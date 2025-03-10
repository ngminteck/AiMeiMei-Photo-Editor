import cv2
import numpy as np
import torch
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QSizePolicy
from PyQt6.QtGui import QPixmap, QPainter, QImage, QPainterPath, QPen, QColor, QBrush
from PyQt6.QtCore import Qt, QBuffer, QIODevice
from providers.sam_model_provider import SAMModelProvider


class CustomGraphicsView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.main_pixmap_item = None  # QGraphicsPixmapItem showing the main image
        self.original_pixmap = None  # QPixmap of the originally loaded image
        self.background_pixmap = None  # QPixmap used as the background layer
        self.selected_pixmap_item = None  # QGraphicsPixmapItem for any selection overlay
        # We'll store contour items for easier removal.
        self.selection_feedback_items = []
        self.dragging = False

        # Modes: "transform" for move/scale and "selection" for prompt-based SAM selection.
        self.mode = "transform"
        self.positive_points = []  # For manual selection: positive seed points
        self.negative_points = []  # For manual selection: negative seed points
        self.auto_selection_mask = None  # Binary mask for auto-selected areas
        self.image_shape = None  # Stores the (height, width) of the cv image

        # Main OpenCV images (BGR format)
        self.cv_image = None  # Current working image in OpenCV (BGR)
        self.original_cv_image = None  # A copy of the originally loaded image (BGR)
        self.base_cv_image = None  # A base copy used for re-applying detection/processing

        # Conversions for display (QImage requires RGBA)
        self.cv_image_rgba = None  # cv_image converted to RGBA (used for QImage conversion)

        # Enhanced image for segmentation (for SAM-based selection).
        self.enhanced_cv_image = None
        self.enhanced_cv_image_rgba = None
        self.enhanced_cv_image_rgb = None

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Ensure scroll bars appear if needed.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Optional checkerboard background for transparency.
        self.checkerboard_pixmap = None

        # Detection overlay item (for detection results, if any).
        self.detection_overlay_item = None

        # Set transformation anchor to under the mouse for better zoom behavior.
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def clear_selection(self):
        """Clears any existing selection state and overlays."""
        # Clear seed points for manual selection
        self.positive_points = []
        self.negative_points = []

        # Remove any selection overlay pixmap item from the scene
        if self.selected_pixmap_item:
            self.scene.removeItem(self.selected_pixmap_item)
            self.selected_pixmap_item = None

        # Remove any feedback contour items from the scene
        for item in self.selection_feedback_items:
            self.scene.removeItem(item)
        self.selection_feedback_items = []

        # Reset the auto selection mask if an image shape is available
        if self.image_shape is not None:
            self.auto_selection_mask = np.zeros(self.image_shape, dtype=np.uint8)
        else:
            self.auto_selection_mask = None

    def clear_detection(self):
        """Clears any active detection overlay."""
        if self.detection_overlay_item:
            self.scene.removeItem(self.detection_overlay_item)
            self.detection_overlay_item = None

    def _update_cv_image_conversions(self):
        if self.cv_image is None or len(self.cv_image.shape) < 3:
            return

        self.image_shape = (self.cv_image.shape[0], self.cv_image.shape[1])
        if self.cv_image.shape[2] == 4:
            self.cv_image_rgba = cv2.cvtColor(self.cv_image, cv2.COLOR_BGRA2RGBA)
        else:
            self.cv_image_rgba = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGBA)

        # Prepare enhanced image for SAM-based prompt selection.
        self.enhanced_cv_image = self.apply_contrast_and_sharpen(self.cv_image)
        if self.cv_image.shape[2] == 4:
            self.enhanced_cv_image_rgba = cv2.cvtColor(self.enhanced_cv_image, cv2.COLOR_BGRA2RGBA)
            self.enhanced_cv_image_rgb = cv2.cvtColor(self.enhanced_cv_image, cv2.COLOR_BGRA2RGB)
        else:
            self.enhanced_cv_image_rgba = cv2.cvtColor(self.enhanced_cv_image, cv2.COLOR_BGR2RGBA)
            self.enhanced_cv_image_rgb = cv2.cvtColor(self.enhanced_cv_image, cv2.COLOR_BGR2RGB)

    def apply_contrast_and_sharpen(self, image):
        """
        Enhances the image by boosting contrast and applying an unsharp mask.
        """
        contrast_image = cv2.convertScaleAbs(image, alpha=1.3, beta=0)
        blurred = cv2.GaussianBlur(contrast_image, (0, 0), sigmaX=3)
        sharpened = cv2.addWeighted(contrast_image, 1.5, blurred, -0.5, 0)
        return sharpened

    def drawBackground(self, painter, rect):
        if self.main_pixmap_item:
            image_rect = self.main_pixmap_item.boundingRect()
        else:
            image_rect = rect

        tile_size = max(20, int(min(image_rect.width(), image_rect.height()) / 40))
        if self.checkerboard_pixmap is None or self.checkerboard_pixmap.width() != tile_size:
            self.checkerboard_pixmap = QPixmap(tile_size, tile_size)
            self.checkerboard_pixmap.fill(Qt.GlobalColor.white)
            tile_painter = QPainter(self.checkerboard_pixmap)
            tile_painter.fillRect(0, 0, tile_size // 2, tile_size // 2, Qt.GlobalColor.lightGray)
            tile_painter.fillRect(tile_size // 2, tile_size // 2, tile_size // 2, tile_size // 2,
                                  Qt.GlobalColor.lightGray)
            tile_painter.end()

        brush = QBrush(self.checkerboard_pixmap)
        painter.fillRect(image_rect, brush)

    def save(self, filepath=None):
        if filepath:
            self.main_pixmap_item.pixmap().save(filepath, None, 100)

    def load_image(self, image_path):
        # Before loading a new image, force-apply merge if any selection is active.
        if self.auto_selection_mask is not None and np.count_nonzero(self.auto_selection_mask) > 0:
            self.apply_merge()

        # Force disable detection overlay.
        self.clear_detection()

        # Create a new scene to clear any lingering items.
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        # Clear selection-related state.
        self.clear_selection()
        self.main_pixmap_item = None
        self.selected_pixmap_item = None

        self.image_path = image_path
        self.cv_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if self.cv_image is None:
            print(f"Error: Could not load image from {image_path}")
            return

        self.original_cv_image = self.cv_image.copy()
        self.base_cv_image = self.cv_image.copy()

        if not image_path.lower().endswith('.png'):
            ret, buf = cv2.imencode('.png', self.cv_image)
            if ret:
                png_bytes = buf.tobytes()
                self.original_pixmap = QPixmap()
                self.original_pixmap.loadFromData(png_bytes, "PNG")
            else:
                print("Error: Could not encode image as PNG.")
                return
        else:
            self.original_pixmap = QPixmap(image_path)

        self.background_pixmap = self.original_pixmap.copy()
        self._update_cv_image_conversions()

        if self.image_shape is not None:
            self.auto_selection_mask = np.zeros(self.image_shape, dtype=np.uint8)
        else:
            self.auto_selection_mask = None

        self.main_pixmap_item = QGraphicsPixmapItem(self.original_pixmap)
        self.scene.addItem(self.main_pixmap_item)
        self.setSceneRect(self.main_pixmap_item.boundingRect())
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def set_mode(self, mode):
        self.mode = mode
        print(f"Mode set to: {mode}")
        if mode != "selection":
            self.positive_points = []
            self.negative_points = []
        if mode != "transform" and self.cv_image is not None:
            self._update_cv_image_conversions()

    def ai_object_selection(self):
        if self.cv_image is None:
            print("No image loaded")
            return
        if not self.positive_points and not self.negative_points:
            print("No selection points provided")
            return

        with torch.no_grad():
            predictor = SAMModelProvider.get_predictor()
            predictor.set_image(self.enhanced_cv_image_rgb)

            points = []
            labels = []
            if self.positive_points:
                points.extend(self.positive_points)
                labels.extend([1] * len(self.positive_points))
            if self.negative_points:
                points.extend(self.negative_points)
                labels.extend([0] * len(self.negative_points))

            points_array = np.array(points)
            labels_array = np.array(labels)

            masks, scores, logits = predictor.predict(
                point_coords=points_array,
                point_labels=labels_array,
                multimask_output=True
            )
            best_idx = np.argmax(scores)
            mask = masks[best_idx]

        mask_uint8 = (mask.astype(np.uint8)) * 255

        # Optionally apply morphology
        kernel = np.ones((5, 5), np.uint8)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

        self.auto_selection_mask = cv2.bitwise_or(self.auto_selection_mask, mask_uint8)

        bridge_kernel = np.ones((25, 25), np.uint8)
        self.auto_selection_mask = cv2.morphologyEx(self.auto_selection_mask, cv2.MORPH_CLOSE, bridge_kernel)

        print("Merged prompt-based selection into union mask with bridging.")
        self.positive_points = []
        self.negative_points = []
        self.update_auto_selection_display()

    def update_auto_selection_display(self):
        if self.cv_image is None or self.auto_selection_mask is None:
            return

        mask_uint8 = self.auto_selection_mask.copy()

        # Create a version of the cv image with selected areas made transparent.
        bg_rgba = self.cv_image_rgba.copy()
        bg_rgba[mask_uint8 == 255, 3] = 0
        bg_h, bg_w, bg_ch = bg_rgba.shape
        bg_bytes_per_line = bg_ch * bg_w
        bg_qimage = QImage(bg_rgba.data, bg_w, bg_h, bg_bytes_per_line, QImage.Format.Format_RGBA8888)
        bg_pixmap = QPixmap.fromImage(bg_qimage)
        self.original_pixmap = bg_pixmap
        self.main_pixmap_item.setPixmap(self.original_pixmap)

        # Create an overlay pixmap for feedback (shows contour outlines).
        overlay_rgba = self.cv_image_rgba.copy()
        overlay_rgba[mask_uint8 == 0] = [0, 0, 0, 0]
        ov_h, ov_w, ov_ch = overlay_rgba.shape
        ov_bytes_per_line = ov_ch * ov_w
        ov_qimage = QImage(overlay_rgba.data, ov_w, ov_h, ov_bytes_per_line, QImage.Format.Format_RGBA8888)
        overlay_pixmap = QPixmap.fromImage(ov_qimage)

        if self.selected_pixmap_item:
            self.scene.removeItem(self.selected_pixmap_item)
        self.selected_pixmap_item = QGraphicsPixmapItem(overlay_pixmap)
        self.selected_pixmap_item.setZValue(10)
        self.scene.addItem(self.selected_pixmap_item)

        for item in self.selection_feedback_items:
            self.scene.removeItem(item)
        self.selection_feedback_items = []

        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        path = QPainterPath()
        for cnt in contours:
            if len(cnt) > 0:
                cnt = cnt.squeeze()
                if cnt.ndim < 2:
                    continue
                start = cnt[0]
                path.moveTo(start[0], start[1])
                for pt in cnt[1:]:
                    path.lineTo(pt[0], pt[1])
                path.closeSubpath()

        white_pen = QPen(QColor("white"), 4)
        item_white = self.scene.addPath(path, white_pen)
        black_pen = QPen(QColor("black"), 2)
        item_black = self.scene.addPath(path, black_pen)
        self.selection_feedback_items = [item_white, item_black]

    def apply_merge(self):
        """
        Forces any active selection (via overlay/mask) to be merged into the main image.
        If no selection exists, this function does nothing.
        """
        if not self.selected_pixmap_item and (
                self.auto_selection_mask is None or np.count_nonzero(self.auto_selection_mask) == 0):
            print("No active selection mask to merge.")
            return

        composite_image = self.main_pixmap_item.pixmap().toImage()
        painter = QPainter(composite_image)
        if self.selected_pixmap_item:
            selected_pixmap = self.selected_pixmap_item.pixmap()
            pos = self.selected_pixmap_item.pos()
            painter.drawPixmap(int(pos.x()), int(pos.y()), selected_pixmap)
        painter.end()

        merged_pixmap = QPixmap.fromImage(composite_image)
        self.main_pixmap_item.setPixmap(merged_pixmap)
        self.background_pixmap = merged_pixmap

        if self.selected_pixmap_item:
            self.scene.removeItem(self.selected_pixmap_item)
            self.selected_pixmap_item = None
        for item in self.selection_feedback_items:
            self.scene.removeItem(item)
        self.selection_feedback_items = []

        # Reset the auto-selection mask
        if self.image_shape is not None:
            self.auto_selection_mask = np.zeros(self.image_shape, dtype=np.uint8)
        else:
            self.auto_selection_mask = None

        print("Merge applied: selection merged into current image.")

        # Update the cv_image from the merged pixmap
        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.ReadWrite)
        merged_pixmap.save(buffer, "PNG")
        arr = np.frombuffer(buffer.data(), np.uint8)
        self.cv_image = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        buffer.close()

        self._update_cv_image_conversions()
        self.base_cv_image = self.cv_image.copy()

    def mousePressEvent(self, event):
        pos = self.mapToScene(event.pos())
        if self.mode == "selection":
            if event.button() == Qt.MouseButton.LeftButton:
                self.positive_points.append([pos.x(), pos.y()])
                print(f"Added positive point: ({pos.x()}, {pos.y()})")
            elif event.button() == Qt.MouseButton.RightButton:
                self.negative_points.append([pos.x(), pos.y()])
                print(f"Added negative point: ({pos.x()}, {pos.y()})")
            self.ai_object_selection()
        elif self.mode == "transform" and self.selected_pixmap_item:
            self.dragging = True
            self.drag_start = pos
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        pos = self.mapToScene(event.pos())
        if self.dragging and self.selected_pixmap_item:
            delta = pos - self.drag_start
            self.selected_pixmap_item.moveBy(delta.x(), delta.y())
            self.drag_start = pos
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        zoomInFactor = 1.25
        zoomOutFactor = 1 / zoomInFactor
        factor = zoomInFactor if event.angleDelta().y() > 0 else zoomOutFactor
        self.scale(factor, factor)
