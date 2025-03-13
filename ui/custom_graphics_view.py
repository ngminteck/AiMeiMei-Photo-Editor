import cv2
import numpy as np
import torch
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QSizePolicy, QGraphicsPathItem
from PyQt6.QtGui import QPixmap, QPainter, QImage, QPainterPath, QPen, QColor, QBrush
from PyQt6.QtCore import Qt, QBuffer, QIODevice
from providers.sam_model_provider import SAMModelProvider

class CustomGraphicsView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.main_pixmap_item = None          # Background layer item
        self.original_pixmap = None           # Original image pixmap
        self.scene_pixmap = None              # current scene layer pixmap (with hole)
        self.selected_pixmap_item = None      # selected overlay item
        self.selection_feedback_items = []    # For drawing contour feedback
        self.dragging = False

        # Modes: "transform", "object selection", "quick selection"
        self.mode = "transform"
        self.positive_points = []             # For SAM prompt-based selection
        self.negative_points = []             # For SAM prompt-based selection
        self.u2net_selection_mask = None      # Binary mask from U2Net auto selection
        self.image_shape = None               # (height, width) of current image

        # OpenCV images (BGR)
        self.cv_image = None                  # Current working image
        self.original_cv_image = None         # Copy of loaded image
        self.base_cv_image = None             # For re-applying detection

        # QImage (RGBA) conversion of cv_image
        self.cv_image_rgba = None

        # Enhanced image for SAM selection
        self.sam_cv_image = None
        self.sam_cv_image_rgba = None
        self.sam_cv_image_rgb = None

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.checkerboard_pixmap = None
        self.detection_overlay_item = None
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def apply_contrast_and_sharpen(self, image):
        contrast_image = cv2.convertScaleAbs(image, alpha=1.3, beta=0)
        blurred = cv2.GaussianBlur(contrast_image, (0, 0), sigmaX=3)
        sharpened = cv2.addWeighted(contrast_image, 1.5, blurred, -0.5, 0)
        return sharpened

    def _update_cv_image_conversions(self):
        if self.cv_image is None or len(self.cv_image.shape) < 3:
            return
        self.image_shape = (self.cv_image.shape[0], self.cv_image.shape[1])
        if self.cv_image.shape[2] == 4:
            self.cv_image_rgba = cv2.cvtColor(self.cv_image, cv2.COLOR_BGRA2RGBA)
        else:
            self.cv_image_rgba = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGBA)

        self.sam_cv_image = self.apply_contrast_and_sharpen(self.cv_image)
        if self.cv_image.shape[2] == 4:
            self.sam_cv_image_rgba = cv2.cvtColor(self.sam_cv_image, cv2.COLOR_BGRA2RGBA)
            self.sam_cv_image_rgb = cv2.cvtColor(self.sam_cv_image, cv2.COLOR_BGRA2RGB)
        else:
            self.sam_cv_image_rgba = cv2.cvtColor(self.sam_cv_image, cv2.COLOR_BGR2RGBA)
            self.sam_cv_image_rgb = cv2.cvtColor(self.sam_cv_image, cv2.COLOR_BGR2RGB)

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
            tile_painter.fillRect(tile_size // 2, tile_size // 2, tile_size // 2, tile_size // 2, Qt.GlobalColor.lightGray)
            tile_painter.end()
        brush = QBrush(self.checkerboard_pixmap)
        painter.fillRect(image_rect, brush)

    def save(self, filepath=None):
        if filepath:
            self.main_pixmap_item.pixmap().save(filepath, None, 100)

    def clear_detection(self):
        if self.detection_overlay_item:
            self.scene.removeItem(self.detection_overlay_item)
            self.detection_overlay_item = None

    def load_image(self, image_path):
        # Merge selection if exists.
        if self.u2net_selection_mask is not None and np.count_nonzero(self.u2net_selection_mask) > 0:
            self.apply_merge()

        self.clear_detection()
        self.scene.clear()
        self.main_pixmap_item = None
        self.selected_pixmap_item = None
        for item in self.selection_feedback_items:
            self.scene.removeItem(item)
        self.selection_feedback_items = []
        self.positive_points = []
        self.negative_points = []
        self.u2net_selection_mask = None

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

        self.scene_pixmap = self.original_pixmap.copy()
        self._update_cv_image_conversions()
        self.u2net_selection_mask = np.zeros(self.image_shape, dtype=np.uint8)

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

    def object_selection(self):
        if self.cv_image is None:
            print("No image loaded")
            return
        if not self.positive_points and not self.negative_points:
            print("No selection points provided")
            return

        with torch.no_grad():
            predictor = SAMModelProvider.get_predictor()
            predictor.set_image(self.sam_cv_image_rgb)
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
        kernel = np.ones((5, 5), np.uint8)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        self.u2net_selection_mask = cv2.bitwise_or(self.u2net_selection_mask, mask_uint8)

        bridge_kernel = np.ones((25, 25), np.uint8)
        self.u2net_selection_mask = cv2.morphologyEx(self.u2net_selection_mask, cv2.MORPH_CLOSE, bridge_kernel)
        print("Merged prompt-based selection into union mask with bridging.")

        self.positive_points = []
        self.negative_points = []
        self.update_u2net_selection_display()

    def _get_outline_path(self, binary_mask):
        kernel = np.ones((3, 3), np.uint8)
        outline_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_GRADIENT, kernel)
        contours, _ = cv2.findContours(outline_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        path = QPainterPath()
        for cnt in contours:
            if len(cnt) > 0:
                cnt = cnt.squeeze()
                if cnt.ndim < 2:
                    continue
                path.moveTo(cnt[0][0], cnt[0][1])
                for pt in cnt[1:]:
                    path.lineTo(pt[0], pt[1])
                path.closeSubpath()
        return path

    def update_u2net_selection_display(self):
        if self.cv_image is None or self.u2net_selection_mask is None:
            return

        bg_rgba = self.cv_image_rgba.copy()
        bg_alpha = np.where(self.u2net_selection_mask == 255, 0, 255).astype(np.uint8)
        bg_rgba[:, :, 3] = bg_alpha

        h, w, ch = bg_rgba.shape
        bytes_per_line = ch * w
        bg_qimage = QImage(bg_rgba.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888)
        bg_pixmap = QPixmap.fromImage(bg_qimage)
        self.main_pixmap_item.setPixmap(bg_pixmap)

        sel_rgba = self.cv_image_rgba.copy()
        sel_rgba[self.u2net_selection_mask != 255] = [0, 0, 0, 0]
        sel_qimage = QImage(sel_rgba.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888)
        sel_pixmap = QPixmap.fromImage(sel_qimage)

        if self.selected_pixmap_item:
            self.scene.removeItem(self.selected_pixmap_item)
        self.selected_pixmap_item = QGraphicsPixmapItem(sel_pixmap)
        self.selected_pixmap_item.setZValue(10)
        self.scene.addItem(self.selected_pixmap_item)

        # Clear previous feedback items; new ones will be children of selected_pixmap_item.
        self.selection_feedback_items = []

        outline_path = self._get_outline_path(self.u2net_selection_mask)

        # Create feedback items as children of selected_pixmap_item so they move together.
        white_pen = QPen(QColor("white"), 2)
        item_white = QGraphicsPathItem(outline_path, self.selected_pixmap_item)
        item_white.setPen(white_pen)

        black_pen = QPen(QColor("black"), 1)
        item_black = QGraphicsPathItem(outline_path, self.selected_pixmap_item)
        item_black.setPen(black_pen)

        self.selection_feedback_items = [item_white, item_black]

    def apply_merge(self):
        if self.selected_pixmap_item is None and (self.u2net_selection_mask is None or np.count_nonzero(self.u2net_selection_mask) == 0):
            print("No active selection mask to merge.")
            self.selection_feedback_items = []
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
        self.scene_pixmap = merged_pixmap

        # Remove the selected item (which will also remove its child feedback items)
        if self.selected_pixmap_item:
            self.scene.removeItem(self.selected_pixmap_item)
            self.selected_pixmap_item = None

        # Since feedback items are children of selected_pixmap_item, clear the list without manual removal.
        self.selection_feedback_items = []

        self.u2net_selection_mask = np.zeros(self.image_shape, dtype=np.uint8)
        print("Merge applied: selection merged into current image.")

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
        if self.mode == "object selection":
            if event.button() == Qt.MouseButton.LeftButton:
                self.positive_points.append([pos.x(), pos.y()])
                print(f"Added positive point: ({pos.x()}, {pos.y()})")
            elif event.button() == Qt.MouseButton.RightButton:
                self.negative_points.append([pos.x(), pos.y()])
                print(f"Added negative point: ({pos.x()}, {pos.y()})")
            self.object_selection()
        elif self.mode == "quick selection":
            print("TODO")
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
