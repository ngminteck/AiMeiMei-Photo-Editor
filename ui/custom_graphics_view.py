import cv2
import numpy as np
import torch
from PyQt6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QSizePolicy,
    QGraphicsPathItem, QGraphicsEllipseItem
)
from PyQt6.QtGui import QPixmap, QPainter, QImage, QPainterPath, QPen, QColor, QBrush
from PyQt6.QtCore import Qt, QBuffer, QIODevice, QRectF
from providers.sam_model_provider import SAMModelProvider


class CustomGraphicsView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        self.selected_pixmap_item = None  # Selected overlay item
        self.background_pixmap_item = None  # Background layer item

        self.selection_feedback_items = []  # For drawing contour feedback
        self.dragging = False

        # Modes: "transform", "object selection", "quick selection", "clone stamp"
        self.mode = "transform"
        self.positive_points = []  # For SAM prompt-based selection
        self.negative_points = []  # For SAM prompt-based selection
        self.selection_mask = None  # Binary mask from U2Net auto selection
        self.image_shape = None  # (height, width) of current image

        # OpenCV images (BGRA)
        self.current_cv_image = None  # Current working image
        self.detection_cv_image = None  # Composite detection image (includes current selection)

        # QImage (RGBA) conversion of current_cv_image
        self.display_cv_image = None

        # Enhanced image for SAM selection
        self.sam_cv_image = None
        self.sam_cv_image_rgb = None

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.checkerboard_pixmap = None
        self.detection_overlay_item = None
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        # Quick selection settings
        self.quick_select_brush_size = 5
        self.quick_selection_overlay = None

        # Clone Stamp mode settings
        self.clone_stamp_brush_size = 5
        self.clone_source_point = None
        self.clone_offset = None
        self.clone_stamp_overlay = None
        self.clone_source_overlay = None

        # Removed score_update_callback since real-time update is managed by MainWindow

    def apply_contrast_and_sharpen(self, image):
        contrast_image = cv2.convertScaleAbs(image, alpha=1.3, beta=0)
        blurred = cv2.GaussianBlur(contrast_image, (0, 0), sigmaX=3)
        sharpened = cv2.addWeighted(contrast_image, 1.5, blurred, -0.5, 0)
        return sharpened

    def update_all_cv_image_conversions(self):
        if self.current_cv_image is None or len(self.current_cv_image.shape) < 3:
            return

        # Update the internal "image_shape" to match the new image
        new_h, new_w = self.current_cv_image.shape[:2]
        self.image_shape = (new_h, new_w)

        # Convert current_cv_image to RGBA for display
        if self.current_cv_image.shape[2] == 4:
            self.display_cv_image = cv2.cvtColor(self.current_cv_image, cv2.COLOR_BGRA2RGBA)
        else:
            self.display_cv_image = cv2.cvtColor(self.current_cv_image, cv2.COLOR_BGR2RGBA)

        # Update sam_cv_image for the SAM predictor
        self.sam_cv_image = self.apply_contrast_and_sharpen(self.current_cv_image)
        if self.current_cv_image.shape[2] == 4:
            self.sam_cv_image_rgb = cv2.cvtColor(self.sam_cv_image, cv2.COLOR_BGRA2RGB)
        else:
            self.sam_cv_image_rgb = cv2.cvtColor(self.sam_cv_image, cv2.COLOR_BGR2RGB)

        # --- Resize the selection mask if the image size has changed ---
        if self.selection_mask is not None:
            mask_h, mask_w = self.selection_mask.shape[:2]
            if (mask_h, mask_w) != (new_h, new_w):
                # Use nearest-neighbor to preserve sharp edges
                self.selection_mask = cv2.resize(
                    self.selection_mask,
                    (new_w, new_h),
                    interpolation=cv2.INTER_NEAREST
                )

        # Update the detection composite whenever conversions update
        self.update_detection_composite()

    def drawBackground(self, painter, rect):
        if self.background_pixmap_item:
            image_rect = self.background_pixmap_item.boundingRect()
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
            self.background_pixmap_item.pixmap().save(filepath, None, 100)

    def clear_detection(self):
        if self.detection_overlay_item:
            self.scene.removeItem(self.detection_overlay_item)
            self.detection_overlay_item = None

    def load_image(self, image_path):
        # Merge selection if exists.
        if self.selection_mask is not None and np.count_nonzero(self.selection_mask) > 0:
            self.apply_merge()

        self.clear_detection()
        self.scene.clear()
        self.background_pixmap_item = None
        self.selected_pixmap_item = None
        for item in self.selection_feedback_items:
            self.scene.removeItem(item)
        self.selection_feedback_items = []
        self.positive_points = []
        self.negative_points = []
        self.selection_mask = None

        if self.clone_stamp_overlay:
            self.scene.removeItem(self.clone_stamp_overlay)
            self.clone_stamp_overlay = None
        if self.clone_source_overlay:
            self.scene.removeItem(self.clone_source_overlay)
            self.clone_source_overlay = None

        self.image_path = image_path
        self.current_cv_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if self.current_cv_image is None:
            print(f"Error: Could not load image from {image_path}")
            return

        # Initialize the selection mask to zeros (preserving any transparency already in the image)
        h, w = self.current_cv_image.shape[:2]
        self.selection_mask = np.zeros((h, w), dtype=np.uint8)

        self.detection_cv_image = self.current_cv_image.copy()

        if not image_path.lower().endswith('.png'):
            ret, buf = cv2.imencode('.png', self.current_cv_image)
            if ret:
                png_bytes = buf.tobytes()
                pixmap = QPixmap()
                pixmap.loadFromData(png_bytes, "PNG")
            else:
                print("Error: Could not encode image as PNG.")
                return
        else:
            pixmap = QPixmap(image_path)

        self.background_pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.background_pixmap_item)
        self.setSceneRect(self.background_pixmap_item.boundingRect())
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def set_mode(self, mode):
        self.mode = mode
        print(f"Mode set to: {mode}")
        if mode != "selection":
            self.positive_points = []
            self.negative_points = []
        if mode != "transform" and self.current_cv_image is not None:
            self.update_all_cv_image_conversions()

        if mode != "quick selection" and self.quick_selection_overlay:
            self.scene.removeItem(self.quick_selection_overlay)
            self.quick_selection_overlay = None
        if mode != "clone stamp":
            if self.clone_stamp_overlay:
                self.scene.removeItem(self.clone_stamp_overlay)
                self.clone_stamp_overlay = None
            if self.clone_source_overlay:
                self.scene.removeItem(self.clone_source_overlay)
                self.clone_source_overlay = None
            self.clone_offset = None

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
            self._quick_select_at_position(pos, event.button())
            self._update_quick_selection_overlay(pos)
        elif self.mode == "clone stamp":
            if event.button() == Qt.MouseButton.RightButton:
                offset = self.background_pixmap_item.pos()
                sample_x = int(pos.x() - offset.x())
                sample_y = int(pos.y() - offset.y())
                self.clone_source_point = (sample_x, sample_y)
                self.clone_offset = None
                self._update_clone_source_overlay(pos)
                print(f"Clone source set at: ({sample_x}, {sample_y})")
            elif event.button() == Qt.MouseButton.LeftButton:
                if self.clone_source_point is None:
                    print("Clone source not set. Right-click to set it.")
                    return
                if self.clone_offset is None:
                    offset = self.background_pixmap_item.pos()
                    dest_x = int(pos.x() - offset.x())
                    dest_y = int(pos.y() - offset.y())
                    self.clone_offset = (self.clone_source_point[0] - dest_x, self.clone_source_point[1] - dest_y)
                    print(f"Clone offset computed as: {self.clone_offset}")
                self._clone_stamp_at_position(pos)
                self._update_clone_stamp_overlay(pos)
        elif self.mode == "transform" and self.selected_pixmap_item:
            self.dragging = True
            self.drag_start = pos
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        pos = self.mapToScene(event.pos())
        if self.mode == "quick selection" and event.buttons() in [Qt.MouseButton.LeftButton,
                                                                  Qt.MouseButton.RightButton]:
            if event.buttons() & Qt.MouseButton.LeftButton:
                self._quick_select_at_position(pos, Qt.MouseButton.LeftButton)
            elif event.buttons() & Qt.MouseButton.RightButton:
                self._quick_select_at_position(pos, Qt.MouseButton.RightButton)
            self._update_quick_selection_overlay(pos)
        elif self.mode == "clone stamp" and (event.buttons() & Qt.MouseButton.LeftButton):
            self._clone_stamp_at_position(pos)
            self._update_clone_stamp_overlay(pos)
        elif self.dragging and self.selected_pixmap_item:
            delta = pos - self.drag_start
            self.selected_pixmap_item.moveBy(delta.x(), delta.y())
            self.drag_start = pos
        else:
            if self.mode == "quick selection":
                self._update_quick_selection_overlay(pos)
            if self.mode == "clone stamp":
                self._update_clone_stamp_overlay(pos)
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

    def object_selection(self):
        if self.current_cv_image is None:
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
        self.selection_mask = cv2.bitwise_or(self.selection_mask, mask_uint8)

        bridge_kernel = np.ones((25, 25), np.uint8)
        self.selection_mask = cv2.morphologyEx(self.selection_mask, cv2.MORPH_CLOSE, bridge_kernel)
        print("Merged prompt-based selection into union mask with bridging.")

        self.positive_points = []
        self.negative_points = []
        self.update_display()

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

    def update_display(self):
        if self.current_cv_image is None or self.selection_mask is None:
            return
        if self.display_cv_image is None:
            self.update_all_cv_image_conversions()

        if self.selected_pixmap_item:
            self.scene.removeItem(self.selected_pixmap_item)
            self.selected_pixmap_item = None
        for item in self.selection_feedback_items:
            self.scene.removeItem(item)
        self.selection_feedback_items = []

        # Preserve the original alpha from display_cv_image
        bg_rgba = self.display_cv_image.copy()
        orig_alpha = self.display_cv_image[..., 3].copy() if self.display_cv_image.shape[2] == 4 else np.full((self.image_shape[0], self.image_shape[1]), 255, dtype=np.uint8)
        bg_rgba[..., 3] = np.where(self.selection_mask == 255, 0, orig_alpha)

        h, w, ch = bg_rgba.shape
        bytes_per_line = ch * w
        bg_qimage = QImage(bg_rgba.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888)
        bg_pixmap = QPixmap.fromImage(bg_qimage)
        self.background_pixmap_item.setPixmap(bg_pixmap)

        # Create foreground image showing only the selected region
        sel_rgba = self.display_cv_image.copy()
        sel_rgba[self.selection_mask != 255] = [0, 0, 0, 0]
        sel_qimage = QImage(sel_rgba.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888)
        sel_pixmap = QPixmap.fromImage(sel_qimage)
        if self.selected_pixmap_item:
            self.scene.removeItem(self.selected_pixmap_item)
        self.selected_pixmap_item = QGraphicsPixmapItem(sel_pixmap)
        self.selected_pixmap_item.setZValue(10)
        self.scene.addItem(self.selected_pixmap_item)

        outline_path = self._get_outline_path(self.selection_mask)
        white_pen = QPen(QColor("white"), 2)
        item_white = QGraphicsPathItem(outline_path, self.selected_pixmap_item)
        item_white.setPen(white_pen)
        black_pen = QPen(QColor("black"), 1)
        item_black = QGraphicsPathItem(outline_path, self.selected_pixmap_item)
        item_black.setPen(black_pen)
        self.selection_feedback_items = [item_white, item_black]

        # Update the detection composite so that detection_cv_image reflects the current selection
        self.update_detection_composite()

    def alpha_composite(self, bg, fg):
        """
        Composite fg over bg using per-pixel alpha blending.
        Both bg and fg are assumed to be RGBA uint8 numpy arrays.
        """
        bg = bg.astype(np.float32) / 255.0
        fg = fg.astype(np.float32) / 255.0
        alpha_fg = fg[..., 3:4]
        alpha_bg = bg[..., 3:4]
        out_alpha = alpha_fg + alpha_bg * (1 - alpha_fg)
        out_rgb = (fg[..., :3] * alpha_fg + bg[..., :3] * alpha_bg * (1 - alpha_fg)) / (out_alpha + 1e-6)
        out = np.concatenate([out_rgb, out_alpha], axis=2)
        out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
        return out

    def update_detection_composite(self):
        """
        Create a composite image that includes the current selection overlay.
        The result is stored in self.detection_cv_image.
        """
        if self.display_cv_image is None or self.selection_mask is None:
            return

        # Background: use original image but make selected areas transparent
        bg_rgba = self.display_cv_image.copy()
        orig_alpha = self.display_cv_image[..., 3].copy() if self.display_cv_image.shape[2] == 4 else np.full((self.image_shape[0], self.image_shape[1]), 255, dtype=np.uint8)
        bg_rgba[..., 3] = np.where(self.selection_mask == 255, 0, orig_alpha)

        # Foreground: only the selected region (rest transparent)
        sel_rgba = self.display_cv_image.copy()
        sel_rgba[self.selection_mask != 255] = [0, 0, 0, 0]

        composite = self.alpha_composite(bg_rgba, sel_rgba)
        self.detection_cv_image = composite

    def apply_merge(self):
        if self.selected_pixmap_item is None and (
                self.selection_mask is None or np.count_nonzero(self.selection_mask) == 0):
            print("No active selection mask to merge.")
            self.selection_feedback_items = []
            return

        composite_image = self.background_pixmap_item.pixmap().toImage()
        painter = QPainter(composite_image)
        if self.selected_pixmap_item:
            selected_pixmap = self.selected_pixmap_item.pixmap()
            pos = self.selected_pixmap_item.pos()
            painter.drawPixmap(int(pos.x()), int(pos.y()), selected_pixmap)
        painter.end()

        merged_pixmap = QPixmap.fromImage(composite_image)
        self.background_pixmap_item.setPixmap(merged_pixmap)

        if self.selected_pixmap_item:
            self.scene.removeItem(self.selected_pixmap_item)
            self.selected_pixmap_item = None

        self.selection_feedback_items = []
        self.selection_mask = np.zeros(self.image_shape, dtype=np.uint8)
        print("Merge applied: selection merged into current image.")

        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.ReadWrite)
        merged_pixmap.save(buffer, "PNG")
        arr = np.frombuffer(buffer.data(), np.uint8)
        self.current_cv_image = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        buffer.close()

        self.update_all_cv_image_conversions()
        self.detection_cv_image = self.current_cv_image.copy()

    def _quick_select_at_position(self, pos, button):
        if not self.background_pixmap_item or self.image_shape is None:
            return
        offset = self.background_pixmap_item.pos()
        x = int(pos.x() - offset.x())
        y = int(pos.y() - offset.y())
        if x < 0 or y < 0 or x >= self.image_shape[1] or y >= self.image_shape[0]:
            return
        brush_radius = self.quick_select_brush_size
        Y, X = np.ogrid[:self.image_shape[0], :self.image_shape[1]]
        dist = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
        mask_area = dist <= brush_radius
        if button == Qt.MouseButton.LeftButton:
            self.selection_mask[mask_area] = 255
        elif button == Qt.MouseButton.RightButton:
            self.selection_mask[mask_area] = 0
        self.update_display()

    def _update_quick_selection_overlay(self, pos):
        radius = self.quick_select_brush_size
        rect = QRectF(pos.x() - radius, pos.y() - radius, 2 * radius, 2 * radius)
        if not self.quick_selection_overlay:
            self.quick_selection_overlay = QGraphicsEllipseItem()
            pen = QPen(QColor("red"))
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setWidth(2)
            self.quick_selection_overlay.setPen(pen)
            self.quick_selection_overlay.setZValue(30)
            self.scene.addItem(self.quick_selection_overlay)
        self.quick_selection_overlay.setRect(rect)

    def _clone_stamp_at_position(self, pos):
        if not self.background_pixmap_item or self.image_shape is None or self.clone_offset is None:
            return
        offset = self.background_pixmap_item.pos()
        dest_x = int(pos.x() - offset.x())
        dest_y = int(pos.y() - offset.y())
        brush_radius = self.clone_stamp_brush_size
        src_x = dest_x + self.clone_offset[0]
        src_y = dest_y + self.clone_offset[1]
        img_h, img_w = self.image_shape
        half_patch = brush_radius
        src_x1 = max(src_x - half_patch, 0)
        src_y1 = max(src_y - half_patch, 0)
        src_x2 = min(src_x + half_patch, img_w)
        src_y2 = min(src_y + half_patch, img_h)
        patch = self.current_cv_image[src_y1:src_y2, src_x1:src_x2].copy()
        dest_x1 = max(dest_x - half_patch, 0)
        dest_y1 = max(dest_y - half_patch, 0)
        dest_x2 = min(dest_x + half_patch, img_w)
        dest_y2 = min(dest_y + half_patch, img_h)
        patch_h, patch_w = patch.shape[:2]
        dest_h = dest_y2 - dest_y1
        dest_w = dest_x2 - dest_x1
        h = min(patch_h, dest_h)
        w = min(patch_w, dest_w)
        if h > 0 and w > 0:
            self.current_cv_image[dest_y1:dest_y1 + h, dest_x1:dest_x1 + w] = patch[:h, :w]
            self.update_all_cv_image_conversions()
            h_img, w_img, ch = self.display_cv_image.shape
            bytes_per_line = ch * w_img
            qimage = QImage(self.display_cv_image.data, w_img, h_img, bytes_per_line, QImage.Format.Format_RGBA8888)
            pixmap = QPixmap.fromImage(qimage)
            self.background_pixmap_item.setPixmap(pixmap)

    def _update_clone_stamp_overlay(self, pos):
        radius = self.clone_stamp_brush_size
        rect = QRectF(pos.x() - radius, pos.y() - radius, 2 * radius, 2 * radius)
        if not self.clone_stamp_overlay:
            self.clone_stamp_overlay = QGraphicsEllipseItem()
            pen = QPen(QColor("green"))
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setWidth(2)
            self.clone_stamp_overlay.setPen(pen)
            self.clone_stamp_overlay.setZValue(30)
            self.scene.addItem(self.clone_stamp_overlay)
        self.clone_stamp_overlay.setRect(rect)

    def _update_clone_source_overlay(self, pos):
        radius = self.clone_stamp_brush_size
        rect = QRectF(pos.x() - radius, pos.y() - radius, 2 * radius, 2 * radius)
        if not self.clone_source_overlay:
            self.clone_source_overlay = QGraphicsEllipseItem()
            pen = QPen(QColor("blue"))
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setWidth(2)
            self.clone_source_overlay.setPen(pen)
            self.clone_source_overlay.setZValue(30)
            self.scene.addItem(self.clone_source_overlay)
        self.clone_source_overlay.setRect(rect)
