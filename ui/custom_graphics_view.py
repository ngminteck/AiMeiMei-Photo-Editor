import cv2
import numpy as np
import torch
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QSizePolicy
from PyQt6.QtGui import QPixmap, QPen, QColor, QPainter, QImage, QPainterPath, QBrush
from PyQt6.QtCore import Qt, QBuffer, QIODevice
from providers.sam_model_provider import SAMModelProvider

class CustomGraphicsView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.main_pixmap_item = None
        self.original_pixmap = None
        self.background_pixmap = None
        self.selected_pixmap_item = None
        self.selection_feedback_item = None
        self.dragging = False

        # Modes: "selection" (prompt-based), "auto" (auto-based), "transform" (move/scale)
        self.mode = "transform"

        # For prompt-based mode.
        self.positive_points = []
        self.negative_points = []

        # Union mask for merged selections (0 or 255).
        self.auto_selection_mask = None
        self.image_shape = None

        # Downscale factor for auto mask generation.
        self.downscale_factor = 0.5

        # Main CV image and its conversions.
        self.cv_image = None
        self.original_cv_image = None
        self.base_cv_image = None

        self.cv_image_rgba = None
        self.cv_image_rgb = None
        self.cv_image_small = None
        self.image_rgba_small = None
        self.image_rgb_small = None

        # Cached masks from the auto mask generator.
        self.cached_masks = None

        # Morphology toggle.
        self.use_morphology = True

        # Rendering hints.
        self.aspectRatioMode = Qt.AspectRatioMode.KeepAspectRatio
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Optional checkerboard background.
        self.checkerboard_pixmap = None

        # Detection overlay item for bounding boxes (non-interactive).
        self.detection_overlay_item = None

    def _update_cv_image_conversions(self):
        if self.cv_image is None:
            return
        if len(self.cv_image.shape) < 3:
            return

        self.image_shape = (self.cv_image.shape[0], self.cv_image.shape[1])

        if self.cv_image.shape[2] == 4:
            self.cv_image_rgba = cv2.cvtColor(self.cv_image, cv2.COLOR_BGRA2RGBA)
            self.cv_image_rgb = cv2.cvtColor(self.cv_image, cv2.COLOR_BGRA2RGB)
        else:
            self.cv_image_rgba = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGBA)
            self.cv_image_rgb = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)

        self.cv_image_small = cv2.resize(self.cv_image, (0, 0),
                                         fx=self.downscale_factor,
                                         fy=self.downscale_factor)
        if self.cv_image_small.shape[2] == 4:
            self.image_rgba_small = cv2.cvtColor(self.cv_image_small, cv2.COLOR_BGRA2RGBA)
            self.image_rgb_small = cv2.cvtColor(self.cv_image_small, cv2.COLOR_BGRA2RGB)
        else:
            self.image_rgba_small = cv2.cvtColor(self.cv_image_small, cv2.COLOR_BGR2RGBA)
            self.image_rgb_small = cv2.cvtColor(self.cv_image_small, cv2.COLOR_BGR2RGB)

        if self.mode != "transform":
            with torch.no_grad():
                self.cached_masks = SAMModelProvider.get_auto_mask_generator().generate(self.image_rgb_small)
        else:
            self.cached_masks = None

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
            # Saves only the main (base) image without overlays.
            self.main_pixmap_item.pixmap().save(filepath, None, 100)

    def load_image(self, image_path):
        self.scene.clear()
        self.main_pixmap_item = None
        self.selected_pixmap_item = None
        self.selection_feedback_item = None
        self.positive_points = []
        self.negative_points = []
        self.cached_masks = None

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

        self.auto_selection_mask = np.zeros(self.image_shape, dtype=np.uint8)

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

    def ai_salient_object_selection(self):
        if self.cv_image is None:
            print("No image loaded")
            return
        if not self.positive_points and not self.negative_points:
            print("No selection points provided")
            return

        with torch.no_grad():
            predictor = SAMModelProvider.get_predictor()
            predictor.set_image(self.cv_image_rgb)

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
                multimask_output=False
            )

        mask = masks[0]
        mask_uint8 = (mask.astype(np.uint8)) * 255
        if self.use_morphology:
            kernel = np.ones((5, 5), np.uint8)
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

        self.auto_selection_mask = cv2.bitwise_or(self.auto_selection_mask, mask_uint8)
        print("Merged prompt-based selection into union mask.")
        self.positive_points = []
        self.negative_points = []
        self.update_auto_selection_display()

    def auto_salient_object_update(self, click_point, action="add"):
        if self.cv_image is None:
            print("No image loaded")
            return
        if self.cached_masks is None:
            print("No masks generated")
            return

        x = int(click_point.x())
        y = int(click_point.y())
        x_small = int(x * self.downscale_factor)
        y_small = int(y * self.downscale_factor)

        selected_mask = None
        best_area = 0

        for m in self.cached_masks:
            seg = m["segmentation"]
            if seg[y_small, x_small]:
                area = m.get("area", np.sum(seg))
                if area > best_area:
                    best_area = area
                    selected_mask = seg

        if selected_mask is None:
            selected_mask = max(self.cached_masks, key=lambda m: m.get("area", np.sum(m["segmentation"])))[
                "segmentation"
            ]

        up_mask = cv2.resize(selected_mask.astype(np.uint8),
                             (self.image_shape[1], self.image_shape[0]),
                             interpolation=cv2.INTER_NEAREST)
        new_mask = up_mask * 255

        if self.use_morphology:
            kernel = np.ones((5, 5), np.uint8)
            new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_CLOSE, kernel)

        if action == "add":
            self.auto_selection_mask = cv2.bitwise_or(self.auto_selection_mask, new_mask)
            print("Added object to selection (auto).")
        elif action == "remove":
            inv = cv2.bitwise_not(new_mask)
            self.auto_selection_mask = cv2.bitwise_and(self.auto_selection_mask, inv)
            print("Removed object from selection (auto).")
        else:
            print("Unknown action")
        self.update_auto_selection_display()

    def update_auto_selection_display(self):
        if self.cv_image is None or self.auto_selection_mask is None:
            return

        mask_uint8 = self.auto_selection_mask.copy()

        # LAYER 1: Background (make selected region transparent)
        bg_rgba = self.cv_image_rgba.copy()
        bg_rgba[mask_uint8 == 255, 3] = 0
        bg_h, bg_w, bg_ch = bg_rgba.shape
        bg_bytes_per_line = bg_ch * bg_w
        bg_qimage = QImage(bg_rgba.data, bg_w, bg_h, bg_bytes_per_line, QImage.Format.Format_RGBA8888)
        bg_pixmap = QPixmap.fromImage(bg_qimage)
        self.original_pixmap = bg_pixmap
        self.main_pixmap_item.setPixmap(self.original_pixmap)

        # LAYER 2: Selection Overlay (only display selected areas)
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

        if self.selection_feedback_item:
            self.scene.removeItem(self.selection_feedback_item)

        self.selection_feedback_item = self.scene.addPath(path, QPen(QColor("black"), 2))

    def apply_merge(self):
        if not self.selected_pixmap_item and np.count_nonzero(self.auto_selection_mask) == 0:
            print("No selected object or active selection mask to merge.")
            if self.selection_feedback_item:
                self.scene.removeItem(self.selection_feedback_item)
                self.selection_feedback_item = None
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
        if self.selection_feedback_item:
            self.scene.removeItem(self.selection_feedback_item)
            self.selection_feedback_item = None

        self.auto_selection_mask = np.zeros(self.image_shape, dtype=np.uint8)
        self.cached_masks = None
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
        if self.mode == "selection":
            if event.button() == Qt.MouseButton.LeftButton:
                self.positive_points.append([pos.x(), pos.y()])
                print(f"Added positive point: ({pos.x()}, {pos.y()})")
            elif event.button() == Qt.MouseButton.RightButton:
                self.negative_points.append([pos.x(), pos.y()])
                print(f"Added negative point: ({pos.x()}, {pos.y()})")
            self.ai_salient_object_selection()
        elif self.mode == "auto":
            if event.button() == Qt.MouseButton.LeftButton:
                self.auto_salient_object_update(pos, action="add")
            elif event.button() == Qt.MouseButton.RightButton:
                self.auto_salient_object_update(pos, action="remove")
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
        if self.mode == "transform" and self.selected_pixmap_item:
            scale_factor = 1.1 if event.angleDelta().y() > 0 else 0.9
            self.selected_pixmap_item.setScale(self.selected_pixmap_item.scale() * scale_factor)
        super().wheelEvent(event)
