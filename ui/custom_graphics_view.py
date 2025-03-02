# ui/custom_graphics_view.py
import cv2
import numpy as np
import torch
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt6.QtGui import QPixmap, QPen, QColor, QPainter, QImage, QPainterPath
from PyQt6.QtCore import Qt

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
        self.mode = "transform"  # Default mode is transform.

        # For prompt-based mode: store positive/negative points
        self.positive_points = []
        self.negative_points = []

        # Union mask for merged selections.
        self.auto_selection_mask = None  # uint8 array: 0 or 255 values.
        self.image_shape = None

        # Cache for full-resolution image.
        self.cv_image = None
        # For speeding up auto mask generation, we downscale the image.
        self.downscale_factor = 0.5  # Adjust factor as needed.
        self.cv_image_small = None  # Downscaled version.
        self.image_rgb_small = None

        # Cache auto masks computed on the downscaled image.
        self.cached_masks = None

        # Toggle for morphological post-processing.
        self.use_morphology = True

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

    def load_image(self, image_path):
        self.image_path = image_path
        self.original_pixmap = QPixmap(image_path)
        self.background_pixmap = QPixmap(image_path)
        if self.original_pixmap.isNull():
            print(f"Error: Could not load image from {image_path}")
            return

        # Load full-resolution image with cv2.
        self.cv_image = cv2.imread(image_path)
        if self.cv_image is not None:
            self.image_shape = (self.cv_image.shape[0], self.cv_image.shape[1])
            self.auto_selection_mask = np.zeros(self.image_shape, dtype=np.uint8)

            # Create a downscaled version to speed up auto mask generation.
            self.cv_image_small = cv2.resize(self.cv_image, (0, 0), fx=self.downscale_factor, fy=self.downscale_factor)
            self.image_rgb_small = cv2.cvtColor(self.cv_image_small, cv2.COLOR_BGR2RGB)

            # Only generate auto masks if mode is not transform.
            if self.mode != "transform":
                with torch.no_grad():
                    self.cached_masks = SAMModelProvider.get_auto_mask_generator().generate(self.image_rgb_small)
        else:
            print("Error loading image with cv2")
            return

        self.main_pixmap_item = QGraphicsPixmapItem(self.original_pixmap)
        self.scene.addItem(self.main_pixmap_item)
        self.setSceneRect(self.main_pixmap_item.boundingRect())

    def set_mode(self, mode):
        self.mode = mode
        print(f"Mode set to: {mode}")
        if mode != "selection":
            self.positive_points = []
            self.negative_points = []
        # If switching to a mode that requires the model (selection or auto) and masks haven't been generated,
        # trigger auto mask generation.
        if mode != "transform" and self.cv_image is not None and self.cached_masks is None:
            with torch.no_grad():
                self.cached_masks = SAMModelProvider.get_auto_mask_generator().generate(self.image_rgb_small)

    # --- Prompt-based selection ---
    def ai_salient_object_selection(self):
        if self.cv_image is None:
            print("No image loaded")
            return
        if not self.positive_points and not self.negative_points:
            print("No selection points provided")
            return

        with torch.no_grad():
            # Use full-resolution RGB image for prompt-based selection.
            image_rgb = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
            predictor = SAMModelProvider.get_predictor()
            predictor.set_image(image_rgb)

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

        # Merge prompt-based mask into the union mask.
        self.auto_selection_mask = cv2.bitwise_or(self.auto_selection_mask, mask_uint8)
        print("Merged prompt-based selection into union mask.")
        self.positive_points = []
        self.negative_points = []
        self.update_auto_selection_display()

    # --- Auto mode: update selection based on auto mask generator ---
    def auto_salient_object_update(self, click_point, action="add"):
        if self.cv_image is None:
            print("No image loaded")
            return
        if self.cached_masks is None:
            print("No masks generated")
            return

        # Convert click coordinates from full-resolution to downscaled image.
        x = int(click_point.x())
        y = int(click_point.y())
        x_small = int(x * self.downscale_factor)
        y_small = int(y * self.downscale_factor)

        selected_mask = None
        best_area = 0

        # Iterate over cached masks (from downscaled image).
        for m in self.cached_masks:
            seg = m["segmentation"]
            # Check the downscaled coordinates.
            if seg[y_small, x_small]:
                area = m.get("area", np.sum(seg))
                if area > best_area:
                    best_area = area
                    selected_mask = seg

        if selected_mask is None:
            selected_mask = max(self.cached_masks, key=lambda m: m.get("area", np.sum(m["segmentation"])))["segmentation"]

        # Upsample the selected mask back to full resolution.
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

    # --- Update display based on the union mask ---
    def update_auto_selection_display(self):
        if self.cv_image is None or self.auto_selection_mask is None:
            return

        # Copy full-resolution image.
        img = self.cv_image.copy()
        mask_uint8 = self.auto_selection_mask.copy()

        # Draw an outline around the union mask.
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

        # Create an overlay pixmap from the union mask.
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgba = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2RGBA)
        img_rgba[:, :, 3] = mask_uint8
        result = cv2.bitwise_and(img_rgba, img_rgba, mask=mask_uint8)

        h, w, ch = result.shape
        bytes_per_line = ch * w
        q_img = QImage(result.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888)
        selected_pixmap = QPixmap.fromImage(q_img)

        if self.selected_pixmap_item:
            self.scene.removeItem(self.selected_pixmap_item)
        self.selected_pixmap_item = QGraphicsPixmapItem(selected_pixmap)
        self.scene.addItem(self.selected_pixmap_item)

        # Update main image: set transparency where selection exists.
        img_rgba[:, :, 3] = cv2.bitwise_not(mask_uint8)
        q_img_main = QImage(img_rgba.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888)
        self.original_pixmap = QPixmap.fromImage(q_img_main)
        self.main_pixmap_item.setPixmap(self.original_pixmap)

    # --- Merge the transformed selection onto the main image ---
    def apply_merge(self):
        if not self.main_pixmap_item:
            print("No current image available.")
            return
        if not self.selected_pixmap_item:
            print("No selected object to merge.")
            return

        composite_image = self.main_pixmap_item.pixmap().toImage()
        painter = QPainter(composite_image)
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

        # Reset the union mask and cached masks (since the base image has changed).
        self.auto_selection_mask = np.zeros(self.image_shape, dtype=np.uint8)
        self.cached_masks = None
        print("Merge applied: selection merged into current image.")

    # --- Event Handling ---
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
