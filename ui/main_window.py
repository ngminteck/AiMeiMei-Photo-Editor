import os
import sys
import shutil
import glob
import uuid
import cv2
import numpy as np
import torch

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QMessageBox, QLabel, QTextEdit,
    QScrollArea, QListWidget, QListWidgetItem, QGraphicsPixmapItem
)
from PyQt6.QtGui import QScreen, QPixmap, QAction, QImage, QIcon
from PyQt6.QtCore import Qt, QRect, QSize

from PIL import Image, ImageDraw
from PIL.ImageQt import ImageQt

# Custom modules
from ui.custom_graphics_view import CustomGraphicsView
from ui.filter_panel_widget import FilterPanelWidget  # Refactored filter panel

# Providers
from providers.controlnet_model_provider import load_controlnet, make_divisible_by_8
from providers.yolo_detection_provider import detect_objects, group_objects, select_focus_object
from providers.realesrgan_provider import RealESRGANProvider
from providers.sam_model_provider import SAMModelProvider
from providers.u2net_provider import U2NetProvider


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_file = None
        self.mode_buttons = {}
        self.detection_enabled = False
        self.reference_dir = os.path.join("images", "reference_images")
        os.makedirs(self.reference_dir, exist_ok=True)
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Image Editor")
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        # --- Top Section: Aesthetic Score Display ---
        self.init_top_section(main_layout)

        # --- Center Section: Layout with Button Panel, Image View, and Filter Panel ---
        self.init_center_section(main_layout)

        # --- Bottom Section: Prompt Area and Reference Images ---
        self.init_bottom_section(main_layout)

        self.setCentralWidget(central_widget)
        self.create_menu_bar()
        self.adjustSize()
        self.set_mode_action("transform")
        self.refresh_reference_list()

    # ============================
    # Layout and UI Setup
    # ============================
    def init_top_section(self, parent_layout):
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        score_label = QLabel("Aesthetic Score: 85 | Position: Good | Angle: Optimal | Brightness: Balanced | Focus: Sharp")
        score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top_layout.addWidget(score_label)
        parent_layout.addWidget(top_widget, 5)

    def init_center_section(self, parent_layout):
        center_widget = QWidget()
        center_layout = QHBoxLayout(center_widget)

        # Left: Button Panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.init_left_buttons(left_layout)

        # Center: Image View
        self.view = CustomGraphicsView()

        # Right: Filter Panel Widget
        self.filter_panel = FilterPanelWidget(self.view)

        center_layout.addWidget(left_panel, 15)
        center_layout.addWidget(self.view, 70)
        center_layout.addWidget(self.filter_panel, 15)
        parent_layout.addWidget(center_widget, 70)

    def init_bottom_section(self, parent_layout):
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        self.init_prompt_area(bottom_layout)
        self.init_reference_panel(bottom_layout)
        parent_layout.addWidget(bottom_widget, 25)

    # ============================
    # Button and Feature Actions
    # ============================
    def init_left_buttons(self, layout):
        # --- YOLO Detection Toggle ---
        self.detection_toggle_button = QPushButton("Detection: OFF")
        self.detection_toggle_button.setCheckable(True)
        self.detection_toggle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.detection_toggle_button.clicked.connect(self.toggle_detection_action)
        layout.addWidget(self.detection_toggle_button)

        # --- U²‑Net Auto Salient Object Selection ---
        auto_select_button = QPushButton("Auto Select Salient Object")
        auto_select_button.setCursor(Qt.CursorShape.PointingHandCursor)
        auto_select_button.clicked.connect(self.u2net_auto_action)
        layout.addWidget(auto_select_button)

        # --- Transform Mode and SAM Selection ---
        mode_map = {
            "Transform": "transform",
            "Select Object": "selection"
        }
        for text, mode in mode_map.items():
            btn = QPushButton(text)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda checked, m=mode: self.set_mode_action(m))
            layout.addWidget(btn)
            self.mode_buttons[mode] = btn

        # --- SAM: Deselect Selection ---
        deselect_button = QPushButton("Deselect Selection")
        deselect_button.setCursor(Qt.CursorShape.PointingHandCursor)
        deselect_button.clicked.connect(self.apply_action)
        layout.addWidget(deselect_button)

        # --- 4k Resolution Upscaling ---
        upscale_button = QPushButton("4k Resolution")
        upscale_button.setCursor(Qt.CursorShape.PointingHandCursor)
        upscale_button.clicked.connect(self.upscale_image_action)
        layout.addWidget(upscale_button)

        # --- Control Net Processing ---
        controlnet_generate_button = QPushButton("Control Net Generate")
        controlnet_generate_button.setCursor(Qt.CursorShape.PointingHandCursor)
        controlnet_generate_button.clicked.connect(self.control_net_action)
        layout.addWidget(controlnet_generate_button)

        layout.addStretch(1)

    def init_prompt_area(self, layout):
        prompt_layout = QVBoxLayout()
        prompt_label = QLabel("Prompt:")
        self.prompt_field = QTextEdit()
        self.default_prompt = (
            "Example: This photo was taken at [Location, Country].\n"
            "Generate a realistic extension of the scene, preserving its color, lighting, and texture.\n"
            "Use reference images (if available) to maintain consistency in style and detail."
        )
        self.prompt_field.setPlainText(self.default_prompt)
        prompt_layout.addWidget(prompt_label)
        prompt_layout.addWidget(self.prompt_field)
        restore_button = QPushButton("Restore Default Prompt")
        restore_button.setCursor(Qt.CursorShape.PointingHandCursor)
        restore_button.clicked.connect(self.restore_default_prompt)
        prompt_layout.addWidget(restore_button)
        layout.addLayout(prompt_layout, 1)

    def init_reference_panel(self, layout):
        reference_container = QWidget()
        reference_vlayout = QVBoxLayout(reference_container)
        reference_vlayout.setContentsMargins(0, 0, 0, 0)
        button_layout = QHBoxLayout()
        add_button = QPushButton("Add Reference Images")
        add_button.setCursor(Qt.CursorShape.PointingHandCursor)
        add_button.clicked.connect(self.add_reference_images)
        button_layout.addWidget(add_button)
        delete_button = QPushButton("Delete Selected Reference Images")
        delete_button.setCursor(Qt.CursorShape.PointingHandCursor)
        delete_button.clicked.connect(self.delete_selected_reference_images)
        button_layout.addWidget(delete_button)
        reference_vlayout.addLayout(button_layout)
        self.reference_list_widget = QListWidget()
        self.reference_list_widget.setIconSize(QSize(50, 50))
        self.reference_list_widget.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        reference_vlayout.addWidget(self.reference_list_widget)
        layout.addWidget(reference_container, 1)

    # ============================
    # File Operations and Image I/O
    # ============================
    def open_image(self):
        file_dialog = QFileDialog()
        image_file, _ = file_dialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if image_file:
            self.view.load_image(image_file)
            self.current_file = image_file
            self.filter_panel.refresh_thumbnails()
            if self.detection_enabled:
                self.update_detection()

    def save_image(self):
        if self.current_file:
            self.view.save(self.current_file)
            QMessageBox.information(self, "Save", "Image saved successfully!")
        else:
            self.save_image_as()

    def save_image_as(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(self, "Save Image As", "", "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            self.view.save(file_path)

    # ============================
    # Feature: Menu Bar and Window Layout
    # ============================
    def create_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        open_action = QAction("Open", self)
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)
        save_action = QAction("Save", self)
        save_action.triggered.connect(self.save_image)
        file_menu.addAction(save_action)
        save_as_action = QAction("Save As", self)
        save_as_action.triggered.connect(self.save_image_as)
        file_menu.addAction(save_as_action)

    def adjustSize(self):
        screen = QApplication.primaryScreen().availableGeometry()
        width = int(screen.width() * 0.6)
        height = int(screen.height() * 0.8)
        self.setGeometry(QRect(0, 0, width, height))
        self.center()

    def center(self):
        qr = self.frameGeometry()
        cp = QScreen.availableGeometry(QApplication.primaryScreen()).center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def restore_default_prompt(self):
        self.prompt_field.setPlainText(self.default_prompt)

    # ============================
    # Feature: Reference Images Management
    # ============================
    def refresh_reference_list(self):
        self.reference_list_widget.clear()
        if os.path.exists(self.reference_dir):
            for file in os.listdir(self.reference_dir):
                file_path = os.path.join(self.reference_dir, file)
                if os.path.isfile(file_path):
                    icon = QIcon(QPixmap(file_path).scaled(
                        50, 50,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    ))
                    item = QListWidgetItem(icon, file)
                    item.setData(Qt.ItemDataRole.UserRole, file_path)
                    self.reference_list_widget.addItem(item)

    def add_reference_images(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_dialog.setNameFilter("Image Files (*.png *.jpg *.jpeg *.bmp *.gif)")
        if file_dialog.exec():
            files = file_dialog.selectedFiles()
            for file in files:
                basename = os.path.basename(file)
                destination = os.path.join(self.reference_dir, basename)
                if os.path.exists(destination):
                    destination = os.path.join(self.reference_dir, f"{uuid.uuid4().hex}_{basename}")
                try:
                    shutil.copy(file, destination)
                except Exception as e:
                    print(f"Error copying file {file} to {destination}: {e}")
            self.refresh_reference_list()

    def delete_selected_reference_images(self):
        selected_items = self.reference_list_widget.selectedItems()
        for item in selected_items:
            file_path = item.data(Qt.ItemDataRole.UserRole)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
        self.refresh_reference_list()

    # ============================
    # Feature: YOLO Detection
    # ============================
    def toggle_detection_action(self):
        self.detection_enabled = not self.detection_enabled
        if self.detection_enabled:
            self.detection_toggle_button.setText("Detection: ON")
            if hasattr(self.view, 'base_cv_image') and self.view.base_cv_image is not None:
                self.update_detection()
        else:
            self.detection_toggle_button.setText("Detection: OFF")
            if hasattr(self.view, 'detection_overlay_item') and self.view.detection_overlay_item is not None:
                self.view.scene.removeItem(self.view.detection_overlay_item)
                self.view.detection_overlay_item = None

    def update_detection(self):
        if not hasattr(self.view, 'base_cv_image') or self.view.base_cv_image is None:
            return

        frame = self.view.base_cv_image.copy()
        height, width = frame.shape[:2]
        has_alpha = (frame.ndim == 3 and frame.shape[2] == 4)
        alpha_channel = frame[:, :, 3] if has_alpha else None
        if has_alpha:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        try:
            objects = detect_objects(frame, alpha_channel)
            if not objects:
                if hasattr(self.view, 'detection_overlay_item') and self.view.detection_overlay_item is not None:
                    self.view.scene.removeItem(self.view.detection_overlay_item)
                    self.view.detection_overlay_item = None
                return

            grouped_clusters = group_objects(objects)
            focus_group = select_focus_object(grouped_clusters, frame.shape)
            overlay = np.zeros((height, width, 4), dtype=np.uint8)
            if focus_group is not None:
                blue = (0, 0, 255, 255)
                red = (255, 0, 0, 255)
                scale_factor = 0.95
                for member in focus_group.get("members", []):
                    bx1, by1, bx2, by2 = member["bbox"]
                    box_width = bx2 - bx1
                    box_height = by2 - by1
                    new_width = int(box_width * scale_factor)
                    new_height = int(box_height * scale_factor)
                    center_x = bx1 + box_width // 2
                    center_y = by1 + box_height // 2
                    new_bx1 = center_x - new_width // 2
                    new_by1 = center_y - new_height // 2
                    new_bx2 = new_bx1 + new_width
                    new_by2 = new_by1 + new_height
                    cv2.rectangle(overlay, (new_bx1, new_by1), (new_bx2, new_by2), blue, thickness=2)
                    label = member.get("label", "object")
                    confidence = member.get("confidence", 0)
                    debug_text = f"{label} {confidence:.2f}"
                    cv2.putText(overlay, debug_text, (new_bx1, new_by1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, blue, 1, cv2.LINE_AA)
                x1, y1, x2, y2 = focus_group["bbox"]
                cv2.rectangle(overlay, (x1, y1), (x2, y2), red, thickness=3)

            q_image = QImage(overlay.data, width, height, width * 4, QImage.Format.Format_RGBA8888)
            overlay_pixmap = QPixmap.fromImage(q_image)
            if hasattr(self.view, 'detection_overlay_item') and self.view.detection_overlay_item is not None:
                self.view.detection_overlay_item.setPixmap(overlay_pixmap)
            else:
                self.view.detection_overlay_item = QGraphicsPixmapItem(overlay_pixmap)
                self.view.detection_overlay_item.setZValue(20)
                self.view.detection_overlay_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                self.view.scene.addItem(self.view.detection_overlay_item)
                self.view.detection_overlay_item.setPos(self.view.main_pixmap_item.pos())

        except Exception as e:
            QMessageBox.critical(self, "Detection Error", f"An error occurred during detection:\n{str(e)}")

    # ============================
    # Feature: U²‑Net Salient Object Segmentation
    # ============================
    def u2net_auto_action(self):
        if not hasattr(self.view, 'cv_image') or self.view.cv_image is None:
            QMessageBox.warning(self, "Auto Salient Object", "No image loaded.")
            return
        try:
            mask = U2NetProvider.get_salient_mask(self.view.cv_image)
            self.view.auto_selection_mask = mask
            self.view.update_auto_selection_display()
            QMessageBox.information(self, "Auto Salient Object", "Salient object segmentation completed.")
            U2NetProvider._session = None
            torch.cuda.empty_cache()
            print("U2NET resources cleaned up.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred:\n{str(e)}")

    # ============================
    # Feature: Transform Mode & SAM Selection
    # ============================

    def apply_action(self):
        """Merge the selection into the main image and update detection if enabled."""
        self.view.apply_merge()
        self.view.base_cv_image = self.view.cv_image.copy()
        if self.detection_enabled:
            self.update_detection()

    def set_mode_action(self, mode: str):
        self.view.set_mode(mode)
        self.update_active_button(mode)
        # Clean up SAM resources when switching away from selection mode
        if mode != "selection" and (SAMModelProvider._model is not None or
                                    SAMModelProvider._predictor is not None or
                                    SAMModelProvider._auto_mask_generator is not None):
            SAMModelProvider._model = None
            SAMModelProvider._predictor = None
            SAMModelProvider._auto_mask_generator = None
            torch.cuda.empty_cache()
            print("SAM resources cleaned up.")

    def update_active_button(self, active_mode: str):
        for mode, btn in self.mode_buttons.items():
            btn.setStyleSheet("background-color: #87CEFA;" if mode == active_mode else "")

    # ============================
    # Feature: 4k Resolution Upscaling
    # ============================
    def upscale_image_action(self):
        if not hasattr(self.view, 'cv_image') or self.view.cv_image is None:
            QMessageBox.warning(self, "4k Resolution", "No image loaded for upscaling.")
            return
        try:
            current_image = self.view.cv_image.copy()
            upscaled_image = RealESRGANProvider.upscale(current_image)
            self.view.cv_image = upscaled_image
            self.view.base_cv_image = upscaled_image.copy()
            self.view._update_cv_image_conversions()

            h, w, ch = self.view.cv_image_rgba.shape
            bytes_per_line = ch * w
            qimage = QImage(self.view.cv_image_rgba.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888)
            pixmap = QPixmap.fromImage(qimage)
            self.view.main_pixmap_item.setPixmap(pixmap)
            self.view.setSceneRect(self.view.main_pixmap_item.boundingRect())
            self.view.fitInView(self.view.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

            QMessageBox.information(self, "4k Resolution", "Image successfully upscaled.")
            RealESRGANProvider._instance = None
            torch.cuda.empty_cache()
            print("RealESRGAN resources cleaned up.")
        except Exception as e:
            QMessageBox.critical(self, "Upscale Error", f"An error occurred during upscaling: {str(e)}")

    # ============================
    # Feature: Control Net Processing
    # ============================
    def control_net_action(self):
        if not hasattr(self.view, 'cv_image') or self.view.cv_image is None:
            QMessageBox.warning(self, "Control Net", "No image loaded for processing.")
            return

        cv_img = self.view.cv_image
        if len(cv_img.shape) == 3:
            if cv_img.shape[2] == 3:
                rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
            elif cv_img.shape[2] == 4:
                rgba_image = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGBA)
                pil_image = Image.fromarray(rgba_image)
            else:
                QMessageBox.warning(self, "Control Net", "Unsupported image format.")
                return
        else:
            QMessageBox.warning(self, "Control Net", "Unsupported image format.")
            return

        pil_image_rgba = pil_image.convert("RGBA")
        original_size = pil_image_rgba.size
        alpha = pil_image_rgba.split()[3]
        mask = alpha.point(lambda p: 255 if p < 128 else 0).convert("L")
        pil_image_rgb = pil_image_rgba.convert("RGB")
        adjusted_size = make_divisible_by_8(original_size)

        reference_images = []
        for img_path in glob.glob(os.path.join(self.reference_dir, "*.*")):
            try:
                ref_img = Image.open(img_path).convert("RGB")
                reference_images.append(ref_img)
            except Exception as e:
                print(f"Error loading reference image {img_path}: {e}")

        prompt = self.prompt_field.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Control Net", "Please enter a prompt for inpainting.")
            return

        pipe = load_controlnet()
        try:
            result = pipe(
                prompt=prompt,
                image=pil_image_rgb.resize(adjusted_size, Image.Resampling.LANCZOS),
                mask_image=mask.resize(adjusted_size, Image.Resampling.LANCZOS),
                conditioning_image=[img.resize(adjusted_size, Image.Resampling.LANCZOS) for img in reference_images] if reference_images else None,
                height=adjusted_size[1],
                width=adjusted_size[0]
            ).images[0]

            result = result.resize(original_size, Image.Resampling.LANCZOS)
            qimage = ImageQt(result)
            pixmap = QPixmap.fromImage(qimage)
            self.view.main_pixmap_item.setPixmap(pixmap)
            self.view.background_pixmap = pixmap

            result_np = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
            self.view.cv_image = result_np
            self.view.base_cv_image = self.view.cv_image.copy()

            QMessageBox.information(self, "Control Net", "Control Net processing completed.")
            del pipe
            torch.cuda.empty_cache()
            print("ControlNet resources cleaned up.")

            if self.detection_enabled:
                self.update_detection()

        except Exception as e:
            QMessageBox.critical(self, "Control Net Error", f"An error occurred: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
