from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QMenuBar, QMessageBox, QLabel, QTextEdit, QScrollArea
)
from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QScreen, QPixmap, QAction
from .custom_graphics_view import CustomGraphicsView
from providers.controlnet_model_provider import load_controlnet, make_divisible_by_8
import cv2
from PIL import Image
from PIL.ImageQt import ImageQt
import glob
import os
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_file = None
        self.mode_buttons = {}  # To store mode buttons
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Image Editor")

        # Main vertical layout for the window.
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        # ---------------------------
        # Top Section (≈5% height)
        # ---------------------------
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        # Display the current image aesthetic score and related info.
        score_label = QLabel("Aesthetic Score: 85 | Position: Good | Angle: Optimal | Brightness: Balanced | Focus: Sharp")
        score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top_layout.addWidget(score_label)

        # ---------------------------
        # Center Section (≈70% height)
        # ---------------------------
        center_widget = QWidget()
        center_layout = QHBoxLayout(center_widget)

        # Left: Button panel (≈15% width)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        # Add mode buttons.
        mode_map = {
            "Transform": "transform",
            "Prompt": "selection",
            "Auto": "auto"
        }
        for text, mode in mode_map.items():
            btn = QPushButton(text)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda checked, m=mode: self.set_mode_action(m))
            left_layout.addWidget(btn)
            self.mode_buttons[mode] = btn

        # Additional control buttons.
        apply_button = QPushButton("Apply")
        apply_button.setCursor(Qt.CursorShape.PointingHandCursor)
        apply_button.clicked.connect(self.apply_action)
        left_layout.addWidget(apply_button)

        controlnet_button = QPushButton("Control Net")
        controlnet_button.setCursor(Qt.CursorShape.PointingHandCursor)
        controlnet_button.clicked.connect(self.control_net_action)
        left_layout.addWidget(controlnet_button)
        left_layout.addStretch(1)

        # Center: Image view (≈70% width)
        self.view = CustomGraphicsView()

        # Right: Layer info panel (≈15% width)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.layer_info = QTextEdit()
        self.layer_info.setReadOnly(True)
        self.layer_info.setPlaceholderText("Layer info will be shown here")
        right_layout.addWidget(self.layer_info)
        right_layout.addStretch(1)

        # Add widgets to center_layout with stretch factors (15, 70, 15)
        center_layout.addWidget(left_panel, 15)
        center_layout.addWidget(self.view, 70)
        center_layout.addWidget(right_panel, 15)

        # ---------------------------
        # Bottom Section (≈25% height)
        # ---------------------------
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        # Left: Prompt area (multi-line)
        prompt_layout = QVBoxLayout()
        prompt_label = QLabel("Prompt:")
        self.prompt_field = QTextEdit()
        self.prompt_field.setPlaceholderText("Enter your prompt here for Control Net inpainting...")
        prompt_layout.addWidget(prompt_label)
        prompt_layout.addWidget(self.prompt_field)
        # Set equal width (50%)
        bottom_layout.addLayout(prompt_layout, 1)

        # Right: Reference thumbnails panel in a scroll area.
        reference_widget = QWidget()
        reference_layout = QHBoxLayout(reference_widget)
        reference_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        reference_images_dir = "images/reference_images"
        if os.path.exists(reference_images_dir):
            for img_path in glob.glob(f"{reference_images_dir}/*.*"):
                try:
                    thumb = QPixmap(img_path).scaled(30, 30, Qt.AspectRatioMode.KeepAspectRatio,
                                                      Qt.TransformationMode.SmoothTransformation)
                    thumb_label = QLabel()
                    thumb_label.setPixmap(thumb)
                    thumb_label.setToolTip(os.path.basename(img_path))
                    reference_layout.addWidget(thumb_label)
                except Exception as e:
                    print(f"Error loading reference image {img_path}: {e}")
        else:
            error_label = QLabel("No reference images found.")
            reference_layout.addWidget(error_label)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(reference_widget)
        scroll_area.setFixedHeight(30)
        # Set equal width (50%)
        bottom_layout.addWidget(scroll_area, 1)

        # ---------------------------
        # Assemble Main Layout
        # ---------------------------
        # Vertical stretch factors: top:5, center:70, bottom:25 (approx 5%, 70%, 25%)
        main_layout.addWidget(top_widget, 5)
        main_layout.addWidget(center_widget, 70)
        main_layout.addWidget(bottom_widget, 25)

        self.setCentralWidget(central_widget)
        self.create_menu_bar()
        self.adjustSize()
        self.set_mode_action("transform")

    def set_mode_action(self, mode):
        self.view.set_mode(mode)
        self.update_active_button(mode)

    def update_active_button(self, active_mode):
        for mode, btn in self.mode_buttons.items():
            if mode == active_mode:
                btn.setStyleSheet("background-color: #87CEFA;")
            else:
                btn.setStyleSheet("")

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

    def open_image(self):
        file_dialog = QFileDialog()
        image_file, _ = file_dialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if image_file:
            self.view.load_image(image_file)
            self.current_file = image_file

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

    def apply_action(self):
        self.view.apply_merge()

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

        reference_images_dir = "images/reference_images"
        reference_images = []
        if os.path.exists(reference_images_dir):
            for img_path in glob.glob(f"{reference_images_dir}/*.*"):
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

            QMessageBox.information(self, "Control Net", "Control Net processing completed.")
        except Exception as e:
            QMessageBox.critical(self, "Control Net Error", f"An error occurred: {str(e)}")
