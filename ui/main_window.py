# main_window.py

import os
import shutil
import glob
import uuid
import cv2
import numpy as np
import pilgram

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QMessageBox, QLabel, QTextEdit,
    QToolButton, QScrollArea, QListWidget, QListWidgetItem, QGraphicsPixmapItem
)
from PyQt6.QtGui import QScreen, QPixmap, QAction, QImage, QPainter, QIcon
from PyQt6.QtCore import Qt, QRect, QSize, QBuffer, QIODevice

from PIL import Image, ImageDraw
from PIL.ImageQt import ImageQt

# Local modules
from ui.custom_graphics_view import CustomGraphicsView
from providers.controlnet_model_provider import load_controlnet, make_divisible_by_8
from providers.yolo_detection_provider import detect_main_object, detect_objects, group_objects, select_focus_object
from providers.realesrgan_provider import RealESRGANProvider


# ------------------------------------------------------------------
# FilterPanelWidget: vertical, scrollable panel on the right side.
# ------------------------------------------------------------------
class FilterPanelWidget(QWidget):
    def __init__(self, view: CustomGraphicsView, parent=None):
        super().__init__(parent)
        self.view = view  # CustomGraphicsView instance
        self.original_image = None  # PIL image will be stored here
        self.available_filters = self.get_available_filters()
        self.buttonIconSize = QSize(200, 200)
        self.initUI()

    def get_available_filters(self):
        """
        Returns an explicit dictionary mapping filter names to pilgram filter functions.
        """
        filters = {
            "1977": pilgram._1977,
            "Aden": pilgram.aden,
            "Brannan": pilgram.brannan,
            "Brooklyn": pilgram.brooklyn,
            "Clarendon": pilgram.clarendon,
            "Earlybird": pilgram.earlybird,
            "Gingham": pilgram.gingham,
            "Hudson": pilgram.hudson,
            "Inkwell": pilgram.inkwell,
            "Kelvin": pilgram.kelvin,
            "Lark": pilgram.lark,
            "Lofi": pilgram.lofi,
            "Maven": pilgram.maven,
            "Mayfair": pilgram.mayfair,
            "Moon": pilgram.moon,
            "Nashville": pilgram.nashville,
            "Perpetua": pilgram.perpetua,
            "Reyes": pilgram.reyes,
            "Rise": pilgram.rise,
            "Slumber": pilgram.slumber,
            "Stinson": pilgram.stinson,
            "Toaster": pilgram.toaster,
            "Valencia": pilgram.valencia,
            "Walden": pilgram.walden,
            "Willow": pilgram.willow,
            "Xpro2": pilgram.xpro2,
        }
        return filters

    def create_checkerboard(self, size):
        """Generate a checkerboard PIL image of the given size."""
        tile_size = 20
        width, height = size
        checkerboard = Image.new("RGBA", size, "white")
        draw = ImageDraw.Draw(checkerboard)
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                if (x // tile_size + y // tile_size) % 2 == 0:
                    draw.rectangle([x, y, x + tile_size - 1, y + tile_size - 1], fill="lightgray")
        return checkerboard

    def composite_with_checkerboard(self, image):
        """Composite image with a checkerboard background if it has transparency."""
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        bg = self.create_checkerboard(image.size)
        return Image.alpha_composite(bg, image)

    def ImageToQPixmap(self, image):
        """
        Convert a PIL image to a QPixmap scaled for thumbnails.
        """
        if image is None:
            image = self.create_checkerboard((self.buttonIconSize.width(), self.buttonIconSize.height()))
        elif "A" in image.getbands():
            image = self.composite_with_checkerboard(image)
        qimage = ImageQt(image.convert("RGBA"))
        pix = QPixmap.fromImage(qimage)
        return pix.scaled(self.buttonIconSize, Qt.AspectRatioMode.KeepAspectRatio,
                          Qt.TransformationMode.SmoothTransformation)

    def PILImageToQPixmap(self, image):
        """
        Convert a PIL image to QPixmap without scaling.
        """
        if image is None:
            rect = self.view.scene.sceneRect()
            image = self.create_checkerboard((int(rect.width()), int(rect.height())))
        elif "A" in image.getbands():
            image = self.composite_with_checkerboard(image)
        qimage = ImageQt(image.convert("RGBA"))
        return QPixmap.fromImage(qimage)

    def initUI(self):
        """Initialize UI with a scroll area to hold filter buttons."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        self.layout = QVBoxLayout(container)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(container)
        mainLayout = QVBoxLayout(self)
        mainLayout.addWidget(scroll)
        self.setLayout(mainLayout)

    def get_filtered_image_with_alpha(self, image, filter_function):
        """
        Apply a Pilgram filter while preserving the alpha channel.
        """
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        r, g, b, a = image.split()
        rgb_image = Image.merge("RGB", (r, g, b))
        filtered_rgb = filter_function(rgb_image)
        filtered_rgba = filtered_rgb.convert("RGBA")
        filtered_rgba.putalpha(a)
        return filtered_rgba

    def refresh_thumbnails(self):
        """Refresh filter thumbnails using the current image from the view."""
        # Clear existing buttons
        while self.layout.count():
            child = self.layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        if self.view.main_pixmap_item is None:
            self.original_image = self.create_checkerboard(
                (self.buttonIconSize.width(), self.buttonIconSize.height())
            )
        else:
            qpixmap = self.view.main_pixmap_item.pixmap()
            pil_image = Image.fromqimage(qpixmap.toImage())
            self.original_image = pil_image

        # Create filter buttons for each available filter.
        for name, f in self.available_filters.items():
            filterButton = QToolButton()
            try:
                filtered = self.get_filtered_image_with_alpha(self.original_image.copy(), f)
            except Exception as e:
                print(f"Error applying filter {name}: {e}")
                filtered = self.original_image.copy()
            filteredPixmap = self.ImageToQPixmap(filtered)
            filterButton.setIcon(QIcon(filteredPixmap))
            filterButton.setIconSize(self.buttonIconSize)
            filterButton.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            filterButton.setText(name)
            filterButton.setObjectName(name)
            filterButton.clicked.connect(self.OnFilterSelect)
            self.layout.addWidget(filterButton)

    def OnFilterSelect(self):
        """
        Apply the selected filter to the original image and update the main view.
        """
        button = self.sender()
        filterName = button.objectName()
        filterFunction = self.available_filters.get(filterName)
        if not filterFunction:
            return

        filtered = self.get_filtered_image_with_alpha(self.original_image.copy(), filterFunction)
        qpixmap_filtered = self.PILImageToQPixmap(filtered)
        if self.view.main_pixmap_item is not None:
            self.view.main_pixmap_item.setPixmap(qpixmap_filtered)
        else:
            self.view.main_pixmap_item = QGraphicsPixmapItem(qpixmap_filtered)
            self.view.scene.addItem(self.view.main_pixmap_item)


# ------------------------------------------------------------------
# MainWindow class
# ------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_file = None
        self.mode_buttons = {}  # Dictionary to store mode buttons
        self.detection_enabled = False  # Toggle for detection overlay
        self.reference_dir = os.path.join("images", "reference_images")
        os.makedirs(self.reference_dir, exist_ok=True)
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Image Editor")
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        # Top Section (~5% height)
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        score_label = QLabel("Aesthetic Score: 85 | Position: Good | Angle: Optimal | Brightness: Balanced | Focus: Sharp")
        score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top_layout.addWidget(score_label)

        # Center Section (~70% height)
        center_widget = QWidget()
        center_layout = QHBoxLayout(center_widget)

        # Left: Button panel (15% width)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        mode_map = {
            "Transform": "transform",
            "Select Object": "selection"
        }
        for text, mode in mode_map.items():
            btn = QPushButton(text)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda checked, m=mode: self.set_mode_action(m))
            left_layout.addWidget(btn)
            self.mode_buttons[mode] = btn

        self.detection_toggle_button = QPushButton("Detection: OFF")
        self.detection_toggle_button.setCheckable(True)
        self.detection_toggle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.detection_toggle_button.clicked.connect(self.toggle_detection_action)
        left_layout.addWidget(self.detection_toggle_button)

        auto_select_button = QPushButton("Auto Select Salient Object")
        auto_select_button.setCursor(Qt.CursorShape.PointingHandCursor)
        auto_select_button.clicked.connect(self.u2net_auto_action)
        left_layout.addWidget(auto_select_button)

        deselect_button = QPushButton("Deselect Selection")
        deselect_button.setCursor(Qt.CursorShape.PointingHandCursor)
        deselect_button.clicked.connect(self.apply_action)
        left_layout.addWidget(deselect_button)

        upscale_button = QPushButton("4k Resolution")
        upscale_button.setCursor(Qt.CursorShape.PointingHandCursor)
        upscale_button.clicked.connect(self.upscale_image_action)
        left_layout.addWidget(upscale_button)

        controlnet_generate_button = QPushButton("Control Net Generate")
        controlnet_generate_button.setCursor(Qt.CursorShape.PointingHandCursor)
        controlnet_generate_button.clicked.connect(self.control_net_action)
        left_layout.addWidget(controlnet_generate_button)

        left_layout.addStretch(1)

        # Center: Image view (70% width)
        self.view = CustomGraphicsView()

        # Right: Filter panel (vertical, scrollable)
        self.filter_panel = FilterPanelWidget(self.view)
        center_layout.addWidget(left_panel, 15)
        center_layout.addWidget(self.view, 70)
        center_layout.addWidget(self.filter_panel, 15)

        # Bottom Section (~25% height)
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        # Left: Prompt area.
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
        bottom_layout.addLayout(prompt_layout, 1)

        # Right: Reference images panel
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
        bottom_layout.addWidget(reference_container, 1)

        main_layout.addWidget(top_widget, 5)
        main_layout.addWidget(center_widget, 70)
        main_layout.addWidget(bottom_widget, 25)

        self.setCentralWidget(central_widget)
        self.create_menu_bar()
        self.adjustSize()
        self.set_mode_action("transform")
        self.refresh_reference_list()

    def restore_default_prompt(self):
        """Restore the default prompt text."""
        self.prompt_field.setPlainText(self.default_prompt)

    def refresh_reference_list(self):
        """Refresh the list of reference images."""
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
        """Add reference images from file dialog."""
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
        """Delete the selected reference images."""
        selected_items = self.reference_list_widget.selectedItems()
        for item in selected_items:
            file_path = item.data(Qt.ItemDataRole.UserRole)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
        self.refresh_reference_list()

    def set_mode_action(self, mode: str):
        """Set the current mode for the view and update active button styling."""
        self.view.set_mode(mode)
        self.update_active_button(mode)

    def update_active_button(self, active_mode: str):
        """Update the styling for mode buttons."""
        for mode, btn in self.mode_buttons.items():
            btn.setStyleSheet("background-color: #87CEFA;" if mode == active_mode else "")

    def create_menu_bar(self):
        """Create a simple menu bar with File actions."""
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
        """Adjust window size based on available screen geometry."""
        screen = QApplication.primaryScreen().availableGeometry()
        width = int(screen.width() * 0.6)
        height = int(screen.height() * 0.8)
        self.setGeometry(QRect(0, 0, width, height))
        self.center()

    def center(self):
        """Center the window on the primary screen."""
        qr = self.frameGeometry()
        cp = QScreen.availableGeometry(QApplication.primaryScreen()).center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def open_image(self):
        """Open an image file and load it into the view."""
        file_dialog = QFileDialog()
        image_file, _ = file_dialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if image_file:
            self.view.load_image(image_file)
            self.current_file = image_file
            self.filter_panel.refresh_thumbnails()
            if self.detection_enabled:
                self.update_detection()

    def save_image(self):
        """Save the current image."""
        if self.current_file:
            self.view.save(self.current_file)
            QMessageBox.information(self, "Save", "Image saved successfully!")
        else:
            self.save_image_as()

    def save_image_as(self):
        """Save the image as a new file."""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(self, "Save Image As", "", "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            self.view.save(file_path)

    def apply_action(self):
        """Merge the selection into the main image and update detection if enabled."""
        self.view.apply_merge()
        self.view.base_cv_image = self.view.cv_image.copy()
        if self.detection_enabled:
            self.update_detection()

    def control_net_action(self):
        """Process the image using Control Net."""
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
            if self.detection_enabled:
                self.update_detection()

        except Exception as e:
            QMessageBox.critical(self, "Control Net Error", f"An error occurred: {str(e)}")

    def toggle_detection_action(self):
        """Toggle the detection overlay on or off."""
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
        """Update the detection overlay using YOLO-based object detection."""
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

    def u2net_auto_action(self):
        """Run U²‑Net based salient object segmentation."""
        if not hasattr(self.view, 'cv_image') or self.view.cv_image is None:
            QMessageBox.warning(self, "Auto Salient Object", "No image loaded.")
            return
        try:
            from providers.u2net_provider import U2NetProvider  # Optionally, keep as function-level import if heavy
            mask = U2NetProvider.get_salient_mask(self.view.cv_image)
            self.view.auto_selection_mask = mask
            self.view.update_auto_selection_display()
            QMessageBox.information(self, "Auto Salient Object", "Salient object segmentation completed.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred:\n{str(e)}")

    def upscale_image_action(self):
        """Upscale the current image using RealESRGAN and update all conversions."""
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

            # Optionally adjust view to show full upscaled image
            self.view.setSceneRect(self.view.main_pixmap_item.boundingRect())
            self.view.fitInView(self.view.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

            QMessageBox.information(self, "4k Resolution", "Image successfully upscaled.")
        except Exception as e:
            QMessageBox.critical(self, "Upscale Error", f"An error occurred during upscaling: {str(e)}")


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
