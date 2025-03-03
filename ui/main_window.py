# ui/main_window.py
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QMenuBar, QMessageBox, QPlainTextEdit, QLabel
)
from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QScreen, QPixmap, QPainter, QAction
from .custom_graphics_view import CustomGraphicsView


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_file = None
        self.mode_buttons = {}  # To store mode buttons
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Editor')

        # Create the main central widget and its layout (vertical)
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        # --- Top area layout: left (buttons), center (image view), right (layer info) ---
        top_layout = QHBoxLayout()

        # Left Panel: Buttons
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        # Mode buttons mapping: button text -> mode string.
        mode_map = {
            "Transform": "transform",
            "Prompt": "selection",
            "Auto": "auto"
        }
        for text, mode in mode_map.items():
            btn = QPushButton(text)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            left_layout.addWidget(btn)
            btn.clicked.connect(lambda checked, m=mode: self.set_mode_action(m))
            self.mode_buttons[mode] = btn

        # Add the Apply button as well
        apply_button = QPushButton("Apply")
        apply_button.setCursor(Qt.CursorShape.PointingHandCursor)
        left_layout.addWidget(apply_button)
        apply_button.clicked.connect(self.apply_action)
        left_layout.addStretch(1)  # Push buttons to the top

        # Center Panel: Image view (CustomGraphicsView)
        self.view = CustomGraphicsView()

        # Right Panel: Layer Info placeholder (using a QPlainTextEdit)
        self.layer_info = QPlainTextEdit()
        self.layer_info.setReadOnly(True)
        self.layer_info.setPlaceholderText("Layer info will be shown here")
        self.layer_info.setMaximumWidth(200)  # Limit the width to 200 pixels

        # Add left, center, right panels to the top layout.
        # Adjust stretch factors as needed.
        top_layout.addWidget(left_panel, 1)
        top_layout.addWidget(self.view, 4)
        top_layout.addWidget(self.layer_info, 2)

        # --- Bottom area: Directory Info ---
        self.directory_info = QLabel("Directory Info: ")

        # Add the top layout and the directory info widget to the main layout.
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.directory_info)

        self.setCentralWidget(central_widget)

        # Create menu bar.
        self.create_menu_bar()

        # Set window size based on screen.
        self.adjustSize()

        # Set default mode to "transform" and update button style.
        self.set_mode_action("transform")

    def set_mode_action(self, mode):
        self.view.set_mode(mode)
        self.update_active_button(mode)

    def update_active_button(self, active_mode):
        # Update style for each mode button.
        for mode, btn in self.mode_buttons.items():
            if mode == active_mode:
                btn.setStyleSheet("background-color: #87CEFA;")  # Light blue for active.
            else:
                btn.setStyleSheet("")

    def create_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')

        open_action = QAction('Open', self)
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)

        save_action = QAction('Save', self)
        save_action.triggered.connect(self.save_image)
        file_menu.addAction(save_action)

        save_as_action = QAction('Save As', self)
        save_as_action.triggered.connect(self.save_image_as)
        file_menu.addAction(save_as_action)

    def adjustSize(self):
        screen = QApplication.primaryScreen().availableGeometry()
        width = int(screen.width() * 0.6)  # 60% of screen width.
        height = int(screen.height() * 0.8)  # 80% of screen height.
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
            # Update directory info when an image is opened.
            self.directory_info.setText(f"Directory Info: {image_file}")

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

