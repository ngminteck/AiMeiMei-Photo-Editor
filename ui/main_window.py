# ui/main_window.py
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QButtonGroup
from ui.custom_graphics_view import CustomGraphicsView

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.view = CustomGraphicsView()

        # Rearranged button order: Transform button appears first.
        self.transform_button = QPushButton("Transform Mode")
        self.transform_button.setCheckable(True)
        self.prompt_selection_button = QPushButton("Prompt Selection Mode")
        self.prompt_selection_button.setCheckable(True)
        self.auto_selection_button = QPushButton("Auto Selection Mode")
        self.auto_selection_button.setCheckable(True)

        # Set default mode to transform so that no model is loaded at startup.
        self.transform_button.setChecked(True)
        self.prompt_selection_button.setChecked(False)
        self.auto_selection_button.setChecked(False)

        # Initialize the view with transform mode.
        self.view.set_mode("transform")

        button_group = QButtonGroup(self)
        button_group.setExclusive(True)
        button_group.addButton(self.transform_button)
        button_group.addButton(self.prompt_selection_button)
        button_group.addButton(self.auto_selection_button)

        self.transform_button.clicked.connect(lambda: self.view.set_mode("transform"))
        self.prompt_selection_button.clicked.connect(lambda: self.view.set_mode("selection"))
        self.auto_selection_button.clicked.connect(lambda: self.view.set_mode("auto"))

        self.apply_merge_button = QPushButton("Apply Merge")
        self.apply_merge_button.clicked.connect(lambda: self.view.apply_merge())

        # Add widgets to layout; transform button is now at the top.
        layout.addWidget(self.view)
        layout.addWidget(self.transform_button)
        layout.addWidget(self.prompt_selection_button)
        layout.addWidget(self.auto_selection_button)
        layout.addWidget(self.apply_merge_button)

        self.setLayout(layout)
        self.setWindowTitle("AiMeiMei Photo Editor")
        self.resize(1920, 1080)
        # Update the image path if needed.
        self.view.load_image("images/test/2_people_together.png")
