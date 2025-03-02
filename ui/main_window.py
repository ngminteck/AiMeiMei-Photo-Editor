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

        self.prompt_selection_button = QPushButton("Prompt Selection Mode")
        self.prompt_selection_button.setCheckable(True)
        self.auto_selection_button = QPushButton("Auto Selection Mode")
        self.auto_selection_button.setCheckable(True)
        self.transform_button = QPushButton("Transform Mode")
        self.transform_button.setCheckable(True)

        # Set default mode.
        self.prompt_selection_button.setChecked(True)
        self.auto_selection_button.setChecked(False)
        self.transform_button.setChecked(False)

        button_group = QButtonGroup(self)
        button_group.setExclusive(True)
        button_group.addButton(self.prompt_selection_button)
        button_group.addButton(self.auto_selection_button)
        button_group.addButton(self.transform_button)

        self.prompt_selection_button.clicked.connect(lambda: self.view.set_mode("selection"))
        self.auto_selection_button.clicked.connect(lambda: self.view.set_mode("auto"))
        self.transform_button.clicked.connect(lambda: self.view.set_mode("transform"))

        self.apply_merge_button = QPushButton("Apply Merge")
        self.apply_merge_button.clicked.connect(lambda: self.view.apply_merge())

        layout.addWidget(self.view)
        layout.addWidget(self.prompt_selection_button)
        layout.addWidget(self.auto_selection_button)
        layout.addWidget(self.transform_button)
        layout.addWidget(self.apply_merge_button)

        self.setLayout(layout)
        self.setWindowTitle("SAM: Merged Selection, Transformation & Apply Tool")
        self.resize(800, 600)
        # Update the image path if needed.
        self.view.load_image("images/test/2_people_together.png")
