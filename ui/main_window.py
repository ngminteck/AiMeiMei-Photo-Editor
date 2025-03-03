# ui/main_window.py
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QPushButton, QGraphicsView, QGraphicsScene, QFileDialog,
                             QMenuBar, QMenu, QMessageBox)
from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QScreen, QPixmap, QPainter, QAction
from .custom_graphics_view import CustomGraphicsView


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_file = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Editor')

        # Create central widget and layout
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)

        # Create CustomGraphicsView
        self.view = CustomGraphicsView()
        layout.addWidget(self.view, 1)  # Give it more vertical space

        # Create buttons
        buttons = ['Transform', 'Prompt', 'Auto', 'Apply']
        for button_text in buttons:
            button = QPushButton(button_text)
            layout.addWidget(button)
            button.clicked.connect(getattr(self, f"{button_text.lower()}_action"))

        self.setCentralWidget(central_widget)

        # Create menu bar
        self.create_menu_bar()

        # Set window size based on screen
        self.adjustSize()

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
        width = int(screen.width() * 0.6)  # 60% of screen width
        height = int(screen.height() * 0.8)  # 80% of screen height
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

    def transform_action(self):
        self.view.set_mode("transform")

    def prompt_action(self):
       self.view.set_mode("selection")

    def auto_action(self):
       self.view.set_mode("auto")

    def apply_action(self):
        self.view.apply_merge()

'''
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.view = CustomGraphicsView()
        self.filepath = None
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
        #self.layout().addChildWidget(button_group)      
        self.layout().add
        self.setWindowTitle("AiMeiMei Photo Editor")
        self.resize(1920, 1080)
        self.createMenu()
        # Update the image path if needed.

    def createMenu(self):
        menubar = self.menuBar()
        fileMenu = menubar.addMenu("&File")

        openAction = QAction("&Open File", self)
        openAction.setShortcut("Ctrl+O")
        openAction.triggered.connect(self.OnOpen)
        fileMenu.addAction(openAction)

        saveAction = QAction("&Save File", self)
        saveAction.setShortcut("Ctrl+S")
        saveAction.triggered.connect(self.OnSave)
        fileMenu.addAction(saveAction)

        saveAsAction = QAction("&Save File As", self)
        saveAsAction.setShortcut("Ctrl+Shift+S")
        saveAsAction.triggered.connect(self.OnSaveAs)
        fileMenu.addAction(saveAsAction)
       

    def OnOpen(self):
        # Load an image file to be displayed (will popup a file dialog).
        self.filepath, dummy = QFileDialog.getOpenFileName(self, "Open image file.")
        if len(self.filepath) and os.path.isfile(self.filepath):
            self.view.load_image(self.filepath)
       
    def OnSave(self):
       self.view.save()

    def OnSaveAs(self):
        name, ext = os.path.splitext(self.image_viewer._current_filename)
        dialog = QFileDialog()
        dialog.setDefaultSuffix("png")
        extension_filter = "Default (*.png);;BMP (*.bmp);;Icon (*.ico);;JPEG (*.jpeg *.jpg);;PBM (*.pbm);;PGM (*.pgm);;PNG (*.png);;PPM (*.ppm);;TIF (*.tif *.tiff);;WBMP (*.wbmp);;XBM (*.xbm);;XPM (*.xpm)"
        name = dialog.getSaveFileName(self, 'Save File', name + ".png", extension_filter)
        print("The saved file name is : ",name[0])
        self.view.save(name[0])
        self.filepath = name[0]

'''