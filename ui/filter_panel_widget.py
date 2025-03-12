import pilgram
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QToolButton
from PyQt6.QtGui import QPixmap, QImage, QIcon
from PyQt6.QtCore import Qt, QSize
from PIL import Image, ImageDraw
from PIL.ImageQt import ImageQt

class FilterPanelWidget(QWidget):
    def __init__(self, view, parent=None):
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
        """Convert a PIL image to a QPixmap scaled for thumbnails."""
        if image is None:
            image = self.create_checkerboard((self.buttonIconSize.width(), self.buttonIconSize.height()))
        elif "A" in image.getbands():
            image = self.composite_with_checkerboard(image)
        qimage = ImageQt(image.convert("RGBA"))
        pix = QPixmap.fromImage(qimage)
        return pix.scaled(self.buttonIconSize, Qt.AspectRatioMode.KeepAspectRatio,
                          Qt.TransformationMode.SmoothTransformation)

    def PILImageToQPixmap(self, image):
        """Convert a PIL image to QPixmap without scaling."""
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
        """Apply a Pilgram filter while preserving the alpha channel."""
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
        """Apply the selected filter to the original image and update the main view."""
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
            from PyQt6.QtWidgets import QGraphicsPixmapItem
            self.view.main_pixmap_item = QGraphicsPixmapItem(qpixmap_filtered)
            self.view.scene.addItem(self.view.main_pixmap_item)
