from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        # self.controller = None
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 2px dashed gray;")
        self.setText("이미지를 드래그하거나 Load 버튼을 사용하세요")

    def set_controller(self, controller):
        self.controller = controller

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self.controller.load_image_from_path(path)