from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import Qt
from views.widgets.ImageLabel import ImageLabel
from controllers.main_controller import MainController

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Find_Head_Top_View")
        self.setGeometry(100, 100, 800, 600)

        self.image_label = ImageLabel()

        self.load_button = QPushButton("Load")
        self.rotate_button = QPushButton("Rotate")
        self.flip_button = QPushButton("Flip")
        self.contour_button = QPushButton("Contours")

        # 버튼 레이아웃
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.rotate_button)
        button_layout.addWidget(self.flip_button)
        button_layout.addWidget(self.contour_button)

        # 전체 레이아웃
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addLayout(button_layout)
        self.setLayout(layout)

        # 컨트롤러 연결
        self.controller = MainController(self)
        self.image_label.set_controller(self.controller)