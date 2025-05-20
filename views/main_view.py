from PyQt5.QtWidgets import QLabel, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView
from PyQt5.QtCore import Qt
from views.widgets.ImageLabel import ImageLabel
from controllers.main_controller import MainController

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Find_Head_Top_View")
        self.setGeometry(100, 100, 800, 600)

        self.image_label = ImageLabel()
        self.image_label.setMinimumHeight(600)
        self.image_label.setMaximumHeight(800)
        self.image_label.setMinimumWidth(600)

        self.load_button = QPushButton("Load")
        self.rotate_button = QPushButton("Rotate")
        self.flip_button = QPushButton("Flip")
        self.contour_button = QPushButton("Contours")

        # CI/CVAI 테이블
        self.table = QTableWidget(2, 3)  # 2행 3열
        self.table.setHorizontalHeaderLabels(["Value", "Normal Range", "Server Range"])
        self.table.setVerticalHeaderLabels(["CI", "CVAI"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # 표 셀 채우기
        self.table.setItem(0, 0, QTableWidgetItem(""))  # CI Value
        self.table.setItem(0, 1, QTableWidgetItem("< 85.0"))  # CI Normal
        self.table.setItem(0, 2, QTableWidgetItem("> 95.0"))  # CI Server

        self.table.setItem(1, 0, QTableWidgetItem(""))  # CVAI Value
        self.table.setItem(1, 1, QTableWidgetItem("< 3.50"))  # CVAI Normal
        self.table.setItem(1, 2, QTableWidgetItem("> 8.75"))  # CVAI Server

        self.table.setEditTriggers(QTableWidget.NoEditTriggers)

        header_height = self.table.horizontalHeader().height()
        total_table_height = header_height * 3 + 2
        self.table.setFixedHeight(total_table_height)

        # 범례
        self.legend_label = QLabel()
        self.legend_label.setText(
            '<span style="color:green;">■</span> Normal &nbsp;&nbsp;'
            '<span style="color:yellow;">■</span> Moderate &nbsp;&nbsp;'
            '<span style="color:red;">■</span> Severe'
        )
        self.legend_label.setAlignment(Qt.AlignLeft)
        self.legend_label.setFixedHeight(30)

        # 버튼 레이아웃
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.rotate_button)
        button_layout.addWidget(self.flip_button)
        button_layout.addWidget(self.contour_button)

        # # table 레이아웃
        table_layout = QVBoxLayout()
        table_layout.addWidget(self.table)
        table_layout.addWidget(self.legend_label)

        # 전체 레이아웃
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addLayout(table_layout)
        layout.addLayout(button_layout)
        self.setLayout(layout)

        # 컨트롤러 연결
        self.controller = MainController(self)
        self.image_label.set_controller(self.controller)