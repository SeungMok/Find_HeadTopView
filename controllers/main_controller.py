import cv2
import numpy as np
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class MainController:
    def __init__(self, view):
        self.view = view
        self.image = None               # 현재 이미지
        self.original_image = None      # Contours 이전 이미지
        self.contour_mode = False

        self.view.load_button.clicked.connect(self.load_image)
        self.view.rotate_button.clicked.connect(self.rotate_image)
        self.view.flip_button.clicked.connect(self.flip_image)
        self.view.contour_button.clicked.connect(self.toggle_contours)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self.view, "이미지 불러오기", "", "Image Files (*.png *.jpg *.bmp)")
        if path:
            self.load_image_from_path(path)

    def load_image_from_path(self, path):
        self.image = cv2.imread(path)
        self.original_image = self.image.copy()
        self.display_image(self.image)
        self.reset_buttons()

    def rotate_image(self):
        if self.image is not None:
            self.image = cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE)
            self.display_image(self.image)

    def flip_image(self):
        if self.image is not None:
            self.image = cv2.flip(self.image, 1)
            self.display_image(self.image)

    def toggle_contours(self):
        if not self.contour_mode:
            # Contours 모드 진입
            if self.image is not None:
                self.original_image = self.image.copy()
                gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour_img = np.zeros_like(self.image)
                cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
                self.image = contour_img
                self.display_image(self.image)

                self.contour_mode = True
                self.view.contour_button.setText("Cancel")
                self.view.flip_button.setEnabled(False)
                self.view.rotate_button.setEnabled(False)
        else:
            # 원래 이미지 복원
            self.image = self.original_image.copy()
            self.display_image(self.image)

            self.contour_mode = False
            self.view.contour_button.setText("Contours")
            self.view.flip_button.setEnabled(True)
            self.view.rotate_button.setEnabled(True)

    def display_image(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.view.image_label.width(), self.view.image_label.height(), Qt.KeepAspectRatio)
        self.view.image_label.setPixmap(pixmap)

    def reset_buttons(self):
        self.contour_mode = False
        self.view.contour_button.setText("Contours")
        self.view.flip_button.setEnabled(True)
        self.view.rotate_button.setEnabled(True)