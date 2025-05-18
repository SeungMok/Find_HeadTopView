import cv2
import numpy as np
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from ai.yolo_segmentor import Yolo_segmentor
from config import USE_AI
from config import DEBUG

import math

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
        self.view.contour_button.setEnabled(False)

        self.ai = Yolo_segmentor()

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
            if USE_AI:
                contours = self.ai.get_contours(self.image)
            else:
                gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cv2.drawContours(self.image, contours, -1, (0, 255, 0), 2)
            self.draw_analysis_lines_points(self.image, contours)
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
        self.view.contour_button.setEnabled(True)
        self.view.contour_button.setText("Contours")
        self.view.flip_button.setEnabled(True)
        self.view.rotate_button.setEnabled(True)

    def draw_analysis_lines_points(self, image, contours):
        x,y,w,h = self.ai.get_roi_coord() # x,y,w,h
        min_distance = 10 # 교차점 그릴 때, 근처 중복 점 방지
        
        # ROI 중심점 계산
        center_x = x + w // 2
        center_y = y + h // 2

        # 분석을 위한 각도 (수직, 수평, 30도, -30도)
        angles = [0, 90, 60, -60]
        length = 800  # 선의 길이

        points_on_contour = []

        for angle in angles:
            rad = math.radians(angle)
            dx = int(math.cos(rad) * length)
            dy = int(math.sin(rad) * length)

            pt1 = (center_x - dx, center_y - dy)
            pt2 = (center_x + dx, center_y + dy)

            # 선 그리기
            cv2.line(image, pt1, pt2, (0, 0, 0), 3)

            if angle in [60, -60]:
                # 선과 contour의 교차점 찾기
                # pt1 = (x1, y1), pt2 = (x2, y2)
                A = pt2[1] - pt1[1]
                B = pt1[0] - pt2[0]
                C = pt2[0]*pt1[1] - pt1[0]*pt2[1]

                for cnt in contours:
                    for pt in cnt:
                        px, py = pt[0]
                        dist = abs(A * px + B * py + C) / math.sqrt(A*A + B*B)

                        if dist < 5:
                            # 선분 범위 체크
                            dot = (px - pt1[0]) * (pt2[0] - pt1[0]) + (py - pt1[1]) * (pt2[1] - pt1[1])
                            len_sq = (pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2
                            if 0 <= dot <= len_sq:
                                
                                # 이미 찍힌 점들과 충분히 멀리 있는지 확인
                                is_far_enough = True
                                for existing_pt in points_on_contour:
                                    if math.hypot(existing_pt[0] - px, existing_pt[1] - py) < min_distance:
                                        is_far_enough = False
                                        break
                                
                                if is_far_enough:
                                    cv2.circle(image, (px, py), 10, (0, 0, 255), -1)
                                    points_on_contour.append((px, py))
        if DEBUG:
            cv2.imwrite("analyzed_result.jpg", image)

        return image, points_on_contour