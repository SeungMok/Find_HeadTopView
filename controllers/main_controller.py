import cv2
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QTableWidgetItem
from PyQt5.QtGui import QPixmap, QImage, QColor, QBrush
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
        self.view.rotate_button.setEnabled(False)
        self.view.flip_button.setEnabled(False)
        self.view.contour_button.setEnabled(False)

        self.intersections = {1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None, 8: None}
        self.ci = None
        self.cvai = None

        self.ai = Yolo_segmentor()

        self.base_image_width = 1000
        self.base_image_height = 1000
        self.scale = 1.0

    def reset_table_UI(self):
        item = QTableWidgetItem("")
        item.setBackground(QBrush(QColor("white")))
        self.view.table.setItem(0, 0, item)
        item = QTableWidgetItem("")
        item.setBackground(QBrush(QColor("white")))
        self.view.table.setItem(1, 0, item)

        self.ci = None
        self.cvai = None
        self.intersections = {
            1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None, 8: None
        }

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self.view, "이미지 불러오기", "", "Image Files (*.png *.jpg *.bmp)")
        if path:
            self.load_image_from_path(path)

    def load_image_from_path(self, path):
        self.image = cv2.imread(path)
        self.original_image = self.image.copy()
        self.display_image(self.image)
        self.reset_table_UI()
        self.reset_buttons()

    def rotate_image(self):
        if self.image is not None:
            self.image = cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE)
            self.original_image = self.image.copy()
            self.display_image(self.image)

    def flip_image(self):
        if self.image is not None:
            self.image = cv2.flip(self.image, 1)
            self.original_image = self.image.copy()
            self.display_image(self.image)

    def toggle_contours(self):
        if self.image is None:
            return

        if not self.contour_mode:
            if USE_AI:
                contours = self.ai.get_contours(self.image)
            else:
                gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            self.set_scale()    # 선, 점, 글자 크기 조정정
            cv2.drawContours(self.image, contours, -1, (0, 255, 0), (int)(2*self.scale))
            self.draw_analysis_lines_points(self.image, contours)
            self.display_image(self.image)

            self.set_table_CI()
            self.set_table_CVAI()

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
            self.reset_table_UI()

    def display_image(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.view.image_label.width(), self.view.image_label.height(), Qt.KeepAspectRatio)
        self.view.image_label.set_image(pixmap)

    def reset_buttons(self):
        self.contour_mode = False
        self.view.contour_button.setEnabled(True)
        self.view.contour_button.setText("Contours")
        self.view.flip_button.setEnabled(True)
        self.view.rotate_button.setEnabled(True)

    def draw_analysis_lines_points(self, image, contours):
        x, y, w, h = self.ai.get_roi_coord()  # x,y,w,h
        min_distance = 10  # 교차점 그릴 때, 근처 중복 점 방지

        # ROI 중심점 계산
        center_x = x + w // 2
        center_y = y + h // 2

        # 분석을 위한 각도 (수평, 수직, 60도, -60도)
        angles = [0, 90, 60, -60]

        intersection_points = [] # 중복 방지 및 거리 계산을 위해 사용

        def is_on_segment(p, a, b):
            #점 p가 선분 a-b 위에 있는지 확인
            return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and
                    min(a[1], b[1]) <= p[1] <= max(a[1], b[1]) and
                    abs((b[1] - a[1]) * (p[0] - a[0]) - (b[0] - a[0]) * (p[1] - a[1])) < 1e-6)

        def find_intersection(contour, p1, p2):
            #윤곽선과 선분의 교차점 찾기
            intersection_pts = []
            for i in range(len(contour)):
                c_p1 = contour[i][0]
                c_p2 = contour[(i + 1) % len(contour)][0]  # 닫힌 윤곽선 처리

                # 선분 교차 판정
                def on_segment(p, a, b):
                    return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and
                            min(a[1], b[1]) <= p[1] <= max(a[1], b[1]) and
                            abs((b[1] - a[1]) * (p[0] - a[0]) - (b[0] - a[0]) * (p[1] - a[1])) < 1e-6)

                def orientation(p, q, r):
                    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
                    if abs(val) < 1e-6: return 0  # Collinear
                    return 1 if val > 0 else 2  # Clockwise or Counterclockwise

                o1 = orientation(p1, p2, c_p1)
                o2 = orientation(p1, p2, c_p2)
                o3 = orientation(c_p1, c_p2, p1)
                o4 = orientation(c_p1, c_p2, p2)

                if o1 != o2 and o3 != o4:
                    # 교점 계산
                    det = (p1[0] - p2[0]) * (c_p1[1] - c_p2[1]) - (p1[1] - p2[1]) * (c_p1[0] - c_p2[0])
                    if abs(det) > 1e-6:
                        t = ((p1[0] - c_p1[0]) * (c_p1[1] - c_p2[1]) - (p1[1] - c_p1[1]) * (c_p1[0] - c_p2[0])) / det
                        u = -((p1[0] - p2[0]) * (p1[1] - c_p1[1]) - (p1[1] - p2[1]) * (p1[0] - c_p1[0])) / det
                        if 0 <= t <= 1 and 0 <= u <= 1:
                            intersection_x = p1[0] + t * (p2[0] - p1[0])
                            intersection_y = p1[1] + t * (p2[1] - p1[1])
                            intersection_pts.append((int(intersection_x), int(intersection_y)))

            return intersection_pts

        for angle in angles:
            rad = math.radians(angle)
            dx = int(math.cos(rad) * w)
            dy = int(math.sin(rad) * h)

            pt1 = (center_x - dx, center_y - dy)
            pt2 = (center_x + dx, center_y + dy)

            # 선 그리기
            cv2.line(image, pt1, pt2, (0, 0, 0), (int)(3*self.scale))

            found_intersections = []
            for cnt in contours:
                found_intersections.extend(find_intersection(cnt, pt1, pt2))

            # 각도에 따라 교차점 분류 및 저장
            if angle == 0:  # 수평선
                left_point = None
                right_point = None
                for pt in found_intersections:
                    if left_point is None or pt[0] < left_point[0]:
                        left_point = pt
                    if right_point is None or pt[0] > right_point[0]:
                        right_point = pt
                if left_point:
                    self.intersections[1] = left_point
                if right_point:
                    self.intersections[2] = right_point
            elif angle == 90:  # 수직선
                top_point = None
                bottom_point = None
                for pt in found_intersections:
                    if top_point is None or pt[1] < top_point[1]:
                        top_point = pt
                    if bottom_point is None or pt[1] > bottom_point[1]:
                        bottom_point = pt
                if top_point:
                    self.intersections[3] = top_point
                if bottom_point:
                    self.intersections[4] = bottom_point
            elif angle == 60:
                top_point = None
                bottom_point = None
                for pt in found_intersections:
                    if top_point is None or pt[1] < top_point[1] - (pt[0] - center_x) * math.tan(math.radians(60)):
                        top_point = pt
                    if bottom_point is None or pt[1] > bottom_point[1] - (pt[0] - center_x) * math.tan(math.radians(60)):
                        bottom_point = pt
                if top_point:
                    self.intersections[5] = top_point
                if bottom_point:
                    self.intersections[7] = bottom_point
            elif angle == -60:
                top_point = None
                bottom_point = None
                for pt in found_intersections:
                    if top_point is None or pt[1] < top_point[1] - (pt[0] - center_x) * math.tan(math.radians(-60)):
                        top_point = pt
                    if bottom_point is None or pt[1] > bottom_point[1] - (pt[0] - center_x) * math.tan(math.radians(-60)):
                        bottom_point = pt
                if top_point:
                    self.intersections[6] = top_point
                if bottom_point:
                    self.intersections[8] = bottom_point

        # 교차점 그리기 및 번호 표시
        for number, point in self.intersections.items():
            if point:
                cv2.circle(image, point, (int)(10*self.scale), (0, 0, 255), -1)
                cv2.putText(image, str(number), (point[0] + 15, point[1] + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, (float)(2*self.scale), (0, 255, 255), (int)(2*self.scale))

        if DEBUG:
            cv2.imwrite("analyzed_result.jpg", image)

    def calculate_CVAI(self):
        # 교차점 좌표를 가져와서 계산
        points = [self.intersections[i] for i in range(5, 9)]
        if None in points:
            print("모든 교차점이 감지되지 않았습니다.")
            return None

        # 교차점 간 거리 계산
        length_5_7 = math.sqrt((points[0][0] - points[2][0]) ** 2 + (points[0][1] - points[2][1]) ** 2)
        length_6_8 = math.sqrt((points[1][0] - points[3][0]) ** 2 + (points[1][1] - points[3][1]) ** 2)        

        # CVAI 계산
        bigger = length_5_7 if length_5_7 > length_6_8 else length_6_8
        self.cvai = abs(length_5_7 - length_6_8) / bigger * 100
    
    def calculate_CI(self):
        # 교차점 좌표를 가져와서 계산
        points = [self.intersections[i] for i in range(1, 5)]
        if None in points:
            print("모든 교차점이 감지되지 않았습니다.")
            return None
        
        # 교차점 간 거리 계산
        length_1_2 = math.sqrt((points[0][0] - points[1][0]) ** 2 + (points[0][1] - points[1][1]) ** 2)
        length_3_4 = math.sqrt((points[2][0] - points[3][0]) ** 2 + (points[2][1] - points[3][1]) ** 2)

        # ML 계산
        self.ci = length_1_2 / length_3_4 * 100
    
    def set_table_CI(self):
        if self.ci is None:
            self.calculate_CI()
        item = QTableWidgetItem(f"{self.ci:.2f}")

        if self.ci < 85.0:
            item.setBackground(QBrush(QColor("green")))
        elif self.ci > 95.0:
            item.setBackground(QBrush(QColor("red")))
        else:
            item.setBackground(QBrush(QColor("yellow")))
        self.view.table.setItem(0, 0, item)

    def set_table_CVAI(self):
        if self.cvai is None:
            self.calculate_CVAI()
        item = QTableWidgetItem(f"{self.cvai:.2f}")

        if self.cvai < 3.5:
            item.setBackground(QBrush(QColor("green")))
        elif self.cvai > 8.75:
            item.setBackground(QBrush(QColor("red")))
        else:
            item.setBackground(QBrush(QColor("yellow")))
        self.view.table.setItem(1, 0, item)

    def set_scale(self):
        _h, _w = self.original_image.shape[:2]

        width_scale = _w / self.base_image_width
        height_scale = _h / self.base_image_height
        self.scale = (width_scale + height_scale) / 2