import cv2
import numpy as np
from ultralytics import YOLO
from config import DEBUG

import math

class Yolo_segmentor:
    def __init__(self):
        self._model = YOLO("ai/YOLODataset/runs/segment/train/weights/best.pt")

    def get_contours(self, image):
        results = self._model(image)
        all_contours = []

        if results[0].masks is None:
            return []

        masks = results[0].masks.data.cpu().numpy()  # shape: (N, h, w) - YOLO 내부 해상도
        mask_shape = masks[0].shape
        original_shape = image.shape[:2]  # (H, W)
        padding = 20

        for mask in masks:
            # 마스크를 원본 이미지 크기로 리사이즈
            mask_resized = cv2.resize(mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_uint8 = (mask_resized * 255).astype(np.uint8)

            if cv2.countNonZero(mask_uint8) == 0:
                continue

            x, y, w, h = cv2.boundingRect(mask_uint8)
            x = max(x - padding, 0)
            y = max(y - padding, 0)
            w = min(w + 2 * padding, original_shape[1] - x)
            h = min(h + 2 * padding, original_shape[0] - y)

            roi = image[y:y+h, x:x+w]
            mask_roi = mask_uint8[y:y+h, x:x+w]

            edges = cv2.Canny(mask_roi, 100, 200)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                cnt[:, 0, 0] += x
                cnt[:, 0, 1] += y
                all_contours.append(cnt)
            if DEBUG:
                 cv2.imwrite(f"roi_debug_{x}_{y}.jpg", roi)

            # 사두 분석선 및 교차점 그리기
            analyzed_img, _ = self.draw_analysis_lines_and_points(image.copy(), all_contours, x, y, w, h)
            cv2.imwrite("analyzed_result.jpg", analyzed_img)

            return all_contours
        # return self.delete_noise(image,all_contours)
    
    # 노이즈 제거
    # 1. contours(애기머리)의 위치가 중앙에 위치한다.
    # 2. 그 중 가장 큰 외곽선만 사용한다. (사두증 검사는 아이머리 하나만 있기 때문)
    # def delete_noise(self, image, contours):
    #     h, w = image.shape[:2]
    #     center_x = w // 2
    #     center_y = h // 2  # 이미지 상단 절반
    #     tolerance = 100

    #     valid_contours = []

    #     for cnt in contours:
    #         M = cv2.moments(cnt)
    #         if M["m00"] == 0:
    #             continue  # 면적이 0이면 중심 계산 불가 → 무시

    #         cx = int(M["m10"] / M["m00"])  # contour의 x 중심
    #         cy = int(M["m01"] / M["m00"])  # contour의 y 중심

    #         if abs(cx - center_x) <= tolerance and abs(cy - center_y) <= tolerance:
    #             valid_contours.append(cnt)

    #     if valid_contours:
    #         # 중심 필터링된 것들 중 가장 큰 contour 하나만 반환
    #         largest_contour = max(valid_contours, key=cv2.contourArea)
    #         return [largest_contour]
    #     else:
    #         return []  # 유효한 contour가 없으면 빈 리스트 반환

    def draw_analysis_lines_and_points(self, image, contours, x, y, w, h):
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
            cv2.line(image, pt1, pt2, (0, 255, 0), 1)

            # 선과 contour의 교차점 찾기
            for cnt in contours:
                for pt in cnt:
                    px, py = pt[0]
                    # 직선의 거리로 근사: |Ax + By + C| < ε
                    dist = abs((pt2[1] - pt1[1]) * px - (pt2[0] - pt1[0]) * py +
                            pt2[0]*pt1[1] - pt2[1]*pt1[0]) / (
                            ((pt2[1] - pt1[1])**2 + (pt2[0] - pt1[0])**2)**0.5)
                    if dist < 2:  # 근사 임계값
                        cv2.circle(image, (px, py), 3, (0, 0, 255), -1)
                        points_on_contour.append((px, py))

        return image, points_on_contour

    def delete_noise(self, image, contours):
        h, w = image.shape[:2]
        center_x = w // 2
        center_y = h // 2  # 이미지 상단 절반
        tolerance = 100

        valid_contours = []

        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue  # 면적이 0이면 중심 계산 불가 → 무시

            cx = int(M["m10"] / M["m00"])  # contour의 x 중심
            cy = int(M["m01"] / M["m00"])  # contour의 y 중심

            if abs(cx - center_x) <= tolerance and abs(cy - center_y) <= tolerance:
                valid_contours.append(cnt)
        
        return valid_contours
        
    def get_roi(self, image):
        results = self._model(image)
        mask_uint8 = results[0].masks.data[0].cpu().numpy()
        x, y, w, h = cv2.boundingRect(mask_uint8)
        roi = image[y:y+h, x:x+w]

        return roi