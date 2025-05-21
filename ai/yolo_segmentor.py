import cv2
import numpy as np
from ultralytics import YOLO
from config import DEBUG

class Yolo_segmentor:
    def __init__(self):
        self._model = YOLO(r"ai\best.pt")
        self._roi = None
        self._roi_coord = None # (x,y,w,h)

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

            self._roi = image[y:y+h, x:x+w]
            self._roi_coord = (x, y, w, h)

            mask_roi = mask_uint8[y:y+h, x:x+w]
            edges = cv2.Canny(mask_roi, 100, 200)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                cnt[:, 0, 0] += x
                cnt[:, 0, 1] += y
                all_contours.append(cnt)
            if DEBUG:
                 cv2.imwrite(f"roi_debug_{x}_{y}.jpg", self._roi)

            return all_contours

    def get_roi(self):
        return self._roi
    
    def get_roi_coord(self):
        return self._roi_coord