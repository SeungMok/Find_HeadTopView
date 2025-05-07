import cv2
import numpy as np
from ultralytics import YOLO

class Yolo_segmentor:
    def __init__(self):
        self._model = YOLO("ai/YOLODataset/runs/segment/train/weights/best.pt")

    def get_contours(self,image):
        results = self._model(image)
        mask = results[0].masks.data[0].cpu().numpy()  # 첫 번째 객체의 마스크 사용

        # 마스크 → ROI
        mask_uint8 = (mask * 255).astype(np.uint8)
        x, y, w, h = cv2.boundingRect(mask_uint8)
        roi = image[y:y+h, x:x+w]

        # ROI → Canny
        edges = cv2.Canny(roi, 100, 200)

        # ROI → Contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 좌표 보정 (ROI → 원본 이미지 위치로)
        for cnt in contours:
            cnt[:, 0, 0] += x
            cnt[:, 0, 1] += y

        return contours