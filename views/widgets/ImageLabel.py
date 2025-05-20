from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPixmap, QWheelEvent, QMouseEvent, QPainter

class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        # self.controller = None
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 2px dashed gray;")
        self.setText("이미지를 드래그하거나 Load 버튼을 사용하세요")

        self.original_pixmap = None
        self.scale_factor = 1.0

        self.dragging = False
        self.last_mouse_pos = QPoint()
        self.offset = QPoint()

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

    def set_image(self, pixmap: QPixmap):
        self.original_pixmap = pixmap
        self.scale_factor = 1.0
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self):
        if self.original_pixmap:
            scaled = self.original_pixmap.scaled(
                self.original_pixmap.size() * self.scale_factor,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            final = QPixmap(self.size())  # 현재 Label 크기에 맞춰 빈 캔버스
            final.fill(Qt.transparent)

            painter = QPainter(final)
            # 이미지를 중앙 + 오프셋으로 그림
            x = (self.width() - scaled.width()) // 2 + self.offset.x()
            y = (self.height() - scaled.height()) // 2 + self.offset.y()
            painter.drawPixmap(x, y, scaled)
            painter.end()

            self.setPixmap(final)

    def wheelEvent(self, event: QWheelEvent):
        # Ctrl 키를 누른 경우에만 확대/축소
        if event.modifiers() == Qt.ControlModifier:
            if event.angleDelta().y() > 0:
                self.scale_factor *= 1.1
            else:
                self.scale_factor /= 1.1

            # 제한
            self.scale_factor = max(0.1, min(self.scale_factor, 10.0))
            self._update_scaled_pixmap()
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
         # Ctrl 키를 누른 경우에만 확대/축소
        if event.modifiers() == Qt.ControlModifier:
            if event.button() == Qt.LeftButton and self.original_pixmap:
                self.dragging = True
                self.last_mouse_pos = event.pos()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        # Ctrl 키를 누른 경우에만 확대/축소
        if event.modifiers() == Qt.ControlModifier:
            if self.dragging and self.original_pixmap:
                delta = event.pos() - self.last_mouse_pos
                self.offset += delta
                self.last_mouse_pos = event.pos()
                self._update_scaled_pixmap()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.modifiers() == Qt.ControlModifier:
            if event.button() == Qt.LeftButton:
                self.dragging = False
        else:
            super().mouseReleaseEvent(event)