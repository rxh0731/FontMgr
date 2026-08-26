"""图片实验室的大图预览与人工清理画布。"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QPointF, QRect, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QWheelEvent
from PySide6.QtWidgets import QSizePolicy, QWidget

import numpy as np


VIEW_ORIGINAL = "original"
VIEW_CLEAN = "clean"
VIEW_LAYER = "layer"
VIEW_REVIEW = "review"


class ImageLabCanvas(QWidget):
    """只持有缩放预览，人工笔画用原图归一化坐标上报。"""

    stroke_finished = Signal(str, float, object)
    zoom_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("imageLabCanvas")
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)
        self._source = QImage()
        self._composite = QImage()
        self._alpha = QImage()
        self._uncertainty = QImage()
        self._layer_visual = QImage()
        self._review_overlay = QImage()
        self._view_mode = VIEW_CLEAN
        self._tool = "cover"
        self._brush_width = 80.0
        self._source_width = 1
        self._source_height = 1
        self._zoom = 1.0
        self._drawing = False
        self._current_points: list[QPointF] = []
        self.setMinimumSize(480, 360)

    @staticmethod
    def _rgb_image(pixels: np.ndarray) -> QImage:
        values = np.ascontiguousarray(pixels, dtype=np.uint8)
        height, width, channels = values.shape
        if channels != 3:
            raise ValueError("预览图片必须是 RGB 格式。")
        return QImage(
            values.data,
            width,
            height,
            int(values.strides[0]),
            QImage.Format.Format_RGB888,
        ).copy()

    @staticmethod
    def _gray_image(pixels: np.ndarray) -> QImage:
        values = np.ascontiguousarray(pixels, dtype=np.uint8)
        height, width = values.shape
        return QImage(
            values.data,
            width,
            height,
            int(values.strides[0]),
            QImage.Format.Format_Alpha8,
        ).copy()

    @property
    def has_image(self) -> bool:
        return not self._source.isNull()

    @property
    def zoom_percent(self) -> int:
        return int(round(self._zoom * 100.0))

    def set_preview(
        self,
        source: np.ndarray,
        composite: np.ndarray,
        cleanup_alpha: np.ndarray,
        uncertainty: np.ndarray,
        *,
        source_width: int,
        source_height: int,
    ) -> None:
        self._source = self._rgb_image(source)
        self._composite = self._rgb_image(composite)
        self._alpha = self._gray_image(cleanup_alpha)
        self._uncertainty = self._gray_image(uncertainty)
        self._layer_visual = self._masked_color_image(
            self._alpha,
            QColor(255, 255, 255, 255),
        )
        self._review_overlay = self._masked_color_image(
            self._uncertainty,
            QColor(220, 62, 55, 105),
        )
        self._source_width = max(1, int(source_width))
        self._source_height = max(1, int(source_height))
        self._update_canvas_size()
        self.update()

    def clear(self) -> None:
        self._source = QImage()
        self._composite = QImage()
        self._alpha = QImage()
        self._uncertainty = QImage()
        self._layer_visual = QImage()
        self._review_overlay = QImage()
        self._current_points.clear()
        self._drawing = False
        self.setFixedSize(480, 360)
        self.update()

    @staticmethod
    def _masked_color_image(mask: QImage, color: QColor) -> QImage:
        overlay = QImage(mask.size(), QImage.Format.Format_ARGB32_Premultiplied)
        overlay.fill(Qt.GlobalColor.transparent)
        painter = QPainter(overlay)
        painter.fillRect(overlay.rect(), color)
        painter.setCompositionMode(
            QPainter.CompositionMode.CompositionMode_DestinationIn
        )
        painter.drawImage(overlay.rect(), mask)
        painter.end()
        return overlay

    def set_view_mode(self, mode: str) -> None:
        if mode not in {VIEW_ORIGINAL, VIEW_CLEAN, VIEW_LAYER, VIEW_REVIEW}:
            raise ValueError("不支持的图片实验室预览模式。")
        self._view_mode = mode
        self.update()

    def set_tool(self, tool: str) -> None:
        if tool not in {"cover", "restore"}:
            raise ValueError("不支持的人工清理工具。")
        self._tool = tool

    def set_brush_width(self, width: float) -> None:
        self._brush_width = max(1.0, min(4096.0, float(width)))
        self.update()

    def set_zoom(self, zoom: float) -> None:
        target = max(0.08, min(8.0, float(zoom)))
        if abs(target - self._zoom) < 0.0001:
            return
        self._zoom = target
        self._update_canvas_size()
        self.zoom_changed.emit(self.zoom_percent)

    def fit_to_size(self, width: int, height: int) -> None:
        if self._source.isNull() or width <= 0 or height <= 0:
            return
        target = min(
            (width - 16) / self._source.width(),
            (height - 16) / self._source.height(),
        )
        self.set_zoom(max(0.08, min(1.0, target)))

    def _update_canvas_size(self) -> None:
        if self._source.isNull():
            return
        self.setFixedSize(
            max(1, int(round(self._source.width() * self._zoom))),
            max(1, int(round(self._source.height() * self._zoom))),
        )

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#25282d"))
        if self._source.isNull():
            painter.setPen(QColor("#aeb4bc"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "打开一张碑文拓片、手稿或文字扫描件",
            )
            return
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        target = self.rect()
        if self._view_mode == VIEW_ORIGINAL:
            painter.drawImage(target, self._source)
        elif self._view_mode == VIEW_LAYER:
            self._draw_checkerboard(painter, target)
            painter.drawImage(target, self._layer_visual)
        else:
            painter.drawImage(target, self._composite)
            if self._view_mode == VIEW_REVIEW:
                painter.drawImage(target, self._review_overlay)
        self._draw_active_stroke(painter)

    @staticmethod
    def _draw_checkerboard(painter: QPainter, target: QRect) -> None:
        size = 14
        light = QColor("#e8e8e8")
        dark = QColor("#c8c8c8")
        painter.fillRect(target, light)
        for y in range(0, target.height(), size):
            for x in range(0, target.width(), size):
                if (x // size + y // size) % 2:
                    painter.fillRect(x, y, size, size, dark)

    def _draw_active_stroke(self, painter: QPainter) -> None:
        if not self._current_points:
            return
        preview_scale = self._source.width() / max(1, self._source_width)
        pen_width = max(1.0, self._brush_width * preview_scale * self._zoom)
        color = QColor(255, 255, 255, 210) if self._tool == "cover" else QColor(28, 125, 163, 210)
        pen = QPen(color, pen_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        if len(self._current_points) == 1:
            painter.drawPoint(self._current_points[0])
        else:
            for first, second in zip(self._current_points, self._current_points[1:]):
                painter.drawLine(first, second)

    def _stroke_dirty_rect(self, first: QPointF, second: QPointF | None = None) -> QRect:
        preview_scale = self._source.width() / max(1, self._source_width)
        radius = max(3, int(self._brush_width * preview_scale * self._zoom / 2.0) + 3)
        other = second or first
        left = int(min(first.x(), other.x())) - radius
        top = int(min(first.y(), other.y())) - radius
        right = int(max(first.x(), other.x())) + radius
        bottom = int(max(first.y(), other.y())) + radius
        return QRect(QPoint(left, top), QPoint(right, bottom)).intersected(self.rect())

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if (
            event.button() == Qt.MouseButton.LeftButton
            and not self._source.isNull()
            and self._view_mode != VIEW_ORIGINAL
        ):
            self._drawing = True
            self._current_points = [QPointF(event.position())]
            self.grabMouse()
            self.update(self._stroke_dirty_rect(self._current_points[0]))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drawing:
            point = QPointF(
                max(0.0, min(float(self.width() - 1), event.position().x())),
                max(0.0, min(float(self.height() - 1), event.position().y())),
            )
            if not self._current_points or (
                QPointF(point) - self._current_points[-1]
            ).manhattanLength() >= 1.0:
                previous = self._current_points[-1]
                self._current_points.append(point)
                self.update(self._stroke_dirty_rect(previous, point))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drawing and event.button() == Qt.MouseButton.LeftButton:
            self._drawing = False
            self.releaseMouse()
            dirty_region = QRect()
            for point in self._current_points:
                dirty_region = dirty_region.united(self._stroke_dirty_rect(point))
            points = tuple(
                (
                    max(0.0, min(1.0, point.x() / max(1, self.width()))),
                    max(0.0, min(1.0, point.y() / max(1, self.height()))),
                )
                for point in self._current_points
            )
            self._current_points.clear()
            self.update(dirty_region)
            if points:
                self.stroke_finished.emit(self._tool, self._brush_width, points)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
            self.set_zoom(self._zoom * factor)
            event.accept()
            return
        event.ignore()
