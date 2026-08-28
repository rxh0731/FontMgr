"""图片实验室的大图预览与人工清理画布。"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QCursor,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QWheelEvent,
)
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
    pan_requested = Signal(QPoint)

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
        self._detail_source = QImage()
        self._detail_composite = QImage()
        self._detail_layer_visual = QImage()
        self._detail_review_overlay = QImage()
        self._detail_source_rect: tuple[int, int, int, int] | None = None
        self._view_mode = VIEW_CLEAN
        self._tool = "cover"
        self._brush_width = 80.0
        self._source_width = 1
        self._source_height = 1
        self._zoom = 1.0
        self._drawing = False
        self._current_points: list[QPointF] = []
        self._space_pan_held = False
        self._panning = False
        self._last_pan_global_position = QPointF()
        self._cursor_before_pan: QCursor | None = None
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
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

    @property
    def zoom_factor(self) -> float:
        return self._zoom

    @property
    def has_reduced_preview(self) -> bool:
        return (
            not self._source.isNull()
            and (
                self._source.width() < self._source_width
                or self._source.height() < self._source_height
            )
        )

    @property
    def detail_source_rect(self) -> tuple[int, int, int, int] | None:
        return self._detail_source_rect

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
        self.clear_detail_preview(update_canvas=False)
        self._update_canvas_size()
        self.update()

    def set_detail_preview(
        self,
        source: np.ndarray,
        composite: np.ndarray,
        cleanup_alpha: np.ndarray,
        uncertainty: np.ndarray,
        source_rect: tuple[int, int, int, int],
    ) -> None:
        """设置覆盖在快速预览上的当前原图高清区域。"""

        if len(source_rect) != 4:
            raise ValueError("高清预览区域无效。")
        left, top, right, bottom = (int(value) for value in source_rect)
        if not (0 <= left < right <= self._source_width):
            raise ValueError("高清预览横向区域超出原图。")
        if not (0 <= top < bottom <= self._source_height):
            raise ValueError("高清预览纵向区域超出原图。")
        if source.shape[:2] != composite.shape[:2]:
            raise ValueError("高清原稿与清理效果尺寸不一致。")
        if source.shape[:2] != cleanup_alpha.shape[:2]:
            raise ValueError("高清原稿与清理层尺寸不一致。")
        if source.shape[:2] != uncertainty.shape[:2]:
            raise ValueError("高清原稿与待核对区域尺寸不一致。")
        self._detail_source = self._rgb_image(source)
        self._detail_composite = self._rgb_image(composite)
        detail_alpha = self._gray_image(cleanup_alpha)
        detail_uncertainty = self._gray_image(uncertainty)
        self._detail_layer_visual = self._masked_color_image(
            detail_alpha,
            QColor(255, 255, 255, 255),
        )
        self._detail_review_overlay = self._masked_color_image(
            detail_uncertainty,
            QColor(220, 62, 55, 105),
        )
        self._detail_source_rect = (left, top, right, bottom)
        self.update(self._canvas_rect_for_source_rect(self._detail_source_rect).toRect())

    def clear_detail_preview(self, *, update_canvas: bool = True) -> None:
        previous = self._detail_source_rect
        self._detail_source = QImage()
        self._detail_composite = QImage()
        self._detail_layer_visual = QImage()
        self._detail_review_overlay = QImage()
        self._detail_source_rect = None
        if update_canvas and previous is not None:
            self.update(self._canvas_rect_for_source_rect(previous).toRect())

    def source_rect_for_canvas_rect(self, rect: QRect) -> tuple[int, int, int, int]:
        """把画布可见区域换算为原图像素区域。"""

        bounded = rect.intersected(self.rect())
        if bounded.isEmpty():
            return (0, 0, 0, 0)
        left = max(0, int(np.floor(bounded.left() * self._source_width / self.width())))
        top = max(0, int(np.floor(bounded.top() * self._source_height / self.height())))
        right = min(
            self._source_width,
            int(np.ceil((bounded.right() + 1) * self._source_width / self.width())),
        )
        bottom = min(
            self._source_height,
            int(np.ceil((bounded.bottom() + 1) * self._source_height / self.height())),
        )
        return (left, top, max(left + 1, right), max(top + 1, bottom))

    def detail_covers(
        self,
        source_rect: tuple[int, int, int, int],
        minimum_scale: float,
    ) -> bool:
        current = self._detail_source_rect
        if current is None or self._detail_source.isNull():
            return False
        if not (
            current[0] <= source_rect[0]
            and current[1] <= source_rect[1]
            and current[2] >= source_rect[2]
            and current[3] >= source_rect[3]
        ):
            return False
        detail_scale = min(
            self._detail_source.width() / max(1, current[2] - current[0]),
            self._detail_source.height() / max(1, current[3] - current[1]),
        )
        return detail_scale >= max(0.0, float(minimum_scale))

    def _canvas_rect_for_source_rect(
        self,
        source_rect: tuple[int, int, int, int],
    ) -> QRectF:
        left, top, right, bottom = source_rect
        return QRectF(
            left * self.width() / self._source_width,
            top * self.height() / self._source_height,
            (right - left) * self.width() / self._source_width,
            (bottom - top) * self.height() / self._source_height,
        )

    def clear(self) -> None:
        self.cancel_pan()
        self._source = QImage()
        self._composite = QImage()
        self._alpha = QImage()
        self._uncertainty = QImage()
        self._layer_visual = QImage()
        self._review_overlay = QImage()
        self.clear_detail_preview(update_canvas=False)
        self._current_points.clear()
        self._drawing = False
        self.setFixedSize(480, 360)
        self.update()

    @property
    def space_pan_active(self) -> bool:
        return self._space_pan_held or self._panning

    def set_pan_modifier_active(self, active: bool) -> bool:
        """临时启用空格抓手，不改变当前人工清理工具。"""

        if active:
            if self._source.isNull() or self._drawing:
                return False
            if self._space_pan_held:
                return True
            self._space_pan_held = True
            if self._cursor_before_pan is None:
                self._cursor_before_pan = QCursor(self.cursor())
            if not self._panning:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            return True

        if not self.space_pan_active:
            return False
        self._space_pan_held = False
        if not self._panning:
            self._restore_cursor_after_pan()
        return True

    def cancel_pan(self) -> None:
        """页面隐藏或应用失焦时强制结束临时平移。"""

        was_active = self.space_pan_active or self._cursor_before_pan is not None
        if self._panning:
            self.releaseMouse()
        self._space_pan_held = False
        self._panning = False
        self._last_pan_global_position = QPointF()
        if was_active:
            self._restore_cursor_after_pan()

    def _restore_cursor_after_pan(self) -> None:
        previous = self._cursor_before_pan
        self._cursor_before_pan = None
        if previous is None:
            self.unsetCursor()
        else:
            self.setCursor(previous)

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

    def zoom_by_wheel_delta(self, delta: int) -> bool:
        """按滚轮方向缩放，供画布和外层滚动视口共用。"""

        if delta == 0 or self._source.isNull():
            return False
        factor = 1.15 if delta > 0 else 1.0 / 1.15
        self.set_zoom(self._zoom * factor)
        return True

    def zoom_in(self) -> bool:
        """按工具栏步长放大画布。"""

        return self.zoom_by_wheel_delta(120)

    def zoom_out(self) -> bool:
        """按工具栏步长缩小画布。"""

        return self.zoom_by_wheel_delta(-120)

    @staticmethod
    def wheel_delta(event: QWheelEvent) -> int:
        """兼容标准滚轮、高精度滚轮及驱动转换后的横向增量。"""

        angle = event.angleDelta()
        delta = angle.y() or angle.x()
        if delta == 0:
            pixel = event.pixelDelta()
            delta = pixel.y() or pixel.x()
        return delta

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
        self._draw_detail_preview(painter)
        self._draw_active_stroke(painter)

    def _draw_detail_preview(self, painter: QPainter) -> None:
        if self._detail_source_rect is None or self._detail_source.isNull():
            return
        target = self._canvas_rect_for_source_rect(self._detail_source_rect)
        if self._view_mode == VIEW_ORIGINAL:
            painter.drawImage(target, self._detail_source)
        elif self._view_mode == VIEW_LAYER:
            painter.drawImage(target, self._detail_layer_visual)
        else:
            painter.drawImage(target, self._detail_composite)
            if self._view_mode == VIEW_REVIEW:
                painter.drawImage(target, self._detail_review_overlay)

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
        if event.button() == Qt.MouseButton.LeftButton and self._space_pan_held:
            self._panning = True
            self._last_pan_global_position = QPointF(event.globalPosition())
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.grabMouse()
            event.accept()
            return
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
        if self._panning:
            current = QPointF(event.globalPosition())
            delta = (current - self._last_pan_global_position).toPoint()
            if not delta.isNull():
                self._last_pan_global_position += QPointF(delta)
                self.pan_requested.emit(delta)
            event.accept()
            return
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
        if self._panning and event.button() == Qt.MouseButton.LeftButton:
            current = QPointF(event.globalPosition())
            delta = (current - self._last_pan_global_position).toPoint()
            if not delta.isNull():
                self.pan_requested.emit(delta)
            self._panning = False
            self._last_pan_global_position = QPointF()
            self.releaseMouse()
            if self._space_pan_held:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self._restore_cursor_after_pan()
            event.accept()
            return
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
        if event.modifiers() & Qt.KeyboardModifier.AltModifier:
            delta = self.wheel_delta(event)
            if self.zoom_by_wheel_delta(delta):
                event.accept()
                return
        event.ignore()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Space:
            handled = (
                self.space_pan_active
                if event.isAutoRepeat()
                else self.set_pan_modifier_active(True)
            )
            if handled:
                event.accept()
                return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Space:
            handled = (
                self.space_pan_active
                if event.isAutoRepeat()
                else self.set_pan_modifier_active(False)
            )
            if handled:
                event.accept()
                return
        super().keyReleaseEvent(event)
