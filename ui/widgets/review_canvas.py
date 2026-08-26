"""手工审核图像画布。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
import math

import cv2
import numpy as np
from core.transform_renderer import (
    TransformLimits,
    calculate_transform_geometry,
    compose_rgba_on_canvas,
    place_transform,
    quad_is_valid,
    render_transformed_rgba,
)
from PySide6.QtCore import QEvent, QPoint, QPointF, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QCursor,
    QImage,
    QInputDevice,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QPointingDevice,
    QPolygonF,
    QTabletEvent,
    QTransform,
    QWheelEvent,
)
from PySide6.QtWidgets import QWidget


@dataclass(frozen=True)
class _TransformState:
    """字形相对原始画布中心的几何变换。"""

    x: float = 0.0
    y: float = 0.0
    scale: float = 1.0
    rotation: float = 0.0
    stretch_w: float = 1.0
    stretch_h: float = 1.0
    distort: tuple[float, ...] = (0.0,) * 8


@dataclass(frozen=True)
class _TransformView:
    """自由变换控制层所在视图的统一坐标映射。"""

    origin: QPointF
    scale: float


@dataclass(frozen=True)
class _TransformGeometry:
    """缩放、透视和旋转共用的几何计算结果。"""

    source_rect: QRect
    scaled_size: QSize
    perspective_size: QSize
    output_size: QSize
    perspective_matrix: np.ndarray
    rotation_matrix: np.ndarray
    origin: QPointF
    polygon: tuple[QPointF, QPointF, QPointF, QPointF]


@dataclass
class _TransformedContent:
    """可直接绘制的变换结果及其逻辑位置。"""

    image: QImage
    origin: QPointF
    polygon: tuple[QPointF, QPointF, QPointF, QPointF]
    alpha_bounds: QRectF


@dataclass
class _CanvasState:
    """一个可撤销的像素与几何状态。"""

    image: QImage
    transform: _TransformState
    image_origin: QPointF


class ReviewCanvas(QWidget):
    """支持自由变换、像素修订、视图控制与撤销重做的 RGBA 画布。"""

    changed = Signal(bool)
    pixels_changed = Signal()
    zoom_changed = Signal(int)
    transform_changed = Signal(dict)
    transform_interaction_started = Signal()
    transform_interaction_finished = Signal(bool)
    ink_color_changed = Signal(QColor)
    brush_size_changed = Signal(int)
    history_changed = Signal(bool, bool)

    TOOL_PAN = "pan"
    TOOL_TRANSFORM = "transform"
    TOOL_BRUSH = "brush"
    TOOL_ERASER = "eraser"

    BACKGROUND_WHITE = "white"
    BACKGROUND_CHECKERBOARD = "checkerboard"

    # 田字格外保留不可见的 130% 越界编辑范围，用于补画但不再绘制外框。
    WORKSPACE_RATIO = 1.30

    _MIN_SCALE = 0.05
    _MAX_SCALE = 5.0
    _MAX_TRANSFORM_DIMENSION = 16_384
    _MAX_TRANSFORM_PIXELS = 64 * 1024 * 1024
    _MAX_DISTORT_OFFSET = 8_192.0
    _HANDLE_RADIUS = 5.0
    _ROTATE_HANDLE_DISTANCE = 24.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        self.setMouseTracking(True)
        self.setTabletTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._image = QImage()
        self._source_image = QImage()
        self._source_image_origin = QPointF()
        self._source_preview_visible = False
        self._reference_image = QImage()
        self._reference_visible = False
        self._reference_opacity = 0.35
        self._canvas_size = QSize()
        self._image_origin = QPointF()
        self._transform = _TransformState()
        self._saved_state = _CanvasState(QImage(), _TransformState(), QPointF())
        self._undo_stack: list[_CanvasState] = []
        self._redo_stack: list[_CanvasState] = []
        self._rendered_image = QImage()
        self._rendered_origin = QPoint()
        self._transformed_content_cache: _TransformedContent | None = None
        self._render_postprocessor: Callable[[np.ndarray], np.ndarray] | None = None
        self._content_bounds = QRectF()

        self._tool = self.TOOL_PAN
        self._brush_size = 10
        self._brush_color = QColor(0, 0, 0, 255)
        self._brush_ink_coverage = 255
        self._pressure_enabled = True
        self._minimum_pressure_ratio = 0.2
        self._zoom = 1.0
        self._offset = QPointF()
        self._view_mode = "fit"
        self._grid_visible = True
        self._background_mode = self.BACKGROUND_WHITE

        self._drag_origin = QPointF()
        self._last_image_point = QPoint()
        self._drawing = False
        self._stroke_tool = self.TOOL_BRUSH
        self._stroke_brush_color = QColor(self._brush_color)
        self._stroke_ink_coverage = self._brush_ink_coverage
        self._last_stroke_width = float(self._brush_size)
        self._stroke_before_state: _CanvasState | None = None
        self._stroke_changed = False
        self._panning = False
        self._space_pan_held = False
        self._space_pan_blocked = False
        self._space_pan_drag = False
        self._tablet_panning = False
        self._tablet_active = False
        self._suppressed_tablet_device_key: tuple[int, int, int] | None = None
        self._suppressed_mouse_sequence_active = False
        self._transform_drag_kind = ""
        self._transform_drag_start = QPointF()
        self._transform_drag_state = _TransformState()
        self._transform_drag_center = QPointF()
        self._transform_drag_polygon: tuple[QPointF, QPointF, QPointF, QPointF] = (
            QPointF(),
            QPointF(),
            QPointF(),
            QPointF(),
        )
        self._transform_drag_anchor_name = ""
        self._transform_drag_anchor = QPointF()
        self._transform_drag_modifiers = Qt.KeyboardModifier.NoModifier
        self._transform_drag_started = False
        self._transform_drag_view: _TransformView | None = None
        self._hover_position: QPointF | None = None
        self._pointer_position: QPointF | None = None
        self._pointer_stroke_width = float(self._brush_size)
        self._pointer_tool = self.TOOL_BRUSH
        self._rotation_cursor: QCursor | None = None
        self._brush_wheel_remainder = 0
        self._dirty = False

    @property
    def has_image(self) -> bool:
        return not self._image.isNull()

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    @property
    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    @property
    def source_preview_visible(self) -> bool:
        return self._source_preview_visible

    @property
    def reference_visible(self) -> bool:
        return self._reference_visible

    @property
    def reference_opacity(self) -> float:
        return self._reference_opacity

    @property
    def grid_visible(self) -> bool:
        return self._grid_visible

    @property
    def background_mode(self) -> str:
        return self._background_mode

    @property
    def tool(self) -> str:
        return self._tool

    @property
    def space_pan_active(self) -> bool:
        """返回空格临时抓手是否正在按住、拖动或等待释放。"""
        return self._space_pan_held or self._space_pan_blocked or self._space_pan_drag

    @property
    def pressure_enabled(self) -> bool:
        return self._pressure_enabled

    @property
    def minimum_pressure_ratio(self) -> float:
        return self._minimum_pressure_ratio

    @property
    def brush_size(self) -> int:
        return self._brush_size

    @property
    def zoom_percent(self) -> int:
        return round(self._zoom * 100)

    def image(self) -> QImage:
        """返回已烘焙自由变换的最终图像，越界部分以透明边对称扩展。"""
        if self._image.isNull():
            return QImage()
        return self._final_image().copy()

    def canvas_size(self) -> QSize:
        """返回田字格对应的逻辑画布尺寸。"""
        return QSize(self._canvas_size)

    def output_origin(self) -> QPoint:
        """返回逻辑田字格左上角在 ``image()`` 输出中的坐标。"""
        if self.has_image:
            self._final_image()
        return QPoint(self._rendered_origin)

    def set_image(
        self,
        image: QImage,
        canvas_size: QSize | tuple[int, int] | None = None,
        source_preview: QImage | None = None,
    ) -> None:
        """载入图像；扩展人工稿可另行传入原始田字格尺寸。"""
        if image.isNull():
            self.clear_image()
            return
        self._cancel_active_input(clear_tablet_suppression=True)
        self._image = image.convertToFormat(QImage.Format.Format_ARGB32)
        self._canvas_size = self._normalized_canvas_size(canvas_size, self._image.size())
        self._image_origin = QPointF(
            (self._canvas_size.width() - self._image.width()) / 2.0,
            (self._canvas_size.height() - self._image.height()) / 2.0,
        )
        if source_preview is not None and not source_preview.isNull():
            self._source_image = source_preview.convertToFormat(
                QImage.Format.Format_ARGB32
            )
            self._source_image_origin = QPointF(
                (self._canvas_size.width() - self._source_image.width()) / 2.0,
                (self._canvas_size.height() - self._source_image.height()) / 2.0,
            )
        else:
            self._source_image = self._image.copy()
            self._source_image_origin = QPointF(self._image_origin)
        self._source_preview_visible = False
        self._transform = _TransformState()
        self._saved_state = self._snapshot()
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._emit_history()
        self._invalidate_render()
        self._refresh_content_bounds()
        self.sample_ink_color()
        self._set_dirty(False)
        self._emit_transform()
        self.fit_to_view()

    def set_render_postprocessor(
        self,
        processor: Callable[[np.ndarray], np.ndarray] | None,
    ) -> None:
        """设置几何重采样后的 RGBA 显示处理器，不改变编辑源图和变换参数。"""
        if processor is self._render_postprocessor:
            return
        self._render_postprocessor = processor
        self._invalidate_render()
        self.update()

    def clear_image(self) -> None:
        self._cancel_active_input(clear_tablet_suppression=True)
        self._clear_pointer_preview()
        self._image = QImage()
        self._source_image = QImage()
        self._source_image_origin = QPointF()
        self._source_preview_visible = False
        self._canvas_size = QSize()
        self._image_origin = QPointF()
        self._transform = _TransformState()
        self._saved_state = _CanvasState(QImage(), _TransformState(), QPointF())
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._emit_history()
        self._rendered_image = QImage()
        self._rendered_origin = QPoint()
        self._transformed_content_cache = None
        self._content_bounds = QRectF()
        self._offset = QPointF()
        self._set_dirty(False)
        self._emit_transform()
        self._update_cursor_at(None)
        self.update()

    def set_tool(self, tool: str) -> None:
        """切换工具；像素编辑、自由变换和平移互斥。"""
        if tool not in {
            self.TOOL_PAN,
            self.TOOL_TRANSFORM,
            self.TOOL_BRUSH,
            self.TOOL_ERASER,
        }:
            return
        self._cancel_active_input(clear_tablet_suppression=True)
        if (
            tool in {self.TOOL_BRUSH, self.TOOL_ERASER}
            and self.has_image
            and self._transform != _TransformState()
        ):
            self._bake_transform(push_undo=True)
        self._tool = tool
        self._brush_wheel_remainder = 0
        self._pointer_tool = tool
        self._pointer_stroke_width = float(self._brush_size)
        self._update_cursor_at(self._pointer_position)
        self.update()

    def set_brush_size(self, size: int) -> None:
        normalized = max(1, min(100, int(size)))
        if normalized == self._brush_size:
            return
        old_rect = self._pointer_preview_rect()
        self._brush_size = normalized
        self._pointer_stroke_width = float(normalized)
        self.brush_size_changed.emit(normalized)
        self._update_pointer_region(old_rect)

    def adjust_brush_size(self, direction: int) -> None:
        """按当前分段步长增大或缩小像素工具笔触。"""
        self._change_brush_size(direction)

    def set_pressure_enabled(self, enabled: bool) -> None:
        """启用或关闭绘图板压力到笔触宽度的映射。"""
        self._pressure_enabled = bool(enabled)

    def set_minimum_pressure_ratio(self, ratio: float) -> None:
        """设置最轻压力对应基础笔触宽度的比例。"""
        try:
            value = float(ratio)
        except (TypeError, ValueError):
            return
        if not math.isfinite(value):
            return
        self._minimum_pressure_ratio = max(0.05, min(1.0, value))

    def brush_color(self) -> QColor:
        return QColor(self._brush_color)

    def set_brush_color(self, color: QColor) -> None:
        """显式指定画笔颜色，透明度固定为完全不透明。"""
        if not color.isValid():
            return
        sampled = QColor(color)
        sampled.setAlpha(255)
        if sampled == self._brush_color:
            return
        self._brush_color = sampled
        self.ink_color_changed.emit(QColor(self._brush_color))

    def sample_ink_color(self) -> QColor:
        """从当前非透明笔画中自动取样出现次数最多的主墨色。"""
        sampled = self._dominant_ink_color(self._image)
        self._brush_color = sampled
        self._brush_ink_coverage = self._dominant_ink_coverage(self._image)
        self.ink_color_changed.emit(QColor(sampled))
        return QColor(sampled)

    def transform(self) -> dict[str, object]:
        """返回变换参数；X/Y 为原始画布坐标系中的像素偏移。"""
        return {
            "x": self._transform.x,
            "y": self._transform.y,
            "scale": self._transform.scale,
            "rotation": self._transform.rotation,
            "stretch_w": self._transform.stretch_w,
            "stretch_h": self._transform.stretch_h,
            "distort": list(self._transform.distort),
        }

    def set_transform(
        self,
        x: float | None = None,
        y: float | None = None,
        scale: float | None = None,
        rotation: float | None = None,
        stretch_w: float | None = None,
        stretch_h: float | None = None,
        distort: Sequence[float] | None = None,
        *,
        record_undo: bool = True,
    ) -> bool:
        """精确设置自由变换；未传入的参数保持不变。"""
        if self._transform_drag_kind:
            self._finish_transform_drag()
        if not self.has_image:
            return False
        current = self._transform
        normalized_distort = self._normalized_distort(distort, current.distort)
        updated = _TransformState(
            x=self._finite_value(x, current.x),
            y=self._finite_value(y, current.y),
            scale=max(
                self._MIN_SCALE,
                min(self._MAX_SCALE, self._finite_value(scale, current.scale)),
            ),
            rotation=self._finite_value(rotation, current.rotation),
            stretch_w=max(
                self._MIN_SCALE,
                min(
                    self._MAX_SCALE,
                    self._finite_value(stretch_w, current.stretch_w),
                ),
            ),
            stretch_h=max(
                self._MIN_SCALE,
                min(
                    self._MAX_SCALE,
                    self._finite_value(stretch_h, current.stretch_h),
                ),
            ),
            distort=normalized_distort,
        )
        if updated == current or not self._transform_state_is_valid(updated):
            return False
        if record_undo:
            self._push_undo()
        self._apply_transform_state(updated)
        return True

    def transform_controls_in_view(
        self,
        origin: QPointF,
        scale: float,
    ) -> tuple[QPolygonF, dict[str, QPointF], QPointF]:
        """返回映射到外部视图的控制框、八个手柄和旋转手柄。"""
        view = self._validated_transform_view(origin, scale)
        polygon = self._control_polygon(view)
        handles, rotate = self._control_handles(view)
        return polygon, handles, rotate

    def transform_hit_test_in_view(
        self,
        position: QPointF,
        origin: QPointF,
        scale: float,
        modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
    ) -> str:
        """在外部视图坐标中命中自由变换控制层。"""
        view = self._validated_transform_view(origin, scale)
        return self._transform_hit_test(position, modifiers, view)

    def transform_cursor_for_hit(self, kind: str) -> QCursor:
        """返回指定自由变换命中类型对应的鼠标光标。"""
        return self._cursor_for_transform_hit(kind)

    def begin_external_transform(
        self,
        position: QPointF,
        origin: QPointF,
        scale: float,
        modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
    ) -> str:
        """从外部视图开始一轮自由变换拖动，并返回命中类型。"""
        self.end_external_transform()
        view = self._validated_transform_view(origin, scale)
        return self._begin_transform_drag(position, modifiers, view)

    def update_external_transform(
        self,
        position: QPointF,
        modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
    ) -> bool:
        """更新外部自由变换拖动；返回本次更新是否改变参数。"""
        if not self._transform_drag_kind or self._transform_drag_view is None:
            return False
        return self._continue_transform_drag(position, modifiers)

    def end_external_transform(self) -> bool:
        """结束外部自由变换；返回本轮是否产生过有效变化。"""
        return self._finish_transform_drag()

    def reset_transform(self) -> None:
        """仅还原几何参数，保留当前像素修订。"""
        if self._transform_drag_kind:
            self._finish_transform_drag()
        if not self.has_image or self._transform == _TransformState():
            return
        self._push_undo()
        self._apply_transform_state(_TransformState())

    def set_grid_visible(self, visible: bool) -> None:
        visible = bool(visible)
        if visible == self._grid_visible:
            return
        self._grid_visible = visible
        self.update()

    def set_background_mode(self, mode: str) -> None:
        if mode not in {self.BACKGROUND_WHITE, self.BACKGROUND_CHECKERBOARD}:
            return
        if mode == self._background_mode:
            return
        self._background_mode = mode
        self.update()

    def set_source_preview_visible(self, visible: bool) -> None:
        """临时显示载入时的原稿，不改变编辑状态和撤销历史。"""
        normalized = bool(visible) and not self._source_image.isNull()
        if normalized == self._source_preview_visible:
            return
        self._source_preview_visible = normalized
        self.update()

    def set_reference_image(
        self,
        image: QImage | None,
        opacity: float = 0.35,
    ) -> None:
        """设置同田字格坐标系的参照字；传入 ``None`` 可清除参照。"""
        if image is None or image.isNull():
            self._reference_image = QImage()
            self._reference_visible = False
            self.update()
            return
        self._reference_image = image.convertToFormat(QImage.Format.Format_ARGB32)
        self._reference_opacity = self._normalized_opacity(opacity)
        self.update()

    def set_reference_visible(self, visible: bool) -> None:
        """开关参照字叠加，不改变编辑状态和撤销历史。"""
        normalized = bool(visible) and not self._reference_image.isNull()
        if normalized == self._reference_visible:
            return
        self._reference_visible = normalized
        self.update()

    def set_reference_opacity(self, opacity: float) -> None:
        """调整参照字叠加透明度。"""
        normalized = self._normalized_opacity(opacity)
        if abs(normalized - self._reference_opacity) <= 1e-9:
            return
        self._reference_opacity = normalized
        if self._reference_visible:
            self.update()

    def fit_to_view(self) -> None:
        """使 130% 编辑工作区及越界后的完整图像适应当前窗口。"""
        if self._image.isNull() or self.width() <= 0 or self.height() <= 0:
            return
        available_w = max(1, self.width() - 64)
        available_h = max(1, self.height() - 64)
        view_bounds = self._logical_view_bounds()
        zoom = min(available_w / view_bounds.width(), available_h / view_bounds.height())
        self._zoom = max(0.1, min(8.0, zoom))
        self._offset = QPointF()
        self._view_mode = "fit"
        self._emit_zoom()
        self.update()

    def actual_size(self) -> None:
        """按图像像素与屏幕像素 1:1 显示。"""
        if not self.has_image:
            return
        self._zoom = 1.0
        self._offset = QPointF()
        self._view_mode = "actual"
        self._emit_zoom()
        self.update()

    def zoom_in(self) -> None:
        self._view_mode = "custom"
        self._set_zoom(self._zoom * 1.2)

    def zoom_out(self) -> None:
        self._view_mode = "custom"
        self._set_zoom(self._zoom / 1.2)

    def undo(self) -> None:
        self._cancel_active_input()
        if not self._undo_stack:
            return
        self._redo_stack.append(self._snapshot())
        self._restore_state(self._undo_stack.pop())
        self._emit_history()

    def redo(self) -> None:
        self._cancel_active_input()
        if not self._redo_stack:
            return
        self._undo_stack.append(self._snapshot())
        self._restore_state(self._redo_stack.pop())
        self._emit_history()

    def reset_image(self) -> None:
        """还原到最后一次载入或保存状态；本次还原本身可以撤销。"""
        self._cancel_active_input()
        if self._image.isNull() or self._state_matches_saved():
            return
        self._push_undo()
        self._restore_state(self._saved_state)

    def discard_changes(self) -> None:
        """丢弃未保存修改并回到保存基线，废弃内容不留入撤销历史。"""
        self._cancel_active_input(clear_tablet_suppression=True)
        if self._image.isNull():
            return
        self._restore_state(self._saved_state)
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._emit_history()
        self._set_dirty(False)

    def mark_saved(self) -> None:
        """烘焙当前变换，并把结果设为新的保存基线。"""
        self.set_saved_baseline(bake=True)

    def set_saved_baseline(self, bake: bool = False) -> bool:
        """把当前状态设为保存基线；默认保留未烘焙的变换参数。"""
        self._cancel_active_input()
        if self._image.isNull():
            return False
        if bake:
            self._bake_transform(push_undo=False)
        self._saved_state = self._snapshot()
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._emit_history()
        self._set_dirty(False)
        return True

    def mark_transform_saved(self) -> bool:
        """兼容名称：保留当前变换并将其设为保存基线。"""
        return self.set_saved_baseline(bake=False)

    def paintEvent(self, _event) -> None:  # type: ignore[no-untyped-def]
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#11151B"))
        if self._image.isNull():
            painter.setPen(QColor("#7E8998"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "请选择一个字形开始审核")
            return

        target = self._canvas_rect()
        workspace_target = self._workspace_rect()
        source_preview = self._source_preview_visible and not self._source_image.isNull()
        if source_preview:
            image_target = self._layer_target_rect(
                target,
                self._source_image,
                self._source_image_origin,
            )
        else:
            output_size, origin = self._output_geometry()
            image_target = QRectF(
                target.left() - origin.x() * self._zoom,
                target.top() - origin.y() * self._zoom,
                output_size.width() * self._zoom,
                output_size.height() * self._zoom,
            )
        display_target = workspace_target.united(image_target)
        reference_target = QRectF()
        if not source_preview and self._reference_visible and not self._reference_image.isNull():
            reference_target = self._reference_target_rect(target)
            display_target = display_target.united(reference_target)
        painter.save()
        painter.setClipRect(display_target)
        self._draw_background(painter, display_target)
        if self._grid_visible:
            self._draw_grid(painter, target)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self._zoom < 1.0)
        if source_preview:
            painter.drawImage(image_target, self._source_image)
        elif self._render_postprocessor is not None:
            painter.drawImage(image_target, self._final_image())
            if not reference_target.isEmpty():
                painter.save()
                painter.setOpacity(self._reference_opacity)
                painter.drawImage(reference_target, self._reference_image)
                painter.restore()
        else:
            transformed = self._transformed_content()
            painter.drawImage(
                QRectF(
                    target.left() + transformed.origin.x() * self._zoom,
                    target.top() + transformed.origin.y() * self._zoom,
                    transformed.image.width() * self._zoom,
                    transformed.image.height() * self._zoom,
                ),
                transformed.image,
            )
            if not reference_target.isEmpty():
                painter.save()
                painter.setOpacity(self._reference_opacity)
                painter.drawImage(reference_target, self._reference_image)
                painter.restore()
        painter.restore()

        if source_preview:
            return
        if self._tool == self.TOOL_TRANSFORM:
            self._draw_transform_controls(painter)
        elif self._tool in {self.TOOL_BRUSH, self.TOOL_ERASER}:
            self._draw_pointer_preview(painter)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._should_suppress_tablet_mouse(event):
            event.accept()
            return
        if self._tablet_active or self._tablet_panning:
            self._cancel_active_input()
        if not self.has_image:
            return
        self._hover_position = QPointF(event.position())
        if (
            event.button() == Qt.MouseButton.RightButton
            and self._tool == self.TOOL_BRUSH
            and not self._space_pan_held
        ):
            self._pick_ink_at(event.position())
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self.setFocus()
        position = event.position()
        if self._space_pan_held:
            self._begin_pan(position, temporary=True)
            return
        if self._tool == self.TOOL_PAN:
            self._begin_pan(position, temporary=False)
            return
        if self._tool == self.TOOL_TRANSFORM:
            self._begin_transform_drag(position, event.modifiers())
            return
        self._set_pointer_preview(position, float(self._brush_size), self._tool)
        if not self._workspace_rect().contains(position):
            return
        image_point = self._widget_to_image_unbounded(position)
        if image_point is None:
            return
        self._begin_stroke(image_point, self._tool, float(self._brush_size))

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._should_suppress_tablet_mouse(event) or self._tablet_active:
            event.accept()
            return
        position = event.position()
        self._hover_position = QPointF(position)
        if not event.buttons() & Qt.MouseButton.LeftButton:
            if self._panning:
                self._finish_pan(position)
            if self._drawing or self._transform_drag_kind:
                self._cancel_active_input()
            if self._space_pan_held:
                self._clear_pointer_preview()
                self._update_cursor_at(position)
                return
            if self._tool in {self.TOOL_BRUSH, self.TOOL_ERASER}:
                self._set_pointer_preview(position, float(self._brush_size), self._tool)
            if self._tool == self.TOOL_TRANSFORM:
                kind = self._transform_hit_test(position, event.modifiers())
                self.setCursor(self._cursor_for_transform_hit(kind))
            else:
                self._update_cursor_at(position)
            return
        if self._panning:
            self._continue_pan(position)
            return
        if self._transform_drag_kind:
            self._continue_transform_drag(position, event.modifiers())
            return
        if not self._drawing:
            return
        self._set_pointer_preview(position, float(self._brush_size), self._tool)
        if not self._workspace_rect().contains(position):
            self._last_stroke_width = 0.0
            return
        image_point = self._widget_to_image_unbounded(position)
        if image_point is None:
            return
        self._continue_stroke(image_point, float(self._brush_size))

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._should_suppress_tablet_mouse(event) or self._tablet_active:
            event.accept()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._hover_position = QPointF(event.position())
        if self._panning:
            self._finish_pan(event.position())
            return
        if self._drawing:
            self._end_stroke()
        self._finish_transform_drag()
        self._update_cursor_at(event.position())

    def tabletEvent(self, event: QTabletEvent) -> None:
        """处理压感笔尖和笔尾；接受事件以抑制随后合成的鼠标事件。"""
        event_type = event.type()
        if event_type not in {
            QEvent.Type.TabletPress,
            QEvent.Type.TabletMove,
            QEvent.Type.TabletRelease,
        }:
            event.ignore()
            return
        self._hover_position = QPointF(event.position())

        if self._space_pan_held or (self._space_pan_drag and self._tablet_panning):
            if not self.has_image:
                event.ignore()
                return
            event.accept()
            if event_type == QEvent.Type.TabletPress:
                self._suppressed_tablet_device_key = self._pointing_device_key(event.device())
                self._suppressed_mouse_sequence_active = False
                if event.button() in {Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton}:
                    self._begin_pan(event.position(), temporary=True, tablet=True)
                return
            if not self._tablet_panning:
                return
            if event_type == QEvent.Type.TabletMove:
                if not event.buttons() & Qt.MouseButton.LeftButton and event.pressure() <= 0.0:
                    self._finish_pan(event.position())
                    return
                self._continue_pan(event.position())
                return
            self._finish_pan(event.position())
            return

        if not self.has_image or self._tool not in {self.TOOL_BRUSH, self.TOOL_ERASER}:
            if event_type == QEvent.Type.TabletPress:
                self._clear_tablet_mouse_suppression()
            event.ignore()
            return
        event.accept()
        if event_type == QEvent.Type.TabletPress:
            self._suppressed_tablet_device_key = self._pointing_device_key(event.device())
            self._suppressed_mouse_sequence_active = False
        pointer_type = event.pointerType()
        stroke_tool = (
            self.TOOL_ERASER
            if pointer_type == QPointingDevice.PointerType.Eraser
            else self._tool
        )
        pressure = self._tablet_pressure(event)
        stroke_width = self._pressure_width(pressure)
        self._set_pointer_preview(event.position(), stroke_width, stroke_tool)
        self._update_cursor_at(event.position())

        if event_type == QEvent.Type.TabletPress:
            if (
                event.button() == Qt.MouseButton.RightButton
                and pointer_type != QPointingDevice.PointerType.Eraser
                and self._tool == self.TOOL_BRUSH
            ):
                self._pick_ink_at(event.position())
                return
            if event.button() not in {Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton}:
                return
            if not self._workspace_rect().contains(event.position()):
                return
            image_point = self._widget_to_image_unbounded(event.position())
            if image_point is None:
                return
            self._tablet_active = True
            self._begin_stroke(image_point, stroke_tool, stroke_width)
            return

        if event_type == QEvent.Type.TabletMove:
            if not self._tablet_active or not self._drawing:
                return
            if not event.buttons() & Qt.MouseButton.LeftButton:
                self._end_stroke()
                self._tablet_active = False
                return
            if not self._workspace_rect().contains(event.position()):
                self._last_stroke_width = 0.0
                return
            image_point = self._widget_to_image_unbounded(event.position())
            if image_point is None:
                return
            self._continue_stroke(image_point, stroke_width)
            return

        if self._tablet_active and self._drawing:
            self._end_stroke()
        self._tablet_active = False
        self._pointer_stroke_width = float(self._brush_size)
        self._update_pointer_region()

    def event(self, event: QEvent) -> bool:
        """输入被窗口状态中断时提交已产生的部分笔画。"""
        if event.type() in {
            QEvent.Type.FocusOut,
            QEvent.Type.Hide,
            QEvent.Type.UngrabMouse,
            QEvent.Type.WindowDeactivate,
        } and hasattr(self, "_drawing"):
            self._cancel_active_input(clear_tablet_suppression=True)
        return super().event(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if not self.has_image:
            return
        if (
            self._tool in {self.TOOL_BRUSH, self.TOOL_ERASER}
            and not self.space_pan_active
            and not event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            delta = event.angleDelta().y()
            if delta == 0:
                delta = event.pixelDelta().y() * 8
            self._brush_wheel_remainder += delta
            steps = math.trunc(self._brush_wheel_remainder / 120)
            if steps:
                self._brush_wheel_remainder -= steps * 120
                for _index in range(abs(steps)):
                    self._change_brush_size(1 if steps > 0 else -1)
            event.accept()
            return
        if event.angleDelta().y() == 0:
            return
        old_rect = self._canvas_rect()
        logical_point = QPointF(
            (event.position().x() - old_rect.left()) / self._zoom,
            (event.position().y() - old_rect.top()) / self._zoom,
        )
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self._view_mode = "custom"
        self._set_zoom(self._zoom * factor)
        new_rect = self._canvas_rect()
        mapped = QPointF(
            new_rect.left() + logical_point.x() * self._zoom,
            new_rect.top() + logical_point.y() * self._zoom,
        )
        self._offset += event.position() - mapped
        self.update()
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """处理临时抓手及像素工具快捷键。"""
        if event.key() == Qt.Key.Key_Space:
            if self.handle_space_pan_key(True, event.isAutoRepeat()):
                event.accept()
                return
        if self._tool in {self.TOOL_BRUSH, self.TOOL_ERASER}:
            if event.key() == Qt.Key.Key_BracketLeft or event.text() == "[":
                self._change_brush_size(-1)
                event.accept()
                return
            if event.key() == Qt.Key.Key_BracketRight or event.text() == "]":
                self._change_brush_size(1)
                event.accept()
                return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space:
            if self.handle_space_pan_key(False, event.isAutoRepeat()):
                event.accept()
                return
        super().keyReleaseEvent(event)

    def handle_space_pan_key(self, pressed: bool, auto_repeat: bool = False) -> bool:
        """由画布或页面路由空格键，不切换当前编辑工具。"""
        if auto_repeat:
            return self.space_pan_active
        if pressed:
            if not self.has_image:
                return False
            if self._space_pan_held or self._space_pan_blocked:
                return True
            if self._drawing or self._tablet_active or self._transform_drag_kind or self._panning:
                self._space_pan_blocked = True
                return True
            self._space_pan_held = True
            self._clear_pointer_preview()
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            return True

        if not self.space_pan_active:
            return False
        self._space_pan_held = False
        self._space_pan_blocked = False
        if self._space_pan_drag and self._panning:
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        else:
            self._restore_tool_hover()
        return True

    def leaveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self._hover_position = None
        self._clear_pointer_preview()
        self._update_cursor_at(None)
        super().leaveEvent(event)

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        """适应窗口模式随可用空间变化重新计算，1:1 与自定义缩放保持不变。"""
        super().resizeEvent(event)
        if self.has_image and self._view_mode == "fit":
            self.fit_to_view()

    def _canvas_rect(self) -> QRectF:
        width = max(1.0, self._canvas_size.width() * self._zoom)
        height = max(1.0, self._canvas_size.height() * self._zoom)
        left = (self.width() - width) / 2.0 + self._offset.x()
        top = (self.height() - height) / 2.0 + self._offset.y()
        return QRectF(left, top, width, height)

    def _layer_target_rect(
        self,
        canvas_target: QRectF,
        image: QImage,
        origin: QPointF,
    ) -> QRectF:
        return QRectF(
            canvas_target.left() + origin.x() * self._zoom,
            canvas_target.top() + origin.y() * self._zoom,
            image.width() * self._zoom,
            image.height() * self._zoom,
        )

    def _reference_target_rect(self, canvas_target: QRectF) -> QRectF:
        origin = QPointF(
            (self._canvas_size.width() - self._reference_image.width()) / 2.0,
            (self._canvas_size.height() - self._reference_image.height()) / 2.0,
        )
        return self._layer_target_rect(canvas_target, self._reference_image, origin)

    def _workspace_rect(self) -> QRectF:
        """返回与田字格同心的 130% 编辑工作区屏幕范围。"""
        target = self._canvas_rect()
        margin_x = target.width() * (self.WORKSPACE_RATIO - 1.0) / 2.0
        margin_y = target.height() * (self.WORKSPACE_RATIO - 1.0) / 2.0
        return target.adjusted(-margin_x, -margin_y, margin_x, margin_y)

    def _image_rect(self) -> QRect:
        """保留旧页面可能使用的整数目标矩形。"""
        return self._canvas_rect().toAlignedRect()

    def _widget_to_image(self, point: QPointF, clamp: bool = False) -> QPoint | None:
        source = self._widget_to_image_unbounded(point)
        if source is None:
            return None
        x = source.x()
        y = source.y()
        if clamp:
            x = max(0, min(self._image.width() - 1, x))
            y = max(0, min(self._image.height() - 1, y))
        elif x < 0 or y < 0 or x >= self._image.width() or y >= self._image.height():
            return None
        return QPoint(x, y)

    def _widget_to_image_unbounded(self, point: QPointF) -> QPoint | None:
        """将屏幕坐标映射到图像坐标，允许结果位于当前图像边界外。"""
        target = self._canvas_rect()
        logical = QPointF(
            (point.x() - target.left()) / self._zoom,
            (point.y() - target.top()) / self._zoom,
        )
        inverse, invertible = self._content_transform().inverted()
        if not invertible:
            return None
        source = inverse.map(logical)
        x = math.floor(source.x())
        y = math.floor(source.y())
        return QPoint(x, y)

    def _begin_stroke(self, point: QPoint, tool: str, width: float) -> None:
        self._drawing = True
        self._stroke_tool = tool
        if tool == self.TOOL_BRUSH:
            self._stroke_brush_color = QColor(self._brush_color)
            self._stroke_ink_coverage = self._local_ink_coverage(
                self._image,
                point,
                width,
                self._brush_ink_coverage,
            )
        self._stroke_before_state = self._snapshot()
        self._stroke_changed = False
        point, _shift = self._expand_image_for_stroke(point, width, tool)
        if point is None:
            self._end_stroke()
            return
        self._last_image_point = point
        self._last_stroke_width = width
        self._draw_line(point, point, width, width, tool)

    def _continue_stroke(self, point: QPoint, width: float) -> None:
        point, shift = self._expand_image_for_stroke(point, width, self._stroke_tool)
        if point is None:
            self._last_image_point = QPoint()
            self._last_stroke_width = 0.0
            return
        if self._last_stroke_width <= 0.0:
            self._last_image_point = point
            self._last_stroke_width = width
        elif not shift.isNull():
            self._last_image_point += shift
        self._draw_line(
            self._last_image_point,
            point,
            self._last_stroke_width,
            width,
            self._stroke_tool,
        )
        self._last_image_point = point
        self._last_stroke_width = width

    def _expand_image_for_stroke(
        self,
        point: QPoint,
        width: float,
        tool: str,
    ) -> tuple[QPoint | None, QPoint]:
        """画笔越出当前图像时扩展透明边，田字格逻辑坐标保持不变。"""
        if tool == self.TOOL_ERASER:
            return QPoint(point), QPoint()
        radius = max(1, math.ceil(float(width) / 2.0 + 1.0))
        left = max(0, radius - point.x())
        top = max(0, radius - point.y())
        right = max(0, point.x() + radius + 1 - self._image.width())
        bottom = max(0, point.y() + radius + 1 - self._image.height())
        if not any((left, top, right, bottom)):
            return QPoint(point), QPoint()
        expanded = QImage(
            self._image.width() + left + right,
            self._image.height() + top + bottom,
            QImage.Format.Format_ARGB32,
        )
        expanded.fill(Qt.GlobalColor.transparent)
        painter = QPainter(expanded)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        painter.drawImage(QPoint(left, top), self._image)
        painter.end()
        self._image = expanded
        self._image_origin -= QPointF(left, top)
        if not self._content_bounds.isEmpty():
            self._content_bounds.translate(float(left), float(top))
        self._invalidate_render()
        shift = QPoint(left, top)
        return point + shift, shift

    def _end_stroke(self) -> None:
        self._drawing = False
        self._refresh_content_bounds()
        self._stroke_before_state = None
        self._stroke_changed = False

    def _begin_pan(
        self,
        position: QPointF,
        *,
        temporary: bool,
        tablet: bool = False,
    ) -> None:
        self._panning = True
        self._space_pan_drag = temporary
        self._tablet_panning = tablet
        self._drag_origin = QPointF(position)
        self._clear_pointer_preview()
        self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def _continue_pan(self, position: QPointF) -> None:
        delta = position - self._drag_origin
        if delta.isNull():
            return
        self._offset += delta
        self._drag_origin = QPointF(position)
        self._view_mode = "custom"
        self.update()

    def _finish_pan(self, position: QPointF | None = None) -> None:
        if position is not None:
            self._hover_position = QPointF(position)
        self._panning = False
        self._tablet_panning = False
        self._space_pan_drag = False
        if self._space_pan_held:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self._restore_tool_hover()

    def _restore_tool_hover(self) -> None:
        if self._tool in {self.TOOL_BRUSH, self.TOOL_ERASER} and self._hover_position is not None:
            self._set_pointer_preview(
                self._hover_position,
                float(self._brush_size),
                self._tool,
            )
        self._update_cursor_at(self._hover_position)

    def _cancel_active_input(self, clear_tablet_suppression: bool = False) -> None:
        if self._drawing:
            self._end_stroke()
        self._drawing = False
        self._panning = False
        self._space_pan_held = False
        self._space_pan_blocked = False
        self._space_pan_drag = False
        self._tablet_panning = False
        self._tablet_active = False
        self._finish_transform_drag()
        if clear_tablet_suppression:
            self._clear_tablet_mouse_suppression()
        self._restore_tool_hover()

    def _change_brush_size(self, direction: int) -> None:
        """按旧版分段步长调整画笔或橡皮的基础笔触。"""
        if (
            self._tool not in {self.TOOL_BRUSH, self.TOOL_ERASER}
            or direction == 0
            or self._drawing
            or self._tablet_active
        ):
            return
        step = 1 if self._brush_size < 10 else 2 if self._brush_size < 30 else 5
        self.set_brush_size(self._brush_size + (step if direction > 0 else -step))

    def _should_suppress_tablet_mouse(self, event: QMouseEvent) -> bool:
        if event.source() == Qt.MouseEventSource.MouseEventNotSynthesized:
            if event.type() == QEvent.Type.MouseButtonPress:
                self._clear_tablet_mouse_suppression()
            return False
        device_key = self._pointing_device_key(event.pointingDevice())
        if device_key is None or device_key != self._suppressed_tablet_device_key:
            return False
        event_type = event.type()
        if event_type == QEvent.Type.MouseButtonPress:
            self._suppressed_mouse_sequence_active = True
            return True
        if event_type == QEvent.Type.MouseMove:
            return (
                self._tablet_active
                or self._tablet_panning
                or self._suppressed_mouse_sequence_active
            )
        if event_type == QEvent.Type.MouseButtonRelease:
            suppress = (
                self._tablet_active
                or self._tablet_panning
                or self._suppressed_mouse_sequence_active
            )
            if self._suppressed_mouse_sequence_active:
                self._clear_tablet_mouse_suppression()
            return suppress
        return False

    def _clear_tablet_mouse_suppression(self) -> None:
        self._suppressed_tablet_device_key = None
        self._suppressed_mouse_sequence_active = False

    @staticmethod
    def _pointing_device_key(device: QPointingDevice | None) -> tuple[int, int, int] | None:
        if device is None:
            return None
        return (
            int(device.systemId()),
            int(device.type().value),
            int(device.pointerType().value),
        )

    def _draw_line(
        self,
        start: QPoint,
        end: QPoint,
        start_width: float | None = None,
        end_width: float | None = None,
        tool: str | None = None,
    ) -> bool:
        start_width = float(self._brush_size) if start_width is None else max(1.0, float(start_width))
        end_width = start_width if end_width is None else max(1.0, float(end_width))
        stroke_tool = tool or self._tool
        dirty_source = self._stroke_source_rect(start, end, max(start_width, end_width))
        if dirty_source.isEmpty():
            return False
        before = self._image.copy(dirty_source)
        if stroke_tool == self.TOOL_ERASER:
            painter = QPainter(self._image)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            distance = max(abs(end.x() - start.x()), abs(end.y() - start.y()))
            steps = max(
                1,
                int(
                    math.ceil(
                        distance / max(1.0, min(start_width, end_width) * 0.35)
                    )
                ),
            )
            for index in range(steps + 1):
                ratio = index / steps
                x = start.x() + (end.x() - start.x()) * ratio
                y = start.y() + (end.y() - start.y()) * ratio
                width = start_width + (end_width - start_width) * ratio
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(0, 0, 0, 0))
                painter.drawEllipse(QPointF(x, y), width / 2.0, width / 2.0)
            painter.end()
        else:
            self._blend_ink_stroke(
                dirty_source,
                start,
                end,
                start_width,
                end_width,
            )
        if before == self._image.copy(dirty_source):
            return False
        self._commit_stroke_undo()
        self._update_content_bounds_after_stroke(dirty_source, stroke_tool)
        self._invalidate_render()
        self._set_dirty(True)
        self.pixels_changed.emit()
        self.update(self._stroke_dirty_rect(start, end, max(start_width, end_width)))
        return True

    def _blend_ink_stroke(
        self,
        dirty_source: QRect,
        start: QPoint,
        end: QPoint,
        start_width: float,
        end_width: float,
    ) -> None:
        """按附近原笔画的视觉墨量融合补笔，避免重叠区域反复变黑。"""

        mask = QImage(
            dirty_source.size(),
            QImage.Format.Format_ARGB32,
        )
        mask.fill(Qt.GlobalColor.transparent)
        painter = QPainter(mask)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        distance = max(abs(end.x() - start.x()), abs(end.y() - start.y()))
        steps = max(
            1,
            int(
                math.ceil(
                    distance / max(1.0, min(start_width, end_width) * 0.35)
                )
            ),
        )
        for index in range(steps + 1):
            ratio = index / steps
            x = start.x() + (end.x() - start.x()) * ratio - dirty_source.x()
            y = start.y() + (end.y() - start.y()) * ratio - dirty_source.y()
            width = start_width + (end_width - start_width) * ratio
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 255, 255, 255))
            painter.drawEllipse(QPointF(x, y), width / 2.0, width / 2.0)
        painter.end()

        pixels = self._qimage_to_rgba(self._image.copy(dirty_source))
        mask_alpha = self._qimage_to_rgba(mask)[..., 3].astype(np.float32)
        desired_coverage = np.rint(
            mask_alpha * float(self._stroke_ink_coverage) / 255.0
        ).astype(np.uint8)
        current_coverage = self._visual_coverage(pixels)
        changed = desired_coverage > current_coverage
        if not np.any(changed):
            return

        color = QColor(self._stroke_brush_color)
        red, green, blue = color.red(), color.green(), color.blue()
        darkness = 255.0 - (
            red * 0.299
            + green * 0.587
            + blue * 0.114
        )
        if darkness < 1.0:
            red = green = blue = 0
            darkness = 255.0
        required_alpha = np.clip(
            np.rint(desired_coverage.astype(np.float32) * 255.0 / darkness),
            0.0,
            255.0,
        ).astype(np.uint8)
        pixels[changed, 0] = red
        pixels[changed, 1] = green
        pixels[changed, 2] = blue
        pixels[changed, 3] = required_alpha[changed]

        updated = self._rgba_to_qimage(pixels)
        image_painter = QPainter(self._image)
        image_painter.setCompositionMode(
            QPainter.CompositionMode.CompositionMode_Source
        )
        image_painter.drawImage(dirty_source.topLeft(), updated)
        image_painter.end()

    def _commit_stroke_undo(self) -> None:
        """首次实际改动像素时提交落笔前快照，整笔只产生一个撤销记录。"""
        if self._stroke_changed or self._stroke_before_state is None:
            return
        self._undo_stack.append(self._stroke_before_state)
        if len(self._undo_stack) > 30:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self._emit_history()
        self._stroke_changed = True

    def _stroke_source_rect(self, start: QPoint, end: QPoint, width: float) -> QRect:
        radius = max(1.0, float(width) / 2.0 + 2.0)
        return (
            QRectF(QPointF(start), QPointF(end))
            .normalized()
            .adjusted(-radius, -radius, radius, radius)
            .toAlignedRect()
            .intersected(self._image.rect())
        )

    def _update_content_bounds_after_stroke(self, dirty: QRect, tool: str) -> None:
        """按局部 Alpha 变化维护活动笔画期间可立即使用的内容边界。"""
        if tool == self.TOOL_BRUSH:
            local_bounds = self._alpha_bounds(dirty)
            if local_bounds.isEmpty():
                return
            self._content_bounds = (
                local_bounds
                if self._content_bounds.isEmpty()
                else self._content_bounds.united(local_bounds)
            )
            return
        if self._content_bounds.isEmpty():
            return
        dirty_bounds = QRectF(dirty)
        content = self._content_bounds
        reaches_edge = (
            dirty_bounds.left() <= content.left()
            or dirty_bounds.top() <= content.top()
            or dirty_bounds.right() >= content.right()
            or dirty_bounds.bottom() >= content.bottom()
        )
        if dirty_bounds.intersects(content) and reaches_edge:
            self._refresh_content_bounds()

    def _alpha_bounds(self, region: QRect) -> QRectF:
        if region.isEmpty():
            return QRectF()
        alpha = self._image.copy(region).convertToFormat(QImage.Format.Format_Alpha8)
        width = alpha.width()
        height = alpha.height()
        stride = alpha.bytesPerLine()
        pixels = np.frombuffer(
            alpha.constBits(),
            dtype=np.uint8,
            count=stride * height,
        ).reshape(height, stride)[:, :width]
        occupied_rows = np.flatnonzero(np.any(pixels, axis=1))
        occupied_columns = np.flatnonzero(np.any(pixels, axis=0))
        if occupied_rows.size == 0 or occupied_columns.size == 0:
            return QRectF()
        left = region.left() + int(occupied_columns[0])
        top = region.top() + int(occupied_rows[0])
        right = region.left() + int(occupied_columns[-1])
        bottom = region.top() + int(occupied_rows[-1])
        return QRectF(float(left), float(top), float(right - left + 1), float(bottom - top + 1))

    def _content_transform(self) -> QTransform:
        """返回像素编辑使用的仿射映射；进入像素工具前复杂变换会先烘焙。"""
        center_x = self._canvas_size.width() / 2.0
        center_y = self._canvas_size.height() / 2.0
        angle = math.radians(self._transform.rotation)
        cos_value = math.cos(angle) * self._transform.scale
        sin_value = math.sin(angle) * self._transform.scale
        offset_x = self._image_origin.x() - center_x
        offset_y = self._image_origin.y() - center_y
        dx = (
            center_x
            + self._transform.x
            + cos_value * offset_x
            - sin_value * offset_y
        )
        dy = (
            center_y
            + self._transform.y
            + sin_value * offset_x
            + cos_value * offset_y
        )
        return QTransform(
            cos_value,
            sin_value,
            -sin_value,
            cos_value,
            dx,
            dy,
        )

    def _render_final_image(self) -> QImage:
        output_size, origin = self._output_geometry()
        if (
            self._transform == _TransformState()
            and output_size == self._image.size()
            and abs(self._image_origin.x() + origin.x()) <= 1e-9
            and abs(self._image_origin.y() + origin.y()) <= 1e-9
            and self._render_postprocessor is None
        ):
            self._rendered_origin = origin
            return QImage(self._image)
        transformed = self._transformed_content()
        rendered = compose_rgba_on_canvas(
            self._qimage_to_rgba(transformed.image),
            (transformed.origin.x(), transformed.origin.y()),
            (self._canvas_size.width(), self._canvas_size.height()),
            expand_symmetric=True,
            limits=self._transform_limits(),
        )
        self._rendered_origin = QPoint(*rendered.geometry.grid_origin)
        pixels = self._postprocess_rendered_rgba(rendered.pixels)
        return self._rgba_to_qimage(pixels).convertToFormat(
            QImage.Format.Format_ARGB32
        )

    def _transformed_content(self) -> _TransformedContent:
        if self._transformed_content_cache is not None:
            return self._transformed_content_cache
        if self._image.isNull():
            empty = _TransformedContent(QImage(), QPointF(), self._empty_polygon(), QRectF())
            self._transformed_content_cache = empty
            return empty
        if self._transform == _TransformState():
            bounds = self._content_bounds
            if bounds.isEmpty():
                bounds = QRectF(self._image.rect())
            origin = QPointF(self._image_origin)
            right = bounds.left() + max(0.0, bounds.width() - 1.0)
            bottom = bounds.top() + max(0.0, bounds.height() - 1.0)
            polygon = tuple(
                origin + point
                for point in (
                    bounds.topLeft(),
                    QPointF(right, bounds.top()),
                    QPointF(right, bottom),
                    QPointF(bounds.left(), bottom),
                )
            )
            cached = _TransformedContent(
                QImage(self._image),
                origin,
                polygon,
                bounds.translated(origin),
            )
            self._transformed_content_cache = cached
            return cached

        geometry = self._calculate_transform_geometry(self._transform)
        source = self._image.copy(geometry.source_rect)
        shared_geometry = calculate_transform_geometry(
            (source.width(), source.height()),
            scale_x=self._transform.scale * self._transform.stretch_w,
            scale_y=self._transform.scale * self._transform.stretch_h,
            rotation=self._transform.rotation,
            distort=self._transform.distort,
            limits=self._transform_limits(),
        )
        pixels = render_transformed_rgba(
            self._qimage_to_rgba(source),
            shared_geometry,
            force_rotation=abs(self._transform.rotation) > 1e-9,
        )
        image = self._rgba_to_qimage(pixels)
        local_alpha = self._image_alpha_bounds(image)
        cached = _TransformedContent(
            image,
            geometry.origin,
            geometry.polygon,
            local_alpha.translated(geometry.origin) if not local_alpha.isEmpty() else QRectF(),
        )
        self._transformed_content_cache = cached
        return cached

    def _postprocess_rendered_rgba(self, pixels: np.ndarray) -> np.ndarray:
        processor = self._render_postprocessor
        if processor is None:
            return pixels
        processed = np.asarray(processor(pixels.copy()))
        if processed.shape != pixels.shape or processed.dtype != np.uint8:
            raise ValueError("画布渲染后处理器必须返回同尺寸的 uint8 RGBA 图像")
        return np.ascontiguousarray(processed)

    def _calculate_transform_geometry(self, state: _TransformState) -> _TransformGeometry:
        source_rect = self._content_source_rect()
        source_width = max(1, source_rect.width())
        source_height = max(1, source_rect.height())
        shared = calculate_transform_geometry(
            (source_width, source_height),
            scale_x=state.scale * state.stretch_w,
            scale_y=state.scale * state.stretch_h,
            rotation=state.rotation,
            distort=state.distort,
            limits=self._transform_limits(),
        )
        source_center = QPointF(
            self._image_origin.x() + source_rect.x() + source_width / 2.0,
            self._image_origin.y() + source_rect.y() + source_height / 2.0,
        )
        placement = place_transform(
            shared,
            (source_center.x(), source_center.y()),
            (state.x, state.y),
        )
        polygon = tuple(
            QPointF(point[0], point[1]) for point in placement.polygon
        )
        return _TransformGeometry(
            source_rect=source_rect,
            scaled_size=QSize(*shared.scaled_size),
            perspective_size=QSize(*shared.perspective_size),
            output_size=QSize(*shared.output_size),
            perspective_matrix=shared.perspective_matrix,
            rotation_matrix=shared.rotation_matrix,
            origin=QPointF(*placement.origin),
            polygon=polygon,
        )

    def _draw_background(self, painter: QPainter, target: QRectF) -> None:
        if self._background_mode == self.BACKGROUND_WHITE:
            painter.fillRect(target, QColor("#FFFFFF"))
            return
        tile = 16
        left = math.floor(target.left())
        top = math.floor(target.top())
        right = math.ceil(target.right())
        bottom = math.ceil(target.bottom())
        colors = (QColor("#FFFFFF"), QColor("#D8DADD"))
        for row, y in enumerate(range(top, bottom, tile)):
            for column, x in enumerate(range(left, right, tile)):
                painter.fillRect(QRect(x, y, tile, tile), colors[(row + column) % 2])

    @staticmethod
    def _draw_grid(painter: QPainter, target: QRectF) -> None:
        grid_color = QColor("#D9A3A3")
        painter.setPen(QPen(grid_color, 1, Qt.PenStyle.SolidLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        center = target.center()
        painter.drawRect(target)
        painter.setPen(QPen(grid_color, 1, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(center.x(), target.top()), QPointF(center.x(), target.bottom()))
        painter.drawLine(QPointF(target.left(), center.y()), QPointF(target.right(), center.y()))
        painter.drawLine(target.topLeft(), target.bottomRight())
        painter.drawLine(target.topRight(), target.bottomLeft())

    def _draw_pointer_preview(self, painter: QPainter) -> None:
        """绘制随笔触宽度、缩放和绘图板压力变化的圆形笔尖。"""
        if self._pointer_position is None or not self.rect().contains(
            self._pointer_position.toPoint()
        ) or not self._workspace_rect().contains(self._pointer_position):
            return
        radius = self._pointer_preview_radius()
        bounds = QRectF(
            self._pointer_position.x() - radius,
            self._pointer_position.y() - radius,
            radius * 2.0,
            radius * 2.0,
        )
        accent = QColor("#FF453A" if self._pointer_tool == self.TOOL_ERASER else "#35C759")
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(17, 17, 17, 220), 3))
        painter.drawEllipse(bounds)
        painter.setPen(QPen(accent, 1))
        painter.drawEllipse(bounds)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(accent)
        painter.drawEllipse(self._pointer_position, 1.8, 1.8)
        painter.restore()

    def _pointer_preview_radius(self) -> float:
        diameter = max(6.0, self._pointer_stroke_width * self._zoom)
        return diameter / 2.0

    def _pointer_preview_rect(self) -> QRect:
        if self._pointer_position is None:
            return QRect()
        radius = self._pointer_preview_radius() + 4.0
        return QRectF(
            self._pointer_position.x() - radius,
            self._pointer_position.y() - radius,
            radius * 2.0,
            radius * 2.0,
        ).toAlignedRect().intersected(self.rect())

    def _update_pointer_region(self, old_rect: QRect | None = None) -> None:
        dirty = self._pointer_preview_rect()
        if old_rect is not None:
            dirty = dirty.united(old_rect)
        if dirty.isEmpty():
            return
        self.update(dirty)

    def _set_pointer_preview(self, position: QPointF, width: float, tool: str) -> None:
        old_rect = self._pointer_preview_rect()
        self._pointer_position = (
            QPointF(position) if self._workspace_rect().contains(position) else None
        )
        self._pointer_stroke_width = max(1.0, float(width))
        self._pointer_tool = tool
        self._update_pointer_region(old_rect)

    def _clear_pointer_preview(self) -> None:
        old_rect = self._pointer_preview_rect()
        self._pointer_position = None
        if not old_rect.isEmpty():
            self.update(old_rect)

    def _control_polygon(self, view: _TransformView | None = None) -> QPolygonF:
        if not self.has_image:
            return QPolygonF()
        geometry = self._calculate_transform_geometry(self._transform)
        active_view = view or self._native_transform_view()
        polygon = QPolygonF()
        for logical in geometry.polygon:
            polygon.append(
                QPointF(
                    active_view.origin.x() + logical.x() * active_view.scale,
                    active_view.origin.y() + logical.y() * active_view.scale,
                )
            )
        return polygon

    def _control_handles(
        self,
        view: _TransformView | None = None,
    ) -> tuple[dict[str, QPointF], QPointF]:
        polygon = self._control_polygon(view)
        if polygon.count() < 4:
            return {}, QPointF()
        corners = [polygon.at(index) for index in range(4)]
        handles = {
            "nw": corners[0],
            "ne": corners[1],
            "se": corners[2],
            "sw": corners[3],
            "n": (corners[0] + corners[1]) / 2.0,
            "e": (corners[1] + corners[2]) / 2.0,
            "s": (corners[2] + corners[3]) / 2.0,
            "w": (corners[3] + corners[0]) / 2.0,
        }
        center = self._polygon_center(tuple(corners))
        top = handles["n"]
        vector = top - center
        length = max(1.0, math.hypot(vector.x(), vector.y()))
        rotate = top + vector * (self._ROTATE_HANDLE_DISTANCE / length)
        return handles, rotate

    def _draw_transform_controls(self, painter: QPainter) -> None:
        polygon = self._control_polygon()
        if polygon.count() < 4:
            return
        handles, rotate = self._control_handles()
        pen = QPen(QColor("#2776C7"), 1.5)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPolygon(polygon)
        painter.drawLine(handles["n"], rotate)
        painter.setPen(QPen(QColor("#174F86"), 1))
        painter.setBrush(QColor("#F7FBFF"))
        radius = self._HANDLE_RADIUS
        for point in handles.values():
            painter.drawRect(QRectF(point.x() - radius, point.y() - radius, radius * 2, radius * 2))
        painter.drawEllipse(rotate, radius, radius)

    def _transform_hit_test(
        self,
        position: QPointF,
        modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
        view: _TransformView | None = None,
    ) -> str:
        """返回自由变换控制层在指定屏幕坐标下的命中类型。"""
        if not self.has_image:
            return ""
        handles, rotate = self._control_handles(view)
        hit_radius = self._HANDLE_RADIUS + 5.0
        if self._distance(position, rotate) <= hit_radius:
            return "rotate"
        for name, point in handles.items():
            if self._distance(position, point) <= hit_radius:
                prefix = "distort" if modifiers & Qt.KeyboardModifier.ControlModifier else "scale"
                return f"{prefix}:{name}"
        if self._control_polygon(view).containsPoint(position, Qt.FillRule.OddEvenFill):
            return "move"
        return ""

    def _cursor_for_transform_hit(self, kind: str) -> QCursor:
        if kind == "move":
            return QCursor(Qt.CursorShape.SizeAllCursor)
        if kind == "rotate":
            return self._rotation_handle_cursor()
        if kind.startswith("distort:"):
            name = kind.partition(":")[2]
            return QCursor(
                Qt.CursorShape.CrossCursor
                if name in {"nw", "ne", "se", "sw"}
                else Qt.CursorShape.SizeAllCursor
            )
        if not kind.startswith("scale:"):
            return QCursor(Qt.CursorShape.ArrowCursor)
        name = kind.partition(":")[2]
        base_angles = {
            "e": 0.0,
            "w": 0.0,
            "nw": 45.0,
            "se": 45.0,
            "n": 90.0,
            "s": 90.0,
            "ne": 135.0,
            "sw": 135.0,
        }
        if name not in base_angles:
            return QCursor(Qt.CursorShape.ArrowCursor)
        angle = (base_angles[name] + self._transform.rotation) % 180.0
        direction = int(round(angle / 45.0)) % 4
        shape = {
            0: Qt.CursorShape.SizeHorCursor,
            1: Qt.CursorShape.SizeFDiagCursor,
            2: Qt.CursorShape.SizeVerCursor,
            3: Qt.CursorShape.SizeBDiagCursor,
        }[direction]
        return QCursor(shape)

    def _rotation_handle_cursor(self) -> QCursor:
        if self._rotation_cursor is not None:
            return self._rotation_cursor
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(QColor("#111111"), 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(QRectF(6, 6, 20, 20), 35 * 16, 275 * 16)
        painter.setPen(QPen(QColor("#F7FBFF"), 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(QRectF(6, 6, 20, 20), 35 * 16, 275 * 16)
        painter.setPen(QPen(QColor("#111111"), 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(22, 4), QPointF(27, 8))
        painter.drawLine(QPointF(27, 8), QPointF(21, 10))
        painter.setPen(QPen(QColor("#F7FBFF"), 1, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(22, 4), QPointF(27, 8))
        painter.drawLine(QPointF(27, 8), QPointF(21, 10))
        painter.end()
        self._rotation_cursor = QCursor(pixmap, 16, 16)
        return self._rotation_cursor

    def _update_cursor_at(self, position: QPointF | None) -> None:
        """根据当前工具和悬停位置刷新系统光标。"""
        if self._space_pan_held or self._space_pan_drag:
            self.setCursor(
                Qt.CursorShape.ClosedHandCursor if self._panning else Qt.CursorShape.OpenHandCursor
            )
            return
        if self._tool == self.TOOL_PAN:
            self.setCursor(
                Qt.CursorShape.ClosedHandCursor if self._panning else Qt.CursorShape.OpenHandCursor
            )
            return
        if self._tool in {self.TOOL_BRUSH, self.TOOL_ERASER}:
            inside = (
                position is not None
                and self.has_image
                and self._workspace_rect().contains(position)
            )
            self.setCursor(Qt.CursorShape.BlankCursor if inside else Qt.CursorShape.ArrowCursor)
            return
        if self._transform_drag_kind:
            self.setCursor(self._cursor_for_transform_hit(self._transform_drag_kind))
            return
        kind = self._transform_hit_test(position) if position is not None else ""
        self.setCursor(self._cursor_for_transform_hit(kind))

    def _begin_transform_drag(
        self,
        position: QPointF,
        modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
        view: _TransformView | None = None,
    ) -> str:
        external_view = view is not None
        active_view = view or self._native_transform_view()
        kind = self._transform_hit_test(position, modifiers, active_view)
        if not kind:
            if not external_view:
                self._update_cursor_at(position)
            return ""
        self._transform_drag_kind = kind
        self._transform_drag_start = QPointF(position)
        self._transform_drag_state = self._transform
        self._transform_drag_view = active_view
        polygon = self._control_polygon(active_view)
        self._transform_drag_polygon = tuple(polygon.at(index) for index in range(4))
        self._transform_drag_center = self._polygon_center(self._transform_drag_polygon)
        self._transform_drag_modifiers = modifiers
        self._set_transform_drag_anchor(kind)
        self._transform_drag_started = False
        self.transform_interaction_started.emit()
        if not external_view:
            self.setCursor(self._cursor_for_transform_hit(kind))
        return kind

    def _continue_transform_drag(
        self,
        position: QPointF,
        modifiers: Qt.KeyboardModifier,
    ) -> bool:
        if not self._transform_drag_kind or self._transform_drag_view is None:
            return False
        view_scale = self._transform_drag_view.scale
        start = self._transform_drag_state
        updated = start
        if self._transform_drag_kind == "move":
            delta = position - self._transform_drag_start
            dx = delta.x() / view_scale
            dy = delta.y() / view_scale
            if modifiers & Qt.KeyboardModifier.ShiftModifier:
                if abs(dx) >= abs(dy):
                    dy = 0.0
                else:
                    dx = 0.0
            updated = replace(start, x=start.x + dx, y=start.y + dy)
        elif self._transform_drag_kind.startswith("scale:"):
            updated = self._scaled_drag_state(position, modifiers)
        elif self._transform_drag_kind.startswith("distort:"):
            updated = self._distorted_drag_state(position)
        elif self._transform_drag_kind == "rotate":
            old_angle = math.degrees(
                math.atan2(
                    self._transform_drag_start.y() - self._transform_drag_center.y(),
                    self._transform_drag_start.x() - self._transform_drag_center.x(),
                )
            )
            new_angle = math.degrees(
                math.atan2(
                    position.y() - self._transform_drag_center.y(),
                    position.x() - self._transform_drag_center.x(),
                )
            )
            rotation = start.rotation + new_angle - old_angle
            if modifiers & Qt.KeyboardModifier.ShiftModifier:
                rotation = round(rotation / 15.0) * 15.0
            updated = replace(start, rotation=rotation)
        if updated == self._transform or not self._transform_state_is_valid(updated):
            return False
        if not self._transform_drag_started:
            self._push_undo()
            self._transform_drag_started = True
        self._apply_transform_state(updated)
        return True

    def _scaled_drag_state(
        self,
        position: QPointF,
        modifiers: Qt.KeyboardModifier,
    ) -> _TransformState:
        """按开始时的局部手柄轴缩放，并保持对侧锚点或中心不动。"""
        start = self._transform_drag_state
        name = self._transform_drag_kind.partition(":")[2]
        handles = self._handles_for_polygon(self._transform_drag_polygon)
        start_handle = handles.get(name)
        if start_handle is None:
            return start
        from_center = bool(modifiers & Qt.KeyboardModifier.AltModifier)
        keep_aspect = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        reference = self._transform_drag_center if from_center else self._transform_drag_anchor
        axis = start_handle - reference
        current = position - reference
        denominator = axis.x() * axis.x() + axis.y() * axis.y()
        if denominator <= 1e-9:
            return start
        factor = (current.x() * axis.x() + current.y() * axis.y()) / denominator
        factor = max(self._MIN_SCALE, min(self._MAX_SCALE, factor))

        if name in {"nw", "ne", "se", "sw"} or keep_aspect:
            updated = replace(
                start,
                scale=max(self._MIN_SCALE, min(self._MAX_SCALE, start.scale * factor)),
            )
        elif name in {"e", "w"}:
            updated = replace(
                start,
                stretch_w=max(
                    self._MIN_SCALE,
                    min(self._MAX_SCALE, start.stretch_w * factor),
                ),
            )
        elif name in {"n", "s"}:
            updated = replace(
                start,
                stretch_h=max(
                    self._MIN_SCALE,
                    min(self._MAX_SCALE, start.stretch_h * factor),
                ),
            )
        else:
            return start
        anchor_name = "center" if from_center else self._transform_drag_anchor_name
        return self._state_with_fixed_anchor(updated, anchor_name, reference)

    def _distorted_drag_state(self, position: QPointF) -> _TransformState:
        """把屏幕拖动逆旋转到透视前坐标，并移动一个角或同边两个角。"""
        if self._transform_drag_view is None:
            return self._transform_drag_state
        start = self._transform_drag_state
        name = self._transform_drag_kind.partition(":")[2]
        indices = {
            "nw": (0,),
            "n": (0, 2),
            "ne": (2,),
            "e": (2, 4),
            "se": (4,),
            "s": (4, 6),
            "sw": (6,),
            "w": (6, 0),
        }.get(name)
        if indices is None:
            return start
        delta = (position - self._transform_drag_start) / self._transform_drag_view.scale
        angle = math.radians(-start.rotation)
        local_dx = delta.x() * math.cos(angle) - delta.y() * math.sin(angle)
        local_dy = delta.x() * math.sin(angle) + delta.y() * math.cos(angle)
        values = list(start.distort)
        for index in indices:
            values[index] = max(
                -self._MAX_DISTORT_OFFSET,
                min(self._MAX_DISTORT_OFFSET, values[index] + local_dx),
            )
            values[index + 1] = max(
                -self._MAX_DISTORT_OFFSET,
                min(self._MAX_DISTORT_OFFSET, values[index + 1] + local_dy),
            )
        updated = replace(start, distort=tuple(values))
        return self._state_with_fixed_anchor(
            updated,
            self._transform_drag_anchor_name,
            self._transform_drag_anchor,
        )

    def _state_with_fixed_anchor(
        self,
        state: _TransformState,
        anchor_name: str,
        target: QPointF,
    ) -> _TransformState:
        """通过平移补偿，让候选变换中的指定控制点保持在拖动前位置。"""
        try:
            polygon = self._calculate_transform_geometry(state).polygon
        except (ValueError, cv2.error):
            return self._transform_drag_state
        current = (
            self._polygon_center(polygon)
            if anchor_name == "center"
            else self._point_for_polygon(polygon, anchor_name)
        )
        if current is None:
            return self._transform
        view = self._transform_drag_view or self._native_transform_view()
        current_widget = QPointF(
            view.origin.x() + current.x() * view.scale,
            view.origin.y() + current.y() * view.scale,
        )
        return replace(
            state,
            x=state.x + (target.x() - current_widget.x()) / view.scale,
            y=state.y + (target.y() - current_widget.y()) / view.scale,
        )

    def _finish_transform_drag(self) -> bool:
        """收束当前自由变换拖动，并清除与所属视图绑定的会话状态。"""
        active = bool(self._transform_drag_kind)
        changed = self._transform_drag_started
        self._transform_drag_kind = ""
        self._transform_drag_started = False
        self._transform_drag_view = None
        self._transform_drag_modifiers = Qt.KeyboardModifier.NoModifier
        if active:
            self.transform_interaction_finished.emit(changed)
        return changed

    def _native_transform_view(self) -> _TransformView:
        target = self._canvas_rect()
        return _TransformView(QPointF(target.left(), target.top()), max(self._zoom, 1e-6))

    @staticmethod
    def _validated_transform_view(origin: QPointF, scale: float) -> _TransformView:
        try:
            mapped_origin = QPointF(origin)
            mapped_scale = float(scale)
        except (TypeError, ValueError) as error:
            raise ValueError("自由变换视图坐标无效。") from error
        if (
            not math.isfinite(mapped_origin.x())
            or not math.isfinite(mapped_origin.y())
            or not math.isfinite(mapped_scale)
            or mapped_scale <= 0.0
        ):
            raise ValueError("自由变换视图坐标必须为有限值，比例必须大于零。")
        return _TransformView(mapped_origin, mapped_scale)

    def _set_transform_drag_anchor(self, kind: str) -> None:
        """记录本轮拖动需固定的对角、对边中点或字形中心。"""
        self._transform_drag_anchor_name = ""
        self._transform_drag_anchor = QPointF(self._transform_drag_center)
        if kind in {"move", "rotate"}:
            return
        name = kind.partition(":")[2]
        opposite = {
            "nw": "se",
            "n": "s",
            "ne": "sw",
            "e": "w",
            "se": "nw",
            "s": "n",
            "sw": "ne",
            "w": "e",
        }.get(name, "")
        point = self._point_for_polygon(self._transform_drag_polygon, opposite)
        if point is None:
            return
        self._transform_drag_anchor_name = opposite
        self._transform_drag_anchor = point

    def _apply_transform_state(self, state: _TransformState) -> None:
        self._transform = state
        self._invalidate_render()
        self._set_dirty_from_state()
        self._emit_transform()
        self.update()

    def _bake_transform(self, push_undo: bool) -> None:
        """把当前几何变换写入像素层，并归一变换参数。"""
        if self._image.isNull() or self._transform == _TransformState():
            return
        if push_undo:
            self._push_undo()
        self._image = self.image()
        self._image_origin = QPointF(-self._rendered_origin.x(), -self._rendered_origin.y())
        self._transform = _TransformState()
        self._invalidate_render()
        self._refresh_content_bounds()
        self.sample_ink_color()
        self._set_dirty_from_state()
        self.pixels_changed.emit()
        self._emit_transform()
        self.update()

    def _pick_ink_at(self, position: QPointF) -> None:
        point = self._widget_to_image(position)
        if point is None:
            return
        color = self._image.pixelColor(point)
        if color.alpha() < 32:
            return
        self._brush_ink_coverage = self._color_visual_coverage(color)
        color.setAlpha(255)
        self.set_brush_color(color)

    def _snapshot(self) -> _CanvasState:
        return _CanvasState(self._image.copy(), self._transform, QPointF(self._image_origin))

    def _restore_state(self, state: _CanvasState) -> None:
        pixels_changed = (
            self._image != state.image
            or self._image_origin != state.image_origin
        )
        self._image = state.image.copy()
        self._transform = state.transform
        self._image_origin = QPointF(state.image_origin)
        self._invalidate_render()
        self._refresh_content_bounds()
        self.sample_ink_color()
        self._set_dirty_from_state()
        if pixels_changed:
            self.pixels_changed.emit()
        self._emit_transform()
        self.update()

    def _push_undo(self) -> None:
        self._undo_stack.append(self._snapshot())
        if len(self._undo_stack) > 30:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self._emit_history()

    def _state_matches_saved(self) -> bool:
        return (
            self._transform == self._saved_state.transform
            and self._image_origin == self._saved_state.image_origin
            and self._image == self._saved_state.image
        )

    def _set_dirty_from_state(self) -> None:
        self._set_dirty(not self._state_matches_saved())

    def _invalidate_render(self) -> None:
        self._rendered_image = QImage()
        self._rendered_origin = QPoint()
        self._transformed_content_cache = None

    def _final_image(self) -> QImage:
        if self._rendered_image.isNull():
            self._rendered_image = self._render_final_image()
        return self._rendered_image

    def _transformed_content_bounds(self) -> QRectF:
        bounds = self._transformed_content().alpha_bounds
        if bounds.isEmpty():
            return QRectF(0.0, 0.0, float(self._canvas_size.width()), float(self._canvas_size.height()))
        return QRectF(bounds)

    def _logical_view_bounds(self) -> QRectF:
        """返回适合窗口模式必须完整容纳的田字格逻辑范围。"""
        canvas_width = float(self._canvas_size.width())
        canvas_height = float(self._canvas_size.height())
        margin_x = canvas_width * (self.WORKSPACE_RATIO - 1.0) / 2.0
        margin_y = canvas_height * (self.WORKSPACE_RATIO - 1.0) / 2.0
        workspace = QRectF(
            -margin_x,
            -margin_y,
            canvas_width * self.WORKSPACE_RATIO,
            canvas_height * self.WORKSPACE_RATIO,
        )
        output_size, origin = self._output_geometry()
        output = QRectF(
            -float(origin.x()),
            -float(origin.y()),
            float(output_size.width()),
            float(output_size.height()),
        )
        return workspace.united(output)

    def _output_geometry(self) -> tuple[QSize, QPoint]:
        """返回完整输出尺寸及田字格左上角坐标。"""
        bounds = self._transformed_content_bounds()
        canvas_width = self._canvas_size.width()
        canvas_height = self._canvas_size.height()
        expand_x = max(
            0,
            math.ceil(-bounds.left()),
            math.ceil(bounds.right() - canvas_width),
        )
        expand_y = max(
            0,
            math.ceil(-bounds.top()),
            math.ceil(bounds.bottom() - canvas_height),
        )
        return (
            QSize(canvas_width + expand_x * 2, canvas_height + expand_y * 2),
            QPoint(expand_x, expand_y),
        )

    def _stroke_dirty_rect(
        self,
        start: QPoint,
        end: QPoint,
        stroke_width: float | None = None,
    ) -> QRect:
        """将笔画像素脏区映射为窗口脏矩形。"""
        width = float(self._brush_size) if stroke_width is None else float(stroke_width)
        radius = width / 2.0 + 3.0
        source_rect = QRectF(QPointF(start), QPointF(end)).normalized().adjusted(
            -radius,
            -radius,
            radius,
            radius,
        )
        logical_rect = self._content_transform().mapRect(source_rect)
        target = self._canvas_rect()
        widget_rect = QRectF(
            target.left() + logical_rect.left() * self._zoom,
            target.top() + logical_rect.top() * self._zoom,
            logical_rect.width() * self._zoom,
            logical_rect.height() * self._zoom,
        ).adjusted(-2.0, -2.0, 2.0, 2.0)
        return widget_rect.toAlignedRect().intersected(self.rect())

    @staticmethod
    def _normalized_pressure(pressure: float) -> float:
        try:
            value = float(pressure)
        except (TypeError, ValueError):
            return 1.0
        if not math.isfinite(value):
            return 1.0
        return max(0.0, min(1.0, value))

    def _tablet_pressure(self, event: QTabletEvent) -> float:
        device = event.device()
        if not device.capabilities() & QInputDevice.Capability.Pressure:
            return 1.0
        return self._normalized_pressure(event.pressure())

    def _pressure_width(self, pressure: float) -> float:
        if not self._pressure_enabled:
            return float(self._brush_size)
        normalized = self._normalized_pressure(pressure)
        ratio = self._minimum_pressure_ratio + (1.0 - self._minimum_pressure_ratio) * normalized
        return max(1.0, self._brush_size * ratio)

    def _set_zoom(self, zoom: float) -> None:
        self._zoom = max(0.1, min(8.0, zoom))
        self._emit_zoom()
        self.update()

    def _emit_zoom(self) -> None:
        self.zoom_changed.emit(self.zoom_percent)

    def _emit_transform(self) -> None:
        self.transform_changed.emit(self.transform())

    def _emit_history(self) -> None:
        self.history_changed.emit(self.can_undo, self.can_redo)

    def _set_dirty(self, dirty: bool) -> None:
        if self._dirty == dirty:
            return
        self._dirty = dirty
        self.changed.emit(dirty)

    @staticmethod
    def _normalized_canvas_size(
        canvas_size: QSize | tuple[int, int] | None,
        fallback: QSize,
    ) -> QSize:
        if isinstance(canvas_size, QSize):
            width = canvas_size.width()
            height = canvas_size.height()
        elif isinstance(canvas_size, tuple) and len(canvas_size) == 2:
            width, height = canvas_size
        else:
            return QSize(fallback)
        try:
            width = int(width)
            height = int(height)
        except (TypeError, ValueError):
            return QSize(fallback)
        if width <= 0 or height <= 0:
            return QSize(fallback)
        return QSize(width, height)

    @staticmethod
    def _normalized_opacity(value: float) -> float:
        try:
            opacity = float(value)
        except (TypeError, ValueError):
            return 0.35
        if not math.isfinite(opacity):
            return 0.35
        return max(0.0, min(1.0, opacity))

    def _content_source_rect(self) -> QRect:
        """返回参与几何变换的像素矩形；透明空图仍保留完整图像尺寸。"""
        if self._image.isNull():
            return QRect()
        if self._content_bounds.isEmpty():
            return self._image.rect()
        return self._content_bounds.toAlignedRect().intersected(self._image.rect())

    @classmethod
    def _normalized_distort(
        cls,
        value: Sequence[float] | None,
        fallback: tuple[float, ...],
    ) -> tuple[float, ...]:
        if value is None:
            return tuple(fallback)
        if isinstance(value, (str, bytes)) or len(value) != 8:
            return tuple(fallback)
        normalized: list[float] = []
        for item in value:
            try:
                number = float(item)
            except (TypeError, ValueError):
                return tuple(fallback)
            if not math.isfinite(number):
                return tuple(fallback)
            normalized.append(
                max(-cls._MAX_DISTORT_OFFSET, min(cls._MAX_DISTORT_OFFSET, number))
            )
        return tuple(normalized)

    def _transform_state_is_valid(self, state: _TransformState) -> bool:
        values = (
            state.x,
            state.y,
            state.scale,
            state.rotation,
            state.stretch_w,
            state.stretch_h,
            *state.distort,
        )
        if not all(math.isfinite(value) for value in values):
            return False
        if not (
            self._MIN_SCALE <= state.scale <= self._MAX_SCALE
            and self._MIN_SCALE <= state.stretch_w <= self._MAX_SCALE
            and self._MIN_SCALE <= state.stretch_h <= self._MAX_SCALE
            and len(state.distort) == 8
        ):
            return False
        try:
            self._calculate_transform_geometry(state)
        except (ValueError, cv2.error, OverflowError):
            return False
        return True

    @classmethod
    def _validate_transform_size(cls, width: int, height: int) -> None:
        if (
            width <= 0
            or height <= 0
            or width > cls._MAX_TRANSFORM_DIMENSION
            or height > cls._MAX_TRANSFORM_DIMENSION
            or width * height > cls._MAX_TRANSFORM_PIXELS
        ):
            raise ValueError("变换后的图像尺寸超出安全范围。")

    @classmethod
    def _transform_limits(cls) -> TransformLimits:
        return TransformLimits(
            max_dimension=cls._MAX_TRANSFORM_DIMENSION,
            max_pixels=cls._MAX_TRANSFORM_PIXELS,
        )

    @staticmethod
    def _quad_is_valid(points: np.ndarray) -> bool:
        """拒绝非有限、自交、凹陷和近零面积四边形。"""
        return quad_is_valid(points)

    @staticmethod
    def _map_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        affine = np.asarray(matrix, dtype=np.float64)
        if affine.shape == (2, 3):
            homogeneous = np.column_stack((values, np.ones(len(values), dtype=np.float64)))
            return homogeneous @ affine.T
        if affine.shape == (3, 3):
            homogeneous = np.column_stack((values, np.ones(len(values), dtype=np.float64)))
            mapped = homogeneous @ affine.T
            denominator = mapped[:, 2:3]
            if np.any(np.abs(denominator) <= 1e-12):
                raise ValueError("透视映射分母无效。")
            return mapped[:, :2] / denominator
        raise ValueError("不支持的变换矩阵尺寸。")

    @staticmethod
    def _qimage_to_rgba(image: QImage) -> np.ndarray:
        """把 Qt 的 BGRA 内存复制为 OpenCV 可安全持有的 RGBA 数组。"""
        converted = image.convertToFormat(QImage.Format.Format_RGBA8888)
        height = converted.height()
        stride = converted.bytesPerLine()
        raw = np.frombuffer(
            converted.constBits(),
            dtype=np.uint8,
            count=stride * height,
        ).reshape(height, stride)
        return raw[:, : converted.width() * 4].reshape(height, converted.width(), 4).copy()

    @staticmethod
    def _rgba_to_qimage(pixels: np.ndarray) -> QImage:
        contiguous = np.ascontiguousarray(pixels, dtype=np.uint8)
        if contiguous.ndim != 3 or contiguous.shape[2] != 4:
            raise ValueError("RGBA 像素数组尺寸无效。")
        height, width, _channels = contiguous.shape
        return QImage(
            contiguous.data,
            width,
            height,
            int(contiguous.strides[0]),
            QImage.Format.Format_RGBA8888,
        ).copy()

    @staticmethod
    def _image_alpha_bounds(image: QImage) -> QRectF:
        if image.isNull():
            return QRectF()
        alpha = image.convertToFormat(QImage.Format.Format_Alpha8)
        height = alpha.height()
        width = alpha.width()
        stride = alpha.bytesPerLine()
        pixels = np.frombuffer(
            alpha.constBits(),
            dtype=np.uint8,
            count=stride * height,
        ).reshape(height, stride)[:, :width]
        rows = np.flatnonzero(np.any(pixels, axis=1))
        columns = np.flatnonzero(np.any(pixels, axis=0))
        if rows.size == 0 or columns.size == 0:
            return QRectF()
        return QRectF(
            float(columns[0]),
            float(rows[0]),
            float(columns[-1] - columns[0] + 1),
            float(rows[-1] - rows[0] + 1),
        )

    @staticmethod
    def _empty_polygon() -> tuple[QPointF, QPointF, QPointF, QPointF]:
        return QPointF(), QPointF(), QPointF(), QPointF()

    @staticmethod
    def _polygon_center(points: Sequence[QPointF]) -> QPointF:
        if not points:
            return QPointF()
        return QPointF(
            sum(point.x() for point in points) / len(points),
            sum(point.y() for point in points) / len(points),
        )

    @classmethod
    def _handles_for_polygon(cls, points: Sequence[QPointF]) -> dict[str, QPointF]:
        if len(points) != 4:
            return {}
        nw, ne, se, sw = points
        return {
            "nw": QPointF(nw),
            "ne": QPointF(ne),
            "se": QPointF(se),
            "sw": QPointF(sw),
            "n": (nw + ne) / 2.0,
            "e": (ne + se) / 2.0,
            "s": (se + sw) / 2.0,
            "w": (sw + nw) / 2.0,
        }

    @classmethod
    def _point_for_polygon(
        cls,
        points: Sequence[QPointF],
        name: str,
    ) -> QPointF | None:
        point = cls._handles_for_polygon(points).get(name)
        return QPointF(point) if point is not None else None

    def _refresh_content_bounds(self) -> None:
        if self._image.isNull():
            self._content_bounds = QRectF()
            return
        alpha = self._image.convertToFormat(QImage.Format.Format_Alpha8)
        width = alpha.width()
        height = alpha.height()
        stride = alpha.bytesPerLine()
        pixels = np.frombuffer(
            alpha.constBits(),
            dtype=np.uint8,
            count=stride * height,
        ).reshape(height, stride)[:, :width]
        occupied_rows = np.flatnonzero(np.any(pixels, axis=1))
        occupied_columns = np.flatnonzero(np.any(pixels, axis=0))
        if occupied_rows.size == 0 or occupied_columns.size == 0:
            self._content_bounds = QRectF()
            return
        left = int(occupied_columns[0])
        right = int(occupied_columns[-1])
        top = int(occupied_rows[0])
        bottom = int(occupied_rows[-1])
        self._content_bounds = QRectF(
            float(left),
            float(top),
            float(right - left + 1),
            float(bottom - top + 1),
        )

    @staticmethod
    def _dominant_ink_color(image: QImage) -> QColor:
        if image.isNull():
            return QColor(0, 0, 0, 255)
        width = image.width()
        height = image.height()
        sample_limit = 65_536
        step = max(1, math.ceil(math.sqrt(width * height / sample_limit)))
        converted = image.convertToFormat(QImage.Format.Format_RGBA8888)
        stride = converted.bytesPerLine()
        pixels = np.frombuffer(
            converted.constBits(),
            dtype=np.uint8,
            count=stride * height,
        ).reshape(height, stride)[:, : width * 4].reshape(height, width, 4)
        sampled = pixels[::step, ::step].reshape(-1, 4)
        visible = sampled[sampled[:, 3] >= 32, :3]
        if visible.size == 0:
            return QColor(0, 0, 0, 255)

        distance = visible.astype(np.int32) - 255
        foreground = visible[
            np.sum(distance * distance, axis=1) >= 24 * 24
        ]
        candidates = foreground if foreground.size else visible
        packed = (
            candidates[:, 0].astype(np.uint32) << 16
            | candidates[:, 1].astype(np.uint32) << 8
            | candidates[:, 2].astype(np.uint32)
        )
        colors, first_indices, counts = np.unique(
            packed,
            return_index=True,
            return_counts=True,
        )
        highest_count = int(counts.max())
        tied = np.flatnonzero(counts == highest_count)
        selected = int(tied[np.argmin(first_indices[tied])])
        color = int(colors[selected])
        red = (color >> 16) & 0xFF
        green = (color >> 8) & 0xFF
        blue = color & 0xFF
        return QColor(red, green, blue, 255)

    @classmethod
    def _dominant_ink_coverage(cls, image: QImage) -> int:
        """返回全图主体笔画的视觉墨量，供远离原笔画时安全回退。"""

        if image.isNull():
            return 255
        pixels = cls._qimage_to_rgba(image)
        height, width = pixels.shape[:2]
        sample_limit = 65_536
        step = max(1, math.ceil(math.sqrt(width * height / sample_limit)))
        coverage = cls._visual_coverage(pixels)[::step, ::step]
        core = coverage[coverage >= 16]
        values = core if core.size else coverage[coverage > 0]
        if not values.size:
            return 255
        return max(1, min(255, int(round(float(np.percentile(values, 70))))))

    @classmethod
    def _local_ink_coverage(
        cls,
        image: QImage,
        point: QPoint,
        width: float,
        fallback: int,
    ) -> int:
        """取样落笔点附近原笔画；空白区域回退到当前字形主体墨量。"""

        if image.isNull():
            return max(1, min(255, int(fallback)))
        radius = max(5, int(math.ceil(float(width) * 1.25)))
        region = QRect(
            point.x() - radius,
            point.y() - radius,
            radius * 2 + 1,
            radius * 2 + 1,
        ).intersected(image.rect())
        if region.isEmpty():
            return max(1, min(255, int(fallback)))
        coverage = cls._visual_coverage(cls._qimage_to_rgba(image.copy(region)))
        core = coverage[coverage >= 16]
        values = core if core.size >= 8 else coverage[coverage > 0]
        if values.size < 3:
            return max(1, min(255, int(fallback)))
        sampled = int(round(float(np.percentile(values, 70))))
        return max(1, min(255, sampled))

    @staticmethod
    def _color_visual_coverage(color: QColor) -> int:
        luminance = color.red() * 0.299 + color.green() * 0.587 + color.blue() * 0.114
        coverage = color.alpha() * (255.0 - luminance) / 255.0
        return max(1, min(255, int(round(coverage))))

    @staticmethod
    def _visual_coverage(pixels: np.ndarray) -> np.ndarray:
        values = np.asarray(pixels, dtype=np.float32)
        if values.ndim != 3 or values.shape[2] != 4:
            raise ValueError("视觉墨量输入必须是 RGBA 像素数组")
        luminance = (
            values[..., 0] * 0.299
            + values[..., 1] * 0.587
            + values[..., 2] * 0.114
        )
        coverage = values[..., 3] * (255.0 - luminance) / 255.0
        return np.clip(np.rint(coverage), 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def _finite_value(value: float | None, fallback: float) -> float:
        if value is None:
            return fallback
        try:
            result = float(value)
        except (TypeError, ValueError):
            return fallback
        return result if math.isfinite(result) else fallback

    @staticmethod
    def _distance(first: QPointF, second: QPointF) -> float:
        return math.hypot(first.x() - second.x(), first.y() - second.y())
