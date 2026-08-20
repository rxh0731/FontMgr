"""字库整体协调工作台。"""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from copy import deepcopy
from threading import Event, Lock
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image
from PySide6.QtCore import (
    QEvent,
    QObject,
    QPoint,
    QPointF,
    QRectF,
    QSignalBlocker,
    QSize,
    QRunnable,
    QThreadPool,
    Qt,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QCursor,
    QIcon,
    QImage,
    QKeyEvent,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPen,
    QPolygonF,
    QPixmap,
    QTransform,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

import config
from services.adjustment_service import AdjustmentService, CoordinationCancelled
from services.glyph_service import GlyphService
from services.workflow_status_service import (
    COORDINATION_STATUS_FILTERS,
    INK_STATUS_ACHIEVED,
    INK_STATUS_EXCEPTION,
    INK_STATUS_PENDING,
    MARKER_FILE_ERROR,
    MARKER_INK_EXCEPTION,
    MARKER_INK_PENDING,
    MARKER_STRUCTURE_REVIEW,
    MARKER_UNSAVED,
    PHASE_FILTER_ALL,
    PHASE_COORDINATION,
    PHASE_STATUS_COLORS,
    STAGE_PENDING_COORDINATION,
    STATUS_COORDINATED,
    WorkflowStageProjection,
    WorkflowStatus,
    project_stage_status,
    resolve_safe_stage_file,
)
from ui.workers import FunctionWorker, log_background_exception
from ui.widgets.adjustable_tree_columns import AdjustableTreeColumns
from ui.widgets.glyph_rename_dialog import run_glyph_rename_dialog
from ui.widgets.export_gallery import decode_thumbnail_image
from ui.widgets.review_canvas import ReviewCanvas
from ui.widgets.two_line_status_delegate import (
    TwoLineStatusDelegate,
    set_two_line_status,
)
from utils.file_utils import natural_key, pinyin_natural_key


class _BaselineAnalysisSignals(QObject):
    """大字库协调基准后台分析信号。"""

    progress = Signal(int, int, str)
    finished = Signal(object)
    failed = Signal(str)


class _BaselineAnalysisTask(QRunnable):
    """从字库状态快照统计协调基准，避免阻塞 GUI 线程。"""

    def __init__(
        self,
        ziku_name: str,
        ziku_dir: str,
        glyph_snapshot: dict[str, Any],
    ) -> None:
        super().__init__()
        self._ziku_name = ziku_name
        self._ziku_dir = ziku_dir
        self._glyph_snapshot = glyph_snapshot
        self.signals = _BaselineAnalysisSignals()

    @Slot()
    def run(self) -> None:
        try:
            worker_glyph = GlyphService.open(self._ziku_name, self._ziku_dir)
            worker_glyph.restore_state(self._glyph_snapshot)
            baseline = AdjustmentService(worker_glyph).analyze(
                progress_callback=self.signals.progress.emit,
            )
        except Exception as exc:
            log_background_exception("整体协调基准分析")
            try:
                self.signals.failed.emit(str(exc))
            except RuntimeError:
                pass
        else:
            try:
                self.signals.finished.emit(baseline)
            except RuntimeError:
                pass


class _CoordinationTaskSignals(QObject):
    """整体协调批量任务信号。"""

    progress = Signal(str, int, int, int, str)
    finished = Signal(object)
    failed = Signal(str)


class _CoordinationTask(QRunnable):
    """在线程池中执行完整的整体协调批次事务。"""

    def __init__(
        self,
        ziku_name: str,
        ziku_dir: str,
        glyph_snapshot: dict[str, Any],
        variant_ids: list[str],
        adjustments: dict[str, dict[str, Any]],
        ink_config: dict[str, Any],
        coordination_baseline: dict[str, Any],
    ) -> None:
        super().__init__()
        self._ziku_name = ziku_name
        self._ziku_dir = ziku_dir
        self._glyph_snapshot = glyph_snapshot
        self._variant_ids = variant_ids
        self._adjustments = adjustments
        self._ink_config = ink_config
        self._coordination_baseline = coordination_baseline
        self._cancel_event = Event()
        self._commit_lock = Lock()
        self._commit_started = False
        self.signals = _CoordinationTaskSignals()

    def request_cancel(self) -> bool:
        """在提交门控关闭前请求停止，返回请求是否被受理。"""
        with self._commit_lock:
            if self._commit_started:
                return False
            self._cancel_event.set()
            return True

    def is_cancel_requested(self) -> bool:
        return self._cancel_event.is_set()

    def is_commit_started(self) -> bool:
        with self._commit_lock:
            return self._commit_started

    def try_begin_commit(self) -> bool:
        """与停止请求互斥地关闭取消窗口。"""
        with self._commit_lock:
            if self._cancel_event.is_set():
                return False
            self._commit_started = True
            return True

    @Slot()
    def run(self) -> None:
        try:
            worker_glyph = GlyphService.open(self._ziku_name, self._ziku_dir)
            worker_glyph.restore_state(self._glyph_snapshot)
            variants: list[dict[str, Any]] = []
            for variant_id in self._variant_ids:
                detail = worker_glyph.get_variant(variant_id)
                if not detail:
                    raise KeyError(f"找不到待协调字形：{variant_id}")
                variants.append(detail)
            result = AdjustmentService(worker_glyph).save_coordinated_variants(
                variants,
                self._adjustments,
                self._ink_config,
                self._coordination_baseline,
                progress_callback=self.signals.progress.emit,
                cancel_check=self.is_cancel_requested,
                commit_gate=self.try_begin_commit,
            )
        except CoordinationCancelled:
            try:
                self.signals.finished.emit(
                    {
                        "已停止": True,
                        "结果": {
                            "已停止": True,
                            "成功": 0,
                            "失败": 0,
                            "失败详情": [],
                        },
                    }
                )
            except RuntimeError:
                pass
        except Exception as exc:
            log_background_exception("整库整体协调")
            try:
                self.signals.failed.emit(str(exc))
            except RuntimeError:
                pass
        else:
            payload: dict[str, Any] = {"结果": result}
            try:
                payload["字库状态"] = worker_glyph.snapshot_state()
            except Exception as exc:
                payload["已提交"] = True
                payload["字库状态错误"] = str(exc) or type(exc).__name__
            try:
                self.signals.finished.emit(payload)
            except RuntimeError:
                pass


class GlyphPreviewCard(QWidget):
    """使用精调画布四边形并提供即时投影反馈的比较卡片。"""

    selected = Signal(str)
    edit_requested = Signal(str)
    transform_started = Signal(str, object, object)
    transform_changed = Signal(str, object, object)
    transform_finished = Signal(str)
    wheel_requested = Signal(str, float)

    FOOTER_HEIGHT = 36
    WORK_RATIO = 1.3
    HANDLE_RADIUS = 5.0
    HANDLE_HIT_RADIUS = 10.0
    ROTATE_HANDLE_DISTANCE = 24.0

    def __init__(
        self,
        variant_id: str,
        canvas_size: QSize | tuple[int, int] | None = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.variant_id = variant_id
        self._pixmap = QPixmap()
        self._selected = False
        self._char = ""
        self._filename = ""
        self._status = "待协调"
        self._preview_size = QSize()
        self._preview_bounds = (0.0, 0.0, 0.0, 0.0)
        self._canvas_size = self._normalized_canvas_size(canvas_size)
        self._transform: dict[str, Any] = {
            "x": 0.0,
            "y": 0.0,
            "scale": 1.0,
            "rotation": 0.0,
            "stretch_w": 1.0,
            "stretch_h": 1.0,
            "distort": [0.0] * 8,
        }
        self._control_polygon_logical = QPolygonF()
        self._live_source_polygon_logical = QPolygonF()
        self._live_control_polygon_logical = QPolygonF()
        self._live_preview_active = False
        self._drag_kind = ""
        self._rotation_cursor: QCursor | None = None
        self._hit_test_handler: Callable[
            [QPointF, Qt.KeyboardModifier],
            str,
        ] | None = None
        self._cursor_handler: Callable[[str], QCursor] | None = None
        self.setMinimumSize(112, 132)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

    @staticmethod
    def _normalized_canvas_size(
        canvas_size: QSize | tuple[int, int] | None,
    ) -> QSize:
        if isinstance(canvas_size, QSize):
            return QSize(max(1, canvas_size.width()), max(1, canvas_size.height()))
        if isinstance(canvas_size, tuple) and len(canvas_size) == 2:
            try:
                return QSize(max(1, int(canvas_size[0])), max(1, int(canvas_size[1])))
            except (TypeError, ValueError):
                pass
        return QSize(250, 250)

    def set_preview(
        self,
        image: QImage,
        bounds: tuple[int, int, int, int] | None = None,
    ) -> None:
        """设置最终质量预览，并结束上一轮临时投影。"""
        self._pixmap = QPixmap.fromImage(image)
        self._preview_size = image.size()
        if bounds is None:
            self._preview_bounds = (
                0.0,
                0.0,
                float(max(0, image.width())),
                float(max(0, image.height())),
            )
        else:
            self._preview_bounds = tuple(float(value) for value in bounds)
        self.finish_live_preview()
        self.update()

    def set_transform(self, transform: dict[str, Any]) -> None:
        for key, default in (
            ("x", 0.0),
            ("y", 0.0),
            ("scale", 1.0),
            ("rotation", 0.0),
            ("stretch_w", 1.0),
            ("stretch_h", 1.0),
        ):
            try:
                value = float(transform.get(key, default))
            except (TypeError, ValueError):
                value = default
            self._transform[key] = value if math.isfinite(value) else default
        distort = transform.get("distort", [0.0] * 8)
        if isinstance(distort, (list, tuple)) and len(distort) == 8:
            normalized: list[float] = []
            for value in distort:
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    number = 0.0
                normalized.append(number if math.isfinite(number) else 0.0)
            self._transform["distort"] = normalized
        else:
            self._transform["distort"] = [0.0] * 8
        self.update()

    def set_control_polygon(self, polygon: QPolygonF | None) -> None:
        """保存标准田字格逻辑坐标中的精调控制四边形。"""
        self._control_polygon_logical = self._copy_quad(polygon)
        self._live_source_polygon_logical = QPolygonF()
        self._live_control_polygon_logical = QPolygonF()
        self._live_preview_active = False
        self.update()

    def begin_live_preview(self) -> bool:
        if self._live_preview_active:
            return True
        if self._pixmap.isNull() or self._control_polygon_logical.count() != 4:
            return False
        self._live_source_polygon_logical = QPolygonF(
            self._control_polygon_logical
        )
        self._live_control_polygon_logical = QPolygonF(
            self._control_polygon_logical
        )
        self._live_preview_active = True
        self.update()
        return True

    def set_live_control_polygon(self, polygon: QPolygonF | None) -> None:
        if not self._live_preview_active:
            return
        normalized = self._copy_quad(polygon)
        if normalized.count() != 4:
            return
        self._live_control_polygon_logical = normalized
        self.update()

    def finish_live_preview(self) -> None:
        self._live_preview_active = False
        self._live_source_polygon_logical = QPolygonF()
        self._live_control_polygon_logical = QPolygonF()
        self.update()

    @staticmethod
    def _copy_quad(polygon: QPolygonF | None) -> QPolygonF:
        if not isinstance(polygon, QPolygonF) or polygon.count() != 4:
            return QPolygonF()
        return QPolygonF(polygon)

    def set_metadata(self, char: str, filename: str, status: str) -> None:
        self._char = char
        self._filename = filename
        self._status = status
        self.setToolTip(f"{char} · {filename}\n{status}")
        self.update()

    def set_selected(self, selected: bool) -> None:
        self._selected = bool(selected)
        if not self._selected and not self._drag_kind:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.update()

    def set_transform_interaction_handlers(
        self,
        hit_test: Callable[[QPointF, Qt.KeyboardModifier], str],
        cursor_for_hit: Callable[[str], QCursor],
    ) -> None:
        """使用精调画布的公开命中与光标规则。"""
        self._hit_test_handler = hit_test
        self._cursor_handler = cursor_for_hit

    def paintEvent(self, _event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        outer = self.rect().adjusted(1, 1, -2, -2)
        painter.fillRect(outer, QColor("#202630"))

        image_area = self._image_area()
        painter.fillRect(image_area, QColor("#EEF0F3"))
        painter.save()
        painter.setClipRect(image_area)
        self._draw_grid(painter)
        self._draw_preview(painter)
        if self._selected:
            self._draw_transform_controls(painter)
        painter.restore()

        footer_top = outer.bottom() - self.FOOTER_HEIGHT + 1
        footer = QRectF(
            float(outer.left()),
            float(footer_top),
            float(outer.width()),
            float(self.FOOTER_HEIGHT),
        )
        painter.fillRect(footer, QColor("#202630"))
        painter.setPen(QColor("#F1F4F8"))
        char_rect = footer.adjusted(9, 0, -9, 0)
        painter.drawText(
            char_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            self._char,
        )
        filename_left = char_rect.left() + max(
            24,
            painter.fontMetrics().horizontalAdvance(self._char) + 14,
        )
        filename_rect = QRectF(
            filename_left,
            char_rect.top(),
            max(1.0, char_rect.right() - filename_left - 16.0),
            char_rect.height(),
        )
        painter.setPen(QColor("#A6B0BE"))
        filename = painter.fontMetrics().elidedText(
            self._filename,
            Qt.TextElideMode.ElideMiddle,
            max(1, int(filename_rect.width())),
        )
        painter.drawText(
            filename_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            filename,
        )
        status_color = QColor(
            PHASE_STATUS_COLORS.get(self._status, "#A6B0BE")
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(status_color)
        painter.drawEllipse(
            footer.right() - 13.0,
            footer.center().y() - 4.0,
            8.0,
            8.0,
        )

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(
            QPen(
                QColor("#4DA3FF") if self._selected else QColor("#505A68"),
                3 if self._selected else 1,
            )
        )
        painter.drawRoundedRect(QRectF(outer), 4.0, 4.0)

    def _image_area(self) -> QRectF:
        outer = self.rect().adjusted(1, 1, -2, -2)
        return QRectF(
            float(outer.left()),
            float(outer.top()),
            float(outer.width()),
            float(max(1, outer.height() - self.FOOTER_HEIGHT)),
        )

    def _work_rect(self) -> QRectF:
        image_area = self._image_area()
        available_width = max(1.0, image_area.width() - 12.0)
        available_height = max(1.0, image_area.height() - 12.0)
        if self._preview_size.isValid():
            source_width = max(1.0, float(self._preview_size.width()))
            source_height = max(1.0, float(self._preview_size.height()))
        else:
            source_width = self._canvas_size.width() * self.WORK_RATIO
            source_height = self._canvas_size.height() * self.WORK_RATIO
        scale = max(
            1e-6,
            min(available_width / source_width, available_height / source_height),
        )
        width = source_width * scale
        height = source_height * scale
        return QRectF(
            image_area.center().x() - width / 2.0,
            image_area.center().y() - height / 2.0,
            width,
            height,
        )

    def transform_view(self) -> tuple[QPointF, float]:
        """返回外部自由变换所需的田字格原点和统一显示比例。"""
        work = self._work_rect()
        preview_width = max(
            1,
            self._preview_size.width()
            if self._preview_size.isValid()
            else round(self._canvas_size.width() * self.WORK_RATIO),
        )
        preview_height = max(
            1,
            self._preview_size.height()
            if self._preview_size.isValid()
            else round(self._canvas_size.height() * self.WORK_RATIO),
        )
        scale = min(
            work.width() / preview_width,
            work.height() / preview_height,
        )
        origin = QPointF(
            work.left()
            + (preview_width - self._canvas_size.width()) * scale / 2.0,
            work.top()
            + (preview_height - self._canvas_size.height()) * scale / 2.0,
        )
        return origin, max(scale, 1e-6)

    def _grid_rect(self) -> QRectF:
        origin, scale = self.transform_view()
        return QRectF(
            origin.x(),
            origin.y(),
            self._canvas_size.width() * scale,
            self._canvas_size.height() * scale,
        )

    def _draw_grid(self, painter: QPainter) -> None:
        grid = self._grid_rect()
        grid_color = QColor("#D9A3A3")
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(grid_color, 1, Qt.PenStyle.SolidLine))
        painter.drawRect(grid)
        painter.setPen(QPen(grid_color, 1, Qt.PenStyle.DashLine))
        center = grid.center()
        painter.drawLine(grid.left(), center.y(), grid.right(), center.y())
        painter.drawLine(center.x(), grid.top(), center.x(), grid.bottom())
        painter.drawLine(grid.topLeft(), grid.bottomRight())
        painter.drawLine(grid.topRight(), grid.bottomLeft())

    def _draw_preview(self, painter: QPainter) -> None:
        if self._pixmap.isNull():
            return
        work = self._work_rect()
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        if self._live_preview_active:
            source = self._mapped_polygon(self._live_source_polygon_logical)
            target = self._mapped_polygon(self._live_control_polygon_logical)
            projection = QTransform()
            if (
                source.count() == 4
                and target.count() == 4
                and QTransform.quadToQuad(source, target, projection)
            ):
                painter.setWorldTransform(projection, True)
        painter.drawPixmap(work, self._pixmap, QRectF(self._pixmap.rect()))
        painter.restore()

    def _mapped_polygon(self, logical: QPolygonF) -> QPolygonF:
        if logical.count() != 4:
            return QPolygonF()
        origin, scale = self.transform_view()
        polygon = QPolygonF()
        for index in range(logical.count()):
            point = logical.at(index)
            polygon.append(
                QPointF(
                    origin.x() + point.x() * scale,
                    origin.y() + point.y() * scale,
                )
            )
        return polygon

    def _control_polygon(self) -> QPolygonF:
        logical = (
            self._live_control_polygon_logical
            if self._live_preview_active
            else self._control_polygon_logical
        )
        return self._mapped_polygon(logical)

    def _control_rect(self) -> QRectF:
        """兼容旧测试和调用方，真实控制层仍使用四边形。"""
        return self._control_polygon().boundingRect()

    @staticmethod
    def _handles(control: QPolygonF | QRectF) -> dict[str, QPointF]:
        if isinstance(control, QRectF):
            corners = [
                control.topLeft(),
                control.topRight(),
                control.bottomRight(),
                control.bottomLeft(),
            ]
        elif control.count() == 4:
            corners = [control.at(index) for index in range(4)]
        else:
            return {}
        return {
            "nw": corners[0],
            "ne": corners[1],
            "se": corners[2],
            "sw": corners[3],
            "n": (corners[0] + corners[1]) / 2.0,
            "e": (corners[1] + corners[2]) / 2.0,
            "s": (corners[2] + corners[3]) / 2.0,
            "w": (corners[3] + corners[0]) / 2.0,
        }

    def _control_handles(self) -> tuple[dict[str, QPointF], QPointF]:
        polygon = self._control_polygon()
        handles = self._handles(polygon)
        if not handles:
            return {}, QPointF()
        corners = [polygon.at(index) for index in range(4)]
        center = QPointF(
            sum(point.x() for point in corners) / 4.0,
            sum(point.y() for point in corners) / 4.0,
        )
        top = handles["n"]
        vector = top - center
        length = max(1.0, math.hypot(vector.x(), vector.y()))
        rotate = top + vector * (self.ROTATE_HANDLE_DISTANCE / length)
        return handles, rotate

    def _draw_transform_controls(self, painter: QPainter) -> None:
        polygon = self._control_polygon()
        if polygon.count() != 4:
            return
        handles, rotate = self._control_handles()
        pen = QPen(QColor("#2776C7"), 1.5)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPolygon(polygon)
        painter.drawLine(handles["n"], rotate)
        painter.setPen(QPen(QColor("#174F86"), 1.0))
        painter.setBrush(QColor("#F7FBFF"))
        radius = self.HANDLE_RADIUS
        for point in handles.values():
            painter.drawRect(
                QRectF(
                    point.x() - radius,
                    point.y() - radius,
                    radius * 2.0,
                    radius * 2.0,
                )
            )
        painter.drawEllipse(rotate, radius, radius)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            was_selected = self._selected
            self.selected.emit(self.variant_id)
            if was_selected:
                kind = self._hit_test(event.position(), event.modifiers())
                if kind:
                    self._drag_kind = kind
                    self.begin_live_preview()
                    self.transform_started.emit(
                        self.variant_id,
                        QPointF(event.position()),
                        event.modifiers(),
                    )
                    self.setCursor(self._cursor_object_for_hit(kind))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        position = event.position()
        if self._drag_kind and event.buttons() & Qt.MouseButton.LeftButton:
            self.transform_changed.emit(
                self.variant_id,
                QPointF(position),
                event.modifiers(),
            )
            event.accept()
            return
        kind = self._hit_test(position, event.modifiers())
        self.setCursor(self._cursor_object_for_hit(kind))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._drag_kind:
            self.cancel_transform_interaction(notify=True)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if not self._selected:
            super().wheelEvent(event)
            return
        delta = float(event.angleDelta().y())
        if delta == 0.0:
            delta = float(event.pixelDelta().y() * 8)
        if delta == 0.0:
            super().wheelEvent(event)
            return
        self.wheel_requested.emit(self.variant_id, delta)
        event.accept()

    def leaveEvent(self, event: Any) -> None:
        if not self._drag_kind:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        super().leaveEvent(event)

    def event(self, event: QEvent) -> bool:
        if (
            hasattr(self, "_drag_kind")
            and self._drag_kind
            and event.type()
            in {
                QEvent.Type.FocusOut,
                QEvent.Type.Hide,
                QEvent.Type.UngrabMouse,
                QEvent.Type.WindowDeactivate,
            }
        ):
            self.cancel_transform_interaction(notify=True)
        return super().event(event)

    def cancel_transform_interaction(self, *, notify: bool = False) -> None:
        """清理被释放或系统中断的卡片变换状态。"""
        was_dragging = bool(self._drag_kind)
        self._drag_kind = ""
        if notify and was_dragging:
            self.transform_finished.emit(self.variant_id)
        self.finish_live_preview()
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def _hit_test(
        self,
        position: QPointF,
        modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
    ) -> str:
        if not self._selected or self._drag_kind:
            return self._drag_kind
        if self._hit_test_handler is not None:
            kind = self._hit_test_handler(QPointF(position), modifiers)
            return kind.partition(":")[2] if kind.startswith("scale:") else kind
        polygon = self._control_polygon()
        if polygon.count() != 4:
            return ""
        handles, rotate = self._control_handles()
        if self._distance(position, rotate) <= self.HANDLE_HIT_RADIUS:
            return "rotate"
        nearest = ""
        nearest_distance = float("inf")
        for name, point in handles.items():
            distance = self._distance(position, point)
            if distance <= self.HANDLE_HIT_RADIUS and distance < nearest_distance:
                nearest = name
                nearest_distance = distance
        if nearest:
            if modifiers & Qt.KeyboardModifier.ControlModifier:
                return f"distort:{nearest}"
            return nearest
        return (
            "move"
            if polygon.containsPoint(position, Qt.FillRule.OddEvenFill)
            else ""
        )

    @staticmethod
    def _distance(first: QPointF, second: QPointF) -> float:
        return math.hypot(first.x() - second.x(), first.y() - second.y())

    def _cursor_for_hit(self, kind: str) -> Qt.CursorShape:
        if kind == "move":
            return Qt.CursorShape.SizeAllCursor
        if kind == "rotate":
            return Qt.CursorShape.CrossCursor
        if kind.startswith("distort:"):
            name = kind.partition(":")[2]
            return (
                Qt.CursorShape.CrossCursor
                if name in {"nw", "ne", "se", "sw"}
                else Qt.CursorShape.SizeAllCursor
            )
        name = kind.partition(":")[2] if kind.startswith("scale:") else kind
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
            return Qt.CursorShape.PointingHandCursor
        angle = (base_angles[name] + float(self._transform["rotation"])) % 180.0
        direction = int(round(angle / 45.0)) % 4
        return {
            0: Qt.CursorShape.SizeHorCursor,
            1: Qt.CursorShape.SizeFDiagCursor,
            2: Qt.CursorShape.SizeVerCursor,
            3: Qt.CursorShape.SizeBDiagCursor,
        }[direction]

    def _cursor_object_for_hit(self, kind: str) -> QCursor:
        if self._cursor_handler is not None:
            normalized = (
                f"scale:{kind}"
                if kind in {"nw", "n", "ne", "e", "se", "s", "sw", "w"}
                else kind
            )
            return self._cursor_handler(normalized)
        if kind == "rotate":
            return self._rotation_handle_cursor()
        return QCursor(self._cursor_for_hit(kind))

    def _rotation_handle_cursor(self) -> QCursor:
        if self._rotation_cursor is not None:
            return self._rotation_cursor
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(
            QPen(
                QColor("#111111"),
                4,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
            )
        )
        painter.drawArc(QRectF(6, 6, 20, 20), 35 * 16, 275 * 16)
        painter.setPen(
            QPen(
                QColor("#F7FBFF"),
                2,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
            )
        )
        painter.drawArc(QRectF(6, 6, 20, 20), 35 * 16, 275 * 16)
        painter.setPen(
            QPen(
                QColor("#111111"),
                3,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
            )
        )
        painter.drawLine(QPointF(22, 4), QPointF(27, 8))
        painter.drawLine(QPointF(27, 8), QPointF(21, 10))
        painter.setPen(
            QPen(
                QColor("#F7FBFF"),
                1,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
            )
        )
        painter.drawLine(QPointF(22, 4), QPointF(27, 8))
        painter.drawLine(QPointF(27, 8), QPointF(21, 10))
        painter.end()
        self._rotation_cursor = QCursor(pixmap, 16, 16)
        return self._rotation_cursor

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.cancel_transform_interaction(notify=True)
            self.selected.emit(self.variant_id)
            self.edit_requested.emit(self.variant_id)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.edit_requested.emit(self.variant_id)
            event.accept()
            return
        super().keyPressEvent(event)


class ConsistencyPage(QWidget):
    """通过整库比较和单字精调统一字形大小、中心与形态。"""

    summary_changed = Signal(object)

    WORK_RATIO = 1.3
    BACKGROUND_ANALYSIS_THRESHOLD = 32
    LARGE_COLUMNS = 5
    LARGE_ROWS = 3
    COMPACT_COLUMNS = 4
    COMPACT_ROWS = 2
    LARGE_VIEWPORT_WIDTH = 760
    LARGE_VIEWPORT_HEIGHT = 520
    LIST_PANEL_MIN_WIDTH = 250
    LIST_PANEL_DEFAULT_WIDTH = 285
    LIST_PANEL_MAX_WIDTH = 400
    TOOL_PANEL_MIN_WIDTH = 218
    TOOL_PANEL_DEFAULT_WIDTH = 278
    TOOL_PANEL_MAX_WIDTH = 330
    TRANSFORM_PERCENT_MIN = 5
    TRANSFORM_PERCENT_MAX = 500
    TRANSFORM_PERCENT_SLIDER_EXTENT = TRANSFORM_PERCENT_MAX - 100
    TRANSFORM_OFFSET_LIMIT = 8192
    LIST_THUMBNAIL_SIZE = 38
    LIST_THUMBNAIL_PREFETCH_ITEMS = 4
    LIST_THUMBNAIL_MAX_REQUESTS = 24
    LIST_THUMBNAIL_CACHE_ITEMS = 512
    LIST_THUMBNAIL_DECODE_LIMIT_BYTES = 32 * 1024 * 1024

    STATUS_FILTERS = COORDINATION_STATUS_FILTERS
    SORT_OPTIONS = ("拼音顺序", "文件名顺序", "导入顺序")
    INK_MODES = ("跟随全库", "保留本字", "人工例外")

    def __init__(
        self,
        glyph_service: GlyphService,
        on_back: Callable[[], None],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._glyph = glyph_service
        self._adjustment_service = AdjustmentService(glyph_service)
        self._on_back = on_back
        saved_summary = self._glyph.get_coordination_summary()
        self._workflow_summary = saved_summary
        self._finished_dir = self._glyph.get_workflow_dirs()["成品"]
        initial_variants = self._adjustment_service.load_reviewed_variants(
            pinyin_order=False
        )
        has_saved_baseline = self._has_current_saved_ink_baseline(saved_summary)
        self._baseline_analysis_pending = (
            not has_saved_baseline
            and len(initial_variants) >= self.BACKGROUND_ANALYSIS_THRESHOLD
        )
        if has_saved_baseline or self._baseline_analysis_pending:
            self._coordination_baseline = self._saved_or_default_baseline(saved_summary)
        else:
            self._coordination_baseline = self._adjustment_service.analyze()
        self._ink_baseline = float(
            self._coordination_baseline.get("墨色基准", 220.0)
        )
        saved_ink_enabled = saved_summary.get("墨色统一启用", True)
        self._initial_ink_enabled = saved_ink_enabled if isinstance(saved_ink_enabled, bool) else True

        metadata = self._glyph.get_metadata()
        self._canvas_width = self._positive_int(metadata.get("画布宽"), 250)
        self._canvas_height = self._positive_int(metadata.get("画布高"), 250)
        self._list_variants: list[dict[str, Any]] = []
        self._list_visible_variants: list[dict[str, Any]] = []
        self._list_variant_by_id: dict[str, dict[str, Any]] = {}
        self._all_variants: list[dict[str, Any]] = []
        self._variants: list[dict[str, Any]] = []
        self._variant_by_id: dict[str, dict[str, Any]] = {}
        self._workflow_status_cache: dict[
            str,
            tuple[bool, WorkflowStageProjection],
        ] = {}
        self._import_order: dict[str, int] = {}
        self._adjustments: dict[str, dict[str, Any]] = {}
        self._saved_adjustments: dict[str, dict[str, Any]] = {}
        self._saved_signatures: dict[str, tuple[tuple[str, Any], ...]] = {}
        self._saved_ink_signatures: dict[str, tuple[Any, ...]] = {}
        self._ink_modes: dict[str, str] = {}
        self._content_sizes: dict[str, tuple[int, int]] = {}
        self._preview_cache: OrderedDict[tuple[Any, ...], QImage] = OrderedDict()
        self._preview_bounds_cache: dict[
            tuple[Any, ...],
            tuple[int, int, int, int],
        ] = {}
        self._preview_cache_bytes = 0
        self._preview_cache_max_bytes = 192 * 1024 * 1024
        self._preview_cache_max_items = 96
        self._preview_workers: dict[str, tuple[tuple[Any, ...], FunctionWorker]] = {}
        self._preview_pool = QThreadPool(self)
        self._preview_pool.setMaxThreadCount(2)
        self._preview_pool.setExpiryTimeout(15_000)
        self._list_thumbnail_cache: OrderedDict[
            tuple[Any, ...],
            QIcon,
        ] = OrderedDict()
        self._list_thumbnail_key_by_variant: dict[str, tuple[Any, ...]] = {}
        self._list_items_by_id: dict[str, QTreeWidgetItem] = {}
        self._list_thumbnail_workers: dict[
            str,
            tuple[str, tuple[Any, ...], FunctionWorker],
        ] = {}
        self._list_thumbnail_failures: set[tuple[Any, ...]] = set()
        self._list_thumbnail_pool = QThreadPool(self)
        self._list_thumbnail_pool.setMaxThreadCount(2)
        self._list_thumbnail_pool.setExpiryTimeout(15_000)
        self._list_thumbnail_timer = QTimer(self)
        self._list_thumbnail_timer.setSingleShot(True)
        self._list_thumbnail_timer.setInterval(35)
        self._list_thumbnail_timer.timeout.connect(self._load_visible_list_thumbnails)
        self._list_placeholder_icon = QIcon()
        self._cards: dict[str, GlyphPreviewCard] = {}
        self._grid_slots: list[QWidget] = []
        self._selected_id = ""
        self._reference_variant_id = ""
        self._page_index = 0
        self._grid_columns = self.COMPACT_COLUMNS
        self._grid_rows = self.COMPACT_ROWS
        self._updating_controls = False
        self._loading_detail = False
        self._comparison_transform_active = False
        self._comparison_transform_mode = ""
        self._comparison_transform_changed = False
        self._comparison_transform_start_dirty = False
        self._capacity_update_pending = False
        self._coordination_busy = False
        self._coordination_task: _CoordinationTask | None = None
        self._coordination_task_total = 0
        self._coordination_task_ink: dict[str, Any] = {}
        self._baseline_task: _BaselineAnalysisTask | None = None
        self._coordination_pool = QThreadPool(self)
        self._coordination_pool.setMaxThreadCount(1)
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(90)
        self._preview_timer.timeout.connect(self._refresh_selected_preview)
        self._comparison_wheel_timer = QTimer(self)
        self._comparison_wheel_timer.setSingleShot(True)
        self._comparison_wheel_timer.setInterval(240)
        self._comparison_wheel_timer.timeout.connect(
            self._finish_comparison_wheel_transform
        )

        self._build_ui()
        self._setup_shortcuts()
        self._install_space_pan_event_filter()
        self._reload_variants(initial_variants)
        if self._baseline_analysis_pending:
            self._show_baseline_loading()
            QTimer.singleShot(0, self._start_baseline_analysis)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)
        root.addWidget(self._build_header())
        root.addWidget(self._build_coordination_progress())

        self._main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._main_splitter.setChildrenCollapsible(False)
        self._main_splitter.addWidget(self._build_glyph_list())
        self._main_splitter.addWidget(self._build_center_panel())
        self._main_splitter.addWidget(self._build_tools())
        self._main_splitter.setStretchFactor(0, 0)
        self._main_splitter.setStretchFactor(1, 1)
        self._main_splitter.setStretchFactor(2, 0)
        self._main_splitter.setSizes(
            [self.LIST_PANEL_DEFAULT_WIDTH, 820, self.TOOL_PANEL_DEFAULT_WIDTH]
        )
        self._main_splitter.splitterMoved.connect(lambda _position, _index: self._schedule_capacity_update())
        root.addWidget(self._main_splitter, 1)

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setProperty("role", "card")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(12, 9, 12, 9)
        layout.setSpacing(10)

        brand = QLabel("协")
        brand.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brand.setFixedSize(36, 36)
        brand.setStyleSheet(
            "background: #315f9a; color: #ffffff; border-radius: 5px; "
            "font-size: 17px; font-weight: 700;"
        )
        layout.addWidget(brand)
        title_box = QVBoxLayout()
        title_box.setSpacing(1)
        title = QLabel("整体协调")
        title.setStyleSheet("font-size: 20px; font-weight: 700;")
        title_box.addWidget(title)
        metadata = self._glyph.get_metadata()
        self._library_summary_label = QLabel(
            f"当前字库：{self._glyph.ziku_name} · {metadata.get('DPI', '--')} DPI · "
            f"{metadata.get('画布宽', '--')}×{metadata.get('画布高', '--')} 像素"
        )
        self._library_summary_label.setProperty("role", "muted")
        title_box.addWidget(self._library_summary_label)
        layout.addLayout(title_box)
        layout.addStretch()

        self._complete_button = QPushButton("批量整体协调")
        self._complete_button.setProperty("role", "primary")
        self._complete_button.clicked.connect(self._complete_coordination)
        layout.addWidget(self._complete_button)
        self._back_button = QPushButton("返回首页")
        self._back_button.clicked.connect(self.request_back)
        layout.addWidget(self._back_button)
        return header

    def _build_coordination_progress(self) -> QWidget:
        panel = QFrame()
        panel.setProperty("role", "card")
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(12, 7, 12, 7)
        layout.setSpacing(10)

        self._task_stage_label = QLabel("本次执行：等待开始")
        self._task_stage_label.setMinimumWidth(132)
        layout.addWidget(self._task_stage_label)
        self._task_progress_bar = QProgressBar()
        self._task_progress_bar.setRange(0, 100)
        self._task_progress_bar.setValue(0)
        self._task_progress_bar.setFormat("本次执行 %p%")
        self._task_progress_bar.setTextVisible(True)
        self._task_progress_bar.setFixedHeight(20)
        layout.addWidget(self._task_progress_bar, 1)
        self._task_detail_label = QLabel("0 / 0")
        self._task_detail_label.setProperty("role", "muted")
        self._task_detail_label.setMinimumWidth(220)
        self._task_detail_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        layout.addWidget(self._task_detail_label)
        self._stop_coordination_button = QPushButton("停止整体协调")
        self._stop_coordination_button.clicked.connect(
            self._request_stop_coordination
        )
        self._stop_coordination_button.hide()
        layout.addWidget(self._stop_coordination_button)
        panel.hide()
        self._task_progress_panel = panel
        return panel

    @classmethod
    def _saved_or_default_baseline(cls, summary: dict[str, Any]) -> dict[str, Any]:
        baseline: dict[str, Any] = {
            "有效数": 0,
            "墨色有效数": 0,
            "目标占比": 0.72,
            "宽中位": 0.0,
            "高中位": 0.0,
            "墨色基准": 220.0,
        }
        saved = summary.get("基准", {}) if isinstance(summary, dict) else {}
        if isinstance(saved, dict):
            baseline.update(deepcopy(saved))
        raw_ink = summary.get("墨色基准") if isinstance(summary, dict) else None
        if raw_ink is not None:
            baseline["墨色基准"] = cls._number(raw_ink, 220.0)
        return baseline

    @classmethod
    def _has_current_saved_ink_baseline(cls, summary: dict[str, Any]) -> bool:
        if not isinstance(summary, dict):
            return False
        baseline = summary.get("基准", {})
        if not isinstance(baseline, dict):
            return False
        method = str(
            summary.get("墨色方法", baseline.get("墨色方法", "")) or ""
        ).strip()
        raw_version = summary.get(
            "墨色方法版本",
            baseline.get("墨色方法版本"),
        )
        try:
            version = int(raw_version)
            ink_count = int(baseline.get("墨色有效数", 0) or 0)
            ink_baseline = float(
                summary.get("墨色基准", baseline.get("墨色基准"))
            )
        except (TypeError, ValueError):
            return False
        return (
            method == AdjustmentService.INK_METHOD
            and version == AdjustmentService.INK_METHOD_VERSION
            and ink_count > 0
            and math.isfinite(ink_baseline)
            and 0.0 < ink_baseline <= 255.0
        )

    def _show_baseline_loading(self) -> None:
        self._task_progress_panel.show()
        self._task_stage_label.setText("载入字库：准备")
        self._task_progress_bar.setRange(0, 0)
        self._task_progress_bar.setFormat("正在统计全库基准")
        self._task_detail_label.setText(
            f"0 / {len(self._all_variants)} · 正在准备后台分析"
        )
        self._task_detail_label.setToolTip("")
        self._main_splitter.setEnabled(False)
        self._complete_button.setEnabled(False)
        for action in self._shortcut_actions:
            action.setEnabled(False)

    def _start_baseline_analysis(self) -> None:
        if not self._baseline_analysis_pending or self._baseline_task is not None:
            return
        task = _BaselineAnalysisTask(
            self._glyph.ziku_name,
            self._glyph.ziku_dir,
            self._glyph.snapshot_state(),
        )
        self._baseline_task = task
        task.signals.progress.connect(self._baseline_progress_changed)
        task.signals.finished.connect(self._baseline_analysis_finished)
        task.signals.failed.connect(self._baseline_analysis_failed)
        try:
            QThreadPool.globalInstance().start(task)
        except Exception as exc:
            self._baseline_analysis_failed(str(exc))

    def _baseline_progress_changed(
        self,
        current: int,
        total: int,
        glyph_label: str,
    ) -> None:
        if not self._baseline_analysis_pending:
            return
        normalized_total = max(1, int(total))
        normalized_current = max(0, min(int(current), normalized_total))
        self._task_stage_label.setText("载入字库：分析")
        self._task_progress_bar.setRange(0, normalized_total)
        self._task_progress_bar.setValue(normalized_current)
        self._task_progress_bar.setFormat("载入进度 %p%")
        detail = f"{normalized_current} / {int(total)}"
        if glyph_label:
            detail += f" · {glyph_label}"
        available_width = max(180, self._task_detail_label.width())
        self._task_detail_label.setText(
            self._task_detail_label.fontMetrics().elidedText(
                detail,
                Qt.TextElideMode.ElideMiddle,
                available_width,
            )
        )
        self._task_detail_label.setToolTip(detail)

    def _baseline_analysis_finished(self, result: object) -> None:
        if not self._baseline_analysis_pending:
            return
        baseline = self._saved_or_default_baseline(
            {"基准": result if isinstance(result, dict) else {}}
        )
        self._coordination_baseline = baseline
        self._ink_baseline = self._number(baseline.get("墨色基准"), 220.0)
        self._finish_baseline_loading(True, "全库协调基准已载入")

    def _baseline_analysis_failed(self, message: str) -> None:
        if not self._baseline_analysis_pending:
            return
        detail = str(message).strip() or "未知错误"
        self._finish_baseline_loading(False, f"后台分析失败：{detail}")

    def _finish_baseline_loading(self, succeeded: bool, detail: str) -> None:
        self._baseline_task = None
        self._baseline_analysis_pending = False
        ink_count = int(self._coordination_baseline.get("墨色有效数", 0) or 0)
        with QSignalBlocker(self._ink_check):
            self._ink_check.setChecked(self._initial_ink_enabled)
        self._ink_check.setEnabled(ink_count > 0)
        if ink_count:
            self._ink_baseline_label.setText(
                f"固定墨色基准：{self._ink_baseline:.2f}\n"
                f"自动取样：{ink_count} 个协调样本，本次进入后保持固定"
            )
        else:
            self._ink_baseline_label.setText("固定墨色基准：暂无有效字形")

        for item in self._all_variants:
            variant_id = str(item.get("变体ID", ""))
            self._saved_ink_signatures[variant_id] = self._stored_ink_signature(item)
        self._clear_preview_cache()
        self._refresh_filtered_view(
            preserve_selection=True,
            reload_detail=False,
        )
        self._main_splitter.setEnabled(not self._coordination_busy)
        for action in self._shortcut_actions:
            action.setEnabled(not self._coordination_busy)
        self._complete_button.setEnabled(
            not self._coordination_busy and bool(self._all_variants)
        )

        if succeeded:
            total = max(1, len(self._all_variants))
            self._task_stage_label.setText("载入字库：完成")
            self._task_progress_bar.setRange(0, total)
            self._task_progress_bar.setValue(total)
            self._task_progress_bar.setFormat("载入完成 %p%")
            self._task_detail_label.setText(detail)
            self._task_detail_label.setToolTip(detail)
            QTimer.singleShot(600, self._hide_completed_baseline_progress)
        else:
            self._task_stage_label.setText("载入字库：失败")
            self._task_progress_bar.setRange(0, 100)
            self._task_progress_bar.setValue(0)
            self._task_progress_bar.setFormat("已使用现有基准")
            self._task_detail_label.setText(detail)
            self._task_detail_label.setToolTip(detail)

    def _hide_completed_baseline_progress(self) -> None:
        if not self._baseline_analysis_pending and not self._coordination_busy:
            self._task_progress_panel.hide()

    def _build_glyph_list(self) -> QWidget:
        panel = QFrame()
        panel.setProperty("role", "card")
        panel.setMinimumWidth(self.LIST_PANEL_MIN_WIDTH)
        panel.setMaximumWidth(self.LIST_PANEL_MAX_WIDTH)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        heading = QHBoxLayout()
        title = QLabel("字形列表")
        title.setStyleSheet("font-size: 18px; font-weight: 700;")
        heading.addWidget(title)
        heading.addStretch()
        self._list_count_label = QLabel("显示 0/0")
        self._list_count_label.setProperty("role", "muted")
        self._list_count_label.setToolTip("当前显示 0 个，共 0 个字形")
        heading.addWidget(self._list_count_label)
        layout.addLayout(heading)

        search_row = QHBoxLayout()
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("搜索字符、字形或文件名")
        self._search_edit.setClearButtonEnabled(True)
        self._search_edit.returnPressed.connect(self._execute_search)
        self._search_edit.textChanged.connect(self._restore_search_when_cleared)
        search_row.addWidget(self._search_edit, 1)
        self._search_button = QPushButton("搜索")
        self._search_button.setObjectName("compactButton")
        self._search_button.clicked.connect(self._execute_search)
        search_row.addWidget(self._search_button)
        layout.addLayout(search_row)

        filter_sort_row = QHBoxLayout()
        filter_sort_row.setSpacing(4)
        self._filter_combo = QComboBox()
        self._filter_combo.addItems(self.STATUS_FILTERS)
        self._filter_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._filter_combo.setMinimumContentsLength(3)
        self._filter_combo.setCurrentText(PHASE_FILTER_ALL)
        self._filter_combo.setToolTip("按整体协调状态筛选")
        self._filter_combo.currentTextChanged.connect(self._apply_filters)
        self._filter_combo.setStyleSheet(
            "QComboBox { padding-left: 4px; padding-right: 4px; }"
        )
        self._filter_combo.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Fixed,
        )
        filter_sort_row.addWidget(self._filter_combo, 5)
        self._order_combo = QComboBox()
        self._order_combo.addItems(self.SORT_OPTIONS)
        self._order_combo.setToolTip("调整字形排序")
        self._order_combo.currentTextChanged.connect(self._change_order)
        self._order_combo.setStyleSheet(
            "QComboBox { padding-left: 4px; padding-right: 4px; }"
        )
        self._order_combo.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Fixed,
        )
        filter_sort_row.addWidget(self._order_combo, 4)
        layout.addLayout(filter_sort_row)

        self._glyph_list = QTreeWidget()
        self._glyph_list.setColumnCount(2)
        self._glyph_list.setHeaderLabels(("字形与文件", "状态与提示"))
        self._glyph_list.setRootIsDecorated(True)
        self._glyph_list.setIndentation(14)
        self._glyph_list.setUniformRowHeights(False)
        self._glyph_list.setAlternatingRowColors(False)
        self._glyph_list.setWordWrap(True)
        self._glyph_list.setAnimated(False)
        self._glyph_list.setIconSize(
            QSize(self.LIST_THUMBNAIL_SIZE, self.LIST_THUMBNAIL_SIZE)
        )
        self._glyph_list.setItemDelegateForColumn(
            1,
            TwoLineStatusDelegate(self._glyph_list),
        )
        self._glyph_list.setStyleSheet(
            "QTreeWidget { background: #171b22; border: 1px solid #37404d; }"
            "QTreeWidget::item { min-height: 26px; padding: 1px 3px; }"
            "QTreeWidget::item:selected { background: #3c4773; }"
        )
        status_width = max(
            self._glyph_list.fontMetrics().horizontalAdvance(value)
            for value in (*self.STATUS_FILTERS[1:], "状态与提示")
        )
        marker_width = max(
            self._glyph_list.fontMetrics().horizontalAdvance(value)
            for value in (
                "未保存修改",
                "结构需核对",
                "墨色待确认",
                "人工例外",
                "文件异常",
                "墨色已达标",
                "问题 99",
            )
        )
        self._glyph_list_columns = AdjustableTreeColumns(
            self._glyph_list,
            {
                0: max(
                    160,
                    self._glyph_list.fontMetrics().horizontalAdvance("字形与文件") + 24,
                ),
                1: max(status_width, marker_width) + 24,
            },
            {
                0: 160,
                1: max(status_width, marker_width) + 24,
            },
        )
        self._glyph_list.currentItemChanged.connect(self._select_list_item)
        self._glyph_list.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self._glyph_list.customContextMenuRequested.connect(
            self._show_glyph_context_menu
        )
        self._glyph_list.itemDoubleClicked.connect(self._open_list_item)
        self._glyph_list.itemExpanded.connect(self._schedule_list_thumbnail_loads)
        self._glyph_list.itemCollapsed.connect(self._schedule_list_thumbnail_loads)
        self._glyph_list.verticalScrollBar().valueChanged.connect(
            self._schedule_list_thumbnail_loads
        )
        layout.addWidget(self._glyph_list, 1)

        self._summary_label = QLabel(
            "待协调 0　已协调 0\n"
            "未保存 0　问题 0"
        )
        self._summary_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._summary_label.setProperty("role", "muted")
        self._summary_label.setWordWrap(True)
        layout.addWidget(self._summary_label)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("完成度 %p%")
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFixedHeight(20)
        layout.addWidget(self._progress_bar)
        return panel

    def _build_center_panel(self) -> QWidget:
        panel = QFrame()
        panel.setProperty("role", "card")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        mode_bar = QWidget()
        mode_layout = QHBoxLayout(mode_bar)
        mode_layout.setContentsMargins(8, 7, 8, 7)
        mode_layout.setSpacing(5)
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._comparison_mode_button = self._segment_button("整库比较")
        self._detail_mode_button = self._segment_button("单字精调")
        self._mode_group.addButton(self._comparison_mode_button)
        self._mode_group.addButton(self._detail_mode_button)
        self._comparison_mode_button.setChecked(True)
        self._comparison_mode_button.clicked.connect(self._leave_detail)
        self._detail_mode_button.clicked.connect(lambda: self._enter_detail(self._selected_id))
        mode_layout.addWidget(self._comparison_mode_button)
        mode_layout.addWidget(self._detail_mode_button)
        mode_layout.addStretch()
        self._mode_status_label = QLabel("请选择字形")
        self._mode_status_label.setProperty("role", "muted")
        mode_layout.addWidget(self._mode_status_label)
        layout.addWidget(mode_bar)

        self._view_stack = QStackedWidget()
        self._comparison_view = self._build_comparison_view()
        self._detail_view = self._build_detail_view()
        self._view_stack.addWidget(self._comparison_view)
        self._view_stack.addWidget(self._detail_view)
        layout.addWidget(self._view_stack, 1)
        return panel

    def _build_comparison_view(self) -> QWidget:
        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        nav = QWidget()
        nav_layout = QHBoxLayout(nav)
        nav_layout.setContentsMargins(8, 6, 8, 6)
        nav_layout.setSpacing(5)
        nav_layout.addStretch()
        self._previous_page_button = QPushButton("上一页")
        self._previous_page_button.setObjectName("compactButton")
        self._previous_page_button.clicked.connect(lambda: self._change_page(-1))
        nav_layout.addWidget(self._previous_page_button)
        self._page_label = QLabel("第 0 / 0 页")
        self._page_label.setMinimumWidth(82)
        self._page_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_layout.addWidget(self._page_label)
        self._next_page_button = QPushButton("下一页")
        self._next_page_button.setObjectName("compactButton")
        self._next_page_button.clicked.connect(lambda: self._change_page(1))
        nav_layout.addWidget(self._next_page_button)
        layout.addWidget(nav)

        self._comparison_scroll = QScrollArea()
        self._comparison_scroll.setWidgetResizable(True)
        self._comparison_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._comparison_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._comparison_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._grid_host = QWidget()
        self._grid = QGridLayout(self._grid_host)
        self._grid.setContentsMargins(10, 10, 10, 10)
        self._grid.setHorizontalSpacing(10)
        self._grid.setVerticalSpacing(10)
        self._comparison_scroll.setWidget(self._grid_host)
        layout.addWidget(self._comparison_scroll, 1)

        self._status_label = QLabel("本页 0 字")
        self._status_label.setContentsMargins(10, 5, 10, 7)
        self._status_label.setProperty("role", "muted")
        layout.addWidget(self._status_label)
        return view

    def _build_detail_view(self) -> QWidget:
        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        nav = QWidget()
        nav_layout = QHBoxLayout(nav)
        nav_layout.setContentsMargins(8, 6, 8, 6)
        nav_layout.addStretch()
        self._detail_position_label = QLabel("0 / 0")
        self._detail_position_label.setProperty("role", "muted")
        nav_layout.addWidget(self._detail_position_label)
        previous = QPushButton("上一字形")
        previous.setObjectName("compactButton")
        previous.clicked.connect(lambda: self._move_detail_selection(-1))
        nav_layout.addWidget(previous)
        next_button = QPushButton("下一字形")
        next_button.setObjectName("compactButton")
        next_button.clicked.connect(lambda: self._move_detail_selection(1))
        nav_layout.addWidget(next_button)
        layout.addWidget(nav)

        toolbar = QWidget()
        toolbar.setStyleSheet(
            "background: #1b2028; border-top: 1px solid #37404d; border-bottom: 1px solid #37404d;"
        )
        toolbar_layout = QVBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(7, 5, 7, 5)
        toolbar_layout.setSpacing(4)
        edit_row = QHBoxLayout()
        edit_row.setSpacing(4)
        self._undo_button = self._toolbar_button("撤销", "撤销（Ctrl+Z）")
        self._undo_button.clicked.connect(self._detail_canvas_undo)
        edit_row.addWidget(self._undo_button)
        self._redo_button = self._toolbar_button("重做", "重做（Ctrl+Y）")
        self._redo_button.clicked.connect(self._detail_canvas_redo)
        edit_row.addWidget(self._redo_button)
        edit_row.addWidget(self._vertical_separator())
        transform = self._toolbar_button("自由变换", "自由变换", checkable=True)
        transform.setChecked(True)
        edit_row.addWidget(transform)
        edit_row.addStretch()
        toolbar_layout.addLayout(edit_row)

        view_row = QHBoxLayout()
        view_row.setSpacing(4)
        fit = self._toolbar_button("适合窗口", "适合窗口")
        fit.clicked.connect(self._detail_canvas_fit)
        view_row.addWidget(fit)
        actual = self._toolbar_button("1:1", "按图像实际像素显示")
        actual.clicked.connect(self._detail_canvas_actual_size)
        view_row.addWidget(actual)
        self._source_button = self._toolbar_button("查看原稿", "按住查看原稿")
        self._source_button.pressed.connect(lambda: self._detail_canvas.set_source_preview_visible(True))
        self._source_button.released.connect(lambda: self._detail_canvas.set_source_preview_visible(False))
        view_row.addWidget(self._source_button)
        self._grid_button = self._toolbar_button("田字格", "显示或隐藏田字格", checkable=True)
        self._grid_button.setChecked(True)
        self._grid_button.toggled.connect(self._detail_canvas_set_grid)
        view_row.addWidget(self._grid_button)
        self._background_group = QButtonGroup(self)
        self._background_group.setExclusive(True)
        self._white_button = self._toolbar_button("白底", "白色预览背景", checkable=True)
        self._transparent_button = self._toolbar_button("透明底", "透明棋盘格背景", checkable=True)
        self._white_button.setChecked(True)
        self._white_button.clicked.connect(
            lambda: self._detail_canvas.set_background_mode(ReviewCanvas.BACKGROUND_WHITE)
        )
        self._transparent_button.clicked.connect(
            lambda: self._detail_canvas.set_background_mode(ReviewCanvas.BACKGROUND_CHECKERBOARD)
        )
        self._background_group.addButton(self._white_button)
        self._background_group.addButton(self._transparent_button)
        view_row.addWidget(self._white_button)
        view_row.addWidget(self._transparent_button)
        view_row.addStretch()
        toolbar_layout.addLayout(view_row)
        layout.addWidget(toolbar)

        self._detail_canvas = ReviewCanvas()
        self._detail_canvas.set_tool(ReviewCanvas.TOOL_TRANSFORM)
        self._detail_canvas.set_background_mode(ReviewCanvas.BACKGROUND_WHITE)
        self._detail_canvas.set_grid_visible(True)
        self._detail_canvas.transform_changed.connect(self._on_detail_transform_changed)
        self._detail_canvas.changed.connect(lambda _dirty: self._refresh_current_labels())
        self._detail_canvas.history_changed.connect(self._update_history_buttons)
        layout.addWidget(self._detail_canvas, 1)

        footer = QWidget()
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(10, 6, 10, 6)
        self._detail_source_label = QLabel("来源：-")
        self._detail_source_label.setProperty("role", "muted")
        footer_layout.addWidget(self._detail_source_label, 1)
        self._zoom_label = QLabel("100%")
        self._zoom_label.setProperty("role", "muted")
        self._detail_canvas.zoom_changed.connect(lambda value: self._zoom_label.setText(f"{value}%"))
        footer_layout.addWidget(self._zoom_label)
        layout.addWidget(footer)
        return view

    def _build_tools(self) -> QWidget:
        panel = QFrame()
        panel.setProperty("role", "card")
        panel.setMinimumWidth(self.TOOL_PANEL_MIN_WIDTH)
        panel.setMaximumWidth(self.TOOL_PANEL_MAX_WIDTH)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        heading = QWidget()
        heading_layout = QHBoxLayout(heading)
        heading_layout.setContentsMargins(11, 10, 11, 9)
        title = QLabel("当前字形")
        title.setStyleSheet("font-size: 16px; font-weight: 700;")
        heading_layout.addWidget(title)
        heading_layout.addStretch()
        self._dirty_label = QLabel("未选择")
        self._dirty_label.setProperty("role", "muted")
        heading_layout.addWidget(self._dirty_label)
        layout.addWidget(heading)

        self._tools_scroll = QScrollArea()
        self._tools_scroll.setWidgetResizable(True)
        self._tools_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._tools_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(11, 7, 11, 12)
        body_layout.setSpacing(10)
        body_layout.addWidget(self._build_current_section())
        body_layout.addWidget(self._horizontal_separator())
        body_layout.addWidget(self._build_reference_section())
        body_layout.addWidget(self._horizontal_separator())
        body_layout.addWidget(self._build_transform_section())
        body_layout.addWidget(self._horizontal_separator())
        body_layout.addWidget(self._build_ink_section())
        body_layout.addStretch()
        self._tools_scroll.setWidget(body)
        layout.addWidget(self._tools_scroll, 1)

        self._action_footer = QWidget()
        self._action_footer.setStyleSheet("background: #1b2028; border-top: 1px solid #37404d;")
        action_layout = QVBoxLayout(self._action_footer)
        action_layout.setContentsMargins(9, 9, 9, 9)
        action_layout.setSpacing(6)
        self._restore_button = QPushButton("还原本字")
        self._restore_button.clicked.connect(self._reset_selected)
        action_layout.addWidget(self._restore_button)
        self._save_button = QPushButton("保存本页")
        self._save_button.setProperty("role", "primary")
        self._save_button.clicked.connect(self._save_action)
        action_layout.addWidget(self._save_button)
        self._save_next_button = QPushButton("保存并下一字")
        self._save_next_button.clicked.connect(self._save_and_next)
        self._save_next_button.hide()
        action_layout.addWidget(self._save_next_button)
        layout.addWidget(self._action_footer)
        return panel

    def _build_current_section(self) -> QWidget:
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        self._current_char_label = QLabel("-")
        self._current_char_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._current_char_label.setFixedSize(42, 42)
        self._current_char_label.setStyleSheet(
            "background: #ffffff; color: #111111; border: 1px solid #596574; font-size: 24px;"
        )
        layout.addWidget(self._current_char_label)
        info = QVBoxLayout()
        info.setSpacing(2)
        self._current_file_label = QLabel("未选择字形")
        self._current_file_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        info.addWidget(self._current_file_label)
        self._current_index_label = QLabel("-")
        self._current_index_label.setProperty("role", "muted")
        info.addWidget(self._current_index_label)
        layout.addLayout(info, 1)
        return panel

    def _build_reference_section(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)
        heading = QHBoxLayout()
        title = QLabel("对照参照")
        title.setStyleSheet("font-weight: 700;")
        heading.addWidget(title)
        heading.addStretch()
        self._reference_pin_button = QToolButton()
        self._reference_pin_button.setText("固定")
        self._reference_pin_button.setCheckable(True)
        self._reference_pin_button.setToolTip("切换当前字形时保持参照字不变")
        heading.addWidget(self._reference_pin_button)
        layout.addLayout(heading)
        self._reference_combo = QComboBox()
        self._reference_combo.currentIndexChanged.connect(self._reference_changed)
        layout.addWidget(self._reference_combo)
        self._reference_overlay_check = QCheckBox("半透明叠加参照")
        self._reference_overlay_check.setChecked(False)
        self._reference_overlay_check.toggled.connect(self._update_reference_overlay)
        layout.addWidget(self._reference_overlay_check)
        return panel

    def _build_transform_section(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        heading = QHBoxLayout()
        title = QLabel("几何变换")
        title.setStyleSheet("font-weight: 700;")
        heading.addWidget(title)
        heading.addStretch()
        reset = QPushButton("重置参数")
        reset.setObjectName("compactButton")
        reset.clicked.connect(self._reset_selected)
        heading.addWidget(reset)
        layout.addLayout(heading)

        positions = QGridLayout()
        positions.setHorizontalSpacing(7)
        positions.setVerticalSpacing(4)
        positions.addWidget(QLabel("水平位置 X"), 0, 0)
        positions.addWidget(QLabel("垂直位置 Y"), 0, 1)
        self._offset_x_spin = QSpinBox()
        self._offset_y_spin = QSpinBox()
        for spin in (self._offset_x_spin, self._offset_y_spin):
            spin.setRange(-self.TRANSFORM_OFFSET_LIMIT, self.TRANSFORM_OFFSET_LIMIT)
            spin.setMinimumWidth(0)
            spin.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self._offset_x_spin.valueChanged.connect(lambda value: self._apply_transform_field("x", float(value)))
        self._offset_y_spin.valueChanged.connect(lambda value: self._apply_transform_field("y", float(value)))
        positions.addWidget(self._offset_x_spin, 1, 0)
        positions.addWidget(self._offset_y_spin, 1, 1)
        layout.addLayout(positions)

        self._scale_slider, self._scale_value_label = self._add_percent_control(
            layout, "等比缩放", "scale"
        )
        self._stretch_w_slider, self._stretch_w_value_label = self._add_percent_control(
            layout, "水平拉伸 / 压缩", "stretch_w"
        )
        self._stretch_h_slider, self._stretch_h_value_label = self._add_percent_control(
            layout, "垂直拉伸 / 压缩", "stretch_h"
        )

        rotation_head = QHBoxLayout()
        rotation_head.addWidget(QLabel("旋转"))
        rotation_head.addStretch()
        self._rotation_value_label = QLabel("0°")
        self._rotation_value_label.setProperty("role", "muted")
        rotation_head.addWidget(self._rotation_value_label)
        layout.addLayout(rotation_head)
        self._rotation_slider = QSlider(Qt.Orientation.Horizontal)
        self._rotation_slider.setRange(-180, 180)
        self._rotation_slider.valueChanged.connect(self._on_rotation_slider_changed)
        self._rotation_slider.sliderReleased.connect(
            lambda: self._apply_transform_field("rotation", float(self._rotation_slider.value()))
        )
        layout.addWidget(self._rotation_slider)

        self._advanced_button = QToolButton()
        self._advanced_button.setText("高级变形：斜切与四角扭曲")
        self._advanced_button.setCheckable(True)
        self._advanced_button.setArrowType(Qt.ArrowType.RightArrow)
        self._advanced_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._advanced_button.toggled.connect(self._toggle_advanced_panel)
        layout.addWidget(self._advanced_button)
        self._advanced_panel = QWidget()
        advanced_layout = QGridLayout(self._advanced_panel)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.setHorizontalSpacing(6)
        advanced_layout.setVerticalSpacing(4)
        advanced_layout.addWidget(QLabel("控制点"), 0, 0)
        advanced_layout.addWidget(QLabel("X"), 0, 1)
        advanced_layout.addWidget(QLabel("Y"), 0, 2)
        self._distort_spins: list[QDoubleSpinBox] = []
        for row, corner in enumerate(("左上", "右上", "右下", "左下"), 1):
            advanced_layout.addWidget(QLabel(corner), row, 0)
            for axis in range(2):
                index = (row - 1) * 2 + axis
                spin = QDoubleSpinBox()
                spin.setRange(-float(self.TRANSFORM_OFFSET_LIMIT), float(self.TRANSFORM_OFFSET_LIMIT))
                spin.setDecimals(1)
                spin.setSingleStep(1.0)
                spin.setMinimumWidth(0)
                spin.valueChanged.connect(
                    lambda value, target=index: self._apply_distort_field(target, value)
                )
                self._distort_spins.append(spin)
                advanced_layout.addWidget(spin, row, axis + 1)
        clear_distort = QPushButton("清除高级变形")
        clear_distort.setObjectName("compactButton")
        clear_distort.clicked.connect(self._reset_distortion)
        advanced_layout.addWidget(clear_distort, 5, 0, 1, 3)
        self._advanced_panel.hide()
        layout.addWidget(self._advanced_panel)
        return panel

    def _build_ink_section(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)
        heading = QHBoxLayout()
        title = QLabel("全库墨色")
        title.setStyleSheet("font-weight: 700;")
        heading.addWidget(title)
        heading.addStretch()
        heading.addWidget(QLabel("最终成品阶段"))
        layout.addLayout(heading)
        self._ink_check = QCheckBox("统一墨色")
        self._ink_check.setChecked(self._initial_ink_enabled)
        self._ink_check.toggled.connect(self._ink_mode_changed)
        layout.addWidget(self._ink_check)
        mode_row = QHBoxLayout()
        mode_row.setSpacing(6)
        mode_row.addWidget(QLabel("本字处理"))
        self._ink_strategy_combo = QComboBox()
        self._ink_strategy_combo.addItems(self.INK_MODES)
        self._ink_strategy_combo.setToolTip(
            "跟随全库：自动向固定基准校正\n"
            "保留本字：不改变本字墨色，未达标时仍需确认\n"
            "人工例外：保留本字并明确接受为全库例外"
        )
        self._ink_strategy_combo.currentTextChanged.connect(
            self._ink_strategy_changed
        )
        mode_row.addWidget(self._ink_strategy_combo, 1)
        layout.addLayout(mode_row)
        ink_count = int(self._coordination_baseline.get("墨色有效数", 0) or 0)
        if self._baseline_analysis_pending:
            ink_text = "正在后台统计全库墨色基准，完成前暂不可编辑或提交"
            self._ink_check.setEnabled(False)
        elif ink_count:
            ink_text = (
                f"固定墨色基准：{self._ink_baseline:.2f}\n"
                f"自动取样：{ink_count} 个协调样本，本次进入后保持固定"
            )
        else:
            ink_text = "固定墨色基准：暂无有效字形"
            self._ink_check.setEnabled(False)
        self._ink_baseline_label = QLabel(ink_text)
        self._ink_baseline_label.setProperty("role", "muted")
        self._ink_baseline_label.setWordWrap(True)
        layout.addWidget(self._ink_baseline_label)
        self._ink_summary_label = QLabel("墨色达标 0　待确认 0　人工例外 0")
        self._ink_summary_label.setProperty("role", "muted")
        self._ink_summary_label.setWordWrap(True)
        layout.addWidget(self._ink_summary_label)
        self._ink_result_label = QLabel("本字墨色：未保存协调结果")
        self._ink_result_label.setProperty("role", "muted")
        self._ink_result_label.setWordWrap(True)
        self._ink_result_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self._ink_result_label)
        return panel

    def _add_percent_control(
        self,
        layout: QVBoxLayout,
        label: str,
        field: str,
    ) -> tuple[QSlider, QLabel]:
        heading = QHBoxLayout()
        heading.addWidget(QLabel(label))
        heading.addStretch()
        value_label = QLabel("100%")
        value_label.setProperty("role", "muted")
        heading.addWidget(value_label)
        layout.addLayout(heading)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(
            -self.TRANSFORM_PERCENT_SLIDER_EXTENT,
            self.TRANSFORM_PERCENT_SLIDER_EXTENT,
        )
        slider.valueChanged.connect(
            lambda _value, control=slider, name=field: self._on_percent_slider_changed(control, name)
        )
        slider.sliderReleased.connect(
            lambda control=slider, name=field: self._commit_percent_slider(control, name)
        )
        layout.addWidget(slider)
        return slider, value_label

    def _setup_shortcuts(self) -> None:
        self._shortcut_actions: list[QAction] = []
        for sequence, callback in (
            (QKeySequence.StandardKey.Save, self._save_action),
            (QKeySequence.StandardKey.Undo, self._detail_canvas_undo),
            (QKeySequence.StandardKey.Redo, self._detail_canvas_redo),
            (QKeySequence(Qt.Key.Key_Return), lambda: self._enter_detail(self._selected_id)),
        ):
            action = QAction(self)
            action.setShortcut(sequence)
            action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            action.triggered.connect(callback)
            self.addAction(action)
            self._shortcut_actions.append(action)

    def _install_space_pan_event_filter(self) -> None:
        application = QApplication.instance()
        if application is not None:
            application.installEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if (
            isinstance(event, QKeyEvent)
            and event.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease)
            and event.key() == Qt.Key.Key_Space
            and self._view_stack.currentWidget() is self._detail_view
            and self._owns_event_target(watched)
            and not self._canvas_owns_event_target(watched)
            and (
                self._detail_canvas.space_pan_active
                or (self._cursor_is_over_canvas() and not self._focused_control_uses_space())
            )
        ):
            handled = self._detail_canvas.handle_space_pan_key(
                event.type() == QEvent.Type.KeyPress,
                auto_repeat=event.isAutoRepeat(),
            )
            if handled:
                event.accept()
                return True
        return super().eventFilter(watched, event)

    def _owns_event_target(self, watched: QObject) -> bool:
        return watched is self or (isinstance(watched, QWidget) and self.isAncestorOf(watched))

    def _canvas_owns_event_target(self, watched: QObject) -> bool:
        return watched is self._detail_canvas or (
            isinstance(watched, QWidget) and self._detail_canvas.isAncestorOf(watched)
        )

    @staticmethod
    def _focused_control_uses_space() -> bool:
        focused = QApplication.focusWidget()
        while focused is not None:
            if isinstance(focused, (QAbstractButton, QLineEdit)):
                return True
            focused = focused.parentWidget()
        return False

    def _cursor_is_over_canvas(self) -> bool:
        position = self._detail_canvas.mapFromGlobal(QCursor.pos())
        return self._detail_canvas.rect().contains(position)

    @staticmethod
    def _default_adjustment() -> dict[str, Any]:
        return {
            "移动X": 0.0,
            "移动Y": 0.0,
            "等比缩放": 1.0,
            "水平拉伸": 1.0,
            "垂直拉伸": 1.0,
            "缩放X": 1.0,
            "缩放Y": 1.0,
            "旋转": 0.0,
            "斜切X": 0.0,
            "斜切Y": 0.0,
            "扭曲": [0.0] * 8,
        }

    def _current_ink_config(
        self,
        variant_id: str = "",
        variant_ids: Optional[set[str]] = None,
    ) -> dict[str, Any]:
        ink_available = (
            not self._baseline_analysis_pending
            and int(self._coordination_baseline.get("墨色有效数", 0) or 0) > 0
        )
        config_data: dict[str, Any] = {
            "启用": ink_available and self._ink_check.isChecked(),
            "基准": self._ink_baseline,
            "方法": AdjustmentService.INK_METHOD,
            "方法版本": AdjustmentService.INK_METHOD_VERSION,
        }
        if variant_id:
            config_data["模式"] = self._ink_modes.get(variant_id, self.INK_MODES[0])
            return config_data
        selected_ids = variant_ids if variant_ids is not None else set(self._variant_by_id)
        config_data["逐字模式"] = {
            item: self._ink_modes.get(item, self.INK_MODES[0])
            for item in selected_ids
        }
        return config_data

    @staticmethod
    def _ink_signature(ink_config: dict[str, Any]) -> tuple[Any, ...]:
        enabled = bool(ink_config.get("启用", False))
        raw_baseline = ink_config.get("基准")
        try:
            baseline = round(float(raw_baseline), 2) if raw_baseline is not None else None
        except (TypeError, ValueError):
            baseline = None
        mode = str(ink_config.get("模式", ConsistencyPage.INK_MODES[0]))
        if mode not in ConsistencyPage.INK_MODES:
            mode = ConsistencyPage.INK_MODES[0]
        raw_version = ink_config.get("方法版本")
        if raw_version is None:
            raw_version = (
                AdjustmentService.INK_METHOD_VERSION
                if "方法" not in ink_config
                else 0
            )
        try:
            method_version = int(raw_version)
        except (TypeError, ValueError):
            method_version = 0
        return (
            enabled,
            baseline,
            str(ink_config.get("方法", AdjustmentService.INK_METHOD)),
            method_version,
            mode,
        )

    @staticmethod
    def _ink_record(detail: dict[str, Any]) -> dict[str, Any]:
        parameters = detail.get("整体协调参数", {})
        if not isinstance(parameters, dict):
            return {}
        record = parameters.get("墨色协调", {})
        return record if isinstance(record, dict) else {}

    @classmethod
    def _stored_ink_mode(cls, detail: dict[str, Any]) -> str:
        record = cls._ink_record(detail)
        mode = str(record.get("模式", "") or "").strip()
        if bool(record.get("人工接受例外")) or str(record.get("状态", "")) in {
            "人工例外",
            "人工接受例外",
        }:
            mode = "人工例外"
        return mode if mode in cls.INK_MODES else cls.INK_MODES[0]

    def _ink_result_category(self, detail: dict[str, Any]) -> str:
        """从统一工作流解析结果读取墨色分类。"""
        ink_status = self._workflow_status(detail).ink_status
        return {
            INK_STATUS_ACHIEVED: "已达标",
            INK_STATUS_PENDING: "待确认",
            INK_STATUS_EXCEPTION: "人工例外",
        }.get(ink_status, "")

    @staticmethod
    def _optional_number_text(value: object) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "--"
        if not math.isfinite(number):
            return "--"
        return f"{number:.2f}"

    def _ink_result_text(self, detail: dict[str, Any]) -> str:
        if str(detail.get("状态", "")) != config.STATUS_FINISHED:
            return "本字墨色：尚无有效协调实测记录"
        record = self._ink_record(detail)
        if not record:
            return "本字墨色：尚无协调实测记录"
        category = self._ink_result_category(detail)
        status = {
            "已达标": "墨色已达标",
            "待确认": "墨色待确认",
            "人工例外": "人工例外",
        }.get(category, "未启用墨色统一")
        before = self._optional_number_text(record.get("调整前墨色"))
        saved_recheck = record.get("保存后复测") is True
        after_value = (
            record.get("保存后墨色")
            if saved_recheck
            else record.get("调整后墨色")
        )
        after = self._optional_number_text(after_value)
        target = self._optional_number_text(record.get("基准"))
        deviation = self._optional_number_text(record.get("目标偏差"))
        pixel_type = str(record.get("像素类型", "") or "未记录")
        measurement_label = "保存后" if saved_recheck else "调整后"
        recheck_label = "（持久化复测）" if saved_recheck else ""
        text = (
            f"本字墨色：{status}\n"
            f"调整前 {before}　{measurement_label} {after}{recheck_label}\n"
            f"基准 {target}　偏差 {deviation}\n"
            f"像素类型：{pixel_type}"
        )
        reason = str(record.get("跳过原因", "") or "").strip()
        if reason:
            text += f"\n说明：{reason}"
        return text

    def _stored_ink_signature(self, detail: dict[str, Any]) -> tuple[Any, ...]:
        record = self._ink_record(detail)
        if isinstance(record, dict) and "启用" in record:
            stored = dict(record)
            stored["模式"] = self._stored_ink_mode(detail)
            return self._ink_signature(stored)
        if str(detail.get("状态", "")) == config.STATUS_FINISHED:
            return self._ink_signature({"启用": False, "基准": None, "方法": ""})
        variant_id = str(detail.get("变体ID", ""))
        return self._ink_signature(self._current_ink_config(variant_id))

    def _get_adjustment(self, variant_id: str) -> dict[str, Any]:
        return self._adjustments.setdefault(variant_id, self._default_adjustment())

    def _reload_variants(
        self,
        loaded: list[dict[str, Any]] | None = None,
    ) -> None:
        selected = self._selected_id
        if loaded is None:
            loaded = self._adjustment_service.load_reviewed_variants(
                pinyin_order=False
            )
        self._workflow_summary = self._glyph.get_coordination_summary()
        self._finished_dir = self._glyph.get_workflow_dirs()["成品"]
        self._workflow_status_cache.clear()
        self._all_variants = list(loaded)
        self._variant_by_id = {
            str(item.get("变体ID", "")): item
            for item in self._all_variants
            if item.get("变体ID")
        }
        self._list_variants = [
            detail
            for detail in self._all_variants
            if self._stage_projection(detail).admitted
        ]
        admitted_ids = {
            str(detail.get("变体ID", "")) for detail in self._list_variants
        }
        self._all_variants = [
            detail
            for detail in self._all_variants
            if str(detail.get("变体ID", "")) in admitted_ids
        ]
        self._list_variant_by_id = {
            str(item.get("变体ID", "")): item
            for item in self._list_variants
            if item.get("变体ID")
        }
        self._variant_by_id = {
            str(item.get("变体ID", "")): item for item in self._all_variants
        }
        for order, detail in enumerate(self._list_variants):
            variant_id = str(detail.get("变体ID", ""))
            self._import_order.setdefault(variant_id, order)
        for detail in loaded:
            variant_id = str(detail.get("变体ID", ""))
            if variant_id not in self._ink_modes:
                self._ink_modes[variant_id] = self._stored_ink_mode(detail)
            if variant_id not in self._adjustments:
                saved = self._adjustment_service.load_saved_coordination_adjustments(detail)
                self._adjustments[variant_id] = deepcopy(saved)
                self._saved_adjustments[variant_id] = deepcopy(saved)
                self._saved_signatures[variant_id] = self._adjustment_signature(saved)
                self._saved_ink_signatures[variant_id] = self._stored_ink_signature(detail)
        if selected not in self._variant_by_id:
            # 首次进入时交给筛选后的当前排序选择第一项，避免导入首项
            # 在拼音序中靠后而把比较墙直接带到后续页面。
            self._selected_id = ""
        self._apply_filters()

    def _apply_filters(self, _value: object = None) -> None:
        self._finish_pending_comparison_transform()
        self._refresh_filtered_view()

    def _execute_search(self, _checked: bool = False) -> None:
        """按回车或按钮执行一次全新的搜索，并定位第一条结果。"""

        self._finish_pending_comparison_transform()
        self._refresh_filtered_view(select_first=True)

    def _restore_search_when_cleared(self, text: str) -> None:
        """删除检索文字后立即恢复当前阶段筛选下的全部字形。"""

        if not text.strip():
            self._apply_filters()

    def _refresh_filtered_view(
        self,
        *,
        preserve_selection: bool = False,
        reload_detail: bool = True,
        select_first: bool = False,
    ) -> None:
        previous_selected_id = self._selected_id
        self._workflow_summary = self._glyph.get_coordination_summary()
        self._workflow_status_cache.clear()
        for detail in self._list_variants:
            self._stage_projection(detail)
        query = self._search_edit.text().strip().casefold()
        status_filter = self._filter_combo.currentText()
        list_filtered: list[dict[str, Any]] = []
        for detail in self._list_variants:
            text = " ".join(
                str(detail.get(key, ""))
                for key in ("归属字", "原始文件", "导入前文件名", "中间文件", "审核文件")
            ).casefold()
            if query and query not in text:
                continue
            if not self._matches_status_filter(detail, status_filter):
                continue
            list_filtered.append(detail)
        self._sort_variants(list_filtered)
        self._list_visible_variants = list_filtered
        self._variants = [
            detail
            for detail in list_filtered
            if str(detail.get("变体ID", "")) in self._variant_by_id
        ]
        filtered_ids = {str(item.get("变体ID", "")) for item in self._variants}
        can_preserve_selection = (
            preserve_selection and self._selected_id in self._variant_by_id
        )
        if select_first and self._variants:
            self._selected_id = str(self._variants[0].get("变体ID", ""))
        elif self._selected_id not in filtered_ids and not can_preserve_selection:
            self._selected_id = (
                str(self._variants[0].get("变体ID", ""))
                if self._variants
                else ""
            )
        selected_index = self._variant_index(self._selected_id)
        self._page_index = selected_index // self._page_size() if selected_index >= 0 else 0
        self._populate_list()
        self._populate_reference_combo()
        self._render_page()
        if self._selected_id:
            if reload_detail or self._selected_id != previous_selected_id:
                self._load_detail_canvas(self._selected_id)
            else:
                self._refresh_current_labels()
        else:
            self._clear_detail()
        self._refresh_statistics()

    def _sort_variants(self, variants: list[dict[str, Any]]) -> None:
        mode = self._order_combo.currentText()
        if mode == "文件名顺序":
            variants.sort(key=lambda item: natural_key(str(item.get("原始文件", ""))))
        elif mode == "导入顺序":
            variants.sort(key=lambda item: self._import_order.get(str(item.get("变体ID", "")), 0))
        else:
            variants.sort(
                key=lambda item: (
                    pinyin_natural_key(str(item.get("归属字", ""))),
                    natural_key(str(item.get("原始文件", ""))),
                )
            )

    def _change_order(self, _value: object = None) -> None:
        self._apply_filters()

    def _populate_list(self) -> None:
        group_positions: dict[str, int] = {}
        variant_positions: dict[str, int] = {}
        for detail in self._list_variants:
            char = str(detail.get("归属字", ""))
            group_positions[char] = group_positions.get(char, 0) + 1
            variant_positions[str(detail.get("变体ID", ""))] = group_positions[char]

        groups: dict[str, list[dict[str, Any]]] = {}
        for detail in self._list_visible_variants:
            groups.setdefault(str(detail.get("归属字", "")) or "?", []).append(detail)
        all_groups: dict[str, list[dict[str, Any]]] = {}
        for detail in self._list_variants:
            all_groups.setdefault(
                str(detail.get("归属字", "")) or "?",
                [],
            ).append(detail)

        self._glyph_list.blockSignals(True)
        self._glyph_list.clear()
        self._list_items_by_id.clear()
        selected_item: QTreeWidgetItem | None = None
        for char, group_items in groups.items():
            all_group_items = all_groups.get(char, group_items)
            projections = [
                self._stage_projection(detail) for detail in all_group_items
            ]
            coordinated_count = sum(
                projection.completed for projection in projections
            )
            problem_count = sum(
                self._is_problem_status(projection)
                for projection in projections
            )
            parent = QTreeWidgetItem(
                [
                    f"{char}（{len(all_group_items)}个字形）",
                    "",
                ]
            )
            parent.setFlags(parent.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            parent_font = parent.font(0)
            parent_font.setBold(True)
            parent.setFont(0, parent_font)
            parent.setFont(1, parent_font)
            set_two_line_status(
                parent,
                1,
                f"已协调 {coordinated_count}/{len(all_group_items)}",
                f"问题 {problem_count}",
                self._coordination_status_color(
                    STATUS_COORDINATED
                    if coordinated_count == len(all_group_items)
                    else STAGE_PENDING_COORDINATION
                ),
                QColor("#F2B84B" if problem_count else "#A6B0BE"),
            )
            parent.setToolTip(
                0,
                f"{char}：共 {len(all_group_items)} 个字形，当前显示 {len(group_items)} 个\n"
                f"已协调 {coordinated_count}，待协调 "
                f"{len(all_group_items) - coordinated_count}\n"
                f"有问题 {problem_count}",
            )
            self._glyph_list.addTopLevelItem(parent)

            for detail in group_items:
                variant_id = str(detail.get("变体ID", ""))
                filename = str(detail.get("原始文件", ""))
                position = variant_positions.get(variant_id, 1)
                projection = self._stage_projection(detail)
                status = projection.status
                markers = self._marker_text(projection)
                child = QTreeWidgetItem(
                    parent,
                    [f"字形{position} · {filename}", ""],
                )
                child.setIcon(0, self._glyph_thumbnail(detail))
                child.setData(0, Qt.ItemDataRole.UserRole, variant_id)
                child.setSizeHint(0, QSize(0, 52))
                child.setToolTip(
                    0,
                    f"{char} · 字形{position}\n文件：{filename}\n"
                    f"整体协调：{status}\n提示：{markers}\n"
                    "协调：可编辑\n"
                    f"{self._ink_result_text(detail)}\n{variant_id}",
                )
                set_two_line_status(
                    child,
                    1,
                    status,
                    markers,
                    self._coordination_status_color(status),
                    self._marker_color(projection),
                )
                self._list_items_by_id[variant_id] = child
                if variant_id == self._selected_id:
                    selected_item = child
            parent.setExpanded(True)
        largest_group = max(
            (len(group_items) for group_items in all_groups.values()),
            default=0,
        )
        self._glyph_list_columns.set_protected_minimum(
            1,
            max(
                self._glyph_list.fontMetrics().horizontalAdvance(
                    f"已协调 {largest_group}/{largest_group}"
                )
                + 24,
                self._glyph_list.fontMetrics().horizontalAdvance("状态与提示") + 24,
            ),
        )
        if selected_item is not None:
            self._glyph_list.setCurrentItem(selected_item)
        self._glyph_list.blockSignals(False)
        visible_count = len(self._list_visible_variants)
        total_count = len(self._list_variants)
        self._list_count_label.setText(f"显示 {visible_count}/{total_count}")
        self._list_count_label.setToolTip(
            f"当前显示 {visible_count} 个，共 {total_count} 个字形"
        )
        self._schedule_list_thumbnail_loads()

    def _populate_reference_combo(self) -> None:
        current = self._reference_variant_id
        reference_variants = sorted(
            self._all_variants,
            key=lambda item: (
                pinyin_natural_key(str(item.get("归属字", ""))),
                natural_key(str(item.get("原始文件", ""))),
            ),
        )
        with QSignalBlocker(self._reference_combo):
            self._reference_combo.clear()
            self._reference_combo.addItem("未选择参照", "")
            target_index = 0
            for detail in reference_variants:
                variant_id = str(detail.get("变体ID", ""))
                char = str(detail.get("归属字", ""))
                filename = str(detail.get("原始文件", ""))
                self._reference_combo.addItem(f"{char} · {filename}", variant_id)
                if variant_id == current:
                    target_index = self._reference_combo.count() - 1
            self._reference_combo.setCurrentIndex(target_index)
        if target_index == 0:
            self._reference_variant_id = ""
        self._reference_overlay_check.setEnabled(bool(self._reference_variant_id))

    def _page_size(self) -> int:
        return self._grid_columns * self._grid_rows

    def _page_variants(self) -> list[dict[str, Any]]:
        start = self._page_index * self._page_size()
        return self._variants[start:start + self._page_size()]

    def _render_page(self) -> None:
        tracked_columns = max(self._grid.columnCount(), self._grid_columns)
        tracked_rows = max(self._grid.rowCount(), self._grid_rows)
        while self._grid.count():
            item = self._grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        # QGridLayout 会保留曾经出现过的行列轨道；缩为紧凑模式时必须显式
        # 清零旧轨道，否则隐藏的第 5 列和第 3 行仍会分走可用空间。
        for column in range(tracked_columns):
            self._grid.setColumnStretch(column, 0)
            self._grid.setColumnMinimumWidth(column, 0)
        for row in range(tracked_rows):
            self._grid.setRowStretch(row, 0)
            self._grid.setRowMinimumHeight(row, 0)
        self._cards.clear()
        self._grid_slots.clear()

        total_pages = math.ceil(len(self._variants) / self._page_size()) if self._variants else 0
        self._page_index = max(0, min(self._page_index, max(0, total_pages - 1)))
        page = self._page_variants()
        for index in range(self._page_size()):
            if index < len(page):
                detail = page[index]
                variant_id = str(detail.get("变体ID", ""))
                slot: QWidget = GlyphPreviewCard(
                    variant_id,
                    (self._canvas_width, self._canvas_height),
                )
                card = slot
                card.selected.connect(self._select_variant)
                card.edit_requested.connect(self._enter_detail)
                card.transform_started.connect(self._begin_comparison_transform)
                card.transform_changed.connect(self._apply_comparison_transform)
                card.transform_finished.connect(self._finish_comparison_transform)
                card.wheel_requested.connect(self._apply_comparison_wheel)
                card.set_transform_interaction_handlers(
                    lambda position, modifiers, target=card: (
                        self._comparison_card_hit_test(
                            target,
                            position,
                            modifiers,
                        )
                    ),
                    self._detail_canvas.transform_cursor_for_hit,
                )
                self._cards[variant_id] = card
                self._update_card(
                    variant_id,
                    render_sync=variant_id == self._selected_id,
                )
            else:
                slot = QWidget()
                slot.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
                slot.setStyleSheet("background: transparent;")
            slot.setMinimumSize(112, 132)
            slot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self._grid_slots.append(slot)
            self._grid.addWidget(slot, index // self._grid_columns, index % self._grid_columns)
        for column in range(self._grid_columns):
            self._grid.setColumnStretch(column, 1)
        for row in range(self._grid_rows):
            self._grid.setRowStretch(row, 1)

        current_page = self._page_index + 1 if total_pages else 0
        self._page_label.setText(f"第 {current_page} / {total_pages} 页")
        self._previous_page_button.setEnabled(self._page_index > 0)
        self._next_page_button.setEnabled(self._page_index + 1 < total_pages)
        dirty = sum(
            self._is_dirty(str(item.get("变体ID", "")))
            for item in page
        )
        self._status_label.setText(f"本页 {len(page)} 字　未保存调整 {dirty} 字")
        self._refresh_selection()

    def _update_card(self, variant_id: str, *, render_sync: bool = True) -> None:
        card = self._cards.get(variant_id)
        detail = self._variant_by_id.get(variant_id)
        if card is None or detail is None:
            return
        cache_key = self._coordinated_preview_cache_key(variant_id)
        image = self._preview_cache.get(cache_key)
        bounds = self._preview_bounds_cache.get(cache_key)
        if image is not None:
            self._preview_cache.move_to_end(cache_key)
        if image is None:
            if render_sync:
                preview = self._adjustment_service.preview_coordinated(
                    detail,
                    self._get_adjustment(variant_id),
                    self.WORK_RATIO,
                    self._current_ink_config(variant_id),
                )
                if preview is not None:
                    try:
                        image = self._pil_to_qimage(preview.image)
                        bounds = preview.bounds
                    finally:
                        preview.image.close()
                    self._store_preview(variant_id, cache_key, image, bounds)
            else:
                self._request_card_preview(variant_id, cache_key)
        if image is not None:
            card.set_preview(image, bounds)
            self._set_list_thumbnail_from_preview(variant_id, image)
        adjustment = self._get_adjustment(variant_id)
        card.set_transform(
            {
                "x": adjustment.get("移动X", 0.0),
                "y": adjustment.get("移动Y", 0.0),
                "scale": adjustment.get("等比缩放", 1.0),
                "rotation": adjustment.get("旋转", 0.0),
                "stretch_w": adjustment.get("水平拉伸", 1.0),
                "stretch_h": adjustment.get("垂直拉伸", 1.0),
                "distort": adjustment.get("扭曲", [0.0] * 8),
            }
        )
        card.set_metadata(
            str(detail.get("归属字", "")),
            str(detail.get("原始文件", "")),
            self._coordination_status(detail),
        )
        card.set_selected(variant_id == self._selected_id)
        if variant_id == self._selected_id:
            self._sync_selected_card_controls(
                live=self._comparison_transform_active,
            )
        else:
            card.set_control_polygon(None)

    def _request_card_preview(
        self,
        variant_id: str,
        cache_key: tuple[Any, ...],
    ) -> None:
        pending = self._preview_workers.get(variant_id)
        if pending is not None and pending[0] == cache_key:
            return
        detail = deepcopy(self._variant_by_id.get(variant_id, {}))
        adjustment = deepcopy(self._get_adjustment(variant_id))
        ink_config = deepcopy(self._current_ink_config(variant_id))

        def render() -> object:
            preview = self._adjustment_service.preview_coordinated(
                detail,
                adjustment,
                self.WORK_RATIO,
                ink_config,
            )
            if preview is None:
                return None
            try:
                return self._pil_to_qimage(preview.image), preview.bounds
            finally:
                preview.image.close()

        worker = FunctionWorker(render)
        worker.signals.finished.connect(
            lambda result, target=variant_id, key=cache_key, task=worker: (
                self._card_preview_finished(target, key, result, task)
            )
        )
        worker.signals.failed.connect(
            lambda _message, target=variant_id, key=cache_key, task=worker: (
                self._card_preview_failed(target, key, task)
            )
        )
        self._preview_workers[variant_id] = (cache_key, worker)
        self._preview_pool.start(worker)

    def _card_preview_finished(
        self,
        variant_id: str,
        cache_key: tuple[Any, ...],
        result: object,
        worker: FunctionWorker,
    ) -> None:
        pending = self._preview_workers.get(variant_id)
        if pending is None or pending[1] is not worker:
            return
        self._preview_workers.pop(variant_id, None)
        if cache_key != self._coordinated_preview_cache_key(variant_id):
            return
        if (
            not isinstance(result, tuple)
            or len(result) != 2
            or not isinstance(result[0], QImage)
        ):
            return
        image, bounds = result
        self._store_preview(variant_id, cache_key, image, bounds)
        card = self._cards.get(variant_id)
        if card is not None:
            card.set_preview(image, bounds)
            self._set_list_thumbnail_from_preview(variant_id, image)

    def _card_preview_failed(
        self,
        variant_id: str,
        cache_key: tuple[Any, ...],
        worker: FunctionWorker,
    ) -> None:
        pending = self._preview_workers.get(variant_id)
        if pending is not None and pending == (cache_key, worker):
            self._preview_workers.pop(variant_id, None)

    def _select_variant(self, variant_id: str) -> None:
        if not variant_id or variant_id not in self._variant_by_id:
            return
        if variant_id != self._selected_id:
            self._finish_pending_comparison_transform()
        selection_changed = variant_id != self._selected_id
        self._selected_id = variant_id
        target_index = self._variant_index(variant_id)
        if target_index >= 0:
            target_page = target_index // self._page_size()
            if target_page != self._page_index:
                self._page_index = target_page
                self._render_page()
            target_item = self._list_items_by_id.get(variant_id)
            if target_item is not None:
                self._glyph_list.blockSignals(True)
                if target_item.parent() is not None:
                    target_item.parent().setExpanded(True)
                self._glyph_list.setCurrentItem(target_item)
                self._glyph_list.scrollToItem(target_item)
                self._glyph_list.blockSignals(False)
        if not self._reference_pin_button.isChecked() and self._reference_variant_id == variant_id:
            self._reference_variant_id = ""
            self._populate_reference_combo()
        if selection_changed or not self._detail_canvas.has_image:
            self._load_detail_canvas(variant_id)
        self._refresh_selection()

    def _select_list_item(
        self,
        current: Optional[QTreeWidgetItem],
        previous: Optional[QTreeWidgetItem],
    ) -> None:
        if current is None:
            return
        variant_id = str(current.data(0, Qt.ItemDataRole.UserRole) or "")
        if variant_id:
            if variant_id not in self._variant_by_id:
                detail = self._list_variant_by_id.get(variant_id, {})
                status = self._workflow_status(detail).stage if detail else "前序阶段"
                char = str(detail.get("归属字", "")) if detail else ""
                self._status_label.setText(
                    f"{char} · {status}　需先完成前序阶段，当前仅供查看"
                )
                self._schedule_list_thumbnail_loads()
                return
            self._select_variant(variant_id)
            self._schedule_list_thumbnail_loads()
            return

        previous_id = (
            str(previous.data(0, Qt.ItemDataRole.UserRole) or "")
            if previous is not None
            else ""
        )
        fallback = self._list_items_by_id.get(previous_id)
        if fallback is None:
            fallback = self._list_items_by_id.get(self._selected_id)
        if fallback is None and current.childCount():
            fallback = current.child(0)
        if fallback is None:
            return

        fallback_id = str(fallback.data(0, Qt.ItemDataRole.UserRole) or "")
        group_was_expanded = current.isExpanded()
        with QSignalBlocker(self._glyph_list):
            self._glyph_list.setCurrentItem(fallback)
            current.setExpanded(group_was_expanded)
        if fallback_id and fallback_id != self._selected_id:
            self._select_variant(fallback_id)

    def _show_glyph_context_menu(self, position: object) -> None:
        node = self._glyph_list.itemAt(position)
        if node is None:
            return
        variant_id = str(node.data(0, Qt.ItemDataRole.UserRole) or "")
        if not variant_id or variant_id not in self._variant_by_id:
            return
        self._glyph_list.setCurrentItem(node)
        menu = QMenu(self)
        action = menu.addAction("修正字形名称…")
        action.setEnabled(
            not self._coordination_busy and not self._baseline_analysis_pending
        )
        action.triggered.connect(self._rename_current_glyph)
        menu.exec(self._glyph_list.viewport().mapToGlobal(position))

    def _rename_current_glyph(self) -> None:
        if not self._selected_id:
            QMessageBox.information(self, "修正字形名称", "请先选择一个具体字形。")
            return
        if self._coordination_busy or self._baseline_analysis_pending:
            QMessageBox.information(
                self,
                "暂时不能修改名称",
                "当前正在分析或批量协调字库，请等待任务结束后重试。",
            )
            return
        if not self._confirm_leave_changes("继续修正字形名称"):
            return
        variant_id = self._selected_id
        result = run_glyph_rename_dialog(self, self._glyph, variant_id)
        if result is None:
            return
        self._preview_timer.stop()
        self._clear_preview_cache()
        self._list_thumbnail_timer.stop()
        self._list_thumbnail_cache.clear()
        self._list_thumbnail_key_by_variant.clear()
        self._reload_variants()
        self.summary_changed.emit(self._glyph)
        QMessageBox.information(
            self,
            "名称修改完成",
            f"字形已修正为“{result.get('新归属字', '')}”，各阶段文件名已同步更新。",
        )

    def _open_list_item(self, item: QTreeWidgetItem, _column: int) -> None:
        variant_id = str(item.data(0, Qt.ItemDataRole.UserRole) or "")
        if variant_id in self._variant_by_id:
            self._enter_detail(variant_id)

    def _refresh_selection(self) -> None:
        for variant_id, card in self._cards.items():
            card.set_selected(variant_id == self._selected_id)
            if variant_id != self._selected_id:
                card.set_control_polygon(None)
        self._sync_selected_card_controls(
            live=self._comparison_transform_active,
        )
        self._refresh_current_labels()

    def _sync_selected_card_controls(self, *, live: bool) -> None:
        """把精调画布的真实四边形同步到当前比较卡片。"""
        card = self._cards.get(self._selected_id)
        if card is None:
            return
        if not self._detail_canvas.has_image:
            card.set_control_polygon(None)
            return
        polygon, _handles, _rotate = self._detail_canvas.transform_controls_in_view(
            QPointF(),
            1.0,
        )
        if live:
            card.set_live_control_polygon(polygon)
        else:
            card.set_transform(self._detail_canvas.transform())
            card.set_control_polygon(polygon)

    def _comparison_card_hit_test(
        self,
        card: GlyphPreviewCard,
        position: QPointF,
        modifiers: Qt.KeyboardModifier,
    ) -> str:
        if (
            card.variant_id != self._selected_id
            or not self._detail_canvas.has_image
        ):
            return ""
        origin, scale = card.transform_view()
        return self._detail_canvas.transform_hit_test_in_view(
            position,
            origin,
            scale,
            modifiers,
        )

    def _enter_detail(self, variant_id: str) -> None:
        self._finish_pending_comparison_transform()
        if not variant_id or variant_id not in self._variant_by_id:
            self._comparison_mode_button.setChecked(True)
            return
        if variant_id != self._selected_id:
            self._select_variant(variant_id)
        elif not self._detail_canvas.has_image:
            self._load_detail_canvas(variant_id)
        self._view_stack.setCurrentWidget(self._detail_view)
        self._detail_mode_button.setChecked(True)
        self._save_button.setText("保存本字")
        self._save_next_button.show()
        self._restore_button.setEnabled(bool(self._selected_id))
        QTimer.singleShot(0, self._detail_canvas.fit_to_view)

    def _leave_detail(self) -> None:
        self._detail_canvas.set_source_preview_visible(False)
        self._view_stack.setCurrentWidget(self._comparison_view)
        self._comparison_mode_button.setChecked(True)
        self._save_button.setText("保存本页")
        self._save_next_button.hide()
        if self._selected_id:
            self._update_card(self._selected_id)
        self._schedule_capacity_update()

    def _load_detail_canvas(self, variant_id: str) -> None:
        detail = self._variant_by_id.get(variant_id)
        if detail is None:
            self._clear_detail()
            return
        source, path = self._adjustment_service.load_reviewed_source(detail)
        if source is None:
            self._clear_detail()
            self._detail_source_label.setText("来源：审核图像无法读取")
            return
        ink_config = self._current_ink_config(variant_id)
        working = self._adjustment_service.prepare_ink_working_copy(
            source,
            ink_config,
        )
        bounding_box = working.getchannel("A").getbbox()
        content_size = (
            max(1, bounding_box[2] - bounding_box[0]),
            max(1, bounding_box[3] - bounding_box[1]),
        ) if bounding_box else (1, 1)
        self._content_sizes[variant_id] = content_size

        saved = deepcopy(self._saved_adjustments.get(variant_id, self._default_adjustment()))
        current = deepcopy(self._get_adjustment(variant_id))
        saved_canvas = self._adjustment_service.coordination_to_canvas_transform(saved, content_size)
        current_canvas = self._adjustment_service.coordination_to_canvas_transform(current, content_size)
        canonical_saved = self._adjustment_service.coordination_from_canvas_transform(saved_canvas)
        canonical_current = self._adjustment_service.coordination_from_canvas_transform(current_canvas)
        self._saved_adjustments[variant_id] = deepcopy(canonical_saved)
        self._saved_signatures[variant_id] = self._adjustment_signature(canonical_saved)
        self._adjustments[variant_id] = deepcopy(canonical_current)

        self._loading_detail = True
        try:
            source_image = self._pil_to_qimage(source)
            self._detail_canvas.set_image(
                self._pil_to_qimage(working),
                (self._canvas_width, self._canvas_height),
                source_preview=source_image,
            )
            self._set_detail_ink_postprocessor(variant_id, ink_config)
            self._detail_canvas.set_tool(ReviewCanvas.TOOL_TRANSFORM)
            if saved_canvas != self._identity_canvas_transform():
                self._detail_canvas.set_transform(**saved_canvas)
            self._detail_canvas.set_saved_baseline(bake=False)
            if current_canvas != saved_canvas:
                self._detail_canvas.set_transform(**current_canvas)
        finally:
            self._loading_detail = False
            working.close()
            source.close()
        self._sync_transform_controls(self._detail_canvas.transform())
        self._sync_selected_card_controls(live=False)
        self._update_reference_overlay()
        stage = "手工审核稿" if os.path.dirname(path) == self._glyph.get_workflow_dirs()["手工审核"] else "自动优化稿"
        self._detail_source_label.setText(f"来源：{stage} · {os.path.basename(path)}")
        self._refresh_current_labels()

    def _clear_detail(self) -> None:
        self._loading_detail = True
        try:
            self._detail_canvas.clear_image()
        finally:
            self._loading_detail = False
        self._current_char_label.setText("-")
        self._current_file_label.setText("未选择字形")
        self._current_index_label.setText("-")
        self._detail_position_label.setText("0 / 0")
        self._detail_source_label.setText("来源：-")
        self._mode_status_label.setText("请选择字形")
        self._dirty_label.setText("未选择")
        self._ink_result_label.setText("本字墨色：未选择字形")
        with QSignalBlocker(self._ink_strategy_combo):
            self._ink_strategy_combo.setCurrentText(self.INK_MODES[0])
        self._ink_strategy_combo.setEnabled(False)
        self._set_transform_controls_enabled(False)

    def _on_detail_transform_changed(self, transform: dict[str, Any]) -> None:
        if self._loading_detail or not self._selected_id:
            return
        was_dirty = self._is_dirty(self._selected_id)
        self._adjustments[self._selected_id] = (
            self._adjustment_service.coordination_from_canvas_transform(transform)
        )
        is_dirty = self._is_dirty(self._selected_id)
        self._sync_transform_controls(transform)
        if self._comparison_transform_active:
            card = self._cards.get(self._selected_id)
            if card is not None:
                self._sync_selected_card_controls(live=True)
        elif was_dirty != is_dirty:
            self._refresh_filtered_view(
                preserve_selection=True,
                reload_detail=False,
            )
        else:
            self._refresh_current_labels()
            self._render_status()
        if not self._comparison_transform_active:
            self._schedule_selected_preview_refresh()

    def _begin_comparison_transform(
        self,
        variant_id: str,
        position: object,
        modifiers: object,
    ) -> None:
        wheel_handoff = False
        start_dirty = self._is_dirty(variant_id)
        carried_change = False
        if self._comparison_transform_active:
            if self._comparison_transform_mode == "wheel":
                wheel_handoff = True
                start_dirty = self._comparison_transform_start_dirty
                carried_change = self._comparison_transform_changed
                self._comparison_wheel_timer.stop()
            else:
                return
        if variant_id != self._selected_id:
            self._select_variant(variant_id)
        if not self._detail_canvas.has_image:
            self._load_detail_canvas(variant_id)
        if not self._detail_canvas.has_image:
            return
        card = self._cards.get(variant_id)
        if card is None or not isinstance(position, QPointF):
            return
        origin, scale = card.transform_view()
        modifier_flags = (
            modifiers
            if isinstance(modifiers, Qt.KeyboardModifier)
            else Qt.KeyboardModifier.NoModifier
        )
        kind = self._detail_canvas.begin_external_transform(
            position,
            origin,
            scale,
            modifier_flags,
        )
        if not kind:
            if wheel_handoff:
                self._comparison_wheel_timer.start()
            else:
                card.finish_live_preview()
            return
        self._preview_timer.stop()
        self._comparison_transform_active = True
        self._comparison_transform_mode = "drag"
        self._comparison_transform_changed = carried_change
        self._comparison_transform_start_dirty = start_dirty

    def _apply_comparison_transform(
        self,
        variant_id: str,
        position: object,
        modifiers: object,
    ) -> None:
        if (
            not self._comparison_transform_active
            or self._comparison_transform_mode != "drag"
            or variant_id != self._selected_id
            or not isinstance(position, QPointF)
        ):
            return
        modifier_flags = (
            modifiers
            if isinstance(modifiers, Qt.KeyboardModifier)
            else Qt.KeyboardModifier.NoModifier
        )
        changed = self._detail_canvas.update_external_transform(
            position,
            modifier_flags,
        )
        if changed:
            self._comparison_transform_changed = True

    def _finish_comparison_transform(self, variant_id: str) -> None:
        if (
            not self._comparison_transform_active
            or self._comparison_transform_mode != "drag"
            or variant_id != self._selected_id
        ):
            return
        changed = self._detail_canvas.end_external_transform()
        self._comparison_transform_changed = (
            self._comparison_transform_changed or changed
        )
        self._complete_comparison_transform(variant_id)

    def _apply_comparison_wheel(self, variant_id: str, delta: float) -> None:
        if variant_id != self._selected_id or not math.isfinite(delta) or delta == 0.0:
            return
        if self._comparison_transform_active:
            if self._comparison_transform_mode != "wheel":
                return
        else:
            if not self._detail_canvas.has_image:
                self._load_detail_canvas(variant_id)
            if not self._detail_canvas.has_image:
                return
            card = self._cards.get(variant_id)
            if card is None:
                return
            self._preview_timer.stop()
            card.begin_live_preview()
            self._comparison_transform_active = True
            self._comparison_transform_mode = "wheel"
            self._comparison_transform_changed = False
            self._comparison_transform_start_dirty = self._is_dirty(variant_id)

        current = self._detail_canvas.transform()
        scale = float(current.get("scale", 1.0))
        updated = max(0.05, min(5.0, scale * math.pow(1.05, delta / 120.0)))
        changed = self._detail_canvas.set_transform(
            scale=updated,
            record_undo=not self._comparison_transform_changed,
        )
        if changed:
            self._comparison_transform_changed = True
        self._comparison_wheel_timer.start()

    def _finish_comparison_wheel_transform(self) -> None:
        if (
            not self._comparison_transform_active
            or self._comparison_transform_mode != "wheel"
        ):
            return
        self._comparison_wheel_timer.stop()
        self._complete_comparison_transform(self._selected_id)

    def _finish_pending_comparison_transform(self) -> None:
        if not self._comparison_transform_active:
            return
        if self._comparison_transform_mode == "drag":
            changed = self._detail_canvas.end_external_transform()
            self._comparison_transform_changed = (
                self._comparison_transform_changed or changed
            )
            self._complete_comparison_transform(self._selected_id)
            return
        self._finish_comparison_wheel_transform()

    def _complete_comparison_transform(self, variant_id: str) -> None:
        start_dirty = self._comparison_transform_start_dirty
        changed = self._comparison_transform_changed
        self._comparison_transform_active = False
        self._comparison_transform_mode = ""
        self._comparison_transform_changed = False
        self._comparison_transform_start_dirty = False
        card = self._cards.get(variant_id)
        if card is not None:
            card.cancel_transform_interaction()
        if not changed:
            self._sync_selected_card_controls(live=False)
            return
        self._preview_timer.stop()
        if start_dirty != self._is_dirty(variant_id):
            self._refresh_filtered_view(
                preserve_selection=True,
                reload_detail=False,
            )
        else:
            self._update_card(variant_id)
            self._refresh_current_labels()
            self._render_status()

    def _schedule_selected_preview_refresh(self) -> None:
        if self._comparison_transform_active:
            return
        self._preview_timer.start()

    def _refresh_selected_preview(self) -> None:
        if not self._comparison_transform_active and self._selected_id in self._cards:
            self._update_card(self._selected_id)

    def _apply_transform_field(self, field: str, value: float) -> None:
        if self._updating_controls or not self._detail_canvas.has_image:
            return
        self._finish_pending_comparison_transform()
        if not self._detail_canvas.set_transform(**{field: value}):
            self._sync_transform_controls(self._detail_canvas.transform())

    def _apply_distort_field(self, index: int, value: float) -> None:
        if self._updating_controls or not self._detail_canvas.has_image:
            return
        self._finish_pending_comparison_transform()
        transform = self._detail_canvas.transform()
        distort = list(transform.get("distort", [0.0] * 8))
        if len(distort) != 8:
            distort = [0.0] * 8
        distort[index] = float(value)
        if not self._detail_canvas.set_transform(distort=distort):
            self._sync_transform_controls(self._detail_canvas.transform())

    def _reset_distortion(self) -> None:
        if self._detail_canvas.has_image:
            self._finish_pending_comparison_transform()
            self._detail_canvas.set_transform(distort=[0.0] * 8)

    def _sync_transform_controls(self, transform: dict[str, Any]) -> None:
        self._updating_controls = True
        try:
            with (
                QSignalBlocker(self._offset_x_spin),
                QSignalBlocker(self._offset_y_spin),
                QSignalBlocker(self._scale_slider),
                QSignalBlocker(self._stretch_w_slider),
                QSignalBlocker(self._stretch_h_slider),
                QSignalBlocker(self._rotation_slider),
            ):
                self._offset_x_spin.setValue(round(self._number(transform.get("x"), 0.0)))
                self._offset_y_spin.setValue(round(self._number(transform.get("y"), 0.0)))
                self._scale_slider.setValue(
                    self._percent_to_slider_position(self._number(transform.get("scale"), 1.0) * 100)
                )
                self._stretch_w_slider.setValue(
                    self._percent_to_slider_position(self._number(transform.get("stretch_w"), 1.0) * 100)
                )
                self._stretch_h_slider.setValue(
                    self._percent_to_slider_position(self._number(transform.get("stretch_h"), 1.0) * 100)
                )
                self._rotation_slider.setValue(round(self._number(transform.get("rotation"), 0.0)))
            distort = transform.get("distort", [0.0] * 8)
            if not isinstance(distort, (list, tuple)) or len(distort) != 8:
                distort = [0.0] * 8
            for spin, value in zip(self._distort_spins, distort):
                with QSignalBlocker(spin):
                    spin.setValue(float(value))
            self._update_transform_value_labels()
        finally:
            self._updating_controls = False

    def _on_percent_slider_changed(self, slider: QSlider, field: str) -> None:
        self._update_transform_value_labels()
        if not slider.isSliderDown():
            self._apply_transform_field(field, self._slider_position_to_percent(slider.value()) / 100.0)

    def _commit_percent_slider(self, slider: QSlider, field: str) -> None:
        self._update_transform_value_labels()
        self._apply_transform_field(field, self._slider_position_to_percent(slider.value()) / 100.0)

    def _on_rotation_slider_changed(self, value: int) -> None:
        self._rotation_value_label.setText(f"{value}°")
        if not self._rotation_slider.isSliderDown():
            self._apply_transform_field("rotation", float(value))

    def _update_transform_value_labels(self) -> None:
        self._scale_value_label.setText(f"{self._slider_position_to_percent(self._scale_slider.value())}%")
        self._stretch_w_value_label.setText(
            f"{self._slider_position_to_percent(self._stretch_w_slider.value())}%"
        )
        self._stretch_h_value_label.setText(
            f"{self._slider_position_to_percent(self._stretch_h_slider.value())}%"
        )
        self._rotation_value_label.setText(f"{self._rotation_slider.value()}°")

    def _toggle_advanced_panel(self, visible: bool) -> None:
        self._advanced_button.setArrowType(
            Qt.ArrowType.DownArrow if visible else Qt.ArrowType.RightArrow
        )
        self._advanced_panel.setVisible(visible)

    def _reference_changed(self, index: int) -> None:
        self._reference_variant_id = str(self._reference_combo.itemData(index) or "")
        self._reference_overlay_check.setEnabled(bool(self._reference_variant_id))
        if not self._reference_variant_id:
            self._reference_overlay_check.setChecked(False)
        self._update_reference_overlay()

    def _update_reference_overlay(self, _checked: object = None) -> None:
        if (
            not self._reference_variant_id
            or not self._reference_overlay_check.isChecked()
            or self._reference_variant_id not in self._variant_by_id
        ):
            self._detail_canvas.set_reference_visible(False)
            if not self._reference_variant_id:
                self._detail_canvas.set_reference_image(None)
            return
        detail = self._variant_by_id[self._reference_variant_id]
        preview = self._adjustment_service.preview_coordinated(
            detail,
            self._get_adjustment(self._reference_variant_id),
            self.WORK_RATIO,
            self._current_ink_config(self._reference_variant_id),
        )
        if preview is None:
            self._detail_canvas.set_reference_image(None)
            return
        self._detail_canvas.set_reference_image(self._pil_to_qimage(preview[0]), opacity=0.35)
        self._detail_canvas.set_reference_visible(True)

    def _ink_mode_changed(self, _checked: bool) -> None:
        self._finish_pending_comparison_transform()
        self._clear_preview_cache()
        self._refresh_filtered_view(
            preserve_selection=True,
            reload_detail=True,
        )
        self._update_reference_overlay()

    def _set_detail_ink_postprocessor(
        self,
        variant_id: str,
        ink_config: dict[str, Any],
    ) -> None:
        if not bool(ink_config.get("启用", False)):
            self._detail_canvas.set_render_postprocessor(None)
            return
        frozen_config = deepcopy(ink_config)

        def process_rendered(pixels: np.ndarray) -> np.ndarray:
            source = Image.fromarray(pixels, "RGBA")
            output: Image.Image | None = None
            try:
                output, _record = self._adjustment_service.apply_ink_preview(
                    source,
                    frozen_config,
                    variant_id,
                )
                return np.array(output.convert("RGBA"), dtype=np.uint8, copy=True)
            except Exception:
                return np.ascontiguousarray(pixels, dtype=np.uint8)
            finally:
                if output is not None:
                    output.close()
                source.close()

        self._detail_canvas.set_render_postprocessor(process_rendered)

    def _ink_strategy_changed(self, mode: str) -> None:
        if self._updating_controls or not self._selected_id or mode not in self.INK_MODES:
            return
        variant_id = self._selected_id
        previous = self._ink_modes.get(variant_id, self.INK_MODES[0])
        if previous == mode:
            return
        self._finish_pending_comparison_transform()
        self._ink_modes[variant_id] = mode
        self._clear_variant_preview_cache(variant_id)
        self._refresh_filtered_view(
            preserve_selection=True,
            reload_detail=True,
        )
        self._update_reference_overlay()

    def _reset_selected(self) -> None:
        if not self._selected_id:
            return
        self._finish_pending_comparison_transform()
        self._adjustments[self._selected_id] = self._default_adjustment()
        self._load_detail_canvas(self._selected_id)
        self._update_card(self._selected_id)
        self._populate_list()
        self._render_status()

    def _signature(self, variant_id: str) -> tuple[tuple[str, Any], ...]:
        return self._adjustment_signature(self._get_adjustment(variant_id))

    @staticmethod
    def _adjustment_signature(adjustment: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
        values: list[tuple[str, Any]] = []
        for key, value in adjustment.items():
            if isinstance(value, (list, tuple)):
                normalized: Any = tuple(round(float(item), 6) for item in value)
            else:
                try:
                    normalized = round(float(value), 6)
                except (TypeError, ValueError):
                    normalized = str(value)
            values.append((key, normalized))
        return tuple(sorted(values))

    def _zero_signature(self) -> tuple[tuple[str, Any], ...]:
        return self._adjustment_signature(self._default_adjustment())

    def _is_dirty(self, variant_id: str) -> bool:
        geometry_dirty = self._signature(variant_id) != self._saved_signatures.get(
            variant_id, self._zero_signature()
        )
        current_ink_signature = self._ink_signature(
            self._current_ink_config(variant_id)
        )
        ink_dirty = current_ink_signature != self._saved_ink_signatures.get(
            variant_id,
            current_ink_signature,
        )
        return geometry_dirty or ink_dirty

    def _save_action(self) -> None:
        if self._view_stack.currentWidget() is self._detail_view:
            self._save_selected()
        else:
            self._save_current_page()

    def _save_selected(self, show_success: bool = True) -> bool:
        detail = self._variant_by_id.get(self._selected_id)
        if detail is None:
            return True
        return self._save_variants([detail], show_success=show_success, title="保存本字")

    def _save_current_page(self, show_success: bool = True) -> bool:
        saved_page_index = self._page_index
        page_variants = self._page_variants()
        next_page_start = (saved_page_index + 1) * self._page_size()
        next_variant_id = (
            str(self._variants[next_page_start].get("变体ID", ""))
            if next_page_start < len(self._variants)
            else ""
        )
        saved = self._save_variants(
            page_variants,
            show_success=show_success,
            title="保存本页",
        )
        if not saved:
            return False

        next_variant_index = self._variant_index(next_variant_id)
        if next_variant_index >= 0:
            self._select_variant(next_variant_id)
            return True

        if self._variants:
            last_page_index = math.ceil(len(self._variants) / self._page_size()) - 1
            target_page_index = min(saved_page_index, last_page_index)
            selected_index = self._variant_index(self._selected_id)
            selected_page_index = (
                selected_index // self._page_size() if selected_index >= 0 else -1
            )
            if selected_page_index != target_page_index:
                target_index = target_page_index * self._page_size()
                target_variant_id = str(
                    self._variants[target_index].get("变体ID", "")
                )
                if target_variant_id:
                    self._select_variant(target_variant_id)
        return saved

    def _save_variants(
        self,
        variants: list[dict[str, Any]],
        *,
        show_success: bool,
        title: str,
    ) -> bool:
        if self._coordination_busy:
            return False
        self._finish_pending_comparison_transform()
        if not variants:
            return True
        requested_ids = {
            str(detail.get("变体ID", "")) for detail in variants
        }
        variants_to_save = list(variants)
        adjustments_to_save = {
            variant_id: deepcopy(self._get_adjustment(variant_id))
            for variant_id in requested_ids
        }
        self._save_button.setEnabled(False)
        self._complete_button.setEnabled(False)
        try:
            result = self._adjustment_service.save_coordinated_variants(
                variants_to_save,
                adjustments_to_save,
                self._current_ink_config(variant_ids=requested_ids),
                self._coordination_baseline,
            )
        except Exception as exc:
            self._reload_variants()
            QMessageBox.critical(self, title, f"整体协调结果保存失败：{exc}")
            return False
        finally:
            self._save_button.setEnabled(True)
            self._complete_button.setEnabled(True)

        failed_ids = {str(item[0]) for item in result["失败详情"]}
        successful_ids: list[str] = []
        for detail in variants_to_save:
            variant_id = str(detail.get("变体ID", ""))
            if variant_id in failed_ids:
                continue
            successful_ids.append(variant_id)
            saved = self._adjustment_service.load_saved_coordination_adjustments(detail)
            self._adjustments[variant_id] = deepcopy(saved)
            self._saved_adjustments[variant_id] = deepcopy(saved)
            self._saved_signatures[variant_id] = self._adjustment_signature(saved)
            self._ink_modes[variant_id] = self._stored_ink_mode(detail)
            self._saved_ink_signatures[variant_id] = self._stored_ink_signature(detail)
        if (
            self._selected_id in successful_ids
            and self._selected_id in requested_ids
            and self._detail_canvas.has_image
        ):
            self._detail_canvas.set_saved_baseline(bake=False)

        self._clear_preview_cache()
        self._reload_variants()

        if result["失败"]:
            details = "\n".join(
                f"{variant_id}：{reason}" for variant_id, reason in result["失败详情"][:8]
            )
            QMessageBox.critical(
                self,
                title,
                f"成功 {result['成功']} 个，失败 {result['失败']} 个。\n\n{details}",
            )
            return False
        self._initial_ink_enabled = self._ink_check.isChecked()
        self.summary_changed.emit(self._glyph)
        if show_success:
            QMessageBox.information(
                self,
                title,
                f"已保存 {len(requested_ids)} 个字形。",
            )
        return True

    def _save_and_next(self) -> None:
        if not self._selected_id:
            return
        current_id = self._selected_id
        next_id = self._next_navigation_id(current_id)
        if not self._save_selected(show_success=False):
            return
        target_id = next_id if self._variant_index(next_id) >= 0 else current_id
        if target_id in self._variant_by_id:
            self._select_variant(target_id)
            self._enter_detail(target_id)

    def _complete_coordination(self) -> None:
        self._finish_pending_comparison_transform()
        if (
            self._coordination_busy
            or self._baseline_analysis_pending
            or not self._all_variants
        ):
            return
        pending_variants = [
            detail
            for detail in self._all_variants
            if self._stage_projection(detail).matches_status(
                STAGE_PENDING_COORDINATION
            )
        ]
        if not pending_variants:
            return
        variant_ids = [
            str(detail.get("变体ID", "")) for detail in pending_variants
        ]
        ink_config = deepcopy(self._current_ink_config(variant_ids=set(variant_ids)))
        ink_config["重算几何后基准"] = True
        zero_signature = self._zero_signature()
        default_count = sum(
            self._adjustment_signature(self._get_adjustment(variant_id))
            == zero_signature
            for variant_id in variant_ids
        )
        if not self._confirm_complete_coordination(
            len(variant_ids),
            default_count,
            bool(ink_config.get("启用", False)),
        ):
            return
        task = _CoordinationTask(
            self._glyph.ziku_name,
            self._glyph.ziku_dir,
            self._glyph.snapshot_state(),
            variant_ids,
            deepcopy(self._adjustments),
            ink_config,
            deepcopy(self._coordination_baseline),
        )
        self._coordination_task = task
        self._coordination_task_total = len(variant_ids)
        self._coordination_task_ink = ink_config
        task.signals.progress.connect(self._coordination_progress_changed)
        task.signals.finished.connect(self._coordination_finished)
        task.signals.failed.connect(self._coordination_failed)

        self._preview_timer.stop()
        self._list_thumbnail_timer.stop()
        self._task_progress_panel.show()
        self._task_stage_label.setText("本次执行：准备")
        # 此进度条也用于载入时的全库基准分析，后者按字形总数设置量程。
        # 批处理使用全局百分比，启动前必须恢复百分比量程。
        self._task_progress_bar.setRange(0, 100)
        self._task_progress_bar.setValue(0)
        self._task_progress_bar.setFormat("本次执行 %p%")
        self._task_detail_label.setText(f"0 / {len(variant_ids)} · 正在核对批次")
        self._task_detail_label.setToolTip("")
        self._set_coordination_busy(True)
        self._set_coordination_stop_state(running=True)
        try:
            self._coordination_pool.start(task)
        except Exception as exc:
            self._coordination_failed(str(exc))

    def _confirm_complete_coordination(
        self,
        total: int,
        default_count: int,
        ink_enabled: bool,
    ) -> bool:
        dialog = QMessageBox(self)
        dialog.setWindowTitle("确认批量整体协调")
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setText(
            f"将批量处理当前字库的 {total} 个字形，"
            f"其中 {default_count} 个使用默认变换参数。"
        )
        ink_message = (
            "当前已启用墨色统一，将对全部字形批量调整墨色。"
            if ink_enabled
            else "当前未启用墨色统一；启用后会对全部字形批量调整墨色。"
        )
        dialog.setInformativeText(
            "请确认以下风险：\n"
            "1. 未逐字调整的字形将使用默认参数生成。\n"
            "2. 批量处理不能替代逐字视觉比较。\n"
            f"3. {ink_message}\n"
            "4. 当前页面草稿会参与处理；整批成功后会更新已有成品和字库索引。\n"
            "任一失败或提交前停止均不会保留本批部分结果。"
        )
        dialog.setStandardButtons(
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
        )
        dialog.setDefaultButton(QMessageBox.StandardButton.Cancel)
        confirm_button = dialog.button(QMessageBox.StandardButton.Ok)
        cancel_button = dialog.button(QMessageBox.StandardButton.Cancel)
        if confirm_button is not None:
            confirm_button.setText("确定")
        if cancel_button is not None:
            cancel_button.setText("取消")
            dialog.setEscapeButton(cancel_button)
        return dialog.exec() == QMessageBox.StandardButton.Ok.value

    def _request_stop_coordination(self) -> None:
        task = self._coordination_task
        if task is None or task.is_cancel_requested():
            return
        if task.is_commit_started():
            self._show_coordination_committing()
            return
        if not self._confirm_stop_coordination():
            return
        if (
            self._coordination_task is not task
            or not self._coordination_busy
            or task.is_cancel_requested()
        ):
            return
        if not task.request_cancel():
            if self._coordination_task is task and self._coordination_busy:
                self._show_coordination_committing()
            return
        self._set_coordination_stop_state(running=True, stopping=True)
        self._task_stage_label.setText("本次执行：正在停止")
        self._task_progress_bar.setFormat("正在停止… %p%")
        detail = "正在停止整体协调，请等待当前操作安全结束……"
        self._task_detail_label.setText(detail)
        self._task_detail_label.setToolTip(detail)

    def _confirm_stop_coordination(self) -> bool:
        dialog = QMessageBox(self)
        dialog.setWindowTitle("停止整体协调")
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setText("确定停止当前整体协调批次吗？")
        dialog.setInformativeText(
            "停止请求会在最终提交前的安全点生效，本批次不会提交任何成品或字库索引。\n"
            "已有成品和当前页面草稿保持不变。若已进入最终提交，将完整提交或回滚。"
        )
        stop_button = dialog.addButton(
            "停止整体协调",
            QMessageBox.ButtonRole.AcceptRole,
        )
        continue_button = dialog.addButton(
            "继续运行",
            QMessageBox.ButtonRole.RejectRole,
        )
        dialog.setDefaultButton(continue_button)
        dialog.setEscapeButton(continue_button)
        dialog.exec()
        return dialog.clickedButton() is stop_button

    def _show_coordination_committing(self) -> None:
        self._set_coordination_stop_state(running=True, committing=True)
        self._task_stage_label.setText("本次执行：提交")
        self._task_progress_bar.setFormat("本次执行 %p%")
        detail = "已进入最终提交，停止请求未受理；请等待完整提交或回滚。"
        self._task_detail_label.setText(detail)
        self._task_detail_label.setToolTip(detail)

    def _set_coordination_stop_state(
        self,
        running: bool,
        stopping: bool = False,
        committing: bool = False,
    ) -> None:
        self._stop_coordination_button.setVisible(running)
        self._stop_coordination_button.setEnabled(
            running and not stopping and not committing
        )
        if committing:
            self._stop_coordination_button.setText("正在提交…")
        else:
            self._stop_coordination_button.setText(
                "正在停止…" if stopping else "停止整体协调"
            )

    def _coordination_progress_changed(
        self,
        stage: str,
        percent: int,
        current: int,
        total: int,
        glyph_label: str,
    ) -> None:
        if not self._coordination_busy:
            return
        task = self._coordination_task
        if task is not None and task.is_commit_started() and stage != "提交":
            self._show_coordination_committing()
            return
        if task is not None and task.is_cancel_requested():
            self._set_coordination_stop_state(running=True, stopping=True)
            self._task_stage_label.setText("本次执行：正在停止")
            self._task_progress_bar.setFormat("正在停止… %p%")
            detail = "正在停止整体协调，请等待当前操作安全结束……"
            self._task_detail_label.setText(detail)
            self._task_detail_label.setToolTip(detail)
            return
        self._set_coordination_stop_state(
            running=True,
            committing=stage == "提交",
        )
        normalized_percent = max(0, min(100, int(percent)))
        normalized_total = max(0, int(total))
        normalized_current = max(0, min(int(current), normalized_total))
        self._task_stage_label.setText(f"本次执行：{stage}")
        self._task_progress_bar.setRange(0, 100)
        self._task_progress_bar.setValue(
            max(self._task_progress_bar.value(), normalized_percent)
        )
        self._task_progress_bar.setFormat("本次执行 %p%")
        detail = f"{normalized_current} / {normalized_total}"
        if glyph_label:
            detail += f" · {glyph_label}"
        available_width = max(180, self._task_detail_label.width())
        self._task_detail_label.setText(
            self._task_detail_label.fontMetrics().elidedText(
                detail,
                Qt.TextElideMode.ElideMiddle,
                available_width,
            )
        )
        self._task_detail_label.setToolTip(detail)

    def _coordination_finished(self, result: object) -> None:
        payload = result if isinstance(result, dict) else {}
        raw_data = payload.get("结果")
        data = raw_data if isinstance(raw_data, dict) else {}
        if bool(payload.get("已停止")) or bool(data.get("已停止")):
            self._finish_coordination_task(
                False,
                "已停止，本批次未提交",
                stopped=True,
            )
            QMessageBox.information(
                self,
                "整体协调已停止",
                "已停止，本批次未提交任何成品或字库索引。"
                "已有成品和页面草稿保持不变。",
            )
            return
        failed = self._result_count(data.get("失败"))
        success = self._result_count(data.get("成功"))
        if failed or success != self._coordination_task_total:
            if not failed:
                failed = max(1, self._coordination_task_total - success)
            reload_error = self._try_reload_after_coordination()
            self._finish_coordination_task(False, f"失败 {failed} 个")
            details = self._coordination_failure_text(data.get("失败详情"))
            message = f"本次批次未提交任何成品。\n成功 {success} 个，失败 {failed} 个。"
            if details:
                message += f"\n\n失败详情：\n{details}"
            if reload_error:
                message += f"\n\n页面刷新失败：{reload_error}。可返回首页后重新进入。"
            QMessageBox.critical(self, "完成整体协调失败", message)
            return

        worker_state = payload.get("字库状态")
        if not isinstance(worker_state, dict):
            state_error = str(
                payload.get("字库状态错误") or "后台任务未返回有效字库状态"
            )
            self._finish_coordination_task(False, "批次已提交，页面刷新失败")
            QMessageBox.critical(
                self,
                "整体协调页面刷新失败",
                f"全库成品已经提交，但后台状态快照失败：{state_error}。\n"
                "页面已经解锁，请返回首页后重新进入以读取最新结果。",
            )
            return
        try:
            self._glyph.restore_state(worker_state)
            self._clear_preview_cache()
            saved_summary = self._glyph.get_coordination_summary()
            self._coordination_baseline = self._saved_or_default_baseline(saved_summary)
            self._ink_baseline = self._number(
                self._coordination_baseline.get("墨色基准"),
                220.0,
            )
            ink_count = int(
                self._coordination_baseline.get("墨色有效数", 0) or 0
            )
            self._initial_ink_enabled = bool(
                saved_summary.get("墨色统一启用", False)
            )
            with QSignalBlocker(self._ink_check):
                self._ink_check.setChecked(self._initial_ink_enabled)
            self._ink_check.setEnabled(ink_count > 0)
            self._ink_baseline_label.setText(
                (
                    f"固定墨色基准：{self._ink_baseline:.2f}\n"
                    f"自动取样：{ink_count} 个协调样本，本次进入后保持固定"
                )
                if ink_count
                else "固定墨色基准：暂无有效字形"
            )
            self._all_variants = self._adjustment_service.load_reviewed_variants(
                pinyin_order=False
            )
            self._workflow_summary = self._glyph.get_coordination_summary()
            self._finished_dir = self._glyph.get_workflow_dirs()["成品"]
            self._workflow_status_cache.clear()
            self._variant_by_id = {
                str(item.get("变体ID", "")): item
                for item in self._all_variants
                if item.get("变体ID")
            }
            self._list_variants = [
                detail
                for detail in self._all_variants
                if self._stage_projection(detail).admitted
            ]
            admitted_ids = {
                str(detail.get("变体ID", ""))
                for detail in self._list_variants
            }
            self._all_variants = [
                detail
                for detail in self._all_variants
                if str(detail.get("变体ID", "")) in admitted_ids
            ]
            self._list_variant_by_id = {
                str(item.get("变体ID", "")): item
                for item in self._list_variants
                if item.get("变体ID")
            }
            self._variant_by_id = {
                str(item.get("变体ID", "")): item for item in self._all_variants
            }
            for detail in self._all_variants:
                variant_id = str(detail.get("变体ID", ""))
                self._ink_modes[variant_id] = self._stored_ink_mode(detail)
                saved = self._adjustment_service.load_saved_coordination_adjustments(detail)
                self._adjustments[variant_id] = deepcopy(saved)
                self._saved_adjustments[variant_id] = deepcopy(saved)
                self._saved_signatures[variant_id] = self._adjustment_signature(saved)
                self._saved_ink_signatures[variant_id] = self._stored_ink_signature(detail)
            self._apply_filters()
        except Exception as exc:
            self._finish_coordination_task(False, "批次已提交，页面刷新失败")
            QMessageBox.critical(
                self,
                "整体协调页面刷新失败",
                f"全库成品已经提交，但页面刷新失败：{exc}。\n"
                "返回首页后重新进入即可读取最新结果。",
            )
            return

        remaining = [
            self._stage_projection(detail)
            for detail in self._list_variants
            if self._stage_projection(detail).status == STAGE_PENDING_COORDINATION
        ]
        remaining_count = len(remaining)
        if remaining_count:
            ink_pending = sum(
                projection.has_marker(MARKER_INK_PENDING)
                for projection in remaining
            )
            file_errors = sum(
                projection.has_marker(MARKER_FILE_ERROR)
                for projection in remaining
            )
            reasons: list[str] = []
            if ink_pending:
                reasons.append(f"墨色待确认 {ink_pending} 个")
            if file_errors:
                reasons.append(f"文件异常 {file_errors} 个")
            classified_count = sum(
                projection.has_marker(MARKER_INK_PENDING)
                or projection.has_marker(MARKER_FILE_ERROR)
                for projection in remaining
            )
            other_count = remaining_count - classified_count
            if other_count:
                reasons.append(f"其他待核对 {other_count} 个")
            reason_text = "，".join(reasons) or f"待协调 {remaining_count} 个"
            coordinated_count = max(0, len(self._list_variants) - remaining_count)
            self._finish_coordination_task(
                True,
                (
                    f"{success} / {self._coordination_task_total} · "
                    f"成品已生成，仍有 {remaining_count} 个待核对"
                ),
                needs_review=True,
            )
            self.summary_changed.emit(self._glyph)
            QMessageBox.warning(
                self,
                "批量整体协调完成，需核对",
                f"本次已生成 {success} 个成品，处理进度为 100%。\n"
                f"当前已协调 {coordinated_count} 个，仍有 {remaining_count} 个待协调："
                f"{reason_text}。\n\n"
                "请在字形列表中核对这些字形；确认可接受的墨色例外后，"
                "再完成其协调状态。",
            )
            return

        self._finish_coordination_task(
            True,
            f"{success} / {self._coordination_task_total} · 批次提交完成",
        )
        self.summary_changed.emit(self._glyph)
        QMessageBox.information(self, "完成整体协调", "全库整体协调结果已生成。")

    def _coordination_failed(self, message: str) -> None:
        reload_error = self._try_reload_after_coordination()
        rollback_incomplete = "回滚未完全完成" in message
        if rollback_incomplete:
            self._finish_coordination_task(False, "回滚未完全完成")
            detail = (
                f"整体协调批次执行失败，且回滚未完全完成：{message}\n\n"
                "成品目录中可能存在未恢复文件，请返回首页并核对字库状态后再继续。"
            )
        else:
            self._finish_coordination_task(False, "事务已回滚")
            detail = f"整体协调批次执行失败，未保留部分结果：{message}"
        if reload_error:
            detail += f"\n\n页面刷新失败：{reload_error}。可返回首页后重新进入。"
        QMessageBox.critical(
            self,
            "完成整体协调失败",
            detail,
        )

    def _try_reload_after_coordination(self) -> str:
        """批次失败后尽力刷新；失败也不能阻断控件解锁。"""
        try:
            self._reload_variants()
        except Exception as exc:
            return str(exc) or type(exc).__name__
        return ""

    def _finish_coordination_task(
        self,
        succeeded: bool,
        detail: str,
        *,
        stopped: bool = False,
        needs_review: bool = False,
    ) -> None:
        total = self._coordination_task_total
        self._coordination_task = None
        self._set_coordination_busy(False)
        self._set_coordination_stop_state(running=False)
        if succeeded:
            self._task_stage_label.setText(
                "本次执行：已完成，需核对" if needs_review else "本次执行：已完成"
            )
            self._task_progress_bar.setRange(0, 100)
            self._task_progress_bar.setValue(100)
            self._task_progress_bar.setFormat("本次执行 %p%")
        elif stopped:
            self._task_stage_label.setText("本次执行：已停止")
            self._task_progress_bar.setFormat("本次执行已停止")
        else:
            self._task_stage_label.setText("本次执行：失败")
            self._task_progress_bar.setFormat("本次执行失败")
        self._task_detail_label.setText(detail or f"0 / {total}")
        self._task_detail_label.setToolTip(detail)
        self._coordination_task_total = 0
        self._coordination_task_ink = {}
        self._task_progress_panel.hide()
        self._schedule_list_thumbnail_loads()

    def _set_coordination_busy(self, busy: bool) -> None:
        self._coordination_busy = bool(busy)
        controls_enabled = not busy and not self._baseline_analysis_pending
        self._main_splitter.setEnabled(controls_enabled)
        self._back_button.setEnabled(not busy)
        for action in self._shortcut_actions:
            action.setEnabled(controls_enabled)
        self._complete_button.setEnabled(
            controls_enabled
            and any(
                self._stage_projection(detail).matches_status(
                    STAGE_PENDING_COORDINATION
                )
                for detail in self._all_variants
            )
        )
    @staticmethod
    def _result_count(value: object) -> int:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _coordination_failure_text(value: object) -> str:
        if not isinstance(value, (list, tuple)):
            return ""
        lines: list[str] = []
        for item in value[:8]:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                lines.append(f"{item[0]}：{item[1]}")
        return "\n".join(lines)

    def _confirm_leave_page(self) -> bool:
        return self._confirm_leave_changes()

    @property
    def is_batch_running(self) -> bool:
        """供主窗口关闭保护只读查询整体协调批次状态。"""
        return self._coordination_busy

    def _confirm_leave_changes(self, action_text: str = "返回首页") -> bool:
        if self._coordination_busy:
            return False
        self._finish_pending_comparison_transform()
        dirty_ids = [
            str(detail.get("变体ID", ""))
            for detail in self._all_variants
            if self._is_dirty(str(detail.get("变体ID", "")))
        ]
        if not dirty_ids:
            return True
        dialog = QMessageBox(self)
        dialog.setWindowTitle("未保存修改")
        dialog.setIcon(QMessageBox.Icon.Question)
        dialog.setText(f"有 {len(dirty_ids)} 个字形存在未保存的整体协调修改。")
        dialog.setInformativeText(f"是否保存后{action_text}？")
        dialog.setStandardButtons(
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel
        )
        dialog.setDefaultButton(QMessageBox.StandardButton.Save)
        save_button = dialog.button(QMessageBox.StandardButton.Save)
        discard_button = dialog.button(QMessageBox.StandardButton.Discard)
        cancel_button = dialog.button(QMessageBox.StandardButton.Cancel)
        if save_button is not None:
            save_button.setText("保存修改")
        if discard_button is not None:
            discard_button.setText("放弃修改")
        if cancel_button is not None:
            cancel_button.setText("取消")
        choice = dialog.exec()
        if choice == QMessageBox.StandardButton.Cancel.value:
            return False
        if choice == QMessageBox.StandardButton.Save.value:
            variants = [self._variant_by_id[item] for item in dirty_ids if item in self._variant_by_id]
            return self._save_variants(
                variants,
                show_success=False,
                title="保存整体协调修改",
            )
        with QSignalBlocker(self._ink_check):
            self._ink_check.setChecked(self._initial_ink_enabled)
        for variant_id in dirty_ids:
            detail = self._variant_by_id.get(variant_id)
            saved = (
                self._adjustment_service.load_saved_coordination_adjustments(detail)
                if detail is not None
                else self._default_adjustment()
            )
            self._adjustments[variant_id] = deepcopy(saved)
            self._saved_adjustments[variant_id] = deepcopy(saved)
            self._saved_signatures[variant_id] = self._adjustment_signature(saved)
            if detail is not None:
                self._ink_modes[variant_id] = self._stored_ink_mode(detail)
                self._saved_ink_signatures[variant_id] = self._stored_ink_signature(detail)
        self._preview_timer.stop()
        self._clear_preview_cache()
        self._apply_filters()
        return True

    def _change_page(self, offset: int) -> None:
        self._finish_pending_comparison_transform()
        total = math.ceil(len(self._variants) / self._page_size()) if self._variants else 0
        target = self._page_index + offset
        if 0 <= target < total:
            self._page_index = target
            self._render_page()

    def _move_detail_selection(self, offset: int) -> None:
        index = self._variant_index(self._selected_id)
        target = index + offset
        if 0 <= target < len(self._variants):
            self._select_variant(str(self._variants[target].get("变体ID", "")))
            self._enter_detail(self._selected_id)

    def request_back(self) -> None:
        """经未保存检查后返回首页。"""
        if self._coordination_busy:
            return
        if self._confirm_leave_changes():
            self._on_back()

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._schedule_capacity_update()
        self._schedule_list_thumbnail_loads()

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        self._schedule_capacity_update()
        self._schedule_list_thumbnail_loads()

    def _schedule_capacity_update(self) -> None:
        if self._capacity_update_pending:
            return
        self._capacity_update_pending = True
        QTimer.singleShot(0, self._update_comparison_capacity)

    def _update_comparison_capacity(self) -> None:
        self._capacity_update_pending = False
        self._finish_pending_comparison_transform()
        columns, rows = self._layout_mode_for_size(self._view_stack.size())
        if (columns, rows) == (self._grid_columns, self._grid_rows):
            return
        selected_index = self._variant_index(self._selected_id)
        self._grid_columns = columns
        self._grid_rows = rows
        self._page_index = selected_index // self._page_size() if selected_index >= 0 else 0
        self._render_page()

    def _layout_mode_for_size(self, size: QSize) -> tuple[int, int]:
        if (
            size.width() >= self.LARGE_VIEWPORT_WIDTH
            and size.height() >= self.LARGE_VIEWPORT_HEIGHT
        ):
            return self.LARGE_COLUMNS, self.LARGE_ROWS
        return self.COMPACT_COLUMNS, self.COMPACT_ROWS

    def _render_status(self) -> None:
        page = self._page_variants()
        dirty = sum(
            self._is_dirty(str(item.get("变体ID", "")))
            for item in page
        )
        self._status_label.setText(f"本页 {len(page)} 字　未保存调整 {dirty} 字")
        self._refresh_statistics()

    def _refresh_statistics(self) -> None:
        projections = [
            self._stage_projection(detail) for detail in self._list_variants
        ]
        pending = sum(
            projection.status == STAGE_PENDING_COORDINATION
            for projection in projections
        )
        coordinated = sum(
            projection.status == STATUS_COORDINATED
            for projection in projections
        )
        completion_count = sum(
            projection.completed
            and not projection.has_marker(MARKER_UNSAVED)
            for projection in projections
        )
        dirty = sum(
            projection.has_marker(MARKER_UNSAVED)
            for projection in projections
        )
        problem_count = sum(
            self._is_problem_status(projection)
            for projection in projections
        )
        self._summary_label.setText(
            f"待协调 {pending}　已协调 {coordinated}\n"
            f"未保存 {dirty}　问题 {problem_count}"
        )
        ink_counts = {"已达标": 0, "待确认": 0, "人工例外": 0}
        for projection in projections:
            category = {
                INK_STATUS_ACHIEVED: "已达标",
                INK_STATUS_PENDING: "待确认",
                INK_STATUS_EXCEPTION: "人工例外",
            }.get(projection.ink_status, "")
            if category in ink_counts:
                ink_counts[category] += 1
        self._ink_summary_label.setText(
            f"墨色达标 {ink_counts['已达标']}　"
            f"待确认 {ink_counts['待确认']}　"
            f"人工例外 {ink_counts['人工例外']}"
        )
        total = len(self._list_variants)
        self._progress_bar.setValue(
            round(completion_count * 100 / total) if total else 0
        )
        self._complete_button.setEnabled(
            bool(pending) and not self._coordination_busy
        )

    def _refresh_current_labels(self) -> None:
        detail = self._variant_by_id.get(self._selected_id)
        if detail is None:
            return
        char = str(detail.get("归属字", ""))
        filename = str(detail.get("原始文件", ""))
        index = self._variant_index(self._selected_id)
        status = self._coordination_status(detail)
        self._current_char_label.setText(char or "-")
        self._current_file_label.setText(filename or self._selected_id)
        self._current_index_label.setText(
            f"第 {index + 1 if index >= 0 else 0} 个字形 · 共 {len(self._variants)} 个"
        )
        self._detail_position_label.setText(
            f"{char} · {index + 1 if index >= 0 else 0} / {len(self._variants)}"
        )
        self._mode_status_label.setText(f"{char} · {filename}")
        self._dirty_label.setText(status)
        self._dirty_label.setStyleSheet(
            f"color: {self._coordination_status_color(status).name()};"
        )
        mode = self._ink_modes.get(self._selected_id, self.INK_MODES[0])
        self._updating_controls = True
        try:
            with QSignalBlocker(self._ink_strategy_combo):
                self._ink_strategy_combo.setCurrentText(mode)
        finally:
            self._updating_controls = False
        self._ink_strategy_combo.setEnabled(
            self._ink_check.isEnabled() and self._ink_check.isChecked()
        )
        result_text = self._ink_result_text(detail)
        if self._is_dirty(self._selected_id):
            result_text += "\n当前设置尚未保存，保存后会重新实测。"
        self._ink_result_label.setText(result_text)
        self._set_transform_controls_enabled(self._detail_canvas.has_image)

    def _coordination_status(self, detail: dict[str, Any]) -> str:
        return self._stage_projection(detail).status

    def _coordination_complete(self, detail: dict[str, Any]) -> bool:
        projection = self._stage_projection(detail)
        return projection.completed and not projection.has_marker(MARKER_UNSAVED)

    def _matches_status_filter(
        self,
        detail: dict[str, Any],
        status_filter: str,
    ) -> bool:
        return self._stage_projection(detail).matches_status(status_filter)

    def _workflow_status(self, detail: dict[str, Any]) -> WorkflowStatus:
        return self._stage_projection(detail).workflow

    def _stage_projection(
        self,
        detail: dict[str, Any],
    ) -> WorkflowStageProjection:
        variant_id = str(detail.get("变体ID", ""))
        dirty = bool(
            variant_id in self._saved_signatures and self._is_dirty(variant_id)
        )
        cached = self._workflow_status_cache.get(variant_id)
        if variant_id and cached is not None and cached[0] == dirty:
            return cached[1]
        projection = project_stage_status(
            detail,
            self._workflow_summary,
            self._finished_dir,
            PHASE_COORDINATION,
            dirty=dirty,
        )
        if variant_id:
            self._workflow_status_cache[variant_id] = (dirty, projection)
        return projection

    @staticmethod
    def _is_problem_status(
        status: WorkflowStageProjection | WorkflowStatus,
    ) -> bool:
        return any(marker != MARKER_INK_EXCEPTION for marker in status.markers)

    @staticmethod
    def _marker_text(
        status: WorkflowStageProjection | WorkflowStatus,
    ) -> str:
        return " · ".join(status.markers) if status.markers else "无"

    @staticmethod
    def _marker_color(
        status: WorkflowStageProjection | WorkflowStatus,
    ) -> QColor:
        if status.has_marker(MARKER_FILE_ERROR):
            return QColor("#E36A6A")
        if any(
            status.has_marker(marker)
            for marker in (MARKER_UNSAVED, MARKER_STRUCTURE_REVIEW, MARKER_INK_PENDING)
        ):
            return QColor("#F2B84B")
        if status.has_marker(MARKER_INK_EXCEPTION):
            return QColor("#4DA3FF")
        return QColor("#A6B0BE")

    @staticmethod
    def _coordination_status_color(status: str) -> QColor:
        return QColor(PHASE_STATUS_COLORS.get(status, "#D7DEE8"))

    def _coordinated_preview_cache_key(self, variant_id: str) -> tuple[Any, ...]:
        return (
            variant_id,
            self._signature(variant_id),
            self._ink_signature(self._current_ink_config(variant_id)),
        )

    def _store_preview(
        self,
        variant_id: str,
        cache_key: tuple[Any, ...],
        image: QImage,
        bounds: tuple[int, int, int, int] | None,
    ) -> None:
        """按图像字节数维护协调预览 LRU，并移除同字形旧版本。"""

        stale_keys = [
            key
            for key in self._preview_cache
            if key[0] == variant_id and key != cache_key
        ]
        for stale_key in stale_keys:
            self._remove_preview_cache_key(stale_key)
        self._remove_preview_cache_key(cache_key)
        self._preview_cache[cache_key] = image
        if bounds is not None:
            self._preview_bounds_cache[cache_key] = bounds
        self._preview_cache_bytes += max(0, int(image.sizeInBytes()))
        while self._preview_cache and (
            len(self._preview_cache) > self._preview_cache_max_items
            or self._preview_cache_bytes > self._preview_cache_max_bytes
        ):
            oldest_key = next(iter(self._preview_cache))
            if oldest_key == cache_key and len(self._preview_cache) == 1:
                break
            self._remove_preview_cache_key(oldest_key)

    def _remove_preview_cache_key(self, cache_key: tuple[Any, ...]) -> None:
        image = self._preview_cache.pop(cache_key, None)
        if image is not None:
            self._preview_cache_bytes = max(
                0,
                self._preview_cache_bytes - max(0, int(image.sizeInBytes())),
            )
        self._preview_bounds_cache.pop(cache_key, None)

    def _clear_preview_cache(self) -> None:
        self._preview_cache.clear()
        self._preview_bounds_cache.clear()
        self._preview_cache_bytes = 0

    def _clear_variant_preview_cache(self, variant_id: str) -> None:
        for key in [key for key in self._preview_cache if key and key[0] == variant_id]:
            self._remove_preview_cache_key(key)

    def _glyph_thumbnail(self, detail: dict[str, Any]) -> QIcon:
        variant_id = str(detail.get("变体ID", ""))
        preview_key: tuple[Any, ...] = (variant_id, "无协调预览")
        if variant_id in self._variant_by_id:
            preview_key = self._coordinated_preview_cache_key(variant_id)
            coordinated_preview = self._preview_cache.get(preview_key)
            if coordinated_preview is not None and not coordinated_preview.isNull():
                return self._thumbnail_icon(variant_id, coordinated_preview, preview_key)

        stored_key = self._list_thumbnail_key_by_variant.get(variant_id)
        if stored_key is not None:
            cached = self._list_thumbnail_cache.get(stored_key)
            if cached is None:
                self._list_thumbnail_key_by_variant.pop(variant_id, None)
            elif stored_key != preview_key:
                _path, current_key = self._list_thumbnail_source(detail)
                if current_key == stored_key:
                    self._list_thumbnail_cache.move_to_end(stored_key)
                    return cached
                self._list_thumbnail_cache.pop(stored_key, None)
                self._list_thumbnail_key_by_variant.pop(variant_id, None)
            else:
                self._list_thumbnail_cache.pop(stored_key, None)
                self._list_thumbnail_key_by_variant.pop(variant_id, None)
        return self._placeholder_thumbnail_icon()

    def _list_thumbnail_source(
        self,
        detail: dict[str, Any],
    ) -> tuple[str, tuple[Any, ...]]:
        variant_id = str(detail.get("变体ID", ""))
        directories = self._glyph.get_workflow_dirs()
        for directory_key, file_key in (
            ("成品", "成品文件"),
            ("手工审核", "审核文件"),
            ("优化预览", "中间文件"),
            ("清洁掩码", "清洁掩码文件"),
            ("灰度母版", "灰度母版文件"),
            ("原图", "原始文件"),
        ):
            raw_filename = detail.get(file_key, "")
            if not str(raw_filename or "").strip():
                continue
            path = resolve_safe_stage_file(
                directories[directory_key],
                raw_filename,
            )
            if not path:
                return "", (variant_id, "文件异常", directory_key, str(raw_filename))
            try:
                stat = os.stat(path)
            except OSError:
                return "", (variant_id, "文件异常", directory_key, str(raw_filename))
            cache_key = (
                variant_id,
                directory_key,
                str(raw_filename),
                stat.st_mtime_ns,
                stat.st_size,
            )
            if cache_key in self._list_thumbnail_failures:
                return "", cache_key
            return path, cache_key
        return "", (variant_id, "无可用图片")

    def _schedule_list_thumbnail_loads(self, _value: object = None) -> None:
        if self._coordination_busy:
            return
        self._list_thumbnail_timer.start()

    def _load_visible_list_thumbnails(self) -> None:
        if self._coordination_busy:
            return
        if not self._list_items_by_id:
            return
        viewport = self._glyph_list.viewport()
        first = self._glyph_list.itemAt(QPoint(2, 2))
        requested: list[str] = []
        item = first
        viewport_bottom = viewport.height()
        while item is not None:
            rect = self._glyph_list.visualItemRect(item)
            if rect.top() > viewport_bottom:
                break
            variant_id = str(item.data(0, Qt.ItemDataRole.UserRole) or "")
            if variant_id:
                requested.append(variant_id)
            item = self._glyph_list.itemBelow(item)

        prefetched = 0
        while item is not None and prefetched < self.LIST_THUMBNAIL_PREFETCH_ITEMS:
            variant_id = str(item.data(0, Qt.ItemDataRole.UserRole) or "")
            if variant_id:
                requested.append(variant_id)
                prefetched += 1
            item = self._glyph_list.itemBelow(item)

        if self._selected_id in self._list_items_by_id:
            requested.insert(0, self._selected_id)
        for variant_id in dict.fromkeys(requested):
            self._request_list_thumbnail(variant_id)

    def _request_list_thumbnail(self, variant_id: str) -> None:
        detail = self._list_variant_by_id.get(variant_id)
        item = self._list_items_by_id.get(variant_id)
        if detail is None or item is None:
            return

        if variant_id in self._variant_by_id:
            preview_key = self._coordinated_preview_cache_key(variant_id)
            coordinated_preview = self._preview_cache.get(preview_key)
            if coordinated_preview is not None and not coordinated_preview.isNull():
                item.setIcon(0, self._thumbnail_icon(variant_id, coordinated_preview, preview_key))
                return

        path, cache_key = self._list_thumbnail_source(detail)
        cached = self._list_thumbnail_cache.get(cache_key)
        if cached is not None:
            self._list_thumbnail_cache.move_to_end(cache_key)
            self._list_thumbnail_key_by_variant[variant_id] = cache_key
            item.setIcon(0, cached)
            return
        if not path or cache_key in self._list_thumbnail_failures:
            return
        if variant_id in self._list_thumbnail_workers:
            return
        if len(self._list_thumbnail_workers) >= self.LIST_THUMBNAIL_MAX_REQUESTS:
            return

        worker = FunctionWorker(
            lambda source_path=path: decode_thumbnail_image(
                source_path,
                QSize(self.LIST_THUMBNAIL_SIZE - 4, self.LIST_THUMBNAIL_SIZE - 4),
                self.LIST_THUMBNAIL_DECODE_LIMIT_BYTES,
            )
        )
        worker.setAutoDelete(False)
        worker.signals.finished.connect(
            lambda result, target=variant_id, source_path=path, key=cache_key, task=worker: (
                self._list_thumbnail_finished(target, source_path, key, result, task)
            )
        )
        worker.signals.failed.connect(
            lambda _message, target=variant_id, key=cache_key, task=worker: (
                self._list_thumbnail_failed(target, key, task)
            )
        )
        self._list_thumbnail_workers[variant_id] = (path, cache_key, worker)
        self._list_thumbnail_pool.start(worker)

    def _list_thumbnail_finished(
        self,
        variant_id: str,
        path: str,
        cache_key: tuple[Any, ...],
        result: object,
        worker: FunctionWorker,
    ) -> None:
        self._release_list_thumbnail_worker(variant_id, worker)
        detail = self._list_variant_by_id.get(variant_id)
        if detail is None or self._list_thumbnail_source(detail) != (path, cache_key):
            self._schedule_list_thumbnail_loads()
            return

        preview_key = self._coordinated_preview_cache_key(variant_id)
        coordinated_preview = self._preview_cache.get(preview_key)
        if coordinated_preview is not None and not coordinated_preview.isNull():
            self._set_list_thumbnail_from_preview(variant_id, coordinated_preview)
            self._schedule_list_thumbnail_loads()
            return
        if not isinstance(result, QImage) or result.isNull():
            self._list_thumbnail_failures.add(cache_key)
            self._schedule_list_thumbnail_loads()
            return

        icon = self._thumbnail_icon(variant_id, result, cache_key)
        item = self._list_items_by_id.get(variant_id)
        if item is not None:
            item.setIcon(0, icon)
        self._schedule_list_thumbnail_loads()

    def _list_thumbnail_failed(
        self,
        variant_id: str,
        cache_key: tuple[Any, ...],
        worker: FunctionWorker,
    ) -> None:
        self._release_list_thumbnail_worker(variant_id, worker)
        self._list_thumbnail_failures.add(cache_key)
        self._schedule_list_thumbnail_loads()

    def _release_list_thumbnail_worker(
        self,
        variant_id: str,
        worker: FunctionWorker,
    ) -> None:
        pending = self._list_thumbnail_workers.get(variant_id)
        if pending is not None and pending[2] is worker:
            self._list_thumbnail_workers.pop(variant_id, None)

    def _placeholder_thumbnail_icon(self) -> QIcon:
        if self._list_placeholder_icon.isNull():
            self._list_placeholder_icon = self._thumbnail_icon("", QImage(), ())
        return self._list_placeholder_icon

    def _set_list_thumbnail_from_preview(
        self,
        variant_id: str,
        image: QImage,
    ) -> None:
        icon = self._thumbnail_icon(
            variant_id,
            image,
            self._coordinated_preview_cache_key(variant_id),
        )
        item = self._list_items_by_id.get(variant_id)
        if item is not None:
            item.setIcon(0, icon)

    def _thumbnail_icon(
        self,
        variant_id: str,
        source: QImage,
        cache_key: tuple[Any, ...],
    ) -> QIcon:
        cached = self._list_thumbnail_cache.get(cache_key)
        if cached is not None:
            self._list_thumbnail_cache.move_to_end(cache_key)
            return cached

        size = QSize(38, 38)
        thumbnail = QPixmap(size)
        thumbnail.fill(QColor("#ffffff"))
        if not source.isNull():
            painter = QPainter(thumbnail)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            preview = QPixmap.fromImage(source).scaled(
                34,
                34,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = (size.width() - preview.width()) // 2
            y = (size.height() - preview.height()) // 2
            painter.drawPixmap(x, y, preview)
            painter.end()
        icon = QIcon(thumbnail)
        if variant_id:
            stale_keys = [
                key
                for key in self._list_thumbnail_cache
                if key and key[0] == variant_id and key != cache_key
            ]
            for stale_key in stale_keys:
                self._list_thumbnail_cache.pop(stale_key, None)
            self._list_thumbnail_cache[cache_key] = icon
            self._list_thumbnail_key_by_variant[variant_id] = cache_key
            self._list_thumbnail_cache.move_to_end(cache_key)
            while len(self._list_thumbnail_cache) > self.LIST_THUMBNAIL_CACHE_ITEMS:
                expired_key, _expired_icon = self._list_thumbnail_cache.popitem(
                    last=False
                )
                if expired_key:
                    expired_id = str(expired_key[0])
                    if self._list_thumbnail_key_by_variant.get(expired_id) == expired_key:
                        self._list_thumbnail_key_by_variant.pop(expired_id, None)
        return icon

    @staticmethod
    def _to_review_image(image: QImage) -> QImage:
        rgba = image.convertToFormat(QImage.Format.Format_ARGB32)
        if image.hasAlphaChannel():
            return rgba
        alpha = image.convertToFormat(QImage.Format.Format_Grayscale8)
        alpha.invertPixels(QImage.InvertMode.InvertRgb)
        result = QImage(image.size(), QImage.Format.Format_ARGB32)
        result.fill(Qt.GlobalColor.black)
        result.setAlphaChannel(alpha)
        return result

    def _set_transform_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self._offset_x_spin,
            self._offset_y_spin,
            self._scale_slider,
            self._stretch_w_slider,
            self._stretch_h_slider,
            self._rotation_slider,
            self._advanced_button,
            self._restore_button,
        ):
            widget.setEnabled(enabled)
        for spin in self._distort_spins:
            spin.setEnabled(enabled)

    def _update_history_buttons(self, can_undo: bool, can_redo: bool) -> None:
        self._undo_button.setEnabled(can_undo)
        self._redo_button.setEnabled(can_redo)

    def _detail_canvas_undo(self) -> None:
        self._finish_pending_comparison_transform()
        self._detail_canvas.undo()

    def _detail_canvas_redo(self) -> None:
        self._finish_pending_comparison_transform()
        self._detail_canvas.redo()

    def _detail_canvas_fit(self) -> None:
        self._detail_canvas.fit_to_view()

    def _detail_canvas_actual_size(self) -> None:
        self._detail_canvas.actual_size()

    def _detail_canvas_set_grid(self, visible: bool) -> None:
        self._detail_canvas.set_grid_visible(visible)

    def _variant_index(self, variant_id: str) -> int:
        return next(
            (
                index
                for index, item in enumerate(self._variants)
                if str(item.get("变体ID", "")) == variant_id
            ),
            -1,
        )

    def _next_navigation_id(self, variant_id: str) -> str:
        current_index = self._variant_index(variant_id)
        if 0 <= current_index < len(self._variants) - 1:
            return str(self._variants[current_index + 1].get("变体ID", ""))

        # 当前精调字可能因实时状态筛选已移出列表，仍按同一搜索和排序规则
        # 在可见字形中寻找它原位置之后的第一个字形。
        query = self._search_edit.text().strip().casefold()
        ordered: list[dict[str, Any]] = []
        for detail in self._all_variants:
            text = " ".join(
                str(detail.get(key, ""))
                for key in ("归属字", "原始文件", "导入前文件名", "中间文件", "审核文件")
            ).casefold()
            if not query or query in text:
                ordered.append(detail)
        self._sort_variants(ordered)
        ordered_ids = [str(item.get("变体ID", "")) for item in ordered]
        try:
            ordered_index = ordered_ids.index(variant_id)
        except ValueError:
            return ""
        visible_ids = {
            str(item.get("变体ID", "")) for item in self._variants
        }
        return next(
            (item for item in ordered_ids[ordered_index + 1:] if item in visible_ids),
            "",
        )

    @staticmethod
    def _identity_canvas_transform() -> dict[str, Any]:
        return {
            "x": 0.0,
            "y": 0.0,
            "scale": 1.0,
            "stretch_w": 1.0,
            "stretch_h": 1.0,
            "rotation": 0.0,
            "distort": [0.0] * 8,
        }

    def shutdown(self) -> None:
        """关闭程序时取消协调任务并释放预览、缩略图后台资源。"""

        if self._coordination_task is not None:
            self._coordination_task.request_cancel()
        self._preview_timer.stop()
        self._comparison_wheel_timer.stop()
        self._list_thumbnail_timer.stop()
        self._preview_pool.clear()
        self._list_thumbnail_pool.clear()
        self._coordination_pool.clear()
        self._preview_workers.clear()
        self._list_thumbnail_workers.clear()
        self._clear_preview_cache()
        self._list_thumbnail_cache.clear()

    @classmethod
    def _percent_to_slider_position(cls, percent: int | float) -> int:
        normalized = max(
            cls.TRANSFORM_PERCENT_MIN,
            min(cls.TRANSFORM_PERCENT_MAX, round(float(percent))),
        )
        if normalized < 100:
            lower_span = 100 - cls.TRANSFORM_PERCENT_MIN
            return round(
                (normalized - 100) * cls.TRANSFORM_PERCENT_SLIDER_EXTENT / lower_span
            )
        return normalized - 100

    @classmethod
    def _slider_position_to_percent(cls, position: int) -> int:
        normalized = max(
            -cls.TRANSFORM_PERCENT_SLIDER_EXTENT,
            min(cls.TRANSFORM_PERCENT_SLIDER_EXTENT, int(position)),
        )
        if normalized < 0:
            lower_span = 100 - cls.TRANSFORM_PERCENT_MIN
            return round(
                100 + normalized * lower_span / cls.TRANSFORM_PERCENT_SLIDER_EXTENT
            )
        return 100 + normalized

    @staticmethod
    def _number(value: object, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _positive_int(value: object, default: int) -> int:
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _pil_to_qimage(image: Image.Image) -> QImage:
        rgba = image.convert("RGBA")
        pixels = rgba.tobytes("raw", "RGBA")
        return QImage(
            pixels,
            rgba.width,
            rgba.height,
            rgba.width * 4,
            QImage.Format.Format_RGBA8888,
        ).copy()

    @staticmethod
    def _segment_button(text: str) -> QPushButton:
        button = QPushButton(text)
        button.setCheckable(True)
        button.setProperty("controlRole", "segment")
        button.setObjectName("compactButton")
        return button

    @staticmethod
    def _toolbar_button(text: str, tooltip: str, checkable: bool = False) -> QToolButton:
        button = QToolButton()
        button.setText(text)
        button.setToolTip(tooltip)
        button.setCheckable(checkable)
        button.setMinimumHeight(30)
        button.setStyleSheet(
            "QToolButton { padding: 0 8px; border: 1px solid #37404d; border-radius: 5px; background: #282f3a; }"
            "QToolButton:hover { border-color: #4da3ff; background: #303947; }"
            "QToolButton:checked { border-color: #4da3ff; background: #294d75; color: #ffffff; }"
            "QToolButton:disabled { color: #68717e; background: #242a33; border-color: #303640; }"
        )
        return button

    @staticmethod
    def _vertical_separator() -> QFrame:
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        return separator

    @staticmethod
    def _horizontal_separator() -> QFrame:
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        return separator
