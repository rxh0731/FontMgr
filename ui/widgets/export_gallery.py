"""字库导出页使用的虚拟化只读字形画廊。"""

from __future__ import annotations

import ctypes
import os
from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import (
    QAbstractListModel,
    QModelIndex,
    QPoint,
    QRectF,
    QSize,
    Qt,
    QThread,
    QThreadPool,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QColor,
    QFontMetrics,
    QImage,
    QImageIOHandler,
    QImageReader,
    QPainter,
    QPen,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QListView,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QWidget,
)

from services.workflow_status_service import PHASE_STATUS_COLORS, STAGE_COLORS
from ui.workers import FunctionWorker


MIB = 1024 * 1024
MIN_CACHE_BUDGET = 2 * MIB
NORMAL_MIN_CACHE_BUDGET = 8 * MIB
MAX_CACHE_BUDGET = 256 * MIB
PINNED_CACHE_OVERFLOW = 16 * MIB


@dataclass(frozen=True, slots=True)
class ExportGalleryEntry:
    """导出画廊中的一项只读字形。"""

    variant_id: str
    char: str
    filename: str
    image_path: str
    status: str = ""
    image_canvas_size: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class ExportPreviewGeometry:
    """导出预览中 130% 工作区与字库田字格的屏幕几何。"""

    workspace_rect: QRectF
    grid_rect: QRectF
    pixel_scale: float


def calculate_export_preview_geometry(
    image_area: QRectF,
    canvas_size: QSize,
    *,
    workspace_ratio: float = 1.3,
    padding: float = 10.0,
) -> ExportPreviewGeometry:
    """按字库画布比例将田字格及其 130% 工作区等比放入预览区。"""
    grid_width = max(1, int(canvas_size.width()))
    grid_height = max(1, int(canvas_size.height()))
    ratio = max(1.0, float(workspace_ratio))
    workspace_width = grid_width * ratio
    workspace_height = grid_height * ratio
    available_width = max(1.0, image_area.width() - max(0.0, padding))
    available_height = max(1.0, image_area.height() - max(0.0, padding))
    pixel_scale = min(
        available_width / workspace_width,
        available_height / workspace_height,
    )
    center = image_area.center()
    workspace_rect = QRectF(
        center.x() - workspace_width * pixel_scale / 2.0,
        center.y() - workspace_height * pixel_scale / 2.0,
        workspace_width * pixel_scale,
        workspace_height * pixel_scale,
    )
    grid_rect = QRectF(
        center.x() - grid_width * pixel_scale / 2.0,
        center.y() - grid_height * pixel_scale / 2.0,
        grid_width * pixel_scale,
        grid_height * pixel_scale,
    )
    return ExportPreviewGeometry(workspace_rect, grid_rect, pixel_scale)


def calculate_thumbnail_cache_budget(
    total_memory_bytes: int,
    available_memory_bytes: int,
) -> int:
    """根据物理内存计算缩略图缓存预算，返回整 MiB 字节数。"""
    total = max(1, int(total_memory_bytes))
    available = max(0, min(total, int(available_memory_bytes)))
    reserve = max(512 * MIB, int(total * 0.05))
    if available <= reserve:
        raw_budget = max(
            MIN_CACHE_BUDGET,
            min(NORMAL_MIN_CACHE_BUDGET, int(available * 0.01)),
        )
    else:
        raw_budget = min(
            int(total * 0.01),
            int(available * 0.04),
            int((available - reserve) * 0.10),
        )
        raw_budget = max(
            NORMAL_MIN_CACHE_BUDGET,
            min(MAX_CACHE_BUDGET, raw_budget),
        )
    return max(MIB, raw_budget // MIB * MIB)


def get_system_memory_status() -> tuple[int, int]:
    """返回物理内存总量和当前可用量；探测失败时使用保守回退值。"""
    if os.name == "nt":
        try:
            class MemoryStatus(ctypes.Structure):
                _fields_ = (
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                )

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.total_physical), int(status.available_physical)
        except (AttributeError, OSError, TypeError, ValueError):
            pass
    else:
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            total_pages = int(os.sysconf("SC_PHYS_PAGES"))
            available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            return page_size * total_pages, page_size * available_pages
        except (AttributeError, OSError, TypeError, ValueError):
            pass
    return 8 * 1024 * MIB, 2 * 1024 * MIB


class WeightedImageCache:
    """按解码后图像字节数约束的 LRU 缓存。"""

    ENTRY_OVERHEAD = 1024

    def __init__(
        self,
        budget_bytes: int,
        *,
        max_items: int = 2048,
        pinned_overflow_bytes: int = PINNED_CACHE_OVERFLOW,
    ) -> None:
        self._budget_bytes = max(1, int(budget_bytes))
        self._max_items = max(1, int(max_items))
        self._pinned_overflow_bytes = max(0, int(pinned_overflow_bytes))
        self._entries: OrderedDict[Hashable, tuple[QImage, int]] = OrderedDict()
        self._pinned: set[Hashable] = set()
        self._used_bytes = 0

    @property
    def budget_bytes(self) -> int:
        return self._budget_bytes

    @property
    def max_items(self) -> int:
        return self._max_items

    @property
    def used_bytes(self) -> int:
        return self._used_bytes

    @property
    def item_count(self) -> int:
        return len(self._entries)

    @property
    def keys(self) -> tuple[Hashable, ...]:
        return tuple(self._entries)

    @classmethod
    def image_cost(cls, image: QImage) -> int:
        return max(0, int(image.sizeInBytes())) + cls.ENTRY_OVERHEAD

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def get(self, key: Hashable) -> QImage | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._entries.move_to_end(key)
        return entry[0]

    def put(self, key: Hashable, image: QImage) -> None:
        previous = self._entries.pop(key, None)
        if previous is not None:
            self._used_bytes -= previous[1]
        stored = QImage(image)
        cost = self.image_cost(stored)
        self._entries[key] = (stored, cost)
        self._used_bytes += cost
        self._trim()

    def set_pinned(self, keys: Iterable[Hashable]) -> None:
        self._pinned = set(keys)
        self._trim()

    def set_limits(self, budget_bytes: int, max_items: int | None = None) -> None:
        self._budget_bytes = max(1, int(budget_bytes))
        if max_items is not None:
            self._max_items = max(1, int(max_items))
        self._trim()

    def remove_if(self, predicate: Callable[[Hashable], bool]) -> None:
        for key in tuple(self._entries):
            if predicate(key):
                self._remove(key)

    def clear(self) -> None:
        self._entries.clear()
        self._pinned.clear()
        self._used_bytes = 0

    def _trim(self) -> None:
        while self._entries:
            over_budget = self._used_bytes > self._budget_bytes
            over_items = len(self._entries) > self._max_items
            if not over_budget and not over_items:
                break
            candidate = next(
                (key for key in self._entries if key not in self._pinned),
                None,
            )
            if candidate is None:
                within_pinned_overflow = (
                    self._used_bytes
                    <= self._budget_bytes + self._pinned_overflow_bytes
                )
                if within_pinned_overflow and not over_items:
                    break
                candidate = next(iter(self._entries))
            self._remove(candidate)

    def _remove(self, key: Hashable) -> None:
        entry = self._entries.pop(key, None)
        if entry is not None:
            self._used_bytes -= entry[1]
        self._pinned.discard(key)


class ExportGalleryModel(QAbstractListModel):
    """仅保存轻量元数据的虚拟画廊模型。"""

    ENTRY_ROLE = int(Qt.ItemDataRole.UserRole) + 1
    VARIANT_ID_ROLE = ENTRY_ROLE + 1

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._entries: list[ExportGalleryEntry] = []

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._entries)

    def data(self, index: QModelIndex, role: int = int(Qt.ItemDataRole.DisplayRole)) -> Any:
        if not index.isValid() or not 0 <= index.row() < len(self._entries):
            return None
        entry = self._entries[index.row()]
        if role == int(Qt.ItemDataRole.DisplayRole):
            return entry.char
        if role == self.ENTRY_ROLE:
            return entry
        if role == self.VARIANT_ID_ROLE:
            return entry.variant_id
        if role == int(Qt.ItemDataRole.ToolTipRole):
            status = f"\n状态：{entry.status}" if entry.status else ""
            return f"{entry.char} · {entry.filename}{status}"
        return None

    def set_entries(self, entries: Iterable[ExportGalleryEntry]) -> None:
        self.beginResetModel()
        self._entries = list(entries)
        self.endResetModel()

    def entry_at(self, row: int) -> ExportGalleryEntry | None:
        if 0 <= row < len(self._entries):
            return self._entries[row]
        return None


class ExportGalleryDelegate(QStyledItemDelegate):
    """直接绘制白底字形、田字格和文件摘要，避免创建成千上万个卡片。"""

    FOOTER_HEIGHT = 34
    WORKSPACE_RATIO = 1.3

    def __init__(
        self,
        image_provider: Callable[[ExportGalleryEntry], tuple[QImage | None, bool]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._image_provider = image_provider
        self._canvas_size = QSize(250, 250)

    @staticmethod
    def status_color(status: str) -> QColor:
        return QColor(
            PHASE_STATUS_COLORS.get(
                status,
                STAGE_COLORS.get(status, "#D95757"),
            )
        )

    @property
    def canvas_size(self) -> QSize:
        return QSize(self._canvas_size)

    def set_canvas_size(self, width: int, height: int) -> None:
        self._canvas_size = QSize(max(1, int(width)), max(1, int(height)))

    def preview_geometry(self, image_area: QRectF) -> ExportPreviewGeometry:
        return calculate_export_preview_geometry(
            image_area,
            self._canvas_size,
            workspace_ratio=self.WORKSPACE_RATIO,
        )

    def image_target_rect(
        self,
        entry: ExportGalleryEntry,
        image: QImage,
        geometry: ExportPreviewGeometry,
    ) -> QRectF:
        """按成品实际画布尺寸映射图片；旧数据则在田字格内等比适配。"""
        logical_size = entry.image_canvas_size
        logical_width = logical_height = 0
        if isinstance(logical_size, tuple) and len(logical_size) == 2:
            try:
                logical_width = int(logical_size[0])
                logical_height = int(logical_size[1])
            except (TypeError, ValueError):
                logical_width = logical_height = 0
        if logical_width > 0 and logical_height > 0:
            width = float(logical_width) * geometry.pixel_scale
            height = float(logical_height) * geometry.pixel_scale
        else:
            scale = min(
                geometry.grid_rect.width() / max(1, image.width()),
                geometry.grid_rect.height() / max(1, image.height()),
            )
            width = image.width() * scale
            height = image.height() * scale
        center = geometry.workspace_rect.center()
        return QRectF(
            center.x() - width / 2.0,
            center.y() - height / 2.0,
            width,
            height,
        )

    def sizeHint(
        self,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ) -> QSize:
        """让项目实际绘制区域填满 QListView 分配的固定网格。"""
        view = self.parent()
        if isinstance(view, QListView):
            grid_size = view.gridSize()
            if grid_size.isValid() and not grid_size.isEmpty():
                return grid_size
        return super().sizeHint(option, index)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ) -> None:
        entry = index.data(ExportGalleryModel.ENTRY_ROLE)
        if not isinstance(entry, ExportGalleryEntry):
            return

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        outer = QRectF(option.rect.adjusted(5, 5, -5, -5))
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#202630"))
        painter.drawRoundedRect(outer, 4.0, 4.0)

        image_area = QRectF(
            outer.left(),
            outer.top(),
            outer.width(),
            max(1.0, outer.height() - self.FOOTER_HEIGHT),
        )
        geometry = self.preview_geometry(image_area)
        work_rect = geometry.workspace_rect
        painter.fillRect(image_area, QColor("#171B22"))
        painter.fillRect(work_rect, QColor("#FFFFFF"))
        image, failed = self._image_provider(entry)
        if image is not None and not image.isNull():
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            image_rect = self.image_target_rect(entry, image, geometry)
            painter.save()
            painter.setClipRect(work_rect)
            painter.drawImage(image_rect, image)
            painter.restore()
        else:
            painter.setPen(QColor("#A6B0BE" if not failed else "#C05B5B"))
            painter.drawText(
                work_rect,
                Qt.AlignmentFlag.AlignCenter,
                "载入中" if not failed else "无法显示",
            )

        grid = geometry.grid_rect
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

        footer = QRectF(
            outer.left(),
            outer.bottom() - self.FOOTER_HEIGHT,
            outer.width(),
            self.FOOTER_HEIGHT,
        )
        painter.fillRect(footer, QColor("#202630"))
        text_rect = footer.adjusted(8, 0, -18, 0)
        painter.setPen(QColor("#F1F4F8"))
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            entry.char,
        )
        metrics = QFontMetrics(painter.font())
        filename_left = text_rect.left() + max(22, metrics.horizontalAdvance(entry.char) + 10)
        filename_rect = QRectF(
            filename_left,
            text_rect.top(),
            max(1.0, text_rect.right() - filename_left),
            text_rect.height(),
        )
        painter.setPen(QColor("#A6B0BE"))
        filename = metrics.elidedText(
            entry.filename,
            Qt.TextElideMode.ElideMiddle,
            max(1, int(filename_rect.width())),
        )
        painter.drawText(
            filename_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            filename,
        )
        status_color = self.status_color(entry.status)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(status_color)
        painter.drawEllipse(footer.right() - 13.0, footer.center().y() - 4.0, 8.0, 8.0)

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#4DA3FF") if selected else QColor("#505A68"), 3 if selected else 1))
        painter.drawRoundedRect(outer, 4.0, 4.0)
        painter.restore()


@dataclass(frozen=True, slots=True)
class _DecodeResult:
    generation: int
    key: Hashable
    row: int
    token: int
    image: QImage
    failed: bool


def decode_thumbnail_image(path: str, target_size: QSize, decode_limit_bytes: int) -> QImage:
    """在工作线程中按目标尺寸解码，拒绝不支持缩放的异常超大图片。"""
    if not path or target_size.isEmpty():
        return QImage()
    reader = QImageReader(path)
    reader.setAutoTransform(True)
    source_size = reader.size()
    supports_scaled = reader.supportsOption(QImageIOHandler.ImageOption.ScaledSize)
    if source_size.isValid():
        estimated_source_bytes = source_size.width() * source_size.height() * 4
        if not supports_scaled and estimated_source_bytes > max(1, decode_limit_bytes):
            return QImage()
        scaled_size = source_size.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
        )
        if supports_scaled and scaled_size.isValid():
            reader.setScaledSize(scaled_size)
    image = reader.read()
    if image.isNull():
        return QImage()
    if image.width() > target_size.width() or image.height() > target_size.height():
        image = image.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    return image.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)


class ExportGallery(QListView):
    """全库滚动浏览使用的八列自适应、内存有界画廊。"""

    variant_selected = Signal(str)
    loading_idle = Signal()

    DEFAULT_COLUMNS = 8
    MIN_COLUMNS = 1
    MAX_COLUMNS = 24
    PREFETCH_VIEWPORTS = 1
    MAX_REQUESTS = 192
    MEMORY_REFRESH_INTERVAL_MS = 30_000

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        memory_provider: Callable[[], tuple[int, int]] | None = None,
        image_loader: Callable[[str, QSize, int], QImage] | None = None,
        cache_budget_bytes: int | None = None,
    ) -> None:
        super().__init__(parent)
        self._memory_provider = memory_provider or get_system_memory_status
        self._image_loader = image_loader or decode_thumbnail_image
        self._fixed_cache_budget = cache_budget_bytes
        self._cache_budget = self._initial_cache_budget()
        self._cache = WeightedImageCache(self._cache_budget)
        self._gallery_model = ExportGalleryModel(self)
        self._delegate = ExportGalleryDelegate(self._cached_image_for_entry, self)
        self._column_count = self.DEFAULT_COLUMNS
        self._target_bucket = 64
        self._generation = 0
        self._token = 0
        self._row_by_variant: dict[str, int] = {}
        self._pending: dict[Hashable, tuple[int, FunctionWorker]] = {}
        self._failed_keys: set[Hashable] = set()
        self._last_requested_count = 0
        self._syncing_selection = False

        self.setModel(self._gallery_model)
        self.setItemDelegate(self._delegate)
        self.setViewMode(QListView.ViewMode.IconMode)
        self.setFlow(QListView.Flow.LeftToRight)
        self.setWrapping(True)
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setLayoutMode(QListView.LayoutMode.Batched)
        self.setBatchSize(64)
        self.setMovement(QListView.Movement.Static)
        self.setUniformItemSizes(True)
        self.setSpacing(0)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # 固定预留滚动条宽度，避免项目增多时视口骤窄并改变每行列数。
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setStyleSheet("QListView { background: #171B22; border: 0; outline: 0; }")

        self._thread_pool = QThreadPool(self)
        self._thread_pool.setMaxThreadCount(1 if self._cache_budget <= NORMAL_MIN_CACHE_BUDGET else 2)
        self._thread_pool.setExpiryTimeout(15_000)

        self._layout_timer = QTimer(self)
        self._layout_timer.setSingleShot(True)
        self._layout_timer.setInterval(40)
        self._layout_timer.timeout.connect(self._apply_grid_layout)
        self._load_timer = QTimer(self)
        self._load_timer.setSingleShot(True)
        self._load_timer.setInterval(40)
        self._load_timer.timeout.connect(self._schedule_visible_loads)
        self._memory_timer = QTimer(self)
        self._memory_timer.setInterval(self.MEMORY_REFRESH_INTERVAL_MS)
        self._memory_timer.timeout.connect(self._refresh_memory_budget)
        if self._fixed_cache_budget is None:
            self._memory_timer.start()

        self.verticalScrollBar().valueChanged.connect(self._schedule_load_update)
        self.selectionModel().currentChanged.connect(self._current_changed)
        self._apply_grid_layout()

    @property
    def column_count(self) -> int:
        return self._column_count

    @property
    def canvas_size(self) -> QSize:
        return self._delegate.canvas_size

    @property
    def entry_count(self) -> int:
        return self._gallery_model.rowCount()

    @property
    def cache_budget_bytes(self) -> int:
        return self._cache.budget_bytes

    @property
    def cached_bytes(self) -> int:
        return self._cache.used_bytes

    @property
    def cached_item_count(self) -> int:
        return self._cache.item_count

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def last_requested_count(self) -> int:
        return self._last_requested_count

    @property
    def worker_thread_limit(self) -> int:
        return self._thread_pool.maxThreadCount()

    def set_entries(self, entries: Iterable[ExportGalleryEntry]) -> None:
        """替换全库元数据；图片保持按可见区域惰性加载。"""
        normalized = list(entries)
        self._generation += 1
        self._cancel_queued_requests()
        self._cache.clear()
        self._failed_keys.clear()
        self._row_by_variant = {
            entry.variant_id: row for row, entry in enumerate(normalized)
        }
        self._syncing_selection = True
        try:
            self._gallery_model.set_entries(normalized)
            self.setCurrentIndex(QModelIndex())
        finally:
            self._syncing_selection = False
        self.verticalScrollBar().setValue(0)
        self._last_requested_count = 0
        self._layout_timer.start(0)
        self._load_timer.start(0)

    def set_canvas_size(self, width: int, height: int) -> None:
        """设置字库田字格尺寸；外围工作区由委托按两轴 130% 推导。"""
        normalized = QSize(max(1, int(width)), max(1, int(height)))
        if normalized == self._delegate.canvas_size:
            return
        self._delegate.set_canvas_size(normalized.width(), normalized.height())
        self.viewport().update()

    def set_column_count(self, columns: int) -> None:
        normalized = max(self.MIN_COLUMNS, min(self.MAX_COLUMNS, int(columns)))
        if normalized == self._column_count:
            return
        anchor = self._first_visible_variant_id()
        self._column_count = normalized
        self._layout_timer.start()
        if anchor:
            QTimer.singleShot(0, lambda target=anchor: self.set_selected_variant(target, center=False))

    def set_selected_variant(self, variant_id: str, *, center: bool = True) -> bool:
        row = self._row_by_variant.get(str(variant_id), -1)
        if row < 0:
            return False
        index = self._gallery_model.index(row, 0)
        self._syncing_selection = True
        try:
            self.setCurrentIndex(index)
        finally:
            self._syncing_selection = False
        hint = (
            QAbstractItemView.ScrollHint.PositionAtCenter
            if center
            else QAbstractItemView.ScrollHint.PositionAtTop
        )
        self.scrollTo(index, hint)
        self._schedule_load_update()
        return True

    def selected_variant_id(self) -> str:
        index = self.currentIndex()
        return str(index.data(ExportGalleryModel.VARIANT_ID_ROLE) or "") if index.isValid() else ""

    def refresh_memory_budget(self) -> None:
        """立即重新采样可用内存，便于页面或测试主动触发。"""
        self._refresh_memory_budget()

    def shutdown(self, *, clear_cache: bool = True) -> None:
        """停止画廊后台活动，并作废尚未完成的载入结果。"""
        self._layout_timer.stop()
        self._load_timer.stop()
        self._memory_timer.stop()
        self._generation += 1
        self._cancel_queued_requests()
        self._cache.set_pinned(())
        if clear_cache:
            self._cache.clear()
            self._failed_keys.clear()

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._layout_timer.start()

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        self._layout_timer.start(0)
        self._load_timer.start(0)

    def closeEvent(self, event: Any) -> None:
        self.shutdown()
        super().closeEvent(event)

    def _initial_cache_budget(self) -> int:
        if self._fixed_cache_budget is not None:
            return max(MIB, int(self._fixed_cache_budget))
        try:
            total, available = self._memory_provider()
        except (OSError, TypeError, ValueError):
            total, available = 8 * 1024 * MIB, 2 * 1024 * MIB
        return calculate_thumbnail_cache_budget(total, available)

    def _refresh_memory_budget(self) -> None:
        if self._fixed_cache_budget is not None:
            return
        try:
            total, available = self._memory_provider()
            proposed = calculate_thumbnail_cache_budget(total, available)
        except (OSError, TypeError, ValueError):
            return
        current = self._cache_budget
        if proposed > current and proposed < int(current * 1.25):
            return
        self._cache_budget = proposed
        self._refresh_cache_limits()
        self._thread_pool.setMaxThreadCount(1 if proposed <= NORMAL_MIN_CACHE_BUDGET else 2)

    def _refresh_cache_limits(self) -> None:
        estimated_item = self._target_bucket * self._target_bucket * 4 + WeightedImageCache.ENTRY_OVERHEAD
        max_items = min(2048, max(1, self._cache_budget // max(1, estimated_item)))
        self._cache.set_limits(self._cache_budget, max_items)

    def _apply_grid_layout(self) -> None:
        available_width = max(1, self.viewport().width())
        # Qt 的 IconMode 在单元格总宽刚好等于视口时可能把末项换行。
        cell_width = max(1, available_width // self._column_count - 1)
        image_side = max(42, min(176, cell_width - 10))
        cell_height = image_side + ExportGalleryDelegate.FOOTER_HEIGHT + 10
        self.setGridSize(QSize(cell_width, cell_height))
        device_ratio = max(1.0, float(self.devicePixelRatioF()))
        raw_target = max(32, round((image_side - 10) * device_ratio))
        new_bucket = max(32, ((raw_target + 15) // 16) * 16)
        if new_bucket != self._target_bucket:
            self._target_bucket = new_bucket
            self._cache.remove_if(
                lambda key: isinstance(key, tuple) and key[-1] != new_bucket
            )
            self._failed_keys = {
                key for key in self._failed_keys
                if not isinstance(key, tuple) or key[-1] == new_bucket
            }
        self._refresh_cache_limits()
        self.scheduleDelayedItemsLayout()
        self.viewport().update()
        self._schedule_load_update()

    def _cache_key(self, entry: ExportGalleryEntry) -> tuple[int, str, str, int]:
        return self._generation, entry.variant_id, entry.image_path, self._target_bucket

    def _cached_image_for_entry(
        self,
        entry: ExportGalleryEntry,
    ) -> tuple[QImage | None, bool]:
        key = self._cache_key(entry)
        return self._cache.get(key), key in self._failed_keys

    def _schedule_load_update(self, _value: int | None = None) -> None:
        self._load_timer.start()

    def _visible_and_requested_rows(self) -> tuple[list[int], list[int]]:
        count = self.entry_count
        if count <= 0:
            return [], []
        grid_height = max(1, self.gridSize().height())
        columns = max(1, self._column_count)
        scroll_value = max(0, self.verticalScrollBar().value())
        first_row = max(0, scroll_value // grid_height)
        visible_row_count = max(1, (self.viewport().height() + grid_height - 1) // grid_height + 1)
        last_row = first_row + visible_row_count
        visible_start = min(count, first_row * columns)
        visible_end = min(count, last_row * columns)
        visible = list(range(visible_start, visible_end))

        requested_first_row = max(0, first_row - visible_row_count * self.PREFETCH_VIEWPORTS)
        requested_last_row = last_row + visible_row_count * self.PREFETCH_VIEWPORTS
        requested_start = min(count, requested_first_row * columns)
        requested_end = min(count, requested_last_row * columns)
        before = list(range(requested_start, visible_start))
        after = list(range(visible_end, requested_end))
        requested = visible + after + list(reversed(before))
        return visible, requested[: max(len(visible), self.MAX_REQUESTS)]

    def _schedule_visible_loads(self) -> None:
        visible_rows, requested_rows = self._visible_and_requested_rows()
        visible_keys: list[Hashable] = []
        requested_keys: set[Hashable] = set()
        for row in visible_rows:
            entry = self._gallery_model.entry_at(row)
            if entry is not None:
                visible_keys.append(self._cache_key(entry))
        self._cache.set_pinned(visible_keys)

        for row in requested_rows:
            entry = self._gallery_model.entry_at(row)
            if entry is not None:
                requested_keys.add(self._cache_key(entry))
        self._last_requested_count = len(requested_keys)
        self._cancel_unneeded_queued_requests(requested_keys)

        for row in requested_rows:
            entry = self._gallery_model.entry_at(row)
            if entry is None:
                continue
            key = self._cache_key(entry)
            if key in self._cache or key in self._failed_keys or key in self._pending:
                continue
            self._start_load(row, entry, key)

    def _start_load(
        self,
        row: int,
        entry: ExportGalleryEntry,
        key: Hashable,
    ) -> None:
        self._token += 1
        token = self._token
        generation = self._generation
        target_size = QSize(self._target_bucket, self._target_bucket)
        decode_limit = min(
            128 * MIB,
            max(16 * MIB, self._cache_budget // 2),
        )

        def load() -> _DecodeResult:
            image = self._image_loader(entry.image_path, target_size, decode_limit)
            if not isinstance(image, QImage):
                image = QImage()
            return _DecodeResult(
                generation=generation,
                key=key,
                row=row,
                token=token,
                image=image,
                failed=image.isNull(),
            )

        worker = FunctionWorker(load)
        # 待处理表负责 QRunnable 的生命周期，避免任务完成与排队信号之间
        # Qt 自动删除 C++ 对象，使快速滚动时无法安全取消或去重。
        worker.setAutoDelete(False)
        worker.signals.finished.connect(self._load_finished)
        worker.signals.failed.connect(
            lambda _message, target_key=key, target_token=token: (
                self._load_failed(target_key, target_token)
            )
        )
        self._pending[key] = (token, worker)
        self._thread_pool.start(worker)

    @Slot(object)
    def _load_finished(self, result: object) -> None:
        if not isinstance(result, _DecodeResult):
            return
        pending = self._pending.get(result.key)
        if pending is None or pending[0] != result.token:
            return
        self._pending.pop(result.key, None)
        if result.generation != self._generation or result.key != self._key_for_row(result.row):
            self._emit_idle_if_needed()
            return
        if result.failed:
            self._failed_keys.add(result.key)
        else:
            self._cache.put(result.key, result.image)
        index = self._gallery_model.index(result.row, 0)
        self._gallery_model.dataChanged.emit(index, index, [ExportGalleryModel.ENTRY_ROLE])
        self.viewport().update(self.visualRect(index))
        self._emit_idle_if_needed()

    def _load_failed(self, key: Hashable, token: int) -> None:
        pending = self._pending.get(key)
        if pending is None or pending[0] != token:
            return
        self._pending.pop(key, None)
        self._failed_keys.add(key)
        self.viewport().update()
        self._emit_idle_if_needed()

    def _key_for_row(self, row: int) -> Hashable | None:
        entry = self._gallery_model.entry_at(row)
        return self._cache_key(entry) if entry is not None else None

    def _cancel_unneeded_queued_requests(self, requested_keys: set[Hashable]) -> None:
        for key, (_token, worker) in tuple(self._pending.items()):
            if key in requested_keys:
                continue
            if self._thread_pool.tryTake(worker):
                self._pending.pop(key, None)

    def _cancel_queued_requests(self) -> None:
        for key, (_token, worker) in tuple(self._pending.items()):
            if self._thread_pool.tryTake(worker):
                self._pending.pop(key, None)

    def _emit_idle_if_needed(self) -> None:
        if not self._pending:
            self.loading_idle.emit()

    @Slot(QModelIndex, QModelIndex)
    def _current_changed(self, current: QModelIndex, _previous: QModelIndex) -> None:
        if self._syncing_selection or not current.isValid():
            return
        variant_id = str(current.data(ExportGalleryModel.VARIANT_ID_ROLE) or "")
        if variant_id:
            self.variant_selected.emit(variant_id)

    def _first_visible_variant_id(self) -> str:
        index = self.indexAt(QPoint(1, 1))
        if not index.isValid():
            visible, _requested = self._visible_and_requested_rows()
            index = self._gallery_model.index(visible[0], 0) if visible else QModelIndex()
        return str(index.data(ExportGalleryModel.VARIANT_ID_ROLE) or "") if index.isValid() else ""
