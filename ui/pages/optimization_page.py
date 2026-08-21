"""自动优化工作台。"""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from threading import Event
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw
from PySide6.QtCore import (
    QEvent,
    QModelIndex,
    QObject,
    QRunnable,
    QSize,
    Qt,
    QThreadPool,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import QBrush, QColor, QIcon, QImage, QPalette, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QListView,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

import config
from core.optimizer import OptimizationCancelled
from core.source_classification import TRANSPARENCY_SOURCE_PHOTOSHOP_ALPHA
from data.log_manager import buffered_log_writes, write_log
from services.batch_persistence import BatchJournalUncertainError
from services.background_model_service import (
    BACKGROUND_MODEL_REGISTRY,
    NO_MODEL_ENGINE_ID,
    BackgroundModelContext,
    build_candidate_cache_key,
)
from services.glyph_service import GlyphService
from services.library_summary_service import summarize_glyph_service
from services.file_transaction_recovery import FileTransactionCommitUncertainError
from services.optimization_service import (
    CANDIDATE_TYPE_ALPHA_DENOISED,
    CANDIDATE_TYPE_DIRECT,
    CANDIDATE_TYPE_OPTIMIZED,
    CANDIDATE_TYPE_TRANSPARENT,
    OptimizationService,
)
from services.workflow_status_service import (
    MARKER_STRUCTURE_REVIEW,
    OPTIMIZATION_STATUS_FILTERS,
    PHASE_FILTER_ALL,
    PHASE_OPTIMIZATION,
    PHASE_STATUS_COLORS,
    STAGE_PENDING_OPTIMIZATION,
    STATUS_OPTIMIZED,
    WORKFLOW_MARKERS,
    project_stage_status,
)
from ui.workers import FunctionWorker, log_background_exception
from ui.widgets.adjustable_tree_columns import AdjustableTreeColumns
from ui.widgets.glyph_rename_dialog import run_glyph_rename_dialog
from ui.widgets.two_line_status_delegate import (
    TwoLineStatusDelegate,
    set_two_line_status,
)
from utils.batch_observability import BatchTiming, ProgressThrottle, format_elapsed_time
from utils.file_utils import pinyin_natural_key


ThumbnailSignature = tuple[str, int, int, bool]


class _OptimizationBatchSignals(QObject):
    """整库自动优化后台任务信号。"""

    progress = Signal(str, int, int)
    finished = Signal(object)
    failed = Signal(str)


class _OptimizationBatchWorker(QRunnable):
    """顺序优化待处理字形，单项失败时继续执行剩余任务。"""

    def __init__(
        self,
        service: OptimizationService,
        items: list[dict[str, Any]],
        engine_context: BackgroundModelContext,
        skipped_count: int,
    ) -> None:
        super().__init__()
        self._service = service
        self._items = tuple(dict(item) for item in items)
        self._engine_context = engine_context
        self._skipped_count = max(0, int(skipped_count))
        self._cancel_event = Event()
        self.signals = _OptimizationBatchSignals()

    def request_cancel(self) -> None:
        """请求在当前安全检查点停止批量任务。"""
        self._cancel_event.set()

    def is_cancel_requested(self) -> bool:
        """返回用户是否已请求停止。"""
        return self._cancel_event.is_set()

    @Slot()
    def run(self) -> None:
        try:
            with buffered_log_writes():
                result = self._run_batch()
        except Exception as exc:
            log_background_exception("整库自动优化")
            try:
                self.signals.failed.emit(str(exc))
            except RuntimeError:
                pass
        else:
            try:
                self.signals.finished.emit(result)
            except RuntimeError:
                pass

    def _run_batch(self) -> dict[str, Any]:
        persistence = self._service.create_batch_persistence()
        timing = BatchTiming()
        progress = ProgressThrottle(self.signals.progress.emit)
        succeeded_ids: list[str] = []
        review_required: list[dict[str, str]] = []
        failures: list[dict[str, str]] = []
        total = len(self._items)
        handled_count = 0
        stopped = False
        try:
            for current, item in enumerate(self._items, 1):
                if self.is_cancel_requested():
                    stopped = True
                    break
                label = self._item_label(item)
                progress.emit(
                    f"正在自动优化 {current}/{total}：{label}",
                    current - 1,
                    total,
                    stage="优化处理",
                )
                try:
                    with timing.measure("候选生成"):
                        candidate = self._service.generate_batch_candidate(
                            item,
                            engine_context=self._engine_context,
                            cancel_check=self.is_cancel_requested,
                        )
                    if self.is_cancel_requested():
                        stopped = True
                        break
                    with timing.measure("结果保存"):
                        self._service.save_selection(
                            item,
                            candidate,
                            round_number=1,
                            persistence=persistence,
                        )
                except OptimizationCancelled:
                    stopped = True
                    break
                except (
                    BatchJournalUncertainError,
                    FileTransactionCommitUncertainError,
                ):
                    # 无法确认数据库提交时必须终止整批，避免后续检查点或
                    # 同字覆盖破坏仍待启动恢复裁决的图片与状态。
                    raise
                except Exception as exc:
                    failures.append({
                        "键": str(item.get("键", "")),
                        "字形": label,
                        "错误": str(exc) or type(exc).__name__,
                    })
                else:
                    succeeded_ids.append(str(item.get("键", "")))
                    if OptimizationService.requires_structure_review(candidate):
                        review = OptimizationService.structure_review_metadata(candidate)
                        review_required.append({
                            "键": str(item.get("键", "")),
                            "字形": label,
                            "原因": str(review.get("原因", "结构保护未通过")),
                        })
                handled_count += 1
                progress.emit(
                    f"已处理 {current}/{total}：{label}",
                    current,
                    total,
                    stage="优化处理",
                )
                with timing.measure("状态提交"):
                    persistence.checkpoint_if_due()
                if self.is_cancel_requested():
                    stopped = True
                    break
            with timing.measure("状态提交"):
                persistence.finish()
        except Exception:
            try:
                persistence.leave_for_recovery()
            except Exception:
                pass
            try:
                progress.flush()
            except Exception:
                pass
            write_log(
                timing.format_summary(
                    "自动优化",
                    {
                        "成功": len(succeeded_ids),
                        "需人工核对": len(review_required),
                        "跳过": self._skipped_count,
                        "失败": len(failures) + 1,
                        "未处理": max(0, total - handled_count),
                    },
                    stopped=self.is_cancel_requested(),
                )
            )
            raise
        unprocessed_count = max(0, total - handled_count)
        progress.emit(
            (
                f"已停止自动优化，完成 {handled_count}/{total}"
                if stopped
                else f"自动优化处理完成 {handled_count}/{total}"
            ),
            handled_count,
            total,
            force=True,
            stage="停止" if stopped else "完成",
        )
        write_log(
            timing.format_summary(
                "自动优化",
                {
                    "成功": len(succeeded_ids),
                    "需人工核对": len(review_required),
                    "跳过": self._skipped_count,
                    "失败": len(failures),
                    "未处理": unprocessed_count,
                },
                stopped=stopped,
            )
        )
        return {
            "已停止": stopped,
            "成功": len(succeeded_ids),
            "成功字形": succeeded_ids,
            "需人工核对": len(review_required),
            "需人工核对详情": review_required,
            "跳过": self._skipped_count,
            "失败": len(failures),
            "失败详情": failures,
            "未处理": unprocessed_count,
            "待处理总数": total,
            "总耗时秒": timing.finish(),
        }

    @staticmethod
    def _item_label(item: dict[str, Any]) -> str:
        filename = os.path.splitext(os.path.basename(str(item.get("原始文件名", ""))))[0]
        if filename:
            return filename
        char = str(item.get("归属字", "?"))
        return f"{char}-字形{item.get('变体序号', 1)}"


class PreserveTextColorDelegate(QStyledItemDelegate):
    """选中列表项时保留数据中设置的文字颜色。"""

    def initStyleOption(self, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        super().initStyleOption(option, index)
        foreground = index.data(Qt.ItemDataRole.ForegroundRole)
        if isinstance(foreground, QBrush):
            option.palette.setBrush(QPalette.ColorRole.HighlightedText, foreground)


class OptimizationPage(QWidget):
    """按字形逐个生成、比较并保存自动优化结果。"""

    home_requested = Signal()
    selection_saved = Signal(str)
    summary_changed = Signal(object)
    status_message = Signal(str)

    MAX_ROUNDS = 5
    PREVIEW_SIZE = (340, 275)
    CARD_SIZE = (150, 105)
    PREVIEW_MIN_HEIGHT = 204
    CANDIDATE_MIN_CELL_WIDTH = 170
    CANDIDATE_CELL_HEIGHT = 170
    CANDIDATE_MAX_COLUMNS = 4
    LIST_PANEL_MIN_WIDTH = 300
    LIST_PANEL_DEFAULT_WIDTH = 340
    LIST_PANEL_MAX_WIDTH = 420
    TREE_COLUMN_PADDING = 8
    LIST_THUMBNAIL_SYNC_LIMIT = 24
    LIST_THUMBNAIL_BATCH_SIZE = 12
    LIST_THUMBNAIL_CACHE_ITEMS = 512
    CANDIDATE_CACHE_MAX_ITEMS = 16
    CANDIDATE_CACHE_MAX_BYTES = 192 * 1024 * 1024
    CANDIDATE_CACHE_ENTRY_OVERHEAD = 4096
    DEFAULT_PREVIEW_BACKGROUND = "白底"
    CANDIDATE_TRANSPARENT_BACKGROUND = False
    PIPELINE_VERSION = "automatic-optimization-v2"
    CANDIDATE_COLORS = {
        CANDIDATE_TYPE_DIRECT: "#39d353",
        CANDIDATE_TYPE_TRANSPARENT: "#f2b84b",
        CANDIDATE_TYPE_OPTIMIZED: "#ffffff",
    }
    STRUCTURE_RISK_COLOR = "#ff8a65"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._glyph_service: GlyphService | None = None
        self._service: OptimizationService | None = None
        self._workflow_summary: dict[str, Any] = {}
        self._finished_dir = ""
        self._items: list[dict[str, Any]] = []
        self._visible_items: list[dict[str, Any]] = []
        self._variant_nodes: list[QTreeWidgetItem] = []
        self._list_thumbnail_cache: OrderedDict[
            str,
            tuple[ThumbnailSignature, QIcon],
        ] = OrderedDict()
        self._list_thumbnail_inflight: set[tuple[str, ThumbnailSignature]] = set()
        self._list_thumbnail_workers: set[FunctionWorker] = set()
        self._list_thumbnail_refresh_pending = False
        self._list_thumbnail_generation = 0
        self._list_thumbnail_placeholder: QIcon | None = None
        self._current_item: dict[str, Any] | None = None
        self._candidates: list[dict[str, Any]] = []
        self._candidate_cache: OrderedDict[
            object,
            list[dict[str, Any]],
        ] = OrderedDict()
        self._candidate_cache_costs: dict[object, int] = {}
        self._candidate_cache_bytes = 0
        self._library_identity = ""
        self._engine_context: BackgroundModelContext = BACKGROUND_MODEL_REGISTRY.create_context()
        self._selected_index = -1
        self._branch_dirty = False
        self._round_number = 1
        self._history: list[str] = []
        self._preview_background = self.DEFAULT_PREVIEW_BACKGROUND
        self._fit_preview = True
        self._request_id = 0
        self._busy = False
        self._preview_refresh_pending = False
        self._candidate_layout_pending = False
        self._candidate_columns = 1
        self._reoptimization_key = ""
        self._original_image_path = ""
        self._original_image: Image.Image | None = None
        self._thread_pool = QThreadPool.globalInstance()
        self._workers: set[FunctionWorker] = set()
        self._bulk_worker: _OptimizationBatchWorker | None = None
        self._bulk_started_at: float | None = None
        self._build_ui()
        self._populate_engines()
        self._set_workspace_enabled(False)

    @property
    def is_batch_running(self) -> bool:
        """返回整库自动优化任务是否仍在运行。"""
        return self._bulk_worker is not None

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        header = QVBoxLayout()
        header.setSpacing(4)
        title_row = QHBoxLayout()
        title = QLabel("自动优化")
        title.setProperty("role", "pageTitle")
        title_row.addWidget(title)
        title_row.addStretch()
        self._complete_button = QPushButton("批量自动优化")
        self._complete_button.setProperty("role", "primary")
        self._complete_button.clicked.connect(self._complete_optimization)
        title_row.addWidget(self._complete_button)
        self._home_button = QPushButton("返回首页")
        self._home_button.clicked.connect(self.home_requested)
        title_row.addWidget(self._home_button)
        header.addLayout(title_row)

        self._library_label = QLabel("请选择字库")
        self._library_label.setProperty("role", "muted")
        self._library_label.setWordWrap(True)
        self._library_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        header.addWidget(self._library_label)
        self._bulk_progress = QProgressBar()
        self._bulk_progress.setRange(0, 1)
        self._bulk_progress.setValue(0)
        self._bulk_progress.setFormat("准备自动优化 0/0　0%")
        self._bulk_progress.setTextVisible(True)
        self._bulk_progress.setFixedHeight(22)
        self._bulk_progress.setVisible(False)
        bulk_row = QHBoxLayout()
        bulk_row.setSpacing(8)
        bulk_row.addWidget(self._bulk_progress, 1)
        self._stop_bulk_button = QPushButton("停止批量优化")
        self._stop_bulk_button.clicked.connect(self._request_stop_bulk)
        self._stop_bulk_button.setVisible(False)
        bulk_row.addWidget(self._stop_bulk_button)
        header.addLayout(bulk_row)
        root.addLayout(header)

        self._main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._main_splitter.addWidget(self._build_list_panel())
        self._main_splitter.addWidget(self._build_workspace())
        self._main_splitter.addWidget(self._build_scheme_panel())
        self._main_splitter.setStretchFactor(0, 0)
        self._main_splitter.setStretchFactor(1, 1)
        self._main_splitter.setStretchFactor(2, 0)
        self._main_splitter.setSizes([self.LIST_PANEL_DEFAULT_WIDTH, 730, 280])
        self._main_splitter.splitterMoved.connect(
            lambda _position, _index: self._schedule_candidate_layout()
        )
        root.addWidget(self._main_splitter, 1)

    def _build_list_panel(self) -> QWidget:
        panel = QFrame()
        self._list_panel = panel
        panel.setProperty("role", "card")
        panel.setMinimumWidth(self.LIST_PANEL_MIN_WIDTH)
        panel.setMaximumWidth(self.LIST_PANEL_MAX_WIDTH)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        heading = QHBoxLayout()
        list_title = QLabel("字形列表")
        list_title.setStyleSheet("font-size: 18px; font-weight: 700;")
        heading.addWidget(list_title)
        heading.addStretch()
        self._list_count_label = QLabel("显示 / 总数：0 / 0")
        self._list_count_label.setProperty("role", "muted")
        heading.addWidget(self._list_count_label)
        layout.addLayout(heading)
        search_row = QHBoxLayout()
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("搜索字符或文件名")
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
        self._status_combo = QComboBox()
        self._status_combo.addItems(OPTIMIZATION_STATUS_FILTERS)
        self._status_combo.setCurrentText(PHASE_FILTER_ALL)
        self._status_combo.currentTextChanged.connect(self._refresh_list)
        self._status_combo.setToolTip("按自动优化状态筛选")
        self._status_combo.setStyleSheet(
            "QComboBox { padding-left: 4px; padding-right: 4px; }"
        )
        self._status_combo.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Fixed,
        )
        filter_sort_row.addWidget(self._status_combo, 5)
        self._sort_combo = QComboBox()
        self._sort_combo.addItems(("拼音顺序", "导入顺序", "低分优先"))
        self._sort_combo.currentTextChanged.connect(self._refresh_list)
        self._sort_combo.setStyleSheet(
            "QComboBox { padding-left: 4px; padding-right: 4px; }"
        )
        self._sort_combo.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Fixed,
        )
        filter_sort_row.addWidget(self._sort_combo, 4)
        layout.addLayout(filter_sort_row)
        self._item_tree = QTreeWidget()
        self._item_tree.setColumnCount(3)
        self._item_tree.setHeaderLabels(("字形与文件", "状态与提示", "得分"))
        self._item_tree.setRootIsDecorated(True)
        self._item_tree.setIndentation(14)
        self._item_tree.setUniformRowHeights(False)
        self._item_tree.setAlternatingRowColors(False)
        self._item_tree.setWordWrap(True)
        self._item_tree.setIconSize(QSize(38, 38))
        self._item_tree.setAnimated(False)
        self._item_tree.setItemDelegate(PreserveTextColorDelegate(self._item_tree))
        self._item_tree.setItemDelegateForColumn(
            1,
            TwoLineStatusDelegate(self._item_tree),
        )
        self._item_tree.setStyleSheet(
            "QTreeWidget { background: #171b22; border: 1px solid #37404d; }"
            "QTreeWidget::item { min-height: 26px; padding: 1px 3px; }"
            "QTreeWidget::item:selected { background: #3c4773; }"
        )
        self._item_tree_columns = AdjustableTreeColumns(
            self._item_tree,
            {
                0: max(
                    160,
                    self._item_tree.fontMetrics().horizontalAdvance("字形与文件") + 24,
                ),
                1: self._optimization_status_column_width("已优化 99/99"),
            },
            {
                0: 160,
                1: self._optimization_status_column_width("状态与提示"),
                2: max(
                    56,
                    self._item_tree.fontMetrics().horizontalAdvance("100分")
                    + self.TREE_COLUMN_PADDING,
                ),
            },
        )
        self._list_row_height_timer = QTimer(self)
        self._list_row_height_timer.setSingleShot(True)
        self._list_row_height_timer.timeout.connect(self._update_list_row_heights)
        self._item_tree.header().sectionResized.connect(
            self._schedule_list_row_height_update
        )
        self._item_tree.currentItemChanged.connect(self._on_item_selected)
        self._item_tree.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self._item_tree.customContextMenuRequested.connect(
            self._show_glyph_context_menu
        )
        self._item_tree.itemExpanded.connect(
            lambda _item: self._schedule_visible_list_thumbnails()
        )
        self._item_tree.itemCollapsed.connect(
            lambda _item: self._schedule_visible_list_thumbnails()
        )
        self._item_tree.verticalScrollBar().valueChanged.connect(
            lambda _value: self._schedule_visible_list_thumbnails()
        )
        self._item_tree.viewport().installEventFilter(self)
        layout.addWidget(self._item_tree, 1)
        self._summary_label = QLabel("待优化 0　已优化 0")
        self._summary_label.setProperty("role", "muted")
        self._summary_label.setWordWrap(True)
        self._summary_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        self._summary_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._summary_label)
        navigation = QHBoxLayout()
        navigation.setSpacing(8)
        self._previous_button = QPushButton("上一字形")
        self._previous_button.clicked.connect(lambda: self._move_current(-1))
        navigation.addWidget(self._previous_button)
        self._next_button = QPushButton("下一字形")
        self._next_button.clicked.connect(lambda: self._move_current(1))
        navigation.addWidget(self._next_button)
        layout.addLayout(navigation)
        return panel

    def _optimization_status_column_width(self, value: str) -> int:
        return max(
            78,
            self._item_tree.fontMetrics().horizontalAdvance(value) + 24,
        )

    def _build_workspace(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(8)
        preview_row = QHBoxLayout()
        preview_row.setSpacing(8)
        preview_row.addWidget(self._make_preview_panel("原始图片", "_original_preview"), 1)
        preview_row.addWidget(self._build_preview_tools())
        preview_row.addWidget(self._make_preview_panel("选中效果", "_selected_preview"), 1)
        layout.addLayout(preview_row, 2)

        candidate_heading = QHBoxLayout()
        self._candidate_title = QLabel("第 1 轮候选 · 共 0 张")
        self._candidate_title.setWordWrap(True)
        self._candidate_title.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        candidate_heading.addWidget(self._candidate_title, 1)
        order_hint = QLabel("直接采用 / 仅透明化 / 寻优优化")
        order_hint.setProperty("role", "muted")
        order_hint.setWordWrap(True)
        order_hint.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        order_hint.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        candidate_heading.addWidget(order_hint, 1)
        layout.addLayout(candidate_heading)
        candidate_panel = QFrame()
        self._candidate_panel = candidate_panel
        candidate_panel.setObjectName("candidatePanel")
        candidate_panel.setStyleSheet(
            "QFrame#candidatePanel { background: #202630; border: 1px solid #37404D; border-radius: 5px; }"
            "QListWidget#candidateList { background: transparent; border: none; border-radius: 0; }"
            "QLabel#candidateMessage { border: none; padding: 4px 9px 7px 9px; }"
        )
        candidate_layout = QVBoxLayout(candidate_panel)
        candidate_layout.setContentsMargins(0, 0, 0, 0)
        candidate_layout.setSpacing(0)
        self._candidate_list = QListWidget()
        self._candidate_list.setObjectName("candidateList")
        self._candidate_list.setViewMode(QListWidget.ViewMode.IconMode)
        self._candidate_list.setFlow(QListView.Flow.LeftToRight)
        self._candidate_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._candidate_list.setMovement(QListWidget.Movement.Static)
        self._candidate_list.setIconSize(QSize(*self.CARD_SIZE))
        self._candidate_list.setGridSize(
            QSize(self.CANDIDATE_MIN_CELL_WIDTH, self.CANDIDATE_CELL_HEIGHT)
        )
        self._candidate_list.setSpacing(0)
        self._candidate_list.setWrapping(True)
        self._candidate_list.setWordWrap(True)
        self._candidate_list.setUniformItemSizes(True)
        self._candidate_list.setVerticalScrollMode(
            QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        self._candidate_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._candidate_list.viewport().installEventFilter(self)
        self._candidate_list.setItemDelegate(PreserveTextColorDelegate(self._candidate_list))
        self._candidate_list.currentRowChanged.connect(self._select_candidate)
        self._candidate_list.itemClicked.connect(lambda _item: self._mark_branch_dirty())
        candidate_layout.addWidget(self._candidate_list, 1)
        self._message_label = QLabel("请选择左侧字形开始处理")
        self._message_label.setObjectName("candidateMessage")
        self._message_label.setProperty("role", "muted")
        self._message_label.setWordWrap(True)
        candidate_layout.addWidget(self._message_label)
        layout.addWidget(candidate_panel, 3)
        return panel

    def _build_preview_tools(self) -> QWidget:
        """构建位于原图和效果图之间的预览控制栏。"""
        tools = QWidget()
        self._preview_tools = tools
        tools.setFixedWidth(104)
        tool_layout = QVBoxLayout(tools)
        tool_layout.setContentsMargins(0, 0, 0, 0)
        tool_layout.setSpacing(5)
        tool_layout.addStretch()

        display_title = QLabel("显示")
        display_title.setProperty("role", "muted")
        display_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tool_layout.addWidget(display_title)
        self._fit_button = QPushButton("适合窗口")
        self._fit_button.setObjectName("compactButton")
        self._fit_button.setProperty("controlRole", "segment")
        self._fit_button.setFixedWidth(104)
        self._fit_button.setCheckable(True)
        self._fit_button.setChecked(True)
        self._fit_button.clicked.connect(lambda: self._set_fit_mode(True))
        tool_layout.addWidget(self._fit_button)
        self._actual_size_button = QPushButton("1:1")
        self._actual_size_button.setObjectName("compactButton")
        self._actual_size_button.setProperty("controlRole", "segment")
        self._actual_size_button.setFixedWidth(104)
        self._actual_size_button.setCheckable(True)
        self._actual_size_button.clicked.connect(lambda: self._set_fit_mode(False))
        tool_layout.addWidget(self._actual_size_button)
        self._hold_original_button = QPushButton("按住查看原图")
        self._hold_original_button.setObjectName("compactButton")
        self._hold_original_button.setFixedWidth(104)
        self._hold_original_button.pressed.connect(self._show_original_as_result)
        self._hold_original_button.released.connect(self._render_selected_preview)
        tool_layout.addWidget(self._hold_original_button)

        background_title = QLabel("效果图背景")
        background_title.setProperty("role", "muted")
        background_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tool_layout.addWidget(background_title)
        self._white_background_button = QPushButton("白底")
        self._white_background_button.setObjectName("compactButton")
        self._white_background_button.setProperty("controlRole", "segment")
        self._white_background_button.setFixedWidth(104)
        self._white_background_button.setCheckable(True)
        self._white_background_button.setChecked(True)
        self._white_background_button.clicked.connect(lambda: self._set_background("白底"))
        tool_layout.addWidget(self._white_background_button)
        self._transparent_background_button = QPushButton("透明底")
        self._transparent_background_button.setObjectName("compactButton")
        self._transparent_background_button.setProperty("controlRole", "segment")
        self._transparent_background_button.setFixedWidth(104)
        self._transparent_background_button.setCheckable(True)
        self._transparent_background_button.clicked.connect(lambda: self._set_background("透明底"))
        tool_layout.addWidget(self._transparent_background_button)
        tool_layout.addStretch()
        return tools

    def _make_preview_panel(self, title: str, attribute: str) -> QWidget:
        panel = QFrame()
        panel.setProperty("role", "card")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(QLabel(title))
        preview = QLabel("暂无图片")
        preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview.setMinimumSize(130, self.PREVIEW_MIN_HEIGHT)
        preview.setStyleSheet("background: #EEF0F3; color: #687080;")
        preview.installEventFilter(self)
        layout.addWidget(preview, 1)
        setattr(self, attribute, preview)
        return panel

    def _build_scheme_panel(self) -> QWidget:
        panel = QFrame()
        self._scheme_panel = panel
        panel.setProperty("role", "card")
        panel.setMinimumWidth(250)
        panel.setMaximumWidth(360)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        content = QWidget()
        self._scheme_fixed_content = content
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(15, 12, 15, 8)
        content_layout.setSpacing(5)

        engine_title = QLabel("处理引擎")
        engine_title.setObjectName("sectionTitle")
        content_layout.addWidget(engine_title)
        self._engine_combo = QComboBox()
        self._engine_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self._engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        content_layout.addWidget(self._engine_combo)
        self._engine_state_label = QLabel("传统图像管线")
        self._engine_state_label.setProperty("role", "muted")
        self._engine_state_label.setWordWrap(True)
        content_layout.addWidget(self._engine_state_label)
        self._engine_detail_label = QLabel("无学习模型 · 传统图像管线")
        self._engine_detail_label.setWordWrap(True)
        self._engine_detail_label.setProperty("role", "muted")
        content_layout.addWidget(self._engine_detail_label)
        self._model_settings_panel = QFrame()
        model_layout = QVBoxLayout(self._model_settings_panel)
        model_layout.setContentsMargins(0, 4, 0, 0)
        model_layout.addWidget(QLabel("模型设置将在安装具体模型后显示。"))
        self._model_settings_panel.setVisible(False)
        content_layout.addWidget(self._model_settings_panel)
        content_layout.addWidget(self._make_separator())

        title = QLabel("当前方案")
        title.setObjectName("sectionTitle")
        content_layout.addWidget(title)
        self._restart_button = QPushButton("从原图重新生成候选")
        self._restart_button.setObjectName("compactButton")
        self._restart_button.setProperty("role", "primary")
        self._restart_button.setToolTip("从当前字形的原图重新生成自动优化候选")
        self._restart_button.clicked.connect(self._restart_candidates)
        content_layout.addWidget(self._restart_button)
        self._scheme_label = QPlainTextEdit()
        self._scheme_label.setReadOnly(True)
        self._scheme_label.setPlainText("请从候选效果中选择一张图片。")
        self._scheme_label.setMinimumHeight(100)
        self._scheme_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self._scheme_label.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self._scheme_label.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scheme_label.setStyleSheet(
            "QPlainTextEdit { background: #171B22; color: #A6B0BE; "
            "border: 1px solid #37404D; border-radius: 5px; padding: 5px; }"
        )
        content_layout.addWidget(self._scheme_label, 1)
        content_layout.addWidget(self._make_separator())

        explore_title = QLabel("方案探索")
        explore_title.setObjectName("sectionTitle")
        content_layout.addWidget(explore_title)
        self._explore_button = QPushButton("围绕选中结果继续探索")
        self._explore_button.setObjectName("compactButton")
        self._explore_button.setProperty("role", "primary")
        self._explore_button.clicked.connect(self._explore_selected)
        content_layout.addWidget(self._explore_button)
        round_row = QHBoxLayout()
        self._round_label = QLabel(f"第 1/{self.MAX_ROUNDS} 轮")
        self._round_label.setProperty("role", "muted")
        round_row.addWidget(self._round_label)
        self._round_progress = QProgressBar()
        self._round_progress.setRange(0, self.MAX_ROUNDS)
        self._round_progress.setValue(1)
        round_row.addWidget(self._round_progress, 1)
        content_layout.addLayout(round_row)
        content_layout.addWidget(self._make_separator())

        history_title = QLabel("处理记录")
        history_title.setObjectName("sectionTitle")
        content_layout.addWidget(history_title)
        self._history_label = QLabel("第 1 轮　基础候选")
        self._history_label.setProperty("role", "muted")
        self._history_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._history_label.setStyleSheet("font-size: 12px;")
        self._history_label.setFixedHeight(86)
        content_layout.addWidget(self._history_label)
        layout.addWidget(content, 1)

        action_panel = QWidget()
        action_layout = QVBoxLayout(action_panel)
        action_layout.setContentsMargins(15, 10, 15, 12)
        action_layout.setSpacing(7)
        action_row = QHBoxLayout()
        self._skip_button = QPushButton("跳过")
        self._skip_button.clicked.connect(lambda: self._move_current(1))
        action_row.addWidget(self._skip_button)
        self._save_button = QPushButton("采用并保存")
        self._save_button.setProperty("role", "primary")
        self._save_button.clicked.connect(self._save_selected)
        action_row.addWidget(self._save_button, 1)
        action_layout.addLayout(action_row)
        action_hint = QLabel("保存为自动优化稿，下一步提交手工审核")
        action_hint.setProperty("role", "muted")
        action_hint.setWordWrap(True)
        action_layout.addWidget(action_hint)
        layout.addWidget(action_panel)
        return panel

    @staticmethod
    def _make_separator() -> QFrame:
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        return separator

    def _populate_engines(self) -> None:
        """从模型注册表加载当前可用的处理引擎。"""
        self._engine_combo.blockSignals(True)
        self._engine_combo.clear()
        selected_index = 0
        for descriptor in BACKGROUND_MODEL_REGISTRY.list_descriptors(installed_only=True):
            self._engine_combo.addItem(descriptor.display_name, descriptor.engine_id)
            if descriptor.engine_id == self._engine_context.engine_id:
                selected_index = self._engine_combo.count() - 1
        self._engine_combo.setCurrentIndex(selected_index)
        self._engine_combo.blockSignals(False)
        self._update_engine_panel()

    def _on_engine_changed(self, index: int) -> None:
        engine_id = str(self._engine_combo.itemData(index) or NO_MODEL_ENGINE_ID)
        next_context = BACKGROUND_MODEL_REGISTRY.create_context(engine_id)
        if next_context == self._engine_context:
            self._update_engine_panel()
            return
        if self._branch_dirty and self._candidates:
            answer = QMessageBox.question(
                self,
                "切换处理引擎",
                "当前探索分支尚未保存。切换处理引擎后将从原图重新生成候选，是否继续？",
            )
            if answer != QMessageBox.StandardButton.Yes:
                previous_index = self._engine_combo.findData(self._engine_context.engine_id)
                self._engine_combo.blockSignals(True)
                self._engine_combo.setCurrentIndex(previous_index)
                self._engine_combo.blockSignals(False)
                return
        self._engine_context = next_context
        self._request_id += 1
        self._round_number = 1
        self._history = ["第 1 轮　基础候选"]
        self._branch_dirty = False
        self._update_engine_panel()
        if self._current_item:
            self._clear_candidates("处理引擎已切换，正在从原图重新生成候选。")
            self._load_candidates()

    def _update_engine_panel(self) -> None:
        descriptor = self._engine_context.descriptor
        if descriptor.engine_id == NO_MODEL_ENGINE_ID:
            state = "传统图像管线"
            detail = "无学习模型 · 传统图像管线"
        else:
            state = f"{descriptor.display_name} · {descriptor.version}"
            detail = f"{descriptor.display_name} · 输出 {descriptor.output_type}"
        self._engine_state_label.setText(state)
        self._engine_detail_label.setText(detail)
        self._model_settings_panel.setVisible(descriptor.engine_id != NO_MODEL_ENGINE_ID)

    def _candidate_cache_key(self, item: dict[str, Any]) -> object:
        source_fingerprint = str(
            item.get("原始MD5")
            or item.get("源文件MD5")
            or item.get("原始路径")
            or item.get("键")
        )
        inference_fingerprint = (
            "无学习模型"
            if self._engine_context.engine_id == NO_MODEL_ENGINE_ID
            else "按当前模型上下文生成"
        )
        return build_candidate_cache_key(
            library_id=self._library_identity or "未打开字库",
            variant_id=str(item.get("键", "")),
            source_fingerprint=source_fingerprint,
            context=self._engine_context,
            inference_fingerprint=inference_fingerprint,
            pipeline_version=self.PIPELINE_VERSION,
        )

    @classmethod
    def _estimate_candidate_cache_bytes(
        cls,
        candidates: list[dict[str, Any]],
    ) -> int:
        """估算候选集实际持有的 PIL 像素和 NumPy 存储空间。"""
        seen: set[int] = set()

        def estimate(value: object) -> int:
            value_id = id(value)
            if value_id in seen:
                return 0
            seen.add(value_id)
            if isinstance(value, np.ndarray):
                owner = value
                while isinstance(owner.base, np.ndarray):
                    owner = owner.base
                owner_id = id(owner)
                if owner_id != value_id:
                    if owner_id in seen:
                        return 0
                    seen.add(owner_id)
                return max(0, int(owner.nbytes))
            if isinstance(value, Image.Image):
                pixels = max(0, int(value.width) * int(value.height))
                mode = str(value.mode)
                if mode == "1":
                    return (pixels + 7) // 8
                if mode.startswith("I;16"):
                    return pixels * 2
                if mode in ("I", "F"):
                    return pixels * 4
                return pixels * max(1, len(value.getbands()))
            if isinstance(value, dict):
                return sum(estimate(item) for item in value.values())
            if isinstance(value, (list, tuple, set, frozenset)):
                return sum(estimate(item) for item in value)
            return 0

        return cls.CANDIDATE_CACHE_ENTRY_OVERHEAD + estimate(candidates)

    def _cached_candidates(
        self,
        cache_key: object,
    ) -> list[dict[str, Any]] | None:
        """读取并提升候选集的 LRU 顺序。"""
        candidates = self._candidate_cache.get(cache_key)
        if candidates is None:
            return None
        self._candidate_cache.move_to_end(cache_key)
        return candidates

    def _store_candidate_cache(
        self,
        cache_key: object,
        candidates: list[dict[str, Any]],
    ) -> bool:
        """按项目数和估算字节数缓存候选；超大单项保持不缓存。"""
        self._remove_candidate_cache(cache_key)
        cost = self._estimate_candidate_cache_bytes(candidates)
        item_limit = max(0, int(self.CANDIDATE_CACHE_MAX_ITEMS))
        byte_limit = max(0, int(self.CANDIDATE_CACHE_MAX_BYTES))
        if item_limit == 0 or byte_limit == 0 or cost > byte_limit:
            return False
        self._candidate_cache[cache_key] = candidates
        self._candidate_cache_costs[cache_key] = cost
        self._candidate_cache_bytes += cost
        while (
            len(self._candidate_cache) > item_limit
            or self._candidate_cache_bytes > byte_limit
        ):
            oldest_key = next(iter(self._candidate_cache))
            self._remove_candidate_cache(oldest_key)
        return cache_key in self._candidate_cache

    def _remove_candidate_cache(
        self,
        cache_key: object,
    ) -> list[dict[str, Any]] | None:
        """只移除缓存引用；当前界面或保存任务仍可继续持有所需对象。"""
        candidates = self._candidate_cache.pop(cache_key, None)
        cost = self._candidate_cache_costs.pop(cache_key, 0)
        self._candidate_cache_bytes = max(0, self._candidate_cache_bytes - cost)
        return candidates

    def _clear_candidate_cache(self) -> None:
        """清空候选缓存并重置字节统计。"""
        self._candidate_cache.clear()
        self._candidate_cache_costs.clear()
        self._candidate_cache_bytes = 0

    def _drop_candidate_cache(self, variant_id: str) -> None:
        stale_keys = [
            cache_key
            for cache_key in self._candidate_cache
            if getattr(cache_key, "variant_id", "") == variant_id
        ]
        for cache_key in stale_keys:
            self._remove_candidate_cache(cache_key)

    def _sync_history_view(self) -> None:
        self._history_label.setText("\n".join(self._history or ["第 1 轮　基础候选"]))

    def _mark_branch_dirty(self) -> None:
        if 0 <= self._selected_index < len(self._candidates):
            self._branch_dirty = True

    def _complete_optimization(self) -> None:
        """确认未保存分支后，启动当前字库的批量自动优化。"""
        if self._bulk_worker is not None or self._busy:
            return
        if not self._service:
            QMessageBox.information(self, "完成自动优化", "请先选择一个字库。")
            return
        if self._workers:
            QMessageBox.information(
                self,
                "完成自动优化",
                "当前字形仍在生成或保存候选，请等待任务结束后再执行整库自动优化。",
            )
            return
        if self._branch_dirty:
            if not self._confirm_save_current_before_bulk():
                return
            self._save_current_before_bulk()
            return
        self._confirm_and_start_bulk()

    def _confirm_save_current_before_bulk(self) -> bool:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("当前候选尚未保存")
        box.setText("当前字形的候选选择或探索分支尚未保存。")
        box.setInformativeText("必须先保存当前候选，才能开始整库自动优化。")
        save_button = box.addButton("保存并继续", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        return box.clickedButton() is save_button

    def _save_current_before_bulk(self) -> None:
        if (
            not self._service
            or not self._current_item
            or not 0 <= self._selected_index < len(self._candidates)
        ):
            QMessageBox.warning(
                self,
                "无法保存当前候选",
                "当前没有可保存的有效候选，整库自动优化尚未开始。",
            )
            return
        candidate = self._candidates[self._selected_index]
        if not self._service.is_candidate_valid(candidate):
            QMessageBox.warning(
                self,
                "无法保存当前候选",
                "当前候选的图片或分层数据无效，整库自动优化尚未开始。",
            )
            return
        if not self._confirm_structure_risk(candidate):
            return
        item = dict(self._current_item)
        key = str(item.get("键", ""))
        service = self._service
        round_number = self._round_number
        self._message_label.setText("正在保存当前候选，随后将准备整库自动优化……")
        self._start_task(
            lambda: service.save_selection(item, candidate, round_number),
            lambda _result: self._current_saved_before_bulk(key),
            lambda message: self._task_failed(
                "保存失败",
                message,
            ),
            lock_page=True,
        )

    def _confirm_structure_risk(self, candidate: dict[str, Any]) -> bool:
        """风险候选允许采用，但必须让用户看到具体结构告警。"""
        if not OptimizationService.requires_structure_review(candidate):
            return True
        review = OptimizationService.structure_review_metadata(candidate)
        reason = str(review.get("原因", "结构保护未通过"))
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("确认采用结构风险候选")
        box.setText("该结果已完成自动优化，但未通过全部结构保护。")
        box.setInformativeText(
            f"风险原因：{reason}\n\n"
            "综合得分只用于候选排序，不代表结构安全。"
            "确定仍然采用并提交手工审核吗？"
        )
        confirm_button = box.addButton("仍然采用", QMessageBox.ButtonRole.AcceptRole)
        cancel_button = box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(cancel_button)
        box.setEscapeButton(cancel_button)
        box.exec()
        return box.clickedButton() is confirm_button

    def _current_saved_before_bulk(self, key: str) -> None:
        if not self._service:
            return
        self._branch_dirty = False
        self._reoptimization_key = ""
        self._items = self._service.list_items()
        self._current_item = next(
            (item for item in self._items if str(item.get("键", "")) == key),
            self._current_item,
        )
        self.selection_saved.emit(key)
        if self._glyph_service is not None:
            self.summary_changed.emit(self._glyph_service)
        self.status_message.emit("当前自动优化稿保存成功")
        self._confirm_and_start_bulk()
        if self._bulk_worker is None:
            self._refresh_list(preferred_key=key)

    def _confirm_and_start_bulk(self) -> None:
        if not self._service or self._bulk_worker is not None:
            return
        all_items = self._service.list_items()
        pending_items, skipped_count = self._service.list_batch_items()
        pending_items = [dict(item) for item in pending_items]
        self._items = all_items
        if not pending_items:
            self._refresh_list()
            QMessageBox.information(
                self,
                "完成自动优化",
                "当前字库没有待优化字形，自动优化阶段已经完成。",
            )
            return
        if not self._confirm_bulk_optimization(len(pending_items), skipped_count):
            return

        worker = _OptimizationBatchWorker(
            self._service,
            pending_items,
            self._engine_context,
            skipped_count,
        )
        self._bulk_worker = worker
        self._bulk_started_at = time.perf_counter()
        self._request_id += 1
        self._busy = True
        worker.signals.progress.connect(self._bulk_progress_changed)
        worker.signals.finished.connect(
            lambda result, active=worker: self._bulk_finished(active, result)
        )
        worker.signals.failed.connect(
            lambda message, active=worker: self._bulk_failed(active, message)
        )
        total = len(pending_items)
        self._bulk_progress.setRange(0, total)
        self._bulk_progress.setValue(0)
        self._bulk_progress.setFormat(f"准备自动优化 0/{total}　%p%")
        self._bulk_progress.setVisible(True)
        self._set_bulk_stop_state(running=True)
        self._message_label.setText("正在准备整库自动优化……")
        self._set_workspace_enabled(False)
        try:
            self._thread_pool.start(worker)
        except Exception as exc:
            elapsed = self._bulk_elapsed_seconds()
            self._bulk_worker = None
            self._bulk_started_at = None
            self._busy = False
            self._bulk_progress.setVisible(False)
            self._set_bulk_stop_state(running=False)
            self._set_workspace_enabled(bool(self._candidates))
            QMessageBox.critical(
                self,
                "整库自动优化失败",
                f"无法启动整库自动优化任务：{exc}\n\n"
                f"总耗时：{format_elapsed_time(elapsed)}",
            )

    def _confirm_bulk_optimization(self, pending_count: int, skipped_count: int) -> bool:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("确认批量自动优化")
        box.setText(f"将自动处理当前字库中的 {pending_count} 个待优化字形。")
        box.setInformativeText(
            "批量处理会自动判断原图类型并选择处理结果，不会逐字等待人工确认。"
            "复杂背景、浅色或断裂笔画以及大块污点可能被误判，导致笔画缺失、"
            "污点残留或边缘发生变化。选中的结果会自动保存为自动优化稿，"
            "并进入待审核阶段。若没有候选通过全部结构保护，程序仍会保存"
            "风险最低的实际优化结果并标记为“结构需人工核对”，不会把它计为失败。\n\n"
            "原始图片不会被修改，处理完成后建议抽查自动优化结果。\n"
            f"完整寻优使用当前处理引擎“{self._engine_context.descriptor.display_name}”；"
            f"已有结果 {skipped_count} 个将保留并跳过。"
        )
        box.setStandardButtons(
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
        )
        box.setDefaultButton(QMessageBox.StandardButton.Cancel)
        confirm_button = box.button(QMessageBox.StandardButton.Ok)
        cancel_button = box.button(QMessageBox.StandardButton.Cancel)
        if confirm_button is not None:
            confirm_button.setText("确定")
        if cancel_button is not None:
            cancel_button.setText("取消")
            box.setEscapeButton(cancel_button)
        return box.exec() == QMessageBox.StandardButton.Ok.value

    def _request_stop_bulk(self) -> None:
        """确认后请求批量线程在安全检查点停止。"""
        worker = self._bulk_worker
        if worker is None or worker.is_cancel_requested():
            return
        if not self._confirm_stop_bulk():
            return
        if self._bulk_worker is not worker or worker.is_cancel_requested():
            return
        worker.request_cancel()
        self._set_bulk_stop_state(running=True, stopping=True)
        self._message_label.setText("正在停止批量优化，请等待当前操作安全结束……")

    def _confirm_stop_bulk(self) -> bool:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("停止批量优化")
        box.setText("确定停止当前批量自动优化任务吗？")
        box.setInformativeText(
            "已经成功保存的字形会保留；当前保存事务会先完成或回滚，"
            "尚未处理的字形仍保持待优化状态。"
        )
        stop_button = box.addButton("停止批量优化", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("继续运行", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        return box.clickedButton() is stop_button

    def _set_bulk_stop_state(self, running: bool, stopping: bool = False) -> None:
        self._stop_bulk_button.setVisible(running)
        self._stop_bulk_button.setEnabled(running and not stopping)
        self._stop_bulk_button.setText("正在停止…" if stopping else "停止批量优化")

    def _bulk_progress_changed(self, message: str, current: int, total: int) -> None:
        worker = self._bulk_worker
        if worker is None:
            return
        safe_total = max(1, int(total))
        safe_current = max(0, min(int(current), safe_total))
        self._bulk_progress.setRange(0, safe_total)
        self._bulk_progress.setValue(safe_current)
        if worker.is_cancel_requested():
            stopping_message = "正在停止批量优化，请等待当前操作安全结束……"
            self._bulk_progress.setFormat(f"{stopping_message}　%p%")
            self._message_label.setText(stopping_message)
            self._set_bulk_stop_state(running=True, stopping=True)
            return
        self._bulk_progress.setFormat(f"{message}　%p%")
        self._message_label.setText(message)

    def _bulk_finished(
        self,
        worker: _OptimizationBatchWorker,
        result: object,
    ) -> None:
        if worker is not self._bulk_worker:
            return
        data = result if isinstance(result, dict) else {}
        elapsed = self._bulk_elapsed_seconds(data)
        self._bulk_worker = None
        self._bulk_started_at = None
        self._busy = False
        self._bulk_progress.setVisible(False)
        self._set_bulk_stop_state(running=False)
        # 先恢复导航，再解析结果和刷新列表；任何后续异常都不能遗留锁定。
        self._set_workspace_enabled(bool(self._candidates))
        stopped = bool(data.get("已停止", False))
        succeeded = int(data.get("成功", 0) or 0)
        skipped = int(data.get("跳过", 0) or 0)
        failed = int(data.get("失败", 0) or 0)
        review_required = int(data.get("需人工核对", 0) or 0)
        unprocessed = int(data.get("未处理", 0) or 0)
        refresh_error = ""
        try:
            succeeded_ids = [str(value) for value in data.get("成功字形", [])]
            for variant_id in succeeded_ids:
                self._drop_candidate_cache(variant_id)
                self.selection_saved.emit(variant_id)
            self._list_thumbnail_cache.clear()
            self._list_thumbnail_generation += 1
            current_key = (
                str(self._current_item.get("键", "")) if self._current_item else ""
            )
            if self._service:
                self._items = self._service.list_items()
                self._current_item = next(
                    (
                        item
                        for item in self._items
                        if str(item.get("键", "")) == current_key
                    ),
                    None,
                )
            self._refresh_list(preferred_key=current_key)
            if (
                self._current_item
                and self._item_phase_status(self._current_item) == STATUS_OPTIMIZED
            ):
                self._show_readonly_current_item()
        except Exception as exc:
            refresh_error = str(exc) or type(exc).__name__
        finally:
            self._set_workspace_enabled(bool(self._candidates))
        summary = (
            f"成功 {succeeded}（已保存），其中结构需人工核对 {review_required}；"
            f"失败 {failed}，未处理 {unprocessed}，跳过 {skipped}。"
        )
        elapsed_text = f"总耗时：{format_elapsed_time(elapsed)}"
        result_state = "已停止" if stopped else "完成"
        self._message_label.setText(f"整库自动优化{result_state}：{summary}")
        self.status_message.emit(f"整库自动优化{result_state}：{summary}")
        if self._glyph_service is not None:
            self.summary_changed.emit(self._glyph_service)
        if refresh_error:
            QMessageBox.critical(
                self,
                "自动优化页面刷新失败",
                f"整库自动优化已经结束，但页面刷新失败：{refresh_error}。\n"
                "返回首页后重新进入即可读取最新结果。\n\n"
                f"总耗时：{format_elapsed_time(elapsed)}",
            )
        elif failed:
            details = data.get("失败详情", [])
            detail_lines = [
                f"{entry.get('字形', '未知字形')}：{entry.get('错误', '未知错误')}"
                for entry in details[:10]
                if isinstance(entry, dict)
            ]
            if len(details) > 10:
                detail_lines.append(f"另有 {len(details) - 10} 项失败。")
            detail_text = "\n".join(detail_lines)
            QMessageBox.warning(
                self,
                f"自动优化{result_state}",
                summary
                + (f"\n\n失败详情：\n{detail_text}" if detail_text else "")
                + f"\n\n{elapsed_text}",
            )
        elif review_required:
            details = data.get("需人工核对详情", [])
            detail_lines = [
                f"{entry.get('字形', '未知字形')}：{entry.get('原因', '结构保护未通过')}"
                for entry in details[:10]
                if isinstance(entry, dict)
            ]
            if len(details) > 10:
                detail_lines.append(f"另有 {len(details) - 10} 项需人工核对。")
            detail_text = "\n".join(detail_lines)
            QMessageBox.warning(
                self,
                f"自动优化{result_state}",
                summary
                + "\n\n这些结果已正常进入待审核阶段，请逐字核对笔画结构。"
                + (f"\n\n核对详情：\n{detail_text}" if detail_text else "")
                + f"\n\n{elapsed_text}",
            )
        else:
            QMessageBox.information(
                self,
                f"自动优化{result_state}",
                f"{summary}\n\n{elapsed_text}",
            )

    def _bulk_failed(
        self,
        worker: _OptimizationBatchWorker,
        message: str,
    ) -> None:
        if worker is not self._bulk_worker:
            return
        elapsed = self._bulk_elapsed_seconds()
        self._bulk_worker = None
        self._bulk_started_at = None
        self._busy = False
        self._bulk_progress.setVisible(False)
        self._set_bulk_stop_state(running=False)
        self._set_workspace_enabled(bool(self._candidates))
        self._message_label.setText("整库自动优化异常中止。")
        if self._glyph_service is not None:
            self.summary_changed.emit(self._glyph_service)
        QMessageBox.critical(
            self,
            "整库自动优化失败",
            f"{message}\n\n总耗时：{format_elapsed_time(elapsed)}",
        )

    def _bulk_elapsed_seconds(self, result: dict[str, Any] | None = None) -> float:
        if result is not None:
            try:
                elapsed = float(result.get("总耗时秒", -1.0))
            except (TypeError, ValueError):
                elapsed = -1.0
            if elapsed >= 0.0:
                return elapsed
        if self._bulk_started_at is None:
            return 0.0
        return max(0.0, time.perf_counter() - self._bulk_started_at)

    def open_library(self, library_path: str) -> None:
        """打开字库目录并选择首个字形。"""
        name = os.path.basename(os.path.normpath(library_path))
        self.open_glyph_service(GlyphService.open(name, library_path))

    def open_glyph_service(self, glyph_service: GlyphService) -> None:
        """使用已有字形服务打开自动优化页面。"""
        self._request_id += 1
        self._glyph_service = glyph_service
        summarize_glyph_service(glyph_service)
        self._service = OptimizationService(glyph_service)
        self._library_identity = os.path.normcase(os.path.abspath(glyph_service.ziku_dir))
        self._reoptimization_key = ""
        self._clear_candidate_cache()
        self._list_thumbnail_cache.clear()
        self._list_thumbnail_inflight.clear()
        self._list_thumbnail_generation += 1
        self._items = self._service.list_items()
        self._workflow_summary = glyph_service.get_coordination_summary()
        self._finished_dir = glyph_service.get_workflow_dirs().get("成品", "")
        self._status_combo.blockSignals(True)
        self._status_combo.setCurrentText(PHASE_FILTER_ALL)
        self._status_combo.blockSignals(False)
        metadata = glyph_service.get_metadata()
        self._library_label.setText(
            f"当前字库：{glyph_service.ziku_name}　{metadata.get('DPI', '--')} DPI · "
            f"{metadata.get('画布宽', '--')}×{metadata.get('画布高', '--')} 像素"
        )
        self._refresh_list(select_first=True)

    def _item_phase_status(self, item: dict[str, Any]) -> str:
        """返回自动优化页面的二态状态，并兼容旧测试桩数据。"""
        source_signature = (
            str(item.get("状态", "")),
            str(item.get("显示状态", "")),
            str(item.get("中间文件", "")),
            str(item.get("审核文件", "")),
            str(item.get("成品文件", "")),
        )
        cached = str(item.get("_自动优化阶段状态", ""))
        if (
            cached in {STAGE_PENDING_OPTIMIZATION, STATUS_OPTIMIZED}
            and item.get("_自动优化阶段状态来源") == source_signature
        ):
            return cached
        if "状态" in item:
            projection = project_stage_status(
                item,
                self._workflow_summary,
                self._finished_dir,
                PHASE_OPTIMIZATION,
            )
            if projection.status:
                item["_自动优化阶段状态"] = projection.status
                item["_自动优化阶段状态来源"] = source_signature
                item["_自动优化阶段提示"] = projection.markers
                return projection.status
        raw = str(item.get("显示状态", "") or item.get("状态", "")).strip()
        status = (
            STAGE_PENDING_OPTIMIZATION
            if raw in {"", "待优化", "待自动优化"}
            else STATUS_OPTIMIZED
        )
        item["_自动优化阶段状态"] = status
        item["_自动优化阶段状态来源"] = source_signature
        return status

    @staticmethod
    def _item_markers(item: dict[str, Any]) -> tuple[str, ...]:
        raw = item.get("_自动优化阶段提示", item.get("提示", ()))
        if isinstance(raw, str):
            values = (raw,)
        elif isinstance(raw, (list, tuple, set)):
            values = tuple(str(value) for value in raw)
        else:
            values = ()
        return tuple(marker for marker in WORKFLOW_MARKERS if marker in values)

    def _refresh_list(
        self,
        _value: str = "",
        select_first: bool = False,
        preferred_key: str | None = None,
    ) -> None:
        selected_key = "" if select_first else (
            preferred_key
            if preferred_key is not None
            else str(self._current_item.get("键", "")) if self._current_item else ""
        )
        query = self._search_edit.text().strip().lower()
        status_filter = self._status_combo.currentText()
        filtered: list[dict[str, Any]] = []
        for item in self._items:
            status = self._item_phase_status(item)
            if status_filter != PHASE_FILTER_ALL and status != status_filter:
                continue
            text = f"{item.get('归属字', '')} {item.get('原始文件名', '')} 字形{item.get('变体序号', '')}".lower()
            if query and query not in text:
                continue
            filtered.append(item)
        ordering = self._sort_combo.currentText()
        if ordering == "低分优先":
            filtered.sort(key=lambda item: (item.get("得分") is None, float(item.get("得分") or 0)))
        elif ordering == "导入顺序":
            filtered.sort(key=lambda item: (item.get("字符顺序", 0), item.get("变体序号", 0)))
        else:
            filtered.sort(
                key=lambda item: (
                    pinyin_natural_key(str(item.get("归属字", ""))),
                    item.get("变体序号", 0),
                )
            )
        groups: dict[str, list[dict[str, Any]]] = {}
        for item in filtered:
            groups.setdefault(str(item.get("归属字", "?")), []).append(item)
        all_groups: dict[str, list[dict[str, Any]]] = {}
        for item in self._items:
            all_groups.setdefault(str(item.get("归属字", "?")), []).append(item)
        self._visible_items = [item for group_items in groups.values() for item in group_items]

        self._item_tree.blockSignals(True)
        self._item_tree.clear()
        self._variant_nodes.clear()
        target_node: QTreeWidgetItem | None = None
        for char, group_items in groups.items():
            all_group_items = all_groups.get(char, group_items)
            completed_count = sum(
                self._item_phase_status(item) == STATUS_OPTIMIZED
                for item in all_group_items
            )
            group_status = f"已优化 {completed_count}/{len(all_group_items)}"
            marked_count = sum(
                bool(self._item_markers(item)) for item in all_group_items
            )
            parent = QTreeWidgetItem(
                [
                    f"{char}（{len(all_group_items)}个字形）",
                    "",
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
                group_status,
                f"提示 {marked_count}" if marked_count else "—",
                PHASE_STATUS_COLORS[
                    STATUS_OPTIMIZED
                    if completed_count == len(all_group_items)
                    else STAGE_PENDING_OPTIMIZATION
                ],
                self.STRUCTURE_RISK_COLOR if marked_count else None,
            )
            self._item_tree.addTopLevelItem(parent)

            for item in group_items:
                status = self._item_phase_status(item)
                markers = self._item_markers(item)
                marker_text = "、".join(markers) or "—"
                score = item.get("得分")
                score_text = f"{float(score):.0f}分" if score is not None else "--"
                item_text = (
                    f"字形{item.get('变体序号', 1)} · "
                    f"{item.get('原始文件名', '')}"
                )
                child = QTreeWidgetItem(
                    parent,
                    [
                        item_text,
                        "",
                        score_text,
                    ],
                )
                key = str(item.get("键", ""))
                child.setData(0, Qt.ItemDataRole.UserRole, key)
                cached_icon = self._cached_glyph_thumbnail(item)
                if cached_icon is not None:
                    child.setIcon(0, cached_icon)
                elif len(filtered) <= self.LIST_THUMBNAIL_SYNC_LIMIT:
                    child.setIcon(0, self._glyph_thumbnail(item))
                else:
                    child.setIcon(0, self._thumbnail_placeholder())
                child.setToolTip(
                    0,
                    f"{char} · 字形{item.get('变体序号', 1)}\n"
                    f"{item.get('原始文件名', '')}\n{score_text} · 自动优化：{status}\n"
                    f"提示：{marker_text}",
                )
                set_two_line_status(
                    child,
                    1,
                    status,
                    marker_text,
                    PHASE_STATUS_COLORS[status],
                    self.STRUCTURE_RISK_COLOR if markers else None,
                )
                if markers:
                    if MARKER_STRUCTURE_REVIEW in markers:
                        child.setForeground(0, QBrush(QColor(self.STRUCTURE_RISK_COLOR)))
                child.setTextAlignment(2, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self._variant_nodes.append(child)
                if key == selected_key:
                    target_node = child
            parent.setExpanded(True)
        largest_group = max(
            (len(group_items) for group_items in all_groups.values()),
            default=0,
        )
        self._item_tree_columns.set_protected_minimum(
            1,
            self._optimization_status_column_width(
                f"已优化 {largest_group}/{largest_group}"
            ),
        )
        self._update_list_row_heights()
        self._item_tree.blockSignals(False)
        self._list_count_label.setText(f"显示 / 总数：{len(filtered)} / {len(self._items)}")
        pending_count = sum(
            self._item_phase_status(item) == STAGE_PENDING_OPTIMIZATION
            for item in self._items
        )
        optimized_count = len(self._items) - pending_count
        self._summary_label.setText(
            f"待优化 {pending_count}　已优化 {optimized_count}"
        )
        if target_node is None and filtered:
            target_node = self._variant_nodes[0]
        if target_node is not None:
            self._item_tree.setCurrentItem(target_node)
        elif not filtered:
            self._clear_current_workspace("当前筛选范围内没有字形")
        if len(filtered) > self.LIST_THUMBNAIL_SYNC_LIMIT:
            self._schedule_visible_list_thumbnails()

    def _execute_search(self, _checked: bool = False) -> None:
        """按回车或按钮执行一次全新的搜索，并定位第一条结果。"""

        self._refresh_list(select_first=True)

    def _restore_search_when_cleared(self, text: str) -> None:
        """删除检索文字后立即恢复当前阶段筛选下的全部字形。"""

        if not text.strip():
            self._refresh_list()

    def _schedule_list_row_height_update(
        self,
        logical_index: int,
        _old_size: int,
        _new_size: int,
    ) -> None:
        """用户调整首列时合并刷新请求，避免拖动期间反复重排整棵树。"""
        if logical_index == 0:
            self._list_row_height_timer.start(40)

    def _update_list_row_heights(self) -> None:
        """按首列实际文字区域计算紧凑行高，不受其他列及面板宽度影响。"""
        header = self._item_tree.header()
        text_area = max(
            72,
            header.sectionSize(0)
            - self._item_tree.indentation() * 2
            - self._item_tree.iconSize().width()
            - 20,
        )
        metrics = self._item_tree.fontMetrics()
        flags = int(
            Qt.TextFlag.TextWordWrap
            | Qt.TextFlag.TextWrapAnywhere
        )
        icon_height = self._item_tree.iconSize().height() + 8
        for child in self._variant_nodes:
            text_height = metrics.boundingRect(
                0,
                0,
                text_area,
                10000,
                flags,
                child.text(0),
            ).height()
            child.setSizeHint(
                0,
                QSize(0, max(46, icon_height, text_height + 6)),
            )

    def _on_item_selected(
        self,
        current: QTreeWidgetItem | None,
        previous: QTreeWidgetItem | None,
    ) -> None:
        if current is None:
            return
        key = str(current.data(0, Qt.ItemDataRole.UserRole) or "")
        if not key:
            self._restore_valid_tree_current(previous)
            return
        item = next((value for value in self._items if str(value.get("键", "")) == key), None)
        if item is not None and item is not self._current_item:
            self._select_item(item)

    def _show_glyph_context_menu(self, position: object) -> None:
        node = self._item_tree.itemAt(position)
        if node is None:
            return
        variant_id = str(node.data(0, Qt.ItemDataRole.UserRole) or "")
        if not variant_id:
            return
        self._item_tree.setCurrentItem(node)
        menu = QMenu(self)
        action = menu.addAction("修正字形名称…")
        action.setEnabled(
            not self._busy and self._bulk_worker is None and not self._workers
        )
        action.triggered.connect(self._rename_current_glyph)
        menu.exec(self._item_tree.viewport().mapToGlobal(position))

    def _rename_current_glyph(self) -> None:
        if not self._glyph_service or not self._service or not self._current_item:
            QMessageBox.information(self, "修正字形名称", "请先选择一个具体字形。")
            return
        if self._busy or self._bulk_worker is not None or self._workers:
            QMessageBox.information(
                self,
                "暂时不能修改名称",
                "当前正在生成、保存或批量处理字形，请等待任务结束后重试。",
            )
            return
        if self._branch_dirty:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle("当前候选尚未保存")
            box.setText("当前字形的候选选择或探索分支尚未保存。")
            box.setInformativeText(
                "继续修正名称会放弃这些未保存候选；需要保留时请先采用当前候选。"
            )
            discard_button = box.addButton(
                "放弃候选并继续",
                QMessageBox.ButtonRole.DestructiveRole,
            )
            box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
            box.exec()
            if box.clickedButton() is not discard_button:
                return
            self._branch_dirty = False

        variant_id = str(self._current_item.get("键", ""))
        result = run_glyph_rename_dialog(
            self,
            self._glyph_service,
            variant_id,
        )
        if result is None:
            return
        self._request_id += 1
        self._release_original_image()
        self._clear_candidate_cache()
        self._list_thumbnail_cache.clear()
        self._list_thumbnail_inflight.clear()
        self._list_thumbnail_generation += 1
        self._candidates = []
        self._current_item = None
        self._reoptimization_key = ""
        self._items = self._service.list_items()
        self._workflow_summary = self._glyph_service.get_coordination_summary()
        self._refresh_list(preferred_key=variant_id)
        self.summary_changed.emit(self._glyph_service)
        self.status_message.emit(f"字形名称已修正为 {result.get('新文件名', '')}")
        QMessageBox.information(
            self,
            "名称修改完成",
            f"字形已修正为“{result.get('新归属字', '')}”，各阶段文件名已同步更新。",
        )

    def _restore_valid_tree_current(self, previous: QTreeWidgetItem | None) -> None:
        """父分组只负责展开折叠，不能取代当前有效字形。"""
        target = previous if previous in self._variant_nodes else None
        if target is None and self._current_item:
            current_key = str(self._current_item.get("键", ""))
            target = next(
                (
                    node
                    for node in self._variant_nodes
                    if str(node.data(0, Qt.ItemDataRole.UserRole) or "") == current_key
                ),
                None,
            )
        if target is None and self._variant_nodes:
            target = self._variant_nodes[0]
        if target is None:
            self._item_tree.setCurrentItem(None)
            return
        self._item_tree.blockSignals(True)
        try:
            self._item_tree.setCurrentItem(target)
        finally:
            self._item_tree.blockSignals(False)

    def _select_item(self, item: dict[str, Any]) -> None:
        selected_key = str(item.get("键", ""))
        if selected_key != self._reoptimization_key:
            self._reoptimization_key = ""
        self._current_item = item
        self._round_number = 1
        self._history = ["第 1 轮　基础候选"]
        self._render_original_preview()
        if not self._current_item_is_optimizable():
            self._show_readonly_current_item()
            return
        self._load_candidates()

    def _load_candidates(self, force: bool = False) -> None:
        if not self._current_item or not self._service:
            return
        if not self._current_item_is_optimizable():
            self._show_readonly_current_item()
            return
        item_key = str(self._current_item.get("键", ""))
        cache_key = self._candidate_cache_key(self._current_item)
        cached_candidates = None if force else self._cached_candidates(cache_key)
        if cached_candidates is not None:
            self._round_number = 1
            self._history = ["第 1 轮　基础候选"]
            self._branch_dirty = False
            self._show_candidates(cached_candidates)
            if any(
                OptimizationService.requires_structure_review(candidate)
                for candidate in cached_candidates
            ):
                self._message_label.setText(
                    "已载入缓存候选；结构风险结果需在手工审核中重点核对。"
                )
            else:
                self._message_label.setText(
                    "已载入缓存候选；所有处理始终基于原始文件执行。"
                )
            return
        item = dict(self._current_item)
        service = self._service
        engine_context = self._engine_context
        self._clear_candidates("正在后台生成候选效果，可继续切换其他字形。")
        self._candidate_title.setText("正在生成候选效果……")
        self._start_task(
            lambda: service.generate_candidates(item, engine_context=engine_context),
            lambda result: self._candidates_ready(cache_key, item_key, result),
            lambda message: self._task_failed(
                "自动优化失败",
                message
                if engine_context.engine_id == NO_MODEL_ENGINE_ID
                else message + "\n\n可切换为“无学习模型”后重新生成候选。",
            ),
            lock_page=False,
        )

    def _candidates_ready(self, cache_key: object, item_key: str, result: object) -> None:
        candidates = list(result) if isinstance(result, list) else []
        if not candidates:
            self._task_failed(
                "自动优化失败",
                "算法未生成有效候选结果。",
            )
            return
        self._store_candidate_cache(cache_key, candidates)
        if (
            self._current_item
            and str(self._current_item.get("键", "")) == item_key
            and self._candidate_cache_key(self._current_item) == cache_key
        ):
            self._round_number = 1
            self._history = ["第 1 轮　基础候选"]
            self._branch_dirty = False
            self._show_candidates(candidates)
            if any(
                OptimizationService.requires_structure_review(candidate)
                for candidate in candidates
            ):
                self._message_label.setText(
                    "没有候选通过全部结构保护，已保留实际优化结果供人工核对。"
                )
            else:
                self._message_label.setText(
                    "请选择效果最满意的候选；所有处理始终基于原始文件执行。"
                )

    def _show_candidates(self, candidates: list[dict[str, Any]]) -> None:
        self._candidates = candidates[:8]
        self._selected_index = -1
        self._candidate_list.blockSignals(True)
        self._candidate_list.clear()
        for index, candidate in enumerate(self._candidates):
            score = float(candidate.get("得分", 0.0))
            candidate_type = str(candidate.get("处理类型", CANDIDATE_TYPE_OPTIMIZED))
            name = self._candidate_caption(candidate, index)
            card = QListWidgetItem(f"候选{index + 1}　{score:.1f}分\n{name}")
            color = (
                self.STRUCTURE_RISK_COLOR
                if OptimizationService.requires_structure_review(candidate)
                else self.CANDIDATE_COLORS.get(candidate_type, "#ffffff")
            )
            card.setForeground(QBrush(QColor(color)))
            card.setTextAlignment(
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop
            )
            card.setToolTip(self._candidate_tooltip(candidate))
            image = candidate.get("图像")
            if isinstance(image, Image.Image):
                card.setIcon(QIcon(QPixmap.fromImage(self._pil_to_qimage(self._compose_preview(
                    image,
                    self.CARD_SIZE,
                    self.CANDIDATE_TRANSPARENT_BACKGROUND,
                )))))
            self._candidate_list.addItem(card)
        self._candidate_list.blockSignals(False)
        self._update_candidate_grid()
        self._schedule_candidate_layout()
        self._candidate_title.setText(f"第 {self._round_number} 轮候选 · 共 {len(self._candidates)} 张")
        self._round_label.setText(f"第 {self._round_number}/{self.MAX_ROUNDS} 轮")
        self._round_progress.setValue(self._round_number)
        self._sync_history_view()
        self._set_workspace_enabled(bool(self._candidates))
        if self._candidates:
            self._candidate_list.setCurrentRow(0)

    def _select_candidate(self, index: int) -> None:
        if not 0 <= index < len(self._candidates):
            return
        previous_index = self._selected_index
        self._selected_index = index
        if previous_index >= 0 and previous_index != index:
            self._branch_dirty = True
        candidate = self._candidates[index]
        self._scheme_label.setPlainText(self._format_candidate_summary(candidate, index))
        self._save_button.setText(
            "确认风险并保存"
            if OptimizationService.requires_structure_review(candidate)
            else "采用并保存"
        )
        self._render_selected_preview()
        self._set_workspace_enabled(True)

    def _explore_selected(self) -> None:
        if (
            not self._service
            or not self._current_item
            or not self._current_item_is_optimizable()
            or self._selected_index < 0
        ):
            QMessageBox.information(self, "继续探索", "请先选择一个候选效果。")
            return
        candidate_type = str(
            self._candidates[self._selected_index].get("处理类型", CANDIDATE_TYPE_OPTIMIZED)
        )
        if candidate_type != CANDIDATE_TYPE_OPTIMIZED:
            QMessageBox.information(self, "继续探索", "当前候选不执行寻优，请直接采用或选择寻优候选。")
            return
        if self._round_number >= self.MAX_ROUNDS:
            QMessageBox.information(self, "探索已完成", "当前分支已达到5轮上限。")
            return
        item = dict(self._current_item)
        base = self._candidates[self._selected_index]
        base_number = self._selected_index + 1
        service = self._service
        engine_context = self._engine_context
        key = str(item.get("键", ""))
        self._message_label.setText("正在围绕选中方案生成下一轮候选……")
        self._branch_dirty = True
        self._start_task(
            lambda: service.explore(item, base, count=8, engine_context=engine_context),
            lambda result: self._explore_ready(key, base_number, result),
            lambda message: self._task_failed(
                "探索失败",
                message,
            ),
            lock_page=True,
        )

    def _explore_ready(self, key: str, base_number: int, result: object) -> None:
        if not self._current_item or str(self._current_item.get("键", "")) != key:
            return
        candidates = list(result) if isinstance(result, list) else []
        if not candidates:
            QMessageBox.information(self, "探索已完成", "当前方案附近没有发现新的有效结果。")
            self._message_label.setText("当前方案附近已基本探索完毕。")
            return
        self._round_number += 1
        self._history.append(f"第 {self._round_number} 轮　基于候选{base_number}")
        self._show_candidates(candidates)
        self._message_label.setText("新一轮候选已生成；重复方案和重复图片已自动排除。")

    def _restart_candidates(self) -> None:
        if not self._current_item or self._busy:
            return
        current_key = str(self._current_item.get("键", ""))
        if (
            self._item_phase_status(self._current_item) == STATUS_OPTIMIZED
            and current_key != self._reoptimization_key
        ):
            if not self._confirm_reoptimization():
                return
            self._reoptimization_key = current_key
            self._branch_dirty = False
            self._remove_candidate_cache(self._candidate_cache_key(self._current_item))
            self._load_candidates(force=True)
            return
        if not self._current_item_is_optimizable():
            return
        answer = QMessageBox.question(
            self,
            "更换基础处理路线",
            "将结束当前探索分支，并从原始图片重新生成基础候选。是否继续？",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._branch_dirty = False
        self._remove_candidate_cache(self._candidate_cache_key(self._current_item))
        self._load_candidates(force=True)

    def _confirm_reoptimization(self) -> bool:
        """已优化字形允许重跑，但必须说明保存会撤销哪些后续结果。"""
        if not self._current_item:
            return False
        has_downstream_result = bool(
            str(self._current_item.get("审核文件", "")).strip()
            or str(self._current_item.get("成品文件", "")).strip()
            or str(self._current_item.get("状态", ""))
            in {config.STATUS_REVIEWED, config.STATUS_FINISHED}
        )
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("重新优化此字形")
        box.setText("将从原始图片重新生成一组候选，现有文件暂时不会改变。")
        detail = "只有采用并保存新候选后，现有自动优化稿才会被覆盖。"
        if has_downstream_result:
            detail += (
                "\n\n该字形已经进入手工审核或整体协调。采用并保存新候选后，"
                "现有人工审核稿和成品将被撤销，需要重新完成后续阶段。"
            )
        box.setInformativeText(detail)
        confirm_button = box.addButton(
            "开始重新优化",
            QMessageBox.ButtonRole.AcceptRole,
        )
        cancel_button = box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(cancel_button)
        box.setEscapeButton(cancel_button)
        box.exec()
        return box.clickedButton() is confirm_button

    def _save_selected(self) -> None:
        if (
            not self._service
            or not self._current_item
            or not self._current_item_is_optimizable()
            or self._selected_index < 0
        ):
            QMessageBox.information(self, "保存结果", "请先选择一个候选效果。")
            return
        item = dict(self._current_item)
        candidate = self._candidates[self._selected_index]
        if not self._confirm_structure_risk(candidate):
            return
        round_number = self._round_number
        service = self._service
        key = str(item.get("键", ""))
        next_pending_key = self._next_pending_key(key)
        self._message_label.setText("正在保存自动优化稿……")
        self._start_task(
            lambda: service.save_selection(item, candidate, round_number),
            lambda _result: self._save_ready(key, next_pending_key),
            lambda message: self._task_failed("保存失败", message),
            lock_page=True,
        )

    def _next_pending_key(self, key: str) -> str | None:
        """保存前捕获当前排序中的后继，避免筛选刷新后错跳一项。"""
        current_index = next(
            (
                index
                for index, item in enumerate(self._visible_items)
                if str(item.get("键", "")) == key
            ),
            -1,
        )
        if current_index < 0:
            return None
        return next(
            (
                str(item.get("键", ""))
                for item in self._visible_items[current_index + 1 :]
                if self._item_phase_status(item) == STAGE_PENDING_OPTIMIZATION
                and item.get("键")
            ),
            None,
        )

    def _save_ready(self, key: str, next_pending_key: str | None = None) -> None:
        if not self._service:
            return
        self._items = self._service.list_items()
        self._current_item = next((item for item in self._items if str(item.get("键", "")) == key), self._current_item)
        self._message_label.setText("已保存为“自动优化稿”，该字形已提交手工审核。")
        self._branch_dirty = False
        self._reoptimization_key = ""
        self.selection_saved.emit(key)
        if self._glyph_service is not None:
            self.summary_changed.emit(self._glyph_service)
        self.status_message.emit("自动优化稿保存成功")
        self._refresh_list(preferred_key=next_pending_key or key)
        if (
            self._current_item
            and str(self._current_item.get("键", "")) == key
            and self._item_phase_status(self._current_item) == STATUS_OPTIMIZED
        ):
            self._show_readonly_current_item()
            self._message_label.setText(
                "已保存为“自动优化稿”，该字形已提交手工审核。"
            )
        if next_pending_key is None:
            self._show_optimization_end_notice()

    def _show_optimization_end_notice(self) -> None:
        """说明当前搜索和筛选范围已经没有后继待优化字形。"""

        pending_count = sum(
            self._item_phase_status(item) == STAGE_PENDING_OPTIMIZATION
            for item in self._visible_items
        )
        if pending_count:
            detail = (
                "当前字形已保存，已到当前搜索和筛选范围的最后一条。\n"
                f"该范围内仍有 {pending_count} 个待优化字形，可从左侧列表重新选择。"
            )
        else:
            detail = (
                "当前字形已保存，已到当前搜索和筛选范围的最后一条。\n"
                "该范围内的待优化字形已全部处理完成。"
            )
        QMessageBox.information(self, "自动优化", detail)

    def _task_failed(self, title: str, message: str) -> None:
        summary = message.strip() or "任务执行失败。"
        detail = (
            summary
            + "\n\n字形记录和原图均已保留，本次失败不会改变处理状态。"
            "请稍后重试；如反复失败，请人工核对原图。"
        )
        QMessageBox.critical(self, title, detail)
        self._message_label.setText(summary.splitlines()[0])

    def _start_task(
        self,
        function: Callable[[], Any],
        success: Callable[[object], None],
        failure: Callable[[str], None],
        lock_page: bool,
    ) -> None:
        self._request_id += 1
        request_id = self._request_id
        worker = FunctionWorker(function)
        self._workers.add(worker)
        if lock_page:
            self._busy = True
            self._set_workspace_enabled(False)
        else:
            self._set_workspace_enabled(bool(self._candidates))

        def finished(result: object) -> None:
            self._release_worker(worker, lock_page)
            if request_id == self._request_id:
                success(result)

        def failed(message: str) -> None:
            self._release_worker(worker, lock_page)
            if request_id == self._request_id:
                failure(message)

        worker.signals.finished.connect(finished)
        worker.signals.failed.connect(failed)
        self._thread_pool.start(worker)

    def _release_worker(self, worker: FunctionWorker, locked: bool) -> None:
        self._workers.discard(worker)
        if locked:
            self._busy = False
        self._set_workspace_enabled(bool(self._candidates))

    def _move_current(self, step: int, pending_only: bool = False, quiet: bool = False) -> None:
        row = self._current_variant_index()
        target = row + step
        while 0 <= target < len(self._variant_nodes):
            if not pending_only:
                node = self._variant_nodes[target]
                if node.parent() is not None:
                    node.parent().setExpanded(True)
                self._item_tree.setCurrentItem(node)
                return
            item = self._visible_items[target]
            if item and self._item_phase_status(item) == STAGE_PENDING_OPTIMIZATION:
                node = self._variant_nodes[target]
                if node.parent() is not None:
                    node.parent().setExpanded(True)
                self._item_tree.setCurrentItem(node)
                return
            target += step
        if pending_only and not quiet:
            QMessageBox.information(self, "自动优化", "当前筛选范围内没有更多待优化字形。")

    def _render_original_preview(self) -> None:
        if self._current_item is None:
            self._release_original_image()
            self._original_preview.setPixmap(QPixmap())
            self._original_preview.setText("暂无图片")
            return
        try:
            image = self._load_original_image()
            if image is None:
                raise FileNotFoundError("未指定原始图片")
            self._set_preview(self._original_preview, image, transparent=False)
        except Exception:
            self._original_preview.setPixmap(QPixmap())
            self._original_preview.setText("图片无法读取")

    def _render_selected_preview(self) -> None:
        if 0 <= self._selected_index < len(self._candidates):
            image = self._candidates[self._selected_index].get("图像")
            if isinstance(image, Image.Image):
                self._set_preview(self._selected_preview, image, self._preview_background == "透明底")

    def _show_original_as_result(self) -> None:
        try:
            image = self._load_original_image()
            if image is not None:
                self._set_preview(self._selected_preview, image, False)
        except Exception:
            pass

    def _load_original_image(self) -> Image.Image | None:
        """缓存当前原图，预览框重排时只重新缩放，不重复读取文件。"""
        path = str(self._current_item.get("原始路径", "")) if self._current_item else ""
        if not path:
            return None
        if path == self._original_image_path and self._original_image is not None:
            return self._original_image
        with Image.open(path) as source:
            image = source.convert("RGBA")
        self._release_original_image()
        self._original_image_path = path
        self._original_image = image
        return image

    @staticmethod
    def _thumbnail_source(item: dict[str, Any]) -> tuple[str, bool]:
        preview_path = str(item.get("优化预览路径", "") or "")
        source_path = str(item.get("原始路径", "") or "")
        path = preview_path if os.path.isfile(preview_path) else source_path
        return path, path == source_path

    @classmethod
    def _thumbnail_cache_key(cls, item: dict[str, Any]) -> ThumbnailSignature:
        path, normalize_source = cls._thumbnail_source(item)
        try:
            stat = os.stat(path)
        except OSError:
            modified_ns = 0
            size = 0
        else:
            modified_ns = stat.st_mtime_ns
            size = stat.st_size
        normalized_path = os.path.normcase(os.path.abspath(path)) if path else ""
        return normalized_path, modified_ns, size, normalize_source

    def _cached_glyph_thumbnail(self, item: dict[str, Any]) -> QIcon | None:
        variant_id = str(item.get("键", ""))
        cached = self._list_thumbnail_cache.get(variant_id)
        if cached is None:
            return None
        signature = self._thumbnail_cache_key(item)
        if cached[0] != signature:
            self._list_thumbnail_cache.pop(variant_id, None)
            return None
        self._list_thumbnail_cache.move_to_end(variant_id)
        return cached[1]

    def _store_glyph_thumbnail(
        self,
        variant_id: str,
        signature: ThumbnailSignature,
        icon: QIcon,
    ) -> None:
        if not variant_id:
            return
        self._list_thumbnail_cache[variant_id] = (signature, icon)
        self._list_thumbnail_cache.move_to_end(variant_id)
        while len(self._list_thumbnail_cache) > self.LIST_THUMBNAIL_CACHE_ITEMS:
            self._list_thumbnail_cache.popitem(last=False)

    def _thumbnail_placeholder(self) -> QIcon:
        if self._list_thumbnail_placeholder is None:
            icon_size = self._item_tree.iconSize()
            image = QImage(
                max(1, icon_size.width()),
                max(1, icon_size.height()),
                QImage.Format.Format_RGB888,
            )
            image.fill(QColor("white"))
            self._list_thumbnail_placeholder = QIcon(QPixmap.fromImage(image))
        return self._list_thumbnail_placeholder

    def _glyph_thumbnail(self, item: dict[str, Any]) -> QIcon:
        """优先显示自动优化稿，否则显示经过极性校正的原图缩略图。"""
        variant_id = str(item.get("键", ""))
        cached = self._cached_glyph_thumbnail(item)
        if cached is not None:
            return cached

        signature = self._thumbnail_cache_key(item)
        icon_size = self._item_tree.iconSize()
        canvas_size = (max(1, icon_size.width()), max(1, icon_size.height()))
        image = self._decode_glyph_thumbnail(item, canvas_size)
        icon = QIcon(QPixmap.fromImage(image))
        self._store_glyph_thumbnail(variant_id, signature, icon)
        return icon

    @classmethod
    def _decode_glyph_thumbnail(
        cls,
        item: dict[str, Any],
        canvas_size: tuple[int, int],
    ) -> QImage:
        """在线程中解码单个缩略图，返回可跨线程传递的 QImage。"""
        path, normalize_source = cls._thumbnail_source(item)
        canvas = Image.new("RGB", canvas_size, "white")
        try:
            if normalize_source:
                source_rgba, gray, transparency_source = OptimizationService._load_source(path)
                try:
                    normalized_gray, _inverted = OptimizationService._normalize_source_polarity(
                        source_rgba,
                        gray,
                        transparency_source,
                    )
                    source = Image.fromarray(
                        normalized_gray.clip(0, 255).astype("uint8"),
                        "L",
                    ).convert("RGB")
                finally:
                    source_rgba.close()
            else:
                with Image.open(path) as opened:
                    rgba = opened.convert("RGBA")
                white = Image.new("RGBA", rgba.size, "white")
                white.alpha_composite(rgba)
                rgba.close()
                source = white.convert("RGB")
                white.close()
            try:
                source.thumbnail(canvas_size, Image.Resampling.LANCZOS)
                left = (canvas_size[0] - source.width) // 2
                top = (canvas_size[1] - source.height) // 2
                canvas.paste(source, (left, top))
            finally:
                source.close()
        except (OSError, ValueError, SyntaxError):
            pass
        image = cls._pil_to_qimage(canvas)
        canvas.close()
        return image

    def _schedule_visible_list_thumbnails(self) -> None:
        if self._list_thumbnail_refresh_pending:
            return
        self._list_thumbnail_refresh_pending = True
        QTimer.singleShot(0, self._load_visible_list_thumbnails)

    def _load_visible_list_thumbnails(self) -> None:
        """只为当前可见的字形行提交后台缩略图解码。"""
        self._list_thumbnail_refresh_pending = False
        if len(self._visible_items) <= self.LIST_THUMBNAIL_SYNC_LIMIT:
            return
        viewport_rect = self._item_tree.viewport().rect()
        jobs: list[tuple[str, ThumbnailSignature, dict[str, Any]]] = []
        queued_keys: set[tuple[str, ThumbnailSignature]] = set()
        items_by_key = {
            str(item.get("键", "")): item
            for item in self._visible_items
            if item.get("键")
        }
        for node in self._variant_nodes:
            if len(jobs) >= self.LIST_THUMBNAIL_BATCH_SIZE:
                break
            rect = self._item_tree.visualItemRect(node)
            if rect.isEmpty() or not rect.intersects(viewport_rect):
                continue
            item = items_by_key.get(
                str(node.data(0, Qt.ItemDataRole.UserRole) or "")
            )
            if item is None:
                continue
            variant_id = str(item.get("键", ""))
            signature = self._thumbnail_cache_key(item)
            cached = self._cached_glyph_thumbnail(item)
            if cached is not None:
                node.setIcon(0, cached)
                continue
            request_key = (variant_id, signature)
            if request_key in self._list_thumbnail_inflight or request_key in queued_keys:
                continue
            jobs.append((variant_id, signature, dict(item)))
            queued_keys.add(request_key)
        if jobs:
            self._start_list_thumbnail_batch(jobs)

    def _start_list_thumbnail_batch(
        self,
        jobs: list[tuple[str, ThumbnailSignature, dict[str, Any]]],
    ) -> None:
        generation = self._list_thumbnail_generation
        canvas_size = (
            max(1, self._item_tree.iconSize().width()),
            max(1, self._item_tree.iconSize().height()),
        )
        requested_keys = {
            (variant_id, signature)
            for variant_id, signature, _item in jobs
        }
        self._list_thumbnail_inflight.update(requested_keys)
        worker = FunctionWorker(
            lambda: [
                (
                    variant_id,
                    signature,
                    self._decode_glyph_thumbnail(item, canvas_size),
                )
                for variant_id, signature, item in jobs
            ]
        )
        self._list_thumbnail_workers.add(worker)

        def release() -> None:
            self._list_thumbnail_workers.discard(worker)
            self._list_thumbnail_inflight.difference_update(requested_keys)

        def finished(result: object) -> None:
            release()
            if generation != self._list_thumbnail_generation:
                return
            loaded = result if isinstance(result, list) else []
            current_items = {
                str(item.get("键", "")): item
                for item in self._items
                if item.get("键")
            }
            for variant_id, signature, image in loaded:
                if not isinstance(image, QImage):
                    continue
                current_item = current_items.get(variant_id)
                if (
                    current_item is None
                    or self._thumbnail_cache_key(current_item) != signature
                ):
                    continue
                self._store_glyph_thumbnail(
                    variant_id,
                    signature,
                    QIcon(QPixmap.fromImage(image)),
                )
            self._apply_cached_list_thumbnails()
            self._schedule_visible_list_thumbnails()

        def failed(_message: str) -> None:
            release()

        worker.signals.finished.connect(finished)
        worker.signals.failed.connect(failed)
        self._thread_pool.start(worker)

    def _apply_cached_list_thumbnails(self) -> None:
        items_by_key = {
            str(item.get("键", "")): item
            for item in self._visible_items
            if item.get("键")
        }
        for node in self._variant_nodes:
            item = items_by_key.get(
                str(node.data(0, Qt.ItemDataRole.UserRole) or "")
            )
            if item is None:
                continue
            cached = self._cached_glyph_thumbnail(item)
            if cached is not None:
                node.setIcon(0, cached)

    def eventFilter(self, watched, event: QEvent) -> bool:
        """视图尺寸稳定后刷新预览和候选排布。"""
        previews = (
            getattr(self, "_original_preview", None),
            getattr(self, "_selected_preview", None),
        )
        if watched in previews and event.type() == QEvent.Type.Resize:
            self._schedule_preview_refresh()
        candidate_list = getattr(self, "_candidate_list", None)
        if (
            candidate_list is not None
            and watched is candidate_list.viewport()
            and event.type() == QEvent.Type.Resize
        ):
            self._schedule_candidate_layout()
        item_tree = getattr(self, "_item_tree", None)
        if (
            item_tree is not None
            and watched is item_tree.viewport()
            and event.type() in (QEvent.Type.Resize, QEvent.Type.Show)
        ):
            self._schedule_visible_list_thumbnails()
        return super().eventFilter(watched, event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._schedule_preview_refresh()
        self._schedule_candidate_layout()

    def _schedule_preview_refresh(self) -> None:
        if self._preview_refresh_pending or not self.isVisible():
            return
        self._preview_refresh_pending = True
        QTimer.singleShot(0, self._refresh_previews_after_layout)

    def _refresh_previews_after_layout(self) -> None:
        self._preview_refresh_pending = False
        if not self.isVisible():
            return
        self._render_original_preview()
        self._render_selected_preview()

    def _schedule_candidate_layout(self) -> None:
        if self._candidate_layout_pending:
            return
        self._candidate_layout_pending = True
        QTimer.singleShot(0, self._refresh_candidate_layout_after_resize)

    def _refresh_candidate_layout_after_resize(self) -> None:
        self._candidate_layout_pending = False
        self._update_candidate_grid()

    def _update_candidate_grid(self) -> None:
        """按视口宽度固定列槽，候选不足一行时保留后续空槽。"""
        viewport = self._candidate_list.viewport()
        viewport_width = max(1, viewport.width())
        scroll_bar = self._candidate_list.verticalScrollBar()
        if not scroll_bar.isVisible():
            viewport_width = max(1, viewport_width - scroll_bar.sizeHint().width())
        columns = min(
            self.CANDIDATE_MAX_COLUMNS,
            max(1, viewport_width // self.CANDIDATE_MIN_CELL_WIDTH),
        )
        grid_size = QSize(
            max(1, (viewport_width - 1) // columns),
            self.CANDIDATE_CELL_HEIGHT,
        )
        self._candidate_columns = columns
        if self._candidate_list.gridSize() != grid_size:
            self._candidate_list.setGridSize(grid_size)
            current = self._candidate_list.currentItem()
            if current is not None:
                self._candidate_list.scrollToItem(
                    current,
                    QAbstractItemView.ScrollHint.EnsureVisible,
                )

    def _set_fit_mode(self, fit: bool) -> None:
        self._fit_preview = fit
        self._fit_button.setChecked(fit)
        self._actual_size_button.setChecked(not fit)
        self._render_original_preview()
        self._render_selected_preview()

    def _set_background(self, background: str) -> None:
        self._preview_background = background
        self._white_background_button.setChecked(background == "白底")
        self._transparent_background_button.setChecked(background == "透明底")
        self._render_selected_preview()

    def _set_preview(self, label: QLabel, image: Image.Image, transparent: bool) -> None:
        width = max(1, label.width() - 8)
        height = max(1, label.height() - 8)
        if self._fit_preview:
            output = self._compose_preview(image, (width, height), transparent)
        else:
            output = self._compose_preview(image, (width, height), transparent, scale=False)
        label.setText("")
        label.setPixmap(QPixmap.fromImage(self._pil_to_qimage(output)))

    @staticmethod
    def _compose_preview(
        image: Image.Image,
        size: tuple[int, int],
        transparent: bool,
        scale: bool = True,
    ) -> Image.Image:
        source = image.convert("RGBA")
        if scale:
            source.thumbnail(size, Image.Resampling.LANCZOS)
        if transparent:
            background = Image.new("RGBA", size, "#F0F0F0")
            draw = ImageDraw.Draw(background)
            block = 12
            for y in range(0, size[1], block):
                for x in range(0, size[0], block):
                    if (x // block + y // block) % 2:
                        draw.rectangle((x, y, x + block - 1, y + block - 1), fill="#D6DADF")
        else:
            background = Image.new("RGBA", size, "white")
        x = (size[0] - source.width) // 2
        y = (size[1] - source.height) // 2
        background.alpha_composite(source, (x, y))
        return background.convert("RGB")

    @staticmethod
    def _pil_to_qimage(image: Image.Image) -> QImage:
        rgb = image.convert("RGB")
        data = rgb.tobytes("raw", "RGB")
        return QImage(data, rgb.width, rgb.height, rgb.width * 3, QImage.Format.Format_RGB888).copy()

    @staticmethod
    def _format_scheme(scheme: dict[str, Any]) -> str:
        lines: list[str] = []
        auto_correction = scheme.get("自动校正", {})
        if isinstance(auto_correction, dict) and auto_correction.get("反相"):
            lines.append("自动校正：检测到深色背景，已反相")
        preprocess = scheme.get("预处理", {})
        labels = {
            "转灰度": "转灰度",
            "反相": "反相",
            "墨色归一": "算法工作归一",
        }
        enabled = [labels[name] for name in labels if preprocess.get(name)]
        lines.append(f"预处理：{'、'.join(enabled) if enabled else '无'}")
        for level in ("L1", "L2", "L3", "L4", "L5"):
            config = scheme.get(level, {})
            method = config.get("算法") or config.get("方法")
            if method and method != "不处理":
                parameter_values = config.get("参数", {})
                if not isinstance(parameter_values, dict):
                    parameter_values = {}
                params = "、".join(f"{key}={value}" for key, value in parameter_values.items())
                lines.append(f"{level}：{method}" + (f"（{params}）" if params else ""))
        return "\n".join(lines) if lines else "基础处理方案"

    @staticmethod
    def _candidate_caption(candidate: dict[str, Any], index: int = 0) -> str:
        candidate_type = str(candidate.get("处理类型", CANDIDATE_TYPE_OPTIMIZED))
        if candidate_type == CANDIDATE_TYPE_DIRECT:
            return "原图已有透明区，直接采用"
        if candidate_type == CANDIDATE_TYPE_ALPHA_DENOISED:
            return "透明层轻度去杂"
        if candidate_type == CANDIDATE_TYPE_TRANSPARENT:
            return "仅做背景透明处理"
        caption = str(candidate.get("方案名", f"候选{index + 1}"))
        if OptimizationService.requires_structure_review(candidate):
            caption += "（结构需核对）"
        return caption

    @staticmethod
    def _candidate_tooltip(candidate: dict[str, Any]) -> str:
        candidate_type = str(candidate.get("处理类型", CANDIDATE_TYPE_OPTIMIZED))
        if candidate_type == CANDIDATE_TYPE_DIRECT:
            source = str(candidate.get("方案", {}).get("透明来源", ""))
            if source == TRANSPARENCY_SOURCE_PHOTOSHOP_ALPHA:
                return "已解码并校验 TIFF Photoshop 图层 Alpha，采用时不执行寻优。"
            return "原图已包含标准 Alpha 透明区，采用时保留原图像素，不执行寻优。"
        if candidate_type == CANDIDATE_TYPE_ALPHA_DENOISED:
            return "只清理远离可靠字形核心的透明层残留；RGB 和可靠核心保持不变。"
        if candidate_type == CANDIDATE_TYPE_TRANSPARENT:
            return "只将图片背景转换为透明，不执行去杂或其他寻优处理。"
        tooltip = f"寻优方案：{candidate.get('方案名', '自动优化')}"
        if OptimizationService.requires_structure_review(candidate):
            review = OptimizationService.structure_review_metadata(candidate)
            tooltip += (
                f"\n结构需人工核对：{review.get('原因', '结构保护未通过')}"
                "\n综合得分只用于候选排序，不代表结构安全。"
            )
        return tooltip

    @classmethod
    def _format_candidate(cls, candidate: dict[str, Any]) -> str:
        candidate_type = str(candidate.get("处理类型", CANDIDATE_TYPE_OPTIMIZED))
        if candidate_type == CANDIDATE_TYPE_DIRECT:
            source = str(candidate.get("方案", {}).get("透明来源", ""))
            if source == TRANSPARENCY_SOURCE_PHOTOSHOP_ALPHA:
                return "处理类型：原图直接采用\n已解码并校验 TIFF Photoshop 图层 Alpha，不执行寻优。"
            return "处理类型：原图直接采用\n保留原图标准 Alpha 透明像素，不执行寻优。"
        if candidate_type == CANDIDATE_TYPE_ALPHA_DENOISED:
            return (
                "处理类型：透明层轻度去杂\n"
                "按笔画尺度清理远离主体的低 Alpha 残留和成片高 Alpha 微小噪点。"
            )
        if candidate_type == CANDIDATE_TYPE_TRANSPARENT:
            return "处理类型：仅背景透明\n只将背景转换为透明，不执行去杂或其他寻优优化。"
        return f"处理类型：寻优优化\n{cls._format_scheme(candidate.get('方案', {}))}"

    def _format_candidate_summary(self, candidate: dict[str, Any], index: int) -> str:
        scheme_value = candidate.get("方案", {})
        scheme = scheme_value if isinstance(scheme_value, dict) else {}
        engine_value = scheme.get("处理引擎", {})
        engine = engine_value if isinstance(engine_value, dict) else {}
        descriptor = self._engine_context.descriptor
        engine_name = str(engine.get("名称") or descriptor.display_name)
        engine_version = str(engine.get("版本") or descriptor.version).strip()
        engine_text = (
            f"{engine_name} · {engine_version}"
            if engine_version
            else engine_name
        )
        candidate_type = str(candidate.get("处理类型", CANDIDATE_TYPE_OPTIMIZED))
        route_source = str(scheme.get("路线来源", "")).strip()
        if not route_source:
            engine_id = str(engine.get("标识", descriptor.engine_id))
            if candidate_type == CANDIDATE_TYPE_DIRECT:
                route_source = "原图直接采用"
            elif candidate_type == CANDIDATE_TYPE_ALPHA_DENOISED:
                route_source = "透明层轻度去杂"
            elif engine_id == NO_MODEL_ENGINE_ID:
                route_source = "传统图像管线"
            else:
                route_source = "学习模型前景 + 传统后处理"
        lines = [
            f"候选名称：候选{index + 1}　{self._candidate_caption(candidate, index)}",
            f"综合得分：{float(candidate.get('得分', 0.0)):.1f}",
            f"处理引擎：{engine_text}",
            f"基础路线：{route_source}",
        ]
        if OptimizationService.requires_structure_review(candidate):
            review = OptimizationService.structure_review_metadata(candidate)
            lines.extend((
                "结构复核：需人工核对",
                f"复核阶段：{review.get('阶段', '原尺寸复核')}",
                f"风险原因：{review.get('原因', '结构保护未通过')}",
                "说明：综合得分只用于候选排序，不代表结构安全。",
            ))
        lines.extend(("", "算法组合", self._format_candidate(candidate)))
        return "\n".join(lines)

    def _clear_candidates(self, message: str) -> None:
        self._candidates = []
        self._selected_index = -1
        self._candidate_list.clear()
        self._update_candidate_grid()
        self._candidate_title.setText("候选效果")
        self._selected_preview.setPixmap(QPixmap())
        self._selected_preview.setText("正在处理" if "正在" in message else "暂无图片")
        self._scheme_label.setPlainText("请从候选效果中选择一张图片。")
        self._save_button.setText("采用并保存")
        self._sync_history_view()
        self._message_label.setText(message)
        self._set_workspace_enabled(False)

    def _clear_current_workspace(self, message: str) -> None:
        """列表没有可选字形时，清除上一字形的全部工作区状态。"""
        self._current_item = None
        self._reoptimization_key = ""
        self._release_original_image()
        self._original_preview.setPixmap(QPixmap())
        self._original_preview.setText("暂无图片")
        self._round_number = 1
        self._history = []
        self._clear_candidates(message)
        self._history_label.setText("暂无处理记录")

    def _current_item_is_optimizable(self) -> bool:
        current_key = str(self._current_item.get("键", "")) if self._current_item else ""
        return bool(
            self._current_item
            and (
                self._item_phase_status(self._current_item)
                == STAGE_PENDING_OPTIMIZATION
                or current_key == self._reoptimization_key
            )
            and str(self._current_item.get("原始路径", "")).strip()
        )

    def _show_readonly_current_item(self) -> None:
        """后续阶段和文件异常记录只用于核对，不启动算法或允许回写。"""
        item = self._current_item
        if item is None:
            return
        status = self._item_phase_status(item)
        source_path = str(item.get("原始路径", "") or "")
        preview_path = str(item.get("优化预览路径", "") or "")
        self._clear_candidates("")
        self._history_label.setText("当前记录为只读")
        if preview_path:
            try:
                with Image.open(preview_path) as source:
                    preview = source.convert("RGBA")
                self._set_preview(
                    self._selected_preview,
                    preview,
                    self._preview_background == "透明底",
                )
                preview.close()
            except Exception:
                self._selected_preview.setPixmap(QPixmap())
                self._selected_preview.setText("图片无法读取")
        if status == STAGE_PENDING_OPTIMIZATION and not source_path:
            message = "原图文件不可用，已保留字形记录；请核对“文件异常”后再优化。"
        else:
            message = "当前字形已经完成自动优化，本页面仅供查看。"
        self._candidate_title.setText("当前优化结果")
        self._scheme_label.setPlainText(message)
        self._message_label.setText(message)
        self._set_workspace_enabled(False)
        can_reoptimize = bool(
            status == STATUS_OPTIMIZED
            and source_path
            and not self._busy
        )
        self._restart_button.setText("重新优化此字形")
        self._restart_button.setEnabled(can_reoptimize)

    def _release_original_image(self) -> None:
        image = self._original_image
        self._original_image = None
        self._original_image_path = ""
        if image is not None:
            image.close()

    def _set_workspace_enabled(self, enabled: bool) -> None:
        active = enabled and not self._busy
        navigation_enabled = not self._busy
        self._engine_combo.setEnabled(navigation_enabled)
        self._home_button.setEnabled(navigation_enabled)
        self._complete_button.setEnabled(
            navigation_enabled
            and self._service is not None
            and self._bulk_worker is None
        )
        self._search_edit.setEnabled(navigation_enabled)
        self._search_button.setEnabled(navigation_enabled)
        self._status_combo.setEnabled(navigation_enabled)
        self._sort_combo.setEnabled(navigation_enabled)
        self._item_tree.setEnabled(navigation_enabled)
        self._candidate_list.setEnabled(active)
        selected_type = ""
        if 0 <= self._selected_index < len(self._candidates):
            selected_type = str(self._candidates[self._selected_index].get("处理类型", CANDIDATE_TYPE_OPTIMIZED))
        self._explore_button.setEnabled(active and selected_type == CANDIDATE_TYPE_OPTIMIZED)
        readonly_reoptimization = bool(
            navigation_enabled
            and not self._workers
            and self._current_item
            and self._item_phase_status(self._current_item) == STATUS_OPTIMIZED
            and str(self._current_item.get("键", "")) != self._reoptimization_key
            and str(self._current_item.get("原始路径", "")).strip()
        )
        self._restart_button.setText(
            "重新优化此字形"
            if readonly_reoptimization
            else "从原图重新生成候选"
        )
        self._restart_button.setEnabled(active or readonly_reoptimization)
        self._save_button.setEnabled(active)
        self._skip_button.setEnabled(not self._busy and bool(self._variant_nodes))
        row = self._current_variant_index()
        self._previous_button.setEnabled(not self._busy and row > 0)
        self._next_button.setEnabled(not self._busy and 0 <= row < len(self._variant_nodes) - 1)

    def _current_variant_index(self) -> int:
        current = self._item_tree.currentItem()
        try:
            return self._variant_nodes.index(current)
        except ValueError:
            return -1

    def shutdown(self) -> None:
        """关闭程序时取消批处理并使候选和缩略图后台结果失效。"""

        self._request_id += 1
        self._list_thumbnail_generation += 1
        self._list_row_height_timer.stop()
        if self._bulk_worker is not None:
            self._bulk_worker.request_cancel()
        self._list_thumbnail_inflight.clear()
        self._list_thumbnail_workers.clear()
        self._workers.clear()
        self._clear_candidate_cache()
        self._release_original_image()
