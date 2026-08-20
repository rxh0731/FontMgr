"""字库最终成品导出工作台。"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Optional

from PySide6.QtCore import QPoint, QObject, QRunnable, QSize, Qt, QThreadPool, QTimer, Signal, Slot
from PySide6.QtGui import QBrush, QColor, QIcon, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QSizePolicy,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

import config
from services.export_service import (
    ExportConflict,
    ExportConflictDecision,
    ExportOptions,
    ExportService,
)
from services.glyph_service import GlyphService
from services.workflow_status_service import (
    COORDINATION_STATUS_FILTERS,
    MARKER_FILE_ERROR,
    MARKER_INK_EXCEPTION,
    MARKER_INK_PENDING,
    MARKER_STRUCTURE_REVIEW,
    MARKER_UNSAVED,
    PHASE_COORDINATION,
    PHASE_STATUS_COLORS,
    STAGE_PENDING_COORDINATION,
    STATUS_COORDINATED,
    WORKFLOW_MARKERS,
    WorkflowStageProjection,
    WorkflowStatus,
    project_stage_status,
    resolve_safe_stage_file,
)
from ui.widgets.export_gallery import (
    ExportGallery,
    ExportGalleryEntry,
    decode_thumbnail_image,
)
from ui.widgets.glyph_rename_dialog import run_glyph_rename_dialog
from ui.workers import FunctionWorker, log_background_exception
from utils.file_utils import natural_key, pinyin_natural_key


class _ExportSignals(QObject):
    """导出后台任务信号。"""

    progress = Signal(str, int, int)
    finished = Signal(object)
    failed = Signal(str)


class _ExportWorker(QRunnable):
    """在线程池执行导出，避免大字库阻塞界面。"""

    def __init__(
        self,
        glyph_service: GlyphService,
        output_dir: str,
        options: ExportOptions,
        cancel_event: threading.Event,
        eligible_variant_ids: set[str],
        conflict_decisions: tuple[ExportConflictDecision, ...] = (),
    ) -> None:
        super().__init__()
        self._glyph = glyph_service
        self._output_dir = output_dir
        self._options = options
        self._cancel_event = cancel_event
        self._eligible_variant_ids = frozenset(eligible_variant_ids)
        self._conflict_decisions = tuple(conflict_decisions)
        self.signals = _ExportSignals()

    @Slot()
    def run(self) -> None:
        try:
            service = ExportService(
                self._glyph,
                progress_callback=self.signals.progress.emit,
            )
            result = service.export(
                self._output_dir,
                name_mode=self._options.name_mode,
                transparent_background=True,
                output_style="灰度保真",
                options=self._options,
                require_ready=False,
                cancel_check=self._cancel_event.is_set,
                eligible_variant_ids=self._eligible_variant_ids,
                conflict_decisions=self._conflict_decisions,
            )
        except Exception as exc:
            log_background_exception("字库导出")
            try:
                self.signals.failed.emit(str(exc))
            except RuntimeError:
                pass
        else:
            try:
                self.signals.finished.emit(result)
            except RuntimeError:
                pass


class ExportPage(QWidget):
    """以三栏工作台预览并导出全库最终成品。"""

    home_requested = Signal()
    summary_changed = Signal(object)
    status_message = Signal(str)

    LIST_PANEL_MIN_WIDTH = 250
    LIST_PANEL_DEFAULT_WIDTH = 275
    LIST_PANEL_MAX_WIDTH = 390
    OPTION_PANEL_MIN_WIDTH = 248
    OPTION_PANEL_DEFAULT_WIDTH = 276
    OPTION_PANEL_MAX_WIDTH = 330
    DEFAULT_COLUMNS = 8
    MIN_COLUMNS = 2
    MAX_COLUMNS = 16
    MAX_CUSTOM_DIMENSION = 16_384
    MAX_CUSTOM_PIXELS = 64 * 1024 * 1024
    LIST_THUMBNAIL_SIZE = 38
    LIST_THUMBNAIL_CACHE_ITEMS = 256
    LIST_THUMBNAIL_PREFETCH_ITEMS = 8

    STATUS_FILTERS = COORDINATION_STATUS_FILTERS
    SORT_OPTIONS = ("拼音顺序", "文件名顺序", "导入顺序")

    def __init__(
        self,
        glyph_service: GlyphService,
        on_back: Optional[Callable[[], None]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._glyph = glyph_service
        self._on_back = on_back
        self._workflow_summary = self._glyph.get_coordination_summary()
        self._finished_dir = self._glyph.get_workflow_dirs()["成品"]
        metadata = self._glyph.get_metadata()
        self._canvas_width = self._positive_int(metadata.get("画布宽"), 250)
        self._canvas_height = self._positive_int(metadata.get("画布高"), 250)
        self._thread_pool = QThreadPool.globalInstance()
        self._audit_workers: set[FunctionWorker] = set()
        self._active_worker: _ExportWorker | None = None
        self._cancel_event = threading.Event()
        self._audit_cancel_event = threading.Event()
        self._audit_in_progress = False
        self._shutdown = False
        self._all_variants: list[dict[str, Any]] = []
        self._phase_variants: list[dict[str, Any]] = []
        self._visible_variants: list[dict[str, Any]] = []
        self._import_order: dict[str, int] = {}
        self._variant_number_by_id: dict[str, int] = {}
        self._variant_ready: dict[str, bool] = {}
        self._audit_issue_ids: set[str] = set()
        self._workflow_status_cache: dict[str, WorkflowStageProjection] = {}
        self._items_by_id: dict[str, QTreeWidgetItem] = {}
        self._details_by_id: dict[str, dict[str, Any]] = {}
        self._selected_id = ""
        self._audit: dict[str, Any] = {}
        self._thumbnail_pool = QThreadPool(self)
        self._thumbnail_pool.setMaxThreadCount(2)
        self._thumbnail_pool.setExpiryTimeout(15_000)
        self._thumbnail_generation = 0
        self._thumbnail_workers: dict[
            str,
            tuple[int, tuple[str, int, int], FunctionWorker],
        ] = {}
        self._thumbnail_cache: OrderedDict[
            str,
            tuple[tuple[str, int, int], QIcon],
        ] = OrderedDict()
        self._thumbnail_failures: set[
            tuple[str, tuple[str, int, int]]
        ] = set()
        self._thumbnail_timer = QTimer(self)
        self._thumbnail_timer.setSingleShot(True)
        self._thumbnail_timer.setInterval(35)
        self._thumbnail_timer.timeout.connect(self._load_visible_list_thumbnails)

        self._build_ui()
        self._reload_variants()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)
        root.addWidget(self._build_header())

        self._main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._main_splitter.setChildrenCollapsible(False)
        self._main_splitter.addWidget(self._build_glyph_list())
        self._main_splitter.addWidget(self._build_gallery_panel())
        self._main_splitter.addWidget(self._build_options_panel())
        self._main_splitter.setStretchFactor(0, 0)
        self._main_splitter.setStretchFactor(1, 1)
        self._main_splitter.setStretchFactor(2, 0)
        self._main_splitter.setSizes(
            [
                self.LIST_PANEL_DEFAULT_WIDTH,
                820,
                self.OPTION_PANEL_DEFAULT_WIDTH,
            ]
        )
        root.addWidget(self._main_splitter, 1)

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setProperty("role", "card")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(12, 9, 12, 9)
        layout.setSpacing(10)

        brand = QLabel("导")
        brand.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brand.setFixedSize(36, 36)
        brand.setStyleSheet(
            "background: #315f9a; color: #ffffff; border-radius: 5px; "
            "font-size: 17px; font-weight: 700;"
        )
        layout.addWidget(brand)

        title_box = QVBoxLayout()
        title_box.setSpacing(1)
        title = QLabel("字库导出")
        title.setStyleSheet("font-size: 20px; font-weight: 700;")
        title_box.addWidget(title)
        metadata = self._glyph.get_metadata()
        summary = QLabel(
            f"当前字库：{self._glyph.ziku_name} · "
            f"{metadata.get('DPI', '--')} DPI · "
            f"{metadata.get('画布宽', '--')}×{metadata.get('画布高', '--')} 像素"
        )
        summary.setProperty("role", "muted")
        title_box.addWidget(summary)
        layout.addLayout(title_box)
        layout.addStretch(1)

        self._readiness_badge = QFrame()
        self._readiness_badge.setObjectName("exportReadinessBadge")
        badge_layout = QHBoxLayout(self._readiness_badge)
        badge_layout.setContentsMargins(10, 6, 10, 6)
        badge_layout.setSpacing(7)
        self._readiness_dot = QLabel()
        self._readiness_dot.setFixedSize(10, 10)
        badge_layout.addWidget(self._readiness_dot)
        self._readiness_label = QLabel("正在核对全库状态")
        self._readiness_label.setStyleSheet("font-weight: 600;")
        badge_layout.addWidget(self._readiness_label)
        layout.addWidget(self._readiness_badge)

        back = QPushButton("返回首页")
        back.clicked.connect(self.request_back)
        layout.addWidget(back)
        return header

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
        heading.addStretch(1)
        self._list_count_label = QLabel("显示 / 总数：0 / 0")
        self._list_count_label.setProperty("role", "muted")
        heading.addWidget(self._list_count_label)
        layout.addLayout(heading)

        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("搜索字符、字形或文件名")
        self._search_edit.setClearButtonEnabled(True)
        self._search_edit.textChanged.connect(self._apply_filters)
        layout.addWidget(self._search_edit)

        filter_sort_row = QHBoxLayout()
        filter_sort_row.setSpacing(4)
        self._filter_combo = QComboBox()
        self._filter_combo.addItems(self.STATUS_FILTERS)
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
        self._sort_combo = QComboBox()
        self._sort_combo.addItems(self.SORT_OPTIONS)
        self._sort_combo.setToolTip("调整字形排序")
        self._sort_combo.currentTextChanged.connect(self._apply_filters)
        self._sort_combo.setStyleSheet(
            "QComboBox { padding-left: 4px; padding-right: 4px; }"
        )
        self._sort_combo.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Fixed,
        )
        filter_sort_row.addWidget(self._sort_combo, 4)
        layout.addLayout(filter_sort_row)

        self._glyph_list = QTreeWidget()
        self._glyph_list.setColumnCount(4)
        self._glyph_list.setHeaderLabels(
            ("字形与文件", "协调状态", "提示", "导出")
        )
        self._glyph_list.setRootIsDecorated(True)
        self._glyph_list.setIndentation(14)
        self._glyph_list.setUniformRowHeights(False)
        self._glyph_list.setAlternatingRowColors(False)
        self._glyph_list.setWordWrap(True)
        self._glyph_list.setAnimated(False)
        self._glyph_list.setIconSize(
            QSize(self.LIST_THUMBNAIL_SIZE, self.LIST_THUMBNAIL_SIZE)
        )
        self._glyph_list.setStyleSheet(
            "QTreeWidget { background: #171b22; border: 1px solid #37404d; }"
            "QTreeWidget::item { min-height: 26px; padding: 1px 3px; }"
            "QTreeWidget::item:selected { background: #3c4773; }"
        )
        header = self._glyph_list.header()
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(48)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        status_width = max(
            self._glyph_list.fontMetrics().horizontalAdvance(value)
            for value in (*self.STATUS_FILTERS[1:], "已协调 99/99")
        )
        marker_width = max(
            self._glyph_list.fontMetrics().horizontalAdvance(value)
            for value in (*WORKFLOW_MARKERS, "问题 99")
        )
        export_width = max(
            self._glyph_list.fontMetrics().horizontalAdvance(value)
            for value in ("可导出", "不可导出", "可导出 99/99")
        )
        header.resizeSection(1, status_width + 20)
        header.resizeSection(2, marker_width + 24)
        header.resizeSection(3, export_width + 20)
        self._glyph_list.currentItemChanged.connect(self._tree_selection_changed)
        self._glyph_list.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self._glyph_list.customContextMenuRequested.connect(
            self._show_glyph_context_menu
        )
        self._glyph_list.itemExpanded.connect(self._schedule_list_thumbnail_loads)
        self._glyph_list.itemCollapsed.connect(self._schedule_list_thumbnail_loads)
        self._glyph_list.verticalScrollBar().valueChanged.connect(
            self._schedule_list_thumbnail_loads
        )
        layout.addWidget(self._glyph_list, 1)

        self._summary_label = QLabel(
            "待协调 0　已协调 0\n可导出 0 / 0"
        )
        self._summary_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._summary_label.setProperty("role", "muted")
        self._summary_label.setWordWrap(True)
        layout.addWidget(self._summary_label)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("成品可用率 %p%")
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFixedHeight(20)
        layout.addWidget(self._progress_bar)
        return panel

    def _build_gallery_panel(self) -> QWidget:
        panel = QFrame()
        panel.setProperty("role", "card")
        panel.setMinimumWidth(300)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(10, 7, 10, 7)
        toolbar_layout.setSpacing(8)
        title = QLabel("成品总览")
        title.setStyleSheet("font-size: 17px; font-weight: 700;")
        toolbar_layout.addWidget(title)
        self._gallery_count_label = QLabel("0 个字形")
        self._gallery_count_label.setProperty("role", "muted")
        toolbar_layout.addWidget(self._gallery_count_label)
        toolbar_layout.addStretch(1)
        toolbar_layout.addWidget(QLabel("每行"))
        self._column_spin = QSpinBox()
        self._column_spin.setRange(self.MIN_COLUMNS, self.MAX_COLUMNS)
        self._column_spin.setValue(self.DEFAULT_COLUMNS)
        self._column_spin.setSuffix(" 个")
        self._column_spin.setFixedWidth(82)
        self._column_spin.valueChanged.connect(self._set_gallery_columns)
        toolbar_layout.addWidget(self._column_spin)
        layout.addWidget(toolbar)

        self._gallery = ExportGallery()
        self._gallery.set_canvas_size(self._canvas_width, self._canvas_height)
        self._gallery.set_column_count(self.DEFAULT_COLUMNS)
        self._gallery.variant_selected.connect(self._gallery_selection_changed)
        layout.addWidget(self._gallery, 1)
        return panel

    def _build_options_panel(self) -> QWidget:
        panel = QFrame()
        panel.setProperty("role", "card")
        panel.setMinimumWidth(self.OPTION_PANEL_MIN_WIDTH)
        panel.setMaximumWidth(self.OPTION_PANEL_MAX_WIDTH)
        root = QVBoxLayout(panel)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._options_scroll = QScrollArea()
        self._options_scroll.setWidgetResizable(True)
        self._options_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._options_host = QWidget()
        layout = QVBoxLayout(self._options_host)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QLabel("导出设置")
        title.setStyleSheet("font-size: 18px; font-weight: 700;")
        layout.addWidget(title)

        layout.addWidget(self._section_title("导出位置"))
        self._directory_edit = QLineEdit()
        self._directory_edit.setPlaceholderText("请选择导出目录")
        self._directory_edit.textChanged.connect(self._refresh_export_button)
        layout.addWidget(self._directory_edit)
        browse = QPushButton("选择目录")
        browse.clicked.connect(self._choose_directory)
        layout.addWidget(browse)

        layout.addWidget(self._horizontal_separator())
        layout.addWidget(self._section_title("保存方式"))
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._mode_buttons: dict[str, QPushButton] = {}
        modes = (
            (ExportService.MODE_LIBRARY_SPEC, "按照字库参数"),
            (ExportService.MODE_TRIM_TRANSPARENT, "去除透明区"),
            (ExportService.MODE_CUSTOM_SPEC, "自定义参数"),
        )
        for mode, text in modes:
            button = QPushButton(text)
            button.setCheckable(True)
            button.setProperty("controlRole", "segment")
            button.clicked.connect(
                lambda _checked=False, selected_mode=mode: self._set_export_mode(
                    selected_mode
                )
            )
            self._mode_group.addButton(button)
            self._mode_buttons[mode] = button
            layout.addWidget(button)
        self._mode_buttons[ExportService.MODE_LIBRARY_SPEC].setChecked(True)
        self._export_mode = ExportService.MODE_LIBRARY_SPEC

        metadata = self._glyph.get_metadata()
        self._library_spec_label = QLabel(
            f"字库参数：{metadata.get('DPI', '--')} DPI，"
            f"{metadata.get('画布宽', '--')}×{metadata.get('画布高', '--')} 像素"
        )
        self._library_spec_label.setWordWrap(True)
        self._library_spec_label.setProperty("role", "muted")
        layout.addWidget(self._library_spec_label)

        self._custom_panel = QWidget()
        custom_form = QFormLayout(self._custom_panel)
        custom_form.setContentsMargins(0, 0, 0, 0)
        custom_form.setHorizontalSpacing(8)
        custom_form.setVerticalSpacing(8)
        self._dpi_spin = QSpinBox()
        self._dpi_spin.setRange(1, 9600)
        self._dpi_spin.setSuffix(" DPI")
        self._dpi_spin.setValue(self._positive_int(metadata.get("DPI"), 300))
        self._width_spin = QSpinBox()
        self._width_spin.setRange(1, self.MAX_CUSTOM_DIMENSION)
        self._width_spin.setSuffix(" px")
        self._width_spin.setValue(
            min(
                self.MAX_CUSTOM_DIMENSION,
                self._positive_int(metadata.get("画布宽"), 250),
            )
        )
        self._height_spin = QSpinBox()
        self._height_spin.setRange(1, self.MAX_CUSTOM_DIMENSION)
        self._height_spin.setSuffix(" px")
        self._height_spin.setValue(
            min(
                self.MAX_CUSTOM_DIMENSION,
                self._positive_int(metadata.get("画布高"), 250),
            )
        )
        custom_form.addRow("画布 DPI", self._dpi_spin)
        custom_form.addRow("画布宽度", self._width_spin)
        custom_form.addRow("画布高度", self._height_spin)
        self._include_transparent_check = QCheckBox("包含透明区")
        self._include_transparent_check.setChecked(True)
        custom_form.addRow("缩放依据", self._include_transparent_check)
        layout.addWidget(self._custom_panel)

        for control in (
            self._dpi_spin,
            self._width_spin,
            self._height_spin,
        ):
            control.valueChanged.connect(self._update_option_summary)
        self._include_transparent_check.toggled.connect(self._update_option_summary)

        self._option_summary_label = QLabel()
        self._option_summary_label.setWordWrap(True)
        self._option_summary_label.setProperty("role", "muted")
        layout.addWidget(self._option_summary_label)

        layout.addWidget(self._horizontal_separator())
        layout.addWidget(self._section_title("文件命名"))
        self._name_mode_combo = QComboBox()
        self._name_mode_combo.addItem("使用字符命名", "字符")
        self._name_mode_combo.addItem("使用原文件名", "原文件名")
        layout.addWidget(self._name_mode_combo)
        format_label = QLabel("文件格式：PNG（保留透明背景）")
        format_label.setProperty("role", "muted")
        format_label.setWordWrap(True)
        layout.addWidget(format_label)
        layout.addStretch(1)
        self._options_scroll.setWidget(self._options_host)
        root.addWidget(self._options_scroll, 1)

        self._action_footer = QWidget()
        footer = QVBoxLayout(self._action_footer)
        footer.setContentsMargins(12, 10, 12, 12)
        footer.setSpacing(7)
        self._export_status_label = QLabel("请选择导出目录")
        self._export_status_label.setProperty("role", "muted")
        self._export_status_label.setWordWrap(True)
        footer.addWidget(self._export_status_label)
        self._export_progress = QProgressBar()
        self._export_progress.setRange(0, 1)
        self._export_progress.setValue(0)
        self._export_progress.setVisible(False)
        footer.addWidget(self._export_progress)
        action_row = QHBoxLayout()
        action_row.setSpacing(6)
        self._cancel_button = QPushButton("取消导出")
        self._cancel_button.setVisible(False)
        self._cancel_button.clicked.connect(self.cancel_export)
        action_row.addWidget(self._cancel_button)
        self._export_button = QPushButton("开始导出")
        self._export_button.setProperty("role", "primary")
        self._export_button.clicked.connect(self._start_export)
        action_row.addWidget(self._export_button, 1)
        footer.addLayout(action_row)
        root.addWidget(self._action_footer)

        self._set_export_mode(ExportService.MODE_LIBRARY_SPEC)
        self._refresh_export_button()
        return panel

    def _reload_variants(self) -> None:
        self._thumbnail_generation += 1
        self._thumbnail_timer.stop()
        self._thumbnail_pool.clear()
        self._thumbnail_workers.clear()
        self._thumbnail_failures.clear()
        self._audit_issue_ids.clear()
        self._workflow_summary = self._glyph.get_coordination_summary()
        self._finished_dir = self._glyph.get_workflow_dirs()["成品"]
        self._workflow_status_cache.clear()
        self._all_variants = list(self._glyph.get_all_variants())
        self._details_by_id = {
            str(detail.get("变体ID", "")): detail
            for detail in self._all_variants
            if detail.get("变体ID")
        }
        self._import_order = {
            str(detail.get("变体ID", "")): index
            for index, detail in enumerate(self._all_variants)
        }
        self._variant_number_by_id = {
            str(variant_id): number
            for variant_ids in self._glyph.get_glyph_groups().values()
            for number, variant_id in enumerate(variant_ids, start=1)
        }
        self._variant_ready = {
            str(detail.get("变体ID", "")): self._is_variant_ready(
                detail,
                self._finished_dir,
            )
            for detail in self._all_variants
        }
        for detail in self._all_variants:
            self._stage_projection(detail)
        self._phase_variants = [
            detail
            for detail in self._all_variants
            if self._stage_projection(detail).admitted
        ]
        phase_ids = {
            str(detail.get("变体ID", "")) for detail in self._phase_variants
        }
        if self._selected_id not in phase_ids:
            self._selected_id = (
                str(self._phase_variants[0].get("变体ID", ""))
                if self._phase_variants
                else ""
            )
        self._audit = self._fast_audit()
        self._apply_audit(self._audit)
        self._apply_filters()
        self._start_readiness_audit()

    def _apply_filters(self, _value: object = None) -> None:
        query = self._search_edit.text().strip().casefold()
        status_filter = self._filter_combo.currentText()
        variants: list[dict[str, Any]] = []
        for detail in self._phase_variants:
            projection = self._stage_projection(detail)
            searchable = " ".join(
                str(detail.get(key, ""))
                for key in ("归属字", "原始文件", "导入前文件名", "成品文件")
            ).casefold()
            if query and query not in searchable:
                continue
            if not projection.matches_status(status_filter):
                continue
            variants.append(detail)
        self._sort_variants(variants)
        self._visible_variants = variants
        if (
            self._selected_id
            not in {str(detail.get("变体ID", "")) for detail in variants}
            and variants
        ):
            self._selected_id = str(variants[0].get("变体ID", ""))
        self._populate_list()
        self._populate_gallery()

    def _sort_variants(self, variants: list[dict[str, Any]]) -> None:
        mode = self._sort_combo.currentText()
        if mode == "文件名顺序":
            variants.sort(key=lambda item: natural_key(str(item.get("原始文件", ""))))
        elif mode == "导入顺序":
            variants.sort(
                key=lambda item: self._import_order.get(
                    str(item.get("变体ID", "")),
                    0,
                )
            )
        else:
            variants.sort(
                key=lambda item: (
                    pinyin_natural_key(str(item.get("归属字", ""))),
                    natural_key(str(item.get("原始文件", ""))),
                )
            )

    def _workflow_status(self, detail: dict[str, Any]) -> WorkflowStatus:
        return self._stage_projection(detail).workflow

    def _stage_projection(
        self,
        detail: dict[str, Any],
    ) -> WorkflowStageProjection:
        variant_id = str(detail.get("变体ID", ""))
        cached = self._workflow_status_cache.get(variant_id)
        if variant_id and cached is not None:
            return cached
        projection = project_stage_status(
            detail,
            self._workflow_summary,
            self._finished_dir,
            PHASE_COORDINATION,
        )
        if variant_id:
            self._workflow_status_cache[variant_id] = projection
        return projection

    def _is_problem_status(
        self,
        status: WorkflowStageProjection | WorkflowStatus,
        variant_id: str,
    ) -> bool:
        return (
            variant_id in self._audit_issue_ids
            or any(marker != MARKER_INK_EXCEPTION for marker in status.markers)
        )

    def _export_marker_text(
        self,
        status: WorkflowStageProjection | WorkflowStatus,
        variant_id: str,
    ) -> str:
        markers = list(status.markers)
        if variant_id in self._audit_issue_ids and MARKER_FILE_ERROR not in markers:
            markers.append(MARKER_FILE_ERROR)
        return " · ".join(markers) if markers else "无"

    @staticmethod
    def _stage_color(status: str) -> QColor:
        return QColor(PHASE_STATUS_COLORS.get(status, "#D7DEE8"))

    def _marker_color(
        self,
        status: WorkflowStageProjection | WorkflowStatus,
        variant_id: str,
    ) -> QColor:
        if variant_id in self._audit_issue_ids or status.has_marker(MARKER_FILE_ERROR):
            return QColor("#E36A6A")
        if any(
            status.has_marker(marker)
            for marker in (MARKER_UNSAVED, MARKER_STRUCTURE_REVIEW, MARKER_INK_PENDING)
        ):
            return QColor("#F2B84B")
        if status.has_marker(MARKER_INK_EXCEPTION):
            return QColor("#4DA3FF")
        return QColor("#A6B0BE")

    def _populate_list(self) -> None:
        self._glyph_list.blockSignals(True)
        self._glyph_list.clear()
        self._items_by_id.clear()
        selected_item: QTreeWidgetItem | None = None
        groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        for detail in self._visible_variants:
            groups.setdefault(str(detail.get("归属字", "?") or "?"), []).append(detail)
        all_groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        for detail in self._phase_variants:
            all_groups.setdefault(
                str(detail.get("归属字", "?") or "?"),
                [],
            ).append(detail)

        for char, group_variants in groups.items():
            all_group_variants = all_groups.get(char, group_variants)
            projections = [
                self._stage_projection(detail) for detail in all_group_variants
            ]
            coordinated_count = sum(
                projection.completed for projection in projections
            )
            ready_count = sum(
                self._variant_ready.get(str(detail.get("变体ID", "")), False)
                for detail in all_group_variants
            )
            problem_count = sum(
                self._is_problem_status(
                    projection,
                    str(detail.get("变体ID", "")),
                )
                for detail, projection in zip(all_group_variants, projections)
            )
            parent = QTreeWidgetItem(
                [
                    f"{char}（{len(all_group_variants)}个字形）",
                    f"已协调 {coordinated_count}/{len(all_group_variants)}",
                    f"问题 {problem_count}",
                    f"可导出 {ready_count}/{len(all_group_variants)}",
                ]
            )
            parent.setFlags(parent.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            parent_font = parent.font(0)
            parent_font.setBold(True)
            parent.setFont(0, parent_font)
            parent.setFont(1, parent_font)
            parent.setFont(2, parent_font)
            parent.setFont(3, parent_font)
            parent.setForeground(
                1,
                QBrush(
                    self._stage_color(
                        STATUS_COORDINATED
                        if coordinated_count == len(all_group_variants)
                        else STAGE_PENDING_COORDINATION
                    )
                ),
            )
            parent.setForeground(
                2,
                QBrush(QColor("#F2B84B" if problem_count else "#A6B0BE")),
            )
            parent.setForeground(
                3,
                QBrush(
                    QColor(
                        "#48C78E"
                        if ready_count == len(all_group_variants)
                        else "#E36A6A"
                    )
                ),
            )
            parent.setTextAlignment(
                1,
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            )
            parent.setTextAlignment(
                2,
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            )
            parent.setTextAlignment(
                3,
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            )
            parent.setToolTip(
                0,
                f"{char}：共 {len(all_group_variants)} 个字形，"
                f"当前显示 {len(group_variants)} 个\n"
                f"已协调 {coordinated_count}，待协调 "
                f"{len(all_group_variants) - coordinated_count}\n"
                f"有问题 {problem_count}，可导出 {ready_count}",
            )
            self._glyph_list.addTopLevelItem(parent)

            for visible_number, detail in enumerate(group_variants, start=1):
                variant_id = str(detail.get("变体ID", ""))
                variant_number = self._variant_number_by_id.get(
                    variant_id,
                    visible_number,
                )
                filename = str(detail.get("原始文件", ""))
                ready = self._variant_ready.get(variant_id, False)
                projection = self._stage_projection(detail)
                status = projection.status
                markers = self._export_marker_text(projection, variant_id)
                item = QTreeWidgetItem(
                    parent,
                    [
                        f"字形{variant_number} · {filename}",
                        status,
                        markers,
                        "可导出" if ready else "不可导出",
                    ],
                )
                item.setIcon(0, self._cached_or_placeholder_icon(detail))
                item.setData(0, Qt.ItemDataRole.UserRole, variant_id)
                item.setSizeHint(0, QSize(0, 52))
                item.setToolTip(
                    0,
                    f"{char} · 字形{variant_number}\n"
                    f"{variant_id}\n文件：{filename}\n"
                    f"整体协调：{status}\n提示：{markers}\n"
                    f"导出：{'可导出' if ready else '不可导出'}",
                )
                item.setForeground(1, QBrush(self._stage_color(status)))
                item.setForeground(
                    2,
                    QBrush(self._marker_color(projection, variant_id)),
                )
                item.setForeground(
                    3,
                    QBrush(QColor("#48C78E" if ready else "#E36A6A")),
                )
                item.setTextAlignment(
                    1,
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                )
                item.setTextAlignment(
                    2,
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                )
                item.setTextAlignment(
                    3,
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                )
                self._items_by_id[variant_id] = item
                if variant_id == self._selected_id:
                    selected_item = item
            parent.setExpanded(True)
        if selected_item is not None:
            self._glyph_list.setCurrentItem(selected_item)
        self._glyph_list.blockSignals(False)
        self._list_count_label.setText(
            f"显示 / 本阶段：{len(self._visible_variants)} / "
            f"{len(self._phase_variants)}"
        )
        self._schedule_list_thumbnail_loads()

    def _populate_gallery(self) -> None:
        finished_dir = self._glyph.get_workflow_dirs()["成品"]
        entries = [
            ExportGalleryEntry(
                variant_id=str(detail.get("变体ID", "")),
                char=str(detail.get("归属字", "")),
                filename=str(detail.get("原始文件", "")),
                image_path=self._finished_image_path(detail, finished_dir),
                image_canvas_size=self._gallery_image_canvas_size(detail),
                status=self._stage_projection(detail).status,
            )
            for detail in self._visible_variants
        ]
        self._gallery.set_entries(entries)
        self._gallery_count_label.setText(f"{len(entries)} 个字形")
        self._select_gallery_variant(self._selected_id)

    def _gallery_image_canvas_size(
        self,
        detail: dict[str, Any],
    ) -> tuple[int, int] | None:
        """返回成品实际画布；旧记录缺少尺寸时交由画廊等比适配。"""
        parameters = detail.get("整体协调参数", {})
        if isinstance(parameters, dict):
            actual_size = parameters.get("实际画布")
            if isinstance(actual_size, (list, tuple)) and len(actual_size) == 2:
                try:
                    width = int(actual_size[0])
                    height = int(actual_size[1])
                except (TypeError, ValueError):
                    pass
                else:
                    if width > 0 and height > 0:
                        return width, height
        return None

    def _tree_selection_changed(
        self,
        current: QTreeWidgetItem | None,
        previous: QTreeWidgetItem | None,
    ) -> None:
        if current is None:
            return
        variant_id = str(current.data(0, Qt.ItemDataRole.UserRole) or "")
        if not variant_id:
            fallback = previous
            if (
                fallback is None
                or not str(fallback.data(0, Qt.ItemDataRole.UserRole) or "")
            ):
                fallback = self._items_by_id.get(self._selected_id)
            if fallback is not None:
                self._glyph_list.blockSignals(True)
                self._glyph_list.setCurrentItem(fallback)
                self._glyph_list.blockSignals(False)
            return
        self._selected_id = variant_id
        self._select_gallery_variant(variant_id)
        self._schedule_list_thumbnail_loads()
    def _show_glyph_context_menu(self, position: object) -> None:
        node = self._glyph_list.itemAt(position)
        if node is None:
            return
        variant_id = str(node.data(0, Qt.ItemDataRole.UserRole) or "")
        if not variant_id:
            return
        self._glyph_list.setCurrentItem(node)
        menu = QMenu(self)
        action = menu.addAction("修正字形名称…")
        action.setEnabled(
            self._active_worker is None and not self._audit_in_progress
        )
        action.triggered.connect(self._rename_current_glyph)
        menu.exec(self._glyph_list.viewport().mapToGlobal(position))

    def _rename_current_glyph(self) -> None:
        if not self._selected_id:
            QMessageBox.information(self, "修正字形名称", "请先选择一个具体字形。")
            return
        if self._active_worker is not None or self._audit_in_progress:
            QMessageBox.information(
                self,
                "暂时不能修改名称",
                "当前正在核对或导出字库，请等待任务结束后重试。",
            )
            return
        variant_id = self._selected_id
        result = run_glyph_rename_dialog(self, self._glyph, variant_id)
        if result is None:
            return
        self._thumbnail_generation += 1
        self._thumbnail_cache.clear()
        self._reload_variants()
        self.summary_changed.emit(self._glyph)
        self.status_message.emit(f"字形名称已修正为 {result.get('新文件名', '')}")
        QMessageBox.information(
            self,
            "名称修改完成",
            f"字形已修正为“{result.get('新归属字', '')}”，各阶段文件名已同步更新。",
        )

    def _gallery_selection_changed(self, variant_id: str) -> None:
        item = self._items_by_id.get(variant_id)
        if item is None:
            return
        self._selected_id = variant_id
        if item.parent() is not None:
            item.parent().setExpanded(True)
        self._glyph_list.setCurrentItem(item)
        self._glyph_list.scrollToItem(item)
        self._schedule_list_thumbnail_loads()

    def _select_gallery_variant(self, variant_id: str) -> None:
        setter = getattr(self._gallery, "set_selected_variant", None)
        if callable(setter):
            setter(variant_id)

    def _schedule_list_thumbnail_loads(self, _value: object = None) -> None:
        """合并滚动和展开事件，只为当前可见子项请求成品缩略图。"""
        if not self._shutdown:
            self._thumbnail_timer.start()

    def _load_visible_list_thumbnails(self) -> None:
        if self._shutdown or not self._items_by_id:
            return
        viewport = self._glyph_list.viewport()
        first = self._glyph_list.itemAt(QPoint(2, 2))
        if first is None:
            return
        item = first
        requested: list[str] = []
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

        if self._selected_id in self._items_by_id:
            requested.insert(0, self._selected_id)
        for variant_id in dict.fromkeys(requested):
            self._request_list_thumbnail(variant_id)

    def _request_list_thumbnail(self, variant_id: str) -> None:
        detail = self._details_by_id.get(variant_id)
        item = self._items_by_id.get(variant_id)
        if detail is None or item is None:
            return
        source = self._list_thumbnail_source(detail)
        if source is None:
            return
        image_path, signature = source
        cached = self._thumbnail_cache.get(variant_id)
        if cached is not None and cached[0] == signature:
            self._thumbnail_cache.move_to_end(variant_id)
            item.setIcon(0, cached[1])
            return
        pending = self._thumbnail_workers.get(variant_id)
        generation = self._thumbnail_generation
        if pending is not None and pending[:2] == (generation, signature):
            return

        worker = FunctionWorker(
            lambda path=image_path: decode_thumbnail_image(
                path,
                QSize(self.LIST_THUMBNAIL_SIZE, self.LIST_THUMBNAIL_SIZE),
                32 * 1024 * 1024,
            )
        )
        worker.setAutoDelete(False)
        worker.signals.finished.connect(
            lambda result,
            target=variant_id,
            version=signature,
            current_generation=generation,
            task=worker: (
                self._list_thumbnail_finished(
                    target,
                    version,
                    current_generation,
                    result,
                    task,
                )
            )
        )
        worker.signals.failed.connect(
            lambda _message,
            target=variant_id,
            version=signature,
            current_generation=generation,
            task=worker: (
                self._list_thumbnail_failed(
                    target,
                    version,
                    current_generation,
                    task,
                )
            )
        )
        self._thumbnail_workers[variant_id] = (generation, signature, worker)
        self._thumbnail_pool.start(worker)

    def _list_thumbnail_finished(
        self,
        variant_id: str,
        signature: tuple[str, int, int],
        generation: int,
        result: object,
        worker: FunctionWorker,
    ) -> None:
        self._release_thumbnail_worker(variant_id, worker)
        if self._shutdown or generation != self._thumbnail_generation:
            return
        if not isinstance(result, QImage) or result.isNull():
            self._thumbnail_failures.add((variant_id, signature))
            item = self._items_by_id.get(variant_id)
            detail = self._details_by_id.get(variant_id)
            if item is not None and detail is not None:
                item.setIcon(0, self._cached_or_placeholder_icon(detail))
            self._schedule_list_thumbnail_loads()
            return
        detail = self._details_by_id.get(variant_id)
        if detail is None:
            return
        current_source = self._list_thumbnail_source(detail)
        if current_source is None or current_source[1] != signature:
            self._schedule_list_thumbnail_loads()
            return
        icon = self._thumbnail_icon(result)
        self._thumbnail_cache[variant_id] = (signature, icon)
        self._thumbnail_cache.move_to_end(variant_id)
        while len(self._thumbnail_cache) > self.LIST_THUMBNAIL_CACHE_ITEMS:
            expired_id, _cached = self._thumbnail_cache.popitem(last=False)
            expired_item = self._items_by_id.get(expired_id)
            expired_detail = self._details_by_id.get(expired_id)
            if expired_item is not None and expired_detail is not None:
                expired_item.setIcon(0, self._cached_or_placeholder_icon(expired_detail))
        item = self._items_by_id.get(variant_id)
        if item is not None:
            item.setIcon(0, icon)

    def _list_thumbnail_failed(
        self,
        variant_id: str,
        signature: tuple[str, int, int],
        generation: int,
        worker: FunctionWorker,
    ) -> None:
        self._release_thumbnail_worker(variant_id, worker)
        if self._shutdown or generation != self._thumbnail_generation:
            return
        self._thumbnail_failures.add((variant_id, signature))
        detail = self._details_by_id.get(variant_id)
        item = self._items_by_id.get(variant_id)
        if detail is not None and item is not None:
            item.setIcon(0, self._cached_or_placeholder_icon(detail))
        self._schedule_list_thumbnail_loads()

    def _release_thumbnail_worker(
        self,
        variant_id: str,
        worker: FunctionWorker,
    ) -> None:
        pending = self._thumbnail_workers.get(variant_id)
        if pending is not None and pending[2] is worker:
            self._thumbnail_workers.pop(variant_id, None)

    def _list_thumbnail_source(
        self,
        detail: dict[str, Any],
    ) -> tuple[str, tuple[str, int, int]] | None:
        """返回当前阶段最可信的真实图片及可失效文件签名。"""
        variant_id = str(detail.get("变体ID", ""))
        directories = self._glyph.get_workflow_dirs()
        candidates = (
            ("成品", "成品文件"),
            ("手工审核", "审核文件"),
            ("优化预览", "中间文件"),
            ("清洁掩码", "清洁掩码文件"),
            ("灰度母版", "灰度母版文件"),
            ("原图", "原始文件"),
        )
        for directory_key, file_key in candidates:
            raw_filename = detail.get(file_key, "")
            if not str(raw_filename or "").strip():
                continue
            directory = directories.get(directory_key, "")
            path = resolve_safe_stage_file(directory, raw_filename)
            if not path:
                return None
            try:
                stat = os.stat(path)
            except OSError:
                return None
            normalized_path = os.path.normcase(os.path.abspath(path))
            signature = (normalized_path, stat.st_mtime_ns, stat.st_size)
            if (variant_id, signature) in self._thumbnail_failures:
                return None
            return path, signature
        return None

    def _cached_or_placeholder_icon(
        self,
        detail: dict[str, Any],
    ) -> QIcon:
        variant_id = str(detail.get("变体ID", ""))
        cached = self._thumbnail_cache.get(variant_id)
        if cached is None:
            return self._placeholder_icon()
        source = self._list_thumbnail_source(detail)
        if source is not None and cached[0] == source[1]:
            self._thumbnail_cache.move_to_end(variant_id)
            return cached[1]
        return self._placeholder_icon()

    def _set_gallery_columns(self, columns: int) -> None:
        self._gallery.set_column_count(columns)

    def _set_export_mode(self, mode: str) -> None:
        if mode not in self._mode_buttons:
            return
        self._export_mode = mode
        self._mode_buttons[mode].setChecked(True)
        self._custom_panel.setVisible(mode == ExportService.MODE_CUSTOM_SPEC)
        self._library_spec_label.setVisible(mode != ExportService.MODE_CUSTOM_SPEC)
        self._update_option_summary()

    def _update_option_summary(self, _value: object = None) -> None:
        metadata = self._glyph.get_metadata()
        if self._export_mode == ExportService.MODE_TRIM_TRANSPARENT:
            text = (
                f"保留 {metadata.get('DPI', '--')} DPI，裁掉文字外围的全透明区域。"
            )
        elif self._export_mode == ExportService.MODE_CUSTOM_SPEC:
            basis = "完整图片（包含透明区）" if self._include_transparent_check.isChecked() else "实际文字（不包含透明区）"
            text = (
                f"{self._dpi_spin.value()} DPI，{self._width_spin.value()}×"
                f"{self._height_spin.value()} 像素；按{basis}等比缩放并居中。"
            )
        else:
            text = (
                f"按字库设定的 {metadata.get('画布宽', '--')}×"
                f"{metadata.get('画布高', '--')} 像素和 "
                f"{metadata.get('DPI', '--')} DPI 原样导出。"
            )
        self._option_summary_label.setText(text)

    def _build_options(self) -> ExportOptions:
        name_mode = str(self._name_mode_combo.currentData() or "字符")
        if self._export_mode == ExportService.MODE_CUSTOM_SPEC:
            return ExportOptions(
                mode=self._export_mode,
                dpi=self._dpi_spin.value(),
                width=self._width_spin.value(),
                height=self._height_spin.value(),
                include_transparent_area=self._include_transparent_check.isChecked(),
                name_mode=name_mode,
            )
        return ExportOptions(
            mode=self._export_mode,
            include_transparent_area=(
                self._export_mode != ExportService.MODE_TRIM_TRANSPARENT
            ),
            name_mode=name_mode,
        )

    def _choose_directory(self) -> None:
        initial = self._directory_edit.text().strip() or os.path.expanduser("~")
        directory = QFileDialog.getExistingDirectory(self, "选择导出目录", initial)
        if directory:
            self._directory_edit.setText(directory)

    def _start_export(self) -> None:
        if self._active_worker is not None:
            return
        if self._audit_in_progress:
            QMessageBox.information(
                self,
                "正在核对全库状态",
                "正在逐项核对最终成品，请等待核对完成后再导出。",
            )
            return
        output_dir = self._directory_edit.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "缺少导出目录", "请先选择导出目录。")
            return
        if not os.path.isdir(output_dir):
            QMessageBox.warning(self, "导出目录无效", "所选导出目录不存在，请重新选择。")
            return
        if (
            self._export_mode == ExportService.MODE_CUSTOM_SPEC
            and self._width_spin.value() * self._height_spin.value()
            > self.MAX_CUSTOM_PIXELS
        ):
            QMessageBox.warning(
                self,
                "画布尺寸过大",
                "自定义画布超过 6400 万像素，请减小宽度或高度。",
            )
            return
        eligible_variant_ids = {
            variant_id
            for variant_id, available in self._variant_ready.items()
            if available
        }
        available_count = len(eligible_variant_ids)
        if available_count <= 0:
            QMessageBox.warning(
                self,
                "没有可导出的成品",
                "当前字库还没有通过完整性核对的最终成品，请先完成前序处理。",
            )
            return
        if not bool(self._audit.get("就绪", False)) and not self._confirm_partial_export(
            available_count,
            len(self._all_variants),
        ):
            return

        options = self._build_options()
        try:
            conflicts = ExportService(self._glyph).find_destination_conflicts(
                output_dir,
                options,
                eligible_variant_ids=eligible_variant_ids,
            )
        except (OSError, TypeError, ValueError) as exc:
            self._export_status_label.setText("无法准备导出")
            QMessageBox.critical(
                self,
                "无法准备导出",
                str(exc),
            )
            return
        conflict_decisions = self._resolve_export_conflicts(conflicts)
        if conflict_decisions is None:
            self._export_status_label.setText("导出已取消")
            self.status_message.emit("导出已取消")
            return

        self._cancel_event = threading.Event()
        worker = _ExportWorker(
            self._glyph,
            output_dir,
            options,
            self._cancel_event,
            eligible_variant_ids,
            conflict_decisions,
        )
        self._active_worker = worker
        worker.signals.progress.connect(self._export_progress_changed)
        worker.signals.finished.connect(self._export_finished)
        worker.signals.failed.connect(self._export_failed)
        self._set_busy(True)
        self._export_progress.setRange(0, max(1, len(self._all_variants)))
        self._export_progress.setValue(0)
        self._export_status_label.setText("正在准备导出…")
        self._thread_pool.start(worker)

    def _export_progress_changed(self, message: str, current: int, total: int) -> None:
        if self._shutdown:
            return
        self._export_progress.setRange(0, max(1, total))
        self._export_progress.setValue(max(0, min(current, max(1, total))))
        self._export_status_label.setText(message)

    def _export_finished(self, result: object) -> None:
        self._active_worker = None
        self._set_busy(False)
        if self._shutdown:
            return
        data = result if isinstance(result, dict) else {}
        cancelled = bool(data.get("已取消", data.get("取消", False)))
        if cancelled:
            self._export_status_label.setText("导出已取消，未留下部分结果")
            self.status_message.emit("导出已取消")
            return
        success = self._result_int(data, "成功", "导出")
        skipped = self._result_int(data, "跳过")
        failed = self._result_int(data, "失败")
        overwritten = self._result_int(data, "覆盖")
        if failed:
            self._export_status_label.setText(
                f"导出失败：成功 {success}，跳过 {skipped}，失败 {failed}"
            )
            self.status_message.emit("导出失败，未留下部分结果")
            message = (
                "本次导出批次未完成，程序没有保留部分结果。"
                f"\n成功 {success} 个，跳过 {skipped} 个，失败 {failed} 个。"
            )
            details_text = self._format_failure_details(data.get("失败详情"))
            if details_text:
                message += f"\n\n失败详情：\n{details_text}"
            QMessageBox.critical(self, "导出失败", message)
            return
        overwrite_text = f"（覆盖 {overwritten}）" if overwritten else ""
        self._export_status_label.setText(
            f"导出完成：成功 {success}{overwrite_text}，"
            f"跳过 {skipped}，失败 {failed}"
        )
        self.status_message.emit(f"已导出 {success} 个字形")
        message = f"已导出 {success} 个文件。"
        if overwritten:
            message += f"\n其中覆盖已有文件 {overwritten} 个。"
        if skipped or failed:
            message += f"\n跳过 {skipped} 个，失败 {failed} 个。"
        QMessageBox.information(self, "导出完成", message)

    def _export_failed(self, message: str) -> None:
        self._active_worker = None
        self._set_busy(False)
        if self._shutdown:
            return
        self._export_status_label.setText("导出失败")
        QMessageBox.critical(self, "导出失败", message)

    def cancel_export(self) -> None:
        if self._active_worker is None:
            return
        self._cancel_event.set()
        self._cancel_button.setEnabled(False)
        self._export_status_label.setText("正在取消导出…")

    def _set_busy(self, busy: bool) -> None:
        self._options_host.setEnabled(not busy)
        self._export_button.setEnabled(
            not busy
            and not self._audit_in_progress
            and bool(self._directory_edit.text().strip())
            and any(self._variant_ready.values())
        )
        self._cancel_button.setVisible(busy)
        self._cancel_button.setEnabled(busy)
        self._export_progress.setVisible(busy)
    def _refresh_export_button(self, _value: object = None) -> None:
        if self._active_worker is None:
            self._export_button.setEnabled(
                not self._audit_in_progress
                and bool(self._directory_edit.text().strip())
                and any(self._variant_ready.values())
            )
        self._export_status_label.setText(
            "准备就绪" if self._directory_edit.text().strip() else "请选择导出目录"
        )
    def _start_readiness_audit(self) -> None:
        self._audit_cancel_event.set()
        cancel_event = threading.Event()
        self._audit_cancel_event = cancel_event
        self._audit_in_progress = True
        self._show_audit_pending()
        service = ExportService(self._glyph)
        worker = FunctionWorker(
            lambda: service.audit_readiness(
                verify_hash=True,
                cancel_check=cancel_event.is_set,
            )
        )
        self._audit_workers.add(worker)
        worker.signals.finished.connect(
            lambda result, task=worker, token=cancel_event: self._audit_finished(
                result,
                task,
                token,
            )
        )
        worker.signals.failed.connect(
            lambda message, task=worker, token=cancel_event: self._audit_failed(
                message,
                task,
                token,
            )
        )
        self._thread_pool.start(worker)

    def _audit_finished(
        self,
        result: object,
        worker: FunctionWorker | None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        if worker is not None:
            self._release_audit_worker(worker)
        if (
            self._shutdown
            or not isinstance(result, dict)
            or (cancel_event is not None and cancel_event is not self._audit_cancel_event)
        ):
            return
        self._audit_in_progress = False
        if bool(result.get("已取消", False)):
            self._refresh_export_button()
            return
        self._audit = dict(result)
        issue_details = result.get("问题详情", [])
        file_issue_types = {
            "状态异常",
            "成品缺失",
            "成品损坏",
            "路径无效",
            "校验不符",
        }
        self._audit_issue_ids = {
            str(item.get("变体ID", ""))
            for item in issue_details
            if (
                isinstance(item, dict)
                and item.get("变体ID")
                and str(item.get("类型", "")) in file_issue_types
            )
        } if isinstance(issue_details, list) else set()
        self._workflow_status_cache.clear()
        for detail in self._all_variants:
            self._stage_projection(detail)
        self._phase_variants = [
            detail
            for detail in self._all_variants
            if self._stage_projection(detail).admitted
        ]
        verified = {
            variant_id: ready and variant_id not in self._audit_issue_ids
            for variant_id, ready in self._variant_ready.items()
        }
        if verified != self._variant_ready:
            self._variant_ready = verified
        self._apply_filters()
        self._apply_audit(self._audit)
        self._refresh_export_button()

    def _audit_failed(
        self,
        message: str,
        worker: FunctionWorker,
        cancel_event: threading.Event,
    ) -> None:
        self._release_audit_worker(worker)
        if self._shutdown or cancel_event is not self._audit_cancel_event:
            return
        self._audit_in_progress = False
        self._audit = {
            "就绪": False,
            "总数": len(self._all_variants),
            "已就绪": sum(self._variant_ready.values()),
            "原因": [f"完整性核对失败：{message}"],
        }
        self._apply_audit(self._audit)
        self._refresh_export_button()

    def _show_audit_pending(self) -> None:
        self._readiness_dot.setStyleSheet(
            "background: #D6A84B; border-radius: 5px;"
        )
        self._readiness_badge.setStyleSheet(
            "QFrame#exportReadinessBadge { background: #3D3524; "
            "border: 1px solid #806B38; border-radius: 6px; }"
        )
        self._readiness_label.setText("正在核对全库状态")
        self._readiness_badge.setToolTip(
            "正在逐项核对成品文件、图像内容和文件校验值。"
        )
        self._refresh_export_button()

    def _release_audit_worker(self, worker: FunctionWorker) -> None:
        self._audit_workers.discard(worker)

    def _fast_audit(self) -> dict[str, Any]:
        total = len(self._all_variants)
        ready_count = sum(self._variant_ready.values())
        summary = self._glyph.get_coordination_summary()
        geometry_completed = bool(summary.get("几何协调完成"))
        ink_enabled = bool(summary.get("墨色统一启用", True))
        ink_completed = not ink_enabled or bool(summary.get("墨色统一完成"))
        ready = (
            total > 0
            and ready_count == total
            and geometry_completed
            and ink_completed
        )
        reasons: list[str] = []
        if total == 0:
            reasons.append("字库中没有字形")
        if ready_count < total:
            reasons.append(f"还有 {total - ready_count} 个字形未生成有效成品")
        if not geometry_completed:
            reasons.append("整体协调尚未完成")
        if not ink_completed:
            reasons.append("全库墨色协调尚未完成")
        return {
            "就绪": ready,
            "总数": total,
            "已就绪": ready_count,
            "原因": reasons,
        }

    def _apply_audit(self, audit: dict[str, Any]) -> None:
        fallback = self._fast_audit() if self._all_variants else {
            "就绪": False,
            "总数": 0,
            "已就绪": 0,
            "原因": ["字库中没有字形"],
        }
        ready = bool(audit.get("就绪", audit.get("ready", fallback["就绪"])))
        total = self._audit_int(audit, "总数", "total", default=fallback["总数"])
        ready_count = self._audit_int(
            audit,
            "已就绪",
            "ready_count",
            default=fallback["已就绪"],
        )
        reasons_value = audit.get(
            "原因",
            audit.get("摘要原因", audit.get("reasons", fallback["原因"])),
        )
        if isinstance(reasons_value, (list, tuple)):
            reasons = [str(item) for item in reasons_value if str(item)]
        elif reasons_value:
            reasons = [str(reasons_value)]
        else:
            reasons = []

        color = "#48C78E" if ready else "#E36A6A"
        background = "#203B32" if ready else "#40272C"
        border = "#397D61" if ready else "#84454E"
        self._readiness_dot.setStyleSheet(
            f"background: {color}; border-radius: 5px;"
        )
        self._readiness_badge.setStyleSheet(
            f"QFrame#exportReadinessBadge {{ background: {background}; "
            f"border: 1px solid {border}; border-radius: 6px; }}"
        )
        self._readiness_label.setText(
            "全库可导出" if ready else "全库尚不可导出"
        )
        tooltip = (
            f"{ready_count} / {total} 个字形已完成审核与整体协调"
        )
        if reasons:
            tooltip += "\n" + "\n".join(reasons)
        self._readiness_badge.setToolTip(tooltip)

        local_ready = sum(self._variant_ready.values())
        local_total = len(self._all_variants)
        projections = [
            self._stage_projection(detail) for detail in self._phase_variants
        ]
        pending = sum(
            projection.status == STAGE_PENDING_COORDINATION
            for projection in projections
        )
        coordinated = sum(
            projection.status == STATUS_COORDINATED
            for projection in projections
        )
        self._summary_label.setText(
            f"待协调 {pending}　已协调 {coordinated}\n"
            f"可导出 {local_ready} / {local_total}"
        )
        self._progress_bar.setValue(
            round(local_ready * 100 / local_total) if local_total else 0
        )
        self._refresh_export_button()

    def _confirm_partial_export(self, ready_count: int, total: int) -> bool:
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle("字库尚未全部完成")
        dialog.setText(
            f"当前字库共 {total} 个字形，其中 {ready_count} 个已有可导出的最终成品。"
        )
        dialog.setInformativeText(
            "继续后只导出完整性核对通过的已有成品；尚未生成、文件损坏或"
            "校验不符的字形会被跳过。全库整体协调未完成时，本次结果仅适合"
            "阶段性检查。"
        )
        continue_button = dialog.addButton(
            "继续导出",
            QMessageBox.ButtonRole.AcceptRole,
        )
        cancel_button = dialog.addButton(
            "取消",
            QMessageBox.ButtonRole.RejectRole,
        )
        dialog.setDefaultButton(cancel_button)
        dialog.setEscapeButton(cancel_button)
        dialog.exec()
        return dialog.clickedButton() is continue_button

    def _resolve_export_conflicts(
        self,
        conflicts: list[ExportConflict],
    ) -> tuple[ExportConflictDecision, ...] | None:
        """在主线程逐项取得决定；勾选后把同一动作应用到剩余冲突。"""
        decisions: list[ExportConflictDecision] = []
        apply_to_all_action: str | None = None
        total = len(conflicts)
        for index, conflict in enumerate(conflicts, start=1):
            if apply_to_all_action is None:
                response = self._ask_export_conflict(conflict, index, total)
                if response is None:
                    return None
                action, apply_to_all = response
                if apply_to_all:
                    apply_to_all_action = action
            else:
                action = apply_to_all_action
            decisions.append(ExportConflictDecision(conflict, action))
        return tuple(decisions)

    def _ask_export_conflict(
        self,
        conflict: ExportConflict,
        index: int,
        total: int,
    ) -> tuple[str, bool] | None:
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle("发现同名文件")
        dialog.setText(
            "导出目录中已存在同名文件：\n"
            f"{conflict.destination_name}"
        )
        char_text = conflict.char or "未知字形"
        dialog.setInformativeText(
            f"字形：{char_text}\n冲突 {index} / {total}\n"
            "请选择如何处理该文件。"
        )
        apply_to_all = QCheckBox("为所有项目执行此操作", dialog)
        dialog.setCheckBox(apply_to_all)
        overwrite_button = dialog.addButton(
            "覆盖",
            QMessageBox.ButtonRole.DestructiveRole,
        )
        skip_button = dialog.addButton(
            "跳过",
            QMessageBox.ButtonRole.ActionRole,
        )
        cancel_button = dialog.addButton(
            "取消",
            QMessageBox.ButtonRole.RejectRole,
        )
        dialog.setDefaultButton(cancel_button)
        dialog.setEscapeButton(cancel_button)
        dialog.exec()
        clicked = dialog.clickedButton()
        if clicked is overwrite_button:
            return ExportService.CONFLICT_OVERWRITE, apply_to_all.isChecked()
        if clicked is skip_button:
            return ExportService.CONFLICT_SKIP, apply_to_all.isChecked()
        return None

    def request_back(self) -> None:
        if self._active_worker is not None:
            QMessageBox.information(
                self,
                "正在导出",
                "请先取消导出并等待任务结束，再返回首页。",
            )
            return
        if self._on_back is not None:
            self._on_back()
        else:
            self.home_requested.emit()

    @property
    def is_running(self) -> bool:
        """返回导出或完整审计任务是否尚未结束。"""

        return self._active_worker is not None or bool(self._audit_workers)

    def shutdown(self) -> None:
        """使后台结果失效，并请求正在运行的导出安全取消。"""
        self._shutdown = True
        self._thumbnail_generation += 1
        self._cancel_event.set()
        self._audit_cancel_event.set()
        self._thumbnail_timer.stop()
        self._thumbnail_pool.clear()
        self._thumbnail_workers.clear()
        self._thumbnail_cache.clear()
        self._thumbnail_failures.clear()
        self._gallery.shutdown()

    @staticmethod
    def _format_failure_details(value: object, limit: int = 5) -> str:
        if not isinstance(value, (list, tuple)):
            return ""
        lines: list[str] = []
        for item in value[:limit]:
            if isinstance(item, dict):
                identifier = str(
                    item.get("字形") or item.get("变体ID") or item.get("文件") or "字形"
                )
                reason = str(item.get("说明") or item.get("原因") or "未知错误")
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                identifier = str(item[0] or "字形")
                reason = str(item[1] or "未知错误")
            else:
                identifier = "字形"
                reason = str(item)
            reason = reason.replace("\r", " ").replace("\n", " ").strip()
            if len(reason) > 160:
                reason = reason[:157] + "..."
            lines.append(f"{identifier}：{reason}")
        if len(value) > limit:
            lines.append(f"另有 {len(value) - limit} 项失败未展开。")
        return "\n".join(lines)

    @staticmethod
    def _is_variant_ready(detail: dict[str, Any], finished_dir: str) -> bool:
        if str(detail.get("状态", "")) != config.STATUS_FINISHED:
            return False
        return bool(
            resolve_safe_stage_file(finished_dir, detail.get("成品文件", ""))
        )

    @staticmethod
    def _finished_image_path(detail: dict[str, Any], finished_dir: str) -> str:
        return resolve_safe_stage_file(finished_dir, detail.get("成品文件", ""))

    @classmethod
    def _placeholder_icon(cls) -> QIcon:
        """异步解码完成前显示中性占位，不用系统字体伪造字形。"""
        size = QSize(
            cls.LIST_THUMBNAIL_SIZE,
            cls.LIST_THUMBNAIL_SIZE,
        )
        thumbnail = QPixmap(size)
        thumbnail.fill(QColor("#ffffff"))
        painter = QPainter(thumbnail)
        painter.setPen(QColor("#c7cdd5"))
        painter.drawRect(thumbnail.rect().adjusted(1, 1, -2, -2))
        painter.end()
        return QIcon(thumbnail)

    @classmethod
    def _thumbnail_icon(cls, image: QImage) -> QIcon:
        """将透明成品合成到固定白底，避免树背景影响字形辨识。"""
        size = QSize(cls.LIST_THUMBNAIL_SIZE, cls.LIST_THUMBNAIL_SIZE)
        thumbnail = QPixmap(size)
        thumbnail.fill(QColor("#ffffff"))
        painter = QPainter(thumbnail)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        target = image.size().scaled(
            QSize(size.width() - 4, size.height() - 4),
            Qt.AspectRatioMode.KeepAspectRatio,
        )
        x = (size.width() - target.width()) // 2
        y = (size.height() - target.height()) // 2
        painter.drawImage(x, y, image.scaled(
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))
        painter.end()
        return QIcon(thumbnail)

    @staticmethod
    def _section_title(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet("font-weight: 700; color: #E8EDF5;")
        return label

    @staticmethod
    def _horizontal_separator() -> QFrame:
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("color: #37404D;")
        return separator

    @staticmethod
    def _positive_int(value: object, default: int) -> int:
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _audit_int(
        audit: dict[str, Any],
        *keys: str,
        default: int,
    ) -> int:
        for key in keys:
            if key not in audit:
                continue
            try:
                return max(0, int(audit[key]))
            except (TypeError, ValueError):
                continue
        return max(0, int(default))

    @staticmethod
    def _result_int(result: dict[str, Any], *keys: str) -> int:
        for key in keys:
            if key not in result:
                continue
            try:
                return max(0, int(result[key]))
            except (TypeError, ValueError):
                continue
        return 0
