"""新建字库与字库添加页面。"""

from __future__ import annotations

import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from PIL import Image
from PySide6.QtCore import (
    QPoint,
    QRect,
    QSize,
    QObject,
    QRunnable,
    QThreadPool,
    QTimer,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLayoutItem,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import config
from services.glyph_service import GlyphService
from services.settings_service import SettingsService
from services.import_service import ImportService
from services.traditional_chinese_service import identify_character
from utils.batch_observability import ProgressThrottle, format_elapsed_time
from utils.file_utils import (
    compute_file_md5,
    is_real_directory,
    is_safe_windows_filename,
    pinyin_natural_key,
    validate_final_char,
)
from ui.workers import log_background_exception


PREVIEW_SIZE = 72


def _decode_preview_image(path: str, width: int = PREVIEW_SIZE, height: int = PREVIEW_SIZE) -> QImage:
    """在工作线程解码小尺寸预览，返回拥有独立像素内存的 QImage。"""
    with Image.open(path) as source:
        source.seek(0)
        source.thumbnail((width, height), Image.Resampling.LANCZOS)
        image = source.convert("RGBA")
        pixels = image.tobytes("raw", "RGBA")
        return QImage(
            pixels,
            image.width,
            image.height,
            image.width * 4,
            QImage.Format.Format_RGBA8888,
        ).copy()


class FlowLayout(QLayout):
    """优先横向排列控件，空间不足时自动换行。"""

    def __init__(self, parent: QWidget | None = None, spacing: int = 8) -> None:
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self._spacing = spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item: QLayoutItem) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QLayoutItem | None:
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> QLayoutItem | None:
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._arrange(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self._arrange(rect, test_only=False)

    def sizeHint(self) -> QSize:
        margins = self.contentsMargins()
        widths = [item.sizeHint().width() for item in self._items]
        heights = [item.sizeHint().height() for item in self._items]
        spacing_width = self._spacing * max(0, len(self._items) - 1)
        return QSize(
            sum(widths) + spacing_width + margins.left() + margins.right(),
            (max(heights) if heights else 0) + margins.top() + margins.bottom(),
        )

    def minimumSize(self) -> QSize:
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(margins.left() + margins.right(), margins.top() + margins.bottom())

    def _arrange(self, rect: QRect, test_only: bool) -> int:
        margins = self.contentsMargins()
        available = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        x = available.x()
        y = available.y()
        line_height = 0
        for item in self._items:
            item_size = item.sizeHint()
            next_x = x + item_size.width()
            if line_height and next_x > available.right() + 1:
                x = available.x()
                y += line_height + self._spacing
                next_x = x + item_size.width()
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), item_size))
            x = next_x + self._spacing
            line_height = max(line_height, item_size.height())
        return y + line_height - rect.y() + margins.bottom()


@dataclass
class ScanItem:
    """一张待导入图片的扫描与人工确认状态。"""

    path: str
    filename: str
    original_char: str
    category: str
    candidates: tuple[str, ...]
    final_char: str
    confirmed: bool
    issue: str = ""
    duplicate_path: str = ""
    duplicate_filename: str = ""
    preview_image: QImage | None = None
    duplicate_preview_image: QImage | None = None
    digest: str = ""
    source_size: int = 0
    source_mtime_ns: int = 0


@dataclass(frozen=True)
class ImportRunContext:
    """一次导入任务的不可变身份及其专属取消、清理状态。"""

    token: int
    append_mode: bool
    library_name: str
    target_directory: str
    cancel_event: threading.Event
    directory_created: threading.Event
    started_at: float = field(default_factory=time.perf_counter)


class TaskSignals(QObject):
    """导入页面后台任务信号。"""

    progress = Signal(int, int, str)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()


class ScanTask(QRunnable):
    """后台扫描目录并识别字符、重复文件和异常图片。"""

    def __init__(
        self,
        paths: list[str],
        existing_files: dict[str, tuple[str, str]],
        cancel_event: threading.Event,
    ) -> None:
        super().__init__()
        self.paths = paths
        self.existing_files = existing_files
        self.cancel_event = cancel_event
        self.signals = TaskSignals()

    @Slot()
    def run(self) -> None:
        items: list[ScanItem] = []
        extractor = ImportService.__new__(ImportService)
        try:
            total = len(self.paths)
            seen_files: dict[str, tuple[str, str]] = {}
            seen_previews: dict[str, QImage] = {}
            progress = ProgressThrottle(self.signals.progress.emit)
            for index, path in enumerate(self.paths, 1):
                if self.cancel_event.is_set():
                    self.signals.cancelled.emit()
                    return
                filename = os.path.basename(path)
                original_char = extractor._extract_char(filename)
                if original_char == "未分类":
                    category = "正确"
                    candidates = (original_char,)
                else:
                    category, candidates = identify_character(original_char)
                if category in {"一对一", "歧义"}:
                    candidates = tuple(dict.fromkeys((*candidates, original_char)))
                    final_char = ""
                else:
                    final_char = original_char
                issue = ""
                duplicate_path = ""
                duplicate_filename = ""
                preview_image = QImage()
                duplicate_preview_image = QImage()
                digest = ""
                source_size = 0
                source_mtime_ns = 0
                try:
                    digest = compute_file_md5(path)
                    source_stat = os.stat(path, follow_symlinks=False)
                    source_size = source_stat.st_size
                    source_mtime_ns = source_stat.st_mtime_ns
                    if digest in self.existing_files:
                        duplicate_path, duplicate_filename = self.existing_files[digest]
                        issue = "重复，将跳过！"
                        category = "重复"
                    elif digest in seen_files:
                        duplicate_path, duplicate_filename = seen_files[digest]
                        issue = "重复，将跳过！"
                        category = "重复"
                    else:
                        seen_files[digest] = (os.path.abspath(path), filename)
                    preview_image = _decode_preview_image(path)
                    if duplicate_path:
                        duplicate_preview_image = seen_previews.get(digest, QImage())
                        if duplicate_preview_image.isNull():
                            try:
                                duplicate_preview_image = _decode_preview_image(duplicate_path)
                            except (OSError, ValueError):
                                duplicate_preview_image = QImage()
                    else:
                        seen_previews[digest] = preview_image
                except (OSError, ValueError) as exc:
                    issue = f"图片异常：{exc}"
                    category = "异常"
                if original_char == "未分类" and category != "重复":
                    issue = f"{issue}；" if issue else ""
                    issue += "文件名中未识别到汉字"
                    category = "异常"
                    final_char = ""
                items.append(
                    ScanItem(
                        path=os.path.abspath(path),
                        filename=filename,
                        original_char=original_char,
                        category=category,
                        candidates=candidates,
                        final_char=final_char,
                        confirmed=category not in {"歧义", "异常"},
                        issue=issue,
                        duplicate_path=duplicate_path,
                        duplicate_filename=duplicate_filename,
                        preview_image=preview_image,
                        duplicate_preview_image=duplicate_preview_image,
                        digest=digest,
                        source_size=source_size,
                        source_mtime_ns=source_mtime_ns,
                    )
                )
                progress.emit(
                    index,
                    total,
                    filename,
                    force=index in {1, total},
                    stage="扫描图片",
                )
            progress.flush()
            self.signals.finished.emit(items)
        except Exception as exc:
            log_background_exception("导入目录扫描")
            self.signals.failed.emit(str(exc))


class ImportTask(QRunnable):
    """后台执行字库创建或追加导入。"""

    def __init__(self, function) -> None:
        super().__init__()
        self.function = function
        self.signals = TaskSignals()

    @Slot()
    def run(self) -> None:
        try:
            self.signals.finished.emit(self.function(self.signals.progress))
        except Exception as exc:
            log_background_exception("字库导入")
            self.signals.failed.emit(str(exc))


class ImportPage(QWidget):
    """在同一页面完成字库规格设置、图片核对与导入。"""

    POPULATION_BATCH_LIMIT = 12
    POPULATION_TIME_SLICE_SECONDS = 0.012
    POPULATION_LAYOUT_SETTLE_MS = 30

    home_requested = Signal()
    import_completed = Signal(str, str, object)
    status_message = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        settings_service = SettingsService()
        try:
            self._application_settings = settings_service.load()
        except (OSError, RuntimeError, ValueError):
            self._application_settings = settings_service.defaults()
        self._append_mode = False
        self._glyph_service: GlyphService | None = None
        self._existing_names: list[str] = []
        self._scan_items: list[ScanItem] = []
        self._column_layouts: dict[str, QVBoxLayout] = {}
        self._thread_pool = QThreadPool.globalInstance()
        self._workers: set[QRunnable] = set()
        self._cancel_event = threading.Event()
        self._scan_generation = 0
        self._import_generation = 0
        self._active_import_token: int | None = None
        self._active_task_kind: str | None = None
        self._return_home_after_import = False
        self._conversion_lock = False
        self._population_timer = QTimer(self)
        self._population_timer.setSingleShot(True)
        self._population_timer.timeout.connect(self._populate_table_batch)
        self._population_active = False
        self._population_prepared = False
        self._population_finishing = False
        self._population_queue: list[tuple[ScanItem, str]] = []
        self._population_index = 0
        self._population_completion_status = ""
        self._population_clear_scan_on_cancel = False
        self._build_ui()
        self._bind_conversions()
        self.configure_create()

    @property
    def is_running(self) -> bool:
        """返回扫描、建表或导入任务是否尚未结束。"""

        return self._active_task_kind is not None or self._population_active

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(22, 18, 22, 20)
        root.setSpacing(12)
        header = QHBoxLayout()
        title_box = QHBoxLayout()
        self._title_label = QLabel("新建字库")
        self._title_label.setProperty("role", "pageTitle")
        self._subtitle_label = QLabel("填写规格、扫描核对文字图片并创建字库")
        self._subtitle_label.setProperty("role", "muted")
        title_box.addWidget(self._title_label)
        title_box.addSpacing(12)
        title_box.addWidget(self._subtitle_label)
        title_box.addStretch()
        header.addLayout(title_box, 1)
        home_button = QPushButton("返回首页")
        home_button.clicked.connect(self._request_home)
        header.addWidget(home_button, 0, Qt.AlignmentFlag.AlignTop)
        root.addLayout(header)

        info_panel = QFrame()
        info_panel.setProperty("role", "card")
        info_layout = QHBoxLayout(info_panel)
        info_layout.setContentsMargins(16, 12, 16, 12)
        info_layout.setSpacing(28)

        left_form = QFormLayout()
        left_form.setHorizontalSpacing(10)
        left_form.setVerticalSpacing(9)
        left_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("请输入新字库名称")
        left_form.addRow("字库名称", self._name_edit)

        directory_widget = QWidget()
        directory_row = QHBoxLayout(directory_widget)
        directory_row.setContentsMargins(0, 0, 0, 0)
        directory_row.setSpacing(8)
        self._directory_edit = QLineEdit()
        self._directory_edit.setReadOnly(True)
        self._directory_edit.setPlaceholderText("请选择包含文字图片的文件夹")
        choose_button = QPushButton("选择目录")
        choose_button.clicked.connect(self._choose_directory)
        directory_row.addWidget(self._directory_edit, 1)
        directory_row.addWidget(choose_button)
        left_form.addRow("图片目录", directory_widget)

        info_layout.addLayout(left_form, 11)

        right_form = QFormLayout()
        right_form.setHorizontalSpacing(10)
        right_form.setVerticalSpacing(9)
        right_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._dpi_spin = QSpinBox()
        self._dpi_spin.setRange(1, 9600)
        self._dpi_spin.setSuffix(" DPI")
        self._dpi_spin.setValue(self._application_settings.default_dpi)
        right_form.addRow("分辨率", self._dpi_spin)

        size_widget = QWidget()
        size_row = QHBoxLayout(size_widget)
        size_row.setContentsMargins(0, 0, 0, 0)
        size_row.setSpacing(8)
        self._width_px_spin = QSpinBox()
        self._width_px_spin.setRange(1, 50000)
        self._width_px_spin.setSuffix(" 像素")
        self._width_px_spin.setValue(
            self._application_settings.default_canvas_width
        )
        self._width_mm_spin = QDoubleSpinBox()
        self._width_mm_spin.setRange(0.01, 10000)
        self._width_mm_spin.setDecimals(2)
        self._width_mm_spin.setSuffix(" 毫米")
        self._height_px_spin = QSpinBox()
        self._height_px_spin.setRange(1, 50000)
        self._height_px_spin.setSuffix(" 像素")
        self._height_px_spin.setValue(
            self._application_settings.default_canvas_height
        )
        self._height_mm_spin = QDoubleSpinBox()
        self._height_mm_spin.setRange(0.01, 10000)
        self._height_mm_spin.setDecimals(2)
        self._height_mm_spin.setSuffix(" 毫米")
        size_row.addWidget(QLabel("宽"))
        size_row.addWidget(self._width_px_spin, 1)
        size_row.addWidget(self._width_mm_spin, 1)
        size_row.addSpacing(4)
        size_row.addWidget(QLabel("高"))
        size_row.addWidget(self._height_px_spin, 1)
        size_row.addWidget(self._height_mm_spin, 1)
        right_form.addRow("成品尺寸", size_widget)
        info_layout.addLayout(right_form, 9)

        root.addWidget(info_panel)

        verify_header = QHBoxLayout()
        verify_title = QLabel("图片字符校对")
        verify_title.setProperty("role", "sectionTitle")
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("输入字符或文件名")
        self._search_edit.setClearButtonEnabled(True)
        self._search_edit.setMinimumWidth(180)
        self._search_edit.setMaximumWidth(280)
        self._search_edit.returnPressed.connect(self._populate_tables)
        self._search_button = QPushButton("搜索")
        self._search_button.clicked.connect(self._populate_tables)
        self._status_label = QLabel("请选择文字图片目录")
        self._status_label.setProperty("role", "muted")
        self._scan_button = QPushButton("扫描并核对")
        self._scan_button.setProperty("role", "primary")
        self._scan_button.clicked.connect(self.start_scan)
        verify_header.addWidget(verify_title)
        verify_header.addWidget(self._search_edit)
        verify_header.addWidget(self._search_button)
        verify_header.addStretch()
        verify_header.addWidget(self._status_label)
        verify_header.addWidget(self._scan_button)
        root.addLayout(verify_header)

        columns = QHBoxLayout()
        columns.setSpacing(10)
        self._column_counts: dict[str, QLabel] = {}
        for category, title, description, color in (
            ("正确", "正常：请核对", "核对并可修改最终字符", "#51cf66"),
            ("一对一", "有歧义：请选择", "选择文件名对应的简体字或繁体字", "#f59f00"),
            ("异常", "重复或异常", "重复文件将跳过；异常图片需处理", "#ff6b6b"),
        ):
            panel = QFrame()
            panel.setProperty("role", "card")
            panel_layout = QVBoxLayout(panel)
            panel_layout.setContentsMargins(8, 8, 8, 8)
            panel_layout.setSpacing(6)
            heading = QHBoxLayout()
            title_label = QLabel(title)
            title_label.setStyleSheet(f"font-size: 16px; font-weight: 700; color: {color};")
            count_label = QLabel("0 项")
            count_label.setProperty("role", "muted")
            self._column_counts[category] = count_label
            heading.addWidget(title_label)
            heading.addStretch()
            heading.addWidget(count_label)
            panel_layout.addLayout(heading)
            description_label = QLabel(description)
            description_label.setProperty("role", "muted")
            panel_layout.addWidget(description_label)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            content = QWidget()
            content_layout = QVBoxLayout(content)
            content_layout.setContentsMargins(0, 0, 0, 0)
            content_layout.setSpacing(7)
            content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
            content_layout.addStretch()
            self._column_layouts[category] = content_layout
            scroll.setWidget(content)
            panel_layout.addWidget(scroll, 1)
            columns.addWidget(panel, 1)
        root.addLayout(columns, 1)

        progress_row = QHBoxLayout()
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        self._cancel_button = QPushButton("取消任务")
        self._cancel_button.clicked.connect(self.cancel_task)
        self._cancel_button.setVisible(False)
        progress_row.addWidget(self._progress_bar, 1)
        progress_row.addWidget(self._cancel_button)
        root.addLayout(progress_row)

        footer = QHBoxLayout()
        self._hint_label = QLabel("源目录和源文件名不会被修改")
        self._hint_label.setProperty("role", "muted")
        footer.addWidget(self._hint_label)
        footer.addStretch()
        self._confirm_check = QCheckBox("已核对重复或异常项")
        self._confirm_check.setVisible(False)
        self._confirm_check.toggled.connect(self._validate_confirmations)
        footer.addWidget(self._confirm_check)
        self._import_button = QPushButton("确认并创建字库")
        self._import_button.setProperty("role", "primary")
        self._import_button.clicked.connect(self.start_import)
        self._import_button.setEnabled(False)
        footer.addWidget(self._import_button)
        root.addLayout(footer)

    def configure_create(self, existing_names: list[str] | None = None) -> None:
        """切换为新建模式，并可传入已有字库名称用于重名校验。"""
        self._append_mode = False
        self._glyph_service = None
        self._existing_names = list(existing_names or [])
        self._title_label.setText("新建字库")
        self._subtitle_label.setText("填写规格、扫描核对文字图片并创建字库")
        self._name_edit.clear()
        self._name_edit.setReadOnly(False)
        self._dpi_spin.setValue(self._application_settings.default_dpi)
        self._width_px_spin.setValue(
            self._application_settings.default_canvas_width
        )
        self._height_px_spin.setValue(
            self._application_settings.default_canvas_height
        )
        self._pixels_to_millimeters()
        self._set_spec_enabled(True)
        self._import_button.setText("确认并创建字库")
        self._reset_scan()
        self._directory_edit.setText(
            SettingsService.usable_directory(
                self._application_settings.default_image_directory
            )
        )

    def configure_append(self, glyph_service: GlyphService, existing_names: list[str] | None = None) -> None:
        """切换为追加模式，读取并锁定现有字库规格。"""
        self._append_mode = True
        self._glyph_service = glyph_service
        self._existing_names = list(existing_names or [])
        metadata = glyph_service.get_metadata()
        self._title_label.setText("字库添加")
        self._subtitle_label.setText("向当前字库继续添加文字图片")
        self._name_edit.setText(glyph_service.ziku_name)
        self._name_edit.setReadOnly(True)
        self._dpi_spin.setValue(int(metadata.get("DPI", metadata.get("分辨率", 300))))
        self._width_px_spin.setValue(int(metadata.get("画布宽", 250)))
        self._height_px_spin.setValue(int(metadata.get("画布高", 250)))
        self._width_mm_spin.setValue(round(float(metadata.get("成品宽度毫米", 21.17)), 2))
        self._height_mm_spin.setValue(round(float(metadata.get("成品高度毫米", 21.17)), 2))
        self._set_spec_enabled(False)
        self._import_button.setText("确认并导入字图")
        self._reset_scan()
        self._directory_edit.setText(
            SettingsService.usable_directory(
                self._application_settings.default_image_directory
            )
        )

    def _bind_conversions(self) -> None:
        self._dpi_spin.valueChanged.connect(self._pixels_to_millimeters)
        self._width_px_spin.valueChanged.connect(self._pixels_to_millimeters)
        self._height_px_spin.valueChanged.connect(self._pixels_to_millimeters)
        self._width_mm_spin.valueChanged.connect(
            lambda _value: self._millimeters_to_pixels("宽")
        )
        self._height_mm_spin.valueChanged.connect(
            lambda _value: self._millimeters_to_pixels("高")
        )
        self._pixels_to_millimeters()

    def _pixels_to_millimeters(self, _value: int | None = None) -> None:
        if self._conversion_lock:
            return
        self._conversion_lock = True
        try:
            dpi = self._dpi_spin.value()
            self._width_mm_spin.setValue(
                round(self._width_px_spin.value() / dpi * 25.4, 2)
            )
            self._height_mm_spin.setValue(
                round(self._height_px_spin.value() / dpi * 25.4, 2)
            )
        finally:
            self._conversion_lock = False

    def _millimeters_to_pixels(self, dimension: str) -> None:
        if self._conversion_lock:
            return
        self._conversion_lock = True
        try:
            dpi = self._dpi_spin.value()
            if dimension == "宽":
                self._width_px_spin.setValue(
                    round(self._width_mm_spin.value() / 25.4 * dpi)
                )
            else:
                self._height_px_spin.setValue(
                    round(self._height_mm_spin.value() / 25.4 * dpi)
                )
        finally:
            self._conversion_lock = False

    def _set_spec_enabled(self, enabled: bool) -> None:
        for widget in (self._dpi_spin, self._width_px_spin, self._height_px_spin, self._width_mm_spin, self._height_mm_spin):
            widget.setEnabled(enabled)

    def _choose_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "选择文字图片目录", self._directory_edit.text())
        if not directory:
            return
        self._directory_edit.setText(directory)
        if not self._append_mode and not self._name_edit.text().strip():
            self._name_edit.setText(os.path.basename(directory.rstrip("/\\")))
        self.start_scan()

    def start_scan(self) -> None:
        """扫描当前目录；结果通过三个分类栏供用户核对。"""
        directory = self._directory_edit.text().strip()
        if not is_real_directory(directory):
            QMessageBox.warning(self, "提示", "请选择真实且有效的文字图片目录。")
            return
        paths = [
            entry.path for entry in sorted(os.scandir(directory), key=lambda value: pinyin_natural_key(value.name))
            if entry.is_file(follow_symlinks=False)
            and os.path.splitext(entry.name)[1].lower() in ImportService.SUPPORTED_EXTENSIONS
        ]
        if not paths:
            QMessageBox.warning(self, "提示", "所选目录中没有支持的图片文件。")
            return
        self._clear_scan_results()
        self._search_edit.clear()
        self._scan_generation += 1
        generation = self._scan_generation
        existing_files: dict[str, tuple[str, str]] = {}
        if self._append_mode and self._glyph_service:
            original_dir = self._glyph_service.get_workflow_dirs()["原图"]
            for detail in self._glyph_service.get_all_variants():
                digest = str(detail.get("原始MD5", ""))
                filename = str(detail.get("原始文件", ""))
                if digest:
                    existing_files[digest] = (os.path.join(original_dir, filename), filename)
        self._cancel_event = threading.Event()
        worker = ScanTask(paths, existing_files, self._cancel_event)
        self._workers.add(worker)
        worker.signals.progress.connect(
            lambda current, total, message, token=generation: self._scan_progress(
                token, current, total, message
            )
        )
        worker.signals.finished.connect(
            lambda result, token=generation: self._scan_finished(result, token)
        )
        worker.signals.failed.connect(
            lambda message, token=generation: self._scan_task_failed(token, message)
        )
        worker.signals.cancelled.connect(
            lambda token=generation: self._scan_task_cancelled(token)
        )
        worker.signals.finished.connect(lambda _result, task=worker: self._release_worker(task))
        worker.signals.failed.connect(lambda _message, task=worker: self._release_worker(task))
        worker.signals.cancelled.connect(lambda task=worker: self._release_worker(task))
        self._active_task_kind = "scan"
        self._set_busy(True, len(paths), "正在扫描图片")
        self._thread_pool.start(worker)

    def _scan_progress(self, generation: int, current: int, total: int, message: str) -> None:
        if generation == self._scan_generation:
            self._update_progress(current, total, message)

    def _scan_finished(self, result: object, generation: int | None = None) -> None:
        if generation is not None and generation != self._scan_generation:
            return
        self._scan_items = list(result) if isinstance(result, list) else []
        self._start_table_population(
            "正在生成校对列表",
            f"已扫描 {len(self._scan_items)} 张图片",
            clear_scan_on_cancel=True,
        )

    def _scan_task_failed(self, generation: int, message: str) -> None:
        if generation == self._scan_generation:
            self._active_task_kind = None
            self._task_failed(message)

    def _scan_task_cancelled(self, generation: int) -> None:
        if generation == self._scan_generation:
            self._active_task_kind = None
            self._task_cancelled()

    def _populate_tables(self, *_args: object) -> None:
        """按当前搜索条件分批刷新三栏，避免大字库阻塞主线程。"""
        if self._population_active:
            return
        self._start_table_population(
            "正在刷新校对列表",
            f"已扫描 {len(self._scan_items)} 张图片",
            clear_scan_on_cancel=False,
        )

    def _start_table_population(
        self,
        busy_message: str,
        completion_status: str,
        *,
        clear_scan_on_cancel: bool,
    ) -> None:
        """启动分批建卡；首个定时器触发前先把第二阶段进度绘制出来。"""
        self._stop_table_population()
        self._population_active = True
        self._population_prepared = False
        self._population_finishing = False
        self._population_queue = []
        self._population_index = 0
        self._population_completion_status = completion_status
        self._population_clear_scan_on_cancel = clear_scan_on_cancel
        self._active_task_kind = "population"
        total = max(1, len(self._scan_items))
        self._set_busy(True, total, busy_message)
        self._progress_bar.setFormat(f"{busy_message} %v / %m")
        self._population_timer.start(0)

    def _prepare_table_population(self) -> None:
        """排序、筛选并轮转三栏队列，让每一栏都尽快显示首屏。"""
        grouped: dict[str, list[ScanItem]] = {"正确": [], "一对一": [], "异常": []}
        for item in self._scan_items:
            if item.category == "正确":
                target = "正确"
            elif item.category in {"一对一", "歧义"}:
                target = "一对一"
            else:
                target = "异常"
            grouped[target].append(item)
        for category_items in grouped.values():
            category_items.sort(
                key=lambda item: (
                    pinyin_natural_key(item.final_char or item.original_char or item.filename),
                    pinyin_natural_key(item.filename),
                )
            )
        exception_count = len(grouped["异常"])
        self._confirm_check.setVisible(exception_count > 0)
        if exception_count == 0:
            self._confirm_check.setChecked(False)
        keyword = self._search_edit.text().strip().lower()
        visible_groups: dict[str, list[ScanItem]] = {}
        self._clear_table_widgets()
        for category in self._column_layouts:
            category_items = grouped[category]
            self._column_counts[category].setText(f"{len(category_items)} 项")
            visible_groups[category] = [
                item
                for item in category_items
                if not keyword
                or keyword in item.filename.lower()
                or keyword in item.original_char.lower()
                or keyword in item.final_char.lower()
            ]

        queue: list[tuple[ScanItem, str]] = []
        longest = max((len(items) for items in visible_groups.values()), default=0)
        for row_index in range(longest):
            for category in self._column_layouts:
                category_items = visible_groups[category]
                if row_index < len(category_items):
                    queue.append((category_items[row_index], category))
        self._population_queue = queue
        self._population_prepared = True
        total = len(queue)
        self._progress_bar.setRange(0, max(1, total))
        self._progress_bar.setValue(0 if total else 1)
        self._progress_bar.setFormat("正在生成校对列表 %v / %m")
        if keyword:
            self._population_completion_status = (
                f"已显示 {total} 项搜索结果（共 {len(self._scan_items)} 张图片）"
            )

    def _populate_table_batch(self) -> None:
        """按时间预算创建卡片，每批结束都把控制权交还 Qt 事件循环。"""
        if not self._population_active:
            return
        try:
            self._populate_table_batch_step()
        except Exception as exc:
            self._fail_table_population(str(exc))

    def _populate_table_batch_step(self) -> None:
        """执行一个受控建卡时间片，异常由外层统一恢复页面。"""
        if self._cancel_event.is_set():
            self._cancel_table_population()
            return
        if not self._population_prepared:
            self._prepare_table_population()
            self._population_timer.start(0)
            return
        if self._population_finishing:
            self._complete_table_population()
            return

        total = len(self._population_queue)
        deadline = time.perf_counter() + self.POPULATION_TIME_SLICE_SECONDS
        batch_count = 0
        while self._population_index < total:
            item, category = self._population_queue[self._population_index]
            layout = self._column_layouts[category]
            layout.insertWidget(layout.count() - 1, self._create_scan_card(item, category))
            self._population_index += 1
            batch_count += 1
            if (
                batch_count >= self.POPULATION_BATCH_LIMIT
                or time.perf_counter() >= deadline
            ):
                break

        self._progress_bar.setValue(self._population_index)
        self._status_label.setText(
            f"正在生成校对列表（{self._population_index}/{total}）"
        )
        if self._population_index >= total:
            self._population_finishing = True
            self._status_label.setText("正在完成校对列表布局…")
            self._progress_bar.setFormat("正在完成校对列表 %v / %m")
            self._population_timer.start(self.POPULATION_LAYOUT_SETTLE_MS)
            return
        self._population_timer.start(0)

    def _fail_table_population(self, message: str) -> None:
        """列表创建失败时清除不完整结果并恢复所有页面控件。"""
        self._stop_table_population()
        self._active_task_kind = None
        self._scan_items.clear()
        self._clear_table_widgets()
        for category in self._column_counts:
            self._column_counts[category].setText("0 项")
        self._confirm_check.setChecked(False)
        self._confirm_check.setVisible(False)
        self._set_busy(False)
        self._import_button.setEnabled(False)
        self._status_label.setText("校对列表生成失败")
        QMessageBox.warning(self, "列表生成失败", f"无法生成图片校对列表：{message}")

    def _complete_table_population(self) -> None:
        """等待最后一批布局事件后，再开放核对与导入操作。"""
        completion_status = self._population_completion_status
        self._stop_table_population()
        self._active_task_kind = None
        self._set_busy(False)
        self._status_label.setText(completion_status)

    def _cancel_table_population(self) -> None:
        """停止尚未完成的建卡任务，不让旧定时器继续写入页面。"""
        clear_scan = self._population_clear_scan_on_cancel
        self._stop_table_population()
        self._active_task_kind = None
        if clear_scan:
            self._scan_items.clear()
            self._clear_table_widgets()
            for category in self._column_counts:
                self._column_counts[category].setText("0 项")
            self._confirm_check.setChecked(False)
            self._confirm_check.setVisible(False)
        else:
            self._cancel_event = threading.Event()
        self._set_busy(False)
        self._import_button.setEnabled(False)
        self._status_label.setText(
            "校对列表生成已取消，请重新扫描"
            if clear_scan
            else "校对列表刷新已取消，可重新搜索"
        )

    def _stop_table_population(self) -> None:
        self._population_timer.stop()
        self._population_active = False
        self._population_prepared = False
        self._population_finishing = False
        self._population_queue = []
        self._population_index = 0

    def _create_scan_card(self, item: ScanItem, category: str) -> QFrame:
        colors = {"正确": "#51cf66", "一对一": "#f59f00", "异常": "#ff6b6b"}
        card = QFrame()
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        card.setStyleSheet(
            f"QFrame {{ border: 1px solid {colors[category]}; border-radius: 6px; background: #252837; }}"
            "QLabel, QLineEdit, QPushButton { border: none; }"
        )
        row = QHBoxLayout(card)
        row.setContentsMargins(8, 8, 8, 8)
        row.setSpacing(10)

        preview = QLabel("图片")
        preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview.setFixedSize(72, 72)
        preview.setStyleSheet("background: #ffffff; color: #5f6473; border-radius: 2px;")
        pixmap = self._preview_pixmap(
            item.preview_image,
            item.path,
            preview.width(),
            preview.height(),
        )
        if not pixmap.isNull():
            preview.setPixmap(pixmap)
        row.addWidget(preview)

        source_box = QVBoxLayout()
        filename_label = QLabel(item.filename)
        filename_label.setWordWrap(True)
        original_label = QLabel(f"原字符：{item.original_char}")
        original_label.setProperty("role", "muted")
        source_box.addWidget(filename_label)
        source_box.addStretch()
        source_box.addWidget(original_label)
        row.addLayout(source_box, 1)

        if category == "异常" and item.category == "重复":
            duplicate_preview = QLabel("已有图片")
            duplicate_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
            duplicate_preview.setFixedSize(72, 72)
            duplicate_preview.setStyleSheet("background: #ffffff; color: #5f6473; border-radius: 2px;")
            duplicate_pixmap = self._preview_pixmap(
                item.duplicate_preview_image,
                item.duplicate_path,
                duplicate_preview.width(),
                duplicate_preview.height(),
            )
            if not duplicate_pixmap.isNull():
                duplicate_preview.setPixmap(duplicate_pixmap)
            row.addWidget(duplicate_preview)

            duplicate_box = QVBoxLayout()
            duplicate_title = QLabel("已有文件")
            duplicate_title.setProperty("role", "muted")
            duplicate_name = QLabel(item.duplicate_filename or "文件名未知")
            duplicate_name.setWordWrap(True)
            duplicate_name.setMaximumWidth(130)
            duplicate_issue = QLabel("重复，将跳过！")
            duplicate_issue.setStyleSheet("color: #ff8787; font-weight: 700;")
            duplicate_box.addWidget(duplicate_title)
            duplicate_box.addWidget(duplicate_name)
            duplicate_box.addStretch()
            duplicate_box.addWidget(duplicate_issue)
            row.addLayout(duplicate_box)
            return card

        action_box = QVBoxLayout()
        action_box.setAlignment(Qt.AlignmentFlag.AlignRight)
        if category == "正确":
            editor = QLineEdit(item.final_char)
            editor.setAlignment(Qt.AlignmentFlag.AlignCenter)
            editor.setFixedWidth(64)
            validation_label = QLabel()
            validation_label.setWordWrap(True)
            validation_label.setMaximumWidth(130)
            validation_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            editor.textChanged.connect(
                lambda text, target=item, field=editor, label=validation_label: self._set_final_char(
                    target, text, field, label
                )
            )
            self._update_final_char_validation(item, editor, validation_label)
            action_box.addWidget(validation_label, 0, Qt.AlignmentFlag.AlignRight)
            action_box.addStretch()
            action_box.addWidget(editor, 0, Qt.AlignmentFlag.AlignRight)
        elif category == "一对一":
            title = QLabel("请选择或输入：")
            title.setStyleSheet("color: #f59f00; font-weight: 700;")
            title.setAlignment(Qt.AlignmentFlag.AlignRight)
            action_box.addWidget(title)
            choices_host = QWidget()
            choices = FlowLayout(choices_host, spacing=8)
            choices_host.setMaximumWidth(220)
            button_group = QButtonGroup(card)
            button_group.setExclusive(True)

            editor = QLineEdit(item.final_char)
            editor.setAlignment(Qt.AlignmentFlag.AlignCenter)
            editor.setFixedSize(48, 54)
            editor.setMaxLength(1)
            editor.setAccessibleName("人工输入最终字符")
            editor.setToolTip("人工输入正确字符")
            for candidate in dict.fromkeys((*item.candidates, item.original_char)):
                button = QPushButton(candidate)
                button.setCheckable(True)
                button.setFixedSize(48, 54)
                button.setStyleSheet(
                    "QPushButton { font-size: 20px; font-weight: 700; padding: 0 2px 4px; "
                    "color: #f1f4f8; background: #282f3a; border: 1px solid #4a5565; border-radius: 6px; }"
                    "QPushButton:hover { background: #343b48; border-color: #f59f00; }"
                    "QPushButton:checked { color: #161a21; background: #f59f00; "
                    "border: 2px solid #ffd166; }"
                )
                button_group.addButton(button)
                button.setChecked(item.final_char == candidate)
                button.clicked.connect(
                    lambda checked=False, value=candidate, field=editor: (
                        field.setText(value) if checked else None
                    )
                )
                choices.addWidget(button)
            choices.addWidget(editor)
            action_box.addWidget(choices_host)
            editor.textChanged.connect(
                lambda text, target=item, field=editor, group=button_group: (
                    self._set_ambiguous_final_char(target, text, field, group)
                )
            )
            self._update_final_char_validation(item, editor, None)
        else:
            issue = QLabel(item.issue or "请检查该文件")
            issue.setWordWrap(True)
            issue.setStyleSheet("color: #ff8787; font-weight: 700;")
            issue.setMaximumWidth(130)
            action_box.addWidget(issue)
        row.addLayout(action_box)
        return card

    @staticmethod
    def _preview_pixmap(
        image: QImage | None,
        fallback_path: str,
        width: int,
        height: int,
    ) -> QPixmap:
        """优先使用扫描线程准备的预览，兼容旧数据的按需回退。"""
        if isinstance(image, QImage):
            return QPixmap() if image.isNull() else QPixmap.fromImage(image)
        return ImportPage._load_preview_pixmap(fallback_path, width, height)

    @staticmethod
    def _load_preview_pixmap(path: str, width: int, height: int) -> QPixmap:
        """使用 Pillow 解码缩略图，避免 Qt TIFF 插件输出无关的元数据告警。"""
        try:
            qimage = _decode_preview_image(path, width, height)
        except (OSError, ValueError):
            return QPixmap()
        return QPixmap.fromImage(qimage)

    def _set_final_char(
        self,
        item: ScanItem,
        text: str,
        editor: QLineEdit,
        validation_label: QLabel | None,
    ) -> None:
        item.final_char = text.strip()
        self._update_final_char_validation(item, editor, validation_label)
        self._validate_confirmations()

    @staticmethod
    def _update_final_char_validation(
        item: ScanItem,
        editor: QLineEdit,
        validation_label: QLabel | None,
    ) -> None:
        valid, message = validate_final_char(item.final_char)
        item.confirmed = valid
        border_color = "#51cf66" if valid else "#ff6b6b"
        editor.setStyleSheet(
            f"font-size: 26px; font-weight: 700; padding: 4px; "
            f"border: 2px solid {border_color}; border-radius: 6px; background: #202630;"
        )
        editor.setToolTip("人工输入正确字符" if valid else message)
        if validation_label is not None:
            validation_label.setText("有效" if valid else message)
            validation_label.setStyleSheet(f"color: {border_color}; font-size: 11px;")

    def _set_ambiguous_final_char(
        self,
        item: ScanItem,
        text: str,
        editor: QLineEdit,
        button_group: QButtonGroup,
    ) -> None:
        """更新歧义字符输入，并同步候选按钮的选中状态。"""
        self._set_final_char(item, text, editor, None)
        button_group.setExclusive(False)
        for button in button_group.buttons():
            button.setChecked(button.text() == item.final_char)
        button_group.setExclusive(True)

    def _validate_confirmations(self) -> bool:
        invalid = [
            item for item in self._scan_items
            if item.category != "重复" and not validate_final_char(item.final_char)[0]
        ]
        broken = [item for item in self._scan_items if item.issue.startswith("图片异常")]
        unselected = [
            item for item in self._scan_items
            if item.category in {"一对一", "歧义"} and not item.final_char
        ]
        has_conflicts = any(item.category not in {"正确", "一对一", "歧义"} for item in self._scan_items)
        conflict_unconfirmed = has_conflicts and not self._confirm_check.isChecked()
        enabled = (
            bool(self._scan_items) and not invalid and not unselected and not broken
            and not conflict_unconfirmed and not self._cancel_button.isVisible()
        )
        self._import_button.setEnabled(enabled)
        if broken:
            self._hint_label.setText(f"有 {len(broken)} 张图片无法读取，请移出目录后重新扫描")
        elif unselected:
            self._hint_label.setText(f"还有 {len(unselected)} 项需要选择简体字或繁体字")
        elif invalid:
            self._hint_label.setText(f"还有 {len(invalid)} 项需要填写并确认有效的最终字符")
        elif conflict_unconfirmed:
            self._hint_label.setText("请核对第三栏中的字符，并勾选确认")
        else:
            self._hint_label.setText("字符确认完成；重复图片会在导入时自动跳过")
        return enabled

    def start_import(self) -> None:
        """校验页面信息并在后台创建或追加字库。"""
        if self._active_task_kind is not None:
            QMessageBox.information(self, "任务进行中", "请等待当前任务结束后再开始导入。")
            return
        name = self._name_edit.text().strip()
        error = self._validate_library_name(name)
        if error:
            QMessageBox.warning(self, "信息不完整", error)
            return
        if not self._validate_confirmations():
            QMessageBox.warning(self, "尚未确认", "请先完成所有字符确认，并处理异常图片。")
            return
        directory = self._directory_edit.text().strip()
        overrides = {item.path: item.final_char for item in self._scan_items}
        scanned_files = {
            os.path.abspath(item.path): {
                "md5": item.digest,
                "size": item.source_size,
                "mtime_ns": item.source_mtime_ns,
            }
            for item in self._scan_items
            if item.digest
        }
        dpi = self._dpi_spin.value()
        width = self._width_px_spin.value()
        height = self._height_px_spin.value()
        width_mm = round(self._width_mm_spin.value(), 2)
        height_mm = round(self._height_mm_spin.value(), 2)
        append_mode = self._append_mode
        existing_glyph = self._glyph_service
        self._import_generation += 1
        target_directory = "" if append_mode else os.path.join(config.ZIKU_ROOT, name)
        context = ImportRunContext(
            token=self._import_generation,
            append_mode=append_mode,
            library_name=name,
            target_directory=target_directory,
            cancel_event=threading.Event(),
            directory_created=threading.Event(),
        )
        self._cancel_event = context.cancel_event

        def import_function(progress_signal: Signal) -> dict[str, Any]:
            if context.append_mode:
                if existing_glyph is None:
                    raise RuntimeError("未指定要追加的字库。")
                glyph = existing_glyph
            else:
                os.makedirs(context.target_directory, exist_ok=False)
                context.directory_created.set()
                glyph = GlyphService.open(context.library_name, context.target_directory)

            def report(message: str, current: int, total: int) -> None:
                progress_signal.emit(current, total, message)

            service = ImportService(
                glyph,
                progress_callback=report,
                cancel_callback=context.cancel_event.is_set,
            )
            result = service.import_batch(
                directory, dpi, width, height, char_overrides=overrides,
                width_mm=width_mm,
                height_mm=height_mm,
                init_meta=not context.append_mode,
                scanned_files=scanned_files,
            )
            result["字库路径"] = glyph.ziku_dir
            return result

        worker = ImportTask(import_function)
        self._workers.add(worker)
        worker.signals.progress.connect(
            lambda current, total, message, token=context.token: self._import_progress(
                token, current, total, message
            )
        )
        worker.signals.finished.connect(
            lambda result, task_context=context: self._import_finished(result, task_context)
        )
        worker.signals.failed.connect(
            lambda message, task_context=context: self._import_failed(message, task_context)
        )
        worker.signals.finished.connect(lambda _result, task=worker: self._release_worker(task))
        worker.signals.failed.connect(lambda _message, task=worker: self._release_worker(task))
        self._active_import_token = context.token
        self._active_task_kind = "import"
        self._return_home_after_import = False
        self._set_busy(True, len(self._scan_items), "正在导入字图")
        self._thread_pool.start(worker)

    def _validate_library_name(self, name: str) -> str:
        if not name:
            return "请输入字库名称。"
        if not is_safe_windows_filename(name):
            return "字库名称不是有效的 Windows 目录名。"
        if not self._append_mode and (name in self._existing_names or os.path.exists(os.path.join(config.ZIKU_ROOT, name))):
            return "该字库名称已存在，请换一个名称。"
        return ""

    def _import_progress(self, token: int, current: int, total: int, message: str) -> None:
        if token == self._active_import_token:
            self._update_progress(current, total, message)

    def _import_finished(
        self,
        result: object,
        context: ImportRunContext | None = None,
    ) -> None:
        if context is not None and context.token != self._active_import_token:
            return
        data = result if isinstance(result, dict) else {}
        append_mode = context.append_mode if context is not None else self._append_mode
        name = context.library_name if context is not None else self._name_edit.text().strip()
        return_home = self._return_home_after_import
        self._active_import_token = None
        self._active_task_kind = None
        self._return_home_after_import = False
        self._set_busy(False)
        action = "字库添加完成" if append_mode else f"字库“{name}”创建完成"
        message = f"{action}。\n\n成功：{data.get('成功', 0)}\n跳过：{data.get('跳过', 0)}\n失败：{data.get('失败', 0)}"
        if data.get("已取消"):
            action = "导入已取消，已完成的数据已安全保存"
            message = f"{action}。\n\n成功：{data.get('成功', 0)}\n跳过：{data.get('跳过', 0)}\n失败：{data.get('失败', 0)}"
        elapsed = max(0.0, time.perf_counter() - context.started_at) if context else 0.0
        message += f"\n\n总耗时：{format_elapsed_time(elapsed)}"
        QMessageBox.information(self, "导入完成", message)
        path = str(data.get("字库路径", ""))
        self.import_completed.emit(name, path, data)
        self.status_message.emit(action)
        if return_home or (not append_mode and not data.get("已取消")):
            self.home_requested.emit()

    def _import_failed(
        self,
        message: str,
        context: ImportRunContext | None = None,
    ) -> None:
        if context is not None and context.token != self._active_import_token:
            return
        return_home = self._return_home_after_import
        if context is not None:
            self._cleanup_failed_import_directory(context)
        elapsed = max(0.0, time.perf_counter() - context.started_at) if context else 0.0
        self._active_import_token = None
        self._active_task_kind = None
        self._return_home_after_import = False
        self._task_failed(f"{message}\n\n总耗时：{format_elapsed_time(elapsed)}")
        if return_home:
            self.home_requested.emit()

    @staticmethod
    def _cleanup_failed_import_directory(context: ImportRunContext) -> None:
        """只清理本任务明确创建、且仍位于新版字库根目录下的目录。"""
        if context.append_mode or not context.directory_created.is_set():
            return
        root = os.path.realpath(config.ZIKU_ROOT)
        target = os.path.realpath(context.target_directory)
        if target != root and os.path.dirname(target) == root:
            shutil.rmtree(target, ignore_errors=True)

    def _update_progress(self, current: int, total: int, message: str) -> None:
        self._progress_bar.setMaximum(max(1, total))
        self._progress_bar.setValue(current)
        self._status_label.setText(f"{message}（{current}/{total}）")

    def _set_busy(self, busy: bool, total: int = 1, message: str = "") -> None:
        self._progress_bar.setVisible(busy)
        self._cancel_button.setVisible(busy)
        self._cancel_button.setEnabled(busy)
        self._scan_button.setEnabled(not busy)
        self._directory_edit.setEnabled(not busy)
        self._search_edit.setEnabled(not busy)
        self._search_button.setEnabled(not busy)
        self._import_button.setEnabled(False if busy else self._validate_confirmations())
        if busy:
            self._progress_bar.setRange(0, max(1, total))
            self._progress_bar.setValue(0)
            self._progress_bar.setFormat(f"{message} %v / %m")
            self._status_label.setText(message)

    def cancel_task(self) -> None:
        """请求取消可中断的扫描；导入阶段仅阻止再次操作以保护数据。"""
        self._cancel_event.set()
        if self._population_active:
            self._cancel_table_population()
            return
        self._cancel_button.setEnabled(False)
        self._status_label.setText("正在取消任务…")

    def _task_cancelled(self) -> None:
        self._set_busy(False)
        self._status_label.setText("扫描已取消")

    def _task_failed(self, message: str) -> None:
        self._set_busy(False)
        self._status_label.setText("任务失败")
        QMessageBox.warning(self, "任务失败", message)

    def _request_home(self) -> None:
        if self._active_task_kind == "import":
            if self._return_home_after_import:
                return
            answer = QMessageBox.question(
                self,
                "导入正在执行",
                "导入正在写入字库。确定停止导入，并在安全保存已完成项目后返回首页吗？",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            self._return_home_after_import = True
            self.cancel_task()
            self._status_label.setText("正在安全停止导入，完成后返回首页…")
            return
        if self._active_task_kind in {"scan", "population"} or self._cancel_button.isVisible():
            answer = QMessageBox.question(self, "任务进行中", "任务仍在进行，确定取消并返回首页吗？")
            if answer != QMessageBox.StandardButton.Yes:
                return
            self.cancel_task()
            if self._active_task_kind == "scan":
                self._active_task_kind = None
                self._set_busy(False)
        self._scan_generation += 1
        self.home_requested.emit()

    def _reset_scan(self) -> None:
        self._scan_generation += 1
        self._stop_table_population()
        self._clear_scan_results()
        self._search_edit.clear()
        self._directory_edit.clear()
        if self._active_task_kind != "import":
            self._active_task_kind = None
            self._cancel_event = threading.Event()
            self._set_busy(False)
        self._status_label.setText("请选择文字图片目录")

    def _clear_scan_results(self) -> None:
        """清除上一批扫描结果，避免确认状态被后续目录复用。"""
        self._stop_table_population()
        self._scan_items.clear()
        self._clear_table_widgets()
        for category in self._column_layouts:
            self._column_counts[category].setText("0 项")
        self._confirm_check.setChecked(False)
        self._confirm_check.setVisible(False)
        self._hint_label.setText("源目录和源文件名不会被修改")
        self._import_button.setEnabled(False)

    def _clear_table_widgets(self) -> None:
        """移除三栏现有卡片，保留每栏末尾的伸缩项。"""
        for layout in self._column_layouts.values():
            while layout.count() > 1:
                child = layout.takeAt(0)
                if child is None:
                    break
                widget = child.widget()
                if widget is not None:
                    widget.deleteLater()

    def _release_worker(self, worker: QRunnable) -> None:
        self._workers.discard(worker)

    def shutdown(self) -> None:
        """关闭程序时使扫描结果失效，并请求当前导入安全停止。"""

        self._scan_generation += 1
        self._active_import_token = None
        self._cancel_event.set()
        self._population_timer.stop()
        self._stop_table_population()
        self._workers.clear()
