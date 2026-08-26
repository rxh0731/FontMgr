"""文字统计 PySide6 页面。"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QIODevice, QMimeData, QSaveFile, QThreadPool, QTimer, Signal, Slot
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

import config
from data.library_database import LIBRARY_DATABASE_FILENAME
from services.scripture_layout_service import GenerationProgress
from services.text_statistics_service import (
    BatchTextExtraction,
    MissingCharacterResult,
    TextStatistics,
    TextStatisticsService,
    analyze_text,
    clean_plain_text,
    source_file_filter,
)
from ui.workers import FunctionWorker
from utils.file_utils import natural_key


class _StatisticsTextEdit(QPlainTextEdit):
    """保留普通文字粘贴，并把文件或图片交给页面后台处理。"""

    external_paste_requested = Signal()

    def insertFromMimeData(self, source: QMimeData) -> None:
        if source.hasUrls() or source.hasImage():
            self.external_paste_requested.emit()
            return
        if source.hasText():
            cursor = self.textCursor()
            cursor.insertText(clean_plain_text(source.text()))
            self.setTextCursor(cursor)
            return
        super().insertFromMimeData(source)


class TextStatisticsPage(QWidget):
    """提取经文正文、统计不重复汉字并分析字库缺字。"""

    home_requested = Signal()
    status_message = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._thread_pool = QThreadPool.globalInstance()
        self._workers: set[FunctionWorker] = set()
        self._file_worker: FunctionWorker | None = None
        self._missing_worker: FunctionWorker | None = None
        self._generation = 0
        self._missing_generation = 0
        self._source_path = ""
        self._last_source_directory = config.SCRIPT_DIR
        self._last_font_directory = config.ZIKU_ROOT
        self._statistics = analyze_text("")
        self._missing_characters: tuple[str, ...] = ()
        self._font_cache_path = ""
        self._font_cache: frozenset[str] | None = None
        self._font_cache_invalid = 0
        self._font_cache_kind = ""
        self._font_cache_variants = 0
        self._font_cache_issues: tuple[str, ...] = ()

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(350)
        self._refresh_timer.timeout.connect(self._refresh_statistics)

        self._build_ui()
        self._load_library_names()
        self._source_mode_changed()
        self._refresh_statistics()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("文字统计")
        title.setProperty("role", "pageTitle")
        home_button = QPushButton("返回首页")
        home_button.clicked.connect(self.home_requested)
        header.addWidget(title)
        header.addStretch()
        header.addWidget(home_button)
        root.addLayout(header)

        self._main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._main_splitter.setObjectName("textStatisticsMainSplitter")
        self._main_splitter.setChildrenCollapsible(False)
        root.addWidget(self._main_splitter, 1)

        self._left_splitter = QSplitter(Qt.Orientation.Vertical)
        self._left_splitter.setObjectName("textStatisticsLeftSplitter")
        self._left_splitter.setChildrenCollapsible(False)
        self._main_splitter.addWidget(self._left_splitter)

        self._source_panel = QFrame()
        self._source_panel.setObjectName("textStatisticsSourcePanel")
        self._source_panel.setProperty("role", "card")
        self._source_panel.setMinimumHeight(210)
        source_layout = QVBoxLayout(self._source_panel)
        source_layout.setContentsMargins(12, 10, 12, 10)
        source_layout.setSpacing(7)

        source_heading = QLabel("信息输入")
        source_heading.setObjectName("sectionTitle")
        source_layout.addWidget(source_heading)

        file_row = QHBoxLayout()
        file_label = QLabel("经文文本")
        file_label.setFixedWidth(72)
        self._source_edit = QLineEdit()
        self._source_edit.setObjectName("textStatisticsSourcePath")
        self._source_edit.setReadOnly(True)
        self._source_edit.setPlaceholderText("可选择多个文档或图片，也可在下方直接输入、粘贴")
        self._select_button = QPushButton("选择文件")
        self._select_button.clicked.connect(self._select_files)
        self._statistics_button = QPushButton("文字统计")
        self._statistics_button.setProperty("role", "primary")
        self._statistics_button.clicked.connect(lambda: self._run_statistics(False))
        file_row.addWidget(file_label)
        file_row.addWidget(self._source_edit, 1)
        file_row.addWidget(self._select_button)
        file_row.addWidget(self._statistics_button)
        source_layout.addLayout(file_row)

        self._source_button_group = QButtonGroup(self)
        self._system_source_radio = QRadioButton("本系统字库")
        self._external_source_radio = QRadioButton("外部字库目录")
        self._system_source_radio.setMinimumWidth(112)
        self._external_source_radio.setMinimumWidth(112)
        self._source_button_group.addButton(self._system_source_radio)
        self._source_button_group.addButton(self._external_source_radio)
        self._system_source_radio.setChecked(True)

        system_row = QHBoxLayout()
        comparison_label = QLabel("比较来源")
        comparison_label.setFixedWidth(72)
        self._system_library_combo = QComboBox()
        self._system_library_combo.setObjectName("textStatisticsSystemLibraryCombo")
        self._system_library_combo.setPlaceholderText("没有可用的系统字库")
        system_row.addWidget(comparison_label)
        system_row.addWidget(self._system_source_radio)
        system_row.addWidget(self._system_library_combo, 1)
        source_layout.addLayout(system_row)

        external_row = QHBoxLayout()
        external_spacer = QLabel()
        external_spacer.setFixedWidth(72)
        self._external_path_edit = QLineEdit()
        self._external_path_edit.setObjectName("textStatisticsExternalFontPath")
        self._external_path_edit.setReadOnly(True)
        self._external_path_edit.setPlaceholderText("选择包含文字图片的外部目录")
        self._browse_font_button = QPushButton("浏览")
        self._browse_font_button.clicked.connect(self._select_font_directory)
        external_row.addWidget(external_spacer)
        external_row.addWidget(self._external_source_radio)
        external_row.addWidget(self._external_path_edit, 1)
        external_row.addWidget(self._browse_font_button)
        source_layout.addLayout(external_row)

        self._comparison_status = QLabel("尚未载入比较字库")
        self._comparison_status.setProperty("role", "muted")
        self._comparison_status.setWordWrap(True)
        source_layout.addWidget(self._comparison_status)
        self._comparison_progress = QProgressBar()
        self._comparison_progress.setRange(0, 0)
        self._comparison_progress.setFixedHeight(22)
        self._comparison_progress.setFormat("正在核对字库…")
        self._comparison_progress.hide()
        source_layout.addWidget(self._comparison_progress)

        self._system_source_radio.toggled.connect(self._source_mode_changed)
        self._system_library_combo.currentIndexChanged.connect(
            self._comparison_source_changed
        )

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFormat("正在准备…")
        self._progress.setFixedHeight(22)
        self._progress.hide()
        source_layout.addWidget(self._progress)
        self._left_splitter.addWidget(self._source_panel)

        self._content_section = QFrame()
        self._content_section.setObjectName("textStatisticsContentPanel")
        self._content_section.setProperty("role", "card")
        self._content_section.setMinimumHeight(220)
        content_layout = QVBoxLayout(self._content_section)
        content_layout.setContentsMargins(12, 10, 12, 12)
        content_layout.setSpacing(7)
        content_header = QHBoxLayout()
        content_title = QLabel("经文内容")
        content_title.setObjectName("sectionTitle")
        content_hint = QLabel("可输入或粘贴文字、文件和图片")
        content_hint.setProperty("role", "muted")
        content_header.addWidget(content_title)
        content_header.addSpacing(8)
        content_header.addWidget(content_hint)
        content_header.addStretch()
        content_layout.addLayout(content_header)
        self._content_edit = _StatisticsTextEdit()
        self._content_edit.setObjectName("textStatisticsContent")
        self._content_edit.setPlaceholderText("在此输入或粘贴需要统计的经文正文")
        self._content_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self._content_edit.textChanged.connect(self._content_changed)
        self._content_edit.external_paste_requested.connect(self._paste_external_content)
        content_layout.addWidget(self._content_edit, 1)
        self._left_splitter.addWidget(self._content_section)
        self._left_splitter.setStretchFactor(0, 0)
        self._left_splitter.setStretchFactor(1, 1)
        self._left_splitter.setSizes([250, 500])

        self._right_splitter = QSplitter(Qt.Orientation.Vertical)
        self._right_splitter.setObjectName("textStatisticsRightSplitter")
        self._right_splitter.setChildrenCollapsible(False)
        self._main_splitter.addWidget(self._right_splitter)

        self._info_label = QLabel("尚未载入文本")
        self._info_label.setProperty("role", "muted")
        self._info_label.setWordWrap(True)
        self._info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self._statistics_section = self._build_result_section(
            "统计结果",
            export_text="全部文字导出",
            export_callback=self._export_all,
            count_attribute="_count_label",
            text_attribute="_all_results",
            summary_widget=self._info_label,
        )
        self._statistics_section.setObjectName("textStatisticsResultsPanel")
        self._statistics_section.setMinimumHeight(220)
        self._right_splitter.addWidget(self._statistics_section)

        self._missing_section = self._build_result_section(
            "缺失文字",
            export_text="缺失文字导出",
            export_callback=self._export_missing,
            count_attribute="_missing_count_label",
            text_attribute="_missing_results",
        )
        self._missing_section.setObjectName("textStatisticsMissingPanel")
        self._missing_section.setMinimumHeight(220)
        self._right_splitter.addWidget(self._missing_section)
        self._right_splitter.setStretchFactor(0, 1)
        self._right_splitter.setStretchFactor(1, 1)
        self._right_splitter.setSizes([360, 390])

        self._main_splitter.setStretchFactor(0, 3)
        self._main_splitter.setStretchFactor(1, 2)
        self._main_splitter.setSizes([650, 430])

    def _build_result_section(
        self,
        title: str,
        *,
        export_text: str,
        export_callback: Callable[[], None],
        count_attribute: str,
        text_attribute: str,
        summary_widget: QWidget | None = None,
    ) -> QWidget:
        section = QFrame()
        section.setProperty("role", "card")
        layout = QVBoxLayout(section)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(7)
        header = QHBoxLayout()
        title_label = QLabel(title)
        title_label.setObjectName("sectionTitle")
        count_label = QLabel()
        count_label.setProperty("role", "muted")
        export_button = QPushButton(export_text)
        export_button.setObjectName("compactButton")
        export_button.clicked.connect(export_callback)
        header.addWidget(title_label)
        header.addWidget(count_label)
        header.addStretch()
        header.addWidget(export_button)
        output = QPlainTextEdit()
        output.setReadOnly(True)
        output.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        output.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addLayout(header)
        if summary_widget is not None:
            layout.addWidget(summary_widget)
        layout.addWidget(output, 1)
        setattr(self, count_attribute, count_label)
        setattr(self, text_attribute, output)
        if text_attribute == "_all_results":
            self._export_all_button = export_button
        else:
            self._export_missing_button = export_button
        return section

    @Slot()
    def _content_changed(self) -> None:
        if self._source_path:
            self._source_path = ""
            self._source_edit.clear()
        self._refresh_timer.start()

    def _refresh_statistics(self) -> None:
        self._run_statistics(True)

    def _run_statistics(self, silent: bool = False) -> None:
        content = self._content_edit.toPlainText()
        if not content.strip():
            self._statistics = analyze_text("")
            self._missing_characters = ()
            self._info_label.setText("尚未载入文本")
            self._count_label.setText("共 0 个不重复汉字")
            self._all_results.clear()
            self._missing_count_label.setText("等待正文和比较字库")
            self._missing_results.setPlainText("（载入正文后分析缺失文字）")
            self._update_export_buttons()
            if not silent:
                QMessageBox.warning(self, "提示", "请先载入文件或在内容窗口粘贴文本。")
            return
        self._statistics = analyze_text(content)
        prefix = f"文件：{os.path.basename(self._source_path)}  |  " if self._source_path else ""
        stats = self._statistics
        self._info_label.setText(
            f"{prefix}总字符：{stats.total_characters}  |  汉字：{stats.chinese_characters}  |  "
            f"英文词：{stats.english_words}  |  标点：{stats.punctuation}  |  "
            f"空白：{stats.whitespace}"
        )
        self._count_label.setText(f"共 {len(stats.unique_chinese)} 个不重复汉字")
        self._all_results.setPlainText(" ".join(stats.unique_chinese))
        self._refresh_missing_results()
        self._update_export_buttons()

    def _select_files(self) -> None:
        if self._file_worker is not None:
            return
        paths, _selected_filter = QFileDialog.getOpenFileNames(
            self,
            "选择经文文件或图片（可多选）",
            self._last_source_directory,
            source_file_filter(),
        )
        if paths:
            self._last_source_directory = os.path.dirname(paths[0])
            self._start_file_task(tuple(paths), replace=True)

    def _start_file_task(
        self,
        paths: tuple[str, ...],
        *,
        replace: bool,
        clipboard_image: Image.Image | None = None,
    ) -> None:
        if self._file_worker is not None:
            if clipboard_image is not None:
                clipboard_image.close()
            QMessageBox.information(self, "正在处理", "请等待当前文件或图片处理完成。")
            return
        self._generation += 1
        generation = self._generation
        self._set_file_busy(True)

        def load(progress_callback: Callable[[object], None]) -> BatchTextExtraction:
            report = lambda ratio, message: progress_callback((ratio, message))
            if clipboard_image is not None:
                return TextStatisticsService.recognize_clipboard_image(
                    clipboard_image,
                    report,
                )
            return TextStatisticsService.extract_files(paths, report)

        worker = FunctionWorker(load, with_progress=True)
        self._file_worker = worker
        self._workers.add(worker)
        worker.signals.progress.connect(
            lambda value, token=generation: self._file_progress(token, value)
        )
        worker.signals.finished.connect(
            lambda result, token=generation, task=worker: self._file_finished(
                token, result, task, paths, replace
            )
        )
        worker.signals.failed.connect(
            lambda message, token=generation, task=worker: self._file_failed(
                token, message, task
            )
        )
        self._thread_pool.start(worker)

    def _file_progress(self, generation: int, value: object) -> None:
        if generation != self._generation or not isinstance(value, tuple) or len(value) != 2:
            return
        ratio, message = value
        try:
            percent = round(float(ratio) * 100)
        except (TypeError, ValueError):
            percent = 0
        self._progress.setValue(max(0, min(100, percent)))
        self._progress.setFormat(f"{message}　%p%")

    def _file_finished(
        self,
        generation: int,
        result: object,
        worker: FunctionWorker,
        paths: tuple[str, ...],
        replace: bool,
    ) -> None:
        self._release_worker(worker)
        if generation != self._generation or not isinstance(result, BatchTextExtraction):
            return
        self._set_file_busy(False)
        if result.text:
            if replace:
                self._content_edit.setPlainText(result.text)
            else:
                cursor = self._content_edit.textCursor()
                if cursor.hasSelection():
                    cursor.removeSelectedText()
                cursor.insertText(result.text)
                self._content_edit.setTextCursor(cursor)
            self._source_path = paths[0] if len(paths) == 1 else ""
            self._source_edit.setText(
                paths[0] if len(paths) == 1 else f"已处理 {len(result.reports)} 个文件或图片"
            )
            self._run_statistics(True)
        lines = [f"成功处理：{len(result.reports)} 个文件或图片", *result.reports]
        if result.failures:
            lines.extend((f"未能处理：{len(result.failures)} 个文件或图片", *result.failures))
        QMessageBox.information(self, "文件处理结果", "\n".join(lines))
        self.status_message.emit(
            f"文字统计：成功处理 {len(result.reports)} 项，失败 {len(result.failures)} 项"
        )

    def _file_failed(
        self,
        generation: int,
        message: str,
        worker: FunctionWorker,
    ) -> None:
        self._release_worker(worker)
        if generation != self._generation:
            return
        self._set_file_busy(False)
        QMessageBox.warning(self, "文件处理失败", message)

    def _release_worker(self, worker: FunctionWorker) -> None:
        self._workers.discard(worker)
        if self._file_worker is worker:
            self._file_worker = None
        if self._missing_worker is worker:
            self._missing_worker = None

    def _set_file_busy(self, busy: bool) -> None:
        self._select_button.setEnabled(not busy)
        self._statistics_button.setEnabled(not busy)
        self._select_button.setText("正在处理" if busy else "选择文件")
        self._progress.setVisible(busy)
        if busy:
            self._progress.setValue(0)
            self._progress.setFormat("正在准备…")

    def _paste_external_content(self) -> None:
        mime = QApplication.clipboard().mimeData()
        paths = tuple(
            url.toLocalFile()
            for url in mime.urls()
            if url.isLocalFile() and os.path.isfile(url.toLocalFile())
        )
        if paths:
            self._start_file_task(paths, replace=False)
            return
        if mime.hasImage():
            qimage = QImage(mime.imageData())
            if not qimage.isNull():
                self._start_file_task(
                    (),
                    replace=False,
                    clipboard_image=self._qimage_to_pil(qimage),
                )
                return
        QMessageBox.warning(self, "粘贴结果", "剪贴板中没有可用于文字统计的内容。")

    @staticmethod
    def _qimage_to_pil(image: QImage) -> Image.Image:
        normalized = image.convertToFormat(QImage.Format.Format_RGBA8888)
        return Image.frombytes(
            "RGBA",
            (normalized.width(), normalized.height()),
            bytes(normalized.bits()),
        )

    def _select_font_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择外部字库目录（含文字图片的文件夹）",
            self._external_path_edit.text() or self._last_font_directory,
        )
        if not directory:
            return
        self._last_font_directory = directory
        self._external_path_edit.setText(directory)
        self._comparison_source_changed()

    def _load_library_names(self) -> None:
        previous = self._system_library_combo.currentData()
        self._system_library_combo.clear()
        root = Path(config.ZIKU_ROOT)
        if root.is_dir():
            names = sorted(
                (
                    entry.name
                    for entry in root.iterdir()
                    if entry.is_dir()
                    and not entry.is_symlink()
                    and (entry / LIBRARY_DATABASE_FILENAME).is_file()
                ),
                key=natural_key,
            )
            for name in names:
                self._system_library_combo.addItem(name, str(root / name))
        if previous:
            index = self._system_library_combo.findData(previous)
            if index >= 0:
                self._system_library_combo.setCurrentIndex(index)
                return
        if self._system_library_combo.count() > 0:
            self._system_library_combo.setCurrentIndex(0)

    @Slot()
    def _source_mode_changed(self) -> None:
        system_mode = self._system_source_radio.isChecked()
        self._system_library_combo.setEnabled(system_mode)
        self._external_path_edit.setEnabled(not system_mode)
        self._browse_font_button.setEnabled(not system_mode)
        self._comparison_source_changed()

    @Slot()
    def _comparison_source_changed(self) -> None:
        self._missing_generation += 1
        self._font_cache_path = ""
        self._font_cache = None
        self._font_cache_invalid = 0
        self._font_cache_kind = ""
        self._font_cache_variants = 0
        self._font_cache_issues = ()
        self._refresh_missing_results()

    def _selected_comparison_source(self) -> tuple[str, str]:
        if self._system_source_radio.isChecked():
            return "system", str(self._system_library_combo.currentData() or "")
        return "external", self._external_path_edit.text().strip()

    def _refresh_missing_results(self) -> None:
        characters = self._statistics.unique_chinese
        if not characters:
            self._missing_characters = ()
            self._missing_count_label.setText("共 0 个缺失")
            self._missing_results.setPlainText("（无统计文字）")
            self._update_export_buttons()
            return
        source_mode, directory = self._selected_comparison_source()
        if not directory:
            self._missing_characters = ()
            message = (
                "没有可用的系统字库"
                if source_mode == "system"
                else "请选择外部字库目录"
            )
            self._comparison_status.setText(message)
            self._comparison_status.setToolTip("")
            self._missing_count_label.setText(message)
            self._missing_results.setPlainText("（选择比较来源后分析缺失文字）")
            self._update_export_buttons()
            return
        source_key = f"{source_mode}|{os.path.normcase(os.path.abspath(directory))}"
        if self._font_cache is not None and source_key == self._font_cache_path:
            self._apply_cached_missing()
            return
        if self._missing_worker is not None:
            self._missing_generation += 1
            return
        self._missing_generation += 1
        generation = self._missing_generation
        self._missing_count_label.setText("正在读取字库…")
        self._missing_results.setPlainText("正在后台分析缺失文字…")
        self._comparison_status.setText("正在核对字库可用成品…")
        self._comparison_progress.setRange(0, 0)
        self._comparison_progress.setFormat("正在打开比较来源…")
        self._comparison_progress.show()

        def analyze(progress_callback: Callable[[object], None]) -> MissingCharacterResult:
            if source_mode == "system":
                return TextStatisticsService.analyze_system_library(
                    characters,
                    directory,
                    progress_callback,
                )
            progress_callback(GenerationProgress(0, 1, "正在扫描外部图片目录", True))
            result = TextStatisticsService.analyze_external_directory(
                characters,
                directory,
            )
            progress_callback(GenerationProgress(1, 1, "外部图片目录核对完成"))
            return result

        worker = FunctionWorker(analyze, with_progress=True)
        self._missing_worker = worker
        self._workers.add(worker)
        worker.signals.progress.connect(
            lambda value, token=generation, task=worker: self._missing_progress(
                token, value, task
            )
        )
        worker.signals.finished.connect(
            lambda result, token=generation, task=worker, source=source_key: self._missing_finished(
                token, result, task, source
            )
        )
        worker.signals.failed.connect(
            lambda message, token=generation, task=worker: self._missing_failed(
                token, message, task
            )
        )
        self._thread_pool.start(worker)

    def _missing_progress(
        self,
        generation: int,
        value: object,
        worker: FunctionWorker,
    ) -> None:
        if (
            generation != self._missing_generation
            or self._missing_worker is not worker
            or not isinstance(value, GenerationProgress)
        ):
            return
        self._comparison_progress.show()
        if value.indeterminate:
            self._comparison_progress.setRange(0, 0)
            self._comparison_progress.setFormat(value.message)
            return
        self._comparison_progress.setRange(0, max(1, value.total))
        self._comparison_progress.setValue(value.completed)
        self._comparison_progress.setFormat(f"{value.message}　%p%")

    def _missing_finished(
        self,
        generation: int,
        result: object,
        worker: FunctionWorker,
        source_path: str,
    ) -> None:
        self._release_worker(worker)
        if not isinstance(result, MissingCharacterResult):
            self._comparison_progress.hide()
            return
        self._comparison_progress.hide()
        mode, current_directory = self._selected_comparison_source()
        current_path = (
            f"{mode}|{os.path.normcase(os.path.abspath(current_directory))}"
            if current_directory
            else ""
        )
        if source_path == current_path:
            self._font_cache_path = source_path
            self._font_cache = frozenset(result.available_characters)
            self._font_cache_invalid = result.invalid_filenames
            self._font_cache_kind = result.source_kind
            self._font_cache_variants = result.valid_variants
            self._font_cache_issues = result.issues
        if generation != self._missing_generation:
            self._refresh_missing_results()
            return
        self._apply_cached_missing()

    def _missing_failed(
        self,
        generation: int,
        message: str,
        worker: FunctionWorker,
    ) -> None:
        self._release_worker(worker)
        self._comparison_progress.hide()
        if generation != self._missing_generation:
            self._refresh_missing_results()
            return
        self._missing_characters = ()
        self._comparison_status.setText(f"比较来源读取失败：{message}")
        self._comparison_status.setToolTip(message)
        self._missing_count_label.setText("比较来源读取失败")
        self._missing_results.setPlainText(f"无法读取比较来源：{message}")
        self._update_export_buttons()

    def _apply_cached_missing(self) -> None:
        available = self._font_cache or frozenset()
        self._missing_characters = tuple(
            character
            for character in self._statistics.unique_chinese
            if character not in available
        )
        issue_count = self._font_cache_invalid + len(self._font_cache_issues)
        suffix = f"，异常 {issue_count} 项" if issue_count else ""
        self._missing_count_label.setText(
            f"共 {len(self._missing_characters)} 个缺失{suffix}"
        )
        self._missing_results.setPlainText(
            " ".join(self._missing_characters)
            if self._missing_characters
            else "（无缺失文字）"
        )
        self._comparison_status.setText(
            f"{self._font_cache_kind}：可用 {len(available)} 个字，"
            f"{self._font_cache_variants} 个变体；异常 {issue_count} 项"
        )
        tooltip_lines = list(self._font_cache_issues[:20])
        if len(self._font_cache_issues) > 20:
            tooltip_lines.append(f"另有 {len(self._font_cache_issues) - 20} 项未展开")
        if self._font_cache_invalid:
            tooltip_lines.append(
                f"外部目录中有 {self._font_cache_invalid} 个图片文件名无法识别"
            )
        self._comparison_status.setToolTip("\n".join(tooltip_lines))
        self._update_export_buttons()
        self.status_message.emit(
            f"文字统计：{self._font_cache_kind}中缺失 {len(self._missing_characters)} 个汉字"
        )

    def _update_export_buttons(self) -> None:
        self._export_all_button.setEnabled(bool(self._statistics.unique_chinese))
        self._export_missing_button.setEnabled(bool(self._missing_characters))

    def _export_all(self) -> None:
        self._export_characters(
            self._statistics.unique_chinese,
            "导出全部文字",
            "统计结果.txt",
        )

    def _export_missing(self) -> None:
        self._export_characters(
            self._missing_characters,
            "导出缺失文字",
            "缺失文字.txt",
        )

    def _export_characters(
        self,
        characters: tuple[str, ...],
        title: str,
        default_name: str,
    ) -> None:
        if not characters:
            QMessageBox.warning(self, "提示", "当前没有可导出的文字。")
            return
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            title,
            os.path.join(config.SCRIPT_DIR, default_name),
            "文本文件（*.txt）",
        )
        if not path:
            return
        if os.path.splitext(path)[1].lower() != ".txt":
            path += ".txt"
        output = QSaveFile(path)
        if not output.open(QIODevice.OpenModeFlag.WriteOnly):
            QMessageBox.critical(self, "导出失败", output.errorString())
            return
        if output.write(" ".join(characters).encode("utf-8")) < 0 or not output.commit():
            QMessageBox.critical(self, "导出失败", output.errorString())
            return
        QMessageBox.information(self, "导出成功", f"已导出到：\n{path}")

    def shutdown(self) -> None:
        """离开页面后废弃仍在运行的后台结果。"""

        self._generation += 1
        self._missing_generation += 1
        self._refresh_timer.stop()
        self._file_worker = None
        self._missing_worker = None
        self._workers.clear()
