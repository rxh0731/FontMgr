"""图片实验室独立工作台。"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

from PySide6.QtCore import QEvent, QPoint, QRect, QSize, QThreadPool, QTimer, Qt, Signal
from PySide6.QtGui import QCursor, QIcon, QKeyEvent, QKeySequence, QShortcut, QWheelEvent
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.image_cleanup import (
    IMAGE_CLEANUP_ALGORITHM_VERSION,
    PROCESSING_MODE_AUTO,
    PROCESSING_MODE_GENERAL,
    PROCESSING_MODE_RUBBING,
    ImageCleanupOptions,
)
from data.image_lab_project_store import (
    IMAGE_LAB_PROJECT_EXTENSION,
    ImageLabProject,
    ImageLabProjectStore,
    ImageLabStroke,
)
from data.log_manager import write_log
from services.image_lab_service import (
    SUPPORTED_IMAGE_FILTER,
    ImageLabDetailPreview,
    ImageLabExportResult,
    ImageLabPreview,
    ImageLabService,
)
from ui.widgets.image_lab_canvas import (
    VIEW_CLEAN,
    VIEW_LAYER,
    VIEW_ORIGINAL,
    VIEW_REVIEW,
    ImageLabCanvas,
)
from ui.workers import FunctionWorker


class ImageLabPreviewScrollArea(QScrollArea):
    """在滚动区域自身的滚轮入口处理图片实验室缩放。"""

    zoom_wheel_requested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._alt_state_provider = lambda: False

    def set_alt_state_provider(self, provider) -> None:  # type: ignore[no-untyped-def]
        self._alt_state_provider = provider

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        modifiers = event.modifiers() | QApplication.queryKeyboardModifiers()
        alt_active = bool(modifiers & Qt.KeyboardModifier.AltModifier)
        try:
            alt_active = alt_active or bool(self._alt_state_provider())
        except (AttributeError, RuntimeError):
            pass
        delta = ImageLabCanvas.wheel_delta(event)
        if alt_active and delta != 0:
            write_log(
                "图片实验室输入诊断｜事件=滚动区域wheelEvent"
                f"｜修饰键={event.modifiers()}"
                f"｜角度增量=({event.angleDelta().x()},{event.angleDelta().y()})"
                f"｜像素增量=({event.pixelDelta().x()},{event.pixelDelta().y()})"
                f"｜采用增量={delta}"
            )
            self.zoom_wheel_requested.emit(delta)
            event.accept()
            return
        super().wheelEvent(event)


def _standard_zoom_icon(name: str) -> QIcon:
    """读取系统主题提供的标准缩放图标，不自行绘制图形。"""

    for theme_name in (
        name,
        f"{name}-symbolic",
        f"gtk-{name}",
    ):
        icon = QIcon.fromTheme(theme_name)
        if not icon.isNull():
            return icon
    return QIcon()


class ImageLabPage(QWidget):
    """面向整幅文献图片的非破坏清理页面。"""

    home_requested = Signal()
    status_message = Signal(str)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        service: ImageLabService | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("imageLabPage")
        self._service = service or ImageLabService()
        self._store: ImageLabProjectStore = self._service.store
        self._thread_pool = QThreadPool.globalInstance()
        self._project: ImageLabProject | None = None
        self._preview: ImageLabPreview | None = None
        self._preview_generation = 0
        self._preview_worker: FunctionWorker | None = None
        self._detail_generation = 0
        self._detail_worker: FunctionWorker | None = None
        self._detail_pending: tuple[
            int,
            ImageLabProject,
            ImageLabPreview,
            tuple[int, int, int, int],
            tuple[int, int],
        ] | None = None
        self._detail_loading = False
        self._detail_timer = QTimer(self)
        self._detail_timer.setSingleShot(True)
        self._detail_timer.setInterval(180)
        self._detail_timer.timeout.connect(self._request_detail_preview)
        self._export_worker: FunctionWorker | None = None
        self._cancel_event = threading.Event()
        self._dirty = False
        self._alt_zoom_held = False
        self._input_diagnostic_budget = 40
        self._application_filter_installed = False
        self._build_ui()
        self._connect_shortcuts()
        self._set_project_available(False)

    @property
    def is_running(self) -> bool:
        return self._preview_worker is not None or self._export_worker is not None

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 16)
        root.setSpacing(12)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title_box.setSpacing(1)
        title = QLabel("图片实验室")
        title.setProperty("role", "pageTitle")
        subtitle = QLabel("整幅文献图片的非破坏背景清理与人工修补")
        subtitle.setProperty("role", "muted")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch(1)
        self._project_label = QLabel("未打开图片")
        self._project_label.setObjectName("imageLabProjectLabel")
        header.addWidget(self._project_label)
        self._open_image_button = QPushButton("打开图片")
        self._open_project_button = QPushButton("打开项目")
        self._save_button = QPushButton("保存项目")
        self._save_button.setObjectName("primaryButton")
        header.addWidget(self._open_image_button)
        header.addWidget(self._open_project_button)
        header.addWidget(self._save_button)
        self._home_button = QPushButton("返回首页")
        self._home_button.setObjectName("secondaryButton")
        self._home_button.clicked.connect(self._request_home)
        header.addWidget(self._home_button)
        self._open_image_button.clicked.connect(self._choose_image)
        self._open_project_button.clicked.connect(self._choose_project)
        self._save_button.clicked.connect(self.save_project)
        root.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_controls())
        splitter.addWidget(self._build_preview_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([310, 1030])
        root.addWidget(splitter, 1)

        footer = QHBoxLayout()
        self._status_label = QLabel("打开图片后可生成清理预览")
        self._status_label.setObjectName("statusLabel")
        self._progress = QProgressBar()
        self._progress.setMinimumWidth(260)
        self._progress.setTextVisible(True)
        self._progress.hide()
        self._stop_button = QPushButton("停止")
        self._stop_button.clicked.connect(self._stop_export)
        self._stop_button.hide()
        footer.addWidget(self._status_label, 1)
        footer.addWidget(self._progress)
        footer.addWidget(self._stop_button)
        root.addLayout(footer)

    def _build_controls(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setObjectName("imageLabControls")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(292)
        scroll.setMaximumWidth(370)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(4, 4, 8, 4)
        layout.setSpacing(10)

        process_group = QGroupBox("智能清理")
        process_layout = QVBoxLayout(process_group)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("处理方式"))
        self._processing_mode = QComboBox()
        self._processing_mode.addItem("自动识别（推荐）", PROCESSING_MODE_AUTO)
        self._processing_mode.addItem("通用彩色文献", PROCESSING_MODE_GENERAL)
        self._processing_mode.addItem("拓片深色文字", PROCESSING_MODE_RUBBING)
        mode_row.addWidget(self._processing_mode, 1)
        process_layout.addLayout(mode_row)
        strength_row = QHBoxLayout()
        strength_row.addWidget(QLabel("清理强度"))
        self._strength_value = QLabel("50")
        strength_row.addStretch(1)
        strength_row.addWidget(self._strength_value)
        process_layout.addLayout(strength_row)
        self._strength_slider = QSlider(Qt.Orientation.Horizontal)
        self._strength_slider.setRange(0, 100)
        self._strength_slider.setValue(50)
        self._strength_slider.valueChanged.connect(
            lambda value: self._strength_value.setText(str(value))
        )
        process_layout.addWidget(self._strength_slider)
        self._preserve_faint = QCheckBox("保护浅色和残损笔迹")
        self._preserve_faint.setChecked(True)
        self._remove_noise = QCheckBox("清除孤立小噪点")
        self._remove_noise.setChecked(True)
        process_layout.addWidget(self._preserve_faint)
        process_layout.addWidget(self._remove_noise)
        self._feather_edges = QCheckBox("羽化清理边缘")
        self._feather_edges.setChecked(True)
        process_layout.addWidget(self._feather_edges)
        self._apply_button = QPushButton("重新生成预览")
        self._apply_button.setObjectName("primaryButton")
        self._apply_button.clicked.connect(self._apply_options)
        process_layout.addWidget(self._apply_button)
        layout.addWidget(process_group)

        manual_group = QGroupBox("人工引导")
        manual_layout = QVBoxLayout(manual_group)
        tool_row = QHBoxLayout()
        self._cover_button = QPushButton("清除背景")
        self._cover_button.setCheckable(True)
        self._cover_button.setChecked(True)
        self._restore_button = QPushButton("保护文字")
        self._restore_button.setCheckable(True)
        tool_group = QButtonGroup(self)
        tool_group.setExclusive(True)
        tool_group.addButton(self._cover_button)
        tool_group.addButton(self._restore_button)
        self._cover_button.clicked.connect(lambda: self._canvas.set_tool("cover"))
        self._restore_button.clicked.connect(lambda: self._canvas.set_tool("restore"))
        tool_row.addWidget(self._cover_button)
        tool_row.addWidget(self._restore_button)
        manual_layout.addLayout(tool_row)
        brush_row = QHBoxLayout()
        brush_row.addWidget(QLabel("笔触大小"))
        self._brush_value = QLabel("80 像素")
        brush_row.addStretch(1)
        brush_row.addWidget(self._brush_value)
        manual_layout.addLayout(brush_row)
        self._brush_slider = QSlider(Qt.Orientation.Horizontal)
        self._brush_slider.setRange(5, 500)
        self._brush_slider.setValue(80)
        self._brush_slider.valueChanged.connect(self._brush_changed)
        manual_layout.addWidget(self._brush_slider)
        edit_row = QHBoxLayout()
        self._undo_button = QPushButton("撤销")
        self._clear_button = QPushButton("清除人工修改")
        self._undo_button.clicked.connect(self._undo_stroke)
        self._clear_button.clicked.connect(self._clear_strokes)
        edit_row.addWidget(self._undo_button)
        edit_row.addWidget(self._clear_button)
        manual_layout.addLayout(edit_row)
        layout.addWidget(manual_group)

        metrics_group = QGroupBox("处理摘要")
        metrics_layout = QVBoxLayout(metrics_group)
        self._metrics_label = QLabel("尚未生成预览")
        self._metrics_label.setWordWrap(True)
        self._metrics_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        metrics_layout.addWidget(self._metrics_label)
        layout.addWidget(metrics_group)

        export_group = QGroupBox("完整尺寸导出")
        export_layout = QVBoxLayout(export_group)
        self._export_photoshop_button = QPushButton("导出 Photoshop 文件")
        self._export_result_button = QPushButton("导出清理效果")
        self._export_layer_button = QPushButton("导出白色清理层")
        self._export_photoshop_button.setObjectName("primaryButton")
        self._export_photoshop_button.clicked.connect(
            lambda: self._choose_export("photoshop")
        )
        self._export_result_button.clicked.connect(
            lambda: self._choose_export("composite")
        )
        self._export_layer_button.clicked.connect(
            lambda: self._choose_export("layer")
        )
        export_layout.addWidget(self._export_photoshop_button)
        export_layout.addWidget(self._export_result_button)
        export_layout.addWidget(self._export_layer_button)
        layout.addWidget(export_group)
        layout.addStretch(1)
        scroll.setWidget(body)
        return scroll

    def _build_preview_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("imageLabPreviewPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        toolbar = QHBoxLayout()
        self._view_buttons: dict[str, QRadioButton] = {}
        view_group = QButtonGroup(self)
        for mode, label in (
            (VIEW_ORIGINAL, "原稿"),
            (VIEW_CLEAN, "清理效果"),
            (VIEW_LAYER, "白色清理层"),
            (VIEW_REVIEW, "待核对区域"),
        ):
            button = QRadioButton(label)
            button.setObjectName("segmentedButton")
            button.clicked.connect(
                lambda _checked=False, selected=mode: self._set_view_mode(selected)
            )
            view_group.addButton(button)
            self._view_buttons[mode] = button
            toolbar.addWidget(button)
        self._view_buttons[VIEW_CLEAN].setChecked(True)
        toolbar.addStretch(1)
        self._zoom_in_button = QToolButton()
        self._zoom_in_button.setObjectName("imageLabZoomButton")
        self._zoom_in_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonIconOnly
        )
        self._zoom_in_button.setIcon(_standard_zoom_icon("zoom-in"))
        self._zoom_in_button.setIconSize(QSize(20, 20))
        self._zoom_in_button.setFixedSize(40, 40)
        self._zoom_in_button.setToolTip("放大")
        self._zoom_in_button.setAccessibleName("放大")
        self._zoom_in_button.clicked.connect(self._zoom_in)
        toolbar.addWidget(self._zoom_in_button)
        self._zoom_out_button = QToolButton()
        self._zoom_out_button.setObjectName("imageLabZoomButton")
        self._zoom_out_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonIconOnly
        )
        self._zoom_out_button.setIcon(_standard_zoom_icon("zoom-out"))
        self._zoom_out_button.setIconSize(QSize(20, 20))
        self._zoom_out_button.setFixedSize(40, 40)
        self._zoom_out_button.setToolTip("缩小")
        self._zoom_out_button.setAccessibleName("缩小")
        self._zoom_out_button.clicked.connect(self._zoom_out)
        toolbar.addWidget(self._zoom_out_button)
        self._fit_button = QPushButton("适合窗口")
        self._fit_button.setFixedSize(40, 40)
        self._zoom_label = QLabel("100%")
        self._fit_button.clicked.connect(self._fit_canvas)
        toolbar.addWidget(self._fit_button)
        toolbar.addWidget(self._zoom_label)
        panel.setStyleSheet(
            "QToolButton#imageLabZoomButton {"
            " padding: 0; border: 1px solid #37404d;"
            " border-radius: 6px; background: #282f3a; }"
            "QToolButton#imageLabZoomButton:hover {"
            " border-color: #4da3ff; background: #303947; }"
            "QToolButton#imageLabZoomButton:pressed { background: #1b75d0; }"
            "QToolButton#imageLabZoomButton:disabled {"
            " background: #242a33; border-color: #303640; }"
        )
        layout.addLayout(toolbar)

        self._canvas_scroll = ImageLabPreviewScrollArea()
        self._canvas_scroll.setObjectName("imageLabPreviewScroll")
        self._canvas_scroll.set_alt_state_provider(
            lambda: self._alt_zoom_held or self._windows_alt_key_is_down()
        )
        self._canvas_scroll.zoom_wheel_requested.connect(
            self._zoom_from_scroll_area
        )
        self._canvas_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._canvas_scroll.setWidgetResizable(False)
        self._canvas = ImageLabCanvas()
        self._canvas.stroke_finished.connect(self._stroke_finished)
        self._canvas.zoom_changed.connect(self._canvas_zoom_changed)
        self._canvas.pan_requested.connect(self._pan_canvas)
        self._canvas_scroll.setWidget(self._canvas)
        self._canvas_scroll.horizontalScrollBar().valueChanged.connect(
            self._schedule_detail_preview
        )
        self._canvas_scroll.verticalScrollBar().valueChanged.connect(
            self._schedule_detail_preview
        )
        self._canvas_scroll.horizontalScrollBar().rangeChanged.connect(
            self._schedule_detail_preview
        )
        self._canvas_scroll.verticalScrollBar().rangeChanged.connect(
            self._schedule_detail_preview
        )
        self._canvas_scroll.viewport().installEventFilter(self)
        layout.addWidget(self._canvas_scroll, 1)
        hint = QLabel("普通滚轮滚屏，Alt+滚轮缩放；空格+鼠标左键拖动画布")
        hint.setObjectName("subtleLabel")
        layout.addWidget(hint)
        return panel

    def _connect_shortcuts(self) -> None:
        save_shortcut = QShortcut(QKeySequence.StandardKey.Save, self)
        save_shortcut.activated.connect(self.save_project)
        undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        undo_shortcut.activated.connect(self._undo_stroke)

    def _create_file_dialog(
        self,
        title: str,
        filter_spec: str,
        *,
        save: bool,
        suggested_path: str = "",
        default_suffix: str = "",
    ) -> QFileDialog:
        """创建按钮和字段均为中文的图片实验室文件对话框。"""

        dialog = QFileDialog(self)
        dialog.setWindowTitle(title)
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dialog.setNameFilters(
            [item for item in filter_spec.split(";;") if item]
        )
        dialog.setAcceptMode(
            QFileDialog.AcceptMode.AcceptSave
            if save
            else QFileDialog.AcceptMode.AcceptOpen
        )
        dialog.setFileMode(
            QFileDialog.FileMode.AnyFile
            if save
            else QFileDialog.FileMode.ExistingFile
        )
        dialog.setLabelText(
            QFileDialog.DialogLabel.LookIn,
            "查找范围",
        )
        dialog.setLabelText(QFileDialog.DialogLabel.FileName, "文件名")
        dialog.setLabelText(QFileDialog.DialogLabel.FileType, "文件类型")
        dialog.setLabelText(
            QFileDialog.DialogLabel.Accept,
            "保存" if save else "打开",
        )
        dialog.setLabelText(QFileDialog.DialogLabel.Reject, "取消")
        if suggested_path:
            dialog.setDirectory(os.path.dirname(suggested_path))
            dialog.selectFile(os.path.basename(suggested_path))
        if default_suffix:
            dialog.setDefaultSuffix(default_suffix.lstrip("."))
        return dialog

    def _select_file(
        self,
        title: str,
        filter_spec: str,
        *,
        save: bool,
        suggested_path: str = "",
        default_suffix: str = "",
    ) -> str:
        dialog = self._create_file_dialog(
            title,
            filter_spec,
            save=save,
            suggested_path=suggested_path,
            default_suffix=default_suffix,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return ""
        selected = dialog.selectedFiles()
        return selected[0] if selected else ""

    def _show_message(
        self,
        icon: QMessageBox.Icon,
        title: str,
        text: str,
    ) -> None:
        dialog = QMessageBox(self)
        dialog.setIcon(icon)
        dialog.setWindowTitle(title)
        dialog.setText(text)
        confirm_button = dialog.addButton(
            "确定",
            QMessageBox.ButtonRole.AcceptRole,
        )
        dialog.setDefaultButton(confirm_button)
        dialog.exec()

    def _choose_image(self) -> None:
        if not self._confirm_replace_project():
            return
        path = self._select_file(
            "打开待处理图片",
            SUPPORTED_IMAGE_FILTER,
            save=False,
        )
        if not path:
            return
        try:
            project = self._service.create_project(path)
        except (OSError, RuntimeError, ValueError) as exc:
            self._show_message(
                QMessageBox.Icon.Warning,
                "打开失败",
                f"无法打开图片：{exc}",
            )
            return
        self._set_project(project, dirty=True)

    def _choose_project(self) -> None:
        if not self._confirm_replace_project():
            return
        path = self._select_file(
            "打开图片实验室项目",
            f"图片实验室项目 (*{IMAGE_LAB_PROJECT_EXTENSION})",
            save=False,
        )
        if not path:
            return
        try:
            project = self._store.load(path)
        except (OSError, RuntimeError, ValueError) as exc:
            self._show_message(
                QMessageBox.Icon.Warning,
                "打开失败",
                f"无法打开项目：{exc}",
            )
            return
        self._set_project(project, dirty=False)

    def _set_project(self, project: ImageLabProject, *, dirty: bool) -> None:
        self._invalidate_detail_preview()
        self._project = project
        self._preview = None
        self._dirty = dirty
        self._strength_slider.setValue(project.options.strength)
        self._preserve_faint.setChecked(project.options.preserve_faint_ink)
        self._remove_noise.setChecked(project.options.remove_small_noise)
        self._feather_edges.setChecked(project.options.feather_edges)
        mode_index = self._processing_mode.findData(project.options.processing_mode)
        self._processing_mode.setCurrentIndex(max(0, mode_index))
        self._project_label.setText(self._project_title())
        self._set_project_available(True)
        self._start_preview()

    def _project_title(self) -> str:
        if self._project is None:
            return "未打开图片"
        suffix = " *" if self._dirty else ""
        return f"{self._project.display_name}{suffix}"

    def _apply_options(self) -> None:
        if self._project is None:
            return
        self._project.options = ImageCleanupOptions(
            strength=self._strength_slider.value(),
            preserve_faint_ink=self._preserve_faint.isChecked(),
            remove_small_noise=self._remove_noise.isChecked(),
            feather_edges=self._feather_edges.isChecked(),
            processing_mode=str(self._processing_mode.currentData()),
        )
        self._mark_dirty()
        self._start_preview()

    def _start_preview(self) -> None:
        if self._project is None:
            return
        self._invalidate_detail_preview()
        self._preview_generation += 1
        generation = self._preview_generation
        project = self._project
        detail_source_cache = (
            self._preview.detail_source if self._preview is not None else None
        )
        self._set_busy(True, "正在后台解码原稿并生成预览…", export=False)
        worker = FunctionWorker(
            lambda: self._service.load_preview(
                project,
                detail_source_cache=detail_source_cache,
            )
        )
        self._preview_worker = worker
        worker.signals.finished.connect(
            lambda result, token=generation, task=worker: self._preview_finished(
                token, task, result
            )
        )
        worker.signals.failed.connect(
            lambda message, token=generation, task=worker: self._preview_failed(
                token, task, message
            )
        )
        self._thread_pool.start(worker)

    def _preview_finished(
        self,
        generation: int,
        worker: FunctionWorker,
        result: object,
    ) -> None:
        if worker is self._preview_worker:
            self._preview_worker = None
        if generation != self._preview_generation or not isinstance(result, ImageLabPreview):
            return
        preserve_view = self._preview is not None and self._canvas.has_image
        previous_zoom = self._canvas.zoom_factor
        horizontal_value = self._canvas_scroll.horizontalScrollBar().value()
        vertical_value = self._canvas_scroll.verticalScrollBar().value()
        self._preview = result
        if self._project is not None:
            self._project.algorithm_version = IMAGE_CLEANUP_ALGORITHM_VERSION
            self._project.resolved_profile = result.cleanup.resolved_profile
        self._canvas.set_preview(
            result.source,
            result.composite,
            result.effective_alpha,
            result.cleanup.uncertainty_mask,
            source_width=result.source_width,
            source_height=result.source_height,
        )
        metrics = result.cleanup.metrics
        adaptive_summary = ""
        if metrics.get("局部自适应") == "是":
            unevenness = float(metrics.get("背景不均匀指数", 0.0))
            adaptive_summary = (
                "\n局部自适应：已启用"
                + ("（检测到背景不均匀）" if unevenness >= 0.03 else "")
            )
        self._metrics_label.setText(
            f"识别方式：{result.cleanup.resolved_profile}\n"
            f"算法版本：{metrics.get('算法版本', IMAGE_CLEANUP_ALGORITHM_VERSION)}\n"
            f"原稿尺寸：{result.source_width} × {result.source_height}\n"
            f"保留前景：{float(metrics['保留前景占比']) * 100:.1f}%\n"
            f"完全清理：{float(metrics['完全清理占比']) * 100:.1f}%\n"
            f"待核对：{float(metrics['待核对占比']) * 100:.1f}%\n"
            f"边缘羽化：{'开启' if self._project and self._project.options.feather_edges else '关闭'}"
            f"{adaptive_summary}"
        )
        self._set_busy(False, f"预览已生成，用时 {result.elapsed_seconds:.2f} 秒")
        if preserve_view:
            def restore_view() -> None:
                self._restore_canvas_view(
                    generation,
                    previous_zoom,
                    horizontal_value,
                    vertical_value,
                )

            QTimer.singleShot(
                0,
                restore_view,
            )
        else:
            QTimer.singleShot(0, self._fit_canvas)

    def _preview_failed(
        self,
        generation: int,
        worker: FunctionWorker,
        message: str,
    ) -> None:
        if worker is self._preview_worker:
            self._preview_worker = None
        if generation != self._preview_generation:
            return
        self._set_busy(False, "预览生成失败")
        self._show_message(
            QMessageBox.Icon.Warning,
            "预览失败",
            f"无法生成清理预览：{message}",
        )

    def _stroke_finished(self, tool: str, width: float, points: object) -> None:
        if self._project is None or self._preview is None:
            return
        try:
            stroke = ImageLabStroke(tool, width, tuple(points))
        except (TypeError, ValueError):
            return
        self._project.strokes.append(stroke)
        self._mark_dirty()
        self._refresh_manual_preview()

    def _refresh_manual_preview(self) -> None:
        if self._project is None or self._preview is None:
            return
        self._invalidate_detail_preview()
        alpha = self._service.apply_strokes(
            self._preview.cleanup.cleanup_layer[:, :, 3],
            self._project.strokes,
            self._project.source_width,
            self._project.source_height,
        )
        composite = self._service.compose(self._preview.source, alpha)
        self._canvas.set_preview(
            self._preview.source,
            composite,
            alpha,
            self._preview.cleanup.uncertainty_mask,
            source_width=self._project.source_width,
            source_height=self._project.source_height,
        )
        self._schedule_detail_preview()
        self._undo_button.setEnabled(bool(self._project.strokes))
        self._clear_button.setEnabled(bool(self._project.strokes))

    def _undo_stroke(self) -> None:
        if self._project is None or not self._project.strokes:
            return
        self._project.strokes.pop()
        self._mark_dirty()
        self._refresh_manual_preview()

    def _clear_strokes(self) -> None:
        if self._project is None or not self._project.strokes:
            return
        self._project.strokes.clear()
        self._mark_dirty()
        self._refresh_manual_preview()

    def _brush_changed(self, value: int) -> None:
        self._brush_value.setText(f"{value} 像素")
        self._canvas.set_brush_width(float(value))

    def _set_view_mode(self, mode: str) -> None:
        self._canvas.set_view_mode(mode)
        self._schedule_detail_preview()

    def _canvas_zoom_changed(self, value: int) -> None:
        self._zoom_label.setText(f"{value}%")
        if self._canvas.zoom_factor <= 1.01:
            self._invalidate_detail_preview()
        else:
            QTimer.singleShot(0, self._schedule_detail_preview)

    def _schedule_detail_preview(self, *_args: object) -> None:
        if self._project is None or self._preview is None:
            return
        if self._preview_worker is not None or self._export_worker is not None:
            return
        self._detail_timer.start()

    def _request_detail_preview(self) -> None:
        project = self._project
        preview = self._preview
        if project is None or preview is None:
            return
        if not self._canvas.has_reduced_preview or self._canvas.zoom_factor <= 1.01:
            self._invalidate_detail_preview()
            return
        viewport = self._canvas_scroll.viewport()
        canvas_origin = self._canvas.mapFrom(viewport, QPoint(0, 0))
        visible = QRect(canvas_origin, viewport.size()).intersected(self._canvas.rect())
        if visible.isEmpty():
            return
        visible_source = self._canvas.source_rect_for_canvas_rect(visible)
        display_scale = min(
            self._canvas.width() / project.source_width,
            self._canvas.height() / project.source_height,
        )
        desired_detail_scale = max(display_scale * 1.25, display_scale)
        if self._canvas.detail_covers(visible_source, desired_detail_scale * 0.9):
            return

        padded = visible.adjusted(-160, -160, 160, 160).intersected(
            self._canvas.rect()
        )
        source_rect = self._canvas.source_rect_for_canvas_rect(padded)
        source_width = source_rect[2] - source_rect[0]
        source_height = source_rect[3] - source_rect[1]
        target_size = (
            max(1, int(round(source_width * desired_detail_scale))),
            max(1, int(round(source_height * desired_detail_scale))),
        )
        self._detail_generation += 1
        request = (
            self._detail_generation,
            project,
            preview,
            source_rect,
            target_size,
        )
        if self._detail_worker is not None:
            self._detail_pending = request
            self._show_detail_loading()
            return
        self._start_detail_worker(request)

    def _start_detail_worker(
        self,
        request: tuple[
            int,
            ImageLabProject,
            ImageLabPreview,
            tuple[int, int, int, int],
            tuple[int, int],
        ],
    ) -> None:
        generation, project, preview, source_rect, target_size = request
        self._show_detail_loading()
        worker = FunctionWorker(
            lambda: self._service.load_detail_preview(
                project,
                preview,
                source_rect,
                target_size,
            )
        )
        self._detail_worker = worker
        worker.signals.finished.connect(
            lambda result, token=generation, task=worker: self._detail_finished(
                token,
                task,
                result,
            )
        )
        worker.signals.failed.connect(
            lambda message, token=generation, task=worker: self._detail_failed(
                token,
                task,
                message,
            )
        )
        self._thread_pool.start(worker)

    def _detail_finished(
        self,
        generation: int,
        worker: FunctionWorker,
        result: object,
    ) -> None:
        if worker is self._detail_worker:
            self._detail_worker = None
        if generation == self._detail_generation and isinstance(
            result,
            ImageLabDetailPreview,
        ):
            self._canvas.set_detail_preview(
                result.source,
                result.composite,
                result.effective_alpha,
                result.uncertainty,
                result.source_rect,
            )
        self._start_pending_detail_request()
        if self._detail_worker is None:
            elapsed = (
                f"，用时 {result.elapsed_seconds:.2f} 秒"
                if isinstance(result, ImageLabDetailPreview)
                else ""
            )
            self._hide_detail_loading(f"高清区域已加载{elapsed}")

    def _detail_failed(
        self,
        generation: int,
        worker: FunctionWorker,
        _message: str,
    ) -> None:
        if worker is self._detail_worker:
            self._detail_worker = None
        if generation == self._detail_generation:
            message = "高清区域生成失败，已继续使用快速预览"
        else:
            message = ""
        self._start_pending_detail_request()
        if self._detail_worker is None:
            self._hide_detail_loading(message)

    def _start_pending_detail_request(self) -> None:
        request = self._detail_pending
        self._detail_pending = None
        if request is not None and request[0] == self._detail_generation:
            self._start_detail_worker(request)

    def _invalidate_detail_preview(self) -> None:
        self._detail_timer.stop()
        self._detail_generation += 1
        self._detail_pending = None
        self._hide_detail_loading()
        if hasattr(self, "_canvas"):
            self._canvas.clear_detail_preview()

    def _fit_canvas(self) -> None:
        viewport = self._canvas_scroll.viewport().size()
        self._canvas.fit_to_size(viewport.width(), viewport.height())
        self._schedule_detail_preview()

    def _zoom_in(self) -> None:
        if self._canvas.zoom_in():
            self._write_input_diagnostic(
                f"事件=点击放大｜缩放后={self._canvas.zoom_percent}%"
            )

    def _zoom_out(self) -> None:
        if self._canvas.zoom_out():
            self._write_input_diagnostic(
                f"事件=点击缩小｜缩放后={self._canvas.zoom_percent}%"
            )

    def _restore_canvas_view(
        self,
        generation: int,
        zoom: float,
        horizontal: int,
        vertical: int,
    ) -> None:
        """预览刷新后恢复用户原有缩放和滚动位置。"""

        if generation != self._preview_generation or self._preview is None:
            return
        self._canvas.set_zoom(zoom)
        self._canvas_scroll.horizontalScrollBar().setValue(horizontal)
        self._canvas_scroll.verticalScrollBar().setValue(vertical)
        self._schedule_detail_preview()

    def _pan_canvas(self, delta: QPoint) -> None:
        """把抓手位移转换为滚动区域偏移，不修改图片数据。"""

        horizontal = self._canvas_scroll.horizontalScrollBar()
        vertical = self._canvas_scroll.verticalScrollBar()
        horizontal.setValue(horizontal.value() - delta.x())
        vertical.setValue(vertical.value() - delta.y())

    def _zoom_from_scroll_area(self, delta: int) -> None:
        before = self._canvas.zoom_percent
        if self._canvas.zoom_by_wheel_delta(delta):
            self._write_input_diagnostic(
                "结果=滚动区域已缩放"
                f"｜缩放前={before}%｜缩放后={self._canvas.zoom_percent}%"
            )

    def _install_application_event_filter(self) -> None:
        if self._application_filter_installed:
            return
        application = QApplication.instance()
        if application is not None:
            application.installEventFilter(self)
            self._application_filter_installed = True

    def _remove_application_event_filter(self) -> None:
        if not self._application_filter_installed:
            return
        application = QApplication.instance()
        if application is not None:
            application.removeEventFilter(self)
        self._application_filter_installed = False

    def _owns_event_target(self, watched: object) -> bool:
        return watched is self or (
            isinstance(watched, QWidget) and self.isAncestorOf(watched)
        )

    def _cursor_is_over_canvas(self) -> bool:
        position = self._canvas.mapFromGlobal(QCursor.pos())
        return self._canvas.rect().contains(position)

    def _wheel_is_over_preview(self, event: QWheelEvent) -> bool:
        viewport = self._canvas_scroll.viewport()
        event_position = viewport.mapFromGlobal(event.globalPosition().toPoint())
        cursor_position = viewport.mapFromGlobal(QCursor.pos())
        return viewport.rect().contains(event_position) or viewport.rect().contains(
            cursor_position
        )

    @staticmethod
    def _windows_alt_key_is_down() -> bool:
        if sys.platform != "win32":
            return False
        try:
            import ctypes

            return bool(ctypes.windll.user32.GetAsyncKeyState(0x12) & 0x8000)
        except (AttributeError, OSError):
            return False

    def _alt_zoom_is_active(self, event: QWheelEvent) -> bool:
        modifiers = event.modifiers() | QApplication.queryKeyboardModifiers()
        return (
            bool(modifiers & Qt.KeyboardModifier.AltModifier)
            or self._alt_zoom_held
            or self._windows_alt_key_is_down()
        )

    def _write_input_diagnostic(self, message: str) -> None:
        if self._input_diagnostic_budget <= 0:
            return
        self._input_diagnostic_budget -= 1
        write_log(f"图片实验室输入诊断｜{message}")

    def eventFilter(self, watched: object, event: QEvent) -> bool:  # noqa: N802
        if hasattr(self, "_canvas_scroll") and isinstance(event, QWheelEvent):
            over_preview = self._wheel_is_over_preview(event)
            alt_active = self._alt_zoom_is_active(event)
            angle_delta = event.angleDelta()
            pixel_delta = event.pixelDelta()
            wheel_delta = ImageLabCanvas.wheel_delta(event)
            before_zoom = self._canvas.zoom_percent
            receiver_name = type(watched).__name__
            object_name = watched.objectName() if isinstance(watched, QWidget) else ""
            self._write_input_diagnostic(
                "事件=滚轮"
                f"｜接收控件={receiver_name}:{object_name or '无名称'}"
                f"｜事件修饰键={event.modifiers()}"
                f"｜当前修饰键={QApplication.queryKeyboardModifiers()}"
                f"｜Alt记录={self._alt_zoom_held}"
                f"｜WindowsAlt={self._windows_alt_key_is_down()}"
                f"｜角度增量=({angle_delta.x()},{angle_delta.y()})"
                f"｜像素增量=({pixel_delta.x()},{pixel_delta.y()})"
                f"｜采用增量={wheel_delta}"
                f"｜位于预览区={over_preview}｜缩放前={before_zoom}%"
            )
        else:
            over_preview = False
            alt_active = False
            wheel_delta = 0
            before_zoom = 0
        if (
            hasattr(self, "_canvas_scroll")
            and isinstance(event, QWheelEvent)
            and over_preview
            and alt_active
        ):
            if self._canvas.zoom_by_wheel_delta(wheel_delta):
                self._write_input_diagnostic(
                    "结果=已缩放"
                    f"｜缩放前={before_zoom}%｜缩放后={self._canvas.zoom_percent}%"
                )
                event.accept()
                return True
            self._write_input_diagnostic("结果=未缩放｜原因=滚轮增量为零或尚未打开图片")
        if (
            hasattr(self, "_canvas_scroll")
            and watched is self._canvas_scroll.viewport()
            and event.type() == QEvent.Type.Resize
        ):
            self._schedule_detail_preview()
        if event.type() == QEvent.Type.ApplicationDeactivate:
            self._alt_zoom_held = False
            self._canvas.cancel_pan()
        if (
            isinstance(event, QKeyEvent)
            and event.key() == Qt.Key.Key_Alt
            and not event.isAutoRepeat()
        ):
            if event.type() == QEvent.Type.KeyPress:
                self._alt_zoom_held = True
                self._write_input_diagnostic(
                    f"事件=Alt按下｜接收对象={type(watched).__name__}"
                )
            elif event.type() == QEvent.Type.KeyRelease:
                self._alt_zoom_held = False
                self._write_input_diagnostic(
                    f"事件=Alt松开｜接收对象={type(watched).__name__}"
                )
        if (
            isinstance(event, QKeyEvent)
            and event.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease)
            and event.key() == Qt.Key.Key_Space
            and self._owns_event_target(watched)
            and watched is not self._canvas
            and (
                self._canvas.space_pan_active
                or self._cursor_is_over_canvas()
            )
        ):
            if event.isAutoRepeat():
                handled = self._canvas.space_pan_active
            else:
                handled = self._canvas.set_pan_modifier_active(
                    event.type() == QEvent.Type.KeyPress
                )
            if handled:
                event.accept()
                return True
        return super().eventFilter(watched, event)

    def showEvent(self, event) -> None:  # type: ignore[no-untyped-def]  # noqa: N802
        super().showEvent(event)
        self._install_application_event_filter()
        self._write_input_diagnostic(
            f"事件=页面显示｜应用事件过滤器={self._application_filter_installed}"
        )

    def hideEvent(self, event) -> None:  # type: ignore[no-untyped-def]  # noqa: N802
        self._alt_zoom_held = False
        self._canvas.cancel_pan()
        self._remove_application_event_filter()
        super().hideEvent(event)

    def save_project(self) -> bool:
        if self._project is None:
            return False
        path = self._project.project_path
        if not path:
            suggested = os.path.join(
                os.path.dirname(self._project.source_path),
                f"{self._project.display_name}{IMAGE_LAB_PROJECT_EXTENSION}",
            )
            path = self._select_file(
                "保存图片实验室项目",
                f"图片实验室项目 (*{IMAGE_LAB_PROJECT_EXTENSION})",
                save=True,
                suggested_path=suggested,
                default_suffix=IMAGE_LAB_PROJECT_EXTENSION,
            )
            if not path:
                return False
        try:
            self._store.save(self._project, path)
        except (OSError, RuntimeError, ValueError) as exc:
            self._show_message(
                QMessageBox.Icon.Warning,
                "保存失败",
                f"无法保存项目：{exc}",
            )
            return False
        self._dirty = False
        self._project_label.setText(self._project_title())
        self._status("项目已保存")
        return True

    def _choose_export(self, kind: str) -> None:
        if self._project is None or self._preview is None or self.is_running:
            return
        if kind == "photoshop":
            suffix = "预处理"
            extension = ".psd"
            file_filter = "Photoshop 文件 (*.psd *.psb)"
        elif kind == "composite":
            suffix = "清理效果"
            extension = ".tif"
            file_filter = "TIFF 图片 (*.tif *.tiff);;PNG 图片 (*.png)"
        else:
            suffix = "白色清理层"
            extension = ".tif"
            file_filter = "TIFF 图片 (*.tif *.tiff);;PNG 图片 (*.png)"
        suggested = os.path.join(
            os.path.dirname(self._project.source_path),
            f"{self._project.display_name}_{suffix}{extension}",
        )
        path = self._select_file(
            f"导出{suffix}",
            file_filter,
            save=True,
            suggested_path=suggested,
            default_suffix=extension,
        )
        if not path:
            return
        if not Path(path).suffix:
            path += extension
        self._start_export(path, kind)

    def _start_export(self, path: str, kind: str) -> None:
        if self._project is None:
            return
        self._cancel_event.clear()
        project = self._project

        def run(progress_callback):  # type: ignore[no-untyped-def]
            return self._service.export_full_resolution(
                project,
                path,
                kind=kind,
                progress_callback=lambda current, total, message: progress_callback(
                    (current, total, message)
                ),
                cancelled=self._cancel_event.is_set,
            )

        worker = FunctionWorker(run, with_progress=True)
        self._export_worker = worker
        worker.signals.progress.connect(self._export_progress)
        worker.signals.finished.connect(
            lambda result, task=worker: self._export_finished(task, result)
        )
        worker.signals.failed.connect(
            lambda message, task=worker: self._export_failed(task, message)
        )
        self._set_busy(True, "正在生成完整尺寸文件…", export=True)
        self._thread_pool.start(worker)

    def _export_progress(self, payload: object) -> None:
        if not isinstance(payload, tuple) or len(payload) != 3:
            return
        current, total, message = payload
        self._progress.setRange(0, max(1, int(total)))
        self._progress.setValue(int(current))
        self._progress.setFormat(str(message))

    def _export_finished(self, worker: FunctionWorker, result: object) -> None:
        if worker is not self._export_worker:
            return
        self._export_worker = None
        self._set_busy(False, "完整尺寸导出完成")
        if not isinstance(result, ImageLabExportResult):
            self._show_message(
                QMessageBox.Icon.Warning,
                "导出失败",
                "后台任务返回了无效结果。",
            )
            return
        self._show_message(
            QMessageBox.Icon.Information,
            "导出完成",
            f"已生成：{result.output_path}\n"
            f"尺寸：{result.width} × {result.height}\n"
            f"用时：{result.elapsed_seconds:.2f} 秒",
        )

    def _export_failed(self, worker: FunctionWorker, message: str) -> None:
        if worker is not self._export_worker:
            return
        self._export_worker = None
        self._set_busy(False, "导出已停止" if self._cancel_event.is_set() else "导出失败")
        if self._cancel_event.is_set() or "停止" in message:
            self._status("完整尺寸导出已安全停止，未覆盖目标文件")
            return
        self._show_message(
            QMessageBox.Icon.Warning,
            "导出失败",
            f"无法生成完整尺寸文件：{message}",
        )

    def _stop_export(self) -> None:
        if self._export_worker is not None:
            self._cancel_event.set()
            self._stop_button.setEnabled(False)
            self._status("正在等待当前分块安全结束…")

    def _show_detail_loading(self) -> None:
        """显示高清区域后台加载提示，不锁定画布交互。"""

        self._detail_loading = True
        if self._preview_worker is not None or self._export_worker is not None:
            return
        message = "正在加载高清图，请稍候…"
        self._progress.setRange(0, 0)
        self._progress.setFormat(message)
        self._progress.setVisible(True)
        self._stop_button.setVisible(False)
        self._stop_button.setEnabled(False)
        self._status(message)

    def _hide_detail_loading(self, message: str = "") -> None:
        """结束高清提示；若有其他任务，保留其他任务对进度条的控制。"""

        self._detail_loading = False
        if self._preview_worker is None and self._export_worker is None:
            self._progress.hide()
            self._stop_button.hide()
            self._stop_button.setEnabled(False)
        if message:
            self._status(message)

    def _set_busy(self, busy: bool, message: str, *, export: bool = False) -> None:
        self._open_image_button.setEnabled(not busy)
        self._open_project_button.setEnabled(not busy)
        self._apply_button.setEnabled(not busy and self._project is not None)
        self._save_button.setEnabled(not busy and self._project is not None)
        self._export_result_button.setEnabled(not busy and self._preview is not None)
        self._export_layer_button.setEnabled(not busy and self._preview is not None)
        self._export_photoshop_button.setEnabled(not busy and self._preview is not None)
        if busy:
            self._progress.setVisible(True)
            self._progress.setRange(0, 0)
            self._progress.setFormat(message)
            self._stop_button.setVisible(export)
            self._stop_button.setEnabled(export)
        else:
            detail_active = self._detail_loading and (
                self._detail_worker is not None or self._detail_pending is not None
            )
            self._progress.setVisible(detail_active)
            self._stop_button.setVisible(False)
            self._stop_button.setEnabled(False)
            if detail_active:
                self._progress.setRange(0, 0)
                self._progress.setFormat("正在加载高清图，请稍候…")
        self._status(message)

    def _set_project_available(self, available: bool) -> None:
        for widget in (
            self._save_button,
            self._strength_slider,
            self._preserve_faint,
            self._remove_noise,
            self._feather_edges,
            self._apply_button,
            self._cover_button,
            self._restore_button,
            self._brush_slider,
            self._undo_button,
            self._clear_button,
            self._export_result_button,
            self._export_layer_button,
            self._export_photoshop_button,
            self._fit_button,
            self._zoom_in_button,
            self._zoom_out_button,
        ):
            widget.setEnabled(available)
        if available and self._project is not None:
            has_strokes = bool(self._project.strokes)
            self._undo_button.setEnabled(has_strokes)
            self._clear_button.setEnabled(has_strokes)

    def _mark_dirty(self) -> None:
        self._dirty = True
        self._project_label.setText(self._project_title())

    def _status(self, message: str) -> None:
        self._status_label.setText(message)
        self.status_message.emit(message)

    def _confirm_replace_project(self) -> bool:
        if self.is_running:
            self._show_message(
                QMessageBox.Icon.Information,
                "后台任务正在执行",
                "请先等待任务完成或停止导出。",
            )
            return False
        if not self._dirty:
            return True
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Question)
        dialog.setWindowTitle("项目尚未保存")
        dialog.setText("当前图片实验室项目有未保存修改，是否先保存？")
        save_button = dialog.addButton(
            "保存并继续",
            QMessageBox.ButtonRole.AcceptRole,
        )
        discard_button = dialog.addButton(
            "放弃修改",
            QMessageBox.ButtonRole.DestructiveRole,
        )
        cancel_button = dialog.addButton(
            "取消",
            QMessageBox.ButtonRole.RejectRole,
        )
        dialog.setDefaultButton(save_button)
        dialog.exec()
        selected = dialog.clickedButton()
        if selected is cancel_button:
            return False
        if selected is save_button:
            return self.save_project()
        return selected is discard_button

    def _request_home(self) -> None:
        if self._confirm_leave_page():
            self.home_requested.emit()

    def _confirm_leave_page(self) -> bool:
        return self._confirm_replace_project()

    def shutdown(self) -> None:
        self._alt_zoom_held = False
        self._canvas.cancel_pan()
        self._remove_application_event_filter()
        self._preview_generation += 1
        self._detail_generation += 1
        self._detail_timer.stop()
        self._detail_pending = None
        self._detail_loading = False
        self._progress.hide()
        self._stop_button.hide()
        self._detail_worker = None
        self._cancel_event.set()
        self._preview_worker = None
