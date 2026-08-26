"""图片实验室独立工作台。"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from PySide6.QtCore import QThreadPool, QTimer, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
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
    QVBoxLayout,
    QWidget,
)

from core.image_cleanup import ImageCleanupOptions
from data.image_lab_project_store import (
    IMAGE_LAB_PROJECT_EXTENSION,
    ImageLabProject,
    ImageLabProjectStore,
    ImageLabStroke,
)
from services.image_lab_service import (
    SUPPORTED_IMAGE_FILTER,
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
        self._export_worker: FunctionWorker | None = None
        self._cancel_event = threading.Event()
        self._dirty = False
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
        title.setObjectName("pageTitle")
        subtitle = QLabel("整幅文献图片的非破坏背景清理与人工修补")
        subtitle.setObjectName("pageSubtitle")
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
        recognition_label = QLabel("多通道通用识别")
        recognition_label.setObjectName("subtleLabel")
        process_layout.addWidget(recognition_label)
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
                lambda _checked=False, selected=mode: self._canvas.set_view_mode(selected)
            )
            view_group.addButton(button)
            self._view_buttons[mode] = button
            toolbar.addWidget(button)
        self._view_buttons[VIEW_CLEAN].setChecked(True)
        toolbar.addStretch(1)
        self._fit_button = QPushButton("适合窗口")
        self._zoom_label = QLabel("100%")
        self._fit_button.clicked.connect(self._fit_canvas)
        toolbar.addWidget(self._fit_button)
        toolbar.addWidget(self._zoom_label)
        layout.addLayout(toolbar)

        self._canvas_scroll = QScrollArea()
        self._canvas_scroll.setObjectName("imageLabPreviewScroll")
        self._canvas_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._canvas_scroll.setWidgetResizable(False)
        self._canvas = ImageLabCanvas()
        self._canvas.stroke_finished.connect(self._stroke_finished)
        self._canvas.zoom_changed.connect(
            lambda value: self._zoom_label.setText(f"{value}%")
        )
        self._canvas_scroll.setWidget(self._canvas)
        layout.addWidget(self._canvas_scroll, 1)
        hint = QLabel("普通滚轮滚屏，Ctrl+滚轮缩放")
        hint.setObjectName("subtleLabel")
        layout.addWidget(hint)
        return panel

    def _connect_shortcuts(self) -> None:
        save_shortcut = QShortcut(QKeySequence.StandardKey.Save, self)
        save_shortcut.activated.connect(self.save_project)
        undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        undo_shortcut.activated.connect(self._undo_stroke)

    def _choose_image(self) -> None:
        if not self._confirm_replace_project():
            return
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "打开待处理图片",
            "",
            SUPPORTED_IMAGE_FILTER,
        )
        if not path:
            return
        try:
            project = self._service.create_project(path)
        except (OSError, RuntimeError, ValueError) as exc:
            QMessageBox.warning(self, "打开失败", f"无法打开图片：{exc}")
            return
        self._set_project(project, dirty=True)

    def _choose_project(self) -> None:
        if not self._confirm_replace_project():
            return
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "打开图片实验室项目",
            "",
            f"图片实验室项目 (*{IMAGE_LAB_PROJECT_EXTENSION})",
        )
        if not path:
            return
        try:
            project = self._store.load(path)
        except (OSError, RuntimeError, ValueError) as exc:
            QMessageBox.warning(self, "打开失败", f"无法打开项目：{exc}")
            return
        self._set_project(project, dirty=False)

    def _set_project(self, project: ImageLabProject, *, dirty: bool) -> None:
        self._project = project
        self._preview = None
        self._dirty = dirty
        self._strength_slider.setValue(project.options.strength)
        self._preserve_faint.setChecked(project.options.preserve_faint_ink)
        self._remove_noise.setChecked(project.options.remove_small_noise)
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
        )
        self._mark_dirty()
        self._start_preview()

    def _start_preview(self) -> None:
        if self._project is None:
            return
        self._preview_generation += 1
        generation = self._preview_generation
        project = self._project
        self._set_busy(True, "正在后台解码原稿并生成预览…", export=False)
        worker = FunctionWorker(lambda: self._service.load_preview(project))
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
        self._preview = result
        self._canvas.set_preview(
            result.source,
            result.composite,
            result.effective_alpha,
            result.cleanup.uncertainty_mask,
            source_width=result.source_width,
            source_height=result.source_height,
        )
        metrics = result.cleanup.metrics
        self._metrics_label.setText(
            f"识别方式：{result.cleanup.resolved_profile}\n"
            f"原稿尺寸：{result.source_width} × {result.source_height}\n"
            f"保留前景：{float(metrics['保留前景占比']) * 100:.1f}%\n"
            f"完全清理：{float(metrics['完全清理占比']) * 100:.1f}%\n"
            f"待核对：{float(metrics['待核对占比']) * 100:.1f}%"
        )
        self._set_busy(False, f"预览已生成，用时 {result.elapsed_seconds:.2f} 秒")
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
        QMessageBox.warning(self, "预览失败", f"无法生成清理预览：{message}")

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

    def _fit_canvas(self) -> None:
        viewport = self._canvas_scroll.viewport().size()
        self._canvas.fit_to_size(viewport.width(), viewport.height())

    def save_project(self) -> bool:
        if self._project is None:
            return False
        path = self._project.project_path
        if not path:
            suggested = os.path.join(
                os.path.dirname(self._project.source_path),
                f"{self._project.display_name}{IMAGE_LAB_PROJECT_EXTENSION}",
            )
            path, _selected_filter = QFileDialog.getSaveFileName(
                self,
                "保存图片实验室项目",
                suggested,
                f"图片实验室项目 (*{IMAGE_LAB_PROJECT_EXTENSION})",
            )
            if not path:
                return False
        try:
            self._store.save(self._project, path)
        except (OSError, RuntimeError, ValueError) as exc:
            QMessageBox.warning(self, "保存失败", f"无法保存项目：{exc}")
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
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            f"导出{suffix}",
            suggested,
            file_filter,
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
            QMessageBox.warning(self, "导出失败", "后台任务返回了无效结果。")
            return
        QMessageBox.information(
            self,
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
        QMessageBox.warning(self, "导出失败", f"无法生成完整尺寸文件：{message}")

    def _stop_export(self) -> None:
        if self._export_worker is not None:
            self._cancel_event.set()
            self._stop_button.setEnabled(False)
            self._status("正在等待当前分块安全结束…")

    def _set_busy(self, busy: bool, message: str, *, export: bool = False) -> None:
        self._open_image_button.setEnabled(not busy)
        self._open_project_button.setEnabled(not busy)
        self._apply_button.setEnabled(not busy and self._project is not None)
        self._save_button.setEnabled(not busy and self._project is not None)
        self._export_result_button.setEnabled(not busy and self._preview is not None)
        self._export_layer_button.setEnabled(not busy and self._preview is not None)
        self._export_photoshop_button.setEnabled(not busy and self._preview is not None)
        self._progress.setVisible(busy)
        self._stop_button.setVisible(busy and export)
        self._stop_button.setEnabled(busy and export)
        if busy:
            self._progress.setRange(0, 0)
            self._progress.setFormat(message)
        self._status(message)

    def _set_project_available(self, available: bool) -> None:
        for widget in (
            self._save_button,
            self._strength_slider,
            self._preserve_faint,
            self._remove_noise,
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
            QMessageBox.information(self, "后台任务正在执行", "请先等待任务完成或停止导出。")
            return False
        if not self._dirty:
            return True
        answer = QMessageBox.question(
            self,
            "项目尚未保存",
            "当前图片实验室项目有未保存修改，是否先保存？",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if answer == QMessageBox.StandardButton.Cancel:
            return False
        if answer == QMessageBox.StandardButton.Save:
            return self.save_project()
        return True

    def _request_home(self) -> None:
        if self._confirm_leave_page():
            self.home_requested.emit()

    def _confirm_leave_page(self) -> bool:
        return self._confirm_replace_project()

    def shutdown(self) -> None:
        self._preview_generation += 1
        self._cancel_event.set()
        self._preview_worker = None
