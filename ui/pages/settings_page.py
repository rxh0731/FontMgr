"""程序级设置与数据维护页面。"""

from __future__ import annotations

import os
from collections.abc import Callable

from PySide6.QtCore import QUrl, Qt, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from services.settings_service import (
    PERFORMANCE_MODES,
    ApplicationSettings,
    SettingsService,
)


class SettingsPage(QWidget):
    """编辑全局偏好，并提供安全的程序数据库检查入口。"""

    home_requested = Signal()
    status_message = Signal(str)
    settings_saved = Signal(object)
    cache_cleanup_requested = Signal()

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        service: SettingsService | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = service or SettingsService()
        self._build_ui()
        self.reload()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(22, 18, 22, 20)
        root.setSpacing(12)

        header = QHBoxLayout()
        title = QLabel("设置")
        title.setProperty("role", "pageTitle")
        header.addWidget(title)
        header.addStretch(1)
        home_button = QPushButton("返回首页")
        home_button.clicked.connect(self.home_requested)
        header.addWidget(home_button)
        root.addLayout(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        host = QWidget()
        grid = QGridLayout(host)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)
        grid.addWidget(self._build_general_panel(), 0, 0)
        grid.addWidget(self._build_directory_panel(), 0, 1)
        grid.addWidget(self._build_performance_panel(), 1, 0)
        grid.addWidget(self._build_maintenance_panel(), 1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(2, 1)
        scroll.setWidget(host)
        root.addWidget(scroll, 1)

        footer = QHBoxLayout()
        self._save_state_label = QLabel("")
        self._save_state_label.setProperty("role", "muted")
        footer.addWidget(self._save_state_label)
        footer.addStretch(1)
        defaults_button = QPushButton("恢复默认值")
        defaults_button.clicked.connect(self._restore_defaults)
        footer.addWidget(defaults_button)
        save_button = QPushButton("保存设置")
        save_button.setProperty("role", "primary")
        save_button.clicked.connect(self._save)
        footer.addWidget(save_button)
        root.addLayout(footer)

    def _build_general_panel(self) -> QFrame:
        panel, layout = self._panel("常规")
        form = QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)
        form.setLabelAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._dpi_spin = QSpinBox()
        self._dpi_spin.setRange(1, 9600)
        self._dpi_spin.setSuffix(" DPI")
        self._width_spin = QSpinBox()
        self._width_spin.setRange(1, 50_000)
        self._width_spin.setSuffix(" 像素")
        self._height_spin = QSpinBox()
        self._height_spin.setRange(1, 50_000)
        self._height_spin.setSuffix(" 像素")
        form.addRow("默认分辨率", self._dpi_spin)
        form.addRow("默认画布宽度", self._width_spin)
        form.addRow("默认画布高度", self._height_spin)
        layout.addLayout(form)
        layout.addStretch(1)
        return panel

    def _build_directory_panel(self) -> QFrame:
        panel, layout = self._panel("目录")
        self._image_directory_edit = self._directory_row(
            layout, "默认图片目录", "选择默认图片目录"
        )
        self._export_directory_edit = self._directory_row(
            layout, "默认导出目录", "选择默认导出目录"
        )
        self._layout_directory_edit = self._directory_row(
            layout, "默认排版输出目录", "选择默认排版输出目录"
        )
        layout.addStretch(1)
        return panel

    def _build_performance_panel(self) -> QFrame:
        panel, layout = self._panel("性能与缓存")
        form = QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)
        self._performance_combo = QComboBox()
        self._performance_combo.addItems(PERFORMANCE_MODES)
        self._performance_combo.setItemData(0, "按处理器核心数自动分配后台任务")
        self._performance_combo.setItemData(1, "减少后台并行任务，降低内存占用")
        form.addRow("性能模式", self._performance_combo)
        layout.addLayout(form)
        cache_row = QHBoxLayout()
        cache_state = QLabel("图片缓存：自动管理")
        cache_state.setProperty("role", "muted")
        cache_row.addWidget(cache_state)
        cache_row.addStretch(1)
        release_button = QPushButton("释放闲置内存")
        release_button.clicked.connect(self.cache_cleanup_requested)
        cache_row.addWidget(release_button)
        layout.addLayout(cache_row)
        layout.addStretch(1)
        return panel

    def _build_maintenance_panel(self) -> QFrame:
        panel, layout = self._panel("数据维护")
        self._readonly_path_row(
            layout,
            "字库目录",
            self._service.library_root,
            lambda: self._open_directory(self._service.library_root),
        )
        self._readonly_path_row(
            layout,
            "程序数据库",
            self._service.database_path,
            lambda: self._open_directory(os.path.dirname(self._service.database_path)),
        )
        self._readonly_path_row(
            layout,
            "日志文件",
            self._service.log_path,
            lambda: self._open_directory(os.path.dirname(self._service.log_path)),
        )
        check_row = QHBoxLayout()
        self._integrity_label = QLabel("尚未核对")
        self._integrity_label.setProperty("role", "muted")
        check_row.addWidget(self._integrity_label)
        check_row.addStretch(1)
        check_button = QPushButton("核对程序数据库")
        check_button.clicked.connect(self._check_database)
        check_row.addWidget(check_button)
        layout.addLayout(check_row)
        layout.addStretch(1)
        return panel

    @staticmethod
    def _panel(title: str) -> tuple[QFrame, QVBoxLayout]:
        panel = QFrame()
        panel.setProperty("role", "card")
        panel.setMinimumHeight(210)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(11)
        heading = QLabel(title)
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)
        return panel, layout

    def _directory_row(
        self,
        layout: QVBoxLayout,
        label_text: str,
        dialog_title: str,
    ) -> QLineEdit:
        label = QLabel(label_text)
        edit = QLineEdit()
        edit.setReadOnly(True)
        edit.setClearButtonEnabled(True)
        edit.setPlaceholderText("使用功能默认位置")
        button = QPushButton("选择")
        button.clicked.connect(
            lambda: self._choose_directory(edit, dialog_title)
        )
        clear_button = QPushButton("清除")
        clear_button.clicked.connect(lambda: self._clear_directory(edit))
        row = QHBoxLayout()
        row.addWidget(edit, 1)
        row.addWidget(button)
        row.addWidget(clear_button)
        layout.addWidget(label)
        layout.addLayout(row)
        return edit

    @staticmethod
    def _readonly_path_row(
        layout: QVBoxLayout,
        label_text: str,
        path: str,
        callback: Callable[[], None],
    ) -> None:
        label = QLabel(label_text)
        edit = QLineEdit(path)
        edit.setReadOnly(True)
        button = QPushButton("打开")
        button.clicked.connect(callback)
        row = QHBoxLayout()
        row.addWidget(edit, 1)
        row.addWidget(button)
        layout.addWidget(label)
        layout.addLayout(row)

    def reload(self) -> None:
        try:
            settings = self._service.load()
        except (OSError, RuntimeError, ValueError) as exc:
            QMessageBox.warning(self, "设置读取失败", f"无法读取程序设置：{exc}")
            settings = self._service.defaults()
        self._apply_settings(settings)
        self._save_state_label.clear()
        self._integrity_label.setText("尚未核对")

    def _apply_settings(self, settings: ApplicationSettings) -> None:
        self._dpi_spin.setValue(settings.default_dpi)
        self._width_spin.setValue(settings.default_canvas_width)
        self._height_spin.setValue(settings.default_canvas_height)
        self._image_directory_edit.setText(settings.default_image_directory)
        self._export_directory_edit.setText(settings.default_export_directory)
        self._layout_directory_edit.setText(settings.default_layout_directory)
        index = self._performance_combo.findText(settings.performance_mode)
        self._performance_combo.setCurrentIndex(max(0, index))

    def _current_settings(self) -> ApplicationSettings:
        return ApplicationSettings(
            default_dpi=self._dpi_spin.value(),
            default_canvas_width=self._width_spin.value(),
            default_canvas_height=self._height_spin.value(),
            default_image_directory=self._image_directory_edit.text().strip(),
            default_export_directory=self._export_directory_edit.text().strip(),
            default_layout_directory=self._layout_directory_edit.text().strip(),
            performance_mode=self._performance_combo.currentText(),
        )

    def _restore_defaults(self) -> None:
        self._apply_settings(self._service.defaults())
        self._save_state_label.setText("默认值尚未保存")

    def _save(self) -> None:
        try:
            settings = self._service.validate(self._current_settings())
            self._service.save(settings)
        except (OSError, RuntimeError, ValueError) as exc:
            QMessageBox.warning(self, "保存失败", str(exc))
            return
        self._apply_settings(settings)
        self._save_state_label.setText("设置已保存")
        self.settings_saved.emit(settings)
        self.status_message.emit("程序设置已保存")

    def _choose_directory(self, edit: QLineEdit, title: str) -> None:
        initial = edit.text().strip() or self._service.library_root
        directory = QFileDialog.getExistingDirectory(self, title, initial)
        if directory:
            edit.setText(directory)
            self._save_state_label.setText("设置尚未保存")

    def _clear_directory(self, edit: QLineEdit) -> None:
        edit.clear()
        self._save_state_label.setText("设置尚未保存")

    def _open_directory(self, path: str) -> None:
        if not os.path.isdir(path):
            QMessageBox.warning(self, "目录不存在", f"无法打开目录：{path}")
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(path)):
            QMessageBox.warning(self, "打开失败", f"无法打开目录：{path}")

    def _check_database(self) -> None:
        try:
            issues = self._service.check_database_integrity()
        except (OSError, RuntimeError, ValueError) as exc:
            self._integrity_label.setText("核对失败")
            QMessageBox.warning(self, "数据库核对失败", str(exc))
            return
        if issues:
            self._integrity_label.setText(f"发现 {len(issues)} 项异常")
            QMessageBox.warning(
                self,
                "数据库核对结果",
                f"程序数据库发现 {len(issues)} 项异常。请先停止继续操作并备份配置目录。",
            )
            return
        self._integrity_label.setText("完整性正常")
        self.status_message.emit("程序数据库完整性正常")
