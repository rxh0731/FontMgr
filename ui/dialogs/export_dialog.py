"""导出最终成品对话框。"""

from __future__ import annotations

import os
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from services.export_service import ExportService
from services.glyph_service import GlyphService


class ExportDialog(QDialog):
    """在导出时选择成品风格、背景和命名方式。"""

    export_completed = Signal(str, int)

    def __init__(self, glyph_service: GlyphService, parent=None) -> None:
        super().__init__(parent)
        self._glyph_service = glyph_service
        self._build_ui()

    def _build_ui(self) -> None:
        self.setWindowTitle("导出最终成品")
        self.setModal(True)
        self.setMinimumWidth(560)

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 22)
        root.setSpacing(16)

        title = QLabel("导出最终成品")
        title.setObjectName("sectionTitle")
        root.addWidget(title)

        description = QLabel("成品风格和背景只作用于本次导出，不会写入或修改字库参数。")
        description.setWordWrap(True)
        description.setObjectName("mutedLabel")
        root.addWidget(description)

        form = QFormLayout()
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(12)

        directory_row = QHBoxLayout()
        self._directory_edit = QLineEdit()
        self._directory_edit.setPlaceholderText("请选择导出目录")
        browse_button = QPushButton("选择目录")
        browse_button.clicked.connect(self._choose_directory)
        directory_row.addWidget(self._directory_edit, 1)
        directory_row.addWidget(browse_button)
        form.addRow("导出目录", directory_row)

        self._style_combo = QComboBox()
        self._style_combo.addItems(("灰度保真", "纯二值", "统一软边"))
        form.addRow("成品风格", self._style_combo)

        self._name_mode_combo = QComboBox()
        self._name_mode_combo.addItem("使用字符命名", "字符")
        self._name_mode_combo.addItem("使用原文件名", "原文件名")
        form.addRow("文件命名", self._name_mode_combo)

        self._transparent_check = QCheckBox("使用透明背景（导出为 PNG）")
        self._transparent_check.setChecked(True)
        form.addRow("背景", self._transparent_check)
        root.addLayout(form)

        hint = QLabel("关闭透明背景后，成品将合成到白色背景并导出为 BMP。")
        hint.setObjectName("mutedLabel")
        hint.setWordWrap(True)
        root.addWidget(hint)

        actions = QHBoxLayout()
        actions.addStretch(1)
        cancel_button = QPushButton("取消")
        cancel_button.clicked.connect(self.reject)
        export_button = QPushButton("开始导出")
        export_button.setObjectName("primaryButton")
        export_button.clicked.connect(self._export)
        actions.addWidget(cancel_button)
        actions.addWidget(export_button)
        root.addLayout(actions)

    def _choose_directory(self) -> None:
        initial = self._directory_edit.text().strip() or os.path.expanduser("~")
        directory = QFileDialog.getExistingDirectory(self, "选择导出目录", initial)
        if directory:
            self._directory_edit.setText(directory)

    def _export(self) -> None:
        output_dir = self._directory_edit.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "缺少导出目录", "请先选择导出目录。")
            return
        if not os.path.isdir(output_dir):
            QMessageBox.warning(self, "导出目录无效", "所选导出目录不存在，请重新选择。")
            return

        name_mode = str(self._name_mode_combo.currentData())
        output_style = self._style_combo.currentText()
        transparent_background = self._transparent_check.isChecked()
        try:
            result: dict[str, Any] = ExportService(self._glyph_service).export(
                output_dir,
                name_mode=name_mode,
                transparent_background=transparent_background,
                output_style=output_style,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            QMessageBox.critical(self, "导出失败", str(exc))
            return

        success_count = int(result.get("成功", 0))
        failure_count = int(result.get("失败", 0))
        message = f"已导出 {success_count} 个成品。"
        if failure_count:
            message += f"\n另有 {failure_count} 个成品导出失败。"
        QMessageBox.information(self, "导出完成", message)
        self.export_completed.emit(output_dir, success_count)
        self.accept()
