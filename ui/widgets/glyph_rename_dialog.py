"""字形归属字修正对话框。"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from services.glyph_service import GlyphService
from utils.file_utils import resolve_safe_child_file


class GlyphRenameDialog(QDialog):
    """只修改归属字，由业务层同步六阶段文件与索引。"""

    PREVIEW_SIZE = 150

    def __init__(
        self,
        glyph_service: GlyphService,
        variant_id: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._glyph = glyph_service
        self._variant_id = variant_id
        self._detail = self._glyph.get_variant(variant_id)
        self._plan: dict[str, Any] | None = None
        self.result_data: dict[str, Any] | None = None
        self.setWindowTitle("修正字形名称")
        self.setModal(True)
        self.setMinimumWidth(560)
        self._build_ui()
        self._load_preview()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 16)
        root.setSpacing(12)

        intro = QLabel(
            "这里只修正图片对应的汉字。文件序号、扩展名和六个阶段文件由系统同步管理。"
        )
        intro.setWordWrap(True)
        intro.setProperty("role", "muted")
        root.addWidget(intro)

        content = QHBoxLayout()
        content.setSpacing(16)
        self._preview_label = QLabel("图片无法读取")
        self._preview_label.setFixedSize(self.PREVIEW_SIZE, self.PREVIEW_SIZE)
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setStyleSheet(
            "background: white; color: #667085; border: 1px solid #AEB8C5;"
        )
        content.addWidget(self._preview_label, 0, Qt.AlignmentFlag.AlignTop)

        form_host = QWidget()
        form = QFormLayout(form_host)
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)
        old_char = str(self._detail.get("归属字", ""))
        old_filename = str(self._detail.get("原始文件", ""))
        form.addRow("当前归属字：", QLabel(old_char or "-"))
        form.addRow("当前原图文件：", QLabel(old_filename or "-"))
        self._char_edit = QLineEdit()
        self._char_edit.setMaxLength(1)
        self._char_edit.setPlaceholderText("输入正确的一个汉字")
        self._char_edit.textChanged.connect(self._update_plan)
        form.addRow("正确归属字：", self._char_edit)
        self._new_filename_label = QLabel("-")
        self._new_filename_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        form.addRow("修改后原图文件：", self._new_filename_label)
        content.addWidget(form_host, 1)
        root.addLayout(content)

        stage_title = QLabel("将同步处理的阶段文件")
        stage_title.setStyleSheet("font-weight: 600;")
        root.addWidget(stage_title)
        self._stage_list = QListWidget()
        self._stage_list.setMinimumHeight(126)
        self._stage_list.setAlternatingRowColors(True)
        root.addWidget(self._stage_list)

        self._validation_label = QLabel("请输入正确的一个汉字。")
        self._validation_label.setWordWrap(True)
        self._validation_label.setProperty("role", "muted")
        root.addWidget(self._validation_label)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        self._confirm_button = self._buttons.button(
            QDialogButtonBox.StandardButton.Ok
        )
        self._cancel_button = self._buttons.button(
            QDialogButtonBox.StandardButton.Cancel
        )
        self._confirm_button.setText("确认修改")
        self._confirm_button.setEnabled(False)
        self._cancel_button.setText("取消")
        self._buttons.accepted.connect(self._commit)
        self._buttons.rejected.connect(self.reject)
        root.addWidget(self._buttons)

    def _load_preview(self) -> None:
        if not self._detail:
            return
        directories = self._glyph.get_workflow_dirs()
        for stage, field in (
            ("成品", "成品文件"),
            ("手工审核", "审核文件"),
            ("优化预览", "中间文件"),
            ("清洁掩码", "清洁掩码文件"),
            ("灰度母版", "灰度母版文件"),
            ("原图", "原始文件"),
        ):
            path = resolve_safe_child_file(
                directories[stage],
                self._detail.get(field, ""),
            )
            if not path:
                continue
            image = QImage(path)
            if image.isNull():
                continue
            target = QImage(
                self.PREVIEW_SIZE,
                self.PREVIEW_SIZE,
                QImage.Format.Format_ARGB32_Premultiplied,
            )
            target.fill(QColor("white"))
            scaled = image.scaled(
                self.PREVIEW_SIZE - 16,
                self.PREVIEW_SIZE - 16,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            painter = QPainter(target)
            painter.drawImage(
                (self.PREVIEW_SIZE - scaled.width()) // 2,
                (self.PREVIEW_SIZE - scaled.height()) // 2,
                scaled,
            )
            painter.end()
            self._preview_label.setPixmap(QPixmap.fromImage(target))
            self._preview_label.setText("")
            return

    def _update_plan(self, value: str) -> None:
        self._plan = None
        self._stage_list.clear()
        self._new_filename_label.setText("-")
        old_char = str(self._detail.get("归属字", ""))
        if not value:
            self._set_validation("请输入正确的一个汉字。", error=False)
            return
        if value == old_char:
            self._set_validation("新名称与当前归属字相同，不需要修改。", error=False)
            return
        try:
            plan = self._glyph.preview_variant_char_change(
                self._variant_id,
                value,
            )
        except Exception as exc:
            self._set_validation(str(exc), error=True)
            return
        self._plan = plan
        self._new_filename_label.setText(str(plan.get("新文件名", "-")))
        for change in plan.get("文件变更", []):
            self._stage_list.addItem(
                f"{change['阶段']}：{change['原文件名']}  →  {change['新文件名']}"
            )
        count = len(plan.get("文件变更", []))
        self._set_validation(
            f"预检通过，将同步修改 {count} 个现有阶段文件；图片内容和制作状态保持不变。",
            error=False,
            ready=True,
        )

    def _set_validation(
        self,
        message: str,
        *,
        error: bool,
        ready: bool = False,
    ) -> None:
        self._validation_label.setText(message)
        self._validation_label.setStyleSheet(
            "color: #E36A6A;" if error else ""
        )
        self._confirm_button.setEnabled(ready)

    def _commit(self) -> None:
        if self._plan is None:
            return
        old_char = str(self._plan.get("原归属字", ""))
        new_char = str(self._plan.get("新归属字", ""))
        answer = QMessageBox.question(
            self,
            "确认修正字形名称",
            f"确定将“{old_char}”修正为“{new_char}”吗？\n\n"
            "系统将同步更新所有现有阶段文件、字形分组和字库记录。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.result_data = self._glyph.move_variant_to_char(
                self._variant_id,
                new_char,
            )
        except Exception as exc:
            QMessageBox.critical(self, "名称修改失败", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.accept()


def run_glyph_rename_dialog(
    parent: QWidget,
    glyph_service: GlyphService,
    variant_id: str,
) -> dict[str, Any] | None:
    """运行共用修正窗口，成功时返回本次同步改名结果。"""
    if not variant_id or not glyph_service.get_variant(variant_id):
        QMessageBox.information(parent, "修正字形名称", "请先选择一个具体字形。")
        return None
    dialog = GlyphRenameDialog(glyph_service, variant_id, parent)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    return dialog.result_data
