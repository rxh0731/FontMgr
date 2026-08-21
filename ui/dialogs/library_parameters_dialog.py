"""字库参数修改对话框。"""

from __future__ import annotations

import os
from collections.abc import Iterable

from PySide6.QtCore import QSignalBlocker, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from services.glyph_service import GlyphService


class LibraryParametersDialog(QDialog):
    """修改字库名称及画布规格。"""

    saved = Signal(str, int)
    return_home_requested = Signal()

    def __init__(
        self,
        service: GlyphService,
        existing_names: Iterable[str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._old_name = service.ziku_name
        self._existing_names = {
            str(name).strip() for name in (existing_names or ()) if str(name).strip()
        }
        self._saved = False
        self._syncing = False
        self.saved_name = self._old_name
        self.invalidated_count = 0

        self.setWindowTitle("修改字库参数")
        self.setModal(True)
        self.setMinimumWidth(500)
        self._build_ui()
        self._load_values()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 22)
        root.setSpacing(16)

        title = QLabel("修改字库参数")
        title.setProperty("role", "pageTitle")
        root.addWidget(title)

        subtitle = QLabel("调整字库名称与画布规格。修改规格后，已有成品需要重新整体协调。")
        subtitle.setWordWrap(True)
        subtitle.setProperty("role", "muted")
        root.addWidget(subtitle)

        card = QFrame()
        card.setProperty("role", "card")
        form = QFormLayout(card)
        form.setContentsMargins(20, 18, 20, 18)
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(12)

        self.name_edit = QLineEdit()
        self.name_edit.setMaxLength(120)
        form.addRow("字库名称", self.name_edit)

        self.width_px_spin = self._make_spin(1, 10000, " 像素")
        self.height_px_spin = self._make_spin(1, 10000, " 像素")
        self.width_mm_spin = self._make_decimal_spin(0.01, 10000.0, " 毫米")
        self.height_mm_spin = self._make_decimal_spin(0.01, 10000.0, " 毫米")
        self.dpi_spin = self._make_spin(1, 9600, " DPI")

        width_row = QHBoxLayout()
        width_row.setSpacing(10)
        width_row.addWidget(self.width_px_spin, 1)
        width_row.addWidget(self.width_mm_spin, 1)
        height_row = QHBoxLayout()
        height_row.setSpacing(10)
        height_row.addWidget(self.height_px_spin, 1)
        height_row.addWidget(self.height_mm_spin, 1)
        form.addRow("成品宽度", width_row)
        form.addRow("成品高度", height_row)
        form.addRow("分辨率", self.dpi_spin)
        root.addWidget(card)

        notice = QLabel("换算公式：毫米 = 像素 ÷ DPI × 25.4。毫米显示与保存到小数点后 2 位。")
        notice.setProperty("role", "muted")
        root.addWidget(notice)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.return_home_button = QPushButton("取消")
        self.save_button = QPushButton("保存")
        self.save_button.setProperty("role", "primary")
        buttons.addWidget(self.return_home_button)
        buttons.addWidget(self.save_button)
        root.addLayout(buttons)

        self.width_px_spin.valueChanged.connect(self._sync_mm_from_px)
        self.height_px_spin.valueChanged.connect(self._sync_mm_from_px)
        self.dpi_spin.valueChanged.connect(self._sync_mm_from_px)
        self.width_mm_spin.valueChanged.connect(self._sync_px_from_mm)
        self.height_mm_spin.valueChanged.connect(self._sync_px_from_mm)
        self.return_home_button.clicked.connect(self._return_home)
        self.save_button.clicked.connect(self._save)

    @staticmethod
    def _make_spin(minimum: int, maximum: int, suffix: str) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSuffix(suffix)
        spin.setKeyboardTracking(False)
        return spin

    @staticmethod
    def _make_decimal_spin(minimum: float, maximum: float, suffix: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(2)
        spin.setSingleStep(0.01)
        spin.setSuffix(suffix)
        spin.setKeyboardTracking(False)
        return spin

    def _load_values(self) -> None:
        metadata = self._service.get_metadata()
        dpi = max(1, int(metadata.get("DPI", 300)))
        width_px = max(1, int(metadata.get("画布宽", 250)))
        height_px = max(1, int(metadata.get("画布高", 250)))
        width_mm = max(0.01, round(float(metadata.get("成品宽度毫米", width_px / dpi * 25.4)), 2))
        height_mm = max(0.01, round(float(metadata.get("成品高度毫米", height_px / dpi * 25.4)), 2))

        self._syncing = True
        self.name_edit.setText(self._old_name)
        self.dpi_spin.setValue(dpi)
        self.width_px_spin.setValue(width_px)
        self.height_px_spin.setValue(height_px)
        self.width_mm_spin.setValue(width_mm)
        self.height_mm_spin.setValue(height_mm)
        self._syncing = False

    def _sync_mm_from_px(self) -> None:
        if self._syncing:
            return
        self._syncing = True
        dpi = max(1, self.dpi_spin.value())
        with QSignalBlocker(self.width_mm_spin), QSignalBlocker(self.height_mm_spin):
            self.width_mm_spin.setValue(max(0.01, round(self.width_px_spin.value() / dpi * 25.4, 2)))
            self.height_mm_spin.setValue(max(0.01, round(self.height_px_spin.value() / dpi * 25.4, 2)))
        self._syncing = False

    def _sync_px_from_mm(self) -> None:
        if self._syncing:
            return
        self._syncing = True
        dpi = max(1, self.dpi_spin.value())
        with QSignalBlocker(self.width_px_spin), QSignalBlocker(self.height_px_spin):
            self.width_px_spin.setValue(max(1, round(self.width_mm_spin.value() / 25.4 * dpi)))
            self.height_px_spin.setValue(max(1, round(self.height_mm_spin.value() / 25.4 * dpi)))
        self._syncing = False

    def _name_conflicts(self, new_name: str) -> bool:
        if new_name == self._old_name:
            return False
        if new_name in self._existing_names:
            return True
        parent_dir = os.path.dirname(os.path.abspath(self._service.ziku_dir))
        return os.path.exists(os.path.join(parent_dir, new_name))

    def _save(self) -> None:
        new_name = self.name_edit.text().strip()
        if not new_name:
            QMessageBox.warning(self, "提示", "字库名称不能为空。")
            self.name_edit.setFocus()
            return
        if self._name_conflicts(new_name):
            QMessageBox.warning(self, "提示", f"字库“{new_name}”已存在，请使用其他名称。")
            self.name_edit.setFocus()
            self.name_edit.selectAll()
            return

        metadata = self._service.get_metadata()
        dpi = self.dpi_spin.value()
        width_px = self.width_px_spin.value()
        height_px = self.height_px_spin.value()
        width_mm = round(float(self.width_mm_spin.value()), 2)
        height_mm = round(float(self.height_mm_spin.value()), 2)
        spec_changed = (
            dpi != int(metadata.get("DPI", 300))
            or width_px != int(metadata.get("画布宽", 250))
            or height_px != int(metadata.get("画布高", 250))
            or width_mm != round(float(metadata.get("成品宽度毫米", width_mm)), 2)
            or height_mm != round(float(metadata.get("成品高度毫米", height_mm)), 2)
        )

        self.save_button.setEnabled(False)
        try:
            if new_name != self._old_name:
                self._service.rename_ziku(new_name)
            if spec_changed:
                self.invalidated_count = self._service.update_output_spec(
                    dpi, width_px, height_px, width_mm, height_mm
                )
            self._service.remove_metadata_keys("成品风格", "透明背景")
            self._service.save()
        except Exception as exc:
            QMessageBox.critical(self, "保存失败", f"参数保存失败：{exc}")
            self.save_button.setEnabled(True)
            return

        self._saved = True
        self.saved_name = self._service.ziku_name
        if self.invalidated_count:
            QMessageBox.information(
                self,
                "参数已保存",
                f"参数已保存。已有 {self.invalidated_count} 个成品失效，请重新执行整体协调。",
            )
        self.saved.emit(self.saved_name, self.invalidated_count)
        self.accept()

    def _return_home(self) -> None:
        self.return_home_requested.emit()
        self.reject()

    @property
    def was_saved(self) -> bool:
        """本次对话框是否成功保存。"""
        return self._saved

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self._saved:
            self.return_home_requested.emit()
        super().closeEvent(event)
