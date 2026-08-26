"""设置页面布局和持久化回归测试。"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel

from services.settings_service import ApplicationSettings
from ui.pages.settings_page import SettingsPage


class _SettingsServiceStub:
    def __init__(self, root: str) -> None:
        self.library_root = root
        self.database_path = str(Path(root) / "配置" / "fontmgr.sqlite3")
        self.log_path = str(Path(root) / "font_editor.log")
        self.saved: list[ApplicationSettings] = []
        self.current = ApplicationSettings(
            default_dpi=600,
            default_canvas_width=320,
            default_canvas_height=480,
        )

    def load(self) -> ApplicationSettings:
        return self.current

    def defaults(self) -> ApplicationSettings:
        return ApplicationSettings()

    def validate(self, settings: ApplicationSettings) -> ApplicationSettings:
        return settings

    def save(self, settings: ApplicationSettings) -> None:
        self.current = settings
        self.saved.append(settings)

    def check_database_integrity(self) -> list[str]:
        return []


class SettingsPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_page_contains_four_confirmed_sections_and_saves(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = _SettingsServiceStub(directory)
            page = SettingsPage(service=service)  # type: ignore[arg-type]
            try:
                headings = {
                    label.text()
                    for label in page.findChildren(QLabel)
                    if label.objectName() == "sectionTitle"
                }
                self.assertEqual(
                    headings,
                    {"常规", "目录", "性能与缓存", "数据维护"},
                )
                self.assertEqual(page._dpi_spin.value(), 600)
                self.assertEqual(page._width_spin.value(), 320)
                self.assertEqual(page._height_spin.value(), 480)

                saved: list[object] = []
                page.settings_saved.connect(saved.append)
                page._dpi_spin.setValue(900)
                page._save()

                self.assertEqual(len(service.saved), 1)
                self.assertEqual(service.saved[0].default_dpi, 900)
                self.assertEqual(saved, [service.saved[0]])
                self.assertEqual(page._save_state_label.text(), "设置已保存")
            finally:
                page.deleteLater()

    def test_database_check_reports_normal_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            page = SettingsPage(  # type: ignore[arg-type]
                service=_SettingsServiceStub(directory)
            )
            try:
                page._check_database()
                self.assertEqual(page._integrity_label.text(), "完整性正常")
            finally:
                page.deleteLater()


if __name__ == "__main__":
    unittest.main()
