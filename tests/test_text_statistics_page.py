from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

import config
from data.library_summary_store import LibrarySummaryStore
from ui.main_window import MainWindow
from ui.pages.text_statistics_page import TextStatisticsPage


class TextStatisticsPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self._library_root = tempfile.TemporaryDirectory()
        self._root_patch = patch.object(config, "ZIKU_ROOT", self._library_root.name)
        self._root_patch.start()

    def tearDown(self) -> None:
        self._root_patch.stop()
        self._library_root.cleanup()

    def test_page_updates_statistics_and_cached_missing_results(self) -> None:
        page = TextStatisticsPage()
        try:
            page._external_source_radio.setChecked(True)
            page._external_path_edit.setText(r"D:\字库")
            page._font_cache_path = (
                "external|" + os.path.normcase(os.path.abspath(r"D:\字库"))
            )
            page._font_cache = frozenset({"甲"})
            page._font_cache_kind = "外部图片目录"
            page._font_cache_variants = 1
            page._content_edit.setPlainText("乙甲甲，ABC。")
            page._run_statistics(True)

            self.assertIn("汉字：3", page._info_label.text())
            self.assertEqual(page._all_results.toPlainText(), "甲 乙")
            self.assertEqual(page._missing_results.toPlainText(), "乙")
            self.assertIn("可用 1 个字", page._comparison_status.text())
            self.assertTrue(page._export_all_button.isEnabled())
            self.assertTrue(page._export_missing_button.isEnabled())
        finally:
            page.shutdown()
            page.deleteLater()
            self.app.processEvents()

    def test_page_uses_two_columns_with_stacked_input_and_result_panels(self) -> None:
        page = TextStatisticsPage()
        try:
            page.resize(1100, 720)
            page.show()
            self.app.processEvents()

            self.assertEqual(
                page._main_splitter.orientation(),
                Qt.Orientation.Horizontal,
            )
            self.assertEqual(page._main_splitter.count(), 2)
            self.assertEqual(page._left_splitter.orientation(), Qt.Orientation.Vertical)
            self.assertEqual(page._right_splitter.orientation(), Qt.Orientation.Vertical)
            self.assertIs(page._left_splitter.widget(0), page._source_panel)
            self.assertIs(page._left_splitter.widget(1), page._content_section)
            self.assertIs(page._right_splitter.widget(0), page._statistics_section)
            self.assertIs(page._right_splitter.widget(1), page._missing_section)

            left_rect = page._left_splitter.geometry()
            right_rect = page._right_splitter.geometry()
            self.assertGreater(right_rect.left(), left_rect.left())
            self.assertGreater(left_rect.width(), right_rect.width())
            self.assertGreater(
                page._content_section.geometry().top(),
                page._source_panel.geometry().top(),
            )
            self.assertGreater(
                page._missing_section.geometry().top(),
                page._statistics_section.geometry().top(),
            )
            for panel in (
                page._source_panel,
                page._content_section,
                page._statistics_section,
                page._missing_section,
            ):
                self.assertGreaterEqual(panel.width(), 300)
                self.assertGreaterEqual(panel.height(), 210)
        finally:
            page.shutdown()
            page.close()
            page.deleteLater()
            self.app.processEvents()

    def test_comparison_sources_are_mutually_exclusive_and_list_sqlite_libraries(
        self,
    ) -> None:
        root = Path(self._library_root.name)
        valid = root / "系统库"
        invalid = root / "普通目录"
        valid.mkdir()
        invalid.mkdir()
        (valid / "font_library.sqlite3").write_bytes(b"database-placeholder")

        page = TextStatisticsPage()
        try:
            self.assertTrue(page._system_source_radio.isChecked())
            self.assertEqual(page._system_library_combo.count(), 1)
            self.assertEqual(page._system_library_combo.currentText(), "系统库")
            self.assertTrue(page._system_library_combo.isEnabled())
            self.assertFalse(page._external_path_edit.isEnabled())
            self.assertFalse(page._browse_font_button.isEnabled())

            page._external_source_radio.setChecked(True)

            self.assertFalse(page._system_library_combo.isEnabled())
            self.assertTrue(page._external_path_edit.isEnabled())
            self.assertTrue(page._browse_font_button.isEnabled())
        finally:
            page.shutdown()
            page.deleteLater()
            self.app.processEvents()

    def test_background_file_load_populates_editor_and_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "经文.txt"
            path.write_text("般若波羅蜜多", encoding="utf-8")
            page = TextStatisticsPage()
            try:
                with (
                    patch.object(page._thread_pool, "start", side_effect=lambda worker: worker.run()),
                    patch.object(QMessageBox, "information"),
                ):
                    page._start_file_task((str(path),), replace=True)
                    self.app.processEvents()

                self.assertEqual(page._content_edit.toPlainText(), "般若波羅蜜多")
                self.assertEqual(page._source_edit.text(), str(path))
                self.assertIn("共 6 个不重复汉字", page._count_label.text())
                self.assertIsNone(page._file_worker)
            finally:
                page.shutdown()
                page.deleteLater()
                self.app.processEvents()

    def test_export_writes_utf8_text_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "统计结果.txt"
            page = TextStatisticsPage()
            try:
                with (
                    patch.object(
                        QFileDialog,
                        "getSaveFileName",
                        return_value=(str(path), "文本文件（*.txt）"),
                    ),
                    patch.object(QMessageBox, "information"),
                ):
                    page._export_characters(("甲", "乙"), "导出全部文字", "统计结果.txt")

                self.assertEqual(path.read_text(encoding="utf-8"), "甲 乙")
            finally:
                page.shutdown()
                page.deleteLater()
                self.app.processEvents()

    def test_home_route_opens_and_releases_real_statistics_page(self) -> None:
        with patch.object(LibrarySummaryStore, "load", return_value=[]):
            window = MainWindow()
        try:
            self.assertIsNone(window._text_statistics_page)

            window._open_tool("statistics")
            page = window._text_statistics_page
            self.assertIsInstance(page, TextStatisticsPage)
            self.assertIs(window._stack.currentWidget(), page)

            window.show_home()
            self.app.processEvents()
            self.assertIsNone(window._text_statistics_page)
        finally:
            window.close()
            window.deleteLater()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
