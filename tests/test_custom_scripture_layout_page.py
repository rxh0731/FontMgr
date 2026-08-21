"""定制经文排版页面交互回归测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PySide6.QtCore import QPointF, QSize, Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QLabel, QScrollArea

import config
from services.scripture_layout_service import GenerationResult, GlyphIndex
from ui.pages.custom_scripture_layout_page import CustomScriptureLayoutPage
from ui.main_window import MainWindow


class CustomScriptureLayoutPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _page(self, directory: str) -> CustomScriptureLayoutPage:
        template_path = str(Path(directory) / "定制经文排版模板.json")
        with (
            patch.object(config, "CUSTOM_LAYOUT_TEMPLATE_FILE", template_path),
            patch.object(config, "ZIKU_ROOT", str(Path(directory) / "字库")),
        ):
            return CustomScriptureLayoutPage()

    def test_page_uses_confirmed_title_output_name_and_default_board(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            page = self._page(directory)
            try:
                titles = [
                    label.text()
                    for label in page.findChildren(QLabel)
                    if label.property("role") == "pageTitle"
                ]
                self.assertIn("定制经文排版", titles)
                self.assertEqual(page._output_name_edit.text(), "定制经文排版")
                self.assertEqual(len(page._custom_board_parameters), 1)
                self.assertEqual(page._template_combo.currentText(), "默认模板")
            finally:
                page.shutdown()
                page.deleteLater()

    def test_generation_completion_feedback_includes_elapsed_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            page = self._page(directory)
            try:
                worker = MagicMock()
                page._generation_worker = worker
                page._workers.add(worker)
                page._generation_started_at = 0.0
                with patch(
                    "ui.pages.custom_scripture_layout_page.QMessageBox.information"
                ) as information:
                    page._generation_finished(GenerationResult((), False), worker)

                self.assertEqual(
                    information.call_args.args[1],
                    "定制经文排版完成",
                )
                self.assertIn("总耗时：", information.call_args.args[2])
            finally:
                page.shutdown()
                page.deleteLater()

    def test_more_text_pages_are_shown_as_ignored_until_parameters_are_added(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            page = self._page(directory)
            try:
                page._glyph_index = GlyphIndex("测试", {})
                page._text_edit.setPlainText("甲乙\n丙丁\n\n戊己")
                with patch.object(page, "_render_current_preview"):
                    page._rebuild_layout()

                self.assertEqual(len(page._boards), 1)
                self.assertEqual(page._custom_result.ignored_pages, 1)
                self.assertEqual(page._board_parameter_tabs.count(), 3)
                self.assertEqual(page._board_parameter_tabs.tabText(0), "第 1 版")
                self.assertEqual(page._board_parameter_tabs.tabText(1), "＋")
                self.assertEqual(page._board_parameter_tabs.tabText(2), "输出参数")
                self.assertIn("无参数，将忽略", page._ignored_pages_label.text())

                with patch(
                    "ui.pages.custom_scripture_layout_page.QInputDialog.getItem",
                    return_value=("复制第 1 版参数", True),
                ):
                    page._add_board_parameters()
                with patch.object(page, "_render_current_preview"):
                    page._rebuild_layout()

                self.assertEqual(len(page._custom_board_parameters), 2)
                self.assertEqual(len(page._boards), 2)
                self.assertEqual(page._board_parameter_tabs.count(), 4)
                self.assertEqual(page._board_parameter_tabs.tabText(1), "第 2 版")
                self.assertEqual(page._board_parameter_tabs.tabText(2), "＋")
                self.assertEqual(page._board_parameter_tabs.tabText(3), "输出参数")
                self.assertNotIn("忽略", page._ignored_pages_label.text())
            finally:
                page.shutdown()
                page.deleteLater()

    def test_parameter_workspace_uses_one_tab_row_for_boards_and_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            page = self._page(directory)
            try:
                page.resize(1600, 900)
                page.show()
                self.app.processEvents()

                self.assertEqual(page._settings_tabs.count(), 2)
                self.assertEqual(page._settings_tabs.tabText(0), "版面参数")
                self.assertEqual(page._settings_tabs.tabText(1), "输出参数")
                self.assertEqual(page._settings_tabs.currentIndex(), 0)
                self.assertTrue(page._settings_tabs.tabBar().isHidden())
                self.assertEqual(page._board_parameter_tabs.count(), 3)
                self.assertEqual(page._board_parameter_tabs.tabText(0), "第 1 版")
                self.assertEqual(page._board_parameter_tabs.tabText(1), "＋")
                self.assertEqual(page._board_parameter_tabs.tabText(2), "输出参数")
                self.assertLess(page._board_parameter_tabs.height(), 48)

                left_width, preview_width, right_width = page._splitter.sizes()
                self.assertGreater(left_width, 0)
                self.assertGreater(preview_width, 0)
                self.assertEqual(right_width, 640)

                parameter_scroll = page.findChild(
                    QScrollArea,
                    "customBoardParameterScroll",
                )
                self.assertIsNotNone(parameter_scroll)
                self.assertGreater(parameter_scroll.viewport().height(), 500)
                self.assertEqual(parameter_scroll.horizontalScrollBar().maximum(), 0)
                self.assertFalse(page._output_name_edit.isVisible())

                page._board_parameter_tabs.setCurrentIndex(2)
                self.app.processEvents()
                self.assertEqual(page._settings_tabs.currentIndex(), 1)
                self.assertTrue(page._output_name_edit.isVisible())
                self.assertFalse(parameter_scroll.isVisible())
            finally:
                page.shutdown()
                page.deleteLater()

    def test_output_tab_moves_after_new_board_and_plus_cancel_restores_board(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            page = self._page(directory)
            try:
                with patch(
                    "ui.pages.custom_scripture_layout_page.QInputDialog.getItem",
                    return_value=("", False),
                ):
                    page._board_parameter_tabs.setCurrentIndex(1)
                self.assertEqual(len(page._custom_board_parameters), 1)
                self.assertEqual(page._board_parameter_tabs.currentIndex(), 0)

                with patch(
                    "ui.pages.custom_scripture_layout_page.QInputDialog.getItem",
                    return_value=("复制第 1 版参数", True),
                ):
                    page._board_parameter_tabs.setCurrentIndex(1)

                self.assertEqual(len(page._custom_board_parameters), 2)
                self.assertEqual(page._board_parameter_tabs.currentIndex(), 1)
                self.assertEqual(page._board_parameter_tabs.tabText(2), "＋")
                self.assertEqual(page._board_parameter_tabs.tabText(3), "输出参数")
                self.assertEqual(page._settings_tabs.currentIndex(), 0)
            finally:
                page.shutdown()
                page.deleteLater()

    def test_refresh_keeps_output_page_selected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            page = self._page(directory)
            try:
                page._board_parameter_tabs.setCurrentIndex(2)
                self.assertEqual(page._settings_tabs.currentIndex(), 1)

                page._refresh_board_parameter_tabs()

                self.assertEqual(page._board_parameter_tabs.currentIndex(), 2)
                self.assertEqual(
                    page._board_parameter_tabs.tabText(
                        page._board_parameter_tabs.currentIndex()
                    ),
                    "输出参数",
                )
                self.assertEqual(page._settings_tabs.currentIndex(), 1)
            finally:
                page.shutdown()
                page.deleteLater()

    def test_custom_preview_inherits_dynamic_fit_zoom_floor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            page = self._page(directory)
            try:
                page.resize(1600, 900)
                page.show()
                page._preview_image = QImage(
                    3600,
                    28000,
                    QImage.Format.Format_RGB32,
                )
                page._preview_image.fill(Qt.GlobalColor.white)
                page._preview_document_size = QSize(3600, 28000)
                page._preview_zoom = page._fit_preview_zoom()
                page._apply_preview_zoom()
                self.app.processEvents()

                self.assertLess(page._preview_zoom, 0.1)
                with patch.object(page, "_schedule_quality_preview") as schedule:
                    page._zoom_preview_at(
                        QPointF(page._preview_scroll.viewport().rect().center()),
                        1 / 1.15,
                    )
                schedule.assert_not_called()
            finally:
                page.shutdown()
                page.deleteLater()

    def test_preview_navigation_keeps_parameter_tab_and_board_number_in_sync(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            page = self._page(directory)
            try:
                page._custom_board_parameters.append(
                    page._custom_board_parameters[0]
                )
                page._refresh_board_parameter_tabs(0)
                first_board = MagicMock(character_count=2)
                second_board = MagicMock(character_count=3)
                page._boards = (first_board, second_board)
                page._current_board = 0
                page._editing_board_index = 0
                page._update_board_controls()

                page._change_board(1)

                self.assertEqual(page._current_board, 1)
                self.assertEqual(page._editing_board_index, 1)
                self.assertEqual(page._board_parameter_tabs.currentIndex(), 1)
                self.assertEqual(page._board_label.text(), "2 / 2")
                self.assertIn("第 2 版", page._board_parameter_heading.text())
            finally:
                page.shutdown()
                page.deleteLater()

    def test_mismatch_notice_does_not_cancel_generation_flow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            page = self._page(directory)
            try:
                with patch(
                    "ui.pages.custom_scripture_layout_page.QMessageBox.information"
                ) as information:
                    result = page._confirm_page_mismatch(3, 2)
                self.assertIsNone(result)
                self.assertEqual(information.call_count, 1)
                self.assertIn("忽略", information.call_args.args[2])
            finally:
                page.shutdown()
                page.deleteLater()

    def test_home_route_opens_custom_layout_after_general_layout(self) -> None:
        page = object()
        window = MagicMock()
        window._custom_scripture_layout_page = page

        MainWindow._open_tool(window, "custom_layout")

        window._stack.setCurrentWidget.assert_called_once_with(page)
        self.assertIn(
            "定制经文排版",
            window.statusBar.return_value.showMessage.call_args.args[0],
        )


if __name__ == "__main__":
    unittest.main()
