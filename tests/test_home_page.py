"""首页快捷入口与制作流程回归测试。"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QApplication, QFrame, QHeaderView, QLabel, QScrollArea

import config
from ui.pages.home_page import HomePage, LibraryScanProgress, scan_library_summaries
from ui.theme import apply_theme


class HomePageTests(unittest.TestCase):
    """验证新建字库作为第 0 步时的入口、状态和最小窗口布局。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])
        apply_theme(cls.app)

    @staticmethod
    def _library() -> dict[str, object]:
        return {
            "name": "测试字库",
            "path": r"D:\字库\测试字库",
            "characters": 20,
            "variants": 24,
            "imported": 24,
            "optimized": 18,
            "pending_optimization": 6,
            "pending_review": 8,
            "review_admitted": 18,
            "pending_coordination": 4,
            "reviewed": 10,
            "coordination_admitted": 10,
            "finished": 6,
            "completed": 6,
            "coordinated": 6,
            "export_ready": 6,
            "metadata": {"DPI": 300, "画布宽": 250, "画布高": 250},
        }

    def test_home_title_displays_version(self) -> None:
        page = HomePage()
        title = page.findChild(QLabel, "homeTitle")
        self.assertIsNotNone(title)
        assert title is not None
        self.assertEqual(title.text(), "欢迎使用字库编辑器-V1.0")
        page.deleteLater()

    def test_scan_library_summaries_counts_stages_and_valid_finished_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            library_dir = root / "阶段统计字库"
            preview_dir = library_dir / config.DIR_INTERMEDIATE_FILES
            reviewed_dir = library_dir / config.DIR_REVIEWED_FILES
            finished_dir = library_dir / config.DIR_FINISHED_FILES
            preview_dir.mkdir(parents=True)
            reviewed_dir.mkdir(parents=True)
            finished_dir.mkdir(parents=True)
            (preview_dir / "已优化.png").write_bytes(b"png")
            for filename in ("已审核.png", "缺成品已审核.png", "越界成品已审核.png"):
                (reviewed_dir / filename).write_bytes(b"png")
            (finished_dir / "可导出.png").write_bytes(b"png")
            (finished_dir / "墨色待确认.png").write_bytes(b"png")
            valid_ink_record = {
                "启用": True,
                "方法": "视觉墨量规范化",
                "方法版本": 2,
                "基准": 215.0,
                "保存后复测": True,
                "保存后墨色": 215.0,
                "是否达标": True,
            }
            details: dict[str, object] = {
                "pending": {"状态": config.STATUS_PENDING_OPTIMIZATION},
                "optimized": {
                    "状态": config.STATUS_PENDING_MANUAL_REVIEW,
                    "中间文件": "已优化.png",
                },
                "reviewed": {
                    "状态": config.STATUS_REVIEWED,
                    "审核文件": "已审核.png",
                },
                "ready": {
                    "状态": config.STATUS_FINISHED,
                    "成品文件": "可导出.png",
                    "整体协调参数": {"墨色协调": valid_ink_record},
                },
                "ink_pending": {
                    "状态": config.STATUS_FINISHED,
                    "成品文件": "墨色待确认.png",
                },
                "missing": {
                    "状态": config.STATUS_FINISHED,
                    "审核文件": "缺成品已审核.png",
                    "成品文件": "不存在.png",
                },
                "unsafe": {
                    "状态": config.STATUS_FINISHED,
                    "审核文件": "越界成品已审核.png",
                    "成品文件": "../目录外.png",
                },
                "unknown": {"状态": "未知状态"},
                "broken": {"状态": ""},
            }
            groups: dict[str, list[str]] = {}
            for index, (variant_id, detail) in enumerate(details.items()):
                char = chr(ord("甲") + index)
                detail["变体ID"] = variant_id
                detail["归属字"] = char
                groups[char] = [variant_id]
            library_data = {
                "数据版本": 3,
                "库名": "阶段统计字库",
                "变体详情": details,
                "字形组索引": groups,
                "元数据": {"DPI": 300, "画布宽": 250, "画布高": 250},
                "会话": {},
                "整体协调": {
                    "墨色统一启用": True,
                    "墨色基准": 215.0,
                    "墨色方法": "视觉墨量规范化",
                    "墨色方法版本": 2,
                },
            }
            (library_dir / "阶段统计字库.json").write_text(
                json.dumps(library_data, ensure_ascii=False),
                encoding="utf-8",
            )

            with patch.object(config, "ZIKU_ROOT", str(root)):
                summaries = scan_library_summaries()

            self.assertEqual(len(summaries), 1)
            summary = summaries[0]
            self.assertEqual(summary["variants"], len(details))
            self.assertEqual(summary["imported"], len(details))
            self.assertEqual(summary["pending_optimization"], 3)
            self.assertEqual(summary["optimized"], 6)
            self.assertEqual(summary["review_admitted"], 6)
            self.assertEqual(summary["pending_review"], 1)
            self.assertEqual(summary["reviewed"], 5)
            self.assertEqual(summary["coordination_admitted"], 5)
            self.assertEqual(summary["pending_coordination"], 4)
            self.assertEqual(summary["coordinated"], 1)
            self.assertEqual(
                summary["pending_optimization"] + summary["optimized"],
                summary["variants"],
            )
            self.assertEqual(
                summary["pending_review"] + summary["reviewed"],
                summary["review_admitted"],
            )
            self.assertEqual(
                summary["pending_coordination"] + summary["coordinated"],
                summary["coordination_admitted"],
            )
            self.assertEqual(summary["export_ready"], 2)

    def test_scan_library_summaries_reports_monotonic_real_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            expected_counts = {"甲字库": 2, "乙字库": 3}
            for library_name, glyph_count in expected_counts.items():
                library_dir = root / library_name
                library_dir.mkdir()
                details = {
                    f"variant-{index}": {
                        "变体ID": f"variant-{index}",
                        "归属字": library_name[0],
                        "状态": config.STATUS_PENDING_OPTIMIZATION,
                    }
                    for index in range(glyph_count)
                }
                data = {
                    "数据版本": 3,
                    "库名": library_name,
                    "变体详情": details,
                    "字形组索引": {library_name[0]: list(details)},
                    "元数据": {},
                    "会话": {},
                    "整体协调": {},
                }
                (library_dir / f"{library_name}.json").write_text(
                    json.dumps(data, ensure_ascii=False),
                    encoding="utf-8",
                )

            updates: list[LibraryScanProgress] = []
            with patch.object(config, "ZIKU_ROOT", str(root)):
                summaries = scan_library_summaries(updates.append)

            self.assertEqual(len(summaries), 2)
            self.assertTrue(updates)
            self.assertEqual(updates[0].phase, "discovering")
            self.assertEqual(updates[-1].phase, "complete")
            self.assertEqual(updates[-1].overall_current, 2000)
            self.assertEqual(updates[-1].overall_total, 2000)
            logical_values = [update.overall_current for update in updates]
            self.assertEqual(logical_values, sorted(logical_values))
            completed_libraries = {
                update.library_name: update.glyph_current
                for update in updates
                if update.phase == "processing"
                and update.glyph_current == update.glyph_total
            }
            self.assertEqual(completed_libraries, expected_counts)

    def test_library_scan_progress_is_visible_and_restores_refresh(self) -> None:
        page = HomePage()
        page.set_loading(True)

        self.assertFalse(page._scan_progress_bar.isHidden())
        self.assertFalse(page._refresh_button.isEnabled())
        progress = LibraryScanProgress(
            "processing",
            "大字库",
            library_index=1,
            library_total=2,
            glyph_current=25,
            glyph_total=100,
        )
        page.set_scan_progress(progress)
        self.assertEqual(page._scan_progress_bar.maximum(), 2000)
        self.assertEqual(page._scan_progress_bar.value(), 250)
        self.assertIn("字形 25/100", page._scan_progress_bar.format())
        self.assertEqual(page._summary_label.text(), "正在核对字库 1/2")

        page.set_libraries([])
        self.assertTrue(page._scan_progress_bar.isHidden())
        self.assertTrue(page._refresh_button.isEnabled())

        page.set_loading(True)
        page.set_loading_failed()
        self.assertTrue(page._scan_progress_bar.isHidden())
        self.assertTrue(page._refresh_button.isEnabled())
        self.assertEqual(
            page._summary_label.text(),
            "字库信息核对失败，请点击重新核对重试",
        )
        page.deleteLater()

    def test_large_library_delete_shows_indeterminate_progress_and_locks_home(self) -> None:
        page = HomePage()
        page.set_libraries([self._library()])

        page.set_deleting(True, "大字库")

        self.assertFalse(page._scan_progress_bar.isHidden())
        self.assertEqual(page._scan_progress_bar.minimum(), 0)
        self.assertEqual(page._scan_progress_bar.maximum(), 0)
        self.assertIn("大字库", page._scan_progress_bar.format())
        self.assertFalse(page._refresh_button.isEnabled())
        self.assertFalse(page._table.isEnabled())
        self.assertFalse(page._flow_group.isEnabled())

        page.set_delete_progress("正在执行系统回收站操作…")
        self.assertEqual(
            page._scan_progress_bar.format(),
            "正在执行系统回收站操作…",
        )

        page.set_deleting(False)
        self.assertTrue(page._scan_progress_bar.isHidden())
        self.assertTrue(page._refresh_button.isEnabled())
        self.assertTrue(page._table.isEnabled())
        self.assertTrue(page._flow_group.isEnabled())
        self.assertIn("当前选择", page._summary_label.text())
        page.deleteLater()

    def test_library_table_shows_stage_completion_counts_and_full_tooltip(self) -> None:
        page = HomePage()
        page.set_libraries([self._library()])

        self.assertEqual(page._table.columnCount(), 10)
        stage_columns = (
            (4, "总字数", "24"),
            (5, "自动优化", "18/24"),
            (6, "手工审核", "10/18"),
            (7, "整体协调", "6/10"),
            (8, "可导出", "6"),
        )
        for column, title, value in stage_columns:
            with self.subTest(column=column, title=title):
                self.assertEqual(page._table.horizontalHeaderItem(column).text(), title)
                item = page._table.item(0, column)
                self.assertIsNotNone(item)
                self.assertEqual(item.text(), value)
                self.assertTrue(
                    item.textAlignment() & Qt.AlignmentFlag.AlignHCenter
                )
                self.assertTrue(
                    item.textAlignment() & Qt.AlignmentFlag.AlignVCenter
                )
                self.assertTrue(item.toolTip().strip())

        action_box = page._table.cellWidget(0, 9)
        self.assertIsNotNone(action_box)
        self.assertIsNotNone(action_box.layout())
        self.assertEqual(action_box.layout().spacing(), 10)
        page.deleteLater()

    def test_select_library_uses_path_and_selects_newly_added_row(self) -> None:
        page = HomePage()
        first = self._library()
        second = dict(first)
        second["name"] = "新建字库"
        second["path"] = r"D:\字库\新建字库"
        page.set_libraries([first, second])

        self.assertEqual(page._selected_name, "测试字库")
        self.assertTrue(page.select_library(str(second["path"])))
        self.assertEqual(page._selected_name, "新建字库")
        self.assertEqual(page._selected_path, second["path"])
        self.assertEqual(page._table.currentRow(), 1)
        self.assertFalse(page.select_library(r"D:\字库\不存在"))
        page.deleteLater()

    def test_library_action_buttons_keep_visible_gap_with_application_theme(self) -> None:
        previous_style = self.app.style().objectName()
        previous_palette = self.app.palette()
        previous_stylesheet = self.app.styleSheet()
        page = None
        try:
            apply_theme(self.app)
            page = HomePage()
            page.resize(1600, 900)
            page.show()
            page.set_libraries([self._library()])
            self.app.processEvents()

            action_box = page._table.cellWidget(0, 9)
            self.assertIsNotNone(action_box)
            action_layout = action_box.layout()
            self.assertIsNotNone(action_layout)
            parameter_button = action_layout.itemAt(0).widget()
            delete_button = action_layout.itemAt(1).widget()
            self.assertIsNotNone(parameter_button)
            self.assertIsNotNone(delete_button)
            actual_gap = (
                delete_button.geometry().left()
                - parameter_button.geometry().right()
                - 1
            )
            self.assertGreaterEqual(actual_gap, 8)
            self.assertGreaterEqual(
                action_box.width(),
                action_layout.minimumSize().width(),
            )
            self.assertTrue(action_box.rect().contains(parameter_button.geometry()))
            self.assertTrue(action_box.rect().contains(delete_button.geometry()))
        finally:
            if page is not None:
                page.close()
                page.deleteLater()
            self.app.setStyle(previous_style)
            self.app.setPalette(previous_palette)
            self.app.setStyleSheet(previous_stylesheet)

    def test_library_table_columns_are_adjustable_and_keep_user_widths(self) -> None:
        page = HomePage()
        library = self._library()
        library["metadata"] = {
            "DPI": 72,
            "画布宽": 369,
            "画布高": 312,
            "成品宽度毫米": 130.0,
            "成品高度毫米": 110.0,
        }
        page.set_libraries([library])

        header = page._table.horizontalHeader()
        for column in range(page._table.columnCount()):
            with self.subTest(column=column):
                self.assertEqual(
                    header.sectionResizeMode(column),
                    QHeaderView.ResizeMode.Interactive,
                )

        for column in (1, 2, 3):
            item = page._table.item(0, column)
            self.assertIsNotNone(item)
            required_width = (
                page._table.fontMetrics().horizontalAdvance(item.text()) + 24
            )
            self.assertGreaterEqual(page._table.columnWidth(column), required_width)
            if column in (2, 3):
                self.assertEqual(item.toolTip(), item.text())

        page._table.setColumnWidth(2, 333)
        page.set_libraries([library])
        self.assertEqual(page._table.columnWidth(2), 333)
        page.deleteLater()

    def test_library_name_column_fills_remaining_width_on_load(self) -> None:
        page = HomePage()
        page.resize(1600, 900)
        page.show()
        page.set_libraries([self._library()])
        self.app.processEvents()

        header = page._table.horizontalHeader()
        self.assertGreater(page._table.columnWidth(0), 180)
        self.assertLessEqual(
            abs(header.length() - page._table.viewport().width()),
            1,
        )
        operation_right = (
            header.sectionViewportPosition(9) + header.sectionSize(9)
        )
        self.assertLessEqual(
            abs(operation_right - page._table.viewport().width()),
            1,
        )

        fitted_width = page._table.columnWidth(0)
        page.resize(1700, 900)
        self.app.processEvents()
        self.assertGreater(page._table.columnWidth(0), fitted_width)
        self.assertLessEqual(
            abs(header.length() - page._table.viewport().width()),
            1,
        )

        page._table.setColumnWidth(0, 240)
        self.app.processEvents()
        page.set_libraries([self._library()])
        self.app.processEvents()
        self.assertEqual(page._table.columnWidth(0), 240)
        page.deleteLater()

    def test_create_moves_from_top_tools_to_stage_zero(self) -> None:
        page = HomePage()
        page.resize(1600, 900)
        page.show()
        page.set_libraries([self._library()])
        self.app.processEvents()

        self.assertNotIn("create", page._tool_cards)
        expected_tools = (
            ("layout", "通用经文排版"),
            ("custom_layout", "定制经文排版"),
            ("statistics", "文字统计"),
            ("image_lab", "图片实验室"),
            ("settings", "设置"),
            ("help", "使用说明"),
        )
        self.assertEqual(
            list(page._tool_cards),
            [key for key, _title in expected_tools],
        )
        tool_cards = [page._tool_cards[key] for key, _title in expected_tools]
        self.assertEqual(
            [
                card.findChild(type(page._summary_label), "cardTitle").text()
                for card in tool_cards
            ],
            [title for _key, title in expected_tools],
        )
        positions = [card.mapTo(page, QPoint()).x() for card in tool_cards]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(len(set(positions)), len(positions))
        self.assertEqual(
            list(page._stage_cards),
            ["create", "import", "optimization", "review", "consistency", "export"],
        )
        self.assertEqual(
            [card._mark_label.text() for card in page._stage_cards.values()],
            ["0", "1", "2", "3", "4", "⇩"],
        )
        self.assertEqual(
            [card._title_label.text() for card in page._stage_cards.values()],
            ["新建字库", "字库添加", "自动优化", "手工审核", "整体协调", "导出最终成品"],
        )

        tool_requests: list[str] = []
        stage_requests: list[tuple[str, str, str]] = []
        page.tool_requested.connect(tool_requests.append)
        page.stage_requested.connect(
            lambda route, name, path: stage_requests.append((route, name, path))
        )
        page._stage_cards["create"]._action_button.click()
        page._stage_cards["import"]._action_button.click()

        self.assertEqual(tool_requests, ["create"])
        self.assertEqual(
            stage_requests,
            [("import", "测试字库", r"D:\字库\测试字库")],
        )
        page.close()
        page.deleteLater()

    def test_stage_cards_use_current_phase_status_pairs(self) -> None:
        page = HomePage()
        page.set_libraries([self._library()])

        expected = {
            "optimization": ("待优化 6", "18 / 24"),
            "review": ("待审核 8 · 前序 6", "10 / 18"),
            "consistency": ("待协调 4 · 前序 14", "6 / 10"),
        }
        for route, (status_text, metric_text) in expected.items():
            with self.subTest(route=route):
                card = page._stage_cards[route]
                self.assertEqual(card._status_label.text(), status_text)
                metric = card.findChild(type(page._summary_label), "metricValue")
                self.assertIsNotNone(metric)
                self.assertEqual(metric.text(), metric_text)
        page.deleteLater()

    def test_empty_library_keeps_only_stage_zero_available(self) -> None:
        page = HomePage()
        page.set_libraries([])

        self.assertEqual(page._table.rowCount(), 0)
        self.assertEqual(
            page._summary_label.text(),
            "暂无可选择的字库，请先新建字库",
        )
        self.assertTrue(page._stage_cards["create"]._action_button.isEnabled())
        for route in ("import", "optimization", "review", "consistency", "export"):
            card = page._stage_cards[route]
            self.assertFalse(card._action_button.isEnabled())
            self.assertEqual(card._status_label.text(), "等待新建字库")

        tool_requests: list[str] = []
        stage_requests: list[tuple[str, str, str]] = []
        page.tool_requested.connect(tool_requests.append)
        page.stage_requested.connect(
            lambda route, name, path: stage_requests.append((route, name, path))
        )
        page._stage_cards["create"]._action_button.click()
        page._stage_cards["import"]._action_button.click()

        self.assertEqual(tool_requests, ["create"])
        self.assertEqual(stage_requests, [])
        page.deleteLater()

    def test_six_stages_fit_without_page_horizontal_scroll(self) -> None:
        page = HomePage()
        page.show()
        for width, height in ((1100, 720), (1600, 900)):
            with self.subTest(size=(width, height)):
                page.resize(width, height)
                page.set_libraries([self._library()])
                self.app.processEvents()

                cards = list(page._stage_cards.values())
                self.assertEqual(len(cards), 6)
                for card in cards:
                    self.assertGreaterEqual(
                        card._title_label.width(),
                        card._title_label.fontMetrics().horizontalAdvance(
                            card._title_label.text()
                        ),
                    )
                    self.assertGreaterEqual(
                        card._action_button.width(),
                        card._action_button.fontMetrics().horizontalAdvance(
                            card._action_button.text()
                        )
                        + 20,
                    )
                self.assertLessEqual(
                    max(card.width() for card in cards)
                    - min(card.width() for card in cards),
                    1,
                )
                self.assertEqual(len({card.height() for card in cards}), 1)
                buttons = [card._action_button for card in cards]
                self.assertLessEqual(
                    max(button.width() for button in buttons)
                    - min(button.width() for button in buttons),
                    1,
                )
                self.assertLessEqual(
                    max(button.height() for button in buttons)
                    - min(button.height() for button in buttons),
                    1,
                )
                marks = [card._mark_label for card in cards]
                self.assertEqual(len({(mark.width(), mark.height()) for mark in marks}), 1)
                self.assertEqual(
                    len({card._title_label.font().toString() for card in cards}),
                    1,
                )
                for left, right in zip(cards, cards[1:]):
                    left_x = left.mapTo(page, QPoint()).x()
                    right_x = right.mapTo(page, QPoint()).x()
                    self.assertLessEqual(left_x + left.width(), right_x)

                create_card = page._stage_cards["create"]
                create_group = page._create_group
                self.assertIsNotNone(create_group)
                self.assertIsInstance(create_group, QFrame)
                self.assertEqual(create_group.objectName(), "libraryCreatePanel")
                flow_group = page._flow_group
                self.assertTrue(create_group.isAncestorOf(create_card))
                self.assertFalse(flow_group.isAncestorOf(create_card))
                for route in ("import", "optimization", "review", "consistency", "export"):
                    card = page._stage_cards[route]
                    self.assertTrue(flow_group.isAncestorOf(card))
                    card_pos = card.mapTo(flow_group, QPoint())
                    self.assertGreaterEqual(card_pos.x(), 0)
                    self.assertGreaterEqual(card_pos.y(), 0)
                    self.assertLessEqual(card_pos.x() + card.width(), flow_group.width())
                    self.assertLessEqual(card_pos.y() + card.height(), flow_group.height())
                self.assertTrue(flow_group.isAncestorOf(page._flow_header))
                self.assertEqual(page._flow_cards_layout.count(), 5)

                create_titles = [
                    label
                    for label in create_group.findChildren(
                        type(page._flow_title), "sectionTitle"
                    )
                    if label.text() == "创建新的字库项目"
                ]
                self.assertEqual(len(create_titles), 1)
                self.assertLessEqual(abs(create_group.height() - flow_group.height()), 1)

                title_x = page._flow_title.mapTo(page, QPoint()).x()
                create_x = create_card.mapTo(page, QPoint()).x()
                import_x = page._stage_cards["import"].mapTo(page, QPoint()).x()
                self.assertEqual(title_x, import_x)
                self.assertLess(create_x, title_x)
                self.assertLessEqual(
                    abs(
                        create_card.mapTo(page, QPoint()).y()
                        + create_card.height()
                        - page._stage_cards["import"].mapTo(page, QPoint()).y()
                        - page._stage_cards["import"].height()
                    ),
                    1,
                )

                scroll = page.findChild(QScrollArea, "homeScroll")
                self.assertIsNotNone(scroll)
                self.assertEqual(scroll.horizontalScrollBar().maximum(), 0)
                if width == 1100:
                    self.assertGreater(page._table.horizontalScrollBar().maximum(), 0)
                else:
                    self.assertEqual(page._table.horizontalScrollBar().maximum(), 0)

        page.set_libraries([])
        page.set_libraries([self._library()])
        self.app.processEvents()
        self.assertTrue(page._flow_title.isVisible())
        self.assertEqual(page._flow_cards_layout.count(), 5)
        self.assertEqual(
            page._flow_title.mapTo(page, QPoint()).x(),
            page._stage_cards["import"].mapTo(page, QPoint()).x(),
        )
        self.assertEqual(
            page._stage_cards["create"].height(),
            page._stage_cards["import"].height(),
        )
        page.deleteLater()


if __name__ == "__main__":
    unittest.main()
