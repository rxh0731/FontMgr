"""自动优化页面首次显示与预览适配回归测试。"""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PIL import Image
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QIcon, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QScrollArea,
)

from core.optimizer import OptimizationCancelled
from services.batch_persistence import BatchJournalUncertainError
from ui.pages.optimization_page import OptimizationPage, _OptimizationBatchWorker
from services.background_model_service import NO_MODEL_ENGINE_ID
from services.workflow_status_service import (
    OPTIMIZATION_STATUS_FILTERS,
    PHASE_FILTER_ALL,
    STATUS_OPTIMIZED,
)
from ui.theme import apply_theme
from services.optimization_service import CANDIDATE_TYPE_DIRECT, CANDIDATE_TYPE_OPTIMIZED


class OptimizationPageTests(unittest.TestCase):
    """验证隐藏页面首次渲染后仍会按最终布局重新适配。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])
        apply_theme(cls.app)

    def test_first_show_refits_original_preview_to_final_label_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "原图.png"
            Image.new("RGB", (1600, 900), "black").save(path)
            page = OptimizationPage()
            page._current_item = {"原始路径": str(path)}

            page._render_original_preview()
            page.resize(1600, 900)
            page.show()
            self.app.processEvents()
            self.app.processEvents()

            pixmap = page._original_preview.pixmap()
            self.assertFalse(pixmap.isNull())
            self.assertEqual(pixmap.width(), max(1, page._original_preview.width() - 8))
            self.assertEqual(pixmap.height(), max(1, page._original_preview.height() - 8))
            page.close()
            page.deleteLater()

    def test_selected_and_candidate_previews_default_to_white_background(self) -> None:
        page = OptimizationPage()
        transparent_image = Image.new("RGBA", (20, 20), (0, 0, 0, 0))

        selected_preview = page._compose_preview(
            transparent_image,
            (40, 30),
            page._preview_background == "透明底",
        )
        candidate_preview = page._compose_preview(
            transparent_image,
            page.CARD_SIZE,
            page.CANDIDATE_TRANSPARENT_BACKGROUND,
        )

        self.assertEqual(page._preview_background, "白底")
        self.assertEqual(selected_preview.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(candidate_preview.getpixel((0, 0)), (255, 255, 255))
        page.deleteLater()

    def test_processing_engine_defaults_to_no_learning_model(self) -> None:
        page = OptimizationPage()

        self.assertEqual(page._engine_combo.currentData(), NO_MODEL_ENGINE_ID)
        self.assertEqual(page._engine_combo.currentText(), "无学习模型")
        self.assertIn("传统图像管线", page._engine_state_label.text())
        self.assertIs(page._engine_combo.parentWidget(), page._scheme_fixed_content)
        self.assertIsInstance(page._scheme_label, QPlainTextEdit)
        self.assertTrue(page._scheme_label.isReadOnly())
        self.assertFalse(page._scheme_panel.findChildren(QScrollArea))
        self.assertFalse(page._model_settings_panel.isVisible())
        page.deleteLater()

    def test_current_glyph_heading_is_removed(self) -> None:
        page = OptimizationPage()
        item = {
            "键": "variant-12",
            "归属字": "永",
            "变体序号": 12,
            "原始文件名": "重污染反相样本_001.tif",
            "原始路径": "",
        }

        page._select_item(item)

        visible_texts = [label.text() for label in page.findChildren(QLabel)]
        self.assertFalse(any(text.startswith("当前字形：") for text in visible_texts))
        self.assertFalse(any("原始文件：" in text for text in visible_texts))
        self.assertFalse(hasattr(page, "_current_label"))
        self.assertFalse(hasattr(page, "_file_label"))
        page.deleteLater()

    def test_current_scheme_details_share_one_scrollable_text_box(self) -> None:
        page = OptimizationPage()
        candidate = {
            "方案名": "结构保护候选",
            "得分": 87.4,
            "处理类型": CANDIDATE_TYPE_OPTIMIZED,
            "方案": {
                "处理引擎": {
                    "标识": "none",
                    "名称": "无学习模型",
                    "版本": "内置",
                },
                "路线来源": "传统图像管线",
                "自动校正": {"反相": True},
                "预处理": {"转灰度": True},
                "L3": {"算法": "Otsu", "参数": {"偏移": 2}},
            },
        }

        page._show_candidates([candidate])

        text = page._scheme_label.toPlainText()
        self.assertIn("候选名称：候选1　结构保护候选", text)
        self.assertIn("综合得分：87.4", text)
        self.assertIn("处理引擎：无学习模型 · 内置", text)
        self.assertIn("基础路线：传统图像管线", text)
        self.assertIn("算法组合", text)
        self.assertIn("L3：Otsu（偏移=2）", text)
        self.assertEqual(text.count("处理引擎："), 1)
        self.assertEqual(
            page._scheme_label.lineWrapMode(),
            QPlainTextEdit.LineWrapMode.WidgetWidth,
        )
        for removed_attribute in ("_scheme_title", "_score_label", "_route_label"):
            self.assertFalse(hasattr(page, removed_attribute))

        direct_candidate = {
            "方案名": "原图已有透明区，直接采用",
            "得分": 90.0,
            "处理类型": CANDIDATE_TYPE_DIRECT,
            "方案": {
                "处理引擎": {
                    "标识": "none",
                    "名称": "无学习模型",
                    "版本": "内置",
                },
                "透明来源": "标准Alpha",
            },
        }
        page._show_candidates([direct_candidate])
        direct_text = page._scheme_label.toPlainText()
        self.assertIn("基础路线：原图直接采用", direct_text)
        self.assertIn("算法组合", direct_text)
        self.assertIn("处理类型：原图直接采用", direct_text)

        page._clear_candidates("等待重新生成候选")
        self.assertEqual(
            page._scheme_label.toPlainText(),
            "请从候选效果中选择一张图片。",
        )
        page.deleteLater()

    def test_structure_risk_candidate_is_explicit_in_card_and_details(self) -> None:
        page = OptimizationPage()
        candidate = {
            "方案名": "保守Otsu",
            "得分": 72.4,
            "处理类型": CANDIDATE_TYPE_OPTIMIZED,
            "结构复核": {
                "状态": "需人工核对",
                "阶段": "原尺寸复核",
                "原因": "有意义孔洞仅保留0.0%",
                "风险等级": 1,
            },
            "方案": {"预处理": {}, "L3": {"算法": "Otsu", "参数": {}}},
            "图像": Image.new("RGBA", (8, 8), (0, 0, 0, 255)),
        }

        page._show_candidates([candidate])

        card = page._candidate_list.item(0)
        self.assertIn("结构需核对", card.text())
        self.assertEqual(card.foreground().color(), QColor(page.STRUCTURE_RISK_COLOR))
        self.assertIn("有意义孔洞仅保留0.0%", card.toolTip())
        self.assertIn("得分只用于候选排序", card.toolTip())
        details = page._scheme_label.toPlainText()
        self.assertIn("结构复核：需人工核对", details)
        self.assertIn("复核阶段：原尺寸复核", details)
        self.assertIn("风险原因：有意义孔洞仅保留0.0%", details)
        self.assertEqual(page._save_button.text(), "确认风险并保存")
        self.assertTrue(page._explore_button.isEnabled())

        safe = dict(candidate)
        safe.pop("结构复核")
        page._show_candidates([safe])
        self.assertNotIn("结构需核对", page._candidate_list.item(0).text())
        self.assertEqual(page._save_button.text(), "采用并保存")
        page.deleteLater()

    def test_risk_candidate_cancel_does_not_start_save(self) -> None:
        page = OptimizationPage()
        page._service = MagicMock()
        page._current_item = {
            "键": "variant-risk",
            "显示状态": "待优化",
            "原始路径": "D:/测试/variant-risk.png",
        }
        page._candidates = [{
            "方案名": "风险候选",
            "处理类型": CANDIDATE_TYPE_OPTIMIZED,
            "结构复核": {"状态": "需人工核对", "原因": "结构告警"},
        }]
        page._selected_index = 0

        with (
            patch.object(page, "_confirm_structure_risk", return_value=False),
            patch.object(page, "_start_task") as start_task,
        ):
            page._save_selected()

        start_task.assert_not_called()
        page._service.save_selection.assert_not_called()
        page.deleteLater()

    def test_candidate_grid_keeps_empty_slots_when_candidates_are_fewer(self) -> None:
        def candidates(count: int) -> list[dict[str, object]]:
            return [
                {
                    "方案名": f"方案{index + 1}",
                    "得分": 80.0 - index,
                    "处理类型": CANDIDATE_TYPE_OPTIMIZED,
                    "方案": {
                        "处理引擎": {
                            "标识": "none",
                            "名称": "无学习模型",
                            "版本": "内置",
                        },
                        "路线来源": "传统图像管线",
                    },
                }
                for index in range(count)
            ]

        page = OptimizationPage()
        page.resize(1600, 900)
        page.show()
        page._show_candidates(candidates(2))
        for _ in range(4):
            self.app.processEvents()

        viewport = page._candidate_list.viewport()
        usable_width = viewport.width()
        if not page._candidate_list.verticalScrollBar().isVisible():
            usable_width -= page._candidate_list.verticalScrollBar().sizeHint().width()
        grid_size = page._candidate_list.gridSize()
        first_rect = page._candidate_list.visualItemRect(page._candidate_list.item(0))
        second_rect = page._candidate_list.visualItemRect(page._candidate_list.item(1))
        self.assertEqual(page._candidate_columns, 4)
        self.assertEqual(grid_size.width(), (usable_width - 1) // 4)
        self.assertEqual(first_rect.top(), second_rect.top())
        self.assertEqual(
            second_rect.center().x() - first_rect.center().x(),
            grid_size.width(),
        )
        self.assertLess(second_rect.center().x(), usable_width // 2)
        two_candidate_centers = (first_rect.center().x(), second_rect.center().x())

        page._show_candidates(candidates(4))
        for _ in range(4):
            self.app.processEvents()
        four_candidate_centers = tuple(
            page._candidate_list.visualItemRect(page._candidate_list.item(index)).center().x()
            for index in range(2)
        )
        self.assertEqual(two_candidate_centers, four_candidate_centers)

        observed_columns: list[int] = []
        for width, height, expected_columns in ((1100, 720, 2), (1600, 900, 4)):
            with self.subTest(size=(width, height)):
                page.resize(width, height)
                page._show_candidates(candidates(8))
                for _ in range(4):
                    self.app.processEvents()
                observed_columns.append(page._candidate_columns)
                viewport_width = page._candidate_list.viewport().width()
                if not page._candidate_list.verticalScrollBar().isVisible():
                    viewport_width -= page._candidate_list.verticalScrollBar().sizeHint().width()
                self.assertEqual(page._candidate_columns, expected_columns)
                row_top = page._candidate_list.visualItemRect(
                    page._candidate_list.item(0)
                ).top()
                first_row = [
                    page._candidate_list.visualItemRect(page._candidate_list.item(index))
                    for index in range(page._candidate_list.count())
                    if page._candidate_list.visualItemRect(
                        page._candidate_list.item(index)
                    ).top() == row_top
                ]
                self.assertEqual(len(first_row), expected_columns)
                self.assertTrue(all(rect.left() >= 0 for rect in first_row))
                self.assertTrue(all(rect.right() < viewport_width for rect in first_row))
                self.assertEqual(page._candidate_list.horizontalScrollBar().maximum(), 0)
                if width == 1100:
                    self.assertTrue(page._candidate_list.verticalScrollBar().isVisible())

        self.assertEqual(observed_columns, [2, 4])
        page._candidate_list.setCurrentRow(5)
        page.resize(1100, 720)
        for _ in range(4):
            self.app.processEvents()
        self.assertEqual(page._candidate_list.currentRow(), 5)
        current_rect = page._candidate_list.visualItemRect(
            page._candidate_list.currentItem()
        )
        self.assertTrue(page._candidate_list.viewport().rect().intersects(current_rect))
        page.close()
        page.deleteLater()

    def test_glyph_list_keeps_long_filename_score_and_state_visible(self) -> None:
        items = [
            {
                "键": "variant-12",
                "归属字": "永",
                "变体序号": 12,
                "原始文件名": "重污染反相样本_001.tif",
                "原始路径": "",
                "显示状态": "待优化",
                "得分": 100.0,
            },
            {
                "键": "variant-13",
                "归属字": "永",
                "变体序号": 13,
                "原始文件名": "普通样本_002.png",
                "原始路径": "",
                "显示状态": "自动优化稿",
                "得分": 88.8,
            },
        ]

        for width, height in ((1100, 720), (1600, 900)):
            with self.subTest(size=(width, height)):
                page = OptimizationPage()
                page._items = items
                page._status_combo.setCurrentText(PHASE_FILTER_ALL)
                page._refresh_list()
                page.resize(width, height)
                page.show()
                self.app.processEvents()
                self.app.processEvents()

                self.assertEqual(page.width(), width)
                self.assertEqual(page.height(), height)
                self.assertGreaterEqual(
                    page._list_panel.width(),
                    page.LIST_PANEL_MIN_WIDTH,
                )
                self.assertLessEqual(
                    page._list_panel.width(),
                    page.LIST_PANEL_DEFAULT_WIDTH,
                )
                self.assertTrue(page._item_tree.wordWrap())
                self.assertFalse(page._item_tree.uniformRowHeights())
                self.assertTrue(page._item_tree.rootIsDecorated())
                self.assertEqual(page._item_tree.indentation(), 14)
                self.assertEqual(page._item_tree.iconSize(), QSize(38, 38))
                self.assertEqual(
                    page._summary_label.text(),
                    "待优化 1　已优化 1",
                )
                self.assertEqual(
                    [
                        page._status_combo.itemText(index)
                        for index in range(page._status_combo.count())
                    ],
                    list(OPTIMIZATION_STATUS_FILTERS),
                )
                self.assertFalse(hasattr(page, "_marker_combo"))
                self.assertEqual(page._item_tree.columnCount(), 3)
                self.assertEqual(
                    [
                        page._item_tree.headerItem().text(column)
                        for column in range(3)
                    ],
                    ["字形与文件", "状态与提示", "得分"],
                )
                self.assertFalse(hasattr(page, "_progress_label"))

                header = page._item_tree.header()
                font_metrics = page._item_tree.fontMetrics()
                score_width = max(
                    48,
                    font_metrics.horizontalAdvance("100分") + page.TREE_COLUMN_PADDING,
                )
                state_width = max(
                    64,
                    font_metrics.horizontalAdvance(STATUS_OPTIMIZED)
                    + page.TREE_COLUMN_PADDING
                    + 16,
                )
                for column in range(3):
                    self.assertEqual(
                        header.sectionResizeMode(column),
                        QHeaderView.ResizeMode.Interactive,
                    )
                self.assertGreaterEqual(header.sectionSize(0), 160)
                self.assertGreaterEqual(header.sectionSize(1), state_width)
                self.assertGreaterEqual(
                    header.sectionSize(2),
                    font_metrics.horizontalAdvance("100分"),
                )
                header.resizeSection(0, 48)
                header.resizeSection(1, 48)
                header.resizeSection(2, 90)
                self.assertGreaterEqual(header.sectionSize(0), 160)
                self.assertGreaterEqual(header.sectionSize(1), state_width)
                self.assertEqual(header.sectionSize(2), 90)

                child = page._variant_nodes[0]
                self.assertEqual(
                    child.text(0),
                    "字形12 · 重污染反相样本_001.tif",
                )
                self.assertEqual(
                    page._variant_nodes[1].text(0),
                    "字形13 · 普通样本_002.png",
                )
                self.assertGreaterEqual(child.sizeHint(0).height(), 46)
                self.assertLess(child.sizeHint(0).height(), 120)
                self.assertLess(page._variant_nodes[1].sizeHint(0).height(), 100)
                self.assertEqual(child.text(1).splitlines()[0], "待优化")
                self.assertEqual(child.text(1).splitlines()[1], "—")
                self.assertEqual(child.text(2), "100分")
                self.assertEqual(page._variant_nodes[1].text(2), "89分")
                text_width = font_metrics.horizontalAdvance(child.text(0))
                text_area = max(
                    1,
                    header.sectionSize(0) - page._item_tree.indentation() * 2 - 8,
                )
                required_lines = (text_width + text_area - 1) // text_area
                required_height = required_lines * font_metrics.height()
                self.assertGreaterEqual(page._item_tree.visualItemRect(child).height(), required_height)

                parent = child.parent()
                self.assertIsNotNone(parent)
                self.assertEqual(parent.text(1).splitlines(), ["已优化 1/2", "—"])
                self.assertFalse(
                    bool(parent.flags() & Qt.ItemFlag.ItemIsSelectable)
                )

                original_panel = page._original_preview.parentWidget()
                selected_panel = page._selected_preview.parentWidget()
                tools_left = page._preview_tools.mapTo(
                    page,
                    page._preview_tools.rect().topLeft(),
                ).x()
                tools_right = page._preview_tools.mapTo(
                    page,
                    page._preview_tools.rect().topRight(),
                ).x()
                original_right = original_panel.mapTo(
                    page,
                    original_panel.rect().topRight(),
                ).x()
                selected_left = selected_panel.mapTo(
                    page,
                    selected_panel.rect().topLeft(),
                ).x()
                self.assertLess(original_right, tools_left)
                self.assertLess(tools_right, selected_left)
                self.assertGreaterEqual(page._original_preview.width(), 130)
                self.assertGreaterEqual(page._selected_preview.width(), 130)
                self.assertGreaterEqual(
                    page._original_preview.height(),
                    page.PREVIEW_MIN_HEIGHT,
                )
                self.assertGreaterEqual(
                    page._selected_preview.height(),
                    page.PREVIEW_MIN_HEIGHT,
                )
                for control in (
                    page._fit_button,
                    page._actual_size_button,
                    page._hold_original_button,
                    page._white_background_button,
                    page._transparent_background_button,
                ):
                    self.assertIs(control.parentWidget(), page._preview_tools)

                list_top = page._list_panel.mapTo(
                    page,
                    page._list_panel.rect().topLeft(),
                ).y()
                preview_top = original_panel.mapTo(page, original_panel.rect().topLeft()).y()
                self.assertLessEqual(abs(preview_top - list_top), 1)
                self.assertFalse(hasattr(page, "_current_label"))
                self.assertIs(page._engine_combo.parentWidget(), page._scheme_fixed_content)

                def widget_bottom(widget) -> int:
                    return widget.mapTo(page, widget.rect().bottomLeft()).y()

                list_bottom = widget_bottom(page._list_panel)
                self.assertLessEqual(abs(widget_bottom(page._candidate_panel) - list_bottom), 1)
                self.assertLessEqual(abs(widget_bottom(page._scheme_panel) - list_bottom), 1)
                self.assertLessEqual(
                    abs(widget_bottom(page._message_label) - widget_bottom(page._candidate_panel)),
                    2,
                )
                self.assertIs(page._message_label.parentWidget(), page._candidate_panel)
                self.assertIs(page._previous_button.parentWidget(), page._list_panel)
                self.assertIs(page._next_button.parentWidget(), page._list_panel)
                self.assertFalse(hasattr(page, "_position_label"))
                self.assertIsInstance(page._scheme_label, QPlainTextEdit)
                self.assertFalse(hasattr(page, "_scheme_title"))
                self.assertFalse(hasattr(page, "_score_label"))
                self.assertFalse(hasattr(page, "_route_label"))
                self.assertIs(page._history_label.parentWidget(), page._scheme_fixed_content)
                self.assertFalse(page._scheme_panel.findChildren(QScrollArea))
                page.close()
                page.deleteLater()

    def test_grouped_list_children_show_corrected_real_image_thumbnails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inverted_path = root / "反相.png"
            optimized_path = root / "优化稿.png"
            inverted = Image.new("RGB", (80, 40), "black")
            for x in range(30, 50):
                for y in range(8, 32):
                    inverted.putpixel((x, y), (255, 255, 255))
            inverted.save(inverted_path)
            Image.new("RGBA", (40, 80), (0, 0, 0, 0)).save(optimized_path)
            with Image.open(optimized_path) as transparent:
                optimized = transparent.convert("RGBA")
            for x in range(8, 32):
                for y in range(30, 50):
                    optimized.putpixel((x, y), (0, 0, 0, 255))
            optimized.save(optimized_path)
            optimized.close()

            page = OptimizationPage()
            page._items = [
                {
                    "键": "variant-1",
                    "归属字": "永",
                    "变体序号": 1,
                    "原始文件名": "反相.png",
                    "原始路径": str(inverted_path),
                    "优化预览路径": "",
                    "显示状态": "待优化",
                    "得分": None,
                },
                {
                    "键": "variant-2",
                    "归属字": "永",
                    "变体序号": 2,
                    "原始文件名": "优化稿.png",
                    "原始路径": str(inverted_path),
                    "优化预览路径": str(optimized_path),
                    "显示状态": "已优化",
                    "得分": 90,
                },
                {
                    "键": "variant-3",
                    "归属字": "中",
                    "变体序号": 1,
                    "原始文件名": "反相.png",
                    "原始路径": str(inverted_path),
                    "优化预览路径": "",
                    "显示状态": "待优化",
                    "得分": None,
                },
            ]
            page._status_combo.setCurrentText(PHASE_FILTER_ALL)
            page._refresh_list()

            self.assertEqual(page._item_tree.topLevelItemCount(), 2)
            parents = [
                page._item_tree.topLevelItem(index)
                for index in range(page._item_tree.topLevelItemCount())
            ]
            parent = next(item for item in parents if item.text(0).startswith("永（"))
            self.assertEqual(parent.text(0), "永（2个字形）")
            self.assertEqual(parent.childCount(), 2)
            self.assertTrue(parent.isExpanded())
            self.assertFalse(bool(parent.flags() & Qt.ItemFlag.ItemIsSelectable))
            self.assertEqual(parent.child(0).text(0), "字形1 · 反相.png")
            self.assertEqual(parent.child(1).text(0), "字形2 · 优化稿.png")
            for index in range(2):
                child = parent.child(index)
                pixmap = child.icon(0).pixmap(page._item_tree.iconSize())
                self.assertFalse(pixmap.isNull())
                self.assertEqual(pixmap.toImage().pixelColor(0, 0), QColor("white"))
                self.assertGreaterEqual(child.sizeHint(0).height(), 46)
                self.assertLess(child.sizeHint(0).height(), 100)
            corrected = parent.child(0).icon(0).pixmap(page._item_tree.iconSize()).toImage()
            self.assertLess(corrected.pixelColor(19, 19).red(), 80)
            self.assertGreater(corrected.pixelColor(19, 10).red(), 180)
            optimized_icon = (
                parent.child(1).icon(0).pixmap(page._item_tree.iconSize()).toImage()
            )
            self.assertLess(optimized_icon.pixelColor(19, 19).red(), 80)
            self.assertGreater(optimized_icon.pixelColor(10, 19).red(), 180)

            single_parent = next(
                item for item in parents if item.text(0).startswith("中（")
            )
            self.assertEqual(single_parent.text(0), "中（1个字形）")
            self.assertEqual(single_parent.childCount(), 1)
            self.assertTrue(single_parent.isExpanded())
            self.assertFalse(
                bool(single_parent.flags() & Qt.ItemFlag.ItemIsSelectable)
            )
            self.assertEqual(page._list_count_label.text(), "显示 / 总数：3 / 3")
            self.assertEqual(
                page._summary_label.text(),
                "待优化 2　已优化 1",
            )

            page._search_edit.setText("优化稿.png")
            QTest.keyClick(page._search_edit, Qt.Key.Key_Return)
            self.assertEqual(page._item_tree.topLevelItemCount(), 1)
            filtered_parent = page._item_tree.topLevelItem(0)
            self.assertEqual(filtered_parent.text(0), "永（2个字形）")
            self.assertEqual(
                filtered_parent.text(1).splitlines(),
                ["已优化 1/2", "—"],
            )
            self.assertEqual(filtered_parent.childCount(), 1)
            self.assertEqual(
                filtered_parent.child(0).data(0, Qt.ItemDataRole.UserRole),
                "variant-2",
            )
            self.assertEqual(page._list_count_label.text(), "显示 / 总数：1 / 3")

            page._search_edit.setText("中")
            QTest.keyClick(page._search_edit, Qt.Key.Key_Return)
            self.assertEqual(page._item_tree.topLevelItemCount(), 1)
            self.assertEqual(
                page._item_tree.topLevelItem(0).child(0).data(
                    0,
                    Qt.ItemDataRole.UserRole,
                ),
                "variant-3",
            )
            self.assertEqual(page._current_item.get("键"), "variant-3")

            page._search_edit.clear()
            self.assertEqual(page._list_count_label.text(), "显示 / 总数：3 / 3")
            page._status_combo.setCurrentText("待优化")
            self.assertEqual(page._item_tree.topLevelItemCount(), 2)
            self.assertTrue(
                all(
                    page._item_tree.topLevelItem(index).childCount() == 1
                    for index in range(page._item_tree.topLevelItemCount())
                )
            )
            self.assertEqual(page._list_count_label.text(), "显示 / 总数：2 / 3")
            page.deleteLater()

    def test_save_in_pending_filter_selects_original_successor_without_skipping(self) -> None:
        page = OptimizationPage()
        original_items = [
            {
                "键": f"variant-{index}",
                "归属字": "永",
                "变体序号": index,
                "原始文件名": f"永_{index}.png",
                "原始路径": f"D:/测试/永_{index}.png",
                "优化预览路径": "",
                "显示状态": "待优化",
                "得分": None,
            }
            for index in range(1, 4)
        ]
        page._items = original_items
        page._status_combo.setCurrentText("待优化")
        page._refresh_list()
        self.assertEqual(
            page._item_tree.currentItem().data(0, Qt.ItemDataRole.UserRole),
            "variant-1",
        )

        saved_items = [dict(item) for item in original_items]
        saved_items[0]["显示状态"] = "自动优化稿"
        saved_items[0]["得分"] = 92.0
        service = MagicMock()
        service.list_items.return_value = saved_items
        page._service = service
        page._candidates = [{"方案名": "候选1", "得分": 92.0}]
        page._selected_index = 0
        page._load_candidates = lambda force=False: None  # type: ignore[method-assign]

        def run_immediately(function, success, _failure, lock_page: bool) -> None:
            self.assertTrue(lock_page)
            success(function())

        page._start_task = run_immediately  # type: ignore[method-assign]
        page._save_selected()

        service.save_selection.assert_called_once()
        self.assertEqual(
            page._item_tree.currentItem().data(0, Qt.ItemDataRole.UserRole),
            "variant-2",
        )
        self.assertEqual(page._current_item["键"], "variant-2")
        self.assertEqual(
            [item["键"] for item in page._visible_items],
            ["variant-2", "variant-3"],
        )
        page.deleteLater()

    def test_saving_last_pending_item_clears_previous_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "末-0001.png"
            Image.new("RGB", (64, 64), "white").save(source_path)
            item = {
                "键": "variant-last",
                "归属字": "末",
                "变体序号": 1,
                "原始文件名": source_path.name,
                "原始路径": str(source_path),
                "优化预览路径": "",
                "显示状态": "待优化",
                "得分": None,
            }
            page = OptimizationPage()
            page._items = [item]
            page._status_combo.setCurrentText("待优化")
            page._refresh_list()
            self.assertFalse(page._original_preview.pixmap().isNull())

            saved_item = dict(item, **{"显示状态": "自动优化稿", "得分": 91.0})
            service = MagicMock()
            service.list_items.return_value = [saved_item]
            page._service = service
            page._candidates = [{"方案名": "最后候选", "得分": 91.0}]
            page._selected_index = 0

            def run_immediately(function, success, _failure, lock_page: bool) -> None:
                self.assertTrue(lock_page)
                success(function())

            page._start_task = run_immediately  # type: ignore[method-assign]
            with patch.object(page, "_show_optimization_end_notice") as end_notice:
                page._save_selected()

            service.save_selection.assert_called_once()
            end_notice.assert_called_once_with()
            self.assertIsNone(page._current_item)
            self.assertEqual(page._visible_items, [])
            self.assertTrue(page._original_preview.pixmap().isNull())
            self.assertEqual(page._original_preview.text(), "暂无图片")
            self.assertIsNone(page._original_image)
            self.assertEqual(page._original_image_path, "")
            self.assertEqual(page._selected_preview.text(), "暂无图片")
            self.assertEqual(page._candidate_title.text(), "候选效果")
            self.assertEqual(page._history_label.text(), "暂无处理记录")
            self.assertIn("当前筛选范围内没有字形", page._message_label.text())
            page.deleteLater()

    def test_empty_pending_filter_keeps_no_image_after_delayed_refresh(self) -> None:
        page = OptimizationPage()
        page._items = [
            {
                "键": "variant-optimized",
                "归属字": "已",
                "变体序号": 1,
                "原始文件名": "已-0001.png",
                "原始路径": "",
                "优化预览路径": "",
                "显示状态": "自动优化稿",
                "得分": 95.0,
            }
        ]
        page._status_combo.setCurrentText("待优化")
        page._refresh_list()
        self.assertIsNone(page._current_item)
        self.assertEqual(page._original_preview.text(), "暂无图片")

        page.resize(1100, 720)
        page.show()
        self.app.processEvents()
        self.app.processEvents()
        page._refresh_previews_after_layout()

        self.assertIsNone(page._current_item)
        self.assertTrue(page._original_preview.pixmap().isNull())
        self.assertEqual(page._original_preview.text(), "暂无图片")
        page.close()
        page.deleteLater()

    def test_parent_group_never_replaces_current_variant(self) -> None:
        page = OptimizationPage()
        page._items = [
            {
                "键": "variant-1",
                "归属字": "永",
                "变体序号": 1,
                "原始文件名": "永_1.png",
                "原始路径": "",
                "显示状态": "待优化",
                "得分": None,
            },
            {
                "键": "variant-2",
                "归属字": "永",
                "变体序号": 2,
                "原始文件名": "永_2.png",
                "原始路径": "",
                "显示状态": "待优化",
                "得分": None,
            },
        ]
        page._refresh_list()
        valid_child = page._item_tree.currentItem()
        parent = valid_child.parent()

        page._item_tree.setCurrentItem(parent)
        self.assertIs(page._item_tree.currentItem(), valid_child)
        self.assertEqual(
            page._item_tree.currentItem().data(0, Qt.ItemDataRole.UserRole),
            "variant-1",
        )

        parent.setExpanded(False)
        self.assertFalse(parent.isExpanded())
        self.assertIs(page._item_tree.currentItem(), valid_child)
        parent.setExpanded(True)
        self.assertTrue(parent.isExpanded())
        self.assertIs(page._item_tree.currentItem(), valid_child)
        page.deleteLater()

    def test_thumbnail_cache_avoids_redecode_on_list_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            page = OptimizationPage()
            page._items = [
                {
                    "键": f"variant-{index}",
                    "归属字": "永",
                    "变体序号": index,
                    "原始文件名": f"永_{index}.png",
                    "原始路径": str(Path(directory) / f"永_{index}.png"),
                    "优化预览路径": "",
                    "显示状态": "待优化",
                    "得分": None,
                }
                for index in range(1, 4)
            ]
            thumbnail = QImage(38, 38, QImage.Format.Format_RGB888)
            thumbnail.fill(QColor("white"))
            decoder = MagicMock(return_value=thumbnail)

            with patch.object(page, "_decode_glyph_thumbnail", decoder):
                page._refresh_list()
                page._refresh_list()

            self.assertEqual(decoder.call_count, 3)
            page.deleteLater()

    def test_large_list_defers_only_visible_thumbnails_to_background(self) -> None:
        page = OptimizationPage()
        page._items = [
            {
                "键": f"variant-{index}",
                "归属字": "永",
                "变体序号": index,
                "原始文件名": f"永_{index}.png",
                "原始路径": f"Z:/不存在/永_{index}.png",
                "优化预览路径": "",
                "显示状态": "待优化",
                "得分": None,
            }
            for index in range(1, 81)
        ]
        decoder = MagicMock()
        with (
            patch.object(page, "_decode_glyph_thumbnail", decoder),
            patch.object(page, "_schedule_visible_list_thumbnails"),
        ):
            page._refresh_list()
            page.resize(1100, 720)
            page.show()
            self.app.processEvents()

        self.assertEqual(decoder.call_count, 0)
        self.assertTrue(
            all(not node.icon(0).isNull() for node in page._variant_nodes)
        )
        with patch.object(page, "_start_list_thumbnail_batch") as start_batch:
            page._load_visible_list_thumbnails()

        start_batch.assert_called_once()
        jobs = start_batch.call_args.args[0]
        self.assertGreater(len(jobs), 0)
        self.assertLessEqual(len(jobs), page.LIST_THUMBNAIL_BATCH_SIZE)
        self.assertLess(len(jobs), len(page._visible_items))
        visible_keys = {
            str(node.data(0, Qt.ItemDataRole.UserRole))
            for node in page._variant_nodes
            if page._item_tree.visualItemRect(node).intersects(
                page._item_tree.viewport().rect()
            )
        }
        self.assertTrue(
            all(
                str(item["键"]) in visible_keys
                for _variant_id, _signature, item in jobs
            )
        )
        page.close()
        page.deleteLater()

    def test_large_list_background_thumbnail_keeps_polarity_correction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "反相.png"
            inverted = Image.new("RGB", (80, 40), "black")
            for x in range(30, 50):
                for y in range(8, 32):
                    inverted.putpixel((x, y), (255, 255, 255))
            inverted.save(source_path)
            inverted.close()

            page = OptimizationPage()
            page._items = [
                {
                    "键": f"variant-{index}",
                    "归属字": "永",
                    "变体序号": index,
                    "原始文件名": f"反相_{index}.png",
                    "原始路径": str(source_path),
                    "优化预览路径": "",
                    "显示状态": "待优化",
                    "得分": None,
                }
                for index in range(1, page.LIST_THUMBNAIL_SYNC_LIMIT + 2)
            ]
            page.resize(1100, 720)
            page.show()
            page._refresh_list()
            self.app.processEvents()
            self.assertTrue(page._thread_pool.waitForDone(2000))
            self.app.processEvents()
            self.app.processEvents()

            variant_id = str(page._items[0]["键"])
            self.assertIn(variant_id, page._list_thumbnail_cache)
            self.assertEqual(
                page._list_thumbnail_cache[variant_id][0],
                page._thumbnail_cache_key(page._items[0]),
            )
            corrected = page._variant_nodes[0].icon(0).pixmap(
                page._item_tree.iconSize()
            ).toImage()
            self.assertLess(corrected.pixelColor(19, 19).red(), 80)
            self.assertGreater(corrected.pixelColor(19, 10).red(), 180)
            page.close()
            page.deleteLater()

    def test_thumbnail_cache_is_bounded_and_same_path_overwrite_invalidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "原图.png"
            Image.new("RGB", (20, 20), "white").save(source_path)
            item = {
                "键": "variant-current",
                "原始路径": str(source_path),
                "优化预览路径": "",
            }
            page = OptimizationPage()
            page.LIST_THUMBNAIL_CACHE_ITEMS = 2
            page._glyph_thumbnail(item)
            old_signature = page._list_thumbnail_cache["variant-current"][0]

            Image.new("RGB", (24, 24), "black").save(source_path)
            timestamp = time.time_ns() + 1_000_000_000
            os.utime(source_path, ns=(timestamp, timestamp))
            replacement = QImage(38, 38, QImage.Format.Format_RGB888)
            replacement.fill(QColor("white"))
            with patch.object(
                page,
                "_decode_glyph_thumbnail",
                return_value=replacement,
            ) as decoder:
                page._glyph_thumbnail(item)

            decoder.assert_called_once()
            self.assertNotEqual(
                page._list_thumbnail_cache["variant-current"][0],
                old_signature,
            )
            page._store_glyph_thumbnail(
                "variant-second",
                ("second", 1, 1, False),
                QIcon(),
            )
            page._store_glyph_thumbnail(
                "variant-third",
                ("third", 1, 1, False),
                QIcon(),
            )
            self.assertEqual(
                tuple(page._list_thumbnail_cache),
                ("variant-second", "variant-third"),
            )
            page.deleteLater()

    def test_candidate_cache_hit_promotes_lru_entry(self) -> None:
        page = OptimizationPage()
        page.CANDIDATE_CACHE_MAX_ITEMS = 2
        page.CANDIDATE_CACHE_MAX_BYTES = 1024 * 1024

        def candidates(name: str) -> list[dict[str, object]]:
            return [{
                "方案名": name,
                "图像": Image.new("RGBA", (4, 4), (0, 0, 0, 255)),
                "灰度母版": np.zeros((4, 4), dtype=np.uint8),
                "清洁掩码": np.ones((4, 4), dtype=np.uint8),
            }]

        first = candidates("第一项")
        second = candidates("第二项")
        third = candidates("第三项")
        page._store_candidate_cache("first", first)
        page._store_candidate_cache("second", second)

        self.assertIs(page._cached_candidates("first"), first)
        page._store_candidate_cache("third", third)

        self.assertEqual(tuple(page._candidate_cache), ("first", "third"))
        self.assertNotIn("second", page._candidate_cache)
        for values in (first, second, third):
            values[0]["图像"].close()
        page.deleteLater()

    def test_candidate_cache_uses_actual_bytes_and_clear_resets_accounting(self) -> None:
        page = OptimizationPage()
        page.CANDIDATE_CACHE_MAX_ITEMS = 10
        shared_layer = np.zeros((10, 10), dtype=np.uint8)
        first_image = Image.new("RGBA", (10, 10), (0, 0, 0, 255))
        first = [{
            "图像": first_image,
            "灰度母版": shared_layer,
            "清洁掩码": shared_layer,
        }]
        first_cost = page._estimate_candidate_cache_bytes(first)
        self.assertEqual(
            first_cost,
            page.CANDIDATE_CACHE_ENTRY_OVERHEAD + 10 * 10 * 4 + shared_layer.nbytes,
        )

        second_image = Image.new("RGBA", (10, 10), (0, 0, 0, 255))
        second = [{
            "图像": second_image,
            "灰度母版": np.zeros((10, 10), dtype=np.uint8),
            "清洁掩码": np.ones((10, 10), dtype=np.uint8),
        }]
        second_cost = page._estimate_candidate_cache_bytes(second)
        page.CANDIDATE_CACHE_MAX_BYTES = first_cost + second_cost - 1
        page._store_candidate_cache("first", first)
        page._store_candidate_cache("second", second)

        self.assertEqual(tuple(page._candidate_cache), ("second",))
        self.assertEqual(page._candidate_cache_bytes, second_cost)
        page._clear_candidate_cache()
        self.assertFalse(page._candidate_cache)
        self.assertFalse(page._candidate_cache_costs)
        self.assertEqual(page._candidate_cache_bytes, 0)
        first_image.close()
        second_image.close()
        page.deleteLater()

    def test_oversized_candidate_is_displayed_without_being_cached(self) -> None:
        page = OptimizationPage()
        page.CANDIDATE_CACHE_MAX_BYTES = 1
        image = Image.new("RGBA", (12, 12), (0, 0, 0, 255))
        candidates = [{
            "方案名": "超预算候选",
            "方案": {},
            "得分": 88.0,
            "处理类型": CANDIDATE_TYPE_OPTIMIZED,
            "图像": image,
            "灰度母版": np.zeros((12, 12), dtype=np.uint8),
            "清洁掩码": np.ones((12, 12), dtype=np.uint8),
        }]
        page._current_item = {"键": "variant-large"}

        with patch.object(page, "_candidate_cache_key", return_value="large-key"):
            page._candidates_ready("large-key", "variant-large", candidates)

        self.assertNotIn("large-key", page._candidate_cache)
        self.assertEqual(page._candidate_cache_bytes, 0)
        self.assertIs(page._candidates[0], candidates[0])
        self.assertEqual(image.getpixel((0, 0)), (0, 0, 0, 255))
        page._candidates = []
        image.close()
        page.deleteLater()

    def test_dynamic_header_still_fits_main_window_minimum_width(self) -> None:
        page = OptimizationPage()
        page._library_label.setText("当前字库：测试字库　300 DPI · 250×250 像素")
        page._summary_label.setText("待优化 128　已优化 256　完成度 67%")
        page.show()
        self.app.processEvents()

        self.assertLessEqual(page.minimumSizeHint().width(), 1100)
        self.assertFalse(hasattr(page, "_progress_label"))
        page.close()
        page.deleteLater()

    def test_no_model_candidate_failure_keeps_pending_variant_and_source(self) -> None:
        page = OptimizationPage()
        service = MagicMock()
        pending = {
            "键": "variant-structure-protected",
            "归属字": "剛",
            "原始文件名": "剛-0001.tif",
            "原始路径": "D:/测试/剛-0001.tif",
            "显示状态": "待优化",
        }
        page._service = service
        page._items = [pending]
        page._visible_items = [pending]
        page._current_item = pending

        def run_failure(_function, _success, failure, lock_page: bool) -> None:
            self.assertFalse(lock_page)
            failure("算法未生成通过结构保护的寻优候选结果。")

        page._start_task = run_failure  # type: ignore[method-assign]
        with (
            patch.object(page, "_candidate_cache_key", return_value="pending-cache"),
            patch.object(QMessageBox, "critical") as critical,
        ):
            page._load_candidates(force=True)

        self.assertIs(page._current_item, pending)
        self.assertEqual(page._items, [pending])
        self.assertEqual(page._visible_items, [pending])
        service.list_items.assert_not_called()
        message = critical.call_args.args[2]
        self.assertIn("结构保护", message)
        self.assertIn("字形记录和原图均已保留", message)
        self.assertIn("不会改变处理状态", message)
        self.assertEqual(page._message_label.text(), "算法未生成通过结构保护的寻优候选结果。")
        page.deleteLater()

    def test_missing_source_is_readonly_without_starting_candidate_task(self) -> None:
        page = OptimizationPage()
        service = MagicMock()
        pending = {
            "键": "variant-missing-source",
            "归属字": "缺",
            "原始文件名": "缺-0001.tif",
            "原始路径": "",
            "显示状态": "待优化",
            "提示": ("文件异常",),
        }
        page._service = service
        page._items = [pending]
        page._visible_items = [pending]
        page._current_item = pending

        with (
            patch.object(page, "_start_task") as start_task,
            patch.object(QMessageBox, "critical") as critical,
        ):
            page._load_candidates(force=True)

        self.assertIs(page._current_item, pending)
        self.assertEqual(page._items, [pending])
        self.assertEqual(page._visible_items, [pending])
        service.list_items.assert_not_called()
        start_task.assert_not_called()
        critical.assert_not_called()
        self.assertIn("原图文件不可用", page._message_label.text())
        self.assertFalse(page._candidate_list.isEnabled())
        page.deleteLater()

    def test_later_stage_item_is_visible_but_readonly(self) -> None:
        page = OptimizationPage()
        service = MagicMock()
        reviewed = {
            "键": "variant-reviewed",
            "归属字": "审",
            "原始文件名": "审-0001.png",
            "原始路径": "D:/测试/审-0001.png",
            "显示状态": "待协调",
        }
        page._service = service

        with patch.object(page, "_start_task") as start_task:
            page._select_item(reviewed)

        start_task.assert_not_called()
        self.assertIs(page._current_item, reviewed)
        self.assertIn("已经完成自动优化", page._message_label.text())
        self.assertIn("仅供查看", page._message_label.text())
        self.assertFalse(page._save_button.isEnabled())
        page.deleteLater()

    def test_optimized_item_can_explicitly_start_reoptimization(self) -> None:
        page = OptimizationPage()
        optimized = {
            "键": "variant-optimized",
            "归属字": "重",
            "原始文件名": "重-0001.png",
            "原始路径": "D:/测试/重-0001.png",
            "显示状态": "已优化",
            "优化预览路径": "",
        }
        page._service = MagicMock()
        page._current_item = optimized
        page._items = [optimized]
        page._show_readonly_current_item()

        self.assertEqual(page._status_combo.currentText(), PHASE_FILTER_ALL)
        self.assertEqual(page._restart_button.text(), "重新优化此字形")
        self.assertTrue(page._restart_button.isEnabled())
        self.assertEqual(page._restart_button.property("role"), "primary")
        self.assertLess(
            page._scheme_fixed_content.layout().indexOf(page._restart_button),
            page._scheme_fixed_content.layout().indexOf(page._scheme_label),
        )

        with (
            patch.object(page, "_confirm_reoptimization", return_value=True),
            patch.object(page, "_candidate_cache_key", return_value="cache-key"),
            patch.object(page, "_remove_candidate_cache") as remove_cache,
            patch.object(page, "_load_candidates") as load_candidates,
        ):
            page._restart_candidates()

        self.assertEqual(page._reoptimization_key, "variant-optimized")
        self.assertTrue(page._current_item_is_optimizable())
        remove_cache.assert_called_once_with("cache-key")
        load_candidates.assert_called_once_with(force=True)
        page.deleteLater()

    def test_reoptimization_warning_mentions_downstream_invalidation(self) -> None:
        page = OptimizationPage()
        page._current_item = {
            "键": "variant-finished",
            "显示状态": "已优化",
            "原始路径": "D:/测试/成-0001.png",
            "状态": "成品已生成",
            "审核文件": "成-0001.png",
            "成品文件": "成-0001.png",
        }
        confirm_button = object()
        cancel_button = object()
        box = MagicMock()
        box.addButton.side_effect = [confirm_button, cancel_button]
        box.clickedButton.return_value = confirm_button

        with patch("ui.pages.optimization_page.QMessageBox", return_value=box):
            self.assertTrue(page._confirm_reoptimization())

        informative_text = box.setInformativeText.call_args.args[0]
        self.assertIn("现有人工审核稿和成品将被撤销", informative_text)
        self.assertIn("重新完成后续阶段", informative_text)
        page.deleteLater()

    def test_empty_candidate_result_does_not_remove_variant_record(self) -> None:
        page = OptimizationPage()
        pending = {"键": "variant-1", "显示状态": "待优化"}
        page._service = MagicMock()
        page._items = [pending]
        page._current_item = pending

        with patch("ui.pages.optimization_page.QMessageBox.critical") as critical:
            page._candidates_ready(object(), "variant-1", [])

        self.assertIs(page._current_item, pending)
        self.assertEqual(page._items, [pending])
        self.assertIn("字形记录和原图均已保留", critical.call_args.args[2])
        self.assertEqual(page._message_label.text(), "算法未生成有效候选结果。")
        page.deleteLater()

    def test_exploration_failure_does_not_remove_variant_record(self) -> None:
        page = OptimizationPage()
        page._service = MagicMock()
        page._current_item = {
            "键": "variant-1",
            "显示状态": "待优化",
            "原始路径": "D:/测试/variant-1.png",
        }
        page._candidates = [{"处理类型": CANDIDATE_TYPE_OPTIMIZED, "方案": {}}]
        page._selected_index = 0

        def run_failure(_function, _success, failure, lock_page: bool) -> None:
            self.assertTrue(lock_page)
            failure("模拟探索失败")

        page._start_task = run_failure  # type: ignore[method-assign]
        with patch("ui.pages.optimization_page.QMessageBox.critical") as critical:
            page._explore_selected()

        self.assertIn("字形记录和原图均已保留", critical.call_args.args[2])
        self.assertEqual(page._message_label.text(), "模拟探索失败")
        page.deleteLater()

    def test_locked_task_disables_context_switching_controls(self) -> None:
        page = OptimizationPage()
        page._busy = True

        page._set_workspace_enabled(True)

        for control in (
            page._engine_combo,
            page._home_button,
            page._search_edit,
            page._search_button,
            page._status_combo,
            page._sort_combo,
            page._item_tree,
        ):
            with self.subTest(control=type(control).__name__):
                self.assertFalse(control.isEnabled())
        page.deleteLater()

    def test_complete_button_precedes_home_and_batch_progress_is_temporary(self) -> None:
        page = OptimizationPage()
        page.resize(1200, 760)
        page.show()
        self.app.processEvents()

        self.assertEqual(page._complete_button.text(), "批量自动优化")
        self.assertLess(page._complete_button.x(), page._home_button.x())
        self.assertFalse(page._bulk_progress.isVisible())
        self.assertFalse(page._stop_bulk_button.isVisible())
        self.assertFalse(page._complete_button.isEnabled())
        page.close()
        page.deleteLater()

    def test_bulk_risk_confirmation_uses_chinese_confirm_and_defaults_to_cancel(self) -> None:
        page = OptimizationPage()
        observed: dict[str, object] = {}

        def cancel_dialog(dialog: QMessageBox) -> int:
            observed["标题"] = dialog.windowTitle()
            observed["正文"] = dialog.text()
            observed["说明"] = dialog.informativeText()
            confirm_button = dialog.button(QMessageBox.StandardButton.Ok)
            cancel_button = dialog.button(QMessageBox.StandardButton.Cancel)
            self.assertIsNotNone(confirm_button)
            self.assertIsNotNone(cancel_button)
            self.assertEqual(
                dialog.standardButtons(),
                QMessageBox.StandardButton.Ok
                | QMessageBox.StandardButton.Cancel,
            )
            self.assertEqual(confirm_button.text(), "确定")
            self.assertEqual(cancel_button.text(), "取消")
            self.assertIs(dialog.defaultButton(), cancel_button)
            self.assertIs(dialog.escapeButton(), cancel_button)
            return QMessageBox.StandardButton.Cancel.value

        with patch.object(QMessageBox, "exec", new=cancel_dialog):
            self.assertFalse(page._confirm_bulk_optimization(12, 3))

        self.assertEqual(observed["标题"], "确认批量自动优化")
        self.assertIn("12 个待优化字形", str(observed["正文"]))
        self.assertIn("不会逐字等待人工确认", str(observed["说明"]))
        self.assertIn("浅色或断裂笔画", str(observed["说明"]))
        self.assertIn("大块污点", str(observed["说明"]))
        self.assertIn("笔画缺失", str(observed["说明"]))
        self.assertIn("污点残留", str(observed["说明"]))
        self.assertIn("边缘发生变化", str(observed["说明"]))
        self.assertIn("自动保存为自动优化稿", str(observed["说明"]))
        self.assertIn("进入待审核阶段", str(observed["说明"]))
        self.assertIn("结构需人工核对", str(observed["说明"]))
        self.assertIn("不会把它计为失败", str(observed["说明"]))
        self.assertIn("原始图片不会被修改", str(observed["说明"]))
        self.assertIn("无学习模型", str(observed["说明"]))
        self.assertIn("已有结果 3 个将保留并跳过", str(observed["说明"]))

        with patch.object(
            QMessageBox,
            "exec",
            new=lambda _dialog: QMessageBox.StandardButton.Ok.value,
        ):
            self.assertTrue(page._confirm_bulk_optimization(12, 3))
        page.deleteLater()

    def test_cancelling_bulk_risk_confirmation_does_not_start_worker(self) -> None:
        pending = {
            "键": "variant-1",
            "归属字": "何",
            "原始文件名": "何-0001.tif",
            "显示状态": "待优化",
        }
        service = MagicMock()
        service.list_items.return_value = [pending]
        service.list_batch_items.return_value = ([pending], 0)
        page = OptimizationPage()
        page._service = service
        page._candidates = [{"方案": {}}]
        page._set_workspace_enabled(True)
        control_states = {
            "首页": page._home_button.isEnabled(),
            "列表": page._item_tree.isEnabled(),
            "候选": page._candidate_list.isEnabled(),
        }

        with (
            patch.object(page, "_confirm_bulk_optimization", return_value=False),
            patch.object(page._thread_pool, "start") as start,
        ):
            page._confirm_and_start_bulk()

        start.assert_not_called()
        self.assertIsNone(page._bulk_worker)
        self.assertFalse(page._busy)
        self.assertFalse(page._bulk_progress.isVisible())
        self.assertEqual(page._home_button.isEnabled(), control_states["首页"])
        self.assertEqual(page._item_tree.isEnabled(), control_states["列表"])
        self.assertEqual(page._candidate_list.isEnabled(), control_states["候选"])
        page.deleteLater()

    def test_no_pending_bulk_items_skips_risk_confirmation_and_worker(self) -> None:
        service = MagicMock()
        service.list_items.return_value = []
        service.list_batch_items.return_value = ([], 7)
        page = OptimizationPage()
        page._service = service

        with (
            patch.object(page, "_refresh_list") as refresh,
            patch.object(page, "_confirm_bulk_optimization") as confirm,
            patch.object(page._thread_pool, "start") as start,
            patch.object(QMessageBox, "information") as information,
        ):
            page._confirm_and_start_bulk()

        refresh.assert_called_once_with()
        confirm.assert_not_called()
        start.assert_not_called()
        information.assert_called_once_with(
            page,
            "完成自动优化",
            "当前字库没有待优化字形，自动优化阶段已经完成。",
        )
        self.assertIsNone(page._bulk_worker)
        self.assertFalse(page._busy)
        page.deleteLater()

    def test_bulk_optimization_uses_batch_candidate_and_continues_failure(self) -> None:
        pending_items = [
            {
                "键": "variant-1",
                "归属字": "何",
                "变体序号": 1,
                "原始文件名": "何-0001.tif",
                "显示状态": "待优化",
            },
            {
                "键": "variant-2",
                "归属字": "是",
                "变体序号": 1,
                "原始文件名": "是-0001.tif",
                "显示状态": "待优化",
            },
        ]
        completed = {
            "键": "variant-3",
            "归属字": "中",
            "变体序号": 1,
            "原始文件名": "中-0001.tif",
            "显示状态": "已优化",
        }
        def valid_candidate(name: str) -> dict[str, object]:
            return {
                "方案名": name,
                "方案": {},
                "得分": 90.0,
                "图像": Image.new("RGBA", (2, 2), (0, 0, 0, 255)),
                "灰度母版": np.zeros((2, 2), dtype=np.uint8),
                "清洁掩码": np.ones((2, 2), dtype=np.uint8),
            }

        first_valid = valid_candidate("第一项有效候选")
        second_valid = valid_candidate("第二项有效候选")
        optimized_first = dict(
            pending_items[0],
            **{"显示状态": "已优化", "得分": 90.0},
        )
        service = MagicMock()
        service.list_items.side_effect = [
            pending_items + [completed],
            [optimized_first, pending_items[1], completed],
        ]
        service.list_batch_items.return_value = (pending_items, 1)
        service.generate_batch_candidate.side_effect = [first_valid, second_valid]
        service.is_candidate_valid.return_value = True
        service.save_selection.side_effect = [None, OSError("模拟单字保存失败")]

        page = OptimizationPage()
        page._service = service
        page._current_item = pending_items[0]
        page._candidates = [first_valid]
        page._selected_index = 0
        page.resize(1200, 760)
        page.show()
        self.app.processEvents()
        started: list[object] = []
        with (
            patch.object(page, "_confirm_bulk_optimization", return_value=True),
            patch.object(page._thread_pool, "start", side_effect=started.append),
        ):
            page._confirm_and_start_bulk()

        self.assertEqual(len(started), 1)
        worker = started[0]
        self.assertIs(worker, page._bulk_worker)
        self.assertTrue(page._bulk_progress.isVisible())
        self.assertTrue(page._stop_bulk_button.isVisible())
        self.assertTrue(page._stop_bulk_button.isEnabled())
        self.assertFalse(page._complete_button.isEnabled())
        self.assertFalse(page._home_button.isEnabled())

        progress: list[tuple[str, int, int]] = []
        worker.signals.progress.connect(
            lambda message, current, total: progress.append((message, current, total))
        )
        with (
            patch.object(QMessageBox, "warning") as warning,
            patch.object(page._thread_pool, "start") as candidate_start,
        ):
            worker.run()
            self.app.processEvents()

        self.assertEqual(service.generate_batch_candidate.call_count, 2)
        self.assertEqual(service.save_selection.call_count, 2)
        self.assertIs(service.save_selection.call_args_list[0].args[1], first_valid)
        self.assertIs(service.save_selection.call_args_list[1].args[1], second_valid)
        self.assertIs(
            service.generate_batch_candidate.call_args_list[0].kwargs["engine_context"],
            page._engine_context,
        )
        self.assertTrue(
            callable(
                service.generate_batch_candidate.call_args_list[0].kwargs[
                    "cancel_check"
                ]
            )
        )
        self.assertGreaterEqual(len(progress), 2)
        self.assertEqual(progress[0][1:], (0, 2))
        self.assertEqual(progress[-1][1:], (2, 2))
        self.assertIn("正在自动优化 1/2：何-0001", progress[0][0])
        self.assertIn("自动优化处理完成 2/2", progress[-1][0])
        self.assertIsNone(page._bulk_worker)
        self.assertFalse(page._bulk_progress.isVisible())
        self.assertFalse(page._stop_bulk_button.isVisible())
        self.assertTrue(page._home_button.isEnabled())
        self.assertTrue(page._complete_button.isEnabled())
        self.assertTrue(page._item_tree.isEnabled())
        self.assertFalse(page._candidate_list.isEnabled())
        candidate_start.assert_not_called()
        warning.assert_called_once()
        summary = warning.call_args.args[2]
        self.assertIn("成功 1", summary)
        self.assertIn("跳过 1", summary)
        self.assertIn("失败 1", summary)
        self.assertIn("是-0001：模拟单字保存失败", summary)
        self.assertIn("总耗时：", summary)
        page.close()
        page.deleteLater()

    def test_bulk_optimization_success_restores_home_and_navigation(self) -> None:
        pending = {
            "键": "variant-1",
            "归属字": "何",
            "变体序号": 1,
            "原始文件名": "何-0001.tif",
            "原始路径": "",
            "显示状态": "待优化",
        }
        optimized = dict(
            pending,
            **{"显示状态": "已优化", "得分": 92.0},
        )
        candidate = {
            "方案名": "第一项有效候选",
            "方案": {},
            "得分": 92.0,
            "图像": Image.new("RGBA", (2, 2), (0, 0, 0, 255)),
            "灰度母版": np.zeros((2, 2), dtype=np.uint8),
            "清洁掩码": np.ones((2, 2), dtype=np.uint8),
        }
        service = MagicMock()
        service.list_items.side_effect = [[pending], [optimized]]
        service.list_batch_items.return_value = ([pending], 0)
        service.generate_batch_candidate.return_value = candidate
        service.is_candidate_valid.return_value = True

        page = OptimizationPage()
        page._service = service
        page._current_item = pending
        page._candidates = [candidate]
        page._selected_index = 0
        started: list[object] = []
        with (
            patch.object(page, "_confirm_bulk_optimization", return_value=True),
            patch.object(page._thread_pool, "start", side_effect=started.append),
        ):
            page._confirm_and_start_bulk()

        worker = started[0]
        self.assertFalse(page._home_button.isEnabled())
        with patch.object(QMessageBox, "information") as information:
            worker.run()
            self.app.processEvents()

        self.assertIsNone(page._bulk_worker)
        self.assertFalse(page._busy)
        self.assertFalse(page._bulk_progress.isVisible())
        self.assertFalse(page._stop_bulk_button.isVisible())
        self.assertTrue(page._home_button.isEnabled())
        self.assertTrue(page._complete_button.isEnabled())
        self.assertTrue(page._engine_combo.isEnabled())
        self.assertTrue(page._search_edit.isEnabled())
        self.assertTrue(page._item_tree.isEnabled())
        self.assertFalse(page._candidate_list.isEnabled())
        information.assert_called_once()
        self.assertIn("总耗时：", information.call_args.args[2])
        page.deleteLater()

    def test_batch_worker_counts_structure_risk_as_saved_not_failed(self) -> None:
        item = {
            "键": "variant-risk",
            "归属字": "险",
            "变体序号": 1,
            "原始文件名": "险-0001.png",
        }
        candidate = {
            "方案名": "风险寻优",
            "方案": {},
            "结构复核": {
                "状态": "需人工核对",
                "阶段": "原尺寸复核",
                "原因": "参考端点仅匹配42.9%",
                "风险等级": 1,
            },
        }
        service = MagicMock()
        service.generate_batch_candidate.return_value = candidate
        service.is_candidate_valid.return_value = True
        page = OptimizationPage()
        worker = _OptimizationBatchWorker(service, [item], page._engine_context, 0)

        with patch("ui.pages.optimization_page.write_log") as write_log:
            result = worker._run_batch()

        self.assertEqual(result["成功"], 1)
        self.assertEqual(result["需人工核对"], 1)
        self.assertEqual(result["失败"], 0)
        self.assertEqual(
            result["需人工核对详情"][0]["原因"],
            "参考端点仅匹配42.9%",
        )
        service.save_selection.assert_called_once()
        self.assertEqual(service.save_selection.call_args.args[:2], (item, candidate))
        self.assertEqual(service.save_selection.call_args.kwargs["round_number"], 1)
        self.assertIs(
            service.save_selection.call_args.kwargs["persistence"],
            service.create_batch_persistence.return_value,
        )
        self.assertIn("批处理耗时汇总", write_log.call_args.args[0])
        self.assertIn("任务=自动优化", write_log.call_args.args[0])
        page.deleteLater()

    def test_batch_checkpoint_failure_keeps_journal_for_recovery(self) -> None:
        item = {
            "键": "variant-checkpoint",
            "归属字": "存",
            "变体序号": 1,
            "原始文件名": "存-0001.png",
        }
        candidate = {
            "方案名": "批量候选",
            "方案": {},
        }
        service = MagicMock()
        service.generate_batch_candidate.return_value = candidate
        persistence = service.create_batch_persistence.return_value
        persistence.checkpoint_if_due.side_effect = OSError("模拟检查点失败")
        page = OptimizationPage()
        worker = _OptimizationBatchWorker(service, [item], page._engine_context, 0)

        with (
            patch("ui.pages.optimization_page.write_log") as write_log,
            self.assertRaisesRegex(OSError, "模拟检查点失败"),
        ):
            worker._run_batch()

        persistence.leave_for_recovery.assert_called_once_with()
        persistence.finish.assert_not_called()
        self.assertIn("批处理耗时汇总", write_log.call_args.args[0])
        self.assertIn("失败=1", write_log.call_args.args[0])
        page.deleteLater()

    def test_uncertain_journal_commit_stops_batch_without_checkpoint(self) -> None:
        items = [
            {
                "键": f"variant-{index}",
                "归属字": "存",
                "变体序号": index,
                "原始文件名": f"存-{index:04d}.png",
            }
            for index in (1, 2)
        ]
        service = MagicMock()
        service.generate_batch_candidate.return_value = {"方案名": "批量候选", "方案": {}}
        service.save_selection.side_effect = BatchJournalUncertainError(
            "模拟日志提交结果未知"
        )
        persistence = service.create_batch_persistence.return_value
        page = OptimizationPage()
        worker = _OptimizationBatchWorker(service, items, page._engine_context, 0)

        with self.assertRaisesRegex(BatchJournalUncertainError, "提交结果未知"):
            worker._run_batch()

        service.generate_batch_candidate.assert_called_once()
        persistence.checkpoint_if_due.assert_not_called()
        persistence.finish.assert_not_called()
        persistence.leave_for_recovery.assert_called_once_with()
        page.deleteLater()

    def test_bulk_stop_requires_confirmation_and_restores_controls(self) -> None:
        pending = {
            "键": "variant-1",
            "归属字": "何",
            "变体序号": 1,
            "原始文件名": "何-0001.tif",
            "显示状态": "待优化",
        }
        service = MagicMock()
        service.list_items.side_effect = [[pending], [pending]]
        service.list_batch_items.return_value = ([pending], 2)
        page = OptimizationPage()
        page._service = service
        page._current_item = pending
        page._candidates = [{"方案": {}}]
        page._selected_index = 0
        started: list[object] = []
        with (
            patch.object(page, "_confirm_bulk_optimization", return_value=True),
            patch.object(page._thread_pool, "start", side_effect=started.append),
        ):
            page._confirm_and_start_bulk()

        worker = started[0]
        with patch.object(page, "_confirm_stop_bulk", return_value=False):
            page._request_stop_bulk()
        self.assertFalse(worker.is_cancel_requested())
        self.assertTrue(page._stop_bulk_button.isEnabled())
        self.assertEqual(page._stop_bulk_button.text(), "停止批量优化")

        with patch.object(page, "_confirm_stop_bulk", return_value=True):
            page._request_stop_bulk()
        self.assertTrue(worker.is_cancel_requested())
        self.assertFalse(page._stop_bulk_button.isEnabled())
        self.assertEqual(page._stop_bulk_button.text(), "正在停止…")
        page._bulk_progress_changed("迟到的普通进度", 0, 1)
        self.assertIn("正在停止批量优化", page._message_label.text())
        self.assertNotIn("迟到的普通进度", page._bulk_progress.format())

        with patch.object(QMessageBox, "information") as information:
            worker.run()
            self.app.processEvents()

        service.generate_batch_candidate.assert_not_called()
        service.save_selection.assert_not_called()
        self.assertIsNone(page._bulk_worker)
        self.assertFalse(page._busy)
        self.assertFalse(page._bulk_progress.isVisible())
        self.assertFalse(page._stop_bulk_button.isVisible())
        self.assertEqual(page._stop_bulk_button.text(), "停止批量优化")
        self.assertTrue(page._home_button.isEnabled())
        self.assertTrue(page._complete_button.isEnabled())
        self.assertTrue(page._engine_combo.isEnabled())
        self.assertTrue(page._search_edit.isEnabled())
        self.assertTrue(page._item_tree.isEnabled())
        information.assert_called_once()
        self.assertEqual(information.call_args.args[1], "自动优化已停止")
        summary = information.call_args.args[2]
        self.assertIn("总耗时：", summary)
        self.assertIn("成功 0", summary)
        self.assertIn("失败 0", summary)
        self.assertIn("未处理 1", summary)
        self.assertIn("跳过 2", summary)
        page.deleteLater()

    def test_bulk_stop_does_not_relock_page_when_task_finishes_during_confirmation(
        self,
    ) -> None:
        page = OptimizationPage()
        worker = MagicMock()
        worker.is_cancel_requested.return_value = False
        page._bulk_worker = worker
        page._set_bulk_stop_state(running=True)

        def finish_while_confirming() -> bool:
            page._bulk_worker = None
            page._set_bulk_stop_state(running=False)
            return True

        with patch.object(
            page,
            "_confirm_stop_bulk",
            side_effect=finish_while_confirming,
        ):
            page._request_stop_bulk()

        worker.request_cancel.assert_not_called()
        self.assertFalse(page._stop_bulk_button.isVisible())
        self.assertEqual(page._stop_bulk_button.text(), "停止批量优化")
        page.deleteLater()

    def test_bulk_stop_after_generation_does_not_save_current_item(self) -> None:
        item = {
            "键": "variant-1",
            "归属字": "何",
            "变体序号": 1,
            "原始文件名": "何-0001.tif",
        }
        candidate = {
            "方案名": "已生成但未保存",
            "图像": Image.new("RGBA", (2, 2), (0, 0, 0, 255)),
        }
        service = MagicMock()

        page = OptimizationPage()
        worker = _OptimizationBatchWorker(service, [item], page._engine_context, 0)

        def generate_then_stop(*_args, **_kwargs):
            worker.request_cancel()
            return candidate

        service.generate_batch_candidate.side_effect = generate_then_stop
        service.is_candidate_valid.return_value = True
        result = worker._run_batch()

        self.assertTrue(result["已停止"])
        self.assertEqual(result["成功"], 0)
        self.assertEqual(result["失败"], 0)
        self.assertEqual(result["未处理"], 1)
        service.save_selection.assert_not_called()
        page.deleteLater()

    def test_bulk_cancelled_generation_is_not_reported_as_failure(self) -> None:
        item = {
            "键": "variant-1",
            "归属字": "何",
            "变体序号": 1,
            "原始文件名": "何-0001.tif",
        }
        service = MagicMock()
        service.generate_batch_candidate.side_effect = OptimizationCancelled(
            "自动优化已由用户停止。"
        )
        page = OptimizationPage()
        worker = _OptimizationBatchWorker(service, [item], page._engine_context, 0)
        result = worker._run_batch()

        self.assertTrue(result["已停止"])
        self.assertEqual(result["失败"], 0)
        self.assertEqual(result["未处理"], 1)
        service.save_selection.assert_not_called()
        page.deleteLater()

    def test_bulk_no_safe_candidate_reports_failure_and_continues(self) -> None:
        failed_item = {
            "键": "variant-failed",
            "归属字": "剛",
            "变体序号": 1,
            "原始文件名": "剛-0001.tif",
        }
        succeeding_item = {
            "键": "variant-succeeded",
            "归属字": "割",
            "变体序号": 1,
            "原始文件名": "割-0001.tif",
        }
        candidate = {
            "方案名": "结构安全候选",
            "图像": Image.new("RGBA", (2, 2), (0, 0, 0, 255)),
        }
        service = MagicMock()
        service.generate_batch_candidate.side_effect = [
            ValueError("算法未生成通过结构保护的寻优候选结果。"),
            candidate,
        ]
        service.is_candidate_valid.return_value = True
        page = OptimizationPage()
        worker = _OptimizationBatchWorker(
            service,
            [failed_item, succeeding_item],
            page._engine_context,
            0,
        )

        result = worker._run_batch()

        self.assertFalse(result["已停止"])
        self.assertEqual(result["成功"], 1)
        self.assertEqual(result["失败"], 1)
        self.assertEqual(result["未处理"], 0)
        self.assertEqual(result["失败详情"][0]["键"], "variant-failed")
        self.assertIn("结构保护", result["失败详情"][0]["错误"])
        service.save_selection.assert_called_once()
        self.assertEqual(service.save_selection.call_args.args[0], succeeding_item)
        self.assertEqual(service.save_selection.call_args.kwargs["round_number"], 1)
        candidate["图像"].close()
        page.deleteLater()

    def test_bulk_stop_during_save_finishes_transaction_before_stopping(self) -> None:
        items = [
            {
                "键": f"variant-{index}",
                "归属字": char,
                "变体序号": 1,
                "原始文件名": f"{char}-0001.tif",
            }
            for index, char in enumerate(("何", "是"), 1)
        ]
        candidate = {
            "方案名": "批量候选",
            "图像": Image.new("RGBA", (2, 2), (0, 0, 0, 255)),
        }
        service = MagicMock()
        service.generate_batch_candidate.return_value = candidate
        service.is_candidate_valid.return_value = True
        page = OptimizationPage()
        worker = _OptimizationBatchWorker(service, items, page._engine_context, 0)
        service.save_selection.side_effect = lambda *_args, **_kwargs: worker.request_cancel()
        result = worker._run_batch()

        self.assertTrue(result["已停止"])
        self.assertEqual(result["成功"], 1)
        self.assertEqual(result["失败"], 0)
        self.assertEqual(result["未处理"], 1)
        service.generate_batch_candidate.assert_called_once()
        service.save_selection.assert_called_once()
        self.assertEqual(
            service.save_selection.call_args.args[:2],
            (items[0], candidate),
        )
        self.assertEqual(service.save_selection.call_args.kwargs["round_number"], 1)
        self.assertIs(
            service.save_selection.call_args.kwargs["persistence"],
            service.create_batch_persistence.return_value,
        )
        page.deleteLater()

    def test_bulk_optimization_worker_failure_restores_home_and_navigation(self) -> None:
        pending = {
            "键": "variant-1",
            "归属字": "何",
            "变体序号": 1,
            "原始文件名": "何-0001.tif",
            "显示状态": "待优化",
        }
        service = MagicMock()
        service.list_items.return_value = [pending]
        service.list_batch_items.return_value = ([pending], 0)

        page = OptimizationPage()
        page._service = service
        page._current_item = pending
        page._candidates = [{"方案": {}}]
        started: list[object] = []
        with (
            patch.object(page, "_confirm_bulk_optimization", return_value=True),
            patch.object(page._thread_pool, "start", side_effect=started.append),
        ):
            page._confirm_and_start_bulk()

        worker = started[0]
        with (
            patch.object(
                worker,
                "_run_batch",
                side_effect=RuntimeError("模拟批量线程异常"),
            ),
            patch.object(QMessageBox, "critical") as critical,
        ):
            worker.run()
            self.app.processEvents()

        self.assertIsNone(page._bulk_worker)
        self.assertFalse(page._busy)
        self.assertFalse(page._bulk_progress.isVisible())
        self.assertFalse(page._stop_bulk_button.isVisible())
        self.assertTrue(page._home_button.isEnabled())
        self.assertTrue(page._complete_button.isEnabled())
        self.assertTrue(page._engine_combo.isEnabled())
        self.assertTrue(page._search_edit.isEnabled())
        self.assertTrue(page._item_tree.isEnabled())
        self.assertTrue(page._candidate_list.isEnabled())
        critical.assert_called_once()
        self.assertEqual(critical.call_args.args[:2], (page, "整库自动优化失败"))
        self.assertIn("模拟批量线程异常", critical.call_args.args[2])
        self.assertIn("总耗时：", critical.call_args.args[2])
        page.deleteLater()

    def test_bulk_optimization_refresh_failure_restores_home_and_navigation(self) -> None:
        service = MagicMock()
        service.list_items.side_effect = RuntimeError("模拟批次后刷新失败")
        page = OptimizationPage()
        page._service = service
        page._current_item = {"键": "variant-1"}
        page._candidates = [{"方案": {}}]
        worker = MagicMock()
        page._bulk_worker = worker
        page._busy = True
        page._set_workspace_enabled(False)

        with patch.object(QMessageBox, "critical") as critical:
            page._bulk_finished(
                worker,
                {
                    "成功": 1,
                    "跳过": 0,
                    "失败": 0,
                    "成功字形": ["variant-1"],
                },
            )

        self.assertIsNone(page._bulk_worker)
        self.assertFalse(page._busy)
        self.assertTrue(page._home_button.isEnabled())
        self.assertTrue(page._complete_button.isEnabled())
        self.assertTrue(page._engine_combo.isEnabled())
        self.assertTrue(page._search_edit.isEnabled())
        self.assertTrue(page._item_tree.isEnabled())
        self.assertIn("页面刷新失败", critical.call_args.args[1])
        self.assertIn("返回首页", critical.call_args.args[2])
        page.deleteLater()

    def test_bulk_optimization_start_failure_restores_home_and_navigation(self) -> None:
        pending = {
            "键": "variant-1",
            "归属字": "何",
            "变体序号": 1,
            "原始文件名": "何-0001.tif",
            "显示状态": "待优化",
        }
        service = MagicMock()
        service.list_items.return_value = [pending]
        service.list_batch_items.return_value = ([pending], 0)
        page = OptimizationPage()
        page._service = service

        with (
            patch.object(page, "_confirm_bulk_optimization", return_value=True),
            patch.object(
                page._thread_pool,
                "start",
                side_effect=RuntimeError("模拟线程池启动失败"),
            ),
            patch.object(QMessageBox, "critical") as critical,
        ):
            page._confirm_and_start_bulk()

        self.assertIsNone(page._bulk_worker)
        self.assertFalse(page._busy)
        self.assertFalse(page._bulk_progress.isVisible())
        self.assertFalse(page._stop_bulk_button.isVisible())
        self.assertTrue(page._home_button.isEnabled())
        self.assertTrue(page._complete_button.isEnabled())
        self.assertTrue(page._engine_combo.isEnabled())
        self.assertTrue(page._search_edit.isEnabled())
        self.assertTrue(page._item_tree.isEnabled())
        self.assertIn("无法启动", critical.call_args.args[2])
        self.assertIn("总耗时：", critical.call_args.args[2])
        page.deleteLater()

    def test_bulk_optimization_reports_missing_original_metadata_as_failure(self) -> None:
        damaged = {
            "键": "variant-damaged",
            "归属字": "缺",
            "变体序号": 1,
            "原始文件名": "",
            "原始路径": "",
            "显示状态": "待优化",
        }
        service = MagicMock()
        service.list_items.return_value = []
        service.list_batch_items.return_value = ([damaged], 0)
        service.generate_batch_candidate.side_effect = FileNotFoundError("缺少原始文件")

        page = OptimizationPage()
        page._service = service
        started: list[object] = []
        with (
            patch.object(page, "_confirm_bulk_optimization", return_value=True),
            patch.object(page._thread_pool, "start", side_effect=started.append),
        ):
            page._confirm_and_start_bulk()

        self.assertEqual(len(started), 1)
        worker = started[0]
        with (
            patch.object(QMessageBox, "warning") as warning,
            patch.object(QMessageBox, "information") as information,
        ):
            worker.run()
            self.app.processEvents()

        service.generate_batch_candidate.assert_called_once_with(
            damaged,
            engine_context=page._engine_context,
            cancel_check=worker.is_cancel_requested,
        )
        warning.assert_called_once()
        summary = warning.call_args.args[2]
        self.assertIn("失败 1", summary)
        self.assertIn("缺-字形1：缺少原始文件", summary)
        information.assert_not_called()
        self.assertIsNone(page._bulk_worker)
        page.deleteLater()

    def test_unsaved_candidate_must_be_saved_or_bulk_is_cancelled(self) -> None:
        candidate = {
            "方案名": "人工选择候选",
            "方案": {},
            "得分": 91.0,
            "图像": Image.new("RGBA", (2, 2), (0, 0, 0, 255)),
            "灰度母版": np.zeros((2, 2), dtype=np.uint8),
            "清洁掩码": np.ones((2, 2), dtype=np.uint8),
        }
        current = {
            "键": "variant-1",
            "归属字": "何",
            "原始文件名": "何-0001.tif",
            "显示状态": "待优化",
        }
        page = OptimizationPage()
        service = MagicMock()
        service.is_candidate_valid.return_value = True
        page._service = service
        page._current_item = current
        page._candidates = [candidate]
        page._selected_index = 0
        page._round_number = 3
        page._branch_dirty = True

        with (
            patch.object(page, "_confirm_save_current_before_bulk", return_value=False),
            patch.object(page, "_save_current_before_bulk") as save_current,
            patch.object(page, "_confirm_and_start_bulk") as start_bulk,
        ):
            page._complete_optimization()
        save_current.assert_not_called()
        start_bulk.assert_not_called()
        service.save_selection.assert_not_called()

        updated = dict(current, **{"显示状态": "已优化"})
        service.list_items.return_value = [updated]

        def run_immediately(function, success, _failure, lock_page: bool) -> None:
            self.assertTrue(lock_page)
            success(function())

        page._start_task = run_immediately  # type: ignore[method-assign]
        with (
            patch.object(page, "_confirm_save_current_before_bulk", return_value=True),
            patch.object(page, "_confirm_and_start_bulk") as start_bulk,
        ):
            page._complete_optimization()

        service.save_selection.assert_called_once_with(current, candidate, 3)
        start_bulk.assert_called_once_with()
        self.assertFalse(page._branch_dirty)
        page.deleteLater()


if __name__ == "__main__":
    unittest.main()
