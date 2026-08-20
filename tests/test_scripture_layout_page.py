"""通用经文排版工作台与首页入口回归测试。"""

from __future__ import annotations

import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPoint, QPointF, QSize, QStandardPaths, Qt
from PySide6.QtGui import QImage, QMouseEvent, QWheelEvent
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QWidget,
)

import config
from core.scripture_layout import LayoutParameters, parse_scripture
from data.layout_template_store import (
    DEFAULT_TEMPLATE_ID,
    DEFAULT_TEMPLATE_PARAMETERS,
    LayoutTemplateStore,
)
from ui.main_window import MainWindow
from ui.pages.scripture_layout_page import (
    CUSTOM_TEMPLATE_ID,
    ScriptureLayoutPage,
    _PreviewCanvas,
    _TemplateSaveDialog,
)
from services.scripture_layout_service import GenerationProgress


class _StackStub:
    def __init__(self) -> None:
        self.indices: list[int] = []

    def setCurrentIndex(self, index: int) -> None:
        self.indices.append(index)


class _StatusStub:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def showMessage(self, message: str) -> None:
        self.messages.append(message)


class _WindowStub:
    PAGE_SCRIPTURE_LAYOUT = MainWindow.PAGE_SCRIPTURE_LAYOUT

    def __init__(self) -> None:
        self._stack = _StackStub()
        self._status = _StatusStub()

    def statusBar(self) -> _StatusStub:
        return self._status


class ScriptureLayoutPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_page_contains_confirmed_workbench_regions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            template_path = str(Path(directory) / "排版模板.json")
            with patch.object(config, "LAYOUT_TEMPLATE_FILE", template_path):
                page = ScriptureLayoutPage()
            try:
                self.assertIsNotNone(page.findChild(QPlainTextEdit, "scriptureTextEdit"))
                self.assertIsNotNone(page.findChild(QComboBox, "layoutTemplateCombo"))
                self.assertIsNotNone(page.findChild(QComboBox, "systemLibraryCombo"))
                self.assertIsNotNone(page.findChild(QProgressBar, "layoutGenerationProgress"))
                self.assertIsNotNone(page.findChild(QProgressBar, "glyphSourceCheckProgress"))
                self.assertIsNotNone(page.findChild(QProgressBar, "layoutPreviewProgress"))
                self.assertTrue(page._preview_guides_check.isChecked())
                self.assertEqual(page._preview_guides_check.text(), "显示田字格和框线")
                self.assertEqual(page._tabs.count(), 3)
                self.assertEqual(
                    [page._tabs.tabText(index) for index in range(page._tabs.count())],
                    ["版面", "文字与段落", "输出参数"],
                )
                self.assertFalse(page._tabs.tabBar().isTabVisible(2))
                self.assertIn("总字数", page._scripture_statistics.text())
                self.assertFalse(page._include_punctuation_check.isChecked())
                self.assertFalse(page._external_path_edit.isEnabled())
                self.assertTrue(page._vertical_layout_radio.isChecked())
                self.assertTrue(page._right_to_left_radio.isChecked())
                self.assertEqual(page._open_text_button.text(), "打开文件")
                self.assertTrue(
                    any(
                        label.text() == "通用经文排版"
                        for label in page.findChildren(QLabel)
                    )
                )
                self.assertEqual(page._output_name_edit.text(), "通用经文排版")
                self.assertTrue(page._compress_psd_check.isChecked())
                self.assertEqual(
                    page._output_path_edit.text(),
                    QStandardPaths.writableLocation(
                        QStandardPaths.StandardLocation.DesktopLocation
                    )
                    or config.SCRIPT_DIR,
                )
                self.assertTrue(
                    any(
                        label.text() == "页行列数："
                        for label in page.findChildren(QLabel)
                    )
                )
                labels = {label.text() for label in page.findChildren(QLabel)}
                self.assertIn("画布 DPI：", labels)
                self.assertIn("单元格：", labels)
                self.assertIn("行：", labels)
                self.assertIn("列：", labels)
                self.assertIn("其他规则", labels)
                self.assertIn("排版方式", labels)
                self.assertIn("排列方向：", labels)
                self.assertIn("行进方向：", labels)
                self.assertIn("PSD 压缩：", labels)
                self.assertNotIn("自动缩放：", labels)
                self.assertNotIn("首经题：", labels)
                self.assertNotIn("尾经题：", labels)
                self.assertNotIn("标点符号：", labels)
                self.assertNotIn("末版处理：", labels)
                self.assertNotIn("尺寸标注：", labels)
                self.assertEqual(page._first_title_check.text(), "首经题单列")
                self.assertEqual(page._last_title_check.text(), "尾经题单列")
                self.assertEqual(page._include_punctuation_check.text(), "标点符号")
                self.assertEqual(page._trim_columns_check.text(), "末版压缩")
                self.assertEqual(page._annotation_check.text(), "尺寸标注")
                self.assertNotIn("画布边距（相对最外侧单元格）", labels)
                self.assertNotIn("特殊行列间距：", labels)
                self.assertNotIn(
                    "每个文字保存为独立图层，并按段落归组；每一版完整写入后才替换目标文件。",
                    labels,
                )
                self.assertEqual(page._frame_margin_group.layout().count(), 2)
                self.assertEqual(page._canvas_margin_group.layout().count(), 2)
                self.assertFalse(page.is_running)
                self.assertEqual(
                    page._collect_parameters(),
                    DEFAULT_TEMPLATE_PARAMETERS,
                )
                self.assertIsNotNone(
                    page.findChild(QWidget, "centeredLayoutTemplateGroup")
                )
                self.assertIsNotNone(
                    page.findChild(QPushButton, "saveLayoutTemplateButton")
                )
                page.show()
                self.app.processEvents()
                save_button = page._save_template_button
                tabs_top = page._tabs.mapTo(page, page._tabs.rect().topLeft()).y()
                button_top = save_button.mapTo(page, save_button.rect().topLeft()).y()
                self.assertLess(button_top, tabs_top)
                parameter_right = page._tabs.mapTo(
                    page, page._tabs.rect().topRight()
                ).x()
                button_right = save_button.mapTo(
                    page, save_button.rect().topRight()
                ).x()
                self.assertLessEqual(button_right, parameter_right)
                for removed_name in (
                    "applyLayoutTemplateButton",
                    "layoutTemplateStatus",
                    "saveLayoutTemplateAsButton",
                    "restoreLayoutTemplateButton",
                    "manageLayoutTemplateButton",
                ):
                    self.assertIsNone(page.findChild(QWidget, removed_name))
            finally:
                page.shutdown()
                page.deleteLater()

    def test_source_and_preview_progress_ignore_stale_tasks_and_source_can_stop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            template_path = str(Path(directory) / "排版模板.json")
            with patch.object(config, "LAYOUT_TEMPLATE_FILE", template_path):
                page = ScriptureLayoutPage()
            try:
                source_worker = object()
                source_cancel = threading.Event()
                page._source_generation = 7
                page._source_worker = source_worker
                page._source_cancel = source_cancel
                page._source_progress_changed(
                    7,
                    GenerationProgress(25, 100, "正在解码：甲.png"),
                    source_worker,
                )
                self.assertEqual(page._source_progress.value(), 25)
                self.assertIn("正在解码", page._source_progress.format())

                page._load_glyph_source()
                self.assertTrue(source_cancel.is_set())
                self.assertEqual(page._check_source_button.text(), "正在停止检查…")

                preview_worker = object()
                page._preview_generation = 9
                page._preview_worker = preview_worker
                page._preview_progress_changed(
                    9,
                    GenerationProgress(60, 100, "正在绘制字图：乙"),
                    preview_worker,
                )
                self.assertEqual(page._preview_progress.value(), 60)
                page._preview_progress_changed(
                    8,
                    GenerationProgress(90, 100, "迟到的旧预览"),
                    preview_worker,
                )
                self.assertEqual(page._preview_progress.value(), 60)

                image = QImage(32, 32, QImage.Format.Format_RGB32)
                image.fill(Qt.GlobalColor.white)
                page._preview_ready(9, image, preview_worker)
                self.assertTrue(page._preview_progress.isHidden())
            finally:
                page._source_worker = None
                page._preview_worker = None
                page.shutdown()
                page.deleteLater()

    def test_parameter_modes_disable_unrelated_controls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            template_path = str(Path(directory) / "排版模板.json")
            with patch.object(config, "LAYOUT_TEMPLATE_FILE", template_path):
                page = ScriptureLayoutPage()
            try:
                self.assertTrue(page._source_scale_group.isEnabled())
                self.assertFalse(page._cell_scale_group.isEnabled())
                self.assertFalse(page._scale_source_radio.property("inactiveOption"))
                self.assertTrue(page._scale_cell_radio.property("inactiveOption"))
                self.assertFalse(page._special_row_gap_label.isEnabled())
                self.assertFalse(page._special_column_gap_label.isEnabled())
                page._scale_cell_radio.setChecked(True)
                self.app.processEvents()
                self.assertFalse(page._source_scale_group.isEnabled())
                self.assertTrue(page._cell_scale_group.isEnabled())
                self.assertTrue(page._scale_source_radio.property("inactiveOption"))
                self.assertFalse(page._scale_cell_radio.property("inactiveOption"))

                page._paragraph_column_radio.setChecked(True)
                self.app.processEvents()
                self.assertFalse(page._paragraph_skip_group.isEnabled())
                self.assertTrue(page._paragraph_skip_radio.property("inactiveOption"))
                self.assertFalse(page._paragraph_column_radio.property("inactiveOption"))

                page._horizontal_layout_radio.setChecked(True)
                page._left_to_right_radio.setChecked(True)
                self.app.processEvents()
                parameters = page._collect_parameters()
                self.assertEqual(parameters.layout_mode, "横排")
                self.assertEqual(parameters.flow_direction, "从左到右")
                self.assertEqual(page._paragraph_column_radio.text(), "段后换行")
                self.assertEqual(page._first_title_check.text(), "首经题单行")
                self.assertEqual(page._last_title_check.text(), "尾经题单行")
                self.assertEqual(page._trim_columns_check.text(), "末版压缩")

                page._draw_frame_check.setChecked(False)
                self.assertFalse(page._frame_margin_label.isEnabled())
                self.assertTrue(
                    all(not control.isEnabled() for control in page._frame_margin_controls)
                )
                self.assertTrue(
                    all(not control.isEnabled() for control in page._special_gap_controls)
                )
                page._special_gaps_check.setChecked(True)
                self.assertTrue(
                    all(control.isEnabled() for control in page._special_gap_controls)
                )
                self.assertTrue(page._special_row_gap_label.isEnabled())
                self.assertTrue(page._special_column_gap_label.isEnabled())

                page._external_source_radio.setChecked(True)
                self.app.processEvents()
                self.assertFalse(page._system_library_combo.isEnabled())
                self.assertTrue(page._external_path_edit.isEnabled())
                self.assertTrue(page._system_source_radio.property("inactiveOption"))
                self.assertFalse(page._external_source_radio.property("inactiveOption"))
            finally:
                page.shutdown()
                page.deleteLater()

    def test_layout_parameter_rows_share_input_and_text_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            template_path = str(Path(directory) / "排版模板.json")
            with patch.object(config, "LAYOUT_TEMPLATE_FILE", template_path):
                page = ScriptureLayoutPage()
            try:
                page.resize(1400, 900)
                page.show()
                page._tabs.setCurrentIndex(1)
                self.app.processEvents()

                left_width, preview_width, right_width = page._splitter.sizes()
                self.assertAlmostEqual(
                    left_width / preview_width,
                    300 / 620,
                    delta=0.03,
                )
                self.assertLess(left_width, 340)
                self.assertEqual(right_width, 480)
                suffix_width = page._cell_height_spin.fontMetrics().horizontalAdvance(
                    " mm"
                )
                self.assertGreaterEqual(
                    page._cell_height_spin.lineEdit().width(),
                    suffix_width + 8,
                )
                self.assertLess(
                    page._special_gaps_check.mapTo(page, QPoint()).x(),
                    page._special_row_gap_label.mapTo(page, QPoint()).x(),
                )

                dpi_x = page._dpi_spin.mapTo(page, QPoint()).x()
                self.assertEqual(
                    dpi_x,
                    page._cell_height_spin.mapTo(page, QPoint()).x(),
                )
                self.assertEqual(
                    dpi_x,
                    page._special_row_gaps_edit.mapTo(page, QPoint()).x(),
                )

                main_label = next(
                    label
                    for label in page.findChildren(QLabel)
                    if label.text() == "单元格："
                )
                inline_label = next(
                    label
                    for label in page._cell_height_spin.parentWidget().findChildren(QLabel)
                    if label.text() == "高："
                )
                main_center = main_label.mapTo(page, QPoint()).y() + main_label.height() // 2
                inline_center = inline_label.mapTo(page, QPoint()).y() + inline_label.height() // 2
                self.assertAlmostEqual(main_center, inline_center, delta=1)

                protected_texts = {"高：", "宽：", "行：", "列：", "阈值：", "目标："}
                for label in page.findChildren(QLabel):
                    if label.text() not in protected_texts:
                        continue
                    metrics = label.fontMetrics()
                    glyph_width = max(
                        metrics.horizontalAdvance(label.text()),
                        metrics.boundingRect(label.text()).width(),
                    )
                    self.assertGreaterEqual(label.width() - glyph_width, 8)

                zoom_metrics = page._zoom_combo.fontMetrics()
                self.assertGreaterEqual(
                    page._zoom_combo.minimumWidth(),
                    zoom_metrics.horizontalAdvance("适合窗口") + 48,
                )
                self.assertEqual(page._shrink_threshold_spin.value(), 150)
                threshold_metrics = page._shrink_threshold_spin.fontMetrics()
                threshold_width = max(
                    threshold_metrics.horizontalAdvance("150%"),
                    threshold_metrics.boundingRect("150%").width(),
                )
                self.assertGreaterEqual(
                    page._shrink_threshold_spin.lineEdit().width(),
                    threshold_width + 8,
                )
                parameter_right = page._tabs.mapTo(
                    page, page._tabs.rect().topRight()
                ).x()
                for control in (
                    page._enlarge_threshold_spin,
                    page._enlarge_fill_spin,
                    page._shrink_threshold_spin,
                    page._shrink_fill_spin,
                ):
                    control_right = control.mapTo(
                        page, control.rect().topRight()
                    ).x()
                    self.assertLessEqual(control_right, parameter_right)
                self.assertEqual(
                    page._first_title_check.mapTo(page, QPoint()).y(),
                    page._last_title_check.mapTo(page, QPoint()).y(),
                )
                self.assertEqual(
                    page._include_punctuation_check.mapTo(page, QPoint()).y(),
                    page._trim_columns_check.mapTo(page, QPoint()).y(),
                )
                self.assertEqual(
                    page._trim_columns_check.mapTo(page, QPoint()).y(),
                    page._annotation_check.mapTo(page, QPoint()).y(),
                )
            finally:
                page.close()
                page.shutdown()
                page.deleteLater()

    def test_parameter_wheel_is_ignored_but_preview_wheel_remains_available(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            template_path = str(Path(directory) / "排版模板.json")
            with patch.object(config, "LAYOUT_TEMPLATE_FILE", template_path):
                page = ScriptureLayoutPage()
            try:
                page.show()
                self.app.processEvents()
                spin = page._dpi_spin
                spin.setValue(300)
                position = QPointF(spin.rect().center())
                wheel = QWheelEvent(
                    position,
                    QPointF(spin.mapToGlobal(position.toPoint())),
                    QPoint(),
                    QPoint(0, 120),
                    Qt.MouseButton.NoButton,
                    Qt.KeyboardModifier.NoModifier,
                    Qt.ScrollPhase.ScrollUpdate,
                    False,
                )

                QApplication.sendEvent(spin, wheel)

                self.assertEqual(spin.value(), 300)
                self.assertFalse(wheel.isAccepted())

                combo = page._system_library_combo
                combo.clear()
                combo.addItems(["字库甲", "字库乙", "字库丙"])
                combo.setCurrentIndex(1)
                combo_position = QPointF(combo.rect().center())
                combo_wheel = QWheelEvent(
                    combo_position,
                    QPointF(combo.mapToGlobal(combo_position.toPoint())),
                    QPoint(),
                    QPoint(0, 120),
                    Qt.MouseButton.NoButton,
                    Qt.KeyboardModifier.NoModifier,
                    Qt.ScrollPhase.ScrollUpdate,
                    False,
                )

                QApplication.sendEvent(combo, combo_wheel)

                self.assertEqual(combo.currentIndex(), 1)
                self.assertFalse(combo_wheel.isAccepted())
            finally:
                page.close()
                page.shutdown()
                page.deleteLater()

    def test_text_menu_statistics_and_missing_summary_are_chinese_and_compact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            template_path = str(Path(directory) / "排版模板.json")
            with patch.object(config, "LAYOUT_TEMPLATE_FILE", template_path):
                page = ScriptureLayoutPage()
            try:
                menu = page._text_edit.createStandardContextMenu()
                self.assertEqual(
                    [action.text() for action in menu.actions() if not action.isSeparator()],
                    ["撤销", "重做", "剪切", "复制", "粘贴", "删除", "全选"],
                )
                menu.deleteLater()

                page._text_edit.setPlainText("一，一無無")
                self.app.processEvents()
                self.assertIn("总字数：4", page._scripture_statistics.text())
                self.assertIn("去重后字数：2", page._scripture_statistics.text())
                self.assertIn("标点符号数：1", page._scripture_statistics.text())

                parsed = parse_scripture("一一無無無", {"二"}, False)
                page._update_missing_results(parsed)
                self.assertEqual(page._source_summary.text(), "缺失 2 个字，共 5 处")
                self.assertEqual(page._check_results.toPlainText(), "一（2）、無（3）")
            finally:
                page.shutdown()
                page.deleteLater()

    def test_high_definition_preview_size_is_memory_bounded(self) -> None:
        size = ScriptureLayoutPage._bounded_preview_render_size(
            QSize(30_000, 20_000),
            QSize(30_000, 20_000),
        )

        self.assertLessEqual(size.width() * size.height(), 12_000_000)
        self.assertGreater(size.width(), size.height())

    def test_template_selection_applies_immediately_and_tracks_custom_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            template_path = str(Path(directory) / "排版模板.json")
            store = LayoutTemplateStore(template_path)
            custom = store.save(
                "十八行横排",
                LayoutParameters(
                    dpi=300,
                    rows=18,
                    columns=24,
                    include_punctuation=True,
                ),
            )
            with patch.object(config, "LAYOUT_TEMPLATE_FILE", template_path):
                page = ScriptureLayoutPage()
            try:
                self.assertEqual(page._applied_template_id, DEFAULT_TEMPLATE_ID)
                self.assertEqual(page._rows_spin.value(), 20)
                self.assertFalse(page._save_template_button.isEnabled())
                index = page._template_combo.findData(custom.template_id)
                page._template_combo.setCurrentIndex(index)

                self.assertEqual(page._applied_template_id, custom.template_id)
                self.assertEqual(page._dpi_spin.value(), 300)
                self.assertEqual(page._rows_spin.value(), 18)
                self.assertTrue(page._include_punctuation_check.isChecked())
                self.assertEqual(page._template_combo.currentText(), "十八行横排")

                page._rows_spin.setValue(19)
                self.assertEqual(page._template_combo.currentText(), "自定义模板")
                self.assertEqual(
                    page._template_combo.currentData(),
                    CUSTOM_TEMPLATE_ID,
                )
                self.assertTrue(page._save_template_button.isEnabled())

                page._rows_spin.setValue(18)
                self.assertEqual(page._template_combo.currentText(), "十八行横排")
                self.assertEqual(
                    page._template_combo.currentData(),
                    custom.template_id,
                )
                self.assertFalse(page._save_template_button.isEnabled())
            finally:
                page.shutdown()
                page.deleteLater()

    def test_template_save_dialog_only_replaces_user_templates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LayoutTemplateStore(str(Path(directory) / "排版模板.json"))
            custom = store.save("测试模板", LayoutParameters(rows=18), "测试说明")
            dialog = _TemplateSaveDialog(store)
            try:
                self.assertEqual(dialog.mode(), _TemplateSaveDialog.MODE_NEW)
                self.assertTrue(dialog.new_radio.isChecked())
                self.assertTrue(dialog.replace_radio.isEnabled())
                self.assertEqual(dialog.replace_combo.count(), 1)
                self.assertEqual(dialog.replace_combo.itemData(0), custom.template_id)
                self.assertEqual(dialog.replace_combo.findData(DEFAULT_TEMPLATE_ID), -1)

                dialog.replace_radio.setChecked(True)
                self.assertEqual(dialog.mode(), _TemplateSaveDialog.MODE_REPLACE)
                self.assertTrue(dialog.replace_combo.isEnabled())
                self.assertFalse(dialog.name_edit.isEnabled())
                self.assertEqual(dialog.description(), "测试说明")
            finally:
                dialog.deleteLater()

        with tempfile.TemporaryDirectory() as directory:
            empty_store = LayoutTemplateStore(str(Path(directory) / "排版模板.json"))
            dialog = _TemplateSaveDialog(empty_store)
            try:
                self.assertFalse(dialog.replace_radio.isEnabled())
                self.assertEqual(dialog.replace_combo.count(), 0)
                self.assertEqual(dialog.mode(), _TemplateSaveDialog.MODE_NEW)
            finally:
                dialog.deleteLater()

    def test_preview_canvas_aspect_fit_rect_preserves_narrow_page_ratio(self) -> None:
        vertical = _PreviewCanvas._aspect_fit_rect(
            QSize(400, 600),
            QSize(100, 1000),
        )
        horizontal = _PreviewCanvas._aspect_fit_rect(
            QSize(600, 400),
            QSize(1000, 100),
        )

        self.assertEqual(vertical.size(), QSize(60, 600))
        self.assertEqual(vertical.x(), 170)
        self.assertEqual(horizontal.size(), QSize(600, 60))
        self.assertEqual(horizontal.y(), 170)

    def test_preview_canvas_allows_narrow_page_and_restores_empty_minimum(self) -> None:
        canvas = _PreviewCanvas()
        try:
            image = QImage(80, 800, QImage.Format.Format_RGB32)
            image.fill(Qt.GlobalColor.white)

            canvas.set_image(image)
            self.assertEqual(canvas.minimumSize(), QSize(1, 1))
            canvas.resize(60, 600)
            self.assertEqual(canvas.size(), QSize(60, 600))

            canvas.set_message("暂无预览")
            self.assertEqual(canvas.minimumSize(), QSize(320, 320))
        finally:
            canvas.deleteLater()

    def test_layout_route_opens_real_page(self) -> None:
        window = _WindowStub()

        MainWindow._open_tool(window, "layout")

        self.assertEqual(window._stack.indices, [MainWindow.PAGE_SCRIPTURE_LAYOUT])
        self.assertIn("通用经文排版", window._status.messages[-1])

    def test_preview_wheel_zoom_and_mouse_pan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            template_path = str(Path(directory) / "排版模板.json")
            with patch.object(config, "LAYOUT_TEMPLATE_FILE", template_path):
                page = ScriptureLayoutPage()
            try:
                page.resize(1400, 900)
                page.show()
                page._preview_image = QImage(1200, 900, QImage.Format.Format_RGB32)
                page._preview_image.fill(Qt.GlobalColor.white)
                page._preview_zoom = 0.0
                page._apply_preview_zoom()
                self.app.processEvents()

                viewport = page._preview_scroll.viewport()
                position = QPointF(viewport.rect().center())
                wheel = QWheelEvent(
                    position,
                    QPointF(viewport.mapToGlobal(position.toPoint())),
                    QPoint(),
                    QPoint(0, 120),
                    Qt.MouseButton.NoButton,
                    Qt.KeyboardModifier.NoModifier,
                    Qt.ScrollPhase.ScrollUpdate,
                    False,
                )
                QApplication.sendEvent(viewport, wheel)
                self.app.processEvents()

                self.assertGreater(page._preview_zoom, 0.0)
                self.assertNotEqual(page._zoom_combo.currentText(), "适合窗口")
                self.assertTrue(wheel.isAccepted())

                page._preview_zoom = 3.0
                page._apply_preview_zoom()
                self.app.processEvents()
                horizontal = page._preview_scroll.horizontalScrollBar()
                vertical = page._preview_scroll.verticalScrollBar()
                self.assertGreater(horizontal.maximum(), 0)
                self.assertGreater(vertical.maximum(), 0)
                horizontal.setValue(min(200, horizontal.maximum() // 2))
                vertical.setValue(min(200, vertical.maximum() // 2))
                old_horizontal = horizontal.value()
                old_vertical = vertical.value()

                press_position = QPointF(180, 160)
                move_position = QPointF(120, 110)
                QApplication.sendEvent(
                    viewport,
                    QMouseEvent(
                        QEvent.Type.MouseButtonPress,
                        press_position,
                        press_position,
                        Qt.MouseButton.LeftButton,
                        Qt.MouseButton.LeftButton,
                        Qt.KeyboardModifier.NoModifier,
                    ),
                )
                self.assertEqual(
                    viewport.cursor().shape(),
                    Qt.CursorShape.ClosedHandCursor,
                )
                QApplication.sendEvent(
                    viewport,
                    QMouseEvent(
                        QEvent.Type.MouseMove,
                        move_position,
                        move_position,
                        Qt.MouseButton.NoButton,
                        Qt.MouseButton.LeftButton,
                        Qt.KeyboardModifier.NoModifier,
                    ),
                )
                QApplication.sendEvent(
                    viewport,
                    QMouseEvent(
                        QEvent.Type.MouseButtonRelease,
                        move_position,
                        move_position,
                        Qt.MouseButton.LeftButton,
                        Qt.MouseButton.NoButton,
                        Qt.KeyboardModifier.NoModifier,
                    ),
                )

                self.assertGreater(horizontal.value(), old_horizontal)
                self.assertGreater(vertical.value(), old_vertical)
                self.assertEqual(
                    viewport.cursor().shape(),
                    Qt.CursorShape.OpenHandCursor,
                )
            finally:
                page.close()
                page.shutdown()
                page.deleteLater()

    def test_missing_glyph_confirmation_can_continue_or_cancel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            template_path = str(Path(directory) / "排版模板.json")
            with patch.object(config, "LAYOUT_TEMPLATE_FILE", template_path):
                page = ScriptureLayoutPage()
            try:
                with patch(
                    "ui.pages.scripture_layout_page.QMessageBox"
                ) as message_box:
                    dialog = message_box.return_value
                    continue_button = object()
                    cancel_button = object()
                    dialog.addButton.side_effect = [continue_button, cancel_button]
                    dialog.clickedButton.return_value = continue_button

                    self.assertTrue(page._confirm_missing_characters({"乙": 2, "丙": 1}))
                    prompt = dialog.setText.call_args.args[0]
                    self.assertIn("2 种、共 3 处缺字", prompt)
                    self.assertIn("单元格将保留空白", prompt)
                    dialog.setDefaultButton.assert_called_once_with(cancel_button)
                    dialog.setEscapeButton.assert_called_once_with(cancel_button)

                    dialog.addButton.side_effect = [continue_button, cancel_button]
                    dialog.clickedButton.return_value = cancel_button
                    self.assertFalse(page._confirm_missing_characters({"乙": 1}))
            finally:
                page.shutdown()
                page.deleteLater()

    def test_large_preview_uses_fit_scale_as_wheel_floor_without_rerender(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            template_path = str(Path(directory) / "排版模板.json")
            with patch.object(config, "LAYOUT_TEMPLATE_FILE", template_path):
                page = ScriptureLayoutPage()
            try:
                page.resize(1400, 900)
                page.show()
                page._preview_image = QImage(
                    3000,
                    24000,
                    QImage.Format.Format_RGB32,
                )
                page._preview_image.fill(Qt.GlobalColor.white)
                page._preview_document_size = QSize(3000, 24000)
                page._preview_zoom = 0.0
                page._apply_preview_zoom()
                self.app.processEvents()

                fit_zoom = page._fit_preview_zoom()
                self.assertLess(fit_zoom, 0.1)
                self.assertAlmostEqual(page._minimum_preview_zoom(), fit_zoom)
                with patch.object(page, "_schedule_quality_preview") as schedule:
                    page._zoom_preview_at(
                        QPointF(page._preview_scroll.viewport().rect().center()),
                        1.15,
                    )
                    self.assertAlmostEqual(page._preview_zoom, fit_zoom * 1.15)
                    self.assertEqual(schedule.call_count, 1)
                    self.assertNotEqual(page._zoom_combo.currentText(), "10%")

                    page._preview_zoom = fit_zoom
                    page._apply_preview_zoom()
                    schedule.reset_mock()
                    page._zoom_preview_at(
                        QPointF(page._preview_scroll.viewport().rect().center()),
                        1 / 1.15,
                    )
                    self.assertAlmostEqual(page._preview_zoom, fit_zoom)
                    schedule.assert_not_called()
            finally:
                page.shutdown()
                page.deleteLater()

    def test_preview_reuses_same_context_when_existing_image_is_clear_enough(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            template_path = str(Path(directory) / "排版模板.json")
            with patch.object(config, "LAYOUT_TEMPLATE_FILE", template_path):
                page = ScriptureLayoutPage()
            try:
                context = ("第1版", "参数", True)
                page._preview_image = QImage(
                    1200,
                    900,
                    QImage.Format.Format_RGB32,
                )
                page._preview_image.fill(Qt.GlobalColor.white)
                page._preview_document_size = QSize(2400, 1800)
                page._preview_render_context = context
                page._preview_cancel = threading.Event()
                page._preview_progress.show()
                generation = page._preview_generation

                self.assertTrue(
                    page._reuse_current_preview(context, QSize(800, 600))
                )
                self.assertTrue(page._preview_cancel.is_set())
                self.assertEqual(page._preview_generation, generation + 1)
                self.assertTrue(page._preview_progress.isHidden())
                self.assertFalse(
                    page._reuse_current_preview(context, QSize(1600, 1200))
                )
            finally:
                page._preview_cancel = None
                page.shutdown()
                page.deleteLater()

    def test_psd_writing_progress_uses_busy_state_then_restores_percentage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            template_path = str(Path(directory) / "排版模板.json")
            with patch.object(config, "LAYOUT_TEMPLATE_FILE", template_path):
                page = ScriptureLayoutPage()
            try:
                page._generation_progress(
                    GenerationProgress(
                        99,
                        100,
                        "正在压缩并写入第 1/1 版 PSD",
                        True,
                    )
                )
                self.assertEqual(page._progress_bar.minimum(), 0)
                self.assertEqual(page._progress_bar.maximum(), 0)
                self.assertIn("正在压缩并写入", page._progress_bar.format())

                page._generation_progress(
                    GenerationProgress(100, 100, "第 1 版已保存")
                )
                self.assertEqual(page._progress_bar.maximum(), 100)
                self.assertEqual(page._progress_bar.value(), 100)
                self.assertIn("%p%", page._progress_bar.format())
            finally:
                page.shutdown()
                page.deleteLater()

    def test_stop_generation_sets_cancel_event_without_waiting_for_worker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            template_path = str(Path(directory) / "排版模板.json")
            with patch.object(config, "LAYOUT_TEMPLATE_FILE", template_path):
                page = ScriptureLayoutPage()
            try:
                cancel_event = threading.Event()
                page._generation_cancel = cancel_event
                page._set_running(True)

                page._stop_generation()

                self.assertTrue(cancel_event.is_set())
                self.assertFalse(page._stop_button.isEnabled())
                self.assertIn("立即停止", page._progress_bar.format())
            finally:
                page._generation_cancel = None
                page.shutdown()
                page.deleteLater()

if __name__ == "__main__":
    unittest.main()
