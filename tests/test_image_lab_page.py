"""图片实验室页面交互测试。"""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QWheelEvent
from PySide6.QtWidgets import QApplication, QFileDialog, QLabel, QMessageBox

from data.image_lab_project_store import ImageLabStroke
from services.image_lab_service import ImageLabService
from ui.pages.image_lab_page import ImageLabPage


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class ImageLabPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _wait_preview(self, page: ImageLabPage) -> None:
        deadline = time.monotonic() + 8.0
        while page._preview_worker is not None and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.app.processEvents()
        self.assertIsNone(page._preview_worker)
        self.assertIsNotNone(page._preview)

    def _wait_detail(self, page: ImageLabPage) -> None:
        deadline = time.monotonic() + 8.0
        while page._detail_worker is not None and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.app.processEvents()
        self.assertIsNone(page._detail_worker)

    def test_workspace_loads_preview_and_tracks_manual_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = os.path.join(temp_dir, "拓片.png")
            source = np.full((320, 240, 3), 220, dtype=np.uint8)
            cv2.putText(
                source,
                "A",
                (65, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                3.0,
                (25, 25, 25),
                8,
                cv2.LINE_AA,
            )
            Image.fromarray(source).save(source_path)
            service = ImageLabService()
            page = ImageLabPage(service=service)
            page.resize(1100, 720)
            page.show()
            project = service.create_project(source_path)

            page._set_project(project, dirty=False)
            self._wait_preview(page)

            self.assertTrue(page._canvas.has_image)
            self.assertIn("原稿尺寸", page._metrics_label.text())
            self.assertFalse(page.is_dirty)
            rubbing_index = page._processing_mode.findData("rubbing_dark")
            page._processing_mode.setCurrentIndex(rubbing_index)
            page._apply_options()
            self._wait_preview(page)
            self.assertEqual(project.options.processing_mode, "rubbing_dark")
            self.assertIn("笔画尺度重建", page._metrics_label.text())
            self.assertTrue(page._feather_edges.isChecked())
            page._feather_edges.setChecked(False)
            page._apply_options()
            self._wait_preview(page)
            self.assertFalse(project.options.feather_edges)
            self.assertIn("边缘羽化：关闭", page._metrics_label.text())
            page._canvas.set_zoom(2.0)
            page._start_preview()
            self._wait_preview(page)
            self.assertAlmostEqual(page._canvas.zoom_factor, 2.0)
            page._stroke_finished("cover", 30, ((0.5, 0.5),))
            self.assertTrue(page.is_dirty)
            self.assertEqual(len(project.strokes), 1)
            page._undo_stroke()
            self.assertEqual(project.strokes, [])
            page.close()
            page.deleteLater()

    def test_restore_stroke_is_available(self) -> None:
        stroke = ImageLabStroke("restore", 20, ((0.2, 0.3), (0.4, 0.5)))
        self.assertEqual(stroke.tool, "restore")

    def test_zoomed_reduced_preview_loads_visible_detail_region(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = os.path.join(temp_dir, "高清滚动.png")
            source = np.full((600, 1200, 3), 225, dtype=np.uint8)
            cv2.putText(
                source,
                "DETAIL",
                (260, 340),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.5,
                (20, 45, 90),
                8,
                cv2.LINE_AA,
            )
            Image.fromarray(source).save(source_path)
            service = ImageLabService()
            project = service.create_project(source_path)
            preview = service.load_preview(project, max_edge=400)
            page = ImageLabPage(service=service)
            page.resize(1100, 720)
            page.show()
            page._project = project
            page._preview = preview
            page._canvas.set_preview(
                preview.source,
                preview.composite,
                preview.effective_alpha,
                preview.cleanup.uncertainty_mask,
                source_width=project.source_width,
                source_height=project.source_height,
            )
            page._canvas.set_zoom(2.0)
            self.app.processEvents()

            original_load_detail = service.load_detail_preview

            def slow_load_detail(*args, **kwargs):  # type: ignore[no-untyped-def]
                time.sleep(0.2)
                return original_load_detail(*args, **kwargs)

            with patch.object(
                service,
                "load_detail_preview",
                side_effect=slow_load_detail,
            ):
                page._request_detail_preview()
                self.assertTrue(page._progress.isVisible())
                self.assertEqual(page._progress.maximum(), 0)
                self.assertIn("正在加载高清图", page._progress.format())
                self.assertFalse(page._stop_button.isVisible())
                self._wait_detail(page)

            self.assertFalse(page._progress.isVisible())

            self.assertIsNotNone(page._canvas.detail_source_rect)
            page._canvas.set_zoom(1.0)
            self.app.processEvents()
            self.assertIsNone(page._canvas.detail_source_rect)
            page.close()
            page.deleteLater()

    def test_home_button_is_the_rightmost_header_action(self) -> None:
        page = ImageLabPage()
        page.resize(1100, 720)
        page.show()
        self.app.processEvents()

        self.assertEqual(page._home_button.text(), "返回首页")
        self.assertGreater(
            page._home_button.geometry().left(),
            page._save_button.geometry().right(),
        )
        self.assertLessEqual(page._home_button.geometry().right(), page.width())

        emissions: list[bool] = []
        page.home_requested.connect(lambda: emissions.append(True))
        page._home_button.click()
        self.assertEqual(emissions, [True])
        page.close()
        page.deleteLater()

    def test_zoom_buttons_change_canvas_scale_and_have_chinese_hints(self) -> None:
        page = ImageLabPage()
        source = np.zeros((40, 60, 3), dtype=np.uint8)
        alpha = np.zeros((40, 60), dtype=np.uint8)
        page._canvas.set_preview(
            source,
            source,
            alpha,
            alpha,
            source_width=60,
            source_height=40,
        )
        page.show()
        self.app.processEvents()
        page._set_project_available(True)

        self.assertEqual(page._zoom_in_button.toolTip(), "放大")
        self.assertEqual(page._zoom_out_button.toolTip(), "缩小")
        self.assertEqual(page._zoom_in_button.accessibleName(), "放大")
        self.assertEqual(page._zoom_out_button.accessibleName(), "缩小")
        self.assertEqual(page._zoom_in_button.height(), page._fit_button.height())
        self.assertEqual(page._zoom_out_button.height(), page._fit_button.height())
        self.assertEqual(
            page._zoom_in_button.toolButtonStyle().name,
            "ToolButtonIconOnly",
        )
        self.assertEqual(
            page._zoom_out_button.toolButtonStyle().name,
            "ToolButtonIconOnly",
        )

        page._zoom_in_button.click()
        self.assertAlmostEqual(page._canvas.zoom_factor, 1.15)
        page._zoom_out_button.click()
        self.assertAlmostEqual(page._canvas.zoom_factor, 1.0)
        page.close()
        page.deleteLater()

    def test_canvas_pan_signal_moves_scrollbars_only(self) -> None:
        page = ImageLabPage()
        page.resize(1100, 720)
        page.show()
        page._canvas.setFixedSize(1800, 1400)
        self.app.processEvents()
        horizontal = page._canvas_scroll.horizontalScrollBar()
        vertical = page._canvas_scroll.verticalScrollBar()
        horizontal.setValue(300)
        vertical.setValue(240)
        dirty_before = page.is_dirty

        page._pan_canvas(QPoint(35, -20))

        self.assertEqual(horizontal.value(), 265)
        self.assertEqual(vertical.value(), 260)
        self.assertEqual(page.is_dirty, dirty_before)
        page.close()
        page.deleteLater()

    def test_alt_wheel_on_scroll_viewport_zooms_canvas(self) -> None:
        page = ImageLabPage()
        source = np.zeros((40, 60, 3), dtype=np.uint8)
        alpha = np.zeros((40, 60), dtype=np.uint8)
        page._canvas.set_preview(
            source,
            source,
            alpha,
            alpha,
            source_width=60,
            source_height=40,
        )
        page.show()
        self.app.processEvents()
        position = QPointF(20.0, 20.0)
        global_position = QPointF(
            page._canvas_scroll.viewport().mapToGlobal(QPoint(20, 20))
        )
        wheel = QWheelEvent(
            position,
            global_position,
            QPoint(),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.AltModifier,
            Qt.ScrollPhase.ScrollUpdate,
            False,
        )

        QApplication.sendEvent(page._canvas_scroll.viewport(), wheel)

        self.assertTrue(wheel.isAccepted())
        self.assertAlmostEqual(page._canvas.zoom_factor, 1.15)
        page.close()
        page.deleteLater()

    def test_preview_scroll_area_override_handles_alt_wheel_directly(self) -> None:
        page = ImageLabPage()
        source = np.zeros((40, 60, 3), dtype=np.uint8)
        alpha = np.zeros((40, 60), dtype=np.uint8)
        page._canvas.set_preview(
            source,
            source,
            alpha,
            alpha,
            source_width=60,
            source_height=40,
        )
        position = QPointF(20.0, 20.0)
        wheel = QWheelEvent(
            position,
            position,
            QPoint(),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.AltModifier,
            Qt.ScrollPhase.ScrollUpdate,
            False,
        )

        page._canvas_scroll.wheelEvent(wheel)

        self.assertTrue(wheel.isAccepted())
        self.assertAlmostEqual(page._canvas.zoom_factor, 1.15)
        page.deleteLater()

    def test_alt_high_resolution_wheel_on_viewport_also_zooms(self) -> None:
        page = ImageLabPage()
        source = np.zeros((40, 60, 3), dtype=np.uint8)
        alpha = np.zeros((40, 60), dtype=np.uint8)
        page._canvas.set_preview(
            source,
            source,
            alpha,
            alpha,
            source_width=60,
            source_height=40,
        )
        page.show()
        self.app.processEvents()
        position = QPointF(20.0, 20.0)
        global_position = QPointF(
            page._canvas_scroll.viewport().mapToGlobal(QPoint(20, 20))
        )
        wheel = QWheelEvent(
            position,
            global_position,
            QPoint(0, -15),
            QPoint(),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.AltModifier,
            Qt.ScrollPhase.ScrollUpdate,
            False,
        )

        QApplication.sendEvent(page._canvas_scroll.viewport(), wheel)

        self.assertTrue(wheel.isAccepted())
        self.assertAlmostEqual(page._canvas.zoom_factor, 1.0 / 1.15)
        page.close()
        page.deleteLater()

    def test_tracked_alt_key_zooms_when_wheel_event_loses_modifier(self) -> None:
        page = ImageLabPage()
        source = np.zeros((40, 60, 3), dtype=np.uint8)
        alpha = np.zeros((40, 60), dtype=np.uint8)
        page._canvas.set_preview(
            source,
            source,
            alpha,
            alpha,
            source_width=60,
            source_height=40,
        )
        page.show()
        self.app.processEvents()
        alt_press = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_Alt,
            Qt.KeyboardModifier.AltModifier,
        )
        QApplication.sendEvent(page._save_button, alt_press)
        self.assertTrue(page._alt_zoom_held)
        position = QPointF(20.0, 20.0)
        global_position = QPointF(
            page._canvas_scroll.viewport().mapToGlobal(QPoint(20, 20))
        )
        wheel = QWheelEvent(
            position,
            global_position,
            QPoint(),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.ScrollUpdate,
            False,
        )

        QApplication.sendEvent(page._canvas_scroll.viewport(), wheel)

        self.assertTrue(wheel.isAccepted())
        self.assertAlmostEqual(page._canvas.zoom_factor, 1.15)
        alt_release = QKeyEvent(
            QEvent.Type.KeyRelease,
            Qt.Key.Key_Alt,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(page._save_button, alt_release)
        self.assertFalse(page._alt_zoom_held)
        page.close()
        page.deleteLater()

    def test_page_routes_space_to_canvas_and_clears_state_when_hidden(self) -> None:
        page = ImageLabPage()
        source = np.zeros((40, 60, 3), dtype=np.uint8)
        alpha = np.zeros((40, 60), dtype=np.uint8)
        page._canvas.set_preview(
            source,
            source,
            alpha,
            alpha,
            source_width=60,
            source_height=40,
        )
        page.show()
        self.app.processEvents()
        self.assertTrue(page._application_filter_installed)
        page._save_button.setFocus()
        press = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_Space,
            Qt.KeyboardModifier.NoModifier,
            " ",
        )
        with patch.object(page, "_cursor_is_over_canvas", return_value=True):
            handled = page.eventFilter(page._save_button, press)

        self.assertTrue(handled)
        self.assertTrue(page._canvas.space_pan_active)
        self.assertEqual(
            page._canvas.cursor().shape(),
            Qt.CursorShape.OpenHandCursor,
        )
        page.hide()
        self.app.processEvents()
        self.assertFalse(page._canvas.space_pan_active)
        self.assertFalse(page._application_filter_installed)
        page.deleteLater()

    def test_header_uses_shared_page_title_style(self) -> None:
        page = ImageLabPage()

        titles = [
            label
            for label in page.findChildren(QLabel)
            if label.property("role") == "pageTitle"
        ]
        self.assertEqual([label.text() for label in titles], ["图片实验室"])
        subtitles = [
            label
            for label in page.findChildren(QLabel)
            if label.text() == "整幅文献图片的非破坏背景清理与人工修补"
        ]
        self.assertEqual(len(subtitles), 1)
        self.assertEqual(subtitles[0].property("role"), "muted")
        page.deleteLater()

    def test_file_dialog_labels_are_chinese(self) -> None:
        page = ImageLabPage()

        open_dialog = page._create_file_dialog(
            "打开待处理图片",
            "图片文件 (*.png)",
            save=False,
        )
        self.assertEqual(
            open_dialog.labelText(QFileDialog.DialogLabel.Accept),
            "打开",
        )
        self.assertEqual(
            open_dialog.labelText(QFileDialog.DialogLabel.Reject),
            "取消",
        )

        save_dialog = page._create_file_dialog(
            "保存图片实验室项目",
            "图片实验室项目 (*.fontlab)",
            save=True,
        )
        self.assertEqual(
            save_dialog.labelText(QFileDialog.DialogLabel.Accept),
            "保存",
        )
        self.assertEqual(
            save_dialog.labelText(QFileDialog.DialogLabel.Reject),
            "取消",
        )
        open_dialog.deleteLater()
        save_dialog.deleteLater()
        page.deleteLater()

    def test_message_dialog_confirm_button_is_chinese(self) -> None:
        page = ImageLabPage()
        button_texts: list[str] = []

        def capture_dialog(dialog: QMessageBox) -> int:
            button_texts.extend(button.text() for button in dialog.buttons())
            return 0

        with patch.object(QMessageBox, "exec", capture_dialog):
            page._show_message(QMessageBox.Icon.Information, "提示", "测试消息")

        self.assertEqual(button_texts, ["确定"])
        page.deleteLater()

    def test_unsaved_project_dialog_buttons_are_chinese(self) -> None:
        page = ImageLabPage()
        page._dirty = True
        button_texts: set[str] = set()

        def cancel_dialog(dialog: QMessageBox) -> int:
            buttons = dialog.buttons()
            button_texts.update(button.text() for button in buttons)
            next(button for button in buttons if button.text() == "取消").click()
            return 0

        with patch.object(QMessageBox, "exec", cancel_dialog):
            should_replace = page._confirm_replace_project()

        self.assertFalse(should_replace)
        self.assertEqual(button_texts, {"保存并继续", "放弃修改", "取消"})
        page.deleteLater()


if __name__ == "__main__":
    unittest.main()
