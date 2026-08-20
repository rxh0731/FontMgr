"""导入核对扫描与确认状态回归测试。"""

from __future__ import annotations

import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication, QMessageBox

from ui.pages.import_page import ImportPage, ImportRunContext, ScanItem, ScanTask
from utils.file_utils import compute_file_md5


class ImportPageTests(unittest.TestCase):
    """验证重复预检不会复用旧状态，也不会阻塞有效批次。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_finished_pixel_size_updates_physical_millimeters(self) -> None:
        page = ImportPage()

        self.assertEqual(page._dpi_spin.value(), 300)
        self.assertAlmostEqual(page._width_mm_spin.value(), 21.17, places=2)
        self.assertAlmostEqual(page._height_mm_spin.value(), 21.17, places=2)

        page._width_px_spin.setValue(600)
        page._height_px_spin.setValue(300)

        self.assertAlmostEqual(page._width_mm_spin.value(), 50.80, places=2)
        self.assertAlmostEqual(page._height_mm_spin.value(), 25.40, places=2)
        page.deleteLater()

    def test_physical_millimeters_update_corresponding_pixel_size(self) -> None:
        page = ImportPage()

        page._width_mm_spin.setValue(25.40)
        page._height_mm_spin.setValue(50.80)

        self.assertEqual(page._width_px_spin.value(), 300)
        self.assertEqual(page._height_px_spin.value(), 600)
        page.deleteLater()

    def test_dpi_change_keeps_pixels_and_recalculates_millimeters(self) -> None:
        page = ImportPage()
        page._width_px_spin.setValue(600)
        page._height_px_spin.setValue(300)

        page._dpi_spin.setValue(600)

        self.assertEqual(page._width_px_spin.value(), 600)
        self.assertEqual(page._height_px_spin.value(), 300)
        self.assertAlmostEqual(page._width_mm_spin.value(), 25.40, places=2)
        self.assertAlmostEqual(page._height_mm_spin.value(), 12.70, places=2)
        page.deleteLater()

    def test_same_batch_ambiguous_duplicate_can_be_confirmed_and_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "爱-1.png"
            duplicate_path = Path(directory) / "爱-2.png"
            Image.new("L", (4, 4), 128).save(first_path)
            duplicate_path.write_bytes(first_path.read_bytes())

            items = self._run_scan([str(first_path), str(duplicate_path)])

        self.assertEqual(items[0].category, "一对一")
        self.assertEqual(items[1].category, "重复")
        self.assertEqual(items[1].duplicate_filename, first_path.name)
        self.assertEqual(items[1].final_char, "")
        self.assertIsInstance(items[0].preview_image, QImage)
        self.assertFalse(items[0].preview_image.isNull())
        self.assertIsInstance(items[1].duplicate_preview_image, QImage)
        self.assertFalse(items[1].duplicate_preview_image.isNull())

        page = ImportPage()
        page._scan_items = items
        page._confirm_check.setChecked(True)
        items[0].final_char = "愛"
        self.assertTrue(page._validate_confirmations())
        page.deleteLater()

    def test_existing_duplicate_retains_comparison_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            existing_path = Path(directory) / "已有.png"
            incoming_path = Path(directory) / "未分类-副本.png"
            Image.new("L", (4, 4), 200).save(existing_path)
            incoming_path.write_bytes(existing_path.read_bytes())
            digest = compute_file_md5(str(existing_path))

            with patch(
                "ui.pages.import_page.identify_character",
                return_value=("正确", ("未分类",)),
            ):
                items = self._run_scan(
                    [str(incoming_path)],
                    {digest: (str(existing_path), existing_path.name)},
                )

        self.assertEqual(items[0].category, "重复")
        self.assertEqual(items[0].duplicate_path, str(existing_path))
        self.assertEqual(items[0].duplicate_filename, existing_path.name)

    def test_clearing_scan_results_resets_exception_confirmation(self) -> None:
        page = ImportPage()
        page._scan_items = [
            ScanItem("路径", "文件.png", "测", "重复", ("测",), "测", True)
        ]
        page._confirm_check.setVisible(True)
        page._confirm_check.setChecked(True)

        page._clear_scan_results()

        self.assertEqual(page._scan_items, [])
        self.assertFalse(page._confirm_check.isChecked())
        self.assertFalse(page._confirm_check.isVisible())
        self.assertFalse(page._import_button.isEnabled())
        page.deleteLater()

    def test_scan_finish_populates_cards_in_bounded_batches(self) -> None:
        """扫描 N/N 后应进入可重绘的列表生成阶段，不能同步阻塞到全部完成。"""
        page = ImportPage()
        page.POPULATION_BATCH_LIMIT = 1
        page.POPULATION_TIME_SLICE_SECONDS = 10.0
        items = [
            self._make_item(1, "正确"),
            self._make_item(2, "正确"),
            self._make_item(3, "正确"),
        ]

        page._scan_finished(items)

        self.assertTrue(page._population_active)
        self.assertFalse(page._progress_bar.isHidden())
        self.assertFalse(page._import_button.isEnabled())
        self.assertEqual(page._population_index, 0)

        page._population_timer.stop()
        page._populate_table_batch()
        page._population_timer.stop()
        self.assertTrue(page._population_prepared)
        self.assertEqual(page._population_index, 0)

        page._populate_table_batch()
        page._population_timer.stop()
        self.assertEqual(page._population_index, 1)
        self.assertTrue(page._population_active)
        self.assertFalse(page._progress_bar.isHidden())

        self._drain_population(page)

        self.assertFalse(page._population_active)
        self.assertTrue(page._progress_bar.isHidden())
        self.assertEqual(page._column_layouts["正确"].count() - 1, 3)
        self.assertEqual(page._status_label.text(), "已扫描 3 张图片")
        self.assertTrue(page._import_button.isEnabled())
        page.deleteLater()

    def test_population_interleaves_columns_and_cancel_discards_partial_scan(self) -> None:
        """大批建卡先覆盖三栏；取消后旧定时器不得继续插入卡片。"""
        page = ImportPage()
        items = [
            self._make_item(1, "正确"),
            self._make_item(2, "正确"),
            self._make_item(3, "一对一", final_char="测"),
            self._make_item(4, "重复"),
        ]
        page._scan_finished(items)
        page._population_timer.stop()
        page._populate_table_batch()
        page._population_timer.stop()

        self.assertEqual(
            [category for _item, category in page._population_queue],
            ["正确", "一对一", "异常", "正确"],
        )

        page.cancel_task()

        self.assertFalse(page._population_active)
        self.assertEqual(page._scan_items, [])
        self.assertTrue(page._progress_bar.isHidden())
        self.assertFalse(page._import_button.isEnabled())
        self.assertEqual(page._status_label.text(), "校对列表生成已取消，请重新扫描")
        self.assertTrue(all(layout.count() == 1 for layout in page._column_layouts.values()))
        page.deleteLater()

    def test_search_reuses_scanned_preview_without_gui_file_decode(self) -> None:
        """搜索重建卡片时复用 QImage，不再从主线程逐张打开源图。"""
        page = ImportPage()
        page.POPULATION_BATCH_LIMIT = 1
        item = self._make_item(1, "正确")
        page._scan_finished([item])
        self._drain_population(page)
        page._search_edit.setText("测")

        with patch.object(
            ImportPage,
            "_load_preview_pixmap",
            side_effect=AssertionError("不应回退到 GUI 文件解码"),
        ):
            page._populate_tables()
            page._population_timer.stop()
            page._populate_table_batch()
            page._population_timer.stop()
            page.cancel_task()

            self.assertFalse(page._cancel_event.is_set())
            self.assertEqual(page._status_label.text(), "校对列表刷新已取消，可重新搜索")

            page._populate_tables()
            self._drain_population(page)

        self.assertEqual(page._column_layouts["正确"].count() - 1, 1)
        self.assertIn("已显示 1 项搜索结果", page._status_label.text())
        page.deleteLater()

    def test_late_scan_result_cannot_replace_newer_page_state(self) -> None:
        """返回或重扫后的旧工作线程结果不能覆盖当前扫描代次。"""
        page = ImportPage()
        current = self._make_item(1, "正确")
        late = self._make_item(2, "正确")
        page._scan_items = [current]
        page._scan_generation = 8

        page._scan_finished([late], generation=7)

        self.assertEqual(page._scan_items, [current])
        self.assertFalse(page._population_active)
        page.deleteLater()

    def test_population_failure_restores_controls_and_clears_partial_cards(self) -> None:
        """单张卡片创建异常不能让页面永久停在忙碌状态。"""
        page = ImportPage()
        page._scan_finished([self._make_item(1, "正确")])
        page._population_timer.stop()
        page._populate_table_batch()
        page._population_timer.stop()

        with (
            patch.object(page, "_create_scan_card", side_effect=RuntimeError("模拟建卡失败")),
            patch("ui.pages.import_page.QMessageBox.warning") as warning,
        ):
            page._populate_table_batch()

        self.assertFalse(page._population_active)
        self.assertTrue(page._progress_bar.isHidden())
        self.assertTrue(page._scan_button.isEnabled())
        self.assertFalse(page._import_button.isEnabled())
        self.assertEqual(page._scan_items, [])
        self.assertEqual(page._status_label.text(), "校对列表生成失败")
        self.assertIn("模拟建卡失败", warning.call_args.args[2])
        page.deleteLater()

    def test_thread_pool_scan_delivers_qimage_and_finishes_population(self) -> None:
        """真实线程池信号可以安全传递 QImage，并完成第二阶段列表生成。"""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "测-0001.png"
            Image.new("L", (32, 32), 80).save(source)
            page = ImportPage()
            page._directory_edit.setText(directory)
            loop = QEventLoop()
            poll = QTimer()
            poll.setInterval(5)
            poll.timeout.connect(
                lambda: loop.quit()
                if page._scan_items and not page._population_active and page._progress_bar.isHidden()
                else None
            )
            timeout = QTimer()
            timeout.setSingleShot(True)
            timeout.timeout.connect(loop.quit)

            page.start_scan()
            poll.start()
            timeout.start(5000)
            loop.exec()

            self.assertTrue(timeout.isActive(), "线程池扫描或列表生成超时")
            self.assertEqual(len(page._scan_items), 1)
            self.assertIsInstance(page._scan_items[0].preview_image, QImage)
            self.assertFalse(page._scan_items[0].preview_image.isNull())
            self.assertEqual(
                sum(layout.count() - 1 for layout in page._column_layouts.values()),
                1,
            )
            self.assertEqual(page._status_label.text(), "已扫描 1 张图片")
            timeout.stop()
            poll.stop()
            page.deleteLater()

    def test_return_home_during_scan_resets_busy_state_before_reentry(self) -> None:
        """扫描中返回首页必须立即恢复复用页面的进度和控件状态。"""
        page = ImportPage()
        page._active_task_kind = "scan"
        page._set_busy(True, 10, "正在扫描图片")
        home_requests: list[bool] = []
        page.home_requested.connect(lambda: home_requests.append(True))

        with patch(
            "ui.pages.import_page.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Yes,
        ):
            page._request_home()

        self.assertEqual(home_requests, [True])
        self.assertTrue(page._cancel_event.is_set())
        self.assertIsNone(page._active_task_kind)
        self.assertTrue(page._progress_bar.isHidden())
        self.assertTrue(page._scan_button.isEnabled())
        self.assertTrue(page._directory_edit.isEnabled())
        page.configure_create([])
        self.assertTrue(page._progress_bar.isHidden())
        self.assertTrue(page._scan_button.isEnabled())
        page.deleteLater()

    def test_return_home_during_import_waits_for_safe_task_completion(self) -> None:
        """导入写盘期间只请求停止，后台安全收尾后才真正返回首页。"""
        page = ImportPage()
        context = ImportRunContext(
            token=4,
            append_mode=True,
            library_name="测试库",
            target_directory="",
            cancel_event=threading.Event(),
            directory_created=threading.Event(),
        )
        page._active_import_token = context.token
        page._active_task_kind = "import"
        page._cancel_event = context.cancel_event
        page._set_busy(True, 10, "正在导入字图")
        home_requests: list[bool] = []
        page.home_requested.connect(lambda: home_requests.append(True))

        with patch(
            "ui.pages.import_page.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Yes,
        ):
            page._request_home()

        self.assertTrue(context.cancel_event.is_set())
        self.assertEqual(home_requests, [])
        self.assertTrue(page._return_home_after_import)
        self.assertFalse(page._progress_bar.isHidden())

        with patch("ui.pages.import_page.QMessageBox.information"):
            page._import_finished(
                {"已取消": True, "成功": 2, "跳过": 0, "失败": 0, "字库路径": "路径"},
                context,
            )

        self.assertEqual(home_requests, [True])
        self.assertIsNone(page._active_import_token)
        self.assertIsNone(page._active_task_kind)
        self.assertTrue(page._progress_bar.isHidden())
        page.deleteLater()

    def test_late_import_failure_cannot_delete_new_task_directory(self) -> None:
        """旧导入失败信号不得按新任务状态清目录或解锁新页面。"""
        with tempfile.TemporaryDirectory() as root:
            new_directory = Path(root) / "新任务"
            new_directory.mkdir()
            marker = new_directory / "保留.txt"
            marker.write_text("保留", encoding="utf-8")
            created = threading.Event()
            created.set()
            late_context = ImportRunContext(
                token=7,
                append_mode=False,
                library_name="旧任务",
                target_directory=str(new_directory),
                cancel_event=threading.Event(),
                directory_created=created,
            )
            page = ImportPage()
            page._active_import_token = 8
            page._active_task_kind = "import"
            page._set_busy(True, 10, "正在导入字图")

            with (
                patch("ui.pages.import_page.config.ZIKU_ROOT", root),
                patch("ui.pages.import_page.QMessageBox.warning") as warning,
            ):
                page._import_failed("迟到失败", late_context)

            self.assertTrue(marker.exists())
            self.assertEqual(page._active_import_token, 8)
            self.assertEqual(page._active_task_kind, "import")
            self.assertFalse(page._progress_bar.isHidden())
            warning.assert_not_called()
            page._set_busy(False)
            page.deleteLater()

    @staticmethod
    def _run_scan(
        paths: list[str],
        existing_files: dict[str, tuple[str, str]] | None = None,
    ) -> list[ScanItem]:
        results: list[list[ScanItem]] = []
        failures: list[str] = []
        task = ScanTask(paths, existing_files or {}, threading.Event())
        task.signals.finished.connect(results.append)
        task.signals.failed.connect(failures.append)
        task.run()
        if failures:
            raise AssertionError(failures[0])
        if not results:
            raise AssertionError("扫描任务未返回结果")
        return results[0]

    @staticmethod
    def _make_item(
        index: int,
        category: str,
        *,
        final_char: str = "测",
    ) -> ScanItem:
        preview = QImage(8, 8, QImage.Format.Format_RGBA8888)
        preview.fill(QColor("black"))
        return ScanItem(
            path=f"不存在-{index}.png",
            filename=f"测-{index:04d}.png",
            original_char="测",
            category=category,
            candidates=("测",),
            final_char=final_char,
            confirmed=True,
            issue="重复，将跳过！" if category == "重复" else "",
            duplicate_path=f"已有-{index}.png" if category == "重复" else "",
            duplicate_filename=f"已有-{index}.png" if category == "重复" else "",
            preview_image=preview,
            duplicate_preview_image=preview if category == "重复" else None,
        )

    @staticmethod
    def _drain_population(page: ImportPage) -> None:
        guard = 0
        while page._population_active:
            guard += 1
            if guard > 100:
                raise AssertionError("列表生成未能结束")
            page._population_timer.stop()
            page._populate_table_batch()


if __name__ == "__main__":
    unittest.main()
