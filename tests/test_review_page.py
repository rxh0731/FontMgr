"""手工审核工作台布局与流程回归测试。"""

from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, call, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPoint, QSize, Qt
from PySide6.QtGui import QColor, QCursor, QImage, QKeyEvent, QPainter
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QHeaderView,
    QAbstractSlider,
    QLabel,
    QMessageBox,
    QSlider,
    QStackedWidget,
    QTreeWidget,
    QWidget,
)

import config
from data.library_database import LibraryDatabase
from services.batch_persistence import (
    JOURNAL_FILENAME,
    BatchJournalUncertainError,
    BatchPersistenceSession,
)
from services.glyph_service import GlyphService
from services.workflow_status_service import (
    MARKER_FILE_ERROR,
    MARKER_UNSAVED,
    PHASE_FILTER_ALL,
    REVIEW_STATUS_FILTERS,
    STAGE_PENDING_REVIEW,
    STATUS_REVIEWED,
)
import ui.pages.review_page as review_page_module
from ui.pages.review_page import ReviewPage
from ui.widgets.review_canvas import ReviewCanvas
from utils.file_utils import safe_read_json


class ReviewPageTests(unittest.TestCase):
    """验证新版三栏页面只处理审核阶段稿件并保持保存契约。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    @classmethod
    def tearDownClass(cls) -> None:
        """销毁页面遗留的快捷键，避免污染后续画布测试。"""
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        cls.app.processEvents()

    def test_workspace_defaults_to_transform_and_white_canvas(self) -> None:
        page = ReviewPage()
        page.resize(1100, 720)
        page.show()
        self.app.processEvents()

        self.assertEqual(page._main_splitter.count(), 3)
        self.assertLessEqual(page.minimumSizeHint().width(), 1100)
        self.assertGreaterEqual(
            page._toolbar_widget.width(),
            page._toolbar_widget.minimumSizeHint().width(),
        )
        self.assertIsInstance(page._item_tree, QTreeWidget)
        self.assertIsInstance(page._parameters_stack, QStackedWidget)
        self.assertEqual(page._canvas.tool, ReviewCanvas.TOOL_TRANSFORM)
        self.assertTrue(page._tool_buttons[ReviewCanvas.TOOL_TRANSFORM].isChecked())
        self.assertEqual(page._canvas.background_mode, ReviewCanvas.BACKGROUND_WHITE)
        self.assertTrue(page._canvas.grid_visible)
        self.assertTrue(page._pressure_checkbox.isChecked())
        self.assertTrue(page._canvas.pressure_enabled)
        self.assertEqual(page._minimum_pressure_slider.value(), 20)
        self.assertAlmostEqual(page._canvas.minimum_pressure_ratio, 0.2)
        self.assertEqual(page._save_button.text(), "保存修改稿")
        self.assertEqual(page._approve_button.text(), "保存并审核通过")
        self.assertEqual(page._previous_button.text(), "上一条")
        self.assertEqual(page._next_button.text(), "下一条")
        for button in (page._previous_button, page._next_button):
            self.assertGreaterEqual(
                button.width(),
                button.fontMetrics().horizontalAdvance(button.text()) + 42,
            )
        self.assertEqual(page._complete_button.text(), "批量手工审核")
        self.assertFalse(page._complete_button.isEnabled())
        self.assertTrue(page._batch_progress_widget.isHidden())
        page.deleteLater()

    def test_bulk_review_includes_missing_sources_and_reports_live_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, variants = self._build_library(Path(directory))
            page = ReviewPage()
            self.assertTrue(page.open_library(service.ziku_dir))
            self.assertTrue(page._complete_button.isEnabled())
            self.assertEqual(
                page._pending_review_ids(),
                [variants["pending"], variants["missing_preview"]],
            )

            main_thread_id = threading.get_ident()
            worker_threads: list[int] = []
            original_save = review_page_module._save_and_approve_review

            def tracked_save(*args, **kwargs):
                worker_threads.append(threading.get_ident())
                return original_save(*args, **kwargs)

            with (
                patch.object(
                    review_page_module,
                    "_save_and_approve_review",
                    side_effect=tracked_save,
                ),
                patch("ui.pages.review_page.QMessageBox.warning") as warning,
                patch("ui.pages.review_page.QMessageBox.information") as information,
            ):
                page._start_bulk_review(page._pending_review_ids())
                self.assertTrue(page._batch_running)
                self.assertFalse(page._main_splitter.isEnabled())
                self.assertFalse(page._home_button.isEnabled())
                self.assertFalse(page._complete_button.isEnabled())
                self.assertTrue(page._shortcut_actions)
                self.assertTrue(
                    all(not action.isEnabled() for action in page._shortcut_actions)
                )
                self.assertFalse(page._batch_progress_widget.isHidden())

                deadline = time.monotonic() + 5.0
                while page._batch_running and time.monotonic() < deadline:
                    self.app.processEvents()
                    time.sleep(0.01)
                self.app.processEvents()

            self.assertFalse(page._batch_running)
            self.assertTrue(worker_threads)
            self.assertTrue(all(thread_id != main_thread_id for thread_id in worker_threads))
            self.assertEqual(page._batch_progress_bar.value(), 100)
            self.assertIn("2 / 2", page._batch_progress_label.text())
            self.assertTrue(page._batch_progress_widget.isHidden())
            self.assertTrue(page._main_splitter.isEnabled())
            self.assertTrue(page._home_button.isEnabled())
            self.assertTrue(page._complete_button.isEnabled())
            self.assertTrue(all(action.isEnabled() for action in page._shortcut_actions))

            reloaded = GlyphService(service.ziku_name, service.ziku_dir)
            approved = reloaded.get_variant(str(variants["pending"]))
            failed = reloaded.get_variant(str(variants["missing_preview"]))
            self.assertEqual(approved["状态"], config.STATUS_REVIEWED)
            self.assertFalse(approved["手工编辑"]["已编辑"])
            self.assertTrue(approved["审核文件"])
            self.assertTrue(
                (Path(reloaded.get_workflow_dirs()["手工审核"]) / approved["审核文件"]).is_file()
            )
            self.assertEqual(failed["状态"], config.STATUS_PENDING_MANUAL_REVIEW)
            self.assertFalse(failed.get("审核文件", ""))
            self.assertEqual(
                reloaded.get_variant(str(variants["reviewed"]))["状态"],
                config.STATUS_REVIEWED,
            )
            self.assertEqual(
                reloaded.get_variant(str(variants["finished"]))["状态"],
                config.STATUS_FINISHED,
            )
            self.assertEqual(
                reloaded.get_variant(str(variants["pending_optimization"]))["状态"],
                config.STATUS_PENDING_OPTIMIZATION,
            )
            warning.assert_called_once()
            self.assertIn("成功 1 个，跳过 0 个，失败 1 个", warning.call_args.args[2])
            self.assertIn(str(variants["missing_preview"]), warning.call_args.args[2])
            self.assertIn("总耗时：", warning.call_args.args[2])
            information.assert_not_called()
            page.deleteLater()

    def test_successful_bulk_review_restores_navigation_editing_and_shortcuts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, variants = self._build_library(Path(directory))
            page = ReviewPage()
            pending_id = str(variants["pending"])
            self.assertTrue(page.open_library(service.ziku_dir, pending_id))

            with (
                patch("ui.pages.review_page.QMessageBox.information") as information,
                patch("ui.pages.review_page.QMessageBox.warning") as warning,
                patch("ui.pages.review_page.QMessageBox.critical") as critical,
            ):
                page._start_bulk_review([pending_id])
                deadline = time.monotonic() + 5.0
                while page._batch_running and time.monotonic() < deadline:
                    self.app.processEvents()
                    time.sleep(0.01)
                self.app.processEvents()

            self.assertFalse(page._batch_running)
            self.assertIsNone(page._batch_worker)
            self.assertTrue(page._home_button.isEnabled())
            self.assertTrue(page._complete_button.isEnabled())
            self.assertTrue(page._main_splitter.isEnabled())
            self.assertTrue(page._toolbar_widget.isEnabled())
            self.assertTrue(page._save_button.isEnabled())
            self.assertTrue(page._approve_button.isEnabled())
            self.assertTrue(all(action.isEnabled() for action in page._shortcut_actions))
            self.assertTrue(page._batch_progress_widget.isHidden())
            information.assert_called_once()
            self.assertIn("总耗时：", information.call_args.args[2])
            warning.assert_not_called()
            critical.assert_not_called()
            page.deleteLater()

    def test_bulk_confirmation_explains_risks_and_defaults_to_cancel(self) -> None:
        page = ReviewPage()
        with patch.object(
            QMessageBox,
            "exec",
            autospec=True,
            return_value=QMessageBox.StandardButton.Cancel.value,
        ):
            self.assertFalse(page._confirm_bulk_review(37))

        dialogs = page.findChildren(QMessageBox)
        self.assertTrue(dialogs)
        dialog = dialogs[-1]
        ok_button = dialog.button(QMessageBox.StandardButton.Ok)
        cancel_button = dialog.button(QMessageBox.StandardButton.Cancel)
        self.assertEqual(
            dialog.standardButtons(),
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
        )
        self.assertEqual(ok_button.text(), "确定")
        self.assertEqual(cancel_button.text(), "取消")
        self.assertIs(dialog.defaultButton(), cancel_button)
        self.assertIs(dialog.escapeButton(), cancel_button)
        self.assertIn("37", dialog.text())
        self.assertIn("不能替代人工判断", dialog.informativeText())
        self.assertIn("字形缺损", dialog.informativeText())

        with patch.object(QMessageBox, "exec", autospec=True, return_value=0):
            self.assertFalse(page._confirm_bulk_review(37))
        page.deleteLater()

    def test_bulk_stop_before_first_item_restores_controls_and_keeps_pending(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, variants = self._build_library(Path(directory))
            page = ReviewPage()
            pending_id = str(variants["pending"])
            self.assertTrue(page.open_library(service.ziku_dir, pending_id))
            started: list[object] = []
            with patch.object(page._batch_pool, "start", side_effect=started.append):
                page._start_bulk_review([pending_id])

            worker = started[0]
            with patch.object(page, "_confirm_stop_bulk_review", return_value=False):
                page._request_stop_bulk_review()
            self.assertFalse(worker.is_cancel_requested())

            with patch.object(page, "_confirm_stop_bulk_review", return_value=True):
                page._request_stop_bulk_review()
            self.assertTrue(worker.is_cancel_requested())
            self.assertFalse(page._stop_batch_button.isEnabled())
            self.assertEqual(page._stop_batch_button.text(), "正在停止…")
            self.assertIn("正在停止批量审核", page._batch_progress_label.text())
            page._bulk_review_progress(
                {"当前": 1, "已处理": 0, "总数": 1, "字形": "迟到进度"}
            )
            self.assertNotIn("迟到进度", page._batch_progress_label.text())

            with patch("ui.pages.review_page.QMessageBox.information") as information:
                worker.run()
                self.app.processEvents()

            self.assertFalse(page._batch_running)
            self.assertIsNone(page._batch_worker)
            self.assertTrue(page._home_button.isEnabled())
            self.assertTrue(page._complete_button.isEnabled())
            self.assertTrue(page._main_splitter.isEnabled())
            self.assertTrue(all(action.isEnabled() for action in page._shortcut_actions))
            self.assertTrue(page._batch_progress_widget.isHidden())
            self.assertEqual(page._stop_batch_button.text(), "停止批量审核")
            information.assert_called_once()
            self.assertEqual(information.call_args.args[1], "手工审核已停止")
            self.assertIn("未处理 1 个", information.call_args.args[2])
            self.assertIn("总耗时：", information.call_args.args[2])
            reloaded = GlyphService(service.ziku_name, service.ziku_dir)
            self.assertEqual(
                reloaded.get_variant(pending_id)["状态"],
                config.STATUS_PENDING_MANUAL_REVIEW,
            )
            page.deleteLater()

    def test_bulk_stop_after_current_transaction_keeps_success_and_leaves_rest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, variants = self._build_library(Path(directory))
            directories = service.get_workflow_dirs()
            second_id = service.add_original(
                "次",
                "次-0001.png",
                "次-0001.png",
                "md5-second-pending",
            )
            self._write_glyph(Path(directories["优化预览"]) / "次-0001.png")
            service.update_variant(
                second_id,
                **{
                    "状态": config.STATUS_PENDING_MANUAL_REVIEW,
                    "中间文件": "次-0001.png",
                },
            )
            service.save()
            stop_event = threading.Event()
            original_save = review_page_module._save_and_approve_review
            original_service_save = GlyphService.save
            save_calls: list[str] = []

            def save_then_stop(*args, **kwargs):
                saved = original_save(*args, **kwargs)
                stop_event.set()
                return saved

            def counted_service_save(glyph: GlyphService) -> None:
                save_calls.append(glyph.ziku_name)
                original_service_save(glyph)

            def create_persistence(glyph: GlyphService) -> BatchPersistenceSession:
                return BatchPersistenceSession(
                    glyph,
                    checkpoint_items=100,
                    checkpoint_seconds=3600.0,
                )

            with (
                patch.object(
                    review_page_module,
                    "BatchPersistenceSession",
                    side_effect=create_persistence,
                ),
                patch.object(GlyphService, "save", new=counted_service_save),
                patch.object(
                    review_page_module,
                    "_save_and_approve_review",
                    side_effect=save_then_stop,
                ) as save,
            ):
                result = review_page_module._run_bulk_review(
                    service.ziku_name,
                    service.ziku_dir,
                    [str(variants["pending"]), second_id],
                    lambda _progress: None,
                    stop_event.is_set,
                )

            self.assertTrue(result["已停止"])
            self.assertEqual(result["成功"], 1)
            self.assertEqual(result["失败"], 0)
            self.assertEqual(result["未处理"], 1)
            save.assert_called_once()
            self.assertEqual(save_calls, [service.ziku_name])
            self.assertFalse((Path(service.ziku_dir) / JOURNAL_FILENAME).exists())
            reloaded = GlyphService(service.ziku_name, service.ziku_dir)
            self.assertEqual(
                reloaded.get_variant(str(variants["pending"]))["状态"],
                config.STATUS_REVIEWED,
            )
            self.assertEqual(
                reloaded.get_variant(second_id)["状态"],
                config.STATUS_PENDING_MANUAL_REVIEW,
            )

    def test_bulk_review_commits_each_variant_to_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, variant_ids = self._build_large_library(
                Path(directory),
                total=6,
            )
            original_service_save = GlyphService.save
            save_calls: list[str] = []

            def counted_service_save(glyph: GlyphService) -> None:
                save_calls.append(glyph.ziku_name)
                original_service_save(glyph)

            def create_persistence(glyph: GlyphService) -> BatchPersistenceSession:
                return BatchPersistenceSession(
                    glyph,
                    checkpoint_items=100,
                    checkpoint_seconds=3600.0,
                )

            with (
                patch.object(
                    review_page_module,
                    "BatchPersistenceSession",
                    side_effect=create_persistence,
                ),
                patch.object(GlyphService, "save", new=counted_service_save),
            ):
                result = review_page_module._run_bulk_review(
                    service.ziku_name,
                    service.ziku_dir,
                    variant_ids,
                    lambda _progress: None,
                )

            self.assertEqual(result["成功"], len(variant_ids))
            self.assertEqual(result["失败"], 0)
            self.assertEqual(save_calls, [service.ziku_name] * len(variant_ids))
            self.assertFalse((Path(service.ziku_dir) / JOURNAL_FILENAME).exists())
            saved = LibraryDatabase.open(service.ziku_dir).load_data()
            self.assertTrue(
                all(
                    saved["变体详情"][variant_id]["状态"]
                    == config.STATUS_REVIEWED
                    for variant_id in variant_ids
                )
            )

    def test_review_batch_commits_without_json_journal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, variants = self._build_library(Path(directory))
            variant_id = str(variants["pending"])
            journal_path = Path(service.ziku_dir) / JOURNAL_FILENAME
            session = BatchPersistenceSession(
                service,
                checkpoint_items=100,
                checkpoint_seconds=3600.0,
            )

            try:
                with (
                    patch.object(
                        service,
                        "snapshot_state",
                        side_effect=AssertionError("批量单字不应复制整库状态"),
                    ),
                ):
                    review_page_module._save_and_approve_review(
                        service,
                        variant_id,
                        (64, 64),
                        300,
                        persistence=session,
                    )
            finally:
                session.leave_for_recovery()

            self.assertFalse(journal_path.exists())
            self.assertEqual(
                service.get_variant(variant_id)["状态"],
                config.STATUS_REVIEWED,
            )
            self.assertEqual(
                LibraryDatabase.open(service.ziku_dir).load_data()["变体详情"][variant_id]["状态"],
                config.STATUS_REVIEWED,
            )
            self.assertFalse(
                (Path(service.ziku_dir) / ".fonteditor_file_transactions").exists()
            )

            recovered = GlyphService(service.ziku_name, service.ziku_dir)

            recovered_detail = recovered.get_variant(variant_id)
            self.assertEqual(recovered_detail["状态"], config.STATUS_REVIEWED)
            self.assertTrue(recovered_detail["审核文件"])
            self.assertTrue(
                (
                    Path(recovered.get_workflow_dirs()["手工审核"])
                    / recovered_detail["审核文件"]
                ).is_file()
            )
            self.assertFalse(journal_path.exists())

    def test_uncertain_review_journal_keeps_new_review_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, variants = self._build_library(Path(directory))
            variant_id = str(variants["pending"])
            persistence = MagicMock()
            persistence.record_variant.side_effect = BatchJournalUncertainError(
                "模拟审核日志提交结果未知"
            )

            with self.assertRaisesRegex(
                BatchJournalUncertainError,
                "提交结果未知",
            ):
                review_page_module._save_and_approve_review(
                    service,
                    variant_id,
                    (64, 64),
                    300,
                    persistence=persistence,
                )

            detail = service.get_variant(variant_id)
            self.assertEqual(detail["状态"], config.STATUS_REVIEWED)
            review_path = (
                Path(service.get_workflow_dirs()["手工审核"])
                / str(detail["审核文件"])
            )
            self.assertTrue(review_path.is_file())

    def test_interactive_review_rejects_library_locked_by_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, variants = self._build_library(Path(directory))
            variant_id = str(variants["pending"])
            session = BatchPersistenceSession(service)
            try:
                with self.assertRaisesRegex(RuntimeError, "正在执行其他批处理任务"):
                    review_page_module._save_and_approve_review(
                        service,
                        variant_id,
                        (64, 64),
                        300,
                    )
            finally:
                session.finish()

            self.assertEqual(
                service.get_variant(variant_id)["状态"],
                config.STATUS_PENDING_MANUAL_REVIEW,
            )

    def test_missing_recorded_review_file_never_falls_back_to_preview(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, variants = self._build_library(Path(directory))
            variant_id = str(variants["pending"])
            service.update_variant(variant_id, **{"审核文件": "丢失审核稿.png"})
            service.save()

            with (
                patch.object(review_page_module.FileTransaction, "begin") as begin,
                self.assertRaisesRegex(FileNotFoundError, "人工修订稿不可用"),
            ):
                review_page_module._save_and_approve_review(
                    service,
                    variant_id,
                    (64, 64),
                    300,
                )

            begin.assert_not_called()
            self.assertEqual(
                service.get_variant(variant_id)["状态"],
                config.STATUS_PENDING_MANUAL_REVIEW,
            )

            page = ReviewPage()
            page.open_library(service.ziku_dir, variant_id)
            record = page._records_by_id[variant_id]
            node = page._node_for_variant(variant_id)
            self.assertEqual(record["source_path"], "")
            self.assertEqual(record["stage"], "手工审核稿")
            self.assertIsNotNone(node)
            self.assertIn(MARKER_FILE_ERROR, node.text(1).splitlines()[1])
            self.assertFalse(page._canvas.has_image)
            page.deleteLater()

    def test_missing_recorded_finished_file_still_uses_transaction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, variants = self._build_library(Path(directory))
            variant_id = str(variants["pending"])
            service.update_variant(variant_id, **{"成品文件": "丢失成品.png"})
            service.save()

            original_begin = review_page_module.FileTransaction.begin
            with patch.object(
                review_page_module.FileTransaction,
                "begin",
                wraps=original_begin,
            ) as begin_transaction:
                review_page_module._save_and_approve_review(
                    service,
                    variant_id,
                    (64, 64),
                    300,
                )

            begin_transaction.assert_called_once()
            saved_detail = service.get_variant(variant_id)
            self.assertEqual(saved_detail["状态"], config.STATUS_REVIEWED)
            self.assertTrue(saved_detail["审核文件"])
            self.assertFalse(
                (
                    Path(service.ziku_dir)
                    / ".fonteditor_file_transactions"
                ).exists()
            )

    def test_bulk_database_failure_keeps_image_transaction_for_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, variants = self._build_library(Path(directory))
            variant_id = str(variants["pending"])
            journal_path = Path(service.ziku_dir) / JOURNAL_FILENAME
            save_calls: list[str] = []

            def create_persistence(glyph: GlyphService) -> BatchPersistenceSession:
                return BatchPersistenceSession(
                    glyph,
                    checkpoint_items=1,
                    checkpoint_seconds=3600.0,
                )

            def fail_checkpoint(glyph: GlyphService) -> None:
                save_calls.append(glyph.ziku_name)
                raise OSError("模拟批量检查点保存失败")

            with (
                patch.object(
                    review_page_module,
                    "BatchPersistenceSession",
                    side_effect=create_persistence,
                ),
                patch.object(GlyphService, "save", new=fail_checkpoint),
                self.assertRaisesRegex(BatchJournalUncertainError, "数据库提交结果无法确认"),
            ):
                review_page_module._run_bulk_review(
                    service.ziku_name,
                    service.ziku_dir,
                    [variant_id],
                    lambda _progress: None,
                )

            self.assertEqual(save_calls, [service.ziku_name])
            self.assertFalse(journal_path.exists())
            self.assertEqual(
                LibraryDatabase.open(service.ziku_dir).load_data()["变体详情"][variant_id]["状态"],
                config.STATUS_PENDING_MANUAL_REVIEW,
            )

            recovered = GlyphService(service.ziku_name, service.ziku_dir)

            self.assertEqual(
                recovered.get_variant(variant_id)["状态"],
                config.STATUS_REVIEWED,
            )
            self.assertFalse(journal_path.exists())

    def test_uncertain_review_journal_stops_batch_before_next_variant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, variants = self._build_library(Path(directory))
            pending_id = str(variants["pending"])
            with patch.object(
                review_page_module,
                "_save_and_approve_review",
                side_effect=BatchJournalUncertainError("模拟审核日志提交结果未知"),
            ) as save_review:
                with self.assertRaisesRegex(
                    BatchJournalUncertainError,
                    "提交结果未知",
                ):
                    review_page_module._run_bulk_review(
                        service.ziku_name,
                        service.ziku_dir,
                        [pending_id, pending_id],
                        lambda _payload: None,
                    )

            save_review.assert_called_once()
            # 异常收尾必须释放同库锁，允许后续恢复或重试。
            retry_session = BatchPersistenceSession(service)
            retry_session.finish()

    def test_bulk_database_failure_is_recovered_on_next_open(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, variants = self._build_library(Path(directory))
            variant_id = str(variants["pending"])
            journal_path = Path(service.ziku_dir) / JOURNAL_FILENAME
            save_calls: list[str] = []

            def create_persistence(glyph: GlyphService) -> BatchPersistenceSession:
                return BatchPersistenceSession(
                    glyph,
                    checkpoint_items=100,
                    checkpoint_seconds=3600.0,
                )

            def fail_finish(glyph: GlyphService) -> None:
                save_calls.append(glyph.ziku_name)
                raise OSError("模拟批量结束保存失败")

            with (
                patch.object(
                    review_page_module,
                    "BatchPersistenceSession",
                    side_effect=create_persistence,
                ),
                patch.object(GlyphService, "save", new=fail_finish),
                self.assertRaisesRegex(BatchJournalUncertainError, "数据库提交结果无法确认"),
            ):
                review_page_module._run_bulk_review(
                    service.ziku_name,
                    service.ziku_dir,
                    [variant_id],
                    lambda _progress: None,
                )

            self.assertEqual(save_calls, [service.ziku_name])
            self.assertFalse(journal_path.exists())
            self.assertEqual(
                LibraryDatabase.open(service.ziku_dir).load_data()["变体详情"][variant_id]["状态"],
                config.STATUS_PENDING_MANUAL_REVIEW,
            )

            recovered = GlyphService(service.ziku_name, service.ziku_dir)

            self.assertEqual(
                recovered.get_variant(variant_id)["状态"],
                config.STATUS_REVIEWED,
            )
            self.assertFalse(journal_path.exists())

    def test_stop_confirmation_race_does_not_relock_finished_page(self) -> None:
        page = ReviewPage()
        worker = review_page_module._BulkReviewWorker(
            lambda _progress, _cancel_check: {}
        )
        page._batch_worker = worker
        page._set_batch_running(True, 1)

        def finish_while_confirming() -> bool:
            page._batch_worker = None
            page._set_batch_running(False)
            return True

        with (
            patch.object(
                page,
                "_confirm_stop_bulk_review",
                side_effect=finish_while_confirming,
            ),
            patch.object(worker, "request_cancel") as request_cancel,
        ):
            page._request_stop_bulk_review()

        request_cancel.assert_not_called()
        self.assertFalse(page._batch_running)
        self.assertTrue(page._home_button.isEnabled())
        self.assertTrue(page._batch_progress_widget.isHidden())
        page.deleteLater()

    def test_bulk_review_worker_exception_restores_all_disabled_controls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, variants = self._build_library(Path(directory))
            page = ReviewPage()
            pending_id = str(variants["pending"])
            self.assertTrue(page.open_library(service.ziku_dir, pending_id))

            with (
                patch.object(
                    review_page_module,
                    "_run_bulk_review",
                    side_effect=RuntimeError("模拟工作线程异常"),
                ),
                patch("ui.pages.review_page.QMessageBox.critical") as critical,
            ):
                page._start_bulk_review([pending_id])
                deadline = time.monotonic() + 5.0
                while page._batch_running and time.monotonic() < deadline:
                    self.app.processEvents()
                    time.sleep(0.01)
                self.app.processEvents()

            self.assertFalse(page._batch_running)
            self.assertIsNone(page._batch_worker)
            self.assertTrue(page._home_button.isEnabled())
            self.assertTrue(page._complete_button.isEnabled())
            self.assertTrue(page._main_splitter.isEnabled())
            self.assertTrue(page._toolbar_widget.isEnabled())
            self.assertTrue(page._save_button.isEnabled())
            self.assertTrue(page._approve_button.isEnabled())
            self.assertTrue(all(action.isEnabled() for action in page._shortcut_actions))
            self.assertTrue(page._batch_progress_widget.isHidden())
            critical.assert_called_once()
            self.assertIn("模拟工作线程异常", critical.call_args.args[2])
            self.assertIn("总耗时：", critical.call_args.args[2])
            page.deleteLater()

    def test_bulk_review_refresh_exception_cannot_leave_page_locked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, variants = self._build_library(Path(directory))
            page = ReviewPage()
            self.assertTrue(
                page.open_library(service.ziku_dir, str(variants["pending"]))
            )
            worker = review_page_module._BulkReviewWorker(
                lambda _progress, _cancel_check: {}
            )
            worker.setAutoDelete(False)
            page._batch_worker = worker
            page._set_batch_running(True, 1)

            with (
                patch.object(
                    page,
                    "_reload_after_bulk_review",
                    side_effect=RuntimeError("模拟页面刷新异常"),
                ),
                patch("ui.pages.review_page.QMessageBox.critical") as critical,
            ):
                page._bulk_review_finished(
                    {"成功": 1, "跳过": 0, "失败": 0},
                    worker,
                )

            self.assertFalse(page._batch_running)
            self.assertIsNone(page._batch_worker)
            self.assertTrue(page._home_button.isEnabled())
            self.assertTrue(page._complete_button.isEnabled())
            self.assertTrue(page._main_splitter.isEnabled())
            self.assertTrue(page._toolbar_widget.isEnabled())
            self.assertTrue(page._save_button.isEnabled())
            self.assertTrue(page._approve_button.isEnabled())
            self.assertTrue(all(action.isEnabled() for action in page._shortcut_actions))
            self.assertTrue(page._batch_progress_widget.isHidden())
            critical.assert_called_once()
            self.assertIn("模拟页面刷新异常", critical.call_args.args[2])
            self.assertIn("总耗时：", critical.call_args.args[2])
            page.deleteLater()

    def test_pending_review_ids_include_detail_missing_from_group_index(self) -> None:
        page = ReviewPage()
        service = MagicMock()
        service.get_variants.return_value = {
            "grouped": {"状态": config.STATUS_PENDING_MANUAL_REVIEW},
            "orphaned": {"状态": config.STATUS_PENDING_MANUAL_REVIEW},
            "reviewed": {"状态": config.STATUS_REVIEWED},
        }
        service.get_glyph_groups.return_value = {
            "组": ["grouped", "grouped"],
        }
        page._service = service

        self.assertEqual(page._pending_review_ids(), ["grouped", "orphaned"])
        page.deleteLater()

    def test_bulk_review_prefers_existing_manual_draft(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, _variants = self._build_library(Path(directory))
            directories = service.get_workflow_dirs()
            variant_id = service.add_original(
                "人",
                "人-0001.png",
                "人-0001.png",
                "md5-manual-preferred",
            )
            preview_path = Path(directories["优化预览"]) / "人-0001.png"
            manual_path = Path(directories["手工审核"]) / "人-0001.png"
            self._write_positioned_glyph(preview_path, 40)
            self._write_positioned_glyph(manual_path, 4)
            manual_bytes = manual_path.read_bytes()
            service.update_variant(
                variant_id,
                **{
                    "状态": config.STATUS_PENDING_MANUAL_REVIEW,
                    "中间文件": preview_path.name,
                    "审核文件": manual_path.name,
                },
            )
            service.save()

            progress_events: list[dict[str, object]] = []
            result = review_page_module._run_bulk_review(
                service.ziku_name,
                service.ziku_dir,
                [variant_id],
                progress_events.append,
            )

            self.assertEqual(result["成功"], 1)
            self.assertEqual(result["失败"], 0)
            self.assertEqual([event["阶段"] for event in progress_events], ["开始", "完成"])
            self.assertEqual(progress_events[0]["已处理"], 0)
            self.assertEqual(progress_events[1]["已处理"], 1)
            reloaded = GlyphService(service.ziku_name, service.ziku_dir)
            detail = reloaded.get_variant(variant_id)
            self.assertTrue(detail["手工编辑"]["已编辑"])
            self.assertEqual(manual_path.read_bytes(), manual_bytes)
            self.assertEqual(
                detail["审核MD5"],
                review_page_module._file_md5(str(manual_path)),
            )
            saved = QImage(str(Path(directories["手工审核"]) / detail["审核文件"]))
            self.assertFalse(saved.isNull())
            self.assertGreater(saved.pixelColor(6, 20).alpha(), 0)
            self.assertEqual(saved.pixelColor(42, 20).alpha(), 0)
            bounds = review_page_module._effective_ink_bounds(saved)
            self.assertIsNotNone(bounds)
            left, top, right, bottom = bounds
            self.assertEqual((right - left, bottom - top), (8, 36))

    def test_existing_manual_draft_skips_png_encoding_and_keeps_edit_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, variants = self._build_library(Path(directory))
            variant_id = str(variants["pending"])
            directories = service.get_workflow_dirs()
            manual_path = Path(directories["手工审核"]) / "何-0001.png"
            self._write_positioned_glyph(manual_path, 6)
            manual_bytes = manual_path.read_bytes()
            service.update_variant(
                variant_id,
                **{
                    "审核文件": manual_path.name,
                    "审核MD5": "过期的摘要",
                    "手工编辑": {"已编辑": False, "最后保存时间": ""},
                },
            )
            state_before = service.snapshot_state()

            with (
                patch.object(service, "save") as save,
                patch.object(
                    review_page_module.tempfile,
                    "mkstemp",
                    side_effect=AssertionError("已有人工稿不应重新编码"),
                ),
                patch.object(
                    review_page_module,
                    "_reserve_review_backup",
                    side_effect=AssertionError("已有人工稿不应替换原文件"),
                ),
            ):
                filename = review_page_module._save_and_approve_review(
                    service,
                    variant_id,
                    (64, 64),
                    300,
                )

            save.assert_called_once_with()
            self.assertEqual(filename, manual_path.name)
            self.assertEqual(manual_path.read_bytes(), manual_bytes)
            detail = service.get_variant(variant_id)
            self.assertEqual(detail["状态"], config.STATUS_REVIEWED)
            self.assertTrue(detail["手工编辑"]["已编辑"])
            self.assertEqual(
                detail["审核MD5"],
                review_page_module._file_md5(str(manual_path)),
            )
            self.assertNotEqual(service.snapshot_state(), state_before)

    def test_existing_manual_draft_metadata_failure_rolls_back_without_touching_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, variants = self._build_library(Path(directory))
            variant_id = str(variants["pending"])
            manual_path = (
                Path(service.get_workflow_dirs()["手工审核"]) / "何-0001.png"
            )
            finished_path = Path(service.get_workflow_dirs()["成品"]) / "何-0001.png"
            self._write_positioned_glyph(manual_path, 6)
            self._write_positioned_glyph(finished_path, 12)
            service.update_variant(
                variant_id,
                **{
                    "审核文件": manual_path.name,
                    "审核MD5": "原摘要",
                    "成品文件": finished_path.name,
                    "成品MD5": "原成品摘要",
                },
            )
            state_before = service.snapshot_state()
            manual_bytes = manual_path.read_bytes()
            finished_bytes = finished_path.read_bytes()

            with patch.object(
                service,
                "save",
                side_effect=OSError("模拟已有人工稿元数据保存失败"),
            ):
                with self.assertRaises(OSError):
                    review_page_module._save_and_approve_review(
                        service,
                        variant_id,
                        (64, 64),
                        300,
                    )

            self.assertEqual(service.snapshot_state(), state_before)
            self.assertEqual(manual_path.read_bytes(), manual_bytes)
            self.assertEqual(finished_path.read_bytes(), finished_bytes)
            self.assertFalse(
                list(finished_path.parent.glob(".fonteditor_finished_rollback_*"))
            )

    def test_bulk_review_renders_automatic_transform_like_review_canvas(self) -> None:
        source = QImage(QSize(64, 64), QImage.Format.Format_ARGB32)
        source.fill(QColor(0, 0, 0, 0))
        painter = QPainter(source)
        painter.fillRect(18, 9, 20, 44, QColor("#111111"))
        painter.end()
        transform = {
            "偏移X": 7.0,
            "偏移Y": -4.0,
            "缩放": 1.2,
            "拉伸W": 1.1,
            "拉伸H": 0.8,
            "旋转": 11.0,
            "扭曲": [2.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -2.0],
        }

        canvas = ReviewCanvas()
        canvas.set_image(source, (64, 64))
        canvas.set_transform(
            x=transform["偏移X"],
            y=transform["偏移Y"],
            scale=transform["缩放"],
            stretch_w=transform["拉伸W"],
            stretch_h=transform["拉伸H"],
            rotation=transform["旋转"],
            distort=transform["扭曲"],
        )

        rendered, origin = review_page_module._render_review_source(
            source,
            (64, 64),
            transform,
        )

        self.assertEqual(rendered, canvas.image())
        self.assertEqual(origin, (canvas.output_origin().x(), canvas.output_origin().y()))
        canvas.deleteLater()

    def test_identity_render_fast_path_is_pixel_identical(self) -> None:
        source = QImage(QSize(64, 64), QImage.Format.Format_ARGB32)
        source.fill(QColor(0, 0, 0, 0))
        painter = QPainter(source)
        painter.fillRect(9, 7, 31, 43, QColor(17, 34, 51, 180))
        painter.end()
        source_pixels = review_page_module._qimage_to_rgba(
            review_page_module._to_review_image(source)
        )
        legacy = review_page_module.compose_rgba_on_canvas(
            source_pixels,
            (0.0, 0.0),
            (64, 64),
            expand_symmetric=True,
        )
        expected = review_page_module._rgba_to_qimage(
            legacy.pixels
        ).convertToFormat(QImage.Format.Format_ARGB32)

        with patch.object(
            review_page_module,
            "_qimage_to_rgba",
            side_effect=AssertionError("同尺寸无变换稿不应进入通用像素合成"),
        ):
            rendered, origin = review_page_module._render_review_source(
                source,
                (64, 64),
                None,
            )

        self.assertEqual(rendered, expected)
        self.assertEqual(origin, (0, 0))

    def test_normal_size_preparation_scans_effective_bounds_once(self) -> None:
        source = QImage(QSize(100, 100), QImage.Format.Format_ARGB32)
        source.fill(QColor(0, 0, 0, 0))
        painter = QPainter(source)
        painter.fillRect(20, 10, 60, 75, QColor("#111111"))
        painter.end()
        original_scan = review_page_module._effective_ink_bounds_from_rgba

        with patch.object(
            review_page_module,
            "_effective_ink_bounds_from_rgba",
            wraps=original_scan,
        ) as scan:
            prepared, origin, bounds = review_page_module._prepare_review_source(
                source,
                (100, 100),
                None,
                normalize_initial=True,
                include_bounds=True,
            )

        self.assertEqual(scan.call_count, 1)
        self.assertEqual(prepared, source)
        self.assertEqual(origin, (0, 0))
        self.assertEqual(bounds, (20, 10, 80, 85))

    def test_initial_size_normalization_thresholds_and_rectangular_canvas(self) -> None:
        def source_image(
            image_size: tuple[int, int],
            bounds: tuple[int, int, int, int],
        ) -> QImage:
            image = QImage(
                QSize(*image_size),
                QImage.Format.Format_ARGB32,
            )
            image.fill(QColor(0, 0, 0, 0))
            left, top, width, height = bounds
            painter = QPainter(image)
            painter.fillRect(left, top, width, height, QColor("#111111"))
            painter.end()
            return image

        def occupancy(image: QImage, canvas: tuple[int, int]) -> float:
            bounds = review_page_module._effective_ink_bounds(image)
            self.assertIsNotNone(bounds)
            left, top, right, bottom = bounds
            return max(
                (right - left) / canvas[0],
                (bottom - top) / canvas[1],
            )

        cases = (
            ((100, 100), (100, 100), (20, 20, 59, 20), True),
            ((100, 100), (100, 100), (20, 20, 60, 20), False),
            ((120, 100), (100, 100), (0, 20, 120, 20), False),
            ((121, 100), (100, 100), (0, 20, 121, 20), True),
            ((200, 100), (200, 100), (20, 20, 20, 50), True),
            ((250, 250), (250, 250), (100, 100, 1, 1), True),
        )
        for image_size, canvas_size, ink_bounds, normalized in cases:
            with self.subTest(
                image_size=image_size,
                canvas_size=canvas_size,
                ink_bounds=ink_bounds,
            ):
                source = source_image(image_size, ink_bounds)
                prepared, _origin, prepared_bounds = review_page_module._prepare_review_source(
                    source,
                    canvas_size,
                    None,
                    normalize_initial=True,
                    include_bounds=True,
                )
                self.assertIsNotNone(prepared_bounds)
                actual = occupancy(prepared, canvas_size)
                if normalized:
                    self.assertGreaterEqual(actual, 0.94)
                    self.assertLessEqual(actual, 0.96)
                else:
                    expected = max(
                        ink_bounds[2] / canvas_size[0],
                        ink_bounds[3] / canvas_size[1],
                    )
                    self.assertAlmostEqual(actual, expected, places=6)

    def test_interactive_and_bulk_share_normalized_load_and_keep_original_preview(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, variants = self._build_library(Path(directory))
            variant_id = str(variants["pending"])
            detail = service.get_variant(variant_id)
            preview_path = (
                Path(service.get_workflow_dirs()["优化预览"])
                / str(detail["中间文件"])
            )
            small = QImage(QSize(64, 64), QImage.Format.Format_ARGB32)
            small.fill(QColor(0, 0, 0, 0))
            painter = QPainter(small)
            painter.fillRect(27, 26, 8, 10, QColor("#111111"))
            painter.end()
            self.assertTrue(small.save(str(preview_path), "PNG"))

            page = ReviewPage()
            self.assertTrue(page.open_library(service.ziku_dir, variant_id))
            interactive = page._canvas.image()
            interactive_bounds = review_page_module._effective_ink_bounds(interactive)
            self.assertIsNotNone(interactive_bounds)
            left, top, right, bottom = interactive_bounds
            self.assertGreaterEqual(max((right - left) / 64, (bottom - top) / 64), 0.94)
            self.assertFalse(page._canvas.is_dirty)
            self.assertEqual(
                page._canvas._source_image,
                review_page_module._to_review_image(QImage(str(preview_path))),
            )
            self.assertNotEqual(page._canvas._source_image, interactive)

            result = review_page_module._run_bulk_review(
                service.ziku_name,
                service.ziku_dir,
                [variant_id],
                lambda _progress: None,
            )
            self.assertEqual(result["成功"], 1)
            reloaded = GlyphService(service.ziku_name, service.ziku_dir)
            saved_detail = reloaded.get_variant(variant_id)
            saved = QImage(
                str(
                    Path(reloaded.get_workflow_dirs()["手工审核"])
                    / str(saved_detail["审核文件"])
                )
            )
            self.assertEqual(saved, interactive)
            page.deleteLater()

    def test_bulk_review_save_failure_rolls_back_single_glyph(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, variants = self._build_library(Path(directory))
            variant_id = str(variants["pending"])
            detail_reference = service.get_variant(variant_id)
            state_before = service.snapshot_state()
            database_before = LibraryDatabase.open(service.ziku_dir).load_data()
            review_dir = Path(service.get_workflow_dirs()["手工审核"])
            output_path = review_dir / "何-0001.png"
            self.assertFalse(output_path.exists())

            with patch.object(
                service,
                "save",
                side_effect=OSError("模拟批量审核 JSON 写入失败"),
            ):
                with self.assertRaises(OSError):
                    review_page_module._save_and_approve_review(
                        service,
                        variant_id,
                        (64, 64),
                        300,
                    )

            self.assertEqual(service.snapshot_state(), state_before)
            self.assertIs(service.get_variant(variant_id), detail_reference)
            self.assertEqual(
                detail_reference,
                state_before["变体详情"][variant_id],
            )
            self.assertEqual(
                LibraryDatabase.open(service.ziku_dir).load_data(),
                database_before,
            )
            self.assertFalse(output_path.exists())
            self.assertFalse(list(review_dir.glob(".fonteditor_review_*")))

    def test_review_save_failure_restores_magicmock_detail_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preview_dir = root / "优化预览"
            review_dir = root / "手工审核"
            finished_dir = root / "成品"
            for path in (preview_dir, review_dir, finished_dir):
                path.mkdir()
            source_path = preview_dir / "测-0001.png"
            self._write_glyph(source_path)
            detail = {
                "归属字": "测",
                "状态": config.STATUS_PENDING_MANUAL_REVIEW,
                "中间文件": source_path.name,
                "审核文件": "",
                "审核MD5": "",
                "成品文件": "",
                "成品MD5": "",
                "变换参数": {},
            }
            detail_before = dict(detail)
            service = MagicMock()
            service.get_variant.return_value = detail
            service.get_workflow_dirs.return_value = {
                "优化预览": str(preview_dir),
                "手工审核": str(review_dir),
                "成品": str(finished_dir),
            }
            service.snapshot_variant_state.return_value = {
                "元数据": {"最后修改": "保存前"},
                "整体协调": {"几何协调完成": True},
            }
            service.default_transform_params.return_value = {
                "缩放": 1.0,
                "旋转": 0.0,
                "偏移X": 0,
                "偏移Y": 0,
                "拉伸W": 1.0,
                "拉伸H": 1.0,
                "扭曲": [0.0] * 8,
            }

            def mark_saved(*_args, **_kwargs) -> None:
                detail.update(
                    {
                        "状态": config.STATUS_REVIEWED,
                        "审核文件": source_path.name,
                        "审核MD5": "新摘要",
                    }
                )

            service.mark_manual_saved.side_effect = mark_saved
            service.approve_manual_review.return_value = True
            service.save.side_effect = OSError("模拟测试替身保存失败")

            with self.assertRaisesRegex(OSError, "模拟测试替身保存失败"):
                review_page_module._save_and_approve_review(
                    service,
                    "mock-variant",
                    (64, 64),
                    300,
                )

            self.assertEqual(detail, detail_before)
            service.restore_variant_state.assert_called_once_with(
                service.snapshot_variant_state.return_value
            )
            self.assertFalse((review_dir / source_path.name).exists())
            self.assertFalse(list(review_dir.glob(".fonteditor_review_*")))

    def test_bulk_review_rejects_transparent_and_white_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, _variants = self._build_library(Path(directory))
            preview_dir = Path(service.get_workflow_dirs()["优化预览"])
            pending_ids: list[str] = []
            for char, filename, color in (
                ("空", "空-0001.png", QColor(0, 0, 0, 0)),
                ("白", "白-0001.png", QColor(255, 255, 255, 255)),
            ):
                variant_id = service.add_original(
                    char,
                    filename,
                    filename,
                    f"blank-{char}",
                )
                image = QImage(QSize(64, 64), QImage.Format.Format_ARGB32)
                image.fill(color)
                self.assertTrue(image.save(str(preview_dir / filename), "PNG"))
                service.update_variant(
                    variant_id,
                    **{
                        "状态": config.STATUS_PENDING_MANUAL_REVIEW,
                        "中间文件": filename,
                    },
                )
                pending_ids.append(variant_id)
            service.save()

            result = review_page_module._run_bulk_review(
                service.ziku_name,
                service.ziku_dir,
                pending_ids,
                lambda _progress: None,
            )

            self.assertEqual(result["成功"], 0)
            self.assertEqual(result["失败"], 2)
            self.assertTrue(
                all(
                    "没有有效文字前景" in reason
                    for _variant_id, reason in result["失败详情"]
                )
            )
            reloaded = GlyphService(service.ziku_name, service.ziku_dir)
            for variant_id in pending_ids:
                detail = reloaded.get_variant(variant_id)
                self.assertEqual(detail["状态"], config.STATUS_PENDING_MANUAL_REVIEW)
                self.assertFalse(detail.get("审核文件", ""))

    def test_single_glyph_save_actions_are_rejected_while_bulk_review_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, variants = self._build_library(Path(directory))
            page = ReviewPage()
            self.assertTrue(page.open_library(service.ziku_dir, str(variants["pending"])))
            page._batch_running = True
            messages: list[str] = []
            page.status_message.connect(messages.append)

            with patch.object(page._service, "save") as save:
                self.assertFalse(page.save_current())
                page.approve_current()

            save.assert_not_called()
            self.assertEqual(len(messages), 2)
            self.assertIn("暂时不能保存单字", messages[0])
            self.assertIn("暂时不能审核单字", messages[1])
            page._batch_running = False
            page.deleteLater()

    def test_bulk_review_requires_saving_dirty_canvas_before_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, variants = self._build_library(Path(directory))
            page = ReviewPage()
            self.assertTrue(page.open_library(service.ziku_dir, str(variants["pending"])))
            page._canvas.set_transform(x=8)
            self.assertTrue(page._canvas.is_dirty)

            with (
                patch.object(
                    QMessageBox,
                    "exec",
                    autospec=True,
                    return_value=QMessageBox.StandardButton.Cancel.value,
                ) as exec_dialog,
                patch.object(page, "_confirm_bulk_review") as confirm_bulk,
                patch.object(page, "_start_bulk_review") as start_bulk,
            ):
                page.complete_all_reviews()

                dialogs = page.findChildren(QMessageBox)
                self.assertTrue(dialogs)
                dialog = dialogs[-1]
                save_button = dialog.button(QMessageBox.StandardButton.Save)
                cancel_button = dialog.button(QMessageBox.StandardButton.Cancel)
                self.assertIsNotNone(save_button)
                self.assertIsNotNone(cancel_button)
                self.assertEqual(save_button.text(), "保存修改并继续")
                self.assertEqual(cancel_button.text(), "取消")

            exec_dialog.assert_called_once()
            confirm_bulk.assert_not_called()
            start_bulk.assert_not_called()
            self.assertTrue(page._canvas.is_dirty)
            page.deleteLater()

    def test_right_tool_panels_do_not_show_operation_instructions(self) -> None:
        page = ReviewPage()

        for attribute in (
            "_transform_help_edit",
            "_tablet_hint_label",
            "_pixel_hint_label",
        ):
            self.assertFalse(hasattr(page, attribute))

        visible_text = "\n".join(
            label.text() for label in page.findChildren(QLabel)
        )
        for instruction in (
            "自由变换说明",
            "支持绘图板压感",
            "在画布上拖动",
            "鼠标滚轮调整笔触",
        ):
            self.assertNotIn(instruction, visible_text)
        page.deleteLater()

    def test_transform_controls_sync_horizontal_and_vertical_stretch(self) -> None:
        page = ReviewPage()
        image = QImage(QSize(64, 64), QImage.Format.Format_ARGB32)
        image.fill(QColor(0, 0, 0, 0))
        for y in range(20, 44):
            for x in range(20, 44):
                image.setPixelColor(x, y, QColor("#111111"))
        page._canvas.set_image(image)
        distort = [2.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -2.0]

        page._canvas.set_transform(
            stretch_w=1.35,
            stretch_h=0.72,
            distort=distort,
        )

        self.assertEqual(page._scale_slider.minimum(), -400)
        self.assertEqual(page._scale_slider.maximum(), 400)
        self.assertEqual(page._stretch_w_slider.minimum(), -400)
        self.assertEqual(page._stretch_w_slider.maximum(), 400)
        self.assertEqual(page._stretch_h_slider.minimum(), -400)
        self.assertEqual(page._stretch_h_slider.maximum(), 400)
        self.assertEqual(page._percent_to_slider_position(5), -400)
        self.assertEqual(page._percent_to_slider_position(100), 0)
        self.assertEqual(page._percent_to_slider_position(500), 400)
        self.assertEqual(page._slider_position_to_percent(-400), 5)
        self.assertEqual(page._slider_position_to_percent(0), 100)
        self.assertEqual(page._slider_position_to_percent(400), 500)
        for percent in range(
            page.TRANSFORM_PERCENT_MIN,
            page.TRANSFORM_PERCENT_MAX + 1,
        ):
            self.assertEqual(
                page._slider_position_to_percent(
                    page._percent_to_slider_position(percent)
                ),
                percent,
            )
        self.assertEqual(page._rotation_slider.minimum(), -180)
        self.assertEqual(page._rotation_slider.maximum(), 180)
        self.assertEqual(page._offset_x_spin.minimum(), -8192)
        self.assertEqual(page._offset_x_spin.maximum(), 8192)
        self.assertEqual(page._offset_y_spin.minimum(), -8192)
        self.assertEqual(page._offset_y_spin.maximum(), 8192)
        self.assertEqual(page._stretch_w_slider.value(), 35)
        self.assertEqual(page._stretch_h_slider.value(), -118)
        self.assertEqual(page._stretch_w_value_label.text(), "135%")
        self.assertEqual(page._stretch_h_value_label.text(), "72%")

        page._stretch_w_slider.setValue(page._percent_to_slider_position(175))
        page._stretch_h_slider.setValue(page._percent_to_slider_position(65))

        transform = page._canvas.transform()
        self.assertAlmostEqual(transform["stretch_w"], 1.75)
        self.assertAlmostEqual(transform["stretch_h"], 0.65)
        self.assertEqual(transform["distort"], distort)
        page.deleteLater()

    def test_out_of_range_control_display_does_not_overwrite_other_fields(self) -> None:
        page = ReviewPage()
        image = QImage(QSize(16, 16), QImage.Format.Format_ARGB32)
        image.fill(QColor("#111111"))
        page._canvas.set_image(image)
        page._sync_transform_controls(
            {
                "x": 20_000.0,
                "y": -20_000.0,
                "scale": 25.0,
                "stretch_w": 0.01,
                "stretch_h": 30.0,
                "rotation": 270.0,
            }
        )

        self.assertEqual(page._offset_x_spin.value(), 8192)
        self.assertEqual(page._offset_y_spin.value(), -8192)
        self.assertEqual(page._scale_slider.value(), 400)
        self.assertEqual(page._stretch_w_slider.value(), -400)
        self.assertEqual(page._stretch_h_slider.value(), 400)
        self.assertEqual(page._rotation_slider.value(), 180)
        self.assertEqual(page._scale_value_label.text(), "500%")
        self.assertEqual(page._stretch_w_value_label.text(), "5%")
        self.assertEqual(page._stretch_h_value_label.text(), "500%")
        self.assertEqual(page._rotation_value_label.text(), "180°")

        with patch.object(page._canvas, "set_transform") as set_transform:
            page._stretch_w_slider.setValue(page._percent_to_slider_position(125))

        set_transform.assert_called_once_with(stretch_w=1.25)
        page.deleteLater()

    def test_transform_slider_drag_commits_once_but_step_action_is_immediate(self) -> None:
        page = ReviewPage()
        image = QImage(QSize(16, 16), QImage.Format.Format_ARGB32)
        image.fill(QColor("#111111"))
        page._canvas.set_image(image)
        slider = page._scale_slider

        with patch.object(
            page._canvas,
            "set_transform",
            wraps=page._canvas.set_transform,
        ) as set_transform:
            slider.setSliderDown(True)
            slider.setValue(page._percent_to_slider_position(110))
            slider.setValue(page._percent_to_slider_position(125))
            slider.setValue(page._percent_to_slider_position(140))

            self.assertEqual(page._scale_value_label.text(), "140%")
            self.assertEqual(set_transform.call_count, 0)
            self.assertAlmostEqual(page._canvas.transform()["scale"], 1.0)

            slider.setSliderDown(False)

            self.assertEqual(set_transform.call_count, 1)
            set_transform.assert_called_with(scale=1.4)
            self.assertAlmostEqual(page._canvas.transform()["scale"], 1.4)
            self.assertEqual(len(page._canvas._undo_stack), 1)

            slider.triggerAction(QAbstractSlider.SliderAction.SliderSingleStepAdd)

            self.assertEqual(set_transform.call_count, 2)
            set_transform.assert_called_with(scale=1.41)
            self.assertAlmostEqual(page._canvas.transform()["scale"], 1.41)
            self.assertEqual(page._scale_value_label.text(), "141%")
            self.assertEqual(len(page._canvas._undo_stack), 2)
        page.deleteLater()

    def test_automatic_preview_loads_legacy_stretch_and_distort_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, variants = self._build_library(Path(directory))
            variant_id = str(variants["pending"])
            detail = service.get_variant(variant_id)
            distort = [4.0, -3.0, 2.0, 1.0, -2.0, 5.0, 3.0, -1.0]
            detail["变换参数"].update(
                {"拉伸W": 1.35, "拉伸H": 0.72, "扭曲": distort}
            )
            service.save()

            page = ReviewPage()
            with patch.object(
                review_page_module,
                "_prepare_review_source",
                wraps=review_page_module._prepare_review_source,
            ) as prepare_source:
                page.open_library(service.ziku_dir, variant_id)

            prepare_source.assert_called_once()
            loaded = prepare_source.call_args.args[2]
            self.assertAlmostEqual(loaded["拉伸W"], 1.35)
            self.assertAlmostEqual(loaded["拉伸H"], 0.72)
            self.assertEqual(loaded["扭曲"], distort)
            self.assertTrue(prepare_source.call_args.kwargs["normalize_initial"])
            self.assertEqual(page._canvas.transform()["stretch_w"], 1.0)
            self.assertEqual(page._canvas.transform()["stretch_h"], 1.0)
            self.assertEqual(page._canvas.transform()["distort"], [0.0] * 8)
            self.assertEqual(page._stretch_w_slider.value(), 0)
            self.assertEqual(page._stretch_h_slider.value(), 0)
            self.assertFalse(page._canvas.is_dirty)
            page.deleteLater()

    def test_canvas_brush_size_updates_slider_label_and_tool_controls(self) -> None:
        page = ReviewPage()
        self.assertEqual(page._brush_slider.minimum(), 1)
        self.assertEqual(page._brush_slider.maximum(), 100)

        page._canvas.set_brush_size(73)

        self.assertEqual(page._brush_slider.value(), 73)
        self.assertEqual(page._brush_value_label.text(), "73 px")
        page._set_tool(ReviewCanvas.TOOL_BRUSH)
        self.assertEqual(page._pixel_title.text(), "画笔")
        self.assertFalse(page._ink_controls.isHidden())
        page._set_tool(ReviewCanvas.TOOL_ERASER)
        self.assertEqual(page._pixel_title.text(), "橡皮")
        self.assertTrue(page._ink_controls.isHidden())
        page.deleteLater()

    def test_pixel_parameters_keep_compact_spacing_at_full_height(self) -> None:
        page = ReviewPage()
        page.resize(1600, 900)
        page.show()

        for tool in (ReviewCanvas.TOOL_BRUSH, ReviewCanvas.TOOL_ERASER):
            page._set_tool(tool)
            self.app.processEvents()

            visible_widgets = [
                page._pixel_title,
                page._brush_slider,
                page._pressure_checkbox,
                page._minimum_pressure_slider,
            ]
            if tool == ReviewCanvas.TOOL_BRUSH:
                visible_widgets.append(page._ink_controls)
            occupied_bottom = max(widget.geometry().bottom() for widget in visible_widgets)
            self.assertLessEqual(
                page._pixel_panel.height() - occupied_bottom,
                16,
            )
            self.assertLessEqual(page._parameters_stack.height(), 320)
            stack_bottom = page._parameters_stack.geometry().bottom()
            draft_top = page._draft_information_panel.geometry().top()
            self.assertLessEqual(draft_top - stack_bottom, 32)

        page._set_tool(ReviewCanvas.TOOL_TRANSFORM)
        self.app.processEvents()
        self.assertLessEqual(page._parameters_stack.height(), 420)
        stack_bottom = page._parameters_stack.geometry().bottom()
        draft_top = page._draft_information_panel.geometry().top()
        self.assertLessEqual(draft_top - stack_bottom, 32)

        page.deleteLater()

    def test_brush_brackets_work_across_review_page_focus(self) -> None:
        page = ReviewPage()
        page.show()
        page._set_controls_enabled(True)
        page._set_tool(ReviewCanvas.TOOL_BRUSH)
        page._brush_slider.setValue(10)

        focus_cases = (
            (page._canvas, Qt.Key.Key_BracketRight, 12),
            (page._tool_buttons[ReviewCanvas.TOOL_BRUSH], Qt.Key.Key_BracketLeft, 10),
            (page._pressure_checkbox, Qt.Key.Key_BracketRight, 12),
            (page._brush_slider, Qt.Key.Key_BracketLeft, 10),
        )
        for widget, key, expected_size in focus_cases:
            with self.subTest(widget=type(widget).__name__, key=key):
                widget.setFocus()
                self.app.processEvents()
                self.assertTrue(widget.hasFocus())

                QTest.keyClick(widget, key)
                self.app.processEvents()

                self.assertEqual(page._canvas.brush_size, expected_size)
                self.assertEqual(page._brush_slider.value(), expected_size)
                self.assertEqual(page._brush_value_label.text(), f"{expected_size} px")
        page.deleteLater()

    def test_space_pan_is_routed_from_child_focus_while_cursor_is_over_canvas(self) -> None:
        page = ReviewPage()
        page.show()
        late_slider = QSlider(Qt.Orientation.Horizontal, page)
        late_slider.show()
        late_slider.setFocus()
        self.app.processEvents()

        with (
            patch.object(page, "_cursor_is_over_canvas", return_value=True),
            patch.object(
                page._canvas,
                "handle_space_pan_key",
                return_value=True,
            ) as handle_space_pan_key,
        ):
            QTest.keyPress(late_slider, Qt.Key.Key_Space)
            QTest.keyRelease(late_slider, Qt.Key.Key_Space)
            self.app.processEvents()

        self.assertEqual(
            handle_space_pan_key.call_args_list,
            [call(True, auto_repeat=False), call(False, auto_repeat=False)],
        )
        late_slider.deleteLater()
        page.deleteLater()

    def test_space_pan_release_is_routed_after_cursor_leaves_canvas(self) -> None:
        page = ReviewPage()
        page.show()
        page._search_edit.setFocus()
        self.app.processEvents()

        with (
            patch.object(page, "_cursor_is_over_canvas", return_value=False),
            patch.object(
                type(page._canvas),
                "space_pan_active",
                new_callable=PropertyMock,
                return_value=True,
            ),
            patch.object(
                page._canvas,
                "handle_space_pan_key",
                return_value=True,
            ) as handle_space_pan_key,
        ):
            QTest.keyRelease(page._search_edit, Qt.Key.Key_Space)
            self.app.processEvents()

        handle_space_pan_key.assert_called_once_with(False, auto_repeat=False)
        page.deleteLater()

    def test_canvas_space_event_is_left_to_canvas_itself(self) -> None:
        page = ReviewPage()
        key_event = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_Space,
            Qt.KeyboardModifier.NoModifier,
            " ",
        )

        with (
            patch.object(page, "_cursor_is_over_canvas", return_value=True),
            patch.object(page._canvas, "handle_space_pan_key") as handle_space_pan_key,
        ):
            self.assertFalse(page.eventFilter(page._canvas, key_event))

        handle_space_pan_key.assert_not_called()
        page.deleteLater()

    def test_canvas_hover_detection_uses_global_cursor_coordinates(self) -> None:
        page = ReviewPage()
        page.resize(1100, 720)
        page.show()
        self.app.processEvents()
        original_position = QCursor.pos()

        try:
            canvas_center = page._canvas.mapToGlobal(page._canvas.rect().center())
            QCursor.setPos(canvas_center)
            self.app.processEvents()
            self.assertTrue(page._cursor_is_over_canvas())

            search_center = page._search_edit.mapToGlobal(
                page._search_edit.rect().center()
            )
            QCursor.setPos(search_center)
            self.app.processEvents()
            self.assertFalse(page._cursor_is_over_canvas())
        finally:
            QCursor.setPos(original_position)

        page.deleteLater()

    def test_discard_on_home_clears_draft_before_other_pages_and_reopen(self) -> None:
        """放弃人工稿后，经过自动优化页再回来也不得重复询问。"""
        with tempfile.TemporaryDirectory() as directory:
            service, variants = self._build_library(Path(directory))
            page = ReviewPage()
            home_page = QWidget()
            optimization_page = QWidget()
            stack = QStackedWidget()
            stack.addWidget(home_page)
            stack.addWidget(page)
            stack.addWidget(optimization_page)
            page.home_requested.connect(lambda: stack.setCurrentWidget(home_page))
            stack.setCurrentWidget(page)
            self.assertTrue(page.open_library(service.ziku_dir, str(variants["pending"])))

            saved_image = page._canvas.image()
            page._canvas.set_transform(x=12, rotation=8)
            page._canvas._begin_stroke(
                QPoint(5, 5),
                ReviewCanvas.TOOL_BRUSH,
                4.0,
            )
            page._canvas._end_stroke()
            self.assertTrue(page._canvas.is_dirty)
            self.assertTrue(page._canvas._undo_stack)
            self.assertNotEqual(page._canvas.image(), saved_image)

            with patch.object(
                QMessageBox,
                "exec",
                return_value=QMessageBox.StandardButton.Discard.value,
            ) as exec_dialog:
                page._request_home()

                dialogs = page.findChildren(QMessageBox)
                self.assertTrue(dialogs)
                dialog = dialogs[-1]
                save_button = dialog.button(QMessageBox.StandardButton.Save)
                discard_button = dialog.button(QMessageBox.StandardButton.Discard)
                cancel_button = dialog.button(QMessageBox.StandardButton.Cancel)
                self.assertIsNotNone(save_button)
                self.assertIsNotNone(discard_button)
                self.assertIsNotNone(cancel_button)
                self.assertEqual(save_button.text(), "保存修改")
                self.assertEqual(discard_button.text(), "放弃修改")
                self.assertEqual(cancel_button.text(), "取消")
                self.assertIs(dialog.defaultButton(), save_button)
                self.assertIs(dialog.escapeButton(), cancel_button)

                self.assertIs(stack.currentWidget(), home_page)
                self.assertFalse(page._canvas.is_dirty)
                self.assertEqual(page._canvas.image(), saved_image)
                self.assertEqual(page._canvas.transform()["x"], 0.0)
                self.assertEqual(page._canvas.transform()["rotation"], 0.0)
                self.assertFalse(page._canvas._undo_stack)
                self.assertFalse(page._canvas._redo_stack)

                page._canvas.undo()
                self.assertFalse(page._canvas.is_dirty)
                self.assertEqual(page._canvas.transform()["x"], 0.0)

                stack.setCurrentWidget(optimization_page)
                stack.setCurrentWidget(home_page)
                self.assertTrue(page.open_library(service.ziku_dir, str(variants["pending"])))
                stack.setCurrentWidget(page)

            self.assertEqual(exec_dialog.call_count, 1)
            self.assertIs(stack.currentWidget(), page)
            self.assertFalse(page._canvas.is_dirty)
            stack.deleteLater()

    def test_save_unsaved_review_before_switching_glyph(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, variants = self._build_library(Path(directory))
            page = ReviewPage()
            current_id = str(variants["pending"])
            next_id = str(variants["reviewed"])
            self.assertTrue(page.open_library(service.ziku_dir, current_id))
            page._canvas.set_transform(x=9)
            self.assertTrue(page._canvas.is_dirty)

            with (
                patch.object(
                    QMessageBox,
                    "exec",
                    return_value=QMessageBox.StandardButton.Save.value,
                ),
                patch.object(page, "save_current", wraps=page.save_current) as save,
            ):
                page._item_tree.setCurrentItem(page._node_for_variant(next_id))
                self.app.processEvents()

            save.assert_called_once_with()
            self.assertEqual(page._current_variant_id, next_id)
            reloaded = GlyphService(service.ziku_name, service.ziku_dir)
            saved_detail = reloaded.get_variant(current_id)
            self.assertTrue(saved_detail.get("审核文件"))
            self.assertEqual(
                saved_detail["变换参数"]["偏移X"],
                0,
            )
            page.deleteLater()

    def test_failed_save_keeps_current_glyph_and_unsaved_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, variants = self._build_library(Path(directory))
            page = ReviewPage()
            current_id = str(variants["pending"])
            next_id = str(variants["reviewed"])
            self.assertTrue(page.open_library(service.ziku_dir, current_id))
            page._canvas.set_transform(x=9)

            with (
                patch.object(
                    QMessageBox,
                    "exec",
                    return_value=QMessageBox.StandardButton.Save.value,
                ),
                patch.object(page, "save_current", return_value=False) as save,
            ):
                page._item_tree.setCurrentItem(page._node_for_variant(next_id))
                self.app.processEvents()

            save.assert_called_once_with()
            self.assertEqual(page._current_variant_id, current_id)
            self.assertIs(
                page._item_tree.currentItem(),
                page._node_for_variant(current_id),
            )
            self.assertTrue(page._canvas.is_dirty)
            page.deleteLater()

    def test_cancel_home_keeps_unsaved_review_draft(self) -> None:
        """取消返回首页时必须保留当前人工稿及撤销历史。"""
        with tempfile.TemporaryDirectory() as directory:
            service, variants = self._build_library(Path(directory))
            page = ReviewPage()
            self.assertTrue(page.open_library(service.ziku_dir, str(variants["pending"])))
            page._canvas.set_transform(x=9)
            undo_count = len(page._canvas._undo_stack)
            home_requests: list[bool] = []
            page.home_requested.connect(lambda: home_requests.append(True))

            with patch.object(
                QMessageBox,
                "exec",
                autospec=True,
                return_value=QMessageBox.StandardButton.Cancel.value,
            ) as exec_dialog:
                page._request_home()

            exec_dialog.assert_called_once()
            self.assertEqual(home_requests, [])
            self.assertTrue(page._canvas.is_dirty)
            self.assertEqual(page._canvas.transform()["x"], 9.0)
            self.assertEqual(len(page._canvas._undo_stack), undo_count)
            page.deleteLater()

    def test_space_keeps_native_text_and_button_behavior_over_canvas(self) -> None:
        page = ReviewPage()
        page.resize(1100, 720)
        page.show()
        self.app.processEvents()

        with (
            patch.object(page, "_cursor_is_over_canvas", return_value=True),
            patch.object(page._canvas, "handle_space_pan_key") as handle_space_pan_key,
        ):
            page._search_edit.setText("字形")
            page._search_edit.setCursorPosition(len(page._search_edit.text()))
            page._search_edit.setFocus()
            QTest.keyClick(page._search_edit, Qt.Key.Key_Space)
            self.assertEqual(page._search_edit.text(), "字形 ")

            home_requests: list[bool] = []
            page.home_requested.connect(lambda: home_requests.append(True))
            page._home_button.setFocus()
            QTest.keyClick(page._home_button, Qt.Key.Key_Space)
            self.assertEqual(home_requests, [True])

            handle_space_pan_key.assert_not_called()

        page.deleteLater()

    def test_list_exposes_only_review_phase_and_defaults_to_all(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, variants = self._build_library(Path(directory))
            page = ReviewPage()
            page.open_library(service.ziku_dir)

            self.assertCountEqual(
                page._variant_ids,
                [
                    variants["pending"],
                    variants["reviewed"],
                    variants["finished"],
                ],
            )
            self.assertEqual(page._filter_combo.currentText(), PHASE_FILTER_ALL)
            self.assertEqual(page._item_tree.columnCount(), 2)
            self.assertEqual(page._item_tree.headerItem().text(0), "字形与文件")
            self.assertEqual(page._item_tree.headerItem().text(1), "状态与提示")
            header = page._item_tree.header()
            for column in range(2):
                self.assertEqual(
                    header.sectionResizeMode(column),
                    QHeaderView.ResizeMode.Interactive,
                )
            status_width = header.sectionSize(1)
            header.resizeSection(0, 48)
            header.resizeSection(1, 48)
            self.assertGreaterEqual(header.sectionSize(0), 160)
            self.assertGreaterEqual(header.sectionSize(1), status_width)
            self.assertTrue(page._item_tree.rootIsDecorated())
            self.assertEqual(page._item_tree.indentation(), 14)
            self.assertEqual(page._item_tree.iconSize(), QSize(38, 38))
            self.assertTrue(all(not item.icon(0).isNull() for item in page._item_nodes))
            self.assertEqual(
                {item.text(1).splitlines()[0] for item in page._item_nodes},
                {STAGE_PENDING_REVIEW, STATUS_REVIEWED},
            )
            self.assertTrue(all(item.parent() is not None for item in page._item_nodes))
            self.assertTrue(
                all(item.sizeHint(0).height() == 52 for item in page._item_nodes)
            )
            self.assertTrue(
                all(item.parent().isExpanded() for item in page._item_nodes)
            )
            self.assertEqual(page._item_tree.topLevelItemCount(), 3)
            for index in range(page._item_tree.topLevelItemCount()):
                parent = page._item_tree.topLevelItem(index)
                self.assertFalse(
                    bool(parent.flags() & Qt.ItemFlag.ItemIsSelectable)
                )
            self.assertEqual(
                page._count_label.text(),
                "待审核 1　已审核 2",
            )

            self.assertEqual(page._list_count_label.text(), "显示 / 总数：3 / 3")
            self.assertFalse(hasattr(page, "_marker_combo"))
            page.deleteLater()

    def test_review_status_filter_does_not_change_persisted_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, variants = self._build_library(Path(directory))
            page = ReviewPage()
            page.open_library(service.ziku_dir)

            self.assertEqual(
                [page._filter_combo.itemText(index) for index in range(page._filter_combo.count())],
                list(REVIEW_STATUS_FILTERS),
            )

            page._filter_combo.setCurrentText(STATUS_REVIEWED)

            self.assertCountEqual(
                page._variant_ids,
                [variants["reviewed"], variants["finished"]],
            )
            expected_color = QColor("#228B22")
            self.assertTrue(
                all(
                    item.text(1).splitlines()[0] == STATUS_REVIEWED
                    for item in page._item_nodes
                )
            )
            self.assertTrue(
                all(item.foreground(1).color() == expected_color for item in page._item_nodes)
            )
            tooltips = {item.toolTip(0) for item in page._item_nodes}
            self.assertTrue(
                all(f"手工审核：{STATUS_REVIEWED}" in text for text in tooltips)
            )

            page._filter_combo.setCurrentText(STAGE_PENDING_REVIEW)
            self.assertCountEqual(
                page._variant_ids,
                [variants["pending"]],
            )

            reloaded = GlyphService(service.ziku_name, service.ziku_dir)
            self.assertEqual(
                reloaded.get_variant(str(variants["reviewed"]))["状态"],
                config.STATUS_REVIEWED,
            )
            self.assertEqual(
                reloaded.get_variant(str(variants["finished"]))["状态"],
                config.STATUS_FINISHED,
            )

    def test_pending_optimization_item_is_not_listed_on_review_page(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, variants = self._build_library(Path(directory))
            variant_id = str(variants["pending_optimization"])
            detail = service.get_variant(variant_id)
            source_path = (
                Path(service.get_workflow_dirs()["原图"])
                / str(detail["原始文件"])
            )
            self._write_glyph(source_path)
            service.save()

            page = ReviewPage()
            page.open_library(service.ziku_dir, variant_id)

            self.assertNotIn(variant_id, page._records_by_id)
            self.assertNotIn(variant_id, page._variant_ids)
            self.assertNotEqual(page._current_variant_id, variant_id)
            page.deleteLater()

    def test_unsafe_pending_source_is_not_admitted_to_review_page(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, variants = self._build_library(Path(directory))
            variant_id = str(variants["pending_optimization"])
            escaped_path = Path(service.ziku_dir) / "越界.png"
            self._write_glyph(escaped_path)
            service.update_variant(variant_id, **{"原始文件": "../越界.png"})
            service.save()

            page = ReviewPage()
            page.open_library(service.ziku_dir)

            self.assertNotIn(variant_id, page._records_by_id)
            self.assertNotIn(variant_id, page._variant_ids)
            page.deleteLater()

    def test_unsaved_canvas_change_updates_marker_without_changing_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, variants = self._build_library(Path(directory))
            variant_id = str(variants["pending"])
            page = ReviewPage()
            page.open_library(service.ziku_dir, variant_id)

            page._canvas.set_transform(x=4)

            node = page._node_for_variant(variant_id)
            self.assertIsNotNone(node)
            self.assertEqual(node.text(1).splitlines()[0], STAGE_PENDING_REVIEW)
            self.assertIn("未保存修改", node.text(1).splitlines()[1])
            self.assertEqual(node.parent().text(1).splitlines()[1], "提示 1")

            self.assertFalse(hasattr(page, "_marker_combo"))
            self.assertTrue(page._canvas.is_dirty)

            page._canvas.discard_changes()
            self.assertNotIn(MARKER_UNSAVED, node.text(1).splitlines()[1])
            page.deleteLater()

    def test_pending_structure_risk_is_visible_in_review_list(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, variants = self._build_library(Path(directory))
            variant_id = str(variants["pending"])
            detail = service.get_variant(variant_id)
            optimization = dict(detail.get("自动优化", {}))
            optimization["方案"] = {
                "结构复核": {
                    "状态": "需人工核对",
                    "阶段": "原尺寸复核",
                    "原因": "参考端点仅匹配42.9%",
                    "风险等级": 1,
                }
            }
            service.update_variant(variant_id, **{"自动优化": optimization})
            service.save()
            page = ReviewPage()
            page.open_library(service.ziku_dir)

            node = page._node_for_variant(variant_id)

            self.assertIsNotNone(node)
            self.assertIn("结构需核对", node.text(1).splitlines()[1])
            self.assertIn("参考端点仅匹配42.9%", node.toolTip(0))
            self.assertEqual(
                node.foreground(0).color(),
                page.STRUCTURE_RISK_COLOR,
            )
            self.assertEqual(
                page._records_by_id[variant_id]["status"],
                config.STATUS_PENDING_MANUAL_REVIEW,
            )
            page.deleteLater()
            page.deleteLater()

    def test_same_character_variants_are_grouped_with_real_thumbnails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, variants = self._build_library(Path(directory))
            directories = service.get_workflow_dirs()
            second_id = service.add_original(
                "何",
                "何-0002.png",
                "何-0002.png",
                "md5-second-he",
            )
            self._write_glyph(Path(directories["优化预览"]) / "何-0002-优化.png")
            self._write_glyph(Path(directories["手工审核"]) / "何-0002.png")
            service.update_variant(
                second_id,
                **{
                    "状态": config.STATUS_REVIEWED,
                    "中间文件": "何-0002-优化.png",
                    "审核文件": "何-0002.png",
                },
            )
            service.save()

            page = ReviewPage()
            page.open_library(service.ziku_dir, second_id)
            page._filter_combo.setCurrentText(PHASE_FILTER_ALL)
            page._populate_variants(select_variant=second_id)

            parents = [
                page._item_tree.topLevelItem(index)
                for index in range(page._item_tree.topLevelItemCount())
            ]
            parent = next(item for item in parents if item.text(0).startswith("何（"))
            self.assertEqual(parent.text(0), "何（2个字形）")
            self.assertEqual(parent.text(1).splitlines()[0], "已审核 1/2")
            self.assertFalse(parent.flags() & Qt.ItemFlag.ItemIsSelectable)
            self.assertTrue(parent.isExpanded())
            self.assertEqual(parent.childCount(), 2)
            self.assertEqual(
                [parent.child(index).data(0, Qt.ItemDataRole.UserRole) for index in range(2)],
                [variants["pending"], second_id],
            )
            self.assertTrue(
                all(not parent.child(index).icon(0).isNull() for index in range(2))
            )
            thumbnail = parent.child(1).icon(0).pixmap(QSize(38, 38)).toImage()
            self.assertLess(thumbnail.pixelColor(19, 19).lightness(), 64)
            self.assertEqual(parent.child(0).text(0), "字形1 · 何-0001.png")
            self.assertEqual(parent.child(1).text(0), "字形2 · 何-0002.png")
            self.assertTrue(
                all(parent.child(index).sizeHint(0).height() == 52 for index in range(2))
            )
            self.assertEqual(
                Path(page._records_by_id[second_id]["source_path"]).parent,
                Path(directories["手工审核"]),
            )
            self.assertEqual(page._records_by_id[second_id]["stage"], "手工审核稿")
            self.assertEqual(
                page._records_by_id[str(variants["pending"])]["stage"],
                "自动优化稿",
            )
            self.assertEqual(page._list_count_label.text(), "显示 / 总数：4 / 4")
            self.assertEqual(
                page._count_label.text(),
                "待审核 1　已审核 3",
            )
            self.assertIs(page._item_tree.currentItem(), parent.child(1))

            parent.setExpanded(False)
            page._move_selection(-1)
            self.assertTrue(parent.isExpanded())
            self.assertIs(page._item_tree.currentItem(), parent.child(0))
            page._move_selection(1)
            self.assertIs(page._item_tree.currentItem(), parent.child(1))

            valid_child = parent.child(1)
            parent.setExpanded(False)
            with patch.object(page, "_confirm_discard") as confirm_discard:
                page._item_tree.setCurrentItem(parent)
                self.app.processEvents()
            confirm_discard.assert_not_called()
            self.assertIs(page._item_tree.currentItem(), valid_child)
            self.assertEqual(page._current_variant_id, second_id)
            self.assertFalse(parent.isExpanded())

            parent.setExpanded(True)
            page.resize(1100, 720)
            page.show()
            self.app.processEvents()
            parent_rect = page._item_tree.visualItemRect(parent)
            branch_position = QPoint(
                max(1, parent_rect.left() - page._item_tree.indentation() // 2),
                parent_rect.center().y(),
            )
            QTest.mouseClick(
                page._item_tree.viewport(),
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
                branch_position,
            )
            self.app.processEvents()
            self.assertFalse(parent.isExpanded())
            self.assertIs(page._item_tree.currentItem(), valid_child)
            QTest.mouseClick(
                page._item_tree.viewport(),
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
                branch_position,
            )
            self.app.processEvents()
            self.assertTrue(parent.isExpanded())
            self.assertIs(page._item_tree.currentItem(), valid_child)

            page._item_tree.setCurrentItem(parent.child(0))
            page._item_tree.setFocus()
            QTest.keyClick(page._item_tree, Qt.Key.Key_Up)
            self.app.processEvents()
            self.assertIs(page._item_tree.currentItem(), parent.child(0))
            self.assertEqual(page._current_variant_id, variants["pending"])
            page.deleteLater()

    def test_list_thumbnail_cache_reuses_file_fingerprint_and_clears_on_switch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_root = root / "第一字库"
            second_root = root / "第二字库"
            first_root.mkdir()
            second_root.mkdir()
            first_service, first_variants = self._build_library(first_root)
            second_service, _second_variants = self._build_library(second_root)

            page = ReviewPage()
            page.open_library(first_service.ziku_dir)

            initial_keys = set(page._list_thumbnail_cache)
            visible_records = [
                page._records_by_id[variant_id]
                for variant_id in page._variant_ids
                if page._records_by_id[variant_id]["source_path"]
            ]
            self.assertEqual(len(initial_keys), len(visible_records))
            for record in visible_records:
                path = str(record["source_path"])
                stat = os.stat(path)
                self.assertIn(
                    (
                        os.path.normcase(os.path.abspath(path)),
                        stat.st_mtime_ns,
                        stat.st_size,
                    ),
                    initial_keys,
                )

            with patch.object(
                page,
                "_render_glyph_thumbnail",
                wraps=page._render_glyph_thumbnail,
            ) as render_thumbnail:
                page._search_edit.setText("何")
                page._search_edit.clear()
                render_thumbnail.assert_not_called()

                source_path = Path(
                    page._records_by_id[str(first_variants["pending"])]["source_path"]
                )
                old_key = page._thumbnail_cache_key(str(source_path))
                source_stat = source_path.stat()
                os.utime(
                    source_path,
                    ns=(
                        source_stat.st_atime_ns,
                        source_stat.st_mtime_ns + 1_000_000_000,
                    ),
                )
                page._search_edit.setText("何")
                QTest.mouseClick(page._search_button, Qt.MouseButton.LeftButton)

                render_thumbnail.assert_called_once_with(str(source_path))
                new_key = page._thumbnail_cache_key(str(source_path))
                self.assertNotEqual(new_key, old_key)
                self.assertNotIn(old_key, page._list_thumbnail_cache)
                self.assertIn(new_key, page._list_thumbnail_cache)

            first_paths = {key[0] for key in page._list_thumbnail_cache}
            page.open_library(second_service.ziku_dir)
            second_paths = {key[0] for key in page._list_thumbnail_cache}
            self.assertTrue(second_paths)
            self.assertTrue(first_paths.isdisjoint(second_paths))
            page.deleteLater()

    def test_large_list_decodes_only_visible_thumbnails_in_background(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, variant_ids = self._build_large_library(Path(directory), total=80)
            page = ReviewPage()
            page.resize(1100, 720)
            page.show()
            self.app.processEvents()

            main_thread_id = threading.get_ident()
            decode_calls: list[tuple[int, str]] = []
            icon_threads: list[int] = []
            original_decode = ReviewPage._decode_glyph_thumbnail
            original_icon = ReviewPage._thumbnail_icon

            def tracked_decode(path: str, size: tuple[int, int]) -> QImage:
                decode_calls.append((threading.get_ident(), path))
                return original_decode(path, size)

            def tracked_icon(image: QImage):
                icon_threads.append(threading.get_ident())
                return original_icon(image)

            with (
                patch.object(
                    ReviewPage,
                    "_decode_glyph_thumbnail",
                    side_effect=tracked_decode,
                ),
                patch.object(
                    ReviewPage,
                    "_thumbnail_icon",
                    side_effect=tracked_icon,
                ),
            ):
                page.open_library(service.ziku_dir)

                self.assertEqual(decode_calls, [])
                self.assertEqual(len(page._list_thumbnail_cache), 0)
                self.assertEqual(page._item_tree.topLevelItemCount(), 1)
                parent = page._item_tree.topLevelItem(0)
                self.assertEqual(parent.childCount(), len(variant_ids))
                placeholder = page._thumbnail_placeholder()
                self.assertEqual(
                    parent.child(0).icon(0).cacheKey(),
                    placeholder.cacheKey(),
                )
                placeholder_image = placeholder.pixmap(
                    page._item_tree.iconSize()
                ).toImage()
                self.assertEqual(placeholder_image.pixelColor(0, 0), QColor("white"))

                deadline = time.monotonic() + 3.0
                while not page._list_thumbnail_cache and time.monotonic() < deadline:
                    self.app.processEvents()
                    time.sleep(0.01)
                self.assertTrue(page._list_thumbnail_cache)

                settle_deadline = time.monotonic() + 3.0
                while page._list_thumbnail_workers and time.monotonic() < settle_deadline:
                    self.app.processEvents()
                    time.sleep(0.01)

            self.assertTrue(decode_calls)
            self.assertTrue(
                all(thread_id != main_thread_id for thread_id, _path in decode_calls)
            )
            self.assertTrue(icon_threads)
            self.assertTrue(
                all(thread_id == main_thread_id for thread_id in icon_threads)
            )
            decoded_paths = {path for _thread_id, path in decode_calls}
            self.assertLess(len(decoded_paths), len(variant_ids))
            self.assertLess(len(page._list_thumbnail_cache), len(variant_ids))
            self.assertLessEqual(
                len(page._list_thumbnail_cache),
                page.LIST_THUMBNAIL_CACHE_ITEMS,
            )

            loaded_key = next(iter(page._list_thumbnail_cache))
            loaded_id = next(
                variant_id
                for variant_id, record in page._records_by_id.items()
                if page._thumbnail_cache_key(str(record["source_path"])) == loaded_key
            )
            self.assertNotEqual(
                page._node_for_variant(loaded_id).icon(0).cacheKey(),
                placeholder.cacheKey(),
            )
            page.close()
            page.deleteLater()

    def test_list_thumbnail_cache_is_bounded_and_rejects_old_generation(self) -> None:
        page = ReviewPage()
        self.assertEqual(page.LIST_THUMBNAIL_CACHE_ITEMS, 512)
        placeholder = page._thumbnail_placeholder()

        page._list_thumbnail_generation = 7
        stale_image = QImage(38, 38, QImage.Format.Format_ARGB32)
        stale_image.fill(QColor("white"))
        stale_key = ("过期字库.png", 1, 1)
        page._list_thumbnail_batch_finished(
            {
                "批次": -1,
                "代次": 6,
                "结果": [(stale_key, stale_image)],
            }
        )
        self.assertNotIn(stale_key, page._list_thumbnail_cache)

        for index in range(page.LIST_THUMBNAIL_CACHE_ITEMS + 8):
            page._store_glyph_thumbnail(
                (f"虚拟缩略图-{index:04d}.png", index, index + 1),
                placeholder,
            )
        self.assertEqual(
            len(page._list_thumbnail_cache),
            page.LIST_THUMBNAIL_CACHE_ITEMS,
        )
        self.assertNotIn(("虚拟缩略图-0000.png", 0, 1), page._list_thumbnail_cache)
        self.assertIn(("虚拟缩略图-0519.png", 519, 520), page._list_thumbnail_cache)
        page.deleteLater()

    def test_search_and_status_filter_keep_matching_group_expanded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, variants = self._build_library(Path(directory))
            directories = service.get_workflow_dirs()
            second_id = service.add_original(
                "何",
                "何-0002.png",
                "何-0002.png",
                "md5-filter-he",
            )
            self._write_glyph(Path(directories["手工审核"]) / "何-0002.png")
            service.update_variant(
                second_id,
                **{"状态": config.STATUS_REVIEWED, "审核文件": "何-0002.png"},
            )
            service.save()

            page = ReviewPage()
            page.open_library(service.ziku_dir)
            page._filter_combo.setCurrentText(PHASE_FILTER_ALL)
            page._search_edit.setText("何-0002")
            self.assertNotEqual(page._variant_ids, [second_id])
            self.assertEqual(page._search_button.text(), "搜索")
            QTest.keyClick(page._search_edit, Qt.Key.Key_Return)

            self.assertEqual(page._variant_ids, [second_id])
            self.assertEqual(page._item_tree.topLevelItemCount(), 1)
            parent = page._item_tree.topLevelItem(0)
            self.assertEqual(parent.text(0), "何（2个字形）")
            self.assertEqual(parent.text(1).splitlines()[0], "已审核 1/2")
            self.assertTrue(parent.isExpanded())
            self.assertEqual(parent.child(0).data(0, Qt.ItemDataRole.UserRole), second_id)

            page._search_edit.setText("何-0001")
            QTest.keyClick(page._search_edit, Qt.Key.Key_Return)
            self.assertEqual(page._variant_ids, [variants["pending"]])
            self.assertEqual(page._current_variant_id, variants["pending"])

            page._search_edit.clear()
            self.assertIn(second_id, page._variant_ids)
            self.assertIn(variants["pending"], page._variant_ids)
            page._filter_combo.setCurrentText(STATUS_REVIEWED)
            parent = next(
                page._item_tree.topLevelItem(index)
                for index in range(page._item_tree.topLevelItemCount())
                if page._item_tree.topLevelItem(index).text(0).startswith("何（")
            )
            self.assertEqual(parent.text(0), "何（2个字形）")
            self.assertEqual(parent.text(1).splitlines()[0], "已审核 1/2")
            self.assertTrue(parent.isExpanded())
            self.assertIn(second_id, page._variant_ids)
            self.assertNotIn(variants["pending"], page._variant_ids)
            page.deleteLater()

    def test_consecutive_search_does_not_match_review_status_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, _variants = self._build_library(Path(directory))
            review_dir = Path(service.get_workflow_dirs()["手工审核"])
            added_ids: dict[str, str] = {}
            for index, char in enumerate(("羅", "已"), start=1):
                filename = f"{char}-0001.png"
                variant_id = service.add_original(
                    char,
                    filename,
                    filename,
                    f"md5-consecutive-search-{index}",
                )
                self._write_glyph(review_dir / filename)
                service.update_variant(
                    variant_id,
                    **{"状态": config.STATUS_REVIEWED, "审核文件": filename},
                )
                added_ids[char] = variant_id
            service.save()

            page = ReviewPage()
            page.open_library(service.ziku_dir)
            all_variant_ids = list(page._variant_ids)

            page._search_edit.setText("羅")
            QTest.keyClick(page._search_edit, Qt.Key.Key_Return)
            self.assertEqual(page._variant_ids, [added_ids["羅"]])
            self.assertEqual(page._current_variant_id, added_ids["羅"])

            page._search_edit.setText("已")
            QTest.keyClick(page._search_edit, Qt.Key.Key_Return)
            self.assertEqual(page._variant_ids, [added_ids["已"]])
            self.assertEqual(page._current_variant_id, added_ids["已"])

            page._search_edit.clear()
            self.assertCountEqual(page._variant_ids, all_variant_ids)
            page.deleteLater()

    def test_save_and_approve_persists_review_draft_and_resets_transform(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, variants = self._build_library(Path(directory))
            variant_id = variants["eligible"][0]
            page = ReviewPage()
            page.open_library(service.ziku_dir, variant_id)
            page._canvas.set_transform(
                x=80,
                scale=1.2,
                stretch_w=1.4,
                stretch_h=0.8,
                distort=[2.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -2.0],
            )

            with (
                patch.object(page, "_populate_variants") as populate,
                patch.object(
                    page,
                    "_advance_after_review_approval",
                    wraps=page._advance_after_review_approval,
                ) as advance,
            ):
                page.approve_current()

            reloaded = GlyphService(service.ziku_name, service.ziku_dir)
            detail = reloaded.get_variant(variant_id)
            reviewed_path = Path(reloaded.get_workflow_dirs()["手工审核"]) / detail["审核文件"]
            self.assertTrue(reviewed_path.is_file())
            self.assertEqual(detail["状态"], config.STATUS_REVIEWED)
            self.assertEqual(detail["变换参数"]["偏移X"], 0)
            self.assertEqual(detail["变换参数"]["缩放"], 1.0)
            self.assertEqual(detail["变换参数"]["拉伸W"], 1.0)
            self.assertEqual(detail["变换参数"]["拉伸H"], 1.0)
            self.assertEqual(detail["变换参数"]["扭曲"], [0.0] * 8)
            self.assertIn("图像原点", detail["变换参数"])
            self.assertGreaterEqual(reviewed_path.stat().st_size, 1)
            self.assertEqual(page._status_label.text(), STATUS_REVIEWED)
            self.assertEqual(
                page._draft_status_label.text(),
                STATUS_REVIEWED,
            )
            populate.assert_not_called()
            advance.assert_called_once()
            page.deleteLater()

    def test_save_button_path_runs_in_background_and_merges_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, variants = self._build_library(Path(directory))
            variant_id = variants["eligible"][0]
            page = ReviewPage()
            self.assertTrue(page.open_library(service.ziku_dir, variant_id))
            page._canvas.set_transform(x=6)

            page._start_save_current()
            deadline = time.monotonic() + 10
            while page._save_running and time.monotonic() < deadline:
                self.app.processEvents()
                QTest.qWait(10)

            self.assertFalse(page._save_running)
            reloaded = GlyphService(service.ziku_name, service.ziku_dir)
            detail = reloaded.get_variant(variant_id)
            reviewed_path = (
                Path(reloaded.get_workflow_dirs()["手工审核"])
                / str(detail["审核文件"])
            )
            self.assertTrue(reviewed_path.is_file())
            self.assertEqual(page._source_stage, "手工审核稿")
            self.assertFalse(page._canvas.is_dirty)
            page.deleteLater()

    def test_worker_reuse_path_holds_library_lock_while_saving_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, variants = self._build_library(Path(directory))
            variant_id = str(variants["eligible"][0])
            page = ReviewPage()
            self.assertTrue(page.open_library(service.ziku_dir, variant_id))
            self.assertTrue(page.save_current())
            reusable = page._reusable_review_file()
            self.assertIsNotNone(reusable)
            if reusable is None:
                self.fail("测试准备失败：审核稿不符合复用条件")
            detail = page._service.get_variant(variant_id) if page._service else None
            filename = str(detail.get("审核文件", "")) if detail else ""
            lock = MagicMock()

            with patch.object(
                review_page_module,
                "acquire_batch_library_lock",
                return_value=lock,
            ) as acquire:
                result = review_page_module._save_interactive_review_in_worker(
                    service.ziku_name,
                    service.ziku_dir,
                    variant_id,
                    page._canvas.image().copy(),
                    filename,
                    (0, 0),
                    300,
                    approve=True,
                    reusable=(reusable[0], reusable[1], True),
                )

            acquire.assert_called_once_with(os.path.abspath(service.ziku_dir))
            lock.release.assert_called_once_with()
            self.assertTrue(result["审核通过"])
            self.assertEqual(
                result["字形状态"]["变体详情"]["状态"],
                config.STATUS_REVIEWED,
            )
            page.deleteLater()

    def test_approving_last_search_result_shows_end_notice(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, variants = self._build_library(Path(directory))
            variant_id = str(variants["pending"])
            page = ReviewPage()
            self.assertTrue(page.open_library(service.ziku_dir, variant_id))
            page._search_edit.setText("何")
            page._execute_search()
            self.assertEqual(page._variant_ids, [variant_id])

            with patch.object(QMessageBox, "information") as information:
                page.approve_current()

            information.assert_called_once()
            self.assertEqual(information.call_args.args[1], "手工审核")
            self.assertIn("最后一条", information.call_args.args[2])
            self.assertIn("全部处理完成", information.call_args.args[2])
            page.deleteLater()

    def test_approve_reuses_only_compliant_review_file_and_saves_state_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, variants = self._build_library(Path(directory))
            variant_id = str(variants["pending"])
            page = ReviewPage()
            self.assertTrue(page.open_library(service.ziku_dir, variant_id))
            self.assertTrue(page.save_current())
            if page._service is None:
                self.fail("手工审核页面应已载入字库服务")
            page._service.get_variant(variant_id)["手工编辑"] = "旧版异常值"

            with (
                patch.object(page, "_save_current_image", wraps=page._save_current_image) as image_save,
                patch.object(page._service, "save", wraps=page._service.save) as state_save,
            ):
                page.approve_current()

            image_save.assert_not_called()
            state_save.assert_called_once()
            self.assertEqual(
                page._service.get_variant(variant_id)["状态"],
                config.STATUS_REVIEWED,
            )
            page.deleteLater()

    def test_approve_regenerates_review_file_when_canvas_contract_changed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, variants = self._build_library(Path(directory))
            variant_id = str(variants["pending"])
            page = ReviewPage()
            self.assertTrue(page.open_library(service.ziku_dir, variant_id))
            self.assertTrue(page.save_current())
            if page._service is None:
                self.fail("手工审核页面应已载入字库服务")
            detail = page._service.get_variant(variant_id)
            review_path = (
                Path(page._service.get_workflow_dirs()["手工审核"])
                / str(detail["审核文件"])
            )
            wrong_size = QImage(QSize(32, 32), QImage.Format.Format_ARGB32)
            wrong_size.fill(QColor(0, 0, 0, 0))
            painter = QPainter(wrong_size)
            painter.fillRect(8, 5, 16, 22, QColor("#111111"))
            painter.end()
            wrong_size.setDotsPerMeterX(round(300 / 0.0254))
            wrong_size.setDotsPerMeterY(round(300 / 0.0254))
            self.assertTrue(wrong_size.save(str(review_path), "PNG"))
            detail["审核MD5"] = review_page_module._file_md5(str(review_path))

            with patch.object(
                page,
                "_save_current_image",
                wraps=page._save_current_image,
            ) as image_save:
                page.approve_current()

            image_save.assert_called_once_with(approve=True)
            reloaded = GlyphService(service.ziku_name, service.ziku_dir)
            saved_detail = reloaded.get_variant(variant_id)
            saved_path = (
                Path(reloaded.get_workflow_dirs()["手工审核"])
                / str(saved_detail["审核文件"])
            )
            self.assertEqual(QImage(str(saved_path)).size(), QSize(64, 64))
            self.assertEqual(saved_detail["状态"], config.STATUS_REVIEWED)
            page.deleteLater()

    def test_save_failure_restores_review_file_finished_file_and_library_state(self) -> None:
        """JSON 写入失败时不得留下半完成的人工稿或失效成品。"""
        with tempfile.TemporaryDirectory() as directory:
            service, variants = self._build_library(Path(directory))
            variant_id = variants["eligible"][1]
            detail = service.get_variant(variant_id)
            directories = service.get_workflow_dirs()
            reviewed_path = Path(directories["手工审核"]) / str(detail["审核文件"])
            finished_path = Path(directories["成品"]) / "是-0001.png"
            self._write_glyph(finished_path)
            detail.update(
                {
                    "状态": config.STATUS_FINISHED,
                    "成品文件": finished_path.name,
                    "成品MD5": "finished-before-save",
                    "整体协调参数": {"整体变换": {"移动X": 2.0}},
                }
            )
            service.set_coordination_summary(
                {"参考字": "是"},
                ink_baseline=128.0,
                geometry_completed=True,
                ink_completed=True,
                ink_enabled=True,
            )
            service.save()

            page = ReviewPage()
            self.assertTrue(page.open_library(service.ziku_dir, variant_id))
            page._canvas.set_transform(x=7, rotation=8)
            self.assertTrue(page._canvas.is_dirty)

            if page._service is None:
                self.fail("手工审核页面应已载入字库服务")
            state_before = page._service.snapshot_state()
            database_before = LibraryDatabase.open(service.ziku_dir).load_data()
            reviewed_before = reviewed_path.read_bytes()
            finished_before = finished_path.read_bytes()

            with (
                patch.object(page._service, "save", side_effect=OSError("模拟 JSON 写入失败")),
                patch("ui.pages.review_page.QMessageBox.critical") as critical,
            ):
                saved = page.save_current()

            self.assertFalse(saved)
            critical.assert_called_once()
            self.assertTrue(page._canvas.is_dirty)
            self.assertEqual(page._service.snapshot_state(), state_before)
            self.assertEqual(
                LibraryDatabase.open(service.ziku_dir).load_data(),
                database_before,
            )
            self.assertEqual(reviewed_path.read_bytes(), reviewed_before)
            self.assertEqual(finished_path.read_bytes(), finished_before)
            self.assertFalse(
                list(Path(directories["手工审核"]).glob(".fonteditor_review_*"))
            )
            self.assertFalse(
                list(Path(directories["成品"]).glob(".fonteditor_finished_rollback_*"))
            )

            reloaded = GlyphService(service.ziku_name, service.ziku_dir)
            self.assertEqual(reloaded.get_variant(variant_id), state_before["变体详情"][variant_id])
            self.assertEqual(reloaded.get_coordination_summary(), state_before["整体协调"])
            page.deleteLater()

    def test_save_rollback_failure_preserves_only_recoverable_review_backup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, variants = self._build_library(Path(directory))
            variant_id = str(variants["reviewed"])
            detail = service.get_variant(variant_id)
            review_dir = Path(service.get_workflow_dirs()["手工审核"])
            reviewed_path = review_dir / str(detail["审核文件"])
            reviewed_before = reviewed_path.read_bytes()

            page = ReviewPage()
            self.assertTrue(page.open_library(service.ziku_dir, variant_id))
            if page._service is None:
                self.fail("手工审核页面应已载入字库服务")
            real_replace = os.replace

            def fail_review_restore(source: str, destination: str) -> None:
                if Path(source).name.startswith(".fonteditor_review_rollback_"):
                    raise OSError("模拟旧审核稿恢复失败")
                real_replace(source, destination)

            original_begin = review_page_module.FileTransaction.begin

            def begin_with_failed_restore(root, changes, old_state):
                return original_begin(
                    root,
                    changes,
                    old_state,
                    replace_func=fail_review_restore,
                )

            with (
                patch.object(page._service, "save", side_effect=OSError("模拟 JSON 写入失败")),
                patch.object(
                    review_page_module.FileTransaction,
                    "begin",
                    side_effect=begin_with_failed_restore,
                ),
                patch("ui.pages.review_page.QMessageBox.critical") as critical,
            ):
                saved = page.save_current()

            self.assertFalse(saved)
            backups = list(review_dir.glob(".fonteditor_review_rollback_*.png"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), reviewed_before)
            manifests = list(
                (Path(service.ziku_dir) / ".fonteditor_file_transactions").glob("*.json")
            )
            self.assertEqual(len(manifests), 1)
            self.assertIn("图片事务回滚未完全完成", critical.call_args.args[2])
            page.deleteLater()

    def test_current_blank_draft_cannot_be_saved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, variants = self._build_library(Path(directory))
            variant_id = str(variants["pending"])
            page = ReviewPage()
            self.assertTrue(page.open_library(service.ziku_dir, variant_id))
            blank = QImage(QSize(64, 64), QImage.Format.Format_ARGB32)
            blank.fill(QColor(255, 255, 255, 255))
            page._canvas.set_image(blank, (64, 64))

            with patch("ui.pages.review_page.QMessageBox.warning") as warning:
                saved = page.save_current()

            self.assertFalse(saved)
            warning.assert_called_once()
            self.assertIn("没有有效文字前景", warning.call_args.args[2])
            reloaded = GlyphService(service.ziku_name, service.ziku_dir)
            self.assertEqual(
                reloaded.get_variant(variant_id)["状态"],
                config.STATUS_PENDING_MANUAL_REVIEW,
            )
            page.deleteLater()

    @staticmethod
    def _build_library(root: Path) -> tuple[GlyphService, dict[str, object]]:
        library_path = root / "审核测试库"
        service = GlyphService("审核测试库", str(library_path))
        service.ensure_dirs()
        service.init_metadata(dpi=300, canvas_w=64, canvas_h=64)
        directories = service.get_workflow_dirs()

        pending_id = service.add_original("何", "何-0001.png", "何-0001.png", "md5-a")
        reviewed_id = service.add_original("是", "是-0001.png", "是-0001.png", "md5-b")
        finished_id = service.add_original("完", "完-0001.png", "完-0001.png", "md5-e")
        optimization_id = service.add_original("无", "无-0001.png", "无-0001.png", "md5-c")
        missing_id = service.add_original("缺", "缺-0001.png", "缺-0001.png", "md5-d")

        ReviewPageTests._write_glyph(Path(directories["优化预览"]) / "何-0001.png")
        ReviewPageTests._write_glyph(Path(directories["手工审核"]) / "是-0001.png")
        ReviewPageTests._write_glyph(Path(directories["手工审核"]) / "完-0001.png")
        service.update_variant(
            pending_id,
            **{"状态": config.STATUS_PENDING_MANUAL_REVIEW, "中间文件": "何-0001.png"},
        )
        service.update_variant(
            reviewed_id,
            **{"状态": config.STATUS_REVIEWED, "审核文件": "是-0001.png"},
        )
        service.update_variant(
            finished_id,
            **{"状态": config.STATUS_FINISHED, "审核文件": "完-0001.png"},
        )
        service.update_variant(
            missing_id,
            **{"状态": config.STATUS_PENDING_MANUAL_REVIEW, "中间文件": "不存在.png"},
        )
        service.save()
        return service, {
            "eligible": [pending_id, reviewed_id, finished_id],
            "pending": pending_id,
            "reviewed": reviewed_id,
            "finished": finished_id,
            "pending_optimization": optimization_id,
            "missing_preview": missing_id,
        }

    @staticmethod
    def _build_large_library(
        root: Path,
        *,
        total: int,
    ) -> tuple[GlyphService, list[str]]:
        library_path = root / "审核大字库"
        service = GlyphService("审核大字库", str(library_path))
        service.ensure_dirs()
        service.init_metadata(dpi=300, canvas_w=64, canvas_h=64)
        preview_dir = Path(service.get_workflow_dirs()["优化预览"])
        variant_ids: list[str] = []
        for index in range(total):
            filename = f"甲-{index + 1:04d}.png"
            variant_id = service.add_original(
                "甲",
                filename,
                filename,
                f"large-review-{index:04d}",
            )
            ReviewPageTests._write_glyph(preview_dir / filename)
            service.update_variant(
                variant_id,
                **{
                    "状态": config.STATUS_PENDING_MANUAL_REVIEW,
                    "中间文件": filename,
                },
            )
            variant_ids.append(variant_id)
        service.save()
        return service, variant_ids

    @staticmethod
    def _write_glyph(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        image = QImage(QSize(64, 64), QImage.Format.Format_ARGB32)
        image.fill(QColor(0, 0, 0, 0))
        painter = QPainter(image)
        painter.fillRect(20, 8, 24, 48, QColor("#111111"))
        painter.end()
        if not image.save(str(path), "PNG"):
            raise AssertionError(f"测试图像写入失败：{path}")

    @staticmethod
    def _write_positioned_glyph(path: Path, left: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        image = QImage(QSize(64, 64), QImage.Format.Format_ARGB32)
        image.fill(QColor(0, 0, 0, 0))
        painter = QPainter(image)
        painter.fillRect(left, 12, 8, 36, QColor("#111111"))
        painter.end()
        if not image.save(str(path), "PNG"):
            raise AssertionError(f"测试图像写入失败：{path}")


if __name__ == "__main__":
    unittest.main()
