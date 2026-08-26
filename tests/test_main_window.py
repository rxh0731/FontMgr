"""主窗口页面导航信号契约回归测试。"""

from __future__ import annotations

import unittest
import tempfile
from collections.abc import Callable
from unittest.mock import MagicMock, call, patch

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from services.glyph_service import GlyphService
from services.settings_service import (
    PERFORMANCE_AUTO,
    PERFORMANCE_CONSERVATIVE,
    ApplicationSettings,
)
from ui.main_window import (
    LibraryScanResult,
    MainWindow,
    _connect_manual_review_navigation,
    _move_library_to_recycle_bin,
)
from ui.pages.home_page import LibraryScanProgress
from ui.workers import FunctionWorker


class _SignalStub:
    def __init__(self) -> None:
        self._callback: Callable[[str], None] | None = None

    def connect(self, callback: Callable[[str], None]) -> None:
        self._callback = callback

    def emit(self, library_path: str) -> None:
        if self._callback is None:
            raise AssertionError("手工审核信号尚未连接")
        self._callback(library_path)


class _HomePageStub:
    def __init__(self) -> None:
        self.manual_review_requested = _SignalStub()


class _LibraryScanHomeStub:
    def __init__(self) -> None:
        self.loading_updates: list[bool] = []
        self.progress_updates: list[LibraryScanProgress] = []
        self.library_results: list[list[dict[str, object]]] = []
        self.selected_libraries: list[str] = []
        self.loading_failures = 0

    def set_loading(self, loading: bool) -> None:
        self.loading_updates.append(loading)

    def set_scan_progress(self, progress: LibraryScanProgress) -> None:
        self.progress_updates.append(progress)

    def set_libraries(self, libraries: list[dict[str, object]]) -> None:
        self.library_results.append(libraries)

    def select_library(self, library_path: str) -> bool:
        self.selected_libraries.append(library_path)
        return True

    def set_loading_failed(self) -> None:
        self.loading_failures += 1


class _LibraryScanWindowStub:
    PAGE_HOME = MainWindow.PAGE_HOME
    refresh_libraries = MainWindow.refresh_libraries
    _start_library_scan = MainWindow._start_library_scan
    _library_scan_progress = MainWindow._library_scan_progress
    _library_scan_finished = MainWindow._library_scan_finished
    _library_scan_failed = MainWindow._library_scan_failed
    _release_worker = MainWindow._release_worker
    _normalized_library_path = staticmethod(MainWindow._normalized_library_path)
    _sync_library_summary = MainWindow._sync_library_summary
    _apply_library_summaries = MainWindow._apply_library_summaries
    _remove_library_summary = MainWindow._remove_library_summary
    _load_library_summary_cache = MainWindow._load_library_summary_cache
    _request_library_reconciliation = MainWindow._request_library_reconciliation

    def __init__(self, generation: int = 0) -> None:
        self._library_scan_generation = generation
        self._library_scan_active = False
        self._library_scan_retrying_unstable = False
        self._library_refresh_force_pending = False
        self._library_cache_ready = False
        self._library_cache_signature: object | None = None
        self._library_cache_summaries: list[dict[str, object]] = []
        self._library_summary_store = _SummaryStoreStub()
        self._summary_store_warning_shown = False
        self._home_page = _LibraryScanHomeStub()
        self._thread_pool = _ThreadPoolStub()
        self._workers: set[object] = set()
        self._export_page = None
        self._stack = _StackStub()
        self._status_bar = _StatusBarStub()
        self.scan_errors: list[str] = []

    def statusBar(self) -> _StatusBarStub:
        return self._status_bar

    def _show_scan_error(self, message: str) -> None:
        self.scan_errors.append(message)


class _ThreadPoolStub:
    def __init__(self) -> None:
        self.started: list[object] = []

    def start(self, worker: object) -> None:
        self.started.append(worker)


class _SummaryStoreStub:
    def __init__(self) -> None:
        self.load_result: list[dict[str, object]] | None = None
        self.loaded_signatures: list[object] = []
        self.saved: list[tuple[list[dict[str, object]], object]] = []

    def load(self, signature: object) -> list[dict[str, object]] | None:
        self.loaded_signatures.append(signature)
        return self.load_result

    def save(
        self,
        summaries: list[dict[str, object]],
        signature: object,
    ) -> None:
        self.saved.append((summaries, signature))


class _ReviewPageStub:
    def __init__(self, open_result: bool) -> None:
        self.open_result = open_result
        self.opened_paths: list[str] = []
        self.cursor_shapes: list[Qt.CursorShape | None] = []

    def open_library(self, library_path: str) -> bool:
        self.opened_paths.append(library_path)
        cursor = QApplication.overrideCursor()
        self.cursor_shapes.append(cursor.shape() if cursor is not None else None)
        return self.open_result


class _StackStub:
    def __init__(
        self,
        current_index: int = MainWindow.PAGE_HOME,
        current_widget: object | None = None,
    ) -> None:
        self.current_index = current_index
        self.current_widget = current_widget
        self.requested_indices: list[int] = []
        self.added_widgets: list[object] = []
        self.removed_widgets: list[object] = []

    def setCurrentIndex(self, index: int) -> None:
        self.requested_indices.append(index)
        self.current_index = index

    def currentWidget(self) -> object | None:
        return self.current_widget

    def addWidget(self, widget: object) -> None:
        self.added_widgets.append(widget)

    def removeWidget(self, widget: object) -> None:
        self.removed_widgets.append(widget)
        if self.current_widget is widget:
            self.current_widget = None

    def setCurrentWidget(self, widget: object) -> None:
        self.current_widget = widget


class _StatusBarStub:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def showMessage(self, message: str) -> None:
        self.messages.append(message)


class _MainWindowStub:
    PAGE_REVIEW = MainWindow.PAGE_REVIEW

    def __init__(self, review_page: _ReviewPageStub) -> None:
        self._review_page = review_page
        self._stack = _StackStub()
        self._status_bar = _StatusBarStub()

    def statusBar(self) -> _StatusBarStub:
        return self._status_bar


class _ConsistencyPageStub:
    def __init__(self, confirm_result: bool, is_batch_running: bool = False) -> None:
        self.confirm_result = confirm_result
        self.is_batch_running = is_batch_running
        self.confirm_calls = 0

    def _confirm_leave_page(self) -> bool:
        self.confirm_calls += 1
        return self.confirm_result


class _BatchPageStub:
    def __init__(self, is_batch_running: bool = False) -> None:
        self.is_batch_running = is_batch_running


class _CloseEventStub:
    def __init__(self) -> None:
        self.accepted = False
        self.ignored = False

    def accept(self) -> None:
        self.accepted = True

    def ignore(self) -> None:
        self.ignored = True


class _ClosingWindowStub:
    def __init__(
        self,
        consistency_page: _ConsistencyPageStub,
        *,
        optimization_running: bool = False,
        review_running: bool = False,
        consistency_running: bool = False,
    ) -> None:
        consistency_page.is_batch_running = consistency_running
        self._consistency_page = consistency_page
        self._optimization_page = _BatchPageStub(optimization_running)
        self._review_page = _BatchPageStub(review_running)
        self._export_page = None
        self._stack = _StackStub(current_widget=consistency_page)


class _ExportPageStub:
    def __init__(self) -> None:
        self.shutdown_calls = 0
        self.delete_calls = 0

    def shutdown(self) -> None:
        self.shutdown_calls += 1

    def deleteLater(self) -> None:
        self.delete_calls += 1


class _HomeWindowStub:
    PAGE_HOME = MainWindow.PAGE_HOME

    def __init__(self, export_page: _ExportPageStub) -> None:
        self._export_page = export_page
        self._stack = _StackStub(current_widget=export_page)


class MainWindowNavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    """验证首页入口与功能页面使用一致的信号参数。"""

    def test_manual_review_signal_opens_selected_library(self) -> None:
        home_page = _HomePageStub()
        received_paths: list[str] = []
        _connect_manual_review_navigation(home_page, received_paths.append)

        home_page.manual_review_requested.emit("D:\\Code\\FontEditor-PySide6\\字库\\测试")

        self.assertEqual(
            received_paths,
            ["D:\\Code\\FontEditor-PySide6\\字库\\测试"],
        )

    def test_recycle_bin_worker_reports_stage_and_returns_identity(self) -> None:
        progress: list[str] = []
        with tempfile.TemporaryDirectory() as directory, patch(
            "ui.main_window.send2trash"
        ) as recycle:
            result = _move_library_to_recycle_bin(
                directory,
                "大字库",
                progress.append,
            )

        recycle.assert_called_once_with(directory)
        self.assertEqual(result, {"字库名称": "大字库", "字库路径": directory})
        self.assertEqual(progress, ["正在将字库“大字库”移入回收站…"])

    def test_delete_library_queues_background_worker_before_recycling(self) -> None:
        window = MagicMock()
        window._library_delete_active = False
        window._library_scan_active = False
        window._workers = set()
        window._thread_pool = _ThreadPoolStub()
        window._home_page = MagicMock()
        target = r"D:\字库\大字库"

        with (
            patch(
                "ui.main_window.resolve_library_directory",
                return_value=target,
            ),
            patch(
                "ui.main_window.QMessageBox.question",
                return_value=QMessageBox.StandardButton.Yes,
            ),
            patch("ui.main_window.send2trash") as recycle,
        ):
            MainWindow._delete_library(window, "大字库", target)

        self.assertTrue(window._library_delete_active)
        self.assertEqual(len(window._thread_pool.started), 1)
        self.assertEqual(len(window._workers), 1)
        window._home_page.set_deleting.assert_called_once_with(True, "大字库")
        recycle.assert_not_called()

    def test_library_scan_ignores_stale_progress_results_and_failures(self) -> None:
        window = _LibraryScanWindowStub(generation=2)
        stable_signature = object()
        progress = LibraryScanProgress(
            "processing",
            "当前字库",
            library_index=1,
            library_total=1,
            glyph_current=1,
            glyph_total=2,
        )

        MainWindow._library_scan_progress(window, 1, progress)
        MainWindow._library_scan_finished(
            window,
            1,
            LibraryScanResult(
                [{"name": "旧结果"}],
                stable_signature,
                stable_signature,
            ),
        )
        MainWindow._library_scan_failed(window, 1, "旧错误")
        self.assertEqual(window._home_page.progress_updates, [])
        self.assertEqual(window._home_page.library_results, [])
        self.assertEqual(window._home_page.loading_failures, 0)
        self.assertEqual(window.scan_errors, [])

        MainWindow._library_scan_progress(window, 2, progress)
        MainWindow._library_scan_finished(
            window,
            2,
            LibraryScanResult(
                [{"name": "当前结果"}],
                stable_signature,
                stable_signature,
            ),
        )
        self.assertEqual(window._home_page.progress_updates, [progress])
        self.assertEqual(
            window._home_page.library_results,
            [[{"name": "当前结果"}]],
        )

    def test_current_library_scan_failure_restores_home_loading_state(self) -> None:
        window = _LibraryScanWindowStub(generation=3)
        cached_signature = object()
        cached_summaries = [{"name": "已有缓存"}]
        window._library_scan_active = True
        window._library_cache_ready = True
        window._library_cache_signature = cached_signature
        window._library_cache_summaries = cached_summaries

        MainWindow._library_scan_failed(window, 3, "读取失败")

        self.assertEqual(window._home_page.loading_failures, 1)
        self.assertEqual(window.scan_errors, ["读取失败"])
        self.assertFalse(window._library_scan_active)
        self.assertTrue(window._library_cache_ready)
        self.assertIs(window._library_cache_signature, cached_signature)
        self.assertIs(window._library_cache_summaries, cached_summaries)

    def test_unchanged_home_cache_skips_loading_and_worker(self) -> None:
        """普通返回首页命中缓存时，不得再次显示进度或启动扫描。"""
        window = _LibraryScanWindowStub(generation=4)
        cached_signature = object()
        window._library_cache_ready = True
        window._library_cache_signature = cached_signature

        with patch(
            "ui.main_window.library_summary_signature",
            return_value=cached_signature,
        ):
            started = MainWindow.refresh_libraries(window)

        self.assertFalse(started)
        self.assertEqual(window._library_scan_generation, 4)
        self.assertEqual(window._home_page.loading_updates, [])
        self.assertEqual(window._thread_pool.started, [])

    def test_startup_uses_persistent_summary_when_signature_matches(self) -> None:
        """启动索引可信时直接显示摘要，不启动后台深度核对。"""
        window = _LibraryScanWindowStub(generation=0)
        signature = object()
        summaries = [
            {"name": "已有字库", "path": r"D:\字库\已有字库", "variants": 12}
        ]
        window._library_summary_store.load_result = summaries

        with patch(
            "ui.main_window.library_summary_signature",
            return_value=signature,
        ):
            MainWindow._load_library_summary_cache(window)

        self.assertEqual(window._library_summary_store.loaded_signatures, [signature])
        self.assertTrue(window._library_cache_ready)
        self.assertIs(window._library_cache_signature, signature)
        self.assertIs(window._library_cache_summaries, summaries)
        self.assertEqual(window._home_page.library_results, [summaries])
        self.assertEqual(window._home_page.loading_updates, [])
        self.assertEqual(window._thread_pool.started, [])

    def test_startup_scans_when_persistent_summary_is_not_trusted(self) -> None:
        """索引缺失或签名不匹配时，启动阶段执行一次深度核对。"""
        window = _LibraryScanWindowStub(generation=0)
        signature = object()

        with patch(
            "ui.main_window.library_summary_signature",
            return_value=signature,
        ):
            MainWindow._load_library_summary_cache(window)

        self.assertEqual(window._library_summary_store.loaded_signatures, [signature])
        self.assertTrue(window._library_scan_active)
        self.assertEqual(window._library_scan_generation, 1)
        self.assertEqual(window._home_page.loading_updates, [True])
        self.assertEqual(len(window._thread_pool.started), 1)

    def test_reconciliation_requires_explicit_confirmation(self) -> None:
        """重新核对入口中，取消不扫描，明确确认才扫描。"""
        for confirmed in (False, True):
            with self.subTest(confirmed=confirmed):
                window = _LibraryScanWindowStub(generation=2)
                window._library_cache_ready = True
                confirm_button = object()
                cancel_button = object()
                dialog = MagicMock()
                dialog.addButton.side_effect = [confirm_button, cancel_button]
                dialog.clickedButton.return_value = (
                    confirm_button if confirmed else cancel_button
                )

                with patch("ui.main_window.QMessageBox", return_value=dialog):
                    MainWindow._request_library_reconciliation(window)

                dialog.exec.assert_called_once_with()
                self.assertEqual(
                    len(window._thread_pool.started),
                    1 if confirmed else 0,
                )
                self.assertEqual(
                    window._library_scan_generation,
                    3 if confirmed else 2,
                )

    def test_force_refresh_bypasses_unchanged_home_cache(self) -> None:
        """用户点击刷新时，即使签名未变也必须启动一次后台核对。"""
        window = _LibraryScanWindowStub(generation=4)
        cached_signature = object()
        window._library_cache_ready = True
        window._library_cache_signature = cached_signature

        with patch(
            "ui.main_window.library_summary_signature",
            return_value=cached_signature,
        ):
            started = MainWindow.refresh_libraries(window, force=True)

        self.assertTrue(started)
        self.assertTrue(window._library_scan_active)
        self.assertEqual(window._library_scan_generation, 5)
        self.assertEqual(window._home_page.loading_updates, [True])
        self.assertEqual(len(window._thread_pool.started), 1)

    def test_changed_signature_does_not_scan_on_normal_home_return(self) -> None:
        """正常返回只显示实时摘要，不因文件签名变化启动扫描。"""
        window = _LibraryScanWindowStub(generation=4)
        window._library_cache_ready = True
        window._library_cache_signature = object()

        with patch(
            "ui.main_window.library_summary_signature",
            return_value=object(),
        ):
            started = MainWindow.refresh_libraries(window)

        self.assertFalse(started)
        self.assertFalse(window._library_scan_active)
        self.assertEqual(window._library_scan_generation, 4)
        self.assertEqual(window._home_page.loading_updates, [])
        self.assertEqual(window._thread_pool.started, [])

    def test_running_scan_coalesces_repeated_returns_and_forced_refreshes(self) -> None:
        """扫描期间的重复请求不得并行，强制刷新最多补跑一次。"""
        window = _LibraryScanWindowStub(generation=1)
        window._library_scan_active = True

        self.assertFalse(MainWindow.refresh_libraries(window))
        self.assertFalse(MainWindow.refresh_libraries(window, force=True))
        self.assertFalse(MainWindow.refresh_libraries(window, force=True))
        self.assertEqual(window._thread_pool.started, [])
        self.assertTrue(window._library_refresh_force_pending)

        stable_signature = object()
        MainWindow._library_scan_finished(
            window,
            1,
            LibraryScanResult(
                [{"name": "稳定结果"}],
                stable_signature,
                stable_signature,
            ),
        )

        self.assertEqual(
            window._home_page.library_results,
            [[{"name": "稳定结果"}]],
        )
        self.assertTrue(window._library_scan_active)
        self.assertFalse(window._library_refresh_force_pending)
        self.assertEqual(window._library_scan_generation, 2)
        self.assertEqual(len(window._thread_pool.started), 1)

    def test_unstable_scan_result_is_not_cached_and_retries_once(self) -> None:
        """扫描期间数据变化时应丢弃混合结果，并自动补扫一次。"""
        window = _LibraryScanWindowStub(generation=7)
        window._library_scan_active = True

        MainWindow._library_scan_finished(
            window,
            7,
            LibraryScanResult(
                [{"name": "混合结果"}],
                object(),
                object(),
            ),
        )

        self.assertFalse(window._library_cache_ready)
        self.assertEqual(window._home_page.library_results, [])
        self.assertTrue(window._library_scan_active)
        self.assertTrue(window._library_scan_retrying_unstable)
        self.assertEqual(window._library_scan_generation, 8)
        self.assertEqual(len(window._thread_pool.started), 1)

        MainWindow._library_scan_finished(
            window,
            8,
            LibraryScanResult(
                [{"name": "仍不稳定"}],
                object(),
                object(),
            ),
        )

        self.assertFalse(window._library_scan_active)
        self.assertEqual(len(window._thread_pool.started), 1)
        self.assertEqual(window._home_page.loading_failures, 1)
        self.assertEqual(
            window.scan_errors,
            ["扫描期间字库数据持续变化，请稍后点击重新核对重试"],
        )

    def test_import_completion_updates_summary_without_scan(self) -> None:
        """导入完成后直接写入实时摘要，返回首页不得启动扫描。"""
        window = _LibraryScanWindowStub(generation=2)
        window._library_cache_ready = True
        window._library_cache_signature = object()
        glyph = object.__new__(GlyphService)
        glyph.ziku_name = "新字库"
        glyph.ziku_dir = r"D:\字库\新字库"
        summary = {
            "name": "新字库",
            "path": glyph.ziku_dir,
            "variants": 12,
        }

        with (
            patch("ui.main_window.GlyphService.open", return_value=glyph),
            patch("ui.main_window.summarize_glyph_service", return_value=summary),
            patch("ui.main_window.library_summary_signature", return_value=object()),
        ):
            MainWindow._import_completed(
                window,
                "create",
                glyph.ziku_dir,
                object(),
            )
            MainWindow.show_home(window)

        self.assertTrue(window._library_cache_ready)
        self.assertEqual(window._library_cache_summaries, [summary])
        self.assertEqual(window._home_page.selected_libraries, [glyph.ziku_dir])
        self.assertEqual(window._thread_pool.started, [])
        self.assertEqual(window._library_scan_generation, 2)

    def test_append_completion_keeps_existing_home_selection(self) -> None:
        """已有字库追加完成时只更新摘要，不强制切换首页选择。"""

        window = _LibraryScanWindowStub(generation=2)
        window._library_cache_ready = True
        glyph = object.__new__(GlyphService)
        glyph.ziku_name = "已有字库"
        glyph.ziku_dir = r"D:\字库\已有字库"
        old_summary = {
            "name": "已有字库",
            "path": glyph.ziku_dir,
            "variants": 10,
        }
        new_summary = dict(old_summary, variants=12)
        window._library_cache_summaries = [old_summary]

        with (
            patch("ui.main_window.GlyphService.open", return_value=glyph),
            patch(
                "ui.main_window.summarize_glyph_service",
                return_value=new_summary,
            ),
            patch(
                "ui.main_window.library_summary_signature",
                return_value=object(),
            ),
        ):
            MainWindow._import_completed(
                window,
                "已有字库",
                glyph.ziku_dir,
                object(),
            )

        self.assertEqual(window._library_cache_summaries, [new_summary])
        self.assertEqual(window._home_page.selected_libraries, [])
        self.assertEqual(window._thread_pool.started, [])

    def test_name_only_entry_points_do_not_run_full_summary_scan(self) -> None:
        """新建、追加和参数对话框只需轻量枚举已有字库名称。"""
        window = MagicMock()
        window.PAGE_IMPORT = MainWindow.PAGE_IMPORT
        glyph_service = object()
        dialog = MagicMock()

        with (
            patch(
                "ui.main_window.scan_library_names",
                return_value=["甲字库", "乙字库"],
            ) as scan_names,
            patch("ui.main_window.scan_library_summaries") as scan_summaries,
            patch("ui.main_window.GlyphService.open", return_value=glyph_service),
            patch(
                "ui.main_window.resolve_library_directory",
                side_effect=lambda _root, path, **_kwargs: path,
            ),
            patch("ui.main_window.set_last_ziku_path"),
            patch(
                "ui.main_window.LibraryParametersDialog",
                return_value=dialog,
            ),
        ):
            MainWindow.show_create_page(window)
            MainWindow._open_stage(
                window,
                "import",
                "甲字库",
                r"D:\字库\甲字库",
            )
            MainWindow._edit_library_parameters(window, r"D:\字库\甲字库")

        self.assertEqual(scan_names.call_count, 3)
        scan_summaries.assert_not_called()
        window._import_page.configure_create.assert_called_once_with(
            ["甲字库", "乙字库"]
        )
        window._import_page.configure_append.assert_called_once_with(
            glyph_service,
            ["甲字库", "乙字库"],
        )
        dialog.exec.assert_called_once_with()

    def test_function_worker_can_inject_progress_callback(self) -> None:
        progress_updates: list[str] = []
        results: list[str] = []

        def run(progress_callback: Callable[[object], None]) -> str:
            progress_callback("正在核对")
            return "完成"

        worker = FunctionWorker(run, with_progress=True)
        worker.signals.progress.connect(progress_updates.append)
        worker.signals.finished.connect(results.append)
        worker.run()

        self.assertEqual(progress_updates, ["正在核对"])
        self.assertEqual(results, ["完成"])

    def test_cancelled_library_switch_keeps_current_page(self) -> None:
        """审核页拒绝切库时，主窗口不得切页或记录新字库。"""
        library_path = "D:\\Code\\FontEditor-PySide6\\字库\\另一个字库"
        review_page = _ReviewPageStub(open_result=False)
        window = _MainWindowStub(review_page)

        with (
            patch("ui.main_window.set_last_ziku_path") as set_last_path,
            patch(
                "ui.main_window.resolve_library_directory",
                return_value=library_path,
            ),
        ):
            MainWindow.open_library(window, library_path)

        self.assertEqual(review_page.opened_paths, [library_path])
        self.assertEqual(review_page.cursor_shapes, [Qt.CursorShape.WaitCursor])
        self.assertIsNone(QApplication.overrideCursor())
        self.assertEqual(window._stack.current_index, MainWindow.PAGE_HOME)
        self.assertEqual(window._stack.requested_indices, [])
        self.assertEqual(window._status_bar.messages, [])
        set_last_path.assert_not_called()

    def test_close_cancel_on_dirty_consistency_page_keeps_window_open(self) -> None:
        """整体协调取消离开时，主窗口关闭事件必须被拒绝。"""
        page = _ConsistencyPageStub(confirm_result=False)
        window = _ClosingWindowStub(page)
        event = _CloseEventStub()

        MainWindow.closeEvent(window, event)

        self.assertEqual(page.confirm_calls, 1)
        self.assertTrue(event.ignored)
        self.assertFalse(event.accepted)

    def test_close_clean_consistency_page_is_allowed(self) -> None:
        """整体协调没有未保存修改时，主窗口可以正常关闭。"""
        page = _ConsistencyPageStub(confirm_result=True)
        window = _ClosingWindowStub(page)
        event = _CloseEventStub()

        MainWindow.closeEvent(window, event)

        self.assertEqual(page.confirm_calls, 1)
        self.assertTrue(event.accepted)
        self.assertFalse(event.ignored)

    def test_close_is_rejected_while_bulk_optimization_is_running(self) -> None:
        """整库自动优化运行期间，标题栏关闭和 Alt+F4 必须被拒绝。"""
        page = _ConsistencyPageStub(confirm_result=True)
        window = _ClosingWindowStub(page, optimization_running=True)
        event = _CloseEventStub()

        with patch("ui.main_window.QMessageBox.information") as information:
            MainWindow.closeEvent(window, event)

        information.assert_called_once_with(
            window,
            "批量任务正在执行",
            "整库自动优化正在执行。如需提前退出，请先点击“停止批量优化”，"
            "等待任务安全停止后再关闭程序。",
        )
        self.assertEqual(page.confirm_calls, 0)
        self.assertTrue(event.ignored)
        self.assertFalse(event.accepted)

    def test_close_is_rejected_while_bulk_review_is_running(self) -> None:
        """整库手工审核运行期间，标题栏关闭和 Alt+F4 必须被拒绝。"""
        page = _ConsistencyPageStub(confirm_result=True)
        window = _ClosingWindowStub(page, review_running=True)
        event = _CloseEventStub()

        with patch("ui.main_window.QMessageBox.information") as information:
            MainWindow.closeEvent(window, event)

        information.assert_called_once_with(
            window,
            "批量任务正在执行",
            "整库手工审核正在执行。如需提前退出，请先点击“停止批量审核”，"
            "等待任务安全停止后再关闭程序。",
        )
        self.assertEqual(page.confirm_calls, 0)
        self.assertTrue(event.ignored)
        self.assertFalse(event.accepted)

    def test_close_is_rejected_while_bulk_coordination_is_running(self) -> None:
        """整库整体协调运行期间，标题栏关闭和 Alt+F4 必须被拒绝。"""
        page = _ConsistencyPageStub(confirm_result=True)
        window = _ClosingWindowStub(page, consistency_running=True)
        event = _CloseEventStub()

        with patch("ui.main_window.QMessageBox.information") as information:
            MainWindow.closeEvent(window, event)

        information.assert_called_once_with(
            window,
            "批量任务正在执行",
            "整库整体协调正在执行。如需提前退出，请先点击“停止整体协调”，"
            "等待任务安全停止后再关闭程序。",
        )
        self.assertEqual(page.confirm_calls, 0)
        self.assertTrue(event.ignored)
        self.assertFalse(event.accepted)

    def test_returning_home_releases_export_page_and_preview_cache(self) -> None:
        """离开导出页时必须停止后台预取并释放动态页面。"""
        page = _ExportPageStub()
        window = _HomeWindowStub(page)

        MainWindow.show_home(window)

        self.assertEqual(page.shutdown_calls, 1)
        self.assertEqual(page.delete_calls, 1)
        self.assertEqual(window._stack.removed_widgets, [page])
        self.assertIsNone(window._export_page)
        self.assertEqual(window._stack.requested_indices, [MainWindow.PAGE_HOME])

    def test_returning_home_destroys_review_state_before_next_library(self) -> None:
        """手工审核返回首页后不得把检索文字带入下一次页面实例。"""

        with patch(
            "ui.main_window.LibrarySummaryStore.load",
            return_value=[],
        ):
            window = MainWindow()
        try:
            old_page = window._ensure_review_page()
            old_page._search_edit.setText("上一次检索")
            window._stack.setCurrentWidget(old_page)

            window.show_home()
            self.app.processEvents()

            self.assertIsNone(window._review_page)
            new_page = window._ensure_review_page()
            self.assertIsNot(new_page, old_page)
            self.assertEqual(new_page._search_edit.text(), "")
        finally:
            window.close()
            window.deleteLater()
            self.app.processEvents()

    def test_startup_keeps_large_feature_pages_unloaded(self) -> None:
        with patch(
            "ui.main_window.LibrarySummaryStore.load",
            return_value=[],
        ):
            window = MainWindow()
        try:
            self.assertIsNone(window._import_page)
            self.assertIsNone(window._review_page)
            self.assertIsNone(window._optimization_page)
            self.assertIsNone(window._scripture_layout_page)
            self.assertIsNone(window._custom_scripture_layout_page)
            self.assertIsNone(window._settings_page)
            self.assertIsNone(window._help_page)
            self.assertEqual(len(window._page_placeholders), 8)
        finally:
            window.close()
            window.deleteLater()
            self.app.processEvents()

    def test_settings_tool_opens_real_page(self) -> None:
        with patch(
            "ui.main_window.LibrarySummaryStore.load",
            return_value=[],
        ):
            window = MainWindow()
        try:
            window._open_tool("settings")

            self.assertIsNotNone(window._settings_page)
            self.assertIs(window._stack.currentWidget(), window._settings_page)
        finally:
            window.close()
            window.deleteLater()
            self.app.processEvents()

    def test_help_tool_opens_searchable_page(self) -> None:
        with patch(
            "ui.main_window.LibrarySummaryStore.load",
            return_value=[],
        ):
            window = MainWindow()
        try:
            window._open_tool("help")

            self.assertIsNotNone(window._help_page)
            self.assertIs(window._stack.currentWidget(), window._help_page)
            self.assertGreater(window._help_page._topic_list.count(), 10)
        finally:
            window.close()
            window.deleteLater()
            self.app.processEvents()

    def test_image_lab_opens_independent_workspace(self) -> None:
        with patch(
            "ui.main_window.LibrarySummaryStore.load",
            return_value=[],
        ):
            window = MainWindow()
            try:
                window._open_tool("image_lab")
                self.assertIsNotNone(window._image_lab_page)
                self.assertIs(window._stack.currentWidget(), window._image_lab_page)
                self.assertIsNone(window._image_lab_page._project)
            finally:
                window.close()
                window.deleteLater()
                self.app.processEvents()

    def test_performance_modes_only_adjust_global_pool_limit(self) -> None:
        window = MagicMock()
        window._automatic_thread_count = 8

        MainWindow._apply_performance_settings(
            window,
            ApplicationSettings(performance_mode=PERFORMANCE_AUTO),
        )
        MainWindow._apply_performance_settings(
            window,
            ApplicationSettings(performance_mode=PERFORMANCE_CONSERVATIVE),
        )

        self.assertEqual(
            window._thread_pool.setMaxThreadCount.call_args_list,
            [call(8), call(4)],
        )


if __name__ == "__main__":
    unittest.main()
