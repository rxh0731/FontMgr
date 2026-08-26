"""应用主窗口与页面导航。"""

from __future__ import annotations

import os
import gc
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QEventLoop, QThreadPool, QTimer, Qt
from PySide6.QtGui import QCloseEvent, QIcon
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QStackedWidget, QWidget
from send2trash import send2trash

import config
from data.config_store import set_last_ziku_path
from data.library_summary_store import LibrarySummaryStore
from services.glyph_service import GlyphService
from services.library_summary_service import summarize_glyph_service
from services.settings_service import (
    PERFORMANCE_CONSERVATIVE,
    ApplicationSettings,
    SettingsService,
)
from ui.dialogs.library_parameters_dialog import LibraryParametersDialog
from ui.pages.home_page import (
    HomePage,
    LibraryScanProgress,
    LibrarySummarySignature,
    library_summary_signature,
    scan_library_names,
    scan_library_summaries,
)
from ui.workers import FunctionWorker
from utils.application_identity import load_application_icon
from utils.file_utils import resolve_library_directory
from utils.file_utils import pinyin_natural_key

if TYPE_CHECKING:
    from ui.pages.consistency_page import ConsistencyPage
    from ui.pages.custom_scripture_layout_page import CustomScriptureLayoutPage
    from ui.pages.export_page import ExportPage
    from ui.pages.help_page import HelpPage
    from ui.pages.import_page import ImportPage
    from ui.pages.image_lab_page import ImageLabPage
    from ui.pages.optimization_page import OptimizationPage
    from ui.pages.review_page import ReviewPage
    from ui.pages.scripture_layout_page import ScriptureLayoutPage
    from ui.pages.text_statistics_page import TextStatisticsPage
    from ui.pages.settings_page import SettingsPage


def _connect_manual_review_navigation(
    home_page: HomePage,
    open_library: Callable[[str], None],
) -> None:
    """按单个字库路径参数连接首页与手工审核入口。"""
    home_page.manual_review_requested.connect(open_library)


@contextmanager
def _feature_startup_cursor():  # type: ignore[no-untyped-def]
    """功能页面同步初始化期间显示系统等待光标，并确保异常后恢复。"""

    application = QApplication.instance()
    if application is None:
        yield
        return
    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    application.processEvents(
        QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents
    )
    try:
        yield
    finally:
        QApplication.restoreOverrideCursor()


@dataclass(frozen=True, slots=True)
class LibraryScanResult:
    """一次首页摘要扫描及其前后数据签名。"""

    summaries: list[dict[str, Any]]
    signature_before: LibrarySummarySignature
    signature_after: LibrarySummarySignature


def _scan_library_summary_result(
    progress_callback: Callable[[LibraryScanProgress], None],
) -> LibraryScanResult:
    """扫描摘要，并检测扫描期间字库数据是否发生变化。"""
    signature_before = library_summary_signature()
    summaries = scan_library_summaries(progress_callback)
    signature_after = library_summary_signature()
    return LibraryScanResult(summaries, signature_before, signature_after)


def _move_library_to_recycle_bin(
    target: str,
    library_name: str,
    progress_callback: Callable[[str], None],
) -> dict[str, str]:
    """在后台把完整字库目录交给系统回收站。"""

    if not os.path.isdir(target):
        raise FileNotFoundError("字库目录不存在，可能已被其他程序移动或删除。")
    progress_callback(f"正在将字库“{library_name}”移入回收站…")
    send2trash(target)
    return {"字库名称": library_name, "字库路径": target}


class MainWindow(QMainWindow):
    """承载全部功能页面和全局导航状态。"""

    PAGE_HOME = 0
    PAGE_IMPORT = 1
    PAGE_REVIEW = 2
    PAGE_OPTIMIZATION = 3
    PAGE_SCRIPTURE_LAYOUT = 4
    PAGE_CUSTOM_SCRIPTURE_LAYOUT = 5
    PAGE_TEXT_STATISTICS = 6
    PAGE_SETTINGS = 7
    PAGE_HELP = 8

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("字库编辑器")
        application = QApplication.instance()
        window_icon = application.windowIcon() if application is not None else QIcon()
        if window_icon.isNull():
            window_icon = load_application_icon()
        if not window_icon.isNull():
            self.setWindowIcon(window_icon)
        self.setMinimumSize(1100, 720)
        self.resize(1380, 880)
        self._thread_pool = QThreadPool.globalInstance()
        self._automatic_thread_count = max(1, self._thread_pool.maxThreadCount())
        self._settings_service = SettingsService()
        try:
            self._apply_performance_settings(self._settings_service.load())
        except (OSError, RuntimeError, ValueError):
            self._apply_performance_settings(self._settings_service.defaults())
        self._workers: set[FunctionWorker] = set()
        self._library_scan_generation = 0
        self._library_scan_active = False
        self._library_scan_retrying_unstable = False
        self._library_refresh_force_pending = False
        self._library_delete_active = False
        self._library_delete_worker: FunctionWorker | None = None
        self._library_delete_name = ""
        self._library_delete_path = ""
        self._library_cache_ready = False
        self._library_cache_signature: LibrarySummarySignature | None = None
        self._library_cache_summaries: list[dict[str, Any]] = []
        self._library_summary_store = LibrarySummaryStore(
            config.LIBRARY_SUMMARY_CACHE_FILE,
            config.ZIKU_ROOT,
        )
        self._summary_store_warning_shown = False
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)
        self._page_placeholders: dict[str, QWidget] = {}

        self._home_page = HomePage()
        self._import_page: ImportPage | None = None
        self._review_page: ReviewPage | None = None
        self._optimization_page: OptimizationPage | None = None
        self._scripture_layout_page: ScriptureLayoutPage | None = None
        self._custom_scripture_layout_page: CustomScriptureLayoutPage | None = None
        self._text_statistics_page: TextStatisticsPage | None = None
        self._settings_page: SettingsPage | None = None
        self._help_page: HelpPage | None = None
        self._consistency_page: ConsistencyPage | None = None
        self._export_page: ExportPage | None = None
        self._image_lab_page: ImageLabPage | None = None
        self._stack.addWidget(self._home_page)
        for attribute, index in (
            ("_import_page", self.PAGE_IMPORT),
            ("_review_page", self.PAGE_REVIEW),
            ("_optimization_page", self.PAGE_OPTIMIZATION),
            ("_scripture_layout_page", self.PAGE_SCRIPTURE_LAYOUT),
            ("_custom_scripture_layout_page", self.PAGE_CUSTOM_SCRIPTURE_LAYOUT),
            ("_text_statistics_page", self.PAGE_TEXT_STATISTICS),
            ("_settings_page", self.PAGE_SETTINGS),
            ("_help_page", self.PAGE_HELP),
        ):
            placeholder = QWidget()
            placeholder.setObjectName(f"unloadedPage{index}")
            self._stack.addWidget(placeholder)
            self._page_placeholders[attribute] = placeholder

        self._home_page.tool_requested.connect(self._open_tool)
        self._home_page.stage_requested.connect(self._open_stage)
        _connect_manual_review_navigation(self._home_page, self.open_library)
        self._home_page.refresh_requested.connect(
            self._request_library_reconciliation
        )
        self.statusBar().hide()
        self._load_library_summary_cache()

    def show_home(self) -> None:
        current_page = self._stack.currentWidget()
        self._stack.setCurrentIndex(self.PAGE_HOME)
        for attribute, index in (
            ("_import_page", MainWindow.PAGE_IMPORT),
            ("_review_page", MainWindow.PAGE_REVIEW),
            ("_optimization_page", MainWindow.PAGE_OPTIMIZATION),
            ("_scripture_layout_page", MainWindow.PAGE_SCRIPTURE_LAYOUT),
            (
                "_custom_scripture_layout_page",
                MainWindow.PAGE_CUSTOM_SCRIPTURE_LAYOUT,
            ),
            ("_text_statistics_page", MainWindow.PAGE_TEXT_STATISTICS),
            ("_settings_page", MainWindow.PAGE_SETTINGS),
            ("_help_page", MainWindow.PAGE_HELP),
        ):
            page = getattr(self, attribute, None)
            if page is not None and current_page is page:
                MainWindow._release_fixed_page(self, attribute, index)
                return
        consistency_page = getattr(self, "_consistency_page", None)
        export_page = getattr(self, "_export_page", None)
        image_lab_page = getattr(self, "_image_lab_page", None)
        if consistency_page is not None and current_page is consistency_page:
            MainWindow._release_dynamic_page(self, "_consistency_page")
        elif export_page is not None and current_page is export_page:
            MainWindow._release_dynamic_page(self, "_export_page")
        elif image_lab_page is not None and current_page is image_lab_page:
            MainWindow._release_dynamic_page(self, "_image_lab_page")

    def _connect_import_page(self, page: ImportPage) -> None:
        page.home_requested.connect(self.show_home)
        page.import_completed.connect(self._import_completed)
        page.status_message.connect(self.statusBar().showMessage)

    def _connect_review_page(self, page: ReviewPage) -> None:
        page.home_requested.connect(self.show_home)
        page.summary_changed.connect(self._sync_library_summary)
        page.status_message.connect(self.statusBar().showMessage)

    def _connect_optimization_page(self, page: OptimizationPage) -> None:
        page.home_requested.connect(self.show_home)
        page.summary_changed.connect(self._sync_library_summary)
        page.status_message.connect(self.statusBar().showMessage)

    def _connect_scripture_layout_page(self, page: ScriptureLayoutPage) -> None:
        page.home_requested.connect(self.show_home)
        page.status_message.connect(self.statusBar().showMessage)

    def _connect_custom_scripture_layout_page(
        self,
        page: CustomScriptureLayoutPage,
    ) -> None:
        page.home_requested.connect(self.show_home)
        page.status_message.connect(self.statusBar().showMessage)

    def _connect_text_statistics_page(self, page: TextStatisticsPage) -> None:
        page.home_requested.connect(self.show_home)
        page.status_message.connect(self.statusBar().showMessage)

    def _connect_settings_page(self, page: SettingsPage) -> None:
        page.home_requested.connect(self.show_home)
        page.status_message.connect(self.statusBar().showMessage)
        page.settings_saved.connect(self._apply_performance_settings)
        page.cache_cleanup_requested.connect(self._release_idle_memory)

    def _connect_help_page(self, page: HelpPage) -> None:
        page.home_requested.connect(self.show_home)
        page.status_message.connect(self.statusBar().showMessage)

    def _release_fixed_page(self, attribute: str, index: int) -> None:
        """销毁固定功能页，并用空占位保持页面索引稳定。"""

        page = getattr(self, attribute, None)
        if page is None:
            return
        setattr(self, attribute, None)
        shutdown = getattr(page, "shutdown", None)
        if callable(shutdown):
            shutdown()
        self._stack.removeWidget(page)
        placeholder = QWidget()
        placeholder.setObjectName(f"releasedPage{index}")
        self._stack.insertWidget(index, placeholder)
        old_placeholder = self._page_placeholders.pop(attribute, None)
        if old_placeholder is not None:
            old_placeholder.deleteLater()
        self._page_placeholders[attribute] = placeholder
        page.deleteLater()
        QTimer.singleShot(0, gc.collect)

    def _release_dynamic_page(self, attribute: str) -> None:
        page = getattr(self, attribute, None)
        if page is None:
            return
        setattr(self, attribute, None)
        shutdown = getattr(page, "shutdown", None)
        if callable(shutdown):
            shutdown()
        self._stack.removeWidget(page)
        page.deleteLater()
        QTimer.singleShot(0, gc.collect)

    def _install_fixed_page(
        self,
        attribute: str,
        index: int,
        page: QWidget,
    ) -> QWidget:
        placeholder = self._page_placeholders.pop(attribute, None)
        if placeholder is not None:
            self._stack.removeWidget(placeholder)
            placeholder.deleteLater()
        self._stack.insertWidget(index, page)
        setattr(self, attribute, page)
        return page

    def _ensure_import_page(self) -> ImportPage:
        if self._import_page is None:
            from ui.pages.import_page import ImportPage

            page = ImportPage()
            self._connect_import_page(page)
            self._install_fixed_page("_import_page", self.PAGE_IMPORT, page)
        return self._import_page

    def _ensure_review_page(self) -> ReviewPage:
        if self._review_page is None:
            from ui.pages.review_page import ReviewPage

            page = ReviewPage()
            self._connect_review_page(page)
            self._install_fixed_page("_review_page", self.PAGE_REVIEW, page)
        return self._review_page

    def _ensure_optimization_page(self) -> OptimizationPage:
        if self._optimization_page is None:
            from ui.pages.optimization_page import OptimizationPage

            page = OptimizationPage()
            self._connect_optimization_page(page)
            self._install_fixed_page(
                "_optimization_page",
                self.PAGE_OPTIMIZATION,
                page,
            )
        return self._optimization_page

    def _ensure_scripture_layout_page(self) -> ScriptureLayoutPage:
        if self._scripture_layout_page is None:
            from ui.pages.scripture_layout_page import ScriptureLayoutPage

            page = ScriptureLayoutPage()
            self._connect_scripture_layout_page(page)
            self._install_fixed_page(
                "_scripture_layout_page",
                self.PAGE_SCRIPTURE_LAYOUT,
                page,
            )
        return self._scripture_layout_page

    def _ensure_custom_scripture_layout_page(self) -> CustomScriptureLayoutPage:
        if self._custom_scripture_layout_page is None:
            from ui.pages.custom_scripture_layout_page import CustomScriptureLayoutPage

            page = CustomScriptureLayoutPage()
            self._connect_custom_scripture_layout_page(page)
            self._install_fixed_page(
                "_custom_scripture_layout_page",
                self.PAGE_CUSTOM_SCRIPTURE_LAYOUT,
                page,
            )
        return self._custom_scripture_layout_page

    def _ensure_text_statistics_page(self) -> TextStatisticsPage:
        if self._text_statistics_page is None:
            from ui.pages.text_statistics_page import TextStatisticsPage

            page = TextStatisticsPage()
            self._connect_text_statistics_page(page)
            self._install_fixed_page(
                "_text_statistics_page",
                self.PAGE_TEXT_STATISTICS,
                page,
            )
        return self._text_statistics_page

    def _ensure_settings_page(self) -> SettingsPage:
        if self._settings_page is None:
            from ui.pages.settings_page import SettingsPage

            page = SettingsPage(service=self._settings_service)
            self._connect_settings_page(page)
            self._install_fixed_page(
                "_settings_page",
                self.PAGE_SETTINGS,
                page,
            )
        return self._settings_page

    def _ensure_help_page(self) -> HelpPage:
        if self._help_page is None:
            from ui.pages.help_page import HelpPage

            page = HelpPage()
            self._connect_help_page(page)
            self._install_fixed_page(
                "_help_page",
                self.PAGE_HELP,
                page,
            )
        return self._help_page

    def _apply_performance_settings(self, settings: ApplicationSettings) -> None:
        processor_count = self._automatic_thread_count
        if settings.performance_mode == PERFORMANCE_CONSERVATIVE:
            processor_count = max(1, processor_count // 2)
        self._thread_pool.setMaxThreadCount(processor_count)

    def _release_idle_memory(self) -> None:
        gc.collect()
        self.statusBar().showMessage("已释放闲置内存")

    def show_create_page(self) -> None:
        with _feature_startup_cursor():
            names = scan_library_names()
            page = getattr(self, "_import_page", None)
            if page is None:
                page = MainWindow._ensure_import_page(self)
            page.configure_create(names)
            self._stack.setCurrentWidget(page)
            self.statusBar().showMessage("新建字库：请选择包含文字图片的目录")

    def _open_tool(self, route: str) -> None:
        if route == "create":
            self.show_create_page()
            return
        if route == "layout":
            with _feature_startup_cursor():
                if hasattr(self, "_scripture_layout_page"):
                    page = getattr(self, "_scripture_layout_page", None)
                    if page is None:
                        page = MainWindow._ensure_scripture_layout_page(self)
                    self._stack.setCurrentWidget(page)
                else:
                    self._stack.setCurrentIndex(self.PAGE_SCRIPTURE_LAYOUT)
                self.statusBar().showMessage("通用经文排版：请选择字图来源并输入经文")
            return
        if route == "custom_layout":
            with _feature_startup_cursor():
                page = getattr(self, "_custom_scripture_layout_page", None)
                if page is None:
                    page = MainWindow._ensure_custom_scripture_layout_page(self)
                self._stack.setCurrentWidget(page)
                self.statusBar().showMessage(
                    "定制经文排版：每个非空行作为一列，使用空行分隔版面"
                )
            return
        if route == "statistics":
            with _feature_startup_cursor():
                page = getattr(self, "_text_statistics_page", None)
                if page is None:
                    page = MainWindow._ensure_text_statistics_page(self)
                self._stack.setCurrentWidget(page)
                self.statusBar().showMessage(
                    "文字统计：请选择经文文件，或直接输入、粘贴文字"
                )
            return
        if route == "settings":
            with _feature_startup_cursor():
                page = getattr(self, "_settings_page", None)
                if page is None:
                    page = MainWindow._ensure_settings_page(self)
                else:
                    page.reload()
                self._stack.setCurrentWidget(page)
                self.statusBar().showMessage("设置：程序级偏好与数据维护")
            return
        if route == "help":
            page = getattr(self, "_help_page", None)
            if page is None:
                page = MainWindow._ensure_help_page(self)
            self._stack.setCurrentWidget(page)
            self.statusBar().showMessage("使用说明：选择主题或搜索具体操作")
            return
        if route == "image_lab":
            with _feature_startup_cursor():
                page = getattr(self, "_image_lab_page", None)
                if page is None:
                    from ui.pages.image_lab_page import ImageLabPage

                    page = ImageLabPage()
                    page.home_requested.connect(self.show_home)
                    page.status_message.connect(self.statusBar().showMessage)
                    self._image_lab_page = page
                    self._stack.addWidget(page)
                self._stack.setCurrentWidget(page)
                self.statusBar().showMessage(
                    "图片实验室：打开整幅拓片、手稿或文字扫描件"
                )
            return
        names = {
            "statistics": "文字统计",
            "layout": "通用经文排版",
            "custom_layout": "定制经文排版",
            "settings": "设置",
        }
        name = names.get(route, "该功能")
        QMessageBox.information(self, name, f"{name}将在后续迁移阶段接入。")

    def _open_stage(self, route: str, library_name: str, library_path: str) -> None:
        if getattr(self, "_library_delete_active", False) is True:
            return
        safe_path = resolve_library_directory(
            config.ZIKU_ROOT,
            library_path,
            expected_name=library_name,
        )
        if not safe_path:
            QMessageBox.warning(self, "打开失败", "字库路径无效或已指向字库目录之外。")
            return
        library_path = safe_path
        if route == "parameters":
            self._edit_library_parameters(library_path)
            return
        if route == "delete":
            self._delete_library(library_name, library_path)
            return
        if route not in {"import", "optimization", "consistency", "export"}:
            return
        with _feature_startup_cursor():
            try:
                glyph_service = GlyphService.open(library_name, library_path)
                set_last_ziku_path(library_path)
                if route == "import":
                    names = scan_library_names()
                    page = getattr(self, "_import_page", None)
                    if page is None:
                        page = MainWindow._ensure_import_page(self)
                    page.configure_append(glyph_service, names)
                    self._stack.setCurrentWidget(page)
                    self.statusBar().showMessage(f"正在向字库“{library_name}”添加文字")
                    return
                if route == "optimization":
                    page = getattr(self, "_optimization_page", None)
                    if page is None:
                        page = MainWindow._ensure_optimization_page(self)
                    page.open_glyph_service(glyph_service)
                    self._stack.setCurrentWidget(page)
                    self.statusBar().showMessage(f"正在自动优化字库“{library_name}”")
                    return
                if route == "consistency":
                    self._open_consistency(glyph_service, library_name)
                    return
                if route == "export":
                    self._open_export(glyph_service, library_name)
                    return
            except (OSError, RuntimeError, ValueError) as exc:
                QMessageBox.warning(self, "打开失败", f"无法打开字库功能：{exc}")
                return

    def _edit_library_parameters(self, library_path: str) -> None:
        try:
            library_name = os.path.basename(os.path.normpath(library_path))
            library_path = resolve_library_directory(
                config.ZIKU_ROOT,
                library_path,
                expected_name=library_name,
            )
            if not library_path:
                raise ValueError("字库路径无效或已指向字库目录之外")
            service = GlyphService.open(library_name, library_path)
            names = scan_library_names()
            dialog = LibraryParametersDialog(service, names, self)
            dialog.saved.connect(
                lambda _name, _count, glyph=service, old_path=library_path: (
                    self._sync_library_summary(glyph, previous_path=old_path)
                )
            )
            dialog.exec()
        except (OSError, RuntimeError, ValueError) as exc:
            QMessageBox.warning(self, "参数修改失败", f"无法读取或保存字库参数：{exc}")

    def _open_consistency(self, glyph_service: GlyphService, library_name: str) -> None:
        from ui.pages.consistency_page import ConsistencyPage

        if self._consistency_page is not None:
            self._consistency_page.shutdown()
            self._stack.removeWidget(self._consistency_page)
            self._consistency_page.deleteLater()
        self._consistency_page = ConsistencyPage(glyph_service, self.show_home)
        self._consistency_page.summary_changed.connect(self._sync_library_summary)
        self._stack.addWidget(self._consistency_page)
        self._stack.setCurrentWidget(self._consistency_page)
        self.statusBar().showMessage(f"正在整体协调字库“{library_name}”")

    def _open_export(self, glyph_service: GlyphService, library_name: str) -> None:
        """打开所选字库的全库导出工作台。"""
        from ui.pages.export_page import ExportPage

        if self._export_page is not None:
            self._export_page.shutdown()
            self._stack.removeWidget(self._export_page)
            self._export_page.deleteLater()
        self._export_page = ExportPage(glyph_service, self.show_home)
        self._export_page.summary_changed.connect(self._sync_library_summary)
        self._stack.addWidget(self._export_page)
        self._stack.setCurrentWidget(self._export_page)
        self.statusBar().showMessage(f"正在导出字库“{library_name}”")

    def _import_completed(self, _library_name: str, library_path: str, _result: object) -> None:
        self.statusBar().showMessage(f"字库导入完成：{library_path}")
        if not library_path:
            return
        try:
            library_name = os.path.basename(os.path.normpath(library_path))
            target_path = self._normalized_library_path(library_path)
            was_known = any(
                self._normalized_library_path(str(item.get("path", "")))
                == target_path
                for item in self._library_cache_summaries
                if item.get("path")
            )
            self._sync_library_summary(GlyphService.open(library_name, library_path))
            if not was_known:
                self._home_page.select_library(library_path)
        except (OSError, RuntimeError, ValueError) as exc:
            QMessageBox.warning(
                self,
                "首页状态更新失败",
                f"字库已完成导入，但首页状态无法更新：{exc}\n\n"
                "请在首页点击“重新核对”。",
            )

    def _delete_library(self, library_name: str, library_path: str) -> None:
        if getattr(self, "_library_delete_active", False) is True:
            return
        if self._library_scan_active:
            QMessageBox.information(
                self,
                "正在核对字库",
                "请等待当前字库数据核对完成后再删除。",
            )
            return
        target = resolve_library_directory(
            config.ZIKU_ROOT,
            library_path,
            expected_name=library_name,
        )
        if not target:
            QMessageBox.warning(self, "删除失败", "字库路径无效，未执行删除。")
            return
        answer = QMessageBox.question(
            self,
            "删除字库",
            f"确定删除字库“{library_name}”吗？\n\n整个字库目录将移入系统回收站，可在回收站中恢复。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._library_delete_active = True
        self._library_delete_name = library_name
        self._library_delete_path = target
        self._home_page.set_deleting(True, library_name)
        worker = FunctionWorker(
            lambda progress_callback: _move_library_to_recycle_bin(
                target,
                library_name,
                progress_callback,
            ),
            with_progress=True,
        )
        self._library_delete_worker = worker
        self._workers.add(worker)
        worker.signals.progress.connect(self._library_delete_progress)
        worker.signals.finished.connect(self._library_delete_finished)
        worker.signals.failed.connect(self._library_delete_failed)
        worker.signals.finished.connect(
            lambda _result, task=worker: self._release_worker(task)
        )
        worker.signals.failed.connect(
            lambda _message, task=worker: self._release_worker(task)
        )
        try:
            self._thread_pool.start(worker)
        except Exception as exc:
            self._workers.discard(worker)
            self._library_delete_failed(str(exc))

    def _library_delete_progress(self, progress: object) -> None:
        if getattr(self, "_library_delete_active", False) is True:
            self._home_page.set_delete_progress(str(progress or ""))

    def _library_delete_finished(self, result: object) -> None:
        if getattr(self, "_library_delete_active", False) is not True:
            return
        payload = result if isinstance(result, dict) else {}
        expected_path = self._normalized_library_path(self._library_delete_path)
        result_path = self._normalized_library_path(str(payload.get("字库路径", "")))
        if not result_path or result_path != expected_path:
            self._library_delete_failed("后台删除任务返回了无效结果。")
            return
        library_name = self._library_delete_name
        library_path = self._library_delete_path
        self._library_delete_active = False
        self._library_delete_worker = None
        self._library_delete_name = ""
        self._library_delete_path = ""
        self._remove_library_summary(library_path)
        self._home_page.set_deleting(False)
        self.statusBar().showMessage(f"字库“{library_name}”已移入回收站")
        QMessageBox.information(
            self,
            "删除完成",
            f"字库“{library_name}”已移入系统回收站。",
        )

    def _library_delete_failed(self, message: str) -> None:
        if getattr(self, "_library_delete_active", False) is not True:
            return
        library_name = self._library_delete_name
        self._library_delete_active = False
        self._library_delete_worker = None
        self._library_delete_name = ""
        self._library_delete_path = ""
        self._home_page.set_deleting(False)
        QMessageBox.warning(
            self,
            "删除失败",
            f"无法将字库“{library_name}”移入回收站：{message}",
        )

    def open_library(self, library_path: str) -> None:
        with _feature_startup_cursor():
            try:
                name = os.path.basename(os.path.normpath(library_path))
                library_path = resolve_library_directory(
                    config.ZIKU_ROOT,
                    library_path,
                    expected_name=name,
                )
                if not library_path:
                    raise ValueError("字库路径无效或已指向字库目录之外")
                page = getattr(self, "_review_page", None)
                if page is None:
                    page = MainWindow._ensure_review_page(self)
                if not page.open_library(library_path):
                    return
                set_last_ziku_path(library_path)
            except (OSError, RuntimeError, ValueError) as exc:
                QMessageBox.warning(self, "打开失败", f"无法打开字库：{exc}")
                return
            self._stack.setCurrentWidget(page)
            self.statusBar().showMessage(f"当前字库：{library_path}")

    def _load_library_summary_cache(self) -> None:
        """启动时优先载入可信摘要，异常时才执行深度核对。"""
        signature = library_summary_signature()
        summaries = self._library_summary_store.load(signature)
        if summaries is None:
            self.refresh_libraries(force=True)
            return
        self._library_cache_ready = True
        self._library_cache_signature = signature
        self._library_cache_summaries = summaries
        self._home_page.set_libraries(summaries)

    def _request_library_reconciliation(self) -> None:
        """由用户明确确认后执行全部字库深度核对。"""
        if (
            self._library_scan_active
            or getattr(self, "_library_delete_active", False) is True
        ):
            return
        dialog = QMessageBox(self)
        dialog.setWindowTitle("重新核对字库数据")
        dialog.setIcon(QMessageBox.Icon.Question)
        dialog.setText(
            "重新核对将逐字检查所有字库的阶段文件并修正首页统计。\n\n"
            "字库较大时可能需要较长时间。仅在程序外修改过字库，"
            "或首页数据可能不准确时使用。"
        )
        confirm_button = dialog.addButton(
            "开始核对",
            QMessageBox.ButtonRole.AcceptRole,
        )
        cancel_button = dialog.addButton(
            "取消",
            QMessageBox.ButtonRole.RejectRole,
        )
        dialog.setDefaultButton(cancel_button)
        dialog.setEscapeButton(cancel_button)
        dialog.exec()
        if dialog.clickedButton() is confirm_button:
            self.refresh_libraries(force=True)

    @staticmethod
    def _normalized_library_path(path: str) -> str:
        return os.path.normcase(os.path.abspath(path))

    def _sync_library_summary(
        self,
        glyph_service: object,
        *,
        previous_path: str = "",
    ) -> None:
        """用已经成功提交的业务状态实时更新一个字库摘要。"""
        required_methods = (
            "get_variants",
            "get_glyph_groups",
            "get_metadata",
            "get_coordination_summary",
        )
        if (
            not getattr(glyph_service, "ziku_name", "")
            or not getattr(glyph_service, "ziku_dir", "")
            or not all(
                callable(getattr(glyph_service, method_name, None))
                for method_name in required_methods
            )
        ):
            return
        summary = summarize_glyph_service(glyph_service)
        target_path = self._normalized_library_path(str(summary["path"]))
        previous_key = (
            self._normalized_library_path(previous_path)
            if previous_path
            else ""
        )
        updated: list[dict[str, Any]] = []
        replaced = False
        for existing in self._library_cache_summaries:
            existing_path = self._normalized_library_path(
                str(existing.get("path", ""))
            )
            if previous_key and existing_path == previous_key:
                if not replaced:
                    updated.append(summary)
                    replaced = True
                continue
            if existing_path == target_path:
                if not replaced:
                    updated.append(summary)
                    replaced = True
                continue
            updated.append(existing)
        if not replaced:
            updated.append(summary)
        updated.sort(key=lambda item: pinyin_natural_key(str(item.get("name", ""))))
        self._apply_library_summaries(updated)

    def _remove_library_summary(self, library_path: str) -> None:
        target_path = self._normalized_library_path(library_path)
        summaries = [
            summary
            for summary in self._library_cache_summaries
            if self._normalized_library_path(str(summary.get("path", "")))
            != target_path
        ]
        self._apply_library_summaries(summaries)

    def _apply_library_summaries(
        self,
        summaries: list[dict[str, Any]],
    ) -> None:
        """同步更新内存、首页和持久摘要索引。"""
        signature = library_summary_signature()
        self._library_cache_ready = True
        self._library_cache_signature = signature
        self._library_cache_summaries = summaries
        self._home_page.set_libraries(summaries)
        try:
            self._library_summary_store.save(summaries, signature)
        except (OSError, TypeError, ValueError) as exc:
            if not self._summary_store_warning_shown:
                self._summary_store_warning_shown = True
                QMessageBox.warning(
                    self,
                    "字库状态索引保存失败",
                    f"首页状态已经实时更新，但状态索引无法保存：{exc}\n\n"
                    "下次启动时程序将重新核对字库数据。",
                )

    def refresh_libraries(self, *, force: bool = False) -> bool:
        """按需刷新首页；普通返回优先复用稳定缓存。"""
        if getattr(self, "_library_delete_active", False) is True:
            return False
        if self._library_scan_active:
            if force:
                self._library_refresh_force_pending = True
            return False

        if not force and self._library_cache_ready:
            return False

        self._start_library_scan(reset_unstable_retry=True)
        return True

    def _start_library_scan(self, *, reset_unstable_retry: bool) -> None:
        """启动唯一的首页摘要任务。"""
        if self._library_scan_active:
            return
        if reset_unstable_retry:
            self._library_scan_retrying_unstable = False
        self._library_scan_generation += 1
        generation = self._library_scan_generation
        self._library_scan_active = True
        self._home_page.set_loading(True)
        worker = FunctionWorker(_scan_library_summary_result, with_progress=True)
        self._workers.add(worker)
        worker.signals.progress.connect(
            lambda progress, token=generation: self._library_scan_progress(
                token,
                progress,
            )
        )
        worker.signals.finished.connect(
            lambda result, token=generation: self._library_scan_finished(
                token,
                result,
            )
        )
        worker.signals.failed.connect(
            lambda message, token=generation: self._library_scan_failed(
                token,
                message,
            )
        )
        worker.signals.finished.connect(lambda _result, task=worker: self._release_worker(task))
        worker.signals.failed.connect(lambda _message, task=worker: self._release_worker(task))
        self._thread_pool.start(worker)

    def _library_scan_progress(
        self,
        generation: int,
        progress: object,
    ) -> None:
        if (
            generation == self._library_scan_generation
            and isinstance(progress, LibraryScanProgress)
        ):
            self._home_page.set_scan_progress(progress)

    def _library_scan_finished(self, generation: int, result: object) -> None:
        if generation != self._library_scan_generation:
            return
        self._library_scan_active = False
        if not isinstance(result, LibraryScanResult):
            self._library_scan_failed(generation, "字库扫描返回了无效结果")
            return
        if result.signature_before != result.signature_after:
            if self._library_scan_retrying_unstable:
                self._library_scan_retrying_unstable = False
                self._library_refresh_force_pending = False
                self._home_page.set_loading_failed()
                self._show_scan_error(
                    "扫描期间字库数据持续变化，请稍后点击重新核对重试"
                )
                return
            self._library_scan_retrying_unstable = True
            self._library_refresh_force_pending = False
            self._start_library_scan(reset_unstable_retry=False)
            return

        self._library_cache_summaries = result.summaries
        self._library_cache_signature = result.signature_after
        self._library_cache_ready = True
        self._library_scan_retrying_unstable = False
        self._home_page.set_libraries(result.summaries)
        try:
            self._library_summary_store.save(
                result.summaries,
                result.signature_after,
            )
        except (OSError, TypeError, ValueError) as exc:
            if not self._summary_store_warning_shown:
                self._summary_store_warning_shown = True
                QMessageBox.warning(
                    self,
                    "字库状态索引保存失败",
                    f"核对结果已经显示，但状态索引无法保存：{exc}",
                )
        if self._library_refresh_force_pending:
            self._library_refresh_force_pending = False
            self._start_library_scan(reset_unstable_retry=True)

    def _library_scan_failed(self, generation: int, message: str) -> None:
        if generation != self._library_scan_generation:
            return
        self._library_scan_active = False
        self._library_scan_retrying_unstable = False
        self._library_refresh_force_pending = False
        self._home_page.set_loading_failed()
        self._show_scan_error(message)

    def _show_scan_error(self, message: str) -> None:
        QMessageBox.warning(self, "扫描失败", f"无法读取字库列表：{message}")

    def _release_worker(self, worker: FunctionWorker) -> None:
        self._workers.discard(worker)

    def closeEvent(self, event: QCloseEvent) -> None:
        """关闭窗口前保护运行中的批量任务和未保存调整。"""
        if getattr(self, "_library_delete_active", False) is True:
            QMessageBox.information(
                self,
                "正在删除字库",
                "请等待字库移入回收站完成后再关闭程序。",
            )
            event.ignore()
            return
        import_page = getattr(self, "_import_page", None)
        if import_page is not None and import_page.is_running:
            QMessageBox.information(
                self,
                "导入任务正在执行",
                "请先停止当前扫描或导入，等待任务安全结束后再关闭程序。",
            )
            event.ignore()
            return
        optimization_page = getattr(self, "_optimization_page", None)
        if optimization_page is not None and optimization_page.is_batch_running:
            QMessageBox.information(
                self,
                "批量任务正在执行",
                "整库自动优化正在执行。如需提前退出，请先点击“停止批量优化”，"
                "等待任务安全停止后再关闭程序。",
            )
            event.ignore()
            return
        review_page = getattr(self, "_review_page", None)
        if review_page is not None and review_page.is_batch_running:
            QMessageBox.information(
                self,
                "批量任务正在执行",
                "整库手工审核正在执行。如需提前退出，请先点击“停止批量审核”，"
                "等待任务安全停止后再关闭程序。",
            )
            event.ignore()
            return
        if (
            self._consistency_page is not None
            and self._consistency_page.is_batch_running
        ):
            QMessageBox.information(
                self,
                "批量任务正在执行",
                "整库整体协调正在执行。如需提前退出，请先点击“停止整体协调”，"
                "等待任务安全停止后再关闭程序。",
            )
            event.ignore()
            return
        scripture_layout_page = getattr(self, "_scripture_layout_page", None)
        if (
            scripture_layout_page is not None
            and scripture_layout_page.is_running
        ):
            QMessageBox.information(
                self,
                "排版任务正在执行",
                "通用经文排版正在生成分层 PSD。如需提前退出，请先点击“停止生成”，"
                "等待当前版安全停止后再关闭程序。",
            )
            event.ignore()
            return
        custom_scripture_layout_page = getattr(
            self,
            "_custom_scripture_layout_page",
            None,
        )
        if (
            custom_scripture_layout_page is not None
            and custom_scripture_layout_page.is_running
        ):
            QMessageBox.information(
                self,
                "排版任务正在执行",
                "定制经文排版正在生成分层 PSD。如需提前退出，请先点击“停止生成”，"
                "等待当前版安全停止后再关闭程序。",
            )
            event.ignore()
            return
        text_statistics_page = getattr(self, "_text_statistics_page", None)
        export_page = getattr(self, "_export_page", None)
        image_lab_page = getattr(self, "_image_lab_page", None)
        if export_page is not None and export_page.is_running:
            QMessageBox.information(
                self,
                "导出任务正在执行",
                "请先取消导出，等待当前文件安全结束后再关闭程序。",
            )
            event.ignore()
            return
        if image_lab_page is not None and image_lab_page.is_running:
            QMessageBox.information(
                self,
                "图片实验室任务正在执行",
                "请先等待预览完成，或停止完整尺寸导出后再关闭程序。",
            )
            event.ignore()
            return
        if (
            self._consistency_page is not None
            and self._stack.currentWidget() is self._consistency_page
            and not self._consistency_page._confirm_leave_page()
        ):
            event.ignore()
            return
        if (
            image_lab_page is not None
            and self._stack.currentWidget() is image_lab_page
            and not image_lab_page._confirm_leave_page()
        ):
            event.ignore()
            return
        for page in (
            import_page,
            optimization_page,
            review_page,
            getattr(self, "_consistency_page", None),
            export_page,
            scripture_layout_page,
            custom_scripture_layout_page,
            text_statistics_page,
            getattr(self, "_settings_page", None),
            image_lab_page,
        ):
            shutdown = getattr(page, "shutdown", None)
            if callable(shutdown):
                shutdown()
        event.accept()
