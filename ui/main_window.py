"""应用主窗口与页面导航。"""

from __future__ import annotations

from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import QMainWindow, QMessageBox, QStackedWidget

from data.config_store import set_last_ziku_path
from ui.pages.home_page import HomePage, scan_library_summaries
from ui.pages.placeholder_page import PlaceholderPage
from ui.workers import FunctionWorker


class MainWindow(QMainWindow):
    """承载全部功能页面和全局导航状态。"""

    PAGE_HOME = 0
    PAGE_CREATE = 1
    PAGE_LIBRARY = 2

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("字库编辑器")
        self.setMinimumSize(1000, 680)
        self.resize(1280, 820)
        self._thread_pool = QThreadPool.globalInstance()
        self._workers: set[FunctionWorker] = set()
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self._home_page = HomePage()
        self._create_page = PlaceholderPage(
            "新建字库",
            "应用骨架已就绪。新建字库向导将在后续迁移阶段接入。",
        )
        self._library_page = PlaceholderPage(
            "字库工作台",
            "字库已选中。手工审核页面将以此导航骨架为基础接入。",
        )
        self._stack.addWidget(self._home_page)
        self._stack.addWidget(self._create_page)
        self._stack.addWidget(self._library_page)

        self._home_page.create_requested.connect(self.show_create_page)
        self._home_page.open_requested.connect(self.open_library)
        self._home_page.refresh_requested.connect(self.refresh_libraries)
        self._create_page.home_requested.connect(self.show_home)
        self._library_page.home_requested.connect(self.show_home)
        self.statusBar().showMessage("应用已就绪")
        self.refresh_libraries()

    def show_home(self) -> None:
        self._stack.setCurrentIndex(self.PAGE_HOME)
        self.refresh_libraries()

    def show_create_page(self) -> None:
        self._stack.setCurrentIndex(self.PAGE_CREATE)
        self.statusBar().showMessage("新建字库功能待迁移")

    def open_library(self, library_path: str) -> None:
        try:
            set_last_ziku_path(library_path)
        except OSError as exc:
            QMessageBox.warning(self, "记录失败", f"无法记录最近打开的字库：{exc}")
        self._stack.setCurrentIndex(self.PAGE_LIBRARY)
        self.statusBar().showMessage(f"当前字库：{library_path}")

    def refresh_libraries(self) -> None:
        self._home_page.set_loading(True)
        worker = FunctionWorker(scan_library_summaries)
        self._workers.add(worker)
        worker.signals.finished.connect(self._home_page.set_libraries)
        worker.signals.failed.connect(self._home_page.show_error)
        worker.signals.finished.connect(lambda _result, task=worker: self._release_worker(task))
        worker.signals.failed.connect(lambda _message, task=worker: self._release_worker(task))
        self._thread_pool.start(worker)

    def _release_worker(self, worker: FunctionWorker) -> None:
        self._workers.discard(worker)
