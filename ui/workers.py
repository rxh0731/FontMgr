"""基于 QThreadPool 的通用后台任务封装。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QRunnable, Signal, Slot


class WorkerSignals(QObject):
    """后台任务统一信号。"""

    finished = Signal(object)
    failed = Signal(str)


class FunctionWorker(QRunnable):
    """在线程池中执行无参数函数。"""

    def __init__(self, function: Callable[[], Any]) -> None:
        super().__init__()
        self._function = function
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            result = self._function()
        except Exception as exc:
            self.signals.failed.emit(str(exc))
        else:
            self.signals.finished.emit(result)
