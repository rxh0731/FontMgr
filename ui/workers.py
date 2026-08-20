"""基于 QThreadPool 的通用后台任务封装。"""

from __future__ import annotations

from collections.abc import Callable
import traceback
from typing import Any

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from data.log_manager import write_log


def log_background_exception(context: str) -> None:
    """把当前后台异常的完整堆栈写入程序日志。"""

    write_log(f"后台任务异常｜任务={context}\n{traceback.format_exc()}")


class WorkerSignals(QObject):
    """后台任务统一信号。"""

    progress = Signal(object)
    finished = Signal(object)
    failed = Signal(str)


class FunctionWorker(QRunnable):
    """在线程池中执行函数，并可注入线程安全的进度回调。"""

    def __init__(
        self,
        function: Callable[..., Any],
        *,
        with_progress: bool = False,
    ) -> None:
        super().__init__()
        self._function = function
        self._with_progress = bool(with_progress)
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            result = (
                self._function(self.signals.progress.emit)
                if self._with_progress
                else self._function()
            )
        except Exception as exc:
            function_name = getattr(self._function, "__qualname__", repr(self._function))
            log_background_exception(function_name)
            try:
                self.signals.failed.emit(str(exc))
            except RuntimeError:
                pass
        else:
            try:
                self.signals.finished.emit(result)
            except RuntimeError:
                pass
