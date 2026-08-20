# log_manager.py — 日志文件管理

import os
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator

import config


_active_manager: "LogManager | None" = None
_BUFFER_LIMIT_BYTES = 64 * 1024


def write_log(message: str) -> None:
    """由业务模块写入当前程序日志；日志尚未启动时静默跳过。"""
    manager = _active_manager
    if manager is not None:
        manager.write(message)


@contextmanager
def buffered_log_writes() -> Iterator[None]:
    """仅在当前线程合并日志写入，最外层上下文退出时必定刷新。"""
    manager = _active_manager
    if manager is None:
        yield
        return
    with manager.buffered_writes():
        yield


class LogManager:
    """全局日志管理器（线程安全，多文件轮转）。"""

    def __init__(self) -> None:
        self._file_path: str = config.LOG_FILE
        self._max_bytes: int = config.LOG_MAX_BYTES
        self._handle = None
        self._lock = threading.Lock()
        self._thread_state = threading.local()

    def open(self) -> None:
        """打开日志文件（追加模式），如超限则轮转。"""
        global _active_manager
        self._rotate_if_needed()
        try:
            self._handle = open(self._file_path, "a", encoding="utf-8")
            _active_manager = self
        except OSError:
            self._handle = None

    def close(self) -> None:
        global _active_manager
        if _active_manager is self:
            _active_manager = None
        self._flush_thread_buffer()
        with self._lock:
            if self._handle is not None:
                try:
                    self._handle.close()
                except OSError:
                    pass
                self._handle = None

    def write(self, message: str) -> None:
        """写入一行日志（自动追加时间戳和换行）。"""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        line = f"[{ts}] {message}\n"
        state = getattr(self._thread_state, "buffer", None)
        if state is not None and state["depth"] > 0:
            state["lines"].append(line)
            state["bytes"] += len(line.encode("utf-8"))
            if state["bytes"] >= _BUFFER_LIMIT_BYTES:
                self._flush_thread_buffer()
            return
        self._write_block(line)

    @contextmanager
    def buffered_writes(self) -> Iterator[None]:
        """为当前线程启用可嵌套的日志缓冲。"""
        state = getattr(self._thread_state, "buffer", None)
        if state is None:
            state = {"depth": 0, "lines": [], "bytes": 0}
            self._thread_state.buffer = state
        state["depth"] += 1
        try:
            yield
        finally:
            state["depth"] -= 1
            if state["depth"] == 0:
                self._flush_thread_buffer()
                try:
                    del self._thread_state.buffer
                except AttributeError:
                    pass

    def _flush_thread_buffer(self) -> None:
        state = getattr(self._thread_state, "buffer", None)
        if state is None or not state["lines"]:
            return
        block = "".join(state["lines"])
        state["lines"].clear()
        state["bytes"] = 0
        self._write_block(block)

    def _write_block(self, block: str) -> None:
        """在线程锁内完成一次日志块写入和刷新。"""
        with self._lock:
            if self._handle is not None:
                try:
                    self._handle.write(block)
                    self._handle.flush()
                except OSError:
                    pass

    def _rotate_if_needed(self) -> None:
        if os.path.exists(self._file_path) and os.path.getsize(self._file_path) >= self._max_bytes:
            bak = self._file_path + ".old"
            if os.path.exists(bak):
                os.unlink(bak)
            os.rename(self._file_path, bak)
