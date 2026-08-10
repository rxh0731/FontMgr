# log_manager.py — 日志文件管理

import os
import threading
from datetime import datetime

import config


_active_manager: "LogManager | None" = None


def write_log(message: str) -> None:
    """由业务模块写入当前程序日志；日志尚未启动时静默跳过。"""
    manager = _active_manager
    if manager is not None:
        manager.write(message)


class LogManager:
    """全局日志管理器（线程安全，多文件轮转）。"""

    def __init__(self) -> None:
        self._file_path: str = config.LOG_FILE
        self._max_bytes: int = config.LOG_MAX_BYTES
        self._handle = None
        self._lock = threading.Lock()

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
        with self._lock:
            if self._handle is not None:
                try:
                    self._handle.write(line)
                    self._handle.flush()
                except OSError:
                    pass

    def _rotate_if_needed(self) -> None:
        if os.path.exists(self._file_path) and os.path.getsize(self._file_path) >= self._max_bytes:
            bak = self._file_path + ".old"
            if os.path.exists(bak):
                os.unlink(bak)
            os.rename(self._file_path, bak)
