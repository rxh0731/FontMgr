# progress_dialog.py — 进度条对话框

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Optional

import config
from ui import theme
from ui.widgets.custom_dialog import show_error
from ui.widgets.dark_helpers import apply_dark_titlebar, set_window_icon


class ProgressDialog(tk.Toplevel):
    """模态进度条对话框，支持标题、消息、进度更新和取消按钮。"""

    def __init__(
        self,
        parent: tk.Widget,
        title: str = "处理中...",
        total: int = 100,
        cancellable: bool = False,
    ) -> None:
        super().__init__(parent)
        self.title(title)
        self.configure(bg=theme.BG_MAIN)
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        set_window_icon(self, config.ICON_FILE)
        apply_dark_titlebar(self)

        self._cancelled: bool = False
        self._cancel_callback: Optional[Callable[[], None]] = None
        self._total: int = max(1, total)

        # 布局
        pad = 20

        self._msg_label = theme.make_label(self, "正在处理...")
        self._msg_label.pack(pady=(pad, 8), padx=pad)

        self._progress_var = tk.IntVar(value=0)
        self._progress_bar = ttk.Progressbar(
            self, variable=self._progress_var, maximum=self._total, length=360,
        )
        self._progress_bar.pack(pady=(0, 8), padx=pad)

        self._detail_label = theme.make_label(self, "", fg=theme.FG_MUTED, wraplength=360)
        self._detail_label.pack(pady=(0, 8), padx=pad)

        if cancellable:
            theme.make_button(self, "取消", command=self._on_cancel).pack(pady=(0, pad))

        # 居中
        self.update_idletasks()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_rootx(), parent.winfo_rooty()
        w = 400
        h = self.winfo_reqheight()
        x = px + (pw - w) // 2
        y = py + (ph - h) // 3
        self.geometry(f"{w}x{h}+{max(0,x)}+{max(0,y)}")

    def update_progress(self, current: int, message: str = "", detail: str = "") -> None:
        """更新进度。

        参数：
            current: 当前进度值
            message: 主消息
            detail: 详情消息
        """
        self._progress_var.set(min(current, self._total))
        if message:
            self._msg_label.configure(text=message)
        if detail:
            self._detail_label.configure(text=detail)
        self.update_idletasks()

    def step(self, message: str = "", detail: str = "") -> None:
        """步进 1。"""
        self.update_progress(self._progress_var.get() + 1, message, detail)

    def set_cancel_callback(self, callback: Callable[[], None]) -> None:
        self._cancel_callback = callback

    def _on_cancel(self) -> None:
        self._cancelled = True
        if self._cancel_callback:
            self._cancel_callback()

    def is_cancelled(self) -> bool:
        return self._cancelled

    def run_task(self, func: Callable[[], Any], on_done: Optional[Callable[[Any], None]] = None) -> None:
        """在后台线程执行任务，并在界面线程处理结果。"""
        import threading

        def worker() -> None:
            try:
                result = func()
                self.after(0, lambda value=result: self._finish_task(value, on_done))
            except Exception as exc:
                self.after(0, lambda error=exc: self._fail_task(error))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_task(self, result: Any, on_done: Optional[Callable[[Any], None]]) -> None:
        self.destroy()
        if on_done:
            on_done(result)

    def _fail_task(self, error: Exception) -> None:
        self.destroy()
        show_error(self.master, "任务失败", str(error))

    def finish(self) -> None:
        self.destroy()
