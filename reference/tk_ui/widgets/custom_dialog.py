"""深色主题对话框组件。"""
from __future__ import annotations

import tkinter as tk
from typing import Optional

import config
from ui import theme
from ui.widgets.dark_helpers import apply_dark_titlebar, set_window_icon


class CustomDialog(tk.Toplevel):
    """与主程序配色一致的模态对话框。"""

    def __init__(
        self,
        parent: tk.Misc,
        title: str,
        message: str,
        dialog_type: str = "info",
        buttons: Optional[list[str]] = None,
        *,
        input_value: Optional[str] = None,
        input_prompt: str = "",
    ) -> None:
        super().__init__(parent)
        self.result: Optional[str] = None
        self._input_var: Optional[tk.StringVar] = None
        self.title(title)
        self.configure(bg=theme.BG_MAIN)
        self.resizable(False, False)
        self.withdraw()
        self.transient(parent)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        set_window_icon(self, config.ICON_FILE)

        if buttons is None:
            buttons = ["确定"]

        frame = tk.Frame(self, bg=theme.BG_MAIN, padx=24, pady=20)
        frame.pack(fill=tk.BOTH, expand=True)

        colors = {
            "info": theme.FG_ACCENT,
            "warning": theme.COLOR_DRAFT,
            "error": theme.BTN_DANGER,
            "question": theme.FG_ACCENT,
            "input": theme.FG_ACCENT,
        }
        symbols = {"info": "i", "warning": "!", "error": "×", "question": "?", "input": "…"}
        color = colors.get(dialog_type, theme.FG_ACCENT)
        symbol = symbols.get(dialog_type, "i")

        icon = tk.Canvas(frame, width=40, height=40, bg=theme.BG_MAIN, highlightthickness=0)
        icon.grid(row=0, column=0, rowspan=3, padx=(0, 16), sticky="n")
        icon.create_oval(3, 3, 37, 37, fill=color, outline="")
        icon.create_text(20, 20, text=symbol, fill="#FFFFFF", font=("Microsoft YaHei UI", 18, "bold"))

        tk.Label(
            frame,
            text=message,
            bg=theme.BG_MAIN,
            fg=theme.FG_PRIMARY,
            font=theme.FONT_NORMAL,
            justify=tk.LEFT,
            wraplength=430,
        ).grid(row=0, column=1, sticky="w")

        if input_value is not None:
            if input_prompt:
                tk.Label(
                    frame, text=input_prompt, bg=theme.BG_MAIN, fg=theme.FG_SECONDARY,
                    font=theme.FONT_SMALL,
                ).grid(row=1, column=1, sticky="w", pady=(14, 4))
            self._input_var = tk.StringVar(value=input_value)
            entry = tk.Entry(
                frame,
                textvariable=self._input_var,
                bg=theme.BG_INPUT,
                fg=theme.FG_PRIMARY,
                insertbackground=theme.FG_PRIMARY,
                selectbackground=theme.FG_ACCENT,
                selectforeground="#FFFFFF",
                relief=tk.FLAT,
                font=theme.FONT_NORMAL,
                width=38,
            )
            entry.grid(row=2, column=1, sticky="ew", pady=(0, 4), ipady=7)
            entry.focus_set()
            entry.select_range(0, tk.END)
            entry.bind("<Return>", lambda _event: self._on_button("确定"))

        button_frame = tk.Frame(frame, bg=theme.BG_MAIN)
        button_frame.grid(row=3, column=0, columnspan=2, sticky="e", pady=(22, 0))
        for index, text in enumerate(buttons):
            primary = index == 0
            button = tk.Button(
                button_frame,
                text=text,
                command=lambda value=text: self._on_button(value),
                bg=theme.BTN_ACCENT if primary else theme.BG_HOVER,
                fg="#FFFFFF" if primary else theme.FG_PRIMARY,
                activebackground=theme.BTN_HOVER if primary else theme.BG_ACTIVE,
                activeforeground="#FFFFFF",
                relief=tk.FLAT,
                bd=0,
                padx=18,
                pady=7,
                font=theme.FONT_NORMAL,
                cursor="hand2",
            )
            button.pack(side=tk.LEFT, padx=(8, 0))

        self.update_idletasks()
        width = max(420, self.winfo_reqwidth())
        height = self.winfo_reqheight()
        parent.update_idletasks()
        x = parent.winfo_rootx() + max(0, (parent.winfo_width() - width) // 2)
        y = parent.winfo_rooty() + max(0, (parent.winfo_height() - height) // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")
        self.deiconify()
        apply_dark_titlebar(self)
        self.lift()
        self.grab_set()
        self.wait_window(self)

    def _on_button(self, value: str) -> None:
        if self._input_var is not None:
            self.result = self._input_var.get() if value == "确定" else None
        else:
            self.result = value
        self.grab_release()
        self.destroy()

    def _on_close(self) -> None:
        self.result = None
        self.grab_release()
        self.destroy()


def show_info(parent: tk.Misc, title: str, message: str) -> None:
    CustomDialog(parent, title, message, "info")


def show_warning(parent: tk.Misc, title: str, message: str) -> None:
    CustomDialog(parent, title, message, "warning")


def show_error(parent: tk.Misc, title: str, message: str) -> None:
    CustomDialog(parent, title, message, "error")


def ask_yes_no(parent: tk.Misc, title: str, message: str) -> bool:
    return CustomDialog(parent, title, message, "question", ["确定", "取消"]).result == "确定"


def ask_save_discard_cancel(parent: tk.Misc, title: str, message: str) -> str:
    """询问未保存修改的处理方式，返回“保存”“不保存”或“取消”。"""
    result = CustomDialog(
        parent,
        title,
        message,
        "question",
        ["保存", "不保存", "取消"],
    ).result
    return result if result in ("保存", "不保存") else "取消"


def show_confirm(parent: tk.Misc, title: str, message: str) -> bool:
    """兼容原有调用的深色确认对话框。"""
    return ask_yes_no(parent, title, message)


def ask_string(
    parent: tk.Misc,
    title: str,
    message: str,
    initial_value: str = "",
    input_prompt: str = "",
) -> Optional[str]:
    """显示深色主题文本输入对话框。"""
    return CustomDialog(
        parent,
        title,
        message,
        "input",
        ["确定", "取消"],
        input_value=initial_value,
        input_prompt=input_prompt,
    ).result
