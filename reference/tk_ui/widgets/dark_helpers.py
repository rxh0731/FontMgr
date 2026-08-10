# dark_helpers.py — 深色窗口辅助函数

import ctypes
import os
import tkinter as tk
from typing import Optional


def apply_dark_titlebar(root: tk.Misc) -> None:
    """在 Windows 11 上设置深色标题栏。
    
    使用 DwmSetWindowAttribute 设置窗口使用深色模式。
    """
    try:
        DWMWA_USE_IMMERSIVE_DARK_MODE = 20
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE,
            ctypes.byref(ctypes.c_int(1)),
            ctypes.sizeof(ctypes.c_int),
        )
    except Exception:
        pass  # 非 Windows 10+ 系统忽略


def set_window_icon(root: tk.Misc, icon_path: Optional[str] = None) -> None:
    """设置窗口图标（.ico 文件）。"""
    if icon_path and os.path.exists(icon_path):
        try:
            root.iconbitmap(icon_path)
        except Exception:
            pass


def center_window(root: tk.Tk, width: int = 1400, height: int = 900) -> None:
    """将窗口居中显示。"""
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    x = (sw - width) // 2
    y = (sh - height) // 2
    root.geometry(f"{width}x{height}+{x}+{y}")
