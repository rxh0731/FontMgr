# theme.py — 深色主题色值、字体、样式工厂

import tkinter as tk
from tkinter import ttk
from typing import Any


# ===== 颜色 =====
BG_MAIN: str = "#1e1e2e"
BG_PANEL: str = "#252535"
BG_CANVAS: str = "#2a2a3a"
BG_INPUT: str = "#333348"
BG_HOVER: str = "#3a3a50"
BG_ACTIVE: str = "#454570"

FG_PRIMARY: str = "#e0e0e8"
FG_SECONDARY: str = "#a0a0b8"
FG_MUTED: str = "#686880"
FG_ACCENT: str = "#7aa2f7"

BORDER: str = "#3a3a55"
DIVIDER: str = "#2a2a40"

# 状态颜色（与 config.STATUS_COLORS 对齐）
COLOR_PENDING: str = "#888888"
COLOR_DRAFT: str = "#FF8C00"
COLOR_REVIEWED: str = "#4169E1"
COLOR_FINALIZED: str = "#228B22"

# 按钮颜色
BTN_DEFAULT: str = "#414174"
BTN_HOVER: str = "#5050a0"
BTN_PRESS: str = "#303060"
BTN_ACCENT: str = "#5a7fbf"
BTN_DANGER: str = "#8b3a3a"

# ===== 字体 =====
FONT_FAMILY: str = "微软雅黑"
FONT_MONO: str = "Consolas"

FONT_SMALL: tuple = (FONT_FAMILY, 8)
FONT_NORMAL: tuple = (FONT_FAMILY, 10)
FONT_BOLD: tuple = (FONT_FAMILY, 10, "bold")
FONT_TITLE: tuple = (FONT_FAMILY, 13, "bold")
FONT_HEADING: tuple = (FONT_FAMILY, 16, "bold")
FONT_CODE: tuple = (FONT_MONO, 9)
FONT_TREE: tuple = (FONT_FAMILY, 10)


# ===== 样式工厂 =====

def apply_dark_theme(root: tk.Tk) -> None:
    """将深色主题应用到 tkinter 根窗口。
    
    在应用初始化时调用一次。
    """
    root.configure(bg=BG_MAIN)

    style = ttk.Style(root)
    style.theme_use("clam")

    # 通用配置
    style.configure(".", background=BG_MAIN, foreground=FG_PRIMARY, font=FONT_NORMAL)

    # Frame
    style.configure("TFrame", background=BG_MAIN)
    style.configure("Panel.TFrame", background=BG_PANEL)
    style.configure("Canvas.TFrame", background=BG_CANVAS)
    style.configure("Toolbar.TFrame", background=BG_PANEL)

    # Label
    style.configure("TLabel", background=BG_MAIN, foreground=FG_PRIMARY)
    style.configure("Panel.TLabel", background=BG_PANEL, foreground=FG_PRIMARY)
    style.configure("Muted.TLabel", foreground=FG_MUTED)
    style.configure("Accent.TLabel", foreground=FG_ACCENT)
    style.configure("Title.TLabel", font=FONT_TITLE)
    style.configure("Heading.TLabel", font=FONT_HEADING, foreground=FG_PRIMARY)

    # Button
    style.configure("TButton", background=BTN_DEFAULT, foreground=FG_PRIMARY, borderwidth=0, padding=(8, 4), font=FONT_NORMAL)
    style.map("TButton", background=[("active", BTN_HOVER), ("pressed", BTN_PRESS)])
    style.configure("Accent.TButton", background=BTN_ACCENT)
    style.configure("Danger.TButton", background=BTN_DANGER)

    # Entry
    style.configure("TEntry", fieldbackground=BG_INPUT, foreground=FG_PRIMARY, insertcolor=FG_PRIMARY)

    # Combobox
    style.configure("TCombobox", fieldbackground=BG_INPUT, foreground=FG_PRIMARY, arrowcolor=FG_PRIMARY)

    # Treeview
    style.configure("Treeview", background=BG_PANEL, foreground=FG_PRIMARY, fieldbackground=BG_PANEL, borderwidth=0, font=FONT_TREE)
    style.configure("Treeview.Heading", background=BG_ACTIVE, foreground=FG_PRIMARY, font=FONT_BOLD)
    style.map("Treeview", background=[("selected", BG_ACTIVE)], foreground=[("selected", FG_ACCENT)])

    # Scrollbar
    style.configure("TScrollbar", background=BG_PANEL, troughcolor=BG_MAIN, borderwidth=0, arrowsize=12)


# ===== 通用控件构建辅助 =====

def make_label(parent: tk.Widget, text: str = "", **kwargs: Any) -> tk.Label:
    """创建统一风格的 Label。

    支持 textvariable 参数（Tkinter Variable 类型）。
    """
    defaults = dict(bg=BG_MAIN, fg=FG_PRIMARY, font=FONT_NORMAL)
    defaults.update(kwargs)
    return tk.Label(parent, text=text, **defaults)


def make_button(parent: tk.Widget, text: str, accent: bool = False, danger: bool = False, **kwargs: Any) -> tk.Button:
    """创建统一风格的 Button。"""
    bg = BTN_ACCENT if accent else (BTN_DANGER if danger else BTN_DEFAULT)
    return tk.Button(
        parent, text=text,
        bg=bg, fg=FG_PRIMARY, font=FONT_NORMAL,
        activebackground=BTN_HOVER, activeforeground=FG_PRIMARY,
        bd=0, padx=8, pady=4, cursor="hand2",
        **kwargs,
    )


def make_entry(parent: tk.Widget, placeholder: str = "", **kwargs: Any) -> tk.Entry:
    """创建统一风格的 Entry。"""
    e = tk.Entry(
        parent,
        bg=BG_INPUT, fg=FG_PRIMARY,
        insertbackground=FG_PRIMARY,
        font=FONT_NORMAL, bd=0,
        **kwargs,
    )
    if placeholder:
        e.insert(0, placeholder)
        e.configure(fg=FG_MUTED)
        e.bind("<FocusIn>", lambda ev: _on_entry_focus_in(ev, placeholder))
        e.bind("<FocusOut>", lambda ev: _on_entry_focus_out(ev, placeholder))
    return e


def _on_entry_focus_in(event: tk.Event, placeholder: str) -> None:
    e = event.widget
    if e.get() == placeholder:
        e.delete(0, tk.END)
        e.configure(fg=FG_PRIMARY)


def _on_entry_focus_out(event: tk.Event, placeholder: str) -> None:
    e = event.widget
    if not e.get():
        e.insert(0, placeholder)
        e.configure(fg=FG_MUTED)
