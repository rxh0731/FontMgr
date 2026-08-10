# import_wizard.py — 单页面新建字库

from __future__ import annotations

import os
import shutil
import threading
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog
from typing import Callable, Optional

from PIL import Image, ImageOps, ImageTk

import config
from services.glyph_service import GlyphService
from services.import_service import ImportService
from services.traditional_chinese_service import identify_character
from ui import theme
from ui.widgets.custom_dialog import show_error, show_info, show_warning
from ui.widgets.dark_helpers import set_window_icon
from ui.widgets.progress_dialog import ProgressDialog
from utils.file_utils import compute_file_md5, pinyin_natural_key, validate_final_char


@dataclass
class ScanItem:
    path: str
    filename: str
    original_char: str
    category: str
    candidates: tuple[str, ...]
    final_char: str
    confirmed: bool
    thumbnail: Optional[Image.Image] = None


class RoundedPanel(tk.Canvas):
    def __init__(self, parent, *, background: str, border: str, radius: int = 12, **kwargs):
        super().__init__(parent, bg=theme.BG_MAIN, highlightthickness=0, bd=0, **kwargs)
        self._background, self._border, self._radius = background, border, radius
        self.content = tk.Frame(self, bg=background)
        self._window = self.create_window(0, 0, anchor="nw", window=self.content)
        self.bind("<Configure>", self._redraw)

    def _redraw(self, _event=None):
        width, height = self.winfo_width(), self.winfo_height()
        if width <= 2 or height <= 2:
            return
        self.delete("圆角背景")
        points = self._rounded_points(2, 2, width - 2, height - 2, self._radius)
        self.create_polygon(points, smooth=True, splinesteps=24, fill=self._background, outline=self._border, width=1, tags="圆角背景")
        self.tag_lower("圆角背景")
        self.coords(self._window, 13, 13)
        self.itemconfigure(self._window, width=max(1, width - 26), height=max(1, height - 26))

    @staticmethod
    def _rounded_points(x1, y1, x2, y2, r):
        return (x1+r,y1, x2-r,y1, x2,y1, x2,y1+r, x2,y2-r, x2,y2, x2-r,y2, x1+r,y2, x1,y2, x1,y2-r, x1,y1+r, x1,y1)


class ScrollableColumn(tk.Frame):
    def __init__(self, parent, *, background: str):
        super().__init__(parent, bg=background)
        self.canvas = tk.Canvas(
            self,
            bg=background,
            highlightthickness=0,
            bd=0,
        )
        scrollbar = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.content = tk.Frame(self.canvas, bg=background)
        self._content_window = self.canvas.create_window(
            0,
            0,
            anchor="nw",
            window=self.content,
        )
        self.content.bind(
            "<Configure>",
            lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.bind(
            "<Configure>",
            lambda event: self.canvas.itemconfigure(self._content_window, width=event.width),
        )
        self.bind_mousewheel_tree(self)

    def bind_mousewheel_tree(self, widget):
        """让鼠标位于栏内任意子控件时都滚动当前栏。"""
        if not getattr(widget, "_column_mousewheel_bound", False):
            widget.bind("<MouseWheel>", self._scroll, add="+")
            widget._column_mousewheel_bound = True
        for child in widget.winfo_children():
            self.bind_mousewheel_tree(child)

    def _scroll(self, event):
        steps = int(-event.delta / 120)
        if steps == 0 and event.delta:
            steps = -1 if event.delta > 0 else 1
        if steps:
            self.canvas.yview_scroll(steps, "units")
        return "break"

    def clear(self):
        for widget in self.content.winfo_children():
            widget.destroy()


class ImportWizard:
    """在一个页面内完成字库信息、目录扫描、字符核对和导入。

    append_mode=True 时进入「字库添加」续加模式：
        - 字库信息从已有的 glyph_service 元数据读取，默认锁定只读
        - 可解锁修改 DPI/尺寸，保存后更新元数据
        - 导入时不覆盖已有元数据（init_meta=False）
    """
    def __init__(
        self,
        parent,
        existing_names: Optional[list[str]] = None,
        on_complete: Optional[Callable[[str], None]] = None,
        append_mode: bool = False,
        glyph_service: Optional[GlyphService] = None,
    ):
        self.parent = parent
        self.existing_names = existing_names or []
        self.on_complete = on_complete
        self._append_mode = append_mode
        self._existing_glyph = glyph_service
        if self._append_mode and self._existing_glyph is None:
            raise ValueError("字库添加必须指定当前字库。")
        self._owns_window = not isinstance(parent, (tk.Frame, tk.Canvas))
        self.win = tk.Toplevel(parent) if self._owns_window else parent
        if self._owns_window:
            self.win.title("字库添加" if self._append_mode else "新建字库")
            self.win.geometry("1460x900")
            self.win.minsize(1180, 720)
            self.win.transient(parent)
            set_window_icon(self.win, config.ICON_FILE)
        metadata = self._existing_glyph.get_metadata() if self._append_mode and self._existing_glyph else {}
        self._result_value: Optional[str] = None
        self._scan_items: list[ScanItem] = []
        self._thumbnail_refs: list[ImageTk.PhotoImage] = []
        self._conversion_lock = False
        self._is_scanning = False
        self._cancelled = False
        self._column_pages = {"正确": 0, "一对一": 0, "歧义": 0}
        self._items_per_page = 200
        self._name_var = tk.StringVar(value=self._existing_glyph.ziku_name if self._append_mode and self._existing_glyph else "")
        self._directory_var = tk.StringVar()
        self._dpi_var = tk.StringVar(value=str(metadata.get("DPI", metadata.get("分辨率", 300))))
        self._width_px_var = tk.StringVar(value=str(metadata.get("画布宽", 250)))
        self._height_px_var = tk.StringVar(value=str(metadata.get("画布高", 250)))
        self._width_mm_var = tk.StringVar(value=f"{float(metadata.get('成品宽度毫米', 21.17)):.2f}")
        self._height_mm_var = tk.StringVar(value=f"{float(metadata.get('成品高度毫米', 21.17)):.2f}")
        self._output_style_var = tk.StringVar(value=str(metadata.get("成品风格", "灰度保真")))
        self._edit_spec_var = tk.BooleanVar(value=False)
        self._spec_entries: list[tk.Entry] = []
        self._existing_hashes = self._load_existing_hashes()
        self._search_var = tk.StringVar()
        self._search_query = ""
        self._status_var = tk.StringVar(value="请选择文字图片目录，然后扫描核对")
        self._scan_progress: Optional[ProgressDialog] = None
        self._build_page()
        self._bind_conversions()
        if self._owns_window:
            self.win.protocol("WM_DELETE_WINDOW", self._cancel)
            self.win.grab_set()


    def result(self) -> Optional[str]:
        return self._result_value

    def show(self) -> Optional[str]:
        if self._owns_window:
            self.parent.wait_window(self.win)
        return self._result_value

    def _build_page(self):
        self.win.configure(bg=theme.BG_MAIN)
        root_frame = tk.Frame(self.win, bg=theme.BG_MAIN)
        root_frame.pack(fill="both", expand=True, padx=22, pady=(14, 10))

        title_row = tk.Frame(root_frame, bg=theme.BG_MAIN)
        title_row.pack(fill="x", pady=(0, 8))
        tk.Label(title_row, text="字库添加" if self._append_mode else "新建字库", bg=theme.BG_MAIN, fg=theme.FG_PRIMARY, font=(theme.FONT_FAMILY, 22, "bold")).pack(side="left")
        subtitle = "向当前字库继续添加文字图片" if self._append_mode else "在本页填写信息、核对文字图片并创建字库"
        tk.Label(title_row, text=subtitle, bg=theme.BG_MAIN, fg=theme.FG_MUTED, font=theme.FONT_SMALL).pack(side="left", padx=18, pady=(7, 0))
        theme.make_button(title_row, "返回首页", command=self._cancel).pack(side="right", padx=10, pady=8)

        info_panel = RoundedPanel(root_frame, background=theme.BG_PANEL, border=theme.BORDER, height=195 if self._append_mode else 170)
        info_panel.pack(fill="x", pady=(0, 8))
        info_panel.pack_propagate(False)
        info_frame = info_panel.content
        tk.Label(info_frame, text="当前字库与成品规格" if self._append_mode else "字库信息与图片目录", bg=theme.BG_PANEL, fg=theme.FG_PRIMARY, font=(theme.FONT_FAMILY, 15, "bold")).grid(row=0, column=0, columnspan=12, sticky="w", pady=(0, 9))
        spec_row = tk.Frame(info_frame, bg=theme.BG_PANEL)
        spec_row.grid(row=1, column=0, columnspan=12, sticky="ew")
        tk.Label(spec_row, text="当前字库" if self._append_mode else "字库名称", bg=theme.BG_PANEL, fg=theme.FG_MUTED, font=theme.FONT_NORMAL).pack(side="left")
        name_entry = tk.Entry(spec_row, textvariable=self._name_var, width=20, bg=theme.BG_INPUT, fg=theme.FG_PRIMARY, insertbackground=theme.FG_PRIMARY, relief="flat", font=theme.FONT_NORMAL)
        name_entry.pack(side="left", padx=(8, 42), ipady=6)
        self._add_spec_input(spec_row, "成品 DPI", self._dpi_var, "DPI", 7)
        self._add_separator(spec_row)
        self._add_size_input(spec_row, "成品宽度", self._width_px_var, self._width_mm_var)
        self._add_separator(spec_row)
        self._add_size_input(spec_row, "成品高度", self._height_px_var, self._height_mm_var)
        tk.Label(spec_row, text="成品风格", bg=theme.BG_PANEL, fg=theme.FG_MUTED, font=theme.FONT_NORMAL).pack(side="left", padx=(28, 6))
        style_menu = tk.OptionMenu(spec_row, self._output_style_var, "灰度保真", "纯二值", "统一软边")
        style_menu.configure(bg=theme.BG_INPUT, fg=theme.FG_PRIMARY, activebackground=theme.BTN_HOVER, activeforeground=theme.FG_PRIMARY, relief="flat", highlightthickness=0, font=theme.FONT_SMALL)
        style_menu["menu"].configure(bg=theme.BG_INPUT, fg=theme.FG_PRIMARY, activebackground=theme.FG_ACCENT, activeforeground=theme.FG_PRIMARY, font=theme.FONT_SMALL)
        style_menu.pack(side="left")
        if self._append_mode:
            style_menu.configure(state="disabled")
            tk.Checkbutton(
                spec_row, text="修改字库规格", variable=self._edit_spec_var, command=self._toggle_spec_editing,
                bg=theme.BG_PANEL, fg=theme.FG_SECONDARY, activebackground=theme.BG_PANEL,
                activeforeground=theme.FG_PRIMARY, selectcolor=theme.BG_INPUT, font=theme.FONT_SMALL,
            ).pack(side="right", padx=(14, 0))
            self._toggle_spec_editing()
            tk.Label(
                info_frame, text="默认仅显示字库现有规格；开启修改后，变更将作用于整个字库，并清除旧规格成品等待重新协调。",
                bg=theme.BG_PANEL, fg=theme.FG_MUTED, font=theme.FONT_SMALL,
            ).grid(row=2, column=0, columnspan=12, sticky="w", pady=(7, 0))
        directory_row = 3 if self._append_mode else 2
        tk.Label(info_frame, text="文字图片目录", bg=theme.BG_PANEL, fg=theme.FG_MUTED, font=theme.FONT_NORMAL).grid(row=directory_row, column=0, sticky="w", pady=(12, 0))
        tk.Entry(info_frame, textvariable=self._directory_var, bg=theme.BG_INPUT, fg=theme.FG_PRIMARY, insertbackground=theme.FG_PRIMARY, relief="flat", font=theme.FONT_NORMAL).grid(row=directory_row, column=1, columnspan=8, sticky="ew", padx=(8, 8), pady=(12, 0), ipady=7)
        choose_directory_button = theme.make_button(info_frame, "选择目录", command=self._choose_directory, width=9, anchor="center")
        choose_directory_button.configure(pady=7)
        choose_directory_button.grid(row=directory_row, column=9, padx=4, pady=(12, 0))
        self._scan_button = theme.make_button(info_frame, "扫描并核对", accent=True, command=self._scan, width=13, anchor="center")
        self._scan_button.configure(pady=7)
        self._scan_button.grid(row=directory_row, column=10, padx=4, pady=(12, 0))
        tk.Label(info_frame, textvariable=self._status_var, bg=theme.BG_PANEL, fg=theme.COLOR_FINALIZED, font=theme.FONT_SMALL).grid(row=directory_row, column=11, sticky="w", padx=(8, 0), pady=(12, 0))
        for column in range(12):
            info_frame.grid_columnconfigure(column, weight=1 if column in (1, 7) else 0)

        toolbar = tk.Frame(root_frame, bg=theme.BG_MAIN)
        toolbar.pack(fill="x", pady=(0, 6))
        tk.Label(toolbar, text="图片字符核对", bg=theme.BG_MAIN, fg=theme.FG_PRIMARY, font=(theme.FONT_FAMILY, 14, "bold")).pack(side="left")
        search_box = tk.Frame(toolbar, bg=theme.BG_MAIN)
        search_box.pack(side="left", padx=18)
        search_entry = tk.Entry(search_box, textvariable=self._search_var, bg=theme.BG_INPUT, fg=theme.FG_PRIMARY, insertbackground=theme.FG_PRIMARY, relief="flat", width=24, font=theme.FONT_NORMAL)
        search_entry.pack(side="left", ipady=6)
        search_entry.bind("<Return>", lambda _event: self._run_search())
        theme.make_button(search_box, "🔍", command=self._run_search, width=3).pack(side="left", padx=(4, 0), ipady=2)
        self._stats_label = tk.Label(toolbar, text="正确 0　·　一对一 0　·　有歧义 0", bg=theme.BG_MAIN, fg=theme.FG_MUTED, font=theme.FONT_SMALL)
        self._stats_label.pack(side="right")

        columns_frame = tk.Frame(root_frame, bg=theme.BG_MAIN)
        columns_frame.pack(fill="both", expand=True)
        columns_frame.grid_columnconfigure((0, 1, 2), weight=1, uniform="三栏")
        columns_frame.grid_rowconfigure(0, weight=1)
        self._columns: dict[str, ScrollableColumn] = {}
        column_config = (("正确", "第一栏　正确的", theme.COLOR_FINALIZED), ("一对一", "第二栏　明确一对一", theme.FG_ACCENT), ("歧义", "第三栏　存在歧义，需要确认", theme.COLOR_DRAFT))
        for column, (category, title, color) in enumerate(column_config):
            panel = RoundedPanel(columns_frame, background=theme.BG_PANEL, border=color)
            panel.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 5, 0 if column == 2 else 5))
            tk.Label(panel.content, text=title, bg=theme.BG_PANEL, fg=theme.FG_PRIMARY, font=(theme.FONT_FAMILY, 14, "bold")).pack(anchor="w")
            description = "默认采用原字符，可逐项核对修改" if category == "正确" else ("可选择原字符或建议繁体，也可直接修改" if category == "一对一" else "必须查看图片并选择最终字符")
            tk.Label(panel.content, text=description, bg=theme.BG_PANEL, fg=theme.FG_MUTED, font=theme.FONT_SMALL).pack(anchor="w", pady=(3, 8))
            scroll = ScrollableColumn(panel.content, background=theme.BG_PANEL)
            scroll.pack(fill="both", expand=True)
            self._columns[category] = scroll

        footer = tk.Frame(root_frame, bg=theme.BG_MAIN)
        footer.pack(fill="x", pady=(10, 0))
        self._footer_hint = tk.Label(footer, text="源目录和源文件名不会被修改", bg=theme.BG_MAIN, fg=theme.FG_MUTED, font=theme.FONT_SMALL)
        self._footer_hint.pack(side="left")
        button_text = "确认并导入字图" if self._append_mode else "确认并创建字库"
        self._create_button = theme.make_button(footer, button_text, accent=True, command=self._create, width=16)
        self._create_button.pack(side="right")
        self._set_create_enabled(False)

    # ── 以下为复用方法 ──

    @staticmethod
    def _add_separator(parent):
        tk.Label(parent, text="│", bg=theme.BG_PANEL, fg=theme.BORDER, font=theme.FONT_NORMAL).pack(side="left", padx=14)

    def _add_spec_input(self, parent, text, variable, unit, entry_width):
        tk.Label(parent, text=text, bg=theme.BG_PANEL, fg=theme.FG_MUTED, font=theme.FONT_NORMAL).pack(side="left")
        entry = tk.Entry(parent, textvariable=variable, width=entry_width, bg=theme.BG_INPUT, fg=theme.FG_PRIMARY, insertbackground=theme.FG_PRIMARY, relief="flat", justify="center")
        entry.pack(side="left", padx=(7, 3), ipady=6)
        self._spec_entries.append(entry)
        tk.Label(parent, text=unit, bg=theme.BG_PANEL, fg=theme.FG_MUTED, font=theme.FONT_SMALL).pack(side="left")

    def _add_size_input(self, parent, text, pixel_var, millimeter_var):
        tk.Label(parent, text=text, bg=theme.BG_PANEL, fg=theme.FG_MUTED, font=theme.FONT_NORMAL).pack(side="left")
        pixel_entry = tk.Entry(parent, textvariable=pixel_var, width=6, bg=theme.BG_INPUT, fg=theme.FG_PRIMARY, insertbackground=theme.FG_PRIMARY, relief="flat", justify="center")
        pixel_entry.pack(side="left", padx=(7, 2), ipady=6)
        self._spec_entries.append(pixel_entry)
        tk.Label(parent, text="像素 ⇄", bg=theme.BG_PANEL, fg=theme.FG_MUTED, font=theme.FONT_SMALL).pack(side="left")
        millimeter_entry = tk.Entry(parent, textvariable=millimeter_var, width=7, bg=theme.BG_INPUT, fg=theme.FG_PRIMARY, insertbackground=theme.FG_PRIMARY, relief="flat", justify="center")
        millimeter_entry.pack(side="left", padx=2, ipady=6)
        self._spec_entries.append(millimeter_entry)
        tk.Label(parent, text="毫米", bg=theme.BG_PANEL, fg=theme.FG_MUTED, font=theme.FONT_SMALL).pack(side="left")

    def _toggle_spec_editing(self):
        status_var = "normal" if not self._append_mode or self._edit_spec_var.get() else "disabled"
        for entry in self._spec_entries:
            entry.configure(
                state=status_var,
                disabledbackground=theme.BG_INPUT,
                disabledforeground=theme.FG_SECONDARY,
            )

    def _load_existing_hashes(self) -> set[str]:
        if not self._append_mode or not self._existing_glyph:
            return set()
        return {
            str(detail.get("原始MD5", ""))
            for detail in self._existing_glyph.get_all_variants()
            if detail.get("原始MD5")
        }

    def _bind_conversions(self):
        self._dpi_var.trace_add("write", lambda *_: self._pixels_to_millimeters())
        self._width_px_var.trace_add("write", lambda *_: self._pixels_to_millimeters())
        self._height_px_var.trace_add("write", lambda *_: self._pixels_to_millimeters())
        self._width_mm_var.trace_add("write", lambda *_: self._millimeters_to_pixels("宽"))
        self._height_mm_var.trace_add("write", lambda *_: self._millimeters_to_pixels("高"))

    def _pixels_to_millimeters(self):
        if self._conversion_lock:
            return
        try:
            dpi, width, height = int(self._dpi_var.get()), int(self._width_px_var.get()), int(self._height_px_var.get())
            if min(dpi, width, height) <= 0:
                return
        except ValueError:
            return
        self._conversion_lock = True
        self._width_mm_var.set(f"{width / dpi * 25.4:.2f}")
        self._height_mm_var.set(f"{height / dpi * 25.4:.2f}")
        self._conversion_lock = False

    def _millimeters_to_pixels(self, dimension: str):
        if self._conversion_lock:
            return
        try:
            dpi = int(self._dpi_var.get())
            millimeters = float(self._width_mm_var.get() if dimension == "宽" else self._height_mm_var.get())
            if dpi <= 0 or millimeters <= 0:
                return
        except ValueError:
            return
        self._conversion_lock = True
        (self._width_px_var if dimension == "宽" else self._height_px_var).set(str(round(millimeters / 25.4 * dpi)))
        self._conversion_lock = False

    def _choose_directory(self):
        directory_var = filedialog.askdirectory(parent=self.win, title="选择文字图片目录")
        if directory_var:
            self._directory_var.set(directory_var)
            if not self._name_var.get().strip():
                self._name_var.set(os.path.basename(directory_var.rstrip("/\\")))
            self._scan()

    def _scan(self):
        directory_var = self._directory_var.get().strip()
        if not os.path.isdir(directory_var):
            show_warning(self.win, "提示", "请选择有效的文字图片目录。")
            return
        supported_extensions = ImportService.SUPPORTED_EXTENSIONS
        source_paths = [item.path for item in sorted(os.scandir(directory_var), key=lambda x: x.name) if item.is_file() and os.path.splitext(item.name)[1].lower() in supported_extensions]
        if not source_paths:
            show_warning(self.win, "提示", "所选目录中没有支持的图片文件。")
            return

        self._is_scanning = True
        self._cancelled = False
        self._scan_button.configure(state="disabled")
        self._set_create_enabled(False)
        self._close_scan_progress()
        self._scan_progress = ProgressDialog(
            self.win,
            "扫描核对",
            len(source_paths),
            cancellable=True,
        )
        self._scan_progress.set_cancel_callback(self._cancel_scan)
        self._scan_progress.protocol("WM_DELETE_WINDOW", self._cancel_scan)
        self._scan_progress.update_progress(0, f"准备扫描，共 {len(source_paths)} 张图片")

        def worker():
            try:
                service = ImportService.__new__(ImportService)
                items: list[ScanItem] = []
                total = len(source_paths)
                for current, path in enumerate(source_paths, 1):
                    if self._cancelled:
                        return
                    filename = os.path.basename(path)
                    original_char = service._extract_char(filename)
                    category, candidates = identify_character(original_char)
                    final_char = candidates[0] if category == "一对一" else ("" if category == "歧义" else original_char)
                    if self._append_mode and compute_file_md5(path) in self._existing_hashes:
                        filename = f"{filename}　（字库已有相同图片，将跳过）"
                    thumbnail = self._load_thumbnail(path)
                    items.append(ScanItem(os.path.abspath(path), filename, original_char, category, candidates, final_char, category != "歧义", thumbnail))
                    if current % 10 == 0 or current == total:
                        self._safe_schedule(lambda count=current, file_label=filename: self._update_scan_progress(count, total, file_label))
                self._safe_schedule(lambda: self._scan_complete(items))
            except Exception as exc:
                self._safe_schedule(lambda error=exc: self._scan_failed(error))

        threading.Thread(target=worker, daemon=True).start()

    def _safe_schedule(self, callback: Callable[[], None]):
        if self._cancelled:
            return
        try:
            self.win.after(0, callback)
        except (tk.TclError, RuntimeError):
            pass

    def _update_scan_progress(self, current: int, total: int, filename: str):
        if self._cancelled:
            return
        if self._scan_progress and self._scan_progress.winfo_exists():
            self._scan_progress.update_progress(current, f"正在扫描 {current}/{total}　{filename}")

    def _scan_complete(self, items: list[ScanItem]):
        if self._cancelled:
            return
        self._scan_items = items
        self._column_pages = {"正确": 0, "一对一": 0, "歧义": 0}
        if self._scan_progress and self._scan_progress.winfo_exists():
            self._scan_progress.update_progress(len(items), "正在生成核对列表……")
        self._refresh_columns()
        self._close_scan_progress()
        self._is_scanning = False
        self._scan_button.configure(state="normal")
        self._status_var.set(f"已扫描 {len(items)} 张")

    def _scan_failed(self, exc: Exception):
        if self._cancelled:
            return
        self._close_scan_progress()
        self._is_scanning = False
        self._scan_button.configure(state="normal")
        self._status_var.set("扫描失败")
        show_error(self.win, "扫描失败", str(exc))

    def _close_scan_progress(self):
        if self._scan_progress and self._scan_progress.winfo_exists():
            self._scan_progress.destroy()
        self._scan_progress = None

    def _cancel_scan(self):
        self._cancelled = True
        self._is_scanning = False
        self._close_scan_progress()
        self._scan_button.configure(state="normal")
        self._status_var.set("扫描已取消")

    def _run_search(self):
        if not self._cancelled:
            self._search_query = self._search_var.get().strip().lower()
            self._column_pages = {category: 0 for category in self._column_pages}
            self._refresh_columns()

    def _refresh_columns(self):
        for scroll in self._columns.values():
            scroll.clear()
        self._thumbnail_refs.clear()
        search_var = self._search_query
        items_by_category: dict[str, list[ScanItem]] = {"正确": [], "一对一": [], "歧义": []}
        for item in self._scan_items:
            if search_var and search_var not in item.filename.lower() and search_var not in item.original_char and search_var not in item.final_char:
                continue
            items_by_category[item.category].append(item)
        for category, items in items_by_category.items():
            items.sort(key=lambda item: pinyin_natural_key(item.final_char or item.original_char or item.filename))
            total = len(items)
            total_pages = max(1, (total + self._items_per_page - 1) // self._items_per_page)
            current_page = min(self._column_pages[category], total_pages - 1)
            self._column_pages[category] = current_page
            start = current_page * self._items_per_page
            end = min(start + self._items_per_page, total)
            for item in items[start:end]:
                self._add_card(self._columns[category].content, item)
            if total_pages > 1:
                self._add_pagination(category, current_page, total_pages, start + 1, end, total)
            self._columns[category].bind_mousewheel_tree(self._columns[category].content)
        self._update_stats_status({category: len(items) for category, items in items_by_category.items()}, bool(search_var))

    def _add_pagination(self, category: str, current_page: int, total_pages: int, start: int, end: int, total: int):
        container = tk.Frame(self._columns[category].content, bg=theme.BG_PANEL)
        container.pack(fill="x", padx=3, pady=8)
        tk.Label(
            container,
            text=f"当前显示 {start}～{end}，共 {total} 个",
            bg=theme.BG_PANEL,
            fg=theme.FG_MUTED,
            font=theme.FONT_SMALL,
        ).pack(pady=(2, 5))
        button_row = tk.Frame(container, bg=theme.BG_PANEL)
        button_row.pack()
        previous_button = theme.make_button(button_row, "上一批", command=lambda selected_category=category: self._change_page(selected_category, -1), width=10)
        previous_button.pack(side="left", padx=3)
        if current_page == 0:
            previous_button.configure(state="disabled")
        next_button = theme.make_button(button_row, "下一批", command=lambda selected_category=category: self._change_page(selected_category, 1), width=10)
        next_button.pack(side="left", padx=3)
        if current_page >= total_pages - 1:
            next_button.configure(state="disabled")

    def _change_page(self, category: str, dimension: int):
        self._column_pages[category] = max(0, self._column_pages[category] + dimension)
        self._refresh_columns()
        self._columns[category].canvas.yview_moveto(0.0)

    def _add_card(self, parent, item: ScanItem):
        border = theme.COLOR_FINALIZED if item.category == "正确" else (theme.FG_ACCENT if item.category == "一对一" else theme.COLOR_DRAFT)
        card = tk.Frame(parent, bg=theme.BG_CANVAS, highlightbackground=border, highlightthickness=1)
        card.pack(fill="x", padx=3, pady=3)
        image = self._create_thumbnail(item)
        image_label = tk.Label(card, image=image, bg=theme.BG_CANVAS, cursor="hand2")
        image_label.grid(row=0, column=0, rowspan=3, padx=7, pady=7)
        image_label.bind("<Button-1>", lambda _e, item=item: self._show_preview(item))
        tk.Label(card, text=item.filename, bg=theme.BG_CANVAS, fg=theme.FG_PRIMARY, font=theme.FONT_NORMAL, anchor="w").grid(row=0, column=1, columnspan=4, sticky="ew", pady=(7, 1))
        tk.Label(card, text=f"原字符：{item.original_char}", bg=theme.BG_CANVAS, fg=theme.FG_MUTED, font=theme.FONT_SMALL).grid(row=1, column=1, sticky="w")
        if item.category in ("正确", "一对一"):
            tk.Label(card, text="最终字符", bg=theme.BG_CANVAS, fg=theme.FG_MUTED, font=theme.FONT_SMALL).grid(row=1, column=2, padx=(12, 4))
            variable = tk.StringVar(value=item.final_char)
            entry = tk.Entry(card, textvariable=variable, width=4, justify="center", bg=theme.BG_INPUT, fg=theme.FG_PRIMARY, insertbackground=theme.FG_PRIMARY, relief="flat", font=(theme.FONT_FAMILY, 16, "bold"))
            entry.grid(row=1, column=3, ipady=3)
            error_label = tk.Label(card, text="", bg=theme.BG_CANVAS, fg=theme.BTN_DANGER, font=theme.FONT_SMALL)
            error_label.grid(row=1, column=4, padx=(6, 4), sticky="w")
            variable.trace_add("write", lambda *_args, item=item, value=variable, widget=entry, label=error_label: self._change_final_char(item, value.get(), widget, label))
            if item.category == "一对一":
                suggestion = item.candidates[0] if item.candidates else item.final_char
                tk.Button(card, text=f"原字 {item.original_char}", command=lambda item=item, value=variable: self._select_final_char(item, item.original_char, value), bg=theme.BG_INPUT, fg=theme.FG_PRIMARY, activebackground=theme.BTN_HOVER, activeforeground=theme.FG_PRIMARY, relief="flat", bd=0, cursor="hand2", font=theme.FONT_SMALL, padx=5, pady=2).grid(row=2, column=2, padx=(12, 2), pady=(2, 7))
                tk.Button(card, text=f"建议 {suggestion}", command=lambda item=item, char=suggestion, value=variable: self._select_final_char(item, char, value), bg=theme.BG_INPUT, fg=theme.FG_PRIMARY, activebackground=theme.BTN_HOVER, activeforeground=theme.FG_PRIMARY, relief="flat", bd=0, cursor="hand2", font=theme.FONT_SMALL, padx=5, pady=2).grid(row=2, column=3, padx=2, pady=(2, 7))
        else:
            status_label = tk.Label(card, text="请选择：" if not item.confirmed else f"已确认：{item.final_char}", bg=theme.BG_CANVAS, fg=theme.COLOR_DRAFT if not item.confirmed else theme.COLOR_FINALIZED, font=theme.FONT_SMALL)
            status_label.grid(row=1, column=2, padx=(10, 3))
            values = list(dict.fromkeys((*item.candidates, item.original_char)))[:4]
            buttons: list[tuple[str, tk.Button]] = []
            for index, value in enumerate(values):
                button = tk.Button(card, text=value, bg=theme.BG_INPUT if not item.confirmed or item.final_char != value else theme.FG_ACCENT, fg=theme.FG_PRIMARY, activebackground=theme.BTN_HOVER, activeforeground=theme.FG_PRIMARY, relief="flat", bd=0, cursor="hand2", font=(theme.FONT_FAMILY, 12, "bold"), padx=7, pady=2)
                button.grid(row=2, column=2 + index, padx=2, pady=(2, 7))
                buttons.append((value, button))
            for value, button in buttons:
                button.configure(command=lambda char=value, item=item, label=status_label, button_group=buttons: self._confirm_ambiguity(item, char, label, button_group))
        card.grid_columnconfigure(1, weight=1)

    @staticmethod
    def _load_thumbnail(path: str) -> Image.Image:
        try:
            with Image.open(path) as source_image:
                display_image = ImageOps.exif_transpose(source_image).convert("RGBA")
                display_image.thumbnail((56, 56), Image.Resampling.LANCZOS)
                background = Image.new("RGBA", (60, 60), "#f3f1ec")
                background.alpha_composite(display_image, ((60 - display_image.width) // 2, (60 - display_image.height) // 2))
                return background
        except Exception:
            return Image.new("RGBA", (60, 60), "#3a2630")

    def _create_thumbnail(self, item: ScanItem):
        background = item.thumbnail if item.thumbnail is not None else self._load_thumbnail(item.path)
        photo = ImageTk.PhotoImage(background)
        self._thumbnail_refs.append(photo)
        return photo

    def _show_preview(self, item: ScanItem):
        preview = tk.Toplevel(self.win)
        preview.title(f"图片核对 - {item.filename}")
        preview.configure(bg=theme.BG_MAIN)
        set_window_icon(preview, config.ICON_FILE)
        try:
            with Image.open(item.path) as source_image:
                display_image = ImageOps.exif_transpose(source_image).convert("RGBA")
                display_image.thumbnail((700, 620), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(display_image)
            label = tk.Label(preview, image=photo, bg=theme.BG_MAIN)
            label.image = photo
            label.pack(padx=18, pady=18)
        except Exception as exc:
            tk.Label(preview, text=f"图片无法预览：{exc}", bg=theme.BG_MAIN, fg=theme.BTN_DANGER).pack(padx=30, pady=30)

    def _change_final_char(self, item: ScanItem, value: str, entry: tk.Entry, error_label: tk.Label):
        item.final_char = value.strip()
        valid, message = validate_final_char(item.final_char)
        entry.configure(
            highlightthickness=2 if not valid else 1,
            highlightbackground=theme.BG_INPUT if valid else theme.BTN_DANGER,
            highlightcolor=theme.FG_ACCENT if valid else theme.BTN_DANGER,
        )
        error_label.configure(text="" if valid else message)
        self._update_stats_status()

    def _select_final_char(self, item: ScanItem, char: str, variable: tk.StringVar):
        item.final_char = char
        variable.set(char)

    def _confirm_ambiguity(self, item: ScanItem, char: str, status_label: tk.Label, buttons: list[tuple[str, tk.Button]]):
        item.final_char = char
        item.confirmed = True
        status_label.configure(text=f"已确认：{char}", fg=theme.COLOR_FINALIZED)
        for value, button in buttons:
            button.configure(bg=theme.FG_ACCENT if value == char else theme.BG_INPUT)
        self._update_stats_status()

    def _update_stats_status(self, display_counts: Optional[dict[str, int]] = None, is_searching: bool = False):
        all_counts = {"正确": 0, "一对一": 0, "歧义": 0}
        for item in self._scan_items:
            all_counts[item.category] += 1
        counts = display_counts if display_counts is not None else all_counts
        prefix = "当前显示：" if is_searching else ""
        self._stats_label.configure(text=f"{prefix}正确 {counts['正确']}　·　一对一 {counts['一对一']}　·　有歧义 {counts['歧义']}")
        unconfirmed_items = [item for item in self._scan_items if item.category == "歧义" and (not item.confirmed or not item.final_char.strip())]
        invalid_items = [item for item in self._scan_items if not validate_final_char(item.final_char)[0]]
        if invalid_items:
            invalid_names = "、".join(item.filename for item in invalid_items[:4])
            ellipsis = "……" if len(invalid_items) > 4 else ""
            self._footer_hint.configure(
                text=f"有 {len(invalid_items)} 个最终字符不合法：{invalid_names}{ellipsis}",
                fg=theme.BTN_DANGER,
            )
        elif unconfirmed_items:
            unconfirmed_chars = "、".join(item.original_char or item.filename for item in unconfirmed_items[:6])
            ellipsis = "……" if len(unconfirmed_items) > 6 else ""
            self._footer_hint.configure(
                text=f"还有 {len(unconfirmed_items)} 个歧义项目未确认：{unconfirmed_chars}{ellipsis}",
                fg=theme.FG_MUTED,
            )
        else:
            self._footer_hint.configure(text="所有歧义项目均已确认，源目录不会被修改", fg=theme.FG_MUTED)
        self._set_create_enabled(bool(self._scan_items) and not unconfirmed_items and not invalid_items)

    def _set_create_enabled(self, enabled: bool):
        self._create_button.configure(state="normal" if enabled else "disabled", bg=theme.FG_ACCENT if enabled else theme.BORDER, activebackground=theme.BTN_HOVER)

    def _validate_info(self):
        name_var = self._name_var.get().strip()
        if not name_var:
            return None, "请输入字库名称。"
        current_name = self._existing_glyph.ziku_name if self._existing_glyph is not None else ""
        name_exists = name_var in self.existing_names or os.path.exists(os.path.join(config.ZIKU_ROOT, name_var))
        if name_var != current_name and name_exists:
            return None, "该字库名称已存在，请换一个名称。"
        if any(char in '<>:"/\\|?*' for char in name_var) or name_var.endswith((" ", ".")):
            return None, "字库名称包含不能用于目录名的字符，或以空格、句点结尾。"
        try:
            dpi, width, height = int(self._dpi_var.get()), int(self._width_px_var.get()), int(self._height_px_var.get())
            width_mm_var, height_mm_var = float(self._width_mm_var.get()), float(self._height_mm_var.get())
            if min(dpi, width, height, width_mm_var, height_mm_var) <= 0:
                raise ValueError
        except ValueError:
            return None, "DPI、像素尺寸和物理尺寸必须是大于零的数字。"
        return (name_var, dpi, width, height, width_mm_var, height_mm_var), ""

    def _create(self):
        info_frame, error = self._validate_info()
        if not info_frame:
            show_warning(self.win, "信息不完整", error)
            return
        if not self._scan_items:
            show_warning(self.win, "尚未扫描", "请先选择目录并扫描图片。")
            return
        unconfirmed = [item for item in self._scan_items if item.category == "歧义" and not item.confirmed]
        if unconfirmed:
            show_warning(self.win, "尚未确认", f"还有 {len(unconfirmed)} 个歧义项目需要确认。")
            return
        invalid_items = []
        for item in self._scan_items:
            valid, message = validate_final_char(item.final_char)
            if not valid:
                invalid_items.append((item, message))
        if invalid_items:
            first_item, message = invalid_items[0]
            show_warning(
                self.win,
                "字符不合法",
                f"“{os.path.basename(first_item.path)}”的{message}\n请修正后再继续。",
            )
            return
        name_var, dpi, width, height, width_mm_var, height_mm_var = info_frame
        if self._append_mode:
            assert self._existing_glyph is not None
            glyph_service = self._existing_glyph
            ziku_dir = glyph_service.ziku_dir
        else:
            ziku_dir = os.path.join(config.ZIKU_ROOT, name_var)
            os.makedirs(ziku_dir, exist_ok=False)
            glyph_service = GlyphService(name_var, ziku_dir)
        progress_dialog = ProgressDialog(self.win, title="正在导入字图" if self._append_mode else "正在创建字库", total=len(self._scan_items))
        char_overrides = {item.path: item.final_char for item in self._scan_items}

        def worker():
            try:
                invalidated_count = 0
                result_ziku_dir = ziku_dir
                if self._append_mode and name_var != glyph_service.ziku_name:
                    result_ziku_dir = glyph_service.rename_ziku(name_var)
                if self._append_mode and self._edit_spec_var.get():
                    invalidated_count = glyph_service.update_output_spec(dpi, width, height, width_mm_var, height_mm_var)

                def report_progress(message, current, total):
                    filename = os.path.basename(self._scan_items[min(current - 1, len(self._scan_items) - 1)].path)
                    self.win.after(0, lambda: progress_dialog.update_progress(current, message, filename))

                import_service = ImportService(glyph_service, progress_callback=report_progress)
                result_value = import_service.import_batch(
                    self._directory_var.get(), dpi, width, height, char_overrides=char_overrides,
                    width_mm=width_mm_var, height_mm=height_mm_var, init_meta=not self._append_mode,
                    output_style=self._output_style_var.get(),
                )
                result_value["失效成品数"] = invalidated_count
                self.win.after(0, lambda: self._import_complete(progress_dialog, name_var, result_ziku_dir, result_value))
            except Exception as exc:
                failed_dir = glyph_service.ziku_dir
                self.win.after(0, lambda error=exc, path=failed_dir: self._import_failed(progress_dialog, path, error))
        threading.Thread(target=worker, daemon=True).start()

    def _import_complete(self, progress_dialog, name_var, ziku_dir, result_value):
        progress_dialog.finish()
        self._result_value = name_var
        skipped_details = [detail for detail in result_value.get("详情", []) if detail.get("状态") == "跳过"]
        action = "字库添加完成" if self._append_mode else f"字库「{name_var}」创建完成"
        prompt = f"{action}。\n\n成功：{result_value['成功']}\n跳过：{result_value['跳过']}\n失败：{result_value['失败']}"
        if result_value.get("失效成品数"):
            prompt += f"\n规格已更新，{result_value['失效成品数']} 个旧规格成品已清除，等待重新整体协调。"
        if skipped_details:
            file_list = "、".join(os.path.basename(detail["路径"]) for detail in skipped_details)
            prompt += f"\n\n跳过原因：图片内容完全重复。\n跳过文件：{file_list}"
        show_info(self.win, "导入完成" if self._append_mode else "创建完成", prompt)
        if self.on_complete:
            self.on_complete(name_var)
        if self._owns_window:
            self.win.destroy()

    def _import_failed(self, progress_dialog, ziku_dir, exc):
        progress_dialog.finish()
        if not self._append_mode:
            shutil.rmtree(ziku_dir, ignore_errors=True)
        show_error(self.win, "导入失败" if self._append_mode else "创建失败", str(exc))

    def _cancel(self):
        self._cancelled = True
        self._is_scanning = False
        self._close_scan_progress()
        self._result_value = None
        self._scan_items.clear()
        self._thumbnail_refs.clear()
        if self._owns_window:
            self.win.destroy()
        elif self.on_complete:
            callback = self.on_complete
            self.on_complete = None
            self.win.after_idle(lambda: callback(""))
