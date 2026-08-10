# welcome_page.py — 引导页：心经水印 + 字库选择与阶段入口

import os
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable

from send2trash import send2trash

import config
from services.glyph_service import GlyphService
from ui import theme
from ui.widgets.custom_dialog import ask_yes_no, show_error, show_info, show_warning
from utils.file_utils import pinyin_natural_key, safe_read_json


# 首页字体独立于全局主题，适配 1920×1080 等常见桌面分辨率。
_HOME_FONT_BODY = (theme.FONT_FAMILY, 11)
_HOME_FONT_BODY_BOLD = (theme.FONT_FAMILY, 11, "bold")
_HOME_FONT_DETAIL = (theme.FONT_FAMILY, 10)
_HOME_FONT_SECTION = (theme.FONT_FAMILY, 13, "bold")
_HOME_FONT_METRIC = (theme.FONT_FAMILY, 14, "bold")


class _RoundedPanel(tk.Canvas):
    """使用画布绘制圆角背景，并提供可放置控件的内容区。"""

    def __init__(
        self,
        parent: tk.Widget,
        *,
        fill: str,
        border: str,
        height: int,
        radius: int = 12,
        cursor: str = "",
    ) -> None:
        parent_bg = str(parent.cget("bg"))
        super().__init__(
            parent,
            bg=parent_bg,
            height=height,
            highlightthickness=0,
            bd=0,
            cursor=cursor,
        )
        self._fill = fill
        self._border = border
        self._radius = radius
        self.body = tk.Frame(self, bg=fill, cursor=cursor)
        self._body_window = self.create_window(0, 0, anchor="nw", window=self.body)
        self.bind("<Configure>", self._redraw)

    @staticmethod
    def _rounded_points(x1: int, y1: int, x2: int, y2: int, radius: int) -> list[int]:
        radius = max(1, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
        return [
            x1 + radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1,
        ]

    def _redraw(self, event: tk.Event) -> None:
        width = max(2, event.width)
        height = max(2, event.height)
        self.delete("圆角背景")
        self.create_polygon(
            self._rounded_points(1, 1, width - 1, height - 1, self._radius),
            smooth=True,
            splinesteps=24,
            fill=self._fill,
            outline=self._border,
            width=1,
            tags="圆角背景",
        )
        inset = max(3, self._radius // 2)
        self.coords(self._body_window, inset, inset)
        self.itemconfigure(
            self._body_window,
            width=max(1, width - inset * 2),
            height=max(1, height - inset * 2),
        )
        self.tag_lower("圆角背景")

    def set_border(self, color: str) -> None:
        self._border = color
        self.itemconfigure("圆角背景", outline=color)


class WelcomePage(tk.Frame):
    """引导页：保留原有心经竖排水印，通过字库选择进入各处理阶段。"""

    def __init__(
        self,
        parent: tk.Widget,
        on_new_ziku: Callable[[], None],
        on_open_lab: Callable[[], None],
        on_open_stats: Callable[[], None],
        on_open_layout: Callable[[], None],
        on_open_help: Callable[[], None],
        on_open_settings: Callable[[], None],
        on_open_import: Callable[[str, str], None],
        on_open_optimization: Callable[[str, str], None],
        on_open_review: Callable[[str, str], None],
        on_open_consistency: Callable[[str, str], None],
        on_export_ziku: Callable[[str, str], None],
        initial_ziku_name: str = "",
    ) -> None:
        super().__init__(parent, bg=theme.BG_MAIN)
        self._new_cb = on_new_ziku
        self._open_lab_cb = on_open_lab
        self._open_stats_cb = on_open_stats
        self._open_layout_cb = on_open_layout
        self._open_help_cb = on_open_help
        self._open_settings_cb = on_open_settings
        self._stage_callbacks = {
            "import": on_open_import,
            "optimization": on_open_optimization,
            "review": on_open_review,
            "consistency": on_open_consistency,
            "export": on_export_ziku,
        }
        self._watermark_fonts: dict[tuple[int], object] = {}
        self._ziku_items: dict[str, tuple[str, str]] = {}
        self._selected_name = tk.StringVar(value=initial_ziku_name)
        self._editing_name = ""
        self._edit_vars: dict[str, tk.StringVar] = {}
        self._conversion_lock = False

        # 下层仍使用原来的全屏心经水印画布。
        self._bg_canvas = tk.Canvas(self, bg=theme.BG_MAIN, highlightthickness=0)
        self._bg_canvas.place(relx=0, rely=0, relwidth=1, relheight=1)

        # 上层采用方案图中的宽幅分区布局，四周留白继续透出心经水印。
        self._content = tk.Frame(self, bg=theme.BG_MAIN)
        self._content.place(relx=0.5, rely=0.5, relwidth=0.92, anchor="center")
        self._build_content()
        self._refresh_ziku_list()

        self.bind("<Configure>", self._on_resize)
        self.after(100, self._redraw_watermark)

    def _build_content(self) -> None:
        inner = tk.Frame(self._content, bg=theme.BG_MAIN)
        inner.pack(fill=tk.X)

        title_row = tk.Frame(inner, bg=theme.BG_MAIN)
        title_row.pack(fill=tk.X, pady=(0, 14))
        theme.make_label(
            title_row,
            "欢迎使用字库编辑器",
            font=(theme.FONT_FAMILY, 20, "bold"),
            fg="#f6f8fc",
        ).pack(anchor="w")
        theme.make_label(
            title_row,
            "请选择功能，或从字库列表中选择项目继续制作流程",
            fg="#8e99aa",
            font=_HOME_FONT_BODY,
        ).pack(anchor="w", pady=(3, 0))

        tools = tk.Frame(inner, bg=theme.BG_MAIN)
        tools.pack(fill=tk.X, pady=(0, 15))
        tool_defs = (
            ("＋", "新建字库", "创建新的字库项目", "#315f9a", "#c8ddff", self._new_cb),
            ("模", "模板工坊", "创建和管理处理模板", "#2d426d", "#9cbdff", self._open_lab_cb),
            ("统", "文字统计", "查看字库与字符统计", "#294d43", "#78d2ad", self._open_stats_cb),
            ("排", "经文排版", "使用字库进行经文排版", "#493b66", "#c6a9ef", self._open_layout_cb),
            ("?", "使用说明", "查看操作方法和说明", "#554630", "#e8c27d", self._open_help_cb),
            ("设", "设置", "目录、显示和程序设置", "#3d4654", "#b7c3d4", self._open_settings_cb),
        )
        for column, definition in enumerate(tool_defs):
            tools.grid_columnconfigure(column, weight=1, uniform="功能入口")
            self._build_tool_card(tools, column, len(tool_defs), *definition)

        selector = _RoundedPanel(inner, fill="#1c222b", border="#343e4c", height=210, radius=12)
        selector.pack(fill=tk.X, pady=(0, 12))
        selector_body = selector.body

        selector_title = tk.Frame(selector_body, bg="#1c222b")
        selector_title.pack(fill=tk.X, padx=10, pady=(3, 5))
        theme.make_label(selector_title, "字库选择", bg="#1c222b", font=_HOME_FONT_BODY_BOLD).pack(side=tk.LEFT)
        self._summary_var = tk.StringVar(value="请从下方选择一个字库")
        theme.make_label(
            selector_title,
            textvariable=self._summary_var,
            bg="#1c222b",
            fg="#aab5c5",
            font=_HOME_FONT_BODY,
        ).pack(side=tk.RIGHT)

        heading = tk.Frame(selector_body, bg="#2c3440", height=29)
        heading.pack(fill=tk.X, padx=(10, 24))
        heading.pack_propagate(False)
        theme.make_label(heading, "字库名称", bg="#2c3440", font=_HOME_FONT_BODY_BOLD, anchor="w", width=18).pack(side=tk.LEFT, padx=(10, 6))
        theme.make_label(heading, "DPI", bg="#2c3440", font=_HOME_FONT_BODY_BOLD, anchor="w", width=10).pack(side=tk.LEFT)
        theme.make_label(heading, "宽（像素 / 毫米）", bg="#2c3440", font=_HOME_FONT_BODY_BOLD, anchor="w", width=19).pack(side=tk.LEFT)
        theme.make_label(heading, "高（像素 / 毫米）", bg="#2c3440", font=_HOME_FONT_BODY_BOLD, anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)
        theme.make_label(heading, "操作", bg="#2c3440", font=_HOME_FONT_BODY_BOLD, width=18).pack(side=tk.RIGHT)

        list_area = tk.Frame(selector_body, bg="#1c222b")
        list_area.pack(fill=tk.BOTH, expand=True, padx=(10, 8), pady=(0, 7))
        self._ziku_list_canvas = tk.Canvas(list_area, bg="#202731", highlightthickness=0, height=122)
        scrollbar = ttk.Scrollbar(list_area, orient=tk.VERTICAL, command=self._ziku_list_canvas.yview)
        self._ziku_list_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._ziku_list_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._ziku_list_body = tk.Frame(self._ziku_list_canvas, bg="#202731")
        self._ziku_list_window = self._ziku_list_canvas.create_window((0, 0), window=self._ziku_list_body, anchor="nw")
        self._ziku_list_body.bind(
            "<Configure>",
            lambda _event: self._ziku_list_canvas.configure(scrollregion=self._ziku_list_canvas.bbox("all")),
        )
        self._ziku_list_canvas.bind(
            "<Configure>",
            lambda event: self._ziku_list_canvas.itemconfigure(self._ziku_list_window, width=event.width),
        )
        self._ziku_list_canvas.bind("<MouseWheel>", self._on_library_mousewheel)
        self._ziku_rows: dict[str, tk.Frame] = {}

        flow_title = tk.Frame(inner, bg=theme.BG_MAIN)
        flow_title.pack(fill=tk.X, pady=(0, 6))
        self._flow_title_var = tk.StringVar(value="制作流程")
        theme.make_label(flow_title, textvariable=self._flow_title_var, font=_HOME_FONT_SECTION).pack(side=tk.LEFT)
        theme.make_label(
            flow_title,
            "点击入口将在当前窗口中进入对应功能",
            fg="#8e99aa",
            font=_HOME_FONT_DETAIL,
        ).pack(side=tk.RIGHT)

        self._stage_frame = tk.Frame(inner, bg=theme.BG_MAIN)
        self._stage_frame.pack(fill=tk.X)
        for column in range(5):
            self._stage_frame.grid_columnconfigure(column, weight=1, uniform="制作流程")

    def _build_tool_card(
        self,
        parent: tk.Widget,
        column: int,
        card_count: int,
        mark: str,
        title: str,
        detail: str,
        mark_bg: str,
        mark_fg: str,
        command: Callable[[], None],
    ) -> None:
        card = _RoundedPanel(
            parent,
            fill="#1c222b",
            border="#343e4c",
            height=82,
            radius=12,
            cursor="hand2",
        )
        card.grid(
            row=0,
            column=column,
            sticky="nsew",
            padx=(0 if column == 0 else 4, 0 if column == card_count - 1 else 4),
        )
        card_body = card.body
        badge = theme.make_label(
            card_body,
            mark,
            bg=mark_bg,
            fg=mark_fg,
            font=(theme.FONT_FAMILY, 14, "bold"),
            width=2,
            pady=8,
            cursor="hand2",
        )
        badge.pack(side=tk.LEFT, padx=(12, 10))
        text = tk.Frame(card_body, bg="#1c222b", cursor="hand2")
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        title_label = theme.make_label(text, title, bg="#1c222b", font=_HOME_FONT_BODY_BOLD, cursor="hand2")
        title_label.pack(anchor="w")
        detail_label = theme.make_label(text, detail, bg="#1c222b", fg="#8e99aa", font=_HOME_FONT_DETAIL, cursor="hand2")
        detail_label.pack(anchor="w", pady=(5, 0))
        arrow = theme.make_label(card_body, "›", bg="#1c222b", fg="#7f8a9c", font=(theme.FONT_FAMILY, 15), cursor="hand2")
        arrow.pack(side=tk.RIGHT, padx=(4, 6))
        for widget in (card, card_body, badge, text, title_label, detail_label, arrow):
            widget.bind("<Button-1>", lambda _event, callback=command: callback())
            widget.bind("<Enter>", lambda _event, target=card: target.set_border("#536176"))
            widget.bind("<Leave>", lambda _event, target=card: target.set_border("#343e4c"))

    @staticmethod
    def _library_counts(meta: dict[str, Any]) -> tuple[int, int, int, int, int]:
        groups = meta.get("字形组索引", {})
        variants = meta.get("变体详情", {})
        total_chars = len(groups) if isinstance(groups, dict) else 0
        status_counts: dict[str, int] = {}
        if isinstance(variants, dict):
            for item_data in variants.values():
                if isinstance(item_data, dict):
                    status = str(item_data.get("状态", config.STATUS_PENDING_OPTIMIZATION))
                    status_counts[status] = status_counts.get(status, 0) + 1
        total_variants = sum(status_counts.values())
        pending_optimization = status_counts.get(config.STATUS_PENDING_OPTIMIZATION, 0)
        reviewed = status_counts.get(config.STATUS_REVIEWED, 0)
        finished = status_counts.get(config.STATUS_FINISHED, 0)
        optimized = max(0, total_variants - pending_optimization)
        reviewed_total = reviewed + finished
        return total_chars, total_variants, optimized, reviewed_total, finished

    def _refresh_ziku_list(self, preferred_name: str = "") -> None:
        """扫描全部有效字库并更新可滚动列表。"""
        previous = preferred_name or self._selected_name.get()
        self._ziku_items.clear()
        self._ziku_rows.clear()
        for widget in self._ziku_list_body.winfo_children():
            widget.destroy()
        os.makedirs(config.ZIKU_ROOT, exist_ok=True)
        for entry in os.scandir(config.ZIKU_ROOT):
            if not entry.is_dir():
                continue
            json_file = os.path.join(entry.path, f"{entry.name}.json")
            if os.path.isfile(json_file):
                self._ziku_items[entry.name] = (entry.path, json_file)

        names = sorted(self._ziku_items, key=pinyin_natural_key)
        for name in names:
            _path, json_file = self._ziku_items[name]
            data = safe_read_json(json_file, default={})
            metadata = data.get("元数据", {}) if isinstance(data, dict) else {}
            dpi = int(metadata.get("DPI") or metadata.get("分辨率") or 300)
            width_px = int(metadata.get("画布宽") or 250)
            height_px = int(metadata.get("画布高") or 250)
            width_mm = float(metadata.get("成品宽度毫米") or width_px / dpi * 25.4)
            height_mm = float(metadata.get("成品高度毫米") or height_px / dpi * 25.4)
            self._build_library_row(name, dpi, width_px, width_mm, height_px, height_mm)

        if previous in self._ziku_items:
            self._select_library(previous)
        else:
            self._selected_name.set("")
            message = "暂无可选择的字库，请先新建字库" if not names else "请从上方列表选择一个字库"
            self._clear_stage_cards(message)

    def _build_library_row(
        self,
        name: str,
        dpi: int,
        width_px: int,
        width_mm: float,
        height_px: int,
        height_mm: float,
    ) -> None:
        editing = self._editing_name == name
        locked = bool(self._editing_name)
        row = tk.Frame(self._ziku_list_body, bg="#202731", height=38, cursor="arrow" if editing else "hand2")
        row.pack(fill=tk.X)
        row.pack_propagate(False)

        clickable: list[tk.Widget] = [row]
        if editing:
            fields = (
                ("name", 16), ("dpi", 7),
                ("width_px", 7), ("width_mm", 7),
                ("height_px", 7), ("height_mm", 7),
            )
            for index, (key, width) in enumerate(fields):
                entry = theme.make_entry(row, textvariable=self._edit_vars[key], width=width)
                entry.configure(font=_HOME_FONT_BODY)
                entry.pack(side=tk.LEFT, padx=((10 if index == 0 else 3), 3), pady=4)
                entry.bind("<MouseWheel>", self._on_library_mousewheel)
                if key in ("width_px", "height_px"):
                    theme.make_label(row, "/", bg="#202731", fg="#9da9ba").pack(side=tk.LEFT)
        else:
            values = (
                (name, 18, _HOME_FONT_BODY_BOLD, "#e5eaf2"),
                (f"{dpi} DPI", 10, _HOME_FONT_BODY, "#9da9ba"),
                (f"{width_px} px / {width_mm:.2f} mm", 19, _HOME_FONT_BODY, "#9da9ba"),
                (f"{height_px} px / {height_mm:.2f} mm", 19, _HOME_FONT_BODY, "#9da9ba"),
            )
            for index, (text, width, font, fg) in enumerate(values):
                label = theme.make_label(
                    row, text, bg="#202731", fg=fg, font=font,
                    anchor="w", width=width, cursor="hand2",
                )
                label.pack(side=tk.LEFT, padx=((10 if index == 0 else 0), 3))
                clickable.append(label)

        delete_button = theme.make_button(
            row, "删除", danger=True, command=lambda library_name=name: self._delete_library(library_name),
        )
        delete_button.configure(font=_HOME_FONT_BODY, padx=8, pady=2, state=tk.DISABLED if locked else tk.NORMAL)
        delete_button.pack(side=tk.RIGHT, padx=(3, 8), pady=3)
        parameter_button = theme.make_button(
            row, "参数保存" if editing else "参数修改",
            command=self._save_parameters if editing else lambda library_name=name: self._begin_parameter_edit(library_name),
        )
        parameter_button.configure(font=_HOME_FONT_BODY, padx=8, pady=2, state=tk.NORMAL if editing or not locked else tk.DISABLED)
        parameter_button.pack(side=tk.RIGHT, padx=(3, 0), pady=3)

        if not editing:
            for widget in clickable:
                widget.bind("<Button-1>", lambda _event, library_name=name: self._select_library(library_name))
                widget.bind("<MouseWheel>", self._on_library_mousewheel)
        self._ziku_rows[name] = row

    def _select_library(self, name: str) -> None:
        if name not in self._ziku_items or (self._editing_name and name != self._editing_name):
            return
        self._selected_name.set(name)
        for row_name, row in self._ziku_rows.items():
            color = "#33445c" if row_name == name else "#202731"
            row.configure(bg=color)
            for child in row.winfo_children():
                if isinstance(child, tk.Label):
                    child.configure(bg=color)
        self._show_selected_ziku()

    def _on_library_mousewheel(self, event: tk.Event) -> str:
        self._ziku_list_canvas.yview_scroll(int(-event.delta / 120), "units")
        return "break"

    def _begin_parameter_edit(self, name: str) -> None:
        if self._editing_name or name not in self._ziku_items:
            return
        self._select_library(name)
        path, _json_file = self._ziku_items[name]
        service = GlyphService(name, path)
        metadata = service.get_metadata()
        dpi = int(metadata.get("DPI") or metadata.get("分辨率") or 300)
        width_px = int(metadata.get("画布宽") or 250)
        height_px = int(metadata.get("画布高") or 250)
        width_mm = float(metadata.get("成品宽度毫米") or width_px / dpi * 25.4)
        height_mm = float(metadata.get("成品高度毫米") or height_px / dpi * 25.4)
        self._editing_name = name
        self._edit_vars = {
            "name": tk.StringVar(value=name),
            "dpi": tk.StringVar(value=str(dpi)),
            "width_px": tk.StringVar(value=str(width_px)),
            "width_mm": tk.StringVar(value=f"{width_mm:.2f}"),
            "height_px": tk.StringVar(value=str(height_px)),
            "height_mm": tk.StringVar(value=f"{height_mm:.2f}"),
        }
        self._edit_vars["dpi"].trace_add("write", lambda *_: self._pixels_to_millimeters())
        self._edit_vars["width_px"].trace_add("write", lambda *_: self._pixels_to_millimeters())
        self._edit_vars["height_px"].trace_add("write", lambda *_: self._pixels_to_millimeters())
        self._edit_vars["width_mm"].trace_add("write", lambda *_: self._millimeters_to_pixels("width"))
        self._edit_vars["height_mm"].trace_add("write", lambda *_: self._millimeters_to_pixels("height"))
        self._refresh_ziku_list(name)

    def _pixels_to_millimeters(self) -> None:
        if self._conversion_lock:
            return
        try:
            dpi = int(self._edit_vars["dpi"].get())
            width_px = int(self._edit_vars["width_px"].get())
            height_px = int(self._edit_vars["height_px"].get())
            if min(dpi, width_px, height_px) <= 0:
                return
        except (KeyError, ValueError):
            return
        self._conversion_lock = True
        self._edit_vars["width_mm"].set(f"{width_px / dpi * 25.4:.2f}")
        self._edit_vars["height_mm"].set(f"{height_px / dpi * 25.4:.2f}")
        self._conversion_lock = False

    def _millimeters_to_pixels(self, dimension: str) -> None:
        if self._conversion_lock:
            return
        try:
            dpi = int(self._edit_vars["dpi"].get())
            millimeters = float(self._edit_vars[f"{dimension}_mm"].get())
            if dpi <= 0 or millimeters <= 0:
                return
        except (KeyError, ValueError):
            return
        self._conversion_lock = True
        self._edit_vars[f"{dimension}_px"].set(str(round(millimeters / 25.4 * dpi)))
        self._conversion_lock = False

    def _save_parameters(self) -> None:
        old_name = self._editing_name
        item = self._ziku_items.get(old_name)
        if not old_name or not item:
            return
        new_name = self._edit_vars["name"].get().strip()
        if not new_name:
            show_warning(self, "输入有误", "请输入字库名称。")
            return
        if any(char in '<>:"/\\|?*' for char in new_name) or new_name.endswith((" ", ".")):
            show_warning(self, "输入有误", "字库名称包含不能用于目录名的字符，或以空格、句点结尾。")
            return
        try:
            dpi = int(self._edit_vars["dpi"].get())
            width_px = int(self._edit_vars["width_px"].get())
            height_px = int(self._edit_vars["height_px"].get())
            width_mm = float(self._edit_vars["width_mm"].get())
            height_mm = float(self._edit_vars["height_mm"].get())
        except ValueError:
            show_warning(self, "输入有误", "DPI 和像素尺寸必须为整数，物理尺寸必须为数字。")
            return
        if not 72 <= dpi <= 2400:
            show_warning(self, "输入有误", "DPI 应在 72 至 2400 之间。")
            return
        if min(width_px, height_px) < 64 or max(width_px, height_px) > 10000:
            show_warning(self, "输入有误", "成品像素尺寸应在 64 至 10000 像素之间。")
            return
        if min(width_mm, height_mm) <= 0:
            show_warning(self, "输入有误", "成品物理尺寸必须大于 0 毫米。")
            return

        path, _json_file = item
        service = GlyphService(old_name, path)
        metadata = service.get_metadata()
        spec_changed = any((
            int(metadata.get("DPI") or metadata.get("分辨率") or 0) != dpi,
            int(metadata.get("画布宽") or 0) != width_px,
            int(metadata.get("画布高") or 0) != height_px,
            abs(float(metadata.get("成品宽度毫米") or 0) - width_mm) > 0.005,
            abs(float(metadata.get("成品高度毫米") or 0) - height_mm) > 0.005,
        ))
        try:
            if new_name != old_name:
                service.rename_ziku(new_name)
            invalidated = 0
            if spec_changed:
                invalidated = service.update_output_spec(dpi, width_px, height_px, width_mm, height_mm)
        except (OSError, ValueError) as exc:
            show_error(self, "保存失败", str(exc))
            return

        changed = new_name != old_name or spec_changed
        self._editing_name = ""
        self._edit_vars = {}
        self._selected_name.set(new_name)
        self._refresh_ziku_list(new_name)
        if changed:
            suffix = f"\n已有 {invalidated} 个最终成品需要重新生成。" if invalidated else ""
            show_info(self, "参数保存完成", f"字库“{new_name}”的参数已保存。{suffix}")

    def _delete_library(self, name: str) -> None:
        item = self._ziku_items.get(name)
        if not item:
            return
        path, json_file = item
        meta = safe_read_json(json_file, default={})
        _chars, total, optimized, reviewed, finished = self._library_counts(meta)
        message = (
            f"确定删除字库“{name}”吗？\n\n"
            f"原图变体：{total}\n已自动优化：{optimized}\n"
            f"已手工审核：{reviewed}\n最终成品：{finished}\n\n"
            "整个字库目录将移入系统回收站，已导出到其他目录的文件不会受影响。"
        )
        if not ask_yes_no(self, "删除字库", message):
            return

        root = os.path.normcase(os.path.realpath(config.ZIKU_ROOT))
        target = os.path.normcase(os.path.realpath(path))
        if os.path.dirname(target) != root or not os.path.isdir(target):
            show_error(self, "删除失败", "字库目录无效，为保护其他数据，程序已取消删除。")
            return
        try:
            send2trash(path)
        except Exception as exc:
            show_error(self, "删除失败", f"无法将字库“{name}”移入回收站：\n{exc}")
            return

        names = sorted((library_name for library_name in self._ziku_items if library_name != name), key=pinyin_natural_key)
        next_name = ""
        if names:
            removed_index = sorted(self._ziku_items, key=pinyin_natural_key).index(name)
            next_name = names[min(removed_index, len(names) - 1)]
        self._selected_name.set("")
        self._refresh_ziku_list(next_name)
        show_info(self, "删除完成", f"字库“{name}”已移入系统回收站。")

    def _show_selected_ziku(self) -> None:
        name = self._selected_name.get()
        item = self._ziku_items.get(name)
        if not item:
            self._clear_stage_cards("请先从上方选择一个字库")
            return
        path, json_file = item
        meta = safe_read_json(json_file, default={})
        variants = meta.get("变体详情", {})
        total_chars, total_variants, optimized, reviewed_total, finished = self._library_counts(meta)
        pending_optimization = max(0, total_variants - optimized)
        pending_review = 0
        if isinstance(variants, dict):
            pending_review = sum(
                1 for item_data in variants.values()
                if isinstance(item_data, dict)
                and item_data.get("状态") == config.STATUS_PENDING_MANUAL_REVIEW
            )

        self._summary_var.set(f"当前选择：{name}    {total_chars} 字 · {total_variants} 个变体")
        self._flow_title_var.set(f"{name} · 制作流程")
        for widget in self._stage_frame.winfo_children():
            widget.destroy()

        review_status = (
            "等待运行自动优化" if optimized == 0
            else (f"待审核 {pending_review}" if pending_review else "已完成")
        )
        consistency_status = (
            "等待审核通过" if reviewed_total == 0
            else (f"待协调 {reviewed_total - finished}" if finished < reviewed_total else "已完成")
        )
        stages = (
            ("import", "1", "字库添加", "源图总数", str(total_variants), "已完成" if total_variants else "尚未导入", "进入字库添加", "#4a618d"),
            ("optimization", "2", "自动优化", "已完成", f"{optimized} / {total_variants}", "待处理" if pending_optimization else ("已完成" if total_variants else "尚未开始"), "运行自动优化", "#9b7530"),
            ("review", "3", "手工审核", "审核通过", f"{reviewed_total} / {optimized}", review_status, "进入手工审核", "#38735f"),
            ("consistency", "4", "整体协调", "最终成品", f"{finished} / {reviewed_total}", consistency_status, "生成最终成品", "#56637a"),
            ("export", "⇩", "导出最终成品", "当前可导出", f"{finished} 字", "可导出" if finished else "暂无最终成品", "进入导出页面", "#315f9a"),
        )
        for column, data in enumerate(stages):
            self._build_stage_card(column, *data, name, path)

    def _clear_stage_cards(self, message: str) -> None:
        self._summary_var.set(message)
        self._flow_title_var.set("制作流程")
        for widget in self._stage_frame.winfo_children():
            widget.destroy()
        empty = _RoundedPanel(
            self._stage_frame,
            fill="#1c222b",
            border="#343e4c",
            height=68,
            radius=12,
        )
        empty.grid(row=0, column=0, columnspan=5, sticky="ew")
        theme.make_label(empty.body, message, bg="#1c222b", fg="#8e99aa", font=_HOME_FONT_BODY).pack(expand=True)

    def _build_stage_card(
        self,
        column: int,
        stage_key: str,
        mark: str,
        title: str,
        metric_name: str,
        metric_value: str,
        status: str,
        button_text: str,
        accent_color: str,
        name: str,
        path: str,
    ) -> None:
        export_card = title == "导出最终成品"
        card_bg = "#1d2939" if export_card else "#1c222b"
        border = "#405c80" if export_card else "#343e4c"
        card = _RoundedPanel(
            self._stage_frame,
            fill=card_bg,
            border=border,
            height=210,
            radius=12,
        )
        card.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 4, 0 if column == 4 else 4))
        card_body = card.body
        stripe_color = "#4d8be5" if export_card else accent_color
        stripe = tk.Frame(card_body, bg=stripe_color, height=4)
        stripe.pack(fill=tk.X)
        stripe.pack_propagate(False)

        heading = tk.Frame(card_body, bg=card_bg)
        heading.pack(fill=tk.X, padx=8, pady=(7, 5))
        theme.make_label(
            heading,
            mark,
            bg=accent_color,
            fg="#d3e4ff",
            font=_HOME_FONT_BODY_BOLD,
            width=2,
            pady=4,
        ).pack(side=tk.LEFT, padx=(0, 8))
        title_box = tk.Frame(heading, bg=card_bg)
        title_box.pack(side=tk.LEFT)
        theme.make_label(title_box, title, bg=card_bg, font=_HOME_FONT_BODY_BOLD).pack(anchor="w")
        theme.make_label(title_box, status, bg=card_bg, fg="#91b8ed" if export_card else "#8e99aa", font=_HOME_FONT_DETAIL).pack(anchor="w", pady=(2, 0))

        metric = tk.Frame(card_body, bg=card_bg)
        metric.pack(fill=tk.X, padx=8, pady=(3, 7))
        theme.make_label(metric, metric_name, bg=card_bg, fg="#8e99aa", font=_HOME_FONT_DETAIL).pack(anchor="w")
        theme.make_label(metric, metric_value, bg=card_bg, font=_HOME_FONT_METRIC).pack(anchor="w", pady=(3, 0))

        button = theme.make_button(
            card_body,
            button_text,
            command=lambda key=stage_key, n=name, p=path: self._stage_callbacks[key](n, p),
        )
        button.configure(bg=stripe_color, activebackground=stripe_color, font=_HOME_FONT_BODY)
        button.pack(fill=tk.X, padx=8, pady=(2, 0))

    # ==================== 原有心经水印绘制 ====================

    def _redraw_watermark(self) -> None:
        """手工触发一次水印重绘（确保在窗口显示后执行）。"""
        w = self.winfo_width()
        h = self.winfo_height()
        if w > 50 and h > 50:
            self._draw_watermark(w, h)

    def _on_resize(self, event: tk.Event) -> None:
        """窗口尺寸变化后重绘水印。"""
        if event.widget is not self:
            return
        if event.width < 100 or event.height < 100:
            return
        self._draw_watermark(event.width, event.height)

    def _draw_watermark(self, w: int, h: int) -> None:
        """按原版规则绘制布满画布的竖排心经水印。"""
        import tkinter.font as tkfont

        self._bg_canvas.delete("心经水印")
        title = config.WELCOME_BG_TITLE
        body = config.WELCOME_BG_BODY
        title_chars = len(title)
        body_chars = len(body)

        margin = 0.96
        best_size = config.WELCOME_MIN_FONT_SIZE
        low = config.WELCOME_MIN_FONT_SIZE
        high = min(w, h)
        while low <= high:
            size = (low + high) // 2
            if size * title_chars > h:
                high = size - 1
                continue
            chars_per_column = h // size
            if chars_per_column <= 0:
                high = size - 1
                continue
            body_columns = (body_chars + chars_per_column - 1) // chars_per_column
            total_columns = 1 + body_columns
            gap = max(2, size // 5)
            needed_width = total_columns * size + (total_columns - 1) * gap
            if needed_width <= int(w * margin):
                best_size = size
                low = size + 1
            else:
                high = size - 1

        chars_per_column = h // best_size
        if chars_per_column <= 0:
            return
        body_columns = (body_chars + chars_per_column - 1) // chars_per_column
        total_columns = 1 + body_columns
        gap = max(2, best_size // 5)
        used_width = total_columns * best_size + (total_columns - 1) * gap
        offset_x = (w - used_width) // 2

        cache_key = (best_size,)
        if cache_key not in self._watermark_fonts:
            font_obj = None
            for family in config.WELCOME_FONT_FAMILIES:
                try:
                    candidate = tkfont.Font(family=family, size=-best_size)
                    if candidate.measure("觀") > 0:
                        font_obj = candidate
                        break
                except tk.TclError:
                    continue
            self._watermark_fonts[cache_key] = font_obj or tkfont.Font(size=-best_size)
        font_obj = self._watermark_fonts[cache_key]

        color = "#3a3a3a"
        column_right = w - offset_x - gap
        center_x = column_right - best_size // 2
        self._bg_canvas.create_text(
            center_x,
            h // 2,
            text="\n".join(title),
            fill=color,
            font=font_obj,
            anchor="center",
            tags=("心经水印", "心经题目"),
        )
        column_right -= best_size + gap

        char_index = 0
        for _ in range(body_columns):
            center_x = column_right - best_size // 2
            segment = body[char_index:char_index + chars_per_column]
            self._bg_canvas.create_text(
                center_x,
                0,
                text="\n".join(segment),
                fill=color,
                font=font_obj,
                anchor="n",
                tags=("心经水印", "心经正文"),
            )
            char_index += len(segment)
            column_right -= best_size + gap

        self._bg_canvas.tag_lower("心经水印")
