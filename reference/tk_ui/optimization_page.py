from __future__ import annotations

import concurrent.futures
import os
import queue
import time
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageTk
from pypinyin import lazy_pinyin

from services.glyph_service import GlyphService
from services.optimization_service import OptimizationService
from data.log_manager import write_log
from ui import theme
from ui.widgets.custom_dialog import ask_string, ask_yes_no, show_error, show_info, show_warning
from utils.file_utils import validate_final_char


class OptimizationPage(tk.Frame):
    """按字形逐个生成、比较并保存自动优化结果。"""

    MAX_ROUNDS = 5
    PREVIEW_SIZE = (330, 285)
    CARD_SIZE = (145, 112)

    def __init__(self, master: tk.Widget, glyph_service: GlyphService, on_close: Callable[[], None]) -> None:
        super().__init__(master, bg=theme.BG_MAIN)
        self.glyph_service = glyph_service
        self.service = OptimizationService(glyph_service)
        self._on_close = on_close
        self.items = self.service.list_items()
        self.visible_items: list[dict[str, Any]] = []
        self.current_index = -1
        self.current_item: dict[str, Any] | None = None
        self.candidates: list[dict[str, Any]] = []
        self.selected_index = -1
        self.round_number = 1
        self._photo_refs: list[ImageTk.PhotoImage] = []
        self._preview_mode = "透明底"
        self._candidate_cache: dict[str, list[dict[str, Any]]] = {}
        self._candidate_results: queue.Queue[tuple[int, str, list[dict[str, Any]] | None, Exception | None]] = queue.Queue()
        self._candidate_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="自动优化")
        self._candidate_request_id = 0
        self._latest_request_by_key: dict[str, int] = {}
        self._pending_request_by_key: dict[str, int] = {}
        self._pending_candidate_tasks = 0
        self._candidate_polling = False
        self._loading_candidates = False
        self._checkerboard_cache: dict[tuple[int, int], Image.Image] = {}

        self.search_var = tk.StringVar()
        self._search_query = ""
        self.status_var = tk.StringVar(value="全部")
        self.sort_var = tk.StringVar(value="拼音顺序")
        self.progress_var = tk.StringVar()
        self.current_var = tk.StringVar(value="请选择字形")
        self.file_var = tk.StringVar()
        self.candidate_title_var = tk.StringVar(value="第 1 轮候选 · 共 0 张")
        self.scheme_title_var = tk.StringVar(value="尚未选择候选")
        self.score_var = tk.StringVar(value="综合得分 --")
        self.scheme_var = tk.StringVar(value="请从候选效果中选择一张图片。")
        self.round_var = tk.StringVar(value=f"当前分支：第 1/{self.MAX_ROUNDS} 轮")
        self.history_var = tk.StringVar(value="第 1 轮　基础候选")
        self.footer_var = tk.StringVar(value="当前字形 0 / 0 · 第1轮")
        self.summary_var = tk.StringVar()
        self.status_message_var = tk.StringVar(value="请选择左侧字形开始处理")

        self._configure_tree_style()
        self._build_ui()
        self._refresh_tree(select_first=True)

    def _configure_tree_style(self) -> None:
        style = ttk.Style(self)
        style.configure(
            "Optimization.Treeview",
            background=theme.BG_PANEL,
            fieldbackground=theme.BG_PANEL,
            foreground=theme.FG_PRIMARY,
            rowheight=30,
            borderwidth=0,
            font=theme.FONT_SMALL,
        )
        style.map(
            "Optimization.Treeview",
            background=[("selected", theme.BG_ACTIVE)],
            foreground=[("selected", "#ffffff")],
        )

    def _build_ui(self) -> None:
        self._build_header()
        body = tk.Frame(self, bg=theme.BG_MAIN)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 8))

        left = tk.Frame(body, width=285, bg=theme.BG_PANEL, highlightthickness=1, highlightbackground=theme.BORDER)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)
        self._build_glyph_list(left)

        right = tk.Frame(body, width=300, bg=theme.BG_PANEL, highlightthickness=1, highlightbackground=theme.BORDER)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right.pack_propagate(False)
        self._build_scheme_panel(right)

        center = tk.Frame(body, bg=theme.BG_MAIN)
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        self._build_workspace(center)
        self._build_footer()

    def _build_header(self) -> None:
        header = tk.Frame(self, bg=theme.BG_PANEL, height=78, highlightthickness=1, highlightbackground=theme.BORDER)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        title_box = tk.Frame(header, bg=theme.BG_PANEL)
        title_box.pack(side=tk.LEFT, padx=16, pady=(8, 10))
        theme.make_label(title_box, "自动优化", bg=theme.BG_PANEL, font=theme.FONT_TITLE).pack(anchor="w")
        library_name = self.glyph_service.ziku_name
        meta = self.glyph_service.get_metadata()
        dpi = meta.get("DPI", meta.get("分辨率", "--"))
        width = meta.get("画布宽", "--")
        height = meta.get("画布高", "--")
        size = f"{width}×{height}像素"
        theme.make_label(
            title_box,
            f"当前字库：{library_name}　{dpi} DPI · {size}",
            bg=theme.BG_PANEL,
            fg=theme.FG_SECONDARY,
            font=theme.FONT_SMALL,
        ).pack(anchor="w")

        theme.make_button(header, "返回首页", command=self._request_close).pack(side=tk.RIGHT, padx=14, pady=14)
        theme.make_label(header, "", textvariable=self.progress_var, bg=theme.BG_PANEL, fg=theme.FG_SECONDARY).pack(side=tk.RIGHT, padx=18)

    def _build_glyph_list(self, parent: tk.Frame) -> None:
        top = tk.Frame(parent, bg=theme.BG_PANEL)
        top.pack(fill=tk.X, padx=12, pady=(12, 8))
        theme.make_label(top, "字形列表", bg=theme.BG_PANEL, font=theme.FONT_BOLD).pack(side=tk.LEFT)
        self.list_count_label = theme.make_label(top, "显示 / 总数：0 / 0", bg=theme.BG_PANEL, fg=theme.FG_SECONDARY, font=theme.FONT_SMALL)
        self.list_count_label.pack(side=tk.RIGHT)

        search_box = tk.Frame(parent, bg=theme.BG_PANEL)
        search_box.pack(fill=tk.X, padx=12, pady=(0, 8))
        search = theme.make_entry(search_box, textvariable=self.search_var)
        search.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4)
        search.bind("<Return>", lambda _event: self._run_search())
        theme.make_button(search_box, "🔍", command=self._run_search, width=3).pack(side=tk.LEFT, padx=(4, 0), ipady=1)

        filters = tk.Frame(parent, bg=theme.BG_PANEL)
        filters.pack(fill=tk.X, padx=12, pady=(0, 8))
        status_menu = tk.OptionMenu(
            filters,
            self.status_var,
            "全部",
            "待优化",
            "已优化",
            command=lambda _value: self._refresh_tree(),
        )
        self._style_option_menu(status_menu)
        status_menu.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))

        order_menu = tk.OptionMenu(
            filters,
            self.sort_var,
            "拼音顺序",
            "导入顺序",
            "低分优先",
            command=lambda _value: self._refresh_tree(),
        )
        self._style_option_menu(order_menu)
        order_menu.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))

        tree_box = tk.Frame(parent, bg=theme.BG_PANEL)
        tree_box.pack(fill=tk.BOTH, expand=True, padx=(8, 4))
        self.tree = ttk.Treeview(tree_box, show="tree", selectmode="browse", style="Optimization.Treeview")
        scrollbar = ttk.Scrollbar(tree_box, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.tag_configure("group", foreground=theme.FG_PRIMARY, font=theme.FONT_BOLD)
        self.tree.tag_configure("pending", foreground=theme.FG_SECONDARY)
        self.tree.tag_configure("completed", foreground=theme.COLOR_FINALIZED)
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        footer = tk.Frame(parent, bg=theme.BG_CANVAS)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        theme.make_label(
            footer,
            "",
            textvariable=self.summary_var,
            bg=theme.BG_CANVAS,
            fg=theme.FG_SECONDARY,
            font=theme.FONT_SMALL,
        ).pack(padx=10, pady=9)

    @staticmethod
    def _style_option_menu(menu: tk.OptionMenu) -> None:
        menu.configure(
            bg=theme.BG_INPUT,
            fg=theme.FG_PRIMARY,
            activebackground=theme.BG_HOVER,
            activeforeground=theme.FG_PRIMARY,
            bd=0,
            highlightthickness=0,
            font=theme.FONT_SMALL,
        )
        menu["menu"].configure(
            bg=theme.BG_INPUT,
            fg=theme.FG_PRIMARY,
            activebackground=theme.BG_ACTIVE,
            activeforeground=theme.FG_PRIMARY,
            font=theme.FONT_SMALL,
        )

    def _build_workspace(self, parent: tk.Frame) -> None:
        heading = tk.Frame(parent, bg=theme.BG_MAIN)
        heading.pack(fill=tk.X, pady=(0, 6))
        theme.make_label(heading, "", textvariable=self.current_var, bg=theme.BG_MAIN, font=theme.FONT_BOLD).pack(side=tk.LEFT)
        self.change_char_button = theme.make_button(heading, "修改字符", command=self._change_current_char, width=8)
        self.change_char_button.pack(side=tk.LEFT, padx=(6, 0))
        self.change_char_button.configure(state=tk.DISABLED)
        theme.make_label(heading, "", textvariable=self.file_var, bg=theme.BG_MAIN, fg=theme.FG_SECONDARY, font=theme.FONT_SMALL).pack(side=tk.RIGHT)

        previews = tk.Frame(parent, bg=theme.BG_MAIN)
        previews.pack(fill=tk.X)
        original_panel, self.original_preview = self._build_preview(previews, "原始图片")
        original_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        tools = tk.Frame(previews, bg=theme.BG_MAIN)
        tools.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=(30, 8))
        for text in ["适合窗口", "1:1", "白底", "透明底"]:
            theme.make_button(tools, text, command=lambda value=text: self._set_preview_mode(value)).pack(fill=tk.X, pady=(0, 6))
        hold = theme.make_button(tools, "按住查看原图")
        hold.pack(fill=tk.X)
        hold.bind("<ButtonPress-1>", lambda _: self._show_original_in_result())
        hold.bind("<ButtonRelease-1>", lambda _: self._render_selected_preview())

        selected_panel, self.selected_preview = self._build_preview(previews, "选中效果")
        selected_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        candidate_header = tk.Frame(parent, bg=theme.BG_MAIN)
        candidate_header.pack(fill=tk.X, pady=(3, 5))
        theme.make_label(candidate_header, "", textvariable=self.candidate_title_var, bg=theme.BG_MAIN, font=theme.FONT_BOLD).pack(side=tk.LEFT)
        theme.make_label(candidate_header, "按得分排序", bg=theme.BG_MAIN, fg=theme.FG_SECONDARY, font=theme.FONT_SMALL).pack(side=tk.RIGHT)

        self.candidate_grid = tk.Frame(parent, bg=theme.BG_MAIN)
        self.candidate_grid.pack(fill=tk.BOTH, expand=True)
        for column in range(4):
            self.candidate_grid.grid_columnconfigure(column, weight=1, uniform="candidate")
        for row in range(2):
            self.candidate_grid.grid_rowconfigure(row, weight=1, uniform="candidate")

        theme.make_label(parent, "", textvariable=self.status_message_var, bg=theme.BG_MAIN, fg=theme.FG_SECONDARY, font=theme.FONT_SMALL).pack(anchor="w", pady=(5, 0))

    def _build_preview(self, parent: tk.Frame, title: str) -> tuple[tk.Frame, tk.Label]:
        panel = tk.Frame(parent, bg=theme.BG_PANEL, highlightthickness=1, highlightbackground=theme.BORDER)
        theme.make_label(panel, title, bg=theme.BG_PANEL, font=theme.FONT_BOLD).pack(anchor="w", padx=10, pady=(7, 3))
        label = tk.Label(panel, text="暂无图片", bg="#eef0f3", fg="#687080", relief=tk.FLAT)
        label.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        return panel, label

    def _build_scheme_panel(self, parent: tk.Frame) -> None:
        theme.make_label(parent, "选中方案", bg=theme.BG_PANEL, font=theme.FONT_TITLE).pack(anchor="w", padx=16, pady=(15, 4))
        theme.make_label(parent, "", textvariable=self.scheme_title_var, bg=theme.BG_PANEL, font=theme.FONT_BOLD).pack(anchor="w", padx=16, pady=(8, 2))
        theme.make_label(parent, "", textvariable=self.score_var, bg=theme.BG_PANEL, fg=theme.FG_ACCENT, font=theme.FONT_TITLE).pack(anchor="w", padx=16)
        tk.Frame(parent, bg=theme.BORDER, height=1).pack(fill=tk.X, padx=16, pady=12)
        theme.make_label(parent, "算法组合", bg=theme.BG_PANEL, font=theme.FONT_BOLD).pack(anchor="w", padx=16)
        theme.make_label(
            parent,
            "",
            textvariable=self.scheme_var,
            bg=theme.BG_PANEL,
            fg=theme.FG_SECONDARY,
            justify=tk.LEFT,
            wraplength=260,
        ).pack(anchor="w", padx=16, pady=(7, 16))

        self.explore_button = theme.make_button(parent, "围绕选中结果继续探索", accent=True, command=self._explore_selected)
        self.explore_button.pack(fill=tk.X, padx=16, pady=(0, 8))
        self.restart_button = theme.make_button(parent, "更换基础处理路线", command=self._restart_candidates)
        self.restart_button.pack(fill=tk.X, padx=16)

        tk.Frame(parent, bg=theme.BORDER, height=1).pack(fill=tk.X, padx=16, pady=16)
        theme.make_label(parent, "探索进度", bg=theme.BG_PANEL, font=theme.FONT_BOLD).pack(anchor="w", padx=16)
        theme.make_label(parent, "", textvariable=self.round_var, bg=theme.BG_PANEL, fg=theme.FG_SECONDARY).pack(anchor="w", padx=16, pady=(7, 2))
        self.round_progress = ttk.Progressbar(parent, maximum=self.MAX_ROUNDS, value=1)
        self.round_progress.pack(fill=tk.X, padx=16, pady=(0, 6))
        theme.make_label(
            parent,
            "每个分支最多5轮，结果趋同时提前结束。",
            bg=theme.BG_PANEL,
            fg=theme.FG_MUTED,
            font=theme.FONT_SMALL,
            wraplength=260,
            justify=tk.LEFT,
        ).pack(anchor="w", padx=16)
        theme.make_label(parent, "探索记录", bg=theme.BG_PANEL, font=theme.FONT_BOLD).pack(anchor="w", padx=16, pady=(18, 5))
        theme.make_label(
            parent,
            "",
            textvariable=self.history_var,
            bg=theme.BG_CANVAS,
            fg=theme.FG_SECONDARY,
            justify=tk.LEFT,
            anchor="nw",
            wraplength=250,
        ).pack(fill=tk.X, padx=16, ipady=8)

    def _build_footer(self) -> None:
        footer = tk.Frame(self, bg=theme.BG_PANEL, height=60, highlightthickness=1, highlightbackground=theme.BORDER)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        footer.pack_propagate(False)
        controls = tk.Frame(footer, bg=theme.BG_PANEL)
        controls.pack(side=tk.LEFT, padx=12, pady=10)
        theme.make_button(controls, "上一字形", command=lambda: self._move_current(-1)).pack(side=tk.LEFT, padx=(0, 6))
        theme.make_button(controls, "下一字形", command=lambda: self._move_current(1)).pack(side=tk.LEFT, padx=(0, 6))
        theme.make_button(controls, "跳过此字形", command=lambda: self._move_current(1)).pack(side=tk.LEFT)
        theme.make_label(footer, "", textvariable=self.footer_var, bg=theme.BG_PANEL, fg=theme.FG_SECONDARY).pack(side=tk.LEFT, padx=24)

        save_box = tk.Frame(footer, bg=theme.BG_PANEL)
        save_box.pack(side=tk.RIGHT, padx=12, pady=8)
        theme.make_label(
            save_box,
            "保存为“自动优化稿”，下一步提交手工审核",
            bg=theme.BG_PANEL,
            fg=theme.FG_MUTED,
            font=theme.FONT_SMALL,
        ).pack(side=tk.LEFT, padx=(0, 10))
        self.save_button = theme.make_button(save_box, "采用选中结果并保存", accent=True, command=self._save_selected)
        self.save_button.pack(side=tk.LEFT)

    def destroy(self) -> None:
        self._candidate_request_id += 1
        self._candidate_executor.shutdown(wait=False, cancel_futures=True)
        super().destroy()

    def _run_search(self) -> None:
        self._search_query = self.search_var.get().strip().lower()
        self._refresh_tree(select_first=True)

    def _refresh_tree(self, select_first: bool = False) -> None:
        selected_key = self.current_item.get("键") if self.current_item else None
        search = self._search_query
        status_filter = self.status_var.get()
        filtered = []
        for item in self.items:
            state = str(item.get("显示状态", "待优化"))
            if status_filter == "待优化" and state != "待优化":
                continue
            if status_filter == "已优化" and state != "已优化":
                continue
            haystack = f"{item.get('归属字', '')} {item.get('原始文件名', '')} 字形{item.get('变体序号', '')}".lower()
            if search and search not in haystack:
                continue
            filtered.append(item)

        ordering = self.sort_var.get()
        if ordering == "拼音顺序":
            filtered.sort(
                key=lambda value: (
                    tuple(lazy_pinyin(str(value.get("归属字", "")))),
                    str(value.get("归属字", "")),
                    value.get("变体序号", 0),
                )
            )
        elif ordering == "低分优先":
            filtered.sort(key=lambda value: (value.get("得分") is None, float(value.get("得分") or 0)))
        else:
            filtered.sort(key=lambda value: (value.get("字符顺序", 0), value.get("变体序号", 0)))
        self.visible_items = filtered

        self.tree.delete(*self.tree.get_children())
        groups: dict[str, list[dict[str, Any]]] = {}
        for item in filtered:
            groups.setdefault(str(item.get("归属字", "?")), []).append(item)
        for char, variants in groups.items():
            pending = sum(1 for value in variants if value.get("显示状态") == "待优化")
            summary = f"　{pending}待优化" if pending else "　已优化"
            group_id = self.tree.insert("", tk.END, text=f"{char}（{len(variants)}个字形）{summary}", open=True, tags=("group",))
            for item in variants:
                state = str(item.get("显示状态", "待优化"))
                score = item.get("得分")
                suffix = f"{float(score):.1f}分 已优化" if score is not None and state != "待优化" else state
                text = f"字形{item.get('变体序号', 1)}　{item.get('原始文件名', '')}　{suffix}"
                tag = "completed" if state != "待优化" else "pending"
                self.tree.insert(group_id, tk.END, iid=str(item["键"]), text=text, tags=(tag,))

        pending = sum(1 for item in self.items if item.get("显示状态") == "待优化")
        completed = len(self.items) - pending
        self.list_count_label.configure(text=f"显示 / 总数：{len(filtered)} / {len(self.items)}")
        self.summary_var.set(f"待优化 {pending}　已优化 {completed}")
        percent = round(completed * 100 / len(self.items)) if self.items else 0
        self.progress_var.set(f"待优化 {pending}　已优化 {completed}　完成度 {percent}%")

        target = selected_key if selected_key in {item.get("键") for item in filtered} else None
        if target:
            self.tree.selection_set(str(target))
            self.tree.see(str(target))
        elif select_first and filtered:
            self.tree.selection_set(str(filtered[0]["键"]))
            self.tree.see(str(filtered[0]["键"]))
            self._select_item(filtered[0])

    def _on_tree_select(self, _: tk.Event) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        key = selection[0]
        match = next((item for item in self.items if str(item.get("键")) == key), None)
        if match is not None and match is not self.current_item:
            self._select_item(match)

    def _change_current_char(self) -> None:
        if not self.current_item:
            return
        old_char = str(self.current_item.get("归属字", ""))
        new_char = ask_string(
            self,
            "修改字符",
            "修改当前字形的归属字符。相关阶段文件将同步更名。",
            initial_value=old_char,
            input_prompt="新字符",
        )
        if new_char is None:
            return
        new_char = new_char.strip()
        valid, message = validate_final_char(new_char)
        if not valid:
            show_warning(self, "字符不合法", message)
            return
        if new_char == old_char:
            return
        target_exists = any(
            item.get("归属字") == new_char and item.get("键") != self.current_item.get("键")
            for item in self.items
        )
        if target_exists and not ask_yes_no(
            self,
            "合并字符",
            f"字符“{new_char}”已存在。是否将当前字形作为该字符的新变体？",
        ):
            return
        variant_id = str(self.current_item.get("键", ""))
        self.change_char_button.configure(state=tk.DISABLED)
        try:
            self.service.change_variant_char(variant_id, new_char)
        except Exception as exc:
            show_error(self, "修改失败", str(exc))
            self.change_char_button.configure(state=tk.NORMAL)
            return
        self._candidate_cache.pop(variant_id, None)
        self.items = self.service.list_items()
        updated_item = next((item for item in self.items if item.get("键") == variant_id), None)
        if updated_item is None:
            show_error(self, "修改失败", "修改后未找到当前字形记录。")
            return
        self.current_item = updated_item
        self._refresh_tree()
        self.tree.selection_set(variant_id)
        self.tree.see(variant_id)
        self._select_item(updated_item)
        self.status_message_var.set(f"字符已由“{old_char}”修改为“{new_char}”。")

    def _select_item(self, item: dict[str, Any]) -> None:
        switch_started = time.perf_counter()
        self.current_item = item
        self.current_index = self.visible_items.index(item) if item in self.visible_items else -1
        self.round_number = 1
        self.current_var.set(f"当前字形：{item.get('归属字', '')} · 字形{item.get('变体序号', 1)}")
        self.file_var.set(f"原始文件：{item.get('原始文件名', '')}")
        self.change_char_button.configure(state=tk.NORMAL)
        self.original_preview.configure(image="", text="正在载入")
        self.after_idle(
            lambda selected_item=item: self._render_image_path(
                selected_item.get("原始路径", ""), self.original_preview, self.PREVIEW_SIZE, transparent=False
            )
            if self.current_item is selected_item
            else None
        )
        self._load_candidates()
        write_log(f"自动优化界面切换字形｜字形={item.get('归属字', '')}｜准备耗时={time.perf_counter() - switch_started:.4f}秒")

    def _load_candidates(self, force: bool = False) -> None:

        if not self.current_item:
            return
        item = self.current_item
        key = str(item.get("键", ""))
        if not force and key in self._candidate_cache:
            self._candidate_request_id += 1
            self._loading_candidates = False
            self.explore_button.configure(state=tk.NORMAL)
            self.restart_button.configure(state=tk.NORMAL)
            self.save_button.configure(state=tk.NORMAL)
            self.candidates = self._candidate_cache[key]
            self.round_number = 1
            self.history_var.set("第 1 轮　基础候选")
            self._render_candidates()
            self.status_message_var.set("已载入缓存候选；所有处理始终基于原始文件执行。")
            return
        if not force and key in self._pending_request_by_key:
            self._candidate_request_id = self._pending_request_by_key[key]
            self._loading_candidates = True
            self.candidates = []
            self._render_candidates()
            self.candidate_title_var.set("正在生成候选效果……")
            self.selected_preview.configure(image="", text="正在处理")
            self.status_message_var.set("正在等待该字形的候选结果。")
            self.explore_button.configure(state=tk.DISABLED)
            self.restart_button.configure(state=tk.DISABLED)
            self.save_button.configure(state=tk.DISABLED)
            return

        self._candidate_request_id += 1
        request_id = self._candidate_request_id
        self._loading_candidates = True
        self.candidates = []
        self._render_candidates()
        self.candidate_title_var.set("正在生成候选效果……")
        self.selected_preview.configure(image="", text="正在处理")
        self.status_message_var.set("正在后台生成候选效果，可继续切换其他字形。")
        self.explore_button.configure(state=tk.DISABLED)
        self.restart_button.configure(state=tk.DISABLED)
        self.save_button.configure(state=tk.DISABLED)
        self._latest_request_by_key[key] = request_id
        self._pending_request_by_key[key] = request_id
        self._pending_candidate_tasks += 1

        submitted_at = time.perf_counter()

        def worker() -> None:
            try:
                candidates = self.service.generate_candidates(item)
                if not candidates:
                    raise RuntimeError("算法未生成有效候选结果。")
                write_log(
                    f"自动优化后台任务结束｜字形={item.get('归属字', '')}｜请求={request_id}｜"
                    f"耗时={time.perf_counter() - submitted_at:.4f}秒"
                )
                self._candidate_results.put((request_id, key, candidates, None))
            except Exception as exc:
                write_log(
                    f"自动优化后台任务失败｜字形={item.get('归属字', '')}｜请求={request_id}｜"
                    f"耗时={time.perf_counter() - submitted_at:.4f}秒｜原因={exc}"
                )
                self._candidate_results.put((request_id, key, None, exc))

        self._candidate_executor.submit(worker)
        if not self._candidate_polling:
            self._candidate_polling = True
            self.after(20, self._poll_candidate_results)

    def _poll_candidate_results(self) -> None:
        current_key = str(self.current_item.get("键", "")) if self.current_item else ""
        current_result: list[dict[str, Any]] | None = None
        failed_results: list[tuple[str, Exception]] = []
        while True:
            try:
                request_id, key, candidates, error = self._candidate_results.get_nowait()
            except queue.Empty:
                break
            self._pending_candidate_tasks = max(0, self._pending_candidate_tasks - 1)
            if self._pending_request_by_key.get(key) == request_id:
                self._pending_request_by_key.pop(key, None)
            is_latest_for_key = self._latest_request_by_key.get(key) == request_id
            if error is not None and is_latest_for_key:
                failed_results.append((key, error))
            elif candidates is not None and is_latest_for_key:
                self._candidate_cache[key] = candidates
                if request_id == self._candidate_request_id and key == current_key:
                    current_result = candidates

        if failed_results:
            self._remove_failed_records(failed_results)
            current_key = str(self.current_item.get("键", "")) if self.current_item else ""

        if current_result is not None and current_key:
            self._loading_candidates = False
            self.restart_button.configure(state=tk.NORMAL)
            self.candidates = current_result
            state = tk.NORMAL if self.candidates else tk.DISABLED
            self.explore_button.configure(state=state)
            self.save_button.configure(state=state)
            self.round_number = 1
            self.history_var.set("第 1 轮　基础候选")
            self._render_candidates()
            self.status_message_var.set("请选择效果最满意的候选；所有处理始终基于原始文件执行。")

        if self._pending_candidate_tasks:
            self.after(30, self._poll_candidate_results)
        else:
            self._candidate_polling = False

    def _remove_failed_records(self, failures: list[tuple[str, Exception]]) -> None:
        """通知候选生成错误，并从字库中删除对应字形记录。"""
        removed: list[str] = []
        messages: list[str] = []
        current_key = str(self.current_item.get("键", "")) if self.current_item else ""
        for key, error in failures:
            item = next((value for value in self.items if str(value.get("键", "")) == key), None)
            if item is None:
                continue
            description = f"{item.get('归属字', '')} · 字形{item.get('变体序号', 1)}"
            try:
                if self.service.remove_failed_variant(key):
                    removed.append(key)
                    messages.append(f"{description}：{error}")
            except Exception as remove_error:
                messages.append(f"{description}：{error}\n删除字库记录失败：{remove_error}")

        if not messages:
            return
        for key in removed:
            self._candidate_cache.pop(key, None)
            self._latest_request_by_key.pop(key, None)
            self._pending_request_by_key.pop(key, None)
        removed_current = current_key in removed
        if removed_current:
            self.current_item = None
            self.current_index = -1
            self.candidates = []
            self._loading_candidates = False
            self._render_candidates()
        self.items = self.service.list_items()
        self._refresh_tree(select_first=removed_current)
        details = "\n\n".join(messages)
        if removed:
            details += "\n\n以上问题字形已从字库记录中删除，原有图片文件予以保留。"
        show_error(self, "自动优化失败", details)
        if removed:
            self.status_message_var.set("问题字形已通知并从字库记录中删除。")
        else:
            self.status_message_var.set("自动优化失败，删除字库记录时发生错误。")

    def _render_candidates(self) -> None:
        render_started = time.perf_counter()
        for child in self.candidate_grid.winfo_children():
            child.destroy()
        self._photo_refs.clear()
        self.selected_index = -1
        self.scheme_title_var.set("尚未选择候选")
        self.score_var.set("综合得分 --")
        self.scheme_var.set("请从候选效果中选择一张图片。")
        self.candidate_title_var.set(f"第 {self.round_number} 轮候选 · 共 {len(self.candidates[:8])} 张")
        self.round_var.set(f"当前分支：第 {self.round_number}/{self.MAX_ROUNDS} 轮")
        self.round_progress.configure(value=self.round_number)
        total = len(self.visible_items)
        position = self.current_index + 1 if self.current_index >= 0 else 0
        self.footer_var.set(f"当前字形 {position} / {total} · 第{self.round_number}轮")

        for index, candidate in enumerate(self.candidates[:8]):
            card = tk.Frame(self.candidate_grid, bg=theme.BG_PANEL, highlightthickness=1, highlightbackground=theme.BORDER)
            card.grid(row=index // 4, column=index % 4, sticky="nsew", padx=4, pady=4)
            card.bind("<Button-1>", lambda _, value=index: self._select_candidate(value))
            image = self._make_photo(candidate["图像"], self.CARD_SIZE, transparent=True)
            image_label = tk.Label(card, image=image, bg="#eef0f3", cursor="hand2")
            image_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=(5, 2))
            image_label.bind("<Button-1>", lambda _, value=index: self._select_candidate(value))
            score = float(candidate.get("得分", 0.0))
            name = str(candidate.get("方案名", f"候选{index + 1}"))
            caption = theme.make_label(
                card,
                f"候选{index + 1}　{score:.1f}分\n{name}",
                bg=theme.BG_PANEL,
                fg=theme.FG_SECONDARY,
                font=theme.FONT_SMALL,
                justify=tk.LEFT,
            )
            caption.pack(anchor="w", padx=6, pady=(0, 5))
            caption.bind("<Button-1>", lambda _, value=index: self._select_candidate(value))
            card._candidate_index = index  # type: ignore[attr-defined]

        if self.candidates:
            self._select_candidate(0)
        else:
            self.selected_preview.configure(image="", text="没有生成可用候选")
        write_log(f"自动优化界面渲染候选｜候选数={len(self.candidates[:8])}｜耗时={time.perf_counter() - render_started:.4f}秒")

    def _select_candidate(self, index: int) -> None:
        if not 0 <= index < len(self.candidates[:8]):
            return
        self.selected_index = index
        for child in self.candidate_grid.winfo_children():
            if not isinstance(child, tk.Frame):
                continue
            selected = getattr(child, "_candidate_index", -1) == index
            child.configure(highlightbackground=theme.FG_ACCENT if selected else theme.BORDER, highlightthickness=2 if selected else 1)
        candidate = self.candidates[index]
        self.scheme_title_var.set(f"候选{index + 1}　{candidate.get('方案名', '')}")
        self.score_var.set(f"综合得分 {float(candidate.get('得分', 0.0)):.1f}")
        self.scheme_var.set(self._format_scheme(candidate.get("方案", {})))
        self._render_selected_preview()

    @staticmethod
    def _format_scheme(scheme: dict[str, Any]) -> str:
        lines = []
        preprocess = scheme.get("预处理", {})
        enabled = [name for name in ["转灰度", "反相", "墨色归一"] if preprocess.get(name)]
        lines.append(f"预处理：{'、'.join(enabled) if enabled else '无'}")
        for level in ["L1", "L2", "L3", "L4", "L5"]:
            config = scheme.get(level, {})
            method = config.get("方法")
            if method and method != "不处理":
                params = "、".join(f"{key}={value}" for key, value in config.items() if key != "方法")
                lines.append(f"{level}：{method}" + (f"（{params}）" if params else ""))
        return "\n".join(lines) if lines else "基础处理方案"

    def _explore_selected(self) -> None:
        if self.selected_index < 0:
            show_info(self, "继续探索", "请先选择一个候选效果。")
            return
        if self.round_number >= self.MAX_ROUNDS:
            show_info(self, "探索已完成", "当前分支已达到5轮上限。请采用当前结果，或更换基础处理路线。")
            return
        base = self.candidates[self.selected_index]
        self.status_message_var.set("正在围绕选中方案生成下一轮候选……")
        self.update_idletasks()
        try:
            new_candidates = self.service.explore(self.current_item or {}, base, count=8)
        except Exception as exc:
            key = str(self.current_item.get("键", "")) if self.current_item else ""
            if key:
                self._remove_failed_records([(key, exc)])
            else:
                show_error(self, "探索失败", str(exc))
            return
        if not new_candidates:
            show_info(self, "探索已完成", "当前方案附近已基本探索完毕，没有发现新的有效结果。")
            return
        self.round_number += 1
        self.candidates = new_candidates[:8]
        self.history_var.set(self.history_var.get() + f"\n第 {self.round_number} 轮　基于候选{self.selected_index + 1}")
        self._render_candidates()
        self.status_message_var.set("新一轮候选已生成；重复方案和重复图片已自动排除。")

    def _restart_candidates(self) -> None:
        if not self.current_item or self._loading_candidates:
            return
        if not ask_yes_no(self, "更换基础处理路线", "将结束当前探索分支，并从原始图片重新生成基础候选。是否继续？"):
            return
        self._candidate_cache.pop(str(self.current_item.get("键", "")), None)
        self._load_candidates(force=True)

    def _save_selected(self) -> None:
        if not self.current_item or self.selected_index < 0:
            show_info(self, "保存结果", "请先选择一个候选效果。")
            return
        try:
            self.service.save_selection(self.current_item, self.candidates[self.selected_index], self.round_number)
        except Exception as exc:
            show_error(self, "保存失败", str(exc))
            return
        key = self.current_item.get("键")
        self.items = self.service.list_items()
        self.current_item = next((item for item in self.items if item.get("键") == key), self.current_item)
        self.status_message_var.set("已保存为“自动优化稿”，该字形已提交手工审核。")
        self._refresh_tree()
        self.after(350, lambda: self._move_current(1, pending_only=True))

    def _move_current(self, step: int, pending_only: bool = False) -> None:
        if not self.visible_items:
            return
        start = self.current_index if self.current_index >= 0 else 0
        indexes = range(start + step, len(self.visible_items), step) if step > 0 else range(start + step, -1, step)
        target = None
        for index in indexes:
            item = self.visible_items[index]
            if not pending_only or item.get("显示状态") == "待优化":
                target = item
                break
        if target is None and pending_only:
            show_info(self, "自动优化", "当前筛选范围内没有更多待优化字形。")
            return
        if target is not None:
            self.tree.selection_set(str(target["键"]))
            self.tree.see(str(target["键"]))
            self._select_item(target)

    def _request_close(self) -> None:
        self._on_close()

    def _set_preview_mode(self, mode: str) -> None:
        self._preview_mode = mode
        self._render_selected_preview()

    def _show_original_in_result(self) -> None:
        if self.current_item:
            self._render_image_path(self.current_item.get("原始路径", ""), self.selected_preview, self.PREVIEW_SIZE, transparent=False)

    def _render_selected_preview(self) -> None:
        if 0 <= self.selected_index < len(self.candidates):
            transparent = self._preview_mode != "白底"
            self._set_label_photo(self.selected_preview, self.candidates[self.selected_index]["图像"], self.PREVIEW_SIZE, transparent)

    def _render_image_path(self, path: str, label: tk.Label, size: tuple[int, int], transparent: bool) -> None:
        started_at = time.perf_counter()
        try:
            with Image.open(path) as source:
                self._set_label_photo(label, source.copy(), size, transparent)
            write_log(f"自动优化界面渲染原图｜文件={os.path.basename(path)}｜耗时={time.perf_counter() - started_at:.4f}秒")
        except Exception:
            label.configure(image="", text="图片无法读取")
            write_log(f"自动优化界面渲染原图失败｜文件={os.path.basename(path)}｜耗时={time.perf_counter() - started_at:.4f}秒")

    def _set_label_photo(self, label: tk.Label, image: Image.Image, size: tuple[int, int], transparent: bool) -> None:
        photo = self._make_photo(image, size, transparent)
        label.configure(image=photo, text="")
        label.image = photo  # type: ignore[attr-defined]

    def _make_photo(self, image: Image.Image, size: tuple[int, int], transparent: bool) -> ImageTk.PhotoImage:
        source = image.convert("RGBA")
        source.thumbnail(size, Image.Resampling.LANCZOS)
        if transparent:
            background = self._checkerboard(size)
        else:
            background = Image.new("RGBA", size, "white")
        x = (size[0] - source.width) // 2
        y = (size[1] - source.height) // 2
        background.alpha_composite(source, (x, y))
        photo = ImageTk.PhotoImage(background.convert("RGB"))
        self._photo_refs.append(photo)
        return photo

    def _checkerboard(self, size: tuple[int, int]) -> Image.Image:
        cached = self._checkerboard_cache.get(size)
        if cached is not None:
            return cached.copy()
        image = Image.new("RGBA", size, "#f0f0f0")
        draw = ImageDraw.Draw(image)
        block = 12
        for y in range(0, size[1], block):
            for x in range(0, size[0], block):
                if (x // block + y // block) % 2:
                    draw.rectangle((x, y, min(x + block - 1, size[0] - 1), min(y + block - 1, size[1] - 1)), fill=(214, 218, 223, 255))
        self._checkerboard_cache[size] = image
        return image.copy()
