# glyph_tree.py — 手工审核字形列表

import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk
from typing import Callable, Optional

import config
from services.glyph_service import GlyphService
from ui import theme
from utils.file_utils import pinyin_natural_key


class GlyphTree(tk.Frame):
    """按汉字分组展示已进入手工审核流程的字形。"""

    SEARCH_PLACEHOLDER = "搜索汉字、文件名或字形序号"
    FILTER_ALL = "全部状态"
    ORDER_PINYIN = "拼音顺序"
    ORDER_ORIGINAL = "原始顺序"

    def __init__(
        self,
        parent: tk.Widget,
        glyph_service: GlyphService,
        on_select: Callable[[str, int], None],
        allowed_statuses: Optional[tuple[str, ...]] = None,
        allow_context_menu: bool = True,
        show_score: bool = True,
        require_intermediate_file: bool = True,
        on_order_change: Optional[Callable[[str], bool]] = None,
        summary_pending_label: str = "待审核",
        summary_completed_label: str = "审核通过",
        summary_pending_statuses: tuple[str, ...] = (config.STATUS_PENDING_MANUAL_REVIEW,),
        summary_completed_statuses: tuple[str, ...] = (
            config.STATUS_REVIEWED,
            config.STATUS_FINISHED,
        ),
    ) -> None:
        super().__init__(parent, bg=theme.BG_PANEL)
        self._glyph = glyph_service
        self._on_select = on_select
        self._allowed_statuses = set(allowed_statuses) if allowed_statuses else None
        self._allow_context_menu = allow_context_menu
        self._show_score = show_score
        self._require_intermediate_file = require_intermediate_file
        self._on_order_change = on_order_change
        self._summary_pending_label = summary_pending_label
        self._summary_completed_label = summary_completed_label
        self._summary_pending_statuses = set(summary_pending_statuses)
        self._summary_completed_statuses = set(summary_completed_statuses)
        self._active_order = self.ORDER_PINYIN
        self._tree: Optional[ttk.Treeview] = None
        self._search_entry: Optional[tk.Entry] = None
        self._count_label: Optional[tk.Label] = None
        self._summary_label: Optional[tk.Label] = None
        self._filter_var = tk.StringVar(value=self.FILTER_ALL)
        self._order_var = tk.StringVar(value=self.ORDER_PINYIN)
        self._node_variants: dict[str, tuple[str, int]] = {}
        self._configure_tree_style()
        self._build()
        self.refresh()

    def _configure_tree_style(self) -> None:
        style = ttk.Style(self)
        style.configure(
            "Review.Treeview",
            background=theme.BG_PANEL,
            fieldbackground=theme.BG_PANEL,
            foreground=theme.FG_PRIMARY,
            rowheight=30,
            borderwidth=0,
            font=theme.FONT_SMALL,
        )
        style.map(
            "Review.Treeview",
            background=[("selected", theme.BG_ACTIVE)],
            foreground=[("selected", "#ffffff")],
        )

    def _build(self) -> None:
        header = tk.Frame(self, bg=theme.BG_PANEL)
        header.pack(fill=tk.X, padx=10, pady=(10, 7))
        theme.make_label(
            header, "字形列表", bg=theme.BG_PANEL, font=theme.FONT_BOLD
        ).pack(side=tk.LEFT)
        self._count_label = theme.make_label(
            header, "显示 / 总数：0 / 0", bg=theme.BG_PANEL, fg=theme.FG_SECONDARY,
            font=theme.FONT_SMALL,
        )
        self._count_label.pack(side=tk.RIGHT)

        search_row = tk.Frame(self, bg=theme.BG_PANEL)
        search_row.pack(fill=tk.X, padx=8, pady=(0, 6))
        self._search_entry = theme.make_entry(
            search_row, placeholder=self.SEARCH_PLACEHOLDER
        )
        self._search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4)
        self._search_entry.bind("<Return>", lambda _event: self.refresh())
        theme.make_button(
            search_row, "🔍", command=self.refresh, width=3
        ).pack(side=tk.LEFT, padx=(5, 0))

        filter_row = tk.Frame(self, bg=theme.BG_PANEL)
        filter_row.pack(fill=tk.X, padx=8, pady=(0, 7))
        status_values = [self.FILTER_ALL]
        if self._allowed_statuses:
            status_values.extend(
                status for status in (
                    config.STATUS_PENDING_MANUAL_REVIEW,
                    config.STATUS_REVIEWED,
                    config.STATUS_FINISHED,
                ) if status in self._allowed_statuses
            )
        status_menu = tk.OptionMenu(
            filter_row, self._filter_var, *status_values,
            command=lambda _value: self.refresh(),
        )
        self._style_option_menu(status_menu)
        status_menu.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))

        order_menu = tk.OptionMenu(
            filter_row, self._order_var, self.ORDER_PINYIN, self.ORDER_ORIGINAL,
            command=lambda value: self._handle_order_change(str(value)),
        )
        self._style_option_menu(order_menu)
        order_menu.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))

        tree_host = tk.Frame(self, bg=theme.BG_PANEL)
        tree_host.pack(fill=tk.BOTH, expand=True, padx=4)
        self._tree = ttk.Treeview(
            tree_host,
            columns=("detail",),
            show=("tree", "headings"),
            selectmode="browse",
            style="Review.Treeview",
        )
        self._tree.heading("#0", text="字形 / 文件", anchor=tk.W)
        self._tree.heading(
            "detail", text="得分 / 状态" if self._show_score else "状态", anchor=tk.E
        )
        self._tree.column("#0", width=155, minwidth=145, stretch=False)
        detail_width = 105 if self._show_score else 75
        detail_minwidth = 95 if self._show_score else 70
        self._tree.column(
            "detail", width=detail_width, minwidth=detail_minwidth,
            stretch=True, anchor=tk.E,
        )
        vertical_scrollbar = ttk.Scrollbar(
            tree_host, orient=tk.VERTICAL, command=self._tree.yview
        )
        horizontal_scrollbar = ttk.Scrollbar(
            tree_host, orient=tk.HORIZONTAL, command=self._tree.xview
        )
        self._tree.configure(
            yscrollcommand=vertical_scrollbar.set,
            xscrollcommand=horizontal_scrollbar.set,
        )
        self._tree.grid(row=0, column=0, sticky="nsew")
        vertical_scrollbar.grid(row=0, column=1, sticky="ns")
        horizontal_scrollbar.grid(row=1, column=0, sticky="ew")
        tree_host.grid_rowconfigure(0, weight=1)
        tree_host.grid_columnconfigure(0, weight=1)
        self._tree.tag_configure(
            "group", foreground=theme.FG_PRIMARY, font=theme.FONT_BOLD
        )
        self._tree.tag_configure(
            config.STATUS_PENDING_MANUAL_REVIEW,
            foreground=config.STATUS_COLORS[config.STATUS_PENDING_MANUAL_REVIEW],
            font=theme.FONT_SMALL,
        )
        self._tree.tag_configure(
            config.STATUS_REVIEWED,
            foreground=config.STATUS_COLORS[config.STATUS_REVIEWED],
            font=theme.FONT_SMALL,
        )
        self._tree.tag_configure(
            config.STATUS_FINISHED,
            foreground=config.STATUS_COLORS[config.STATUS_FINISHED],
            font=theme.FONT_SMALL,
        )
        self._ignore_tree_select = False
        self._tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        self._summary_label = theme.make_label(
            self,
            f"{self._summary_pending_label} 0　{self._summary_completed_label} 0　完成度 0%",
            bg=theme.BG_PANEL, fg=theme.FG_SECONDARY, font=theme.FONT_SMALL,
        )
        self._summary_label.pack(fill=tk.X, padx=10, pady=(6, 9))

    @staticmethod
    def _style_option_menu(menu: tk.OptionMenu) -> None:
        menu.configure(
            bg=theme.BG_INPUT, fg=theme.FG_PRIMARY,
            activebackground=theme.BG_HOVER, activeforeground=theme.FG_PRIMARY,
            bd=0, highlightthickness=0, font=theme.FONT_SMALL,
        )
        menu["menu"].configure(
            bg=theme.BG_INPUT, fg=theme.FG_PRIMARY,
            activebackground=theme.BG_ACTIVE, activeforeground=theme.FG_PRIMARY,
            font=theme.FONT_SMALL,
        )

    def _handle_order_change(self, value: str) -> None:
        if value == self._active_order:
            return
        if self._on_order_change is not None and not self._on_order_change(value):
            self._order_var.set(self._active_order)
            return
        self._active_order = value
        self.refresh()

    def _review_variants(self, char: str) -> list[dict]:
        variants = self._glyph.get_char_variants(char)
        return [
            variant for variant in variants
            if (self._allowed_statuses is None or variant.get("状态") in self._allowed_statuses)
            and (not self._require_intermediate_file or variant.get("中间文件"))
        ]

    def _search_text(self) -> str:
        if not self._search_entry:
            return ""
        value = self._search_entry.get().strip()
        return "" if value == self.SEARCH_PLACEHOLDER else value.lower()

    def refresh(self, _filter_status: Optional[str] = None) -> None:
        if not self._tree:
            return
        self._tree.delete(*self._tree.get_children())
        self._node_variants.clear()

        search_text = self._search_text()
        filter_status = self._filter_var.get()
        groups: list[tuple[str, list[dict]]] = []
        all_total = 0
        pending_total = 0
        reviewed_total = 0

        chars = self._glyph.get_all_chars()
        if self._order_var.get() == self.ORDER_PINYIN:
            chars = sorted(chars, key=pinyin_natural_key)

        for char in chars:
            variants = self._review_variants(char)
            all_total += len(variants)
            pending_total += sum(
                variant.get("状态") in self._summary_pending_statuses
                for variant in variants
            )
            reviewed_total += sum(
                variant.get("状态") in self._summary_completed_statuses
                for variant in variants
            )
            visible: list[dict] = []
            for index, variant in enumerate(variants):
                status = variant.get("状态", "")
                filename = variant.get("原始文件") or variant.get("中间文件", "")
                haystack = f"{char} {filename} 字形{index + 1} {status}".lower()
                if filter_status != self.FILTER_ALL and status != filter_status:
                    continue
                if search_text and search_text not in haystack:
                    continue
                visible.append(variant)
            if visible:
                groups.append((char, visible))

        visible_total = sum(len(variants) for _, variants in groups)
        if self._count_label:
            self._count_label.configure(text=f"显示 / 总数：{visible_total} / {all_total}")
        completion = round(reviewed_total * 100 / all_total) if all_total else 0
        if self._summary_label:
            self._summary_label.configure(
                text=(
                    f"{self._summary_pending_label} {pending_total}　"
                    f"{self._summary_completed_label} {reviewed_total}　"
                    f"完成度 {completion}%"
                )
            )

        for char, visible_variants in groups:
            pending = sum(
                variant.get("状态") == config.STATUS_PENDING_MANUAL_REVIEW
                for variant in visible_variants
            )
            parent_detail = f"{pending}待审核" if pending else "审核通过"
            parent = self._tree.insert(
                "", tk.END, text=f"{char}（{len(visible_variants)}个字形）",
                values=(parent_detail,), open=True, tags=("group",),
            )
            all_review_variants = self._review_variants(char)
            for variant in visible_variants:
                actual_index = all_review_variants.index(variant)
                filename = variant.get("原始文件") or variant.get("中间文件", "")
                status = variant.get("状态", "—")
                score = variant.get("自动优化", {}).get("得分")
                detail = (
                    f"{score}分 · {status}"
                    if self._show_score and score is not None
                    else status
                )
                node = self._tree.insert(
                    parent, tk.END,
                    text=f"字形{actual_index + 1}　{filename}", values=(detail,),
                    tags=(status,),
                )
                self._node_variants[node] = (char, actual_index)

    def recommended_width(self) -> int:
        """依据当前列表的最长显示内容返回合理的左栏初始宽度。"""
        if not self._tree:
            return 250
        regular_font = tkfont.Font(font=theme.FONT_SMALL)
        bold_font = tkfont.Font(font=theme.FONT_BOLD)
        tree_width = regular_font.measure("字形 / 文件") + 28
        detail_width = regular_font.measure(
            "得分 / 状态" if self._show_score else "状态"
        ) + 28
        for parent in self._tree.get_children(""):
            tree_width = max(
                tree_width,
                bold_font.measure(str(self._tree.item(parent, "text"))) + 30,
            )
            values = self._tree.item(parent, "values")
            if values:
                detail_width = max(detail_width, regular_font.measure(str(values[0])) + 24)
            for child in self._tree.get_children(parent):
                tree_width = max(
                    tree_width,
                    regular_font.measure(str(self._tree.item(child, "text"))) + 50,
                )
                values = self._tree.item(child, "values")
                if values:
                    detail_width = max(
                        detail_width, regular_font.measure(str(values[0])) + 24
                    )
        tree_width = min(330, max(145, tree_width))
        detail_width = min(130, max(70, detail_width))
        self._tree.column("#0", width=tree_width)
        self._tree.column("detail", width=detail_width)
        return min(470, max(250, tree_width + detail_width + 25))

    def _on_tree_select(self, _event: tk.Event) -> None:
        if not self._tree or self._ignore_tree_select:
            return
        selection = self._tree.selection()
        if not selection:
            return
        selected = self._node_variants.get(selection[0])
        if selected:
            self._on_select(*selected)

    def next_visible_after(self, char: str, variant_index: int) -> Optional[tuple[str, int]]:
        """按左侧当前筛选和排序后的列表返回下一个字形，到末尾后从头继续。"""
        visible = list(self._node_variants.values())
        if len(visible) <= 1:
            return None
        current = (char, variant_index)
        try:
            start = visible.index(current) + 1
        except ValueError:
            start = 0
        target = visible[start] if start < len(visible) else visible[0]
        return target if target != current else None

    def select_variant(self, char: str, variant_index: int) -> None:
        """选中并显示指定字形。"""
        if not self._tree:
            return
        target = (char, variant_index)
        for node, selected in self._node_variants.items():
            if selected == target:
                self._tree.selection_set(node)
                self._tree.focus(node)
                self._tree.see(node)
                self._on_select(char, variant_index)
                return

    def restore_selection(self, char: str, variant_index: int) -> None:
        """仅恢复树选择，不重新加载字形。"""
        if not self._tree:
            return
        target = (char, variant_index)
        for node, selected in self._node_variants.items():
            if selected == target:
                self._ignore_tree_select = True
                self._tree.selection_set(node)
                self._tree.focus(node)
                self._tree.see(node)
                # TreeviewSelect 可能排入事件队列，空闲时再解除抑制。
                self.after_idle(self._finish_restore_selection)
                return

    def _finish_restore_selection(self) -> None:
        self._ignore_tree_select = False

    def set_filter(self, text: str) -> None:
        if not self._search_entry:
            return
        self._search_entry.delete(0, tk.END)
        self._search_entry.insert(0, text)
        self._search_entry.configure(fg=theme.FG_PRIMARY)
        self.refresh()
