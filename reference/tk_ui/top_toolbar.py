# top_toolbar.py — 顶部工具栏：搜索、筛选、仪表板、菜单、导出

import tkinter as tk
from typing import Callable, Optional

import config
from services.glyph_service import GlyphService
from ui import theme


class TopToolbar(tk.Frame):
    """顶部工具栏（打开字库后显示）。"""

    def __init__(
        self,
        parent: tk.Widget,
        glyph_service: GlyphService,
        on_back: Callable[[], None],
        on_filter: Optional[Callable[[str], None]] = None,
        on_save: Optional[Callable[[], None]] = None,
        on_review: Optional[Callable[[], None]] = None,
        on_finalize: Optional[Callable[[], None]] = None,
        on_export: Optional[Callable[[], None]] = None,
        on_menu: Optional[Callable[[], None]] = None,
        review_mode: bool = False,
        page_title: str = "手工审核",
    ) -> None:
        super().__init__(parent, bg=theme.BG_PANEL)
        self._glyph: GlyphService = glyph_service
        self._on_back: Callable[[], None] = on_back
        self._on_filter = on_filter
        self._on_save = on_save
        self._on_review = on_review
        self._on_finalize = on_finalize
        self._on_export = on_export
        self._on_menu = on_menu
        self._review_mode = review_mode
        self._page_title = page_title
        self._status_filter: str = ""

        self._build()

    def _build(self) -> None:
        theme.make_button(self, "返回首页", command=self._on_back, width=10).pack(
            side=tk.RIGHT, padx=(8, 12), pady=12
        )

        if self._review_mode and self._glyph:
            title_box = tk.Frame(self, bg=theme.BG_PANEL)
            title_box.pack(side=tk.LEFT, padx=(16, 12), pady=(8, 7))
            theme.make_label(
                title_box, self._page_title, bg=theme.BG_PANEL,
                font=theme.FONT_TITLE, fg=theme.FG_PRIMARY,
            ).pack(anchor=tk.W)
            meta = self._glyph.get_metadata()
            dpi = meta.get("DPI", meta.get("分辨率", "--"))
            width = meta.get("画布宽", "--")
            height = meta.get("画布高", "--")
            theme.make_label(
                title_box,
                f"当前字库：{self._glyph.ziku_name}　{dpi} DPI · {width}×{height}像素",
                bg=theme.BG_PANEL, fg=theme.FG_SECONDARY, font=theme.FONT_SMALL,
            ).pack(anchor=tk.W, pady=(2, 0))
            return

        logo = theme.make_label(self, "字库编辑", font=theme.FONT_BOLD, fg=theme.FG_ACCENT)
        logo.pack(side=tk.LEFT, padx=(8, 6))

        # 搜索框
        self._search_var = tk.StringVar()
        self._search_entry = tk.Entry(
            self, textvariable=self._search_var,
            bg=theme.BG_INPUT, fg=theme.FG_PRIMARY,
            insertbackground=theme.FG_PRIMARY, bd=0, font=theme.FONT_NORMAL, width=20,
        )
        self._search_entry.pack(side=tk.LEFT, padx=(12, 2))
        self._search_entry.insert(0, "搜索字或状态...")
        self._search_entry.configure(fg=theme.FG_MUTED)
        self._search_entry.bind("<FocusIn>", self._on_search_focus_in)
        self._search_entry.bind("<FocusOut>", self._on_search_focus_out)
        self._search_entry.bind("<Return>", lambda e: self._do_search())

        theme.make_button(self, "🔍", command=self._do_search).pack(side=tk.LEFT)

        # 筛选下拉
        filter_statuses = (
            config.STATUS_PENDING_MANUAL_REVIEW,
            config.STATUS_REVIEWED,
            config.STATUS_FINISHED,
        ) if self._review_mode else config.ALL_STATUSES
        self._filter_var = tk.StringVar(value="全部")
        filter_cb = tk.OptionMenu(
            self, self._filter_var, "全部", *filter_statuses,
            command=lambda value: self._on_filter_change(str(value)),
        )
        filter_cb.configure(bg=theme.BG_INPUT, fg=theme.FG_PRIMARY, bd=0,
                            activebackground=theme.BG_HOVER, activeforeground=theme.FG_PRIMARY,
                            font=theme.FONT_SMALL)
        filter_cb["menu"].configure(bg=theme.BG_PANEL, fg=theme.FG_PRIMARY, font=theme.FONT_SMALL)
        filter_cb.pack(side=tk.LEFT, padx=4)

        # 仪表板数字行
        self._counts_frame = tk.Frame(self, bg=theme.BG_PANEL)
        self._counts_frame.pack(side=tk.LEFT, padx=(16, 0))
        self._count_labels: dict[str, tk.Label] = {}
        self.refresh_counts()

        # 工作区主要操作（右侧）
        if self._on_menu:
            theme.make_button(self, "更多", command=self._on_menu_click).pack(side=tk.RIGHT, padx=4)
        if self._on_export:
            theme.make_button(self, "导出", command=self._on_export_click).pack(side=tk.RIGHT, padx=4)
        if self._on_review:
            theme.make_button(self, "保存并审核通过", command=self._on_review).pack(side=tk.RIGHT, padx=4)
        if self._on_save:
            theme.make_button(self, "保存修订稿", accent=True, command=self._on_save).pack(side=tk.RIGHT, padx=4)

    def refresh_counts(self) -> None:
        """刷新仪表板数字。"""
        if self._review_mode:
            return
        for w in self._counts_frame.winfo_children():
            w.destroy()
        counts = self._glyph.get_status_counts() if self._glyph else {}
        total = self._glyph.get_total_count() if self._glyph else 0
        if self._review_mode:
            total = sum(
                counts.get(status, 0)
                for status in (
                    config.STATUS_PENDING_MANUAL_REVIEW,
                    config.STATUS_REVIEWED,
                    config.STATUS_FINISHED,
                )
            )

        def make_count_label(text: str, color: str, val: int) -> None:
            frame = tk.Frame(self._counts_frame, bg=theme.BG_PANEL, cursor="hand2")
            frame.pack(side=tk.LEFT, padx=2)
            tk.Label(frame, text=text, bg=theme.BG_PANEL, fg=theme.FG_MUTED, font=theme.FONT_SMALL).pack(side=tk.LEFT)
            lbl = tk.Label(frame, text=str(val), bg=theme.BG_PANEL, fg=color, font=theme.FONT_BOLD)
            lbl.pack(side=tk.LEFT)
            frame.bind("<Button-1>", lambda e, s=text: self._on_count_click(s))
            lbl.bind("<Button-1>", lambda e, s=text: self._on_count_click(s))

        make_count_label("总 ", theme.FG_PRIMARY, total)
        if not self._review_mode:
            make_count_label("优化 ", config.STATUS_COLORS[config.STATUS_PENDING_OPTIMIZATION], counts.get(config.STATUS_PENDING_OPTIMIZATION, 0))
        make_count_label("手审 ", config.STATUS_COLORS[config.STATUS_PENDING_MANUAL_REVIEW], counts.get(config.STATUS_PENDING_MANUAL_REVIEW, 0))
        make_count_label("通过 ", config.STATUS_COLORS[config.STATUS_REVIEWED], counts.get(config.STATUS_REVIEWED, 0))
        make_count_label("成品 ", config.STATUS_COLORS[config.STATUS_FINISHED], counts.get(config.STATUS_FINISHED, 0))

    # ==================== 事件处理 ====================

    def _on_search_focus_in(self, _e: tk.Event) -> None:
        if self._search_entry.get() == "搜索字或状态...":
            self._search_entry.delete(0, tk.END)
            self._search_entry.configure(fg=theme.FG_PRIMARY)

    def _on_search_focus_out(self, _e: tk.Event) -> None:
        if not self._search_entry.get():
            self._search_entry.insert(0, "搜索字或状态...")
            self._search_entry.configure(fg=theme.FG_MUTED)

    def _do_search(self) -> None:
        query = self._search_var.get().strip()
        if query == "搜索字或状态...":
            query = ""
        self._status_filter = query
        if self._on_filter:
            self._on_filter(query)

    def _on_filter_change(self, value: str) -> None:
        self._status_filter = value if value != "全部" else ""
        if self._on_filter:
            self._on_filter(self._status_filter)

    def _on_count_click(self, status: str) -> None:
        """点击仪表板数字 → 设对应状态为当前筛选。"""
        status_map = {
            "总 ": "",
            "手审 ": config.STATUS_PENDING_MANUAL_REVIEW,
            "通过 ": config.STATUS_REVIEWED,
            "成品 ": config.STATUS_FINISHED,
        }
        if not self._review_mode:
            status_map["优化 "] = config.STATUS_PENDING_OPTIMIZATION
        self._status_filter = status_map.get(status, "")
        self._filter_var.set(self._status_filter if self._status_filter else "全部")
        if self._on_filter:
            self._on_filter(self._status_filter)

    def _on_export_click(self) -> None:
        if self._on_export:
            self._on_export()

    def _on_menu_click(self) -> None:
        if self._on_menu:
            self._on_menu()
