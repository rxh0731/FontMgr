# tool_panel.py — 右侧工具面板：变换、去杂、详情、缩略条

import os
import tkinter as tk
from typing import Callable, Optional, TYPE_CHECKING

from services.glyph_service import GlyphService
from ui import theme
import config

if TYPE_CHECKING:
    from ui.edit_canvas import EditCanvas


class ToolPanel(tk.Frame):
    """右侧工具面板。"""

    def __init__(
        self,
        parent: tk.Widget,
        glyph_service: GlyphService,
        on_select_variant: Optional[Callable[[int], None]] = None,
        on_revert: Optional[Callable[[], None]] = None,
        on_refresh: Optional[Callable[[], None]] = None,
        allowed_statuses: Optional[tuple[str, ...]] = None,
        allow_variant_management: bool = True,
        show_variant_thumbnails: bool = True,
        on_save: Optional[Callable[[bool], bool]] = None,
        on_review_approved: Optional[Callable[[str, int], None]] = None,
        edit_canvas: Optional["EditCanvas"] = None,
    ) -> None:
        super().__init__(parent, bg=theme.BG_PANEL)
        self._glyph: GlyphService = glyph_service
        self._current_char: str = ""
        self._current_variant_index: int = 0
        self._on_select_variant = on_select_variant
        self._on_revert_canvas = on_revert
        self._on_refresh = on_refresh
        self._allowed_statuses = set(allowed_statuses) if allowed_statuses else None
        self._allow_variant_management = allow_variant_management
        self._show_variant_thumbnails = show_variant_thumbnails
        self._on_save = on_save
        self._on_review_approved = on_review_approved
        self._edit_canvas = edit_canvas

        self._build()

    def _build(self) -> None:
        """构建面板内容。"""
        pad = 8

        if self._edit_canvas is not None:
            self._edit_canvas.build_tool_panel(self).pack(fill=tk.X)
        else:
            # 其他流程继续使用原有参数面板；手工审核页由画布自由变换接管。
            sect = self._add_section("变换")
            self._scale_var = tk.DoubleVar(value=1.0)
            self._build_slider(sect, "缩放", self._scale_var, 0.5, 2.0)
            self._rotate_var = tk.DoubleVar(value=0.0)
            self._build_slider(sect, "旋转", self._rotate_var, -180, 180)
            self._offset_x_var = tk.IntVar(value=0)
            self._build_spin(sect, "偏移X", self._offset_x_var, -100, 100)
            self._offset_y_var = tk.IntVar(value=0)
            self._build_spin(sect, "偏移Y", self._offset_y_var, -100, 100)

        theme.make_button(
            self, "还原至上次保存", command=self._on_revert
        ).pack(fill=tk.X, padx=6, pady=(8, 0))

        # === 审核操作 ===
        sect3 = self._add_section("审核操作")
        theme.make_button(sect3, "保存并审核通过", accent=True, command=self._on_promote).pack(fill=tk.X, pady=2)

        # === 变体缩略条 ===
        self._thumb_canvas = tk.Canvas(self, bg=theme.BG_PANEL, height=0, highlightthickness=0)
        self._thumb_frame = tk.Frame(self._thumb_canvas, bg=theme.BG_PANEL)
        if self._show_variant_thumbnails:
            sect5 = self._add_section("变体缩略条")
            self._thumb_canvas = tk.Canvas(sect5, bg=theme.BG_PANEL, height=70, highlightthickness=0)
            self._thumb_canvas.pack(fill=tk.X, pady=2)
            self._thumb_frame = tk.Frame(self._thumb_canvas, bg=theme.BG_PANEL)
            self._thumb_canvas.create_window((0, 0), window=self._thumb_frame, anchor=tk.NW)
            self._thumb_frame.bind(
                "<Configure>",
                lambda e: self._thumb_canvas.configure(scrollregion=self._thumb_canvas.bbox("all")),
            )

            nav_row = tk.Frame(sect5, bg=theme.BG_PANEL)
            nav_row.pack(fill=tk.X)
            theme.make_button(nav_row, "←", command=self._prev_variant).pack(side=tk.LEFT, padx=2)
            theme.make_button(nav_row, "→", command=self._next_variant).pack(side=tk.RIGHT, padx=2)

            if self._allow_variant_management:
                theme.make_button(sect5, "移除此字", danger=True, command=self._on_remove_char).pack(fill=tk.X, pady=4)

        if self._on_save:
            action_box = tk.Frame(self, bg=theme.BG_PANEL)
            action_box.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=10)
            theme.make_button(
                action_box, "保存修订稿", accent=True, command=self._on_save
            ).pack(fill=tk.X, ipady=4)

    def _add_section(self, title: str) -> tk.Frame:
        """创建带标题的分区。"""
        frame = tk.Frame(self, bg=theme.BG_PANEL, bd=0, highlightthickness=0)
        frame.pack(fill=tk.X, padx=6, pady=(8, 4))
        theme.make_label(frame, title, font=theme.FONT_BOLD).pack(anchor="w", pady=(0, 4))
        return frame

    def _build_slider(self, parent: tk.Widget, label: str, var: tk.DoubleVar, lo: float, hi: float) -> None:
        row = tk.Frame(parent, bg=theme.BG_PANEL)
        row.pack(fill=tk.X, pady=1)
        theme.make_label(row, label, width=6, anchor="w", fg=theme.FG_MUTED).pack(side=tk.LEFT)
        scale = tk.Scale(row, from_=lo, to=hi, variable=var, orient=tk.HORIZONTAL,
                         resolution=0.01 if isinstance(var, tk.DoubleVar) else 1,
                         bg=theme.BG_PANEL, fg=theme.FG_PRIMARY, troughcolor=theme.BG_INPUT,
                         highlightthickness=0, bd=0, length=140)
        scale.pack(side=tk.LEFT)

    def _build_spin(self, parent: tk.Widget, label: str, var: tk.IntVar, lo: int, hi: int) -> None:
        row = tk.Frame(parent, bg=theme.BG_PANEL)
        row.pack(fill=tk.X, pady=1)
        theme.make_label(row, label, width=6, anchor="w", fg=theme.FG_MUTED).pack(side=tk.LEFT)
        sp = tk.Spinbox(row, from_=lo, to=hi, textvariable=var, width=6,
                        bg=theme.BG_INPUT, fg=theme.FG_PRIMARY, buttonbackground=theme.BG_PANEL, bd=0)
        sp.pack(side=tk.LEFT)

    # ==================== 加载/刷新 ====================

    def _get_variants(self, char: str) -> list[dict]:
        """返回当前流程可操作且已有自动优化预览的变体。"""
        variants = self._glyph.get_char_variants(char) if char else []
        if self._allowed_statuses is None:
            return variants
        return [
            variant
            for variant in variants
            if variant.get("状态") in self._allowed_statuses and variant.get("中间文件")
        ]

    def load_char(self, char: str, variant_index: int = 0) -> None:
        """加载指定汉字及字形详情。"""
        self._current_char = char
        variants = self._get_variants(char)
        if not variants:
            return

        self._current_variant_index = min(max(0, variant_index), len(variants) - 1)
        if self._show_variant_thumbnails:
            self._build_thumbnails(variants)
        self._refresh_detail()

    def refresh_current(self, char: str, variant_index: int = 0) -> None:
        """刷新当前汉字及字形详情。"""
        self.load_char(char, variant_index)

    def _build_thumbnails(self, variants: list) -> None:
        """构建缩略条。"""
        for w in self._thumb_frame.winfo_children():
            w.destroy()

        for i, v in enumerate(variants):
            status = v.get("状态", "")
            color = config.STATUS_COLORS.get(status, theme.FG_MUTED)

            fname = v.get("原始文件", "")
            lbl = tk.Label(self._thumb_frame, text=fname.split(".")[0], bg=theme.BG_PANEL,
                          fg=color, font=theme.FONT_SMALL, width=10, anchor="center", cursor="hand2")
            lbl.pack(side=tk.LEFT, padx=1, pady=2)
            lbl.bind("<Button-1>", lambda e, idx=i: self._on_thumb_click(idx))
            if self._allow_variant_management:
                lbl.bind("<Button-3>", lambda e, idx=i: self._on_thumb_rclick(idx))

    def _on_thumb_click(self, index: int) -> None:
        """点击缩略图 → 切换变体。"""
        self._current_variant_index = index
        if self._on_select_variant:
            self._on_select_variant(index)
        self._refresh_detail()

    def _on_thumb_rclick(self, index: int) -> None:
        """右键缩略图 → 确认后移除该变体。"""
        from ui.widgets.custom_dialog import show_confirm
        variants = self._get_variants(self._current_char)
        if index >= len(variants):
            return
        variant_id = variants[index].get("变体ID", "")
        if variant_id and show_confirm(self, "移除变体", "确定移除这个变体？"):
            self._glyph.remove_variant(variant_id)
            self._glyph.save()
            remaining = self._get_variants(self._current_char)
            self._current_variant_index = min(index, max(0, len(remaining) - 1))
            self._build_thumbnails(remaining)
            if remaining and self._on_select_variant:
                self._on_select_variant(self._current_variant_index)
            if self._on_refresh:
                self._on_refresh()

    # ==================== 操作回调 ====================

    def _on_promote(self) -> None:
        """保存当前修订稿并审核通过。"""
        variants = self._get_variants(self._current_char)
        if self._current_variant_index >= len(variants):
            return
        variant_id = variants[self._current_variant_index].get("变体ID", "")
        reviewed_char = self._current_char
        reviewed_index = self._current_variant_index
        if self._on_save and not self._on_save(False):
            return
        if variant_id and self._glyph.approve_manual_review(variant_id):
            self._glyph.save()
            self.load_char(self._current_char)
            if self._on_review_approved:
                self._on_review_approved(reviewed_char, reviewed_index)
            elif self._on_refresh:
                self._on_refresh()
        elif self._on_refresh:
            self._on_refresh()

    def _on_revert(self) -> None:
        """还原编辑。"""
        if self._on_revert_canvas:
            self._on_revert_canvas()

    def _on_remove_char(self) -> None:
        """移除此字。"""
        from ui.widgets.custom_dialog import show_confirm
        if show_confirm(self, "移除确认", f"确定移除「{self._current_char}」及其全部变体？"):
            self._glyph.remove_char(self._current_char)

    def _prev_variant(self) -> None:
        variants = self._get_variants(self._current_char)
        if variants:
            self._on_thumb_click((self._current_variant_index - 1) % len(variants))

    def _next_variant(self) -> None:
        variants = self._get_variants(self._current_char)
        if variants:
            self._on_thumb_click((self._current_variant_index + 1) % len(variants))

    def _refresh_detail(self) -> None:
        variants = self._get_variants(self._current_char)
        if not variants:
            return
        self._current_variant_index = min(self._current_variant_index, len(variants) - 1)
