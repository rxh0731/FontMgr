# consistency_page.py — 字库整体协调页面

import copy
import math
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Optional

from PIL import Image, ImageTk

import config
from services.adjustment_service import AdjustmentService
from services.glyph_service import GlyphService
from ui import theme
from ui.glyph_tree import GlyphTree
from ui.top_toolbar import TopToolbar
from ui.widgets.custom_dialog import ask_save_discard_cancel, show_error, show_info


class ConsistencyPage(tk.Frame):
    """审核通过字形的分页总览与逐字几何协调。"""

    WORK_RATIO = 1.3
    MAX_PAGE_SIZE = 30
    MIN_CELL_SIZE = 112

    def __init__(
        self,
        parent: tk.Widget,
        glyph_service: GlyphService,
        on_back: Callable[[], None],
    ) -> None:
        super().__init__(parent, bg=theme.BG_MAIN)
        self._glyph = glyph_service
        self._adjustment = AdjustmentService(glyph_service)
        self._on_back = on_back
        self._order = GlyphTree.ORDER_PINYIN
        self._variants = self._adjustment.load_reviewed_variants(pinyin_order=True)
        self._variant_by_id = {
            str(item.get("变体ID", "")): item for item in self._variants
        }
        self._selected_id = str(self._variants[0].get("变体ID", "")) if self._variants else ""
        self._page_index = 0
        self._page_size = 1
        self._columns = 1
        self._rows = 1
        self._cell_width = 1
        self._cell_height = 1
        self._gap_x = 1
        self._gap_y = 1
        self._photo_refs: dict[str, ImageTk.PhotoImage] = {}
        self._base_preview_cache: dict[str, tuple[Image.Image, tuple[int, int, int, int]]] = {}
        self._cells: dict[str, dict[str, Any]] = {}
        self._adjustments: dict[str, dict[str, Any]] = {}
        self._saved_signatures: dict[str, tuple[tuple[str, Any], ...]] = {}
        self._layout_after: Optional[str] = None
        self._drag: Optional[dict[str, Any]] = None
        self._edit_enabled = tk.BooleanVar(value=False)
        self._background_mode = tk.StringVar(value="白底")
        self._requested_columns = tk.IntVar(value=5)
        self._page_text = tk.StringVar(value="第 0 / 0 页")
        self._status_text = tk.StringVar(value="")
        self._build()
        self.after_idle(self._schedule_layout)

    def _build(self) -> None:
        TopToolbar(
            self,
            self._glyph,
            on_back=self._request_back,
            review_mode=True,
            page_title="整体协调",
        ).pack(fill=tk.X)

        body = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 10))

        left = tk.Frame(body, bg=theme.BG_PANEL, width=285)
        center = tk.Frame(body, bg=theme.BG_MAIN)
        right = tk.Frame(body, bg=theme.BG_PANEL, width=235)
        body.add(left, weight=0)
        body.add(center, weight=1)
        body.add(right, weight=0)

        self._tree = GlyphTree(
            left,
            self._glyph,
            on_select=self._select_from_tree,
            allowed_statuses=(config.STATUS_REVIEWED, config.STATUS_FINISHED),
            allow_context_menu=False,
            show_score=False,
            require_intermediate_file=False,
            on_order_change=self._on_order_change,
            summary_pending_label="待协调",
            summary_completed_label="已协调",
            summary_pending_statuses=(config.STATUS_REVIEWED,),
            summary_completed_statuses=(config.STATUS_FINISHED,),
        )
        self._tree.pack(fill=tk.BOTH, expand=True)

        navigation = tk.Frame(center, bg=theme.BG_PANEL)
        navigation.pack(fill=tk.X)
        theme.make_button(navigation, "上一页", command=lambda: self._change_page(-1), width=8).pack(
            side=tk.LEFT, padx=(10, 4), pady=7
        )
        theme.make_label(
            navigation, textvariable=self._page_text, bg=theme.BG_PANEL, fg=theme.FG_SECONDARY
        ).pack(side=tk.LEFT, padx=8)
        theme.make_button(navigation, "下一页", command=lambda: self._change_page(1), width=8).pack(
            side=tk.LEFT, padx=4, pady=7
        )
        background_menu = tk.OptionMenu(
            navigation,
            self._background_mode,
            "白底",
            "棋盘格",
            command=lambda _value: self._render_page(),
        )
        self._style_option_menu(background_menu)
        background_menu.pack(side=tk.RIGHT, padx=(4, 10), pady=7)
        theme.make_label(navigation, "背景", bg=theme.BG_PANEL, fg=theme.FG_SECONDARY).pack(
            side=tk.RIGHT
        )
        columns_spin = tk.Spinbox(
            navigation,
            from_=1,
            to=12,
            textvariable=self._requested_columns,
            width=4,
            command=self._schedule_layout,
            bg=theme.BG_INPUT,
            fg=theme.FG_PRIMARY,
            buttonbackground=theme.BG_PANEL,
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=theme.BORDER,
            font=theme.FONT_SMALL,
        )
        columns_spin.pack(side=tk.RIGHT, padx=(4, 14), pady=7)
        columns_spin.bind("<Return>", lambda _event: self._schedule_layout())
        columns_spin.bind("<FocusOut>", lambda _event: self._schedule_layout())
        theme.make_label(
            navigation,
            "每行字数",
            bg=theme.BG_PANEL,
            fg=theme.FG_SECONDARY,
        ).pack(side=tk.RIGHT)

        self._grid_canvas = tk.Canvas(
            center,
            bg=theme.BG_MAIN,
            highlightthickness=0,
            takefocus=True,
        )
        self._grid_canvas.pack(fill=tk.BOTH, expand=True)
        self._grid_canvas.bind("<Configure>", lambda _event: self._schedule_layout())
        self._grid_canvas.bind("<Button-1>", self._on_canvas_press)
        self._grid_canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self._grid_canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self._grid_canvas.bind("<Motion>", self._update_cursor)
        for key in ("Alt_L", "Alt_R", "Control_L", "Control_R", "Shift_L", "Shift_R"):
            self._grid_canvas.bind(f"<KeyPress-{key}>", self._update_cursor)
            self._grid_canvas.bind(f"<KeyRelease-{key}>", self._update_cursor)

        theme.make_label(
            center,
            textvariable=self._status_text,
            bg=theme.BG_MAIN,
            fg=theme.FG_MUTED,
            font=theme.FONT_SMALL,
        ).pack(fill=tk.X, padx=8, pady=(2, 0))

        self._body_panes = body
        self.after_idle(lambda: self._set_initial_panes(left, right))
        self._build_tools(right)

    def _set_initial_panes(self, left: tk.Widget, right: tk.Widget) -> None:
        total_width = self._body_panes.winfo_width()
        if total_width <= 560:
            return
        try:
            self._body_panes.sashpos(0, min(285, max(210, total_width // 5)))
            self._body_panes.sashpos(1, max(360, total_width - min(235, max(190, total_width // 6))))
        except tk.TclError:
            left.configure({"width": 285})
            right.configure({"width": 235})

    def _build_tools(self, parent: tk.Widget) -> None:
        row = tk.Frame(parent, bg=theme.BG_PANEL)
        row.pack(fill=tk.X, padx=14, pady=(14, 8))
        theme.make_label(row, "编辑开关", bg=theme.BG_PANEL, font=theme.FONT_BOLD).pack(side=tk.LEFT)
        switch = tk.Checkbutton(
            row,
            text="开启",
            variable=self._edit_enabled,
            command=self._on_edit_toggle,
            bg=theme.BG_PANEL,
            fg=theme.FG_PRIMARY,
            activebackground=theme.BG_PANEL,
            activeforeground=theme.FG_PRIMARY,
            selectcolor=theme.BG_INPUT,
            font=theme.FONT_SMALL,
        )
        switch.pack(side=tk.RIGHT)

        theme.make_label(parent, "自由变换说明", bg=theme.BG_PANEL, font=theme.FONT_BOLD).pack(
            anchor=tk.W, padx=14, pady=(12, 6)
        )
        shortcuts = (
            "拖动字形：移动\n"
            "拖动旋转手柄：旋转\n"
            "拖动四角手柄：等比缩放\n"
            "拖动四边手柄：拉伸、压缩\n"
            "Shift + 四边手柄：等比缩放\n"
            "Alt + 缩放手柄：从中心缩放\n"
            "Ctrl + 任意缩放手柄：自由扭曲"
        )
        theme.make_label(
            parent,
            shortcuts,
            bg=theme.BG_PANEL,
            fg=theme.FG_SECONDARY,
            justify=tk.LEFT,
            wraplength=205,
        ).pack(anchor=tk.W, padx=14)

        tk.Frame(parent, height=1, bg=theme.BORDER).pack(fill=tk.X, padx=14, pady=16)
        self._restore_button = theme.make_button(
            parent, "还原本字", command=self._restore_selected, width=16
        )
        self._restore_button.pack(fill=tk.X, padx=14, pady=4)
        self._save_button = theme.make_button(
            parent, "保存当前页", accent=True, command=self._save_current_page, width=16
        )
        self._save_button.pack(fill=tk.X, padx=14, pady=4)
        if not self._variants:
            self._restore_button.configure(state=tk.DISABLED)
            self._save_button.configure(state=tk.DISABLED)

    @staticmethod
    def _style_option_menu(menu: tk.OptionMenu) -> None:
        menu.configure(
            bg=theme.BG_INPUT,
            fg=theme.FG_PRIMARY,
            activebackground=theme.BG_HOVER,
            activeforeground=theme.FG_PRIMARY,
            relief=tk.FLAT,
            bd=0,
            highlightthickness=1,
            highlightbackground=theme.BORDER,
            font=theme.FONT_SMALL,
            width=7,
        )
        menu["menu"].configure(
            bg=theme.BG_PANEL,
            fg=theme.FG_PRIMARY,
            activebackground=theme.BG_HOVER,
            activeforeground=theme.FG_PRIMARY,
            font=theme.FONT_SMALL,
        )

    def _schedule_layout(self) -> None:
        if self._layout_after is not None:
            self.after_cancel(self._layout_after)
        self._layout_after = self.after(160, self._recalculate_layout)

    def _recalculate_layout(self) -> None:
        self._layout_after = None
        width = max(1, self._grid_canvas.winfo_width())
        height = max(1, self._grid_canvas.winfo_height())
        metadata = self._glyph.get_metadata()
        grid_width = max(1, int(metadata.get("画布宽", 250)))
        grid_height = max(1, int(metadata.get("画布高", 250)))
        aspect = grid_width / grid_height
        try:
            requested_columns = int(self._requested_columns.get())
        except (tk.TclError, ValueError):
            requested_columns = 5
        requested_columns = max(1, min(12, requested_columns))
        if self._requested_columns.get() != requested_columns:
            self._requested_columns.set(requested_columns)

        # 优先使用目标列数；窗口过窄时只临时减少实际列数，不覆盖用户设置。
        best: Optional[tuple[int, int, int, int, int, int, int]] = None
        base_gap = 8
        for columns in range(requested_columns, 0, -1):
            maximum_width = (width - (columns + 1) * base_gap) / columns
            cell_width = max(1, int(maximum_width))
            cell_height = max(1, int(cell_width / aspect))
            if cell_height > height - base_gap * 2:
                cell_height = max(1, height - base_gap * 2)
                cell_width = max(1, int(cell_height * aspect))
            if columns > 1 and min(cell_width, cell_height) < self.MIN_CELL_SIZE:
                continue
            rows = max(1, min(self.MAX_PAGE_SIZE // columns, (height - base_gap) // (cell_height + base_gap)))
            gap_x = max(1, int((width - columns * cell_width) / (columns + 1)))
            gap_y = max(1, int((height - rows * cell_height) / (rows + 1)))
            best = (columns * rows, columns, rows, cell_width, cell_height, gap_x, gap_y)
            break
        if best is None:
            cell_height = max(1, min(height - 2, int((width - 2) / aspect)))
            best = (1, 1, 1, max(1, int(cell_height * aspect)), cell_height, 1, 1)

        old_size = self._page_size
        selected_index = self._selected_index()
        self._page_size, self._columns, self._rows, self._cell_width, self._cell_height, self._gap_x, self._gap_y = best
        if selected_index >= 0 and old_size != self._page_size:
            self._page_index = selected_index // self._page_size
        self._render_page()

    def _render_page(self) -> None:
        canvas = self._grid_canvas
        canvas.delete("all")
        self._photo_refs.clear()
        self._cells.clear()
        total_pages = max(1, math.ceil(len(self._variants) / max(1, self._page_size)))
        self._page_index = max(0, min(self._page_index, total_pages - 1))
        start = self._page_index * self._page_size
        page = self._variants[start:start + self._page_size]
        gap_x = max(1.0, float(self._gap_x))
        gap_y = max(1.0, float(self._gap_y))
        for index, detail in enumerate(page):
            row, column = divmod(index, self._columns)
            left = int(round(gap_x + column * (self._cell_width + gap_x)))
            top = int(round(gap_y + row * (self._cell_height + gap_y)))
            variant_id = str(detail.get("变体ID", ""))
            self._draw_cell(detail, variant_id, left, top, self._cell_width, self._cell_height)
        self._page_text.set(f"第 {self._page_index + 1 if self._variants else 0} / {total_pages if self._variants else 0} 页")
        changed = sum(1 for item in page if self._is_dirty(str(item.get("变体ID", ""))))
        self._status_text.set(f"本页 {len(page)} 字　未保存调整 {changed} 字")

    def _draw_cell(
        self,
        detail: dict[str, Any],
        variant_id: str,
        left: int,
        top: int,
        width: int,
        height: int,
    ) -> None:
        selected = variant_id == self._selected_id
        border = theme.FG_ACCENT if selected else theme.BORDER
        self._grid_canvas.create_rectangle(
            left,
            top,
            left + width,
            top + height,
            fill="#FFFFFF" if self._background_mode.get() == "白底" else "#E6E6E6",
            outline=border,
            width=3 if selected else 1,
            tags=("cell", variant_id),
        )
        if self._background_mode.get() == "棋盘格":
            self._draw_checkerboard(left, top, width, height, variant_id)
        metadata = self._glyph.get_metadata()
        grid_width = max(1, int(metadata.get("画布宽", 250)))
        grid_height = max(1, int(metadata.get("画布高", 250)))
        scale = min(width / (grid_width * self.WORK_RATIO), height / (grid_height * self.WORK_RATIO))
        display_grid_width = grid_width * scale
        display_grid_height = grid_height * scale
        grid_left = left + (width - display_grid_width) / 2
        grid_top = top + (height - display_grid_height) / 2
        grid_right = grid_left + display_grid_width
        grid_bottom = grid_top + display_grid_height
        grid_color = "#D88787"
        self._grid_canvas.create_rectangle(grid_left, grid_top, grid_right, grid_bottom, outline=grid_color, tags=("cell", variant_id))
        self._grid_canvas.create_line((grid_left + grid_right) / 2, grid_top, (grid_left + grid_right) / 2, grid_bottom, fill=grid_color, dash=(4, 3), tags=("cell", variant_id))
        self._grid_canvas.create_line(grid_left, (grid_top + grid_bottom) / 2, grid_right, (grid_top + grid_bottom) / 2, fill=grid_color, dash=(4, 3), tags=("cell", variant_id))
        self._grid_canvas.create_line(grid_left, grid_top, grid_right, grid_bottom, fill="#EABABA", dash=(3, 4), tags=("cell", variant_id))
        self._grid_canvas.create_line(grid_right, grid_top, grid_left, grid_bottom, fill="#EABABA", dash=(3, 4), tags=("cell", variant_id))

        adjustments = self._adjustments.get(variant_id)
        if adjustments is None and variant_id in self._base_preview_cache:
            preview = self._base_preview_cache[variant_id]
        else:
            preview = self._adjustment.preview_coordinated(detail, adjustments, self.WORK_RATIO)
            if adjustments is None and preview is not None:
                self._base_preview_cache[variant_id] = preview
        display_bbox: Optional[tuple[float, float, float, float]] = None
        if preview is not None:
            image, bounding_box = preview
            image_width = max(1, int(round(image.width * scale)))
            image_height = max(1, int(round(image.height * scale)))
            image = image.resize((image_width, image_height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self._photo_refs[variant_id] = photo
            image_left = left + (width - image_width) / 2
            image_top = top + (height - image_height) / 2
            self._grid_canvas.create_image(image_left, image_top, image=photo, anchor=tk.NW, tags=("cell", variant_id))
            display_bbox = (
                image_left + bounding_box[0] * scale,
                image_top + bounding_box[1] * scale,
                image_left + bounding_box[2] * scale,
                image_top + bounding_box[3] * scale,
            )
        self._cells[variant_id] = {
            "rect": (left, top, left + width, top + height),
            "bbox": display_bbox,
            "scale": scale,
        }
        if selected and self._edit_enabled.get() and display_bbox is not None:
            self._draw_handles(variant_id, display_bbox)

    def _draw_checkerboard(self, left: int, top: int, width: int, height: int, variant_id: str) -> None:
        size = max(7, min(width, height) // 14)
        for y in range(top, top + height, size):
            for x in range(left, left + width, size):
                if ((x - left) // size + (y - top) // size) % 2:
                    self._grid_canvas.create_rectangle(
                        x,
                        y,
                        min(x + size, left + width),
                        min(y + size, top + height),
                        fill="#CFCFCF",
                        outline="",
                        tags=("cell", variant_id),
                    )

    def _draw_handles(self, variant_id: str, bounding_box: tuple[float, float, float, float]) -> None:
        left, top, right, bottom = bounding_box
        handle_color = "#5a7fbf"
        handle_size = 4
        self._grid_canvas.create_rectangle(
            left,
            top,
            right,
            bottom,
            outline=handle_color,
            dash=(4, 3),
            tags=("handle-box", variant_id),
        )
        points = {
            "nw": (left, top), "n": ((left + right) / 2, top), "ne": (right, top),
            "e": (right, (top + bottom) / 2), "se": (right, bottom),
            "s": ((left + right) / 2, bottom), "sw": (left, bottom), "w": (left, (top + bottom) / 2),
        }
        for name, (x, y) in points.items():
            self._grid_canvas.create_rectangle(
                x - handle_size,
                y - handle_size,
                x + handle_size,
                y + handle_size,
                fill=handle_color,
                outline=theme.FG_PRIMARY,
                tags=("handle", f"handle:{name}", variant_id),
            )
        top_mid_x, top_mid_y = points["n"]
        rotate_y = top_mid_y - handle_size * 2
        self._grid_canvas.create_line(
            top_mid_x,
            top_mid_y,
            top_mid_x,
            rotate_y,
            fill=handle_color,
            tags=("handle", "handle:rotate", variant_id),
        )
        self._grid_canvas.create_oval(
            top_mid_x - handle_size,
            rotate_y - handle_size,
            top_mid_x + handle_size,
            rotate_y + handle_size,
            fill=handle_color,
            outline=theme.FG_PRIMARY,
            tags=("handle", "handle:rotate", variant_id),
        )

    def _select_from_tree(self, char: str, variant_index: int) -> None:
        group = self._glyph.get_char_variants(char)
        if variant_index < 0 or variant_index >= len(group):
            return
        variant_id = str(group[variant_index].get("变体ID", ""))
        if variant_id not in self._variant_by_id:
            return
        index = next(
            (item_index for item_index, item in enumerate(self._variants) if str(item.get("变体ID", "")) == variant_id),
            -1,
        )
        target_page = index // max(1, self._page_size) if index >= 0 else self._page_index
        if target_page != self._page_index and not self._confirm_leave_page():
            current_index = self._selected_index()
            if current_index >= 0:
                current = self._variants[current_index]
                self._tree.restore_selection(str(current.get("归属字", "")), int(current.get("变体序号", 1)) - 1)
            return
        self._selected_id = variant_id
        self._page_index = target_page
        self._render_page()

    def _on_order_change(self, order: str) -> bool:
        if order == self._order:
            return True
        if not self._confirm_leave_page():
            return False
        self._order = order
        self._variants = self._adjustment.load_reviewed_variants(
            pinyin_order=order == GlyphTree.ORDER_PINYIN,
        )
        self._variant_by_id = {str(item.get("变体ID", "")): item for item in self._variants}
        selected_index = self._selected_index()
        if selected_index < 0:
            self._selected_id = str(self._variants[0].get("变体ID", "")) if self._variants else ""
            selected_index = 0
        self._page_index = selected_index // max(1, self._page_size)
        self._render_page()
        return True

    def _select_canvas_variant(self, variant_id: str) -> None:
        if variant_id not in self._variant_by_id:
            return
        self._selected_id = variant_id
        detail = self._variant_by_id[variant_id]
        char = str(detail.get("归属字", ""))
        group = self._glyph.get_char_variants(char)
        variant_index = next(
            (index for index, item in enumerate(group) if str(item.get("变体ID", "")) == variant_id),
            0,
        )
        self._tree.restore_selection(char, variant_index)
        self._render_page()

    def _selected_index(self) -> int:
        for index, detail in enumerate(self._variants):
            if str(detail.get("变体ID", "")) == self._selected_id:
                return index
        return -1

    def _variant_at(self, x: int, y: int) -> str:
        for variant_id, cell in self._cells.items():
            left, top, right, bottom = cell["rect"]
            if left <= x <= right and top <= y <= bottom:
                return variant_id
        return ""

    def _handle_at(self, x: int, y: int) -> str:
        current = self._grid_canvas.find_overlapping(x - 3, y - 3, x + 3, y + 3)
        for item in reversed(current):
            for tag in self._grid_canvas.gettags(item):
                if tag.startswith("handle:"):
                    return tag.split(":", 1)[1]
        return ""

    @staticmethod
    def _modifier_state(state: int | str) -> tuple[bool, bool, bool]:
        value = int(state)
        alt = bool(value & 0x0008) or bool(value & 0x20000)
        control = bool(value & 0x0004)
        shift = bool(value & 0x0001)
        return alt, control, shift

    def _on_canvas_press(self, event: tk.Event) -> None:
        self._grid_canvas.focus_set()
        variant_id = self._variant_at(event.x, event.y)
        if not variant_id:
            return
        if variant_id != self._selected_id:
            self._select_canvas_variant(variant_id)
        if not self._edit_enabled.get():
            return
        handle = self._handle_at(event.x, event.y)
        alt, control, shift = self._modifier_state(event.state)
        cell = self._cells.get(variant_id)
        bounding_box = cell.get("bbox") if cell is not None else None
        rotate_angle = 0.0
        rotate_center = (0.0, 0.0)
        if handle == "rotate" and bounding_box is not None:
            center_x = (bounding_box[0] + bounding_box[2]) / 2.0
            center_y = (bounding_box[1] + bounding_box[3]) / 2.0
            rotate_center = (center_x, center_y)
            rotate_angle = math.degrees(math.atan2(event.y - center_y, event.x - center_x))
        self._drag = {
            "x": event.x,
            "y": event.y,
            "handle": handle or "move",
            "alt": alt,
            "control": control,
            "shift": shift,
            "start": copy.deepcopy(self._get_adjustment(variant_id)),
            "bbox": bounding_box,
            "rotate_angle": rotate_angle,
            "rotate_center": rotate_center,
        }

    def _on_canvas_drag(self, event: tk.Event) -> None:
        if self._drag is None or not self._selected_id:
            return
        cell = self._cells.get(self._selected_id)
        if cell is None:
            return
        scale = max(float(cell["scale"]), 0.0001)
        screen_dx = event.x - self._drag["x"]
        screen_dy = event.y - self._drag["y"]
        dx = screen_dx / scale
        dy = screen_dy / scale
        start = self._drag["start"]
        value = copy.deepcopy(start)
        handle = self._drag["handle"]
        if self._drag["control"] and handle not in ("move", "rotate"):
            self._apply_corner_distort(value, start, handle, dx, dy)
        elif handle == "move":
            value["移动X"] = float(start["移动X"]) + dx
            value["移动Y"] = float(start["移动Y"]) + dy
        elif handle == "rotate":
            center_x, center_y = self._drag["rotate_center"]
            angle = math.degrees(math.atan2(event.y - center_y, event.x - center_x))
            delta = angle - float(self._drag["rotate_angle"])
            while delta > 180.0:
                delta -= 360.0
            while delta < -180.0:
                delta += 360.0
            value["旋转"] = float(start["旋转"]) + delta
        else:
            bounding_box = self._drag.get("bbox")
            if bounding_box is not None:
                self._apply_handle_scale(value, start, handle, screen_dx, screen_dy, bounding_box)
        self._adjustments[self._selected_id] = value
        self._redraw_selected()

    def _apply_handle_scale(
        self,
        value: dict[str, Any],
        start: dict[str, Any],
        handle: str,
        dx: float,
        dy: float,
        bounding_box: tuple[float, float, float, float],
    ) -> None:
        width = max(1.0, bounding_box[2] - bounding_box[0])
        height = max(1.0, bounding_box[3] - bounding_box[1])
        direction_x = -1.0 if "w" in handle else (1.0 if "e" in handle else 0.0)
        direction_y = -1.0 if "n" in handle else (1.0 if "s" in handle else 0.0)
        center_factor = 2.0 if self._drag and self._drag["alt"] else 1.0
        factor_x = 1.0 + direction_x * dx * center_factor / width
        factor_y = 1.0 + direction_y * dy * center_factor / height
        proportional = len(handle) == 2 or bool(self._drag and self._drag["shift"])
        if proportional:
            candidates = []
            if direction_x:
                candidates.append(factor_x)
            if direction_y:
                candidates.append(factor_y)
            factor = max(candidates, key=lambda item: abs(item - 1.0)) if candidates else 1.0
            factor_x = factor_y = factor
        elif not direction_x:
            factor_x = 1.0
        elif not direction_y:
            factor_y = 1.0
        value["缩放X"] = max(0.15, min(5.0, float(start["缩放X"]) * factor_x))
        value["缩放Y"] = max(0.15, min(5.0, float(start["缩放Y"]) * factor_y))

    @staticmethod
    def _apply_corner_distort(
        value: dict[str, Any],
        start: dict[str, Any],
        handle: str,
        dx: float,
        dy: float,
    ) -> None:
        corner_by_handle = {
            "nw": 0, "n": 0, "w": 0,
            "ne": 1, "e": 1,
            "se": 2, "s": 2,
            "sw": 3,
        }
        corner = corner_by_handle.get(handle)
        if corner is None:
            return
        distort = list(start.get("扭曲", [0.0] * 8))
        distort[corner * 2] += dx
        distort[corner * 2 + 1] += dy
        value["扭曲"] = distort

    def _redraw_selected(self) -> None:
        variant_id = self._selected_id
        cell = self._cells.get(variant_id)
        detail = self._variant_by_id.get(variant_id)
        if cell is None or detail is None:
            return
        left, top, right, bottom = cell["rect"]
        self._grid_canvas.delete(variant_id)
        self._photo_refs.pop(variant_id, None)
        self._draw_cell(
            detail,
            variant_id,
            int(left),
            int(top),
            max(1, int(right - left)),
            max(1, int(bottom - top)),
        )

    def _on_canvas_release(self, _event: tk.Event) -> None:
        self._drag = None
        self._update_cursor(_event)

    def _update_cursor(self, event: tk.Event) -> None:
        if not self._edit_enabled.get():
            self._grid_canvas.configure(cursor="arrow")
            return
        _alt, control, _shift = self._modifier_state(getattr(event, "state", 0))
        handle = self._handle_at(getattr(event, "x", 0), getattr(event, "y", 0))
        if control and handle not in (None, "", "rotate"):
            cursor = "crosshair"
        elif handle == "rotate":
            cursor = "exchange"
        elif handle in ("nw", "se"):
            cursor = "size_nw_se"
        elif handle in ("ne", "sw"):
            cursor = "size_ne_sw"
        elif handle in ("e", "w"):
            cursor = "sb_h_double_arrow"
        elif handle in ("n", "s"):
            cursor = "sb_v_double_arrow"
        elif handle:
            cursor = "sizing"
        elif self._variant_at(getattr(event, "x", 0), getattr(event, "y", 0)) == self._selected_id:
            cursor = "fleur"
        else:
            cursor = "arrow"
        self._grid_canvas.configure(cursor=cursor)

    @staticmethod
    def _default_adjustment() -> dict[str, Any]:
        return {
            "移动X": 0.0,
            "移动Y": 0.0,
            "缩放X": 1.0,
            "缩放Y": 1.0,
            "旋转": 0.0,
            "斜切X": 0.0,
            "斜切Y": 0.0,
            "扭曲": [0.0] * 8,
        }

    def _get_adjustment(self, variant_id: str) -> dict[str, Any]:
        return self._adjustments.setdefault(variant_id, self._default_adjustment())

    def _signature(self, variant_id: str) -> tuple[tuple[str, Any], ...]:
        result: list[tuple[str, Any]] = []
        for key, value in self._get_adjustment(variant_id).items():
            if isinstance(value, (list, tuple)):
                result.append((key, tuple(round(float(item), 6) for item in value)))
            else:
                result.append((key, round(float(value), 6)))
        return tuple(sorted(result))

    def _is_dirty(self, variant_id: str) -> bool:
        return self._signature(variant_id) != self._saved_signatures.get(variant_id, self._zero_signature())

    @classmethod
    def _zero_signature(cls) -> tuple[tuple[str, Any], ...]:
        result: list[tuple[str, Any]] = []
        for key, value in cls._default_adjustment().items():
            result.append((key, tuple(value) if isinstance(value, list) else value))
        return tuple(sorted(result))

    def _restore_selected(self) -> None:
        if not self._selected_id:
            return
        self._adjustments.pop(self._selected_id, None)
        self._render_page()

    def _save_current_page(self, show_success: bool = True) -> bool:
        start = self._page_index * self._page_size
        page = self._variants[start:start + self._page_size]
        if not page:
            return True
        self._save_button.configure(state=tk.DISABLED)
        self.update_idletasks()
        result = self._adjustment.save_coordinated_variants(page, self._adjustments)
        for detail in page:
            variant_id = str(detail.get("变体ID", ""))
            if not any(item[0] == variant_id for item in result["失败详情"]):
                self._saved_signatures[variant_id] = self._signature(variant_id)
        self._save_button.configure(state=tk.NORMAL)
        self._refresh_variants()
        if result["失败"]:
            failed = "\n".join(f"{variant_id}：{reason}" for variant_id, reason in result["失败详情"][:8])
            show_error(self, "保存当前页", f"成功 {result['成功']} 个，失败 {result['失败']} 个。\n\n{failed}")
            return False
        if show_success:
            show_info(self, "保存当前页", f"已将当前页 {result['成功']} 个字形保存为最终成品。")
        return True

    def _change_page(self, offset: int) -> None:
        total_pages = max(1, math.ceil(len(self._variants) / max(1, self._page_size)))
        target = max(0, min(self._page_index + offset, total_pages - 1))
        if target != self._page_index and self._confirm_leave_page():
            self._page_index = target
            self._render_page()

    def _page_variant_ids(self) -> list[str]:
        start = self._page_index * self._page_size
        return [
            str(detail.get("变体ID", ""))
            for detail in self._variants[start:start + self._page_size]
        ]

    def _confirm_leave_page(self) -> bool:
        dirty_ids = [variant_id for variant_id in self._page_variant_ids() if self._is_dirty(variant_id)]
        if not dirty_ids:
            return True
        choice = ask_save_discard_cancel(
            self,
            "未保存调整",
            "当前页面存在尚未保存的调整，是否保存后继续？",
        )
        if choice == "cancel":
            return False
        if choice == "save":
            return self._save_current_page(show_success=False)
        for variant_id in dirty_ids:
            saved = self._saved_signatures.get(variant_id)
            if saved is None or saved == self._zero_signature():
                self._adjustments.pop(variant_id, None)
            else:
                self._adjustments[variant_id] = {
                    key: list(value) if isinstance(value, tuple) else float(value)
                    for key, value in saved
                }
        return True

    def _on_edit_toggle(self) -> None:
        self._schedule_layout()

    def _refresh_variants(self) -> None:
        selected = self._selected_id
        self._variants = self._adjustment.load_reviewed_variants(
            pinyin_order=self._order == GlyphTree.ORDER_PINYIN,
        )
        self._variant_by_id = {str(item.get("变体ID", "")): item for item in self._variants}
        valid_ids = set(self._variant_by_id)
        self._base_preview_cache = {
            variant_id: preview
            for variant_id, preview in self._base_preview_cache.items()
            if variant_id in valid_ids
        }
        if selected not in self._variant_by_id:
            self._selected_id = str(self._variants[0].get("变体ID", "")) if self._variants else ""
        self._tree.refresh()
        self._render_page()

    def _request_back(self) -> None:
        if self._confirm_leave_page():
            self._on_back()
