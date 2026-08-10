# edit_canvas.py — 中央编辑画布（田字格 + 文字 + 8手柄 + 画笔 + 橡皮擦 + 快捷键）

import ctypes
import logging
import math
import os
import time
import tkinter as tk
from typing import Callable, Optional, cast

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from services.glyph_service import GlyphService
from ui import theme


_PERFORMANCE_LOGGER = logging.getLogger("手工审核耗时")
if not _PERFORMANCE_LOGGER.handlers:
    _PERFORMANCE_LOGGER.setLevel(logging.INFO)
    _PERFORMANCE_LOGGER.propagate = False
    _performance_formatter = logging.Formatter(
        "%(asctime)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _performance_file_handler = logging.FileHandler(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "font_editor.log"),
        encoding="utf-8",
    )
    _performance_file_handler.setFormatter(_performance_formatter)
    _PERFORMANCE_LOGGER.addHandler(_performance_file_handler)
    _performance_console_handler = logging.StreamHandler()
    _performance_console_handler.setFormatter(_performance_formatter)
    _PERFORMANCE_LOGGER.addHandler(_performance_console_handler)


class EditCanvas(tk.Frame):
    """中央编辑画布：支持手柄拖拽、画笔、橡皮擦、键盘快捷键。"""

    # 工具模式：像素编辑工具与自由变换工具始终互斥
    MODE_TRANSFORM = "transform"
    MODE_SELECT = MODE_TRANSFORM
    MODE_DRAW = "draw"
    MODE_ERASER = "eraser"
    WORKSPACE_RATIO = 1.30
    DEFAULT_DPI = 300.0

    def __init__(
        self,
        parent: tk.Widget,
        glyph_service: GlyphService,
        on_status_change: Optional[Callable[[], None]] = None,
        on_variant_change: Optional[Callable[[int], None]] = None,
        allowed_statuses: Optional[tuple[str, ...]] = None,
    ) -> None:
        super().__init__(parent, bg=theme.BG_CANVAS)
        self._glyph: GlyphService = glyph_service
        self._on_status_change: Optional[Callable[[], None]] = on_status_change
        self._on_variant_change: Optional[Callable[[int], None]] = on_variant_change
        self._allowed_statuses = set(allowed_statuses) if allowed_statuses else None

        # 状态
        self._current_char: str = ""
        self._current_variant_index: int = 0
        self._canvas_w: int = 250
        self._canvas_h: int = 250
        self._view_mode: str = "fit"
        self._view_scale: float = 1.0
        self._base_img: Optional[Image.Image] = None
        self._display_img: Optional[ImageTk.PhotoImage] = None
        self._workspace_buffer: Optional[Image.Image] = None
        self._display_buffer: Optional[Image.Image] = None
        self._last_transformed: Optional[Image.Image] = None
        self._last_transformed_origin: tuple[float, float] = (0.0, 0.0)
        self._last_transformed_bbox: Optional[tuple[int, int, int, int]] = None
        self._last_transform_signature: Optional[tuple] = None
        self._edit_img: Optional[Image.Image] = None  # 已按目标DPI换算并裁去外围透明区
        self._image_origin_x: float = 0.0
        self._image_origin_y: float = 0.0
        self._source_dpi_x: float = self.DEFAULT_DPI
        self._source_dpi_y: float = self.DEFAULT_DPI
        self._target_dpi: float = self.DEFAULT_DPI
        self._dirty: bool = False
        self._brush_changed: bool = False

        # 几何状态
        self._ox: int = 0
        self._oy: int = 0
        self._disp_w: int = 0
        self._disp_h: int = 0
        self._workspace_w: int = 1
        self._workspace_h: int = 1
        self._workspace_canvas_x: float = 0.0
        self._workspace_canvas_y: float = 0.0
        self._workspace_scale_x: float = 1.0
        self._workspace_scale_y: float = 1.0
        self._workspace_paste_x: int = 0
        self._workspace_paste_y: int = 0

        # 变换参数（缩放/旋转/偏移）
        self._tx_scale: float = 1.0
        self._tx_rotate: float = 0.0
        self._tx_offset_x: float = 0.0
        self._tx_offset_y: float = 0.0
        self._tx_stretch_w: float = 1.0
        self._tx_stretch_h: float = 1.0
        self._tx_distort: list[float] = [0.0] * 8

        # 模式
        self._mode: str = self.MODE_SELECT
        self._brush_size: int = 12
        self._brush_color: tuple[int, int, int, int] = (0, 0, 0, 255)
        self._stroke_color: tuple[int, int, int, int] = self._brush_color
        self._color_pick_gesture: bool = False
        self._alt_pressed: bool = False
        self._suppress_brush_until: float = 0.0
        self._last_brush_point: Optional[tuple[int, int]] = None
        self._pressure: float = 1.0
        self._brush_cursor_x: Optional[int] = None
        self._brush_cursor_y: Optional[int] = None
        self._brush_render_job: Optional[str] = None
        self._pending_brush_points: list[tuple[int, int]] = []
        self._brush_schedule_time: Optional[float] = None
        self._stroke_start_time: Optional[float] = None
        self._stroke_frame_count = 0
        self._stroke_point_count = 0
        self._glyph_canvas_item: Optional[int] = None
        self._brush_cursor_items: tuple[int, int, int] | None = None
        self._brush_size_var = tk.IntVar(value=self._brush_size)
        self._checkerboard_enabled: bool = False
        self._background_tk: Optional[ImageTk.PhotoImage] = None
        self._background_cache_key: Optional[tuple[int, int]] = None
        self._tool_buttons: dict[str, tk.Button] = {}
        self._brush_size_scale: Optional[tk.Scale] = None

        # 交互状态
        self._drag_active: bool = False
        self._drag_handle: str = ""  # 空=移动, nw/n/ne/e/se/s/sw/w=手柄, rotate=旋转
        self._drag_start_x: int = 0
        self._drag_start_y: int = 0
        self._drag_saved_params: dict = {}
        self._drag_start_bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
        self._drag_fixed_anchor: Optional[tuple[float, float]] = None
        self._transform_session_snapshot: Optional[dict] = None

        # 撤销栈
        self._undo_stack: list[dict] = []
        self._undo_index: int = -1

        # 手柄大小
        self._handle_size: int = 8

        self._build()

    # ==================== 构建 ====================

    def _build(self) -> None:
        """构建编辑画布。"""
        # 未选择字形时使用与容器相同的背景，视觉上保持透明
        self._canvas = tk.Canvas(self, bg=theme.BG_CANVAS, highlightthickness=0)
        self._canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        self._canvas.bind("<Button-1>", self._on_mouse_down)
        self._canvas.bind("<Motion>", self._on_mouse_motion)
        self._canvas.bind("<Leave>", self._on_mouse_leave)
        self._canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self._canvas.bind("<Control-Button-1>", self._on_ctrl_down)
        self._canvas.bind("<Control-B1-Motion>", self._on_ctrl_drag)
        self._canvas.bind("<MouseWheel>", self._on_brush_wheel)
        self._canvas.bind("<ButtonPress-3>", self._on_pick_color)
        self._canvas.bind("<B3-Motion>", self._on_pick_color)
        self._canvas.bind("<ButtonRelease-3>", self._on_color_pick_release)
        self._canvas.bind_all("<KeyPress-Alt_L>", self._on_alt_key_press, add="+")
        self._canvas.bind_all("<KeyPress-Alt_R>", self._on_alt_key_press, add="+")
        self._canvas.bind_all("<KeyRelease-Alt_L>", self._on_alt_key_release, add="+")
        self._canvas.bind_all("<KeyRelease-Alt_R>", self._on_alt_key_release, add="+")

        # Windows绘图板通常映射为鼠标事件；额外接收可用的笔压力虚拟事件。
        self._canvas.bind("<<TabletPressure>>", self._on_tablet_pressure)

        # 键盘快捷键
        self.bind_all("<Control-z>", self._on_undo)
        self.bind_all("<Control-Z>", self._on_undo)
        self.bind_all("<Control-y>", self._on_redo)
        self.bind_all("<Control-Y>", self._on_redo)
        self.bind_all("<Control-Shift-z>", self._on_redo)
        self.bind_all("<Control-Shift-Z>", self._on_redo)
        self.bind_all("<bracketleft>", self._on_brush_smaller)
        self.bind_all("<bracketright>", self._on_brush_larger)
        self._canvas.bind("<Return>", self._on_transform_confirm)
        self._canvas.bind("<KP_Enter>", self._on_transform_confirm)
        self._canvas.bind("<Escape>", self._on_transform_cancel)
        self._canvas.bind("<r>", self._on_revert)
        self._canvas.bind("<R>", self._on_revert)

        self._canvas.focus_set()

    def build_tool_panel(self, parent: tk.Widget) -> tk.Frame:
        """在指定父容器中构建画布工具面板。"""
        panel = tk.Frame(parent, bg=theme.BG_PANEL)

        transform_group = tk.LabelFrame(
            panel, text="自由变换", bg=theme.BG_PANEL, fg=theme.FG_PRIMARY,
            font=theme.FONT_NORMAL, padx=8, pady=8,
        )
        transform_group.pack(fill=tk.X, padx=10, pady=(10, 6))
        self._btn_select = theme.make_button(
            transform_group, "自由变换", accent=True,
            command=lambda: self._set_mode(self.MODE_TRANSFORM),
        )
        self._btn_select.pack(fill=tk.X)
        theme.make_label(
            transform_group,
            "自由变换说明",
            bg=theme.BG_PANEL,
            fg=theme.FG_PRIMARY,
            font=theme.FONT_BOLD,
        ).pack(anchor="w", pady=(9, 3))
        theme.make_label(
            transform_group,
            "拖动字形：移动\n"
            "拖动旋转手柄：旋转\n"
            "拖动四角手柄：等比缩放\n"
            "拖动四边手柄：拉伸、压缩\n"
            "Shift + 四边手柄：等比缩放\n"
            "Alt + 缩放手柄：从中心缩放\n"
            "Ctrl + 任意缩放手柄：自由扭曲",
            bg=theme.BG_PANEL,
            fg=theme.FG_SECONDARY,
            font=theme.FONT_SMALL,
            justify=tk.LEFT,
            wraplength=230,
        ).pack(anchor="w", fill=tk.X)

        pixel_group = tk.LabelFrame(
            panel, text="像素编辑", bg=theme.BG_PANEL, fg=theme.FG_PRIMARY,
            font=theme.FONT_NORMAL, padx=8, pady=8,
        )
        pixel_group.pack(fill=tk.X, padx=10, pady=6)
        for text, mode in (("画笔", self.MODE_DRAW), ("橡皮擦", self.MODE_ERASER)):
            button = theme.make_button(pixel_group, text, command=lambda value=mode: self._set_mode(value))
            button.pack(fill=tk.X, pady=2)
            self._tool_buttons[mode] = button
        self._btn_draw = self._tool_buttons[self.MODE_DRAW]
        self._btn_eraser = self._tool_buttons[self.MODE_ERASER]

        brush_size_row = tk.Frame(pixel_group, bg=theme.BG_PANEL)
        brush_size_row.pack(fill=tk.X, pady=(8, 0))
        tk.Label(
            brush_size_row, text="笔触大小", bg=theme.BG_PANEL,
            fg=theme.FG_SECONDARY, font=theme.FONT_SMALL,
        ).pack(side=tk.LEFT)
        self._brush_size_scale = tk.Scale(
            brush_size_row, from_=1, to=100, orient=tk.HORIZONTAL, showvalue=True,
            variable=self._brush_size_var, command=self._on_brush_size_change,
            bg=theme.BG_PANEL, fg=theme.FG_PRIMARY, troughcolor=theme.BG_INPUT,
            highlightthickness=0, bd=0,
        )
        self._brush_size_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))
        theme.make_label(
            pixel_group,
            "快捷键：[ 缩小笔触，] 放大笔触",
            bg=theme.BG_PANEL,
            fg=theme.FG_SECONDARY,
            font=theme.FONT_SMALL,
        ).pack(anchor=tk.W)

        view_group = tk.LabelFrame(
            panel, text="画布视图", bg=theme.BG_PANEL, fg=theme.FG_PRIMARY,
            font=theme.FONT_NORMAL, padx=8, pady=8,
        )
        view_group.pack(fill=tk.X, padx=10, pady=6)
        for text, command in (
            ("适应窗口", self._fit_to_window),
            ("1:1", self._zoom_actual),
            ("白底/棋盘格", self._toggle_bg),
        ):
            theme.make_button(view_group, text, command=command).pack(fill=tk.X, pady=2)

        self._update_mode_buttons()
        if self._edit_img is not None:
            self._update_global_ink_color()
        return panel

    def _update_mode_buttons(self) -> None:
        """刷新当前工具的选中状态。"""
        if not hasattr(self, "_btn_select"):
            return
        if self._brush_size_scale is not None:
            self._brush_size_scale.configure(
                state=tk.NORMAL if self._mode in (self.MODE_DRAW, self.MODE_ERASER) else tk.DISABLED
            )
        self._btn_select.configure(bg=theme.BTN_ACCENT if self._mode == self.MODE_TRANSFORM else theme.BTN_DEFAULT)
        for mode, button in self._tool_buttons.items():
            button.configure(bg=theme.BTN_ACCENT if self._mode == mode else theme.BTN_DEFAULT)

    # ==================== 工具模式 ====================

    def _set_mode(self, mode: str) -> None:
        """在像素编辑与自由变换之间互斥切换。"""
        if mode == self._mode:
            return
        if mode in (self.MODE_DRAW, self.MODE_ERASER):
            self._confirm_transform(bake=True)
        else:
            self._finish_pixel_edit()
            self._begin_transform_session()
        self._mode = mode
        self._update_mode_buttons()

        self._set_canvas_tool_cursor()
        if mode != self.MODE_TRANSFORM:
            pointer_x = self._canvas.winfo_pointerx() - self._canvas.winfo_rootx()
            pointer_y = self._canvas.winfo_pointery() - self._canvas.winfo_rooty()
            if 0 <= pointer_x < self._canvas.winfo_width() and 0 <= pointer_y < self._canvas.winfo_height():
                self._brush_cursor_x = pointer_x
                self._brush_cursor_y = pointer_y
                self._draw_brush_cursor(pointer_x, pointer_y)
        self._render()

    def _begin_transform_session(self) -> None:
        """记录本轮自由变换起点，供 Esc 完整取消。"""
        if self._transform_session_snapshot is None and self._edit_img is not None:
            self._transform_session_snapshot = self._snapshot()

    def _confirm_transform(self, bake: bool = False) -> None:
        """确认当前自由变换；进入像素工具时同步烘焙编辑图。"""
        if self._edit_img is None:
            return
        if bake:
            self._bake_transform_for_painting()
        self._transform_session_snapshot = None

    def _on_transform_confirm(self, _event: tk.Event) -> str:
        if self._mode == self.MODE_TRANSFORM:
            self._confirm_transform()
            self._begin_transform_session()
        return "break"

    def _on_transform_cancel(self, _event: tk.Event) -> str:
        if self._mode == self.MODE_TRANSFORM and self._transform_session_snapshot is not None:
            self._restore_snapshot(self._transform_session_snapshot)
            self._transform_session_snapshot = None
            self._begin_transform_session()
            self._render()
        return "break"

    # ==================== 加载字形 ====================

    def _get_variants(self, char: str) -> list[dict]:
        """返回当前流程可操作且已有自动优化预览的变体。"""
        variants = self._glyph.get_char_variants(char)
        if self._allowed_statuses is None:
            return variants
        return [
            variant
            for variant in variants
            if variant.get("状态") in self._allowed_statuses and variant.get("中间文件")
        ]

    def load_char(self, char: str, variant_index: int = 0) -> None:
        """加载指定汉字到画布。"""
        load_start = time.perf_counter()
        _PERFORMANCE_LOGGER.info(
            "[载图] 开始：文字=%s，字形序号=%d", char, variant_index + 1
        )

        stage_start = time.perf_counter()
        self._cancel_brush_render()
        self._last_brush_point = None
        self._brush_changed = False
        self._current_char = char
        self._current_variant_index = variant_index
        self._transform_session_snapshot = None
        _PERFORMANCE_LOGGER.info(
            "[载图] 清理上一字形状态：%.2f 毫秒",
            (time.perf_counter() - stage_start) * 1000,
        )

        stage_start = time.perf_counter()
        variants = self._get_variants(char)
        _PERFORMANCE_LOGGER.info(
            "[载图] 查询字形数据：%.2f 毫秒，数量=%d",
            (time.perf_counter() - stage_start) * 1000,
            len(variants),
        )
        if not variants or variant_index >= len(variants):
            self._clear_canvas()
            return

        variant = variants[variant_index]
        self._canvas.configure(bg="#ffffff")
        metadata = self._glyph._data.get("元数据", {})
        self._canvas_w = int(metadata.get("画布宽", 250))
        self._canvas_h = int(metadata.get("画布高", 250))
        self._target_dpi = self._valid_dpi(
            metadata.get("DPI", metadata.get("分辨率", self.DEFAULT_DPI))
        )

        # 加载变换参数
        params = variant.get("变换参数", {})
        self._tx_scale = params.get("缩放", 1.0)
        self._tx_rotate = params.get("旋转", 0.0)
        self._tx_offset_x = params.get("偏移X", 0)
        self._tx_offset_y = params.get("偏移Y", 0)
        self._tx_stretch_w = params.get("拉伸W", 1.0)
        self._tx_stretch_h = params.get("拉伸H", 1.0)
        self._tx_distort = list(params.get("扭曲", [0.0] * 8))
        if len(self._tx_distort) != 8:
            self._tx_distort = [0.0] * 8

        # 手工审核优先读取独立审核层；未编辑时使用自动优化预览。
        workflow_dirs = self._glyph.get_workflow_dirs()
        reviewed_name = variant.get("审核文件", "")
        intermediate_name = variant.get("中间文件", "")
        image_paths = []
        if reviewed_name:
            image_paths.append(os.path.join(workflow_dirs["手工审核"], reviewed_name))
        if intermediate_name:
            image_paths.append(os.path.join(workflow_dirs["优化预览"], intermediate_name))
        img = None
        loaded_review = False
        loaded_path = ""
        stage_start = time.perf_counter()
        for image_path in image_paths:
            exists_start = time.perf_counter()
            image_exists = os.path.exists(image_path)
            _PERFORMANCE_LOGGER.info(
                "[载图] 检查文件：%.2f 毫秒，存在=%s，路径=%s",
                (time.perf_counter() - exists_start) * 1000,
                "是" if image_exists else "否",
                image_path,
            )
            if image_exists:
                try:
                    decode_start = time.perf_counter()
                    with Image.open(image_path) as opened:
                        open_elapsed = (time.perf_counter() - decode_start) * 1000
                        convert_start = time.perf_counter()
                        img = opened.convert("RGBA")
                        convert_elapsed = (time.perf_counter() - convert_start) * 1000
                    _PERFORMANCE_LOGGER.info(
                        "[载图] 打开文件头：%.2f 毫秒；解码并转RGBA：%.2f 毫秒；尺寸=%s，格式=%s",
                        open_elapsed,
                        convert_elapsed,
                        img.size,
                        opened.format or "未知",
                    )
                    loaded_path = image_path
                    loaded_review = bool(
                        reviewed_name
                        and os.path.dirname(os.path.normcase(image_path))
                        == os.path.normcase(workflow_dirs["手工审核"])
                    )
                    break
                except (OSError, ValueError) as exc:
                    _PERFORMANCE_LOGGER.info(
                        "[载图] 文件读取失败：%s，路径=%s", exc, image_path
                    )
                    img = None
        _PERFORMANCE_LOGGER.info(
            "[载图] 文件定位与读取合计：%.2f 毫秒，来源=%s，路径=%s",
            (time.perf_counter() - stage_start) * 1000,
            "手工审核" if loaded_review else "优化预览",
            loaded_path or "未找到",
        )

        image_info = variant.get("图像信息", {})
        self._source_dpi_x = self._valid_dpi(
            image_info.get("水平DPI", self.DEFAULT_DPI)
        )
        self._source_dpi_y = self._valid_dpi(
            image_info.get("垂直DPI", self._source_dpi_x)
        )
        saved_origin = params.get("图像原点")
        stage_start = time.perf_counter()
        if loaded_review:
            if isinstance(saved_origin, (list, tuple)) and len(saved_origin) == 2:
                full_origin_x = float(saved_origin[0])
                full_origin_y = float(saved_origin[1])
            else:
                full_origin_x = (self._canvas_w - img.width) / 2.0
                full_origin_y = (self._canvas_h - img.height) / 2.0
            # 审核稿按保存画布读取后重新裁掉透明边，并补偿逻辑原点。
            # 编辑层因此只包含文字区域，控制框不会包住整张透明画布。
            alpha_mask = img.getchannel("A").point([0] * 16 + [255] * 240)
            glyph_bbox = alpha_mask.getbbox()
            if glyph_bbox is not None:
                prepared = img.crop(glyph_bbox)
                self._image_origin_x = full_origin_x + glyph_bbox[0]
                self._image_origin_y = full_origin_y + glyph_bbox[1]
            else:
                prepared = img
                self._image_origin_x = full_origin_x
                self._image_origin_y = full_origin_y
        else:
            prepared, self._image_origin_x, self._image_origin_y = self._prepare_source_image(
                img, self._source_dpi_x, self._source_dpi_y
            )
        _PERFORMANCE_LOGGER.info(
            "[载图] DPI归一化与透明边界裁剪：%.2f 毫秒，处理后尺寸=%s，逻辑原点=(%.2f, %.2f)",
            (time.perf_counter() - stage_start) * 1000,
            prepared.size if prepared is not None else "无",
            self._image_origin_x,
            self._image_origin_y,
        )

        stage_start = time.perf_counter()
        self._base_img = prepared
        self._edit_img = prepared.copy() if prepared else None
        # 切换字形后必须清空渲染缓存。仅用图像对象 id 作为签名时，
        # Python 可能复用上一张已释放图像的 id，导致画面和控制框仍引用旧字形。
        self._last_transformed = None
        self._last_transformed_origin = (0.0, 0.0)
        self._last_transformed_bbox = None
        self._last_transform_signature = None
        self._update_global_ink_color()
        self._dirty = False
        self._brush_changed = False
        self._reset_undo()
        if self._mode == self.MODE_TRANSFORM:
            self._begin_transform_session()
        _PERFORMANCE_LOGGER.info(
            "[载图] 编辑副本、墨色分析、撤销栈与缓存初始化：%.2f 毫秒",
            (time.perf_counter() - stage_start) * 1000,
        )

        # 通知面板和缩略条
        stage_start = time.perf_counter()
        if self._on_variant_change:
            self._on_variant_change(variant_index)
        _PERFORMANCE_LOGGER.info(
            "[载图] 更新工具面板与缩略条：%.2f 毫秒",
            (time.perf_counter() - stage_start) * 1000,
        )

        stage_start = time.perf_counter()
        self._render()
        _PERFORMANCE_LOGGER.info(
            "[载图] 首帧渲染与画布上传：%.2f 毫秒",
            (time.perf_counter() - stage_start) * 1000,
        )
        _PERFORMANCE_LOGGER.info(
            "[载图] 完成：文字=%s，字形序号=%d，总耗时=%.2f 毫秒",
            char,
            variant_index + 1,
            (time.perf_counter() - load_start) * 1000,
        )

    @classmethod
    def _valid_dpi(cls, value: object) -> float:
        """返回可用于物理尺寸换算的DPI。"""
        if not isinstance(value, (str, int, float)):
            return cls.DEFAULT_DPI
        try:
            dpi = float(value)
        except (TypeError, ValueError):
            return cls.DEFAULT_DPI
        return dpi if 1.0 <= dpi <= 9600.0 else cls.DEFAULT_DPI

    def _prepare_source_image(
        self,
        image: Optional[Image.Image],
        source_dpi_x: float,
        source_dpi_y: float,
    ) -> tuple[Optional[Image.Image], float, float]:
        """按源图毫米尺寸贴入目标DPI，并裁掉外围透明区但保留逻辑位置。"""
        if image is None:
            return None, 0.0, 0.0
        target_w = max(1, int(round(image.width * self._target_dpi / source_dpi_x)))
        target_h = max(1, int(round(image.height * self._target_dpi / source_dpi_y)))
        normalized = image.convert("RGBA")
        if normalized.size != (target_w, target_h):
            normalized = normalized.resize(
                (target_w, target_h), Image.Resampling.LANCZOS
            )

        alpha_mask = normalized.getchannel("A").point(
            [0] * 16 + [255] * 240
        )
        bbox = alpha_mask.getbbox()
        if bbox is not None:
            glyph_width = bbox[2] - bbox[0]
            glyph_height = bbox[3] - bbox[1]
            size_ratio = max(
                glyph_width / max(1, self._canvas_w),
                glyph_height / max(1, self._canvas_h),
            )
            if size_ratio > 1.0 or size_ratio < 0.6:
                size_scale = 0.95 / max(size_ratio, 1e-6)
                target_w = max(1, int(round(target_w * size_scale)))
                target_h = max(1, int(round(target_h * size_scale)))
                normalized = normalized.resize(
                    (target_w, target_h), Image.Resampling.LANCZOS
                )
                alpha_mask = normalized.getchannel("A").point(
                    [0] * 16 + [255] * 240
                )
                bbox = alpha_mask.getbbox()

        full_origin_x = (self._canvas_w - target_w) / 2.0
        full_origin_y = (self._canvas_h - target_h) / 2.0
        if bbox is None:
            return normalized, full_origin_x, full_origin_y
        cropped = normalized.crop(bbox)
        return cropped, full_origin_x + bbox[0], full_origin_y + bbox[1]

    def _clear_canvas(self) -> None:
        self._cancel_brush_render()
        self._canvas.delete("all")
        self._glyph_canvas_item = None
        self._brush_cursor_items = None
        self._workspace_buffer = None
        self._display_buffer = None
        self._canvas.configure(bg=theme.BG_CANVAS)
        self._base_img = None
        self._edit_img = None

    # ==================== 渲染 ====================

    def _render(self, draw_handles: bool = True) -> None:
        """渲染画布：背景 + 田字格 + 文字 + 手柄。"""
        render_start = time.perf_counter()
        stage_start = render_start
        self._canvas.delete("all")
        self._glyph_canvas_item = None
        self._brush_cursor_items = None
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        clear_elapsed = (time.perf_counter() - stage_start) * 1000
        if cw <= 1 or ch <= 1:
            return
        self._draw_canvas_background(cw, ch)
        if self._edit_img is None:
            return

        # 130%编辑工作区整体适应窗口；田字格始终位于其中心。
        workspace_w = self._canvas_w * self.WORKSPACE_RATIO
        workspace_h = self._canvas_h * self.WORKSPACE_RATIO
        fit_scale = min(cw / max(workspace_w, 1), ch / max(workspace_h, 1)) * 0.95
        self._view_scale = fit_scale if self._view_mode == "fit" else 1.0

        self._disp_w = max(1, int(round(self._canvas_w * self._view_scale)))
        self._disp_h = max(1, int(round(self._canvas_h * self._view_scale)))
        self._ox = int(round((cw - self._disp_w) / 2.0))
        self._oy = int(round((ch - self._disp_h) / 2.0))

        # 田字格
        self._draw_grid()

        # 文字图像（应用变换参数居中显示）
        transform_elapsed = 0.0
        compose_elapsed = 0.0
        photo_elapsed = 0.0
        upload_elapsed = 0.0
        decoration_elapsed = 0.0
        if self._edit_img:
            signature = (
                id(self._edit_img),
                self._tx_scale,
                self._tx_stretch_w,
                self._tx_stretch_h,
                self._tx_rotate,
                tuple(self._tx_distort),
            )
            stage_start = time.perf_counter()
            if self._last_transformed is None or signature != self._last_transform_signature:
                self._last_transformed = self._transformed_glyph(self._edit_img)
                self._last_transformed_bbox = self._last_transformed.getchannel("A").getbbox()
                self._last_transform_signature = signature
            self._last_transformed_origin = self._transformed_origin(self._last_transformed)
            transform_elapsed = (time.perf_counter() - stage_start) * 1000

            stage_start = time.perf_counter()
            disp_img = self._apply_transform(
                self._edit_img,
                self._disp_w,
                self._disp_h,
                transformed=self._last_transformed,
            )
            compose_elapsed = (time.perf_counter() - stage_start) * 1000
            self._workspace_buffer = None
            self._display_buffer = disp_img

            stage_start = time.perf_counter()
            self._display_img = ImageTk.PhotoImage(disp_img, master=self._canvas)
            photo_elapsed = (time.perf_counter() - stage_start) * 1000
            workspace_margin_x = (disp_img.width - self._disp_w) / 2.0
            workspace_margin_y = (disp_img.height - self._disp_h) / 2.0
            self._workspace_canvas_x = self._ox - workspace_margin_x
            self._workspace_canvas_y = self._oy - workspace_margin_y
            self._workspace_scale_x = disp_img.width / max(1, self._workspace_w)
            self._workspace_scale_y = disp_img.height / max(1, self._workspace_h)

            stage_start = time.perf_counter()
            self._glyph_canvas_item = self._canvas.create_image(
                self._workspace_canvas_x,
                self._workspace_canvas_y,
                anchor=tk.NW,
                image=self._display_img,
                tags=("glyph",),
            )
            upload_elapsed = (time.perf_counter() - stage_start) * 1000

        # 手柄或画笔类工具的圆形笔尖预览
        stage_start = time.perf_counter()
        if draw_handles and self._mode == self.MODE_SELECT:
            self._draw_handles()
        elif self._brush_cursor_x is not None and self._brush_cursor_y is not None:
            self._draw_brush_cursor(self._brush_cursor_x, self._brush_cursor_y)
        decoration_elapsed = (time.perf_counter() - stage_start) * 1000
        _PERFORMANCE_LOGGER.info(
            "[完整渲染] 模式=%s，画布=%dx%d，工作区=%dx%d，显示图=%s，清空与尺寸=%.2f 毫秒，字形变换=%.2f 毫秒，工作区合成与缩放=%.2f 毫秒，创建PhotoImage=%.2f 毫秒，Tk画布挂载=%.2f 毫秒，网格与控件=%.2f 毫秒，总计=%.2f 毫秒",
            self._mode,
            cw,
            ch,
            self._workspace_w,
            self._workspace_h,
            self._display_buffer.size if self._display_buffer is not None else "无",
            clear_elapsed,
            transform_elapsed,
            compose_elapsed,
            photo_elapsed,
            upload_elapsed,
            decoration_elapsed,
            (time.perf_counter() - render_start) * 1000,
        )

    def _prepare_pixel_display(self) -> bool:
        """建立像素工具使用的持久工作区和显示缓冲。"""
        prepare_start = time.perf_counter()
        if self._edit_img is None or self._glyph_canvas_item is None:
            _PERFORMANCE_LOGGER.info("[像素显示准备] 缺少编辑图或画布图元，无法建立缓冲")
            return False
        workspace_w = max(self._canvas_w, int(math.ceil(self._canvas_w * self.WORKSPACE_RATIO)))
        workspace_h = max(self._canvas_h, int(math.ceil(self._canvas_h * self.WORKSPACE_RATIO)))
        margin_x = (workspace_w - self._canvas_w) / 2.0
        margin_y = (workspace_h - self._canvas_h) / 2.0
        origin_x, origin_y = self._transformed_origin(self._edit_img)
        self._workspace_w = workspace_w
        self._workspace_h = workspace_h
        self._workspace_paste_x = self._round_pixel(margin_x + origin_x)
        self._workspace_paste_y = self._round_pixel(margin_y + origin_y)

        stage_start = time.perf_counter()
        workspace = Image.new("RGBA", (workspace_w, workspace_h), (0, 0, 0, 0))
        workspace.alpha_composite(
            self._edit_img,
            (self._workspace_paste_x, self._workspace_paste_y),
        )
        workspace_elapsed = (time.perf_counter() - stage_start) * 1000

        display_w = max(1, int(round(workspace_w * self._view_scale)))
        display_h = max(1, int(round(workspace_h * self._view_scale)))
        stage_start = time.perf_counter()
        display = workspace
        if workspace.size != (display_w, display_h):
            display = workspace.resize((display_w, display_h), Image.Resampling.BILINEAR)
        scale_elapsed = (time.perf_counter() - stage_start) * 1000
        self._workspace_buffer = workspace
        self._display_buffer = display

        stage_start = time.perf_counter()
        self._display_img = ImageTk.PhotoImage(display, master=self._canvas)
        photo_elapsed = (time.perf_counter() - stage_start) * 1000
        workspace_margin_x = (display_w - self._disp_w) / 2.0
        workspace_margin_y = (display_h - self._disp_h) / 2.0
        self._workspace_canvas_x = self._ox - workspace_margin_x
        self._workspace_canvas_y = self._oy - workspace_margin_y
        self._workspace_scale_x = display_w / max(1, workspace_w)
        self._workspace_scale_y = display_h / max(1, workspace_h)
        stage_start = time.perf_counter()
        try:
            self._canvas.coords(
                self._glyph_canvas_item,
                self._workspace_canvas_x,
                self._workspace_canvas_y,
            )
            self._canvas.itemconfigure(self._glyph_canvas_item, image=self._display_img)
        except tk.TclError:
            _PERFORMANCE_LOGGER.exception("[像素显示准备] Tk画布更新失败")
            return False
        upload_elapsed = (time.perf_counter() - stage_start) * 1000
        _PERFORMANCE_LOGGER.info(
            "[像素显示准备] 工作区=%dx%d，显示区=%dx%d，工作区合成=%.2f 毫秒，交互快速缩放=%.2f 毫秒，创建PhotoImage=%.2f 毫秒，Tk画布更新=%.2f 毫秒，总计=%.2f 毫秒",
            workspace_w,
            workspace_h,
            display_w,
            display_h,
            workspace_elapsed,
            scale_elapsed,
            photo_elapsed,
            upload_elapsed,
            (time.perf_counter() - prepare_start) * 1000,
        )
        return True

    def _render_pixel_edit(
        self,
        dirty_region: Optional[tuple[int, int, int, int]] = None,
    ) -> None:
        """仅重采样并上传本帧发生变化的局部像素。"""
        render_start = time.perf_counter()
        if self._edit_img is None or self._glyph_canvas_item is None:
            fallback_start = time.perf_counter()
            self._render()
            _PERFORMANCE_LOGGER.info(
                "[画笔帧] 缺少显示缓存，回退完整渲染：%.2f 毫秒",
                (time.perf_counter() - fallback_start) * 1000,
            )
            return
        self._last_transformed = self._edit_img
        self._last_transformed_origin = self._transformed_origin(self._edit_img)
        self._last_transformed_bbox = None
        self._last_transform_signature = (
            id(self._edit_img),
            self._tx_scale,
            self._tx_stretch_w,
            self._tx_stretch_h,
            self._tx_rotate,
            tuple(self._tx_distort),
        )
        if (
            dirty_region is None
            or self._workspace_buffer is None
            or self._display_buffer is None
            or self._display_img is None
        ):
            prepare_start = time.perf_counter()
            prepared = self._prepare_pixel_display()
            prepare_elapsed = (time.perf_counter() - prepare_start) * 1000
            fallback_elapsed = 0.0
            if not prepared:
                fallback_start = time.perf_counter()
                self._render()
                fallback_elapsed = (time.perf_counter() - fallback_start) * 1000
            _PERFORMANCE_LOGGER.info(
                "[画笔帧] 重建完整像素显示：缓冲准备=%.2f 毫秒，回退完整渲染=%.2f 毫秒，总计=%.2f 毫秒",
                prepare_elapsed,
                fallback_elapsed,
                (time.perf_counter() - render_start) * 1000,
            )
            return

        x1, y1, x2, y2 = dirty_region
        x1 = max(0, min(self._edit_img.width, x1))
        y1 = max(0, min(self._edit_img.height, y1))
        x2 = max(x1, min(self._edit_img.width, x2))
        y2 = max(y1, min(self._edit_img.height, y2))
        if x2 <= x1 or y2 <= y1:
            return

        wx1 = self._workspace_paste_x + x1
        wy1 = self._workspace_paste_y + y1
        wx2 = self._workspace_paste_x + x2
        wy2 = self._workspace_paste_y + y2
        stage_start = time.perf_counter()
        self._workspace_buffer.paste(self._edit_img.crop((x1, y1, x2, y2)), (wx1, wy1))
        workspace_elapsed = (time.perf_counter() - stage_start) * 1000

        stage_start = time.perf_counter()
        filter_pad = 4
        sx1 = max(0, wx1 - filter_pad)
        sy1 = max(0, wy1 - filter_pad)
        sx2 = min(self._workspace_w, wx2 + filter_pad)
        sy2 = min(self._workspace_h, wy2 + filter_pad)
        scale_x = self._workspace_scale_x
        scale_y = self._workspace_scale_y
        dx1 = max(0, int(math.floor(sx1 * scale_x)))
        dy1 = max(0, int(math.floor(sy1 * scale_y)))
        dx2 = min(self._display_buffer.width, int(math.ceil(sx2 * scale_x)))
        dy2 = min(self._display_buffer.height, int(math.ceil(sy2 * scale_y)))
        if dx2 <= dx1 or dy2 <= dy1:
            return
        source_box = (
            max(0, int(math.floor(dx1 / scale_x)) - filter_pad),
            max(0, int(math.floor(dy1 / scale_y)) - filter_pad),
            min(self._workspace_w, int(math.ceil(dx2 / scale_x)) + filter_pad),
            min(self._workspace_h, int(math.ceil(dy2 / scale_y)) + filter_pad),
        )
        patch = self._workspace_buffer.crop(source_box)
        patch_size = (
            max(1, int(round(patch.width * scale_x))),
            max(1, int(round(patch.height * scale_y))),
        )
        if patch.size != patch_size:
            patch = patch.resize(patch_size, Image.Resampling.LANCZOS)
        crop_x = max(0, dx1 - int(round(source_box[0] * scale_x)))
        crop_y = max(0, dy1 - int(round(source_box[1] * scale_y)))
        patch = patch.crop((crop_x, crop_y, crop_x + dx2 - dx1, crop_y + dy2 - dy1))
        self._display_buffer.paste(patch, (dx1, dy1))
        scale_elapsed = (time.perf_counter() - stage_start) * 1000

        photo_elapsed = 0.0
        upload_elapsed = 0.0
        fallback_elapsed = 0.0
        try:
            photo_start = time.perf_counter()
            patch_image = ImageTk.PhotoImage(patch, master=self._canvas)
            photo_elapsed = (time.perf_counter() - photo_start) * 1000
            upload_start = time.perf_counter()
            self._canvas.tk.call(
                str(self._display_img),
                "copy",
                str(patch_image),
                "-to",
                dx1,
                dy1,
                "-compositingrule",
                "set",
            )
            upload_elapsed = (time.perf_counter() - upload_start) * 1000
        except tk.TclError:
            fallback_start = time.perf_counter()
            if not self._prepare_pixel_display():
                self._render()
            fallback_elapsed = (time.perf_counter() - fallback_start) * 1000
        _PERFORMANCE_LOGGER.info(
            "[画笔帧] 局部显示：脏区=%dx%d，显示补丁=%dx%d，工作区复制=%.2f 毫秒，局部缩放=%.2f 毫秒，创建PhotoImage=%.2f 毫秒，Tk上传=%.2f 毫秒，异常回退=%.2f 毫秒，总计=%.2f 毫秒",
            x2 - x1,
            y2 - y1,
            patch.width,
            patch.height,
            workspace_elapsed,
            scale_elapsed,
            photo_elapsed,
            upload_elapsed,
            fallback_elapsed,
            (time.perf_counter() - render_start) * 1000,
        )

    def _transformed_glyph(self, img: Optional[Image.Image] = None) -> Image.Image:
        """返回完成缩放、扭曲和旋转后的字形，不限制在田字格内。"""
        source = img or self._edit_img
        if source is None:
            return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        if (
            abs(self._tx_scale - 1.0) <= 1e-9
            and abs(self._tx_stretch_w - 1.0) <= 1e-9
            and abs(self._tx_stretch_h - 1.0) <= 1e-9
            and abs(self._tx_rotate) <= 0.01
            and not any(abs(value) > 0.01 for value in self._tx_distort)
        ):
            return source
        nw = max(1, int(round(source.width * self._tx_scale * self._tx_stretch_w)))
        nh = max(1, int(round(source.height * self._tx_scale * self._tx_stretch_h)))
        transformed = source.resize((nw, nh), Image.Resampling.LANCZOS)
        if any(abs(value) > 0.01 for value in self._tx_distort):
            transformed = self._warp_distort(transformed, self._tx_distort)
        if abs(self._tx_rotate) > 0.01:
            transformed = transformed.rotate(
                -self._tx_rotate,
                expand=True,
                resample=Image.Resampling.BICUBIC,
            )
        return transformed

    @staticmethod
    def _warp_distort(
        image: Image.Image,
        offsets: list[float],
    ) -> Image.Image:
        """按四角位移执行透视扭曲，并扩展图像以完整保留结果。"""
        try:
            import cv2
        except ImportError:
            return image
        width, height = image.size
        source = np.asarray(
            [[0.0, 0.0], [width - 1.0, 0.0],
             [width - 1.0, height - 1.0], [0.0, height - 1.0]],
            dtype=np.float32,
        )
        target = source + np.asarray(
            [[offsets[0], offsets[1]], [offsets[2], offsets[3]],
             [offsets[4], offsets[5]], [offsets[6], offsets[7]]],
            dtype=np.float32,
        )
        min_x = min(0.0, float(target[:, 0].min()))
        min_y = min(0.0, float(target[:, 1].min()))
        target[:, 0] -= min_x
        target[:, 1] -= min_y
        out_w = max(1, int(math.ceil(float(target[:, 0].max()))) + 1)
        out_h = max(1, int(math.ceil(float(target[:, 1].max()))) + 1)
        matrix = cv2.getPerspectiveTransform(source, target)
        warped = cv2.warpPerspective(
            np.asarray(image),
            matrix,
            (out_w, out_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        return Image.fromarray(warped, "RGBA")

    def _transformed_origin(self, transformed: Image.Image) -> tuple[float, float]:
        """返回变换后字形左上角在田字格逻辑坐标中的位置。"""
        source_center_x = self._image_origin_x + self._edit_img.width / 2.0 if self._edit_img else 0.0
        source_center_y = self._image_origin_y + self._edit_img.height / 2.0 if self._edit_img else 0.0
        return (
            source_center_x - transformed.width / 2.0 + self._tx_offset_x,
            source_center_y - transformed.height / 2.0 + self._tx_offset_y,
        )

    def _glyph_screen_bbox(self) -> tuple[float, float, float, float]:
        """返回当前有效文字变换后的屏幕包围盒。"""
        transformed = self._last_transformed or self._transformed_glyph()
        bbox = self._last_transformed_bbox or transformed.getchannel("A").getbbox()
        bbox = bbox or (0, 0, transformed.width, transformed.height)
        origin_x, origin_y = (
            self._last_transformed_origin
            if self._last_transformed is not None
            else self._transformed_origin(transformed)
        )
        return (
            self._ox + (origin_x + bbox[0]) * self._view_scale,
            self._oy + (origin_y + bbox[1]) * self._view_scale,
            self._ox + (origin_x + bbox[2]) * self._view_scale,
            self._oy + (origin_y + bbox[3]) * self._view_scale,
        )

    @staticmethod
    def _round_pixel(value: float) -> int:
        """将逻辑坐标稳定量化到像素，保证整数平移前后结果一致。"""
        return int(math.floor(value + 0.5))

    def _apply_transform(
        self,
        img: Image.Image,
        target_w: int,
        target_h: int,
        transformed: Optional[Image.Image] = None,
    ) -> Image.Image:
        """将完整变换字形绘入不裁切的130%显示工作区。"""
        transformed = transformed or self._transformed_glyph(img)
        workspace_w = max(self._canvas_w, int(math.ceil(self._canvas_w * self.WORKSPACE_RATIO)))
        workspace_h = max(self._canvas_h, int(math.ceil(self._canvas_h * self.WORKSPACE_RATIO)))
        self._workspace_w = workspace_w
        self._workspace_h = workspace_h
        workspace = Image.new("RGBA", (workspace_w, workspace_h), (0, 0, 0, 0))
        margin_x = (workspace_w - self._canvas_w) / 2.0
        margin_y = (workspace_h - self._canvas_h) / 2.0
        origin_x, origin_y = self._transformed_origin(transformed)
        paste_x = self._round_pixel(margin_x + origin_x)
        paste_y = self._round_pixel(margin_y + origin_y)
        self._workspace_paste_x = paste_x
        self._workspace_paste_y = paste_y
        workspace.alpha_composite(transformed, (paste_x, paste_y))
        display_w = max(1, int(round(workspace_w * self._view_scale)))
        display_h = max(1, int(round(workspace_h * self._view_scale)))
        if workspace.size != (display_w, display_h):
            workspace = workspace.resize(
                (display_w, display_h),
                Image.Resampling.LANCZOS,
            )
        return workspace

    def _draw_canvas_background(self, width: int, height: int) -> None:
        """绘制白底或表示透明区域的棋盘格。"""
        if not self._checkerboard_enabled:
            self._canvas.create_rectangle(
                0, 0, width, height,
                fill="#ffffff", outline="", tags=("canvas_background",),
            )
            return
        cache_key = (width, height)
        if self._background_tk is None or self._background_cache_key != cache_key:
            tile_size = 16
            pattern = Image.new("RGB", (width, height), "#ffffff")
            drawer = ImageDraw.Draw(pattern)
            for row, y in enumerate(range(0, height, tile_size)):
                for column, x in enumerate(range(0, width, tile_size)):
                    if (row + column) % 2:
                        drawer.rectangle(
                            (x, y, min(x + tile_size - 1, width - 1),
                             min(y + tile_size - 1, height - 1)),
                            fill="#d8d8d8",
                        )
            self._background_tk = ImageTk.PhotoImage(pattern)
            self._background_cache_key = cache_key
        self._canvas.create_image(
            0, 0, anchor="nw", image=self._background_tk,
            tags=("canvas_background",),
        )

    def _draw_grid(self) -> None:
        """绘制130%编辑范围框和田字格。"""
        ox, oy = self._ox, self._oy
        w, h = self._disp_w, self._disp_h
        margin_x = w * (self.WORKSPACE_RATIO - 1.0) / 2.0
        margin_y = h * (self.WORKSPACE_RATIO - 1.0) / 2.0
        self._canvas.create_rectangle(
            ox - margin_x,
            oy - margin_y,
            ox + w + margin_x,
            oy + h + margin_y,
            outline="#b86f6f",
            width=1,
            tags=("workspace_boundary",),
        )
        color = "#3a3a50"

        self._canvas.create_rectangle(ox, oy, ox + w, oy + h, outline=color, width=1)
        self._canvas.create_line(ox + w // 2, oy, ox + w // 2, oy + h, fill=color, width=1)
        self._canvas.create_line(ox, oy + h // 2, ox + w, oy + h // 2, fill=color, width=1)
        self._canvas.create_line(ox, oy, ox + w, oy + h, fill=color, width=1)
        self._canvas.create_line(ox + w, oy, ox, oy + h, fill=color, width=1)

    def _control_screen_corners(self) -> dict[str, tuple[float, float]]:
        """返回随缩放、旋转和透视扭曲同步变化的四个控制角点。"""
        if self._edit_img is None:
            return {name: (0.0, 0.0) for name in ("nw", "ne", "se", "sw")}
        width = max(1, int(round(self._edit_img.width * self._tx_scale * self._tx_stretch_w)))
        height = max(1, int(round(self._edit_img.height * self._tx_scale * self._tx_stretch_h)))
        offsets = self._tx_distort
        points = [
            [offsets[0], offsets[1]],
            [width - 1.0 + offsets[2], offsets[3]],
            [width - 1.0 + offsets[4], height - 1.0 + offsets[5]],
            [offsets[6], height - 1.0 + offsets[7]],
        ]
        min_x = min(0.0, *(point[0] for point in points))
        min_y = min(0.0, *(point[1] for point in points))
        points = [[point[0] - min_x, point[1] - min_y] for point in points]
        warped_w = max(1.0, max(point[0] for point in points) + 1.0)
        warped_h = max(1.0, max(point[1] for point in points) + 1.0)

        # Pillow 的正角度在屏幕上逆时针旋转；控制框使用屏幕坐标矩阵时
        # 必须采用相反的数学符号，才能与实际图像保持同向。
        angle = math.radians(self._tx_rotate)
        if abs(angle) > 1e-8:
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            cx = warped_w / 2.0
            cy = warped_h / 2.0
            source_corners = [(0.0, 0.0), (warped_w, 0.0), (warped_w, warped_h), (0.0, warped_h)]

            def rotate(point: tuple[float, float] | list[float]) -> tuple[float, float]:
                px, py = point
                return (
                    (px - cx) * cos_a - (py - cy) * sin_a,
                    (px - cx) * sin_a + (py - cy) * cos_a,
                )

            rotated_bounds = [rotate(point) for point in source_corners]
            rotate_min_x = min(point[0] for point in rotated_bounds)
            rotate_min_y = min(point[1] for point in rotated_bounds)
            points = [
                [rotated[0] - rotate_min_x, rotated[1] - rotate_min_y]
                for rotated in (rotate(point) for point in points)
            ]

        transformed = self._last_transformed or self._transformed_glyph()
        origin_x, origin_y = (
            self._last_transformed_origin
            if self._last_transformed is not None
            else self._transformed_origin(transformed)
        )
        names = ("nw", "ne", "se", "sw")
        return {
            name: (
                self._ox + (origin_x + point[0]) * self._view_scale,
                self._oy + (origin_y + point[1]) * self._view_scale,
            )
            for name, point in zip(names, points)
        }

    def _draw_handles(self) -> None:
        """绘制随自由变换同步变化的四边形控制框及8个手柄。"""
        corners = self._control_screen_corners()
        nw, ne, se, sw = (corners[name] for name in ("nw", "ne", "se", "sw"))
        mid = lambda first, second: ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)
        points = {
            "nw": nw,
            "n": mid(nw, ne),
            "ne": ne,
            "e": mid(ne, se),
            "se": se,
            "s": mid(se, sw),
            "sw": sw,
            "w": mid(sw, nw),
        }
        self._canvas.create_polygon(
            *(coordinate for name in ("nw", "ne", "se", "sw") for coordinate in corners[name]),
            fill="",
            outline="#5a7fbf",
            dash=(4, 3),
            tags=("bounds",),
        )
        hs = self._handle_size
        for tag, (px, py) in points.items():
            self._canvas.create_rectangle(
                px - hs, py - hs, px + hs, py + hs,
                fill="#5a7fbf", outline=theme.FG_PRIMARY, tags=("handle", tag),
            )

        top_mid = points["n"]
        center = mid(mid(nw, se), mid(ne, sw))
        vx = top_mid[0] - center[0]
        vy = top_mid[1] - center[1]
        length = max(1.0, math.hypot(vx, vy))
        rotate_x = top_mid[0] + vx / length * hs * 2.0
        rotate_y = top_mid[1] + vy / length * hs * 2.0
        self._canvas.create_line(top_mid[0], top_mid[1], rotate_x, rotate_y, fill="#5a7fbf")
        self._canvas.create_oval(
            rotate_x - hs,
            rotate_y - hs,
            rotate_x + hs,
            rotate_y + hs,
            fill="#5a7fbf",
            outline=theme.FG_PRIMARY,
            tags=("handle", "rotate"),
        )

    # ==================== 鼠标事件 ====================

    def _get_handle_at(self, x: int, y: int) -> str:
        """检测鼠标位置的手柄标签，忽略 Tk 动态附加的 current 标签。"""
        handle_names = {"nw", "n", "ne", "e", "se", "s", "sw", "w", "rotate"}
        items = self._canvas.find_overlapping(x - 2, y - 2, x + 2, y + 2)
        for item_id in reversed(items):
            tags = self._canvas.gettags(item_id)
            if "handle" not in tags:
                continue
            for tag in tags:
                if tag in handle_names:
                    return tag
        return ""

    def _on_mouse_motion(self, event: tk.Event) -> None:
        """根据鼠标或绘图板位置实时更新手柄光标或圆形笔尖。"""
        if self._mode != self.MODE_TRANSFORM:
            self._brush_cursor_x = int(event.x)
            self._brush_cursor_y = int(event.y)
            self._draw_brush_cursor(int(event.x), int(event.y))
            return
        if self._drag_active:
            return
        handle = self._get_handle_at(event.x, event.y)
        ctrl_pressed = bool(int(event.state) & 0x0004)
        cursors = {
            "nw": "bottom_right_corner",
            "se": "bottom_right_corner",
            "ne": "bottom_left_corner",
            "sw": "bottom_left_corner",
            "n": "sb_v_double_arrow",
            "s": "sb_v_double_arrow",
            "e": "sb_h_double_arrow",
            "w": "sb_h_double_arrow",
            "rotate": "exchange",
        }
        if handle:
            if ctrl_pressed and handle != "rotate":
                cursor = "sizing" if handle in ("n", "e", "s", "w") else "target"
            else:
                cursor = cursors.get(handle, "crosshair")
            self._canvas.configure(cursor=cursor)
        elif self._edit_img is not None and self._is_inside_glyph(event.x, event.y):
            self._canvas.configure(cursor="fleur")
        else:
            self._canvas.configure(cursor="")

    def _on_mouse_leave(self, _event: tk.Event) -> None:
        """鼠标离开画布后恢复当前工具的默认光标。"""
        self._brush_cursor_x = None
        self._brush_cursor_y = None
        if self._brush_cursor_items is not None:
            for item in self._brush_cursor_items:
                self._canvas.itemconfigure(item, state=tk.HIDDEN)
        if not self._drag_active:
            self._set_canvas_tool_cursor()

    def _draw_brush_cursor(self, x: int, y: int) -> None:
        """移动持久笔尖图元，避免高频输入时反复创建Canvas对象。"""
        if self._mode == self.MODE_SELECT:
            return
        image_radius = max(
            1,
            int(round(self._brush_size * (0.45 + 0.55 * self._pressure) / 2.0)),
        )
        radius = max(
            3.0,
            (image_radius + 0.5) * (self._workspace_scale_x + self._workspace_scale_y) * 0.5,
        )
        accent = {
            self.MODE_DRAW: "#35C759",
            self.MODE_ERASER: "#FF453A",
        }.get(self._mode, "#FFFFFF")
        if self._brush_cursor_items is None:
            outer = self._canvas.create_oval(
                0, 0, 0, 0, outline="#111111", width=3,
                tags=("brush_cursor",),
            )
            inner = self._canvas.create_oval(
                0, 0, 0, 0, outline=accent, width=1,
                tags=("brush_cursor",),
            )
            center = self._canvas.create_oval(
                0, 0, 0, 0, fill=accent, outline="#111111",
                tags=("brush_cursor",),
            )
            self._brush_cursor_items = (outer, inner, center)
        outer, inner, center = self._brush_cursor_items
        bounds = (x - radius, y - radius, x + radius, y + radius)
        self._canvas.coords(outer, *bounds)
        self._canvas.coords(inner, *bounds)
        self._canvas.coords(center, x - 1.5, y - 1.5, x + 1.5, y + 1.5)
        self._canvas.itemconfigure(outer, state=tk.NORMAL)
        self._canvas.itemconfigure(inner, outline=accent, state=tk.NORMAL)
        self._canvas.itemconfigure(center, fill=accent, state=tk.NORMAL)
        self._canvas.tag_raise("brush_cursor")

    def _set_canvas_tool_cursor(self) -> None:
        """恢复当前工具模式对应的默认光标。"""
        cursor = "" if self._mode == self.MODE_SELECT else "none"
        self._canvas.configure(cursor=cursor)
        if self._mode == self.MODE_SELECT and self._brush_cursor_items is not None:
            for item in self._brush_cursor_items:
                self._canvas.itemconfigure(item, state=tk.HIDDEN)

    def _is_inside_canvas(self, x: int, y: int) -> bool:
        """编辑命中范围为田字格130%的逻辑工作区。"""
        margin_x = self._disp_w * (self.WORKSPACE_RATIO - 1.0) / 2.0
        margin_y = self._disp_h * (self.WORKSPACE_RATIO - 1.0) / 2.0
        return (
            self._ox - margin_x <= x <= self._ox + self._disp_w + margin_x
            and self._oy - margin_y <= y <= self._oy + self._disp_h + margin_y
        )

    def _is_inside_glyph(self, x: int, y: int) -> bool:
        x1, y1, x2, y2 = self._glyph_screen_bbox()
        return x1 <= x <= x2 and y1 <= y <= y2

    @staticmethod
    def _is_alt_pressed(event: tk.Event) -> bool:
        """兼容 Windows 鼠标与绘图板事件中的 Alt 状态位。"""
        state = int(event.state)
        if bool(state & 0x0008) or bool(state & 0x20000):
            return True
        try:
            return bool(ctypes.windll.user32.GetAsyncKeyState(0x12) & 0x8000)
        except (AttributeError, OSError):
            return False

    def _on_mouse_down(self, event: tk.Event) -> Optional[str]:
        """选择模式：检测手柄 / 拖拽移动。"""
        if self._mode != self.MODE_SELECT:
            if self._alt_pressed or self._is_alt_pressed(event):
                return self._start_color_pick(event)
            if time.monotonic() < self._suppress_brush_until:
                return "break"
            self._color_pick_gesture = False
            self._drag_active = True
            self._start_brush(event)
            return None

        handle = self._get_handle_at(event.x, event.y)
        if handle:
            self._drag_active = True
            self._drag_handle = handle
            self._drag_start_x = event.x
            self._drag_start_y = event.y
            self._drag_start_bbox = self._glyph_screen_bbox()
            self._save_current_params()
        elif self._is_inside_glyph(event.x, event.y):
            self._drag_active = True
            self._drag_handle = ""
            self._drag_start_x = event.x
            self._drag_start_y = event.y
            self._drag_start_bbox = self._glyph_screen_bbox()
            self._save_current_params()

    def _on_mouse_drag(self, event: tk.Event) -> Optional[str]:
        """处理自由变换拖动或绘图板笔迹移动。"""
        if self._mode != self.MODE_TRANSFORM:
            self._brush_cursor_x = int(event.x)
            self._brush_cursor_y = int(event.y)
            self._draw_brush_cursor(int(event.x), int(event.y))
            if self._color_pick_gesture or self._alt_pressed or self._is_alt_pressed(event):
                return self._start_color_pick(event)
            if time.monotonic() < self._suppress_brush_until:
                return "break"
            if self._drag_active:
                self._continue_brush(event)
            return None
        if not self._drag_active:
            return

        dx = event.x - self._drag_start_x
        dy = event.y - self._drag_start_y
        event_state = int(event.state)
        shift_pressed = bool(event_state & 0x0001)
        alt_pressed = bool(event_state & 0x0008) or bool(event_state & 0x20000)

        if self._drag_handle == "":
            if shift_pressed:
                if abs(dx) >= abs(dy):
                    dy = 0
                else:
                    dx = 0
            self._adjust_offset(dx, dy)
        elif self._drag_handle == "rotate":
            # 旋转中心在本次拖动期间保持不变；Shift 按15度吸附。
            x1, y1, x2, y2 = self._drag_start_bbox
            cx = int(round((x1 + x2) / 2.0))
            cy = int(round((y1 + y2) / 2.0))
            old_angle = self._angle_from_center(cx, cy, self._drag_start_x, self._drag_start_y)
            new_angle = self._angle_from_center(cx, cy, event.x, event.y)
            # 屏幕坐标Y轴向下，鼠标角增量本身就是视觉旋转方向；
            # 渲染层会转换为Pillow所需的相反角度。
            angle = self._drag_saved_params.get("旋转", 0.0) + (new_angle - old_angle)
            self._tx_rotate = round(angle / 15.0) * 15.0 if shift_pressed else angle
        else:
            self._handle_scale(
                self._drag_handle,
                dx,
                dy,
                keep_aspect=shift_pressed,
                from_center=alt_pressed,
            )

        self._dirty = True
        self._render()

    def _on_mouse_up(self, event: tk.Event) -> Optional[str]:
        """松开鼠标：保存撤销快照并恢复高清显示。"""
        if self._color_pick_gesture or time.monotonic() < self._suppress_brush_until:
            self._color_pick_gesture = False
            self._drag_active = False
            self._last_brush_point = None
            self._set_canvas_tool_cursor()
            return "break"
        was_glyph_drag = self._drag_active and self._mode == self.MODE_SELECT
        was_transform_drag = was_glyph_drag and self._drag_handle != ""
        if was_glyph_drag:
            self._push_undo()
        self._drag_active = False
        self._drag_handle = ""
        self._end_brush(event)
        if was_transform_drag:
            self._last_transform_signature = None
        if was_glyph_drag:
            self._render()
        if self._mode == self.MODE_SELECT:
            self._on_mouse_motion(event)

    def _on_ctrl_down(self, event: tk.Event) -> None:
        """Ctrl + 手柄 = 扭曲/斜切模式启动。"""
        if self._mode != self.MODE_SELECT:
            return
        handle = self._get_handle_at(event.x, event.y)
        if handle and handle not in ("rotate",):
            self._drag_active = True
            self._drag_handle = handle
            self._drag_start_x = event.x
            self._drag_start_y = event.y
            self._drag_start_bbox = self._glyph_screen_bbox()
            self._save_current_params()
            self._drag_fixed_anchor = None
            corners = self._control_screen_corners()
            if handle in ("n", "e", "s", "w"):
                opposite_edges = {
                    "n": ("sw", "se"),
                    "e": ("nw", "sw"),
                    "s": ("nw", "ne"),
                    "w": ("ne", "se"),
                }
                first_name, second_name = opposite_edges[handle]
                first = corners[first_name]
                second = corners[second_name]
                self._drag_fixed_anchor = (
                    (first[0] + second[0]) / 2.0,
                    (first[1] + second[1]) / 2.0,
                )
            elif handle in ("nw", "ne", "se", "sw"):
                opposite_corners = {"nw": "se", "ne": "sw", "se": "nw", "sw": "ne"}
                self._drag_fixed_anchor = corners[opposite_corners[handle]]

    def _on_ctrl_drag(self, event: tk.Event) -> None:
        """Ctrl拖动角点执行自由扭曲，拖动边中点执行斜切并缩放。"""
        if not self._drag_active:
            return
        dx = event.x - self._drag_start_x
        dy = event.y - self._drag_start_y
        self._handle_stretch(self._drag_handle, dx, dy)
        self._render()

    # ==================== 画笔工具 ====================

    def _start_brush(self, event: tk.Event) -> None:
        start = time.perf_counter()
        self._stroke_start_time = start
        self._stroke_frame_count = 0
        self._stroke_point_count = 0
        self._brush_changed = False
        self._last_brush_point = None
        self._cancel_brush_render()
        if not self._is_inside_canvas(event.x, event.y):
            self._drag_active = False
            self._stroke_start_time = None
            return
        stage_start = time.perf_counter()
        self._bake_transform_for_painting()
        bake_elapsed = (time.perf_counter() - stage_start) * 1000
        stage_start = time.perf_counter()
        point = self._canvas_to_image(event.x, event.y, allow_outside=True)
        coordinate_elapsed = (time.perf_counter() - stage_start) * 1000
        if point is None:
            self._stroke_start_time = None
            return
        if self._mode == self.MODE_DRAW:
            self._stroke_color = self._brush_color
        stage_start = time.perf_counter()
        self._paint_stroke_point(point)
        _PERFORMANCE_LOGGER.info(
            "[笔画] 开始：模式=%s，烘焙变换=%.2f 毫秒，坐标换算=%.2f 毫秒，首点入队=%.2f 毫秒，总计=%.2f 毫秒",
            "画笔" if self._mode == self.MODE_DRAW else "橡皮擦",
            bake_elapsed,
            coordinate_elapsed,
            (time.perf_counter() - stage_start) * 1000,
            (time.perf_counter() - start) * 1000,
        )

    def _continue_brush(self, event: tk.Event) -> None:
        if not self._is_inside_canvas(event.x, event.y):
            if self._pending_brush_points:
                self._flush_brush_render()
            self._last_brush_point = None
            return
        point = self._canvas_to_image(event.x, event.y, allow_outside=True)
        if point is not None:
            self._paint_stroke_point(point)

    def _end_brush(self, event: tk.Event) -> None:
        if self._mode in (self.MODE_DRAW, self.MODE_ERASER) and self._is_inside_canvas(event.x, event.y):
            # 松键事件补齐最后一段，主线程繁忙时也不会丢失笔画末端。
            point = self._canvas_to_image(event.x, event.y, allow_outside=True)
            if point is not None and point != self._last_brush_point:
                self._paint_stroke_point(point, render=False)
        self._commit_active_brush()

    def _commit_active_brush(self) -> None:
        """提交当前像素笔画，并在不重建整图显示的情况下写入撤销栈。"""
        commit_start = time.perf_counter()
        flush_elapsed = 0.0
        undo_elapsed = 0.0
        if self._pending_brush_points:
            stage_start = time.perf_counter()
            self._flush_brush_render(refresh=True)
            flush_elapsed = (time.perf_counter() - stage_start) * 1000
        self._cancel_brush_render()
        if self._mode in (self.MODE_DRAW, self.MODE_ERASER) and self._brush_changed:
            stage_start = time.perf_counter()
            self._push_undo()
            undo_elapsed = (time.perf_counter() - stage_start) * 1000
            self._brush_changed = False
            self._last_brush_point = None
        if self._stroke_start_time is not None:
            _PERFORMANCE_LOGGER.info(
                "[笔画] 结束：模式=%s，帧数=%d，采样点=%d，末帧局部刷新=%.2f 毫秒，透明裁剪=%.2f 毫秒，撤销快照=%.2f 毫秒，完整高清渲染=已跳过，提交总计=%.2f 毫秒，整笔历时=%.2f 毫秒",
                "画笔" if self._mode == self.MODE_DRAW else "橡皮擦",
                self._stroke_frame_count,
                self._stroke_point_count,
                flush_elapsed,
                0.0,
                undo_elapsed,
                (time.perf_counter() - commit_start) * 1000,
                (time.perf_counter() - self._stroke_start_time) * 1000,
            )
            self._stroke_start_time = None

    def _finish_pixel_edit(self) -> None:
        """结束像素编辑，按当前透明内容收紧字形并清除旧变换缓存。"""
        self._commit_active_brush()
        self._last_brush_point = None
        self._crop_transparent_preserve_origin()
        self._last_transformed = None
        self._last_transformed_origin = (0.0, 0.0)
        self._last_transformed_bbox = None
        self._last_transform_signature = None

    def _bake_transform_for_painting(self) -> None:
        """像素编辑前烘焙几何变换，使笔尖坐标与所见字形完全一致。"""
        if self._edit_img is None:
            return
        transformed = self._transformed_glyph()
        origin_x, origin_y = self._transformed_origin(transformed)
        self._edit_img = transformed
        self._image_origin_x = origin_x
        self._image_origin_y = origin_y
        self._tx_scale = 1.0
        self._tx_rotate = 0.0
        self._tx_offset_x = 0.0
        self._tx_offset_y = 0.0
        self._tx_stretch_w = 1.0
        self._tx_stretch_h = 1.0
        self._tx_distort = [0.0] * 8
        self._last_transform_signature = None

    def _canvas_to_image(
        self, cx: int, cy: int, allow_outside: bool = False
    ) -> Optional[tuple[int, int]]:
        """按实际工作区贴图位置反算当前编辑图的像素坐标。"""
        if self._edit_img is None or self._workspace_scale_x <= 0 or self._workspace_scale_y <= 0:
            return None
        workspace_x = (cx - self._workspace_canvas_x) / self._workspace_scale_x
        workspace_y = (cy - self._workspace_canvas_y) / self._workspace_scale_y
        ix = math.floor(workspace_x) - self._workspace_paste_x
        iy = math.floor(workspace_y) - self._workspace_paste_y
        if allow_outside:
            return ix, iy
        if 0 <= ix < self._edit_img.width and 0 <= iy < self._edit_img.height:
            return ix, iy
        return None

    def _expand_edit_image_for_point(self, point: tuple[int, int], radius: int) -> tuple[int, int]:
        """画笔落在现有包围盒外时扩展局部图像并保持田字格坐标不变。"""
        if self._edit_img is None:
            return point
        x, y = point
        left = max(0, radius - x)
        top = max(0, radius - y)
        right = max(0, x + radius + 1 - self._edit_img.width)
        bottom = max(0, y + radius + 1 - self._edit_img.height)
        if not any((left, top, right, bottom)):
            return point
        expanded = Image.new(
            "RGBA",
            (self._edit_img.width + left + right, self._edit_img.height + top + bottom),
            (0, 0, 0, 0),
        )
        expanded.alpha_composite(self._edit_img, (left, top))
        self._edit_img = expanded
        self._image_origin_x -= left
        self._image_origin_y -= top
        return x + left, y + top

    def _cancel_brush_render(self) -> None:
        """取消尚未执行的笔迹批处理与显示刷新。"""
        if self._brush_render_job is not None:
            try:
                self.after_cancel(self._brush_render_job)
            except tk.TclError:
                pass
            self._brush_render_job = None
        self._brush_schedule_time = None
        self._pending_brush_points.clear()

    def _flush_brush_render(self, refresh: bool = True) -> None:
        """按帧合并高频采样点，一次修改局部像素并按需刷新字形。"""
        if self._brush_render_job is not None:
            try:
                self.after_cancel(self._brush_render_job)
            except tk.TclError:
                pass
        flush_start = time.perf_counter()
        schedule_wait = (
            (flush_start - self._brush_schedule_time) * 1000
            if self._brush_schedule_time is not None
            else 0.0
        )
        self._brush_render_job = None
        self._brush_schedule_time = None
        points = self._pending_brush_points
        self._pending_brush_points = []
        self._stroke_frame_count += 1
        self._stroke_point_count += len(points)
        dirty_region: Optional[tuple[int, int, int, int]] = None
        image_size_changed = False
        mask_elapsed = 0.0
        pixel_elapsed = 0.0
        render_elapsed = 0.0
        if points and self._edit_img is not None:
            previous_size = self._edit_img.size
            stage_start = time.perf_counter()
            region, mask, adjusted_points = self._brush_region_for_points(points)
            mask_elapsed = (time.perf_counter() - stage_start) * 1000
            image_size_changed = self._edit_img.size != previous_size
            if adjusted_points:
                stage_start = time.perf_counter()
                if self._mode == self.MODE_DRAW:
                    self._draw_region(region, mask)
                elif self._mode == self.MODE_ERASER:
                    self._erase_region(region, mask)
                pixel_elapsed = (time.perf_counter() - stage_start) * 1000
                dirty_region = region
                self._last_brush_point = adjusted_points[-1]
                self._dirty = True
                self._brush_changed = True
                self._last_transform_signature = None
        if refresh and dirty_region is not None:
            stage_start = time.perf_counter()
            self._render_pixel_edit(None if image_size_changed else dirty_region)
            render_elapsed = (time.perf_counter() - stage_start) * 1000
        _PERFORMANCE_LOGGER.info(
            "[画笔帧] 模式=%s，采样点=%d，调度等待=%.2f 毫秒，轨迹掩码=%.2f 毫秒，像素修改=%.2f 毫秒，显示调用=%.2f 毫秒，总计=%.2f 毫秒，刷新=%s",
            "画笔" if self._mode == self.MODE_DRAW else "橡皮擦",
            len(points),
            schedule_wait,
            mask_elapsed,
            pixel_elapsed,
            render_elapsed,
            (time.perf_counter() - flush_start) * 1000,
            "是" if refresh else "否",
        )

    def _schedule_brush_render(self) -> None:
        """将像素计算与显示刷新合并到约60帧，避免高频绘图板事件积压。"""
        if self._brush_render_job is None:
            self._brush_schedule_time = time.perf_counter()
            self._brush_render_job = self.after(16, self._flush_brush_render)

    def _paint_stroke_point(
        self, point: tuple[int, int], render: bool = True
    ) -> None:
        """收集连续笔迹采样点，并在下一显示帧中统一处理。"""
        self._pending_brush_points.append(point)
        if render:
            self._schedule_brush_render()

    def _brush_region_for_points(
        self, points: list[tuple[int, int]]
    ) -> tuple[tuple[int, int, int, int], Image.Image, list[tuple[int, int]]]:
        """为一帧内的轨迹生成局部脏矩形和连续笔迹掩码。"""
        if self._edit_img is None or not points:
            return (0, 0, 1, 1), Image.new("L", (1, 1), 0), points
        radius = max(1, int(round(self._brush_size * (0.45 + 0.55 * self._pressure) / 2.0)))
        start = self._last_brush_point
        raw_points = ([start] if start is not None else []) + points
        min_x = min(point[0] for point in raw_points)
        min_y = min(point[1] for point in raw_points)
        max_x = max(point[0] for point in raw_points)
        max_y = max(point[1] for point in raw_points)
        expand_left = max(0, radius - min_x)
        expand_top = max(0, radius - min_y)
        expand_right = max(0, max_x + radius + 1 - self._edit_img.width)
        expand_bottom = max(0, max_y + radius + 1 - self._edit_img.height)
        if any((expand_left, expand_top, expand_right, expand_bottom)):
            expanded = Image.new(
                "RGBA",
                (
                    self._edit_img.width + expand_left + expand_right,
                    self._edit_img.height + expand_top + expand_bottom,
                ),
                (0, 0, 0, 0),
            )
            expanded.alpha_composite(self._edit_img, (expand_left, expand_top))
            self._edit_img = expanded
            self._image_origin_x -= expand_left
            self._image_origin_y -= expand_top
            self._workspace_paste_x -= expand_left
            self._workspace_paste_y -= expand_top
            raw_points = [(x + expand_left, y + expand_top) for x, y in raw_points]
            points = [(x + expand_left, y + expand_top) for x, y in points]
            if start is not None:
                start = (start[0] + expand_left, start[1] + expand_top)
                self._last_brush_point = start

        stroke_points: list[tuple[int, int]] = []
        previous = start
        for current in points:
            if previous is None:
                stroke_points.append(current)
            else:
                distance = max(abs(current[0] - previous[0]), abs(current[1] - previous[1]))
                spacing = max(1, int(self._brush_size * max(0.4, self._pressure) / 4))
                steps = max(1, int(math.ceil(distance / spacing)))
                stroke_points.extend(
                    (
                        int(round(previous[0] + (current[0] - previous[0]) * step / steps)),
                        int(round(previous[1] + (current[1] - previous[1]) * step / steps)),
                    )
                    for step in range(1, steps + 1)
                )
            previous = current
        if not stroke_points:
            stroke_points = points
        left = max(0, min(point[0] for point in raw_points) - radius)
        top = max(0, min(point[1] for point in raw_points) - radius)
        right = min(self._edit_img.width, max(point[0] for point in raw_points) + radius + 1)
        bottom = min(self._edit_img.height, max(point[1] for point in raw_points) + radius + 1)
        mask = Image.new("L", (max(1, right - left), max(1, bottom - top)), 0)
        draw = ImageDraw.Draw(mask)
        for x, y in stroke_points:
            draw.ellipse(
                (x - left - radius, y - top - radius, x - left + radius, y - top + radius),
                fill=255,
            )
        return (left, top, right, bottom), mask, points

    def _draw_region(self, region: tuple[int, int, int, int], mask: Image.Image) -> None:
        """仅在脏矩形内绘制画笔颜色。"""
        if self._edit_img is None:
            return
        base = self._edit_img.crop(region)
        ink = Image.new("RGBA", base.size, self._stroke_color)
        if self._stroke_color[3] < 255:
            alpha = self._stroke_color[3]
            mask = mask.point([value * alpha // 255 for value in range(256)])
        ink.putalpha(mask)
        base.alpha_composite(ink)
        self._edit_img.paste(base, region[:2])

    def _erase_region(self, region: tuple[int, int, int, int], mask: Image.Image) -> None:
        """仅清除脏矩形内的Alpha通道，避免整图合成。"""
        if self._edit_img is None:
            return
        base = self._edit_img.crop(region)
        alpha = base.getchannel("A")
        alpha.paste(0, (0, 0), mask)
        base.putalpha(alpha)
        self._edit_img.paste(base, region[:2])

    def _estimate_ink_color(self, _point: tuple[int, int]) -> tuple[int, int, int, int]:
        """返回当前画笔颜色。"""
        return self._brush_color

    def _update_global_ink_color(self) -> None:
        """将当前字形笔画中出现次数最多的颜色设为默认画笔颜色。"""
        if self._edit_img is None:
            self._brush_color = (0, 0, 0, 255)
            return
        array = np.asarray(self._edit_img)
        pixels = array[array[:, :, 3] >= 160, :3]
        if len(pixels) == 0:
            self._brush_color = (0, 0, 0, 255)
            return
        colors, counts = np.unique(pixels, axis=0, return_counts=True)
        color = colors[int(np.argmax(counts))]
        self._brush_color = (int(color[0]), int(color[1]), int(color[2]), 255)
        self._stroke_color = self._brush_color

    def _pick_color_at(self, x: int, y: int) -> None:
        """从字形的非透明像素指定画笔颜色。"""
        point = self._canvas_to_image(x, y)
        if point is None or self._edit_img is None:
            return
        color = cast(tuple[int, int, int, int], self._edit_img.getpixel(point))
        if color[3] >= 32:
            self._brush_color = (int(color[0]), int(color[1]), int(color[2]), 255)
            self._stroke_color = self._brush_color

    def _on_alt_key_press(self, _event: tk.Event) -> None:
        """记录 Alt 实际按下状态，兼容绘图板驱动未上报修饰键状态。"""
        self._alt_pressed = True

    def _on_alt_key_release(self, _event: tk.Event) -> None:
        """记录 Alt 松开状态。"""
        self._alt_pressed = False

    def _start_color_pick(self, event: tk.Event) -> str:
        """开始或继续取色手势，并阻断随后可能重复到达的落笔事件。"""
        if self._mode == self.MODE_TRANSFORM:
            self._confirm_transform(bake=True)
            self._render()
        if self._pending_brush_points or self._brush_changed:
            self._commit_active_brush()
        self._color_pick_gesture = True
        self._drag_active = False
        self._last_brush_point = None
        self._suppress_brush_until = time.monotonic() + 0.12
        self._pick_color_at(event.x, event.y)
        return "break"

    def _on_pick_color(self, event: tk.Event) -> str:
        """使用鼠标右键从当前所见字形上指定画笔颜色。"""
        return self._start_color_pick(event)

    def _on_color_pick_release(self, _event: tk.Event) -> str:
        """结束右键取色手势。"""
        self._color_pick_gesture = False
        self._drag_active = False
        self._last_brush_point = None
        self._set_canvas_tool_cursor()
        return "break"

    def _on_tablet_pressure(self, event: tk.Event) -> None:
        """接收驱动或桥接层提供的0～1压力值。"""
        value = getattr(event, "pressure", getattr(event, "data", 1.0))
        try:
            self._pressure = max(0.05, min(1.0, float(value)))
        except (TypeError, ValueError):
            self._pressure = 1.0

    def _on_brush_size_change(self, value: str) -> None:
        """应用工具栏中的笔触大小。"""
        self._brush_size = max(1, min(100, int(round(float(value)))))
        if self._brush_cursor_x is not None and self._brush_cursor_y is not None:
            self._draw_brush_cursor(self._brush_cursor_x, self._brush_cursor_y)

    def _change_brush_size(self, delta: int) -> None:
        """按步长调整画笔或橡皮擦的笔触大小。"""
        if self._mode not in (self.MODE_DRAW, self.MODE_ERASER):
            return
        step = 1 if self._brush_size < 10 else 2 if self._brush_size < 30 else 5
        self._brush_size = max(1, min(100, self._brush_size + delta * step))
        self._brush_size_var.set(self._brush_size)
        if self._brush_cursor_x is not None and self._brush_cursor_y is not None:
            self._draw_brush_cursor(self._brush_cursor_x, self._brush_cursor_y)

    def _on_brush_wheel(self, event: tk.Event) -> Optional[str]:
        """画笔类工具直接滚轮调整笔触大小。"""
        if self._mode not in (self.MODE_DRAW, self.MODE_ERASER):
            return None
        self._change_brush_size(1 if event.delta > 0 else -1)
        return "break"

    def _on_brush_smaller(self, _event: tk.Event) -> str:
        self._change_brush_size(-1)
        return "break"

    def _on_brush_larger(self, _event: tk.Event) -> str:
        self._change_brush_size(1)
        return "break"

    def _crop_transparent_preserve_origin(self) -> None:
        if self._edit_img is None:
            return
        bbox = self._edit_img.getchannel("A").getbbox()
        if bbox is None:
            return
        self._edit_img = self._edit_img.crop(bbox)
        self._image_origin_x += bbox[0]
        self._image_origin_y += bbox[1]

    # ==================== 手柄变换 ====================

    def _save_current_params(self) -> None:
        self._drag_saved_params = {
            "缩放": self._tx_scale,
            "旋转": self._tx_rotate,
            "偏移X": self._tx_offset_x,
            "偏移Y": self._tx_offset_y,
            "拉伸W": self._tx_stretch_w,
            "拉伸H": self._tx_stretch_h,
            "扭曲": list(self._tx_distort),
        }

    def _adjust_offset(self, dx: int, dy: int) -> None:
        self._tx_offset_x = self._drag_saved_params["偏移X"] + dx / max(self._view_scale, 1e-6)
        self._tx_offset_y = self._drag_saved_params["偏移Y"] + dy / max(self._view_scale, 1e-6)

    def _handle_scale(
        self,
        handle: str,
        dx: int,
        dy: int,
        *,
        keep_aspect: bool = False,
        from_center: bool = False,
    ) -> None:
        """八方向缩放：Shift 约束比例，Alt 从中心缩放。"""
        if self._edit_img is None:
            return
        x1, y1, x2, y2 = self._drag_start_bbox
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        saved = self._drag_saved_params
        center_dx = 0.0
        center_dy = 0.0
        center_factor = 2.0 if from_center else 1.0

        if handle in ("nw", "ne", "se", "sw"):
            vectors = {
                "nw": (x1 - x2, y1 - y2),
                "ne": (x2 - x1, y1 - y2),
                "se": (x2 - x1, y2 - y1),
                "sw": (x1 - x2, y2 - y1),
            }
            vx, vy = vectors[handle]
            factor = (
                (vx + center_factor * dx) * vx + (vy + center_factor * dy) * vy
            ) / max(1.0, vx * vx + vy * vy)
            factor = max(0.05, min(20.0, factor))
            self._tx_scale = max(0.05, min(20.0, saved["缩放"] * factor))
            if not from_center:
                center_dx = vx * (factor - 1.0) / 2.0
                center_dy = vy * (factor - 1.0) / 2.0
        elif handle in ("e", "w"):
            signed_dx = dx if handle == "e" else -dx
            factor = max(0.05, min(20.0, 1.0 + center_factor * signed_dx / width))
            if keep_aspect:
                self._tx_scale = max(0.05, min(20.0, saved["缩放"] * factor))
            else:
                self._tx_stretch_w = max(0.05, min(20.0, saved["拉伸W"] * factor))
            if not from_center:
                center_dx = dx / 2.0
        elif handle in ("n", "s"):
            signed_dy = dy if handle == "s" else -dy
            factor = max(0.05, min(20.0, 1.0 + center_factor * signed_dy / height))
            if keep_aspect:
                self._tx_scale = max(0.05, min(20.0, saved["缩放"] * factor))
            else:
                self._tx_stretch_h = max(0.05, min(20.0, saved["拉伸H"] * factor))
            if not from_center:
                center_dy = dy / 2.0

        self._tx_offset_x = saved["偏移X"] + center_dx / max(0.01, self._view_scale)
        self._tx_offset_y = saved["偏移Y"] + center_dy / max(0.01, self._view_scale)

    def _handle_stretch(self, handle: str, dx: int, dy: int) -> None:
        """扭曲：Ctrl 拖动角点或边中点，并固定边中点的对边。"""
        handle_indices = {
            "nw": (0,),
            "n": (0, 2),
            "ne": (2,),
            "e": (2, 4),
            "se": (4,),
            "s": (4, 6),
            "sw": (6,),
            "w": (6, 0),
        }
        indices = handle_indices.get(handle)
        if indices is None:
            return
        saved = list(self._drag_saved_params.get("扭曲", [0.0] * 8))
        image_dx = dx / max(self._view_scale, 1e-6)
        image_dy = dy / max(self._view_scale, 1e-6)
        for corner_index in indices:
            saved[corner_index] += image_dx
            saved[corner_index + 1] += image_dy
        self._tx_distort = saved

        if self._drag_fixed_anchor is not None:
            # 每帧都从拖动开始时的偏移重新计算，避免累计取整造成固定顶点抖动。
            self._tx_offset_x = self._drag_saved_params["偏移X"]
            self._tx_offset_y = self._drag_saved_params["偏移Y"]
            self._last_transformed = None
            self._last_transformed_bbox = None
            self._last_transform_signature = None
            corners = self._control_screen_corners()
            if handle in ("n", "e", "s", "w"):
                opposite_edges = {
                    "n": ("sw", "se"),
                    "e": ("nw", "sw"),
                    "s": ("nw", "ne"),
                    "w": ("ne", "se"),
                }
                first_name, second_name = opposite_edges[handle]
                first = corners[first_name]
                second = corners[second_name]
                current_anchor_x = (first[0] + second[0]) / 2.0
                current_anchor_y = (first[1] + second[1]) / 2.0
            else:
                opposite_corners = {"nw": "se", "ne": "sw", "se": "nw", "sw": "ne"}
                current_anchor_x, current_anchor_y = corners[opposite_corners[handle]]
            scale = max(self._view_scale, 1e-6)
            self._tx_offset_x += (self._drag_fixed_anchor[0] - current_anchor_x) / scale
            self._tx_offset_y += (self._drag_fixed_anchor[1] - current_anchor_y) / scale
        self._dirty = True

    def _angle_from_center(self, cx: int, cy: int, px: int, py: int) -> float:
        import math
        return math.degrees(math.atan2(py - cy, px - cx))

    # ==================== 撤销 ====================

    def _reset_undo(self) -> None:
        """重置撤销栈。"""
        self._undo_stack = [self._snapshot()]
        self._undo_index = 0

    def _snapshot(self) -> dict:
        """创建当前状态快照。"""
        return {
            "image": self._edit_img.copy() if self._edit_img else None,
            "缩放": self._tx_scale,
            "旋转": self._tx_rotate,
            "偏移X": self._tx_offset_x,
            "偏移Y": self._tx_offset_y,
            "拉伸W": self._tx_stretch_w,
            "拉伸H": self._tx_stretch_h,
            "扭曲": list(self._tx_distort),
            "图像原点X": self._image_origin_x,
            "图像原点Y": self._image_origin_y,
        }

    def _push_undo(self) -> None:
        """保存撤销快照。"""
        self._undo_stack = self._undo_stack[:self._undo_index + 1]
        self._undo_stack.append(self._snapshot())
        self._undo_index = len(self._undo_stack) - 1

    def _on_undo(self, _event: tk.Event) -> None:
        """Ctrl+Z 撤销。"""
        if self._undo_index > 0:
            self._undo_index -= 1
            self._restore_snapshot(self._undo_stack[self._undo_index])
            self._render()

    def _on_redo(self, _event: tk.Event) -> None:
        """Ctrl+Shift+Z 或 Ctrl+Y 重做。"""
        if self._undo_index + 1 < len(self._undo_stack):
            self._undo_index += 1
            self._restore_snapshot(self._undo_stack[self._undo_index])
            self._render()

    def _on_revert(self, _event: tk.Event) -> None:
        """R 还原到最后一次保存。"""
        if self._undo_stack:
            self._undo_index = 0
            self._restore_snapshot(self._undo_stack[0])
            self._render()

    def _restore_snapshot(self, snap: dict) -> None:
        self._cancel_brush_render()
        self._last_brush_point = None
        self._brush_changed = False
        if snap.get("image"):
            self._edit_img = snap["image"].copy()
        self._tx_scale = snap["缩放"]
        self._tx_rotate = snap["旋转"]
        self._tx_offset_x = snap["偏移X"]
        self._tx_offset_y = snap["偏移Y"]
        self._tx_stretch_w = snap["拉伸W"]
        self._tx_stretch_h = snap["拉伸H"]
        self._tx_distort = list(snap.get("扭曲", [0.0] * 8))
        self._image_origin_x = float(snap.get("图像原点X", self._image_origin_x))
        self._image_origin_y = float(snap.get("图像原点Y", self._image_origin_y))
        self._last_transform_signature = None
        self._update_global_ink_color()
        self._dirty = True

    def save_current(self, notify_status_change: bool = True) -> bool:
        """保存当前手工审核结果到独立审核层，并使旧成品失效。"""
        if not self._current_char or self._edit_img is None:
            return False
        if self._mode in (self.MODE_DRAW, self.MODE_ERASER):
            self._commit_active_brush()
        if self._mode == self.MODE_TRANSFORM:
            self._confirm_transform(bake=True)
            self._begin_transform_session()
        variants = self._get_variants(self._current_char)
        if self._current_variant_index >= len(variants):
            return False
        variant = variants[self._current_variant_index]
        filename = variant.get("审核文件") or variant.get("中间文件", "")
        variant_id = variant.get("变体ID", "")
        if not filename or not variant_id:
            return False

        reviewed_dir = self._glyph.get_workflow_dirs()["手工审核"]
        os.makedirs(reviewed_dir, exist_ok=True)
        from utils.file_utils import compute_file_md5
        save_path = os.path.join(reviewed_dir, filename)
        save_image = self.build_output_image()
        expand_x = max(0, (save_image.width - self._canvas_w) // 2)
        expand_y = max(0, (save_image.height - self._canvas_h) // 2)
        save_image.save(
            save_path,
            "PNG",
            dpi=(self._target_dpi, self._target_dpi),
        )
        self._edit_img = save_image
        self._base_img = save_image.copy()
        self._original_img = save_image.copy()
        self._original_base_img = save_image.copy()
        self._image_origin_x = -float(expand_x)
        self._image_origin_y = -float(expand_y)
        self._tx_scale = 1.0
        self._tx_rotate = 0.0
        self._tx_offset_x = 0
        self._tx_offset_y = 0
        self._tx_stretch_w = 1.0
        self._tx_stretch_h = 1.0
        self._tx_distort = [0.0] * 8
        self._transform_session_snapshot = None
        variant["变换参数"] = {
            "缩放": 1.0,
            "旋转": 0.0,
            "偏移X": 0,
            "偏移Y": 0,
            "拉伸W": 1.0,
            "拉伸H": 1.0,
            "扭曲": [0.0] * 8,
            "图像原点": [
                round(float(self._image_origin_x), 6),
                round(float(self._image_origin_y), 6),
            ],
        }
        self._glyph.mark_manual_saved(
            variant_id,
            filename,
            compute_file_md5(save_path),
            edited=True,
        )
        self._glyph.save()
        self._base_img = self._edit_img.copy()
        self._dirty = False
        self._reset_undo()
        self._begin_transform_session()
        if notify_status_change and self._on_status_change:
            self._on_status_change()
        return True

    def build_output_image(self) -> Image.Image:
        """按文字越界量围绕田字格中心对称扩展，完整生成成品图。"""
        transformed = self._transformed_glyph()
        origin_x, origin_y = self._transformed_origin(transformed)
        bbox = transformed.getchannel("A").getbbox()
        if bbox is None:
            return Image.new(
                "RGBA", (self._canvas_w, self._canvas_h), (0, 0, 0, 0)
            )
        left = origin_x + bbox[0]
        top = origin_y + bbox[1]
        right = origin_x + bbox[2]
        bottom = origin_y + bbox[3]
        expand_x = int(
            math.ceil(max(0.0, -left, right - self._canvas_w))
        )
        expand_y = int(
            math.ceil(max(0.0, -top, bottom - self._canvas_h))
        )
        output = Image.new(
            "RGBA",
            (self._canvas_w + expand_x * 2, self._canvas_h + expand_y * 2),
            (0, 0, 0, 0),
        )
        output.alpha_composite(
            transformed,
            (
                int(round(origin_x + expand_x)),
                int(round(origin_y + expand_y)),
            ),
        )
        output.info["dpi"] = (self._target_dpi, self._target_dpi)
        return output

    def has_unsaved_changes(self) -> bool:
        """返回当前变体是否存在未保存修改。"""
        return self._dirty


    # ==================== 画布事件 ====================

    def _on_canvas_configure(self, _event: tk.Event) -> None:
        if self._edit_img:
            self._render()

    # ==================== 视图控制 ====================

    def _fit_to_window(self) -> None:
        self._view_mode = "fit"
        self._render()

    def _zoom_actual(self) -> None:
        self._view_mode = "actual"
        self._render()

    def _toggle_bg(self) -> None:
        """在白底与透明棋盘格之间切换。"""
        self._checkerboard_enabled = not self._checkerboard_enabled
        if not self._checkerboard_enabled:
            self._background_tk = None
            self._background_cache_key = None
        self._render()
