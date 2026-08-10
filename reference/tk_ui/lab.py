# lab.py — 模板工坊：选样图调参 → 预览 → 保存模板

import os
import tkinter as tk
from tkinter import ttk, filedialog
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image, ImageTk

from core import pipeline, scoring, registry
from data.registry_store import load_registry
from ui import theme
from ui.widgets.params_editor import ParamsEditor
from utils.file_utils import natural_key, safe_read_json, atomic_write_json
import config


class LabPage(tk.Frame):
    """嵌入主窗口的模板工坊页面。"""

    def __init__(self, parent: tk.Misc, on_close: Optional[Callable[[], None]] = None) -> None:
        super().__init__(parent, bg=theme.BG_MAIN)
        self._on_close = on_close

        # 状态
        self._samples: list[dict[str, Any]] = []  # [{"路径": ..., "缩略图": PIL, ...}]
        self._scheme: dict[str, Any] = self._default_scheme()
        self._current_layer: str = "L3"
        self._current_algo: str = ""

        # 注册表
        self._registry = load_registry()
        self._groups = self._registry.get("分组", {})

        # 内置模板
        self._templates: dict[str, Any] = self._load_templates()

        self._build()
        self._refresh_preview()
        self.pack(fill=tk.BOTH, expand=True)

    def _close_page(self) -> None:
        """关闭模板工坊并返回进入前页面。"""
        if self._on_close:
            self._on_close()

    # ==================== 默认方案 ====================

    def _default_scheme(self) -> dict[str, Any]:
        return {
            "版本": 3,
            "预处理": {"转灰度": False, "反相": False, "墨色归一": False, "墨色基准": 60},
            "L1": None, "L2": None,
            "L3": {"算法": "Otsu", "参数": {"偏移": 0}},
            "L4": None,
            "L5": {"算法": "面积过滤", "参数": {"min_area": 60, "连通类型": 8, "相对模式": False}},
        }

    # ==================== 模板加载 ====================

    def _load_templates(self) -> dict[str, Any]:
        data = safe_read_json(config.TEMPLATE_FILE, default={})
        if not isinstance(data, dict):
            data = {}
        # 补齐内置模板
        builtins = {
            "通用去杂": {
                "版本": 3, "预处理": {"转灰度": False, "反相": False, "墨色归一": False, "墨色基准": 60},
                "L3": {"算法": "Otsu", "参数": {"偏移": 0}},
                "L5": {"算法": "面积过滤", "参数": {"min_area": 60, "连通类型": 8, "相对模式": False, "相对比例": 0.002}},
            },
            "拓片清散": {
                "版本": 3, "预处理": {"转灰度": True, "反相": True, "墨色归一": True, "墨色基准": 55},
                "L3": {"算法": "Otsu", "参数": {"偏移": 0}},
                "L4": {"算法": "黑帽扣除", "参数": {"核大小": 11, "强度": 1.0}},
                "L5": {"算法": "面积+形状过滤", "参数": {"min_area": 60, "连通类型": 8, "仅孤立": True}},
            },
            "古籍归一": {
                "版本": 3, "预处理": {"转灰度": False, "反相": False, "墨色归一": False, "墨色基准": 60},
                "L2": {"算法": "形态学背景归一", "参数": {"核大小": 51}},
                "L3": {"算法": "Sauvola", "参数": {"窗口": 25, "k": 0.15, "R": 128}},
                "L5": {"算法": "面积过滤", "参数": {"min_area": 60, "连通类型": 8, "相对模式": False, "相对比例": 0.002}},
            },
        }
        for name, tmpl in builtins.items():
            if name not in data:
                data[name] = tmpl
        return data

    def _save_templates(self) -> None:
        atomic_write_json(self._templates, config.TEMPLATE_FILE)

    # ==================== 界面构建 ====================

    def _build(self) -> None:
        # 页面顶栏
        top_bar = tk.Frame(self, bg=theme.BG_PANEL, height=52)
        top_bar.pack(fill=tk.X, side=tk.TOP)
        top_bar.pack_propagate(False)
        theme.make_label(top_bar, "模板工坊", font=theme.FONT_TITLE).pack(side=tk.LEFT, padx=14, pady=8)
        if self._on_close:
            theme.make_button(top_bar, "返回首页", command=self._close_page).pack(side=tk.RIGHT, padx=10, pady=8)

        content = tk.Frame(self, bg=theme.BG_MAIN)
        content.pack(fill=tk.BOTH, expand=True)

        # 左侧：配置区
        self._left = tk.Frame(content, bg=theme.BG_PANEL, width=360)
        self._left.pack(fill=tk.Y, side=tk.LEFT, padx=(0, 2))
        self._left.pack_propagate(False)

        # 标题
        header = tk.Frame(self._left, bg=theme.BG_PANEL)
        header.pack(fill=tk.X, padx=8, pady=(8, 4))
        theme.make_label(header, "模板配置", font=theme.FONT_TITLE).pack(side=tk.LEFT)

        # 样图选择
        self._build_sample_section()

        # 预处理
        self._build_preprocess()

        # 分层选择
        self._build_layer_selector()

        # 参数编辑区
        self._params_frame = tk.Frame(self._left, bg=theme.BG_PANEL)
        self._params_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # 底部按钮
        self._build_buttons()

        # 右侧：可滚动、自适应排列的对照预览区
        self._right = tk.Frame(content, bg=theme.BG_MAIN)
        self._right.pack(fill=tk.BOTH, expand=True, side=tk.RIGHT)

        self._preview_label = theme.make_label(
            self._right, "原图与优化图对照", font=theme.FONT_TITLE,
        )
        self._preview_label.pack(pady=(6, 4))

        preview_container = tk.Frame(self._right, bg=theme.BG_MAIN)
        preview_container.pack(fill=tk.BOTH, expand=True)
        self._preview_canvas = tk.Canvas(
            preview_container, bg=theme.BG_MAIN, highlightthickness=0, bd=0,
        )
        self._preview_scrollbar = ttk.Scrollbar(
            preview_container, orient=tk.VERTICAL, command=self._preview_canvas.yview,
        )
        self._preview_canvas.configure(yscrollcommand=self._preview_scrollbar.set)
        self._preview_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._preview_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._preview_inner = tk.Frame(self._preview_canvas, bg=theme.BG_MAIN)
        self._preview_window = self._preview_canvas.create_window(
            (0, 0), window=self._preview_inner, anchor="nw",
        )
        self._preview_cards: list[tk.Frame] = []
        self._preview_columns = 1
        self._preview_canvas.bind("<Configure>", self._on_preview_resize)
        self._preview_inner.bind("<Configure>", self._on_preview_content_resize)
        self._bind_preview_mousewheel(self._preview_canvas)
        self._bind_preview_mousewheel(self._preview_inner)

    def _build_sample_section(self) -> None:
        """样图选择区。"""
        sect = tk.Frame(self._left, bg=theme.BG_PANEL)
        sect.pack(fill=tk.X, padx=8, pady=2)
        theme.make_label(sect, "📷 样图", font=theme.FONT_BOLD).pack(anchor="w")
        row = tk.Frame(sect, bg=theme.BG_PANEL)
        row.pack(fill=tk.X, pady=2)
        theme.make_button(row, "选择样图...", command=self._on_select_samples).pack(side=tk.LEFT, padx=2)
        self._sample_count_label = theme.make_label(row, "0 张", fg=theme.FG_MUTED)
        self._sample_count_label.pack(side=tk.LEFT, padx=6)

    def _build_preprocess(self) -> None:
        """预处理选项。"""
        sect = tk.Frame(self._left, bg=theme.BG_PANEL)
        sect.pack(fill=tk.X, padx=8, pady=2)
        theme.make_label(sect, "⚙ 预处理", font=theme.FONT_BOLD).pack(anchor="w")

        pre = self._scheme["预处理"]
        self._pre_gray = tk.BooleanVar(value=pre["转灰度"])
        self._pre_invert = tk.BooleanVar(value=pre["反相"])
        self._pre_inknorm = tk.BooleanVar(value=pre["墨色归一"])
        self._pre_inkval = tk.DoubleVar(value=pre.get("墨色基准", 60))

        row = tk.Frame(sect, bg=theme.BG_PANEL)
        row.pack(fill=tk.X)
        tk.Checkbutton(row, text="转灰度", variable=self._pre_gray, bg=theme.BG_PANEL, fg=theme.FG_PRIMARY,
                      selectcolor=theme.BG_INPUT, activebackground=theme.BG_PANEL,
                      command=self._on_scheme_change).pack(side=tk.LEFT, padx=4)
        tk.Checkbutton(row, text="反相", variable=self._pre_invert, bg=theme.BG_PANEL, fg=theme.FG_PRIMARY,
                      selectcolor=theme.BG_INPUT, activebackground=theme.BG_PANEL,
                      command=self._on_scheme_change).pack(side=tk.LEFT, padx=4)
        tk.Checkbutton(row, text="墨色归一", variable=self._pre_inknorm, bg=theme.BG_PANEL, fg=theme.FG_PRIMARY,
                      selectcolor=theme.BG_INPUT, activebackground=theme.BG_PANEL,
                      command=self._on_scheme_change).pack(side=tk.LEFT, padx=4)
        ink_row = tk.Frame(sect, bg=theme.BG_PANEL)
        ink_row.pack(fill=tk.X)
        theme.make_label(ink_row, "基准:", fg=theme.FG_MUTED, font=theme.FONT_SMALL).pack(side=tk.LEFT)
        scale = tk.Scale(ink_row, from_=5, to=250, variable=self._pre_inkval, orient=tk.HORIZONTAL,
                        bg=theme.BG_PANEL, fg=theme.FG_PRIMARY, troughcolor=theme.BG_INPUT,
                        highlightthickness=0, bd=0, length=120, command=lambda v: self._on_scheme_change())
        scale.pack(side=tk.LEFT)

    def _build_layer_selector(self) -> None:
        """分层下拉选择。"""
        sect = tk.Frame(self._left, bg=theme.BG_PANEL)
        sect.pack(fill=tk.X, padx=8, pady=(4, 0))
        theme.make_label(sect, "📊 管线配置", font=theme.FONT_BOLD).pack(anchor="w")

        group_names = ["L1 降噪", "L2 背景分离", "L3 二值化", "L4 形态清理", "L5 连通域过滤"]
        layer_keys = ["L1", "L2", "L3", "L4", "L5"]

        for gname, lkey in zip(group_names, layer_keys):
            row = tk.Frame(sect, bg=theme.BG_PANEL)
            row.pack(fill=tk.X, pady=1)
            theme.make_label(row, f"{lkey}:", width=4, fg=theme.FG_MUTED).pack(side=tk.LEFT)

            algos = list(self._groups.get(gname, {}).get("算法", {}).keys())
            algo_names = ["(无)"] + algos
            var = tk.StringVar(value="(无)")
            current_cfg = self._scheme.get(lkey)
            if current_cfg and current_cfg.get("算法"):
                var.set(current_cfg["算法"])

            cb = ttk.Combobox(row, textvariable=var, values=algo_names, state="readonly", width=14)
            cb.pack(side=tk.LEFT, padx=2)
            cb.bind("<<ComboboxSelected>>", lambda e, l=lkey, g=gname: self._on_layer_select(l, g))
            setattr(self, f"_layer_{lkey}_var", var)
            setattr(self, f"_layer_{lkey}_combobox", cb)

    def _build_buttons(self) -> None:
        """底部操作按钮。"""
        bar = tk.Frame(self._left, bg=theme.BG_PANEL)
        bar.pack(fill=tk.X, padx=8, pady=(8, 8))

        theme.make_button(bar, "🪄 自动优化", accent=True, command=self._on_auto_optimize).pack(side=tk.LEFT, padx=2)
        theme.make_button(bar, "💾 保存模板", command=self._on_save_template).pack(side=tk.RIGHT, padx=2)
        theme.make_button(bar, "🔄 重置", command=self._on_reset).pack(side=tk.RIGHT, padx=2)

    # ==================== 事件处理 ====================

    @staticmethod
    def _open_as_white_background_grayscale(path: str) -> Image.Image:
        """打开图片，将透明区域铺为白色后转换为灰度图。"""
        with Image.open(path) as source_image:
            source_image.seek(0)
            rgba = source_image.convert("RGBA")
            white_background = Image.new("RGBA", rgba.size, "white")
            white_background.alpha_composite(rgba)
            return white_background.convert("L")

    def _on_select_samples(self) -> None:
        """选择样图。"""
        files = filedialog.askopenfilenames(
            title="选择样图",
            filetypes=[
                ("所有支持的图片", "*.png *.jpg *.jpeg *.jpe *.jfif *.bmp *.dib *.tif *.tiff *.webp *.gif *.tga *.ppm *.pgm *.pbm *.pnm *.ico"),
                ("便携式网络图片", "*.png"),
                ("JPEG 图片", "*.jpg *.jpeg *.jpe *.jfif"),
                ("位图", "*.bmp *.dib"),
                ("TIFF 图片", "*.tif *.tiff"),
                ("WebP 图片", "*.webp"),
                ("动图", "*.gif"),
                ("其他常用图片", "*.tga *.ppm *.pgm *.pbm *.pnm *.ico"),
                ("所有文件", "*.*"),
            ],
        )
        if not files:
            return
        self._samples = []
        for fp in files:
            try:
                img = self._open_as_white_background_grayscale(fp)
                thumb = img.copy()
                thumb.thumbnail((80, 80), Image.Resampling.LANCZOS)
                self._samples.append({"路径": fp, "图像": img, "缩略图": thumb})
            except Exception:
                pass
        self._sample_count_label.configure(text=f"{len(self._samples)} 张")
        self._refresh_preview()

    def _on_layer_select(self, layer_key: str, group_name: str) -> None:
        """层算法选择变化时更新参数编辑器和预览。"""
        var: tk.StringVar = getattr(self, f"_layer_{layer_key}_var")
        algo_name = var.get()
        self._current_layer = layer_key
        self._current_algo = algo_name

        # 更新方案
        if algo_name == "(无)":
            self._scheme[layer_key] = None
        else:
            self._scheme[layer_key] = {"算法": algo_name, "参数": {}}

        # 重建参数编辑器
        for w in self._params_frame.winfo_children():
            w.destroy()

        if algo_name != "(无)":
            algos = self._groups.get(group_name, {}).get("算法", {})
            algo_def = algos.get(algo_name, {})
            if algo_def.get("参数"):
                theme.make_label(self._params_frame, algo_name, font=theme.FONT_BOLD).pack(anchor="w", pady=(4, 0))
                editor = ParamsEditor(self._params_frame, group_name, algo_name, {},
                                     on_change=lambda p: self._on_params_change(layer_key, p))
                editor.pack(fill=tk.X)

                # 加载已有参数
                existing = (self._scheme.get(layer_key) or {}).get("参数", {})
                if existing:
                    editor.set_values(existing)
                setattr(self, f"_editor_{layer_key}", editor)

        self._on_scheme_change()

    def _on_params_change(self, layer_key: str, params: dict[str, Any]) -> None:
        """参数变更时更新方案并预览。"""
        cfg = self._scheme.get(layer_key)
        if cfg:
            cfg["参数"] = params
        self._refresh_preview()

    def _on_scheme_change(self) -> None:
        """方案变更（预处理/层选择/参数）→ 刷新预览。"""
        pre = self._scheme["预处理"]
        pre["转灰度"] = self._pre_gray.get()
        pre["反相"] = self._pre_invert.get()
        pre["墨色归一"] = self._pre_inknorm.get()
        pre["墨色基准"] = self._pre_inkval.get()
        self._refresh_preview()

    @staticmethod
    def _create_white_background_thumbnail(image: Image.Image, max_size: tuple[int, int]) -> Image.Image:
        """按原比例缩放图片，并将透明区域合成到白色背景。"""
        thumbnail = image.convert("RGBA")
        thumbnail.thumbnail(max_size, Image.Resampling.LANCZOS)
        white_background = Image.new("RGB", thumbnail.size, "white")
        white_background.paste(thumbnail, mask=thumbnail.getchannel("A"))
        return white_background

    def _on_preview_content_resize(self, _event: tk.Event) -> None:
        """内容尺寸变化时同步滚动范围。"""
        self._preview_canvas.configure(scrollregion=self._preview_canvas.bbox("all"))

    def _on_preview_resize(self, event: tk.Event) -> None:
        """根据可用宽度自动计算每行卡片数。"""
        available_width = max(1, int(event.width))
        self._preview_canvas.itemconfigure(self._preview_window, width=available_width)
        card_width = 408
        new_column_count = max(1, available_width // card_width)
        if new_column_count != self._preview_columns:
            self._preview_columns = new_column_count
            self._layout_preview_cards()

    def _layout_preview_cards(self) -> None:
        """按当前列数重新排列全部预览卡片。"""
        for index, card in enumerate(self._preview_cards):
            card.grid_forget()
            card.grid(
                row=index // self._preview_columns,
                column=index % self._preview_columns,
                padx=6, pady=6, sticky="n",
            )
        for column_index in range(self._preview_columns):
            self._preview_inner.grid_columnconfigure(column_index, weight=1)
        self._preview_inner.update_idletasks()
        self._preview_canvas.configure(scrollregion=self._preview_canvas.bbox("all"))

    def _bind_preview_mousewheel(self, widget: tk.Misc) -> None:
        """为预览区控件绑定 Windows 鼠标滚轮事件。"""
        widget.bind("<MouseWheel>", self._on_preview_mousewheel, add="+")

    def _bind_preview_child_mousewheel(self, widget: tk.Misc) -> None:
        """递归绑定卡片内全部控件，保证指针位于图片上也能滚动。"""
        self._bind_preview_mousewheel(widget)
        for child_widget in widget.winfo_children():
            self._bind_preview_child_mousewheel(child_widget)

    def _on_preview_mousewheel(self, event: tk.Event) -> str:
        """鼠标位于预览区时滚动图片列表。"""
        if self._preview_canvas.yview() != (0.0, 1.0):
            scroll_amount = -max(1, abs(int(event.delta)) // 120) if event.delta > 0 else max(1, abs(int(event.delta)) // 120)
            self._preview_canvas.yview_scroll(scroll_amount, "units")
        return "break"

    def _refresh_preview(self) -> None:
        """刷新原图与优化图对照预览区。"""
        for widget in self._preview_inner.winfo_children():
            widget.destroy()
        self._preview_cards.clear()
        self._preview_canvas.yview_moveto(0)

        if not self._samples:
            theme.make_label(
                self._preview_inner, "（选择样图后可对照查看原图与优化效果）",
                fg=theme.FG_MUTED,
            ).pack(expand=True, pady=80)
            return

        for sample in self._samples:
            img = sample["图像"]
            gray = np.array(img, dtype=np.float32)
            _, mask = pipeline.run_pipeline(gray, self._scheme)
            if mask is not None:
                h, w = gray.shape
                rgba = np.zeros((h, w, 4), dtype=np.uint8)
                g_u8 = np.clip(gray, 0, 255).astype(np.uint8)
                text = mask > 0
                rgba[text, 0] = g_u8[text]
                rgba[text, 1] = g_u8[text]
                rgba[text, 2] = g_u8[text]
                rgba[text, 3] = 255
                result = Image.fromarray(rgba, "RGBA")
            else:
                result = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8), "L").convert("RGBA")

            if mask is not None:
                s = scoring.auto_score(mask, gray)
            else:
                s = 0

            card = tk.Frame(
                self._preview_inner, bg=theme.BG_PANEL,
                highlightbackground=theme.BORDER, highlightthickness=1,
            )
            self._preview_cards.append(card)
            theme.make_label(
                card, os.path.basename(sample["路径"]), font=theme.FONT_BOLD,
            ).pack(padx=6, pady=(5, 2))

            comparison_area = tk.Frame(card, bg=theme.BG_PANEL)
            comparison_area.pack(padx=6, pady=2)
            original_thumbnail = self._create_white_background_thumbnail(img, (180, 180))
            optimized_thumbnail = self._create_white_background_thumbnail(result, (180, 180))
            original_photo = ImageTk.PhotoImage(original_thumbnail)
            optimized_photo = ImageTk.PhotoImage(optimized_thumbnail)

            for column_index, (title, photo) in enumerate((("原图", original_photo), ("优化后", optimized_photo))):
                column = tk.Frame(comparison_area, bg=theme.BG_PANEL)
                column.grid(row=0, column=column_index, padx=4, sticky="n")
                theme.make_label(column, title, font=theme.FONT_SMALL).pack()
                image_label = tk.Label(
                    column, image=photo, bg="white", bd=1,
                    relief=tk.SOLID, highlightthickness=0,
                )
                image_label._preview_photo = photo  # type: ignore[attr-defined]
                image_label.pack(pady=(2, 0))

            theme.make_label(
                card, f"优化评分：{s}", font=theme.FONT_SMALL,
                fg=theme.FG_ACCENT,
            ).pack(pady=(2, 5))

        self._layout_preview_cards()
        self._bind_preview_child_mousewheel(self._preview_inner)

    def _on_auto_optimize(self) -> None:
        """一键自动优化：逐样图寻优取最优。"""
        from core.optimizer import auto_pick_for_image

        if not self._samples:
            return

        best_score = -1.0
        best_scheme = None
        best_name = ""

        for sample in self._samples:
            gray = np.array(sample["图像"], dtype=np.float32)
            name, scheme, score, _ = auto_pick_for_image(gray)
            if score > best_score:
                best_score = score
                best_scheme = scheme
                best_name = name

        if best_scheme:
            self._scheme = best_scheme
            self._reload_ui_from_scheme()
            self._refresh_preview()
            theme.make_label(self._left, f"✓ 自动优化: {best_name} ({best_score}分)", fg=theme.FG_ACCENT).pack()

    def _on_save_template(self) -> None:
        """保存当前方案为模板。"""
        from ui.widgets.custom_dialog import ask_string
        name = ask_string(self, "保存模板", "请输入模板名称：")
        if not name or not name.strip():
            return
        name = name.strip()
        self._templates[name] = self._scheme
        self._save_templates()
        theme.make_label(self._left, f"✓ 已保存模板: {name}", fg=theme.FG_ACCENT).pack()

    def _on_reset(self) -> None:
        """重置为默认方案。"""
        self._scheme = self._default_scheme()
        self._reload_ui_from_scheme()
        self._refresh_preview()

    def _reload_ui_from_scheme(self) -> None:
        """从方案回填 UI 控件。"""
        pre = self._scheme["预处理"]
        self._pre_gray.set(pre.get("转灰度", False))
        self._pre_invert.set(pre.get("反相", False))
        self._pre_inknorm.set(pre.get("墨色归一", False))
        self._pre_inkval.set(pre.get("墨色基准", 60))

        for lkey in ("L1", "L2", "L3", "L4", "L5"):
            cfg = self._scheme.get(lkey)
            algo = cfg.get("算法", "") if cfg else ""
            var: tk.StringVar = getattr(self, f"_layer_{lkey}_var", None)
            if var:
                var.set(algo if algo else "(无)")
