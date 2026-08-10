# app_window.py — 主窗口：欢迎页与各功能页面的单窗口切换

import importlib.util
import logging
import os
import sys
import time
import tkinter as tk
from tkinter import filedialog
from typing import Callable, Optional

import config
from data.config_store import load_global_config, save_global_config, set_last_ziku_path
from services.export_service import ExportService
from services.glyph_service import GlyphService
from ui import theme
from ui.consistency_page import ConsistencyPage
from ui.edit_canvas import EditCanvas
from ui.glyph_tree import GlyphTree
from ui.import_wizard import ImportWizard
from ui.lab import LabPage
from ui.optimization_page import OptimizationPage
from ui.text_stats import TextStatsPage
from ui.tool_panel import ToolPanel
from ui.top_toolbar import TopToolbar
from ui.welcome_page import WelcomePage
from ui.widgets.custom_dialog import ask_save_discard_cancel, show_error, show_info
from ui.widgets.dark_helpers import apply_dark_titlebar, set_window_icon


_PERFORMANCE_LOGGER = logging.getLogger("手工审核耗时")


class AppWindow:
    """应用主窗口；任一时刻只显示一个嵌入式功能页面。"""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("字库编辑 V2.0")
        self.root.geometry("1200x800")
        self.root.minsize(920, 640)
        self.root.state("zoomed")
        self.root.configure(bg=theme.BG_MAIN)
        set_window_icon(self.root, config.ICON_FILE)
        self.root.after(0, lambda: apply_dark_titlebar(self.root))

        self._settings = load_global_config()
        self._current_page: Optional[tk.Widget] = None
        self._current_glyph: Optional[GlyphService] = None
        self._current_ziku_name = ""
        self._current_ziku_path = ""
        self._show_welcome()

    def run(self) -> None:
        self.root.mainloop()

    def _replace_page(self, page_factory: Callable[[tk.Widget], tk.Widget]) -> None:
        old_page = self._current_page
        new_page: Optional[tk.Widget] = None
        try:
            new_page = page_factory(self.root)
            new_page.pack(fill=tk.BOTH, expand=True)
        except Exception:
            if new_page is not None:
                new_page.destroy()
            raise
        if old_page is not None:
            old_page.destroy()
        self._current_page = new_page

    def _show_welcome(self) -> None:
        self._current_glyph = None
        self._replace_page(
            lambda parent: WelcomePage(
                parent,
                on_new_ziku=self._show_new_ziku,
                on_open_lab=self._show_lab,
                on_open_stats=self._show_stats,
                on_open_layout=self._show_layout,
                on_open_help=self._show_help,
                on_open_settings=self._show_settings,
                on_open_import=self._show_import,
                on_open_optimization=self._show_optimization,
                on_open_review=self._show_review,
                on_open_consistency=self._show_consistency,
                on_export_ziku=self._show_export,
                initial_ziku_name=self._current_ziku_name,
            )
        )

    def _activate_ziku(self, name: str, path: str) -> GlyphService:
        self._current_ziku_name = name
        self._current_ziku_path = path
        self._current_glyph = GlyphService(name, path)
        set_last_ziku_path(path)
        return self._current_glyph

    def _show_new_ziku(self) -> None:
        def build_new_ziku_page(parent: tk.Widget) -> tk.Widget:
            container = tk.Frame(parent, bg=theme.BG_MAIN)
            ImportWizard(
                container,
                existing_names=[
                    entry.name for entry in os.scandir(config.ZIKU_ROOT) if entry.is_dir()
                ] if os.path.isdir(config.ZIKU_ROOT) else [],
                on_complete=self._on_new_ziku_complete,
            )
            return container

        self._replace_page(build_new_ziku_page)

    def _on_new_ziku_complete(self, name: str) -> None:
        if name:
            path = os.path.join(config.ZIKU_ROOT, name)
            self._activate_ziku(name, path)
        self._show_welcome()

    def _show_import(self, name: str, path: str) -> None:
        glyph = self._activate_ziku(name, path)

        def build_import_page(parent: tk.Widget) -> tk.Widget:
            container = tk.Frame(parent, bg=theme.BG_MAIN)
            ImportWizard(
                container,
                existing_names=[
                    entry.name for entry in os.scandir(config.ZIKU_ROOT) if entry.is_dir()
                ] if os.path.isdir(config.ZIKU_ROOT) else [],
                on_complete=lambda new_name: self._on_append_import_complete(new_name, glyph),
                append_mode=True,
                glyph_service=glyph,
            )
            return container

        self._replace_page(build_import_page)

    def _on_append_import_complete(self, name: str, glyph: GlyphService) -> None:
        if name:
            self._current_ziku_name = name
            self._current_ziku_path = glyph.ziku_dir
            self._current_glyph = glyph
            set_last_ziku_path(glyph.ziku_dir)
        self._show_welcome()

    def _show_optimization(self, name: str, path: str) -> None:
        glyph = self._activate_ziku(name, path)
        self._replace_page(lambda parent: OptimizationPage(parent, glyph, self._show_welcome))

    def _show_review(self, name: str, path: str) -> None:
        glyph = self._activate_ziku(name, path)

        def build(parent: tk.Widget) -> tk.Widget:
            page = tk.Frame(parent, bg=theme.BG_MAIN)
            toolbar_host = tk.Frame(page, bg=theme.BG_PANEL)
            toolbar_host.pack(fill=tk.X)
            workspace = tk.PanedWindow(
                page,
                orient=tk.HORIZONTAL,
                bg=theme.BORDER,
                sashwidth=5,
                bd=0,
                relief=tk.FLAT,
            )
            workspace.pack(fill=tk.BOTH, expand=True)

            tree_host = tk.Frame(workspace, bg=theme.BG_MAIN, width=250)
            canvas_host = tk.Frame(workspace, bg=theme.BG_MAIN)
            tool_host = tk.Frame(workspace, bg=theme.BG_PANEL, width=270)
            workspace.add(tree_host, minsize=250, width=250, stretch="never")
            workspace.add(canvas_host, minsize=420, stretch="always")
            workspace.add(tool_host, minsize=270, width=270, stretch="never")

            state: dict[str, object] = {}

            def refresh_status() -> None:
                tree = state.get("tree")
                tools = state.get("tools")
                canvas = state.get("canvas")
                toolbar = state.get("toolbar")
                if isinstance(tree, GlyphTree):
                    tree.refresh()
                if isinstance(tools, ToolPanel) and isinstance(canvas, EditCanvas):
                    tools.refresh_current(canvas._current_char, canvas._current_variant_index)
                if isinstance(toolbar, TopToolbar):
                    toolbar.refresh_counts()

            def confirm_switch(canvas: EditCanvas) -> bool:
                if not canvas.has_unsaved_changes():
                    return True
                action = ask_save_discard_cancel(
                    self.root,
                    "尚未保存",
                    "当前文字已有修改，是否保存后再切换？",
                )
                if action == "保存":
                    # 切换过程中不刷新并重建左侧树，否则刚点击的目标节点会丢失。
                    return canvas.save_current(notify_status_change=False)
                return action == "不保存"

            def select_char(char: str, variant_index: int) -> None:
                select_start = time.perf_counter()
                _PERFORMANCE_LOGGER.info(
                    "[审核入口] 开始选择：文字=%s，字形序号=%d", char, variant_index + 1
                )
                canvas = state.get("canvas")
                tools = state.get("tools")
                tree = state.get("tree")
                confirm_elapsed = 0.0
                load_elapsed = 0.0
                if isinstance(canvas, EditCanvas):
                    current = (canvas._current_char, canvas._current_variant_index)
                    target = (char, variant_index)
                    confirm_start = time.perf_counter()
                    if current[0] and target != current and not confirm_switch(canvas):
                        if isinstance(tree, GlyphTree):
                            tree.restore_selection(*current)
                        _PERFORMANCE_LOGGER.info(
                            "[审核入口] 用户取消切换：确认耗时=%.2f 毫秒",
                            (time.perf_counter() - confirm_start) * 1000,
                        )
                        return
                    confirm_elapsed = (time.perf_counter() - confirm_start) * 1000
                    load_start = time.perf_counter()
                    canvas.load_char(char, variant_index)
                    load_elapsed = (time.perf_counter() - load_start) * 1000
                tools_start = time.perf_counter()
                if isinstance(tools, ToolPanel):
                    tools.refresh_current(char, variant_index)
                tools_elapsed = (time.perf_counter() - tools_start) * 1000
                _PERFORMANCE_LOGGER.info(
                    "[审核入口] 完成：切换确认=%.2f 毫秒，画布载图=%.2f 毫秒，工具面板刷新=%.2f 毫秒，总耗时=%.2f 毫秒",
                    confirm_elapsed,
                    load_elapsed,
                    tools_elapsed,
                    (time.perf_counter() - select_start) * 1000,
                )

            def select_variant(index: int) -> None:
                select_start = time.perf_counter()
                canvas = state.get("canvas")
                tools = state.get("tools")
                char = canvas._current_char if isinstance(canvas, EditCanvas) else ""
                _PERFORMANCE_LOGGER.info(
                    "[审核入口] 开始切换字形：文字=%s，字形序号=%d", char, index + 1
                )
                confirm_elapsed = 0.0
                load_elapsed = 0.0
                if isinstance(canvas, EditCanvas):
                    confirm_start = time.perf_counter()
                    if index != canvas._current_variant_index and not confirm_switch(canvas):
                        _PERFORMANCE_LOGGER.info(
                            "[审核入口] 用户取消切换字形：确认耗时=%.2f 毫秒",
                            (time.perf_counter() - confirm_start) * 1000,
                        )
                        return
                    confirm_elapsed = (time.perf_counter() - confirm_start) * 1000
                    load_start = time.perf_counter()
                    canvas.load_char(canvas._current_char, index)
                    load_elapsed = (time.perf_counter() - load_start) * 1000
                tools_start = time.perf_counter()
                if isinstance(tools, ToolPanel) and isinstance(canvas, EditCanvas):
                    tools.refresh_current(canvas._current_char, index)
                tools_elapsed = (time.perf_counter() - tools_start) * 1000
                _PERFORMANCE_LOGGER.info(
                    "[审核入口] 字形切换完成：切换确认=%.2f 毫秒，画布载图=%.2f 毫秒，工具面板刷新=%.2f 毫秒，总耗时=%.2f 毫秒",
                    confirm_elapsed,
                    load_elapsed,
                    tools_elapsed,
                    (time.perf_counter() - select_start) * 1000,
                )

            def select_next_visible(char: str, variant_index: int) -> None:
                tree = state.get("tree")
                toolbar = state.get("toolbar")
                if not isinstance(tree, GlyphTree):
                    return
                target = tree.next_visible_after(char, variant_index)
                tree.refresh()
                if isinstance(toolbar, TopToolbar):
                    toolbar.refresh_counts()
                if target:
                    tree.select_variant(*target)

            canvas = EditCanvas(
                canvas_host,
                glyph,
                on_status_change=refresh_status,
                allowed_statuses=(
                    config.STATUS_PENDING_MANUAL_REVIEW,
                    config.STATUS_REVIEWED,
                    config.STATUS_FINISHED,
                ),
            )
            canvas.pack(fill=tk.BOTH, expand=True)
            tree = GlyphTree(
                tree_host,
                glyph,
                on_select=select_char,
                allowed_statuses=(
                    config.STATUS_PENDING_MANUAL_REVIEW,
                    config.STATUS_REVIEWED,
                    config.STATUS_FINISHED,
                ),
                allow_context_menu=False,
                show_score=False,
            )
            tree.pack(fill=tk.BOTH, expand=True)
            tree_width = tree.recommended_width()
            tree_host.configure(width=tree_width)
            workspace.paneconfigure(tree_host, minsize=250, width=tree_width)
            tools = ToolPanel(
                tool_host,
                glyph,
                on_select_variant=select_variant,
                on_revert=lambda: canvas._on_revert(None),
                on_refresh=refresh_status,
                allowed_statuses=(
                    config.STATUS_PENDING_MANUAL_REVIEW,
                    config.STATUS_REVIEWED,
                    config.STATUS_FINISHED,
                ),
                allow_variant_management=False,
                show_variant_thumbnails=False,
                on_save=canvas.save_current,
                on_review_approved=select_next_visible,
                edit_canvas=canvas,
            )
            tools.pack(fill=tk.BOTH, expand=True)
            state.update(tree=tree, canvas=canvas, tools=tools)

            toolbar = TopToolbar(
                toolbar_host,
                glyph,
                on_back=self._show_welcome,
                review_mode=True,
            )
            toolbar.pack(fill=tk.X)
            state["toolbar"] = toolbar
            return page

        self._replace_page(build)

    def _show_consistency(self, name: str, path: str) -> None:
        glyph = self._activate_ziku(name, path)
        self._replace_page(lambda parent: ConsistencyPage(parent, glyph, self._show_welcome))

    def _show_export(self, name: str, path: str) -> None:
        glyph = self._activate_ziku(name, path)

        def build(parent: tk.Widget) -> tk.Widget:
            page = tk.Frame(parent, bg=theme.BG_MAIN)
            self._build_page_header(page, "导出最终成品", self._show_welcome)
            panel = tk.Frame(page, bg=theme.BG_PANEL, highlightthickness=1, highlightbackground=theme.BORDER)
            panel.pack(fill=tk.X, padx=80, pady=60, ipady=28)
            theme.make_label(panel, f"当前字库：{name}", bg=theme.BG_PANEL, font=theme.FONT_TITLE).pack(pady=(0, 8))
            counts = glyph.get_status_counts()
            finished = counts.get(config.STATUS_FINISHED, 0)
            theme.make_label(
                panel,
                f"共 {len(glyph.get_all_chars())} 字 · {glyph.get_total_count()} 个变体 · {finished} 个最终成品可导出",
                bg=theme.BG_PANEL,
                fg=theme.FG_SECONDARY,
            ).pack(pady=(0, 22))
            theme.make_button(panel, "选择目录并导出最终成品", accent=True, command=lambda: self._export_to_folder(glyph, name)).pack()
            return page

        self._replace_page(build)

    def _export_to_folder(self, glyph: GlyphService, name: str) -> None:
        output_dir = filedialog.askdirectory(parent=self.root, title="选择导出目录")
        if not output_dir:
            return
        try:
            result = ExportService(glyph).export(output_dir)
            show_info(
                self.root,
                "导出完成",
                f"已导出 {result['导出']} 个文件，跳过 {result['跳过']} 个，失败 {result['失败']} 个。\n保存位置：{output_dir}",
            )
        except Exception as exc:
            show_error(self.root, "导出失败", str(exc))

    def _show_lab(self) -> None:
        self._replace_page(lambda parent: LabPage(parent, on_close=self._show_welcome))

    def _show_stats(self) -> None:
        self._replace_page(lambda parent: TextStatsPage(parent, on_close=self._show_welcome))

    def _show_layout(self) -> None:
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        module_path = os.path.abspath(os.path.join(project_dir, "..", "paiban", "通用经文_Python.py"))
        if not os.path.isfile(module_path):
            self._replace_page(lambda parent: self._build_placeholder(parent, "经文排版", "未找到经文排版模块。", self._show_welcome))
            return
        try:
            spec = importlib.util.spec_from_file_location("经文排版模块", module_path)
            if spec is None or spec.loader is None:
                raise RuntimeError("无法加载经文排版模块")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        except Exception as exc:
            self._replace_page(lambda parent: self._build_placeholder(parent, "经文排版", f"经文排版模块加载失败：{exc}", self._show_welcome))
            return

        def build(parent: tk.Widget) -> tk.Widget:
            host = tk.Frame(parent, bg=theme.BG_MAIN)
            self._build_page_header(host, "经文排版", self._show_welcome)
            body = tk.Frame(host, bg=theme.BG_MAIN)
            body.pack(fill=tk.BOTH, expand=True)
            adapter = _EmbeddedRoot(body, self.root, self._show_welcome)
            layout_app = module.App.__new__(module.App)
            layout_app.params = module.load_config()
            layout_app._dirty = False
            layout_app.root = adapter
            style = __import__("tkinter.ttk", fromlist=["Style"]).Style()
            style.theme_use("vista" if "vista" in style.theme_names() else "default")
            layout_app._build_buttons()
            layout_app.notebook = __import__("tkinter.ttk", fromlist=["Notebook"]).Notebook(body)
            layout_app.notebook.pack(fill="both", expand=True, padx=10, pady=(8, 2))
            layout_app._build_tab1()
            layout_app._build_tab2()
            layout_app._build_tab3()
            layout_app._build_tab4()
            layout_app.notebook.bind("<<NotebookTabChanged>>", layout_app._on_tab_changed)
            host._layout_app = layout_app  # type: ignore[attr-defined]
            return host

        self._replace_page(build)

    def _show_help(self) -> None:
        content = (
            "一、从首页新建字库或在“字库选择”中选择已有字库。\n\n"
            "二、按字库添加、运行自动优化、手工审核、整体协调的顺序处理。\n\n"
            "三、所有功能都在当前主窗口中运行；点击“返回首页”后再进入其他功能。\n\n"
            "四、审核通过并生成最终成品后，通过“导出最终成品”输出文件。"
        )
        self._replace_page(lambda parent: self._build_text_page(parent, "使用说明", content, self._show_welcome))

    def _show_settings(self) -> None:
        def build(parent: tk.Widget) -> tk.Widget:
            page = tk.Frame(parent, bg=theme.BG_MAIN)
            self._build_page_header(page, "设置", self._show_welcome)
            panel = tk.Frame(page, bg=theme.BG_PANEL, highlightthickness=1, highlightbackground=theme.BORDER)
            panel.pack(fill=tk.X, padx=100, pady=60, ipady=24)
            theme.make_label(panel, "界面主题", bg=theme.BG_PANEL, font=theme.FONT_BOLD).pack(anchor="w", padx=30)
            theme_var = tk.StringVar(value=str(self._settings.get("主题", "深色")))
            combo = __import__("tkinter.ttk", fromlist=["Combobox"]).Combobox(panel, textvariable=theme_var, state="readonly", values=["深色"], width=24)
            combo.pack(anchor="w", padx=30, pady=(8, 24))

            def save() -> None:
                self._settings["主题"] = theme_var.get()
                save_global_config(self._settings)
                show_info(page, "设置", "设置已保存。")

            theme.make_button(panel, "保存设置", accent=True, command=save).pack(anchor="w", padx=30)
            return page
        self._replace_page(build)

    @staticmethod
    def _build_page_header(parent: tk.Widget, title: str, on_back: Callable[[], None]) -> None:
        bar = tk.Frame(parent, bg=theme.BG_PANEL, height=50)
        bar.pack(fill=tk.X)
        theme.make_button(bar, "返回首页", command=on_back).pack(side=tk.LEFT, padx=12, pady=8)
        theme.make_label(bar, title, bg=theme.BG_PANEL, font=theme.FONT_TITLE).pack(side=tk.LEFT, padx=12)

    def _build_placeholder(self, parent: tk.Widget, title: str, message: str, on_back: Callable[[], None]) -> tk.Widget:
        return self._build_text_page(parent, title, message, on_back)

    def _build_text_page(self, parent: tk.Widget, title: str, content: str, on_back: Callable[[], None]) -> tk.Widget:
        page = tk.Frame(parent, bg=theme.BG_MAIN)
        self._build_page_header(page, title, on_back)
        text = tk.Text(page, bg=theme.BG_PANEL, fg=theme.FG_PRIMARY, insertbackground=theme.FG_PRIMARY, relief=tk.FLAT, font=theme.FONT_NORMAL, wrap=tk.WORD, padx=28, pady=24)
        text.pack(fill=tk.BOTH, expand=True, padx=50, pady=40)
        text.insert("1.0", content)
        text.configure(state=tk.DISABLED)
        return page


class _EmbeddedRoot:
    """为原有经文排版界面提供主窗口内嵌容器兼容接口。"""

    def __init__(self, frame: tk.Frame, window: tk.Tk, on_close: Callable[[], None]) -> None:
        self._frame = frame
        self._window = window
        self._on_close = on_close
        self.tk = frame.tk
        self._w = frame._w
        self.children = frame.children
        self.master = frame.master

    def __str__(self) -> str:
        return str(self._frame)

    def destroy(self) -> None:
        self._on_close()

    def title(self, _text: str) -> None:
        return None

    def geometry(self, _value: str) -> None:
        return None

    def minsize(self, width: int, height: int) -> None:
        self._window.minsize(width, height)

    def resizable(self, _width: bool, _height: bool) -> None:
        return None

    def protocol(self, _name: str, _callback: Callable[[], None]) -> None:
        return None

    def mainloop(self) -> None:
        return None
