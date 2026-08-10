# params_editor.py — 注册表驱动的参数编辑器

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Optional

from ui import theme
from core.registry import get_layer_algos


class ParamsEditor(tk.Frame):
    """根据算法注册表的参数定义，动态生成参数编辑控件。

    可在模板工坊和实时调参面板中复用。
    """

    def __init__(
        self,
        parent: tk.Widget,
        layer_name: str,
        algo_name: str,
        params: dict[str, Any],
        on_change: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> None:
        super().__init__(parent, bg=theme.BG_PANEL)
        self._layer_name: str = layer_name
        self._algo_name: str = algo_name
        self._params: dict[str, Any] = dict(params)
        self._on_change: Optional[Callable[[dict[str, Any]], None]] = on_change
        self._widgets: dict[str, Any] = {}

        self._build()

    def _build(self) -> None:
        """根据注册表动态创建控件。"""
        algos = get_layer_algos(self._layer_name)
        algo_def = algos.get(self._algo_name, {})
        param_defs = algo_def.get("参数", {})

        if not param_defs:
            theme.make_label(self, "(无参数)", fg=theme.FG_MUTED).pack(pady=(0, 4))
            return

        for pname, pdef in param_defs.items():
            ptype = pdef.get("类型", "")
            default = pdef.get("默认", 0)
            val_range = pdef.get("范围", None)
            current = self._params.get(pname, default)

            row = tk.Frame(self, bg=theme.BG_PANEL)
            row.pack(fill=tk.X, pady=1)

            theme.make_label(row, pname, width=10, anchor="w").pack(side=tk.LEFT, padx=(0, 6))

            if ptype == "bool":
                var = tk.BooleanVar(value=bool(current))
                cb = tk.Checkbutton(row, variable=var, bg=theme.BG_PANEL, fg=theme.FG_PRIMARY,
                                    selectcolor=theme.BG_INPUT, activebackground=theme.BG_PANEL,
                                    activeforeground=theme.FG_PRIMARY)
                cb.pack(side=tk.LEFT)
                self._widgets[pname] = ("bool", var)
                var.trace_add("write", lambda *a, p=pname: self._on_param_change(p))

            elif ptype in ("int", "float"):
                var = tk.StringVar(value=str(current))
                entry = tk.Entry(row, textvariable=var, width=8,
                                 bg=theme.BG_INPUT, fg=theme.FG_PRIMARY,
                                 insertbackground=theme.FG_PRIMARY, bd=0, font=theme.FONT_CODE)
                entry.pack(side=tk.LEFT)
                self._widgets[pname] = (ptype, var)
                var.trace_add("write", lambda *a, p=pname: self._on_param_change(p))

            elif ptype == "str":
                var = tk.StringVar(value=str(current))
                entry = tk.Entry(row, textvariable=var, width=16,
                                 bg=theme.BG_INPUT, fg=theme.FG_PRIMARY,
                                 insertbackground=theme.FG_PRIMARY, bd=0, font=theme.FONT_CODE)
                entry.pack(side=tk.LEFT)
                self._widgets[pname] = ("str", var)
                var.trace_add("write", lambda *a, p=pname: self._on_param_change(p))

    def _on_param_change(self, pname: str) -> None:
        if self._on_change:
            self._on_change(self.get_values())

    def get_values(self) -> dict[str, Any]:
        """获取当前所有参数值。"""
        result = {}
        for pname, (ptype, var) in self._widgets.items():
            val_str = str(var.get())
            if ptype == "bool":
                result[pname] = bool(var.get())
            elif ptype == "int":
                try:
                    result[pname] = int(float(val_str))
                except ValueError:
                    result[pname] = 0
            elif ptype == "float":
                try:
                    result[pname] = float(val_str)
                except ValueError:
                    result[pname] = 0.0
            else:
                result[pname] = val_str
        return result

    def set_values(self, params: dict[str, Any]) -> None:
        """批量设置参数值。"""
        for pname, val in params.items():
            if pname in self._widgets:
                ptype, var = self._widgets[pname]
                var.set(str(val))
