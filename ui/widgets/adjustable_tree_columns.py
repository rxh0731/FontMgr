"""字形树的可调列宽与关键列内容保护。"""

from __future__ import annotations

from collections.abc import Mapping

from PySide6.QtCore import QObject, QSignalBlocker
from PySide6.QtWidgets import QHeaderView, QTreeWidget


class AdjustableTreeColumns(QObject):
    """让所有列可调整，并阻止关键列缩到无法显示主要信息。"""

    def __init__(
        self,
        tree: QTreeWidget,
        protected_minimums: Mapping[int, int],
        initial_widths: Mapping[int, int] | None = None,
    ) -> None:
        super().__init__(tree)
        self._tree = tree
        self._minimums = {
            int(column): max(1, int(width))
            for column, width in protected_minimums.items()
        }
        header = tree.header()
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(48)
        for column in range(tree.columnCount()):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Interactive)
        for column, width in (initial_widths or {}).items():
            header.resizeSection(int(column), max(1, int(width)))
        for column, width in self._minimums.items():
            if header.sectionSize(column) < width:
                header.resizeSection(column, width)
        header.sectionResized.connect(self._enforce_minimum)

    def set_protected_minimum(self, column: int, width: int) -> None:
        """更新关键列下限；只在现有宽度不足时扩宽，不覆盖用户加宽。"""
        normalized = max(
            self._minimums.get(int(column), 1),
            int(width),
        )
        self._minimums[int(column)] = normalized
        header = self._tree.header()
        if header.sectionSize(int(column)) < normalized:
            with QSignalBlocker(header):
                header.resizeSection(int(column), normalized)

    def _enforce_minimum(
        self,
        logical_index: int,
        _old_size: int,
        new_size: int,
    ) -> None:
        minimum = self._minimums.get(logical_index)
        if minimum is None or new_size >= minimum:
            return
        header = self._tree.header()
        with QSignalBlocker(header):
            header.resizeSection(logical_index, minimum)
