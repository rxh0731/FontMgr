"""字形列表双行状态绘制回归测试。"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSize
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QTreeWidget, QTreeWidgetItem

from ui.theme import apply_theme
from ui.widgets.two_line_status_delegate import (
    STATUS_LINE_GAP,
    STATUS_VERTICAL_MARGIN,
    TwoLineStatusDelegate,
    set_two_line_status,
)


class TwoLineStatusDelegateTests(unittest.TestCase):
    """确保主状态和提示不会因行高不足被裁掉。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])
        apply_theme(cls.app)

    def test_parent_and_thumbnail_rows_reserve_complete_two_line_height(self) -> None:
        tree = QTreeWidget()
        tree.setColumnCount(2)
        tree.setHeaderLabels(("字形与文件", "状态与提示"))
        tree.setItemDelegateForColumn(1, TwoLineStatusDelegate(tree))
        parent = QTreeWidgetItem(["阿（4个字形）", ""])
        set_two_line_status(parent, 1, "已协调 4/4", "问题 0", "#228B22")
        tree.addTopLevelItem(parent)
        child = QTreeWidgetItem(parent, ["字形1 · 阿-0001.png", ""])
        child.setSizeHint(0, QSize(0, 52))
        set_two_line_status(child, 1, "已协调", "无", "#228B22")
        parent.setExpanded(True)
        tree.resize(300, 240)
        tree.show()
        self.app.processEvents()

        metrics = tree.fontMetrics()
        minimum_height = (
            metrics.lineSpacing() * 2
            + STATUS_LINE_GAP
            + STATUS_VERTICAL_MARGIN * 2
        )
        self.assertGreaterEqual(tree.visualItemRect(parent).height(), minimum_height)
        self.assertGreaterEqual(tree.visualItemRect(child).height(), 52)

        rendered = tree.grab()
        self.assertFalse(rendered.isNull())

        tree.close()
        tree.deleteLater()


if __name__ == "__main__":
    unittest.main()
