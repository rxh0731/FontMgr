"""尚未迁移功能的统一占位页面。"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget


class PlaceholderPage(QWidget):
    """为后续页面迁移保留稳定导航位置。"""

    home_requested = Signal()

    def __init__(self, title: str, description: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 36, 40, 36)
        layout.setSpacing(14)
        title_label = QLabel(title)
        title_label.setProperty("role", "pageTitle")
        description_label = QLabel(description)
        description_label.setProperty("role", "muted")
        description_label.setWordWrap(True)
        home_button = QPushButton("返回首页")
        home_button.clicked.connect(self.home_requested)
        layout.addWidget(title_label)
        layout.addWidget(description_label)
        layout.addSpacing(10)
        layout.addWidget(home_button, 0)
        layout.addStretch()
