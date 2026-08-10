"""首页导航与字库概览。"""

from __future__ import annotations

import os
from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

import config
from utils.file_utils import pinyin_natural_key, safe_read_json


@dataclass(frozen=True)
class LibrarySummary:
    """首页显示所需的轻量字库摘要。"""

    name: str
    path: str
    total: int
    reviewed: int
    pending: int
    modified: str


def scan_library_summaries() -> list[LibrarySummary]:
    """扫描本项目字库目录，仅读取摘要数据。"""
    summaries: list[LibrarySummary] = []
    if not os.path.isdir(config.ZIKU_ROOT):
        return summaries
    for entry in os.scandir(config.ZIKU_ROOT):
        if not entry.is_dir():
            continue
        json_path = os.path.join(entry.path, f"{entry.name}.json")
        data = safe_read_json(json_path, default={})
        if not isinstance(data, dict):
            continue
        details = data.get("变体详情", {})
        if not isinstance(details, dict):
            details = {}
        statuses = [detail.get("状态", "") for detail in details.values() if isinstance(detail, dict)]
        reviewed = sum(status in {config.STATUS_REVIEWED, config.STATUS_FINISHED} for status in statuses)
        pending = sum(status == config.STATUS_PENDING_MANUAL_REVIEW for status in statuses)
        modified = str(data.get("元数据", {}).get("最后修改", "未记录"))
        summaries.append(
            LibrarySummary(
                name=entry.name,
                path=entry.path,
                total=len(details),
                reviewed=reviewed,
                pending=pending,
                modified=modified.replace("T", " "),
            )
        )
    return sorted(summaries, key=lambda item: pinyin_natural_key(item.name))


class HomePage(QWidget):
    """应用首页，承载入口导航和字库概览。"""

    create_requested = Signal()
    open_requested = Signal(str)
    manual_review_requested = Signal(str)
    refresh_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cards_layout: QGridLayout
        self._empty_label: QLabel
        self._refresh_button: QPushButton
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(36, 30, 36, 28)
        root.setSpacing(22)

        heading = QHBoxLayout()
        titles = QVBoxLayout()
        title = QLabel("字库编辑器")
        title.setProperty("role", "pageTitle")
        subtitle = QLabel("管理字库工作流，并为手工审核与成品输出提供统一入口")
        subtitle.setProperty("role", "muted")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        heading.addLayout(titles)
        heading.addStretch()

        create_button = QPushButton("新建字库")
        create_button.setProperty("role", "primary")
        create_button.clicked.connect(self.create_requested)
        heading.addWidget(create_button)
        root.addLayout(heading)

        actions = QFrame()
        actions.setProperty("role", "card")
        actions_layout = QHBoxLayout(actions)
        actions_layout.setContentsMargins(22, 18, 22, 18)
        actions_layout.setSpacing(12)
        action_title = QLabel("工作台")
        action_title.setStyleSheet("font-size: 17px; font-weight: 600;")
        actions_layout.addWidget(action_title)
        actions_layout.addStretch()
        review_button = QPushButton("进入手工审核")
        review_button.setEnabled(False)
        review_button.setToolTip("手工审核页面将在下一迁移阶段接入")
        actions_layout.addWidget(review_button)
        export_button = QPushButton("成品导出")
        export_button.setEnabled(False)
        export_button.setToolTip("成品导出页面将在后续阶段接入")
        actions_layout.addWidget(export_button)
        root.addWidget(actions)

        section_heading = QHBoxLayout()
        section_title = QLabel("现有字库")
        section_title.setStyleSheet("font-size: 18px; font-weight: 600;")
        section_heading.addWidget(section_title)
        section_heading.addStretch()
        self._refresh_button = QPushButton("刷新")
        self._refresh_button.clicked.connect(self.refresh_requested)
        section_heading.addWidget(self._refresh_button)
        root.addLayout(section_heading)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        container = QWidget()
        self._cards_layout = QGridLayout(container)
        self._cards_layout.setContentsMargins(0, 0, 8, 0)
        self._cards_layout.setHorizontalSpacing(16)
        self._cards_layout.setVerticalSpacing(16)
        self._cards_layout.setColumnStretch(0, 1)
        self._cards_layout.setColumnStretch(1, 1)
        self._empty_label = QLabel("正在读取字库…")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setProperty("role", "muted")
        self._empty_label.setMinimumHeight(180)
        self._cards_layout.addWidget(self._empty_label, 0, 0, 1, 2)
        scroll.setWidget(container)
        root.addWidget(scroll, 1)

    def set_loading(self, loading: bool) -> None:
        self._refresh_button.setEnabled(not loading)
        if loading and self._cards_layout.count() <= 1:
            self._empty_label.setText("正在读取字库…")

    def show_error(self, message: str) -> None:
        self._clear_cards()
        self._empty_label.setText(f"字库读取失败：{message}")
        self._cards_layout.addWidget(self._empty_label, 0, 0, 1, 2)
        self._refresh_button.setEnabled(True)

    def set_libraries(self, summaries: object) -> None:
        items = list(summaries) if isinstance(summaries, (list, tuple)) else []
        self._clear_cards()
        if not items:
            self._empty_label.setText("暂无字库，请先新建字库。")
            self._cards_layout.addWidget(self._empty_label, 0, 0, 1, 2)
        else:
            for index, summary in enumerate(items):
                if isinstance(summary, LibrarySummary):
                    self._cards_layout.addWidget(self._build_card(summary), index // 2, index % 2)
            self._cards_layout.setRowStretch((len(items) + 1) // 2, 1)
        self._refresh_button.setEnabled(True)

    def _build_card(self, summary: LibrarySummary) -> QFrame:
        card = QFrame()
        card.setProperty("role", "card")
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        title_row = QHBoxLayout()
        name = QLabel(summary.name)
        name.setStyleSheet("font-size: 18px; font-weight: 700;")
        title_row.addWidget(name)
        title_row.addStretch()
        total = QLabel(f"{summary.total} 个字形")
        total.setProperty("role", "muted")
        title_row.addWidget(total)
        layout.addLayout(title_row)

        stats = QLabel(f"待审核 {summary.pending}    已审核/成品 {summary.reviewed}")
        stats.setProperty("role", "muted")
        layout.addWidget(stats)
        modified = QLabel(f"最后修改：{summary.modified}")
        modified.setProperty("role", "muted")
        layout.addWidget(modified)

        buttons = QHBoxLayout()
        buttons.addStretch()
        review = QPushButton("手工审核")
        review.setEnabled(False)
        review.setToolTip("手工审核页面将在下一迁移阶段接入")
        buttons.addWidget(review)
        open_button = QPushButton("打开字库")
        open_button.setProperty("role", "primary")
        open_button.clicked.connect(lambda checked=False, path=summary.path: self.open_requested.emit(path))
        buttons.addWidget(open_button)
        layout.addLayout(buttons)
        return card

    def _clear_cards(self) -> None:
        while self._cards_layout.count():
            item = self._cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
