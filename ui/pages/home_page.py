"""应用首页：字库选择、快捷工具与制作流程入口。"""

from __future__ import annotations

import os
import stat
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QResizeEvent, QShowEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import config
from data.library_database import LIBRARY_DATABASE_FILENAME
from services.glyph_service import GlyphService
from services.library_summary_service import build_library_summary
from utils.batch_observability import ProgressThrottle
from utils.file_utils import (
    is_real_directory,
    pinyin_natural_key,
    resolve_library_directory,
)


@dataclass(frozen=True, slots=True)
class DirectorySignature:
    """目录是否存在及其轻量修改时间戳。"""

    exists: bool
    mtime_ns: int = 0


@dataclass(frozen=True, slots=True)
class FileSignature:
    """文件是否存在及其无需读取内容即可取得的属性。"""

    exists: bool
    mtime_ns: int = 0
    size: int = 0


@dataclass(frozen=True, slots=True)
class LibraryEntrySignature:
    """单个字库中会影响首页阶段统计的轻量属性。"""

    name: str
    directory: DirectorySignature
    database: FileSignature
    database_wal: FileSignature
    stage_directories: tuple[tuple[str, DirectorySignature], ...]


@dataclass(frozen=True, slots=True)
class LibrarySummarySignature:
    """首页摘要缓存的会话内文件系统签名。"""

    root_path: str
    root: DirectorySignature
    libraries: tuple[LibraryEntrySignature, ...]


_SUMMARY_STAGE_DIRECTORY_NAMES: tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            config.DIR_ORIGINAL_FILES,
            "原始文件",
            config.DIR_INTERMEDIATE_FILES,
            "中间文件",
            config.DIR_REVIEWED_FILES,
            "审核文件",
            config.DIR_FINISHED_FILES,
            "成品文件",
        )
    )
)


def _normalized_library_root() -> str:
    """返回适合稳定比较、但不解析链接目标的字库根路径。"""

    try:
        return os.path.normcase(os.path.abspath(os.fspath(config.ZIKU_ROOT)))
    except (OSError, TypeError, ValueError):
        return str(config.ZIKU_ROOT)


def _directory_signature(path: str) -> DirectorySignature:
    try:
        path_stat = os.stat(path)
    except (OSError, TypeError, ValueError):
        return DirectorySignature(False)
    if not stat.S_ISDIR(path_stat.st_mode):
        return DirectorySignature(False)
    return DirectorySignature(True, path_stat.st_mtime_ns)


def _file_signature(path: str) -> FileSignature:
    try:
        path_stat = os.stat(path)
    except (OSError, TypeError, ValueError):
        return FileSignature(False)
    if not stat.S_ISREG(path_stat.st_mode):
        return FileSignature(False)
    return FileSignature(True, path_stat.st_mtime_ns, path_stat.st_size)


def _scan_library_names(root_path: str) -> list[str]:
    try:
        with os.scandir(root_path) as iterator:
            names: list[str] = []
            for entry in iterator:
                try:
                    if entry.is_dir(follow_symlinks=False) and is_real_directory(entry.path):
                        names.append(entry.name)
                except OSError:
                    continue
    except (OSError, TypeError, ValueError):
        return []
    names.sort(key=pinyin_natural_key)
    return names


def scan_library_names() -> list[str]:
    """仅枚举字库根目录的一层名称，不读取 JSON 或阶段图片。"""

    return _scan_library_names(_normalized_library_root())


def library_summary_signature() -> LibrarySummarySignature:
    """生成首页摘要缓存签名，不枚举或读取任何阶段图片。"""

    root_path = _normalized_library_root()
    libraries: list[LibraryEntrySignature] = []
    for library_name in _scan_library_names(root_path):
        library_path = os.path.join(root_path, library_name)
        database_path = os.path.join(library_path, LIBRARY_DATABASE_FILENAME)
        stage_directories = tuple(
            (
                directory_name,
                _directory_signature(
                    os.path.join(library_path, directory_name)
                ),
            )
            for directory_name in _SUMMARY_STAGE_DIRECTORY_NAMES
        )
        libraries.append(
            LibraryEntrySignature(
                name=library_name,
                # SQLite 打开 WAL 时会改变目录时间，目录自身只记录存在性。
                directory=DirectorySignature(os.path.isdir(library_path)),
                database=_file_signature(database_path),
                database_wal=_file_signature(f"{database_path}-wal"),
                stage_directories=stage_directories,
            )
        )
    return LibrarySummarySignature(
        root_path=root_path,
        root=_directory_signature(root_path),
        libraries=tuple(libraries),
    )


@dataclass(frozen=True)
class LibraryScanProgress:
    """首页字库摘要扫描的单调逻辑进度。"""

    phase: str
    library_name: str = ""
    library_index: int = 0
    library_total: int = 0
    glyph_current: int = 0
    glyph_total: int = 0

    UNITS_PER_LIBRARY = 1000

    @property
    def overall_total(self) -> int:
        return max(1, self.library_total * self.UNITS_PER_LIBRARY)

    @property
    def overall_current(self) -> int:
        if self.phase == "complete":
            return self.overall_total
        if self.library_total <= 0 or self.library_index <= 0:
            return 0
        base = (
            min(self.library_total, self.library_index) - 1
        ) * self.UNITS_PER_LIBRARY
        if self.phase != "processing":
            return base
        if self.glyph_total <= 0:
            return min(self.overall_total, base + self.UNITS_PER_LIBRARY)
        fraction = max(0.0, min(1.0, self.glyph_current / self.glyph_total))
        return min(
            self.overall_total,
            base + round(fraction * self.UNITS_PER_LIBRARY),
        )


def scan_library_summaries(
    progress_callback: Callable[[LibraryScanProgress], None] | None = None,
) -> list[dict[str, Any]]:
    """扫描字库目录，生成首页展示所需的完整摘要。"""
    summaries: list[dict[str, Any]] = []
    progress = (
        ProgressThrottle(progress_callback, interval_seconds=0.1)
        if progress_callback is not None
        else None
    )

    def report(
        update: LibraryScanProgress,
        *,
        force: bool = False,
    ) -> None:
        if progress is not None:
            progress.emit(
                update,
                force=force,
                stage=f"{update.library_index}:{update.phase}",
            )

    if not os.path.isdir(config.ZIKU_ROOT):
        report(LibraryScanProgress("complete"), force=True)
        return summaries
    with os.scandir(config.ZIKU_ROOT) as iterator:
        entries = [
            entry
            for entry in iterator
            if entry.is_dir(follow_symlinks=False)
            and resolve_library_directory(
                config.ZIKU_ROOT,
                entry.path,
                expected_name=entry.name,
            )
        ]
    entries.sort(key=lambda entry: pinyin_natural_key(entry.name))
    library_total = len(entries)
    report(
        LibraryScanProgress("discovering", library_total=library_total),
        force=True,
    )
    for library_index, entry in enumerate(entries, start=1):
        report(
            LibraryScanProgress(
                "loading",
                entry.name,
                library_index,
                library_total,
            ),
            force=True,
        )
        try:
            glyph_service = GlyphService.open(entry.name, entry.path)
        except Exception as exc:
            summaries.append(
                {
                    "name": entry.name,
                    "path": entry.path,
                    "characters": 0,
                    "variants": 0,
                    "metadata": {},
                    "data_error": str(exc),
                }
            )
            report(
                LibraryScanProgress(
                    "processing",
                    entry.name,
                    library_index,
                    library_total,
                ),
                force=True,
            )
            continue
        def report_glyph_progress(glyph_current: int, glyph_total: int) -> None:
            report(
                LibraryScanProgress(
                    "processing",
                    entry.name,
                    library_index,
                    library_total,
                    glyph_current,
                    glyph_total,
                ),
                force=glyph_current == glyph_total,
            )
        summaries.append(
            build_library_summary(
                entry.name,
                entry.path,
                glyph_service.get_variants(),
                glyph_service.get_glyph_groups(),
                glyph_service.get_metadata(),
                glyph_service.get_coordination_summary(),
                verify_files=True,
                progress_callback=report_glyph_progress,
            )
        )
    report(
        LibraryScanProgress(
            "complete",
            library_index=library_total,
            library_total=library_total,
        ),
        force=True,
    )
    return summaries


class WatermarkWidget(QWidget):
    """在整个首页背景自适应绘制一遍竖排心经水印。"""

    def _watermark_layout(self, width: int, height: int) -> tuple[QFont, int, int, int, int]:
        """计算可完整容纳全部经文的最大字号及排版尺寸。"""
        title_length = len(config.WELCOME_BG_TITLE)
        body_length = len(config.WELCOME_BG_BODY)
        best: tuple[QFont, int, int, int, int] | None = None
        low = config.WELCOME_MIN_FONT_SIZE
        high = max(low, min(width, height))
        while low <= high:
            pixel_size = (low + high) // 2
            font = QFont(config.WELCOME_FONT_FAMILIES[0])
            font.setPixelSize(pixel_size)
            metrics = QFontMetrics(font)
            line_height = max(1, metrics.height())
            column_width = max(1, metrics.horizontalAdvance("觀"))
            gap = max(2, pixel_size // 5)
            rows = height // line_height
            if rows <= 0 or title_length * line_height > height:
                high = pixel_size - 1
                continue
            body_columns = (body_length + rows - 1) // rows
            used_width = (body_columns + 1) * column_width + body_columns * gap
            if used_width <= width:
                best = (font, line_height, column_width, gap, rows)
                low = pixel_size + 1
            else:
                high = pixel_size - 1

        if best is not None:
            return best
        font = QFont(config.WELCOME_FONT_FAMILIES[0])
        font.setPixelSize(config.WELCOME_MIN_FONT_SIZE)
        metrics = QFontMetrics(font)
        return font, max(1, metrics.height()), max(1, metrics.horizontalAdvance("觀")), 2, max(1, height // max(1, metrics.height()))

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        if self.width() <= 0 or self.height() <= 0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        painter.setPen(QColor("#252b34"))
        font, line_height, column_width, gap, rows = self._watermark_layout(self.width(), self.height())
        painter.setFont(font)

        body = config.WELCOME_BG_BODY
        body_columns = (len(body) + rows - 1) // rows
        total_columns = body_columns + 1
        used_width = total_columns * column_width + body_columns * gap
        start_x = (self.width() + used_width) // 2 - column_width

        title_height = len(config.WELCOME_BG_TITLE) * line_height
        title_y = (self.height() - title_height) // 2
        for row, char in enumerate(config.WELCOME_BG_TITLE):
            painter.drawText(
                start_x,
                title_y + row * line_height,
                column_width,
                line_height,
                Qt.AlignmentFlag.AlignCenter,
                char,
            )

        for column in range(body_columns):
            x = start_x - (column + 1) * (column_width + gap)
            segment = body[column * rows:(column + 1) * rows]
            for row, char in enumerate(segment):
                painter.drawText(
                    x,
                    row * line_height,
                    column_width,
                    line_height,
                    Qt.AlignmentFlag.AlignCenter,
                    char,
                )


class ToolCard(QFrame):
    """首页顶部快捷功能卡片。"""

    clicked = Signal()

    def __init__(self, mark: str, title: str, detail: str, color: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("toolCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(82)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 11, 8, 11)
        layout.setSpacing(6)

        badge = QLabel(mark)
        badge.setObjectName("toolBadge")
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setFixedSize(32, 32)
        badge.setStyleSheet(f"background: {color}; color: #dce9ff; border-radius: 6px; font-weight: 700; font-size: 16px;")

        text_box = QVBoxLayout()
        text_box.setSpacing(2)
        title_label = QLabel(title)
        title_label.setObjectName("cardTitle")
        title_label.setWordWrap(True)
        title_label.setMinimumWidth(0)
        title_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        detail_label = QLabel(detail)
        detail_label.setObjectName("cardDetail")
        detail_label.setWordWrap(True)
        detail_label.setMinimumWidth(0)
        detail_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        text_box.addWidget(title_label)
        text_box.addWidget(detail_label)

        arrow = QLabel("›")
        arrow.setObjectName("cardArrow")
        layout.addWidget(badge)
        layout.addLayout(text_box, 1)
        layout.addWidget(arrow)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class StageCard(QFrame):
    """单个字库制作阶段卡片。"""

    clicked = Signal()

    def __init__(
        self,
        mark: str,
        title: str,
        metric_name: str,
        metric_value: str,
        status: str,
        button_text: str,
        accent: str,
        emphasized: bool = False,
        available: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("exportStageCard" if emphasized else "stageCard")
        self.setProperty("available", available)
        self.setFixedHeight(148)
        self.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Fixed,
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 9)
        layout.setSpacing(4)

        stripe = QFrame()
        stripe.setFixedHeight(4)
        stripe.setStyleSheet(f"background: {accent}; border: 0;")
        layout.addWidget(stripe)

        heading = QHBoxLayout()
        self._mark_label = QLabel(mark)
        self._mark_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._mark_label.setFixedSize(30, 30)
        self._mark_label.setStyleSheet(f"background: {accent}; color: #e4efff; border-radius: 5px; font-weight: 700;")
        title_box = QVBoxLayout()
        title_box.setSpacing(1)
        self._title_label = QLabel(title)
        self._title_label.setObjectName("cardTitle")
        self._status_label = QLabel(status)
        self._status_label.setObjectName("stageStatus")
        title_box.addWidget(self._title_label)
        title_box.addWidget(self._status_label)
        heading.addWidget(self._mark_label)
        heading.addLayout(title_box, 1)
        layout.addLayout(heading)

        metric_label = QLabel(metric_name)
        metric_label.setObjectName("cardDetail")
        value_label = QLabel(metric_value)
        value_label.setObjectName("metricValue")
        layout.addWidget(metric_label)
        layout.addWidget(value_label)
        layout.addStretch()

        self._action_button = QPushButton(button_text)
        self._action_button.setCursor(
            Qt.CursorShape.PointingHandCursor
            if available
            else Qt.CursorShape.ArrowCursor
        )
        self._action_button.setStyleSheet(
            f"QPushButton {{ background: {accent}; border-color: {accent}; }} "
            f"QPushButton:hover {{ background: {accent}; border-color: #7fa8df; }} "
            "QPushButton:disabled { color: #68717e; background: #242a33; "
            "border-color: #303640; }"
        )
        self._action_button.setEnabled(available)
        self._action_button.clicked.connect(self.clicked)
        layout.addWidget(self._action_button)


class HomePage(WatermarkWidget):
    """复刻旧版首页的功能入口、字库表格与阶段流程。"""

    refresh_requested = Signal()
    tool_requested = Signal(str)
    stage_requested = Signal(str, str, str)
    manual_review_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._libraries: list[dict[str, Any]] = []
        self._selected_path = ""
        self._selected_name = ""
        self._tool_cards: dict[str, ToolCard] = {}
        self._stage_cards: dict[str, StageCard] = {}
        self._create_group: QFrame | None = None
        self._table_columns_initialized = False
        self._table_name_user_adjusted = False
        self._table_resize_in_progress = False
        self._table_name_fit_pending = False
        self._loading = False
        self._deleting = False
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setObjectName("homeScroll")
        scroll.setStyleSheet("QScrollArea#homeScroll { background: transparent; } QScrollArea#homeScroll > QWidget > QWidget { background: transparent; }")
        root.addWidget(scroll)

        content = QWidget()
        content.setObjectName("homeContent")
        content.setStyleSheet("QWidget#homeContent { background: transparent; }")
        body = QVBoxLayout(content)
        body.setContentsMargins(54, 30, 54, 34)
        body.setSpacing(13)
        scroll.setWidget(content)
        body.addStretch(1)

        title = QLabel("欢迎使用字库编辑器-V1.0")
        title.setObjectName("homeTitle")
        subtitle = QLabel("请选择功能，或从字库列表中选择项目继续制作流程")
        subtitle.setObjectName("homeSubtitle")
        body.addWidget(title)
        body.addWidget(subtitle)

        tools = QGridLayout()
        tools.setHorizontalSpacing(8)
        tool_defs = (
            ("排", "通用经文排版", "使用字库进行通用经文排版", "#493b66", "layout"),
            ("定", "定制经文排版", "按空行分版和逐列参数生成版面", "#35556A", "custom_layout"),
            ("统", "文字统计", "查看字库与字符统计", "#294d43", "statistics"),
            ("图", "图片实验室", "处理整幅拓片和文字扫描件", "#2d426d", "image_lab"),
            ("设", "设置", "目录、显示和程序设置", "#3d4654", "settings"),
            ("?", "使用说明", "查看操作方法和说明", "#554630", "help"),
        )
        for column, definition in enumerate(tool_defs):
            mark, title_text, detail, color, key = definition
            card = ToolCard(mark, title_text, detail, color)
            card.setProperty("route", key)
            card.clicked.connect(lambda checked=False, route=key: self.tool_requested.emit(route))
            self._tool_cards[key] = card
            tools.addWidget(card, 0, column)
            tools.setColumnStretch(column, 1)
        body.addLayout(tools)

        selector = QFrame()
        selector.setObjectName("selectorPanel")
        selector_layout = QVBoxLayout(selector)
        selector_layout.setContentsMargins(12, 10, 12, 12)
        selector_layout.setSpacing(7)
        selector_header = QHBoxLayout()
        selector_title = QLabel("字库选择")
        selector_title.setObjectName("sectionTitle")
        self._summary_label = QLabel("正在扫描字库…")
        self._summary_label.setObjectName("homeSubtitle")
        self._refresh_button = QPushButton("重新核对")
        self._refresh_button.setToolTip("逐字核对全部字库数据并修正首页统计")
        self._refresh_button.setObjectName("compactButton")
        self._refresh_button.clicked.connect(self.refresh_requested)
        selector_header.addWidget(selector_title)
        selector_header.addStretch()
        selector_header.addWidget(self._summary_label)
        selector_header.addSpacing(8)
        selector_header.addWidget(self._refresh_button)
        selector_layout.addLayout(selector_header)

        self._scan_progress_bar = QProgressBar()
        self._scan_progress_bar.setRange(0, 0)
        self._scan_progress_bar.setTextVisible(True)
        self._scan_progress_bar.setFormat("正在准备字库信息…")
        self._scan_progress_bar.setFixedHeight(22)
        self._scan_progress_bar.hide()
        selector_layout.addWidget(self._scan_progress_bar)

        self._table = QTableWidget(0, 10)
        self._table.setHorizontalHeaderLabels(
            (
                "字库名称",
                "DPI",
                "宽（像素 / 毫米）",
                "高（像素 / 毫米）",
                "总字数",
                "自动优化",
                "手工审核",
                "整体协调",
                "可导出",
                "操作",
            )
        )
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setShowGrid(False)
        self._table.verticalHeader().hide()
        self._table.verticalHeader().setDefaultSectionSize(46)
        self._table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self._table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        header = self._table.horizontalHeader()
        header.setMinimumSectionSize(44)
        for column in range(self._table.columnCount()):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Interactive)
        initial_widths = (
            180, 92, 190, 190, 72, 112, 112, 112, 76, 184,
        )
        for column, width in enumerate(initial_widths):
            self._table.setColumnWidth(column, width)
        header.sectionResized.connect(self._on_table_section_resized)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        self._table.cellDoubleClicked.connect(lambda row, _column: self._open_selected_review(row))
        selector_layout.addWidget(self._table)
        body.addWidget(selector)

        self._flow_group = QFrame()
        self._flow_group.setObjectName("libraryFlowPanel")
        flow_group_layout = QVBoxLayout(self._flow_group)
        flow_group_layout.setContentsMargins(8, 6, 8, 8)
        flow_group_layout.setSpacing(8)

        self._flow_header = QWidget()
        flow_header = QHBoxLayout(self._flow_header)
        flow_header.setContentsMargins(0, 0, 0, 0)
        self._flow_title = QLabel("制作流程")
        self._flow_title.setObjectName("sectionTitle")
        self._flow_tip = QLabel("点击入口将在当前窗口中进入对应功能")
        self._flow_tip.setObjectName("homeSubtitle")
        flow_header.addWidget(self._flow_title)
        flow_header.addStretch()
        flow_header.addWidget(self._flow_tip)
        flow_group_layout.addWidget(self._flow_header)

        self._flow_cards_layout = QGridLayout()
        self._flow_cards_layout.setContentsMargins(0, 0, 0, 0)
        self._flow_cards_layout.setHorizontalSpacing(8)
        for column in range(5):
            self._flow_cards_layout.setColumnStretch(column, 1)
        flow_group_layout.addLayout(self._flow_cards_layout)

        self._stage_layout = QGridLayout()
        self._stage_layout.setContentsMargins(0, 0, 0, 0)
        self._stage_layout.setHorizontalSpacing(8)
        for column in range(6):
            self._stage_layout.setColumnStretch(column, 1)
        self._stage_layout.addWidget(self._flow_group, 0, 1, 1, 5)
        body.addLayout(self._stage_layout)
        body.addStretch(1)
        self._show_stages(None)

    def set_loading(self, loading: bool) -> None:
        self._loading = bool(loading)
        self._refresh_button.setEnabled(not self._loading and not self._deleting)
        if loading:
            self._summary_label.setText("正在准备字库信息…")
            self._summary_label.setToolTip("")
            self._scan_progress_bar.setRange(0, 0)
            self._scan_progress_bar.setFormat("正在查找字库…")
            self._scan_progress_bar.setToolTip("正在查找可用字库")
            self._scan_progress_bar.show()
        elif not self._deleting:
            self._scan_progress_bar.hide()

    def set_deleting(self, deleting: bool, library_name: str = "") -> None:
        """在系统回收站处理大字库期间锁定首页入口并显示活动进度。"""

        self._deleting = bool(deleting)
        enabled = not self._deleting
        self._refresh_button.setEnabled(enabled and not self._loading)
        self._table.setEnabled(enabled)
        self._flow_group.setEnabled(enabled)
        if self._create_group is not None:
            self._create_group.setEnabled(enabled)
        for card in self._tool_cards.values():
            card.setEnabled(enabled)
        if self._deleting:
            name = str(library_name or "当前字库")
            self._summary_label.setText(f"正在删除字库“{name}”")
            self._summary_label.setToolTip("正在将整个字库目录移入系统回收站")
            self._scan_progress_bar.setRange(0, 0)
            self._scan_progress_bar.setFormat(f"正在将字库“{name}”移入回收站…")
            self._scan_progress_bar.setToolTip(
                "Windows 回收站不提供可靠的逐文件百分比，完成后会自动提示。"
            )
            self._scan_progress_bar.show()
            return
        if not self._loading:
            self._scan_progress_bar.hide()
        if self._libraries:
            self._on_selection_changed()
        else:
            self._summary_label.setText("暂无可选择的字库，请先新建字库")
            self._summary_label.setToolTip("")

    def set_delete_progress(self, message: str) -> None:
        """更新删除任务的当前阶段，进度保持为系统活动状态。"""

        if not self._deleting:
            return
        text = str(message or "正在移入回收站…")
        self._scan_progress_bar.setRange(0, 0)
        self._scan_progress_bar.setFormat(text)
        self._scan_progress_bar.setToolTip(text)

    def set_scan_progress(self, progress: LibraryScanProgress) -> None:
        """显示后台扫描的真实字库和字形进度。"""
        if not self._loading or not isinstance(progress, LibraryScanProgress):
            return
        if progress.phase == "discovering":
            self._summary_label.setText("正在准备字库信息…")
            detail = f"发现 {progress.library_total} 个字库，准备核对"
        elif progress.phase == "loading":
            self._summary_label.setText(
                f"正在读取字库 {progress.library_index}/{progress.library_total}"
            )
            detail = f"正在读取：{progress.library_name}"
        elif progress.phase == "processing":
            self._summary_label.setText(
                f"正在核对字库 {progress.library_index}/{progress.library_total}"
            )
            if progress.glyph_total:
                detail = (
                    f"{progress.library_name} · 字形 "
                    f"{progress.glyph_current}/{progress.glyph_total} · %p%"
                )
            else:
                detail = f"{progress.library_name} · 暂无字形 · %p%"
        else:
            self._summary_label.setText("字库信息加载完成")
            detail = "字库信息加载完成 · %p%"
        self._scan_progress_bar.setRange(0, progress.overall_total)
        self._scan_progress_bar.setValue(progress.overall_current)
        self._scan_progress_bar.setFormat(detail)
        self._scan_progress_bar.setToolTip(detail.replace("%p%", ""))

    def set_loading_failed(self) -> None:
        """结束失败的扫描，并恢复可重试状态。"""
        self.set_loading(False)
        self._summary_label.setText("字库信息核对失败，请点击重新核对重试")
        self._summary_label.setToolTip("")

    def set_libraries(self, libraries: list[dict[str, Any]]) -> None:
        self.set_loading(False)
        previous = self._selected_name
        self._libraries = libraries
        self._table.blockSignals(True)
        self._table.setRowCount(len(libraries))
        visible_rows = max(1, min(len(libraries), 5))
        header_height = self._table.horizontalHeader().sizeHint().height()
        row_height = self._table.verticalHeader().defaultSectionSize()
        frame_width = self._table.frameWidth() * 2
        self._table.setFixedHeight(header_height + visible_rows * row_height + frame_width)
        selected_row = -1
        for row, library in enumerate(libraries):
            data_error = str(library.get("data_error", ""))
            metadata = library.get("metadata", {})
            dpi = int(metadata.get("DPI") or metadata.get("分辨率") or 300)
            width_px = int(metadata.get("画布宽") or 250)
            height_px = int(metadata.get("画布高") or 250)
            width_mm = float(metadata.get("成品宽度毫米") or width_px / dpi * 25.4)
            height_mm = float(metadata.get("成品高度毫米") or height_px / dpi * 25.4)
            values = (
                str(library.get("name", "")),
                f"{dpi} DPI",
                f"{width_px} px/{width_mm:.2f} mm",
                f"{height_px} px/{height_mm:.2f} mm",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
                if column in (0, 2, 3):
                    item.setToolTip(value)
                self._table.setItem(row, column, item)
                if data_error:
                    item.setForeground(QColor("#E36A6A"))
                    item.setToolTip(f"数据异常：{data_error}")
            total = int(library.get("variants", 0))
            pending_optimization = int(
                library.get(
                    "pending_optimization",
                    max(0, total - int(library.get("optimized", 0))),
                )
            )
            optimized = int(
                library.get("optimized", max(0, total - pending_optimization))
            )
            reviewed = int(library.get("reviewed", 0))
            pending_review = int(
                library.get("pending_review", max(0, optimized - reviewed))
            )
            coordinated = int(
                library.get(
                    "coordinated",
                    library.get("completed", library.get("finished", 0)),
                )
            )
            pending_coordination = int(
                library.get(
                    "pending_coordination",
                    max(0, reviewed - coordinated),
                )
            )
            export_ready = int(library.get("export_ready", coordinated))
            review_admitted = int(library.get("review_admitted", optimized))
            coordination_admitted = int(
                library.get("coordination_admitted", reviewed)
            )
            review_blocked = max(0, total - review_admitted)
            coordination_blocked = max(
                0,
                total - coordination_admitted,
            )
            stage_counts = (
                (str(total), f"总字数：{total} 个字形（按字形变体统计）"),
                (
                    f"{optimized}/{total}",
                    f"自动优化：已优化 {optimized} / {total}；"
                    f"待优化 {pending_optimization} 个",
                ),
                (
                    f"{reviewed}/{review_admitted}",
                    f"手工审核：已审核 {reviewed} / {review_admitted} 个已进入本阶段字形；"
                    f"待审核 {pending_review} 个，前序未完成 {review_blocked} 个",
                ),
                (
                    f"{coordinated}/{coordination_admitted}",
                    f"整体协调：已协调 {coordinated} / {coordination_admitted} 个已进入本阶段字形；"
                    f"待协调 {pending_coordination} 个，"
                    f"前序未完成 {coordination_blocked} 个",
                ),
                (str(export_ready), f"当前可导出：{export_ready} / {total} 个字形"),
            )
            for column, (value, tooltip) in enumerate(stage_counts, start=4):
                stage_item = QTableWidgetItem(str(value))
                stage_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                stage_item.setToolTip(tooltip)
                self._table.setItem(row, column, stage_item)
            action_box = QWidget()
            action_layout = QHBoxLayout(action_box)
            action_layout.setContentsMargins(3, 3, 3, 3)
            action_layout.setSpacing(10)
            parameter_button = QPushButton("参数修改")
            parameter_button.setObjectName("compactButton")
            parameter_button.setFixedSize(88, 36)
            parameter_button.setEnabled(not data_error)
            if data_error:
                parameter_button.setToolTip("字库数据异常，重新核对或恢复数据后才能修改参数")
            parameter_button.clicked.connect(lambda checked=False, index=row: self._request_library_action("parameters", index))
            delete_button = QPushButton("删除")
            delete_button.setObjectName("dangerCompactButton")
            delete_button.setFixedSize(64, 36)
            delete_button.clicked.connect(lambda checked=False, index=row: self._request_library_action("delete", index))
            action_layout.addWidget(parameter_button)
            action_layout.addWidget(delete_button)
            self._table.setCellWidget(row, 9, action_box)
            if library.get("name") == previous:
                selected_row = row
        self._initialize_table_column_widths()
        self._table.blockSignals(False)

        if not libraries:
            self._selected_name = ""
            self._selected_path = ""
            self._summary_label.setText("暂无可选择的字库，请先新建字库")
            self._summary_label.setToolTip("")
            self._flow_title.setText("制作流程")
            self._show_stages(None)
            return
        self._table.selectRow(selected_row if selected_row >= 0 else 0)
        self._on_selection_changed()
        self._schedule_table_name_column_fit()

    def select_library(self, library_path: str) -> bool:
        """按完整路径选中字库，供新建完成后的首页定位使用。"""

        if not library_path:
            return False
        target = os.path.normcase(os.path.abspath(library_path))
        for row, library in enumerate(self._libraries):
            path = str(library.get("path", ""))
            if path and os.path.normcase(os.path.abspath(path)) == target:
                self._table.setCurrentCell(row, 0)
                self._table.selectRow(row)
                self._on_selection_changed()
                return True
        return False

    def _initialize_table_column_widths(self) -> None:
        """首次载入数据时完整显示关键列，后续刷新保留用户调整。"""
        if self._table_columns_initialized or self._table.rowCount() <= 0:
            return
        header = self._table.horizontalHeader()
        font_metrics = self._table.fontMetrics()
        # 留出少量平台样式余量，避免 Windows 原生表头取整后压住末尾文字。
        cell_padding = 28
        for column in range(1, 9):
            header_item = self._table.horizontalHeaderItem(column)
            required_width = (
                font_metrics.horizontalAdvance(header_item.text()) + cell_padding
                if header_item is not None
                else 0
            )
            for row in range(self._table.rowCount()):
                item = self._table.item(row, column)
                if item is not None:
                    required_width = max(
                        required_width,
                        font_metrics.horizontalAdvance(item.text()) + cell_padding,
                    )
            required_width = max(
                required_width,
                header.sectionSizeHint(column) + 6,
            )
            self._table.setColumnWidth(column, required_width)
        self._table_columns_initialized = True

    def _on_table_section_resized(
        self,
        logical_index: int,
        _old_size: int,
        _new_size: int,
    ) -> None:
        if self._table_resize_in_progress or not self._table_columns_initialized:
            return
        if logical_index == 0:
            self._table_name_user_adjusted = True
            return
        self._schedule_table_name_column_fit()

    def _schedule_table_name_column_fit(self) -> None:
        if (
            not hasattr(self, "_table")
            or not self._libraries
            or not self.isVisible()
            or self._table_name_user_adjusted
            or self._table_name_fit_pending
        ):
            return
        self._table_name_fit_pending = True
        QTimer.singleShot(0, self._fit_table_name_column)

    def _fit_table_name_column(self) -> None:
        self._table_name_fit_pending = False
        if (
            not self._libraries
            or not self.isVisible()
            or self._table_name_user_adjusted
        ):
            return
        header = self._table.horizontalHeader()
        viewport_width = self._table.viewport().width()
        if viewport_width <= 0:
            return
        other_width = sum(
            header.sectionSize(column)
            for column in range(1, self._table.columnCount())
        )
        name_width = max(
            180,
            header.sectionSizeHint(0),
            viewport_width - other_width,
        )
        if header.sectionSize(0) == name_width:
            return
        self._table_resize_in_progress = True
        try:
            self._table.setColumnWidth(0, name_width)
        finally:
            self._table_resize_in_progress = False

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._schedule_table_name_column_fit()

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self._schedule_table_name_column_fit()

    def _on_selection_changed(self) -> None:
        row = self._table.currentRow()
        if not 0 <= row < len(self._libraries):
            return
        library = self._libraries[row]
        self._selected_name = str(library.get("name", ""))
        self._selected_path = str(library.get("path", ""))
        total_chars = int(library.get("characters", 0))
        total = int(library.get("variants", 0))
        self._summary_label.setText(f"当前选择：{self._selected_name}    {total_chars} 字 · {total} 个变体")
        self._summary_label.setToolTip("")
        self._flow_title.setText(f"{self._selected_name} · 制作流程")
        self._show_stages(library)

    def _show_stages(self, library: dict[str, Any] | None) -> None:
        self._clear_stage_layout()
        data_error = str((library or {}).get("data_error", ""))
        has_library = library is not None and not data_error
        library = library or {}
        total = int(library.get("variants", 0))
        pending_optimization = int(
            library.get(
                "pending_optimization",
                max(0, total - int(library.get("optimized", 0))),
            )
        )
        optimized = int(
            library.get("optimized", max(0, total - pending_optimization))
        )
        reviewed = int(library.get("reviewed", 0))
        pending_review = int(
            library.get("pending_review", max(0, optimized - reviewed))
        )
        coordinated = int(
            library.get(
                "coordinated",
                library.get("completed", library.get("finished", 0)),
            )
        )
        pending_coordination = int(
            library.get(
                "pending_coordination",
                max(0, reviewed - coordinated),
            )
        )
        export_ready = int(library.get("export_ready", coordinated))
        review_blocked = max(0, total - int(library.get("review_admitted", optimized)))
        coordination_blocked = max(
            0,
            total - int(library.get("coordination_admitted", reviewed)),
        )
        review_status = (
            f"待审核 {pending_review} · 前序 {review_blocked}"
            if pending_review and review_blocked
            else (
                f"待审核 {pending_review}"
                if pending_review
                else (
                    f"前序待优化 {review_blocked}"
                    if review_blocked
                    else "已审核"
                )
            )
        )
        coordination_status = (
            f"待协调 {pending_coordination} · 前序 {coordination_blocked}"
            if pending_coordination and coordination_blocked
            else (
                f"待协调 {pending_coordination}"
                if pending_coordination
                else (
                    f"前序待审核 {coordination_blocked}"
                    if coordination_blocked
                    else "已协调"
                )
            )
        )
        waiting_status = "数据异常" if data_error else "等待新建字库"
        stages = (
            (
                "create", "0", "新建字库", "新项目", "开始", "随时可用",
                "新建字库", "#315f9a", False, True,
            ),
            (
                "import", "1", "字库添加", "源图总数",
                str(total) if has_library else "--",
                ("已导入" if total else "尚未导入") if has_library else waiting_status,
                "进入字库添加", "#4a618d", False, has_library,
            ),
            (
                "optimization", "2", "自动优化", "已优化",
                f"{optimized} / {total}" if has_library else "-- / --",
                (f"待优化 {pending_optimization}" if pending_optimization else "已优化")
                if has_library else waiting_status,
                "运行自动优化", "#9b7530", False, has_library,
            ),
            (
                "review", "3", "手工审核", "已审核",
                f"{reviewed} / {optimized}" if has_library else "-- / --",
                review_status if has_library else waiting_status,
                "进入手工审核", "#38735f", False, has_library,
            ),
            (
                "consistency", "4", "整体协调", "已协调",
                f"{coordinated} / {reviewed}" if has_library else "-- / --",
                coordination_status if has_library else waiting_status,
                "生成最终成品", "#56637a", False, has_library,
            ),
            (
                "export", "⇩", "导出最终成品", "当前可导出",
                f"{export_ready} 个字形" if has_library else "--",
                ("可导出" if export_ready else "暂无有效最终成品")
                if has_library else waiting_status,
                "进入导出页面", "#315f9a", True, has_library,
            ),
        )
        for column, definition in enumerate(stages):
            route, mark, title, metric_name, metric_value, status, button, accent, emphasized, available = definition
            card = StageCard(
                mark,
                title,
                metric_name,
                metric_value,
                status,
                button,
                accent,
                emphasized,
                available,
            )
            card.setProperty("route", route)
            card.clicked.connect(lambda checked=False, key=route: self._emit_stage(key))
            self._stage_cards[route] = card
            if route == "create":
                create_group = QFrame()
                create_group.setObjectName("libraryCreatePanel")
                create_group.ensurePolished()
                side_margin = max(0, 5 - create_group.frameWidth())
                create_layout = QVBoxLayout(create_group)
                create_layout.setContentsMargins(side_margin, 6, side_margin, 8)
                create_layout.setSpacing(8)
                create_title = QLabel("创建新的字库项目")
                create_title.setObjectName("sectionTitle")
                create_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
                create_layout.addWidget(create_title)
                create_layout.addWidget(card)
                self._create_group = create_group
                self._stage_layout.addWidget(create_group, 0, 0)
            else:
                self._flow_cards_layout.addWidget(card, 0, column - 1)

    def _emit_stage(self, route: str) -> None:
        if route == "create":
            self.tool_requested.emit("create")
            return
        if route == "review":
            self.manual_review_requested.emit(self._selected_path)
        else:
            self.stage_requested.emit(route, self._selected_name, self._selected_path)

    def _clear_stage_layout(self) -> None:
        self._stage_cards = {}
        self._create_group = None
        for index in reversed(range(self._stage_layout.count())):
            item = self._stage_layout.itemAt(index)
            widget = item.widget()
            if widget is None or widget is self._flow_group:
                continue
            self._stage_layout.takeAt(index)
            widget.hide()
            widget.deleteLater()
        while self._flow_cards_layout.count():
            item = self._flow_cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.deleteLater()

    def _select_and_open(self, row: int) -> None:
        self._table.selectRow(row)
        self._open_selected_review(row)

    def _open_selected_review(self, row: int) -> None:
        if 0 <= row < len(self._libraries):
            self.manual_review_requested.emit(str(self._libraries[row].get("path", "")))

    def _request_library_action(self, action: str, row: int) -> None:
        if 0 <= row < len(self._libraries):
            library = self._libraries[row]
            self.stage_requested.emit(action, str(library.get("name", "")), str(library.get("path", "")))
