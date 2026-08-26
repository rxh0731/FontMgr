"""通用经文排版三栏工作台。"""

from __future__ import annotations

import os
import re
import threading
import time
from collections import Counter
from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL.ImageQt import ImageQt
from PySide6.QtCore import (
    QEvent,
    QPoint,
    QPointF,
    QRect,
    QSize,
    QSignalBlocker,
    QStandardPaths,
    Qt,
    QThreadPool,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QAction,
    QColor,
    QContextMenuEvent,
    QImage,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QResizeEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import config
from core.scripture_layout import (
    FLOW_LEFT_TO_RIGHT,
    FLOW_RIGHT_TO_LEFT,
    LAYOUT_HORIZONTAL,
    LAYOUT_VERTICAL,
    PARAGRAPH_NEW_COLUMN,
    PARAGRAPH_SKIP_CELLS,
    SCALE_BY_DPI,
    SCALE_TO_CELL,
    BoardLayout,
    ColumnGapAdjustment,
    LayoutParameters,
    RowGapAdjustment,
    ParsedScripture,
    allocate_boards,
    canvas_size_mm,
    compute_grid,
    parse_scripture,
)
from data.library_database import LIBRARY_DATABASE_FILENAME
from data.layout_template_store import (
    DEFAULT_TEMPLATE_ID,
    LayoutTemplate,
    LayoutTemplateStore,
)
from services.glyph_service import GlyphService
from services.scripture_layout_service import (
    CONFLICT_CANCEL,
    CONFLICT_OVERWRITE,
    CONFLICT_SKIP,
    OUTPUT_FORMAT_AUTO,
    OUTPUT_FORMAT_PSB,
    OUTPUT_FORMAT_PSD,
    BoardOutputPlan,
    GenerationCancelled,
    GenerationProgress,
    GenerationResult,
    GlyphIndex,
    board_output_path,
    build_external_glyph_index,
    build_system_glyph_index,
    generate_psd_boards,
    plan_board_output,
    render_board_preview,
)
from services.settings_service import SettingsService
from services.scripture_text_service import (
    ScriptureTextResult,
    load_scripture_text,
    scripture_document_filter,
)
from ui.workers import FunctionWorker
from utils.batch_observability import format_elapsed_time
from utils.file_utils import natural_key


CUSTOM_TEMPLATE_ID = "custom:current"


@dataclass(frozen=True, slots=True)
class _GlyphSourceLoadResult:
    index: GlyphIndex
    scripture_text: str
    include_punctuation: bool
    parsed: ParsedScripture


class _NoWheelSpinBox(QSpinBox):
    """忽略滚轮，避免滚动参数页时意外改值。"""

    def wheelEvent(self, event: QWheelEvent) -> None:
        event.ignore()


class _NoWheelDoubleSpinBox(QDoubleSpinBox):
    """忽略滚轮，避免滚动参数页时意外改值。"""

    def wheelEvent(self, event: QWheelEvent) -> None:
        event.ignore()


class _NoWheelComboBox(QComboBox):
    """忽略组合框本身的滚轮事件，避免未展开时误切换选项。"""

    def wheelEvent(self, event: QWheelEvent) -> None:
        event.ignore()


class _ChinesePlainTextEdit(QPlainTextEdit):
    """提供不依赖系统语言环境的中文编辑菜单。"""

    def createStandardContextMenu(self) -> QMenu:
        menu = QMenu(self)
        cursor = self.textCursor()
        has_selection = cursor.hasSelection()

        def add_action(
            text: str,
            callback: Callable[[], None],
            shortcut: QKeySequence.StandardKey | None = None,
            *,
            enabled: bool = True,
        ) -> QAction:
            action = menu.addAction(text)
            action.triggered.connect(callback)
            action.setEnabled(enabled)
            if shortcut is not None:
                action.setShortcuts(shortcut)
            return action

        document = self.document()
        add_action(
            "撤销",
            self.undo,
            QKeySequence.StandardKey.Undo,
            enabled=document.isUndoAvailable(),
        )
        add_action(
            "重做",
            self.redo,
            QKeySequence.StandardKey.Redo,
            enabled=document.isRedoAvailable(),
        )
        menu.addSeparator()
        add_action(
            "剪切",
            self.cut,
            QKeySequence.StandardKey.Cut,
            enabled=has_selection and not self.isReadOnly(),
        )
        add_action(
            "复制",
            self.copy,
            QKeySequence.StandardKey.Copy,
            enabled=has_selection,
        )
        add_action(
            "粘贴",
            self.paste,
            QKeySequence.StandardKey.Paste,
            enabled=self.canPaste() and not self.isReadOnly(),
        )

        def remove_selection() -> None:
            selected = self.textCursor()
            if selected.hasSelection():
                selected.removeSelectedText()

        add_action("删除", remove_selection, enabled=has_selection and not self.isReadOnly())
        menu.addSeparator()
        add_action("全选", self.selectAll, QKeySequence.StandardKey.SelectAll)
        return menu

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:
        menu = self.createStandardContextMenu()
        menu.exec(event.globalPos())
        menu.deleteLater()


class _ProportionalSplitter(QSplitter):
    """窗口缩放时维持三栏比例，同时记住用户拖动后的新宽度。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent)
        self._left_share = 300 / (300 + 620)
        self._right_width = 480
        self._adjusting_sizes = False
        self.splitterMoved.connect(self._remember_user_sizes)

    def set_preferred_sizes(self, left: int, middle: int, right: int) -> None:
        if left + middle > 0:
            self._left_share = left / (left + middle)
        self._right_width = right
        self._set_sizes([left, middle, right])

    def _set_sizes(self, sizes: list[int]) -> None:
        self._adjusting_sizes = True
        try:
            self.setSizes(sizes)
        finally:
            self._adjusting_sizes = False

    @Slot(int, int)
    def _remember_user_sizes(self, _position: int, _index: int) -> None:
        if self._adjusting_sizes:
            return
        sizes = self.sizes()
        if len(sizes) != 3:
            return
        left, middle, right = sizes
        if left + middle > 0:
            self._left_share = left / (left + middle)
        self._right_width = right

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._adjusting_sizes or self.count() != 3:
            return
        sizes = self.sizes()
        total = sum(sizes)
        if total <= 0:
            return
        right = min(self._right_width, total)
        flexible = max(0, total - right)
        left = round(flexible * self._left_share)
        self._set_sizes([left, flexible - left, right])


class _PreviewScrollArea(QScrollArea):
    """支持滚轮缩放和抓手平移的版面预览视窗。"""

    zoom_requested = Signal(object, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._drag_position: QPointF | None = None
        self._drag_button = Qt.MouseButton.NoButton
        self.viewport().installEventFilter(self)
        self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)

    def eventFilter(self, watched: object, event: QEvent) -> bool:
        if watched is not self.viewport():
            return super().eventFilter(watched, event)
        if event.type() == QEvent.Type.Wheel:
            wheel_event = event
            if not isinstance(wheel_event, QWheelEvent):
                return False
            delta = wheel_event.angleDelta().y()
            if not delta:
                delta = wheel_event.pixelDelta().y() * 8
            if delta:
                factor = 1.15 ** (delta / 120.0)
                self.zoom_requested.emit(QPointF(wheel_event.position()), factor)
                wheel_event.accept()
                return True
        if event.type() == QEvent.Type.MouseButtonPress:
            mouse_event = event
            if (
                isinstance(mouse_event, QMouseEvent)
                and mouse_event.button()
                in {Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton}
            ):
                self._drag_position = QPointF(mouse_event.position())
                self._drag_button = mouse_event.button()
                self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                mouse_event.accept()
                return True
        if event.type() == QEvent.Type.MouseMove and self._drag_position is not None:
            mouse_event = event
            if not isinstance(mouse_event, QMouseEvent):
                return False
            delta = QPointF(mouse_event.position()) - self._drag_position
            self._drag_position = QPointF(mouse_event.position())
            horizontal = self.horizontalScrollBar()
            vertical = self.verticalScrollBar()
            horizontal.setValue(horizontal.value() - round(delta.x()))
            vertical.setValue(vertical.value() - round(delta.y()))
            mouse_event.accept()
            return True
        if event.type() == QEvent.Type.MouseButtonRelease:
            mouse_event = event
            if (
                isinstance(mouse_event, QMouseEvent)
                and self._drag_position is not None
                and mouse_event.button() == self._drag_button
            ):
                self._drag_position = None
                self._drag_button = Qt.MouseButton.NoButton
                self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
                mouse_event.accept()
                return True
        if event.type() in {QEvent.Type.Hide, QEvent.Type.UngrabMouse}:
            self._drag_position = None
            self._drag_button = Qt.MouseButton.NoButton
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
        return super().eventFilter(watched, event)


class _PreviewCanvas(QWidget):
    """只绘制当前脏区的持久预览画布，缩放时不复制整张大图。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image = QImage()
        self._message = "请先选择字图来源并输入经文"
        self.setMinimumSize(320, 320)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

    def set_image(self, image: QImage) -> None:
        self._image = image
        self._message = ""
        self.setMinimumSize(1, 1)
        self.update()

    def set_message(self, message: str) -> None:
        self._image = QImage()
        self._message = message
        self.setMinimumSize(320, 320)
        self.resize(max(320, self.width()), max(320, self.height()))
        self.update()

    @staticmethod
    def _aspect_fit_rect(container: QSize, image: QSize) -> QRect:
        """返回在容器内等比居中的绘制区域。"""

        if container.isEmpty() or image.isEmpty():
            return QRect()
        target = QSize(image)
        target.scale(container, Qt.AspectRatioMode.KeepAspectRatio)
        return QRect(
            (container.width() - target.width()) // 2,
            (container.height() - target.height()) // 2,
            target.width(),
            target.height(),
        )

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setClipRegion(event.region())
        if self._image.isNull():
            painter.fillRect(self.rect(), QColor("#151A21"))
            painter.setPen(QColor("#B9C3CF"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                self._message,
            )
            return
        painter.fillRect(self.rect(), QColor("#10141A"))
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawImage(
            self._aspect_fit_rect(self.size(), self._image.size()),
            self._image,
        )


class _TemplatePreviewCanvas(QWidget):
    """绘制模板的轻量版面示意，不生成或缓存正式预览图。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._parameters: LayoutParameters | None = None
        self.setMinimumSize(280, 210)

    def set_parameters(self, parameters: LayoutParameters | None) -> None:
        self._parameters = parameters
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setClipRegion(event.region())
        painter.fillRect(self.rect(), QColor("#151A21"))
        if self._parameters is None:
            painter.setPen(QColor("#B9C3CF"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "请选择模板")
            return
        width_mm, height_mm = canvas_size_mm(self._parameters)
        available = self.rect().adjusted(28, 18, -28, -18)
        if width_mm <= 0 or height_mm <= 0 or available.isEmpty():
            return
        scale = min(available.width() / width_mm, available.height() / height_mm)
        page_width = max(1, round(width_mm * scale))
        page_height = max(1, round(height_mm * scale))
        page = QRect(
            available.center().x() - page_width // 2,
            available.center().y() - page_height // 2,
            page_width,
            page_height,
        )
        painter.fillRect(page, QColor("#F8F8F6"))
        painter.setPen(QColor("#30343A"))
        painter.drawRect(page.adjusted(0, 0, -1, -1))

        rows = min(self._parameters.rows, 24)
        columns = min(self._parameters.columns, 24)
        grid = page.adjusted(
            max(2, round(page.width() * 0.08)),
            max(2, round(page.height() * 0.08)),
            -max(2, round(page.width() * 0.08)),
            -max(2, round(page.height() * 0.08)),
        )
        painter.setPen(QColor("#D5A6A6"))
        for index in range(rows + 1):
            y = grid.top() + round(grid.height() * index / max(1, rows))
            painter.drawLine(grid.left(), y, grid.right(), y)
        for index in range(columns + 1):
            x = grid.left() + round(grid.width() * index / max(1, columns))
            painter.drawLine(x, grid.top(), x, grid.bottom())
        if self._parameters.draw_outer_frame:
            painter.setPen(QColor("#8C5555"))
            painter.drawRect(grid)


class _TemplateDetailsDialog(QDialog):
    """收集模板名称和可选说明。"""

    def __init__(
        self,
        title: str,
        *,
        name: str = "",
        description: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(460)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name_edit = QLineEdit(name)
        self.name_edit.setObjectName("templateNameEdit")
        self.name_edit.setMaxLength(40)
        self.description_edit = QPlainTextEdit(description)
        self.description_edit.setObjectName("templateDescriptionEdit")
        self.description_edit.setPlaceholderText("可填写适用场景、纸张或排版用途")
        self.description_edit.setMaximumHeight(110)
        form.addRow("模板名称：", self.name_edit)
        form.addRow("模板说明：", self.description_edit)
        layout.addLayout(form)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        save_button = QPushButton("保存")
        save_button.setDefault(True)
        save_button.clicked.connect(self.accept)
        cancel_button = QPushButton("取消")
        cancel_button.clicked.connect(self.reject)
        buttons.addWidget(save_button)
        buttons.addWidget(cancel_button)
        layout.addLayout(buttons)
        self.name_edit.selectAll()
        self.name_edit.setFocus()

    def values(self) -> tuple[str, str]:
        return (
            self.name_edit.text().strip(),
            self.description_edit.toPlainText().strip(),
        )


class _TemplateSaveDialog(QDialog):
    """将当前自定义参数保存为新模板或替换用户模板。"""

    MODE_NEW = "new"
    MODE_REPLACE = "replace"

    def __init__(
        self,
        store: LayoutTemplateStore,
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("保存版面模板")
        self.setModal(True)
        self.setMinimumWidth(500)
        self._store = store
        self._user_templates = tuple(
            template for template in store.list_templates() if not template.builtin
        )

        root = QVBoxLayout(self)
        mode_row = QHBoxLayout()
        self.new_radio = QRadioButton("保存为新模板")
        self.new_radio.setObjectName("saveAsNewTemplateRadio")
        self.replace_radio = QRadioButton("替换现有模板")
        self.replace_radio.setObjectName("replaceTemplateRadio")
        self.new_radio.setChecked(True)
        mode_group = QButtonGroup(self)
        mode_group.addButton(self.new_radio)
        mode_group.addButton(self.replace_radio)
        mode_row.addWidget(self.new_radio)
        mode_row.addWidget(self.replace_radio)
        mode_row.addStretch(1)
        root.addLayout(mode_row)

        form = QFormLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setObjectName("newTemplateNameEdit")
        self.name_edit.setMaxLength(40)
        self.name_edit.setPlaceholderText("输入新的模板名称")
        self.replace_combo = QComboBox()
        self.replace_combo.setObjectName("replaceTemplateCombo")
        for template in self._user_templates:
            self.replace_combo.addItem(template.name, template.template_id)
        self.description_edit = QPlainTextEdit()
        self.description_edit.setObjectName("savedTemplateDescriptionEdit")
        self.description_edit.setPlaceholderText("可填写模板用途或适用版面")
        self.description_edit.setMaximumHeight(100)
        form.addRow("新模板名称：", self.name_edit)
        form.addRow("替换模板：", self.replace_combo)
        form.addRow("模板说明：", self.description_edit)
        root.addLayout(form)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        save_button = QPushButton("保存模板")
        save_button.setDefault(True)
        save_button.clicked.connect(self.accept)
        cancel_button = QPushButton("取消")
        cancel_button.clicked.connect(self.reject)
        buttons.addWidget(save_button)
        buttons.addWidget(cancel_button)
        root.addLayout(buttons)

        self.replace_radio.setEnabled(bool(self._user_templates))
        self.new_radio.toggled.connect(self._sync_mode)
        self.replace_combo.currentIndexChanged.connect(self._load_description)
        self._sync_mode()

    def _sync_mode(self) -> None:
        use_new = self.new_radio.isChecked()
        self.name_edit.setEnabled(use_new)
        self.replace_combo.setEnabled(not use_new and bool(self._user_templates))
        if use_new:
            self.description_edit.clear()
            self.name_edit.setFocus()
        else:
            self._load_description()

    def _load_description(self) -> None:
        template_id = str(self.replace_combo.currentData() or "")
        try:
            description = self._store.get(template_id).description
        except KeyError:
            description = ""
        self.description_edit.setPlainText(description)

    def mode(self) -> str:
        return self.MODE_NEW if self.new_radio.isChecked() else self.MODE_REPLACE

    def new_name(self) -> str:
        return self.name_edit.text().strip()

    def replacement_id(self) -> str:
        return str(self.replace_combo.currentData() or "")

    def description(self) -> str:
        return self.description_edit.toPlainText().strip()


def _template_summary(template: LayoutTemplate) -> str:
    parameters = template.parameters
    width_mm, height_mm = canvas_size_mm(parameters)
    if parameters.scale_mode == SCALE_BY_DPI:
        scale = f"按源图尺寸，{parameters.scale_percent}%"
        if parameters.auto_scale_enabled:
            scale += "，启用自动缩放"
    else:
        scale = f"相对单元格，填充 {parameters.cell_fill_percent}%"
    if parameters.paragraph_mode == PARAGRAPH_SKIP_CELLS:
        paragraph = f"段后跳 {parameters.paragraph_skip_cells} 格"
    else:
        paragraph = "段后换列" if parameters.layout_mode == LAYOUT_VERTICAL else "段后换行"
    type_text = "内置模板" if template.builtin else "用户模板"
    lines = [
        f"类型：{type_text}",
        f"画布：{width_mm:.2f} × {height_mm:.2f} mm，{parameters.dpi} DPI",
        f"版面：{parameters.rows} 行 × {parameters.columns} 列",
        (
            f"单元格：{parameters.cell_width_mm:g} × "
            f"{parameters.cell_height_mm:g} mm"
        ),
        f"方向：{parameters.layout_mode}，{parameters.flow_direction}",
        f"文字：{scale}",
        f"段落：{paragraph}",
        "其他："
        + "、".join(
            (
                "包含标点" if parameters.include_punctuation else "不含标点",
                "末版压缩" if parameters.trim_empty_columns else "末版保留空位",
                "尺寸标注" if parameters.add_annotations else "无尺寸标注",
            )
        ),
    ]
    if template.description:
        lines.extend(("", f"说明：{template.description}"))
    if template.updated_at:
        lines.extend(("", f"修改时间：{template.updated_at}"))
    return "\n".join(lines)


class _TemplateManagerDialog(QDialog):
    """集中浏览、整理和交换通用经文排版模板。"""

    def __init__(
        self,
        store: LayoutTemplateStore,
        *,
        current_template_id: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("版面模板管理")
        self.setModal(True)
        self.setMinimumSize(900, 580)
        self._store = store
        self._current_template_id = current_template_id
        self.selected_template_id = ""
        self.changed = False
        self._last_directory = str(Path(store.file_path).parent)

        root = QVBoxLayout(self)
        filters = QHBoxLayout()
        filters.addWidget(QLabel("搜索模板："))
        self._search_edit = QLineEdit()
        self._search_edit.setObjectName("templateSearchEdit")
        self._search_edit.setClearButtonEnabled(True)
        self._search_edit.textChanged.connect(lambda _text: self._reload())
        filters.addWidget(self._search_edit, 1)
        self._filter_combo = QComboBox()
        self._filter_combo.setObjectName("templateTypeFilter")
        self._filter_combo.addItems(("全部模板", "内置模板", "我的模板"))
        self._filter_combo.currentIndexChanged.connect(
            lambda _index: self._reload()
        )
        filters.addWidget(self._filter_combo)
        root.addLayout(filters)

        content = QHBoxLayout()
        self._template_list = QListWidget()
        self._template_list.setObjectName("templateManagerList")
        self._template_list.setMinimumWidth(260)
        self._template_list.currentItemChanged.connect(self._selection_changed)
        content.addWidget(self._template_list, 0)
        details = QVBoxLayout()
        self._name_label = QLabel("请选择模板")
        self._name_label.setObjectName("sectionTitle")
        details.addWidget(self._name_label)
        self._preview = _TemplatePreviewCanvas()
        self._preview.setObjectName("templatePreviewCanvas")
        details.addWidget(self._preview, 1)
        self._summary = QPlainTextEdit()
        self._summary.setObjectName("templateSummaryEdit")
        self._summary.setReadOnly(True)
        self._summary.setMinimumHeight(190)
        details.addWidget(self._summary)
        content.addLayout(details, 1)
        root.addLayout(content, 1)

        actions = QHBoxLayout()
        self._apply_button = QPushButton("应用")
        self._apply_button.clicked.connect(self._apply_current)
        self._copy_button = QPushButton("复制")
        self._copy_button.clicked.connect(self._copy_current)
        self._rename_button = QPushButton("编辑信息")
        self._rename_button.clicked.connect(self._rename_current)
        self._delete_button = QPushButton("删除")
        self._delete_button.clicked.connect(self._delete_current)
        self._import_button = QPushButton("导入")
        self._import_button.clicked.connect(self._import_template)
        self._export_button = QPushButton("导出")
        self._export_button.clicked.connect(self._export_current)
        for button in (
            self._apply_button,
            self._copy_button,
            self._rename_button,
            self._delete_button,
            self._import_button,
            self._export_button,
        ):
            actions.addWidget(button)
        actions.addStretch(1)
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.reject)
        actions.addWidget(close_button)
        root.addLayout(actions)
        self._reload(current_template_id)

    def _selected_template(self) -> LayoutTemplate | None:
        item = self._template_list.currentItem()
        if item is None:
            return None
        try:
            return self._store.get(str(item.data(Qt.ItemDataRole.UserRole)))
        except KeyError:
            return None

    def _reload(self, selected_id: str = "") -> None:
        if isinstance(selected_id, bool):
            selected_id = ""
        previous = selected_id or (
            str(self._template_list.currentItem().data(Qt.ItemDataRole.UserRole))
            if self._template_list.currentItem() is not None
            else ""
        )
        search = self._search_edit.text().strip().lower()
        filter_index = self._filter_combo.currentIndex()
        with QSignalBlocker(self._template_list):
            self._template_list.clear()
            target_row = -1
            for template in self._store.list_templates():
                if search and search not in template.name.lower() and search not in template.description.lower():
                    continue
                if filter_index == 1 and not template.builtin:
                    continue
                if filter_index == 2 and template.builtin:
                    continue
                type_text = "内置" if template.builtin else "我的模板"
                item = QListWidgetItem(f"{template.name}\n{type_text}")
                item.setData(Qt.ItemDataRole.UserRole, template.template_id)
                self._template_list.addItem(item)
                if template.template_id == previous:
                    target_row = self._template_list.count() - 1
            if target_row < 0 and self._template_list.count():
                target_row = 0
            self._template_list.setCurrentRow(target_row)
        self._selection_changed(self._template_list.currentItem(), None)

    def _selection_changed(self, current: QListWidgetItem | None, _previous: object) -> None:
        template = self._selected_template() if current is not None else None
        available = template is not None
        self._apply_button.setEnabled(available)
        self._copy_button.setEnabled(available)
        self._export_button.setEnabled(available)
        editable = available and not template.builtin
        self._rename_button.setEnabled(editable)
        self._delete_button.setEnabled(editable)
        if template is None:
            self._name_label.setText("请选择模板")
            self._summary.clear()
            self._preview.set_parameters(None)
            return
        self._name_label.setText(template.name)
        self._summary.setPlainText(_template_summary(template))
        self._preview.set_parameters(template.parameters)

    def _apply_current(self) -> None:
        template = self._selected_template()
        if template is None:
            return
        self.selected_template_id = template.template_id
        self.accept()

    def _copy_current(self) -> None:
        template = self._selected_template()
        if template is None:
            return
        base = template.name.replace("（默认）", "").strip()
        proposed = f"{base} 副本"[:40]
        suffix = 2
        while self._store.find_by_name(proposed) is not None:
            tail = f" 副本 {suffix}"
            proposed = f"{base[:40 - len(tail)]}{tail}"
            suffix += 1
        dialog = _TemplateDetailsDialog(
            "复制版面模板",
            name=proposed,
            description=template.description,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        name, description = dialog.values()
        try:
            copied = self._store.duplicate(
                template.template_id,
                name,
                description=description,
            )
        except (OSError, RuntimeError, FileExistsError, ValueError) as exc:
            QMessageBox.warning(self, "模板复制失败", str(exc))
            return
        self.changed = True
        self._reload(copied.template_id)

    def _rename_current(self) -> None:
        template = self._selected_template()
        if template is None or template.builtin:
            return
        dialog = _TemplateDetailsDialog(
            "编辑模板信息",
            name=template.name,
            description=template.description,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        name, description = dialog.values()
        try:
            renamed = self._store.update_details(
                template.template_id,
                name=name,
                description=description,
            )
        except (OSError, RuntimeError, FileExistsError, ValueError) as exc:
            QMessageBox.warning(self, "模板重命名失败", str(exc))
            return
        self.changed = True
        self._reload(renamed.template_id)

    def _delete_current(self) -> None:
        template = self._selected_template()
        if template is None or template.builtin:
            return
        dialog = QMessageBox(self)
        dialog.setWindowTitle("删除模板")
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setText(f"确定删除模板“{template.name}”吗？")
        delete_button = dialog.addButton(
            "删除模板",
            QMessageBox.ButtonRole.DestructiveRole,
        )
        cancel_button = dialog.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        dialog.setDefaultButton(cancel_button)
        dialog.exec()
        if dialog.clickedButton() is not delete_button:
            return
        try:
            self._store.delete(template.template_id)
        except (OSError, RuntimeError, KeyError, ValueError) as exc:
            QMessageBox.warning(self, "模板删除失败", str(exc))
            return
        self.changed = True
        self._reload(DEFAULT_TEMPLATE_ID)

    def _import_template(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "导入版面模板",
            self._last_directory,
            "通用经文排版模板 (*.json);;JSON 文件 (*.json)",
        )
        if not path:
            return
        self._last_directory = str(Path(path).parent)
        try:
            imported = self._store.read_import_file(path)
        except (OSError, RuntimeError, ValueError) as exc:
            QMessageBox.warning(self, "模板导入失败", str(exc))
            return
        existing = self._store.find_by_name(imported.name)
        target_name = imported.name
        overwrite = False
        if existing is not None:
            dialog = QMessageBox(self)
            dialog.setWindowTitle("模板名称重复")
            dialog.setIcon(QMessageBox.Icon.Question)
            dialog.setText(f"模板“{imported.name}”已经存在，请选择处理方式。")
            overwrite_button = dialog.addButton(
                "覆盖",
                QMessageBox.ButtonRole.AcceptRole,
            )
            save_as_button = dialog.addButton(
                "另存为",
                QMessageBox.ButtonRole.ActionRole,
            )
            cancel_button = dialog.addButton("取消", QMessageBox.ButtonRole.RejectRole)
            if existing.builtin:
                overwrite_button.setEnabled(False)
            dialog.setDefaultButton(cancel_button)
            dialog.exec()
            clicked = dialog.clickedButton()
            if clicked is overwrite_button:
                overwrite = True
            elif clicked is save_as_button:
                target_name, accepted = QInputDialog.getText(
                    self,
                    "导入模板另存为",
                    "新的模板名称：",
                    text=f"{imported.name} 副本"[:40],
                )
                if not accepted:
                    return
            else:
                return
        try:
            saved = self._store.import_template(
                path,
                target_name=target_name,
                overwrite=overwrite,
            )
        except (OSError, RuntimeError, FileExistsError, ValueError) as exc:
            QMessageBox.warning(self, "模板导入失败", str(exc))
            return
        self.changed = True
        self._reload(saved.template_id)

    def _export_current(self) -> None:
        template = self._selected_template()
        if template is None:
            return
        proposed = self._store.suggested_export_path(template, self._last_directory)
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "导出版面模板",
            proposed,
            "通用经文排版模板 (*.json)",
        )
        if not path:
            return
        if not Path(path).suffix:
            path += ".json"
        self._last_directory = str(Path(path).parent)
        try:
            self._store.export_template(template.template_id, path)
        except (OSError, RuntimeError, ValueError) as exc:
            QMessageBox.warning(self, "模板导出失败", str(exc))
            return
        QMessageBox.information(self, "模板导出完成", f"模板已导出到：\n{path}")


class ScriptureLayoutPage(QWidget):
    """模板化通用经文排版、预览和分层 PSD 输出页面。"""

    home_requested = Signal()
    status_message = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("scriptureLayoutPage")
        self._thread_pool = QThreadPool.globalInstance()
        self._workers: set[FunctionWorker] = set()
        self._template_store = self._create_template_store()
        self._applied_template_id = ""
        self._template_baseline: LayoutParameters | None = None
        self._template_dirty = False
        self._applying_template = False
        self._glyph_index: GlyphIndex | None = None
        self._boards: tuple[BoardLayout, ...] = ()
        self._current_board = 0
        self._preview_generation = 0
        self._source_generation = 0
        self._source_cancel: threading.Event | None = None
        self._source_worker: FunctionWorker | None = None
        self._text_load_generation = 0
        self._text_load_worker: FunctionWorker | None = None
        self._last_scripture_directory = config.SCRIPT_DIR
        self._generation_cancel: threading.Event | None = None
        self._generation_worker: FunctionWorker | None = None
        self._generation_started_at: float | None = None
        self._preview_zoom = 0.0
        self._preview_image = QImage()
        self._preview_document_size = QSize()
        self._preview_device_ratio = 1.0
        self._preview_render_context: object | None = None
        self._preview_cancel: threading.Event | None = None
        self._preview_worker: FunctionWorker | None = None
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(220)
        self._preview_timer.timeout.connect(self._rebuild_layout)
        self._preview_quality_timer = QTimer(self)
        self._preview_quality_timer.setSingleShot(True)
        self._preview_quality_timer.setInterval(180)
        self._preview_quality_timer.timeout.connect(self._render_current_preview)
        self._build_ui()
        self._load_library_names()
        self._source_mode_changed()
        default_template_id = self._default_template_id()
        self._reload_template_names(default_template_id)
        self._apply_template(default_template_id)
        self._set_running(False)

    def _create_template_store(self) -> LayoutTemplateStore:
        return LayoutTemplateStore(
            config.LAYOUT_TEMPLATE_FILE,
            legacy_file_path=config.LEGACY_LAYOUT_TEMPLATE_FILE,
        )

    def _default_template_id(self) -> str:
        return DEFAULT_TEMPLATE_ID

    @property
    def is_running(self) -> bool:
        return self._generation_worker is not None

    def _build_ui(self) -> None:
        self.setStyleSheet(
            """
            QWidget#scriptureLayoutPage QLabel:disabled,
            QWidget#scriptureLayoutPage QCheckBox:disabled,
            QWidget#scriptureLayoutPage QRadioButton:disabled,
            QWidget#scriptureLayoutPage QRadioButton[inactiveOption="true"] {
                color: #68717E;
            }
            QWidget#scriptureLayoutPage QLineEdit:disabled,
            QWidget#scriptureLayoutPage QSpinBox:disabled,
            QWidget#scriptureLayoutPage QDoubleSpinBox:disabled,
            QWidget#scriptureLayoutPage QComboBox:disabled {
                color: #68717E;
                background: #242A33;
                border-color: #303640;
            }
            QWidget#scriptureLayoutPage QDoubleSpinBox {
                padding-left: 2px;
                padding-right: 0;
            }
            """
        )
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(10)

        toolbar = QGridLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setColumnStretch(0, 1)
        toolbar.setColumnStretch(2, 1)
        title = QLabel("通用经文排版")
        title.setProperty("role", "pageTitle")
        toolbar.addWidget(title, 0, 0, Qt.AlignmentFlag.AlignLeft)
        template_group = QWidget()
        template_group.setObjectName("centeredLayoutTemplateGroup")
        template_layout = QHBoxLayout(template_group)
        template_layout.setContentsMargins(0, 0, 0, 0)
        template_layout.setSpacing(8)
        template_layout.addWidget(QLabel("版面模板："))
        self._template_combo = QComboBox()
        self._template_combo.setObjectName("layoutTemplateCombo")
        self._template_combo.setMinimumWidth(240)
        self._template_combo.currentIndexChanged.connect(self._template_choice_changed)
        template_layout.addWidget(self._template_combo)
        toolbar.addWidget(template_group, 0, 1, Qt.AlignmentFlag.AlignCenter)
        self._home_button = QPushButton("返回首页")
        self._home_button.clicked.connect(self._request_home)
        toolbar.addWidget(self._home_button, 0, 2, Qt.AlignmentFlag.AlignRight)
        root.addLayout(toolbar)

        splitter = _ProportionalSplitter()
        splitter.setObjectName("scriptureLayoutSplitter")
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_preview_panel())
        splitter.addWidget(self._build_parameter_panel())
        splitter.setStretchFactor(0, 30)
        splitter.setStretchFactor(1, 62)
        splitter.setStretchFactor(2, 0)
        self._splitter = splitter
        splitter.set_preferred_sizes(300, 620, 480)
        root.addWidget(splitter, 1)
        root.addWidget(self._build_bottom_bar())

    def _card(self) -> QFrame:
        frame = QFrame()
        frame.setProperty("role", "card")
        return frame

    def _build_left_panel(self) -> QWidget:
        panel = self._card()
        panel.setMinimumWidth(280)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        text_header = QHBoxLayout()
        heading = QLabel("经文正文")
        heading.setObjectName("sectionTitle")
        text_header.addWidget(heading)
        text_header.addStretch(1)
        self._open_text_button = QPushButton("打开文件")
        self._open_text_button.setObjectName("openScriptureFileButton")
        self._open_text_button.setToolTip("只读取文件中的文字，不会修改源文件")
        self._open_text_button.clicked.connect(self._open_scripture_file)
        text_header.addWidget(self._open_text_button)
        layout.addLayout(text_header)
        self._text_edit = _ChinesePlainTextEdit()
        self._text_edit.setObjectName("scriptureTextEdit")
        self._text_edit.setPlaceholderText("在这里粘贴经文正文，或点击右上角打开文件。每一行作为一个段落。")
        self._text_edit.textChanged.connect(self._scripture_text_changed)
        layout.addWidget(self._text_edit, 3)
        self._scripture_statistics = QLabel("总字数：0　去重后字数：0　标点符号数：0")
        self._scripture_statistics.setObjectName("scriptureStatistics")
        self._scripture_statistics.setWordWrap(True)
        layout.addWidget(self._scripture_statistics)

        source_heading = QLabel("字图来源")
        source_heading.setObjectName("sectionTitle")
        layout.addWidget(source_heading)
        self._source_button_group = QButtonGroup(self)
        self._system_source_radio = QRadioButton("本系统字库")
        self._external_source_radio = QRadioButton("外部字库目录")
        self._source_button_group.addButton(self._system_source_radio)
        self._source_button_group.addButton(self._external_source_radio)
        self._system_source_radio.setChecked(True)
        self._system_source_radio.toggled.connect(self._source_mode_changed)
        system_row = QHBoxLayout()
        system_row.addWidget(self._system_source_radio)
        self._system_library_combo = _NoWheelComboBox()
        self._system_library_combo.setObjectName("systemLibraryCombo")
        self._system_library_combo.currentIndexChanged.connect(self._source_changed)
        system_row.addWidget(self._system_library_combo, 1)
        layout.addLayout(system_row)
        external_row = QHBoxLayout()
        external_row.addWidget(self._external_source_radio)
        self._external_path_edit = QLineEdit()
        self._external_path_edit.setObjectName("externalGlyphDirectoryEdit")
        self._external_path_edit.setPlaceholderText("选择外部字库目录")
        self._external_path_edit.textChanged.connect(self._source_changed)
        self._browse_source_button = QPushButton("浏览")
        self._browse_source_button.clicked.connect(self._browse_external_source)
        external_row.addWidget(self._external_path_edit, 1)
        external_row.addWidget(self._browse_source_button)
        layout.addLayout(external_row)
        self._check_source_button = QPushButton("载入并检查字图")
        self._check_source_button.setProperty("role", "primary")
        self._check_source_button.clicked.connect(self._load_glyph_source)
        layout.addWidget(self._check_source_button)
        self._source_progress = QProgressBar()
        self._source_progress.setObjectName("glyphSourceCheckProgress")
        self._source_progress.setRange(0, 100)
        self._source_progress.setValue(0)
        self._source_progress.setFormat("等待检查")
        self._source_progress.hide()
        layout.addWidget(self._source_progress)

        check_heading = QLabel("检查结果")
        check_heading.setObjectName("sectionTitle")
        layout.addWidget(check_heading)
        self._source_summary = QLabel("待载入字图后检查")
        self._source_summary.setWordWrap(True)
        self._source_summary.setProperty("role", "muted")
        layout.addWidget(self._source_summary)
        self._check_results = _ChinesePlainTextEdit()
        self._check_results.setObjectName("scriptureCheckList")
        self._check_results.setReadOnly(True)
        self._check_results.setPlaceholderText("缺失文字会显示在这里")
        self._check_results.setMinimumHeight(100)
        self._check_list = self._check_results
        layout.addWidget(self._check_results, 2)
        return panel

    def _build_preview_panel(self) -> QWidget:
        panel = self._card()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        header = QHBoxLayout()
        heading = QLabel("版面预览")
        heading.setObjectName("sectionTitle")
        self._board_summary = QLabel("暂无版面")
        self._board_summary.setProperty("role", "muted")
        header.addWidget(heading)
        header.addStretch(1)
        header.addWidget(self._board_summary)
        layout.addLayout(header)
        self._preview_scroll = _PreviewScrollArea()
        self._preview_scroll.setObjectName("scripturePreviewScroll")
        self._preview_scroll.setWidgetResizable(False)
        self._preview_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_scroll.setStyleSheet("QScrollArea { background: #10141A; }")
        self._preview_scroll.zoom_requested.connect(self._zoom_preview_at)
        self._preview_label = _PreviewCanvas()
        self._preview_label.setObjectName("scripturePreview")
        self._preview_scroll.setWidget(self._preview_label)
        layout.addWidget(self._preview_scroll, 1)
        self._preview_progress = QProgressBar()
        self._preview_progress.setObjectName("layoutPreviewProgress")
        self._preview_progress.setRange(0, 100)
        self._preview_progress.setValue(0)
        self._preview_progress.setFormat("等待预览")
        self._preview_progress.hide()
        layout.addWidget(self._preview_progress)
        controls = QHBoxLayout()
        self._previous_button = QPushButton("上一版")
        self._previous_button.clicked.connect(lambda: self._change_board(-1))
        self._next_button = QPushButton("下一版")
        self._next_button.clicked.connect(lambda: self._change_board(1))
        self._board_label = QLabel("0 / 0")
        controls.addWidget(self._previous_button)
        controls.addWidget(self._next_button)
        controls.addWidget(self._board_label)
        controls.addStretch(1)
        self._preview_guides_check = QCheckBox("显示田字格和框线")
        self._preview_guides_check.setObjectName("previewGuidesCheck")
        self._preview_guides_check.setChecked(True)
        self._preview_guides_check.toggled.connect(self._preview_guides_changed)
        controls.addWidget(self._preview_guides_check)
        controls.addStretch(1)
        controls.addWidget(QLabel("缩放"))
        self._zoom_combo = QComboBox()
        self._zoom_combo.setEditable(True)
        self._zoom_combo.addItems(["适合窗口", "50%", "75%", "100%", "150%", "200%"])
        self._zoom_combo.setMinimumWidth(
            max(
                116,
                self.fontMetrics().horizontalAdvance("适合窗口") + 52,
            )
        )
        if self._zoom_combo.lineEdit() is not None:
            self._zoom_combo.lineEdit().setReadOnly(True)
        self._zoom_combo.currentTextChanged.connect(self._zoom_changed)
        controls.addWidget(self._zoom_combo)
        layout.addLayout(controls)
        self._update_board_controls()
        return panel

    def _build_parameter_panel(self) -> QWidget:
        panel = self._card()
        panel.setMinimumWidth(480)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        heading_row = QWidget()
        heading_layout = QHBoxLayout(heading_row)
        heading_layout.setContentsMargins(0, 0, 0, 0)
        heading = QLabel("排版参数")
        heading.setObjectName("sectionTitle")
        heading_layout.addWidget(heading)
        heading_layout.addStretch(1)
        self._save_template_button = QPushButton("模板保存")
        self._save_template_button.setObjectName("saveLayoutTemplateButton")
        self._save_template_button.setEnabled(False)
        self._save_template_button.clicked.connect(self._save_custom_template)
        heading_layout.addWidget(self._save_template_button)
        layout.addWidget(heading_row)
        self._tabs = QTabWidget()
        self._tabs.setObjectName("scriptureParameterTabs")
        self._tabs.addTab(self._build_layout_tab(), "版面")
        self._tabs.addTab(self._build_text_paragraph_tab(), "文字与段落")
        # 保留旧页面对象契约，但将“输出参数”从标签页提升为同级区块。
        # 标签页本身隐藏，避免用户误以为输出参数仍属于排版参数的页签。
        self._tabs.addTab(QWidget(), "输出参数")
        self._tabs.tabBar().setTabVisible(2, False)
        layout.addWidget(self._tabs, 1)

        output_heading = QLabel("输出参数")
        output_heading.setObjectName("sectionTitle")
        output_heading.setProperty("role", "parameterSectionHeading")
        layout.addWidget(output_heading, 0, Qt.AlignmentFlag.AlignLeft)
        output_form = self._build_output_form()
        output_form.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        layout.addWidget(output_form, 0)
        return panel

    def _scrollable_form(self) -> tuple[QScrollArea, QFormLayout]:
        content = QWidget()
        form = QFormLayout(content)
        form.setContentsMargins(10, 10, 10, 10)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setLabelAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        form.setFormAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter
        )
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll.setWidget(content)
        return scroll, form

    def _spin(self, minimum: int, maximum: int, suffix: str = "") -> QSpinBox:
        control = _NoWheelSpinBox()
        control.setRange(minimum, maximum)
        control.setSuffix(suffix)
        control.valueChanged.connect(self._schedule_preview)
        return control

    def _double_spin(self, minimum: float, maximum: float, suffix: str = "") -> QDoubleSpinBox:
        control = _NoWheelDoubleSpinBox()
        control.setRange(minimum, maximum)
        control.setDecimals(2)
        control.setSuffix(suffix)
        control.valueChanged.connect(self._schedule_preview)
        return control

    def _form_label(self, text: str, field: QWidget) -> QLabel:
        """创建与字段等高、垂直居中的参数名称标签。"""

        label = QLabel(text if text.endswith("：") else f"{text}：")
        label.setContentsMargins(4, 0, 4, 0)
        label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        label.setMinimumHeight(field.sizeHint().height())
        label.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Fixed,
        )
        label.setMinimumWidth(self._protected_text_width(label.text(), 20, label))
        return label

    def _protected_text_width(
        self,
        text: str,
        padding: int = 10,
        widget: QWidget | None = None,
    ) -> int:
        """为字体左右边缘、抗锯齿和高 DPI 取整预留安全宽度。"""

        metrics = (widget or self).fontMetrics()
        return max(
            metrics.horizontalAdvance(text),
            metrics.boundingRect(text).width(),
        ) + max(0, padding)

    def _add_form_row(
        self,
        form: QFormLayout,
        text: str,
        field: QWidget,
    ) -> QLabel:
        label = self._form_label(text, field)
        form.addRow(label, field)
        return label

    def _inline_label(self, text: str, width: int | None = None) -> QLabel:
        display_text = text if text.endswith("：") else f"{text}："
        label = QLabel(display_text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        protected_width = self._protected_text_width(display_text, 20, label)
        label.setFixedWidth(max(protected_width, int(width or 0)))
        return label

    def _single_labeled_control(
        self,
        label_text: str,
        control: QWidget,
        *,
        label_width: int | None = None,
    ) -> tuple[QWidget, QLabel]:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        label = self._inline_label(label_text, label_width)
        control.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        layout.addWidget(label)
        layout.addWidget(control, 1)
        return row, label

    def _aligned_single_control(
        self,
        control: QWidget,
        reference_label: str = "高",
    ) -> QWidget:
        """让单个输入框与成对参数中的第一个输入框左对齐。"""

        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        placeholder = QWidget()
        placeholder.setFixedWidth(
            self._protected_text_width(f"{reference_label}：", 20)
        )
        control.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        layout.addWidget(placeholder)
        layout.addWidget(control, 1)
        return row

    def _paired_controls(
        self,
        first_label: str,
        first_control: QWidget,
        second_label: str,
        second_control: QWidget,
    ) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        first_control.setMinimumWidth(max(54, first_control.minimumWidth()))
        second_control.setMinimumWidth(max(54, second_control.minimumWidth()))
        first_control.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        second_control.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        first_display = f"{first_label}："
        second_display = f"{second_label}："
        label_width = max(
            self._protected_text_width(first_display, 20),
            self._protected_text_width(second_display, 20),
        )
        first_text = self._inline_label(first_display, label_width)
        second_text = self._inline_label(second_display, label_width)
        layout.addWidget(first_text)
        layout.addWidget(first_control, 1)
        layout.addWidget(second_text)
        layout.addWidget(second_control, 1)
        return row

    def _stacked_controls(self, *rows: QWidget) -> QWidget:
        """把同一参数的多行控件放进一个字段，使主标签垂直居中。"""

        group = QWidget()
        layout = QVBoxLayout(group)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        for row in rows:
            layout.addWidget(row)
        return group

    def _named_percentage_pair(
        self,
        name: str,
        first_control: QSpinBox,
        second_control: QSpinBox,
        *,
        name_width: int,
    ) -> QWidget:
        """构建适合右栏最小宽度的“名称、阈值、目标”参数行。"""

        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        name_label = QLabel(f"{name}：")
        name_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        name_label.setFixedWidth(name_width)
        threshold_label = QLabel("阈值：")
        threshold_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        threshold_label.setFixedWidth(
            self._protected_text_width("阈值：", 14, threshold_label)
        )
        target_label = QLabel("目标：")
        target_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        target_label.setFixedWidth(
            self._protected_text_width("目标：", 14, target_label)
        )
        for control in (first_control, second_control):
            control.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed,
            )
        layout.addWidget(name_label)
        layout.addWidget(threshold_label)
        layout.addWidget(first_control, 1)
        layout.addWidget(target_label)
        layout.addWidget(second_control, 1)
        return row

    def _radio_row(self, *controls: QRadioButton) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        for control in controls:
            layout.addWidget(control)
        layout.addStretch(1)
        return row

    def _checkbox_row(self, *controls: QCheckBox) -> QWidget:
        """把相关复选项放在同一行并保持左对齐。"""

        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(18)
        for control in controls:
            layout.addWidget(control)
        layout.addStretch(1)
        return row

    def _form_group(self) -> tuple[QWidget, QFormLayout]:
        group = QWidget()
        form = QFormLayout(group)
        form.setContentsMargins(14, 4, 0, 8)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setLabelAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        return group, form

    def _build_layout_tab(self) -> QWidget:
        scroll, form = self._scrollable_form()
        self._dpi_spin = self._spin(72, 1200, " DPI")
        self._cell_width_spin = self._double_spin(0.1, 1000, " mm")
        self._cell_height_spin = self._double_spin(0.1, 1000, " mm")
        self._rows_spin = self._spin(1, 500)
        self._columns_spin = self._spin(1, 500)
        self._row_gap_spin = self._double_spin(0, 500, " mm")
        self._column_gap_spin = self._double_spin(0, 500, " mm")
        self._draw_frame_check = QCheckBox("绘制大框")
        self._draw_frame_check.toggled.connect(self._parameter_mode_changed)
        self._frame_top_spin = self._double_spin(0, 1000, " mm")
        self._frame_bottom_spin = self._double_spin(0, 1000, " mm")
        self._frame_left_spin = self._double_spin(0, 1000, " mm")
        self._frame_right_spin = self._double_spin(0, 1000, " mm")
        self._canvas_top_spin = self._double_spin(0, 1000, " mm")
        self._canvas_bottom_spin = self._double_spin(0, 1000, " mm")
        self._canvas_left_spin = self._double_spin(0, 1000, " mm")
        self._canvas_right_spin = self._double_spin(0, 1000, " mm")
        self._special_gaps_check = QCheckBox("启用特殊行列间距")
        self._special_gaps_check.toggled.connect(self._parameter_mode_changed)
        self._special_row_gaps_edit = QLineEdit()
        self._special_row_gaps_edit.setPlaceholderText("例如：2=1.5, 18=3")
        self._special_row_gaps_edit.textChanged.connect(self._schedule_preview)
        self._special_column_gaps_edit = QLineEdit()
        self._special_column_gaps_edit.setPlaceholderText("例如：3=2, 12=2.5")
        self._special_column_gaps_edit.textChanged.connect(self._schedule_preview)
        self._frame_margin_controls = (
            self._frame_top_spin,
            self._frame_bottom_spin,
            self._frame_left_spin,
            self._frame_right_spin,
        )
        self._special_gap_controls = (
            self._special_row_gaps_edit,
            self._special_column_gaps_edit,
        )
        self._frame_margin_group = self._stacked_controls(
            self._paired_controls(
                "上", self._frame_top_spin, "下", self._frame_bottom_spin
            ),
            self._paired_controls(
                "左", self._frame_left_spin, "右", self._frame_right_spin
            ),
        )
        self._frame_margin_label = self._form_label(
            "大框边距", self._frame_margin_group
        )
        self._canvas_margin_group = self._stacked_controls(
            self._paired_controls(
                "上", self._canvas_top_spin, "下", self._canvas_bottom_spin
            ),
            self._paired_controls(
                "左", self._canvas_left_spin, "右", self._canvas_right_spin
            ),
        )
        self._canvas_margin_label = self._form_label(
            "画布边距", self._canvas_margin_group
        )
        inline_label_width = self._protected_text_width("高：")
        special_row, self._special_row_gap_label = self._single_labeled_control(
            "行",
            self._special_row_gaps_edit,
            label_width=inline_label_width,
        )
        special_column, self._special_column_gap_label = self._single_labeled_control(
            "列",
            self._special_column_gaps_edit,
            label_width=inline_label_width,
        )
        self._special_gap_labels = (
            self._special_row_gap_label,
            self._special_column_gap_label,
        )
        dpi_field = self._aligned_single_control(self._dpi_spin)
        self._add_form_row(form, "画布 DPI", dpi_field)
        self._add_form_row(
            form,
            "单元格",
            self._paired_controls(
                "高", self._cell_height_spin, "宽", self._cell_width_spin
            ),
        )
        self._add_form_row(
            form,
            "页行列数",
            self._paired_controls("行", self._rows_spin, "列", self._columns_spin),
        )
        self._add_form_row(
            form,
            "行列间距",
            self._paired_controls("行", self._row_gap_spin, "列", self._column_gap_spin),
        )
        self._add_form_row(form, "大框", self._draw_frame_check)
        form.addRow(self._frame_margin_label, self._frame_margin_group)
        form.addRow(self._canvas_margin_label, self._canvas_margin_group)
        form.addRow(self._special_gaps_check)
        form.addRow("", special_row)
        form.addRow("", special_column)
        return scroll

    def _build_text_paragraph_tab(self) -> QWidget:
        scroll, form = self._scrollable_form()
        layout_heading_row = QWidget()
        layout_heading_layout = QHBoxLayout(layout_heading_row)
        layout_heading_layout.setContentsMargins(0, 0, 0, 0)
        layout_heading = QLabel("排版方式")
        layout_heading.setObjectName("sectionTitle")
        layout_heading_layout.addWidget(layout_heading)
        form.addRow(layout_heading_row)
        self._vertical_layout_radio = QRadioButton("竖排")
        self._horizontal_layout_radio = QRadioButton("横排")
        self._vertical_layout_radio.setObjectName("verticalLayoutRadio")
        self._horizontal_layout_radio.setObjectName("horizontalLayoutRadio")
        self._vertical_layout_radio.setChecked(True)
        self._layout_mode_button_group = QButtonGroup(self)
        self._layout_mode_button_group.addButton(self._vertical_layout_radio)
        self._layout_mode_button_group.addButton(self._horizontal_layout_radio)
        self._vertical_layout_radio.toggled.connect(self._parameter_mode_changed)
        self._left_to_right_radio = QRadioButton("从左到右")
        self._right_to_left_radio = QRadioButton("从右到左")
        self._left_to_right_radio.setObjectName("leftToRightRadio")
        self._right_to_left_radio.setObjectName("rightToLeftRadio")
        self._right_to_left_radio.setChecked(True)
        self._flow_direction_button_group = QButtonGroup(self)
        self._flow_direction_button_group.addButton(self._left_to_right_radio)
        self._flow_direction_button_group.addButton(self._right_to_left_radio)
        self._left_to_right_radio.toggled.connect(self._parameter_mode_changed)
        self._add_form_row(
            form,
            "排列方向",
            self._radio_row(self._vertical_layout_radio, self._horizontal_layout_radio),
        )
        self._add_form_row(
            form,
            "行进方向",
            self._radio_row(self._left_to_right_radio, self._right_to_left_radio),
        )

        self._scale_source_radio = QRadioButton("按源图尺寸")
        self._scale_cell_radio = QRadioButton("相对单元格")
        self._scale_source_radio.setObjectName("scaleBySourceRadio")
        self._scale_cell_radio.setObjectName("scaleToCellRadio")
        self._scale_source_radio.setChecked(True)
        self._scale_button_group = QButtonGroup(self)
        self._scale_button_group.addButton(self._scale_source_radio)
        self._scale_button_group.addButton(self._scale_cell_radio)
        self._scale_source_radio.toggled.connect(self._parameter_mode_changed)
        self._scale_percent_spin = self._spin(1, 500, "%")
        self._cell_fill_spin = self._spin(1, 500, "%")
        self._auto_scale_check = QCheckBox("启用自动缩放")
        self._auto_scale_check.toggled.connect(self._parameter_mode_changed)
        self._enlarge_threshold_spin = self._spin(1, 500, "%")
        self._enlarge_fill_spin = self._spin(1, 500, "%")
        self._shrink_threshold_spin = self._spin(1, 500, "%")
        self._shrink_fill_spin = self._spin(1, 500, "%")
        percentage_width = self._protected_text_width("500%", 28)
        for control in (
            self._scale_percent_spin,
            self._cell_fill_spin,
            self._enlarge_threshold_spin,
            self._enlarge_fill_spin,
            self._shrink_threshold_spin,
            self._shrink_fill_spin,
        ):
            control.setMinimumWidth(percentage_width)
        self._source_scale_group, source_form = self._form_group()
        self._add_form_row(source_form, "文字全局缩放", self._scale_percent_spin)
        source_form.addRow(self._auto_scale_check)
        self._auto_scale_group = QWidget()
        auto_layout = QVBoxLayout(self._auto_scale_group)
        auto_layout.setContentsMargins(14, 4, 0, 8)
        auto_layout.setSpacing(6)
        auto_name_width = max(
            self._protected_text_width("自动放大：", 8),
            self._protected_text_width("自动缩小：", 8),
        )
        auto_layout.addWidget(
            self._named_percentage_pair(
                "自动放大",
                self._enlarge_threshold_spin,
                self._enlarge_fill_spin,
                name_width=auto_name_width,
            )
        )
        auto_layout.addWidget(
            self._named_percentage_pair(
                "自动缩小",
                self._shrink_threshold_spin,
                self._shrink_fill_spin,
                name_width=auto_name_width,
            )
        )
        source_form.addRow(self._auto_scale_group)
        self._cell_scale_group, cell_form = self._form_group()
        self._add_form_row(cell_form, "缩放至单元格", self._cell_fill_spin)
        self._add_form_row(
            form,
            "文字缩放方式",
            self._radio_row(self._scale_source_radio, self._scale_cell_radio),
        )
        form.addRow(self._source_scale_group)
        form.addRow(self._cell_scale_group)

        paragraph_heading = QLabel("段落规则")
        paragraph_heading.setObjectName("sectionTitle")
        form.addRow(paragraph_heading)
        self._paragraph_skip_radio = QRadioButton("段后跳格")
        self._paragraph_column_radio = QRadioButton("段后换列")
        self._paragraph_skip_radio.setObjectName("paragraphSkipRadio")
        self._paragraph_column_radio.setObjectName("paragraphNewColumnRadio")
        self._paragraph_skip_radio.setChecked(True)
        self._paragraph_button_group = QButtonGroup(self)
        self._paragraph_button_group.addButton(self._paragraph_skip_radio)
        self._paragraph_button_group.addButton(self._paragraph_column_radio)
        self._paragraph_skip_radio.toggled.connect(self._parameter_mode_changed)
        self._paragraph_skip_spin = self._spin(0, 250000, " 格")
        self._first_title_check = QCheckBox("首经题单列")
        self._last_title_check = QCheckBox("尾经题单列")
        self._first_title_check.toggled.connect(self._schedule_preview)
        self._last_title_check.toggled.connect(self._schedule_preview)
        self._add_form_row(
            form,
            "处理方式",
            self._radio_row(self._paragraph_skip_radio, self._paragraph_column_radio),
        )
        self._paragraph_skip_group, skip_form = self._form_group()
        self._add_form_row(skip_form, "段后跳格数", self._paragraph_skip_spin)
        skip_form.addRow(
            self._checkbox_row(self._first_title_check, self._last_title_check)
        )
        form.addRow(self._paragraph_skip_group)
        self._trim_columns_check = QCheckBox("末版压缩")
        self._trim_columns_check.toggled.connect(self._schedule_preview)
        self._annotation_check = QCheckBox("尺寸标注")
        self._annotation_check.toggled.connect(self._schedule_preview)
        self._other_rules_heading = QLabel("其他规则")
        self._other_rules_heading.setObjectName("sectionTitle")
        form.addRow(self._other_rules_heading)
        self._include_punctuation_check = QCheckBox("标点符号")
        self._include_punctuation_check.setObjectName("includePunctuationCheck")
        self._include_punctuation_check.setChecked(False)
        self._include_punctuation_check.toggled.connect(
            lambda _checked: self._scripture_text_changed()
        )
        form.addRow(
            self._checkbox_row(
                self._include_punctuation_check,
                self._trim_columns_check,
                self._annotation_check,
            )
        )
        return scroll

    def _build_output_form(self) -> QWidget:
        content = QWidget()
        form = QFormLayout(content)
        form.setContentsMargins(10, 0, 10, 8)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setLabelAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        form.setFormAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter
        )
        directory_row = QWidget()
        directory_layout = QHBoxLayout(directory_row)
        directory_layout.setContentsMargins(0, 0, 0, 0)
        desktop_path = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.DesktopLocation
        )
        settings_service = SettingsService()
        try:
            default_layout_directory = (
                settings_service.load().default_layout_directory
            )
        except (OSError, RuntimeError, ValueError):
            default_layout_directory = ""
        output_directory = SettingsService.usable_directory(
            default_layout_directory,
            desktop_path or config.SCRIPT_DIR,
        )
        self._output_path_edit = QLineEdit(output_directory)
        self._output_path_edit.setObjectName("layoutOutputDirectoryEdit")
        self._output_path_edit.setPlaceholderText("选择 PSD/PSB 输出目录")
        browse = QPushButton("浏览")
        browse.clicked.connect(self._browse_output)
        directory_layout.addWidget(self._output_path_edit, 1)
        directory_layout.addWidget(browse)
        self._output_name_edit = QLineEdit("通用经文排版")
        self._output_name_edit.setObjectName("layoutOutputFileNameEdit")
        self._board_selection_edit = QLineEdit("全部")
        self._board_selection_edit.setPlaceholderText("全部，或 1,3-5")
        self._compress_psd_check = QCheckBox("启用")
        self._compress_psd_check.setObjectName("compressPsdCheck")
        self._compress_psd_check.setChecked(True)
        self._output_format_combo = QComboBox()
        self._output_format_combo.setObjectName("layoutOutputFormatCombo")
        self._output_format_combo.addItems(
            [OUTPUT_FORMAT_AUTO, OUTPUT_FORMAT_PSD, OUTPUT_FORMAT_PSB]
        )
        self._add_form_row(form, "输出目录", directory_row)
        self._add_form_row(form, "文件名", self._output_name_edit)
        self._add_form_row(form, "生成版面", self._board_selection_edit)
        self._add_form_row(form, "文件格式", self._output_format_combo)
        self._add_form_row(form, "PSD 压缩", self._compress_psd_check)
        return content

    def _build_bottom_bar(self) -> QWidget:
        bar = self._card()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 8, 12, 8)
        self._progress_bar = QProgressBar()
        self._progress_bar.setObjectName("layoutGenerationProgress")
        self._progress_bar.setMinimumWidth(360)
        self._progress_bar.setFormat("等待生成")
        layout.addWidget(self._progress_bar, 1)
        self._start_button = QPushButton("开始生成")
        self._start_button.setProperty("role", "primary")
        self._start_button.clicked.connect(self._start_generation)
        layout.addWidget(self._start_button)
        self._stop_button = QPushButton("停止生成")
        self._stop_button.clicked.connect(self._stop_generation)
        layout.addWidget(self._stop_button)
        return bar

    def _load_library_names(self) -> None:
        previous = self._system_library_combo.currentData()
        self._system_library_combo.clear()
        root = Path(config.ZIKU_ROOT)
        if root.is_dir():
            names = sorted(
                (
                    entry.name
                    for entry in root.iterdir()
                    if entry.is_dir()
                    and (entry / LIBRARY_DATABASE_FILENAME).is_file()
                ),
                key=natural_key,
            )
            for name in names:
                self._system_library_combo.addItem(name, str(root / name))
        if previous:
            index = self._system_library_combo.findData(previous)
            if index >= 0:
                self._system_library_combo.setCurrentIndex(index)

    def _reload_template_names(self, selected: str = "") -> None:
        try:
            selected_id = self._template_store.get(selected).template_id if selected else ""
        except KeyError:
            selected_id = selected
        self._applying_template = True
        with QSignalBlocker(self._template_combo):
            self._template_combo.clear()
            for template in self._template_store.list_templates():
                self._template_combo.addItem(template.name, template.template_id)
            index = self._template_combo.findData(selected_id)
            self._template_combo.setCurrentIndex(max(0, index))
        self._applying_template = False

    def _selected_template_id(self) -> str:
        return str(self._template_combo.currentData() or "")

    @Slot()
    def _template_choice_changed(self) -> None:
        if self._applying_template:
            return
        template_id = self._selected_template_id()
        if template_id == CUSTOM_TEMPLATE_ID:
            return
        try:
            template = self._template_store.get(template_id)
            self._template_combo.setToolTip(_template_summary(template))
        except KeyError as exc:
            self._template_combo.setToolTip("")
            QMessageBox.warning(self, "模板载入失败", str(exc))
            return
        self._apply_template(template_id)

    def _apply_template(self, template_id: str) -> bool:
        try:
            template = self._template_store.get(template_id)
        except (KeyError, RuntimeError, ValueError) as exc:
            QMessageBox.warning(self, "模板载入失败", str(exc))
            return False
        self._applying_template = True
        try:
            self._apply_parameters(template.parameters)
            self._applied_template_id = template.template_id
            self._template_baseline = self._collect_parameters()
            index = self._template_combo.findData(template.template_id)
            if index >= 0:
                with QSignalBlocker(self._template_combo):
                    self._template_combo.setCurrentIndex(index)
        finally:
            self._applying_template = False
        self._update_template_state()
        self._schedule_preview()
        self.status_message.emit(f"已应用版面模板“{template.name}”")
        return True

    def _update_template_state(self) -> None:
        if self._applying_template or not hasattr(self, "_save_template_button"):
            return
        if self._template_baseline is None:
            return
        try:
            dirty = self._collect_parameters() != self._template_baseline
        except ValueError:
            dirty = True
        self._template_dirty = dirty
        custom_index = self._template_combo.findData(CUSTOM_TEMPLATE_ID)
        with QSignalBlocker(self._template_combo):
            if dirty:
                if custom_index < 0:
                    self._template_combo.insertItem(0, "自定义模板", CUSTOM_TEMPLATE_ID)
                    custom_index = 0
                self._template_combo.setCurrentIndex(custom_index)
                self._template_combo.setToolTip("当前参数尚未保存为模板")
            else:
                if custom_index >= 0:
                    self._template_combo.removeItem(custom_index)
                applied_index = self._template_combo.findData(self._applied_template_id)
                if applied_index >= 0:
                    self._template_combo.setCurrentIndex(applied_index)
                    try:
                        template = self._template_store.get(self._applied_template_id)
                        self._template_combo.setToolTip(_template_summary(template))
                    except KeyError:
                        self._template_combo.setToolTip("")
        self._save_template_button.setEnabled(not self.is_running and dirty)

    @Slot()
    def _save_custom_template(self) -> None:
        if not self._template_dirty:
            return
        try:
            parameters = self._collect_parameters()
        except ValueError as exc:
            QMessageBox.warning(self, "参数无效", str(exc))
            return
        dialog = _TemplateSaveDialog(self._template_store, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            if dialog.mode() == _TemplateSaveDialog.MODE_NEW:
                name = dialog.new_name()
                if self._template_store.find_by_name(name) is not None:
                    raise FileExistsError(
                        f"模板“{name}”已经存在，请选择“替换现有模板”。"
                    )
                saved = self._template_store.save(
                    name,
                    parameters,
                    dialog.description(),
                )
            else:
                target_id = dialog.replacement_id()
                target = self._template_store.get(target_id)
                confirm = QMessageBox(self)
                confirm.setWindowTitle("替换模板")
                confirm.setIcon(QMessageBox.Icon.Warning)
                confirm.setText(f"确定用当前参数替换模板“{target.name}”吗？")
                replace_button = confirm.addButton(
                    "替换模板",
                    QMessageBox.ButtonRole.AcceptRole,
                )
                cancel_button = confirm.addButton(
                    "取消",
                    QMessageBox.ButtonRole.RejectRole,
                )
                confirm.setDefaultButton(cancel_button)
                confirm.exec()
                if confirm.clickedButton() is not replace_button:
                    return
                saved = self._template_store.update(
                    target_id,
                    parameters,
                    description=dialog.description(),
                )
        except (OSError, RuntimeError, FileExistsError, KeyError, ValueError) as exc:
            QMessageBox.warning(self, "模板保存失败", str(exc))
            return
        self._applied_template_id = saved.template_id
        self._template_baseline = parameters
        self._reload_template_names(saved.template_id)
        self._update_template_state()
        self.status_message.emit(f"版面模板“{saved.name}”已保存")

    def _parse_special_row_gaps(self) -> tuple[RowGapAdjustment, ...]:
        text = self._special_row_gaps_edit.text().strip()
        if not text:
            return ()
        result: list[RowGapAdjustment] = []
        for part in re.split(r"[,，;；]", text):
            if not part.strip():
                continue
            match = re.fullmatch(r"\s*(\d+)\s*[=:：]\s*(\d+(?:\.\d+)?)\s*", part)
            if not match:
                raise ValueError("特殊行距格式应为“行号=毫米”，多个项目用逗号分隔。")
            result.append(RowGapAdjustment(int(match.group(1)), float(match.group(2))))
        return tuple(result)

    def _parse_special_column_gaps(self) -> tuple[ColumnGapAdjustment, ...]:
        text = self._special_column_gaps_edit.text().strip()
        if not text:
            return ()
        result: list[ColumnGapAdjustment] = []
        for part in re.split(r"[,，;；]", text):
            if not part.strip():
                continue
            match = re.fullmatch(r"\s*(\d+)\s*[=:：]\s*(\d+(?:\.\d+)?)\s*", part)
            if not match:
                raise ValueError("特殊列距格式应为“列号=毫米”，多个项目用逗号分隔。")
            result.append(
                ColumnGapAdjustment(int(match.group(1)), float(match.group(2)))
            )
        return tuple(result)

    def _collect_parameters(self) -> LayoutParameters:
        parameters = LayoutParameters(
            dpi=self._dpi_spin.value(),
            cell_width_mm=self._cell_width_spin.value(),
            cell_height_mm=self._cell_height_spin.value(),
            rows=self._rows_spin.value(),
            columns=self._columns_spin.value(),
            row_gap_mm=self._row_gap_spin.value(),
            column_gap_mm=self._column_gap_spin.value(),
            draw_outer_frame=self._draw_frame_check.isChecked(),
            frame_top_mm=self._frame_top_spin.value(),
            frame_bottom_mm=self._frame_bottom_spin.value(),
            frame_left_mm=self._frame_left_spin.value(),
            frame_right_mm=self._frame_right_spin.value(),
            canvas_top_mm=self._canvas_top_spin.value(),
            canvas_bottom_mm=self._canvas_bottom_spin.value(),
            canvas_left_mm=self._canvas_left_spin.value(),
            canvas_right_mm=self._canvas_right_spin.value(),
            include_punctuation=self._include_punctuation_check.isChecked(),
            trim_empty_columns=self._trim_columns_check.isChecked(),
            layout_mode=(
                LAYOUT_VERTICAL
                if self._vertical_layout_radio.isChecked()
                else LAYOUT_HORIZONTAL
            ),
            flow_direction=(
                FLOW_LEFT_TO_RIGHT
                if self._left_to_right_radio.isChecked()
                else FLOW_RIGHT_TO_LEFT
            ),
            scale_mode=(
                SCALE_BY_DPI
                if self._scale_source_radio.isChecked()
                else SCALE_TO_CELL
            ),
            scale_percent=self._scale_percent_spin.value(),
            cell_fill_percent=self._cell_fill_spin.value(),
            auto_scale_enabled=self._auto_scale_check.isChecked(),
            auto_enlarge_threshold=self._enlarge_threshold_spin.value(),
            auto_enlarge_fill_percent=self._enlarge_fill_spin.value(),
            auto_shrink_threshold=self._shrink_threshold_spin.value(),
            auto_shrink_fill_percent=self._shrink_fill_spin.value(),
            paragraph_mode=(
                PARAGRAPH_SKIP_CELLS
                if self._paragraph_skip_radio.isChecked()
                else PARAGRAPH_NEW_COLUMN
            ),
            paragraph_skip_cells=self._paragraph_skip_spin.value(),
            first_title_new_column=self._first_title_check.isChecked(),
            last_title_new_column=self._last_title_check.isChecked(),
            add_annotations=self._annotation_check.isChecked(),
            special_gaps_enabled=self._special_gaps_check.isChecked(),
            row_gap_adjustments=(
                self._parse_special_row_gaps()
                if self._special_gaps_check.isChecked()
                else ()
            ),
            column_gap_adjustments=(
                self._parse_special_column_gaps()
                if self._special_gaps_check.isChecked()
                else ()
            ),
        )
        parameters.validate()
        return parameters

    def _apply_parameters(self, parameters: LayoutParameters) -> None:
        controls: tuple[tuple[Any, Any], ...] = (
            (self._dpi_spin, parameters.dpi),
            (self._cell_width_spin, parameters.cell_width_mm),
            (self._cell_height_spin, parameters.cell_height_mm),
            (self._rows_spin, parameters.rows),
            (self._columns_spin, parameters.columns),
            (self._row_gap_spin, parameters.row_gap_mm),
            (self._column_gap_spin, parameters.column_gap_mm),
            (self._draw_frame_check, parameters.draw_outer_frame),
            (self._frame_top_spin, parameters.frame_top_mm),
            (self._frame_bottom_spin, parameters.frame_bottom_mm),
            (self._frame_left_spin, parameters.frame_left_mm),
            (self._frame_right_spin, parameters.frame_right_mm),
            (self._canvas_top_spin, parameters.canvas_top_mm),
            (self._canvas_bottom_spin, parameters.canvas_bottom_mm),
            (self._canvas_left_spin, parameters.canvas_left_mm),
            (self._canvas_right_spin, parameters.canvas_right_mm),
            (self._include_punctuation_check, parameters.include_punctuation),
            (self._trim_columns_check, parameters.trim_empty_columns),
            (self._scale_percent_spin, parameters.scale_percent),
            (self._cell_fill_spin, parameters.cell_fill_percent),
            (self._auto_scale_check, parameters.auto_scale_enabled),
            (self._enlarge_threshold_spin, parameters.auto_enlarge_threshold),
            (self._enlarge_fill_spin, parameters.auto_enlarge_fill_percent),
            (self._shrink_threshold_spin, parameters.auto_shrink_threshold),
            (self._shrink_fill_spin, parameters.auto_shrink_fill_percent),
            (self._paragraph_skip_spin, parameters.paragraph_skip_cells),
            (self._first_title_check, parameters.first_title_new_column),
            (self._last_title_check, parameters.last_title_new_column),
            (self._annotation_check, parameters.add_annotations),
            (self._special_gaps_check, parameters.special_gaps_enabled),
        )
        blockers = [QSignalBlocker(control) for control, _value in controls]
        blockers.extend(
            [
                QSignalBlocker(self._scale_source_radio),
                QSignalBlocker(self._scale_cell_radio),
                QSignalBlocker(self._vertical_layout_radio),
                QSignalBlocker(self._horizontal_layout_radio),
                QSignalBlocker(self._left_to_right_radio),
                QSignalBlocker(self._right_to_left_radio),
                QSignalBlocker(self._paragraph_skip_radio),
                QSignalBlocker(self._paragraph_column_radio),
                QSignalBlocker(self._special_row_gaps_edit),
                QSignalBlocker(self._special_column_gaps_edit),
            ]
        )
        for control, value in controls:
            if isinstance(control, QCheckBox):
                control.setChecked(bool(value))
            else:
                control.setValue(value)
        self._scale_source_radio.setChecked(parameters.scale_mode == SCALE_BY_DPI)
        self._scale_cell_radio.setChecked(parameters.scale_mode == SCALE_TO_CELL)
        self._vertical_layout_radio.setChecked(parameters.layout_mode == LAYOUT_VERTICAL)
        self._horizontal_layout_radio.setChecked(parameters.layout_mode == LAYOUT_HORIZONTAL)
        self._left_to_right_radio.setChecked(
            parameters.flow_direction == FLOW_LEFT_TO_RIGHT
        )
        self._right_to_left_radio.setChecked(
            parameters.flow_direction == FLOW_RIGHT_TO_LEFT
        )
        self._paragraph_skip_radio.setChecked(
            parameters.paragraph_mode == PARAGRAPH_SKIP_CELLS
        )
        self._paragraph_column_radio.setChecked(
            parameters.paragraph_mode == PARAGRAPH_NEW_COLUMN
        )
        self._special_row_gaps_edit.setText(
            ", ".join(f"{item.row}={item.gap_mm:g}" for item in parameters.row_gap_adjustments)
        )
        self._special_column_gaps_edit.setText(
            ", ".join(
                f"{item.column}={item.gap_mm:g}"
                for item in parameters.column_gap_adjustments
            )
        )
        del blockers
        self._sync_parameter_controls()
        self._scripture_text_changed()

    @Slot()
    def _parameter_mode_changed(self) -> None:
        self._sync_parameter_controls()
        self._schedule_preview()

    @staticmethod
    def _set_inactive_option(control: QRadioButton, inactive: bool) -> None:
        """灰显未选互斥项，同时保留单击切换能力。"""

        value = bool(inactive)
        if control.property("inactiveOption") == value:
            return
        control.setProperty("inactiveOption", value)
        style = control.style()
        style.unpolish(control)
        style.polish(control)
        control.update()

    def _sync_parameter_controls(self) -> None:
        vertical = self._vertical_layout_radio.isChecked()
        left_to_right = self._left_to_right_radio.isChecked()
        self._set_inactive_option(self._vertical_layout_radio, not vertical)
        self._set_inactive_option(self._horizontal_layout_radio, vertical)
        self._set_inactive_option(self._left_to_right_radio, not left_to_right)
        self._set_inactive_option(self._right_to_left_radio, left_to_right)
        self._paragraph_column_radio.setText("段后换列" if vertical else "段后换行")
        self._first_title_check.setText(
            "首经题单列" if vertical else "首经题单行"
        )
        self._last_title_check.setText(
            "尾经题单列" if vertical else "尾经题单行"
        )
        source_mode = self._scale_source_radio.isChecked()
        self._set_inactive_option(self._scale_source_radio, not source_mode)
        self._set_inactive_option(self._scale_cell_radio, source_mode)
        self._source_scale_group.setEnabled(source_mode)
        self._cell_scale_group.setEnabled(not source_mode)
        self._auto_scale_group.setEnabled(
            source_mode and self._auto_scale_check.isChecked()
        )
        self._paragraph_skip_group.setEnabled(
            self._paragraph_skip_radio.isChecked()
        )
        paragraph_skip = self._paragraph_skip_radio.isChecked()
        self._set_inactive_option(self._paragraph_skip_radio, not paragraph_skip)
        self._set_inactive_option(self._paragraph_column_radio, paragraph_skip)
        frame_enabled = self._draw_frame_check.isChecked()
        self._frame_margin_label.setEnabled(frame_enabled)
        self._frame_margin_group.setEnabled(frame_enabled)
        special_enabled = self._special_gaps_check.isChecked()
        for control in self._special_gap_controls:
            control.setEnabled(special_enabled)
        for label in self._special_gap_labels:
            label.setEnabled(special_enabled)

    @Slot()
    def _source_mode_changed(self) -> None:
        system = self._system_source_radio.isChecked()
        self._set_inactive_option(self._system_source_radio, not system)
        self._set_inactive_option(self._external_source_radio, system)
        self._system_library_combo.setEnabled(system)
        self._external_path_edit.setEnabled(not system)
        self._browse_source_button.setEnabled(not system)
        self._source_changed()

    @Slot()
    def _source_changed(self) -> None:
        if self._source_cancel is not None:
            self._source_cancel.set()
        self._source_generation += 1
        self._glyph_index = None
        if not self.is_running and self._source_worker is None:
            self._check_source_button.setEnabled(True)
            self._check_source_button.setText("载入并检查字图")
        self._source_progress.hide()
        self._source_summary.setText("待载入字图后检查")
        self._check_results.clear()
        self._schedule_preview()

    @Slot()
    def _scripture_text_changed(self) -> None:
        """正文或标点规则变化后立即刷新统计，并延迟重绘预览。"""

        available = self._glyph_index.characters if self._glyph_index is not None else None
        parsed = parse_scripture(
            self._text_edit.toPlainText(),
            available,
            self._include_punctuation_check.isChecked(),
        )
        self._update_scripture_statistics(parsed, source_loaded=self._glyph_index is not None)
        self._update_missing_results(parsed if self._glyph_index is not None else None)
        self._schedule_preview()

    def _open_scripture_file(self) -> None:
        if self.is_running or self._text_load_worker is not None:
            return
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "打开经文正文文件",
            self._last_scripture_directory,
            scripture_document_filter(),
        )
        if not path:
            return
        self._last_scripture_directory = os.path.dirname(path)
        self._text_load_generation += 1
        generation = self._text_load_generation
        self._open_text_button.setEnabled(False)
        self._open_text_button.setText("正在读取…")

        worker = FunctionWorker(lambda source=path: load_scripture_text(source))
        self._text_load_worker = worker
        self._workers.add(worker)
        worker.signals.finished.connect(
            lambda result, token=generation, task=worker: self._scripture_file_loaded(
                token, result, task
            )
        )
        worker.signals.failed.connect(
            lambda message, token=generation, task=worker: self._scripture_file_failed(
                token, message, task
            )
        )
        self._thread_pool.start(worker)

    def _scripture_file_loaded(
        self,
        generation: int,
        result: object,
        worker: FunctionWorker,
    ) -> None:
        self._workers.discard(worker)
        if self._text_load_worker is worker:
            self._text_load_worker = None
        if generation != self._text_load_generation or not isinstance(
            result, ScriptureTextResult
        ):
            return
        self._open_text_button.setText("打开文件")
        self._open_text_button.setEnabled(not self.is_running)
        self._text_edit.setPlainText(result.text)
        self._text_edit.document().setModified(False)
        self.status_message.emit(f"已读取正文文件：{result.source_name}；{result.detail}")

    def _scripture_file_failed(
        self,
        generation: int,
        message: str,
        worker: FunctionWorker,
    ) -> None:
        self._workers.discard(worker)
        if self._text_load_worker is worker:
            self._text_load_worker = None
        if generation != self._text_load_generation:
            return
        self._open_text_button.setText("打开文件")
        self._open_text_button.setEnabled(not self.is_running)
        QMessageBox.warning(self, "正文文件读取失败", message)

    def _browse_external_source(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择外部字库目录",
            self._external_path_edit.text() or config.SCRIPT_DIR,
        )
        if directory:
            self._external_path_edit.setText(directory)

    def _browse_output(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择 PSD 输出目录",
            self._output_path_edit.text() or config.SCRIPT_DIR,
        )
        if directory:
            self._output_path_edit.setText(directory)

    def _load_glyph_source(self) -> None:
        if self.is_running:
            return
        if self._source_worker is not None:
            if self._source_cancel is not None:
                self._source_cancel.set()
            self._check_source_button.setEnabled(False)
            self._check_source_button.setText("正在停止检查…")
            self._source_progress.setRange(0, 0)
            self._source_progress.setFormat("正在停止字图检查…")
            return
        system_mode = self._system_source_radio.isChecked()
        source_path = (
            str(self._system_library_combo.currentData() or "")
            if system_mode
            else self._external_path_edit.text().strip()
        )
        if not source_path:
            QMessageBox.warning(self, "字图来源", "请先选择一个字图来源。")
            return
        self._source_generation += 1
        generation = self._source_generation
        cancel_event = threading.Event()
        captured_text = self._text_edit.toPlainText()
        captured_include_punctuation = self._include_punctuation_check.isChecked()
        self._source_cancel = cancel_event
        self._check_source_button.setEnabled(True)
        self._check_source_button.setText("停止检查")
        self._source_progress.show()
        self._source_progress.setRange(0, 0)
        self._source_progress.setFormat("正在打开字图来源…")
        self._source_summary.setText("正在载入并核对字图…")
        self._check_results.clear()

        def load(
            progress_callback: Callable[[object], None],
        ) -> _GlyphSourceLoadResult | None:
            progress_callback(GenerationProgress(0, 1, "正在打开字图来源…", True))

            def index_progress(value: GenerationProgress) -> None:
                if value.indeterminate:
                    progress_callback(value)
                    return
                completed = round(value.completed / max(1, value.total) * 90)
                progress_callback(
                    GenerationProgress(completed, 100, value.message)
                )

            try:
                if system_mode:
                    name = os.path.basename(os.path.normpath(source_path))
                    service = GlyphService.open(name, source_path)
                    if cancel_event.is_set():
                        raise GenerationCancelled("用户停止了字图检查。")
                    index = build_system_glyph_index(
                        service,
                        progress_callback=index_progress,
                        cancel_check=cancel_event.is_set,
                    )
                else:
                    index = build_external_glyph_index(
                        source_path,
                        progress_callback=index_progress,
                        cancel_check=cancel_event.is_set,
                    )
                if cancel_event.is_set():
                    raise GenerationCancelled("用户停止了字图检查。")
                progress_callback(
                    GenerationProgress(90, 100, "正在比对正文与字图库…")
                )
                parsed = parse_scripture(
                    captured_text,
                    index.characters,
                    captured_include_punctuation,
                )
                progress_callback(
                    GenerationProgress(100, 100, "字图与正文核对完成")
                )
                return _GlyphSourceLoadResult(
                    index,
                    captured_text,
                    captured_include_punctuation,
                    parsed,
                )
            except GenerationCancelled:
                return None

        worker = FunctionWorker(load, with_progress=True)
        self._source_worker = worker
        self._workers.add(worker)
        worker.signals.progress.connect(
            lambda value, token=generation, task=worker: self._source_progress_changed(
                token, value, task
            )
        )
        worker.signals.finished.connect(
            lambda result, token=generation, task=worker, cancel=cancel_event: self._source_loaded(
                token, result, task, cancel
            )
        )
        worker.signals.failed.connect(
            lambda message, token=generation, task=worker, cancel=cancel_event: self._source_failed(
                token, message, task, cancel
            )
        )
        self._thread_pool.start(worker)

    def _source_progress_changed(
        self,
        generation: int,
        value: object,
        worker: FunctionWorker,
    ) -> None:
        if (
            generation != self._source_generation
            or self._source_worker is not worker
            or not isinstance(value, GenerationProgress)
        ):
            return
        self._source_progress.show()
        if value.indeterminate:
            self._source_progress.setRange(0, 0)
            self._source_progress.setFormat(value.message)
            return
        self._source_progress.setRange(0, max(1, value.total))
        self._source_progress.setValue(value.completed)
        self._source_progress.setFormat(f"{value.message} · %p%")

    def _finish_source_worker(self, worker: FunctionWorker) -> bool:
        self._workers.discard(worker)
        if self._source_worker is not worker:
            return False
        self._source_worker = None
        self._source_cancel = None
        self._check_source_button.setText("载入并检查字图")
        self._check_source_button.setEnabled(not self.is_running)
        return True

    def _source_loaded(
        self,
        generation: int,
        result: object,
        worker: FunctionWorker,
        cancel_event: threading.Event,
    ) -> None:
        active = self._finish_source_worker(worker)
        if not active or generation != self._source_generation:
            return
        if cancel_event.is_set():
            self._source_progress.setRange(0, 1)
            self._source_progress.setValue(0)
            self._source_progress.setFormat("字图检查已停止")
            self._source_summary.setText("检查已停止")
            return
        if not isinstance(result, _GlyphSourceLoadResult):
            self._source_progress.hide()
            self._source_summary.setText("检查失败")
            return
        self._glyph_index = result.index
        self._source_progress.setRange(0, 1)
        self._source_progress.setValue(1)
        self._source_progress.setFormat(
            f"检查完成：{len(result.index.characters)} 个字，"
            f"{result.index.variant_count} 个字形"
        )
        parsed = (
            result.parsed
            if result.scripture_text == self._text_edit.toPlainText()
            and result.include_punctuation
            == self._include_punctuation_check.isChecked()
            else None
        )
        self._rebuild_layout(parsed)

    def _source_failed(
        self,
        generation: int,
        message: str,
        worker: FunctionWorker,
        cancel_event: threading.Event,
    ) -> None:
        active = self._finish_source_worker(worker)
        if not active or generation != self._source_generation:
            return
        if cancel_event.is_set():
            self._source_progress.setRange(0, 1)
            self._source_progress.setValue(0)
            self._source_progress.setFormat("字图检查已停止")
            self._source_summary.setText("检查已停止")
            return
        self._source_summary.setText("检查失败")
        self._check_results.clear()
        QMessageBox.warning(self, "字图来源载入失败", message)

    @Slot()
    def _schedule_preview(self) -> None:
        self._update_template_state()
        if not self.is_running:
            if self._preview_cancel is not None:
                self._preview_cancel.set()
                self._preview_generation += 1
            self._preview_timer.start()

    def _rebuild_layout(self, preparsed: ParsedScripture | None = None) -> None:
        text = self._text_edit.toPlainText()
        include_punctuation = self._include_punctuation_check.isChecked()
        if self._glyph_index is None:
            # 未载入字图时仍独立统计正文，缺字信息等待来源核对。
            parsed_without_source = parse_scripture(
                text,
                None,
                include_punctuation,
            )
            self._update_scripture_statistics(parsed_without_source, source_loaded=False)
            self._update_missing_results(None)
            self._boards = ()
            self._show_preview_message("请先载入并检查字图来源")
            self._update_board_controls()
            return
        parsed = preparsed or parse_scripture(
            text,
            self._glyph_index.characters,
            include_punctuation,
        )
        self._update_scripture_statistics(parsed, source_loaded=True)
        self._update_missing_results(parsed)
        try:
            parameters = self._collect_parameters()
            self._boards = allocate_boards(parsed, parameters)
        except ValueError as exc:
            self._boards = ()
            self._show_preview_message(str(exc))
            self._update_board_controls()
            return
        if not self._boards:
            self._current_board = 0
            self._show_preview_message("请输入经文正文")
            self._update_board_controls()
            return
        self._current_board = min(self._current_board, len(self._boards) - 1)
        self._update_board_controls()
        self._render_current_preview(parameters)

    def _update_scripture_statistics(
        self,
        parsed: ParsedScripture,
        *,
        source_loaded: bool,
    ) -> None:
        """更新正文框下方的即时文字统计。"""

        self._scripture_statistics.setText(
            f"总字数：{parsed.total_characters}　"
            f"去重后字数：{parsed.unique_characters}　"
            f"标点符号数：{parsed.punctuation}"
        )

    def _update_missing_results(self, parsed: ParsedScripture | None) -> None:
        if parsed is None:
            self._source_summary.setText("待载入字图后检查")
            self._check_results.clear()
            return
        missing_kinds = len(parsed.missing)
        missing_occurrences = sum(parsed.missing.values())
        if missing_kinds:
            self._source_summary.setText(
                f"缺失 {missing_kinds} 个字，共 {missing_occurrences} 处"
            )
            self._check_results.setPlainText(
                "、".join(
                    f"{character}（{count}）"
                    for character, count in parsed.missing.items()
                )
            )
        else:
            self._source_summary.setText("无缺失文字")
            self._check_results.clear()

    def _render_current_preview(self, parameters: LayoutParameters | None = None) -> None:
        if not self._boards or self._glyph_index is None:
            return
        try:
            captured_parameters = parameters or self._collect_parameters()
        except ValueError as exc:
            self._show_preview_message(str(exc))
            return
        board = self._boards[self._current_board]
        index = self._glyph_index
        show_guides = self._preview_guides_check.isChecked()
        grid = compute_grid(
            captured_parameters,
            board.effective_columns,
            board.effective_rows,
        )
        document_size = QSize(grid.canvas_width, grid.canvas_height)
        self._preview_document_size = document_size
        display_size = self._preview_target_size(document_size)
        device_ratio = max(1.0, float(self.devicePixelRatioF()))
        requested_size = QSize(
            max(1, round(display_size.width() * device_ratio)),
            max(1, round(display_size.height() * device_ratio)),
        )
        render_size = self._bounded_preview_render_size(
            requested_size,
            document_size,
        )
        render_context = (
            board,
            captured_parameters,
            show_guides,
            id(index),
        )
        if self._reuse_current_preview(render_context, render_size):
            return
        self._preview_generation += 1
        generation = self._preview_generation
        if self._preview_cancel is not None:
            self._preview_cancel.set()
        cancel_event = threading.Event()
        self._preview_cancel = cancel_event
        if self._preview_image.isNull():
            self._preview_label.set_message("正在生成高清字图预览…")
        self._preview_progress.show()
        self._preview_progress.setRange(0, 0)
        self._preview_progress.setFormat("正在准备高清预览…")

        def render(progress_callback: Callable[[object], None]) -> QImage:
            try:
                image = render_board_preview(
                    board,
                    index,
                    captured_parameters,
                    (render_size.width(), render_size.height()),
                    show_guides=show_guides,
                    progress_callback=progress_callback,
                    cancel_check=cancel_event.is_set,
                )
                try:
                    return QImage(ImageQt(image)).copy()
                finally:
                    image.close()
            except GenerationCancelled:
                return QImage()

        worker = FunctionWorker(render, with_progress=True)
        self._preview_worker = worker
        self._workers.add(worker)
        worker.signals.progress.connect(
            lambda value, token=generation, task=worker: self._preview_progress_changed(
                token, value, task
            )
        )
        worker.signals.finished.connect(
            lambda result,
            token=generation,
            task=worker,
            ratio=device_ratio,
            context=render_context: self._preview_ready(
                token,
                result,
                task,
                ratio,
                context,
            )
        )
        worker.signals.failed.connect(
            lambda message, token=generation, task=worker: self._preview_failed(
                token,
                message,
                task,
            )
        )
        self._thread_pool.start(worker)

    @Slot(bool)
    def _preview_guides_changed(self, _checked: bool) -> None:
        """只刷新当前预览，不改变模板参数或最终输出设置。"""

        if self._boards and self._glyph_index is not None and not self.is_running:
            self._render_current_preview()

    def _preview_ready(
        self,
        generation: int,
        result: object,
        worker: FunctionWorker,
        device_ratio: float = 1.0,
        render_context: object | None = None,
    ) -> None:
        self._workers.discard(worker)
        if self._preview_worker is worker:
            self._preview_worker = None
            self._preview_cancel = None
        if generation != self._preview_generation or not isinstance(result, QImage):
            return
        self._preview_image = result
        self._preview_device_ratio = max(1.0, device_ratio)
        if render_context is not None:
            self._preview_render_context = render_context
        self._apply_preview_zoom()
        self._preview_progress.hide()

    def _preview_progress_changed(
        self,
        generation: int,
        value: object,
        worker: FunctionWorker,
    ) -> None:
        if (
            generation != self._preview_generation
            or self._preview_worker is not worker
            or not isinstance(value, GenerationProgress)
        ):
            return
        self._preview_progress.show()
        if value.indeterminate:
            self._preview_progress.setRange(0, 0)
            self._preview_progress.setFormat(value.message)
            return
        self._preview_progress.setRange(0, max(1, value.total))
        self._preview_progress.setValue(value.completed)
        self._preview_progress.setFormat(f"{value.message} · %p%")

    def _preview_failed(self, generation: int, message: str, worker: FunctionWorker) -> None:
        self._workers.discard(worker)
        if self._preview_worker is worker:
            self._preview_worker = None
            self._preview_cancel = None
        if generation == self._preview_generation:
            self._preview_progress.hide()
            self._show_preview_message(f"预览生成失败：{message}")

    def _show_preview_message(self, message: str) -> None:
        self._preview_quality_timer.stop()
        if self._preview_cancel is not None:
            self._preview_cancel.set()
        self._preview_generation += 1
        self._preview_image = QImage()
        self._preview_document_size = QSize()
        self._preview_render_context = None
        self._preview_label.set_message(message)
        self._preview_progress.hide()

    def _zoom_changed(self, text: str) -> None:
        previous_zoom = self._preview_zoom
        if text == "适合窗口":
            self._preview_zoom = 0.0
        else:
            try:
                requested_zoom = float(text.rstrip("%")) / 100
                self._preview_zoom = max(
                    self._minimum_preview_zoom(),
                    min(8.0, requested_zoom),
                )
            except ValueError:
                return
        self._apply_preview_zoom()
        if abs(self._preview_zoom - previous_zoom) > 1e-9:
            self._schedule_quality_preview()

    @Slot(object, float)
    def _zoom_preview_at(self, viewport_position: object, factor: float) -> None:
        """以鼠标位置为锚点连续缩放预览。"""

        if self._preview_image.isNull() or factor <= 0:
            return
        position = (
            QPointF(viewport_position)
            if isinstance(viewport_position, (QPoint, QPointF))
            else QPointF(self._preview_scroll.viewport().rect().center())
        )
        label_size = self._preview_label.size()
        label_origin = self._preview_label.mapTo(
            self._preview_scroll.viewport(),
            QPoint(0, 0),
        )
        anchor_x = max(
            0.0,
            min(1.0, (position.x() - label_origin.x()) / max(1, label_size.width())),
        )
        anchor_y = max(
            0.0,
            min(1.0, (position.y() - label_origin.y()) / max(1, label_size.height())),
        )
        current_zoom = (
            self._preview_zoom
            if self._preview_zoom > 0
            else self._fit_preview_zoom()
        )
        new_zoom = max(
            self._minimum_preview_zoom(),
            min(8.0, current_zoom * factor),
        )
        if abs(new_zoom - current_zoom) <= 1e-9:
            return
        self._preview_zoom = new_zoom
        with QSignalBlocker(self._zoom_combo):
            self._zoom_combo.setEditText(
                self._format_preview_zoom(self._preview_zoom)
            )
        self._apply_preview_zoom()
        self._schedule_quality_preview()

        def restore_anchor() -> None:
            new_origin = self._preview_label.mapTo(
                self._preview_scroll.viewport(),
                QPoint(0, 0),
            )
            target_x = new_origin.x() + anchor_x * self._preview_label.width()
            target_y = new_origin.y() + anchor_y * self._preview_label.height()
            horizontal = self._preview_scroll.horizontalScrollBar()
            vertical = self._preview_scroll.verticalScrollBar()
            horizontal.setValue(
                horizontal.value() + round(target_x - position.x())
            )
            vertical.setValue(vertical.value() + round(target_y - position.y()))

        QTimer.singleShot(0, restore_anchor)

    def _apply_preview_zoom(self) -> None:
        if self._preview_image.isNull():
            return
        target = self._preview_target_size()
        self._preview_label.set_image(self._preview_image)
        self._preview_label.resize(target)

    def _preview_target_size(self, document_size: QSize | None = None) -> QSize:
        source_size = document_size or self._preview_document_size
        if source_size.isEmpty():
            source_size = self._preview_image.size()
        if source_size.isEmpty():
            return QSize(320, 320)
        if self._preview_zoom <= 0:
            target = QSize(source_size)
            viewport = self._preview_scroll.viewport().size()
            target.scale(
                max(1, viewport.width() - 24),
                max(1, viewport.height() - 24),
                Qt.AspectRatioMode.KeepAspectRatio,
            )
            return target
        return QSize(
            max(1, round(source_size.width() * self._preview_zoom)),
            max(1, round(source_size.height() * self._preview_zoom)),
        )

    def _fit_preview_zoom(self, document_size: QSize | None = None) -> float:
        """返回完整显示当前版面所需的真实缩放比例。"""

        source_size = document_size or self._preview_document_size
        if source_size.isEmpty():
            source_size = self._preview_image.size()
        if source_size.isEmpty():
            return 0.1
        viewport = self._preview_scroll.viewport().size()
        available_width = max(1, viewport.width() - 24)
        available_height = max(1, viewport.height() - 24)
        return max(
            0.0001,
            min(
                available_width / source_size.width(),
                available_height / source_size.height(),
            ),
        )

    def _minimum_preview_zoom(self) -> float:
        """大版面允许缩小到适合窗口比例，普通版面仍以 10% 为下限。"""

        return min(0.1, self._fit_preview_zoom())

    @staticmethod
    def _format_preview_zoom(zoom: float) -> str:
        percent = max(0.0, zoom * 100)
        if percent >= 10 or abs(percent - round(percent)) < 0.05:
            value = f"{percent:.0f}"
        elif percent >= 1:
            value = f"{percent:.1f}".rstrip("0").rstrip(".")
        else:
            value = f"{percent:.2f}".rstrip("0").rstrip(".")
        return f"{value}%"

    @staticmethod
    def _bounded_preview_render_size(requested: QSize, document: QSize) -> QSize:
        """限制当前页高清位图约为 1200 万像素，避免预览挤占输出内存。"""

        result = QSize(
            max(1, min(requested.width(), document.width())),
            max(1, min(requested.height(), document.height())),
        )
        maximum_pixels = 12_000_000
        pixels = result.width() * result.height()
        if pixels > maximum_pixels:
            factor = (maximum_pixels / pixels) ** 0.5
            result = QSize(
                max(1, round(result.width() * factor)),
                max(1, round(result.height() * factor)),
            )
        return result

    def _reuse_current_preview(
        self,
        render_context: object,
        requested_size: QSize,
    ) -> bool:
        """现有位图足够清晰时只调整显示尺寸，不重复后台绘制。"""

        current_size = self._preview_image.size()
        if (
            self._preview_image.isNull()
            or self._preview_render_context != render_context
            or current_size.width() < requested_size.width()
            or current_size.height() < requested_size.height()
        ):
            return False
        if self._preview_cancel is not None:
            self._preview_cancel.set()
            self._preview_generation += 1
        self._apply_preview_zoom()
        self._preview_progress.hide()
        return True

    def _schedule_quality_preview(self) -> None:
        if self._boards and self._glyph_index is not None and not self.is_running:
            self._preview_quality_timer.start()

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        if self._preview_zoom <= 0:
            QTimer.singleShot(0, self._apply_preview_zoom)
            self._schedule_quality_preview()

    def _change_board(self, offset: int) -> None:
        if not self._boards:
            return
        new_index = max(0, min(len(self._boards) - 1, self._current_board + offset))
        if new_index == self._current_board:
            return
        self._current_board = new_index
        self._update_board_controls()
        self._render_current_preview()

    def _update_board_controls(self) -> None:
        total = len(self._boards)
        current = self._current_board + 1 if total else 0
        self._board_label.setText(f"{current} / {total}")
        self._previous_button.setEnabled(total > 0 and self._current_board > 0)
        self._next_button.setEnabled(total > 0 and self._current_board + 1 < total)
        characters = sum(board.character_count for board in self._boards)
        self._board_summary.setText(
            f"共 {total} 版 · {characters} 个文字位置" if total else "暂无版面"
        )

    def _parse_board_selection(self, total_boards: int | None = None) -> set[int]:
        total = len(self._boards) if total_boards is None else total_boards
        text = self._board_selection_edit.text().strip()
        if not text or text in {"全部", "all", "ALL"}:
            return set(range(1, total + 1))
        selected: set[int] = set()
        for part in re.split(r"[,，;；]", text):
            value = part.strip()
            if not value:
                continue
            match = re.fullmatch(r"(\d+)\s*[-~至]\s*(\d+)", value)
            if match:
                start, end = int(match.group(1)), int(match.group(2))
                if start > end:
                    start, end = end, start
                selected.update(range(start, end + 1))
            elif value.isdigit():
                selected.add(int(value))
            else:
                raise ValueError("生成版面应填写“全部”或类似“1,3-5”的版号。")
        invalid = sorted(number for number in selected if number < 1 or number > total)
        if invalid:
            raise ValueError(f"版号超出范围：{', '.join(map(str, invalid))}")
        if not selected:
            raise ValueError("至少选择一个要生成的版面。")
        return selected

    def _confirm_missing_characters(self, missing: Mapping[str, int]) -> bool:
        """确认本次生成中缺字留空的风险。"""

        if not missing:
            return True
        total = sum(missing.values())
        items = list(missing.items())
        detail = "、".join(
            f"“{character}”×{count}" for character, count in items[:24]
        )
        if len(items) > 24:
            detail += f"，另有 {len(items) - 24} 种未展开"
        dialog = QMessageBox(self)
        dialog.setWindowTitle("正文存在缺字")
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setText(
            f"本次选择的版面中有 {len(missing)} 种、共 {total} 处缺字：\n\n"
            f"{detail}\n\n"
            "继续生成时，缺字对应的单元格将保留空白，其他文字正常生成。"
        )
        continue_button = dialog.addButton(
            "继续生成",
            QMessageBox.ButtonRole.AcceptRole,
        )
        cancel_button = dialog.addButton(
            "取消",
            QMessageBox.ButtonRole.RejectRole,
        )
        dialog.setDefaultButton(cancel_button)
        dialog.setEscapeButton(cancel_button)
        dialog.exec()
        return dialog.clickedButton() is continue_button

    def _resolve_conflicts(
        self,
        output_dir: str,
        selected: set[int],
        total_boards: int,
        dpi: int,
        output_base_name: str,
        plans: Mapping[int, BoardOutputPlan],
    ) -> dict[int, str] | None:
        conflicts = [
            number
            for number in sorted(selected)
            if os.path.exists(
                board_output_path(
                    output_dir,
                    number,
                    dpi,
                    output_base_name,
                    total_boards=total_boards,
                    extension=plans[number].extension,
                )
            )
        ]
        decisions: dict[int, str] = {}
        apply_all = ""
        for number in conflicts:
            if apply_all:
                decisions[number] = apply_all
                continue
            dialog = QMessageBox(self)
            dialog.setWindowTitle("输出文件已存在")
            dialog.setIcon(QMessageBox.Icon.Question)
            dialog.setText(
                f"第 {number} 版的 {plans[number].format_name} 文件已经存在，"
                "请选择处理方式。"
            )
            overwrite = dialog.addButton("覆盖", QMessageBox.ButtonRole.AcceptRole)
            skip = dialog.addButton("跳过", QMessageBox.ButtonRole.ActionRole)
            cancel = dialog.addButton("取消", QMessageBox.ButtonRole.RejectRole)
            apply_check = QCheckBox("为所有重复文件执行此操作")
            dialog.setCheckBox(apply_check)
            dialog.setDefaultButton(skip)
            dialog.setEscapeButton(cancel)
            dialog.exec()
            clicked = dialog.clickedButton()
            if clicked is overwrite:
                decision = CONFLICT_OVERWRITE
            elif clicked is skip:
                decision = CONFLICT_SKIP
            else:
                return None
            decisions[number] = decision
            if apply_check.isChecked():
                apply_all = decision
        return decisions

    def _start_generation(self) -> None:
        if self.is_running:
            return
        if self._glyph_index is None or not self._boards:
            QMessageBox.warning(self, "无法生成", "请先载入字图、输入经文并生成有效预览。")
            return
        try:
            parameters = self._collect_parameters()
            parsed = parse_scripture(
                self._text_edit.toPlainText(),
                self._glyph_index.characters,
                self._include_punctuation_check.isChecked(),
            )
            boards = allocate_boards(parsed, parameters)
            if not boards:
                raise ValueError("请输入要生成的经文正文。")
            selected = self._parse_board_selection(len(boards))
            output_dir = self._output_path_edit.text().strip()
            if not output_dir:
                raise ValueError("请选择 PSD 输出目录。")
            output_base_name = self._output_name_edit.text().strip()
            compress_psd = self._compress_psd_check.isChecked()
            output_format = self._output_format_combo.currentText()
            board_output_path(
                output_dir,
                1,
                parameters.dpi,
                output_base_name,
                total_boards=len(boards),
            )
            os.makedirs(output_dir, exist_ok=True)
            selected_missing: Counter[str] = Counter(
                placement.character
                for board in boards
                if board.number in selected
                for placement in board.placements
                if placement.missing
            )
            if not self._confirm_missing_characters(selected_missing):
                return
            plans = {
                board.number: plan_board_output(
                    board,
                    self._glyph_index,
                    parameters,
                    output_format,
                )
                for board in boards
                if board.number in selected
            }
            decisions = self._resolve_conflicts(
                output_dir,
                selected,
                len(boards),
                parameters.dpi,
                output_base_name,
                plans,
            )
            if decisions is None:
                return
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "无法生成", str(exc))
            return
        glyph_index = self._glyph_index
        cancel_event = threading.Event()
        self._generation_cancel = cancel_event

        def generate(progress_callback: Callable[[object], None]) -> GenerationResult:
            try:
                return generate_psd_boards(
                    boards,
                    glyph_index,
                    parameters,
                    output_dir,
                    output_base_name=output_base_name,
                    compress_psd=compress_psd,
                    output_format=output_format,
                    selected_boards=selected,
                    conflict_decisions=decisions,
                    progress_callback=progress_callback,
                    cancel_event=cancel_event,
                )
            except GenerationCancelled:
                return GenerationResult((), True)

        worker = FunctionWorker(generate, with_progress=True)
        self._generation_worker = worker
        self._generation_started_at = time.perf_counter()
        self._workers.add(worker)
        worker.signals.progress.connect(self._generation_progress)
        worker.signals.finished.connect(
            lambda result, task=worker: self._generation_finished(result, task)
        )
        worker.signals.failed.connect(
            lambda message, task=worker: self._generation_failed(message, task)
        )
        self._set_running(True)
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setFormat("正在准备分层 PSD…")
        self._thread_pool.start(worker)

    @Slot(object)
    def _generation_progress(self, value: object) -> None:
        if not isinstance(value, GenerationProgress):
            return
        if value.indeterminate:
            self._progress_bar.setRange(0, 0)
            self._progress_bar.setFormat(value.message)
            return
        self._progress_bar.setRange(0, value.total)
        self._progress_bar.setValue(value.completed)
        self._progress_bar.setFormat(f"{value.message} · %p%")

    def _generation_finished(self, result: object, worker: FunctionWorker) -> None:
        elapsed_text = self._generation_elapsed_text()
        self._finish_worker(worker)
        if not isinstance(result, GenerationResult):
            QMessageBox.warning(
                self,
                "生成失败",
                f"排版任务返回了无法识别的结果。\n\n{elapsed_text}",
            )
            return
        if result.stopped:
            self._progress_bar.setRange(0, 1)
            self._progress_bar.setValue(0)
            self._progress_bar.setFormat("任务已停止；已完整生成的版面予以保留")
            self.status_message.emit("通用经文排版已停止")
            QMessageBox.information(
                self,
                "通用经文排版已停止",
                "生成任务已停止，已完整生成的版面予以保留。\n\n"
                f"{elapsed_text}",
            )
            return
        completed = sum(not board.skipped for board in result.boards)
        skipped = sum(board.skipped for board in result.boards)
        missing = sum(board.missing_characters for board in result.boards)
        self._progress_bar.setValue(self._progress_bar.maximum())
        summary = f"生成完成：新增或覆盖 {completed} 版，跳过 {skipped} 版"
        if missing:
            summary += f"，缺字留空 {missing} 处"
        self._progress_bar.setFormat(summary)
        detail = f"已生成 {completed} 个分层文件，跳过 {skipped} 个已有文件。"
        if missing:
            detail += f"\n\n其中 {missing} 处缺字单元格已按确认保留空白。"
        detail += f"\n\n{elapsed_text}"
        QMessageBox.information(
            self,
            "通用经文排版完成",
            detail,
        )

    def _generation_failed(self, message: str, worker: FunctionWorker) -> None:
        elapsed_text = self._generation_elapsed_text()
        self._finish_worker(worker)
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("生成失败")
        QMessageBox.critical(
            self,
            "通用经文排版失败",
            f"{message}\n\n{elapsed_text}",
        )

    def _generation_elapsed_text(self) -> str:
        if self._generation_started_at is None:
            elapsed = 0.0
        else:
            elapsed = max(0.0, time.perf_counter() - self._generation_started_at)
        return f"总耗时：{format_elapsed_time(elapsed)}"

    def _finish_worker(self, worker: FunctionWorker) -> None:
        self._workers.discard(worker)
        if self._generation_worker is worker:
            self._generation_worker = None
            self._generation_cancel = None
            self._generation_started_at = None
        self._set_running(False)

    def _stop_generation(self) -> None:
        if self._generation_cancel is None:
            return
        self._generation_cancel.set()
        self._stop_button.setEnabled(False)
        self._progress_bar.setFormat("正在立即停止并清理当前版临时文件…")

    def _set_running(self, running: bool) -> None:
        self._start_button.setEnabled(not running)
        self._stop_button.setEnabled(running)
        self._home_button.setEnabled(not running)
        self._check_source_button.setEnabled(not running and self._source_worker is None)
        self._open_text_button.setEnabled(not running and self._text_load_worker is None)
        self._template_combo.setEnabled(not running)
        self._update_template_state()

    def _request_home(self) -> None:
        if self.is_running:
            QMessageBox.information(
                self,
                "排版任务正在执行",
                "请先点击“停止生成”，任务停止后即可返回首页。",
            )
            return
        self.home_requested.emit()

    def shutdown(self) -> None:
        """程序关闭时请求立即终止后台生成。"""

        self._preview_generation += 1
        self._source_generation += 1
        self._text_load_generation += 1
        self._preview_timer.stop()
        self._preview_quality_timer.stop()
        if self._preview_cancel is not None:
            self._preview_cancel.set()
        if self._source_cancel is not None:
            self._source_cancel.set()
        if self._generation_cancel is not None:
            self._generation_cancel.set()
