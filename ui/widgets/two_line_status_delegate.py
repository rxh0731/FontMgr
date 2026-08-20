"""字形列表中主状态与辅助提示的双行绘制。"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QSize, Qt
from PySide6.QtGui import QColor, QFontMetrics, QPainter
from PySide6.QtWidgets import QApplication, QStyle, QStyledItemDelegate, QStyleOptionViewItem


PRIMARY_COLOR_ROLE = int(Qt.ItemDataRole.UserRole) + 41
SECONDARY_COLOR_ROLE = int(Qt.ItemDataRole.UserRole) + 42
STATUS_LINE_GAP = 2
STATUS_VERTICAL_MARGIN = 4


def set_two_line_status(
    item: object,
    column: int,
    primary: str,
    secondary: str,
    primary_color: QColor | str | None = None,
    secondary_color: QColor | str | None = None,
) -> None:
    """设置可测试的双行文字及两行独立颜色。"""

    primary_text = str(primary or "-")
    secondary_text = str(secondary or "-")
    item.setText(column, f"{primary_text}\n{secondary_text}")
    item.setData(column, PRIMARY_COLOR_ROLE, _color_name(primary_color))
    item.setData(column, SECONDARY_COLOR_ROLE, _color_name(secondary_color))
    if primary_color is not None:
        value = primary_color if isinstance(primary_color, QColor) else QColor(str(primary_color))
        if value.isValid():
            item.setForeground(column, value)
    item.setTextAlignment(
        column,
        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
    )
    metrics = QFontMetrics(item.font(column))
    minimum_height = (
        metrics.lineSpacing() * 2
        + STATUS_LINE_GAP
        + STATUS_VERTICAL_MARGIN * 2
    )
    current_hint = item.sizeHint(column)
    item.setSizeHint(
        column,
        QSize(current_hint.width(), max(current_hint.height(), minimum_height)),
    )


class TwoLineStatusDelegate(QStyledItemDelegate):
    """在同一列中分别绘制主状态和辅助提示，避免为大字库创建控件。"""

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: object,
    ) -> None:
        prepared = QStyleOptionViewItem(option)
        self.initStyleOption(prepared, index)
        lines = str(prepared.text or "").split("\n", 1)
        primary = lines[0]
        secondary = lines[1] if len(lines) > 1 else ""
        prepared.text = ""
        style = prepared.widget.style() if prepared.widget is not None else QApplication.style()
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, prepared, painter, prepared.widget)

        metrics = QFontMetrics(prepared.font)
        text_rect = prepared.rect.adjusted(
            6,
            STATUS_VERTICAL_MARGIN,
            -6,
            -STATUS_VERTICAL_MARGIN,
        )
        line_height = metrics.lineSpacing()
        block_height = line_height * 2 + STATUS_LINE_GAP
        block_top = text_rect.top() + max(0, (text_rect.height() - block_height) // 2)
        horizontal_alignment = (
            prepared.displayAlignment & Qt.AlignmentFlag.AlignHorizontal_Mask
        ) or Qt.AlignmentFlag.AlignLeft
        painter.save()
        painter.setClipRect(prepared.rect)
        painter.setFont(prepared.font)
        painter.setPen(
            _role_color(index.data(PRIMARY_COLOR_ROLE), prepared.palette.text().color())
        )
        painter.drawText(
            QPoint(
                _line_x(text_rect, metrics, primary, horizontal_alignment),
                block_top + metrics.ascent(),
            ),
            primary,
        )
        painter.setPen(
            _role_color(index.data(SECONDARY_COLOR_ROLE), prepared.palette.text().color())
        )
        painter.drawText(
            QPoint(
                _line_x(text_rect, metrics, secondary, horizontal_alignment),
                block_top + line_height + STATUS_LINE_GAP + metrics.ascent(),
            ),
            secondary,
        )
        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index: object) -> QSize:
        size = super().sizeHint(option, index)
        metrics = option.fontMetrics
        lines = str(index.data(Qt.ItemDataRole.DisplayRole) or "").split("\n", 1)
        content_width = max(
            (metrics.horizontalAdvance(line) for line in lines),
            default=0,
        ) + 12
        content_height = (
            metrics.lineSpacing() * 2
            + STATUS_LINE_GAP
            + STATUS_VERTICAL_MARGIN * 2
        )
        return QSize(max(size.width(), content_width), max(size.height(), content_height))


def _color_name(color: QColor | str | None) -> str:
    if color is None:
        return ""
    value = color if isinstance(color, QColor) else QColor(str(color))
    return value.name() if value.isValid() else ""


def _role_color(value: object, fallback: QColor) -> QColor:
    color = QColor(str(value or ""))
    return color if color.isValid() else fallback


def _line_x(
    rect: object,
    metrics: QFontMetrics,
    text: str,
    alignment: Qt.AlignmentFlag,
) -> int:
    text_width = metrics.horizontalAdvance(text)
    if alignment & Qt.AlignmentFlag.AlignRight:
        return max(rect.left(), rect.right() - text_width + 1)
    if alignment & Qt.AlignmentFlag.AlignHCenter:
        return rect.left() + max(0, (rect.width() - text_width) // 2)
    return rect.left()
