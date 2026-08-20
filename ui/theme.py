"""应用统一视觉主题。"""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


COLORS = {
    "background": "#161A21",
    "surface": "#202630",
    "surface_alt": "#282F3A",
    "border": "#37404D",
    "text": "#F1F4F8",
    "muted": "#A6B0BE",
    "accent": "#4DA3FF",
    "accent_hover": "#6AB2FF",
    "success": "#48C78E",
    "warning": "#F2B84B",
}


def apply_theme(app: QApplication) -> None:
    """应用全局深色调色板与控件样式。"""
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(COLORS["background"]))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(COLORS["text"]))
    palette.setColor(QPalette.ColorRole.Base, QColor(COLORS["surface"]))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(COLORS["surface_alt"]))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(COLORS["surface_alt"]))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(COLORS["text"]))
    palette.setColor(QPalette.ColorRole.Text, QColor(COLORS["text"]))
    palette.setColor(QPalette.ColorRole.Button, QColor(COLORS["surface_alt"]))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(COLORS["text"]))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(COLORS["accent"]))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
    app.setPalette(palette)
    app.setStyleSheet(
        f"""
        QWidget {{
            color: {COLORS['text']};
            font-family: "Microsoft YaHei UI", "Microsoft YaHei", sans-serif;
            font-size: 14px;
        }}
        QMainWindow, QDialog {{ background: {COLORS['background']}; }}
        QLabel[role="muted"] {{ color: {COLORS['muted']}; }}
        QLabel[role="pageTitle"] {{ font-size: 26px; font-weight: 700; }}
        QPushButton {{
            min-height: 38px;
            padding: 0 18px;
            border: 1px solid {COLORS['border']};
            border-radius: 7px;
            background: {COLORS['surface_alt']};
        }}
        QPushButton:hover {{ border-color: {COLORS['accent']}; background: #303947; }}
        QPushButton:pressed {{ background: #1B75D0; }}
        QPushButton:disabled {{ color: #68717E; background: #242A33; border-color: #303640; }}
        QPushButton[role="primary"] {{
            border-color: {COLORS['accent']};
            background: {COLORS['accent']};
            color: #FFFFFF;
            font-weight: 600;
        }}
        QPushButton[role="primary"]:hover {{ background: {COLORS['accent_hover']}; }}
        QPushButton[role="primary"]:disabled {{
            color: #737D8A;
            background: #252B34;
            border-color: #343C47;
        }}
        QFrame[role="card"] {{
            background: {COLORS['surface']};
            border: 1px solid {COLORS['border']};
            border-radius: 10px;
        }}
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
            min-height: 36px;
            padding: 0 10px;
            border: 1px solid {COLORS['border']};
            border-radius: 6px;
            background: {COLORS['surface']};
            selection-background-color: {COLORS['accent']};
        }}
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
            border-color: {COLORS['accent']};
        }}
        QScrollArea {{ border: none; background: transparent; }}
        QScrollBar:vertical {{ width: 10px; background: {COLORS['background']}; }}
        QScrollBar::handle:vertical {{ min-height: 28px; border-radius: 5px; background: #46515F; }}
        QStatusBar {{ color: {COLORS['muted']}; background: #12161C; }}
        QToolTip {{
            color: {COLORS['text']};
            background: {COLORS['surface_alt']};
            border: 1px solid {COLORS['border']};
        }}
        QLabel#homeTitle {{ font-size: 27px; font-weight: 700; color: #F4F7FB; }}
        QLabel#homeSubtitle {{ color: {COLORS['muted']}; font-size: 12px; }}
        QLabel#sectionTitle {{ font-size: 16px; font-weight: 700; color: #E8EDF5; }}
        QLabel#cardTitle {{ font-size: 14px; font-weight: 700; color: #E8EDF5; }}
        QLabel#cardDetail, QLabel#stageStatus {{ color: {COLORS['muted']}; font-size: 11px; }}
        QLabel#cardArrow {{ color: #768396; font-size: 25px; }}
        QLabel#metricValue {{ color: #EAF1FA; font-size: 21px; font-weight: 700; }}
        QFrame#toolCard, QFrame#stageCard, QFrame#exportStageCard, QFrame#selectorPanel {{
            background: rgba(31, 37, 47, 242);
            border: 1px solid {COLORS['border']};
            border-radius: 7px;
        }}
        QFrame#libraryFlowPanel, QFrame#libraryCreatePanel {{
            background: rgba(24, 29, 37, 218);
            border: 1px solid #4A5666;
            border-radius: 7px;
        }}
        QFrame#toolCard:hover {{ background: #29313D; border-color: #526177; }}
        QFrame#exportStageCard {{ background: rgba(28, 48, 72, 246); border-color: #416A9D; }}
        QFrame#stageCard[available="false"], QFrame#exportStageCard[available="false"] {{
            background: rgba(26, 31, 39, 232);
            border-color: #303844;
        }}
        QFrame#stageCard[available="false"] QLabel#cardTitle,
        QFrame#stageCard[available="false"] QLabel#cardDetail,
        QFrame#stageCard[available="false"] QLabel#stageStatus,
        QFrame#stageCard[available="false"] QLabel#metricValue,
        QFrame#exportStageCard[available="false"] QLabel#cardTitle,
        QFrame#exportStageCard[available="false"] QLabel#cardDetail,
        QFrame#exportStageCard[available="false"] QLabel#stageStatus,
        QFrame#exportStageCard[available="false"] QLabel#metricValue {{ color: #68717E; }}
        QPushButton#compactButton, QPushButton#dangerCompactButton {{ min-height: 28px; padding: 0 11px; border-radius: 5px; font-size: 12px; }}
        QPushButton[controlRole="segment"]:checked {{
            color: #FFFFFF;
            background: #294D75;
            border-color: {COLORS['accent']};
        }}
        QPushButton#dangerCompactButton {{ color: #F2B6B6; border-color: #734545; background: #482D31; }}
        QPushButton#dangerCompactButton:hover {{ color: #FFFFFF; border-color: #B65B5B; background: #68373B; }}
        QTableWidget {{
            border: 1px solid {COLORS['border']};
            border-radius: 5px;
            background: rgba(26, 31, 39, 245);
            alternate-background-color: #202630;
            selection-background-color: #294D75;
            gridline-color: transparent;
        }}
        QTableWidget::item {{ padding: 5px 8px; border-bottom: 1px solid #2B333E; }}
        QHeaderView::section {{
            min-height: 31px;
            padding: 0 8px;
            color: #D7DEE8;
            background: #252C36;
            border: 0;
            border-right: 1px solid {COLORS['border']};
            border-bottom: 1px solid {COLORS['border']};
            font-weight: 600;
        }}
        """
    )
