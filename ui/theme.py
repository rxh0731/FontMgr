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
        """
    )
