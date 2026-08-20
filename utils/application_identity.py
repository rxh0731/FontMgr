"""应用身份与窗口图标的跨运行环境配置。"""

from __future__ import annotations

import ctypes
import os
import sys

from PySide6.QtGui import QIcon

import config


def configure_windows_app_identity() -> bool:
    """在创建 QApplication 前设置稳定的 Windows 任务栏应用身份。"""

    if sys.platform != "win32":
        return False
    try:
        result = ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            config.WINDOWS_APP_USER_MODEL_ID
        )
    except (AttributeError, OSError):
        return False
    return result == 0


def load_application_icon() -> QIcon:
    """优先载入 Qt 稳定支持的 PNG，缺失时回退到 EXE 使用的 ICO。"""

    for path in (config.WINDOW_ICON_FILE, config.ICON_FILE):
        if not os.path.isfile(path):
            continue
        icon = QIcon(path)
        if not icon.isNull():
            return icon
    return QIcon()
