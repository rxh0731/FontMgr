"""字库编辑器 PySide6 应用入口。"""

from __future__ import annotations

import sys

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication

import config
from data.log_manager import LogManager
from ui.main_window import MainWindow
from ui.theme import apply_theme
from utils.crash_handler import setup_crash_handler


def create_application(arguments: list[str] | None = None) -> QApplication:
    """创建并配置全局 QApplication。"""
    QCoreApplication.setOrganizationName("字库编辑器")
    QCoreApplication.setApplicationName("字库编辑器")
    app = QApplication(arguments if arguments is not None else sys.argv)
    app.setApplicationDisplayName("字库编辑器")
    apply_theme(app)
    return app


def main() -> int:
    """启动主窗口并进入 Qt 事件循环。"""
    log_manager = LogManager()
    log_manager.open()
    setup_crash_handler(config.LOG_FILE)
    log_manager.write("正在启动 PySide6 应用")

    app = create_application()
    window = MainWindow()
    window.show()
    exit_code = app.exec()

    log_manager.write(f"应用退出，退出码：{exit_code}")
    log_manager.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
