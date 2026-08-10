# main.py — 字库编辑 V2.0 程序入口

import ctypes
import os
import sys
import time
import warnings

# 静默 PIL 的 iCCP 警告
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

import config
from data.log_manager import LogManager
from utils.crash_handler import setup_crash_handler


def _enable_high_dpi() -> None:
    """避免 Windows 对界面进行位图拉伸，保持文字和控件清晰。"""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass


def main() -> None:
    """程序主入口。"""

    _enable_high_dpi()

    # 确保关键目录存在
    os.makedirs(config.CONFIG_DIR, exist_ok=True)
    os.makedirs(config.ZIKU_ROOT, exist_ok=True)

    # 日志系统
    log_mgr = LogManager()
    log_mgr.open()
    setup_crash_handler(config.LOG_FILE)
    started_at = time.perf_counter()
    log_mgr.write("程序启动｜性能诊断日志已启用")

    # 启动主窗口
    try:
        from ui.app_window import AppWindow
        app = AppWindow()
        app.run()
    except Exception as e:
        log_mgr.write(f"致命错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        log_mgr.write(f"程序退出｜运行耗时={time.perf_counter() - started_at:.3f}秒")
        log_mgr.close()


if __name__ == "__main__":
    main()
