# crash_handler.py — 全局异常捕获与崩溃日志

import sys
import threading
import traceback
from datetime import datetime

_log_file: str = "font_editor.log"


def setup_crash_handler(log_file: str) -> None:
    """安装全局异常钩子，未捕获异常写入日志并弹窗提示。

    仅在主线程中调用一次。
    """
    global _log_file
    _log_file = log_file

    original_excepthook = sys.excepthook
    original_threadexcepthook = getattr(threading, "excepthook", None)
    original_unraisablehook = getattr(sys, "unraisablehook", None)

    def _log_exception(exc_type: type, exc_value: BaseException, exc_tb: object) -> None:
        today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
        msg = f"[崩溃 {today}]\n{''.join(tb_lines)}\n"
        try:
            with open(_log_file, "a", encoding="utf-8") as f:
                f.write(msg)
        except OSError:
            pass  # 日志写入失败不做二次处理

    def _custom_excepthook(exc_type, exc_value, exc_tb):
        _log_exception(exc_type, exc_value, exc_tb)
        original_excepthook(exc_type, exc_value, exc_tb)

    def _custom_threadexcepthook(args):
        _log_exception(args.exc_type, args.exc_value, args.exc_traceback)
        if original_threadexcepthook is not None:
            original_threadexcepthook(args)

    def _custom_unraisablehook(args):
        exc_type = args.exc_type or type(args.exc_value)
        _log_exception(exc_type, args.exc_value, args.exc_traceback)
        if original_unraisablehook is not None:
            original_unraisablehook(args)

    sys.excepthook = _custom_excepthook
    if hasattr(threading, "excepthook"):
        threading.excepthook = _custom_threadexcepthook
    if hasattr(sys, "unraisablehook"):
        sys.unraisablehook = _custom_unraisablehook
