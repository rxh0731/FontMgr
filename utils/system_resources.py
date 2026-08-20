"""系统资源探测辅助函数。"""

from __future__ import annotations

import ctypes
import os


MIB = 1024 * 1024


def get_system_memory_status() -> tuple[int, int]:
    """返回物理内存总量和当前可用量；探测失败时使用保守值。"""

    if os.name == "nt":
        try:
            class MemoryStatus(ctypes.Structure):
                _fields_ = (
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                )

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.total_physical), int(status.available_physical)
        except (AttributeError, OSError, TypeError, ValueError):
            pass
    else:
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            total_pages = int(os.sysconf("SC_PHYS_PAGES"))
            available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            return page_size * total_pages, page_size * available_pages
        except (AttributeError, OSError, TypeError, ValueError):
            pass
    return 8 * 1024 * MIB, 2 * 1024 * MIB
