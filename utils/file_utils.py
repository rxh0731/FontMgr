# file_utils.py — 文件工具：自然排序、MD5、原子写入

import hashlib
import json
import os
import re
import shutil
import tempfile
from typing import Any

from pypinyin import lazy_pinyin


def is_cjk_character(value: str) -> bool:
    """判断是否为程序支持的单个 CJK 汉字。"""
    if len(value) != 1:
        return False
    codepoint = ord(value)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2EBEF
        or 0x2F800 <= codepoint <= 0x2FA1F
        or 0x30000 <= codepoint <= 0x323AF
    )


def validate_final_char(value: str) -> tuple[bool, str]:
    """校验用户指定的最终字符，返回是否合法及中文错误说明。"""
    if not value:
        return False, "最终字符不能为空。"
    if len(value) != 1:
        return False, "最终字符只能填写一个汉字。"
    if not is_cjk_character(value):
        return False, "最终字符必须是有效的汉字。"
    return True, ""


def natural_key(text: str) -> list:
    """自然排序键：将数字片段转为整数，字母/中文按原序。

    示例：['图10', '图2', '图1'] → ['图1', '图2', '图10']
    """
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', text)]


def pinyin_natural_key(text: str) -> tuple[list, list]:
    """汉字按拼音、数字按数值排序，并以原名称作为同音项的次级排序键。"""
    pinyin_text = "".join(lazy_pinyin(text, errors=lambda chars: list(chars)))
    return natural_key(pinyin_text), natural_key(text)


def compute_file_md5(file_path: str, chunk_size: int = 8192) -> str:
    """计算文件的 MD5 哈希值（hex 摘要）。

    参数：
        file_path: 文件绝对路径
        chunk_size: 每次读取的字节数
    返回：
        32 字符小写 hex 字符串
    """
    digest = hashlib.md5()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(data: Any, file_path: str, indent: int = 2) -> None:
    """原子写入 JSON 文件（先写临时文件，再 os.replace）。

    参数：
        data: 可序列化的 Python 对象
        file_path: 目标文件绝对路径
        indent: JSON 缩进空格数
    """
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    # 临时文件放在目标目录下，保证同盘 os.replace() 是原子的
    fd, tmp_path = tempfile.mkstemp(suffix=".tmp", prefix=".tmp_", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            json.dump(data, fp, ensure_ascii=False, indent=indent)
        # 先写 .bak（如果已有文件），再原子替换
        if os.path.exists(file_path):
            bak_path = file_path + ".bak"
            shutil.copy2(file_path, bak_path)
        os.replace(tmp_path, file_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def safe_read_json(file_path: str, default: Any = None) -> Any:
    """安全读取 JSON 文件，损坏时回退到 .bak。

    参数：
        file_path: JSON 文件绝对路径
        default: 文件不存在时的默认返回值
    返回：
        反序列化后的 Python 对象
    """
    for candidate in (file_path, file_path + ".bak"):
        if not os.path.exists(candidate):
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
    return default


def ensure_dir(dir_path: str) -> None:
    """确保目录存在（递归创建）。"""
    os.makedirs(dir_path, exist_ok=True)
