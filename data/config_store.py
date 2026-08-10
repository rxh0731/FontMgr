# config_store.py — 全局用户配置读写

import os
from typing import Any

import config
from utils.file_utils import atomic_write_json, safe_read_json


_DEFAULTS: dict[str, Any] = {
    "默认DPI": 300,
    "默认画布宽": 250,
    "默认画布高": 250,
    "默认目标分": 90,
    "最后一次打开的字库": "",
}


def load_global_config() -> dict[str, Any]:
    """加载全局用户设置（不存在或损坏时返回默认值）。"""
    data = safe_read_json(config.GLOBAL_CONFIG_FILE, default={})
    if not isinstance(data, dict):
        data = {}
    merged = dict(_DEFAULTS)
    merged.update(data)
    return merged


def save_global_config(data: dict[str, Any]) -> None:
    """保存全局用户设置（原子写入）。"""
    atomic_write_json(data, config.GLOBAL_CONFIG_FILE)


def get_last_ziku_path() -> str:
    """获取最后一次打开的字库路径（如果字库已不存在，返回空字符串）。"""
    cfg = load_global_config()
    path = cfg.get("最后一次打开的字库", "")
    if path and not os.path.isdir(path):
        return ""
    return path


def set_last_ziku_path(ziku_dir: str) -> None:
    """记录最后一次打开的字库路径。"""
    cfg = load_global_config()
    cfg["最后一次打开的字库"] = ziku_dir
    save_global_config(cfg)
