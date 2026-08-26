# config_store.py — 全局用户配置读写

import os
from typing import Any

import config
from data.application_database import ApplicationDatabase, resolve_application_database_path
from utils.file_utils import safe_read_json


_DEFAULTS: dict[str, Any] = {
    "默认DPI": 300,
    "默认画布宽": 250,
    "默认画布高": 250,
    "默认目标分": 90,
    "默认图片目录": "",
    "默认导出目录": "",
    "默认排版输出目录": "",
    "性能模式": "自动",
    "最后一次打开的字库": "",
}


def load_global_config() -> dict[str, Any]:
    """从程序数据库加载全局用户设置。"""
    database = ApplicationDatabase(
        resolve_application_database_path(config.GLOBAL_CONFIG_FILE)
    )
    data = database.read_document("用户设置")
    imported = data is None
    if imported:
        data = safe_read_json(config.GLOBAL_CONFIG_FILE, default={})
    if not isinstance(data, dict):
        data = {}
    merged = dict(_DEFAULTS)
    merged.update(data)
    if imported:
        database.write_document("用户设置", merged)
    return merged


def save_global_config(data: dict[str, Any]) -> None:
    """将全局用户设置事务化保存到程序数据库。"""
    ApplicationDatabase(
        resolve_application_database_path(config.GLOBAL_CONFIG_FILE)
    ).write_document("用户设置", data)


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
