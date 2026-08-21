"""程序级结构化数据的 SQLite 初始化入口。"""

from __future__ import annotations

import config
from data.application_database import ApplicationDatabase
from data.config_store import load_global_config
from data.custom_layout_template_store import CustomLayoutTemplateStore
from data.layout_template_store import LayoutTemplateStore
from data.registry_store import load_registry
from utils.file_utils import safe_read_json


def initialize_application_storage() -> None:
    """一次性导入五类旧配置，并确保后续只使用程序数据库。"""
    load_global_config()
    load_registry()
    LayoutTemplateStore(
        config.LAYOUT_TEMPLATE_FILE,
        legacy_file_path=config.LEGACY_LAYOUT_TEMPLATE_FILE,
    )
    CustomLayoutTemplateStore(config.CUSTOM_LAYOUT_TEMPLATE_FILE)

    database = ApplicationDatabase()
    if database.read_document("字库状态索引") is None:
        payload = safe_read_json(config.LIBRARY_SUMMARY_CACHE_FILE, default={})
        if not isinstance(payload, dict):
            payload = {}
        database.write_document("字库状态索引", payload)
