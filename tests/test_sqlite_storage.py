"""程序级与字库级 SQLite 存储回归测试。"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import config
from data.application_database import ApplicationDatabase
from data.library_database import LIBRARY_DATABASE_FILENAME, LibraryDatabase
from data.registry_store import get_builtin_registry
from data.storage_initializer import initialize_application_storage
from services.batch_persistence import BatchPersistenceSession, JOURNAL_FILENAME
from services.glyph_service import GlyphService


def _legacy_library(name: str) -> dict[str, object]:
    return {
        "数据版本": 3,
        "库名": name,
        "元数据": {
            "DPI": 300,
            "画布宽": 250,
            "画布高": 250,
            "成品宽度毫米": 21.17,
            "成品高度毫米": 21.17,
            "创建时间": "2026-01-01T00:00:00",
            "最后修改": "2026-01-01T00:00:00",
        },
        "会话": {},
        "字形组索引": {"永": ["variant-1"]},
        "变体详情": {
            "variant-1": {
                "变体ID": "variant-1",
                "归属字": "永",
                "状态": config.STATUS_PENDING_OPTIMIZATION,
                "原始文件": "永-0001.png",
                "原始MD5": "a" * 32,
                "导入前文件名": "来源.png",
                "图像信息": {"宽": 250, "高": 250},
                "自动优化": {},
                "手工编辑": {},
                "变换参数": {},
                "整体协调参数": {},
                "备注": "",
            }
        },
        "整体协调": GlyphService._default_coordination_summary(),
    }


class ApplicationDatabaseTests(unittest.TestCase):
    def test_documents_are_transactionally_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "fontmgr.sqlite3")
            database = ApplicationDatabase(path)
            database.write_document("用户设置", {"默认DPI": 300})
            database.write_document("用户设置", {"默认DPI": 600})

            self.assertEqual(database.read_document("用户设置"), {"默认DPI": 600})
            with closing(sqlite3.connect(path)) as connection:
                count = connection.execute(
                    "SELECT COUNT(*) FROM application_documents"
                ).fetchone()[0]
            self.assertEqual(count, 1)

    def test_startup_imports_all_five_configuration_files_into_one_database(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_dir = Path(directory) / "配置"
            config_dir.mkdir()
            paths = {
                "GLOBAL_CONFIG_FILE": config_dir / "用户设置.json",
                "REGISTRY_FILE": config_dir / "算法注册表.json",
                "LAYOUT_TEMPLATE_FILE": config_dir / "通用经文排版模板.json",
                "LEGACY_LAYOUT_TEMPLATE_FILE": config_dir / "排版模板.json",
                "CUSTOM_LAYOUT_TEMPLATE_FILE": config_dir / "定制经文排版模板.json",
                "LIBRARY_SUMMARY_CACHE_FILE": config_dir / "字库状态索引.json",
            }
            payloads = {
                "GLOBAL_CONFIG_FILE": {"默认DPI": 600},
                "REGISTRY_FILE": get_builtin_registry(),
                "LAYOUT_TEMPLATE_FILE": {"数据版本": 2, "用户模板": {}},
                "CUSTOM_LAYOUT_TEMPLATE_FILE": {"数据版本": 1, "用户模板": {}},
                "LIBRARY_SUMMARY_CACHE_FILE": {"版本": 1, "字库摘要": []},
            }
            for key, payload in payloads.items():
                paths[key].write_text(
                    json.dumps(payload, ensure_ascii=False),
                    encoding="utf-8",
                )
            database_path = config_dir / "fontmgr.sqlite3"
            patches = [
                patch.object(config, "CONFIG_DIR", str(config_dir)),
                patch.object(config, "APP_DATABASE_FILE", str(database_path)),
                *(patch.object(config, key, str(path)) for key, path in paths.items()),
            ]
            for active_patch in patches:
                active_patch.start()
            try:
                initialize_application_storage()
            finally:
                for active_patch in reversed(patches):
                    active_patch.stop()

            database = ApplicationDatabase(str(database_path))
            for key in (
                "用户设置",
                "算法注册表",
                "通用经文排版模板",
                "定制经文排版模板",
                "字库状态索引",
            ):
                self.assertIsNotNone(database.read_document(key), key)
            self.assertEqual(
                json.loads(paths["GLOBAL_CONFIG_FILE"].read_text(encoding="utf-8")),
                payloads["GLOBAL_CONFIG_FILE"],
            )


class LibraryDatabaseTests(unittest.TestCase):
    def test_new_library_uses_database_without_main_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = GlyphService.open("新字库", directory)
            variant_id = service.add_original(
                "永",
                "永-0001.png",
                "来源.png",
                "b" * 32,
            )
            service.save()

            self.assertTrue((Path(directory) / LIBRARY_DATABASE_FILENAME).is_file())
            self.assertFalse((Path(directory) / "新字库.json").exists())
            reopened = GlyphService.open("新字库", directory)
            self.assertEqual(reopened.get_variant(variant_id)["归属字"], "永")

    def test_legacy_json_is_imported_once_and_left_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "旧字库.json"
            original = _legacy_library("旧字库")
            source.write_text(json.dumps(original, ensure_ascii=False), encoding="utf-8")

            first = GlyphService.open("旧字库", directory)
            self.assertEqual(first.get_total_count(), 1)
            source_before = source.read_bytes()
            changed = _legacy_library("旧字库")
            changed["字形组索引"] = {}
            changed["变体详情"] = {}
            source.write_text(json.dumps(changed, ensure_ascii=False), encoding="utf-8")

            second = GlyphService.open("旧字库", directory)
            self.assertEqual(second.get_total_count(), 1)
            self.assertNotEqual(source.read_bytes(), source_before)

    def test_failed_import_does_not_install_database(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = _legacy_library("失败字库")
            with patch.object(
                LibraryDatabase,
                "replace_all",
                side_effect=OSError("模拟导入失败"),
            ):
                with self.assertRaisesRegex(OSError, "模拟导入失败"):
                    LibraryDatabase.install_from_data(directory, data)

            self.assertFalse((Path(directory) / LIBRARY_DATABASE_FILENAME).exists())
            self.assertFalse(
                any(path.name.endswith(".tmp") for path in Path(directory).iterdir())
            )

    def test_batch_session_commits_without_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = GlyphService.open("批处理字库", directory)
            variant_id = service.add_original(
                "永",
                "永-0001.png",
                "来源.png",
                "c" * 32,
            )
            session = BatchPersistenceSession(service)
            session.record_variant(variant_id)
            session.finish()

            self.assertFalse((Path(directory) / JOURNAL_FILENAME).exists())
            self.assertEqual(
                GlyphService.open("批处理字库", directory).get_total_count(),
                1,
            )


if __name__ == "__main__":
    unittest.main()
