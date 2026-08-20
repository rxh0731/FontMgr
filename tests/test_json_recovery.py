from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import config
from services.batch_persistence import BatchPersistenceSession, JOURNAL_FILENAME
from services.glyph_service import GlyphService
from utils.file_utils import atomic_write_json, safe_read_json


class JsonRecoveryTests(unittest.TestCase):
    def _build_service(self, directory: str) -> tuple[GlyphService, str, str]:
        service = GlyphService("恢复测试", directory)
        service.ensure_dirs()
        first_id = service.add_original("甲", "甲-0001.png", "甲.png", "md5-a")
        second_id = service.add_original("乙", "乙-0001.png", "乙.png", "md5-b")
        service.save()
        service.save()
        return service, first_id, second_id

    def _leave_reviewed_journal(
        self,
        service: GlyphService,
        variant_id: str,
    ) -> Path:
        session = BatchPersistenceSession(
            service,
            checkpoint_items=100,
            checkpoint_seconds=3600.0,
        )
        service.update_variant(variant_id, **{"状态": config.STATUS_REVIEWED})
        session.record_variant(variant_id)
        session.leave_for_recovery()
        return Path(service.ziku_dir) / JOURNAL_FILENAME

    def test_corrupt_main_uses_backup_then_replays_journal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, first_id, second_id = self._build_service(directory)
            json_path = Path(directory) / "恢复测试.json"
            backup_path = Path(str(json_path) + ".bak")
            second_before = dict(service.get_variant(second_id))
            journal_path = self._leave_reviewed_journal(service, first_id)
            json_path.write_bytes(b"{corrupt-main")

            recovered = GlyphService("恢复测试", directory)

            self.assertEqual(recovered.get_variant(first_id)["状态"], config.STATUS_REVIEWED)
            self.assertEqual(recovered.get_variant(second_id), second_before)
            self.assertIsInstance(json.loads(json_path.read_text(encoding="utf-8")), dict)
            self.assertIsInstance(json.loads(backup_path.read_text(encoding="utf-8")), dict)
            self.assertFalse(journal_path.exists())

    def test_missing_main_uses_backup_then_replays_journal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, first_id, second_id = self._build_service(directory)
            json_path = Path(directory) / "恢复测试.json"
            journal_path = self._leave_reviewed_journal(service, first_id)
            json_path.unlink()

            recovered = GlyphService("恢复测试", directory)

            self.assertTrue(json_path.is_file())
            self.assertIsNotNone(recovered.get_variant(second_id))
            self.assertEqual(recovered.get_variant(first_id)["状态"], config.STATUS_REVIEWED)
            self.assertFalse(journal_path.exists())

    def test_journal_without_readable_base_refuses_empty_library(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal_path = Path(directory) / JOURNAL_FILENAME
            journal_path.write_text("未完成恢复记录", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "避免生成空字库"):
                GlyphService("恢复测试", directory)

            self.assertTrue(journal_path.exists())
            self.assertFalse((Path(directory) / "恢复测试.json").exists())

    def test_corrupt_main_and_backup_raise_instead_of_initializing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "恢复测试.json"
            backup_path = Path(str(json_path) + ".bak")
            json_path.write_bytes(b"{bad-main")
            backup_path.write_bytes(b"{bad-backup")

            with self.assertRaisesRegex(RuntimeError, "均缺失或损坏"):
                GlyphService("恢复测试", directory)

            self.assertEqual(json_path.read_bytes(), b"{bad-main")
            self.assertEqual(backup_path.read_bytes(), b"{bad-backup")

    def test_backup_restore_failure_never_overwrites_good_backup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "恢复测试.json"
            backup_path = Path(str(json_path) + ".bak")
            json_path.write_bytes(b"{bad-main")
            good_data = {"数据版本": 3, "内容": "唯一备份"}
            backup_path.write_text(
                json.dumps(good_data, ensure_ascii=False),
                encoding="utf-8",
            )
            backup_before = backup_path.read_bytes()

            with (
                patch("utils.file_utils.os.replace", side_effect=OSError("模拟替换失败")),
                self.assertRaisesRegex(OSError, "模拟替换失败"),
            ):
                GlyphService("恢复测试", directory)

            self.assertEqual(json_path.read_bytes(), b"{bad-main")
            self.assertEqual(backup_path.read_bytes(), backup_before)

    def test_atomic_write_can_repair_main_without_touching_backup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "data.json"
            backup_path = Path(str(json_path) + ".bak")
            json_path.write_bytes(b"bad-main")
            backup_path.write_bytes(b"good-backup")

            atomic_write_json(
                {"状态": "已恢复"},
                str(json_path),
                backup_existing=False,
            )

            self.assertEqual(safe_read_json(str(json_path))["状态"], "已恢复")
            self.assertEqual(backup_path.read_bytes(), b"good-backup")


if __name__ == "__main__":
    unittest.main()
