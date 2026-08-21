from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import config
from services.batch_persistence import (
    JOURNAL_FILENAME,
    JOURNAL_VERSION,
    BatchJournalUncertainError,
    BatchPersistenceSession,
    _compact_json,
    _payload_checksum,
)
from data.library_database import LibraryDatabase
from services.glyph_service import GlyphService


class BatchPersistenceTests(unittest.TestCase):
    @staticmethod
    def _write_legacy_record(
        service: GlyphService,
        variant_id: str,
        *,
        newline: bool = True,
    ) -> Path:
        payload = {
            "版本": JOURNAL_VERSION,
            "库名": service.ziku_name,
            "状态": service.snapshot_variant_state(variant_id),
        }
        record = {"载荷": payload, "校验": _payload_checksum(payload)}
        path = Path(service.ziku_dir) / JOURNAL_FILENAME
        suffix = "\n" if newline else ""
        path.write_text(_compact_json(record) + suffix, encoding="utf-8")
        return path

    def _build_service(self, directory: str) -> tuple[GlyphService, str, str]:
        service = GlyphService("测试库", directory)
        service.ensure_dirs()
        first_id = service.add_original(
            "甲",
            "甲-0001.png",
            "甲.png",
            "md5-first",
        )
        second_id = service.add_original(
            "乙",
            "乙-0001.png",
            "乙.png",
            "md5-second",
        )
        service.save()
        return service, first_id, second_id

    def test_local_snapshot_restores_only_touched_batch_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, first_id, second_id = self._build_service(directory)
            snapshot = service.snapshot_variant_state(first_id)
            second_before = dict(service.get_variant(second_id))

            service.update_variant(
                first_id,
                **{"状态": config.STATUS_REVIEWED, "审核文件": "甲-0001.png"},
            )
            service.set_coordination_summary({}, geometry_completed=True)
            service.restore_variant_state(snapshot)

            self.assertEqual(
                service.get_variant(first_id)["状态"],
                config.STATUS_PENDING_OPTIMIZATION,
            )
            self.assertEqual(service.get_variant(second_id), second_before)
            self.assertFalse(
                service.get_coordination_summary()["几何协调完成"]
            )

    def test_checkpoint_groups_multiple_variant_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, first_id, second_id = self._build_service(directory)
            session = BatchPersistenceSession(
                service,
                checkpoint_items=2,
                checkpoint_seconds=60.0,
            )

            service.update_variant(first_id, **{"状态": config.STATUS_REVIEWED})
            session.record_variant(first_id)
            self.assertFalse(session.checkpoint_if_due())
            on_disk = LibraryDatabase.open(directory).load_data()
            self.assertEqual(
                on_disk["变体详情"][first_id]["状态"],
                config.STATUS_REVIEWED,
            )

            service.update_variant(second_id, **{"状态": config.STATUS_REVIEWED})
            session.record_variant(second_id)
            self.assertTrue(session.checkpoint_if_due())
            session.finish()

            saved = LibraryDatabase.open(directory).load_data()
            self.assertEqual(
                saved["变体详情"][first_id]["状态"],
                config.STATUS_REVIEWED,
            )
            self.assertEqual(
                saved["变体详情"][second_id]["状态"],
                config.STATUS_REVIEWED,
            )
            self.assertFalse((Path(directory) / JOURNAL_FILENAME).exists())

    def test_large_library_default_checkpoint_waits_for_one_hundred_items(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, first_id, _second_id = self._build_service(directory)
            with patch.object(service, "get_total_count", return_value=1000):
                session = BatchPersistenceSession(service)
            try:
                with (
                    patch(
                        "services.batch_persistence.time.monotonic",
                        return_value=session._last_checkpoint,
                    ),
                    patch.object(service, "save", wraps=service.save) as save_mock,
                ):
                    for _index in range(99):
                        session.record_variant(first_id)
                        self.assertFalse(session.checkpoint_if_due())

                    self.assertEqual(save_mock.call_count, 99)
                    session.record_variant(first_id)
                    self.assertTrue(session.checkpoint_if_due())
                    self.assertEqual(save_mock.call_count, 100)
            finally:
                session.finish()

    def test_large_library_slow_items_do_not_save_full_json_per_item(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, first_id, _second_id = self._build_service(directory)
            clock = [0.0]
            with (
                patch.object(service, "get_total_count", return_value=1000),
                patch(
                    "services.batch_persistence.time.monotonic",
                    side_effect=lambda: clock[0],
                ),
                patch.object(service, "save", wraps=service.save) as save_mock,
            ):
                session = BatchPersistenceSession(service)
                try:
                    for _index in range(9):
                        clock[0] += 3.0
                        session.record_variant(first_id)
                        session.checkpoint_if_due()
                    session.finish()
                finally:
                    session.leave_for_recovery()

            self.assertEqual(save_mock.call_count, 9)

    def test_finish_forces_remaining_large_library_state_to_full_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, first_id, _second_id = self._build_service(directory)
            with patch.object(service, "get_total_count", return_value=1000):
                session = BatchPersistenceSession(service)

            service.update_variant(first_id, **{"状态": config.STATUS_REVIEWED})
            session.record_variant(first_id)
            self.assertEqual(
                LibraryDatabase.open(directory).load_data()["变体详情"][first_id]["状态"],
                config.STATUS_REVIEWED,
            )

            session.finish()

            self.assertEqual(
                LibraryDatabase.open(directory).load_data()["变体详情"][first_id]["状态"],
                config.STATUS_REVIEWED,
            )
            self.assertFalse((Path(directory) / JOURNAL_FILENAME).exists())

    def test_same_library_rejects_a_second_batch_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, _first_id, _second_id = self._build_service(directory)
            session = BatchPersistenceSession(service)
            try:
                with self.assertRaisesRegex(RuntimeError, "正在执行其他批处理任务"):
                    BatchPersistenceSession(service)
            finally:
                session.finish()

    def test_released_library_lock_can_be_acquired_again(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, _first_id, _second_id = self._build_service(directory)
            first_session = BatchPersistenceSession(service)
            first_session.finish()

            second_session = BatchPersistenceSession(service)
            second_session.finish()

    def test_first_fsync_failure_truncates_record_before_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, first_id, _second_id = self._build_service(directory)
            session = BatchPersistenceSession(service)
            service.update_variant(first_id, **{"状态": config.STATUS_REVIEWED})

            with patch.object(service, "save", side_effect=OSError("模拟数据库提交失败")):
                with self.assertRaisesRegex(BatchJournalUncertainError, "数据库提交结果无法确认"):
                    session.record_variant(first_id)

            self.assertEqual(session.pending_count, 0)
            journal_path = Path(directory) / JOURNAL_FILENAME
            self.assertFalse(journal_path.exists())
            session.leave_for_recovery()

            recovered = GlyphService("测试库", directory)

            self.assertEqual(
                recovered.get_variant(first_id)["状态"],
                config.STATUS_PENDING_OPTIMIZATION,
            )
            self.assertFalse(journal_path.exists())

    def test_fsync_and_rollback_fsync_failure_reports_uncertain_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, first_id, _second_id = self._build_service(directory)
            session = BatchPersistenceSession(service)
            service.update_variant(first_id, **{"状态": config.STATUS_REVIEWED})

            with patch.object(service, "save", side_effect=OSError("模拟数据库提交失败")):
                with self.assertRaisesRegex(
                    BatchJournalUncertainError,
                    "数据库提交结果无法确认",
                ):
                    session.record_variant(first_id)

            self.assertEqual(session.pending_count, 0)
            session.leave_for_recovery()

    def test_uncheckpointed_record_is_recovered_on_next_open(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, first_id, _second_id = self._build_service(directory)
            session = BatchPersistenceSession(
                service,
                checkpoint_items=20,
                checkpoint_seconds=60.0,
            )
            service.update_variant(
                first_id,
                **{"状态": config.STATUS_REVIEWED, "审核文件": "甲-0001.png"},
            )
            session.record_variant(first_id)
            session.leave_for_recovery()

            recovered = GlyphService("测试库", directory)

            self.assertEqual(
                recovered.get_variant(first_id)["状态"],
                config.STATUS_REVIEWED,
            )
            self.assertEqual(
                recovered.get_variant(first_id)["审核文件"],
                "甲-0001.png",
            )
            self.assertFalse((Path(directory) / JOURNAL_FILENAME).exists())

    def test_recovery_ignores_only_a_partial_final_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, first_id, _second_id = self._build_service(directory)
            session = BatchPersistenceSession(service)
            service.update_variant(first_id, **{"状态": config.STATUS_REVIEWED})
            session.leave_for_recovery()
            journal_path = self._write_legacy_record(service, first_id)
            with open(journal_path, "ab") as handle:
                handle.write(b'{"incomplete"')
                handle.flush()
                os.fsync(handle.fileno())

            recovered = GlyphService("测试库", directory)

            self.assertEqual(
                recovered.get_variant(first_id)["状态"],
                config.STATUS_REVIEWED,
            )
            self.assertFalse(journal_path.exists())

    def test_recovery_ignores_valid_final_record_without_newline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, first_id, _second_id = self._build_service(directory)
            session = BatchPersistenceSession(service)
            service.update_variant(first_id, **{"状态": config.STATUS_REVIEWED})
            session.leave_for_recovery()
            journal_path = self._write_legacy_record(service, first_id, newline=False)

            recovered = GlyphService("测试库", directory)

            self.assertEqual(
                recovered.get_variant(first_id)["状态"],
                config.STATUS_PENDING_OPTIMIZATION,
            )
            self.assertFalse(journal_path.exists())

    def test_recovery_rejects_a_corrupt_final_line_with_newline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, first_id, _second_id = self._build_service(directory)
            session = BatchPersistenceSession(service)
            service.update_variant(first_id, **{"状态": config.STATUS_REVIEWED})
            session.leave_for_recovery()
            journal_path = self._write_legacy_record(service, first_id)
            with open(journal_path, "ab") as handle:
                handle.write(b'{"incomplete"\n')
                handle.flush()
                os.fsync(handle.fileno())

            with self.assertRaisesRegex(RuntimeError, "第 2 行损坏"):
                GlyphService("测试库", directory)

            self.assertTrue(journal_path.exists())


if __name__ == "__main__":
    unittest.main()
