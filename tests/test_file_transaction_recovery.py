"""单字图片事务的进程异常恢复测试。"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services.batch_persistence import acquire_batch_library_lock
from services.file_transaction_recovery import (
    FileChange,
    FileTransaction,
    FileTransactionCommitUncertainError,
    PHASE_CLEANUP,
    PHASE_ROLLFORWARD,
    TRANSACTION_DIRNAME,
    ensure_file_transactions_ready,
    recover_file_transactions,
    recovery_full_state_snapshot,
)
from utils.file_utils import atomic_write_json


class FileTransactionRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.stage = self.root / "04_自动优化稿"
        self.stage.mkdir()
        self.json_path = self.root / "临时字库.json"
        self.old_state = {
            "变体ID": "variant-1",
            "变体存在": True,
            "变体详情": {"状态": "待优化", "中间文件": ""},
            "元数据": {"最后修改": "旧时间"},
            "整体协调": {"几何协调完成": True},
        }
        self.new_state = {
            "变体ID": "variant-1",
            "变体存在": True,
            "变体详情": {"状态": "待审核", "中间文件": "字.png"},
            "元数据": {"最后修改": "新时间"},
            "整体协调": {"几何协调完成": False},
        }
        self.data = {
            "数据版本": 3,
            "变体详情": {"variant-1": copy.deepcopy(self.old_state["变体详情"])},
            "元数据": copy.deepcopy(self.old_state["元数据"]),
            "整体协调": copy.deepcopy(self.old_state["整体协调"]),
        }
        atomic_write_json(self.data, str(self.json_path))

    def tearDown(self) -> None:
        self._temporary.cleanup()

    @staticmethod
    def _md5(content: bytes) -> str:
        return hashlib.md5(content).hexdigest()

    def _change(
        self,
        name: str,
        old_content: bytes,
        new_content: bytes,
    ) -> tuple[FileChange, Path, Path]:
        target = self.stage / name
        temporary = self.stage / f".new-{name}"
        target.write_bytes(old_content)
        temporary.write_bytes(new_content)
        return (
            FileChange(
                target_path=str(target),
                temporary_path=str(temporary),
                new_md5=self._md5(new_content),
            ),
            target,
            temporary,
        )

    def _recover(self) -> bool:
        return recover_file_transactions(
            self.data,
            ziku_dir=str(self.root),
            json_path=str(self.json_path),
        )

    def _assert_state(self, expected: dict[str, object]) -> None:
        self.assertEqual(self.data["变体详情"]["variant-1"], expected["变体详情"])
        self.assertEqual(self.data["元数据"], expected["元数据"])
        self.assertEqual(self.data["整体协调"], expected["整体协调"])
        with self.json_path.open("r", encoding="utf-8") as handle:
            saved = json.load(handle)
        self.assertEqual(saved["变体详情"]["variant-1"], expected["变体详情"])

    def test_exit_after_backups_rolls_back_old_files_and_state(self) -> None:
        change, target, temporary = self._change("字.png", b"old", b"new")
        transaction = FileTransaction.begin(
            str(self.root),
            [change],
            self.old_state,
        )
        transaction.backup_targets()

        self.assertFalse(target.exists())
        self.assertTrue(temporary.exists())
        self.assertTrue(self._recover())

        self.assertEqual(target.read_bytes(), b"old")
        self.assertFalse(temporary.exists())
        self._assert_state(self.old_state)
        self.assertFalse((self.root / TRANSACTION_DIRNAME).exists())

    def test_exit_after_partial_replacement_finishes_verified_new_files(self) -> None:
        first, first_target, first_temporary = self._change(
            "甲.png", b"old-a", b"new-a"
        )
        second, second_target, _second_temporary = self._change(
            "乙.png", b"old-b", b"new-b"
        )
        transaction = FileTransaction.begin(
            str(self.root),
            [first, second],
            self.old_state,
        )
        transaction.backup_targets()
        transaction.mark_rollforward(self.new_state)
        os.replace(first_temporary, first_target)

        self.assertTrue(self._recover())

        self.assertEqual(first_target.read_bytes(), b"new-a")
        self.assertEqual(second_target.read_bytes(), b"new-b")
        self._assert_state(self.new_state)

    def test_all_images_installed_before_json_commit_restores_new_state(self) -> None:
        change, target, _temporary = self._change("字.png", b"old", b"new")
        transaction = FileTransaction.begin(
            str(self.root),
            [change],
            self.old_state,
        )
        transaction.backup_targets()
        transaction.mark_rollforward(self.new_state)
        transaction.install_new_files()

        self._assert_state(self.old_state)
        self.assertTrue(self._recover())

        self.assertEqual(target.read_bytes(), b"new")
        self._assert_state(self.new_state)

    def test_full_library_state_recovers_multi_variant_batch(self) -> None:
        change, target, _temporary = self._change("字.png", b"old", b"new")
        old_full_state = copy.deepcopy(self.data)
        new_full_state = copy.deepcopy(self.data)
        new_full_state["变体详情"]["variant-1"] = copy.deepcopy(
            self.new_state["变体详情"]
        )
        new_full_state["变体详情"]["variant-2"] = {
            "状态": "成品已生成",
            "成品文件": "乙.png",
        }
        new_full_state["整体协调"] = {"几何协调完成": True}
        transaction = FileTransaction.begin(
            str(self.root),
            [change],
            recovery_full_state_snapshot(old_full_state),
        )
        transaction.backup_targets()
        transaction.mark_rollforward(
            recovery_full_state_snapshot(new_full_state)
        )
        transaction.install_new_files()

        self.assertTrue(self._recover())

        self.assertEqual(target.read_bytes(), b"new")
        self.assertEqual(self.data, new_full_state)
        with self.json_path.open("r", encoding="utf-8") as handle:
            self.assertEqual(json.load(handle), new_full_state)

    def test_mismatched_new_file_md5_rolls_back_old_file_and_state(self) -> None:
        change, target, _temporary = self._change("字.png", b"old", b"new")
        transaction = FileTransaction.begin(
            str(self.root),
            [change],
            self.old_state,
        )
        transaction.backup_targets()
        transaction.mark_rollforward(self.new_state)
        transaction.install_new_files()
        target.write_bytes(b"corrupted")

        self.assertTrue(self._recover())

        self.assertEqual(target.read_bytes(), b"old")
        self._assert_state(self.old_state)

    def test_active_batch_lock_prevents_another_session_from_recovering(self) -> None:
        change, target, _temporary = self._change("字.png", b"old", b"new")
        transaction = FileTransaction.begin(
            str(self.root),
            [change],
            self.old_state,
        )
        transaction.backup_targets()
        batch_lock = acquire_batch_library_lock(str(self.root))
        try:
            with self.assertRaisesRegex(RuntimeError, "正在执行其他批处理任务"):
                self._recover()
            self.assertFalse(target.exists())
            self.assertTrue(Path(transaction.manifest_path).exists())
        finally:
            batch_lock.release()

        self.assertTrue(self._recover())
        self.assertEqual(target.read_bytes(), b"old")

    def test_cleanup_failure_then_newer_save_only_cleans_old_transaction(self) -> None:
        change, target, _temporary = self._change("字.png", b"old", b"new-1")

        def fail_old_backup_cleanup(path: str) -> None:
            if Path(path).name.startswith(".fonteditor_rollback_"):
                raise OSError("模拟旧备份清理失败")
            os.remove(path)

        first = FileTransaction.begin(
            str(self.root),
            [change],
            self.old_state,
            remove_func=fail_old_backup_cleanup,
        )
        first.backup_targets()
        first.mark_rollforward(self.new_state)
        first.install_new_files()
        self.data["变体详情"]["variant-1"] = copy.deepcopy(
            self.new_state["变体详情"]
        )
        self.data["元数据"] = copy.deepcopy(self.new_state["元数据"])
        self.data["整体协调"] = copy.deepcopy(self.new_state["整体协调"])
        atomic_write_json(self.data, str(self.json_path))

        cleanup_errors = first.finalize()

        self.assertTrue(cleanup_errors)
        with open(first.manifest_path, "r", encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["载荷"]["阶段"], PHASE_CLEANUP)

        newest_state = copy.deepcopy(self.new_state)
        newest_state["变体详情"]["中间文件"] = "字-新版.png"
        newest_state["元数据"]["最后修改"] = "更新后的时间"
        second_temporary = self.stage / ".newer-字.png"
        second_temporary.write_bytes(b"new-2")
        second = FileTransaction.begin(
            str(self.root),
            [
                FileChange(
                    target_path=str(target),
                    temporary_path=str(second_temporary),
                    new_md5=self._md5(b"new-2"),
                )
            ],
            self.new_state,
        )
        second.backup_targets()
        second.mark_rollforward(newest_state)
        second.install_new_files()
        self.data["变体详情"]["variant-1"] = copy.deepcopy(
            newest_state["变体详情"]
        )
        self.data["元数据"] = copy.deepcopy(newest_state["元数据"])
        self.data["整体协调"] = copy.deepcopy(newest_state["整体协调"])
        atomic_write_json(self.data, str(self.json_path))
        self.assertEqual(second.finalize(), [])

        self.assertTrue(self._recover())

        self.assertEqual(target.read_bytes(), b"new-2")
        self._assert_state(newest_state)
        self.assertFalse((self.root / TRANSACTION_DIRNAME).exists())

    def test_missing_backup_never_treats_new_target_as_old_file(self) -> None:
        change, target, _temporary = self._change("字.png", b"old", b"new")
        transaction = FileTransaction.begin(
            str(self.root),
            [change],
            self.old_state,
        )
        transaction.backup_targets()
        transaction.mark_rollforward(self.new_state)
        transaction.install_new_files()
        backup_path = next(self.stage.glob(".fonteditor_rollback_*"))
        backup_path.unlink()

        rollback_errors = transaction.rollback()

        self.assertTrue(rollback_errors)
        self.assertIn("目标内容不是事务开始时的旧文件", "；".join(rollback_errors))
        with self.assertRaisesRegex(RuntimeError, "目标内容不是事务开始时的旧文件"):
            self._recover()
        self.assertEqual(target.read_bytes(), b"new")
        self._assert_state(self.old_state)
        self.assertTrue(Path(transaction.manifest_path).exists())

    def test_cleanup_phase_write_failure_blocks_new_writes_until_recovery(self) -> None:
        change, _target, _temporary = self._change("字.png", b"old", b"new")
        transaction = FileTransaction.begin(
            str(self.root),
            [change],
            self.old_state,
        )
        transaction.backup_targets()
        transaction.mark_rollforward(self.new_state)
        transaction.install_new_files()

        with patch(
            "services.file_transaction_recovery._write_manifest",
            side_effect=OSError("模拟清理阶段标记失败"),
        ):
            with self.assertRaisesRegex(
                FileTransactionCommitUncertainError,
                "重新打开字库",
            ):
                transaction.finalize()

        self.assertEqual(transaction.phase, PHASE_ROLLFORWARD)
        with open(transaction.manifest_path, "r", encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["载荷"]["阶段"], PHASE_ROLLFORWARD)
        with self.assertRaisesRegex(RuntimeError, "尚未恢复"):
            ensure_file_transactions_ready(str(self.root))


if __name__ == "__main__":
    unittest.main()
