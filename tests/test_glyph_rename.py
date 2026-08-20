from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import config
from services.glyph_service import GlyphService


class GlyphRenameTests(unittest.TestCase):
    def _build_library(self, root: Path) -> tuple[GlyphService, str, str]:
        library = root / "名称修正测试"
        service = GlyphService("名称修正测试", str(library))
        service.ensure_dirs()
        service.init_metadata(300, 250, 250)

        source_id = service.add_original(
            "錯",
            "錯-0001.tif",
            "原始错字.tif",
            "1" * 32,
        )
        target_id = service.add_original(
            "正",
            "正-0001.png",
            "已有正字.png",
            "2" * 32,
        )
        directories = service.get_workflow_dirs()
        (Path(directories["原图"]) / "正-0001.png").write_bytes(b"target")

        detail = service.get_variant(source_id)
        stage_fields = (
            ("原图", "原始文件", "tif"),
            ("灰度母版", "灰度母版文件", "png"),
            ("清洁掩码", "清洁掩码文件", "png"),
            ("优化预览", "中间文件", "png"),
            ("手工审核", "审核文件", "png"),
            ("成品", "成品文件", "png"),
        )
        for index, (stage, field, suffix) in enumerate(stage_fields, start=1):
            filename = f"錯-0001.{suffix}"
            detail[field] = filename
            (Path(directories[stage]) / filename).write_bytes(
                f"stage-{index}".encode("ascii")
            )
        detail["状态"] = config.STATUS_FINISHED
        detail["自动优化"]["得分"] = 97.5
        detail["备注"] = "保持状态"
        service.save()
        return service, source_id, target_id

    def test_move_variant_renames_all_existing_stage_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service, source_id, target_id = self._build_library(Path(temporary))
            before = dict(service.get_variant(source_id))

            preview = service.preview_variant_char_change(source_id, "正")
            self.assertEqual(preview["新文件名"], "正-0002.tif")
            self.assertEqual(len(preview["文件变更"]), 6)

            result = service.move_variant_to_char(source_id, "正")
            self.assertEqual(result["新变体序号"], 2)
            detail = service.get_variant(source_id)
            self.assertEqual(detail["归属字"], "正")
            self.assertEqual(detail["变体序号"], 2)
            self.assertEqual(detail["状态"], config.STATUS_FINISHED)
            self.assertEqual(detail["自动优化"]["得分"], 97.5)
            self.assertEqual(detail["备注"], "保持状态")

            groups = service.get_glyph_groups()
            self.assertNotIn("錯", groups)
            self.assertEqual(groups["正"], [target_id, source_id])
            for change in result["文件变更"]:
                self.assertFalse(os.path.exists(change["原路径"]))
                self.assertTrue(os.path.isfile(change["新路径"]))
            self.assertEqual(before["原始MD5"], detail["原始MD5"])

            reopened = GlyphService.open(service.ziku_name, service.ziku_dir)
            self.assertEqual(reopened.get_variant(source_id)["归属字"], "正")
            self.assertEqual(reopened.get_glyph_groups()["正"], [target_id, source_id])

    def test_missing_referenced_stage_file_blocks_without_side_effects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service, source_id, _target_id = self._build_library(Path(temporary))
            detail = service.get_variant(source_id)
            missing_path = Path(service.get_workflow_dirs()["优化预览"]) / str(
                detail["中间文件"]
            )
            missing_path.unlink()
            original_path = Path(service.get_workflow_dirs()["原图"]) / "錯-0001.tif"

            with self.assertRaisesRegex(FileNotFoundError, "重新核对字库数据"):
                service.move_variant_to_char(source_id, "正")

            self.assertEqual(service.get_variant(source_id)["归属字"], "錯")
            self.assertIn(source_id, service.get_glyph_groups()["錯"])
            self.assertTrue(original_path.is_file())
            self.assertFalse(
                (Path(service.get_workflow_dirs()["原图"]) / "正-0002.tif").exists()
            )

    def test_json_save_failure_rolls_back_files_and_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service, source_id, target_id = self._build_library(Path(temporary))
            directories = service.get_workflow_dirs()

            with patch.object(service, "save", side_effect=OSError("模拟写盘失败")):
                with self.assertRaisesRegex(OSError, "模拟写盘失败"):
                    service.move_variant_to_char(source_id, "正")

            detail = service.get_variant(source_id)
            self.assertEqual(detail["归属字"], "錯")
            self.assertEqual(service.get_glyph_groups()["錯"], [source_id])
            self.assertEqual(service.get_glyph_groups()["正"], [target_id])
            for stage, suffix in (
                ("原图", "tif"),
                ("灰度母版", "png"),
                ("清洁掩码", "png"),
                ("优化预览", "png"),
                ("手工审核", "png"),
                ("成品", "png"),
            ):
                self.assertTrue(
                    (Path(directories[stage]) / f"錯-0001.{suffix}").is_file()
                )
                self.assertFalse(
                    (Path(directories[stage]) / f"正-0002.{suffix}").exists()
                )
            transaction_dir = Path(service.ziku_dir) / ".fonteditor_file_transactions"
            self.assertFalse(transaction_dir.exists())

    def test_process_exit_before_json_save_recovers_renamed_group_on_open(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service, source_id, target_id = self._build_library(Path(temporary))

            with patch.object(service, "save", side_effect=SystemExit("模拟进程退出")):
                with self.assertRaises(SystemExit):
                    service.move_variant_to_char(source_id, "正")

            transaction_dir = Path(service.ziku_dir) / ".fonteditor_file_transactions"
            self.assertTrue(transaction_dir.is_dir())
            recovered = GlyphService.open(service.ziku_name, service.ziku_dir)
            self.assertEqual(recovered.get_variant(source_id)["归属字"], "正")
            self.assertNotIn("錯", recovered.get_glyph_groups())
            self.assertEqual(
                recovered.get_glyph_groups()["正"],
                [target_id, source_id],
            )
            self.assertFalse(transaction_dir.exists())
            original_dir = Path(recovered.get_workflow_dirs()["原图"])
            self.assertFalse((original_dir / "錯-0001.tif").exists())
            self.assertTrue((original_dir / "正-0002.tif").is_file())


if __name__ == "__main__":
    unittest.main()
