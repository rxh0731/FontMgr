"""最终成品三模式导出、就绪审计与事务安全回归测试。"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import config
from services.export_service import (
    ExportConflictDecision,
    ExportOptions,
    ExportService,
)
from services.glyph_service import GlyphService
from utils.file_utils import compute_file_md5


class ExportServiceTests(unittest.TestCase):
    """验证导出只读取最终成品，并且失败时不留下部分结果。"""

    @staticmethod
    def _source_image(size: tuple[int, int] = (20, 10)) -> Image.Image:
        image = Image.new("RGBA", size, (120, 20, 40, 0))
        for y in range(2, 8):
            for x in range(5, 15):
                alpha = 96 if x in (5, 14) else 220
                image.putpixel((x, y), (0, 0, 0, alpha))
        return image

    def _build_library(
        self,
        root: Path,
        *,
        variants: int = 1,
        complete_summary: bool = True,
    ) -> tuple[GlyphService, list[dict[str, object]]]:
        library_dir = root / "测试导出库"
        glyph = GlyphService("测试导出库", str(library_dir))
        glyph.ensure_dirs()
        glyph.init_metadata(dpi=300, canvas_w=40, canvas_h=30)
        finished_dir = Path(glyph.get_workflow_dirs()["成品"])
        details: list[dict[str, object]] = []
        for index in range(variants):
            original_name = f"甲-{index + 1:04d}.tif"
            detail_id = glyph.add_original(
                "甲",
                original_name,
                original_name,
                f"{index + 1:032x}",
            )
            detail = glyph.get_variant(detail_id)
            finished_name = f"甲-{index + 1:04d}.png"
            finished_path = finished_dir / finished_name
            image = self._source_image((20 + index * 2, 10 + index))
            image.save(finished_path, "PNG", dpi=(300, 300))
            glyph.mark_finished(
                detail_id,
                finished_name,
                compute_file_md5(str(finished_path)),
                {
                    "标准画布": [40, 30],
                    "实际画布": list(image.size),
                    "墨色协调": {
                        "启用": True,
                        "方法": "视觉墨量",
                        "方法版本": 1,
                        "基准": 180.0,
                        "保存后墨色": 180.0,
                        "保存后复测": True,
                        "是否达标": True,
                        "人工接受例外": False,
                    },
                },
            )
            details.append(detail)
        if complete_summary:
            glyph.set_coordination_summary(
                {},
                180.0,
                geometry_completed=True,
                ink_completed=True,
                ink_enabled=True,
                ink_method="视觉墨量",
                ink_method_version=1,
                ink_counts={
                    "总数": variants,
                    "已达标": variants,
                    "待确认": 0,
                    "人工例外": 0,
                },
            )
        return glyph, details

    def test_audit_rejects_legacy_ink_completion_without_versioned_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph, _details = self._build_library(
                root,
                complete_summary=False,
            )
            legacy_state = glyph.snapshot_state()
            legacy_state["整体协调"] = {
                "基准": {},
                "墨色基准": 180.0,
                "墨色统一启用": True,
                "几何协调完成": True,
                "墨色统一完成": True,
                "最后生成时间": "2026-08-15 12:00:00",
            }
            glyph.restore_state(legacy_state)
            glyph.save()
            reloaded = GlyphService("测试导出库", str(root / "测试导出库"))

            audit = ExportService(reloaded).audit_readiness()

            self.assertFalse(audit["就绪"])
            self.assertFalse(audit["墨色统一完成"])
            self.assertIsNone(audit["墨色方法版本"])
            self.assertTrue(any("缺少新方法及版本" in reason for reason in audit["原因"]))

    def test_audit_rejects_stale_per_glyph_ink_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph, details = self._build_library(Path(directory))
            ink_record = details[0]["整体协调参数"]["墨色协调"]
            ink_record["方法版本"] = 2

            audit = ExportService(glyph).audit_readiness()

            self.assertFalse(audit["就绪"])
            self.assertEqual(audit["已就绪"], 0)
            self.assertEqual(audit["待协调"], 1)
            self.assertEqual(audit["成品缺失"], 0)
            self.assertEqual(audit["成品损坏"], 0)
            self.assertTrue(
                any(
                    item["类型"] == "待协调"
                    and "墨色待确认" in item["说明"]
                    for item in audit["问题详情"]
                )
            )

    def test_audit_rejects_pending_and_unaccounted_ink_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph, _details = self._build_library(
                Path(directory),
                variants=2,
                complete_summary=False,
            )
            glyph.set_coordination_summary(
                {},
                180.0,
                geometry_completed=True,
                ink_completed=True,
                ink_enabled=True,
                ink_method="视觉墨量",
                ink_method_version=1,
                ink_counts={
                    "总数": 2,
                    "已达标": 0,
                    "待确认": 1,
                    "人工例外": 0,
                },
            )

            audit = ExportService(glyph).audit_readiness()

            self.assertFalse(audit["就绪"])
            self.assertFalse(audit["墨色统一完成"])
            self.assertEqual(audit["墨色待确认"], 1)
            self.assertEqual(audit["墨色未达标"], 1)
            self.assertTrue(any("墨色待确认 1 个" in reason for reason in audit["原因"]))
            self.assertTrue(any("墨色未达标 1 个" in reason for reason in audit["原因"]))

    def test_audit_accepts_explicit_manual_ink_exceptions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph, details = self._build_library(
                Path(directory),
                variants=2,
                complete_summary=False,
            )
            exception_record = details[1]["整体协调参数"]["墨色协调"]
            exception_record["是否达标"] = False
            exception_record["人工接受例外"] = True
            glyph.set_coordination_summary(
                {},
                180.0,
                geometry_completed=True,
                ink_completed=True,
                ink_enabled=True,
                ink_method="视觉墨量",
                ink_method_version=1,
                ink_counts={
                    "总数": 2,
                    "已达标": 1,
                    "待确认": 0,
                    "人工例外": 1,
                },
            )

            audit = ExportService(glyph).audit_readiness()

            self.assertTrue(audit["就绪"])
            self.assertTrue(audit["墨色统一完成"])
            self.assertEqual(audit["墨色已达标"], 1)
            self.assertEqual(audit["墨色人工例外"], 1)

    def test_versioned_ink_summary_survives_save_and_reload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph, _details = self._build_library(
                root,
                variants=2,
                complete_summary=False,
            )
            glyph.set_coordination_summary(
                {"字面宽度": 20},
                188.25,
                geometry_completed=True,
                ink_completed=True,
                ink_enabled=True,
                ink_method="视觉墨量",
                ink_method_version=2,
                ink_counts={
                    "总数": 2,
                    "已达标": 1,
                    "待确认": 0,
                    "人工例外": 1,
                },
            )
            glyph.save()

            reloaded = GlyphService("测试导出库", str(root / "测试导出库"))
            summary = reloaded.get_coordination_summary()

            self.assertEqual(summary["墨色方法"], "视觉墨量")
            self.assertEqual(summary["墨色方法版本"], 2)
            self.assertEqual(
                summary["墨色统计"],
                {"总数": 2, "已达标": 1, "待确认": 0, "人工例外": 1},
            )
            self.assertTrue(summary["墨色统一完成"])

    def test_audit_does_not_require_or_apply_ink_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph, details = self._build_library(
                root,
                complete_summary=False,
            )
            glyph.set_coordination_summary(
                {},
                geometry_completed=True,
                ink_completed=False,
                ink_enabled=False,
            )
            details[0]["整体协调参数"]["墨色协调"] = {
                "启用": False,
                "保存后墨色": 180.0,
                "保存后复测": True,
            }
            source_path = (
                Path(glyph.get_workflow_dirs()["成品"])
                / str(details[0]["成品文件"])
            )
            before = source_path.read_bytes()

            audit = ExportService(glyph).audit_readiness(verify_hash=True)

            self.assertTrue(audit["就绪"])
            self.assertFalse(audit["墨色统一启用"])
            self.assertEqual(source_path.read_bytes(), before)

    def test_audit_requires_every_variant_and_complete_coordination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph, details = self._build_library(root, complete_summary=False)
            pending_id = glyph.add_original(
                "乙", "乙-0001.tif", "乙.tif", "f" * 32
            )
            glyph.set_status(pending_id, config.STATUS_PENDING_MANUAL_REVIEW)

            audit = ExportService(glyph).audit_readiness()

            self.assertFalse(audit["就绪"])
            self.assertEqual(audit["总数"], 2)
            self.assertEqual(audit["已就绪"], 0)
            self.assertEqual(audit["待审核"], 1)
            self.assertEqual(audit["待协调"], 1)
            self.assertFalse(audit["几何协调完成"])
            self.assertTrue(any("一致性调整" in reason for reason in audit["原因"]))
            self.assertTrue(any(item["变体ID"] == pending_id for item in audit["问题详情"]))
            self.assertEqual(details[0]["状态"], config.STATUS_FINISHED)

    def test_audit_detects_missing_corrupt_and_hash_mismatched_finished_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph, details = self._build_library(root, variants=3)
            finished_dir = Path(glyph.get_workflow_dirs()["成品"])
            (finished_dir / str(details[0]["成品文件"])).unlink()
            (finished_dir / str(details[1]["成品文件"])).write_bytes(b"not-an-image")
            details[2]["成品MD5"] = "0" * 32

            audit = ExportService(glyph).audit_readiness(verify_hash=True)

            self.assertFalse(audit["就绪"])
            self.assertEqual(audit["成品缺失"], 1)
            self.assertEqual(audit["成品损坏"], 1)
            self.assertEqual(audit["校验不符"], 1)
            self.assertEqual(audit["已就绪"], 0)

    def test_audit_and_required_export_honor_cancellation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph, _details = self._build_library(root, variants=2)
            service = ExportService(glyph)

            audit = service.audit_readiness(
                verify_hash=True,
                cancel_check=lambda: True,
            )
            output_dir = root / "核对取消"
            result = service.export(
                str(output_dir),
                options=ExportOptions(mode=ExportService.MODE_LIBRARY_SPEC),
                require_ready=True,
                cancel_check=lambda: True,
            )

            self.assertTrue(audit["已取消"])
            self.assertFalse(audit["就绪"])
            self.assertTrue(result["已取消"])
            self.assertFalse(output_dir.exists())

    def test_audit_reports_per_variant_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph, _details = self._build_library(root, variants=2)
            progress: list[tuple[str, int, int]] = []

            audit = ExportService(glyph).audit_readiness(
                progress_callback=lambda message, current, total: progress.append(
                    (message, current, total)
                )
            )

            self.assertTrue(audit["就绪"])
            self.assertEqual(progress[0][1:], (0, 2))
            self.assertEqual(progress[-1][1:], (2, 2))
            self.assertEqual([item[1] for item in progress[1:-1]], [1, 2])

    def test_audit_and_export_reject_finished_path_outside_library_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph, details = self._build_library(root)
            outside = root / "外部.png"
            self._source_image().save(outside)
            details[0]["成品文件"] = "..\\..\\外部.png"

            service = ExportService(glyph)
            audit = service.audit_readiness()
            result = service.export(
                str(root / "输出"),
                options=ExportOptions(mode=ExportService.MODE_LIBRARY_SPEC),
            )

            self.assertEqual(audit["路径无效"], 1)
            self.assertFalse(audit["就绪"])
            self.assertEqual(result["成功"], 0)
            self.assertEqual(result["失败"], 1)
            self.assertFalse((root / "输出").exists())

    def test_audit_rejects_unsafe_finished_filename_without_normalizing_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph, details = self._build_library(Path(directory))
            details[0]["成品文件"] = f" {details[0]['成品文件']}"

            audit = ExportService(glyph).audit_readiness()

            self.assertEqual(audit["路径无效"], 1)
            self.assertEqual(audit["成品缺失"], 0)
            self.assertFalse(audit["就绪"])

    def test_audit_honors_unified_safe_stage_resolver_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph, details = self._build_library(Path(directory))
            with patch(
                "services.export_service.resolve_safe_stage_file",
                return_value="",
            ) as resolver:
                audit = ExportService(glyph).audit_readiness()

            resolver.assert_called_once_with(
                glyph.get_workflow_dirs()["成品"],
                details[0]["成品文件"],
            )
            self.assertEqual(audit["路径无效"], 1)
            self.assertEqual(audit["成品缺失"], 0)
            self.assertFalse(audit["就绪"])

    def test_library_spec_exports_exact_library_canvas_and_dpi(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph, details = self._build_library(root)
            options = ExportOptions(mode=ExportService.MODE_LIBRARY_SPEC)
            service = ExportService(glyph)

            preview = service.preview_image(details[0], options)
            output_dir = root / "按字库参数"
            result = service.export(str(output_dir), options=options, require_ready=True)

            source_path = (
                Path(glyph.get_workflow_dirs()["成品"])
                / str(details[0]["成品文件"])
            )
            self.assertEqual(preview.size, (20, 10))
            self.assertEqual(result["成功"], 1)
            self.assertFalse(result["已取消"])
            self.assertEqual(
                (output_dir / "甲.png").read_bytes(),
                source_path.read_bytes(),
            )
            with Image.open(output_dir / "甲.png") as exported:
                self.assertEqual(exported.size, (20, 10))
                self.assertAlmostEqual(exported.info["dpi"][0], 300, delta=0.1)
                self.assertEqual(exported.mode, "RGBA")

    def test_export_creates_missing_destination_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph, _details = self._build_library(root)
            output_dir = root / "尚不存在" / "多级目录" / "导出结果"

            result = ExportService(glyph).export(
                str(output_dir),
                options=ExportOptions(mode=ExportService.MODE_LIBRARY_SPEC),
            )

            self.assertEqual(result["成功"], 1)
            self.assertTrue((output_dir / "甲.png").is_file())
            self.assertFalse(list(output_dir.parent.glob(".fonteditor_export_*")))

    def test_trim_transparent_crops_alpha_bbox_without_resizing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph, details = self._build_library(root)
            options = ExportOptions(mode=ExportService.MODE_TRIM_TRANSPARENT)
            service = ExportService(glyph)

            preview = service.preview_image(details[0], options)
            output_dir = root / "去透明区"
            result = service.export(str(output_dir), options=options)

            self.assertEqual(preview.size, (10, 6))
            self.assertEqual(preview.getchannel("A").getbbox(), (0, 0, 10, 6))
            self.assertEqual(result["成功"], 1)
            with Image.open(output_dir / "甲.png") as exported:
                self.assertEqual(exported.size, (10, 6))
                self.assertAlmostEqual(exported.info["dpi"][0], 300, delta=0.1)

    def test_custom_spec_requires_library_aspect_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph, details = self._build_library(root)
            service = ExportService(glyph)
            with self.assertRaisesRegex(ValueError, "宽高比例"):
                service.preview_image(
                    details[0],
                    ExportOptions(
                        mode=ExportService.MODE_CUSTOM_SPEC,
                        dpi=600,
                        width=100,
                        height=100,
                    ),
                )

    def test_custom_larger_canvas_can_center_original_or_scale_up(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph, details = self._build_library(root)
            finished_path = (
                Path(glyph.get_workflow_dirs()["成品"])
                / str(details[0]["成品文件"])
            )
            expanded_source = self._source_image((90, 50))
            expanded_source.save(finished_path, "PNG", dpi=(300, 300))
            service = ExportService(glyph)
            centered = service.preview_image(
                details[0],
                ExportOptions(
                    mode=ExportService.MODE_CUSTOM_SPEC,
                    dpi=600,
                    width=80,
                    height=60,
                    allow_upscale=False,
                ),
            )
            enlarged = service.preview_image(
                details[0],
                ExportOptions(
                    mode=ExportService.MODE_CUSTOM_SPEC,
                    dpi=600,
                    width=80,
                    height=60,
                    allow_upscale=True,
                ),
            )

            self.assertEqual(centered.size, (90, 60))
            self.assertEqual(enlarged.size, (180, 100))
            self.assertEqual(centered.getchannel("A").getbbox(), (5, 7, 15, 13))
            centered_bbox = centered.getchannel("A").getbbox()
            enlarged_bbox = enlarged.getchannel("A").getbbox()
            self.assertIsNotNone(centered_bbox)
            self.assertIsNotNone(enlarged_bbox)
            if centered_bbox is None or enlarged_bbox is None:
                self.fail("自定义导出必须保留可见文字")
            self.assertGreater(
                enlarged_bbox[2] - enlarged_bbox[0],
                centered_bbox[2] - centered_bbox[0],
            )
            self.assertGreater(
                enlarged_bbox[3] - enlarged_bbox[1],
                centered_bbox[3] - centered_bbox[1],
            )
            output_dir = root / "扩展画布保持原尺寸"
            result = service.export(
                str(output_dir),
                options=ExportOptions(
                    mode=ExportService.MODE_CUSTOM_SPEC,
                    dpi=600,
                    width=80,
                    height=60,
                    allow_upscale=False,
                ),
            )
            self.assertEqual(result["成功"], 1)
            with Image.open(output_dir / "甲.png") as exported:
                self.assertEqual(exported.size, (90, 60))

    def test_custom_smaller_canvas_scales_full_product_proportionally(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph, details = self._build_library(root)
            finished_path = (
                Path(glyph.get_workflow_dirs()["成品"])
                / str(details[0]["成品文件"])
            )
            expanded_source = self._source_image((60, 40))
            expanded_source.save(finished_path, "PNG", dpi=(300, 300))
            reduced = ExportService(glyph).preview_image(
                details[0],
                ExportOptions(
                    mode=ExportService.MODE_CUSTOM_SPEC,
                    dpi=600,
                    width=20,
                    height=15,
                ),
            )
            self.assertEqual(reduced.size, (30, 20))
            bbox = reduced.getchannel("A").getbbox()
            self.assertIsNotNone(bbox)
            if bbox is None:
                self.fail("缩小导出必须保留可见文字")
            self.assertLessEqual(bbox[0], 3)
            self.assertGreaterEqual(bbox[2], 7)

    def test_custom_export_scales_complete_product_and_writes_dpi(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph, _details = self._build_library(root)
            output_dir = root / "自定义"
            options = ExportOptions(
                mode=ExportService.MODE_CUSTOM_SPEC,
                dpi=720,
                width=64,
                height=48,
                allow_upscale=True,
            )

            result = ExportService(glyph).export(str(output_dir), options=options)

            self.assertEqual(result["成功"], 1)
            with Image.open(output_dir / "甲.png") as exported:
                self.assertEqual(exported.size, (32, 16))
                self.assertAlmostEqual(exported.info["dpi"][0], 720, delta=0.1)

    def test_new_character_names_are_deterministic_for_multiple_variants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph, _details = self._build_library(root, variants=3)
            output_dir = root / "多变体"

            result = ExportService(glyph).export(
                str(output_dir),
                options=ExportOptions(mode=ExportService.MODE_TRIM_TRANSPARENT),
            )

            self.assertEqual(result["成功"], 3)
            self.assertEqual(
                sorted(path.name for path in output_dir.glob("*.png")),
                ["甲-0002.png", "甲-0003.png", "甲.png"],
            )

    def test_new_sequence_modes_follow_confirmed_numbering_rules(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph, _details = self._build_library(root, variants=10)
            normal_dir = root / "普通序号"
            padded_dir = root / "等宽序号"

            ExportService(glyph).export(
                str(normal_dir),
                options=ExportOptions(sequence_mode="普通序号"),
            )
            ExportService(glyph).export(
                str(padded_dir),
                options=ExportOptions(sequence_mode="自动等宽序号"),
            )

            self.assertEqual(
                {path.name for path in normal_dir.iterdir()},
                {"甲.png", *(f"甲-{index}.png" for index in range(1, 10))},
            )
            self.assertEqual(
                {path.name for path in padded_dir.iterdir()},
                {f"甲-{index:02d}.png" for index in range(1, 11)},
            )

    def test_new_export_supports_common_image_formats(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph, _details = self._build_library(root)
            expected = {
                "PNG": (".png", "RGBA"),
                "JPEG": (".jpg", "RGB"),
                "TIFF": (".tif", "RGBA"),
                "BMP": (".bmp", "RGB"),
                "WEBP": (".webp", "RGBA"),
            }
            for image_format, (extension, expected_mode) in expected.items():
                with self.subTest(image_format=image_format):
                    output_dir = root / image_format
                    result = ExportService(glyph).export(
                        str(output_dir),
                        options=ExportOptions(
                            sequence_mode="普通序号",
                            image_format=image_format,
                        ),
                    )
                    self.assertEqual(result["成功"], 1)
                    target = output_dir / f"甲{extension}"
                    self.assertTrue(target.is_file())
                    with Image.open(target) as exported:
                        self.assertEqual(exported.mode, expected_mode)

    def test_partial_export_only_writes_eligible_variants_and_keeps_full_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph, details = self._build_library(root, variants=3)
            output_dir = root / "部分成品"

            result = ExportService(glyph).export(
                str(output_dir),
                options=ExportOptions(mode=ExportService.MODE_LIBRARY_SPEC),
                eligible_variant_ids={str(details[1]["变体ID"])},
            )

            self.assertEqual(result["成功"], 1)
            self.assertEqual(result["跳过"], 2)
            self.assertEqual(
                sorted(path.name for path in output_dir.glob("*.png")),
                ["甲-0002.png"],
            )

    def test_original_name_mode_uses_name_before_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph, details = self._build_library(root)
            details[0]["导入前文件名"] = "用户原稿.tif"
            output_dir = root / "原文件名"

            result = ExportService(glyph).export(
                str(output_dir),
                options=ExportOptions(
                    mode=ExportService.MODE_LIBRARY_SPEC,
                    name_mode="原文件名",
                ),
            )

            self.assertEqual(result["成功"], 1)
            self.assertTrue((output_dir / "用户原稿.png").is_file())
            self.assertFalse((output_dir / "甲-0001.png").exists())

    def test_character_name_mode_rejects_path_escape_from_corrupt_char(self) -> None:
        for unsafe_char in ("../逃逸字符", "..\\逃逸字符"):
            with self.subTest(unsafe_char=unsafe_char):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    glyph, details = self._build_library(root)
                    details[0]["归属字"] = unsafe_char
                    output_dir = root / "字符命名输出"
                    output_dir.mkdir()

                    result = ExportService(glyph).export(
                        str(output_dir),
                        options=ExportOptions(
                            mode=ExportService.MODE_LIBRARY_SPEC,
                            name_mode="字符",
                        ),
                    )

                    self.assertEqual(result["成功"], 0)
                    self.assertEqual(result["失败"], 1)
                    self.assertEqual(list(output_dir.iterdir()), [])
                    self.assertFalse((root / "逃逸字符.png").exists())
                    self.assertFalse(list(root.glob(".fonteditor_export_*")))

    def test_original_name_mode_rejects_path_escape_from_corrupt_metadata(self) -> None:
        for unsafe_name in ("../逃逸原稿.tif", "..\\逃逸原稿.tif"):
            with self.subTest(unsafe_name=unsafe_name):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    glyph, details = self._build_library(root)
                    details[0]["导入前文件名"] = unsafe_name
                    output_dir = root / "原文件名输出"
                    output_dir.mkdir()

                    result = ExportService(glyph).export(
                        str(output_dir),
                        options=ExportOptions(
                            mode=ExportService.MODE_LIBRARY_SPEC,
                            name_mode="原文件名",
                        ),
                    )

                    self.assertEqual(result["成功"], 0)
                    self.assertEqual(result["失败"], 1)
                    self.assertEqual(list(output_dir.iterdir()), [])
                    self.assertFalse((root / "逃逸原稿.png").exists())
                    self.assertFalse(list(root.glob(".fonteditor_export_*")))

    def test_new_export_refuses_existing_same_name_without_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph, _details = self._build_library(root)
            output_dir = root / "已有文件"
            output_dir.mkdir()
            existing = output_dir / "甲.png"
            existing.write_bytes(b"existing-content")

            result = ExportService(glyph).export(
                str(output_dir),
                options=ExportOptions(mode=ExportService.MODE_LIBRARY_SPEC),
            )

            self.assertEqual(result["成功"], 0)
            self.assertEqual(result["失败"], 1)
            self.assertEqual(existing.read_bytes(), b"existing-content")
            self.assertFalse(list(root.glob(".fonteditor_export_*")))

    def test_conflict_preflight_reports_deterministic_targets_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph, _details = self._build_library(root, variants=2)
            output_dir = root / "冲突预检"
            output_dir.mkdir()
            (output_dir / "甲.png").write_bytes(b"first")
            (output_dir / "甲-0002.png").write_bytes(b"second")

            conflicts = ExportService(glyph).find_destination_conflicts(
                str(output_dir),
                ExportOptions(mode=ExportService.MODE_LIBRARY_SPEC),
            )

            self.assertEqual(
                [conflict.destination_name for conflict in conflicts],
                ["甲.png", "甲-0002.png"],
            )
            self.assertEqual([conflict.char for conflict in conflicts], ["甲", "甲"])
            self.assertEqual([conflict.file_size for conflict in conflicts], [5, 6])
            self.assertTrue(
                all(Path(conflict.destination_path).is_file() for conflict in conflicts)
            )

    def test_mixed_overwrite_skip_and_new_export_preserves_user_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph, _details = self._build_library(root, variants=3)
            output_dir = root / "混合冲突"
            output_dir.mkdir()
            first = output_dir / "甲.png"
            second = output_dir / "甲-0002.png"
            first.write_bytes(b"replace-me")
            second.write_bytes(b"keep-me")
            service = ExportService(glyph)
            options = ExportOptions(mode=ExportService.MODE_LIBRARY_SPEC)
            conflicts = service.find_destination_conflicts(str(output_dir), options)
            decisions = (
                ExportConflictDecision(
                    conflicts[0],
                    ExportService.CONFLICT_OVERWRITE,
                ),
                ExportConflictDecision(
                    conflicts[1],
                    ExportService.CONFLICT_SKIP,
                ),
            )

            result = service.export(
                str(output_dir),
                options=options,
                conflict_decisions=decisions,
            )

            self.assertEqual(result["成功"], 2)
            self.assertEqual(result["覆盖"], 1)
            self.assertEqual(result["跳过"], 1)
            self.assertNotEqual(first.read_bytes(), b"replace-me")
            self.assertEqual(second.read_bytes(), b"keep-me")
            self.assertTrue((output_dir / "甲-0003.png").is_file())
            with Image.open(first) as exported:
                self.assertEqual(exported.size, (20, 10))
            self.assertFalse(list(root.glob(".fonteditor_export_*")))
            self.assertFalse(list(root.glob(".fonteditor_export_backup_*")))

    def test_changed_conflict_after_confirmation_is_refused_without_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph, _details = self._build_library(root)
            output_dir = root / "冲突变化"
            output_dir.mkdir()
            existing = output_dir / "甲.png"
            existing.write_bytes(b"before")
            service = ExportService(glyph)
            options = ExportOptions(mode=ExportService.MODE_LIBRARY_SPEC)
            conflict = service.find_destination_conflicts(str(output_dir), options)[0]
            existing.write_bytes(b"changed-after-confirmation")

            result = service.export(
                str(output_dir),
                options=options,
                conflict_decisions=(
                    ExportConflictDecision(
                        conflict,
                        ExportService.CONFLICT_OVERWRITE,
                    ),
                ),
            )

            self.assertEqual(result["成功"], 0)
            self.assertEqual(result["失败"], 1)
            self.assertEqual(existing.read_bytes(), b"changed-after-confirmation")
            self.assertTrue(
                any("确认后发生变化" in reason for _item, reason in result["失败详情"])
            )
            self.assertFalse(list(root.glob(".fonteditor_export_*")))

    def test_overwrite_commit_failure_restores_old_file_and_removes_new_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph, _details = self._build_library(root, variants=2)
            output_dir = root / "覆盖回滚"
            output_dir.mkdir()
            existing = output_dir / "甲.png"
            existing.write_bytes(b"original-bytes")
            service = ExportService(glyph)
            options = ExportOptions(mode=ExportService.MODE_LIBRARY_SPEC)
            conflict = service.find_destination_conflicts(str(output_dir), options)[0]

            with patch(
                "services.export_service.os.rename",
                side_effect=OSError("模拟新增文件提交失败"),
            ):
                with self.assertRaisesRegex(OSError, "新增文件提交失败"):
                    service.export(
                        str(output_dir),
                        options=options,
                        conflict_decisions=(
                            ExportConflictDecision(
                                conflict,
                                ExportService.CONFLICT_OVERWRITE,
                            ),
                        ),
                    )

            self.assertEqual(existing.read_bytes(), b"original-bytes")
            self.assertFalse((output_dir / "甲-0002.png").exists())
            self.assertFalse(list(root.glob(".fonteditor_export_*")))
            self.assertFalse(list(root.glob(".fonteditor_export_backup_*")))

    def test_cancelled_overwrite_keeps_existing_file_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph, _details = self._build_library(root)
            output_dir = root / "覆盖前取消"
            output_dir.mkdir()
            existing = output_dir / "甲.png"
            existing.write_bytes(b"do-not-change")
            service = ExportService(glyph)
            options = ExportOptions(mode=ExportService.MODE_LIBRARY_SPEC)
            conflict = service.find_destination_conflicts(str(output_dir), options)[0]

            result = service.export(
                str(output_dir),
                options=options,
                conflict_decisions=(
                    ExportConflictDecision(
                        conflict,
                        ExportService.CONFLICT_OVERWRITE,
                    ),
                ),
                cancel_check=lambda: True,
            )

            self.assertTrue(result["已取消"])
            self.assertEqual(existing.read_bytes(), b"do-not-change")
            self.assertFalse(list(root.glob(".fonteditor_export_*")))

    def test_overwrite_backup_failure_changes_no_destination_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph, _details = self._build_library(root)
            output_dir = root / "备份失败"
            output_dir.mkdir()
            existing = output_dir / "甲.png"
            existing.write_bytes(b"original-before-backup")
            service = ExportService(glyph)
            options = ExportOptions(mode=ExportService.MODE_LIBRARY_SPEC)
            conflict = service.find_destination_conflicts(str(output_dir), options)[0]

            with patch(
                "services.export_service.shutil.copy2",
                side_effect=OSError("模拟备份失败"),
            ):
                with self.assertRaisesRegex(OSError, "备份失败"):
                    service.export(
                        str(output_dir),
                        options=options,
                        conflict_decisions=(
                            ExportConflictDecision(
                                conflict,
                                ExportService.CONFLICT_OVERWRITE,
                            ),
                        ),
                    )

            self.assertEqual(existing.read_bytes(), b"original-before-backup")
            self.assertFalse(list(root.glob(".fonteditor_export_*")))
            self.assertFalse(list(root.glob(".fonteditor_export_backup_*")))

    def test_same_name_directory_is_not_an_overwritable_file_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph, _details = self._build_library(root)
            output_dir = root / "目录冲突"
            output_dir.mkdir()
            (output_dir / "甲.png").mkdir()

            with self.assertRaisesRegex(ValueError, "同名项不是文件"):
                ExportService(glyph).find_destination_conflicts(
                    str(output_dir),
                    ExportOptions(mode=ExportService.MODE_LIBRARY_SPEC),
                )

    def test_output_directory_cannot_be_inside_current_library(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph, _details = self._build_library(root)
            target = Path(glyph.ziku_dir) / "导出"

            with self.assertRaisesRegex(ValueError, "不能位于当前字库内部"):
                ExportService(glyph).export(
                    str(target),
                    options=ExportOptions(mode=ExportService.MODE_LIBRARY_SPEC),
                )

            self.assertFalse(target.exists())

    def test_second_image_failure_leaves_no_partial_export(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph, _details = self._build_library(root, variants=2)
            output_dir = root / "事务失败"
            output_dir.mkdir()
            sentinel = output_dir / "已有.txt"
            sentinel.write_text("保留", encoding="utf-8")
            original_copy = shutil.copyfile
            calls = 0

            def fail_second(source, target):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("模拟第二张保存失败")
                return original_copy(source, target)

            with patch("services.export_service.shutil.copyfile", side_effect=fail_second):
                result = ExportService(glyph).export(
                    str(output_dir),
                    options=ExportOptions(mode=ExportService.MODE_LIBRARY_SPEC),
                )

            self.assertEqual(result["成功"], 0)
            self.assertEqual(result["失败"], 2)
            self.assertTrue(any("第二张保存失败" in reason for _item, reason in result["失败详情"]))
            self.assertIn("第二张保存失败", result["失败详情"][0][1])
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "保留")
            self.assertFalse(list(output_dir.glob("*.png")))
            self.assertFalse(list(root.glob(".fonteditor_export_*")))

    def test_existing_destination_commit_failure_rolls_back_first_installed_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph, _details = self._build_library(root, variants=2)
            output_dir = root / "提交失败"
            output_dir.mkdir()
            sentinel = output_dir / "已有.txt"
            sentinel.write_text("保留", encoding="utf-8")
            original_rename = os.rename
            rename_calls = 0

            def fail_second_commit(source, destination):
                nonlocal rename_calls
                rename_calls += 1
                if rename_calls == 2:
                    raise OSError("模拟第二个文件提交失败")
                return original_rename(source, destination)

            with patch("services.export_service.os.rename", side_effect=fail_second_commit):
                with self.assertRaisesRegex(OSError, "第二个文件提交失败"):
                    ExportService(glyph).export(
                        str(output_dir),
                        options=ExportOptions(mode=ExportService.MODE_LIBRARY_SPEC),
                    )

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "保留")
            self.assertFalse(list(output_dir.glob("*.png")))
            self.assertFalse(list(root.glob(".fonteditor_export_*")))

    def test_cancel_removes_staging_and_returns_explicit_cancelled_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph, _details = self._build_library(root, variants=2)
            output_dir = root / "取消导出"
            checks = 0

            def cancel_after_first() -> bool:
                nonlocal checks
                checks += 1
                return checks >= 2

            result = ExportService(glyph).export(
                str(output_dir),
                options=ExportOptions(mode=ExportService.MODE_LIBRARY_SPEC),
                cancel_check=cancel_after_first,
            )

            self.assertTrue(result["已取消"])
            self.assertEqual(result["成功"], 0)
            self.assertEqual(result["失败"], 0)
            self.assertFalse(output_dir.exists())
            self.assertFalse(list(root.glob(".fonteditor_export_*")))

    def test_cancel_after_render_does_not_enter_png_save(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph, _details = self._build_library(root)
            output_dir = root / "渲染后取消"
            checks = 0

            def cancel_after_render() -> bool:
                nonlocal checks
                checks += 1
                return checks >= 2

            with patch.object(ExportService, "_save_output") as save_output:
                result = ExportService(glyph).export(
                    str(output_dir),
                    options=ExportOptions(mode=ExportService.MODE_LIBRARY_SPEC),
                    cancel_check=cancel_after_render,
                )

            self.assertTrue(result["已取消"])
            save_output.assert_not_called()
            self.assertFalse(output_dir.exists())
            self.assertFalse(list(root.glob(".fonteditor_export_*")))

    def test_oversized_source_is_rejected_before_rgba_decode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph, details = self._build_library(root)
            output_dir = root / "超大源图"
            original_limit = ExportService.MAX_SOURCE_PIXELS

            with patch.object(ExportService, "MAX_SOURCE_PIXELS", 100):
                result = ExportService(glyph).export(
                    str(output_dir),
                    options=ExportOptions(mode=ExportService.MODE_LIBRARY_SPEC),
                )

            self.assertEqual(ExportService.MAX_SOURCE_PIXELS, original_limit)
            self.assertEqual(result["成功"], 0)
            self.assertEqual(result["失败"], 1)
            self.assertTrue(
                any(
                    str(details[0]["变体ID"]) == variant_id
                    and "源图尺寸过大" in reason
                    for variant_id, reason in result["失败详情"]
                )
            )
            self.assertFalse(output_dir.exists())

    def test_all_transparent_image_fails_without_creating_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph, details = self._build_library(root)
            finished_path = (
                Path(glyph.get_workflow_dirs()["成品"])
                / str(details[0]["成品文件"])
            )
            Image.new("RGBA", (20, 20), (0, 0, 0, 0)).save(finished_path)
            output_dir = root / "空图失败"

            result = ExportService(glyph).export(
                str(output_dir),
                options=ExportOptions(mode=ExportService.MODE_TRIM_TRANSPARENT),
            )

            self.assertEqual(result["失败"], 1)
            self.assertFalse(output_dir.exists())

    def test_legacy_export_signature_still_supports_bmp_and_unique_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph, _details = self._build_library(root)
            output_dir = root / "旧接口"
            output_dir.mkdir()
            (output_dir / "甲.bmp").write_bytes(b"old-file")

            result = ExportService(glyph).export(
                str(output_dir),
                name_mode="字符",
                transparent_background=False,
                output_style="灰度保真",
            )

            self.assertEqual(result["成功"], 1)
            self.assertEqual((output_dir / "甲.bmp").read_bytes(), b"old-file")
            with Image.open(output_dir / "甲-1.bmp") as exported:
                self.assertEqual(exported.mode, "RGB")

    def test_invalid_custom_dimensions_are_rejected_before_disk_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph, _details = self._build_library(root)
            output_dir = root / "非法尺寸"

            with self.assertRaisesRegex(ValueError, "画布宽度.*正整数"):
                ExportService(glyph).export(
                    str(output_dir),
                    options=ExportOptions(
                        mode=ExportService.MODE_CUSTOM_SPEC,
                        dpi=300,
                        width=0,
                        height=100,
                    ),
                )

            self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
