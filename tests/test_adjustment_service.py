"""整体协调墨色归一、状态摘要与页面恢复回归测试。"""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PIL import Image
from PySide6.QtCore import QPoint
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

import config
import services.adjustment_service as adjustment_service_module
from data.library_database import LibraryDatabase
from services.adjustment_service import AdjustmentService, CoordinationCancelled
from services.batch_persistence import BatchPersistenceSession
from services.glyph_service import GlyphService
from ui.pages.consistency_page import ConsistencyPage
from ui.widgets.review_canvas import ReviewCanvas


class AdjustmentServiceTests(unittest.TestCase):
    """验证整体协调始终从审核来源生成，并如实记录墨色处理。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_coordination_rejects_library_locked_by_other_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph, variants = self._build_reviewed_library(Path(directory), (100,))
            service = AdjustmentService(glyph)
            session = BatchPersistenceSession(glyph)
            try:
                with self.assertRaisesRegex(RuntimeError, "正在执行其他批处理任务"):
                    service.save_coordinated_variants(variants, {})
            finally:
                session.finish()

    def test_coordination_defaults_expose_new_and_compatible_scale_fields(self) -> None:
        normalized = AdjustmentService._normalized_coordination(None)

        self.assertEqual(normalized["等比缩放"], 1.0)
        self.assertEqual(normalized["水平拉伸"], 1.0)
        self.assertEqual(normalized["垂直拉伸"], 1.0)
        self.assertEqual(normalized["缩放X"], 1.0)
        self.assertEqual(normalized["缩放Y"], 1.0)

    def test_legacy_coordination_scale_fields_restore_without_visual_change(self) -> None:
        normalized = AdjustmentService._normalized_coordination(
            {"缩放X": 1.25, "缩放Y": 0.8}
        )

        self.assertEqual(normalized["等比缩放"], 1.0)
        self.assertEqual(normalized["水平拉伸"], 1.25)
        self.assertEqual(normalized["垂直拉伸"], 0.8)
        self.assertEqual(normalized["缩放X"], 1.25)
        self.assertEqual(normalized["缩放Y"], 0.8)

    def test_new_coordination_scales_write_legacy_derived_fields(self) -> None:
        normalized = AdjustmentService._normalized_coordination(
            {"等比缩放": 1.5, "水平拉伸": 1.2, "垂直拉伸": 0.75}
        )

        self.assertEqual(normalized["等比缩放"], 1.5)
        self.assertEqual(normalized["水平拉伸"], 1.2)
        self.assertEqual(normalized["垂直拉伸"], 0.75)
        self.assertAlmostEqual(normalized["缩放X"], 1.8)
        self.assertAlmostEqual(normalized["缩放Y"], 1.125)

    def test_coordination_values_match_review_canvas_limits(self) -> None:
        normalized = AdjustmentService._normalized_coordination(
            {
                "移动X": 20_000,
                "移动Y": -20_000,
                "等比缩放": 100,
                "水平拉伸": 0,
                "垂直拉伸": float("nan"),
                "扭曲": [20_000, -20_000, "bad", 1, 2, 3, 4, 5],
            }
        )

        self.assertEqual(normalized["移动X"], 8_192.0)
        self.assertEqual(normalized["移动Y"], -8_192.0)
        self.assertEqual(normalized["等比缩放"], 5.0)
        self.assertEqual(normalized["水平拉伸"], 0.05)
        self.assertEqual(normalized["垂直拉伸"], 1.0)
        self.assertEqual(normalized["扭曲"][:3], [8_192.0, -8_192.0, 0.0])

        upper = AdjustmentService._normalized_coordination(
            {"等比缩放": 100, "水平拉伸": 100, "垂直拉伸": 100}
        )
        self.assertEqual(upper["等比缩放"], 5.0)
        self.assertEqual(upper["水平拉伸"], 5.0)
        self.assertEqual(upper["垂直拉伸"], 5.0)

    def test_coordination_transform_uses_shared_renderer(self) -> None:
        source = Image.new("RGBA", (8, 6), (0, 0, 0, 255))

        with patch(
            "services.adjustment_service.render_transformed_rgba",
            wraps=__import__(
                "services.adjustment_service",
                fromlist=["render_transformed_rgba"],
            ).render_transformed_rgba,
        ) as render:
            result = AdjustmentService._apply_coordination_transform(
                source,
                {
                    "等比缩放": 1.2,
                    "水平拉伸": 0.9,
                    "垂直拉伸": 1.1,
                    "旋转": 12.0,
                    "扭曲": [-1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 0.0],
                },
            )

        self.assertEqual(render.call_count, 1)
        self.assertEqual(result.mode, "RGBA")
        self.assertGreater(result.width, 0)
        self.assertGreater(result.height, 0)

    def test_legacy_shear_converts_to_canvas_corner_distortion(self) -> None:
        transform = AdjustmentService.coordination_to_canvas_transform(
            {
                "缩放X": 1.0,
                "缩放Y": 1.0,
                "斜切X": 10.0,
                "斜切Y": 0.0,
                "扭曲": [0.0] * 8,
            },
            (100, 80),
        )
        distort = transform["distort"]

        self.assertAlmostEqual(transform["scale"], 1.0)
        self.assertAlmostEqual(transform["stretch_w"], 1.0)
        self.assertLess(distort[0], 0.0)
        self.assertAlmostEqual(distort[0], distort[2])
        self.assertGreater(distort[4], 0.0)
        self.assertAlmostEqual(distort[4], distort[6])
        self.assertAlmostEqual(sum(distort[::2]), 0.0, places=6)
        self.assertTrue(all(math.isclose(value, 0.0) for value in distort[1::2]))

    def test_canvas_transform_writes_new_model_with_zero_legacy_shear(self) -> None:
        normalized = AdjustmentService.coordination_from_canvas_transform(
            {
                "x": 8.0,
                "y": -4.0,
                "scale": 1.25,
                "stretch_w": 0.8,
                "stretch_h": 1.1,
                "rotation": 3.0,
                "distort": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            }
        )

        self.assertEqual(normalized["移动X"], 8.0)
        self.assertEqual(normalized["移动Y"], -4.0)
        self.assertEqual(normalized["等比缩放"], 1.25)
        self.assertEqual(normalized["水平拉伸"], 0.8)
        self.assertEqual(normalized["垂直拉伸"], 1.1)
        self.assertAlmostEqual(normalized["缩放X"], 1.0)
        self.assertAlmostEqual(normalized["缩放Y"], 1.375)
        self.assertEqual(normalized["斜切X"], 0.0)
        self.assertEqual(normalized["斜切Y"], 0.0)
        self.assertEqual(normalized["扭曲"], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    def test_reviewed_source_public_api_prefers_manual_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph, variants = self._build_reviewed_library(Path(directory), (90,))
            detail = variants[0]
            manual_dir = Path(glyph.get_workflow_dirs()["手工审核"])
            manual_dir.mkdir(parents=True, exist_ok=True)
            manual_path = manual_dir / "manual.png"
            Image.new("RGBA", (2, 2), (10, 20, 30, 222)).save(manual_path)
            detail["审核文件"] = manual_path.name
            service = AdjustmentService(glyph)

            self.assertEqual(service.reviewed_source_path(detail), str(manual_path))
            loaded = service.load_reviewed_image(detail)
            self.assertIsNotNone(loaded)
            if loaded is None:
                self.fail("应加载手工审核文件")
            self.assertEqual(loaded.mode, "RGBA")
            self.assertEqual(loaded.getpixel((0, 0)), (10, 20, 30, 222))

    def test_reviewed_source_load_decodes_selected_file_only_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph, variants = self._build_reviewed_library(Path(directory), (90,))
            service = AdjustmentService(glyph)

            with patch.object(
                service,
                "_open_rgba",
                wraps=service._open_rgba,
            ) as open_rgba:
                image, path = service.load_reviewed_source(variants[0])

            self.assertIsNotNone(image)
            self.assertTrue(path.endswith("甲-0001.png"))
            self.assertEqual(open_rgba.call_count, 1)
            if image is not None:
                image.close()

    def test_analyze_reports_each_glyph_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph, _variants = self._build_reviewed_library(
                Path(directory),
                (80, 120, 200),
            )
            progress: list[tuple[int, int, str]] = []

            baseline = AdjustmentService(glyph).analyze(
                progress_callback=lambda current, total, label: progress.append(
                    (current, total, label)
                )
            )

            self.assertEqual([item[:2] for item in progress], [(1, 3), (2, 3), (3, 3)])
            self.assertTrue(all(item[2] for item in progress))
            self.assertEqual(baseline["有效数"], 3)

    def test_reviewed_source_public_api_does_not_hide_corrupt_manual_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph, variants = self._build_reviewed_library(Path(directory), (90,))
            detail = variants[0]
            manual_dir = Path(glyph.get_workflow_dirs()["手工审核"])
            manual_dir.mkdir(parents=True, exist_ok=True)
            corrupt_path = manual_dir / "corrupt.png"
            corrupt_path.write_bytes(b"not a png")
            detail["审核文件"] = corrupt_path.name
            service = AdjustmentService(glyph)

            self.assertEqual(service.reviewed_source_path(detail), "")
            self.assertIsNone(service.load_reviewed_image(detail))

    def test_saved_coordination_records_new_and_compatible_scale_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph, variants = self._build_reviewed_library(Path(directory), (100,))
            service = AdjustmentService(glyph)
            detail = variants[0]
            variant_id = str(detail["变体ID"])

            result = service.save_coordinated_variants(
                [detail],
                {
                    variant_id: {
                        "等比缩放": 1.25,
                        "水平拉伸": 0.8,
                        "垂直拉伸": 1.1,
                        "斜切X": 0.0,
                        "斜切Y": 0.0,
                        "扭曲": [0.0] * 8,
                    }
                },
                {"启用": False, "基准": 100.0},
            )
            saved = detail["整体协调参数"]["整体变换"]

            self.assertEqual(result["成功"], 1)
            self.assertEqual(saved["等比缩放"], 1.25)
            self.assertEqual(saved["水平拉伸"], 0.8)
            self.assertEqual(saved["垂直拉伸"], 1.1)
            self.assertAlmostEqual(saved["缩放X"], 1.0)
            self.assertAlmostEqual(saved["缩放Y"], 1.375)
            self.assertEqual(service.load_saved_coordination_adjustments(detail), saved)

    def test_invalid_perspective_quadrilateral_is_rejected(self) -> None:
        source = Image.new("RGBA", (10, 10), (0, 0, 0, 255))
        adjustments = AdjustmentService._normalized_coordination(
            {"扭曲": [9.0, 0.0, -9.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
        )

        with self.assertRaisesRegex(ValueError, "四边形无效"):
            AdjustmentService._apply_coordination_transform(source, adjustments)

    def test_asymmetric_combined_transform_matches_canvas_and_saved_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph, detail, source_path = self._build_asymmetric_library(Path(directory))
            service = AdjustmentService(glyph)
            variant_id = str(detail["变体ID"])
            adjustments = {
                "移动X": 11.0,
                "移动Y": -7.0,
                "等比缩放": 1.18,
                "水平拉伸": 0.86,
                "垂直拉伸": 1.23,
                "旋转": 17.0,
                "扭曲": [-4.5, 2.0, 7.0, -3.5, 3.5, 6.0, -6.0, 1.5],
            }
            with Image.open(source_path) as source:
                bounding_box = source.convert("RGBA").getchannel("A").getbbox()
            self.assertIsNotNone(bounding_box)
            if bounding_box is None:
                self.fail("非对称测试字形不应为空")
            content_size = (
                bounding_box[2] - bounding_box[0],
                bounding_box[3] - bounding_box[1],
            )

            canvas = ReviewCanvas()
            canvas.set_image(QImage(str(source_path)), (128, 128))
            transform = service.coordination_to_canvas_transform(
                adjustments,
                content_size,
            )
            self.assertTrue(canvas.set_transform(**transform))
            canvas_pixels = self._qimage_rgba(canvas.image())
            canvas_origin = canvas.output_origin()

            result = service.save_coordinated_variants(
                [detail],
                {variant_id: adjustments},
                {"启用": False, "基准": 180.0},
                service.analyze(),
            )
            finished_path = Path(glyph.get_workflow_dirs()["成品"]) / str(
                detail["成品文件"]
            )
            with Image.open(finished_path) as saved:
                saved_pixels = np.asarray(saved.convert("RGBA"), dtype=np.uint8).copy()
            parameters = detail["整体协调参数"]

            self.assertEqual(result["失败"], 0)
            self.assertEqual(canvas_pixels.shape, saved_pixels.shape)
            self.assertEqual(
                canvas_origin,
                QPoint(parameters["对称扩展X"], parameters["对称扩展Y"]),
            )
            np.testing.assert_array_equal(canvas_pixels, saved_pixels)
            canvas.close()
            canvas.deleteLater()

    def test_legacy_shear_open_without_editing_resaves_identically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph, detail, source_path = self._build_asymmetric_library(Path(directory))
            service = AdjustmentService(glyph)
            variant_id = str(detail["变体ID"])
            legacy = {
                "移动X": -5.0,
                "移动Y": 4.0,
                "缩放X": 1.12,
                "缩放Y": 0.91,
                "旋转": -13.0,
                "斜切X": 12.0,
                "斜切Y": -7.0,
                "扭曲": [0.0] * 8,
            }
            with Image.open(source_path) as source:
                bounding_box = source.convert("RGBA").getchannel("A").getbbox()
            self.assertIsNotNone(bounding_box)
            if bounding_box is None:
                self.fail("非对称测试字形不应为空")
            content_size = (
                bounding_box[2] - bounding_box[0],
                bounding_box[3] - bounding_box[1],
            )

            first = service.save_coordinated_variants(
                [detail],
                {variant_id: legacy},
                {"启用": False, "基准": 180.0},
                service.analyze(),
            )
            finished_path = Path(glyph.get_workflow_dirs()["成品"]) / str(
                detail["成品文件"]
            )
            with Image.open(finished_path) as saved:
                before = np.asarray(saved.convert("RGBA"), dtype=np.uint8).copy()
            before_origin = (
                detail["整体协调参数"]["对称扩展X"],
                detail["整体协调参数"]["对称扩展Y"],
            )

            canvas_transform = service.coordination_to_canvas_transform(
                legacy,
                content_size,
            )
            canvas = ReviewCanvas()
            canvas.set_image(QImage(str(source_path)), (128, 128))
            self.assertTrue(canvas.set_transform(**canvas_transform))
            canvas_pixels = self._qimage_rgba(canvas.image())
            canvas_origin = canvas.output_origin()
            reopened_without_edits = service.coordination_from_canvas_transform(
                canvas.transform()
            )
            second = service.save_coordinated_variants(
                [detail],
                {variant_id: reopened_without_edits},
                {"启用": False, "基准": 180.0},
                service.analyze(),
            )
            with Image.open(finished_path) as saved:
                after = np.asarray(saved.convert("RGBA"), dtype=np.uint8).copy()
            after_origin = (
                detail["整体协调参数"]["对称扩展X"],
                detail["整体协调参数"]["对称扩展Y"],
            )

            self.assertEqual(first["失败"], 0)
            self.assertEqual(second["失败"], 0)
            self.assertEqual(canvas_origin, QPoint(*before_origin))
            self.assertEqual(before_origin, after_origin)
            np.testing.assert_array_equal(canvas_pixels, before)
            np.testing.assert_array_equal(before, after)
            canvas.close()
            canvas.deleteLater()

    def test_disabled_ink_coordination_keeps_rgba_pixels(self) -> None:
        source = Image.new("RGBA", (3, 2), (12, 34, 56, 0))
        source.putpixel((1, 0), (78, 90, 123, 80))
        source.putpixel((2, 1), (45, 67, 89, 180))

        result, record = AdjustmentService._apply_ink_coordination(
            source,
            {"启用": False, "基准": 160.0},
        )

        self.assertEqual(result.tobytes(), source.tobytes())
        self.assertFalse(record["启用"])
        self.assertEqual(record["基准"], 160.0)
        self.assertEqual(record["调整前墨色"], record["调整后墨色"])
        self.assertEqual(record["调整方式"], "关闭")
        self.assertFalse(record["触及限制"])
        self.assertEqual(record["跳过原因"], "已关闭墨色统一")

    def test_enabled_ink_coordination_records_before_and_after_values(self) -> None:
        source = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        for y in range(1, 3):
            for x in range(1, 3):
                source.putpixel((x, y), (0, 0, 0, 90 + x * 10 + y * 5))

        result, record = AdjustmentService._apply_ink_coordination(
            source,
            {"启用": True, "基准": 190.0},
        )

        self.assertNotEqual(result.getchannel("A").tobytes(), source.getchannel("A").tobytes())
        self.assertTrue(record["启用"])
        self.assertTrue(record["已应用"])
        self.assertIsNone(record["Gamma"])
        self.assertIsNotNone(record["增益"])
        self.assertEqual(record["调整方式"], "比例增益")
        self.assertEqual(record["方法版本"], AdjustmentService.INK_METHOD_VERSION)
        self.assertIsNotNone(record["调整前墨色"])
        self.assertIsNotNone(record["调整后墨色"])
        self.assertTrue(record["是否达标"])
        self.assertEqual(record["状态"], "墨色已达标")
        self.assertEqual(record["跳过原因"], "")

    def test_visual_coverage_combines_rgb_and_alpha_without_mutating_source(self) -> None:
        source = Image.new("RGBA", (3, 1), (255, 255, 255, 255))
        source.putpixel((1, 0), (106, 106, 106, 255))
        source.putpixel((2, 0), (0, 0, 0, 128))
        original = source.tobytes()

        working = AdjustmentService.prepare_ink_working_copy(
            source,
            {"启用": True, "基准": 180.0},
        )

        self.assertEqual(source.tobytes(), original)
        working_pixels = np.asarray(working, dtype=np.uint8)
        self.assertEqual(working_pixels[..., 3].tolist(), [[0, 149, 128]])
        self.assertTrue(np.all(working_pixels[..., :3] == 0))

    def test_true_binary_ink_can_follow_gray_library_baseline(self) -> None:
        source = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        for y in range(1, 3):
            for x in range(1, 3):
                source.putpixel((x, y), (0, 0, 0, 255))

        result, record = AdjustmentService.apply_ink_preview(
            source,
            {"启用": True, "基准": 197.0},
        )
        output_alpha = np.array(result.getchannel("A"), dtype=np.uint8, copy=True)
        source_alpha = np.array(source.getchannel("A"), dtype=np.uint8, copy=True)

        self.assertEqual(record["像素类型"], "视觉纯二值")
        self.assertAlmostEqual(record["增益"], 197.0 / 255.0, places=5)
        self.assertEqual(record["调整方式"], "比例增益")
        self.assertFalse(record["触及限制"])
        self.assertEqual(record["调整后墨色"], 197.0)
        self.assertTrue(record["是否达标"])
        self.assertTrue(np.all(output_alpha[source_alpha > 0] == 197))
        self.assertTrue(np.all(output_alpha[source_alpha == 0] == 0))

    def test_very_light_149_ink_reaches_197_without_gamma_limit(self) -> None:
        source = Image.new("RGBA", (5, 5), (0, 0, 0, 0))
        for y in range(1, 4):
            for x in range(1, 4):
                source.putpixel((x, y), (0, 0, 0, 149))

        result, record = AdjustmentService.apply_ink_preview(
            source,
            {"启用": True, "基准": 197.0},
        )

        self.assertEqual(AdjustmentService._glyph_ink_value(result), 197.0)
        self.assertAlmostEqual(record["增益"], 197.0 / 149.0, places=5)
        self.assertEqual(record["目标偏差"], 0.0)
        self.assertTrue(record["是否达标"])
        self.assertEqual(record["状态"], "墨色已达标")

    def test_ink_gain_keeps_low_alpha_background_outside_fixed_core(self) -> None:
        source = Image.new("RGBA", (20, 20), (0, 0, 0, 14))
        for y in range(6, 14):
            for x in range(8, 12):
                source.putpixel((x, y), (0, 0, 0, 80))

        result, record = AdjustmentService.apply_ink_preview(
            source,
            {"启用": True, "基准": 196.0},
        )

        output_alpha = np.array(result.getchannel("A"), dtype=np.uint8, copy=True)
        self.assertEqual(int(output_alpha[0, 0]), 14)
        self.assertEqual(int(output_alpha[10, 10]), 196)
        self.assertEqual(record["调整前墨色"], 80.0)
        self.assertEqual(record["调整后墨色"], 196.0)
        self.assertTrue(record["是否达标"])

    def test_batch_rejects_upstream_inverted_alpha_image_before_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph, variants = self._build_reviewed_library(Path(directory), (90,))
            service = AdjustmentService(glyph)
            detail = variants[0]
            preview_path = (
                Path(glyph.get_workflow_dirs()["优化预览"])
                / str(detail["中间文件"])
            )
            malformed = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
            for y in range(3, 37):
                for x in range(3, 37):
                    malformed.putpixel((x, y), (0, 0, 0, 14))
            for y in range(10, 30):
                for x in range(16, 24):
                    malformed.putpixel((x, y), (230, 230, 230, 255))
            malformed.save(preview_path)

            result = service.save_coordinated_variants(
                variants,
                {},
                {"启用": True, "基准": 196.0},
                {"目标占比": 0.72, "墨色基准": 196.0},
            )

            self.assertEqual(result["成功"], 0)
            self.assertEqual(result["失败"], 1)
            self.assertIn("前序图像异常", result["失败详情"][0][1])
            self.assertEqual(detail["状态"], config.STATUS_REVIEWED)
            self.assertFalse(
                list(Path(glyph.get_workflow_dirs()["成品"]).glob("*.png"))
            )

    def test_near_white_residual_does_not_replace_glyph_core_metric(self) -> None:
        source = Image.new("RGBA", (7, 7), (254, 254, 254, 255))
        for y in range(2, 5):
            for x in range(2, 5):
                source.putpixel((x, y), (0, 0, 0, 149))

        result, record = AdjustmentService.apply_ink_preview(
            source,
            {"启用": True, "基准": 197.0},
        )

        self.assertEqual(record["调整前墨色"], 149.0)
        self.assertEqual(record["调整后墨色"], 197.0)
        self.assertTrue(record["是否达标"])
        self.assertEqual(AdjustmentService._ink_bounding_box(result), (2, 2, 5, 5))
        self.assertLessEqual(result.getpixel((0, 0))[3], 2)

    def test_unreached_keep_mode_does_not_complete_until_manually_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph, variants = self._build_reviewed_library(Path(directory), (90,))
            service = AdjustmentService(glyph)
            baseline = service.analyze()

            service.save_coordinated_variants(
                variants,
                {},
                {"启用": True, "基准": 190.0, "模式": "保留本字"},
                baseline,
            )
            pending_record = variants[0]["整体协调参数"]["墨色协调"]
            pending_summary = glyph.get_coordination_summary()

            self.assertFalse(pending_record["是否达标"])
            self.assertEqual(pending_record["调整方式"], "保留本字")
            self.assertEqual(pending_record["状态"], "保留本字，待确认")
            self.assertFalse(pending_summary["墨色统一完成"])
            self.assertEqual(pending_summary["墨色统计"]["待确认"], 1)

            service.save_coordinated_variants(
                variants,
                {},
                {
                    "启用": True,
                    "基准": 190.0,
                    "逐字模式": {
                        str(variants[0]["变体ID"]): "人工例外",
                    },
                },
                baseline,
            )
            accepted_record = variants[0]["整体协调参数"]["墨色协调"]
            accepted_summary = glyph.get_coordination_summary()

            self.assertTrue(accepted_record["人工接受例外"])
            self.assertEqual(accepted_record["调整方式"], "人工例外")
            self.assertEqual(accepted_record["状态"], "人工接受例外")
            self.assertTrue(accepted_summary["墨色统一完成"])
            self.assertEqual(accepted_summary["墨色统计"]["人工例外"], 1)

    def test_page_save_only_marks_summary_complete_after_every_variant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph, variants = self._build_reviewed_library(Path(directory), (80, 150))
            service = AdjustmentService(glyph)
            baseline = service.analyze()
            ink_config = {"启用": True, "基准": baseline["墨色基准"]}

            first_result = service.save_coordinated_variants(
                [variants[0]],
                {},
                ink_config,
                baseline,
            )
            first_summary = glyph.get_coordination_summary()

            self.assertEqual(first_result["成功"], 1)
            self.assertFalse(first_summary["几何协调完成"])
            self.assertFalse(first_summary["墨色统一完成"])

            second_result = service.save_coordinated_variants(
                [variants[1]],
                {},
                ink_config,
                baseline,
            )
            final_summary = glyph.get_coordination_summary()

            self.assertEqual(second_result["成功"], 1)
            self.assertTrue(final_summary["几何协调完成"])
            self.assertTrue(final_summary["墨色统一完成"])
            for detail in variants:
                record = detail["整体协调参数"]["墨色协调"]
                self.assertTrue(record["启用"])
                self.assertEqual(record["基准"], baseline["墨色基准"])
                self.assertIn("调整前墨色", record)
                self.assertIn("调整后墨色", record)
            glyph.mark_manual_saved(str(variants[0]["变体ID"]), "人工修改.png", "new-md5")
            invalidated = glyph.get_coordination_summary()
            self.assertFalse(invalidated["几何协调完成"])
            self.assertFalse(invalidated["墨色统一完成"])

    def test_preview_and_saved_result_use_the_same_ink_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph, variants = self._build_reviewed_library(Path(directory), (105,))
            service = AdjustmentService(glyph)
            detail = variants[0]
            source_path = (
                Path(glyph.get_workflow_dirs()["优化预览"])
                / str(detail["中间文件"])
            )
            rgb_alpha_source = Image.new("RGBA", (6, 6), (255, 255, 255, 255))
            for y in range(2, 4):
                for x in range(2, 4):
                    rgb_alpha_source.putpixel((x, y), (60, 80, 100, 180))
            rgb_alpha_source.save(source_path)
            baseline = service.analyze()
            ink_config = {"启用": True, "基准": 190.0}
            adjustments = {"等比缩放": 1.2, "旋转": 13.0}

            preview_result = service.preview_coordinated(
                detail,
                adjustments,
                1.3,
                ink_config,
            )
            self.assertIsNotNone(preview_result)
            if preview_result is None:
                self.fail("整体协调预览不应为空")
            preview_image, preview_bounds = preview_result
            self.assertEqual(len(preview_result), 2)
            self.assertEqual(len(preview_bounds), 4)
            self.assertEqual(len(preview_result.control_polygon), 4)
            service.save_coordinated_variants(
                [detail],
                {str(detail["变体ID"]): adjustments},
                ink_config,
                baseline,
            )
            finished_path = Path(glyph.get_workflow_dirs()["成品"]) / detail["成品文件"]
            with Image.open(finished_path) as saved:
                saved_rgba = np.array(
                    saved.convert("RGBA"),
                    dtype=np.uint8,
                    copy=True,
                )
                saved_alpha = saved_rgba[..., 3]
            preview_alpha = np.array(
                preview_image.getchannel("A"),
                dtype=np.uint8,
                copy=True,
            )

            self.assertEqual(
                sorted(preview_alpha[preview_alpha > 0].tolist()),
                sorted(saved_alpha[saved_alpha > 0].tolist()),
            )
            self.assertTrue(np.all(saved_rgba[..., :3] == 0))
            record = detail["整体协调参数"]["墨色协调"]
            self.assertEqual(record["像素类型"], "RGB+Alpha混合墨色")
            self.assertTrue(record["是否达标"])
            self.assertTrue(record["保存后复测"])
            self.assertEqual(record["保存后墨色"], record["调整后墨色"])
            self.assertEqual(
                record["调整后墨色"],
                round(
                    float(
                        AdjustmentService._glyph_ink_value(
                            Image.fromarray(saved_rgba, "RGBA")
                        )
                    ),
                    2,
                ),
            )

    def test_saved_png_recheck_controls_follow_mode_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph, variants = self._build_reviewed_library(Path(directory), (149,))
            service = AdjustmentService(glyph)
            detail = variants[0]
            baseline = service.analyze()
            real_save = service._save_coordination_temp_png

            def save_with_changed_alpha(
                image: Image.Image,
                target_path: str,
                dpi: tuple[float, float],
            ) -> str:
                temporary_path = real_save(image, target_path, dpi)
                with Image.open(temporary_path) as reopened:
                    changed = reopened.convert("RGBA")
                alpha = np.array(
                    changed.getchannel("A"),
                    dtype=np.uint8,
                    copy=True,
                )
                alpha[alpha > 0] = 80
                changed.putalpha(Image.fromarray(alpha, "L"))
                changed.save(temporary_path, "PNG", dpi=dpi)
                return temporary_path

            with patch.object(
                service,
                "_save_coordination_temp_png",
                side_effect=save_with_changed_alpha,
            ):
                result = service.save_coordinated_variants(
                    variants,
                    {},
                    {"启用": True, "基准": 197.0},
                    baseline,
                )

            record = detail["整体协调参数"]["墨色协调"]
            self.assertEqual(result["失败"], 0)
            self.assertEqual(record["调整后墨色"], 197.0)
            self.assertEqual(record["保存后墨色"], 80.0)
            self.assertTrue(record["保存后复测"])
            self.assertFalse(record["是否达标"])
            self.assertEqual(record["状态"], "调整受限，待确认")
            self.assertEqual(record["目标偏差"], -117.0)
            self.assertFalse(glyph.get_coordination_summary()["墨色统一完成"])

    def test_unreadable_saved_png_fails_batch_before_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph, variants = self._build_reviewed_library(Path(directory), (149,))
            service = AdjustmentService(glyph)
            baseline = service.analyze()
            real_open = service._open_rgba

            def reject_temporary(path: str) -> Image.Image | None:
                if Path(path).name.startswith(".fonteditor_coordination_"):
                    return None
                return real_open(path)

            with patch.object(service, "_open_rgba", side_effect=reject_temporary):
                result = service.save_coordinated_variants(
                    variants,
                    {},
                    {"启用": True, "基准": 197.0},
                    baseline,
                )

            finished_dir = Path(glyph.get_workflow_dirs()["成品"])
            self.assertEqual(result["成功"], 0)
            self.assertEqual(result["失败"], 1)
            self.assertIn("无法重新解码", result["失败详情"][0][1])
            self.assertEqual(variants[0]["状态"], config.STATUS_REVIEWED)
            self.assertFalse(list(finished_dir.glob("*.png")))
            self.assertFalse(list(finished_dir.glob(".fonteditor_coordination_*")))

    def test_analyze_can_measure_formal_baseline_after_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph, variants = self._build_reviewed_library(Path(directory), (120, 180))
            adjustments = {
                str(variants[0]["变体ID"]): {"等比缩放": 1.6, "旋转": 13.0},
                str(variants[1]["变体ID"]): {"水平拉伸": 0.75, "旋转": -9.0},
            }

            baseline = AdjustmentService(glyph).analyze(
                adjustments_by_id=adjustments,
            )

            self.assertEqual(baseline["墨色统计阶段"], "几何变换后")
            self.assertEqual(baseline["墨色方法"], AdjustmentService.INK_METHOD)
            self.assertEqual(
                baseline["墨色方法版本"],
                AdjustmentService.INK_METHOD_VERSION,
            )
            self.assertEqual(baseline["墨色有效数"], 2)

    def test_full_save_can_lock_recomputed_post_geometry_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph, variants = self._build_reviewed_library(Path(directory), (80, 200))
            service = AdjustmentService(glyph)
            adjustments = {
                str(variants[0]["变体ID"]): {"等比缩放": 1.4, "旋转": 17.0},
                str(variants[1]["变体ID"]): {"水平拉伸": 0.8, "旋转": -11.0},
            }
            expected = service.analyze(adjustments_by_id=adjustments)

            result = service.save_coordinated_variants(
                variants,
                adjustments,
                {"启用": True, "基准": 100.0, "重算几何后基准": True},
                service.analyze(),
            )
            summary = glyph.get_coordination_summary()

            self.assertEqual(result["失败"], 0)
            self.assertEqual(summary["墨色基准"], expected["墨色基准"])
            self.assertEqual(summary["基准"]["墨色统计阶段"], "几何变换后")
            for detail in variants:
                record = detail["整体协调参数"]["墨色协调"]
                self.assertEqual(record["基准"], expected["墨色基准"])
                self.assertTrue(record["是否达标"])

    def test_resave_with_ink_disabled_uses_reviewed_source_not_old_finished_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph, variants = self._build_reviewed_library(Path(directory), (95,))
            service = AdjustmentService(glyph)
            detail = variants[0]
            baseline = service.analyze()
            service.save_coordinated_variants(
                [detail],
                {},
                {"启用": True, "基准": 200.0},
                baseline,
            )

            finished_path = Path(glyph.get_workflow_dirs()["成品"]) / detail["成品文件"]
            Image.new("RGBA", (10, 10), (0, 0, 0, 250)).save(finished_path)
            service.save_coordinated_variants(
                [detail],
                {},
                {"启用": False, "基准": baseline["墨色基准"]},
                baseline,
            )

            with Image.open(finished_path) as saved:
                alpha_values = np.array(
                    saved.convert("RGBA").getchannel("A"),
                    dtype=np.uint8,
                    copy=True,
                )
            nonzero = alpha_values[alpha_values > 0]
            self.assertTrue(nonzero.size)
            self.assertEqual(int(nonzero.max()), 95)
            record = detail["整体协调参数"]["墨色协调"]
            self.assertFalse(record["启用"])
            self.assertEqual(record["跳过原因"], "已关闭墨色统一")
            self.assertFalse(glyph.get_coordination_summary()["墨色统一完成"])

    def test_disabling_ink_requires_every_finished_variant_to_be_regenerated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph, variants = self._build_reviewed_library(Path(directory), (95, 175))
            service = AdjustmentService(glyph)
            baseline = service.analyze()
            enabled = {"启用": True, "基准": baseline["墨色基准"]}
            disabled = {"启用": False, "基准": baseline["墨色基准"]}

            service.save_coordinated_variants(variants, {}, enabled, baseline)
            self.assertTrue(glyph.get_coordination_summary()["几何协调完成"])

            service.save_coordinated_variants([variants[0]], {}, disabled, baseline)
            mixed_summary = glyph.get_coordination_summary()

            self.assertFalse(mixed_summary["墨色统一启用"])
            self.assertFalse(mixed_summary["几何协调完成"])
            self.assertFalse(
                variants[0]["整体协调参数"]["墨色协调"]["启用"]
            )
            self.assertTrue(
                variants[1]["整体协调参数"]["墨色协调"]["启用"]
            )

            service.save_coordinated_variants([variants[1]], {}, disabled, baseline)
            completed_summary = glyph.get_coordination_summary()

            self.assertFalse(completed_summary["墨色统一启用"])
            self.assertTrue(completed_summary["几何协调完成"])
            self.assertFalse(completed_summary["墨色统一完成"])
            self.assertTrue(
                all(
                    detail["整体协调参数"]["墨色协调"]["启用"] is False
                    for detail in variants
                )
            )

    def test_batch_preparation_failure_writes_no_partial_finished_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph, variants = self._build_reviewed_library(Path(directory), (90, 140))
            service = AdjustmentService(glyph)
            baseline = service.analyze()
            initial = service.save_coordinated_variants(
                variants,
                {},
                {"启用": False, "基准": baseline["墨色基准"]},
                baseline,
            )
            self.assertEqual(initial["成功"], 2)

            finished_dir = Path(glyph.get_workflow_dirs()["成品"])
            existing_files = {
                str(detail["变体ID"]): (
                    finished_dir / str(detail["成品文件"])
                ).read_bytes()
                for detail in variants
            }
            state_before = glyph.snapshot_state()
            source_to_remove = (
                Path(glyph.get_workflow_dirs()["优化预览"])
                / str(variants[1]["中间文件"])
            )
            source_to_remove.unlink()
            progress: list[tuple[str, int, int, int, str]] = []

            result = service.save_coordinated_variants(
                variants,
                {
                    str(variants[0]["变体ID"]): {"移动X": 3.0},
                    str(variants[1]["变体ID"]): {"移动X": -4.0},
                },
                {"启用": False, "基准": baseline["墨色基准"]},
                baseline,
                progress_callback=lambda *event: progress.append(event),
            )

            self.assertEqual(result["成功"], 0)
            self.assertEqual(result["失败"], 2)
            self.assertIn("找不到审核通过的文字图片", result["失败详情"][1][1])
            self.assertEqual(glyph.snapshot_state(), state_before)
            self.assertNotIn(100, [event[1] for event in progress])
            for detail in variants:
                path = finished_dir / str(detail["成品文件"])
                self.assertEqual(
                    path.read_bytes(),
                    existing_files[str(detail["变体ID"])],
                )
            self.assertFalse(list(finished_dir.glob(".fonteditor_coordination_*")))

    def test_batch_cancel_during_record_preparation_preserves_existing_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph, variants = self._build_reviewed_library(Path(directory), (90, 140))
            service = AdjustmentService(glyph)
            baseline = service.analyze()
            initial = service.save_coordinated_variants(
                variants,
                {},
                {"启用": False, "基准": baseline["墨色基准"]},
                baseline,
            )
            self.assertEqual(initial["成功"], len(variants))
            finished_dir = Path(glyph.get_workflow_dirs()["成品"])
            state_before = glyph.snapshot_state()
            files_before = {
                path.name: path.read_bytes()
                for path in finished_dir.glob("*.png")
            }
            cancel_requested = False

            def progress(stage: str, _percent: int, current: int, _total: int, _label: str) -> None:
                nonlocal cancel_requested
                if stage == "准备" and current == 1:
                    cancel_requested = True

            with self.assertRaisesRegex(CoordinationCancelled, "本批次未提交"):
                service.save_coordinated_variants(
                    variants,
                    {str(variants[0]["变体ID"]): {"移动X": 8.0}},
                    {"启用": False, "基准": baseline["墨色基准"]},
                    baseline,
                    progress_callback=progress,
                    cancel_check=lambda: cancel_requested,
                )

            self.assertEqual(glyph.snapshot_state(), state_before)
            self.assertEqual(
                {path.name: path.read_bytes() for path in finished_dir.glob("*.png")},
                files_before,
            )
            self.assertFalse(list(finished_dir.glob(".fonteditor_coordination_*")))

    def test_batch_cancel_during_render_removes_all_temporary_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph, variants = self._build_reviewed_library(Path(directory), (90, 140))
            service = AdjustmentService(glyph)
            baseline = service.analyze()
            state_before = glyph.snapshot_state()
            finished_dir = Path(glyph.get_workflow_dirs()["成品"])
            cancel_requested = False
            real_save_temp = service._save_coordination_temp_png

            def save_then_cancel(
                image: Image.Image,
                target_path: str,
                dpi: tuple[float, float],
            ) -> str:
                nonlocal cancel_requested
                temporary_path = real_save_temp(image, target_path, dpi)
                cancel_requested = True
                return temporary_path

            with (
                patch.object(
                    service,
                    "_save_coordination_temp_png",
                    side_effect=save_then_cancel,
                ),
                self.assertRaisesRegex(CoordinationCancelled, "本批次未提交"),
            ):
                service.save_coordinated_variants(
                    variants,
                    {},
                    {"启用": False, "基准": baseline["墨色基准"]},
                    baseline,
                    cancel_check=lambda: cancel_requested,
                )

            self.assertEqual(glyph.snapshot_state(), state_before)
            self.assertFalse(list(finished_dir.glob("*.png")))
            self.assertFalse(list(finished_dir.glob(".fonteditor_coordination_*")))

    def test_cancel_requested_during_state_snapshot_prevents_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph, variants = self._build_reviewed_library(Path(directory), (90, 140))
            service = AdjustmentService(glyph)
            baseline = service.analyze()
            state_before = glyph.snapshot_state()
            finished_dir = Path(glyph.get_workflow_dirs()["成品"])
            cancel_requested = False
            commit_gate_calls = 0
            real_snapshot = glyph.snapshot_variant_state

            def snapshot_then_cancel(variant_id: str) -> dict[str, object]:
                nonlocal cancel_requested
                snapshot = real_snapshot(variant_id)
                cancel_requested = True
                return snapshot

            def try_begin_commit() -> bool:
                nonlocal commit_gate_calls
                commit_gate_calls += 1
                return not cancel_requested

            with (
                patch.object(
                    glyph,
                    "snapshot_variant_state",
                    side_effect=snapshot_then_cancel,
                ),
                self.assertRaisesRegex(CoordinationCancelled, "本批次未提交"),
            ):
                service.save_coordinated_variants(
                    variants,
                    {},
                    {"启用": False, "基准": baseline["墨色基准"]},
                    baseline,
                    cancel_check=lambda: cancel_requested,
                    commit_gate=try_begin_commit,
                )

            self.assertEqual(glyph.snapshot_state(), state_before)
            self.assertEqual(commit_gate_calls, 0)
            self.assertFalse(list(finished_dir.glob("*.png")))
            self.assertFalse(list(finished_dir.glob(".fonteditor_coordination_*")))

    def test_state_snapshot_failure_removes_all_temporary_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph, variants = self._build_reviewed_library(Path(directory), (90, 140))
            service = AdjustmentService(glyph)
            baseline = service.analyze()
            finished_dir = Path(glyph.get_workflow_dirs()["成品"])

            with (
                patch.object(
                    glyph,
                    "snapshot_variant_state",
                    side_effect=RuntimeError("模拟状态快照失败"),
                ),
                self.assertRaisesRegex(RuntimeError, "模拟状态快照失败"),
            ):
                service.save_coordinated_variants(
                    variants,
                    {},
                    {"启用": False, "基准": baseline["墨色基准"]},
                    baseline,
                )

            self.assertFalse(list(finished_dir.glob("*.png")))
            self.assertFalse(list(finished_dir.glob(".fonteditor_coordination_*")))

    def test_cancel_after_commit_stage_begins_does_not_interrupt_transaction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph, variants = self._build_reviewed_library(Path(directory), (90, 140))
            service = AdjustmentService(glyph)
            baseline = service.analyze()
            cancel_requested = False
            commit_events: list[tuple[int, str]] = []

            def progress(stage: str, percent: int, _current: int, _total: int, label: str) -> None:
                nonlocal cancel_requested
                if stage == "提交":
                    cancel_requested = True
                    commit_events.append((percent, label))

            result = service.save_coordinated_variants(
                variants,
                {},
                {"启用": False, "基准": baseline["墨色基准"]},
                baseline,
                progress_callback=progress,
                cancel_check=lambda: cancel_requested,
            )

            self.assertEqual(result, {"成功": 2, "失败": 0, "失败详情": []})
            self.assertTrue(commit_events)
            self.assertEqual(commit_events[-1], (100, "批次提交完成"))
            self.assertTrue(
                all(detail["状态"] == config.STATUS_FINISHED for detail in variants)
            )
            finished_dir = Path(glyph.get_workflow_dirs()["成品"])
            self.assertEqual(len(list(finished_dir.glob("*.png"))), len(variants))
            self.assertFalse(list(finished_dir.glob(".fonteditor_coordination_*")))

    def test_coordination_transaction_snapshot_scales_with_saved_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph, variants = self._build_reviewed_library(Path(directory), (90, 140))
            for index in range(500):
                glyph.add_original(
                    chr(0x3400 + index),
                    f"扩展-{index:04d}.png",
                    f"扩展-{index:04d}.png",
                    f"extra-{index:04d}",
                )
            service = AdjustmentService(glyph)
            baseline = service.analyze()
            captured: list[dict[str, object]] = []
            real_snapshot = (
                adjustment_service_module.recovery_variant_batch_state_snapshot
            )

            def capture_snapshot(snapshots):
                payload = real_snapshot(snapshots)
                captured.append(payload)
                return payload

            with patch.object(
                adjustment_service_module,
                "recovery_variant_batch_state_snapshot",
                side_effect=capture_snapshot,
            ):
                result = service.save_coordinated_variants(
                    variants,
                    {},
                    {"启用": False, "基准": baseline["墨色基准"]},
                    baseline,
                )

            self.assertEqual(result, {"成功": 2, "失败": 0, "失败详情": []})
            self.assertEqual(len(captured), 2)
            self.assertTrue(
                all(payload["快照类型"] == "字形批次" for payload in captured)
            )
            self.assertTrue(
                all(len(payload["字形状态"]) == 2 for payload in captured)
            )
            batch_size = len(json.dumps(captured[0], ensure_ascii=False))
            full_size = len(json.dumps(glyph.snapshot_state(), ensure_ascii=False))
            self.assertLess(batch_size * 10, full_size)

    def test_empty_identity_fields_reject_batch_before_rendering(self) -> None:
        invalid_cases = (
            ("空变体ID", "变体ID", "", "缺少变体ID"),
            ("空白变体ID", "变体ID", "   ", "缺少变体ID"),
            ("空原始文件名", "原始文件", "", "缺少原始文件名"),
            ("空原始文件值", "原始文件", None, "缺少原始文件名"),
        )
        for case_name, field_name, invalid_value, expected_error in invalid_cases:
            with self.subTest(case_name=case_name), tempfile.TemporaryDirectory() as directory:
                glyph, variants = self._build_reviewed_library(Path(directory), (90, 140))
                service = AdjustmentService(glyph)
                variants[1][field_name] = invalid_value
                state_before = glyph.snapshot_state()
                progress: list[tuple[str, int, int, int, str]] = []

                result = service.save_coordinated_variants(
                    variants,
                    {},
                    {"启用": False, "基准": 120.0},
                    {"目标占比": 0.72, "墨色基准": 120.0},
                    progress_callback=lambda *event: progress.append(event),
                )

                finished_dir = Path(glyph.get_workflow_dirs()["成品"])
                self.assertEqual(result["成功"], 0)
                self.assertEqual(result["失败"], len(variants))
                self.assertTrue(
                    any(
                        expected_error in reason
                        for _variant_id, reason in result["失败详情"]
                    )
                )
                self.assertEqual(glyph.snapshot_state(), state_before)
                self.assertTrue(
                    all(detail["状态"] == config.STATUS_REVIEWED for detail in variants)
                )
                self.assertFalse(list(finished_dir.glob("*.png")))
                self.assertFalse(list(finished_dir.glob(".fonteditor_coordination_*")))
                self.assertFalse(any(event[0] == "渲染" for event in progress))
                self.assertNotIn(100, [event[1] for event in progress])

    def test_blank_reviewed_image_rejects_entire_batch_without_finished_files(self) -> None:
        blank_images = {
            "全透明": Image.new("RGBA", (6, 6), (0, 0, 0, 0)),
            "纯白": Image.new("RGBA", (6, 6), (255, 255, 255, 255)),
        }
        for blank_name, blank_image in blank_images.items():
            with self.subTest(blank_name=blank_name), tempfile.TemporaryDirectory() as directory:
                glyph, variants = self._build_reviewed_library(Path(directory), (90, 140))
                service = AdjustmentService(glyph)
                invalid_detail = variants[1]
                invalid_path = (
                    Path(glyph.get_workflow_dirs()["优化预览"])
                    / str(invalid_detail["中间文件"])
                )
                blank_image.save(invalid_path)
                state_before = glyph.snapshot_state()
                progress: list[tuple[str, int, int, int, str]] = []

                result = service.save_coordinated_variants(
                    variants,
                    {},
                    {"启用": False, "基准": 120.0},
                    {"目标占比": 0.72, "墨色基准": 120.0},
                    progress_callback=lambda *event: progress.append(event),
                )

                failures = dict(result["失败详情"])
                finished_dir = Path(glyph.get_workflow_dirs()["成品"])
                self.assertEqual(result["成功"], 0)
                self.assertEqual(result["失败"], len(variants))
                self.assertIn(
                    "没有有效文字前景",
                    failures[str(invalid_detail["变体ID"])],
                )
                self.assertEqual(glyph.snapshot_state(), state_before)
                self.assertTrue(
                    all(detail["状态"] == config.STATUS_REVIEWED for detail in variants)
                )
                self.assertFalse(list(finished_dir.glob("*.png")))
                self.assertFalse(list(finished_dir.glob(".fonteditor_coordination_*")))
                self.assertNotIn(100, [event[1] for event in progress])

    def test_batch_progress_reports_prepare_render_and_commit_monotonically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph, variants = self._build_reviewed_library(Path(directory), (90, 140))
            service = AdjustmentService(glyph)
            baseline = service.analyze()
            progress: list[tuple[str, int, int, int, str]] = []

            result = service.save_coordinated_variants(
                variants,
                {},
                {"启用": False, "基准": baseline["墨色基准"]},
                baseline,
                progress_callback=lambda *event: progress.append(event),
            )

            self.assertEqual(result, {"成功": 2, "失败": 0, "失败详情": []})
            self.assertTrue(progress)
            self.assertEqual(progress[0], ("准备", 0, 0, 2, "正在核对批次"))
            self.assertEqual(progress[-1], ("提交", 100, 2, 2, "批次提交完成"))
            self.assertEqual(
                list(dict.fromkeys(event[0] for event in progress)),
                ["准备", "渲染", "提交"],
            )
            percentages = [event[1] for event in progress]
            self.assertEqual(percentages, sorted(percentages))
            self.assertTrue(
                all(event[3] == len(variants) for event in progress)
            )
            rendered_labels = {
                event[4] for event in progress if event[0] == "渲染"
            }
            for detail in variants:
                self.assertTrue(
                    any(str(detail["归属字"]) in label for label in rendered_labels)
                )

    def test_unchanged_finished_output_is_reused_without_rendering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph, variants = self._build_reviewed_library(Path(directory), (100,))
            service = AdjustmentService(glyph)
            baseline = service.analyze()
            ink_config = {"启用": False, "基准": baseline["墨色基准"]}
            first = service.save_coordinated_variants(
                variants,
                {},
                ink_config,
                baseline,
            )
            finished = Path(glyph.get_workflow_dirs()["成品"]) / str(
                variants[0]["成品文件"]
            )
            before = finished.read_bytes()

            with patch.object(
                service,
                "_render_coordination_item",
                wraps=service._render_coordination_item,
            ) as render:
                second = service.save_coordinated_variants(
                    variants,
                    {},
                    ink_config,
                    baseline,
                )

            self.assertEqual(first["成功"], 1)
            self.assertEqual(second["成功"], 1)
            render.assert_not_called()
            self.assertEqual(finished.read_bytes(), before)

    def test_coordination_rendering_uses_bounded_worker_threads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph, variants = self._build_reviewed_library(Path(directory), (90, 140))
            service = AdjustmentService(glyph)
            baseline = service.analyze()
            barrier = threading.Barrier(2)
            thread_names: set[str] = set()
            original = service._render_coordination_item

            def render(*args: object, **kwargs: object) -> dict[str, object]:
                thread_names.add(threading.current_thread().name)
                barrier.wait(timeout=5)
                return original(*args, **kwargs)

            with patch.object(service, "_render_coordination_item", side_effect=render):
                result = service.save_coordinated_variants(
                    variants,
                    {},
                    {"启用": False, "基准": baseline["墨色基准"]},
                    baseline,
                )

            self.assertEqual(result["成功"], 2)
            self.assertEqual(len(thread_names), 2)
            self.assertTrue(
                all(name.startswith("整体协调渲染") for name in thread_names)
            )

    def test_progress_callback_failure_does_not_break_batch_transaction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph, variants = self._build_reviewed_library(Path(directory), (110,))
            service = AdjustmentService(glyph)
            baseline = service.analyze()
            callback_calls = 0

            def broken_progress(*_event: object) -> None:
                nonlocal callback_calls
                callback_calls += 1
                raise RuntimeError("模拟进度接收端已关闭")

            result = service.save_coordinated_variants(
                variants,
                {},
                {"启用": False, "基准": baseline["墨色基准"]},
                baseline,
                progress_callback=broken_progress,
            )

            self.assertGreater(callback_calls, 0)
            self.assertEqual(result, {"成功": 1, "失败": 0, "失败详情": []})
            detail = glyph.get_variant(str(variants[0]["变体ID"]))
            self.assertEqual(detail["状态"], config.STATUS_FINISHED)

    def test_json_commit_failure_rolls_back_entire_finished_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph, variants = self._build_reviewed_library(Path(directory), (85, 155))
            service = AdjustmentService(glyph)
            baseline = service.analyze()
            initial = service.save_coordinated_variants(
                variants,
                {},
                {"启用": False, "基准": baseline["墨色基准"]},
                baseline,
            )
            self.assertEqual(initial["成功"], 2)

            finished_dir = Path(glyph.get_workflow_dirs()["成品"])
            state_before = glyph.snapshot_state()
            database_before = LibraryDatabase.open(glyph.ziku_dir).load_data()
            existing_files = {
                str(detail["变体ID"]): (
                    finished_dir / str(detail["成品文件"])
                ).read_bytes()
                for detail in variants
            }

            with patch.object(glyph, "save", side_effect=OSError("模拟 JSON 提交失败")):
                with self.assertRaisesRegex(OSError, "模拟 JSON 提交失败"):
                    service.save_coordinated_variants(
                        variants,
                        {
                            str(variants[0]["变体ID"]): {"移动X": 5.0},
                            str(variants[1]["变体ID"]): {"旋转": 9.0},
                        },
                        {"启用": True, "基准": baseline["墨色基准"]},
                        baseline,
                    )

            self.assertEqual(glyph.snapshot_state(), state_before)
            self.assertEqual(
                LibraryDatabase.open(glyph.ziku_dir).load_data(),
                database_before,
            )
            for variant_id, file_bytes in existing_files.items():
                detail = glyph.get_variant(variant_id)
                path = finished_dir / str(detail["成品文件"])
                self.assertEqual(path.read_bytes(), file_bytes)
            self.assertFalse(list(finished_dir.glob(".fonteditor_coordination_*")))

    def test_legacy_json_receives_complete_coordination_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = {
                "数据版本": 3,
                "库名": "旧库",
                "元数据": {"DPI": 300, "画布宽": 20, "画布高": 20},
                "会话": {},
                "字形组索引": {"旧": ["old-1"]},
                "变体详情": {
                    "old-1": {
                        "变体ID": "old-1",
                        "归属字": "旧",
                        "状态": config.STATUS_REVIEWED,
                    }
                },
                "整体协调": {"基准": {}},
            }
            (root / "旧库.json").write_text(
                json.dumps(data, ensure_ascii=False),
                encoding="utf-8",
            )

            glyph = GlyphService("旧库", str(root))
            summary = glyph.get_coordination_summary()

            self.assertTrue(summary["墨色统一启用"])
            self.assertFalse(summary["几何协调完成"])
            self.assertFalse(summary["墨色统一完成"])
            self.assertEqual(glyph.get_variant("old-1")["整体协调参数"], {})

    @staticmethod
    def _build_reviewed_library(
        root: Path,
        ink_values: tuple[int, ...],
    ) -> tuple[GlyphService, list[dict[str, object]]]:
        glyph = GlyphService("测试", str(root))
        glyph.ensure_dirs()
        glyph.init_metadata(dpi=300, canvas_w=12, canvas_h=12)
        preview_dir = Path(glyph.get_workflow_dirs()["优化预览"])
        variants: list[dict[str, object]] = []
        for index, ink in enumerate(ink_values, 1):
            char = chr(ord("甲") + index - 1)
            filename = f"{char}-0001.png"
            variant_id = glyph.add_original(char, filename, filename, f"md5-{index}")
            image = Image.new("RGBA", (6, 6), (0, 0, 0, 0))
            for y in range(2, 4):
                for x in range(2, 4):
                    image.putpixel((x, y), (0, 0, 0, ink))
            image.save(preview_dir / filename)
            detail = glyph.get_variant(variant_id)
            detail["中间文件"] = filename
            detail["状态"] = config.STATUS_REVIEWED
            variants.append(detail)
        glyph.save()
        return glyph, variants

    @staticmethod
    def _build_asymmetric_library(
        root: Path,
    ) -> tuple[GlyphService, dict[str, object], Path]:
        glyph = GlyphService("几何核对", str(root))
        glyph.ensure_dirs()
        glyph.init_metadata(dpi=300, canvas_w=128, canvas_h=128)
        filename = "测-0001.png"
        variant_id = glyph.add_original("测", filename, filename, "geometry-md5")
        source = Image.new("RGBA", (96, 84), (0, 0, 0, 0))
        for y in range(10, 70):
            for x in range(16, 34):
                source.putpixel((x, y), (18, 35, 52, 255))
        for y in range(22, 43):
            for x in range(33, 72):
                source.putpixel((x, y), (18, 35, 52, 255))
        for y in range(52, 65):
            for x in range(30, 64):
                source.putpixel((x, y), (18, 35, 52, 230))
        for y in range(9, 21):
            for x in range(66, 79):
                if (x - 72) ** 2 + (y - 15) ** 2 <= 36:
                    source.putpixel((x, y), (72, 25, 25, 210))
        source_path = Path(glyph.get_workflow_dirs()["优化预览"]) / filename
        source.save(source_path)
        detail = glyph.get_variant(variant_id)
        detail["中间文件"] = filename
        detail["状态"] = config.STATUS_REVIEWED
        glyph.save()
        return glyph, detail, source_path

    @staticmethod
    def _qimage_rgba(image: QImage) -> np.ndarray:
        converted = image.convertToFormat(QImage.Format.Format_RGBA8888)
        height = converted.height()
        width = converted.width()
        stride = converted.bytesPerLine()
        pixels = np.frombuffer(
            converted.constBits(),
            dtype=np.uint8,
            count=stride * height,
        ).reshape(height, stride)
        return pixels[:, : width * 4].reshape(height, width, 4).copy()


class ConsistencyPageTests(unittest.TestCase):
    """验证页面默认模式、固定基准和既有几何参数恢复。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_page_defaults_to_ink_coordination_and_restores_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph, variants = AdjustmentServiceTests._build_reviewed_library(
                Path(directory),
                (120,),
            )
            detail = variants[0]
            detail["状态"] = config.STATUS_FINISHED
            detail["成品文件"] = str(detail["原始文件"])
            detail["整体协调参数"] = {
                "整体变换": {
                    "移动X": 3.0,
                    "移动Y": -2.0,
                    "缩放X": 1.1,
                    "缩放Y": 0.9,
                    "旋转": 2.0,
                    "斜切X": 0.0,
                    "斜切Y": 0.0,
                    "扭曲": [0.0] * 8,
                }
            }
            glyph.save()

            page = ConsistencyPage(glyph, lambda: None)
            variant_id = str(detail["变体ID"])

            self.assertTrue(page._ink_check.isChecked())
            self.assertIn("本次进入后保持固定", page._ink_baseline_label.text())
            self.assertEqual(page._adjustments[variant_id]["移动X"], 3.0)
            self.assertEqual(page._adjustments[variant_id]["移动Y"], -2.0)
            self.assertIn("待协调 1", page._summary_label.text())
            page.close()
            page.deleteLater()


if __name__ == "__main__":
    unittest.main()
