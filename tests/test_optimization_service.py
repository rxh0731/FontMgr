"""自动优化候选分类与透明图像处理回归测试。"""

from __future__ import annotations

import hashlib
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from PIL import Image

import config
from core.optimizer import OptimizationCancelled
from core.source_classification import (
    SOURCE_TYPE_TRANSPARENT,
    SOURCE_TYPE_UNPROCESSED,
    SOURCE_TYPE_WHITE_CLEANED,
    classify_source,
)
from services.optimization_service import (
    CANDIDATE_TYPE_ALPHA_DENOISED,
    CANDIDATE_TYPE_DIRECT,
    CANDIDATE_TYPE_OPTIMIZED,
    CANDIDATE_TYPE_TRANSPARENT,
    OptimizationService,
)
from services.batch_persistence import (
    BatchJournalUncertainError,
    acquire_batch_library_lock,
)
from services.background_model_service import (
    BACKGROUND_MODEL_REGISTRY,
    MODEL_OUTPUT_PROBABILITY_MASK,
    BackgroundModelContext,
    BackgroundModelDescriptor,
    BackgroundModelInferenceResult,
)
from services.file_transaction_recovery import FileTransaction
from services.workflow_status_service import (
    MARKER_FILE_ERROR,
    STAGE_COMPLETED,
    STAGE_PENDING_COORDINATION,
    STAGE_PENDING_OPTIMIZATION,
    STAGE_PENDING_REVIEW,
)


class OptimizationServiceTests(unittest.TestCase):
    """验证不依赖真实字库写入的候选生成语义。"""

    def setUp(self) -> None:
        self.service = OptimizationService(None)  # type: ignore[arg-type]

    def test_batch_items_keep_pending_record_without_original_filename(self) -> None:
        glyph = MagicMock()
        glyph.get_workflow_dirs.return_value = {
            "原图": "D:/临时字库/01_原图",
            "优化预览": "D:/临时字库/04_自动优化稿",
        }
        glyph.get_all_variants.return_value = [
            {
                "变体ID": "variant-damaged",
                "归属字": "缺",
                "原始文件": "",
                "状态": config.STATUS_PENDING_OPTIMIZATION,
            },
            {
                "变体ID": "variant-reviewed",
                "归属字": "已",
                "原始文件": "已.png",
                "状态": config.STATUS_REVIEWED,
            },
        ]

        pending, skipped = OptimizationService(glyph).list_batch_items()

        self.assertEqual(skipped, 1)
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["键"], "variant-damaged")
        self.assertEqual(pending[0]["原始文件名"], "")
        self.assertEqual(pending[0]["原始路径"], "")
        self.assertIn(MARKER_FILE_ERROR, pending[0]["提示"])

    def test_list_items_uses_unified_stages_and_rejects_unsafe_source_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "统一阶段字库"
            source_dir = root / config.DIR_ORIGINAL_FILES
            preview_dir = root / config.DIR_INTERMEDIATE_FILES
            review_dir = root / config.DIR_REVIEWED_FILES
            finished_dir = root / config.DIR_FINISHED_FILES
            for stage_dir in (source_dir, preview_dir, review_dir, finished_dir):
                stage_dir.mkdir(parents=True, exist_ok=True)
            (source_dir / "待优化.png").write_bytes(b"source")
            (preview_dir / "待审核.png").write_bytes(b"preview")
            (review_dir / "待协调.png").write_bytes(b"review")
            (finished_dir / "已完成.png").write_bytes(b"finished")
            (root / "越界.png").write_bytes(b"outside")

            details = {
                "优": {
                    "变体ID": "variant-opt",
                    "归属字": "优",
                    "原始文件": "待优化.png",
                    "状态": config.STATUS_PENDING_OPTIMIZATION,
                },
                "审": {
                    "变体ID": "variant-review",
                    "归属字": "审",
                    "原始文件": "",
                    "中间文件": "待审核.png",
                    "状态": config.STATUS_PENDING_MANUAL_REVIEW,
                },
                "协": {
                    "变体ID": "variant-coordinate",
                    "归属字": "协",
                    "原始文件": "",
                    "审核文件": "待协调.png",
                    "状态": config.STATUS_REVIEWED,
                },
                "完": {
                    "变体ID": "variant-complete",
                    "归属字": "完",
                    "原始文件": "",
                    "审核文件": "待协调.png",
                    "成品文件": "已完成.png",
                    "状态": config.STATUS_FINISHED,
                    "整体协调参数": {
                        "墨色协调": {
                            "启用": False,
                            "保存后复测": True,
                            "保存后墨色": 128.0,
                        }
                    },
                },
                "险": {
                    "变体ID": "variant-unsafe",
                    "归属字": "险",
                    "原始文件": "../越界.png",
                    "状态": config.STATUS_PENDING_OPTIMIZATION,
                },
            }
            glyph = MagicMock()
            glyph.get_workflow_dirs.return_value = {
                "原图": str(source_dir),
                "优化预览": str(preview_dir),
                "手工审核": str(review_dir),
                "成品": str(finished_dir),
            }
            glyph.get_coordination_summary.return_value = {"墨色统一启用": False}
            glyph.get_all_chars.return_value = list(details)
            glyph.get_char_variants.side_effect = lambda char: [details[char]]
            glyph.get_all_variants.return_value = list(details.values())

            service = OptimizationService(glyph)
            items = service.list_items()
            pending, skipped = service.list_batch_items()

        by_id = {item["键"]: item for item in items}
        self.assertEqual(by_id["variant-opt"]["显示状态"], STAGE_PENDING_OPTIMIZATION)
        self.assertEqual(by_id["variant-review"]["显示状态"], STAGE_PENDING_REVIEW)
        self.assertEqual(
            by_id["variant-coordinate"]["显示状态"],
            STAGE_PENDING_COORDINATION,
        )
        self.assertEqual(by_id["variant-complete"]["显示状态"], STAGE_COMPLETED)
        self.assertEqual(by_id["variant-unsafe"]["原始路径"], "")
        self.assertIn(MARKER_FILE_ERROR, by_id["variant-unsafe"]["提示"])
        pending_by_id = {item["键"]: item for item in pending}
        self.assertEqual(skipped, 3)
        self.assertEqual(pending_by_id["variant-unsafe"]["原始路径"], "")
        self.assertIn(MARKER_FILE_ERROR, pending_by_id["variant-unsafe"]["提示"])

    def test_load_source_preserves_standard_alpha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "透明.png"
            source = Image.new("RGBA", (2, 2), (18, 52, 86, 255))
            source.putpixel((1, 1), (120, 90, 60, 0))
            source.save(path)

            rgba, gray, transparency_source = self.service._load_source(str(path))

        self.assertEqual(transparency_source, "标准Alpha")
        self.assertEqual(rgba.getpixel((0, 0)), (18, 52, 86, 255))
        self.assertEqual(rgba.getpixel((1, 1)), (120, 90, 60, 0))
        self.assertEqual(gray.shape, (2, 2))
        self.assertEqual(int(gray[1, 1]), 255)

    def test_source_classification_distinguishes_three_input_types(self) -> None:
        transparent = Image.new("RGBA", (100, 100), (255, 255, 255, 0))
        for y in range(30, 70):
            for x in range(42, 58):
                transparent.putpixel((x, y), (24, 24, 24, 255))
        transparent_gray = np.full((100, 100), 255, dtype=np.float32)
        transparent_gray[30:70, 42:58] = 24

        cleaned_gray = np.full((100, 100), 255, dtype=np.float32)
        cleaned_gray[30:70, 42:58] = 24
        cleaned = Image.fromarray(cleaned_gray.astype(np.uint8), "L").convert("RGBA")

        gradient = np.linspace(205, 255, 100, dtype=np.float32)[:, None]
        screenshot_gray = np.repeat(gradient, 100, axis=1)
        screenshot_gray[30:70, 42:58] = 24
        screenshot = Image.fromarray(screenshot_gray.astype(np.uint8), "L").convert("RGBA")

        transparent_result = classify_source(
            transparent,
            transparent_gray,
            "标准Alpha",
        )
        cleaned_result = classify_source(cleaned, cleaned_gray)
        screenshot_result = classify_source(screenshot, screenshot_gray)

        self.assertEqual(transparent_result.source_type, SOURCE_TYPE_TRANSPARENT)
        self.assertEqual(cleaned_result.source_type, SOURCE_TYPE_WHITE_CLEANED)
        self.assertEqual(screenshot_result.source_type, SOURCE_TYPE_UNPROCESSED)
        self.assertIn("边缘近白占比", cleaned_result.metrics)
        self.assertIn("背景灰度标准差", screenshot_result.metrics)
        self.assertIn("光照变化", screenshot_result.metrics)
        self.assertIn("污染指标", screenshot_result.metrics)

    def test_white_background_with_many_dark_specks_is_unprocessed(self) -> None:
        gray = np.full((120, 120), 255, dtype=np.float32)
        gray[38:82, 52:68] = 24
        placed = 0
        for y in range(8, 112, 9):
            for x in range(8, 112, 11):
                if 34 <= y <= 86 and 46 <= x <= 74:
                    continue
                gray[y, x] = 40
                placed += 1
        self.assertGreater(placed, 40)
        image = Image.fromarray(gray.astype(np.uint8), "L").convert("RGBA")

        result = classify_source(image, gray)

        self.assertEqual(result.source_type, SOURCE_TYPE_UNPROCESSED)
        self.assertGreater(int(result.metrics["散点数量"]), 8)
        self.assertIn("孤立散点污染", "".join(result.reasons))

    def test_white_background_with_large_external_block_is_unprocessed(self) -> None:
        gray = np.full((160, 160), 255, dtype=np.float32)
        gray[48:112, 70:92] = 24
        gray[8:34, 10:38] = 36
        image = Image.fromarray(gray.astype(np.uint8), "L").convert("RGBA")

        result = classify_source(image, gray)

        self.assertEqual(result.source_type, SOURCE_TYPE_UNPROCESSED)
        self.assertGreaterEqual(int(result.metrics["疑似大块外部污染数量"]), 1)
        self.assertIn("大块外部污染", "".join(result.reasons))

    def test_white_cleaned_glyph_with_legitimate_separated_dots_stays_on_fast_path(self) -> None:
        gray = np.full((160, 160), 255, dtype=np.float32)
        gray[34:112, 72:88] = 24
        gray[58:76, 46:114] = 24
        for left in (50, 68, 86, 104):
            gray[138:143, left:left + 5] = 24
        image = Image.fromarray(gray.astype(np.uint8), "L").convert("RGBA")

        result = classify_source(image, gray)

        self.assertEqual(result.source_type, SOURCE_TYPE_WHITE_CLEANED)
        self.assertEqual(int(result.metrics["疑似大块外部污染数量"]), 0)

    def test_tight_full_glyph_uses_highlight_background_instead_of_edge_average(self) -> None:
        gray = np.full((160, 160), 255, dtype=np.float32)
        gray[:, 72:88] = 24
        for top in (20, 40, 60, 80, 100, 120, 140):
            gray[top:top + 8, 20:140] = 24
        image = Image.fromarray(gray.astype(np.uint8), "L").convert("RGBA")

        result = classify_source(image, gray)

        self.assertLess(float(result.metrics["边缘平均灰度"]), 244.0)
        self.assertLess(float(result.metrics["全图近白占比"]), 0.70)
        self.assertGreater(float(result.metrics["边缘高亮平均灰度"]), 250.0)
        self.assertEqual(result.source_type, SOURCE_TYPE_WHITE_CLEANED)

    def test_small_or_internal_alpha_defects_are_not_valid_transparent_background(self) -> None:
        gray = np.full((100, 100), 255, dtype=np.float32)
        gray[32:68, 44:56] = 24
        for name, alpha_slice in (
            ("少量边缘瑕疵", (slice(0, 1), slice(0, 5))),
            ("内部透明块", (slice(8, 28), slice(8, 28))),
        ):
            with self.subTest(name=name):
                rgba = Image.fromarray(gray.astype(np.uint8), "L").convert("RGBA")
                alpha = np.full((100, 100), 255, dtype=np.uint8)
                alpha[alpha_slice] = 0
                rgba.putalpha(Image.fromarray(alpha, "L"))

                result = classify_source(rgba, gray, "标准Alpha")

                self.assertNotEqual(result.source_type, SOURCE_TYPE_TRANSPARENT)
                self.assertIn("边缘连通透明像素占比", result.metrics)

    def test_nearly_opaque_alpha_is_not_effective_transparent_background(self) -> None:
        gray = np.full((100, 100), 255, dtype=np.float32)
        gray[32:68, 44:56] = 24
        for background_alpha in (249, 240):
            with self.subTest(background_alpha=background_alpha):
                rgba = Image.new(
                    "RGBA",
                    (100, 100),
                    (255, 255, 255, background_alpha),
                )
                try:
                    for y in range(32, 68):
                        for x in range(44, 56):
                            rgba.putpixel((x, y), (24, 24, 24, 255))
                    result = classify_source(rgba, gray, "标准Alpha")
                finally:
                    rgba.close()

                self.assertNotEqual(result.source_type, SOURCE_TYPE_TRANSPARENT)
                self.assertEqual(float(result.metrics["全透明像素占比"]), 0.0)

    def test_single_edge_transparent_band_is_not_valid_background(self) -> None:
        gray = np.full((100, 100), 255, dtype=np.float32)
        gray[32:68, 44:56] = 24
        rgba = Image.fromarray(gray.astype(np.uint8), "L").convert("RGBA")
        alpha = np.full((100, 100), 255, dtype=np.uint8)
        alpha[:3, :] = 0
        alpha_image = Image.fromarray(alpha, "L")
        try:
            rgba.putalpha(alpha_image)
        finally:
            alpha_image.close()
        try:
            result = classify_source(rgba, gray, "标准Alpha")
        finally:
            rgba.close()

        self.assertEqual(int(result.metrics["边缘连通透明触边数"]), 3)
        self.assertEqual(float(result.metrics["透明像素边缘连通率"]), 1.0)
        self.assertNotEqual(result.source_type, SOURCE_TYPE_TRANSPARENT)

    def test_disconnected_corner_alpha_regions_do_not_collectively_touch_four_edges(self) -> None:
        gray = np.full((100, 100), 255, dtype=np.float32)
        gray[32:68, 44:56] = 24
        rgba = Image.fromarray(gray.astype(np.uint8), "L").convert("RGBA")
        alpha = np.full((100, 100), 255, dtype=np.uint8)
        alpha[:20, :20] = 0
        alpha[:20, -20:] = 0
        alpha[-20:, :20] = 0
        alpha[-20:, -20:] = 0
        alpha_image = Image.fromarray(alpha, "L")
        try:
            rgba.putalpha(alpha_image)
        finally:
            alpha_image.close()
        try:
            result = classify_source(rgba, gray, "标准Alpha")
        finally:
            rgba.close()

        self.assertEqual(int(result.metrics["边缘连通透明触边数"]), 2)
        self.assertNotEqual(result.source_type, SOURCE_TYPE_TRANSPARENT)

    def test_tiny_alpha_image_keeps_single_pixel_transparency_exception(self) -> None:
        rgba = Image.new("RGBA", (2, 2), (24, 24, 24, 255))
        rgba.putpixel((1, 1), (255, 255, 255, 0))
        gray = np.full((2, 2), 24, dtype=np.float32)
        gray[1, 1] = 255
        try:
            result = classify_source(rgba, gray, "标准Alpha")
        finally:
            rgba.close()

        self.assertEqual(result.source_type, SOURCE_TYPE_TRANSPARENT)

    def test_large_inner_counter_does_not_invalidate_edge_transparent_background(self) -> None:
        rgba = Image.new("RGBA", (100, 100), (255, 255, 255, 0))
        for y in range(5, 95):
            for x in range(5, 95):
                if x < 15 or x >= 85 or y < 15 or y >= 85:
                    rgba.putpixel((x, y), (24, 24, 24, 255))
        gray = np.full((100, 100), 255, dtype=np.float32)
        gray[5:95, 5:15] = 24
        gray[5:95, 85:95] = 24
        gray[5:15, 15:85] = 24
        gray[85:95, 15:85] = 24

        result = classify_source(rgba, gray, "标准Alpha")

        self.assertLess(float(result.metrics["透明像素边缘连通率"]), 0.80)
        self.assertGreater(float(result.metrics["边缘连通透明像素占比"]), 0.12)
        self.assertEqual(result.source_type, SOURCE_TYPE_TRANSPARENT)
        rgba.close()

    def test_alpha_denoise_accepts_small_medium_alpha_speckles(self) -> None:
        alpha = np.zeros((100, 100), dtype=np.uint8)
        alpha[25:75, 45:55] = 255
        alpha[8:10, 10:12] = 60
        alpha[84, 82] = 42

        cleaned, mask, cleanup = self.service._lightly_denoise_alpha(alpha)

        self.assertTrue(cleanup["有变化"])
        self.assertTrue(cleanup["批量安全"])
        self.assertEqual(cleanup["需保护连通域数"], 0)
        self.assertEqual(int(cleaned[8:10, 10:12].max()), 0)
        self.assertEqual(int(cleaned[84, 82]), 0)
        self.assertTrue(np.all(cleaned[25:75, 45:55] == 255))
        self.assertTrue(np.all(mask[25:75, 45:55] == 1))

    def test_alpha_denoise_removes_repeated_high_alpha_micro_speckles(self) -> None:
        alpha = np.zeros((100, 100), dtype=np.uint8)
        alpha[25:75, 35:65] = 255
        for index in range(8):
            y = 5 + index * 11
            alpha[y:y + 2, 82:84] = 255

        cleaned, _mask, cleanup = self.service._lightly_denoise_alpha(alpha)

        self.assertTrue(cleanup["有变化"])
        self.assertTrue(cleanup["批量安全"])
        self.assertTrue(cleanup["清理充分"])
        self.assertEqual(cleanup["高Alpha微小域清理数"], 8)
        self.assertEqual(int(np.count_nonzero(cleaned[:, 82:84])), 0)

    def test_alpha_denoise_preserves_single_high_alpha_point_stroke(self) -> None:
        alpha = np.zeros((100, 100), dtype=np.uint8)
        alpha[25:75, 45:55] = 255
        alpha[10:14, 70:74] = 255

        cleaned, _mask, cleanup = self.service._lightly_denoise_alpha(alpha)

        self.assertFalse(cleanup["有变化"])
        self.assertTrue(cleanup["批量安全"])
        self.assertTrue(cleanup["清理充分"])
        self.assertTrue(np.all(cleaned[10:14, 70:74] == 255))

    def test_alpha_denoise_reports_high_alpha_residue_connected_by_weak_bridges(
        self,
    ) -> None:
        alpha = np.zeros((100, 100), dtype=np.uint8)
        alpha[20:80, 35:65] = 255
        for index in range(8):
            y = 8 + index * 11
            alpha[y:y + 2, 68:70] = 255
            alpha[y, 64:69] = 16

        _cleaned, _mask, cleanup = self.service._lightly_denoise_alpha(alpha)

        self.assertFalse(cleanup["有变化"])
        self.assertTrue(cleanup["批量安全"])
        self.assertFalse(cleanup["清理充分"])
        self.assertGreaterEqual(cleanup["剩余高Alpha微小域数"], 6)
        self.assertIn("转入完整寻优", cleanup["剩余污染说明"])

    def test_alpha_denoise_protects_independent_point_stroke(self) -> None:
        alpha = np.zeros((100, 100), dtype=np.uint8)
        alpha[25:75, 45:55] = 255
        alpha[10:14, 70:74] = 60

        _cleaned, _mask, cleanup = self.service._lightly_denoise_alpha(alpha)

        self.assertTrue(cleanup["有变化"])
        self.assertFalse(cleanup["批量安全"])
        self.assertEqual(cleanup["需保护连通域数"], 1)
        self.assertIn("独立点画或笔画片段尺度", cleanup["人工核对原因"])

    def test_alpha_denoise_protects_elongated_stroke_fragment(self) -> None:
        alpha = np.zeros((100, 100), dtype=np.uint8)
        alpha[25:75, 45:55] = 255
        alpha[10:12, 65:79] = 52

        _cleaned, _mask, cleanup = self.service._lightly_denoise_alpha(alpha)

        self.assertFalse(cleanup["批量安全"])
        self.assertEqual(cleanup["需保护连通域数"], 1)

    def test_alpha_denoise_protects_large_faint_region(self) -> None:
        alpha = np.zeros((100, 100), dtype=np.uint8)
        alpha[25:75, 45:55] = 255
        alpha[10:12, 65:85] = 10

        _cleaned, _mask, cleanup = self.service._lightly_denoise_alpha(alpha)

        self.assertFalse(cleanup["批量安全"])
        self.assertEqual(cleanup["需保护连通域数"], 1)

    def test_alpha_denoise_rejects_excessive_total_residue_mass(self) -> None:
        alpha = np.zeros((100, 100), dtype=np.uint8)
        alpha[25:75, 45:55] = 255
        for y in range(2, 98, 5):
            for x in range(2, 38, 5):
                alpha[y, x] = 60

        _cleaned, _mask, cleanup = self.service._lightly_denoise_alpha(alpha)

        self.assertFalse(cleanup["批量安全"])
        self.assertIn("透明残留总量", cleanup["人工核对原因"])

    def test_luo_0004_batch_routes_residual_alpha_noise_to_full_optimizer(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "字库"
            / "小品-庆兰"
            / "01_源图"
            / "羅-0004.png"
        )
        if not path.is_file():
            self.skipTest("只读真实样本不存在")

        candidate = self.service.generate_batch_candidate(
            {"原始路径": str(path), "归属字": "羅"}
        )
        try:
            self.assertEqual(candidate["处理类型"], CANDIDATE_TYPE_OPTIMIZED)
            result_alpha = np.array(candidate["图像"], dtype=np.uint8)[..., 3]
            count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
                (result_alpha >= 64).astype(np.uint8),
                connectivity=8,
                ltype=cv2.CV_32S,
            )
            areas = stats[1:, cv2.CC_STAT_AREA]
            self.assertEqual(count - 1, 6)
            self.assertEqual(int(np.count_nonzero(areas <= 12)), 0)
        finally:
            candidate["图像"].close()

    def test_alpha_connectivity_uses_bounded_analysis_image(self) -> None:
        rgba = Image.new("RGBA", (1600, 1200), (255, 255, 255, 0))
        for y in range(360, 840):
            for x in range(680, 920):
                rgba.putpixel((x, y), (24, 24, 24, 255))
        gray = np.full((1200, 1600), 255, dtype=np.float32)
        gray[360:840, 680:920] = 24

        result = classify_source(rgba, gray, "标准Alpha")

        self.assertEqual(result.source_type, SOURCE_TYPE_TRANSPARENT)
        self.assertLessEqual(int(result.metrics["Alpha分析宽度"]), 512)
        self.assertLessEqual(int(result.metrics["Alpha分析高度"]), 512)

    def test_generate_candidates_keeps_direct_candidate_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "透明.png"
            source = Image.new("RGBA", (2, 2), (0, 0, 0, 255))
            source.putpixel((1, 1), (32, 64, 96, 0))
            source.save(path)
            algorithm_results = [
                self._algorithm_result("原图保护", [[1, 1], [1, 0]], protect_original=True),
                self._algorithm_result("寻优结果", [[1, 0], [0, 0]]),
            ]

            with (
                patch(
                    "services.optimization_service.generate_candidate_results",
                    return_value=algorithm_results,
                ),
                patch("services.optimization_service.write_log"),
            ):
                candidates = self.service.generate_candidates(
                    {"原始路径": str(path), "归属字": "测"}, limit=8
                )

        direct = candidates[0]
        self.assertEqual(direct["处理类型"], CANDIDATE_TYPE_DIRECT)
        self.assertEqual(direct["方案"]["处理类型"], CANDIDATE_TYPE_DIRECT)
        self.assertTrue(direct["保留原图"])
        self.assertEqual(direct["图像"].getpixel((1, 1)), (32, 64, 96, 0))
        self.assertEqual(candidates[1]["处理类型"], CANDIDATE_TYPE_OPTIMIZED)
        self.assertTrue(candidates[1]["图像指纹"])
        self.assertEqual(direct["原图分类"]["类型"], SOURCE_TYPE_TRANSPARENT)
        self.assertEqual(
            candidates[1]["方案"]["原图分类"]["类型"],
            SOURCE_TYPE_TRANSPARENT,
        )

    def test_standard_alpha_direct_mask_ignores_nearly_transparent_background_residue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "透明残留.png"
            source = Image.new("RGBA", (40, 40), (0, 0, 0, 1))
            for y in range(10, 30):
                for x in range(16, 24):
                    source.putpixel((x, y), (0, 0, 0, 255))
            source.save(path)
            mask = np.zeros((40, 40), dtype=np.uint8)
            mask[10:30, 16:24] = 1

            with (
                patch(
                    "services.optimization_service.generate_candidate_results",
                    return_value=[self._algorithm_result("残留寻优", mask.tolist())],
                ),
                patch("services.optimization_service.write_log"),
            ):
                candidates = self.service.generate_candidates(
                    {"原始路径": str(path), "归属字": "残"}
                )

        direct = candidates[0]
        self.assertEqual(direct["处理类型"], CANDIDATE_TYPE_DIRECT)
        self.assertEqual(direct["图像"].getpixel((0, 0)), (0, 0, 0, 1))
        self.assertEqual(int(direct["清洁掩码"][0, 0]), 0)
        self.assertEqual(int(direct["清洁掩码"][20, 20]), 1)

    def test_low_alpha_white_haze_does_not_invert_dark_transparent_glyph(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "低透明白雾.png"
            source = Image.new("RGBA", (40, 40), (255, 255, 255, 0))
            for y in range(4, 36):
                for x in range(4, 36):
                    source.putpixel((x, y), (255, 255, 255, 14))
            for y in range(10, 30):
                for x in range(16, 24):
                    source.putpixel((x, y), (24, 24, 24, 255))
            source.save(path)
            algorithm_mask = np.zeros((40, 40), dtype=np.uint8)
            algorithm_mask[10:30, 16:24] = 1

            with (
                patch(
                    "services.optimization_service.generate_candidate_results",
                    return_value=[
                        self._algorithm_result("白雾寻优", algorithm_mask.tolist())
                    ],
                ),
                patch("services.optimization_service.write_log"),
            ):
                candidates = self.service.generate_candidates(
                    {"原始路径": str(path), "归属字": "雾"}
                )

        direct = candidates[0]
        self.assertEqual(direct["处理类型"], CANDIDATE_TYPE_DIRECT)
        self.assertNotIn("自动校正", direct["方案"])
        self.assertEqual(direct["图像"].getpixel((20, 20)), (24, 24, 24, 255))
        self.assertEqual(direct["图像"].getpixel((6, 6)), (255, 255, 255, 14))
        self.assertEqual(int(direct["清洁掩码"][6, 6]), 0)
        self.assertEqual(int(direct["清洁掩码"][20, 20]), 1)

    def test_generate_candidates_keeps_distinct_types_with_identical_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "透明.png"
            source = Image.new("RGBA", (2, 2), (0, 0, 0, 255))
            source.putpixel((1, 1), (0, 0, 0, 0))
            source.save(path)
            algorithm_results = [
                self._algorithm_result("原图保护", [[1, 1], [1, 0]], protect_original=True),
                self._algorithm_result("寻优结果", [[1, 1], [1, 0]]),
            ]

            with (
                patch(
                    "services.optimization_service.generate_candidate_results",
                    return_value=algorithm_results,
                ),
                patch("services.optimization_service.write_log"),
            ):
                candidates = self.service.generate_candidates(
                    {"原始路径": str(path), "归属字": "测"}, limit=8
                )

        self.assertEqual(candidates[0]["图像"].tobytes(), candidates[1]["图像"].tobytes())
        self.assertEqual(
            [candidate["处理类型"] for candidate in candidates],
            [CANDIDATE_TYPE_DIRECT, CANDIDATE_TYPE_OPTIMIZED],
        )

    def test_generate_candidates_classifies_opaque_source_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "普通.png"
            source = Image.new("L", (20, 20), 255)
            for y in range(6, 14):
                for x in range(8, 12):
                    source.putpixel((x, y), 32)
            source.save(path)
            mask = np.zeros((20, 20), dtype=np.uint8)
            mask[6:14, 8:12] = 1
            algorithm_results = [
                self._algorithm_result("原图保护", mask.tolist(), protect_original=True),
                self._algorithm_result("寻优结果", mask.tolist()),
            ]

            with (
                patch(
                    "services.optimization_service.generate_candidate_results",
                    return_value=algorithm_results,
                ),
                patch("services.optimization_service.write_log"),
            ):
                candidates = self.service.generate_candidates(
                    {"原始路径": str(path), "归属字": "测"}, limit=8
                )

        self.assertEqual(
            [candidate["处理类型"] for candidate in candidates],
            [CANDIDATE_TYPE_TRANSPARENT, CANDIDATE_TYPE_OPTIMIZED],
        )
        for candidate in candidates:
            self.assertEqual(candidate["方案"]["处理类型"], candidate["处理类型"])

    def test_unprocessed_source_returns_only_score_sorted_optimized_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "截图.png"
            gradient = np.linspace(205, 255, 60, dtype=np.uint8)[:, None]
            source_arr = np.repeat(gradient, 80, axis=1)
            source_arr[18:44, 32:48] = 24
            Image.fromarray(source_arr, "L").save(path)
            narrow = np.zeros(source_arr.shape, dtype=np.uint8)
            narrow[18:44, 34:46] = 1
            wide = np.zeros(source_arr.shape, dtype=np.uint8)
            wide[18:44, 32:48] = 1
            original = self._algorithm_result("原图保护", wide.tolist(), protect_original=True)
            lower = self._algorithm_result("较低分", narrow.tolist())
            higher = self._algorithm_result("较高分", wide.tolist())
            original["得分"] = 99.0
            lower["得分"] = 72.0
            higher["得分"] = 91.0

            with (
                patch(
                    "services.optimization_service.generate_candidate_results",
                    return_value=[lower, original, higher],
                ),
                patch("services.optimization_service.write_log"),
            ):
                candidates = self.service.generate_candidates(
                    {"原始路径": str(path), "归属字": "截"},
                    limit=4,
                )

        self.assertTrue(candidates)
        self.assertTrue(
            all(value["处理类型"] == CANDIDATE_TYPE_OPTIMIZED for value in candidates)
        )
        self.assertEqual([value["得分"] for value in candidates], [91.0, 72.0])
        self.assertEqual(candidates[0]["原图分类"]["类型"], SOURCE_TYPE_UNPROCESSED)

    def test_generate_candidates_rejects_baseline_only_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "透明.png"
            source = Image.new("RGBA", (20, 20), (255, 255, 255, 0))
            for y in range(6, 14):
                for x in range(8, 12):
                    source.putpixel((x, y), (0, 0, 0, 255))
            source.save(path)
            mask = np.zeros((20, 20), dtype=np.uint8)
            mask[6:14, 8:12] = 1
            only_original = self._algorithm_result(
                "原图保护",
                mask.tolist(),
                protect_original=True,
            )

            with (
                patch(
                    "services.optimization_service.generate_candidate_results",
                    return_value=[only_original],
                ),
                patch("services.optimization_service.write_log"),
            ):
                with self.assertRaisesRegex(ValueError, "有效文字前景.*寻优候选"):
                    self.service.generate_candidates(
                        {"原始路径": str(path), "归属字": "透"}
                    )

    def test_generate_candidates_preserves_structure_risk_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "截图.png"
            source_arr = np.full((24, 24), 245, dtype=np.uint8)
            source_arr[6:18, 9:15] = 24
            Image.fromarray(source_arr, "L").save(path)
            mask = np.zeros((24, 24), dtype=np.uint8)
            mask[6:18, 9:15] = 1
            result = self._algorithm_result("风险寻优", mask.tolist())
            result["结构复核"] = {
                "状态": "需人工核对",
                "阶段": "原尺寸复核",
                "原因": "有意义孔洞仅保留0.0%",
                "风险等级": 1,
            }

            with (
                patch(
                    "services.optimization_service.generate_candidate_results",
                    return_value=[result],
                ),
                patch("services.optimization_service.write_log"),
            ):
                candidates = self.service.generate_candidates(
                    {"原始路径": str(path), "归属字": "险"}
                )

        optimized = next(
            candidate
            for candidate in candidates
            if candidate["处理类型"] == CANDIDATE_TYPE_OPTIMIZED
        )
        self.assertTrue(self.service.is_candidate_valid(optimized))
        self.assertTrue(self.service.requires_structure_review(optimized))
        self.assertEqual(optimized["结构复核"], optimized["方案"]["结构复核"])
        self.assertIn("得分不代表结构安全", optimized["方案"]["评分方式"]["说明"])

    def test_structure_risk_sort_prefers_lower_risk_before_score(self) -> None:
        low_risk = {
            "方案名": "覆盖完整",
            "得分": 70.0,
            "结构复核": {"状态": "需人工核对", "风险等级": 1},
        }
        high_risk = {
            "方案名": "覆盖受损",
            "得分": 99.0,
            "结构复核": {"状态": "需人工核对", "风险等级": 2},
        }
        safe = {"方案名": "安全结果", "得分": 60.0}

        ordered = sorted(
            [high_risk, low_risk, safe],
            key=self.service._optimized_result_sort_key,
        )

        self.assertEqual(
            [candidate["方案名"] for candidate in ordered],
            ["安全结果", "覆盖完整", "覆盖受损"],
        )

    def test_generate_candidates_auto_corrects_inverted_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "反相.png"
            source = Image.new("L", (20, 20), 0)
            for y in range(6, 14):
                for x in range(6, 14):
                    source.putpixel((x, y), 255)
            source.save(path)
            mask = np.zeros((20, 20), dtype=np.uint8)
            mask[6:14, 6:14] = 1

            def build_results(gray: np.ndarray, **_kwargs) -> list[dict[str, object]]:
                self.assertEqual(int(gray[0, 0]), 255)
                self.assertEqual(int(gray[10, 10]), 0)
                return [self._algorithm_result("反相寻优", mask.tolist())]

            with (
                patch(
                    "services.optimization_service.generate_candidate_results",
                    side_effect=build_results,
                ),
                patch("services.optimization_service.write_log"),
            ):
                candidates = self.service.generate_candidates(
                    {"原始路径": str(path), "归属字": "反"}, limit=8
                )

        candidate = candidates[0]
        self.assertEqual(candidate["图像"].getpixel((0, 0))[3], 0)
        self.assertEqual(candidate["图像"].getpixel((10, 10))[3], 255)
        self.assertTrue(candidate["方案"]["自动校正"]["反相"])
        self.assertEqual(int(candidate["灰度母版"][0, 0]), 255)
        self.assertEqual(int(candidate["灰度母版"][10, 10]), 0)

    def test_transparent_white_glyph_direct_candidate_is_corrected_and_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "透明白字.png"
            source = Image.new("RGBA", (40, 40), (255, 255, 255, 0))
            for y in range(10, 30):
                for x in range(16, 24):
                    source.putpixel((x, y), (255, 255, 255, 255))
            source.save(path)
            mask = np.zeros((40, 40), dtype=np.uint8)
            mask[10:30, 16:24] = 1

            with (
                patch(
                    "services.optimization_service.generate_candidate_results",
                    return_value=[self._algorithm_result("白字寻优", mask.tolist())],
                ),
                patch("services.optimization_service.write_log"),
            ):
                candidates = self.service.generate_candidates(
                    {"原始路径": str(path), "归属字": "白"}
                )

        direct = candidates[0]
        self.assertEqual(direct["处理类型"], CANDIDATE_TYPE_DIRECT)
        self.assertTrue(self.service.is_candidate_valid(direct))
        self.assertEqual(direct["图像"].getpixel((20, 20)), (0, 0, 0, 255))
        self.assertEqual(direct["图像"].getpixel((0, 0))[3], 0)
        self.assertTrue(direct["方案"]["自动校正"]["反相"])
        self.assertEqual(int(direct["灰度母版"][20, 20]), 0)

    def test_model_inference_is_cached_and_records_engine_metadata(self) -> None:
        class ProbabilityAdapter:
            descriptor = BackgroundModelDescriptor(
                engine_id="test-probability-model",
                display_name="测试去背景模型",
                version="1.2",
                output_type=MODEL_OUTPUT_PROBABILITY_MASK,
                installed=True,
                model_fingerprint="model-sha256",
            )

            def __init__(self) -> None:
                self.call_count = 0

            def infer(self, source, _context):
                self.call_count += 1
                probability = np.zeros(source.shape, dtype=np.float32)
                probability[1:3, 1:3] = 1.0
                return BackgroundModelInferenceResult(
                    MODEL_OUTPUT_PROBABILITY_MASK,
                    probability,
                )

        adapter = ProbabilityAdapter()
        BACKGROUND_MODEL_REGISTRY.register(adapter, replace=True)
        context = BACKGROUND_MODEL_REGISTRY.create_context(adapter.descriptor.engine_id)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "模型输入.png"
            source = Image.new("L", (4, 4), 255)
            source.putpixel((1, 1), 32)
            source.save(path)
            mask = np.zeros((4, 4), dtype=np.uint8)
            mask[1:3, 1:3] = 1

            def build_results(working_gray, **kwargs):
                self.assertEqual(int(working_gray[1, 1]), 0)
                self.assertIn("reference_gray_arr", kwargs)
                return [self._algorithm_result("模型结果", mask.tolist())]

            with (
                patch(
                    "services.optimization_service.generate_candidate_results",
                    side_effect=build_results,
                ),
                patch("services.optimization_service.write_log"),
            ):
                item = {
                    "原始路径": str(path),
                    "原始MD5": "source-sha256",
                    "归属字": "模",
                }
                first = self.service.generate_candidates(item, engine_context=context)
                second = self.service.generate_candidates(item, engine_context=context)

        self.assertEqual(adapter.call_count, 1)
        first_optimized = next(
            candidate for candidate in first
            if candidate["处理类型"] == CANDIDATE_TYPE_OPTIMIZED
        )
        second_optimized = next(
            candidate for candidate in second
            if candidate["处理类型"] == CANDIDATE_TYPE_OPTIMIZED
        )
        self.assertEqual(
            first_optimized["方案"]["处理引擎"]["标识"],
            adapter.descriptor.engine_id,
        )
        self.assertEqual(
            first_optimized["方案"]["处理引擎"]["模型指纹"],
            "model-sha256",
        )
        self.assertEqual(
            first_optimized["灰度母版"].tolist(),
            second_optimized["灰度母版"].tolist(),
        )

    def test_exploration_restores_stored_model_context_and_rejects_version_change(self) -> None:
        class ContextAdapter:
            descriptor = BackgroundModelDescriptor(
                engine_id="test-context-model",
                display_name="上下文测试模型",
                version="1.0",
                output_type=MODEL_OUTPUT_PROBABILITY_MASK,
                installed=True,
                model_fingerprint="context-model-sha256",
            )

            def infer(self, source, _context):
                return BackgroundModelInferenceResult(
                    MODEL_OUTPUT_PROBABILITY_MASK,
                    np.zeros(source.shape, dtype=np.float32),
                )

        adapter = ContextAdapter()
        BACKGROUND_MODEL_REGISTRY.register(adapter, replace=True)
        stored_context = BACKGROUND_MODEL_REGISTRY.create_context(
            adapter.descriptor.engine_id,
            {"阈值": 0.5},
        )
        changed_context = BackgroundModelContext(
            adapter.descriptor,
            {"阈值": 0.8},
        )

        restored = self.service._resolve_exploration_context(
            stored_context.to_metadata(),
            changed_context,
        )

        self.assertEqual(restored.configuration_hash, stored_context.configuration_hash)
        self.assertEqual(dict(restored.configuration), {"阈值": 0.5})

        changed_metadata = stored_context.to_metadata()
        changed_metadata["版本"] = "2.0"
        with self.assertRaisesRegex(RuntimeError, "版本或配置已经变化"):
            self.service._resolve_exploration_context(changed_metadata, changed_context)

    def test_photoshop_metadata_without_decoded_alpha_is_not_direct_candidate(self) -> None:
        descriptor = BackgroundModelDescriptor(
            engine_id="test-photoshop-model",
            display_name="Photoshop测试模型",
            version="1.0",
            output_type=MODEL_OUTPUT_PROBABILITY_MASK,
            installed=True,
            model_fingerprint="photoshop-model-sha256",
        )
        context = BackgroundModelContext(descriptor)
        rgba = Image.new("RGBA", (2, 2), (0, 0, 0, 255))
        gray = np.array([[0, 255], [255, 255]], dtype=np.float32)
        model_mask = [[1, 1], [1, 1]]
        model_results = [self._algorithm_result("模型原始结果", model_mask, protect_original=True)]

        with (
            patch.object(
                self.service,
                "_load_source",
                return_value=(rgba, gray, "Photoshop图层"),
            ),
            patch.object(
                self.service,
                "_normalize_source_polarity",
                return_value=(gray, False),
            ),
            patch.object(
                self.service,
                "_prepare_engine_input",
                return_value=(gray, "model-result-sha256"),
            ),
            patch(
                "services.optimization_service.generate_candidate_results",
                return_value=model_results,
            ) as generate,
            patch("services.optimization_service.write_log"),
        ):
            candidates = self.service.generate_candidates(
                {"原始路径": "不读取的测试路径", "归属字": "模"},
                engine_context=context,
            )

        self.assertEqual(generate.call_count, 1)
        self.assertEqual(
            [candidate["处理类型"] for candidate in candidates],
            [CANDIDATE_TYPE_OPTIMIZED],
        )
        self.assertEqual(
            candidates[0]["原图分类"]["指标"]["Photoshop透明元数据未解码"],
            1,
        )

    def test_algorithm_work_ink_normalization_does_not_replace_gray_master(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "墨色.png"
            source = Image.new("L", (3, 3), 255)
            source.putpixel((1, 1), 96)
            source.save(path)
            result = self._algorithm_result("工作归一", [[0, 0, 0], [0, 1, 0], [0, 0, 0]])
            result["方案"] = {
                "预处理": {"墨色归一": True, "墨色基准": 60},
                "L3": {"算法": "Otsu", "参数": {"偏移": 0}},
            }

            with (
                patch(
                    "services.optimization_service.generate_candidate_results",
                    return_value=[result],
                ),
                patch("services.optimization_service.write_log"),
            ):
                candidates = self.service.generate_candidates(
                    {"原始路径": str(path), "归属字": "墨"}
                )

        candidate = next(
            value for value in candidates
            if value["处理类型"] == CANDIDATE_TYPE_OPTIMIZED
        )
        self.assertEqual(int(candidate["灰度母版"][1, 1]), 96)
        self.assertIn("算法工作归一", candidate["方案"]["墨色归一用途"])

    def test_batch_transparent_fast_path_skips_model_and_optimizer(self) -> None:
        descriptor = BackgroundModelDescriptor(
            engine_id="unused-batch-model",
            display_name="不应执行的模型",
            version="1.0",
            output_type=MODEL_OUTPUT_PROBABILITY_MASK,
            installed=True,
            model_fingerprint="unused-model-sha256",
        )
        context = BackgroundModelContext(descriptor)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "大图透明.png"
            source = Image.new("RGBA", (1000, 1200), (255, 255, 255, 0))
            for y in range(360, 840):
                for x in range(420, 580):
                    source.putpixel((x, y), (18, 18, 18, 255))
            source.save(path)

            with (
                patch(
                    "services.optimization_service.generate_candidate_results"
                ) as generate,
                patch.object(BACKGROUND_MODEL_REGISTRY, "infer") as infer,
                patch.object(
                    self.service,
                    "_score_candidate_bounded",
                ) as full_score,
                patch("services.optimization_service.write_log"),
            ):
                cancel_check = MagicMock(return_value=False)
                candidate = self.service.generate_batch_candidate(
                    {"原始路径": str(path), "归属字": "大"},
                    engine_context=context,
                    cancel_check=cancel_check,
                )

        generate.assert_not_called()
        infer.assert_not_called()
        full_score.assert_not_called()
        self.assertGreaterEqual(cancel_check.call_count, 4)
        self.assertEqual(candidate["处理类型"], CANDIDATE_TYPE_DIRECT)
        scoring_method = candidate["方案"]["评分方式"]
        self.assertEqual(scoring_method["模式"], "批量快速分类评分")
        self.assertEqual(scoring_method["原图分类"], SOURCE_TYPE_TRANSPARENT)
        self.assertEqual(
            scoring_method["分类置信度"],
            round(candidate["原图分类"]["置信度"], 4),
        )
        self.assertAlmostEqual(
            candidate["得分"],
            scoring_method["分类置信度"] * 100.0,
            places=2,
        )
        self.assertTrue(np.isfinite(candidate["得分"]))
        self.assertGreaterEqual(candidate["得分"], 0.0)
        self.assertLessEqual(candidate["得分"], 100.0)

    def test_batch_transparent_residual_pollution_falls_back_to_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "残留污染.png"
            rgba = np.full((100, 100, 4), 255, dtype=np.uint8)
            rgba[..., :3] = 255
            rgba[..., 3] = 0
            rgba[20:80, 35:65, :3] = 24
            rgba[20:80, 35:65, 3] = 255
            for index in range(8):
                y = 8 + index * 11
                rgba[y:y + 2, 68:70, :3] = 24
                rgba[y:y + 2, 68:70, 3] = 255
                rgba[y, 64:69, :3] = 24
                rgba[y, 64:69, 3] = 16
            Image.fromarray(rgba, "RGBA").save(path)
            mask = np.zeros((100, 100), dtype=np.uint8)
            mask[20:80, 35:65] = 1
            optimized = self._algorithm_result("残留污染完整寻优", mask.tolist())

            with (
                patch(
                    "services.optimization_service.generate_candidate_results",
                    return_value=[optimized],
                ) as generate,
                patch("services.optimization_service.write_log"),
            ):
                candidate = self.service.generate_batch_candidate(
                    {"原始路径": str(path), "归属字": "测"}
                )

        try:
            generate.assert_called_once()
            self.assertEqual(candidate["处理类型"], CANDIDATE_TYPE_OPTIMIZED)
            self.assertEqual(candidate["方案名"], "残留污染完整寻优")
        finally:
            candidate["图像"].close()

    def test_manual_baseline_keeps_full_structure_score(self) -> None:
        source = Image.new("RGBA", (20, 20), (255, 255, 255, 0))
        for y in range(5, 15):
            for x in range(8, 12):
                source.putpixel((x, y), (24, 24, 24, 255))
        gray = np.full((20, 20), 255, dtype=np.float32)
        gray[5:15, 8:12] = 24
        classification = classify_source(source, gray, "标准Alpha")

        with patch.object(
            self.service,
            "_score_candidate_bounded",
            return_value=(73.5, {"模式": "完整结构评分测试"}),
        ) as full_score:
            candidate = self.service._build_baseline_candidate(
                source,
                gray,
                "标准Alpha",
                classification,
                False,
                None,
            )

        source.close()
        full_score.assert_called_once()
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["得分"], 73.5)
        self.assertEqual(candidate["方案"]["评分方式"]["模式"], "完整结构评分测试")

    def test_batch_white_cleaned_fast_path_uses_otsu_for_near_white_background(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "近白底.png"
            source_arr = np.fromfunction(
                lambda y, x: 248 + ((x + y) % 5),
                (80, 80),
                dtype=int,
            ).astype(np.uint8)
            source_arr[24:56, 34:46] = 36
            Image.fromarray(source_arr, "L").save(path)

            with (
                patch(
                    "services.optimization_service.generate_candidate_results"
                ) as generate,
                patch.object(
                    self.service,
                    "_score_candidate_bounded",
                ) as full_score,
                patch("services.optimization_service.write_log"),
            ):
                candidate = self.service.generate_batch_candidate(
                    {"原始路径": str(path), "归属字": "白"}
                )

        generate.assert_not_called()
        full_score.assert_not_called()
        self.assertEqual(candidate["处理类型"], CANDIDATE_TYPE_TRANSPARENT)
        self.assertEqual(candidate["原图分类"]["类型"], SOURCE_TYPE_WHITE_CLEANED)
        scoring_method = candidate["方案"]["评分方式"]
        self.assertEqual(scoring_method["模式"], "批量快速分类评分")
        self.assertEqual(scoring_method["原图分类"], SOURCE_TYPE_WHITE_CLEANED)
        self.assertAlmostEqual(
            candidate["得分"],
            float(scoring_method["分类置信度"]) * 100.0,
            places=2,
        )
        self.assertTrue(np.isfinite(candidate["得分"]))
        self.assertEqual(int(candidate["清洁掩码"][0, 0]), 0)
        self.assertEqual(candidate["图像"].getpixel((0, 0))[3], 0)
        self.assertEqual(int(candidate["清洁掩码"][40, 40]), 1)

    def test_batch_unprocessed_source_selects_highest_valid_optimized_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "截图.png"
            gradient = np.linspace(205, 255, 40, dtype=np.uint8)[:, None]
            source_arr = np.repeat(gradient, 50, axis=1)
            source_arr[12:30, 21:29] = 24
            Image.fromarray(source_arr, "L").save(path)
            mask = np.ones((2, 2), dtype=np.uint8)

            def candidate(score: float) -> dict[str, object]:
                return {
                    "方案名": f"寻优{score}",
                    "方案": {},
                    "得分": score,
                    "图像": Image.new("RGBA", (2, 2), (0, 0, 0, 255)),
                    "灰度母版": np.zeros((2, 2), dtype=np.uint8),
                    "清洁掩码": mask.copy(),
                    "处理类型": CANDIDATE_TYPE_OPTIMIZED,
                }

            lower = candidate(72.0)
            higher = candidate(93.0)
            item = {"原始路径": str(path), "归属字": "截"}
            engine_context = MagicMock(spec=BackgroundModelContext)
            cancel_check = MagicMock(return_value=False)
            with (
                patch.object(
                    self.service,
                    "generate_candidates",
                    return_value=[lower, higher],
                ) as generate,
                patch("services.optimization_service.write_log"),
            ):
                selected = self.service.generate_batch_candidate(
                    item,
                    engine_context=engine_context,
                    cancel_check=cancel_check,
                )

        self.assertIs(selected, higher)
        generate.assert_called_once()
        self.assertEqual(generate.call_args.args, (item,))
        self.assertEqual(generate.call_args.kwargs["limit"], 1)
        self.assertIs(
            generate.call_args.kwargs["engine_context"],
            engine_context,
        )
        self.assertIs(generate.call_args.kwargs["cancel_check"], cancel_check)
        self.assertTrue(generate.call_args.kwargs["_batch_final_only"])
        prepared = generate.call_args.kwargs["_prepared_source"]
        self.assertEqual(prepared[3].source_type, SOURCE_TYPE_UNPROCESSED)

    def test_batch_final_only_skips_unused_baseline_and_pixel_fingerprint(self) -> None:
        source = Image.new("RGBA", (20, 20), (255, 255, 255, 0))
        for y in range(6, 14):
            for x in range(8, 12):
                source.putpixel((x, y), (24, 24, 24, 255))
        gray = np.full((20, 20), 255, dtype=np.float32)
        gray[6:14, 8:12] = 24
        classification = classify_source(source, gray, "标准Alpha")
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[6:14, 8:12] = 1
        algorithm_result = self._algorithm_result("批量最终寻优", mask.tolist())
        optimized_image = source.copy()

        try:
            with (
                patch.object(
                    self.service,
                    "_build_baseline_candidate",
                    side_effect=AssertionError("批量最终模式不应构造不会采用的基准候选"),
                ),
                patch(
                    "services.optimization_service.generate_candidate_results",
                    return_value=[algorithm_result],
                ) as generate,
                patch.object(
                    self.service,
                    "_gray_to_transparent_image",
                    return_value=optimized_image,
                ),
                patch.object(
                    optimized_image,
                    "tobytes",
                    side_effect=AssertionError("单个最终候选不应计算去重指纹"),
                ),
                patch("services.optimization_service.write_log"),
            ):
                candidates = self.service.generate_candidates(
                    {"原始路径": "不读取的路径", "归属字": "批"},
                    limit=8,
                    _prepared_source=(
                        source,
                        gray,
                        "标准Alpha",
                        classification,
                        False,
                    ),
                    _batch_final_only=True,
                )
        finally:
            source.close()

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["处理类型"], CANDIDATE_TYPE_OPTIMIZED)
        self.assertEqual(candidates[0]["图像指纹"], "")
        self.assertEqual(generate.call_args.kwargs["limit"], 1)

    def test_batch_unprocessed_source_reuses_first_decode_and_classification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "复用.png"
            gradient = np.linspace(205, 255, 40, dtype=np.uint8)[:, None]
            source_arr = np.repeat(gradient, 50, axis=1)
            source_arr[12:30, 21:29] = 24
            Image.fromarray(source_arr, "L").save(path)
            mask = np.zeros((40, 50), dtype=np.uint8)
            mask[12:30, 21:29] = 1
            result = self._algorithm_result("单次解码", mask.tolist())

            with (
                patch.object(
                    self.service,
                    "_load_source",
                    wraps=self.service._load_source,
                ) as load_source,
                patch(
                    "services.optimization_service.generate_candidate_results",
                    return_value=[result],
                ),
                patch("services.optimization_service.write_log"),
            ):
                selected = self.service.generate_batch_candidate(
                    {"原始路径": str(path), "归属字": "复"}
                )

        load_source.assert_called_once_with(str(path))
        self.assertEqual(selected["处理类型"], CANDIDATE_TYPE_OPTIMIZED)

    def test_non_deduplicated_baseline_skips_pixel_fingerprint(self) -> None:
        source = Image.new("RGBA", (20, 20), (255, 255, 255, 0))
        for y in range(5, 15):
            for x in range(8, 12):
                source.putpixel((x, y), (24, 24, 24, 255))
        gray = np.full((20, 20), 255, dtype=np.float32)
        gray[5:15, 8:12] = 24
        classification = classify_source(source, gray, "标准Alpha")
        candidate_image = source.copy()

        with (
            patch.object(
                self.service,
                "_score_candidate_bounded",
                return_value=(90.0, {"模式": "测试"}),
            ),
            patch.object(source, "copy", return_value=candidate_image),
            patch.object(candidate_image, "tobytes") as tobytes,
        ):
            candidate = self.service._build_baseline_candidate(
                source,
                gray,
                "标准Alpha",
                classification,
                False,
                None,
            )

        source.close()
        tobytes.assert_not_called()
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["图像指纹"], "")

    def test_batch_uses_lower_structure_risk_before_higher_score(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "截图.png"
            gradient = np.linspace(205, 255, 40, dtype=np.uint8)[:, None]
            source_arr = np.repeat(gradient, 50, axis=1)
            source_arr[12:30, 21:29] = 24
            Image.fromarray(source_arr, "L").save(path)

            def candidate(score: float, risk_level: int) -> dict[str, object]:
                review = {
                    "状态": "需人工核对",
                    "阶段": "原尺寸复核",
                    "原因": "结构测试告警",
                    "风险等级": risk_level,
                }
                return {
                    "方案名": f"风险{risk_level}",
                    "方案": {"结构复核": dict(review)},
                    "结构复核": review,
                    "得分": score,
                    "图像": Image.new("RGBA", (2, 2), (0, 0, 0, 255)),
                    "灰度母版": np.zeros((2, 2), dtype=np.uint8),
                    "清洁掩码": np.ones((2, 2), dtype=np.uint8),
                    "处理类型": CANDIDATE_TYPE_OPTIMIZED,
                }

            lower_risk = candidate(72.0, 1)
            higher_risk = candidate(96.0, 2)
            with (
                patch.object(
                    self.service,
                    "generate_candidates",
                    return_value=[higher_risk, lower_risk],
                ),
                patch("services.optimization_service.write_log"),
            ):
                selected = self.service.generate_batch_candidate(
                    {"原始路径": str(path), "归属字": "险"}
                )

        self.assertIs(selected, lower_risk)
        self.assertTrue(self.service.requires_structure_review(selected))

    def test_batch_closes_loaded_source_when_cancelled_after_classification(self) -> None:
        source = Image.new("RGBA", (20, 20), (255, 255, 255, 0))
        for y in range(6, 14):
            for x in range(8, 12):
                source.putpixel((x, y), (0, 0, 0, 255))
        gray = np.full((20, 20), 255, dtype=np.float32)
        gray[6:14, 8:12] = 0
        checks = iter((False, True))

        with patch.object(
            self.service,
            "_load_source",
            return_value=(source, gray, "标准Alpha"),
        ):
            with self.assertRaises(OptimizationCancelled):
                self.service.generate_batch_candidate(
                    {"原始路径": "不读取的路径"},
                    cancel_check=lambda: next(checks, True),
                )

        with self.assertRaises(ValueError):
            source.getpixel((0, 0))

    def test_batch_cancel_after_fast_score_closes_unreturned_candidate(self) -> None:
        source = Image.new("RGBA", (20, 20), (255, 255, 255, 0))
        for y in range(6, 14):
            for x in range(8, 12):
                source.putpixel((x, y), (0, 0, 0, 255))
        gray = np.full((20, 20), 255, dtype=np.float32)
        gray[6:14, 8:12] = 0
        classification = classify_source(source, gray, "标准Alpha")
        candidate_image = source.copy()
        checks = iter((False, True))

        try:
            with patch.object(source, "copy", return_value=candidate_image):
                with self.assertRaises(OptimizationCancelled):
                    self.service._build_baseline_candidate(
                        source,
                        gray,
                        "标准Alpha",
                        classification,
                        False,
                        lambda: next(checks, True),
                        fast_batch_score=True,
                    )
        finally:
            source.close()

        with self.assertRaises(ValueError):
            candidate_image.getpixel((0, 0))

    def test_explore_rejects_non_optimized_candidate(self) -> None:
        for candidate_type in (CANDIDATE_TYPE_DIRECT, CANDIDATE_TYPE_TRANSPARENT):
            with self.subTest(candidate_type=candidate_type):
                with self.assertRaisesRegex(ValueError, "只有.*寻优优化"):
                    self.service.explore({}, {"处理类型": candidate_type})

    def test_save_selection_persists_candidate_type(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph = MagicMock()
            glyph.get_variant.return_value = {
                "原始文件": "测.tif",
                "图像信息": {"水平DPI": 300, "垂直DPI": 300},
            }
            workflow_dirs: dict[str, str] = {}
            for name in ("优化预览", "灰度母版", "清洁掩码", "手工审核", "成品"):
                path = Path(directory) / name
                path.mkdir()
                workflow_dirs[name] = str(path)
            glyph.get_workflow_dirs.return_value = workflow_dirs
            service = OptimizationService(glyph)
            candidate = {
                "方案名": "仅背景透明",
                "方案": {},
                "得分": 72.0,
                "图像": Image.new("RGBA", (2, 2), (0, 0, 0, 255)),
                "灰度母版": np.zeros((2, 2), dtype=np.uint8),
                "清洁掩码": np.ones((2, 2), dtype=np.uint8),
                "处理类型": CANDIDATE_TYPE_TRANSPARENT,
            }

            service.save_selection({"键": "variant-1"}, candidate)

        saved_scheme = glyph.confirm_optimization.call_args.args[4]
        self.assertEqual(saved_scheme["处理类型"], CANDIDATE_TYPE_TRANSPARENT)
        glyph.save.assert_called_once_with()

    def test_save_selection_rejects_unsafe_original_filename_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph = MagicMock()
            glyph.get_variant.return_value = {
                "原始文件": "../越界.tif",
                "图像信息": {"水平DPI": 300, "垂直DPI": 300},
            }
            workflow_dirs: dict[str, str] = {}
            for name in ("优化预览", "灰度母版", "清洁掩码", "手工审核", "成品"):
                path = Path(directory) / name
                path.mkdir()
                workflow_dirs[name] = str(path)
            glyph.get_workflow_dirs.return_value = workflow_dirs
            service = OptimizationService(glyph)
            candidate = {
                "方案名": "越界候选",
                "方案": {},
                "得分": 80.0,
                "图像": Image.new("RGBA", (2, 2), (0, 0, 0, 255)),
                "灰度母版": np.zeros((2, 2), dtype=np.uint8),
                "清洁掩码": np.ones((2, 2), dtype=np.uint8),
            }

            with self.assertRaisesRegex(ValueError, "安全的纯文件名"):
                service.save_selection({"键": "variant-unsafe"}, candidate)

            glyph.confirm_optimization.assert_not_called()
            glyph.save.assert_not_called()
            self.assertTrue(
                all(not list(Path(path).iterdir()) for path in workflow_dirs.values())
            )

    def test_interactive_save_rejects_library_locked_by_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph = MagicMock()
            glyph.ziku_dir = directory
            glyph.get_variant.return_value = {
                "原始文件": "锁测试.tif",
                "图像信息": {"水平DPI": 300, "垂直DPI": 300},
            }
            workflow_dirs: dict[str, str] = {}
            for name in ("优化预览", "灰度母版", "清洁掩码", "手工审核", "成品"):
                path = Path(directory) / name
                path.mkdir()
                workflow_dirs[name] = str(path)
            glyph.get_workflow_dirs.return_value = workflow_dirs
            service = OptimizationService(glyph)
            candidate = {
                "方案名": "锁测试",
                "方案": {},
                "得分": 80.0,
                "图像": Image.new("RGBA", (4, 4), (0, 0, 0, 255)),
                "灰度母版": np.zeros((4, 4), dtype=np.uint8),
                "清洁掩码": np.ones((4, 4), dtype=np.uint8),
            }
            batch_lock = acquire_batch_library_lock(directory)
            try:
                with self.assertRaisesRegex(RuntimeError, "正在执行其他批处理任务"):
                    service.save_selection({"键": "variant-lock"}, candidate)
            finally:
                batch_lock.release()

            glyph.confirm_optimization.assert_not_called()

    def test_save_selection_reuses_png_stream_digests_without_file_reread(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph = MagicMock()
            glyph.get_variant.return_value = {
                "原始文件": "流式摘要.tif",
                "图像信息": {"水平DPI": 300, "垂直DPI": 300},
            }
            workflow_dirs: dict[str, str] = {}
            for name in ("优化预览", "灰度母版", "清洁掩码", "手工审核", "成品"):
                path = Path(directory) / name
                path.mkdir()
                workflow_dirs[name] = str(path)
            glyph.get_workflow_dirs.return_value = workflow_dirs
            service = OptimizationService(glyph)
            candidate = {
                "方案名": "流式摘要候选",
                "方案": {},
                "得分": 88.0,
                "图像": Image.new("RGBA", (16, 12), (0, 0, 0, 255)),
                "灰度母版": np.zeros((12, 16), dtype=np.uint8),
                "清洁掩码": np.ones((12, 16), dtype=np.uint8),
                "处理类型": CANDIDATE_TYPE_OPTIMIZED,
            }

            with (
                patch(
                    "services.optimization_service.compute_file_md5",
                    side_effect=AssertionError("PNG 编码完成后不应再次回读计算摘要"),
                ),
                patch(
                    "services.optimization_service.FileTransaction.begin",
                    wraps=FileTransaction.begin,
                ) as begin_transaction,
            ):
                service.save_selection(
                    {"键": "variant-stream-digest"},
                    candidate,
                    persistence=MagicMock(),
                )

            begin_transaction.assert_called_once()

            call_args = glyph.confirm_optimization.call_args
            preview_path = Path(workflow_dirs["优化预览"]) / "流式摘要.png"
            gray_path = Path(workflow_dirs["灰度母版"]) / "流式摘要.png"
            mask_path = Path(workflow_dirs["清洁掩码"]) / "流式摘要.png"
            expected_hashes = [
                hashlib.md5(path.read_bytes()).hexdigest()
                for path in (preview_path, gray_path, mask_path)
            ]
            self.assertEqual(call_args.args[2], expected_hashes[0])
            self.assertEqual(call_args.kwargs["gray_master_md5"], expected_hashes[1])
            self.assertEqual(call_args.kwargs["clean_mask_md5"], expected_hashes[2])

    def test_uncertain_batch_journal_keeps_new_images_for_startup_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph = MagicMock()
            detail = {
                "原始文件": "待恢复.tif",
                "图像信息": {"水平DPI": 300, "垂直DPI": 300},
                "状态": config.STATUS_PENDING_OPTIMIZATION,
            }
            glyph.get_variant.return_value = detail
            workflow_dirs: dict[str, str] = {}
            for name in ("优化预览", "灰度母版", "清洁掩码", "手工审核", "成品"):
                path = Path(directory) / name
                path.mkdir()
                workflow_dirs[name] = str(path)
            glyph.get_workflow_dirs.return_value = workflow_dirs

            def mutate_detail(*_args, **_kwargs) -> None:
                detail.update({"状态": config.STATUS_PENDING_MANUAL_REVIEW})

            glyph.confirm_optimization.side_effect = mutate_detail
            persistence = MagicMock()
            persistence.record_variant.side_effect = BatchJournalUncertainError(
                "模拟日志提交结果未知"
            )
            service = OptimizationService(glyph)
            candidate = {
                "方案名": "待恢复候选",
                "方案": {},
                "得分": 88.0,
                "图像": Image.new("RGBA", (8, 8), (0, 0, 0, 255)),
                "灰度母版": np.zeros((8, 8), dtype=np.uint8),
                "清洁掩码": np.ones((8, 8), dtype=np.uint8),
                "处理类型": CANDIDATE_TYPE_OPTIMIZED,
            }

            with self.assertRaisesRegex(
                BatchJournalUncertainError,
                "提交结果未知",
            ):
                service.save_selection(
                    {"键": "variant-uncertain"},
                    candidate,
                    persistence=persistence,
                )

            self.assertEqual(detail["状态"], config.STATUS_PENDING_MANUAL_REVIEW)
            glyph.restore_variant_state.assert_not_called()
            for stage in ("优化预览", "灰度母版", "清洁掩码"):
                self.assertTrue((Path(workflow_dirs[stage]) / "待恢复.png").is_file())

    def test_save_selection_persists_structure_review_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph = MagicMock()
            glyph.get_variant.return_value = {
                "原始文件": "险.tif",
                "图像信息": {"水平DPI": 300, "垂直DPI": 300},
            }
            workflow_dirs: dict[str, str] = {}
            for name in ("优化预览", "灰度母版", "清洁掩码", "手工审核", "成品"):
                path = Path(directory) / name
                path.mkdir()
                workflow_dirs[name] = str(path)
            glyph.get_workflow_dirs.return_value = workflow_dirs
            service = OptimizationService(glyph)
            review = {
                "状态": "需人工核对",
                "阶段": "原尺寸复核",
                "原因": "参考端点仅匹配42.9%",
                "风险等级": 1,
            }
            candidate = {
                "方案名": "风险寻优",
                "方案": {},
                "结构复核": review,
                "得分": 72.0,
                "图像": Image.new("RGBA", (2, 2), (0, 0, 0, 255)),
                "灰度母版": np.zeros((2, 2), dtype=np.uint8),
                "清洁掩码": np.ones((2, 2), dtype=np.uint8),
                "处理类型": CANDIDATE_TYPE_OPTIMIZED,
            }

            service.save_selection({"键": "variant-risk"}, candidate)

        saved_scheme = glyph.confirm_optimization.call_args.args[4]
        self.assertEqual(saved_scheme["结构复核"], review)
        self.assertEqual(saved_scheme["处理类型"], CANDIDATE_TYPE_OPTIMIZED)

    def test_save_selection_restores_all_files_when_metadata_save_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph = MagicMock()
            detail = {
                "原始文件": "测.tif",
                "图像信息": {"水平DPI": 300, "垂直DPI": 300},
                "状态": "审核通过",
                "中间文件": "测.png",
                "审核文件": "测.png",
                "成品文件": "测.png",
            }
            original_detail = dict(detail)
            glyph.get_variant.return_value = detail
            workflow_dirs: dict[str, str] = {}
            old_contents: dict[Path, bytes] = {}
            for index, name in enumerate(("优化预览", "灰度母版", "清洁掩码", "手工审核", "成品"), 1):
                path = Path(directory) / name
                path.mkdir()
                workflow_dirs[name] = str(path)
                target = path / "测.png"
                target.write_bytes(f"旧文件{index}".encode("utf-8"))
                old_contents[target] = target.read_bytes()
            glyph.get_workflow_dirs.return_value = workflow_dirs

            def mutate_detail(*_args, **_kwargs) -> None:
                detail.update({"状态": "待审核", "中间文件": "新文件.png", "成品文件": ""})

            glyph.confirm_optimization.side_effect = mutate_detail
            glyph.save.side_effect = OSError("模拟数据文件写入失败")
            service = OptimizationService(glyph)
            candidate = {
                "方案名": "寻优结果",
                "方案": {},
                "得分": 88.0,
                "图像": Image.new("RGBA", (2, 2), (0, 0, 0, 255)),
                "灰度母版": np.zeros((2, 2), dtype=np.uint8),
                "清洁掩码": np.ones((2, 2), dtype=np.uint8),
                "处理类型": CANDIDATE_TYPE_OPTIMIZED,
            }

            with self.assertRaisesRegex(OSError, "模拟数据文件写入失败"):
                service.save_selection({"键": "variant-1"}, candidate)

            self.assertEqual(detail, original_detail)
            for target, old_content in old_contents.items():
                self.assertEqual(target.read_bytes(), old_content)
            self.assertEqual(list(Path(directory).rglob(".fonteditor_*")), [])

    def test_save_failure_restores_differently_named_stage_files_and_full_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph = MagicMock()
            detail = {
                "原始文件": "测.tif",
                "图像信息": {"水平DPI": 300, "垂直DPI": 300},
                "状态": "成品已生成",
                "中间文件": "旧优化.png",
                "灰度母版文件": "旧灰度.png",
                "清洁掩码文件": "旧掩码.png",
                "审核文件": "旧审核.png",
                "成品文件": "旧成品.png",
            }
            original_detail = dict(detail)
            snapshot = {
                "元数据": {"最后修改": "保存前"},
                "整体协调": {"几何协调完成": True, "墨色统一完成": True},
            }
            glyph.get_variant.return_value = detail
            glyph.snapshot_variant_state.return_value = snapshot
            workflow_dirs: dict[str, str] = {}
            for name in ("优化预览", "灰度母版", "清洁掩码", "手工审核", "成品"):
                path = Path(directory) / name
                path.mkdir()
                workflow_dirs[name] = str(path)
            glyph.get_workflow_dirs.return_value = workflow_dirs

            old_stage_files = {
                Path(workflow_dirs["优化预览"]) / "旧优化.png": b"old-preview",
                Path(workflow_dirs["灰度母版"]) / "旧灰度.png": b"old-gray",
                Path(workflow_dirs["清洁掩码"]) / "旧掩码.png": b"old-mask",
                Path(workflow_dirs["手工审核"]) / "旧审核.png": b"old-reviewed",
                Path(workflow_dirs["成品"]) / "旧成品.png": b"old-finished",
            }
            for path, content in old_stage_files.items():
                path.write_bytes(content)

            def mutate_detail(*_args, **_kwargs) -> None:
                detail.update({"状态": "待审核", "审核文件": "", "成品文件": ""})

            glyph.confirm_optimization.side_effect = mutate_detail
            glyph.save.side_effect = OSError("模拟完整状态写入失败")
            service = OptimizationService(glyph)
            candidate = {
                "方案名": "寻优结果",
                "方案": {},
                "得分": 88.0,
                "图像": Image.new("RGBA", (2, 2), (0, 0, 0, 255)),
                "灰度母版": np.zeros((2, 2), dtype=np.uint8),
                "清洁掩码": np.ones((2, 2), dtype=np.uint8),
                "处理类型": CANDIDATE_TYPE_OPTIMIZED,
            }

            with self.assertRaisesRegex(OSError, "模拟完整状态写入失败"):
                service.save_selection({"键": "variant-1"}, candidate)

            self.assertEqual(detail, original_detail)
            for path, content in old_stage_files.items():
                self.assertEqual(path.read_bytes(), content)
            glyph.restore_variant_state.assert_called_once_with(snapshot)
            self.assertEqual(list(Path(directory).rglob(".fonteditor_*")), [])

    def test_photoshop_layer_detection_requires_actual_transparency(self) -> None:
        transparent_payload = self._photoshop_payload([
            ((0, 0, 20, 20), [-1, 0], 255, False),
        ])
        opaque_payload = self._photoshop_payload([
            ((0, 0, 20, 20), [-1, 0], 255, False),
            ((0, 0, 20, 20), [0], 255, False),
        ])
        transparent_image = self._TaggedImage((20, 20), transparent_payload)
        opaque_image = self._TaggedImage((20, 20), opaque_payload)

        self.assertTrue(self.service._has_photoshop_transparency(transparent_image))
        self.assertFalse(self.service._has_photoshop_transparency(opaque_image))

    def test_candidate_validation_rejects_empty_or_mismatched_layers(self) -> None:
        valid = {
            "方案名": "有效候选",
            "方案": {},
            "得分": 88.0,
            "图像": Image.new("RGBA", (3, 2), (0, 0, 0, 255)),
            "灰度母版": np.zeros((2, 3), dtype=np.uint8),
            "清洁掩码": np.ones((2, 3), dtype=np.uint8),
        }
        self.assertTrue(self.service.is_candidate_valid(valid))
        self.assertEqual(self.service.candidate_validation_error(valid), "")

        cases = {
            "没有前景": dict(valid, **{"清洁掩码": np.zeros((2, 3), dtype=np.uint8)}),
            "分层尺寸不同": dict(valid, **{"清洁掩码": np.ones((3, 2), dtype=np.uint8)}),
            "图片尺寸不同": dict(valid, **{"图像": Image.new("RGBA", (2, 2))}),
            "包含无效数值": dict(
                valid,
                **{"灰度母版": np.full((2, 3), np.nan, dtype=np.float32)},
            ),
        }
        for name, candidate in cases.items():
            with self.subTest(name=name):
                self.assertFalse(self.service.is_candidate_valid(candidate))
                self.assertTrue(self.service.candidate_validation_error(candidate))

    def test_candidate_validation_requires_visible_image_ink(self) -> None:
        clean_mask = np.ones((2, 3), dtype=np.uint8)
        base_candidate = {
            "方案名": "图片前景校验",
            "方案": {},
            "得分": 88.0,
            "灰度母版": np.zeros((2, 3), dtype=np.uint8),
            "清洁掩码": clean_mask,
        }
        invalid_images = {
            "全透明图片": Image.new("RGBA", (3, 2), (0, 0, 0, 0)),
            "纯白图片": Image.new("RGBA", (3, 2), (255, 255, 255, 255)),
        }
        for name, image in invalid_images.items():
            with self.subTest(name=name):
                candidate = dict(base_candidate, **{"图像": image})
                self.assertFalse(self.service.is_candidate_valid(candidate))
                self.assertIn(
                    "没有可见的非白文字前景",
                    self.service.candidate_validation_error(candidate),
                )

        visible_image = Image.new("RGBA", (3, 2), (255, 255, 255, 0))
        visible_image.putpixel((1, 1), (0, 0, 0, 255))
        valid_candidate = dict(base_candidate, **{"图像": visible_image})
        with patch(
            "services.optimization_service.np.asarray",
            side_effect=AssertionError("可见墨迹校验不应构造 RGBA NumPy 数组"),
        ):
            self.assertTrue(self.service.is_candidate_valid(valid_candidate))
            self.assertEqual(self.service.candidate_validation_error(valid_candidate), "")

        threshold_cases = (
            ("仅1级墨差", Image.new("RGBA", (3, 2), (254, 254, 254, 255)), False),
            ("2级墨差", Image.new("RGBA", (3, 2), (253, 253, 253, 255)), True),
        )
        for name, image, expected_valid in threshold_cases:
            with self.subTest(name=name):
                candidate = dict(base_candidate, **{"图像": image})
                self.assertEqual(self.service.is_candidate_valid(candidate), expected_valid)

    @staticmethod
    def _algorithm_result(
        name: str,
        mask: list[list[int]],
        protect_original: bool = False,
    ) -> dict[str, object]:
        return {
            "方案名": name,
            "方案": {"预处理": {}},
            "得分": 80.0,
            "掩码": np.asarray(mask, dtype=np.uint8),
            "质量等级": "低污染",
            "保留原图": protect_original,
        }

    @staticmethod
    def _photoshop_payload(
        layers: list[tuple[tuple[int, int, int, int], list[int], int, bool]],
    ) -> bytes:
        records = bytearray(struct.pack(">h", len(layers)))
        for bounds, channels, opacity, hidden in layers:
            records.extend(struct.pack(">4iH", *bounds, len(channels)))
            for channel_id in channels:
                records.extend(struct.pack(">hI", channel_id, 0))
            flags = 0x02 if hidden else 0
            records.extend(b"8BIMnorm")
            records.extend(bytes((opacity, 0, flags, 0)))
            records.extend(struct.pack(">I", 0))
        header = b"Adobe Photoshop Document Data Block\x00"
        return header + b"8BIMLayr" + struct.pack(">I", len(records)) + bytes(records)

    class _TaggedImage:
        def __init__(self, size: tuple[int, int], payload: bytes) -> None:
            self.size = size
            self.tag_v2 = {37724: payload}


if __name__ == "__main__":
    unittest.main()
