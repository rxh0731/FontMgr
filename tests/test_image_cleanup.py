"""图片实验室共享背景清理算法测试。"""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from core.image_cleanup import (
    PROCESSING_MODE_GENERAL,
    PROCESSING_MODE_RUBBING,
    ImageCleanupOptions,
    _edge_risk_map,
    clean_document_image,
)


class ImageCleanupTests(unittest.TestCase):
    def test_edge_feathering_can_be_disabled_without_intermediate_alpha(self) -> None:
        source = np.full((180, 260, 3), 230, dtype=np.uint8)
        cv2.rectangle(source, (50, 40), (210, 140), (30, 30, 30), -1)

        feathered = clean_document_image(
            source,
            ImageCleanupOptions(
                detect_page=False,
                remove_small_noise=False,
                processing_mode=PROCESSING_MODE_GENERAL,
                feather_edges=True,
            ),
        )
        hard = clean_document_image(
            source,
            ImageCleanupOptions(
                detect_page=False,
                remove_small_noise=False,
                processing_mode=PROCESSING_MODE_GENERAL,
                feather_edges=False,
            ),
        )

        feather_alpha = feathered.cleanup_layer[:, :, 3]
        hard_alpha = hard.cleanup_layer[:, :, 3]
        self.assertGreater(
            int(np.count_nonzero((feather_alpha > 0) & (feather_alpha < 255))),
            0,
        )
        self.assertEqual(
            int(np.count_nonzero((hard_alpha > 0) & (hard_alpha < 255))),
            0,
        )
        self.assertEqual(feathered.metrics["边缘羽化"], "是")
        self.assertEqual(hard.metrics["边缘羽化"], "否")

    def test_zero_strength_is_an_exact_original_preservation_mode(self) -> None:
        source = np.full((180, 260, 3), 220, dtype=np.uint8)
        cv2.putText(
            source,
            "ORIGINAL",
            (18, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (30, 45, 70),
            5,
            cv2.LINE_AA,
        )
        result = clean_document_image(
            source,
            ImageCleanupOptions(strength=0),
        )

        self.assertTrue(np.array_equal(result.composite, source))
        self.assertTrue(np.all(result.cleanup_layer[:, :, 3] == 0))
        self.assertTrue(np.all(result.foreground_mask == 255))
        self.assertEqual(result.resolved_profile, "原稿保真（未自动清理）")
        self.assertEqual(result.metrics["完全清理占比"], 0.0)

    def test_low_rubbing_strength_keeps_ambiguous_thin_strokes(self) -> None:
        height, width = 280, 420
        source = np.full((height, width, 3), 238, dtype=np.uint8)
        text_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.line(source, (35, 75), (385, 75), (85, 85, 85), 2, cv2.LINE_AA)
        cv2.line(text_mask, (35, 75), (385, 75), 255, 2, cv2.LINE_8)
        cv2.putText(
            source,
            "细笔",
            (75, 205),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            (42, 42, 42),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            text_mask,
            "细笔",
            (75, 205),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            255,
            2,
            cv2.LINE_8,
        )
        random = np.random.default_rng(5)
        rows = random.integers(0, height, 1200)
        columns = random.integers(0, width, 1200)
        source[rows, columns] = random.integers(130, 205, (1200, 1), dtype=np.uint8)
        low = clean_document_image(
            source,
            ImageCleanupOptions(
                strength=1,
                processing_mode=PROCESSING_MODE_RUBBING,
                detect_page=False,
            ),
        )
        higher = clean_document_image(
            source,
            ImageCleanupOptions(
                strength=25,
                processing_mode=PROCESSING_MODE_RUBBING,
                detect_page=False,
            ),
        )

        text = text_mask > 0
        self.assertGreater(float(np.mean(low.foreground_mask[text] > 0)), 0.98)
        self.assertGreaterEqual(
            int(np.count_nonzero(low.foreground_mask)),
            int(np.count_nonzero(higher.foreground_mask)),
        )

    def test_rubbing_strength_transition_does_not_have_a_hard_cutoff(self) -> None:
        height, width = 480, 640
        random = np.random.default_rng(20260827)
        source = np.full((height, width, 3), 242, dtype=np.int16)
        source += random.normal(0.0, 10.0, (height, width, 1)).astype(np.int16)
        source = np.clip(source, 0, 255).astype(np.uint8)
        cv2.putText(
            source,
            "RUBBING",
            (35, 280),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.4,
            (28, 28, 28),
            9,
            cv2.LINE_AA,
        )

        foreground_ratios = {}
        for strength in (25, 26, 27):
            result = clean_document_image(
                source,
                ImageCleanupOptions(
                    strength=strength,
                    processing_mode=PROCESSING_MODE_RUBBING,
                    detect_page=False,
                ),
            )
            foreground_ratios[strength] = float(
                result.metrics["保留前景占比"]
            )

        self.assertLess(
            foreground_ratios[25] - foreground_ratios[26],
            0.25,
        )
        self.assertLess(
            foreground_ratios[26] - foreground_ratios[27],
            0.10,
        )

    def test_rubbing_mode_adapts_to_uneven_background_by_global_position(
        self,
    ) -> None:
        height, width = 600, 1000
        source = np.full((height, width, 3), 242, dtype=np.uint8)
        text_mask = np.zeros((height, width), dtype=np.uint8)
        for text, origin in (("LEFT", (40, 320)), ("RIGHT", (540, 320))):
            cv2.putText(
                source,
                text,
                origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                2.2,
                (25, 25, 25),
                10,
                cv2.LINE_AA,
            )
            cv2.putText(
                text_mask,
                text,
                origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                2.2,
                255,
                6,
                cv2.LINE_8,
            )
        random = np.random.default_rng(112)
        for _index in range(2500):
            center = (
                int(random.integers(0, 480)),
                int(random.integers(0, height)),
            )
            radius = int(random.integers(1, 3))
            value = int(random.integers(100, 200))
            cv2.circle(source, center, radius, (value, value, value), -1)

        options = ImageCleanupOptions(
            strength=37,
            processing_mode=PROCESSING_MODE_RUBBING,
            detect_page=False,
        )
        result = clean_document_image(source, options)
        kept = result.foreground_mask > 0
        text = text_mask > 0
        columns = np.indices((height, width))[1]
        background = cv2.dilate(
            text_mask,
            np.ones((31, 31), dtype=np.uint8),
        ) == 0
        grid = np.asarray(result.calibration.rubbing_difficulty_grid)

        self.assertGreater(float(np.mean(kept[text])), 0.995)
        self.assertLess(float(np.mean(kept[background & (columns < 480)])), 0.10)
        self.assertGreater(
            float(np.mean(grid[:, :3])),
            float(np.mean(grid[:, -3:])) + 0.35,
        )
        self.assertEqual(result.metrics["局部自适应"], "是")
        self.assertGreater(float(result.metrics["背景不均匀指数"]), 0.03)

        left = clean_document_image(
            source[:, :500],
            options,
            calibration=result.calibration,
            source_region=(0, 0, 500, height),
            source_size=(width, height),
        )
        right = clean_document_image(
            source[:, 500:],
            options,
            calibration=result.calibration,
            source_region=(500, 0, width, height),
            source_size=(width, height),
        )
        self.assertGreater(
            float(left.metrics["局部强阈值最大值"]),
            float(right.metrics["局部强阈值最大值"]) + 10.0,
        )

    def test_rubbing_mode_removes_dense_speckles_and_keeps_stroke_cores(self) -> None:
        height, width = 620, 820
        source = np.full((height, width, 3), 242, dtype=np.uint8)
        text_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.putText(
            source,
            "RUBBING",
            (35, 330),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.2,
            (28, 28, 28),
            10,
            cv2.LINE_AA,
        )
        cv2.putText(
            text_mask,
            "RUBBING",
            (35, 330),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.2,
            255,
            6,
            cv2.LINE_8,
        )
        rng = np.random.default_rng(139)
        rows = rng.integers(0, height, 9000)
        columns = rng.integers(0, width, 9000)
        source[rows, columns] = rng.integers(80, 190, (9000, 1), dtype=np.uint8)
        for column in range(70, width, 95):
            cv2.line(source, (column, 0), (column, height - 1), (185, 185, 185), 1)

        result = clean_document_image(
            source,
            ImageCleanupOptions(
                processing_mode=PROCESSING_MODE_RUBBING,
                detect_page=False,
            ),
        )
        kept = result.foreground_mask > 0
        text_pixels = text_mask > 0
        background = cv2.dilate(text_mask, np.ones((41, 41), np.uint8)) == 0

        self.assertIn("笔画尺度重建", result.resolved_profile)
        self.assertGreater(float(np.mean(kept[text_pixels])), 0.985)
        self.assertLess(float(np.mean(kept[background])), 0.035)
        self.assertEqual(
            result.calibration.resolved_mode,
            PROCESSING_MODE_RUBBING,
        )

    def test_rubbing_mode_cleans_uneven_border_without_losing_center_strokes(
        self,
    ) -> None:
        height, width = 360, 520
        random = np.random.default_rng(20260827)
        source = np.full((height, width, 3), 242, dtype=np.uint8)
        text_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.putText(
            source,
            "TEST",
            (120, 230),
            cv2.FONT_HERSHEY_SIMPLEX,
            3.0,
            (28, 28, 28),
            10,
            cv2.LINE_AA,
        )
        cv2.putText(
            text_mask,
            "TEST",
            (120, 230),
            cv2.FONT_HERSHEY_SIMPLEX,
            3.0,
            255,
            6,
            cv2.LINE_8,
        )
        # 模拟四边底色和漏墨明显深于中心的拓片污染。
        for top, bottom, left, right in (
            (0, 70, 0, width),
            (height - 70, height, 0, width),
            (0, height, 0, 45),
            (0, height, width - 45, width),
        ):
            source[top:bottom, left:right] = random.integers(
                80,
                190,
                (bottom - top, right - left, 1),
                dtype=np.uint8,
            )
        for _index in range(500):
            row = int(random.integers(0, height))
            column = int(random.integers(0, width))
            if 60 < row < height - 60 and 50 < column < width - 50:
                continue
            value = int(random.integers(90, 200))
            cv2.circle(
                source,
                (column, row),
                int(random.integers(1, 4)),
                (value, value, value),
                -1,
            )

        result = clean_document_image(
            source,
            ImageCleanupOptions(
                strength=34,
                processing_mode=PROCESSING_MODE_RUBBING,
                detect_page=False,
            ),
        )
        kept = result.foreground_mask > 127
        edge = _edge_risk_map(
            kept.shape,
            (0, 0, width, height),
            (width, height),
        )

        self.assertGreater(float(np.mean(kept[text_mask > 0])), 0.98)
        self.assertLess(float(np.mean(kept[edge > 0.5])), 0.08)
        self.assertLess(
            float(np.mean(kept[edge == 0])),
            0.35,
        )
        self.assertEqual(result.metrics["边缘污染自适应"], "是")
        self.assertGreater(float(result.metrics["边缘阈值增量最大值"]), 20.0)

    def test_edge_risk_uses_original_coordinates_for_chunks(self) -> None:
        height, width = 240, 360
        full = _edge_risk_map(
            (height, width),
            (0, 0, width, height),
            (width, height),
        )
        internal = _edge_risk_map(
            (height, 120),
            (120, 0, 240, height),
            (width, height),
        )

        self.assertGreater(float(full.max()), 0.99)
        self.assertEqual(float(internal[30:-30].max()), 0.0)

    def test_general_mode_can_be_selected_explicitly(self) -> None:
        source = np.full((180, 260, 3), 225, dtype=np.uint8)
        cv2.putText(
            source,
            "A",
            (85, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.5,
            (25, 25, 25),
            7,
        )

        result = clean_document_image(
            source,
            ImageCleanupOptions(
                processing_mode=PROCESSING_MODE_GENERAL,
                detect_page=False,
            ),
        )

        self.assertEqual(result.calibration.resolved_mode, PROCESSING_MODE_GENERAL)
        self.assertEqual(result.resolved_profile, "多通道通用识别")

    def test_textured_yellow_paper_is_removed_without_losing_colored_text(self) -> None:
        height, width = 520, 760
        rng = np.random.default_rng(20260826)
        coarse = rng.normal(0.0, 1.0, (26, 38)).astype(np.float32)
        texture = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
        texture = cv2.GaussianBlur(texture, (0, 0), 4.0)
        base = np.empty((height, width, 3), dtype=np.float32)
        base[:, :, 0] = 236.0 + texture * 12.0
        base[:, :, 1] = 220.0 + texture * 10.0
        base[:, :, 2] = 168.0 + texture * 5.0
        source = np.clip(base, 0, 255).astype(np.uint8)
        text_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.putText(
            source,
            "BLUE",
            (45, 235),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.8,
            (38, 74, 190),
            8,
            cv2.LINE_AA,
        )
        cv2.putText(
            text_mask,
            "BLUE",
            (45, 235),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.8,
            255,
            5,
            cv2.LINE_8,
        )
        cv2.putText(
            source,
            "RED",
            (245, 420),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.8,
            (205, 42, 58),
            8,
            cv2.LINE_AA,
        )
        cv2.putText(
            text_mask,
            "RED",
            (245, 420),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.8,
            255,
            5,
            cv2.LINE_8,
        )

        result = clean_document_image(
            source,
            ImageCleanupOptions(detect_page=False),
        )
        kept = result.foreground_mask > 0
        text_pixels = text_mask > 0
        background_pixels = cv2.dilate(text_mask, np.ones((31, 31), np.uint8)) == 0

        self.assertGreater(float(np.mean(kept[text_pixels])), 0.995)
        self.assertLess(float(np.mean(kept[background_pixels])), 0.03)
        self.assertTrue(result.calibration.colorful_document)
        self.assertGreater(len(result.calibration.background_palette), 1)

    def test_uniform_background_is_white_and_dark_text_is_preserved(self) -> None:
        source = np.full((240, 320, 3), (220, 210, 190), dtype=np.uint8)
        cv2.putText(
            source,
            "TEXT",
            (45, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.2,
            (35, 30, 25),
            5,
            cv2.LINE_AA,
        )
        original = source.copy()

        result = clean_document_image(
            source,
            ImageCleanupOptions(),
        )

        self.assertTrue(np.array_equal(source, original))
        self.assertGreater(float(result.composite[20, 20].mean()), 250.0)
        self.assertLess(float(result.composite[120, 100].mean()), 150.0)
        self.assertGreater(int(result.cleanup_layer[20, 20, 3]), 245)
        self.assertLess(int(result.cleanup_layer[120, 100, 3]), 20)

    def test_colored_writing_is_preserved_on_yellow_paper(self) -> None:
        source = np.full((260, 360, 3), (236, 222, 174), dtype=np.uint8)
        cv2.line(source, (20, 80), (340, 80), (228, 178, 178), 1)
        cv2.putText(
            source,
            "Blue",
            (45, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.8,
            (45, 80, 180),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            source,
            "R",
            (270, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (190, 45, 55),
            4,
            cv2.LINE_AA,
        )

        result = clean_document_image(
            source,
            ImageCleanupOptions(),
        )

        self.assertGreater(float(result.composite[20, 20].mean()), 250.0)
        self.assertLess(int(result.composite[140, 70, 2]), 240)
        self.assertLess(int(result.cleanup_layer[140, 70, 3]), 80)

    def test_dark_area_outside_light_document_is_removed(self) -> None:
        source = np.full((300, 240, 3), 55, dtype=np.uint8)
        source[30:270, 35:205] = 215
        cv2.putText(
            source,
            "A",
            (85, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.5,
            (30, 30, 30),
            6,
            cv2.LINE_AA,
        )

        result = clean_document_image(source)

        self.assertGreater(int(result.cleanup_layer[5, 5, 3]), 245)
        self.assertLess(int(result.cleanup_layer[90:190, 70:160, 3].min()), 30)

    def test_light_text_on_dark_background_is_preserved(self) -> None:
        source = np.full((240, 340, 3), (28, 35, 46), dtype=np.uint8)
        cv2.putText(
            source,
            "LIGHT",
            (35, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.8,
            (222, 210, 182),
            5,
            cv2.LINE_AA,
        )

        result = clean_document_image(
            source,
            ImageCleanupOptions(detect_page=False),
        )

        self.assertGreater(int(result.cleanup_layer[20, 20, 3]), 245)
        text_region = result.cleanup_layer[95:150, 35:290, 3]
        self.assertLess(int(np.percentile(text_region, 5)), 30)

    def test_equal_luminance_colored_text_is_preserved(self) -> None:
        source = np.full((220, 320, 3), (78, 145, 110), dtype=np.uint8)
        cv2.putText(
            source,
            "COLOR",
            (35, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.7,
            (173, 92, 142),
            5,
            cv2.LINE_AA,
        )

        result = clean_document_image(
            source,
            ImageCleanupOptions(detect_page=False),
        )

        self.assertGreater(int(result.cleanup_layer[15, 15, 3]), 245)
        self.assertLess(int(result.cleanup_layer[120, 75:260, 3].min()), 30)

    def test_textured_paper_does_not_become_one_foreground_region(self) -> None:
        rng = np.random.default_rng(42)
        coarse = rng.normal(0, 1, (28, 38)).astype(np.float32)
        texture = cv2.resize(coarse, (380, 280), interpolation=cv2.INTER_CUBIC)
        texture = cv2.GaussianBlur(texture, (0, 0), 3.0)
        paper = np.clip(218 + texture * 16, 0, 255).astype(np.uint8)
        source = np.repeat(paper[:, :, None], 3, axis=2)
        cv2.putText(
            source,
            "INK",
            (65, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.8,
            (35, 45, 80),
            7,
            cv2.LINE_AA,
        )

        result = clean_document_image(
            source,
            ImageCleanupOptions(detect_page=False),
        )

        self.assertLess(float(np.mean(result.foreground_mask > 0)), 0.25)
        self.assertGreater(float(result.composite[30, 30].mean()), 245.0)
        self.assertLess(int(result.cleanup_layer[150, 85:280, 3].min()), 30)

    def test_stronger_setting_never_preserves_more_weak_background(self) -> None:
        rng = np.random.default_rng(12)
        source = np.full((220, 300, 3), 225, dtype=np.int16)
        noise = rng.normal(0, 8, (220, 300, 1))
        source = np.clip(source + noise, 0, 255).astype(np.uint8)
        cv2.putText(source, "A", (105, 155), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (20, 20, 20), 6)

        conservative = clean_document_image(
            source,
            ImageCleanupOptions(strength=20),
        )
        strong = clean_document_image(
            source,
            ImageCleanupOptions(strength=85),
        )

        self.assertLessEqual(
            np.count_nonzero(strong.foreground_mask),
            np.count_nonzero(conservative.foreground_mask),
        )

    def test_results_are_immutable_and_options_are_validated(self) -> None:
        result = clean_document_image(np.full((32, 40), 220, dtype=np.uint8))

        self.assertFalse(result.composite.flags.writeable)
        self.assertFalse(result.cleanup_layer.flags.writeable)
        self.assertIsInstance(result.calibration.background_palette, tuple)
        with self.assertRaises(ValueError):
            ImageCleanupOptions(strength=101)
        with self.assertRaises(ValueError):
            ImageCleanupOptions(processing_mode="不存在的方式")
        with self.assertRaises(ValueError):
            clean_document_image(np.zeros((1, 1), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
