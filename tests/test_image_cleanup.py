"""图片实验室共享背景清理算法测试。"""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from core.image_cleanup import ImageCleanupOptions, clean_document_image


class ImageCleanupTests(unittest.TestCase):
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
            clean_document_image(np.zeros((1, 1), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
