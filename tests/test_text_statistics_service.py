from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from services.glyph_service import GlyphService
from services.text_statistics_service import (
    TextStatisticsService,
    analyze_text,
    clean_plain_text,
    is_chinese_character,
)
from utils.file_utils import compute_file_md5


class TextStatisticsServiceTests(unittest.TestCase):
    def test_text_analysis_matches_legacy_counting_rules(self) -> None:
        statistics = analyze_text("乙甲甲，ABC def。\n")

        self.assertEqual(statistics.total_characters, 13)
        self.assertEqual(statistics.chinese_characters, 3)
        self.assertEqual(statistics.english_words, 2)
        self.assertEqual(statistics.punctuation, 2)
        self.assertEqual(statistics.whitespace, 2)
        self.assertEqual(statistics.unique_chinese, ("甲", "乙"))

    def test_chinese_ranges_and_plain_text_cleanup_are_preserved(self) -> None:
        self.assertTrue(is_chinese_character("字"))
        self.assertTrue(is_chinese_character("𠀀"))
        self.assertFalse(is_chinese_character("A"))
        self.assertEqual(clean_plain_text("\ufeff甲\x00\r\n\r\n\r\n\r\n乙"), "甲\n\n\n乙")

    def test_external_directory_uses_legacy_filename_rules(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "甲-0001.png").write_bytes(b"not-decoded")
            (root / "乙.png").write_bytes(b"not-decoded")
            (root / "名称异常.png").write_bytes(b"not-decoded")
            (root / "忽略.txt").write_text("丙", encoding="utf-8")

            result = TextStatisticsService.analyze_missing(("甲", "乙", "丙"), directory)

            self.assertEqual(result.source_kind, "外部图片目录")
            self.assertEqual(result.existing_characters, 2)
            self.assertEqual(result.invalid_filenames, 1)
            self.assertEqual(result.missing_characters, ("丙",))

    def test_system_library_uses_real_readable_finished_variants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            library_directory = Path(directory) / "数据库字库"
            glyph = GlyphService("数据库字库", str(library_directory))
            glyph.ensure_dirs()
            first_id = glyph.add_original("甲", "甲-0001.png", "甲.png", "md5-a")
            second_id = glyph.add_original("乙", "乙-0001.png", "乙.png", "md5-b")
            glyph.add_original("丙", "丙-0001.png", "丙.png", "md5-c")
            broken_id = glyph.add_original("丁", "丁-0001.png", "丁.png", "md5-d")
            finished_dir = Path(glyph.get_workflow_dirs()["成品"])
            for variant_id, filename in (
                (first_id, "甲-0001.png"),
                (second_id, "乙-0001.png"),
            ):
                path = finished_dir / filename
                Image.new("RGBA", (8, 8), (0, 0, 0, 255)).save(path)
                glyph.mark_finished(variant_id, filename, compute_file_md5(path), {})
            broken_path = finished_dir / "丁-0001.png"
            broken_path.write_bytes(b"invalid-image")
            glyph.mark_finished(
                broken_id,
                broken_path.name,
                compute_file_md5(broken_path),
                {},
            )
            glyph.save()

            result = TextStatisticsService.analyze_system_library(
                ("甲", "乙", "丙", "丁"),
                str(library_directory),
            )

            self.assertEqual(result.source_kind, "本系统字库“数据库字库”")
            self.assertEqual(result.existing_characters, 2)
            self.assertEqual(result.invalid_filenames, 0)
            self.assertEqual(result.valid_variants, 2)
            self.assertEqual(result.missing_characters, ("丙", "丁"))
            self.assertEqual(len(result.issues), 1)
            self.assertIn("丁-0001.png", result.issues[0])

    def test_ocr_order_restores_vertical_columns_from_right_to_left(self) -> None:
        texts = ["甲", "乙", "丙", "丁"]
        boxes = [
            [[80, 10], [90, 10], [90, 35], [80, 35]],
            [[80, 45], [90, 45], [90, 70], [80, 70]],
            [[40, 10], [50, 10], [50, 35], [40, 35]],
            [[40, 45], [50, 45], [50, 70], [40, 70]],
        ]

        text, layout = TextStatisticsService.arrange_ocr_order(texts, boxes)

        self.assertEqual(text, "甲乙\n丙丁")
        self.assertIn("竖排", layout)

    def test_clipboard_ocr_uses_existing_engine_and_closes_image(self) -> None:
        class Result:
            txts = ["般若"]
            boxes = None

        class Engine:
            def __call__(self, _pixels: object) -> Result:
                return Result()

        previous = TextStatisticsService._ocr_engine
        TextStatisticsService._ocr_engine = Engine()
        image = Image.new("RGB", (10, 10), "white")
        try:
            result = TextStatisticsService.recognize_clipboard_image(image)
        finally:
            TextStatisticsService._ocr_engine = previous

        self.assertEqual(result.text, "般若")
        self.assertEqual(len(result.reports), 1)
        with self.assertRaises(ValueError):
            image.getpixel((0, 0))

    def test_batch_file_extraction_keeps_successes_and_reports_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text_path = root / "经文.txt"
            text_path.write_text("般若波羅蜜多", encoding="utf-8")
            missing_path = root / "不存在.txt"

            result = TextStatisticsService.extract_files((str(text_path), str(missing_path)))

            self.assertEqual(result.text, "般若波羅蜜多")
            self.assertEqual(len(result.reports), 1)
            self.assertEqual(len(result.failures), 1)
            self.assertIn("不存在.txt", result.failures[0])


if __name__ == "__main__":
    unittest.main()
