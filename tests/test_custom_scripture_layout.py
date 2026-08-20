"""定制经文排版核心规则回归测试。"""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from PIL import Image

from core.custom_scripture_layout import (
    CustomBoardParameters,
    CustomLayoutTemplateParameters,
    allocate_custom_boards,
    parse_custom_scripture,
)
from services.scripture_layout_service import (
    GlyphImage,
    GlyphIndex,
    _target_image_size_for_cell,
    generate_psd_boards,
    render_board_preview,
)


class CustomScriptureLayoutCoreTests(unittest.TestCase):
    def test_glyph_size_uses_physical_dimensions_instead_of_dpi_ratio(self) -> None:
        parameters = CustomBoardParameters(dpi=200).as_layout_parameters(
            rows=1,
            columns=1,
            include_punctuation=False,
            add_annotations=False,
        )

        target = _target_image_size_for_cell(
            240,
            120,
            600,
            300,
            300,
            parameters,
            source_width_mm=12.7,
            source_height_mm=6.35,
            use_physical_size=True,
        )

        self.assertEqual(target, (100, 50))
        self.assertNotEqual(target, (80, 40))

    def test_blank_lines_split_pages_and_each_nonempty_line_is_a_column(self) -> None:
        parsed = parse_custom_scripture(
            "甲乙丙\n丁戊\n\n己庚\n辛壬癸\n\n\n子丑",
            set("甲乙丙丁戊己庚辛壬癸子丑"),
        )

        self.assertEqual(len(parsed.pages), 3)
        self.assertEqual([len(page.columns) for page in parsed.pages], [2, 2, 1])
        self.assertEqual(
            [[len(column) for column in page.columns] for page in parsed.pages],
            [[3, 2], [2, 3], [2]],
        )

    def test_extra_text_pages_are_ignored_and_first_line_is_rightmost(self) -> None:
        parsed = parse_custom_scripture("甲乙\n丙丁\n\n戊己", set("甲乙丙丁戊己"))
        parameters = CustomLayoutTemplateParameters(
            boards=(CustomBoardParameters(base_column_characters=2),)
        )

        result = allocate_custom_boards(parsed, parameters)

        self.assertEqual(len(result.boards), 1)
        self.assertEqual(result.ignored_pages, 1)
        first_character = next(
            item for item in result.boards[0].placements if item.character == "甲"
        )
        self.assertEqual(first_character.column, 1)

    def test_columns_at_or_above_threshold_keep_baseline_height(self) -> None:
        parsed = parse_custom_scripture(
            "甲乙丙\n甲乙丙丁戊己\n甲乙丙丁",
            set("甲乙丙丁戊己"),
        )
        board_parameters = CustomBoardParameters(
            cell_height_mm=10.0,
            base_row_gap_mm=2.0,
            base_column_characters=4,
        )
        result = allocate_custom_boards(
            parsed,
            CustomLayoutTemplateParameters(boards=(board_parameters,)),
        )

        metrics = result.geometries[0].column_metrics
        baseline = board_parameters.baseline_height_mm
        for metric in metrics:
            if metric.character_count < board_parameters.base_column_characters * 0.9:
                continue
            total = (
                metric.character_count * metric.cell_height_mm
                + (metric.character_count - 1) * metric.row_gap_mm
            )
            self.assertAlmostEqual(total, baseline, places=6)
        crowded = next(item for item in metrics if item.character_count == 6)
        self.assertEqual(crowded.row_gap_mm, 0.0)
        self.assertAlmostEqual(crowded.cell_height_mm, baseline / 6, places=6)

    def test_short_columns_inherit_previous_column_vertical_rule(self) -> None:
        parsed = parse_custom_scripture(
            "甲乙丙丁戊己庚辛壬\n甲乙丙丁戊己庚辛\n甲乙丙丁戊己庚",
            set("甲乙丙丁戊己庚辛壬"),
        )
        board_parameters = CustomBoardParameters(
            cell_height_mm=10.0,
            base_row_gap_mm=2.0,
            base_column_characters=10,
        )

        result = allocate_custom_boards(
            parsed,
            CustomLayoutTemplateParameters(boards=(board_parameters,)),
        )

        # 物理列从左到右保存，正文列从右到左显示，因此反转后才是输入顺序。
        first, second, third = reversed(result.geometries[0].column_metrics)
        self.assertEqual(
            [first.character_count, second.character_count, third.character_count],
            [9, 8, 7],
        )
        self.assertAlmostEqual(first.row_gap_mm, 3.5, places=6)
        self.assertEqual(second.cell_height_mm, first.cell_height_mm)
        self.assertEqual(second.row_gap_mm, first.row_gap_mm)
        self.assertEqual(third.cell_height_mm, first.cell_height_mm)
        self.assertEqual(third.row_gap_mm, first.row_gap_mm)

    def test_first_short_column_uses_default_rule_without_stretching(self) -> None:
        parsed = parse_custom_scripture(
            "甲乙丙丁戊己庚辛\n甲乙丙丁戊己庚",
            set("甲乙丙丁戊己庚辛"),
        )
        board_parameters = CustomBoardParameters(
            cell_height_mm=10.0,
            base_row_gap_mm=2.0,
            base_column_characters=10,
        )

        result = allocate_custom_boards(
            parsed,
            CustomLayoutTemplateParameters(boards=(board_parameters,)),
        )

        first, second = reversed(result.geometries[0].column_metrics)
        self.assertEqual(first.character_count, 8)
        self.assertEqual(first.cell_height_mm, 10.0)
        self.assertEqual(first.row_gap_mm, 2.0)
        self.assertEqual(second.character_count, 7)
        self.assertEqual(second.cell_height_mm, first.cell_height_mm)
        self.assertEqual(second.row_gap_mm, first.row_gap_mm)

    def test_last_column_below_baseline_count_inherits_previous_rule(self) -> None:
        parsed = parse_custom_scripture(
            "甲乙丙丁戊己庚辛壬癸\n甲乙丙丁戊己庚辛壬",
            set("甲乙丙丁戊己庚辛壬癸"),
        )
        board_parameters = CustomBoardParameters(
            cell_height_mm=10.0,
            base_row_gap_mm=2.0,
            base_column_characters=10,
        )

        result = allocate_custom_boards(
            parsed,
            CustomLayoutTemplateParameters(boards=(board_parameters,)),
        )

        first, last = reversed(result.geometries[0].column_metrics)
        self.assertEqual(first.character_count, 10)
        self.assertEqual(first.row_gap_mm, 2.0)
        self.assertEqual(last.character_count, 9)
        self.assertEqual(last.cell_height_mm, first.cell_height_mm)
        self.assertEqual(last.row_gap_mm, first.row_gap_mm)

    def test_last_column_at_or_above_baseline_count_uses_existing_rules(self) -> None:
        equal_parsed = parse_custom_scripture(
            "甲乙丙丁戊己庚辛壬\n甲乙丙丁戊己庚辛壬癸",
            set("甲乙丙丁戊己庚辛壬癸子丑"),
        )
        board_parameters = CustomBoardParameters(
            cell_height_mm=10.0,
            base_row_gap_mm=2.0,
            base_column_characters=10,
        )
        equal_result = allocate_custom_boards(
            equal_parsed,
            CustomLayoutTemplateParameters(boards=(board_parameters,)),
        )
        equal_last = equal_result.geometries[0].column_metrics[0]
        self.assertEqual(equal_last.character_count, 10)
        self.assertEqual(equal_last.cell_height_mm, 10.0)
        self.assertEqual(equal_last.row_gap_mm, 2.0)

        crowded_parsed = parse_custom_scripture(
            "甲乙丙丁戊己庚辛壬癸\n甲乙丙丁戊己庚辛壬癸子丑",
            set("甲乙丙丁戊己庚辛壬癸子丑"),
        )
        crowded_result = allocate_custom_boards(
            crowded_parsed,
            CustomLayoutTemplateParameters(boards=(board_parameters,)),
        )
        last = crowded_result.geometries[0].column_metrics[0]
        self.assertEqual(last.character_count, 12)
        self.assertEqual(last.row_gap_mm, 0.0)
        self.assertAlmostEqual(
            last.cell_height_mm,
            board_parameters.baseline_height_mm / 12,
            places=6,
        )

    def test_single_short_last_column_uses_default_rule(self) -> None:
        parsed = parse_custom_scripture(
            "甲乙丙丁戊己庚辛壬",
            set("甲乙丙丁戊己庚辛壬"),
        )
        board_parameters = CustomBoardParameters(
            cell_height_mm=10.0,
            base_row_gap_mm=2.0,
            base_column_characters=10,
        )

        result = allocate_custom_boards(
            parsed,
            CustomLayoutTemplateParameters(boards=(board_parameters,)),
        )

        metric = result.geometries[0].column_metrics[0]
        self.assertEqual(metric.character_count, 9)
        self.assertEqual(metric.cell_height_mm, 10.0)
        self.assertEqual(metric.row_gap_mm, 2.0)

    def test_punctuation_can_be_excluded_without_changing_page_boundaries(self) -> None:
        excluded = parse_custom_scripture("甲，乙\n\n丙。", None, False)
        included = parse_custom_scripture("甲，乙\n\n丙。", None, True)

        self.assertEqual(len(excluded.pages), 2)
        self.assertEqual(excluded.punctuation, 2)
        self.assertEqual(excluded.characters, 3)
        self.assertEqual(included.characters, 5)

    def test_custom_geometry_is_used_by_preview_and_layered_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph_path = Path(directory) / "甲.png"
            image = Image.new("RGBA", (24, 24), (0, 0, 0, 0))
            for x in range(6, 18):
                for y in range(3, 21):
                    image.putpixel((x, y), (0, 0, 0, 255))
            image.save(glyph_path, dpi=(300, 300))
            image.close()
            parsed = parse_custom_scripture("甲甲\n甲甲甲甲", {"甲"})
            template = CustomLayoutTemplateParameters(
                boards=(
                    CustomBoardParameters(
                        dpi=150,
                        cell_width_mm=8,
                        cell_height_mm=9,
                        base_column_characters=3,
                        base_row_gap_mm=1,
                    ),
                )
            )
            layout = allocate_custom_boards(parsed, template)
            glyph_index = GlyphIndex(
                "测试",
                {
                    "甲": (
                        GlyphImage("甲", str(glyph_path), 1, "甲", 24, 24, 300),
                    )
                },
            )

            preview = render_board_preview(
                layout.boards[0],
                glyph_index,
                layout.parameters[0],
                (800, 800),
                geometry=layout.geometries[0],
            )
            try:
                self.assertEqual(
                    preview.size,
                    (
                        layout.geometries[0].canvas_width,
                        layout.geometries[0].canvas_height,
                    ),
                )
                self.assertNotEqual(preview.getbbox(), None)
            finally:
                preview.close()

            generated = generate_psd_boards(
                layout.boards,
                glyph_index,
                layout.parameters[0],
                directory,
                board_parameters={1: layout.parameters[0]},
                board_geometries={1: layout.geometries[0]},
                output_base_name="定制测试",
            )
            self.assertFalse(generated.stopped)
            self.assertEqual(len(generated.boards), 1)
            self.assertTrue(Path(generated.boards[0].path).is_file())


if __name__ == "__main__":
    unittest.main()
