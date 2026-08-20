"""经文排版纯核心回归测试。"""

from __future__ import annotations

import unittest

from core.scripture_layout import (
    FLOW_LEFT_TO_RIGHT,
    FLOW_RIGHT_TO_LEFT,
    LAYOUT_HORIZONTAL,
    LAYOUT_VERTICAL,
    SCALE_BY_DPI,
    ColumnGapAdjustment,
    PARAGRAPH_NEW_COLUMN,
    LayoutParameters,
    RowGapAdjustment,
    allocate_boards,
    canvas_size_mm,
    compute_grid,
    parse_scripture,
    target_image_size,
)


class ScriptureLayoutCoreTests(unittest.TestCase):
    def test_parse_reports_missing_and_keeps_spaces(self) -> None:
        parsed = parse_scripture("甲 乙，\n丙\x00", {"甲", "乙", "丙"})

        self.assertEqual(parsed.characters, 4)
        self.assertEqual(parsed.total_characters, 3)
        self.assertEqual(parsed.unique_characters, 3)
        self.assertEqual(parsed.punctuation, 1)
        self.assertEqual(parsed.paragraphs, 2)
        self.assertEqual(parsed.missing, {"，": 1})
        self.assertEqual(parsed.ignored, {"\x00": 1})
        self.assertEqual(sum(token.kind == "skip" for token in parsed.tokens), 1)

        without_punctuation = parse_scripture(
            "甲，乙。",
            {"甲", "乙"},
            include_punctuation=False,
        )
        self.assertEqual(without_punctuation.characters, 2)
        self.assertEqual(without_punctuation.total_characters, 2)
        self.assertEqual(without_punctuation.unique_characters, 2)
        self.assertEqual(without_punctuation.punctuation, 2)
        self.assertEqual(without_punctuation.missing, {})
        self.assertEqual(
            [token.character for token in without_punctuation.tokens],
            ["甲", "乙"],
        )

    def test_allocate_is_vertical_and_right_to_left(self) -> None:
        parameters = LayoutParameters(
            rows=2,
            columns=2,
            first_title_new_column=False,
            last_title_new_column=False,
            trim_empty_columns=False,
        )
        boards = allocate_boards(parse_scripture("甲乙丙丁戊"), parameters)

        self.assertEqual(len(boards), 2)
        first = boards[0].placements
        self.assertEqual(
            [(item.character, item.row, item.column) for item in first],
            [("甲", 0, 1), ("乙", 1, 1), ("丙", 0, 0), ("丁", 1, 0)],
        )
        self.assertEqual(boards[1].placements[0].character, "戊")

    def test_allocate_supports_vertical_and_horizontal_flow_directions(self) -> None:
        common = dict(
            rows=2,
            columns=3,
            first_title_new_column=False,
            last_title_new_column=False,
            trim_empty_columns=False,
        )
        cases = (
            (
                LAYOUT_VERTICAL,
                FLOW_LEFT_TO_RIGHT,
                [(0, 0), (1, 0), (0, 1), (1, 1)],
            ),
            (
                LAYOUT_VERTICAL,
                FLOW_RIGHT_TO_LEFT,
                [(0, 2), (1, 2), (0, 1), (1, 1)],
            ),
            (
                LAYOUT_HORIZONTAL,
                FLOW_LEFT_TO_RIGHT,
                [(0, 0), (0, 1), (0, 2), (1, 0)],
            ),
            (
                LAYOUT_HORIZONTAL,
                FLOW_RIGHT_TO_LEFT,
                [(0, 2), (0, 1), (0, 0), (1, 2)],
            ),
        )
        for layout_mode, flow_direction, expected in cases:
            with self.subTest(layout_mode=layout_mode, flow_direction=flow_direction):
                parameters = LayoutParameters(
                    **common,
                    layout_mode=layout_mode,
                    flow_direction=flow_direction,
                )
                board = allocate_boards(parse_scripture("甲乙丙丁"), parameters)[0]
                self.assertEqual(
                    [(item.row, item.column) for item in board.placements],
                    expected,
                )

    def test_horizontal_last_board_trims_empty_rows(self) -> None:
        parameters = LayoutParameters(
            rows=3,
            columns=2,
            layout_mode=LAYOUT_HORIZONTAL,
            flow_direction=FLOW_LEFT_TO_RIGHT,
            first_title_new_column=False,
            last_title_new_column=False,
            trim_empty_columns=True,
        )

        board = allocate_boards(parse_scripture("甲乙丙"), parameters)[0]
        grid = compute_grid(
            parameters,
            board.effective_columns,
            board.effective_rows,
        )

        self.assertEqual(board.effective_rows, 2)
        self.assertEqual(board.effective_columns, 2)
        self.assertEqual(len(grid.row_tops), 2)

    def test_horizontal_paragraph_new_track_starts_a_new_row(self) -> None:
        parameters = LayoutParameters(
            rows=2,
            columns=3,
            layout_mode=LAYOUT_HORIZONTAL,
            flow_direction=FLOW_LEFT_TO_RIGHT,
            paragraph_mode=PARAGRAPH_NEW_COLUMN,
            first_title_new_column=False,
            last_title_new_column=False,
            trim_empty_columns=False,
        )

        board = allocate_boards(parse_scripture("甲乙\n丙"), parameters)[0]

        self.assertEqual(
            [(item.character, item.row, item.column) for item in board.placements],
            [("甲", 0, 0), ("乙", 0, 1), ("丙", 1, 0)],
        )

    def test_paragraph_new_column_and_variant_occurrence_continue_across_boards(self) -> None:
        parameters = LayoutParameters(
            rows=3,
            columns=2,
            paragraph_mode=PARAGRAPH_NEW_COLUMN,
            first_title_new_column=False,
            last_title_new_column=False,
            trim_empty_columns=False,
        )
        boards = allocate_boards(parse_scripture("甲乙\n甲甲甲甲甲"), parameters)

        occurrences = [
            item.occurrence
            for board in boards
            for item in board.placements
            if item.character == "甲"
        ]
        self.assertEqual(occurrences, [0, 1, 2, 3, 4, 5])
        self.assertEqual(boards[0].placements[2].column, 0)

    def test_grid_and_target_size_are_stable(self) -> None:
        parameters = LayoutParameters(
            dpi=254,
            cell_width_mm=10,
            cell_height_mm=20,
            rows=2,
            columns=2,
            row_gap_mm=1,
            column_gap_mm=2,
            frame_top_mm=1,
            frame_bottom_mm=1,
            frame_left_mm=1,
            frame_right_mm=1,
            canvas_top_mm=2,
            canvas_bottom_mm=2,
            canvas_left_mm=2,
            canvas_right_mm=2,
        )
        grid = compute_grid(parameters)

        self.assertEqual((grid.cell_width, grid.cell_height), (100, 200))
        self.assertEqual(canvas_size_mm(parameters), (26, 45))
        self.assertEqual((grid.canvas_width, grid.canvas_height), (260, 450))
        self.assertEqual(grid.frame_rect, (10, 10, 250, 440))
        self.assertEqual(target_image_size(50, 100, 254, grid, parameters), (95, 190))

    def test_special_row_and_column_gaps_only_apply_when_enabled(self) -> None:
        disabled = LayoutParameters(
            dpi=254,
            cell_width_mm=10,
            cell_height_mm=10,
            rows=3,
            columns=3,
            row_gap_mm=1,
            column_gap_mm=1,
            draw_outer_frame=False,
            canvas_top_mm=0,
            canvas_bottom_mm=0,
            canvas_left_mm=0,
            canvas_right_mm=0,
            special_gaps_enabled=False,
            row_gap_adjustments=(RowGapAdjustment(2, 4),),
            column_gap_adjustments=(ColumnGapAdjustment(2, 5),),
        )
        enabled = LayoutParameters.from_dict(
            {**disabled.to_dict(), "special_gaps_enabled": True}
        )

        self.assertEqual(canvas_size_mm(disabled), (32, 32))
        self.assertEqual(canvas_size_mm(enabled), (36, 35))

    def test_source_size_auto_scale_can_be_disabled(self) -> None:
        common = dict(
            dpi=100,
            cell_width_mm=25.4,
            cell_height_mm=25.4,
            rows=1,
            columns=1,
            draw_outer_frame=False,
            canvas_top_mm=0,
            canvas_bottom_mm=0,
            canvas_left_mm=0,
            canvas_right_mm=0,
            scale_mode=SCALE_BY_DPI,
        )
        automatic = LayoutParameters(**common, auto_scale_enabled=True)
        manual = LayoutParameters(**common, auto_scale_enabled=False)
        grid = compute_grid(automatic)

        self.assertEqual(target_image_size(50, 50, 100, grid, automatic), (95, 95))
        self.assertEqual(target_image_size(50, 50, 100, grid, manual), (50, 50))

    def test_legacy_source_scale_name_is_migrated(self) -> None:
        parameters = LayoutParameters.from_dict(
            {"scale_mode": "按源图实际DPI"}
        )

        self.assertEqual(parameters.scale_mode, SCALE_BY_DPI)


if __name__ == "__main__":
    unittest.main()
