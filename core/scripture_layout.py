"""经文排版的参数、解析、分版与几何计算。"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping
import unicodedata


TOKEN_CHAR = "char"
TOKEN_SKIP = "skip"
TOKEN_BREAK = "break"

SCALE_BY_DPI = "按源图尺寸"
SCALE_TO_CELL = "相对单元格"
LEGACY_SCALE_BY_DPI = "按源图实际DPI"
PARAGRAPH_NEW_COLUMN = "段后换列"
PARAGRAPH_SKIP_CELLS = "段后跳格"
LAYOUT_VERTICAL = "竖排"
LAYOUT_HORIZONTAL = "横排"
FLOW_LEFT_TO_RIGHT = "从左到右"
FLOW_RIGHT_TO_LEFT = "从右到左"


@dataclass(frozen=True, slots=True)
class RowGapAdjustment:
    """指定行上方的特殊行距，行号从 2 开始。"""

    row: int
    gap_mm: float


@dataclass(frozen=True, slots=True)
class ColumnGapAdjustment:
    """指定列左侧的特殊列距，列号从 2 开始。"""

    column: int
    gap_mm: float


@dataclass(frozen=True, slots=True)
class LayoutParameters:
    """一套可保存为模板的版面参数。"""

    dpi: int = 150
    cell_width_mm: float = 13.0
    cell_height_mm: float = 11.0
    rows: int = 33
    columns: int = 40
    row_gap_mm: float = 1.2
    column_gap_mm: float = 1.0
    draw_outer_frame: bool = True
    frame_top_mm: float = 2.0
    frame_bottom_mm: float = 2.0
    frame_left_mm: float = 2.0
    frame_right_mm: float = 2.0
    canvas_top_mm: float = 8.0
    canvas_bottom_mm: float = 8.0
    canvas_left_mm: float = 8.0
    canvas_right_mm: float = 8.0
    include_punctuation: bool = False
    trim_empty_columns: bool = True
    layout_mode: str = LAYOUT_VERTICAL
    flow_direction: str = FLOW_RIGHT_TO_LEFT
    scale_mode: str = SCALE_BY_DPI
    scale_percent: int = 100
    cell_fill_percent: int = 95
    auto_scale_enabled: bool = True
    auto_enlarge_threshold: int = 75
    auto_enlarge_fill_percent: int = 95
    auto_shrink_threshold: int = 150
    auto_shrink_fill_percent: int = 95
    paragraph_mode: str = PARAGRAPH_SKIP_CELLS
    paragraph_skip_cells: int = 2
    first_title_new_column: bool = True
    last_title_new_column: bool = True
    add_annotations: bool = True
    special_gaps_enabled: bool = False
    row_gap_adjustments: tuple[RowGapAdjustment, ...] = field(default_factory=tuple)
    column_gap_adjustments: tuple[ColumnGapAdjustment, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        """拒绝会生成空版、无效尺寸或不可控超大画布的参数。"""

        if not 72 <= self.dpi <= 1200:
            raise ValueError("画布 DPI 必须在 72 到 1200 之间。")
        if not 1 <= self.rows <= 500 or not 1 <= self.columns <= 500:
            raise ValueError("行数和列数必须在 1 到 500 之间。")
        positive_values = {
            "单元格宽度": self.cell_width_mm,
            "单元格高度": self.cell_height_mm,
        }
        for label, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{label}必须大于 0。")
        nonnegative_values = {
            "行间距": self.row_gap_mm,
            "列间距": self.column_gap_mm,
            "大框上边距": self.frame_top_mm,
            "大框下边距": self.frame_bottom_mm,
            "大框左边距": self.frame_left_mm,
            "大框右边距": self.frame_right_mm,
            "画布上边距": self.canvas_top_mm,
            "画布下边距": self.canvas_bottom_mm,
            "画布左边距": self.canvas_left_mm,
            "画布右边距": self.canvas_right_mm,
        }
        for label, value in nonnegative_values.items():
            if value < 0:
                raise ValueError(f"{label}不能小于 0。")
        if self.draw_outer_frame:
            frame_canvas_pairs = (
                ("上", self.frame_top_mm, self.canvas_top_mm),
                ("下", self.frame_bottom_mm, self.canvas_bottom_mm),
                ("左", self.frame_left_mm, self.canvas_left_mm),
                ("右", self.frame_right_mm, self.canvas_right_mm),
            )
            for direction, frame_margin, canvas_margin in frame_canvas_pairs:
                if frame_margin > canvas_margin:
                    raise ValueError(
                        f"大框{direction}边距不能超过画布{direction}边距。"
                    )
        if self.scale_mode not in {SCALE_BY_DPI, SCALE_TO_CELL}:
            raise ValueError("字形缩放方式无效。")
        if self.layout_mode not in {LAYOUT_VERTICAL, LAYOUT_HORIZONTAL}:
            raise ValueError("排版方向无效。")
        if self.flow_direction not in {FLOW_LEFT_TO_RIGHT, FLOW_RIGHT_TO_LEFT}:
            raise ValueError("文字行进方向无效。")
        if self.paragraph_mode not in {
            PARAGRAPH_NEW_COLUMN,
            PARAGRAPH_SKIP_CELLS,
        }:
            raise ValueError("分段方式无效。")
        for label, value in {
            "整体缩放": self.scale_percent,
            "单元格填充": self.cell_fill_percent,
            "自动放大阈值": self.auto_enlarge_threshold,
            "自动放大目标": self.auto_enlarge_fill_percent,
            "自动缩小阈值": self.auto_shrink_threshold,
            "自动缩小目标": self.auto_shrink_fill_percent,
        }.items():
            if not 1 <= value <= 500:
                raise ValueError(f"{label}必须在 1% 到 500% 之间。")
        if self.paragraph_skip_cells < 0:
            raise ValueError("段后跳格数不能小于 0。")
        adjusted_rows: set[int] = set()
        for adjustment in self.row_gap_adjustments:
            if not 2 <= adjustment.row <= self.rows:
                raise ValueError("特殊行距的行号必须在 2 到总行数之间。")
            if adjustment.row in adjusted_rows:
                raise ValueError("同一行只能设置一次特殊行距。")
            if adjustment.gap_mm < 0:
                raise ValueError("特殊行距不能小于 0。")
            adjusted_rows.add(adjustment.row)
        adjusted_columns: set[int] = set()
        for adjustment in self.column_gap_adjustments:
            if not 2 <= adjustment.column <= self.columns:
                raise ValueError("特殊列距的列号必须在 2 到总列数之间。")
            if adjustment.column in adjusted_columns:
                raise ValueError("同一列只能设置一次特殊列距。")
            if adjustment.gap_mm < 0:
                raise ValueError("特殊列距不能小于 0。")
            adjusted_columns.add(adjustment.column)
        width, height = canvas_size_mm(self)
        px = self.dpi / 25.4
        if round(width * px) > 300_000 or round(height * px) > 300_000:
            raise ValueError("当前参数生成的画布过大，请降低尺寸、行列数或 DPI。")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["row_gap_adjustments"] = [
            asdict(item) for item in self.row_gap_adjustments
        ]
        result["column_gap_adjustments"] = [
            asdict(item) for item in self.column_gap_adjustments
        ]
        return result

    @classmethod
    def from_dict(cls, source: Mapping[str, Any]) -> "LayoutParameters":
        allowed = set(cls.__dataclass_fields__)
        values = {key: value for key, value in source.items() if key in allowed}
        values["row_gap_adjustments"] = tuple(
            RowGapAdjustment(
                row=int(item["row"]),
                gap_mm=float(item["gap_mm"]),
            )
            for item in values.get("row_gap_adjustments", ())
            if isinstance(item, Mapping)
        )
        values["column_gap_adjustments"] = tuple(
            ColumnGapAdjustment(
                column=int(item["column"]),
                gap_mm=float(item["gap_mm"]),
            )
            for item in values.get("column_gap_adjustments", ())
            if isinstance(item, Mapping)
        )
        if values.get("scale_mode") == LEGACY_SCALE_BY_DPI:
            values["scale_mode"] = SCALE_BY_DPI
        if "special_gaps_enabled" not in source:
            values["special_gaps_enabled"] = bool(
                values["row_gap_adjustments"]
                or values["column_gap_adjustments"]
            )
        result = cls(**values)
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class ScriptureToken:
    kind: str
    paragraph: int
    character: str = ""
    missing: bool = False
    last_break: bool = False


@dataclass(frozen=True, slots=True)
class ParsedScripture:
    tokens: tuple[ScriptureToken, ...]
    characters: int
    paragraphs: int
    missing: Mapping[str, int]
    ignored: Mapping[str, int]
    # 正文统计独立于排版是否包含标点：字数统计不把标点计入字数，
    # 但保留标点总数，便于界面明确显示本次排版是否排除标点。
    total_characters: int = 0
    unique_characters: int = 0
    punctuation: int = 0


@dataclass(frozen=True, slots=True)
class Placement:
    kind: str
    row: int
    column: int
    paragraph: int
    character: str = ""
    occurrence: int = 0
    missing: bool = False


@dataclass(frozen=True, slots=True)
class BoardLayout:
    number: int
    placements: tuple[Placement, ...]
    effective_columns: int
    effective_rows: int
    full: bool

    @property
    def character_count(self) -> int:
        return sum(item.kind == TOKEN_CHAR for item in self.placements)


@dataclass(frozen=True, slots=True)
class GridGeometry:
    canvas_width: int
    canvas_height: int
    cell_width: int
    cell_height: int
    row_tops: tuple[int, ...]
    column_lefts: tuple[int, ...]
    frame_rect: tuple[int, int, int, int] | None


def parse_scripture(
    text: str,
    available_characters: Iterable[str] | None = None,
    include_punctuation: bool = True,
) -> ParsedScripture:
    """解析正文；空格占格，控制字符忽略，其余可见字符都参与缺字检查。"""

    available = (
        set(available_characters) if available_characters is not None else None
    )
    paragraphs: list[list[ScriptureToken]] = []
    missing: Counter[str] = Counter()
    ignored: Counter[str] = Counter()
    total_characters = 0
    unique_characters: set[str] = set()
    punctuation = 0
    for raw_line in text.splitlines():
        paragraph_index = len(paragraphs)
        paragraph: list[ScriptureToken] = []
        for character in raw_line:
            if character in {" ", "\u3000", "\t"}:
                paragraph.append(
                    ScriptureToken(TOKEN_SKIP, paragraph_index)
                )
                continue
            if not character.isprintable() or character in {"\r", "\n"}:
                ignored[character] += 1
                continue
            is_punctuation = unicodedata.category(character).startswith("P")
            if is_punctuation:
                punctuation += 1
                if not include_punctuation:
                    continue
            else:
                total_characters += 1
                unique_characters.add(character)
            if is_punctuation and not include_punctuation:
                continue
            is_missing = available is not None and character not in available
            if is_missing:
                missing[character] += 1
            paragraph.append(
                ScriptureToken(
                    TOKEN_CHAR,
                    paragraph_index,
                    character=character,
                    missing=is_missing,
                )
            )
        if paragraph:
            paragraphs.append(paragraph)

    tokens: list[ScriptureToken] = []
    for paragraph_index, paragraph in enumerate(paragraphs):
        tokens.extend(paragraph)
        if paragraph_index < len(paragraphs) - 1:
            tokens.append(
                ScriptureToken(
                    TOKEN_BREAK,
                    paragraph_index,
                    last_break=paragraph_index == len(paragraphs) - 2,
                )
            )
    return ParsedScripture(
        tokens=tuple(tokens),
        characters=sum(token.kind == TOKEN_CHAR for token in tokens),
        paragraphs=len(paragraphs),
        missing=dict(sorted(missing.items())),
        ignored=dict(sorted(ignored.items())),
        total_characters=total_characters,
        unique_characters=len(unique_characters),
        punctuation=punctuation,
    )


def allocate_boards(
    parsed: ParsedScripture | Iterable[ScriptureToken],
    parameters: LayoutParameters,
) -> tuple[BoardLayout, ...]:
    """按排版方向和行进方向把经文分配到多块版面。"""

    parameters.validate()
    tokens = list(parsed.tokens if isinstance(parsed, ParsedScripture) else parsed)
    if not tokens:
        return ()
    occurrences: Counter[str] = Counter()
    boards: list[BoardLayout] = []
    token_index = 0
    horizontal = parameters.layout_mode == LAYOUT_HORIZONTAL
    track_limit = parameters.rows if horizontal else parameters.columns
    cells_per_track = parameters.columns if horizontal else parameters.rows
    while token_index < len(tokens):
        logical: list[tuple[ScriptureToken, int, int, int]] = []
        track = 0
        cell = 0
        start_index = token_index

        def advance(count: int = 1) -> None:
            nonlocal track, cell
            for _ in range(count):
                cell += 1
                if cell >= cells_per_track:
                    cell = 0
                    track += 1

        while token_index < len(tokens) and track < track_limit:
            token = tokens[token_index]
            if token.kind == TOKEN_BREAK:
                new_track = parameters.paragraph_mode == PARAGRAPH_NEW_COLUMN
                if (
                    token.paragraph == 0
                    and parameters.first_title_new_column
                ) or (token.last_break and parameters.last_title_new_column):
                    new_track = True
                if new_track:
                    if cell:
                        track += 1
                        cell = 0
                else:
                    for _ in range(parameters.paragraph_skip_cells):
                        if track >= track_limit:
                            break
                        logical_row, logical_column = (
                            (track, cell) if horizontal else (cell, track)
                        )
                        logical.append((token, logical_row, logical_column, 0))
                        advance()
                token_index += 1
                continue
            occurrence = occurrences[token.character]
            logical_row, logical_column = (
                (track, cell) if horizontal else (cell, track)
            )
            logical.append((token, logical_row, logical_column, occurrence))
            if token.kind == TOKEN_CHAR:
                occurrences[token.character] += 1
            advance()
            token_index += 1

        if token_index == start_index:
            raise RuntimeError("分版参数未能容纳任何经文内容。")
        used_tracks = max(track + (1 if cell else 0), 1)
        if parameters.trim_empty_columns and logical:
            effective_rows = used_tracks if horizontal else parameters.rows
            effective_columns = parameters.columns if horizontal else used_tracks
        else:
            effective_rows = parameters.rows
            effective_columns = parameters.columns

        def physical_column(logical_column: int) -> int:
            if parameters.flow_direction == FLOW_RIGHT_TO_LEFT:
                return effective_columns - 1 - logical_column
            return logical_column

        placements = tuple(
            Placement(
                kind=(TOKEN_SKIP if token.kind == TOKEN_BREAK else token.kind),
                row=item_row,
                column=physical_column(item_column),
                paragraph=token.paragraph,
                character=token.character,
                occurrence=occurrence,
                missing=token.missing,
            )
            for token, item_row, item_column, occurrence in logical
        )
        boards.append(
            BoardLayout(
                number=len(boards) + 1,
                placements=placements,
                effective_columns=effective_columns,
                effective_rows=effective_rows,
                full=track >= track_limit,
            )
        )
    return tuple(boards)


def row_gap_mm(parameters: LayoutParameters, row_index: int) -> float:
    """返回指定行上方的行距，row_index 使用从 0 开始的行索引。"""

    row_number = row_index + 1
    if parameters.special_gaps_enabled:
        for adjustment in parameters.row_gap_adjustments:
            if adjustment.row == row_number:
                return adjustment.gap_mm
    return parameters.row_gap_mm


def column_gap_mm(parameters: LayoutParameters, column_index: int) -> float:
    """返回指定列左侧的列距，column_index 使用从 0 开始的列索引。"""

    column_number = column_index + 1
    if parameters.special_gaps_enabled:
        for adjustment in parameters.column_gap_adjustments:
            if adjustment.column == column_number:
                return adjustment.gap_mm
    return parameters.column_gap_mm


def canvas_size_mm(
    parameters: LayoutParameters,
    effective_columns: int | None = None,
    effective_rows: int | None = None,
) -> tuple[float, float]:
    columns = effective_columns or parameters.columns
    rows = effective_rows or parameters.rows
    content_width = columns * parameters.cell_width_mm + sum(
        column_gap_mm(parameters, column) for column in range(1, columns)
    )
    content_height = rows * parameters.cell_height_mm + sum(
        row_gap_mm(parameters, row) for row in range(1, rows)
    )
    return (
        parameters.canvas_left_mm + content_width + parameters.canvas_right_mm,
        parameters.canvas_top_mm + content_height + parameters.canvas_bottom_mm,
    )


def compute_grid(
    parameters: LayoutParameters,
    effective_columns: int | None = None,
    effective_rows: int | None = None,
) -> GridGeometry:
    parameters.validate()
    columns = effective_columns or parameters.columns
    rows = effective_rows or parameters.rows
    px = parameters.dpi / 25.4
    cell_width = max(1, round(parameters.cell_width_mm * px))
    cell_height = max(1, round(parameters.cell_height_mm * px))
    origin_x_mm = parameters.canvas_left_mm
    origin_y_mm = parameters.canvas_top_mm
    row_tops = [round(origin_y_mm * px)]
    for row in range(1, rows):
        row_tops.append(
            row_tops[-1] + cell_height + round(row_gap_mm(parameters, row) * px)
        )
    column_lefts = [round(origin_x_mm * px)]
    for column in range(1, columns):
        column_lefts.append(
            column_lefts[-1]
            + cell_width
            + round(column_gap_mm(parameters, column) * px)
        )
    width_mm, height_mm = canvas_size_mm(parameters, columns, rows)
    frame_rect = None
    if parameters.draw_outer_frame:
        left = round((parameters.canvas_left_mm - parameters.frame_left_mm) * px)
        top = round((parameters.canvas_top_mm - parameters.frame_top_mm) * px)
        right = column_lefts[-1] + cell_width + round(parameters.frame_right_mm * px)
        bottom = row_tops[-1] + cell_height + round(parameters.frame_bottom_mm * px)
        frame_rect = (
            left,
            top,
            right,
            bottom,
        )
    return GridGeometry(
        canvas_width=max(1, round(width_mm * px)),
        canvas_height=max(1, round(height_mm * px)),
        cell_width=cell_width,
        cell_height=cell_height,
        row_tops=tuple(row_tops),
        column_lefts=tuple(column_lefts),
        frame_rect=frame_rect,
    )


def target_image_size(
    source_width: int,
    source_height: int,
    source_dpi: float,
    grid: GridGeometry,
    parameters: LayoutParameters,
) -> tuple[int, int]:
    """按模板规则等比计算单个字图的目标尺寸。"""

    if source_width <= 0 or source_height <= 0:
        raise ValueError("字图尺寸无效。")
    if parameters.scale_mode == SCALE_BY_DPI:
        safe_source_dpi = source_dpi if source_dpi > 0 else parameters.dpi
        scale = parameters.dpi / safe_source_dpi
        scale *= parameters.scale_percent / 100
    else:
        scale = min(
            grid.cell_width / source_width,
            grid.cell_height / source_height,
        ) * parameters.cell_fill_percent / 100
    width = source_width * scale
    height = source_height * scale
    if parameters.scale_mode == SCALE_BY_DPI and parameters.auto_scale_enabled:
        occupied = max(width / grid.cell_width, height / grid.cell_height) * 100
        if occupied <= parameters.auto_enlarge_threshold:
            scale *= parameters.auto_enlarge_fill_percent / max(occupied, 0.001)
        elif occupied >= parameters.auto_shrink_threshold:
            scale *= parameters.auto_shrink_fill_percent / occupied
        width = source_width * scale
        height = source_height * scale
    return max(1, round(width)), max(1, round(height))
