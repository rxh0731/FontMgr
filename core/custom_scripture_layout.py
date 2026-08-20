"""定制经文排版的空行分版、逐列等高计算与版面几何。"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping
import unicodedata

from core.scripture_layout import (
    FLOW_RIGHT_TO_LEFT,
    LAYOUT_VERTICAL,
    PARAGRAPH_NEW_COLUMN,
    SCALE_BY_DPI,
    TOKEN_CHAR,
    TOKEN_SKIP,
    BoardLayout,
    LayoutParameters,
    Placement,
    ScriptureToken,
)


@dataclass(frozen=True, slots=True)
class CustomBoardParameters:
    """单块定制版面的尺寸参数。"""

    dpi: int = 300
    cell_width_mm: float = 18.0
    cell_height_mm: float = 20.0
    base_row_gap_mm: float = 2.0
    base_column_characters: int = 21
    column_gap_mm: float = 3.0
    draw_outer_frame: bool = True
    frame_top_mm: float = 2.0
    frame_bottom_mm: float = 2.0
    frame_left_mm: float = 2.0
    frame_right_mm: float = 2.0
    canvas_top_mm: float = 8.0
    canvas_bottom_mm: float = 8.0
    canvas_left_mm: float = 8.0
    canvas_right_mm: float = 8.0

    def validate(self) -> None:
        if not 72 <= self.dpi <= 1200:
            raise ValueError("画布 DPI 必须在 72 到 1200 之间。")
        if not 1 <= self.base_column_characters <= 500:
            raise ValueError("基准列字数必须在 1 到 500 之间。")
        if self.cell_width_mm <= 0 or self.cell_height_mm <= 0:
            raise ValueError("单元格宽度和高度必须大于 0。")
        for label, value in {
            "基准行间距": self.base_row_gap_mm,
            "固定列间距": self.column_gap_mm,
            "大框上边距": self.frame_top_mm,
            "大框下边距": self.frame_bottom_mm,
            "大框左边距": self.frame_left_mm,
            "大框右边距": self.frame_right_mm,
            "画布上边距": self.canvas_top_mm,
            "画布下边距": self.canvas_bottom_mm,
            "画布左边距": self.canvas_left_mm,
            "画布右边距": self.canvas_right_mm,
        }.items():
            if value < 0:
                raise ValueError(f"{label}不能小于 0。")
        if self.draw_outer_frame:
            for direction, frame_margin, canvas_margin in (
                ("上", self.frame_top_mm, self.canvas_top_mm),
                ("下", self.frame_bottom_mm, self.canvas_bottom_mm),
                ("左", self.frame_left_mm, self.canvas_left_mm),
                ("右", self.frame_right_mm, self.canvas_right_mm),
            ):
                if frame_margin > canvas_margin:
                    raise ValueError(
                        f"大框{direction}边距不能超过画布{direction}边距。"
                    )

    @property
    def baseline_height_mm(self) -> float:
        return (
            self.base_column_characters * self.cell_height_mm
            + max(0, self.base_column_characters - 1) * self.base_row_gap_mm
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, source: Mapping[str, Any]) -> "CustomBoardParameters":
        allowed = set(cls.__dataclass_fields__)
        result = cls(**{key: value for key, value in source.items() if key in allowed})
        result.validate()
        return result

    def as_layout_parameters(
        self,
        *,
        rows: int,
        columns: int,
        include_punctuation: bool,
        add_annotations: bool,
    ) -> LayoutParameters:
        """生成共享字图缩放和 PSD 输出所需的兼容参数。"""

        self.validate()
        return LayoutParameters(
            dpi=self.dpi,
            cell_width_mm=self.cell_width_mm,
            cell_height_mm=self.cell_height_mm,
            rows=max(1, rows),
            columns=max(1, columns),
            row_gap_mm=self.base_row_gap_mm,
            column_gap_mm=self.column_gap_mm,
            draw_outer_frame=self.draw_outer_frame,
            frame_top_mm=self.frame_top_mm,
            frame_bottom_mm=self.frame_bottom_mm,
            frame_left_mm=self.frame_left_mm,
            frame_right_mm=self.frame_right_mm,
            canvas_top_mm=self.canvas_top_mm,
            canvas_bottom_mm=self.canvas_bottom_mm,
            canvas_left_mm=self.canvas_left_mm,
            canvas_right_mm=self.canvas_right_mm,
            include_punctuation=include_punctuation,
            trim_empty_columns=True,
            layout_mode=LAYOUT_VERTICAL,
            flow_direction=FLOW_RIGHT_TO_LEFT,
            scale_mode=SCALE_BY_DPI,
            scale_percent=100,
            auto_scale_enabled=False,
            paragraph_mode=PARAGRAPH_NEW_COLUMN,
            first_title_new_column=False,
            last_title_new_column=False,
            add_annotations=add_annotations,
            special_gaps_enabled=False,
        )


@dataclass(frozen=True, slots=True)
class CustomLayoutTemplateParameters:
    """可保存为模板的全部定制版面参数。"""

    boards: tuple[CustomBoardParameters, ...] = field(
        default_factory=lambda: (CustomBoardParameters(),)
    )
    include_punctuation: bool = False
    add_annotations: bool = False

    def validate(self) -> None:
        if not self.boards:
            raise ValueError("至少需要保留一块版面参数。")
        if len(self.boards) > 200:
            raise ValueError("单个模板最多保存 200 块版面参数。")
        for board in self.boards:
            board.validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "boards": [board.to_dict() for board in self.boards],
            "include_punctuation": self.include_punctuation,
            "add_annotations": self.add_annotations,
        }

    @classmethod
    def from_dict(
        cls,
        source: Mapping[str, Any],
    ) -> "CustomLayoutTemplateParameters":
        raw_boards = source.get("boards", ())
        if not isinstance(raw_boards, (list, tuple)):
            raise ValueError("模板中的版面参数列表无效。")
        result = cls(
            boards=tuple(
                CustomBoardParameters.from_dict(item)
                for item in raw_boards
                if isinstance(item, Mapping)
            ),
            include_punctuation=bool(source.get("include_punctuation", False)),
            add_annotations=bool(source.get("add_annotations", False)),
        )
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class CustomScripturePage:
    number: int
    columns: tuple[tuple[ScriptureToken, ...], ...]

    @property
    def character_count(self) -> int:
        return sum(
            token.kind == TOKEN_CHAR
            for column in self.columns
            for token in column
        )


@dataclass(frozen=True, slots=True)
class ParsedCustomScripture:
    pages: tuple[CustomScripturePage, ...]
    missing: Mapping[str, int]
    ignored: Mapping[str, int]
    total_characters: int
    unique_characters: int
    punctuation: int

    @property
    def characters(self) -> int:
        return sum(page.character_count for page in self.pages)


@dataclass(frozen=True, slots=True)
class CellGeometry:
    row: int
    column: int
    left: int
    top: int
    width: int
    height: int

    @property
    def rect(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.width, self.height


@dataclass(frozen=True, slots=True)
class CustomColumnMetric:
    column: int
    character_count: int
    cell_height_mm: float
    row_gap_mm: float


@dataclass(frozen=True, slots=True)
class CustomGridGeometry:
    canvas_width: int
    canvas_height: int
    cell_width: int
    cell_height: int
    frame_rect: tuple[int, int, int, int] | None
    cells: tuple[CellGeometry, ...]
    column_metrics: tuple[CustomColumnMetric, ...]

    def cell_rect(self, row: int, column: int) -> tuple[int, int, int, int]:
        for cell in self.cells:
            if cell.row == row and cell.column == column:
                return cell.rect
        raise IndexError(f"找不到第 {column + 1} 列第 {row + 1} 个单元格。")


@dataclass(frozen=True, slots=True)
class CustomLayoutResult:
    boards: tuple[BoardLayout, ...]
    geometries: tuple[CustomGridGeometry, ...]
    parameters: tuple[LayoutParameters, ...]
    ignored_pages: int


def parse_custom_scripture(
    text: str,
    available_characters: Iterable[str] | None = None,
    include_punctuation: bool = False,
) -> ParsedCustomScripture:
    """按空行分版，并把每个非空正文行解释为一列。"""

    available = set(available_characters) if available_characters is not None else None
    raw_pages: list[list[str]] = []
    current_page: list[str] = []
    for raw_line in text.splitlines():
        if not raw_line.strip():
            if current_page:
                raw_pages.append(current_page)
                current_page = []
            continue
        current_page.append(raw_line)
    if current_page:
        raw_pages.append(current_page)

    missing: Counter[str] = Counter()
    ignored: Counter[str] = Counter()
    unique_characters: set[str] = set()
    total_characters = 0
    punctuation = 0
    pages: list[CustomScripturePage] = []
    for page_index, raw_columns in enumerate(raw_pages):
        columns: list[tuple[ScriptureToken, ...]] = []
        for column_index, raw_column in enumerate(raw_columns):
            tokens: list[ScriptureToken] = []
            for character in raw_column:
                if character in {" ", "\u3000", "\t"}:
                    tokens.append(ScriptureToken(TOKEN_SKIP, column_index))
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
                is_missing = available is not None and character not in available
                if is_missing:
                    missing[character] += 1
                tokens.append(
                    ScriptureToken(
                        TOKEN_CHAR,
                        column_index,
                        character=character,
                        missing=is_missing,
                    )
                )
            if tokens:
                columns.append(tuple(tokens))
        if columns:
            pages.append(CustomScripturePage(page_index + 1, tuple(columns)))
    return ParsedCustomScripture(
        pages=tuple(pages),
        missing=dict(sorted(missing.items())),
        ignored=dict(sorted(ignored.items())),
        total_characters=total_characters,
        unique_characters=len(unique_characters),
        punctuation=punctuation,
    )


def _column_dimensions(
    character_count: int,
    parameters: CustomBoardParameters,
) -> tuple[float, float, float]:
    """返回单元格高、行距和相对内容顶边。"""

    baseline = parameters.baseline_height_mm
    if character_count <= 0:
        return parameters.cell_height_mm, 0.0, 0.0
    if character_count == 1:
        return (
            parameters.cell_height_mm,
            0.0,
            max(0.0, (baseline - parameters.cell_height_mm) / 2.0),
        )
    gap = (
        baseline - character_count * parameters.cell_height_mm
    ) / (character_count - 1)
    if gap >= 0:
        return parameters.cell_height_mm, gap, 0.0
    return baseline / character_count, 0.0, 0.0


def compute_custom_grid(
    board: BoardLayout,
    parameters: CustomBoardParameters,
) -> CustomGridGeometry:
    """根据每列实际字数生成自适应行距、固定列距的像素几何。"""

    parameters.validate()
    columns = max(1, board.effective_columns)
    px = parameters.dpi / 25.4
    cell_width = max(1, round(parameters.cell_width_mm * px))
    baseline_height = parameters.baseline_height_mm
    content_width = (
        columns * parameters.cell_width_mm
        + max(0, columns - 1) * parameters.column_gap_mm
    )
    canvas_width = max(
        1,
        round(
            (parameters.canvas_left_mm + content_width + parameters.canvas_right_mm)
            * px
        ),
    )
    canvas_height = max(
        1,
        round(
            (
                parameters.canvas_top_mm
                + baseline_height
                + parameters.canvas_bottom_mm
            )
            * px
        ),
    )
    counts = [0] * columns
    for placement in board.placements:
        counts[placement.column] = max(counts[placement.column], placement.row + 1)
    dimensions: list[tuple[float, float, float]] = [
        (parameters.cell_height_mm, parameters.base_row_gap_mm, 0.0)
    ] * columns
    previous_rule: tuple[float, float] | None = None
    short_column_limit = parameters.base_column_characters * 0.9
    # 正文第一列位于最右侧，因此按物理列号倒序计算“前一列”的继承关系。
    for column in range(columns - 1, -1, -1):
        count = counts[column]
        is_last_text_column = column == 0
        inherits_previous = (
            count < parameters.base_column_characters
            if is_last_text_column
            else count < short_column_limit
        )
        if inherits_previous:
            height_mm, gap_mm = previous_rule or (
                parameters.cell_height_mm,
                parameters.base_row_gap_mm,
            )
            dimensions[column] = (height_mm, gap_mm, 0.0)
        else:
            dimensions[column] = _column_dimensions(count, parameters)
        height_mm, gap_mm, _top_offset_mm = dimensions[column]
        previous_rule = (height_mm, gap_mm)
    cells: list[CellGeometry] = []
    metrics: list[CustomColumnMetric] = []
    for column, count in enumerate(counts):
        height_mm, gap_mm, top_offset_mm = dimensions[column]
        height = max(1, round(height_mm * px))
        left = round(
            (
                parameters.canvas_left_mm
                + column * (parameters.cell_width_mm + parameters.column_gap_mm)
            )
            * px
        )
        top = round((parameters.canvas_top_mm + top_offset_mm) * px)
        gap = max(0, round(gap_mm * px))
        for row in range(count):
            cells.append(CellGeometry(row, column, left, top, cell_width, height))
            top += height + gap
        metrics.append(CustomColumnMetric(column, count, height_mm, gap_mm))
    frame_rect = None
    if parameters.draw_outer_frame:
        frame_rect = (
            round((parameters.canvas_left_mm - parameters.frame_left_mm) * px),
            round((parameters.canvas_top_mm - parameters.frame_top_mm) * px),
            round(
                (
                    parameters.canvas_left_mm
                    + content_width
                    + parameters.frame_right_mm
                )
                * px
            ),
            round(
                (
                    parameters.canvas_top_mm
                    + baseline_height
                    + parameters.frame_bottom_mm
                )
                * px
            ),
        )
    return CustomGridGeometry(
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        cell_width=cell_width,
        cell_height=max(1, round(parameters.cell_height_mm * px)),
        frame_rect=frame_rect,
        cells=tuple(cells),
        column_metrics=tuple(metrics),
    )


def allocate_custom_boards(
    parsed: ParsedCustomScripture,
    parameters: CustomLayoutTemplateParameters,
) -> CustomLayoutResult:
    """只匹配已有参数的正文版，多出的正文按确认规则忽略。"""

    parameters.validate()
    matched = min(len(parsed.pages), len(parameters.boards))
    occurrences: Counter[str] = Counter()
    boards: list[BoardLayout] = []
    geometries: list[CustomGridGeometry] = []
    compatible_parameters: list[LayoutParameters] = []
    for page_index in range(matched):
        page = parsed.pages[page_index]
        board_parameters = parameters.boards[page_index]
        column_count = max(1, len(page.columns))
        placements: list[Placement] = []
        for logical_column, tokens in enumerate(page.columns):
            physical_column = column_count - 1 - logical_column
            for row, token in enumerate(tokens):
                occurrence = occurrences[token.character] if token.kind == TOKEN_CHAR else 0
                placements.append(
                    Placement(
                        kind=token.kind,
                        row=row,
                        column=physical_column,
                        paragraph=logical_column,
                        character=token.character,
                        occurrence=occurrence,
                        missing=token.missing,
                    )
                )
                if token.kind == TOKEN_CHAR:
                    occurrences[token.character] += 1
        board = BoardLayout(
            number=page_index + 1,
            placements=tuple(placements),
            effective_columns=column_count,
            effective_rows=max((len(column) for column in page.columns), default=1),
            full=True,
        )
        compatible = board_parameters.as_layout_parameters(
            rows=board.effective_rows,
            columns=board.effective_columns,
            include_punctuation=parameters.include_punctuation,
            add_annotations=parameters.add_annotations,
        )
        boards.append(board)
        geometries.append(compute_custom_grid(board, board_parameters))
        compatible_parameters.append(compatible)
    return CustomLayoutResult(
        boards=tuple(boards),
        geometries=tuple(geometries),
        parameters=tuple(compatible_parameters),
        ignored_pages=max(0, len(parsed.pages) - len(parameters.boards)),
    )
