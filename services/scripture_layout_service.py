"""经文字图索引、高清预览与分层 PSD 输出服务。"""

from __future__ import annotations

import gc
import multiprocessing
import os
import queue
import re
import shutil
import tempfile
import threading
import time
import traceback
import unicodedata
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

import config
from core.custom_scripture_layout import CustomGridGeometry
from core.scripture_layout import (
    FLOW_RIGHT_TO_LEFT,
    LAYOUT_HORIZONTAL,
    PARAGRAPH_SKIP_CELLS,
    SCALE_BY_DPI,
    BoardLayout,
    GridGeometry,
    LayoutParameters,
    TOKEN_CHAR,
    compute_grid,
)
from services.glyph_service import GlyphService
from data.log_manager import write_log
from services.workflow_status_service import (
    PHASE_COORDINATION,
    project_stage_status,
    resolve_safe_stage_file,
)
from utils.file_utils import natural_key, pinyin_natural_key
from utils.system_resources import get_system_memory_status


SUPPORTED_IMAGE_EXTENSIONS = frozenset(
    {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".webp", ".psd"}
)
CONFLICT_OVERWRITE = "覆盖"
CONFLICT_SKIP = "跳过"
CONFLICT_CANCEL = "取消"
OUTPUT_FORMAT_AUTO = "自动（推荐）"
OUTPUT_FORMAT_PSD = "PSD"
OUTPUT_FORMAT_PSB = "PSB"
PSD_MAX_DIMENSION = 30_000
PSB_MAX_DIMENSION = 300_000
PSD_SAFE_FILE_BYTES = 1_800_000_000
ABSOLUTE_PEAK_MEMORY_LIMIT = 8 * 1024 * 1024 * 1024
DEFAULT_EXTERNAL_SOURCE_DPI = 300.0
MINIMUM_DISK_WORKSPACE_BYTES = 64 * 1024 * 1024
DISK_WORKSPACE_SAFETY_FACTOR = 1.10
MINIMUM_GLYPH_CACHE_BYTES = 8 * 1024 * 1024
PREVIEW_GLYPH_CACHE_LIMIT = 64 * 1024 * 1024
OUTPUT_GLYPH_CACHE_LIMIT = 192 * 1024 * 1024

LayoutGrid = GridGeometry | CustomGridGeometry


class GenerationCancelled(RuntimeError):
    """用户安全停止排版任务。"""


@dataclass(frozen=True, slots=True)
class GlyphImage:
    character: str
    path: str
    version: int
    display_name: str
    source_width: int = 0
    source_height: int = 0
    source_dpi: float = 0.0
    source_width_mm: float = 0.0
    source_height_mm: float = 0.0


@dataclass(frozen=True, slots=True)
class GlyphIndex:
    source_name: str
    images: Mapping[str, tuple[GlyphImage, ...]]
    issues: tuple[str, ...] = ()

    @property
    def characters(self) -> frozenset[str]:
        return frozenset(self.images)

    @property
    def variant_count(self) -> int:
        return sum(len(items) for items in self.images.values())

    def resolve(self, character: str, occurrence: int) -> GlyphImage | None:
        variants = self.images.get(character, ())
        if not variants:
            return None
        return variants[occurrence % len(variants)]


@dataclass(frozen=True, slots=True)
class GenerationProgress:
    completed: int
    total: int
    message: str
    indeterminate: bool = False


@dataclass(frozen=True, slots=True)
class GeneratedBoard:
    board_number: int
    path: str
    characters: int
    skipped: bool = False
    missing_characters: int = 0


@dataclass(frozen=True, slots=True)
class BoardOutputPlan:
    board_number: int
    format_name: str
    extension: str
    psb: bool
    width: int
    height: int
    estimated_file_bytes: int
    estimated_peak_bytes: int
    memory_budget_bytes: int = 0
    memory_warning: str = ""


@dataclass(frozen=True, slots=True)
class GenerationResult:
    boards: tuple[GeneratedBoard, ...]
    stopped: bool


@dataclass(frozen=True, slots=True)
class _BoardProcessResult:
    placed: int
    missing_characters: int
    source_layer_pixels: int
    retained_layer_pixels: int
    stage_times: tuple[tuple[str, float], ...]


def _probe_image(path: str) -> tuple[int, int, float, float]:
    with Image.open(path) as image:
        width, height = image.size
        dpi_value = image.info.get("dpi", (0, 0))
    if width <= 0 or height <= 0:
        raise ValueError("图片尺寸无效")
    if isinstance(dpi_value, (tuple, list)) and dpi_value:
        source_dpi_x = float(dpi_value[0] or 0)
        source_dpi_y = float(
            dpi_value[1] or source_dpi_x
            if len(dpi_value) > 1
            else source_dpi_x
        )
    else:
        source_dpi_x = float(dpi_value or 0)
        source_dpi_y = source_dpi_x
    return width, height, source_dpi_x, source_dpi_y


def _physical_size_from_resolution(
    width: int,
    height: int,
    dpi_x: float,
    dpi_y: float,
) -> tuple[float, float]:
    """把图片分辨率元数据换算为物理尺寸；未知轴保持为零。"""

    width_mm = width / dpi_x * 25.4 if dpi_x > 0 else 0.0
    height_mm = height / dpi_y * 25.4 if dpi_y > 0 else 0.0
    return width_mm, height_mm


def _system_glyph_physical_size(
    metadata: Mapping[str, Any],
    width: int,
    height: int,
    dpi_x: float,
    dpi_y: float,
) -> tuple[float, float]:
    """按字库设定的标准画布毫米尺寸推导当前成品的实际物理尺寸。"""

    try:
        canvas_width = int(metadata.get("画布宽", 0))
        canvas_height = int(metadata.get("画布高", 0))
        configured_width_mm = float(metadata.get("成品宽度毫米", 0))
        configured_height_mm = float(metadata.get("成品高度毫米", 0))
    except (TypeError, ValueError):
        canvas_width = canvas_height = 0
        configured_width_mm = configured_height_mm = 0.0
    width_mm = (
        width * configured_width_mm / canvas_width
        if canvas_width > 0 and configured_width_mm > 0
        else 0.0
    )
    height_mm = (
        height * configured_height_mm / canvas_height
        if canvas_height > 0 and configured_height_mm > 0
        else 0.0
    )
    fallback_width_mm, fallback_height_mm = _physical_size_from_resolution(
        width,
        height,
        dpi_x,
        dpi_y,
    )
    return width_mm or fallback_width_mm, height_mm or fallback_height_mm


def _is_real_regular_file(path: str) -> bool:
    is_junction = getattr(os.path, "isjunction", lambda _path: False)
    return (
        os.path.isfile(path)
        and not os.path.islink(path)
        and not is_junction(path)
    )


def build_system_glyph_index(
    glyph_service: GlyphService,
    *,
    progress_callback: Callable[[GenerationProgress], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> GlyphIndex:
    """索引系统字库中真实存在且可读取的成品字图。

    墨色统一是整体协调的辅助契约，不应影响通用经文排版对字库字符的
    缺字核对。只要记录为成品且成品文件有效，就可以作为排版来源；
    这样墨色统一尚未完成的字库不会被误报为“全部缺字”。
    """

    variants = glyph_service.get_variants()
    groups = glyph_service.get_glyph_groups()
    metadata = glyph_service.get_metadata()
    summary = glyph_service.get_coordination_summary()
    finished_dir = glyph_service.get_workflow_dirs()["成品"]
    indexed: dict[str, tuple[GlyphImage, ...]] = {}
    issues: list[str] = []
    total = sum(len(items) for items in groups.values())
    completed = 0
    last_notification = 0.0

    def cancelled() -> bool:
        return bool(cancel_check and cancel_check())

    def notify(message: str, *, done: bool = False) -> None:
        nonlocal last_notification
        now = time.monotonic()
        if not done and completed > 0 and now - last_notification < 0.05:
            return
        last_notification = now
        if progress_callback:
            progress_callback(
                GenerationProgress(
                    max(1, total) if done else completed,
                    max(1, total),
                    message,
                )
            )

    notify(f"正在核对系统字库，共 {total} 个字形")
    for character in sorted(groups, key=pinyin_natural_key):
        images: list[GlyphImage] = []
        for ordinal, variant_id in enumerate(groups[character]):
            if cancelled():
                raise GenerationCancelled("用户停止了字图检查。")
            detail = variants.get(variant_id)
            if not isinstance(detail, Mapping):
                issues.append(f"{character} / {variant_id}：字形记录不存在")
                completed += 1
                notify(f"正在核对：{character} / {variant_id}")
                continue
            projection = project_stage_status(
                detail,
                summary,
                finished_dir,
                PHASE_COORDINATION,
            )
            if not projection.has_valid_finished:
                completed += 1
                notify(f"正在核对：{character} / {variant_id}")
                continue
            path = resolve_safe_stage_file(finished_dir, detail.get("成品文件"))
            if not path:
                issues.append(f"{character} / {variant_id}：成品文件缺失或路径不安全")
                completed += 1
                notify(f"正在核对：{character} / {variant_id}")
                continue
            try:
                source_width, source_height, source_dpi, source_dpi_y = _probe_image(path)
            except (OSError, ValueError, UnidentifiedImageError) as exc:
                issues.append(f"{character} / {os.path.basename(path)}：{exc}")
                completed += 1
                notify(f"正在解码：{character} / {os.path.basename(path)}")
                continue
            try:
                version = int(detail.get("变体序号", ordinal + 1))
            except (TypeError, ValueError):
                version = ordinal + 1
            source_width_mm, source_height_mm = _system_glyph_physical_size(
                metadata,
                source_width,
                source_height,
                source_dpi,
                source_dpi_y,
            )
            images.append(
                GlyphImage(
                    character,
                    path,
                    version,
                    os.path.basename(path),
                    source_width,
                    source_height,
                    source_dpi,
                    source_width_mm,
                    source_height_mm,
                )
            )
            completed += 1
            notify(f"正在归类：{character} / {os.path.basename(path)}")
        if images:
            indexed[character] = tuple(
                sorted(images, key=lambda item: (item.version, natural_key(item.display_name)))
            )
    notify(
        f"系统字库核对完成：{len(indexed)} 个字，{sum(len(items) for items in indexed.values())} 个字形",
        done=True,
    )
    return GlyphIndex(glyph_service.ziku_name, indexed, tuple(issues))


_EXTERNAL_VERSION_PATTERN = re.compile(r"(?:[-_ ]?)(?P<version>\d+)$")


def build_external_glyph_index(
    directory: str,
    *,
    progress_callback: Callable[[GenerationProgress], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> GlyphIndex:
    """只读扫描外部字图目录，按文件名首字归类为字形变体。"""

    root = os.path.abspath(directory)
    is_junction = getattr(os.path, "isjunction", lambda _path: False)
    if not os.path.isdir(root) or os.path.islink(root) or is_junction(root):
        raise ValueError("外部字库目录不存在，或目录是链接/联接。")
    grouped: dict[str, list[GlyphImage]] = defaultdict(list)
    issues: list[str] = []
    used_versions: defaultdict[str, set[int]] = defaultdict(set)
    next_versions: defaultdict[str, int] = defaultdict(int)
    if progress_callback:
        progress_callback(GenerationProgress(0, 1, "正在扫描外部字库目录…", True))
    with os.scandir(root) as iterator:
        entries = sorted(iterator, key=lambda item: natural_key(item.name))
    total = len(entries)
    completed = 0
    last_notification = 0.0

    def notify(message: str, *, done: bool = False) -> None:
        nonlocal last_notification
        now = time.monotonic()
        if not done and completed > 0 and now - last_notification < 0.05:
            return
        last_notification = now
        if progress_callback:
            progress_callback(
                GenerationProgress(
                    max(1, total) if done else completed,
                    max(1, total),
                    message,
                )
            )

    if progress_callback:
        progress_callback(
            GenerationProgress(0, max(1, total), f"发现 {total} 个目录项目，正在核对图片")
        )
    for entry in entries:
        if cancel_check and cancel_check():
            raise GenerationCancelled("用户停止了字图检查。")
        extension = Path(entry.name).suffix.lower()
        if extension not in SUPPORTED_IMAGE_EXTENSIONS:
            completed += 1
            notify(f"正在扫描：{entry.name}")
            continue
        if not entry.is_file(follow_symlinks=False) or not _is_real_regular_file(entry.path):
            issues.append(f"{entry.name}：不是可安全读取的普通文件")
            completed += 1
            notify(f"正在核对：{entry.name}")
            continue
        stem = Path(entry.name).stem
        if not stem or not stem[0].isprintable() or stem[0].isspace():
            issues.append(f"{entry.name}：文件名首字不能用于字形归类")
            completed += 1
            notify(f"正在核对：{entry.name}")
            continue
        character = stem[0]
        version_match = _EXTERNAL_VERSION_PATTERN.search(stem[1:])
        desired_version = int(version_match.group("version")) if version_match else None
        if desired_version is not None and desired_version not in used_versions[character]:
            version = desired_version
        else:
            version = next_versions[character]
            while version in used_versions[character]:
                version += 1
            if desired_version is not None:
                issues.append(
                    f"{entry.name}：{character} 的编号 {desired_version} 重复，"
                    f"已按文件顺序作为变体 {version} 载入"
                )
        try:
            source_width, source_height, source_dpi, source_dpi_y = _probe_image(
                entry.path
            )
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            issues.append(f"{entry.name}：图片损坏或不受支持（{exc}）")
            completed += 1
            notify(f"正在解码：{entry.name}")
            continue
        used_versions[character].add(version)
        next_versions[character] = max(next_versions[character], version + 1)
        source_width_mm, source_height_mm = _physical_size_from_resolution(
            source_width,
            source_height,
            source_dpi,
            source_dpi_y,
        )
        if source_width_mm <= 0 or source_height_mm <= 0:
            fallback_width_mm, fallback_height_mm = _physical_size_from_resolution(
                source_width,
                source_height,
                DEFAULT_EXTERNAL_SOURCE_DPI,
                DEFAULT_EXTERNAL_SOURCE_DPI,
            )
            source_width_mm = source_width_mm or fallback_width_mm
            source_height_mm = source_height_mm or fallback_height_mm
            issues.append(
                f"{entry.name}：缺少完整物理尺寸元数据，"
                f"已按 {DEFAULT_EXTERNAL_SOURCE_DPI:g} DPI 换算"
            )
        grouped[character].append(
            GlyphImage(
                character,
                entry.path,
                version,
                entry.name,
                source_width,
                source_height,
                source_dpi,
                source_width_mm,
                source_height_mm,
            )
        )
        completed += 1
        notify(f"正在归类：{entry.name}")
    indexed = {
        character: tuple(
            sorted(items, key=lambda item: (item.version, natural_key(item.display_name)))
        )
        for character, items in sorted(grouped.items(), key=lambda pair: pinyin_natural_key(pair[0]))
    }
    notify(
        f"外部字库核对完成：{len(indexed)} 个字，{sum(len(items) for items in indexed.values())} 个字形",
        done=True,
    )
    return GlyphIndex(os.path.basename(root) or root, indexed, tuple(issues))


def _prepare_glyph_image(path: str) -> tuple[Image.Image, float]:
    """读取字图并将无透明通道的白底图转换为保留灰度墨色的透明字形。"""

    with Image.open(path) as source:
        source.load()
        dpi_value = source.info.get("dpi", (0, 0))
        source_dpi = (
            float(dpi_value[0] or 0)
            if isinstance(dpi_value, (tuple, list)) and dpi_value
            else float(dpi_value or 0)
        )
        has_alpha = source.mode in {"RGBA", "LA"} or "transparency" in source.info
        rgba = source.convert("RGBA")
    if has_alpha:
        return rgba, source_dpi
    luminance = rgba.convert("L")
    alpha = luminance.point(lambda value: 255 - value)
    result = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    result.putalpha(alpha)
    rgba.close()
    luminance.close()
    return result, source_dpi


class _GlyphBitmapCache:
    """按实际像素字节控制的任务内 LRU，所有位图由缓存统一关闭。"""

    def __init__(self, byte_limit: int) -> None:
        self.byte_limit = max(0, int(byte_limit))
        self.byte_size = 0
        self._items: OrderedDict[tuple[Any, ...], tuple[Image.Image, int]] = (
            OrderedDict()
        )

    def get(self, key: tuple[Any, ...]) -> Image.Image | None:
        item = self._items.get(key)
        if item is None:
            return None
        self._items.move_to_end(key)
        return item[0]

    def put(self, key: tuple[Any, ...], image: Image.Image) -> bool:
        cost = max(1, image.width * image.height * len(image.getbands()))
        if cost > self.byte_limit:
            return False
        old = self._items.pop(key, None)
        if old is not None:
            self.byte_size -= old[1]
            old[0].close()
        while self._items and self.byte_size + cost > self.byte_limit:
            _old_key, (old_image, old_cost) = self._items.popitem(last=False)
            self.byte_size -= old_cost
            old_image.close()
        self._items[key] = (image, cost)
        self.byte_size += cost
        return True

    def close(self) -> None:
        for image, _cost in self._items.values():
            image.close()
        self._items.clear()
        self.byte_size = 0


def _glyph_cache_budget(maximum_bytes: int) -> int:
    """缓存最多占当前可用内存的 5%，并受场景上限约束。"""

    _total_memory, available_memory = get_system_memory_status()
    available_share = max(0, int(available_memory)) // 20
    return max(
        MINIMUM_GLYPH_CACHE_BYTES,
        min(int(maximum_bytes), available_share or MINIMUM_GLYPH_CACHE_BYTES),
    )


def _glyph_bitmap_cache_key(
    image_ref: GlyphImage,
    target_width: int,
    target_height: int,
) -> tuple[Any, ...]:
    try:
        stat_result = os.stat(image_ref.path)
        stamp = (stat_result.st_mtime_ns, stat_result.st_size)
    except OSError:
        stamp = (0, 0)
    return (
        os.path.normcase(os.path.abspath(image_ref.path)),
        *stamp,
        int(target_width),
        int(target_height),
    )


def _visible_alpha_bounds(image: Image.Image) -> tuple[int, int, int, int] | None:
    """返回非全透明像素范围，供 PSD 在定位后丢弃外围透明画布。"""

    alpha = image.getchannel("A")
    try:
        return alpha.getbbox()
    finally:
        alpha.close()


def _install_composite_preview(
    psd: Any,
    composite_image: Image.Image,
    compression: Any,
) -> None:
    """写入已完成的整版预览，避免保存时再次遍历全部 PSD 图层。"""

    preview = composite_image.convert(psd.pil_mode)
    channels = preview.split()
    try:
        psd._record.image_data.compression = compression
        psd._record.image_data.set_data(
            [channel.tobytes() for channel in channels],
            psd._record.header,
        )
        psd._updated = False
    finally:
        for channel in channels:
            channel.close()
        preview.close()


def _iter_cell_rects(grid: LayoutGrid) -> tuple[tuple[int, int, int, int], ...]:
    """统一枚举普通网格和定制逐列网格中的单元格。"""

    if isinstance(grid, CustomGridGeometry):
        return tuple(cell.rect for cell in grid.cells)
    return tuple(
        (column_left, row_top, grid.cell_width, grid.cell_height)
        for row_top in grid.row_tops
        for column_left in grid.column_lefts
    )


def _placement_cell_rect(
    grid: LayoutGrid,
    row: int,
    column: int,
) -> tuple[int, int, int, int]:
    if isinstance(grid, CustomGridGeometry):
        return grid.cell_rect(row, column)
    return (
        grid.column_lefts[column],
        grid.row_tops[row],
        grid.cell_width,
        grid.cell_height,
    )


def _target_image_size_for_cell(
    source_width: int,
    source_height: int,
    source_dpi: float,
    cell_width: int,
    cell_height: int,
    parameters: LayoutParameters,
    *,
    source_width_mm: float = 0.0,
    source_height_mm: float = 0.0,
    use_physical_size: bool = False,
) -> tuple[int, int]:
    """按指定单元格计算字图尺寸，支持定制版面的逐列高度。"""

    if source_width <= 0 or source_height <= 0:
        raise ValueError("字图尺寸无效。")
    if use_physical_size:
        if source_width_mm <= 0 or source_height_mm <= 0:
            source_width_mm, source_height_mm = _physical_size_from_resolution(
                source_width,
                source_height,
                DEFAULT_EXTERNAL_SOURCE_DPI,
                DEFAULT_EXTERNAL_SOURCE_DPI,
            )
        pixels_per_mm = parameters.dpi / 25.4
        width = source_width_mm * pixels_per_mm
        height = source_height_mm * pixels_per_mm
    elif parameters.scale_mode == SCALE_BY_DPI:
        safe_source_dpi = source_dpi if source_dpi > 0 else parameters.dpi
        scale = parameters.dpi / safe_source_dpi
        scale *= parameters.scale_percent / 100
        width = source_width * scale
        height = source_height * scale
    else:
        scale = min(cell_width / source_width, cell_height / source_height)
        scale *= parameters.cell_fill_percent / 100
        width = source_width * scale
        height = source_height * scale
    if (
        not use_physical_size
        and parameters.scale_mode == SCALE_BY_DPI
        and parameters.auto_scale_enabled
    ):
        occupied = max(width / cell_width, height / cell_height) * 100
        if occupied <= parameters.auto_enlarge_threshold:
            scale *= parameters.auto_enlarge_fill_percent / max(occupied, 0.001)
        elif occupied >= parameters.auto_shrink_threshold:
            scale *= parameters.auto_shrink_fill_percent / occupied
        width = source_width * scale
        height = source_height * scale
    return max(1, round(width)), max(1, round(height))


def render_board_preview(
    board: BoardLayout,
    glyph_index: GlyphIndex,
    parameters: LayoutParameters,
    maximum_size: tuple[int, int] = (1100, 760),
    *,
    geometry: LayoutGrid | None = None,
    show_guides: bool = True,
    progress_callback: Callable[[GenerationProgress], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> Image.Image:
    """按请求尺寸直接绘制当前版，避免放大低清缩略图。"""

    grid = geometry or compute_grid(
        parameters,
        board.effective_columns,
        board.effective_rows,
    )
    scale = min(
        maximum_size[0] / grid.canvas_width,
        maximum_size[1] / grid.canvas_height,
        1.0,
    )
    preview_width = max(1, round(grid.canvas_width * scale))
    preview_height = max(1, round(grid.canvas_height * scale))
    preview = Image.new("RGB", (preview_width, preview_height), "white")
    draw = ImageDraw.Draw(preview)
    character_placements = tuple(
        placement for placement in board.placements if placement.kind == TOKEN_CHAR
    )
    total_steps = max(
        1,
        (
            len(_iter_cell_rects(grid))
            if show_guides
            else 0
        )
        + len(character_placements),
    )
    completed_steps = 0
    last_notification = 0.0

    def check_cancelled() -> None:
        if cancel_check and cancel_check():
            preview.close()
            raise GenerationCancelled("预览任务已取消。")

    def notify(message: str) -> None:
        nonlocal last_notification
        now = time.monotonic()
        if completed_steps < total_steps and now - last_notification < 0.05:
            return
        last_notification = now
        if progress_callback:
            progress_callback(
                GenerationProgress(completed_steps, total_steps, message)
            )

    def point(value: int) -> int:
        return round(value * scale)

    if show_guides:
        for cell_left, cell_top, cell_width, cell_height in _iter_cell_rects(grid):
            check_cancelled()
            left = point(cell_left)
            top = point(cell_top)
            right = point(cell_left + cell_width)
            bottom = point(cell_top + cell_height)
            draw.rectangle(
                (left, top, right, bottom),
                outline=(155, 74, 74),
                width=1,
            )
            draw.line(
                ((left + right) // 2, top, (left + right) // 2, bottom),
                fill=(220, 160, 160),
            )
            draw.line(
                (left, (top + bottom) // 2, right, (top + bottom) // 2),
                fill=(220, 160, 160),
            )
            completed_steps += 1
            notify("正在绘制田字格")
        if grid.frame_rect:
            draw.rectangle(
                tuple(point(value) for value in grid.frame_rect),
                outline=(125, 40, 40),
                width=1,
            )

    glyph_cache = _GlyphBitmapCache(
        _glyph_cache_budget(PREVIEW_GLYPH_CACHE_LIMIT)
    )
    try:
        for placement in character_placements:
            check_cancelled()
            cell_left, cell_top, cell_width, cell_height = _placement_cell_rect(
                grid,
                placement.row,
                placement.column,
            )
            left = point(cell_left)
            top = point(cell_top)
            image_ref = glyph_index.resolve(placement.character, placement.occurrence)
            if image_ref is None:
                right = point(cell_left + cell_width)
                bottom = point(cell_top + cell_height)
                draw.rectangle((left + 2, top + 2, right - 2, bottom - 2), outline=(204, 45, 45), width=2)
                completed_steps += 1
                notify(f"正在标记缺字：{placement.character}")
                continue
            glyph: Image.Image | None = None
            resized: Image.Image | None = None
            cached = False
            try:
                source_width = image_ref.source_width
                source_height = image_ref.source_height
                source_dpi = image_ref.source_dpi
                if source_width <= 0 or source_height <= 0:
                    glyph, decoded_dpi = _prepare_glyph_image(image_ref.path)
                    source_width, source_height = glyph.size
                    source_dpi = source_dpi or decoded_dpi
                target_width, target_height = _target_image_size_for_cell(
                    source_width,
                    source_height,
                    source_dpi,
                    cell_width,
                    cell_height,
                    parameters,
                    source_width_mm=image_ref.source_width_mm,
                    source_height_mm=image_ref.source_height_mm,
                    use_physical_size=isinstance(grid, CustomGridGeometry),
                )
                target_width = max(1, round(target_width * scale))
                target_height = max(1, round(target_height * scale))
                cache_key = _glyph_bitmap_cache_key(
                    image_ref,
                    target_width,
                    target_height,
                )
                resized = glyph_cache.get(cache_key)
                if resized is None:
                    if glyph is None:
                        glyph, _decoded_dpi = _prepare_glyph_image(image_ref.path)
                    resized = glyph.resize(
                        (target_width, target_height),
                        Image.Resampling.LANCZOS,
                    )
                    cached = glyph_cache.put(cache_key, resized)
                else:
                    cached = True
                if glyph is not None:
                    glyph.close()
                    glyph = None
                x = left + (point(cell_width) - target_width) // 2
                y = top + (point(cell_height) - target_height) // 2
                preview.paste(resized, (x, y), resized)
            except (OSError, ValueError, UnidentifiedImageError):
                draw.line((left, top, left + point(cell_width), top + point(cell_height)), fill=(204, 45, 45), width=2)
            finally:
                if glyph is not None:
                    glyph.close()
                if resized is not None and not cached:
                    resized.close()
            completed_steps += 1
            notify(f"正在绘制字图：{placement.character}")
    finally:
        glyph_cache.close()
    if progress_callback:
        progress_callback(GenerationProgress(total_steps, total_steps, "高清预览已完成"))
    return preview


def plan_board_output(
    board: BoardLayout,
    glyph_index: GlyphIndex,
    parameters: LayoutParameters,
    output_format: str = OUTPUT_FORMAT_AUTO,
    *,
    geometry: LayoutGrid | None = None,
    memory_status: tuple[int, int] | None = None,
) -> BoardOutputPlan:
    """预估单版文件格式和峰值内存，不分配完整画布。"""

    if output_format not in {OUTPUT_FORMAT_AUTO, OUTPUT_FORMAT_PSD, OUTPUT_FORMAT_PSB}:
        raise ValueError("输出格式无效。")
    grid = geometry or compute_grid(
        parameters,
        board.effective_columns,
        board.effective_rows,
    )
    width, height = grid.canvas_width, grid.canvas_height
    if width > PSB_MAX_DIMENSION or height > PSB_MAX_DIMENSION:
        raise ValueError(
            f"第 {board.number} 版画布达到 {width}×{height}，超过 PSB 单边 "
            f"{PSB_MAX_DIMENSION:,} 像素的格式上限。"
        )

    canvas_pixels = width * height
    glyph_pixels = 0
    for placement in board.placements:
        if placement.kind != TOKEN_CHAR:
            continue
        image_ref = glyph_index.resolve(placement.character, placement.occurrence)
        if image_ref is None:
            continue
        _left, _top, cell_width, cell_height = _placement_cell_rect(
            grid,
            placement.row,
            placement.column,
        )
        if image_ref.source_width > 0 and image_ref.source_height > 0:
            target_width, target_height = _target_image_size_for_cell(
                image_ref.source_width,
                image_ref.source_height,
                image_ref.source_dpi,
                cell_width,
                cell_height,
                parameters,
                source_width_mm=image_ref.source_width_mm,
                source_height_mm=image_ref.source_height_mm,
                use_physical_size=isinstance(grid, CustomGridGeometry),
            )
        else:
            target_width, target_height = cell_width, cell_height
        glyph_pixels += max(1, target_width) * max(1, target_height)

    full_canvas_layers = 2 + int(parameters.add_annotations)
    layer_pixels = canvas_pixels * full_canvas_layers + glyph_pixels
    estimated_file_bytes = canvas_pixels * 4 + layer_pixels * 4 + 4 * 1024 * 1024
    estimated_peak_bytes = (
        canvas_pixels * 24
        + layer_pixels * 6
        + 192 * 1024 * 1024
    )
    requires_psb = (
        width > PSD_MAX_DIMENSION
        or height > PSD_MAX_DIMENSION
        or estimated_file_bytes >= PSD_SAFE_FILE_BYTES
    )
    if output_format == OUTPUT_FORMAT_PSD and requires_psb:
        reason = (
            "画布单边超过 30,000 像素"
            if width > PSD_MAX_DIMENSION or height > PSD_MAX_DIMENSION
            else "预计未压缩数据接近 PSD 的 2 GB 上限"
        )
        raise ValueError(f"第 {board.number} 版{reason}，不能强制使用 PSD，请改用自动或 PSB。")
    psb = output_format == OUTPUT_FORMAT_PSB or (
        output_format == OUTPUT_FORMAT_AUTO and requires_psb
    )

    total_memory, available_memory = memory_status or get_system_memory_status()
    memory_budget = min(
        ABSOLUTE_PEAK_MEMORY_LIMIT,
        max(256 * 1024 * 1024, int(max(1, total_memory) * 0.40)),
        max(256 * 1024 * 1024, int(max(1, available_memory) * 0.70)),
    )
    memory_warning = ""
    if estimated_peak_bytes > memory_budget:
        memory_warning = (
            f"第 {board.number} 版预计需要约 {estimated_peak_bytes / 1024**3:.2f} GiB "
            f"峰值内存，当前安全额度约 {memory_budget / 1024**3:.2f} GiB。"
            "程序仍会逐版在独立进程中生成，并在每版后释放内存。"
        )
    return BoardOutputPlan(
        board.number,
        OUTPUT_FORMAT_PSB if psb else OUTPUT_FORMAT_PSD,
        ".psb" if psb else ".psd",
        psb,
        width,
        height,
        estimated_file_bytes,
        estimated_peak_bytes,
        memory_budget,
        memory_warning,
    )


def board_output_path(
    output_dir: str,
    board_number: int,
    dpi: int,
    base_name: str = "通用经文排版",
    *,
    total_boards: int = 1,
    extension: str = ".psd",
) -> str:
    """根据用户文件名生成 PSD/PSB 路径，多版时追加逐版编号。"""

    del dpi
    if board_number < 1 or total_boards < 1 or board_number > total_boards:
        raise ValueError("版号或总版数无效。")
    stem = str(base_name or "").strip()
    if stem.lower().endswith((".psd", ".psb")):
        stem = stem[:-4].rstrip()
    if not stem:
        raise ValueError("输出文件名不能为空。")
    if any(character in stem for character in '\\/:*?"<>|'):
        raise ValueError("输出文件名包含 Windows 不允许的字符。")
    if stem.endswith("."):
        raise ValueError("输出文件名不能以句点结尾。")
    extension = str(extension).lower()
    if extension not in {".psd", ".psb"}:
        raise ValueError("输出文件扩展名必须是 .psd 或 .psb。")
    if total_boards > 1:
        width = max(2, len(str(total_boards)))
        stem = f"{stem}-{board_number:0{width}d}"
    return os.path.join(output_dir, f"{stem}{extension}")


def _generate_board_process(
    board: BoardLayout,
    glyph_index: GlyphIndex,
    parameters: LayoutParameters,
    geometry: LayoutGrid | None,
    temporary_path: str,
    compress_psd: bool,
    psb: bool,
    total_jobs: int,
    message_queue: Any,
) -> None:
    """在可终止的独立进程中生成单版 PSD 临时文件。"""

    psd: Any | None = None
    composite_image: Image.Image | None = None
    glyph_cache: _GlyphBitmapCache | None = None
    try:
        import struct

        from psd_tools import PSDImage
        from psd_tools.constants import Compression, Resource
        from psd_tools.psd.image_resources import ImageResource

        stage_times: dict[str, float] = {}
        stage_started = time.perf_counter()
        plan = plan_board_output(
            board,
            glyph_index,
            parameters,
            OUTPUT_FORMAT_PSB if psb else OUTPUT_FORMAT_PSD,
            geometry=geometry,
        )
        grid = geometry or compute_grid(
            parameters,
            board.effective_columns,
            board.effective_rows,
        )
        message_queue.put(("progress", f"正在创建第 {board.number} 版", 1, False))
        layer_compression = Compression.RLE if compress_psd else Compression.RAW
        psd = PSDImage.new(
            "RGBA",
            (grid.canvas_width, grid.canvas_height),
            color=(255, 255, 255, 255),
            compression=layer_compression,
        )
        if plan.psb:
            psd._record.header.version = 2
        composite_image = Image.new(
            "RGB", (grid.canvas_width, grid.canvas_height), "white"
        )
        resolution = int(parameters.dpi * 0x10000)
        resolution_data = struct.pack(
            ">IHHIHH", resolution, 1, 2, resolution, 1, 2
        )
        psd.image_resources[Resource.RESOLUTION_INFO] = ImageResource(
            signature=b"8BIM",
            key=Resource.RESOLUTION_INFO,
            name="",
            data=resolution_data,
        )
        background = Image.new("RGB", (grid.canvas_width, grid.canvas_height), "white")
        psd.create_pixel_layer(
            name="背景",
            image=background,
            top=0,
            left=0,
            compression=layer_compression,
        )
        background.close()
        frame_group = psd.create_group(name="框线", open_folder=False)
        scripture_group = psd.create_group(name="经文", open_folder=False)
        stage_times["画布创建"] = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        frame = Image.new(
            "RGBA", (grid.canvas_width, grid.canvas_height), (0, 0, 0, 0)
        )
        frame_draw = ImageDraw.Draw(frame)
        for cell_left, cell_top, cell_width, cell_height in _iter_cell_rects(grid):
            rect = (
                cell_left,
                cell_top,
                cell_left + cell_width,
                cell_top + cell_height,
            )
            frame_draw.rectangle(rect, outline=(139, 0, 0, 255), width=1)
            center_x = cell_left + cell_width // 2
            center_y = cell_top + cell_height // 2
            frame_draw.line(
                (center_x, cell_top, center_x, cell_top + cell_height),
                fill=(190, 105, 105, 180),
                width=1,
            )
            frame_draw.line(
                (cell_left, center_y, cell_left + cell_width, center_y),
                fill=(190, 105, 105, 180),
                width=1,
            )
        if grid.frame_rect:
            frame_draw.rectangle(grid.frame_rect, outline=(139, 0, 0, 255), width=1)
        frame_layer = psd.create_pixel_layer(
            name="版面框线",
            image=frame,
            top=0,
            left=0,
            compression=layer_compression,
        )
        frame_layer.move_to_group(frame_group)
        composite_image.paste(frame, (0, 0), frame)
        frame.close()
        message_queue.put(
            ("progress", f"第 {board.number} 版框线已完成", 1, False)
        )
        stage_times["框线"] = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        paragraph_groups: dict[int, Any] = {}
        glyph_cache = _GlyphBitmapCache(
            _glyph_cache_budget(OUTPUT_GLYPH_CACHE_LIMIT)
        )
        placed = 0
        missing_characters = 0
        processed_characters = 0
        source_layer_pixels = 0
        retained_layer_pixels = 0
        for placement in board.placements:
            if placement.kind != TOKEN_CHAR:
                continue
            processed_characters += 1
            image_ref = glyph_index.resolve(placement.character, placement.occurrence)
            if image_ref is None:
                missing_characters += 1
                message_queue.put(
                    (
                        "progress",
                        f"第 {board.number} 版：缺字“{placement.character}”已留空 "
                        f"({processed_characters}/{board.character_count})",
                        1,
                        False,
                    )
                )
                continue
            cell_left, cell_top, cell_width, cell_height = _placement_cell_rect(
                grid,
                placement.row,
                placement.column,
            )
            source_width = image_ref.source_width
            source_height = image_ref.source_height
            source_dpi = image_ref.source_dpi
            glyph: Image.Image | None = None
            if source_width <= 0 or source_height <= 0:
                glyph, decoded_dpi = _prepare_glyph_image(image_ref.path)
                source_width, source_height = glyph.size
                source_dpi = source_dpi or decoded_dpi
            target_width, target_height = _target_image_size_for_cell(
                source_width,
                source_height,
                source_dpi,
                cell_width,
                cell_height,
                parameters,
                source_width_mm=image_ref.source_width_mm,
                source_height_mm=image_ref.source_height_mm,
                use_physical_size=isinstance(grid, CustomGridGeometry),
            )
            cache_key = _glyph_bitmap_cache_key(
                image_ref,
                target_width,
                target_height,
            )
            layer_image = glyph_cache.get(cache_key)
            cached = layer_image is not None
            visible_offset = (0, 0)
            if layer_image is None:
                if glyph is None:
                    glyph, _decoded_dpi = _prepare_glyph_image(image_ref.path)
                resized = glyph.resize(
                    (target_width, target_height), Image.Resampling.LANCZOS
                )
                glyph.close()
                glyph = None
                visible_bounds = _visible_alpha_bounds(resized)
                if visible_bounds is None:
                    resized.close()
                    missing_characters += 1
                    message_queue.put(
                        (
                            "progress",
                            f"第 {board.number} 版：“{placement.character}”没有可见内容，已留空 "
                            f"({processed_characters}/{board.character_count})",
                            1,
                            False,
                        )
                    )
                    continue
                visible_offset = (visible_bounds[0], visible_bounds[1])
                if visible_bounds != (0, 0, resized.width, resized.height):
                    layer_image = resized.crop(visible_bounds)
                    resized.close()
                else:
                    layer_image = resized
                layer_image.info["fonteditor_visible_offset"] = visible_offset
                cached = glyph_cache.put(cache_key, layer_image)
            else:
                cached_offset = layer_image.info.get(
                    "fonteditor_visible_offset",
                    (0, 0),
                )
                visible_offset = (
                    int(cached_offset[0]),
                    int(cached_offset[1]),
                )
            left = cell_left + (cell_width - target_width) // 2
            top = cell_top + (cell_height - target_height) // 2
            left += visible_offset[0]
            top += visible_offset[1]
            source_layer_pixels += target_width * target_height
            retained_layer_pixels += layer_image.width * layer_image.height
            column_number = (
                board.effective_columns - placement.column
                if parameters.flow_direction == FLOW_RIGHT_TO_LEFT
                else placement.column + 1
            )
            layer_name = _safe_psd_name(
                f"{placement.character}_{placement.row + 1}_{column_number}"
            )
            layer = psd.create_pixel_layer(
                name=layer_name,
                image=layer_image,
                top=top,
                left=left,
                compression=layer_compression,
            )
            composite_image.paste(layer_image, (left, top), layer_image)
            if not cached:
                layer_image.close()
            group = paragraph_groups.get(placement.paragraph)
            if group is None:
                group_name = (
                    "首经题"
                    if placement.paragraph == 0
                    else f"第{placement.paragraph}段"
                )
                group = psd.create_group(name=group_name, open_folder=False)
                group.move_to_group(scripture_group)
                paragraph_groups[placement.paragraph] = group
            layer.move_to_group(group)
            placed += 1
            message_queue.put(
                (
                    "progress",
                    f"第 {board.number} 版：正在放置“{placement.character}” "
                    f"({processed_characters}/{board.character_count})",
                    1,
                    False,
                )
            )
        stage_times["字形图层"] = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        if parameters.add_annotations:
            annotation = Image.new(
                "RGBA",
                (grid.canvas_width, grid.canvas_height),
                (0, 0, 0, 0),
            )
            annotation_draw = ImageDraw.Draw(annotation)
            font_size = max(10, min(18, grid.canvas_width // 120))
            font = _annotation_font(font_size)
            lines = _layout_annotation_lines(board.number, parameters, grid)
            line_height = font_size + 5
            text_width = max(
                (
                    annotation_draw.textbbox((0, 0), line, font=font)[2]
                    for line in lines
                ),
                default=0,
            )
            panel_width = min(grid.canvas_width, text_width + 24)
            panel_height = min(
                grid.canvas_height,
                line_height * len(lines) + 16,
            )
            annotation_draw.rectangle(
                (0, 0, panel_width, panel_height),
                fill=(255, 255, 255, 225),
            )
            for line_index, line in enumerate(lines):
                annotation_draw.text(
                    (12, 8 + line_index * line_height),
                    line,
                    fill=(70, 70, 70, 235),
                    font=font,
                )
            psd.create_pixel_layer(
                name="尺寸标注",
                image=annotation,
                top=0,
                left=0,
                compression=layer_compression,
            )
            composite_image.paste(annotation, (0, 0), annotation)
            annotation.close()
        stage_times["尺寸标注"] = time.perf_counter() - stage_started
        message_queue.put(
            (
                "progress",
                f"正在编码第 {board.number}/{total_jobs} 版兼容预览",
                0,
                True,
            )
        )
        stage_started = time.perf_counter()
        _install_composite_preview(psd, composite_image, layer_compression)
        composite_image.close()
        composite_image = None
        stage_times["兼容预览编码"] = time.perf_counter() - stage_started
        message_queue.put(
            (
                "progress",
                (
                    f"正在压缩并写入第 {board.number}/{total_jobs} 版 {plan.format_name}，"
                    "图层较多时可能需要较长时间"
                    if compress_psd
                    else f"正在写入第 {board.number}/{total_jobs} 版无压缩 {plan.format_name}"
                ),
                0,
                True,
            )
        )
        stage_started = time.perf_counter()
        psd.save(
            temporary_path,
            encoding="gb18030",
            compression=layer_compression,
        )
        stage_times["PSD写盘"] = time.perf_counter() - stage_started

        cleanup_started = time.perf_counter()
        del psd
        psd = None
        gc.collect()
        stage_times["内存回收"] = time.perf_counter() - cleanup_started
        message_queue.put(
            (
                "result",
                _BoardProcessResult(
                    placed,
                    missing_characters,
                    source_layer_pixels,
                    retained_layer_pixels,
                    tuple(stage_times.items()),
                ),
            )
        )
    except BaseException as exc:
        message_queue.put(
            ("error", type(exc).__name__, str(exc), traceback.format_exc())
        )
    finally:
        if glyph_cache is not None:
            glyph_cache.close()
        if composite_image is not None:
            composite_image.close()
        if psd is not None:
            del psd
            gc.collect()


def _terminate_process(process: multiprocessing.Process) -> None:
    """立即终止生成子进程，并确保不遗留后台写盘任务。"""

    if process.is_alive():
        process.terminate()
    process.join(timeout=1.0)
    if process.is_alive():
        process.kill()
        process.join(timeout=1.0)


def _run_board_process(
    board: BoardLayout,
    glyph_index: GlyphIndex,
    parameters: LayoutParameters,
    geometry: LayoutGrid | None,
    temporary_path: str,
    compress_psd: bool,
    psb: bool,
    total_jobs: int,
    *,
    task_name: str,
    notify: Callable[..., None],
    cancelled: Callable[[], bool],
) -> _BoardProcessResult:
    """监听单版生成进程；收到停止请求时立即终止。"""

    context = multiprocessing.get_context("spawn")
    message_queue = context.Queue()
    process = context.Process(
        target=_generate_board_process,
        args=(
            board,
            glyph_index,
            parameters,
            geometry,
            temporary_path,
            compress_psd,
            psb,
            total_jobs,
            message_queue,
        ),
        name=f"{task_name}-第{board.number}版",
        daemon=True,
    )
    result: _BoardProcessResult | None = None
    error: tuple[str, str, str] | None = None

    def receive_messages() -> None:
        nonlocal result, error
        while True:
            try:
                message = message_queue.get_nowait()
            except queue.Empty:
                return
            if not isinstance(message, tuple) or not message:
                continue
            if message[0] == "progress":
                notify(
                    str(message[1]),
                    int(message[2]),
                    indeterminate=bool(message[3]),
                )
            elif message[0] == "result" and isinstance(
                message[1], _BoardProcessResult
            ):
                result = message[1]
            elif message[0] == "error":
                error = (str(message[1]), str(message[2]), str(message[3]))

    try:
        process.start()
        while process.is_alive():
            receive_messages()
            if cancelled():
                _terminate_process(process)
                raise GenerationCancelled("用户停止了通用经文排版。")
            process.join(timeout=0.02)
        process.join()
        deadline = time.monotonic() + 1.0
        while result is None and error is None and time.monotonic() < deadline:
            receive_messages()
            if result is None and error is None:
                time.sleep(0.01)
        receive_messages()
        if cancelled():
            raise GenerationCancelled("用户停止了通用经文排版。")
        if error is not None:
            error_name, error_message, error_traceback = error
            raise RuntimeError(
                f"生成子进程异常（{error_name}）：{error_message}\n{error_traceback}"
            )
        if process.exitcode != 0:
            raise RuntimeError(f"生成子进程异常退出，退出码：{process.exitcode}")
        if result is None:
            raise RuntimeError("生成子进程没有返回版面结果。")
        return result
    finally:
        if process.is_alive():
            _terminate_process(process)
        if process.pid is not None:
            process.close()
        message_queue.close()
        message_queue.cancel_join_thread()


def generate_psd_boards(
    boards: Iterable[BoardLayout],
    glyph_index: GlyphIndex,
    parameters: LayoutParameters,
    output_dir: str,
    *,
    board_parameters: Mapping[int, LayoutParameters] | None = None,
    board_geometries: Mapping[int, LayoutGrid] | None = None,
    task_name: str = "通用经文排版",
    output_base_name: str = "通用经文排版",
    compress_psd: bool = True,
    output_format: str = OUTPUT_FORMAT_AUTO,
    selected_boards: Iterable[int] | None = None,
    conflict_decisions: Mapping[int, str] | None = None,
    progress_callback: Callable[[GenerationProgress], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> GenerationResult:
    """逐版原子生成分层 PSD/PSB；停止时不留下当前版半成品。"""

    try:
        import psd_tools  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "缺少分层 PSD 组件 psd-tools，请先安装 requirements.txt 中的依赖。"
        ) from exc

    parameters.validate()
    output_root = os.path.abspath(output_dir)
    os.makedirs(output_root, exist_ok=True)
    board_list = list(boards)
    total_boards = len(board_list)
    selected = set(selected_boards or (board.number for board in board_list))
    jobs = [board for board in board_list if board.number in selected]
    parameter_map = dict(board_parameters or {})
    geometry_map = dict(board_geometries or {})
    for board in jobs:
        parameter_map.get(board.number, parameters).validate()
    plans = {
        board.number: plan_board_output(
            board,
            glyph_index,
            parameter_map.get(board.number, parameters),
            output_format,
            geometry=geometry_map.get(board.number),
        )
        for board in jobs
    }
    decisions = dict(conflict_decisions or {})
    generated: list[GeneratedBoard] = []
    total_characters = sum(board.character_count for board in jobs)
    compression_name = "RLE" if compress_psd else "无压缩"
    batch_started = time.perf_counter()
    stage_totals: defaultdict[str, float] = defaultdict(float)
    write_log(
        f"{task_name}生成开始｜"
        f"版数={len(jobs)}｜总文字={total_characters}｜DPI={parameters.dpi}｜"
        f"文件格式={output_format}｜PSD压缩={compression_name}｜输出目录={output_root}"
    )

    def log_batch_summary(status: str) -> None:
        stage_text = "、".join(
            f"{stage}={seconds:.4f}秒"
            for stage, seconds in stage_totals.items()
        ) or "无"
        write_log(
            f"{task_name}耗时汇总｜"
            f"状态={status}｜已保存={sum(not item.skipped for item in generated)}｜"
            f"已跳过={sum(item.skipped for item in generated)}｜"
            f"总耗时={time.perf_counter() - batch_started:.4f}秒｜"
            f"PSD压缩={compression_name}｜阶段耗时={stage_text}"
        )

    progress_total = max(1, total_characters + len(jobs) * 5)
    progress_value = 0

    def cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    def notify(
        message: str,
        increment: int = 0,
        *,
        indeterminate: bool = False,
    ) -> None:
        nonlocal progress_value
        progress_value = min(progress_total, progress_value + increment)
        if progress_callback:
            progress_callback(
                GenerationProgress(
                    progress_value,
                    progress_total,
                    message,
                    indeterminate,
                )
            )

    for board in jobs:
        if cancelled():
            log_batch_summary("已停止")
            return GenerationResult(tuple(generated), True)
        board_started = time.perf_counter()
        stage_times: dict[str, float] = {}
        board_parameter = parameter_map.get(board.number, parameters)
        board_geometry = geometry_map.get(board.number)
        plan = plans[board.number]
        if plan.memory_warning:
            write_log(f"{task_name}资源提示｜{plan.memory_warning}")
            notify(plan.memory_warning)
        output_path = board_output_path(
            output_root,
            board.number,
            board_parameter.dpi,
            output_base_name,
            total_boards=total_boards,
            extension=plan.extension,
        )
        if os.path.exists(output_path):
            decision = decisions.get(board.number, CONFLICT_CANCEL)
            if decision == CONFLICT_SKIP:
                generated.append(GeneratedBoard(board.number, output_path, 0, True))
                notify(f"已跳过第 {board.number} 版", board.character_count + 5)
                write_log(
                    f"{task_name}单版跳过｜版号={board.number}/{total_boards}｜"
                    f"文件={os.path.basename(output_path)}"
                )
                continue
            if decision != CONFLICT_OVERWRITE:
                log_batch_summary("已停止")
                return GenerationResult(tuple(generated), True)
        file_descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".第{board.number:02d}版_",
            suffix=f".tmp{plan.extension}",
            dir=output_root,
        )
        os.close(file_descriptor)
        try:
            free_bytes = shutil.disk_usage(output_root).free
            minimum_workspace = _required_disk_workspace(plan)
            if free_bytes < minimum_workspace:
                raise OSError(
                    f"输出磁盘可用空间仅 {free_bytes / 1024**2:.1f} MiB，"
                    f"第 {board.number} 版至少需要预留约 "
                    f"{minimum_workspace / 1024**2:.1f} MiB 临时空间。"
                )
            process_result = _run_board_process(
                board,
                glyph_index,
                board_parameter,
                board_geometry,
                temporary_path,
                compress_psd,
                plan.psb,
                len(jobs),
                task_name=task_name,
                notify=notify,
                cancelled=cancelled,
            )
            notify(
                f"第 {board.number}/{len(jobs)} 版生成进程已退出，内存资源已释放",
                1,
            )
            stage_times.update(process_result.stage_times)
            stage_started = time.perf_counter()
            if cancelled():
                raise GenerationCancelled("用户停止了通用经文排版。")
            if (
                os.path.exists(output_path)
                and decisions.get(board.number) != CONFLICT_OVERWRITE
            ):
                raise FileExistsError(
                    f"第 {board.number} 版目标文件在生成期间出现，未获覆盖授权。"
                )
            os.replace(temporary_path, output_path)
            stage_times["文件替换"] = time.perf_counter() - stage_started
        except GenerationCancelled:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
            write_log(
                f"{task_name}单版停止｜"
                f"版号={board.number}/{total_boards}｜"
                f"已耗时={time.perf_counter() - board_started:.4f}秒｜"
                f"PSD压缩={compression_name}"
            )
            log_batch_summary("已停止")
            return GenerationResult(tuple(generated), True)
        except Exception as exc:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
            write_log(
                f"{task_name}单版失败｜"
                f"版号={board.number}/{total_boards}｜"
                f"已耗时={time.perf_counter() - board_started:.4f}秒｜"
                f"PSD压缩={compression_name}｜错误={type(exc).__name__}: {exc}"
            )
            raise
        generated.append(
            GeneratedBoard(
                board.number,
                output_path,
                process_result.placed,
                missing_characters=process_result.missing_characters,
            )
        )
        for stage, seconds in stage_times.items():
            stage_totals[stage] += seconds
        stage_text = "、".join(
            f"{stage}={seconds:.4f}秒"
            for stage, seconds in stage_times.items()
        )
        try:
            output_size = os.path.getsize(output_path)
        except OSError:
            output_size = 0
        grid = board_geometry or compute_grid(
            board_parameter,
            board.effective_columns,
            board.effective_rows,
        )
        write_log(
            f"{task_name}单版耗时｜"
            f"版号={board.number}/{total_boards}｜画布={grid.canvas_width}x{grid.canvas_height}｜"
            f"文字={board.character_count}｜PSD压缩={compression_name}｜"
            f"字形图层像素={process_result.retained_layer_pixels}/"
            f"{process_result.source_layer_pixels}"
            f"({process_result.retained_layer_pixels / max(process_result.source_layer_pixels, 1) * 100:.2f}%)｜"
            f"文件大小={output_size / (1024 * 1024):.2f}MiB｜"
            f"总耗时={time.perf_counter() - board_started:.4f}秒｜阶段耗时={stage_text}"
        )
        notify(f"第 {board.number} 版已保存", 2)
    log_batch_summary("完成")
    return GenerationResult(tuple(generated), False)


def _safe_psd_name(name: str) -> str:
    normalized = unicodedata.normalize("NFC", str(name))
    return "".join(
        character if character >= " " and character != "\x7f" else "_"
        for character in normalized
    )


def _required_disk_workspace(plan: BoardOutputPlan) -> int:
    """为临时 PSD/PSB 文件预留完整预计大小和少量文件系统余量。"""

    estimated = max(0, int(plan.estimated_file_bytes))
    return max(
        MINIMUM_DISK_WORKSPACE_BYTES,
        int(estimated * DISK_WORKSPACE_SAFETY_FACTOR + 0.999),
    )


def _layout_annotation_lines(
    board_number: int,
    parameters: LayoutParameters,
    grid: GridGeometry,
) -> tuple[str, ...]:
    row_special = "、".join(
        f"{item.row}={item.gap_mm:g}"
        for item in parameters.row_gap_adjustments
    ) or "无"
    column_special = "、".join(
        f"{item.column}={item.gap_mm:g}"
        for item in parameters.column_gap_adjustments
    ) or "无"
    frame_text = (
        "绘制；上/下/左/右 "
        f"{parameters.frame_top_mm:g}/{parameters.frame_bottom_mm:g}/"
        f"{parameters.frame_left_mm:g}/{parameters.frame_right_mm:g} mm"
        if parameters.draw_outer_frame
        else "不绘制"
    )
    if parameters.scale_mode == SCALE_BY_DPI:
        scale_text = f"按源图尺寸；全局 {parameters.scale_percent}%"
        if parameters.auto_scale_enabled:
            scale_text += (
                f"；自动放大 {parameters.auto_enlarge_threshold}%→"
                f"{parameters.auto_enlarge_fill_percent}%，自动缩小 "
                f"{parameters.auto_shrink_threshold}%→"
                f"{parameters.auto_shrink_fill_percent}%"
            )
        else:
            scale_text += "；自动缩放关闭"
    else:
        scale_text = f"相对单元格；目标 {parameters.cell_fill_percent}%"
    paragraph_text = parameters.paragraph_mode
    if parameters.layout_mode == LAYOUT_HORIZONTAL:
        paragraph_text = paragraph_text.replace("换列", "换行")
    if parameters.paragraph_mode == PARAGRAPH_SKIP_CELLS:
        paragraph_text += f" {parameters.paragraph_skip_cells} 格"
    yes_no = lambda value: "是" if value else "否"
    track_name = "行" if parameters.layout_mode == LAYOUT_HORIZONTAL else "列"
    return (
        f"第 {board_number} 版；{parameters.dpi} DPI；画布 {grid.canvas_width}×{grid.canvas_height} px",
        f"排版方式：{parameters.layout_mode}；行进方向：{parameters.flow_direction}",
        f"单元格 高×宽 {parameters.cell_height_mm:g}×{parameters.cell_width_mm:g} mm；每页 行×列 {parameters.rows}×{parameters.columns}",
        f"行列间距 行/列 {parameters.row_gap_mm:g}/{parameters.column_gap_mm:g} mm",
        f"大框：{frame_text}",
        "画布边距 上/下/左/右 "
        f"{parameters.canvas_top_mm:g}/{parameters.canvas_bottom_mm:g}/"
        f"{parameters.canvas_left_mm:g}/{parameters.canvas_right_mm:g} mm",
        f"特殊行距：{row_special}；特殊列距：{column_special}",
        f"文字缩放：{scale_text}",
        f"段落：{paragraph_text}；首经题换{track_name} "
        f"{yes_no(parameters.first_title_new_column)}；尾经题换{track_name} "
        f"{yes_no(parameters.last_title_new_column)}",
        f"末版删除空{track_name}：{yes_no(parameters.trim_empty_columns)}",
    )


def _annotation_font(size: int) -> ImageFont.ImageFont:
    font_root = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")
    for filename in ("msyh.ttc", "simsun.ttc", "simhei.ttf"):
        path = os.path.join(font_root, filename)
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()
