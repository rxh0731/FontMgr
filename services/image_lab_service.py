"""图片实验室的预览、人工图层和完整尺寸导出服务。"""

from __future__ import annotations

import os
import struct
import tempfile
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps

from core.image_cleanup import ImageCleanupResult, clean_document_image
from data.image_lab_project_store import (
    ImageLabProject,
    ImageLabProjectStore,
    ImageLabStroke,
)
from utils.system_resources import MIB, get_system_memory_status


SUPPORTED_IMAGE_FILTER = (
    "文字图片 (*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.webp);;所有文件 (*)"
)
PSD_MAX_DIMENSION = 30_000
PSB_MAX_DIMENSION = 300_000
PSD_SAFE_FILE_BYTES = 1_800_000_000
DETAIL_CACHE_MIN_BYTES = 64 * MIB
DETAIL_CACHE_MAX_BYTES = 512 * MIB
DETAIL_CACHE_EMERGENCY_BYTES = 16 * MIB


class ImageLabCancelled(RuntimeError):
    """用户安全取消了图片实验室后台任务。"""


@dataclass(frozen=True, slots=True)
class ImageLabSourceInfo:
    path: str
    width: int
    height: int
    mode: str
    dpi_x: float
    dpi_y: float


@dataclass(frozen=True, slots=True)
class ImageLabPreview:
    source: np.ndarray
    detail_source: np.ndarray
    cleanup: ImageCleanupResult
    effective_alpha: np.ndarray
    composite: np.ndarray
    source_width: int
    source_height: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class ImageLabDetailPreview:
    """按原图区域生成、用于放大查看的临时高清预览。"""

    source: np.ndarray
    composite: np.ndarray
    effective_alpha: np.ndarray
    uncertainty: np.ndarray
    source_rect: tuple[int, int, int, int]
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class ImageLabExportResult:
    output_path: str
    kind: str
    elapsed_seconds: float
    width: int
    height: int


class ImageLabService:
    """不依赖字库状态的整图处理服务。"""

    def __init__(self, store: ImageLabProjectStore | None = None) -> None:
        self.store = store or ImageLabProjectStore()

    @staticmethod
    def _open_image(path: str) -> Image.Image:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            image = Image.open(path)
        return image

    @staticmethod
    def _apply_exif_orientation(image: Image.Image) -> Image.Image:
        oriented = ImageOps.exif_transpose(image)
        if oriented is not image:
            image.close()
        return oriented

    def inspect_source(self, path: str) -> ImageLabSourceInfo:
        source_path = os.path.abspath(os.fspath(path))
        if not os.path.isfile(source_path):
            raise FileNotFoundError("待处理的原稿不存在。")
        image = self._open_image(source_path)
        try:
            width, height = image.size
            orientation = int(image.getexif().get(274, 1))
            if orientation in {5, 6, 7, 8}:
                width, height = height, width
            dpi = image.info.get("dpi", (0.0, 0.0))
            try:
                dpi_x, dpi_y = float(dpi[0]), float(dpi[1])
            except (IndexError, TypeError, ValueError):
                dpi_x = dpi_y = 0.0
            mode = str(image.mode)
        finally:
            image.close()
        if width <= 0 or height <= 0:
            raise ValueError("原稿尺寸无效。")
        return ImageLabSourceInfo(
            path=source_path,
            width=width,
            height=height,
            mode=mode,
            dpi_x=dpi_x,
            dpi_y=dpi_y,
        )

    def create_project(self, path: str) -> ImageLabProject:
        info = self.inspect_source(path)
        return self.store.create(
            info.path,
            width=info.width,
            height=info.height,
            mode=info.mode,
            dpi_x=info.dpi_x,
            dpi_y=info.dpi_y,
        )

    @staticmethod
    def _detail_cache_pixel_budget() -> int:
        """按当前可用内存限制单个图片实验室高清工作副本。"""

        _total_memory, available_memory = get_system_memory_status()
        if available_memory <= 0:
            cache_bytes = DETAIL_CACHE_MIN_BYTES
        else:
            cache_bytes = max(
                DETAIL_CACHE_EMERGENCY_BYTES,
                min(DETAIL_CACHE_MAX_BYTES, available_memory // 8),
            )
        return max(1, cache_bytes // 3)

    def load_preview(
        self,
        project: ImageLabProject,
        *,
        max_edge: int = 2200,
        detail_source_cache: np.ndarray | None = None,
        build_detail_cache: bool = True,
    ) -> ImageLabPreview:
        if max_edge < 320:
            raise ValueError("预览尺寸过小。")
        started = time.perf_counter()
        if detail_source_cache is not None:
            if (
                not isinstance(detail_source_cache, np.ndarray)
                or detail_source_cache.dtype != np.uint8
                or detail_source_cache.ndim != 3
                or detail_source_cache.shape[2] != 3
            ):
                raise ValueError("高清工作缓存格式无效。")
            detail_source = detail_source_cache
            cache_height, cache_width = detail_source.shape[:2]
            aspect_error = abs(
                cache_width * project.source_height
                - cache_height * project.source_width
            )
            if aspect_error > max(project.source_width, project.source_height):
                raise ValueError("高清工作缓存与当前原稿尺寸不匹配。")
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", Image.DecompressionBombWarning)
                image = self._open_image(project.source_path)
                try:
                    image = self._apply_exif_orientation(image)
                    source_width, source_height = image.size
                    if build_detail_cache:
                        pixel_budget = self._detail_cache_pixel_budget()
                        detail_scale = min(
                            1.0,
                            np.sqrt(
                                pixel_budget
                                / max(1, source_width * source_height)
                            ),
                        )
                    else:
                        detail_scale = min(
                            1.0,
                            max_edge / max(source_width, source_height),
                        )
                    detail_size = (
                        max(1, int(round(source_width * detail_scale))),
                        max(1, int(round(source_height * detail_scale))),
                    )
                    if image.size != detail_size:
                        image.thumbnail(detail_size, Image.Resampling.LANCZOS)
                    detail_image = image.convert("RGB")
                    try:
                        detail_source = np.array(
                            detail_image,
                            dtype=np.uint8,
                            copy=True,
                        )
                    finally:
                        detail_image.close()
                finally:
                    image.close()
        detail_height, detail_width = detail_source.shape[:2]
        if max(detail_width, detail_height) <= max_edge:
            source = detail_source
        else:
            preview_scale = min(max_edge / detail_width, max_edge / detail_height)
            preview_size = (
                max(1, int(round(detail_width * preview_scale))),
                max(1, int(round(detail_height * preview_scale))),
            )
            source = cv2.resize(
                detail_source,
                preview_size,
                interpolation=cv2.INTER_AREA,
            )
        cleanup = clean_document_image(
            source,
            project.options,
            source_region=(
                0,
                0,
                project.source_width,
                project.source_height,
            ),
            source_size=(project.source_width, project.source_height),
        )
        effective_alpha = self.apply_strokes(
            cleanup.cleanup_layer[:, :, 3],
            project.strokes,
            project.source_width,
            project.source_height,
        )
        composite = self.compose(source, effective_alpha)
        for array in (source, detail_source, effective_alpha, composite):
            array.setflags(write=False)
        return ImageLabPreview(
            source=source,
            detail_source=detail_source,
            cleanup=cleanup,
            effective_alpha=effective_alpha,
            composite=composite,
            source_width=project.source_width,
            source_height=project.source_height,
            elapsed_seconds=time.perf_counter() - started,
        )

    def load_detail_preview(
        self,
        project: ImageLabProject,
        preview: ImageLabPreview,
        source_rect: tuple[int, int, int, int],
        target_size: tuple[int, int],
        *,
        processing_overlap: int = 96,
        max_output_edge: int = 2600,
    ) -> ImageLabDetailPreview:
        """从原稿读取当前可见区域，并生成适合屏幕显示的高清处理结果。"""

        started = time.perf_counter()
        if len(source_rect) != 4 or len(target_size) != 2:
            raise ValueError("高清预览区域参数无效。")
        left, top, right, bottom = (int(value) for value in source_rect)
        left = max(0, min(project.source_width - 1, left))
        top = max(0, min(project.source_height - 1, top))
        right = max(left + 1, min(project.source_width, right))
        bottom = max(top + 1, min(project.source_height, bottom))
        target_width = max(1, int(target_size[0]))
        target_height = max(1, int(target_size[1]))
        if processing_overlap < 16 or max_output_edge < 320:
            raise ValueError("高清预览处理参数无效。")
        output_scale = min(
            1.0,
            max_output_edge / max(target_width, target_height),
        )
        target_width = max(1, int(round(target_width * output_scale)))
        target_height = max(1, int(round(target_height * output_scale)))
        source_region_width = right - left
        source_region_height = bottom - top
        scale_x = target_width / source_region_width
        scale_y = target_height / source_region_height
        source_overlap_x = max(1, int(np.ceil(processing_overlap / scale_x)))
        source_overlap_y = max(1, int(np.ceil(processing_overlap / scale_y)))
        read_left = max(0, left - source_overlap_x)
        read_top = max(0, top - source_overlap_y)
        read_right = min(project.source_width, right + source_overlap_x)
        read_bottom = min(project.source_height, bottom + source_overlap_y)
        read_target_width = max(
            target_width,
            int(np.ceil((read_right - read_left) * scale_x)),
        )
        read_target_height = max(
            target_height,
            int(np.ceil((read_bottom - read_top) * scale_y)),
        )

        cache_height, cache_width = preview.detail_source.shape[:2]
        cache_left = max(
            0,
            int(np.floor(read_left * cache_width / project.source_width)),
        )
        cache_top = max(
            0,
            int(np.floor(read_top * cache_height / project.source_height)),
        )
        cache_right = min(
            cache_width,
            int(np.ceil(read_right * cache_width / project.source_width)),
        )
        cache_bottom = min(
            cache_height,
            int(np.ceil(read_bottom * cache_height / project.source_height)),
        )
        cached_region = preview.detail_source[
            cache_top:cache_bottom,
            cache_left:cache_right,
        ]
        if cached_region.size == 0:
            raise ValueError("高清工作缓存未覆盖当前原图区域。")
        tile = cv2.resize(
            cached_region,
            (read_target_width, read_target_height),
            interpolation=(
                cv2.INTER_AREA
                if cached_region.shape[1] > read_target_width
                or cached_region.shape[0] > read_target_height
                else cv2.INTER_LANCZOS4
            ),
        )

        core_left = int(round((left - read_left) * scale_x))
        core_top = int(round((top - read_top) * scale_y))
        core_right = min(tile.shape[1], core_left + target_width)
        core_bottom = min(tile.shape[0], core_top + target_height)

        def exact_size(values: np.ndarray, interpolation: int) -> np.ndarray:
            core = np.array(
                values[core_top:core_bottom, core_left:core_right],
                copy=True,
            )
            if core.shape[:2] != (target_height, target_width):
                core = cv2.resize(
                    core,
                    (target_width, target_height),
                    interpolation=interpolation,
                )
            return core

        source = exact_size(tile, cv2.INTER_LANCZOS4)
        # 高清预览只提升原图显示清晰度，清理判断统一复用快速预览蒙版，
        # 避免同一位置因处理尺寸和局部背景不同而产生两套结果。
        preview_height, preview_width = preview.effective_alpha.shape
        mask_left = max(
            0,
            min(
                preview_width - 1,
                int(np.floor(left * preview_width / project.source_width)),
            ),
        )
        mask_top = max(
            0,
            min(
                preview_height - 1,
                int(np.floor(top * preview_height / project.source_height)),
            ),
        )
        mask_right = max(
            mask_left + 1,
            min(
                preview_width,
                int(np.ceil(right * preview_width / project.source_width)),
            ),
        )
        mask_bottom = max(
            mask_top + 1,
            min(
                preview_height,
                int(np.ceil(bottom * preview_height / project.source_height)),
            ),
        )
        mask_region = preview.cleanup.cleanup_layer[
            mask_top:mask_bottom,
            mask_left:mask_right,
            3,
        ]
        uncertainty_region = preview.cleanup.uncertainty_mask[
            mask_top:mask_bottom,
            mask_left:mask_right,
        ]
        if mask_region.size == 0 or uncertainty_region.size == 0:
            raise ValueError("快速预览清理蒙版未覆盖当前原图区域。")
        alpha = cv2.resize(
            mask_region,
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR,
        )
        uncertainty = cv2.resize(
            uncertainty_region,
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        )
        alpha = self.apply_strokes(
            alpha,
            list(project.strokes),
            project.source_width,
            project.source_height,
            source_region=(left, top, right, bottom),
        )
        composite = self.compose(source, alpha)
        for array in (source, composite, alpha, uncertainty):
            array.setflags(write=False)
        return ImageLabDetailPreview(
            source=source,
            composite=composite,
            effective_alpha=alpha,
            uncertainty=uncertainty,
            source_rect=(left, top, right, bottom),
            elapsed_seconds=time.perf_counter() - started,
        )

    @staticmethod
    def compose(source: np.ndarray, cleanup_alpha: np.ndarray) -> np.ndarray:
        if source.shape[:2] != cleanup_alpha.shape:
            raise ValueError("原稿与清理层尺寸不一致。")
        alpha = cleanup_alpha.astype(np.float32)[:, :, None] / 255.0
        return np.clip(
            source.astype(np.float32) * (1.0 - alpha) + 255.0 * alpha,
            0,
            255,
        ).astype(np.uint8)

    @staticmethod
    def apply_strokes(
        cleanup_alpha: np.ndarray,
        strokes: list[ImageLabStroke],
        source_width: int,
        source_height: int,
        *,
        source_offset: tuple[int, int] = (0, 0),
        source_region: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray:
        """按原图坐标把白色覆盖和还原笔画作用到清理层。"""

        result = np.array(cleanup_alpha, dtype=np.uint8, copy=True)
        target_height, target_width = result.shape
        if source_region is not None and source_offset != (0, 0):
            raise ValueError("人工笔画不能同时指定区域和分块偏移。")
        offset_x, offset_y = source_offset
        scale_x = target_width / max(1, source_width)
        scale_y = target_height / max(1, source_height)
        if source_region is not None:
            region_left, region_top, region_right, region_bottom = source_region
            region_width = max(1, region_right - region_left)
            region_height = max(1, region_bottom - region_top)
            scale_x = target_width / region_width
            scale_y = target_height / region_height
            offset_x = region_left * scale_x
            offset_y = region_top * scale_y
        elif source_offset != (0, 0):
            scale_x = scale_y = 1.0
        for stroke in strokes:
            points = [
                (
                    int(round(point[0] * source_width * scale_x - offset_x)),
                    int(round(point[1] * source_height * scale_y - offset_y)),
                )
                for point in stroke.points
            ]
            width_scale = (scale_x + scale_y) / 2.0
            line_width = max(1, int(round(stroke.width * width_scale)))
            value = 255 if stroke.tool == "cover" else 0
            if len(points) == 1:
                cv2.circle(result, points[0], max(1, line_width // 2), value, -1)
                continue
            cv2.polylines(
                result,
                [np.asarray(points, dtype=np.int32)],
                False,
                value,
                line_width,
                cv2.LINE_AA,
            )
        return result

    def export_full_resolution(
        self,
        project: ImageLabProject,
        output_path: str,
        *,
        kind: str = "composite",
        progress_callback: Callable[[int, int, str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
        tile_size: int = 2048,
        overlap: int = 128,
    ) -> ImageLabExportResult:
        """后台分块生成清理效果、白色清理层或分层 Photoshop 文件。"""

        if kind not in {"composite", "layer", "photoshop"}:
            raise ValueError("导出类型必须是清理效果、透明清理层或 Photoshop 文件。")
        if tile_size < 512 or overlap < 32 or overlap * 2 >= tile_size:
            raise ValueError("分块参数无效。")
        target = os.path.abspath(os.fspath(output_path))
        suffix = Path(target).suffix.lower()
        if kind == "photoshop":
            if suffix not in {".psd", ".psb"}:
                raise ValueError("Photoshop 分层导出仅支持 PSD 或 PSB。")
        elif suffix not in {".tif", ".tiff", ".png"}:
            raise ValueError("图片导出仅支持 TIFF 或 PNG。")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        started = time.perf_counter()
        preview = self.load_preview(
            project,
            max_edge=1600,
            build_detail_cache=False,
        )
        guide_mask = preview.cleanup.page_mask
        image = self._open_image(project.source_path)
        temporary_raw = ""
        temporary_composite_raw = ""
        temporary_output = ""
        output: np.memmap | None = None
        composite_output: np.memmap | None = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", Image.DecompressionBombWarning)
                image = self._apply_exif_orientation(image)
            width, height = image.size
            if width > PSB_MAX_DIMENSION or height > PSB_MAX_DIMENSION:
                raise ValueError(
                    f"原稿达到 {width}×{height}，超过 PSB 单边 300,000 像素上限。"
                )
            psb = False
            if kind == "photoshop":
                estimated_bytes = width * height * 12 + 16 * 1024 * 1024
                psb = (
                    suffix == ".psb"
                    or width > PSD_MAX_DIMENSION
                    or height > PSD_MAX_DIMENSION
                    or estimated_bytes >= PSD_SAFE_FILE_BYTES
                )
                if psb and suffix != ".psb":
                    target = str(Path(target).with_suffix(".psb"))
                    suffix = ".psb"
            channels = 4 if kind in {"layer", "photoshop"} else 3
            raw_handle = tempfile.NamedTemporaryFile(
                prefix="fontmgr_image_lab_",
                suffix=".raw",
                dir=os.path.dirname(target),
                delete=False,
            )
            temporary_raw = raw_handle.name
            raw_handle.close()
            output = np.memmap(
                temporary_raw,
                dtype=np.uint8,
                mode="w+",
                shape=(height, width, channels),
            )
            if kind == "photoshop":
                composite_handle = tempfile.NamedTemporaryFile(
                    prefix="fontmgr_image_lab_composite_",
                    suffix=".raw",
                    dir=os.path.dirname(target),
                    delete=False,
                )
                temporary_composite_raw = composite_handle.name
                composite_handle.close()
                composite_output = np.memmap(
                    temporary_composite_raw,
                    dtype=np.uint8,
                    mode="w+",
                    shape=(height, width, 3),
                )
            columns = (width + tile_size - 1) // tile_size
            rows = (height + tile_size - 1) // tile_size
            total = max(1, columns * rows)
            current = 0
            tile_options = replace(project.options, detect_page=False)
            for top in range(0, height, tile_size):
                for left in range(0, width, tile_size):
                    if cancelled is not None and cancelled():
                        raise ImageLabCancelled("已停止完整尺寸导出。")
                    right = min(width, left + tile_size)
                    bottom = min(height, top + tile_size)
                    read_left = max(0, left - overlap)
                    read_top = max(0, top - overlap)
                    read_right = min(width, right + overlap)
                    read_bottom = min(height, bottom + overlap)
                    tile = np.array(
                        image.crop((read_left, read_top, read_right, read_bottom)).convert("RGB"),
                        dtype=np.uint8,
                        copy=True,
                    )
                    tile_cleanup = clean_document_image(
                        tile,
                        tile_options,
                        calibration=preview.cleanup.calibration,
                        source_region=(read_left, read_top, read_right, read_bottom),
                        source_size=(width, height),
                    )
                    core_x = left - read_left
                    core_y = top - read_top
                    core_width = right - left
                    core_height = bottom - top
                    alpha = np.array(
                        tile_cleanup.cleanup_layer[
                            core_y:core_y + core_height,
                            core_x:core_x + core_width,
                            3,
                        ],
                        copy=True,
                    )
                    page = self._page_guide_tile(
                        guide_mask,
                        left,
                        top,
                        right,
                        bottom,
                        width,
                        height,
                    )
                    alpha[page == 0] = 255
                    alpha = self.apply_strokes(
                        alpha,
                        project.strokes,
                        width,
                        height,
                        source_offset=(left, top),
                    )
                    if kind == "composite":
                        source_core = tile[
                            core_y:core_y + core_height,
                            core_x:core_x + core_width,
                        ]
                        output[top:bottom, left:right] = self.compose(source_core, alpha)
                    else:
                        output[top:bottom, left:right, :3] = 255
                        output[top:bottom, left:right, 3] = alpha
                        if composite_output is not None:
                            source_core = tile[
                                core_y:core_y + core_height,
                                core_x:core_x + core_width,
                            ]
                            composite_output[top:bottom, left:right] = self.compose(
                                source_core,
                                alpha,
                            )
                    current += 1
                    if progress_callback is not None:
                        progress_callback(current, total, f"正在处理分块 {current}/{total}")
            output.flush()
            if composite_output is not None:
                composite_output.flush()
            temporary_output = os.path.join(
                os.path.dirname(target),
                f".{Path(target).stem}.tmp{Path(target).suffix}",
            )
            if kind == "photoshop":
                if composite_output is None:
                    raise RuntimeError("Photoshop 兼容预览未生成。")
                if progress_callback is not None:
                    progress_callback(total, total, "正在创建 Photoshop 分层文件")
                self._save_photoshop_document(
                    image,
                    output,
                    composite_output,
                    temporary_output,
                    project,
                    psb=psb,
                    cancelled=cancelled,
                )
            else:
                pil_output = Image.fromarray(output, "RGB" if channels == 3 else "RGBA")
                save_options: dict[str, object] = {}
                if suffix in {".tif", ".tiff"}:
                    save_options.update(compression="tiff_lzw", big_tiff=True)
                else:
                    save_options.update(compress_level=4)
                pil_output.save(temporary_output, **save_options)
                del pil_output
            output._mmap.close()
            output = None
            if composite_output is not None:
                composite_output._mmap.close()
                composite_output = None
            os.replace(temporary_output, target)
            temporary_output = ""
        finally:
            image.close()
            if output is not None:
                output._mmap.close()
                output = None
            if composite_output is not None:
                composite_output._mmap.close()
                composite_output = None
            if temporary_output and os.path.exists(temporary_output):
                os.remove(temporary_output)
            if temporary_raw and os.path.exists(temporary_raw):
                os.remove(temporary_raw)
            if temporary_composite_raw and os.path.exists(temporary_composite_raw):
                os.remove(temporary_composite_raw)
        return ImageLabExportResult(
            output_path=target,
            kind=kind,
            elapsed_seconds=time.perf_counter() - started,
            width=project.source_width,
            height=project.source_height,
        )

    @staticmethod
    def _install_psd_preview(psd: Any, composite: Image.Image, compression: Any) -> None:
        preview = composite.convert(psd.pil_mode)
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

    def _save_photoshop_document(
        self,
        source: Image.Image,
        cleanup_pixels: np.memmap,
        composite_pixels: np.memmap,
        target: str,
        project: ImageLabProject,
        *,
        psb: bool,
        cancelled: Callable[[], bool] | None,
    ) -> None:
        """写入适合后续精修的原稿、白色清理层和空白修补层。"""

        if cancelled is not None and cancelled():
            raise ImageLabCancelled("已停止 Photoshop 分层导出。")
        try:
            from psd_tools import PSDImage
            from psd_tools.constants import Compression, ProtectedFlags, Resource
            from psd_tools.psd.image_resources import ImageResource
        except ImportError as exc:
            raise RuntimeError("缺少 PSD 写入组件，请重新安装程序依赖。") from exc

        width, height = source.size
        compression = Compression.RLE
        psd = PSDImage.new(
            "RGBA",
            (width, height),
            color=(255, 255, 255, 255),
            compression=compression,
        )
        if psb:
            psd._record.header.version = 2

        dpi_x = project.source_dpi_x if project.source_dpi_x > 0 else 300.0
        dpi_y = project.source_dpi_y if project.source_dpi_y > 0 else dpi_x
        resolution_data = struct.pack(
            ">IHHIHH",
            int(round(dpi_x * 0x10000)),
            1,
            2,
            int(round(dpi_y * 0x10000)),
            1,
            2,
        )
        psd.image_resources[Resource.RESOLUTION_INFO] = ImageResource(
            signature=b"8BIM",
            key=Resource.RESOLUTION_INFO,
            name="",
            data=resolution_data,
        )
        srgb_profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
        psd.image_resources[Resource.ICC_PROFILE] = ImageResource(
            signature=b"8BIM",
            key=Resource.ICC_PROFILE,
            name="",
            data=srgb_profile.tobytes(),
        )

        source_rgb = source.convert("RGB")
        original_layer = psd.create_pixel_layer(
            name="原稿（锁定）",
            image=source_rgb,
            top=0,
            left=0,
            compression=compression,
        )
        original_lock_flags = (
            ProtectedFlags.TRANSPARENCY
            | ProtectedFlags.COMPOSITE
            | ProtectedFlags.POSITION
        )
        original_layer.lock(original_lock_flags)
        # psd-tools 首次 lock() 只建立记录，需要对已安装记录写入实际位值。
        if original_layer.locks is not None:
            original_layer.locks.lock(original_lock_flags)
        source_rgb.close()

        cleanup_image = Image.fromarray(cleanup_pixels, "RGBA")
        psd.create_pixel_layer(
            name="白色清理层",
            image=cleanup_image,
            top=0,
            left=0,
            compression=compression,
        )
        cleanup_image.close()

        repair_image = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        psd.create_pixel_layer(
            name="笔画修补",
            image=repair_image,
            top=0,
            left=0,
            compression=compression,
        )
        repair_image.close()

        if cancelled is not None and cancelled():
            raise ImageLabCancelled("已停止 Photoshop 分层导出。")
        composite_image = Image.fromarray(composite_pixels, "RGB")
        self._install_psd_preview(psd, composite_image, compression)
        composite_image.close()
        psd.save(target, encoding="gb18030", compression=compression)

    @staticmethod
    def _page_guide_tile(
        guide: np.ndarray,
        left: int,
        top: int,
        right: int,
        bottom: int,
        width: int,
        height: int,
        *,
        output_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        guide_height, guide_width = guide.shape
        x0 = max(0, int(left * guide_width / width) - 1)
        y0 = max(0, int(top * guide_height / height) - 1)
        x1 = min(guide_width, int(np.ceil(right * guide_width / width)) + 1)
        y1 = min(guide_height, int(np.ceil(bottom * guide_height / height)) + 1)
        crop = guide[y0:y1, x0:x1]
        target_size = output_size or (right - left, bottom - top)
        return cv2.resize(
            crop,
            target_size,
            interpolation=cv2.INTER_NEAREST,
        )
