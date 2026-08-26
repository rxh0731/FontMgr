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


SUPPORTED_IMAGE_FILTER = (
    "文字图片 (*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.webp);;所有文件 (*)"
)
PSD_MAX_DIMENSION = 30_000
PSB_MAX_DIMENSION = 300_000
PSD_SAFE_FILE_BYTES = 1_800_000_000


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
    cleanup: ImageCleanupResult
    effective_alpha: np.ndarray
    composite: np.ndarray
    source_width: int
    source_height: int
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

    def load_preview(
        self,
        project: ImageLabProject,
        *,
        max_edge: int = 2200,
    ) -> ImageLabPreview:
        if max_edge < 320:
            raise ValueError("预览尺寸过小。")
        started = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            image = self._open_image(project.source_path)
            try:
                image = self._apply_exif_orientation(image)
                image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
                rgb_image = image.convert("RGB")
                try:
                    source = np.array(rgb_image, dtype=np.uint8, copy=True)
                finally:
                    rgb_image.close()
            finally:
                image.close()
        cleanup = clean_document_image(source, project.options)
        effective_alpha = self.apply_strokes(
            cleanup.cleanup_layer[:, :, 3],
            project.strokes,
            project.source_width,
            project.source_height,
        )
        composite = self.compose(source, effective_alpha)
        for array in (source, effective_alpha, composite):
            array.setflags(write=False)
        return ImageLabPreview(
            source=source,
            cleanup=cleanup,
            effective_alpha=effective_alpha,
            composite=composite,
            source_width=project.source_width,
            source_height=project.source_height,
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
    ) -> np.ndarray:
        """按原图坐标把白色覆盖和还原笔画作用到清理层。"""

        result = np.array(cleanup_alpha, dtype=np.uint8, copy=True)
        target_height, target_width = result.shape
        offset_x, offset_y = source_offset
        scale_x = target_width / max(1, source_width)
        scale_y = target_height / max(1, source_height)
        if source_offset != (0, 0):
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
        preview = self.load_preview(project, max_edge=1600)
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
    ) -> np.ndarray:
        guide_height, guide_width = guide.shape
        x0 = max(0, int(left * guide_width / width) - 1)
        y0 = max(0, int(top * guide_height / height) - 1)
        x1 = min(guide_width, int(np.ceil(right * guide_width / width)) + 1)
        y1 = min(guide_height, int(np.ceil(bottom * guide_height / height)) + 1)
        crop = guide[y0:y1, x0:x1]
        return cv2.resize(
            crop,
            (right - left, bottom - top),
            interpolation=cv2.INTER_NEAREST,
        )
