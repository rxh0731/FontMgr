"""整幅文献图片的多通道非破坏背景清理算法。"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import cv2
import numpy as np

from core.algorithms import sauvola_binarize, wolf_binarize


RECOGNITION_METHOD = "多通道通用识别"


@dataclass(frozen=True, slots=True)
class ImageCleanupOptions:
    """一次背景清理的稳定参数。"""

    strength: int = 50
    preserve_faint_ink: bool = True
    remove_small_noise: bool = True
    detect_page: bool = True

    def __post_init__(self) -> None:
        if not 0 <= int(self.strength) <= 100:
            raise ValueError("清理强度必须在 0 到 100 之间。")
        if not isinstance(self.preserve_faint_ink, bool):
            raise TypeError("保护浅色和残损笔迹必须是布尔值。")
        if not isinstance(self.remove_small_noise, bool):
            raise TypeError("清除孤立小噪点必须是布尔值。")
        if not isinstance(self.detect_page, bool):
            raise TypeError("文献边界检测必须是布尔值。")


@dataclass(frozen=True, slots=True)
class ImageCleanupCalibration:
    """可跨预览和完整尺寸分块复用的背景色域校准。"""

    background_palette: tuple[tuple[int, int], ...]
    color_support_distance_sq: int
    color_seed_distance_sq: int
    colorful_document: bool

    def __post_init__(self) -> None:
        if not self.background_palette:
            raise ValueError("背景色域调色板不能为空。")
        for center in self.background_palette:
            if len(center) != 2 or any(not 0 <= int(value) <= 255 for value in center):
                raise ValueError("背景色域调色板的 Lab 色度值无效。")
        if int(self.color_support_distance_sq) < 0:
            raise ValueError("彩色笔迹弱阈值不能为负数。")
        if int(self.color_seed_distance_sq) < int(self.color_support_distance_sq):
            raise ValueError("彩色笔迹强阈值不能小于弱阈值。")


@dataclass(frozen=True, slots=True)
class ImageCleanupResult:
    """背景清理结果；所有数组均为只读副本。"""

    cleanup_layer: np.ndarray
    composite: np.ndarray
    foreground_mask: np.ndarray
    uncertainty_mask: np.ndarray
    page_mask: np.ndarray
    resolved_profile: str
    metrics: Mapping[str, float | int | str]
    calibration: ImageCleanupCalibration

    def __post_init__(self) -> None:
        arrays = {
            "cleanup_layer": (self.cleanup_layer, 3, 4),
            "composite": (self.composite, 3, 3),
            "foreground_mask": (self.foreground_mask, 2, None),
            "uncertainty_mask": (self.uncertainty_mask, 2, None),
            "page_mask": (self.page_mask, 2, None),
        }
        shape: tuple[int, int] | None = None
        for field_name, (value, dimensions, channels) in arrays.items():
            if not isinstance(value, np.ndarray) or value.ndim != dimensions:
                raise TypeError(f"{field_name} 必须是 {dimensions} 维 NumPy 数组。")
            if value.dtype != np.uint8:
                raise TypeError(f"{field_name} 必须使用 uint8。")
            if channels is not None and value.shape[2] != channels:
                raise ValueError(f"{field_name} 的通道数量必须为 {channels}。")
            current_shape = tuple(value.shape[:2])
            if shape is None:
                shape = current_shape
            elif current_shape != shape:
                raise ValueError("背景清理结果中的全部图像尺寸必须一致。")
            stored = np.array(value, copy=True)
            stored.setflags(write=False)
            object.__setattr__(self, field_name, stored)
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))


def _odd_size(value: float, minimum: int, maximum: int) -> int:
    size = max(minimum, min(maximum, int(round(value))))
    return size if size % 2 == 1 else size + 1


def _normalize_rgb(source: np.ndarray) -> np.ndarray:
    if not isinstance(source, np.ndarray) or source.size == 0:
        raise ValueError("待清理图片不能为空。")
    if source.ndim == 2:
        rgb = np.repeat(source[:, :, None], 3, axis=2)
    elif source.ndim == 3 and source.shape[2] in (3, 4):
        rgb = source[:, :, :3]
    else:
        raise ValueError("待清理图片必须是灰度、RGB 或 RGBA 图像。")
    if rgb.shape[0] < 3 or rgb.shape[1] < 3:
        raise ValueError("待清理图片尺寸过小。")
    if rgb.dtype == np.uint8:
        return np.array(rgb, copy=True)
    if not np.issubdtype(rgb.dtype, np.number) or not np.isfinite(rgb).all():
        raise TypeError("待清理图片必须使用有限数值像素。")
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _estimate_multichannel_background(
    lab: np.ndarray,
    kernel_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """分别估计深字、浅字和颜色通道的局部背景。"""

    lightness = lab[:, :, 0]
    element = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    dark_background = cv2.morphologyEx(lightness, cv2.MORPH_CLOSE, element)
    light_background = cv2.morphologyEx(lightness, cv2.MORPH_OPEN, element)
    sigma = max(1.0, kernel_size / 12.0)
    dark_background = cv2.GaussianBlur(dark_background, (0, 0), sigmaX=sigma)
    light_background = cv2.GaussianBlur(light_background, (0, 0), sigmaX=sigma)
    chroma = lab[:, :, 1:].astype(np.float32)
    chroma_background = cv2.GaussianBlur(
        chroma,
        (0, 0),
        sigmaX=max(2.0, kernel_size / 7.0),
    )
    return dark_background, light_background, chroma_background


def _foreground_evidence(
    lab: np.ndarray,
    dark_background: np.ndarray,
    light_background: np.ndarray,
    chroma_background: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成与颜色和明暗方向无关的前景证据。"""

    lightness = lab[:, :, 0].astype(np.float32)
    dark_delta = np.maximum(dark_background.astype(np.float32) - lightness, 0.0)
    light_delta = np.maximum(lightness - light_background.astype(np.float32), 0.0)
    luminance_delta = np.maximum(dark_delta, light_delta)
    chroma = lab[:, :, 1:].astype(np.float32)
    chroma_delta = np.sqrt(np.sum((chroma - chroma_background) ** 2, axis=2))
    evidence = np.sqrt(luminance_delta**2 + (chroma_delta * 1.15) ** 2)
    return evidence, luminance_delta, chroma_delta


def _stroke_detail_evidence(
    lab: np.ndarray,
    short_side: int,
) -> np.ndarray:
    """提取笔画尺度的局部明暗和颜色变化，抑制缓慢变化的纸张底纹。"""

    kernel_size = _odd_size(short_side * 0.012, 9, 21)
    element = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    lightness = lab[:, :, 0]
    dark_detail = cv2.morphologyEx(lightness, cv2.MORPH_BLACKHAT, element)
    light_detail = cv2.morphologyEx(lightness, cv2.MORPH_TOPHAT, element)

    chroma = lab[:, :, 1:].astype(np.float32)
    chroma_background = cv2.GaussianBlur(
        chroma,
        (0, 0),
        sigmaX=max(1.0, kernel_size / 3.5),
    )
    chroma_detail = np.sqrt(
        np.sum((chroma - chroma_background) ** 2, axis=2)
    )
    return np.maximum.reduce(
        (
            dark_detail.astype(np.float32),
            light_detail.astype(np.float32),
            chroma_detail * 1.25,
        )
    )


def _background_chroma_palette(
    lab: np.ndarray,
    page_mask: np.ndarray,
    stroke_detail: np.ndarray,
) -> tuple[tuple[int, int], ...]:
    """从各区域低细节像素提取纸张背景色，保留缓慢变化的底色范围。"""

    height, width = page_mask.shape
    page_detail = stroke_detail[page_mask]
    detail_limit = (
        float(np.percentile(page_detail, 55.0))
        if page_detail.size
        else float(np.percentile(stroke_detail, 55.0))
    )
    low_detail = page_mask & (stroke_detail <= max(1.0, detail_limit))
    centers: list[tuple[int, int]] = []

    def append_center(center: tuple[int, int]) -> None:
        if center not in centers:
            centers.append(center)

    for row in range(4):
        top = height * row // 4
        bottom = height * (row + 1) // 4
        for column in range(4):
            left = width * column // 4
            right = width * (column + 1) // 4
            region_mask = low_detail[top:bottom, left:right]
            if np.count_nonzero(region_mask) < 16:
                region_mask = page_mask[top:bottom, left:right]
            chroma = lab[top:bottom, left:right, 1:][region_mask]
            if chroma.size == 0:
                continue
            center = tuple(
                int(round(float(value)))
                for value in np.median(chroma, axis=0)
            )
            append_center(center)

    background_chroma = lab[:, :, 1:][low_detail]
    if background_chroma.size:
        sample_step = max(1, len(background_chroma) // 250_000)
        sample = background_chroma[::sample_step]
        ranges = np.percentile(sample, 95.0, axis=0) - np.percentile(
            sample,
            5.0,
            axis=0,
        )
        direction = int(np.argmax(ranges))
        direction_values = sample[:, direction].astype(np.float32)
        for quantile in (5.0, 20.0, 40.0, 60.0, 80.0, 92.0, 97.0, 99.5):
            target = float(np.percentile(direction_values, quantile))
            band = sample[np.abs(direction_values - target) <= 1.0]
            if band.size == 0:
                continue
            center = tuple(
                int(round(float(value)))
                for value in np.median(band, axis=0)
            )
            append_center(center)
    if centers:
        return tuple(centers)
    fallback = np.median(lab[:, :, 1:].reshape(-1, 2), axis=0)
    return ((int(round(float(fallback[0]))), int(round(float(fallback[1])))),)


def _background_chroma_distance_sq(
    lab: np.ndarray,
    palette: tuple[tuple[int, int], ...],
) -> np.ndarray:
    """通过 256×256 查找表计算像素到最近背景色的平方距离。"""

    values = np.arange(256, dtype=np.int32)
    first = values[:, None]
    second = values[None, :]
    distance_table = np.full((256, 256), np.iinfo(np.int32).max, dtype=np.int32)
    for center_a, center_b in palette:
        distance = (first - int(center_a)) ** 2 + (second - int(center_b)) ** 2
        np.minimum(distance_table, distance, out=distance_table)
    return distance_table[lab[:, :, 1], lab[:, :, 2]]


def _build_cleanup_calibration(
    lab: np.ndarray,
    page_mask: np.ndarray,
    stroke_detail: np.ndarray,
    strength_ratio: float,
    preserve_faint_ink: bool,
) -> ImageCleanupCalibration:
    palette = _background_chroma_palette(lab, page_mask, stroke_detail)
    distance_sq = _background_chroma_distance_sq(lab, palette)
    support_distance = 6.0 + strength_ratio * 8.0
    seed_distance = 12.0 + strength_ratio * 11.8
    if preserve_faint_ink:
        support_distance -= 0.8
        seed_distance -= 1.3
    page_distances = distance_sq[page_mask]
    color_peak = (
        float(np.sqrt(np.percentile(page_distances, 99.0)))
        if page_distances.size
        else 0.0
    )
    colorful_document = color_peak >= seed_distance
    return ImageCleanupCalibration(
        background_palette=palette,
        color_support_distance_sq=int(round(support_distance**2)),
        color_seed_distance_sq=int(round(seed_distance**2)),
        colorful_document=colorful_document,
    )


def _component_touch_count(mask: np.ndarray) -> int:
    return sum(
        bool(edge.any())
        for edge in (mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1])
    )


def _document_surface(lab: np.ndarray) -> np.ndarray:
    """对亮纸、暗纸和满幅扫描统一定位文献主体。"""

    height, width = lab.shape[:2]
    lightness = lab[:, :, 0]
    smooth = cv2.GaussianBlur(
        lightness,
        (0, 0),
        sigmaX=max(2.0, min(height, width) / 90.0),
    )
    threshold, _binary = cv2.threshold(
        smooth,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    candidates = (smooth >= int(threshold), smooth < int(threshold))
    center_y, center_x = height // 2, width // 2
    best_mask: np.ndarray | None = None
    best_score = float("-inf")
    close_size = _odd_size(min(height, width) * 0.05, 9, 121)
    element = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (close_size, close_size),
    )
    for candidate in candidates:
        closed = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_CLOSE, element)
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            closed,
            8,
            cv2.CV_32S,
        )
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            ratio = area / max(1, height * width)
            if ratio < 0.18 or ratio > 0.985:
                continue
            component = labels == label
            contains_center = bool(component[center_y, center_x])
            touches = _component_touch_count(component)
            score = ratio * 4.0 + (1.5 if contains_center else 0.0) - touches * 0.3
            if score > best_score:
                best_score = score
                best_mask = component
    if best_mask is None:
        return np.ones((height, width), dtype=bool)
    contours, _hierarchy = cv2.findContours(
        best_mask.astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    filled = np.zeros((height, width), dtype=np.uint8)
    if contours:
        points = np.vstack(contours)
        hull = cv2.convexHull(points)
        x, y, component_width, component_height = cv2.boundingRect(hull)
        if (
            component_width >= int(width * 0.85)
            and component_height >= int(height * 0.85)
        ):
            filled[y : y + component_height, x : x + component_width] = 255
        else:
            cv2.drawContours(filled, [hull], -1, 255, thickness=cv2.FILLED)
    return filled > 0


def _reconstruct_from_seeds(
    seed: np.ndarray,
    support: np.ndarray,
    max_distance: int,
) -> np.ndarray:
    """在有限笔画距离内恢复弱边缘，禁止沿连续纸纹扩散到整页。"""

    marker = seed.astype(np.uint8)
    support_u8 = support.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _step in range(max(1, int(max_distance))):
        grown = cv2.dilate(marker, kernel)
        current = cv2.bitwise_and(grown, support_u8)
        current = cv2.bitwise_or(current, marker)
        if np.array_equal(current, marker):
            break
        marker = current
    return marker > 0


def _remove_isolated_noise(
    foreground: np.ndarray,
    evidence: np.ndarray,
    seed_threshold: float,
    short_side: int,
    preserve_faint_ink: bool,
) -> np.ndarray:
    """只删除尺寸很小且缺乏强前景证据的孤立域。"""

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        foreground.astype(np.uint8),
        8,
        cv2.CV_32S,
    )
    if count <= 1:
        return foreground
    area_limit = max(2, int(round((short_side / 900.0) ** 2 * 3.0)))
    if preserve_faint_ink:
        area_limit = max(1, area_limit // 2)
    keep = np.ones(count, dtype=bool)
    keep[0] = False
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area > area_limit or max(component_width, component_height) > area_limit * 2:
            continue
        component_values = evidence[labels == label]
        if component_values.size and float(component_values.max()) < seed_threshold * 1.45:
            keep[label] = False
    return keep[labels]


def clean_document_image(
    source: np.ndarray,
    options: ImageCleanupOptions | None = None,
    calibration: ImageCleanupCalibration | None = None,
) -> ImageCleanupResult:
    """生成白色透明清理层、文字保留蒙版和合成预览。

    算法只比较像素与局部背景的多通道差异，不假定文字是黑色、背景是
    白色，也不修改输入数组。清理层 RGB 固定为白色，Alpha 表示覆盖强度。
    """

    settings = options or ImageCleanupOptions()
    rgb = _normalize_rgb(source)
    height, width = rgb.shape[:2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness = lab[:, :, 0]
    short_side = min(height, width)
    background_kernel = _odd_size(short_side * 0.04, 31, 181)
    local_window = _odd_size(short_side * 0.016, 21, 81)
    dark_background, light_background, chroma_background = (
        _estimate_multichannel_background(lab, background_kernel)
    )
    evidence, luminance_delta, chroma_delta = _foreground_evidence(
        lab,
        dark_background,
        light_background,
        chroma_background,
    )
    stroke_detail = _stroke_detail_evidence(lab, short_side)
    page_mask = (
        _document_surface(lab)
        if settings.detect_page
        else np.ones((height, width), dtype=bool)
    )

    strength_ratio = int(settings.strength) / 100.0
    resolved_calibration = calibration or _build_cleanup_calibration(
        lab,
        page_mask,
        stroke_detail,
        strength_ratio,
        settings.preserve_faint_ink,
    )
    background_chroma_distance_sq = _background_chroma_distance_sq(
        lab,
        resolved_calibration.background_palette,
    )
    support_threshold = 5.5 + strength_ratio * 11.0
    seed_threshold = 15.0 + strength_ratio * 22.0
    if settings.preserve_faint_ink:
        support_threshold -= 2.0
        seed_threshold -= 4.5

    page_evidence = evidence[page_mask]
    support_quantile = 58.0 + strength_ratio * 25.0
    seed_quantile = 82.0 + strength_ratio * 12.0
    if settings.preserve_faint_ink:
        support_quantile -= 8.0
        seed_quantile -= 4.0
    if page_evidence.size:
        support_threshold = max(
            support_threshold,
            float(np.percentile(page_evidence, support_quantile)),
        )
        seed_threshold = max(
            seed_threshold,
            float(np.percentile(page_evidence, seed_quantile)),
        )

    dark_local = sauvola_binarize(
        lightness,
        window=local_window,
        k=0.14 + strength_ratio * 0.10,
    ) > 0
    light_local = sauvola_binarize(
        255 - lightness,
        window=local_window,
        k=0.14 + strength_ratio * 0.10,
    ) > 0
    dark_wolf = wolf_binarize(
        lightness,
        window=local_window,
        k=0.22 + strength_ratio * 0.16,
    ) > 0
    light_wolf = wolf_binarize(
        255 - lightness,
        window=local_window,
        k=0.22 + strength_ratio * 0.16,
    ) > 0
    local_structure = dark_local | light_local | dark_wolf | light_wolf

    detail_support_threshold = 2.8 + strength_ratio * 4.2
    detail_seed_threshold = 7.0 + strength_ratio * 7.0
    if settings.preserve_faint_ink:
        detail_support_threshold -= 0.9
        detail_seed_threshold -= 1.8

    structural_detail = local_structure & (
        stroke_detail >= detail_support_threshold
    )
    detail_neighborhood = cv2.dilate(
        (stroke_detail >= detail_support_threshold).astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    ) > 0
    colorful_document = resolved_calibration.colorful_document
    if colorful_document:
        palette_center = np.median(
            np.asarray(resolved_calibration.background_palette, dtype=np.float32),
            axis=0,
        )
        neutral_background = bool(
            np.sum((palette_center - np.array((128.0, 128.0))) ** 2) <= 100.0
        )
        neutral_support_quantile = 84.0 + strength_ratio * 10.0
        neutral_seed_quantile = 92.0 + strength_ratio * 6.0
        if settings.preserve_faint_ink:
            neutral_support_quantile -= 5.0
            neutral_seed_quantile -= 2.0
        page_luminance = luminance_delta[page_mask]
        neutral_support_threshold = max(
            support_threshold,
            float(np.percentile(page_luminance, neutral_support_quantile)),
        )
        neutral_seed_threshold = max(
            seed_threshold,
            float(np.percentile(page_luminance, neutral_seed_quantile)),
        )
        color_support = (
            (
                background_chroma_distance_sq
                >= resolved_calibration.color_support_distance_sq
            )
            & detail_neighborhood
        )
        neutral_support = np.zeros((height, width), dtype=bool)
        if neutral_background:
            neutral_support = (
                (luminance_delta >= neutral_support_threshold)
                & structural_detail
            )
        seed = (
            (
                background_chroma_distance_sq
                >= resolved_calibration.color_seed_distance_sq
            )
            & (stroke_detail >= detail_support_threshold)
        )
        if neutral_background:
            seed |= (
                (luminance_delta >= neutral_seed_threshold)
                & (stroke_detail >= detail_seed_threshold)
            )
        support = color_support | neutral_support | seed
    else:
        seed = (
            (stroke_detail >= detail_seed_threshold)
            & (evidence >= support_threshold * 0.7)
        )
        seed |= (
            structural_detail
            & (evidence >= seed_threshold)
        )
        support = (
            detail_neighborhood
            & (evidence >= support_threshold)
        )
        support |= (
            structural_detail
            & (evidence >= support_threshold * 0.55)
        )
        support |= seed
    seed &= page_mask
    support &= page_mask
    reconstruction_distance = max(1, min(6, int(round(short_side / 450.0))))
    foreground = _reconstruct_from_seeds(
        seed,
        support,
        reconstruction_distance,
    )
    if settings.remove_small_noise:
        foreground = _remove_isolated_noise(
            foreground,
            evidence,
            seed_threshold,
            short_side,
            settings.preserve_faint_ink,
        )

    edge_support = cv2.dilate(
        foreground.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    ) > 0
    foreground |= edge_support & (evidence >= max(1.5, support_threshold * 0.42))

    foreground_u8 = foreground.astype(np.uint8) * 255
    foreground_alpha = cv2.GaussianBlur(foreground_u8, (3, 3), 0.55)
    foreground_alpha[foreground] = 255
    foreground_alpha[~page_mask] = 0
    cleanup_alpha = 255 - foreground_alpha

    cleanup_layer = np.empty((height, width, 4), dtype=np.uint8)
    cleanup_layer[:, :, :3] = 255
    cleanup_layer[:, :, 3] = cleanup_alpha
    alpha = foreground_alpha.astype(np.float32)[:, :, None] / 255.0
    composite = np.clip(
        rgb.astype(np.float32) * alpha + 255.0 * (1.0 - alpha),
        0,
        255,
    ).astype(np.uint8)

    uncertain_low = max(0.0, support_threshold * 0.75)
    uncertain_high = seed_threshold * 1.05
    uncertainty = page_mask & (evidence >= uncertain_low) & ~foreground
    uncertainty &= evidence < uncertain_high
    if colorful_document:
        uncertainty &= (
            (
                background_chroma_distance_sq
                >= int(resolved_calibration.color_support_distance_sq * 0.42)
            )
            | (
                local_structure
                & (luminance_delta >= neutral_support_threshold * 0.8)
            )
        )
    uncertainty |= local_structure & ~foreground & page_mask
    uncertainty_mask = uncertainty.astype(np.uint8) * 255

    metrics: dict[str, float | int | str] = {
        "识别方式": RECOGNITION_METHOD,
        "彩色笔迹优先": "是" if colorful_document else "否",
        "清理强度": int(settings.strength),
        "背景核大小": background_kernel,
        "局部窗口": local_window,
        "弱笔迹证据阈值": round(float(support_threshold), 4),
        "强笔迹证据阈值": round(float(seed_threshold), 4),
        "文献区域占比": round(float(np.mean(page_mask)), 6),
        "保留前景占比": round(float(np.mean(foreground_alpha > 0)), 6),
        "完全清理占比": round(float(np.mean(cleanup_alpha == 255)), 6),
        "待核对占比": round(float(np.mean(uncertainty)), 6),
        "亮度差异90分位": round(float(np.percentile(luminance_delta, 90.0)), 4),
        "颜色差异90分位": round(float(np.percentile(chroma_delta, 90.0)), 4),
        "背景色域数量": len(resolved_calibration.background_palette),
        "背景色域弱阈值": round(
            float(np.sqrt(resolved_calibration.color_support_distance_sq)),
            4,
        ),
        "背景色域强阈值": round(
            float(np.sqrt(resolved_calibration.color_seed_distance_sq)),
            4,
        ),
    }
    return ImageCleanupResult(
        cleanup_layer=cleanup_layer,
        composite=composite,
        foreground_mask=foreground_alpha,
        uncertainty_mask=uncertainty_mask,
        page_mask=page_mask.astype(np.uint8) * 255,
        resolved_profile=RECOGNITION_METHOD,
        metrics=metrics,
        calibration=resolved_calibration,
    )
