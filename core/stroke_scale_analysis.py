"""基于笔画尺度的密集细噪分析与核心重建。"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from functools import wraps
import hashlib
from threading import RLock
from typing import Callable, Iterator, ParamSpec, TypeVar

import cv2
import numpy as np


class ReconstructionStrength(str, Enum):
    """笔画尺度重建强度。"""

    CONSERVATIVE = "保守"
    BALANCED = "均衡"
    STRONG = "强力"


@dataclass(frozen=True)
class StrokeScaleAnalysis:
    """灰度图的笔画与细噪尺度分析结果。

    掩码均为 ``uint8`` 二值数组（0/1）。``base_mask`` 是低置信回退基准；
    ``core_mask``、``body_mask`` 和 ``support_mask`` 可供后续优化或评分复用。
    """

    source_gray: np.ndarray
    base_mask: np.ndarray
    core_mask: np.ndarray
    safety_mask: np.ndarray
    body_mask: np.ndarray
    conservative_body_mask: np.ndarray
    support_mask: np.ndarray
    independent_noise_mask: np.ndarray
    confidence: float
    applicable: bool
    reason: str
    stroke_scale: float
    noise_scale: float
    otsu_threshold: int
    core_threshold: int
    support_threshold: int
    noise_component_count: int
    metrics: dict[str, float]


@dataclass(frozen=True)
class StrokeScaleReconstruction:
    """单档笔画尺度重建结果。"""

    strength: ReconstructionStrength
    mask: np.ndarray
    removed_mask: np.ndarray
    added_mask: np.ndarray
    confidence: float
    applied: bool
    reason: str
    core_retention: float
    removed_ratio: float
    added_ratio: float


_CacheKey = tuple[tuple[int, int], bytes, float, int]
_P = ParamSpec("_P")
_R = TypeVar("_R")


def _analysis_arrays(analysis: StrokeScaleAnalysis) -> tuple[np.ndarray, ...]:
    return (
        analysis.source_gray,
        analysis.base_mask,
        analysis.core_mask,
        analysis.safety_mask,
        analysis.body_mask,
        analysis.conservative_body_mask,
        analysis.support_mask,
        analysis.independent_noise_mask,
    )


def _copy_analysis(
    analysis: StrokeScaleAnalysis,
    *,
    readonly: bool,
) -> StrokeScaleAnalysis:
    arrays = [np.array(item, copy=True, order="C") for item in _analysis_arrays(analysis)]
    if readonly:
        for item in arrays:
            item.setflags(write=False)
    return StrokeScaleAnalysis(
        source_gray=arrays[0],
        base_mask=arrays[1],
        core_mask=arrays[2],
        safety_mask=arrays[3],
        body_mask=arrays[4],
        conservative_body_mask=arrays[5],
        support_mask=arrays[6],
        independent_noise_mask=arrays[7],
        confidence=analysis.confidence,
        applicable=analysis.applicable,
        reason=analysis.reason,
        stroke_scale=analysis.stroke_scale,
        noise_scale=analysis.noise_scale,
        otsu_threshold=analysis.otsu_threshold,
        core_threshold=analysis.core_threshold,
        support_threshold=analysis.support_threshold,
        noise_component_count=analysis.noise_component_count,
        metrics=dict(analysis.metrics),
    )


class _StrokeScaleSessionCache:
    """一次寻优任务内使用的线程安全、有界分析缓存。"""

    def __init__(
        self,
        max_items: int = 8,
        max_bytes: int = 32 * 1024 * 1024,
        max_entry_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        self.max_items = max_items
        self.max_bytes = max_bytes
        self.max_entry_bytes = max_entry_bytes
        self._entries: OrderedDict[_CacheKey, tuple[StrokeScaleAnalysis, int]] = OrderedDict()
        self._total_bytes = 0
        self._lock = RLock()

    def get(
        self,
        key: _CacheKey,
        source: np.ndarray,
    ) -> StrokeScaleAnalysis | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            cached, _size = entry
            if not np.array_equal(cached.source_gray, source):
                return None
            self._entries.move_to_end(key)
        return _copy_analysis(cached, readonly=False)

    def put(self, key: _CacheKey, analysis: StrokeScaleAnalysis) -> None:
        cached = _copy_analysis(analysis, readonly=True)
        size = sum(item.nbytes for item in _analysis_arrays(cached))
        if size > self.max_entry_bytes or size > self.max_bytes:
            return
        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._total_bytes -= previous[1]
            self._entries[key] = (cached, size)
            self._total_bytes += size
            while (
                len(self._entries) > self.max_items
                or self._total_bytes > self.max_bytes
            ):
                _old_key, (_old_value, old_size) = self._entries.popitem(last=False)
                self._total_bytes -= old_size

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._total_bytes = 0


_SESSION_CACHE: ContextVar[_StrokeScaleSessionCache | None] = ContextVar(
    "stroke_scale_session_cache",
    default=None,
)


@contextmanager
def stroke_scale_analysis_session() -> Iterator[None]:
    """在一次自动优化任务内复用相同输入的尺度分析。"""
    existing = _SESSION_CACHE.get()
    if existing is not None:
        yield
        return
    cache = _StrokeScaleSessionCache()
    token = _SESSION_CACHE.set(cache)
    try:
        yield
    finally:
        cache.clear()
        _SESSION_CACHE.reset(token)


def use_stroke_scale_analysis_cache(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """让一个入口函数在执行期间启用任务内尺度分析缓存。"""

    @wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with stroke_scale_analysis_session():
            return func(*args, **kwargs)

    return wrapper


@dataclass(frozen=True)
class _StrengthProfile:
    center_radius_ratio: float
    growth_radius_ratio: float
    component_seed_ratio: float
    component_retention: float
    minimum_core_retention: float
    minimum_safety_component_retention: float


_STRENGTH_PROFILES = {
    ReconstructionStrength.CONSERVATIVE: _StrengthProfile(
        center_radius_ratio=0.14,
        growth_radius_ratio=0.60,
        component_seed_ratio=0.30,
        component_retention=0.92,
        minimum_core_retention=0.985,
        minimum_safety_component_retention=0.98,
    ),
    ReconstructionStrength.BALANCED: _StrengthProfile(
        center_radius_ratio=0.20,
        growth_radius_ratio=0.52,
        component_seed_ratio=0.38,
        component_retention=0.84,
        minimum_core_retention=0.965,
        minimum_safety_component_retention=0.84,
    ),
    ReconstructionStrength.STRONG: _StrengthProfile(
        center_radius_ratio=0.26,
        growth_radius_ratio=0.46,
        component_seed_ratio=0.46,
        component_retention=0.74,
        minimum_core_retention=0.940,
        minimum_safety_component_retention=0.74,
    ),
}


def _as_gray_u8(gray: np.ndarray) -> np.ndarray:
    """统一二维灰度输入；PIL 灰度图可直接经 ``np.asarray`` 转换。"""
    source = np.asarray(gray)
    if source.ndim != 2:
        raise ValueError("笔画尺度分析仅支持二维灰度图")
    if source.size == 0:
        raise ValueError("灰度图不能为空")
    return np.clip(source, 0, 255).astype(np.uint8)


def _otsu_separability(gray: np.ndarray) -> float:
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = float(histogram.sum())
    if total <= 0.0:
        return 0.0
    probabilities = histogram / total
    levels = np.arange(256, dtype=np.float64)
    global_mean = float(np.sum(probabilities * levels))
    total_variance = float(np.sum(probabilities * (levels - global_mean) ** 2))
    if total_variance <= 1e-9:
        return 0.0
    weights = np.cumsum(probabilities)
    means = np.cumsum(probabilities * levels)
    denominator = weights * (1.0 - weights)
    between = np.zeros_like(denominator)
    valid = denominator > 1e-12
    between[valid] = (global_mean * weights[valid] - means[valid]) ** 2 / denominator[valid]
    return float(np.clip(float(between.max()) / total_variance, 0.0, 1.0))


def _mask_stroke_scale(mask: np.ndarray) -> tuple[float, int]:
    """从主要连通域的距离脊线和内接峰值估算典型笔画宽度。"""
    source = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        source,
        connectivity=8,
        ltype=cv2.CV_32S,
    )
    if count <= 1:
        return 0.0, 0
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64, copy=False)
    largest = int(areas.max())
    minimum_area = max(4, int(round(largest * 0.012)), int(round(source.size * 0.00008)))
    eligible = np.flatnonzero(areas >= minimum_area)
    if not eligible.size:
        return 0.0, 0
    eligible = eligible[np.argsort(areas[eligible])[::-1]]
    target_area = max(1, int(round(float(areas[eligible].sum()) * 0.82)))
    selected: list[int] = []
    accumulated = 0
    for offset in eligible:
        selected.append(int(offset) + 1)
        accumulated += int(areas[offset])
        if accumulated >= target_area:
            break

    selected_mask = np.isin(labels, selected)
    distance = cv2.distanceTransform(source, cv2.DIST_L2, 5)
    peak_radii: list[float] = []
    peak_weights: list[int] = []
    for label in selected:
        values = distance[labels == label]
        if values.size:
            peak_radii.append(float(values.max()))
            peak_weights.append(int(stats[label, cv2.CC_STAT_AREA]))

    weighted_peak = 0.0
    if peak_radii:
        peak_array = np.asarray(peak_radii, dtype=np.float64)
        weight_array = np.asarray(peak_weights, dtype=np.float64)
        order = np.argsort(peak_array)
        cumulative = np.cumsum(weight_array[order])
        midpoint = float(weight_array.sum()) * 0.5
        weighted_peak = float(peak_array[order[np.searchsorted(cumulative, midpoint)]])

    local_maximum = distance >= cv2.dilate(distance, np.ones((3, 3), np.uint8)) - 0.05
    samples = distance[selected_mask & local_maximum & (distance >= 0.9)] * 2.0
    if samples.size < 3:
        samples = distance[selected_mask] * 2.0
    if not samples.size:
        return 0.0, 0
    lower, upper = np.percentile(samples, (15.0, 85.0))
    trimmed = samples[(samples >= lower) & (samples <= upper)]
    values = trimmed if trimmed.size else samples
    ridge_scale = float(np.percentile(values, 68.0))
    # 密噪会制造大量低脊线；主要主体的内接半径为典型笔画提供稳健下限。
    return max(ridge_scale, weighted_peak), int(values.size)


def _estimate_multilevel_stroke(
    filtered: np.ndarray,
    otsu_threshold: int,
) -> tuple[float, float, int, list[float]]:
    estimates: list[float] = []
    sample_total = 0
    for offset in (-8, 0, 8):
        threshold = int(np.clip(otsu_threshold + offset, 1, 254))
        estimate, sample_count = _mask_stroke_scale(filtered <= threshold)
        if estimate > 0.0:
            estimates.append(estimate)
            sample_total += sample_count
    if not estimates:
        return 0.0, 0.0, 0, []
    stroke_scale = float(np.median(np.asarray(estimates, dtype=np.float64)))
    spread = float(np.ptp(np.asarray(estimates, dtype=np.float64))) if len(estimates) > 1 else 0.0
    stability = float(np.clip(1.0 - spread / max(stroke_scale, 1.0), 0.0, 1.0))
    return stroke_scale, stability, sample_total, estimates


def _component_scale_data(
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        source,
        connectivity=8,
        ltype=cv2.CV_32S,
    )
    distance = cv2.distanceTransform(source, cv2.DIST_L2, 5)
    widths = np.zeros(count, dtype=np.float32)
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        component = labels[y : y + height, x : x + width] == label
        values = distance[y : y + height, x : x + width][component]
        widths[label] = float(values.max() * 2.0) if values.size else 0.0
    return labels, stats, widths, distance


def _bbox_group_gap(label: int, group: set[int], stats: np.ndarray) -> float:
    """返回一个连通域与既有主体组的最小包围盒间距。"""
    if not group:
        return float("inf")
    x = float(stats[label, cv2.CC_STAT_LEFT])
    y = float(stats[label, cv2.CC_STAT_TOP])
    right = x + float(stats[label, cv2.CC_STAT_WIDTH]) - 1.0
    bottom = y + float(stats[label, cv2.CC_STAT_HEIGHT]) - 1.0
    best = float("inf")
    for other in group:
        other_x = float(stats[other, cv2.CC_STAT_LEFT])
        other_y = float(stats[other, cv2.CC_STAT_TOP])
        other_right = other_x + float(stats[other, cv2.CC_STAT_WIDTH]) - 1.0
        other_bottom = other_y + float(stats[other, cv2.CC_STAT_HEIGHT]) - 1.0
        dx = max(x - other_right - 1.0, other_x - right - 1.0, 0.0)
        dy = max(y - other_bottom - 1.0, other_y - bottom - 1.0, 0.0)
        best = min(best, float(np.hypot(dx, dy)))
    return best


def _select_primary_coarse_labels(
    coarse_labels: list[int],
    stats: np.ndarray,
    stroke_scale: float,
    image_shape: tuple[int, int],
) -> tuple[list[int], list[int], bool]:
    """先聚合粗部件组，再按主体组总面积和组间距离建立主体。"""
    remaining = set(coarse_labels)
    groups: list[set[int]] = []
    link_distance = stroke_scale * 1.35
    while remaining:
        start = remaining.pop()
        group = {start}
        stack = [start]
        while stack:
            current = stack.pop()
            neighbors = {
                label
                for label in remaining
                if _bbox_group_gap(label, {current}, stats) <= link_distance
            }
            if neighbors:
                remaining.difference_update(neighbors)
                group.update(neighbors)
                stack.extend(neighbors)
        groups.append(group)

    def group_area(group: set[int]) -> int:
        return sum(int(stats[label, cv2.CC_STAT_AREA]) for label in group)

    groups.sort(key=group_area, reverse=True)
    primary = set(groups[0])
    anchor_area = group_area(groups[0])

    def group_bounds(group: set[int]) -> tuple[int, int, int, int]:
        indexes = np.asarray(sorted(group), dtype=np.int32)
        left = int(stats[indexes, cv2.CC_STAT_LEFT].min())
        top = int(stats[indexes, cv2.CC_STAT_TOP].min())
        right = int(
            np.max(
                stats[indexes, cv2.CC_STAT_LEFT]
                + stats[indexes, cv2.CC_STAT_WIDTH]
            )
        )
        bottom = int(
            np.max(
                stats[indexes, cv2.CC_STAT_TOP]
                + stats[indexes, cv2.CC_STAT_HEIGHT]
            )
        )
        return left, top, right, bottom

    image_height, image_width = image_shape
    short_side = float(min(image_height, image_width))
    border_margin = max(2, int(round(short_side * 0.04)))
    inner_margin = max(3, int(round(short_side * 0.08)))
    anchor_left, anchor_top, anchor_right, anchor_bottom = group_bounds(groups[0])
    spans_horizontal_frame = (
        anchor_left <= border_margin
        and image_width - anchor_right <= border_margin
        and anchor_right - anchor_left >= image_width * 0.85
    )
    spans_vertical_frame = (
        anchor_top <= border_margin
        and image_height - anchor_bottom <= border_margin
        and anchor_bottom - anchor_top >= image_height * 0.85
    )
    anchor_ambiguous = False
    if spans_horizontal_frame or spans_vertical_frame:
        for group in groups[1:]:
            left, top, right, bottom = group_bounds(group)
            clearly_inside = (
                left >= inner_margin
                and top >= inner_margin
                and image_width - right >= inner_margin
                and image_height - bottom >= inner_margin
            )
            separated = min(
                _bbox_group_gap(label, primary, stats)
                for label in group
            ) > stroke_scale * 1.42
            if clearly_inside and separated:
                anchor_ambiguous = True
                break

    changed = True
    while changed:
        changed = False
        for group in groups[1:]:
            if group.issubset(primary):
                continue
            gap = min(_bbox_group_gap(label, primary, stats) for label in group)
            if gap <= stroke_scale * 1.42 or group_area(group) >= anchor_area * 0.22:
                primary.update(group)
                changed = True
    peripheral = [label for label in coarse_labels if label not in primary]
    return sorted(primary), peripheral, anchor_ambiguous


def _fallback_analysis(
    gray: np.ndarray,
    base_mask: np.ndarray,
    otsu_threshold: int,
    reason: str,
) -> StrokeScaleAnalysis:
    empty = np.zeros_like(base_mask, dtype=np.uint8)
    return StrokeScaleAnalysis(
        source_gray=gray,
        base_mask=base_mask,
        core_mask=base_mask.copy(),
        safety_mask=base_mask.copy(),
        body_mask=base_mask.copy(),
        conservative_body_mask=base_mask.copy(),
        support_mask=base_mask.copy(),
        independent_noise_mask=empty,
        confidence=0.0,
        applicable=False,
        reason=reason,
        stroke_scale=0.0,
        noise_scale=0.0,
        otsu_threshold=otsu_threshold,
        core_threshold=otsu_threshold,
        support_threshold=otsu_threshold,
        noise_component_count=0,
        metrics={},
    )


def _analyze_stroke_scale_uncached(
    gray: np.ndarray,
    min_confidence: float = 0.78,
    minimum_noise_components: int = 8,
) -> StrokeScaleAnalysis:
    """分析灰度图中的主体笔画尺度和密集细噪尺度。

    不使用字符或方向信息。只有细噪数量、尺度分离、灰度可分性和多阈值笔画
    稳定性共同达到门槛时，``applicable`` 才为 True。
    """
    source = _as_gray_u8(gray)
    otsu_value, _ = cv2.threshold(
        source,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    otsu_threshold = int(round(float(otsu_value)))
    base_mask = (source <= otsu_threshold).astype(np.uint8)
    foreground_total = int(base_mask.sum())
    if foreground_total < 8:
        return _fallback_analysis(source, base_mask, otsu_threshold, "有效前景不足")

    filtered = cv2.medianBlur(source, 3)
    stroke_scale, stroke_stability, stroke_samples, stroke_estimates = _estimate_multilevel_stroke(
        filtered,
        otsu_threshold,
    )
    short_side = float(min(source.shape))
    if stroke_scale < 2.0 or stroke_scale > short_side * 0.18:
        return _fallback_analysis(source, base_mask, otsu_threshold, "参考笔画尺度不稳定")

    labels, stats, component_widths, distance = _component_scale_data(base_mask)
    component_count = stats.shape[0] - 1
    stroke_area = stroke_scale * stroke_scale
    foreground_values = source[base_mask > 0]
    foreground_spread = float(np.std(foreground_values)) if foreground_values.size else 0.0
    core_offset = max(8, int(round(foreground_spread * 0.22)))
    core_threshold = int(np.clip(otsu_threshold - core_offset, 1, 253))
    support_threshold = int(np.clip(otsu_threshold + 18, core_threshold + 1, 254))
    support_mask = (source <= support_threshold).astype(np.uint8)
    support_coverage = float(support_mask.sum()) / source.size

    component_radii = component_widths * 0.5
    baseline_core_radius = stroke_scale * 0.40
    subscale_radii = component_radii[
        (component_radii > 0.0) & (component_radii < baseline_core_radius)
    ]
    noise_upper_radius = (
        float(np.percentile(subscale_radii, 95.0))
        if subscale_radii.size
        else 0.0
    )
    coarse_radius_threshold = max(baseline_core_radius, noise_upper_radius * 1.10)
    scale_coarse_labels = [
        label
        for label in range(1, component_count + 1)
        if float(component_radii[label]) >= coarse_radius_threshold
    ]
    if not scale_coarse_labels:
        return _fallback_analysis(source, base_mask, otsu_threshold, "无法定位粗笔画核心")

    coarse_labels, peripheral_coarse_labels, anchor_ambiguous = _select_primary_coarse_labels(
        scale_coarse_labels,
        stats,
        stroke_scale,
        source.shape,
    )
    coarse_mask = np.isin(labels, coarse_labels).astype(np.uint8)
    distance_to_coarse = cv2.distanceTransform(1 - coarse_mask, cv2.DIST_L2, 3)
    deep_stable = (source <= core_threshold) & (filtered <= otsu_threshold)
    coarse_ink_values = source[coarse_mask > 0]
    coarse_ink_median = (
        float(np.median(coarse_ink_values))
        if coarse_ink_values.size
        else float(otsu_threshold)
    )
    coarse_ink_iqr = (
        float(np.percentile(coarse_ink_values, 75.0) - np.percentile(coarse_ink_values, 25.0))
        if coarse_ink_values.size
        else 0.0
    )
    coarse_ink_mad = (
        float(np.median(np.abs(coarse_ink_values.astype(np.float32) - coarse_ink_median)))
        if coarse_ink_values.size
        else 0.0
    )
    ink_margin = max(
        12.0,
        min(24.0, 8.0 + coarse_ink_iqr * 0.45),
    )
    tone_mad_limit = max(16.0, coarse_ink_mad * 1.50)
    thin_candidates: list[tuple[float, int, int, bool]] = []
    coarse_label_set = set(coarse_labels)
    peripheral_coarse_set = set(peripheral_coarse_labels)
    image_height, image_width = source.shape
    short_side = float(min(source.shape))
    for label in range(1, component_count + 1):
        if label in coarse_label_set:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        radius = float(component_radii[label])
        span = max(component_width, component_height)
        minor_span = max(1, min(component_width, component_height))
        aspect = span / minor_span
        possible_dot = (
            radius >= stroke_scale * 0.12
            and area >= stroke_area * 0.08
            and span <= stroke_scale * 3.20
            and aspect <= 3.50
        )
        possible_stroke = (
            span >= stroke_scale * 1.20
            and area >= stroke_area * 0.15
            and minor_span <= stroke_scale * 1.70
        )
        ambiguous_part = possible_dot or possible_stroke or label in peripheral_coarse_set
        if not ambiguous_part:
            continue

        local_labels = labels[y : y + component_height, x : x + component_width]
        component = local_labels == label
        local_source = source[y : y + component_height, x : x + component_width]
        local_deep = deep_stable[y : y + component_height, x : x + component_width]
        local_distance = distance_to_coarse[y : y + component_height, x : x + component_width]
        component_values = local_source[component]
        deep_ratio = float(np.count_nonzero(local_deep[component])) / max(area, 1)
        component_ink_median = float(np.median(component_values))
        component_gray_mad = float(
            np.median(
                np.abs(
                    component_values.astype(np.float32) - component_ink_median
                )
            )
        )
        gap = max(0.0, float(local_distance[component].min()) - 1.0)
        fill_ratio = area / max(1.0, float(component_width * component_height))
        component_u8 = component.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            component_u8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contour = max(contours, key=cv2.contourArea) if contours else None
        contour_area = float(cv2.contourArea(contour)) if contour is not None else 0.0
        perimeter = float(cv2.arcLength(contour, True)) if contour is not None else 0.0
        hull_area = (
            float(cv2.contourArea(cv2.convexHull(contour)))
            if contour is not None
            else 0.0
        )
        solidity = float(np.clip(area / max(hull_area, 1.0), 0.0, 1.0))
        circularity = float(
            np.clip(4.0 * np.pi * contour_area / max(perimeter * perimeter, 1.0), 0.0, 1.0)
        )
        points_yx = np.column_stack(np.where(component))
        if points_yx.shape[0] >= 3:
            points_xy = points_yx[:, ::-1].astype(np.float64)
            centered_points = points_xy - points_xy.mean(axis=0, keepdims=True)
            covariance = np.cov(centered_points, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            eigenvalues = np.maximum(eigenvalues, 1e-6)
            pca_elongation = float(np.sqrt(eigenvalues[-1] / eigenvalues[0]))
            projections = centered_points @ eigenvectors
            minor_oriented_span = float(np.ptp(projections[:, 0]) + 1.0)
            major_oriented_span = float(np.ptp(projections[:, 1]) + 1.0)
        else:
            pca_elongation = 1.0
            minor_oriented_span = float(minor_span)
            major_oriented_span = float(span)
        compact_dot = (
            radius >= stroke_scale * 0.17
            and area >= stroke_area * 0.10
            and span <= stroke_scale * 1.75
            and fill_ratio >= 0.42
            and solidity >= 0.75
            and circularity >= 0.45
        )
        slender_long_stroke = (
            major_oriented_span >= stroke_scale * 1.60
            and area >= stroke_area * 0.16
            and minor_oriented_span <= stroke_scale * 0.90
            and pca_elongation >= 2.35
            and solidity >= 0.58
        )
        ink_consistent = component_ink_median <= coarse_ink_median + ink_margin
        tone_stable = component_gray_mad <= tone_mad_limit
        coherent_shallow = component_ink_median <= otsu_threshold - 8
        edge_distance = min(
            x,
            y,
            image_width - (x + component_width),
            image_height - (y + component_height),
        )
        peripheral_coarse = (
            label in peripheral_coarse_set
            and gap > stroke_scale * 2.20
            and edge_distance < short_side * 0.18
        )
        high_confidence_part = bool(
            (compact_dot or slender_long_stroke)
            and (ink_consistent or coherent_shallow)
            and tone_stable
            and gap <= stroke_scale * 3.00
            and not peripheral_coarse
        )
        proximity_score = float(np.clip(1.0 - gap / max(stroke_scale * 2.0, 1.0), 0.0, 1.0))
        radius_score = float(np.clip(radius / max(stroke_scale * 0.40, 1.0), 0.0, 1.0))
        ink_score = float(
            np.clip(
                (coarse_ink_median + ink_margin - component_ink_median)
                / max(ink_margin, 1.0),
                0.0,
                1.0,
            )
        )
        evidence_score = (
            (2.5 if slender_long_stroke else 0.6)
            + deep_ratio * 1.4
            + proximity_score * 1.2
            + radius_score * 0.8
            + ink_score * 1.2
            + (1.2 if coherent_shallow else 0.0)
        )
        thin_candidates.append((evidence_score, label, area, high_confidence_part))

    thin_candidates.sort(reverse=True)
    maximum_protected_count = 16
    coarse_area = int(coarse_mask.sum())
    protected_area_budget = max(
        int(round(coarse_area * 0.18)),
        int(round(stroke_area * 6.0)),
    )
    protected_thin_labels: list[int] = []
    protected_area = 0
    safety_overflow = False
    for _score, label, area, high_confidence_part in thin_candidates:
        if not high_confidence_part:
            continue
        if len(protected_thin_labels) >= maximum_protected_count:
            safety_overflow = True
            continue
        if protected_area + area > protected_area_budget:
            safety_overflow = True
            continue
        protected_thin_labels.append(label)
        protected_area += area

    conservative_thin_labels = list(protected_thin_labels)
    for _score, label, _area, _high_confidence_part in thin_candidates:
        if label in conservative_thin_labels:
            continue
        conservative_thin_labels.append(label)

    body_labels = coarse_labels + protected_thin_labels
    body_label_set = set(body_labels)
    body_mask = np.isin(labels, body_labels).astype(np.uint8)
    safety_mask = body_mask.copy()
    conservative_body_mask = np.isin(
        labels,
        coarse_labels + conservative_thin_labels,
    ).astype(np.uint8)
    body_total = int(body_mask.sum())
    if body_total < max(8, int(round(stroke_area * 0.5))):
        return _fallback_analysis(source, base_mask, otsu_threshold, "无法建立稳定文字主体")

    noise_labels = [
        label
        for label in range(1, component_count + 1)
        if label not in body_label_set
    ]
    noise_widths = [float(component_widths[label]) for label in noise_labels]
    noise_component_count = len(noise_labels)
    noise_scale = float(np.median(np.asarray(noise_widths))) if noise_widths else 0.0
    noise_mask = np.isin(labels, noise_labels).astype(np.uint8)
    fine_area = int(noise_mask.sum())

    scale_core = (distance >= stroke_scale * 0.32) & (coarse_mask > 0)
    core_halo_radius = max(1, int(round(stroke_scale * 0.18)))
    core_halo_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (core_halo_radius * 2 + 1, core_halo_radius * 2 + 1),
    )
    scale_core_halo = cv2.dilate(scale_core.astype(np.uint8), core_halo_kernel) > 0
    core_mask = (
        (scale_core | (deep_stable & scale_core_halo))
        & (coarse_mask > 0)
    ).astype(np.uint8)
    if protected_thin_labels:
        core_mask |= np.isin(labels, protected_thin_labels).astype(np.uint8)
    core_total = int(core_mask.sum())
    if core_total < max(4, int(round(body_total * 0.04))):
        return _fallback_analysis(source, base_mask, otsu_threshold, "稳定笔画核心不足")

    separability = _otsu_separability(source)
    scale_ratio = stroke_scale / max(noise_scale, 1.0) if noise_scale > 0.0 else 1.0
    adaptive_minimum_count = max(1, int(minimum_noise_components))
    count_score = float(
        np.clip(
            (noise_component_count - adaptive_minimum_count + 1) / max(12.0, adaptive_minimum_count * 1.5),
            0.0,
            1.0,
        )
    )
    scale_score = float(np.clip((scale_ratio - 1.45) / 1.80, 0.0, 1.0))
    noise_area_ratio = fine_area / max(foreground_total, 1)
    noise_area_score = float(np.clip(noise_area_ratio / 0.04, 0.0, 1.0))
    body_dominance = body_total / max(foreground_total, 1)
    body_score = float(np.clip((body_dominance - 0.55) / 0.40, 0.0, 1.0))
    sample_score = float(np.clip(stroke_samples / 24.0, 0.0, 1.0))
    confidence = float(
        np.clip(
            0.10
            + separability * 0.17
            + stroke_stability * 0.16
            + sample_score * 0.07
            + scale_score * 0.22
            + count_score * 0.16
            + noise_area_score * 0.07
            + body_score * 0.05,
            0.0,
            1.0,
        )
    )
    applicable = bool(
        confidence >= float(min_confidence)
        and noise_component_count >= adaptive_minimum_count
        and scale_ratio >= 1.65
        and stroke_stability >= 0.45
        and separability >= 0.25
        and support_coverage <= 0.65
        and not safety_overflow
        and not anchor_ambiguous
    )
    if applicable:
        reason = (
            f"密集细噪与笔画尺度分离明确：{noise_component_count}域，"
            f"尺度比{scale_ratio:.2f}"
        )
    elif safety_overflow:
        reason = "待保护结构超过安全预算，保持基准掩码"
    elif anchor_ambiguous:
        reason = "跨边框粗部件与内区字形并存，主体锚点归属不明确"
    else:
        reason = "细噪密度或尺度分离不足，保持基准掩码"

    return StrokeScaleAnalysis(
        source_gray=source,
        base_mask=base_mask,
        core_mask=core_mask,
        safety_mask=safety_mask,
        body_mask=body_mask,
        conservative_body_mask=conservative_body_mask,
        support_mask=support_mask,
        independent_noise_mask=noise_mask,
        confidence=confidence,
        applicable=applicable,
        reason=reason,
        stroke_scale=stroke_scale,
        noise_scale=noise_scale,
        otsu_threshold=otsu_threshold,
        core_threshold=core_threshold,
        support_threshold=support_threshold,
        noise_component_count=noise_component_count,
        metrics={
            "Otsu可分性": separability,
            "笔画尺度稳定性": stroke_stability,
            "笔画尺度样本数": float(stroke_samples),
            "笔画尺度最小值": min(stroke_estimates) if stroke_estimates else 0.0,
            "笔画尺度最大值": max(stroke_estimates) if stroke_estimates else 0.0,
            "噪声与笔画尺度比": scale_ratio,
            "细噪面积比例": noise_area_ratio,
            "主体前景比例": body_dominance,
            "支持区图像占比": support_coverage,
            "自适应最小噪声域数": float(adaptive_minimum_count),
            "基准前景像素": float(foreground_total),
            "主体前景像素": float(body_total),
            "稳定核心像素": float(core_total),
            "粗核心半径阈值": coarse_radius_threshold,
            "噪声半径上界": noise_upper_radius,
            "尺度粗部件数": float(len(scale_coarse_labels)),
            "主体粗部件数": float(len(coarse_labels)),
            "外围粗部件数": float(len(peripheral_coarse_labels)),
            "粗核心连通域数": float(len(coarse_labels)),
            "受保护细笔连通域数": float(len(protected_thin_labels)),
            "保形档歧义部件数": float(
                len(conservative_thin_labels) - len(protected_thin_labels)
            ),
            "细笔保护候选数": float(len(thin_candidates)),
            "细笔保护面积": float(protected_area),
            "细笔保护面积预算": float(protected_area_budget),
            "主体墨色中位数": coarse_ink_median,
            "主体墨色四分位距": coarse_ink_iqr,
            "主体墨色中位绝对偏差": coarse_ink_mad,
            "细笔墨色容差": ink_margin,
            "浅墨区域MAD上限": tone_mad_limit,
            "安全部件溢出": 1.0 if safety_overflow else 0.0,
            "主体锚点歧义": 1.0 if anchor_ambiguous else 0.0,
        },
    )


def analyze_stroke_scale(
    gray: np.ndarray,
    min_confidence: float = 0.78,
    minimum_noise_components: int = 8,
) -> StrokeScaleAnalysis:
    """分析笔画尺度，并在当前寻优任务内复用完全相同的输入。"""
    source = np.ascontiguousarray(_as_gray_u8(gray))
    cache = _SESSION_CACHE.get()
    if cache is None:
        return _analyze_stroke_scale_uncached(
            source,
            min_confidence=min_confidence,
            minimum_noise_components=minimum_noise_components,
        )

    digest = hashlib.blake2b(memoryview(source), digest_size=16).digest()
    key: _CacheKey = (
        (int(source.shape[0]), int(source.shape[1])),
        digest,
        float(min_confidence),
        int(minimum_noise_components),
    )
    cached = cache.get(key, source)
    if cached is not None:
        return cached
    analysis = _analyze_stroke_scale_uncached(
        source,
        min_confidence=min_confidence,
        minimum_noise_components=minimum_noise_components,
    )
    cache.put(key, analysis)
    return analysis


def _strength_value(
    strength: ReconstructionStrength | str,
) -> ReconstructionStrength:
    if isinstance(strength, ReconstructionStrength):
        return strength
    aliases = {
        "conservative": ReconstructionStrength.CONSERVATIVE,
        "balanced": ReconstructionStrength.BALANCED,
        "strong": ReconstructionStrength.STRONG,
    }
    if strength in aliases:
        return aliases[strength]
    try:
        return ReconstructionStrength(strength)
    except ValueError as exc:
        raise ValueError(f"未知重建强度：{strength}") from exc


def _finite_geodesic_growth(
    marker: np.ndarray,
    allowed: np.ndarray,
    iterations: int,
) -> np.ndarray:
    current = ((marker > 0) & (allowed > 0)).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for _ in range(max(1, int(iterations))):
        grown = cv2.dilate(current, kernel)
        grown = ((grown > 0) & (allowed > 0)).astype(np.uint8)
        if np.array_equal(grown, current):
            break
        current = grown
    return current


def _restore_meaningful_components(
    reconstructed: np.ndarray,
    body_mask: np.ndarray,
    stroke_scale: float,
    minimum_retention: float,
) -> np.ndarray:
    """保底恢复被整体削弱的合法分离部件，不恢复局部细长附着噪声。"""
    result = reconstructed.copy()
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        body_mask,
        connectivity=8,
        ltype=cv2.CV_32S,
    )
    meaningful_area = max(5, int(round(stroke_scale * stroke_scale * 0.45)))
    maximum_protected_area = max(meaningful_area, int(round(stroke_scale * stroke_scale * 10.0)))
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < meaningful_area or area > maximum_protected_area:
            continue
        component = labels == label
        retained = int(np.count_nonzero(result[component])) / max(area, 1)
        if retained < minimum_retention:
            result[component] = 1
    return result


def _minimum_component_coverage(reference: np.ndarray, candidate: np.ndarray) -> float:
    """逐域计算独立安全结构的最低覆盖率，避免小点画被全局面积掩盖。"""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (reference > 0).astype(np.uint8),
        connectivity=8,
        ltype=cv2.CV_32S,
    )
    minimum = 1.0
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        local_reference = labels[y : y + height, x : x + width] == label
        local_candidate = candidate[y : y + height, x : x + width] > 0
        area = max(1, int(stats[label, cv2.CC_STAT_AREA]))
        coverage = float(np.count_nonzero(local_reference & local_candidate)) / area
        minimum = min(minimum, coverage)
    return minimum


def _restore_connected_terminal_branches(
    reconstructed: np.ndarray,
    body_mask: np.ndarray,
    source_gray: np.ndarray,
    stroke_scale: float,
) -> np.ndarray:
    """仅补回与现有结果相接、被长距离截断且墨色一致的细长端枝。"""
    result = (reconstructed > 0).astype(np.uint8)
    body = (body_mask > 0).astype(np.uint8)
    removed = ((body > 0) & (result == 0)).astype(np.uint8)
    if not removed.any() or not result.any():
        return result

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        removed,
        connectivity=8,
        ltype=cv2.CV_32S,
    )
    if count <= 1:
        return result

    distance_to_result = cv2.distanceTransform(
        (result == 0).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    retained_values = source_gray[(body > 0) & (result > 0)].astype(np.float32)
    if not retained_values.size:
        return result
    retained_median = float(np.median(retained_values))
    retained_mad = float(np.median(np.abs(retained_values - retained_median)))
    tone_mad_limit = max(16.0, retained_mad * 1.50)
    tone_difference_limit = max(24.0, retained_mad * 2.0)
    contact_kernel = np.ones((3, 3), dtype=np.uint8)

    for label in range(1, count):
        component = labels == label
        if float(distance_to_result[component].max()) <= stroke_scale * 1.25:
            continue

        contact = (
            (cv2.dilate(component.astype(np.uint8), contact_kernel) > 0)
            & (result > 0)
        )
        if not contact.any():
            continue
        if cv2.connectedComponents(contact.astype(np.uint8), connectivity=8)[0] - 1 != 1:
            continue

        points_yx = np.column_stack(np.where(component))
        if points_yx.shape[0] < 3:
            continue
        points_xy = points_yx[:, ::-1].astype(np.float64)
        centered_points = points_xy - points_xy.mean(axis=0, keepdims=True)
        covariance = np.cov(centered_points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 1e-6)
        projections = centered_points @ eigenvectors
        minor_span = float(np.ptp(projections[:, 0]) + 1.0)
        major_span = float(np.ptp(projections[:, 1]) + 1.0)
        elongation = float(np.sqrt(eigenvalues[-1] / eigenvalues[0]))
        if (
            major_span < stroke_scale * 2.0
            or minor_span > stroke_scale * 0.95
            or elongation < 2.35
        ):
            continue

        component_values = source_gray[component].astype(np.float32)
        contact_values = source_gray[contact].astype(np.float32)
        component_median = float(np.median(component_values))
        component_mad = float(
            np.median(np.abs(component_values - component_median))
        )
        contact_median = float(np.median(contact_values))
        if (
            component_mad > tone_mad_limit
            or abs(component_median - contact_median) > tone_difference_limit
        ):
            continue

        result[component] = 1

    return result


def _protect_body_topology(
    reconstructed: np.ndarray,
    body_mask: np.ndarray,
    core_mask: np.ndarray,
    stroke_scale: float,
) -> np.ndarray:
    """保护连接多个稳定核心的窄桥，并保留原主体中的有意义孔洞。"""
    result = reconstructed.copy()
    body_count, body_labels, body_stats, _ = cv2.connectedComponentsWithStats(
        body_mask,
        connectivity=8,
        ltype=cv2.CV_32S,
    )
    output_count, output_labels = cv2.connectedComponents(result, connectivity=8)
    if output_count > 1:
        meaningful_area = max(6, int(round(stroke_scale * stroke_scale * 0.45)))
        for label in range(1, body_count):
            if int(body_stats[label, cv2.CC_STAT_AREA]) < meaningful_area:
                continue
            component = body_labels == label
            core_output_labels = np.unique(output_labels[component & (core_mask > 0)])
            core_output_labels = core_output_labels[core_output_labels > 0]
            if core_output_labels.size > 1:
                result[component] = 1

    background = (body_mask == 0).astype(np.uint8)
    hole_count, hole_labels, hole_stats, _ = cv2.connectedComponentsWithStats(
        background,
        connectivity=8,
        ltype=cv2.CV_32S,
    )
    height, width = body_mask.shape
    minimum_hole_area = max(4, int(round(stroke_scale * stroke_scale * 0.12)))
    for label in range(1, hole_count):
        x = int(hole_stats[label, cv2.CC_STAT_LEFT])
        y = int(hole_stats[label, cv2.CC_STAT_TOP])
        item_width = int(hole_stats[label, cv2.CC_STAT_WIDTH])
        item_height = int(hole_stats[label, cv2.CC_STAT_HEIGHT])
        area = int(hole_stats[label, cv2.CC_STAT_AREA])
        touches_edge = x == 0 or y == 0 or x + item_width >= width or y + item_height >= height
        if not touches_edge and area >= minimum_hole_area:
            result[hole_labels == label] = 0
    return result


def reconstruct_stroke_scale(
    analysis: StrokeScaleAnalysis,
    strength: ReconstructionStrength | str = ReconstructionStrength.BALANCED,
) -> StrokeScaleReconstruction:
    """按指定强度执行有限距离的笔画核心重建。"""
    strength_value = _strength_value(strength)
    profile = _STRENGTH_PROFILES[strength_value]
    base = analysis.base_mask
    empty = np.zeros_like(base, dtype=np.uint8)
    if not analysis.applicable:
        return StrokeScaleReconstruction(
            strength=strength_value,
            mask=base.copy(),
            removed_mask=empty,
            added_mask=empty.copy(),
            confidence=analysis.confidence,
            applied=False,
            reason=analysis.reason,
            core_retention=1.0,
            removed_ratio=0.0,
            added_ratio=0.0,
        )

    stroke_scale = analysis.stroke_scale
    body = analysis.body_mask
    distance = cv2.distanceTransform(body, cv2.DIST_L2, 5)
    center_radius = max(0.9, stroke_scale * profile.center_radius_ratio)
    marker = distance >= center_radius
    marker |= (analysis.core_mask > 0) & (distance >= center_radius * 0.55)

    # 每个达到笔画尺度的独立合法部件至少保留一个生长种子。
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        body,
        connectivity=8,
        ltype=cv2.CV_32S,
    )
    for label in range(1, count):
        component = labels == label
        values = distance[component]
        if not values.size or marker[component].any():
            continue
        maximum_radius = float(values.max())
        area = int(stats[label, cv2.CC_STAT_AREA])
        scale_eligible = maximum_radius * 2.0 >= stroke_scale * profile.component_seed_ratio
        area_eligible = area >= stroke_scale * stroke_scale * 0.45
        if scale_eligible or area_eligible:
            marker[component & (distance >= maximum_radius - 0.05)] = True

    # 去噪候选不凭宽松支持阈值新增墨点；支持掩码仅作为后续恢复型管线入口。
    allowed = body

    growth_iterations = max(1, int(round(stroke_scale * profile.growth_radius_ratio)))
    if strength_value is ReconstructionStrength.CONSERVATIVE:
        reconstructed = analysis.conservative_body_mask.copy()
    else:
        reconstructed = _finite_geodesic_growth(marker.astype(np.uint8), allowed, growth_iterations)
    reconstructed = _restore_meaningful_components(
        reconstructed,
        body,
        stroke_scale,
        profile.component_retention,
    )
    reconstructed = _protect_body_topology(
        reconstructed,
        body,
        analysis.core_mask,
        stroke_scale,
    )
    reconstructed = _restore_connected_terminal_branches(
        reconstructed,
        body,
        analysis.source_gray,
        stroke_scale,
    )

    core_total = int(analysis.core_mask.sum())
    core_retention = (
        float((reconstructed & analysis.core_mask).sum()) / core_total
        if core_total
        else 1.0
    )
    safety_component_retention = _minimum_component_coverage(
        analysis.safety_mask,
        reconstructed,
    )
    if (
        core_retention < profile.minimum_core_retention
        or safety_component_retention < profile.minimum_safety_component_retention
    ):
        return StrokeScaleReconstruction(
            strength=strength_value,
            mask=base.copy(),
            removed_mask=empty,
            added_mask=empty.copy(),
            confidence=analysis.confidence,
            applied=False,
            reason=(
                f"安全结构最低保留{safety_component_retention:.1%}、"
                f"稳定核心保留{core_retention:.1%}，已回退基准掩码"
            ),
            core_retention=core_retention,
            removed_ratio=0.0,
            added_ratio=0.0,
        )

    removed_mask = ((base > 0) & (reconstructed == 0)).astype(np.uint8)
    added_mask = ((reconstructed > 0) & (base == 0)).astype(np.uint8)
    base_total = max(1, int(base.sum()))
    removed_ratio = float(removed_mask.sum()) / base_total
    added_ratio = float(added_mask.sum()) / base_total
    changed = bool(removed_mask.any() or added_mask.any())
    confidence_adjustment = {
        ReconstructionStrength.CONSERVATIVE: 0.0,
        ReconstructionStrength.BALANCED: 0.015,
        ReconstructionStrength.STRONG: 0.035,
    }[strength_value]
    result_confidence = float(np.clip(analysis.confidence - confidence_adjustment, 0.0, 1.0))
    reason = (
        f"{strength_value.value}重建：移除{removed_ratio:.1%}，补回{added_ratio:.1%}"
        if changed
        else f"{strength_value.value}重建与基准掩码一致"
    )
    return StrokeScaleReconstruction(
        strength=strength_value,
        mask=reconstructed,
        removed_mask=removed_mask,
        added_mask=added_mask,
        confidence=result_confidence,
        applied=changed,
        reason=reason,
        core_retention=core_retention,
        removed_ratio=removed_ratio,
        added_ratio=added_ratio,
    )


def reconstruct_three_strengths(
    gray: np.ndarray,
    min_confidence: float = 0.78,
    minimum_noise_components: int = 8,
) -> tuple[StrokeScaleAnalysis, dict[str, StrokeScaleReconstruction]]:
    """一次分析并返回保守、均衡、强力三档重建结果。"""
    analysis = analyze_stroke_scale(
        gray,
        min_confidence=min_confidence,
        minimum_noise_components=minimum_noise_components,
    )
    results = {
        strength.value: reconstruct_stroke_scale(analysis, strength)
        for strength in ReconstructionStrength
    }
    return analysis, results
