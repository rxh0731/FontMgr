# scoring.py — 无参考自动评分体系（满分 100）

import time
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

from core.foreground_analysis import analyze_external_pollution
from core.stroke_scale_analysis import analyze_stroke_scale
from core.component_policy import STRUCTURE_PROTECTION_COMPONENT_LIMIT


CancelCheck = Optional[Callable[[], None]]


def _check_cancelled(cancel_check: CancelCheck) -> None:
    """调用取消检查；具体异常类型由上层回调决定，避免评分层反向依赖优化器。"""
    if cancel_check is not None:
        cancel_check()


@dataclass(frozen=True)
class StructureMetrics:
    """候选或参考掩码的统一结构指标。"""

    components: int
    holes: int
    stroke_width: float
    skeleton: np.ndarray
    skeleton_total: int
    endpoints: np.ndarray
    endpoint_count: int
    branch_count: int
    hole_centroids: np.ndarray
    hole_areas: np.ndarray
    component_areas: np.ndarray

    @property
    def comparison_min_component_area(self) -> Optional[int]:
        """返回候选拓扑分析应采用的参考主体面积下限。"""
        if not self.component_areas.size:
            return None
        return max(1, int(self.component_areas.min()) // 2)


@dataclass(frozen=True)
class StructureComparison:
    """候选结构相对参考结构的保留情况。"""

    skeleton_coverage: float
    endpoint_retention: float
    endpoint_growth: int
    branch_retention: float
    hole_retention: float
    extra_holes: int
    stroke_ratio: float
    component_delta: int


@dataclass(frozen=True)
class ScoreBreakdown:
    """综合得分及用于多目标排序的独立质量维度。"""

    score: float
    stroke_retention: float
    core_retention: float
    background_cleanliness: float
    topology_stability: float
    ratio_plausibility: float
    noise_penalty: float
    background_penalty: float
    structure_penalty: float
    noise_components: int
    structure: StructureMetrics
    comparison: StructureComparison

    @property
    def objectives(self) -> tuple[float, float, float, float]:
        """返回需同时最大化的排序目标。"""
        return (
            round(self.topology_stability, 4),
            round(min(self.stroke_retention, self.core_retention), 4),
            round(self.background_cleanliness, 4),
            round(self.ratio_plausibility, 4),
        )

    def as_dict(self) -> dict[str, float | int]:
        """返回可记录、展示和持久化的精简评分明细。"""
        return {
            "综合得分": self.score,
            "笔画保留": round(self.stroke_retention, 4),
            "深墨核心保留": round(self.core_retention, 4),
            "背景清洁": round(self.background_cleanliness, 4),
            "拓扑稳定": round(self.topology_stability, 4),
            "前景占比合理": round(self.ratio_plausibility, 4),
            "噪点扣分": round(self.noise_penalty, 2),
            "背景残留扣分": round(self.background_penalty, 2),
            "结构扣分": round(self.structure_penalty, 2),
            "连通域": self.structure.components,
            "有意义孔洞": self.structure.holes,
            "端点": self.structure.endpoint_count,
            "分支点": self.structure.branch_count,
        }


@dataclass(frozen=True)
class ScoreContext:
    """同一字图全部候选共享的只读评分参考数据。"""

    ref_mask: np.ndarray
    ref_total: int
    core_mask: np.ndarray
    core_total: int
    ref_centroids: np.ndarray
    text_median: float
    dist_threshold: float
    small_threshold: int
    ref_components: int
    ref_holes: int
    ref_stroke: float
    total_pixels: int
    heavy_noise: bool
    reference_structure: StructureMetrics
    structure_confidence: float
    ref_label_map: np.ndarray
    ref_component_ids: np.ndarray
    ref_component_areas: np.ndarray


def _thin_mask(mask: np.ndarray, cancel_check: CancelCheck = None) -> np.ndarray:
    """使用 Zhang-Suen 算法把二值主体细化为单像素骨架。"""
    _check_cancelled(cancel_check)
    source = (mask > 0).astype(bool)
    if not source.any():
        return np.zeros_like(source, dtype=np.uint8)

    ys, xs = np.where(source)
    top = max(0, int(ys.min()) - 1)
    bottom = min(source.shape[0], int(ys.max()) + 2)
    left = max(0, int(xs.min()) - 1)
    right = min(source.shape[1], int(xs.max()) + 2)
    image = source[top:bottom, left:right].astype(np.uint8, copy=True)
    padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2), dtype=np.uint8)

    def neighbors(current: np.ndarray) -> tuple[np.ndarray, ...]:
        padded[1:-1, 1:-1] = current
        return (
            padded[:-2, 1:-1],
            padded[:-2, 2:],
            padded[1:-1, 2:],
            padded[2:, 2:],
            padded[2:, 1:-1],
            padded[2:, :-2],
            padded[1:-1, :-2],
            padded[:-2, :-2],
        )

    for _ in range(max(image.shape) * 2):
        _check_cancelled(cancel_check)
        changed = False
        p2, p3, p4, p5, p6, p7, p8, p9 = neighbors(image)
        neighbor_count = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        transitions = (
            (1 - p2) * p3 + (1 - p3) * p4
            + (1 - p4) * p5 + (1 - p5) * p6
            + (1 - p6) * p7 + (1 - p7) * p8
            + (1 - p8) * p9 + (1 - p9) * p2
        )
        remove = (
            (image > 0)
            & (neighbor_count >= 2)
            & (neighbor_count <= 6)
            & (transitions == 1)
            & ((p2 * p4 * p6) == 0)
            & ((p4 * p6 * p8) == 0)
        )
        if remove.any():
            image[remove] = False
            changed = True

        _check_cancelled(cancel_check)
        p2, p3, p4, p5, p6, p7, p8, p9 = neighbors(image)
        neighbor_count = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        transitions = (
            (1 - p2) * p3 + (1 - p3) * p4
            + (1 - p4) * p5 + (1 - p5) * p6
            + (1 - p6) * p7 + (1 - p7) * p8
            + (1 - p8) * p9 + (1 - p9) * p2
        )
        remove = (
            (image > 0)
            & (neighbor_count >= 2)
            & (neighbor_count <= 6)
            & (transitions == 1)
            & ((p2 * p4 * p8) == 0)
            & ((p2 * p6 * p8) == 0)
        )
        if remove.any():
            image[remove] = False
            changed = True
        if not changed:
            break

    result = np.zeros_like(source, dtype=np.uint8)
    result[top:bottom, left:right] = image
    _check_cancelled(cancel_check)
    return result


def _meaningful_components(
    mask: np.ndarray,
    minimum_component_area: Optional[int] = None,
    cancel_check: CancelCheck = None,
) -> tuple[np.ndarray, int, np.ndarray]:
    """仅保留足以参与拓扑判断的连通域。"""
    _check_cancelled(cancel_check)
    source = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(source, connectivity=8, ltype=cv2.CV_32S)
    _check_cancelled(cancel_check)
    if count <= 1:
        return source, int(source.any()), np.empty(0, dtype=np.int32)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(areas.max()) if areas.size else 0
    threshold = max(3, int(source.size * 0.00003), int(largest * 0.003))
    if minimum_component_area is not None:
        threshold = max(threshold, int(minimum_component_area))
    valid = areas >= threshold
    keep_lookup = np.zeros(count, dtype=np.uint8)
    keep_lookup[1:] = valid.astype(np.uint8)
    keep = keep_lookup[labels]
    component_areas = areas[valid].astype(np.int32, copy=False)
    return keep, int(valid.sum()), component_areas


def compute_structure_metrics(
    mask: np.ndarray,
    minimum_component_area: Optional[int] = None,
    cancel_check: CancelCheck = None,
) -> StructureMetrics:
    """统一计算连通域、骨架、端点、孔洞和笔画宽度。"""
    source, components, component_areas = _meaningful_components(
        mask,
        minimum_component_area,
        cancel_check,
    )
    skeleton = _thin_mask(source, cancel_check)
    skeleton_total = int(skeleton.sum())

    _check_cancelled(cancel_check)
    if skeleton_total:
        neighbor_kernel = np.ones((3, 3), dtype=np.uint8)
        neighbor_count = cv2.filter2D(skeleton, cv2.CV_16S, neighbor_kernel) - skeleton
        endpoint_mask = ((skeleton > 0) & (neighbor_count == 1)).astype(np.uint8)
        branch_pixels = ((skeleton > 0) & (neighbor_count >= 3)).astype(np.uint8)
        branch_count, _ = cv2.connectedComponents(branch_pixels, connectivity=8)
        endpoints_yx = np.column_stack(np.where(endpoint_mask > 0)).astype(np.float64)
        endpoints = endpoints_yx[:, ::-1] if endpoints_yx.size else np.empty((0, 2), dtype=np.float64)
        distance = cv2.distanceTransform(source, cv2.DIST_L2, 5)
        _check_cancelled(cancel_check)
        stroke_values = distance[skeleton > 0]
        stroke_width = float(np.median(stroke_values) * 2.0) if stroke_values.size else 0.0
    else:
        endpoints = np.empty((0, 2), dtype=np.float64)
        branch_count = 1
        stroke_width = 0.0

    background = (source == 0).astype(np.uint8)
    _check_cancelled(cancel_check)
    hole_count, _, hole_stats, hole_centroids = cv2.connectedComponentsWithStats(
        background, connectivity=8, ltype=cv2.CV_32S
    )
    _check_cancelled(cancel_check)
    height, width = source.shape
    minimum_hole_area = max(4, int(round(stroke_width * stroke_width * 0.12)))
    meaningful_centroids: list[np.ndarray] = []
    meaningful_areas: list[int] = []
    for label in range(1, hole_count):
        if label % 32 == 1:
            _check_cancelled(cancel_check)
        x = int(hole_stats[label, cv2.CC_STAT_LEFT])
        y = int(hole_stats[label, cv2.CC_STAT_TOP])
        item_width = int(hole_stats[label, cv2.CC_STAT_WIDTH])
        item_height = int(hole_stats[label, cv2.CC_STAT_HEIGHT])
        area = int(hole_stats[label, cv2.CC_STAT_AREA])
        touches_edge = x == 0 or y == 0 or x + item_width >= width or y + item_height >= height
        if touches_edge or area < minimum_hole_area:
            continue
        meaningful_centroids.append(np.asarray(hole_centroids[label], dtype=np.float64))
        meaningful_areas.append(area)

    centroid_array = (
        np.vstack(meaningful_centroids)
        if meaningful_centroids
        else np.empty((0, 2), dtype=np.float64)
    )
    area_array = np.asarray(meaningful_areas, dtype=np.int32)
    _check_cancelled(cancel_check)
    return StructureMetrics(
        components=components,
        holes=len(meaningful_areas),
        stroke_width=stroke_width,
        skeleton=skeleton,
        skeleton_total=skeleton_total,
        endpoints=endpoints,
        endpoint_count=int(endpoints.shape[0]),
        branch_count=max(0, int(branch_count) - 1),
        hole_centroids=centroid_array,
        hole_areas=area_array,
        component_areas=component_areas,
    )


def _otsu_separability(gray_arr: np.ndarray) -> float:
    """估算灰度直方图的双类可分性，供结构硬保护判断置信度。"""
    arr_u8 = np.clip(gray_arr, 0, 255).astype(np.uint8)
    histogram = np.bincount(arr_u8.ravel(), minlength=256).astype(np.float64)
    total = float(histogram.sum())
    if total <= 0:
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
    return float(np.clip(between.max() / total_variance, 0.0, 1.0))


def _reference_component_data(
    ref_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """缓存参考主体的有效连通域，供逐域覆盖检查复用。"""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (ref_mask > 0).astype(np.uint8),
        connectivity=8,
        ltype=cv2.CV_32S,
    )
    if count <= 1:
        return labels, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int32, copy=False)
    largest = int(areas.max()) if areas.size else 0
    minimum_area = max(3, int(round(largest * 0.003)))
    valid_offsets = np.flatnonzero(areas >= minimum_area)
    component_ids = (valid_offsets + 1).astype(np.int32, copy=False)
    return labels, component_ids, areas[valid_offsets]


def minimum_reference_component_coverage(
    mask: np.ndarray,
    context: ScoreContext,
) -> float:
    """返回候选对每个有效参考部件的最低覆盖率。"""
    if not context.ref_component_ids.size:
        return 1.0
    candidate = np.asarray(mask) > 0
    covered = np.bincount(
        context.ref_label_map[candidate].ravel(),
        minlength=int(context.ref_label_map.max()) + 1,
    )
    ratios = covered[context.ref_component_ids] / np.maximum(
        context.ref_component_areas,
        1,
    )
    return float(np.min(ratios)) if ratios.size else 1.0


def build_score_context(
    gray_arr: np.ndarray,
    total_pixels: Optional[int] = None,
    cancel_check: CancelCheck = None,
) -> ScoreContext:
    """预计算候选评分时保持不变的参考数据。"""
    _check_cancelled(cancel_check)
    total = total_pixels or gray_arr.size
    ref_mask = _build_reference_mask(gray_arr, cancel_check)
    _check_cancelled(cancel_check)
    core_mask = _build_stable_core_mask(gray_arr, ref_mask)
    _check_cancelled(cancel_check)
    ref_total = int(ref_mask.sum())
    core_total = int(core_mask.sum())
    ref_label_map, ref_component_ids, ref_component_areas = _reference_component_data(ref_mask)
    _check_cancelled(cancel_check)
    raw_noise_ratio, raw_noise_count = _estimate_raw_noise(gray_arr)
    _check_cancelled(cancel_check)
    heavy_noise = raw_noise_ratio >= 0.04 or raw_noise_count >= 40
    if ref_total:
        _, _, _, centroids = cv2.connectedComponentsWithStats(ref_mask, connectivity=8, ltype=cv2.CV_32S)
        ref_centroids = np.asarray(centroids[1:], dtype=np.float64)
        text_median = float(np.median(gray_arr[ref_mask > 0]))
        ys, xs = np.where(ref_mask > 0)
        diagonal = float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min()))
        reference_structure = compute_structure_metrics(
            ref_mask,
            cancel_check=cancel_check,
        )
        ref_components = reference_structure.components
        ref_holes = reference_structure.holes
        ref_stroke = reference_structure.stroke_width
    else:
        ref_centroids = np.empty((0, 2), dtype=np.float64)
        text_median = 60.0
        diagonal = 100.0
        reference_structure = compute_structure_metrics(
            ref_mask,
            cancel_check=cancel_check,
        )
        ref_components, ref_holes, ref_stroke = 0, 0, 0.0
    return ScoreContext(
        ref_mask=ref_mask,
        ref_total=ref_total,
        core_mask=core_mask,
        core_total=core_total,
        ref_centroids=ref_centroids,
        text_median=text_median,
        dist_threshold=diagonal * 0.08,
        small_threshold=max(20, int(np.sqrt(total) * 0.01)),
        ref_components=ref_components,
        ref_holes=ref_holes,
        ref_stroke=ref_stroke,
        total_pixels=total,
        heavy_noise=heavy_noise,
        reference_structure=reference_structure,
        structure_confidence=_otsu_separability(gray_arr),
        ref_label_map=ref_label_map,
        ref_component_ids=ref_component_ids,
        ref_component_areas=ref_component_areas,
    )


def auto_score(
    mask: np.ndarray,
    gray_arr: np.ndarray,
    total_pixels: Optional[int] = None,
    context: Optional[ScoreContext] = None,
    cancel_check: CancelCheck = None,
) -> float:
    """对去杂结果进行百分制评分。"""
    score, _ = auto_score_with_timing(
        mask,
        gray_arr,
        total_pixels,
        context,
        cancel_check,
    )
    return score


def auto_score_with_timing(
    mask: np.ndarray,
    gray_arr: np.ndarray,
    total_pixels: Optional[int] = None,
    context: Optional[ScoreContext] = None,
    cancel_check: CancelCheck = None,
) -> tuple[float, str]:
    """评分并返回可直接写入诊断日志的子阶段耗时。"""
    breakdown, timing = evaluate_candidate(
        mask,
        gray_arr,
        total_pixels,
        context,
        cancel_check,
    )
    return breakdown.score, timing


def evaluate_candidate(
    mask: np.ndarray,
    gray_arr: np.ndarray,
    total_pixels: Optional[int] = None,
    context: Optional[ScoreContext] = None,
    cancel_check: CancelCheck = None,
) -> tuple[ScoreBreakdown, str]:
    """返回兼顾显示分数与多目标排序的候选评分明细。"""
    total_started = time.perf_counter()
    _check_cancelled(cancel_check)
    if mask is None or not mask.any():
        empty_structure = compute_structure_metrics(
            np.zeros_like(gray_arr, dtype=np.uint8),
            cancel_check=cancel_check,
        )
        empty_comparison = StructureComparison(0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0)
        return ScoreBreakdown(
            score=0.0,
            stroke_retention=0.0,
            core_retention=0.0,
            background_cleanliness=0.0,
            topology_stability=0.0,
            ratio_plausibility=0.0,
            noise_penalty=40.0,
            background_penalty=0.0,
            structure_penalty=12.0,
            noise_components=0,
            structure=empty_structure,
            comparison=empty_comparison,
        ), "评分总计=0.0000秒｜空掩码"

    prepare_started = time.perf_counter()
    mask_bin = (mask > 0).astype(np.uint8)
    text_pixels = int(mask_bin.sum())
    _check_cancelled(cancel_check)
    prepare_elapsed = time.perf_counter() - prepare_started

    reference_started = time.perf_counter()
    score_context = context or build_score_context(
        gray_arr,
        total_pixels,
        cancel_check,
    )
    _check_cancelled(cancel_check)
    ref_mask = score_context.ref_mask
    reference_elapsed = time.perf_counter() - reference_started
    if not ref_mask.any():
        coverage = 1.0
        core_coverage = 1.0
        stroke_score = 40.0
    else:
        overlap = int((mask_bin & ref_mask).sum())
        coverage = overlap / max(1, score_context.ref_total)
        core_overlap = int((mask_bin & score_context.core_mask).sum())
        core_coverage = core_overlap / max(1, score_context.core_total)
        component_coverage = minimum_reference_component_coverage(mask_bin, score_context)
        core_coverage = min(
            core_coverage,
            float(np.clip(component_coverage / 0.90, 0.0, 1.0)),
        )
        if score_context.heavy_noise and score_context.core_total:
            # 重噪图以多阈值共同存在的深墨核心为主，避免把浅色背景颗粒当成应保留笔画。
            combined_coverage = core_coverage * 0.72 + min(1.0, coverage / 0.90) * 0.28
            stroke_score = float(np.clip((combined_coverage - 0.68) / 0.30 * 40.0, 0.0, 40.0))
        elif coverage >= 0.98:
            stroke_score = 40.0
        elif coverage < 0.70:
            stroke_score = 0.0
        else:
            stroke_score = (coverage - 0.70) / (0.98 - 0.70) * 40.0

    noise_started = time.perf_counter()
    penalty, noise_components = _compute_noise_penalty_details(mask_bin, gray_arr, score_context)
    _check_cancelled(cancel_check)
    noise_elapsed = time.perf_counter() - noise_started
    noise_score = max(0.0, 40.0 - penalty)

    ratio_started = time.perf_counter()
    ratio_score = _compute_ratio_score(text_pixels, score_context.total_pixels, score_context.heavy_noise)
    background_penalty = _compute_background_residue_penalty(mask_bin, score_context) if score_context.heavy_noise else 0.0
    ratio_elapsed = time.perf_counter() - ratio_started

    structure_started = time.perf_counter()
    candidate_structure = compute_structure_metrics(
        mask_bin,
        score_context.reference_structure.comparison_min_component_area,
        cancel_check,
    )
    _check_cancelled(cancel_check)
    comparison = compare_structure(mask_bin, score_context, candidate_structure)
    structure_penalty = _compute_structure_penalty(mask_bin, score_context, candidate_structure, comparison)
    _check_cancelled(cancel_check)
    structure_elapsed = time.perf_counter() - structure_started

    total_score = stroke_score + noise_score + ratio_score - structure_penalty - background_penalty
    score = round(min(100.0, max(0.0, total_score)), 1)
    background_cleanliness = float(np.clip(1.0 - (penalty + background_penalty) / 52.0, 0.0, 1.0))
    topology_stability = float(np.clip(1.0 - structure_penalty / 12.0, 0.0, 1.0))
    breakdown = ScoreBreakdown(
        score=score,
        stroke_retention=float(np.clip(coverage, 0.0, 1.0)),
        core_retention=float(np.clip(core_coverage, 0.0, 1.0)),
        background_cleanliness=background_cleanliness,
        topology_stability=topology_stability,
        ratio_plausibility=float(np.clip(ratio_score / 20.0, 0.0, 1.0)),
        noise_penalty=float(penalty),
        background_penalty=float(background_penalty),
        structure_penalty=float(structure_penalty),
        noise_components=int(noise_components),
        structure=candidate_structure,
        comparison=comparison,
    )
    total_elapsed = time.perf_counter() - total_started
    timing = (
        f"评分总计={total_elapsed:.4f}秒｜准备={prepare_elapsed:.4f}秒｜参考掩码={reference_elapsed:.4f}秒｜"
        f"噪点评分={noise_elapsed:.4f}秒｜结构评分={structure_elapsed:.4f}秒｜占比评分={ratio_elapsed:.4f}秒｜"
        f"前景像素={text_pixels}｜噪声连通域={noise_components}｜噪点扣分={penalty:.1f}｜"
        f"背景残留扣分={background_penalty:.1f}｜结构扣分={structure_penalty:.1f}｜重噪={'是' if score_context.heavy_noise else '否'}"
    )
    return breakdown, timing


def _keep_main_components(
    mask: np.ndarray,
    limit: int = STRUCTURE_PROTECTION_COMPONENT_LIMIT,
    min_ratio: float = 0.04,
) -> np.ndarray:
    """保留主要连通域，排除分散背景颗粒。"""
    source = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(source, connectivity=8, ltype=cv2.CV_32S)
    if num_labels <= 1:
        return source
    areas = [(i, int(stats[i, cv2.CC_STAT_AREA])) for i in range(1, num_labels)]
    areas.sort(key=lambda item: item[1], reverse=True)
    max_area = areas[0][1] if areas else 0
    keep = np.zeros_like(source)
    for idx, area in areas[:limit]:
        if area >= max_area * min_ratio:
            keep[labels == idx] = 1
    return keep


def _build_reference_mask(
    gray_arr: np.ndarray,
    cancel_check: CancelCheck = None,
) -> np.ndarray:
    """构建经过空间污染和笔画尺度清理的主体参考掩码。"""
    _check_cancelled(cancel_check)
    arr_u8 = np.clip(gray_arr, 0, 255).astype(np.uint8)
    filtered = cv2.medianBlur(arr_u8, 3)
    threshold, _ = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    raw_mask = (filtered <= threshold).astype(np.uint8)
    _check_cancelled(cancel_check)
    external = analyze_external_pollution(raw_mask, min_confidence=0.92)
    reference = external.cleaned_mask

    _check_cancelled(cancel_check)
    dense_analysis = analyze_stroke_scale(
        arr_u8,
        min_confidence=0.84,
        minimum_noise_components=10,
    )
    _check_cancelled(cancel_check)
    if dense_analysis.applicable:
        # 安全主体由原始灰度和连通域证据独立建立，不复用候选重建结果。
        return dense_analysis.safety_mask.copy()
    return _keep_main_components(reference)


def _build_stable_core_mask(gray_arr: np.ndarray, ref_mask: np.ndarray) -> np.ndarray:
    """构建多个保守阈值下都存在的稳定深墨核心。"""
    arr_u8 = np.clip(gray_arr, 0, 255).astype(np.uint8)
    filtered = cv2.medianBlur(arr_u8, 3)
    otsu, _ = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    q18 = float(np.percentile(filtered, 18.0))
    threshold = int(np.clip(min(float(otsu) - 12.0, q18), 10, 235))
    core = ((filtered <= threshold) & (ref_mask > 0)).astype(np.uint8)
    if int(core.sum()) < max(10, int(ref_mask.sum() * 0.25)):
        core = ((filtered <= max(10, int(otsu) - 6)) & (ref_mask > 0)).astype(np.uint8)
    if int(core.sum()) < max(10, int(ref_mask.sum() * 0.25)):
        core = ref_mask.copy()
    return core


def _estimate_raw_noise(gray_arr: np.ndarray) -> tuple[float, int]:
    """估算原图 Otsu 掩码中小连通域的面积比例和数量。"""
    arr_u8 = np.clip(gray_arr, 0, 255).astype(np.uint8)
    threshold, _ = cv2.threshold(arr_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    raw = (arr_u8 <= threshold).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8, ltype=cv2.CV_32S)
    if count <= 1:
        return 0.0, 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    small_limit = max(12, int(np.sqrt(raw.size) * 0.045))
    small = areas[areas < small_limit]
    return float(small.sum()) / max(1, int(raw.sum())), int(small.size)


def _compute_noise_penalty(
    mask_bin: np.ndarray,
    ref_mask: np.ndarray,
    gray_arr: np.ndarray,
    total: int,
) -> float:
    """计算噪点惩罚分。"""
    context = build_score_context(gray_arr, total)
    if not np.array_equal(context.ref_mask, ref_mask):
        context = _build_context_from_reference(ref_mask, gray_arr, total)
    penalty, _ = _compute_noise_penalty_details(mask_bin, gray_arr, context)
    return penalty


def _build_context_from_reference(ref_mask: np.ndarray, gray_arr: np.ndarray, total: int) -> ScoreContext:
    """为兼容内部旧调用，从指定参考掩码建立评分上下文。"""
    ref_total = int(ref_mask.sum())
    ref_label_map, ref_component_ids, ref_component_areas = _reference_component_data(ref_mask)
    if ref_total:
        _, _, _, centroids = cv2.connectedComponentsWithStats(ref_mask, connectivity=8, ltype=cv2.CV_32S)
        ref_centroids = np.asarray(centroids[1:], dtype=np.float64)
        text_median = float(np.median(gray_arr[ref_mask > 0]))
        ys, xs = np.where(ref_mask > 0)
        diagonal = float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min()))
        reference_structure = compute_structure_metrics(ref_mask)
        ref_components = reference_structure.components
        ref_holes = reference_structure.holes
        ref_stroke = reference_structure.stroke_width
    else:
        ref_centroids = np.empty((0, 2), dtype=np.float64)
        text_median = 60.0
        diagonal = 100.0
        reference_structure = compute_structure_metrics(ref_mask)
        ref_components, ref_holes, ref_stroke = 0, 0, 0.0
    return ScoreContext(
        ref_mask=ref_mask,
        ref_total=ref_total,
        core_mask=ref_mask.copy(),
        core_total=ref_total,
        ref_centroids=ref_centroids,
        text_median=text_median,
        dist_threshold=diagonal * 0.08,
        small_threshold=max(20, int(np.sqrt(total) * 0.01)),
        ref_components=ref_components,
        ref_holes=ref_holes,
        ref_stroke=ref_stroke,
        total_pixels=total,
        heavy_noise=False,
        reference_structure=reference_structure,
        structure_confidence=_otsu_separability(gray_arr),
        ref_label_map=ref_label_map,
        ref_component_ids=ref_component_ids,
        ref_component_areas=ref_component_areas,
    )


def _compute_noise_penalty_details(
    mask_bin: np.ndarray,
    gray_arr: np.ndarray,
    context: ScoreContext,
) -> tuple[float, int]:
    """计算噪点惩罚分及参与分析的噪声连通域数量。"""
    if not context.ref_mask.any():
        return 0.0, 0

    non_ref = mask_bin.copy()
    non_ref[context.ref_mask > 0] = 0
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(non_ref, connectivity=8, ltype=cv2.CV_32S)
    noise_count = num_labels - 1
    if num_labels <= 1 or context.ref_centroids.size == 0:
        return 0.0, max(0, noise_count)

    deltas = centroids[1:, None, :] - context.ref_centroids[None, :, :]
    min_distances = np.sqrt(np.sum(deltas * deltas, axis=2)).min(axis=1)

    penalty = 0.0
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if min_distances[i - 1] <= context.dist_threshold:
            left = int(stats[i, cv2.CC_STAT_LEFT])
            top = int(stats[i, cv2.CC_STAT_TOP])
            width = int(stats[i, cv2.CC_STAT_WIDTH])
            height = int(stats[i, cv2.CC_STAT_HEIGHT])
            local_labels = labels[top:top + height, left:left + width]
            local_gray = gray_arr[top:top + height, left:left + width]
            blob_values = local_gray[local_labels == i]
            if blob_values.size and np.median(blob_values) <= context.text_median + 60:
                continue
        if area < 20:
            penalty += 1.0
        elif area < context.small_threshold:
            penalty += 3.0
        else:
            penalty += min(20.0, 8.0 * area / context.small_threshold)
        if penalty >= 40.0:
            return 40.0, noise_count

    return min(40.0, penalty), noise_count


def _structure_features(mask: np.ndarray) -> tuple[int, int, float]:
    """兼容旧内部调用，实际指标由统一结构分析提供。"""
    metrics = compute_structure_metrics(mask)
    return metrics.components, metrics.holes, metrics.stroke_width


def _match_points(reference: np.ndarray, candidate: np.ndarray, tolerance: float) -> int:
    """按最近距离贪心匹配端点或孔洞中心。"""
    if reference.size == 0 or candidate.size == 0:
        return 0
    distances = np.sqrt(np.sum((reference[:, None, :] - candidate[None, :, :]) ** 2, axis=2))
    pairs = np.argwhere(distances <= tolerance)
    if not pairs.size:
        return 0
    pair_distances = distances[pairs[:, 0], pairs[:, 1]]
    order = np.argsort(pair_distances, kind="stable")
    matches = 0
    used_reference = np.zeros(reference.shape[0], dtype=bool)
    used_candidate = np.zeros(candidate.shape[0], dtype=bool)
    for pair_index in order:
        ref_index = int(pairs[pair_index, 0])
        candidate_index = int(pairs[pair_index, 1])
        if used_reference[ref_index] or used_candidate[candidate_index]:
            continue
        used_reference[ref_index] = True
        used_candidate[candidate_index] = True
        matches += 1
    return matches


def compare_structure(
    mask_bin: np.ndarray,
    context: ScoreContext,
    metrics: Optional[StructureMetrics] = None,
) -> StructureComparison:
    """比较候选与参考的稳定拓扑关系。"""
    candidate = (mask_bin > 0).astype(np.uint8)
    current = metrics or compute_structure_metrics(
        candidate,
        context.reference_structure.comparison_min_component_area,
    )
    reference = context.reference_structure
    if reference.skeleton_total:
        tolerance_mask = cv2.dilate(candidate, np.ones((3, 3), dtype=np.uint8), iterations=1)
        skeleton_coverage = float((tolerance_mask & reference.skeleton).sum()) / reference.skeleton_total
    else:
        skeleton_coverage = 1.0

    point_tolerance = max(2.0, reference.stroke_width * 0.75)
    endpoint_matches = _match_points(reference.endpoints, current.endpoints, point_tolerance)
    if reference.endpoint_count:
        endpoint_retention = endpoint_matches / reference.endpoint_count
    else:
        endpoint_retention = 1.0 if current.endpoint_count == 0 else 0.0
    endpoint_growth = max(0, current.endpoint_count - reference.endpoint_count)
    if reference.branch_count:
        branch_retention = min(1.0, current.branch_count / reference.branch_count)
    else:
        branch_retention = 1.0 if current.branch_count <= 1 else 0.5

    hole_tolerance = max(3.0, reference.stroke_width * 1.5)
    hole_matches = _match_points(reference.hole_centroids, current.hole_centroids, hole_tolerance)
    hole_retention = hole_matches / max(1, reference.holes) if reference.holes else 1.0
    extra_holes = max(0, current.holes - hole_matches)
    stroke_ratio = current.stroke_width / reference.stroke_width if reference.stroke_width > 0 else 1.0
    return StructureComparison(
        skeleton_coverage=float(np.clip(skeleton_coverage, 0.0, 1.0)),
        endpoint_retention=float(np.clip(endpoint_retention, 0.0, 1.0)),
        endpoint_growth=endpoint_growth,
        branch_retention=float(np.clip(branch_retention, 0.0, 1.0)),
        hole_retention=float(np.clip(hole_retention, 0.0, 1.0)),
        extra_holes=extra_holes,
        stroke_ratio=float(stroke_ratio),
        component_delta=current.components - reference.components,
    )


def _compute_structure_penalty(
    mask_bin: np.ndarray,
    context: ScoreContext,
    metrics: Optional[StructureMetrics] = None,
    comparison: Optional[StructureComparison] = None,
) -> float:
    """惩罚明显断笔、误填字腔和整体削薄/增粗。"""
    if not context.ref_mask.any():
        return 0.0
    current = metrics or compute_structure_metrics(
        mask_bin,
        context.reference_structure.comparison_min_component_area,
    )
    relation = comparison or compare_structure(mask_bin, context, current)
    penalty = min(3.0, abs(relation.component_delta) * 0.7)
    penalty += min(2.5, (1.0 - relation.skeleton_coverage) * 10.0)
    penalty += min(2.0, (1.0 - relation.endpoint_retention) * 3.0 + relation.endpoint_growth * 0.25)
    penalty += min(2.5, (1.0 - relation.hole_retention) * 3.0 + relation.extra_holes * 0.6)
    if current.stroke_width > 0 and context.ref_stroke > 0:
        change = abs(relation.stroke_ratio - 1.0)
        if change > 0.18:
            penalty += min(2.0, (change - 0.18) * 5.0)
    return min(12.0, penalty)


def pareto_front_ranks(breakdowns: list[ScoreBreakdown]) -> list[int]:
    """计算多目标候选的 Pareto 层级，0 表示第一前沿。"""
    if not breakdowns:
        return []
    objectives = [item.objectives for item in breakdowns]
    remaining = set(range(len(breakdowns)))
    ranks = [0] * len(breakdowns)
    rank = 0
    while remaining:
        front: list[int] = []
        for candidate_index in sorted(remaining):
            candidate = objectives[candidate_index]
            dominated = False
            for other_index in remaining:
                if other_index == candidate_index:
                    continue
                other = objectives[other_index]
                if all(left >= right for left, right in zip(other, candidate)) and any(
                    left > right for left, right in zip(other, candidate)
                ):
                    dominated = True
                    break
            if not dominated:
                front.append(candidate_index)
        if not front:
            front = [min(remaining)]
        for index in front:
            ranks[index] = rank
            remaining.remove(index)
        rank += 1
    return ranks


def _compute_background_residue_penalty(mask_bin: np.ndarray, context: ScoreContext) -> float:
    """惩罚主体参考外围仍成片保留的前景，重点识别重噪背景残留。"""
    if not context.ref_mask.any():
        return 0.0
    dilation = cv2.dilate(context.ref_mask, np.ones((5, 5), dtype=np.uint8), iterations=1)
    outside = (mask_bin > 0) & (dilation == 0)
    residue_ratio = float(np.count_nonzero(outside)) / max(1, context.total_pixels)
    return min(12.0, max(0.0, residue_ratio - 0.003) * 180.0)


def _compute_ratio_score(text_pixels: int, total: int, heavy_noise: bool = False) -> float:
    """计算文字占比合理性得分；重噪图收紧背景占比过高的容忍范围。"""
    ratio = text_pixels / max(total, 1)
    if heavy_noise:
        if 0.025 <= ratio <= 0.38:
            return 20.0
        if ratio < 0.025:
            return max(0.0, ratio / 0.025 * 20.0)
        return max(0.0, 20.0 - (ratio - 0.38) * 70.0)
    if 0.005 <= ratio <= 0.60:
        return 20.0
    if ratio < 0.005:
        return max(0.0, ratio / 0.005 * 20.0)
    return max(0.0, 20.0 - (ratio - 0.60) * 50.0)
