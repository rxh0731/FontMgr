# scoring.py — 无参考自动评分体系（满分 100）

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


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


def build_score_context(gray_arr: np.ndarray, total_pixels: Optional[int] = None) -> ScoreContext:
    """预计算候选评分时保持不变的参考数据。"""
    total = total_pixels or gray_arr.size
    ref_mask = _build_reference_mask(gray_arr)
    core_mask = _build_stable_core_mask(gray_arr, ref_mask)
    ref_total = int(ref_mask.sum())
    core_total = int(core_mask.sum())
    raw_noise_ratio, raw_noise_count = _estimate_raw_noise(gray_arr)
    heavy_noise = raw_noise_ratio >= 0.04 or raw_noise_count >= 40
    if ref_total:
        _, _, _, centroids = cv2.connectedComponentsWithStats(ref_mask, connectivity=8, ltype=cv2.CV_32S)
        ref_centroids = np.asarray(centroids[1:], dtype=np.float64)
        text_median = float(np.median(gray_arr[ref_mask > 0]))
        ys, xs = np.where(ref_mask > 0)
        diagonal = float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min()))
        ref_components, ref_holes, ref_stroke = _structure_features(ref_mask)
    else:
        ref_centroids = np.empty((0, 2), dtype=np.float64)
        text_median = 60.0
        diagonal = 100.0
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
    )


def auto_score(
    mask: np.ndarray,
    gray_arr: np.ndarray,
    total_pixels: Optional[int] = None,
    context: Optional[ScoreContext] = None,
) -> float:
    """对去杂结果进行百分制评分。"""
    score, _ = auto_score_with_timing(mask, gray_arr, total_pixels, context)
    return score


def auto_score_with_timing(
    mask: np.ndarray,
    gray_arr: np.ndarray,
    total_pixels: Optional[int] = None,
    context: Optional[ScoreContext] = None,
) -> tuple[float, str]:
    """评分并返回可直接写入诊断日志的子阶段耗时。"""
    total_started = time.perf_counter()
    if mask is None or not mask.any():
        return 0.0, "评分总计=0.0000秒｜空掩码"

    prepare_started = time.perf_counter()
    mask_bin = (mask > 0).astype(np.uint8)
    text_pixels = int(mask_bin.sum())
    prepare_elapsed = time.perf_counter() - prepare_started

    reference_started = time.perf_counter()
    score_context = context or build_score_context(gray_arr, total_pixels)
    ref_mask = score_context.ref_mask
    reference_elapsed = time.perf_counter() - reference_started
    if not ref_mask.any():
        stroke_score = 40.0
    else:
        overlap = int((mask_bin & ref_mask).sum())
        coverage = overlap / max(1, score_context.ref_total)
        core_overlap = int((mask_bin & score_context.core_mask).sum())
        core_coverage = core_overlap / max(1, score_context.core_total)
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
    noise_elapsed = time.perf_counter() - noise_started
    noise_score = max(0.0, 40.0 - penalty)

    ratio_started = time.perf_counter()
    ratio_score = _compute_ratio_score(text_pixels, score_context.total_pixels, score_context.heavy_noise)
    background_penalty = _compute_background_residue_penalty(mask_bin, score_context) if score_context.heavy_noise else 0.0
    ratio_elapsed = time.perf_counter() - ratio_started

    structure_started = time.perf_counter()
    structure_penalty = _compute_structure_penalty(mask_bin, score_context)
    structure_elapsed = time.perf_counter() - structure_started

    total_score = stroke_score + noise_score + ratio_score - structure_penalty - background_penalty
    score = round(min(100.0, max(0.0, total_score)), 1)
    total_elapsed = time.perf_counter() - total_started
    timing = (
        f"评分总计={total_elapsed:.4f}秒｜准备={prepare_elapsed:.4f}秒｜参考掩码={reference_elapsed:.4f}秒｜"
        f"噪点评分={noise_elapsed:.4f}秒｜结构评分={structure_elapsed:.4f}秒｜占比评分={ratio_elapsed:.4f}秒｜"
        f"前景像素={text_pixels}｜噪声连通域={noise_components}｜噪点扣分={penalty:.1f}｜"
        f"背景残留扣分={background_penalty:.1f}｜结构扣分={structure_penalty:.1f}｜重噪={'是' if score_context.heavy_noise else '否'}"
    )
    return score, timing


def _keep_main_components(mask: np.ndarray, limit: int = 8, min_ratio: float = 0.04) -> np.ndarray:
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


def _build_reference_mask(gray_arr: np.ndarray) -> np.ndarray:
    """构建经过中值清噪的主体参考掩码。"""
    arr_u8 = np.clip(gray_arr, 0, 255).astype(np.uint8)
    filtered = cv2.medianBlur(arr_u8, 3)
    threshold, _ = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    raw_mask = (filtered < threshold).astype(np.uint8)
    return _keep_main_components(raw_mask)


def _build_stable_core_mask(gray_arr: np.ndarray, ref_mask: np.ndarray) -> np.ndarray:
    """构建多个保守阈值下都存在的稳定深墨核心。"""
    arr_u8 = np.clip(gray_arr, 0, 255).astype(np.uint8)
    filtered = cv2.medianBlur(arr_u8, 3)
    otsu, _ = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    q18 = float(np.percentile(filtered, 18.0))
    threshold = int(np.clip(min(float(otsu) - 12.0, q18), 10, 235))
    core = ((filtered < threshold) & (ref_mask > 0)).astype(np.uint8)
    if int(core.sum()) < max(10, int(ref_mask.sum() * 0.25)):
        core = ((filtered < max(10, int(otsu) - 6)) & (ref_mask > 0)).astype(np.uint8)
    return core


def _estimate_raw_noise(gray_arr: np.ndarray) -> tuple[float, int]:
    """估算原图 Otsu 掩码中小连通域的面积比例和数量。"""
    arr_u8 = np.clip(gray_arr, 0, 255).astype(np.uint8)
    threshold, _ = cv2.threshold(arr_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    raw = (arr_u8 < threshold).astype(np.uint8)
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
    if ref_total:
        _, _, _, centroids = cv2.connectedComponentsWithStats(ref_mask, connectivity=8, ltype=cv2.CV_32S)
        ref_centroids = np.asarray(centroids[1:], dtype=np.float64)
        text_median = float(np.median(gray_arr[ref_mask > 0]))
        ys, xs = np.where(ref_mask > 0)
        diagonal = float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min()))
        ref_components, ref_holes, ref_stroke = _structure_features(ref_mask)
    else:
        ref_centroids = np.empty((0, 2), dtype=np.float64)
        text_median = 60.0
        diagonal = 100.0
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
    """返回主体连通域数、内部孔洞数和中位半笔画宽度。"""
    source = (mask > 0).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(source, connectivity=8, ltype=cv2.CV_32S)
    areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.array([], dtype=np.int32)
    meaningful = int(np.count_nonzero(areas >= max(3, source.size * 0.00003)))
    contours, hierarchy = cv2.findContours(source * 255, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = 0
    if hierarchy is not None:
        holes = sum(1 for item in hierarchy[0] if item[3] >= 0)
    distance = cv2.distanceTransform(source, cv2.DIST_L2, 3)
    values = distance[distance > 0]
    stroke = float(np.median(values)) if values.size else 0.0
    return meaningful, holes, stroke


def _compute_structure_penalty(mask_bin: np.ndarray, context: ScoreContext) -> float:
    """惩罚明显断笔、误填字腔和整体削薄/增粗。"""
    if not context.ref_mask.any():
        return 0.0
    comp, holes, stroke = _structure_features(mask_bin)
    penalty = min(5.0, abs(comp - context.ref_components) * 0.8)
    penalty += min(4.0, abs(holes - context.ref_holes) * 1.5)
    if context.ref_stroke > 0:
        change = abs(stroke - context.ref_stroke) / context.ref_stroke
        if change > 0.18:
            penalty += min(3.0, (change - 0.18) * 8.0)
    return min(12.0, penalty)


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
