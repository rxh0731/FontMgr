# optimizer.py — 逐字寻优引擎

import copy
import cv2
import math
import time
import uuid
import numpy as np
from PIL import Image
from typing import Any, Optional, Tuple

from core import pipeline
from core import scoring
from core import imaging
from data.log_manager import write_log

# 寻优派生上限（达到目标分即提前停止）
_DERIVE_LIMIT: int = 24


def auto_pick_for_image(
    gray_arr: np.ndarray,
    target_score: float = 90.0,
    derive_limit: int = _DERIVE_LIMIT,
) -> Tuple[str, dict[str, Any], float, bool]:
    """逐字寻优：为目标灰度图自动找到最优去杂方案。

    参数：
        gray_arr: 灰度 numpy float32 数组 (0~255)，已归一化极性
        target_score: 目标跑分 (0~100)
        derive_limit: 派生加试上限

    返回：
        (方案名, 方案dict, 得分, 是否达标)
    """
    diagnosis_id = uuid.uuid4().hex[:8]
    started_at = time.perf_counter()
    h, w = gray_arr.shape[:2]
    write_log(f"自动优化开始｜编号={diagnosis_id}｜尺寸={w}x{h}｜目标分={target_score}")

    stage_started = time.perf_counter()
    features = _auto_analyze(gray_arr)
    write_log(
        f"自动优化阶段｜编号={diagnosis_id}｜特征分析耗时={time.perf_counter() - stage_started:.4f}秒｜"
        f"光照={features['illum']:.4f}｜散点比例={features['speck']:.4f}｜散点数={features['speck_cnt']}"
    )

    stage_started = time.perf_counter()
    thumb, area_k = _auto_thumb(gray_arr)
    write_log(f"自动优化阶段｜编号={diagnosis_id}｜缩略图耗时={time.perf_counter() - stage_started:.4f}秒｜缩略图尺寸={thumb.shape[1]}x{thumb.shape[0]}")

    stage_started = time.perf_counter()
    candidates = _auto_build_candidates(features, area_k)
    if _is_heavy_noise(features):
        strong_candidates = _build_heavy_noise_beam_candidates(thumb, area_k, diagnosis_id)
        candidates.extend(strong_candidates)
    write_log(f"自动优化阶段｜编号={diagnosis_id}｜基础候选构建耗时={time.perf_counter() - stage_started:.4f}秒｜候选数={len(candidates)}")

    stage_started = time.perf_counter()
    best_name, best_scheme, best_score = _evaluate_candidates(thumb, candidates, diagnosis_id, "基础")
    write_log(f"自动优化阶段｜编号={diagnosis_id}｜基础候选总耗时={time.perf_counter() - stage_started:.4f}秒｜最佳={best_name}｜得分={best_score:.1f}")

    if best_score >= target_score:
        write_log(f"自动优化结束｜编号={diagnosis_id}｜总耗时={time.perf_counter() - started_at:.4f}秒｜派生=否")
        return best_name, best_scheme, best_score, True

    stage_started = time.perf_counter()
    derived = _auto_derive_candidates(best_scheme, best_name, derive_limit)
    write_log(f"自动优化阶段｜编号={diagnosis_id}｜派生候选构建耗时={time.perf_counter() - stage_started:.4f}秒｜派生数={len(derived)}")
    if derived:
        stage_started = time.perf_counter()
        d_name, d_scheme, d_score = _evaluate_candidates(thumb, derived, diagnosis_id, "派生")
        write_log(f"自动优化阶段｜编号={diagnosis_id}｜派生候选总耗时={time.perf_counter() - stage_started:.4f}秒｜最佳={d_name}｜得分={d_score:.1f}")
        if d_score >= target_score:
            write_log(f"自动优化结束｜编号={diagnosis_id}｜总耗时={time.perf_counter() - started_at:.4f}秒｜派生=达标")
            return d_name, d_scheme, d_score, True
        if d_score > best_score:
            best_name, best_scheme, best_score = d_name, d_scheme, d_score

    write_log(f"自动优化结束｜编号={diagnosis_id}｜总耗时={time.perf_counter() - started_at:.4f}秒｜派生=完成｜最终得分={best_score:.1f}")
    return best_name, best_scheme, best_score, best_score >= target_score


def generate_candidate_results(
    gray_arr: np.ndarray,
    parent_scheme: Optional[dict[str, Any]] = None,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """生成兼顾质量、处理路线和结果差异的候选结果。"""
    diagnosis_id = uuid.uuid4().hex[:8]
    started_at = time.perf_counter()
    h, w = gray_arr.shape[:2]
    write_log(f"候选生成开始｜编号={diagnosis_id}｜尺寸={w}x{h}｜模式={'继续探索' if parent_scheme else '基础候选'}｜显示上限={limit}")

    stage_started = time.perf_counter()
    features = _auto_analyze(gray_arr)
    quality_level = _classify_quality(features)
    analyze_elapsed = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    thumb, area_factor = _auto_thumb(gray_arr)
    thumb_elapsed = time.perf_counter() - stage_started
    write_log(
        f"候选生成特征｜编号={diagnosis_id}｜分析耗时={analyze_elapsed:.4f}秒｜缩略图耗时={thumb_elapsed:.4f}秒｜"
        f"缩略图={thumb.shape[1]}x{thumb.shape[0]}｜质量等级={quality_level}｜光照={features['illum']:.4f}｜噪声={features['noise']:.3f}｜"
        f"散点比例={features['speck']:.4f}｜散点数={features['speck_cnt']}｜文字占比={features['est_ratio']:.4f}"
    )

    stage_started = time.perf_counter()
    if parent_scheme:
        protect_original = _is_original_protection_scheme(parent_scheme)
        allow_structure_changes = quality_level in ("中度污染", "重度污染") and not protect_original
        candidate_schemes = [("当前选择", parent_scheme)] + _auto_derive_candidates(
            parent_scheme, "细化", _DERIVE_LIMIT, allow_structure_changes=allow_structure_changes
        )
        base_count = 1
    else:
        base_schemes = _auto_build_candidates(features, area_factor, quality_level)
        if quality_level == "重度污染":
            base_schemes.extend(_build_heavy_noise_beam_candidates(thumb, area_factor, diagnosis_id))
        base_count = len(base_schemes)
        candidate_schemes = list(base_schemes)
        if quality_level != "已足够干净":
            for scheme_name, scheme in base_schemes:
                if not _is_original_protection_scheme(scheme):
                    candidate_schemes.extend(_auto_derive_candidates(scheme, scheme_name, 6))
    write_log(
        f"候选生成方案｜编号={diagnosis_id}｜构建耗时={time.perf_counter() - stage_started:.4f}秒｜"
        f"基础数={base_count}｜实际评估数={len(candidate_schemes)}"
    )

    selected: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []
    evaluation_started = time.perf_counter()
    pipeline_total = 0.0
    scoring_total = 0.0
    slowest: list[tuple[float, str, float, float]] = []
    score_context = scoring.build_score_context(thumb)
    score_cache: dict[tuple[tuple[int, ...], bytes], tuple[float, str]] = {}
    reused_score_count = 0
    original_score: Optional[float] = None
    for scheme_name, scheme in candidate_schemes:
        candidate_started = time.perf_counter()
        pipeline_started = time.perf_counter()
        protect_original = _is_original_protection_scheme(scheme)
        if protect_original:
            mask = _original_foreground_mask(thumb)
        else:
            _, mask = pipeline.run_pipeline(thumb, scheme, timing_label=f"{diagnosis_id}/缩略图/{scheme_name}")
        pipeline_elapsed = time.perf_counter() - pipeline_started
        pipeline_total += pipeline_elapsed
        if mask is None:
            write_log(f"候选明细｜编号={diagnosis_id}｜方案={scheme_name}｜管线={pipeline_elapsed:.4f}秒｜结果=无掩码")
            continue
        mask_u8 = (mask > 0).astype(np.uint8)
        cache_key = (mask_u8.shape, np.packbits(mask_u8, axis=None).tobytes())
        scoring_started = time.perf_counter()
        cached_score = score_cache.get(cache_key)
        if cached_score is None:
            score, score_timing = scoring.auto_score_with_timing(mask_u8, thumb, context=score_context)
            score_cache[cache_key] = (score, score_timing)
        else:
            score, cached_timing = cached_score
            score_timing = f"复用相同掩码评分｜{cached_timing}"
            reused_score_count += 1
        if protect_original:
            original_score = float(score)
        rejected, protection_note = _reject_structure_damage(mask_u8, score_context)
        if not protect_original and quality_level in ("已足够干净", "低污染") and original_score is not None:
            minimum_gain = 4.0 if quality_level == "已足够干净" else 2.0
            if float(score) < original_score + minimum_gain:
                rejected = True
                protection_note = f"相对原图收益不足{minimum_gain:.1f}分"
        scoring_elapsed = time.perf_counter() - scoring_started
        scoring_total += scoring_elapsed
        candidate_elapsed = time.perf_counter() - candidate_started
        slowest.append((candidate_elapsed, scheme_name, pipeline_elapsed, scoring_elapsed))
        write_log(
            f"候选明细｜编号={diagnosis_id}｜方案={scheme_name}｜总耗时={candidate_elapsed:.4f}秒｜"
            f"管线={pipeline_elapsed:.4f}秒｜评分={scoring_elapsed:.4f}秒｜得分={score:.1f}｜"
            f"结构保护={'淘汰' if rejected else '通过'}({protection_note})｜{score_timing}"
        )
        if rejected and not protect_original:
            continue
        all_results.append({
            "方案名": scheme_name,
            "方案": scheme,
            "得分": float(score),
            "掩码": mask_u8,
            "质量等级": quality_level,
            "保留原图": protect_original,
        })
    all_results.sort(key=lambda item: item["得分"], reverse=True)
    write_log(
        f"候选生成评估｜编号={diagnosis_id}｜总耗时={time.perf_counter() - evaluation_started:.4f}秒｜"
        f"管线累计={pipeline_total:.4f}秒｜评分累计={scoring_total:.4f}秒｜有效结果={len(all_results)}｜"
        f"相同掩码复用={reused_score_count}"
    )

    dedup_started = time.perf_counter()
    compare_count = 0
    for result in all_results:
        is_similar = False
        for existing in selected:
            compare_count += 1
            if _masks_too_similar(result["掩码"], existing["掩码"]):
                is_similar = True
                break
        if is_similar:
            continue
        selected.append(result)
        if len(selected) >= max(1, limit):
            break
    original_result = next((item for item in all_results if item.get("保留原图")), None)
    original_selected = original_result is not None and any(item is original_result for item in selected)
    if original_result is not None and not original_selected:
        if len(selected) >= max(1, limit):
            selected[-1] = original_result
        else:
            selected.append(original_result)
        original_selected = True
    write_log(
        f"候选生成去重｜编号={diagnosis_id}｜耗时={time.perf_counter() - dedup_started:.4f}秒｜"
        f"比较次数={compare_count}｜保留数={len(selected)}｜原图基准={'已保留' if original_selected else '无'}"
    )

    full_size_started = time.perf_counter()
    full_size_total = 0.0
    full_score_context = scoring.build_score_context(gray_arr)
    verified: list[dict[str, Any]] = []
    for result in selected:
        item_started = time.perf_counter()
        if result.get("保留原图"):
            full_size_mask = _original_foreground_mask(gray_arr)
        else:
            _, full_size_mask = pipeline.run_pipeline(
                gray_arr, result["方案"], timing_label=f"{diagnosis_id}/全尺寸/{result['方案名']}"
            )
        item_elapsed = time.perf_counter() - item_started
        full_size_total += item_elapsed
        if full_size_mask is None:
            write_log(f"全尺寸明细｜编号={diagnosis_id}｜方案={result['方案名']}｜结果=无掩码，已淘汰")
            continue
        full_size_mask = (full_size_mask > 0).astype(np.uint8)
        rejected, protection_note = _reject_structure_damage(full_size_mask, full_score_context)
        if rejected and not result.get("保留原图"):
            write_log(
                f"全尺寸明细｜编号={diagnosis_id}｜方案={result['方案名']}｜管线耗时={item_elapsed:.4f}秒｜"
                f"原尺寸复核=淘汰({protection_note})"
            )
            continue
        result["掩码"] = full_size_mask
        if result.get("保留原图"):
            result["原始灰度"] = np.clip(gray_arr, 0, 255).astype(np.uint8)
        verified.append(result)
        write_log(
            f"全尺寸明细｜编号={diagnosis_id}｜方案={result['方案名']}｜管线耗时={item_elapsed:.4f}秒｜"
            f"原尺寸复核=通过({protection_note})"
        )
    selected = verified
    slowest.sort(reverse=True)
    slow_text = "；".join(f"{name}:{elapsed:.4f}秒(管线{pipe:.4f}/评分{score_time:.4f})" for elapsed, name, pipe, score_time in slowest[:5])
    write_log(
        f"候选生成结束｜编号={diagnosis_id}｜总耗时={time.perf_counter() - started_at:.4f}秒｜"
        f"全尺寸阶段={time.perf_counter() - full_size_started:.4f}秒｜全尺寸管线累计={full_size_total:.4f}秒｜最慢前五={slow_text}"
    )
    return selected


def _original_foreground_mask(gray_arr: np.ndarray) -> np.ndarray:
    """提取未经清噪的原图前景，使原图评分真实计入背景污染。"""
    source = np.clip(gray_arr, 0, 255).astype(np.uint8)
    threshold, _ = cv2.threshold(source, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return (source < threshold).astype(np.uint8)


def _structure_features(mask: np.ndarray) -> tuple[int, int, float]:
    """计算连通域、孔洞和平均笔画宽度，用于结构硬保护。"""
    source = (mask > 0).astype(np.uint8)
    count, _ = cv2.connectedComponents(source, connectivity=8)
    contours, hierarchy = cv2.findContours(source, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = 0
    if hierarchy is not None:
        holes = sum(1 for item in hierarchy[0] if item[3] >= 0)
    stroke = 0.0
    if source.any():
        distance = cv2.distanceTransform(source, cv2.DIST_L2, 5)
        values = distance[source > 0]
        stroke = float(np.median(values) * 2.0) if values.size else 0.0
    return max(0, count - 1), holes, stroke


def _reject_structure_damage(mask: np.ndarray, context: scoring.ScoreContext) -> tuple[bool, str]:
    """硬性淘汰明显断笔、丢失深墨核心或异常削细的候选。"""
    candidate = (mask > 0).astype(np.uint8)
    if not candidate.any():
        return True, "结果为空"
    core_coverage = float((candidate & context.core_mask).sum()) / max(1, context.core_total)
    ref_coverage = float((candidate & context.ref_mask).sum()) / max(1, context.ref_total)
    components, holes, stroke = _structure_features(candidate)
    if context.core_total and core_coverage < 0.94:
        return True, f"深墨核心仅保留{core_coverage:.1%}"
    if context.ref_total and ref_coverage < 0.82:
        return True, f"主体笔画仅保留{ref_coverage:.1%}"
    if context.ref_components > 0 and components > max(context.ref_components + 3, int(context.ref_components * 1.8)):
        return True, f"连通域由{context.ref_components}增至{components}"
    if context.ref_holes > 0 and holes < max(0, context.ref_holes - 2):
        return True, f"孔洞由{context.ref_holes}减至{holes}"
    if context.ref_stroke > 0 and stroke < context.ref_stroke * 0.72:
        return True, f"笔画宽度降至原参考的{stroke / context.ref_stroke:.1%}"
    return False, "结构完整"


def _mask_difference(left: np.ndarray, right: np.ndarray) -> float:
    """计算两个二值结果的像素差异比例。"""
    if left.shape != right.shape:
        return 1.0
    return float(np.mean((left > 0) != (right > 0)))


def _masks_too_similar(left: np.ndarray, right: np.ndarray) -> bool:
    """同时依据像素差异和前景交并比判定结构重复候选。"""
    if left.shape != right.shape:
        return False
    lhs = left > 0
    rhs = right > 0
    union = int(np.count_nonzero(lhs | rhs))
    iou = float(np.count_nonzero(lhs & rhs)) / max(1, union)
    return _mask_difference(left, right) < 0.012 or iou > 0.985


# ============================================================
# 阶段 A：特征分析
# ============================================================

def _auto_analyze(gray_arr: np.ndarray) -> dict[str, Any]:
    """提取 8 维特征向量。"""
    arr = np.clip(gray_arr, 0, 255).astype(np.uint8)
    h, w = arr.shape
    total = h * w

    med = float(np.median(arr))
    avg = float(np.mean(arr))

    # Otsu 初步二值化
    thresh, _ = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = arr < thresh
    text_pixels = int(mask.sum())

    # 文字灰度中位
    if mask.any():
        ink = float(np.median(arr[mask]))
    else:
        ink = 60.0

    # 前景对比度
    bg_pix = arr[~mask]
    bg_med = float(np.median(bg_pix)) if bg_pix.size > 0 else 255.0
    contrast = bg_med - ink

    # 前 8 大连通域面积估算文字占比
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S)
    areas = sorted(
        (int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)),
        reverse=True,
    )
    top8_sum = sum(areas[:8]) if areas else text_pixels
    est_ratio = top8_sum / max(total, 1)

    # 光照不均检测
    illum = _compute_illumination(arr, h, w)

    # 背景噪点
    noise, speck, speck_cnt = _compute_noise_features(arr, mask, h, w)

    return {
        "med": med, "avg": avg,
        "contrast": contrast, "est_ratio": est_ratio,
        "illum": illum, "noise": noise,
        "speck": speck, "speck_cnt": speck_cnt,
        "ink": ink, "text_pixels": text_pixels, "total_pixels": total,
    }


def _compute_illumination(arr: np.ndarray, h: int, w: int) -> float:
    """4x4 分块中位数的变异系数，衡量光照不均。"""
    py = max(1, h // 4)
    px = max(1, w // 4)
    medians = []
    for y in range(0, h, py):
        for x in range(0, w, px):
            patch = arr[y : y + py, x : x + px]
            if patch.size > 0:
                medians.append(np.median(patch))
    if not medians or np.mean(medians) == 0:
        return 0.0
    return float(np.std(medians) / np.mean(medians))


def _compute_noise_features(arr: np.ndarray, mask: np.ndarray, h: int, w: int) -> Tuple[float, float, int]:
    """计算背景噪点和散点特征。"""
    # 背景噪点：高亮区域（灰度 > 245）的拉普拉斯能量
    bg_zone = arr > 245
    if bg_zone.any():
        lap = cv2.Laplacian(arr.astype(np.float32), cv2.CV_32F)
        noise = float(np.mean(np.abs(lap[bg_zone])))
    else:
        noise = 0.0

    # 散点
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S)
    if num_labels <= 1:
        return noise, 0.0, 0
    areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
    areas.sort(key=lambda x: x[1], reverse=True)
    main_ids = {idx for idx, _ in areas[:5]}
    total_text = sum(a for _, a in areas)
    speck_px = sum(a for idx, a in areas if idx not in main_ids)
    speck_cnt = sum(1 for idx, _ in areas if idx not in main_ids)
    speck_ratio = speck_px / max(total_text, 1)
    return noise, speck_ratio, speck_cnt


# ============================================================
# 阶段 B：缩略图
# ============================================================

def _auto_thumb(gray_arr: np.ndarray) -> Tuple[np.ndarray, float]:
    """生成缩略图供快速跑分。"""
    h, w = gray_arr.shape
    max_dim = max(w, h)
    if max_dim <= 512:
        return gray_arr.copy(), 1.0
    ratio = 512.0 / max_dim
    nw = max(1, int(w * ratio))
    nh = max(1, int(h * ratio))
    arr_u8 = np.clip(gray_arr, 0, 255).astype(np.uint8)
    thumb = cv2.resize(arr_u8, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32)
    return thumb, ratio ** 2


# ============================================================
# 阶段 C：生成候选方案
# ============================================================

def _classify_quality(features: dict[str, Any]) -> str:
    """综合多个污染指标判定处理强度，避免单项波动触发激进路线。"""
    speck = float(features.get("speck", 0.0))
    speck_count = int(features.get("speck_cnt", 0))
    noise = float(features.get("noise", 0.0))
    illumination = float(features.get("illum", 0.0))
    heavy_signals = sum((speck >= 0.04, speck_count >= 40, noise >= 5.0, illumination >= 0.10))
    if heavy_signals >= 2 or speck >= 0.12 or speck_count >= 120:
        return "重度污染"
    if speck >= 0.04 or speck_count >= 40 or noise >= 5.0 or illumination >= 0.06:
        return "中度污染"
    if speck >= 0.01 or speck_count >= 8 or noise >= 2.5 or illumination >= 0.035:
        return "低污染"
    return "已足够干净"


def _is_original_protection_scheme(scheme: dict[str, Any]) -> bool:
    """判断方案是否要求完整保留原始灰度字形。"""
    return bool(scheme.get("保护原图", False))


def _auto_build_candidates(
    features: dict[str, Any], area_k: float, quality_level: str
) -> list[tuple[str, dict[str, Any]]]:
    """根据原图质量分级生成候选，处理强度只随确切污染逐级增加。"""
    est = max(0.04, min(0.4, float(features["est_ratio"])))
    base_min = max(10, int(60 * area_k))
    filter_params = {"min_area": base_min, "连通类型": 8, "相对模式": True, "相对比例": 0.002}
    shape_params = {**filter_params, "仅孤立": True, "最大长宽比": 3.0, "最小凸包比": 0.7, "最小实体比": 0.4}
    cands: list[tuple[str, dict[str, Any]]] = [
        ("原图保护·不去杂", {"保护原图": True, "质量等级": quality_level, "预处理": {}}),
        ("低污染·保守Otsu", {"预处理": {}, "L3": {"算法": "Otsu", "参数": {"偏移": 0}}}),
    ]
    if quality_level == "已足够干净":
        return cands

    cands.append(("干净·Otsu", {"预处理": {}, "L3": {"算法": "Otsu", "参数": {"偏移": 0}}, "L5": {"算法": "面积+形状过滤", "参数": shape_params}}))
    if quality_level == "低污染":
        return cands

    cands.append(("轻量·percentile", {"预处理": {}, "L3": {"算法": "percentile硬切", "参数": {"暗色比例": min(0.38, est * 1.2)}}, "L5": {"算法": "面积过滤", "参数": filter_params}}))
    if features["illum"] >= 0.06:
        cands.extend([
            ("光照不均·背景归一", {"预处理": {}, "L2": {"算法": "形态学背景归一", "参数": {"核大小": 51}}, "L3": {"算法": "Otsu", "参数": {"偏移": 0}}, "L5": {"算法": "面积过滤", "参数": filter_params}}),
            ("退化文档·Wolf", {"预处理": {}, "L3": {"算法": "Wolf-Jolion", "参数": {"窗口": 31, "k": 0.35}}, "L5": {"算法": "面积+形状过滤", "参数": shape_params}}),
            ("低对比·Phansalkar", {"预处理": {}, "L3": {"算法": "Phansalkar", "参数": {"窗口": 25, "k": 0.25, "R": 128, "p": 2.0, "q": 10.0}}, "L5": {"算法": "面积+形状过滤", "参数": shape_params}}),
        ])
    else:
        cands.append(("均衡·Sauvola", {"预处理": {}, "L3": {"算法": "Sauvola", "参数": {"窗口": 25, "k": 0.2, "R": 128}}, "L5": {"算法": "面积过滤", "参数": filter_params}}))
    if features.get("speck", 0) >= 0.01 or features.get("speck_cnt", 0) >= 8:
        cands.append(("散点拓片·灰度黑帽", {"预处理": {}, "L2": {"算法": "灰度黑帽增强", "参数": {"核大小": 15, "强度": 0.8}}, "L3": {"算法": "Otsu", "参数": {"偏移": 0}}, "L5": {"算法": "面积+形状过滤", "参数": shape_params}}))
    if quality_level == "重度污染":
        cands.append(("重噪·中值重建", {"预处理": {}, "L1": {"算法": "中值滤波", "参数": {"核大小": 3}}, "L3": {"算法": "Otsu", "参数": {"偏移": 0}}, "L4": {"算法": "形态学重建", "参数": {"半径": 1}}, "L5": {"算法": "面积+形状过滤", "参数": shape_params}}))
    if features.get("speck_cnt", 0) >= 8:
        cands.append(("边界污染·主体保护", {"预处理": {}, "L3": {"算法": "Otsu", "参数": {"偏移": 0}}, "L5": {"算法": "边界污染过滤", "参数": {"最大宽度比例": 0.18, "最大高度比例": 0.18, "保护主体数": 8}}}))
    return cands[:8]


# ============================================================
# 阶段 D：缩略图跑分
# ============================================================

def _evaluate_candidates(
    thumb: np.ndarray,
    candidates: list[tuple[str, dict[str, Any]]],
    diagnosis_id: str = "",
    stage_name: str = "寻优",
) -> Tuple[str, dict[str, Any], float]:
    """在缩略图上批量跑分，返回最高分候选。"""
    best_name, best_scheme, best_score = "", {}, -1.0
    for name, scheme in candidates:
        started_at = time.perf_counter()
        pipeline_started = time.perf_counter()
        _, mask = pipeline.run_pipeline(thumb, scheme, timing_label=f"{diagnosis_id}/{stage_name}/{name}")
        pipeline_elapsed = time.perf_counter() - pipeline_started
        if mask is not None:
            scoring_started = time.perf_counter()
            score, score_timing = scoring.auto_score_with_timing(mask, thumb)
            scoring_elapsed = time.perf_counter() - scoring_started
            write_log(
                f"寻优候选明细｜编号={diagnosis_id}｜阶段={stage_name}｜方案={name}｜"
                f"总耗时={time.perf_counter() - started_at:.4f}秒｜管线={pipeline_elapsed:.4f}秒｜"
                f"评分={scoring_elapsed:.4f}秒｜得分={score:.1f}｜{score_timing}"
            )
            if score > best_score:
                best_score = score
                best_name = name
                best_scheme = scheme
    return best_name, best_scheme, best_score


# ============================================================
# 阶段 E：重噪束搜索与梯度派生加试
# ============================================================

def _is_heavy_noise(features: dict[str, float]) -> bool:
    """判断是否需要启用强去杂搜索。"""
    return bool(features.get("speck", 0.0) >= 0.04 or features.get("speck_cnt", 0.0) >= 40 or features.get("noise", 0.0) >= 5.0)


def _clone_scheme(scheme: dict[str, Any]) -> dict[str, Any]:
    """深拷贝方案，避免派生候选共享嵌套参数。"""
    return copy.deepcopy(scheme)


def _beam_rank(thumb: np.ndarray, candidates: list[tuple[str, dict[str, Any]]], keep: int) -> list[tuple[str, dict[str, Any]]]:
    """评分并保留当前阶段最优且结果不同的少量方案。"""
    context = scoring.build_score_context(thumb)
    ranked: list[tuple[float, str, dict[str, Any], np.ndarray]] = []
    for name, scheme in candidates:
        _, mask = pipeline.run_pipeline(thumb, scheme)
        if mask is None or not mask.any():
            continue
        mask_u8 = (mask > 0).astype(np.uint8)
        ranked.append((scoring.auto_score(mask_u8, thumb, context=context), name, scheme, mask_u8))
    ranked.sort(key=lambda item: item[0], reverse=True)
    selected: list[tuple[str, dict[str, Any]]] = []
    selected_masks: list[np.ndarray] = []
    for _, name, scheme, mask in ranked:
        if any(_masks_too_similar(mask, existing) for existing in selected_masks):
            continue
        selected.append((name, scheme))
        selected_masks.append(mask)
        if len(selected) >= keep:
            break
    return selected


def _build_heavy_noise_beam_candidates(thumb: np.ndarray, area_factor: float, diagnosis_id: str) -> list[tuple[str, dict[str, Any]]]:
    """分阶段搜索中值、阈值、开运算和主体过滤，避免一次穷举数万组合。"""
    stage1: list[tuple[str, dict[str, Any]]] = []
    for kernel in (3, 5, 7, 9, 11):
        stage1.append((f"强去杂·中值{kernel}", {
            "预处理": {}, "L1": {"算法": "中值滤波", "参数": {"核大小": kernel}},
            "L3": {"算法": "Otsu", "参数": {"偏移": -12}},
        }))
    beam = _beam_rank(thumb, stage1, 4)

    stage2: list[tuple[str, dict[str, Any]]] = []
    for parent_name, parent in beam:
        for offset in (-35, -24, -12, 0, 12, 24, 40):
            scheme = _clone_scheme(parent)
            scheme["L3"] = {"算法": "Otsu", "参数": {"偏移": offset}}
            stage2.append((f"{parent_name}·Otsu{offset:+d}", scheme))
        for threshold in (120, 140, 160, 180, 200, 220):
            scheme = _clone_scheme(parent)
            scheme["L3"] = {"算法": "固定阈值", "参数": {"阈值": threshold}}
            stage2.append((f"{parent_name}·阈值{threshold}", scheme))
    beam = _beam_rank(thumb, stage2, 5)

    stage3: list[tuple[str, dict[str, Any]]] = []
    for parent_name, parent in beam:
        stage3.append((parent_name, parent))
        for shape_name, shape in (("椭圆", 1), ("矩形", 0), ("十字", 2)):
            for radius in (1, 2, 3, 4):
                scheme = _clone_scheme(parent)
                scheme["L4"] = {"算法": "开运算", "参数": {"半径": radius, "迭代": 1, "核形状": shape}}
                stage3.append((f"{parent_name}·{shape_name}开运算{radius}", scheme))
    beam = _beam_rank(thumb, stage3, 5)

    stage4: list[tuple[str, dict[str, Any]]] = []
    for parent_name, parent in beam:
        stage4.append((parent_name, parent))
        for min_area in (20, 40, 80, 140):
            scheme = _clone_scheme(parent)
            scheme["L5"] = {"算法": "面积+形状过滤", "参数": {
                "min_area": max(8, int(min_area * area_factor)), "相对模式": True, "相对比例": 0.0015,
                "仅孤立": True, "最大长宽比": 4.5, "最小凸包比": 0.35, "最小实体比": 0.25,
            }}
            stage4.append((f"{parent_name}·主体过滤{min_area}", scheme))
    final = _beam_rank(thumb, stage4, 6)
    write_log(f"强去杂束搜索｜编号={diagnosis_id}｜阶段候选={len(stage1)}/{len(stage2)}/{len(stage3)}/{len(stage4)}｜最终保留={len(final)}")
    return final


def _auto_derive_candidates(
    base_scheme: dict[str, Any],
    base_name: str,
    limit: int = 24,
    allow_structure_changes: bool = False,
) -> list[tuple[str, dict[str, Any]]]:
    """基于最优候选做参数微调派生。"""
    derived: list[tuple[str, dict[str, Any]]] = []
    name_prefix = base_name + "·派生"

    for layer_key in ("L1", "L2", "L3", "L4", "L5"):
        cfg = base_scheme.get(layer_key)
        if not cfg:
            continue
        algo = cfg.get("算法", "")
        params = dict(cfg.get("参数", {}))

        variants: list[dict[str, Any]] = []
        if algo == "percentile硬切" and len(derived) < limit:
            r = params.get("暗色比例", 0.2)
            for dr in (-0.03, 0.03, -0.06, 0.06):
                nr = max(0.05, min(0.5, r + dr))
                if abs(nr - r) > 0.001:
                    p = dict(params)
                    p["暗色比例"] = round(nr, 3)
                    variants.append(p)
        elif algo == "Otsu" and len(derived) < limit:
            off = params.get("偏移", 0)
            for do in (-20, -10, 10, 20):
                no = max(-50, min(50, off + do))
                if no != off:
                    p = dict(params)
                    p["偏移"] = no
                    variants.append(p)
        elif algo == "固定阈值" and len(derived) < limit:
            threshold = params.get("阈值", 160)
            for delta in (-30, -15, 15, 30):
                value = max(20, min(245, threshold + delta))
                if value != threshold:
                    p = dict(params)
                    p["阈值"] = value
                    variants.append(p)
        elif algo in ("Sauvola", "Wolf-Jolion", "Phansalkar") and len(derived) < limit:
            k = params.get("k", 0.2)
            w = params.get("窗口", 25)
            k_values = (0.15, 0.25, 0.35, 0.45) if algo == "Wolf-Jolion" else (0.1, 0.15, 0.25, 0.3)
            for nk in k_values:
                if abs(nk - k) < 0.001:
                    continue
                for nw in (19, 31, 41):
                    if abs(nw - w) < 2:
                        continue
                    p = dict(params)
                    p["k"] = nk
                    p["窗口"] = nw
                    variants.append(p)
        elif algo in ("黑帽扣除", "灰度黑帽增强") and len(derived) < limit:
            ks = params.get("核大小", 11)
            for nks in (7, 9, 13):
                if nks <= 13 and nks != ks:
                    p = dict(params)
                    p["核大小"] = nks
                    variants.append(p)
        elif algo == "中值滤波" and len(derived) < limit:
            ks = params.get("核大小", 3)
            for nks in (3, 5, 7, 9, 11):
                if nks != ks:
                    p = dict(params)
                    p["核大小"] = nks
                    variants.append(p)
        elif algo == "开运算" and len(derived) < limit:
            radius = params.get("半径", 1)
            shape = params.get("核形状", 1)
            for next_radius in (1, 2, 3, 4):
                for next_shape in (0, 1, 2):
                    if next_radius == radius and next_shape == shape:
                        continue
                    p = dict(params)
                    p["半径"] = next_radius
                    p["核形状"] = next_shape
                    variants.append(p)
        elif algo in ("面积过滤", "面积+形状过滤") and len(derived) < limit:
            ma = params.get("min_area", 60)
            for mul in (0.5, 2, 4):
                nma = int(ma * mul)
                if nma != ma and nma >= 10:
                    p = dict(params)
                    p["min_area"] = nma
                    variants.append(p)

        for i, vp in enumerate(variants):
            if len(derived) >= limit:
                break
            new_scheme = dict(base_scheme)
            new_layer = dict(cfg)
            new_layer["参数"] = vp
            new_scheme[layer_key] = new_layer
            derived.append((f"{name_prefix}{len(derived)+1}", new_scheme))

    if allow_structure_changes and len(derived) < limit:
        structural_variants: list[tuple[str, str, dict[str, Any]]] = []
        if not base_scheme.get("L1"):
            structural_variants.extend([
                ("新增中值3", "L1", {"算法": "中值滤波", "参数": {"核大小": 3}}),
                ("新增中值5", "L1", {"算法": "中值滤波", "参数": {"核大小": 5}}),
            ])
        if base_scheme.get("L3", {}).get("算法") != "固定阈值":
            for threshold in (140, 160, 180, 200):
                structural_variants.append((f"改固定阈值{threshold}", "L3", {"算法": "固定阈值", "参数": {"阈值": threshold}}))
        if base_scheme.get("L3", {}).get("算法") != "Otsu":
            structural_variants.append(("改Otsu", "L3", {"算法": "Otsu", "参数": {"偏移": -12}}))
        if base_scheme.get("L4", {}).get("算法") != "开运算":
            structural_variants.extend([
                ("改椭圆开运算1", "L4", {"算法": "开运算", "参数": {"半径": 1, "迭代": 1, "核形状": 1}}),
                ("改椭圆开运算2", "L4", {"算法": "开运算", "参数": {"半径": 2, "迭代": 1, "核形状": 1}}),
                ("改椭圆开运算3", "L4", {"算法": "开运算", "参数": {"半径": 3, "迭代": 1, "核形状": 1}}),
            ])
        if not base_scheme.get("L5"):
            structural_variants.append(("新增主体过滤", "L5", {"算法": "面积+形状过滤", "参数": {
                "min_area": 40, "相对模式": True, "相对比例": 0.0015, "仅孤立": True,
                "最大长宽比": 4.5, "最小凸包比": 0.35, "最小实体比": 0.25,
            }}))
        for change_name, layer_key, layer in structural_variants:
            if len(derived) >= limit:
                break
            new_scheme = _clone_scheme(base_scheme)
            new_scheme[layer_key] = layer
            derived.append((f"{base_name}·{change_name}", new_scheme))

    return derived
