# optimizer.py — 逐字寻优引擎

import copy
import cv2
import json
import math
import time
import uuid
import numpy as np
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

from core import pipeline
from core import scoring
from core import imaging
from core import foreground_analysis
from core import stroke_scale_analysis
from core.component_policy import (
    PRIMARY_CLUSTER_COMPONENT_LIMIT,
    STRUCTURE_PROTECTION_COMPONENT_LIMIT,
)
from data.log_manager import write_log

# 寻优派生上限（达到目标分即提前停止）
_DERIVE_LIMIT: int = 24


class OptimizationCancelled(RuntimeError):
    """调用方请求停止当前自动优化任务。"""


def _raise_if_cancelled(cancel_check: Optional[Callable[[], bool]]) -> None:
    """在可安全中断的计算边界响应调用方取消请求。"""
    if cancel_check is not None and cancel_check():
        raise OptimizationCancelled("自动优化已由用户停止。")


def _scoring_cancel_check(
    cancel_check: Optional[Callable[[], bool]],
) -> Optional[Callable[[], None]]:
    """把布尔取消查询转换为评分层可调用的异常回调。"""
    if cancel_check is None:
        return None

    def check() -> None:
        _raise_if_cancelled(cancel_check)

    return check


@dataclass(frozen=True)
class ScaleProfile:
    """自动方案在原图尺度下使用的像素参数基准。"""

    width: int
    height: int
    stroke_width: float
    local_window: int
    background_kernel: int
    noise_kernel: int
    blackhat_kernel: int
    morph_radius: int
    min_area: int


@dataclass(frozen=True)
class _PreviewEvaluation:
    """单次候选生成任务内可复用的缩略图计算结果。"""

    mask: Optional[np.ndarray]
    breakdown: Optional[scoring.ScoreBreakdown]
    score_timing: str
    rejected: bool
    protection_note: str
    pipeline_elapsed: float
    scoring_elapsed: float
    score_reused: bool = False


def _odd_value(value: float, minimum: int = 3, maximum: int = 201) -> int:
    result = max(minimum, min(maximum, int(round(value))))
    return result if result % 2 == 1 else min(maximum, result + 1)


def _build_scale_profile(gray_arr: np.ndarray, structure: scoring.StructureMetrics) -> ScaleProfile:
    """根据原图尺寸和参考笔画宽度建立自动参数基准。"""
    height, width = gray_arr.shape[:2]
    short_side = max(1, min(width, height))
    stroke = float(np.clip(structure.stroke_width or short_side * 0.015, 1.0, short_side * 0.12))
    return ScaleProfile(
        width=width,
        height=height,
        stroke_width=stroke,
        local_window=_odd_value(max(stroke * 4.0, short_side * 0.04), 15, 101),
        background_kernel=_odd_value(max(stroke * 8.0, short_side * 0.10), 31, 201),
        noise_kernel=_odd_value(stroke * 0.45, 3, 11),
        blackhat_kernel=_odd_value(max(stroke * 1.5, short_side * 0.025), 7, 31),
        morph_radius=max(1, min(6, int(round(stroke * 0.12)))),
        min_area=max(10, min(500, int(round(stroke * stroke * 0.35)))),
    )


def _mark_adaptive_scheme(scheme: dict[str, Any], profile: ScaleProfile) -> dict[str, Any]:
    """给新生成的自动方案写入尺度元数据，旧方案不会自动获得该标记。"""
    marked = _clone_scheme(scheme)
    marked["自适应尺度"] = {
        "基准宽度": profile.width,
        "基准高度": profile.height,
        "参考笔画宽度": round(profile.stroke_width, 3),
        "最小连通域面积": profile.min_area,
    }
    return marked


@stroke_scale_analysis.use_stroke_scale_analysis_cache
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
    full_context = scoring.build_score_context(gray_arr)
    features = _auto_analyze(gray_arr)
    quality_level = _classify_quality(features)
    scale_profile = _build_scale_profile(gray_arr, full_context.reference_structure)
    write_log(
        f"自动优化阶段｜编号={diagnosis_id}｜特征分析耗时={time.perf_counter() - stage_started:.4f}秒｜"
        f"光照={features['illum']:.4f}｜散点比例={features['speck']:.4f}｜散点数={features['speck_cnt']}"
    )

    stage_started = time.perf_counter()
    thumb, _ = _auto_thumb(gray_arr)
    write_log(f"自动优化阶段｜编号={diagnosis_id}｜缩略图耗时={time.perf_counter() - stage_started:.4f}秒｜缩略图尺寸={thumb.shape[1]}x{thumb.shape[0]}")

    stage_started = time.perf_counter()
    candidates = _auto_build_candidates(features, scale_profile, quality_level)
    if _is_heavy_noise(features):
        strong_candidates = _build_heavy_noise_beam_candidates(thumb, scale_profile, diagnosis_id)
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
        d_name, d_scheme, d_score = _evaluate_candidates(
            thumb,
            candidates + derived,
            diagnosis_id,
            "基础+派生",
        )
        write_log(f"自动优化阶段｜编号={diagnosis_id}｜派生候选总耗时={time.perf_counter() - stage_started:.4f}秒｜最佳={d_name}｜得分={d_score:.1f}")
        if d_score >= target_score:
            write_log(f"自动优化结束｜编号={diagnosis_id}｜总耗时={time.perf_counter() - started_at:.4f}秒｜派生=达标")
            return d_name, d_scheme, d_score, True
        best_name, best_scheme, best_score = d_name, d_scheme, d_score

    write_log(f"自动优化结束｜编号={diagnosis_id}｜总耗时={time.perf_counter() - started_at:.4f}秒｜派生=完成｜最终得分={best_score:.1f}")
    return best_name, best_scheme, best_score, best_score >= target_score


@stroke_scale_analysis.use_stroke_scale_analysis_cache
def generate_candidate_results(
    gray_arr: np.ndarray,
    parent_scheme: Optional[dict[str, Any]] = None,
    limit: int = 8,
    reference_gray_arr: Optional[np.ndarray] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> list[dict[str, Any]]:
    """生成候选；可用独立参考灰度评价模型或其他前置处理结果。"""
    diagnosis_id = uuid.uuid4().hex[:8]
    started_at = time.perf_counter()
    reference_gray = gray_arr if reference_gray_arr is None else np.asarray(reference_gray_arr)
    if gray_arr.ndim != 2 or reference_gray.ndim != 2:
        raise ValueError("自动优化只接受二维灰度图。")
    if reference_gray.shape[:2] != gray_arr.shape[:2]:
        raise ValueError("评分参考图与管线输入图尺寸必须一致。")
    h, w = gray_arr.shape[:2]
    scoring_cancel_check = _scoring_cancel_check(cancel_check)
    write_log(f"候选生成开始｜编号={diagnosis_id}｜尺寸={w}x{h}｜模式={'继续探索' if parent_scheme else '基础候选'}｜显示上限={limit}")

    stage_started = time.perf_counter()
    _raise_if_cancelled(cancel_check)
    full_score_context = scoring.build_score_context(
        reference_gray,
        cancel_check=scoring_cancel_check,
    )
    _raise_if_cancelled(cancel_check)
    features = _auto_analyze(gray_arr)
    quality_level = _classify_quality(features)
    scale_profile = _build_scale_profile(gray_arr, full_score_context.reference_structure)
    analyze_elapsed = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    thumb, _ = _auto_thumb(gray_arr)
    if reference_gray_arr is None:
        reference_thumb = thumb
    else:
        reference_u8 = np.clip(reference_gray, 0, 255).astype(np.uint8)
        reference_thumb = cv2.resize(
            reference_u8,
            (thumb.shape[1], thumb.shape[0]),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)
    if reference_gray_arr is None and reference_thumb.shape == reference_gray.shape:
        score_context = full_score_context
    else:
        score_context = scoring.build_score_context(
            reference_thumb,
            cancel_check=scoring_cancel_check,
        )
    preview_cache: dict[str, _PreviewEvaluation] = {}
    score_cache: dict[tuple[tuple[int, ...], bytes], tuple[scoring.ScoreBreakdown, str]] = {}
    thumb_elapsed = time.perf_counter() - stage_started
    write_log(
        f"候选生成特征｜编号={diagnosis_id}｜分析耗时={analyze_elapsed:.4f}秒｜缩略图耗时={thumb_elapsed:.4f}秒｜"
        f"缩略图={thumb.shape[1]}x{thumb.shape[0]}｜质量等级={quality_level}｜光照={features['illum']:.4f}｜噪声={features['noise']:.3f}｜"
        f"散点比例={features['speck']:.4f}｜散点数={features['speck_cnt']}｜文字占比={features['est_ratio']:.4f}"
    )

    stage_started = time.perf_counter()
    fallback_candidate_schemes: list[tuple[str, dict[str, Any]]] = []
    if parent_scheme:
        protect_original = _is_original_protection_scheme(parent_scheme)
        allow_structure_changes = quality_level in ("中度污染", "重度污染") and not protect_original
        candidate_schemes = [("当前选择", parent_scheme)] + _auto_derive_candidates(
            parent_scheme, "细化", _DERIVE_LIMIT, allow_structure_changes=allow_structure_changes
        )
        base_count = 1
    else:
        base_schemes = _auto_build_candidates(features, scale_profile, quality_level)
        derivation_bases = [
            (name, scheme)
            for name, scheme in base_schemes
            if not _is_original_protection_scheme(scheme)
        ]
        derivation_parents: list[tuple[str, dict[str, Any]]] = []
        if quality_level != "已足够干净":
            derivation_parents = _beam_rank(
                thumb,
                derivation_bases,
                min(4, len(derivation_bases)),
                reference_thumb,
                score_context,
                preview_cache,
                score_cache,
                family_limit=1,
                cancel_check=cancel_check,
            )
        if quality_level == "重度污染":
            strict_base_count = sum(
                1
                for _, scheme in derivation_bases
                if (entry := preview_cache.get(_scheme_cache_key(scheme))) is not None
                and entry.breakdown is not None
                and not entry.rejected
            )
            if strict_base_count < 3:
                base_schemes.extend(_build_heavy_noise_beam_candidates(
                    thumb,
                    scale_profile,
                    diagnosis_id,
                    reference_thumb,
                    score_context,
                    preview_cache,
                    score_cache,
                    cancel_check=cancel_check,
                ))
                derivation_bases = [
                    (name, scheme)
                    for name, scheme in base_schemes
                    if not _is_original_protection_scheme(scheme)
                ]
                derivation_parents = _beam_rank(
                    thumb,
                    derivation_bases,
                    min(4, len(derivation_bases)),
                    reference_thumb,
                    score_context,
                    preview_cache,
                    score_cache,
                    family_limit=1,
                    cancel_check=cancel_check,
                )
            else:
                write_log(
                    f"强去杂束搜索跳过｜编号={diagnosis_id}｜"
                    f"通用基础路线已通过={strict_base_count}"
                )
        base_count = len(base_schemes)
        candidate_schemes = list(base_schemes)
        if quality_level != "已足够干净":
            parent_keys = {_scheme_cache_key(scheme) for _, scheme in derivation_parents}
            for scheme_name, scheme in derivation_parents:
                candidate_schemes.extend(_auto_derive_candidates(scheme, scheme_name, 4))
            for scheme_name, scheme in derivation_bases:
                if _scheme_cache_key(scheme) in parent_keys:
                    continue
                fallback_candidate_schemes.extend(
                    _auto_derive_candidates(scheme, scheme_name, 3)
                )
    write_log(
        f"候选生成方案｜编号={diagnosis_id}｜构建耗时={time.perf_counter() - stage_started:.4f}秒｜"
        f"基础数={base_count}｜首轮评估数={len(candidate_schemes)}｜"
        f"候补派生数={len(fallback_candidate_schemes)}"
    )

    all_results: list[dict[str, Any]] = []
    preview_risky_results: list[dict[str, Any]] = []
    evaluation_started = time.perf_counter()
    pipeline_total = 0.0
    scoring_total = 0.0
    slowest: list[tuple[float, str, float, float]] = []
    reused_score_count = 0
    original_score: Optional[float] = None
    prefilter_budget = max(6, min(12, max(1, int(limit)) * 2))
    ranked_candidate_schemes = _prefilter_candidate_schemes(
        thumb,
        candidate_schemes,
        score_context,
        preview_cache,
        prefilter_budget,
        cancel_check=cancel_check,
    )
    pending_schemes = ranked_candidate_schemes[:prefilter_budget]
    deferred_candidate_schemes = ranked_candidate_schemes[prefilter_budget:]
    if deferred_candidate_schemes:
        write_log(
            f"候选快速初筛｜编号={diagnosis_id}｜"
            f"初筛前={len(candidate_schemes)}｜进入完整评分={len(pending_schemes)}"
        )
    fallback_added = False
    candidate_index = 0
    minimum_result_count = max(2, min(max(1, limit), 4))
    while True:
        if candidate_index >= len(pending_schemes):
            if (
                len(all_results) < minimum_result_count
                and deferred_candidate_schemes
            ):
                supplement = deferred_candidate_schemes[:4]
                del deferred_candidate_schemes[:4]
                pending_schemes.extend(supplement)
                write_log(
                    f"候选快速初筛补位｜编号={diagnosis_id}｜"
                    f"当前有效={len(all_results)}｜补位={len(supplement)}"
                )
            elif (
                not fallback_added
                and len(all_results) < minimum_result_count
                and fallback_candidate_schemes
            ):
                pending_schemes.extend(fallback_candidate_schemes)
                fallback_added = True
                write_log(
                    f"候选生成补充探索｜编号={diagnosis_id}｜当前有效={len(all_results)}｜"
                    f"新增候补={len(fallback_candidate_schemes)}"
                )
            else:
                break
        scheme_name, scheme = pending_schemes[candidate_index]
        candidate_index += 1
        candidate_started = time.perf_counter()
        protect_original = _is_original_protection_scheme(scheme)
        _raise_if_cancelled(cancel_check)
        evaluation, cache_hit = _evaluate_preview_scheme(
            thumb,
            reference_thumb,
            score_context,
            scheme,
            preview_cache,
            score_cache,
            timing_label=f"{diagnosis_id}/缩略图/{scheme_name}",
            cancel_check=cancel_check,
        )
        _raise_if_cancelled(cancel_check)
        pipeline_elapsed = 0.0 if cache_hit else evaluation.pipeline_elapsed
        scoring_elapsed = 0.0 if cache_hit else evaluation.scoring_elapsed
        pipeline_total += pipeline_elapsed
        scoring_total += scoring_elapsed
        if cache_hit or evaluation.score_reused:
            reused_score_count += 1
        if evaluation.mask is None:
            write_log(f"候选明细｜编号={diagnosis_id}｜方案={scheme_name}｜管线={pipeline_elapsed:.4f}秒｜结果=无掩码")
            continue
        if evaluation.breakdown is None:
            candidate_elapsed = time.perf_counter() - candidate_started
            slowest.append((candidate_elapsed, scheme_name, pipeline_elapsed, scoring_elapsed))
            write_log(
                f"候选明细｜编号={diagnosis_id}｜方案={scheme_name}｜总耗时={candidate_elapsed:.4f}秒｜"
                f"管线={pipeline_elapsed:.4f}秒｜评分=0.0000秒｜得分=跳过｜"
                f"结构保护=淘汰({evaluation.protection_note})｜{evaluation.score_timing}"
            )
            if not protect_original and np.any(evaluation.mask > 0):
                fast_score = _fast_beam_score(evaluation.mask, score_context)
                preview_risky_results.append({
                    "方案名": scheme_name,
                    "方案": scheme,
                    "得分": float(fast_score),
                    "掩码": evaluation.mask,
                    "质量等级": quality_level,
                    "保留原图": False,
                    "_收益不足": False,
                    "评分明细": {
                        "综合得分": float(fast_score),
                        "评分方式": "覆盖预筛快速评分",
                    },
                    "结构复核": _build_structure_review(
                        evaluation.mask,
                        score_context,
                        evaluation.protection_note,
                        "缩略图初筛",
                    ),
                    "_结构风险候选": True,
                })
            continue
        mask_u8 = evaluation.mask
        breakdown = evaluation.breakdown
        score = breakdown.score
        if protect_original:
            original_score = float(score)
        rejected = evaluation.rejected
        protection_note = evaluation.protection_note
        gain_shortfall = False
        if not protect_original and quality_level in ("已足够干净", "低污染") and original_score is not None:
            minimum_gain = 4.0 if quality_level == "已足够干净" else 2.0
            if float(score) < original_score + minimum_gain:
                gain_shortfall = True
        candidate_elapsed = time.perf_counter() - candidate_started
        slowest.append((candidate_elapsed, scheme_name, pipeline_elapsed, scoring_elapsed))
        write_log(
            f"候选明细｜编号={diagnosis_id}｜方案={scheme_name}｜总耗时={candidate_elapsed:.4f}秒｜"
            f"管线={pipeline_elapsed:.4f}秒｜评分={scoring_elapsed:.4f}秒｜得分={score:.1f}｜"
            f"结构保护={'淘汰' if rejected else '通过'}({protection_note})｜"
            f"收益排序={'降级' if gain_shortfall else '正常'}｜"
            f"缓存={'复用' if cache_hit else '新算'}｜{evaluation.score_timing}"
        )
        result = {
            "方案名": scheme_name,
            "方案": scheme,
            "得分": float(score),
            "掩码": mask_u8,
            "质量等级": quality_level,
            "保留原图": protect_original,
            "_收益不足": gain_shortfall,
            "评分明细": breakdown.as_dict(),
            "_评分对象": breakdown,
        }
        if rejected and not protect_original:
            result["结构复核"] = _build_structure_review(
                mask_u8,
                score_context,
                protection_note,
                "缩略图初筛",
            )
            result["_结构风险候选"] = True
            preview_risky_results.append(result)
            continue
        all_results.append(result)
    _rank_results(all_results)
    _rank_risky_results(preview_risky_results)
    write_log(
        f"候选生成评估｜编号={diagnosis_id}｜总耗时={time.perf_counter() - evaluation_started:.4f}秒｜"
        f"管线累计={pipeline_total:.4f}秒｜评分累计={scoring_total:.4f}秒｜有效结果={len(all_results)}｜"
        f"结构风险候选={len(preview_risky_results)}｜相同掩码复用={reused_score_count}"
    )

    dedup_started = time.perf_counter()
    compare_count = 0
    original_result = next((item for item in all_results if item.get("保留原图")), None)
    optimized_results = [item for item in all_results if not item.get("保留原图")]
    primary_optimized: list[dict[str, Any]] = []
    duplicate_fallbacks: list[dict[str, Any]] = []
    for result in optimized_results:
        is_similar = False
        for existing in primary_optimized:
            compare_count += 1
            if _candidate_results_too_similar(result, existing):
                is_similar = True
                break
        if is_similar:
            duplicate_fallbacks.append(result)
        else:
            primary_optimized.append(result)
    verification_queue = ([original_result] if original_result is not None else [])
    verification_queue.extend(primary_optimized)
    verification_queue.extend(duplicate_fallbacks)
    risk_primary: list[dict[str, Any]] = []
    risk_duplicates: list[dict[str, Any]] = []
    for result in preview_risky_results:
        if any(
            _candidate_results_too_similar(result, existing)
            for existing in risk_primary
        ):
            risk_duplicates.append(result)
        else:
            risk_primary.append(result)
    verification_queue.extend(risk_primary)
    verification_queue.extend(risk_duplicates)
    write_log(
        f"候选生成去重｜编号={diagnosis_id}｜耗时={time.perf_counter() - dedup_started:.4f}秒｜"
        f"比较次数={compare_count}｜优先复核={len(primary_optimized)}｜"
        f"重复候补={len(duplicate_fallbacks)}｜风险复核={len(risk_primary)}｜"
        f"风险重复候补={len(risk_duplicates)}｜原图基准={'已保留' if original_result is not None else '无'}"
    )

    full_size_started = time.perf_counter()
    full_size_total = 0.0
    verified: list[dict[str, Any]] = []
    verified_optimized: list[dict[str, Any]] = []
    verified_risky: list[dict[str, Any]] = []
    full_size_score_cache: dict[
        tuple[tuple[int, ...], bytes],
        tuple[scoring.ScoreBreakdown, str],
    ] = {}
    full_size_reused_score_count = 0
    requested_count = max(1, int(limit))
    require_optimized = parent_scheme is None and bool(
        optimized_results or preview_risky_results
    )
    if requested_count == 1 and require_optimized and original_result is not None:
        # 单结果基础寻优最终只会返回寻优结果，原图评分不会参与选择或低收益排序。
        verification_queue = [
            result for result in verification_queue if result is not original_result
        ]
        write_log(
            f"全尺寸复核优化｜编号={diagnosis_id}｜单结果寻优跳过不返回的原图基准"
        )
    for result in verification_queue:
        is_preview_risk = bool(result.get("_结构风险候选", False))
        if is_preview_risk and verified_optimized:
            # 已有原尺寸安全寻优结果时，不再执行或混入风险候选。
            break
        if is_preview_risk and len(verified_risky) >= requested_count:
            break
        if len(verified) >= requested_count and (
            not require_optimized or verified_optimized
        ):
            break
        _raise_if_cancelled(cancel_check)
        item_started = time.perf_counter()
        if result.get("保留原图"):
            full_size_mask = _original_foreground_mask(gray_arr)
        else:
            _, full_size_mask = pipeline.run_pipeline(
                gray_arr, result["方案"], timing_label=f"{diagnosis_id}/全尺寸/{result['方案名']}"
            )
        _raise_if_cancelled(cancel_check)
        item_elapsed = time.perf_counter() - item_started
        full_size_total += item_elapsed
        if full_size_mask is None:
            write_log(f"全尺寸明细｜编号={diagnosis_id}｜方案={result['方案名']}｜结果=无掩码，已淘汰")
            continue
        full_size_mask = (full_size_mask > 0).astype(np.uint8)
        mask_key = _mask_content_key(full_size_mask)
        cached_full_score = full_size_score_cache.get(mask_key)
        if cached_full_score is None:
            full_breakdown, _ = scoring.evaluate_candidate(
                full_size_mask,
                reference_gray,
                context=full_score_context,
                cancel_check=scoring_cancel_check,
            )
            full_size_score_cache[mask_key] = (
                full_breakdown,
                str(result["方案名"]),
            )
        else:
            full_breakdown, first_scheme_name = cached_full_score
            full_size_reused_score_count += 1
            write_log(
                f"全尺寸明细｜编号={diagnosis_id}｜方案={result['方案名']}｜"
                f"原尺寸评分=复用完全相同掩码（首次方案={first_scheme_name}）"
            )
        rejected, protection_note = _reject_structure_damage(
            full_size_mask,
            full_score_context,
            full_breakdown,
        )
        _raise_if_cancelled(cancel_check)
        result["掩码"] = full_size_mask
        result["得分"] = full_breakdown.score
        result["评分明细"] = full_breakdown.as_dict()
        result["_评分对象"] = full_breakdown
        if rejected and not result.get("保留原图"):
            result["结构复核"] = _build_structure_review(
                full_size_mask,
                full_score_context,
                protection_note,
                "原尺寸复核",
            )
            result["_结构风险候选"] = True
            write_log(
                f"全尺寸明细｜编号={diagnosis_id}｜方案={result['方案名']}｜管线耗时={item_elapsed:.4f}秒｜"
                f"原尺寸复核=需人工核对({protection_note})"
            )
            if not any(
                _candidate_results_too_similar(result, existing)
                for existing in verified_risky
            ):
                verified_risky.append(result)
            continue
        result.pop("结构复核", None)
        result.pop("_结构风险候选", None)
        if result.get("保留原图"):
            result["原始灰度"] = np.clip(gray_arr, 0, 255).astype(np.uint8)
        elif any(
            _candidate_results_too_similar(result, existing)
            for existing in verified_optimized
        ):
            write_log(
                f"全尺寸明细｜编号={diagnosis_id}｜方案={result['方案名']}｜"
                "原尺寸复核=通过但与已入围寻优结果重复，继续补位"
            )
            continue
        else:
            verified_optimized.append(result)
        verified.append(result)
        write_log(
            f"全尺寸明细｜编号={diagnosis_id}｜方案={result['方案名']}｜管线耗时={item_elapsed:.4f}秒｜"
            f"原尺寸复核=通过({protection_note})｜重评分={full_breakdown.score:.1f}"
        )

    if not verified_optimized and verified_risky:
        risk_slots = requested_count
        if any(item.get("保留原图") for item in verified) and requested_count > 1:
            risk_slots -= 1
        selected_risks = _select_lowest_risk_results(
            verified_risky,
            max(1, risk_slots),
        )
        verified.extend(selected_risks)
        write_log(
            f"全尺寸结构风险回退｜编号={diagnosis_id}｜"
            f"安全寻优=0｜风险候选={len(verified_risky)}｜采用={len(selected_risks)}"
        )

    full_original = next((item for item in verified if item.get("保留原图")), None)
    if full_original is not None and quality_level in ("已足够干净", "低污染"):
        minimum_gain = 4.0 if quality_level == "已足够干净" else 2.0
        baseline = float(full_original["得分"])
        for item in verified:
            item["_收益不足"] = bool(
                not item.get("保留原图")
                and float(item["得分"]) < baseline + minimum_gain
            )

    _rank_results(verified)
    full_original = next((item for item in verified if item.get("保留原图")), None)
    ranked_optimized = [item for item in verified if not item.get("保留原图")]
    if ranked_optimized and not verified_optimized:
        _rank_risky_results(ranked_optimized)
    if parent_scheme is None and ranked_optimized:
        if requested_count == 1:
            selected = _select_diverse_results(ranked_optimized, 1)
        elif full_original is not None:
            selected = [full_original]
            selected.extend(_select_diverse_results(ranked_optimized, requested_count - 1))
        else:
            selected = _select_diverse_results(ranked_optimized, requested_count)
    else:
        selected = _select_diverse_results(verified, requested_count)
        if full_original is not None and not any(item is full_original for item in selected):
            if len(selected) >= requested_count:
                selected[-1] = full_original
            else:
                selected.append(full_original)
    for result in selected:
        result.pop("_评分对象", None)
        result.pop("_收益不足", None)
        result.pop("_结构风险候选", None)
        result.pop("_结构风险等级", None)
    slowest.sort(reverse=True)
    slow_text = "；".join(f"{name}:{elapsed:.4f}秒(管线{pipe:.4f}/评分{score_time:.4f})" for elapsed, name, pipe, score_time in slowest[:5])
    write_log(
        f"候选生成结束｜编号={diagnosis_id}｜总耗时={time.perf_counter() - started_at:.4f}秒｜"
        f"全尺寸阶段={time.perf_counter() - full_size_started:.4f}秒｜全尺寸管线累计={full_size_total:.4f}秒｜"
        f"全尺寸相同掩码评分复用={full_size_reused_score_count}｜最慢前五={slow_text}"
    )
    return selected


def _original_foreground_mask(gray_arr: np.ndarray) -> np.ndarray:
    """提取未经清噪的原图前景，使原图评分真实计入背景污染。"""
    source = np.clip(gray_arr, 0, 255).astype(np.uint8)
    threshold, _ = cv2.threshold(source, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return (source <= threshold).astype(np.uint8)


def _structure_features(mask: np.ndarray) -> tuple[int, int, float]:
    """兼容旧内部调用，结构指标统一由评分模块计算。"""
    metrics = scoring.compute_structure_metrics(mask)
    return metrics.components, metrics.holes, metrics.stroke_width


def _reject_coverage_damage(
    mask: np.ndarray,
    context: scoring.ScoreContext,
) -> tuple[bool, str]:
    """在骨架分析前淘汰必然无法通过结构保护的候选。"""
    candidate = (mask > 0).astype(np.uint8)
    if not candidate.any():
        return True, "结果为空"
    core_coverage = float((candidate & context.core_mask).sum()) / max(1, context.core_total)
    ref_coverage = float((candidate & context.ref_mask).sum()) / max(1, context.ref_total)
    component_coverage = scoring.minimum_reference_component_coverage(candidate, context)
    if context.core_total and core_coverage < 0.94:
        return True, f"深墨核心仅保留{core_coverage:.1%}"
    if context.ref_total and ref_coverage < 0.82:
        return True, f"主体笔画仅保留{ref_coverage:.1%}"
    if context.ref_component_ids.size and component_coverage < 0.70:
        return True, f"独立笔画最低仅保留{component_coverage:.1%}"
    return False, "覆盖预筛通过"


def _reject_structure_damage(
    mask: np.ndarray,
    context: scoring.ScoreContext,
    breakdown: Optional[scoring.ScoreBreakdown] = None,
) -> tuple[bool, str]:
    """硬性淘汰明显断笔、丢失深墨核心或异常削细的候选。"""
    candidate = (mask > 0).astype(np.uint8)
    coverage_rejected, coverage_note = _reject_coverage_damage(candidate, context)
    if coverage_rejected:
        return True, coverage_note
    if breakdown is None:
        metrics = scoring.compute_structure_metrics(
            candidate,
            context.reference_structure.comparison_min_component_area,
        )
        relation = scoring.compare_structure(candidate, context, metrics)
    else:
        metrics = breakdown.structure
        relation = breakdown.comparison
    components = metrics.components
    stroke = metrics.stroke_width
    if context.ref_components > 0 and components > max(context.ref_components + 3, int(context.ref_components * 1.8)):
        return True, f"连通域由{context.ref_components}增至{components}"
    if context.ref_stroke > 0 and stroke < context.ref_stroke * 0.72:
        return True, f"笔画宽度降至原参考的{stroke / context.ref_stroke:.1%}"
    if context.ref_stroke > 0 and stroke > context.ref_stroke * 1.45:
        return True, f"笔画宽度增至原参考的{stroke / context.ref_stroke:.1%}"
    if context.structure_confidence >= 0.45:
        if relation.skeleton_coverage < 0.90:
            return True, f"参考骨架仅保留{relation.skeleton_coverage:.1%}"
    if context.structure_confidence >= 0.45 and not context.heavy_noise:
        reference = context.reference_structure
        endpoint_limit = max(2, int(math.ceil(reference.endpoint_count * 0.35)))
        if reference.endpoint_count >= 2 and relation.endpoint_retention < 0.65:
            return True, f"参考端点仅匹配{relation.endpoint_retention:.1%}"
        if relation.endpoint_growth > endpoint_limit:
            return True, f"端点异常增加{relation.endpoint_growth}个"
        if reference.holes and relation.hole_retention < 0.999:
            return True, f"有意义孔洞仅保留{relation.hole_retention:.1%}"
        if relation.extra_holes > max(1, reference.holes):
            return True, f"新增有意义孔洞{relation.extra_holes}个"
    return False, "结构完整"


def _build_structure_review(
    mask: np.ndarray,
    context: scoring.ScoreContext,
    reason: str,
    stage: str,
) -> dict[str, Any]:
    """把结构硬门槛失败转换为可追溯的人工复核提示。"""
    coverage_rejected, _coverage_note = _reject_coverage_damage(mask, context)
    return {
        "状态": "需人工核对",
        "阶段": stage,
        "原因": str(reason).strip() or "结构保护未通过",
        "风险等级": 2 if coverage_rejected else 1,
    }


def _scheme_cache_key(scheme: dict[str, Any]) -> str:
    """把方案规范化为单次寻优任务内稳定的缓存键。"""
    return json.dumps(scheme, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _mask_content_key(mask: np.ndarray) -> tuple[tuple[int, ...], bytes]:
    """按二值尺寸和完整像素内容生成无碰撞语义的任务内缓存键。"""
    binary = np.asarray(mask) > 0
    shape = tuple(int(value) for value in binary.shape)
    return shape, np.packbits(binary, axis=None).tobytes()


def _prepare_preview_scheme(
    thumb: np.ndarray,
    context: scoring.ScoreContext,
    scheme: dict[str, Any],
    evaluation_cache: dict[str, _PreviewEvaluation],
    timing_label: Optional[str] = None,
) -> tuple[_PreviewEvaluation, bool]:
    """只运行管线和覆盖预筛，供束搜索前三层快速缩小范围。"""
    scheme_key = _scheme_cache_key(scheme)
    cached = evaluation_cache.get(scheme_key)
    if cached is not None:
        return cached, True

    pipeline_started = time.perf_counter()
    if _is_original_protection_scheme(scheme):
        mask = _original_foreground_mask(thumb)
    else:
        _, mask = pipeline.run_pipeline(thumb, scheme, timing_label=timing_label)
    pipeline_elapsed = time.perf_counter() - pipeline_started
    if mask is None:
        result = _PreviewEvaluation(
            mask=None,
            breakdown=None,
            score_timing="未生成掩码",
            rejected=True,
            protection_note="结果为空",
            pipeline_elapsed=pipeline_elapsed,
            scoring_elapsed=0.0,
        )
    else:
        mask_u8 = (mask > 0).astype(np.uint8)
        rejected, note = _reject_coverage_damage(mask_u8, context)
        result = _PreviewEvaluation(
            mask=mask_u8,
            breakdown=None,
            score_timing="廉价覆盖预筛，未执行完整结构评分",
            rejected=rejected,
            protection_note=note,
            pipeline_elapsed=pipeline_elapsed,
            scoring_elapsed=0.0,
        )
    evaluation_cache[scheme_key] = result
    return result, False


def _fast_beam_score(mask: np.ndarray, context: scoring.ScoreContext) -> float:
    """使用覆盖、污染残留和连通域计算无骨架的束搜索近似分。"""
    candidate = (mask > 0).astype(np.uint8)
    core_coverage = float((candidate & context.core_mask).sum()) / max(1, context.core_total)
    ref_coverage = float((candidate & context.ref_mask).sum()) / max(1, context.ref_total)
    tolerance = cv2.dilate(context.ref_mask, np.ones((5, 5), dtype=np.uint8), iterations=1)
    outside_ratio = float(np.count_nonzero(candidate & (tolerance == 0))) / max(1, int(candidate.sum()))
    cleanliness = 1.0 - min(1.0, outside_ratio * 2.5)
    component_count, _ = cv2.connectedComponents(candidate, connectivity=8)
    extra_components = max(0, component_count - 1 - context.ref_components)
    component_score = 1.0 - min(1.0, extra_components / 40.0)
    foreground_ratio = float(candidate.sum()) / max(1, context.ref_total)
    ratio_score = 1.0 - min(1.0, abs(foreground_ratio - 1.0) / 0.8)
    return (
        min(1.0, core_coverage) * 40.0
        + min(1.0, ref_coverage) * 25.0
        + cleanliness * 20.0
        + component_score * 10.0
        + ratio_score * 5.0
    )


def _evaluate_preview_scheme(
    thumb: np.ndarray,
    reference: np.ndarray,
    context: scoring.ScoreContext,
    scheme: dict[str, Any],
    evaluation_cache: dict[str, _PreviewEvaluation],
    score_cache: dict[tuple[tuple[int, ...], bytes], tuple[scoring.ScoreBreakdown, str]],
    timing_label: Optional[str] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> tuple[_PreviewEvaluation, bool]:
    """执行并缓存候选缩略图；廉价硬门槛失败时跳过骨架评分。"""
    scheme_key = _scheme_cache_key(scheme)
    cached = evaluation_cache.get(scheme_key)
    protect_original = _is_original_protection_scheme(scheme)
    if cached is not None and (
        cached.breakdown is not None or (cached.rejected and not protect_original)
    ):
        return cached, True
    prepared, prepared_from_cache = _prepare_preview_scheme(
        thumb,
        context,
        scheme,
        evaluation_cache,
        timing_label,
    )
    if prepared.mask is None or (prepared.rejected and not protect_original):
        return prepared, prepared_from_cache
    mask_u8 = prepared.mask
    pipeline_elapsed = 0.0 if prepared_from_cache else prepared.pipeline_elapsed

    cache_key = _mask_content_key(mask_u8)
    scoring_started = time.perf_counter()
    cached_score = score_cache.get(cache_key)
    score_reused = cached_score is not None
    if cached_score is None:
        breakdown, score_timing = scoring.evaluate_candidate(
            mask_u8,
            reference,
            context=context,
            cancel_check=_scoring_cancel_check(cancel_check),
        )
        score_cache[cache_key] = (breakdown, score_timing)
    else:
        breakdown, previous_timing = cached_score
        score_timing = f"复用相同掩码评分｜{previous_timing}"
    scoring_elapsed = time.perf_counter() - scoring_started
    rejected, protection_note = _reject_structure_damage(mask_u8, context, breakdown)
    result = _PreviewEvaluation(
        mask=mask_u8,
        breakdown=breakdown,
        score_timing=score_timing,
        rejected=rejected,
        protection_note=protection_note,
        pipeline_elapsed=pipeline_elapsed,
        scoring_elapsed=scoring_elapsed,
        score_reused=score_reused,
    )
    evaluation_cache[scheme_key] = result
    return result, False


def _rank_results(results: list[dict[str, Any]]) -> None:
    """按 Pareto 层级和综合分稳定排序，并写入可追溯层级。"""
    if not results:
        return
    breakdowns = [item["_评分对象"] for item in results]
    ranks = scoring.pareto_front_ranks(breakdowns)
    for result, rank in zip(results, ranks):
        result["Pareto层级"] = rank + 1
        details = result.get("评分明细")
        if isinstance(details, dict):
            details["Pareto层级"] = rank + 1
    results.sort(
        key=lambda item: (
            bool(item.get("_收益不足", False)),
            int(item.get("Pareto层级", 999)),
            -float(item.get("得分", 0.0)),
            str(item.get("方案名", "")),
        )
    )


def _rank_risky_results(results: list[dict[str, Any]]) -> None:
    """风险候选先保护笔画覆盖，再按 Pareto 层级和得分排序。"""
    if not results:
        return
    fully_scored = [
        item
        for item in results
        if isinstance(item.get("_评分对象"), scoring.ScoreBreakdown)
    ]
    _rank_results(fully_scored)
    for result in results:
        review = result.get("结构复核", {})
        if not isinstance(review, dict):
            review = {}
        try:
            risk_level = int(review.get("风险等级", 2))
        except (TypeError, ValueError):
            risk_level = 2
        result["_结构风险等级"] = max(1, risk_level)
    results.sort(
        key=lambda item: (
            int(item.get("_结构风险等级", 2)),
            bool(item.get("_收益不足", False)),
            int(item.get("Pareto层级", 999)),
            -float(item.get("得分", 0.0)),
            str(item.get("方案名", "")),
        )
    )


def _select_lowest_risk_results(
    results: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """只从当前最低风险层选取候选，避免混入覆盖受损结果。"""
    if not results:
        return []
    _rank_risky_results(results)
    lowest_level = int(results[0].get("_结构风险等级", 2))
    lowest_tier = [
        result
        for result in results
        if int(result.get("_结构风险等级", 2)) == lowest_level
    ]
    return _select_diverse_results(lowest_tier, limit)


def _select_diverse_results(results: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """按当前排序保留结果不同的有限候选。"""
    selected: list[dict[str, Any]] = []
    for result in results:
        if any(_candidate_results_too_similar(result, existing) for existing in selected):
            continue
        selected.append(result)
        if len(selected) >= max(1, limit):
            break
    return selected


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


def _stroke_scale_level(result: dict[str, Any]) -> Optional[int]:
    """返回笔画尺度候选的重建级别，其他方案返回 None。"""
    scheme = result.get("方案")
    if not isinstance(scheme, dict):
        return None
    layer = scheme.get("L3")
    if not isinstance(layer, dict) or layer.get("算法") != "笔画尺度核心重建":
        return None
    params = layer.get("参数", {})
    try:
        return max(0, min(2, int(params.get("重建级别", 1))))
    except (TypeError, ValueError):
        return None


def _candidate_results_too_similar(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """不同尺度档按前景相对变化去重，避免小字图被全图占比掩盖。"""
    left_level = _stroke_scale_level(left)
    right_level = _stroke_scale_level(right)
    left_mask = left["掩码"]
    right_mask = right["掩码"]
    if left_level is not None and right_level is not None and left_level != right_level:
        lhs = left_mask > 0
        rhs = right_mask > 0
        difference = int(np.count_nonzero(lhs != rhs))
        foreground = max(1, min(int(lhs.sum()), int(rhs.sum())))
        union = max(1, int(np.count_nonzero(lhs | rhs)))
        relative_difference = difference / foreground
        iou = float(np.count_nonzero(lhs & rhs)) / union
        return relative_difference < 0.010 or iou > 0.995
    return _masks_too_similar(left_mask, right_mask)


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
    mask = arr <= thresh
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
    p10, p90 = np.percentile(arr, (10.0, 90.0))
    robust_span = float(p90 - p10)

    # 使用共享的结构保护域数量估算文字占比。
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S)
    areas = sorted(
        (int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)),
        reverse=True,
    )
    top8_sum = (
        sum(areas[:STRUCTURE_PROTECTION_COMPONENT_LIMIT])
        if areas
        else text_pixels
    )
    est_ratio = top8_sum / max(total, 1)

    # 光照不均检测
    illum = _compute_illumination(arr, h, w)

    # 背景噪点
    noise, speck, speck_cnt = _compute_noise_features(arr, mask, h, w)
    external_analysis = foreground_analysis.analyze_external_pollution(
        mask.astype(np.uint8),
        min_confidence=0.78,
    )
    dense_analysis = stroke_scale_analysis.analyze_stroke_scale(
        arr,
        min_confidence=0.78,
        minimum_noise_components=8,
    )

    low_contrast = bool(
        text_pixels >= max(16, int(total * 0.002))
        and 8.0 <= contrast <= 75.0
        and robust_span <= 130.0
        and 0.005 <= est_ratio <= 0.60
    )

    return {
        "med": med, "avg": avg,
        "contrast": contrast, "est_ratio": est_ratio,
        "robust_span": robust_span, "low_contrast": low_contrast,
        "illum": illum, "noise": noise,
        "speck": speck, "speck_cnt": speck_cnt,
        "external_pollution": external_analysis.applied,
        "external_confidence": external_analysis.confidence,
        "external_removed_ratio": external_analysis.removed_ratio,
        "external_component_count": external_analysis.pollution_component_count,
        "dense_fine_noise": dense_analysis.applicable,
        "dense_confidence": dense_analysis.confidence,
        "dense_stroke_scale": dense_analysis.stroke_scale,
        "dense_noise_scale": dense_analysis.noise_scale,
        "dense_component_count": dense_analysis.noise_component_count,
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
    main_ids = {
        idx for idx, _ in areas[:PRIMARY_CLUSTER_COMPONENT_LIMIT]
    }
    total_text = sum(a for _, a in areas)
    speck_px = sum(a for idx, a in areas if idx not in main_ids)
    speck_cnt = sum(1 for idx, _ in areas if idx not in main_ids)
    speck_ratio = speck_px / max(total_text, 1)
    return noise, speck_ratio, speck_cnt


# ============================================================
# 阶段 B：缩略图
# ============================================================

def _auto_thumb(gray_arr: np.ndarray) -> Tuple[np.ndarray, float]:
    """生成缩略图供快速跑分，并返回线性缩放比。"""
    h, w = gray_arr.shape
    max_dim = max(w, h)
    if max_dim <= 512:
        return gray_arr.copy(), 1.0
    ratio = 512.0 / max_dim
    nw = max(1, int(w * ratio))
    nh = max(1, int(h * ratio))
    arr_u8 = np.clip(gray_arr, 0, 255).astype(np.uint8)
    thumb = cv2.resize(arr_u8, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32)
    return thumb, ratio


# ============================================================
# 阶段 C：生成候选方案
# ============================================================

def _classify_quality(features: dict[str, Any]) -> str:
    """综合多个污染指标判定处理强度，避免单项波动触发激进路线。"""
    speck = float(features.get("speck", 0.0))
    speck_count = int(features.get("speck_cnt", 0))
    noise = float(features.get("noise", 0.0))
    illumination = float(features.get("illum", 0.0))
    external_pollution = bool(features.get("external_pollution", False))
    external_removed_ratio = float(features.get("external_removed_ratio", 0.0))
    external_component_count = int(features.get("external_component_count", 0))
    dense_fine_noise = bool(features.get("dense_fine_noise", False))
    if dense_fine_noise:
        return "重度污染"
    heavy_signals = sum((speck >= 0.04, speck_count >= 40, noise >= 5.0, illumination >= 0.10))
    if heavy_signals >= 2 or speck >= 0.12 or speck_count >= 120:
        return "重度污染"
    if external_pollution and (external_removed_ratio >= 0.12 or external_component_count >= 8):
        return "重度污染"
    if external_pollution:
        return "中度污染"
    if speck >= 0.04 or speck_count >= 40 or noise >= 5.0 or illumination >= 0.06:
        return "中度污染"
    if speck >= 0.01 or speck_count >= 8 or noise >= 2.5 or illumination >= 0.035:
        return "低污染"
    return "已足够干净"


def _is_original_protection_scheme(scheme: dict[str, Any]) -> bool:
    """判断方案是否要求完整保留原始灰度字形。"""
    return bool(scheme.get("保护原图", False))


def _auto_build_candidates(
    features: dict[str, Any], profile: ScaleProfile, quality_level: str
) -> list[tuple[str, dict[str, Any]]]:
    """根据原图质量分级生成候选，处理强度只随确切污染逐级增加。"""
    est = max(0.04, min(0.4, float(features["est_ratio"])))
    filter_params = {
        "min_area": profile.min_area,
        "连通类型": 8,
        "相对模式": True,
        "相对比例": 0.002,
    }
    shape_params = {**filter_params, "仅孤立": True, "最大长宽比": 3.0, "最小凸包比": 0.7, "最小实体比": 0.4}
    external_params = {
        "最小置信度": 0.78,
        "最大污染面积比": 0.20,
        "最小间隔笔宽": 1.25,
        "外围边距比例": 0.18,
        "清理孤立小点": True,
        "min_area": profile.min_area,
    }
    cands: list[tuple[str, dict[str, Any]]] = [
        ("原图保护·不去杂", {"保护原图": True, "质量等级": quality_level, "预处理": {}}),
        ("低污染·保守Otsu", {"预处理": {}, "L3": {"算法": "Otsu", "参数": {"偏移": 0}}}),
    ]

    dense_fine_noise = bool(features.get("dense_fine_noise", False))
    if dense_fine_noise:
        dense_candidates = [
            ("密集细噪·保形重建", {"预处理": {}, "L3": {"算法": "笔画尺度核心重建", "参数": {
                "重建级别": 0, "最小置信度": 0.78, "最少细噪域": 8,
            }}}),
            ("密集细噪·平衡重建", {
                "预处理": {},
                "L3": {"算法": "笔画尺度核心重建", "参数": {
                    "重建级别": 1, "最小置信度": 0.78, "最少细噪域": 8,
                }},
                "L5": {"算法": "主体外污染过滤", "参数": external_params},
            }),
            ("密集细噪·强清理重建", {
                "预处理": {},
                "L3": {"算法": "笔画尺度核心重建", "参数": {
                    "重建级别": 2, "最小置信度": 0.78, "最少细噪域": 8,
                }},
                "L5": {"算法": "主体外污染过滤", "参数": external_params},
            }),
        ]
        cands[2:2] = dense_candidates

    if features.get("low_contrast", False):
        correction = {
            "背景核大小": profile.background_kernel,
            "CLAHE限制": 1.4,
            "网格数": 8,
        }
        cands.extend([
            ("低对比·受限CLAHE", {
                "预处理": {},
                "L2": {"算法": "低对比背景校正", "参数": correction},
                "L3": {"算法": "Otsu", "参数": {"偏移": 0}},
                "L5": {"算法": "面积+形状过滤", "参数": shape_params},
            }),
            ("低对比·背景校正Triangle", {
                "预处理": {},
                "L2": {"算法": "低对比背景校正", "参数": {**correction, "CLAHE限制": 0.0}},
                "L3": {"算法": "Triangle", "参数": {}},
                "L5": {"算法": "面积+形状过滤", "参数": shape_params},
            }),
            ("低对比·Phansalkar", {
                "预处理": {},
                "L3": {"算法": "Phansalkar", "参数": {
                    "窗口": profile.local_window, "k": 0.25, "R": 128, "p": 2.0, "q": 10.0,
                }},
                "L5": {"算法": "面积+形状过滤", "参数": shape_params},
            }),
        ])

    def finish(items: list[tuple[str, dict[str, Any]]]) -> list[tuple[str, dict[str, Any]]]:
        return [(name, _mark_adaptive_scheme(scheme, profile)) for name, scheme in items[:8]]

    if quality_level == "已足够干净":
        return finish(cands)

    cands.insert(5 if dense_fine_noise else 2, ("主体外污染·保形清理", {
        "预处理": {},
        "L3": {"算法": "Otsu", "参数": {"偏移": 0}},
        "L5": {"算法": "主体外污染过滤", "参数": external_params},
    }))
    cands.append(("干净·Otsu", {"预处理": {}, "L3": {"算法": "Otsu", "参数": {"偏移": 0}}, "L5": {"算法": "面积+形状过滤", "参数": shape_params}}))
    if quality_level == "低污染":
        return finish(cands)

    cands.append(("轻量·percentile", {"预处理": {}, "L3": {"算法": "percentile硬切", "参数": {"暗色比例": min(0.38, est * 1.2)}}, "L5": {"算法": "面积过滤", "参数": filter_params}}))
    if features["illum"] >= 0.06:
        cands.extend([
            ("光照不均·背景归一", {"预处理": {}, "L2": {"算法": "形态学背景归一", "参数": {"核大小": profile.background_kernel}}, "L3": {"算法": "Otsu", "参数": {"偏移": 0}}, "L5": {"算法": "面积过滤", "参数": filter_params}}),
            ("退化文档·Wolf", {"预处理": {}, "L3": {"算法": "Wolf-Jolion", "参数": {"窗口": profile.local_window, "k": 0.35}}, "L5": {"算法": "面积+形状过滤", "参数": shape_params}}),
        ])
    else:
        cands.append(("均衡·Sauvola", {"预处理": {}, "L3": {"算法": "Sauvola", "参数": {"窗口": profile.local_window, "k": 0.2, "R": 128}}, "L5": {"算法": "面积过滤", "参数": filter_params}}))
    if features.get("speck", 0) >= 0.01 or features.get("speck_cnt", 0) >= 8:
        cands.append(("散点拓片·灰度黑帽", {"预处理": {}, "L2": {"算法": "灰度黑帽增强", "参数": {"核大小": profile.blackhat_kernel, "强度": 0.8}}, "L3": {"算法": "Otsu", "参数": {"偏移": 0}}, "L5": {"算法": "面积+形状过滤", "参数": shape_params}}))
    if quality_level == "重度污染":
        cands.insert(6 if dense_fine_noise else 3, ("主体外污染·重噪清理", {
            "预处理": {},
            "L1": {"算法": "中值滤波", "参数": {"核大小": profile.noise_kernel}},
            "L3": {"算法": "Otsu", "参数": {"偏移": 0}},
            "L4": {"算法": "开运算", "参数": {
                "半径": profile.morph_radius,
                "迭代": 1,
                "核形状": 1,
            }},
            "L5": {"算法": "主体外污染过滤", "参数": external_params},
        }))
        cands.append(("重噪·双阈值种子重建", {"预处理": {}, "L1": {"算法": "中值滤波", "参数": {"核大小": profile.noise_kernel}}, "L3": {"算法": "双阈值种子重建", "参数": {"核心偏移": -28, "生长偏移": 18}}, "L5": {"算法": "面积+形状过滤", "参数": shape_params}}))
    if features.get("speck_cnt", 0) >= 8:
        cands.append(("边界污染·主体保护", {"预处理": {}, "L3": {"算法": "Otsu", "参数": {"偏移": 0}}, "L5": {"算法": "边界污染过滤", "参数": {"最大宽度比例": 0.18, "最大高度比例": 0.18, "保护主体数": 8}}}))
    return finish(cands)


# ============================================================
# 阶段 D：缩略图跑分
# ============================================================

def _evaluate_candidates(
    thumb: np.ndarray,
    candidates: list[tuple[str, dict[str, Any]]],
    diagnosis_id: str = "",
    stage_name: str = "寻优",
    reference_gray_arr: Optional[np.ndarray] = None,
) -> Tuple[str, dict[str, Any], float]:
    """在缩略图上按统一硬保护和 Pareto 层级返回最佳候选。"""
    reference = thumb if reference_gray_arr is None else reference_gray_arr
    context = scoring.build_score_context(reference)
    evaluated: list[dict[str, Any]] = []
    for name, scheme in candidates:
        started_at = time.perf_counter()
        pipeline_started = time.perf_counter()
        protect_original = _is_original_protection_scheme(scheme)
        if protect_original:
            mask = _original_foreground_mask(thumb)
        else:
            _, mask = pipeline.run_pipeline(thumb, scheme, timing_label=f"{diagnosis_id}/{stage_name}/{name}")
        pipeline_elapsed = time.perf_counter() - pipeline_started
        if mask is not None:
            mask_u8 = (mask > 0).astype(np.uint8)
            scoring_started = time.perf_counter()
            breakdown, score_timing = scoring.evaluate_candidate(mask_u8, reference, context=context)
            rejected, protection_note = _reject_structure_damage(mask_u8, context, breakdown)
            scoring_elapsed = time.perf_counter() - scoring_started
            write_log(
                f"寻优候选明细｜编号={diagnosis_id}｜阶段={stage_name}｜方案={name}｜"
                f"总耗时={time.perf_counter() - started_at:.4f}秒｜管线={pipeline_elapsed:.4f}秒｜"
                f"评分={scoring_elapsed:.4f}秒｜得分={breakdown.score:.1f}｜"
                f"结构保护={'淘汰' if rejected else '通过'}({protection_note})｜{score_timing}"
            )
            if rejected and not protect_original:
                continue
            evaluated.append({
                "方案名": name,
                "方案": scheme,
                "得分": breakdown.score,
                "掩码": mask_u8,
                "评分明细": breakdown.as_dict(),
                "_评分对象": breakdown,
            })
    _rank_results(evaluated)
    if not evaluated:
        return "", {}, -1.0
    best = evaluated[0]
    return str(best["方案名"]), dict(best["方案"]), float(best["得分"])


# ============================================================
# 阶段 E：重噪束搜索与梯度派生加试
# ============================================================

def _is_heavy_noise(features: dict[str, float]) -> bool:
    """判断是否需要启用强去杂搜索。"""
    return bool(
        features.get("dense_fine_noise", False)
        or features.get("speck", 0.0) >= 0.04
        or features.get("speck_cnt", 0.0) >= 40
        or features.get("noise", 0.0) >= 5.0
    )


def _clone_scheme(scheme: dict[str, Any]) -> dict[str, Any]:
    """深拷贝方案，避免派生候选共享嵌套参数。"""
    return copy.deepcopy(scheme)


def _prefilter_candidate_schemes(
    thumb: np.ndarray,
    candidates: list[tuple[str, dict[str, Any]]],
    context: scoring.ScoreContext,
    evaluation_cache: dict[str, _PreviewEvaluation],
    keep: int,
    *,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> list[tuple[str, dict[str, Any]]]:
    """用覆盖、污染残留和连通域快速初筛，再把少量方案交给骨架评分。"""

    if len(candidates) <= keep:
        return list(candidates)
    protected: list[tuple[str, dict[str, Any]]] = []
    ranked: list[tuple[float, str, dict[str, Any], np.ndarray]] = []
    soft_ranked: list[tuple[float, str, dict[str, Any], np.ndarray]] = []
    for name, scheme in candidates:
        _raise_if_cancelled(cancel_check)
        evaluation, _cache_hit = _prepare_preview_scheme(
            thumb,
            context,
            scheme,
            evaluation_cache,
        )
        if _is_original_protection_scheme(scheme):
            protected.append((name, scheme))
            continue
        if evaluation.mask is None:
            continue
        score = (
            float(evaluation.breakdown.score)
            if evaluation.breakdown is not None
            else _fast_beam_score(evaluation.mask, context)
        )
        item = (score, name, scheme, evaluation.mask)
        if evaluation.rejected:
            soft_ranked.append(item)
        else:
            ranked.append(item)

    ranked.sort(key=lambda item: (-item[0], item[1]))
    soft_ranked.sort(key=lambda item: (-item[0], item[1]))
    pool = ranked or (soft_ranked if context.heavy_noise else [])
    selected = list(protected)
    selected_masks: list[np.ndarray] = []
    family_counts: dict[str, int] = {}
    for _score, name, scheme, mask in pool:
        if len(selected) >= keep:
            break
        family = name.split("·", 1)[0]
        if family_counts.get(family, 0) >= 2:
            continue
        if any(_masks_too_similar(mask, existing) for existing in selected_masks):
            continue
        selected.append((name, scheme))
        selected_masks.append(mask)
        family_counts[family] = family_counts.get(family, 0) + 1

    if len(selected) < min(keep, len(candidates)):
        selected_keys = {_scheme_cache_key(scheme) for _name, scheme in selected}
        for name, scheme in candidates:
            if len(selected) >= keep:
                break
            key = _scheme_cache_key(scheme)
            if key in selected_keys:
                continue
            evaluation = evaluation_cache.get(key)
            if evaluation is None or evaluation.mask is None:
                continue
            selected.append((name, scheme))
            selected_keys.add(key)
    selected_keys = {_scheme_cache_key(scheme) for _name, scheme in selected}
    for name, scheme in candidates:
        key = _scheme_cache_key(scheme)
        if key in selected_keys:
            continue
        selected.append((name, scheme))
        selected_keys.add(key)
    return selected


def _beam_rank(
    thumb: np.ndarray,
    candidates: list[tuple[str, dict[str, Any]]],
    keep: int,
    reference_gray_arr: Optional[np.ndarray] = None,
    score_context: Optional[scoring.ScoreContext] = None,
    evaluation_cache: Optional[dict[str, _PreviewEvaluation]] = None,
    score_cache: Optional[
        dict[tuple[tuple[int, ...], bytes], tuple[scoring.ScoreBreakdown, str]]
    ] = None,
    family_limit: Optional[int] = None,
    full_structure: bool = True,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> list[tuple[str, dict[str, Any]]]:
    """评分并保留少量方案；中间层可跳过骨架，最终层执行完整保护。"""
    reference = thumb if reference_gray_arr is None else reference_gray_arr
    context = score_context or scoring.build_score_context(
        reference,
        cancel_check=_scoring_cancel_check(cancel_check),
    )
    evaluations = evaluation_cache if evaluation_cache is not None else {}
    scores = score_cache if score_cache is not None else {}
    ranked: list[dict[str, Any]] = []
    soft_ranked: list[dict[str, Any]] = []
    for name, scheme in candidates:
        _raise_if_cancelled(cancel_check)
        if full_structure:
            evaluation, _ = _evaluate_preview_scheme(
                thumb,
                reference,
                context,
                scheme,
                evaluations,
                scores,
                cancel_check=cancel_check,
            )
            if evaluation.mask is None or evaluation.breakdown is None:
                _raise_if_cancelled(cancel_check)
                continue
            item = {
                "方案名": name,
                "方案": scheme,
                "得分": evaluation.breakdown.score,
                "掩码": evaluation.mask,
                "评分明细": evaluation.breakdown.as_dict(),
                "_评分对象": evaluation.breakdown,
            }
            if evaluation.rejected:
                soft_ranked.append(item)
            else:
                ranked.append(item)
        else:
            evaluation, _ = _prepare_preview_scheme(
                thumb,
                context,
                scheme,
                evaluations,
            )
            if evaluation.mask is None:
                _raise_if_cancelled(cancel_check)
                continue
            item = {
                "方案名": name,
                "方案": scheme,
                "得分": _fast_beam_score(evaluation.mask, context),
                "掩码": evaluation.mask,
            }
            if evaluation.rejected:
                soft_ranked.append(item)
            else:
                ranked.append(item)
        _raise_if_cancelled(cancel_check)

    fallback = not ranked and context.heavy_noise
    pool = soft_ranked if fallback else ranked
    if full_structure:
        _rank_results(pool)
    else:
        pool.sort(key=lambda item: (-float(item["得分"]), str(item["方案名"])))
    selected: list[tuple[str, dict[str, Any]]] = []
    selected_masks: list[np.ndarray] = []
    family_counts: dict[str, int] = {}
    effective_keep = min(2, keep) if fallback else keep
    for result in pool:
        name = str(result["方案名"])
        scheme = result["方案"]
        mask = result["掩码"]
        if any(_masks_too_similar(mask, existing) for existing in selected_masks):
            continue
        family = name.split("·", 1)[0]
        if family_limit is not None and family_counts.get(family, 0) >= family_limit:
            continue
        selected.append((name, scheme))
        selected_masks.append(mask)
        family_counts[family] = family_counts.get(family, 0) + 1
        if len(selected) >= effective_keep:
            break
    return selected


def _build_heavy_noise_beam_candidates(
    thumb: np.ndarray,
    profile: ScaleProfile,
    diagnosis_id: str,
    reference_gray_arr: Optional[np.ndarray] = None,
    score_context: Optional[scoring.ScoreContext] = None,
    evaluation_cache: Optional[dict[str, _PreviewEvaluation]] = None,
    score_cache: Optional[
        dict[tuple[tuple[int, ...], bytes], tuple[scoring.ScoreBreakdown, str]]
    ] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> list[tuple[str, dict[str, Any]]]:
    """分阶段搜索中值、阈值、开运算和主体过滤，避免一次穷举数万组合。"""
    reference = thumb if reference_gray_arr is None else reference_gray_arr
    context = score_context or scoring.build_score_context(
        reference,
        cancel_check=_scoring_cancel_check(cancel_check),
    )
    evaluations = evaluation_cache if evaluation_cache is not None else {}
    scores = score_cache if score_cache is not None else {}
    stage1: list[tuple[str, dict[str, Any]]] = []
    kernels = sorted({_odd_value(profile.noise_kernel * factor, 3, 11) for factor in (0.6, 0.8, 1.0, 1.4, 1.8)})
    for kernel in kernels:
        scheme = _mark_adaptive_scheme({
            "预处理": {}, "L1": {"算法": "中值滤波", "参数": {"核大小": kernel}},
            "L3": {"算法": "Otsu", "参数": {"偏移": -12}},
        }, profile)
        stage1.append((f"强去杂·中值{kernel}", scheme))
    beam = _beam_rank(
        thumb,
        stage1,
        4,
        reference_gray_arr,
        context,
        evaluations,
        scores,
        full_structure=False,
        cancel_check=cancel_check,
    )

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
        for level, level_name in ((0, "保形"), (1, "平衡"), (2, "强清理")):
            scheme = _clone_scheme(parent)
            scheme["L3"] = {"算法": "笔画尺度核心重建", "参数": {
                "重建级别": level,
                "最小置信度": 0.78,
                "最少细噪域": 8,
            }}
            stage2.append((f"{parent_name}·尺度{level_name}", scheme))
    beam = _beam_rank(
        thumb,
        stage2,
        5,
        reference_gray_arr,
        context,
        evaluations,
        scores,
        full_structure=False,
        cancel_check=cancel_check,
    )

    stage3: list[tuple[str, dict[str, Any]]] = []
    for parent_name, parent in beam:
        stage3.append((parent_name, parent))
        for shape_name, shape in (("椭圆", 1), ("矩形", 0), ("十字", 2)):
            radii = sorted({
                max(1, min(6, profile.morph_radius + delta))
                for delta in (-1, 0, 1, 2)
            })
            for radius in radii:
                scheme = _clone_scheme(parent)
                scheme["L4"] = {"算法": "开运算", "参数": {"半径": radius, "迭代": 1, "核形状": shape}}
                stage3.append((f"{parent_name}·{shape_name}开运算{radius}", scheme))
    beam = _beam_rank(
        thumb,
        stage3,
        5,
        reference_gray_arr,
        context,
        evaluations,
        scores,
        full_structure=False,
        cancel_check=cancel_check,
    )

    stage4: list[tuple[str, dict[str, Any]]] = []
    for parent_name, parent in beam:
        stage4.append((parent_name, parent))
        external_scheme = _clone_scheme(parent)
        external_scheme["L5"] = {"算法": "主体外污染过滤", "参数": {
            "最小置信度": 0.78,
            "最大污染面积比": 0.20,
            "最小间隔笔宽": 1.25,
            "外围边距比例": 0.18,
            "清理孤立小点": True,
            "min_area": profile.min_area,
        }}
        stage4.append((f"{parent_name}·主体外污染过滤", external_scheme))
        for factor in (0.5, 1.0, 2.0, 3.0):
            min_area = max(4, int(round(profile.min_area * factor)))
            scheme = _clone_scheme(parent)
            scheme["L5"] = {"算法": "面积+形状过滤", "参数": {
                "min_area": min_area, "相对模式": True, "相对比例": 0.0015,
                "仅孤立": True, "最大长宽比": 4.5, "最小凸包比": 0.35, "最小实体比": 0.25,
            }}
            stage4.append((f"{parent_name}·主体过滤{min_area}", scheme))
    final = _beam_rank(
        thumb,
        stage4,
        6,
        reference_gray_arr,
        context,
        evaluations,
        scores,
        cancel_check=cancel_check,
    )
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
        if algo == "笔画尺度核心重建" and len(derived) < limit:
            level = int(params.get("重建级别", 1))
            confidence = float(params.get("最小置信度", 0.78))
            minimum_count = int(params.get("最少细噪域", 8))
            for next_level, next_confidence, next_count in (
                (0, confidence, minimum_count),
                (1, confidence, minimum_count),
                (2, confidence, minimum_count),
                (level, confidence - 0.06, minimum_count),
                (level, confidence + 0.08, minimum_count + 4),
            ):
                p = dict(params)
                p["重建级别"] = next_level
                p["最小置信度"] = round(max(0.60, min(0.98, next_confidence)), 2)
                p["最少细噪域"] = max(4, min(200, next_count))
                if p != params:
                    variants.append(p)
        elif algo == "percentile硬切" and len(derived) < limit:
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
        elif algo == "双阈值种子重建" and len(derived) < limit:
            seed_offset = int(params.get("核心偏移", -28))
            support_offset = int(params.get("生长偏移", 18))
            for seed_delta, support_delta in ((-10, -6), (-6, 8), (8, -6), (8, 8)):
                p = dict(params)
                p["核心偏移"] = max(-80, min(-4, seed_offset + seed_delta))
                p["生长偏移"] = max(0, min(60, support_offset + support_delta))
                if p != params:
                    variants.append(p)
        elif algo in ("Sauvola", "Wolf-Jolion", "Phansalkar") and len(derived) < limit:
            k = params.get("k", 0.2)
            w = params.get("窗口", 25)
            k_values = (0.15, 0.25, 0.35, 0.45) if algo == "Wolf-Jolion" else (0.1, 0.15, 0.25, 0.3)
            for nk in k_values:
                if abs(nk - k) < 0.001:
                    continue
                for nw in sorted({_odd_value(w * factor, 7, 101) for factor in (0.7, 1.0, 1.3)}):
                    if abs(nw - w) < 2:
                        continue
                    p = dict(params)
                    p["k"] = nk
                    p["窗口"] = nw
                    variants.append(p)
        elif algo in ("黑帽扣除", "灰度黑帽增强") and len(derived) < limit:
            ks = params.get("核大小", 11)
            for nks in sorted({_odd_value(ks * factor, 3, 31) for factor in (0.7, 1.0, 1.3)}):
                if nks != ks:
                    p = dict(params)
                    p["核大小"] = nks
                    variants.append(p)
        elif algo == "低对比背景校正" and len(derived) < limit:
            background_kernel = params.get("背景核大小", 51)
            clip_limit = float(params.get("CLAHE限制", 1.4))
            for next_clip in (0.0, 1.0, 1.4, 1.8):
                for next_kernel in sorted({
                    _odd_value(background_kernel * factor, 15, 201) for factor in (0.8, 1.0, 1.2)
                }):
                    if next_kernel == background_kernel and abs(next_clip - clip_limit) < 0.001:
                        continue
                    p = dict(params)
                    p["背景核大小"] = next_kernel
                    p["CLAHE限制"] = next_clip
                    variants.append(p)
        elif algo == "中值滤波" and len(derived) < limit:
            ks = params.get("核大小", 3)
            for nks in sorted({_odd_value(ks * factor, 3, 11) for factor in (0.6, 0.8, 1.0, 1.4, 1.8)}):
                if nks != ks:
                    p = dict(params)
                    p["核大小"] = nks
                    variants.append(p)
        elif algo == "开运算" and len(derived) < limit:
            radius = params.get("半径", 1)
            shape = params.get("核形状", 1)
            for next_radius in sorted({max(1, min(6, radius + delta)) for delta in (-1, 0, 1, 2)}):
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
        elif algo == "主体外污染过滤" and len(derived) < limit:
            confidence = float(params.get("最小置信度", 0.78))
            area_ratio = float(params.get("最大污染面积比", 0.20))
            gap_ratio = float(params.get("最小间隔笔宽", 1.25))
            for next_confidence, next_area, next_gap in (
                (confidence + 0.08, area_ratio, gap_ratio),
                (confidence - 0.06, area_ratio, gap_ratio),
                (confidence, area_ratio - 0.05, gap_ratio + 0.3),
                (confidence, area_ratio + 0.05, gap_ratio - 0.3),
            ):
                p = dict(params)
                p["最小置信度"] = round(max(0.60, min(0.98, next_confidence)), 2)
                p["最大污染面积比"] = round(max(0.05, min(0.35, next_area)), 2)
                p["最小间隔笔宽"] = round(max(0.8, min(4.0, next_gap)), 2)
                if p != params:
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
        scale_meta = base_scheme.get("自适应尺度", {})
        if isinstance(scale_meta, dict):
            reference_stroke = float(scale_meta.get("参考笔画宽度", 6.0))
            adaptive_noise = _odd_value(reference_stroke * 0.45, 3, 11)
            adaptive_radius = max(1, min(6, int(round(reference_stroke * 0.12))))
        else:
            adaptive_noise = 3
            adaptive_radius = 1
        if base_scheme.get("L5", {}).get("算法") != "主体外污染过滤":
            structural_variants.append(("改主体外污染过滤", "L5", {
                "算法": "主体外污染过滤",
                "参数": {
                    "最小置信度": 0.78,
                    "最大污染面积比": 0.20,
                    "最小间隔笔宽": 1.25,
                    "外围边距比例": 0.18,
                    "清理孤立小点": True,
                    "min_area": max(10, int(round(reference_stroke * reference_stroke * 0.35))),
                },
            }))
        if not base_scheme.get("L1"):
            structural_variants.extend([
                (f"新增中值{adaptive_noise}", "L1", {"算法": "中值滤波", "参数": {"核大小": adaptive_noise}}),
                (f"新增中值{_odd_value(adaptive_noise * 1.5, 3, 11)}", "L1", {"算法": "中值滤波", "参数": {"核大小": _odd_value(adaptive_noise * 1.5, 3, 11)}}),
            ])
        if base_scheme.get("L3", {}).get("算法") != "笔画尺度核心重建":
            structural_variants.append(("改笔画尺度核心重建", "L3", {
                "算法": "笔画尺度核心重建",
                "参数": {"重建级别": 1, "最小置信度": 0.78, "最少细噪域": 8},
            }))
        if base_scheme.get("L3", {}).get("算法") != "固定阈值":
            for threshold in (140, 160, 180, 200):
                structural_variants.append((f"改固定阈值{threshold}", "L3", {"算法": "固定阈值", "参数": {"阈值": threshold}}))
        if base_scheme.get("L3", {}).get("算法") != "Otsu":
            structural_variants.append(("改Otsu", "L3", {"算法": "Otsu", "参数": {"偏移": -12}}))
        if base_scheme.get("L4", {}).get("算法") != "开运算":
            structural_variants.extend([
                (f"改椭圆开运算{radius}", "L4", {"算法": "开运算", "参数": {"半径": radius, "迭代": 1, "核形状": 1}})
                for radius in sorted({adaptive_radius, min(6, adaptive_radius + 1), min(6, adaptive_radius + 2)})
            ])
        if not base_scheme.get("L5"):
            scale_meta = base_scheme.get("自适应尺度", {})
            adaptive_min_area = int(scale_meta.get("最小连通域面积", 40)) if isinstance(scale_meta, dict) else 40
            structural_variants.append(("新增主体过滤", "L5", {"算法": "面积+形状过滤", "参数": {
                "min_area": adaptive_min_area, "相对模式": True, "相对比例": 0.0015, "仅孤立": True,
                "最大长宽比": 4.5, "最小凸包比": 0.35, "最小实体比": 0.25,
            }}))
        for change_name, layer_key, layer in structural_variants:
            if len(derived) >= limit:
                break
            new_scheme = _clone_scheme(base_scheme)
            new_scheme[layer_key] = layer
            derived.append((f"{base_name}·{change_name}", new_scheme))

    return derived
