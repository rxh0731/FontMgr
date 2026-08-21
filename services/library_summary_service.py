"""首页字库摘要的统一计算服务。"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any

import config
from services.glyph_service import GlyphService
from services.workflow_status_service import (
    PHASE_COORDINATION,
    PHASE_OPTIMIZATION,
    PHASE_REVIEW,
    project_all_stage_statuses,
)


_SUMMARY_CACHE_ATTRIBUTE = "_fonteditor_library_summary_cache"


def _coordination_fingerprint(summary: Mapping[str, Any]) -> tuple[Any, ...]:
    """只包含会改变单字协调投影的全局契约字段。"""

    return (
        summary.get("墨色基准"),
        bool(summary.get("墨色统一启用", True)),
        summary.get("墨色方法"),
        summary.get("墨色方法版本"),
    )


def _variant_summary_fingerprint(detail: Mapping[str, Any]) -> tuple[Any, ...]:
    parameters = detail.get("整体协调参数")
    safe_parameters = parameters if isinstance(parameters, Mapping) else {}
    ink = safe_parameters.get("墨色协调")
    safe_ink = ink if isinstance(ink, Mapping) else {}
    return (
        detail.get("状态"),
        detail.get("原始文件"),
        detail.get("中间文件"),
        detail.get("审核文件"),
        detail.get("成品文件"),
        safe_ink.get("启用"),
        safe_ink.get("模式"),
        safe_ink.get("方法"),
        safe_ink.get("方法版本"),
        safe_ink.get("基准"),
        safe_ink.get("保存后复测"),
        safe_ink.get("保存后墨色"),
        safe_ink.get("是否达标"),
        safe_ink.get("人工接受例外"),
    )


def _summary_contribution(
    detail: Mapping[str, Any],
    coordination_summary: Mapping[str, Any],
    finished_dir: str,
    *,
    verify_files: bool,
) -> tuple[int, ...]:
    statuses = project_all_stage_statuses(
        detail,
        coordination_summary,
        finished_dir,
        verify_files=verify_files,
    )
    optimization = statuses[PHASE_OPTIMIZATION]
    review = statuses[PHASE_REVIEW]
    coordination = statuses[PHASE_COORDINATION]
    return (
        int(optimization.completed),
        int(optimization.admitted and not optimization.completed),
        int(review.completed),
        int(review.admitted and not review.completed),
        int(review.admitted),
        int(coordination.completed),
        int(coordination.admitted and not coordination.completed),
        int(coordination.admitted),
        int(coordination.has_valid_finished),
    )


def build_library_summary(
    library_name: str,
    library_path: str,
    details: Mapping[str, Any] | None,
    groups: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None,
    coordination_summary: Mapping[str, Any] | None,
    *,
    verify_files: bool,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """按统一阶段规则生成一个字库的首页摘要。"""

    safe_details = details if isinstance(details, Mapping) else {}
    safe_groups = groups if isinstance(groups, Mapping) else {}
    safe_metadata = metadata if isinstance(metadata, Mapping) else {}
    safe_coordination = (
        coordination_summary
        if isinstance(coordination_summary, Mapping)
        else {}
    )
    glyph_total = len(safe_details)
    if progress_callback is not None:
        progress_callback(0, glyph_total)
    finished_dir = os.path.join(library_path, config.DIR_FINISHED_FILES)
    totals = [0] * 9
    for glyph_current, detail in enumerate(safe_details.values(), start=1):
        source = detail if isinstance(detail, Mapping) else None
        contribution = _summary_contribution(
            source or {},
            safe_coordination,
            finished_dir,
            verify_files=verify_files,
        )
        totals = [left + right for left, right in zip(totals, contribution)]
        if progress_callback is not None:
            progress_callback(glyph_current, glyph_total)
    if glyph_total == 0 and progress_callback is not None:
        progress_callback(0, 0)

    (
        optimized,
        pending_optimization,
        reviewed,
        pending_review,
        review_admitted,
        coordinated,
        pending_coordination,
        coordination_admitted,
        export_ready,
    ) = totals
    summary = {
        "name": library_name,
        "path": library_path,
        "characters": len(safe_groups),
        "variants": glyph_total,
        "imported": glyph_total,
        "optimized": optimized,
        "pending_optimization": pending_optimization,
        "pending_review": pending_review,
        "review_admitted": review_admitted,
        "pending_coordination": pending_coordination,
        "reviewed": reviewed,
        "coordination_admitted": coordination_admitted,
        "finished": coordinated,
        "completed": coordinated,
        "coordinated": coordinated,
        "export_ready": export_ready,
        "metadata": dict(safe_metadata),
    }
    return summary


def summarize_glyph_service(glyph_service: GlyphService) -> dict[str, Any]:
    """快速汇总本程序已经成功提交的字库内存状态。"""

    details = glyph_service.get_variants()
    groups = glyph_service.get_glyph_groups()
    metadata = glyph_service.get_metadata()
    coordination = glyph_service.get_coordination_summary()
    finished_dir = os.path.join(glyph_service.ziku_dir, config.DIR_FINISHED_FILES)
    coordination_key = _coordination_fingerprint(coordination)
    cache = getattr(glyph_service, _SUMMARY_CACHE_ATTRIBUTE, None)
    if not isinstance(cache, dict) or cache.get("协调签名") != coordination_key:
        cache = {"协调签名": coordination_key, "字形": {}}
    cached_variants = cache.setdefault("字形", {})
    active_ids = set(details)
    for stale_id in set(cached_variants) - active_ids:
        cached_variants.pop(stale_id, None)
    for variant_id, detail in details.items():
        fingerprint = _variant_summary_fingerprint(detail)
        cached = cached_variants.get(variant_id)
        if not isinstance(cached, tuple) or cached[0] != fingerprint:
            cached_variants[variant_id] = (
                fingerprint,
                _summary_contribution(
                    detail,
                    coordination,
                    finished_dir,
                    verify_files=False,
                ),
            )
    setattr(glyph_service, _SUMMARY_CACHE_ATTRIBUTE, cache)
    totals = [0] * 9
    for _fingerprint, contribution in cached_variants.values():
        totals = [left + right for left, right in zip(totals, contribution)]
    (
        optimized,
        pending_optimization,
        reviewed,
        pending_review,
        review_admitted,
        coordinated,
        pending_coordination,
        coordination_admitted,
        export_ready,
    ) = totals
    glyph_total = len(details)
    summary = {
        "name": glyph_service.ziku_name,
        "path": glyph_service.ziku_dir,
        "characters": len(groups),
        "variants": glyph_total,
        "imported": glyph_total,
        "optimized": optimized,
        "pending_optimization": pending_optimization,
        "pending_review": pending_review,
        "review_admitted": review_admitted,
        "pending_coordination": pending_coordination,
        "reviewed": reviewed,
        "coordination_admitted": coordination_admitted,
        "finished": coordinated,
        "completed": coordinated,
        "coordinated": coordinated,
        "export_ready": export_ready,
        "metadata": dict(metadata),
    }
    glyph_service.save_library_summary(summary)
    return summary
