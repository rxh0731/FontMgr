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
    project_stage_status,
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
    phases = (PHASE_OPTIMIZATION, PHASE_REVIEW, PHASE_COORDINATION)
    phase_statuses = {phase: [] for phase in phases}
    glyph_total = len(safe_details)
    if progress_callback is not None:
        progress_callback(0, glyph_total)
    finished_dir = os.path.join(library_path, config.DIR_FINISHED_FILES)
    for glyph_current, detail in enumerate(safe_details.values(), start=1):
        source = detail if isinstance(detail, Mapping) else None
        for phase in phases:
            phase_statuses[phase].append(
                project_stage_status(
                    source,
                    safe_coordination,
                    finished_dir,
                    phase,
                    verify_files=verify_files,
                )
            )
        if progress_callback is not None:
            progress_callback(glyph_current, glyph_total)
    if glyph_total == 0 and progress_callback is not None:
        progress_callback(0, 0)

    optimization_statuses = phase_statuses[PHASE_OPTIMIZATION]
    review_statuses = phase_statuses[PHASE_REVIEW]
    coordination_statuses = phase_statuses[PHASE_COORDINATION]
    optimized = sum(status.completed for status in optimization_statuses)
    pending_optimization = sum(
        status.admitted and not status.completed
        for status in optimization_statuses
    )
    reviewed = sum(status.completed for status in review_statuses)
    pending_review = sum(
        status.admitted and not status.completed for status in review_statuses
    )
    coordinated = sum(status.completed for status in coordination_statuses)
    pending_coordination = sum(
        status.admitted and not status.completed
        for status in coordination_statuses
    )
    export_ready = sum(
        status.has_valid_finished for status in coordination_statuses
    )
    return {
        "name": library_name,
        "path": library_path,
        "characters": len(safe_groups),
        "variants": glyph_total,
        "imported": glyph_total,
        "optimized": optimized,
        "pending_optimization": pending_optimization,
        "pending_review": pending_review,
        "review_admitted": sum(status.admitted for status in review_statuses),
        "pending_coordination": pending_coordination,
        "reviewed": reviewed,
        "coordination_admitted": sum(
            status.admitted for status in coordination_statuses
        ),
        "finished": coordinated,
        "completed": coordinated,
        "coordinated": coordinated,
        "export_ready": export_ready,
        "metadata": dict(safe_metadata),
    }


def summarize_glyph_service(glyph_service: GlyphService) -> dict[str, Any]:
    """快速汇总本程序已经成功提交的字库内存状态。"""

    return build_library_summary(
        glyph_service.ziku_name,
        glyph_service.ziku_dir,
        glyph_service.get_variants(),
        glyph_service.get_glyph_groups(),
        glyph_service.get_metadata(),
        glyph_service.get_coordination_summary(),
        verify_files=False,
    )
