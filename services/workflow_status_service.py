"""统一解析字形的用户阶段、辅助标记与墨色结果。"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import config
from utils.file_utils import (
    is_real_directory,
    is_safe_windows_filename,
    resolve_safe_child_file,
)


STAGE_PENDING_OPTIMIZATION = "待优化"
STAGE_PENDING_REVIEW = "待审核"
STAGE_PENDING_COORDINATION = "待协调"
STAGE_COMPLETED = "已完成"

WORKFLOW_STAGES: tuple[str, ...] = (
    STAGE_PENDING_OPTIMIZATION,
    STAGE_PENDING_REVIEW,
    STAGE_PENDING_COORDINATION,
    STAGE_COMPLETED,
)
STAGE_COLORS: Mapping[str, str] = MappingProxyType(
    {
        STAGE_PENDING_OPTIMIZATION: "#888888",
        STAGE_PENDING_REVIEW: "#FF8C00",
        STAGE_PENDING_COORDINATION: "#4169E1",
        STAGE_COMPLETED: "#228B22",
    }
)
STAGE_FILTER_ALL = "全部阶段"
STAGE_FILTERS: tuple[str, ...] = (STAGE_FILTER_ALL, *WORKFLOW_STAGES)

PHASE_OPTIMIZATION = "自动优化"
PHASE_REVIEW = "手工审核"
PHASE_COORDINATION = "整体协调"
WORKFLOW_PHASES: tuple[str, ...] = (
    PHASE_OPTIMIZATION,
    PHASE_REVIEW,
    PHASE_COORDINATION,
)

PHASE_FILTER_ALL = "全部（本阶段）"
STATUS_OPTIMIZED = "已优化"
STATUS_REVIEWED = "已审核"
STATUS_COORDINATED = "已协调"

OPTIMIZATION_STATUS_FILTERS: tuple[str, ...] = (
    PHASE_FILTER_ALL,
    STAGE_PENDING_OPTIMIZATION,
    STATUS_OPTIMIZED,
)
REVIEW_STATUS_FILTERS: tuple[str, ...] = (
    PHASE_FILTER_ALL,
    STAGE_PENDING_REVIEW,
    STATUS_REVIEWED,
)
COORDINATION_STATUS_FILTERS: tuple[str, ...] = (
    PHASE_FILTER_ALL,
    STAGE_PENDING_COORDINATION,
    STATUS_COORDINATED,
)
PHASE_STATUS_FILTERS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        PHASE_OPTIMIZATION: OPTIMIZATION_STATUS_FILTERS,
        PHASE_REVIEW: REVIEW_STATUS_FILTERS,
        PHASE_COORDINATION: COORDINATION_STATUS_FILTERS,
    }
)
PHASE_PENDING_STATUSES: Mapping[str, str] = MappingProxyType(
    {
        PHASE_OPTIMIZATION: STAGE_PENDING_OPTIMIZATION,
        PHASE_REVIEW: STAGE_PENDING_REVIEW,
        PHASE_COORDINATION: STAGE_PENDING_COORDINATION,
    }
)
PHASE_COMPLETED_STATUSES: Mapping[str, str] = MappingProxyType(
    {
        PHASE_OPTIMIZATION: STATUS_OPTIMIZED,
        PHASE_REVIEW: STATUS_REVIEWED,
        PHASE_COORDINATION: STATUS_COORDINATED,
    }
)
PHASE_STATUS_COLORS: Mapping[str, str] = MappingProxyType(
    {
        STAGE_PENDING_OPTIMIZATION: STAGE_COLORS[STAGE_PENDING_OPTIMIZATION],
        STAGE_PENDING_REVIEW: STAGE_COLORS[STAGE_PENDING_REVIEW],
        STAGE_PENDING_COORDINATION: STAGE_COLORS[STAGE_PENDING_COORDINATION],
        STATUS_OPTIMIZED: "#228B22",
        STATUS_REVIEWED: "#228B22",
        STATUS_COORDINATED: "#228B22",
    }
)

MARKER_UNSAVED = "未保存修改"
MARKER_STRUCTURE_REVIEW = "结构需核对"
MARKER_INK_PENDING = "墨色待确认"
MARKER_INK_EXCEPTION = "人工例外"
MARKER_FILE_ERROR = "文件异常"

WORKFLOW_MARKERS: tuple[str, ...] = (
    MARKER_UNSAVED,
    MARKER_STRUCTURE_REVIEW,
    MARKER_INK_PENDING,
    MARKER_INK_EXCEPTION,
    MARKER_FILE_ERROR,
)
MARKER_FILTER_ALL = "全部提示"
MARKER_FILTERS: tuple[str, ...] = (
    MARKER_FILTER_ALL,
    *WORKFLOW_MARKERS,
)

INK_STATUS_NOT_APPLICABLE = "不适用"
INK_STATUS_DISABLED = "未启用"
INK_STATUS_ACHIEVED = "墨色已达标"
INK_STATUS_PENDING = MARKER_INK_PENDING
INK_STATUS_EXCEPTION = MARKER_INK_EXCEPTION
INK_STATUSES: tuple[str, ...] = (
    INK_STATUS_NOT_APPLICABLE,
    INK_STATUS_DISABLED,
    INK_STATUS_ACHIEVED,
    INK_STATUS_PENDING,
    INK_STATUS_EXCEPTION,
)

LEGACY_STATUS_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "待自动优化": config.STATUS_PENDING_OPTIMIZATION,
        "待手工审核": config.STATUS_PENDING_MANUAL_REVIEW,
        "已审核": config.STATUS_REVIEWED,
    }
)

_PRIMARY_STAGE_MAP: Mapping[str, str] = MappingProxyType(
    {
        config.STATUS_PENDING_OPTIMIZATION: STAGE_PENDING_OPTIMIZATION,
        config.STATUS_PENDING_MANUAL_REVIEW: STAGE_PENDING_REVIEW,
        config.STATUS_REVIEWED: STAGE_PENDING_COORDINATION,
        config.STATUS_FINISHED: STAGE_COMPLETED,
    }
)
_STRUCTURE_REVIEW_REQUIRED = "需人工核对"
_INK_BASELINE_TOLERANCE = 0.01
_LEGACY_STAGE_DIRS: Mapping[str, str] = MappingProxyType(
    {
        config.DIR_ORIGINAL_FILES: "原始文件",
        config.DIR_INTERMEDIATE_FILES: "中间文件",
        config.DIR_REVIEWED_FILES: "审核文件",
        config.DIR_FINISHED_FILES: "成品文件",
    }
)


@dataclass(frozen=True, slots=True)
class WorkflowStatus:
    """一次只读解析结果；元组字段保证结果本身不可变。"""

    stage: str
    markers: tuple[str, ...]
    ink_status: str
    has_valid_finished: bool

    @property
    def is_completed(self) -> bool:
        return self.stage == STAGE_COMPLETED

    def has_marker(self, marker: str) -> bool:
        return marker in self.markers

    def matches_stage(self, stage_filter: str) -> bool:
        return stage_filter == STAGE_FILTER_ALL or self.stage == stage_filter

    def matches_marker(self, marker_filter: str) -> bool:
        if marker_filter == MARKER_FILTER_ALL:
            return True
        return marker_filter in self.markers


@dataclass(frozen=True, slots=True)
class WorkflowStageProjection:
    """一个字形在指定制作阶段页面中的只读显示状态。"""

    phase: str
    status: str
    admitted: bool
    completed: bool
    markers: tuple[str, ...]
    workflow: WorkflowStatus

    @property
    def ink_status(self) -> str:
        return self.workflow.ink_status

    @property
    def has_valid_finished(self) -> bool:
        return self.workflow.has_valid_finished

    def has_marker(self, marker: str) -> bool:
        return marker in self.markers

    def matches_status(self, status_filter: str) -> bool:
        """未进入本阶段的记录不参与本阶段的任何列表筛选。"""

        return self.admitted and (
            status_filter == PHASE_FILTER_ALL or self.status == status_filter
        )


def resolve_workflow_status(
    detail: Mapping[str, Any] | None,
    coordination_summary: Mapping[str, Any] | None,
    finished_dir: str | os.PathLike[str],
    dirty: bool = False,
    *,
    verify_files: bool = True,
) -> WorkflowStatus:
    """按当前持久化记录解析用户阶段，不修改输入字典。"""

    source = detail if isinstance(detail, Mapping) else {}
    summary = (
        coordination_summary
        if isinstance(coordination_summary, Mapping)
        else {}
    )
    marker_set: set[str] = set()
    if dirty:
        marker_set.add(MARKER_UNSAVED)

    primary_status = _normalize_primary_status(source.get("状态"))
    stage = _PRIMARY_STAGE_MAP.get(primary_status)
    if stage is None:
        marker_set.add(MARKER_FILE_ERROR)
        return _result(
            STAGE_PENDING_OPTIMIZATION,
            marker_set,
            INK_STATUS_NOT_APPLICABLE,
            False,
        )

    if (
        primary_status == config.STATUS_PENDING_MANUAL_REVIEW
        and _requires_structure_review(source)
    ):
        marker_set.add(MARKER_STRUCTURE_REVIEW)

    if primary_status != config.STATUS_FINISHED:
        valid_source = (
            _has_valid_stage_source(source, primary_status, finished_dir)
            if verify_files
            else _has_valid_stage_reference(source, primary_status)
        )
        if not valid_source:
            marker_set.add(MARKER_FILE_ERROR)
        return _result(
            stage,
            marker_set,
            INK_STATUS_NOT_APPLICABLE,
            False,
        )

    has_valid_finished = (
        _has_valid_finished_file(source, finished_dir)
        if verify_files
        else is_safe_stage_filename(source.get("成品文件"))
    )
    if not has_valid_finished:
        marker_set.add(MARKER_FILE_ERROR)
        return _result(
            STAGE_PENDING_COORDINATION,
            marker_set,
            INK_STATUS_NOT_APPLICABLE,
            False,
        )

    ink_status, ink_completed = _resolve_ink_status(source, summary)
    if ink_status == INK_STATUS_PENDING:
        marker_set.add(MARKER_INK_PENDING)
    elif ink_status == INK_STATUS_EXCEPTION:
        marker_set.add(MARKER_INK_EXCEPTION)

    return _result(
        STAGE_COMPLETED if ink_completed else STAGE_PENDING_COORDINATION,
        marker_set,
        ink_status,
        True,
    )


class WorkflowStatusService:
    """供页面和其他服务统一调用的无状态解析入口。"""

    @staticmethod
    def resolve(
        detail: Mapping[str, Any] | None,
        coordination_summary: Mapping[str, Any] | None,
        finished_dir: str | os.PathLike[str],
        dirty: bool = False,
        *,
        verify_files: bool = True,
    ) -> WorkflowStatus:
        return resolve_workflow_status(
            detail,
            coordination_summary,
            finished_dir,
            dirty,
            verify_files=verify_files,
        )

    @staticmethod
    def project_stage(
        detail: Mapping[str, Any] | None,
        coordination_summary: Mapping[str, Any] | None,
        finished_dir: str | os.PathLike[str],
        phase: str,
        dirty: bool = False,
        *,
        verify_files: bool = True,
    ) -> WorkflowStageProjection:
        return project_stage_status(
            detail,
            coordination_summary,
            finished_dir,
            phase,
            dirty,
            verify_files=verify_files,
        )


def project_stage_status(
    detail: Mapping[str, Any] | None,
    coordination_summary: Mapping[str, Any] | None,
    finished_dir: str | os.PathLike[str],
    phase: str,
    dirty: bool = False,
    *,
    verify_files: bool = True,
) -> WorkflowStageProjection:
    """把内部流水线状态投影为指定页面的一组待处理/已完成状态。"""

    if phase not in WORKFLOW_PHASES:
        raise ValueError(f"未知制作阶段：{phase}")

    source = detail if isinstance(detail, Mapping) else {}
    workflow = resolve_workflow_status(
        source,
        coordination_summary,
        finished_dir,
        dirty,
        verify_files=verify_files,
    )
    primary_status = _normalize_primary_status(source.get("状态"))
    library_dir = _library_dir_from_finished_dir(finished_dir)
    valid_original = (
        bool(library_dir)
        and _has_stage_file(
            library_dir,
            config.DIR_ORIGINAL_FILES,
            source.get("原始文件"),
        )
        if verify_files
        else is_safe_stage_filename(source.get("原始文件"))
    )
    valid_optimized = (
        bool(library_dir)
        and _has_stage_file(
            library_dir,
            config.DIR_INTERMEDIATE_FILES,
            source.get("中间文件"),
        )
        if verify_files
        else is_safe_stage_filename(source.get("中间文件"))
    )
    reviewed_filename = str(source.get("审核文件", "") or "").strip()
    valid_reviewed_file = (
        bool(reviewed_filename)
        and (
            (
                bool(library_dir)
                and _has_stage_file(
                    library_dir,
                    config.DIR_REVIEWED_FILES,
                    reviewed_filename,
                )
            )
            if verify_files
            else is_safe_stage_filename(reviewed_filename)
        )
    )
    # 未另存人工稿时，审核通过允许继续复用自动优化稿；一旦记录了人工稿
    # 文件名，就必须使用该文件，不能用自动稿掩盖人工稿损坏。
    valid_review_source = (
        valid_reviewed_file if reviewed_filename else valid_optimized
    )

    review_claimed_complete = primary_status in {
        config.STATUS_REVIEWED,
        config.STATUS_FINISHED,
    }
    finished_proves_previous = (
        primary_status == config.STATUS_FINISHED
        and workflow.has_valid_finished
    )
    review_completed = review_claimed_complete and (
        valid_review_source or finished_proves_previous
    )
    optimization_claimed_complete = primary_status in {
        config.STATUS_PENDING_MANUAL_REVIEW,
        config.STATUS_REVIEWED,
        config.STATUS_FINISHED,
    }
    optimization_completed = optimization_claimed_complete and (
        valid_optimized
        or valid_reviewed_file
        or review_completed
        or finished_proves_previous
    )

    if phase == PHASE_OPTIMIZATION:
        admitted = True
        completed = optimization_completed
        file_error = (
            primary_status not in _PRIMARY_STAGE_MAP
            or (
                primary_status == config.STATUS_PENDING_OPTIMIZATION
                and not valid_original
            )
            or (
                primary_status != config.STATUS_PENDING_OPTIMIZATION
                and not optimization_completed
            )
        )
    elif phase == PHASE_REVIEW:
        admitted = optimization_completed
        completed = admitted and review_completed
        file_error = (
            admitted
            and (
                (
                    primary_status == config.STATUS_PENDING_MANUAL_REVIEW
                    and bool(reviewed_filename)
                    and not valid_reviewed_file
                )
                or (
                    review_claimed_complete
                    and not review_completed
                )
            )
        )
    else:
        admitted = review_completed
        completed = admitted and workflow.is_completed
        file_error = (
            admitted
            and primary_status == config.STATUS_FINISHED
            and not workflow.has_valid_finished
        )

    markers = _projection_markers(workflow, phase, file_error=file_error)

    status = ""
    if admitted:
        status = (
            PHASE_COMPLETED_STATUSES[phase]
            if completed
            else PHASE_PENDING_STATUSES[phase]
        )
    return WorkflowStageProjection(
        phase=phase,
        status=status,
        admitted=admitted,
        completed=completed,
        markers=markers,
        workflow=workflow,
    )


def _normalize_primary_status(raw_status: object) -> str:
    value = str(raw_status or "").strip()
    return LEGACY_STATUS_ALIASES.get(value, value)


def _library_dir_from_finished_dir(
    finished_dir: str | os.PathLike[str],
) -> str:
    try:
        return os.path.dirname(os.path.abspath(os.fspath(finished_dir)))
    except (OSError, TypeError, ValueError):
        return ""


def _projection_markers(
    workflow: WorkflowStatus,
    phase: str,
    *,
    file_error: bool,
) -> tuple[str, ...]:
    values: set[str] = set()
    if workflow.has_marker(MARKER_UNSAVED):
        values.add(MARKER_UNSAVED)
    if (
        phase in {PHASE_OPTIMIZATION, PHASE_REVIEW}
        and workflow.has_marker(MARKER_STRUCTURE_REVIEW)
    ):
        values.add(MARKER_STRUCTURE_REVIEW)
    if phase == PHASE_COORDINATION:
        if workflow.has_marker(MARKER_INK_PENDING):
            values.add(MARKER_INK_PENDING)
        if workflow.has_marker(MARKER_INK_EXCEPTION):
            values.add(MARKER_INK_EXCEPTION)
    if file_error:
        values.add(MARKER_FILE_ERROR)
    return tuple(marker for marker in WORKFLOW_MARKERS if marker in values)


def _result(
    stage: str,
    marker_set: set[str],
    ink_status: str,
    has_valid_finished: bool,
) -> WorkflowStatus:
    markers = tuple(
        marker for marker in WORKFLOW_MARKERS if marker in marker_set
    )
    return WorkflowStatus(
        stage=stage,
        markers=markers,
        ink_status=ink_status,
        has_valid_finished=has_valid_finished,
    )


def _requires_structure_review(detail: Mapping[str, Any]) -> bool:
    optimization = detail.get("自动优化")
    if not isinstance(optimization, Mapping):
        return False
    scheme = optimization.get("方案")
    if not isinstance(scheme, Mapping):
        return False
    review = scheme.get("结构复核")
    return (
        isinstance(review, Mapping)
        and str(review.get("状态", "") or "").strip()
        == _STRUCTURE_REVIEW_REQUIRED
    )


def _has_valid_finished_file(
    detail: Mapping[str, Any],
    finished_dir: str | os.PathLike[str],
) -> bool:
    filename = detail.get("成品文件")
    try:
        normalized_dir = os.path.normpath(os.path.abspath(os.fspath(finished_dir)))
    except (OSError, TypeError, ValueError):
        return False
    if os.path.basename(normalized_dir) != config.DIR_FINISHED_FILES:
        return _has_safe_existing_file(normalized_dir, filename)
    return _has_stage_file(
        os.path.dirname(normalized_dir),
        config.DIR_FINISHED_FILES,
        filename,
    )


def _has_valid_stage_source(
    detail: Mapping[str, Any],
    primary_status: str,
    finished_dir: str | os.PathLike[str],
) -> bool:
    """核对当前阶段至少有一个可继续处理的真实文件。"""

    try:
        library_dir = os.path.dirname(
            os.path.abspath(os.fspath(finished_dir))
        )
    except (OSError, TypeError, ValueError):
        return False
    if primary_status == config.STATUS_PENDING_OPTIMIZATION:
        return _has_stage_file(
            library_dir,
            config.DIR_ORIGINAL_FILES,
            detail.get("原始文件"),
        )
    elif primary_status in {
        config.STATUS_PENDING_MANUAL_REVIEW,
        config.STATUS_REVIEWED,
    }:
        reviewed_filename = str(detail.get("审核文件", "") or "").strip()
        if reviewed_filename:
            return _has_stage_file(
                library_dir,
                config.DIR_REVIEWED_FILES,
                reviewed_filename,
            )
        return _has_stage_file(
            library_dir,
            config.DIR_INTERMEDIATE_FILES,
            detail.get("中间文件"),
        )
    return False


def _has_valid_stage_reference(
    detail: Mapping[str, Any],
    primary_status: str,
) -> bool:
    """本程序已提交事务的快速核对，仅验证阶段文件名契约。"""

    if primary_status == config.STATUS_PENDING_OPTIMIZATION:
        return is_safe_stage_filename(detail.get("原始文件"))
    if primary_status in {
        config.STATUS_PENDING_MANUAL_REVIEW,
        config.STATUS_REVIEWED,
    }:
        reviewed_filename = str(detail.get("审核文件", "") or "").strip()
        return is_safe_stage_filename(
            reviewed_filename or detail.get("中间文件")
        )
    return False


def _has_stage_file(
    library_dir: str,
    current_name: str,
    filename: object,
) -> bool:
    """兼容尚未执行目录迁移的旧字库，并显式暴露目录冲突。"""

    current_dir = os.path.join(library_dir, current_name)
    legacy_name = _LEGACY_STAGE_DIRS.get(current_name, "")
    legacy_dir = os.path.join(library_dir, legacy_name) if legacy_name else ""
    current_exists = os.path.isdir(current_dir)
    legacy_exists = bool(legacy_dir) and os.path.isdir(legacy_dir)
    if current_exists and legacy_exists:
        return False
    target_dir = legacy_dir if legacy_exists else current_dir
    return _has_safe_existing_file(target_dir, filename)


def _has_safe_existing_file(
    directory: str | os.PathLike[str],
    raw_filename: object,
) -> bool:
    return bool(resolve_safe_stage_file(directory, raw_filename))


def resolve_safe_stage_file(
    directory: str | os.PathLike[str],
    raw_filename: object,
) -> str:
    """返回阶段目录内可安全读取的真实文件；非法、缺失或链接返回空串。"""

    return resolve_safe_child_file(directory, raw_filename)


def is_safe_stage_filename(raw_filename: object) -> bool:
    """判断阶段文件名是否为不带路径语义的合法普通文件名。"""

    return is_safe_windows_filename(raw_filename)


def is_safe_stage_directory(directory: str | os.PathLike[str]) -> bool:
    """判断阶段目录是否为存在且不经符号链接或目录联接的真实目录。"""

    return is_real_directory(directory)


def _resolve_ink_status(
    detail: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> tuple[str, bool]:
    parameters = detail.get("整体协调参数")
    record = parameters.get("墨色协调") if isinstance(parameters, Mapping) else None
    if not isinstance(record, Mapping):
        return INK_STATUS_PENDING, False

    enabled_value = summary.get("墨色统一启用", True)
    if enabled_value is False:
        matches_disabled = (
            record.get("启用") is False
            and _has_saved_recheck(record)
        )
        return (
            (INK_STATUS_DISABLED, True)
            if matches_disabled
            else (INK_STATUS_PENDING, False)
        )
    if enabled_value is not True:
        return INK_STATUS_PENDING, False

    if not _matches_enabled_ink_contract(record, summary):
        return INK_STATUS_PENDING, False
    if record.get("人工接受例外") is True:
        return INK_STATUS_EXCEPTION, True
    if record.get("是否达标") is True:
        return INK_STATUS_ACHIEVED, True
    return INK_STATUS_PENDING, False


def _matches_enabled_ink_contract(
    record: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> bool:
    if record.get("启用") is not True or not _has_saved_recheck(record):
        return False

    summary_method = str(summary.get("墨色方法", "") or "").strip()
    record_method = str(record.get("方法", "") or "").strip()
    if not summary_method or record_method != summary_method:
        return False

    summary_version = _positive_int(summary.get("墨色方法版本"))
    record_version = _positive_int(record.get("方法版本"))
    if summary_version is None or record_version != summary_version:
        return False

    summary_baseline = _finite_number(summary.get("墨色基准"))
    record_baseline = _finite_number(record.get("基准"))
    return (
        summary_baseline is not None
        and record_baseline is not None
        and math.isclose(
            record_baseline,
            summary_baseline,
            rel_tol=0.0,
            abs_tol=_INK_BASELINE_TOLERANCE,
        )
    )


def _has_saved_recheck(record: Mapping[str, Any]) -> bool:
    return (
        record.get("保存后复测") is True
        and _finite_number(record.get("保存后墨色")) is not None
    )


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number > 0 and number == value else None


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None
