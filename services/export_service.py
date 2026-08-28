"""最终成品审计、预览与批次导出服务。"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image, ImageFilter, UnidentifiedImageError

import config
from services.glyph_service import GlyphService
from services.workflow_status_service import (
    STAGE_COMPLETED,
    is_safe_stage_directory,
    is_safe_stage_filename,
    resolve_safe_stage_file,
    resolve_workflow_status,
)
from utils.file_utils import compute_file_md5, natural_key, pinyin_natural_key


@dataclass(frozen=True)
class ExportOptions:
    """描述一次不会写回字库的导出画布设置。"""

    mode: str = "library_spec"
    dpi: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    include_transparent_area: bool = True
    name_mode: str = "字符"
    sequence_mode: str | None = None
    image_format: str = "PNG"
    allow_upscale: bool = False


@dataclass(frozen=True)
class ExportConflict:
    """导出目标目录中一个需要用户决定的同名文件。"""

    variant_id: str
    char: str
    destination_name: str
    destination_path: str
    file_size: int
    modified_ns: int
    device: int
    inode: int


@dataclass(frozen=True)
class ExportConflictDecision:
    """用户对一个已核对同名文件作出的本次导出决定。"""

    conflict: ExportConflict
    action: str


@dataclass(frozen=True)
class _ExportItem:
    detail: dict[str, Any]
    source_path: str
    destination_name: str
    overwrite_conflict: ExportConflict | None = None


class ExportService:
    """从最终成品生成只读预览，并以完整批次导出图片。"""

    MODE_LIBRARY_SPEC = "library_spec"
    MODE_TRIM_TRANSPARENT = "trim_transparent"
    MODE_CUSTOM_SPEC = "custom_spec"
    EXPORT_MODES = (
        MODE_LIBRARY_SPEC,
        MODE_TRIM_TRANSPARENT,
        MODE_CUSTOM_SPEC,
    )
    NAME_MODES = ("普通序号", "自动等宽序号")
    IMAGE_FORMATS = ("PNG", "JPEG", "TIFF", "BMP", "WEBP")
    FORMAT_EXTENSIONS = {
        "PNG": ".png",
        "JPEG": ".jpg",
        "TIFF": ".tif",
        "BMP": ".bmp",
        "WEBP": ".webp",
    }
    SEQUENCE_MODES = NAME_MODES
    OUTPUT_STYLES = ("灰度保真", "纯二值", "统一软边")
    CONFLICT_OVERWRITE = "overwrite"
    CONFLICT_SKIP = "skip"
    CONFLICT_ACTIONS = (CONFLICT_OVERWRITE, CONFLICT_SKIP)
    MAX_DIMENSION = 16_384
    MAX_PIXELS = 64 * 1024 * 1024
    MAX_SOURCE_DIMENSION = 32_768
    MAX_SOURCE_PIXELS = 64 * 1024 * 1024

    def __init__(
        self,
        glyph_service: GlyphService,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        self._glyph = glyph_service
        self._progress = progress_callback

    def audit_readiness(
        self,
        verify_hash: bool = False,
        cancel_check: Optional[Callable[[], bool]] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> dict[str, Any]:
        """核对全库流程、协调摘要和成品文件，返回可直接展示的中文摘要。"""
        variants = self._glyph.get_all_variants()
        progress = progress_callback if progress_callback is not None else self._progress
        total_variants = len(variants)
        if progress:
            progress("核对：准备开始", 0, total_variants)
        counts = {
            "待优化": 0,
            "待审核": 0,
            "待协调": 0,
            "状态异常": 0,
            "成品缺失": 0,
            "成品损坏": 0,
            "路径无效": 0,
            "校验不符": 0,
        }
        issue_details: list[dict[str, str]] = []
        ready_count = 0
        cancelled = False
        summary = self._glyph.get_coordination_summary()
        finished_dir = self._glyph.get_workflow_dirs()["成品"]

        for index, detail in enumerate(variants, 1):
            if cancel_check is not None and cancel_check():
                cancelled = True
                break
            status = str(detail.get("状态", ""))
            if status == config.STATUS_PENDING_OPTIMIZATION:
                counts["待优化"] += 1
                self._append_issue(issue_details, detail, "待优化")
                if progress:
                    progress(f"核对：{detail.get('归属字', '')}", index, total_variants)
                continue
            if status == config.STATUS_PENDING_MANUAL_REVIEW:
                counts["待审核"] += 1
                self._append_issue(issue_details, detail, "待审核")
                if progress:
                    progress(f"核对：{detail.get('归属字', '')}", index, total_variants)
                continue
            if status == config.STATUS_REVIEWED:
                counts["待协调"] += 1
                self._append_issue(issue_details, detail, "待协调")
                if progress:
                    progress(f"核对：{detail.get('归属字', '')}", index, total_variants)
                continue
            if status != config.STATUS_FINISHED:
                counts["状态异常"] += 1
                self._append_issue(issue_details, detail, "状态异常", status or "状态为空")
                if progress:
                    progress(f"核对：{detail.get('归属字', '')}", index, total_variants)
                continue

            try:
                source_path = self._resolve_finished_source(detail)
            except ValueError as exc:
                counts["路径无效"] += 1
                self._append_issue(issue_details, detail, "路径无效", str(exc))
                if progress:
                    progress(f"核对：{detail.get('归属字', '')}", index, total_variants)
                continue
            except FileNotFoundError as exc:
                counts["成品缺失"] += 1
                self._append_issue(issue_details, detail, "成品缺失", str(exc))
                if progress:
                    progress(f"核对：{detail.get('归属字', '')}", index, total_variants)
                continue

            try:
                with Image.open(source_path) as source:
                    self._validate_source_size(source)
                    if not self._has_visible_content(source):
                        raise ValueError("成品图像没有可见文字")
            except (OSError, ValueError, SyntaxError, UnidentifiedImageError) as exc:
                counts["成品损坏"] += 1
                self._append_issue(issue_details, detail, "成品损坏", str(exc))
                if progress:
                    progress(f"核对：{detail.get('归属字', '')}", index, total_variants)
                continue

            expected_hash = str(detail.get("成品MD5", "")).strip()
            if verify_hash and expected_hash:
                try:
                    hash_matches = compute_file_md5(source_path) == expected_hash
                except OSError as exc:
                    counts["成品损坏"] += 1
                    self._append_issue(issue_details, detail, "成品损坏", str(exc))
                    if progress:
                        progress(f"核对：{detail.get('归属字', '')}", index, total_variants)
                    continue
                if not hash_matches:
                    counts["校验不符"] += 1
                    self._append_issue(issue_details, detail, "校验不符", "成品 MD5 与记录不一致")
                    if progress:
                        progress(f"核对：{detail.get('归属字', '')}", index, total_variants)
                    continue
            if cancel_check is not None and cancel_check():
                cancelled = True
                break
            workflow = resolve_workflow_status(
                detail,
                summary,
                finished_dir,
            )
            if workflow.stage != STAGE_COMPLETED:
                counts["待协调"] += 1
                marker_text = "、".join(workflow.markers)
                message = marker_text or workflow.ink_status or "逐字协调契约未完成"
                self._append_issue(
                    issue_details,
                    detail,
                    "待协调",
                    message,
                )
                if progress:
                    progress(f"核对：{detail.get('归属字', '')}", index, total_variants)
                continue
            ready_count += 1
            if progress:
                progress(f"核对：{detail.get('归属字', '')}", index, total_variants)

        if progress and not cancelled:
            progress("核对完成", total_variants, total_variants)

        geometry_completed = bool(summary.get("几何协调完成", False))
        ink_enabled = bool(summary.get("墨色统一启用", True))
        ink_completed = bool(summary.get("墨色统一完成", False))
        ink_audit, ink_reasons = self._audit_ink_summary(
            summary,
            len(variants),
            enabled=ink_enabled,
            completed=ink_completed,
        )
        reasons = (
            ["全库状态核对已取消"]
            if cancelled
            else self._readiness_reasons(
                len(variants),
                counts,
                geometry_completed,
                ink_enabled,
                ink_completed,
                ink_reasons,
            )
        )
        return {
            "就绪": not reasons and not cancelled,
            "已取消": cancelled,
            "总数": len(variants),
            "已就绪": ready_count,
            **counts,
            "几何协调完成": geometry_completed,
            "墨色统一启用": ink_enabled,
            "墨色统一完成": ink_completed,
            **ink_audit,
            "原因": reasons,
            "问题详情": issue_details,
        }

    def preview_image(
        self,
        detail_or_variant_id: dict[str, Any] | str,
        options: ExportOptions,
    ) -> Image.Image:
        """按导出设置生成单张独立 RGBA 预览，不写入磁盘。"""
        detail = self._resolve_detail(detail_or_variant_id)
        source_path = self._resolve_finished_source(detail)
        normalized = self._validate_options(options)
        with Image.open(source_path) as source:
            output, dpi = self._render_new_mode(source, normalized)
        output.info["dpi"] = (float(dpi), float(dpi))
        return output

    def find_destination_conflicts(
        self,
        output_dir: str,
        options: ExportOptions,
        *,
        eligible_variant_ids: Optional[Iterable[str]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> list[ExportConflict]:
        """只读核对确定性目标文件名，返回需要用户处理的同名文件。"""
        destination_dir = self._validate_output_dir(output_dir)
        normalized_options = self._validate_options(options)
        eligible_ids = (
            None
            if eligible_variant_ids is None
            else {str(variant_id) for variant_id in eligible_variant_ids}
        )
        sequence_mode, _legacy_naming = self._normalized_naming(normalized_options)
        extension = self.FORMAT_EXTENSIONS[self._normalized_format(normalized_options)]
        _items, _skipped, failures, conflicts = self._build_items(
            destination_dir,
            sequence_mode,
            legacy_mode=False,
            extension=extension,
            eligible_variant_ids=eligible_ids,
            collect_conflicts=True,
            cancel_check=cancel_check,
            progress_callback=progress_callback,
        )
        if failures:
            details = "；".join(
                f"{variant_id or '字形'}：{reason}"
                for variant_id, reason in failures[:5]
            )
            if len(failures) > 5:
                details += f"；另有 {len(failures) - 5} 项"
            raise ValueError(f"无法准备导出：{details}")
        return conflicts

    def export(
        self,
        output_dir: str,
        name_mode: str = "字符",
        transparent_background: bool = True,
        output_style: str = "灰度保真",
        *,
        options: Optional[ExportOptions] = None,
        require_ready: bool = False,
        cancel_check: Optional[Callable[[], bool]] = None,
        eligible_variant_ids: Optional[Iterable[str]] = None,
        conflict_decisions: Optional[Iterable[ExportConflictDecision]] = None,
    ) -> dict[str, Any]:
        """完整批次导出；新页面传 ``options``，旧调用仍沿用原背景与风格参数。"""
        legacy_mode = options is None
        if legacy_mode:
            if output_style not in self.OUTPUT_STYLES:
                raise ValueError(f"不支持的成品风格：{output_style}")
            if name_mode not in (*self.NAME_MODES, "字符", "原文件名"):
                raise ValueError(f"不支持的文件命名方式：{name_mode}")
            normalized_options = ExportOptions(name_mode=name_mode)
        else:
            normalized_options = self._validate_options(options)

        decision_map = self._normalize_conflict_decisions(conflict_decisions)
        if legacy_mode and decision_map:
            raise ValueError("旧版导出接口不接受同名文件处理决定")

        if require_ready:
            audit = self.audit_readiness(
                verify_hash=True,
                cancel_check=cancel_check,
            )
            if audit.get("已取消"):
                return self._result(
                    0,
                    0,
                    0,
                    str(output_dir),
                    cancelled=True,
                )
            if not audit["就绪"]:
                reason = "；".join(audit["原因"]) or "字库尚未完成全部处理"
                raise RuntimeError(f"当前字库不能完整导出：{reason}")

        destination_dir = self._validate_output_dir(output_dir)
        eligible_ids = (
            None
            if eligible_variant_ids is None
            else {str(variant_id) for variant_id in eligible_variant_ids}
        )
        sequence_mode, _legacy_naming = self._normalized_naming(normalized_options)
        output_format = self._normalized_format(normalized_options)
        items, skipped, plan_failures, _conflicts = self._build_items(
            destination_dir,
            sequence_mode,
            legacy_mode=legacy_mode,
            extension=(
                self.FORMAT_EXTENSIONS[output_format]
                if not legacy_mode
                else (".png" if transparent_background else ".bmp")
            ),
            eligible_variant_ids=eligible_ids,
            conflict_decisions=decision_map,
        )
        if plan_failures:
            return self._result(
                success=0,
                skipped=skipped,
                failure=len(items) + len(plan_failures),
                output_dir=destination_dir,
                failure_details=plan_failures,
            )
        if not items:
            return self._result(0, skipped, 0, destination_dir)

        parent_dir = os.path.dirname(destination_dir)
        os.makedirs(parent_dir, exist_ok=True)
        staging_dir = tempfile.mkdtemp(prefix=".fonteditor_export_", dir=parent_dir)
        try:
            for index, item in enumerate(items):
                if cancel_check is not None and cancel_check():
                    return self._result(
                        0, skipped, 0, destination_dir, cancelled=True
                    )
                image: Image.Image | None = None
                try:
                    staging_path = self._safe_child_path(
                        staging_dir,
                        item.destination_name,
                        "导出暂存文件",
                    )
                    if (
                        not legacy_mode
                        and normalized_options.mode == self.MODE_LIBRARY_SPEC
                        and output_format == "PNG"
                    ):
                        # 默认导出保持最终成品文件的编码、像素、透明区和附加信息不变。
                        with Image.open(item.source_path) as source:
                            self._validate_source_size(source)
                            if not self._has_visible_content(source):
                                raise ValueError("成品图像没有可见文字")
                        shutil.copyfile(item.source_path, staging_path)
                        self._report(
                            f"导出：{item.detail.get('归属字', '')}",
                            index + 1,
                            len(items),
                        )
                        continue
                    with Image.open(item.source_path) as source:
                        if legacy_mode:
                            image, image_format, dpi = self._render_legacy_mode(
                                source,
                                transparent_background,
                                output_style,
                            )
                        else:
                            image, dpi = self._render_new_mode(
                                source, normalized_options
                            )
                            image_format = output_format
                            if output_format in {"JPEG", "BMP"}:
                                flattened = Image.new("RGB", image.size, "white")
                                rgb_image = image.convert("RGB")
                                alpha = image.getchannel("A")
                                try:
                                    flattened.paste(rgb_image, mask=alpha)
                                finally:
                                    rgb_image.close()
                                    alpha.close()
                                image.close()
                                image = flattened
                    if cancel_check is not None and cancel_check():
                        return self._result(
                            0, skipped, 0, destination_dir, cancelled=True
                        )
                    self._save_output(
                        image,
                        staging_path,
                        image_format,
                        dpi,
                    )
                except (OSError, ValueError, RuntimeError, SyntaxError, UnidentifiedImageError) as exc:
                    failures = self._batch_failure_details(items, item, str(exc))
                    return self._result(
                        0,
                        skipped,
                        len(items),
                        destination_dir,
                        failure_details=failures,
                    )
                finally:
                    if image is not None:
                        image.close()
                self._report(
                    f"导出：{item.detail.get('归属字', '')}", index + 1, len(items)
                )

            if cancel_check is not None and cancel_check():
                return self._result(0, skipped, 0, destination_dir, cancelled=True)
            self._commit_staging(staging_dir, destination_dir, items)
            overwritten = sum(item.overwrite_conflict is not None for item in items)
            return self._result(
                len(items),
                skipped,
                0,
                destination_dir,
                overwritten=overwritten,
            )
        finally:
            if os.path.isdir(staging_dir):
                shutil.rmtree(staging_dir, ignore_errors=True)

    def _build_items(
        self,
        output_dir: str,
        sequence_mode: str,
        *,
        legacy_mode: bool,
        extension: str,
        eligible_variant_ids: set[str] | None = None,
        conflict_decisions: Mapping[str, ExportConflictDecision] | None = None,
        collect_conflicts: bool = False,
        cancel_check: Optional[Callable[[], bool]] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> tuple[
        list[_ExportItem],
        int,
        list[tuple[str, str]],
        list[ExportConflict],
    ]:
        items: list[_ExportItem] = []
        skipped = 0
        failures: list[tuple[str, str]] = []
        conflicts: list[ExportConflict] = []
        reserved: set[str] = set()
        group_positions: dict[str, int] = {}
        decisions = conflict_decisions or {}
        existing_paths = (
            {} if legacy_mode else self._existing_destination_paths(output_dir)
        )

        ordered_variants = self._ordered_variants()
        total = len(ordered_variants)
        max_group_count = max(
            (len(self._glyph.get_char_variants(char)) for char in self._glyph.get_all_chars()),
            default=1,
        )
        for index, detail in enumerate(ordered_variants, 1):
            if cancel_check is not None and cancel_check():
                break
            if progress_callback is not None:
                progress_callback(
                    f"准备：{detail.get('归属字', '')}",
                    index,
                    total,
                )
            char = str(detail.get("归属字", "")) or "字形"
            group_positions[char] = group_positions.get(char, 0) + 1
            variant_index = group_positions[char] - 1
            variant_id = str(detail.get("变体ID", ""))
            if (
                eligible_variant_ids is not None
                and variant_id not in eligible_variant_ids
            ):
                skipped += 1
                continue
            if detail.get("状态") != config.STATUS_FINISHED:
                skipped += 1
                continue
            try:
                source_path = self._resolve_finished_source(detail)
                original_name = (
                    self._preferred_original_name(detail, source_path)
                    if sequence_mode not in self.SEQUENCE_MODES
                    else ""
                )
                if legacy_mode:
                    base_name = self._make_dest_name(
                        char,
                        original_name,
                        variant_index,
                        sequence_mode,
                        extension,
                        max_group_count=max_group_count,
                    )
                    self._validate_destination_name(base_name)
                    destination_name = self._allocate_legacy_name(
                        output_dir, base_name, reserved
                    )
                else:
                    destination_name = self._deterministic_dest_name(
                        char,
                        original_name,
                        variant_index,
                        sequence_mode,
                        extension,
                        max_group_count=max_group_count,
                    )
                    self._validate_destination_name(destination_name)
                    self._safe_child_path(
                        output_dir,
                        destination_name,
                        "导出目标文件",
                    )
                    normalized_name = self._destination_key(destination_name)
                    if normalized_name in reserved:
                        raise ValueError(f"导出文件名重复：{destination_name}")
                    reserved.add(normalized_name)
                    existing_path = existing_paths.get(normalized_name)
                    decision = decisions.get(normalized_name)
                    overwrite_conflict: ExportConflict | None = None
                    if existing_path is not None:
                        if not os.path.isfile(existing_path):
                            raise IsADirectoryError(
                                f"导出目标同名项不是文件：{destination_name}"
                            )
                        current_conflict = self._make_export_conflict(
                            detail,
                            destination_name,
                            existing_path,
                        )
                        conflicts.append(current_conflict)
                        if collect_conflicts:
                            items.append(
                                _ExportItem(detail, source_path, destination_name)
                            )
                            continue
                        if decision is None:
                            raise FileExistsError(
                                f"导出目录已有同名文件：{destination_name}"
                            )
                        if decision.action == self.CONFLICT_SKIP:
                            skipped += 1
                            continue
                        if not self._same_export_conflict(
                            decision.conflict,
                            current_conflict,
                        ):
                            raise FileExistsError(
                                f"同名文件在确认后发生变化：{destination_name}"
                            )
                        overwrite_conflict = decision.conflict
                    elif decision is not None:
                        if decision.action == self.CONFLICT_SKIP:
                            skipped += 1
                            continue
                        raise FileNotFoundError(
                            f"准备覆盖的同名文件在确认后已不存在：{destination_name}"
                        )
                    items.append(
                        _ExportItem(
                            detail,
                            source_path,
                            destination_name,
                            overwrite_conflict,
                        )
                    )
                    continue
                items.append(_ExportItem(detail, source_path, destination_name))
            except (OSError, ValueError) as exc:
                failures.append((variant_id, str(exc)))
        return items, skipped, failures, conflicts

    @classmethod
    def _normalize_conflict_decisions(
        cls,
        decisions: Optional[Iterable[ExportConflictDecision]],
    ) -> dict[str, ExportConflictDecision]:
        normalized: dict[str, ExportConflictDecision] = {}
        for decision in decisions or ():
            if not isinstance(decision, ExportConflictDecision):
                raise TypeError("同名文件处理决定类型无效")
            if decision.action not in cls.CONFLICT_ACTIONS:
                raise ValueError(f"不支持的同名文件处理方式：{decision.action}")
            name = str(decision.conflict.destination_name)
            try:
                cls._validate_destination_name(name)
            except ValueError as exc:
                raise ValueError("同名文件处理决定包含不安全的目标文件名") from exc
            key = cls._destination_key(name)
            if key in normalized:
                raise ValueError(f"同一目标文件存在重复处理决定：{name}")
            normalized[key] = decision
        return normalized

    @staticmethod
    def _destination_key(filename: str) -> str:
        return os.path.normcase(str(filename))

    @classmethod
    def _existing_destination_paths(cls, output_dir: str) -> dict[str, str]:
        if not os.path.isdir(output_dir):
            return {}
        paths: dict[str, str] = {}
        with os.scandir(output_dir) as entries:
            for entry in entries:
                paths.setdefault(cls._destination_key(entry.name), entry.path)
        return paths

    @staticmethod
    def _make_export_conflict(
        detail: dict[str, Any],
        destination_name: str,
        destination_path: str,
    ) -> ExportConflict:
        stat = os.stat(destination_path)
        return ExportConflict(
            variant_id=str(detail.get("变体ID", "")),
            char=str(detail.get("归属字", "")),
            destination_name=destination_name,
            destination_path=os.path.abspath(destination_path),
            file_size=int(stat.st_size),
            modified_ns=int(stat.st_mtime_ns),
            device=int(stat.st_dev),
            inode=int(stat.st_ino),
        )

    @staticmethod
    def _same_export_conflict(
        expected: ExportConflict,
        current: ExportConflict,
    ) -> bool:
        return (
            os.path.normcase(os.path.abspath(expected.destination_path))
            == os.path.normcase(os.path.abspath(current.destination_path))
            and expected.variant_id == current.variant_id
            and expected.destination_name == current.destination_name
            and expected.file_size == current.file_size
            and expected.modified_ns == current.modified_ns
            and expected.device == current.device
            and expected.inode == current.inode
        )

    def _ordered_variants(self) -> list[dict[str, Any]]:
        ordered: list[dict[str, Any]] = []
        seen: set[str] = set()
        for char in self._glyph.get_all_chars():
            for detail in self._glyph.get_char_variants(char):
                variant_id = str(detail.get("变体ID", ""))
                if variant_id in seen:
                    continue
                ordered.append(detail)
                seen.add(variant_id)
        remaining = [
            detail
            for detail in self._glyph.get_all_variants()
            if str(detail.get("变体ID", "")) not in seen
        ]
        remaining.sort(
            key=lambda detail: (
                pinyin_natural_key(str(detail.get("归属字", ""))),
                natural_key(str(detail.get("原始文件", ""))),
                str(detail.get("变体ID", "")),
            )
        )
        return ordered + remaining

    @classmethod
    def _preferred_original_name(
        cls,
        detail: dict[str, Any],
        fallback_path: str,
    ) -> str:
        """优先使用用户导入前的文件名，兼容没有该字段的历史字库。"""
        for key in ("导入前文件名", "原始文件"):
            name = str(detail.get(key, "") or "").strip()
            if name:
                return cls._validate_destination_name(name, label=key)
        return cls._validate_destination_name(os.path.basename(fallback_path))

    def _resolve_detail(
        self, detail_or_variant_id: dict[str, Any] | str
    ) -> dict[str, Any]:
        if isinstance(detail_or_variant_id, dict):
            detail = detail_or_variant_id
        else:
            detail = self._glyph.get_variant(str(detail_or_variant_id))
        if not detail:
            raise KeyError("找不到指定字形")
        return detail

    def _resolve_finished_source(self, detail: dict[str, Any]) -> str:
        raw_source_name = detail.get("成品文件", "")
        source_name = str(raw_source_name or "")
        if not source_name:
            raise FileNotFoundError("没有成品文件记录")
        if not is_safe_stage_filename(raw_source_name):
            raise ValueError("成品文件路径不安全")
        final_dir = self._glyph.get_workflow_dirs()["成品"]
        if not is_safe_stage_directory(final_dir):
            raise ValueError("最终成品目录无效或使用了链接")
        source_path = resolve_safe_stage_file(final_dir, raw_source_name)
        if source_path:
            return source_path

        candidate = os.path.join(os.path.abspath(final_dir), source_name)
        is_junction = getattr(os.path, "isjunction", lambda _path: False)
        if (
            os.path.islink(candidate)
            or is_junction(candidate)
            or os.path.lexists(candidate)
        ):
            raise ValueError("成品文件不是安全的普通文件")
        raise FileNotFoundError(f"找不到成品文件：{source_name}")

    def _validate_output_dir(self, output_dir: str) -> str:
        if not str(output_dir).strip():
            raise ValueError("导出目录不能为空")
        destination = self._real_path(str(output_dir))
        library_dir = self._real_path(self._glyph.ziku_dir)
        if self._is_within(destination, library_dir):
            raise ValueError("导出目录不能位于当前字库内部")
        if os.path.exists(destination) and not os.path.isdir(destination):
            raise NotADirectoryError(f"导出目标不是目录：{destination}")
        return destination

    def _validate_options(self, options: ExportOptions) -> ExportOptions:
        if not isinstance(options, ExportOptions):
            raise TypeError("导出设置必须使用 ExportOptions")
        if options.mode not in self.EXPORT_MODES:
            raise ValueError(f"不支持的导出模式：{options.mode}")
        self._normalized_naming(options)
        image_format = self._normalized_format(options)
        if options.mode == self.MODE_CUSTOM_SPEC:
            dpi = self._positive_int(options.dpi, "DPI")
            width = self._positive_int(options.width, "画布宽度")
            height = self._positive_int(options.height, "画布高度")
            self._validate_size(width, height)
            metadata = self._glyph.get_metadata()
            library_width = self._positive_int(
                metadata.get("画布宽", 250), "字库画布宽度"
            )
            library_height = self._positive_int(
                metadata.get("画布高", 250), "字库画布高度"
            )
            self._custom_scale_ratio(
                width,
                height,
                library_width,
                library_height,
            )
            return ExportOptions(
                mode=options.mode,
                dpi=dpi,
                width=width,
                height=height,
                include_transparent_area=bool(options.include_transparent_area),
                name_mode=options.name_mode,
                sequence_mode=options.sequence_mode,
                image_format=image_format,
                allow_upscale=bool(options.allow_upscale),
            )
        return ExportOptions(
            mode=options.mode,
            include_transparent_area=bool(options.include_transparent_area),
            name_mode=options.name_mode,
            sequence_mode=options.sequence_mode,
            image_format=image_format,
            allow_upscale=bool(options.allow_upscale),
        )

    def _normalized_naming(self, options: ExportOptions) -> tuple[str, bool]:
        """返回新序号方式及是否为旧接口调用。"""
        if options.sequence_mode is not None:
            mode = str(options.sequence_mode)
            if mode not in self.SEQUENCE_MODES:
                raise ValueError(f"不支持的序号编制方式：{mode}")
            return mode, False
        legacy_mode = str(options.name_mode or "字符")
        if legacy_mode in {"字符", "原文件名"}:
            return legacy_mode, True
        if legacy_mode in self.SEQUENCE_MODES:
            return legacy_mode, False
        raise ValueError(f"不支持的序号编制方式：{legacy_mode}")

    def _normalized_format(self, options: ExportOptions) -> str:
        image_format = str(options.image_format or "PNG").upper()
        if image_format not in self.IMAGE_FORMATS:
            raise ValueError(f"不支持的导出格式：{image_format}")
        return image_format

    @staticmethod
    def _custom_scale_ratio(
        width: int,
        height: int,
        library_width: int,
        library_height: int,
    ) -> float:
        """校验自定义宽高等比约束，并返回唯一的完整画布缩放比例。"""
        ratio = width / library_width
        expected_height = max(1, round(library_height * ratio))
        if height != expected_height:
            raise ValueError("自定义画布必须保持字库参数的宽高比例")
        return ratio

    def _render_new_mode(
        self, source: Image.Image, options: ExportOptions
    ) -> tuple[Image.Image, int]:
        self._validate_source_size(source)
        source_rgba = source if source.mode == "RGBA" else source.convert("RGBA")
        owns_source_rgba = source_rgba is not source
        try:
            bounding_box = source_rgba.getchannel("A").getbbox()
            if bounding_box is None:
                raise ValueError("成品图像没有可见文字")
            metadata = self._glyph.get_metadata()
            metadata_dpi = self._positive_int(
                metadata.get("DPI", metadata.get("分辨率", 300)), "字库 DPI"
            )
            source_dpi = self._image_dpi(source, metadata_dpi)

            if options.mode == self.MODE_TRIM_TRANSPARENT:
                return source_rgba.crop(bounding_box), source_dpi

            if options.mode == self.MODE_LIBRARY_SPEC:
                return source_rgba.copy(), source_dpi

            width = self._positive_int(options.width, "画布宽度")
            height = self._positive_int(options.height, "画布高度")
            dpi = self._positive_int(options.dpi, "DPI")
            library_width = self._positive_int(
                metadata.get("画布宽", 250), "字库画布宽度"
            )
            library_height = self._positive_int(
                metadata.get("画布高", 250), "字库画布高度"
            )
            self._validate_size(width, height)
            ratio = self._custom_scale_ratio(
                width,
                height,
                library_width,
                library_height,
            )
            if ratio > 1.0 and not options.allow_upscale:
                return self._center_on_minimum_transparent_canvas(
                    source_rgba,
                    width,
                    height,
                ), dpi
            return self._scale_complete_product(
                source_rgba,
                ratio,
            ), dpi
        finally:
            if owns_source_rgba:
                source_rgba.close()

    @staticmethod
    def _image_dpi(source: Image.Image, fallback: int) -> int:
        value = source.info.get("dpi", (fallback, fallback))
        try:
            dpi = round(float(value[0]))
        except (IndexError, TypeError, ValueError):
            return fallback
        return dpi if dpi > 0 else fallback

    @classmethod
    def _center_on_minimum_transparent_canvas(
        cls,
        source: Image.Image,
        target_width: int,
        target_height: int,
    ) -> Image.Image:
        """把自定义宽高作为最小画布，完整保留更大的最终成品画布。"""
        output_width = max(target_width, source.width)
        output_height = max(target_height, source.height)
        cls._validate_generated_size(output_width, output_height)
        output = Image.new(
            "RGBA",
            (output_width, output_height),
            (0, 0, 0, 0),
        )
        left = (output_width - source.width) // 2
        top = (output_height - source.height) // 2
        output.alpha_composite(source, (left, top))
        return output

    def _render_legacy_mode(
        self,
        source: Image.Image,
        transparent_background: bool,
        output_style: str,
    ) -> tuple[Image.Image, str, int]:
        self._validate_source_size(source)
        source_rgba = source if source.mode == "RGBA" else source.convert("RGBA")
        try:
            output = self._apply_output_style(source_rgba, output_style)
            metadata = self._glyph.get_metadata()
            dpi = self._positive_int(
                metadata.get("DPI", metadata.get("分辨率", 300)), "字库 DPI"
            )
            if transparent_background:
                return output, "PNG", dpi
            background = Image.new("RGB", output.size, "white")
            background.paste(output.convert("RGB"), mask=output.getchannel("A"))
            output.close()
            return background, "BMP", dpi
        finally:
            if source_rgba is not source:
                source_rgba.close()

    @classmethod
    def _scale_complete_product(
        cls,
        source: Image.Image,
        ratio: float,
    ) -> Image.Image:
        """按字库参数得到的单一比例缩放完整成品及其透明画布。"""
        source_width, source_height = source.size
        if source_width <= 0 or source_height <= 0:
            raise ValueError("成品图像尺寸无效")
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError("自定义缩放比例无效")
        scaled_width = max(1, round(source_width * ratio))
        scaled_height = max(1, round(source_height * ratio))
        cls._validate_generated_size(scaled_width, scaled_height)
        if (scaled_width, scaled_height) == source.size:
            return source.copy()
        # 预乘 Alpha 后缩放，避免透明边缘的 RGB 颜色污染抗锯齿像素。
        premultiplied = source.convert("RGBa")
        try:
            resized = premultiplied.resize(
                (scaled_width, scaled_height),
                Image.Resampling.LANCZOS,
            )
        finally:
            premultiplied.close()
        try:
            return resized.convert("RGBA")
        finally:
            resized.close()

    @staticmethod
    def _save_output(
        image: Image.Image,
        path: str,
        image_format: str,
        dpi: int,
    ) -> None:
        image.save(path, image_format, dpi=(float(dpi), float(dpi)))

    @classmethod
    def _commit_staging(
        cls,
        staging_dir: str,
        destination_dir: str,
        items: list[_ExportItem],
    ) -> None:
        for item in items:
            cls._safe_child_path(
                staging_dir,
                item.destination_name,
                "导出暂存文件",
            )
            cls._safe_child_path(
                destination_dir,
                item.destination_name,
                "导出目标文件",
            )
        if not os.path.exists(destination_dir):
            os.replace(staging_dir, destination_dir)
            return
        targets: list[tuple[_ExportItem, str]] = []
        for item in items:
            target_path = cls._safe_child_path(
                destination_dir,
                item.destination_name,
                "导出目标文件",
            )
            if item.overwrite_conflict is None:
                if os.path.exists(target_path):
                    raise FileExistsError(
                        f"导出目录新出现同名文件：{item.destination_name}"
                    )
            else:
                if not os.path.isfile(target_path):
                    raise FileExistsError(
                        f"准备覆盖的同名文件已不存在：{item.destination_name}"
                    )
                current = cls._make_export_conflict(
                    item.detail,
                    item.destination_name,
                    target_path,
                )
                if not cls._same_export_conflict(item.overwrite_conflict, current):
                    raise FileExistsError(
                        f"同名文件在导出期间发生变化：{item.destination_name}"
                    )
            targets.append((item, target_path))

        backup_dir: str | None = None
        backups: dict[str, str] = {}
        installed: list[tuple[_ExportItem, str]] = []
        try:
            overwrite_targets = [
                (item, target_path)
                for item, target_path in targets
                if item.overwrite_conflict is not None
            ]
            if overwrite_targets:
                backup_dir = tempfile.mkdtemp(
                    prefix=".fonteditor_export_backup_",
                    dir=os.path.dirname(destination_dir),
                )
                for index, (_item, target_path) in enumerate(overwrite_targets):
                    backup_path = os.path.join(backup_dir, f"{index:08d}.bak")
                    shutil.copy2(target_path, backup_path)
                    backups[target_path] = backup_path

            for item, target_path in targets:
                source_path = cls._safe_child_path(
                    staging_dir,
                    item.destination_name,
                    "导出暂存文件",
                )
                if item.overwrite_conflict is not None:
                    current = cls._make_export_conflict(
                        item.detail,
                        item.destination_name,
                        target_path,
                    )
                    if not cls._same_export_conflict(
                        item.overwrite_conflict,
                        current,
                    ):
                        raise FileExistsError(
                            f"同名文件在提交前发生变化：{item.destination_name}"
                        )
                    os.replace(source_path, target_path)
                else:
                    if os.path.exists(target_path):
                        raise FileExistsError(
                            f"导出目录新出现同名文件：{item.destination_name}"
                        )
                    os.rename(source_path, target_path)
                installed.append((item, target_path))
        except Exception as exc:
            rollback_errors: list[str] = []
            for item, path in reversed(installed):
                try:
                    if item.overwrite_conflict is not None:
                        backup_path = backups.get(path, "")
                        if not backup_path or not os.path.isfile(backup_path):
                            raise FileNotFoundError("找不到被覆盖文件的备份")
                        os.replace(backup_path, path)
                    elif os.path.isfile(path):
                        os.remove(path)
                except OSError as rollback_exc:
                    rollback_errors.append(f"无法恢复导出目标 {path}：{rollback_exc}")
            if rollback_errors:
                backup_hint = f"；备份保留在 {backup_dir}" if backup_dir else ""
                raise RuntimeError(
                    "导出提交失败，且回滚未完全完成："
                    + "；".join(rollback_errors)
                    + backup_hint
                ) from exc
            if backup_dir and os.path.isdir(backup_dir):
                shutil.rmtree(backup_dir, ignore_errors=True)
            raise
        else:
            if backup_dir and os.path.isdir(backup_dir):
                shutil.rmtree(backup_dir, ignore_errors=True)

    @staticmethod
    def _deterministic_dest_name(
        target_char: str,
        original_name: str,
        variant_index: int,
        sequence_mode: str,
        extension: str,
        max_group_count: int,
    ) -> str:
        if sequence_mode == "原文件名":
            stem = os.path.splitext(os.path.basename(original_name))[0] or "字形"
            return f"{stem}{extension}"
        if sequence_mode == "字符":
            suffix = "" if variant_index == 0 else f"-{variant_index + 1:04d}"
            return f"{target_char}{suffix}{extension}"
        if sequence_mode == "自动等宽序号":
            width = len(str(max(1, max_group_count)))
            return f"{target_char}-{variant_index + 1:0{width}d}{extension}"
        suffix = "" if variant_index == 0 else f"-{variant_index}"
        return f"{target_char}{suffix}{extension}"

    @classmethod
    def _allocate_legacy_name(
        cls,
        output_dir: str,
        filename: str,
        reserved: set[str],
    ) -> str:
        cls._validate_destination_name(filename)
        stem, extension = os.path.splitext(filename)
        sequence = 0
        while True:
            candidate = filename if sequence == 0 else f"{stem}-{sequence}{extension}"
            cls._validate_destination_name(candidate)
            candidate_path = cls._safe_child_path(
                output_dir,
                candidate,
                "导出目标文件",
            )
            normalized = cls._destination_key(candidate)
            if (
                normalized not in reserved
                and not os.path.exists(candidate_path)
            ):
                reserved.add(normalized)
                return candidate
            sequence += 1

    @staticmethod
    def _batch_failure_details(
        items: list[_ExportItem], failed_item: _ExportItem, message: str
    ) -> list[tuple[str, str]]:
        failed_id = str(failed_item.detail.get("变体ID", ""))
        details: list[tuple[str, str]] = [(failed_id, message)]
        for item in items:
            if item is failed_item:
                continue
            variant_id = str(item.detail.get("变体ID", ""))
            details.append((variant_id, "同批次存在失败字形，本字未导出。"))
        return details

    @staticmethod
    def _result(
        success: int,
        skipped: int,
        failure: int,
        output_dir: str,
        *,
        cancelled: bool = False,
        overwritten: int = 0,
        failure_details: Optional[list[tuple[str, str]]] = None,
    ) -> dict[str, Any]:
        return {
            "成功": int(success),
            "跳过": int(skipped),
            "失败": int(failure),
            "已取消": bool(cancelled),
            "覆盖": int(overwritten),
            "失败详情": list(failure_details or []),
            "输出目录": output_dir,
        }

    @staticmethod
    def _append_issue(
        issues: list[dict[str, str]],
        detail: dict[str, Any],
        issue_type: str,
        message: str = "",
    ) -> None:
        issues.append(
            {
                "变体ID": str(detail.get("变体ID", "")),
                "字形": str(detail.get("归属字", "")),
                "类型": issue_type,
                "说明": message,
            }
        )

    @staticmethod
    def _readiness_reasons(
        total: int,
        counts: dict[str, int],
        geometry_completed: bool,
        ink_enabled: bool,
        ink_completed: bool,
        ink_reasons: Iterable[str] = (),
    ) -> list[str]:
        reasons: list[str] = []
        if total <= 0:
            reasons.append("字库中没有文字图片")
        labels = (
            "待优化",
            "待审核",
            "待协调",
            "状态异常",
            "成品缺失",
            "成品损坏",
            "路径无效",
            "校验不符",
        )
        for label in labels:
            count = int(counts.get(label, 0))
            if count:
                reasons.append(f"{label} {count} 个")
        if not geometry_completed:
            reasons.append("整体协调的一致性调整尚未全部完成")
        if ink_enabled and not ink_completed:
            reasons.append("已启用墨色统一，但尚未全部完成")
        reasons.extend(str(reason) for reason in ink_reasons if str(reason))
        return reasons

    @staticmethod
    def _audit_ink_summary(
        summary: Mapping[str, Any],
        library_total: int,
        *,
        enabled: bool,
        completed: bool,
    ) -> tuple[dict[str, Any], list[str]]:
        """只审计已保存结果，不在导出阶段重新计算或改变墨色。"""
        method = str(summary.get("墨色方法", "") or "").strip()
        raw_version = summary.get("墨色方法版本")
        version: int | None = None
        if not isinstance(raw_version, bool):
            try:
                candidate = int(raw_version)
            except (TypeError, ValueError):
                pass
            else:
                if candidate > 0 and candidate == raw_version:
                    version = candidate

        count_keys = ("总数", "已达标", "待确认", "人工例外")
        raw_counts = summary.get("墨色统计")
        counts: dict[str, int] = {key: 0 for key in count_keys}
        counts_valid = isinstance(raw_counts, Mapping)
        if counts_valid:
            for key in count_keys:
                value = raw_counts.get(key)
                if isinstance(value, bool):
                    counts_valid = False
                    break
                try:
                    number = int(value)
                except (TypeError, ValueError):
                    counts_valid = False
                    break
                if number < 0 or number != value:
                    counts_valid = False
                    break
                counts[key] = number

        unaccounted = 0
        reasons: list[str] = []
        if enabled:
            if not method or version is None:
                reasons.append("墨色统一记录缺少新方法及版本，请重新执行整体协调")
            if not counts_valid:
                reasons.append("墨色统一统计缺失或格式不正确，请重新执行整体协调")
            else:
                recorded_total = counts["总数"]
                accounted = (
                    counts["已达标"]
                    + counts["待确认"]
                    + counts["人工例外"]
                )
                if recorded_total != library_total:
                    reasons.append(
                        f"墨色统一统计仅覆盖 {recorded_total} 个，与全库 {library_total} 个不一致"
                    )
                if accounted > recorded_total:
                    reasons.append("墨色统一统计数量相互矛盾，请重新执行整体协调")
                else:
                    unaccounted = recorded_total - accounted
                    if unaccounted:
                        reasons.append(f"墨色未达标 {unaccounted} 个")
                if counts["待确认"]:
                    reasons.append(f"墨色待确认 {counts['待确认']} 个")

        return {
            "墨色方法": method,
            "墨色方法版本": version,
            "墨色统计": dict(counts),
            "墨色已达标": counts["已达标"],
            "墨色待确认": counts["待确认"],
            "墨色人工例外": counts["人工例外"],
            "墨色未达标": unaccounted,
        }, reasons

    @classmethod
    def _validate_size(cls, width: int, height: int) -> None:
        if (
            width <= 0
            or height <= 0
            or width > cls.MAX_DIMENSION
            or height > cls.MAX_DIMENSION
            or width * height > cls.MAX_PIXELS
        ):
            raise ValueError(
                f"导出画布过大，单边不得超过 {cls.MAX_DIMENSION} 像素，"
                f"总像素不得超过 {cls.MAX_PIXELS}"
            )

    @classmethod
    def _validate_generated_size(cls, width: int, height: int) -> None:
        """限制由扩展画布或缩放产生的实际输出，防止异常尺寸耗尽内存。"""
        if (
            width <= 0
            or height <= 0
            or width > cls.MAX_SOURCE_DIMENSION
            or height > cls.MAX_SOURCE_DIMENSION
            or width * height > cls.MAX_PIXELS
        ):
            raise ValueError(
                "导出结果画布过大，"
                f"单边不得超过 {cls.MAX_SOURCE_DIMENSION} 像素，"
                f"总像素不得超过 {cls.MAX_PIXELS}"
            )

    @classmethod
    def _validate_source_size(cls, source: Image.Image) -> None:
        width, height = source.size
        if (
            width <= 0
            or height <= 0
            or width > cls.MAX_SOURCE_DIMENSION
            or height > cls.MAX_SOURCE_DIMENSION
            or width * height > cls.MAX_SOURCE_PIXELS
        ):
            raise ValueError(
                "成品源图尺寸过大，"
                f"单边不得超过 {cls.MAX_SOURCE_DIMENSION} 像素，"
                f"总像素不得超过 {cls.MAX_SOURCE_PIXELS}"
            )

    @staticmethod
    def _has_visible_content(source: Image.Image) -> bool:
        bands = source.getbands()
        if "A" in bands:
            alpha = source.getchannel("A")
            try:
                return alpha.getbbox() is not None
            finally:
                alpha.close()
        if "transparency" in source.info:
            rgba = source.convert("RGBA")
            try:
                alpha = rgba.getchannel("A")
                try:
                    return alpha.getbbox() is not None
                finally:
                    alpha.close()
            finally:
                rgba.close()
        return source.size[0] > 0 and source.size[1] > 0

    @staticmethod
    def _positive_int(value: object, label: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{label} 必须是正整数")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} 必须是正整数") from exc
        if not math.isfinite(number) or number <= 0 or not number.is_integer():
            raise ValueError(f"{label} 必须是正整数")
        return int(number)

    @staticmethod
    def _real_path(path: str) -> str:
        return os.path.normcase(os.path.realpath(os.path.abspath(path)))

    @staticmethod
    def _validate_destination_name(
        filename: str,
        *,
        label: str = "导出文件名",
    ) -> str:
        name = str(filename)
        if (
            not name
            or name in {".", ".."}
            or "\x00" in name
            or "/" in name
            or "\\" in name
            or os.path.isabs(name)
            or os.path.basename(name) != name
        ):
            raise ValueError(f"{label}不安全：必须是不含路径的文件名")
        return name

    @classmethod
    def _safe_child_path(
        cls,
        directory: str,
        filename: str,
        label: str,
    ) -> str:
        name = cls._validate_destination_name(filename)
        absolute_directory = os.path.abspath(directory)
        directory_root = cls._real_path(absolute_directory)
        candidate = os.path.abspath(os.path.join(absolute_directory, name))
        resolved_candidate = cls._real_path(candidate)
        if (
            resolved_candidate == directory_root
            or not cls._is_within(resolved_candidate, directory_root)
        ):
            raise ValueError(f"{label}超出允许目录：{name}")
        return candidate

    @staticmethod
    def _is_within(path: str, directory: str) -> bool:
        try:
            return os.path.commonpath((path, directory)) == directory
        except ValueError:
            return False

    @staticmethod
    def _apply_output_style(image: Image.Image, output_style: str) -> Image.Image:
        alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)
        if output_style == "纯二值":
            output_alpha = np.where(alpha >= 16, 255, 0).astype(np.uint8)
        elif output_style == "统一软边":
            hard_mask = Image.fromarray(
                np.where(alpha >= 32, 255, 0).astype(np.uint8), "L"
            )
            output_alpha = np.asarray(
                hard_mask.filter(ImageFilter.GaussianBlur(radius=0.7)),
                dtype=np.uint8,
            )
        else:
            output_alpha = alpha
        result = Image.new("RGBA", image.size, (0, 0, 0, 0))
        result.putalpha(Image.fromarray(output_alpha, "L"))
        return result

    @staticmethod
    def _make_dest_name(
        target_char: str,
        original_name: str,
        variant_index: int,
        name_mode: str,
        extension: str,
        max_group_count: int = 1,
    ) -> str:
        """兼容旧命名入口，并支持新版普通序号与自动等宽序号。"""
        if name_mode == "原文件名":
            stem = os.path.splitext(os.path.basename(original_name))[0] or "字形"
            return f"{stem}{extension}"
        if name_mode == "自动等宽序号":
            width = len(str(max(1, max_group_count)))
            return f"{target_char}-{variant_index + 1:0{width}d}{extension}"
        suffix = "" if variant_index == 0 else f"-{variant_index}"
        return f"{target_char}{suffix}{extension}"

    @classmethod
    def _unique_path(cls, output_dir: str, filename: str) -> str:
        """保留旧工具接口，新三模式不会静默改变确定性文件名。"""
        cls._validate_destination_name(filename)
        candidate = cls._safe_child_path(output_dir, filename, "导出目标文件")
        if not os.path.exists(candidate):
            return candidate
        stem, extension = os.path.splitext(filename)
        sequence = 1
        while True:
            candidate_name = f"{stem}-{sequence}{extension}"
            candidate = cls._safe_child_path(
                output_dir,
                candidate_name,
                "导出目标文件",
            )
            if not os.path.exists(candidate):
                return candidate
            sequence += 1

    def _report(self, message: str, current: int, total: int) -> None:
        if self._progress:
            self._progress(message, current, total)
