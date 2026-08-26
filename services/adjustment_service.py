# adjustment_service.py — 字库整体协调与成品生成

import hashlib
import json
import math
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Optional

import cv2
import numpy as np
from PIL import Image, ImageFilter

import config
from core.transform_renderer import (
    TransformLimits,
    calculate_transform_geometry,
    compose_rgba_on_canvas,
    place_transform,
    quad_is_valid,
    render_transformed_rgba,
)
from services.glyph_service import GlyphService
from data.log_manager import write_log
from services.batch_persistence import acquire_batch_library_lock
from services.file_transaction_recovery import (
    FileChange,
    FileTransaction,
    ensure_file_transactions_ready,
    recovery_variant_batch_state_snapshot,
)
from utils.batch_observability import BatchTiming
from services.workflow_status_service import resolve_safe_stage_file
from utils.file_utils import (
    compute_file_md5,
    is_safe_windows_filename,
    pinyin_natural_key,
)


class CoordinationCancelled(RuntimeError):
    """整体协调在最终提交前按用户请求安全停止。"""


@dataclass(frozen=True)
class CoordinationPreview:
    """兼容原二元预览契约，并附带真实控制四边形。"""

    image: Image.Image
    bounds: tuple[int, int, int, int]
    control_polygon: tuple[tuple[float, float], ...]

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter((self.image, self.bounds))

    def __getitem__(self, index: int):  # type: ignore[no-untyped-def]
        return (self.image, self.bounds)[index]

    def __len__(self) -> int:
        return 2


class AdjustmentService:
    """统一审核字形的几何、墨色、边缘风格与成品画布。"""

    INK_METHOD = "白底视觉墨量有效前景第70百分位比例增益"
    INK_METHOD_VERSION = 2
    INK_TOLERANCE = 3.0
    INK_CORE_THRESHOLD = 16
    INK_MODE_FOLLOW = "跟随全库"
    INK_MODE_KEEP = "保留本字"
    INK_MODE_EXCEPTION = "人工例外"
    COORDINATION_SCALE_MIN = 0.05
    COORDINATION_SCALE_MAX = 5.0
    COORDINATION_MOVE_LIMIT = 8_192.0
    COORDINATION_DISTORT_LIMIT = 8_192.0
    COORDINATION_MAX_DIMENSION = 16_384
    COORDINATION_MAX_PIXELS = 64 * 1024 * 1024
    COORDINATION_RENDER_VERSION = 1

    def __init__(self, glyph_service: GlyphService) -> None:
        self._glyph = glyph_service

    def load_reviewed_variants(self, pinyin_order: bool = True) -> list[dict[str, Any]]:
        variants = self._glyph.get_variants_by_status(config.STATUS_REVIEWED, config.STATUS_FINISHED)
        if pinyin_order:
            variants.sort(key=lambda item: pinyin_natural_key(str(item.get("归属字", ""))))
        return variants

    def analyze(
        self,
        target_ratio: Optional[float] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        adjustments_by_id: Optional[dict[str, dict[str, Any]]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        *,
        source_images_by_id: Optional[dict[str, Image.Image]] = None,
    ) -> dict[str, Any]:
        """统计审核结果；提供几何参数时按最终几何后的视觉墨量取基准。"""
        width_ratios: list[float] = []
        height_ratios: list[float] = []
        ink_values: list[float] = []
        variants = self.load_reviewed_variants()
        total = len(variants)
        metadata = self._glyph.get_metadata()
        canvas_width = max(1, int(metadata.get("画布宽", 250)))
        canvas_height = max(1, int(metadata.get("画布高", 250)))
        for current, detail in enumerate(variants, 1):
            self._raise_if_coordination_cancelled(cancel_check)
            variant_id = str(detail.get("变体ID", ""))
            source_override = (source_images_by_id or {}).get(variant_id)
            image = (
                source_override.copy()
                if source_override is not None
                else self._load_reviewed_image(detail)
            )
            if image is not None:
                try:
                    if self._upstream_ink_issue(image):
                        if progress_callback is not None:
                            label = str(detail.get("归属字", ""))
                            filename = str(detail.get("原始文件", ""))
                            progress_callback(
                                current,
                                total,
                                f"{label} · {filename}".strip(" ·"),
                            )
                        self._raise_if_coordination_cancelled(cancel_check)
                        continue
                    working = self.prepare_ink_working_copy(
                        image,
                        {"启用": True, "基准": None},
                    )
                    bounding_box = self._ink_bounding_box(working)
                    if bounding_box:
                        width_ratios.append(
                            (bounding_box[2] - bounding_box[0]) / max(image.width, 1)
                        )
                        height_ratios.append(
                            (bounding_box[3] - bounding_box[1]) / max(image.height, 1)
                        )
                        ink_image = working
                        if adjustments_by_id is not None:
                            applied = self._normalized_coordination(
                                adjustments_by_id.get(variant_id)
                            )
                            source_left = (canvas_width - working.width) // 2
                            source_top = (canvas_height - working.height) // 2
                            center_x = source_left + (
                                bounding_box[0] + bounding_box[2]
                            ) / 2.0
                            center_y = source_top + (
                                bounding_box[1] + bounding_box[3]
                            ) / 2.0
                            glyph, content_origin, _polygon = self._render_coordination_glyph(
                                working.crop(bounding_box),
                                applied,
                                (center_x, center_y),
                            )
                            rendered = compose_rgba_on_canvas(
                                np.asarray(glyph, dtype=np.uint8),
                                content_origin,
                                (canvas_width, canvas_height),
                                expand_symmetric=True,
                                limits=self._coordination_limits(),
                            )
                            ink_image = Image.fromarray(rendered.pixels, "RGBA")
                        ink = self._coverage_ink_value(
                            np.array(
                                ink_image.getchannel("A"),
                                dtype=np.uint8,
                                copy=True,
                            )
                        )
                        if ink is not None:
                            ink_values.append(ink)
                finally:
                    image.close()
            if progress_callback is not None:
                label = str(detail.get("归属字", ""))
                filename = str(detail.get("原始文件", ""))
                progress_callback(current, total, f"{label} · {filename}".strip(" ·"))
            self._raise_if_coordination_cancelled(cancel_check)
        valid_count = len(width_ratios)
        if valid_count == 0:
            return {
                "有效数": 0,
                "墨色有效数": 0,
                "目标占比": 0.72,
                "宽中位": 0.0,
                "高中位": 0.0,
                "墨色基准": 220.0,
                "墨色方法": self.INK_METHOD,
                "墨色方法版本": self.INK_METHOD_VERSION,
                "墨色前景阈值": self.INK_CORE_THRESHOLD,
                "墨色统计阶段": "几何变换后" if adjustments_by_id is not None else "审核稿",
            }
        median_width = float(np.median(width_ratios))
        median_height = float(np.median(height_ratios))
        default_ratio = max(0.35, min(0.9, max(median_width, median_height)))
        ink_baseline = float(np.median(ink_values)) if ink_values else 220.0
        return {
            "有效数": valid_count,
            "墨色有效数": len(ink_values),
            "目标占比": round(float(target_ratio if target_ratio is not None else default_ratio), 4),
            "宽中位": round(median_width, 4),
            "高中位": round(median_height, 4),
            "墨色基准": round(ink_baseline, 2),
            "墨色方法": self.INK_METHOD,
            "墨色方法版本": self.INK_METHOD_VERSION,
            "墨色前景阈值": self.INK_CORE_THRESHOLD,
            "墨色统计阶段": "几何变换后" if adjustments_by_id is not None else "审核稿",
        }

    def preview_variant(
        self,
        detail: dict[str, Any],
        target_ratio: float,
        adjustments: Optional[dict[str, Any]] = None,
        ink_baseline: Optional[float] = None,
    ) -> Optional[tuple[Image.Image, dict[str, Any]]]:
        image = self._load_reviewed_image(detail)
        if image is None:
            return None
        ink_profile = self._normalized_ink_config(
            {"启用": ink_baseline is not None, "基准": ink_baseline}
        )
        ink_profile["像素类型"] = self._classify_ink_pixels(image)
        if self._should_block_upstream_ink(image, ink_profile):
            ink_profile["启用"] = False
        image = self.prepare_ink_working_copy(image, ink_profile)
        metadata = self._glyph.get_metadata()
        canvas_width = max(1, int(metadata.get("画布宽", 250)))
        canvas_height = max(1, int(metadata.get("画布高", 250)))
        bounding_box = self._ink_bounding_box(image)
        if not bounding_box:
            return Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0)), {"缩放": 1.0, "偏移X": 0, "偏移Y": 0}

        # 完整图片（含透明边）先与田字格中心对齐，实际文字包围盒只决定目标大小。
        glyph_width = bounding_box[2] - bounding_box[0]
        glyph_height = bounding_box[3] - bounding_box[1]
        target_width = max(1, int(canvas_width * target_ratio))
        target_height = max(1, int(canvas_height * target_ratio))
        scale = min(target_width / max(glyph_width, 1), target_height / max(glyph_height, 1))
        new_width = max(1, int(round(image.width * scale)))
        new_height = max(1, int(round(image.height * scale)))
        glyph = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        applied = self._normalized_adjustments(adjustments)
        glyph = self._apply_global_transform(glyph, applied)
        left = (
            (canvas_width - glyph.width) // 2
            + int(round(applied["移动X"]))
        )
        top = (
            (canvas_height - glyph.height) // 2
            + int(round(applied["移动Y"]))
        )
        output_bbox = glyph.getchannel("A").getbbox()
        if output_bbox:
            glyph_left = left + output_bbox[0]
            glyph_top = top + output_bbox[1]
            glyph_right = left + output_bbox[2]
            glyph_bottom = top + output_bbox[3]
            expand_x = int(math.ceil(max(0, -glyph_left, glyph_right - canvas_width)))
            expand_y = int(math.ceil(max(0, -glyph_top, glyph_bottom - canvas_height)))
        else:
            expand_x = expand_y = 0
        finished_image = Image.new(
            "RGBA",
            (canvas_width + expand_x * 2, canvas_height + expand_y * 2),
            (0, 0, 0, 0),
        )
        finished_image.alpha_composite(
            glyph, (left + expand_x, top + expand_y)
        )
        finished_image, ink_record = self._apply_ink_coordination(
            finished_image,
            ink_profile,
        )
        parameters = {
            "缩放": round(scale, 6),
            "偏移X": left + expand_x,
            "偏移Y": top + expand_y,
            "对称扩展X": expand_x,
            "对称扩展Y": expand_y,
            "标准画布": [canvas_width, canvas_height],
            "实际画布": list(finished_image.size),
            "目标占比": round(target_ratio, 4),
            "原包围盒": list(bounding_box),
            "整体变换": applied,
            "墨色基准": round(float(ink_baseline), 2) if ink_baseline is not None else None,
            "墨色协调": ink_record,
        }
        return finished_image, parameters

    def preview_coordinated(
        self,
        detail: dict[str, Any],
        adjustments: Optional[dict[str, Any]] = None,
        work_ratio: float = 1.3,
        ink_config: Optional[dict[str, Any]] = None,
    ) -> Optional[CoordinationPreview]:
        """生成整体协调预览，并返回与精调画布一致的控制四边形。"""
        source = self._load_reviewed_image(detail)
        if source is None:
            return None
        variant_id = str(detail.get("变体ID", ""))
        ink_profile = self._ink_config_for_variant(ink_config, variant_id)
        ink_profile["像素类型"] = self._classify_ink_pixels(source)
        if self._should_block_upstream_ink(source, ink_profile):
            ink_profile["启用"] = False
        source = self.prepare_ink_working_copy(source, ink_profile)
        metadata = self._glyph.get_metadata()
        grid_width = max(1, int(metadata.get("画布宽", 250)))
        grid_height = max(1, int(metadata.get("画布高", 250)))
        work_width = max(grid_width, int(round(grid_width * work_ratio)))
        work_height = max(grid_height, int(round(grid_height * work_ratio)))
        source_left = (work_width - source.width) // 2
        source_top = (work_height - source.height) // 2
        bounding_box = self._ink_bounding_box(source)
        if not bounding_box:
            return CoordinationPreview(
                image=Image.new(
                    "RGBA",
                    (work_width, work_height),
                    (0, 0, 0, 0),
                ),
                bounds=(0, 0, 0, 0),
                control_polygon=(),
            )
        center_x = source_left + (bounding_box[0] + bounding_box[2]) / 2.0
        center_y = source_top + (bounding_box[1] + bounding_box[3]) / 2.0
        applied = self._normalized_coordination(adjustments)
        glyph, content_origin, control_polygon = self._render_coordination_glyph(
            source.crop(bounding_box),
            applied,
            (center_x, center_y),
        )
        rendered = compose_rgba_on_canvas(
            np.asarray(glyph, dtype=np.uint8),
            content_origin,
            (work_width, work_height),
            expand_symmetric=False,
            limits=self._coordination_limits(),
        )
        preview = Image.fromarray(rendered.pixels, "RGBA")
        preview, _ink_record = self._apply_ink_coordination(preview, ink_profile)
        preview_alpha = preview.getchannel("A").point([0] * 16 + [255] * 240)
        preview_bbox = preview_alpha.getbbox()
        if preview_bbox is None:
            return CoordinationPreview(preview, (0, 0, 0, 0), control_polygon)
        return CoordinationPreview(
            image=preview,
            bounds=tuple(int(value) for value in preview_bbox),
            control_polygon=control_polygon,
        )

    def save_coordinated_variants(
        self,
        variants: list[dict[str, Any]],
        adjustments_by_id: dict[str, dict[str, Any]],
        ink_config: Optional[dict[str, Any]] = None,
        coordination_baseline: Optional[dict[str, Any]] = None,
        progress_callback: Optional[
            Callable[[str, int, int, int, str], None]
        ] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        commit_gate: Optional[Callable[[], bool]] = None,
        *,
        source_images_by_id: Optional[dict[str, Image.Image]] = None,
        include_metrics: bool = False,
    ) -> dict[str, Any]:
        """以批次事务生成成品，任一失败都不留下部分保存结果。"""
        library_lock = acquire_batch_library_lock(self._glyph.ziku_dir)
        try:
            ensure_file_transactions_ready(self._glyph.ziku_dir)
            return self._save_coordinated_variants_locked(
                variants,
                adjustments_by_id,
                ink_config,
                coordination_baseline,
                progress_callback,
                cancel_check,
                commit_gate,
                source_images_by_id=source_images_by_id,
                include_metrics=include_metrics,
            )
        finally:
            library_lock.release()

    def _save_coordinated_variants_locked(
        self,
        variants: list[dict[str, Any]],
        adjustments_by_id: dict[str, dict[str, Any]],
        ink_config: Optional[dict[str, Any]] = None,
        coordination_baseline: Optional[dict[str, Any]] = None,
        progress_callback: Optional[
            Callable[[str, int, int, int, str], None]
        ] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        commit_gate: Optional[Callable[[], bool]] = None,
        *,
        source_images_by_id: Optional[dict[str, Image.Image]] = None,
        include_metrics: bool = False,
    ) -> dict[str, Any]:
        """调用方持有字库独占锁时完成整批渲染与持久提交。"""
        source_images_by_id = source_images_by_id or {}
        if not variants:
            return {"成功": 0, "失败": 0, "失败详情": []}
        total = len(variants)
        timing = BatchTiming()
        preparation_started = time.perf_counter()
        self._raise_if_coordination_cancelled(cancel_check)
        self._report_coordination_progress(
            progress_callback,
            "准备",
            0,
            0,
            total,
            "正在核对批次",
        )
        normalized_ink = self._normalized_ink_config(ink_config)
        self._validate_ink_baseline_scope(variants, normalized_ink)
        if normalized_ink["启用"] and (
            normalized_ink["重算几何后基准"] or normalized_ink["基准"] is None
        ):
            requested_ratio = None
            if isinstance(coordination_baseline, dict):
                requested_ratio = coordination_baseline.get("目标占比")
            formal_baseline = self.analyze(
                target_ratio=requested_ratio,
                adjustments_by_id=adjustments_by_id,
                cancel_check=cancel_check,
                source_images_by_id=source_images_by_id,
            )
            normalized_ink["基准"] = formal_baseline["墨色基准"]
            coordination_baseline = formal_baseline
        metadata = self._glyph.get_metadata()
        canvas_width = max(1, int(metadata.get("画布宽", 250)))
        canvas_height = max(1, int(metadata.get("画布高", 250)))
        target_dpi = float(metadata.get("DPI", metadata.get("分辨率", 300)) or 300)
        finished_dir = self._glyph.get_workflow_dirs()["成品"]
        os.makedirs(finished_dir, exist_ok=True)
        prepared: list[dict[str, Any]] = []
        temporary_paths: list[str] = []
        preparation_failures: dict[int, str] = {}
        reserved_targets: set[str] = set()
        plans: dict[int, dict[str, Any]] = {}

        for index, detail in enumerate(variants):
            self._raise_if_coordination_cancelled(cancel_check)
            label = self._coordination_progress_label(detail)
            try:
                raw_variant_id = detail.get("变体ID")
                variant_id = (
                    "" if raw_variant_id is None else str(raw_variant_id).strip()
                )
                if not variant_id:
                    raise ValueError("字形缺少变体ID，不能生成整体协调成品")
                raw_source_name = detail.get("原始文件")
                source_value = (
                    "" if raw_source_name is None else str(raw_source_name).strip()
                )
                source_name = os.path.basename(source_value)
                if (
                    not source_name
                    or source_value != source_value.strip()
                    or source_name != source_value
                    or ":" in source_value
                ):
                    raise ValueError("字形缺少原始文件名，不能生成整体协调成品")
                filename = os.path.splitext(source_name)[0] + ".png"
                path = os.path.join(finished_dir, filename)
                normalized_path = os.path.normcase(os.path.abspath(path))
                if normalized_path in reserved_targets:
                    raise ValueError(f"多个字形使用了同一成品文件名：{filename}")
                reserved_targets.add(normalized_path)
                plans[index] = {
                    "detail": detail,
                    "variant_id": variant_id,
                    "filename": filename,
                    "target_path": path,
                    "label": label,
                }
            except Exception as exc:
                preparation_failures[index] = str(exc)
            self._report_coordination_progress(
                progress_callback,
                "准备",
                round((index + 1) * 10 / total),
                index + 1,
                total,
                label,
            )
            self._raise_if_coordination_cancelled(cancel_check)

        if preparation_failures:
            timing.add("准备", time.perf_counter() - preparation_started)
            write_log(
                timing.format_summary(
                    "整体协调保存",
                    {"请求": total, "成功": 0, "失败": len(preparation_failures)},
                )
            )
            return self._coordination_batch_failure_result(
                variants,
                preparation_failures,
            )

        timing.add("准备", time.perf_counter() - preparation_started)
        rendering_started = time.perf_counter()
        render_requests: list[
            tuple[
                int,
                dict[str, Any],
                dict[str, Any],
                dict[str, Any],
                dict[str, Any],
                str,
                Image.Image | None,
            ]
        ] = []
        for index, detail in enumerate(variants):
            self._raise_if_coordination_cancelled(cancel_check, temporary_paths)
            label = self._coordination_progress_label(detail)
            self._raise_if_coordination_cancelled(cancel_check, temporary_paths)
            plan = plans.get(index)
            if plan is None:
                continue
            variant_id = str(plan["variant_id"])
            try:
                applied = self._normalized_coordination(
                    adjustments_by_id.get(variant_id)
                )
                variant_ink = self._ink_config_for_variant(
                    normalized_ink,
                    variant_id,
                )
                generation_signature = self._coordination_generation_signature(
                    detail,
                    applied,
                    variant_ink,
                    (canvas_width, canvas_height),
                    target_dpi,
                )
                if (
                    variant_id not in source_images_by_id
                    and self._reusable_coordination_output(
                    detail,
                    str(plan["target_path"]),
                    str(plan["filename"]),
                    generation_signature,
                    )
                ):
                    prepared.append(
                        {
                            "detail": detail,
                            "variant_id": variant_id,
                            "filename": str(plan["filename"]),
                            "target_path": str(plan["target_path"]),
                            "temporary_path": "",
                            "md5": str(detail.get("成品MD5", "")),
                            "parameters": detail.get("整体协调参数", {}),
                            "label": label,
                            "reused": True,
                        }
                    )
                    continue
                render_requests.append(
                    (
                        index,
                        detail,
                        plan,
                        applied,
                        variant_ink,
                        generation_signature,
                        source_images_by_id.get(variant_id),
                    )
                )
            except Exception as exc:
                preparation_failures[index] = str(exc)
            self._raise_if_coordination_cancelled(cancel_check, temporary_paths)
            self._raise_if_coordination_cancelled(cancel_check, temporary_paths)

        completed_renders = 0
        reused_count = len(prepared)
        render_cancelled = False
        if reused_count:
            self._report_coordination_progress(
                progress_callback,
                "渲染",
                10 + round(reused_count * 75 / total),
                reused_count,
                total,
                "正在复用未变化成品",
            )
        if render_requests and not preparation_failures:
            worker_count = min(2, len(render_requests))
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="整体协调渲染",
            ) as executor:
                futures = {
                    executor.submit(
                        self._render_coordination_item,
                        detail,
                        plan,
                        applied,
                        variant_ink,
                        generation_signature,
                        (canvas_width, canvas_height),
                        target_dpi,
                        source_override,
                    ): (index, str(plan["label"]))
                    for (
                        index,
                        detail,
                        plan,
                        applied,
                        variant_ink,
                        generation_signature,
                        source_override,
                    ) in render_requests
                }
                for future in as_completed(futures):
                    index, label = futures[future]
                    if future.cancelled():
                        continue
                    try:
                        item = future.result()
                        prepared.append(item)
                        temporary_paths.append(str(item["temporary_path"]))
                    except Exception as exc:
                        preparation_failures[index] = str(exc)
                    completed_renders += 1
                    self._report_coordination_progress(
                        progress_callback,
                        "渲染",
                        10 + round(
                            (reused_count + completed_renders) * 75 / total
                        ),
                        reused_count + completed_renders,
                        total,
                        label,
                    )
                    if cancel_check is not None and cancel_check():
                        render_cancelled = True
                        for pending in futures:
                            pending.cancel()
        if render_cancelled:
            self._remove_paths(temporary_paths)
            raise CoordinationCancelled("已停止，本批次未提交")
        plan_order = {
            str(plan["variant_id"]): index for index, plan in plans.items()
        }
        prepared.sort(key=lambda item: plan_order[str(item["variant_id"])])

        if preparation_failures:
            timing.add("渲染编码", time.perf_counter() - rendering_started)
            self._remove_paths(temporary_paths)
            write_log(
                timing.format_summary(
                    "整体协调保存",
                    {"请求": total, "成功": 0, "失败": len(preparation_failures)},
                )
            )
            return self._coordination_batch_failure_result(
                variants,
                preparation_failures,
            )

        timing.add("渲染编码", time.perf_counter() - rendering_started)
        changed_items = [item for item in prepared if not item.get("reused")]
        review_items: list[dict[str, Any]] = []
        review_dir = self._glyph.get_workflow_dirs()["手工审核"]
        os.makedirs(review_dir, exist_ok=True)
        for item in changed_items:
            variant_id = str(item["variant_id"])
            source_override = source_images_by_id.get(variant_id)
            if source_override is None:
                continue
            detail = item["detail"]
            review_filename = os.path.basename(
                str(detail.get("审核文件", "") or "")
            )
            if not review_filename:
                review_filename = os.path.splitext(
                    os.path.basename(str(detail.get("原始文件", "")))
                )[0] + ".png"
            review_target = os.path.join(review_dir, review_filename)
            review_temp = self._save_coordination_temp_png(
                source_override,
                review_target,
                (target_dpi, target_dpi),
            )
            review_items.append(
                {
                    "variant_id": variant_id,
                    "detail": detail,
                    "filename": review_filename,
                    "target_path": review_target,
                    "temporary_path": review_temp,
                    "md5": compute_file_md5(review_temp),
                }
            )
            temporary_paths.append(review_temp)
        transaction_preparation_started = time.perf_counter()
        try:
            baseline = (
                dict(coordination_baseline)
                if coordination_baseline is not None
                else self.analyze(
                    adjustments_by_id=adjustments_by_id,
                    source_images_by_id=source_images_by_id,
                )
            )
            state_ids = {
                str(item["variant_id"]) for item in changed_items
            } | {str(item["variant_id"]) for item in review_items}
            state_backups = [self._glyph.snapshot_variant_state(variant_id) for variant_id in state_ids]
            self._raise_if_coordination_cancelled(cancel_check)
            # 快照准备完成后再原子关闭取消窗口。通过后必须完整提交或完整回滚，
            # 不再观察取消标志，避免把成品文件和 JSON 留在半提交状态。
            if commit_gate is not None:
                if not commit_gate():
                    raise CoordinationCancelled("已停止，本批次未提交")
            else:
                self._raise_if_coordination_cancelled(cancel_check)
        except Exception:
            self._remove_paths(temporary_paths)
            raise
        timing.add(
            "事务准备",
            time.perf_counter() - transaction_preparation_started,
        )
        if not changed_items:
            self._report_coordination_progress(
                progress_callback,
                "提交",
                100,
                total,
                total,
                "成品未变化，已直接复用",
            )
            write_log(
                timing.format_summary(
                    "整体协调保存",
                    {"请求": total, "成功": total, "复用": total, "失败": 0},
                )
            )
            return self._coordination_success_result(
                total,
                reused=total,
                include_metrics=include_metrics,
            )
        transaction: FileTransaction | None = None
        state_persisted = False
        commit_started = time.perf_counter()
        try:
            self._report_coordination_progress(
                progress_callback,
                "提交",
                85,
                0,
                total,
                "正在备份原成品",
            )
            transaction = FileTransaction.begin(
                self._glyph.ziku_dir,
                [
                    FileChange(
                        target_path=str(item["target_path"]),
                        temporary_path=str(item["temporary_path"]),
                        new_md5=str(item["md5"]),
                        backup_prefix=".fonteditor_coordination_rollback_",
                    )
                    for item in changed_items
                ]
                + [
                    FileChange(
                        target_path=str(item["target_path"]),
                        temporary_path=str(item["temporary_path"]),
                        new_md5=str(item["md5"]),
                        backup_prefix=".fonteditor_review_rollback_",
                    )
                    for item in review_items
                ],
                recovery_variant_batch_state_snapshot(state_backups),
            )
            transaction.backup_targets()

            for item in review_items:
                self._glyph.mark_manual_saved(
                    str(item["variant_id"]),
                    str(item["filename"]),
                    str(item["md5"]),
                )
                for rendered_item in changed_items:
                    if str(rendered_item["variant_id"]) != str(item["variant_id"]):
                        continue
                    rendered_item["parameters"]["生成签名"] = (
                        self._coordination_generation_signature(
                            self._glyph.get_variant(str(item["variant_id"])),
                            rendered_item["parameters"].get("整体变换", {}),
                            self._ink_config_for_variant(
                                normalized_ink,
                                str(item["variant_id"]),
                            ),
                            (canvas_width, canvas_height),
                            target_dpi,
                        )
                    )
                    break
            for index, item in enumerate(changed_items):
                self._glyph.mark_finished(
                    str(item["variant_id"]),
                    str(item["filename"]),
                    str(item["md5"]),
                    item["parameters"],
                )
                self._report_coordination_progress(
                    progress_callback,
                    "提交",
                    93 + round((index + 1) * 6 / total),
                    index + 1,
                    total,
                    str(item["label"]),
                )
            self._refresh_coordination_summary(
                baseline,
                normalized_ink,
                verify_variant_ids={
                    str(item["variant_id"]) for item in changed_items
                },
                pending_finished_paths={
                    os.path.normcase(os.path.abspath(str(item["target_path"])))
                    for item in changed_items
                },
            )
            transaction.mark_rollforward(
                recovery_variant_batch_state_snapshot(
                    self._glyph.snapshot_variant_state(str(item["variant_id"]))
                    for item in changed_items
                )
            )
            transaction.install_new_files()
            self._report_coordination_progress(
                progress_callback,
                "提交",
                99,
                total,
                total,
                "正在提交字库索引",
            )
            self._glyph.save()
            state_persisted = True
            cleanup_errors = transaction.finalize()
            if cleanup_errors:
                write_log(
                    "整体协调批次事务已提交，清理将于下次打开继续｜"
                    + "；".join(cleanup_errors)
                )
        except Exception as exc:
            if state_persisted:
                raise
            for snapshot in state_backups:
                self._glyph.restore_variant_state(snapshot)
            rollback_errors = transaction.rollback() if transaction is not None else []
            if rollback_errors:
                details = "；".join(rollback_errors)
                raise RuntimeError(
                    f"整体协调保存失败，且回滚未完全完成：{details}"
                ) from exc
            raise
        finally:
            self._remove_paths(temporary_paths)
        timing.add("事务提交", time.perf_counter() - commit_started)

        self._report_coordination_progress(
            progress_callback,
            "提交",
            100,
            total,
            total,
            "批次提交完成",
        )
        write_log(
            timing.format_summary(
                "整体协调保存",
                {
                    "请求": total,
                    "成功": len(prepared),
                    "复用": len(prepared) - len(changed_items),
                    "失败": 0,
                },
            )
        )
        return self._coordination_success_result(
            len(prepared),
            reused=len(prepared) - len(changed_items),
            include_metrics=include_metrics,
        )

    def _validate_ink_baseline_scope(
        self,
        variants: list[dict[str, Any]],
        ink_config: dict[str, Any],
    ) -> None:
        """禁止部分批次重算或替换已经生效的全库墨色基准。"""

        if not ink_config.get("启用"):
            return
        submitted_ids = {
            str(detail.get("变体ID", "")).strip()
            for detail in variants
            if str(detail.get("变体ID", "")).strip()
        }
        eligible_ids = {
            str(detail.get("变体ID", "")).strip()
            for detail in self.load_reviewed_variants(pinyin_order=False)
            if str(detail.get("变体ID", "")).strip()
        }
        full_scope = bool(eligible_ids) and submitted_ids == eligible_ids
        if ink_config.get("重算几何后基准") and not full_scope:
            raise ValueError(
                "重新计算全库墨色基准必须提交全部可协调字形，不能只处理部分字形。"
            )

        summary = self._glyph.get_coordination_summary()
        if summary.get("墨色统一启用") is not True:
            return
        current_baseline = self._finite_ink_baseline(summary.get("墨色基准"))
        requested_baseline = self._finite_ink_baseline(ink_config.get("基准"))
        if current_baseline is None or requested_baseline is None:
            return
        current_method = str(summary.get("墨色方法", "") or "").strip()
        requested_method = str(ink_config.get("方法", "") or "").strip()
        try:
            current_version = int(summary.get("墨色方法版本"))
            requested_version = int(ink_config.get("方法版本"))
        except (TypeError, ValueError):
            current_version = requested_version = 0
        contract_changed = (
            not math.isclose(
                current_baseline,
                requested_baseline,
                rel_tol=0.0,
                abs_tol=0.01,
            )
            or current_method != requested_method
            or current_version != requested_version
        )
        if contract_changed and not full_scope:
            raise ValueError(
                "全库墨色基准或算法契约发生变化，必须提交全部可协调字形。"
            )

    @staticmethod
    def _finite_ink_baseline(value: object) -> float | None:
        if isinstance(value, bool):
            return None
        try:
            baseline = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return baseline if math.isfinite(baseline) and 0.0 < baseline <= 255.0 else None

    @staticmethod
    def _coordination_success_result(
        success: int,
        *,
        reused: int,
        include_metrics: bool,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "成功": int(success),
            "失败": 0,
            "失败详情": [],
        }
        if include_metrics:
            normalized_reused = max(0, min(int(reused), int(success)))
            result["复用"] = normalized_reused
            result["重新生成"] = max(0, int(success) - normalized_reused)
        return result

    @classmethod
    def _raise_if_coordination_cancelled(
        cls,
        cancel_check: Optional[Callable[[], bool]],
        temporary_paths: Optional[list[str]] = None,
    ) -> None:
        if cancel_check is None or not cancel_check():
            return
        if temporary_paths:
            cls._remove_paths(temporary_paths)
        raise CoordinationCancelled("已停止，本批次未提交")

    @staticmethod
    def _coordination_progress_label(detail: dict[str, Any]) -> str:
        char = str(detail.get("归属字", "")).strip()
        filename = str(detail.get("原始文件", "")).strip()
        return " · ".join(item for item in (char, filename) if item) or "未命名字形"

    @staticmethod
    def _coordination_batch_failure_result(
        variants: list[dict[str, Any]],
        failures_by_index: dict[int, str],
    ) -> dict[str, Any]:
        failure_details = []
        for index, detail in enumerate(variants):
            raw_variant_id = detail.get("变体ID")
            variant_id = (
                "" if raw_variant_id is None else str(raw_variant_id).strip()
            )
            failure_details.append(
                (
                    variant_id,
                    failures_by_index.get(
                        index,
                        "同批次存在失败字形，本字未写入成品。",
                    ),
                )
            )
        return {
            "成功": 0,
            "失败": len(failure_details),
            "失败详情": failure_details,
        }

    @staticmethod
    def _report_coordination_progress(
        callback: Optional[Callable[[str, int, int, int, str], None]],
        stage: str,
        percent: int,
        current: int,
        total: int,
        glyph_label: str,
    ) -> None:
        if callback is None:
            return
        try:
            callback(
                str(stage),
                max(0, min(100, int(percent))),
                max(0, int(current)),
                max(0, int(total)),
                str(glyph_label),
            )
        except Exception:
            pass

    @staticmethod
    def _save_coordination_temp_png(
        image: Image.Image,
        target_path: str,
        dpi: tuple[float, float],
    ) -> str:
        """在成品目录写完整临时 PNG，提交前不覆盖任何现有文件。"""
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=".fonteditor_coordination_",
            suffix=".png",
            dir=os.path.dirname(target_path),
        )
        os.close(descriptor)
        try:
            image.save(temporary_path, "PNG", dpi=dpi)
        except Exception:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
            raise
        return temporary_path

    def _render_coordination_item(
        self,
        detail: dict[str, Any],
        plan: dict[str, Any],
        applied: dict[str, Any],
        variant_ink: dict[str, Any],
        generation_signature: str,
        canvas_size: tuple[int, int],
        target_dpi: float,
        source_override: Image.Image | None = None,
    ) -> dict[str, Any]:
        """渲染并验证一个临时成品，不修改字库状态。"""

        temporary_path = ""
        try:
            canvas_width, canvas_height = canvas_size
            source = (
                source_override.copy()
                if source_override is not None
                else self._load_reviewed_image(detail)
            )
            if source is None:
                raise FileNotFoundError("找不到审核通过的文字图片")
            if not self._has_visible_ink(source):
                raise ValueError("审核通过的文字图片没有有效文字前景")
            variant_ink = dict(variant_ink)
            variant_ink["像素类型"] = self._classify_ink_pixels(source)
            upstream_issue = self._upstream_ink_issue(source)
            if (
                upstream_issue
                and variant_ink["启用"]
                and variant_ink["模式"] == self.INK_MODE_FOLLOW
            ):
                raise ValueError(
                    f"前序图像异常：{upstream_issue}；"
                    "请重新自动优化并完成手工审核后再整体协调"
                )
            source = self.prepare_ink_working_copy(source, variant_ink)
            source_left = (canvas_width - source.width) // 2
            source_top = (canvas_height - source.height) // 2
            bounding_box = self._ink_bounding_box(source)
            if not bounding_box:
                raise ValueError("审核通过的文字图片没有有效文字前景")
            glyph = source.crop(bounding_box)
            center_x = source_left + (bounding_box[0] + bounding_box[2]) / 2.0
            center_y = source_top + (bounding_box[1] + bounding_box[3]) / 2.0
            glyph, content_origin, _control_polygon = self._render_coordination_glyph(
                glyph,
                applied,
                (center_x, center_y),
            )
            rendered = compose_rgba_on_canvas(
                np.asarray(glyph, dtype=np.uint8),
                content_origin,
                canvas_size,
                expand_symmetric=True,
                limits=self._coordination_limits(),
            )
            finished = Image.fromarray(rendered.pixels, "RGBA")
            finished, ink_record = self._apply_ink_coordination(
                finished,
                variant_ink,
            )
            expand_x, expand_y = rendered.geometry.grid_origin
            temporary_path = self._save_coordination_temp_png(
                finished,
                str(plan["target_path"]),
                (target_dpi, target_dpi),
            )
            saved = self._open_rgba(temporary_path)
            if saved is None:
                raise OSError("临时成品 PNG 写入后无法重新解码")
            try:
                self._record_saved_ink_verification(ink_record, saved)
            finally:
                saved.close()
            parameters = {
                "标准画布": [canvas_width, canvas_height],
                "实际画布": list(finished.size),
                "对称扩展X": expand_x,
                "对称扩展Y": expand_y,
                "整体变换": applied,
                "原包围盒": list(bounding_box),
                "墨色协调": ink_record,
                "生成签名": generation_signature,
            }
            return {
                "detail": detail,
                "variant_id": str(plan["variant_id"]),
                "filename": str(plan["filename"]),
                "target_path": str(plan["target_path"]),
                "temporary_path": temporary_path,
                "md5": compute_file_md5(temporary_path),
                "parameters": parameters,
                "label": str(plan["label"]),
                "reused": False,
            }
        except Exception:
            if temporary_path and os.path.exists(temporary_path):
                try:
                    os.remove(temporary_path)
                except OSError:
                    pass
            raise

    def _coordination_generation_signature(
        self,
        detail: dict[str, Any],
        adjustments: dict[str, Any],
        ink_config: dict[str, Any],
        canvas_size: tuple[int, int],
        dpi: float,
    ) -> str:
        reviewed_filename = str(detail.get("审核文件", "") or "")
        workflow_dirs = self._glyph.get_workflow_dirs()
        source_path = resolve_safe_stage_file(
            workflow_dirs["手工审核" if reviewed_filename else "优化预览"],
            reviewed_filename or detail.get("中间文件"),
        )
        if not source_path:
            return ""
        source_md5 = str(
            detail.get("审核MD5" if reviewed_filename else "中间MD5", "")
            or ""
        ).lower()
        if not source_md5:
            source_md5 = compute_file_md5(source_path)
        if not source_md5:
            return ""
        profile = self._normalized_ink_config(ink_config)
        payload = {
            "版本": self.COORDINATION_RENDER_VERSION,
            "源图MD5": source_md5,
            "画布": [int(canvas_size[0]), int(canvas_size[1])],
            "DPI": round(float(dpi), 4),
            "整体变换": self._normalized_coordination(adjustments),
            "墨色": {
                key: profile.get(key)
                for key in (
                    "启用",
                    "基准",
                    "方法",
                    "方法版本",
                    "前景阈值",
                    "容差",
                    "模式",
                )
            },
        }
        packed = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(packed).hexdigest()

    @staticmethod
    def _reusable_coordination_output(
        detail: dict[str, Any],
        target_path: str,
        filename: str,
        generation_signature: str,
    ) -> bool:
        parameters = detail.get("整体协调参数")
        return bool(
            generation_signature
            and str(detail.get("状态", "")) == config.STATUS_FINISHED
            and str(detail.get("成品文件", "")) == filename
            and str(detail.get("成品MD5", ""))
            and isinstance(parameters, dict)
            and parameters.get("生成签名") == generation_signature
            and os.path.isfile(target_path)
            and os.path.getsize(target_path) > 0
        )

    @staticmethod
    def _reserve_coordination_backup(target_path: str) -> str:
        """在同目录预留旧成品回滚路径，确保替换不跨磁盘。"""
        descriptor, backup_path = tempfile.mkstemp(
            prefix=".fonteditor_coordination_rollback_",
            suffix=os.path.splitext(target_path)[1],
            dir=os.path.dirname(target_path),
        )
        os.close(descriptor)
        return backup_path

    @staticmethod
    def _rollback_coordination_files(
        installed_paths: list[str],
        backup_paths: list[tuple[str, str]],
    ) -> list[str]:
        """删除本批新文件并恢复全部旧成品，返回无法恢复的错误。"""
        errors: list[str] = []
        for target_path in reversed(installed_paths):
            try:
                if os.path.exists(target_path):
                    os.remove(target_path)
            except OSError as exc:
                errors.append(f"无法移除新成品 {target_path}：{exc}")
        for target_path, backup_path in reversed(backup_paths):
            try:
                if os.path.exists(backup_path):
                    os.replace(backup_path, target_path)
            except OSError as exc:
                errors.append(f"无法恢复旧成品 {target_path}：{exc}")
        return errors

    @staticmethod
    def _remove_paths(paths: list[str]) -> None:
        """尽力清理已完成或已回滚事务的临时文件。"""
        for path in paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass

    @classmethod
    def _normalized_coordination(cls, adjustments: Optional[dict[str, Any]]) -> dict[str, Any]:
        source = adjustments if isinstance(adjustments, dict) else {}
        raw_distort = source.get("扭曲", [0.0] * 8)
        if not isinstance(raw_distort, (list, tuple)) or len(raw_distort) != 8:
            raw_distort = [0.0] * 8
        has_uniform_scale = "等比缩放" in source
        uniform_scale = cls._bounded_number(
            source.get("等比缩放", 1.0),
            cls.COORDINATION_SCALE_MIN,
            cls.COORDINATION_SCALE_MAX,
            1.0,
        )
        legacy_scale_x = cls._bounded_number(
            source.get("缩放X", source.get("额外缩放", 1.0)),
            cls.COORDINATION_SCALE_MIN,
            cls.COORDINATION_SCALE_MAX,
            1.0,
        )
        legacy_scale_y = cls._bounded_number(
            source.get("缩放Y", source.get("额外缩放", 1.0)),
            cls.COORDINATION_SCALE_MIN,
            cls.COORDINATION_SCALE_MAX,
            1.0,
        )
        has_legacy_scale_x = "缩放X" in source or "额外缩放" in source
        has_legacy_scale_y = "缩放Y" in source or "额外缩放" in source

        if "水平拉伸" in source:
            stretch_w = cls._bounded_number(
                source["水平拉伸"],
                cls.COORDINATION_SCALE_MIN,
                cls.COORDINATION_SCALE_MAX,
                1.0,
            )
        elif has_uniform_scale and has_legacy_scale_x:
            stretch_w = cls._bounded_number(
                legacy_scale_x / uniform_scale,
                cls.COORDINATION_SCALE_MIN,
                cls.COORDINATION_SCALE_MAX,
                1.0,
            )
        else:
            stretch_w = legacy_scale_x

        if "垂直拉伸" in source:
            stretch_h = cls._bounded_number(
                source["垂直拉伸"],
                cls.COORDINATION_SCALE_MIN,
                cls.COORDINATION_SCALE_MAX,
                1.0,
            )
        elif has_uniform_scale and has_legacy_scale_y:
            stretch_h = cls._bounded_number(
                legacy_scale_y / uniform_scale,
                cls.COORDINATION_SCALE_MIN,
                cls.COORDINATION_SCALE_MAX,
                1.0,
            )
        else:
            stretch_h = legacy_scale_y

        return {
            "移动X": cls._bounded_number(
                source.get("移动X", 0.0),
                -cls.COORDINATION_MOVE_LIMIT,
                cls.COORDINATION_MOVE_LIMIT,
                0.0,
            ),
            "移动Y": cls._bounded_number(
                source.get("移动Y", 0.0),
                -cls.COORDINATION_MOVE_LIMIT,
                cls.COORDINATION_MOVE_LIMIT,
                0.0,
            ),
            "等比缩放": uniform_scale,
            "水平拉伸": stretch_w,
            "垂直拉伸": stretch_h,
            # 保留派生字段，供旧页面和已保存字库继续读取。
            "缩放X": uniform_scale * stretch_w,
            "缩放Y": uniform_scale * stretch_h,
            "旋转": cls._bounded_number(source.get("旋转", 0.0), -180.0, 180.0, 0.0),
            "斜切X": cls._bounded_number(source.get("斜切X", 0.0), -25.0, 25.0, 0.0),
            "斜切Y": cls._bounded_number(source.get("斜切Y", 0.0), -25.0, 25.0, 0.0),
            "扭曲": [
                cls._bounded_number(
                    value,
                    -cls.COORDINATION_DISTORT_LIMIT,
                    cls.COORDINATION_DISTORT_LIMIT,
                    0.0,
                )
                for value in raw_distort
            ],
        }

    @staticmethod
    def _bounded_number(value: Any, minimum: float, maximum: float, default: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(number):
            return default
        return max(minimum, min(maximum, number))

    @classmethod
    def coordination_to_canvas_transform(
        cls,
        adjustments: Optional[dict[str, Any]],
        content_size: tuple[int, int],
    ) -> dict[str, Any]:
        """把协调参数转换为 ``ReviewCanvas.set_transform`` 可直接使用的参数。"""
        applied = cls._normalized_coordination(adjustments)
        try:
            source_width = max(1, int(content_size[0]))
            source_height = max(1, int(content_size[1]))
        except (IndexError, TypeError, ValueError):
            source_width = source_height = 1
        scaled_width = max(1, int(round(source_width * applied["缩放X"])))
        scaled_height = max(1, int(round(source_height * applied["缩放Y"])))
        cls._validate_coordination_size(scaled_width, scaled_height)

        shear_x = math.tan(math.radians(applied["斜切X"]))
        shear_y = math.tan(math.radians(applied["斜切Y"]))
        denominator = 1.0 - shear_x * shear_y
        shear_distort = np.zeros((4, 2), dtype=np.float64)
        if not math.isclose(denominator, 0.0, abs_tol=1e-9):
            source_quad = np.asarray(
                [
                    [0.0, 0.0],
                    [float(scaled_width - 1), 0.0],
                    [float(scaled_width - 1), float(scaled_height - 1)],
                    [0.0, float(scaled_height - 1)],
                ],
                dtype=np.float64,
            )
            forward = np.asarray(
                [[1.0, shear_x], [shear_y, 1.0]],
                dtype=np.float64,
            ) / denominator
            sheared_quad = source_quad @ forward.T
            # ReviewCanvas 会保持四边形几何中心不动，因此这里只编码形状差异。
            sheared_quad += source_quad.mean(axis=0) - sheared_quad.mean(axis=0)
            shear_distort = sheared_quad - source_quad

        raw_distort = np.asarray(applied["扭曲"], dtype=np.float64).reshape(4, 2)
        combined_distort = (shear_distort + raw_distort).reshape(-1)
        return {
            "x": applied["移动X"],
            "y": applied["移动Y"],
            "scale": applied["等比缩放"],
            "stretch_w": applied["水平拉伸"],
            "stretch_h": applied["垂直拉伸"],
            "rotation": applied["旋转"],
            "distort": [
                cls._bounded_number(
                    value,
                    -cls.COORDINATION_DISTORT_LIMIT,
                    cls.COORDINATION_DISTORT_LIMIT,
                    0.0,
                )
                for value in combined_distort
            ],
        }

    @classmethod
    def coordination_from_canvas_transform(
        cls,
        transform: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """把 ReviewCanvas 状态写回协调参数，并以四角扭曲统一表达斜切。"""
        source = transform if isinstance(transform, dict) else {}
        return cls._normalized_coordination(
            {
                "移动X": source.get("x", source.get("移动X", 0.0)),
                "移动Y": source.get("y", source.get("移动Y", 0.0)),
                "等比缩放": source.get("scale", source.get("等比缩放", 1.0)),
                "水平拉伸": source.get("stretch_w", source.get("水平拉伸", 1.0)),
                "垂直拉伸": source.get("stretch_h", source.get("垂直拉伸", 1.0)),
                "旋转": source.get("rotation", source.get("旋转", 0.0)),
                "斜切X": 0.0,
                "斜切Y": 0.0,
                "扭曲": source.get("distort", source.get("扭曲", [0.0] * 8)),
            }
        )

    def load_saved_coordination_adjustments(self, detail: dict[str, Any]) -> dict[str, Any]:
        """读取单字已保存的几何参数；旧成品没有记录时返回默认值。"""
        parameters = detail.get("整体协调参数", {})
        if not isinstance(parameters, dict):
            parameters = {}
        adjustments = parameters.get("整体变换", {})
        return self._normalized_coordination(adjustments if isinstance(adjustments, dict) else None)

    @classmethod
    def _normalized_ink_config(cls, ink_config: Optional[dict[str, Any]]) -> dict[str, Any]:
        source = ink_config or {}
        raw_baseline = source.get("基准", source.get("墨色基准"))
        try:
            baseline = float(raw_baseline) if raw_baseline is not None else None
        except (TypeError, ValueError):
            baseline = None
        if baseline is not None and not math.isfinite(baseline):
            baseline = None
        raw_tolerance = source.get("容差", cls.INK_TOLERANCE)
        try:
            tolerance = float(raw_tolerance)
        except (TypeError, ValueError):
            tolerance = cls.INK_TOLERANCE
        if not math.isfinite(tolerance):
            tolerance = cls.INK_TOLERANCE
        mode = str(source.get("模式", cls.INK_MODE_FOLLOW)).strip()
        if mode not in {cls.INK_MODE_FOLLOW, cls.INK_MODE_KEEP, cls.INK_MODE_EXCEPTION}:
            mode = cls.INK_MODE_FOLLOW
        raw_modes = source.get("逐字模式", {})
        variant_modes = dict(raw_modes) if isinstance(raw_modes, dict) else {}
        return {
            "启用": bool(source.get("启用", False)),
            "基准": round(max(1.0, min(255.0, baseline)), 2) if baseline is not None else None,
            "方法": cls.INK_METHOD,
            "方法版本": cls.INK_METHOD_VERSION,
            "前景阈值": cls.INK_CORE_THRESHOLD,
            "容差": round(max(0.5, min(20.0, tolerance)), 2),
            "模式": mode,
            "逐字模式": variant_modes,
            "重算几何后基准": bool(source.get("重算几何后基准", False)),
            "像素类型": str(source.get("像素类型", "")).strip(),
        }

    @classmethod
    def _ink_config_for_variant(
        cls,
        ink_config: Optional[dict[str, Any]],
        variant_id: str,
    ) -> dict[str, Any]:
        """解析逐字模式，同时保留全库墨色基准和算法参数。"""
        profile = cls._normalized_ink_config(ink_config)
        raw_mode = profile["逐字模式"].get(variant_id, profile["模式"])
        mode = str(raw_mode).strip()
        if mode not in {cls.INK_MODE_FOLLOW, cls.INK_MODE_KEEP, cls.INK_MODE_EXCEPTION}:
            mode = cls.INK_MODE_FOLLOW
        profile["模式"] = mode
        return profile

    @classmethod
    def prepare_ink_working_copy(
        cls,
        image: Image.Image,
        ink_config: Optional[dict[str, Any]],
    ) -> Image.Image:
        """在几何变换前建立黑色 RGB + 白底视觉墨量 Alpha 工作副本。"""
        profile = cls._normalized_ink_config(ink_config)
        if not profile["启用"]:
            return image.copy()
        coverage = cls._visual_coverage(image)
        pixels = np.zeros((*coverage.shape, 4), dtype=np.uint8)
        pixels[..., 3] = coverage
        return Image.fromarray(pixels, "RGBA")

    @classmethod
    def apply_ink_preview(
        cls,
        image: Image.Image,
        ink_config: Optional[dict[str, Any]],
        variant_id: str = "",
    ) -> tuple[Image.Image, dict[str, Any]]:
        """对几何渲染结果执行与正式保存完全相同的墨色处理。"""
        profile = (
            cls._ink_config_for_variant(ink_config, variant_id)
            if variant_id
            else cls._normalized_ink_config(ink_config)
        )
        return cls._apply_ink_coordination(image, profile)

    @classmethod
    def _apply_ink_coordination(
        cls,
        image: Image.Image,
        ink_config: Optional[dict[str, Any]],
    ) -> tuple[Image.Image, dict[str, Any]]:
        """在成品工作副本上统一墨色，并返回可追溯的逐字记录。"""
        profile = cls._normalized_ink_config(ink_config)
        source = image.convert("RGBA")
        pixel_type = profile["像素类型"] or cls._classify_ink_pixels(source)
        before = cls._glyph_ink_value(source)
        target = profile["基准"]
        tolerance = float(profile["容差"])
        record: dict[str, Any] = {
            "启用": bool(profile["启用"]),
            "基准": target,
            "方法": profile["方法"],
            "方法版本": profile["方法版本"],
            "前景阈值": profile["前景阈值"],
            "像素类型": pixel_type,
            "模式": profile["模式"],
            "调整方式": "关闭",
            "Gamma": None,
            "增益": None,
            "调整前墨色": round(before, 2) if before is not None else None,
            "调整后墨色": round(before, 2) if before is not None else None,
            "目标偏差": None,
            "绝对偏差": None,
            "容差": tolerance,
            "已应用": False,
            "是否达标": False,
            "状态": "待确认",
            "人工接受例外": profile["模式"] == cls.INK_MODE_EXCEPTION,
            "触及限制": False,
            "保存后墨色": None,
            "保存后复测": False,
            "跳过原因": "",
        }
        if not profile["启用"]:
            record["状态"] = "未启用"
            record["跳过原因"] = "已关闭墨色统一"
            return source.copy(), record

        working = cls.prepare_ink_working_copy(source, profile)
        record["调整方式"] = "视觉墨量规范化"
        before = cls._coverage_ink_value(
            np.array(working.getchannel("A"), dtype=np.uint8, copy=True)
        )
        record["调整前墨色"] = round(before, 2) if before is not None else None
        record["调整后墨色"] = record["调整前墨色"]
        if target is None:
            record["跳过原因"] = "缺少有效的全库墨色基准"
            return working, record
        if before is None:
            record["状态"] = "像素表达异常"
            record["跳过原因"] = "字形没有有效前景像素"
            return working, record

        mode = str(profile["模式"])
        if mode in {cls.INK_MODE_KEEP, cls.INK_MODE_EXCEPTION}:
            after = before
            difference = after - float(target)
            achieved = abs(difference) <= tolerance
            accepted = mode == cls.INK_MODE_EXCEPTION
            record.update({
                "调整后墨色": round(after, 2),
                "目标偏差": round(difference, 2),
                "绝对偏差": round(abs(difference), 2),
                "已应用": working.tobytes() != source.tobytes(),
                "是否达标": achieved,
                "调整方式": "人工例外" if accepted else "保留本字",
                "状态": (
                    "墨色已达标"
                    if achieved
                    else ("人工接受例外" if accepted else "保留本字，待确认")
                ),
                "跳过原因": "用户已接受墨色例外" if accepted else "用户选择保留本字",
            })
            return working, record

        alpha = np.array(working.getchannel("A"), dtype=np.uint8, copy=True)
        output_alpha, gain, reason = cls._normalize_ink_with_details(
            alpha,
            float(target),
            tolerance=tolerance,
        )
        output = working.copy()
        output.putalpha(Image.fromarray(output_alpha, "L"))
        after = cls._coverage_ink_value(output_alpha)
        difference = after - float(target) if after is not None else None
        achieved = difference is not None and abs(difference) <= tolerance
        applied = output.tobytes() != source.tobytes()
        hit_limit = bool(
            gain is not None
            and (
                (
                    gain > 1.0
                    and np.any((alpha > 0) & (alpha < 255) & (output_alpha == 255))
                )
                or (
                    gain < 1.0
                    and np.any((alpha > 1) & (output_alpha == 1))
                )
            )
        )
        record.update({
            "调整方式": (
                "比例增益"
                if gain is not None and not math.isclose(gain, 1.0, abs_tol=1e-9)
                else "视觉墨量规范化"
            ),
            "增益": round(gain, 6) if gain is not None else None,
            "调整后墨色": round(after, 2) if after is not None else None,
            "目标偏差": round(difference, 2) if difference is not None else None,
            "绝对偏差": round(abs(difference), 2) if difference is not None else None,
            "已应用": applied,
            "是否达标": achieved,
            "状态": "墨色已达标" if achieved else "调整受限，待确认",
            "触及限制": hit_limit,
            "跳过原因": "" if achieved and applied else reason,
        })
        return output, record

    @classmethod
    def _record_saved_ink_verification(
        cls,
        record: dict[str, Any],
        saved_image: Image.Image,
    ) -> None:
        """以重新解码的临时 PNG 更新可审计墨色结论。"""
        saved_ink = cls._glyph_ink_value(saved_image)
        if saved_ink is None:
            raise ValueError("临时成品 PNG 没有可复测的有效文字前景")
        record["保存后墨色"] = round(saved_ink, 2)
        record["保存后复测"] = True
        if record.get("启用") is not True:
            return

        raw_target = record.get("基准")
        try:
            target = float(raw_target)
        except (TypeError, ValueError):
            target = math.nan
        if not math.isfinite(target):
            record["是否达标"] = False
            record["状态"] = "待确认"
            record["跳过原因"] = "保存后复测缺少有效的全库墨色基准"
            return

        tolerance = float(record.get("容差", cls.INK_TOLERANCE))
        difference = saved_ink - target
        achieved = abs(difference) <= tolerance
        record["目标偏差"] = round(difference, 2)
        record["绝对偏差"] = round(abs(difference), 2)
        record["是否达标"] = achieved
        mode = str(record.get("模式", cls.INK_MODE_FOLLOW))
        if mode == cls.INK_MODE_EXCEPTION:
            record["状态"] = "墨色已达标" if achieved else "人工接受例外"
            return
        if mode == cls.INK_MODE_KEEP:
            record["状态"] = "墨色已达标" if achieved else "保留本字，待确认"
            return
        record["状态"] = "墨色已达标" if achieved else "调整受限，待确认"
        if not achieved:
            record["跳过原因"] = "保存后复测未进入墨色容差"

    def _refresh_coordination_summary(
        self,
        baseline: dict[str, Any],
        ink_config: dict[str, Any],
        *,
        verify_variant_ids: set[str] | None = None,
        pending_finished_paths: set[str] | None = None,
    ) -> None:
        """刷新协调摘要；普通保存只深查本次变化的成品。"""
        profile = self._normalized_ink_config(ink_config)
        variants = self.load_reviewed_variants(pinyin_order=False)
        finished_dir = self._glyph.get_workflow_dirs()["成品"]
        verify_ids = verify_variant_ids
        pending_paths = pending_finished_paths or set()
        geometry_completed = bool(variants)
        counts = {"总数": len(variants), "已达标": 0, "待确认": 0, "人工例外": 0}
        for detail in variants:
            filename = str(detail.get("成品文件", ""))
            variant_id = str(detail.get("变体ID", ""))
            expected_path = (
                os.path.normcase(
                    os.path.abspath(os.path.join(finished_dir, filename))
                )
                if is_safe_windows_filename(filename)
                else ""
            )
            if verify_ids is None or variant_id in verify_ids:
                valid_finished_reference = bool(
                    expected_path in pending_paths
                    or resolve_safe_stage_file(finished_dir, filename)
                )
            else:
                # 页面内保存期间字库持有独占锁，页外文件未发生变化；
                # 外部修改由“重新核对字库数据”执行深度审计。
                valid_finished_reference = bool(expected_path)
            has_finished = (
                str(detail.get("状态", "")) == config.STATUS_FINISHED
                and valid_finished_reference
            )
            geometry_completed = geometry_completed and has_finished
            parameters = detail.get("整体协调参数", {})
            record = parameters.get("墨色协调", {}) if isinstance(parameters, dict) else {}
            if not profile["启用"]:
                # 关闭统一墨色也是一种全库输出配置。只重存当前页时，
                # 页外仍启用墨色的旧成品不能与新成品混合后通过导出审计。
                geometry_completed = (
                    geometry_completed
                    and isinstance(record, dict)
                    and record.get("启用") is False
                    and record.get("保存后复测") is True
                )
            record_baseline = record.get("基准") if isinstance(record, dict) else None
            try:
                baseline_matches = (
                    record_baseline is not None
                    and profile["基准"] is not None
                    and math.isclose(float(record_baseline), float(profile["基准"]), abs_tol=0.01)
                )
            except (TypeError, ValueError):
                baseline_matches = False
            common_matches = (
                has_finished
                and isinstance(record, dict)
                and record.get("启用") is True
                and record.get("方法") == profile["方法"]
                and record.get("方法版本") == profile["方法版本"]
                and record.get("保存后复测") is True
                and record.get("保存后墨色") is not None
                and baseline_matches
            )
            accepted = common_matches and record.get("人工接受例外") is True
            achieved = common_matches and record.get("是否达标") is True
            if accepted:
                counts["人工例外"] += 1
            elif achieved:
                counts["已达标"] += 1
            else:
                counts["待确认"] += 1
        ink_completed = (
            bool(variants)
            and bool(profile["启用"])
            and counts["待确认"] == 0
        )
        summary_baseline = dict(baseline)
        summary_baseline["墨色方法"] = profile["方法"]
        summary_baseline["墨色方法版本"] = profile["方法版本"]
        summary_baseline["墨色统计"] = dict(counts)
        self._glyph.set_coordination_summary(
            summary_baseline,
            profile["基准"],
            geometry_completed=geometry_completed,
            ink_completed=ink_completed,
            ink_enabled=bool(profile["启用"]),
            ink_method=profile["方法"],
            ink_method_version=profile["方法版本"],
            ink_counts=counts,
        )

    @classmethod
    def _apply_coordination_transform(cls, image: Image.Image, adjustments: dict[str, Any]) -> Image.Image:
        rendered, _origin, _control_polygon = cls._render_coordination_glyph(
            image,
            adjustments,
            (image.width / 2.0, image.height / 2.0),
        )
        return rendered

    @classmethod
    def _render_coordination_glyph(
        cls,
        image: Image.Image,
        adjustments: dict[str, Any],
        source_center: tuple[float, float],
    ) -> tuple[
        Image.Image,
        tuple[float, float],
        tuple[tuple[float, float], ...],
    ]:
        """使用与 ReviewCanvas 相同的矩阵渲染字形并返回几何信息。"""
        applied = cls._normalized_coordination(adjustments)
        canvas_transform = cls.coordination_to_canvas_transform(applied, image.size)
        geometry = calculate_transform_geometry(
            image.size,
            scale_x=canvas_transform["scale"] * canvas_transform["stretch_w"],
            scale_y=canvas_transform["scale"] * canvas_transform["stretch_h"],
            rotation=canvas_transform["rotation"],
            distort=canvas_transform["distort"],
            limits=cls._coordination_limits(),
        )
        pixels = render_transformed_rgba(
            np.array(image.convert("RGBA"), dtype=np.uint8, copy=True),
            geometry,
            force_rotation=abs(canvas_transform["rotation"]) > 1e-9,
        )
        placement = place_transform(
            geometry,
            source_center,
            (canvas_transform["x"], canvas_transform["y"]),
        )
        polygon = tuple(
            (float(point[0]), float(point[1]))
            for point in placement.polygon
        )
        return Image.fromarray(pixels, "RGBA"), placement.origin, polygon

    def generate_finished(
        self,
        target_ratio: Optional[float] = None,
        adjustments: Optional[dict[str, Any]] = None,
    ) -> dict[str, int]:
        baseline = self.analyze(target_ratio)
        ratio = float(baseline["目标占比"])
        ink_baseline = float(baseline["墨色基准"])
        metadata = self._glyph.get_metadata()
        target_dpi = float(
            metadata.get("DPI", metadata.get("分辨率", 300)) or 300
        )
        finished_dir = self._glyph.get_workflow_dirs()["成品"]
        os.makedirs(finished_dir, exist_ok=True)
        success_count = failure_count = 0
        for detail in self.load_reviewed_variants():
            preview = self.preview_variant(detail, ratio, adjustments, ink_baseline)
            if preview is None:
                failure_count += 1
                continue
            image, parameters = preview
            filename = os.path.splitext(detail.get("原始文件", "字形"))[0] + ".png"
            path = os.path.join(finished_dir, filename)
            try:
                image.save(path, "PNG", dpi=(target_dpi, target_dpi))
                self._glyph.mark_finished(detail["变体ID"], filename, compute_file_md5(path), parameters)
                success_count += 1
            except OSError:
                failure_count += 1
        baseline["整体变换"] = self._normalized_adjustments(adjustments)
        self._refresh_coordination_summary(
            baseline,
            {"启用": True, "基准": ink_baseline},
        )
        self._glyph.save()
        return {"成功": success_count, "失败": failure_count}

    def reviewed_source_path(self, detail: dict[str, Any]) -> str:
        """返回整体协调实际采用的审核源图路径，找不到时返回空字符串。"""
        image, path = self.load_reviewed_source(detail)
        if image is not None:
            image.close()
        return path

    def load_reviewed_source(
        self,
        detail: dict[str, Any],
    ) -> tuple[Optional[Image.Image], str]:
        """单次解码并返回整体协调实际采用的审核源图及路径。"""
        workflow_dirs = self._glyph.get_workflow_dirs()
        reviewed_filename = str(detail.get("审核文件", "") or "")
        if reviewed_filename:
            path = resolve_safe_stage_file(
                workflow_dirs["手工审核"],
                reviewed_filename,
            )
            if not path:
                return None, ""
            image = self._open_rgba(path)
            return (image, path) if image is not None else (None, "")

        preview_path = resolve_safe_stage_file(
            workflow_dirs["优化预览"],
            detail.get("中间文件"),
        )
        if preview_path:
            image = self._open_rgba(preview_path)
            if image is not None:
                return image, preview_path
        return None, ""

    def load_reviewed_image(self, detail: dict[str, Any]) -> Optional[Image.Image]:
        """读取整体协调使用的审核源图，并返回独立的 RGBA 图像。"""
        image, _path = self.load_reviewed_source(detail)
        return image

    def _load_reviewed_image(self, detail: dict[str, Any]) -> Optional[Image.Image]:
        """兼容原有内部调用。"""
        return self.load_reviewed_image(detail)

    @staticmethod
    def _normalized_adjustments(adjustments: Optional[dict[str, Any]]) -> dict[str, float]:
        source = adjustments or {}
        return {
            "移动X": float(source.get("移动X", 0.0)),
            "移动Y": float(source.get("移动Y", 0.0)),
            "额外缩放": float(source.get("额外缩放", 1.0)),
            "旋转": float(source.get("旋转", 0.0)),
            "斜切X": float(source.get("斜切X", 0.0)),
            "斜切Y": float(source.get("斜切Y", 0.0)),
            "扭曲X": float(source.get("扭曲X", 0.0)),
            "扭曲Y": float(source.get("扭曲Y", 0.0)),
        }

    @classmethod
    def _apply_global_transform(cls, image: Image.Image, adjustments: dict[str, Any]) -> Image.Image:
        scale = max(0.5, min(1.5, adjustments["额外缩放"]))
        if not math.isclose(scale, 1.0):
            width = max(1, int(round(image.width * scale)))
            height = max(1, int(round(image.height * scale)))
            cls._validate_coordination_size(width, height)
            image = image.resize((width, height), Image.Resampling.BICUBIC)
        image = cls._apply_shear_transform(image, adjustments)
        image = cls._apply_perspective_transform(image, adjustments)
        return cls._apply_rotation_transform(image, adjustments["旋转"])

    @classmethod
    def _apply_shear_transform(cls, image: Image.Image, adjustments: dict[str, Any]) -> Image.Image:
        shear_x = math.tan(math.radians(max(-25.0, min(25.0, adjustments["斜切X"]))))
        shear_y = math.tan(math.radians(max(-25.0, min(25.0, adjustments["斜切Y"]))))
        if not math.isclose(shear_x, 0.0) or not math.isclose(shear_y, 0.0):
            margin_x = int(abs(shear_x) * image.height) + 2
            margin_y = int(abs(shear_y) * image.width) + 2
            output_size = (image.width + margin_x, image.height + margin_y)
            cls._validate_coordination_size(*output_size)
            image = image.transform(
                output_size,
                Image.Transform.AFFINE,
                (1.0, -shear_x, margin_x // 2, -shear_y, 1.0, margin_y // 2),
                Image.Resampling.BICUBIC,
            )
        return image

    @classmethod
    def _apply_perspective_transform(cls, image: Image.Image, adjustments: dict[str, Any]) -> Image.Image:
        raw_distort = adjustments.get("扭曲")
        if isinstance(raw_distort, (list, tuple)) and len(raw_distort) == 8:
            distort = [
                cls._bounded_number(
                    value,
                    -cls.COORDINATION_DISTORT_LIMIT,
                    cls.COORDINATION_DISTORT_LIMIT,
                    0.0,
                )
                for value in raw_distort
            ]
            if any(not math.isclose(value, 0.0) for value in distort):
                rgba = np.asarray(image, dtype=np.uint8)
                height, width = rgba.shape[:2]
                source = np.empty((4, 2), dtype=np.float32)
                source[0] = (0.0, 0.0)
                source[1] = (float(width - 1), 0.0)
                source[2] = (float(width - 1), float(height - 1))
                source[3] = (0.0, float(height - 1))
                target = source + np.asarray(distort, dtype=np.float32).reshape(4, 2)
                if width < 2 or height < 2 or not cls._coordination_quad_is_valid(target):
                    raise ValueError("扭曲后的控制四边形无效。")
                target[:, 0] -= min(0.0, float(target[:, 0].min()))
                target[:, 1] -= min(0.0, float(target[:, 1].min()))
                output_size = (
                    max(1, int(math.ceil(float(target[:, 0].max()))) + 1),
                    max(1, int(math.ceil(float(target[:, 1].max()))) + 1),
                )
                cls._validate_coordination_size(*output_size)
                matrix = cv2.getPerspectiveTransform(source, target)
                rgba = cv2.warpPerspective(
                    rgba,
                    matrix,
                    output_size,
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0, 0),
                )
                image = Image.fromarray(rgba, "RGBA")
        else:
            distort_x = max(-0.3, min(0.3, float(adjustments.get("扭曲X", 0.0))))
            distort_y = max(-0.3, min(0.3, float(adjustments.get("扭曲Y", 0.0))))
            if not math.isclose(distort_x, 0.0) or not math.isclose(distort_y, 0.0):
                rgba = np.asarray(image, dtype=np.uint8)
                height, width = rgba.shape[:2]
                source = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
                dx = abs(distort_x) * width * 0.5
                dy = abs(distort_y) * height * 0.5
                target = np.float32([
                    [dx if distort_x > 0 else 0, dy if distort_y > 0 else 0],
                    [width - 1 - (dx if distort_x < 0 else 0), dy if distort_y < 0 else 0],
                    [width - 1 - (dx if distort_x > 0 else 0), height - 1 - (dy if distort_y > 0 else 0)],
                    [dx if distort_x < 0 else 0, height - 1 - (dy if distort_y < 0 else 0)],
                ])
                matrix = cv2.getPerspectiveTransform(source, target)
                rgba = cv2.warpPerspective(rgba, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
                image = Image.fromarray(rgba, "RGBA")
        return image

    @classmethod
    def _apply_rotation_transform(cls, image: Image.Image, rotation: Any) -> Image.Image:
        angle = cls._bounded_number(rotation, -180.0, 180.0, 0.0)
        if math.isclose(angle, 0.0):
            return image
        radians = math.radians(angle)
        estimated_width = max(
            1,
            int(math.ceil(abs(image.width * math.cos(radians)) + abs(image.height * math.sin(radians)))) + 2,
        )
        estimated_height = max(
            1,
            int(math.ceil(abs(image.width * math.sin(radians)) + abs(image.height * math.cos(radians)))) + 2,
        )
        cls._validate_coordination_size(estimated_width, estimated_height)
        return image.rotate(-angle, Image.Resampling.BICUBIC, expand=True)

    @classmethod
    def _validate_coordination_size(cls, width: int, height: int) -> None:
        if (
            width < 1
            or height < 1
            or width > cls.COORDINATION_MAX_DIMENSION
            or height > cls.COORDINATION_MAX_DIMENSION
            or width * height > cls.COORDINATION_MAX_PIXELS
        ):
            raise ValueError("变换后的字形尺寸超出安全范围。")

    @classmethod
    def _coordination_limits(cls) -> TransformLimits:
        return TransformLimits(
            max_dimension=cls.COORDINATION_MAX_DIMENSION,
            max_pixels=cls.COORDINATION_MAX_PIXELS,
        )

    @staticmethod
    def _coordination_quad_is_valid(points: np.ndarray) -> bool:
        """拒绝非有限、自交、凹陷和近零面积的透视四边形。"""
        return quad_is_valid(points)

    def _apply_output_style(self, image: Image.Image, style: str, ink_baseline: Optional[float]) -> Image.Image:
        alpha = np.array(image.getchannel("A"), dtype=np.uint8, copy=True)
        if style == "纯二值":
            output_alpha = np.where(alpha >= 16, 255, 0).astype(np.uint8)
        elif style == "统一软边":
            hard_mask = Image.fromarray(np.where(alpha >= 32, 255, 0).astype(np.uint8), "L")
            output_alpha = np.array(
                hard_mask.filter(ImageFilter.GaussianBlur(radius=0.7)),
                dtype=np.uint8,
                copy=True,
            )
        else:
            output_alpha = self._normalize_ink(alpha, ink_baseline)
        result = Image.new("RGBA", image.size, (0, 0, 0, 0))
        result.putalpha(Image.fromarray(output_alpha, "L"))
        return result

    @classmethod
    def _normalize_ink(cls, alpha: np.ndarray, target: Optional[float]) -> np.ndarray:
        normalized, _gamma, _reason = cls._normalize_ink_with_details(alpha, target)
        return normalized

    @classmethod
    def _normalize_ink_with_details(
        cls,
        alpha: np.ndarray,
        target: Optional[float],
        *,
        tolerance: Optional[float] = None,
    ) -> tuple[np.ndarray, Optional[float], str]:
        if not np.any(alpha > 0):
            return alpha.copy(), None, "字形没有有效前景像素"
        if target is None:
            return alpha.copy(), None, "缺少有效的全库墨色基准"
        source = np.asarray(alpha, dtype=np.uint8)
        support = source >= cls.INK_CORE_THRESHOLD
        if not np.any(support):
            support = source > 0
        current = cls._coverage_ink_value(source, support)
        if current is None or current <= 0:
            return alpha.copy(), None, "当前墨色统计无效"
        target_value = max(1.0, min(255.0, float(target)))
        allowed = cls.INK_TOLERANCE if tolerance is None else float(tolerance)
        if abs(current - target_value) <= allowed:
            return alpha.copy(), 1.0, "已在墨色容差内，无需调整"

        gain = target_value / current
        normalized = cls._apply_coverage_gain(source, gain, support)
        measured = cls._coverage_ink_value(normalized, support)
        if measured is not None and abs(measured - target_value) <= allowed:
            return normalized, gain, ""

        # 分位数插值和 uint8 量化可能使直接比例落在相邻整数上，
        # 以单调二分找到实际输出中最接近目标的一档。
        lower = max(1.0 / 255.0, gain * 0.5)
        upper = max(1.0, gain * 2.0)
        candidates: list[tuple[float, float, np.ndarray]] = []
        for _index in range(24):
            probe = (lower + upper) / 2.0
            candidate = cls._apply_coverage_gain(source, probe, support)
            candidate_value = cls._coverage_ink_value(candidate, support)
            if candidate_value is None:
                break
            candidates.append((abs(candidate_value - target_value), probe, candidate))
            if candidate_value < target_value:
                lower = probe
            else:
                upper = probe
        if candidates:
            _difference, gain, normalized = min(candidates, key=lambda item: item[0])
            measured = cls._coverage_ink_value(normalized, support)
        reason = ""
        if measured is None or abs(measured - target_value) > allowed:
            reason = "比例增益受量化或饱和影响，复测未进入墨色容差"
        return normalized, gain, reason

    @staticmethod
    def _apply_coverage_gain(
        alpha: np.ndarray,
        gain: float,
        support: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """只调整固定字形核心，不放大核心之外的低透明背景残留。"""
        source = np.asarray(alpha, dtype=np.uint8)
        active = source > 0 if support is None else np.asarray(support, dtype=bool)
        if active.shape != source.shape:
            raise ValueError("墨色支持域尺寸与图像不一致")
        scaled = np.rint(source.astype(np.float32) * float(gain))
        scaled = np.clip(scaled, 0.0, 255.0).astype(np.uint8)
        output = source.copy()
        output[active] = np.maximum(scaled[active], 1)
        return output

    @classmethod
    def _coverage_ink_value(
        cls,
        coverage: np.ndarray,
        support: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        source = np.asarray(coverage, dtype=np.uint8)
        if support is None:
            values = source[source >= cls.INK_CORE_THRESHOLD]
            if values.size == 0:
                values = source[source > 0]
        else:
            active = np.asarray(support, dtype=bool)
            if active.shape != source.shape:
                raise ValueError("墨色支持域尺寸与图像不一致")
            values = source[active]
        return float(np.percentile(values, 70)) if values.size else None

    @classmethod
    def _should_block_upstream_ink(
        cls,
        image: Image.Image,
        ink_profile: dict[str, Any],
    ) -> bool:
        return bool(
            ink_profile.get("启用")
            and ink_profile.get("模式") == cls.INK_MODE_FOLLOW
            and cls._upstream_ink_issue(image)
        )

    @classmethod
    def _upstream_ink_issue(cls, image: Image.Image) -> str:
        """识别误反相特征，避免把低透明黑背景当作字形墨迹放大。"""
        pixels = np.array(image.convert("RGBA"), dtype=np.uint8, copy=True)
        alpha = pixels[..., 3]
        if alpha.size == 0 or float(np.mean(alpha <= 5)) < 0.02:
            return ""
        maximum = int(alpha.max())
        if maximum <= cls.INK_CORE_THRESHOLD:
            return ""
        core_threshold = max(
            cls.INK_CORE_THRESHOLD,
            min(64, int(math.ceil(maximum * 0.25))),
        )
        core = alpha >= core_threshold
        low_alpha = (alpha > 5) & (alpha < core_threshold)
        core_fraction = float(np.mean(core))
        low_fraction = float(np.mean(low_alpha))
        if (
            not np.any(core)
            or not np.any(low_alpha)
            or core_fraction > 0.65
            or low_fraction < 0.05
        ):
            return ""
        rgb = pixels[..., :3].astype(np.float32)
        luminance = (
            rgb[..., 0] * 0.299
            + rgb[..., 1] * 0.587
            + rgb[..., 2] * 0.114
        )
        core_median = float(np.median(luminance[core]))
        low_median = float(np.median(luminance[low_alpha]))
        low_dark_ratio = float(np.mean(luminance[low_alpha] < 64.0))
        if core_median > 192.0 and low_median < 64.0 and low_dark_ratio >= 0.80:
            return "检测到高不透明浅色笔画和大面积低透明黑色背景，疑似自动优化误反相"
        return ""

    @classmethod
    def _ink_bounding_box(
        cls,
        image: Image.Image,
    ) -> Optional[tuple[int, int, int, int]]:
        coverage = cls._visual_coverage(image)
        mask = coverage >= cls.INK_CORE_THRESHOLD
        if not np.any(mask):
            mask = coverage > 0
        if not np.any(mask):
            return None
        return Image.fromarray(mask.astype(np.uint8) * 255, "L").getbbox()

    @classmethod
    def _glyph_ink_value(cls, image: Image.Image) -> Optional[float]:
        return cls._coverage_ink_value(cls._visual_coverage(image))

    @staticmethod
    def _visual_coverage(image: Image.Image) -> np.ndarray:
        """计算图像合成到白底后的视觉墨量，返回 0～255 coverage。"""
        pixels = np.array(
            image.convert("RGBA"),
            dtype=np.float32,
            copy=True,
        )
        luminance = (
            pixels[..., 0] * 0.299
            + pixels[..., 1] * 0.587
            + pixels[..., 2] * 0.114
        )
        coverage = pixels[..., 3] * (255.0 - luminance) / 255.0
        return np.clip(np.rint(coverage), 0.0, 255.0).astype(np.uint8)

    @classmethod
    def _classify_ink_pixels(cls, image: Image.Image) -> str:
        pixels = np.array(
            image.convert("RGBA"),
            dtype=np.uint8,
            copy=True,
        )
        coverage = cls._visual_coverage(image)
        visible = coverage > 0
        if not np.any(visible):
            return "无有效前景"
        values = np.unique(coverage[visible])
        if values.size == 1 and int(values[0]) == 255:
            return "视觉纯二值"
        alpha_values = pixels[..., 3][visible]
        rgb = pixels[..., :3].astype(np.float32)
        luminance = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
        has_rgb_tone = bool(np.any(luminance[visible] > 0.5))
        has_alpha_tone = bool(np.any((alpha_values > 0) & (alpha_values < 255)))
        if has_rgb_tone and has_alpha_tone:
            return "RGB+Alpha混合墨色"
        if has_rgb_tone:
            return "RGB墨色"
        if has_alpha_tone:
            return "Alpha墨色"
        return "视觉灰度"

    @classmethod
    def _has_visible_ink(cls, image: Image.Image) -> bool:
        """确认图像合成到白底后至少包含一个可见墨点。"""
        return bool(np.any(cls._visual_coverage(image) > 0))

    @staticmethod
    def _open_rgba(path: str) -> Optional[Image.Image]:
        if not path or not os.path.exists(path):
            return None
        try:
            with Image.open(path) as image:
                return image.convert("RGBA")
        except (OSError, ValueError):
            return None
