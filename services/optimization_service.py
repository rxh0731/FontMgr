# optimization_service.py — 多候选自动优化服务

import copy
import hashlib
import math
import os
import struct
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import PureWindowsPath
from typing import Any, Callable, Optional

import cv2
import numpy as np
from PIL import Image, ImageChops

import config
from core import pipeline, scoring
from core.imaging import normalize_text_polarity
from core.optimizer import OptimizationCancelled, generate_candidate_results
from core.photoshop_tiff import decode_single_layer_rgba
from core.source_classification import (
    ACTUAL_ALPHA_SOURCES,
    ALPHA_VISIBLE_THRESHOLD,
    SOURCE_TYPE_TRANSPARENT,
    SOURCE_TYPE_UNPROCESSED,
    SOURCE_TYPE_WHITE_CLEANED,
    TRANSPARENCY_SOURCE_PHOTOSHOP_ALPHA,
    TRANSPARENCY_SOURCE_PHOTOSHOP_METADATA,
    TRANSPARENCY_SOURCE_STANDARD_ALPHA,
    SourceClassification,
    classify_source,
)
from data.log_manager import write_log
from services.batch_persistence import (
    BatchJournalUncertainError,
    BatchPersistenceSession,
    acquire_batch_library_lock,
)
from services.background_model_service import (
    BACKGROUND_MODEL_REGISTRY,
    MODEL_OUTPUT_BINARY_MASK,
    MODEL_OUTPUT_CLEAN_GRAY,
    MODEL_OUTPUT_PROBABILITY_MASK,
    NO_MODEL_ENGINE_ID,
    BackgroundModelContext,
    BackgroundModelInferenceResult,
    InferenceCacheKey,
    build_inference_cache_key,
)
from services.glyph_service import GlyphService
from services.workflow_status_service import (
    resolve_safe_stage_file,
    resolve_workflow_status,
)
from services.file_transaction_recovery import (
    FileChange,
    FileTransaction,
    ensure_file_transactions_ready,
    library_root_from_paths,
    recovery_state_snapshot,
)
from utils.file_utils import compute_file_md5


CANDIDATE_TYPE_DIRECT = "原图直接采用"
CANDIDATE_TYPE_ALPHA_DENOISED = "透明层轻度去杂"
CANDIDATE_TYPE_TRANSPARENT = "仅背景透明"
CANDIDATE_TYPE_OPTIMIZED = "寻优优化"
STRUCTURE_REVIEW_REQUIRED = "需人工核对"


def _validated_stage_filename(raw_filename: object, field_name: str) -> str:
    """返回可安全拼入阶段目录的纯文件名，否则拒绝保存。"""
    filename = str(raw_filename or "")
    if (
        not filename
        or filename != filename.strip()
        or filename in {".", ".."}
        or "\x00" in filename
        or ":" in filename
        or filename.endswith((" ", "."))
        or any(ord(character) < 32 for character in filename)
        or "/" in filename
        or "\\" in filename
        or os.path.isabs(filename)
        or os.path.basename(filename) != filename
        or PureWindowsPath(filename).is_reserved()
    ):
        raise ValueError(f"{field_name}不是安全的纯文件名。")
    return filename


class _DigestingWriter:
    """把 PNG 字节顺序写入文件，同时累计与落盘内容一致的摘要。"""

    def __init__(self, raw: Any) -> None:
        self._raw = raw
        self._digest = hashlib.md5()

    def write(self, data: bytes) -> int:
        written = self._raw.write(data)
        if written:
            self._digest.update(memoryview(data)[:written])
        return written

    def hexdigest(self) -> str:
        return self._digest.hexdigest()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw, name)


class OptimizationService:
    """始终读取原始文件，生成多路线候选并确认到中间文件。"""

    MAX_INFERENCE_CACHE_ITEMS = 24
    MAX_BASELINE_SCORE_PIXELS = 96_000
    MAX_BASELINE_SCORE_SIDE = 384

    def __init__(self, glyph_service: GlyphService) -> None:
        self._glyph = glyph_service
        self._inference_cache: OrderedDict[
            InferenceCacheKey,
            BackgroundModelInferenceResult,
        ] = OrderedDict()
        self._inference_lock = threading.RLock()

    def change_variant_char(self, variant_id: str, new_char: str) -> dict[str, Any]:
        """修改当前字形的归属字符，并同步全部阶段文件。"""
        return self._glyph.move_variant_to_char(variant_id, new_char)

    def create_batch_persistence(self) -> BatchPersistenceSession:
        """为整库任务创建逐字数据库提交会话。"""
        return BatchPersistenceSession(self._glyph)

    def list_items(self) -> list[dict[str, Any]]:
        """按字形组顺序返回自动优化页面所需的字形任务。"""
        items: list[dict[str, Any]] = []
        workflow_dirs = self._glyph.get_workflow_dirs()
        source_dir = workflow_dirs["原图"]
        preview_dir = workflow_dirs["优化预览"]
        finished_dir = workflow_dirs.get("成品", "")
        coordination_summary = self._glyph.get_coordination_summary()
        for char_order, char in enumerate(self._glyph.get_all_chars()):
            for index, detail in enumerate(self._glyph.get_char_variants(char)):
                filename = str(detail.get("原始文件", ""))
                optimization = detail.get("自动优化", {})
                source_path = resolve_safe_stage_file(source_dir, filename)
                preview_path = resolve_safe_stage_file(
                    preview_dir,
                    detail.get("中间文件"),
                )
                workflow_status = resolve_workflow_status(
                    detail,
                    coordination_summary,
                    finished_dir,
                )
                items.append({
                    **detail,
                    "键": str(detail.get("变体ID", "")),
                    "归属字": char,
                    "字符顺序": char_order,
                    "变体序号": index + 1,
                    "原始文件名": filename,
                    "原始路径": source_path,
                    "优化预览路径": preview_path,
                    "显示状态": workflow_status.stage,
                    "提示": workflow_status.markers,
                    "墨色状态": workflow_status.ink_status,
                    "得分": optimization.get("得分"),
                })
        return items

    def list_batch_items(self) -> tuple[list[dict[str, Any]], int]:
        """返回全部真实待优化记录及后续阶段跳过数，包括损坏元数据。"""
        workflow_dirs = self._glyph.get_workflow_dirs()
        source_dir = workflow_dirs["原图"]
        preview_dir = workflow_dirs["优化预览"]
        finished_dir = workflow_dirs.get("成品", "")
        coordination_summary = self._glyph.get_coordination_summary()
        pending: list[dict[str, Any]] = []
        skipped = 0
        for order, detail in enumerate(self._glyph.get_all_variants()):
            state = str(detail.get("状态", config.STATUS_PENDING_OPTIMIZATION))
            if state != config.STATUS_PENDING_OPTIMIZATION:
                skipped += 1
                continue
            filename = str(detail.get("原始文件", "") or "")
            source_path = resolve_safe_stage_file(source_dir, filename)
            preview_path = resolve_safe_stage_file(
                preview_dir,
                detail.get("中间文件"),
            )
            optimization = detail.get("自动优化", {})
            if not isinstance(optimization, dict):
                optimization = {}
            workflow_status = resolve_workflow_status(
                detail,
                coordination_summary,
                finished_dir,
            )
            pending.append({
                **detail,
                "键": str(detail.get("变体ID", "")),
                "归属字": str(detail.get("归属字", "")),
                "字符顺序": order,
                "变体序号": detail.get("变体序号", 1),
                "原始文件名": filename,
                "原始路径": source_path,
                "优化预览路径": preview_path,
                "显示状态": workflow_status.stage,
                "提示": workflow_status.markers,
                "墨色状态": workflow_status.ink_status,
                "得分": optimization.get("得分"),
            })
        return pending, skipped

    def generate_candidates(
        self,
        item: dict[str, Any],
        parent_scheme: Optional[dict[str, Any]] = None,
        limit: int = 8,
        engine_context: Optional[BackgroundModelContext] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        _prepared_source: Optional[
            tuple[
                Image.Image,
                np.ndarray,
                str,
                SourceClassification,
                bool,
            ]
        ] = None,
        _batch_final_only: bool = False,
    ) -> list[dict[str, Any]]:
        """生成候选；批量内部模式只包装最终会采用的寻优结果。"""
        path = str(item.get("原始路径", ""))
        context = engine_context or BACKGROUND_MODEL_REGISTRY.create_context()
        optimized_limit = 1 if _batch_final_only else max(1, int(limit))
        started_at = time.perf_counter()
        load_started = time.perf_counter()
        self._raise_if_cancelled(cancel_check)
        if _prepared_source is None:
            source_rgba, gray, transparency_source = self._load_source(path)
            try:
                classification = classify_source(
                    source_rgba,
                    gray,
                    transparency_source,
                )
                self._raise_if_cancelled(cancel_check)
                gray, auto_inverted = self._normalize_source_polarity(
                    source_rgba,
                    gray,
                    transparency_source,
                )
            except Exception:
                source_rgba.close()
                raise
        else:
            (
                source_rgba,
                gray,
                transparency_source,
                classification,
                auto_inverted,
            ) = _prepared_source
        candidates: list[dict[str, Any]] = []
        seen_images: set[str] = set()
        try:
            self._raise_if_cancelled(cancel_check)
            if (
                not _batch_final_only
                and parent_scheme is None
                and classification.source_type != SOURCE_TYPE_UNPROCESSED
            ):
                baseline = self._build_baseline_candidate(
                    source_rgba,
                    gray,
                    transparency_source,
                    classification,
                    auto_inverted,
                    cancel_check,
                )
                if baseline is not None:
                    candidates.append(baseline)
                alpha_denoised, _batch_safe = self._build_alpha_denoised_candidate(
                    source_rgba,
                    gray,
                    transparency_source,
                    classification,
                    auto_inverted,
                    cancel_check,
                )
                if alpha_denoised is not None:
                    candidates.append(alpha_denoised)
        finally:
            if _prepared_source is None:
                source_rgba.close()
        working_gray, inference_fingerprint = self._prepare_engine_input(
            path,
            item,
            gray,
            context,
        )
        self._raise_if_cancelled(cancel_check)
        load_elapsed = time.perf_counter() - load_started
        optimize_started = time.perf_counter()
        if context.engine_id == NO_MODEL_ENGINE_ID:
            results = generate_candidate_results(
                working_gray,
                parent_scheme=parent_scheme,
                limit=optimized_limit,
                cancel_check=cancel_check,
            )
        else:
            results = generate_candidate_results(
                working_gray,
                parent_scheme=parent_scheme,
                limit=optimized_limit,
                reference_gray_arr=gray,
                cancel_check=cancel_check,
            )
        self._raise_if_cancelled(cancel_check)
        optimize_elapsed = time.perf_counter() - optimize_started

        optimized_results = sorted(results, key=self._optimized_result_sort_key)
        optimized_count = 0
        for result in optimized_results:
            self._raise_if_cancelled(cancel_check)
            if optimized_count >= optimized_limit:
                break
            mask = np.asarray(result["掩码"], dtype=np.uint8)
            if mask.ndim != 2 or not np.any(mask > 0):
                continue
            algorithm_protects_original = bool(result.get("保留原图", False))
            if context.engine_id == NO_MODEL_ENGINE_ID and algorithm_protects_original:
                continue
            stored_scheme = copy.deepcopy(result["方案"])
            stored_scheme.pop("结构复核", None)
            structure_review = self.structure_review_metadata(result)
            review_required = self.requires_structure_review(result)
            if structure_review:
                stored_scheme["结构复核"] = copy.deepcopy(structure_review)
            scheme_name = str(result["方案名"])
            if context.engine_id != NO_MODEL_ENGINE_ID:
                # 模型输出已经是前景依据，“原图保护”在这里表示模型直接结果。
                if algorithm_protects_original:
                    stored_scheme.pop("保护原图", None)
                    stored_scheme["L3"] = {"算法": "Otsu", "参数": {"偏移": 0}}
                    stored_scheme["模型直接结果"] = True
                scheme_name = f"{context.descriptor.display_name}·{scheme_name}"
            image = self._gray_to_transparent_image(gray, mask, False)
            added = self._append_candidate(
                candidates,
                seen_images,
                image=image,
                scheme_name=scheme_name,
                scheme=stored_scheme,
                score=float(result["得分"]),
                gray=gray,
                mask=mask,
                quality_level=str(result.get("质量等级", "")),
                candidate_type=CANDIDATE_TYPE_OPTIMIZED,
                protect_original=False,
                deduplicate=not _batch_final_only,
                auto_inverted=auto_inverted,
                engine_context=context,
                inference_fingerprint=inference_fingerprint,
                route_source=(
                    "传统图像管线"
                    if context.engine_id == NO_MODEL_ENGINE_ID
                    else "学习模型前景 + 传统后处理"
                ),
                source_classification=classification,
                score_method={
                    "模式": "寻优管线评分",
                    "说明": (
                        "由自动寻优管线生成；结构保护提示需人工核对，综合得分不代表结构安全"
                        if review_required
                        else "由自动寻优管线生成并通过结构复核"
                    ),
                },
                structure_review=structure_review,
            )
            if added:
                optimized_count += 1

        if optimized_count == 0:
            raise ValueError("算法未生成包含有效文字前景的寻优候选结果。")
        self._raise_if_cancelled(cancel_check)
        write_log(
            f"自动优化服务结束｜字形={item.get('归属字', '')}｜文件={os.path.basename(path)}｜"
            f"读取={load_elapsed:.4f}秒｜算法={optimize_elapsed:.4f}秒｜"
            f"结果包装={time.perf_counter() - optimize_started - optimize_elapsed:.4f}秒｜"
            f"总耗时={time.perf_counter() - started_at:.4f}秒｜自动反相={'是' if auto_inverted else '否'}｜"
            f"处理引擎={context.descriptor.display_name}｜"
            f"原图分类={classification.source_type}({classification.confidence:.2f})｜"
            f"基准候选数={len(candidates) - optimized_count}｜寻优候选数={optimized_count}"
        )
        return candidates

    def generate_batch_candidate(
        self,
        item: dict[str, Any],
        engine_context: Optional[BackgroundModelContext] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> dict[str, Any]:
        """生成批量默认结果；已清理原图走不调用模型和寻优器的快速路径。"""
        path = str(item.get("原始路径", ""))
        started_at = time.perf_counter()
        self._raise_if_cancelled(cancel_check)
        source_rgba, gray, transparency_source = self._load_source(path)
        candidate: Optional[dict[str, Any]] = None
        force_full_optimization = False
        fallback_reason = ""
        try:
            classification = classify_source(source_rgba, gray, transparency_source)
            self._raise_if_cancelled(cancel_check)
            gray, auto_inverted = self._normalize_source_polarity(
                source_rgba,
                gray,
                transparency_source,
            )
            self._raise_if_cancelled(cancel_check)
            if classification.source_type != SOURCE_TYPE_UNPROCESSED:
                candidate = self._build_baseline_candidate(
                    source_rgba,
                    gray,
                    transparency_source,
                    classification,
                    auto_inverted,
                    cancel_check,
                    fast_batch_score=True,
                )
            if classification.source_type == SOURCE_TYPE_TRANSPARENT:
                denoised, batch_direct = self._build_alpha_denoised_candidate(
                    source_rgba,
                    gray,
                    transparency_source,
                    classification,
                    auto_inverted,
                    cancel_check,
                    fast_batch_score=True,
                )
                if (
                    denoised is not None
                    and batch_direct
                    and self.is_candidate_valid(denoised)
                ):
                    if candidate is not None:
                        candidate["图像"].close()
                    candidate = denoised
                elif denoised is not None:
                    cleanup = denoised.get("方案", {}).get("清理统计", {})
                    if isinstance(cleanup, dict):
                        if not bool(cleanup.get("清理充分", True)):
                            fallback_reason = str(
                                cleanup.get("剩余污染说明")
                                or "透明层轻度去杂后仍有疑似噪点"
                            )
                        elif not bool(cleanup.get("批量安全", False)):
                            fallback_reason = str(
                                cleanup.get("人工核对原因")
                                or "透明层轻度去杂存在结构歧义"
                            )
                    force_full_optimization = True
                    denoised["图像"].close()
                elif not batch_direct:
                    force_full_optimization = True
                    fallback_reason = "透明层残留污染需要完整寻优"
            if (
                classification.source_type != SOURCE_TYPE_UNPROCESSED
                and not force_full_optimization
            ):
                if candidate is not None and self.is_candidate_valid(candidate):
                    write_log(
                        f"批量自动优化快速完成｜字形={item.get('归属字', '')}｜"
                        f"文件={os.path.basename(path)}｜原图分类={classification.source_type}｜"
                        f"处理类型={candidate['处理类型']}｜总耗时={time.perf_counter() - started_at:.4f}秒"
                    )
                    return candidate
                write_log(
                    f"批量自动优化快速结果无效，回退完整寻优｜字形={item.get('归属字', '')}｜"
                    f"文件={os.path.basename(path)}｜原图分类={classification.source_type}"
                )
                if candidate is not None:
                    candidate["图像"].close()
                    candidate = None

            if force_full_optimization:
                if candidate is not None:
                    candidate["图像"].close()
                    candidate = None
                write_log(
                    f"批量自动优化透明快速路径清理不足，转入完整寻优｜"
                    f"字形={item.get('归属字', '')}｜文件={os.path.basename(path)}｜"
                    f"原因={fallback_reason or '透明残留污染审计未通过'}"
                )

            self._raise_if_cancelled(cancel_check)
            candidates = self.generate_candidates(
                item,
                limit=1,
                engine_context=engine_context,
                cancel_check=cancel_check,
                _prepared_source=(
                    source_rgba,
                    gray,
                    transparency_source,
                    classification,
                    auto_inverted,
                ),
                _batch_final_only=True,
            )
        finally:
            source_rgba.close()
        valid_optimized = [
            candidate
            for candidate in candidates
            if candidate.get("处理类型") == CANDIDATE_TYPE_OPTIMIZED
            and self.is_candidate_valid(candidate)
        ]
        self._raise_if_cancelled(cancel_check)
        if not valid_optimized:
            raise ValueError("算法未生成包含有效文字前景的寻优候选结果。")
        selected = min(valid_optimized, key=self._optimized_result_sort_key)
        structure_review = self.structure_review_metadata(selected)
        write_log(
            f"批量自动优化完整寻优完成｜字形={item.get('归属字', '')}｜"
            f"文件={os.path.basename(path)}｜原图分类={classification.source_type}｜"
            f"得分={float(selected.get('得分', 0.0)):.1f}｜"
            f"结构复核={structure_review.get('状态', '通过') if structure_review else '通过'}｜"
            f"总耗时={time.perf_counter() - started_at:.4f}秒"
        )
        return selected

    def _build_baseline_candidate(
        self,
        source_rgba: Image.Image,
        gray: np.ndarray,
        transparency_source: str,
        classification: SourceClassification,
        auto_inverted: bool,
        cancel_check: Optional[Callable[[], bool]],
        *,
        fast_batch_score: bool = False,
    ) -> Optional[dict[str, Any]]:
        """独立构造原图或仅去背景基准，不依赖寻优器返回原图保护方案。"""
        self._raise_if_cancelled(cancel_check)
        if classification.source_type == SOURCE_TYPE_TRANSPARENT:
            candidate_type = CANDIDATE_TYPE_DIRECT
            scheme_name = "原图已有有效透明区，直接采用"
            if transparency_source in ACTUAL_ALPHA_SOURCES:
                alpha_image = source_rgba.getchannel("A")
                try:
                    alpha = np.array(alpha_image, dtype=np.uint8, copy=True)
                finally:
                    alpha_image.close()
                mask = self._alpha_foreground_mask(alpha).astype(np.uint8)
                if auto_inverted:
                    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
                    corrected_rgba = np.empty((*gray_u8.shape, 4), dtype=np.uint8)
                    corrected_rgba[..., :3] = gray_u8[..., None]
                    corrected_rgba[..., 3] = alpha
                    image = Image.fromarray(corrected_rgba, "RGBA")
                else:
                    image = source_rgba.copy()
            else:
                mask = self._background_only_mask(gray)
                image = self._gray_to_transparent_image(gray, mask, protect_original=False)
            scheme = {
                "直接采用原图": True,
                "透明来源": transparency_source,
            }
            route_source = "原图直接采用"
        elif classification.source_type == SOURCE_TYPE_WHITE_CLEANED:
            candidate_type = CANDIDATE_TYPE_TRANSPARENT
            scheme_name = "白底已清理，仅转换透明背景"
            mask = self._background_only_mask(gray)
            image = self._gray_to_transparent_image(gray, mask, protect_original=False)
            scheme = {"仅背景透明": True}
            route_source = "仅背景透明"
        else:
            return None

        try:
            if fast_batch_score:
                confidence = float(classification.confidence)
                if not np.isfinite(confidence):
                    confidence = 0.0
                confidence = min(1.0, max(0.0, confidence))
                stored_confidence = round(confidence, 4)
                score = stored_confidence * 100.0
                score_method = {
                    "模式": "批量快速分类评分",
                    "原图分类": classification.source_type,
                    "分类置信度": stored_confidence,
                    "说明": "批量快速采用分类置信度，不执行结构或骨架评分",
                }
            else:
                score, score_method = self._score_candidate_bounded(mask, gray)
            self._raise_if_cancelled(cancel_check)
            candidates: list[dict[str, Any]] = []
            added = self._append_candidate(
                candidates,
                set(),
                image=image,
                scheme_name=scheme_name,
                scheme=scheme,
                score=score,
                gray=gray,
                mask=mask,
                quality_level="基准候选",
                candidate_type=candidate_type,
                protect_original=True,
                deduplicate=False,
                auto_inverted=auto_inverted,
                engine_context=BACKGROUND_MODEL_REGISTRY.create_context(),
                route_source=route_source,
                source_classification=classification,
                score_method=score_method,
            )
            if not added:
                image.close()
                return None
            return candidates[0]
        except Exception:
            image.close()
            raise

    def _build_alpha_denoised_candidate(
        self,
        source_rgba: Image.Image,
        gray: np.ndarray,
        transparency_source: str,
        classification: SourceClassification,
        auto_inverted: bool,
        cancel_check: Optional[Callable[[], bool]],
        *,
        fast_batch_score: bool = False,
    ) -> tuple[Optional[dict[str, Any]], bool]:
        """构造保守的 Alpha 去杂候选，并返回是否可由批量流程自动采用。"""
        if (
            classification.source_type != SOURCE_TYPE_TRANSPARENT
            or transparency_source not in ACTUAL_ALPHA_SOURCES
        ):
            return None, False
        self._raise_if_cancelled(cancel_check)
        rgba = np.array(source_rgba, dtype=np.uint8, copy=True)
        cleaned_alpha, mask, cleanup = self._lightly_denoise_alpha(rgba[..., 3])
        if not cleanup["有变化"]:
            return None, bool(
                cleanup.get("批量安全", False)
                and cleanup.get("清理充分", True)
            )
        self._raise_if_cancelled(cancel_check)

        if auto_inverted:
            gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
            rgba[..., :3] = gray_u8[..., None]
        rgba[..., 3] = cleaned_alpha
        image = Image.fromarray(rgba, "RGBA")
        review_reason = str(cleanup.get("人工核对原因", ""))
        structure_review = (
            {
                "状态": STRUCTURE_REVIEW_REQUIRED,
                "阶段": "透明层轻度去杂",
                "原因": review_reason,
                "风险等级": 1,
            }
            if review_reason
            else None
        )
        try:
            if fast_batch_score:
                confidence = float(classification.confidence)
                if not np.isfinite(confidence):
                    confidence = 0.0
                score = min(1.0, max(0.0, confidence)) * 100.0
                score_method = {
                    "模式": "批量快速Alpha去杂评分",
                    "原图分类": classification.source_type,
                    "分类置信度": round(min(1.0, max(0.0, confidence)), 4),
                    "说明": "只执行透明层安全清理，不执行结构或骨架评分",
                }
            else:
                score, score_method = self._score_candidate_bounded(mask, gray)
            candidates: list[dict[str, Any]] = []
            added = self._append_candidate(
                candidates,
                set(),
                image=image,
                scheme_name="透明层轻度去杂",
                scheme={
                    "透明层轻度去杂": True,
                    "透明来源": transparency_source,
                    "清理统计": cleanup,
                },
                score=score,
                gray=gray,
                mask=mask,
                quality_level="轻度去杂候选",
                candidate_type=CANDIDATE_TYPE_ALPHA_DENOISED,
                protect_original=False,
                deduplicate=False,
                auto_inverted=auto_inverted,
                engine_context=BACKGROUND_MODEL_REGISTRY.create_context(),
                route_source="透明层轻度去杂",
                source_classification=classification,
                score_method=score_method,
                structure_review=structure_review,
            )
            if not added:
                image.close()
                return None, False
            return candidates[0], bool(
                cleanup["批量安全"] and cleanup.get("清理充分", True)
            )
        except Exception:
            image.close()
            raise

    @classmethod
    def _lightly_denoise_alpha(
        cls,
        alpha: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """仅移除远离可靠核心的 Alpha 残留，不腐蚀笔画或合并连通域。"""
        source = np.asarray(alpha, dtype=np.uint8)
        core_threshold = cls._alpha_core_threshold(source)
        foreground_threshold = max(
            ALPHA_VISIBLE_THRESHOLD + 1,
            core_threshold // 4,
        )
        core = source >= core_threshold
        primary = source >= foreground_threshold
        if not core.any() or not primary.any():
            return source.copy(), primary.astype(np.uint8), {
                "有变化": False,
                "批量安全": False,
                "人工核对原因": "",
            }

        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            primary.astype(np.uint8),
            connectivity=8,
            ltype=cv2.CV_32S,
        )
        stroke_width = cls._estimate_alpha_stroke_width(primary)
        stroke_area = max(1.0, stroke_width * stroke_width)
        trusted_area = max(2.0, stroke_area * 0.30)
        largest_label = (
            int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
            if count > 1
            else 0
        )
        trusted_labels: list[int] = []
        core_micro_labels: list[int] = []
        for label in range(1, count):
            left = int(stats[label, cv2.CC_STAT_LEFT])
            top = int(stats[label, cv2.CC_STAT_TOP])
            area = int(stats[label, cv2.CC_STAT_AREA])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            component = labels[top:top + height, left:left + width] == label
            longest_span = float(max(width, height))
            shortest_span = float(min(width, height))
            has_core = bool(
                np.any(
                    core[top:top + height, left:left + width]
                    & component
                )
            )
            scale_reliable = has_core and (
                area >= trusted_area
                or (
                    area >= stroke_area * 0.18
                    and longest_span >= stroke_width * 1.20
                    and shortest_span >= max(2.0, stroke_width * 0.18)
                )
            )
            if label == largest_label or scale_reliable:
                trusted_labels.append(label)
            elif has_core:
                core_micro_labels.append(label)

        reliable_area = int(
            sum(int(stats[label, cv2.CC_STAT_AREA]) for label in trusted_labels)
        )
        micro_areas = np.asarray(
            [int(stats[label, cv2.CC_STAT_AREA]) for label in core_micro_labels],
            dtype=np.float64,
        )
        confident_core_speckles = bool(
            micro_areas.size >= 6
            and float(np.median(micro_areas)) <= stroke_area * 0.10
            and float(micro_areas.sum())
            <= max(stroke_area * 2.5, reliable_area * 0.03)
        )
        if not confident_core_speckles:
            trusted_labels.extend(core_micro_labels)
        kept_primary = (
            np.isin(labels, np.asarray(trusted_labels, dtype=np.int32))
            if trusted_labels
            else primary
        )
        confident_speckle_mask = (
            np.isin(labels, np.asarray(core_micro_labels, dtype=np.int32))
            if confident_core_speckles
            else np.zeros(source.shape, dtype=bool)
        )
        del labels, stats

        longest_side = max(source.shape) if source.ndim == 2 and source.size else 1
        halo_radius = max(1, min(4, int(math.ceil(longest_side / 500.0))))
        kernel_size = halo_radius * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        halo = cv2.dilate(kept_primary.astype(np.uint8), kernel) > 0
        visible = source > ALPHA_VISIBLE_THRESHOLD
        keep = kept_primary | (visible & halo)
        removed = visible & ~keep
        if not removed.any():
            residual = cls._alpha_residual_pollution(
                source,
                core_threshold,
                stroke_width,
            )
            return source.copy(), cls._alpha_foreground_mask(source).astype(np.uint8), {
                "有变化": False,
                "批量安全": True,
                "人工核对原因": "",
                "清理充分": not residual["需要完整寻优"],
                "剩余污染说明": residual["说明"],
                **residual["统计"],
            }

        cleaned = source.copy()
        cleaned[removed] = 0
        cleaned_mask = cls._alpha_foreground_mask(cleaned).astype(np.uint8)
        removed_values = source[removed].astype(np.uint64)
        core_mass = max(1, int(source[core].astype(np.uint64).sum()))
        removed_mass = int(removed_values.sum())
        removed_mass_ratio = removed_mass / core_mass
        removed_maximum = int(removed_values.max())
        component_count, removed_labels, removed_stats, _ = cv2.connectedComponentsWithStats(
            removed.astype(np.uint8),
            connectivity=8,
            ltype=cv2.CV_32S,
        )
        distance_to_core = cv2.distanceTransform(
            (~kept_primary).astype(np.uint8),
            cv2.DIST_L2,
            5,
        )
        protected_area = max(2.0, stroke_area * 0.12)
        elongated_area = max(2.0, stroke_area * 0.08)
        protected_components: list[dict[str, Any]] = []
        maximum_component_area = 0
        for label in range(1, component_count):
            left = int(removed_stats[label, cv2.CC_STAT_LEFT])
            top = int(removed_stats[label, cv2.CC_STAT_TOP])
            area = int(removed_stats[label, cv2.CC_STAT_AREA])
            width = int(removed_stats[label, cv2.CC_STAT_WIDTH])
            height = int(removed_stats[label, cv2.CC_STAT_HEIGHT])
            component = (
                removed_labels[top:top + height, left:left + width] == label
            )
            maximum_component_area = max(maximum_component_area, area)
            component_values = source[
                top:top + height,
                left:left + width,
            ][component].astype(np.uint64)
            component_mass_ratio = float(component_values.sum()) / core_mass
            alpha_p90 = float(np.percentile(component_values, 90.0))
            longest_span = float(max(width, height))
            shortest_span = float(min(width, height))
            aspect_ratio = longest_span / max(1.0, shortest_span)
            minimum_distance = float(
                distance_to_core[
                    top:top + height,
                    left:left + width,
                ][component].min()
            )
            gap_ratio = max(0.0, minimum_distance - 1.0) / max(stroke_width, 1.0)
            confident_speckle_component = bool(
                np.any(
                    confident_speckle_mask[
                        top:top + height,
                        left:left + width,
                    ][component]
                )
            )

            if (
                confident_speckle_component
                and area < stroke_area * 0.45
                and component_mass_ratio < 0.005
            ):
                continue

            point_stroke_like = (
                area >= protected_area
                and shortest_span >= max(2.0, stroke_width * 0.25)
                and alpha_p90 >= foreground_threshold
            )
            elongated_stroke_like = (
                area >= elongated_area
                and longest_span >= stroke_width * 1.20
                and aspect_ratio >= 2.0
                and alpha_p90 >= foreground_threshold
            )
            nearby_fragment = (
                area >= stroke_area * 0.10
                and gap_ratio <= 0.35
                and alpha_p90 >= max(foreground_threshold, core_threshold * 0.55)
            )
            large_faint_region = area >= stroke_area * 0.30
            meaningful_alpha_mass = component_mass_ratio >= 0.0025
            if not (
                point_stroke_like
                or elongated_stroke_like
                or nearby_fragment
                or large_faint_region
                or meaningful_alpha_mass
            ):
                continue
            protected_components.append(
                {
                    "面积": area,
                    "宽": width,
                    "高": height,
                    "Alpha九十分位": round(alpha_p90, 2),
                    "距核心笔画宽度比": round(gap_ratio, 3),
                    "Alpha总量占核心": round(component_mass_ratio, 6),
                }
            )

        removal_too_large = removed_mass_ratio > 0.03
        reasons: list[str] = []
        if protected_components:
            reasons.append(
                f"待清理区域中有{len(protected_components)}个连通域达到独立点画或笔画片段尺度"
            )
        if removal_too_large:
            reasons.append("移除的透明残留总量相对字形核心较大")
        residual = cls._alpha_residual_pollution(
            cleaned,
            core_threshold,
            stroke_width,
        )
        return cleaned, cleaned_mask, {
            "有变化": True,
            "批量安全": not reasons,
            "人工核对原因": "；".join(reasons),
            "清理充分": not residual["需要完整寻优"],
            "剩余污染说明": residual["说明"],
            "核心阈值": core_threshold,
            "主要前景阈值": foreground_threshold,
            "边缘恢复半径": halo_radius,
            "移除像素数": int(np.count_nonzero(removed)),
            "移除连通域数": max(0, component_count - 1),
            "最大移除连通域面积": maximum_component_area,
            "移除Alpha最大值": removed_maximum,
            "移除Alpha总量占核心": round(removed_mass_ratio, 6),
            "估算笔画宽度": round(stroke_width, 3),
            "需保护连通域数": len(protected_components),
            "需保护连通域": protected_components[:8],
            "高Alpha微小域清理数": (
                len(core_micro_labels) if confident_core_speckles else 0
            ),
            **residual["统计"],
        }

    @staticmethod
    def _alpha_residual_pollution(
        alpha: np.ndarray,
        core_threshold: int,
        stroke_width: float,
    ) -> dict[str, Any]:
        """检查轻度清理后仍存在的高 Alpha 微小域，决定是否继续完整寻优。"""
        core = (np.asarray(alpha, dtype=np.uint8) >= int(core_threshold)).astype(
            np.uint8
        )
        count, _labels, stats, _ = cv2.connectedComponentsWithStats(
            core,
            connectivity=8,
            ltype=cv2.CV_32S,
        )
        if count <= 2:
            suspicious_count = 0
            suspicious_pixels = 0
        else:
            areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64, copy=False)
            largest_area = max(1, int(areas.max()))
            stroke_area = max(1.0, float(stroke_width) ** 2)
            area_limit = max(2.0, min(stroke_area * 0.24, largest_area * 0.05))
            span_limit = max(2.0, float(stroke_width) * 1.20)
            widths = stats[1:, cv2.CC_STAT_WIDTH].astype(np.float64, copy=False)
            heights = stats[1:, cv2.CC_STAT_HEIGHT].astype(np.float64, copy=False)
            suspicious = (
                (areas.astype(np.float64) <= area_limit)
                & (np.maximum(widths, heights) <= span_limit)
            )
            suspicious_count = int(np.count_nonzero(suspicious))
            suspicious_pixels = int(areas[suspicious].sum())
        needs_full = suspicious_count >= 6
        explanation = (
            f"清理后仍有{suspicious_count}个高Alpha微小连通域，转入完整寻优"
            if needs_full
            else ""
        )
        return {
            "需要完整寻优": needs_full,
            "说明": explanation,
            "统计": {
                "剩余高Alpha微小域数": suspicious_count,
                "剩余高Alpha微小域像素数": suspicious_pixels,
            },
        }

    @staticmethod
    def _estimate_alpha_stroke_width(mask: np.ndarray) -> float:
        """从可靠主体的距离脊线估算笔画宽度，供 Alpha 小域风险归一化。"""
        source = (np.asarray(mask) > 0).astype(np.uint8)
        if source.ndim != 2 or not source.any():
            return 1.0
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            source,
            connectivity=8,
            ltype=cv2.CV_32S,
        )
        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64, copy=False)
        selected_mask = source > 0
        if count > 2 and areas.size:
            order = np.argsort(areas)[::-1]
            target = max(1, int(round(float(areas.sum()) * 0.70)))
            selected: list[int] = []
            accumulated = 0
            for offset in order:
                selected.append(int(offset) + 1)
                accumulated += int(areas[offset])
                if accumulated >= target:
                    break
            selected_mask = np.isin(labels, np.asarray(selected, dtype=np.int32))
        distance = cv2.distanceTransform(source, cv2.DIST_L2, 5)
        local_maximum = distance >= cv2.dilate(
            distance,
            np.ones((3, 3), dtype=np.uint8),
        ) - 0.05
        ridge_values = distance[selected_mask & local_maximum & (distance >= 0.9)]
        if ridge_values.size >= 3:
            estimate = float(np.median(ridge_values) * 2.0)
        else:
            foreground_values = distance[selected_mask]
            estimate = (
                float(np.percentile(foreground_values, 80.0) * 2.0)
                if foreground_values.size
                else 1.0
            )
        short_side = float(min(source.shape))
        return float(np.clip(estimate, 1.0, max(2.0, short_side * 0.12)))

    @classmethod
    def _score_candidate_bounded(
        cls,
        mask: np.ndarray,
        gray: np.ndarray,
    ) -> tuple[float, dict[str, Any]]:
        """限制基准评分尺寸，避免百万像素原图反复执行骨架分析。"""
        mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
        gray_u8 = np.clip(np.asarray(gray), 0, 255).astype(np.uint8)
        height, width = gray_u8.shape
        longest_ratio = min(1.0, cls.MAX_BASELINE_SCORE_SIDE / max(1, height, width))
        area_ratio = min(1.0, (cls.MAX_BASELINE_SCORE_PIXELS / max(1, height * width)) ** 0.5)
        ratio = min(longest_ratio, area_ratio)
        if ratio < 1.0:
            target_size = (
                max(1, int(round(width * ratio))),
                max(1, int(round(height * ratio))),
            )
            gray_image = Image.fromarray(gray_u8, "L")
            mask_image = Image.fromarray(mask_u8 * 255, "L")
            try:
                resized_gray = gray_image.resize(target_size, Image.Resampling.BOX)
                resized_mask = mask_image.resize(target_size, Image.Resampling.NEAREST)
                try:
                    score_gray = np.array(resized_gray, dtype=np.uint8, copy=True)
                    score_mask = np.array(resized_mask, dtype=np.uint8, copy=True) > 0
                finally:
                    resized_mask.close()
                    resized_gray.close()
            finally:
                mask_image.close()
                gray_image.close()
            mode = "有界缩略图"
        else:
            score_gray = gray_u8
            score_mask = mask_u8
            target_size = (width, height)
            mode = "原尺寸"
        breakdown, _timing = scoring.evaluate_candidate(
            score_mask.astype(np.uint8),
            score_gray.astype(np.float32),
        )
        return float(breakdown.score), {
            "模式": mode,
            "原始尺寸": [width, height],
            "评分尺寸": [target_size[0], target_size[1]],
            "像素上限": cls.MAX_BASELINE_SCORE_PIXELS,
        }

    @staticmethod
    def _background_only_mask(gray: np.ndarray) -> np.ndarray:
        """用 Otsu 区分字与背景，不执行去杂或形态学修改。"""
        _processed, mask = pipeline.run_pipeline(
            np.clip(gray, 0, 255).astype(np.float32),
            {"预处理": {}, "L3": {"算法": "Otsu", "参数": {"偏移": 0}}},
        )
        if mask is None:
            return np.zeros(np.asarray(gray).shape, dtype=np.uint8)
        return (np.asarray(mask) > 0).astype(np.uint8)

    @staticmethod
    def _raise_if_cancelled(cancel_check: Optional[Callable[[], bool]]) -> None:
        if cancel_check is not None and cancel_check():
            raise OptimizationCancelled("自动优化已由用户停止。")

    @staticmethod
    def _normalize_structure_review(value: object) -> dict[str, Any]:
        """规范化结构复核元数据，忽略历史或损坏的非字典值。"""
        if not isinstance(value, dict):
            return {}
        status = str(value.get("状态", "")).strip()
        if not status:
            return {}
        review = {
            "状态": status,
            "阶段": str(value.get("阶段", "")).strip(),
            "原因": str(value.get("原因", "")).strip(),
        }
        try:
            risk_level = int(value.get("风险等级", 1))
        except (TypeError, ValueError):
            risk_level = 1
        review["风险等级"] = max(1, risk_level)
        return review

    @classmethod
    def structure_review_metadata(cls, candidate: object) -> dict[str, Any]:
        """从候选顶层或持久化方案中读取结构复核信息。"""
        if not isinstance(candidate, dict):
            return {}
        review = cls._normalize_structure_review(candidate.get("结构复核"))
        if review:
            return review
        scheme = candidate.get("方案", {})
        if not isinstance(scheme, dict):
            return {}
        return cls._normalize_structure_review(scheme.get("结构复核"))

    @classmethod
    def requires_structure_review(cls, candidate: object) -> bool:
        """判断候选是否需要在手工审核阶段重点核对结构。"""
        review = cls.structure_review_metadata(candidate)
        return review.get("状态") == STRUCTURE_REVIEW_REQUIRED

    @classmethod
    def _optimized_result_sort_key(cls, candidate: dict[str, Any]) -> tuple[Any, ...]:
        """安全候选优先；无安全结果时再按风险等级和得分选择。"""
        review = cls.structure_review_metadata(candidate)
        required = review.get("状态") == STRUCTURE_REVIEW_REQUIRED
        try:
            risk_level = int(review.get("风险等级", 1)) if required else 0
        except (TypeError, ValueError):
            risk_level = 1
        try:
            score = float(candidate.get("得分", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        return (
            1 if required else 0,
            max(1, risk_level) if required else 0,
            -score,
            str(candidate.get("方案名", "")),
        )

    @staticmethod
    def _append_candidate(
        candidates: list[dict[str, Any]],
        seen_images: set[str],
        *,
        image: Image.Image,
        scheme_name: str,
        scheme: dict[str, Any],
        score: float,
        gray: np.ndarray,
        mask: np.ndarray,
        quality_level: str,
        candidate_type: str,
        protect_original: bool = False,
        deduplicate: bool = True,
        auto_inverted: bool = False,
        engine_context: Optional[BackgroundModelContext] = None,
        inference_fingerprint: str = "",
        route_source: str = "传统图像管线",
        source_classification: Optional[SourceClassification] = None,
        score_method: Optional[dict[str, Any]] = None,
        structure_review: Optional[dict[str, Any]] = None,
    ) -> bool:
        """包装候选数据，特殊候选不参与普通寻优结果的像素去重。"""
        digest = ""
        if deduplicate:
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            if digest in seen_images:
                return False
            seen_images.add(digest)
        stored_scheme = copy.deepcopy(scheme)
        stored_scheme.pop("结构复核", None)
        normalized_review = OptimizationService._normalize_structure_review(
            structure_review
        )
        if normalized_review:
            stored_scheme["结构复核"] = copy.deepcopy(normalized_review)
        stored_scheme["处理类型"] = candidate_type
        context = engine_context or BACKGROUND_MODEL_REGISTRY.create_context()
        engine_metadata = context.to_metadata()
        if inference_fingerprint:
            engine_metadata["推理结果指纹"] = inference_fingerprint
        stored_scheme["处理引擎"] = engine_metadata
        stored_scheme["路线来源"] = route_source
        classification_metadata: dict[str, Any] = {}
        if source_classification is not None:
            classification_metadata = source_classification.as_metadata()
            stored_scheme["原图分类"] = classification_metadata
        if score_method:
            stored_scheme["评分方式"] = dict(score_method)
        preprocess = stored_scheme.get("预处理", {})
        if isinstance(preprocess, dict) and preprocess.get("墨色归一"):
            stored_scheme["墨色归一用途"] = "算法工作归一，仅用于分割与评分，不写回灰度母版"
        if auto_inverted:
            stored_scheme["自动校正"] = {
                "反相": True,
                "说明": "检测到深色背景，已自动校正为白底深字",
            }
        candidate = {
            "方案名": scheme_name,
            "方案": stored_scheme,
            "得分": float(score),
            "图像": image,
            "图像指纹": digest,
            "质量等级": quality_level,
            "灰度母版": np.clip(gray, 0, 255).astype(np.uint8),
            "清洁掩码": (mask > 0).astype(np.uint8),
            "保留原图": protect_original,
            "处理类型": candidate_type,
            "原图分类": classification_metadata,
        }
        if normalized_review:
            candidate["结构复核"] = copy.deepcopy(normalized_review)
        candidates.append(candidate)
        return True

    def explore(
        self,
        item: dict[str, Any],
        candidate: dict[str, Any],
        count: int = 8,
        engine_context: Optional[BackgroundModelContext] = None,
    ) -> list[dict[str, Any]]:
        """围绕选中候选的方案参数生成下一轮，并排除基准结果。"""
        candidate_type = str(candidate.get("处理类型", CANDIDATE_TYPE_OPTIMIZED))
        if candidate_type != CANDIDATE_TYPE_OPTIMIZED:
            raise ValueError("只有“寻优优化”候选可以继续探索。")
        scheme = candidate.get("方案", {})
        if not isinstance(scheme, dict):
            scheme = {}
        stored_engine = scheme.get("处理引擎", {})
        if not isinstance(stored_engine, dict):
            stored_engine = {}
        context = self._resolve_exploration_context(stored_engine, engine_context)
        results = self.generate_candidates(
            item,
            parent_scheme=scheme,
            limit=count + 1,
            engine_context=context,
        )
        base_digest = candidate.get("图像指纹")
        return [result for result in results if result.get("图像指纹") != base_digest][:count]

    @staticmethod
    def _resolve_exploration_context(
        stored_engine: dict[str, Any],
        engine_context: Optional[BackgroundModelContext],
    ) -> BackgroundModelContext:
        """恢复候选固定的模型配置，并拒绝用已变化的模型继续旧分支。"""
        stored_engine_id = str(stored_engine.get("标识", "")).strip()
        if not stored_engine_id:
            return engine_context or BACKGROUND_MODEL_REGISTRY.create_context()
        configuration = stored_engine.get("推理参数", {})
        if not isinstance(configuration, dict):
            raise ValueError("候选记录中的模型推理参数无效，请重新生成基础候选。")
        try:
            context = BACKGROUND_MODEL_REGISTRY.create_context(
                stored_engine_id,
                configuration,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "候选使用的处理引擎已不可用，请从原图重新生成基础候选。"
            ) from exc
        current_metadata = context.to_metadata()
        for key in ("版本", "输出类型", "模型指纹", "配置指纹"):
            if key in stored_engine and stored_engine.get(key) != current_metadata.get(key):
                raise RuntimeError(
                    "候选使用的处理引擎版本或配置已经变化，请从原图重新生成基础候选。"
                )
        return context

    def _prepare_engine_input(
        self,
        path: str,
        item: dict[str, Any],
        gray: np.ndarray,
        context: BackgroundModelContext,
    ) -> tuple[np.ndarray, str]:
        """执行一次可缓存的模型推理，并转换为寻优工作灰度。"""
        if context.engine_id == NO_MODEL_ENGINE_ID:
            return gray, ""
        source_fingerprint = str(item.get("原始MD5") or "")
        if not source_fingerprint:
            source_fingerprint = compute_file_md5(path)
        cache_key = build_inference_cache_key(source_fingerprint, context)
        with self._inference_lock:
            result = self._inference_cache.get(cache_key)
            if result is None:
                result = BACKGROUND_MODEL_REGISTRY.infer(gray.copy(), context)
                self._inference_cache[cache_key] = result
                while len(self._inference_cache) > self.MAX_INFERENCE_CACHE_ITEMS:
                    self._inference_cache.popitem(last=False)
            else:
                self._inference_cache.move_to_end(cache_key)
        result.validate(expected_shape=tuple(gray.shape))
        if result.data is None:
            raise RuntimeError(f"处理引擎“{context.descriptor.display_name}”没有返回图像结果。")
        if result.output_type == MODEL_OUTPUT_CLEAN_GRAY:
            working = np.asarray(result.data, dtype=np.float32)
        elif result.output_type == MODEL_OUTPUT_PROBABILITY_MASK:
            probability = np.clip(np.asarray(result.data, dtype=np.float32), 0.0, 1.0)
            working = (1.0 - probability) * 255.0
        elif result.output_type == MODEL_OUTPUT_BINARY_MASK:
            working = np.where(np.asarray(result.data) > 0, 0.0, 255.0).astype(np.float32)
        else:
            raise ValueError(f"不支持的模型输出类型：{result.output_type}。")
        return np.clip(working, 0, 255).astype(np.float32), result.fingerprint

    def save_selection(
        self,
        item: dict[str, Any],
        candidate: dict[str, Any],
        round_number: int = 1,
        *,
        persistence: BatchPersistenceSession | None = None,
    ) -> str:
        """保存所选方案；交互保存与批处理共用同一字库独占边界。"""
        if persistence is not None:
            return self._save_selection_locked(
                item,
                candidate,
                round_number,
                persistence=persistence,
            )
        workflow_dirs = self._glyph.get_workflow_dirs()
        library_lock = acquire_batch_library_lock(
            library_root_from_paths(self._glyph, workflow_dirs.values())
        )
        try:
            return self._save_selection_locked(
                item,
                candidate,
                round_number,
                persistence=None,
            )
        finally:
            library_lock.release()

    def _save_selection_locked(
        self,
        item: dict[str, Any],
        candidate: dict[str, Any],
        round_number: int = 1,
        *,
        persistence: BatchPersistenceSession | None = None,
    ) -> str:
        variant_id = str(item.get("键", ""))
        detail = self._glyph.get_variant(variant_id)
        if not detail:
            raise ValueError("字形记录不存在。")
        validation_error = self.candidate_validation_error(candidate)
        if validation_error:
            raise ValueError(validation_error)
        workflow_dirs = self._glyph.get_workflow_dirs()
        ensure_file_transactions_ready(
            library_root_from_paths(self._glyph, workflow_dirs.values())
        )
        source_filename = _validated_stage_filename(
            detail.get("原始文件"),
            "原始文件名",
        )
        filename = os.path.splitext(source_filename)[0] + ".png"
        preview_path = os.path.join(workflow_dirs["优化预览"], filename)
        gray_master_path = os.path.join(workflow_dirs["灰度母版"], filename)
        clean_mask_path = os.path.join(workflow_dirs["清洁掩码"], filename)
        reviewed_path = os.path.join(workflow_dirs["手工审核"], filename)
        finished_path = os.path.join(workflow_dirs["成品"], filename)
        image = candidate.get("图像")
        gray_master = candidate.get("灰度母版")
        clean_mask = candidate.get("清洁掩码")
        scheme = copy.deepcopy(candidate.get("方案", {}))
        structure_review = self.structure_review_metadata(candidate)
        if structure_review:
            scheme["结构复核"] = copy.deepcopy(structure_review)
        else:
            scheme.pop("结构复核", None)
        candidate_type = str(
            candidate.get("处理类型") or scheme.get("处理类型") or CANDIDATE_TYPE_OPTIMIZED
        )
        scheme["处理类型"] = candidate_type
        dpi = self._source_dpi(detail)
        output_images = (
            (image, preview_path),
            (Image.fromarray(np.clip(gray_master, 0, 255).astype(np.uint8), "L"), gray_master_path),
            (Image.fromarray((clean_mask > 0).astype(np.uint8) * 255, "L"), clean_mask_path),
        )
        output_paths = tuple(path for _output_image, path in output_images)
        recorded_stage_paths: list[str] = []
        for directory_key, detail_key in (
            ("优化预览", "中间文件"),
            ("灰度母版", "灰度母版文件"),
            ("清洁掩码", "清洁掩码文件"),
            ("手工审核", "审核文件"),
            ("成品", "成品文件"),
        ):
            raw_recorded_filename = detail.get(detail_key, "")
            recorded_filename = (
                _validated_stage_filename(raw_recorded_filename, detail_key)
                if str(raw_recorded_filename or "")
                else ""
            )
            if recorded_filename:
                recorded_stage_paths.append(
                    os.path.join(workflow_dirs[directory_key], recorded_filename)
                )
        transaction_paths = tuple(dict.fromkeys(
            path
            for path in (
                *output_paths,
                reviewed_path,
                finished_path,
                *recorded_stage_paths,
            )
            if path
        ))
        temporary_paths: list[str] = []
        direct_installed_paths: list[str] = []
        detail_backup = copy.deepcopy(detail)
        raw_state_backup = self._glyph.snapshot_variant_state(variant_id)
        state_backup = recovery_state_snapshot(
            raw_state_backup,
            variant_id,
            detail_backup,
        )
        transaction: FileTransaction | None = None
        state_persisted = False
        try:
            output_hashes: list[str] = []
            for output_image, target_path in output_images:
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                temporary_path, output_hash = self._save_temp_png(
                    output_image,
                    target_path,
                    dpi,
                )
                temporary_paths.append(temporary_path)
                output_hashes.append(output_hash)

            output_changes = {
                os.path.normcase(os.path.abspath(target_path)): (
                    temporary_path,
                    output_hash,
                )
                for temporary_path, target_path, output_hash in zip(
                    temporary_paths,
                    output_paths,
                    output_hashes,
                )
            }
            # SQLite 提交失败时仍需独立清单恢复，首次生成也使用图片事务。
            requires_transaction = True
            if requires_transaction:
                changes: list[FileChange] = []
                seen_targets: set[str] = set()
                for target_path in transaction_paths:
                    target_key = os.path.normcase(os.path.abspath(target_path))
                    if target_key in seen_targets:
                        continue
                    seen_targets.add(target_key)
                    replacement = output_changes.get(target_key)
                    if replacement is None:
                        changes.append(FileChange(target_path=target_path))
                    else:
                        temporary_path, output_hash = replacement
                        changes.append(
                            FileChange(
                                target_path=target_path,
                                temporary_path=temporary_path,
                                new_md5=output_hash,
                            )
                        )

                transaction = FileTransaction.begin(
                    library_root_from_paths(self._glyph, workflow_dirs.values()),
                    changes,
                    state_backup,
                )
                transaction.backup_targets()

            self._glyph.confirm_optimization(
                variant_id,
                filename,
                output_hashes[0],
                str(candidate.get("方案名", "自动优化")),
                scheme,
                float(candidate.get("得分", 0.0)),
                round_number,
                gray_master_filename=filename,
                gray_master_md5=output_hashes[1],
                clean_mask_filename=filename,
                clean_mask_md5=output_hashes[2],
            )
            new_state = recovery_state_snapshot(
                self._glyph.snapshot_variant_state(variant_id),
                variant_id,
                detail,
            )
            if transaction is not None:
                transaction.mark_rollforward(new_state)
                transaction.install_new_files()
            else:
                for temporary_path, target_path in zip(
                    temporary_paths,
                    output_paths,
                ):
                    os.replace(temporary_path, target_path)
                    direct_installed_paths.append(target_path)
            if persistence is None:
                self._glyph.save()
            else:
                try:
                    persistence.record_variant(variant_id)
                except BatchJournalUncertainError:
                    # 数据库提交结果未知，保留图片和清单，由启动恢复统一裁决。
                    state_persisted = True
                    raise
            state_persisted = True
            if transaction is not None:
                cleanup_errors = transaction.finalize()
                if cleanup_errors:
                    write_log(
                        "自动优化图片事务已提交，清理将于下次打开继续｜"
                        + "；".join(cleanup_errors)
                    )
        except Exception as exc:
            if state_persisted:
                raise
            if isinstance(raw_state_backup, dict):
                self._glyph.restore_variant_state(raw_state_backup)
            detail.clear()
            detail.update(detail_backup)
            if transaction is not None:
                rollback_errors = transaction.rollback()
            else:
                rollback_errors = []
                for target_path in reversed(direct_installed_paths):
                    try:
                        if os.path.exists(target_path):
                            os.remove(target_path)
                    except OSError as rollback_exc:
                        rollback_errors.append(
                            f"无法移除未提交的新文件 {target_path}：{rollback_exc}"
                        )
            if rollback_errors:
                joined = "；".join(rollback_errors)
                raise RuntimeError(f"自动优化稿保存失败，且回滚未完全完成：{joined}") from exc
            raise
        finally:
            if transaction is None:
                for temporary_path in temporary_paths:
                    if os.path.exists(temporary_path):
                        try:
                            os.remove(temporary_path)
                        except OSError:
                            pass
        return preview_path

    @classmethod
    def is_candidate_valid(cls, candidate: object) -> bool:
        """判断候选分层数据是否完整并可保存，不代表结构保护已通过。"""
        return not cls.candidate_validation_error(candidate)

    @staticmethod
    def candidate_validation_error(candidate: object) -> str:
        """返回候选分层数据错误；空字符串表示数据有效。"""
        if not isinstance(candidate, dict):
            return "候选结果格式无效。"
        image = candidate.get("图像")
        gray_master = candidate.get("灰度母版")
        clean_mask = candidate.get("清洁掩码")
        if not isinstance(image, Image.Image):
            return "候选图片无效。"
        if image.width <= 0 or image.height <= 0:
            return "候选图片尺寸无效。"
        luminance = image.convert("L")
        alpha = (
            image.getchannel("A")
            if "A" in image.getbands()
            else Image.new("L", image.size, 255)
        )
        inverted = ImageChops.invert(luminance)
        visible_ink = ImageChops.multiply(inverted, alpha)
        try:
            maximum_visible_ink = int(visible_ink.getextrema()[1])
        finally:
            visible_ink.close()
            inverted.close()
            alpha.close()
            luminance.close()
        if maximum_visible_ink <= 1:
            return "候选图片没有可见的非白文字前景。"
        if not isinstance(gray_master, np.ndarray) or gray_master.ndim != 2:
            return "候选灰度母版无效。"
        if not isinstance(clean_mask, np.ndarray) or clean_mask.ndim != 2:
            return "候选清洁掩码无效。"
        if gray_master.shape != clean_mask.shape:
            return "候选灰度母版与清洁掩码尺寸不一致。"
        if image.size != (gray_master.shape[1], gray_master.shape[0]):
            return "候选图片与分层数据尺寸不一致。"
        if not np.issubdtype(gray_master.dtype, np.number):
            return "候选灰度母版数据类型无效。"
        if (
            not np.issubdtype(clean_mask.dtype, np.number)
            and not np.issubdtype(clean_mask.dtype, np.bool_)
        ):
            return "候选清洁掩码数据类型无效。"
        if not np.isfinite(gray_master).all() or not np.isfinite(clean_mask).all():
            return "候选分层数据包含无效数值。"
        if not np.any(clean_mask > 0):
            return "候选结果没有有效文字前景。"
        scheme = candidate.get("方案", {})
        if not isinstance(scheme, dict):
            return "候选方案数据无效。"
        try:
            score = float(candidate.get("得分", 0.0))
        except (TypeError, ValueError):
            return "候选得分无效。"
        if not np.isfinite(score):
            return "候选得分无效。"
        return ""

    @staticmethod
    def _save_temp_png(
        image: Image.Image,
        target_path: str,
        dpi: tuple[float, float],
    ) -> tuple[str, str]:
        """生成临时 PNG，并在编码写入时同步计算落盘字节摘要。"""
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=".fonteditor_save_",
            suffix=".png",
            dir=os.path.dirname(target_path),
        )
        try:
            with os.fdopen(descriptor, "wb") as raw:
                writer = _DigestingWriter(raw)
                image.save(writer, "PNG", dpi=dpi)
                writer.flush()
                os.fsync(raw.fileno())
                digest = writer.hexdigest()
        except Exception:
            try:
                os.close(descriptor)
            except OSError:
                pass
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
            raise
        return temporary_path, digest

    @staticmethod
    def _reserve_backup_path(target_path: str) -> str:
        """在同一目录预留回滚文件，保证替换操作不跨磁盘。"""
        descriptor, backup_path = tempfile.mkstemp(
            prefix=".fonteditor_rollback_",
            suffix=os.path.splitext(target_path)[1],
            dir=os.path.dirname(target_path),
        )
        os.close(descriptor)
        return backup_path

    @staticmethod
    def _rollback_files(
        installed_paths: list[str],
        backup_paths: list[tuple[str, str]],
    ) -> list[str]:
        """删除本次新文件并恢复全部旧文件，返回无法恢复的错误。"""
        errors: list[str] = []
        for target_path in reversed(installed_paths):
            try:
                if os.path.exists(target_path):
                    os.remove(target_path)
            except OSError as exc:
                errors.append(f"无法移除新文件 {target_path}：{exc}")
        for target_path, backup_path in reversed(backup_paths):
            try:
                if os.path.exists(backup_path):
                    os.replace(backup_path, target_path)
            except OSError as exc:
                errors.append(f"无法恢复旧文件 {target_path}：{exc}")
        return errors

    @staticmethod
    def _source_dpi(detail: dict[str, Any]) -> tuple[float, float]:
        """读取原图记录的 DPI，并为无效数据提供安全默认值。"""
        image_info = detail.get("图像信息", {})
        if not isinstance(image_info, dict):
            image_info = {}

        def valid_dpi(value: Any, fallback: float) -> float:
            try:
                dpi = float(value)
            except (TypeError, ValueError):
                return fallback
            return dpi if 1.0 <= dpi <= 9600.0 else fallback

        dpi_x = valid_dpi(image_info.get("水平DPI"), 300.0)
        dpi_y = valid_dpi(image_info.get("垂直DPI"), dpi_x)
        return dpi_x, dpi_y

    @staticmethod
    def _load_source(path: str) -> tuple[Image.Image, np.ndarray, str]:
        """读取原图，并区分标准 Alpha、已解码图层 Alpha 和仅元数据。"""
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"找不到原始图片：{path}")
        with Image.open(path) as source_image:
            source_image.seek(0)
            decoded_photoshop = decode_single_layer_rgba(source_image)
            has_photoshop_metadata = (
                decoded_photoshop is None
                and OptimizationService._has_photoshop_transparency(source_image)
            )
            rgba = (
                decoded_photoshop
                if decoded_photoshop is not None
                else source_image.convert("RGBA")
            )
        try:
            if decoded_photoshop is not None:
                transparency_source = TRANSPARENCY_SOURCE_PHOTOSHOP_ALPHA
            else:
                alpha_image = rgba.getchannel("A")
                try:
                    has_standard_alpha = alpha_image.getextrema()[0] < 255
                finally:
                    alpha_image.close()
                if has_standard_alpha:
                    transparency_source = TRANSPARENCY_SOURCE_STANDARD_ALPHA
                elif has_photoshop_metadata:
                    transparency_source = TRANSPARENCY_SOURCE_PHOTOSHOP_METADATA
                else:
                    transparency_source = ""
            white_background = Image.new("RGBA", rgba.size, "white")
            try:
                white_background.alpha_composite(rgba)
                gray_image = white_background.convert("L")
                try:
                    gray = np.array(gray_image, dtype=np.float32, copy=True)
                finally:
                    gray_image.close()
            finally:
                white_background.close()
        except Exception:
            rgba.close()
            raise
        return rgba, gray, transparency_source

    @staticmethod
    def _normalize_source_polarity(
        source_rgba: Image.Image,
        gray: np.ndarray,
        transparency_source: str,
    ) -> tuple[np.ndarray, bool]:
        """按不透明背景或实际 Alpha 的高置信字形核心判断文字极性。"""
        if transparency_source not in ACTUAL_ALPHA_SOURCES:
            return normalize_text_polarity(gray)

        rgba = np.asarray(source_rgba, dtype=np.uint8)
        alpha = rgba[..., 3]
        core = OptimizationService._alpha_core_mask(alpha)
        if not core.any():
            return np.clip(gray, 0, 255).astype(np.float32), False
        rgb = rgba[..., :3].astype(np.float32)
        luminance = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
        should_invert = float(np.median(luminance[core])) > 127.5
        if not should_invert:
            return np.clip(gray, 0, 255).astype(np.float32), False
        core_threshold = OptimizationService._alpha_core_threshold(alpha)
        foreground_threshold = max(
            ALPHA_VISIBLE_THRESHOLD + 1,
            core_threshold // 4,
        )
        visible = alpha >= foreground_threshold
        corrected = np.full(gray.shape, 255.0, dtype=np.float32)
        corrected[visible] = 255.0 - np.clip(gray[visible], 0, 255)
        return corrected, True

    @staticmethod
    def _alpha_core_threshold(alpha: np.ndarray) -> int:
        """返回相对最大不透明度的字形核心阈值，排除低 Alpha 背景薄雾。"""
        source = np.asarray(alpha, dtype=np.uint8)
        maximum = int(source.max()) if source.size else 0
        if maximum <= ALPHA_VISIBLE_THRESHOLD:
            return ALPHA_VISIBLE_THRESHOLD + 1
        return max(
            ALPHA_VISIBLE_THRESHOLD + 1,
            min(64, int(math.ceil(maximum * 0.25))),
        )

    @classmethod
    def _alpha_core_mask(cls, alpha: np.ndarray) -> np.ndarray:
        source = np.asarray(alpha, dtype=np.uint8)
        return source >= cls._alpha_core_threshold(source)

    @classmethod
    def _alpha_foreground_mask(cls, alpha: np.ndarray) -> np.ndarray:
        """保留字形和主要抗锯齿边缘，不把接近透明的背景残留写入掩码。"""
        source = np.asarray(alpha, dtype=np.uint8)
        threshold = max(
            ALPHA_VISIBLE_THRESHOLD + 1,
            cls._alpha_core_threshold(source) // 4,
        )
        return source >= threshold

    @staticmethod
    def _has_photoshop_transparency(image: Image.Image) -> bool:
        """解析 Photoshop 图层，判断可见图层合成后是否可能保留透明区。"""
        tags = getattr(image, "tag_v2", None)
        if tags is None:
            return False
        layer_data = tags.get(37724)
        if not isinstance(layer_data, (bytes, bytearray)):
            return False
        payload = bytes(layer_data)
        layers = OptimizationService._parse_photoshop_layers(payload)
        if not layers:
            return False

        canvas_width, canvas_height = image.size
        visible_layers = [
            layer for layer in layers
            if layer["不透明度"] > 0 and not layer["隐藏"]
        ]
        for layer in visible_layers:
            top, left, bottom, right = layer["范围"]
            covers_canvas = (
                top <= 0 and left <= 0
                and bottom >= canvas_height and right >= canvas_width
            )
            has_transparency_channel = any(
                channel_id in (-1, -2, -3) for channel_id in layer["通道"]
            )
            if covers_canvas and layer["不透明度"] == 255 and not has_transparency_channel:
                return False

        return any(
            any(channel_id in (-1, -2, -3) for channel_id in layer["通道"])
            for layer in visible_layers
        )

    @staticmethod
    def _parse_photoshop_layers(payload: bytes) -> list[dict[str, Any]]:
        """读取 TIFF ImageSourceData 中的基础图层记录，不解码图层像素。"""
        header = b"Adobe Photoshop Document Data Block\x00"
        if not payload.startswith(header):
            return []

        position = len(header)
        layer_block = b""
        byte_order = "<"
        while position + 12 <= len(payload):
            signature = payload[position:position + 4]
            key = payload[position + 4:position + 8]
            if signature == b"MIB8":
                byte_order = "<"
                layer_key = b"ryaL"
            elif signature == b"8BIM":
                byte_order = ">"
                layer_key = b"Layr"
            else:
                break
            block_length = struct.unpack_from(f"{byte_order}I", payload, position + 8)[0]
            block_start = position + 12
            block_end = block_start + block_length
            if block_end > len(payload):
                return []
            if key == layer_key:
                layer_block = payload[block_start:block_end]
                break
            position = block_end + (block_length % 2)

        if len(layer_block) < 2:
            return []
        layer_count = abs(struct.unpack_from(f"{byte_order}h", layer_block, 0)[0])
        if not 1 <= layer_count <= 2048:
            return []

        layers: list[dict[str, Any]] = []
        position = 2
        try:
            for _ in range(layer_count):
                if position + 18 > len(layer_block):
                    return []
                top, left, bottom, right = struct.unpack_from(
                    f"{byte_order}4i", layer_block, position
                )
                position += 16
                channel_count = struct.unpack_from(f"{byte_order}H", layer_block, position)[0]
                position += 2
                if channel_count > 64 or position + channel_count * 6 + 16 > len(layer_block):
                    return []
                channel_ids: list[int] = []
                for _channel in range(channel_count):
                    channel_ids.append(struct.unpack_from(f"{byte_order}h", layer_block, position)[0])
                    position += 6

                opacity = layer_block[position + 8]
                flags = layer_block[position + 10]
                extra_length = struct.unpack_from(f"{byte_order}I", layer_block, position + 12)[0]
                position += 16
                if position + extra_length > len(layer_block):
                    return []
                layers.append({
                    "范围": (top, left, bottom, right),
                    "通道": channel_ids,
                    "不透明度": opacity,
                    "隐藏": bool(flags & 0x02),
                })
                position += extra_length
        except (IndexError, struct.error):
            return []
        return layers

    @staticmethod
    def _load_white_background_gray(path: str) -> np.ndarray:
        """兼容原有调用：读取合成到白底后的灰度图。"""
        source_rgba, gray, _ = OptimizationService._load_source(path)
        source_rgba.close()
        return gray

    @staticmethod
    def _gray_to_transparent_image(
        gray: np.ndarray, mask: np.ndarray, protect_original: bool = False
    ) -> Image.Image:
        """以灰度母版提供墨色、清洁掩码限定字形，生成阶段性透明预览。"""
        gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
        alpha = 255 - gray_u8
        if not protect_original:
            alpha = np.where(mask > 0, alpha, 0).astype(np.uint8)
        rgba = np.zeros((*gray_u8.shape, 4), dtype=np.uint8)
        rgba[..., 3] = alpha
        return Image.fromarray(rgba, "RGBA")

    @staticmethod
    def _mask_to_transparent_image(mask: np.ndarray) -> Image.Image:
        foreground = mask > 0
        rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        rgba[foreground, :3] = 0
        rgba[foreground, 3] = 255
        return Image.fromarray(rgba, "RGBA")
