"""自动寻优尺度、低对比、结构保护与多目标排序回归测试。"""

from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from core import optimizer, pipeline, scoring


def _gray_from_mask(mask: np.ndarray, ink: int = 24, background: int = 244) -> np.ndarray:
    gray = np.full(mask.shape, background, dtype=np.float32)
    gray[mask > 0] = ink
    return gray


def _ring_mask(size: int = 96) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.rectangle(mask, (18, 18), (size - 19, size - 19), 1, thickness=10)
    cv2.line(mask, (size // 2, 18), (size // 2, size - 19), 1, thickness=7)
    return mask


def _preview_evaluation(
    mask: np.ndarray,
    breakdown: scoring.ScoreBreakdown,
    *,
    rejected: bool = False,
    protection_note: str = "结构完整",
) -> optimizer._PreviewEvaluation:
    return optimizer._PreviewEvaluation(
        mask=mask,
        breakdown=breakdown,
        score_timing="测试缩略图评分",
        rejected=rejected,
        protection_note=protection_note,
        pipeline_elapsed=0.0,
        scoring_elapsed=0.0,
    )


class OptimizerCoreTests(unittest.TestCase):
    def test_auto_pick_passes_quality_level_to_candidate_builder(self) -> None:
        gray = _gray_from_mask(_ring_mask(64))
        with (
            patch("core.optimizer._evaluate_candidates", return_value=("保守方案", {"预处理": {}}, 95.0)),
            patch("core.optimizer.write_log"),
        ):
            name, _scheme, score, reached = optimizer.auto_pick_for_image(gray)

        self.assertEqual(name, "保守方案")
        self.assertEqual(score, 95.0)
        self.assertTrue(reached)

    def test_adaptive_scale_only_changes_marked_automatic_scheme(self) -> None:
        old_scheme = {
            "预处理": {},
            "L3": {"算法": "Sauvola", "参数": {"窗口": 25, "k": 0.2}},
            "L5": {"算法": "面积过滤", "参数": {"min_area": 60}},
        }
        self.assertIs(pipeline.resolve_adaptive_scheme(old_scheme, (256, 256)), old_scheme)

        automatic = {
            **old_scheme,
            "自适应尺度": {"基准宽度": 512, "基准高度": 512},
        }
        resolved = pipeline.resolve_adaptive_scheme(automatic, (256, 256))

        self.assertEqual(resolved["L3"]["参数"]["窗口"], 13)
        self.assertEqual(resolved["L5"]["参数"]["min_area"], 15)
        self.assertEqual(automatic["L3"]["参数"]["窗口"], 25)
        self.assertEqual(automatic["L5"]["参数"]["min_area"], 60)

    def test_low_contrast_routes_are_strictly_gated(self) -> None:
        low_mask = _ring_mask(96)
        low_gray = _gray_from_mask(low_mask, ink=145, background=205)
        normal_gray = _gray_from_mask(low_mask, ink=20, background=245)

        low_features = optimizer._auto_analyze(low_gray)
        low_context = scoring.build_score_context(low_gray)
        low_profile = optimizer._build_scale_profile(low_gray, low_context.reference_structure)
        low_names = {
            name
            for name, _scheme in optimizer._auto_build_candidates(
                low_features,
                low_profile,
                optimizer._classify_quality(low_features),
            )
        }

        normal_features = optimizer._auto_analyze(normal_gray)
        normal_context = scoring.build_score_context(normal_gray)
        normal_profile = optimizer._build_scale_profile(normal_gray, normal_context.reference_structure)
        normal_names = {
            name
            for name, _scheme in optimizer._auto_build_candidates(
                normal_features,
                normal_profile,
                optimizer._classify_quality(normal_features),
            )
        }

        self.assertTrue(low_features["low_contrast"])
        self.assertIn("低对比·受限CLAHE", low_names)
        self.assertIn("低对比·背景校正Triangle", low_names)
        self.assertFalse(normal_features["low_contrast"])
        self.assertFalse(any(name.startswith("低对比·") for name in normal_names))

    def test_structure_guard_rejects_cut_stroke_and_filled_hole(self) -> None:
        reference_mask = _ring_mask(96)
        gray = _gray_from_mask(reference_mask)
        context = scoring.build_score_context(gray)

        cut = reference_mask.copy()
        cut[43:53, 12:84] = 0
        cut_breakdown, _ = scoring.evaluate_candidate(cut, gray, context=context)
        cut_rejected, _ = optimizer._reject_structure_damage(cut, context, cut_breakdown)

        filled = reference_mask.copy()
        filled[24:72, 24:72] = 1
        filled_breakdown, _ = scoring.evaluate_candidate(filled, gray, context=context)
        filled_rejected, _ = optimizer._reject_structure_damage(filled, context, filled_breakdown)

        self.assertTrue(cut_rejected)
        self.assertTrue(filled_rejected)
        self.assertGreaterEqual(context.reference_structure.holes, 1)

    def test_structure_guard_treats_small_residue_as_noise_not_breakage(self) -> None:
        reference_mask = _ring_mask(128)
        reference_mask[1:10, 105:127] = 1
        gray = _gray_from_mask(reference_mask)
        context = scoring.build_score_context(gray)
        candidate = context.ref_mask.copy()
        for top, left in ((2, 2), (2, 32), (2, 50), (2, 68), (120, 2), (120, 32), (120, 50)):
            candidate[top:top + 6, left:left + 6] = 1

        unscaled_metrics = scoring.compute_structure_metrics(candidate)
        breakdown, _ = scoring.evaluate_candidate(candidate, gray, context=context)
        rejected, note = optimizer._reject_structure_damage(candidate, context, breakdown)

        self.assertEqual(context.ref_components, 2)
        self.assertGreater(unscaled_metrics.components, context.ref_components + 3)
        self.assertEqual(breakdown.structure.components, context.ref_components)
        self.assertGreater(breakdown.noise_components, 0)
        self.assertGreater(breakdown.noise_penalty, 0.0)
        self.assertFalse(rejected, note)

    def test_custom_reference_context_initializes_component_coverage(self) -> None:
        mask = _ring_mask(96)
        gray = _gray_from_mask(mask)
        custom_reference = mask.copy()
        custom_reference[18:28, 18:28] = 0

        context = scoring._build_context_from_reference(
            custom_reference,
            gray,
            gray.size,
        )

        self.assertEqual(context.ref_total, int(custom_reference.sum()))
        self.assertGreater(context.ref_component_ids.size, 0)
        self.assertEqual(
            scoring.minimum_reference_component_coverage(custom_reference, context),
            1.0,
        )

    def test_score_breakdown_exposes_pareto_fronts(self) -> None:
        mask = _ring_mask(64)
        gray = _gray_from_mask(mask)
        base, _ = scoring.evaluate_candidate(mask, gray)
        clean_but_weaker_structure = replace(
            base,
            score=91.0,
            topology_stability=0.72,
            stroke_retention=0.78,
            core_retention=0.80,
            background_cleanliness=1.0,
            ratio_plausibility=1.0,
        )
        stable_but_less_clean = replace(
            base,
            score=86.0,
            topology_stability=1.0,
            stroke_retention=1.0,
            core_retention=1.0,
            background_cleanliness=0.80,
            ratio_plausibility=1.0,
        )
        dominated = replace(
            base,
            score=70.0,
            topology_stability=0.60,
            stroke_retention=0.60,
            core_retention=0.60,
            background_cleanliness=0.70,
            ratio_plausibility=0.80,
        )

        ranks = scoring.pareto_front_ranks([
            clean_but_weaker_structure,
            stable_but_less_clean,
            dominated,
        ])

        self.assertEqual(ranks[:2], [0, 0])
        self.assertGreater(ranks[2], 0)
        self.assertIn("拓扑稳定", base.as_dict())

    def test_reference_gray_is_separate_and_shape_checked(self) -> None:
        processing_mask = _ring_mask(64)
        processing_mask[28:36, 23:29] = 1
        processing = _gray_from_mask(processing_mask, ink=90, background=220)
        reference = _gray_from_mask(_ring_mask(64), ink=20, background=245)
        original_builder = scoring.build_score_context

        with (
            patch("core.optimizer._auto_build_candidates", return_value=[
                ("原图保护", {"保护原图": True, "预处理": {}}),
            ]),
            patch("core.optimizer.scoring.build_score_context", wraps=original_builder) as build_context,
            patch("core.optimizer.write_log"),
        ):
            results = optimizer.generate_candidate_results(
                processing,
                limit=1,
                reference_gray_arr=reference,
            )

        self.assertTrue(results)
        self.assertTrue(np.array_equal(build_context.call_args_list[0].args[0], reference))
        self.assertEqual(int(results[0]["掩码"][32, 25]), 1)
        self.assertEqual(int(_ring_mask(64)[32, 25]), 0)
        with self.assertRaisesRegex(ValueError, "尺寸必须一致"):
            optimizer.generate_candidate_results(processing, reference_gray_arr=reference[:32, :32])

    def test_full_size_stage_replaces_preview_score(self) -> None:
        gray = _gray_from_mask(_ring_mask(64))
        template, _ = scoring.evaluate_candidate(_ring_mask(64), gray)
        preview = replace(template, score=41.0)
        full_size = replace(template, score=88.0)

        with (
            patch("core.optimizer._auto_build_candidates", return_value=[
                ("原图保护", {"保护原图": True, "预处理": {}}),
            ]),
            patch("core.optimizer.scoring.evaluate_candidate", side_effect=[
                (preview, "缩略图评分"),
                (full_size, "全尺寸评分"),
            ]),
            patch("core.optimizer.write_log"),
        ):
            results = optimizer.generate_candidate_results(gray, limit=1)

        self.assertEqual(results[0]["得分"], 88.0)
        self.assertEqual(results[0]["评分明细"]["综合得分"], 88.0)

    def test_limit_one_skips_unreturned_original_full_size_score(self) -> None:
        original_mask = _ring_mask(64)
        optimized_mask = original_mask.copy()
        optimized_mask[2:16, 2:16] = 1
        gray = _gray_from_mask(original_mask)
        template, _ = scoring.evaluate_candidate(original_mask, gray)
        schemes = [
            ("原图保护", {"测试编号": "原图", "保护原图": True, "预处理": {}}),
            ("唯一寻优结果", {"测试编号": "寻优", "预处理": {}}),
        ]

        def evaluate_preview(
            _thumb: np.ndarray,
            _reference: np.ndarray,
            _context: scoring.ScoreContext,
            scheme: dict[str, object],
            *_args: object,
            **_kwargs: object,
        ) -> tuple[optimizer._PreviewEvaluation, bool]:
            is_original = scheme["测试编号"] == "原图"
            mask = original_mask if is_original else optimized_mask
            score = 90.0 if is_original else 88.0
            return _preview_evaluation(mask.copy(), replace(template, score=score)), False

        with (
            patch("core.optimizer._auto_build_candidates", return_value=schemes),
            patch("core.optimizer._classify_quality", return_value="已足够干净"),
            patch("core.optimizer._evaluate_preview_scheme", side_effect=evaluate_preview),
            patch("core.optimizer._original_foreground_mask") as original_foreground,
            patch(
                "core.optimizer.pipeline.run_pipeline",
                return_value=(gray, optimized_mask.copy()),
            ) as run_pipeline,
            patch(
                "core.optimizer.scoring.evaluate_candidate",
                return_value=(replace(template, score=86.0), "寻优全尺寸评分"),
            ) as evaluate_candidate,
            patch("core.optimizer._reject_structure_damage", return_value=(False, "结构完整")),
            patch("core.optimizer.write_log"),
        ):
            results = optimizer.generate_candidate_results(gray, limit=1)

        original_foreground.assert_not_called()
        run_pipeline.assert_called_once()
        evaluate_candidate.assert_called_once()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["方案名"], "唯一寻优结果")
        self.assertEqual(results[0]["方案"]["测试编号"], "寻优")

    def test_full_size_identical_masks_reuse_score_and_keep_first_scheme(self) -> None:
        base_mask = _ring_mask(64)
        distinct_mask = base_mask.copy()
        distinct_mask[1:17, 1:17] = 1
        gray = _gray_from_mask(base_mask)
        template, _ = scoring.evaluate_candidate(base_mask, gray)
        schemes = [
            ("首选方案", {"测试编号": "A", "预处理": {}}),
            ("重复方案", {"测试编号": "B", "预处理": {}}),
            ("补位方案", {"测试编号": "C", "预处理": {}}),
        ]
        preview_masks: dict[str, np.ndarray] = {}
        for key, bounds in {
            "A": (3, 3, 23, 23),
            "B": (37, 3, 60, 27),
            "C": (15, 37, 49, 60),
        }.items():
            mask = np.zeros_like(base_mask)
            cv2.rectangle(mask, bounds[:2], bounds[2:], 1, thickness=-1)
            preview_masks[key] = mask
        preview_scores = {"A": 96.0, "B": 92.0, "C": 88.0}

        def evaluate_preview(
            _thumb: np.ndarray,
            _reference: np.ndarray,
            _context: scoring.ScoreContext,
            scheme: dict[str, object],
            *_args: object,
            **_kwargs: object,
        ) -> tuple[optimizer._PreviewEvaluation, bool]:
            key = str(scheme["测试编号"])
            breakdown = replace(template, score=preview_scores[key])
            return _preview_evaluation(preview_masks[key].copy(), breakdown), False

        def run_full_size(
            _gray: np.ndarray,
            scheme: dict[str, object],
            **_kwargs: object,
        ) -> tuple[np.ndarray, np.ndarray]:
            mask = distinct_mask if scheme["测试编号"] == "C" else base_mask
            return gray, mask.copy()

        with (
            patch("core.optimizer._auto_build_candidates", return_value=schemes),
            patch("core.optimizer._classify_quality", return_value="已足够干净"),
            patch("core.optimizer._evaluate_preview_scheme", side_effect=evaluate_preview),
            patch(
                "core.optimizer.pipeline.run_pipeline",
                side_effect=run_full_size,
            ) as run_pipeline,
            patch(
                "core.optimizer.scoring.evaluate_candidate",
                side_effect=[
                    (replace(template, score=90.0), "首选全尺寸评分"),
                    (replace(template, score=82.0), "补位全尺寸评分"),
                ],
            ) as evaluate_candidate,
            patch("core.optimizer._reject_structure_damage", return_value=(False, "结构完整")),
            patch("core.optimizer.write_log") as write_log,
        ):
            results = optimizer.generate_candidate_results(gray, limit=2)

        self.assertEqual(run_pipeline.call_count, 3)
        self.assertEqual(evaluate_candidate.call_count, 2)
        self.assertEqual(
            [result["方案"]["测试编号"] for result in results],
            ["A", "C"],
        )
        self.assertNotIn("B", {result["方案"]["测试编号"] for result in results})
        self.assertTrue(
            any(
                "重复方案" in str(call.args[0])
                and "复用完全相同掩码" in str(call.args[0])
                for call in write_log.call_args_list
            )
        )

    def test_full_size_duplicate_rechecks_structure_guard_without_rescoring(self) -> None:
        safe_mask = _ring_mask(64)
        unsafe_mask = np.zeros_like(safe_mask)
        cv2.rectangle(unsafe_mask, (3, 3), (30, 60), 1, thickness=-1)
        gray = _gray_from_mask(safe_mask)
        template, _ = scoring.evaluate_candidate(safe_mask, gray)
        schemes = [
            ("不安全方案A", {"测试编号": "A", "预处理": {}}),
            ("不安全方案B", {"测试编号": "B", "预处理": {}}),
            ("安全补位", {"测试编号": "C", "预处理": {}}),
        ]
        preview_masks = {
            "A": np.pad(np.ones((20, 20), dtype=np.uint8), ((2, 42), (2, 42))),
            "B": np.pad(np.ones((20, 20), dtype=np.uint8), ((2, 42), (42, 2))),
            "C": np.pad(np.ones((20, 20), dtype=np.uint8), ((42, 2), (22, 22))),
        }
        preview_scores = {"A": 96.0, "B": 92.0, "C": 88.0}

        def evaluate_preview(
            _thumb: np.ndarray,
            _reference: np.ndarray,
            _context: scoring.ScoreContext,
            scheme: dict[str, object],
            *_args: object,
            **_kwargs: object,
        ) -> tuple[optimizer._PreviewEvaluation, bool]:
            key = str(scheme["测试编号"])
            breakdown = replace(template, score=preview_scores[key])
            return _preview_evaluation(preview_masks[key].copy(), breakdown), False

        def run_full_size(
            _gray: np.ndarray,
            scheme: dict[str, object],
            **_kwargs: object,
        ) -> tuple[np.ndarray, np.ndarray]:
            mask = safe_mask if scheme["测试编号"] == "C" else unsafe_mask
            return gray, mask.copy()

        def reject_unsafe(
            candidate: np.ndarray,
            _context: scoring.ScoreContext,
            _breakdown: scoring.ScoreBreakdown,
        ) -> tuple[bool, str]:
            rejected = np.array_equal(candidate, unsafe_mask)
            return rejected, "结构损坏" if rejected else "结构完整"

        with (
            patch("core.optimizer._auto_build_candidates", return_value=schemes),
            patch("core.optimizer._classify_quality", return_value="已足够干净"),
            patch("core.optimizer._evaluate_preview_scheme", side_effect=evaluate_preview),
            patch("core.optimizer.pipeline.run_pipeline", side_effect=run_full_size),
            patch(
                "core.optimizer.scoring.evaluate_candidate",
                side_effect=[
                    (replace(template, score=90.0), "不安全全尺寸评分"),
                    (replace(template, score=84.0), "安全全尺寸评分"),
                ],
            ) as evaluate_candidate,
            patch(
                "core.optimizer._reject_structure_damage",
                side_effect=reject_unsafe,
            ) as reject_structure,
            patch("core.optimizer.write_log"),
        ):
            results = optimizer.generate_candidate_results(gray, limit=1)

        self.assertEqual(evaluate_candidate.call_count, 2)
        self.assertEqual(reject_structure.call_count, 3)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["方案"]["测试编号"], "C")

    def test_low_gain_safe_optimized_result_is_not_removed(self) -> None:
        mask = _ring_mask(64)
        gray = _gray_from_mask(mask)
        template, _ = scoring.evaluate_candidate(mask, gray)
        original_score = replace(template, score=95.0)
        optimized_score = replace(template, score=90.0)
        original_scheme = {"测试编号": "原图", "保护原图": True, "预处理": {}}
        optimized_scheme = {"测试编号": "寻优", "预处理": {}}

        def evaluate_preview(
            _thumb: np.ndarray,
            _reference: np.ndarray,
            _context: scoring.ScoreContext,
            scheme: dict[str, object],
            *_args: object,
            **_kwargs: object,
        ) -> tuple[optimizer._PreviewEvaluation, bool]:
            breakdown = original_score if scheme["测试编号"] == "原图" else optimized_score
            return _preview_evaluation(mask.copy(), breakdown), False

        with (
            patch("core.optimizer._auto_build_candidates", return_value=[
                ("原图保护", original_scheme),
                ("安全寻优", optimized_scheme),
            ]),
            patch("core.optimizer._classify_quality", return_value="已足够干净"),
            patch("core.optimizer._evaluate_preview_scheme", side_effect=evaluate_preview),
            patch("core.optimizer.pipeline.run_pipeline", return_value=(gray, mask.copy())),
            patch("core.optimizer.scoring.evaluate_candidate", side_effect=[
                (original_score, "原图全尺寸评分"),
                (optimized_score, "寻优全尺寸评分"),
            ]),
            patch("core.optimizer._reject_structure_damage", return_value=(False, "结构完整")),
            patch("core.optimizer.write_log"),
        ):
            results = optimizer.generate_candidate_results(gray, limit=2)

        self.assertEqual(len(results), 2)
        optimized = next(result for result in results if not result["保留原图"])
        self.assertEqual(optimized["方案名"], "安全寻优")

    def test_full_size_verification_backfills_after_unsafe_candidate(self) -> None:
        original_mask = _ring_mask(64)
        unsafe_mask = np.zeros_like(original_mask)
        cv2.rectangle(unsafe_mask, (4, 4), (30, 58), 1, thickness=-1)
        gray = _gray_from_mask(original_mask)
        template, _ = scoring.evaluate_candidate(original_mask, gray)
        scores = {
            "原图": replace(template, score=90.0),
            "首选": replace(template, score=96.0),
            "补位": replace(template, score=85.0),
        }
        schemes = [
            ("原图保护", {"测试编号": "原图", "保护原图": True, "预处理": {}}),
            ("首选但结构不安全", {"测试编号": "首选", "预处理": {}}),
            ("安全补位", {"测试编号": "补位", "预处理": {}}),
        ]

        def evaluate_preview(
            _thumb: np.ndarray,
            _reference: np.ndarray,
            _context: scoring.ScoreContext,
            scheme: dict[str, object],
            *_args: object,
            **_kwargs: object,
        ) -> tuple[optimizer._PreviewEvaluation, bool]:
            key = str(scheme["测试编号"])
            preview_mask = unsafe_mask if key == "首选" else original_mask
            return _preview_evaluation(preview_mask.copy(), scores[key]), False

        def run_full_size(
            _gray: np.ndarray,
            scheme: dict[str, object],
            **_kwargs: object,
        ) -> tuple[np.ndarray, np.ndarray]:
            mask = unsafe_mask if scheme["测试编号"] == "首选" else original_mask
            return gray, mask.copy()

        def reject_unsafe(
            candidate: np.ndarray,
            _context: scoring.ScoreContext,
            _breakdown: scoring.ScoreBreakdown,
        ) -> tuple[bool, str]:
            rejected = np.array_equal(candidate, unsafe_mask)
            return rejected, "结构损坏" if rejected else "结构完整"

        with (
            patch("core.optimizer._auto_build_candidates", return_value=schemes),
            patch("core.optimizer._classify_quality", return_value="中度污染"),
            patch("core.optimizer._beam_rank", return_value=[]),
            patch("core.optimizer._evaluate_preview_scheme", side_effect=evaluate_preview),
            patch("core.optimizer.pipeline.run_pipeline", side_effect=run_full_size) as run_pipeline,
            patch("core.optimizer.scoring.evaluate_candidate", side_effect=[
                (scores["原图"], "原图全尺寸评分"),
                (scores["首选"], "首选全尺寸评分"),
                (scores["补位"], "补位全尺寸评分"),
            ]),
            patch("core.optimizer._reject_structure_damage", side_effect=reject_unsafe),
            patch("core.optimizer.write_log"),
        ):
            results = optimizer.generate_candidate_results(gray, limit=2)

        self.assertEqual(run_pipeline.call_count, 2)
        self.assertIn("安全补位", {result["方案名"] for result in results})
        self.assertNotIn("首选但结构不安全", {result["方案名"] for result in results})

    def test_all_structure_rejections_return_real_risk_candidates(self) -> None:
        base_mask = _ring_mask(64)
        alternate_mask = base_mask.copy()
        alternate_mask[2:10, 2:10] = 1
        gray = _gray_from_mask(base_mask)
        template, _ = scoring.evaluate_candidate(base_mask, gray)
        schemes = [
            ("保守Otsu", {"测试编号": "A", "预处理": {}}),
            ("清理方案", {"测试编号": "B", "预处理": {}}),
        ]

        def evaluate_preview(
            _thumb: np.ndarray,
            _reference: np.ndarray,
            _context: scoring.ScoreContext,
            scheme: dict[str, object],
            *_args: object,
            **_kwargs: object,
        ) -> tuple[optimizer._PreviewEvaluation, bool]:
            mask = base_mask if scheme["测试编号"] == "A" else alternate_mask
            score = 72.0 if scheme["测试编号"] == "A" else 81.0
            return _preview_evaluation(
                mask.copy(),
                replace(template, score=score),
                rejected=True,
                protection_note="有意义孔洞仅保留0.0%",
            ), False

        def run_full_size(
            _gray: np.ndarray,
            scheme: dict[str, object],
            **_kwargs: object,
        ) -> tuple[np.ndarray, np.ndarray]:
            mask = base_mask if scheme["测试编号"] == "A" else alternate_mask
            return gray, mask.copy()

        with (
            patch("core.optimizer._auto_build_candidates", return_value=schemes),
            patch("core.optimizer._classify_quality", return_value="已足够干净"),
            patch("core.optimizer._evaluate_preview_scheme", side_effect=evaluate_preview),
            patch("core.optimizer.pipeline.run_pipeline", side_effect=run_full_size),
            patch(
                "core.optimizer.scoring.evaluate_candidate",
                side_effect=[
                    (replace(template, score=74.0), "保守全尺寸评分"),
                    (replace(template, score=83.0), "清理全尺寸评分"),
                ],
            ),
            patch(
                "core.optimizer._reject_structure_damage",
                return_value=(True, "有意义孔洞仅保留0.0%"),
            ),
            patch("core.optimizer.write_log"),
        ):
            results = optimizer.generate_candidate_results(gray, limit=2)

        self.assertTrue(results)
        self.assertTrue(all(not result.get("保留原图") for result in results))
        self.assertTrue(all(np.any(result["掩码"] > 0) for result in results))
        self.assertTrue(
            all(result["结构复核"]["状态"] == "需人工核对" for result in results)
        )
        self.assertTrue(
            all(result["结构复核"]["阶段"] == "原尺寸复核" for result in results)
        )

    def test_safe_candidate_prevents_risk_candidate_from_being_returned(self) -> None:
        base_mask = _ring_mask(64)
        risk_mask = base_mask.copy()
        risk_mask[2:14, 2:14] = 1
        gray = _gray_from_mask(base_mask)
        template, _ = scoring.evaluate_candidate(base_mask, gray)
        schemes = [
            ("安全方案", {"测试编号": "safe", "预处理": {}}),
            ("风险方案", {"测试编号": "risk", "预处理": {}}),
        ]

        def evaluate_preview(
            _thumb: np.ndarray,
            _reference: np.ndarray,
            _context: scoring.ScoreContext,
            scheme: dict[str, object],
            *_args: object,
            **_kwargs: object,
        ) -> tuple[optimizer._PreviewEvaluation, bool]:
            is_risk = scheme["测试编号"] == "risk"
            return _preview_evaluation(
                risk_mask.copy() if is_risk else base_mask.copy(),
                replace(template, score=99.0 if is_risk else 82.0),
                rejected=is_risk,
                protection_note="端点异常增加" if is_risk else "结构完整",
            ), False

        with (
            patch("core.optimizer._auto_build_candidates", return_value=schemes),
            patch("core.optimizer._classify_quality", return_value="已足够干净"),
            patch("core.optimizer._evaluate_preview_scheme", side_effect=evaluate_preview),
            patch(
                "core.optimizer.pipeline.run_pipeline",
                return_value=(gray, base_mask.copy()),
            ) as run_pipeline,
            patch(
                "core.optimizer.scoring.evaluate_candidate",
                return_value=(replace(template, score=84.0), "安全全尺寸评分"),
            ),
            patch(
                "core.optimizer._reject_structure_damage",
                return_value=(False, "结构完整"),
            ),
            patch("core.optimizer.write_log"),
        ):
            results = optimizer.generate_candidate_results(gray, limit=2)

        self.assertEqual(run_pipeline.call_count, 1)
        self.assertEqual([result["方案名"] for result in results], ["安全方案"])
        self.assertNotIn("结构复核", results[0])

    def test_risk_ranking_prefers_coverage_safe_result_before_score(self) -> None:
        mask = _ring_mask(64)
        gray = _gray_from_mask(mask)
        template, _ = scoring.evaluate_candidate(mask, gray)
        lower_risk = {
            "方案名": "覆盖完整",
            "得分": 70.0,
            "掩码": mask.copy(),
            "评分明细": template.as_dict(),
            "_评分对象": replace(template, score=70.0),
            "结构复核": {"状态": "需人工核对", "风险等级": 1},
        }
        higher_risk = {
            "方案名": "覆盖受损",
            "得分": 99.0,
            "掩码": mask.copy(),
            "评分明细": template.as_dict(),
            "_评分对象": replace(template, score=99.0),
            "结构复核": {"状态": "需人工核对", "风险等级": 2},
        }
        results = [higher_risk, lower_risk]

        optimizer._rank_risky_results(results)
        selected = optimizer._select_lowest_risk_results(results, 2)

        self.assertEqual(results[0]["方案名"], "覆盖完整")
        self.assertEqual([result["方案名"] for result in selected], ["覆盖完整"])

    def test_cancellation_is_checked_around_full_score_context(self) -> None:
        gray = _gray_from_mask(_ring_mask(64))
        checks = iter((False, True))

        with (
            patch("core.optimizer.scoring.build_score_context", wraps=scoring.build_score_context) as build_context,
            patch("core.optimizer.write_log"),
            self.assertRaises(optimizer.OptimizationCancelled),
        ):
            optimizer.generate_candidate_results(gray, cancel_check=lambda: next(checks))

        build_context.assert_called_once()

    def test_thinning_can_cancel_between_iteration_phases(self) -> None:
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.rectangle(mask, (16, 16), (239, 239), 1, thickness=-1)
        check_count = 0

        def cancel_during_thinning() -> None:
            nonlocal check_count
            check_count += 1
            if check_count >= 3:
                raise optimizer.OptimizationCancelled("测试在骨架细化中停止")

        with self.assertRaisesRegex(optimizer.OptimizationCancelled, "骨架细化"):
            scoring._thin_mask(mask, cancel_check=cancel_during_thinning)

        self.assertEqual(check_count, 3)

    def test_optimizer_cancel_query_is_forwarded_into_score_context(self) -> None:
        gray = _gray_from_mask(_ring_mask(128))
        query_count = 0

        def cancel_query() -> bool:
            nonlocal query_count
            query_count += 1
            return query_count >= 3

        with (
            patch("core.optimizer.write_log"),
            self.assertRaises(optimizer.OptimizationCancelled),
        ):
            optimizer.generate_candidate_results(gray, cancel_check=cancel_query)

        self.assertEqual(query_count, 3)

    def test_cancellation_is_checked_after_preview_candidate(self) -> None:
        mask = _ring_mask(64)
        gray = _gray_from_mask(mask)
        template, _ = scoring.evaluate_candidate(mask, gray)
        cancelled = False

        def evaluate_preview(*_args: object, **_kwargs: object) -> tuple[optimizer._PreviewEvaluation, bool]:
            nonlocal cancelled
            cancelled = True
            return _preview_evaluation(mask.copy(), template), False

        with (
            patch("core.optimizer._auto_build_candidates", return_value=[
                ("安全寻优", {"测试编号": "寻优", "预处理": {}}),
            ]),
            patch("core.optimizer._classify_quality", return_value="已足够干净"),
            patch("core.optimizer._evaluate_preview_scheme", side_effect=evaluate_preview),
            patch("core.optimizer.pipeline.run_pipeline") as run_pipeline,
            patch("core.optimizer.write_log"),
            self.assertRaises(optimizer.OptimizationCancelled),
        ):
            optimizer.generate_candidate_results(
                gray,
                cancel_check=lambda: cancelled,
            )

        run_pipeline.assert_not_called()


if __name__ == "__main__":
    unittest.main()
