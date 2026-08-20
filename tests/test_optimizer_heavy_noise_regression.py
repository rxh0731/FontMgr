"""重污染候选生成的结果、结构保护与束搜索保底回归测试。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from core import optimizer, pipeline, scoring


def _ring_mask(size: int = 96) -> np.ndarray:
    """构造带端点和孔洞、同时便于稳定评分的测试字形。"""
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.rectangle(mask, (18, 18), (size - 19, size - 19), 1, thickness=10)
    cv2.line(mask, (size // 2, 18), (size // 2, size - 19), 1, thickness=7)
    return mask


def _gray_from_mask(mask: np.ndarray) -> np.ndarray:
    gray = np.full(mask.shape, 244, dtype=np.float32)
    gray[mask > 0] = 24
    return gray


def _load_gray_image(path: Path) -> np.ndarray:
    """通过内存解码兼容 Windows 中文路径。"""
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise AssertionError(f"无法读取回归样本：{path}")
    return image.astype(np.float32)


class OptimizerHeavyNoiseRegressionTests(unittest.TestCase):
    def test_test1_he_glyph_keeps_at_least_one_processed_candidate(self) -> None:
        sample_path = (
            Path(__file__).resolve().parent
            / "fixtures"
            / "optimizer"
            / "何-0001.tif"
        )
        self.assertTrue(sample_path.is_file(), f"缺少回归样本：{sample_path}")
        gray = _load_gray_image(sample_path)

        with (
            patch("core.optimizer.write_log"),
            patch("core.pipeline.write_log"),
            patch(
                "core.optimizer.scoring.evaluate_candidate",
                wraps=scoring.evaluate_candidate,
            ) as evaluate_candidate,
        ):
            results = optimizer.generate_candidate_results(gray, limit=4)

        processed = [item for item in results if not item.get("保留原图", False)]
        self.assertTrue(
            processed,
            f"“何”字候选不能只剩原图，实际方案：{[item.get('方案名') for item in results]}",
        )
        self.assertLess(
            evaluate_candidate.call_count,
            60,
            "重污染首轮候选不应对所有中间束方案执行完整骨架评分",
        )

    def test_seeded_reconstruction_only_grows_from_deep_ink_core(self) -> None:
        gray = np.full((64, 64), 240, dtype=np.float32)
        gray[26:38, 18:30] = 20
        gray[26:38, 30:44] = 120
        gray[6:12, 50:56] = 120

        mask = pipeline.algo_run(
            gray,
            "L3",
            "双阈值种子重建",
            {"核心偏移": -28, "生长偏移": 18},
        )

        self.assertEqual(int(mask[30, 22]), 255)
        self.assertEqual(int(mask[30, 38]), 255)
        self.assertEqual(int(mask[8, 52]), 0)

    def test_heavy_noise_does_not_use_unreliable_endpoints_as_hard_rejection(self) -> None:
        gray = _gray_from_mask(_ring_mask())
        context = scoring.build_score_context(gray)
        candidate = context.ref_mask.copy()
        breakdown, _ = scoring.evaluate_candidate(candidate, gray, context=context)
        reference_structure = replace(
            context.reference_structure,
            endpoint_count=4,
        )
        context = replace(context, reference_structure=reference_structure)

        unreliable_topology = replace(
            breakdown.comparison,
            endpoint_retention=0.0,
            endpoint_growth=8,
            hole_retention=0.0,
            extra_holes=max(3, context.reference_structure.holes + 2),
        )
        unreliable_breakdown = replace(breakdown, comparison=unreliable_topology)
        trusted_context = replace(context, heavy_noise=False, structure_confidence=1.0)
        heavy_noise_context = replace(context, heavy_noise=True, structure_confidence=1.0)

        trusted_rejected, _ = optimizer._reject_structure_damage(
            candidate,
            trusted_context,
            unreliable_breakdown,
        )
        heavy_rejected, heavy_note = optimizer._reject_structure_damage(
            candidate,
            heavy_noise_context,
            unreliable_breakdown,
        )

        self.assertTrue(trusted_rejected)
        self.assertFalse(heavy_rejected, heavy_note)

    def test_heavy_noise_beam_rank_keeps_a_fallback_route(self) -> None:
        gray = _gray_from_mask(_ring_mask())
        context = replace(
            scoring.build_score_context(gray),
            heavy_noise=True,
            structure_confidence=1.0,
        )
        candidate = context.ref_mask.copy()
        breakdown, _ = scoring.evaluate_candidate(candidate, gray, context=context)
        schemes = [
            ("重噪路线A", {"测试编号": "A"}),
            ("重噪路线B", {"测试编号": "B"}),
        ]

        with (
            patch("core.optimizer.scoring.build_score_context", return_value=context),
            patch("core.optimizer.scoring.evaluate_candidate", return_value=(breakdown, "模拟评分")),
            patch("core.optimizer.pipeline.run_pipeline", return_value=(gray, candidate)),
            patch(
                "core.optimizer._reject_structure_damage",
                return_value=(True, "端点异常增加19个"),
            ),
        ):
            selected = optimizer._beam_rank(gray, schemes, keep=2)

        self.assertTrue(selected, "重污染束搜索不能因端点软风险在首层完全坍缩")
        self.assertIn(selected[0][0], {"重噪路线A", "重噪路线B"})

        coverage_damaged = np.zeros_like(candidate)
        coverage_damaged[30:40, 30:40] = 1
        with patch(
            "core.optimizer.pipeline.run_pipeline",
            return_value=(gray, coverage_damaged),
        ):
            fast_selected = optimizer._beam_rank(
                gray,
                schemes,
                keep=2,
                score_context=context,
                full_structure=False,
            )

        self.assertTrue(
            fast_selected,
            "重污染中间束应允许少量软路线继续到可修复覆盖的后续阶段",
        )


if __name__ == "__main__":
    unittest.main()
