"""笔画尺度核心重建的通用行为、结构保护与真实样本回归测试。"""

from __future__ import annotations

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from core import algorithms, optimizer, pipeline, scoring, stroke_scale_analysis
from core.stroke_scale_analysis import (
    ReconstructionStrength,
    analyze_stroke_scale,
    reconstruct_three_strengths,
)
from data.registry_store import get_builtin_registry


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _binary(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask) > 0).astype(np.uint8)


def _recall(expected: np.ndarray, actual: np.ndarray) -> float:
    expected_mask = _binary(expected)
    total = int(expected_mask.sum())
    if total == 0:
        return 1.0
    return float((_binary(actual) & expected_mask).sum()) / total


def _add_sparse_noise(
    gray: np.ndarray,
    protected_mask: np.ndarray,
    *,
    ink: int = 112,
) -> int:
    """在合法结构之外加入确定性、互不连通的单像素细噪。"""
    exclusion = cv2.dilate(
        _binary(protected_mask),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
    )
    count = 0
    for row_index, y in enumerate(range(7, gray.shape[0] - 7, 9)):
        offset = 4 if row_index % 2 else 0
        for x in range(7 + offset, gray.shape[1] - 7, 9):
            if exclusion[y, x]:
                continue
            gray[y, x] = ink
            count += 1
    return count


def _dense_noise_gray(size: int = 192) -> np.ndarray:
    body = np.zeros((size, size), dtype=np.uint8)
    cv2.rectangle(body, (46, 38), (146, 145), 1, thickness=14)
    cv2.line(body, (96, 40), (96, 143), 1, thickness=11)
    cv2.line(body, (48, 91), (144, 91), 1, thickness=9)

    gray = np.full(body.shape, 245, dtype=np.uint8)
    gray[body > 0] = 38
    noise_count = _add_sparse_noise(gray, body)
    if noise_count < 80:
        raise AssertionError("合成图未生成足够的独立细噪")
    return gray


def _protected_structure_gray() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """构造带合法离散部件、窄桥和孔洞的非特定字符结构。"""
    size = 224
    main = np.zeros((size, size), dtype=np.uint8)
    cv2.rectangle(main, (42, 38), (104, 122), 1, thickness=14)
    cv2.rectangle(main, (120, 38), (182, 122), 1, thickness=14)

    bridge = np.zeros_like(main)
    cv2.line(bridge, (103, 80), (121, 80), 1, thickness=3)
    main |= bridge

    dot = np.zeros_like(main)
    cv2.circle(dot, (54, 146), 4, 1, thickness=-1)

    long_stroke = np.zeros_like(main)
    cv2.line(long_stroke, (116, 149), (174, 149), 1, thickness=3)

    protected = main | dot | long_stroke
    gray = np.full(protected.shape, 245, dtype=np.uint8)
    gray[main > 0] = 38
    gray[(dot | long_stroke) > 0] = 85
    noise_count = _add_sparse_noise(gray, protected)
    if noise_count < 100:
        raise AssertionError("结构保护合成图未生成足够的独立细噪")

    left_hole = np.zeros_like(main)
    left_hole[52:109, 56:91] = 1
    right_hole = np.zeros_like(main)
    right_hole[52:109, 134:169] = 1
    return gray, {
        "独立点画": dot,
        "长细笔": long_stroke,
        "窄桥": bridge,
        "孔洞": left_hole | right_hole,
    }


def _load_gray_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise AssertionError(f"无法读取回归样本：{path}")
    return image.astype(np.float32)


class StrokeScaleReconstructionTests(unittest.TestCase):
    def test_session_cache_reuses_analysis_without_array_aliases(self) -> None:
        gray = _dense_noise_gray(160)
        original_runner = stroke_scale_analysis._analyze_stroke_scale_uncached

        with patch.object(
            stroke_scale_analysis,
            "_analyze_stroke_scale_uncached",
            wraps=original_runner,
        ) as uncached:
            with stroke_scale_analysis.stroke_scale_analysis_session():
                first = analyze_stroke_scale(gray)
                second = analyze_stroke_scale(gray.astype(np.float32))
                self.assertEqual(uncached.call_count, 1)
                self.assertFalse(np.shares_memory(first.base_mask, second.base_mask))

                expected = int(first.base_mask[0, 0])
                second.base_mask[0, 0] = 1 - expected
                third = analyze_stroke_scale(gray)
                self.assertEqual(int(third.base_mask[0, 0]), expected)
                self.assertEqual(uncached.call_count, 1)

            analyze_stroke_scale(gray)
            self.assertEqual(uncached.call_count, 2)

    def test_three_strengths_are_subsets_and_remove_monotonically(self) -> None:
        gray = _dense_noise_gray()

        analysis, result_map = reconstruct_three_strengths(gray)
        results = [result_map[strength.value] for strength in ReconstructionStrength]

        self.assertTrue(analysis.applicable, analysis.reason)
        self.assertGreaterEqual(analysis.noise_component_count, 80)
        base = _binary(analysis.base_mask)
        areas: list[int] = []
        for result in results:
            mask = _binary(result.mask)
            areas.append(int(mask.sum()))
            self.assertEqual(
                int(np.count_nonzero(mask & (1 - base))),
                0,
                f"{result.strength.value}档不得在基准掩码外新增墨点",
            )
            self.assertEqual(int(result.added_mask.sum()), 0)
            self.assertGreaterEqual(
                _recall(analysis.core_mask, mask),
                0.94,
                f"{result.strength.value}档必须保留稳定笔画核心",
            )

        self.assertLess(areas[0], int(base.sum()), "保守档也应清除明确独立细噪")
        self.assertGreaterEqual(areas[0], areas[1])
        self.assertGreaterEqual(areas[1], areas[2])

    def test_low_confidence_thin_input_falls_back_pixel_for_pixel(self) -> None:
        gray = np.full((160, 160), 245, dtype=np.uint8)
        cv2.line(gray, (24, 80), (136, 80), 30, thickness=1)
        cv2.line(gray, (80, 46), (80, 114), 30, thickness=1)

        analysis, result_map = reconstruct_three_strengths(gray)

        self.assertFalse(analysis.applicable)
        for strength in ReconstructionStrength:
            result = result_map[strength.value]
            self.assertFalse(result.applied)
            self.assertTrue(
                np.array_equal(result.mask, analysis.base_mask),
                f"低置信时{strength.value}档必须逐像素回退到基准掩码",
            )

    def test_preserves_legitimate_parts_narrow_bridge_and_holes(self) -> None:
        gray, regions = _protected_structure_gray()

        analysis, result_map = reconstruct_three_strengths(gray)

        self.assertTrue(analysis.applicable, analysis.reason)
        self.assertGreaterEqual(analysis.metrics["受保护细笔连通域数"], 2.0)
        for strength in ReconstructionStrength:
            result = result_map[strength.value]
            for name in ("独立点画", "长细笔", "窄桥"):
                self.assertGreaterEqual(
                    _recall(regions[name], result.mask),
                    0.98,
                    f"{strength.value}档不应破坏{name}",
                )
            self.assertEqual(
                int(np.count_nonzero(_binary(result.mask) & regions["孔洞"])),
                0,
                f"{strength.value}档不应填平主体孔洞",
            )

    def test_preserves_many_coherent_separated_points(self) -> None:
        body = np.zeros((224, 224), dtype=np.uint8)
        cv2.rectangle(body, (55, 40), (175, 135), 1, thickness=14)
        points = np.zeros_like(body)
        centers = [(35 + index % 5 * 38, 160 + index // 5 * 28) for index in range(9)]
        for center in centers:
            cv2.circle(points, center, 4, 1, thickness=-1)
        gray = np.full(body.shape, 245, dtype=np.uint8)
        gray[body > 0] = 38
        gray[points > 0] = 60
        _add_sparse_noise(gray, body | points)

        analysis, result_map = reconstruct_three_strengths(gray)

        self.assertTrue(analysis.applicable, analysis.reason)
        self.assertEqual(analysis.metrics["受保护细笔连通域数"], 9.0)
        for strength in ReconstructionStrength:
            self.assertGreaterEqual(
                _recall(points, result_map[strength.value].mask),
                0.98,
                f"{strength.value}档不应因固定数量上限删除合法离散点画",
            )

    def test_conservative_strength_has_no_fixed_ambiguous_part_limit(self) -> None:
        body = np.zeros((256, 256), dtype=np.uint8)
        cv2.rectangle(body, (55, 40), (175, 135), 1, thickness=14)
        points = np.zeros_like(body)
        centers = [
            (28 + column * 40, y)
            for y in (160, 190, 220, 238)
            for column in range(5)
        ][:17]
        gray = np.full(body.shape, 245, dtype=np.uint8)
        gray[body > 0] = 38
        for center in centers:
            point = np.zeros_like(body)
            cv2.circle(point, center, 5, 1, thickness=-1)
            points |= point
            rows, columns = np.where(point > 0)
            gray[rows, columns] = np.linspace(60, 110, rows.size).astype(np.uint8)
        _add_sparse_noise(gray, body | points)

        analysis, result_map = reconstruct_three_strengths(gray)

        self.assertTrue(analysis.applicable, analysis.reason)
        self.assertGreater(len(centers), 16)
        self.assertGreaterEqual(
            _recall(points, result_map["保守"].mask),
            0.98,
            "保形档不得按固定数量截断低置信歧义点画",
        )

    def test_preserves_shallow_points_and_diagonal_short_strokes(self) -> None:
        for ink in (70, 85, 100):
            with self.subTest(kind="浅墨点画", ink=ink):
                body = np.zeros((224, 224), dtype=np.uint8)
                cv2.rectangle(body, (55, 40), (175, 135), 1, thickness=14)
                part = np.zeros_like(body)
                cv2.circle(part, (45, 165), 5, 1, thickness=-1)
                gray = np.full(body.shape, 245, dtype=np.uint8)
                gray[body > 0] = 38
                gray[part > 0] = ink
                _add_sparse_noise(gray, body | part)
                analysis, result_map = reconstruct_three_strengths(gray)
                self.assertTrue(analysis.applicable, analysis.reason)
                for strength in ReconstructionStrength:
                    self.assertGreaterEqual(
                        _recall(part, result_map[strength.value].mask),
                        0.98,
                    )

        with self.subTest(kind="灰度波动浅墨点画", ink="100±12"):
            body = np.zeros((224, 224), dtype=np.uint8)
            cv2.rectangle(body, (55, 40), (175, 135), 1, thickness=14)
            part = np.zeros_like(body)
            cv2.circle(part, (45, 165), 5, 1, thickness=-1)
            gray = np.full(body.shape, 245, dtype=np.uint8)
            gray[body > 0] = 38
            point_y, point_x = np.where(part > 0)
            gray[point_y, point_x] = 100
            outside_center = (point_y != 165) | (point_x != 45)
            varying_y = point_y[outside_center]
            varying_x = point_x[outside_center]
            gray[varying_y[::2], varying_x[::2]] = 88
            gray[varying_y[1::2], varying_x[1::2]] = 112
            gray[165, 45] = 100
            _add_sparse_noise(gray, body | part)

            analysis, result_map = reconstruct_three_strengths(gray)

            self.assertTrue(analysis.applicable, analysis.reason)
            for strength in ReconstructionStrength:
                self.assertGreaterEqual(
                    _recall(part, result_map[strength.value].mask),
                    0.98,
                    f"{strength.value}档不应删除带灰度波动的浅墨点画",
                )

        for length in (30, 34, 38):
            with self.subTest(kind="斜向短笔", length=length):
                body = np.zeros((224, 224), dtype=np.uint8)
                cv2.rectangle(body, (55, 40), (175, 135), 1, thickness=14)
                part = np.zeros_like(body)
                delta = int(round(length / np.sqrt(2.0)))
                cv2.line(part, (100, 155), (100 + delta, 155 + delta), 1, thickness=3)
                gray = np.full(body.shape, 245, dtype=np.uint8)
                gray[body > 0] = 38
                gray[part > 0] = 85
                _add_sparse_noise(gray, body | part)
                analysis, result_map = reconstruct_three_strengths(gray)
                self.assertTrue(analysis.applicable, analysis.reason)
                for strength in ReconstructionStrength:
                    self.assertGreaterEqual(
                        _recall(part, result_map[strength.value].mask),
                        0.98,
                    )

    def test_preserves_textured_shallow_point(self) -> None:
        body = np.zeros((224, 224), dtype=np.uint8)
        cv2.rectangle(body, (55, 40), (175, 135), 1, thickness=14)
        part = np.zeros_like(body)
        cv2.circle(part, (45, 165), 5, 1, thickness=-1)
        gray = np.full(body.shape, 245, dtype=np.uint8)
        gray[body > 0] = 38
        rows, columns = np.where(part > 0)
        order = np.lexsort((columns, rows))
        gray[rows[order], columns[order]] = np.linspace(
            60,
            110,
            order.size,
        ).astype(np.uint8)
        _add_sparse_noise(gray, body | part)

        analysis, result_map = reconstruct_three_strengths(gray)

        self.assertTrue(analysis.applicable, analysis.reason)
        self.assertGreaterEqual(analysis.metrics["浅墨区域MAD上限"], 16.0)
        for strength in ReconstructionStrength:
            self.assertGreaterEqual(
                _recall(part, result_map[strength.value].mask),
                0.98,
                f"{strength.value}档不应删除内部灰度有自然波动的浅墨点画",
            )

    def test_preserves_connected_long_thin_terminal_stroke(self) -> None:
        main = np.zeros((224, 224), dtype=np.uint8)
        cv2.rectangle(main, (45, 35), (165, 135), 1, thickness=14)
        for thickness in (2, 3, 4):
            with self.subTest(thickness=thickness):
                tail = np.zeros_like(main)
                cv2.line(
                    tail,
                    (163, 128),
                    (215, 205),
                    1,
                    thickness=thickness,
                    lineType=cv2.LINE_8,
                )
                glyph = main | tail
                gray = np.full(glyph.shape, 245, dtype=np.uint8)
                gray[glyph > 0] = 38
                _add_sparse_noise(gray, glyph)
                terminal = tail & (
                    1 - cv2.dilate(main, np.ones((3, 3), dtype=np.uint8))
                )

                analysis, result_map = reconstruct_three_strengths(gray)

                self.assertTrue(analysis.applicable, analysis.reason)
                self.assertEqual(
                    cv2.connectedComponents(glyph, connectivity=8)[0] - 1,
                    1,
                    "测试末笔必须与粗主体属于同一连通域",
                )
                for strength in ReconstructionStrength:
                    self.assertGreaterEqual(
                        _recall(terminal, result_map[strength.value].mask),
                        0.98,
                        f"{strength.value}档不应截断连接主体的长细末笔",
                    )

    def test_scoring_rejects_candidate_missing_one_shallow_component(self) -> None:
        body = np.zeros((224, 224), dtype=np.uint8)
        cv2.rectangle(body, (55, 40), (175, 135), 1, thickness=14)
        points = np.zeros_like(body)
        for index in range(9):
            cv2.circle(
                points,
                (35 + index % 5 * 38, 160 + index // 5 * 28),
                4,
                1,
                thickness=-1,
            )
        gray = np.full(body.shape, 245, dtype=np.uint8)
        gray[body > 0] = 38
        gray[points > 0] = 85
        _add_sparse_noise(gray, body | points)

        context = scoring.build_score_context(gray.astype(np.float32))
        self.assertGreaterEqual(context.ref_component_ids.size, 9)
        smallest_index = int(np.argmin(context.ref_component_areas))
        missing_id = int(context.ref_component_ids[smallest_index])
        candidate = context.ref_mask.copy()
        candidate[context.ref_label_map == missing_id] = 0
        breakdown, _ = scoring.evaluate_candidate(candidate, gray, context=context)

        rejected, reason = optimizer._reject_structure_damage(candidate, context, breakdown)
        self.assertTrue(rejected)
        self.assertIn("独立笔画", reason)
        self.assertEqual(
            scoring.minimum_reference_component_coverage(candidate, context),
            0.0,
        )

    def test_peripheral_same_scale_blob_is_not_forced_into_stable_body(self) -> None:
        body = np.zeros((224, 224), dtype=np.uint8)
        for x, y in ((55, 45), (85, 45), (55, 105), (85, 105)):
            cv2.rectangle(body, (x, y), (x + 16, y + 42), 1, thickness=-1)
        pollution = np.zeros_like(body)
        cv2.circle(pollution, (205, 200), 10, 1, thickness=-1)
        gray = np.full(body.shape, 245, dtype=np.uint8)
        gray[body > 0] = 38
        gray[pollution > 0] = 38
        _add_sparse_noise(gray, body | pollution)

        analysis, result_map = reconstruct_three_strengths(gray)

        self.assertTrue(analysis.applicable, analysis.reason)
        self.assertEqual(analysis.metrics["主体粗部件数"], 4.0)
        self.assertEqual(analysis.metrics["外围粗部件数"], 1.0)
        self.assertGreaterEqual(_recall(body, result_map["保守"].mask), 0.98)
        self.assertGreaterEqual(_recall(pollution, result_map["保守"].mask), 0.98)
        for strength in ("均衡", "强力"):
            self.assertGreaterEqual(_recall(body, result_map[strength].mask), 0.94)
            self.assertLessEqual(_recall(pollution, result_map[strength].mask), 0.02)

    def test_falls_back_when_frame_pollution_becomes_largest_anchor(self) -> None:
        pollution = np.zeros((256, 256), dtype=np.uint8)
        cv2.rectangle(pollution, (5, 8), (100, 245), 1, thickness=14)
        for y in (70, 145, 210):
            cv2.line(pollution, (20, y), (86, y), 1, thickness=14)
        glyph = np.zeros_like(pollution)
        cv2.line(glyph, (185, 80), (185, 175), 1, thickness=8)
        cv2.line(glyph, (150, 125), (225, 125), 1, thickness=8)
        source = pollution | glyph
        gray = np.full(source.shape, 245, dtype=np.uint8)
        gray[source > 0] = 38
        _add_sparse_noise(gray, source)

        analysis, result_map = reconstruct_three_strengths(gray)

        self.assertFalse(analysis.applicable)
        self.assertEqual(analysis.metrics["主体锚点歧义"], 1.0)
        self.assertIn("主体锚点", analysis.reason)
        for strength in ReconstructionStrength:
            result = result_map[strength.value]
            self.assertFalse(result.applied)
            self.assertTrue(np.array_equal(result.mask, analysis.base_mask))
            self.assertGreaterEqual(_recall(glyph, result.mask), 0.98)

    def test_pipeline_builtin_registry_and_json_registry_are_consistent(self) -> None:
        algorithm_name = "笔画尺度核心重建"
        expected_parameters = {
            "重建级别": {"类型": "int", "范围": [0, 2], "默认": 1},
            "最小置信度": {"类型": "float", "范围": [0.6, 0.98], "默认": 0.78},
            "最少细噪域": {"类型": "int", "范围": [4, 200], "默认": 8},
        }
        builtin_entry = get_builtin_registry()["分组"]["L3 二值化"]["算法"][algorithm_name]
        registry_path = PROJECT_ROOT / "配置" / "算法注册表.json"
        disk_registry = json.loads(registry_path.read_text(encoding="utf-8-sig"))
        disk_entry = disk_registry["分组"]["L3 二值化"]["算法"][algorithm_name]

        self.assertIn(algorithm_name, pipeline._ALGO_DISPATCH["L3"])
        self.assertEqual(builtin_entry, disk_entry)
        self.assertEqual(builtin_entry["参数"], expected_parameters)

        gray = _dense_noise_gray(160)
        for level in range(3):
            params = {
                "重建级别": level,
                "最小置信度": 0.78,
                "最少细噪域": 8,
            }
            via_pipeline = pipeline.algo_run(gray, "L3", algorithm_name, params)
            direct = algorithms.stroke_scale_core_reconstruct(
                gray,
                strength_level=level,
                min_confidence=0.78,
                minimum_noise_components=8,
            )
            self.assertTrue(np.array_equal(via_pipeline, direct))

    def test_test1_nai_generates_four_candidates_with_three_processed(self) -> None:
        sample_path = PROJECT_ROOT / "tests" / "fixtures" / "optimizer" / "乃-0001.tif"
        self.assertTrue(sample_path.is_file(), f"缺少回归样本：{sample_path}")
        gray = _load_gray_image(sample_path)

        with (
            patch("core.optimizer.write_log"),
            patch("core.pipeline.write_log"),
        ):
            results = optimizer.generate_candidate_results(gray, limit=4)

        processed = [item for item in results if not item.get("保留原图", False)]
        self.assertGreaterEqual(
            len(results),
            4,
            f"“乃”应生成至少4个候选，实际方案：{[item.get('方案名') for item in results]}",
        )
        self.assertGreaterEqual(
            len(processed),
            3,
            f"“乃”应至少有3个非原图候选，实际方案：{[item.get('方案名') for item in results]}",
        )
        original_mask = algorithms.otsu_binarize(gray)
        original_foreground = int(_binary(original_mask).sum())
        filtered = cv2.medianBlur(np.clip(gray, 0, 255).astype(np.uint8), 3)
        _, filtered_otsu = cv2.threshold(
            filtered,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(
            _binary(filtered_otsu),
            connectivity=8,
            ltype=cv2.CV_32S,
        )
        ordered_labels = sorted(
            range(1, component_count),
            key=lambda label: int(component_stats[label, cv2.CC_STAT_AREA]),
            reverse=True,
        )
        expected_body = np.isin(component_labels, ordered_labels[:2]).astype(np.uint8)
        known_pollution = np.isin(component_labels, ordered_labels[2:4]).astype(np.uint8)
        clean_candidates: list[str] = []
        for item in processed:
            mask = _binary(item["掩码"])
            component_count = cv2.connectedComponents(mask, connectivity=8)[0] - 1
            body_recall = _recall(expected_body, mask)
            pollution_recall = _recall(known_pollution, mask)
            if (
                component_count <= 3
                and int(mask.sum()) <= original_foreground * 0.48
                and body_recall >= 0.94
                and pollution_recall <= 0.02
            ):
                clean_candidates.append(str(item.get("方案名", "")))
        self.assertTrue(
            clean_candidates,
            "“乃”至少应有一个清除密集背景和外围粗噪团的结构完整候选",
        )


if __name__ == "__main__":
    unittest.main()
