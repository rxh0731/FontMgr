"""通用外围空间污染过滤及其自动优化接入回归测试。"""

from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from core import algorithms, foreground_analysis, optimizer


def _binary(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask) > 0).astype(np.uint8)


def _draw_polyline(
    mask: np.ndarray,
    points: list[tuple[int, int]],
    thickness: int,
) -> None:
    cv2.polylines(
        mask,
        [np.asarray(points, dtype=np.int32)],
        False,
        255,
        thickness=thickness,
        lineType=cv2.LINE_8,
    )


def _central_separated_glyph(size: int = 192) -> np.ndarray:
    """构造由多个合法分离部件组成、但空间上形成稳定主体簇的字形。"""
    mask = np.zeros((size, size), dtype=np.uint8)
    _draw_polyline(mask, [(49, 59), (58, 112), (72, 130), (83, 91)], 11)
    cv2.ellipse(mask, (99, 67), (9, 13), -35, 0, 360, 255, -1)
    _draw_polyline(mask, [(128, 45), (122, 82), (108, 123)], 12)
    _draw_polyline(mask, [(144, 91), (163, 112)], 12)
    return mask


def _bottom_fragment_band(size: int = 192) -> np.ndarray:
    """构造不贴边、单块面积足以逃过普通面积过滤的碎裂污染带。"""
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.rectangle(mask, (27, 165), (39, 171), 255, -1)
    cv2.ellipse(mask, (61, 168), (5, 7), 20, 0, 360, 255, -1)
    cv2.rectangle(mask, (80, 164), (94, 170), 255, -1)
    _draw_polyline(mask, [(112, 171), (120, 163), (131, 170)], 4)
    cv2.rectangle(mask, (150, 165), (164, 171), 255, -1)
    return mask


def _single_outer_block(size: int = 192) -> np.ndarray:
    """构造面积不小、但薄长且与主体明显隔离的单块外围污染。"""
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.rectangle(mask, (55, 164), (109, 170), 255, -1)
    return mask


def _four_dot_glyph(size: int = 192) -> np.ndarray:
    """构造下方带四个合法分离点画的字形。"""
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.rectangle(mask, (44, 37), (148, 124), 255, thickness=10)
    cv2.line(mask, (62, 76), (132, 76), 255, thickness=9)
    cv2.line(mask, (96, 42), (96, 119), 255, thickness=9)
    for center in ((55, 145), (82, 147), (111, 147), (139, 145)):
        cv2.ellipse(mask, center, (7, 5), -20, 0, 360, 255, -1)
    return mask


def _heart_like_glyph(size: int = 192) -> np.ndarray:
    """构造含多个离散点画和弯钩主体的“心”式结构。"""
    mask = np.zeros((size, size), dtype=np.uint8)
    _draw_polyline(mask, [(64, 101), (69, 129), (87, 145), (128, 143), (145, 126)], 11)
    cv2.ellipse(mask, (42, 108), (7, 10), -30, 0, 360, 255, -1)
    cv2.ellipse(mask, (91, 81), (6, 11), -20, 0, 360, 255, -1)
    cv2.ellipse(mask, (148, 93), (7, 12), -35, 0, 360, 255, -1)
    return mask


def _long_bottom_stroke_glyph(size: int = 192) -> np.ndarray:
    """构造上部离散笔画与下部长底笔共同组成的“辶”式结构。"""
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(mask, (66, 48), (7, 12), -40, 0, 360, 255, -1)
    _draw_polyline(mask, [(91, 64), (121, 58), (137, 83)], 10)
    _draw_polyline(mask, [(56, 92), (69, 111), (57, 128)], 11)
    _draw_polyline(mask, [(43, 145), (67, 158), (112, 160), (158, 153)], 12)
    return mask


def _two_large_components(size: int = 192) -> np.ndarray:
    """构造两个面积相近的合法部件，避免把远离最大域等同于污染。"""
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.rectangle(mask, (31, 50), (70, 140), 255, thickness=11)
    cv2.rectangle(mask, (112, 43), (160, 143), 255, thickness=11)
    cv2.line(mask, (119, 93), (153, 93), 255, thickness=9)
    return mask


def _ambiguous_two_cluster_mask(size: int = 192) -> np.ndarray:
    """构造没有稳定主体归属的两个等价簇，检测应低置信回退。"""
    mask = np.zeros((size, size), dtype=np.uint8)
    for top in (34, 116):
        cv2.rectangle(mask, (54, top), (82, top + 35), 255, thickness=8)
        cv2.rectangle(mask, (111, top), (139, top + 35), 255, thickness=8)
    return mask


def _load_gray_image(path: Path) -> np.ndarray:
    """通过内存解码兼容 Windows 中文路径。"""
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise AssertionError(f"无法读取回归样本：{path}")
    return image.astype(np.float32)


def _find_lower_separator(mask: np.ndarray) -> int:
    """在图像下半部寻找主体和底部污染之间的水平空白分隔区。"""
    source = _binary(mask)
    height, width = source.shape
    row_ink = np.count_nonzero(source, axis=1)
    sparse_limit = max(1, int(round(width * 0.01)))
    search_start = int(round(height * 0.55))
    search_end = int(round(height * 0.90))
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for row in range(search_start, search_end):
        if int(row_ink[row]) <= sparse_limit:
            if run_start is None:
                run_start = row
        elif run_start is not None:
            runs.append((run_start, row))
            run_start = None
    if run_start is not None:
        runs.append((run_start, search_end))

    minimum_gap = max(3, int(round(height * 0.02)))
    minimum_lower_ink = max(20, int(round(np.count_nonzero(source) * 0.01)))
    valid = [
        (start, end)
        for start, end in runs
        if end - start >= minimum_gap
        and np.count_nonzero(source[end:]) >= minimum_lower_ink
    ]
    if not valid:
        raise AssertionError("真实样本中未找到主体与底部污染带之间的空白分隔区")
    _start, end = max(valid, key=lambda item: (item[1] - item[0], item[0]))
    return end


def _significant_lower_component_pixels(mask: np.ndarray, separator: int) -> int:
    """统计分隔区以下仍可见的显著连通域像素，忽略少量阈值毛刺。"""
    source = _binary(mask)
    lower = source[separator:]
    if lower.size == 0:
        return 0
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        lower,
        8,
        cv2.CV_32S,
    )
    minimum_area = max(6, int(round(source.size * 0.00012)))
    return sum(
        int(stats[label, cv2.CC_STAT_AREA])
        for label in range(1, count)
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area
    )


class ExternalPollutionFilterTests(unittest.TestCase):
    def test_removes_fragment_bands_from_all_four_directions(self) -> None:
        glyph = _central_separated_glyph()
        pollution = _bottom_fragment_band()
        direction_names = ("下方", "右侧", "上方", "左侧")

        for quarter_turns, direction_name in enumerate(direction_names):
            with self.subTest(direction=direction_name):
                source = np.rot90(glyph | pollution, quarter_turns).copy()
                expected = np.rot90(glyph, quarter_turns).copy()
                result = algorithms.external_pollution_filter(source)

                self.assertTrue(
                    np.array_equal(_binary(result), _binary(expected)),
                    f"{direction_name}远端碎裂污染带应被完整删除，文字主体必须保持不变",
                )

    def test_removes_one_large_thin_isolated_outer_block(self) -> None:
        glyph = _central_separated_glyph()
        source = glyph | _single_outer_block()

        result = algorithms.external_pollution_filter(source)
        strict_analysis = foreground_analysis.analyze_external_pollution(
            source,
            min_confidence=0.92,
        )

        self.assertTrue(
            np.array_equal(_binary(result), _binary(glyph)),
            "面积不小的外围污染块不能因普通面积阈值保护而残留",
        )
        self.assertTrue(strict_analysis.applied)
        self.assertGreaterEqual(strict_analysis.confidence, 0.92)

    def test_optimizer_exposes_a_clean_candidate_for_one_large_outer_block(self) -> None:
        glyph = _central_separated_glyph()
        source = glyph | _single_outer_block()
        gray = np.full(source.shape, 245, dtype=np.float32)
        gray[source > 0] = 25

        with (
            patch("core.optimizer.write_log"),
            patch("core.pipeline.write_log"),
        ):
            results = optimizer.generate_candidate_results(gray, limit=8)

        processed = [item for item in results if not item.get("保留原图", False)]
        self.assertTrue(
            any(np.array_equal(_binary(item["掩码"]), _binary(glyph)) for item in processed),
            "单个大块外围污染必须进入自动优化候选，而不是只在底层算法中可清理",
        )

    def test_preserves_four_legitimate_bottom_dots(self) -> None:
        source = _four_dot_glyph()

        result = algorithms.external_pollution_filter(source)

        self.assertTrue(np.array_equal(_binary(result), _binary(source)))

    def test_preserves_heart_like_separated_dots(self) -> None:
        source = _heart_like_glyph()

        result = algorithms.external_pollution_filter(source)

        self.assertTrue(np.array_equal(_binary(result), _binary(source)))

    def test_preserves_long_bottom_stroke(self) -> None:
        source = _long_bottom_stroke_glyph()

        result = algorithms.external_pollution_filter(source)

        self.assertTrue(np.array_equal(_binary(result), _binary(source)))

    def test_preserves_single_large_separated_component(self) -> None:
        source = _two_large_components()

        result = algorithms.external_pollution_filter(source)

        self.assertTrue(np.array_equal(_binary(result), _binary(source)))

    def test_returns_original_when_body_assignment_is_low_confidence(self) -> None:
        source = _ambiguous_two_cluster_mask()

        analysis = foreground_analysis.analyze_external_pollution(
            source,
            min_confidence=0.78,
        )

        self.assertFalse(analysis.applied)
        self.assertLess(analysis.confidence, 0.78)
        self.assertTrue(
            np.array_equal(_binary(analysis.cleaned_mask), _binary(source))
        )

    def test_optimizer_has_a_processed_candidate_without_bottom_pollution_band(self) -> None:
        source = _central_separated_glyph() | _bottom_fragment_band()
        gray = np.full(source.shape, 245, dtype=np.float32)
        gray[source > 0] = 25
        source_mask = optimizer._original_foreground_mask(gray)
        separator = _find_lower_separator(source_mask)
        original_lower_pixels = _significant_lower_component_pixels(
            source_mask,
            separator,
        )
        self.assertGreater(original_lower_pixels, 20, "回归样本应包含可识别的底部污染带")

        with (
            patch("core.optimizer.write_log"),
            patch("core.pipeline.write_log"),
        ):
            results = optimizer.generate_candidate_results(gray, limit=8)

        processed = [item for item in results if not item.get("保留原图", False)]
        clean_candidates = [
            item
            for item in processed
            if _significant_lower_component_pixels(item["掩码"], separator) == 0
        ]
        residuals = [
            (
                item.get("方案名"),
                _significant_lower_component_pixels(item["掩码"], separator),
            )
            for item in processed
        ]
        self.assertTrue(
            clean_candidates,
            "底部污染样本至少应生成一个完整清除污染带的非原图候选；"
            f"当前非原图候选残留像素：{residuals}",
        )


if __name__ == "__main__":
    unittest.main()
