"""自动优化原图的通用质量分类。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image

from core.foreground_analysis import analyze_external_pollution


SOURCE_TYPE_TRANSPARENT = "已有有效透明区"
SOURCE_TYPE_WHITE_CLEANED = "白底已清理"
SOURCE_TYPE_UNPROCESSED = "未处理截图"
TRANSPARENCY_SOURCE_STANDARD_ALPHA = "标准Alpha"
TRANSPARENCY_SOURCE_PHOTOSHOP_ALPHA = "已解码Photoshop Alpha"
TRANSPARENCY_SOURCE_PHOTOSHOP_METADATA = "Photoshop图层元数据"
ACTUAL_ALPHA_SOURCES = frozenset((
    TRANSPARENCY_SOURCE_STANDARD_ALPHA,
    TRANSPARENCY_SOURCE_PHOTOSHOP_ALPHA,
))
ALPHA_TRANSPARENT_THRESHOLD = 250
ALPHA_VISIBLE_THRESHOLD = 5

_METRIC_MAX_SIDE = 512


@dataclass(frozen=True)
class SourceClassification:
    """原图分类结果及可追溯判定信息。"""

    source_type: str
    confidence: float
    metrics: dict[str, float | int | str]
    reasons: tuple[str, ...]

    def as_metadata(self) -> dict[str, Any]:
        return {
            "类型": self.source_type,
            "置信度": round(float(self.confidence), 4),
            "指标": dict(self.metrics),
            "判定依据": list(self.reasons),
        }


def classify_source(
    source_rgba: Image.Image,
    gray: np.ndarray,
    transparency_source: str = "",
) -> SourceClassification:
    """按有效透明、白底稳定度及背景污染指标划分三类原图。"""
    alpha_image = source_rgba.getchannel("A")
    try:
        alpha = np.array(alpha_image, dtype=np.uint8, copy=True)
    finally:
        alpha_image.close()
    alpha_thumb = _bounded_alpha(alpha)
    alpha_total = max(1, int(alpha.size))
    transparent_count = int(np.count_nonzero(alpha < ALPHA_TRANSPARENT_THRESHOLD))
    fully_transparent_count = int(np.count_nonzero(alpha <= ALPHA_VISIBLE_THRESHOLD))
    visible_count = int(np.count_nonzero(alpha > ALPHA_VISIBLE_THRESHOLD))
    transparent_fraction = transparent_count / alpha_total
    fully_transparent_fraction = fully_transparent_count / alpha_total
    visible_fraction = visible_count / alpha_total
    alpha_analysis_total = max(1, int(alpha_thumb.size))
    (
        edge_alpha_count,
        edge_alpha_fraction,
        edge_alpha_share,
        edge_alpha_sides,
    ) = _edge_connected_alpha_metrics(alpha_thumb)
    min_edge_fraction = 1.0 / alpha_analysis_total if alpha_analysis_total <= 16 else 0.02
    min_edge_count = (
        1
        if alpha_analysis_total <= 16
        else max(4, int(np.ceil(alpha_analysis_total * 0.002)))
    )
    # “口、国、因”等字形的大内腔会拉低外缘连通率。正常图片必须由同一个
    # 强透明连通域触达画布四边；微型测试图没有足够几何信息，保留单点特例。
    if alpha_analysis_total <= 16:
        edge_background_valid = edge_alpha_share >= 0.80
    else:
        edge_background_valid = (
            edge_alpha_sides == 4
            and (edge_alpha_share >= 0.80 or edge_alpha_fraction >= 0.12)
        )
    decoded_alpha_valid = (
        transparency_source in ACTUAL_ALPHA_SOURCES
        and fully_transparent_count >= min_edge_count
        and edge_alpha_count >= min_edge_count
        and edge_alpha_fraction >= min_edge_fraction
        and edge_background_valid
        and visible_fraction >= 0.001
    )

    alpha_metrics: dict[str, float | int | str] = {
        "透明来源": transparency_source or "无",
        "透明像素占比": round(transparent_fraction, 6),
        "全透明像素占比": round(fully_transparent_fraction, 6),
        "可见像素占比": round(visible_fraction, 6),
        "边缘连通透明像素数": edge_alpha_count,
        "边缘连通透明像素占比": round(edge_alpha_fraction, 6),
        "透明像素边缘连通率": round(edge_alpha_share, 6),
        "边缘连通透明触边数": edge_alpha_sides,
        "Photoshop透明元数据未解码": int(
            transparency_source in (
                TRANSPARENCY_SOURCE_PHOTOSHOP_METADATA,
                "Photoshop图层",
            )
            and transparent_count == 0
        ),
        "Alpha分析宽度": int(alpha_thumb.shape[1]),
        "Alpha分析高度": int(alpha_thumb.shape[0]),
        "分析宽度": int(alpha_thumb.shape[1]),
        "分析高度": int(alpha_thumb.shape[0]),
    }
    if decoded_alpha_valid:
        alpha_strength = min(1.0, edge_alpha_fraction / 0.20)
        confidence = 0.90 + 0.09 * alpha_strength
        alpha_label = (
            "Photoshop 图层 Alpha"
            if transparency_source == TRANSPARENCY_SOURCE_PHOTOSHOP_ALPHA
            else "标准 Alpha"
        )
        return SourceClassification(
            SOURCE_TYPE_TRANSPARENT,
            min(0.99, confidence),
            alpha_metrics,
            (f"检测到面积充足且与画布边缘连通的{alpha_label}透明背景",),
        )

    gray_thumb = _bounded_gray(gray)
    background_metrics = _background_metrics(gray_thumb)
    metrics: dict[str, float | int | str] = {
        **background_metrics,
        **alpha_metrics,
        "分析宽度": int(gray_thumb.shape[1]),
        "分析高度": int(gray_thumb.shape[0]),
    }

    white_score = float(background_metrics["白底可信分"])
    edge_white_ratio = float(background_metrics["边缘近白占比"])
    edge_highlight_mean = float(background_metrics["边缘高亮平均灰度"])
    background_std = float(background_metrics["背景灰度标准差"])
    illumination = float(background_metrics["光照变化"])
    midtone_pollution = float(background_metrics["中间调污染占比"])
    speck_count = int(background_metrics["散点数量"])
    speck_ratio = float(background_metrics["散点前景占比"])
    external_block_count = int(background_metrics["疑似大块外部污染数量"])
    white_cleaned = (
        white_score >= 0.78
        and edge_white_ratio >= 0.88
        and edge_highlight_mean >= 248.0
        and background_std <= 12.0
        and illumination <= 10.0
        and midtone_pollution <= 0.08
        and speck_count <= 8
        and speck_ratio <= 0.025
        and external_block_count == 0
    )
    if white_cleaned:
        confidence = min(0.98, max(0.72, 0.58 + (white_score - 0.78) * 1.5))
        reasons = (
            "画布边缘接近纯白",
            "背景灰度与光照变化较小",
            "中间调和孤立散点污染比例较低",
        )
        return SourceClassification(SOURCE_TYPE_WHITE_CLEANED, confidence, metrics, reasons)

    confidence = min(0.98, max(0.55, 0.56 + (0.78 - white_score) * 0.85))
    reasons_list: list[str] = []
    if edge_white_ratio < 0.88 or edge_highlight_mean < 248.0:
        reasons_list.append("边缘背景不是稳定纯白")
    if background_std > 12.0 or illumination > 10.0:
        reasons_list.append("背景存在明显灰度或光照变化")
    if midtone_pollution > 0.08:
        reasons_list.append("背景中间调污染比例较高")
    if speck_count > 8 or speck_ratio > 0.025:
        reasons_list.append("存在较多孤立散点污染")
    if external_block_count:
        reasons_list.append("存在与文字主体明显分离的大块外部污染")
    if transparency_source in ACTUAL_ALPHA_SOURCES and transparent_count:
        if fully_transparent_count == 0:
            reasons_list.append("Alpha 只有接近不透明的残差，没有强透明背景")
        else:
            reasons_list.append("Alpha 强透明区面积不足或未由同一连通域触达画布四边")
    if (
        transparency_source
        in (TRANSPARENCY_SOURCE_PHOTOSHOP_METADATA, "Photoshop图层")
        and transparent_count == 0
    ):
        reasons_list.append("Photoshop 透明元数据未实际解码出 Alpha")
    if not reasons_list:
        reasons_list.append("白底清理证据不足")
    return SourceClassification(
        SOURCE_TYPE_UNPROCESSED,
        confidence,
        metrics,
        tuple(reasons_list),
    )


def _bounded_gray(gray: np.ndarray) -> np.ndarray:
    source = np.clip(np.asarray(gray), 0, 255).astype(np.uint8)
    if source.ndim != 2:
        raise ValueError("原图分类只接受二维灰度图。")
    height, width = source.shape
    longest = max(height, width)
    if longest <= _METRIC_MAX_SIDE:
        return source
    ratio = _METRIC_MAX_SIDE / longest
    target = (max(1, int(round(width * ratio))), max(1, int(round(height * ratio))))
    source_image = Image.fromarray(source, "L")
    try:
        resized = source_image.resize(target, Image.Resampling.BOX)
        try:
            return np.array(resized, dtype=np.uint8, copy=True)
        finally:
            resized.close()
    finally:
        source_image.close()


def _bounded_alpha(alpha: np.ndarray) -> np.ndarray:
    """限制 Alpha 连通域分析尺寸，最近邻缩放保持透明拓扑。"""
    source = np.clip(np.asarray(alpha), 0, 255).astype(np.uint8)
    if source.ndim != 2:
        raise ValueError("Alpha 分类只接受二维数组。")
    height, width = source.shape
    longest = max(height, width)
    if longest <= _METRIC_MAX_SIDE:
        return source
    ratio = _METRIC_MAX_SIDE / longest
    target = (max(1, int(round(width * ratio))), max(1, int(round(height * ratio))))
    source_image = Image.fromarray(source, "L")
    try:
        resized = source_image.resize(target, Image.Resampling.NEAREST)
        try:
            return np.array(resized, dtype=np.uint8, copy=True)
        finally:
            resized.close()
    finally:
        source_image.close()


def _background_metrics(gray: np.ndarray) -> dict[str, float | int]:
    height, width = gray.shape
    band = max(1, min(height, width) // 12)
    edge_mask = np.zeros(gray.shape, dtype=bool)
    edge_mask[:band, :] = True
    edge_mask[-band:, :] = True
    edge_mask[:, :band] = True
    edge_mask[:, -band:] = True
    edge = gray[edge_mask].astype(np.float32)
    light_background = gray[gray >= 180].astype(np.float32)
    if light_background.size == 0:
        light_background = gray.reshape(-1).astype(np.float32)

    cell_means: list[float] = []
    for row in np.array_split(gray, min(4, height), axis=0):
        for cell in np.array_split(row, min(4, width), axis=1):
            values = cell[cell >= 180]
            if values.size:
                cell_means.append(float(np.mean(values)))
    illumination = max(cell_means) - min(cell_means) if len(cell_means) >= 2 else 0.0
    edge_white_ratio = float(np.mean(edge >= 245))
    edge_mean = float(np.mean(edge))
    edge_highlight = edge[edge >= 245]
    edge_highlight_mean = (
        float(np.mean(edge_highlight)) if edge_highlight.size else edge_mean
    )
    background_std = float(np.std(light_background))
    global_white_ratio = float(np.mean(gray >= 245))
    midtone_pollution = float(np.mean((gray >= 180) & (gray < 245)))
    pollution_metrics = _foreground_pollution_metrics(gray)
    component_count = int(pollution_metrics["前景连通域数量"])
    speck_count = int(pollution_metrics["散点数量"])
    speck_ratio = float(pollution_metrics["散点前景占比"])
    external_block_count = int(pollution_metrics["疑似大块外部污染数量"])
    external_block_ratio = float(pollution_metrics["大块外部污染前景占比"])

    edge_ratio_score = np.clip((edge_white_ratio - 0.72) / 0.28, 0.0, 1.0)
    edge_mean_score = np.clip((edge_highlight_mean - 248.0) / 7.0, 0.0, 1.0)
    variation_score = 1.0 - np.clip(background_std / 20.0, 0.0, 1.0)
    illumination_score = 1.0 - np.clip(illumination / 18.0, 0.0, 1.0)
    global_white_score = np.clip((global_white_ratio - 0.40) / 0.40, 0.0, 1.0)
    base_white_score = (
        0.34 * edge_ratio_score
        + 0.20 * edge_mean_score
        + 0.18 * variation_score
        + 0.14 * illumination_score
        + 0.14 * global_white_score
    )
    pollution_penalty = min(
        0.65,
        speck_count / 80.0
        + speck_ratio * 1.8
        + (0.28 if external_block_count else 0.0)
        + external_block_ratio * 0.8,
    )
    white_score = max(0.0, float(base_white_score) - pollution_penalty)
    return {
        "边缘近白占比": round(edge_white_ratio, 6),
        "边缘平均灰度": round(edge_mean, 4),
        "边缘高亮平均灰度": round(edge_highlight_mean, 4),
        "背景灰度标准差": round(background_std, 4),
        "光照变化": round(float(illumination), 4),
        "全图近白占比": round(global_white_ratio, 6),
        "中间调污染占比": round(midtone_pollution, 6),
        "前景连通域数量": component_count,
        "散点数量": speck_count,
        "散点前景占比": round(speck_ratio, 6),
        "疑似大块外部污染数量": external_block_count,
        "大块外部污染前景占比": round(external_block_ratio, 6),
        "污染指标": round(max(1.0 - float(base_white_score), pollution_penalty), 6),
        "白底可信分": round(float(white_score), 6),
    }


def _edge_connected_alpha_metrics(alpha: np.ndarray) -> tuple[int, float, float, int]:
    """统计最可信的单个外缘强透明连通域，排除内部或分散 Alpha 瑕疵。"""
    transparent = (
        np.asarray(alpha) <= ALPHA_VISIBLE_THRESHOLD
    ).astype(np.uint8)
    transparent_total = int(transparent.sum())
    if not transparent_total:
        return 0, 0.0, 0.0, 0
    count, labels = cv2.connectedComponents(transparent, connectivity=8)
    if count <= 1:
        return 0, 0.0, 0.0, 0
    edge_labels = np.unique(
        np.concatenate((labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]))
    )
    edge_labels = edge_labels[edge_labels > 0]
    if not edge_labels.size:
        return 0, 0.0, 0.0, 0
    edge_components: list[tuple[int, int]] = []
    for label in edge_labels:
        component = labels == int(label)
        component_count = int(component.sum())
        touched_sides = sum((
            bool(component[0, :].any()),
            bool(component[-1, :].any()),
            bool(component[:, 0].any()),
            bool(component[:, -1].any()),
        ))
        edge_components.append((component_count, touched_sides))
    edge_connected_count, touched_sides = max(
        edge_components,
        key=lambda item: (item[1], item[0]),
    )
    total = max(1, int(alpha.size))
    return (
        edge_connected_count,
        edge_connected_count / total,
        edge_connected_count / transparent_total,
        touched_sides,
    )


def _foreground_pollution_metrics(gray: np.ndarray) -> dict[str, float | int]:
    """用有界 Otsu 前景统计孤立散点及远离主体的大块外围前景。"""
    threshold, _binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    foreground = (gray <= threshold).astype(np.uint8)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        foreground,
        connectivity=8,
        ltype=cv2.CV_32S,
    )
    component_ids = list(range(1, count))
    component_ids.sort(key=lambda index: int(stats[index, cv2.CC_STAT_AREA]), reverse=True)
    areas = [int(stats[index, cv2.CC_STAT_AREA]) for index in component_ids]
    if not areas:
        return {
            "前景连通域数量": 0,
            "散点数量": 0,
            "散点前景占比": 0.0,
            "疑似大块外部污染数量": 0,
            "大块外部污染前景占比": 0.0,
        }
    # 一个汉字可能包含多个合法分离部件，先保护最大的八个，再统计剩余碎片。
    speck_areas = areas[8:]
    foreground_total = max(1, sum(areas))
    height, width = gray.shape
    short_side = float(min(height, width))
    strict_external = analyze_external_pollution(
        foreground,
        min_confidence=0.92,
    )

    def edge_distance(component_id: int) -> int:
        left = int(stats[component_id, cv2.CC_STAT_LEFT])
        top = int(stats[component_id, cv2.CC_STAT_TOP])
        right = left + int(stats[component_id, cv2.CC_STAT_WIDTH]) - 1
        bottom = top + int(stats[component_id, cv2.CC_STAT_HEIGHT]) - 1
        return min(left, top, width - 1 - right, height - 1 - bottom)

    central_ids = [
        component_id
        for component_id in component_ids
        if edge_distance(component_id) > short_side * 0.18
    ]
    main_id = max(
        central_ids or component_ids,
        key=lambda component_id: int(stats[component_id, cv2.CC_STAT_AREA]),
    )
    main_left = int(stats[main_id, cv2.CC_STAT_LEFT])
    main_top = int(stats[main_id, cv2.CC_STAT_TOP])
    main_right = main_left + int(stats[main_id, cv2.CC_STAT_WIDTH]) - 1
    main_bottom = main_top + int(stats[main_id, cv2.CC_STAT_HEIGHT]) - 1
    main_area = max(1, int(stats[main_id, cv2.CC_STAT_AREA]))
    minimum_block_area = max(
        12,
        int(round(gray.size * 0.0015)),
        int(round(main_area * 0.12)),
    )
    external_areas: list[int] = []
    for component_id in component_ids:
        if component_id == main_id:
            continue
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < minimum_block_area:
            continue
        left = int(stats[component_id, cv2.CC_STAT_LEFT])
        top = int(stats[component_id, cv2.CC_STAT_TOP])
        right = left + int(stats[component_id, cv2.CC_STAT_WIDTH]) - 1
        bottom = top + int(stats[component_id, cv2.CC_STAT_HEIGHT]) - 1
        gap_x = max(main_left - right - 1, left - main_right - 1, 0)
        gap_y = max(main_top - bottom - 1, top - main_bottom - 1, 0)
        gap = float(np.hypot(gap_x, gap_y))
        component_edge_distance = min(left, top, width - 1 - right, height - 1 - bottom)
        near_outer_area = component_edge_distance <= short_side * 0.18
        clearly_separated = gap >= max(3.0, short_side * 0.08)
        if near_outer_area and clearly_separated:
            external_areas.append(area)
    external_count = len(external_areas)
    external_area = sum(external_areas)
    strict_external_area = int(strict_external.pollution_mask.sum())
    if strict_external.applied and strict_external.confidence >= 0.92 and strict_external_area:
        external_count = max(external_count, strict_external.pollution_component_count)
        external_area = max(external_area, strict_external_area)
    return {
        "前景连通域数量": len(areas),
        "散点数量": len(speck_areas),
        "散点前景占比": sum(speck_areas) / foreground_total,
        "疑似大块外部污染数量": external_count,
        "大块外部污染前景占比": min(1.0, external_area / foreground_total),
    }
