"""前景空间结构分析，供清理算法和评分参考共同复用。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ExternalPollutionAnalysis:
    """方向无关的外围污染分析结果。

    所有掩码均为 ``uint8`` 二值数组（0/1）。只有 ``applied`` 为 True 时，
    ``cleaned_mask`` 才会删除像素；低置信分析始终原样回退。
    """

    cleaned_mask: np.ndarray
    pollution_mask: np.ndarray
    body_mask: np.ndarray
    confidence: float
    applied: bool
    reason: str
    stroke_width: float
    body_component_count: int
    pollution_component_count: int
    removed_ratio: float
    metrics: dict[str, float]


def _empty_result(source: np.ndarray, reason: str) -> ExternalPollutionAnalysis:
    empty = np.zeros_like(source, dtype=np.uint8)
    return ExternalPollutionAnalysis(
        cleaned_mask=source.copy(),
        pollution_mask=empty,
        body_mask=source.copy(),
        confidence=0.0,
        applied=False,
        reason=reason,
        stroke_width=0.0,
        body_component_count=0,
        pollution_component_count=0,
        removed_ratio=0.0,
        metrics={},
    )


def _estimate_stroke_width(
    source: np.ndarray,
    labels: np.ndarray,
    stats: np.ndarray,
) -> float:
    """从累计占主要墨量的连通域估算笔画宽度，避免散点拉低尺度。"""
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64, copy=False)
    if not areas.size:
        return 1.0
    order = np.argsort(areas)[::-1]
    target = max(1, int(round(float(areas.sum()) * 0.70)))
    selected: list[int] = []
    accumulated = 0
    for offset in order:
        label = int(offset) + 1
        selected.append(label)
        accumulated += int(areas[offset])
        if accumulated >= target:
            break

    selected_mask = np.isin(labels, selected)
    distance = cv2.distanceTransform(source, cv2.DIST_L2, 5)
    local_maximum = distance >= cv2.dilate(distance, np.ones((3, 3), np.uint8)) - 0.05
    ridge_values = distance[selected_mask & local_maximum & (distance >= 0.9)]
    if ridge_values.size >= 3:
        estimate = float(np.median(ridge_values) * 2.0)
    else:
        foreground_values = distance[selected_mask]
        estimate = float(np.percentile(foreground_values, 80.0) * 2.0) if foreground_values.size else 1.0
    short_side = float(min(source.shape))
    return float(np.clip(estimate, 1.0, max(2.0, short_side * 0.12)))


def _bbox_distance_matrix(component_ids: list[int], stats: np.ndarray) -> np.ndarray:
    if not component_ids:
        return np.empty((0, 0), dtype=np.float32)
    indexes = np.asarray(component_ids, dtype=np.int32)
    left = stats[indexes, cv2.CC_STAT_LEFT].astype(np.float32)
    top = stats[indexes, cv2.CC_STAT_TOP].astype(np.float32)
    right = left + stats[indexes, cv2.CC_STAT_WIDTH].astype(np.float32) - 1.0
    bottom = top + stats[indexes, cv2.CC_STAT_HEIGHT].astype(np.float32) - 1.0
    dx = np.maximum(
        np.maximum(left[:, None] - right[None, :] - 1.0, left[None, :] - right[:, None] - 1.0),
        0.0,
    )
    dy = np.maximum(
        np.maximum(top[:, None] - bottom[None, :] - 1.0, top[None, :] - bottom[:, None] - 1.0),
        0.0,
    )
    return np.hypot(dx, dy).astype(np.float32, copy=False)


def _minimum_group_distance(
    first: list[int] | set[int],
    second: list[int] | set[int],
    stats: np.ndarray,
) -> float:
    left_ids = list(first)
    right_ids = list(second)
    if not left_ids or not right_ids:
        return float("inf")
    combined = left_ids + right_ids
    distances = _bbox_distance_matrix(combined, stats)
    return float(distances[: len(left_ids), len(left_ids) :].min())


def _group_components(
    component_ids: list[int],
    stats: np.ndarray,
    link_distance: float,
) -> list[list[int]]:
    """按包围盒间距建立组件图，并返回其连通分组。"""
    if not component_ids:
        return []
    distances = _bbox_distance_matrix(component_ids, stats)
    adjacency = distances <= float(link_distance)
    unseen = np.ones(len(component_ids), dtype=bool)
    groups: list[list[int]] = []
    for start in range(len(component_ids)):
        if not unseen[start]:
            continue
        unseen[start] = False
        stack = [start]
        indexes: list[int] = []
        while stack:
            current = stack.pop()
            indexes.append(current)
            neighbors = np.flatnonzero(adjacency[current] & unseen)
            if neighbors.size:
                unseen[neighbors] = False
                stack.extend(int(item) for item in neighbors)
        groups.append([component_ids[index] for index in indexes])
    return groups


def _component_area(component_ids: list[int] | set[int], stats: np.ndarray) -> int:
    indexes = np.asarray(list(component_ids), dtype=np.int32)
    if not indexes.size:
        return 0
    return int(stats[indexes, cv2.CC_STAT_AREA].sum())


def _mask_for_components(labels: np.ndarray, component_ids: list[int] | set[int]) -> np.ndarray:
    if not component_ids:
        return np.zeros_like(labels, dtype=np.uint8)
    return np.isin(labels, list(component_ids)).astype(np.uint8)


def _ray_distance_to_frame(
    center: np.ndarray,
    direction: np.ndarray,
    width: int,
    height: int,
) -> float:
    distances: list[float] = []
    if direction[0] > 1e-6:
        distances.append((width - 1.0 - center[0]) / direction[0])
    elif direction[0] < -1e-6:
        distances.append(-center[0] / direction[0])
    if direction[1] > 1e-6:
        distances.append((height - 1.0 - center[1]) / direction[1])
    elif direction[1] < -1e-6:
        distances.append(-center[1] / direction[1])
    positive = [value for value in distances if value > 0.0]
    return min(positive) if positive else 1.0


def _cluster_linearity(component_ids: list[int], centroids: np.ndarray) -> float:
    if len(component_ids) < 3:
        return 0.55 if len(component_ids) == 2 else 0.0
    points = centroids[np.asarray(component_ids, dtype=np.int32)].astype(np.float64)
    covariance = np.cov(points, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    largest = float(max(eigenvalues[-1], 1e-6))
    return float(np.clip(1.0 - float(eigenvalues[0]) / largest, 0.0, 1.0))


def _cluster_geometry(
    cluster_ids: list[int],
    body_mask: np.ndarray,
    labels: np.ndarray,
    stats: np.ndarray,
    centroids: np.ndarray,
    stroke_width: float,
    body_area: int,
    edge_margin_ratio: float,
) -> dict[str, float]:
    cluster_mask = _mask_for_components(labels, cluster_ids)
    body_y, body_x = np.where(body_mask > 0)
    cluster_y, cluster_x = np.where(cluster_mask > 0)
    body_center = np.array([float(body_x.mean()), float(body_y.mean())], dtype=np.float64)
    cluster_center = np.array([float(cluster_x.mean()), float(cluster_y.mean())], dtype=np.float64)
    vector = cluster_center - body_center
    center_distance = float(np.linalg.norm(vector))
    direction = vector / max(center_distance, 1e-6)

    body_projection = body_x.astype(np.float64) * direction[0] + body_y.astype(np.float64) * direction[1]
    cluster_projection = cluster_x.astype(np.float64) * direction[0] + cluster_y.astype(np.float64) * direction[1]
    directional_gap = max(0.0, float(cluster_projection.min() - body_projection.max()))

    distance_to_body = cv2.distanceTransform(1 - body_mask, cv2.DIST_L2, 3)
    pixel_gap = max(0.0, float(distance_to_body[cluster_mask > 0].min()) - 1.0)
    gap_ratio = pixel_gap / max(stroke_width, 1.0)
    directional_gap_ratio = directional_gap / max(stroke_width, 1.0)

    height, width = labels.shape
    edge_distance = float(
        min(
            int(cluster_x.min()),
            int(cluster_y.min()),
            width - 1 - int(cluster_x.max()),
            height - 1 - int(cluster_y.max()),
        )
    )
    short_side = float(min(height, width))
    edge_span = max(1.0, short_side * max(0.01, edge_margin_ratio))
    edge_score = float(np.clip((edge_span - edge_distance) / edge_span, 0.0, 1.0))
    ray_distance = _ray_distance_to_frame(body_center, direction, width, height)
    radial_position = float(np.clip(center_distance / max(ray_distance, 1.0), 0.0, 1.5))

    component_areas = stats[np.asarray(cluster_ids, dtype=np.int32), cv2.CC_STAT_AREA].astype(np.float64)
    indexes = np.asarray(cluster_ids, dtype=np.int32)
    widths = stats[indexes, cv2.CC_STAT_WIDTH].astype(np.float64)
    heights = stats[indexes, cv2.CC_STAT_HEIGHT].astype(np.float64)
    cluster_width = float(cluster_x.max() - cluster_x.min() + 1)
    cluster_height = float(cluster_y.max() - cluster_y.min() + 1)
    cluster_aspect = max(cluster_width, cluster_height) / max(1.0, min(cluster_width, cluster_height))

    perpendicular = np.array([-direction[1], direction[0]], dtype=np.float64)
    body_perpendicular = body_x.astype(np.float64) * perpendicular[0] + body_y.astype(np.float64) * perpendicular[1]
    cluster_perpendicular = cluster_x.astype(np.float64) * perpendicular[0] + cluster_y.astype(np.float64) * perpendicular[1]
    perpendicular_overhang = max(0.0, float(body_perpendicular.min() - cluster_perpendicular.min()))
    perpendicular_overhang += max(0.0, float(cluster_perpendicular.max() - body_perpendicular.max()))

    area = int(component_areas.sum())
    return {
        "area": float(area),
        "area_ratio": float(area / max(body_area, 1)),
        "gap_ratio": gap_ratio,
        "directional_gap_ratio": directional_gap_ratio,
        "edge_score": edge_score,
        "radial_position": radial_position,
        "component_count": float(len(cluster_ids)),
        "median_component_area": float(np.median(component_areas)),
        "component_area_cv": float(np.std(component_areas) / max(float(np.mean(component_areas)), 1.0)),
        "median_minor_width_ratio": float(np.median(np.minimum(widths, heights)) / max(stroke_width, 1.0)),
        "cluster_aspect": float(cluster_aspect),
        "perpendicular_overhang_ratio": float(perpendicular_overhang / max(stroke_width, 1.0)),
        "linearity": _cluster_linearity(cluster_ids, centroids),
    }


def analyze_external_pollution(
    mask: np.ndarray,
    min_confidence: float = 0.78,
    max_area_ratio: float = 0.20,
    gap_stroke_ratio: float = 1.25,
    edge_margin_ratio: float = 0.18,
) -> ExternalPollutionAnalysis:
    """识别与文字主体存在明确空白分隔的外围污染簇。

    算法不读取字符、不假定污染方向。所有距离按估计笔画宽度归一；只有同时满足
    主体占优、外围位置、空白分隔、较小面积和多碎片聚集时才执行删除。
    """
    source_array = np.asarray(mask)
    if source_array.ndim != 2:
        raise ValueError("外围污染分析仅支持二维掩码")
    source = (source_array > 0).astype(np.uint8)
    foreground_total = int(source.sum())
    if foreground_total == 0:
        return _empty_result(source, "前景为空")

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        source,
        connectivity=8,
        ltype=cv2.CV_32S,
    )
    component_count = count - 1
    if component_count < 3:
        return _empty_result(source, "连通域不足，保留原掩码")

    stroke_width = _estimate_stroke_width(source, labels, stats)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64, copy=False)
    largest_area = int(areas.max())
    active_min_area = max(
        4,
        int(round(stroke_width * stroke_width * 0.16)),
        int(round(largest_area * 0.012)),
    )
    active_ids = [
        label
        for label in range(1, count)
        if int(stats[label, cv2.CC_STAT_AREA]) >= active_min_area
    ]
    if len(active_ids) < 3:
        result = _empty_result(source, "有效连通域不足，保留原掩码")
        return ExternalPollutionAnalysis(
            **{**result.__dict__, "stroke_width": stroke_width}
        )

    short_side = float(min(source.shape))
    # 主体聚合半径必须小于污染判定的空白门槛，否则外围簇会先被并入主体。
    body_link_distance = max(stroke_width * 1.15, short_side * 0.012)
    active_groups = _group_components(active_ids, stats, body_link_distance)
    active_groups.sort(key=lambda group: _component_area(group, stats), reverse=True)
    body_ids = set(active_groups[0])
    body_area = _component_area(body_ids, stats)

    # 面积接近主体的分离结构一律纳入保护，避免删除偏旁、点画或独立长笔。
    changed = True
    while changed:
        changed = False
        for group in active_groups[1:]:
            if body_ids.intersection(group):
                continue
            group_area = _component_area(group, stats)
            nearby = _minimum_group_distance(body_ids, group, stats) <= body_link_distance * 1.02
            structurally_large = group_area > body_area * max_area_ratio
            if nearby or structurally_large:
                body_ids.update(group)
                body_area = _component_area(body_ids, stats)
                changed = True

    body_mask = _mask_for_components(labels, body_ids)
    remaining_active = [label for label in active_ids if label not in body_ids]
    if not remaining_active:
        return ExternalPollutionAnalysis(
            cleaned_mask=source.copy(),
            pollution_mask=np.zeros_like(source),
            body_mask=body_mask,
            confidence=0.0,
            applied=False,
            reason="全部有效连通域均属于主体保护范围",
            stroke_width=stroke_width,
            body_component_count=len(body_ids),
            pollution_component_count=0,
            removed_ratio=0.0,
            metrics={"有效面积阈值": float(active_min_area)},
        )

    cluster_link_distance = max(stroke_width * 2.8, short_side * 0.025)
    seed_groups = _group_components(remaining_active, stats, cluster_link_distance)
    inactive_ids = [label for label in range(1, count) if label not in active_ids]
    candidate_records: list[tuple[list[int], float, dict[str, float], str]] = []
    body_active_areas = stats[np.asarray(list(body_ids)), cv2.CC_STAT_AREA].astype(np.float64)
    body_median_area = float(np.median(body_active_areas))

    for seed_group in seed_groups:
        cluster_ids = list(seed_group)
        # 吸收污染簇附近的小碎片，但不允许小碎片把两个远处区域串联起来。
        for label in inactive_ids:
            near_cluster = _minimum_group_distance(seed_group, [label], stats) <= cluster_link_distance
            clear_of_body = (
                _minimum_group_distance(body_ids, [label], stats)
                >= stroke_width * gap_stroke_ratio
            )
            if near_cluster and clear_of_body:
                cluster_ids.append(label)
        geometry = _cluster_geometry(
            cluster_ids,
            body_mask,
            labels,
            stats,
            centroids,
            stroke_width,
            body_area,
            edge_margin_ratio,
        )
        significant_count = len(seed_group)
        area_ratio = geometry["area_ratio"]
        gap_ratio = geometry["gap_ratio"]
        directional_gap_ratio = geometry["directional_gap_ratio"]
        radial_position = geometry["radial_position"]
        single_thin_block = (
            significant_count == 1
            and len(cluster_ids) == 1
            and geometry["cluster_aspect"] >= 4.5
            and geometry["median_minor_width_ratio"] <= 1.8
            and gap_ratio >= max(gap_stroke_ratio, 1.75)
            and radial_position >= 0.52
        )
        high_confidence_single_block = (
            single_thin_block
            and geometry["cluster_aspect"] >= 6.0
            and geometry["median_minor_width_ratio"] <= 1.0
            and directional_gap_ratio >= 1.0
            and area_ratio <= 0.15
            and geometry["edge_score"] >= 0.25
            and radial_position >= 0.65
        )
        repeated_stroke_pattern = (
            significant_count >= 3
            and geometry["perpendicular_overhang_ratio"] <= 0.75
            and geometry["component_area_cv"] <= 0.45
            and 0.55 <= geometry["median_minor_width_ratio"] <= 1.8
        )
        wide_irregular_band = (
            significant_count >= 3
            and len(cluster_ids) >= 6
            and geometry["component_area_cv"] >= 0.55
            and geometry["perpendicular_overhang_ratio"] >= 1.0
            and geometry["edge_score"] >= 0.50
            and radial_position >= 0.62
        )
        dense_fragment_line = (
            significant_count >= 2
            and len(cluster_ids) >= 8
            and geometry["component_area_cv"] >= 0.75
            and geometry["median_minor_width_ratio"] <= 0.45
            and geometry["cluster_aspect"] >= 4.0
            and geometry["linearity"] >= 0.85
            and gap_ratio >= gap_stroke_ratio
            and geometry["edge_score"] >= 0.50
            and radial_position >= 0.70
        )
        irregular_fragment_band = wide_irregular_band or dense_fragment_line

        hard_reject = ""
        if area_ratio > max_area_ratio:
            hard_reject = "面积接近合法结构"
        elif gap_ratio < gap_stroke_ratio:
            hard_reject = "与主体距离不足"
        elif directional_gap_ratio < 0.35 and not irregular_fragment_band:
            hard_reject = "未完整位于主体外围"
        elif radial_position < 0.40:
            hard_reject = "外围位置不明确"
        elif repeated_stroke_pattern:
            hard_reject = "规则分离笔画位于主体投影内"
        elif significant_count < 2 and len(cluster_ids) < 3 and not single_thin_block:
            hard_reject = "单个分离部件需保护"

        separation_score = float(
            np.clip((gap_ratio - gap_stroke_ratio) / 1.5, 0.0, 1.0)
        )
        outside_score = float(
            np.clip((directional_gap_ratio - 0.35) / 1.8, 0.0, 1.0)
        )
        area_score = float(np.clip(1.0 - area_ratio / max(max_area_ratio, 1e-6), 0.0, 1.0))
        fragment_score = float(np.clip((len(cluster_ids) - 1) / 3.0, 0.0, 1.0))
        size_ratio = geometry["median_component_area"] / max(body_median_area, 1.0)
        size_mismatch_score = float(np.clip((0.55 - size_ratio) / 0.50, 0.0, 1.0))
        confidence = float(
            np.clip(
                0.46
                + separation_score * 0.18
                + outside_score * 0.13
                + geometry["edge_score"] * 0.12
                + area_score * 0.08
                + fragment_score * 0.12
                + size_mismatch_score * 0.10
                + geometry["linearity"] * 0.06,
                0.0,
                1.0,
            )
        )
        if single_thin_block and not hard_reject:
            confidence = max(
                confidence,
                float(
                    np.clip(
                        0.76
                        + separation_score * 0.08
                        + outside_score * 0.07
                        + geometry["edge_score"] * 0.05
                        + min(1.0, (geometry["cluster_aspect"] - 4.5) / 4.0) * 0.04,
                        0.0,
                        1.0,
                    )
                ),
            )
        if high_confidence_single_block and not hard_reject:
            confidence = max(confidence, 0.94)
        if irregular_fragment_band and not hard_reject:
            confidence = max(confidence, 0.94)
        if hard_reject:
            confidence = min(confidence, max(0.0, min_confidence - 0.01))
        candidate_records.append((cluster_ids, confidence, geometry, hard_reject))

    accepted = [
        record
        for record in candidate_records
        if not record[3] and record[1] >= min_confidence
    ]
    accepted.sort(key=lambda record: record[1], reverse=True)
    removal_limit = max(1, int(round(body_area * max_area_ratio)))
    selected: list[tuple[list[int], float, dict[str, float], str]] = []
    selected_area = 0
    for record in accepted:
        area = int(record[2]["area"])
        if selected_area + area > removal_limit:
            continue
        selected.append(record)
        selected_area += area

    best_confidence = max((record[1] for record in candidate_records), default=0.0)
    common_metrics = {
        "参考笔画宽度": stroke_width,
        "有效面积阈值": float(active_min_area),
        "前景连通域数": float(component_count),
        "有效连通域数": float(len(active_ids)),
        "主体连通域数": float(len(body_ids)),
        "候选污染簇数": float(len(candidate_records)),
        "最大候选置信度": best_confidence,
    }
    diagnostic_record = max(
        candidate_records,
        key=lambda record: (record[1], record[2]["area"]),
        default=None,
    )
    if diagnostic_record is not None:
        diagnostic_geometry = diagnostic_record[2]
        common_metrics.update(
            {
                "诊断候选面积比": diagnostic_geometry["area_ratio"],
                "诊断候选间隔笔宽": diagnostic_geometry["gap_ratio"],
                "诊断候选方向间隔笔宽": diagnostic_geometry["directional_gap_ratio"],
                "诊断候选径向位置": diagnostic_geometry["radial_position"],
                "诊断候选长宽比": diagnostic_geometry["cluster_aspect"],
                "诊断候选短边笔宽比": diagnostic_geometry["median_minor_width_ratio"],
                "诊断候选面积变异": diagnostic_geometry["component_area_cv"],
                "诊断候选横向越界笔宽": diagnostic_geometry["perpendicular_overhang_ratio"],
            }
        )
    if not selected:
        reject_reason = "未发现达到高置信门槛的外围污染簇"
        if diagnostic_record is not None and diagnostic_record[3]:
            reject_reason += f"：{diagnostic_record[3]}"
        return ExternalPollutionAnalysis(
            cleaned_mask=source.copy(),
            pollution_mask=np.zeros_like(source),
            body_mask=body_mask,
            confidence=best_confidence,
            applied=False,
            reason=reject_reason,
            stroke_width=stroke_width,
            body_component_count=len(body_ids),
            pollution_component_count=0,
            removed_ratio=0.0,
            metrics=common_metrics,
        )

    removed_ids: set[int] = set()
    for cluster_ids, _, _, _ in selected:
        removed_ids.update(cluster_ids)
    pollution_mask = _mask_for_components(labels, removed_ids)
    cleaned_mask = source.copy()
    cleaned_mask[pollution_mask > 0] = 0
    removed_pixels = int(pollution_mask.sum())
    removed_ratio = removed_pixels / max(foreground_total, 1)
    confidence = min(record[1] for record in selected)
    common_metrics.update(
        {
            "命中污染簇数": float(len(selected)),
            "移除前景像素": float(removed_pixels),
            "移除面积比例": removed_ratio,
        }
    )
    return ExternalPollutionAnalysis(
        cleaned_mask=cleaned_mask,
        pollution_mask=pollution_mask,
        body_mask=body_mask,
        confidence=confidence,
        applied=True,
        reason=(
            f"高置信外围污染：{len(selected)}簇/{len(removed_ids)}个连通域，"
            f"移除{removed_ratio:.1%}前景"
        ),
        stroke_width=stroke_width,
        body_component_count=len(body_ids),
        pollution_component_count=len(removed_ids),
        removed_ratio=removed_ratio,
        metrics=common_metrics,
    )


def remove_external_pollution(
    mask: np.ndarray,
    min_confidence: float = 0.78,
    max_area_ratio: float = 0.20,
    gap_stroke_ratio: float = 1.25,
    edge_margin_ratio: float = 0.18,
) -> np.ndarray:
    """返回方向无关的高置信外围污染清理结果（0/1）。"""
    return analyze_external_pollution(
        mask,
        min_confidence=min_confidence,
        max_area_ratio=max_area_ratio,
        gap_stroke_ratio=gap_stroke_ratio,
        edge_margin_ratio=edge_margin_ratio,
    ).cleaned_mask
