"""自由变换共用的 RGBA 几何计算与渲染。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class TransformLimits:
    """限制变换产生的中间位图，避免异常参数耗尽内存。"""

    max_dimension: int = 16_384
    max_pixels: int = 64 * 1024 * 1024


@dataclass
class TransformGeometry:
    """缩放、透视和旋转共用的像素中心坐标几何。"""

    source_size: tuple[int, int]
    scaled_size: tuple[int, int]
    perspective_size: tuple[int, int]
    output_size: tuple[int, int]
    perspective_matrix: np.ndarray
    rotation_matrix: np.ndarray
    normalized_quad: np.ndarray
    rotated_quad: np.ndarray
    transformed_center: tuple[float, float]
    has_distortion: bool


@dataclass(frozen=True)
class TransformPlacement:
    """变换位图在逻辑田字格坐标系中的位置。"""

    origin: tuple[float, float]
    polygon: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]


@dataclass(frozen=True)
class CanvasGeometry:
    """最终位图尺寸以及逻辑田字格左上角。"""

    output_size: tuple[int, int]
    grid_origin: tuple[int, int]


@dataclass(frozen=True)
class CanvasRender:
    """合成后的 RGBA 像素及其逻辑田字格原点。"""

    pixels: np.ndarray
    geometry: CanvasGeometry


def calculate_transform_geometry(
    source_size: tuple[int, int],
    *,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    rotation: float = 0.0,
    distort: Sequence[float] = (0.0,) * 8,
    limits: TransformLimits = TransformLimits(),
) -> TransformGeometry:
    """按像素中心坐标计算缩放、四角透视和旋转几何。"""
    source_width, source_height = _normalized_size(source_size)
    values = (scale_x, scale_y, rotation, *distort)
    if len(distort) != 8 or not all(math.isfinite(float(value)) for value in values):
        raise ValueError("变换参数包含无效数值。")
    if scale_x <= 0.0 or scale_y <= 0.0:
        raise ValueError("缩放比例必须大于零。")

    scaled_width = max(1, round(source_width * float(scale_x)))
    scaled_height = max(1, round(source_height * float(scale_y)))
    validate_size((scaled_width, scaled_height), limits)

    source_quad = np.asarray(
        [
            [0.0, 0.0],
            [float(scaled_width - 1), 0.0],
            [float(scaled_width - 1), float(scaled_height - 1)],
            [0.0, float(scaled_height - 1)],
        ],
        dtype=np.float32,
    )
    distort_array = np.asarray(distort, dtype=np.float32).reshape(4, 2)
    target_quad = source_quad + distort_array
    has_distortion = bool(np.any(np.abs(distort_array) > 1e-9))
    if has_distortion:
        if scaled_width < 2 or scaled_height < 2 or not quad_is_valid(target_quad):
            raise ValueError("扭曲后的控制四边形无效。")
        min_x = min(0.0, float(target_quad[:, 0].min()))
        min_y = min(0.0, float(target_quad[:, 1].min()))
        normalized_quad = target_quad.copy()
        normalized_quad[:, 0] -= min_x
        normalized_quad[:, 1] -= min_y
        perspective_width = max(
            1,
            int(math.ceil(float(normalized_quad[:, 0].max()))) + 1,
        )
        perspective_height = max(
            1,
            int(math.ceil(float(normalized_quad[:, 1].max()))) + 1,
        )
        validate_size((perspective_width, perspective_height), limits)
        perspective_matrix = cv2.getPerspectiveTransform(source_quad, normalized_quad)
        if not np.isfinite(perspective_matrix).all():
            raise ValueError("透视矩阵包含无效数值。")
    else:
        normalized_quad = source_quad.copy()
        perspective_width = scaled_width
        perspective_height = scaled_height
        perspective_matrix = np.eye(3, dtype=np.float64)

    if has_distortion:
        rotation_center = (
            float(normalized_quad[:, 0].mean()),
            float(normalized_quad[:, 1].mean()),
        )
    else:
        rotation_center = (perspective_width / 2.0, perspective_height / 2.0)
    rotation_matrix = cv2.getRotationMatrix2D(rotation_center, -float(rotation), 1.0)
    perspective_corners = np.asarray(
        [
            [0.0, 0.0],
            [float(perspective_width - 1), 0.0],
            [float(perspective_width - 1), float(perspective_height - 1)],
            [0.0, float(perspective_height - 1)],
        ],
        dtype=np.float64,
    )
    rotated_bounds = map_points(perspective_corners, rotation_matrix)
    rotate_min_x = float(rotated_bounds[:, 0].min())
    rotate_min_y = float(rotated_bounds[:, 1].min())
    rotate_max_x = float(rotated_bounds[:, 0].max())
    rotate_max_y = float(rotated_bounds[:, 1].max())
    output_width = max(1, int(math.ceil(rotate_max_x - rotate_min_x)) + 1)
    output_height = max(1, int(math.ceil(rotate_max_y - rotate_min_y)) + 1)
    validate_size((output_width, output_height), limits)

    rotation_matrix = rotation_matrix.copy()
    rotation_matrix[0, 2] -= rotate_min_x
    rotation_matrix[1, 2] -= rotate_min_y
    rotated_quad = map_points(normalized_quad, rotation_matrix)
    transformed_center_values = map_points(
        np.asarray([rotation_center], dtype=np.float64),
        rotation_matrix,
    )[0]
    transformed_center = (
        float(transformed_center_values[0]),
        float(transformed_center_values[1]),
    )
    return TransformGeometry(
        source_size=(source_width, source_height),
        scaled_size=(scaled_width, scaled_height),
        perspective_size=(perspective_width, perspective_height),
        output_size=(output_width, output_height),
        perspective_matrix=perspective_matrix,
        rotation_matrix=rotation_matrix,
        normalized_quad=normalized_quad,
        rotated_quad=rotated_quad,
        transformed_center=transformed_center,
        has_distortion=has_distortion,
    )


def render_transformed_rgba(
    source: np.ndarray,
    geometry: TransformGeometry,
    *,
    force_rotation: bool = False,
) -> np.ndarray:
    """按已计算的共享几何渲染一张 RGBA 字形。"""
    pixels = _normalized_rgba(source)
    source_height, source_width = pixels.shape[:2]
    if (source_width, source_height) != geometry.source_size:
        raise ValueError("源图尺寸与变换几何不一致。")

    if geometry.scaled_size != geometry.source_size:
        pixels = cv2.resize(
            pixels,
            geometry.scaled_size,
            interpolation=(
                cv2.INTER_LANCZOS4
                if geometry.scaled_size[0] > source_width
                or geometry.scaled_size[1] > source_height
                else cv2.INTER_AREA
            ),
        )
    if geometry.has_distortion:
        pixels = cv2.warpPerspective(
            pixels,
            geometry.perspective_matrix,
            geometry.perspective_size,
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
    if force_rotation or geometry.output_size != geometry.perspective_size:
        pixels = cv2.warpAffine(
            pixels,
            geometry.rotation_matrix,
            geometry.output_size,
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
    return np.ascontiguousarray(pixels, dtype=np.uint8)


def place_transform(
    geometry: TransformGeometry,
    source_center: tuple[float, float],
    movement: tuple[float, float] = (0.0, 0.0),
) -> TransformPlacement:
    """把变换结果的几何中心锚定到原字形中心，再叠加用户移动量。"""
    origin_x = (
        float(source_center[0])
        - geometry.transformed_center[0]
        + float(movement[0])
    )
    origin_y = (
        float(source_center[1])
        - geometry.transformed_center[1]
        + float(movement[1])
    )
    polygon_values = geometry.rotated_quad + np.asarray(
        [origin_x, origin_y],
        dtype=np.float64,
    )
    polygon = tuple(
        (float(point[0]), float(point[1])) for point in polygon_values
    )
    return TransformPlacement(
        origin=(origin_x, origin_y),
        polygon=polygon,  # type: ignore[arg-type]
    )


def alpha_bounds(pixels: np.ndarray) -> tuple[int, int, int, int] | None:
    """返回非透明像素的左上闭、右下开包围盒。"""
    rgba = _normalized_rgba(pixels)
    rows, columns = np.nonzero(rgba[:, :, 3])
    if rows.size == 0 or columns.size == 0:
        return None
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    )


def calculate_canvas_geometry(
    pixels: np.ndarray,
    content_origin: tuple[float, float],
    canvas_size: tuple[int, int],
    *,
    expand_symmetric: bool,
) -> CanvasGeometry:
    """计算固定田字格或对称扩展成品的位图尺寸。"""
    canvas_width, canvas_height = _normalized_size(canvas_size)
    bounds = alpha_bounds(pixels)
    if not expand_symmetric or bounds is None:
        return CanvasGeometry((canvas_width, canvas_height), (0, 0))

    left = float(content_origin[0]) + bounds[0]
    top = float(content_origin[1]) + bounds[1]
    right = float(content_origin[0]) + bounds[2]
    bottom = float(content_origin[1]) + bounds[3]
    expand_x = max(0, math.ceil(-left), math.ceil(right - canvas_width))
    expand_y = max(0, math.ceil(-top), math.ceil(bottom - canvas_height))
    return CanvasGeometry(
        (canvas_width + expand_x * 2, canvas_height + expand_y * 2),
        (expand_x, expand_y),
    )


def compose_rgba_on_canvas(
    pixels: np.ndarray,
    content_origin: tuple[float, float],
    canvas_size: tuple[int, int],
    *,
    expand_symmetric: bool,
    limits: TransformLimits = TransformLimits(),
) -> CanvasRender:
    """把局部变换位图合成到田字格，并按需对称扩展。"""
    source = _normalized_rgba(pixels)
    canvas_geometry = calculate_canvas_geometry(
        source,
        content_origin,
        canvas_size,
        expand_symmetric=expand_symmetric,
    )
    validate_size(canvas_geometry.output_size, limits)
    output_width, output_height = canvas_geometry.output_size
    translation_x = float(content_origin[0]) + canvas_geometry.grid_origin[0]
    translation_y = float(content_origin[1]) + canvas_geometry.grid_origin[1]
    rounded_x = round(translation_x)
    rounded_y = round(translation_y)
    if (
        math.isclose(translation_x, rounded_x, abs_tol=1e-9)
        and math.isclose(translation_y, rounded_y, abs_tol=1e-9)
    ):
        output = np.zeros((output_height, output_width, 4), dtype=np.uint8)
        source_height, source_width = source.shape[:2]
        source_left = max(0, -rounded_x)
        source_top = max(0, -rounded_y)
        source_right = min(source_width, output_width - rounded_x)
        source_bottom = min(source_height, output_height - rounded_y)
        if source_right > source_left and source_bottom > source_top:
            destination_left = source_left + rounded_x
            destination_top = source_top + rounded_y
            output[
                destination_top : destination_top + source_bottom - source_top,
                destination_left : destination_left + source_right - source_left,
            ] = source[source_top:source_bottom, source_left:source_right]
    else:
        matrix = np.asarray(
            [[1.0, 0.0, translation_x], [0.0, 1.0, translation_y]],
            dtype=np.float64,
        )
        output = cv2.warpAffine(
            source,
            matrix,
            canvas_geometry.output_size,
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
    return CanvasRender(
        np.ascontiguousarray(output, dtype=np.uint8),
        canvas_geometry,
    )


def validate_size(size: tuple[int, int], limits: TransformLimits) -> None:
    """验证图像尺寸是否位于统一的安全范围。"""
    width, height = _normalized_size(size)
    if (
        width > limits.max_dimension
        or height > limits.max_dimension
        or width * height > limits.max_pixels
    ):
        raise ValueError("变换后的图像尺寸超出安全范围。")


def quad_is_valid(points: np.ndarray) -> bool:
    """拒绝非有限、自交、凹陷和近零面积的透视四边形。"""
    quad = np.asarray(points, dtype=np.float64).reshape(4, 2)
    if not np.isfinite(quad).all():
        return False
    cross_products: list[float] = []
    for index in range(4):
        first = quad[(index + 1) % 4] - quad[index]
        second = quad[(index + 2) % 4] - quad[(index + 1) % 4]
        cross_products.append(float(first[0] * second[1] - first[1] * second[0]))
    epsilon = 1e-4
    if any(abs(value) <= epsilon for value in cross_products):
        return False
    return all(value > 0.0 for value in cross_products) or all(
        value < 0.0 for value in cross_products
    )


def map_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """使用仿射或透视矩阵映射一组二维点。"""
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape == (2, 3):
        homogeneous = np.column_stack((values, np.ones(len(values), dtype=np.float64)))
        return homogeneous @ transform.T
    if transform.shape == (3, 3):
        homogeneous = np.column_stack((values, np.ones(len(values), dtype=np.float64)))
        mapped = homogeneous @ transform.T
        denominator = mapped[:, 2:3]
        if np.any(np.abs(denominator) <= 1e-12):
            raise ValueError("透视映射分母无效。")
        return mapped[:, :2] / denominator
    raise ValueError("不支持的变换矩阵尺寸。")


def _normalized_rgba(pixels: np.ndarray) -> np.ndarray:
    values = np.asarray(pixels, dtype=np.uint8)
    if values.ndim != 3 or values.shape[2] != 4:
        raise ValueError("RGBA 像素数组尺寸无效。")
    return np.ascontiguousarray(values)


def _normalized_size(size: tuple[int, int]) -> tuple[int, int]:
    try:
        width = int(size[0])
        height = int(size[1])
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError("图像尺寸无效。") from exc
    if width <= 0 or height <= 0:
        raise ValueError("图像尺寸必须大于零。")
    return width, height
