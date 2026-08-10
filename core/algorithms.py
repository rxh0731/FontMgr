# algorithms.py — 16+ 种图像处理算法，纯计算无 UI 依赖

import cv2
import numpy as np
from typing import Optional, Tuple


# ============================================================
# L1 降噪层
# ============================================================

def _odd_kernel(value: int, minimum: int = 3) -> int:
    """把核大小限制为不小于 minimum 的奇数。"""
    kernel = max(minimum, int(value))
    return kernel if kernel % 2 == 1 else kernel + 1


def gaussian_blur(arr: np.ndarray, kernel: int = 5) -> np.ndarray:
    """高斯滤波降噪。"""
    arr_u8 = np.clip(arr, 0, 255).astype(np.uint8)
    size = _odd_kernel(kernel)
    return cv2.GaussianBlur(arr_u8, (size, size), 0).astype(np.float32)


def median_blur(arr: np.ndarray, kernel: int = 3) -> np.ndarray:
    """中值滤波降噪。"""
    arr_u8 = np.clip(arr, 0, 255).astype(np.uint8)
    return cv2.medianBlur(arr_u8, _odd_kernel(kernel)).astype(np.float32)


def bilateral_filter(arr: np.ndarray, d: int = 9, sigma_color: float = 75.0, sigma_space: float = 75.0) -> np.ndarray:
    """双边滤波降噪（边缘保持）。"""
    arr_u8 = np.clip(arr, 0, 255).astype(np.uint8)
    return cv2.bilateralFilter(arr_u8, d, sigma_color, sigma_space).astype(np.float32)


def nlm_denoise(arr: np.ndarray, h: float = 10.0, template_window: int = 7, search_window: int = 21) -> np.ndarray:
    """非局部均值降噪。"""
    arr_u8 = np.clip(arr, 0, 255).astype(np.uint8)
    return cv2.fastNlMeansDenoising(arr_u8, None, h, template_window, search_window).astype(np.float32)


# ============================================================
# L2 背景分离层
# ============================================================

def bg_subtract(
    arr: np.ndarray,
    kernel: int = 31,
    threshold: int = 30,
    amplify: float = 2.0,
    normalize: bool = True,
) -> np.ndarray:
    """背景差分：大核模糊估计背景→差分→放大→扣除。

    返回 0~255 float32 灰度，背景已被压暗。
    """
    arr_u8 = np.clip(arr, 0, 255).astype(np.uint8)
    bg = cv2.GaussianBlur(arr_u8, (kernel, kernel), 0).astype(np.float32)
    diff = (arr.astype(np.float32) - bg) * amplify + 128.0
    if normalize:
        mn, mx = diff.min(), diff.max()
        if mx > mn:
            diff = (diff - mn) / (mx - mn) * 255.0
    return np.clip(diff, 0, 255)


def bg_morph_normalize(arr: np.ndarray, kernel: int = 51) -> np.ndarray:
    """形态学背景归一：大核开运算估计背景→原图除以背景。"""
    arr_u8 = np.clip(arr, 0, 255).astype(np.uint8)
    size = _odd_kernel(kernel)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    bg = cv2.morphologyEx(arr_u8, cv2.MORPH_OPEN, element).astype(np.float32)
    bg = np.maximum(bg, 1.0)
    result = arr.astype(np.float32) / bg * 128.0
    return np.clip(result, 0, 255)


def blackhat_background_enhance(arr: np.ndarray, kernel: int = 15, strength: float = 1.0) -> np.ndarray:
    """灰度黑帽背景增强，在二值化前增强小尺度深色笔画与背景的差异。"""
    arr_u8 = np.clip(arr, 0, 255).astype(np.uint8)
    size = _odd_kernel(kernel)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    closed = cv2.morphologyEx(arr_u8, cv2.MORPH_CLOSE, element)
    blackhat = cv2.subtract(closed, arr_u8).astype(np.float32)
    result = arr_u8.astype(np.float32) - blackhat * max(0.0, float(strength))
    return np.clip(result, 0, 255).astype(np.float32)


# ============================================================
# L3 二值化层
# ============================================================

def otsu_binarize(arr: np.ndarray, offset: int = 0) -> np.ndarray:
    """Otsu 大津二值化。

    返回：二值掩码 uint8（0=背景, 255=文字）。
    """
    arr_u8 = np.clip(arr, 0, 255).astype(np.uint8)
    thresh, _ = cv2.threshold(arr_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = max(1, min(254, thresh + offset))
    _, mask = cv2.threshold(arr_u8, thresh, 255, cv2.THRESH_BINARY_INV)
    return mask


def fixed_threshold_binarize(arr: np.ndarray, threshold: int = 160) -> np.ndarray:
    """按指定灰度阈值二值化，返回 0=背景、255=文字的掩码。"""
    arr_u8 = np.clip(arr, 0, 255).astype(np.uint8)
    value = max(1, min(254, int(threshold)))
    return (arr_u8 < value).astype(np.uint8) * 255


def sauvola_binarize(arr: np.ndarray, window: int = 25, k: float = 0.2, R: int = 128) -> np.ndarray:
    """Sauvola 局部自适应二值化：T = mean * (1 + k * (std/R - 1))。

    返回：二值掩码 uint8（0=背景, 255=文字）。
    """
    arr_u8 = np.clip(arr, 0, 255).astype(np.uint8)
    win = max(3, window if window % 2 == 1 else window + 1)
    from cv2 import meanStdDev as msd
    mean_arr = cv2.blur(arr_u8.astype(np.float32), (win, win))
    # 高效计算局部标准差：std = sqrt(E(X²) - E(X)²)
    sqr = cv2.blur((arr_u8.astype(np.float32)) ** 2, (win, win))
    var = sqr - mean_arr ** 2
    var[var < 0] = 0
    std_arr = np.sqrt(var)
    R_f = float(R)
    k_f = float(k)
    thresh = mean_arr * (1.0 + k_f * (std_arr / R_f - 1.0))
    thresh = np.clip(thresh, 0, 255).astype(np.uint8)
    mask = np.zeros_like(arr_u8, dtype=np.uint8)
    mask[arr_u8 < thresh] = 255
    return mask


def niblack_binarize(arr: np.ndarray, window: int = 25, k: float = -0.2) -> np.ndarray:
    """Niblack 局部二值化：T = mean + k * std。

    k 取负值使得阈值低于均值，对细笔画/枯笔保留更好。
    """
    arr_u8 = np.clip(arr, 0, 255).astype(np.uint8)
    win = max(3, window if window % 2 == 1 else window + 1)
    mean_arr = cv2.blur(arr_u8.astype(np.float32), (win, win))
    sqr = cv2.blur((arr_u8.astype(np.float32)) ** 2, (win, win))
    var = sqr - mean_arr ** 2
    var[var < 0] = 0
    std_arr = np.sqrt(var)
    thresh = mean_arr + k * std_arr
    thresh = np.clip(thresh, 0, 255).astype(np.uint8)
    mask = np.zeros_like(arr_u8, dtype=np.uint8)
    mask[arr_u8 < thresh] = 255
    return mask


def phansalkar_binarize(
    arr: np.ndarray,
    window: int = 25,
    k: float = 0.25,
    R: int = 128,
    p: float = 2.0,
    q: float = 10.0,
) -> np.ndarray:
    """Phansalkar 局部二值化：在低光区添加指数补偿。

    T = mean * (1 + p * exp(-q * mean) + k * (std/R - 1))
    适合极暗背景区域的文字。
    """
    arr_u8 = np.clip(arr, 0, 255).astype(np.uint8)
    win = max(3, window if window % 2 == 1 else window + 1)
    mean_arr = cv2.blur(arr_u8.astype(np.float32), (win, win))
    sqr = cv2.blur((arr_u8.astype(np.float32)) ** 2, (win, win))
    var = sqr - mean_arr ** 2
    var[var < 0] = 0
    std_arr = np.sqrt(var)
    R_f = float(R)
    mean_norm = mean_arr / 255.0
    compensation = p * np.exp(-q * mean_norm)
    thresh = mean_arr * (1.0 + compensation + float(k) * (std_arr / R_f - 1.0))
    thresh = np.clip(thresh, 0, 255).astype(np.uint8)
    mask = np.zeros_like(arr_u8, dtype=np.uint8)
    mask[arr_u8 < thresh] = 255
    return mask


def percentile_binarize(arr: np.ndarray, dark_ratio: float = 0.2) -> np.ndarray:
    """百分位硬切二值化：按灰度直方图的暗色比例切分。

    返回：二值掩码 uint8（0=背景, 255=文字）。
    """
    arr_u8 = np.clip(arr, 0, 255).astype(np.uint8).ravel()
    nonzero = arr_u8[arr_u8 < 255]
    if len(nonzero) == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    thresh = np.percentile(nonzero, max(0.1, dark_ratio * 100.0))
    thresh = max(1, min(254, int(thresh)))
    return (arr < thresh).astype(np.uint8) * 255


def wolf_binarize(arr: np.ndarray, window: int = 31, k: float = 0.35) -> np.ndarray:
    """Wolf-Jolion 局部二值化，适合背景不均和字迹深浅变化。"""
    arr_u8 = np.clip(arr, 0, 255).astype(np.float32)
    win = _odd_kernel(window)
    mean_arr = cv2.blur(arr_u8, (win, win))
    sqr = cv2.blur(arr_u8 ** 2, (win, win))
    std_arr = np.sqrt(np.maximum(sqr - mean_arr ** 2, 0.0))
    max_std = max(float(std_arr.max()), 1e-6)
    min_gray = float(arr_u8.min())
    threshold = mean_arr + float(k) * (std_arr / max_std - 1.0) * (mean_arr - min_gray)
    return (arr_u8 < threshold).astype(np.uint8) * 255


# ============================================================
# L4 形态清理层
# ============================================================

def morph_open(mask: np.ndarray, radius: int = 2, iterations: int = 1, shape: int = 1) -> np.ndarray:
    """开运算（先腐蚀再膨胀），清除小噪点。

    shape: 0=矩形, 1=椭圆, 2=十字
    """
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    shapes = [cv2.MORPH_RECT, cv2.MORPH_ELLIPSE, cv2.MORPH_CROSS]
    s = shapes[max(0, min(2, shape))]
    kernel = cv2.getStructuringElement(s, (radius * 2 + 1, radius * 2 + 1))
    return cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=iterations)


def morph_close(mask: np.ndarray, radius: int = 2, iterations: int = 1, shape: int = 1) -> np.ndarray:
    """闭运算（先膨胀再腐蚀），闭合小孔洞。"""
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    shapes = [cv2.MORPH_RECT, cv2.MORPH_ELLIPSE, cv2.MORPH_CROSS]
    s = shapes[max(0, min(2, shape))]
    kernel = cv2.getStructuringElement(s, (radius * 2 + 1, radius * 2 + 1))
    return cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def hole_fill(mask: np.ndarray, max_area: int = 80, max_ratio: float = 0.003) -> np.ndarray:
    """仅填充小孔洞，避免破坏“日、目、田”等字的固有字腔。"""
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    background = (mask_u8 == 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(background, 8, cv2.CV_32S)
    h, w = mask_u8.shape
    limit = max(1, max(int(max_area), int(mask_u8.sum() / 255 * max(0.0, float(max_ratio)))))
    result = mask_u8.copy()
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        touches_edge = x == 0 or y == 0 or x + width >= w or y + height >= h
        if not touches_edge and int(area) <= limit:
            result[labels == label] = 255
    return result


def morphological_reconstruct(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """受原掩码约束恢复腐蚀后的可信核心，去点同时尽量保持笔画轮廓。"""
    source = (mask > 0).astype(np.uint8) * 255
    size = max(1, int(radius)) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))
    marker = cv2.erode(source, kernel)
    if not np.any(marker):
        return source
    while True:
        grown = cv2.dilate(marker, kernel)
        current = cv2.bitwise_and(grown, source)
        if np.array_equal(current, marker):
            return current
        marker = current


def blackhat_subtract(mask: np.ndarray, kernel: int = 11, strength: float = 1.0) -> np.ndarray:
    """黑帽扣除：形态学黑帽检测暗色散点→从掩码中扣除。

    黑帽 = 闭运算 - 原图，标识原图中比闭运算暗的地方（即小散点）。
    """
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, element)
    blackhat = closed - mask_u8
    blackhat[blackhat < 0] = 0
    reduction = (blackhat * strength).astype(np.uint8)
    result = mask_u8.copy()
    result = cv2.subtract(result, reduction)
    return result


# ============================================================
# L5 连通域过滤层
# ============================================================

def area_filter(
    mask: np.ndarray,
    min_area: int = 60,
    connectivity: int = 8,
    relative_mode: bool = False,
    relative_ratio: float = 0.002,
    total_text_pixels: Optional[int] = None,
) -> np.ndarray:
    """连通域面积过滤：删除面积小于 min_area 的连通域。

    参数：
        mask: 二值掩码（0=背景, >0=文字）
        min_area: 绝对最小面积
        connectivity: 连通类型 4 或 8
        relative_mode: True 时 min_area = max(min_area, total_pixels * relative_ratio)
        total_text_pixels: 相对模式下计算绝对值时使用，None 则自动从 mask 算

    返回：过滤后的掩码 uint8
    """
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity, cv2.CV_32S)

    if relative_mode:
        ref = total_text_pixels if total_text_pixels is not None else int(mask_u8.sum() / 255)
        min_area = max(min_area, int(ref * relative_ratio))

    keep = np.zeros_like(mask_u8, dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 255
    return keep


def border_component_filter(
    mask: np.ndarray,
    max_width_ratio: float = 0.18,
    max_height_ratio: float = 0.18,
    preserve_largest: int = 8,
) -> np.ndarray:
    """删除贴边的窄条、版框和扫描阴影，同时保护主要文字连通域。"""
    source = (mask > 0).astype(np.uint8) * 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(source, 8, cv2.CV_32S)
    h, w = source.shape
    areas = [(label, int(stats[label, cv2.CC_STAT_AREA])) for label in range(1, count)]
    protected = {label for label, _ in sorted(areas, key=lambda item: item[1], reverse=True)[:max(1, int(preserve_largest))]}
    result = np.zeros_like(source)
    for label in range(1, count):
        x, y, width, height, _ = stats[label]
        touches = x == 0 or y == 0 or x + width >= w or y + height >= h
        narrow_vertical = width <= max(2, int(w * float(max_width_ratio))) and height >= int(h * 0.45)
        narrow_horizontal = height <= max(2, int(h * float(max_height_ratio))) and width >= int(w * 0.45)
        if touches and label not in protected and (narrow_vertical or narrow_horizontal):
            continue
        result[labels == label] = 255
    return result


def area_shape_filter(
    mask: np.ndarray,
    min_area: int = 60,
    connectivity: int = 8,
    relative_mode: bool = False,
    relative_ratio: float = 0.002,
    only_isolated: bool = True,
    max_aspect: float = 3.0,
    min_convexity: float = 0.7,
    min_solidity: float = 0.4,
    total_text_pixels: Optional[int] = None,
) -> np.ndarray:
    """面积+形状特征联合过滤。

    参数除面积过滤基础参数外，增补形状约束：
        only_isolated: True 时仅过滤孤立连通域（远离文字主体的一定距离）
        max_aspect: 最大允许长宽比
        min_convexity: 最小凸包面积比 (area / convex_hull_area)
        min_solidity: 最小实体面积比 (area / bbox_area)

    返回：过滤后的掩码 uint8
    """
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity, cv2.CV_32S)

    if relative_mode:
        ref = total_text_pixels if total_text_pixels is not None else int(mask_u8.sum() / 255)
        min_area = max(min_area, int(ref * relative_ratio))

    # 计算文字主体区域（前 5 大连通域）的质心
    areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
    areas.sort(key=lambda x: x[1], reverse=True)
    main_centroids = []
    main_areas = []
    main_ids = {idx for idx, _ in areas[:5]}
    for idx in main_ids:
        cx, cy = centroids[idx]
        main_centroids.append(np.array([cx, cy]))
        main_areas.append(stats[idx, cv2.CC_STAT_AREA])
    if not main_centroids:
        return mask_u8
    main_centroids = np.array(main_centroids)
    main_areas = np.array(main_areas, dtype=np.float64)

    keep = np.zeros_like(mask_u8, dtype=np.uint8)
    for i in range(1, num_labels):
        area_i = stats[i, cv2.CC_STAT_AREA]
        if area_i < min_area:
            # 面积不达标，直接丢弃
            continue
        # 计算形状指标
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        aspect = max(w, max(h, 1)) / max(min(w, h), 1)
        bbox_area = max(w * h, 1)
        solidity = area_i / bbox_area
        # 凸包比
        comp_mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        convexity = 1.0
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                convexity = area_i / hull_area

        # 孤立判定：质心是否远离所有文字主体
        if only_isolated:
            ci = np.array([centroids[i][0], centroids[i][1]])
            dists = np.linalg.norm(main_centroids - ci, axis=1)
            # 若该域本身是前5大之一，视为非孤立
            is_main = i in main_ids or any(d < 3 for d in dists)
            if not is_main and (aspect > max_aspect or convexity < min_convexity or solidity < min_solidity):
                continue
        else:
            if aspect > max_aspect or convexity < min_convexity or solidity < min_solidity:
                continue

        keep[labels == i] = 255
    return keep
