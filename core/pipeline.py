# pipeline.py — V3 分层管线执行器

import time
import numpy as np
from typing import Any, Optional

from core import algorithms as alg
from data.log_manager import write_log


# 管线层顺序
_LAYER_ORDER: list[str] = ["L1", "L2", "L3", "L4", "L5"]

# 每一层对应的算法分发映射
_ALGO_DISPATCH: dict[str, dict[str, Any]] = {
    "L1": {
        "高斯滤波": lambda a, p: alg.gaussian_blur(a, kernel=p.get("核大小", 5)),
        "中值滤波": lambda a, p: alg.median_blur(a, kernel=p.get("核大小", 3)),
        "双边滤波": lambda a, p: alg.bilateral_filter(a, d=p.get("d", 9), sigma_color=p.get("颜色σ", 75.0), sigma_space=p.get("空间σ", 75.0)),
        "NLM降噪": lambda a, p: alg.nlm_denoise(a, h=p.get("h", 10.0), template_window=p.get("模板窗", 7), search_window=p.get("搜索窗", 21)),
    },
    "L2": {
        "背景差分": lambda a, p: alg.bg_subtract(a, kernel=p.get("核大小", 31), threshold=p.get("阈值", 30), amplify=p.get("放大倍率", 2.0), normalize=p.get("归一化", True)),
        "形态学背景归一": lambda a, p: alg.bg_morph_normalize(a, kernel=p.get("核大小", 51)),
        "灰度黑帽增强": lambda a, p: alg.blackhat_background_enhance(a, kernel=p.get("核大小", 15), strength=p.get("强度", 1.0)),
    },
    "L3": {
        "Otsu": lambda a, p: alg.otsu_binarize(a, offset=p.get("偏移", 0)),
        "固定阈值": lambda a, p: alg.fixed_threshold_binarize(a, threshold=p.get("阈值", 160)),
        "Sauvola": lambda a, p: alg.sauvola_binarize(a, window=p.get("窗口", 25), k=p.get("k", 0.2), R=p.get("R", 128)),
        "percentile硬切": lambda a, p: alg.percentile_binarize(a, dark_ratio=p.get("暗色比例", 0.2)),
        "Niblack": lambda a, p: alg.niblack_binarize(a, window=p.get("窗口", 25), k=p.get("k", -0.2)),
        "Phansalkar": lambda a, p: alg.phansalkar_binarize(a, window=p.get("窗口", 25), k=p.get("k", 0.25), R=p.get("R", 128), p=p.get("p", 2.0), q=p.get("q", 10.0)),
        "Wolf-Jolion": lambda a, p: alg.wolf_binarize(a, window=p.get("窗口", 31), k=p.get("k", 0.35)),
    },
    "L4": {
        "黑帽扣除": lambda a, p: alg.blackhat_subtract(a, kernel=p.get("核大小", 11), strength=p.get("强度", 1.0)),
        "开运算": lambda a, p: alg.morph_open(a, radius=p.get("半径", 1), iterations=p.get("迭代", 1), shape=p.get("核形状", 1)),
        "闭运算": lambda a, p: alg.morph_close(a, radius=p.get("半径", 2), iterations=p.get("迭代", 1), shape=p.get("核形状", 1)),
        "孔洞填充": lambda a, p: alg.hole_fill(a, max_area=p.get("最大孔洞面积", 80), max_ratio=p.get("最大孔洞比例", 0.003)),
        "形态学重建": lambda a, p: alg.morphological_reconstruct(a, radius=p.get("半径", 1)),
    },
    "L5": {
        "面积过滤": lambda a, p: alg.area_filter(
            a, min_area=p.get("min_area", 60), connectivity=p.get("连通类型", 8),
            relative_mode=p.get("相对模式", False), relative_ratio=p.get("相对比例", 0.002),
            total_text_pixels=p.get("_total_text_pixels", None),
        ),
        "面积+形状过滤": lambda a, p: alg.area_shape_filter(
            a, min_area=p.get("min_area", 60), connectivity=p.get("连通类型", 8),
            relative_mode=p.get("相对模式", False), relative_ratio=p.get("相对比例", 0.002),
            only_isolated=p.get("仅孤立", True), max_aspect=p.get("最大长宽比", 3.0),
            min_convexity=p.get("最小凸包比", 0.7), min_solidity=p.get("最小实体比", 0.4),
            total_text_pixels=p.get("_total_text_pixels", None),
        ),
        "边界污染过滤": lambda a, p: alg.border_component_filter(
            a, max_width_ratio=p.get("最大宽度比例", 0.18),
            max_height_ratio=p.get("最大高度比例", 0.18),
            preserve_largest=p.get("保护主体数", 8),
        ),
    },
}


def algo_run(arr: np.ndarray, layer: str, algo_name: str, params: dict[str, Any]) -> np.ndarray:
    """在指定层上执行指定算法。

    参数：
        arr: 当前灰度数组 float32
        layer: "L1"~"L5"
        algo_name: 算法名称（中文，如"Otsu"）
        params: 参数字典
    返回：
        处理后的数组
     """
    dispatch = _ALGO_DISPATCH.get(layer, {})
    runner = dispatch.get(algo_name)
    if runner is None:
        return arr
    return runner(arr, params)


def run_pipeline(
    gray_arr: np.ndarray,
    scheme: dict[str, Any],
    total_text_pixels: Optional[int] = None,
    timing_label: Optional[str] = None,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """执行完整的 V3 分层去杂管线。

    参数：
        gray_arr: 灰度图像数组 (H, W) float32，范围 0~255
        scheme: V3 方案字典，格式 {'预处理': {...}, 'L1': {...}, 'L3': {...}, 'L5': {...}}
        total_text_pixels: 相对模式下 L5 使用的全文字像素估计值

    返回：
        (灰度结果 0~255 float32, 二值掩码 uint8) 或 (None, None) 空方案
    """
    pipeline_started = time.perf_counter()
    layer_timings: list[str] = []
    pre_started = time.perf_counter()
    # 预处理
    pre = scheme.get("预处理", {})
    arr = gray_arr.copy().astype(np.float32)

    if pre.get("转灰度", False):
        arr_255 = np.clip(arr, 0, 255).astype(np.float32)
        arr = arr_255  # 已是灰度，保持

    if pre.get("反相", False):
        arr = 255.0 - arr

    if pre.get("墨色归一", False):
        target = float(pre.get("墨色基准", 60))
        arr_u8 = np.clip(arr, 0, 255).astype(np.uint8)
        thresh, _ = cv2.threshold(arr_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        text_mask = arr_u8 < thresh
        if text_mask.any():
            cur_ink = np.median(arr_u8[text_mask])
        else:
            cur_ink = 60.0
        if cur_ink > 0 and cur_ink != target:
            import math
            gamma = math.log(target / 255.0) / math.log(max(cur_ink, 1.0) / 255.0)
            gamma = max(0.4, min(2.5, gamma))
            arr = 255.0 * (arr / 255.0) ** gamma
    if timing_label:
        layer_timings.append(f"预处理={time.perf_counter() - pre_started:.4f}秒")

    # 逐层执行
    current = arr
    mask = None
    for layer in _LAYER_ORDER:
        layer_cfg = scheme.get(layer)
        if not layer_cfg:
            continue
        algo_name = layer_cfg.get("算法", "")
        params = dict(layer_cfg.get("参数", {}))
        if total_text_pixels is not None and layer in ("L5",):
            params["_total_text_pixels"] = total_text_pixels
        layer_started = time.perf_counter()
        current = algo_run(current, layer, algo_name, params)
        layer_timings.append(f"{layer}:{algo_name}={time.perf_counter() - layer_started:.4f}秒")
        if layer in ("L3", "L4", "L5"):
            # L3 及其后的清理层都输出二值掩码，最终掩码必须反映完整管线结果。
            mask = current.copy()

    if timing_label:
        write_log(
            f"管线明细｜标签={timing_label}｜总耗时={time.perf_counter() - pipeline_started:.4f}秒｜"
            + "｜".join(layer_timings)
        )
    return arr, mask


# 延迟导入循环
import cv2  # noqa: E402
