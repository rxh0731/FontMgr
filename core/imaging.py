# imaging.py — 通用图像工具函数（纯计算，无 UI 依赖）

import cv2
import numpy as np
from PIL import Image, ImageOps
from typing import Optional, Tuple


# ============================================================
# 灰度与极性
# ============================================================

def normalize_text_polarity(gray_arr: np.ndarray) -> tuple[np.ndarray, bool]:
    """将文字图自动校正为白底深字，返回校正结果及是否执行过反相。"""
    arr = np.clip(gray_arr, 0, 255).astype(np.float32)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("极性判断需要非空的二维灰度图。")

    height, width = arr.shape
    border_width = max(1, min(height, width) // 20)
    border = np.concatenate((
        arr[:border_width, :].ravel(),
        arr[-border_width:, :].ravel(),
        arr[:, :border_width].ravel(),
        arr[:, -border_width:].ravel(),
    ))
    low = float(np.percentile(arr, 5))
    high = float(np.percentile(arr, 95))
    if high - low < 12.0:
        should_invert = float(np.median(border)) < 127.5
    else:
        pivot = (low + high) / 2.0
        dark_border_ratio = float(np.mean(border < pivot))
        if dark_border_ratio >= 0.65:
            should_invert = True
        elif dark_border_ratio <= 0.35:
            should_invert = False
        else:
            should_invert = float(np.median(arr)) < pivot
    return (255.0 - arr if should_invert else arr), should_invert


# ============================================================
# 缩放与裁剪
# ============================================================

def resize_premul(img: Image.Image, max_pixels: int) -> Image.Image:
    """等比缩放到最大像素数以内（宽或高不超过 sqrt(max_pixels) 的近似）。"""
    w, h = img.size
    pixels = w * h
    if pixels <= max_pixels:
        return img.copy()
    ratio = (max_pixels / pixels) ** 0.5
    nw = max(1, int(w * ratio))
    nh = max(1, int(h * ratio))
    return img.resize((nw, nh), Image.LANCZOS)


def find_text_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """在二值掩码（0=背景，>0=文字）中找出文字包围盒。

    返回：(min_x, min_y, max_x, max_y) 或 None（全背景时）。
    """
    if mask is None or not mask.any():
        return None
    from PIL import Image
    pil_img = Image.fromarray(mask.astype(np.uint8) * 255)
    bbox = pil_img.getbbox()
    return bbox


# ============================================================
# 透视变换
# ============================================================

def warp_distort(
    src_img: np.ndarray,
    out_quad: np.ndarray,
    out_size: Tuple[int, int],
) -> np.ndarray:
    """OpenCV 透视变换扭曲。

    参数：
        src_img: 源图像 (H, W) 或 (H, W, C)
        out_quad: 四个输出角点坐标 [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]
        out_size: (width, height) 输出画布尺寸
    返回：
        扭曲后的 numpy 数组
    """
    h, w = out_size[1], out_size[0]
    in_quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    out_quad = out_quad.astype(np.float32)
    matrix = cv2.getPerspectiveTransform(in_quad, out_quad)
    if src_img.ndim == 2:
        result = cv2.warpPerspective(src_img, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    else:
        result = cv2.warpPerspective(src_img, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return result


# ============================================================
# 画布合成
# ============================================================

def compose_on_canvas(
    glyph: np.ndarray,
    mask: np.ndarray,
    canvas_w: int,
    canvas_h: int,
    offset_x: int = 0,
    offset_y: int = 0,
) -> Image.Image:
    """将去杂后的文字合成到画布上。

    参数：
        glyph: 去杂后的 RGBA numpy 数组 (H, W, 4)
        mask: 二值文字掩码 (H, W)
        canvas_w, canvas_h: 画布尺寸
        offset_x, offset_y: 文字在画布上的偏移
    返回：
        RGBA PIL Image
    """
    canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
    gh, gw = glyph.shape[:2]
    # 裁剪文字区域使其不超出画布
    src_x1 = max(0, -offset_x)
    src_y1 = max(0, -offset_y)
    src_x2 = min(gw, canvas_w - offset_x)
    src_y2 = min(gh, canvas_h - offset_y)
    dst_x1 = max(0, offset_x)
    dst_y1 = max(0, offset_y)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    if src_x2 > src_x1 and src_y2 > src_y1 and dst_x2 > dst_x1 and dst_y2 > dst_y1:
        if mask is not None:
            m = mask[src_y1:src_y2, src_x1:src_x2] > 0
            canvas[dst_y1:dst_y2, dst_x1:dst_x2, :][m] = glyph[src_y1:src_y2, src_x1:src_x2, :][m]
        else:
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = glyph[src_y1:src_y2, src_x1:src_x2]
    return Image.fromarray(canvas, "RGBA")
