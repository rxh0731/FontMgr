# adjustment_service.py — 字库整体协调与成品生成

import math
import os
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image, ImageFilter

import config
from services.glyph_service import GlyphService
from utils.file_utils import compute_file_md5, pinyin_natural_key


class AdjustmentService:
    """统一审核字形的几何、墨色、边缘风格与成品画布。"""

    def __init__(self, glyph_service: GlyphService) -> None:
        self._glyph = glyph_service

    def load_reviewed_variants(self, pinyin_order: bool = True) -> list[dict[str, Any]]:
        variants = self._glyph.get_variants_by_status(config.STATUS_REVIEWED, config.STATUS_FINISHED)
        if pinyin_order:
            variants.sort(key=lambda item: pinyin_natural_key(str(item.get("归属字", ""))))
        return variants

    def analyze(self, target_ratio: Optional[float] = None) -> dict[str, Any]:
        """统计审核结果的尺寸与可信墨色，作为整库协调基准。"""
        width_ratios: list[float] = []
        height_ratios: list[float] = []
        ink_values: list[float] = []
        for detail in self.load_reviewed_variants():
            image = self._load_reviewed_image(detail)
            if image is None:
                continue
            bounding_box = image.getchannel("A").getbbox()
            if not bounding_box:
                continue
            width_ratios.append((bounding_box[2] - bounding_box[0]) / max(image.width, 1))
            height_ratios.append((bounding_box[3] - bounding_box[1]) / max(image.height, 1))
            ink = self._glyph_ink_value(image)
            if ink is not None:
                ink_values.append(ink)
        valid_count = len(width_ratios)
        if valid_count == 0:
            return {"有效数": 0, "目标占比": 0.72, "宽中位": 0.0, "高中位": 0.0, "墨色基准": 220.0}
        median_width = float(np.median(width_ratios))
        median_height = float(np.median(height_ratios))
        default_ratio = max(0.35, min(0.9, max(median_width, median_height)))
        ink_baseline = float(np.median(ink_values)) if ink_values else 220.0
        return {
            "有效数": valid_count,
            "目标占比": round(float(target_ratio if target_ratio is not None else default_ratio), 4),
            "宽中位": round(median_width, 4),
            "高中位": round(median_height, 4),
            "墨色基准": round(ink_baseline, 2),
        }

    def preview_variant(
        self,
        detail: dict[str, Any],
        target_ratio: float,
        adjustments: Optional[dict[str, Any]] = None,
        ink_baseline: Optional[float] = None,
    ) -> Optional[tuple[Image.Image, dict[str, Any]]]:
        image = self._load_reviewed_image(detail)
        if image is None:
            return None
        metadata = self._glyph.get_metadata()
        canvas_width = max(1, int(metadata.get("画布宽", 250)))
        canvas_height = max(1, int(metadata.get("画布高", 250)))
        bounding_box = image.getchannel("A").getbbox()
        if not bounding_box:
            return Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0)), {"缩放": 1.0, "偏移X": 0, "偏移Y": 0}

        # 完整图片（含透明边）先与田字格中心对齐，实际文字包围盒只决定目标大小。
        glyph_width = bounding_box[2] - bounding_box[0]
        glyph_height = bounding_box[3] - bounding_box[1]
        target_width = max(1, int(canvas_width * target_ratio))
        target_height = max(1, int(canvas_height * target_ratio))
        scale = min(target_width / max(glyph_width, 1), target_height / max(glyph_height, 1))
        new_width = max(1, int(round(image.width * scale)))
        new_height = max(1, int(round(image.height * scale)))
        glyph = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        applied = self._normalized_adjustments(adjustments)
        glyph = self._apply_global_transform(glyph, applied)
        glyph = self._apply_output_style(glyph, metadata.get("成品风格", "灰度保真"), ink_baseline)
        left = (
            (canvas_width - glyph.width) // 2
            + int(round(applied["移动X"]))
        )
        top = (
            (canvas_height - glyph.height) // 2
            + int(round(applied["移动Y"]))
        )
        output_bbox = glyph.getchannel("A").getbbox()
        if output_bbox:
            glyph_left = left + output_bbox[0]
            glyph_top = top + output_bbox[1]
            glyph_right = left + output_bbox[2]
            glyph_bottom = top + output_bbox[3]
            expand_x = int(math.ceil(max(0, -glyph_left, glyph_right - canvas_width)))
            expand_y = int(math.ceil(max(0, -glyph_top, glyph_bottom - canvas_height)))
        else:
            expand_x = expand_y = 0
        finished_image = Image.new(
            "RGBA",
            (canvas_width + expand_x * 2, canvas_height + expand_y * 2),
            (0, 0, 0, 0),
        )
        finished_image.alpha_composite(
            glyph, (left + expand_x, top + expand_y)
        )
        parameters = {
            "缩放": round(scale, 6),
            "偏移X": left + expand_x,
            "偏移Y": top + expand_y,
            "对称扩展X": expand_x,
            "对称扩展Y": expand_y,
            "标准画布": [canvas_width, canvas_height],
            "实际画布": list(finished_image.size),
            "目标占比": round(target_ratio, 4),
            "原包围盒": list(bounding_box),
            "整体变换": applied,
            "成品风格": metadata.get("成品风格", "灰度保真"),
            "墨色基准": round(float(ink_baseline), 2) if ink_baseline is not None else None,
        }
        return finished_image, parameters

    def preview_coordinated(
        self,
        detail: dict[str, Any],
        adjustments: Optional[dict[str, Any]] = None,
        work_ratio: float = 1.3,
    ) -> Optional[tuple[Image.Image, tuple[int, int, int, int]]]:
        """生成整体协调工作预览，先按完整审核图居中，再裁透明边并保留位置。"""
        source = self._load_reviewed_image(detail)
        if source is None:
            return None
        metadata = self._glyph.get_metadata()
        grid_width = max(1, int(metadata.get("画布宽", 250)))
        grid_height = max(1, int(metadata.get("画布高", 250)))
        work_width = max(grid_width, int(round(grid_width * work_ratio)))
        work_height = max(grid_height, int(round(grid_height * work_ratio)))
        source_left = (work_width - source.width) // 2
        source_top = (work_height - source.height) // 2
        source_alpha = source.getchannel("A").point([0] * 16 + [255] * 240)
        bounding_box = source_alpha.getbbox()
        if not bounding_box:
            return Image.new("RGBA", (work_width, work_height), (0, 0, 0, 0)), (0, 0, 0, 0)
        glyph = source.crop(bounding_box)
        center_x = source_left + (bounding_box[0] + bounding_box[2]) / 2.0
        center_y = source_top + (bounding_box[1] + bounding_box[3]) / 2.0
        applied = self._normalized_coordination(adjustments)
        glyph = self._apply_coordination_transform(glyph, applied)
        left = int(round(center_x - glyph.width / 2.0 + applied["移动X"]))
        top = int(round(center_y - glyph.height / 2.0 + applied["移动Y"]))
        preview = Image.new("RGBA", (work_width, work_height), (0, 0, 0, 0))
        preview.alpha_composite(glyph, (left, top))
        glyph_alpha = glyph.getchannel("A").point([0] * 16 + [255] * 240)
        glyph_bbox = glyph_alpha.getbbox()
        if glyph_bbox is None:
            return preview, (0, 0, 0, 0)
        return preview, (
            left + glyph_bbox[0],
            top + glyph_bbox[1],
            left + glyph_bbox[2],
            top + glyph_bbox[3],
        )

    def save_coordinated_variants(
        self,
        variants: list[dict[str, Any]],
        adjustments_by_id: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """逐字生成当前页成品，沿用手工审核的标准画布和对称扩展原则。"""
        metadata = self._glyph.get_metadata()
        canvas_width = max(1, int(metadata.get("画布宽", 250)))
        canvas_height = max(1, int(metadata.get("画布高", 250)))
        target_dpi = float(metadata.get("DPI", metadata.get("分辨率", 300)) or 300)
        finished_dir = self._glyph.get_workflow_dirs()["成品"]
        os.makedirs(finished_dir, exist_ok=True)
        success_count = 0
        failures: list[tuple[str, str]] = []
        for detail in variants:
            variant_id = str(detail.get("变体ID", ""))
            try:
                source = self._load_reviewed_image(detail)
                if source is None:
                    raise FileNotFoundError("找不到审核通过的文字图片")
                source_left = (canvas_width - source.width) // 2
                source_top = (canvas_height - source.height) // 2
                bounding_box = source.getchannel("A").getbbox()
                if not bounding_box:
                    glyph = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
                    center_x = canvas_width / 2.0
                    center_y = canvas_height / 2.0
                else:
                    glyph = source.crop(bounding_box)
                    center_x = source_left + (bounding_box[0] + bounding_box[2]) / 2.0
                    center_y = source_top + (bounding_box[1] + bounding_box[3]) / 2.0
                applied = self._normalized_coordination(adjustments_by_id.get(variant_id))
                glyph = self._apply_coordination_transform(glyph, applied)
                left = int(round(center_x - glyph.width / 2.0 + applied["移动X"]))
                top = int(round(center_y - glyph.height / 2.0 + applied["移动Y"]))
                output_bbox = glyph.getchannel("A").getbbox()
                if output_bbox:
                    glyph_left = left + output_bbox[0]
                    glyph_top = top + output_bbox[1]
                    glyph_right = left + output_bbox[2]
                    glyph_bottom = top + output_bbox[3]
                    expand_x = int(math.ceil(max(0, -glyph_left, glyph_right - canvas_width)))
                    expand_y = int(math.ceil(max(0, -glyph_top, glyph_bottom - canvas_height)))
                else:
                    expand_x = expand_y = 0
                finished = Image.new(
                    "RGBA",
                    (canvas_width + expand_x * 2, canvas_height + expand_y * 2),
                    (0, 0, 0, 0),
                )
                finished.alpha_composite(glyph, (left + expand_x, top + expand_y))
                filename = os.path.splitext(detail.get("原始文件", "字形"))[0] + ".png"
                path = os.path.join(finished_dir, filename)
                finished.save(path, "PNG", dpi=(target_dpi, target_dpi))
                parameters = {
                    "标准画布": [canvas_width, canvas_height],
                    "实际画布": list(finished.size),
                    "对称扩展X": expand_x,
                    "对称扩展Y": expand_y,
                    "整体变换": applied,
                    "原包围盒": list(bounding_box) if bounding_box else None,
                }
                self._glyph.mark_finished(variant_id, filename, compute_file_md5(path), parameters)
                success_count += 1
            except (OSError, ValueError, RuntimeError) as exc:
                failures.append((variant_id, str(exc)))
        self._glyph.save()
        return {"成功": success_count, "失败": len(failures), "失败详情": failures}

    @staticmethod
    def _normalized_coordination(adjustments: Optional[dict[str, Any]]) -> dict[str, Any]:
        source = adjustments or {}
        raw_distort = source.get("扭曲", [0.0] * 8)
        if not isinstance(raw_distort, (list, tuple)) or len(raw_distort) != 8:
            raw_distort = [0.0] * 8
        return {
            "移动X": float(source.get("移动X", 0.0)),
            "移动Y": float(source.get("移动Y", 0.0)),
            "缩放X": max(0.15, min(5.0, float(source.get("缩放X", source.get("额外缩放", 1.0))))),
            "缩放Y": max(0.15, min(5.0, float(source.get("缩放Y", source.get("额外缩放", 1.0))))),
            "旋转": max(-180.0, min(180.0, float(source.get("旋转", 0.0)))),
            "斜切X": max(-25.0, min(25.0, float(source.get("斜切X", 0.0)))),
            "斜切Y": max(-25.0, min(25.0, float(source.get("斜切Y", 0.0)))),
            "扭曲": [float(value) for value in raw_distort],
        }

    @staticmethod
    def _apply_coordination_transform(image: Image.Image, adjustments: dict[str, Any]) -> Image.Image:
        width = max(1, int(round(image.width * adjustments["缩放X"])))
        height = max(1, int(round(image.height * adjustments["缩放Y"])))
        if (width, height) != image.size:
            image = image.resize((width, height), Image.Resampling.BICUBIC)
        compatible = {
            "额外缩放": 1.0,
            "旋转": adjustments["旋转"],
            "斜切X": adjustments["斜切X"],
            "斜切Y": adjustments["斜切Y"],
            "扭曲": adjustments["扭曲"],
        }
        return AdjustmentService._apply_global_transform(image, compatible)

    def generate_finished(
        self,
        target_ratio: Optional[float] = None,
        adjustments: Optional[dict[str, Any]] = None,
    ) -> dict[str, int]:
        baseline = self.analyze(target_ratio)
        ratio = float(baseline["目标占比"])
        ink_baseline = float(baseline["墨色基准"])
        metadata = self._glyph.get_metadata()
        target_dpi = float(
            metadata.get("DPI", metadata.get("分辨率", 300)) or 300
        )
        finished_dir = self._glyph.get_workflow_dirs()["成品"]
        os.makedirs(finished_dir, exist_ok=True)
        success_count = failure_count = 0
        for detail in self.load_reviewed_variants():
            preview = self.preview_variant(detail, ratio, adjustments, ink_baseline)
            if preview is None:
                failure_count += 1
                continue
            image, parameters = preview
            filename = os.path.splitext(detail.get("原始文件", "字形"))[0] + ".png"
            path = os.path.join(finished_dir, filename)
            try:
                image.save(path, "PNG", dpi=(target_dpi, target_dpi))
                self._glyph.mark_finished(detail["变体ID"], filename, compute_file_md5(path), parameters)
                success_count += 1
            except OSError:
                failure_count += 1
        baseline["整体变换"] = self._normalized_adjustments(adjustments)
        baseline["成品风格"] = self._glyph.get_metadata().get("成品风格", "灰度保真")
        self._glyph.set_coordination_summary(baseline, ink_baseline)
        self._glyph.save()
        return {"成功": success_count, "失败": failure_count}

    def _load_reviewed_image(self, detail: dict[str, Any]) -> Optional[Image.Image]:
        workflow_dirs = self._glyph.get_workflow_dirs()
        candidates = (
            os.path.join(workflow_dirs["手工审核"], detail.get("审核文件", "")),
            os.path.join(workflow_dirs["优化预览"], detail.get("中间文件", "")),
        )
        for path in candidates:
            image = self._open_rgba(path)
            if image is not None:
                return image
        return None

    @staticmethod
    def _normalized_adjustments(adjustments: Optional[dict[str, Any]]) -> dict[str, float]:
        source = adjustments or {}
        return {
            "移动X": float(source.get("移动X", 0.0)),
            "移动Y": float(source.get("移动Y", 0.0)),
            "额外缩放": float(source.get("额外缩放", 1.0)),
            "旋转": float(source.get("旋转", 0.0)),
            "斜切X": float(source.get("斜切X", 0.0)),
            "斜切Y": float(source.get("斜切Y", 0.0)),
            "扭曲X": float(source.get("扭曲X", 0.0)),
            "扭曲Y": float(source.get("扭曲Y", 0.0)),
        }

    @staticmethod
    def _apply_global_transform(image: Image.Image, adjustments: dict[str, Any]) -> Image.Image:
        scale = max(0.5, min(1.5, adjustments["额外缩放"]))
        if not math.isclose(scale, 1.0):
            image = image.resize(
                (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale)))),
                Image.Resampling.BICUBIC,
            )
        if not math.isclose(adjustments["旋转"], 0.0):
            image = image.rotate(-adjustments["旋转"], Image.Resampling.BICUBIC, expand=True)
        shear_x = math.tan(math.radians(max(-25.0, min(25.0, adjustments["斜切X"]))))
        shear_y = math.tan(math.radians(max(-25.0, min(25.0, adjustments["斜切Y"]))))
        if not math.isclose(shear_x, 0.0) or not math.isclose(shear_y, 0.0):
            margin_x = int(abs(shear_x) * image.height) + 2
            margin_y = int(abs(shear_y) * image.width) + 2
            output_size = (image.width + margin_x, image.height + margin_y)
            image = image.transform(
                output_size,
                Image.Transform.AFFINE,
                (1.0, -shear_x, margin_x // 2, -shear_y, 1.0, margin_y // 2),
                Image.Resampling.BICUBIC,
            )
        raw_distort = adjustments.get("扭曲")
        if isinstance(raw_distort, (list, tuple)) and len(raw_distort) == 8:
            distort = [float(value) for value in raw_distort]
            if any(not math.isclose(value, 0.0) for value in distort):
                rgba = np.asarray(image, dtype=np.uint8)
                height, width = rgba.shape[:2]
                source = np.empty((4, 2), dtype=np.float32)
                source[0] = (0.0, 0.0)
                source[1] = (float(width - 1), 0.0)
                source[2] = (float(width - 1), float(height - 1))
                source[3] = (0.0, float(height - 1))
                margin = int(math.ceil(max(abs(value) for value in distort))) + 2
                target = np.empty((4, 2), dtype=np.float32)
                target[0] = (distort[0] + margin, distort[1] + margin)
                target[1] = (width - 1 + distort[2] + margin, distort[3] + margin)
                target[2] = (width - 1 + distort[4] + margin, height - 1 + distort[5] + margin)
                target[3] = (distort[6] + margin, height - 1 + distort[7] + margin)
                output_size = (width + margin * 2, height + margin * 2)
                matrix = cv2.getPerspectiveTransform(source, target)
                rgba = cv2.warpPerspective(
                    rgba,
                    matrix,
                    output_size,
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0, 0),
                )
                image = Image.fromarray(rgba, "RGBA")
        else:
            distort_x = max(-0.3, min(0.3, float(adjustments.get("扭曲X", 0.0))))
            distort_y = max(-0.3, min(0.3, float(adjustments.get("扭曲Y", 0.0))))
            if not math.isclose(distort_x, 0.0) or not math.isclose(distort_y, 0.0):
                rgba = np.asarray(image, dtype=np.uint8)
                height, width = rgba.shape[:2]
                source = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
                dx = abs(distort_x) * width * 0.5
                dy = abs(distort_y) * height * 0.5
                target = np.float32([
                    [dx if distort_x > 0 else 0, dy if distort_y > 0 else 0],
                    [width - 1 - (dx if distort_x < 0 else 0), dy if distort_y < 0 else 0],
                    [width - 1 - (dx if distort_x > 0 else 0), height - 1 - (dy if distort_y > 0 else 0)],
                    [dx if distort_x < 0 else 0, height - 1 - (dy if distort_y < 0 else 0)],
                ])
                matrix = cv2.getPerspectiveTransform(source, target)
                rgba = cv2.warpPerspective(rgba, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
                image = Image.fromarray(rgba, "RGBA")
        return image

    def _apply_output_style(self, image: Image.Image, style: str, ink_baseline: Optional[float]) -> Image.Image:
        alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)
        if style == "纯二值":
            output_alpha = np.where(alpha >= 16, 255, 0).astype(np.uint8)
        elif style == "统一软边":
            hard_mask = Image.fromarray(np.where(alpha >= 32, 255, 0).astype(np.uint8), "L")
            output_alpha = np.asarray(hard_mask.filter(ImageFilter.GaussianBlur(radius=0.7)), dtype=np.uint8)
        else:
            output_alpha = self._normalize_ink(alpha, ink_baseline)
        result = Image.new("RGBA", image.size, (0, 0, 0, 0))
        result.putalpha(Image.fromarray(output_alpha, "L"))
        return result

    @staticmethod
    def _normalize_ink(alpha: np.ndarray, target: Optional[float]) -> np.ndarray:
        values = alpha[alpha >= 16]
        if values.size == 0 or target is None:
            return alpha.copy()
        current = float(np.percentile(values, 70))
        if current <= 0 or current >= 255:
            return alpha.copy()
        target_value = max(32.0, min(250.0, float(target)))
        gamma = math.log(target_value / 255.0) / math.log(current / 255.0)
        gamma = max(0.75, min(1.35, gamma))
        normalized = 255.0 * np.power(alpha.astype(np.float32) / 255.0, gamma)
        normalized[normalized < 6.0] = 0.0
        return np.clip(normalized, 0, 255).astype(np.uint8)

    @staticmethod
    def _glyph_ink_value(image: Image.Image) -> Optional[float]:
        alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)
        values = alpha[alpha >= 16]
        return float(np.percentile(values, 70)) if values.size else None

    @staticmethod
    def _open_rgba(path: str) -> Optional[Image.Image]:
        if not path or not os.path.exists(path):
            return None
        try:
            with Image.open(path) as image:
                return image.convert("RGBA")
        except (OSError, ValueError):
            return None
