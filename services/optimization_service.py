# optimization_service.py — 多候选自动优化服务

import hashlib
import os
import time
from typing import Any, Optional

import numpy as np
from PIL import Image

import config
from core.optimizer import generate_candidate_results
from data.log_manager import write_log
from services.glyph_service import GlyphService
from utils.file_utils import compute_file_md5


class OptimizationService:
    """始终读取原始文件，生成多路线候选并确认到中间文件。"""

    def __init__(self, glyph_service: GlyphService) -> None:
        self._glyph = glyph_service

    def change_variant_char(self, variant_id: str, new_char: str) -> None:
        """修改当前字形的归属字符，并同步全部阶段文件。"""
        self._glyph.move_variant_to_char(variant_id, new_char)

    def remove_failed_variant(self, variant_id: str) -> bool:
        """删除无法自动优化的字形记录，并保留各阶段图片文件。"""
        return self._glyph.remove_variant_record(variant_id)

    def list_items(self) -> list[dict[str, Any]]:
        """按字形组顺序返回自动优化页面所需的字形任务。"""
        items: list[dict[str, Any]] = []
        source_dir = self._glyph.get_three_dirs()[0]
        for char_order, char in enumerate(self._glyph.get_all_chars()):
            for index, detail in enumerate(self._glyph.get_char_variants(char)):
                filename = str(detail.get("原始文件", ""))
                if not filename:
                    continue
                optimization = detail.get("自动优化", {})
                state = str(detail.get("状态", config.STATUS_PENDING_OPTIMIZATION))
                items.append({
                    **detail,
                    "键": str(detail.get("变体ID", "")),
                    "归属字": char,
                    "字符顺序": char_order,
                    "变体序号": index + 1,
                    "原始文件名": filename,
                    "原始路径": os.path.join(source_dir, filename),
                    "显示状态": self._display_status(state),
                    "得分": optimization.get("得分"),
                })
        return items

    def generate_candidates(
        self,
        item: dict[str, Any],
        parent_scheme: Optional[dict[str, Any]] = None,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        path = str(item.get("原始路径", ""))
        started_at = time.perf_counter()
        load_started = time.perf_counter()
        gray = self._load_white_background_gray(path)
        load_elapsed = time.perf_counter() - load_started
        optimize_started = time.perf_counter()
        results = generate_candidate_results(gray, parent_scheme=parent_scheme, limit=limit)
        optimize_elapsed = time.perf_counter() - optimize_started
        candidates: list[dict[str, Any]] = []
        seen_images: set[str] = set()
        for result in results:
            mask = np.asarray(result["掩码"], dtype=np.uint8)
            protect_original = bool(result.get("保留原图", False))
            image = self._gray_to_transparent_image(gray, mask, protect_original)
            digest_source = image.tobytes()
            digest = hashlib.sha256(digest_source).hexdigest()
            if digest in seen_images:
                continue
            seen_images.add(digest)
            candidates.append({
                "方案名": result["方案名"],
                "方案": result["方案"],
                "得分": float(result["得分"]),
                "图像": image,
                "图像指纹": digest,
                "质量等级": result.get("质量等级", ""),
                "灰度母版": np.clip(gray, 0, 255).astype(np.uint8),
                "清洁掩码": (mask > 0).astype(np.uint8),
                "保留原图": protect_original,
            })
            if len(candidates) >= limit:
                break
        write_log(
            f"自动优化服务结束｜字形={item.get('归属字', '')}｜文件={os.path.basename(path)}｜"
            f"读取={load_elapsed:.4f}秒｜算法={optimize_elapsed:.4f}秒｜"
            f"结果包装={time.perf_counter() - optimize_started - optimize_elapsed:.4f}秒｜"
            f"总耗时={time.perf_counter() - started_at:.4f}秒｜候选数={len(candidates)}"
        )
        return candidates

    def explore(self, item: dict[str, Any], candidate: dict[str, Any], count: int = 8) -> list[dict[str, Any]]:
        """围绕选中候选的方案参数生成下一轮，并排除基准结果。"""
        results = self.generate_candidates(item, parent_scheme=candidate.get("方案", {}), limit=count + 1)
        base_digest = candidate.get("图像指纹")
        return [result for result in results if result.get("图像指纹") != base_digest][:count]

    def save_selection(self, item: dict[str, Any], candidate: dict[str, Any], round_number: int = 1) -> str:
        variant_id = str(item.get("键", ""))
        detail = self._glyph.get_variant(variant_id)
        if not detail:
            raise ValueError("字形记录不存在。")
        workflow_dirs = self._glyph.get_workflow_dirs()
        filename = os.path.splitext(str(detail.get("原始文件", "字形")))[0] + ".png"
        preview_path = os.path.join(workflow_dirs["优化预览"], filename)
        gray_master_path = os.path.join(workflow_dirs["灰度母版"], filename)
        clean_mask_path = os.path.join(workflow_dirs["清洁掩码"], filename)
        reviewed_path = os.path.join(workflow_dirs["手工审核"], filename)
        finished_path = os.path.join(workflow_dirs["成品"], filename)
        image = candidate.get("图像")
        gray_master = candidate.get("灰度母版")
        clean_mask = candidate.get("清洁掩码")
        if not isinstance(image, Image.Image) or not isinstance(gray_master, np.ndarray) or not isinstance(clean_mask, np.ndarray):
            raise ValueError("候选分层数据无效。")
        dpi = self._source_dpi(detail)
        image.save(preview_path, "PNG", dpi=dpi)
        Image.fromarray(np.clip(gray_master, 0, 255).astype(np.uint8), "L").save(
            gray_master_path, "PNG", dpi=dpi
        )
        Image.fromarray((clean_mask > 0).astype(np.uint8) * 255, "L").save(
            clean_mask_path, "PNG", dpi=dpi
        )
        for stale_path in (reviewed_path, finished_path):
            if os.path.isfile(stale_path):
                os.remove(stale_path)
        self._glyph.confirm_optimization(
            variant_id,
            filename,
            compute_file_md5(preview_path),
            str(candidate.get("方案名", "自动优化")),
            dict(candidate.get("方案", {})),
            float(candidate.get("得分", 0.0)),
            round_number,
            gray_master_filename=filename,
            gray_master_md5=compute_file_md5(gray_master_path),
            clean_mask_filename=filename,
            clean_mask_md5=compute_file_md5(clean_mask_path),
        )
        self._glyph.save()
        return preview_path

    @staticmethod
    def _source_dpi(detail: dict[str, Any]) -> tuple[float, float]:
        """读取原图记录的 DPI，并为无效数据提供安全默认值。"""
        image_info = detail.get("图像信息", {})
        if not isinstance(image_info, dict):
            image_info = {}

        def valid_dpi(value: Any, fallback: float) -> float:
            try:
                dpi = float(value)
            except (TypeError, ValueError):
                return fallback
            return dpi if 1.0 <= dpi <= 9600.0 else fallback

        dpi_x = valid_dpi(image_info.get("水平DPI"), 300.0)
        dpi_y = valid_dpi(image_info.get("垂直DPI"), dpi_x)
        return dpi_x, dpi_y

    @staticmethod
    def _display_status(state: str) -> str:
        if state == config.STATUS_PENDING_OPTIMIZATION:
            return "待优化"
        if state == config.STATUS_PENDING_MANUAL_REVIEW:
            return "已优化"
        if state == config.STATUS_REVIEWED:
            return "审核通过"
        if state == config.STATUS_FINISHED:
            return "成品已生成"
        return state or "待优化"

    @staticmethod
    def _load_white_background_gray(path: str) -> np.ndarray:
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"找不到原始图片：{path}")
        with Image.open(path) as source_image:
            source_image.seek(0)
            rgba = source_image.convert("RGBA")
            white_background = Image.new("RGBA", rgba.size, "white")
            white_background.alpha_composite(rgba)
            return np.asarray(white_background.convert("L"), dtype=np.float32)

    @staticmethod
    def _gray_to_transparent_image(
        gray: np.ndarray, mask: np.ndarray, protect_original: bool = False
    ) -> Image.Image:
        """以灰度母版提供墨色、清洁掩码限定字形，生成阶段性透明预览。"""
        gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
        alpha = 255 - gray_u8
        if not protect_original:
            alpha = np.where(mask > 0, alpha, 0).astype(np.uint8)
        rgba = np.zeros((*gray_u8.shape, 4), dtype=np.uint8)
        rgba[..., 3] = alpha
        return Image.fromarray(rgba, "RGBA")

    @staticmethod
    def _mask_to_transparent_image(mask: np.ndarray) -> Image.Image:
        foreground = mask > 0
        rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        rgba[foreground, :3] = 0
        rgba[foreground, 3] = 255
        return Image.fromarray(rgba, "RGBA")
