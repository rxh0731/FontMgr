# export_service.py — 按字库字符导出成品图片

import os
import shutil
from typing import Any, Callable, Optional

import config
from services.glyph_service import GlyphService
from utils.file_utils import ensure_dir


class ExportService:
    """按照字库登记字符导出成品图片。"""

    def __init__(
        self,
        glyph_service: GlyphService,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        self._glyph: GlyphService = glyph_service
        self._progress: Optional[Callable[[str, int, int], None]] = progress_callback

    def export(self, output_dir: str) -> dict[str, int]:
        """按照字库中登记的字符原样导出已完成的图片。"""
        ensure_dir(output_dir)
        _, _, final_dir = self._glyph.get_three_dirs()

        if not os.path.isdir(final_dir):
            return {"导出": 0, "跳过": 0, "失败": 0}

        exported = 0
        skipped = 0
        failed = 0

        chars = self._glyph.get_all_chars()
        total = len(chars)

        for idx, char in enumerate(chars):
            variants = self._glyph.get_char_variants(char)
            if not variants:
                continue

            for vi, v in enumerate(variants):
                if v.get("状态") != config.STATUS_FINISHED:
                    skipped += 1
                    continue

                fname_png = v.get("成品文件", "")
                if not fname_png:
                    skipped += 1
                    continue

                src_path = os.path.join(final_dir, fname_png)
                if not os.path.exists(src_path):
                    skipped += 1
                    continue

                dest_name = self._make_dest_name(char, fname_png, vi)

                dest_path = os.path.join(output_dir, dest_name)
                try:
                    shutil.copy2(src_path, dest_path)
                    exported += 1
                except OSError:
                    failed += 1

            self._report(f"导出: {char}", idx + 1, total)

        return {"导出": exported, "跳过": skipped, "失败": failed}

    def _make_dest_name(self, target_char: str, _orig_fname: str, variant_index: int) -> str:
        """生成排版程序可直接匹配的「字.png、字-1.png……」文件名。"""
        suffix = "" if variant_index == 0 else f"-{variant_index}"
        return f"{target_char}{suffix}.png"

    def _report(self, msg: str, current: int, total: int) -> None:
        if self._progress:
            self._progress(msg, current, total)
