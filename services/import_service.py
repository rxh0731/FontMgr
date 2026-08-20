# import_service.py — 原始文件无损导入服务

import os
import shutil
import hashlib
import tempfile
from typing import Any, Callable, Optional

from PIL import Image

from data.log_manager import write_log
from services.glyph_service import GlyphService
from utils.file_utils import (
    compute_file_md5,
    is_cjk_character,
    is_real_directory,
    natural_key,
)


class ImportService:
    """只负责扫描、校验、标准命名、无损复制和原始文件登记。"""

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".jpe", ".jfif", ".bmp", ".dib", ".tif", ".tiff", ".webp", ".gif", ".tga", ".ppm", ".pgm", ".pbm", ".pnm", ".ico"}

    def __init__(
        self,
        glyph_service: GlyphService,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        cancel_callback: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._glyph = glyph_service
        self._progress = progress_callback
        self._cancel = cancel_callback
        self._md5_index: dict[str, dict[str, Any]] = {}
        self._used_stems: set[str] = set()
        self._next_char_index: dict[str, int] = {}

    def import_batch(
        self,
        source_dir: str,
        dpi: int = 300,
        canvas_w: int = 250,
        canvas_h: int = 250,
        force: bool = False,
        char_overrides: Optional[dict[str, str]] = None,
        width_mm: Optional[float] = None,
        height_mm: Optional[float] = None,
        init_meta: bool = True,
        scanned_files: Optional[dict[str, dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """无损导入目录中的图片；可按扫描确认结果覆盖每个文件的归属字符。"""
        if init_meta:
            self._glyph.init_metadata(dpi, canvas_w, canvas_h, width_mm, height_mm)
        self._glyph.ensure_dirs()
        original_dir, _, _ = self._glyph.get_three_dirs()
        self._initialize_indexes(original_dir)
        files = self._scan_source_dir(source_dir)
        if not files:
            return {"成功": 0, "跳过": 0, "失败": 0, "详情": []}

        details: list[dict[str, Any]] = []
        success_count = skipped_count = failed_count = 0
        total = len(files)
        for index, source_path in enumerate(files, 1):
            if self._cancel and self._cancel():
                self._glyph.save()
                return {
                    "成功": success_count,
                    "跳过": skipped_count,
                    "失败": failed_count,
                    "详情": details,
                    "已取消": True,
                }
            source_filename = os.path.basename(source_path)
            target_char = self._extract_char(source_filename)
            if char_overrides:
                target_char = char_overrides.get(os.path.abspath(source_path), target_char)
            record: dict[str, Any] = {"路径": source_path, "归属字": target_char}
            target_path = ""
            variant_id = ""
            try:
                scan_record = (scanned_files or {}).get(os.path.abspath(source_path), {})
                digest = self._reusable_scanned_digest(source_path, scan_record)
                if not digest:
                    digest = compute_file_md5(source_path)
                existing = self._md5_index.get(digest)
                if existing and not force:
                    skipped_count += 1
                    record.update({"状态": "跳过", "原因": "文件内容已经导入", "原始MD5": digest})
                    details.append(record)
                    self._report(f"跳过重复文件：{source_filename}", index, total)
                    continue

                with Image.open(source_path) as image:
                    image.load()
                    dpi_info = image.info.get("dpi", (300.0, 300.0))
                    if isinstance(dpi_info, (int, float)):
                        dpi_x = dpi_y = float(dpi_info)
                    elif isinstance(dpi_info, (tuple, list)) and dpi_info:
                        dpi_x = float(dpi_info[0] or 300.0)
                        dpi_y = float(dpi_info[1] if len(dpi_info) > 1 and dpi_info[1] else dpi_x)
                    else:
                        dpi_x = dpi_y = 300.0
                    dpi_x = dpi_x if 1.0 <= dpi_x <= 9600.0 else 300.0
                    dpi_y = dpi_y if 1.0 <= dpi_y <= 9600.0 else 300.0
                    image_info = {
                        "宽": image.width,
                        "高": image.height,
                        "格式": image.format or "未知",
                        "模式": image.mode,
                        "水平DPI": round(dpi_x, 4),
                        "垂直DPI": round(dpi_y, 4),
                        "物理宽度毫米": round(image.width / dpi_x * 25.4, 4),
                        "物理高度毫米": round(image.height / dpi_y * 25.4, 4),
                    }

                standard_stem = self._alloc_filename(target_char)
                extension = os.path.splitext(source_path)[1].lower() or ".png"
                standard_filename = standard_stem + extension
                target_path = os.path.join(original_dir, standard_filename)
                copied_digest = self._copy_atomic_with_md5(source_path, target_path)
                if copied_digest != digest:
                    os.remove(target_path)
                    raise OSError("复制后的文件摘要与源文件不一致")

                variant_id = self._glyph.add_original(target_char, standard_filename, source_filename, digest, image_info)
                self._md5_index.setdefault(digest, self._glyph.get_variant(variant_id))
                success_count += 1
                record.update({"状态": "成功", "原始文件": standard_filename, "原始MD5": digest, "变体ID": variant_id})
                self._report(f"已归档：{standard_filename}", index, total)
            except Exception as exc:
                if variant_id:
                    self._glyph.discard_unsaved_original(variant_id)
                    indexed = self._md5_index.get(digest)
                    if isinstance(indexed, dict) and indexed.get("变体ID") == variant_id:
                        self._md5_index.pop(digest, None)
                if target_path and os.path.isfile(target_path):
                    try:
                        os.remove(target_path)
                    except OSError as cleanup_exc:
                        record["清理错误"] = str(cleanup_exc)
                failed_count += 1
                record.update({"状态": "失败", "错误": str(exc)})
                self._report(f"导入失败：{source_filename}", index, total)
            details.append(record)

        self._glyph.save()
        return {"成功": success_count, "跳过": skipped_count, "失败": failed_count, "详情": details}

    def _scan_source_dir(self, source_dir: str) -> list[str]:
        if not is_real_directory(source_dir):
            raise ValueError("源图片目录不存在，或目录是链接/联接。")
        files = []
        for entry in sorted(os.scandir(source_dir), key=lambda entry: natural_key(entry.name)):
            if entry.is_file(follow_symlinks=False) and os.path.splitext(entry.name)[1].lower() in self.SUPPORTED_EXTENSIONS:
                files.append(entry.path)
        return files

    def _extract_char(self, filename: str) -> str:
        stem = os.path.splitext(filename)[0]
        if stem == "未分类" or stem.startswith("未分类-"):
            return "未分类"
        for char in stem:
            if is_cjk_character(char):
                return char
        return "未分类"

    def _alloc_filename(self, char: str) -> str:
        index = self._next_char_index.get(char, 1)
        while f"{char}-{index:04d}" in self._used_stems:
            index += 1
        stem = f"{char}-{index:04d}"
        self._used_stems.add(stem)
        self._next_char_index[char] = index + 1
        return stem

    def _initialize_indexes(self, original_dir: str) -> None:
        variants = self._glyph.get_all_variants()
        self._md5_index = {
            str(detail.get("原始MD5")): detail
            for detail in variants
            if detail.get("原始MD5")
        }
        self._used_stems = {
            os.path.splitext(str(detail.get("原始文件", "")))[0]
            for detail in variants
            if detail.get("原始文件")
        }
        try:
            self._used_stems.update(
                os.path.splitext(entry.name)[0]
                for entry in os.scandir(original_dir)
                if entry.is_file(follow_symlinks=False)
            )
        except OSError:
            pass
        self._next_char_index.clear()

    @staticmethod
    def _reusable_scanned_digest(
        source_path: str,
        record: dict[str, Any],
    ) -> str:
        digest = str(record.get("md5", ""))
        if not digest:
            return ""
        try:
            source_stat = os.stat(source_path, follow_symlinks=False)
        except OSError:
            return ""
        if (
            int(record.get("size", -1)) != source_stat.st_size
            or int(record.get("mtime_ns", -1)) != source_stat.st_mtime_ns
        ):
            return ""
        return digest

    @staticmethod
    def _copy_atomic_with_md5(source_path: str, target_path: str) -> str:
        if os.path.exists(target_path):
            raise FileExistsError(f"目标文件已经存在：{os.path.basename(target_path)}")
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=".import_",
            suffix=".tmp",
            dir=os.path.dirname(target_path),
        )
        digest = hashlib.md5()
        try:
            with open(source_path, "rb") as source, os.fdopen(descriptor, "wb") as target:
                descriptor = -1
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
            shutil.copystat(source_path, temporary_path, follow_symlinks=False)
            try:
                os.link(temporary_path, target_path)
            except FileExistsError:
                raise FileExistsError(
                    f"目标文件已经存在：{os.path.basename(target_path)}"
                ) from None
            os.remove(temporary_path)
            return digest.hexdigest()
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if os.path.exists(temporary_path):
                try:
                    os.remove(temporary_path)
                except OSError:
                    pass

    def _report(self, message: str, current: int, total: int) -> None:
        if self._progress:
            try:
                self._progress(message, current, total)
            except Exception as exc:
                write_log(f"导入进度通知失败｜{type(exc).__name__}: {exc}")
