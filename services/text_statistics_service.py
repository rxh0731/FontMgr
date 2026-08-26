"""文字提取、字符统计与字库缺字分析服务。"""

from __future__ import annotations

import os
import re
import threading
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageSequence

from services.glyph_service import GlyphService
from services.scripture_layout_service import build_system_glyph_index
from services.scripture_text_service import (
    MAX_EXTRACTED_CHARACTERS,
    MAX_PDF_PAGES,
    MAX_SOURCE_FILE_BYTES,
    SUPPORTED_DOCUMENT_EXTENSIONS,
    load_scripture_text,
)
from utils.file_utils import pinyin_natural_key


IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
)
FONT_IMAGE_EXTENSIONS = frozenset(
    {*IMAGE_EXTENSIONS, ".gif", ".psd", ".tga", ".ico"}
)
SUPPORTED_SOURCE_EXTENSIONS = frozenset(
    {*SUPPORTED_DOCUMENT_EXTENSIONS, *IMAGE_EXTENSIONS}
)
MAX_IMAGE_FRAMES = 10_000

ProgressCallback = Callable[[float, str], None]


@dataclass(frozen=True, slots=True)
class TextStatistics:
    total_characters: int
    chinese_characters: int
    english_words: int
    punctuation: int
    whitespace: int
    unique_chinese: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TextExtraction:
    text: str
    source_name: str
    detail: str


@dataclass(frozen=True, slots=True)
class BatchTextExtraction:
    text: str
    reports: tuple[str, ...]
    failures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MissingCharacterResult:
    missing_characters: tuple[str, ...]
    existing_characters: int
    invalid_filenames: int
    source_kind: str
    available_characters: tuple[str, ...]
    valid_variants: int = 0
    issues: tuple[str, ...] = ()


def is_chinese_character(character: str) -> bool:
    """识别统一表意文字及兼容、扩展区字符。"""

    if len(character) != 1:
        return False
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2EE5F
        or 0x2F800 <= codepoint <= 0x2FA1F
        or 0x30000 <= codepoint <= 0x323AF
    )


def clean_plain_text(text: object) -> str:
    value = str(text or "").replace("\ufeff", "").replace("\u200b", "")
    value = value.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    value = "".join(
        character
        for character in value
        if character in "\n\t" or ord(character) >= 32
    )
    return re.sub(r"\n{4,}", "\n\n\n", value).strip()


def analyze_text(text: str) -> TextStatistics:
    """按旧版口径统计正文，并输出按拼音排序的不重复汉字。"""

    unique_chinese = tuple(
        sorted(
            {character for character in text if is_chinese_character(character)},
            key=pinyin_natural_key,
        )
    )
    return TextStatistics(
        total_characters=len(text),
        chinese_characters=sum(is_chinese_character(character) for character in text),
        english_words=len(re.findall(r"[a-zA-Z]+", text)),
        punctuation=sum(
            unicodedata.category(character).startswith("P") for character in text
        ),
        whitespace=sum(character.isspace() for character in text),
        unique_chinese=unique_chinese,
    )


def source_file_filter() -> str:
    extensions = " ".join(f"*{suffix}" for suffix in sorted(SUPPORTED_SOURCE_EXTENSIONS))
    images = " ".join(f"*{suffix}" for suffix in sorted(IMAGE_EXTENSIONS))
    return (
        f"支持的文件（{extensions}）;;"
        f"图片文件（{images}）;;"
        "所有文件（*.*）"
    )


class TextStatisticsService:
    """提供可在线程中调用的文字统计业务。"""

    _ocr_engine: Any = None
    _ocr_lock = threading.Lock()

    @classmethod
    def extract_files(
        cls,
        paths: Iterable[str],
        progress: ProgressCallback | None = None,
    ) -> BatchTextExtraction:
        normalized_paths = tuple(os.path.abspath(os.fspath(path)) for path in paths)
        values: list[str] = []
        reports: list[str] = []
        failures: list[str] = []
        total = max(1, len(normalized_paths))
        for index, path in enumerate(normalized_paths):
            name = os.path.basename(path) or path
            base = index / total
            span = 1.0 / total
            if progress is not None:
                progress(base, f"正在处理：{name}")

            def item_progress(ratio: float, detail: str) -> None:
                if progress is not None:
                    progress(base + span * max(0.0, min(1.0, ratio)), f"{name}：{detail}")

            try:
                result = cls.extract_file(path, item_progress)
                if result.text.strip():
                    values.append(result.text)
                    reports.append(f"{name}：{result.detail}")
                else:
                    failures.append(f"{name}：未提取或识别到正文")
            except Exception as exc:
                failures.append(f"{name}：{exc}")
            if progress is not None:
                progress((index + 1) / total, f"已处理：{name}")
        return BatchTextExtraction(
            clean_plain_text("\n\n".join(values)),
            tuple(reports),
            tuple(failures),
        )

    @classmethod
    def extract_file(
        cls,
        path: str,
        progress: ProgressCallback | None = None,
    ) -> TextExtraction:
        normalized = os.path.abspath(os.fspath(path))
        if not os.path.isfile(normalized) or os.path.islink(normalized):
            raise ValueError("所选文件不存在，或不是可安全读取的普通文件。")
        if os.path.getsize(normalized) > MAX_SOURCE_FILE_BYTES:
            raise ValueError(
                f"文件大小超过安全上限（{MAX_SOURCE_FILE_BYTES // 1024 // 1024} MiB）。"
            )
        extension = Path(normalized).suffix.lower()
        if extension in IMAGE_EXTENSIONS:
            text = cls._recognize_image_file(normalized, progress)
            detail = "已使用本地识别引擎提取图片文字"
        elif extension == ".pdf":
            text = cls._read_pdf_with_ocr(normalized, progress)
            detail = "已提取 PDF 文本层，扫描页面已自动识别"
        else:
            result = load_scripture_text(normalized)
            text, detail = result.text, result.detail
            if progress is not None:
                progress(1.0, "文件文字提取完成")
        cleaned = clean_plain_text(text)
        if not cleaned:
            raise ValueError("文件中没有提取或识别到可用文字。")
        if len(cleaned) > MAX_EXTRACTED_CHARACTERS:
            raise ValueError(
                f"提取文字超过安全上限（{MAX_EXTRACTED_CHARACTERS:,} 个字符）。"
            )
        return TextExtraction(cleaned, os.path.basename(normalized), detail)

    @classmethod
    def recognize_image(
        cls,
        image: Image.Image,
        progress: ProgressCallback | None = None,
    ) -> str:
        if progress is not None:
            progress(0.05, "正在加载图片识别引擎")
        try:
            from rapidocr import RapidOCR
        except ImportError as exc:
            raise ValueError("图片识别引擎不可用，请安装 RapidOCR。") from exc
        with cls._ocr_lock:
            if cls._ocr_engine is None:
                cls._ocr_engine = RapidOCR()
            if progress is not None:
                progress(0.25, "正在分析图片文字")
            result = cls._ocr_engine(np.asarray(image.convert("RGB")))
        texts = list(getattr(result, "txts", None) or [])
        boxes = getattr(result, "boxes", None)
        text, layout = cls.arrange_ocr_order(texts, boxes)
        if progress is not None:
            progress(1.0, f"识别完成，已判断为{layout}")
        return clean_plain_text(text)

    @classmethod
    def recognize_clipboard_image(
        cls,
        image: Image.Image,
        progress: ProgressCallback | None = None,
    ) -> BatchTextExtraction:
        try:
            text = cls.recognize_image(image, progress)
        finally:
            image.close()
        if not text:
            return BatchTextExtraction("", (), ("剪贴板图片：未识别到正文",))
        return BatchTextExtraction(text, ("剪贴板图片：已使用本地识别引擎提取文字",), ())

    @staticmethod
    def arrange_ocr_order(texts: list[str], boxes: Any) -> tuple[str, str]:
        """依据识别框恢复横排或从右向左竖排的阅读顺序。"""

        if not texts or boxes is None or len(texts) != len(boxes):
            return "\n".join(texts or []), "横排"
        entries: list[dict[str, Any]] = []
        for text, box in zip(texts, boxes):
            coordinates = np.asarray(box, dtype=float)
            if coordinates.shape != (4, 2) or not str(text).strip():
                continue
            left, right = float(coordinates[:, 0].min()), float(coordinates[:, 0].max())
            top, bottom = float(coordinates[:, 1].min()), float(coordinates[:, 1].max())
            entries.append(
                {
                    "文字": str(text).strip(),
                    "横心": (left + right) / 2,
                    "纵心": (top + bottom) / 2,
                    "宽": max(1.0, right - left),
                    "高": max(1.0, bottom - top),
                }
            )
        if not entries:
            return "", "横排"
        median_width = float(np.median([entry["宽"] for entry in entries]))
        median_height = float(np.median([entry["高"] for entry in entries]))
        horizontal_evidence = sum(
            max(0.0, entry["宽"] / entry["高"] - 1.25) for entry in entries
        )
        vertical_evidence = sum(
            max(0.0, entry["高"] / entry["宽"] - 1.25) for entry in entries
        )
        horizontal_alignment = 0
        vertical_alignment = 0
        for index, current in enumerate(entries):
            for other in entries[index + 1 :]:
                if abs(current["纵心"] - other["纵心"]) <= median_height * 0.65:
                    horizontal_alignment += 1
                if abs(current["横心"] - other["横心"]) <= median_width * 0.65:
                    vertical_alignment += 1
        vertical = (
            vertical_evidence * 2 + vertical_alignment
            > (horizontal_evidence * 2 + horizontal_alignment) * 1.2
            and (vertical_evidence > 0 or vertical_alignment >= 2)
        )
        groups: list[list[dict[str, Any]]] = []
        grouping_key = "横心" if vertical else "纵心"
        tolerance = (median_width if vertical else median_height) * 0.7
        for current in sorted(
            entries,
            key=lambda entry: entry[grouping_key],
            reverse=vertical,
        ):
            best_group: list[dict[str, Any]] | None = None
            minimum_distance = float("inf")
            for group in groups:
                center = sum(entry[grouping_key] for entry in group) / len(group)
                distance = abs(current[grouping_key] - center)
                if distance <= tolerance and distance < minimum_distance:
                    best_group, minimum_distance = group, distance
            if best_group is None:
                groups.append([current])
            else:
                best_group.append(current)
        if vertical:
            groups.sort(
                key=lambda group: sum(entry["横心"] for entry in group) / len(group),
                reverse=True,
            )
            lines = [
                "".join(entry["文字"] for entry in sorted(group, key=lambda item: item["纵心"]))
                for group in groups
            ]
            return "\n".join(lines), "竖排（从右向左、从上到下）"
        groups.sort(key=lambda group: sum(entry["纵心"] for entry in group) / len(group))
        lines = [
            "".join(entry["文字"] for entry in sorted(group, key=lambda item: item["横心"]))
            for group in groups
        ]
        return "\n".join(lines), "横排（从上到下、从左向右）"

    @classmethod
    def analyze_system_library(
        cls,
        characters: Iterable[str],
        library_directory: str,
        progress: Callable[[object], None] | None = None,
    ) -> MissingCharacterResult:
        """按排版功能的相同规则核对系统字库真实可用成品。"""

        directory = os.path.abspath(os.fspath(library_directory))
        if not os.path.isdir(directory) or os.path.islink(directory):
            raise ValueError("系统字库目录不存在，或不是可安全读取的普通目录。")
        library_name = os.path.basename(os.path.normpath(directory))
        glyph_service = GlyphService.open(library_name, directory)
        index = build_system_glyph_index(
            glyph_service,
            progress_callback=progress,
        )
        return cls._missing_result(
            characters,
            index.characters,
            source_kind=f"本系统字库“{index.source_name}”",
            valid_variants=index.variant_count,
            issues=index.issues,
        )

    @classmethod
    def analyze_external_directory(
        cls,
        characters: Iterable[str],
        font_directory: str,
    ) -> MissingCharacterResult:
        """按旧版文件名规则只读分析外部图片目录。"""

        directory = os.path.abspath(os.fspath(font_directory))
        if not os.path.isdir(directory) or os.path.islink(directory):
            raise ValueError("字库目录不存在，或不是可安全读取的普通目录。")
        invalid = 0
        valid_variants = 0
        existing: set[str] = set()
        for root, directory_names, filenames in os.walk(directory, followlinks=False):
            directory_names[:] = [
                name
                for name in directory_names
                if not os.path.islink(os.path.join(root, name))
            ]
            for filename in filenames:
                if Path(filename).suffix.lower() not in FONT_IMAGE_EXTENSIONS:
                    continue
                stem = Path(filename).stem
                candidate = stem.rsplit("-", 1)[0] if "-" in stem else stem
                if is_chinese_character(candidate):
                    existing.add(candidate)
                    valid_variants += 1
                else:
                    invalid += 1
        return cls._missing_result(
            characters,
            existing,
            source_kind="外部图片目录",
            invalid_filenames=invalid,
            valid_variants=valid_variants,
        )

    @classmethod
    def analyze_missing(
        cls,
        characters: Iterable[str],
        font_directory: str,
    ) -> MissingCharacterResult:
        """保留旧调用名称；目录模式固定按外部图片规则分析。"""

        return cls.analyze_external_directory(characters, font_directory)

    @staticmethod
    def _missing_result(
        characters: Iterable[str],
        available_characters: Iterable[str],
        *,
        source_kind: str,
        invalid_filenames: int = 0,
        valid_variants: int = 0,
        issues: Iterable[str] = (),
    ) -> MissingCharacterResult:
        existing = {
            str(character)
            for character in available_characters
            if is_chinese_character(str(character))
        }
        requested = {character for character in characters if is_chinese_character(character)}
        missing = tuple(sorted(requested - existing, key=pinyin_natural_key))
        return MissingCharacterResult(
            missing,
            len(existing),
            int(invalid_filenames),
            source_kind,
            tuple(sorted(existing, key=pinyin_natural_key)),
            int(valid_variants),
            tuple(str(issue) for issue in issues),
        )

    @classmethod
    def _recognize_image_file(
        cls,
        path: str,
        progress: ProgressCallback | None,
    ) -> str:
        texts: list[str] = []
        with Image.open(path) as image:
            frame_count = int(getattr(image, "n_frames", 1) or 1)
            if frame_count > MAX_IMAGE_FRAMES:
                raise ValueError(f"图片页数超过安全上限（{MAX_IMAGE_FRAMES:,} 页）。")
            for index, frame in enumerate(ImageSequence.Iterator(image)):
                copied = frame.copy()
                try:
                    def frame_progress(ratio: float, detail: str) -> None:
                        if progress is not None:
                            progress(
                                (index + ratio) / max(1, frame_count),
                                f"第 {index + 1}/{frame_count} 页：{detail}",
                            )

                    text = cls.recognize_image(copied, frame_progress)
                    if text:
                        texts.append(text)
                finally:
                    copied.close()
        return clean_plain_text("\n\n".join(texts))

    @classmethod
    def _read_pdf_with_ocr(
        cls,
        path: str,
        progress: ProgressCallback | None,
    ) -> str:
        import fitz

        document = fitz.open(path)
        texts: list[str] = []
        try:
            page_count = int(document.page_count)
            if page_count > MAX_PDF_PAGES:
                raise ValueError(f"PDF 页数超过安全上限（{MAX_PDF_PAGES:,} 页）。")
            for index, page in enumerate(document):
                if progress is not None:
                    progress(index / max(1, page_count), f"正在读取第 {index + 1}/{page_count} 页")
                page_text = page.get_text("text", sort=True)
                if sum(is_chinese_character(character) for character in page_text) < 5:
                    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                    image = Image.frombytes(
                        "RGB",
                        (pixmap.width, pixmap.height),
                        pixmap.samples,
                    )
                    try:
                        def page_progress(ratio: float, detail: str) -> None:
                            if progress is not None:
                                progress(
                                    (index + ratio) / max(1, page_count),
                                    f"第 {index + 1}/{page_count} 页：{detail}",
                                )

                        page_text = cls.recognize_image(image, page_progress)
                    finally:
                        image.close()
                if page_text.strip():
                    texts.append(page_text)
                if sum(len(value) for value in texts) > MAX_EXTRACTED_CHARACTERS:
                    raise ValueError("PDF 提取文字超过安全上限。")
        finally:
            document.close()
        if progress is not None:
            progress(1.0, "PDF 处理完成")
        text = clean_plain_text("\n\n".join(texts))
        if not text:
            raise ValueError("PDF 文本层和扫描页面均未识别出文字。")
        return text
