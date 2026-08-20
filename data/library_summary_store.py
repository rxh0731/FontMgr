"""首页字库摘要索引的持久化存储。"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any

from utils.file_utils import atomic_write_json, safe_read_json
from utils.file_utils import resolve_library_directory


class LibrarySummaryStore:
    """保存可由深度核对重建的首页摘要，不替代字库主数据。"""

    VERSION = 1

    def __init__(self, file_path: str, library_root: str) -> None:
        self._file_path = file_path
        self._library_root = os.path.normcase(os.path.abspath(library_root))

    @staticmethod
    def signature_fingerprint(signature: object) -> str:
        value = asdict(signature) if is_dataclass(signature) else signature
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def load(self, signature: object) -> list[dict[str, Any]] | None:
        payload = safe_read_json(self._file_path, default={})
        if not isinstance(payload, dict):
            return None
        if payload.get("版本") != self.VERSION:
            return None
        if payload.get("字库根目录") != self._library_root:
            return None
        if payload.get("文件系统签名") != self.signature_fingerprint(signature):
            return None
        summaries = payload.get("字库摘要")
        if not isinstance(summaries, list):
            return None
        result: list[dict[str, Any]] = []
        for summary in summaries:
            if not isinstance(summary, dict):
                return None
            if not all(key in summary for key in ("name", "path", "variants")):
                return None
            name = str(summary.get("name", ""))
            path = resolve_library_directory(
                self._library_root,
                str(summary.get("path", "")),
                expected_name=name,
            )
            if not path:
                return None
            summary = dict(summary)
            summary["path"] = path
            result.append(dict(summary))
        return result

    def save(self, summaries: list[dict[str, Any]], signature: object) -> None:
        payload = {
            "版本": self.VERSION,
            "字库根目录": self._library_root,
            "文件系统签名": self.signature_fingerprint(signature),
            "字库摘要": summaries,
        }
        atomic_write_json(payload, self._file_path)
