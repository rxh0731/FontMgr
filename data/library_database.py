"""单字库 SQLite 数据库、增量提交与旧主数据导入。"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


LIBRARY_DATABASE_FILENAME = "font_library.sqlite3"
LIBRARY_DATABASE_VERSION = 1

_STAGE_FIELDS = {
    "原图": ("原始文件", "原始MD5"),
    "灰度母版": ("灰度母版文件", "灰度母版MD5"),
    "清洁掩码": ("清洁掩码文件", "清洁掩码MD5"),
    "优化预览": ("中间文件", "中间MD5"),
    "手工审核": ("审核文件", "审核MD5"),
    "成品": ("成品文件", "成品MD5"),
}


def library_database_path(library_dir: str) -> str:
    return os.path.join(os.path.abspath(library_dir), LIBRARY_DATABASE_FILENAME)


def _pack(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _unpack(value: object, label: str) -> Any:
    try:
        return json.loads(str(value))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"字库数据库中的{label}损坏。") from exc


class LibraryDatabase:
    """为一个字库提供短连接、事务化、可跨线程使用的数据访问。"""

    def __init__(self, path: str) -> None:
        self.path = os.path.abspath(path)

    @classmethod
    def open(cls, library_dir: str) -> "LibraryDatabase":
        path = library_database_path(library_dir)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"字库数据库不存在：{path}")
        database = cls(path)
        database.verify()
        return database

    @classmethod
    def install_from_data(
        cls,
        library_dir: str,
        data: Mapping[str, Any],
        *,
        source_path: str = "",
    ) -> "LibraryDatabase":
        """将完整状态写入临时数据库，核对成功后原子安装。"""
        root = os.path.abspath(library_dir)
        Path(root).mkdir(parents=True, exist_ok=True)
        final_path = library_database_path(root)
        if os.path.isfile(final_path):
            return cls.open(root)
        token = uuid.uuid4().hex
        temporary_path = os.path.join(root, f".{LIBRARY_DATABASE_FILENAME}.{token}.tmp")
        temporary = cls(temporary_path)
        try:
            temporary._initialize()
            temporary.replace_all(data, source_path=source_path)
            loaded = temporary.load_data()
            if loaded != dict(data):
                raise RuntimeError("旧字库导入核对失败：数据库重建结果与来源数据不一致。")
            temporary.verify()
            temporary.checkpoint()
            os.replace(temporary_path, final_path)
        except Exception:
            for suffix in ("", "-wal", "-shm"):
                try:
                    os.remove(temporary_path + suffix)
                except FileNotFoundError:
                    pass
            raise
        return cls.open(root)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self.transaction() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS library (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    schema_version INTEGER NOT NULL,
                    data_version INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    dpi INTEGER NOT NULL,
                    canvas_width INTEGER NOT NULL,
                    canvas_height INTEGER NOT NULL,
                    finished_width_mm REAL NOT NULL,
                    finished_height_mm REAL NOT NULL,
                    metadata_json TEXT NOT NULL,
                    session_json TEXT NOT NULL,
                    coordination_json TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS glyph_variants (
                    variant_id TEXT PRIMARY KEY,
                    glyph_char TEXT NOT NULL,
                    variant_order INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    original_md5 TEXT NOT NULL,
                    source_filename TEXT NOT NULL,
                    image_info_json TEXT NOT NULL,
                    notes TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    UNIQUE(glyph_char, variant_order)
                );
                CREATE INDEX IF NOT EXISTS idx_glyph_variants_status
                    ON glyph_variants(status, glyph_char, variant_order);
                CREATE TABLE IF NOT EXISTS variant_stage_files (
                    variant_id TEXT NOT NULL REFERENCES glyph_variants(variant_id) ON DELETE CASCADE,
                    stage TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    content_md5 TEXT NOT NULL,
                    PRIMARY KEY(variant_id, stage)
                );
                CREATE INDEX IF NOT EXISTS idx_variant_stage_path
                    ON variant_stage_files(stage, relative_path);
                CREATE TABLE IF NOT EXISTS optimization_results (
                    variant_id TEXT PRIMARY KEY REFERENCES glyph_variants(variant_id) ON DELETE CASCADE,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS manual_edits (
                    variant_id TEXT PRIMARY KEY REFERENCES glyph_variants(variant_id) ON DELETE CASCADE,
                    edit_json TEXT NOT NULL,
                    transform_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS coordination_results (
                    variant_id TEXT PRIMARY KEY REFERENCES glyph_variants(variant_id) ON DELETE CASCADE,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS library_summary (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    revision INTEGER NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS committed_operations (
                    operation_id TEXT PRIMARY KEY,
                    revision INTEGER NOT NULL,
                    committed_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS migration_history (
                    migration_id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL,
                    variant_count INTEGER NOT NULL,
                    migrated_at TEXT NOT NULL
                );
                """
            )

    def replace_all(self, data: Mapping[str, Any], *, source_path: str = "") -> None:
        self._validate_data(data)
        with self.transaction() as connection:
            connection.execute("DELETE FROM glyph_variants")
            self._write_library(connection, data, revision=1)
            self._write_variants(connection, data, data["变体详情"].keys())
            source_hash = self._source_hash(source_path, data)
            connection.execute(
                """
                INSERT OR REPLACE INTO migration_history(
                    migration_id, source_path, source_sha256, variant_count, migrated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    f"初始导入:{source_hash}",
                    os.path.abspath(source_path) if source_path else "",
                    source_hash,
                    len(data["变体详情"]),
                    self._now(),
                ),
            )

    def save_data(
        self,
        data: Mapping[str, Any],
        *,
        dirty_variant_ids: Iterable[str],
        deleted_variant_ids: Iterable[str] = (),
    ) -> int:
        dirty = tuple(dict.fromkeys(str(item) for item in dirty_variant_ids))
        deleted = tuple(dict.fromkeys(str(item) for item in deleted_variant_ids))
        self._validate_incremental_data(data, dirty)
        with self.transaction() as connection:
            row = connection.execute("SELECT revision FROM library WHERE id = 1").fetchone()
            revision = int(row[0]) + 1 if row else 1
            for variant_id in deleted:
                connection.execute(
                    "DELETE FROM glyph_variants WHERE variant_id = ?",
                    (variant_id,),
                )
            self._write_variants(connection, data, dirty)
            self._write_library(connection, data, revision=revision)
        return revision

    def save_full_data(self, data: Mapping[str, Any]) -> int:
        """完整替换字库实体，用于启动时恢复旧事务。"""
        self._validate_data(data)
        with self.transaction() as connection:
            row = connection.execute("SELECT revision FROM library WHERE id = 1").fetchone()
            revision = int(row[0]) + 1 if row else 1
            connection.execute("DELETE FROM glyph_variants")
            self._write_variants(connection, data, data["变体详情"].keys())
            self._write_library(connection, data, revision=revision)
        return revision

    def load_data(self) -> dict[str, Any]:
        connection = self._connect()
        try:
            library = connection.execute("SELECT * FROM library WHERE id = 1").fetchone()
            if library is None:
                raise RuntimeError("字库数据库缺少字库参数。")
            metadata = _unpack(library["metadata_json"], "字库参数")
            session = _unpack(library["session_json"], "会话状态")
            coordination = _unpack(library["coordination_json"], "整体协调状态")
            details: dict[str, dict[str, Any]] = {}
            groups: dict[str, list[str]] = {}
            rows = connection.execute(
                """
                SELECT variant_id, glyph_char, payload_json
                FROM glyph_variants
                ORDER BY glyph_char, variant_order
                """
            ).fetchall()
            for row in rows:
                variant_id = str(row["variant_id"])
                detail = _unpack(row["payload_json"], f"字形“{variant_id}”")
                if not isinstance(detail, dict):
                    raise RuntimeError(f"字形“{variant_id}”记录格式无效。")
                details[variant_id] = detail
                groups.setdefault(str(row["glyph_char"]), []).append(variant_id)
            return {
                "数据版本": int(library["data_version"]),
                "库名": str(library["name"]),
                "元数据": metadata,
                "会话": session,
                "字形组索引": groups,
                "变体详情": details,
                "整体协调": coordination,
            }
        finally:
            connection.close()

    def save_summary(self, summary: Mapping[str, Any]) -> None:
        with self.transaction() as connection:
            row = connection.execute("SELECT revision FROM library WHERE id = 1").fetchone()
            revision = int(row[0]) if row else 0
            connection.execute(
                """
                INSERT INTO library_summary(id, revision, payload_json)
                VALUES (1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    revision = excluded.revision,
                    payload_json = excluded.payload_json
                """,
                (revision, _pack(dict(summary))),
            )

    def verify(self) -> None:
        connection = self._connect()
        try:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            if integrity is None or str(integrity[0]).lower() != "ok":
                raise RuntimeError("字库数据库完整性检查失败。")
            foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
            if foreign_keys:
                raise RuntimeError("字库数据库存在无效关联记录。")
            version = connection.execute("PRAGMA user_version").fetchone()
            del version
            library = connection.execute(
                "SELECT schema_version FROM library WHERE id = 1"
            ).fetchone()
            if library is None or int(library[0]) != LIBRARY_DATABASE_VERSION:
                raise RuntimeError("字库数据库版本不受支持。")
        finally:
            connection.close()

    def checkpoint(self) -> None:
        connection = self._connect()
        try:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        finally:
            connection.close()

    def _write_library(
        self,
        connection: sqlite3.Connection,
        data: Mapping[str, Any],
        *,
        revision: int,
    ) -> None:
        metadata = data["元数据"]
        now = self._now()
        connection.execute(
            """
            INSERT INTO library(
                id, schema_version, data_version, name, dpi, canvas_width,
                canvas_height, finished_width_mm, finished_height_mm,
                metadata_json, session_json, coordination_json, revision,
                created_at, updated_at
            ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                schema_version = excluded.schema_version,
                data_version = excluded.data_version,
                name = excluded.name,
                dpi = excluded.dpi,
                canvas_width = excluded.canvas_width,
                canvas_height = excluded.canvas_height,
                finished_width_mm = excluded.finished_width_mm,
                finished_height_mm = excluded.finished_height_mm,
                metadata_json = excluded.metadata_json,
                session_json = excluded.session_json,
                coordination_json = excluded.coordination_json,
                revision = excluded.revision,
                updated_at = excluded.updated_at
            """,
            (
                LIBRARY_DATABASE_VERSION,
                int(data["数据版本"]),
                str(data["库名"]),
                int(metadata.get("DPI", 300)),
                int(metadata.get("画布宽", 250)),
                int(metadata.get("画布高", 250)),
                float(metadata.get("成品宽度毫米", 0.0)),
                float(metadata.get("成品高度毫米", 0.0)),
                _pack(metadata),
                _pack(data.get("会话", {})),
                _pack(data.get("整体协调", {})),
                revision,
                str(metadata.get("创建时间") or now),
                str(metadata.get("最后修改") or now),
            ),
        )

    def _write_variants(
        self,
        connection: sqlite3.Connection,
        data: Mapping[str, Any],
        variant_ids: Iterable[str],
    ) -> None:
        details = data["变体详情"]
        groups = data["字形组索引"]
        requested_ids = tuple(variant_ids)
        order_map: dict[str, tuple[str, int]] = {}
        if len(requested_ids) > 32:
            order_map = {
                str(variant_id): (str(char), order)
                for char, items in groups.items()
                for order, variant_id in enumerate(items, start=1)
            }
        for variant_id in requested_ids:
            detail = details.get(variant_id)
            if not isinstance(detail, Mapping):
                continue
            glyph_char = str(detail.get("归属字", ""))
            if variant_id in order_map:
                glyph_char, order = order_map[variant_id]
            else:
                group = groups.get(glyph_char, [])
                try:
                    order = group.index(variant_id) + 1
                except ValueError:
                    order = int(detail.get("变体序号", 0) or 0)
            connection.execute(
                """
                INSERT INTO glyph_variants(
                    variant_id, glyph_char, variant_order, status,
                    original_filename, original_md5, source_filename,
                    image_info_json, notes, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(variant_id) DO UPDATE SET
                    glyph_char = excluded.glyph_char,
                    variant_order = excluded.variant_order,
                    status = excluded.status,
                    original_filename = excluded.original_filename,
                    original_md5 = excluded.original_md5,
                    source_filename = excluded.source_filename,
                    image_info_json = excluded.image_info_json,
                    notes = excluded.notes,
                    payload_json = excluded.payload_json
                """,
                (
                    variant_id,
                    glyph_char,
                    order,
                    str(detail.get("状态", "")),
                    str(detail.get("原始文件", "")),
                    str(detail.get("原始MD5", "")),
                    str(detail.get("导入前文件名", "")),
                    _pack(detail.get("图像信息", {})),
                    str(detail.get("备注", "")),
                    _pack(dict(detail)),
                ),
            )
            connection.execute("DELETE FROM variant_stage_files WHERE variant_id = ?", (variant_id,))
            for stage, (file_field, md5_field) in _STAGE_FIELDS.items():
                filename = str(detail.get(file_field, ""))
                checksum = str(detail.get(md5_field, ""))
                if filename:
                    connection.execute(
                        """
                        INSERT INTO variant_stage_files(
                            variant_id, stage, relative_path, content_md5
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (variant_id, stage, filename, checksum),
                    )
            connection.execute(
                "INSERT OR REPLACE INTO optimization_results VALUES (?, ?)",
                (variant_id, _pack(detail.get("自动优化", {}))),
            )
            connection.execute(
                "INSERT OR REPLACE INTO manual_edits VALUES (?, ?, ?)",
                (
                    variant_id,
                    _pack(detail.get("手工编辑", {})),
                    _pack(detail.get("变换参数", {})),
                ),
            )
            connection.execute(
                "INSERT OR REPLACE INTO coordination_results VALUES (?, ?)",
                (variant_id, _pack(detail.get("整体协调参数", {}))),
            )

    @staticmethod
    def _validate_data(data: Mapping[str, Any]) -> None:
        for key, expected in (
            ("元数据", Mapping),
            ("字形组索引", Mapping),
            ("变体详情", Mapping),
            ("整体协调", Mapping),
        ):
            if not isinstance(data.get(key), expected):
                raise RuntimeError(f"字库数据缺少有效的{key}。")
        details = data["变体详情"]
        seen: set[str] = set()
        for char, variant_ids in data["字形组索引"].items():
            if not isinstance(variant_ids, list):
                raise RuntimeError(f"字形“{char}”的分组索引无效。")
            for variant_id in variant_ids:
                key = str(variant_id)
                if key in seen or key not in details:
                    raise RuntimeError(f"字形分组索引包含重复或悬空记录：{key}")
                detail = details[key]
                if not isinstance(detail, Mapping) or str(detail.get("归属字", "")) != str(char):
                    raise RuntimeError(f"字形分组索引与记录归属不一致：{key}")
                seen.add(key)
        if seen != {str(item) for item in details}:
            raise RuntimeError("字库存在未加入字形组的变体记录。")

    @staticmethod
    def _validate_incremental_data(
        data: Mapping[str, Any],
        variant_ids: Iterable[str],
    ) -> None:
        for key in ("元数据", "字形组索引", "变体详情", "整体协调"):
            if not isinstance(data.get(key), Mapping):
                raise RuntimeError(f"字库数据缺少有效的{key}。")
        details = data["变体详情"]
        groups = data["字形组索引"]
        for variant_id in variant_ids:
            detail = details.get(variant_id)
            if not isinstance(detail, Mapping):
                raise RuntimeError(f"待保存的字形记录无效：{variant_id}")
            char = str(detail.get("归属字", ""))
            members = groups.get(char)
            if not isinstance(members, list) or variant_id not in members:
                raise RuntimeError(f"字形分组索引与待保存记录不一致：{variant_id}")

    @staticmethod
    def _source_hash(source_path: str, data: Mapping[str, Any]) -> str:
        digest = hashlib.sha256()
        if source_path and os.path.isfile(source_path):
            with open(source_path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        else:
            digest.update(_pack(dict(data)).encode("utf-8"))
        return digest.hexdigest()

    @staticmethod
    def _now() -> str:
        return datetime.now().astimezone().isoformat(timespec="seconds")
