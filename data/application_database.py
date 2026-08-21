"""程序级 SQLite 数据库连接与小型结构化文档存储。"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import config


APP_DATABASE_VERSION = 1


def resolve_application_database_path(source_path: str | None = None) -> str:
    """返回程序数据库路径；测试传入临时配置文件时使用其所在目录。"""
    if source_path:
        source_parent = os.path.normcase(os.path.abspath(os.path.dirname(source_path)))
        config_parent = os.path.normcase(os.path.abspath(config.CONFIG_DIR))
        if source_parent != config_parent:
            source = Path(os.path.abspath(source_path))
            return str(source.with_suffix(".sqlite3"))
    return config.APP_DATABASE_FILE


class ApplicationDatabase:
    """管理用户设置、算法、模板和可重建的字库摘要。"""

    def __init__(self, path: str | None = None) -> None:
        self.path = os.path.abspath(path or config.APP_DATABASE_FILE)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _initialize(self) -> None:
        with self.transaction() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS application_meta (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    schema_version INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS application_documents (
                    document_key TEXT PRIMARY KEY,
                    data_version INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            now = self._now()
            connection.execute(
                """
                INSERT INTO application_meta(id, schema_version, created_at, updated_at)
                VALUES (1, ?, ?, ?)
                ON CONFLICT(id) DO NOTHING
                """,
                (APP_DATABASE_VERSION, now, now),
            )
            row = connection.execute(
                "SELECT schema_version FROM application_meta WHERE id = 1"
            ).fetchone()
            if row is None or int(row[0]) != APP_DATABASE_VERSION:
                raise RuntimeError("程序数据库版本不受支持。")

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

    def read_document(self, key: str) -> Any | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT payload_json FROM application_documents WHERE document_key = ?",
                (key,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        try:
            return json.loads(str(row[0]))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"程序数据库中的“{key}”记录损坏。") from exc

    def write_document(self, key: str, payload: Any, *, version: int = 1) -> None:
        packed = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO application_documents(
                    document_key, data_version, payload_json, updated_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(document_key) DO UPDATE SET
                    data_version = excluded.data_version,
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (key, int(version), packed, self._now()),
            )

    @staticmethod
    def _now() -> str:
        return datetime.now().astimezone().isoformat(timespec="seconds")
