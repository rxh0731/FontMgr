"""图片实验室独立项目文件存储。"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from core.image_cleanup import IMAGE_CLEANUP_ALGORITHM_VERSION, ImageCleanupOptions


IMAGE_LAB_PROJECT_EXTENSION = ".fontlab"
IMAGE_LAB_SCHEMA_VERSION = 4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class ImageLabStroke:
    """一条按原图像素坐标记录的人工清理笔画。"""

    tool: str
    width: float
    points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if self.tool not in {"cover", "restore"}:
            raise ValueError("人工笔画工具必须是覆盖或还原。")
        if not 0.5 <= float(self.width) <= 4096.0:
            raise ValueError("人工笔画宽度超出有效范围。")
        if not self.points:
            raise ValueError("人工笔画至少需要一个坐标点。")
        normalized: list[tuple[float, float]] = []
        for point in self.points:
            if len(point) != 2:
                raise ValueError("人工笔画坐标格式无效。")
            x, y = float(point[0]), float(point[1])
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError("人工笔画坐标必须位于图片范围内。")
            normalized.append((x, y))
        object.__setattr__(self, "width", float(self.width))
        object.__setattr__(self, "points", tuple(normalized))


@dataclass(slots=True)
class ImageLabProject:
    """图片实验室项目的完整可编辑状态。"""

    source_path: str
    source_width: int
    source_height: int
    source_mode: str
    source_mtime_ns: int
    source_size: int
    source_dpi_x: float = 0.0
    source_dpi_y: float = 0.0
    options: ImageCleanupOptions = field(default_factory=ImageCleanupOptions)
    strokes: list[ImageLabStroke] = field(default_factory=list)
    algorithm_version: int = IMAGE_CLEANUP_ALGORITHM_VERSION
    resolved_profile: str = ""
    project_path: str = ""
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)

    @property
    def display_name(self) -> str:
        return Path(self.source_path).stem


class ImageLabProjectStore:
    """用单个 SQLite 文件保存图片实验室项目。"""

    def create(
        self,
        source_path: str,
        *,
        width: int,
        height: int,
        mode: str,
        dpi_x: float = 0.0,
        dpi_y: float = 0.0,
    ) -> ImageLabProject:
        source = os.path.abspath(os.fspath(source_path))
        source_stat = os.stat(source)
        if not os.path.isfile(source):
            raise ValueError("原稿路径不是有效文件。")
        return ImageLabProject(
            source_path=source,
            source_width=int(width),
            source_height=int(height),
            source_mode=str(mode),
            source_mtime_ns=int(source_stat.st_mtime_ns),
            source_size=int(source_stat.st_size),
            source_dpi_x=max(0.0, float(dpi_x)),
            source_dpi_y=max(0.0, float(dpi_y)),
        )

    def save(self, project: ImageLabProject, path: str | None = None) -> str:
        target = os.path.abspath(os.fspath(path or project.project_path))
        if not target:
            raise ValueError("请先指定图片实验室项目文件。")
        if not target.lower().endswith(IMAGE_LAB_PROJECT_EXTENSION):
            target += IMAGE_LAB_PROJECT_EXTENSION
        os.makedirs(os.path.dirname(target), exist_ok=True)
        connection = sqlite3.connect(target)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            with connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS project_meta (
                        key TEXT PRIMARY KEY NOT NULL,
                        value TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS strokes (
                        sequence INTEGER PRIMARY KEY NOT NULL,
                        tool TEXT NOT NULL CHECK(tool IN ('cover', 'restore')),
                        width REAL NOT NULL,
                        points_json TEXT NOT NULL
                    );
                    """
                )
                updated_at = _utc_now()
                metadata = {
                    "schema_version": str(IMAGE_LAB_SCHEMA_VERSION),
                    "source_path": project.source_path,
                    "source_width": str(project.source_width),
                    "source_height": str(project.source_height),
                    "source_mode": project.source_mode,
                    "source_mtime_ns": str(project.source_mtime_ns),
                    "source_size": str(project.source_size),
                    "source_dpi_x": str(project.source_dpi_x),
                    "source_dpi_y": str(project.source_dpi_y),
                    "strength": str(project.options.strength),
                    "preserve_faint_ink": (
                        "1" if project.options.preserve_faint_ink else "0"
                    ),
                    "remove_small_noise": (
                        "1" if project.options.remove_small_noise else "0"
                    ),
                    "feather_edges": (
                        "1" if project.options.feather_edges else "0"
                    ),
                    "processing_mode": project.options.processing_mode,
                    "algorithm_version": str(IMAGE_CLEANUP_ALGORITHM_VERSION),
                    "resolved_profile": project.resolved_profile,
                    "created_at": project.created_at,
                    "updated_at": updated_at,
                }
                connection.executemany(
                    "INSERT OR REPLACE INTO project_meta(key, value) VALUES (?, ?)",
                    metadata.items(),
                )
                connection.execute("DELETE FROM strokes")
                connection.executemany(
                    """
                    INSERT INTO strokes(sequence, tool, width, points_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        (
                            index,
                            stroke.tool,
                            stroke.width,
                            json.dumps(stroke.points, ensure_ascii=False),
                        )
                        for index, stroke in enumerate(project.strokes)
                    ),
                )
            project.project_path = target
            project.updated_at = updated_at
            project.algorithm_version = IMAGE_CLEANUP_ALGORITHM_VERSION
            return target
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def load(self, path: str) -> ImageLabProject:
        target = os.path.abspath(os.fspath(path))
        if not os.path.isfile(target):
            raise FileNotFoundError("图片实验室项目文件不存在。")
        connection = sqlite3.connect(f"file:{Path(target).as_posix()}?mode=ro", uri=True)
        try:
            metadata = dict(
                connection.execute("SELECT key, value FROM project_meta").fetchall()
            )
            version = int(metadata.get("schema_version", "0"))
            if version not in {2, 3, IMAGE_LAB_SCHEMA_VERSION}:
                raise ValueError(f"不支持的图片实验室项目版本：{version}。")
            strokes = [
                ImageLabStroke(
                    tool=str(row[0]),
                    width=float(row[1]),
                    points=tuple(tuple(point) for point in json.loads(str(row[2]))),
                )
                for row in connection.execute(
                    "SELECT tool, width, points_json FROM strokes ORDER BY sequence"
                )
            ]
        except sqlite3.DatabaseError as exc:
            raise ValueError(f"图片实验室项目文件已损坏：{exc}") from exc
        finally:
            connection.close()
        source_path = metadata.get("source_path", "")
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"项目引用的原稿不存在：{source_path}")
        source_stat = os.stat(source_path)
        project = ImageLabProject(
            source_path=source_path,
            source_width=int(metadata["source_width"]),
            source_height=int(metadata["source_height"]),
            source_mode=metadata.get("source_mode", ""),
            source_mtime_ns=int(metadata.get("source_mtime_ns", "0")),
            source_size=int(metadata.get("source_size", "0")),
            source_dpi_x=float(metadata.get("source_dpi_x", "0")),
            source_dpi_y=float(metadata.get("source_dpi_y", "0")),
            options=ImageCleanupOptions(
                strength=int(metadata.get("strength", "50")),
                preserve_faint_ink=metadata.get("preserve_faint_ink", "1") == "1",
                remove_small_noise=metadata.get("remove_small_noise", "1") == "1",
                feather_edges=metadata.get("feather_edges", "1") == "1",
                processing_mode=metadata.get("processing_mode", "auto"),
            ),
            strokes=strokes,
            algorithm_version=int(metadata.get("algorithm_version", "1")),
            resolved_profile=metadata.get("resolved_profile", ""),
            project_path=target,
            created_at=metadata.get("created_at", _utc_now()),
            updated_at=metadata.get("updated_at", _utc_now()),
        )
        if (
            source_stat.st_mtime_ns != project.source_mtime_ns
            or source_stat.st_size != project.source_size
        ):
            raise ValueError("原稿在项目保存后发生过变化，请重新导入并核对结果。")
        return project
