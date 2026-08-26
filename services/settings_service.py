"""程序级用户设置的校验、持久化与维护服务。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import config
from data.application_database import (
    ApplicationDatabase,
    resolve_application_database_path,
)
from data.config_store import load_global_config, save_global_config


PERFORMANCE_AUTO = "自动"
PERFORMANCE_CONSERVATIVE = "保守"
PERFORMANCE_MODES = (PERFORMANCE_AUTO, PERFORMANCE_CONSERVATIVE)


@dataclass(frozen=True, slots=True)
class ApplicationSettings:
    """设置页允许修改的程序级偏好。"""

    default_dpi: int = 300
    default_canvas_width: int = 250
    default_canvas_height: int = 250
    default_image_directory: str = ""
    default_export_directory: str = ""
    default_layout_directory: str = ""
    performance_mode: str = PERFORMANCE_AUTO


class SettingsService:
    """集中管理跨字库、跨功能并长期生效的程序设置。"""

    def load(self) -> ApplicationSettings:
        data = load_global_config()
        return ApplicationSettings(
            default_dpi=self._bounded_int(data.get("默认DPI"), 300, 1, 9600),
            default_canvas_width=self._bounded_int(
                data.get("默认画布宽"), 250, 1, 50_000
            ),
            default_canvas_height=self._bounded_int(
                data.get("默认画布高"), 250, 1, 50_000
            ),
            default_image_directory=self._path_text(data.get("默认图片目录")),
            default_export_directory=self._path_text(data.get("默认导出目录")),
            default_layout_directory=self._path_text(
                data.get("默认排版输出目录")
            ),
            performance_mode=(
                str(data.get("性能模式"))
                if str(data.get("性能模式")) in PERFORMANCE_MODES
                else PERFORMANCE_AUTO
            ),
        )

    def save(self, settings: ApplicationSettings) -> None:
        validated = self.validate(settings)
        data = load_global_config()
        data.update(
            {
                "默认DPI": validated.default_dpi,
                "默认画布宽": validated.default_canvas_width,
                "默认画布高": validated.default_canvas_height,
                "默认图片目录": validated.default_image_directory,
                "默认导出目录": validated.default_export_directory,
                "默认排版输出目录": validated.default_layout_directory,
                "性能模式": validated.performance_mode,
            }
        )
        save_global_config(data)

    def validate(self, settings: ApplicationSettings) -> ApplicationSettings:
        if not 1 <= int(settings.default_dpi) <= 9600:
            raise ValueError("默认分辨率必须在 1 至 9600 DPI 之间。")
        if not 1 <= int(settings.default_canvas_width) <= 50_000:
            raise ValueError("默认画布宽度必须在 1 至 50000 像素之间。")
        if not 1 <= int(settings.default_canvas_height) <= 50_000:
            raise ValueError("默认画布高度必须在 1 至 50000 像素之间。")
        if settings.performance_mode not in PERFORMANCE_MODES:
            raise ValueError("性能模式无效。")

        directories = (
            ("默认图片目录", settings.default_image_directory),
            ("默认导出目录", settings.default_export_directory),
            ("默认排版输出目录", settings.default_layout_directory),
        )
        normalized: dict[str, str] = {}
        for name, value in directories:
            path = self._path_text(value)
            if path and not os.path.isdir(path):
                raise ValueError(f"{name}不存在，请重新选择。")
            normalized[name] = os.path.abspath(path) if path else ""

        return ApplicationSettings(
            default_dpi=int(settings.default_dpi),
            default_canvas_width=int(settings.default_canvas_width),
            default_canvas_height=int(settings.default_canvas_height),
            default_image_directory=normalized["默认图片目录"],
            default_export_directory=normalized["默认导出目录"],
            default_layout_directory=normalized["默认排版输出目录"],
            performance_mode=settings.performance_mode,
        )

    def check_database_integrity(self) -> list[str]:
        database = ApplicationDatabase(self.database_path)
        return database.check_integrity()

    @property
    def database_path(self) -> str:
        return resolve_application_database_path(config.GLOBAL_CONFIG_FILE)

    @property
    def library_root(self) -> str:
        return os.path.abspath(config.ZIKU_ROOT)

    @property
    def log_path(self) -> str:
        return os.path.abspath(config.LOG_FILE)

    @staticmethod
    def defaults() -> ApplicationSettings:
        return ApplicationSettings()

    @staticmethod
    def usable_directory(path: str, fallback: str = "") -> str:
        return path if path and os.path.isdir(path) else fallback

    @staticmethod
    def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            return default
        return number if minimum <= number <= maximum else default

    @staticmethod
    def _path_text(value: Any) -> str:
        return str(value or "").strip()
