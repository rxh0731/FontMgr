"""程序级设置服务回归测试。"""

from __future__ import annotations

import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import config
from data.config_store import load_global_config, save_global_config
from services.settings_service import (
    PERFORMANCE_CONSERVATIVE,
    ApplicationSettings,
    SettingsService,
)


class SettingsServiceTests(unittest.TestCase):
    def test_save_preserves_unmanaged_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            config_dir = root / "配置"
            config_dir.mkdir()
            image_dir = root / "图片"
            export_dir = root / "导出"
            layout_dir = root / "排版"
            for path in (image_dir, export_dir, layout_dir):
                path.mkdir()
            stack.enter_context(patch.object(config, "CONFIG_DIR", str(config_dir)))
            stack.enter_context(
                patch.object(
                    config,
                    "APP_DATABASE_FILE",
                    str(config_dir / "fontmgr.sqlite3"),
                )
            )
            stack.enter_context(
                patch.object(
                    config,
                    "GLOBAL_CONFIG_FILE",
                    str(config_dir / "用户设置.json"),
                )
            )
            save_global_config(
                {
                    "最后一次打开的字库": r"D:\字库\保留项",
                    "未公开设置": "继续保留",
                }
            )

            service = SettingsService()
            settings = ApplicationSettings(
                default_dpi=600,
                default_canvas_width=320,
                default_canvas_height=480,
                default_image_directory=str(image_dir),
                default_export_directory=str(export_dir),
                default_layout_directory=str(layout_dir),
                performance_mode=PERFORMANCE_CONSERVATIVE,
            )
            service.save(settings)

            self.assertEqual(service.load(), settings)
            raw = load_global_config()
            self.assertEqual(raw["最后一次打开的字库"], r"D:\字库\保留项")
            self.assertEqual(raw["未公开设置"], "继续保留")
            self.assertEqual(service.check_database_integrity(), [])

    def test_missing_default_directory_is_rejected(self) -> None:
        settings = ApplicationSettings(default_image_directory=r"Z:\不存在目录")

        with self.assertRaisesRegex(ValueError, "默认图片目录不存在"):
            SettingsService().validate(settings)


if __name__ == "__main__":
    unittest.main()
