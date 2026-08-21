"""Windows 应用身份与任务栏图标回归测试。"""

from __future__ import annotations

import os
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

import config
from utils.application_identity import (
    configure_windows_app_identity,
    load_application_icon,
)


class ApplicationIdentityTests(unittest.TestCase):
    """确保开发版和便携版使用同一应用身份与彩色方块字图标。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_window_icon_asset_can_be_loaded(self) -> None:
        self.assertTrue(os.path.isfile(config.WINDOW_ICON_FILE))
        self.assertTrue(os.path.isfile(config.ICON_FILE))
        icon = load_application_icon()
        self.assertFalse(icon.isNull())
        for size in (16, 24, 32, 48, 64, 128, 256):
            self.assertFalse(icon.pixmap(size, size).isNull())

    def test_windows_app_id_uses_fontmgr_product_identity(self) -> None:
        self.assertEqual(
            config.WINDOWS_APP_USER_MODEL_ID,
            "RuanXiaohua.FontMgr",
        )

    def test_windows_app_id_is_set_explicitly(self) -> None:
        setter = MagicMock(return_value=0)
        windows_api = SimpleNamespace(
            shell32=SimpleNamespace(
                SetCurrentProcessExplicitAppUserModelID=setter,
            )
        )
        with (
            patch("utils.application_identity.sys.platform", "win32"),
            patch("utils.application_identity.ctypes.windll", windows_api),
        ):
            self.assertTrue(configure_windows_app_identity())

        setter.assert_called_once_with(config.WINDOWS_APP_USER_MODEL_ID)

    def test_non_windows_platform_skips_windows_api(self) -> None:
        with patch("utils.application_identity.sys.platform", "linux"):
            self.assertFalse(configure_windows_app_identity())


if __name__ == "__main__":
    unittest.main()
