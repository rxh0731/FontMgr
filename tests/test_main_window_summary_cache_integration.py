"""首页摘要索引与真实主窗口启动流程集成测试。"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import QApplication

import config
from ui.main_window import MainWindow


class MainWindowSummaryCacheIntegrationTests(unittest.TestCase):
    """验证首次核对、持久缓存命中和普通返回的完整链路。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_second_startup_and_home_return_do_not_scan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "字库"
            library_dir = root / "空字库"
            library_dir.mkdir(parents=True)
            (library_dir / "空字库.json").write_text(
                json.dumps(
                    {
                        "数据版本": 3,
                        "库名": "空字库",
                        "变体详情": {},
                        "字形组索引": {},
                        "会话": {},
                        "元数据": {
                            "DPI": 300,
                            "画布宽": 250,
                            "画布高": 250,
                        },
                        "整体协调": {},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            cache_file = Path(temp_dir) / "配置" / "字库状态索引.json"

            with (
                patch.object(config, "ZIKU_ROOT", str(root)),
                patch.object(
                    config,
                    "LIBRARY_SUMMARY_CACHE_FILE",
                    str(cache_file),
                ),
            ):
                first_window = MainWindow()
                self.assertFalse(first_window.windowIcon().isNull())
                self.assertTrue(first_window._library_scan_active)
                self.assertEqual(first_window._library_scan_generation, 1)
                for _ in range(2):
                    self.assertTrue(QThreadPool.globalInstance().waitForDone(5000))
                    for _event in range(3):
                        self.app.processEvents()

                self.assertFalse(first_window._library_scan_active)
                self.assertTrue(first_window._library_cache_ready)
                self.assertTrue(cache_file.with_suffix(".sqlite3").is_file())
                first_window.close()
                first_window.deleteLater()
                self.app.processEvents()

                second_window = MainWindow()
                self.assertFalse(second_window._library_scan_active)
                self.assertTrue(second_window._library_cache_ready)
                self.assertEqual(second_window._library_scan_generation, 0)
                self.assertEqual(second_window._home_page._table.rowCount(), 1)

                second_window.show_home()
                self.app.processEvents()
                self.assertEqual(second_window._library_scan_generation, 0)
                second_window.close()
                second_window.deleteLater()
                self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
