"""图片实验室页面交互测试。"""

from __future__ import annotations

import os
import tempfile
import time
import unittest

import cv2
import numpy as np
from PIL import Image
from PySide6.QtWidgets import QApplication

from data.image_lab_project_store import ImageLabStroke
from services.image_lab_service import ImageLabService
from ui.pages.image_lab_page import ImageLabPage


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class ImageLabPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _wait_preview(self, page: ImageLabPage) -> None:
        deadline = time.monotonic() + 8.0
        while page._preview_worker is not None and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.app.processEvents()
        self.assertIsNone(page._preview_worker)
        self.assertIsNotNone(page._preview)

    def test_workspace_loads_preview_and_tracks_manual_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = os.path.join(temp_dir, "拓片.png")
            source = np.full((320, 240, 3), 220, dtype=np.uint8)
            cv2.putText(
                source,
                "A",
                (65, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                3.0,
                (25, 25, 25),
                8,
                cv2.LINE_AA,
            )
            Image.fromarray(source).save(source_path)
            service = ImageLabService()
            page = ImageLabPage(service=service)
            page.resize(1100, 720)
            page.show()
            project = service.create_project(source_path)

            page._set_project(project, dirty=False)
            self._wait_preview(page)

            self.assertTrue(page._canvas.has_image)
            self.assertIn("原稿尺寸", page._metrics_label.text())
            self.assertFalse(page.is_dirty)
            page._stroke_finished("cover", 30, ((0.5, 0.5),))
            self.assertTrue(page.is_dirty)
            self.assertEqual(len(project.strokes), 1)
            page._undo_stroke()
            self.assertEqual(project.strokes, [])
            page.close()
            page.deleteLater()

    def test_restore_stroke_is_available(self) -> None:
        stroke = ImageLabStroke("restore", 20, ((0.2, 0.3), (0.4, 0.5)))
        self.assertEqual(stroke.tool, "restore")

    def test_home_button_is_the_rightmost_header_action(self) -> None:
        page = ImageLabPage()
        page.resize(1100, 720)
        page.show()
        self.app.processEvents()

        self.assertEqual(page._home_button.text(), "返回首页")
        self.assertGreater(
            page._home_button.geometry().left(),
            page._save_button.geometry().right(),
        )
        self.assertLessEqual(page._home_button.geometry().right(), page.width())

        emissions: list[bool] = []
        page.home_requested.connect(lambda: emissions.append(True))
        page._home_button.click()
        self.assertEqual(emissions, [True])
        page.close()
        page.deleteLater()


if __name__ == "__main__":
    unittest.main()
