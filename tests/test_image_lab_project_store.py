"""图片实验室独立项目文件测试。"""

from __future__ import annotations

import os
import tempfile
import unittest

from PIL import Image

from core.image_cleanup import ImageCleanupOptions
from data.image_lab_project_store import ImageLabProjectStore, ImageLabStroke


class ImageLabProjectStoreTests(unittest.TestCase):
    def test_project_round_trip_and_repeated_save(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = os.path.join(temp_dir, "原稿.png")
            Image.new("RGB", (120, 80), (220, 210, 180)).save(source_path)
            project_path = os.path.join(temp_dir, "整理项目.fontlab")
            store = ImageLabProjectStore()
            project = store.create(source_path, width=120, height=80, mode="RGB")
            project.options = ImageCleanupOptions(
                strength=72,
                preserve_faint_ink=False,
                remove_small_noise=False,
                feather_edges=False,
                processing_mode="rubbing_dark",
            )
            project.strokes.append(
                ImageLabStroke("cover", 18, ((0.1, 0.2), (0.4, 0.6)))
            )

            self.assertEqual(store.save(project, project_path), project_path)
            self.assertEqual(store.save(project), project_path)
            loaded = store.load(project_path)

            self.assertEqual(loaded.source_path, source_path)
            self.assertEqual(loaded.options, project.options)
            self.assertGreaterEqual(loaded.algorithm_version, 2)
            self.assertEqual(loaded.strokes, project.strokes)
            self.assertEqual(loaded.source_width, 120)
            self.assertEqual(loaded.source_height, 80)

    def test_changed_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = os.path.join(temp_dir, "原稿.png")
            Image.new("RGB", (40, 30), "white").save(source_path)
            store = ImageLabProjectStore()
            project = store.create(source_path, width=40, height=30, mode="RGB")
            project_path = store.save(project, os.path.join(temp_dir, "项目.fontlab"))
            with open(source_path, "ab") as stream:
                stream.write(b"changed")

            with self.assertRaisesRegex(ValueError, "发生过变化"):
                store.load(project_path)


if __name__ == "__main__":
    unittest.main()
