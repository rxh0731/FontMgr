"""图片实验室预览和完整尺寸导出测试。"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
from psd_tools import PSDImage

from core.image_cleanup import clean_document_image
from data.image_lab_project_store import ImageLabStroke
from services.image_lab_service import ImageLabCancelled, ImageLabService


class ImageLabServiceTests(unittest.TestCase):
    @staticmethod
    def _source(path: str) -> np.ndarray:
        source = np.full((360, 480, 3), (232, 218, 184), dtype=np.uint8)
        cv2.putText(
            source,
            "TEST",
            (70, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.3,
            (30, 45, 70),
            7,
            cv2.LINE_AA,
        )
        Image.fromarray(source).save(path)
        return source

    def test_preview_preserves_source_and_applies_manual_strokes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = os.path.join(temp_dir, "样本.png")
            original = self._source(source_path)
            service = ImageLabService()
            project = service.create_project(source_path)
            first = service.load_preview(project, max_edge=600)
            row, column = np.unravel_index(
                int(np.argmin(first.effective_alpha)),
                first.effective_alpha.shape,
            )
            project.strokes.append(
                ImageLabStroke(
                    "cover",
                    40,
                    ((column / 480.0, row / 360.0),),
                )
            )
            second = service.load_preview(project, max_edge=600)

            self.assertTrue(np.array_equal(np.array(Image.open(source_path)), original))
            self.assertGreater(
                int(second.effective_alpha[row, column]),
                int(first.effective_alpha[row, column]),
            )

    def test_full_export_is_atomic_and_keeps_original_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = os.path.join(temp_dir, "样本.png")
            original = self._source(source_path)
            service = ImageLabService()
            project = service.create_project(source_path)
            output_path = os.path.join(temp_dir, "清理结果.tif")

            result = service.export_full_resolution(
                project,
                output_path,
                tile_size=512,
                overlap=64,
            )

            self.assertEqual((result.width, result.height), (480, 360))
            self.assertTrue(os.path.isfile(output_path))
            with Image.open(output_path) as exported:
                self.assertEqual(exported.size, (480, 360))
            self.assertTrue(np.array_equal(np.array(Image.open(source_path)), original))
            self.assertFalse(
                any(name.endswith(".raw") or ".tmp" in name for name in os.listdir(temp_dir))
            )

    def test_full_export_reuses_preview_background_calibration_for_every_tile(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = os.path.join(temp_dir, "彩色跨分块样本.png")
            height, width = 720, 1280
            source = np.full((height, width, 3), (238, 221, 168), dtype=np.uint8)
            cv2.line(source, (80, 350), (1200, 350), (35, 72, 190), 12, cv2.LINE_AA)
            Image.fromarray(source).save(source_path)
            output_path = os.path.join(temp_dir, "跨分块结果.png")
            service = ImageLabService()
            project = service.create_project(source_path)
            calibrations = []

            def tracked_cleanup(source_array, options=None, calibration=None):
                calibrations.append(calibration)
                return clean_document_image(source_array, options, calibration)

            with patch(
                "services.image_lab_service.clean_document_image",
                side_effect=tracked_cleanup,
            ):
                service.export_full_resolution(
                    project,
                    output_path,
                    kind="layer",
                    tile_size=512,
                    overlap=64,
                )

            self.assertIsNone(calibrations[0])
            self.assertGreater(len(calibrations), 2)
            self.assertTrue(
                all(calibration is calibrations[1] for calibration in calibrations[1:])
            )
            with Image.open(output_path) as exported:
                alpha = np.array(exported.convert("RGBA"))[:, :, 3]
            for boundary in (512, 1024):
                self.assertLess(int(alpha[350, boundary]), 40)
                self.assertGreater(int(alpha[250, boundary]), 245)

    def test_cancel_does_not_replace_existing_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = os.path.join(temp_dir, "样本.png")
            self._source(source_path)
            output_path = os.path.join(temp_dir, "结果.png")
            with open(output_path, "wb") as stream:
                stream.write(b"existing-target")
            service = ImageLabService()
            project = service.create_project(source_path)

            with self.assertRaises(ImageLabCancelled):
                service.export_full_resolution(
                    project,
                    output_path,
                    cancelled=lambda: True,
                    tile_size=512,
                    overlap=64,
                )

            with open(output_path, "rb") as stream:
                self.assertEqual(stream.read(), b"existing-target")

    def test_transparent_layer_export_contains_white_rgba_layer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = os.path.join(temp_dir, "样本.png")
            self._source(source_path)
            output_path = os.path.join(temp_dir, "清理层.png")
            service = ImageLabService()
            project = service.create_project(source_path)

            service.export_full_resolution(
                project,
                output_path,
                kind="layer",
                tile_size=512,
                overlap=64,
            )

            with Image.open(output_path) as exported:
                rgba = np.array(exported.convert("RGBA"))
            self.assertTrue(np.all(rgba[:, :, :3] == 255))
            self.assertGreater(int(rgba[:, :, 3].max()), 0)
            self.assertLess(int(rgba[:, :, 3].min()), 255)

    def test_photoshop_export_has_editable_preprocessing_layers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = os.path.join(temp_dir, "样本.png")
            self._source(source_path)
            output_path = os.path.join(temp_dir, "预处理.psd")
            service = ImageLabService()
            project = service.create_project(source_path)

            result = service.export_full_resolution(
                project,
                output_path,
                kind="photoshop",
                tile_size=512,
                overlap=64,
            )

            self.assertEqual(result.output_path, output_path)
            psd = PSDImage.open(output_path, encoding="gb18030")
            self.assertEqual(psd.size, (480, 360))
            self.assertEqual(
                [layer.name for layer in psd],
                ["原稿（锁定）", "白色清理层", "笔画修补"],
            )
            original_layer = psd[0]
            cleanup_layer = psd[1]
            repair_layer = psd[2]
            self.assertIsNotNone(original_layer.locks)
            self.assertTrue(original_layer.locks.transparency)
            self.assertTrue(original_layer.locks.composite)
            self.assertTrue(original_layer.locks.position)
            self.assertEqual(cleanup_layer.size, (480, 360))
            self.assertEqual(repair_layer.size, (1, 1))
            cleanup_rgba = np.array(cleanup_layer.topil().convert("RGBA"))
            self.assertTrue(np.all(cleanup_rgba[:, :, :3] == 255))
            self.assertGreater(int(cleanup_rgba[:, :, 3].max()), 0)
            self.assertLess(int(cleanup_rgba[:, :, 3].min()), 255)

    def test_explicit_psb_export_uses_large_document_header(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = os.path.join(temp_dir, "样本.png")
            self._source(source_path)
            output_path = os.path.join(temp_dir, "预处理.psb")
            service = ImageLabService()
            project = service.create_project(source_path)

            service.export_full_resolution(
                project,
                output_path,
                kind="photoshop",
                tile_size=512,
                overlap=64,
            )

            psb = PSDImage.open(output_path, encoding="gb18030")
            self.assertEqual(psb._record.header.version, 2)


if __name__ == "__main__":
    unittest.main()
