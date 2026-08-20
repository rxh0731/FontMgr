"""Photoshop TIFF 单图层 Alpha 解码与安全回退测试。"""

from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, TiffImagePlugin, TiffTags

from core.photoshop_tiff import decode_single_layer_rgba
from core.source_classification import (
    SOURCE_TYPE_TRANSPARENT,
    TRANSPARENCY_SOURCE_PHOTOSHOP_ALPHA,
    TRANSPARENCY_SOURCE_PHOTOSHOP_METADATA,
    TRANSPARENCY_SOURCE_STANDARD_ALPHA,
    classify_source,
)
from services.optimization_service import (
    CANDIDATE_TYPE_DIRECT,
    OptimizationService,
)


class PhotoshopTiffTests(unittest.TestCase):
    """验证只采用能够完整重建并复核的 Photoshop 单图层。"""

    def setUp(self) -> None:
        self.service = OptimizationService(None)  # type: ignore[arg-type]

    def test_raw_single_layer_decodes_exact_rgba(self) -> None:
        layer = self._sample_layer()
        payload = self._photoshop_payload([((1, 1, 4, 5), layer)], compression=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.tif"
            self._save_tagged_tiff(path, (6, 5), [((1, 1), layer)], payload)
            with Image.open(path) as source:
                decoded = decode_single_layer_rgba(source)

        self.assertIsNotNone(decoded)
        assert decoded is not None
        try:
            self.assertEqual(decoded.size, (6, 5))
            self.assertEqual(decoded.getpixel((0, 0)), (0, 0, 0, 0))
            self.assertEqual(decoded.getpixel((2, 2)), tuple(layer[1, 1]))
        finally:
            decoded.close()

    def test_packbits_layer_is_direct_batch_fast_path(self) -> None:
        layer = self._sample_layer()
        payload = self._photoshop_payload([((1, 1, 4, 5), layer)], compression=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "packbits.tif"
            self._save_tagged_tiff(path, (6, 5), [((1, 1), layer)], payload)

            rgba, _gray, source_type = self.service._load_source(str(path))
            try:
                self.assertEqual(source_type, TRANSPARENCY_SOURCE_PHOTOSHOP_ALPHA)
                self.assertEqual(rgba.getpixel((2, 2)), tuple(layer[1, 1]))
            finally:
                rgba.close()

            with (
                patch(
                    "services.optimization_service.generate_candidate_results",
                    side_effect=AssertionError("已解码 Alpha 不应进入完整寻优"),
                ),
                patch("services.optimization_service.write_log"),
            ):
                candidate = self.service.generate_batch_candidate(
                    {"原始路径": str(path), "归属字": "测"}
                )

        self.assertEqual(candidate["处理类型"], CANDIDATE_TYPE_DIRECT)
        self.assertEqual(
            candidate["方案"]["透明来源"],
            TRANSPARENCY_SOURCE_PHOTOSHOP_ALPHA,
        )

    def test_corrupt_channel_length_falls_back_to_flattened_image(self) -> None:
        self._assert_unsupported_payload_falls_back(
            self._photoshop_payload(
                [((1, 1, 4, 5), self._sample_layer())],
                compression=1,
                corrupt_length=True,
            )
        )

    def test_multilayer_payload_falls_back_to_flattened_image(self) -> None:
        layer = self._sample_layer()
        self._assert_unsupported_payload_falls_back(
            self._photoshop_payload(
                [((1, 1, 4, 5), layer), ((1, 1, 4, 5), layer)],
                compression=1,
            )
        )

    def test_unsupported_zip_compression_falls_back_to_flattened_image(self) -> None:
        self._assert_unsupported_payload_falls_back(
            self._photoshop_payload(
                [((1, 1, 4, 5), self._sample_layer())],
                compression=2,
            )
        )

    def test_composite_mismatch_falls_back_to_flattened_image(self) -> None:
        layer = self._sample_layer()
        payload = self._photoshop_payload([((1, 1, 4, 5), layer)], compression=0)
        flattened = Image.new("RGB", (6, 5), "white")
        flattened.putpixel((0, 0), (0, 0, 0))
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "合成不一致.tif"
                self._save_rgb_with_tag(path, flattened, payload)
                rgba, _gray, source_type = self.service._load_source(str(path))
            try:
                self.assertEqual(source_type, TRANSPARENCY_SOURCE_PHOTOSHOP_METADATA)
                alpha_image = rgba.getchannel("A")
                try:
                    self.assertEqual(alpha_image.getextrema(), (255, 255))
                finally:
                    alpha_image.close()
            finally:
                rgba.close()
        finally:
            flattened.close()

    def test_oversized_layer_falls_back_before_decoding_channels(self) -> None:
        layer = self._sample_layer()
        payload = self._photoshop_payload([((1, 1, 4, 5), layer)], compression=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "超限.tif"
            self._save_tagged_tiff(path, (6, 5), [((1, 1), layer)], payload)
            with Image.open(path) as source:
                with (
                    patch("core.photoshop_tiff._MAX_LAYER_PIXELS", 4),
                    patch("core.photoshop_tiff._decode_channel") as decode_channel,
                ):
                    decoded = decode_single_layer_rgba(source)

        self.assertIsNone(decoded)
        decode_channel.assert_not_called()

    def test_oversized_channel_declaration_falls_back_before_decoding(self) -> None:
        layer = self._sample_layer()
        payload = self._photoshop_payload([((1, 1, 4, 5), layer)], compression=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "通道超限.tif"
            self._save_tagged_tiff(path, (6, 5), [((1, 1), layer)], payload)
            with Image.open(path) as source:
                with (
                    patch("core.photoshop_tiff._MAX_CHANNEL_BYTES", 4),
                    patch("core.photoshop_tiff._decode_channel") as decode_channel,
                ):
                    decoded = decode_single_layer_rgba(source)

        self.assertIsNone(decoded)
        decode_channel.assert_not_called()

    def test_oversized_canvas_falls_back_before_decoding_channels(self) -> None:
        layer = self._sample_layer()
        payload = self._photoshop_payload([((1, 1, 4, 5), layer)], compression=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "超大画布.tif"
            self._save_tagged_tiff(path, (6, 5), [((1, 1), layer)], payload)
            for limit_name, limit in (
                ("_MAX_CANVAS_PIXELS", 20),
                ("_MAX_CANVAS_SIDE", 5),
            ):
                with self.subTest(limit_name=limit_name):
                    with Image.open(path) as source:
                        with (
                            patch(f"core.photoshop_tiff.{limit_name}", limit),
                            patch("core.photoshop_tiff._decode_channel") as decode_channel,
                        ):
                            decoded = decode_single_layer_rgba(source)

                    self.assertIsNone(decoded)
                    decode_channel.assert_not_called()

    def test_canvas_is_closed_when_composite_verification_raises(self) -> None:
        layer = self._sample_layer()
        payload = self._photoshop_payload([((1, 1, 4, 5), layer)], compression=0)
        created: list[Image.Image] = []
        real_image_new = Image.new

        def tracked_image_new(*args: object, **kwargs: object) -> Image.Image:
            result = real_image_new(*args, **kwargs)
            created.append(result)
            return result

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "复核异常.tif"
            self._save_tagged_tiff(path, (6, 5), [((1, 1), layer)], payload)
            with Image.open(path) as source:
                with (
                    patch("core.photoshop_tiff.Image.new", side_effect=tracked_image_new),
                    patch(
                        "core.photoshop_tiff._matches_flattened_white_composite",
                        side_effect=OSError("模拟复核异常"),
                    ),
                ):
                    decoded = decode_single_layer_rgba(source)

        self.assertIsNone(decoded)
        self.assertGreaterEqual(len(created), 1)
        for image in created:
            with self.assertRaises(ValueError):
                image.getpixel((0, 0))

    def test_single_edge_band_cannot_hide_large_internal_alpha_defect(self) -> None:
        rgba = Image.new("RGBA", (100, 100), (24, 24, 24, 255))
        alpha = np.full((100, 100), 255, dtype=np.uint8)
        alpha[:, :12] = 0
        alpha[20:80, 20:80] = 0
        alpha_image = Image.fromarray(alpha, "L")
        try:
            rgba.putalpha(alpha_image)
        finally:
            alpha_image.close()
        gray = np.full((100, 100), 255, dtype=np.float32)
        gray[30:70, 44:56] = 24
        try:
            result = classify_source(
                rgba,
                gray,
                TRANSPARENCY_SOURCE_STANDARD_ALPHA,
            )
        finally:
            rgba.close()

        self.assertEqual(int(result.metrics["边缘连通透明触边数"]), 3)
        self.assertNotEqual(result.source_type, SOURCE_TYPE_TRANSPARENT)

    def test_nearly_transparent_dark_residue_does_not_hide_white_glyph(self) -> None:
        source = Image.new("RGBA", (20, 20), (0, 0, 0, 1))
        for y in range(6, 14):
            for x in range(8, 12):
                source.putpixel((x, y), (255, 255, 255, 255))
        gray = np.full((20, 20), 255, dtype=np.float32)

        corrected, inverted = self.service._normalize_source_polarity(
            source,
            gray,
            TRANSPARENCY_SOURCE_STANDARD_ALPHA,
        )

        self.assertTrue(inverted)
        self.assertEqual(int(corrected[10, 10]), 0)
        self.assertEqual(int(corrected[0, 0]), 255)
        source.close()

    def _assert_unsupported_payload_falls_back(self, payload: bytes) -> None:
        flattened = Image.new("RGB", (6, 5), "white")
        flattened.putpixel((2, 2), (24, 24, 24))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fallback.tif"
            self._save_rgb_with_tag(path, flattened, payload)
            rgba, _gray, source_type = self.service._load_source(str(path))
        try:
            self.assertEqual(source_type, TRANSPARENCY_SOURCE_PHOTOSHOP_METADATA)
            self.assertEqual(rgba.getchannel("A").getextrema(), (255, 255))
            self.assertEqual(rgba.getpixel((2, 2)), (24, 24, 24, 255))
        finally:
            rgba.close()
            flattened.close()

    @staticmethod
    def _sample_layer() -> np.ndarray:
        layer = np.zeros((3, 4, 4), dtype=np.uint8)
        layer[..., :3] = 32
        layer[..., 3] = 0
        layer[1, 1] = (18, 18, 18, 255)
        layer[1, 2] = (36, 36, 36, 255)
        layer[2, 1] = (72, 72, 72, 160)
        return layer

    @classmethod
    def _photoshop_payload(
        cls,
        layers: list[tuple[tuple[int, int, int, int], np.ndarray]],
        *,
        compression: int,
        corrupt_length: bool = False,
    ) -> bytes:
        records = bytearray(struct.pack("<h", len(layers)))
        channel_payloads: list[bytes] = []
        for layer_index, (bounds, rgba) in enumerate(layers):
            height, width = rgba.shape[:2]
            self_channels = (
                (-1, rgba[..., 3]),
                (0, rgba[..., 0]),
                (1, rgba[..., 1]),
                (2, rgba[..., 2]),
            )
            encoded = [
                (channel_id, cls._encode_channel(channel, compression))
                for channel_id, channel in self_channels
            ]
            records.extend(struct.pack("<4iH", *bounds, len(encoded)))
            for channel_index, (channel_id, data) in enumerate(encoded):
                length = len(data)
                if corrupt_length and layer_index == 0 and channel_index == 0:
                    length += 1
                records.extend(struct.pack("<hI", channel_id, length))
            records.extend(b"MIB8mron")
            records.extend(bytes((255, 0, 0, 0)))
            extra = struct.pack("<II", 0, 0)
            records.extend(struct.pack("<I", len(extra)))
            records.extend(extra)
            channel_payloads.extend(data for _channel_id, data in encoded)
            if (height, width) != (
                bounds[2] - bounds[0],
                bounds[3] - bounds[1],
            ):
                raise AssertionError("测试图层尺寸与边界不一致")
        block = bytes(records) + b"".join(channel_payloads)
        header = b"Adobe Photoshop Document Data Block\x00"
        return header + b"MIB8ryaL" + struct.pack("<I", len(block)) + block

    @staticmethod
    def _encode_channel(channel: np.ndarray, compression: int) -> bytes:
        source = np.asarray(channel, dtype=np.uint8)
        if compression == 0:
            return struct.pack("<H", 0) + source.tobytes()
        if compression == 1:
            rows: list[bytes] = []
            for row in source:
                encoded = bytearray()
                row_bytes = row.tobytes()
                for start in range(0, len(row_bytes), 128):
                    chunk = row_bytes[start:start + 128]
                    encoded.append(len(chunk) - 1)
                    encoded.extend(chunk)
                rows.append(bytes(encoded))
            lengths = struct.pack("<" + "H" * len(rows), *(len(row) for row in rows))
            return struct.pack("<H", 1) + lengths + b"".join(rows)
        return struct.pack("<H", compression) + b"unsupported"

    @classmethod
    def _save_tagged_tiff(
        cls,
        path: Path,
        canvas_size: tuple[int, int],
        layers: list[tuple[tuple[int, int], np.ndarray]],
        payload: bytes,
    ) -> None:
        canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
        for position, rgba in layers:
            layer = Image.fromarray(np.asarray(rgba, dtype=np.uint8), "RGBA")
            canvas.alpha_composite(layer, dest=position)
            layer.close()
        white = Image.new("RGBA", canvas_size, "white")
        white.alpha_composite(canvas)
        flattened = white.convert("RGB")
        try:
            cls._save_rgb_with_tag(path, flattened, payload)
        finally:
            flattened.close()
            white.close()
            canvas.close()

    @staticmethod
    def _save_rgb_with_tag(path: Path, image: Image.Image, payload: bytes) -> None:
        tiff_info = TiffImagePlugin.ImageFileDirectory_v2()
        tiff_info.tagtype[37724] = TiffTags.UNDEFINED
        tiff_info[37724] = payload
        image.save(path, "TIFF", compression="tiff_lzw", tiffinfo=tiff_info)


if __name__ == "__main__":
    unittest.main()
