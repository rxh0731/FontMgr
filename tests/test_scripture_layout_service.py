"""经文字图库索引、预览和分层 PSD 输出回归测试。"""

from __future__ import annotations

import io
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageChops
from psd_tools import PSDImage
from psd_tools.api.layers import PixelLayer
from psd_tools.constants import Compression

from core.scripture_layout import (
    SCALE_TO_CELL,
    LayoutParameters,
    allocate_boards,
    compute_grid,
    parse_scripture,
)
from services.glyph_service import GlyphService
from services import scripture_layout_service as layout_service
from services.scripture_layout_service import (
    CONFLICT_OVERWRITE,
    CONFLICT_SKIP,
    GenerationCancelled,
    GlyphIndex,
    GenerationProgress,
    BoardOutputPlan,
    _required_disk_workspace,
    _safe_psd_name,
    board_output_path,
    build_external_glyph_index,
    build_system_glyph_index,
    generate_psd_boards,
    _install_composite_preview,
    render_board_preview,
)


class ScriptureLayoutServiceTests(unittest.TestCase):
    def _write_glyph(self, path: Path, inset: int = 4) -> None:
        image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        for x in range(inset, 32 - inset):
            for y in range(inset, 32 - inset):
                if x in {inset, 31 - inset} or y in {inset, 31 - inset}:
                    image.putpixel((x, y), (18, 18, 18, 230))
        image.save(path, dpi=(300, 300))
        image.close()

    def test_external_index_groups_every_supported_file_by_first_character(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_glyph(root / "甲-10.png")
            self._write_glyph(root / "甲-2.tif")
            self._write_glyph(root / "乙.png")
            self._write_glyph(root / "甲乙.png")
            self._write_glyph(root / "甲说明.bmp")

            index = build_external_glyph_index(directory)

            self.assertEqual(
                [item.version for item in index.images["甲"]],
                [2, 10, 11, 12],
            )
            self.assertEqual(index.resolve("甲", 4).version, 2)
            self.assertEqual(index.variant_count, 5)
            self.assertFalse(any("文件名应为" in issue for issue in index.issues))

    def test_external_index_assigns_stable_physical_size_without_dpi(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "甲.png"
            image = Image.new("RGBA", (300, 150), (0, 0, 0, 255))
            image.save(path)
            image.close()

            index = build_external_glyph_index(directory)
            glyph = index.resolve("甲", 0)

            self.assertIsNotNone(glyph)
            assert glyph is not None
            self.assertAlmostEqual(glyph.source_width_mm, 25.4)
            self.assertAlmostEqual(glyph.source_height_mm, 12.7)
            self.assertTrue(any("按 300 DPI 换算" in issue for issue in index.issues))

    def test_psd_layer_name_keeps_rare_unicode_characters(self) -> None:
        self.assertEqual(_safe_psd_name("𠀀_1_1"), "𠀀_1_1")
        self.assertEqual(_safe_psd_name("甲\x00乙"), "甲_乙")

    def test_disk_workspace_uses_full_estimate_instead_of_256_mib_cap(self) -> None:
        plan = BoardOutputPlan(
            1,
            "PSB",
            ".psb",
            True,
            1000,
            1000,
            2 * 1024**3,
            3 * 1024**3,
        )

        self.assertGreater(_required_disk_workspace(plan), 2 * 1024**3)

    def test_external_index_reports_real_progress_and_supports_cancellation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_glyph(root / "甲.png")
            self._write_glyph(root / "乙.png")
            progress: list[GenerationProgress] = []

            index = build_external_glyph_index(
                directory,
                progress_callback=progress.append,
            )

            self.assertEqual(index.characters, frozenset({"甲", "乙"}))
            self.assertTrue(progress[0].indeterminate)
            self.assertEqual(progress[-1].completed, progress[-1].total)
            self.assertIn("核对完成", progress[-1].message)

            with self.assertRaisesRegex(GenerationCancelled, "停止了字图检查"):
                build_external_glyph_index(
                    directory,
                    cancel_check=lambda: True,
                )

    def test_preview_reuses_decoded_and_resized_repeated_glyph(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "甲.png"
            self._write_glyph(path)
            index = build_external_glyph_index(directory)
            parameters = LayoutParameters(
                dpi=72,
                rows=4,
                columns=5,
                cell_width_mm=8,
                cell_height_mm=8,
                first_title_new_column=False,
                last_title_new_column=False,
                add_annotations=False,
            )
            board = allocate_boards(
                parse_scripture("甲" * 20, index.characters),
                parameters,
            )[0]

            with patch.object(
                layout_service,
                "_prepare_glyph_image",
                wraps=layout_service._prepare_glyph_image,
            ) as prepare:
                preview = render_board_preview(board, index, parameters, (800, 800))

            preview.close()
            self.assertEqual(prepare.call_count, 1)

    def test_glyph_bitmap_cache_evicts_by_bytes(self) -> None:
        cache = layout_service._GlyphBitmapCache(600)
        first = Image.new("RGBA", (10, 10))
        second = Image.new("RGBA", (10, 10))

        self.assertTrue(cache.put(("甲",), first))
        self.assertTrue(cache.put(("乙",), second))

        self.assertIsNone(cache.get(("甲",)))
        self.assertIs(cache.get(("乙",)), second)
        self.assertLessEqual(cache.byte_size, cache.byte_limit)
        cache.close()

    def test_output_file_name_uses_user_base_and_board_sequence(self) -> None:
        self.assertEqual(
            Path(board_output_path("输出", 1, 300)).name,
            "通用经文排版.psd",
        )
        self.assertEqual(
            Path(board_output_path("输出", 1, 300, "经文排版")).name,
            "经文排版.psd",
        )
        self.assertEqual(
            Path(
                board_output_path(
                    "输出",
                    1,
                    300,
                    "经文排版",
                    total_boards=12,
                )
            ).name,
            "经文排版-01.psd",
        )
        self.assertEqual(
            Path(
                board_output_path(
                    "输出",
                    12,
                    300,
                    "经文排版-01.psd",
                    total_boards=12,
                )
            ).name,
            "经文排版-01-12.psd",
        )
        self.assertEqual(
            Path(
                board_output_path(
                    "输出",
                    1,
                    300,
                    "11",
                    total_boards=12,
                )
            ).name,
            "11-01.psd",
        )
        self.assertEqual(
            Path(
                board_output_path(
                    "输出",
                    1,
                    300,
                    "22",
                    total_boards=12,
                )
            ).name,
            "22-01.psd",
        )
        with self.assertRaises(ValueError):
            board_output_path("输出", 1, 300, "错误:名称")

    def test_preview_uses_real_images_and_marks_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_glyph(root / "甲.png")
            index = build_external_glyph_index(directory)
            parameters = LayoutParameters(
                dpi=72,
                cell_width_mm=8,
                cell_height_mm=8,
                rows=1,
                columns=2,
                first_title_new_column=False,
                last_title_new_column=False,
            )
            board = allocate_boards(parse_scripture("甲乙", index.characters), parameters)[0]

            preview = render_board_preview(board, index, parameters, (400, 300))
            try:
                self.assertGreater(preview.width, 1)
                self.assertGreater(preview.height, 1)
                colors = preview.getcolors(maxcolors=preview.width * preview.height)
                self.assertIsNotNone(colors)
                self.assertTrue(any(color[1][0] > 150 and color[1][1] < 100 for color in colors))
            finally:
                preview.close()

    def test_preview_reports_completion_and_discards_cancelled_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_glyph(root / "甲.png")
            index = build_external_glyph_index(directory)
            parameters = LayoutParameters(
                dpi=72,
                cell_width_mm=8,
                cell_height_mm=8,
                rows=2,
                columns=2,
                first_title_new_column=False,
                last_title_new_column=False,
            )
            board = allocate_boards(
                parse_scripture("甲甲", index.characters), parameters
            )[0]
            progress: list[GenerationProgress] = []

            preview = render_board_preview(
                board,
                index,
                parameters,
                (400, 300),
                progress_callback=progress.append,
            )
            preview.close()

            self.assertEqual(progress[-1].completed, progress[-1].total)
            self.assertIn("预览已完成", progress[-1].message)
            with self.assertRaisesRegex(GenerationCancelled, "预览任务已取消"):
                render_board_preview(
                    board,
                    index,
                    parameters,
                    (400, 300),
                    cancel_check=lambda: True,
                )

    def test_preview_can_hide_guides_without_hiding_glyphs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_glyph(root / "甲.png")
            index = build_external_glyph_index(directory)
            parameters = LayoutParameters(
                dpi=72,
                cell_width_mm=8,
                cell_height_mm=8,
                rows=1,
                columns=1,
                first_title_new_column=False,
                last_title_new_column=False,
            )
            board = allocate_boards(
                parse_scripture("甲", index.characters), parameters
            )[0]

            visible = render_board_preview(
                board,
                index,
                parameters,
                (400, 300),
            )
            hidden = render_board_preview(
                board,
                index,
                parameters,
                (400, 300),
                show_guides=False,
            )
            try:
                self.assertIsNotNone(ImageChops.difference(visible, hidden).getbbox())
                grayscale = hidden.convert("L")
                try:
                    self.assertLess(grayscale.getextrema()[0], 255)
                finally:
                    grayscale.close()
            finally:
                visible.close()
                hidden.close()

    def test_system_index_uses_valid_finished_files_even_when_ink_is_pending(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "系统字库"
            root.mkdir()
            glyph = GlyphService("系统字库", str(root))
            glyph.ensure_dirs()
            glyph.init_metadata(
                dpi=300,
                canvas_w=250,
                canvas_h=250,
                width_mm=50,
                height_mm=25,
            )
            finished_dir = Path(glyph.get_workflow_dirs()["成品"])
            completed_id = glyph.add_original("甲", "甲-0001.png", "甲.png", "a" * 32)
            pending_id = glyph.add_original("乙", "乙-0001.png", "乙.png", "b" * 32)
            self._write_glyph(finished_dir / "甲-0001.png")
            glyph.mark_finished(
                completed_id,
                "甲-0001.png",
                "",
                {
                    "墨色协调": {
                        "启用": False,
                        "保存后复测": True,
                        "保存后墨色": 210.0,
                    }
                },
            )
            glyph.set_coordination_summary(
                {},
                None,
                geometry_completed=True,
                ink_completed=False,
                ink_enabled=True,
            )
            glyph.save()

            index = build_system_glyph_index(glyph)

            self.assertEqual(index.characters, frozenset({"甲"}))
            indexed_glyph = index.resolve("甲", 0)
            self.assertIsNotNone(indexed_glyph)
            assert indexed_glyph is not None
            self.assertAlmostEqual(indexed_glyph.source_width_mm, 6.4)
            self.assertAlmostEqual(indexed_glyph.source_height_mm, 3.2)
            self.assertIsNone(index.resolve("乙", 0))
            self.assertTrue(glyph.get_variant(pending_id))

    def test_generate_layered_psd_and_read_it_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph_dir = root / "字图"
            output_dir = root / "输出"
            glyph_dir.mkdir()
            self._write_glyph(glyph_dir / "甲.png")
            self._write_glyph(glyph_dir / "乙.png", inset=6)
            index = build_external_glyph_index(str(glyph_dir))
            parameters = LayoutParameters(
                dpi=72,
                cell_width_mm=8,
                cell_height_mm=8,
                rows=2,
                columns=1,
                row_gap_mm=1,
                column_gap_mm=1,
                frame_top_mm=1,
                frame_bottom_mm=1,
                frame_left_mm=1,
                frame_right_mm=1,
                canvas_top_mm=1,
                canvas_bottom_mm=1,
                canvas_left_mm=1,
                canvas_right_mm=1,
                first_title_new_column=False,
                last_title_new_column=False,
                add_annotations=True,
            )
            boards = allocate_boards(parse_scripture("甲乙", index.characters), parameters)
            progress: list[GenerationProgress] = []

            result = generate_psd_boards(
                boards,
                index,
                parameters,
                str(output_dir),
                progress_callback=progress.append,
            )

            self.assertFalse(result.stopped)
            self.assertEqual(len(result.boards), 1)
            output_path = Path(result.boards[0].path)
            self.assertTrue(output_path.is_file())
            self.assertFalse(any(output_dir.glob("*.tmp.psd")))
            psd = PSDImage.open(output_path, encoding="gbk")
            names = [layer.name for layer in psd.descendants()]
            self.assertIn("经文", names)
            self.assertIn("框线", names)
            self.assertTrue(any(name.startswith("甲_") for name in names))
            self.assertTrue(any(name.startswith("乙_") for name in names))
            glyph_layers = [
                layer
                for layer in psd.descendants()
                if layer.name.startswith(("甲_", "乙_"))
            ]
            self.assertTrue(glyph_layers)
            self.assertTrue(all(layer.width < 32 and layer.height < 32 for layer in glyph_layers))
            self.assertEqual(psd._record.image_data.compression, Compression.RLE)
            self.assertTrue(
                all(
                    channel.compression == Compression.RLE
                    for layer in psd.descendants()
                    if isinstance(layer, PixelLayer)
                    for channel in layer._channels
                )
            )
            embedded_preview = psd.topil()
            rendered_preview = psd.composite(force=True)
            self.assertIsNotNone(embedded_preview)
            difference = ImageChops.difference(
                embedded_preview.convert("RGB"),
                rendered_preview.convert("RGB"),
            )
            extrema = difference.getextrema()
            self.assertLessEqual(
                max(channel_maximum for _channel_minimum, channel_maximum in extrema),
                1,
                f"内嵌预览与图层合成差异过大：{extrema}",
            )
            difference.close()
            embedded_preview.close()
            rendered_preview.close()
            writing = [item for item in progress if item.indeterminate]
            self.assertEqual(len(writing), 2)
            self.assertIn("正在编码第 1/1 版兼容预览", writing[0].message)
            self.assertIn("正在压缩并写入第 1/1 版 PSD", writing[1].message)
            self.assertFalse(progress[-1].indeterminate)
            self.assertEqual(progress[-1].completed, progress[-1].total)

    def test_installed_composite_preview_skips_generic_layer_composite(self) -> None:
        psd = PSDImage.new("RGBA", (16, 16), color=(255, 255, 255, 255))
        layer_image = Image.new("RGBA", (8, 8), (0, 0, 0, 255))
        psd.create_pixel_layer(name="测试", image=layer_image, top=4, left=4)
        layer_image.close()
        preview = Image.new("RGB", (16, 16), "white")
        preview.paste((0, 0, 0), (4, 4, 12, 12))

        _install_composite_preview(psd, preview, Compression.RLE)

        self.assertFalse(psd.is_updated())
        output = io.BytesIO()
        with patch.object(
            psd,
            "composite",
            side_effect=AssertionError("保存阶段不得再次进行通用图层合成"),
        ):
            psd.save(output, encoding="gbk")
        output.seek(0)
        reopened = PSDImage.open(output, encoding="gbk")
        embedded = reopened.topil()
        self.assertEqual(embedded.getpixel((0, 0))[:3], (255, 255, 255))
        self.assertEqual(embedded.getpixel((6, 6))[:3], (0, 0, 0))
        embedded.close()
        preview.close()

    def test_generate_psd_keeps_missing_glyph_cells_blank(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph_dir = root / "字图"
            output_dir = root / "输出"
            glyph_dir.mkdir()
            self._write_glyph(glyph_dir / "甲.png")
            index = build_external_glyph_index(str(glyph_dir))
            parameters = LayoutParameters(
                dpi=72,
                cell_width_mm=8,
                cell_height_mm=8,
                rows=2,
                columns=1,
                first_title_new_column=False,
                last_title_new_column=False,
                add_annotations=False,
            )
            boards = allocate_boards(
                parse_scripture("甲乙", index.characters),
                parameters,
            )

            result = generate_psd_boards(
                boards,
                index,
                parameters,
                str(output_dir),
            )

            self.assertFalse(result.stopped)
            self.assertEqual(result.boards[0].characters, 1)
            self.assertEqual(result.boards[0].missing_characters, 1)
            psd = PSDImage.open(result.boards[0].path, encoding="gbk")
            names = [layer.name for layer in psd.descendants()]
            self.assertTrue(any(name.startswith("甲_") for name in names))
            self.assertFalse(any(name.startswith("乙_") for name in names))

    def test_psd_trims_transparent_margin_after_cell_centering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph_dir = root / "字图"
            output_dir = root / "输出"
            glyph_dir.mkdir()
            self._write_glyph(glyph_dir / "甲.png", inset=4)
            index = build_external_glyph_index(str(glyph_dir))
            parameters = LayoutParameters(
                dpi=72,
                cell_width_mm=32 * 25.4 / 72,
                cell_height_mm=32 * 25.4 / 72,
                rows=1,
                columns=1,
                draw_outer_frame=False,
                canvas_top_mm=0,
                canvas_bottom_mm=0,
                canvas_left_mm=0,
                canvas_right_mm=0,
                scale_mode=SCALE_TO_CELL,
                cell_fill_percent=100,
                first_title_new_column=False,
                last_title_new_column=False,
                add_annotations=False,
            )
            boards = allocate_boards(
                parse_scripture("甲", index.characters),
                parameters,
            )

            result = generate_psd_boards(
                boards,
                index,
                parameters,
                str(output_dir),
            )

            psd = PSDImage.open(result.boards[0].path, encoding="gbk")
            layer = next(
                item for item in psd.descendants() if item.name.startswith("甲_")
            )
            self.assertEqual((layer.left, layer.top), (4, 4))
            self.assertEqual((layer.width, layer.height), (24, 24))

    def test_generate_uncompressed_psd_and_write_timing_logs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph_dir = root / "字图"
            output_dir = root / "输出"
            glyph_dir.mkdir()
            self._write_glyph(glyph_dir / "甲.png")
            index = build_external_glyph_index(str(glyph_dir))
            parameters = LayoutParameters(
                dpi=72,
                cell_width_mm=8,
                cell_height_mm=8,
                rows=1,
                columns=1,
                first_title_new_column=False,
                last_title_new_column=False,
                add_annotations=False,
            )
            boards = allocate_boards(
                parse_scripture("甲", index.characters),
                parameters,
            )
            progress: list[GenerationProgress] = []

            with patch(
                "services.scripture_layout_service.write_log"
            ) as write_log:
                result = generate_psd_boards(
                    boards,
                    index,
                    parameters,
                    str(output_dir),
                    compress_psd=False,
                    progress_callback=progress.append,
                )

            self.assertFalse(result.stopped)
            psd = PSDImage.open(result.boards[0].path, encoding="gbk")
            grid = compute_grid(
                parameters,
                boards[0].effective_columns,
                boards[0].effective_rows,
            )
            self.assertEqual(psd.size, (grid.canvas_width, grid.canvas_height))
            self.assertEqual(psd._record.image_data.compression, Compression.RAW)
            self.assertTrue(
                all(
                    channel.compression == Compression.RAW
                    for layer in psd.descendants()
                    if isinstance(layer, PixelLayer)
                    for channel in layer._channels
                )
            )
            messages = [str(call.args[0]) for call in write_log.call_args_list]
            self.assertTrue(any("PSD压缩=无压缩" in message for message in messages))
            self.assertTrue(any("PSD写盘=" in message for message in messages))
            self.assertTrue(any("兼容预览编码=" in message for message in messages))
            self.assertTrue(any("内存回收=" in message for message in messages))
            self.assertTrue(any("通用经文排版单版耗时" in message for message in messages))
            self.assertTrue(any("通用经文排版耗时汇总" in message for message in messages))
            self.assertTrue(
                any(
                    item.indeterminate and "无压缩 PSD" in item.message
                    for item in progress
                )
            )

    def test_cancel_before_first_board_and_skip_conflict(self) -> None:
        parameters = LayoutParameters(rows=1, columns=1)
        boards = allocate_boards(parse_scripture("甲"), parameters)
        cancel = threading.Event()
        cancel.set()
        with tempfile.TemporaryDirectory() as directory:
            result = generate_psd_boards(
                boards,
                GlyphIndex("空", {}),
                parameters,
                directory,
                cancel_event=cancel,
            )
            self.assertTrue(result.stopped)

            target = Path(board_output_path(directory, 1, parameters.dpi))
            target.write_bytes(b"existing")
            result = generate_psd_boards(
                boards,
                GlyphIndex("空", {}),
                parameters,
                directory,
                conflict_decisions={1: CONFLICT_SKIP},
            )
            self.assertTrue(result.boards[0].skipped)
            self.assertEqual(target.read_bytes(), b"existing")

    def test_cancel_during_generation_immediately_terminates_and_keeps_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph_dir = root / "字图"
            output_dir = root / "输出"
            glyph_dir.mkdir()
            output_dir.mkdir()
            self._write_glyph(glyph_dir / "甲.png")
            index = build_external_glyph_index(str(glyph_dir))
            parameters = LayoutParameters(
                dpi=72,
                cell_width_mm=3,
                cell_height_mm=3,
                rows=100,
                columns=50,
                row_gap_mm=0,
                column_gap_mm=0,
                first_title_new_column=False,
                last_title_new_column=False,
                add_annotations=False,
            )
            boards = allocate_boards(
                parse_scripture("甲" * 5000, index.characters),
                parameters,
            )
            target = Path(board_output_path(str(output_dir), 1, parameters.dpi))
            target.write_bytes(b"existing-target")
            cancel = threading.Event()
            timer = threading.Timer(0.2, cancel.set)
            started = time.perf_counter()
            timer.start()
            try:
                result = generate_psd_boards(
                    boards,
                    index,
                    parameters,
                    str(output_dir),
                    conflict_decisions={1: CONFLICT_OVERWRITE},
                    cancel_event=cancel,
                )
            finally:
                timer.cancel()

            self.assertTrue(result.stopped)
            self.assertLess(time.perf_counter() - started, 3.0)
            self.assertEqual(target.read_bytes(), b"existing-target")
            self.assertFalse(any(output_dir.glob("*.tmp.psd")))


if __name__ == "__main__":
    unittest.main()
