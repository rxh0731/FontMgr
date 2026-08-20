"""本轮安全、资源和导入风险的集中回归测试。"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import threading
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from core.scripture_layout import BoardLayout, LayoutParameters
from services.glyph_service import GlyphService
from services.import_service import ImportService
from services.scripture_layout_service import (
    OUTPUT_FORMAT_AUTO,
    OUTPUT_FORMAT_PSB,
    OUTPUT_FORMAT_PSD,
    GlyphIndex,
    board_output_path,
    generate_psd_boards,
    plan_board_output,
)
from services.scripture_text_service import load_scripture_text
from utils.crash_handler import setup_crash_handler
from utils.file_utils import (
    is_safe_windows_filename,
    resolve_library_directory,
    resolve_safe_child_file,
)
from ui.pages.home_page import scan_library_summaries
from ui.workers import FunctionWorker


class PathBoundaryTests(unittest.TestCase):
    def test_windows_names_and_child_paths_are_strict(self) -> None:
        for value in ("", ".", "..", "NUL", "CON.txt", "a/b", "a\\b", "尾点."):
            with self.subTest(value=value):
                self.assertFalse(is_safe_windows_filename(value))
        self.assertTrue(is_safe_windows_filename("小品0806"))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "测试库"
            stage = library / "04_自动优化稿"
            stage.mkdir(parents=True)
            image_path = stage / "甲-0001.png"
            image_path.write_bytes(b"png")
            self.assertEqual(
                resolve_library_directory(root, library, expected_name="测试库"),
                str(library),
            )
            self.assertEqual(resolve_safe_child_file(stage, image_path.name), str(image_path))
            self.assertEqual(resolve_safe_child_file(stage, "../甲-0001.png"), "")


class ImportRiskTests(unittest.TestCase):
    @staticmethod
    def _png(path: Path) -> str:
        Image.new("L", (12, 12), 0).save(path)
        return hashlib.md5(path.read_bytes()).hexdigest()

    def test_extension_b_character_and_scanned_digest_are_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "源图"
            library = root / "字库"
            source.mkdir()
            library.mkdir()
            image_path = source / "𠀀-原图.png"
            digest = self._png(image_path)
            stat = image_path.stat()
            service = GlyphService("字库", str(library))
            scanned = {
                str(image_path.resolve()): {
                    "md5": digest,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            }

            with patch(
                "services.import_service.compute_file_md5",
                side_effect=AssertionError("不应重复计算扫描阶段摘要"),
            ):
                result = ImportService(service).import_batch(
                    str(source),
                    scanned_files=scanned,
                )

            self.assertEqual(result["成功"], 1)
            detail = service.get_all_variants()[0]
            self.assertEqual(detail["归属字"], "𠀀")

    def test_registration_failure_removes_copied_original(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "源图"
            library = root / "字库"
            source.mkdir()
            library.mkdir()
            self._png(source / "甲.png")
            glyph = GlyphService("字库", str(library))

            with patch.object(glyph, "add_original", side_effect=RuntimeError("登记失败")):
                result = ImportService(glyph).import_batch(str(source))

            self.assertEqual(result["失败"], 1)
            original_dir = Path(glyph.get_workflow_dirs()["原图"])
            self.assertEqual(list(original_dir.iterdir()), [])
            self.assertEqual(glyph.get_all_variants(), [])

    def test_existing_orphan_file_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "源图"
            library = root / "字库"
            source.mkdir()
            library.mkdir()
            self._png(source / "甲.png")
            glyph = GlyphService("字库", str(library))
            glyph.ensure_dirs()
            original_dir = Path(glyph.get_workflow_dirs()["原图"])
            orphan = original_dir / "甲-0001.png"
            orphan_payload = "不可覆盖".encode("utf-8")
            orphan.write_bytes(orphan_payload)

            result = ImportService(glyph).import_batch(str(source))

            self.assertEqual(result["成功"], 1)
            self.assertEqual(orphan.read_bytes(), orphan_payload)
            self.assertTrue((original_dir / "甲-0002.png").is_file())


class ScriptureResourceTests(unittest.TestCase):
    def test_text_source_and_extracted_character_limits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "经文.txt"
            path.write_text("甲乙丙丁", encoding="utf-8")
            with patch("services.scripture_text_service.MAX_TEXT_SOURCE_BYTES", 3):
                with self.assertRaisesRegex(ValueError, "文本类文档大小超过"):
                    load_scripture_text(str(path))
            with patch("services.scripture_text_service.MAX_EXTRACTED_CHARACTERS", 3):
                with self.assertRaisesRegex(ValueError, "提取文字超过"):
                    load_scripture_text(str(path))

    def test_office_archive_member_limit_is_checked_before_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "经文.docx"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("word/document.xml", "<document />")
            with patch("services.scripture_text_service.MAX_ARCHIVE_MEMBERS", 0):
                with self.assertRaisesRegex(ValueError, "内部文件数量超过"):
                    load_scripture_text(str(path))

    def test_auto_selects_psb_and_forced_psd_rejects_oversize(self) -> None:
        parameters = LayoutParameters(
            dpi=300,
            cell_width_mm=900,
            cell_height_mm=1,
            rows=1,
            columns=3,
            row_gap_mm=0,
            column_gap_mm=0,
            draw_outer_frame=False,
            frame_top_mm=0,
            frame_bottom_mm=0,
            frame_left_mm=0,
            frame_right_mm=0,
            canvas_top_mm=0,
            canvas_bottom_mm=0,
            canvas_left_mm=0,
            canvas_right_mm=0,
            add_annotations=False,
        )
        board = BoardLayout(1, (), 3, 1, True)
        memory = (128 * 1024**3, 128 * 1024**3)

        plan = plan_board_output(
            board,
            GlyphIndex("空", {}),
            parameters,
            OUTPUT_FORMAT_AUTO,
            memory_status=memory,
        )

        self.assertEqual(plan.format_name, OUTPUT_FORMAT_PSB)
        self.assertEqual(plan.extension, ".psb")
        self.assertTrue(plan.psb)
        self.assertEqual(
            Path(
                board_output_path(
                    "输出",
                    1,
                    300,
                    "经文排版.psd",
                    extension=plan.extension,
                )
            ).suffix,
            ".psb",
        )
        with self.assertRaisesRegex(ValueError, "不能强制使用 PSD"):
            plan_board_output(
                board,
                GlyphIndex("空", {}),
                parameters,
                OUTPUT_FORMAT_PSD,
                memory_status=memory,
            )

    def test_preflight_warns_but_does_not_reject_high_memory_board(self) -> None:
        parameters = LayoutParameters(
            dpi=300,
            cell_width_mm=423,
            cell_height_mm=423,
            rows=1,
            columns=1,
            row_gap_mm=0,
            column_gap_mm=0,
            draw_outer_frame=False,
            frame_top_mm=0,
            frame_bottom_mm=0,
            frame_left_mm=0,
            frame_right_mm=0,
            canvas_top_mm=0,
            canvas_bottom_mm=0,
            canvas_left_mm=0,
            canvas_right_mm=0,
            add_annotations=False,
        )
        plan = plan_board_output(
            BoardLayout(1, (), 1, 1, True),
            GlyphIndex("空", {}),
            parameters,
            OUTPUT_FORMAT_AUTO,
            memory_status=(512 * 1024**2, 256 * 1024**2),
        )

        self.assertGreater(plan.estimated_peak_bytes, plan.memory_budget_bytes)
        self.assertIn("峰值内存", plan.memory_warning)
        self.assertIn("逐版", plan.memory_warning)

    def test_forced_psb_writes_version_two_header_and_psb_suffix(self) -> None:
        try:
            from psd_tools import PSDImage
        except ImportError:
            self.skipTest("未安装 psd-tools")
        parameters = LayoutParameters(
            dpi=72,
            cell_width_mm=10,
            cell_height_mm=10,
            rows=1,
            columns=1,
            row_gap_mm=0,
            column_gap_mm=0,
            draw_outer_frame=False,
            add_annotations=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            result = generate_psd_boards(
                (BoardLayout(1, (), 1, 1, True),),
                GlyphIndex("空", {}),
                parameters,
                directory,
                output_format=OUTPUT_FORMAT_PSB,
            )
            output = Path(result.boards[0].path)
            self.assertEqual(output.suffix, ".psb")
            psd = PSDImage.open(output)
            self.assertEqual(psd._record.header.version, 2)


class CrashHookTests(unittest.TestCase):
    def test_process_thread_and_unraisable_hooks_use_their_own_arguments(self) -> None:
        previous_process = sys.excepthook
        previous_thread = threading.excepthook
        previous_unraisable = sys.unraisablehook
        process_hook = Mock()
        thread_hook = Mock()
        unraisable_hook = Mock()
        with tempfile.TemporaryDirectory() as directory:
            log_path = str(Path(directory) / "崩溃.log")
            try:
                sys.excepthook = process_hook
                threading.excepthook = thread_hook
                sys.unraisablehook = unraisable_hook
                setup_crash_handler(log_path)
                error = RuntimeError("测试异常")
                sys.excepthook(RuntimeError, error, None)
                thread_args = types.SimpleNamespace(
                    exc_type=RuntimeError,
                    exc_value=error,
                    exc_traceback=None,
                    thread=None,
                )
                threading.excepthook(thread_args)
                unraisable_args = types.SimpleNamespace(
                    exc_type=RuntimeError,
                    exc_value=error,
                    exc_traceback=None,
                    err_msg=None,
                    object=None,
                )
                sys.unraisablehook(unraisable_args)
            finally:
                sys.excepthook = previous_process
                threading.excepthook = previous_thread
                sys.unraisablehook = previous_unraisable

            process_hook.assert_called_once_with(RuntimeError, error, None)
            thread_hook.assert_called_once_with(thread_args)
            unraisable_hook.assert_called_once_with(unraisable_args)
            self.assertGreaterEqual(Path(log_path).read_text(encoding="utf-8").count("测试异常"), 3)

    def test_function_worker_logs_full_traceback_but_emits_short_message(self) -> None:
        failures: list[str] = []
        worker = FunctionWorker(lambda: (_ for _ in ()).throw(RuntimeError("后台失败")))
        worker.signals.failed.connect(failures.append)
        with patch("ui.workers.write_log") as write_log:
            worker.run()

        self.assertEqual(failures, ["后台失败"])
        logged = write_log.call_args.args[0]
        self.assertIn("Traceback", logged)
        self.assertIn("RuntimeError: 后台失败", logged)


class HomeDamageTests(unittest.TestCase):
    def test_corrupt_library_is_reported_instead_of_hidden(self) -> None:
        import config

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "损坏字库"
            library.mkdir()
            (library / "损坏字库.json").write_text("{损坏", encoding="utf-8")
            (library / "损坏字库.json.bak").write_text("{仍损坏", encoding="utf-8")
            with patch.object(config, "ZIKU_ROOT", str(root)):
                summaries = scan_library_summaries()

        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["name"], "损坏字库")
        self.assertIn("损坏", summaries[0]["data_error"])


if __name__ == "__main__":
    unittest.main()
