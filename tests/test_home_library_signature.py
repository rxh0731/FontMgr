"""首页字库摘要轻量签名回归测试。"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import config
from ui.pages.home_page import (
    library_summary_signature,
    scan_library_names,
)


class HomeLibrarySignatureTests(unittest.TestCase):
    """验证首页可在不读取阶段图片的前提下判断摘要是否过期。"""

    @staticmethod
    def _advance_mtime(path: Path) -> None:
        current = path.stat().st_mtime_ns
        os.utime(path, ns=(current + 1_000_000_000, current + 1_000_000_000))

    def test_names_only_scan_root_and_use_pinyin_natural_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "测试10").mkdir()
            (root / "测试2").mkdir()
            nested = root / "测试2" / config.DIR_ORIGINAL_FILES
            nested.mkdir()
            (nested / "不应读取.png").write_bytes(b"image")
            (root / "不是字库.txt").write_text("忽略", encoding="utf-8")
            real_scandir = os.scandir
            scanned_paths: list[str] = []

            def tracking_scandir(path):
                scanned_paths.append(os.path.normcase(os.path.abspath(os.fspath(path))))
                return real_scandir(path)

            with (
                patch.object(config, "ZIKU_ROOT", str(root)),
                patch("ui.pages.home_page.os.scandir", side_effect=tracking_scandir),
            ):
                names = scan_library_names()
                signature = library_summary_signature()

            self.assertEqual(names, ["测试2", "测试10"])
            self.assertEqual(
                [library.name for library in signature.libraries],
                names,
            )
            normalized_root = os.path.normcase(os.path.abspath(temp_dir))
            self.assertEqual(scanned_paths, [normalized_root, normalized_root])

    def test_missing_root_has_stable_empty_signature(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_root = Path(temp_dir) / "不存在"
            with patch.object(config, "ZIKU_ROOT", str(missing_root)):
                first = library_summary_signature()
                second = library_summary_signature()
                names = scan_library_names()

            self.assertEqual(first, second)
            self.assertFalse(first.root.exists)
            self.assertEqual(first.libraries, ())
            self.assertEqual(names, [])

    def test_json_backup_and_library_directory_changes_invalidate_signature(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            library = root / "甲字库"
            library.mkdir()
            json_path = library / "甲字库.json"
            backup_path = library / "甲字库.json.bak"
            json_path.write_text("{}", encoding="utf-8")
            with patch.object(config, "ZIKU_ROOT", str(root)):
                initial = library_summary_signature()

                json_path.write_text('{"版本": 2}', encoding="utf-8")
                json_changed = library_summary_signature()

                backup_path.write_text("{}", encoding="utf-8")
                backup_added = library_summary_signature()

                added_library = root / "乙字库"
                added_library.mkdir()
                library_added = library_summary_signature()

                added_library.rename(root / "丙字库")
                library_renamed = library_summary_signature()

            self.assertNotEqual(initial, json_changed)
            self.assertNotEqual(json_changed, backup_added)
            self.assertNotEqual(backup_added, library_added)
            self.assertNotEqual(library_added, library_renamed)

    def test_current_and_legacy_stage_directories_invalidate_signature(self) -> None:
        stage_names = (
            config.DIR_ORIGINAL_FILES,
            "原始文件",
            config.DIR_INTERMEDIATE_FILES,
            "中间文件",
            config.DIR_REVIEWED_FILES,
            "审核文件",
            config.DIR_FINISHED_FILES,
            "成品文件",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            library = root / "阶段字库"
            library.mkdir()
            (library / "阶段字库.json").write_text("{}", encoding="utf-8")
            with patch.object(config, "ZIKU_ROOT", str(root)):
                previous = library_summary_signature()
                for stage_name in stage_names:
                    with self.subTest(stage_name=stage_name):
                        stage_dir = library / stage_name
                        stage_dir.mkdir()
                        created = library_summary_signature()
                        self.assertNotEqual(previous, created)

                        self._advance_mtime(stage_dir)
                        touched = library_summary_signature()
                        self.assertNotEqual(created, touched)
                        previous = touched


if __name__ == "__main__":
    unittest.main()
