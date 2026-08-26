"""字库导出三栏工作台回归测试。"""

from __future__ import annotations

from contextlib import contextmanager
import os
import tempfile
import threading
import time
import unittest
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image, ImageDraw
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QHeaderView, QMessageBox, QTreeWidget

import config
from services.adjustment_service import AdjustmentService
from services.export_service import (
    ExportConflict,
    ExportConflictDecision,
    ExportOptions,
    ExportService,
)
from services.glyph_service import GlyphService
from services.settings_service import ApplicationSettings
import ui.pages.export_page as export_page_module
from ui.pages.export_page import ExportPage
from ui.theme import apply_theme
from ui.widgets.export_gallery import ExportGallery


class ExportPageTests(unittest.TestCase):
    """锁定导出页面布局、选项、状态与后台任务契约。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])
        apply_theme(cls.app)

    def test_export_directory_uses_program_setting(self) -> None:
        with tempfile.TemporaryDirectory() as output_directory, patch(
            "ui.pages.export_page.SettingsService.load",
            return_value=ApplicationSettings(
                default_export_directory=output_directory
            ),
        ):
            with self._page_with_variants(total=1) as (page, _ids, _root):
                self.assertEqual(page._directory_edit.text(), output_directory)

    @contextmanager
    def _page_with_variants(
        self,
        *,
        total: int = 18,
        finished: int = 18,
        summary_completed: bool = True,
        characters: str | None = None,
        canvas_w: int = 96,
        canvas_h: int = 112,
    ) -> Iterator[tuple[ExportPage, list[str], Path]]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library_dir = root / "导出页面测试"
            library_dir.mkdir()
            glyph = GlyphService("导出页面测试", str(library_dir))
            glyph.ensure_dirs()
            glyph.init_metadata(dpi=600, canvas_w=canvas_w, canvas_h=canvas_h)
            final_dir = Path(glyph.get_workflow_dirs()["成品"])
            preview_dir = Path(glyph.get_workflow_dirs()["优化预览"])
            variant_ids: list[str] = []
            chars = characters or "天地玄黄宇宙洪荒日月盈昃辰宿列张寒来暑往"
            if len(chars) < total:
                raise ValueError("测试字符数量不能少于字形总数。")
            for index in range(total):
                char = chars[index]
                filename = f"{char}-{index + 1:04d}.png"
                variant_id = glyph.add_original(
                    char,
                    filename,
                    filename,
                    f"export-page-{index:04d}",
                )
                variant_ids.append(variant_id)
                detail = glyph.get_variant(variant_id)
                if index >= finished:
                    preview = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
                    preview_draw = ImageDraw.Draw(preview)
                    preview_draw.rectangle(
                        (12, 12, canvas_w - 13, canvas_h - 13),
                        fill=(20, 20, 20, 220),
                    )
                    preview.save(preview_dir / filename)
                    detail["中间文件"] = filename
                    detail["状态"] = config.STATUS_REVIEWED
                    continue
                image = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
                draw = ImageDraw.Draw(image)
                inset = 12 + index % 5
                draw.rectangle(
                    (
                        inset,
                        inset,
                        canvas_w - 1 - inset,
                        canvas_h - 1 - inset,
                    ),
                    fill=(20, 20, 20, 180 + index % 70),
                )
                image.save(final_dir / filename, dpi=(600, 600))
                glyph.mark_finished(
                    variant_id,
                    filename,
                    "",
                    {
                        "标准画布": [canvas_w, canvas_h],
                        "实际画布": [canvas_w, canvas_h],
                        "整体变换": {},
                        "墨色协调": {
                            "启用": True,
                            "基准": 210.0,
                            "方法": AdjustmentService.INK_METHOD,
                            "方法版本": AdjustmentService.INK_METHOD_VERSION,
                            "保存后墨色": 210.0,
                            "保存后复测": True,
                            "是否达标": True,
                            "状态": "已达标",
                        },
                    },
                )
            glyph.set_coordination_summary(
                {},
                210.0,
                geometry_completed=summary_completed,
                ink_completed=summary_completed,
                ink_enabled=True,
                ink_method=AdjustmentService.INK_METHOD,
                ink_method_version=AdjustmentService.INK_METHOD_VERSION,
                ink_counts={
                    "总数": total,
                    "已达标": total if summary_completed else 0,
                    "待确认": 0 if summary_completed else total,
                    "人工例外": 0,
                },
            )
            glyph.save()

            page = ExportPage(glyph)
            page.resize(1380, 880)
            page.show()
            for _ in range(12):
                self.app.processEvents()
            try:
                yield page, variant_ids, root
            finally:
                page.shutdown()
                page.close()
                page.deleteLater()
                for _ in range(3):
                    self.app.processEvents()

    @staticmethod
    def _gallery_entry_count(gallery: ExportGallery) -> int:
        for name in ("entry_count", "count"):
            value = getattr(gallery, name, None)
            if callable(value):
                return int(value())
            if value is not None:
                return int(value)
        entries = getattr(gallery, "_entries", None)
        if entries is None:
            raise AssertionError("ExportGallery 未暴露条目数量。")
        return len(entries)

    @staticmethod
    def _gallery_columns(gallery: ExportGallery) -> int:
        value = getattr(gallery, "column_count", None)
        if callable(value):
            return int(value())
        if value is not None:
            return int(value)
        return int(getattr(gallery, "_column_count"))

    def test_grouped_list_matches_optimization_style_and_full_gallery(self) -> None:
        with self._page_with_variants() as (page, _variant_ids, _root):
            self.assertEqual(page._main_splitter.count(), 3)
            tree = page._glyph_list
            self.assertIsInstance(tree, QTreeWidget)
            self.assertEqual(tree.columnCount(), 4)
            self.assertEqual(
                [tree.headerItem().text(index) for index in range(4)],
                ["字形与文件", "协调状态", "提示", "导出"],
            )
            self.assertEqual(page._filter_combo.currentText(), "全部（本阶段）")
            self.assertEqual(
                [page._filter_combo.itemText(index) for index in range(page._filter_combo.count())],
                ["全部（本阶段）", "待协调", "已协调"],
            )
            self.assertFalse(hasattr(page, "_problem_filter_combo"))
            self.assertTrue(tree.rootIsDecorated())
            self.assertFalse(tree.uniformRowHeights())
            self.assertFalse(tree.alternatingRowColors())
            self.assertTrue(tree.wordWrap())
            self.assertEqual(tree.iconSize(), QSize(38, 38))
            header = tree.header()
            self.assertEqual(
                header.sectionResizeMode(0),
                QHeaderView.ResizeMode.Stretch,
            )
            self.assertEqual(
                header.sectionResizeMode(1),
                QHeaderView.ResizeMode.Fixed,
            )
            self.assertEqual(
                header.sectionResizeMode(2),
                QHeaderView.ResizeMode.Fixed,
            )
            self.assertEqual(
                header.sectionResizeMode(3),
                QHeaderView.ResizeMode.Fixed,
            )
            self.assertEqual(tree.topLevelItemCount(), 18)
            first_group = tree.topLevelItem(0)
            first_detail = page._visible_variants[0]
            first_char = str(first_detail["归属字"])
            first_filename = str(first_detail["原始文件"])
            self.assertEqual(first_group.text(0), f"{first_char}（1个字形）")
            self.assertEqual(first_group.text(1), "已协调 1/1")
            self.assertEqual(first_group.text(2), "问题 0")
            self.assertEqual(first_group.text(3), "可导出 1/1")
            self.assertFalse(first_group.flags() & Qt.ItemFlag.ItemIsSelectable)
            self.assertTrue(first_group.isExpanded())
            self.assertEqual(first_group.childCount(), 1)
            self.assertEqual(
                first_group.child(0).text(0),
                f"字形1 · {first_filename}",
            )
            self.assertEqual(first_group.child(0).text(1), "已协调")
            self.assertEqual(first_group.child(0).text(2), "无")
            self.assertEqual(first_group.child(0).text(3), "可导出")
            self.assertEqual(first_group.child(0).sizeHint(0).height(), 52)
            self.assertIn("整体协调：已协调", first_group.child(0).toolTip(0))
            self.assertNotIn("阶段：", first_group.child(0).toolTip(0))
            self.assertEqual(page._list_count_label.text(), "显示 / 本阶段：18 / 18")
            self.assertIsInstance(page._gallery, ExportGallery)
            self.assertEqual(self._gallery_entry_count(page._gallery), 18)
            first_entry = page._gallery._gallery_model.entry_at(0)
            self.assertIsNotNone(first_entry)
            assert first_entry is not None
            self.assertEqual(first_entry.status, "已协调")
            delegate = page._gallery.itemDelegate()
            self.assertEqual(delegate.status_color("待协调"), QColor("#4169E1"))
            self.assertEqual(delegate.status_color("已协调"), QColor("#228B22"))
            self.assertEqual(page._column_spin.value(), 8)
            self.assertEqual(self._gallery_columns(page._gallery), 8)

            page._column_spin.setValue(6)
            self.app.processEvents()
            self.assertEqual(self._gallery_columns(page._gallery), 6)
            self.assertEqual(self._gallery_entry_count(page._gallery), 18)

    def test_gallery_receives_square_and_rectangular_library_canvas_sizes(self) -> None:
        for width, height in ((250, 250), (300, 200)):
            with self.subTest(canvas=(width, height)):
                with self._page_with_variants(
                    total=1,
                    finished=0,
                    summary_completed=False,
                    canvas_w=width,
                    canvas_h=height,
                ) as (page, _variant_ids, _root):
                    self.assertEqual(page._gallery.canvas_size, QSize(width, height))
                    entry = page._gallery._gallery_model.entry_at(0)
                    self.assertIsNotNone(entry)
                    assert entry is not None
                    self.assertIsNone(entry.image_canvas_size)

                    detail = page._visible_variants[0]
                    detail["整体协调参数"] = {
                        "标准画布": [width, height],
                        "实际画布": [width + 20, height + 10],
                    }
                    page._populate_gallery()
                    entry = page._gallery._gallery_model.entry_at(0)
                    self.assertIsNotNone(entry)
                    assert entry is not None
                    self.assertEqual(
                        entry.image_canvas_size,
                        (width + 20, height + 10),
                    )

    def test_same_character_variants_share_parent_and_keep_child_statistics(self) -> None:
        with self._page_with_variants(
            total=5,
            finished=3,
            summary_completed=False,
            characters="甲甲甲乙乙",
        ) as (page, variant_ids, _root):
            tree = page._glyph_list
            self.assertEqual(tree.topLevelItemCount(), 2)
            first_group = tree.topLevelItem(0)
            second_group = tree.topLevelItem(1)
            self.assertEqual(first_group.text(0), "甲（3个字形）")
            self.assertEqual(first_group.childCount(), 3)
            self.assertEqual(first_group.text(1), "已协调 3/3")
            self.assertEqual(first_group.text(2), "问题 0")
            self.assertEqual(first_group.text(3), "可导出 3/3")
            self.assertEqual(
                [first_group.child(index).text(0) for index in range(3)],
                [
                    "字形1 · 甲-0001.png",
                    "字形2 · 甲-0002.png",
                    "字形3 · 甲-0003.png",
                ],
            )
            self.assertEqual(second_group.text(0), "乙（2个字形）")
            self.assertEqual(second_group.childCount(), 2)
            self.assertEqual(second_group.text(1), "已协调 0/2")
            self.assertEqual(second_group.text(2), "问题 0")
            self.assertEqual(second_group.text(3), "可导出 0/2")
            self.assertEqual(page._list_count_label.text(), "显示 / 本阶段：5 / 5")

            page._search_edit.setText("甲-0002")
            self.app.processEvents()
            self.assertEqual(tree.topLevelItemCount(), 1)
            matched_group = tree.topLevelItem(0)
            self.assertEqual(matched_group.text(0), "甲（3个字形）")
            self.assertEqual(matched_group.childCount(), 1)
            self.assertEqual(
                matched_group.child(0).data(0, Qt.ItemDataRole.UserRole),
                variant_ids[1],
            )
            self.assertEqual(page._list_count_label.text(), "显示 / 本阶段：1 / 5")

    def test_filtering_and_selection_stay_synchronized(self) -> None:
        with self._page_with_variants(total=8, finished=5, summary_completed=False) as (
            page,
            variant_ids,
            _root,
        ):
            page._filter_combo.setCurrentText("待协调")
            self.app.processEvents()
            self.assertEqual(page._glyph_list.topLevelItemCount(), 3)
            self.assertEqual(self._gallery_entry_count(page._gallery), 3)
            self.assertEqual(page._list_count_label.text(), "显示 / 本阶段：3 / 8")
            for row in range(3):
                entry = page._gallery._gallery_model.entry_at(row)
                self.assertIsNotNone(entry)
                assert entry is not None
                self.assertEqual(entry.status, "待协调")
            for row in range(page._glyph_list.topLevelItemCount()):
                parent = page._glyph_list.topLevelItem(row)
                self.assertEqual(parent.text(1), "已协调 0/1")
                item = parent.child(0)
                self.assertEqual(item.text(1), "待协调")
                self.assertEqual(item.text(2), "无")
                self.assertEqual(item.text(3), "不可导出")
                self.assertEqual(item.foreground(1).color(), QColor("#4169E1"))

            page._filter_combo.setCurrentText("全部（本阶段）")
            self.app.processEvents()
            target_item = page._items_by_id[variant_ids[2]]
            page._glyph_list.setCurrentItem(target_item)
            self.app.processEvents()
            self.assertEqual(page._selected_id, variant_ids[2])

            page._gallery.variant_selected.emit(variant_ids[4])
            self.app.processEvents()
            self.assertEqual(page._selected_id, variant_ids[4])
            self.assertIs(page._glyph_list.currentItem(), page._items_by_id[variant_ids[4]])

            selected_item = page._items_by_id[variant_ids[4]]
            parent = selected_item.parent()
            self.assertIsNotNone(parent)
            page._glyph_list.setCurrentItem(parent)
            self.app.processEvents()
            self.assertIs(page._glyph_list.currentItem(), selected_item)
            self.assertEqual(page._selected_id, variant_ids[4])

    def test_left_list_loads_only_visible_real_thumbnails_in_background(self) -> None:
        original_decode = export_page_module.decode_thumbnail_image
        main_thread_id = threading.get_ident()
        decode_calls: list[tuple[int, str]] = []

        def tracked_decode(*args: object, **kwargs: object):
            decode_calls.append((threading.get_ident(), str(args[0])))
            return original_decode(*args, **kwargs)

        with patch.object(
            export_page_module,
            "decode_thumbnail_image",
            side_effect=tracked_decode,
        ):
            with self._page_with_variants(
                total=60,
                characters="甲" * 60,
            ) as (page, variant_ids, _root):
                self.assertEqual(page._glyph_list.topLevelItemCount(), 1)
                deadline = time.monotonic() + 3.0
                while not page._thumbnail_cache and time.monotonic() < deadline:
                    self.app.processEvents()
                    time.sleep(0.01)
                self.assertTrue(page._thumbnail_cache)
                self.assertTrue(decode_calls)
                self.assertTrue(
                    all(thread_id != main_thread_id for thread_id, _path in decode_calls)
                )
                self.assertLess(
                    len({path for _thread_id, path in decode_calls}),
                    len(variant_ids),
                )
                self.assertLess(len(page._thumbnail_cache), len(variant_ids))
                loaded_id = next(iter(page._thumbnail_cache))
                child = page._items_by_id[loaded_id]
                cached_icon = page._thumbnail_cache[loaded_id][1]
                self.assertEqual(child.icon(0).cacheKey(), cached_icon.cacheKey())
                thumbnail = cached_icon.pixmap(page._glyph_list.iconSize()).toImage()
                self.assertEqual(thumbnail.pixelColor(0, 0), QColor("white"))
                self.assertLess(thumbnail.pixelColor(19, 19).lightness(), 180)
                self.assertLessEqual(
                    len(page._thumbnail_cache),
                    page.LIST_THUMBNAIL_CACHE_ITEMS,
                )

    def test_coordination_projection_excludes_upstream_records(self) -> None:
        with self._page_with_variants(
            total=4,
            finished=1,
            summary_completed=False,
        ) as (page, variant_ids, _root):
            optimization_id, review_id, coordination_id = variant_ids[1:]
            optimization_detail = page._glyph.get_variant(optimization_id)
            review_detail = page._glyph.get_variant(review_id)
            original_dir = Path(page._glyph.get_workflow_dirs()["原图"])
            preview_dir = Path(page._glyph.get_workflow_dirs()["优化预览"])
            optimization_filename = str(optimization_detail["原始文件"])
            with Image.open(preview_dir / optimization_filename) as source:
                source.save(original_dir / optimization_filename)
            optimization_detail["中间文件"] = ""
            optimization_detail["状态"] = config.STATUS_PENDING_OPTIMIZATION
            review_detail["状态"] = config.STATUS_PENDING_MANUAL_REVIEW
            review_detail["自动优化"] = {
                "方案": {"结构复核": {"状态": "需人工核对"}}
            }
            page._glyph.get_variant(coordination_id)["状态"] = config.STATUS_REVIEWED
            page._glyph.save()
            page._reload_variants()
            self.app.processEvents()

            expected_ids = {
                "待协调": coordination_id,
                "已协调": variant_ids[0],
            }
            for stage, expected_id in expected_ids.items():
                page._filter_combo.setCurrentText(stage)
                self.app.processEvents()
                self.assertEqual(
                    [str(item["变体ID"]) for item in page._visible_variants],
                    [expected_id],
                )

            page._filter_combo.setCurrentText("全部（本阶段）")
            self.assertEqual(
                {str(item["变体ID"]) for item in page._visible_variants},
                {coordination_id, variant_ids[0]},
            )
            self.assertNotIn(optimization_id, page._items_by_id)
            self.assertNotIn(review_id, page._items_by_id)
            self.assertFalse(hasattr(page, "_problem_filter_combo"))

    def test_incomplete_variant_uses_latest_real_stage_thumbnail(self) -> None:
        with self._page_with_variants(
            total=2,
            finished=0,
            summary_completed=False,
        ) as (page, variant_ids, _root):
            variant_id = variant_ids[0]
            detail = page._details_by_id[variant_id]
            filename = "甲-人工稿.png"
            reviewed_dir = Path(page._glyph.get_workflow_dirs()["手工审核"])
            image = Image.new("RGBA", (96, 112), (0, 0, 0, 0))
            ImageDraw.Draw(image).rectangle((22, 18, 74, 94), fill=(15, 15, 15, 255))
            image.save(reviewed_dir / filename)
            detail["审核文件"] = filename

            page._apply_filters()
            deadline = time.monotonic() + 3.0
            while variant_id not in page._thumbnail_cache and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.01)

            self.assertIn(variant_id, page._thumbnail_cache)
            item = page._items_by_id[variant_id]
            thumbnail = item.icon(0).pixmap(page._glyph_list.iconSize()).toImage()
            self.assertEqual(item.text(1), "待协调")
            self.assertEqual(item.text(2), "无")
            self.assertEqual(item.text(3), "不可导出")
            self.assertLess(thumbnail.pixelColor(19, 19).lightness(), 80)

    def test_coordination_projection_is_resolved_once_per_filter_cycle(self) -> None:
        with self._page_with_variants(total=12) as (page, _variant_ids, _root):
            page._workflow_status_cache.clear()
            with patch.object(
                export_page_module,
                "project_stage_status",
                wraps=export_page_module.project_stage_status,
            ) as resolver:
                page._apply_filters()
                self.assertEqual(resolver.call_count, len(page._all_variants))
                page._populate_list()
                page._populate_gallery()
                self.assertEqual(resolver.call_count, len(page._all_variants))

    def test_thumbnail_rejects_unsafe_or_missing_high_priority_file(self) -> None:
        with self._page_with_variants(total=2) as (page, variant_ids, _root):
            detail = page._details_by_id[variant_ids[0]]
            final_dir = Path(page._glyph.get_workflow_dirs()["成品"])
            Image.new("RGBA", (96, 112), (10, 10, 10, 255)).save(
                final_dir / "同名合法文件.png"
            )

            detail["成品文件"] = "../同名合法文件.png"
            self.assertIsNone(page._list_thumbnail_source(detail))

            detail["成品文件"] = "不存在的成品.png"
            self.assertIsNone(page._list_thumbnail_source(detail))

    def test_thumbnail_cache_invalidates_when_same_path_is_overwritten(self) -> None:
        with self._page_with_variants(total=2) as (page, variant_ids, _root):
            variant_id = variant_ids[0]
            deadline = time.monotonic() + 3.0
            while variant_id not in page._thumbnail_cache and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.01)
            self.assertIn(variant_id, page._thumbnail_cache)
            old_signature = page._thumbnail_cache[variant_id][0]

            detail = page._details_by_id[variant_id]
            finished_dir = Path(page._glyph.get_workflow_dirs()["成品"])
            image_path = finished_dir / str(detail["成品文件"])
            replacement = Image.new("RGBA", (96, 112), (255, 255, 255, 255))
            ImageDraw.Draw(replacement).rectangle(
                (22, 18, 74, 94),
                fill=(220, 20, 20, 255),
            )
            replacement.save(image_path)
            timestamp = time.time_ns() + 1_000_000_000
            os.utime(image_path, ns=(timestamp, timestamp))

            page._apply_filters()
            deadline = time.monotonic() + 3.0
            while (
                page._thumbnail_cache.get(variant_id, (old_signature,))[0]
                == old_signature
                and time.monotonic() < deadline
            ):
                self.app.processEvents()
                time.sleep(0.01)

            self.assertNotEqual(page._thumbnail_cache[variant_id][0], old_signature)
            thumbnail = page._items_by_id[variant_id].icon(0).pixmap(
                page._glyph_list.iconSize()
            ).toImage()
            center = thumbnail.pixelColor(19, 19)
            self.assertGreater(center.red(), 180)
            self.assertLess(center.green(), 80)

    def test_three_export_modes_build_expected_options(self) -> None:
        with self._page_with_variants(total=4) as (page, _variant_ids, _root):
            options = page._build_options()
            self.assertIsInstance(options, ExportOptions)
            self.assertEqual(options.mode, ExportService.MODE_LIBRARY_SPEC)
            self.assertFalse(page._custom_panel.isVisible())
            self.assertTrue(page._library_spec_label.isVisible())

            page._mode_buttons[ExportService.MODE_TRIM_TRANSPARENT].click()
            self.app.processEvents()
            options = page._build_options()
            self.assertEqual(options.mode, ExportService.MODE_TRIM_TRANSPARENT)
            self.assertFalse(options.include_transparent_area)
            self.assertIn("裁掉", page._option_summary_label.text())

            page._mode_buttons[ExportService.MODE_CUSTOM_SPEC].click()
            page._dpi_spin.setValue(1200)
            page._width_spin.setValue(640)
            page._height_spin.setValue(720)
            page._include_transparent_check.setChecked(False)
            page._name_mode_combo.setCurrentIndex(1)
            self.app.processEvents()
            options = page._build_options()
            self.assertEqual(options.mode, ExportService.MODE_CUSTOM_SPEC)
            self.assertEqual(options.dpi, 1200)
            self.assertEqual(options.width, 640)
            self.assertEqual(options.height, 720)
            self.assertFalse(options.include_transparent_area)
            self.assertEqual(options.name_mode, "原文件名")
            self.assertTrue(page._custom_panel.isVisible())
            self.assertFalse(page._library_spec_label.isVisible())
            self.assertIn("实际文字（不包含透明区）", page._option_summary_label.text())

    def test_readiness_badge_requires_files_and_coordination_summary(self) -> None:
        with self._page_with_variants(total=5) as (page, _variant_ids, _root):
            self.assertEqual(page._readiness_label.text(), "全库可导出")
            self.assertIn("5 / 5", page._readiness_badge.toolTip())
            self.assertEqual(
                page._summary_label.text(),
                "待协调 0　已协调 5\n可导出 5 / 5",
            )
            self.assertEqual(page._progress_bar.value(), 100)
            self.assertIn("#48c78e", page._readiness_dot.styleSheet().lower())

        with self._page_with_variants(
            total=5,
            finished=4,
            summary_completed=False,
        ) as (page, _variant_ids, _root):
            self.assertEqual(page._readiness_label.text(), "全库尚不可导出")
            self.assertEqual(
                page._summary_label.text(),
                "待协调 1　已协调 4\n可导出 4 / 5",
            )
            self.assertEqual(page._progress_bar.value(), 80)
            self.assertIn("#e36a6a", page._readiness_dot.styleSheet().lower())
            tooltip = page._readiness_badge.toolTip()
            self.assertIn("4 / 5", tooltip)
            self.assertIn("整体协调", tooltip)

    def test_verified_audit_marks_corrupt_finished_file_as_incomplete(self) -> None:
        with self._page_with_variants(total=3) as (page, variant_ids, root):
            page._audit_finished(
                {
                    "就绪": False,
                    "总数": 3,
                    "已就绪": 2,
                    "原因": ["1 个成品损坏"],
                    "问题详情": [
                        {
                            "变体ID": variant_ids[1],
                            "类型": "成品损坏",
                            "说明": "无法解码",
                        }
                    ],
                },
                next(iter(page._audit_workers), None),
            )
            self.app.processEvents()

            item = page._items_by_id[variant_ids[1]]
            self.assertEqual(item.text(1), "已协调")
            self.assertEqual(item.text(2), "文件异常")
            self.assertEqual(item.text(3), "不可导出")
            self.assertEqual(item.foreground(1).color(), QColor("#228B22"))
            self.assertEqual(item.foreground(2).color(), QColor("#E36A6A"))
            self.assertEqual(
                page._summary_label.text(),
                "待协调 0　已协调 3\n可导出 2 / 3",
            )
            self.assertEqual(page._readiness_label.text(), "全库尚不可导出")

            output_dir = root / "损坏成品部分输出"
            output_dir.mkdir()
            page._directory_edit.setText(str(output_dir))
            started: list[object] = []
            with (
                patch.object(page, "_confirm_partial_export", return_value=True),
                patch.object(page._thread_pool, "start", side_effect=started.append),
            ):
                page._start_export()
            worker = started[0]
            self.assertNotIn(variant_ids[1], worker._eligible_variant_ids)
            self.assertEqual(
                worker._eligible_variant_ids,
                frozenset((variant_ids[0], variant_ids[2])),
            )
            page._active_worker = None

    def test_verified_audit_keeps_safe_ink_pending_product_exportable(self) -> None:
        with self._page_with_variants(total=3) as (page, variant_ids, _root):
            detail = page._glyph.get_variant(variant_ids[1])
            ink_record = detail["整体协调参数"]["墨色协调"]
            ink_record["方法版本"] = 0
            page._glyph.save()

            page._workflow_summary = page._glyph.get_coordination_summary()
            page._workflow_status_cache.clear()
            audit = ExportService(page._glyph).audit_readiness(verify_hash=True)
            page._audit_finished(audit, None)
            self.app.processEvents()

            item = page._items_by_id[variant_ids[1]]
            self.assertEqual(item.text(1), "待协调")
            self.assertEqual(item.text(2), "墨色待确认")
            self.assertEqual(item.text(3), "可导出")
            self.assertTrue(page._variant_ready[variant_ids[1]])
            self.assertNotIn(variant_ids[1], page._audit_issue_ids)
            self.assertEqual(
                page._summary_label.text(),
                "待协调 1　已协调 2\n可导出 3 / 3",
            )
            self.assertEqual(page._readiness_label.text(), "全库尚不可导出")

    def test_readiness_worker_enables_hash_verification_and_cancellation(self) -> None:
        with self._page_with_variants(total=2) as (page, _variant_ids, _root):
            started: list[object] = []
            expected = page._fast_audit()
            with (
                patch.object(
                    ExportService,
                    "audit_readiness",
                    return_value=expected,
                ) as audit,
                patch.object(page._thread_pool, "start", side_effect=started.append),
            ):
                page._start_readiness_audit()
                self.assertTrue(page._audit_in_progress)
                self.assertFalse(page._export_button.isEnabled())
                started[0].run()
                self.app.processEvents()

            audit.assert_called_once()
            _args, kwargs = audit.call_args
            self.assertIs(kwargs["verify_hash"], True)
            self.assertTrue(callable(kwargs["cancel_check"]))
            self.assertFalse(page._audit_in_progress)

    def test_background_export_exposes_progress_cancel_and_completion(self) -> None:
        with self._page_with_variants(total=4) as (page, _variant_ids, root):
            output_dir = root / "输出"
            output_dir.mkdir()
            page._directory_edit.setText(str(output_dir))

            started: list[object] = []
            with patch.object(page._thread_pool, "start", side_effect=started.append):
                page._start_export()

            self.assertEqual(len(started), 1)
            worker = started[0]
            self.assertIs(worker, page._active_worker)
            self.assertTrue(page._cancel_button.isVisible())
            self.assertTrue(page._export_progress.isVisible())
            self.assertFalse(page._options_host.isEnabled())
            self.assertEqual(worker._options.mode, ExportService.MODE_LIBRARY_SPEC)
            self.assertEqual(worker._eligible_variant_ids, frozenset(_variant_ids))

            worker.signals.progress.emit("导出：天", 2, 4)
            self.app.processEvents()
            self.assertEqual(page._export_progress.maximum(), 4)
            self.assertEqual(page._export_progress.value(), 2)
            self.assertEqual(page._export_status_label.text(), "导出：天")

            page.cancel_export()
            self.assertTrue(page._cancel_event.is_set())
            self.assertFalse(page._cancel_button.isEnabled())

            with patch.object(QMessageBox, "information"):
                worker.signals.finished.emit(
                    {
                        "成功": 0,
                        "跳过": 0,
                        "失败": 0,
                        "已取消": True,
                    }
                )
                self.app.processEvents()
            self.assertIsNone(page._active_worker)
            self.assertFalse(page._cancel_button.isVisible())
            self.assertFalse(page._export_progress.isVisible())
            self.assertIn("已取消", page._export_status_label.text())

    def test_conflict_resolver_applies_checked_action_to_remaining_items(self) -> None:
        with self._page_with_variants(total=1) as (page, _variant_ids, root):
            conflicts = [
                ExportConflict(
                    variant_id=f"冲突-{index}",
                    char="甲",
                    destination_name=f"甲-{index}.png",
                    destination_path=str(root / f"甲-{index}.png"),
                    file_size=index,
                    modified_ns=index,
                    device=1,
                    inode=index,
                )
                for index in range(1, 4)
            ]
            with patch.object(
                page,
                "_ask_export_conflict",
                side_effect=(
                    (ExportService.CONFLICT_OVERWRITE, False),
                    (ExportService.CONFLICT_SKIP, True),
                ),
            ) as ask:
                decisions = page._resolve_export_conflicts(conflicts)

            self.assertIsNotNone(decisions)
            assert decisions is not None
            self.assertEqual(ask.call_count, 2)
            self.assertEqual(
                [decision.action for decision in decisions],
                [
                    ExportService.CONFLICT_OVERWRITE,
                    ExportService.CONFLICT_SKIP,
                    ExportService.CONFLICT_SKIP,
                ],
            )

    def test_conflict_dialog_uses_chinese_actions_apply_all_and_safe_default(self) -> None:
        with self._page_with_variants(total=1) as (page, _variant_ids, root):
            conflict = ExportConflict(
                variant_id="冲突字形",
                char="甲",
                destination_name="甲.png",
                destination_path=str(root / "甲.png"),
                file_size=8,
                modified_ns=10,
                device=1,
                inode=2,
            )
            captured: dict[str, object] = {}

            def choose_skip(dialog: QMessageBox) -> int:
                buttons = {button.text(): button for button in dialog.buttons()}
                checkbox = dialog.checkBox()
                captured["title"] = dialog.windowTitle()
                captured["buttons"] = set(buttons)
                captured["checkbox"] = checkbox.text() if checkbox else ""
                captured["default"] = dialog.defaultButton().text()
                captured["escape"] = dialog.escapeButton().text()
                assert checkbox is not None
                checkbox.setChecked(True)
                buttons["跳过"].click()
                return 0

            with patch.object(QMessageBox, "exec", new=choose_skip):
                response = page._ask_export_conflict(conflict, 1, 3)

            self.assertEqual(
                response,
                (ExportService.CONFLICT_SKIP, True),
            )
            self.assertEqual(captured["title"], "发现同名文件")
            self.assertEqual(captured["buttons"], {"覆盖", "跳过", "取消"})
            self.assertEqual(captured["checkbox"], "为所有项目执行此操作")
            self.assertEqual(captured["default"], "取消")
            self.assertEqual(captured["escape"], "取消")

    def test_conflict_cancel_stops_before_worker_and_decisions_reach_worker(self) -> None:
        with self._page_with_variants(total=1) as (page, variant_ids, root):
            output_dir = root / "冲突输出"
            output_dir.mkdir()
            existing = output_dir / "天.png"
            existing.write_bytes(b"keep-existing")
            page._directory_edit.setText(str(output_dir))

            with (
                patch.object(page, "_resolve_export_conflicts", return_value=None) as resolve,
                patch.object(page._thread_pool, "start") as start,
            ):
                page._start_export()
            resolve.assert_called_once()
            self.assertEqual(len(resolve.call_args.args[0]), 1)
            start.assert_not_called()
            self.assertIsNone(page._active_worker)
            self.assertEqual(existing.read_bytes(), b"keep-existing")

            captured_decisions: list[tuple[ExportConflictDecision, ...]] = []

            def skip_conflicts(
                conflicts: list[ExportConflict],
            ) -> tuple[ExportConflictDecision, ...]:
                decisions = tuple(
                    ExportConflictDecision(
                        conflict,
                        ExportService.CONFLICT_SKIP,
                    )
                    for conflict in conflicts
                )
                captured_decisions.append(decisions)
                return decisions

            started: list[object] = []
            with (
                patch.object(page, "_resolve_export_conflicts", side_effect=skip_conflicts),
                patch.object(page._thread_pool, "start", side_effect=started.append),
            ):
                page._start_export()

            self.assertEqual(len(started), 1)
            worker = started[0]
            self.assertEqual(worker._eligible_variant_ids, frozenset(variant_ids))
            self.assertEqual(worker._conflict_decisions, captured_decisions[0])
            page._active_worker = None

    def test_service_failure_result_uses_critical_dialog_with_details(self) -> None:
        with self._page_with_variants(total=2) as (page, _variant_ids, _root):
            page._active_worker = object()  # type: ignore[assignment]
            with (
                patch.object(QMessageBox, "critical") as critical,
                patch.object(QMessageBox, "information") as information,
            ):
                page._export_finished(
                    {
                        "成功": 0,
                        "跳过": 1,
                        "失败": 1,
                        "失败详情": [("甲-0001", "成品文件无法解码")],
                    }
                )

            critical.assert_called_once()
            information.assert_not_called()
            self.assertEqual(critical.call_args.args[1], "导出失败")
            self.assertIn("甲-0001：成品文件无法解码", critical.call_args.args[2])
            self.assertEqual(page._export_status_label.text(), "导出失败：成功 0，跳过 1，失败 1")

    def test_partial_export_requires_confirmation_and_zero_ready_is_blocked(self) -> None:
        with self._page_with_variants(total=4, finished=2, summary_completed=False) as (
            page,
            _variant_ids,
            root,
        ):
            output_dir = root / "部分输出"
            output_dir.mkdir()
            page._directory_edit.setText(str(output_dir))
            started: list[object] = []
            with (
                patch.object(page, "_confirm_partial_export", return_value=False) as confirm,
                patch.object(page._thread_pool, "start", side_effect=started.append),
            ):
                page._start_export()
            confirm.assert_called_once_with(2, 4)
            self.assertEqual(started, [])

            with (
                patch.object(page, "_confirm_partial_export", return_value=True),
                patch.object(page._thread_pool, "start", side_effect=started.append),
            ):
                page._start_export()
            self.assertEqual(len(started), 1)
            self.assertIsNotNone(page._active_worker)
            self.assertEqual(
                page._active_worker._eligible_variant_ids,
                frozenset(_variant_ids[:2]),
            )
            page._active_worker = None

        with self._page_with_variants(total=3, finished=0, summary_completed=False) as (
            page,
            _variant_ids,
            root,
        ):
            output_dir = root / "空输出"
            output_dir.mkdir()
            page._directory_edit.setText(str(output_dir))
            self.assertFalse(page._export_button.isEnabled())
            with (
                patch.object(QMessageBox, "warning") as warning,
                patch.object(page._thread_pool, "start") as start,
            ):
                page._start_export()
            warning.assert_called_once()
            start.assert_not_called()

    def test_right_footer_is_fixed_and_minimum_window_has_no_overlap(self) -> None:
        with self._page_with_variants(total=12) as (page, _variant_ids, _root):
            page.resize(1100, 720)
            for _ in range(6):
                self.app.processEvents()

            self.assertFalse(page._options_scroll.isAncestorOf(page._action_footer))
            self.assertTrue(page._action_footer.isAncestorOf(page._export_button))
            self.assertFalse(page._options_scroll.isAncestorOf(page._export_button))
            self.assertTrue(page._action_footer.isVisible())
            self.assertGreater(page._action_footer.height(), 0)
            for index in range(page._main_splitter.count()):
                widget = page._main_splitter.widget(index)
                self.assertGreater(widget.width(), 0)
                self.assertGreater(widget.height(), 0)

    def test_return_signal_and_shutdown_cancel_pending_work(self) -> None:
        with self._page_with_variants(total=2) as (page, _variant_ids, _root):
            emitted: list[bool] = []
            page.home_requested.connect(lambda: emitted.append(True))
            page.request_back()
            self.assertEqual(emitted, [True])

            page._active_worker = object()  # type: ignore[assignment]
            with patch.object(
                page._gallery,
                "shutdown",
                wraps=page._gallery.shutdown,
            ) as gallery_shutdown:
                page.shutdown()
            gallery_shutdown.assert_called_once_with()
            self.assertTrue(page._shutdown)
            self.assertTrue(page._cancel_event.is_set())


if __name__ == "__main__":
    unittest.main()
