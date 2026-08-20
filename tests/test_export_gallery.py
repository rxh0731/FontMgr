"""导出全库画廊的虚拟化、缓存和后台加载测试。"""

from __future__ import annotations

import os
import threading
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRect, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QListView, QStyleOptionViewItem

from services.workflow_status_service import PHASE_STATUS_COLORS, STATUS_COORDINATED
from ui.widgets.export_gallery import (
    MIB,
    ExportGallery,
    ExportGalleryDelegate,
    ExportGalleryEntry,
    ExportGalleryModel,
    WeightedImageCache,
    calculate_export_preview_geometry,
    calculate_thumbnail_cache_budget,
)


class ThumbnailCacheTests(unittest.TestCase):
    """锁定内存预算和加权 LRU 的硬边界。"""

    def test_memory_budget_uses_total_and_available_memory(self) -> None:
        gib = 1024 * MIB
        self.assertEqual(calculate_thumbnail_cache_budget(40 * gib, 26 * gib), 256 * MIB)
        self.assertEqual(calculate_thumbnail_cache_budget(16 * gib, 8 * gib), 163 * MIB)
        self.assertEqual(calculate_thumbnail_cache_budget(8 * gib, 700 * MIB), 18 * MIB)
        self.assertEqual(calculate_thumbnail_cache_budget(8 * gib, 400 * MIB), 4 * MIB)

    def test_weighted_lru_evicts_oldest_and_touch_promotes_entry(self) -> None:
        image = QImage(16, 16, QImage.Format.Format_ARGB32)
        item_cost = WeightedImageCache.image_cost(image)
        cache = WeightedImageCache(item_cost * 2, max_items=2)
        cache.put("一", image)
        cache.put("二", image)
        self.assertIsNotNone(cache.get("一"))
        cache.put("三", image)
        self.assertIn("一", cache)
        self.assertNotIn("二", cache)
        self.assertIn("三", cache)
        self.assertLessEqual(cache.used_bytes, cache.budget_bytes)

    def test_visible_pins_have_bounded_overflow(self) -> None:
        image = QImage(32, 32, QImage.Format.Format_ARGB32)
        item_cost = WeightedImageCache.image_cost(image)
        cache = WeightedImageCache(
            item_cost,
            max_items=4,
            pinned_overflow_bytes=item_cost,
        )
        cache.put("一", image)
        cache.set_pinned(("一", "二"))
        cache.put("二", image)
        self.assertEqual(cache.item_count, 2)
        cache.put("三", image)
        self.assertNotIn("三", cache)
        self.assertLessEqual(cache.used_bytes, cache.budget_bytes + item_cost)


class ExportGalleryTests(unittest.TestCase):
    """锁定八列布局、虚拟模型和可见区异步加载。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _entries(count: int) -> list[ExportGalleryEntry]:
        return [
            ExportGalleryEntry(
                variant_id=f"字-{index:05d}",
                char=chr(0x4E00 + index % 2000),
                filename=f"字-{index:05d}.png",
                image_path=f"不存在-{index:05d}.png",
                status="已协调",
            )
            for index in range(count)
        ]

    def _wait_until(self, predicate, timeout: float = 2.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return True
            # QTest.qWait 在部分 Windows Qt 构建中会长时间持有 GIL，
            # 这里显式让 Python QRunnable 得到执行机会。
            time.sleep(0.01)
        self.app.processEvents()
        return bool(predicate())

    def test_default_layout_is_eight_columns_without_horizontal_scroll(self) -> None:
        gallery = ExportGallery(cache_budget_bytes=8 * MIB)
        try:
            gallery.resize(960, 600)
            gallery.show()
            QTest.qWait(80)
            self.assertIsInstance(gallery, QListView)
            self.assertEqual(gallery.column_count, 8)
            self.assertEqual(
                gallery.horizontalScrollBarPolicy(),
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
            )
            self.assertEqual(
                gallery.gridSize().width(),
                gallery.viewport().width() // 8 - 1,
            )
            gallery.set_entries(self._entries(9))
            QTest.qWait(80)
            first_rect = gallery.visualRect(gallery.model().index(0, 0))
            eighth_rect = gallery.visualRect(gallery.model().index(7, 0))
            ninth_rect = gallery.visualRect(gallery.model().index(8, 0))
            self.assertGreaterEqual(first_rect.width(), gallery.gridSize().width() - 1)
            self.assertGreaterEqual(first_rect.height(), gallery.gridSize().height() - 1)
            self.assertEqual(eighth_rect.top(), first_rect.top())
            self.assertGreater(ninth_rect.top(), first_rect.top())
        finally:
            gallery.close()

    def test_preview_geometry_uses_library_size_and_130_percent_workspace(self) -> None:
        image_area = QRectF(0.0, 0.0, 420.0, 320.0)
        for width, height in ((250, 250), (300, 200)):
            with self.subTest(canvas=(width, height)):
                geometry = calculate_export_preview_geometry(
                    image_area,
                    QSize(width, height),
                )
                self.assertAlmostEqual(
                    geometry.workspace_rect.center().x(),
                    geometry.grid_rect.center().x(),
                )
                self.assertAlmostEqual(
                    geometry.workspace_rect.center().y(),
                    geometry.grid_rect.center().y(),
                )
                self.assertAlmostEqual(
                    geometry.workspace_rect.width() / geometry.grid_rect.width(),
                    1.3,
                )
                self.assertAlmostEqual(
                    geometry.workspace_rect.height() / geometry.grid_rect.height(),
                    1.3,
                )
                self.assertAlmostEqual(
                    geometry.grid_rect.width() / geometry.grid_rect.height(),
                    width / height,
                )

    def test_delegate_maps_standard_image_to_non_square_grid(self) -> None:
        ink = QColor("#D000FF")
        source = QImage(300, 200, QImage.Format.Format_ARGB32_Premultiplied)
        source.fill(ink)
        entry = ExportGalleryEntry(
            variant_id="宽图",
            char="一",
            filename="宽图.png",
            image_path="宽图.png",
            status="已协调",
            image_canvas_size=(300, 200),
        )
        model = ExportGalleryModel()
        model.set_entries([entry])
        delegate = ExportGalleryDelegate(lambda _entry: (source, False))
        delegate.set_canvas_size(300, 200)
        option = QStyleOptionViewItem()
        option.rect = QRect(0, 0, 400, 320)
        canvas = QImage(400, 320, QImage.Format.Format_ARGB32_Premultiplied)
        canvas.fill(Qt.GlobalColor.transparent)

        painter = QPainter(canvas)
        delegate.paint(painter, option, model.index(0, 0))
        painter.end()

        outer = QRectF(option.rect.adjusted(5, 5, -5, -5))
        image_area = QRectF(
            outer.left(),
            outer.top(),
            outer.width(),
            outer.height() - delegate.FOOTER_HEIGHT,
        )
        geometry = delegate.preview_geometry(image_area)
        target = delegate.image_target_rect(entry, source, geometry)
        self.assertAlmostEqual(target.left(), geometry.grid_rect.left())
        self.assertAlmostEqual(target.top(), geometry.grid_rect.top())
        self.assertAlmostEqual(target.width(), geometry.grid_rect.width())
        self.assertAlmostEqual(target.height(), geometry.grid_rect.height())

        ink_x = round(geometry.grid_rect.left() + geometry.grid_rect.width() * 0.2)
        ink_y = round(geometry.grid_rect.top() + geometry.grid_rect.height() * 0.35)
        margin_x = round(
            (geometry.workspace_rect.left() + geometry.grid_rect.left()) / 2.0
        )
        outside_y = round(
            (image_area.top() + geometry.workspace_rect.top()) / 2.0
        )
        self.assertEqual(canvas.pixelColor(ink_x, ink_y), ink)
        self.assertEqual(canvas.pixelColor(margin_x, ink_y), QColor("#FFFFFF"))
        self.assertEqual(
            canvas.pixelColor(round(geometry.workspace_rect.center().x()), outside_y),
            QColor("#171B22"),
        )

    def test_delegate_keeps_expanded_image_on_same_logical_pixel_scale(self) -> None:
        source = QImage(300, 275, QImage.Format.Format_ARGB32_Premultiplied)
        source.fill(QColor("#202020"))
        entry = ExportGalleryEntry(
            variant_id="越界成品",
            char="永",
            filename="越界成品.png",
            image_path="越界成品.png",
            status="已有成品",
            image_canvas_size=(300, 275),
        )
        delegate = ExportGalleryDelegate(lambda _entry: (source, False))
        delegate.set_canvas_size(250, 250)
        geometry = delegate.preview_geometry(QRectF(0.0, 0.0, 360.0, 300.0))
        target = delegate.image_target_rect(entry, source, geometry)

        self.assertEqual(target.center(), geometry.grid_rect.center())
        self.assertAlmostEqual(target.width() / geometry.grid_rect.width(), 1.2)
        self.assertAlmostEqual(target.height() / geometry.grid_rect.height(), 1.1)
        self.assertTrue(geometry.workspace_rect.contains(target))

    def test_coordinated_phase_status_uses_green_marker(self) -> None:
        entry = ExportGalleryEntry(
            variant_id=STATUS_COORDINATED,
            char="永",
            filename="永.png",
            image_path="永.png",
            status=STATUS_COORDINATED,
        )
        model = ExportGalleryModel()
        model.set_entries([entry])
        delegate = ExportGalleryDelegate(lambda _entry: (None, True))
        option = QStyleOptionViewItem()
        option.rect = QRect(0, 0, 200, 200)
        canvas = QImage(200, 200, QImage.Format.Format_ARGB32_Premultiplied)
        canvas.fill(Qt.GlobalColor.transparent)

        painter = QPainter(canvas)
        delegate.paint(painter, option, model.index(0, 0))
        painter.end()

        self.assertEqual(
            canvas.pixelColor(186, 178),
            QColor(PHASE_STATUS_COLORS[STATUS_COORDINATED]),
        )

    def test_ten_thousand_entries_only_request_visible_and_prefetch_rows(self) -> None:
        gate = threading.Event()

        def blocked_loader(_path: str, _size: QSize, _limit: int) -> QImage:
            gate.wait(0.25)
            return QImage(16, 16, QImage.Format.Format_ARGB32)

        gallery = ExportGallery(
            cache_budget_bytes=8 * MIB,
            image_loader=blocked_loader,
        )
        try:
            gallery.resize(960, 600)
            gallery.set_entries(self._entries(10_000))
            gallery.show()
            QTest.qWait(90)
            self.assertEqual(gallery.entry_count, 10_000)
            self.assertGreater(gallery.last_requested_count, 0)
            self.assertLessEqual(gallery.last_requested_count, gallery.MAX_REQUESTS)
            self.assertLess(gallery.last_requested_count, gallery.entry_count)
            self.assertLessEqual(gallery.pending_count, gallery.MAX_REQUESTS)
        finally:
            gate.set()
            gallery.close()
            QTest.qWait(30)

    def test_loader_runs_off_gui_thread_and_only_returns_qimage(self) -> None:
        main_thread_id = threading.get_ident()
        loader_thread_ids: list[int] = []

        def loader(_path: str, size: QSize, _limit: int) -> QImage:
            loader_thread_ids.append(threading.get_ident())
            image = QImage(size, QImage.Format.Format_ARGB32)
            image.fill(Qt.GlobalColor.transparent)
            return image

        gallery = ExportGallery(
            cache_budget_bytes=8 * MIB,
            image_loader=loader,
        )
        try:
            gallery.resize(800, 420)
            gallery.set_entries(self._entries(4))
            gallery.show()
            self.assertTrue(
                self._wait_until(lambda: gallery.cached_item_count == 4),
                "可见字形缩略图未在期限内完成后台载入",
            )
            self.assertEqual(gallery.cached_item_count, 4)
            self.assertTrue(loader_thread_ids)
            self.assertTrue(
                all(thread_id != main_thread_id for thread_id in loader_thread_ids)
            )
            self.assertLessEqual(gallery.cached_bytes, gallery.cache_budget_bytes)
        finally:
            gallery.close()

    def test_shutdown_stops_timers_and_can_preserve_or_clear_cache(self) -> None:
        gallery = ExportGallery(cache_budget_bytes=8 * MIB)
        image = QImage(16, 16, QImage.Format.Format_ARGB32)
        gallery._cache.put("测试", image)
        gallery._memory_timer.start()
        gallery.shutdown(clear_cache=False)
        self.assertFalse(gallery._memory_timer.isActive())
        self.assertEqual(gallery.cached_item_count, 1)
        gallery.shutdown()
        self.assertEqual(gallery.cached_item_count, 0)

    def test_column_count_and_programmatic_selection_are_public_contracts(self) -> None:
        gallery = ExportGallery(cache_budget_bytes=8 * MIB)
        selected: list[str] = []
        gallery.variant_selected.connect(selected.append)
        try:
            gallery.resize(900, 500)
            gallery.set_entries(self._entries(20))
            gallery.show()
            QTest.qWait(60)
            gallery.set_column_count(6)
            QTest.qWait(60)
            self.assertEqual(gallery.column_count, 6)
            self.assertTrue(gallery.set_selected_variant("字-00012"))
            self.assertEqual(gallery.selected_variant_id(), "字-00012")
            self.assertEqual(selected, [])
            self.assertFalse(gallery.set_selected_variant("缺失"))
        finally:
            gallery.close()


if __name__ == "__main__":
    unittest.main()
