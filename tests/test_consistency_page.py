"""整体协调工作台的响应式比较墙与单字精调回归测试。"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
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
from PySide6.QtCore import QEvent, QPoint, QPointF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QFocusEvent, QImage, QMouseEvent, QPainter, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QHeaderView,
    QHBoxLayout,
    QMessageBox,
    QScrollArea,
    QSlider,
    QTreeWidget,
    QWidget,
)

import config
from services.adjustment_service import AdjustmentService, CoordinationCancelled
from services.glyph_service import GlyphService
import ui.pages.consistency_page as consistency_page_module
from ui.pages.consistency_page import ConsistencyPage
from ui.theme import apply_theme
from ui.widgets.review_canvas import ReviewCanvas
from ui.widgets.two_line_status_delegate import SECONDARY_COLOR_ROLE


class ConsistencyPageResponsiveTests(unittest.TestCase):
    """锁定整体比较与单字精调之间的布局、选择和草稿契约。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])
        apply_theme(cls.app)

    @contextmanager
    def _page_with_sixteen_variants(
        self,
        saved_first_adjustment: dict[str, object] | None = None,
    ) -> Iterator[tuple[ConsistencyPage, list[str]]]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glyph = GlyphService("响应式测试", str(root))
            glyph.ensure_dirs()
            glyph.init_metadata(dpi=300, canvas_w=64, canvas_h=64)
            preview_dir = Path(glyph.get_workflow_dirs()["优化预览"])
            variant_ids: list[str] = []
            for index, char in enumerate("天地玄黄宇宙洪荒日月盈昃辰宿列张", 1):
                filename = f"{char}-{index:04d}.png"
                variant_id = glyph.add_original(
                    char,
                    filename,
                    filename,
                    f"responsive-{index:04d}",
                )
                image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
                draw = ImageDraw.Draw(image)
                inset = 8 + index % 4
                draw.rectangle(
                    (inset, inset, 63 - inset, 63 - inset),
                    fill=(20, 20, 20, 150 + index * 5),
                )
                image.save(preview_dir / filename)
                detail = glyph.get_variant(variant_id)
                detail["中间文件"] = filename
                detail["状态"] = config.STATUS_REVIEWED
                variant_ids.append(variant_id)
            glyph.save()

            if saved_first_adjustment is not None:
                service = AdjustmentService(glyph)
                baseline = service.analyze()
                result = service.save_coordinated_variants(
                    [glyph.get_variant(variant_ids[0])],
                    {variant_ids[0]: saved_first_adjustment},
                    {"启用": True, "基准": baseline["墨色基准"]},
                    baseline,
                )
                self.assertEqual(result, {"成功": 1, "失败": 0, "失败详情": []})

            page = ConsistencyPage(glyph, lambda: None)
            try:
                yield page, variant_ids
            finally:
                page.close()
                page.deleteLater()
                self.app.processEvents()

    def _show_at(self, page: ConsistencyPage, width: int, height: int) -> None:
        page.resize(width, height)
        page.show()
        for _ in range(5):
            self.app.processEvents()

    def _wait_until(self, predicate: object, timeout: float = 3.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            if callable(predicate) and predicate():
                return True
            time.sleep(0.01)
        self.app.processEvents()
        return bool(callable(predicate) and predicate())

    @staticmethod
    def _grid_position(page: ConsistencyPage, widget: QWidget) -> tuple[int, int]:
        index = page._grid.indexOf(widget)
        if index < 0:
            raise AssertionError("比较卡片没有加入比较墙网格。")
        row, column, _row_span, _column_span = page._grid.getItemPosition(index)
        return row, column

    def _drag_card(
        self,
        card: QWidget,
        start: QPointF,
        delta: QPointF,
        *,
        steps: int = 3,
        modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
    ) -> None:
        self._send_card_mouse_event(
            card,
            QEvent.Type.MouseButtonPress,
            start,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            modifiers,
        )
        end = QPointF(start)
        for step in range(1, max(1, steps) + 1):
            end = start + delta * (step / max(1, steps))
            self._send_card_mouse_event(
                card,
                QEvent.Type.MouseMove,
                end,
                Qt.MouseButton.NoButton,
                Qt.MouseButton.LeftButton,
                modifiers,
            )
        self._send_card_mouse_event(
            card,
            QEvent.Type.MouseButtonRelease,
            end,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
            modifiers,
        )

    @staticmethod
    def _send_card_mouse_event(
        card: QWidget,
        event_type: QEvent.Type,
        position: QPointF,
        button: Qt.MouseButton,
        buttons: Qt.MouseButton,
        modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
    ) -> None:
        QApplication.sendEvent(
            card,
            QMouseEvent(
                event_type,
                position,
                position,
                QPointF(card.mapToGlobal(position.toPoint())),
                button,
                buttons,
                modifiers,
            ),
        )

    @staticmethod
    def _card_control_polygon(card: QWidget) -> object:
        for name in ("control_polygon", "_control_polygon"):
            getter = getattr(card, name, None)
            if callable(getter):
                return getter()
        raise AssertionError("比较卡片必须提供当前自由变换控制四边形。")

    @staticmethod
    def _polygon_points(polygon: object) -> tuple[QPointF, ...]:
        count = getattr(polygon, "count", None)
        at = getattr(polygon, "at", None)
        if callable(count) and callable(at):
            return tuple(QPointF(at(index)) for index in range(count()))
        try:
            return tuple(QPointF(point) for point in polygon)  # type: ignore[union-attr]
        except TypeError as exc:
            raise AssertionError("控制四边形必须是可迭代的四点坐标。") from exc

    @staticmethod
    def _card_control_handles(card: QWidget) -> tuple[dict[str, QPointF], QPointF]:
        for name in ("control_handles", "_control_handles"):
            getter = getattr(card, name, None)
            if not callable(getter):
                continue
            result = getter()
            if isinstance(result, tuple) and len(result) == 2:
                handles, rotate = result
                return dict(handles), QPointF(rotate)
        raise AssertionError("比较卡片必须提供八个控制点和旋转手柄。")

    @staticmethod
    def _preview_control_polygon(preview: object) -> tuple[tuple[float, float], ...]:
        polygon = getattr(preview, "control_polygon", None)
        if polygon is None and isinstance(preview, dict):
            polygon = preview.get("control_polygon") or preview.get("控制四边形")
        if polygon is None and isinstance(preview, tuple) and len(preview) >= 3:
            polygon = preview[2]
        if polygon is None:
            raise AssertionError("协调预览必须返回实际控制四边形，而不只是透明像素包围盒。")
        return tuple((float(point[0]), float(point[1])) for point in polygon)

    @staticmethod
    def _preview_image(preview: object) -> Image.Image:
        image = getattr(preview, "image", None)
        if image is None and isinstance(preview, dict):
            image = preview.get("image") or preview.get("图像")
        if image is None and isinstance(preview, tuple) and preview:
            image = preview[0]
        if not isinstance(image, Image.Image):
            raise AssertionError("协调预览没有返回 Pillow 图像。")
        return image

    @staticmethod
    def _render_widget(widget: QWidget) -> QImage:
        image = QImage(widget.size(), QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        widget.render(painter, QPoint())
        painter.end()
        return image

    @staticmethod
    def _dark_pixel_centroid(image: QImage, rect: object) -> QPointF:
        left = max(0, int(rect.left()))
        top = max(0, int(rect.top()))
        right = min(image.width(), int(rect.right()) + 1)
        bottom = min(image.height(), int(rect.bottom()) + 1)
        total_x = 0.0
        total_y = 0.0
        count = 0
        for y in range(top, bottom):
            for x in range(left, right):
                color = image.pixelColor(x, y)
                if max(color.red(), color.green(), color.blue()) >= 96:
                    continue
                total_x += x
                total_y += y
                count += 1
        if not count:
            raise AssertionError("比较卡片工作区内没有可用于定位的深色字形像素。")
        return QPointF(total_x / count, total_y / count)

    @staticmethod
    def _wheel_card(
        card: QWidget,
        delta: int,
        modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
    ) -> None:
        position = QPointF(card.rect().center())
        QApplication.sendEvent(
            card,
            QWheelEvent(
                position,
                QPointF(card.mapToGlobal(position.toPoint())),
                QPoint(),
                QPoint(0, delta),
                Qt.MouseButton.NoButton,
                modifiers,
                Qt.ScrollPhase.ScrollUpdate,
                False,
            ),
        )

    def test_large_library_baseline_analysis_runs_in_background_with_progress(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def slow_analyze(
            _service: AdjustmentService,
            _target_ratio: float | None = None,
            progress_callback: object = None,
        ) -> dict[str, object]:
            started.set()
            if callable(progress_callback):
                progress_callback(3, 16, "测 · test.png")
            release.wait(3.0)
            return {
                "有效数": 16,
                "墨色有效数": 16,
                "目标占比": 0.72,
                "宽中位": 0.6,
                "高中位": 0.7,
                "墨色基准": 188.0,
            }

        try:
            with (
                patch.object(ConsistencyPage, "BACKGROUND_ANALYSIS_THRESHOLD", 1),
                patch.object(
                    AdjustmentService,
                    "analyze",
                    autospec=True,
                    side_effect=slow_analyze,
                ),
                self._page_with_sixteen_variants() as (page, _variant_ids),
            ):
                self._show_at(page, 1100, 720)
                self.assertTrue(self._wait_until(started.is_set))
                self.assertTrue(page._baseline_analysis_pending)
                self.assertTrue(page._task_progress_panel.isVisible())
                self.assertFalse(page._main_splitter.isEnabled())
                self.assertFalse(page._complete_button.isEnabled())

                heartbeats: list[int] = []
                heartbeat = QTimer()
                heartbeat.setInterval(10)
                heartbeat.timeout.connect(lambda: heartbeats.append(1))
                heartbeat.start()
                deadline = time.monotonic() + 0.12
                while time.monotonic() < deadline:
                    self.app.processEvents()
                    time.sleep(0.005)
                heartbeat.stop()
                self.assertGreaterEqual(len(heartbeats), 3)
                self.assertEqual(page._task_progress_bar.value(), 3)

                release.set()
                self.assertTrue(
                    self._wait_until(
                        lambda: not page._baseline_analysis_pending,
                    )
                )
                self.assertTrue(page._main_splitter.isEnabled())
                self.assertTrue(page._complete_button.isEnabled())
                self.assertEqual(page._ink_baseline, 188.0)
                self.assertIn("16 个协调样本", page._ink_baseline_label.text())
        finally:
            release.set()

    def test_comparison_card_handles_move_and_stretch_with_single_undo(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            self._show_at(page, 1100, 720)
            variant_id = page._selected_id
            card = page._cards[variant_id]
            control = card._control_rect()
            self.assertFalse(control.isEmpty())
            self.assertEqual(card._hit_test(control.center()), "move")
            self.assertFalse(page._detail_canvas.can_undo)

            self._drag_card(card, control.center(), QPointF(18.0, 10.0))
            moved = page._detail_canvas.transform()
            self.assertNotEqual(moved["x"], 0.0)
            self.assertNotEqual(moved["y"], 0.0)
            self.assertEqual(len(page._detail_canvas._undo_stack), 1)
            self.assertEqual(page._offset_x_spin.value(), round(float(moved["x"])))
            self.assertEqual(page._offset_y_spin.value(), round(float(moved["y"])))

            page._detail_canvas.undo()
            self.assertAlmostEqual(float(page._detail_canvas.transform()["x"]), 0.0)
            self.assertAlmostEqual(float(page._detail_canvas.transform()["y"]), 0.0)
            self.assertFalse(page._detail_canvas.can_undo)

            card = page._cards[variant_id]
            east = card._handles(card._control_rect())["e"]
            self.assertEqual(card._hit_test(east), "e")
            self.assertEqual(
                card._cursor_for_hit("e"),
                Qt.CursorShape.SizeHorCursor,
            )
            self._drag_card(card, east, QPointF(20.0, 0.0))
            stretched = page._detail_canvas.transform()
            self.assertGreater(float(stretched["stretch_w"]), 1.0)
            self.assertAlmostEqual(float(stretched["stretch_h"]), 1.0)
            self.assertAlmostEqual(float(stretched["scale"]), 1.0)
            self.assertEqual(len(page._detail_canvas._undo_stack), 1)

            page._detail_canvas.undo()
            card = page._cards[variant_id]
            southeast = card._handles(card._control_rect())["se"]
            self._drag_card(card, southeast, QPointF(16.0, 14.0))
            scaled = page._detail_canvas.transform()
            self.assertGreater(float(scaled["scale"]), 1.0)
            self.assertAlmostEqual(float(scaled["stretch_w"]), 1.0)
            self.assertAlmostEqual(float(scaled["stretch_h"]), 1.0)
            self.assertEqual(len(page._detail_canvas._undo_stack), 1)

    def test_comparison_control_polygon_matches_high_quality_preview_geometry(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            self._show_at(page, 1100, 720)
            variant_id = page._selected_id
            transform = {
                "x": 5.0,
                "y": -4.0,
                "scale": 1.08,
                "rotation": 27.0,
                "stretch_w": 1.15,
                "stretch_h": 0.86,
                "distort": [-4.0, 2.0, 5.0, -3.0, 3.0, 6.0, -2.0, 4.0],
            }

            self.assertTrue(page._detail_canvas.set_transform(**transform))
            self.app.processEvents()
            page._preview_timer.stop()
            detail = page._variant_by_id[variant_id]
            preview = page._adjustment_service.preview_coordinated(
                detail,
                page._get_adjustment(variant_id),
                page.WORK_RATIO,
                page._current_ink_config(),
            )
            self.assertIsNotNone(preview)
            preview_polygon = self._preview_control_polygon(preview)
            preview_image = self._preview_image(preview)
            self.assertEqual(len(preview_polygon), 4)

            page._update_card(variant_id)
            self._card_control_polygon(page._cards[variant_id])

            def current_geometry() -> tuple[tuple[QPointF, ...], tuple[QPointF, ...]]:
                current_card = page._cards[variant_id]
                actual_points = self._polygon_points(
                    self._card_control_polygon(current_card)
                )
                work_rect = current_card._work_rect()
                expected_points = tuple(
                    QPointF(
                        work_rect.left() + x * work_rect.width() / preview_image.width,
                        work_rect.top() + y * work_rect.height() / preview_image.height,
                    )
                    for x, y in preview_polygon
                )
                return actual_points, expected_points

            def geometry_matches() -> bool:
                actual_points, expected_points = current_geometry()
                return len(actual_points) == len(expected_points) == 4 and all(
                    abs(actual.x() - target.x()) <= 1.0
                    and abs(actual.y() - target.y()) <= 1.0
                    for actual, target in zip(actual_points, expected_points)
                )

            self.assertTrue(
                self._wait_until(geometry_matches),
                "比较卡片控制四边形没有更新为最新高质量预览几何。",
            )
            card_polygon, expected = current_geometry()
            self.assertEqual(len(card_polygon), 4)
            for actual, target in zip(card_polygon, expected):
                self.assertAlmostEqual(actual.x(), target.x(), delta=1.0)
                self.assertAlmostEqual(actual.y(), target.y(), delta=1.0)

            edges = [
                card_polygon[(index + 1) % 4] - card_polygon[index]
                for index in range(4)
            ]
            self.assertTrue(
                any(abs(edge.x()) > 1.0 and abs(edge.y()) > 1.0 for edge in edges),
                "旋转和扭曲后的控制层不能退化成轴对齐矩形。",
            )

    def test_comparison_card_supports_rotation_distortion_and_shear(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            self._show_at(page, 1100, 720)
            variant_id = page._selected_id

            card = page._cards[variant_id]
            _handles, rotate = self._card_control_handles(card)
            self._drag_card(card, rotate, QPointF(18.0, 8.0), steps=4)
            rotated = page._detail_canvas.transform()
            self.assertNotAlmostEqual(float(rotated["rotation"]), 0.0)
            self.assertEqual(len(page._detail_canvas._undo_stack), 1)

            page._detail_canvas.undo()
            self.app.processEvents()
            card = page._cards[variant_id]
            handles, _rotate = self._card_control_handles(card)
            self._drag_card(
                card,
                handles["ne"],
                QPointF(8.0, -5.0),
                steps=4,
                modifiers=Qt.KeyboardModifier.ControlModifier,
            )
            distorted = page._detail_canvas.transform()
            changed_corner_values = [
                value for value in distorted["distort"] if abs(float(value)) > 0.01
            ]
            self.assertEqual(len(changed_corner_values), 2)
            self.assertAlmostEqual(float(distorted["scale"]), 1.0)
            self.assertEqual(len(page._detail_canvas._undo_stack), 1)

            page._detail_canvas.undo()
            self.app.processEvents()
            card = page._cards[variant_id]
            handles, _rotate = self._card_control_handles(card)
            self._drag_card(
                card,
                handles["e"],
                QPointF(6.0, 9.0),
                steps=4,
                modifiers=Qt.KeyboardModifier.ControlModifier,
            )
            sheared = page._detail_canvas.transform()
            changed_edge_values = [
                value for value in sheared["distort"] if abs(float(value)) > 0.01
            ]
            self.assertEqual(len(changed_edge_values), 4)
            self.assertAlmostEqual(float(sheared["stretch_w"]), 1.0)
            self.assertAlmostEqual(float(sheared["stretch_h"]), 1.0)
            self.assertEqual(len(page._detail_canvas._undo_stack), 1)

    def test_comparison_drag_defers_high_quality_preview_until_release(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            self._show_at(page, 1100, 720)
            variant_id = page._selected_id
            card = page._cards[variant_id]
            start = card._control_rect().center()
            end = QPointF(start)
            render_calls: list[float] = []
            original_preview = AdjustmentService.preview_coordinated

            def counted_preview(service: AdjustmentService, *args, **kwargs):
                render_calls.append(time.monotonic())
                return original_preview(service, *args, **kwargs)

            with patch.object(
                AdjustmentService,
                "preview_coordinated",
                new=counted_preview,
            ):
                self._send_card_mouse_event(
                    card,
                    QEvent.Type.MouseButtonPress,
                    start,
                    Qt.MouseButton.LeftButton,
                    Qt.MouseButton.LeftButton,
                )
                for step in range(1, 101):
                    end = start + QPointF(24.0, 10.0) * (step / 100.0)
                    self._send_card_mouse_event(
                        card,
                        QEvent.Type.MouseMove,
                        end,
                        Qt.MouseButton.NoButton,
                    Qt.MouseButton.LeftButton,
                )
                QTest.qWait(page._preview_timer.interval() + 40)
                render_count_during_drag = len(render_calls)

                self._send_card_mouse_event(
                    card,
                    QEvent.Type.MouseButtonRelease,
                    end,
                    Qt.MouseButton.LeftButton,
                    Qt.MouseButton.NoButton,
                )
                self.assertTrue(
                    self._wait_until(lambda: len(render_calls) >= 1),
                    "松手后没有提交高质量协调预览。",
                )
                QTest.qWait(page._preview_timer.interval() + 40)
                final_render_count = len(render_calls)

            self.assertEqual(
                render_count_during_drag,
                0,
                "持续拖动时不得运行 Pillow/OpenCV 高质量预览。",
            )
            self.assertEqual(final_render_count, 1, "一次拖动松手后只能重算一次预览。")

    def test_drag_release_keeps_final_live_position_until_preview_arrives(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            self._show_at(page, 1100, 720)
            variant_id = page._selected_id
            card = page._cards[variant_id]
            self.assertTrue(
                self._wait_until(
                    lambda: page._coordinated_preview_cache_key(variant_id)
                    in page._preview_cache
                )
            )
            start = card._control_rect().center()
            end = start + QPointF(24.0, 0.0)
            preview_started = threading.Event()
            allow_preview = threading.Event()
            original_preview = AdjustmentService.preview_coordinated

            def delayed_preview(service: AdjustmentService, *args, **kwargs):
                detail = args[0] if args else {}
                adjustment = args[1] if len(args) > 1 else {}
                if (
                    str(detail.get("变体ID", "")) == variant_id
                    and abs(float(adjustment.get("移动X", 0.0))) > 0.01
                ):
                    preview_started.set()
                    allow_preview.wait(3.0)
                return original_preview(service, *args, **kwargs)

            try:
                with patch.object(
                    AdjustmentService,
                    "preview_coordinated",
                    new=delayed_preview,
                ):
                    self._send_card_mouse_event(
                        card,
                        QEvent.Type.MouseButtonPress,
                        start,
                        Qt.MouseButton.LeftButton,
                        Qt.MouseButton.LeftButton,
                    )
                    self._send_card_mouse_event(
                        card,
                        QEvent.Type.MouseMove,
                        end,
                        Qt.MouseButton.NoButton,
                        Qt.MouseButton.LeftButton,
                    )
                    self.app.processEvents()
                    during = self._render_widget(card)
                    during_center = self._dark_pixel_centroid(during, card._work_rect())
                    live_controls = [
                        (
                            card._live_control_polygon_logical.at(index).x(),
                            card._live_control_polygon_logical.at(index).y(),
                        )
                        for index in range(
                            card._live_control_polygon_logical.count()
                        )
                    ]
                    self.assertEqual(len(live_controls), 4)

                    self._send_card_mouse_event(
                        card,
                        QEvent.Type.MouseButtonRelease,
                        end,
                        Qt.MouseButton.LeftButton,
                        Qt.MouseButton.NoButton,
                    )
                    self.assertTrue(self._wait_until(preview_started.is_set))
                    after_release = self._render_widget(card)
                    after_release_center = self._dark_pixel_centroid(
                        after_release,
                        card._work_rect(),
                    )
                    self.assertTrue(card._live_preview_active)
                    self.assertAlmostEqual(
                        after_release_center.x(),
                        during_center.x(),
                        delta=1.5,
                        msg="等待高清预览时不得短暂跳回旧位置。",
                    )
                    self.assertAlmostEqual(
                        after_release_center.y(),
                        during_center.y(),
                        delta=1.5,
                        msg="等待高清预览时不得短暂跳回旧位置。",
                    )

                    allow_preview.set()
                    self.assertTrue(
                        self._wait_until(
                            lambda: variant_id not in page._comparison_preview_pending
                            and not card._live_preview_active
                        )
                    )
                    committed_controls = [
                        (
                            card._control_polygon_logical.at(index).x(),
                            card._control_polygon_logical.at(index).y(),
                        )
                        for index in range(card._control_polygon_logical.count())
                    ]
                    self.assertEqual(len(committed_controls), 4)
                    for committed, live in zip(committed_controls, live_controls):
                        self.assertAlmostEqual(committed[0], live[0], delta=1e-6)
                        self.assertAlmostEqual(committed[1], live[1], delta=1e-6)
            finally:
                allow_preview.set()

    def test_selected_high_quality_preview_never_blocks_gui_thread(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            self._show_at(page, 1100, 720)
            variant_id = page._selected_id
            page._discard_queued_card_previews()
            page._clear_variant_preview_cache(variant_id)
            main_thread_id = threading.get_ident()
            render_threads: list[int] = []
            rendered = threading.Event()
            original_preview = AdjustmentService.preview_coordinated

            def tracked_preview(service: AdjustmentService, *args, **kwargs):
                render_threads.append(threading.get_ident())
                try:
                    return original_preview(service, *args, **kwargs)
                finally:
                    rendered.set()

            with patch.object(
                AdjustmentService,
                "preview_coordinated",
                new=tracked_preview,
            ):
                started_at = time.perf_counter()
                page._refresh_selected_preview()
                elapsed = time.perf_counter() - started_at
                self.assertLess(elapsed, 0.1)
                self.assertTrue(self._wait_until(rendered.is_set))

            self.assertTrue(render_threads)
            self.assertTrue(
                all(thread_id != main_thread_id for thread_id in render_threads),
                "高质量协调预览不得在 GUI 主线程执行。",
            )

    def test_comparison_drag_moves_cached_glyph_pixels_before_release(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            self._show_at(page, 1100, 720)
            variant_id = page._selected_id
            card = page._cards[variant_id]
            control = card._control_rect()
            start = control.center()
            before = self._render_widget(card)
            before_center = self._dark_pixel_centroid(before, card._work_rect())
            end = start + QPointF(24.0, 0.0)

            self._send_card_mouse_event(
                card,
                QEvent.Type.MouseButtonPress,
                start,
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
            )
            try:
                self._send_card_mouse_event(
                    card,
                    QEvent.Type.MouseMove,
                    end,
                    Qt.MouseButton.NoButton,
                    Qt.MouseButton.LeftButton,
                )
                self.app.processEvents()
                during = self._render_widget(card)
                during_center = self._dark_pixel_centroid(during, card._work_rect())
            finally:
                self._send_card_mouse_event(
                    card,
                    QEvent.Type.MouseButtonRelease,
                    end,
                    Qt.MouseButton.LeftButton,
                    Qt.MouseButton.NoButton,
                )

            self.assertGreater(
                during_center.x(),
                before_center.x() + 8.0,
                "拖动中的缓存字形像素必须跟随控制框移动。",
            )

    def test_comparison_wheel_scale_merges_into_one_undo_step(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            self._show_at(page, 1100, 720)
            variant_id = page._selected_id
            card = page._cards[variant_id]
            self.assertFalse(page._detail_canvas.can_undo)
            with patch.object(
                page._adjustment_service,
                "preview_coordinated",
                wraps=page._adjustment_service.preview_coordinated,
            ) as render_preview:
                for _index in range(5):
                    self._wheel_card(card, 120, Qt.KeyboardModifier.ControlModifier)
                    self.app.processEvents()
                self.assertEqual(render_preview.call_count, 0)
                QTest.qWait(page._comparison_wheel_timer.interval() + 40)
                self.assertTrue(
                    self._wait_until(lambda: render_preview.call_count == 1),
                    "滚轮会话结束后没有在后台生成高质量预览。",
                )

            scaled = page._detail_canvas.transform()
            self.assertGreater(float(scaled["scale"]), 1.0)
            self.assertEqual(
                len(page._detail_canvas._undo_stack),
                1,
                "同一轮连续滚轮缩放必须合并为一个撤销动作。",
            )
            page._detail_canvas.undo()
            self.assertAlmostEqual(float(page._detail_canvas.transform()["scale"]), 1.0)
            self.assertFalse(page._detail_canvas.can_undo)

    def test_comparison_plain_wheel_scrolls_without_scaling_selected_card(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            self._show_at(page, 1100, 720)
            card = page._cards[page._selected_id]
            scroll_bar = page._comparison_scroll.verticalScrollBar()
            self.assertGreater(scroll_bar.maximum(), 0)
            before_value = scroll_bar.value()
            before_transform = page._detail_canvas.transform()

            self._wheel_card(card, -120)
            self.app.processEvents()

            self.assertEqual(
                page._detail_canvas.transform(),
                before_transform,
                "普通滚轮不应改变选中字形的缩放参数。",
            )
            self.assertGreater(
                scroll_bar.value(),
                before_value,
                "普通滚轮应交给整体协调列表滚动。",
            )

    def test_plain_wheel_after_move_scrolls_monotonically_without_rebuilding_buffer(
        self,
    ) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            self._show_at(page, 1100, 720)
            variant_id = page._selected_id
            self.assertTrue(self._wait_until(lambda: page._loaded_detail_id == variant_id))
            card = page._cards[variant_id]
            self._drag_card(
                card,
                card._control_rect().center(),
                QPointF(18.0, 6.0),
            )
            self.app.processEvents()
            self.assertTrue(page._is_dirty(variant_id))

            cards_before = dict(page._cards)
            scroll_bar = page._comparison_scroll.verticalScrollBar()
            values = [scroll_bar.value()]
            for _ in range(3):
                current_card = page._cards.get(variant_id, next(iter(page._cards.values())))
                self._wheel_card(current_card, -120)
                self.app.processEvents()
                values.append(scroll_bar.value())

            self.assertEqual(values, sorted(values))
            self.assertGreater(values[-1], values[0])
            for visible_id, old_card in cards_before.items():
                if visible_id in page._cards:
                    self.assertIs(
                        page._cards[visible_id],
                        old_card,
                        "同一虚拟缓冲区内滚动不得销毁并重建卡片。",
                    )

    def test_virtual_rows_use_cell_height_plus_grid_spacing(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            self._show_at(page, 1100, 720)
            spacing = page.COMPARISON_ROW_GAP
            row_stride = page.COMPARISON_CELL_HEIGHT + spacing
            self.assertEqual(page._comparison_row_stride(), row_stride)
            self.assertEqual(page._grid.verticalSpacing(), 0)

            total_rows = len(page._variants) // page._grid_columns
            margins = page._grid.contentsMargins()
            expected_height = (
                margins.top()
                + margins.bottom()
                + total_rows * page.COMPARISON_CELL_HEIGHT
                + max(0, total_rows - 1) * spacing
            )
            self.assertGreaterEqual(page._grid_host.minimumHeight(), expected_height)

            page._scroll_to_variant_index(page._grid_columns * 2)
            self.app.processEvents()
            self.assertEqual(
                page._comparison_scroll.verticalScrollBar().value(),
                row_stride,
                "定位字形时必须把网格行距计入滚动位置。",
            )

    def test_responsive_column_change_discards_historical_grid_rows(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            self._show_at(page, 1100, 720)
            page._grid_columns = 1
            page._render_virtual_view()
            self.app.processEvents()
            self.assertEqual(page._grid.rowCount(), len(page._variants))

            page._grid_columns = 4
            page._render_virtual_view()
            self.app.processEvents()
            expected_rows = (len(page._variants) + 3) // 4
            self.assertEqual(
                page._grid.rowCount(),
                expected_rows,
                "响应式换列后不得保留旧的一列网格行数。",
            )
            self.assertTrue(page._cards)
            self.assertTrue(
                all(
                    card.height() == page.COMPARISON_CELL_HEIGHT
                    for card in page._cards.values()
                ),
                "历史空行不得把当前比较卡片拉高。",
            )

    def test_crossing_virtual_buffer_reuses_overlapping_cards(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            self._show_at(page, 1100, 720)
            originals = list(page._variants)
            expanded = list(originals)
            for cycle in range(1, 4):
                for detail in originals:
                    clone = deepcopy(detail)
                    original_id = str(detail["变体ID"])
                    clone_id = f"{original_id}-virtual-{cycle}"
                    clone["变体ID"] = clone_id
                    expanded.append(clone)
                    page._variant_by_id[clone_id] = clone
            page._variants = expanded
            page._render_virtual_view()
            self.app.processEvents()

            cards_before = dict(page._cards)
            scroll_bar = page._comparison_scroll.verticalScrollBar()
            self.assertGreater(scroll_bar.maximum(), page._comparison_row_stride() * 2)
            scroll_bar.setValue(page._comparison_row_stride() * 2)
            self.app.processEvents()
            cards_after = dict(page._cards)

            overlapping_ids = set(cards_before) & set(cards_after)
            self.assertTrue(overlapping_ids)
            self.assertNotEqual(set(cards_before), set(cards_after))
            for variant_id in overlapping_ids:
                self.assertIs(cards_after[variant_id], cards_before[variant_id])
            self.assertTrue(all(not card.isWindow() for card in cards_after.values()))

    def test_scrolling_focused_card_out_of_buffer_does_not_leave_blank_rows(
        self,
    ) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            self._show_at(page, 1100, 720)
            originals = list(page._variants)
            expanded = list(originals)
            for cycle in range(1, 4):
                for detail in originals:
                    clone = deepcopy(detail)
                    original_id = str(detail["变体ID"])
                    clone_id = f"{original_id}-focus-{cycle}"
                    clone["变体ID"] = clone_id
                    expanded.append(clone)
                    page._variant_by_id[clone_id] = clone
            page._variants = expanded
            page._render_virtual_view()
            self.app.processEvents()

            focused_card = page._cards[page._selected_id]
            focused_card.setFocus(Qt.FocusReason.MouseFocusReason)
            self.app.processEvents()
            self.assertTrue(focused_card.hasFocus())
            scroll_bar = page._comparison_scroll.verticalScrollBar()
            target = min(page._comparison_row_stride() * 6, scroll_bar.maximum())
            scroll_bar.setValue(target)
            self.app.processEvents()

            self.assertEqual(
                scroll_bar.value(),
                target,
                "焦点卡片离开缓冲区时不得把滚动条拉回旧位置。",
            )
            self.assertNotIn(page._selected_id, page._cards)
            visible_tops = [
                card.mapTo(page._comparison_scroll.viewport(), QPoint(0, 0)).y()
                for card in page._cards.values()
                if card.mapTo(
                    page._comparison_scroll.viewport(),
                    QPoint(0, 0),
                ).y()
                + card.height()
                > 0
            ]
            self.assertTrue(visible_tops)
            self.assertLessEqual(
                min(visible_tops),
                page.COMPARISON_ROW_GAP,
                "虚拟卡片首行必须覆盖视口顶部，不得留下空行。",
            )

    def test_clicking_visible_comparison_card_keeps_scroll_position(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            self._show_at(page, 1100, 720)
            scroll_bar = page._comparison_scroll.verticalScrollBar()
            scroll_bar.setValue(min(220, scroll_bar.maximum()))
            self.app.processEvents()
            visible_ids = list(page._cards)
            self.assertTrue(visible_ids)
            target_id = visible_ids[-1]
            before_value = scroll_bar.value()

            page._select_variant_deferred(target_id)
            self.app.processEvents()

            self.assertEqual(
                scroll_bar.value(),
                before_value,
                "点击当前可视区内的字图不应重新定位比较区滚动条。",
            )

    def test_moving_one_card_then_selecting_another_keeps_scroll_position(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            self._show_at(page, 1100, 720)
            scroll_bar = page._comparison_scroll.verticalScrollBar()
            scroll_bar.setValue(min(220, scroll_bar.maximum()))
            self.app.processEvents()
            visible_ids = list(page._cards)
            self.assertGreaterEqual(len(visible_ids), 2)
            target_id = visible_ids[-1]
            page._select_variant_deferred(target_id)
            self.assertTrue(
                self._wait_until(lambda: page._loaded_detail_id == target_id),
            )
            card = page._cards[target_id]
            with (
                patch.object(
                    page,
                    "_refresh_filtered_view",
                    wraps=page._refresh_filtered_view,
                ) as full_refresh,
                patch.object(
                    page,
                    "_render_virtual_view",
                    wraps=page._render_virtual_view,
                ) as rebuild_cards,
            ):
                self._drag_card(
                    card,
                    card._control_rect().center(),
                    QPointF(18.0, 6.0),
                )
                self.app.processEvents()
                full_refresh.assert_not_called()
                rebuild_cards.assert_not_called()
            before_value = scroll_bar.value()

            next_id = next(item for item in page._cards if item != target_id)
            page._select_variant_deferred(next_id)
            self.app.processEvents()

            self.assertEqual(
                scroll_bar.value(),
                before_value,
                "移动字图后切换到另一个可视字图不应造成列表跳行。",
            )

    def test_pending_wheel_session_hands_off_to_drag_as_separate_undo_step(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            self._show_at(page, 1100, 720)
            variant_id = page._selected_id
            card = page._cards[variant_id]

            self._wheel_card(card, 120, Qt.KeyboardModifier.ControlModifier)
            self.app.processEvents()
            wheel_transform = page._detail_canvas.transform()
            self.assertEqual(page._comparison_transform_mode, "wheel")
            self.assertTrue(page._comparison_wheel_timer.isActive())
            self.assertGreater(float(wheel_transform["scale"]), 1.0)
            self.assertEqual(len(page._detail_canvas._undo_stack), 1)

            handles, _rotate = self._card_control_handles(card)
            start = handles["e"]
            self._send_card_mouse_event(
                card,
                QEvent.Type.MouseButtonPress,
                start,
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
            )
            self.app.processEvents()

            self.assertIs(
                page._cards[variant_id],
                card,
                "滚轮会话交给拖动时不能替换仍持有鼠标序列的卡片。",
            )
            self.assertEqual(page._comparison_transform_mode, "drag")
            self.assertTrue(page._comparison_transform_active)
            self.assertFalse(page._comparison_wheel_timer.isActive())
            self.assertTrue(card._drag_kind)

            end = start + QPointF(18.0, 0.0)
            self._send_card_mouse_event(
                card,
                QEvent.Type.MouseMove,
                end,
                Qt.MouseButton.NoButton,
                Qt.MouseButton.LeftButton,
            )
            self._send_card_mouse_event(
                card,
                QEvent.Type.MouseButtonRelease,
                end,
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.NoButton,
            )

            transformed = page._detail_canvas.transform()
            self.assertFalse(page._comparison_transform_active)
            self.assertEqual(page._comparison_transform_mode, "")
            self.assertGreater(float(transformed["stretch_w"]), 1.0)
            self.assertAlmostEqual(
                float(transformed["scale"]),
                float(wheel_transform["scale"]),
                delta=1e-6,
            )
            self.assertEqual(
                len(page._detail_canvas._undo_stack),
                2,
                "滚轮缩放和随后按下的手柄拖动应是两个独立手势。",
            )

            page._detail_canvas.undo()
            after_drag_undo = page._detail_canvas.transform()
            self.assertAlmostEqual(float(after_drag_undo["stretch_w"]), 1.0)
            self.assertAlmostEqual(
                float(after_drag_undo["scale"]),
                float(wheel_transform["scale"]),
                delta=1e-6,
            )
            page._detail_canvas.undo()
            self.assertAlmostEqual(float(page._detail_canvas.transform()["scale"]), 1.0)
            self.assertFalse(page._detail_canvas.can_undo)

    def test_wheel_after_completed_wheel_undo_starts_new_undo_session(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            self._show_at(page, 1100, 720)
            variant_id = page._selected_id
            card = page._cards[variant_id]

            for _index in range(3):
                self._wheel_card(card, 120, Qt.KeyboardModifier.ControlModifier)
            QTest.qWait(page._comparison_wheel_timer.interval() + 40)
            first_scale = float(page._detail_canvas.transform()["scale"])
            self.assertGreater(first_scale, 1.0)
            self.assertEqual(len(page._detail_canvas._undo_stack), 1)

            page._detail_canvas.undo()
            self.app.processEvents()
            self.assertAlmostEqual(float(page._detail_canvas.transform()["scale"]), 1.0)
            self.assertTrue(page._detail_canvas.can_redo)
            self.assertFalse(page._detail_canvas.can_undo)

            card = page._cards[variant_id]
            for _index in range(2):
                self._wheel_card(card, -120, Qt.KeyboardModifier.ControlModifier)
            QTest.qWait(page._comparison_wheel_timer.interval() + 40)
            second_scale = float(page._detail_canvas.transform()["scale"])

            self.assertLess(second_scale, 1.0)
            self.assertEqual(len(page._detail_canvas._undo_stack), 1)
            self.assertFalse(
                page._detail_canvas.can_redo,
                "撤销后开始新的滚轮手势必须废弃旧重做分支。",
            )
            page._detail_canvas.undo()
            self.assertAlmostEqual(float(page._detail_canvas.transform()["scale"]), 1.0)
            self.assertFalse(page._detail_canvas.can_undo)

    def test_card_focus_out_finishes_active_external_transform_session(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            self._show_at(page, 1100, 720)
            variant_id = page._selected_id
            card = page._cards[variant_id]
            start = card._control_rect().center()

            self._send_card_mouse_event(
                card,
                QEvent.Type.MouseButtonPress,
                start,
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
            )
            self._send_card_mouse_event(
                card,
                QEvent.Type.MouseMove,
                start + QPointF(12.0, 5.0),
                Qt.MouseButton.NoButton,
                Qt.MouseButton.LeftButton,
            )
            self.assertTrue(page._comparison_transform_active)
            self.assertTrue(card._drag_kind)
            self.assertTrue(page._detail_canvas._transform_drag_kind)

            QApplication.sendEvent(
                card,
                QFocusEvent(
                    QEvent.Type.FocusOut,
                    Qt.FocusReason.ActiveWindowFocusReason,
                ),
            )
            self.app.processEvents()

            self.assertFalse(page._comparison_transform_active)
            self.assertEqual(page._comparison_transform_mode, "")
            self.assertFalse(card._drag_kind)
            self.assertFalse(page._detail_canvas._transform_drag_kind)
            self.assertIsNone(page._detail_canvas._transform_drag_view)
            self.assertEqual(len(page._detail_canvas._undo_stack), 1)
            self.assertTrue(
                self._wait_until(lambda: not card._live_preview_active),
                "焦点中断后应在高清预览返回时结束临时投影。",
            )
            page._detail_canvas.undo()
            self.assertAlmostEqual(float(page._detail_canvas.transform()["x"]), 0.0)
            self.assertAlmostEqual(float(page._detail_canvas.transform()["y"]), 0.0)

    def test_small_card_hit_matches_review_canvas_public_hit_test(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            self._show_at(page, 1100, 720)
            variant_id = page._selected_id
            page._loading_detail = True
            try:
                self.assertTrue(
                    page._detail_canvas.set_transform(
                        scale=0.05,
                        record_undo=False,
                    )
                )
            finally:
                page._loading_detail = False
            page._sync_selected_card_controls(live=False)

            card = page._cards[variant_id]
            card_handles, _card_rotate = self._card_control_handles(card)
            origin, view_scale = card.transform_view()
            _polygon, canvas_handles, _canvas_rotate = (
                page._detail_canvas.transform_controls_in_view(origin, view_scale)
            )
            self.assertLess(
                page._detail_canvas._distance(card_handles["nw"], card_handles["n"]),
                card.HANDLE_HIT_RADIUS,
                "测试前提要求角点与边点的命中圈发生重叠。",
            )
            self.assertAlmostEqual(
                card_handles["n"].x(),
                canvas_handles["n"].x(),
                delta=0.01,
            )
            self.assertAlmostEqual(
                card_handles["n"].y(),
                canvas_handles["n"].y(),
                delta=0.01,
            )

            position = card_handles["n"]
            for modifiers in (
                Qt.KeyboardModifier.NoModifier,
                Qt.KeyboardModifier.ControlModifier,
            ):
                with self.subTest(modifiers=modifiers):
                    expected = page._detail_canvas.transform_hit_test_in_view(
                        position,
                        origin,
                        view_scale,
                        modifiers,
                    )
                    actual = card._hit_test(position, modifiers)
                    if actual in card_handles:
                        actual = f"scale:{actual}"
                    self.assertEqual(
                        actual,
                        expected,
                        "卡片不得在命中圈重叠时自行选择与 ReviewCanvas 不同的手柄。",
                    )

    def test_initial_load_opens_unpaginated_results_in_pinyin_order(self) -> None:
        with self._page_with_sixteen_variants() as (page, variant_ids):
            self.assertEqual(page._complete_button.text(), "批量整体协调")
            self.assertEqual(
                page._recalculate_baseline_button.text(),
                "重新计算全库基准",
            )
            self.assertTrue(page._recalculate_baseline_button.isEnabled())
            self.assertEqual(page._order_combo.currentText(), "拼音顺序")
            self.assertNotEqual(page._selected_id, variant_ids[0])
            self.assertEqual(
                page._selected_id,
                str(page._variants[0]["变体ID"]),
            )
            self.assertEqual(page._page_index, 0)
            self.assertIs(page._page_variants()[0], page._variants[0])

            self._show_at(page, 1100, 720)

            self.assertGreaterEqual(
                page._grid_columns,
                page.COMPARISON_MIN_COLUMNS,
            )
            self.assertEqual(page._page_index, 0)
            self.assertIs(page._page_variants()[0], page._variants[0])
            self.assertEqual(len(page._page_variants()), len(page._variants))
            self.assertIn(
                "可滚动查看全部字形",
                page._comparison_scroll_label.text(),
            )

    def test_batch_coordination_keeps_fixed_baseline_and_submits_only_pending(
        self,
    ) -> None:
        with self._page_with_sixteen_variants() as (page, variant_ids):
            with (
                patch.object(
                    page,
                    "_confirm_complete_coordination",
                    return_value=True,
                ),
                patch.object(page._coordination_pool, "start") as start,
            ):
                page._complete_coordination()

            start.assert_called_once()
            task = page._coordination_task
            self.assertIsNotNone(task)
            self.assertEqual(set(task._variant_ids), set(variant_ids))
            self.assertFalse(task._ink_config.get("重算几何后基准", False))
            self.assertEqual(task._ink_config["基准"], page._ink_baseline)
            self.assertEqual(page._coordination_task_context["类型"], "批量协调")
            page._finish_coordination_task(False, "测试结束")

    def test_explicit_baseline_recalculation_submits_the_whole_library(self) -> None:
        with self._page_with_sixteen_variants() as (page, variant_ids):
            with (
                patch.object(
                    page,
                    "_confirm_recalculate_coordination_baseline",
                    return_value=True,
                ),
                patch.object(page._coordination_pool, "start") as start,
            ):
                page._recalculate_coordination_baseline()

            start.assert_called_once()
            task = page._coordination_task
            self.assertIsNotNone(task)
            self.assertEqual(set(task._variant_ids), set(variant_ids))
            self.assertTrue(task._ink_config["重算几何后基准"])
            self.assertEqual(
                page._coordination_task_context["类型"],
                "全库基准重算",
            )
            page._finish_coordination_task(False, "测试结束")

    def test_comparison_grid_columns_follow_available_viewport_width(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            ordered_ids = [str(item["变体ID"]) for item in page._variants]
            observed_columns: list[int] = []
            for width, height in ((1100, 720), (1600, 900), (1920, 1080)):
                self._show_at(page, width, height)
                observed_columns.append(page._grid_columns)
                margins = page._grid.contentsMargins()
                spacing = max(0, page._grid.horizontalSpacing())
                usable_width = (
                    page._comparison_scroll.viewport().width()
                    - margins.left()
                    - margins.right()
                )
                expected = max(
                    page.COMPARISON_MIN_COLUMNS,
                    (usable_width + spacing)
                    // (page.COMPARISON_CARD_MIN_WIDTH + spacing),
                )
                self.assertEqual(page._grid_columns, expected)
                self.assertTrue(page._cards)
                self.assertGreaterEqual(
                    min(card.width() for card in page._cards.values()),
                    page.COMPARISON_CARD_MIN_WIDTH,
                )

            self.assertEqual(observed_columns, sorted(observed_columns))
            self.assertGreater(observed_columns[-1], observed_columns[0])
            self.assertEqual(len(page._page_variants()), len(page._variants))
            first_card = page._cards[ordered_ids[0]]
            self.assertEqual(self._grid_position(page, first_card), (0, 0))

            first_group = page._glyph_list.topLevelItem(0)
            page._glyph_list.setCurrentItem(first_group.child(0))
            for _ in range(3):
                self.app.processEvents()
            page.resize(1100, 720)
            for _ in range(5):
                self.app.processEvents()

            self.assertEqual(page._selected_id, ordered_ids[0])
            self.assertEqual(page._grid_columns, observed_columns[0])
            self.assertEqual(len(page._page_variants()), len(page._variants))
            self.assertEqual(page._grid.columnCount(), observed_columns[0])

    def test_comparison_selection_loads_detail_in_background_and_keeps_latest_result(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            ordered_ids = [str(item["变体ID"]) for item in page._variants]
            first_target, latest_target = ordered_ids[1:3]
            started = threading.Event()
            release = threading.Event()
            original_load = page._adjustment_service.load_reviewed_source

            def delayed_load(detail: dict[str, object]) -> object:
                if str(detail.get("变体ID", "")) == first_target:
                    started.set()
                    release.wait(3.0)
                return original_load(detail)

            page._clear_detail_cache()
            try:
                with patch.object(
                    page._adjustment_service,
                    "load_reviewed_source",
                    side_effect=delayed_load,
                ):
                    selected_at = time.perf_counter()
                    page._select_variant_deferred(first_target)
                    selection_elapsed = time.perf_counter() - selected_at
                    self.assertEqual(page._selected_id, first_target)
                    self.assertLess(selection_elapsed, 0.25)
                    self.assertTrue(self._wait_until(started.is_set))
                    self.assertNotEqual(page._loaded_detail_id, first_target)

                    page._select_variant_deferred(latest_target)
                    self.assertEqual(page._selected_id, latest_target)
                    release.set()
                    self.assertTrue(
                        self._wait_until(
                            lambda: page._loaded_detail_id == latest_target,
                        )
                    )
                    self.assertEqual(page._loaded_detail_id, latest_target)
            finally:
                release.set()

    def test_comparison_selection_selects_and_reveals_matching_list_item(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            self._show_at(page, 1100, 720)
            target_id = next(
                variant_id
                for variant_id in page._cards
                if variant_id != page._selected_id
            )
            target_item = page._list_items_by_id[target_id]
            last_group = page._glyph_list.topLevelItem(
                page._glyph_list.topLevelItemCount() - 1
            )
            page._glyph_list.scrollToItem(last_group.child(last_group.childCount() - 1))
            self.app.processEvents()
            self.assertFalse(
                page._glyph_list.viewport().rect().intersects(
                    page._glyph_list.visualItemRect(target_item)
                )
            )

            page._cards[target_id].selected.emit(target_id)
            self.app.processEvents()

            self.assertEqual(page._selected_id, target_id)
            self.assertIs(page._glyph_list.currentItem(), target_item)
            self.assertTrue(target_item.parent().isExpanded())
            self.assertTrue(
                page._glyph_list.viewport().rect().intersects(
                    page._glyph_list.visualItemRect(target_item)
                )
            )

    def test_entering_detail_loads_selected_glyph_even_when_background_load_is_pending(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            target_id = str(page._variants[1]["变体ID"])
            started = threading.Event()
            release = threading.Event()
            main_thread_id = threading.get_ident()
            original_load = page._adjustment_service.load_reviewed_source

            def delay_background_only(detail: dict[str, object]) -> object:
                if threading.get_ident() != main_thread_id:
                    started.set()
                    release.wait(3.0)
                return original_load(detail)

            page._clear_detail_cache()
            try:
                with patch.object(
                    page._adjustment_service,
                    "load_reviewed_source",
                    side_effect=delay_background_only,
                ):
                    page._select_variant_deferred(target_id)
                    self.assertTrue(self._wait_until(started.is_set))
                    page._enter_detail(target_id)
                    self.assertEqual(page._loaded_detail_id, target_id)
                    self.assertTrue(page._detail_canvas.has_image)
                    self.assertIs(page._view_stack.currentWidget(), page._detail_view)
            finally:
                release.set()
                self._wait_until(lambda: page._detail_worker is None)

    def test_detail_source_lru_avoids_reloading_recent_glyph(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            ordered_ids = [str(item["变体ID"]) for item in page._variants]
            first_target, second_target = ordered_ids[1:3]
            main_thread_id = threading.get_ident()
            loaded_on_main_thread: list[str] = []
            original_load = page._adjustment_service.load_reviewed_source

            def track_detail_load(detail: dict[str, object]) -> object:
                if threading.get_ident() == main_thread_id:
                    loaded_on_main_thread.append(str(detail.get("变体ID", "")))
                return original_load(detail)

            page._clear_detail_cache()
            with patch.object(
                page._adjustment_service,
                "load_reviewed_source",
                side_effect=track_detail_load,
            ):
                page._select_variant(first_target)
                page._select_variant(second_target)
                self.assertEqual(loaded_on_main_thread, [first_target, second_target])
                page._select_variant(first_target)
                self.assertEqual(loaded_on_main_thread, [first_target, second_target])
                self.assertEqual(page._loaded_detail_id, first_target)

    def test_double_click_signal_enters_single_glyph_detail_mode(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            self._show_at(page, 1600, 900)
            variant_id = str(page._variants[0]["变体ID"])
            card = page._cards[variant_id]
            emitted: list[str] = []
            card.edit_requested.connect(emitted.append)

            QTest.mouseDClick(
                card,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
                card.rect().center(),
            )
            for _ in range(3):
                self.app.processEvents()

            self.assertEqual(emitted, [variant_id])
            self.assertEqual(page._selected_id, variant_id)
            self.assertIs(page._view_stack.currentWidget(), page._detail_view)
            self.assertTrue(page._detail_canvas.has_image)
            self.assertEqual(page._detail_canvas.tool, ReviewCanvas.TOOL_TRANSFORM)

    def test_detail_navigation_uses_top_row_and_brush_size_uses_slider(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            self._show_at(page, 1100, 720)
            self.assertFalse(page._detail_navigation.isVisible())

            page._enter_detail(page._selected_id)
            self.app.processEvents()

            self.assertTrue(page._detail_navigation.isVisible())
            self.assertIs(
                page._detail_navigation.parentWidget(),
                page._comparison_mode_button.parentWidget(),
            )
            self.assertIsInstance(page._detail_brush_size, QSlider)
            self.assertIsInstance(page._detail_toolbar.layout(), QHBoxLayout)
            self.assertEqual(page._detail_brush_size.orientation(), Qt.Orientation.Horizontal)
            self.assertEqual(
                (page._detail_brush_size.minimum(), page._detail_brush_size.maximum()),
                (1, 100),
            )
            page._detail_brush_size.setValue(24)
            self.assertEqual(page._detail_canvas.brush_size, 24)
            self.assertEqual(page._undo_button.text(), "↶")
            self.assertEqual(page._redo_button.text(), "↷")
            self.assertEqual(
                page._detail_tool_buttons[ReviewCanvas.TOOL_TRANSFORM].text(),
                "变换",
            )
            self.assertEqual(
                page._detail_tool_buttons[ReviewCanvas.TOOL_ERASER].text(),
                "橡皮",
            )
            self.assertEqual(page._source_button.text(), "原稿")
            self.assertEqual(page._grid_button.text(), "网格")
            self.assertEqual(page._transparent_button.text(), "透明")

            page._leave_detail()
            self.app.processEvents()
            self.assertFalse(page._detail_navigation.isVisible())

    def test_detail_transform_defers_library_refresh_until_interaction_finishes(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            self._show_at(page, 1100, 720)
            page._enter_detail(page._selected_id)
            self.app.processEvents()
            canvas = page._detail_canvas
            exact_processor = canvas._render_postprocessor

            with patch.object(page, "_render_status") as render_status:
                page._on_detail_transform_started()
                interactive_processor = canvas._render_postprocessor
                self.assertTrue(page._detail_transform_active)
                self.assertIsNot(interactive_processor, exact_processor)

                self.assertTrue(canvas.set_transform(x=12.0))
                self.assertIn(page._selected_id, page._dirty_variant_ids)
                render_status.assert_not_called()

                page._on_detail_transform_finished(True)
                self.assertFalse(page._detail_transform_active)
                self.assertIsNot(canvas._render_postprocessor, interactive_processor)
                render_status.assert_called_once()

    def test_status_refresh_uses_cached_dirty_set_without_rescanning_signatures(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            with patch.object(
                page,
                "_is_dirty",
                side_effect=AssertionError("状态刷新不应重新计算全部字形签名"),
            ):
                page._render_status()

    def test_glyph_list_matches_review_tree_presentation(self) -> None:
        with self._page_with_sixteen_variants(
            {"移动X": 2.0, "移动Y": 0.0, "等比缩放": 1.0}
        ) as (page, variant_ids):
            coordinated_id = variant_ids[0]
            coordinated_detail = page._variant_by_id[coordinated_id]
            duplicate_char = str(coordinated_detail["归属字"])
            duplicate_filename = f"{duplicate_char}-补充字形.png"
            duplicate_id = page._glyph.add_original(
                duplicate_char,
                duplicate_filename,
                duplicate_filename,
                "duplicate-for-grouping",
            )
            duplicate_image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
            ImageDraw.Draw(duplicate_image).ellipse(
                (12, 12, 52, 52),
                fill=(25, 25, 25, 230),
            )
            duplicate_image.save(
                Path(page._glyph.get_workflow_dirs()["优化预览"])
                / duplicate_filename
            )
            duplicate_detail = page._glyph.get_variant(duplicate_id)
            duplicate_detail["中间文件"] = duplicate_filename
            duplicate_detail["状态"] = config.STATUS_REVIEWED
            page._glyph.save()
            page._reload_variants()

            tree = page._glyph_list
            self.assertIsInstance(tree, QTreeWidget)
            self.assertEqual(page._filter_combo.currentText(), "全部（本阶段）")
            self.app.processEvents()
            self.assertEqual(tree.columnCount(), 2)
            self.assertEqual(
                [tree.headerItem().text(column) for column in range(2)],
                ["字形与文件", "状态与提示"],
            )
            self.assertEqual(
                [page._filter_combo.itemText(index) for index in range(page._filter_combo.count())],
                ["全部（本阶段）", "待协调", "已协调"],
            )
            self.assertFalse(hasattr(page, "_problem_filter_combo"))
            self.assertTrue(tree.rootIsDecorated())
            self.assertEqual(tree.indentation(), 14)
            self.assertFalse(tree.uniformRowHeights())
            self.assertFalse(tree.alternatingRowColors())
            self.assertTrue(tree.wordWrap())
            self.assertEqual(tree.iconSize(), QSize(38, 38))
            self.assertEqual(page.LIST_THUMBNAIL_CACHE_ITEMS, 512)

            header = tree.header()
            for column in range(2):
                self.assertEqual(
                    header.sectionResizeMode(column),
                    QHeaderView.ResizeMode.Interactive,
                )
            status_text_width = max(
                tree.fontMetrics().horizontalAdvance(value)
                for value in ("待协调", "已协调", "已协调 1/1", "状态与提示")
            )
            self.assertGreater(header.sectionSize(1), status_text_width)
            protected_status_width = header.sectionSize(1)
            header.resizeSection(0, 48)
            header.resizeSection(1, 48)
            self.assertGreaterEqual(header.sectionSize(0), 160)
            self.assertGreaterEqual(header.sectionSize(1), protected_status_width)
            self.assertEqual(tree.topLevelItemCount(), 16)
            self.assertEqual(page._list_count_label.text(), "显示 17/17")
            self.assertEqual(
                page._search_edit.placeholderText(),
                "搜索归属字、文件名或变体ID",
            )

            single_detail = page._variant_by_id[variant_ids[1]]
            single_char = str(single_detail["归属字"])
            single_parent = next(
                tree.topLevelItem(row)
                for row in range(tree.topLevelItemCount())
                if tree.topLevelItem(row).text(0).startswith(f"{single_char}（")
            )
            self.assertEqual(single_parent.text(0), f"{single_char}（1个字形）")
            self.assertEqual(single_parent.childCount(), 1)
            self.assertTrue(single_parent.isExpanded())
            self.assertFalse(
                bool(single_parent.flags() & Qt.ItemFlag.ItemIsSelectable)
            )
            self.assertEqual(
                single_parent.child(0).text(0),
                f"字形1 · {single_detail['原始文件']}",
            )
            self.assertEqual(single_parent.child(0).sizeHint(0).height(), 52)

            duplicate_parent = next(
                tree.topLevelItem(row)
                for row in range(tree.topLevelItemCount())
                if tree.topLevelItem(row).text(0).startswith(f"{duplicate_char}（")
            )
            self.assertEqual(duplicate_parent.text(0), f"{duplicate_char}（2个字形）")
            self.assertEqual(
                duplicate_parent.text(1).splitlines(),
                ["已协调 1/2", "问题 0"],
            )
            self.assertEqual(
                duplicate_parent.foreground(1).color(),
                QColor("#4169E1"),
            )
            self.assertEqual(duplicate_parent.childCount(), 2)
            self.assertTrue(duplicate_parent.isExpanded())
            self.assertFalse(
                bool(duplicate_parent.flags() & Qt.ItemFlag.ItemIsSelectable)
            )
            self.assertTrue(duplicate_parent.font(0).bold())

            finished_path = (
                Path(page._glyph.get_workflow_dirs()["成品"])
                / str(coordinated_detail["成品文件"])
            )
            Image.new("RGBA", (64, 64), (20, 140, 50, 255)).save(finished_path)
            page._preview_cache.clear()
            page._list_thumbnail_cache.clear()
            with patch.object(
                page._adjustment_service,
                "preview_coordinated",
                wraps=page._adjustment_service.preview_coordinated,
            ) as render_preview:
                page._populate_list()
            render_preview.assert_not_called()

            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                self.app.processEvents()
                coordinated = page._list_items_by_id[coordinated_id]
                finished_icon = coordinated.icon(0).pixmap(QSize(38, 38)).toImage()
                if finished_icon.pixelColor(19, 19) == QColor(20, 140, 50):
                    break
                time.sleep(0.01)

            coordinated = page._list_items_by_id[coordinated_id]
            self.assertTrue(coordinated.text(0).startswith("字形1 · "))
            self.assertIn(str(coordinated_detail["原始文件"]), coordinated.text(0))
            self.assertEqual(coordinated.text(1).splitlines(), ["已协调", "无"])
            self.assertFalse(coordinated.icon(0).isNull())
            self.assertEqual(coordinated.sizeHint(0).height(), 52)
            self.assertIn(str(coordinated_detail["原始文件"]), coordinated.toolTip(0))
            self.assertIn("整体协调：已协调", coordinated.toolTip(0))
            self.assertNotIn("阶段：", coordinated.toolTip(0))
            self.assertEqual(coordinated.foreground(1).color(), QColor("#228B22"))
            self.assertEqual(coordinated.foreground(0).style(), Qt.BrushStyle.NoBrush)
            finished_icon = coordinated.icon(0).pixmap(QSize(38, 38)).toImage()
            self.assertEqual(finished_icon.pixelColor(19, 19), QColor(20, 140, 50))

            live_preview = QImage(64, 64, QImage.Format.Format_ARGB32)
            live_preview.fill(QColor(45, 90, 210))
            page._preview_cache[
                page._coordinated_preview_cache_key(coordinated_id)
            ] = live_preview
            page._populate_list()
            coordinated = page._list_items_by_id[coordinated_id]
            preview_icon = coordinated.icon(0).pixmap(QSize(38, 38)).toImage()
            self.assertEqual(preview_icon.pixelColor(19, 19), QColor(45, 90, 210))

            dirty_id = variant_ids[1]
            page._adjustments[dirty_id]["移动X"] = 3.0
            page._sync_dirty_variant(dirty_id)
            page._populate_list()
            statuses = page._list_items_by_id
            self.assertEqual(
                statuses[dirty_id].text(1).splitlines(),
                ["待协调", "未保存修改"],
            )
            self.assertEqual(statuses[dirty_id].foreground(1).color(), QColor("#4169E1"))
            self.assertEqual(
                statuses[dirty_id].data(1, SECONDARY_COLOR_ROLE),
                QColor("#F2B84B").name(),
            )
            pending_id = variant_ids[2]
            self.assertEqual(
                statuses[pending_id].text(1).splitlines(),
                ["待协调", "无"],
            )
            self.assertEqual(statuses[pending_id].foreground(1).color(), QColor("#4169E1"))

            page._search_edit.setText(duplicate_filename)
            self.app.processEvents()
            self.assertGreater(len(page._variants), 1)
            self.assertEqual(page._search_button.text(), "搜索")
            QTest.mouseClick(page._search_button, Qt.MouseButton.LeftButton)
            self.app.processEvents()
            self.assertEqual(len(page._variants), 1)
            self.assertEqual(tree.topLevelItemCount(), 1)
            self.assertEqual(tree.topLevelItem(0).text(0), f"{duplicate_char}（2个字形）")
            self.assertEqual(tree.topLevelItem(0).childCount(), 1)
            page._select_variant(duplicate_id)
            self.assertIs(tree.currentItem(), page._list_items_by_id[duplicate_id])
            self.assertTrue(tree.currentItem().parent().isExpanded())

            second_filename = str(page._variant_by_id[pending_id]["原始文件"])
            page._search_edit.setText(second_filename)
            page._search_edit.setFocus()
            QTest.keyClick(page._search_edit, Qt.Key.Key_Return)
            self.app.processEvents()
            self.assertEqual(page._selected_id, pending_id)
            self.assertEqual(len(page._variants), 1)
            self.assertIs(page._view_stack.currentWidget(), page._comparison_view)

            page._search_edit.clear()
            self.app.processEvents()
            self.assertGreater(len(page._variants), 1)

            tree.setFocus()
            QTest.keyClick(tree, Qt.Key.Key_Return)
            self.app.processEvents()
            self.assertIs(page._view_stack.currentWidget(), page._detail_view)
            page._leave_detail()

            tree.itemDoubleClicked.emit(page._list_items_by_id[pending_id], 0)
            self.app.processEvents()
            self.assertEqual(page._selected_id, pending_id)
            self.assertIs(page._view_stack.currentWidget(), page._detail_view)

    def test_group_parent_cannot_replace_active_child_current_item(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            self._show_at(page, 1100, 720)
            tree = page._glyph_list
            selected_id = page._selected_id
            child = page._list_items_by_id[selected_id]
            parent = child.parent()
            self.assertIsNotNone(parent)
            tree.setCurrentItem(child)
            tree.setFocus()

            with patch.object(page, "_load_detail_canvas") as load_detail:
                QTest.mouseClick(
                    tree.viewport(),
                    Qt.MouseButton.LeftButton,
                    Qt.KeyboardModifier.NoModifier,
                    tree.visualItemRect(parent).center(),
                )
                self.app.processEvents()
                self.assertIs(tree.currentItem(), child)
                self.assertEqual(page._selected_id, selected_id)

                QTest.keyClick(tree, Qt.Key.Key_Up)
                self.app.processEvents()
                self.assertIs(tree.currentItem(), child)
                self.assertEqual(page._selected_id, selected_id)
                load_detail.assert_not_called()

            parent.setExpanded(False)
            tree.setCurrentItem(parent)
            self.app.processEvents()
            self.assertIs(tree.currentItem(), child)
            self.assertFalse(parent.isExpanded())
            parent.setExpanded(True)

    def test_list_file_thumbnails_decode_only_after_visible_area_is_scheduled(self) -> None:
        with self._page_with_sixteen_variants() as (page, variant_ids):
            page._preview_cache.clear()
            page._list_thumbnail_cache.clear()
            page._list_thumbnail_key_by_variant.clear()
            page._list_thumbnail_failures.clear()
            original_decode = consistency_page_module.decode_thumbnail_image
            with (
                patch.object(
                    consistency_page_module,
                    "decode_thumbnail_image",
                    wraps=original_decode,
                ) as decode_thumbnail,
                patch.object(
                    page,
                    "_list_thumbnail_source",
                    wraps=page._list_thumbnail_source,
                ) as thumbnail_source,
            ):
                page._list_thumbnail_timer.stop()
                page._populate_list()
                self.assertEqual(decode_thumbnail.call_count, 0)
                self.assertEqual(thumbnail_source.call_count, 0)

                self._show_at(page, 1100, 720)
                page._glyph_list.verticalScrollBar().setValue(
                    page._glyph_list.verticalScrollBar().maximum()
                )
                page._schedule_list_thumbnail_loads()
                deadline = time.monotonic() + 3.0
                while decode_thumbnail.call_count == 0 and time.monotonic() < deadline:
                    self.app.processEvents()
                    time.sleep(0.01)
                self.assertGreater(decode_thumbnail.call_count, 0)

                while page._list_thumbnail_workers and time.monotonic() < deadline:
                    self.app.processEvents()
                    time.sleep(0.01)
                self.assertLess(decode_thumbnail.call_count, len(variant_ids))

    def test_list_thumbnail_does_not_hide_corrupt_finished_image_with_fallback(self) -> None:
        with self._page_with_sixteen_variants(
            {"移动X": 2.0, "移动Y": 0.0, "等比缩放": 1.0}
        ) as (page, variant_ids):
            page._filter_combo.setCurrentText("全部（本阶段）")
            self.app.processEvents()
            variant_id = variant_ids[0]
            detail = page._variant_by_id[variant_id]
            finished_path = (
                Path(page._glyph.get_workflow_dirs()["成品"])
                / str(detail["成品文件"])
            )
            finished_path.write_bytes("损坏的成品文件".encode("utf-8"))
            page._preview_cache.clear()
            page._list_thumbnail_cache.clear()
            page._list_thumbnail_failures.clear()
            page._populate_list()

            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                self.app.processEvents()
                if any(
                    key and key[0] == variant_id and key[1] == "成品"
                    for key in page._list_thumbnail_failures
                ):
                    break
                time.sleep(0.01)

            self.assertTrue(
                any(
                    key and key[0] == variant_id and key[1] == "成品"
                    for key in page._list_thumbnail_failures
                )
            )
            self.assertFalse(
                any(
                    key and key[0] == variant_id and key[1] == "优化预览"
                    for key in page._list_thumbnail_cache
                )
            )

    def test_upstream_record_enters_left_tree_only_after_coordination_admission(self) -> None:
        with self._page_with_sixteen_variants() as (page, variant_ids):
            variant_id = variant_ids[0]
            detail = page._glyph.get_variant(variant_id)
            original_dir = Path(page._glyph.get_workflow_dirs()["原图"])
            preview_dir = Path(page._glyph.get_workflow_dirs()["优化预览"])
            filename = str(detail["原始文件"])
            with Image.open(preview_dir / filename) as source:
                source.save(original_dir / filename)
            detail["中间文件"] = ""
            detail["状态"] = config.STATUS_PENDING_OPTIMIZATION
            page._glyph.save()
            page._reload_variants()

            self.assertNotIn(variant_id, page._variant_by_id)
            self.assertNotIn(variant_id, page._list_variant_by_id)
            self.assertNotIn(variant_id, page._list_items_by_id)

            detail["中间文件"] = filename
            detail["状态"] = config.STATUS_REVIEWED
            page._glyph.save()
            page._reload_variants()

            self.assertIn(variant_id, page._variant_by_id)
            list_detail = page._list_variant_by_id[variant_id]
            path, cache_key = page._list_thumbnail_source(list_detail)
            self.assertTrue(path)
            self.assertEqual(cache_key[1], "优化预览")

    def test_resize_keeps_unsaved_detail_transform_parameters(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            self._show_at(page, 1600, 900)
            variant_id = str(page._variants[0]["变体ID"])
            page._cards[variant_id].edit_requested.emit(variant_id)
            self.app.processEvents()

            self.assertEqual(page.TRANSFORM_PERCENT_MAX, 500)
            self.assertEqual(page._scale_slider.minimum(), -400)
            self.assertEqual(page._scale_slider.maximum(), 400)
            self.assertEqual(page._stretch_w_slider.minimum(), -400)
            self.assertEqual(page._stretch_w_slider.maximum(), 400)
            self.assertEqual(page._stretch_h_slider.minimum(), -400)
            self.assertEqual(page._stretch_h_slider.maximum(), 400)
            self.assertEqual(page._percent_to_slider_position(5), -400)
            self.assertEqual(page._percent_to_slider_position(100), 0)
            self.assertEqual(page._percent_to_slider_position(500), 400)
            self.assertTrue(
                page._detail_canvas.set_transform(
                    scale=50.0,
                    stretch_w=50.0,
                    stretch_h=50.0,
                )
            )
            limited = page._detail_canvas.transform()
            self.assertEqual(limited["scale"], 5.0)
            self.assertEqual(limited["stretch_w"], 5.0)
            self.assertEqual(limited["stretch_h"], 5.0)

            changed = page._detail_canvas.set_transform(
                x=17.0,
                y=-9.0,
                scale=1.12,
                stretch_w=1.08,
                stretch_h=0.93,
                rotation=4.0,
            )
            self.assertTrue(changed)
            before = dict(page._adjustments[variant_id])
            self.assertTrue(page._is_dirty(variant_id))
            self.assertEqual(before["移动X"], 17.0)
            self.assertEqual(before["移动Y"], -9.0)
            self.assertEqual(before["等比缩放"], 1.12)
            self.assertEqual(before["水平拉伸"], 1.08)
            self.assertEqual(before["垂直拉伸"], 0.93)
            self.assertEqual(before["旋转"], 4.0)

            page.resize(1100, 720)
            for _ in range(5):
                self.app.processEvents()

            after = page._adjustments[variant_id]
            for key in (
                "移动X",
                "移动Y",
                "等比缩放",
                "水平拉伸",
                "垂直拉伸",
                "旋转",
            ):
                self.assertEqual(after[key], before[key])
            self.assertTrue(page._is_dirty(variant_id))
            self.assertIs(page._view_stack.currentWidget(), page._detail_view)
            self.assertEqual(page._grid_columns, 4)

    def test_right_action_footer_stays_outside_scroll_area(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            self._show_at(page, 1100, 720)

            self.assertIsInstance(page._tools_scroll, QScrollArea)
            self.assertIsInstance(page._action_footer, QWidget)
            self.assertFalse(page._tools_scroll.isAncestorOf(page._action_footer))
            self.assertTrue(page._action_footer.isAncestorOf(page._save_button))
            self.assertFalse(page._tools_scroll.isAncestorOf(page._save_button))
            self.assertTrue(page._action_footer.isVisible())
            self.assertGreater(page._action_footer.height(), 0)

    def test_reference_overlay_and_source_preview_do_not_create_edits(self) -> None:
        with self._page_with_sixteen_variants() as (page, variant_ids):
            self._show_at(page, 1600, 900)
            selected_id = page._selected_id
            reference_id = next(item for item in variant_ids if item != selected_id)
            page._enter_detail(selected_id)
            self.app.processEvents()

            saved_signature = page._signature(selected_id)
            saved_transform = page._detail_canvas.transform()
            self.assertFalse(page._is_dirty(selected_id))
            self.assertFalse(page._detail_canvas.is_dirty)

            self.assertFalse(page._detail_canvas.can_undo)
            self.assertFalse(page._detail_canvas.can_redo)

            reference_index = page._reference_combo.findData(reference_id)
            self.assertGreater(reference_index, 0)
            page._reference_combo.setCurrentIndex(reference_index)
            page._reference_overlay_check.setChecked(True)
            self.app.processEvents()

            self.assertTrue(page._detail_canvas.reference_visible)
            self.assertAlmostEqual(page._detail_canvas.reference_opacity, 0.35)
            self.assertEqual(page._signature(selected_id), saved_signature)
            self.assertEqual(page._detail_canvas.transform(), saved_transform)
            self.assertFalse(page._is_dirty(selected_id))
            self.assertFalse(page._detail_canvas.is_dirty)
            self.assertFalse(page._detail_canvas.can_undo)
            self.assertFalse(page._detail_canvas.can_redo)

            QTest.mousePress(page._source_button, Qt.MouseButton.LeftButton)
            self.app.processEvents()
            self.assertTrue(page._detail_canvas.source_preview_visible)
            self.assertFalse(page._is_dirty(selected_id))
            self.assertFalse(page._detail_canvas.is_dirty)
            self.assertFalse(page._detail_canvas.can_undo)

            QTest.mouseRelease(page._source_button, Qt.MouseButton.LeftButton)
            page._reference_overlay_check.setChecked(False)
            self.app.processEvents()
            self.assertFalse(page._detail_canvas.source_preview_visible)
            self.assertFalse(page._detail_canvas.reference_visible)
            self.assertEqual(page._signature(selected_id), saved_signature)
            self.assertEqual(page._detail_canvas.transform(), saved_transform)
            self.assertFalse(page._is_dirty(selected_id))
            self.assertFalse(page._detail_canvas.is_dirty)
            self.assertFalse(page._detail_canvas.can_undo)
            self.assertFalse(page._detail_canvas.can_redo)

    def test_coordination_projection_is_resolved_once_per_refresh_cycle(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            with patch.object(
                consistency_page_module,
                "project_stage_status",
                wraps=consistency_page_module.project_stage_status,
            ) as resolver:
                page._apply_filters()
                self.assertEqual(resolver.call_count, len(page._list_variants))
                page._populate_list()
                page._refresh_statistics()
                self.assertEqual(resolver.call_count, len(page._list_variants))

    def test_thumbnail_rejects_unsafe_or_missing_high_priority_file(self) -> None:
        with self._page_with_sixteen_variants(
            {"移动X": 2.0, "移动Y": 0.0, "等比缩放": 1.0}
        ) as (page, variant_ids):
            detail = page._list_variant_by_id[variant_ids[0]]
            final_dir = Path(page._glyph.get_workflow_dirs()["成品"])
            Image.new("RGBA", (64, 64), (10, 10, 10, 255)).save(
                final_dir / "同名合法文件.png"
            )

            detail["成品文件"] = "../同名合法文件.png"
            source_path, _cache_key = page._list_thumbnail_source(detail)
            self.assertEqual(source_path, "")

            detail["成品文件"] = "不存在的成品.png"
            source_path, _cache_key = page._list_thumbnail_source(detail)
            self.assertEqual(source_path, "")

    def test_detail_canvas_uses_canonical_source_then_post_geometry_ink_preview(self) -> None:
        with self._page_with_sixteen_variants() as (page, variant_ids):
            variant_id = variant_ids[0]
            with (
                patch.object(
                    page._adjustment_service,
                    "prepare_ink_working_copy",
                    wraps=page._adjustment_service.prepare_ink_working_copy,
                ) as prepare_working,
                patch.object(
                    page._adjustment_service,
                    "apply_ink_preview",
                    wraps=page._adjustment_service.apply_ink_preview,
                ) as apply_preview,
            ):
                page._select_variant(variant_id)
                page._load_detail_canvas(variant_id)
                rendered = page._detail_canvas.image()

            self.assertFalse(rendered.isNull())
            prepare_working.assert_called()
            apply_preview.assert_called()
            self.assertEqual(apply_preview.call_args.args[2], variant_id)
            self.assertEqual(
                apply_preview.call_args.args[1]["模式"],
                "跟随全库",
            )
            comparison = page._adjustment_service.preview_coordinated(
                page._variant_by_id[variant_id],
                page._adjustments[variant_id],
                page.WORK_RATIO,
                page._current_ink_config(variant_id),
            )
            self.assertIsNotNone(comparison)
            comparison_image = self._preview_image(comparison)
            detail_image = Image.fromarray(
                ReviewCanvas._qimage_to_rgba(rendered),
                "RGBA",
            )
            left = (comparison_image.width - detail_image.width) // 2
            top = (comparison_image.height - detail_image.height) // 2
            comparison_crop = comparison_image.crop(
                (
                    left,
                    top,
                    left + detail_image.width,
                    top + detail_image.height,
                )
            )
            self.assertEqual(comparison_crop.tobytes(), detail_image.tobytes())
            self.assertFalse(page._detail_canvas.source_preview_visible)
            page._detail_canvas.set_source_preview_visible(True)
            self.assertTrue(page._detail_canvas.source_preview_visible)
            self.assertFalse(page._is_dirty(variant_id))
            self.assertFalse(page._detail_canvas.is_dirty)

    def test_reference_combo_keeps_pinyin_order_independent_from_comparison(self) -> None:
        with self._page_with_sixteen_variants() as (page, variant_ids):
            pinyin_ids = [str(item["变体ID"]) for item in page._variants]
            reference_ids = [
                str(page._reference_combo.itemData(index))
                for index in range(1, page._reference_combo.count())
            ]
            self.assertNotEqual(pinyin_ids, variant_ids)
            self.assertEqual(reference_ids, pinyin_ids)

            page._order_combo.setCurrentText("导入顺序")
            self.app.processEvents()

            self.assertEqual(
                [str(item["变体ID"]) for item in page._variants],
                variant_ids,
            )
            self.assertEqual(
                [
                    str(page._reference_combo.itemData(index))
                    for index in range(1, page._reference_combo.count())
                ],
                pinyin_ids,
            )

    def test_save_selected_updates_baseline_and_save_next_advances(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            self._show_at(page, 1600, 900)
            page._filter_combo.setCurrentText("全部（本阶段）")
            first_id = page._selected_id
            page._enter_detail(first_id)
            self.app.processEvents()

            self.assertTrue(page._detail_canvas.set_transform(x=8.0, y=-5.0, scale=1.12))
            self.assertTrue(page._is_dirty(first_id))
            self.assertEqual(page._coordination_status(page._variant_by_id[first_id]), "待协调")
            self.assertIn(
                "未保存修改",
                page._marker_text(page._workflow_status(page._variant_by_id[first_id])),
            )

            self.assertTrue(page._save_selected(show_success=False))
            self.assertFalse(page._is_dirty(first_id))
            self.assertFalse(page._detail_canvas.is_dirty)
            self.assertFalse(page._detail_canvas.can_undo)
            self.assertFalse(page._detail_canvas.can_redo)
            self.assertEqual(page._saved_signatures[first_id], page._signature(first_id))
            self.assertEqual(
                page._coordination_status(page._variant_by_id[first_id]),
                "已协调",
            )

            current_index = page._variant_index(first_id)
            next_id = str(page._variants[current_index + 1]["变体ID"])
            self.assertTrue(page._detail_canvas.set_transform(x=14.0, rotation=6.0))
            self.assertTrue(page._is_dirty(first_id))
            page._save_and_next()
            self.app.processEvents()

            self.assertEqual(page._selected_id, next_id)
            self.assertIs(page._view_stack.currentWidget(), page._detail_view)
            self.assertFalse(page._is_dirty(first_id))
            self.assertFalse(page._detail_canvas.is_dirty)
            self.assertFalse(page._detail_canvas.can_undo)
            self.assertFalse(page._detail_canvas.can_redo)
            saved_detail = page._glyph.get_variant(first_id)
            self.assertEqual(saved_detail["状态"], config.STATUS_FINISHED)
            self.assertEqual(saved_detail["整体协调参数"]["整体变换"]["移动X"], 14.0)
            self.assertEqual(saved_detail["整体协调参数"]["整体变换"]["旋转"], 6.0)
            finished_path = (
                Path(page._glyph.get_workflow_dirs()["成品"])
                / str(saved_detail["成品文件"])
            )
            self.assertTrue(finished_path.is_file())

    def test_save_button_uses_background_task_without_full_library_reload(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            self._show_at(page, 1600, 900)
            page._filter_combo.setCurrentText("全部（本阶段）")
            page_ids = [
                str(detail["变体ID"])
                for detail in page._page_variants()
            ]
            dirty_ids = page_ids[:2]
            for offset, variant_id in enumerate(dirty_ids, 1):
                page._adjustments[variant_id]["移动X"] = float(offset)
                page._sync_dirty_variant(variant_id)

            with (
                patch.object(page, "_reload_variants", wraps=page._reload_variants) as reload_all,
                patch.object(page, "_clear_preview_cache", wraps=page._clear_preview_cache) as clear_all,
                patch.object(QMessageBox, "information") as information,
            ):
                page._save_action()
                self.assertTrue(page._coordination_busy)
                self.assertTrue(
                    self._wait_until(
                        lambda: not page._coordination_busy,
                        timeout=8.0,
                    )
                )

            reload_all.assert_not_called()
            clear_all.assert_not_called()
            information.assert_called_once()
            for variant_id in page_ids:
                expected = (
                    config.STATUS_FINISHED
                    if variant_id in dirty_ids
                    else config.STATUS_REVIEWED
                )
                self.assertEqual(page._glyph.get_variant(variant_id)["状态"], expected)
            self.assertEqual(page._page_index, 0)

    def test_refreshing_same_virtual_range_keeps_cards_attached_and_reuses_them(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            self._show_at(page, 1600, 900)
            old_cards = dict(page._cards)
            self.assertTrue(old_cards)
            self.assertTrue(all(not card.isWindow() for card in old_cards.values()))

            page._render_page()

            self.assertEqual(set(page._cards), set(old_cards))
            self.assertTrue(
                all(page._cards[variant_id] is card for variant_id, card in old_cards.items())
            )
            self.assertTrue(all(not card.isWindow() for card in old_cards.values()))
            top_level_widgets = set(QApplication.topLevelWidgets())
            self.assertTrue(all(card not in top_level_widgets for card in old_cards.values()))
            self.assertTrue(all(not card.isWindow() for card in page._cards.values()))

    def test_save_and_next_on_last_result_shows_end_notice(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            self._show_at(page, 1600, 900)
            last_id = str(page._variants[-1]["变体ID"])
            page._select_variant(last_id)
            page._enter_detail(last_id)
            self.app.processEvents()

            with patch.object(QMessageBox, "information") as information:
                page._save_and_next_async()
                self.assertTrue(
                    self._wait_until(
                        lambda: not page._coordination_busy,
                        timeout=8.0,
                    )
                )

            information.assert_called_once()
            self.assertEqual(information.call_args.args[1], "整体协调")
            self.assertIn("最后一条", information.call_args.args[2])
            self.assertEqual(page._selected_id, last_id)

    def test_saving_dirty_comparison_result_keeps_unpaginated_position(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            self._show_at(page, 1600, 900)
            selected_id = page._selected_id
            page._adjustments[selected_id]["移动X"] = 3.0
            page._sync_dirty_variant(selected_id)

            with patch.object(QMessageBox, "information") as information:
                page._save_action()
                self.assertTrue(
                    self._wait_until(
                        lambda: not page._coordination_busy,
                        timeout=8.0,
                    )
                )

            information.assert_called_once()
            self.assertEqual(information.call_args.args[1], "保存全部修改")
            self.assertIn("已保存 1 个字形", information.call_args.args[2])
            self.assertEqual(page._page_index, 0)
            self.assertEqual(page._selected_id, selected_id)

    def test_complete_coordination_runs_in_background_and_restores_after_failure(self) -> None:
        with self._page_with_sixteen_variants() as (page, variant_ids):
            self._show_at(page, 1100, 720)
            started = threading.Event()
            release = threading.Event()

            def blocking_failure(
                _service: AdjustmentService,
                variants: list[dict[str, object]],
                _adjustments: dict[str, dict[str, object]],
                _ink_config: dict[str, object] | None = None,
                _baseline: dict[str, object] | None = None,
                source_images_by_id: dict[str, Image.Image] | None = None,
                progress_callback: object = None,
                cancel_check: object = None,
                commit_gate: object = None,
                include_metrics: bool = False,
            ) -> dict[str, object]:
                started.set()
                if callable(progress_callback):
                    progress_callback(
                        "准备",
                        5,
                        1,
                        len(variants),
                        "天 · 天-0001.png",
                    )
                release.wait(2.0)
                raise RuntimeError("模拟后台事务失败")

            try:
                with (
                    patch.object(
                        page,
                        "_confirm_complete_coordination",
                        return_value=True,
                    ),
                    patch.object(
                        AdjustmentService,
                        "save_coordinated_variants",
                        new=blocking_failure,
                    ),
                    patch.object(QMessageBox, "critical") as critical,
                ):
                    page._complete_coordination()

                    self.assertTrue(self._wait_until(started.is_set))
                    self.assertTrue(page._coordination_busy)
                    self.assertTrue(page._task_progress_panel.isVisible())
                    self.assertFalse(page._main_splitter.isEnabled())
                    self.assertFalse(page._complete_button.isEnabled())
                    self.assertFalse(page._back_button.isEnabled())
                    self.assertFalse(page._confirm_leave_page())
                    self.assertIsNot(page._progress_bar, page._task_progress_bar)
                    self.assertEqual(page._progress_bar.format(), "完成度 %p%")

                    heartbeat: list[bool] = []
                    QTimer.singleShot(0, lambda: heartbeat.append(True))
                    self.assertTrue(self._wait_until(lambda: bool(heartbeat)))
                    self.assertIn("准备", page._task_stage_label.text())
                    self.assertIn("1 / 16", page._task_detail_label.toolTip())

                    release.set()
                    self.assertTrue(
                        self._wait_until(lambda: not page._coordination_busy)
                    )

                    self.assertTrue(page._main_splitter.isEnabled())
                    self.assertTrue(page._complete_button.isEnabled())
                    self.assertTrue(page._back_button.isEnabled())
                    self.assertEqual(page._task_stage_label.text(), "本次执行：失败")
                    self.assertFalse(page._task_progress_panel.isVisible())
                    self.assertIn("回滚", page._task_detail_label.text())
                    self.assertEqual(len(page._all_variants), len(variant_ids))
                    critical.assert_called_once()
            finally:
                release.set()

    def test_complete_coordination_applies_worker_state_only_after_success(self) -> None:
        with self._page_with_sixteen_variants() as (page, variant_ids):
            self._show_at(page, 1100, 720)
            page._task_progress_bar.setRange(0, len(variant_ids))
            page._task_progress_bar.setValue(len(variant_ids))
            original_state = page._glyph.snapshot_state()
            selected_id = page._selected_id
            page._adjustments[selected_id]["移动X"] = 7.0

            with patch.object(QMessageBox, "information") as information:
                with patch.object(
                    page,
                    "_confirm_complete_coordination",
                    return_value=True,
                ):
                    page._complete_coordination()
                    self.assertTrue(page._coordination_busy)
                    self.assertEqual(page._glyph.snapshot_state(), original_state)
                    self.assertTrue(
                        self._wait_until(
                            lambda: not page._coordination_busy,
                            timeout=8.0,
                        )
                    )

            self.assertEqual(page._task_stage_label.text(), "本次执行：已完成")
            self.assertFalse(page._task_progress_panel.isVisible())
            self.assertEqual(page._task_progress_bar.minimum(), 0)
            self.assertEqual(page._task_progress_bar.maximum(), 100)
            self.assertEqual(page._task_progress_bar.value(), 100)
            self.assertEqual(page._progress_bar.value(), 100)
            self.assertTrue(page._main_splitter.isEnabled())
            self.assertFalse(page._complete_button.isEnabled())
            self.assertTrue(page._back_button.isEnabled())
            self.assertFalse(any(page._is_dirty(item) for item in variant_ids))
            self.assertTrue(
                all(
                    page._glyph.get_variant(item)["状态"] == config.STATUS_FINISHED
                    for item in variant_ids
                )
            )
            self.assertEqual(
                page._glyph.get_variant(selected_id)["整体协调参数"]["整体变换"]["移动X"],
                7.0,
            )
            information.assert_called_once()
            self.assertEqual(
                information.call_args.args[:2],
                (page, "完成整体协调"),
            )
            self.assertIn("全库整体协调结果已生成。", information.call_args.args[2])
            self.assertIn("总耗时：", information.call_args.args[2])

    def test_coordination_progress_resets_baseline_range_and_never_regresses(self) -> None:
        with self._page_with_sixteen_variants() as (page, variant_ids):
            page._task_progress_bar.setRange(0, len(variant_ids))
            page._task_progress_bar.setValue(0)
            page._coordination_task = None
            page._coordination_task_total = len(variant_ids)
            page._set_coordination_busy(True)

            page._coordination_progress_changed(
                "准备",
                5,
                1,
                len(variant_ids),
                "天 · 天-0001.png",
            )
            self.assertEqual(page._task_progress_bar.maximum(), 100)
            self.assertEqual(page._task_progress_bar.value(), 5)

            page._coordination_progress_changed(
                "准备",
                3,
                1,
                len(variant_ids),
                "天 · 天-0001.png",
            )
            self.assertEqual(page._task_progress_bar.value(), 5)
            page._finish_coordination_task(False, "测试结束")

    def test_coordination_finished_warns_when_post_commit_audit_remains_pending(
        self,
    ) -> None:
        with self._page_with_sixteen_variants() as (page, variant_ids):
            result = page._adjustment_service.save_coordinated_variants(
                page._all_variants,
                page._adjustments,
                {"启用": True, "基准": page._ink_baseline},
                page._coordination_baseline,
            )
            self.assertEqual(result["成功"], len(variant_ids))
            pending_detail = page._glyph.get_variant(variant_ids[0])
            pending_record = pending_detail["整体协调参数"]["墨色协调"]
            pending_record["是否达标"] = False
            pending_record["人工接受例外"] = False

            page._coordination_task = object()  # type: ignore[assignment]
            page._coordination_task_total = len(variant_ids)
            page._set_coordination_busy(True)
            payload = {
                "结果": result,
                "字库状态": page._glyph.snapshot_state(),
            }
            with (
                patch.object(QMessageBox, "warning") as warning,
                patch.object(QMessageBox, "information") as information,
            ):
                page._coordination_finished(payload)

            self.assertFalse(page.is_batch_running)
            self.assertEqual(
                page._task_stage_label.text(),
                "本次执行：已完成，需核对",
            )
            self.assertEqual(page._task_progress_bar.maximum(), 100)
            self.assertEqual(page._task_progress_bar.value(), 100)
            self.assertIn("仍有 1 个待核对", page._task_detail_label.text())
            self.assertIn("待协调 1", page._summary_label.text())
            self.assertIn("墨色待确认 1 个", warning.call_args.args[2])
            self.assertIn("总耗时：", warning.call_args.args[2])
            information.assert_not_called()

    def test_complete_coordination_only_submits_pending_admitted_records(self) -> None:
        with self._page_with_sixteen_variants(
            {"移动X": 2.0, "移动Y": 0.0, "等比缩放": 1.0}
        ) as (page, variant_ids):
            coordinated_id = variant_ids[0]
            started: list[object] = []
            with (
                patch.object(
                    page,
                    "_confirm_complete_coordination",
                    return_value=True,
                ),
                patch.object(
                    page._coordination_pool,
                    "start",
                    side_effect=started.append,
                ),
            ):
                page._complete_coordination()

            self.assertEqual(len(started), 1)
            task = started[0]
            self.assertNotIn(coordinated_id, task._variant_ids)
            self.assertEqual(set(task._variant_ids), set(variant_ids[1:]))
            self.assertEqual(page._coordination_task_total, len(variant_ids) - 1)
            page._finish_coordination_task(False, "测试结束")

    def test_reopen_restores_current_saved_post_geometry_ink_baseline(self) -> None:
        with self._page_with_sixteen_variants() as (page, variant_ids):
            with (
                patch.object(page, "_confirm_complete_coordination", return_value=True),
                patch.object(QMessageBox, "information"),
            ):
                page._complete_coordination()
                self.assertTrue(
                    self._wait_until(
                        lambda: not page._coordination_busy,
                        timeout=8.0,
                    )
                )
            saved_baseline = page._ink_baseline
            saved_summary = page._glyph.get_coordination_summary()
            self.assertEqual(
                saved_summary["墨色方法版本"],
                AdjustmentService.INK_METHOD_VERSION,
            )

            with patch.object(
                AdjustmentService,
                "analyze",
                side_effect=AssertionError("有效正式基准不应重新分析"),
            ):
                reopened = ConsistencyPage(page._glyph, lambda: None)
            try:
                self.assertFalse(reopened._baseline_analysis_pending)
                self.assertEqual(reopened._ink_baseline, saved_baseline)
                self.assertFalse(any(reopened._is_dirty(item) for item in variant_ids))
                self.assertEqual(reopened._progress_bar.value(), 100)
            finally:
                reopened.close()
                reopened.deleteLater()
                self.app.processEvents()

    def test_coordination_reload_failure_still_restores_navigation(self) -> None:
        with self._page_with_sixteen_variants() as (page, variant_ids):
            page._coordination_task = object()  # type: ignore[assignment]
            page._coordination_task_total = len(variant_ids)
            page._set_coordination_busy(True)

            with (
                patch.object(
                    page,
                    "_reload_variants",
                    side_effect=RuntimeError("模拟页面重载失败"),
                ),
                patch.object(QMessageBox, "critical") as critical,
            ):
                page._coordination_failed("模拟后台事务失败")

            self.assertFalse(page._coordination_busy)
            self.assertIsNone(page._coordination_task)
            self.assertTrue(page._main_splitter.isEnabled())
            self.assertTrue(page._complete_button.isEnabled())
            self.assertTrue(page._back_button.isEnabled())
            self.assertTrue(all(action.isEnabled() for action in page._shortcut_actions))
            self.assertIn("页面重载失败", critical.call_args.args[2])
            self.assertIn("返回首页", critical.call_args.args[2])
            self.assertIn("总耗时：", critical.call_args.args[2])

    def test_incomplete_rollback_is_reported_without_false_success_claim(self) -> None:
        with self._page_with_sixteen_variants() as (page, variant_ids):
            page._coordination_task = object()  # type: ignore[assignment]
            page._coordination_task_total = len(variant_ids)
            page._set_coordination_busy(True)

            with patch.object(QMessageBox, "critical") as critical:
                page._coordination_failed(
                    "整体协调保存失败，且回滚未完全完成：旧成品恢复失败"
                )

            self.assertFalse(page.is_batch_running)
            self.assertEqual(page._task_detail_label.text(), "回滚未完全完成")
            message = critical.call_args.args[2]
            self.assertIn("回滚未完全完成", message)
            self.assertIn("可能存在未恢复文件", message)
            self.assertNotIn("未保留部分结果", message)
            self.assertIn("总耗时：", message)

    def test_post_commit_snapshot_failure_emits_terminal_result_and_unlocks_page(
        self,
    ) -> None:
        with self._page_with_sixteen_variants() as (page, variant_ids):
            task = consistency_page_module._CoordinationTask(
                page._glyph.ziku_name,
                page._glyph.ziku_dir,
                page._glyph.snapshot_state(),
                variant_ids,
                deepcopy(page._adjustments),
                page._current_ink_config(),
                deepcopy(page._coordination_baseline),
            )
            payloads: list[object] = []
            failures: list[str] = []
            task.signals.finished.connect(payloads.append)
            task.signals.failed.connect(failures.append)
            result = {
                "成功": len(variant_ids),
                "失败": 0,
                "失败详情": [],
            }
            with (
                patch.object(
                    AdjustmentService,
                    "save_coordinated_variants",
                    return_value=result,
                ),
                patch.object(
                    GlyphService,
                    "snapshot_state",
                    side_effect=MemoryError("模拟提交后状态快照失败"),
                ),
            ):
                task.run()

            self.assertFalse(failures)
            self.assertEqual(len(payloads), 1)
            payload = payloads[0]
            self.assertIsInstance(payload, dict)
            self.assertTrue(payload["已提交"])
            self.assertIn("状态快照失败", payload["字库状态错误"])

            page._coordination_task = task
            page._coordination_task_total = len(variant_ids)
            page._set_coordination_busy(True)
            with patch.object(QMessageBox, "critical") as critical:
                page._coordination_finished(payload)

            self.assertFalse(page.is_batch_running)
            self.assertIsNone(page._coordination_task)
            self.assertTrue(page._main_splitter.isEnabled())
            self.assertTrue(page._complete_button.isEnabled())
            self.assertTrue(page._back_button.isEnabled())
            self.assertIn("批次已提交", page._task_detail_label.text())
            self.assertIn("已经提交", critical.call_args.args[2])
            self.assertIn("返回首页", critical.call_args.args[2])

    def test_coordination_success_refresh_failure_still_restores_navigation(self) -> None:
        with self._page_with_sixteen_variants() as (page, variant_ids):
            page._coordination_task = object()  # type: ignore[assignment]
            page._coordination_task_total = len(variant_ids)
            page._coordination_task_ink = page._current_ink_config()
            page._set_coordination_busy(True)
            payload = {
                "结果": {
                    "成功": len(variant_ids),
                    "失败": 0,
                    "失败详情": [],
                },
                "字库状态": page._glyph.snapshot_state(),
            }

            with (
                patch.object(
                    page,
                    "_apply_filters",
                    side_effect=RuntimeError("模拟成功后页面刷新失败"),
                ),
                patch.object(QMessageBox, "critical") as critical,
                patch.object(QMessageBox, "information") as information,
            ):
                page._coordination_finished(payload)

            self.assertFalse(page._coordination_busy)
            self.assertIsNone(page._coordination_task)
            self.assertTrue(page._main_splitter.isEnabled())
            self.assertTrue(page._complete_button.isEnabled())
            self.assertTrue(page._back_button.isEnabled())
            self.assertTrue(all(action.isEnabled() for action in page._shortcut_actions))
            self.assertIn("已经提交", critical.call_args.args[2])
            self.assertIn("返回首页", critical.call_args.args[2])
            information.assert_not_called()

    def test_complete_coordination_confirmation_cancel_does_not_start_task(self) -> None:
        with self._page_with_sixteen_variants() as (page, variant_ids):
            self._show_at(page, 1100, 720)
            state_before = page._glyph.snapshot_state()

            with patch.object(
                QMessageBox,
                "exec",
                autospec=True,
                return_value=QMessageBox.StandardButton.Cancel.value,
            ) as exec_dialog:
                page._complete_coordination()

            exec_dialog.assert_called_once()
            self.assertFalse(page._coordination_busy)
            self.assertIsNone(page._coordination_task)
            self.assertFalse(page._task_progress_panel.isVisible())
            self.assertEqual(page._glyph.snapshot_state(), state_before)
            dialogs = page.findChildren(QMessageBox)
            self.assertTrue(dialogs)
            dialog = dialogs[-1]
            self.assertIn(str(len(variant_ids)), dialog.text())
            self.assertIn(str(len(variant_ids)), dialog.text())
            self.assertIn("默认参数", dialog.informativeText())
            self.assertIn("不能替代逐字视觉比较", dialog.informativeText())
            self.assertIn("墨色统一", dialog.informativeText())
            self.assertIn("更新已有成品", dialog.informativeText())
            confirm_button = dialog.button(QMessageBox.StandardButton.Ok)
            cancel_button = dialog.button(QMessageBox.StandardButton.Cancel)
            self.assertEqual(len(dialog.buttons()), 2)
            self.assertIsNotNone(confirm_button)
            self.assertIsNotNone(cancel_button)
            self.assertEqual(confirm_button.text(), "确定")
            self.assertEqual(cancel_button.text(), "取消")
            self.assertIs(dialog.defaultButton(), cancel_button)
            self.assertIs(dialog.escapeButton(), cancel_button)

    def test_stop_coordination_restores_ui_without_applying_worker_snapshot(self) -> None:
        with self._page_with_sixteen_variants() as (page, variant_ids):
            self._show_at(page, 1100, 720)
            selected_id = page._selected_id
            page._adjustments[selected_id]["移动X"] = 9.0
            state_before = page._glyph.snapshot_state()
            draft_before = deepcopy(page._adjustments)
            started = threading.Event()

            def wait_for_cancel(
                _service: AdjustmentService,
                _variants: list[dict[str, object]],
                _adjustments: dict[str, dict[str, object]],
                _ink_config: dict[str, object] | None = None,
                _baseline: dict[str, object] | None = None,
                source_images_by_id: dict[str, Image.Image] | None = None,
                progress_callback: object = None,
                cancel_check: object = None,
                commit_gate: object = None,
                include_metrics: bool = False,
            ) -> dict[str, object]:
                started.set()
                while not (callable(cancel_check) and cancel_check()):
                    time.sleep(0.005)
                if callable(commit_gate) and commit_gate():
                    raise AssertionError("停止请求已受理后不得进入提交阶段")
                raise CoordinationCancelled("已停止，本批次未提交")

            with (
                patch.object(page, "_confirm_complete_coordination", return_value=True),
                patch.object(page, "_confirm_stop_coordination", return_value=True),
                patch.object(
                    AdjustmentService,
                    "save_coordinated_variants",
                    new=wait_for_cancel,
                ),
                patch.object(QMessageBox, "information") as information,
            ):
                page._complete_coordination()
                self.assertTrue(self._wait_until(started.is_set))
                self.assertTrue(page.is_batch_running)
                self.assertTrue(page._stop_coordination_button.isVisible())
                self.assertTrue(page._stop_coordination_button.isEnabled())

                page._request_stop_coordination()

                self.assertIn("正在停止", page._task_stage_label.text())
                self.assertEqual(page._stop_coordination_button.text(), "正在停止…")
                self.assertFalse(page._stop_coordination_button.isEnabled())
                self.assertTrue(
                    self._wait_until(lambda: not page.is_batch_running)
                )

            self.assertEqual(page._task_stage_label.text(), "本次执行：已停止")
            self.assertFalse(page._task_progress_panel.isVisible())
            self.assertIn("已停止，本批次未提交", page._task_detail_label.text())
            self.assertFalse(page._stop_coordination_button.isVisible())
            self.assertTrue(page._main_splitter.isEnabled())
            self.assertTrue(page._complete_button.isEnabled())
            self.assertTrue(page._back_button.isEnabled())
            self.assertTrue(all(action.isEnabled() for action in page._shortcut_actions))
            self.assertEqual(page._glyph.snapshot_state(), state_before)
            self.assertEqual(page._adjustments, draft_before)
            self.assertTrue(page._is_dirty(selected_id))
            self.assertTrue(all(page._glyph.get_variant(item)["状态"] == config.STATUS_REVIEWED for item in variant_ids))
            information.assert_called_once()
            self.assertIn("本批次未提交", information.call_args.args[2])
            self.assertIn("总耗时：", information.call_args.args[2])

    def test_stop_confirmation_cannot_overwrite_a_task_that_finished_in_dialog(self) -> None:
        with self._page_with_sixteen_variants() as (page, variant_ids):
            class FinishedDuringDialogTask:
                def __init__(self) -> None:
                    self.cancel_requested = False

                def is_cancel_requested(self) -> bool:
                    return self.cancel_requested

                def is_commit_started(self) -> bool:
                    return False

                def request_cancel(self) -> bool:
                    self.cancel_requested = True
                    return True

            task = FinishedDuringDialogTask()
            page._coordination_task = task  # type: ignore[assignment]
            page._coordination_task_total = len(variant_ids)
            page._set_coordination_busy(True)
            page._set_coordination_stop_state(running=True)

            def finish_while_confirming() -> bool:
                page._finish_coordination_task(
                    True,
                    f"{len(variant_ids)} / {len(variant_ids)} · 批次提交完成",
                )
                return True

            with patch.object(
                page,
                "_confirm_stop_coordination",
                side_effect=finish_while_confirming,
            ):
                page._request_stop_coordination()

            self.assertFalse(task.cancel_requested)
            self.assertFalse(page.is_batch_running)
            self.assertIsNone(page._coordination_task)
            self.assertEqual(page._task_stage_label.text(), "本次执行：已完成")
            self.assertNotIn("正在停止", page._task_progress_bar.format())
            self.assertFalse(page._stop_coordination_button.isVisible())

    def test_late_stop_request_shows_commit_state(self) -> None:
        with self._page_with_sixteen_variants() as (page, variant_ids):
            task = consistency_page_module._CoordinationTask(
                page._glyph.ziku_name,
                page._glyph.ziku_dir,
                page._glyph.snapshot_state(),
                variant_ids,
                deepcopy(page._adjustments),
                page._current_ink_config(),
                deepcopy(page._coordination_baseline),
            )
            page._coordination_task = task
            page._coordination_task_total = len(variant_ids)
            page._set_coordination_busy(True)
            page._set_coordination_stop_state(running=True)

            def begin_commit_while_confirming() -> bool:
                self.assertTrue(task.try_begin_commit())
                return True

            with patch.object(
                page,
                "_confirm_stop_coordination",
                side_effect=begin_commit_while_confirming,
            ):
                page._request_stop_coordination()

            self.assertTrue(task.is_commit_started())
            self.assertFalse(task.is_cancel_requested())
            self.assertEqual(page._task_stage_label.text(), "本次执行：提交")
            self.assertEqual(page._stop_coordination_button.text(), "正在提交…")
            self.assertFalse(page._stop_coordination_button.isEnabled())
            self.assertIn("停止请求未受理", page._task_detail_label.text())
            page._finish_coordination_task(False, "测试结束")

    def test_save_and_next_uses_successor_before_status_filter_shrinks(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            self._show_at(page, 1600, 900)
            ordered_ids = [str(item["变体ID"]) for item in page._variants]
            dirty_ids = ordered_ids[:3]
            for offset, variant_id in enumerate(dirty_ids, 1):
                page._adjustments[variant_id]["移动X"] = float(offset)

            page._filter_combo.setCurrentText("待协调")
            page._select_variant(dirty_ids[0])
            page._enter_detail(dirty_ids[0])
            self.app.processEvents()
            self.assertEqual(len(page._variants), len(ordered_ids))

            page._save_and_next()
            self.app.processEvents()

            self.assertEqual(page._selected_id, dirty_ids[1])
            self.assertNotIn(
                dirty_ids[0],
                [str(item["变体ID"]) for item in page._variants],
            )
            self.assertEqual(len(page._variants), len(ordered_ids) - 1)
            self.assertIs(page._view_stack.currentWidget(), page._detail_view)
            self.assertFalse(page._is_dirty(dirty_ids[0]))
            self.assertTrue(page._is_dirty(dirty_ids[1]))

    def test_geometry_status_filter_reapplies_without_losing_detail_draft(self) -> None:
        with self._page_with_sixteen_variants(
            {"移动X": 0.0, "移动Y": 0.0, "等比缩放": 1.0}
        ) as (page, variant_ids):
            self._show_at(page, 1600, 900)
            coordinated_id = variant_ids[0]
            page._filter_combo.setCurrentText("已协调")
            page._enter_detail(coordinated_id)
            self.app.processEvents()
            self.assertEqual(
                [str(item["变体ID"]) for item in page._variants],
                [coordinated_id],
            )

            self.assertTrue(page._detail_canvas.set_transform(x=19.0, rotation=5.0))
            self.app.processEvents()

            self.assertEqual(
                [str(item["变体ID"]) for item in page._variants],
                [coordinated_id],
            )
            self.assertEqual(page._glyph_list.topLevelItemCount(), 1)
            self.assertEqual(page._selected_id, coordinated_id)
            self.assertTrue(page._is_dirty(coordinated_id))
            self.assertAlmostEqual(page._adjustments[coordinated_id]["移动X"], 19.0)
            self.assertAlmostEqual(page._detail_canvas.transform()["x"], 19.0)

            page._apply_filters()
            self.app.processEvents()

            self.assertEqual(
                [str(item["变体ID"]) for item in page._variants],
                [coordinated_id],
            )
            self.assertEqual(page._selected_id, coordinated_id)
            self.assertAlmostEqual(page._adjustments[coordinated_id]["移动X"], 19.0)
            self.assertAlmostEqual(page._detail_canvas.transform()["x"], 19.0)
            self.assertAlmostEqual(page._detail_canvas.transform()["rotation"], 5.0)

    def test_ink_option_changes_preserve_phase_filter_and_selection(self) -> None:
        with self._page_with_sixteen_variants() as (page, variant_ids):
            self._show_at(page, 1600, 900)
            selected_id = page._selected_id
            page._enter_detail(selected_id)
            page._filter_combo.setCurrentText("待协调")
            self.assertEqual(len(page._variants), len(variant_ids))

            page._ink_check.setChecked(False)
            self.app.processEvents()

            self.assertEqual(len(page._variants), len(variant_ids))
            self.assertEqual(page._glyph_list.topLevelItemCount(), len(variant_ids))
            self.assertEqual(page._selected_id, selected_id)
            self.assertTrue(page._detail_canvas.has_image)

            page._ink_check.setChecked(True)
            self.app.processEvents()

            self.assertEqual(len(page._variants), len(variant_ids))
            self.assertEqual(page._glyph_list.topLevelItemCount(), len(variant_ids))
            self.assertEqual(page._selected_id, selected_id)
            self.assertTrue(page._detail_canvas.has_image)
            self.assertFalse(hasattr(page, "_problem_filter_combo"))

    def test_left_tree_only_lists_records_admitted_to_coordination(self) -> None:
        with self._page_with_sixteen_variants() as (page, variant_ids):
            optimization_id, review_id = variant_ids[:2]
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
            page._glyph.save()
            page._reload_variants()

            self.assertEqual(page._filter_combo.currentText(), "全部（本阶段）")
            self.assertEqual(len(page._list_visible_variants), 14)
            self.assertEqual(len(page._variants), 14)
            self.assertEqual(page._summary_label.text(), "待协调 14　已协调 0\n未保存 0　问题 0")
            self.assertNotIn(optimization_id, page._list_variant_by_id)
            self.assertNotIn(review_id, page._list_variant_by_id)
            self.assertNotIn(optimization_id, page._list_items_by_id)
            self.assertNotIn(review_id, page._list_items_by_id)
            self.assertEqual(
                [page._filter_combo.itemText(index) for index in range(page._filter_combo.count())],
                ["全部（本阶段）", "待协调", "已协调"],
            )
            self.assertFalse(hasattr(page, "_problem_filter_combo"))

    def test_ink_records_distinguish_reached_pending_and_manual_exception(self) -> None:
        with self._page_with_sixteen_variants() as (page, variant_ids):
            reached_id, pending_id, exception_id = variant_ids[:3]
            ink_baseline = page._ink_baseline
            records = {
                reached_id: {
                    "启用": True,
                    "基准": ink_baseline,
                    "方法": AdjustmentService.INK_METHOD,
                    "方法版本": AdjustmentService.INK_METHOD_VERSION,
                    "模式": "跟随全库",
                    "调整前墨色": 170.0,
                    "调整后墨色": 189.0,
                    "保存后墨色": 188.5,
                    "保存后复测": True,
                    "目标偏差": 1.0,
                    "像素类型": "灰度",
                    "是否达标": True,
                    "状态": "已达标",
                    "已应用": True,
                    "跳过原因": "",
                },
                pending_id: {
                    "启用": True,
                    "基准": ink_baseline,
                    "方法": AdjustmentService.INK_METHOD,
                    "方法版本": AdjustmentService.INK_METHOD_VERSION,
                    "模式": "跟随全库",
                    "调整前墨色": 150.0,
                    "调整后墨色": 168.0,
                    "目标偏差": 22.0,
                    "像素类型": "灰度",
                    # 旧记录即使写过“已应用”，没有实测达标也必须待确认。
                    "已应用": True,
                    "跳过原因": "调整达到限制",
                },
                exception_id: {
                    "启用": True,
                    "基准": ink_baseline,
                    "方法": AdjustmentService.INK_METHOD,
                    "方法版本": AdjustmentService.INK_METHOD_VERSION,
                    "模式": "人工例外",
                    "调整前墨色": 255.0,
                    "调整后墨色": 255.0,
                    "保存后墨色": 255.0,
                    "保存后复测": True,
                    "目标偏差": 65.0,
                    "像素类型": "纯二值",
                    "是否达标": False,
                    "状态": "人工例外",
                    "人工接受例外": True,
                    "已应用": False,
                    "跳过原因": "用户保留纯黑",
                },
            }
            final_dir = Path(page._glyph.get_workflow_dirs()["成品"])
            for variant_id, record in records.items():
                detail = page._variant_by_id[variant_id]
                detail["状态"] = config.STATUS_FINISHED
                detail["成品文件"] = f"{variant_id}.png"
                Image.new("RGBA", (64, 64), (20, 20, 20, 220)).save(
                    final_dir / detail["成品文件"]
                )
                detail["整体协调参数"] = {
                    "整体变换": deepcopy(page._adjustments[variant_id]),
                    "墨色协调": record,
                }
                page._ink_modes[variant_id] = page._stored_ink_mode(detail)
                page._saved_ink_signatures[variant_id] = page._stored_ink_signature(detail)

            page._glyph.set_coordination_summary(
                page._coordination_baseline,
                ink_baseline,
                geometry_completed=False,
                ink_completed=False,
                ink_enabled=True,
                ink_method=AdjustmentService.INK_METHOD,
                ink_method_version=AdjustmentService.INK_METHOD_VERSION,
            )
            page._filter_combo.setCurrentText("全部（本阶段）")
            page._apply_filters()
            self.assertEqual(page._coordination_status(page._variant_by_id[reached_id]), "已协调")
            self.assertEqual(page._coordination_status(page._variant_by_id[pending_id]), "待协调")
            self.assertEqual(page._coordination_status(page._variant_by_id[exception_id]), "已协调")
            self.assertEqual(
                page._list_items_by_id[reached_id].text(1).splitlines()[1],
                "无",
            )
            self.assertEqual(
                page._list_items_by_id[pending_id].text(1).splitlines()[1],
                "墨色待确认",
            )
            self.assertEqual(
                page._list_items_by_id[exception_id].text(1).splitlines()[1],
                "人工例外",
            )
            self.assertIn("墨色达标 1", page._ink_summary_label.text())
            self.assertIn("待确认 1", page._ink_summary_label.text())
            self.assertIn("人工例外 1", page._ink_summary_label.text())
            self.assertIn("已协调 2", page._summary_label.text())
            self.assertEqual(page._progress_bar.value(), 12)
            page._select_variant(reached_id)
            self.assertIn("保存后 188.50（持久化复测）", page._ink_result_label.text())

            page._filter_combo.setCurrentText("待协调")
            self.app.processEvents()
            pending_ids = {
                str(detail["变体ID"]) for detail in page._variants
            }
            self.assertIn(pending_id, pending_ids)
            self.assertNotIn(reached_id, pending_ids)
            self.assertNotIn(exception_id, pending_ids)

            page._filter_combo.setCurrentText("已协调")
            self.app.processEvents()
            self.assertEqual(
                {str(detail["变体ID"]) for detail in page._variants},
                {reached_id, exception_id},
            )

    def test_per_glyph_ink_mode_is_dirty_and_saved_only_for_requested_glyph(self) -> None:
        with self._page_with_sixteen_variants() as (page, variant_ids):
            selected_id, untouched_id = variant_ids[:2]
            page._select_variant(selected_id)
            self.assertFalse(page._is_dirty(selected_id))
            self.assertFalse(page._is_dirty(untouched_id))

            page._ink_strategy_combo.setCurrentText("保留本字")
            self.app.processEvents()

            self.assertTrue(page._is_dirty(selected_id))
            self.assertFalse(page._is_dirty(untouched_id))
            config_data = page._current_ink_config(
                variant_ids={selected_id, untouched_id}
            )
            self.assertEqual(
                config_data["逐字模式"],
                {selected_id: "保留本字", untouched_id: "跟随全库"},
            )

            real_save = page._adjustment_service.save_coordinated_variants
            with patch.object(
                page._adjustment_service,
                "save_coordinated_variants",
                wraps=real_save,
            ) as save_spy:
                self.assertTrue(page._save_selected(show_success=False))

            submitted = save_spy.call_args.args[2]
            self.assertEqual(submitted["逐字模式"], {selected_id: "保留本字"})
            self.assertFalse(page._is_dirty(selected_id))
            self.assertFalse(page._is_dirty(untouched_id))

    def test_leave_confirmation_uses_chinese_button_labels(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            self._show_at(page, 1100, 720)
            selected_id = page._selected_id
            page._enter_detail(selected_id)
            self.assertTrue(page._detail_canvas.set_transform(x=9.0))

            with patch.object(
                QMessageBox,
                "exec",
                autospec=True,
                return_value=QMessageBox.StandardButton.Cancel.value,
            ) as exec_dialog:
                self.assertFalse(page._confirm_leave_changes())

            exec_dialog.assert_called_once()
            dialogs = page.findChildren(QMessageBox)
            self.assertTrue(dialogs)
            dialog = dialogs[-1]
            save_button = dialog.button(QMessageBox.StandardButton.Save)
            discard_button = dialog.button(QMessageBox.StandardButton.Discard)
            cancel_button = dialog.button(QMessageBox.StandardButton.Cancel)
            self.assertIsNotNone(save_button)
            self.assertIsNotNone(discard_button)
            self.assertIsNotNone(cancel_button)
            self.assertEqual(save_button.text(), "保存修改")
            self.assertEqual(discard_button.text(), "放弃修改")
            self.assertEqual(cancel_button.text(), "取消")
            self.assertIs(dialog.defaultButton(), save_button)
            self.assertTrue(page._is_dirty(selected_id))
            self.assertTrue(page._detail_canvas.is_dirty)

    def test_comparison_view_never_exposes_horizontal_scrolling(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            for width, height in ((1600, 900), (1100, 720)):
                with self.subTest(size=(width, height)):
                    self._show_at(page, width, height)
                    self.assertEqual(
                        page._comparison_scroll.horizontalScrollBarPolicy(),
                        Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
                    )
                    self.assertEqual(
                        page._comparison_scroll.horizontalScrollBar().maximum(),
                        0,
                    )

    def test_phase_filter_does_not_force_page_wider_than_main_window(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            self._show_at(page, 1100, 720)

            self.assertLessEqual(page.minimumSizeHint().width(), 1100)
            self.assertEqual(page.size(), QSize(1100, 720))
            self.assertFalse(hasattr(page, "_problem_filter_combo"))
            self.assertGreater(page._filter_combo.width(), 0)
            self.assertGreaterEqual(
                page._filter_combo.width(),
                page._filter_combo.minimumSizeHint().width(),
            )

    def test_saved_nonzero_adjustment_loads_as_clean_canvas_baseline(self) -> None:
        saved = {
            "移动X": 13.0,
            "移动Y": -7.0,
            "等比缩放": 1.15,
            "水平拉伸": 0.9,
            "垂直拉伸": 1.07,
            "旋转": 6.0,
            "斜切X": 0.0,
            "斜切Y": 0.0,
            "扭曲": [1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0],
        }
        with self._page_with_sixteen_variants(saved) as (page, variant_ids):
            self._show_at(page, 1600, 900)
            variant_id = variant_ids[0]
            page._select_variant(variant_id)
            page._enter_detail(variant_id)
            self.app.processEvents()

            transform = page._detail_canvas.transform()
            self.assertAlmostEqual(transform["x"], 13.0)
            self.assertAlmostEqual(transform["y"], -7.0)
            self.assertAlmostEqual(transform["scale"], 1.15)
            self.assertAlmostEqual(transform["stretch_w"], 0.9)
            self.assertAlmostEqual(transform["stretch_h"], 1.07)
            self.assertAlmostEqual(transform["rotation"], 6.0)
            for actual, expected in zip(transform["distort"], saved["扭曲"]):
                self.assertAlmostEqual(actual, expected)
            self.assertFalse(page._is_dirty(variant_id))
            self.assertFalse(page._detail_canvas.is_dirty)
            self.assertFalse(page._detail_canvas.can_undo)
            self.assertFalse(page._detail_canvas.can_redo)
            self.assertFalse(page._undo_button.isEnabled())
            self.assertFalse(page._redo_button.isEnabled())

            page._detail_canvas.undo()
            self.assertEqual(page._detail_canvas.transform(), transform)
            self.assertFalse(page._detail_canvas.is_dirty)

    def test_repeated_preview_updates_keep_only_latest_cache_per_glyph(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            self._show_at(page, 1600, 900)
            variant_id = page._selected_id

            for offset in range(1, 8):
                page._adjustments[variant_id]["移动X"] = float(offset)
                page._update_card(variant_id)

            keys = [key for key in page._preview_cache if key[0] == variant_id]
            self.assertEqual(len(keys), 1)
            self.assertEqual(keys[0][1], page._signature(variant_id))

    def test_save_all_changes_applies_global_ink_change_to_all_results(self) -> None:
        with self._page_with_sixteen_variants() as (page, variant_ids):
            self._show_at(page, 1600, 900)
            result_ids = [
                str(detail["变体ID"])
                for detail in page._page_variants()
            ]
            self.assertEqual(set(result_ids), set(variant_ids))
            page._ink_check.setChecked(False)
            self.app.processEvents()

            real_save = page._adjustment_service.save_coordinated_variants
            with patch.object(
                page._adjustment_service,
                "save_coordinated_variants",
                wraps=real_save,
            ) as save_spy:
                self.assertTrue(page._save_current_page(show_success=False))

            submitted_ids = [
                str(detail["变体ID"])
                for detail in save_spy.call_args.args[0]
            ]
            self.assertEqual(set(submitted_ids), set(result_ids))

            for variant_id in result_ids:
                detail = page._variant_by_id[variant_id]
                self.assertEqual(detail["状态"], config.STATUS_FINISHED)
                record = detail["整体协调参数"]["墨色协调"]
                self.assertFalse(record["启用"])
                self.assertEqual(record["跳过原因"], "已关闭墨色统一")
                self.assertFalse(page._is_dirty(variant_id))
            self.assertFalse(page._initial_ink_enabled)
            self.assertEqual(page._page_index, 0)

    def test_save_all_changes_keeps_selection_or_when_save_fails(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            self._show_at(page, 1600, 900)
            initial_id = page._selected_id

            with patch.object(page, "_save_variants", return_value=False):
                self.assertFalse(page._save_current_page(show_success=False))
            self.assertEqual(page._page_index, 0)
            self.assertEqual(page._selected_id, initial_id)

            dirty_id = str(page._variants[-1]["变体ID"])
            page._adjustments[dirty_id]["移动X"] = 4.0
            page._sync_dirty_variant(dirty_id)
            real_save = page._adjustment_service.save_coordinated_variants
            with patch.object(
                page._adjustment_service,
                "save_coordinated_variants",
                wraps=real_save,
            ) as save_spy:
                self.assertTrue(page._save_current_page(show_success=False))
            self.assertEqual(page._page_index, 0)
            self.assertEqual(
                [str(item["变体ID"]) for item in save_spy.call_args.args[0]],
                [dirty_id],
            )
            self.assertEqual(page._selected_id, initial_id)
            self.assertEqual(
                page._coordination_status(page._variant_by_id[dirty_id]),
                "已协调",
            )

    def test_legacy_page_navigation_keeps_unpaginated_results(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            ordered_ids = [str(item["变体ID"]) for item in page._variants]
            page._change_page(3)
            self.assertEqual(page._page_index, 0)
            self.assertEqual(
                [str(item["变体ID"]) for item in page._page_variants()],
                ordered_ids,
            )

    def test_discard_restores_persisted_baselines_and_refreshes_page_state(self) -> None:
        with self._page_with_sixteen_variants() as (page, _variant_ids):
            self._show_at(page, 1600, 900)
            variant_id = page._selected_id
            page._enter_detail(variant_id)
            self.assertTrue(page._detail_canvas.set_transform(x=11.0, rotation=4.0))
            page._ink_check.setChecked(False)
            self.app.processEvents()
            self.assertTrue(page._is_dirty(variant_id))

            # 模拟内存保存基线缺失，放弃操作仍须以字库记录为准恢复。
            page._saved_adjustments.pop(variant_id)
            page._saved_signatures.pop(variant_id)
            stale_cache_key = ("过期预览",)
            page._preview_cache[stale_cache_key] = page._pil_to_qimage(
                Image.new("RGBA", (1, 1), (0, 0, 0, 0))
            )

            with patch.object(
                QMessageBox,
                "exec",
                autospec=True,
                return_value=QMessageBox.StandardButton.Discard.value,
            ):
                self.assertTrue(page._confirm_leave_changes())
            self.app.processEvents()

            persisted = page._adjustment_service.load_saved_coordination_adjustments(
                page._variant_by_id[variant_id]
            )
            self.assertEqual(page._adjustments[variant_id], persisted)
            self.assertEqual(page._saved_adjustments[variant_id], persisted)
            self.assertFalse(any(page._is_dirty(item) for item in page._variant_by_id))
            self.assertTrue(page._ink_check.isChecked())
            self.assertNotIn(stale_cache_key, page._preview_cache)
            self.assertAlmostEqual(page._detail_canvas.transform()["x"], 0.0)
            self.assertAlmostEqual(page._detail_canvas.transform()["rotation"], 0.0)
            self.assertFalse(page._detail_canvas.is_dirty)
            self.assertIn("未保存 0", page._summary_label.text())


if __name__ == "__main__":
    unittest.main()
