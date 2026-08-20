"""手工审核画布的自由变换与状态语义回归测试。"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QImage,
    QInputDevice,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPointingDevice,
    QTabletEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import QApplication

from ui.widgets.review_canvas import ReviewCanvas


class ReviewCanvasTests(unittest.TestCase):
    """验证最终输出、保存基线和纯视图操作不会互相污染。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.canvas = ReviewCanvas()
        self.canvas.resize(600, 520)
        self.canvas.set_image(self._sample_image())

    def tearDown(self) -> None:
        self.canvas.close()
        self.canvas.deleteLater()

    def test_transform_api_bakes_translation_into_final_image(self) -> None:
        self.canvas.set_transform(x=2, y=-1, scale=1.0, rotation=0.0)

        result = self.canvas.image()

        self.assertEqual(result.size(), self._sample_image().size())
        self.assertEqual(result.pixelColor(6, 3), QColor(24, 68, 112, 255))
        self.assertEqual(result.pixelColor(4, 4).alpha(), 0)
        self.assertEqual(
            self.canvas.transform(),
            {
                "x": 2.0,
                "y": -1.0,
                "scale": 1.0,
                "rotation": 0.0,
                "stretch_w": 1.0,
                "stretch_h": 1.0,
                "distort": [0.0] * 8,
            },
        )
        self.assertTrue(self.canvas.is_dirty)

    def test_scale_and_rotation_are_baked_without_changing_canvas_size(self) -> None:
        self.canvas.set_transform(scale=1.5, rotation=90)

        result = self.canvas.image()

        self.assertEqual(result.size(), self._sample_image().size())
        self.assertGreater(self._opaque_pixel_count(result), 0)
        self.assertNotEqual(result, self._sample_image())

    def test_render_postprocessor_runs_after_geometry_without_changing_edit_state(self) -> None:
        processed_sizes: list[tuple[int, int]] = []

        def normalize_ink(pixels):  # type: ignore[no-untyped-def]
            processed_sizes.append((pixels.shape[1], pixels.shape[0]))
            result = pixels.copy()
            foreground = result[:, :, 3] > 0
            result[:, :, :3] = 0
            result[:, :, 3] = 0
            result[:, :, 3][foreground] = 123
            return result

        self.canvas.set_render_postprocessor(normalize_ink)
        self.assertTrue(self.canvas.set_transform(scale=2.0))
        transform_before = self.canvas.transform()
        undo_count = len(self.canvas._undo_stack)

        result = self.canvas.image()

        self.assertTrue(processed_sizes)
        self.assertGreaterEqual(processed_sizes[-1][0], 6)
        self.assertGreaterEqual(processed_sizes[-1][1], 6)
        foreground = [
            result.pixelColor(x, y)
            for y in range(result.height())
            for x in range(result.width())
            if result.pixelColor(x, y).alpha() > 0
        ]
        self.assertTrue(foreground)
        self.assertTrue(all(color.alpha() == 123 for color in foreground))
        self.assertTrue(all(color.red() == 0 for color in foreground))
        self.assertEqual(self.canvas.transform(), transform_before)
        self.assertEqual(len(self.canvas._undo_stack), undo_count)

        self.canvas.actual_size()
        postprocessed_view = self._render_canvas()
        reference = QImage(4, 4, QImage.Format.Format_ARGB32)
        reference.fill(QColor(220, 40, 60, 255))
        self.canvas.set_reference_image(reference, opacity=0.5)
        self.canvas.set_reference_visible(True)
        self.assertNotEqual(self._render_canvas(), postprocessed_view)
        self.canvas.set_reference_visible(False)

        self.canvas.set_render_postprocessor(None)
        restored = self.canvas.image()
        restored_foreground = [
            restored.pixelColor(x, y)
            for y in range(restored.height())
            for x in range(restored.width())
            if restored.pixelColor(x, y).alpha() > 0
        ]
        self.assertTrue(any(color.alpha() != 123 for color in restored_foreground))

    def test_out_of_bounds_transform_expands_symmetrically_without_clipping(self) -> None:
        image = QImage(12, 12, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        image.setPixelColor(0, 5, QColor(20, 40, 60, 255))
        self.canvas.set_image(image)

        self.canvas.set_transform(x=-4)
        result = self.canvas.image()

        self.assertEqual(result.size().width(), 20)
        self.assertEqual(result.size().height(), 12)
        self.assertEqual(self.canvas.output_origin(), self._point(4, 0))
        self.assertEqual(result.pixelColor(0, 5), QColor(20, 40, 60, 255))
        self.assertEqual(self._opaque_pixel_count(result), 1)

    def test_expanded_saved_image_restores_original_logical_canvas(self) -> None:
        self.canvas.set_transform(x=-8, scale=1.4, rotation=20)
        result = self.canvas.image()
        origin = self.canvas.output_origin()
        self.assertGreater(result.width(), 12)

        reopened = ReviewCanvas()
        reopened.set_image(result, (12, 12))

        self.assertEqual(reopened.canvas_size().width(), 12)
        self.assertEqual(reopened.canvas_size().height(), 12)
        self.assertEqual(reopened.image(), result)
        self.assertEqual(reopened.output_origin(), origin)
        reopened.deleteLater()

    def test_undo_redo_reset_and_mark_saved_share_one_state_history(self) -> None:
        self.canvas.set_transform(x=2)
        self.assertTrue(self.canvas.is_dirty)

        self.canvas.undo()
        self.assertEqual(self.canvas.transform()["x"], 0.0)
        self.assertFalse(self.canvas.is_dirty)

        self.canvas.redo()
        self.assertEqual(self.canvas.transform()["x"], 2.0)
        self.assertTrue(self.canvas.is_dirty)

        self.canvas.mark_saved()
        saved_result = self.canvas.image()
        self.assertFalse(self.canvas.is_dirty)
        self.assertEqual(
            self.canvas.transform(),
            self._default_transform(),
        )

        self.canvas.set_transform(x=4)
        self.canvas.reset_image()
        self.assertEqual(self.canvas.transform()["x"], 0.0)
        self.assertEqual(self.canvas.image(), saved_result)
        self.assertFalse(self.canvas.is_dirty)

        self.canvas.undo()
        self.assertEqual(self.canvas.transform()["x"], 4.0)
        self.assertTrue(self.canvas.is_dirty)

    def test_switching_to_brush_bakes_transform_once(self) -> None:
        transformed = None
        self.canvas.set_transform(x=2, rotation=15)
        transformed = self.canvas.image()

        self.canvas.set_tool(ReviewCanvas.TOOL_BRUSH)

        self.assertEqual(self.canvas.image(), transformed)
        self.assertEqual(
            self.canvas.transform(),
            self._default_transform(),
        )
        self.canvas.undo()
        self.assertEqual(self.canvas.transform()["x"], 2.0)
        self.assertEqual(self.canvas.transform()["rotation"], 15.0)

    def test_view_options_do_not_mark_image_dirty(self) -> None:
        self.canvas.set_grid_visible(False)
        self.canvas.set_background_mode(ReviewCanvas.BACKGROUND_CHECKERBOARD)
        self.canvas.actual_size()
        self.canvas.zoom_in()
        self.canvas.fit_to_view()

        self.assertFalse(self.canvas.grid_visible)
        self.assertEqual(
            self.canvas.background_mode,
            ReviewCanvas.BACKGROUND_CHECKERBOARD,
        )
        self.assertFalse(self.canvas.is_dirty)

    def test_source_preview_temporarily_renders_loaded_image_without_editing(self) -> None:
        self.canvas.actual_size()
        loaded_view = self._render_canvas()
        self.assertTrue(self.canvas.set_transform(x=2, y=-1))
        edited_view = self._render_canvas()
        dirty = self.canvas.is_dirty
        undo_count = len(self.canvas._undo_stack)
        redo_count = len(self.canvas._redo_stack)

        self.canvas.set_source_preview_visible(True)

        self.assertTrue(self.canvas.source_preview_visible)
        self.assertEqual(self._render_canvas(), loaded_view)
        self.assertNotEqual(edited_view, loaded_view)
        self.assertEqual(self.canvas.transform()["x"], 2.0)
        self.assertEqual(self.canvas.transform()["y"], -1.0)
        self.assertEqual(self.canvas.is_dirty, dirty)
        self.assertEqual(len(self.canvas._undo_stack), undo_count)
        self.assertEqual(len(self.canvas._redo_stack), redo_count)

        self.canvas.set_source_preview_visible(False)

        self.assertFalse(self.canvas.source_preview_visible)
        self.assertEqual(self._render_canvas(), edited_view)
        self.assertEqual(self.canvas.is_dirty, dirty)

    def test_prepared_baseline_keeps_separate_pre_normalization_source_preview(self) -> None:
        original = self._sample_image()
        prepared = QImage(12, 12, QImage.Format.Format_ARGB32)
        prepared.fill(Qt.GlobalColor.transparent)
        for y in range(2, 10):
            for x in range(2, 10):
                prepared.setPixelColor(x, y, QColor(24, 68, 112, 255))

        self.canvas.set_image(
            prepared,
            (12, 12),
            source_preview=original,
        )
        self.canvas.actual_size()
        prepared_view = self._render_canvas()

        self.assertEqual(self.canvas.image(), prepared)
        self.assertEqual(self.canvas._source_image, original)
        self.assertFalse(self.canvas.is_dirty)
        self.canvas.set_source_preview_visible(True)
        original_view = self._render_canvas()
        self.assertNotEqual(original_view, prepared_view)
        self.assertFalse(self.canvas.is_dirty)
        self.canvas.set_source_preview_visible(False)
        self.assertEqual(self._render_canvas(), prepared_view)

    def test_reference_overlay_is_centered_and_remains_a_view_only_layer(self) -> None:
        self.canvas.actual_size()
        reference = QImage(4, 4, QImage.Format.Format_ARGB32)
        reference.fill(QColor(220, 40, 60, 255))
        baseline_view = self._render_canvas()
        original_image = self.canvas.image()
        original_transform = self.canvas.transform()
        history_events: list[tuple[bool, bool]] = []
        self.canvas.history_changed.connect(
            lambda can_undo, can_redo: history_events.append((can_undo, can_redo))
        )

        self.canvas.set_reference_image(reference, opacity=0.5)
        self.assertFalse(self.canvas.reference_visible)
        self.assertAlmostEqual(self.canvas.reference_opacity, 0.5)
        self.assertEqual(self._render_canvas(), baseline_view)

        self.canvas.set_reference_visible(True)
        overlaid_view = self._render_canvas()
        sample = self._widget_position(4, 4).toPoint()

        self.assertTrue(self.canvas.reference_visible)
        self.assertNotEqual(overlaid_view, baseline_view)
        self.assertGreater(overlaid_view.pixelColor(sample).red(), baseline_view.pixelColor(sample).red())
        self.assertEqual(self.canvas.image(), original_image)
        self.assertEqual(self.canvas.transform(), original_transform)
        self.assertFalse(self.canvas.is_dirty)
        self.assertFalse(self.canvas.can_undo)
        self.assertFalse(self.canvas.can_redo)
        self.assertEqual(history_events, [])

        self.canvas.set_reference_opacity(2.0)
        self.assertEqual(self.canvas.reference_opacity, 1.0)
        self.canvas.set_reference_image(None)
        self.assertFalse(self.canvas.reference_visible)
        self.assertEqual(self._render_canvas(), baseline_view)

    def test_non_baking_saved_baseline_preserves_transform_and_restores_to_it(self) -> None:
        self.assertTrue(
            self.canvas.set_transform(
                x=3,
                scale=1.25,
                rotation=12,
                stretch_w=1.1,
            )
        )
        saved_transform = self.canvas.transform()
        saved_result = self.canvas.image()

        self.assertTrue(self.canvas.set_saved_baseline())

        self.assertEqual(self.canvas.transform(), saved_transform)
        self.assertEqual(self.canvas.image(), saved_result)
        self.assertFalse(self.canvas.is_dirty)
        self.assertFalse(self.canvas.can_undo)
        self.assertFalse(self.canvas.can_redo)

        self.assertTrue(self.canvas.set_transform(y=-2, rotation=20))
        self.assertTrue(self.canvas.is_dirty)
        self.canvas.discard_changes()

        self.assertEqual(self.canvas.transform(), saved_transform)
        self.assertEqual(self.canvas.image(), saved_result)
        self.assertFalse(self.canvas.is_dirty)

    def test_transform_feedback_and_history_state_signal(self) -> None:
        history_events: list[tuple[bool, bool]] = []
        self.canvas.history_changed.connect(
            lambda can_undo, can_redo: history_events.append((can_undo, can_redo))
        )

        self.assertTrue(self.canvas.set_transform(x=2))
        self.assertTrue(self.canvas.can_undo)
        self.assertFalse(self.canvas.can_redo)
        self.assertEqual(history_events[-1], (True, False))

        event_count = len(history_events)
        self.assertFalse(self.canvas.set_transform(x=2))
        self.assertEqual(len(history_events), event_count)

        self.canvas.undo()
        self.assertFalse(self.canvas.can_undo)
        self.assertTrue(self.canvas.can_redo)
        self.assertEqual(history_events[-1], (False, True))

        self.canvas.redo()
        self.assertTrue(self.canvas.can_undo)
        self.assertFalse(self.canvas.can_redo)
        self.assertEqual(history_events[-1], (True, False))

    def test_workspace_boundary_is_centered_and_130_percent_for_rectangular_canvas(self) -> None:
        image = QImage(200, 100, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        self.canvas.set_image(image, (200, 100))
        self.canvas.actual_size()

        grid = self.canvas._canvas_rect()
        workspace = self.canvas._workspace_rect()

        self.assertAlmostEqual(workspace.width(), grid.width() * 1.30)
        self.assertAlmostEqual(workspace.height(), grid.height() * 1.30)
        self.assertAlmostEqual(workspace.center().x(), grid.center().x())
        self.assertAlmostEqual(workspace.center().y(), grid.center().y())
        self.assertAlmostEqual(grid.left() - workspace.left(), grid.width() * 0.15)
        self.assertAlmostEqual(grid.top() - workspace.top(), grid.height() * 0.15)

    def test_fit_to_view_contains_workspace_and_transformed_output(self) -> None:
        image = QImage(200, 100, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        image.setPixelColor(0, 50, QColor(20, 40, 60, 255))
        self.canvas.resize(600, 520)
        self.canvas.set_image(image, (200, 100))
        self.canvas.set_transform(x=-120)

        self.canvas.fit_to_view()

        bounds = self.canvas._logical_view_bounds()
        expected_zoom = min((600 - 64) / bounds.width(), (520 - 64) / bounds.height())
        self.assertAlmostEqual(self.canvas._zoom, expected_zoom)
        self.assertLessEqual(self.canvas._workspace_rect().left(), self.canvas._canvas_rect().left())
        self.assertGreaterEqual(self.canvas._workspace_rect().right(), self.canvas._canvas_rect().right())

        output_size, origin = self.canvas._output_geometry()
        grid = self.canvas._canvas_rect()
        output_rect = QRectF(
            grid.left() - origin.x() * self.canvas._zoom,
            grid.top() - origin.y() * self.canvas._zoom,
            output_size.width() * self.canvas._zoom,
            output_size.height() * self.canvas._zoom,
        )
        visible_bounds = self.canvas._workspace_rect().united(output_rect)
        self.assertGreaterEqual(visible_bounds.left(), 32.0 - 1e-6)
        self.assertLessEqual(visible_bounds.right(), 600.0 - 32.0 + 1e-6)
        self.assertGreaterEqual(visible_bounds.top(), 32.0 - 1e-6)
        self.assertLessEqual(visible_bounds.bottom(), 520.0 - 32.0 + 1e-6)

    def test_workspace_boundary_is_not_drawn(self) -> None:
        image = QImage(100, 80, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        self.canvas.set_image(image, (100, 80))
        self.canvas.actual_size()

        workspace = self.canvas._workspace_rect()
        samples = (
            QPoint(round(workspace.left()), round(workspace.center().y())),
            QPoint(round(workspace.right()) - 1, round(workspace.center().y())),
            QPoint(round(workspace.center().x()), round(workspace.top())),
            QPoint(round(workspace.center().x()), round(workspace.bottom()) - 1),
        )
        visible = self._render_canvas()
        self.canvas.set_grid_visible(False)
        hidden = self._render_canvas()

        for sample in samples:
            with self.subTest(sample=sample):
                self.assertEqual(visible.pixelColor(sample), hidden.pixelColor(sample))
                self.assertEqual(visible.pixelColor(sample), QColor("#FFFFFF"))

    def test_grid_uses_light_red_solid_border_and_dashed_inner_lines(self) -> None:
        image = QImage(120, 100, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        self.canvas.set_image(image, (120, 100))
        self.canvas.actual_size()

        rendered = self._render_canvas()
        target = self.canvas._canvas_rect()
        left = round(target.left())
        top = round(target.top())
        right = round(target.right())
        bottom = round(target.bottom())
        center_x = round(target.center().x())

        border_color = rendered.pixelColor(left, top + 11)
        self.assertGreater(border_color.red(), border_color.green())
        self.assertGreater(border_color.red(), border_color.blue())
        self.assertGreaterEqual(border_color.green(), 120)
        self.assertGreaterEqual(border_color.blue(), 120)

        border_pixels = [
            rendered.pixelColor(left, y)
            for y in range(top + 4, bottom - 3)
        ]
        self.assertTrue(all(color == border_color for color in border_pixels))

        internal_pixels = [
            rendered.pixelColor(center_x, y)
            for y in range(top + 7, top + 35)
        ]
        self.assertIn(border_color, internal_pixels)
        self.assertIn(QColor("#FFFFFF"), internal_pixels)

    def test_space_drag_temporarily_pans_in_every_tool_without_editing(self) -> None:
        position = QPointF(180.0, 180.0)
        delta = QPointF(34.0, -21.0)
        for tool in (
            ReviewCanvas.TOOL_PAN,
            ReviewCanvas.TOOL_TRANSFORM,
            ReviewCanvas.TOOL_BRUSH,
            ReviewCanvas.TOOL_ERASER,
        ):
            with self.subTest(tool=tool):
                self.canvas.set_image(self._sample_image())
                self.canvas.actual_size()
                self.canvas.set_tool(tool)
                self.canvas._update_cursor_at(position)
                original_cursor = self.canvas.cursor().shape()
                original_image = self.canvas.image()
                original_transform = self.canvas.transform()
                original_offset = QPointF(self.canvas._offset)
                original_dirty = self.canvas.is_dirty
                undo_count = len(self.canvas._undo_stack)
                redo_count = len(self.canvas._redo_stack)

                space_press = QKeyEvent(
                    QEvent.Type.KeyPress,
                    Qt.Key.Key_Space,
                    Qt.KeyboardModifier.NoModifier,
                    " ",
                )
                QApplication.sendEvent(self.canvas, space_press)
                self.assertTrue(space_press.isAccepted())
                self.assertEqual(self.canvas.cursor().shape(), Qt.CursorShape.OpenHandCursor)

                press = QMouseEvent(
                    QEvent.Type.MouseButtonPress,
                    position,
                    position,
                    position,
                    Qt.MouseButton.LeftButton,
                    Qt.MouseButton.LeftButton,
                    Qt.KeyboardModifier.NoModifier,
                )
                move_position = position + delta
                move = QMouseEvent(
                    QEvent.Type.MouseMove,
                    move_position,
                    move_position,
                    move_position,
                    Qt.MouseButton.NoButton,
                    Qt.MouseButton.LeftButton,
                    Qt.KeyboardModifier.NoModifier,
                )
                release = QMouseEvent(
                    QEvent.Type.MouseButtonRelease,
                    move_position,
                    move_position,
                    move_position,
                    Qt.MouseButton.LeftButton,
                    Qt.MouseButton.NoButton,
                    Qt.KeyboardModifier.NoModifier,
                )
                QApplication.sendEvent(self.canvas, press)
                self.assertEqual(self.canvas.cursor().shape(), Qt.CursorShape.ClosedHandCursor)
                QApplication.sendEvent(self.canvas, move)
                QApplication.sendEvent(self.canvas, release)
                self.assertEqual(self.canvas.cursor().shape(), Qt.CursorShape.OpenHandCursor)

                space_release = QKeyEvent(
                    QEvent.Type.KeyRelease,
                    Qt.Key.Key_Space,
                    Qt.KeyboardModifier.NoModifier,
                    " ",
                )
                QApplication.sendEvent(self.canvas, space_release)
                self.assertTrue(space_release.isAccepted())
                self.assertEqual(self.canvas.tool, tool)
                self.assertEqual(self.canvas.cursor().shape(), original_cursor)
                self._assert_point_close(self.canvas._offset, original_offset + delta, 0.01)
                self.assertEqual(self.canvas.image(), original_image)
                self.assertEqual(self.canvas.transform(), original_transform)
                self.assertEqual(self.canvas.is_dirty, original_dirty)
                self.assertEqual(len(self.canvas._undo_stack), undo_count)
                self.assertEqual(len(self.canvas._redo_stack), redo_count)

    def test_space_pan_stays_locked_until_mouse_release_when_space_is_released_first(self) -> None:
        self.canvas.actual_size()
        self.canvas.set_tool(ReviewCanvas.TOOL_BRUSH)
        original_image = self.canvas.image()
        original_offset = QPointF(self.canvas._offset)
        start = QPointF(180.0, 180.0)
        first = QPointF(205.0, 190.0)
        second = QPointF(220.0, 205.0)

        QApplication.sendEvent(
            self.canvas,
            QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_Space,
                Qt.KeyboardModifier.NoModifier,
                " ",
            ),
        )
        QApplication.sendEvent(
            self.canvas,
            QMouseEvent(
                QEvent.Type.MouseButtonPress,
                start,
                start,
                start,
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        QApplication.sendEvent(
            self.canvas,
            QMouseEvent(
                QEvent.Type.MouseMove,
                first,
                first,
                first,
                Qt.MouseButton.NoButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        QApplication.sendEvent(
            self.canvas,
            QKeyEvent(
                QEvent.Type.KeyRelease,
                Qt.Key.Key_Space,
                Qt.KeyboardModifier.NoModifier,
                " ",
            ),
        )

        self.assertTrue(self.canvas._panning)
        self.assertEqual(self.canvas.cursor().shape(), Qt.CursorShape.ClosedHandCursor)

        QApplication.sendEvent(
            self.canvas,
            QMouseEvent(
                QEvent.Type.MouseMove,
                second,
                second,
                second,
                Qt.MouseButton.NoButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        QApplication.sendEvent(
            self.canvas,
            QMouseEvent(
                QEvent.Type.MouseButtonRelease,
                second,
                second,
                second,
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
            ),
        )

        self._assert_point_close(self.canvas._offset, original_offset + second - start, 0.01)
        self.assertEqual(self.canvas.image(), original_image)
        self.assertFalse(self.canvas.is_dirty)
        self.assertFalse(self.canvas.space_pan_active)

    def test_space_auto_repeat_and_wheel_keep_pan_state_and_brush_size(self) -> None:
        self.canvas.actual_size()
        self.canvas.set_tool(ReviewCanvas.TOOL_BRUSH)
        self.canvas.set_brush_size(20)
        original_zoom = self.canvas._zoom

        QApplication.sendEvent(
            self.canvas,
            QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_Space,
                Qt.KeyboardModifier.NoModifier,
                " ",
            ),
        )
        QApplication.sendEvent(
            self.canvas,
            QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_Space,
                Qt.KeyboardModifier.NoModifier,
                " ",
                True,
                2,
            ),
        )
        QApplication.sendEvent(
            self.canvas,
            self._wheel_event(120, Qt.KeyboardModifier.NoModifier),
        )

        self.assertEqual(self.canvas.brush_size, 20)
        self.assertGreater(self.canvas._zoom, original_zoom)
        self.assertTrue(self.canvas.space_pan_active)

        QApplication.sendEvent(
            self.canvas,
            QKeyEvent(
                QEvent.Type.KeyRelease,
                Qt.Key.Key_Space,
                Qt.KeyboardModifier.NoModifier,
                " ",
                True,
                2,
            ),
        )
        self.assertTrue(self.canvas.space_pan_active)
        QApplication.sendEvent(
            self.canvas,
            QKeyEvent(
                QEvent.Type.KeyRelease,
                Qt.Key.Key_Space,
                Qt.KeyboardModifier.NoModifier,
                " ",
            ),
        )
        self.assertFalse(self.canvas.space_pan_active)

    def test_space_pressed_during_brush_stroke_does_not_cancel_or_start_pan(self) -> None:
        image = QImage(80, 80, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        self.canvas.set_image(image)
        self.canvas.actual_size()
        self.canvas.set_tool(ReviewCanvas.TOOL_BRUSH)
        self.canvas.set_brush_size(8)
        start = self.canvas._canvas_rect().center()
        end = start + QPointF(16.0, 0.0)
        original_offset = QPointF(self.canvas._offset)

        QApplication.sendEvent(
            self.canvas,
            QMouseEvent(
                QEvent.Type.MouseButtonPress,
                start,
                start,
                start,
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        space = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_Space,
            Qt.KeyboardModifier.NoModifier,
            " ",
        )
        QApplication.sendEvent(self.canvas, space)

        self.assertTrue(space.isAccepted())
        self.assertTrue(self.canvas._drawing)
        self.assertTrue(self.canvas._space_pan_blocked)
        self.assertFalse(self.canvas._panning)

        QApplication.sendEvent(
            self.canvas,
            QMouseEvent(
                QEvent.Type.MouseMove,
                end,
                end,
                end,
                Qt.MouseButton.NoButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        QApplication.sendEvent(
            self.canvas,
            QMouseEvent(
                QEvent.Type.MouseButtonRelease,
                end,
                end,
                end,
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        QApplication.sendEvent(
            self.canvas,
            QKeyEvent(
                QEvent.Type.KeyRelease,
                Qt.Key.Key_Space,
                Qt.KeyboardModifier.NoModifier,
                " ",
            ),
        )

        self.assertGreater(self._opaque_pixel_count(self.canvas.image()), 0)
        self.assertEqual(len(self.canvas._undo_stack), 1)
        self._assert_point_close(self.canvas._offset, original_offset, 0.01)
        self.assertFalse(self.canvas.space_pan_active)

    def test_space_pan_state_is_cleared_when_input_is_interrupted(self) -> None:
        outside = QPointF(12.0, 12.0)
        for interruption in (QEvent.Type.FocusOut, QEvent.Type.WindowDeactivate):
            with self.subTest(interruption=interruption):
                self.canvas.set_image(self._sample_image())
                self.canvas.actual_size()
                self.canvas.set_tool(ReviewCanvas.TOOL_BRUSH)
                self.canvas._update_cursor_at(outside)
                original_offset = QPointF(self.canvas._offset)

                QApplication.sendEvent(
                    self.canvas,
                    QKeyEvent(
                        QEvent.Type.KeyPress,
                        Qt.Key.Key_Space,
                        Qt.KeyboardModifier.NoModifier,
                        " ",
                    ),
                )
                self.assertEqual(self.canvas.cursor().shape(), Qt.CursorShape.OpenHandCursor)
                QApplication.sendEvent(self.canvas, QEvent(interruption))
                self.assertEqual(self.canvas.cursor().shape(), Qt.CursorShape.ArrowCursor)

                press = QMouseEvent(
                    QEvent.Type.MouseButtonPress,
                    outside,
                    outside,
                    outside,
                    Qt.MouseButton.LeftButton,
                    Qt.MouseButton.LeftButton,
                    Qt.KeyboardModifier.NoModifier,
                )
                moved = outside + QPointF(25.0, 20.0)
                move = QMouseEvent(
                    QEvent.Type.MouseMove,
                    moved,
                    moved,
                    moved,
                    Qt.MouseButton.NoButton,
                    Qt.MouseButton.LeftButton,
                    Qt.KeyboardModifier.NoModifier,
                )
                release = QMouseEvent(
                    QEvent.Type.MouseButtonRelease,
                    moved,
                    moved,
                    moved,
                    Qt.MouseButton.LeftButton,
                    Qt.MouseButton.NoButton,
                    Qt.KeyboardModifier.NoModifier,
                )
                QApplication.sendEvent(self.canvas, press)
                QApplication.sendEvent(self.canvas, move)
                QApplication.sendEvent(self.canvas, release)
                self._assert_point_close(self.canvas._offset, original_offset, 0.01)

    def test_brush_can_expand_into_workspace_ring_and_undo_restores_image(self) -> None:
        image = QImage(100, 80, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        image.setPixelColor(50, 40, QColor("#222222"))
        self.canvas.set_image(image, (100, 80))
        self.canvas.actual_size()
        self.canvas.set_tool(ReviewCanvas.TOOL_BRUSH)
        self.canvas.set_brush_size(6)
        workspace = self.canvas._workspace_rect()
        grid = self.canvas._canvas_rect()
        position = QPointF(grid.left() - 8.0, grid.center().y())
        self.assertTrue(workspace.contains(position))
        self.assertFalse(grid.contains(position))

        press = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            position,
            position,
            position,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        release = QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            position,
            position,
            position,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(self.canvas, press)
        QApplication.sendEvent(self.canvas, release)

        result = self.canvas.image()
        self.assertGreater(result.width(), 100)
        self.assertGreater(self._opaque_pixel_count(result), 1)
        self.assertGreater(self.canvas.output_origin().x(), 0)

        self.canvas.undo()

        self.assertEqual(self.canvas.image(), image)

    def test_active_outside_brush_updates_output_before_release(self) -> None:
        image = QImage(101, 81, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        image.setPixelColor(50, 40, QColor("#222222"))
        self.canvas.set_image(image, (101, 81))
        self.canvas.actual_size()
        self.canvas.set_tool(ReviewCanvas.TOOL_BRUSH)
        self.canvas.set_brush_size(5)
        position = self._widget_position(-8, 40)

        press = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            position,
            position,
            position,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(self.canvas, press)

        active_image = self.canvas.image()
        active_origin = self.canvas.output_origin()
        self.assertTrue(self.canvas._drawing)
        self.assertGreater(active_image.width(), 101)
        self.assertGreater(active_origin.x(), 0)
        self.assertGreater(self._opaque_pixel_count(active_image), 1)
        self.assertEqual(len(self.canvas._undo_stack), 1)

        release = QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            position,
            position,
            position,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(self.canvas, release)

        self.assertEqual(self.canvas.image(), active_image)
        self.assertEqual(self.canvas.output_origin(), active_origin)
        self.canvas.undo()
        self.assertEqual(self.canvas.image(), image)
        self.assertEqual(self.canvas.output_origin(), QPoint())

    def test_one_stroke_can_expand_multiple_directions_and_redo_exactly(self) -> None:
        image = QImage(100, 80, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        image.setPixelColor(50, 40, QColor("#222222"))
        self.canvas.set_image(image, (100, 80))
        self.canvas.actual_size()
        self.canvas.set_tool(ReviewCanvas.TOOL_BRUSH)
        self.canvas.set_brush_size(4)
        points = ((-8, 40), (-13, 40), (50, -10), (112, 40), (50, 90))

        for index, (x, y) in enumerate(points):
            position = self._widget_position(x, y)
            event = QMouseEvent(
                QEvent.Type.MouseButtonPress if index == 0 else QEvent.Type.MouseMove,
                position,
                position,
                position,
                Qt.MouseButton.LeftButton if index == 0 else Qt.MouseButton.NoButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
            QApplication.sendEvent(self.canvas, event)
        last_position = self._widget_position(*points[-1])
        release = QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            last_position,
            last_position,
            last_position,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(self.canvas, release)

        result = self.canvas.image()
        origin = self.canvas.output_origin()
        self.assertGreater(result.width(), 100)
        self.assertGreater(result.height(), 80)
        self.assertGreater(origin.x(), 0)
        self.assertGreater(origin.y(), 0)
        for x, y in points:
            self.assertGreater(result.pixelColor(origin.x() + x, origin.y() + y).alpha(), 0)
        self.assertEqual(len(self.canvas._undo_stack), 1)

        self.canvas.undo()
        self.assertEqual(self.canvas.image(), image)
        self.canvas.redo()
        self.assertEqual(self.canvas.image(), result)
        self.assertEqual(self.canvas.output_origin(), origin)

    def test_eraser_on_transparent_regions_creates_no_dirty_state_or_undo(self) -> None:
        image = QImage(100, 80, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        image.setPixelColor(50, 40, QColor("#222222"))
        self.canvas.set_image(image, (100, 80))
        self.canvas.actual_size()
        self.canvas.set_tool(ReviewCanvas.TOOL_ERASER)
        self.canvas.set_brush_size(10)

        for x, y in ((10, 10), (-8, 10)):
            position = self._widget_position(x, y)
            press = QMouseEvent(
                QEvent.Type.MouseButtonPress,
                position,
                position,
                position,
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
            release = QMouseEvent(
                QEvent.Type.MouseButtonRelease,
                position,
                position,
                position,
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
            )
            QApplication.sendEvent(self.canvas, press)
            QApplication.sendEvent(self.canvas, release)

        self.assertEqual(self.canvas.image(), image)
        self.assertFalse(self.canvas.is_dirty)
        self.assertFalse(self.canvas._undo_stack)

        ink_position = self._widget_position(50, 40)
        press = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            ink_position,
            ink_position,
            ink_position,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        release = QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            ink_position,
            ink_position,
            ink_position,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(self.canvas, press)
        QApplication.sendEvent(self.canvas, release)

        self.assertTrue(self.canvas.is_dirty)
        self.assertEqual(len(self.canvas._undo_stack), 1)
        self.canvas.undo()
        self.assertEqual(self.canvas.image(), image)

    def test_brush_outside_workspace_does_not_modify_image(self) -> None:
        image = QImage(100, 80, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        image.setPixelColor(50, 40, QColor("#222222"))
        self.canvas.set_image(image, (100, 80))
        self.canvas.actual_size()
        self.canvas.set_tool(ReviewCanvas.TOOL_BRUSH)
        workspace = self.canvas._workspace_rect()
        position = QPointF(workspace.left() - 10.0, workspace.center().y())

        press = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            position,
            position,
            position,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(self.canvas, press)

        self.assertEqual(self.canvas.image(), image)
        self.assertFalse(self.canvas.is_dirty)
        self.assertFalse(self.canvas._undo_stack)
        self.assertEqual(self.canvas.cursor().shape(), Qt.CursorShape.ArrowCursor)

    def test_dominant_ink_color_is_sampled_for_brush(self) -> None:
        image = QImage(12, 12, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        for y in range(2, 9):
            for x in range(2, 9):
                image.setPixelColor(x, y, QColor(36, 74, 108, 190))
        image.setPixelColor(10, 10, QColor(220, 20, 40, 255))

        self.canvas.set_image(image)

        self.assertEqual(self.canvas.brush_color(), QColor(36, 74, 108, 255))
        self.assertEqual(self.canvas.sample_ink_color(), QColor(36, 74, 108, 255))

    def test_opaque_white_background_is_not_sampled_as_ink(self) -> None:
        image = QImage(30, 30, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.white)
        for y in range(12, 18):
            for x in range(12, 18):
                image.setPixelColor(x, y, QColor(92, 88, 84, 255))

        self.canvas.set_image(image)

        self.assertEqual(self.canvas.brush_color(), QColor(92, 88, 84, 255))

    def test_transform_signal_reports_exact_parameter_values(self) -> None:
        emitted: list[dict[str, object]] = []
        self.canvas.transform_changed.connect(emitted.append)

        distort = [2.0, -1.0, 3.0, 0.5, -2.0, 4.0, -1.5, -3.0]
        self.canvas.set_transform(
            x=-3.5,
            y=4.25,
            scale=0.8,
            rotation=-12.5,
            stretch_w=1.2,
            stretch_h=0.75,
            distort=distort,
        )

        self.assertEqual(
            emitted[-1],
            {
                "x": -3.5,
                "y": 4.25,
                "scale": 0.8,
                "rotation": -12.5,
                "stretch_w": 1.2,
                "stretch_h": 0.75,
                "distort": distort,
            },
        )
        distort[0] = 999.0
        emitted[-1]["distort"][1] = 999.0  # type: ignore[index]
        self.assertEqual(
            self.canvas.transform()["distort"],
            [2.0, -1.0, 3.0, 0.5, -2.0, 4.0, -1.5, -3.0],
        )

        self.assertTrue(
            self.canvas.set_transform(scale=50.0, stretch_w=50.0, stretch_h=50.0)
        )
        limited = self.canvas.transform()
        self.assertEqual(limited["scale"], 5.0)
        self.assertEqual(limited["stretch_w"], 5.0)
        self.assertEqual(limited["stretch_h"], 5.0)

    def test_transform_rejects_degenerate_or_non_finite_distortion(self) -> None:
        self._prepare_transform_canvas()
        baseline = self.canvas.transform()

        self.canvas.set_transform(
            distort=[0.0, 0.0, -120.0, 80.0, -120.0, -80.0, 0.0, 0.0]
        )

        self.assertEqual(self.canvas.transform(), baseline)
        self.assertFalse(self.canvas.is_dirty)
        self.assertFalse(self.canvas._undo_stack)

        self.canvas.set_transform(distort=[float("nan")] * 8)
        self.canvas.set_transform(distort=[0.0] * 7)

        self.assertEqual(self.canvas.transform(), baseline)
        self.assertFalse(self.canvas.is_dirty)
        self.assertFalse(self.canvas._undo_stack)

    def test_side_handles_stretch_and_compress_one_axis_with_opposite_edge_fixed(self) -> None:
        cases = (
            ("e", "w", QPointF(24.0, 0.0), "stretch_w", True),
            ("w", "e", QPointF(24.0, 0.0), "stretch_w", False),
            ("s", "n", QPointF(0.0, 18.0), "stretch_h", True),
            ("n", "s", QPointF(0.0, 18.0), "stretch_h", False),
        )
        for handle, anchor, delta, parameter, should_grow in cases:
            with self.subTest(handle=handle):
                self._prepare_transform_canvas()
                undo_before = len(self.canvas._undo_stack)
                before, after = self._drag_transform_handle(handle, delta)
                transform = self.canvas.transform()

                if should_grow:
                    self.assertGreater(transform[parameter], 1.0)
                else:
                    self.assertLess(transform[parameter], 1.0)
                other_parameter = "stretch_h" if parameter == "stretch_w" else "stretch_w"
                self.assertAlmostEqual(float(transform[other_parameter]), 1.0)
                self.assertAlmostEqual(float(transform["scale"]), 1.0)
                self._assert_point_close(before[anchor], after[anchor])
                self.assertEqual(len(self.canvas._undo_stack), undo_before + 1)

    def test_rotated_side_handle_uses_glyph_local_axis(self) -> None:
        self._prepare_transform_canvas()
        self.canvas.set_transform(rotation=35.0)
        handles, _rotate = self.canvas._control_handles()
        axis = handles["e"] - handles["w"]
        length = max(1.0, (axis.x() ** 2 + axis.y() ** 2) ** 0.5)
        delta = QPointF(axis.x() * 22.0 / length, axis.y() * 22.0 / length)

        before, after = self._drag_transform_handle("e", delta)

        transform = self.canvas.transform()
        self.assertGreater(transform["stretch_w"], 1.0)
        self.assertAlmostEqual(float(transform["stretch_h"]), 1.0)
        self.assertAlmostEqual(float(transform["scale"]), 1.0)
        self.assertAlmostEqual(float(transform["rotation"]), 35.0)
        self._assert_point_close(before["w"], after["w"])

    def test_corner_handles_scale_equally_and_keep_opposite_corner_fixed(self) -> None:
        opposite = {"nw": "se", "ne": "sw", "se": "nw", "sw": "ne"}
        for handle, anchor in opposite.items():
            with self.subTest(handle=handle):
                self._prepare_transform_canvas()
                handles, _rotate = self.canvas._control_handles()
                vector = handles[handle] - handles[anchor]
                length = max(1.0, (vector.x() ** 2 + vector.y() ** 2) ** 0.5)
                delta = QPointF(vector.x() * 18.0 / length, vector.y() * 18.0 / length)

                before, after = self._drag_transform_handle(handle, delta)

                transform = self.canvas.transform()
                self.assertGreater(transform["scale"], 1.0)
                self.assertAlmostEqual(float(transform["stretch_w"]), 1.0)
                self.assertAlmostEqual(float(transform["stretch_h"]), 1.0)
                self.assertEqual(transform["distort"], [0.0] * 8)
                self._assert_point_close(before[anchor], after[anchor])

    def test_shift_side_handle_scales_equally_and_alt_scales_from_center(self) -> None:
        self._prepare_transform_canvas()
        before, after = self._drag_transform_handle(
            "e",
            QPointF(20.0, 0.0),
            Qt.KeyboardModifier.ShiftModifier,
        )
        transform = self.canvas.transform()
        self.assertGreater(transform["scale"], 1.0)
        self.assertAlmostEqual(float(transform["stretch_w"]), 1.0)
        self.assertAlmostEqual(float(transform["stretch_h"]), 1.0)
        self._assert_point_close(before["w"], after["w"])

        self._prepare_transform_canvas()
        before_center = self._control_center()
        self._drag_transform_handle(
            "e",
            QPointF(20.0, 0.0),
            Qt.KeyboardModifier.AltModifier,
        )
        transform = self.canvas.transform()
        self.assertGreater(transform["stretch_w"], 1.0)
        self.assertAlmostEqual(float(transform["scale"]), 1.0)
        self._assert_point_close(before_center, self._control_center())

    def test_ctrl_corner_handles_distort_only_selected_corner_and_fix_opposite(self) -> None:
        corner_indices = {"nw": 0, "ne": 2, "se": 4, "sw": 6}
        opposite = {"nw": "se", "ne": "sw", "se": "nw", "sw": "ne"}
        delta = QPointF(7.0, 5.0)
        for handle, index in corner_indices.items():
            with self.subTest(handle=handle):
                self._prepare_transform_canvas()
                undo_before = len(self.canvas._undo_stack)
                before, after = self._drag_transform_handle(
                    handle,
                    delta,
                    Qt.KeyboardModifier.ControlModifier,
                )
                distort = self.canvas.transform()["distort"]

                for value_index, value in enumerate(distort):
                    if value_index == index:
                        self.assertAlmostEqual(value, delta.x(), delta=0.01)
                    elif value_index == index + 1:
                        self.assertAlmostEqual(value, delta.y(), delta=0.01)
                    else:
                        self.assertAlmostEqual(value, 0.0, delta=0.01)
                self._assert_point_close(before[opposite[handle]], after[opposite[handle]])
                self.assertEqual(len(self.canvas._undo_stack), undo_before + 1)

    def test_ctrl_side_handles_shear_selected_edge_and_fix_opposite_edge(self) -> None:
        edge_indices = {
            "n": (0, 2),
            "e": (2, 4),
            "s": (4, 6),
            "w": (6, 0),
        }
        opposite = {"n": "s", "e": "w", "s": "n", "w": "e"}
        delta = QPointF(8.0, 4.0)
        for handle, indices in edge_indices.items():
            with self.subTest(handle=handle):
                self._prepare_transform_canvas()
                undo_before = len(self.canvas._undo_stack)
                before, after = self._drag_transform_handle(
                    handle,
                    delta,
                    Qt.KeyboardModifier.ControlModifier,
                )
                distort = self.canvas.transform()["distort"]
                selected = {index for corner in indices for index in (corner, corner + 1)}

                for value_index, value in enumerate(distort):
                    if value_index in selected:
                        expected = delta.x() if value_index % 2 == 0 else delta.y()
                        self.assertAlmostEqual(value, expected, delta=0.01)
                    else:
                        self.assertAlmostEqual(value, 0.0, delta=0.01)
                self._assert_point_close(before[opposite[handle]], after[opposite[handle]])
                self.assertEqual(len(self.canvas._undo_stack), undo_before + 1)

    def test_complex_handle_drag_is_one_undo_and_supports_redo_and_reset(self) -> None:
        self._prepare_transform_canvas()
        undo_before = len(self.canvas._undo_stack)
        self._drag_transform_handle(
            "ne",
            QPointF(10.0, -7.0),
            Qt.KeyboardModifier.ControlModifier,
            intermediate_steps=4,
        )
        transformed = self.canvas.transform()

        self.assertNotEqual(transformed, self._default_transform())
        self.assertEqual(len(self.canvas._undo_stack), undo_before + 1)
        self.canvas.undo()
        self.assertEqual(self.canvas.transform(), self._default_transform())
        self.assertFalse(self.canvas.is_dirty)

        self.canvas.redo()
        self.assertEqual(self.canvas.transform(), transformed)
        self.assertTrue(self.canvas.is_dirty)

        self.canvas.reset_transform()
        self.assertEqual(self.canvas.transform(), self._default_transform())
        self.canvas.undo()
        self.assertEqual(self.canvas.transform(), transformed)

    def test_combined_stretch_distort_rotation_and_offset_output_is_not_clipped(self) -> None:
        image = QImage(120, 80, QImage.Format.Format_ARGB32)
        image.fill(QColor(24, 68, 112, 255))
        self.canvas.set_image(image, (120, 80))
        self.canvas.set_transform(
            x=-68.0,
            y=-52.0,
            scale=1.25,
            rotation=27.0,
            stretch_w=1.35,
            stretch_h=0.8,
            distort=[-14.0, -9.0, 12.0, -5.0, 18.0, 11.0, -10.0, 15.0],
        )

        result = self.canvas.image()
        origin = self.canvas.output_origin()

        self.assertFalse(result.isNull())
        self.assertGreater(origin.x(), 0)
        self.assertGreater(origin.y(), 0)
        self.assertEqual(result.width(), 120 + origin.x() * 2)
        self.assertEqual(result.height(), 80 + origin.y() * 2)
        polygon = self.canvas._transformed_content().polygon
        for point in polygon:
            self.assertGreaterEqual(point.x() + origin.x(), -1e-6)
            self.assertLessEqual(point.x() + origin.x(), result.width() + 1e-6)
            self.assertGreaterEqual(point.y() + origin.y(), -1e-6)
            self.assertLessEqual(point.y() + origin.y(), result.height() + 1e-6)
        self.assertGreater(self._opaque_pixel_count(result), 0)

    def test_asymmetric_perspective_rotation_keeps_control_center_fixed(self) -> None:
        self._prepare_transform_canvas()
        self.canvas.set_transform(
            distort=[-9.0, 3.0, 14.0, -6.0, 5.0, 11.0, -4.0, 7.0]
        )
        center_before = self._control_center()

        self.canvas.set_transform(rotation=38.0)

        self._assert_point_close(center_before, self._control_center(), delta=0.01)

    def test_transform_hover_cursor_matches_handle_direction_and_rotation(self) -> None:
        image = QImage(180, 40, QImage.Format.Format_ARGB32)
        image.fill(QColor("#222222"))
        self.canvas.set_image(image)
        self.canvas.actual_size()
        self.canvas.set_tool(ReviewCanvas.TOOL_TRANSFORM)
        handles, rotate = self.canvas._control_handles()

        expected = {
            "n": Qt.CursorShape.SizeVerCursor,
            "e": Qt.CursorShape.SizeHorCursor,
            "nw": Qt.CursorShape.SizeFDiagCursor,
            "ne": Qt.CursorShape.SizeBDiagCursor,
        }
        for name, shape in expected.items():
            self.canvas._update_cursor_at(handles[name])
            self.assertEqual(self.canvas.cursor().shape(), shape)
        self.canvas._update_cursor_at(self.canvas._control_polygon().boundingRect().center())
        self.assertEqual(self.canvas.cursor().shape(), Qt.CursorShape.SizeAllCursor)
        self.canvas._update_cursor_at(rotate)
        self.assertFalse(self.canvas.cursor().pixmap().isNull())

        self.canvas.set_transform(rotation=90)
        handles, _rotate = self.canvas._control_handles()
        self.canvas._update_cursor_at(handles["e"])
        self.assertEqual(self.canvas.cursor().shape(), Qt.CursorShape.SizeVerCursor)

    def test_external_transform_controls_map_to_requested_view_and_hit_test(self) -> None:
        self._prepare_transform_canvas()
        self.canvas.set_transform(x=3.0, y=-2.0, rotation=18.0)
        origin = QPointF(37.5, 52.25)
        scale = 1.75

        polygon, handles, rotate = self.canvas.transform_controls_in_view(origin, scale)

        logical_polygon = self.canvas._transformed_content().polygon
        self.assertEqual(polygon.count(), 4)
        for index, logical in enumerate(logical_polygon):
            expected = QPointF(
                origin.x() + logical.x() * scale,
                origin.y() + logical.y() * scale,
            )
            self._assert_point_close(polygon.at(index), expected)
        self.assertEqual(
            set(handles),
            {"nw", "n", "ne", "e", "se", "s", "sw", "w"},
        )
        self.assertAlmostEqual(
            self.canvas._distance(handles["n"], rotate),
            self.canvas._ROTATE_HANDLE_DISTANCE,
            delta=0.01,
        )
        self.assertEqual(
            self.canvas.transform_hit_test_in_view(handles["nw"], origin, scale),
            "scale:nw",
        )
        self.assertEqual(
            self.canvas.transform_hit_test_in_view(
                handles["nw"],
                origin,
                scale,
                Qt.KeyboardModifier.ControlModifier,
            ),
            "distort:nw",
        )
        self.assertEqual(
            self.canvas.transform_hit_test_in_view(
                handles["e"],
                origin,
                scale,
                Qt.KeyboardModifier.ControlModifier,
            ),
            "distort:e",
        )
        self.assertEqual(
            self.canvas.transform_hit_test_in_view(rotate, origin, scale),
            "rotate",
        )
        center = self.canvas._polygon_center(tuple(polygon.at(i) for i in range(4)))
        self.assertEqual(
            self.canvas.transform_hit_test_in_view(center, origin, scale),
            "move",
        )
        self.assertFalse(self.canvas.transform_cursor_for_hit("rotate").pixmap().isNull())

        with self.assertRaisesRegex(ValueError, "比例必须大于零"):
            self.canvas.transform_controls_in_view(origin, 0.0)
        with self.assertRaisesRegex(ValueError, "比例必须大于零"):
            self.canvas.transform_controls_in_view(origin, float("nan"))
        with self.assertRaisesRegex(ValueError, "必须为有限值"):
            self.canvas.transform_controls_in_view(QPointF(float("inf"), 0.0), 1.0)

    def test_external_move_uses_view_scale_and_one_history_entry(self) -> None:
        self._prepare_transform_canvas()
        origin, scale = self._external_transform_view()
        polygon, _handles, _rotate = self.canvas.transform_controls_in_view(origin, scale)
        start = self.canvas._polygon_center(tuple(polygon.at(i) for i in range(4)))
        undo_before = len(self.canvas._undo_stack)

        self.assertEqual(
            self.canvas.begin_external_transform(start, origin, scale),
            "move",
        )
        self.assertTrue(self.canvas.update_external_transform(start + QPointF(18.0, 4.0)))
        self.assertTrue(self.canvas.update_external_transform(start + QPointF(36.0, 8.0)))
        moved = self.canvas.transform()

        self.assertAlmostEqual(float(moved["x"]), 18.0)
        self.assertAlmostEqual(float(moved["y"]), 4.0)
        self.assertEqual(len(self.canvas._undo_stack), undo_before + 1)
        self.assertTrue(self.canvas.end_external_transform())
        self.assertFalse(self.canvas.end_external_transform())
        self.assertFalse(self.canvas.update_external_transform(start))

        self.canvas.undo()
        self.assertEqual(self.canvas.transform(), self._default_transform())
        self.canvas.redo()
        self.assertEqual(self.canvas.transform(), moved)

    def test_external_shift_move_and_shift_rotation_constraints(self) -> None:
        self._prepare_transform_canvas()
        origin, scale = self._external_transform_view()
        polygon, _handles, _rotate = self.canvas.transform_controls_in_view(origin, scale)
        start = self.canvas._polygon_center(tuple(polygon.at(i) for i in range(4)))

        self.assertEqual(self.canvas.begin_external_transform(start, origin, scale), "move")
        self.assertTrue(
            self.canvas.update_external_transform(
                start + QPointF(30.0, 10.0),
                Qt.KeyboardModifier.ShiftModifier,
            )
        )
        self.canvas.end_external_transform()
        moved = self.canvas.transform()
        self.assertAlmostEqual(float(moved["x"]), 15.0)
        self.assertAlmostEqual(float(moved["y"]), 0.0)

        self._prepare_transform_canvas()
        polygon, _handles, rotate = self.canvas.transform_controls_in_view(origin, scale)
        center = self.canvas._polygon_center(tuple(polygon.at(i) for i in range(4)))
        radius = self.canvas._distance(center, rotate)
        target = center + QPointF(radius, -radius * 0.2)
        self.assertEqual(
            self.canvas.begin_external_transform(rotate, origin, scale),
            "rotate",
        )
        self.assertTrue(
            self.canvas.update_external_transform(
                target,
                Qt.KeyboardModifier.ShiftModifier,
            )
        )
        self.canvas.end_external_transform()
        self.assertAlmostEqual(float(self.canvas.transform()["rotation"]), 75.0)

    def test_external_corner_and_side_scaling_keep_expected_anchor(self) -> None:
        origin, scale = self._external_transform_view()

        self._prepare_transform_canvas()
        _kind, before, after = self._drag_external_transform(
            "se",
            QPointF(24.0, 16.0),
            origin=origin,
            scale=scale,
        )
        transformed = self.canvas.transform()
        self.assertGreater(float(transformed["scale"]), 1.0)
        self.assertAlmostEqual(float(transformed["stretch_w"]), 1.0)
        self.assertAlmostEqual(float(transformed["stretch_h"]), 1.0)
        self._assert_point_close(before["nw"], after["nw"])

        self._prepare_transform_canvas()
        _kind, before, after = self._drag_external_transform(
            "e",
            QPointF(28.0, 0.0),
            origin=origin,
            scale=scale,
        )
        transformed = self.canvas.transform()
        self.assertGreater(float(transformed["stretch_w"]), 1.0)
        self.assertAlmostEqual(float(transformed["scale"]), 1.0)
        self._assert_point_close(before["w"], after["w"])

    def test_external_shift_side_scales_equally_and_alt_keeps_center(self) -> None:
        origin, scale = self._external_transform_view()

        self._prepare_transform_canvas()
        self._drag_external_transform(
            "e",
            QPointF(26.0, 0.0),
            modifiers=Qt.KeyboardModifier.ShiftModifier,
            origin=origin,
            scale=scale,
        )
        transformed = self.canvas.transform()
        self.assertGreater(float(transformed["scale"]), 1.0)
        self.assertAlmostEqual(float(transformed["stretch_w"]), 1.0)
        self.assertAlmostEqual(float(transformed["stretch_h"]), 1.0)

        self._prepare_transform_canvas()
        _kind, before, after = self._drag_external_transform(
            "e",
            QPointF(26.0, 0.0),
            modifiers=Qt.KeyboardModifier.AltModifier,
            origin=origin,
            scale=scale,
        )
        transformed = self.canvas.transform()
        self.assertGreater(float(transformed["stretch_w"]), 1.0)
        self._assert_point_close(before["center"], after["center"])

    def test_external_ctrl_corner_and_edge_reuse_distortion_pipeline(self) -> None:
        origin, scale = self._external_transform_view()

        self._prepare_transform_canvas()
        undo_before = len(self.canvas._undo_stack)
        _kind, before, after = self._drag_external_transform(
            "nw",
            QPointF(14.0, 10.0),
            modifiers=Qt.KeyboardModifier.ControlModifier,
            origin=origin,
            scale=scale,
            intermediate_steps=4,
        )
        distort = self.canvas.transform()["distort"]
        self.assertAlmostEqual(distort[0], 7.0, delta=0.01)
        self.assertAlmostEqual(distort[1], 5.0, delta=0.01)
        for value in distort[2:]:
            self.assertAlmostEqual(value, 0.0, delta=0.01)
        self._assert_point_close(before["se"], after["se"])
        self.assertEqual(len(self.canvas._undo_stack), undo_before + 1)

        self._prepare_transform_canvas()
        _kind, before, after = self._drag_external_transform(
            "n",
            QPointF(16.0, 8.0),
            modifiers=Qt.KeyboardModifier.ControlModifier,
            origin=origin,
            scale=scale,
        )
        distort = self.canvas.transform()["distort"]
        for index, value in enumerate(distort):
            expected = 8.0 if index in {0, 2} else 4.0 if index in {1, 3} else 0.0
            self.assertAlmostEqual(value, expected, delta=0.01)
        self._assert_point_close(before["s"], after["s"])

    def test_external_transform_session_is_cleared_by_state_changes(self) -> None:
        self._prepare_transform_canvas()
        origin, scale = self._external_transform_view()
        polygon, _handles, _rotate = self.canvas.transform_controls_in_view(origin, scale)
        center = self.canvas._polygon_center(tuple(polygon.at(i) for i in range(4)))

        self.assertEqual(
            self.canvas.begin_external_transform(center, origin, scale),
            "move",
        )
        self.canvas.undo()
        self.assertFalse(self.canvas.update_external_transform(center + QPointF(20.0, 0.0)))
        self.assertIsNone(self.canvas._transform_drag_view)

        self.assertEqual(
            self.canvas.begin_external_transform(center, origin, scale),
            "move",
        )
        self.canvas.clear_image()
        self.assertFalse(self.canvas.update_external_transform(center + QPointF(20.0, 0.0)))
        polygon, handles, rotate = self.canvas.transform_controls_in_view(origin, scale)
        self.assertTrue(polygon.isEmpty())
        self.assertFalse(handles)
        self.assertTrue(rotate.isNull())

    def test_tablet_hover_hides_system_cursor_and_clear_restores_it(self) -> None:
        self._prepare_tablet_canvas()
        device = self._tablet_device(QPointingDevice.PointerType.Pen)
        position = self._widget_position(40, 40)
        hover = self._tablet_event(
            QEvent.Type.TabletMove,
            device,
            position,
            0.5,
            Qt.MouseButton.NoButton,
            Qt.MouseButton.NoButton,
        )

        QApplication.sendEvent(self.canvas, hover)

        self.assertTrue(hover.isAccepted())
        self.assertEqual(self.canvas.cursor().shape(), Qt.CursorShape.BlankCursor)
        self.assertEqual(self.canvas._pointer_position, position)

        self.canvas.clear_image()

        self.assertEqual(self.canvas.cursor().shape(), Qt.CursorShape.ArrowCursor)
        self.assertIsNone(self.canvas._pointer_position)

    def test_brush_wheel_and_brackets_change_size_without_zooming(self) -> None:
        self.canvas.set_tool(ReviewCanvas.TOOL_BRUSH)
        self.canvas.actual_size()
        self.canvas.set_brush_size(10)
        changed: list[int] = []
        self.canvas.brush_size_changed.connect(changed.append)
        original_zoom = self.canvas._zoom

        QApplication.sendEvent(
            self.canvas,
            self._wheel_event(120, Qt.KeyboardModifier.NoModifier),
        )

        self.assertEqual(self.canvas.brush_size, 12)
        self.assertEqual(self.canvas._zoom, original_zoom)
        self.assertEqual(changed, [12])

        QApplication.sendEvent(
            self.canvas,
            QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_BracketLeft,
                Qt.KeyboardModifier.NoModifier,
                "[",
            ),
        )
        self.assertEqual(self.canvas.brush_size, 10)

        self.canvas.set_brush_size(30)
        QApplication.sendEvent(
            self.canvas,
            QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_BracketRight,
                Qt.KeyboardModifier.NoModifier,
                "]",
            ),
        )
        self.assertEqual(self.canvas.brush_size, 35)

        QApplication.sendEvent(
            self.canvas,
            self._wheel_event(120, Qt.KeyboardModifier.ControlModifier),
        )
        self.assertEqual(self.canvas.brush_size, 35)
        self.assertGreater(self.canvas._zoom, original_zoom)

    def test_high_resolution_wheel_accumulates_partial_steps(self) -> None:
        self.canvas.set_tool(ReviewCanvas.TOOL_BRUSH)
        self.canvas.set_brush_size(10)
        changed: list[int] = []
        self.canvas.brush_size_changed.connect(changed.append)

        QApplication.sendEvent(
            self.canvas,
            self._wheel_event(60, Qt.KeyboardModifier.NoModifier),
        )
        self.assertEqual(self.canvas.brush_size, 10)
        self.assertEqual(changed, [])

        QApplication.sendEvent(
            self.canvas,
            self._wheel_event(60, Qt.KeyboardModifier.NoModifier),
        )
        self.assertEqual(self.canvas.brush_size, 12)
        self.assertEqual(changed, [12])

    def test_round_brush_preview_tracks_width_zoom_tool_and_leave(self) -> None:
        image = QImage(100, 80, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        self.canvas.set_image(image)
        self.canvas.actual_size()
        self.canvas.set_tool(ReviewCanvas.TOOL_BRUSH)
        self.canvas.set_brush_size(20)
        position = self.canvas._canvas_rect().center()

        self.canvas._set_pointer_preview(position, 20.0, ReviewCanvas.TOOL_BRUSH)

        self.assertAlmostEqual(self.canvas._pointer_preview_radius(), 10.0)
        self.canvas._update_cursor_at(position)
        self.assertEqual(self.canvas.cursor().shape(), Qt.CursorShape.BlankCursor)
        self.canvas.zoom_in()
        self.assertAlmostEqual(self.canvas._pointer_preview_radius(), 12.0)
        self.canvas._set_pointer_preview(position, 8.0, ReviewCanvas.TOOL_ERASER)
        self.assertEqual(self.canvas._pointer_tool, ReviewCanvas.TOOL_ERASER)
        self.assertEqual(self.canvas.tool, ReviewCanvas.TOOL_BRUSH)

        QApplication.sendEvent(self.canvas, QEvent(QEvent.Type.Leave))

        self.assertIsNone(self.canvas._pointer_position)
        self.assertEqual(self.canvas.cursor().shape(), Qt.CursorShape.ArrowCursor)

    def test_tablet_pressure_changes_width_and_one_stroke_uses_one_undo(self) -> None:
        self._prepare_tablet_canvas()
        device = self._tablet_device(QPointingDevice.PointerType.Pen)

        self._send_tablet_stroke(device, ((20, 20, 0.1), (35, 20, 0.5), (50, 20, 1.0)))

        result = self.canvas.image()
        light_width = self._column_ink_height(result, 20)
        firm_width = self._column_ink_height(result, 50)
        self.assertGreater(firm_width, light_width)
        self.assertEqual(len(self.canvas._undo_stack), 1)
        self.canvas.undo()
        self.assertEqual(self._opaque_pixel_count(self.canvas.image()), 0)

    def test_space_tablet_tip_and_eraser_pan_without_editing(self) -> None:
        for pointer_type in (
            QPointingDevice.PointerType.Pen,
            QPointingDevice.PointerType.Eraser,
        ):
            with self.subTest(pointer_type=pointer_type):
                self._prepare_tablet_canvas()
                device = self._tablet_device(pointer_type)
                original_image = self.canvas.image()
                original_offset = QPointF(self.canvas._offset)
                start = self._widget_position(24, 28)
                end = start + QPointF(18.0, 13.0)

                QApplication.sendEvent(
                    self.canvas,
                    QKeyEvent(
                        QEvent.Type.KeyPress,
                        Qt.Key.Key_Space,
                        Qt.KeyboardModifier.NoModifier,
                        " ",
                    ),
                )
                press = self._tablet_event(
                    QEvent.Type.TabletPress,
                    device,
                    start,
                    0.5,
                    Qt.MouseButton.LeftButton,
                    Qt.MouseButton.LeftButton,
                )
                move = self._tablet_event(
                    QEvent.Type.TabletMove,
                    device,
                    end,
                    0.5,
                    Qt.MouseButton.NoButton,
                    Qt.MouseButton.LeftButton,
                )
                release = self._tablet_event(
                    QEvent.Type.TabletRelease,
                    device,
                    end,
                    0.0,
                    Qt.MouseButton.LeftButton,
                    Qt.MouseButton.NoButton,
                )
                QApplication.sendEvent(self.canvas, press)
                QApplication.sendEvent(self.canvas, move)
                QApplication.sendEvent(self.canvas, release)
                QApplication.sendEvent(
                    self.canvas,
                    QKeyEvent(
                        QEvent.Type.KeyRelease,
                        Qt.Key.Key_Space,
                        Qt.KeyboardModifier.NoModifier,
                        " ",
                    ),
                )

                self.assertTrue(press.isAccepted())
                self.assertTrue(move.isAccepted())
                self.assertTrue(release.isAccepted())
                self._assert_point_close(self.canvas._offset, original_offset + end - start, 0.01)
                self.assertEqual(self.canvas.image(), original_image)
                self.assertFalse(self.canvas.is_dirty)
                self.assertEqual(len(self.canvas._undo_stack), 0)
                self.assertEqual(self.canvas.tool, ReviewCanvas.TOOL_BRUSH)

    def test_disabled_pressure_uses_fixed_width(self) -> None:
        self._prepare_tablet_canvas()
        self.canvas.set_pressure_enabled(False)
        device = self._tablet_device(QPointingDevice.PointerType.Pen)

        self._send_tablet_stroke(device, ((20, 20, 0.1), (50, 20, 1.0)))

        result = self.canvas.image()
        self.assertLessEqual(
            abs(self._column_ink_height(result, 20) - self._column_ink_height(result, 50)),
            1,
        )

    def test_tablet_eraser_tip_temporarily_erases_without_changing_tool(self) -> None:
        image = QImage(80, 80, QImage.Format.Format_ARGB32)
        image.fill(QColor("#222222"))
        self.canvas.set_image(image)
        self.canvas.actual_size()
        self.canvas.set_tool(ReviewCanvas.TOOL_BRUSH)
        self.canvas.set_brush_size(18)
        device = self._tablet_device(QPointingDevice.PointerType.Eraser)

        self._send_tablet_stroke(device, ((40, 40, 1.0),))

        self.assertEqual(self.canvas.tool, ReviewCanvas.TOOL_BRUSH)
        self.assertEqual(self.canvas.image().pixelColor(40, 40).alpha(), 0)
        self.canvas.undo()
        self.assertEqual(self.canvas.image().pixelColor(40, 40).alpha(), 255)

    def test_synthesized_mouse_events_after_tablet_stroke_are_ignored(self) -> None:
        self._prepare_tablet_canvas()
        device = self._tablet_device(QPointingDevice.PointerType.Pen)
        points = ((20, 30, 0.5), (40, 30, 0.75), (60, 30, 1.0))

        self._send_tablet_stroke(device, points)
        tablet_result = self.canvas.image()
        undo_count = len(self.canvas._undo_stack)
        self._send_mouse_stroke(
            tuple((x, y) for x, y, _pressure in points),
            Qt.MouseEventSource.MouseEventSynthesizedByQt,
            device,
        )

        self.assertEqual(self.canvas.image(), tablet_result)
        self.assertEqual(len(self.canvas._undo_stack), undo_count)
        self.canvas.undo()
        self.assertEqual(self._opaque_pixel_count(self.canvas.image()), 0)

    def test_real_mouse_stroke_after_tablet_stroke_is_not_suppressed(self) -> None:
        self._prepare_tablet_canvas()
        tablet = self._tablet_device(QPointingDevice.PointerType.Pen)
        mouse = self._pointing_device(
            "测试鼠标",
            201,
            QInputDevice.DeviceType.Mouse,
            QPointingDevice.PointerType.Generic,
        )
        self._send_tablet_stroke(tablet, ((20, 20, 0.5), (40, 20, 0.8)))
        tablet_result = self.canvas.image()
        undo_count = len(self.canvas._undo_stack)

        self._send_mouse_stroke(
            ((20, 60), (40, 60)),
            Qt.MouseEventSource.MouseEventNotSynthesized,
            mouse,
        )

        self.assertNotEqual(self.canvas.image(), tablet_result)
        self.assertGreater(self.canvas.image().pixelColor(30, 60).alpha(), 0)
        self.assertEqual(len(self.canvas._undo_stack), undo_count + 1)
        self.canvas.undo()
        self.assertEqual(self.canvas.image(), tablet_result)

    def test_touchscreen_synthesized_mouse_stroke_after_tablet_is_not_suppressed(self) -> None:
        self._prepare_tablet_canvas()
        tablet = self._tablet_device(QPointingDevice.PointerType.Pen)
        touchscreen = self._pointing_device(
            "测试触摸屏",
            202,
            QInputDevice.DeviceType.TouchScreen,
            QPointingDevice.PointerType.Finger,
        )
        self._send_tablet_stroke(tablet, ((20, 20, 0.5), (40, 20, 0.8)))
        tablet_result = self.canvas.image()
        undo_count = len(self.canvas._undo_stack)

        self._send_mouse_stroke(
            ((50, 55), (65, 55)),
            Qt.MouseEventSource.MouseEventSynthesizedByQt,
            touchscreen,
        )

        self.assertNotEqual(self.canvas.image(), tablet_result)
        self.assertGreater(self.canvas.image().pixelColor(58, 55).alpha(), 0)
        self.assertEqual(len(self.canvas._undo_stack), undo_count + 1)
        self.canvas.undo()
        self.assertEqual(self.canvas.image(), tablet_result)

    def test_transform_accepts_stylus_mouse_compatibility_after_tool_switch(self) -> None:
        self._prepare_tablet_canvas()
        device = self._tablet_device(QPointingDevice.PointerType.Pen)
        self._send_tablet_stroke(device, ((20, 20, 0.5), (60, 60, 0.8)))
        self.assertIsNotNone(self.canvas._suppressed_tablet_device_key)

        self.canvas.set_tool(ReviewCanvas.TOOL_TRANSFORM)

        self.assertIsNone(self.canvas._suppressed_tablet_device_key)
        ignored_press = self._tablet_event(
            QEvent.Type.TabletPress,
            device,
            self._widget_position(40, 40),
            0.8,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
        )
        QApplication.sendEvent(self.canvas, ignored_press)
        self.assertFalse(ignored_press.isAccepted())

        self._send_mouse_stroke(
            ((40, 40), (55, 40)),
            Qt.MouseEventSource.MouseEventSynthesizedByQt,
            device,
        )

        self.assertGreater(self.canvas.transform()["x"], 0.0)

    def test_tablet_without_pressure_capability_uses_base_brush_width(self) -> None:
        self._prepare_tablet_canvas()
        device = self._tablet_device(
            QPointingDevice.PointerType.Pen,
            QInputDevice.Capability.Position,
        )

        self._send_tablet_stroke(device, ((20, 30, 0.05),))
        self._send_tablet_stroke(device, ((60, 30, 1.0),))

        result = self.canvas.image()
        light_width = self._column_ink_height(result, 20)
        firm_width = self._column_ink_height(result, 60)
        self.assertLessEqual(abs(light_width - firm_width), 1)
        self.assertAlmostEqual(light_width, 20, delta=2)

    def test_undo_during_active_tablet_stroke_cancels_input_and_redo_restores_it(self) -> None:
        self._prepare_tablet_canvas()
        device = self._tablet_device(QPointingDevice.PointerType.Pen)
        self._send_tablet_input(
            QEvent.Type.TabletPress,
            device,
            20,
            30,
            0.5,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
        )
        self._send_tablet_input(
            QEvent.Type.TabletMove,
            device,
            40,
            30,
            0.8,
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
        )
        partial_stroke = self.canvas.image()

        self.canvas.undo()

        self.assertEqual(self._opaque_pixel_count(self.canvas.image()), 0)
        self.assertFalse(self.canvas._drawing)
        self.assertFalse(self.canvas._tablet_active)
        self._send_tablet_input(
            QEvent.Type.TabletMove,
            device,
            65,
            30,
            1.0,
            Qt.MouseButton.NoButton,
            Qt.MouseButton.NoButton,
        )
        self.assertEqual(self._opaque_pixel_count(self.canvas.image()), 0)

        self.canvas.redo()

        self.assertEqual(self.canvas.image(), partial_stroke)

    def test_tablet_move_without_pressed_button_ends_missing_release_stroke(self) -> None:
        self._prepare_tablet_canvas()
        device = self._tablet_device(QPointingDevice.PointerType.Pen)
        self._send_tablet_input(
            QEvent.Type.TabletPress,
            device,
            20,
            30,
            0.5,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
        )
        self._send_tablet_input(
            QEvent.Type.TabletMove,
            device,
            35,
            30,
            0.8,
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
        )
        completed_stroke = self.canvas.image()

        self._send_tablet_input(
            QEvent.Type.TabletMove,
            device,
            50,
            30,
            0.6,
            Qt.MouseButton.NoButton,
            Qt.MouseButton.NoButton,
        )

        self.assertEqual(self.canvas.image(), completed_stroke)
        self.assertFalse(self.canvas._drawing)
        self.assertFalse(self.canvas._tablet_active)
        self._send_tablet_input(
            QEvent.Type.TabletMove,
            device,
            70,
            30,
            1.0,
            Qt.MouseButton.NoButton,
            Qt.MouseButton.NoButton,
        )
        self.assertEqual(self.canvas.image(), completed_stroke)

    def _prepare_transform_canvas(self) -> None:
        self.canvas.clear_image()
        image = QImage(120, 80, QImage.Format.Format_ARGB32)
        image.fill(QColor(24, 68, 112, 255))
        self.canvas.set_image(image, (120, 80))
        self.canvas.actual_size()
        self.canvas.set_tool(ReviewCanvas.TOOL_TRANSFORM)

    @staticmethod
    def _external_transform_view() -> tuple[QPointF, float]:
        return QPointF(41.0, 63.0), 2.0

    def _drag_external_transform(
        self,
        handle: str,
        delta: QPointF,
        modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
        *,
        origin: QPointF,
        scale: float,
        intermediate_steps: int = 1,
    ) -> tuple[str, dict[str, QPointF], dict[str, QPointF]]:
        polygon, handles, rotate = self.canvas.transform_controls_in_view(origin, scale)
        center = self.canvas._polygon_center(tuple(polygon.at(i) for i in range(4)))
        before = {name: QPointF(point) for name, point in handles.items()}
        before["center"] = QPointF(center)
        before["rotate"] = QPointF(rotate)
        start = center if handle == "move" else rotate if handle == "rotate" else handles[handle]

        kind = self.canvas.begin_external_transform(start, origin, scale, modifiers)
        steps = max(1, int(intermediate_steps))
        for step in range(1, steps + 1):
            self.assertTrue(
                self.canvas.update_external_transform(
                    start + delta * (step / steps),
                    modifiers,
                )
            )
        self.assertTrue(self.canvas.end_external_transform())

        after_polygon, after_handles, after_rotate = self.canvas.transform_controls_in_view(
            origin,
            scale,
        )
        after_center = self.canvas._polygon_center(
            tuple(after_polygon.at(i) for i in range(4))
        )
        after = {name: QPointF(point) for name, point in after_handles.items()}
        after["center"] = QPointF(after_center)
        after["rotate"] = QPointF(after_rotate)
        return kind, before, after

    def _drag_transform_handle(
        self,
        handle: str,
        delta: QPointF,
        modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
        *,
        intermediate_steps: int = 1,
    ) -> tuple[dict[str, QPointF], dict[str, QPointF]]:
        handles, _rotate = self.canvas._control_handles()
        before = {name: QPointF(point) for name, point in handles.items()}
        start = QPointF(before[handle])
        press = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            start,
            start,
            start,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            modifiers,
        )
        QApplication.sendEvent(self.canvas, press)
        steps = max(1, int(intermediate_steps))
        end = QPointF(start)
        for step in range(1, steps + 1):
            ratio = step / steps
            end = start + delta * ratio
            move = QMouseEvent(
                QEvent.Type.MouseMove,
                end,
                end,
                end,
                Qt.MouseButton.NoButton,
                Qt.MouseButton.LeftButton,
                modifiers,
            )
            QApplication.sendEvent(self.canvas, move)
        release = QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            end,
            end,
            end,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
            modifiers,
        )
        QApplication.sendEvent(self.canvas, release)
        after_handles, _rotate = self.canvas._control_handles()
        after = {name: QPointF(point) for name, point in after_handles.items()}
        return before, after

    def _control_center(self) -> QPointF:
        polygon = self.canvas._control_polygon()
        count = polygon.count()
        if count == 0:
            return QPointF()
        return QPointF(
            sum(polygon.at(index).x() for index in range(count)) / count,
            sum(polygon.at(index).y() for index in range(count)) / count,
        )

    def _assert_point_close(
        self,
        first: QPointF,
        second: QPointF,
        delta: float = 1.25,
    ) -> None:
        self.assertAlmostEqual(first.x(), second.x(), delta=delta)
        self.assertAlmostEqual(first.y(), second.y(), delta=delta)

    @staticmethod
    def _default_transform() -> dict[str, object]:
        return {
            "x": 0.0,
            "y": 0.0,
            "scale": 1.0,
            "rotation": 0.0,
            "stretch_w": 1.0,
            "stretch_h": 1.0,
            "distort": [0.0] * 8,
        }

    def _prepare_tablet_canvas(self) -> None:
        image = QImage(80, 80, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        self.canvas.set_image(image)
        self.canvas.actual_size()
        self.canvas.set_tool(ReviewCanvas.TOOL_BRUSH)
        self.canvas.set_brush_size(20)

    def _send_tablet_stroke(
        self,
        device: QPointingDevice,
        points: tuple[tuple[int, int, float], ...],
    ) -> None:
        for index, (x, y, pressure) in enumerate(points):
            event_type = QEvent.Type.TabletPress if index == 0 else QEvent.Type.TabletMove
            button = Qt.MouseButton.LeftButton if index == 0 else Qt.MouseButton.NoButton
            event = self._tablet_event(
                event_type,
                device,
                self._widget_position(x, y),
                pressure,
                button,
                Qt.MouseButton.LeftButton,
            )
            self.assertTrue(QApplication.sendEvent(self.canvas, event))
            self.assertTrue(event.isAccepted())
        last_x, last_y, _pressure = points[-1]
        release = self._tablet_event(
            QEvent.Type.TabletRelease,
            device,
            self._widget_position(last_x, last_y),
            0.0,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
        )
        self.assertTrue(QApplication.sendEvent(self.canvas, release))
        self.assertTrue(release.isAccepted())

    def _send_tablet_input(
        self,
        event_type: QEvent.Type,
        device: QPointingDevice,
        x: int,
        y: int,
        pressure: float,
        button: Qt.MouseButton,
        buttons: Qt.MouseButton,
    ) -> None:
        event = self._tablet_event(
            event_type,
            device,
            self._widget_position(x, y),
            pressure,
            button,
            buttons,
        )
        self.assertTrue(QApplication.sendEvent(self.canvas, event))
        self.assertTrue(event.isAccepted())

    def _tablet_device(
        self,
        pointer_type: QPointingDevice.PointerType,
        capabilities: QInputDevice.Capability | None = None,
    ) -> QPointingDevice:
        if capabilities is None:
            capabilities = QInputDevice.Capability.Position | QInputDevice.Capability.Pressure
        return QPointingDevice(
            "测试绘图笔",
            100 + int(pointer_type.value),
            QInputDevice.DeviceType.Stylus,
            pointer_type,
            capabilities,
            1,
            3,
            parent=self.canvas,
        )

    def _pointing_device(
        self,
        name: str,
        system_id: int,
        device_type: QInputDevice.DeviceType,
        pointer_type: QPointingDevice.PointerType,
    ) -> QPointingDevice:
        return QPointingDevice(
            name,
            system_id,
            device_type,
            pointer_type,
            QInputDevice.Capability.Position | QInputDevice.Capability.MouseEmulation,
            1,
            3,
            parent=self.canvas,
        )

    def _send_mouse_stroke(
        self,
        points: tuple[tuple[int, int], ...],
        source: Qt.MouseEventSource,
        device: QPointingDevice,
    ) -> None:
        for index, (x, y) in enumerate(points):
            event_type = QEvent.Type.MouseButtonPress if index == 0 else QEvent.Type.MouseMove
            button = Qt.MouseButton.LeftButton if index == 0 else Qt.MouseButton.NoButton
            event = self._mouse_event(
                event_type,
                self._widget_position(x, y),
                button,
                Qt.MouseButton.LeftButton,
                source,
                device,
            )
            self.assertTrue(QApplication.sendEvent(self.canvas, event))
            self.assertTrue(event.isAccepted())
        last_x, last_y = points[-1]
        release = self._mouse_event(
            QEvent.Type.MouseButtonRelease,
            self._widget_position(last_x, last_y),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
            source,
            device,
        )
        self.assertTrue(QApplication.sendEvent(self.canvas, release))
        self.assertTrue(release.isAccepted())

    def _widget_position(self, x: int, y: int) -> QPointF:
        offset_x = (self.canvas.width() - self.canvas.canvas_size().width()) / 2.0
        offset_y = (self.canvas.height() - self.canvas.canvas_size().height()) / 2.0
        return QPointF(offset_x + x, offset_y + y)

    def _render_canvas(self) -> QImage:
        rendered = QImage(self.canvas.size(), QImage.Format.Format_ARGB32)
        rendered.fill(Qt.GlobalColor.transparent)
        painter = QPainter(rendered)
        self.canvas.render(painter, QPoint())
        painter.end()
        return rendered

    def _wheel_event(
        self,
        delta: int,
        modifiers: Qt.KeyboardModifier,
    ) -> QWheelEvent:
        position = self.canvas.rect().center()
        return QWheelEvent(
            QPointF(position),
            QPointF(position),
            QPoint(),
            QPoint(0, delta),
            Qt.MouseButton.NoButton,
            modifiers,
            Qt.ScrollPhase.ScrollUpdate,
            False,
        )

    @staticmethod
    def _mouse_event(
        event_type: QEvent.Type,
        position: QPointF,
        button: Qt.MouseButton,
        buttons: Qt.MouseButton,
        source: Qt.MouseEventSource,
        device: QPointingDevice,
    ) -> QMouseEvent:
        return QMouseEvent(
            event_type,
            position,
            position,
            position,
            button,
            buttons,
            Qt.KeyboardModifier.NoModifier,
            source,
            device=device,
        )

    @staticmethod
    def _tablet_event(
        event_type: QEvent.Type,
        device: QPointingDevice,
        position: QPointF,
        pressure: float,
        button: Qt.MouseButton,
        buttons: Qt.MouseButton,
    ) -> QTabletEvent:
        return QTabletEvent(
            event_type,
            device,
            position,
            position,
            pressure,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            Qt.KeyboardModifier.NoModifier,
            button,
            buttons,
        )

    @staticmethod
    def _sample_image() -> QImage:
        image = QImage(12, 12, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        for y in range(4, 7):
            for x in range(4, 7):
                image.setPixelColor(x, y, QColor(24, 68, 112, 255))
        return image

    @staticmethod
    def _opaque_pixel_count(image: QImage) -> int:
        return sum(
            image.pixelColor(x, y).alpha() > 0
            for y in range(image.height())
            for x in range(image.width())
        )

    @staticmethod
    def _column_ink_height(image: QImage, x: int) -> int:
        return sum(image.pixelColor(x, y).alpha() > 0 for y in range(image.height()))

    @staticmethod
    def _point(x: int, y: int):
        return QPoint(x, y)


if __name__ == "__main__":
    unittest.main()
