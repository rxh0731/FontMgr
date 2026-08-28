"""图片实验室高清区域画布测试。"""

from __future__ import annotations

import os
import unittest

import numpy as np
from PySide6.QtCore import QEvent, QPoint, QPointF, QRect, Qt
from PySide6.QtGui import QImage, QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication

from ui.widgets.image_lab_canvas import VIEW_ORIGINAL, ImageLabCanvas


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class ImageLabCanvasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _canvas_with_preview() -> ImageLabCanvas:
        canvas = ImageLabCanvas()
        source = np.zeros((100, 200, 3), dtype=np.uint8)
        alpha = np.zeros((100, 200), dtype=np.uint8)
        canvas.set_preview(
            source,
            source,
            alpha,
            alpha,
            source_width=200,
            source_height=100,
        )
        return canvas

    @staticmethod
    def _wheel_event(
        delta: int,
        modifiers: Qt.KeyboardModifier,
    ) -> QWheelEvent:
        position = QPointF(50.0, 50.0)
        return QWheelEvent(
            position,
            position,
            QPoint(),
            QPoint(0, delta),
            Qt.MouseButton.NoButton,
            modifiers,
            Qt.ScrollPhase.ScrollUpdate,
            False,
        )

    def test_only_alt_wheel_changes_zoom(self) -> None:
        canvas = self._canvas_with_preview()

        control_wheel = self._wheel_event(
            120,
            Qt.KeyboardModifier.ControlModifier,
        )
        canvas.wheelEvent(control_wheel)
        self.assertAlmostEqual(canvas.zoom_factor, 1.0)

        alt_wheel = self._wheel_event(120, Qt.KeyboardModifier.AltModifier)
        canvas.wheelEvent(alt_wheel)
        self.assertAlmostEqual(canvas.zoom_factor, 1.15)
        canvas.deleteLater()

    def test_alt_horizontal_wheel_delta_from_mouse_driver_changes_zoom(self) -> None:
        canvas = self._canvas_with_preview()
        position = QPointF(50.0, 50.0)
        wheel = QWheelEvent(
            position,
            position,
            QPoint(),
            QPoint(-120, 0),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.AltModifier,
            Qt.ScrollPhase.ScrollUpdate,
            False,
        )

        canvas.wheelEvent(wheel)

        self.assertTrue(wheel.isAccepted())
        self.assertAlmostEqual(canvas.zoom_factor, 1.0 / 1.15)
        canvas.deleteLater()

    def test_space_left_drag_emits_pan_without_creating_stroke(self) -> None:
        canvas = self._canvas_with_preview()
        pan_deltas: list[QPoint] = []
        strokes: list[tuple[object, ...]] = []
        canvas.pan_requested.connect(pan_deltas.append)
        canvas.stroke_finished.connect(lambda *args: strokes.append(args))

        self.assertTrue(canvas.set_pan_modifier_active(True))
        self.assertEqual(canvas.cursor().shape(), Qt.CursorShape.OpenHandCursor)
        start = QPointF(40.0, 30.0)
        start_global = QPointF(140.0, 130.0)
        press = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            start,
            start,
            start_global,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        canvas.mousePressEvent(press)
        self.assertEqual(canvas.cursor().shape(), Qt.CursorShape.ClosedHandCursor)

        moved = QPointF(53.0, 39.0)
        moved_global = QPointF(153.0, 139.0)
        move = QMouseEvent(
            QEvent.Type.MouseMove,
            moved,
            moved,
            moved_global,
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        canvas.mouseMoveEvent(move)
        release = QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            moved,
            moved,
            moved_global,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        canvas.mouseReleaseEvent(release)

        self.assertEqual(pan_deltas, [QPoint(13, 9)])
        self.assertEqual(strokes, [])
        self.assertEqual(canvas.cursor().shape(), Qt.CursorShape.OpenHandCursor)
        self.assertTrue(canvas.set_pan_modifier_active(False))
        self.assertEqual(canvas.cursor().shape(), Qt.CursorShape.ArrowCursor)
        canvas.deleteLater()

    def test_releasing_space_during_drag_keeps_closed_hand_until_mouse_release(self) -> None:
        canvas = self._canvas_with_preview()
        canvas.set_pan_modifier_active(True)
        position = QPointF(30.0, 25.0)
        press = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            position,
            position,
            position,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        canvas.mousePressEvent(press)

        canvas.set_pan_modifier_active(False)
        self.assertEqual(canvas.cursor().shape(), Qt.CursorShape.ClosedHandCursor)
        release = QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            position,
            position,
            position,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        canvas.mouseReleaseEvent(release)
        self.assertEqual(canvas.cursor().shape(), Qt.CursorShape.ArrowCursor)
        self.assertFalse(canvas.space_pan_active)
        canvas.deleteLater()

    def test_detail_region_uses_source_coordinates_and_overlays_preview(self) -> None:
        canvas = ImageLabCanvas()
        source = np.zeros((100, 200, 3), dtype=np.uint8)
        alpha = np.zeros((100, 200), dtype=np.uint8)
        canvas.set_preview(
            source,
            source,
            alpha,
            alpha,
            source_width=1000,
            source_height=500,
        )
        canvas.set_zoom(2.0)
        canvas.set_view_mode(VIEW_ORIGINAL)

        visible_source = canvas.source_rect_for_canvas_rect(
            QRect(100, 50, 200, 100)
        )
        self.assertEqual(visible_source, (250, 125, 750, 375))
        detail = np.empty((100, 200, 3), dtype=np.uint8)
        detail[:, :, :] = (240, 15, 20)
        detail_alpha = np.zeros((100, 200), dtype=np.uint8)
        canvas.set_detail_preview(
            detail,
            detail,
            detail_alpha,
            detail_alpha,
            visible_source,
        )

        self.assertTrue(canvas.detail_covers(visible_source, 0.39))
        self.assertFalse(canvas.detail_covers(visible_source, 0.41))
        rendered = QImage(canvas.size(), QImage.Format.Format_ARGB32)
        rendered.fill(0)
        canvas.render(rendered)
        center = rendered.pixelColor(200, 100)
        outside = rendered.pixelColor(20, 20)
        self.assertGreater(center.red(), 220)
        self.assertLess(center.green(), 40)
        self.assertLess(outside.red(), 10)
        canvas.deleteLater()


if __name__ == "__main__":
    unittest.main()
