from __future__ import annotations

import logging
import math
import sys
import time
from collections import deque
from pathlib import Path
from statistics import mean, median

from PySide6.QtCore import QPointF, QRect, QRectF, Qt, QTimer
from PySide6.QtGui import QAction, QColor, QImage, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "测试数据"
LOG_DIR = BASE_DIR / "测试日志"
DEFAULT_IMAGE = DATA_DIR / "是-0001.png"


class PerformanceRecorder:
    def __init__(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / f"PySide6画布性能_{time.strftime('%Y%m%d_%H%M%S')}.log"
        self.logger = logging.getLogger(f"PySide6画布性能.{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        formatter = logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", "%Y-%m-%d %H:%M:%S")
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.log_path = log_path
        self.draw_times: deque[float] = deque(maxlen=2000)
        self.paint_times: deque[float] = deque(maxlen=2000)
        self.stroke_draw_times: list[float] = []
        self.stroke_start = 0.0
        self.stroke_points = 0
        self.stroke_dirty_pixels = 0

    def start_stroke(self) -> None:
        self.stroke_start = time.perf_counter()
        self.stroke_draw_times = []
        self.stroke_points = 0
        self.stroke_dirty_pixels = 0

    def add_draw(self, elapsed_ms: float, dirty_rect: QRect) -> None:
        self.draw_times.append(elapsed_ms)
        self.stroke_draw_times.append(elapsed_ms)
        self.stroke_points += 1
        self.stroke_dirty_pixels += max(0, dirty_rect.width()) * max(0, dirty_rect.height())

    def add_paint(self, elapsed_ms: float) -> None:
        self.paint_times.append(elapsed_ms)

    def finish_stroke(self, tool: str, brush_size: int, dirty_refresh: bool) -> None:
        total_ms = (time.perf_counter() - self.stroke_start) * 1000.0
        draw_total = sum(self.stroke_draw_times)
        draw_avg = mean(self.stroke_draw_times) if self.stroke_draw_times else 0.0
        self.logger.info(
            "笔画结束｜工具=%s｜笔宽=%d｜刷新=%s｜输入点=%d｜绘制总计=%.3f毫秒｜"
            "绘制平均=%.3f毫秒｜笔画总计=%.3f毫秒｜累计脏区=%d像素",
            tool,
            brush_size,
            "脏矩形" if dirty_refresh else "全画布",
            self.stroke_points,
            draw_total,
            draw_avg,
            total_ms,
            self.stroke_dirty_pixels,
        )

    @staticmethod
    def summary(values: deque[float]) -> str:
        if not values:
            return "暂无数据"
        ordered = sorted(values)
        p95_index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
        return (
            f"平均 {mean(values):.2f} ms｜中位 {median(values):.2f} ms｜"
            f"P95 {ordered[p95_index]:.2f} ms｜最大 {max(values):.2f} ms"
        )


class PaintCanvas(QWidget):
    def __init__(self, recorder: PerformanceRecorder, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.recorder = recorder
        self.original_image = QImage()
        self.working_image = QImage()
        self.tool = "画笔"
        self.brush_size = 32
        self.dirty_refresh = True
        self.antialiasing = True
        self.drawing = False
        self.last_image_point = QPointF()
        self.scale = 1.0
        self.image_origin = QPointF()
        self.setMinimumSize(640, 640)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def load_image(self, path: Path) -> bool:
        started = time.perf_counter()
        image = QImage(str(path))
        if image.isNull():
            return False
        self.original_image = image.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
        self.working_image = self.original_image.copy()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.recorder.logger.info(
            "载入图片｜路径=%s｜尺寸=%dx%d｜转换QImage=%.3f毫秒",
            path,
            image.width(),
            image.height(),
            elapsed_ms,
        )
        self.update_geometry_cache()
        self.update()
        return True

    def reset_image(self) -> None:
        if self.original_image.isNull():
            return
        started = time.perf_counter()
        self.working_image = self.original_image.copy()
        self.recorder.logger.info("重置图片｜复制QImage=%.3f毫秒", (time.perf_counter() - started) * 1000.0)
        self.update()

    def update_geometry_cache(self) -> None:
        if self.working_image.isNull():
            return
        margin = 32
        available_w = max(1, self.width() - margin * 2)
        available_h = max(1, self.height() - margin * 2)
        self.scale = min(
            available_w / self.working_image.width(),
            available_h / self.working_image.height(),
        )
        shown_w = self.working_image.width() * self.scale
        shown_h = self.working_image.height() * self.scale
        self.image_origin = QPointF((self.width() - shown_w) / 2.0, (self.height() - shown_h) / 2.0)

    def image_rect_on_widget(self) -> QRectF:
        return QRectF(
            self.image_origin.x(),
            self.image_origin.y(),
            self.working_image.width() * self.scale,
            self.working_image.height() * self.scale,
        )

    def widget_to_image(self, point: QPointF) -> QPointF:
        return QPointF(
            (point.x() - self.image_origin.x()) / self.scale,
            (point.y() - self.image_origin.y()) / self.scale,
        )

    def image_to_widget_rect(self, rect: QRect) -> QRect:
        padding = 3
        return QRectF(
            self.image_origin.x() + rect.x() * self.scale - padding,
            self.image_origin.y() + rect.y() * self.scale - padding,
            rect.width() * self.scale + padding * 2,
            rect.height() * self.scale + padding * 2,
        ).toAlignedRect()

    def point_in_image(self, point: QPointF) -> bool:
        return 0 <= point.x() < self.working_image.width() and 0 <= point.y() < self.working_image.height()

    def draw_segment(self, start: QPointF, end: QPointF) -> None:
        started = time.perf_counter_ns()
        radius = self.brush_size / 2.0 + 3.0
        dirty = QRectF(start, end).normalized().adjusted(-radius, -radius, radius, radius).toAlignedRect()
        dirty = dirty.intersected(self.working_image.rect())

        painter = QPainter(self.working_image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, self.antialiasing)
        if self.tool == "橡皮擦":
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            color = QColor(0, 0, 0, 0)
        else:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            color = QColor(22, 22, 22, 255)
        pen = QPen(color, self.brush_size, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(start, end)
        if start == end:
            painter.drawPoint(start)
        painter.end()

        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        self.recorder.add_draw(elapsed_ms, dirty)
        if self.dirty_refresh:
            self.update(self.image_to_widget_rect(dirty))
        else:
            self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        started = time.perf_counter_ns()
        painter = QPainter(self)
        painter.fillRect(event.rect(), QColor("#17191d"))
        if not self.working_image.isNull():
            image_rect = self.image_rect_on_widget()
            painter.fillRect(image_rect, QColor("#f3f3f3"))
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
            painter.drawImage(image_rect, self.working_image)
            painter.setPen(QPen(QColor("#555b66"), 1))
            painter.drawRect(image_rect)
        painter.end()
        self.recorder.add_paint((time.perf_counter_ns() - started) / 1_000_000.0)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self.update_geometry_cache()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton or self.working_image.isNull():
            return
        point = self.widget_to_image(event.position())
        if not self.point_in_image(point):
            return
        self.drawing = True
        self.last_image_point = point
        self.recorder.start_stroke()
        self.draw_segment(point, point)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if not self.drawing:
            return
        point = self.widget_to_image(event.position())
        point.setX(min(max(point.x(), 0), self.working_image.width() - 1))
        point.setY(min(max(point.y(), 0), self.working_image.height() - 1))
        self.draw_segment(self.last_image_point, point)
        self.last_image_point = point

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton or not self.drawing:
            return
        self.drawing = False
        self.recorder.finish_stroke(self.tool, self.brush_size, self.dirty_refresh)


class TestWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.recorder = PerformanceRecorder()
        self.canvas = PaintCanvas(self.recorder)
        self.setWindowTitle("PySide6 画笔与橡皮擦性能测试（独立窗口）")
        self.resize(1180, 820)
        self.build_ui()
        if DEFAULT_IMAGE.exists():
            self.canvas.load_image(DEFAULT_IMAGE)
        self.stats_timer = QTimer(self)
        self.stats_timer.timeout.connect(self.refresh_stats)
        self.stats_timer.start(250)
        self.recorder.logger.info("测试窗口启动｜Qt版本=%s｜日志=%s", QApplication.instance().applicationVersion(), self.recorder.log_path)

    def build_ui(self) -> None:
        toolbar = QToolBar("测试工具", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        open_action = QAction("打开测试图片", self)
        open_action.triggered.connect(self.open_image)
        toolbar.addAction(open_action)

        reset_action = QAction("重置图片", self)
        reset_action.triggered.connect(self.canvas.reset_image)
        toolbar.addAction(reset_action)
        toolbar.addSeparator()

        brush_action = QAction("画笔", self)
        brush_action.triggered.connect(lambda: self.set_tool("画笔"))
        toolbar.addAction(brush_action)
        eraser_action = QAction("橡皮擦", self)
        eraser_action.triggered.connect(lambda: self.set_tool("橡皮擦"))
        toolbar.addAction(eraser_action)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(self.canvas, 1)

        panel = QWidget()
        panel.setFixedWidth(330)
        panel_layout = QVBoxLayout(panel)
        title = QLabel("实时性能数据")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        panel_layout.addWidget(title)

        self.tool_label = QLabel("当前工具：画笔")
        panel_layout.addWidget(self.tool_label)
        panel_layout.addWidget(QLabel("笔刷大小"))
        size_row = QHBoxLayout()
        size_slider = QSlider(Qt.Orientation.Horizontal)
        size_slider.setRange(1, 100)
        size_slider.setValue(32)
        size_spin = QSpinBox()
        size_spin.setRange(1, 100)
        size_spin.setValue(32)
        size_slider.valueChanged.connect(size_spin.setValue)
        size_spin.valueChanged.connect(size_slider.setValue)
        size_spin.valueChanged.connect(self.set_brush_size)
        size_row.addWidget(size_slider)
        size_row.addWidget(size_spin)
        panel_layout.addLayout(size_row)

        dirty_check = QCheckBox("仅刷新脏矩形（推荐）")
        dirty_check.setChecked(True)
        dirty_check.toggled.connect(self.set_dirty_refresh)
        panel_layout.addWidget(dirty_check)
        antialias_check = QCheckBox("抗锯齿")
        antialias_check.setChecked(True)
        antialias_check.toggled.connect(self.set_antialiasing)
        panel_layout.addWidget(antialias_check)

        self.draw_stats = QLabel("像素绘制：暂无数据")
        self.draw_stats.setWordWrap(True)
        self.paint_stats = QLabel("窗口重绘：暂无数据")
        self.paint_stats.setWordWrap(True)
        self.frame_hint = QLabel("帧预算：暂无数据")
        self.frame_hint.setWordWrap(True)
        panel_layout.addSpacing(14)
        panel_layout.addWidget(self.draw_stats)
        panel_layout.addWidget(self.paint_stats)
        panel_layout.addWidget(self.frame_hint)

        instructions = QLabel(
            "验证方法：\n"
            "1. 分别使用画笔和橡皮擦连续快速拖动。\n"
            "2. 对比勾选与取消“仅刷新脏矩形”。\n"
            "3. 观察动作是否即时显示及 P95 耗时。\n"
            "4. 每次启动会在 test\\测试日志 中生成独立日志。\n\n"
            "本窗口只编辑内存副本，不会保存或改动原图，也不会导入主程序代码。"
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #aeb4bf; line-height: 1.4;")
        panel_layout.addSpacing(18)
        panel_layout.addWidget(instructions)
        panel_layout.addStretch(1)

        open_log_button = QPushButton("显示日志位置")
        open_log_button.clicked.connect(self.show_log_path)
        panel_layout.addWidget(open_log_button)
        layout.addWidget(panel)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage(f"测试数据：{DEFAULT_IMAGE}")
        self.setStyleSheet(
            "QMainWindow, QWidget { background: #24272d; color: #e8eaf0; }"
            "QToolBar { background: #1d2025; spacing: 8px; padding: 7px; }"
            "QPushButton { background: #3b72d9; border: 0; border-radius: 5px; padding: 8px; }"
            "QCheckBox, QLabel { padding: 3px; }"
        )

    def set_tool(self, tool: str) -> None:
        self.canvas.tool = tool
        self.tool_label.setText(f"当前工具：{tool}")

    def set_brush_size(self, size: int) -> None:
        self.canvas.brush_size = size

    def set_dirty_refresh(self, enabled: bool) -> None:
        self.canvas.dirty_refresh = enabled
        self.recorder.logger.info("切换刷新方式｜%s", "脏矩形" if enabled else "全画布")

    def set_antialiasing(self, enabled: bool) -> None:
        self.canvas.antialiasing = enabled
        self.recorder.logger.info("切换抗锯齿｜%s", "开启" if enabled else "关闭")

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择测试图片（只读载入）",
            str(DATA_DIR),
            "图片文件 (*.png *.tif *.tiff *.jpg *.jpeg *.bmp)",
        )
        if path and not self.canvas.load_image(Path(path)):
            QMessageBox.warning(self, "载入失败", "无法读取所选图片。")

    def refresh_stats(self) -> None:
        self.draw_stats.setText(f"像素绘制：{self.recorder.summary(self.recorder.draw_times)}")
        self.paint_stats.setText(f"窗口重绘：{self.recorder.summary(self.recorder.paint_times)}")
        over_16 = sum(value > 16.67 for value in self.recorder.paint_times)
        total = len(self.recorder.paint_times)
        self.frame_hint.setText(f"超过 16.67 ms：{over_16}/{total} 帧")

    def show_log_path(self) -> None:
        QMessageBox.information(self, "日志位置", str(self.recorder.log_path))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("PySide6画布性能测试")
    app.setApplicationVersion("1.0")
    window = TestWindow()
    window.show()
    sys.exit(app.exec())
