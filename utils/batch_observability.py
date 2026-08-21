"""批处理进度节流与阶段耗时汇总。"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager


def format_elapsed_time(seconds: float) -> str:
    """将耗时格式化为适合用户反馈的简体中文文本。"""
    elapsed = float(seconds)
    if not math.isfinite(elapsed) or elapsed < 0.0:
        elapsed = 0.0
    elapsed = round(elapsed, 2)
    hours, remainder = divmod(elapsed, 3600.0)
    minutes, remaining_seconds = divmod(remainder, 60.0)
    if hours >= 1.0:
        return f"{int(hours)} 小时 {int(minutes):02d} 分 {remaining_seconds:05.2f} 秒"
    if minutes >= 1.0:
        return f"{int(minutes)} 分 {remaining_seconds:05.2f} 秒"
    return f"{remaining_seconds:.2f} 秒"


class ProgressThrottle:
    """限制高频进度通知，同时允许关键状态立即送达。"""

    def __init__(
        self,
        callback: Callable[..., None],
        *,
        interval_seconds: float = 0.1,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        interval = float(interval_seconds)
        if not math.isfinite(interval) or interval <= 0.0:
            raise ValueError("进度节流间隔必须是大于零的有限数。")
        self._callback = callback
        self._interval_seconds = interval
        self._clock = clock
        self._last_emit_at: float | None = None
        self._last_stage: str | None = None
        self._pending: tuple[tuple[object, ...], str | None] | None = None

    def emit(
        self,
        *args: object,
        force: bool = False,
        stage: str | None = None,
    ) -> bool:
        """按节流规则通知进度；返回本次是否实际调用了回调。"""
        now = float(self._clock())
        stage_changed = stage is not None and stage != self._last_stage
        elapsed = (
            math.inf
            if self._last_emit_at is None
            else max(0.0, now - self._last_emit_at)
        )
        should_emit = (
            force
            or self._last_emit_at is None
            or stage_changed
            or elapsed >= self._interval_seconds
        )
        if not should_emit:
            self._pending = (args, stage)
            return False

        self._callback(*args)
        self._last_emit_at = now
        if stage is not None:
            self._last_stage = stage
        self._pending = None
        return True

    def flush(self) -> bool:
        """立即发送最近一条被节流的进度，适合正常结束或停止前调用。"""
        pending = self._pending
        if pending is None:
            return False
        args, stage = pending
        return self.emit(*args, force=True, stage=stage)


class BatchTiming:
    """累计批处理总耗时及若干可选阶段耗时。"""

    def __init__(self, *, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._started_at = float(clock())
        self._finished_at: float | None = None
        self._stage_seconds: dict[str, float] = {}

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        """累计一个代码块的耗时；代码块抛错时仍保留本段计时。"""
        started_at = float(self._clock())
        try:
            yield
        finally:
            self.add(stage, max(0.0, float(self._clock()) - started_at))

    def add(self, stage: str, seconds: float) -> None:
        """加入调用方已经测得的阶段耗时。"""
        name = str(stage).strip()
        elapsed = float(seconds)
        if not name:
            raise ValueError("阶段名称不能为空。")
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("阶段耗时必须是非负有限数。")
        self._stage_seconds[name] = self._stage_seconds.get(name, 0.0) + elapsed

    def finish(self) -> float:
        """冻结并返回总耗时；重复调用不会改变结果。"""
        if self._finished_at is None:
            self._finished_at = float(self._clock())
        return self.total_seconds

    @property
    def total_seconds(self) -> float:
        ended_at = (
            float(self._clock())
            if self._finished_at is None
            else self._finished_at
        )
        return max(0.0, ended_at - self._started_at)

    @property
    def stage_seconds(self) -> dict[str, float]:
        return dict(self._stage_seconds)

    def format_summary(
        self,
        task_name: str,
        counters: Mapping[str, int],
        *,
        stopped: bool = False,
    ) -> str:
        """生成适合写入程序日志的单行中文汇总。"""
        total = self.finish()
        name = str(task_name).strip() or "未命名批处理"
        sections = [
            "批处理耗时汇总",
            f"任务={name}",
            f"状态={'已停止' if stopped else '完成'}",
            f"总耗时={total:.4f}秒",
        ]
        for label, value in counters.items():
            sections.append(f"{str(label).strip()}={max(0, int(value))}")
        if self._stage_seconds:
            stage_text = "、".join(
                f"{stage}={seconds:.4f}秒"
                for stage, seconds in self._stage_seconds.items()
            )
            sections.append(f"阶段耗时={stage_text}")
        return "｜".join(sections)
