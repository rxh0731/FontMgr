from __future__ import annotations

import unittest

from utils.batch_observability import (
    BatchTiming,
    ProgressThrottle,
    format_elapsed_time,
)


class _FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class ProgressThrottleTests(unittest.TestCase):
    def test_rate_limits_intermediate_updates_and_forces_terminal_update(self) -> None:
        clock = _FakeClock()
        observed: list[tuple[str, int]] = []
        progress = ProgressThrottle(
            lambda message, current: observed.append((str(message), int(current))),
            interval_seconds=0.1,
            clock=clock,
        )

        self.assertTrue(progress.emit("开始", 0, stage="处理"))
        clock.advance(0.02)
        self.assertFalse(progress.emit("中间一", 1, stage="处理"))
        clock.advance(0.03)
        self.assertFalse(progress.emit("中间二", 2, stage="处理"))
        clock.advance(0.05)
        self.assertTrue(progress.emit("定时进度", 3, stage="处理"))
        clock.advance(0.01)
        self.assertTrue(progress.emit("已停止", 3, force=True, stage="停止"))

        self.assertEqual(
            observed,
            [("开始", 0), ("定时进度", 3), ("已停止", 3)],
        )

    def test_stage_change_and_flush_deliver_key_updates(self) -> None:
        clock = _FakeClock()
        observed: list[str] = []
        progress = ProgressThrottle(
            observed.append,
            interval_seconds=0.1,
            clock=clock,
        )

        self.assertTrue(progress.emit("准备开始", stage="准备"))
        clock.advance(0.01)
        self.assertFalse(progress.emit("准备中", stage="准备"))
        self.assertTrue(progress.emit("渲染开始", stage="渲染"))
        clock.advance(0.01)
        self.assertFalse(progress.emit("渲染中一", stage="渲染"))
        self.assertFalse(progress.emit("渲染中二", stage="渲染"))
        self.assertTrue(progress.flush())
        self.assertFalse(progress.flush())

        self.assertEqual(
            observed,
            ["准备开始", "渲染开始", "渲染中二"],
        )

    def test_sustained_same_stage_updates_stay_near_ten_per_second(self) -> None:
        clock = _FakeClock()
        emitted_at: list[float] = []
        progress = ProgressThrottle(
            lambda _value: emitted_at.append(clock()),
            interval_seconds=0.1,
            clock=clock,
        )

        for value in range(100):
            progress.emit(value, stage="处理")
            clock.advance(0.01)

        self.assertLessEqual(len(emitted_at), 11)
        self.assertTrue(
            all(
                later - earlier >= 0.1 - 1e-9
                for earlier, later in zip(emitted_at, emitted_at[1:])
            )
        )

    def test_rejects_invalid_interval(self) -> None:
        for interval in (0.0, -0.1, float("inf"), float("nan")):
            with self.subTest(interval=interval):
                with self.assertRaises(ValueError):
                    ProgressThrottle(lambda: None, interval_seconds=interval)


class BatchTimingTests(unittest.TestCase):
    def test_formats_elapsed_time_for_user_feedback(self) -> None:
        self.assertEqual(format_elapsed_time(0.346), "0.35 秒")
        self.assertEqual(format_elapsed_time(83.45), "1 分 23.45 秒")
        self.assertEqual(
            format_elapsed_time(3723.45),
            "1 小时 02 分 03.45 秒",
        )
        self.assertEqual(format_elapsed_time(3599.999), "1 小时 00 分 00.00 秒")
        for invalid in (-1.0, float("inf"), float("nan")):
            with self.subTest(invalid=invalid):
                self.assertEqual(format_elapsed_time(invalid), "0.00 秒")

    def test_accumulates_stages_and_formats_stable_chinese_summary(self) -> None:
        clock = _FakeClock()
        timing = BatchTiming(clock=clock)

        with timing.measure("候选生成"):
            clock.advance(0.2)
        with timing.measure("候选生成"):
            clock.advance(0.1)
        timing.add("保存", 0.05)
        clock.advance(0.2)

        summary = timing.format_summary(
            "自动优化",
            {"处理": 8, "跳过": 2, "失败": 1},
            stopped=True,
        )

        self.assertAlmostEqual(timing.total_seconds, 0.5)
        stage_seconds = timing.stage_seconds
        self.assertAlmostEqual(stage_seconds["候选生成"], 0.3)
        self.assertAlmostEqual(stage_seconds["保存"], 0.05)
        self.assertIn("任务=自动优化", summary)
        self.assertIn("状态=已停止", summary)
        self.assertIn("总耗时=0.5000秒", summary)
        self.assertIn("处理=8｜跳过=2｜失败=1", summary)
        self.assertIn("阶段耗时=候选生成=0.3000秒、保存=0.0500秒", summary)

        clock.advance(10.0)
        self.assertAlmostEqual(timing.finish(), 0.5)

    def test_measure_keeps_elapsed_time_when_stage_raises(self) -> None:
        clock = _FakeClock()
        timing = BatchTiming(clock=clock)

        with self.assertRaisesRegex(RuntimeError, "模拟失败"):
            with timing.measure("保存"):
                clock.advance(0.125)
                raise RuntimeError("模拟失败")

        self.assertEqual(timing.stage_seconds, {"保存": 0.125})

    def test_rejects_invalid_stage_values(self) -> None:
        timing = BatchTiming(clock=_FakeClock())
        with self.assertRaises(ValueError):
            timing.add("", 0.1)
        for elapsed in (-0.1, float("inf"), float("nan")):
            with self.subTest(elapsed=elapsed):
                with self.assertRaises(ValueError):
                    timing.add("处理", elapsed)


if __name__ == "__main__":
    unittest.main()
