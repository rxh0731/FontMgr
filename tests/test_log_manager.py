"""日志管理器的线程内批量写入回归测试。"""

from __future__ import annotations

import threading
import unittest
from unittest.mock import MagicMock, patch

from data import log_manager


class LogManagerBufferingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = log_manager.LogManager()
        self.handle = MagicMock()
        self.manager._handle = self.handle
        self.active_manager = patch.object(
            log_manager,
            "_active_manager",
            self.manager,
        )
        self.active_manager.start()

    def tearDown(self) -> None:
        self.active_manager.stop()
        self.manager._handle = None

    def test_buffer_reduces_write_and_flush_count(self) -> None:
        with log_manager.buffered_log_writes():
            for index in range(200):
                log_manager.write_log(f"批量日志 {index}")

        self.handle.write.assert_called_once()
        self.handle.flush.assert_called_once()
        block = self.handle.write.call_args.args[0]
        self.assertIn("批量日志 0", block)
        self.assertIn("批量日志 199", block)
        self.assertEqual(block.count("批量日志"), 200)

    def test_threshold_flushes_in_blocks(self) -> None:
        message = "批量内容" * 512
        with log_manager.buffered_log_writes():
            for _index in range(80):
                log_manager.write_log(message)

        self.assertGreater(self.handle.write.call_count, 1)
        self.assertLess(self.handle.write.call_count, 80)
        self.assertEqual(self.handle.flush.call_count, self.handle.write.call_count)

    def test_context_exit_flushes_when_body_raises(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "模拟异常"):
            with log_manager.buffered_log_writes():
                log_manager.write_log("异常前仍需写入")
                raise RuntimeError("模拟异常")

        self.handle.write.assert_called_once()
        self.handle.flush.assert_called_once()
        self.assertIn("异常前仍需写入", self.handle.write.call_args.args[0])

    def test_nested_context_only_flushes_at_outer_exit(self) -> None:
        with log_manager.buffered_log_writes():
            log_manager.write_log("外层开始")
            with log_manager.buffered_log_writes():
                log_manager.write_log("内层")
            self.handle.write.assert_not_called()
            log_manager.write_log("外层结束")

        self.handle.write.assert_called_once()
        block = self.handle.write.call_args.args[0]
        self.assertLess(block.index("外层开始"), block.index("内层"))
        self.assertLess(block.index("内层"), block.index("外层结束"))

    def test_other_thread_keeps_immediate_write_behavior(self) -> None:
        worker_finished = threading.Event()

        def worker() -> None:
            log_manager.write_log("工作线程即时日志")
            worker_finished.set()

        with log_manager.buffered_log_writes():
            log_manager.write_log("主线程缓冲日志")
            thread = threading.Thread(target=worker)
            thread.start()
            self.assertTrue(worker_finished.wait(timeout=2.0))
            thread.join(timeout=2.0)
            self.assertFalse(thread.is_alive())
            self.handle.write.assert_called_once()
            self.assertIn("工作线程即时日志", self.handle.write.call_args.args[0])

        self.assertEqual(self.handle.write.call_count, 2)
        written_blocks = [call.args[0] for call in self.handle.write.call_args_list]
        self.assertTrue(any("主线程缓冲日志" in block for block in written_blocks))

    def test_existing_immediate_write_api_is_unchanged(self) -> None:
        log_manager.write_log("即时日志")

        self.handle.write.assert_called_once()
        self.handle.flush.assert_called_once()
        self.assertIn("即时日志", self.handle.write.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
