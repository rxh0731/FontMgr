"""首页实时摘要和持久索引回归测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import config
from data.library_summary_store import LibrarySummaryStore
from services.library_summary_service import build_library_summary


class LibrarySummaryStoreTests(unittest.TestCase):
    def test_store_only_loads_matching_filesystem_signature(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "字库"
            root.mkdir()
            store = LibrarySummaryStore(
                str(Path(temp_dir) / "摘要.json"),
                str(root),
            )
            signature = {"根目录": str(root), "修订": 1}
            summaries = [
                {
                    "name": "测试字库",
                    "path": str(root / "测试字库"),
                    "variants": 12,
                }
            ]
            (root / "测试字库").mkdir()

            store.save(summaries, signature)

            self.assertEqual(store.load(signature), summaries)
            self.assertIsNone(store.load({"根目录": str(root), "修订": 2}))

    def test_trusted_summary_uses_committed_records_without_file_scan(self) -> None:
        coordination_summary = {
            "墨色统一启用": True,
            "墨色方法": "视觉墨量规范化",
            "墨色方法版本": 2,
            "墨色基准": 215.0,
        }
        valid_ink = {
            "启用": True,
            "方法": "视觉墨量规范化",
            "方法版本": 2,
            "基准": 215.0,
            "保存后复测": True,
            "保存后墨色": 214.8,
            "是否达标": True,
        }
        details = {
            "pending": {
                "状态": config.STATUS_PENDING_OPTIMIZATION,
                "原始文件": "待优化.tif",
            },
            "optimized": {
                "状态": config.STATUS_PENDING_MANUAL_REVIEW,
                "中间文件": "已优化.png",
            },
            "reviewed": {
                "状态": config.STATUS_REVIEWED,
                "中间文件": "已审核.png",
            },
            "finished": {
                "状态": config.STATUS_FINISHED,
                "成品文件": "已协调.png",
                "整体协调参数": {"墨色协调": valid_ink},
            },
        }

        summary = build_library_summary(
            "测试字库",
            r"D:\不存在\测试字库",
            details,
            {"字": list(details)},
            {"DPI": 300, "画布宽": 250, "画布高": 250},
            coordination_summary,
            verify_files=False,
        )

        self.assertEqual(summary["variants"], 4)
        self.assertEqual(summary["pending_optimization"], 1)
        self.assertEqual(summary["optimized"], 3)
        self.assertEqual(summary["review_admitted"], 3)
        self.assertEqual(summary["pending_review"], 1)
        self.assertEqual(summary["reviewed"], 2)
        self.assertEqual(summary["coordination_admitted"], 2)
        self.assertEqual(summary["pending_coordination"], 1)
        self.assertEqual(summary["coordinated"], 1)
        self.assertEqual(summary["export_ready"], 1)


if __name__ == "__main__":
    unittest.main()
