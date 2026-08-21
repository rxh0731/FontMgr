"""首页摘要快速更新回归测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import config
import services.library_summary_service as summary_module
from services.glyph_service import GlyphService
from services.library_summary_service import summarize_glyph_service


class LibrarySummaryServiceTests(unittest.TestCase):
    def test_only_changed_variant_rebuilds_cached_contribution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph = GlyphService("摘要测试", directory)
            glyph.ensure_dirs()
            glyph.init_metadata(dpi=300, canvas_w=20, canvas_h=20)
            first_id = glyph.add_original(
                "甲", "甲-0001.png", "甲.png", "md5-a"
            )
            glyph.add_original("乙", "乙-0001.png", "乙.png", "md5-b")

            with patch.object(
                summary_module,
                "_summary_contribution",
                wraps=summary_module._summary_contribution,
            ) as contribution:
                initial = summarize_glyph_service(glyph)
                self.assertEqual(contribution.call_count, 2)
                contribution.reset_mock()

                glyph.update_variant(
                    first_id,
                    状态=config.STATUS_PENDING_MANUAL_REVIEW,
                    中间文件="甲-0001.png",
                )
                updated = summarize_glyph_service(glyph)

            self.assertEqual(contribution.call_count, 1)
            self.assertEqual(initial["optimized"], 0)
            self.assertEqual(updated["optimized"], 1)
            self.assertEqual(updated["review_admitted"], 1)

    def test_coordination_count_refresh_does_not_invalidate_all_variants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glyph = GlyphService("摘要测试", directory)
            glyph.ensure_dirs()
            glyph.init_metadata(dpi=300, canvas_w=20, canvas_h=20)
            glyph.add_original("甲", "甲-0001.png", "甲.png", "md5-a")
            glyph.add_original("乙", "乙-0001.png", "乙.png", "md5-b")
            summarize_glyph_service(glyph)
            summary = glyph.get_coordination_summary()
            summary["墨色统计"] = {
                "总数": 2,
                "已达标": 1,
                "待确认": 1,
                "人工例外": 0,
            }
            glyph._data["整体协调"] = summary

            with patch.object(
                summary_module,
                "_summary_contribution",
                wraps=summary_module._summary_contribution,
            ) as contribution:
                summarize_glyph_service(glyph)

            contribution.assert_not_called()


if __name__ == "__main__":
    unittest.main()
