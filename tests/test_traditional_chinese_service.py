"""OpenCC 简繁候选识别回归测试。"""

from __future__ import annotations

import gc
import unittest
import warnings
from unittest.mock import patch

import services.traditional_chinese_service as service


class TraditionalChineseServiceTests(unittest.TestCase):
    """验证三套 OpenCC 规则与项目歧义表会保守合并。"""

    def test_three_profiles_and_project_candidates_are_merged_with_sources(self) -> None:
        result = service.identify_character_with_sources("台")

        self.assertEqual(result.category, "歧义")
        self.assertEqual(len(result.candidates), len(set(result.candidates)))
        self.assertIn("臺", result.candidates)
        self.assertIn("檯", result.candidates)
        self.assertIn("颱", result.candidates)
        self.assertIn("台", result.candidates)
        self.assertIn("项目歧义表", result.candidate_sources["臺"])
        self.assertIn("s2t（通用繁体）", result.candidate_sources["臺"])
        self.assertIn("s2tw（台湾用字）", result.candidate_sources["臺"])
        self.assertIn("s2hk（香港用字）", result.candidate_sources["台"])
        self.assertIn("保留原字", result.candidate_sources["台"])

    def test_project_ambiguity_table_keeps_candidate_opencc_does_not_return(self) -> None:
        result = service.identify_character_with_sources("发")

        self.assertEqual(result.category, "歧义")
        self.assertIn("發", result.candidates)
        self.assertIn("髮", result.candidates)
        self.assertIn("发", result.candidates)
        self.assertEqual(result.candidate_sources["髮"], ("项目歧义表",))

    def test_known_opencc_difference_still_enters_manual_review(self) -> None:
        result = service.identify_character_with_sources("喂")

        self.assertEqual(result.category, "歧义")
        self.assertIn("餵", result.candidates)
        self.assertIn("喂", result.candidates)

    def test_legacy_two_value_api_remains_available(self) -> None:
        category, candidates = service.identify_character("爱")

        self.assertEqual(category, "一对一")
        self.assertEqual(candidates[0], "愛")
        self.assertIn("爱", candidates)

    def test_missing_opencc_fails_explicitly(self) -> None:
        with patch.object(service, "OpenCC", None):
            with self.assertRaisesRegex(RuntimeError, "缺少 OpenCC"):
                service._get_converters()

    def test_opencc_conversion_emits_no_resource_warning(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ResourceWarning)
            for _index in range(100):
                service.identify_character_with_sources("台")
            gc.collect()


if __name__ == "__main__":
    unittest.main()
