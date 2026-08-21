"""定制经文排版模板存储回归测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.custom_scripture_layout import (
    CustomBoardParameters,
    CustomLayoutTemplateParameters,
)
from data.application_database import ApplicationDatabase
from data.custom_layout_template_store import (
    DEFAULT_CUSTOM_TEMPLATE_ID,
    CustomLayoutTemplateStore,
)


class CustomLayoutTemplateStoreTests(unittest.TestCase):
    def test_missing_file_is_rebuilt_with_hardcoded_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "定制经文排版模板.json"

            store = CustomLayoutTemplateStore(str(path))

            self.assertFalse(path.exists())
            default = store.get(DEFAULT_CUSTOM_TEMPLATE_ID)
            self.assertTrue(default.builtin)
            self.assertEqual(len(default.parameters.boards), 1)
            payload = ApplicationDatabase(str(path.with_suffix(".sqlite3"))).read_document(
                "定制经文排版模板"
            )
            self.assertEqual(payload["数据版本"], 1)

    def test_multi_board_template_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "定制经文排版模板.json"
            store = CustomLayoutTemplateStore(str(path))
            parameters = CustomLayoutTemplateParameters(
                boards=(
                    CustomBoardParameters(base_column_characters=21),
                    CustomBoardParameters(base_column_characters=18, dpi=600),
                ),
                include_punctuation=True,
                add_annotations=True,
            )

            saved = store.save("双版模板", parameters, "两块版面")
            reloaded = CustomLayoutTemplateStore(str(path)).get(saved.template_id)

            self.assertEqual(reloaded.parameters, parameters)
            self.assertEqual(reloaded.description, "两块版面")


if __name__ == "__main__":
    unittest.main()
