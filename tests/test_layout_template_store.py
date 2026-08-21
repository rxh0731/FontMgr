"""通用经文排版模板存储回归测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.scripture_layout import (
    FLOW_LEFT_TO_RIGHT,
    FLOW_RIGHT_TO_LEFT,
    LAYOUT_HORIZONTAL,
    LAYOUT_VERTICAL,
    LayoutParameters,
)
from data.application_database import ApplicationDatabase
from data.layout_template_store import (
    DEFAULT_TEMPLATE_ID,
    DEFAULT_TEMPLATE_NAME,
    DEFAULT_TEMPLATE_PARAMETERS,
    TEMPLATE_DATA_VERSION,
    LayoutTemplateStore,
)


class LayoutTemplateStoreTests(unittest.TestCase):
    def test_legacy_configuration_is_imported_without_modifying_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            legacy_path = Path(directory) / "排版模板.json"
            target_path = Path(directory) / "通用经文排版模板.json"
            legacy_path.write_text(
                json.dumps(
                    {
                        "数据版本": 1,
                        "用户模板": {"旧文件中的模板": {"rows": 18}},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            migrated = LayoutTemplateStore(
                str(target_path),
                legacy_file_path=str(legacy_path),
            )

            self.assertFalse(target_path.exists())
            self.assertTrue(legacy_path.is_file())
            self.assertTrue(target_path.with_suffix(".sqlite3").is_file())
            self.assertEqual(migrated.get("旧文件中的模板").parameters.rows, 18)

    def test_missing_configuration_rebuilds_hardcoded_default_template(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "排版模板.json"

            store = LayoutTemplateStore(str(path))

            self.assertFalse(path.exists())
            self.assertEqual(
                store.get(DEFAULT_TEMPLATE_ID).parameters,
                DEFAULT_TEMPLATE_PARAMETERS,
            )
            stored = ApplicationDatabase(str(path.with_suffix(".sqlite3"))).read_document(
                "通用经文排版模板"
            )
            self.assertEqual(stored["数据版本"], TEMPLATE_DATA_VERSION)
            self.assertIn(DEFAULT_TEMPLATE_ID, stored["用户模板"])

    def test_corrupted_configuration_is_archived_and_rebuilt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "排版模板.json"
            path.write_text("{损坏", encoding="utf-8")

            store = LayoutTemplateStore(str(path))

            self.assertEqual(
                store.get(DEFAULT_TEMPLATE_NAME).parameters,
                DEFAULT_TEMPLATE_PARAMETERS,
            )
            self.assertEqual(path.read_text(encoding="utf-8"), "{损坏")
            rebuilt = ApplicationDatabase(str(path.with_suffix(".sqlite3"))).read_document(
                "通用经文排版模板"
            )
            self.assertEqual(rebuilt["数据版本"], TEMPLATE_DATA_VERSION)
            self.assertEqual(len(list(Path(directory).glob("排版模板.json.损坏-*"))), 0)

    def test_file_cannot_override_hardcoded_default_template(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "排版模板.json"
            changed = DEFAULT_TEMPLATE_PARAMETERS.to_dict()
            changed["dpi"] = 72
            path.write_text(
                json.dumps(
                    {
                        "数据版本": TEMPLATE_DATA_VERSION,
                        "用户模板": {
                            DEFAULT_TEMPLATE_ID: {
                                "名称": DEFAULT_TEMPLATE_NAME,
                                "说明": "外部修改",
                                "创建时间": "",
                                "修改时间": "",
                                "参数": changed,
                            }
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            store = LayoutTemplateStore(str(path))

            self.assertEqual(store.get(DEFAULT_TEMPLATE_ID).parameters.dpi, 300)
            self.assertEqual(
                store.get(DEFAULT_TEMPLATE_ID).parameters,
                DEFAULT_TEMPLATE_PARAMETERS,
            )

    def test_user_template_round_trip_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "排版模板.json"
            store = LayoutTemplateStore(str(path))
            parameters = LayoutParameters(
                rows=18,
                columns=24,
                dpi=300,
                layout_mode=LAYOUT_HORIZONTAL,
                flow_direction=FLOW_LEFT_TO_RIGHT,
            )

            store.save("十八行", parameters)
            reloaded = LayoutTemplateStore(str(path))
            self.assertEqual(reloaded.get("十八行").parameters, parameters)
            self.assertTrue(reloaded.get(DEFAULT_TEMPLATE_NAME).builtin)

            reloaded.delete("十八行")
            with self.assertRaises(KeyError):
                reloaded.get("十八行")

    def test_builtin_template_cannot_be_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LayoutTemplateStore(str(Path(directory) / "排版模板.json"))
            with self.assertRaises(ValueError):
                store.save(DEFAULT_TEMPLATE_NAME, LayoutParameters())

    def test_invalid_version_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "排版模板.json"
            path.write_text(
                json.dumps({"数据版本": 999, "用户模板": {}}, ensure_ascii=False),
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                LayoutTemplateStore(str(path))

    def test_legacy_template_defaults_to_vertical_right_to_left(self) -> None:
        parameters = LayoutParameters.from_dict({"rows": 18, "columns": 24})

        self.assertEqual(parameters.layout_mode, LAYOUT_VERTICAL)
        self.assertEqual(parameters.flow_direction, FLOW_RIGHT_TO_LEFT)

    def test_version_one_templates_migrate_on_next_user_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "排版模板.json"
            path.write_text(
                json.dumps(
                    {
                        "数据版本": 1,
                        "用户模板": {
                            "旧模板": {"rows": 19, "columns": 23},
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            store = LayoutTemplateStore(str(path))
            legacy = store.get("旧模板")
            self.assertEqual(legacy.parameters.rows, 19)
            self.assertFalse(legacy.parameters.include_punctuation)

            store.update(legacy.template_id, legacy.parameters)
            migrated = ApplicationDatabase(str(path.with_suffix(".sqlite3"))).read_document(
                "通用经文排版模板"
            )
            self.assertEqual(migrated["数据版本"], TEMPLATE_DATA_VERSION)
            self.assertIn(legacy.template_id, migrated["用户模板"])
            self.assertEqual(
                migrated["用户模板"][legacy.template_id]["名称"],
                "旧模板",
            )

    def test_template_id_survives_name_and_description_edits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "排版模板.json"
            store = LayoutTemplateStore(str(path))
            created = store.save("竖排模板", LayoutParameters(rows=20), "初始说明")

            updated = store.update_details(
                created.template_id,
                name="大字竖排",
                description="适用于大字版面",
            )

            self.assertEqual(updated.template_id, created.template_id)
            self.assertEqual(updated.name, "大字竖排")
            self.assertEqual(updated.description, "适用于大字版面")
            self.assertEqual(
                LayoutTemplateStore(str(path)).get(created.template_id).name,
                "大字竖排",
            )

    def test_template_export_import_preserves_layout_rules(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "源模板.json"
            export_path = Path(directory) / "交换模板.json"
            target_path = Path(directory) / "目标模板.json"
            source = LayoutTemplateStore(str(source_path))
            parameters = LayoutParameters(
                dpi=300,
                rows=18,
                columns=24,
                include_punctuation=True,
                layout_mode=LAYOUT_HORIZONTAL,
                flow_direction=FLOW_LEFT_TO_RIGHT,
            )
            created = source.save("横排标点", parameters, "保留标点的横排模板")
            source.export_template(created.template_id, str(export_path))

            target = LayoutTemplateStore(str(target_path))
            imported = target.import_template(str(export_path))

            self.assertEqual(imported.parameters, parameters)
            self.assertEqual(imported.description, "保留标点的横排模板")
            self.assertNotEqual(imported.template_id, created.template_id)


if __name__ == "__main__":
    unittest.main()
