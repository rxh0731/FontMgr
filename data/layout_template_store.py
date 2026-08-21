"""通用经文排版模板的版本化、原子化存储。"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Mapping
from uuid import NAMESPACE_URL, uuid4, uuid5

from core.scripture_layout import LayoutParameters
from data.application_database import (
    ApplicationDatabase,
    resolve_application_database_path,
)
from utils.file_utils import atomic_write_json, safe_read_json, safe_read_json_with_source


TEMPLATE_DATA_VERSION = 2
TEMPLATE_EXPORT_VERSION = 1
TEMPLATE_EXPORT_KIND = "FontEditor-PySide6 经文排版模板"
DEFAULT_TEMPLATE_ID = "builtin:default"
DEFAULT_TEMPLATE_NAME = "默认模板"
DEFAULT_TEMPLATE_PARAMETERS = LayoutParameters(
    dpi=300,
    cell_width_mm=21.17,
    cell_height_mm=21.17,
    rows=20,
    columns=20,
    row_gap_mm=2.0,
    column_gap_mm=2.0,
    draw_outer_frame=True,
    frame_top_mm=2.0,
    frame_bottom_mm=2.0,
    frame_left_mm=2.0,
    frame_right_mm=2.0,
    canvas_top_mm=8.0,
    canvas_bottom_mm=8.0,
    canvas_left_mm=8.0,
    canvas_right_mm=8.0,
    include_punctuation=False,
    trim_empty_columns=True,
    layout_mode="竖排",
    flow_direction="从右到左",
    scale_mode="按源图尺寸",
    scale_percent=100,
    cell_fill_percent=95,
    auto_scale_enabled=False,
    auto_enlarge_threshold=75,
    auto_enlarge_fill_percent=95,
    auto_shrink_threshold=150,
    auto_shrink_fill_percent=95,
    paragraph_mode="段后换列",
    paragraph_skip_cells=2,
    first_title_new_column=True,
    last_title_new_column=True,
    add_annotations=False,
    special_gaps_enabled=False,
)


def _current_time() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class LayoutTemplate:
    template_id: str
    name: str
    parameters: LayoutParameters
    builtin: bool = False
    description: str = ""
    created_at: str = ""
    updated_at: str = ""


class LayoutTemplateStore:
    """管理只读内置模板、用户模板及单模板交换文件。"""

    def __init__(self, file_path: str, *, legacy_file_path: str = "") -> None:
        self.file_path = file_path
        self.legacy_file_path = legacy_file_path
        self._database = ApplicationDatabase(
            resolve_application_database_path(file_path)
        )
        self._load_source = ""
        self._needs_repair = False
        self._user_templates = self._load()
        if self._needs_repair:
            self._repair_configuration()

    def _migrate_legacy_configuration(self) -> None:
        """旧入口保留为空操作；配置文件只读导入，不再移动。"""

    def _load(self) -> dict[str, LayoutTemplate]:
        raw = self._database.read_document("通用经文排版模板")
        source = "数据库"
        if raw is None:
            raw, source = safe_read_json_with_source(self.file_path, {})
            if not raw and self.legacy_file_path:
                raw, source = safe_read_json_with_source(self.legacy_file_path, {})
            self._needs_repair = True
        self._load_source = source
        if not raw:
            self._needs_repair = True
            return {}
        if not isinstance(raw, Mapping):
            raise RuntimeError("排版模板文件格式无效。")
        version = raw.get("数据版本")
        templates = raw.get("用户模板", {})
        if not isinstance(templates, Mapping):
            raise RuntimeError("排版模板列表格式无效。")
        if version == 1:
            result = self._load_legacy_templates(templates)
            if os.path.abspath(source) != os.path.abspath(self.file_path):
                self._needs_repair = True
            return result
        if version != TEMPLATE_DATA_VERSION:
            raise RuntimeError("排版模板数据版本不受支持。")
        result = self._load_current_templates(templates)
        if source != "数据库":
            self._needs_repair = True
        return result

    def _load_legacy_templates(
        self,
        templates: Mapping[object, object],
    ) -> dict[str, LayoutTemplate]:
        result: dict[str, LayoutTemplate] = {}
        for raw_name, values in templates.items():
            name = self.validate_name(raw_name)
            if name == DEFAULT_TEMPLATE_NAME:
                continue
            if not isinstance(values, Mapping):
                raise RuntimeError(f"模板“{name}”参数格式无效。")
            template_id = uuid5(
                NAMESPACE_URL,
                f"fonteditor-layout-template:{name}",
            ).hex
            result[template_id] = LayoutTemplate(
                template_id=template_id,
                name=name,
                parameters=LayoutParameters.from_dict(values),
            )
        return result

    def _load_current_templates(
        self,
        templates: Mapping[object, object],
    ) -> dict[str, LayoutTemplate]:
        result: dict[str, LayoutTemplate] = {}
        names: set[str] = set()
        for raw_id, values in templates.items():
            template_id = str(raw_id or "").strip()
            if not template_id:
                raise RuntimeError("排版模板标识无效。")
            if not isinstance(values, Mapping):
                raise RuntimeError("排版模板记录格式无效。")
            name = self.validate_name(values.get("名称"))
            if template_id == DEFAULT_TEMPLATE_ID or name == DEFAULT_TEMPLATE_NAME:
                continue
            if name in names:
                raise RuntimeError(f"模板名称重复：{name}")
            raw_parameters = values.get("参数")
            if not isinstance(raw_parameters, Mapping):
                raise RuntimeError(f"模板“{name}”参数格式无效。")
            names.add(name)
            result[template_id] = LayoutTemplate(
                template_id=template_id,
                name=name,
                parameters=LayoutParameters.from_dict(raw_parameters),
                description=self.validate_description(values.get("说明", "")),
                created_at=str(values.get("创建时间") or ""),
                updated_at=str(values.get("修改时间") or ""),
            )
        return result

    @staticmethod
    def validate_name(raw_name: object) -> str:
        name = str(raw_name or "").strip()
        if not name:
            raise ValueError("模板名称不能为空。")
        if len(name) > 40:
            raise ValueError("模板名称不能超过 40 个字符。")
        if any(character in name for character in "\\/:*?\"<>|"):
            raise ValueError("模板名称包含不允许的字符。")
        return name

    @staticmethod
    def validate_description(raw_description: object) -> str:
        description = str(raw_description or "").strip()
        if len(description) > 500:
            raise ValueError("模板说明不能超过 500 个字符。")
        return description

    @staticmethod
    def _builtin_template() -> LayoutTemplate:
        return LayoutTemplate(
            template_id=DEFAULT_TEMPLATE_ID,
            name=DEFAULT_TEMPLATE_NAME,
            parameters=DEFAULT_TEMPLATE_PARAMETERS,
            builtin=True,
            description="程序内置的托底排版参数，不能覆盖、改名或删除。",
        )

    def list_templates(self) -> tuple[LayoutTemplate, ...]:
        templates = [self._builtin_template()]
        templates.extend(
            sorted(self._user_templates.values(), key=lambda item: item.name)
        )
        return tuple(templates)

    def get(self, identity: str) -> LayoutTemplate:
        if identity in {DEFAULT_TEMPLATE_ID, DEFAULT_TEMPLATE_NAME}:
            return self._builtin_template()
        if identity in self._user_templates:
            return self._user_templates[identity]
        for template in self._user_templates.values():
            if template.name == identity:
                return template
        raise KeyError(f"找不到排版模板：{identity}")

    def find_by_name(self, name: str) -> LayoutTemplate | None:
        if name == DEFAULT_TEMPLATE_NAME:
            return self._builtin_template()
        return next(
            (item for item in self._user_templates.values() if item.name == name),
            None,
        )

    def save(
        self,
        name: str,
        parameters: LayoutParameters,
        description: str | None = None,
    ) -> LayoutTemplate:
        normalized = self.validate_name(name)
        if normalized == DEFAULT_TEMPLATE_NAME:
            raise ValueError("内置默认模板不能覆盖，请使用其他名称。")
        parameters.validate()
        existing = self.find_by_name(normalized)
        if existing is None:
            now = _current_time()
            template = LayoutTemplate(
                template_id=uuid4().hex,
                name=normalized,
                parameters=parameters,
                description=self.validate_description(description),
                created_at=now,
                updated_at=now,
            )
        else:
            template = replace(
                existing,
                parameters=parameters,
                description=(
                    existing.description
                    if description is None
                    else self.validate_description(description)
                ),
                updated_at=_current_time(),
            )
        self._user_templates[template.template_id] = template
        self._write()
        return template

    def update(
        self,
        template_id: str,
        parameters: LayoutParameters,
        *,
        description: str | None = None,
    ) -> LayoutTemplate:
        template = self.get(template_id)
        if template.builtin:
            raise ValueError("内置默认模板不能覆盖，请使用“另存为”。")
        parameters.validate()
        updated = replace(
            template,
            parameters=parameters,
            description=(
                template.description
                if description is None
                else self.validate_description(description)
            ),
            updated_at=_current_time(),
        )
        self._user_templates[template_id] = updated
        self._write()
        return updated

    def delete(self, identity: str) -> None:
        template = self.get(identity)
        if template.builtin:
            raise ValueError("内置默认模板不能删除。")
        del self._user_templates[template.template_id]
        self._write()

    def rename(self, identity: str, new_name: str) -> LayoutTemplate:
        template = self.get(identity)
        return self.update_details(
            identity,
            name=new_name,
            description=template.description,
        )

    def update_details(
        self,
        identity: str,
        *,
        name: str,
        description: str,
    ) -> LayoutTemplate:
        template = self.get(identity)
        if template.builtin:
            raise ValueError("内置默认模板不能修改。")
        normalized = self.validate_name(name)
        collision = self.find_by_name(normalized)
        if collision is not None and collision.template_id != template.template_id:
            raise FileExistsError(f"模板“{normalized}”已存在。")
        updated = replace(
            template,
            name=normalized,
            description=self.validate_description(description),
            updated_at=_current_time(),
        )
        self._user_templates[template.template_id] = updated
        self._write()
        return updated

    def duplicate(
        self,
        identity: str,
        new_name: str,
        *,
        description: str | None = None,
    ) -> LayoutTemplate:
        source = self.get(identity)
        normalized = self.validate_name(new_name)
        if self.find_by_name(normalized) is not None:
            raise FileExistsError(f"模板“{normalized}”已存在。")
        return self.save(
            normalized,
            source.parameters,
            source.description if description is None else description,
        )

    def export_template(self, identity: str, file_path: str) -> None:
        template = self.get(identity)
        atomic_write_json(
            {
                "文件类型": TEMPLATE_EXPORT_KIND,
                "数据版本": TEMPLATE_EXPORT_VERSION,
                "模板": {
                    "名称": template.name,
                    "说明": template.description,
                    "参数": template.parameters.to_dict(),
                },
            },
            file_path,
        )

    @classmethod
    def read_import_file(cls, file_path: str) -> LayoutTemplate:
        raw = safe_read_json(file_path, {})
        if not isinstance(raw, Mapping) or raw.get("文件类型") != TEMPLATE_EXPORT_KIND:
            raise RuntimeError("所选文件不是有效的通用经文排版模板。")
        if raw.get("数据版本") != TEMPLATE_EXPORT_VERSION:
            raise RuntimeError("导入模板的数据版本不受支持。")
        values = raw.get("模板")
        if not isinstance(values, Mapping):
            raise RuntimeError("导入模板的记录格式无效。")
        name = cls.validate_name(values.get("名称"))
        parameters = values.get("参数")
        if not isinstance(parameters, Mapping):
            raise RuntimeError(f"模板“{name}”参数格式无效。")
        return LayoutTemplate(
            template_id="",
            name=name,
            description=cls.validate_description(values.get("说明", "")),
            parameters=LayoutParameters.from_dict(parameters),
        )

    def import_template(
        self,
        file_path: str,
        *,
        target_name: str | None = None,
        overwrite: bool = False,
    ) -> LayoutTemplate:
        imported = self.read_import_file(file_path)
        name = self.validate_name(target_name or imported.name)
        existing = self.find_by_name(name)
        if existing is not None and not overwrite:
            raise FileExistsError(f"模板“{name}”已存在。")
        if existing is not None and existing.builtin:
            raise ValueError("内置默认模板不能覆盖，请使用其他名称。")
        return self.save(name, imported.parameters, imported.description)

    def _repair_configuration(self) -> None:
        """将有效模板写入数据库，不改动原配置文件。"""
        self._write(backup_existing=False)

    @staticmethod
    def _serialize_template(template: LayoutTemplate) -> dict[str, object]:
        return {
            "名称": template.name,
            "说明": template.description,
            "创建时间": template.created_at,
            "修改时间": template.updated_at,
            "参数": template.parameters.to_dict(),
        }

    def _write(self, *, backup_existing: bool = True) -> None:
        templates = {
            DEFAULT_TEMPLATE_ID: self._serialize_template(self._builtin_template())
        }
        templates.update(
            {
                template.template_id: self._serialize_template(template)
                for template in sorted(
                    self._user_templates.values(),
                    key=lambda item: item.name,
                )
            }
        )
        self._database.write_document(
            "通用经文排版模板",
            {
                "数据版本": TEMPLATE_DATA_VERSION,
                "用户模板": templates,
            },
            version=TEMPLATE_DATA_VERSION,
        )

    @staticmethod
    def suggested_export_path(template: LayoutTemplate, directory: str) -> str:
        return str(Path(directory) / f"{template.name}.json")
