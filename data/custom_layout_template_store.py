"""定制经文排版模板的版本化、原子化存储。"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Mapping
from uuid import uuid4

from core.custom_scripture_layout import (
    CustomBoardParameters,
    CustomLayoutTemplateParameters,
)
from data.application_database import (
    ApplicationDatabase,
    resolve_application_database_path,
)
from utils.file_utils import atomic_write_json, safe_read_json_with_source


CUSTOM_TEMPLATE_DATA_VERSION = 1
DEFAULT_CUSTOM_TEMPLATE_ID = "builtin:default"
DEFAULT_CUSTOM_TEMPLATE_NAME = "默认模板"
DEFAULT_CUSTOM_TEMPLATE_PARAMETERS = CustomLayoutTemplateParameters(
    boards=(CustomBoardParameters(),),
    include_punctuation=False,
    add_annotations=False,
)


def _current_time() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class CustomLayoutTemplate:
    template_id: str
    name: str
    parameters: CustomLayoutTemplateParameters
    builtin: bool = False
    description: str = ""
    created_at: str = ""
    updated_at: str = ""


class CustomLayoutTemplateStore:
    """管理内置托底模板和用户保存的多版参数模板。"""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self._database = ApplicationDatabase(
            resolve_application_database_path(file_path)
        )
        self._load_source = ""
        self._needs_repair = False
        self._user_templates = self._load()
        if self._needs_repair:
            self._repair_configuration()

    @staticmethod
    def validate_name(raw_name: object) -> str:
        name = str(raw_name or "").strip()
        if not name:
            raise ValueError("模板名称不能为空。")
        if len(name) > 40:
            raise ValueError("模板名称不能超过 40 个字符。")
        if any(character in name for character in '\\/:*?"<>|'):
            raise ValueError("模板名称包含不允许的字符。")
        return name

    @staticmethod
    def validate_description(raw_description: object) -> str:
        description = str(raw_description or "").strip()
        if len(description) > 500:
            raise ValueError("模板说明不能超过 500 个字符。")
        return description

    @staticmethod
    def _builtin_template() -> CustomLayoutTemplate:
        return CustomLayoutTemplate(
            DEFAULT_CUSTOM_TEMPLATE_ID,
            DEFAULT_CUSTOM_TEMPLATE_NAME,
            DEFAULT_CUSTOM_TEMPLATE_PARAMETERS,
            builtin=True,
            description="程序内置的定制排版托底参数，不能覆盖或删除。",
        )

    def _load(self) -> dict[str, CustomLayoutTemplate]:
        raw = self._database.read_document("定制经文排版模板")
        source = "数据库"
        if raw is None:
            raw, source = safe_read_json_with_source(self.file_path, {})
            self._needs_repair = True
        self._load_source = source
        if not raw:
            self._needs_repair = True
            return {}
        if not isinstance(raw, Mapping):
            raise RuntimeError("定制经文排版模板文件格式无效。")
        if raw.get("数据版本") != CUSTOM_TEMPLATE_DATA_VERSION:
            raise RuntimeError("定制经文排版模板数据版本不受支持。")
        records = raw.get("用户模板", {})
        if not isinstance(records, Mapping):
            raise RuntimeError("定制经文排版模板列表格式无效。")
        result: dict[str, CustomLayoutTemplate] = {}
        names: set[str] = set()
        for raw_id, value in records.items():
            template_id = str(raw_id or "").strip()
            if not template_id or not isinstance(value, Mapping):
                raise RuntimeError("定制经文排版模板记录无效。")
            name = self.validate_name(value.get("名称"))
            if template_id == DEFAULT_CUSTOM_TEMPLATE_ID or name == DEFAULT_CUSTOM_TEMPLATE_NAME:
                continue
            if name in names:
                raise RuntimeError(f"模板名称重复：{name}")
            raw_parameters = value.get("参数")
            if not isinstance(raw_parameters, Mapping):
                raise RuntimeError(f"模板“{name}”参数格式无效。")
            names.add(name)
            result[template_id] = CustomLayoutTemplate(
                template_id=template_id,
                name=name,
                parameters=CustomLayoutTemplateParameters.from_dict(raw_parameters),
                description=self.validate_description(value.get("说明", "")),
                created_at=str(value.get("创建时间") or ""),
                updated_at=str(value.get("修改时间") or ""),
            )
        if source != "数据库":
            self._needs_repair = True
        return result

    def list_templates(self) -> tuple[CustomLayoutTemplate, ...]:
        return (
            self._builtin_template(),
            *sorted(self._user_templates.values(), key=lambda item: item.name),
        )

    def get(self, identity: str) -> CustomLayoutTemplate:
        if identity in {DEFAULT_CUSTOM_TEMPLATE_ID, DEFAULT_CUSTOM_TEMPLATE_NAME}:
            return self._builtin_template()
        if identity in self._user_templates:
            return self._user_templates[identity]
        for template in self._user_templates.values():
            if template.name == identity:
                return template
        raise KeyError(f"找不到定制排版模板：{identity}")

    def find_by_name(self, name: str) -> CustomLayoutTemplate | None:
        if name == DEFAULT_CUSTOM_TEMPLATE_NAME:
            return self._builtin_template()
        return next(
            (item for item in self._user_templates.values() if item.name == name),
            None,
        )

    def save(
        self,
        name: str,
        parameters: CustomLayoutTemplateParameters,
        description: str | None = None,
    ) -> CustomLayoutTemplate:
        normalized = self.validate_name(name)
        if normalized == DEFAULT_CUSTOM_TEMPLATE_NAME:
            raise ValueError("内置默认模板不能覆盖，请使用其他名称。")
        parameters.validate()
        existing = self.find_by_name(normalized)
        if existing is None:
            now = _current_time()
            template = CustomLayoutTemplate(
                uuid4().hex,
                normalized,
                parameters,
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
        parameters: CustomLayoutTemplateParameters,
        *,
        description: str | None = None,
    ) -> CustomLayoutTemplate:
        template = self.get(template_id)
        if template.builtin:
            raise ValueError("内置默认模板不能覆盖，请保存为新模板。")
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

    def _repair_configuration(self) -> None:
        """将有效模板写入数据库，不改动原配置文件。"""
        self._write(backup_existing=False)

    def _write(self, *, backup_existing: bool = True) -> None:
        builtin = self._builtin_template()
        records = {
            builtin.template_id: self._serialize_template(builtin),
            **{
                template.template_id: self._serialize_template(template)
                for template in sorted(
                    self._user_templates.values(), key=lambda item: item.name
                )
            },
        }
        self._database.write_document(
            "定制经文排版模板",
            {"数据版本": CUSTOM_TEMPLATE_DATA_VERSION, "用户模板": records},
            version=CUSTOM_TEMPLATE_DATA_VERSION,
        )

    @staticmethod
    def _serialize_template(template: CustomLayoutTemplate) -> dict[str, object]:
        return {
            "名称": template.name,
            "说明": template.description,
            "创建时间": template.created_at,
            "修改时间": template.updated_at,
            "参数": template.parameters.to_dict(),
        }

    @staticmethod
    def suggested_export_path(template: CustomLayoutTemplate, directory: str) -> str:
        return str(Path(directory) / f"{template.name}.json")
