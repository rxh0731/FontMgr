# glyph_service.py — 字形数据与六阶段流程管理

import copy
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Optional, Tuple

from send2trash import send2trash

import config
from data.library_database import LibraryDatabase, library_database_path
from services.batch_persistence import (
    JOURNAL_FILENAME,
    acquire_batch_library_lock,
    recover_batch_journal,
)
from services.file_transaction_recovery import (
    FileChange,
    FileTransaction,
    has_file_transaction_artifacts,
    ensure_file_transactions_ready,
    recover_file_transactions,
)
from utils.file_utils import (
    atomic_write_json,
    compute_file_md5,
    ensure_dir,
    is_safe_windows_filename,
    resolve_safe_child_file,
    safe_read_json_with_source,
    validate_final_char,
)


class GlyphService:
    """管理字库、字形记录、六层工作文件以及四阶段状态。"""

    @staticmethod
    def _default_ink_counts() -> dict[str, int]:
        """返回可安全识别旧记录的墨色结果统计。"""
        return {
            "总数": 0,
            "已达标": 0,
            "待确认": 0,
            "人工例外": 0,
        }

    @classmethod
    def _default_coordination_summary(cls) -> dict[str, Any]:
        """返回完整的整体协调摘要，兼容旧字库缺少墨色字段的情况。"""
        return {
            "基准": {},
            "墨色基准": None,
            "墨色统一启用": True,
            "几何协调完成": False,
            "墨色统一完成": False,
            "墨色方法": "",
            "墨色方法版本": None,
            "墨色统计": cls._default_ink_counts(),
            "最后生成时间": "",
        }

    _LEGACY_DIR_NAMES = {
        config.DIR_ORIGINAL_FILES: "原始文件",
        config.DIR_GRAY_MASTER_FILES: "灰度母版",
        config.DIR_CLEAN_MASK_FILES: "清洁掩码",
        config.DIR_INTERMEDIATE_FILES: "中间文件",
        config.DIR_REVIEWED_FILES: "审核文件",
        config.DIR_FINISHED_FILES: "成品文件",
    }
    _LEGACY_STATUSES = {
        "待自动优化": config.STATUS_PENDING_OPTIMIZATION,
        "待手工审核": config.STATUS_PENDING_MANUAL_REVIEW,
        "已审核": config.STATUS_REVIEWED,
    }

    @classmethod
    def open(cls, ziku_name: str, ziku_dir: str) -> "GlyphService":
        """显式打开字库，并执行必要的数据恢复与目录迁移。"""

        return cls(ziku_name, ziku_dir)

    def __init__(self, ziku_name: str, ziku_dir: str) -> None:
        """打开 SQLite 字库；首次打开旧字库时执行一次性导入。"""
        self.ziku_name = ziku_name
        self.ziku_dir = ziku_dir
        self._json_path = os.path.join(ziku_dir, f"{ziku_name}.json")
        self._database: LibraryDatabase | None = None
        self._dirty_variant_ids: set[str] = set()
        self._deleted_variant_ids: set[str] = set()
        self._all_variants_dirty = False
        self._data = self._load_or_init()
        if self._database is None:
            self._database = LibraryDatabase.install_from_data(
                self.ziku_dir,
                self._data,
                source_path=self._json_path if os.path.isfile(self._json_path) else "",
            )
        self._migrate_workflow_dirs()

    def _migrate_workflow_dirs(self) -> None:
        """无损迁移旧版六阶段目录；发生冲突时不执行任何改名。"""
        rename_pairs: list[tuple[str, str]] = []
        for new_name, old_name in self._LEGACY_DIR_NAMES.items():
            old_path = os.path.join(self.ziku_dir, old_name)
            new_path = os.path.join(self.ziku_dir, new_name)
            if not os.path.isdir(old_path):
                continue
            if os.path.exists(new_path):
                raise RuntimeError(f"阶段目录冲突：{old_name} 与 {new_name} 同时存在，请先合并文件")
            rename_pairs.append((old_path, new_path))
        for old_path, new_path in rename_pairs:
            os.replace(old_path, new_path)

    def _load_or_init(self) -> dict[str, Any]:
        """读取数据库，或将当前 JSON 字库一次性导入数据库。"""
        database_path = library_database_path(self.ziku_dir)
        if os.path.isfile(database_path):
            self._database = LibraryDatabase.open(self.ziku_dir)
            data = self._database.load_data()
            if self._normalize_database_data(data):
                self._database.save_full_data(data)
            recover_file_transactions(
                data,
                ziku_dir=self.ziku_dir,
                persist_callback=self._database.save_full_data,
            )
            recover_batch_journal(
                data,
                ziku_name=self.ziku_name,
                ziku_dir=self.ziku_dir,
                persist_callback=self._database.save_full_data,
            )
            return data
        backup_path = self._json_path + ".bak"
        journal_path = os.path.join(self.ziku_dir, JOURNAL_FILENAME)
        file_transaction_exists = has_file_transaction_artifacts(self.ziku_dir)
        recovery_artifact_exists = any(
            os.path.exists(path)
            for path in (self._json_path, backup_path, journal_path)
        ) or file_transaction_exists
        if recovery_artifact_exists:
            data, source_path = safe_read_json_with_source(
                self._json_path,
                expected_type=dict,
            )
            if not source_path:
                if os.path.exists(journal_path) or file_transaction_exists:
                    raise RuntimeError(
                        "发现批处理或图片事务恢复记录，但字库主数据和备份均缺失或损坏；"
                        "为避免生成空字库，已保留现有恢复文件。"
                    )
                raise RuntimeError(
                    "字库主数据和备份均缺失或损坏，无法安全打开字库。"
                )
            if source_path == backup_path:
                # 先从有效备份修复主文件，不能用损坏主文件覆盖唯一好备份。
                atomic_write_json(
                    data,
                    self._json_path,
                    indent=None,
                    backup_existing=False,
                )
            if data.get("数据版本") != 3:
                raise RuntimeError(
                    f"字库数据版本不受支持：{data.get('数据版本', '未知')}。"
                )
            recover_file_transactions(
                data,
                ziku_dir=self.ziku_dir,
                json_path=self._json_path,
            )
            recover_batch_journal(
                data,
                ziku_name=self.ziku_name,
                ziku_dir=self.ziku_dir,
                json_path=self._json_path,
            )
            data.pop("简繁映射表", None)
            data_changed = False
            metadata = data.setdefault("元数据", {})
            if "DPI" not in metadata and "分辨率" in metadata:
                metadata["DPI"] = metadata.pop("分辨率")
                data_changed = True
            for key in ("成品风格", "透明背景"):
                if key in metadata:
                    metadata.pop(key)
                    data_changed = True
            for key in ("成品宽度毫米", "成品高度毫米"):
                if key in metadata:
                    rounded_value = round(float(metadata[key]), 2)
                    if metadata[key] != rounded_value:
                        metadata[key] = rounded_value
                        data_changed = True
            coordination = data.get("整体协调")
            if not isinstance(coordination, dict):
                coordination = {}
                data["整体协调"] = coordination
                data_changed = True
            ink_contract_fields = ("墨色方法", "墨色方法版本", "墨色统计")
            missing_ink_contract = any(
                key not in coordination for key in ink_contract_fields
            )
            for key, default_value in self._default_coordination_summary().items():
                if key not in coordination:
                    coordination[key] = copy.deepcopy(default_value)
                    data_changed = True
            if missing_ink_contract and coordination.get("墨色统一完成") is True:
                coordination["墨色统一完成"] = False
                data_changed = True
            for detail in data.get("变体详情", {}).values():
                old_status = detail.get("状态")
                if old_status in self._LEGACY_STATUSES:
                    detail["状态"] = self._LEGACY_STATUSES[old_status]
                    data_changed = True
                if "整体协调参数" not in detail:
                    detail["整体协调参数"] = {}
                    data_changed = True
            if data_changed:
                atomic_write_json(data, self._json_path, indent=None)
            return data
        current_time = self._now()
        return {
            "数据版本": 3,
            "库名": self.ziku_name,
            "元数据": {
                "DPI": 300,
                "画布宽": 250,
                "画布高": 250,
                "成品宽度毫米": round(250 / 300 * 25.4, 2),
                "成品高度毫米": round(250 / 300 * 25.4, 2),
                "创建时间": current_time,
                "最后修改": current_time,
            },
            "会话": {},
            "字形组索引": {},
            "变体详情": {},
            "整体协调": self._default_coordination_summary(),
        }

    def _normalize_database_data(self, data: dict[str, Any]) -> bool:
        """补齐当前数据契约，避免缺失字段被错误判定为已完成。"""
        changed = False
        coordination = data.get("整体协调")
        if not isinstance(coordination, dict):
            coordination = {}
            data["整体协调"] = coordination
            changed = True
        contract_fields = ("墨色方法", "墨色方法版本", "墨色统计")
        missing_contract = any(key not in coordination for key in contract_fields)
        for key, default_value in self._default_coordination_summary().items():
            if key not in coordination:
                coordination[key] = copy.deepcopy(default_value)
                changed = True
        if missing_contract and coordination.get("墨色统一完成") is True:
            coordination["墨色统一完成"] = False
            changed = True
        for detail in data.get("变体详情", {}).values():
            if isinstance(detail, dict) and "整体协调参数" not in detail:
                detail["整体协调参数"] = {}
                changed = True
        return changed

    def init_metadata(
        self,
        dpi: int = 300,
        canvas_w: int = 250,
        canvas_h: int = 250,
        width_mm: Optional[float] = None,
        height_mm: Optional[float] = None,
    ) -> None:
        old_metadata = self._data.get("元数据", {})
        self._data["元数据"] = {
            "DPI": dpi,
            "画布宽": canvas_w,
            "画布高": canvas_h,
            "成品宽度毫米": round(float(width_mm if width_mm is not None else canvas_w / dpi * 25.4), 2),
            "成品高度毫米": round(float(height_mm if height_mm is not None else canvas_h / dpi * 25.4), 2),
            "创建时间": old_metadata.get("创建时间") or self._now(),
            "最后修改": self._now(),
        }

    def get_metadata(self) -> dict[str, Any]:
        return dict(self._data.get("元数据", {}))

    def remove_metadata_keys(self, *keys: str) -> None:
        """删除不再使用的字库参数字段。"""
        metadata = self._data.setdefault("元数据", {})
        for key in keys:
            metadata.pop(key, None)
        self._update_mtime()

    def snapshot_state(self) -> dict[str, Any]:
        """复制完整字库状态，供跨文件保存事务失败时恢复。"""
        return copy.deepcopy(self._data)

    def restore_state(self, snapshot: dict[str, Any]) -> None:
        """原位恢复字库状态，保留持有根字典引用的服务对象。"""
        if not isinstance(snapshot, dict):
            raise TypeError("字库状态快照必须是字典。")
        self._data.clear()
        self._data.update(copy.deepcopy(snapshot))
        self._all_variants_dirty = True

    def snapshot_variant_state(
        self,
        variant_id: str,
        group_chars: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """只复制单字事务可能修改的状态，避免批量时复制整库。"""
        details = self._data.setdefault("变体详情", {})
        detail = details.get(variant_id)
        snapshot = {
            "变体ID": variant_id,
            "变体存在": isinstance(detail, dict),
            "变体详情": copy.deepcopy(detail) if isinstance(detail, dict) else {},
            "元数据": copy.deepcopy(self._data.get("元数据", {})),
            "整体协调": copy.deepcopy(self._data.get("整体协调", {})),
        }
        if group_chars:
            groups = self._data.setdefault("字形组索引", {})
            snapshot["字形组索引片段"] = {
                char: copy.deepcopy(groups.get(char))
                for char in dict.fromkeys(group_chars)
            }
        return snapshot

    def restore_variant_state(self, snapshot: dict[str, Any]) -> None:
        """原位恢复单字、元数据时间及整体协调摘要。"""
        if not isinstance(snapshot, dict):
            raise TypeError("单字状态快照必须是字典。")
        variant_id = str(snapshot.get("变体ID", ""))
        if not variant_id:
            raise ValueError("单字状态快照缺少变体ID。")
        details = self._data.setdefault("变体详情", {})
        if snapshot.get("变体存在", True):
            restored_detail = copy.deepcopy(snapshot.get("变体详情", {}))
            current_detail = details.get(variant_id)
            if isinstance(current_detail, dict):
                current_detail.clear()
                current_detail.update(restored_detail)
            else:
                details[variant_id] = restored_detail
        else:
            details.pop(variant_id, None)

        for key in ("元数据", "整体协调"):
            restored = copy.deepcopy(snapshot.get(key, {}))
            current = self._data.get(key)
            if isinstance(current, dict):
                current.clear()
                current.update(restored)
            else:
                self._data[key] = restored

        group_fragment = snapshot.get("字形组索引片段")
        if isinstance(group_fragment, dict):
            groups = self._data.setdefault("字形组索引", {})
            for char, variant_ids in group_fragment.items():
                if isinstance(variant_ids, list) and variant_ids:
                    groups[str(char)] = copy.deepcopy(variant_ids)
                else:
                    groups.pop(str(char), None)
        self._dirty_variant_ids.add(variant_id)

    def rename_ziku(self, new_name: str) -> str:
        """改名字库目录和数据库内库名，返回改名后的目录。"""
        new_name = new_name.strip()
        if not new_name:
            raise ValueError("字库名称不能为空。")
        if new_name == self.ziku_name:
            return self.ziku_dir
        if not is_safe_windows_filename(new_name):
            raise ValueError("字库名称不是有效的 Windows 目录名。")

        old_name = self.ziku_name
        old_dir = os.path.abspath(self.ziku_dir)
        parent_dir = os.path.dirname(old_dir)
        new_dir = os.path.join(parent_dir, new_name)
        if os.path.exists(new_dir):
            raise FileExistsError(f"字库“{new_name}”已存在。")

        self._data["库名"] = new_name
        self._data.setdefault("元数据", {})["最后修改"] = self._now()
        try:
            os.replace(old_dir, new_dir)
            self.ziku_name = new_name
            self.ziku_dir = new_dir
            self._json_path = os.path.join(new_dir, f"{new_name}.json")
            self._database = LibraryDatabase.open(new_dir)
            self.save()
        except Exception:
            self._data["库名"] = old_name
            if os.path.exists(new_dir) and not os.path.exists(old_dir):
                os.replace(new_dir, old_dir)
            self.ziku_name = old_name
            self.ziku_dir = old_dir
            self._json_path = os.path.join(old_dir, f"{old_name}.json")
            self._database = LibraryDatabase.open(old_dir)
            raise
        return new_dir

    def update_output_spec(
        self,
        dpi: int,
        canvas_w: int,
        canvas_h: int,
        width_mm: float,
        height_mm: float,
    ) -> int:
        """更新全库成品规格，并使已有成品回到待整体协调状态。"""
        self.init_metadata(dpi, canvas_w, canvas_h, width_mm, height_mm)
        invalidated_count = 0
        for detail in self._data.get("变体详情", {}).values():
            if not detail.get("成品文件"):
                continue
            self._remove_finished_file(detail)
            detail["成品文件"] = ""
            detail["成品MD5"] = ""
            detail["整体协调参数"] = {}
            if detail.get("中间文件"):
                detail["状态"] = config.STATUS_REVIEWED
            else:
                detail["状态"] = config.STATUS_PENDING_OPTIMIZATION
            invalidated_count += 1
        self._all_variants_dirty = True
        self._data["整体协调"] = self._default_coordination_summary()
        self.save()
        return invalidated_count

    def get_three_dirs(self) -> Tuple[str, str, str]:
        """保留旧接口：原始文件、自动优化预览、最终成品。"""
        return (
            os.path.join(self.ziku_dir, config.DIR_ORIGINAL_FILES),
            os.path.join(self.ziku_dir, config.DIR_INTERMEDIATE_FILES),
            os.path.join(self.ziku_dir, config.DIR_FINISHED_FILES),
        )

    def get_workflow_dirs(self) -> dict[str, str]:
        """返回六阶段流程中互不覆盖的数据目录。"""
        return {
            "原图": os.path.join(self.ziku_dir, config.DIR_ORIGINAL_FILES),
            "灰度母版": os.path.join(self.ziku_dir, config.DIR_GRAY_MASTER_FILES),
            "清洁掩码": os.path.join(self.ziku_dir, config.DIR_CLEAN_MASK_FILES),
            "优化预览": os.path.join(self.ziku_dir, config.DIR_INTERMEDIATE_FILES),
            "手工审核": os.path.join(self.ziku_dir, config.DIR_REVIEWED_FILES),
            "成品": os.path.join(self.ziku_dir, config.DIR_FINISHED_FILES),
        }

    def ensure_dirs(self) -> None:
        ensure_dir(self.ziku_dir)
        for directory in self.get_workflow_dirs().values():
            ensure_dir(directory)

    def add_original(
        self,
        target_char: str,
        original_filename: str,
        source_filename: str,
        md5_value: str,
        image_info: Optional[dict[str, Any]] = None,
    ) -> str:
        """登记一个已无损复制到原始文件目录的文件。"""
        self._require_safe_stage_filename(original_filename, "原始文件")
        valid, message = validate_final_char(target_char)
        if not valid:
            raise ValueError(message)
        variant_id = self._gen_variant_id(target_char, md5_value)
        index = 1
        base_id = variant_id
        while variant_id in self._data["变体详情"]:
            index += 1
            variant_id = f"{base_id}_{index}"
        detail = {
            "变体ID": variant_id,
            "归属字": target_char,
            "原始文件": original_filename,
            "导入前文件名": source_filename,
            "原始MD5": md5_value,
            "图像信息": image_info or {},
            "状态": config.STATUS_PENDING_OPTIMIZATION,
            "中间文件": "",
            "中间MD5": "",
            "灰度母版文件": "",
            "灰度母版MD5": "",
            "清洁掩码文件": "",
            "清洁掩码MD5": "",
            "自动优化": {"方案名": "", "方案": {}, "得分": None, "轮次": 0},
            "手工编辑": {"已编辑": False, "最后保存时间": ""},
            "变换参数": self.default_transform_params(),
            "成品文件": "",
            "成品MD5": "",
            "整体协调参数": {},
            "备注": "",
        }
        self._data["变体详情"][variant_id] = detail
        self._data["字形组索引"].setdefault(target_char, []).append(variant_id)
        self._dirty_variant_ids.add(variant_id)
        self._update_mtime()
        return variant_id

    def discard_unsaved_original(self, variant_id: str) -> None:
        """撤销尚未保存的导入登记，不触碰任何阶段文件。"""

        detail = self._data["变体详情"].pop(variant_id, None)
        if not detail:
            return
        self._deleted_variant_ids.add(variant_id)
        target_char = str(detail.get("归属字", ""))
        variants = self._data["字形组索引"].get(target_char, [])
        remaining = [item for item in variants if item != variant_id]
        if remaining:
            self._data["字形组索引"][target_char] = remaining
        else:
            self._data["字形组索引"].pop(target_char, None)
        self._update_mtime()

    def default_transform_params(self) -> dict[str, Any]:
        return {
            "缩放": 1.0, "旋转": 0.0, "偏移X": 0, "偏移Y": 0,
            "拉伸W": 1.0, "拉伸H": 1.0, "扭曲": [0.0] * 8,
        }

    def find_by_md5(self, md5_value: str) -> Optional[dict[str, Any]]:
        for detail in self._data["变体详情"].values():
            if detail.get("原始MD5") == md5_value:
                return detail
        return None

    def confirm_optimization(
        self,
        variant_id: str,
        intermediate_filename: str,
        intermediate_md5: str,
        scheme_name: str,
        scheme: dict[str, Any],
        score: float,
        round_number: int,
        gray_master_filename: str = "",
        gray_master_md5: str = "",
        clean_mask_filename: str = "",
        clean_mask_md5: str = "",
    ) -> None:
        for label, filename in (
            ("优化预览", intermediate_filename),
            ("灰度母版", gray_master_filename),
            ("清洁掩码", clean_mask_filename),
        ):
            if filename:
                self._require_safe_stage_filename(filename, label)
        detail = self.get_variant(variant_id)
        if not detail:
            return
        self._remove_finished_file(detail)
        detail.update({
            "中间文件": intermediate_filename,
            "中间MD5": intermediate_md5,
            "灰度母版文件": gray_master_filename,
            "灰度母版MD5": gray_master_md5,
            "清洁掩码文件": clean_mask_filename,
            "清洁掩码MD5": clean_mask_md5,
            "审核文件": "",
            "审核MD5": "",
            "状态": config.STATUS_PENDING_MANUAL_REVIEW,
            "自动优化": {
                "方案名": scheme_name,
                "方案": scheme,
                "得分": round(float(score), 2),
                "轮次": int(round_number),
            },
            "手工编辑": {"已编辑": False, "最后保存时间": ""},
            "变换参数": self.default_transform_params(),
            "成品文件": "",
            "成品MD5": "",
            "整体协调参数": {},
        })
        self._update_mtime()

    def mark_manual_saved(
        self,
        variant_id: str,
        reviewed_filename: str,
        reviewed_md5: str,
        edited: bool = True,
    ) -> None:
        self._require_safe_stage_filename(reviewed_filename, "手工审核")
        detail = self.get_variant(variant_id)
        if not detail:
            return
        self._remove_finished_file(detail)
        detail["审核文件"] = reviewed_filename
        detail["审核MD5"] = reviewed_md5
        detail["手工编辑"] = {"已编辑": bool(edited), "最后保存时间": self._now()}
        detail["状态"] = config.STATUS_PENDING_MANUAL_REVIEW
        detail["成品文件"] = ""
        detail["成品MD5"] = ""
        self._update_mtime()

    def approve_manual_review(self, variant_id: str) -> bool:
        detail = self.get_variant(variant_id)
        if not detail:
            return False
        workflow_dirs = self.get_workflow_dirs()
        reviewed_filename = detail.get("审核文件", "")
        preview_filename = detail.get("中间文件", "")
        reviewed_exists = bool(
            reviewed_filename
            and resolve_safe_child_file(
                workflow_dirs["手工审核"], reviewed_filename
            )
        )
        preview_exists = bool(
            preview_filename
            and resolve_safe_child_file(
                workflow_dirs["优化预览"], preview_filename
            )
        )
        if not reviewed_exists and not preview_exists:
            return False
        detail["状态"] = config.STATUS_REVIEWED
        self._update_mtime()
        return True

    def mark_finished(
        self,
        variant_id: str,
        finished_filename: str,
        finished_md5: str,
        adjustment_params: dict[str, Any],
    ) -> None:
        self._require_safe_stage_filename(finished_filename, "成品")
        detail = self.get_variant(variant_id)
        if not detail:
            return
        detail["成品文件"] = finished_filename
        detail["成品MD5"] = finished_md5
        detail["整体协调参数"] = adjustment_params
        detail["状态"] = config.STATUS_FINISHED
        self._update_mtime()

    def set_coordination_summary(
        self,
        baseline: dict[str, Any],
        ink_baseline: Optional[float] = None,
        *,
        geometry_completed: bool = True,
        ink_completed: Optional[bool] = None,
        ink_enabled: Optional[bool] = None,
        ink_method: Optional[str] = None,
        ink_method_version: Optional[int] = None,
        ink_counts: Optional[Mapping[str, Any]] = None,
    ) -> None:
        enabled = bool(ink_baseline is not None) if ink_enabled is None else bool(ink_enabled)
        completed = bool(ink_baseline is not None) if ink_completed is None else bool(ink_completed)
        normalized_counts = self._normalize_ink_counts(ink_counts)
        if ink_counts is not None:
            counts_completed = (
                normalized_counts["总数"]
                == normalized_counts["已达标"] + normalized_counts["人工例外"]
                and normalized_counts["待确认"] == 0
            )
            completed = completed and counts_completed
        self._data["整体协调"] = {
            "基准": copy.deepcopy(baseline),
            "墨色基准": round(float(ink_baseline), 2) if ink_baseline is not None else None,
            "墨色统一启用": enabled,
            "几何协调完成": bool(geometry_completed),
            "墨色统一完成": enabled and completed,
            "墨色方法": str(ink_method or "").strip(),
            "墨色方法版本": self._normalize_ink_method_version(ink_method_version),
            "墨色统计": normalized_counts,
            "最后生成时间": self._now(),
        }
        self._update_mtime()

    @staticmethod
    def _require_safe_stage_filename(filename: object, label: str) -> None:
        if not is_safe_windows_filename(filename):
            raise ValueError(f"{label}文件名不安全。")

    def get_coordination_summary(self) -> dict[str, Any]:
        """返回整体协调摘要副本，供页面恢复全局选项。"""
        summary = self._data.get("整体协调", {})
        if not isinstance(summary, dict):
            return self._default_coordination_summary()
        result = self._default_coordination_summary()
        result.update(copy.deepcopy(summary))
        return result

    @classmethod
    def _normalize_ink_counts(
        cls,
        counts: Optional[Mapping[str, Any]],
    ) -> dict[str, int]:
        """规范墨色统计；未提供时保留旧调用的空统计语义。"""
        if counts is None:
            return cls._default_ink_counts()
        if not isinstance(counts, Mapping):
            raise TypeError("墨色统计必须是映射。")
        result: dict[str, int] = {}
        for key in ("已达标", "待确认", "人工例外"):
            value = counts.get(key, 0)
            if isinstance(value, bool):
                raise ValueError(f"墨色统计“{key}”必须是非负整数。")
            try:
                number = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"墨色统计“{key}”必须是非负整数。") from exc
            if number < 0 or number != value:
                raise ValueError(f"墨色统计“{key}”必须是非负整数。")
            result[key] = number
        total_value = counts.get(
            "总数",
            result["已达标"] + result["待确认"] + result["人工例外"],
        )
        if isinstance(total_value, bool):
            raise ValueError("墨色统计“总数”必须是非负整数。")
        try:
            total = int(total_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("墨色统计“总数”必须是非负整数。") from exc
        if total < 0 or total != total_value:
            raise ValueError("墨色统计“总数”必须是非负整数。")
        return {
            "总数": total,
            "已达标": result["已达标"],
            "待确认": result["待确认"],
            "人工例外": result["人工例外"],
        }

    @staticmethod
    def _normalize_ink_method_version(version: Optional[int]) -> Optional[int]:
        if version is None:
            return None
        if isinstance(version, bool):
            raise ValueError("墨色方法版本必须是正整数。")
        try:
            normalized = int(version)
        except (TypeError, ValueError) as exc:
            raise ValueError("墨色方法版本必须是正整数。") from exc
        if normalized <= 0 or normalized != version:
            raise ValueError("墨色方法版本必须是正整数。")
        return normalized

    def get_variant(self, variant_id: str) -> dict[str, Any]:
        if variant_id in self._data["变体详情"]:
            self._dirty_variant_ids.add(variant_id)
        return self._data["变体详情"].get(variant_id, {})

    def get_variants(self) -> dict[str, dict[str, Any]]:
        """返回全部变体详情的浅拷贝，供界面只读展示。"""
        return {variant_id: dict(detail) for variant_id, detail in self._data["变体详情"].items()}

    def get_glyph_groups(self) -> dict[str, list[str]]:
        """返回字形组索引副本，避免界面直接访问内部数据。"""
        return {char: list(variant_ids) for char, variant_ids in self._data["字形组索引"].items()}

    def preview_variant_char_change(
        self,
        variant_id: str,
        new_char: str,
    ) -> dict[str, Any]:
        """预检归属字修改并返回不会写盘的文件改名计划。"""
        return self._build_variant_char_move_plan(variant_id, new_char)

    def move_variant_to_char(
        self,
        variant_id: str,
        new_char: str,
    ) -> dict[str, Any]:
        """修改单个字形的归属字符，并事务化同步六阶段文件名。"""
        library_lock = acquire_batch_library_lock(self.ziku_dir)
        try:
            ensure_file_transactions_ready(self.ziku_dir)
            return self._move_variant_to_char_locked(variant_id, new_char)
        finally:
            library_lock.release()

    def _move_variant_to_char_locked(
        self,
        variant_id: str,
        new_char: str,
    ) -> dict[str, Any]:
        plan = self._build_variant_char_move_plan(variant_id, new_char)
        if not plan["文件变更"]:
            return plan

        old_char = str(plan["原归属字"])
        target_char = str(plan["新归属字"])
        group_chars = (old_char, target_char)
        old_state = self.snapshot_variant_state(variant_id, group_chars)
        temporary_paths: list[str] = []
        transaction: FileTransaction | None = None
        state_persisted = False
        try:
            changes: list[FileChange] = []
            for change in plan["文件变更"]:
                source_path = str(change["原路径"])
                target_path = str(change["新路径"])
                descriptor, temporary_path = tempfile.mkstemp(
                    prefix=".fonteditor_rename_",
                    suffix=os.path.splitext(target_path)[1],
                    dir=os.path.dirname(target_path),
                )
                os.close(descriptor)
                shutil.copyfile(source_path, temporary_path)
                temporary_paths.append(temporary_path)
                changes.append(
                    FileChange(
                        target_path=source_path,
                        backup_prefix=".fonteditor_rename_old_",
                    )
                )
                changes.append(
                    FileChange(
                        target_path=target_path,
                        temporary_path=temporary_path,
                        new_md5=compute_file_md5(temporary_path),
                        backup_prefix=".fonteditor_rename_target_",
                    )
                )

            transaction = FileTransaction.begin(
                self.ziku_dir,
                changes,
                old_state,
            )
            transaction.backup_targets()

            detail = self.get_variant(variant_id)
            groups = self._data["字形组索引"]
            groups[old_char].remove(variant_id)
            if not groups[old_char]:
                groups.pop(old_char, None)
            groups.setdefault(target_char, []).append(variant_id)
            detail["归属字"] = target_char
            detail["变体序号"] = int(plan["新变体序号"])
            for change in plan["文件变更"]:
                detail[str(change["字段"])] = str(change["新文件名"])
            self._update_mtime()

            new_state = self.snapshot_variant_state(variant_id, group_chars)
            transaction.mark_rollforward(new_state)
            transaction.install_new_files()
            self.save()
            state_persisted = True
            plan["清理提示"] = tuple(transaction.finalize())
            return plan
        except Exception as exc:
            if state_persisted:
                raise
            self.restore_variant_state(old_state)
            rollback_errors = transaction.rollback() if transaction is not None else []
            if rollback_errors:
                raise RuntimeError(
                    "字形名称修改失败，且文件回滚未完全完成："
                    + "；".join(rollback_errors)
                ) from exc
            raise
        finally:
            if transaction is None:
                for temporary_path in temporary_paths:
                    try:
                        if os.path.exists(temporary_path):
                            os.remove(temporary_path)
                    except OSError:
                        pass

    def _build_variant_char_move_plan(
        self,
        variant_id: str,
        new_char: str,
    ) -> dict[str, Any]:
        """生成六阶段同步改名计划，并拒绝缺失或越界的引用文件。"""
        new_char = new_char.strip()
        valid, message = validate_final_char(new_char)
        if not valid:
            raise ValueError(message)
        detail = self.get_variant(variant_id)
        if not detail:
            raise ValueError("字形记录不存在。")
        old_char = str(detail.get("归属字", ""))
        if new_char == old_char:
            return {
                "变体ID": variant_id,
                "原归属字": old_char,
                "新归属字": new_char,
                "新变体序号": int(detail.get("变体序号", 0) or 0),
                "原始文件名": str(detail.get("原始文件", "")),
                "新文件名": str(detail.get("原始文件", "")),
                "文件变更": [],
            }

        groups = self._data["字形组索引"]
        old_variant_ids = groups.get(old_char, [])
        if variant_id not in old_variant_ids:
            raise ValueError("字形组索引与字形记录不一致。")
        target_variant_ids = groups.get(new_char, [])
        new_index = self._next_char_file_index(new_char, target_variant_ids)
        new_base_name = f"{new_char}-{new_index:04d}"
        workflow_dirs = self.get_workflow_dirs()
        layer_fields = (
            ("原图", "原始文件"),
            ("灰度母版", "灰度母版文件"),
            ("清洁掩码", "清洁掩码文件"),
            ("优化预览", "中间文件"),
            ("手工审核", "审核文件"),
            ("成品", "成品文件"),
        )
        rename_plan: list[dict[str, str]] = []
        target_paths: set[str] = set()
        for layer, field in layer_fields:
            old_filename = str(detail.get(field, ""))
            if not old_filename:
                continue
            source_path = resolve_safe_child_file(
                workflow_dirs[layer],
                old_filename,
                require_exists=False,
            )
            if not source_path:
                raise ValueError(f"{layer}引用的文件名不安全：{old_filename}")
            if not os.path.isfile(source_path):
                raise FileNotFoundError(
                    f"{layer}引用的文件不存在：{old_filename}。请先重新核对字库数据。"
                )
            if not resolve_safe_child_file(workflow_dirs[layer], old_filename):
                raise ValueError(f"{layer}引用的文件不是安全的普通文件：{old_filename}")
            extension = os.path.splitext(old_filename)[1]
            new_filename = new_base_name + extension
            target_path = resolve_safe_child_file(
                workflow_dirs[layer],
                new_filename,
                require_exists=False,
            )
            if not target_path:
                raise ValueError(f"目标阶段文件名不安全：{new_filename}")
            normalized_target = os.path.normcase(os.path.abspath(target_path))
            if normalized_target in target_paths or (
                os.path.exists(target_path)
                and os.path.normcase(os.path.abspath(source_path)) != normalized_target
            ):
                raise FileExistsError(f"目标文件已存在：{new_filename}")
            target_paths.add(normalized_target)
            rename_plan.append(
                {
                    "阶段": layer,
                    "字段": field,
                    "原文件名": old_filename,
                    "新文件名": new_filename,
                    "原路径": source_path,
                    "新路径": target_path,
                }
            )

        if not any(change["字段"] == "原始文件" for change in rename_plan):
            raise FileNotFoundError("字形没有可用的原图文件，请先重新核对字库数据。")
        original_change = next(
            change for change in rename_plan if change["字段"] == "原始文件"
        )
        return {
            "变体ID": variant_id,
            "原归属字": old_char,
            "新归属字": new_char,
            "新变体序号": new_index,
            "原始文件名": original_change["原文件名"],
            "新文件名": original_change["新文件名"],
            "文件变更": rename_plan,
        }

    def _next_char_file_index(self, char: str, variant_ids: list[str]) -> int:
        """按目标字现有记录及六阶段文件的最大编号生成下一编号。"""
        max_index = 0
        filename_pattern = re.compile(rf"^{re.escape(char)}-(\d{{4}})(?:\.[^.]+)?$")
        for target_variant_id in variant_ids:
            target_detail = self.get_variant(target_variant_id)
            try:
                max_index = max(max_index, int(target_detail.get("变体序号", 0)))
            except (TypeError, ValueError):
                pass
            for field in ("原始文件", "灰度母版文件", "清洁掩码文件", "中间文件", "审核文件", "成品文件"):
                match = filename_pattern.fullmatch(str(target_detail.get(field, "")))
                if match:
                    max_index = max(max_index, int(match.group(1)))

        for directory in self.get_workflow_dirs().values():
            if not os.path.isdir(directory):
                continue
            for filename in os.listdir(directory):
                match = filename_pattern.fullmatch(filename)
                if match:
                    max_index = max(max_index, int(match.group(1)))
        return max_index + 1

    def update_variant(self, variant_id: str, **updates: Any) -> None:
        detail = self.get_variant(variant_id)
        if detail:
            detail.update(updates)
            self._update_mtime()

    def set_status(self, variant_id: str, new_status: str) -> None:
        detail = self.get_variant(variant_id)
        if detail and new_status in config.ALL_STATUSES:
            detail["状态"] = new_status
            self._update_mtime()

    def get_variants_by_status(self, *statuses: str) -> list[dict[str, Any]]:
        status_set = set(statuses)
        return [detail for detail in self._data["变体详情"].values() if detail.get("状态") in status_set]

    def remove_variant_record(self, variant_id: str) -> bool:
        """仅删除字形数据记录并立即保存，保留各阶段图片文件。"""
        data_backup = copy.deepcopy(self._data)
        detail = self._data["变体详情"].pop(variant_id, None)
        if not detail:
            return False
        self._deleted_variant_ids.add(variant_id)
        target_char = str(detail.get("归属字", ""))
        index = self._data["字形组索引"].get(target_char, [])
        self._data["字形组索引"][target_char] = [number for number in index if number != variant_id]
        if not self._data["字形组索引"].get(target_char):
            self._data["字形组索引"].pop(target_char, None)
        self._update_mtime()
        try:
            self.save()
        except Exception:
            self._data = data_backup
            raise
        return True

    def remove_variant(self, variant_id: str, target_char: Optional[str] = None) -> None:
        detail = self._data["变体详情"].pop(variant_id, None)
        if not detail:
            return
        self._deleted_variant_ids.add(variant_id)
        workflow_dirs = self.get_workflow_dirs()
        layer_fields = (
            ("原图", "原始文件"),
            ("灰度母版", "灰度母版文件"),
            ("清洁掩码", "清洁掩码文件"),
            ("优化预览", "中间文件"),
            ("手工审核", "审核文件"),
            ("成品", "成品文件"),
        )
        removed_paths: set[str] = set()
        for layer, field in layer_fields:
            filename = detail.get(field, "")
            file_path = resolve_safe_child_file(workflow_dirs[layer], filename)
            if file_path and file_path not in removed_paths and os.path.exists(file_path):
                try:
                    send2trash(file_path)
                    removed_paths.add(file_path)
                except OSError:
                    pass
        target_char = target_char or detail.get("归属字", "")
        index = self._data["字形组索引"].get(target_char, [])
        self._data["字形组索引"][target_char] = [number for number in index if number != variant_id]
        if not self._data["字形组索引"].get(target_char):
            self._data["字形组索引"].pop(target_char, None)
        self._update_mtime()

    def remove_char(self, char: str) -> None:
        for variant_id in list(self._data["字形组索引"].get(char, [])):
            self.remove_variant(variant_id, char)

    def get_char_variants(self, char: str) -> list[dict[str, Any]]:
        variant_ids = self._data["字形组索引"].get(char, [])
        self._dirty_variant_ids.update(variant_ids)
        return [self._data["变体详情"][number] for number in variant_ids if number in self._data["变体详情"]]

    def get_all_variants(self) -> list[dict[str, Any]]:
        self._all_variants_dirty = True
        return list(self._data["变体详情"].values())

    def get_all_chars(self) -> list[str]:
        from utils.file_utils import natural_key
        return sorted(self._data["字形组索引"].keys(), key=natural_key)

    def get_status_counts(self) -> dict[str, int]:
        stats = {status: 0 for status in config.ALL_STATUSES}
        for detail in self._data["变体详情"].values():
            status = detail.get("状态", "")
            if status in stats:
                stats[status] += 1
        return stats

    def get_total_count(self) -> int:
        return len(self._data["变体详情"])

    def save_library_summary(self, summary: Mapping[str, Any]) -> None:
        """保存可重建的字库摘要，供首页和核对流程复用。"""
        if self._database is None:
            raise RuntimeError("字库数据库尚未初始化。")
        self._database.save_summary(summary)

    def save_session(self, char: str, variant_index: int = 0) -> None:
        self._data["会话"] = {"上次编辑字": char, "变体索引": variant_index}
        self.save()

    def load_session(self) -> Optional[dict[str, Any]]:
        session = self._data.get("会话", {})
        return session if session.get("上次编辑字") else None

    def save(self) -> None:
        if self._database is None:
            raise RuntimeError("字库数据库尚未初始化。")
        dirty_ids = (
            self._data["变体详情"].keys()
            if self._all_variants_dirty
            else self._dirty_variant_ids
        )
        self._database.save_data(
            self._data,
            dirty_variant_ids=dirty_ids,
            deleted_variant_ids=self._deleted_variant_ids,
        )
        self._dirty_variant_ids.clear()
        self._deleted_variant_ids.clear()
        self._all_variants_dirty = False

    def _remove_finished_file(self, detail: dict[str, Any]) -> None:
        filename = str(detail.get("成品文件", ""))
        if is_safe_windows_filename(filename):
            summary = self._data.setdefault("整体协调", self._default_coordination_summary())
            if isinstance(summary, dict):
                summary["几何协调完成"] = False
                summary["墨色统一完成"] = False
            file_path = resolve_safe_child_file(self.get_three_dirs()[2], filename)
            if file_path:
                try:
                    os.remove(file_path)
                except OSError:
                    pass

    def _gen_variant_id(self, char: str, md5_value: str) -> str:
        encoding = hex(ord(char))[2:] if len(char) == 1 else "未分类"
        return f"{encoding}_{md5_value[:12]}"

    def _update_mtime(self) -> None:
        self._data["元数据"]["最后修改"] = self._now()

    @staticmethod
    def _now() -> str:
        return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
