# glyph_service.py — 字形数据与六阶段流程管理

import copy
import os
import re
from datetime import datetime
from typing import Any, Optional, Tuple

from send2trash import send2trash

import config
from utils.file_utils import atomic_write_json, ensure_dir, validate_final_char


class GlyphService:
    """管理字库、字形记录、六层工作文件以及四阶段状态。"""

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

    def __init__(self, ziku_name: str, ziku_dir: str) -> None:
        self.ziku_name = ziku_name
        self.ziku_dir = ziku_dir
        self._json_path = os.path.join(ziku_dir, f"{ziku_name}.json")
        self._migrate_workflow_dirs()
        self._data = self._load_or_init()

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
        """读取当前数据，并兼容迁移旧版流程状态名称。"""
        if os.path.exists(self._json_path):
            from utils.file_utils import safe_read_json
            data = safe_read_json(self._json_path)
            if isinstance(data, dict) and data.get("数据版本") == 3:
                data.pop("简繁映射表", None)
                data_changed = False
                metadata = data.setdefault("元数据", {})
                if "DPI" not in metadata and "分辨率" in metadata:
                    metadata["DPI"] = metadata.pop("分辨率")
                    data_changed = True
                for detail in data.get("变体详情", {}).values():
                    old_status = detail.get("状态")
                    if old_status in self._LEGACY_STATUSES:
                        detail["状态"] = self._LEGACY_STATUSES[old_status]
                        data_changed = True
                if data_changed:
                    atomic_write_json(data, self._json_path)
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
            "整体协调": {
                "基准": {},
                "墨色基准": None,
                "几何协调完成": False,
                "墨色统一完成": False,
                "最后生成时间": "",
            },
        }

    def init_metadata(
        self,
        dpi: int = 300,
        canvas_w: int = 250,
        canvas_h: int = 250,
        width_mm: Optional[float] = None,
        height_mm: Optional[float] = None,
        output_style: Optional[str] = None,
    ) -> None:
        old_metadata = self._data.get("元数据", {})
        self._data["元数据"] = {
            "DPI": dpi,
            "画布宽": canvas_w,
            "画布高": canvas_h,
            "成品宽度毫米": round(width_mm if width_mm is not None else canvas_w / dpi * 25.4, 2),
            "成品高度毫米": round(height_mm if height_mm is not None else canvas_h / dpi * 25.4, 2),
            "成品风格": output_style or old_metadata.get("成品风格", "灰度保真"),
            "透明背景": True,
            "创建时间": old_metadata.get("创建时间") or self._now(),
            "最后修改": self._now(),
        }

    def get_metadata(self) -> dict[str, Any]:
        return dict(self._data.get("元数据", {}))

    def rename_ziku(self, new_name: str) -> str:
        """改名字库目录、数据文件和库名，返回改名后的目录。"""
        new_name = new_name.strip()
        if not new_name:
            raise ValueError("字库名称不能为空。")
        if new_name == self.ziku_name:
            return self.ziku_dir
        if any(char in new_name for char in '<>:"/\\|?*') or new_name.endswith((" ", ".")):
            raise ValueError("字库名称包含 Windows 不允许的字符，或以空格、句点结尾。")

        old_name = self.ziku_name
        old_dir = os.path.abspath(self.ziku_dir)
        parent_dir = os.path.dirname(old_dir)
        new_dir = os.path.join(parent_dir, new_name)
        old_json_path = self._json_path
        renamed_json_path = os.path.join(old_dir, f"{new_name}.json")
        final_json_path = os.path.join(new_dir, f"{new_name}.json")
        if os.path.exists(new_dir):
            raise FileExistsError(f"字库“{new_name}”已存在。")
        if os.path.exists(renamed_json_path):
            raise FileExistsError(f"数据文件“{new_name}.json”已存在。")

        self._data["库名"] = new_name
        self._data.setdefault("元数据", {})["最后修改"] = self._now()
        try:
            os.replace(old_json_path, renamed_json_path)
            os.replace(old_dir, new_dir)
            self.ziku_name = new_name
            self.ziku_dir = new_dir
            self._json_path = final_json_path
            self.save()
        except Exception:
            self._data["库名"] = old_name
            if os.path.exists(new_dir) and not os.path.exists(old_dir):
                os.replace(new_dir, old_dir)
            if os.path.exists(renamed_json_path) and not os.path.exists(old_json_path):
                os.replace(renamed_json_path, old_json_path)
            self.ziku_name = old_name
            self.ziku_dir = old_dir
            self._json_path = old_json_path
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
        self._data["整体协调"] = {"基准": {}, "最后生成时间": ""}
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
        self._update_mtime()
        return variant_id

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
        reviewed_exists = bool(reviewed_filename) and os.path.exists(
            os.path.join(workflow_dirs["手工审核"], reviewed_filename)
        )
        preview_exists = bool(preview_filename) and os.path.exists(
            os.path.join(workflow_dirs["优化预览"], preview_filename)
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
        detail = self.get_variant(variant_id)
        if not detail:
            return
        detail["成品文件"] = finished_filename
        detail["成品MD5"] = finished_md5
        detail["整体协调参数"] = adjustment_params
        detail["状态"] = config.STATUS_FINISHED
        self._update_mtime()

    def set_coordination_summary(
        self, baseline: dict[str, Any], ink_baseline: Optional[float] = None
    ) -> None:
        self._data["整体协调"] = {
            "基准": baseline,
            "墨色基准": round(float(ink_baseline), 2) if ink_baseline is not None else None,
            "几何协调完成": True,
            "墨色统一完成": ink_baseline is not None,
            "最后生成时间": self._now(),
        }
        self._update_mtime()

    def get_variant(self, variant_id: str) -> dict[str, Any]:
        return self._data["变体详情"].get(variant_id, {})

    def move_variant_to_char(self, variant_id: str, new_char: str) -> None:
        """修改单个字形的归属字符，并同步六阶段文件名。"""
        new_char = new_char.strip()
        valid, message = validate_final_char(new_char)
        if not valid:
            raise ValueError(message)
        detail = self.get_variant(variant_id)
        if not detail:
            raise ValueError("字形记录不存在。")
        old_char = str(detail.get("归属字", ""))
        if new_char == old_char:
            return

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
        rename_plan: list[tuple[str, str, str, str]] = []
        target_paths: set[str] = set()
        for layer, field in layer_fields:
            old_filename = str(detail.get(field, ""))
            if not old_filename:
                continue
            source_path = os.path.join(workflow_dirs[layer], old_filename)
            if not os.path.exists(source_path):
                continue
            extension = os.path.splitext(old_filename)[1]
            new_filename = new_base_name + extension
            target_path = os.path.join(workflow_dirs[layer], new_filename)
            normalized_target = os.path.normcase(os.path.abspath(target_path))
            if normalized_target in target_paths or (
                os.path.exists(target_path)
                and os.path.normcase(os.path.abspath(source_path)) != normalized_target
            ):
                raise FileExistsError(f"目标文件已存在：{new_filename}")
            target_paths.add(normalized_target)
            rename_plan.append((source_path, target_path, field, new_filename))

        data_backup = copy.deepcopy(self._data)
        completed_renames: list[tuple[str, str]] = []
        try:
            for source_path, target_path, _field, _filename in rename_plan:
                os.replace(source_path, target_path)
                completed_renames.append((target_path, source_path))
            old_variant_ids.remove(variant_id)
            if not old_variant_ids:
                groups.pop(old_char, None)
            groups.setdefault(new_char, []).append(variant_id)
            detail["归属字"] = new_char
            detail["变体序号"] = new_index
            for _source_path, _target_path, field, new_filename in rename_plan:
                detail[field] = new_filename
            self._update_mtime()
            self.save()
        except Exception:
            self._data = data_backup
            for target_path, source_path in reversed(completed_renames):
                if os.path.exists(target_path):
                    os.replace(target_path, source_path)
            raise

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
            file_path = os.path.join(workflow_dirs[layer], filename) if filename else ""
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
        return [self._data["变体详情"][number] for number in self._data["字形组索引"].get(char, []) if number in self._data["变体详情"]]

    def get_all_variants(self) -> list[dict[str, Any]]:
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

    def save_session(self, char: str, variant_index: int = 0) -> None:
        self._data["会话"] = {"上次编辑字": char, "变体索引": variant_index}
        self.save()

    def load_session(self) -> Optional[dict[str, Any]]:
        session = self._data.get("会话", {})
        return session if session.get("上次编辑字") else None

    def save(self) -> None:
        atomic_write_json(self._data, self._json_path)

    def _remove_finished_file(self, detail: dict[str, Any]) -> None:
        filename = detail.get("成品文件", "")
        if filename:
            file_path = os.path.join(self.get_three_dirs()[2], filename)
            if os.path.exists(file_path):
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
