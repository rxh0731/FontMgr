"""字形图片保存事务及进程异常恢复。"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from services.batch_persistence import acquire_batch_library_lock
from utils.file_utils import atomic_write_json


TRANSACTION_DIRNAME = ".fonteditor_file_transactions"
TRANSACTION_VERSION = 2
PHASE_ROLLBACK = "rollback"
PHASE_ROLLFORWARD = "rollforward"
PHASE_CLEANUP = "cleanup"

_TRANSACTION_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_MD5_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_ReplaceFunction = Callable[[str, str], Any]
_RemoveFunction = Callable[[str], Any]


@dataclass(frozen=True)
class FileChange:
    """描述一个目标文件在事务完成后的内容。"""

    target_path: str
    temporary_path: str = ""
    new_md5: str = ""
    backup_prefix: str = ".fonteditor_rollback_"


class FileTransactionCommitUncertainError(RuntimeError):
    """状态已保存，但事务无法确认进入仅清理阶段。"""


def library_root_from_paths(
    service: Any,
    workflow_dirs: Iterable[str],
) -> str:
    """取得字库根目录，并兼容只提供阶段目录的测试替身。"""
    configured = getattr(service, "ziku_dir", "")
    if isinstance(configured, (str, os.PathLike)) and os.fspath(configured):
        configured_path = os.path.abspath(os.fspath(configured))
        if os.path.isdir(configured_path):
            return configured_path
    directories = [os.path.abspath(os.fspath(path)) for path in workflow_dirs]
    if not directories:
        raise ValueError("无法确定字库目录。")
    return os.path.commonpath(directories)


def recovery_state_snapshot(
    snapshot: Any,
    variant_id: str,
    detail: dict[str, Any],
) -> dict[str, Any]:
    """补齐单字恢复快照；真实服务已有完整字段，测试替身使用安全默认值。"""
    result = copy.deepcopy(snapshot) if isinstance(snapshot, dict) else {}
    result.setdefault("变体ID", variant_id)
    result.setdefault("变体存在", True)
    result.setdefault("变体详情", copy.deepcopy(detail))
    result.setdefault("元数据", {})
    result.setdefault("整体协调", {})
    return result


def recovery_full_state_snapshot(snapshot: Any) -> dict[str, Any]:
    """封装整批协调事务的完整字库状态，供异常退出后原位恢复。"""

    if not isinstance(snapshot, dict):
        raise TypeError("完整字库状态快照必须是字典。")
    return {
        "快照类型": "完整字库",
        "完整状态": copy.deepcopy(snapshot),
    }


def recovery_variant_batch_state_snapshot(
    snapshots: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """封装只涉及若干字形的增量状态，避免事务清单复制整库。"""

    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            raise TypeError("字形批次状态快照必须是字典。")
        variant_id = str(snapshot.get("变体ID", "")).strip()
        if not variant_id:
            raise ValueError("字形批次状态快照缺少变体ID。")
        if variant_id in seen:
            raise ValueError(f"字形批次状态快照包含重复变体：{variant_id}")
        seen.add(variant_id)
        items.append(copy.deepcopy(snapshot))
    if not items:
        raise ValueError("字形批次状态快照不能为空。")
    return {
        "快照类型": "字形批次",
        "字形状态": items,
    }


class FileTransaction:
    """管理一次跨多个字形图片的可恢复保存事务。"""

    def __init__(
        self,
        root_dir: str,
        manifest_path: str,
        payload: dict[str, Any],
        *,
        replace_func: _ReplaceFunction = os.replace,
        remove_func: _RemoveFunction = os.remove,
        trusted_old_files: dict[str, tuple[int, int, int, int]] | None = None,
    ) -> None:
        self.root_dir = os.path.abspath(root_dir)
        self.manifest_path = manifest_path
        self._payload = payload
        self._replace = replace_func
        self._remove = remove_func
        self._trusted_old_files = trusted_old_files or {}

    @classmethod
    def begin(
        cls,
        root_dir: str,
        changes: Iterable[FileChange],
        old_state: dict[str, Any],
        *,
        replace_func: _ReplaceFunction = os.replace,
        remove_func: _RemoveFunction = os.remove,
    ) -> "FileTransaction":
        """先原子写入回滚清单；此方法返回前不会移动任何目标文件。"""
        root = os.path.abspath(root_dir)
        if not isinstance(old_state, dict):
            raise TypeError("图片事务的旧状态快照必须是字典。")
        transaction_id = uuid.uuid4().hex
        transaction_dir = os.path.join(root, TRANSACTION_DIRNAME)
        manifest_path = os.path.join(transaction_dir, f"{transaction_id}.json")
        operations: list[dict[str, Any]] = []
        trusted_old_files: dict[str, tuple[int, int, int, int]] = {}
        seen_targets: set[str] = set()
        for index, change in enumerate(changes):
            target_path = _validate_absolute_member(
                root,
                change.target_path,
                role="目标文件",
            )
            target_key = os.path.normcase(target_path)
            if target_key in seen_targets:
                raise ValueError(f"图片事务包含重复目标文件：{target_path}")
            seen_targets.add(target_key)
            if os.path.isdir(target_path):
                raise ValueError(f"图片事务目标不能是目录：{target_path}")

            temporary_path = ""
            new_md5 = str(change.new_md5 or "").lower()
            if change.temporary_path:
                absolute_temporary = _validate_absolute_member(
                    root,
                    change.temporary_path,
                    role="临时新文件",
                )
                if os.path.normcase(absolute_temporary) == target_key:
                    raise ValueError("临时新文件不能与目标文件相同。")
                if not os.path.isfile(absolute_temporary):
                    raise FileNotFoundError(
                        f"图片事务临时新文件不存在：{absolute_temporary}"
                    )
                temporary_path = _relative_member(root, absolute_temporary)
                if not _MD5_PATTERN.fullmatch(new_md5):
                    raise ValueError(
                        f"图片事务临时新文件缺少有效 MD5：{absolute_temporary}"
                    )
                _flush_file(absolute_temporary)
            elif new_md5:
                raise ValueError("删除文件的事务操作不能设置新文件 MD5。")

            prefix = _safe_backup_prefix(change.backup_prefix)
            suffix = os.path.splitext(target_path)[1]
            backup_name = f"{prefix}{transaction_id}_{index:03d}{suffix}"
            backup_path = os.path.join(os.path.dirname(target_path), backup_name)
            _validate_absolute_member(root, backup_path, role="备份文件")
            if os.path.exists(backup_path):
                raise FileExistsError(f"图片事务备份路径已存在：{backup_path}")
            target_existed = os.path.isfile(target_path)
            old_md5 = _compute_md5(target_path) if target_existed else ""
            if target_existed:
                trusted_old_files[target_key] = _file_identity(target_path)
            operations.append(
                {
                    "目标文件": _relative_member(root, target_path),
                    "临时新文件": temporary_path,
                    "新文件MD5": new_md5,
                    "备份文件": _relative_member(root, backup_path),
                    "原目标存在": target_existed,
                    "旧文件MD5": old_md5,
                }
            )
        if not operations:
            raise ValueError("图片事务至少需要一个文件操作。")

        payload = {
            "版本": TRANSACTION_VERSION,
            "事务ID": transaction_id,
            "创建序号": time.time_ns(),
            "阶段": PHASE_ROLLBACK,
            "旧状态": copy.deepcopy(old_state),
            "新状态": None,
            "文件操作": operations,
        }
        os.makedirs(transaction_dir, exist_ok=True)
        _write_manifest(manifest_path, payload)
        return cls(
            root,
            manifest_path,
            payload,
            replace_func=replace_func,
            remove_func=remove_func,
            trusted_old_files=trusted_old_files,
        )

    @property
    def phase(self) -> str:
        return str(self._payload["阶段"])

    def backup_targets(self) -> None:
        """把事务开始时存在的全部旧目标移动到预定备份路径。"""
        if self.phase != PHASE_ROLLBACK:
            raise RuntimeError("只有回滚阶段可以备份旧文件。")
        validation_errors = self._live_backup_validation_errors()
        if validation_errors is None:
            validation_errors = _rollback_validation_errors(
                self.root_dir,
                self._payload,
                allow_target_with_backup=False,
            )
        if validation_errors:
            raise RuntimeError("图片事务旧文件校验失败：" + "；".join(validation_errors))
        for operation in self._payload["文件操作"]:
            target_path, _temporary_path, backup_path = self._operation_paths(operation)
            if operation["原目标存在"]:
                if not os.path.exists(backup_path):
                    self._replace(target_path, backup_path)

    def mark_rollforward(self, new_state: dict[str, Any]) -> None:
        """确认旧文件均已备份，并原子切换为向前完成阶段。"""
        if self.phase != PHASE_ROLLBACK:
            raise RuntimeError("图片事务已经切换到向前完成阶段。")
        if not isinstance(new_state, dict):
            raise TypeError("图片事务的新状态快照必须是字典。")
        for operation in self._payload["文件操作"]:
            target_path, _temporary_path, backup_path = self._operation_paths(operation)
            if operation["原目标存在"] and not os.path.isfile(backup_path):
                raise RuntimeError(f"旧文件尚未完成备份：{target_path}")
            old_md5 = str(operation.get("旧文件MD5", ""))
            trusted_identity = self._trusted_old_files.get(
                os.path.normcase(target_path)
            )
            trusted_backup = (
                trusted_identity is not None
                and _try_file_identity(backup_path) == trusted_identity
            )
            if old_md5 and not trusted_backup and not _matches_md5(backup_path, old_md5):
                raise RuntimeError(f"旧文件备份内容校验失败：{target_path}")
            if os.path.exists(target_path):
                raise RuntimeError(f"目标路径尚未腾空：{target_path}")
        self._payload["新状态"] = copy.deepcopy(new_state)
        self._payload["阶段"] = PHASE_ROLLFORWARD
        _write_manifest(self.manifest_path, self._payload)

    def _live_backup_validation_errors(self) -> list[str] | None:
        """实时事务用文件标识确认旧目标；不确定时回退完整 MD5。"""

        if not self._trusted_old_files:
            return None
        errors: list[str] = []
        for operation in self._payload["文件操作"]:
            target_path, _temporary_path, backup_path = self._operation_paths(operation)
            if os.path.exists(backup_path):
                return None
            if not operation["原目标存在"]:
                if os.path.exists(target_path):
                    errors.append(f"事务开始后出现了意外目标文件：{target_path}")
                continue
            expected = self._trusted_old_files.get(os.path.normcase(target_path))
            if expected is None or _try_file_identity(target_path) != expected:
                return None
        return errors

    def install_new_files(self) -> None:
        """安装全部临时新文件；空临时路径表示提交后目标应不存在。"""
        if self.phase != PHASE_ROLLFORWARD:
            raise RuntimeError("图片事务尚未切换到向前完成阶段。")
        for operation in self._payload["文件操作"]:
            target_path, temporary_path, _backup_path = self._operation_paths(operation)
            if not temporary_path:
                if os.path.exists(target_path):
                    self._remove(target_path)
                continue
            if not os.path.isfile(temporary_path):
                expected_md5 = str(operation.get("新文件MD5", ""))
                if os.path.isfile(target_path) and _matches_md5(target_path, expected_md5):
                    continue
                if os.path.isfile(target_path):
                    raise RuntimeError(f"已安装的新文件内容校验失败：{target_path}")
                raise FileNotFoundError(f"待安装的临时新文件不存在：{temporary_path}")
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            self._replace(temporary_path, target_path)

    def update_rollforward_state(self, new_state: dict[str, Any]) -> None:
        """文件安装后用最终复核结果刷新前滚状态，保持崩溃恢复精确。"""

        if self.phase != PHASE_ROLLFORWARD:
            raise RuntimeError("只有向前完成阶段可以刷新事务状态。")
        if not isinstance(new_state, dict):
            raise TypeError("图片事务的新状态快照必须是字典。")
        self._payload["新状态"] = copy.deepcopy(new_state)
        _write_manifest(self.manifest_path, self._payload)

    def rollback(self) -> list[str]:
        """把清单切回回滚阶段并恢复旧文件；失败时保留清单和备份。"""
        errors: list[str] = []
        try:
            if self.phase != PHASE_ROLLBACK:
                self._payload["阶段"] = PHASE_ROLLBACK
                _write_manifest(self.manifest_path, self._payload)
        except Exception as exc:
            return [f"无法把图片事务切换为回滚阶段：{exc}"]
        errors.extend(
            _restore_rollback_files(
                self.root_dir,
                self._payload,
                replace_func=self._replace,
                remove_func=self._remove,
            )
        )
        if errors:
            return errors
        return _cleanup_transaction(
            self.root_dir,
            self.manifest_path,
            self._payload,
            remove_func=self._remove,
        )

    def finalize(self) -> list[str]:
        """状态已持久化后清理旧备份，并最后删除事务清单。"""
        if self.phase == PHASE_ROLLFORWARD:
            self._payload["阶段"] = PHASE_CLEANUP
            try:
                _write_manifest(self.manifest_path, self._payload)
            except Exception as exc:
                self._payload["阶段"] = PHASE_ROLLFORWARD
                raise FileTransactionCommitUncertainError(
                    "图片和状态已保存，但无法确认图片事务进入清理阶段；"
                    "已停止继续写入，请重新打开字库完成恢复。"
                ) from exc
        elif self.phase != PHASE_CLEANUP:
            return ["图片事务尚未提交，不能清理回滚依据。"]
        return _cleanup_transaction(
            self.root_dir,
            self.manifest_path,
            self._payload,
            remove_func=self._remove,
        )

    def _operation_paths(
        self,
        operation: dict[str, Any],
    ) -> tuple[str, str, str]:
        return (
            _resolve_relative_member(self.root_dir, operation["目标文件"], role="目标文件"),
            _resolve_optional_relative_member(
                self.root_dir,
                operation["临时新文件"],
                role="临时新文件",
            ),
            _resolve_relative_member(self.root_dir, operation["备份文件"], role="备份文件"),
        )


def recover_file_transactions(
    data: dict[str, Any],
    *,
    ziku_dir: str,
    json_path: str | None = None,
    persist_callback: Callable[[dict[str, Any]], None] | None = None,
) -> bool:
    """恢复字库内全部未完成图片事务，并同步其单字状态。"""
    if not isinstance(data, dict):
        raise TypeError("字库恢复基础数据必须是字典。")
    root = os.path.abspath(ziku_dir)
    transaction_dir = os.path.join(root, TRANSACTION_DIRNAME)
    if not os.path.isdir(transaction_dir):
        return False
    manifest_paths = [
        os.path.join(transaction_dir, name)
        for name in os.listdir(transaction_dir)
        if name.endswith(".json")
    ]
    if not manifest_paths:
        return False

    batch_lock = acquire_batch_library_lock(root)
    try:
        manifests = [
            (path, _read_and_validate_manifest(path, root))
            for path in manifest_paths
        ]
        manifests.sort(
            key=lambda item: (int(item[1]["创建序号"]), item[1]["事务ID"])
        )
        recovered = False
        for manifest_path, payload in manifests:
            phase = payload["阶段"]
            if phase == PHASE_CLEANUP:
                cleanup_errors = _cleanup_transaction(root, manifest_path, payload)
                if cleanup_errors:
                    raise RuntimeError(
                        f"图片事务 {payload['事务ID']} 已提交，但清理失败："
                        + "；".join(cleanup_errors)
                    )
                recovered = True
                continue
            if phase == PHASE_ROLLFORWARD:
                can_rollforward, validation_errors = _can_rollforward(root, payload)
            else:
                can_rollforward, validation_errors = False, []
            if can_rollforward:
                errors = _restore_rollforward_files(root, payload)
                state = payload["新状态"]
            else:
                errors = _restore_rollback_files(root, payload)
                state = payload["旧状态"]
                if errors and validation_errors:
                    errors = [*validation_errors, *errors]
            if errors:
                raise RuntimeError(
                    f"图片事务 {payload['事务ID']} 恢复失败："
                    + "；".join(errors)
                )
            _apply_variant_state(data, state)
            if persist_callback is not None:
                persist_callback(data)
            elif json_path:
                atomic_write_json(data, json_path, indent=None)
            else:
                raise RuntimeError("图片事务恢复缺少状态持久化入口。")
            payload["阶段"] = PHASE_CLEANUP
            _write_manifest(manifest_path, payload)
            cleanup_errors = _cleanup_transaction(root, manifest_path, payload)
            if cleanup_errors:
                raise RuntimeError(
                    f"图片事务 {payload['事务ID']} 已恢复，但清理失败："
                    + "；".join(cleanup_errors)
                )
            recovered = True
        return recovered
    finally:
        batch_lock.release()


def has_file_transaction_artifacts(ziku_dir: str) -> bool:
    """判断字库是否存在需要恢复或核查的图片事务文件。"""
    transaction_dir = os.path.join(os.path.abspath(ziku_dir), TRANSACTION_DIRNAME)
    if not os.path.isdir(transaction_dir):
        return False
    try:
        return any(name.endswith(".json") for name in os.listdir(transaction_dir))
    except OSError:
        return True


def ensure_file_transactions_ready(ziku_dir: str) -> None:
    """拒绝在同库仍有待恢复事务时继续写入；仅清理事务不阻塞。"""
    root = os.path.abspath(ziku_dir)
    transaction_dir = os.path.join(root, TRANSACTION_DIRNAME)
    if not os.path.isdir(transaction_dir):
        return
    try:
        manifest_paths = [
            os.path.join(transaction_dir, name)
            for name in os.listdir(transaction_dir)
            if name.endswith(".json")
        ]
    except OSError as exc:
        raise RuntimeError(
            "无法核查字库图片事务，请重新打开字库完成恢复。"
        ) from exc
    for manifest_path in manifest_paths:
        try:
            payload = _read_and_validate_manifest(manifest_path, root)
        except Exception as exc:
            raise RuntimeError(
                "字库存在无法核查的图片事务，请重新打开字库完成恢复。"
            ) from exc
        if payload["阶段"] != PHASE_CLEANUP:
            raise RuntimeError(
                "字库存在尚未恢复的图片事务，请重新打开字库完成恢复后再写入。"
            )


def _read_and_validate_manifest(path: str, root: str) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            wrapper = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"图片事务清单损坏：{path}：{exc}") from exc
    if not isinstance(wrapper, dict):
        raise RuntimeError(f"图片事务清单格式无效：{path}")
    payload = wrapper.get("载荷")
    checksum = wrapper.get("校验")
    if not isinstance(payload, dict) or not isinstance(checksum, str):
        raise RuntimeError(f"图片事务清单缺少载荷或校验：{path}")
    if checksum != _payload_checksum(payload):
        raise RuntimeError(f"图片事务清单校验失败：{path}")
    _validate_payload(payload, path, root)
    return payload


def _validate_payload(payload: dict[str, Any], manifest_path: str, root: str) -> None:
    try:
        version = int(payload["版本"])
        transaction_id = str(payload["事务ID"])
        creation_order = int(payload["创建序号"])
        phase = str(payload["阶段"])
        old_state = payload["旧状态"]
        new_state = payload["新状态"]
        operations = payload["文件操作"]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"图片事务清单字段无效：{manifest_path}") from exc
    if version not in (1, TRANSACTION_VERSION):
        raise RuntimeError(f"图片事务清单版本不受支持：{manifest_path}")
    if not _TRANSACTION_ID_PATTERN.fullmatch(transaction_id):
        raise RuntimeError(f"图片事务编号无效：{manifest_path}")
    if os.path.basename(manifest_path) != f"{transaction_id}.json":
        raise RuntimeError(f"图片事务文件名与事务编号不一致：{manifest_path}")
    if creation_order <= 0 or phase not in (
        PHASE_ROLLBACK,
        PHASE_ROLLFORWARD,
        PHASE_CLEANUP,
    ):
        raise RuntimeError(f"图片事务阶段或创建序号无效：{manifest_path}")
    if not isinstance(old_state, dict):
        raise RuntimeError(f"图片事务旧状态无效：{manifest_path}")
    if phase in (PHASE_ROLLFORWARD, PHASE_CLEANUP) and not isinstance(new_state, dict):
        raise RuntimeError(f"图片事务新状态无效：{manifest_path}")
    if not isinstance(operations, list) or not operations:
        raise RuntimeError(f"图片事务没有文件操作：{manifest_path}")

    seen_targets: set[str] = set()
    for operation in operations:
        if not isinstance(operation, dict):
            raise RuntimeError(f"图片事务文件操作格式无效：{manifest_path}")
        try:
            target_value = operation["目标文件"]
            temporary_value = operation["临时新文件"]
            new_md5 = operation["新文件MD5"]
            backup_value = operation["备份文件"]
            target_existed = operation["原目标存在"]
        except KeyError as exc:
            raise RuntimeError(f"图片事务文件操作字段不完整：{manifest_path}") from exc
        if not isinstance(target_existed, bool):
            raise RuntimeError(f"图片事务原目标标记无效：{manifest_path}")
        if not isinstance(new_md5, str):
            raise RuntimeError(f"图片事务新文件 MD5 无效：{manifest_path}")
        old_md5 = operation.get("旧文件MD5", "")
        if not isinstance(old_md5, str):
            raise RuntimeError(f"图片事务旧文件 MD5 无效：{manifest_path}")
        if target_existed:
            if version >= 2 and not _MD5_PATTERN.fullmatch(old_md5):
                raise RuntimeError(f"图片事务旧文件 MD5 无效：{manifest_path}")
            if old_md5 and not _MD5_PATTERN.fullmatch(old_md5):
                raise RuntimeError(f"图片事务旧文件 MD5 无效：{manifest_path}")
        elif old_md5:
            raise RuntimeError(f"原本不存在的目标含有旧文件 MD5：{manifest_path}")
        target_path = _resolve_relative_member(root, target_value, role="目标文件")
        temporary_path = _resolve_optional_relative_member(
            root,
            temporary_value,
            role="临时新文件",
        )
        backup_path = _resolve_relative_member(root, backup_value, role="备份文件")
        target_key = os.path.normcase(target_path)
        if target_key in seen_targets:
            raise RuntimeError(f"图片事务目标文件重复：{manifest_path}")
        seen_targets.add(target_key)
        if temporary_path and os.path.normcase(temporary_path) == target_key:
            raise RuntimeError(f"图片事务临时文件与目标相同：{manifest_path}")
        if temporary_path:
            if not _MD5_PATTERN.fullmatch(new_md5):
                raise RuntimeError(f"图片事务新文件 MD5 无效：{manifest_path}")
        elif new_md5:
            raise RuntimeError(f"图片事务删除操作含有新文件 MD5：{manifest_path}")
        if os.path.normcase(backup_path) in (
            target_key,
            os.path.normcase(temporary_path) if temporary_path else "",
        ):
            raise RuntimeError(f"图片事务备份路径冲突：{manifest_path}")


def _restore_rollback_files(
    root: str,
    payload: dict[str, Any],
    *,
    replace_func: _ReplaceFunction = os.replace,
    remove_func: _RemoveFunction = os.remove,
) -> list[str]:
    errors = _rollback_validation_errors(root, payload)
    if errors:
        return errors
    for operation in reversed(payload["文件操作"]):
        target_path = _resolve_relative_member(root, operation["目标文件"], role="目标文件")
        backup_path = _resolve_relative_member(root, operation["备份文件"], role="备份文件")
        if operation["原目标存在"]:
            if os.path.isfile(backup_path):
                try:
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    replace_func(backup_path, target_path)
                except OSError as exc:
                    errors.append(f"无法恢复旧文件 {target_path}：{exc}")
        else:
            if os.path.exists(target_path):
                try:
                    remove_func(target_path)
                except OSError as exc:
                    errors.append(f"无法移除新文件 {target_path}：{exc}")
    return errors


def _rollback_validation_errors(
    root: str,
    payload: dict[str, Any],
    *,
    allow_target_with_backup: bool = True,
) -> list[str]:
    """在移动任何文件前确认每个旧目标都有可信恢复来源。"""
    errors: list[str] = []
    for operation in payload["文件操作"]:
        target_path = _resolve_relative_member(root, operation["目标文件"], role="目标文件")
        backup_path = _resolve_relative_member(root, operation["备份文件"], role="备份文件")
        if not operation["原目标存在"]:
            if os.path.exists(backup_path):
                errors.append(f"原本不存在的目标出现了异常备份：{backup_path}")
            if not allow_target_with_backup and os.path.exists(target_path):
                errors.append(f"事务开始后出现了意外目标文件：{target_path}")
            continue

        expected_md5 = str(operation.get("旧文件MD5", ""))
        if os.path.isfile(backup_path):
            if expected_md5 and not _matches_md5(backup_path, expected_md5):
                errors.append(f"旧文件备份内容校验失败：{target_path}")
            if not allow_target_with_backup and os.path.exists(target_path):
                errors.append(f"目标文件和备份同时存在：{target_path}")
            continue
        if not os.path.isfile(target_path):
            errors.append(f"旧文件及其备份均不存在：{target_path}")
            continue
        if not expected_md5:
            errors.append(f"旧文件备份缺失，且清单没有旧文件校验值：{target_path}")
        elif not _matches_md5(target_path, expected_md5):
            errors.append(f"旧文件备份缺失，且目标内容不是事务开始时的旧文件：{target_path}")
    return errors


def _restore_rollforward_files(
    root: str,
    payload: dict[str, Any],
    *,
    replace_func: _ReplaceFunction = os.replace,
    remove_func: _RemoveFunction = os.remove,
) -> list[str]:
    errors: list[str] = []
    for operation in payload["文件操作"]:
        target_path = _resolve_relative_member(root, operation["目标文件"], role="目标文件")
        temporary_path = _resolve_optional_relative_member(
            root,
            operation["临时新文件"],
            role="临时新文件",
        )
        if temporary_path:
            expected_md5 = operation["新文件MD5"]
            target_matches = _matches_md5(target_path, expected_md5)
            temporary_matches = _matches_md5(temporary_path, expected_md5)
            if target_matches:
                continue
            if temporary_matches:
                try:
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    replace_func(temporary_path, target_path)
                except OSError as exc:
                    errors.append(f"无法安装新文件 {target_path}：{exc}")
            else:
                errors.append(f"新文件内容校验失败：{target_path}")
        elif os.path.exists(target_path):
            try:
                remove_func(target_path)
            except OSError as exc:
                errors.append(f"无法完成文件删除 {target_path}：{exc}")
    return errors


def _can_rollforward(root: str, payload: dict[str, Any]) -> tuple[bool, list[str]]:
    """先校验全部新内容，避免前滚一半后才发现某个文件损坏。"""
    errors: list[str] = []
    for operation in payload["文件操作"]:
        temporary_path = _resolve_optional_relative_member(
            root,
            operation["临时新文件"],
            role="临时新文件",
        )
        if not temporary_path:
            continue
        target_path = _resolve_relative_member(
            root,
            operation["目标文件"],
            role="目标文件",
        )
        expected_md5 = operation["新文件MD5"]
        if _matches_md5(target_path, expected_md5):
            continue
        if _matches_md5(temporary_path, expected_md5):
            continue
        errors.append(f"新文件及临时文件均缺失或 MD5 不匹配：{target_path}")
    return not errors, errors


def _apply_variant_state(data: dict[str, Any], snapshot: dict[str, Any]) -> None:
    if not isinstance(snapshot, dict):
        raise RuntimeError("图片事务状态快照无效。")
    if snapshot.get("快照类型") == "完整字库":
        full_state = snapshot.get("完整状态")
        if not isinstance(full_state, dict):
            raise RuntimeError("图片事务的完整字库状态快照无效。")
        data.clear()
        data.update(copy.deepcopy(full_state))
        return
    if snapshot.get("快照类型") == "字形批次":
        items = snapshot.get("字形状态")
        if not isinstance(items, list) or not items:
            raise RuntimeError("图片事务的字形批次状态快照无效。")
        for item in items:
            if not isinstance(item, dict) or item.get("快照类型") in {
                "完整字库",
                "字形批次",
            }:
                raise RuntimeError("图片事务的字形批次成员无效。")
            _apply_variant_state(data, item)
        return
    variant_id = str(snapshot.get("变体ID", ""))
    if not variant_id:
        raise RuntimeError("图片事务状态快照缺少变体ID。")
    details = data.setdefault("变体详情", {})
    if not isinstance(details, dict):
        raise RuntimeError("字库变体详情格式无效，无法恢复图片事务。")
    if snapshot.get("变体存在", True):
        detail = snapshot.get("变体详情")
        if not isinstance(detail, dict):
            raise RuntimeError("图片事务的变体详情快照无效。")
        details[variant_id] = copy.deepcopy(detail)
    else:
        details.pop(variant_id, None)
    for key in ("元数据", "整体协调"):
        value = snapshot.get(key)
        if isinstance(value, dict):
            data[key] = copy.deepcopy(value)
    group_fragment = snapshot.get("字形组索引片段")
    if isinstance(group_fragment, dict):
        groups = data.setdefault("字形组索引", {})
        if not isinstance(groups, dict):
            raise RuntimeError("字库字形组索引格式无效，无法恢复图片事务。")
        for char, variant_ids in group_fragment.items():
            if isinstance(variant_ids, list) and variant_ids:
                groups[str(char)] = copy.deepcopy(variant_ids)
            else:
                groups.pop(str(char), None)


def _cleanup_transaction(
    root: str,
    manifest_path: str,
    payload: dict[str, Any],
    *,
    remove_func: _RemoveFunction = os.remove,
) -> list[str]:
    errors: list[str] = []
    cleanup_paths: list[str] = []
    for operation in payload["文件操作"]:
        cleanup_paths.append(
            _resolve_relative_member(root, operation["备份文件"], role="备份文件")
        )
        temporary_path = _resolve_optional_relative_member(
            root,
            operation["临时新文件"],
            role="临时新文件",
        )
        if temporary_path:
            cleanup_paths.append(temporary_path)
    for path in dict.fromkeys(cleanup_paths):
        if not os.path.exists(path):
            continue
        try:
            remove_func(path)
        except OSError as exc:
            errors.append(f"无法清理事务文件 {path}：{exc}")
    if errors:
        return errors
    try:
        remove_func(manifest_path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        return [f"无法清理事务清单 {manifest_path}：{exc}"]
    try:
        os.rmdir(os.path.join(root, TRANSACTION_DIRNAME))
    except OSError:
        pass
    return []


def _write_manifest(path: str, payload: dict[str, Any]) -> None:
    wrapper = {
        "载荷": payload,
        "校验": _payload_checksum(payload),
    }
    atomic_write_json(wrapper, path, indent=None, backup_existing=False)


def _flush_file(path: str) -> None:
    """确保临时新文件内容先于事务清单持久化。"""
    with open(path, "r+b") as handle:
        os.fsync(handle.fileno())


def _matches_md5(path: str, expected_md5: str) -> bool:
    if not expected_md5 or not os.path.isfile(path):
        return False
    try:
        return _compute_md5(path) == expected_md5
    except OSError:
        return False


def _file_identity(path: str) -> tuple[int, int, int, int]:
    stat = os.stat(path, follow_symlinks=False)
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )


def _try_file_identity(path: str) -> tuple[int, int, int, int] | None:
    try:
        return _file_identity(path)
    except OSError:
        return None


def _compute_md5(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_checksum(payload: dict[str, Any]) -> str:
    packed = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(packed).hexdigest()


def _safe_backup_prefix(prefix: str) -> str:
    value = str(prefix or ".fonteditor_rollback_")
    if os.path.basename(value) != value or value in (".", ".."):
        raise ValueError("图片事务备份前缀不能包含路径。")
    return value


def _validate_absolute_member(root: str, value: Any, *, role: str) -> str:
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError(f"{role}路径无效。")
    absolute = os.path.abspath(os.fspath(value))
    _ensure_within_root(root, absolute, role=role)
    transaction_dir = os.path.join(root, TRANSACTION_DIRNAME)
    if role == "目标文件" and _is_within(transaction_dir, absolute):
        raise ValueError(f"{role}不能位于图片事务目录中：{absolute}")
    return absolute


def _relative_member(root: str, path: str) -> str:
    relative = os.path.relpath(path, root)
    if relative == ".." or relative.startswith(".." + os.sep):
        raise ValueError(f"路径不在字库目录内：{path}")
    return relative


def _resolve_relative_member(root: str, value: Any, *, role: str) -> str:
    if not isinstance(value, str) or not value or os.path.isabs(value):
        raise RuntimeError(f"图片事务{role}必须是非空相对路径。")
    normalized = os.path.normpath(value)
    if normalized == ".." or normalized.startswith(".." + os.sep):
        raise RuntimeError(f"图片事务{role}越过字库目录：{value}")
    absolute = os.path.abspath(os.path.join(root, normalized))
    _ensure_within_root(root, absolute, role=role)
    transaction_dir = os.path.join(root, TRANSACTION_DIRNAME)
    if role == "目标文件" and _is_within(transaction_dir, absolute):
        raise RuntimeError(f"图片事务{role}不能位于事务目录中：{value}")
    return absolute


def _resolve_optional_relative_member(root: str, value: Any, *, role: str) -> str:
    if value == "":
        return ""
    return _resolve_relative_member(root, value, role=role)


def _ensure_within_root(root: str, path: str, *, role: str) -> None:
    if not _is_within(root, path):
        raise ValueError(f"{role}不在字库目录内：{path}")


def _is_within(root: str, path: str) -> bool:
    normalized_root = os.path.normcase(os.path.realpath(root))
    normalized_path = os.path.normcase(os.path.realpath(path))
    try:
        return os.path.commonpath((normalized_root, normalized_path)) == normalized_root
    except ValueError:
        return False
