"""批处理字形状态的轻量日志与分组持久化。"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from typing import TYPE_CHECKING, Any

from utils.file_utils import atomic_write_json

if TYPE_CHECKING:
    from services.glyph_service import GlyphService


JOURNAL_FILENAME = ".fonteditor_batch_state.jsonl"
JOURNAL_VERSION = 1
_LOCK_ANCHOR_FILENAME = ".fonteditor_batch_lock"
_LOCK_CONFLICT_MESSAGE = "当前字库正在执行其他批处理任务，请等待该任务结束后重试。"
_PROCESS_LOCKS: set[str] = set()
_PROCESS_LOCKS_GUARD = threading.Lock()
_LEGACY_CHECKPOINT_ITEMS = 20
_LEGACY_CHECKPOINT_SECONDS = 2.0
_MEDIUM_LIBRARY_MIN_VARIANTS = 200
_LARGE_LIBRARY_MIN_VARIANTS = 1000
_MEDIUM_CHECKPOINT_ITEMS = 50
_MEDIUM_CHECKPOINT_SECONDS = 6.0
_LARGE_CHECKPOINT_ITEMS = 100
_LARGE_CHECKPOINT_SECONDS = 12.0


class BatchJournalUncertainError(RuntimeError):
    """日志提交结果无法确认；调用方必须保留当前图片并停止批处理。"""


class _BatchLibraryLock:
    """以系统锁独占字库；进程退出时由操作系统自动释放。"""

    def __init__(self, ziku_dir: str) -> None:
        self._key = os.path.normcase(os.path.abspath(ziku_dir))
        self._handle: Any = None
        self._locked = False

    def acquire(self) -> None:
        if self._locked:
            return
        with _PROCESS_LOCKS_GUARD:
            if self._key in _PROCESS_LOCKS:
                raise RuntimeError(_LOCK_CONFLICT_MESSAGE)
            _PROCESS_LOCKS.add(self._key)
        try:
            if os.name == "nt":
                self._handle = self._acquire_windows_mutex()
            else:
                self._handle = self._acquire_posix_lock()
        except Exception:
            with _PROCESS_LOCKS_GUARD:
                _PROCESS_LOCKS.discard(self._key)
            raise
        self._locked = True

    def release(self) -> None:
        if not self._locked:
            return
        handle = self._handle
        self._handle = None
        self._locked = False
        try:
            if os.name == "nt":
                self._release_windows_mutex(handle)
            elif handle is not None:
                self._release_posix_lock(handle)
        finally:
            with _PROCESS_LOCKS_GUARD:
                _PROCESS_LOCKS.discard(self._key)

    def _acquire_windows_mutex(self) -> Any:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.argtypes = (
            wintypes.LPVOID,
            wintypes.BOOL,
            wintypes.LPCWSTR,
        )
        kernel32.CreateMutexW.restype = wintypes.HANDLE
        kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.ReleaseMutex.argtypes = (wintypes.HANDLE,)
        kernel32.ReleaseMutex.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL

        lock_digest = hashlib.sha256(self._key.encode("utf-8")).hexdigest()
        mutex_name = f"Local\\FontEditorPySide6_Batch_{lock_digest}"
        handle = kernel32.CreateMutexW(None, False, mutex_name)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        wait_result = int(kernel32.WaitForSingleObject(handle, 0))
        if wait_result in (0x00000000, 0x00000080):
            return (kernel32, handle)
        kernel32.CloseHandle(handle)
        if wait_result == 0x00000102:
            raise RuntimeError(_LOCK_CONFLICT_MESSAGE)
        raise OSError(f"无法取得字库批处理系统锁，系统返回值：{wait_result}。")

    @staticmethod
    def _release_windows_mutex(value: Any) -> None:
        if not value:
            return
        kernel32, handle = value
        try:
            kernel32.ReleaseMutex(handle)
        finally:
            kernel32.CloseHandle(handle)

    def _acquire_posix_lock(self) -> Any:
        import fcntl

        os.makedirs(self._key, exist_ok=True)
        anchor_path = os.path.join(self._key, _LOCK_ANCHOR_FILENAME)
        handle = open(anchor_path, "a+b")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise RuntimeError(_LOCK_CONFLICT_MESSAGE) from exc
        return handle

    @staticmethod
    def _release_posix_lock(handle: Any) -> None:
        import fcntl

        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def acquire_batch_library_lock(ziku_dir: str) -> _BatchLibraryLock:
    """取得字库批处理独占锁，调用方结束后必须调用 ``release``。"""
    lock = _BatchLibraryLock(ziku_dir)
    lock.acquire()
    return lock


def _default_checkpoint_limits(glyph_service: GlyphService) -> tuple[int, float]:
    """按字库规模放宽完整 JSON 检查点，逐字日志仍同步落盘。"""
    variant_count = max(0, int(glyph_service.get_total_count()))
    if variant_count >= _LARGE_LIBRARY_MIN_VARIANTS:
        return _LARGE_CHECKPOINT_ITEMS, _LARGE_CHECKPOINT_SECONDS
    if variant_count >= _MEDIUM_LIBRARY_MIN_VARIANTS:
        return _MEDIUM_CHECKPOINT_ITEMS, _MEDIUM_CHECKPOINT_SECONDS
    return _LEGACY_CHECKPOINT_ITEMS, _LEGACY_CHECKPOINT_SECONDS


def _compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _payload_checksum(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_compact_json(payload).encode("utf-8")).hexdigest()


def _apply_snapshot(data: dict[str, Any], snapshot: dict[str, Any]) -> None:
    variant_id = str(snapshot.get("变体ID", ""))
    if not variant_id:
        raise ValueError("批处理恢复记录缺少变体ID。")
    details = data.setdefault("变体详情", {})
    if not isinstance(details, dict):
        raise ValueError("字库变体详情格式无效，无法恢复批处理记录。")
    if snapshot.get("变体存在", True):
        detail = snapshot.get("变体详情")
        if not isinstance(detail, dict):
            raise ValueError("批处理恢复记录中的变体详情无效。")
        details[variant_id] = detail
    else:
        details.pop(variant_id, None)

    metadata = snapshot.get("元数据")
    coordination = snapshot.get("整体协调")
    if isinstance(metadata, dict):
        data["元数据"] = metadata
    if isinstance(coordination, dict):
        data["整体协调"] = coordination


def recover_batch_journal(
    data: dict[str, Any],
    *,
    ziku_name: str,
    ziku_dir: str,
    json_path: str,
) -> bool:
    """重放异常退出前已落盘的完整单字记录。"""
    journal_path = os.path.join(ziku_dir, JOURNAL_FILENAME)
    if not os.path.isfile(journal_path):
        return False
    batch_lock = acquire_batch_library_lock(ziku_dir)
    try:
        with open(journal_path, "rb") as handle:
            raw_data = handle.read()
        final_line_terminated = raw_data.endswith((b"\n", b"\r"))
        raw_lines = raw_data.splitlines()
        if raw_lines and not final_line_terminated:
            # 换行是单条记录的提交标志。即使最后半行恰好能解析并通过校验，
            # 也可能来自只完成部分 write 的异常退出，不能据此推进字形状态。
            raw_lines = raw_lines[:-1]

        applied = 0
        for index, raw_line in enumerate(raw_lines):
            if not raw_line.strip():
                continue
            try:
                decoded_line = raw_line.decode("utf-8")
                record = json.loads(decoded_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                # 未换行末行已在上方整体丢弃，其余损坏都必须明确报错。
                raise RuntimeError(f"批处理恢复日志第 {index + 1} 行损坏：{exc}") from exc
            try:
                payload = record["载荷"]
                checksum = str(record["校验"])
                if not isinstance(payload, dict):
                    raise ValueError("载荷不是字典")
                if int(payload.get("版本", 0)) != JOURNAL_VERSION:
                    raise ValueError("日志版本不受支持")
                if str(payload.get("库名", "")) != ziku_name:
                    raise ValueError("日志字库名称不匹配")
                if checksum != _payload_checksum(payload):
                    raise ValueError("日志校验失败")
                snapshot = payload.get("状态")
                if not isinstance(snapshot, dict):
                    raise ValueError("状态快照无效")
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f"批处理恢复日志第 {index + 1} 行损坏：{exc}") from exc
            _apply_snapshot(data, snapshot)
            applied += 1

        if applied:
            atomic_write_json(data, json_path)
        try:
            os.remove(journal_path)
        except FileNotFoundError:
            pass
        return bool(applied)
    finally:
        batch_lock.release()


class BatchPersistenceSession:
    """每字写小型恢复日志，定期合并到完整字库 JSON。"""

    def __init__(
        self,
        glyph_service: GlyphService,
        *,
        checkpoint_items: int | None = None,
        checkpoint_seconds: float | None = None,
    ) -> None:
        self._glyph = glyph_service
        self._journal_path = os.path.join(
            glyph_service.ziku_dir,
            JOURNAL_FILENAME,
        )
        if checkpoint_items is None and checkpoint_seconds is None:
            checkpoint_items, checkpoint_seconds = _default_checkpoint_limits(
                glyph_service
            )
        else:
            # 兼容只覆盖一个阈值的旧调用：另一个阈值仍使用原固定默认值。
            if checkpoint_items is None:
                checkpoint_items = _LEGACY_CHECKPOINT_ITEMS
            if checkpoint_seconds is None:
                checkpoint_seconds = _LEGACY_CHECKPOINT_SECONDS
        self._checkpoint_items = max(1, int(checkpoint_items))
        self._checkpoint_seconds = max(0.1, float(checkpoint_seconds))
        self._pending_count = 0
        self._last_checkpoint = time.monotonic()
        self._handle = None
        self._closed = False
        self._batch_lock = acquire_batch_library_lock(glyph_service.ziku_dir)

    @property
    def pending_count(self) -> int:
        return self._pending_count

    @property
    def journal_path(self) -> str:
        return self._journal_path

    def record_variant(self, variant_id: str) -> None:
        """在图片提交后持久记录该字形的新状态。"""
        if self._closed:
            raise RuntimeError("批处理持久化会话已经结束。")
        snapshot = self._glyph.snapshot_variant_state(variant_id)
        payload = {
            "版本": JOURNAL_VERSION,
            "库名": self._glyph.ziku_name,
            "状态": snapshot,
        }
        record = {
            "载荷": payload,
            "校验": _payload_checksum(payload),
        }
        handle = self._ensure_handle()
        record_bytes = (_compact_json(record) + "\n").encode("utf-8")
        start_offset = handle.seek(0, os.SEEK_END)
        try:
            written = handle.write(record_bytes)
            if written != len(record_bytes):
                raise OSError("批处理恢复日志没有完整写入。")
            handle.flush()
            os.fsync(handle.fileno())
        except Exception as exc:
            self._rollback_failed_append(handle, start_offset, exc)
        self._pending_count += 1

    def checkpoint_if_due(self) -> bool:
        """达到字数或时间阈值时合并一次完整 JSON。"""
        if not self._pending_count:
            return False
        elapsed = time.monotonic() - self._last_checkpoint
        if (
            self._pending_count < self._checkpoint_items
            and elapsed < self._checkpoint_seconds
        ):
            return False
        self.checkpoint()
        return True

    def checkpoint(self) -> None:
        """原子保存完整 JSON；失败时保留日志供下次恢复。"""
        if self._closed:
            raise RuntimeError("批处理持久化会话已经结束。")
        if not self._pending_count:
            self._last_checkpoint = time.monotonic()
            return
        self._close_handle()
        self._glyph.save()
        try:
            os.remove(self._journal_path)
        except FileNotFoundError:
            pass
        self._pending_count = 0
        self._last_checkpoint = time.monotonic()

    def finish(self) -> None:
        """任务正常结束或停止时强制合并剩余记录。"""
        if self._closed:
            return
        try:
            self.checkpoint()
        finally:
            self._close_handle()
            self._batch_lock.release()
            self._closed = True

    def leave_for_recovery(self) -> None:
        """关闭文件但保留日志，供异常退出恢复或回归测试使用。"""
        if self._closed:
            return
        self._close_handle()
        self._batch_lock.release()
        self._closed = True

    def _ensure_handle(self):
        if self._handle is None:
            os.makedirs(os.path.dirname(self._journal_path), exist_ok=True)
            self._handle = open(
                self._journal_path,
                "a+b",
                buffering=0,
            )
        return self._handle

    def _rollback_failed_append(
        self,
        handle: Any,
        start_offset: int,
        original_error: Exception,
    ) -> None:
        """截断未确认的日志记录，避免下次启动误重放失败状态。"""
        rollback_error: Exception | None = None
        try:
            handle.truncate(start_offset)
            handle.flush()
            os.fsync(handle.fileno())
        except Exception as exc:
            rollback_error = exc
        finally:
            self._close_handle()
        if rollback_error is not None:
            raise BatchJournalUncertainError(
                "批处理状态日志写入失败，且未能安全回退失败记录："
                f"写入错误={original_error}；回退错误={rollback_error}"
            ) from original_error
        raise original_error

    def _close_handle(self) -> None:
        if self._handle is None:
            return
        try:
            self._handle.close()
        finally:
            self._handle = None

    def __del__(self) -> None:
        try:
            self._close_handle()
            self._batch_lock.release()
        except Exception:
            pass
