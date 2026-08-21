"""手工审核工作台。"""

from __future__ import annotations

import copy
import hashlib
import math
import os
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from threading import Event
from typing import Any, Callable

import numpy as np

from PySide6.QtCore import (
    QEvent,
    QObject,
    QPoint,
    QRunnable,
    QSignalBlocker,
    QSize,
    Qt,
    QThreadPool,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QCursor,
    QIcon,
    QImage,
    QKeyEvent,
    QKeySequence,
    QPainter,
    QPixmap,
)
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

import config
from core.transform_renderer import (
    TransformLimits,
    alpha_bounds,
    calculate_transform_geometry,
    compose_rgba_on_canvas,
    place_transform,
    render_transformed_rgba,
)
from data.log_manager import buffered_log_writes, write_log
from services.batch_persistence import (
    BatchJournalUncertainError,
    BatchPersistenceSession,
    acquire_batch_library_lock,
)
from services.file_transaction_recovery import (
    FileChange,
    FileTransaction,
    FileTransactionCommitUncertainError,
    ensure_file_transactions_ready,
    library_root_from_paths,
    recovery_state_snapshot,
)
from services.glyph_service import GlyphService
from services.library_summary_service import summarize_glyph_service
from services.workflow_status_service import (
    INK_STATUS_NOT_APPLICABLE,
    MARKER_STRUCTURE_REVIEW,
    MARKER_UNSAVED,
    PHASE_FILTER_ALL,
    PHASE_REVIEW,
    PHASE_STATUS_COLORS,
    REVIEW_STATUS_FILTERS,
    STAGE_COMPLETED,
    STAGE_PENDING_COORDINATION,
    STAGE_PENDING_OPTIMIZATION,
    STAGE_PENDING_REVIEW,
    STATUS_REVIEWED,
    WorkflowStatus,
    WorkflowStageProjection,
    project_stage_status,
    resolve_safe_stage_file,
    resolve_workflow_status,
)
from ui.workers import FunctionWorker, log_background_exception
from ui.widgets.adjustable_tree_columns import AdjustableTreeColumns
from ui.widgets.glyph_rename_dialog import run_glyph_rename_dialog
from ui.widgets.two_line_status_delegate import (
    TwoLineStatusDelegate,
    set_two_line_status,
)
from ui.widgets.review_canvas import ReviewCanvas
from utils.batch_observability import BatchTiming, ProgressThrottle, format_elapsed_time
from utils.file_utils import ensure_dir, natural_key, pinyin_natural_key


class _BulkReviewSignals(QObject):
    """批量审核任务的跨线程通知。"""

    progress = Signal(object)
    finished = Signal(object)
    failed = Signal(str)


class _BulkReviewWorker(QRunnable):
    """在线程池中逐字完成审核，并实时报告进度。"""

    def __init__(
        self,
        function: Callable[
            [Callable[[dict[str, Any]], None], Callable[[], bool]],
            dict[str, Any],
        ],
    ) -> None:
        super().__init__()
        self._function = function
        self._cancel_event = Event()
        self.signals = _BulkReviewSignals()

    def request_cancel(self) -> None:
        """请求任务在当前单字事务完成后安全停止。"""
        self._cancel_event.set()

    def is_cancel_requested(self) -> bool:
        return self._cancel_event.is_set()

    @Slot()
    def run(self) -> None:
        try:
            with buffered_log_writes():
                result = self._function(
                    self.signals.progress.emit,
                    self.is_cancel_requested,
                )
        except Exception as exc:
            log_background_exception("整库手工审核")
            try:
                self.signals.failed.emit(str(exc))
            except RuntimeError:
                pass
        else:
            try:
                self.signals.finished.emit(result)
            except RuntimeError:
                pass


def _qimage_to_rgba(image: QImage) -> np.ndarray:
    """将 QImage 复制成后台线程可独立持有的 RGBA 数组。"""
    converted = image.convertToFormat(QImage.Format.Format_RGBA8888)
    height = converted.height()
    stride = converted.bytesPerLine()
    raw = np.frombuffer(
        converted.constBits(),
        dtype=np.uint8,
        count=stride * height,
    ).reshape(height, stride)
    return raw[:, : converted.width() * 4].reshape(
        height,
        converted.width(),
        4,
    ).copy()


def _rgba_to_qimage(pixels: np.ndarray) -> QImage:
    """将独立 RGBA 数组转换成持有自身内存的 QImage。"""
    contiguous = np.ascontiguousarray(pixels, dtype=np.uint8)
    if contiguous.ndim != 3 or contiguous.shape[2] != 4:
        raise ValueError("RGBA 像素数组尺寸无效。")
    height, width, _channels = contiguous.shape
    return QImage(
        contiguous.data,
        width,
        height,
        int(contiguous.strides[0]),
        QImage.Format.Format_RGBA8888,
    ).copy()


def _to_review_image(image: QImage) -> QImage:
    """将白底灰度稿转换成透明底深色字稿。"""
    rgba = image.convertToFormat(QImage.Format.Format_ARGB32)
    if image.hasAlphaChannel():
        return rgba
    alpha = image.convertToFormat(QImage.Format.Format_Grayscale8)
    alpha.invertPixels(QImage.InvertMode.InvertRgb)
    result = QImage(image.size(), QImage.Format.Format_ARGB32)
    result.fill(Qt.GlobalColor.black)
    result.setAlphaChannel(alpha)
    return result


def _finite_number(value: object, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _normalized_distort(value: object) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 8:
        return [0.0] * 8
    return [_finite_number(item, 0.0) for item in value]


def _render_review_source(
    image: QImage,
    canvas_size: tuple[int, int],
    transform: dict[str, Any] | None,
) -> tuple[QImage, tuple[int, int]]:
    """按手工审核画布的保存规则烘焙自动优化稿。"""
    source = _to_review_image(image)
    canvas_width, canvas_height = canvas_size
    image_origin = (
        (canvas_width - source.width()) / 2.0,
        (canvas_height - source.height()) / 2.0,
    )
    params = transform if isinstance(transform, dict) else {}
    scale = max(0.05, min(5.0, _finite_number(params.get("缩放"), 1.0)))
    stretch_w = max(0.05, min(5.0, _finite_number(params.get("拉伸W"), 1.0)))
    stretch_h = max(0.05, min(5.0, _finite_number(params.get("拉伸H"), 1.0)))
    rotation = _finite_number(params.get("旋转"), 0.0)
    offset = (
        _finite_number(params.get("偏移X"), 0.0),
        _finite_number(params.get("偏移Y"), 0.0),
    )
    distort = _normalized_distort(params.get("扭曲"))
    is_identity = (
        math.isclose(scale, 1.0)
        and math.isclose(stretch_w, 1.0)
        and math.isclose(stretch_h, 1.0)
        and math.isclose(rotation, 0.0)
        and math.isclose(offset[0], 0.0)
        and math.isclose(offset[1], 0.0)
        and all(math.isclose(value, 0.0) for value in distort)
    )
    if is_identity and source.size() == QSize(canvas_width, canvas_height):
        return source, (0, 0)

    pixels = _qimage_to_rgba(source)
    limits = TransformLimits(max_dimension=16_384, max_pixels=64 * 1024 * 1024)
    if is_identity:
        rendered = compose_rgba_on_canvas(
            pixels,
            image_origin,
            canvas_size,
            expand_symmetric=True,
            limits=limits,
        )
        return (
            _rgba_to_qimage(rendered.pixels).convertToFormat(
                QImage.Format.Format_ARGB32
            ),
            rendered.geometry.grid_origin,
        )

    bounds = alpha_bounds(pixels)
    if bounds is None:
        bounds = (0, 0, source.width(), source.height())
    left, top, right, bottom = bounds
    cropped = pixels[top:bottom, left:right].copy()
    geometry = calculate_transform_geometry(
        (cropped.shape[1], cropped.shape[0]),
        scale_x=scale * stretch_w,
        scale_y=scale * stretch_h,
        rotation=rotation,
        distort=distort,
        limits=limits,
    )
    source_center = (
        image_origin[0] + left + cropped.shape[1] / 2.0,
        image_origin[1] + top + cropped.shape[0] / 2.0,
    )
    placement = place_transform(geometry, source_center, offset)
    transformed = render_transformed_rgba(
        cropped,
        geometry,
        force_rotation=abs(rotation) > 1e-9,
    )
    rendered = compose_rgba_on_canvas(
        transformed,
        placement.origin,
        canvas_size,
        expand_symmetric=True,
        limits=limits,
    )
    return (
        _rgba_to_qimage(rendered.pixels).convertToFormat(
            QImage.Format.Format_ARGB32
        ),
        rendered.geometry.grid_origin,
    )


def _file_md5(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _has_visible_ink(image: QImage) -> bool:
    """确认审核稿至少包含一个可见的非白色前景像素。"""
    return _effective_ink_bounds(image) is not None


def _effective_ink_bounds(image: QImage) -> tuple[int, int, int, int] | None:
    """返回有效非透明文字的左上及右下开区间包围盒。"""
    return _effective_ink_bounds_from_rgba(_qimage_to_rgba(image))


def _effective_ink_bounds_from_rgba(
    pixels: np.ndarray,
) -> tuple[int, int, int, int] | None:
    """直接扫描已有 RGBA 数组，避免尺寸归一阶段重复转换图像。"""
    pixels = np.asarray(pixels, dtype=np.uint8).astype(np.float32)
    alpha = pixels[..., 3] / 255.0
    luminance = (
        pixels[..., 0] * 0.299
        + pixels[..., 1] * 0.587
        + pixels[..., 2] * 0.114
    )
    foreground = (255.0 - luminance) * alpha > 1.0
    rows = np.flatnonzero(np.any(foreground, axis=1))
    columns = np.flatnonzero(np.any(foreground, axis=0))
    if rows.size == 0 or columns.size == 0:
        return None
    return (
        int(columns[0]),
        int(rows[0]),
        int(columns[-1]) + 1,
        int(rows[-1]) + 1,
    )


def _normalize_initial_review_image(
    image: QImage,
    canvas_size: tuple[int, int],
    output_origin: tuple[int, int],
    pixels: np.ndarray,
    bounds: tuple[int, int, int, int] | None,
) -> tuple[
    QImage,
    tuple[int, int],
    tuple[int, int, int, int] | None,
]:
    """把异常大小的首次自动稿等比烘焙到田字格的 95%。"""
    if bounds is None:
        return image, output_origin, None
    canvas_width, canvas_height = canvas_size
    left, top, right, bottom = bounds
    width = right - left
    height = bottom - top
    occupancy = max(width / canvas_width, height / canvas_height)
    if 0.60 <= occupancy <= 1.20:
        return image, output_origin, bounds

    scale = 0.95 / occupancy
    cropped = pixels[top:bottom, left:right].copy()
    limits = TransformLimits(max_dimension=16_384, max_pixels=64 * 1024 * 1024)
    geometry = calculate_transform_geometry(
        (width, height),
        scale_x=scale,
        scale_y=scale,
        limits=limits,
    )
    source_center = (
        left + width / 2.0 - output_origin[0],
        top + height / 2.0 - output_origin[1],
    )
    placement = place_transform(geometry, source_center, (0.0, 0.0))
    transformed = render_transformed_rgba(cropped, geometry)
    rendered = compose_rgba_on_canvas(
        transformed,
        placement.origin,
        canvas_size,
        expand_symmetric=True,
        limits=limits,
    )
    normalized = _rgba_to_qimage(rendered.pixels).convertToFormat(
        QImage.Format.Format_ARGB32
    )
    return (
        normalized,
        rendered.geometry.grid_origin,
        _effective_ink_bounds_from_rgba(rendered.pixels),
    )


def _prepare_review_source(
    image: QImage,
    canvas_size: tuple[int, int],
    transform: dict[str, Any] | None,
    *,
    normalize_initial: bool,
    include_bounds: bool = False,
) -> (
    tuple[QImage, tuple[int, int]]
    | tuple[
        QImage,
        tuple[int, int],
        tuple[int, int, int, int] | None,
    ]
):
    """按交互和批量共用路径烘焙来源，并按需执行首次尺寸归一。"""
    rendered, output_origin = _render_review_source(image, canvas_size, transform)
    bounds: tuple[int, int, int, int] | None = None
    if normalize_initial:
        pixels = _qimage_to_rgba(rendered)
        bounds = _effective_ink_bounds_from_rgba(pixels)
        rendered, output_origin, bounds = _normalize_initial_review_image(
            rendered,
            canvas_size,
            output_origin,
            pixels,
            bounds,
        )
    elif include_bounds:
        bounds = _effective_ink_bounds(rendered)
    if include_bounds:
        return rendered, output_origin, bounds
    return rendered, output_origin


def _reserve_review_backup(target_path: str, prefix: str) -> str:
    """在同目录安全预留旧文件；移动失败时不把空占位文件当成备份。"""
    descriptor, backup_path = tempfile.mkstemp(
        prefix=prefix,
        suffix=Path(target_path).suffix,
        dir=os.path.dirname(target_path),
    )
    os.close(descriptor)
    try:
        os.replace(target_path, backup_path)
    except Exception:
        try:
            os.remove(backup_path)
        except OSError:
            pass
        raise
    return backup_path


def _rollback_review_files(
    output_path: str,
    installed: bool,
    backup_path: str,
    finished_path: str,
    finished_backup: str,
) -> list[str]:
    """恢复审核事务文件，无法恢复的备份必须留在原处。"""
    errors: list[str] = []
    if backup_path and os.path.exists(backup_path):
        try:
            os.replace(backup_path, output_path)
        except OSError as exc:
            errors.append(f"无法恢复旧审核稿 {output_path}：{exc}")
    elif installed and os.path.exists(output_path):
        try:
            os.remove(output_path)
        except OSError as exc:
            errors.append(f"无法移除新审核稿 {output_path}：{exc}")
    if finished_backup and os.path.exists(finished_backup):
        try:
            os.replace(finished_backup, finished_path)
        except OSError as exc:
            errors.append(f"无法恢复旧成品 {finished_path}：{exc}")
    return errors


def _review_source(
    detail: dict[str, Any],
    directories: dict[str, str],
) -> tuple[str, bool]:
    reviewed_name = str(detail.get("审核文件", "") or "")
    if reviewed_name:
        reviewed_path = resolve_safe_stage_file(
            directories["手工审核"],
            reviewed_name,
        )
        if reviewed_path:
            return reviewed_path, False
        raise FileNotFoundError("记录的人工修订稿不可用")
    preview_path = resolve_safe_stage_file(
        directories["优化预览"],
        detail.get("中间文件"),
    )
    if preview_path:
        return preview_path, True
    raise FileNotFoundError("找不到人工修订稿或自动优化稿")


def _save_and_approve_review(
    service: GlyphService,
    variant_id: str,
    canvas_size: tuple[int, int],
    dpi: int,
    *,
    persistence: BatchPersistenceSession | None = None,
) -> str:
    """以单字事务保存审核稿，并与批处理共享字库独占边界。"""
    if persistence is not None:
        return _save_and_approve_review_locked(
            service,
            variant_id,
            canvas_size,
            dpi,
            persistence=persistence,
        )
    directories = service.get_workflow_dirs()
    library_lock = acquire_batch_library_lock(
        library_root_from_paths(service, directories.values())
    )
    try:
        return _save_and_approve_review_locked(
            service,
            variant_id,
            canvas_size,
            dpi,
            persistence=None,
        )
    finally:
        library_lock.release()


def _save_and_approve_review_locked(
    service: GlyphService,
    variant_id: str,
    canvas_size: tuple[int, int],
    dpi: int,
    *,
    persistence: BatchPersistenceSession | None = None,
) -> str:
    """调用方持有字库独占锁时完成审核稿和状态提交。"""
    detail = service.get_variant(variant_id)
    if not detail:
        raise ValueError("字形记录不存在")
    directories = service.get_workflow_dirs()
    ensure_file_transactions_ready(
        library_root_from_paths(service, directories.values())
    )
    source_path, apply_transform = _review_source(detail, directories)
    source_image = QImage(source_path)
    if source_image.isNull():
        raise ValueError(f"无法读取字形图像：{os.path.basename(source_path)}")
    output_dir = directories["手工审核"]
    ensure_dir(output_dir)
    output_image: QImage | None = None
    output_origin: tuple[int, int] | None = None
    if apply_transform:
        prepared = _prepare_review_source(
            source_image,
            canvas_size,
            detail.get("变换参数"),
            normalize_initial=(
                detail.get("状态") == config.STATUS_PENDING_MANUAL_REVIEW
            ),
            include_bounds=True,
        )
        output_image, output_origin, ink_bounds = prepared
        filename = Path(source_path).stem + ".png"
        output_path = os.path.join(output_dir, filename)
        md5_value = ""
    else:
        # 人工稿已经是烘焙后的最终像素，批量审核只需校验并推进状态。
        ink_bounds = _effective_ink_bounds(source_image)
        filename = os.path.basename(source_path)
        output_path = source_path
        md5_value = _file_md5(source_path)
    if ink_bounds is None:
        raise ValueError("字形图像没有有效文字前景")

    detail_backup = copy.deepcopy(detail)
    raw_state_backup = service.snapshot_variant_state(variant_id)
    state_backup = recovery_state_snapshot(
        raw_state_backup,
        variant_id,
        detail_backup,
    )
    temporary_path = ""
    finished_path = ""
    reviewed_name = os.path.basename(str(detail.get("审核文件", "")))
    finished_name = os.path.basename(str(detail.get("成品文件", "")))
    if finished_name:
        finished_path = os.path.join(directories["成品"], finished_name)
    transaction: FileTransaction | None = None
    direct_installed = False
    state_persisted = False
    try:
        if apply_transform:
            if output_image is None:
                raise RuntimeError("自动优化稿没有生成可保存的审核图像")
            descriptor, temporary_path = tempfile.mkstemp(
                prefix=".fonteditor_review_",
                suffix=".png",
                dir=output_dir,
            )
            os.close(descriptor)
            output_image.setDotsPerMeterX(round(dpi / 0.0254))
            output_image.setDotsPerMeterY(round(dpi / 0.0254))
            if not output_image.save(temporary_path, "PNG"):
                raise OSError("无法写入临时人工修订稿。")
            md5_value = _file_md5(temporary_path)

        changes: list[FileChange] = []
        if apply_transform:
            changes.append(
                FileChange(
                    target_path=output_path,
                    temporary_path=temporary_path,
                    new_md5=md5_value,
                    backup_prefix=".fonteditor_review_rollback_",
                )
            )
        if finished_path:
            changes.append(
                FileChange(
                    target_path=finished_path,
                    backup_prefix=".fonteditor_finished_rollback_",
                )
            )
        # SQLite 逐字提交失败时必须依靠独立清单恢复，所有图片变更都进入事务。
        requires_transaction = bool(changes)
        if changes and requires_transaction:
            transaction = FileTransaction.begin(
                library_root_from_paths(service, directories.values()),
                changes,
                state_backup,
            )
            transaction.backup_targets()

        service.mark_manual_saved(
            variant_id,
            filename,
            md5_value,
            edited=not apply_transform,
        )
        saved_detail = service.get_variant(variant_id)
        params = service.default_transform_params()
        if output_origin is not None:
            params["图像原点"] = [-int(output_origin[0]), -int(output_origin[1])]
        else:
            existing_params = detail.get("变换参数", {})
            existing_origin = (
                existing_params.get("图像原点")
                if isinstance(existing_params, dict)
                else None
            )
            if isinstance(existing_origin, (list, tuple)) and len(existing_origin) == 2:
                params["图像原点"] = [
                    round(_finite_number(existing_origin[0], 0.0)),
                    round(_finite_number(existing_origin[1], 0.0)),
                ]
            else:
                params["图像原点"] = [
                    round((canvas_size[0] - source_image.width()) / 2.0),
                    round((canvas_size[1] - source_image.height()) / 2.0),
                ]
        saved_detail["变换参数"] = params
        if not service.approve_manual_review(variant_id):
            raise RuntimeError("人工审核稿保存后仍无法审核通过")
        if transaction is not None:
            new_state = recovery_state_snapshot(
                service.snapshot_variant_state(variant_id),
                variant_id,
                saved_detail,
            )
            transaction.mark_rollforward(new_state)
            transaction.install_new_files()
        elif apply_transform:
            os.replace(temporary_path, output_path)
            direct_installed = True
            temporary_path = ""
        if persistence is None:
            service.save()
        else:
            try:
                persistence.record_variant(variant_id)
            except BatchJournalUncertainError:
                # 数据库提交结果未知，保留图片和清单，由启动恢复统一裁决。
                state_persisted = True
                raise
        state_persisted = True
        if transaction is not None:
            cleanup_errors = transaction.finalize()
            if cleanup_errors:
                write_log(
                    "手工审核图片事务已提交，清理将于下次打开继续｜"
                    + "；".join(cleanup_errors)
                )
    except Exception as exc:
        if state_persisted:
            raise
        if isinstance(raw_state_backup, dict):
            service.restore_variant_state(raw_state_backup)
        # 保留调用方已经持有的详情字典引用，也兼容测试替身不执行恢复。
        detail.clear()
        detail.update(detail_backup)
        rollback_errors = transaction.rollback() if transaction is not None else []
        if transaction is None and direct_installed:
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
            except OSError as rollback_exc:
                rollback_errors.append(
                    f"无法移除未提交的新审核稿 {output_path}：{rollback_exc}"
                )
        if rollback_errors:
            raise RuntimeError(
                "手工审核保存失败，且回滚未完全完成："
                + "；".join(rollback_errors)
                + f"；图片事务清单已保留：{transaction.manifest_path if transaction else '无'}"
            ) from exc
        raise
    finally:
        if transaction is None and temporary_path and os.path.exists(temporary_path):
            try:
                os.remove(temporary_path)
            except OSError:
                pass
    return filename


def _save_interactive_review(
    service: GlyphService,
    variant_id: str,
    output_image: QImage,
    filename: str,
    output_origin: tuple[int, int],
    dpi: int,
    *,
    approve: bool = False,
) -> str:
    """以持久图片事务保存交互修改稿，并可在同次事务中审核通过。"""

    timing = BatchTiming()
    directories = service.get_workflow_dirs()
    root = library_root_from_paths(service, directories.values())
    library_lock = acquire_batch_library_lock(root)
    transaction: FileTransaction | None = None
    temporary_path = ""
    state_persisted = False
    try:
        ensure_file_transactions_ready(root)
        ensure_dir(directories["手工审核"])
        detail = service.get_variant(variant_id)
        if not detail:
            raise ValueError("字形记录不存在")
        detail_backup = copy.deepcopy(detail)
        old_state = recovery_state_snapshot(
            service.snapshot_variant_state(variant_id),
            variant_id,
            detail_backup,
        )
        output_path = os.path.join(directories["手工审核"], filename)
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=".fonteditor_review_",
            suffix=".png",
            dir=directories["手工审核"],
        )
        os.close(descriptor)
        output_image.setDotsPerMeterX(round(dpi / 0.0254))
        output_image.setDotsPerMeterY(round(dpi / 0.0254))
        stage_started = time.perf_counter()
        if not output_image.save(temporary_path, "PNG"):
            raise OSError("无法写入临时人工修订稿。")
        timing.add("PNG编码", time.perf_counter() - stage_started)
        stage_started = time.perf_counter()
        md5_value = _file_md5(temporary_path)
        timing.add("文件摘要", time.perf_counter() - stage_started)

        changes = [
            FileChange(
                target_path=output_path,
                temporary_path=temporary_path,
                new_md5=md5_value,
                backup_prefix=".fonteditor_review_rollback_",
            )
        ]
        finished_name = os.path.basename(str(detail.get("成品文件", "")))
        if finished_name:
            changes.append(
                FileChange(
                    target_path=os.path.join(directories["成品"], finished_name),
                    backup_prefix=".fonteditor_finished_rollback_",
                )
            )
        stage_started = time.perf_counter()
        transaction = FileTransaction.begin(root, changes, old_state)
        transaction.backup_targets()

        service.mark_manual_saved(variant_id, filename, md5_value)
        saved_detail = service.get_variant(variant_id)
        params = service.default_transform_params()
        params["图像原点"] = [int(output_origin[0]), int(output_origin[1])]
        saved_detail["变换参数"] = params
        if approve and not service.approve_manual_review(variant_id):
            raise RuntimeError("人工审核稿保存后仍无法审核通过")
        new_state = recovery_state_snapshot(
            service.snapshot_variant_state(variant_id),
            variant_id,
            saved_detail,
        )
        transaction.mark_rollforward(new_state)
        transaction.install_new_files()
        temporary_path = ""
        timing.add("图片事务", time.perf_counter() - stage_started)
        stage_started = time.perf_counter()
        service.save()
        timing.add("字库索引", time.perf_counter() - stage_started)
        state_persisted = True
        stage_started = time.perf_counter()
        cleanup_errors = transaction.finalize()
        timing.add("事务清理", time.perf_counter() - stage_started)
        if cleanup_errors:
            write_log(
                "手工审核图片事务已提交，清理将于下次打开继续｜"
                + "；".join(cleanup_errors)
            )
        write_log(
            timing.format_summary(
                "手工审核保存",
                {"字形": 1, "审核通过": int(approve)},
            )
        )
        return output_path
    except Exception:
        if not state_persisted:
            try:
                service.restore_variant_state(old_state)
            except (NameError, TypeError, ValueError):
                pass
            if transaction is not None:
                rollback_errors = transaction.rollback()
                if rollback_errors:
                    raise RuntimeError(
                        "手工审核保存失败，且图片事务回滚未完全完成："
                        + "；".join(rollback_errors)
                    )
        raise
    finally:
        if transaction is None and temporary_path and os.path.exists(temporary_path):
            try:
                os.remove(temporary_path)
            except OSError:
                pass
        library_lock.release()


def _save_interactive_review_in_worker(
    library_name: str,
    library_path: str,
    variant_id: str,
    output_image: QImage,
    filename: str,
    output_origin: tuple[int, int],
    dpi: int,
    *,
    approve: bool,
    reusable: tuple[str, str, bool] | None = None,
) -> dict[str, Any]:
    """在线程中保存单字，并返回供页面合并的局部状态。"""

    if reusable is None:
        service = GlyphService.open(library_name, library_path)
        output_path = _save_interactive_review(
            service,
            variant_id,
            output_image,
            filename,
            output_origin,
            dpi,
            approve=approve,
        )
    else:
        library_lock = acquire_batch_library_lock(os.path.abspath(library_path))
        try:
            ensure_file_transactions_ready(os.path.abspath(library_path))
            # 必须在取得独占锁后重新读取数据库，避免使用等待锁期间过期的状态。
            service = GlyphService.open(library_name, library_path)
            reusable_path, expected_md5, edited = reusable
            detail = service.get_variant(variant_id)
            reviewed_path = resolve_safe_stage_file(
                service.get_workflow_dirs()["手工审核"],
                filename,
            )
            if (
                not detail
                or not reviewed_path
                or os.path.normcase(reviewed_path) != os.path.normcase(reusable_path)
                or _file_md5(reviewed_path) != expected_md5
            ):
                raise RuntimeError("原审核稿已变化，请重新载入后再审核")
            timing = BatchTiming()
            with timing.measure("状态提交"):
                service.mark_manual_saved(
                    variant_id,
                    filename,
                    expected_md5,
                    edited=edited,
                )
                if approve and not service.approve_manual_review(variant_id):
                    raise RuntimeError("当前审核稿无法审核通过")
                service.save()
            write_log(
                timing.format_summary(
                    "手工审核复用",
                    {"字形": 1, "复用PNG": 1},
                )
            )
            output_path = reviewed_path
        finally:
            library_lock.release()
    return {
        "字形状态": service.snapshot_variant_state(variant_id),
        "输出路径": output_path,
        "文件名": filename,
        "审核通过": approve,
    }


def _run_bulk_review(
    library_name: str,
    library_path: str,
    variant_ids: list[str],
    progress_callback: Callable[[dict[str, Any]], None],
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """在后台逐字审核；单字失败不影响后续字形。"""
    service = GlyphService.open(library_name, library_path)
    metadata = service.get_metadata()
    canvas_size = (
        max(1, int(metadata.get("画布宽", 250) or 250)),
        max(1, int(metadata.get("画布高", 250) or 250)),
    )
    dpi = max(1, int(metadata.get("DPI", 300) or 300))
    result: dict[str, Any] = {
        "成功": 0,
        "跳过": 0,
        "失败": 0,
        "失败详情": [],
        "已停止": False,
        "未处理": 0,
    }
    total = len(variant_ids)
    handled_count = 0
    persistence = BatchPersistenceSession(service)
    timing = BatchTiming()
    progress = ProgressThrottle(progress_callback)
    try:
        for current, variant_id in enumerate(variant_ids, start=1):
            if cancel_check is not None and cancel_check():
                result["已停止"] = True
                break
            detail = service.get_variant(variant_id)
            char = str(detail.get("归属字", variant_id.split("-")[0])) if detail else variant_id
            source_name = str(
                detail.get("审核文件") or detail.get("中间文件") or variant_id
            ) if detail else variant_id
            label = f"{char} · {os.path.basename(source_name)}"
            progress.emit(
                {
                    "当前": current,
                    "已处理": current - 1,
                    "总数": total,
                    "字形": label,
                    "阶段": "开始",
                    "结果": "",
                    "原因": "",
                },
                stage="审核处理",
            )
            if cancel_check is not None and cancel_check():
                result["已停止"] = True
                break
            outcome = "成功"
            reason = ""
            if not detail or detail.get("状态") != config.STATUS_PENDING_MANUAL_REVIEW:
                result["跳过"] += 1
                outcome = "跳过"
            else:
                try:
                    with timing.measure("图像与审核保存"):
                        _save_and_approve_review(
                            service,
                            variant_id,
                            canvas_size,
                            dpi,
                            persistence=persistence,
                        )
                except (
                    BatchJournalUncertainError,
                    FileTransactionCommitUncertainError,
                ):
                    # 当前记录需要在下次打开时由图片事务裁决，后续字形
                    # 不得继续处理或触发数据库提交。
                    raise
                except Exception as exc:
                    reason = str(exc) or exc.__class__.__name__
                    result["失败"] += 1
                    result["失败详情"].append((variant_id, reason))
                    outcome = "失败"
                else:
                    result["成功"] += 1
            handled_count += 1
            progress.emit(
                {
                    "当前": current,
                    "已处理": current,
                    "总数": total,
                    "字形": label,
                    "阶段": "完成",
                    "结果": outcome,
                    "原因": reason,
                },
                stage="审核处理",
            )
            with timing.measure("状态提交"):
                persistence.checkpoint_if_due()
            if cancel_check is not None and cancel_check():
                result["已停止"] = True
                break
    except BaseException:
        # 循环或提交异常时不重试保存，避免掩盖原异常；图片事务留待重开恢复。
        try:
            persistence.leave_for_recovery()
        except Exception:
            pass
        write_log(
            timing.format_summary(
                "手工审核",
                {
                    "成功": result["成功"],
                    "跳过": result["跳过"],
                    "失败": result["失败"] + 1,
                    "未处理": max(0, total - handled_count),
                },
                stopped=result["已停止"],
            )
        )
        raise
    try:
        # 正常完成和用户停止都必须提交已经完成的单字。
        with timing.measure("状态提交"):
            persistence.finish()
    except BaseException:
        # finish 清理异常不得替换真正的保存异常。
        try:
            persistence.leave_for_recovery()
        except Exception:
            pass
        write_log(
            timing.format_summary(
                "手工审核",
                {
                    "成功": result["成功"],
                    "跳过": result["跳过"],
                    "失败": result["失败"] + 1,
                    "未处理": max(0, total - handled_count),
                },
                stopped=result["已停止"],
            )
        )
        raise
    result["未处理"] = max(0, total - handled_count)
    progress.flush()
    write_log(
        timing.format_summary(
            "手工审核",
            {
                "成功": result["成功"],
                "跳过": result["跳过"],
                "失败": result["失败"],
                "未处理": result["未处理"],
            },
            stopped=result["已停止"],
        )
    )
    result["总耗时秒"] = timing.finish()
    return result


class ReviewPage(QWidget):
    """用于逐字检查、修订并通过自动优化稿。"""

    home_requested = Signal()
    summary_changed = Signal(object)
    status_message = Signal(str)

    STATUS_FILTER_ALL = PHASE_FILTER_ALL
    STATUS_FILTERS = REVIEW_STATUS_FILTERS
    SORT_OPTIONS = ("拼音顺序", "文件名顺序", "导入顺序")
    REVIEW_STATUSES = {
        config.STATUS_PENDING_MANUAL_REVIEW,
        config.STATUS_REVIEWED,
        config.STATUS_FINISHED,
    }
    LIST_PANEL_MIN_WIDTH = 260
    LIST_PANEL_DEFAULT_WIDTH = 285
    LIST_PANEL_MAX_WIDTH = 400
    TOOL_PANEL_MIN_WIDTH = 230
    TOOL_PANEL_DEFAULT_WIDTH = 245
    TOOL_PANEL_MAX_WIDTH = 320
    TRANSFORM_PERCENT_MIN = 5
    TRANSFORM_PERCENT_MAX = 500
    TRANSFORM_PERCENT_SLIDER_EXTENT = TRANSFORM_PERCENT_MAX - 100
    TRANSFORM_ROTATION_MIN = -180
    TRANSFORM_ROTATION_MAX = 180
    TRANSFORM_OFFSET_LIMIT = 8192
    LIST_THUMBNAIL_SYNC_LIMIT = 24
    LIST_THUMBNAIL_BATCH_SIZE = 12
    LIST_THUMBNAIL_CACHE_ITEMS = 512
    STRUCTURE_RISK_COLOR = QColor("#ff8a65")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._service: GlyphService | None = None
        self._variant_ids: list[str] = []
        self._item_nodes: list[QTreeWidgetItem] = []
        self._records_by_id: dict[str, dict[str, Any]] = {}
        self._list_thumbnail_cache: OrderedDict[
            tuple[str, int, int],
            QIcon,
        ] = OrderedDict()
        self._list_thumbnail_key_by_path: dict[
            str,
            tuple[str, int, int],
        ] = {}
        self._list_thumbnail_inflight: set[
            tuple[int, tuple[str, int, int]]
        ] = set()
        self._list_thumbnail_workers: dict[
            int,
            tuple[
                FunctionWorker,
                set[tuple[int, tuple[str, int, int]]],
            ],
        ] = {}
        self._list_thumbnail_generation = 0
        self._list_thumbnail_batch_id = 0
        self._list_thumbnail_placeholder: QIcon | None = None
        self._list_thumbnail_pool = QThreadPool(self)
        self._list_thumbnail_pool.setMaxThreadCount(2)
        self._list_thumbnail_pool.setExpiryTimeout(15_000)
        self._list_thumbnail_timer = QTimer(self)
        self._list_thumbnail_timer.setSingleShot(True)
        self._list_thumbnail_timer.setInterval(20)
        self._list_thumbnail_timer.timeout.connect(
            self._load_visible_list_thumbnails
        )
        self._current_variant_id = ""
        self._current_status = ""
        self._source_path = ""
        self._source_stage = ""
        self._canvas_width = 250
        self._canvas_height = 250
        self._batch_running = False
        self._batch_worker: _BulkReviewWorker | None = None
        self._batch_started_at: float | None = None
        self._batch_pool = QThreadPool(self)
        self._batch_pool.setMaxThreadCount(1)
        self._batch_pool.setExpiryTimeout(15_000)
        self._save_running = False
        self._save_worker: FunctionWorker | None = None
        self._save_pool = QThreadPool(self)
        self._save_pool.setMaxThreadCount(1)
        self._save_pool.setExpiryTimeout(15_000)
        self._build_ui()
        self._set_tool(ReviewCanvas.TOOL_TRANSFORM)
        self._setup_shortcuts()
        self._install_space_pan_event_filter()

    @property
    def is_batch_running(self) -> bool:
        """返回整库手工审核任务是否仍在运行。"""
        return self._batch_running

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        title_row = QHBoxLayout()
        brand_mark = QLabel("字")
        brand_mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brand_mark.setFixedSize(38, 38)
        brand_mark.setStyleSheet(
            "background: #315f9a; color: #ffffff; border-radius: 5px; "
            "font-size: 18px; font-weight: 700;"
        )
        title_row.addWidget(brand_mark)
        title_box = QVBoxLayout()
        title_box.setSpacing(3)
        self._title_label = QLabel("手工审核")
        self._title_label.setProperty("role", "pageTitle")
        title_box.addWidget(self._title_label)
        self._summary_label = QLabel("请选择字库")
        self._summary_label.setProperty("role", "muted")
        title_box.addWidget(self._summary_label)
        title_row.addLayout(title_box)
        title_row.addStretch()
        self._complete_button = QPushButton("批量手工审核")
        self._complete_button.setProperty("role", "primary")
        self._complete_button.clicked.connect(self.complete_all_reviews)
        self._complete_button.setEnabled(False)
        title_row.addWidget(self._complete_button)
        self._home_button = QPushButton("返回首页")
        self._home_button.clicked.connect(self._request_home)
        title_row.addWidget(self._home_button)
        root.addLayout(title_row)

        self._batch_progress_widget = QWidget()
        batch_progress_layout = QHBoxLayout(self._batch_progress_widget)
        batch_progress_layout.setContentsMargins(0, 0, 0, 0)
        batch_progress_layout.setSpacing(10)
        self._batch_progress_label = QLabel("准备批量审核")
        self._batch_progress_label.setMinimumWidth(260)
        batch_progress_layout.addWidget(self._batch_progress_label)
        self._batch_progress_bar = QProgressBar()
        self._batch_progress_bar.setRange(0, 100)
        self._batch_progress_bar.setValue(0)
        self._batch_progress_bar.setFormat("0% · 0 / 0")
        self._batch_progress_bar.setTextVisible(True)
        self._batch_progress_bar.setFixedHeight(20)
        batch_progress_layout.addWidget(self._batch_progress_bar, 1)
        self._stop_batch_button = QPushButton("停止批量审核")
        self._stop_batch_button.clicked.connect(self._request_stop_bulk_review)
        batch_progress_layout.addWidget(self._stop_batch_button)
        self._batch_progress_widget.hide()
        root.addWidget(self._batch_progress_widget)

        self._main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._main_splitter.addWidget(self._build_sidebar())
        self._main_splitter.addWidget(self._build_canvas_panel())
        self._main_splitter.addWidget(self._build_tool_panel())
        self._main_splitter.setStretchFactor(0, 0)
        self._main_splitter.setStretchFactor(1, 1)
        self._main_splitter.setStretchFactor(2, 0)
        self._main_splitter.setSizes(
            [self.LIST_PANEL_DEFAULT_WIDTH, 720, self.TOOL_PANEL_DEFAULT_WIDTH]
        )
        root.addWidget(self._main_splitter, 1)
        self._set_controls_enabled(False)

    def _build_sidebar(self) -> QWidget:
        panel = QFrame()
        panel.setProperty("role", "card")
        panel.setMinimumWidth(self.LIST_PANEL_MIN_WIDTH)
        panel.setMaximumWidth(self.LIST_PANEL_MAX_WIDTH)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        heading = QHBoxLayout()
        title = QLabel("字形列表")
        title.setStyleSheet("font-size: 18px; font-weight: 700;")
        heading.addWidget(title)
        heading.addStretch()
        self._list_count_label = QLabel("显示 / 总数：0 / 0")
        self._list_count_label.setProperty("role", "muted")
        heading.addWidget(self._list_count_label)
        layout.addLayout(heading)

        search_row = QHBoxLayout()
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("搜索字符、字形或文件名")
        self._search_edit.setClearButtonEnabled(True)
        self._search_edit.returnPressed.connect(self._execute_search)
        self._search_edit.textChanged.connect(self._restore_search_when_cleared)
        search_row.addWidget(self._search_edit, 1)
        self._search_button = QPushButton("搜索")
        self._search_button.setObjectName("compactButton")
        self._search_button.clicked.connect(self._execute_search)
        search_row.addWidget(self._search_button)
        layout.addLayout(search_row)

        filter_sort_row = QHBoxLayout()
        filter_sort_row.setSpacing(4)
        self._filter_combo = QComboBox()
        self._filter_combo.addItems(self.STATUS_FILTERS)
        self._filter_combo.setCurrentText(PHASE_FILTER_ALL)
        self._filter_combo.currentTextChanged.connect(self._populate_variants)
        self._filter_combo.setToolTip("按手工审核状态筛选")
        self._filter_combo.setStyleSheet(
            "QComboBox { padding-left: 4px; padding-right: 4px; }"
        )
        self._filter_combo.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Fixed,
        )
        filter_sort_row.addWidget(self._filter_combo, 5)
        self._sort_combo = QComboBox()
        self._sort_combo.addItems(self.SORT_OPTIONS)
        self._sort_combo.currentTextChanged.connect(self._populate_variants)
        self._sort_combo.setStyleSheet(
            "QComboBox { padding-left: 4px; padding-right: 4px; }"
        )
        self._sort_combo.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Fixed,
        )
        filter_sort_row.addWidget(self._sort_combo, 4)
        layout.addLayout(filter_sort_row)

        self._item_tree = QTreeWidget()
        self._item_tree.setColumnCount(2)
        self._item_tree.setHeaderLabels(("字形与文件", "状态与提示"))
        self._item_tree.setRootIsDecorated(True)
        self._item_tree.setIndentation(14)
        self._item_tree.setUniformRowHeights(False)
        self._item_tree.setAlternatingRowColors(False)
        self._item_tree.setWordWrap(True)
        self._item_tree.setAnimated(False)
        self._item_tree.setIconSize(QSize(38, 38))
        self._item_tree.setItemDelegateForColumn(
            1,
            TwoLineStatusDelegate(self._item_tree),
        )
        self._item_tree.setStyleSheet(
            "QTreeWidget { background: #171b22; border: 1px solid #37404d; }"
            "QTreeWidget::item { min-height: 26px; padding: 1px 3px; }"
            "QTreeWidget::item:selected { background: #3c4773; }"
        )
        status_width = max(
            self._item_tree.fontMetrics().horizontalAdvance(value)
            for value in (STAGE_PENDING_REVIEW, STATUS_REVIEWED, "状态与提示")
        )
        marker_width = self._item_tree.fontMetrics().horizontalAdvance(
            MARKER_STRUCTURE_REVIEW
        )
        self._item_tree_columns = AdjustableTreeColumns(
            self._item_tree,
            {
                0: max(
                    160,
                    self._item_tree.fontMetrics().horizontalAdvance("字形与文件") + 24,
                ),
                1: max(status_width, marker_width) + 36,
            },
            {
                0: 160,
                1: max(status_width, marker_width) + 36,
            },
        )
        self._item_tree.currentItemChanged.connect(self._on_variant_selected)
        self._item_tree.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self._item_tree.customContextMenuRequested.connect(
            self._show_glyph_context_menu
        )
        self._item_tree.itemExpanded.connect(self._schedule_list_thumbnail_loads)
        self._item_tree.itemCollapsed.connect(self._schedule_list_thumbnail_loads)
        self._item_tree.verticalScrollBar().valueChanged.connect(
            self._schedule_list_thumbnail_loads
        )
        self._item_tree.viewport().installEventFilter(self)
        layout.addWidget(self._item_tree, 1)

        self._count_label = QLabel("待审核 0　已审核 0")
        self._count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._count_label.setProperty("role", "muted")
        layout.addWidget(self._count_label)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("完成度 %p%")
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFixedHeight(20)
        layout.addWidget(self._progress_bar)

        return panel

    def _build_canvas_panel(self) -> QWidget:
        panel = QFrame()
        panel.setProperty("role", "card")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        info_widget = QWidget()
        info_row = QHBoxLayout(info_widget)
        info_row.setContentsMargins(12, 8, 12, 8)
        info_row.setSpacing(9)
        self._glyph_label = QLabel("未选择字形")
        self._glyph_label.setStyleSheet("font-size: 19px; font-weight: 700;")
        info_row.addWidget(self._glyph_label)
        self._file_label = QLabel("")
        self._file_label.setProperty("role", "muted")
        self._file_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        info_row.addWidget(self._file_label, 1)
        self._status_label = QLabel("")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        info_row.addWidget(self._status_label)
        self._previous_button = self._toolbar_button("上一条", "上一字形")
        self._previous_button.setAccessibleName("上一条")
        self._fit_navigation_button(self._previous_button)
        self._previous_button.clicked.connect(lambda: self._move_selection(-1))
        info_row.addWidget(self._previous_button)
        self._next_button = self._toolbar_button("下一条", "下一字形")
        self._next_button.setAccessibleName("下一条")
        self._fit_navigation_button(self._next_button)
        self._next_button.clicked.connect(lambda: self._move_selection(1))
        info_row.addWidget(self._next_button)
        layout.addWidget(info_widget)

        self._toolbar_widget = self._build_canvas_toolbar()
        layout.addWidget(self._toolbar_widget)

        self._canvas = ReviewCanvas()
        self._canvas.changed.connect(self._on_canvas_changed)
        self._canvas.zoom_changed.connect(lambda value: self._zoom_label.setText(f"{value}%"))
        self._canvas.transform_changed.connect(self._sync_transform_controls)
        self._canvas.ink_color_changed.connect(self._update_ink_swatch)
        self._canvas.brush_size_changed.connect(self._sync_brush_size)
        self._canvas.set_background_mode("white")
        self._canvas.set_grid_visible(True)
        layout.addWidget(self._canvas, 1)

        footer_widget = QWidget()
        footer = QHBoxLayout(footer_widget)
        footer.setContentsMargins(12, 7, 12, 7)
        self._source_label = QLabel("来源：-")
        self._source_label.setProperty("role", "muted")
        self._source_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        footer.addWidget(self._source_label, 1)
        self._save_state_label = QLabel("无未保存修改")
        self._save_state_label.setProperty("role", "muted")
        footer.addWidget(self._save_state_label)
        self._zoom_label = QLabel("100%")
        self._zoom_label.setMinimumWidth(50)
        self._zoom_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        footer.addWidget(self._zoom_label)
        layout.addWidget(footer_widget)
        return panel

    def _build_canvas_toolbar(self) -> QWidget:
        toolbar = QWidget()
        toolbar.setStyleSheet("background: #1b2028; border-top: 1px solid #37404d; border-bottom: 1px solid #37404d;")
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(7, 5, 7, 5)
        layout.setSpacing(4)

        self._undo_button = self._toolbar_button("↶", "撤销（Ctrl+Z）")
        self._undo_button.setAccessibleName("撤销")
        self._undo_button.clicked.connect(self._canvas_undo)
        layout.addWidget(self._undo_button)
        self._redo_button = self._toolbar_button("↷", "重做（Ctrl+Y）")
        self._redo_button.setAccessibleName("重做")
        self._redo_button.clicked.connect(self._canvas_redo)
        layout.addWidget(self._redo_button)
        layout.addWidget(self._vertical_separator())

        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(True)
        self._tool_buttons: dict[str, QToolButton] = {}
        for label, tool, tip in (
            ("变换", ReviewCanvas.TOOL_TRANSFORM, "移动、缩放或旋转字形"),
            ("画笔", ReviewCanvas.TOOL_BRUSH, "补充缺损笔画"),
            ("橡皮", ReviewCanvas.TOOL_ERASER, "清除多余墨迹"),
        ):
            button = self._toolbar_button(label, tip, checkable=True)
            button.clicked.connect(lambda _checked=False, value=tool: self._set_tool(value))
            self._tool_group.addButton(button)
            self._tool_buttons[tool] = button
            layout.addWidget(button)
        self._tool_buttons[ReviewCanvas.TOOL_TRANSFORM].setChecked(True)
        layout.addWidget(self._vertical_separator())

        self._fit_button = self._toolbar_button("适应", "适应窗口：完整显示当前画布")
        self._fit_button.clicked.connect(self._canvas_fit)
        layout.addWidget(self._fit_button)
        self._actual_size_button = self._toolbar_button("1:1", "按图像实际像素显示")
        self._actual_size_button.clicked.connect(self._canvas_actual_size)
        layout.addWidget(self._actual_size_button)
        self._grid_button = self._toolbar_button("网格", "显示或隐藏字格参考线", checkable=True)
        self._grid_button.setChecked(True)
        self._grid_button.toggled.connect(self._set_grid_visible)
        layout.addWidget(self._grid_button)

        self._background_group = QButtonGroup(self)
        self._background_group.setExclusive(True)
        self._white_button = self._toolbar_button("白底", "使用白色预览背景", checkable=True)
        self._checker_button = self._toolbar_button("透明", "使用透明棋盘格背景", checkable=True)
        self._white_button.setChecked(True)
        self._white_button.clicked.connect(lambda: self._set_background_mode("white"))
        self._checker_button.clicked.connect(lambda: self._set_background_mode("checkerboard"))
        self._background_group.addButton(self._white_button)
        self._background_group.addButton(self._checker_button)
        layout.addWidget(self._white_button)
        layout.addWidget(self._checker_button)
        layout.addStretch()
        return toolbar

    def _build_tool_panel(self) -> QWidget:
        panel = QFrame()
        panel.setProperty("role", "card")
        panel.setMinimumWidth(self.TOOL_PANEL_MIN_WIDTH)
        panel.setMaximumWidth(self.TOOL_PANEL_MAX_WIDTH)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        heading_widget = QWidget()
        heading = QHBoxLayout(heading_widget)
        heading.setContentsMargins(12, 11, 12, 10)
        title = QLabel("编辑参数")
        title.setStyleSheet("font-size: 17px; font-weight: 700;")
        heading.addWidget(title)
        heading.addStretch()
        self._tool_mode_label = QLabel("自由变换")
        self._tool_mode_label.setStyleSheet("color: #4da3ff;")
        heading.addWidget(self._tool_mode_label)
        layout.addWidget(heading_widget)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(12, 10, 12, 12)
        body_layout.setSpacing(12)

        self._parameters_stack = QStackedWidget()
        self._transform_panel = self._build_transform_parameters()
        self._pixel_panel = self._build_pixel_parameters()
        self._parameters_stack.addWidget(self._transform_panel)
        self._parameters_stack.addWidget(self._pixel_panel)
        body_layout.addWidget(self._parameters_stack)
        body_layout.addWidget(self._horizontal_separator())
        self._draft_information_panel = self._build_draft_information()
        body_layout.addWidget(self._draft_information_panel)
        body_layout.addStretch()
        scroll.setWidget(body)
        layout.addWidget(scroll, 1)

        actions = QWidget()
        actions.setStyleSheet("background: #1b2028; border-top: 1px solid #37404d;")
        action_layout = QVBoxLayout(actions)
        action_layout.setContentsMargins(10, 10, 10, 10)
        action_layout.setSpacing(7)
        self._restore_button = QPushButton("恢复上次保存")
        self._restore_button.clicked.connect(self._canvas_reset)
        action_layout.addWidget(self._restore_button)
        self._save_button = QPushButton("保存修改稿")
        self._save_button.clicked.connect(self._start_save_current)
        action_layout.addWidget(self._save_button)
        self._approve_button = QPushButton("保存并审核通过")
        self._approve_button.setProperty("role", "primary")
        self._approve_button.clicked.connect(self._start_approve_current)
        action_layout.addWidget(self._approve_button)
        layout.addWidget(actions)
        return panel

    def _build_transform_parameters(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(self._section_title("自由变换"))

        position_label = QLabel("位置偏移（像素）")
        position_label.setProperty("role", "muted")
        layout.addWidget(position_label)
        positions = QGridLayout()
        positions.setHorizontalSpacing(8)
        positions.setVerticalSpacing(5)
        positions.addWidget(QLabel("水平 X"), 0, 0)
        positions.addWidget(QLabel("垂直 Y"), 0, 1)
        self._offset_x_spin = QSpinBox()
        self._offset_x_spin.setRange(
            -self.TRANSFORM_OFFSET_LIMIT,
            self.TRANSFORM_OFFSET_LIMIT,
        )
        self._offset_x_spin.setMinimumWidth(0)
        self._offset_x_spin.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self._offset_y_spin = QSpinBox()
        self._offset_y_spin.setRange(
            -self.TRANSFORM_OFFSET_LIMIT,
            self.TRANSFORM_OFFSET_LIMIT,
        )
        self._offset_y_spin.setMinimumWidth(0)
        self._offset_y_spin.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        positions.addWidget(self._offset_x_spin, 1, 0)
        positions.addWidget(self._offset_y_spin, 1, 1)
        layout.addLayout(positions)

        scale_head = QHBoxLayout()
        scale_head.addWidget(QLabel("等比缩放"))
        scale_head.addStretch()
        self._scale_value_label = QLabel("100%")
        self._scale_value_label.setProperty("role", "muted")
        scale_head.addWidget(self._scale_value_label)
        layout.addLayout(scale_head)
        self._scale_slider = QSlider(Qt.Orientation.Horizontal)
        self._scale_slider.setRange(
            -self.TRANSFORM_PERCENT_SLIDER_EXTENT,
            self.TRANSFORM_PERCENT_SLIDER_EXTENT,
        )
        self._scale_slider.setValue(0)
        layout.addWidget(self._scale_slider)

        stretch_w_head = QHBoxLayout()
        stretch_w_head.addWidget(QLabel("水平拉伸 / 压缩"))
        stretch_w_head.addStretch()
        self._stretch_w_value_label = QLabel("100%")
        self._stretch_w_value_label.setProperty("role", "muted")
        stretch_w_head.addWidget(self._stretch_w_value_label)
        layout.addLayout(stretch_w_head)
        self._stretch_w_slider = QSlider(Qt.Orientation.Horizontal)
        self._stretch_w_slider.setRange(
            -self.TRANSFORM_PERCENT_SLIDER_EXTENT,
            self.TRANSFORM_PERCENT_SLIDER_EXTENT,
        )
        self._stretch_w_slider.setValue(0)
        layout.addWidget(self._stretch_w_slider)

        stretch_h_head = QHBoxLayout()
        stretch_h_head.addWidget(QLabel("垂直拉伸 / 压缩"))
        stretch_h_head.addStretch()
        self._stretch_h_value_label = QLabel("100%")
        self._stretch_h_value_label.setProperty("role", "muted")
        stretch_h_head.addWidget(self._stretch_h_value_label)
        layout.addLayout(stretch_h_head)
        self._stretch_h_slider = QSlider(Qt.Orientation.Horizontal)
        self._stretch_h_slider.setRange(
            -self.TRANSFORM_PERCENT_SLIDER_EXTENT,
            self.TRANSFORM_PERCENT_SLIDER_EXTENT,
        )
        self._stretch_h_slider.setValue(0)
        layout.addWidget(self._stretch_h_slider)

        rotation_head = QHBoxLayout()
        rotation_head.addWidget(QLabel("旋转"))
        rotation_head.addStretch()
        self._rotation_value_label = QLabel("0°")
        self._rotation_value_label.setProperty("role", "muted")
        rotation_head.addWidget(self._rotation_value_label)
        layout.addLayout(rotation_head)
        self._rotation_slider = QSlider(Qt.Orientation.Horizontal)
        self._rotation_slider.setRange(
            self.TRANSFORM_ROTATION_MIN,
            self.TRANSFORM_ROTATION_MAX,
        )
        self._rotation_slider.setValue(0)
        layout.addWidget(self._rotation_slider)

        self._offset_x_spin.valueChanged.connect(
            lambda value: self._apply_transform_field("x", float(value))
        )
        self._offset_y_spin.valueChanged.connect(
            lambda value: self._apply_transform_field("y", float(value))
        )
        for slider, field in (
            (self._scale_slider, "scale"),
            (self._stretch_w_slider, "stretch_w"),
            (self._stretch_h_slider, "stretch_h"),
        ):
            slider.valueChanged.connect(
                lambda _value, control=slider, name=field: (
                    self._on_percent_slider_changed(control, name)
                )
            )
            slider.sliderReleased.connect(
                lambda control=slider, name=field: (
                    self._commit_percent_slider(control, name)
                )
            )
        self._rotation_slider.valueChanged.connect(
            lambda value: self._on_transform_slider_changed(
                self._rotation_slider,
                "rotation",
                float(value),
            )
        )
        self._rotation_slider.sliderReleased.connect(
            lambda: self._commit_transform_slider(
                self._rotation_slider,
                "rotation",
                1.0,
            )
        )

        self._reset_transform_button = QPushButton("重置变换")
        self._reset_transform_button.setObjectName("compactButton")
        self._reset_transform_button.clicked.connect(self._reset_transform)
        layout.addWidget(self._reset_transform_button)
        return panel

    def _build_pixel_parameters(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self._pixel_title = self._section_title("画笔")
        layout.addWidget(self._pixel_title)

        brush_head = QHBoxLayout()
        brush_head.addWidget(QLabel("笔触大小"))
        brush_head.addStretch()
        self._brush_value_label = QLabel("10 px")
        self._brush_value_label.setProperty("role", "muted")
        brush_head.addWidget(self._brush_value_label)
        layout.addLayout(brush_head)
        self._brush_slider = QSlider(Qt.Orientation.Horizontal)
        self._brush_slider.setRange(1, 100)
        self._brush_slider.setValue(10)
        self._brush_slider.valueChanged.connect(self._set_brush_size)
        layout.addWidget(self._brush_slider)

        self._pressure_checkbox = QCheckBox("笔压控制粗细")
        self._pressure_checkbox.setChecked(True)
        self._pressure_checkbox.setToolTip("绘图板压力越大，实际笔触越接近设定的笔触大小")
        self._pressure_checkbox.toggled.connect(self._set_pressure_enabled)
        layout.addWidget(self._pressure_checkbox)

        pressure_head = QHBoxLayout()
        self._minimum_pressure_title = QLabel("最轻笔触")
        pressure_head.addWidget(self._minimum_pressure_title)
        pressure_head.addStretch()
        self._minimum_pressure_value_label = QLabel("20%")
        self._minimum_pressure_value_label.setProperty("role", "muted")
        pressure_head.addWidget(self._minimum_pressure_value_label)
        layout.addLayout(pressure_head)
        self._minimum_pressure_slider = QSlider(Qt.Orientation.Horizontal)
        self._minimum_pressure_slider.setRange(5, 100)
        self._minimum_pressure_slider.setValue(20)
        self._minimum_pressure_slider.setToolTip("绘图板最轻压力对应的基础笔触宽度")
        self._minimum_pressure_slider.valueChanged.connect(self._set_minimum_pressure_ratio)
        layout.addWidget(self._minimum_pressure_slider)

        self._ink_controls = QWidget()
        ink_layout = QVBoxLayout(self._ink_controls)
        ink_layout.setContentsMargins(0, 2, 0, 0)
        ink_layout.setSpacing(7)
        ink_label = QLabel("当前墨色")
        ink_label.setProperty("role", "muted")
        ink_layout.addWidget(ink_label)
        ink_row = QHBoxLayout()
        self._ink_swatch = QFrame()
        self._ink_swatch.setFixedSize(28, 28)
        ink_row.addWidget(self._ink_swatch)
        self._sample_ink_button = QPushButton("自动取墨")
        self._sample_ink_button.setObjectName("compactButton")
        self._sample_ink_button.clicked.connect(self._sample_ink_color)
        ink_row.addWidget(self._sample_ink_button, 1)
        ink_layout.addLayout(ink_row)
        layout.addWidget(self._ink_controls)
        layout.addStretch()
        self._update_ink_swatch(QColor("#000000"))
        return panel

    def _build_draft_information(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(9)
        layout.addWidget(self._section_title("当前稿件"))
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        labels = (("来源", "_draft_source_label"), ("文件", "_draft_file_label"), ("状态", "_draft_status_label"), ("保存", "_draft_save_label"))
        for row, (caption, attribute) in enumerate(labels):
            name_label = QLabel(caption)
            name_label.setProperty("role", "muted")
            value_label = QLabel("-")
            value_label.setWordWrap(True)
            value_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            setattr(self, attribute, value_label)
            grid.addWidget(name_label, row, 0, Qt.AlignmentFlag.AlignTop)
            grid.addWidget(value_label, row, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)
        return panel

    def _setup_shortcuts(self) -> None:
        self._shortcut_actions: list[QAction] = []
        for shortcut, callback in (
            (QKeySequence.StandardKey.Save, self._start_save_current),
            (QKeySequence.StandardKey.Undo, self._canvas.undo),
            (QKeySequence.StandardKey.Redo, self._canvas.redo),
            (QKeySequence("["), lambda: self._canvas.adjust_brush_size(-1)),
            (QKeySequence("]"), lambda: self._canvas.adjust_brush_size(1)),
        ):
            action = QAction(self)
            action.setShortcut(shortcut)
            action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            action.triggered.connect(callback)
            self.addAction(action)
            self._shortcut_actions.append(action)

    def _install_space_pan_event_filter(self) -> None:
        """在页面子控件持有焦点时，仍允许画布接收临时平移按键。"""
        application = QApplication.instance()
        if application is not None:
            application.installEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """仅在指针位于画布或平移已激活时接管空格键。"""
        item_tree = getattr(self, "_item_tree", None)
        if (
            item_tree is not None
            and watched is item_tree.viewport()
            and event.type() in (QEvent.Type.Resize, QEvent.Type.Show)
        ):
            self._schedule_list_thumbnail_loads()
        if (
            isinstance(event, QKeyEvent)
            and event.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease)
            and event.key() == Qt.Key.Key_Space
            and self._owns_event_target(watched)
            and not self._canvas_owns_event_target(watched)
            and (
                self._canvas.space_pan_active
                or (
                    self._cursor_is_over_canvas()
                    and not self._focused_control_uses_space()
                )
            )
        ):
            handled = self._canvas.handle_space_pan_key(
                event.type() == QEvent.Type.KeyPress,
                auto_repeat=event.isAutoRepeat(),
            )
            if handled:
                event.accept()
                return True
        return super().eventFilter(watched, event)

    def _owns_event_target(self, watched: QObject) -> bool:
        return watched is self or (
            isinstance(watched, QWidget) and self.isAncestorOf(watched)
        )

    def _canvas_owns_event_target(self, watched: QObject) -> bool:
        return watched is self._canvas or (
            isinstance(watched, QWidget) and self._canvas.isAncestorOf(watched)
        )

    @staticmethod
    def _focused_control_uses_space() -> bool:
        """输入与按钮的空格语义优先于尚未开始的临时平移。"""
        focused = QApplication.focusWidget()
        while focused is not None:
            if isinstance(focused, (QAbstractButton, QLineEdit)):
                return True
            parent = focused.parentWidget()
            focused = parent if isinstance(parent, QWidget) else None
        return False

    def _cursor_is_over_canvas(self) -> bool:
        position = self._canvas.mapFromGlobal(QCursor.pos())
        return self._canvas.rect().contains(position)

    def open_library(self, library_path: str, variant_id: str = "") -> bool:
        if self._batch_running or self._save_running:
            self.status_message.emit("正在完成手工审核，请等待任务结束。")
            return False
        if self._canvas.is_dirty and not self._confirm_discard():
            return False
        name = os.path.basename(os.path.normpath(library_path))
        self._service = GlyphService.open(name, library_path)
        summarize_glyph_service(self._service)
        self._list_thumbnail_timer.stop()
        self._list_thumbnail_generation += 1
        self._list_thumbnail_cache.clear()
        self._list_thumbnail_key_by_path.clear()
        self._current_variant_id = ""
        self._current_status = ""
        self._source_path = ""
        self._source_stage = ""
        self._filter_combo.blockSignals(True)
        self._filter_combo.setCurrentText(PHASE_FILTER_ALL)
        self._filter_combo.blockSignals(False)
        self._search_edit.blockSignals(True)
        self._search_edit.clear()
        self._search_edit.blockSignals(False)
        metadata = self._service.get_metadata()
        self._canvas_width = self._positive_int(metadata.get("画布宽"), 250)
        self._canvas_height = self._positive_int(metadata.get("画布高"), 250)
        self._summary_label.setText(
            f"当前字库：{name}　{metadata.get('DPI', '--')} DPI · "
            f"{metadata.get('画布宽', '--')}×{metadata.get('画布高', '--')} 像素"
        )
        self._complete_button.setEnabled(True)
        self._populate_variants(select_variant=variant_id)
        return True

    def save_current(self) -> bool:
        return self._save_current_image(approve=False)

    def _start_save_current(self) -> None:
        self._start_interactive_save(approve=False)

    def _start_approve_current(self) -> None:
        self._start_interactive_save(approve=True)

    def _start_interactive_save(self, *, approve: bool) -> None:
        """从界面按钮启动后台单字保存，避免阻塞 Qt 主线程。"""

        if self._save_running or self._batch_running:
            return
        if not self._service or not self._current_variant_id:
            return
        if not self._current_record_is_editable() or not self._canvas.has_image:
            self.status_message.emit("当前字形还不能保存。")
            return
        output_image = self._canvas.image().copy()
        if not _has_visible_ink(output_image):
            QMessageBox.warning(
                self,
                "无法保存",
                "当前稿件没有有效文字前景，请保留文字内容后再保存。",
            )
            return
        output_dir = self._service.get_workflow_dirs()["手工审核"]
        try:
            ensure_dir(output_dir)
        except OSError as exc:
            QMessageBox.critical(self, "保存失败", f"无法创建手工审核目录：{exc}")
            return
        variant_id = self._current_variant_id
        filename = Path(self._source_path).stem + ".png"
        output_origin = self._canvas.output_origin()
        origin = (-int(output_origin.x()), -int(output_origin.y()))
        dpi = self._positive_int(self._service.get_metadata().get("DPI"), 300)
        current_index = (
            self._variant_ids.index(variant_id)
            if variant_id in self._variant_ids
            else -1
        )
        next_variant = (
            self._variant_ids[current_index + 1]
            if approve and 0 <= current_index < len(self._variant_ids) - 1
            else ""
        )
        reusable_payload: tuple[str, str, bool] | None = None
        reusable = self._reusable_review_file() if approve else None
        if reusable is not None:
            detail = self._service.get_variant(variant_id)
            manual_edit = detail.get("手工编辑", {}) if detail else {}
            edited = (
                bool(manual_edit.get("已编辑", True))
                if isinstance(manual_edit, dict)
                else True
            )
            reusable_payload = (reusable[0], reusable[1], edited)
        context = {
            "字形ID": variant_id,
            "下一字形": next_variant,
            "图像": output_image,
            "文件名": filename,
            "审核通过": approve,
        }
        library_name = self._service.ziku_name
        library_path = self._service.ziku_dir
        worker = FunctionWorker(
            lambda: _save_interactive_review_in_worker(
                library_name,
                library_path,
                variant_id,
                output_image,
                filename,
                origin,
                dpi,
                approve=approve,
                reusable=reusable_payload,
            )
        )
        worker.signals.finished.connect(
            lambda result, task=worker, state=context: self._interactive_save_finished(
                task,
                state,
                result,
            )
        )
        worker.signals.failed.connect(
            lambda message, task=worker: self._interactive_save_failed(task, message)
        )
        self._save_worker = worker
        self._set_single_save_running(True, approve=approve)
        self._save_pool.start(worker)

    def _interactive_save_finished(
        self,
        worker: FunctionWorker,
        context: dict[str, Any],
        result: object,
    ) -> None:
        if worker is not self._save_worker:
            return
        self._save_worker = None
        self._set_single_save_running(False)
        if not self._service or not isinstance(result, dict):
            QMessageBox.critical(self, "保存失败", "后台保存没有返回有效结果。")
            return
        try:
            snapshot = result.get("字形状态")
            if not isinstance(snapshot, dict):
                raise TypeError("后台保存没有返回字形状态")
            self._service.restore_variant_state(snapshot)
            output_path = str(result.get("输出路径", ""))
            filename = str(context["文件名"])
            approve = bool(context["审核通过"])
            self._apply_saved_review_ui(
                context["图像"],
                output_path,
                filename,
                approve=approve,
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "保存后刷新失败",
                f"稿件已经保存，但页面状态刷新失败：{exc}\n请重新进入手工审核。",
            )
            return
        if approve:
            variant_id = str(context["字形ID"])
            next_variant = str(context["下一字形"])
            self.summary_changed.emit(self._service)
            self.status_message.emit(f"已审核通过：{variant_id}")
            self._advance_after_review_approval(variant_id, next_variant)
            if not next_variant:
                self._show_review_end_notice()

    def _interactive_save_failed(
        self,
        worker: FunctionWorker,
        message: str,
    ) -> None:
        if worker is not self._save_worker:
            return
        self._save_worker = None
        self._set_single_save_running(False)
        QMessageBox.critical(self, "保存失败", f"无法保存手工审核图像：{message}")

    def _set_single_save_running(self, running: bool, *, approve: bool = False) -> None:
        self._save_running = bool(running)
        self._main_splitter.setEnabled(not running)
        self._home_button.setEnabled(not running)
        self._complete_button.setEnabled(not running and self._service is not None)
        for action in self._shortcut_actions:
            action.setEnabled(not running)
        if running:
            self.status_message.emit(
                "正在后台保存并审核通过…" if approve else "正在后台保存修改稿…"
            )

    def _save_current_image(self, *, approve: bool) -> bool:
        if self._batch_running or self._save_running:
            self.status_message.emit("正在完成手工审核，暂时不能保存单字。")
            return False
        if not self._service or not self._current_variant_id:
            return False
        if not self._current_record_is_editable():
            self.status_message.emit("当前字形仍在待优化阶段，请先完成自动优化。")
            return False
        if not self._canvas.has_image:
            return False
        output_dir = self._service.get_workflow_dirs()["手工审核"]
        try:
            ensure_dir(output_dir)
        except OSError as exc:
            QMessageBox.critical(self, "保存失败", f"无法创建手工审核目录：{exc}")
            return False
        filename = Path(self._source_path).stem + ".png"
        output_path = os.path.join(output_dir, filename)
        output_image = self._canvas.image()
        if not _has_visible_ink(output_image):
            QMessageBox.warning(
                self,
                "无法保存",
                "当前稿件没有有效文字前景，请保留文字内容后再保存。",
            )
            return False
        output_origin = self._canvas.output_origin()
        try:
            dpi = self._positive_int(self._service.get_metadata().get("DPI"), 300)
            output_path = _save_interactive_review(
                self._service,
                self._current_variant_id,
                output_image,
                filename,
                (-int(output_origin.x()), -int(output_origin.y())),
                dpi,
                approve=approve,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            QMessageBox.critical(
                self,
                "保存失败",
                f"无法保存手工审核图像：{exc}",
            )
            return False

        self._apply_saved_review_ui(
            output_image,
            output_path,
            filename,
            approve=approve,
        )
        return True

    def _apply_saved_review_ui(
        self,
        output_image: QImage,
        output_path: str,
        filename: str,
        *,
        approve: bool,
    ) -> None:
        """把已成功提交的单字结果合并到当前页面。"""

        self._canvas.set_image(output_image, (self._canvas_width, self._canvas_height))
        self._canvas.mark_saved()
        self._source_path = output_path
        self._source_stage = "手工审核稿"
        self._current_status = (
            config.STATUS_REVIEWED
            if approve
            else config.STATUS_PENDING_MANUAL_REVIEW
        )
        self._source_label.setText(f"来源：手工审核稿 · {filename}")
        self._draft_source_label.setText("手工审核稿")
        self._draft_file_label.setText(filename)
        phase_status = STATUS_REVIEWED if approve else STAGE_PENDING_REVIEW
        self._draft_status_label.setText(phase_status)
        self._status_label.setText(phase_status)
        self._status_label.setStyleSheet(
            f"color: {PHASE_STATUS_COLORS[phase_status]};"
        )
        self._file_label.setText(f"{filename} · {self._current_variant_id}")
        self._update_current_list_item(self._current_status, filename)
        self._refresh_progress(list(self._records_by_id.values()))
        if self._service is not None and not approve:
            self.summary_changed.emit(self._service)
        if not approve:
            self.status_message.emit(f"已保存手工修改：{filename}")

    def _reusable_review_file(self) -> tuple[str, str] | None:
        """返回已符合当前字库尺寸契约的审核稿及摘要。"""

        if (
            not self._service
            or not self._current_variant_id
            or self._canvas.is_dirty
            or self._source_stage != "手工审核稿"
        ):
            return None
        detail = self._service.get_variant(self._current_variant_id)
        if not detail or str(detail.get("成品文件", "") or ""):
            return None
        filename = str(detail.get("审核文件", "") or "")
        path = resolve_safe_stage_file(
            self._service.get_workflow_dirs()["手工审核"],
            filename,
        )
        if not path or os.path.normcase(path) != os.path.normcase(self._source_path):
            return None
        expected_md5 = str(detail.get("审核MD5", "") or "").lower()
        if not expected_md5 or _file_md5(path) != expected_md5:
            return None
        parameters = detail.get("变换参数", {})
        origin = parameters.get("图像原点") if isinstance(parameters, dict) else None
        if not isinstance(origin, (list, tuple)) or len(origin) != 2:
            return None
        try:
            origin_x = int(origin[0])
            origin_y = int(origin[1])
        except (TypeError, ValueError):
            return None
        if origin_x > 0 or origin_y > 0:
            return None
        image = QImage(path)
        if image.isNull():
            return None
        expected_size = QSize(
            self._canvas_width - origin_x * 2,
            self._canvas_height - origin_y * 2,
        )
        if image.size() != expected_size:
            return None
        dpi = self._positive_int(self._service.get_metadata().get("DPI"), 300)
        target_dpm = round(dpi / 0.0254)
        if (
            abs(image.dotsPerMeterX() - target_dpm) > 2
            or abs(image.dotsPerMeterY() - target_dpm) > 2
        ):
            return None
        return path, expected_md5

    def approve_current(self) -> None:
        if self._batch_running or self._save_running:
            self.status_message.emit("正在完成手工审核，暂时不能审核单字。")
            return
        if not self._service or not self._current_variant_id:
            return
        if not self._current_record_is_editable():
            self.status_message.emit("当前字形仍在待优化阶段，请先完成自动优化。")
            return
        current_index = self._variant_ids.index(self._current_variant_id) if self._current_variant_id in self._variant_ids else -1
        next_variant = self._variant_ids[current_index + 1] if 0 <= current_index < len(self._variant_ids) - 1 else ""
        approved_variant = self._current_variant_id
        reusable = self._reusable_review_file()
        if reusable is None:
            if not self._save_current_image(approve=True):
                return
        else:
            timing = BatchTiming()
            state_backup = self._service.snapshot_variant_state(approved_variant)
            detail = self._service.get_variant(approved_variant)
            manual_edit = detail.get("手工编辑", {}) if detail else {}
            if not isinstance(manual_edit, dict):
                manual_edit = {}
            try:
                with timing.measure("状态提交"):
                    self._service.mark_manual_saved(
                        approved_variant,
                        os.path.basename(reusable[0]),
                        reusable[1],
                        edited=bool(manual_edit.get("已编辑", True)),
                    )
                    if not self._service.approve_manual_review(approved_variant):
                        raise RuntimeError("当前审核稿无法审核通过")
                    self._service.save()
            except (OSError, RuntimeError, ValueError) as exc:
                self._service.restore_variant_state(state_backup)
                QMessageBox.critical(self, "审核失败", f"无法保存审核状态：{exc}")
                return
            write_log(
                timing.format_summary(
                    "手工审核复用",
                    {"字形": 1, "复用PNG": 1},
                )
            )
        self._current_status = config.STATUS_REVIEWED
        self._status_label.setText(STATUS_REVIEWED)
        self._status_label.setStyleSheet(
            f"color: {PHASE_STATUS_COLORS[STATUS_REVIEWED]};"
        )
        self._draft_status_label.setText(STATUS_REVIEWED)
        self._update_current_list_item(config.STATUS_REVIEWED)
        if self._service is not None:
            self.summary_changed.emit(self._service)
        self.status_message.emit(f"已审核通过：{approved_variant}")
        self._advance_after_review_approval(approved_variant, next_variant)
        if not next_variant:
            self._show_review_end_notice()

    def _advance_after_review_approval(
        self,
        approved_variant: str,
        next_variant: str,
    ) -> None:
        """局部更新审核结果，并按当前筛选定位下一字形。"""

        self._refresh_progress(list(self._records_by_id.values()))
        if self._filter_combo.currentText() == STAGE_PENDING_REVIEW:
            node = self._node_for_variant(approved_variant)
            if node is not None:
                parent = node.parent()
                row = self._variant_ids.index(approved_variant)
                with QSignalBlocker(self._item_tree):
                    if parent is not None:
                        parent.removeChild(node)
                        if parent.childCount() == 0:
                            parent_row = self._item_tree.indexOfTopLevelItem(parent)
                            if parent_row >= 0:
                                self._item_tree.takeTopLevelItem(parent_row)
                        else:
                            self._update_group_item(parent)
                    self._variant_ids.pop(row)
                    self._item_nodes.pop(row)
            self._list_count_label.setText(
                f"显示 / 总数：{len(self._variant_ids)} / {len(self._records_by_id)}"
            )

        target = self._node_for_variant(next_variant) if next_variant else None
        if target is not None:
            self._item_tree.setCurrentItem(target)
        elif approved_variant not in self._variant_ids:
            self._clear_current()
        else:
            self._update_navigation_buttons()

    def _show_review_end_notice(self) -> None:
        """说明当前搜索和筛选范围已经没有下一字形。"""

        pending_count = sum(
            self._record_phase_status(self._records_by_id[variant_id])
            == STAGE_PENDING_REVIEW
            for variant_id in self._variant_ids
            if variant_id in self._records_by_id
        )
        if pending_count:
            detail = (
                "当前字形已审核通过，已到当前搜索和筛选范围的最后一条。\n"
                f"该范围内仍有 {pending_count} 个待审核字形，可从左侧列表重新选择。"
            )
        else:
            detail = (
                "当前字形已审核通过，已到当前搜索和筛选范围的最后一条。\n"
                "该范围内的待审核字形已全部处理完成。"
            )
        QMessageBox.information(self, "手工审核", detail)

    def complete_all_reviews(self) -> None:
        """确认后在后台完成当前字库的全部待审核字形。"""
        if self._batch_running or self._save_running or not self._service:
            return
        if self._canvas.is_dirty and not self._save_unsaved_before_batch():
            return
        pending_ids = self._pending_review_ids()
        if not pending_ids:
            QMessageBox.information(
                self,
                "完成手工审核",
                "当前字库没有待审核字形。",
            )
            return
        if not self._confirm_bulk_review(len(pending_ids)):
            return
        self._start_bulk_review(pending_ids)

    def _save_unsaved_before_batch(self) -> bool:
        """批量开始前只允许保存当前修改或取消，不静默放弃。"""
        dialog = QMessageBox(self)
        dialog.setWindowTitle("尚未保存")
        dialog.setIcon(QMessageBox.Icon.Question)
        dialog.setText("当前字形有未保存的修改。")
        dialog.setInformativeText("必须先保存当前修改，才能完成全库手工审核。")
        dialog.setStandardButtons(
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Cancel
        )
        dialog.setDefaultButton(QMessageBox.StandardButton.Save)
        save_button = dialog.button(QMessageBox.StandardButton.Save)
        cancel_button = dialog.button(QMessageBox.StandardButton.Cancel)
        if save_button is not None:
            save_button.setText("保存修改并继续")
        if cancel_button is not None:
            cancel_button.setText("取消")
            dialog.setEscapeButton(cancel_button)
        if dialog.exec() != QMessageBox.StandardButton.Save.value:
            return False
        return self.save_current()

    def _confirm_bulk_review(self, pending_count: int) -> bool:
        dialog = QMessageBox(self)
        dialog.setWindowTitle("完成手工审核")
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setText(f"确定批量审核 {pending_count} 个待审核字形吗？")
        dialog.setInformativeText(
            "批量审核会优先采用已有人工修订稿；没有人工稿时，将按载入规则直接采用"
            "自动优化稿并标记为审核通过。\n\n"
            "此操作不能替代人工判断，可能遗漏字形缺损、误删笔画、残留污点、大小失衡"
            "或视觉中心偏移，批量通过后仍可能需要逐字返工。建议先抽查代表性字形。"
        )
        dialog.setStandardButtons(
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
        )
        dialog.setDefaultButton(QMessageBox.StandardButton.Cancel)
        start_button = dialog.button(QMessageBox.StandardButton.Ok)
        cancel_button = dialog.button(QMessageBox.StandardButton.Cancel)
        if start_button is not None:
            start_button.setText("确定")
        if cancel_button is not None:
            cancel_button.setText("取消")
            dialog.setEscapeButton(cancel_button)
        return dialog.exec() == QMessageBox.StandardButton.Ok.value

    def _pending_review_ids(self) -> list[str]:
        """按字库顺序返回全部待审核记录，包括来源缺失的记录。"""
        if not self._service:
            return []
        variants = self._service.get_variants()
        pending_ids: list[str] = []
        seen_ids: set[str] = set()
        for variant_ids in self._service.get_glyph_groups().values():
            for variant_id in variant_ids:
                if variant_id in seen_ids:
                    continue
                seen_ids.add(variant_id)
                detail = variants.get(variant_id, {})
                if detail.get("状态") == config.STATUS_PENDING_MANUAL_REVIEW:
                    pending_ids.append(variant_id)
        # 损坏字库可能仍有变体详情但缺少分组索引；批处理仍须处理
        # 这类记录，不能把阶段误报为已经完成。
        for variant_id, detail in variants.items():
            if variant_id in seen_ids:
                continue
            if detail.get("状态") == config.STATUS_PENDING_MANUAL_REVIEW:
                pending_ids.append(variant_id)
        return pending_ids

    def _start_bulk_review(self, pending_ids: list[str]) -> None:
        if self._batch_running or not self._service or not pending_ids:
            return
        library_name = self._service.ziku_name
        library_path = self._service.ziku_dir
        total = len(pending_ids)
        self._set_batch_running(True, total)
        self._batch_started_at = time.perf_counter()
        try:
            worker = _BulkReviewWorker(
                lambda progress, cancel_check: _run_bulk_review(
                    library_name,
                    library_path,
                    list(pending_ids),
                    progress,
                    cancel_check,
                )
            )
            worker.setAutoDelete(False)
            worker.signals.progress.connect(self._bulk_review_progress)
            worker.signals.finished.connect(
                lambda result, task=worker: self._bulk_review_finished(result, task)
            )
            worker.signals.failed.connect(
                lambda message, task=worker: self._bulk_review_failed(message, task)
            )
            self._batch_worker = worker
            self._batch_pool.start(worker)
        except Exception as exc:
            elapsed = self._batch_elapsed_seconds()
            self._batch_worker = None
            self._batch_started_at = None
            self._set_batch_running(False)
            QMessageBox.critical(
                self,
                "完成手工审核失败",
                f"无法启动批量审核任务：{exc}\n\n"
                f"总耗时：{format_elapsed_time(elapsed)}",
            )

    def _set_batch_running(
        self,
        running: bool,
        total: int = 0,
        *,
        stopping: bool = False,
    ) -> None:
        self._batch_running = bool(running)
        self._main_splitter.setEnabled(not running)
        for action in self._shortcut_actions:
            action.setEnabled(not running)
        self._complete_button.setEnabled(not running and self._service is not None)
        self._home_button.setEnabled(not running)
        self._batch_progress_widget.setVisible(running)
        self._stop_batch_button.setEnabled(running and not stopping)
        self._stop_batch_button.setText(
            "正在停止…" if stopping else "停止批量审核"
        )
        if running:
            if stopping:
                self._batch_progress_label.setText(
                    "正在停止批量审核，请等待当前单字安全结束…"
                )
            else:
                self._batch_progress_label.setText(f"准备手工审核 0 / {total}")
                self._batch_progress_bar.setValue(0)
                self._batch_progress_bar.setFormat(f"0% · 0 / {total}")

    def _request_stop_bulk_review(self) -> None:
        """确认后请求工作线程在单字事务边界停止。"""
        worker = self._batch_worker
        if worker is None or worker.is_cancel_requested():
            return
        if not self._confirm_stop_bulk_review():
            return
        # 确认框显示期间任务可能已经结束，不能重新锁定已恢复的页面。
        if worker is not self._batch_worker or not self._batch_running:
            return
        worker.request_cancel()
        self._set_batch_running(True, stopping=True)
        self._batch_progress_bar.setFormat("正在停止… · %p%")

    def _confirm_stop_bulk_review(self) -> bool:
        dialog = QMessageBox(self)
        dialog.setWindowTitle("停止批量审核")
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setText("确定停止当前批量手工审核任务吗？")
        dialog.setInformativeText(
            "已经成功保存的字形会保留；当前单字保存事务会先完整提交或回滚，"
            "尚未处理的字形仍保持待审核状态。"
        )
        stop_button = dialog.addButton(
            "停止批量审核",
            QMessageBox.ButtonRole.AcceptRole,
        )
        continue_button = dialog.addButton(
            "继续运行",
            QMessageBox.ButtonRole.RejectRole,
        )
        dialog.setDefaultButton(continue_button)
        dialog.setEscapeButton(continue_button)
        dialog.exec()
        return dialog.clickedButton() is stop_button

    def _bulk_review_progress(self, progress: object) -> None:
        if not isinstance(progress, dict) or not self._batch_running:
            return
        worker = self._batch_worker
        if worker is not None and worker.is_cancel_requested():
            self._set_batch_running(True, stopping=True)
            self._batch_progress_bar.setFormat("正在停止… · %p%")
            return
        current = max(0, int(progress.get("当前", 0)))
        processed = max(0, int(progress.get("已处理", current)))
        total = max(0, int(progress.get("总数", 0)))
        label = str(progress.get("字形", ""))
        percent = round(processed * 100 / total) if total else 0
        self._batch_progress_label.setText(
            f"正在自动审核 {current} / {total}：{label}"
        )
        self._batch_progress_bar.setValue(percent)
        self._batch_progress_bar.setFormat(f"%p% · {processed} / {total}")

    def _bulk_review_finished(
        self,
        result: object,
        worker: _BulkReviewWorker,
    ) -> None:
        if self._batch_worker is not worker:
            return
        summary = result if isinstance(result, dict) else {}
        elapsed = self._batch_elapsed_seconds(summary)
        self._batch_started_at = None
        reload_error = self._finish_bulk_review(worker)
        stopped = bool(summary.get("已停止", False))
        succeeded = int(summary.get("成功", 0))
        skipped = int(summary.get("跳过", 0))
        failed = int(summary.get("失败", 0))
        unprocessed = int(summary.get("未处理", 0))
        state = "已停止" if stopped else "已完成"
        message = (
            f"批量手工审核{state}：成功 {succeeded} 个，跳过 {skipped} 个，"
            f"失败 {failed} 个，未处理 {unprocessed} 个。"
        )
        failures = summary.get("失败详情", [])
        if failed:
            details = "\n".join(
                f"{variant_id}：{reason}"
                for variant_id, reason in list(failures)[:8]
            )
            if details:
                message += f"\n\n{details}"
        message += f"\n\n总耗时：{format_elapsed_time(elapsed)}"
        if reload_error:
            message += f"\n\n批量审核已经结束，但页面刷新失败：{reload_error}"
            QMessageBox.critical(self, "完成手工审核", message)
        elif failed:
            QMessageBox.warning(self, "手工审核已停止" if stopped else "完成手工审核", message)
        else:
            QMessageBox.information(
                self,
                "手工审核已停止" if stopped else "完成手工审核",
                message,
            )
        self.status_message.emit(message.splitlines()[0])

    def _bulk_review_failed(
        self,
        message: str,
        worker: _BulkReviewWorker,
    ) -> None:
        if self._batch_worker is not worker:
            return
        elapsed = self._batch_elapsed_seconds()
        self._batch_started_at = None
        reload_error = self._finish_bulk_review(worker)
        detail = (
            f"批量审核任务异常终止：{message}\n\n"
            f"总耗时：{format_elapsed_time(elapsed)}"
        )
        if reload_error:
            detail += f"\n\n页面刷新失败：{reload_error}"
        QMessageBox.critical(
            self,
            "完成手工审核失败",
            detail,
        )
        self.status_message.emit("批量审核任务异常终止。")

    def _batch_elapsed_seconds(self, result: dict[str, Any] | None = None) -> float:
        if result is not None:
            try:
                elapsed = float(result.get("总耗时秒", -1.0))
            except (TypeError, ValueError):
                elapsed = -1.0
            if elapsed >= 0.0:
                return elapsed
        if self._batch_started_at is None:
            return 0.0
        return max(0.0, time.perf_counter() - self._batch_started_at)

    def _finish_bulk_review(self, worker: _BulkReviewWorker) -> str:
        """刷新批处理结果，并确保任何刷新异常都不会遗留页面锁定。"""
        if self._batch_worker is not worker:
            return ""
        self._batch_worker = None
        reload_error = ""
        try:
            self._reload_after_bulk_review()
        except Exception as exc:
            reload_error = str(exc) or exc.__class__.__name__
        finally:
            self._set_batch_running(False)
        if self._service is not None:
            self.summary_changed.emit(self._service)
        return reload_error

    def _reload_after_bulk_review(self) -> None:
        if not self._service:
            return
        current_variant = self._current_variant_id
        library_name = self._service.ziku_name
        library_path = self._service.ziku_dir
        self._service = GlyphService.open(library_name, library_path)
        self._list_thumbnail_timer.stop()
        self._list_thumbnail_generation += 1
        self._list_thumbnail_cache.clear()
        self._list_thumbnail_key_by_path.clear()
        self._current_variant_id = ""
        self._current_status = ""
        self._source_path = ""
        self._source_stage = ""
        self._populate_variants(select_variant=current_variant)

    def _record_workflow(self, record: dict[str, Any]) -> WorkflowStatus:
        projection = record.get("projection")
        if isinstance(projection, WorkflowStageProjection):
            return projection.workflow
        workflow = record.get("workflow")
        if isinstance(workflow, WorkflowStatus):
            return workflow
        status = str(record.get("status", ""))
        stage = {
            config.STATUS_PENDING_MANUAL_REVIEW: STAGE_PENDING_REVIEW,
            config.STATUS_REVIEWED: STAGE_PENDING_COORDINATION,
            config.STATUS_FINISHED: STAGE_COMPLETED,
        }.get(status, STAGE_PENDING_REVIEW)
        markers = (
            (MARKER_STRUCTURE_REVIEW,)
            if self._record_requires_structure_review(record)
            else ()
        )
        return WorkflowStatus(
            stage=stage,
            markers=markers,
            ink_status=INK_STATUS_NOT_APPLICABLE,
            has_valid_finished=stage == STAGE_COMPLETED,
        )

    def _record_projection(
        self,
        record: dict[str, Any],
    ) -> WorkflowStageProjection | None:
        projection = record.get("projection")
        return projection if isinstance(projection, WorkflowStageProjection) else None

    def _record_phase_status(self, record: dict[str, Any]) -> str:
        projection = self._record_projection(record)
        if projection is not None and projection.admitted:
            return projection.status
        raw_status = str(record.get("status", ""))
        return (
            STATUS_REVIEWED
            if raw_status in {config.STATUS_REVIEWED, config.STATUS_FINISHED}
            else STAGE_PENDING_REVIEW
        )

    def _record_phase_markers(self, record: dict[str, Any]) -> tuple[str, ...]:
        projection = self._record_projection(record)
        return projection.markers if projection is not None else self._record_workflow(record).markers

    def _resolve_record_projection(
        self,
        record: dict[str, Any],
        *,
        dirty: bool = False,
    ) -> WorkflowStageProjection | None:
        detail = record.get("detail")
        if self._service is None or not isinstance(detail, dict):
            return None
        directories = self._service.get_workflow_dirs()
        return project_stage_status(
            detail,
            self._service.get_coordination_summary(),
            directories.get("成品", ""),
            PHASE_REVIEW,
            dirty=dirty,
        )

    def _resolve_record_workflow(
        self,
        record: dict[str, Any],
        *,
        dirty: bool = False,
    ) -> WorkflowStatus:
        projection = self._resolve_record_projection(record, dirty=dirty)
        if projection is not None:
            return projection.workflow
        workflow = self._record_workflow(record)
        if not dirty or MARKER_UNSAVED in workflow.markers:
            return workflow
        return WorkflowStatus(
            stage=workflow.stage,
            markers=(MARKER_UNSAVED, *workflow.markers),
            ink_status=workflow.ink_status,
            has_valid_finished=workflow.has_valid_finished,
        )

    def _populate_variants(
        self,
        _value: str = "",
        select_variant: str = "",
        *,
        select_first: bool = False,
    ) -> None:
        if not self._service:
            return
        desired = "" if select_first else (select_variant or self._current_variant_id)
        records = self._eligible_records()
        self._records_by_id = {str(record["variant_id"]): record for record in records}
        status_filter = self._filter_combo.currentText()
        query = self._search_edit.text().strip().lower()
        filtered: list[dict[str, Any]] = []
        for record in records:
            status = self._record_phase_status(record)
            if status_filter != PHASE_FILTER_ALL and status != status_filter:
                continue
            searchable = " ".join(
                str(record.get(key, ""))
                for key in (
                    "char",
                    "variant_id",
                    "filename",
                    "original_filename",
                )
            ).lower()
            if query and query not in searchable:
                continue
            filtered.append(record)
        self._sort_records(filtered)
        groups: dict[str, list[dict[str, Any]]] = {}
        for record in filtered:
            groups.setdefault(str(record["char"]), []).append(record)
        all_groups: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            all_groups.setdefault(str(record["char"]), []).append(record)

        self._item_tree.blockSignals(True)
        self._item_tree.clear()
        self._variant_ids.clear()
        self._item_nodes.clear()
        target_node: QTreeWidgetItem | None = None
        for char, group_records in groups.items():
            all_group_records = all_groups.get(char, group_records)
            completed_count = sum(
                self._record_phase_status(record) == STATUS_REVIEWED
                for record in all_group_records
            )
            group_status = f"已审核 {completed_count}/{len(all_group_records)}"
            marked_count = sum(
                bool(self._record_phase_markers(record))
                for record in all_group_records
            )
            group_color_status = (
                STATUS_REVIEWED
                if completed_count == len(all_group_records)
                else STAGE_PENDING_REVIEW
            )
            parent = QTreeWidgetItem(
                [
                    f"{char}（{len(all_group_records)}个字形）",
                    "",
                ]
            )
            parent.setFlags(parent.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            parent_font = parent.font(0)
            parent_font.setBold(True)
            parent.setFont(0, parent_font)
            parent.setFont(1, parent_font)
            set_two_line_status(
                parent,
                1,
                group_status,
                f"提示 {marked_count}" if marked_count else "—",
                PHASE_STATUS_COLORS[group_color_status],
                self.STRUCTURE_RISK_COLOR if marked_count else None,
            )
            parent.setToolTip(
                0,
                f"{char}，当前显示 {len(group_records)} / 全部 {len(all_group_records)} 个字形",
            )
            self._item_tree.addTopLevelItem(parent)

            for record in group_records:
                variant_id = str(record["variant_id"])
                status = self._record_phase_status(record)
                markers = self._record_phase_markers(record)
                marker_text = "、".join(markers) or "—"
                item = QTreeWidgetItem(
                    parent,
                    [self._variant_list_text(record), ""],
                )
                source_path = str(record["source_path"])
                cached_icon = (
                    self._cached_glyph_thumbnail(source_path)
                    if source_path
                    else None
                )
                if not source_path:
                    item.setIcon(0, self._thumbnail_placeholder())
                elif cached_icon is not None:
                    item.setIcon(0, cached_icon)
                elif len(filtered) <= self.LIST_THUMBNAIL_SYNC_LIMIT:
                    item.setIcon(0, self._glyph_thumbnail(source_path))
                else:
                    item.setIcon(0, self._thumbnail_placeholder())
                item.setData(0, Qt.ItemDataRole.UserRole, variant_id)
                item.setSizeHint(0, QSize(0, 52))
                item.setToolTip(0, self._record_tooltip(record))
                if MARKER_STRUCTURE_REVIEW in markers:
                    item.setForeground(0, QBrush(self.STRUCTURE_RISK_COLOR))
                set_two_line_status(
                    item,
                    1,
                    status,
                    marker_text,
                    PHASE_STATUS_COLORS[status],
                    self.STRUCTURE_RISK_COLOR if markers else None,
                )
                self._variant_ids.append(variant_id)
                self._item_nodes.append(item)
                if variant_id == desired:
                    target_node = item
            parent.setExpanded(True)
        largest_group = max(
            (len(group_records) for group_records in all_groups.values()),
            default=0,
        )
        self._item_tree_columns.set_protected_minimum(
            1,
            max(
                self._item_tree.fontMetrics().horizontalAdvance(
                    f"已审核 {largest_group}/{largest_group}"
                )
                + 36,
                self._item_tree.fontMetrics().horizontalAdvance("状态与提示") + 24,
            ),
        )
        self._item_tree.blockSignals(False)

        self._list_count_label.setText(f"显示 / 总数：{len(filtered)} / {len(records)}")
        self._refresh_progress(records)
        if target_node is None and filtered and (
            select_first
            or not (self._current_variant_id and self._canvas.is_dirty)
        ):
            target_node = self._item_nodes[0]
        if target_node is not None:
            if target_node.parent() is not None:
                target_node.parent().setExpanded(True)
            self._item_tree.setCurrentItem(target_node)
            if str(target_node.data(0, Qt.ItemDataRole.UserRole)) == self._current_variant_id:
                self._update_navigation_buttons()
        elif not filtered and not self._canvas.is_dirty:
            self._clear_current()
        elif not self._current_variant_id:
            self._clear_current()
        else:
            self._update_navigation_buttons()
        if len(filtered) > self.LIST_THUMBNAIL_SYNC_LIMIT:
            self._schedule_list_thumbnail_loads()

    def _execute_search(self, _checked: bool = False) -> None:
        """按回车或按钮执行一次全新的搜索，并定位第一条结果。"""

        self._populate_variants(select_first=True)

    def _restore_search_when_cleared(self, text: str) -> None:
        """删除检索文字后立即恢复当前阶段筛选下的全部字形。"""

        if not text.strip():
            self._populate_variants()

    def _eligible_records(self) -> list[dict[str, Any]]:
        if not self._service:
            return []
        directories = self._service.get_workflow_dirs()
        coordination_summary = self._service.get_coordination_summary()
        finished_dir = directories.get("成品", "")
        records: list[dict[str, Any]] = []
        variants = self._service.get_variants()
        order = 0
        for char, variant_ids in self._service.get_glyph_groups().items():
            for variant_index, variant_id in enumerate(variant_ids, start=1):
                detail = variants.get(variant_id, {})
                status = str(detail.get("状态", ""))
                projection = project_stage_status(
                    detail,
                    coordination_summary,
                    finished_dir,
                    PHASE_REVIEW,
                    dirty=(
                        variant_id == self._current_variant_id
                        and self._canvas.is_dirty
                    ),
                )
                if not projection.admitted:
                    continue
                workflow_status = projection.workflow
                source_path = ""
                stage = ""
                filename = ""
                if str(detail.get("审核文件", "") or ""):
                    source_options = (("手工审核", "审核文件", "手工审核稿"),)
                else:
                    source_options = (("优化预览", "中间文件", "自动优化稿"),)
                for directory_key, file_key, display_stage in source_options:
                    candidate = str(detail.get(file_key, "") or "")
                    if not candidate:
                        continue
                    if not filename:
                        filename = candidate
                        stage = display_stage
                    path = resolve_safe_stage_file(
                        directories[directory_key],
                        candidate,
                    )
                    if path:
                        source_path = path
                        stage = display_stage
                        filename = candidate
                        break
                if not filename:
                    filename = str(detail.get("原始文件", "") or "（文件信息缺失）")
                if not stage:
                    stage = "阶段文件"
                optimization = detail.get("自动优化", {})
                if not isinstance(optimization, dict):
                    optimization = {}
                scheme = optimization.get("方案", {})
                if not isinstance(scheme, dict):
                    scheme = {}
                structure_review = scheme.get("结构复核", {})
                if not isinstance(structure_review, dict):
                    structure_review = {}
                records.append(
                    {
                        "variant_id": variant_id,
                        "detail": detail,
                        "projection": projection,
                        "workflow": workflow_status,
                        "char": str(detail.get("归属字", char)),
                        "status": status,
                        "source_path": source_path,
                        "stage": stage,
                        "filename": filename,
                        "original_filename": str(detail.get("原始文件", "")),
                        "variant_index": variant_index,
                        "order": order,
                        "structure_review_status": str(
                            structure_review.get("状态", "")
                        ),
                        "structure_review_reason": str(
                            structure_review.get("原因", "")
                        ),
                    }
                )
                order += 1
        return records

    def _sort_records(self, records: list[dict[str, Any]]) -> None:
        ordering = self._sort_combo.currentText()
        if ordering == "文件名顺序":
            records.sort(key=lambda record: natural_key(str(record["filename"])))
        elif ordering == "导入顺序":
            records.sort(key=lambda record: int(record["order"]))
        else:
            records.sort(
                key=lambda record: (
                    pinyin_natural_key(str(record["char"])),
                    natural_key(str(record["filename"])),
                )
            )

    def _on_variant_selected(
        self,
        current: QTreeWidgetItem | None,
        previous: QTreeWidgetItem | None,
    ) -> None:
        if current is None or not self._service:
            return
        variant_id = str(current.data(0, Qt.ItemDataRole.UserRole) or "")
        if not variant_id:
            self._restore_valid_tree_selection(previous)
            return
        if variant_id == self._current_variant_id:
            self._update_navigation_buttons()
            return
        if self._canvas.is_dirty and not self._confirm_discard():
            self._item_tree.blockSignals(True)
            previous_node = self._node_for_variant(self._current_variant_id)
            self._item_tree.setCurrentItem(previous_node)
            self._item_tree.blockSignals(False)
            self._update_navigation_buttons()
            return
        detail = self._service.get_variant(variant_id) or {}
        record = self._records_by_id.get(variant_id)
        workflow = self._record_workflow(record) if record else resolve_workflow_status(
            detail,
            self._service.get_coordination_summary(),
            self._service.get_workflow_dirs().get("成品", ""),
        )
        phase_status = (
            self._record_phase_status(record)
            if record is not None
            else STAGE_PENDING_REVIEW
        )
        source_path = (
            str(record.get("source_path", ""))
            if record
            else self._resolve_source_path(detail)
        )
        editable = workflow.stage != STAGE_PENDING_OPTIMIZATION
        image = QImage(source_path)
        if image.isNull():
            self._show_unavailable_record(
                variant_id,
                detail,
                record,
                workflow,
            )
            return

        self._current_variant_id = variant_id
        self._current_status = str(detail.get("状态", ""))
        self._source_path = source_path
        self._source_stage = str(record.get("stage", "")) if record else self._source_stage_for_path(source_path)
        source_preview = self._to_review_image(image)
        is_automatic_source = self._source_stage != "手工审核稿"
        params = detail.get("变换参数", {}) if is_automatic_source else {}
        if not isinstance(params, dict):
            params = {}
        prepared_image, _prepared_origin = _prepare_review_source(
            source_preview,
            (self._canvas_width, self._canvas_height),
            params if editable and is_automatic_source else None,
            normalize_initial=(
                editable
                and is_automatic_source
                and self._current_status == config.STATUS_PENDING_MANUAL_REVIEW
            ),
        )
        self._canvas.set_image(
            prepared_image,
            (self._canvas_width, self._canvas_height),
            source_preview=source_preview,
        )
        self._sync_transform_controls(self._canvas.transform())
        self._update_ink_swatch(self._canvas.brush_color())

        char = str(detail.get("归属字", variant_id.split("-")[0]))
        filename = os.path.basename(source_path)
        self._glyph_label.setText(char)
        self._file_label.setText(f"{filename} · {variant_id}")
        self._status_label.setText(phase_status)
        self._status_label.setStyleSheet(
            f"color: {PHASE_STATUS_COLORS[phase_status]};"
        )
        self._source_label.setText(f"来源：{self._source_stage} · {filename}")
        self._draft_source_label.setText(self._source_stage)
        self._draft_file_label.setText(filename)
        self._draft_status_label.setText(phase_status)
        if editable:
            self._draft_save_label.setText("无未保存修改")
            self._save_state_label.setText("无未保存修改")
        else:
            readonly_message = "只读：请先完成自动优化"
            self._draft_save_label.setText(readonly_message)
            self._save_state_label.setText(readonly_message)
            self._source_label.setText(
                f"来源：{self._source_stage} · {filename}　请先完成自动优化"
            )
            self.status_message.emit("当前字形仍在待优化阶段，请先完成自动优化。")
        self._set_controls_enabled(editable)
        self._update_navigation_buttons()

    def _show_glyph_context_menu(self, position: object) -> None:
        node = self._item_tree.itemAt(position)
        if node is None:
            return
        variant_id = str(node.data(0, Qt.ItemDataRole.UserRole) or "")
        if not variant_id:
            return
        self._item_tree.setCurrentItem(node)
        menu = QMenu(self)
        action = menu.addAction("修正字形名称…")
        action.setEnabled(not self._batch_running)
        action.triggered.connect(self._rename_current_glyph)
        menu.exec(self._item_tree.viewport().mapToGlobal(position))

    def _rename_current_glyph(self) -> None:
        if not self._service or not self._current_variant_id:
            QMessageBox.information(self, "修正字形名称", "请先选择一个具体字形。")
            return
        if self._batch_running:
            QMessageBox.information(
                self,
                "暂时不能修改名称",
                "当前正在执行批量手工审核，请等待任务结束后重试。",
            )
            return
        if self._canvas.is_dirty:
            dialog = QMessageBox(self)
            dialog.setWindowTitle("当前稿件尚未保存")
            dialog.setIcon(QMessageBox.Icon.Question)
            dialog.setText("当前字形有未保存修改，请先决定如何处理。")
            dialog.setStandardButtons(
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel
            )
            dialog.setDefaultButton(QMessageBox.StandardButton.Save)
            save_button = dialog.button(QMessageBox.StandardButton.Save)
            discard_button = dialog.button(QMessageBox.StandardButton.Discard)
            cancel_button = dialog.button(QMessageBox.StandardButton.Cancel)
            if save_button is not None:
                save_button.setText("保存修改")
            if discard_button is not None:
                discard_button.setText("放弃修改")
            if cancel_button is not None:
                cancel_button.setText("取消")
            choice = dialog.exec()
            if choice == QMessageBox.StandardButton.Cancel.value:
                return
            if choice == QMessageBox.StandardButton.Save.value:
                if not self.save_current():
                    return
            else:
                self._canvas.discard_changes()

        variant_id = self._current_variant_id
        result = run_glyph_rename_dialog(self, self._service, variant_id)
        if result is None:
            return
        self._list_thumbnail_timer.stop()
        self._list_thumbnail_generation += 1
        self._list_thumbnail_cache.clear()
        self._list_thumbnail_key_by_path.clear()
        self._current_variant_id = ""
        self._current_status = ""
        self._source_path = ""
        self._source_stage = ""
        self._canvas.clear_image()
        self._populate_variants(select_variant=variant_id)
        self.summary_changed.emit(self._service)
        self.status_message.emit(f"字形名称已修正为 {result.get('新文件名', '')}")
        QMessageBox.information(
            self,
            "名称修改完成",
            f"字形已修正为“{result.get('新归属字', '')}”，各阶段文件名已同步更新。",
        )

    def _show_unavailable_record(
        self,
        variant_id: str,
        detail: dict[str, Any],
        record: dict[str, Any] | None,
        workflow: WorkflowStatus,
    ) -> None:
        """缺少阶段文件时清空旧稿，保留列表导航并明确当前阻塞原因。"""
        self._current_variant_id = variant_id
        self._current_status = str(detail.get("状态", ""))
        self._source_path = ""
        self._source_stage = str(record.get("stage", "阶段文件")) if record else "阶段文件"
        self._canvas.clear_image()
        char = str(detail.get("归属字", variant_id.split("-")[0]))
        filename = str(record.get("filename", "")) if record else ""
        phase_status = (
            self._record_phase_status(record)
            if record is not None
            else STAGE_PENDING_REVIEW
        )
        self._glyph_label.setText(char)
        self._file_label.setText(f"{filename or '文件不可用'} · {variant_id}")
        self._status_label.setText(phase_status)
        self._status_label.setStyleSheet(
            f"color: {PHASE_STATUS_COLORS[phase_status]};"
        )
        self._source_label.setText("来源：文件不可用")
        self._draft_source_label.setText(self._source_stage)
        self._draft_file_label.setText(filename or "-")
        self._draft_status_label.setText(phase_status)
        prompt = "审核稿文件不可用，请核对字库文件"
        self._draft_save_label.setText(prompt)
        self._save_state_label.setText(prompt)
        self._set_controls_enabled(False)
        self._update_navigation_buttons()
        self.status_message.emit(prompt)

    def _current_record_is_editable(self) -> bool:
        record = self._records_by_id.get(self._current_variant_id)
        return bool(
            record is not None
            and self._record_workflow(record).stage != STAGE_PENDING_OPTIMIZATION
        )

    def _restore_valid_tree_selection(
        self,
        previous: QTreeWidgetItem | None,
    ) -> None:
        """父组只负责折叠展开，树的 current 始终停留在有效字形上。"""
        previous_id = (
            str(previous.data(0, Qt.ItemDataRole.UserRole) or "")
            if previous is not None
            else ""
        )
        target_node: QTreeWidgetItem | None = None
        for variant_id in (previous_id, self._current_variant_id):
            target_node = self._node_for_variant(variant_id)
            if target_node is not None:
                break
        with QSignalBlocker(self._item_tree):
            self._item_tree.setCurrentItem(target_node)
        self._update_navigation_buttons()

    def _resolve_source_path(self, detail: dict[str, Any]) -> str:
        if not self._service:
            return ""
        directories = self._service.get_workflow_dirs()
        reviewed_name = str(detail.get("审核文件", "") or "")
        if reviewed_name:
            return resolve_safe_stage_file(
                directories["手工审核"],
                reviewed_name,
            )
        return resolve_safe_stage_file(
            directories["优化预览"],
            detail.get("中间文件"),
        )

    def _source_stage_for_path(self, path: str) -> str:
        if not self._service or not path:
            return ""
        reviewed_dir = os.path.normcase(os.path.abspath(self._service.get_workflow_dirs()["手工审核"]))
        parent = os.path.normcase(os.path.abspath(os.path.dirname(path)))
        return "手工审核稿" if parent == reviewed_dir else "自动优化稿"

    @staticmethod
    def _to_review_image(image: QImage) -> QImage:
        return _to_review_image(image)

    def _move_selection(self, offset: int) -> None:
        current = self._item_tree.currentItem()
        if current is None:
            return
        variant_id = str(current.data(0, Qt.ItemDataRole.UserRole) or "")
        if variant_id not in self._variant_ids:
            return
        target = self._variant_ids.index(variant_id) + offset
        if 0 <= target < len(self._item_nodes):
            target_node = self._item_nodes[target]
            if target_node.parent() is not None:
                target_node.parent().setExpanded(True)
            self._item_tree.setCurrentItem(target_node)

    def _request_home(self) -> None:
        if self._batch_running or self._save_running:
            self.status_message.emit("正在保存手工审核结果，请等待任务结束。")
            return
        if self._canvas.is_dirty and not self._confirm_discard():
            return
        self.home_requested.emit()

    def _confirm_discard(self) -> bool:
        dialog = QMessageBox(self)
        dialog.setWindowTitle("尚未保存")
        dialog.setIcon(QMessageBox.Icon.Question)
        dialog.setText("当前字形有未保存的修改，请先决定如何处理。")
        dialog.setStandardButtons(
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel
        )
        dialog.setDefaultButton(QMessageBox.StandardButton.Save)
        save_button = dialog.button(QMessageBox.StandardButton.Save)
        discard_button = dialog.button(QMessageBox.StandardButton.Discard)
        cancel_button = dialog.button(QMessageBox.StandardButton.Cancel)
        if save_button is not None:
            save_button.setText("保存修改")
        if discard_button is not None:
            discard_button.setText("放弃修改")
        if cancel_button is not None:
            cancel_button.setText("取消")
            dialog.setEscapeButton(cancel_button)
        choice = dialog.exec()
        if choice == QMessageBox.StandardButton.Save.value:
            return self.save_current()
        if choice != QMessageBox.StandardButton.Discard.value:
            return False
        self._canvas.discard_changes()
        return True

    def _set_tool(self, tool: str) -> None:
        self._canvas.set_tool(tool)
        if tool == ReviewCanvas.TOOL_TRANSFORM:
            self._parameters_stack.setCurrentWidget(self._transform_panel)
            self._tool_mode_label.setText("自由变换")
            self._fit_parameters_stack_to_current()
            return
        self._parameters_stack.setCurrentWidget(self._pixel_panel)
        is_brush = tool == ReviewCanvas.TOOL_BRUSH
        self._tool_mode_label.setText("画笔" if is_brush else "橡皮")
        self._pixel_title.setText("画笔" if is_brush else "橡皮")
        self._ink_controls.setVisible(is_brush)
        self._fit_parameters_stack_to_current()

    def _fit_parameters_stack_to_current(self) -> None:
        """让参数栈只占当前工具所需高度，剩余空间留到稿件信息之后。"""
        current = self._parameters_stack.currentWidget()
        if current is None:
            return
        current_layout = current.layout()
        if current_layout is not None:
            current_layout.invalidate()
            current_layout.activate()
        height = max(
            current.minimumSizeHint().height(),
            current.sizeHint().height(),
        )
        self._parameters_stack.setFixedHeight(height)
        self._parameters_stack.updateGeometry()

    def _update_transform_value_labels(self) -> None:
        self._scale_value_label.setText(
            f"{self._slider_position_to_percent(self._scale_slider.value())}%"
        )
        self._stretch_w_value_label.setText(
            f"{self._slider_position_to_percent(self._stretch_w_slider.value())}%"
        )
        self._stretch_h_value_label.setText(
            f"{self._slider_position_to_percent(self._stretch_h_slider.value())}%"
        )
        self._rotation_value_label.setText(f"{self._rotation_slider.value()}°")

    def _on_percent_slider_changed(self, slider: QSlider, field: str) -> None:
        percent = self._slider_position_to_percent(slider.value())
        self._on_transform_slider_changed(slider, field, percent / 100.0)

    def _commit_percent_slider(self, slider: QSlider, field: str) -> None:
        self._update_transform_value_labels()
        percent = self._slider_position_to_percent(slider.value())
        self._apply_transform_field(field, percent / 100.0)

    def _on_transform_slider_changed(
        self,
        slider: QSlider,
        field: str,
        value: float,
    ) -> None:
        """拖动期间只更新读数，键盘、轨道点击等非拖动操作立即提交。"""
        self._update_transform_value_labels()
        if slider.isSliderDown():
            return
        self._apply_transform_field(field, value)

    def _commit_transform_slider(
        self,
        slider: QSlider,
        field: str,
        divisor: float,
    ) -> None:
        """一次提交滑块拖动的最终值，避免每个刻度生成一条撤销记录。"""
        self._update_transform_value_labels()
        self._apply_transform_field(field, slider.value() / divisor)

    def _apply_transform_field(self, field: str, value: float) -> None:
        """只更新用户正在操作的字段，保留控件范围外的既有参数。"""
        if not self._canvas.has_image:
            return
        self._canvas.set_transform(**{field: value})

    def _sync_transform_controls(self, transform: dict[str, Any]) -> None:
        if not hasattr(self, "_offset_x_spin"):
            return
        x = round(self._number(transform.get("x"), 0.0))
        y = round(self._number(transform.get("y"), 0.0))
        scale = round(self._number(transform.get("scale"), 1.0) * 100)
        stretch_w = round(self._number(transform.get("stretch_w"), 1.0) * 100)
        stretch_h = round(self._number(transform.get("stretch_h"), 1.0) * 100)
        rotation = round(self._number(transform.get("rotation"), 0.0))
        with (
            QSignalBlocker(self._offset_x_spin),
            QSignalBlocker(self._offset_y_spin),
            QSignalBlocker(self._scale_slider),
            QSignalBlocker(self._stretch_w_slider),
            QSignalBlocker(self._stretch_h_slider),
            QSignalBlocker(self._rotation_slider),
        ):
            self._offset_x_spin.setValue(x)
            self._offset_y_spin.setValue(y)
            self._scale_slider.setValue(self._percent_to_slider_position(scale))
            self._stretch_w_slider.setValue(self._percent_to_slider_position(stretch_w))
            self._stretch_h_slider.setValue(self._percent_to_slider_position(stretch_h))
            self._rotation_slider.setValue(rotation)
        self._update_transform_value_labels()

    def _reset_transform(self) -> None:
        self._canvas.reset_transform()

    def _set_brush_size(self, value: int) -> None:
        self._brush_value_label.setText(f"{value} px")
        self._canvas.set_brush_size(value)

    def _sync_brush_size(self, value: int) -> None:
        """把画布快捷操作产生的笔触大小变化同步回参数面板。"""
        normalized = max(
            self._brush_slider.minimum(),
            min(self._brush_slider.maximum(), int(value)),
        )
        with QSignalBlocker(self._brush_slider):
            self._brush_slider.setValue(normalized)
        self._brush_value_label.setText(f"{normalized} px")

    def _set_pressure_enabled(self, enabled: bool) -> None:
        self._minimum_pressure_title.setEnabled(enabled)
        self._minimum_pressure_value_label.setEnabled(enabled)
        self._minimum_pressure_slider.setEnabled(enabled)
        self._canvas.set_pressure_enabled(enabled)

    def _set_minimum_pressure_ratio(self, value: int) -> None:
        self._minimum_pressure_value_label.setText(f"{value}%")
        self._canvas.set_minimum_pressure_ratio(value / 100.0)

    def _sample_ink_color(self) -> None:
        color = self._canvas.sample_ink_color()
        self._update_ink_swatch(color)
        self.status_message.emit("已从当前字形自动取得主墨色")

    def _update_ink_swatch(self, color: QColor) -> None:
        if not isinstance(color, QColor) or not color.isValid():
            color = QColor("#000000")
        self._ink_swatch.setStyleSheet(
            f"background: {color.name(QColor.NameFormat.HexRgb)}; border: 2px solid #d7dee8; border-radius: 3px;"
        )
        self._ink_swatch.setToolTip(f"当前墨色：{color.name(QColor.NameFormat.HexRgb).upper()}")

    def _set_grid_visible(self, visible: bool) -> None:
        self._canvas.set_grid_visible(visible)

    def _set_background_mode(self, mode: str) -> None:
        self._canvas.set_background_mode(mode)

    def _canvas_undo(self) -> None:
        self._canvas.undo()

    def _canvas_redo(self) -> None:
        self._canvas.redo()

    def _canvas_reset(self) -> None:
        self._canvas.reset_image()

    def _canvas_fit(self) -> None:
        self._canvas.fit_to_view()

    def _canvas_actual_size(self) -> None:
        self._canvas.actual_size()

    def _on_canvas_changed(self, dirty: bool) -> None:
        self._save_button.setText("保存修改稿 *" if dirty else "保存修改稿")
        save_state = "有未保存修改" if dirty else "无未保存修改"
        self._save_state_label.setText(save_state)
        self._draft_save_label.setText(save_state)
        record = self._records_by_id.get(self._current_variant_id)
        item = self._node_for_variant(self._current_variant_id)
        if record is not None and item is not None:
            projection = self._resolve_record_projection(record, dirty=dirty)
            if projection is not None:
                record["projection"] = projection
                workflow = projection.workflow
            else:
                workflow = self._resolve_record_workflow(record, dirty=dirty)
            record["workflow"] = workflow
            self._apply_record_workflow_to_node(record, item, workflow)
            self._update_group_item(item.parent())

    def _set_controls_enabled(self, enabled: bool) -> None:
        self._canvas.setEnabled(enabled)
        self._toolbar_widget.setEnabled(enabled)
        self._parameters_stack.setEnabled(enabled)
        self._restore_button.setEnabled(enabled)
        self._save_button.setEnabled(enabled)
        self._approve_button.setEnabled(enabled)
        if enabled:
            self._update_navigation_buttons()
        else:
            self._previous_button.setEnabled(False)
            self._next_button.setEnabled(False)

    def _update_navigation_buttons(self) -> None:
        current = self._item_tree.currentItem()
        variant_id = str(current.data(0, Qt.ItemDataRole.UserRole) or "") if current else ""
        row = self._variant_ids.index(variant_id) if variant_id in self._variant_ids else -1
        self._previous_button.setEnabled(row > 0)
        self._next_button.setEnabled(0 <= row < len(self._variant_ids) - 1)

    def _update_current_list_item(self, status: str, filename: str = "") -> None:
        item = self._node_for_variant(self._current_variant_id)
        if item is None:
            return
        record = self._records_by_id.get(self._current_variant_id)
        if record is not None:
            record["status"] = status
            detail = record.get("detail")
            if isinstance(detail, dict):
                detail["状态"] = status
            if filename:
                record["filename"] = filename
                record["stage"] = "手工审核稿"
                record["source_path"] = self._source_path
            projection = self._resolve_record_projection(record)
            if projection is not None:
                record["projection"] = projection
                workflow = projection.workflow
            else:
                workflow = self._resolve_record_workflow(record)
            record["workflow"] = workflow
            self._apply_record_workflow_to_node(record, item, workflow)
        if filename:
            item.setIcon(0, self._glyph_thumbnail(self._source_path))
        self._update_group_item(item.parent())

    def _apply_record_workflow_to_node(
        self,
        record: dict[str, Any],
        item: QTreeWidgetItem,
        workflow: WorkflowStatus,
    ) -> None:
        status = self._record_phase_status(record)
        markers = self._record_phase_markers(record)
        item.setText(0, self._variant_list_text(record))
        item.setToolTip(0, self._record_tooltip(record))
        item.setForeground(
            0,
            QBrush(self.STRUCTURE_RISK_COLOR)
            if MARKER_STRUCTURE_REVIEW in markers
            else QBrush(),
        )
        set_two_line_status(
            item,
            1,
            status,
            "、".join(markers) or "—",
            PHASE_STATUS_COLORS[status],
            self.STRUCTURE_RISK_COLOR if markers else None,
        )

    def _refresh_progress(self, records: list[dict[str, Any]] | None = None) -> None:
        records = records if records is not None else self._eligible_records()
        approved = sum(
            self._record_phase_status(record) == STATUS_REVIEWED
            for record in records
        )
        total = len(records)
        percent = round(approved * 100 / total) if total else 0
        self._count_label.setText(f"待审核 {total - approved}　已审核 {approved}")
        self._progress_bar.setValue(percent)

    def _node_for_variant(self, variant_id: str) -> QTreeWidgetItem | None:
        if variant_id not in self._variant_ids:
            return None
        return self._item_nodes[self._variant_ids.index(variant_id)]

    @staticmethod
    def _variant_list_text(record: dict[str, Any]) -> str:
        return f"字形{record.get('variant_index', 1)} · {record.get('filename', '')}"

    @staticmethod
    def _record_requires_structure_review(record: dict[str, Any]) -> bool:
        return bool(
            record.get("status") == config.STATUS_PENDING_MANUAL_REVIEW
            and record.get("structure_review_status") == "需人工核对"
        )

    def _record_tooltip(self, record: dict[str, Any]) -> str:
        status = self._record_phase_status(record)
        markers = self._record_phase_markers(record)
        tooltip = (
            f"{record.get('char', '')} · 字形{record.get('variant_index', 1)}\n"
            f"{record.get('stage', '')}：{record.get('filename', '')}\n"
            f"手工审核：{status}\n"
            f"提示：{'、'.join(markers) or '无'}"
        )
        if MARKER_STRUCTURE_REVIEW in markers:
            reason = str(record.get("structure_review_reason", "")).strip()
            tooltip += f"\n结构需核对：{reason or '结构保护未通过'}"
        return tooltip

    def _update_group_item(self, parent: QTreeWidgetItem | None) -> None:
        if parent is None:
            return
        visible_record = next(
            (
                self._records_by_id.get(
                    str(parent.child(index).data(0, Qt.ItemDataRole.UserRole) or "")
                )
                for index in range(parent.childCount())
            ),
            None,
        )
        if visible_record is None:
            return
        char = str(visible_record.get("char", ""))
        records = [
            record
            for record in self._records_by_id.values()
            if str(record.get("char", "")) == char
        ]
        completed = sum(
            self._record_phase_status(record) == STATUS_REVIEWED
            for record in records
        )
        group_status = (
            STATUS_REVIEWED
            if completed == len(records)
            else STAGE_PENDING_REVIEW
        )
        marked = sum(bool(self._record_phase_markers(record)) for record in records)
        set_two_line_status(
            parent,
            1,
            f"已审核 {completed}/{len(records)}",
            f"提示 {marked}" if marked else "—",
            PHASE_STATUS_COLORS[group_status],
            self.STRUCTURE_RISK_COLOR if marked else None,
        )

    def _clear_current(self) -> None:
        self._current_variant_id = ""
        self._current_status = ""
        self._source_path = ""
        self._source_stage = ""
        self._canvas.clear_image()
        self._glyph_label.setText("当前筛选无字形")
        self._file_label.clear()
        self._status_label.clear()
        self._source_label.setText("来源：-")
        self._draft_source_label.setText("-")
        self._draft_file_label.setText("-")
        self._draft_status_label.setText("-")
        self._draft_save_label.setText("-")
        self._set_controls_enabled(False)

    @staticmethod
    def _apply_status_color(label: QLabel, status: str) -> None:
        label.setStyleSheet(f"color: {config.STATUS_COLORS.get(status, '#d7dee8')};")

    def _glyph_thumbnail(self, path: str) -> QIcon:
        cached = self._cached_glyph_thumbnail(path)
        if cached is not None:
            return cached
        cache_key = self._thumbnail_cache_key(path)
        icon = self._render_glyph_thumbnail(path)
        if cache_key is not None:
            self._store_glyph_thumbnail(cache_key, icon)
        return icon

    def _cached_glyph_thumbnail(self, path: str) -> QIcon | None:
        normalized_path = self._normalized_thumbnail_path(path)
        if not normalized_path:
            return None
        stored_key = self._list_thumbnail_key_by_path.get(normalized_path)
        if stored_key is None:
            return None
        cache_key = self._thumbnail_cache_key(path)
        if cache_key is None or cache_key != stored_key:
            self._list_thumbnail_cache.pop(stored_key, None)
            self._list_thumbnail_key_by_path.pop(normalized_path, None)
            return None
        cached = self._list_thumbnail_cache.get(cache_key)
        if cached is not None:
            self._list_thumbnail_cache.move_to_end(cache_key)
        return cached

    def _store_glyph_thumbnail(
        self,
        cache_key: tuple[str, int, int],
        icon: QIcon,
    ) -> None:
        normalized_path = cache_key[0]
        stale_keys = [
            key
            for key in self._list_thumbnail_cache
            if key[0] == normalized_path and key != cache_key
        ]
        for key in stale_keys:
            self._list_thumbnail_cache.pop(key, None)
        self._list_thumbnail_cache[cache_key] = icon
        self._list_thumbnail_key_by_path[normalized_path] = cache_key
        self._list_thumbnail_cache.move_to_end(cache_key)
        while len(self._list_thumbnail_cache) > self.LIST_THUMBNAIL_CACHE_ITEMS:
            expired_key, _expired_icon = self._list_thumbnail_cache.popitem(last=False)
            if self._list_thumbnail_key_by_path.get(expired_key[0]) == expired_key:
                self._list_thumbnail_key_by_path.pop(expired_key[0], None)

    @staticmethod
    def _normalized_thumbnail_path(path: str) -> str:
        if not path:
            return ""
        return os.path.normcase(os.path.realpath(os.path.abspath(path)))

    @classmethod
    def _thumbnail_cache_key(cls, path: str) -> tuple[str, int, int] | None:
        normalized_path = cls._normalized_thumbnail_path(path)
        if not normalized_path:
            return None
        try:
            stat = os.stat(normalized_path)
        except OSError:
            return None
        return normalized_path, stat.st_mtime_ns, stat.st_size

    def _thumbnail_placeholder(self) -> QIcon:
        if self._list_thumbnail_placeholder is None:
            size = self._item_tree.iconSize()
            thumbnail = QPixmap(size)
            thumbnail.fill(QColor("#ffffff"))
            painter = QPainter(thumbnail)
            painter.setPen(QColor("#c7cdd5"))
            painter.drawRect(7, 7, max(1, size.width() - 15), max(1, size.height() - 15))
            painter.end()
            self._list_thumbnail_placeholder = QIcon(thumbnail)
        return self._list_thumbnail_placeholder

    @staticmethod
    def _render_glyph_thumbnail(path: str) -> QIcon:
        image = ReviewPage._decode_glyph_thumbnail(path, (38, 38))
        return ReviewPage._thumbnail_icon(image)

    @staticmethod
    def _decode_glyph_thumbnail(
        path: str,
        canvas_size: tuple[int, int],
    ) -> QImage:
        """后台只解码并合成 QImage，不创建依赖 GUI 线程的 QPixmap。"""
        width = max(1, int(canvas_size[0]))
        height = max(1, int(canvas_size[1]))
        thumbnail = QImage(width, height, QImage.Format.Format_ARGB32)
        thumbnail.fill(QColor("#ffffff"))
        source = ReviewPage._to_review_image(QImage(path))
        if not source.isNull():
            painter = QPainter(thumbnail)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            preview = source.scaled(
                max(1, width - 4),
                max(1, height - 4),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = (width - preview.width()) // 2
            y = (height - preview.height()) // 2
            painter.drawImage(x, y, preview)
            painter.end()
        return thumbnail

    @staticmethod
    def _thumbnail_icon(image: QImage) -> QIcon:
        return QIcon(QPixmap.fromImage(image))

    def _schedule_list_thumbnail_loads(self, _value: object = None) -> None:
        if len(self._variant_ids) > self.LIST_THUMBNAIL_SYNC_LIMIT:
            self._list_thumbnail_timer.start()

    def _visible_list_nodes(self) -> list[QTreeWidgetItem]:
        viewport = self._item_tree.viewport()
        viewport_rect = viewport.rect()
        current = self._item_tree.itemAt(QPoint(2, 2))
        visible: list[QTreeWidgetItem] = []
        while current is not None:
            rect = self._item_tree.visualItemRect(current)
            if not rect.isEmpty() and rect.top() > viewport_rect.bottom():
                break
            if not rect.isEmpty() and rect.intersects(viewport_rect):
                visible.append(current)
            current = self._item_tree.itemBelow(current)
        return visible

    def _load_visible_list_thumbnails(self) -> None:
        """按可见行分批提交缩略图后台解码。"""
        if len(self._variant_ids) <= self.LIST_THUMBNAIL_SYNC_LIMIT:
            return
        generation = self._list_thumbnail_generation
        jobs: list[tuple[tuple[str, int, int], str]] = []
        queued: set[tuple[str, int, int]] = set()
        for node in self._visible_list_nodes():
            variant_id = str(node.data(0, Qt.ItemDataRole.UserRole) or "")
            record = self._records_by_id.get(variant_id)
            if record is None:
                continue
            path = str(record.get("source_path", ""))
            cache_key = self._thumbnail_cache_key(path)
            if cache_key is None:
                continue
            cached = self._list_thumbnail_cache.get(cache_key)
            if cached is not None:
                self._list_thumbnail_cache.move_to_end(cache_key)
                node.setIcon(0, cached)
                continue
            tagged_key = (generation, cache_key)
            if tagged_key in self._list_thumbnail_inflight or cache_key in queued:
                continue
            jobs.append((cache_key, path))
            queued.add(cache_key)
            if len(jobs) >= self.LIST_THUMBNAIL_BATCH_SIZE:
                break
        if jobs:
            self._start_list_thumbnail_batch(jobs)

    def _start_list_thumbnail_batch(
        self,
        jobs: list[tuple[tuple[str, int, int], str]],
    ) -> None:
        generation = self._list_thumbnail_generation
        self._list_thumbnail_batch_id += 1
        batch_id = self._list_thumbnail_batch_id
        tagged_keys = {(generation, cache_key) for cache_key, _path in jobs}
        self._list_thumbnail_inflight.update(tagged_keys)
        canvas_size = (
            max(1, self._item_tree.iconSize().width()),
            max(1, self._item_tree.iconSize().height()),
        )

        def decode_batch() -> dict[str, object]:
            loaded = [
                (
                    cache_key,
                    ReviewPage._decode_glyph_thumbnail(path, canvas_size),
                )
                for cache_key, path in jobs
            ]
            return {
                "批次": batch_id,
                "代次": generation,
                "结果": loaded,
            }

        worker = FunctionWorker(decode_batch)
        self._list_thumbnail_workers[batch_id] = (worker, tagged_keys)
        worker.signals.finished.connect(self._list_thumbnail_batch_finished)
        worker.signals.failed.connect(self._list_thumbnail_batch_failed)
        self._list_thumbnail_pool.start(worker)

    def _list_thumbnail_batch_finished(self, result: object) -> None:
        payload = result if isinstance(result, dict) else {}
        batch_id = int(payload.get("批次", -1))
        generation = int(payload.get("代次", -1))
        self._release_list_thumbnail_batch(batch_id)
        if generation != self._list_thumbnail_generation:
            return
        loaded = payload.get("结果", [])
        loaded_keys: set[tuple[str, int, int]] = set()
        if isinstance(loaded, list):
            for entry in loaded:
                if not isinstance(entry, tuple) or len(entry) != 2:
                    continue
                cache_key, image = entry
                if (
                    not isinstance(cache_key, tuple)
                    or len(cache_key) != 3
                    or not isinstance(image, QImage)
                    or image.isNull()
                ):
                    continue
                if self._thumbnail_cache_key(str(cache_key[0])) != cache_key:
                    continue
                self._store_glyph_thumbnail(
                    cache_key,
                    self._thumbnail_icon(image),
                )
                loaded_keys.add(cache_key)
        if loaded_keys:
            self._apply_cached_list_thumbnails(loaded_keys)
        self._schedule_list_thumbnail_loads()

    def _list_thumbnail_batch_failed(self, _message: str) -> None:
        sender = self.sender()
        batch_id = next(
            (
                candidate
                for candidate, (worker, _keys) in self._list_thumbnail_workers.items()
                if worker.signals is sender
            ),
            -1,
        )
        pending = self._list_thumbnail_workers.get(batch_id)
        is_current = bool(
            pending
            and any(
                generation == self._list_thumbnail_generation
                for generation, _cache_key in pending[1]
            )
        )
        self._release_list_thumbnail_batch(batch_id)
        if is_current:
            self._schedule_list_thumbnail_loads()

    def _release_list_thumbnail_batch(self, batch_id: int) -> None:
        pending = self._list_thumbnail_workers.pop(batch_id, None)
        if pending is not None:
            self._list_thumbnail_inflight.difference_update(pending[1])

    def _apply_cached_list_thumbnails(
        self,
        loaded_keys: set[tuple[str, int, int]],
    ) -> None:
        for node in self._visible_list_nodes():
            variant_id = str(node.data(0, Qt.ItemDataRole.UserRole) or "")
            record = self._records_by_id.get(variant_id)
            if record is None:
                continue
            cache_key = self._thumbnail_cache_key(str(record.get("source_path", "")))
            if cache_key not in loaded_keys:
                continue
            cached = self._list_thumbnail_cache.get(cache_key)
            if cached is not None:
                node.setIcon(0, cached)

    @staticmethod
    def _toolbar_button(text: str, tooltip: str, checkable: bool = False) -> QToolButton:
        button = QToolButton()
        button.setText(text)
        button.setToolTip(tooltip)
        button.setCheckable(checkable)
        button.setMinimumHeight(30)
        width = button.fontMetrics().horizontalAdvance(text) + 22
        button.setFixedWidth(width)
        button.setStyleSheet(
            "QToolButton { padding: 0 8px; border: 1px solid #37404d; border-radius: 5px; background: #282f3a; }"
            "QToolButton:hover { border-color: #4da3ff; background: #303947; }"
            "QToolButton:checked { border-color: #4da3ff; background: #294d75; color: #ffffff; }"
            "QToolButton:disabled { color: #68717e; background: #242a33; border-color: #303640; }"
        )
        return button

    @staticmethod
    def _fit_navigation_button(button: QToolButton) -> None:
        """为中文导航文字预留边框、内边距和系统缩放余量。"""

        text_width = button.fontMetrics().horizontalAdvance(button.text())
        button.setFixedWidth(max(76, text_width + 42))

    def showEvent(self, event: Any) -> None:
        """主题字体在首次显示时才稳定，随后重新核算导航文字宽度。"""

        super().showEvent(event)
        self._fit_navigation_button(self._previous_button)
        self._fit_navigation_button(self._next_button)

    def shutdown(self) -> None:
        """关闭程序时收拢保存任务并释放后台缩略图。"""

        self._list_thumbnail_generation += 1
        self._list_thumbnail_timer.stop()
        if self._batch_worker is not None:
            self._batch_worker.request_cancel()
        self._list_thumbnail_pool.clear()
        self._batch_pool.clear()
        self._save_pool.clear()
        self._save_pool.waitForDone()
        self._list_thumbnail_workers.clear()
        self._list_thumbnail_inflight.clear()
        self._list_thumbnail_cache.clear()

    @staticmethod
    def _section_title(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet("font-size: 15px; font-weight: 700;")
        return label

    @staticmethod
    def _vertical_separator() -> QFrame:
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        return separator

    @staticmethod
    def _horizontal_separator() -> QFrame:
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        return separator

    @staticmethod
    def _number(value: object, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _percent_to_slider_position(cls, percent: int | float) -> int:
        """把百分比映射到以 100% 为几何中点的滑块位置。"""
        normalized = max(
            cls.TRANSFORM_PERCENT_MIN,
            min(cls.TRANSFORM_PERCENT_MAX, round(float(percent))),
        )
        if normalized < 100:
            available = cls.TRANSFORM_PERCENT_SLIDER_EXTENT
            lower_span = 100 - cls.TRANSFORM_PERCENT_MIN
            return round((normalized - 100) * available / lower_span)
        return normalized - 100

    @classmethod
    def _slider_position_to_percent(cls, position: int) -> int:
        """把滑块位置还原为百分比，左右端分别对应 5% 和 500%。"""
        normalized = max(
            -cls.TRANSFORM_PERCENT_SLIDER_EXTENT,
            min(cls.TRANSFORM_PERCENT_SLIDER_EXTENT, int(position)),
        )
        if normalized < 0:
            available = cls.TRANSFORM_PERCENT_SLIDER_EXTENT
            lower_span = 100 - cls.TRANSFORM_PERCENT_MIN
            return round(100 + normalized * lower_span / available)
        return 100 + normalized

    @classmethod
    def _distort_values(cls, value: object) -> list[float]:
        """把旧字库中的四角偏移规范为固定八项，异常数据按零处理。"""
        if not isinstance(value, (list, tuple)) or len(value) != 8:
            return [0.0] * 8
        return [cls._number(item, 0.0) for item in value]

    @staticmethod
    def _positive_int(value: object, default: int) -> int:
        try:
            result = int(value)
        except (TypeError, ValueError):
            return default
        return result if result > 0 else default

    @staticmethod
    def _file_md5(path: str) -> str:
        return _file_md5(path)
