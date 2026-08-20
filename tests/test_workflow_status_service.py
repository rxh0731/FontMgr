"""统一工作流状态解析服务回归测试。"""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

import config
from services.workflow_status_service import (
    COORDINATION_STATUS_FILTERS,
    INK_STATUS_ACHIEVED,
    INK_STATUS_DISABLED,
    INK_STATUS_EXCEPTION,
    INK_STATUS_NOT_APPLICABLE,
    INK_STATUS_PENDING,
    MARKER_FILE_ERROR,
    MARKER_FILTER_ALL,
    MARKER_FILTERS,
    MARKER_INK_EXCEPTION,
    MARKER_INK_PENDING,
    MARKER_STRUCTURE_REVIEW,
    MARKER_UNSAVED,
    OPTIMIZATION_STATUS_FILTERS,
    PHASE_COMPLETED_STATUSES,
    PHASE_COORDINATION,
    PHASE_FILTER_ALL,
    PHASE_OPTIMIZATION,
    PHASE_PENDING_STATUSES,
    PHASE_REVIEW,
    PHASE_STATUS_COLORS,
    PHASE_STATUS_FILTERS,
    REVIEW_STATUS_FILTERS,
    STAGE_COLORS,
    STAGE_COMPLETED,
    STAGE_FILTER_ALL,
    STAGE_FILTERS,
    STAGE_PENDING_COORDINATION,
    STAGE_PENDING_OPTIMIZATION,
    STAGE_PENDING_REVIEW,
    STATUS_COORDINATED,
    STATUS_OPTIMIZED,
    STATUS_REVIEWED,
    WORKFLOW_MARKERS,
    WORKFLOW_PHASES,
    WORKFLOW_STAGES,
    WorkflowStageProjection,
    WorkflowStatusService,
    project_stage_status,
    resolve_safe_stage_file,
    resolve_workflow_status,
)


class WorkflowStatusServiceTests(unittest.TestCase):
    """验证阶段、标记、成品引用和墨色契约保持正交。"""

    @staticmethod
    def _summary(*, enabled: object = True) -> dict[str, object]:
        return {
            "墨色统一启用": enabled,
            "墨色基准": 180.0,
            "墨色方法": "视觉墨量",
            "墨色方法版本": 2,
        }

    @staticmethod
    def _ink_record(**updates: object) -> dict[str, object]:
        record: dict[str, object] = {
            "启用": True,
            "基准": 180.0,
            "方法": "视觉墨量",
            "方法版本": 2,
            "保存后复测": True,
            "保存后墨色": 179.5,
            "是否达标": True,
            "人工接受例外": False,
        }
        record.update(updates)
        return record

    @classmethod
    def _finished_detail(
        cls,
        filename: str = "甲-0001.png",
        **record_updates: object,
    ) -> dict[str, object]:
        return {
            "状态": config.STATUS_FINISHED,
            "成品文件": filename,
            "整体协调参数": {
                "墨色协调": cls._ink_record(**record_updates),
            },
        }

    @staticmethod
    def _install_finished(directory: Path, filename: str = "甲-0001.png") -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / filename).write_bytes(b"png-placeholder")

    @staticmethod
    def _install_stage_file(
        library_dir: Path,
        stage_dir: str,
        filename: str,
    ) -> None:
        directory = library_dir / stage_dir
        directory.mkdir(parents=True, exist_ok=True)
        (directory / filename).write_bytes(b"image-placeholder")

    def test_public_stage_and_filter_constants_are_fixed(self) -> None:
        self.assertEqual(
            WORKFLOW_STAGES,
            ("待优化", "待审核", "待协调", "已完成"),
        )
        self.assertEqual(STAGE_FILTERS, (STAGE_FILTER_ALL, *WORKFLOW_STAGES))
        self.assertEqual(
            WORKFLOW_MARKERS,
            (
                "未保存修改",
                "结构需核对",
                "墨色待确认",
                "人工例外",
                "文件异常",
            ),
        )
        self.assertEqual(
            MARKER_FILTERS,
            ("全部提示", *WORKFLOW_MARKERS),
        )
        self.assertEqual(set(STAGE_COLORS), set(WORKFLOW_STAGES))

    def test_public_phase_projection_constants_are_fixed(self) -> None:
        self.assertEqual(
            WORKFLOW_PHASES,
            ("自动优化", "手工审核", "整体协调"),
        )
        self.assertEqual(
            OPTIMIZATION_STATUS_FILTERS,
            (PHASE_FILTER_ALL, "待优化", "已优化"),
        )
        self.assertEqual(
            REVIEW_STATUS_FILTERS,
            (PHASE_FILTER_ALL, "待审核", "已审核"),
        )
        self.assertEqual(
            COORDINATION_STATUS_FILTERS,
            (PHASE_FILTER_ALL, "待协调", "已协调"),
        )
        self.assertEqual(
            PHASE_STATUS_FILTERS,
            {
                PHASE_OPTIMIZATION: OPTIMIZATION_STATUS_FILTERS,
                PHASE_REVIEW: REVIEW_STATUS_FILTERS,
                PHASE_COORDINATION: COORDINATION_STATUS_FILTERS,
            },
        )
        self.assertEqual(
            PHASE_PENDING_STATUSES,
            {
                PHASE_OPTIMIZATION: STAGE_PENDING_OPTIMIZATION,
                PHASE_REVIEW: STAGE_PENDING_REVIEW,
                PHASE_COORDINATION: STAGE_PENDING_COORDINATION,
            },
        )
        self.assertEqual(
            PHASE_COMPLETED_STATUSES,
            {
                PHASE_OPTIMIZATION: STATUS_OPTIMIZED,
                PHASE_REVIEW: STATUS_REVIEWED,
                PHASE_COORDINATION: STATUS_COORDINATED,
            },
        )
        self.assertEqual(PHASE_STATUS_COLORS[STATUS_OPTIMIZED], "#228B22")
        self.assertEqual(PHASE_STATUS_COLORS[STATUS_REVIEWED], "#228B22")
        self.assertEqual(PHASE_STATUS_COLORS[STATUS_COORDINATED], "#228B22")

    def test_pending_primary_states_map_to_fixed_user_stages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            library_dir = Path(directory)
            source_dir = library_dir / config.DIR_ORIGINAL_FILES
            preview_dir = library_dir / config.DIR_INTERMEDIATE_FILES
            source_dir.mkdir()
            preview_dir.mkdir()
            (source_dir / "甲.tif").write_bytes(b"source")
            (preview_dir / "甲.png").write_bytes(b"preview")
            cases = (
                (
                    {"状态": config.STATUS_PENDING_OPTIMIZATION, "原始文件": "甲.tif"},
                    STAGE_PENDING_OPTIMIZATION,
                ),
                (
                    {"状态": config.STATUS_PENDING_MANUAL_REVIEW, "中间文件": "甲.png"},
                    STAGE_PENDING_REVIEW,
                ),
                (
                    {"状态": config.STATUS_REVIEWED, "中间文件": "甲.png"},
                    STAGE_PENDING_COORDINATION,
                ),
                (
                    {"状态": "待自动优化", "原始文件": "甲.tif"},
                    STAGE_PENDING_OPTIMIZATION,
                ),
                (
                    {"状态": "待手工审核", "中间文件": "甲.png"},
                    STAGE_PENDING_REVIEW,
                ),
                (
                    {"状态": "已审核", "中间文件": "甲.png"},
                    STAGE_PENDING_COORDINATION,
                ),
            )
            finished_dir = library_dir / config.DIR_FINISHED_FILES
            for detail, expected_stage in cases:
                with self.subTest(primary_status=detail["状态"]):
                    result = resolve_workflow_status(
                        detail,
                        self._summary(),
                        finished_dir,
                    )
                    self.assertEqual(result.stage, expected_stage)
                    self.assertEqual(result.markers, ())
                    self.assertEqual(result.ink_status, INK_STATUS_NOT_APPLICABLE)
                    self.assertFalse(result.has_valid_finished)

    def test_phase_projection_follows_strict_pipeline_layers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            library_dir = Path(directory)
            finished_dir = library_dir / config.DIR_FINISHED_FILES
            self._install_stage_file(
                library_dir,
                config.DIR_ORIGINAL_FILES,
                "甲.tif",
            )
            self._install_stage_file(
                library_dir,
                config.DIR_INTERMEDIATE_FILES,
                "甲.png",
            )
            self._install_stage_file(
                library_dir,
                config.DIR_REVIEWED_FILES,
                "甲-审核.png",
            )
            self._install_finished(finished_dir)

            details = (
                (
                    {
                        "状态": config.STATUS_PENDING_OPTIMIZATION,
                        "原始文件": "甲.tif",
                    },
                    ("待优化", "", ""),
                    (True, False, False),
                ),
                (
                    {
                        "状态": config.STATUS_PENDING_MANUAL_REVIEW,
                        "中间文件": "甲.png",
                    },
                    ("已优化", "待审核", ""),
                    (True, True, False),
                ),
                (
                    {
                        "状态": config.STATUS_REVIEWED,
                        "中间文件": "甲.png",
                        "审核文件": "甲-审核.png",
                    },
                    ("已优化", "已审核", "待协调"),
                    (True, True, True),
                ),
                (
                    {
                        **self._finished_detail(),
                        "中间文件": "甲.png",
                        "审核文件": "甲-审核.png",
                    },
                    ("已优化", "已审核", "已协调"),
                    (True, True, True),
                ),
            )
            for detail, expected_statuses, expected_admission in details:
                with self.subTest(primary_status=detail["状态"]):
                    projections = tuple(
                        project_stage_status(
                            detail,
                            self._summary(),
                            finished_dir,
                            phase,
                        )
                        for phase in WORKFLOW_PHASES
                    )
                    self.assertEqual(
                        tuple(item.status for item in projections),
                        expected_statuses,
                    )
                    self.assertEqual(
                        tuple(item.admitted for item in projections),
                        expected_admission,
                    )
                    self.assertEqual(
                        tuple(item.completed for item in projections),
                        tuple(
                            status in {
                                STATUS_OPTIMIZED,
                                STATUS_REVIEWED,
                                STATUS_COORDINATED,
                            }
                            for status in expected_statuses
                        ),
                    )

    def test_optimization_projection_accepts_valid_draft_or_later_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            library_dir = Path(directory)
            finished_dir = library_dir / config.DIR_FINISHED_FILES
            self._install_stage_file(
                library_dir,
                config.DIR_INTERMEDIATE_FILES,
                "甲.png",
            )
            self._install_stage_file(
                library_dir,
                config.DIR_REVIEWED_FILES,
                "乙-审核.png",
            )
            self._install_finished(finished_dir, "丙-0001.png")

            valid_draft = project_stage_status(
                {
                    "状态": config.STATUS_PENDING_MANUAL_REVIEW,
                    "中间文件": "甲.png",
                },
                self._summary(),
                finished_dir,
                PHASE_OPTIMIZATION,
            )
            reviewed = project_stage_status(
                {
                    "状态": config.STATUS_REVIEWED,
                    "审核文件": "乙-审核.png",
                },
                self._summary(),
                finished_dir,
                PHASE_OPTIMIZATION,
            )
            finished = project_stage_status(
                self._finished_detail("丙-0001.png"),
                self._summary(),
                finished_dir,
                PHASE_OPTIMIZATION,
            )

            for projection in (valid_draft, reviewed, finished):
                self.assertTrue(projection.admitted)
                self.assertTrue(projection.completed)
                self.assertEqual(projection.status, STATUS_OPTIMIZED)

    def test_uncommitted_automatic_draft_does_not_advance_projection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            library_dir = Path(directory)
            finished_dir = library_dir / config.DIR_FINISHED_FILES
            self._install_stage_file(
                library_dir,
                config.DIR_ORIGINAL_FILES,
                "甲.tif",
            )
            self._install_stage_file(
                library_dir,
                config.DIR_INTERMEDIATE_FILES,
                "甲.png",
            )
            detail = {
                "状态": config.STATUS_PENDING_OPTIMIZATION,
                "原始文件": "甲.tif",
                "中间文件": "甲.png",
            }

            optimization = project_stage_status(
                detail,
                self._summary(),
                finished_dir,
                PHASE_OPTIMIZATION,
            )
            review = project_stage_status(
                detail,
                self._summary(),
                finished_dir,
                PHASE_REVIEW,
            )

            self.assertTrue(optimization.admitted)
            self.assertFalse(optimization.completed)
            self.assertEqual(optimization.status, STAGE_PENDING_OPTIMIZATION)
            self.assertEqual(optimization.markers, ())
            self.assertFalse(review.admitted)
            self.assertEqual(review.status, "")

    def test_missing_automatic_draft_blocks_review_admission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            finished_dir = Path(directory) / config.DIR_FINISHED_FILES
            detail = {
                "状态": config.STATUS_PENDING_MANUAL_REVIEW,
                "中间文件": "不存在.png",
            }
            optimization = project_stage_status(
                detail,
                self._summary(),
                finished_dir,
                PHASE_OPTIMIZATION,
            )
            review = project_stage_status(
                detail,
                self._summary(),
                finished_dir,
                PHASE_REVIEW,
            )
            self.assertFalse(optimization.completed)
            self.assertEqual(optimization.status, STAGE_PENDING_OPTIMIZATION)
            self.assertEqual(optimization.markers, (MARKER_FILE_ERROR,))
            self.assertFalse(review.admitted)
            self.assertEqual(review.status, "")
            self.assertFalse(review.matches_status(PHASE_FILTER_ALL))

    def test_review_completion_uses_manual_draft_or_legal_automatic_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            library_dir = Path(directory)
            finished_dir = library_dir / config.DIR_FINISHED_FILES
            self._install_stage_file(
                library_dir,
                config.DIR_INTERMEDIATE_FILES,
                "甲.png",
            )
            self._install_stage_file(
                library_dir,
                config.DIR_REVIEWED_FILES,
                "甲-审核.png",
            )
            manual = project_stage_status(
                {
                    "状态": config.STATUS_REVIEWED,
                    "中间文件": "甲.png",
                    "审核文件": "甲-审核.png",
                },
                self._summary(),
                finished_dir,
                PHASE_REVIEW,
            )
            fallback = project_stage_status(
                {
                    "状态": config.STATUS_REVIEWED,
                    "中间文件": "甲.png",
                    "审核文件": "",
                },
                self._summary(),
                finished_dir,
                PHASE_REVIEW,
            )
            broken_manual = project_stage_status(
                {
                    "状态": config.STATUS_REVIEWED,
                    "中间文件": "甲.png",
                    "审核文件": "不存在.png",
                },
                self._summary(),
                finished_dir,
                PHASE_REVIEW,
            )

            for projection in (manual, fallback):
                self.assertTrue(projection.admitted)
                self.assertTrue(projection.completed)
                self.assertEqual(projection.status, STATUS_REVIEWED)
                self.assertNotIn(MARKER_FILE_ERROR, projection.markers)
            self.assertTrue(broken_manual.admitted)
            self.assertFalse(broken_manual.completed)
            self.assertEqual(broken_manual.status, STAGE_PENDING_REVIEW)
            self.assertEqual(broken_manual.markers, (MARKER_FILE_ERROR,))

    def test_pending_review_keeps_explicit_missing_manual_draft_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            library_dir = Path(directory)
            finished_dir = library_dir / config.DIR_FINISHED_FILES
            self._install_stage_file(
                library_dir,
                config.DIR_INTERMEDIATE_FILES,
                "甲.png",
            )
            projection = project_stage_status(
                {
                    "状态": config.STATUS_PENDING_MANUAL_REVIEW,
                    "中间文件": "甲.png",
                    "审核文件": "不存在.png",
                },
                self._summary(),
                finished_dir,
                PHASE_REVIEW,
            )

            self.assertTrue(projection.admitted)
            self.assertFalse(projection.completed)
            self.assertEqual(projection.status, STAGE_PENDING_REVIEW)
            self.assertEqual(projection.markers, (MARKER_FILE_ERROR,))

    def test_pending_review_manual_draft_can_prove_optimization_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            library_dir = Path(directory)
            finished_dir = library_dir / config.DIR_FINISHED_FILES
            self._install_stage_file(
                library_dir,
                config.DIR_REVIEWED_FILES,
                "甲-审核.png",
            )
            detail = {
                "状态": config.STATUS_PENDING_MANUAL_REVIEW,
                "中间文件": "不存在.png",
                "审核文件": "甲-审核.png",
            }
            optimization = project_stage_status(
                detail,
                self._summary(),
                finished_dir,
                PHASE_OPTIMIZATION,
            )
            review = project_stage_status(
                detail,
                self._summary(),
                finished_dir,
                PHASE_REVIEW,
            )

            self.assertTrue(optimization.completed)
            self.assertEqual(optimization.status, STATUS_OPTIMIZED)
            self.assertTrue(review.admitted)
            self.assertFalse(review.completed)
            self.assertEqual(review.status, STAGE_PENDING_REVIEW)
            self.assertEqual(review.markers, ())

    def test_future_file_damage_does_not_roll_back_previous_phase_projection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            library_dir = Path(directory)
            finished_dir = library_dir / config.DIR_FINISHED_FILES
            self._install_stage_file(
                library_dir,
                config.DIR_INTERMEDIATE_FILES,
                "甲.png",
            )
            self._install_stage_file(
                library_dir,
                config.DIR_REVIEWED_FILES,
                "甲-审核.png",
            )
            detail = {
                **self._finished_detail("不存在.png"),
                "中间文件": "甲.png",
                "审核文件": "甲-审核.png",
            }
            optimization = project_stage_status(
                detail,
                self._summary(),
                finished_dir,
                PHASE_OPTIMIZATION,
            )
            review = project_stage_status(
                detail,
                self._summary(),
                finished_dir,
                PHASE_REVIEW,
            )
            coordination = project_stage_status(
                detail,
                self._summary(),
                finished_dir,
                PHASE_COORDINATION,
            )

            self.assertEqual(optimization.status, STATUS_OPTIMIZED)
            self.assertEqual(optimization.markers, ())
            self.assertEqual(review.status, STATUS_REVIEWED)
            self.assertEqual(review.markers, ())
            self.assertEqual(coordination.status, STAGE_PENDING_COORDINATION)
            self.assertEqual(coordination.markers, (MARKER_FILE_ERROR,))

    def test_pending_stages_report_missing_or_unsafe_required_files(self) -> None:
        cases = (
            {"状态": config.STATUS_PENDING_OPTIMIZATION, "原始文件": "不存在.tif"},
            {"状态": config.STATUS_PENDING_MANUAL_REVIEW, "中间文件": "../外部.png"},
            {"状态": config.STATUS_REVIEWED, "审核文件": "不存在.png"},
        )
        with tempfile.TemporaryDirectory() as directory:
            finished_dir = Path(directory) / config.DIR_FINISHED_FILES
            for detail in cases:
                with self.subTest(detail=detail):
                    result = resolve_workflow_status(
                        detail,
                        self._summary(),
                        finished_dir,
                    )
                    self.assertEqual(result.markers, (MARKER_FILE_ERROR,))

    def test_legacy_stage_directories_are_recognized_before_migration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            library_dir = Path(directory)
            (library_dir / "原始文件").mkdir()
            (library_dir / "中间文件").mkdir()
            (library_dir / "成品文件").mkdir()
            (library_dir / "原始文件" / "甲.tif").write_bytes(b"source")
            (library_dir / "中间文件" / "甲.png").write_bytes(b"preview")
            (library_dir / "成品文件" / "甲-0001.png").write_bytes(b"finished")
            finished_dir = library_dir / config.DIR_FINISHED_FILES

            pending = resolve_workflow_status(
                {"状态": config.STATUS_PENDING_OPTIMIZATION, "原始文件": "甲.tif"},
                self._summary(),
                finished_dir,
            )
            reviewed = resolve_workflow_status(
                {"状态": config.STATUS_REVIEWED, "中间文件": "甲.png"},
                self._summary(),
                finished_dir,
            )
            completed = resolve_workflow_status(
                self._finished_detail(),
                self._summary(),
                finished_dir,
            )

            self.assertEqual(pending.markers, ())
            self.assertEqual(reviewed.markers, ())
            self.assertEqual(completed.stage, STAGE_COMPLETED)
            self.assertTrue(completed.has_valid_finished)

    def test_phase_projection_recognizes_legacy_statuses_and_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            library_dir = Path(directory)
            (library_dir / "中间文件").mkdir()
            (library_dir / "审核文件").mkdir()
            (library_dir / "中间文件" / "甲.png").write_bytes(b"preview")
            (library_dir / "审核文件" / "乙.png").write_bytes(b"reviewed")
            finished_dir = library_dir / config.DIR_FINISHED_FILES

            pending_review = project_stage_status(
                {"状态": "待手工审核", "中间文件": "甲.png"},
                self._summary(),
                finished_dir,
                PHASE_REVIEW,
            )
            reviewed = project_stage_status(
                {"状态": "已审核", "审核文件": "乙.png"},
                self._summary(),
                finished_dir,
                PHASE_COORDINATION,
            )

            self.assertTrue(pending_review.admitted)
            self.assertEqual(pending_review.status, STAGE_PENDING_REVIEW)
            self.assertTrue(reviewed.admitted)
            self.assertEqual(reviewed.status, STAGE_PENDING_COORDINATION)

    def test_invalid_reviewed_file_is_not_hidden_by_preview_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            library_dir = Path(directory)
            preview_dir = library_dir / config.DIR_INTERMEDIATE_FILES
            preview_dir.mkdir()
            (preview_dir / "甲.png").write_bytes(b"preview")
            result = resolve_workflow_status(
                {
                    "状态": config.STATUS_REVIEWED,
                    "审核文件": "../外部.png",
                    "中间文件": "甲.png",
                },
                self._summary(),
                library_dir / config.DIR_FINISHED_FILES,
            )
            self.assertEqual(result.stage, STAGE_PENDING_COORDINATION)
            self.assertEqual(result.markers, (MARKER_FILE_ERROR,))

    def test_unknown_or_missing_primary_status_is_explicit_error(self) -> None:
        for detail in ({}, {"状态": "未知阶段"}, None):
            with self.subTest(detail=detail):
                result = resolve_workflow_status(
                    detail,
                    self._summary(),
                    "不存在",
                )
                self.assertEqual(result.stage, STAGE_PENDING_OPTIMIZATION)
                self.assertEqual(result.markers, (MARKER_FILE_ERROR,))

    def test_safe_stage_file_resolver_rejects_escape_and_missing_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stage_dir = Path(directory) / "阶段"
            stage_dir.mkdir()
            expected = stage_dir / "甲.png"
            expected.write_bytes(b"image")
            self.assertEqual(
                resolve_safe_stage_file(stage_dir, "甲.png"),
                str(expected.resolve()),
            )
            for filename in (
                "",
                ".",
                "..",
                "../甲.png",
                "子目录/甲.png",
                "不存在.png",
                "甲.png::$DATA",
                "甲.png.",
                "甲.png ",
                "NUL.png",
            ):
                with self.subTest(filename=filename):
                    self.assertEqual(resolve_safe_stage_file(stage_dir, filename), "")

            with patch(
                "services.workflow_status_service.os.path.islink",
                side_effect=lambda path: Path(path) == stage_dir,
            ):
                self.assertEqual(resolve_safe_stage_file(stage_dir, expected.name), "")
            with patch(
                "services.workflow_status_service.os.path.islink",
                side_effect=lambda path: Path(path) == expected,
            ):
                self.assertEqual(resolve_safe_stage_file(stage_dir, expected.name), "")
            with patch(
                "services.workflow_status_service.os.path.isjunction",
                side_effect=lambda path: Path(path) == stage_dir,
                create=True,
            ):
                self.assertEqual(resolve_safe_stage_file(stage_dir, expected.name), "")

    def test_structure_warning_is_marker_only_while_waiting_for_review(self) -> None:
        structure = {
            "自动优化": {
                "方案": {
                    "结构复核": {"状态": "需人工核对"},
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            library_dir = Path(directory)
            preview_dir = library_dir / config.DIR_INTERMEDIATE_FILES
            preview_dir.mkdir()
            (preview_dir / "甲.png").write_bytes(b"preview")
            finished_dir = library_dir / config.DIR_FINISHED_FILES
            pending = resolve_workflow_status(
                {
                    "状态": config.STATUS_PENDING_MANUAL_REVIEW,
                    "中间文件": "甲.png",
                    **structure,
                },
                self._summary(),
                finished_dir,
            )
            reviewed = resolve_workflow_status(
                {
                    "状态": config.STATUS_REVIEWED,
                    "中间文件": "甲.png",
                    **structure,
                },
                self._summary(),
                finished_dir,
            )
            self.assertEqual(pending.stage, STAGE_PENDING_REVIEW)
            self.assertEqual(pending.markers, (MARKER_STRUCTURE_REVIEW,))
            self.assertEqual(reviewed.stage, STAGE_PENDING_COORDINATION)
            self.assertEqual(reviewed.markers, ())

    def test_finished_requires_safe_existing_file(self) -> None:
        invalid_names = ("", "..", "../外部.png", "..\\外部.png", "子目录/甲.png")
        with tempfile.TemporaryDirectory() as directory:
            finished_dir = Path(directory)
            for filename in invalid_names:
                with self.subTest(filename=filename):
                    result = resolve_workflow_status(
                        self._finished_detail(filename),
                        self._summary(),
                        finished_dir,
                    )
                    self.assertEqual(result.stage, STAGE_PENDING_COORDINATION)
                    self.assertEqual(result.markers, (MARKER_FILE_ERROR,))
                    self.assertEqual(result.ink_status, INK_STATUS_NOT_APPLICABLE)
                    self.assertFalse(result.has_valid_finished)

    def test_matching_enabled_ink_result_is_completed_without_problem_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            finished_dir = Path(directory)
            self._install_finished(finished_dir)
            result = resolve_workflow_status(
                self._finished_detail(),
                self._summary(),
                finished_dir,
            )
            self.assertEqual(result.stage, STAGE_COMPLETED)
            self.assertTrue(result.is_completed)
            self.assertEqual(result.markers, ())
            self.assertEqual(result.ink_status, INK_STATUS_ACHIEVED)
            self.assertTrue(result.has_valid_finished)

    def test_manual_exception_is_completed_but_remains_visible_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            finished_dir = Path(directory)
            self._install_finished(finished_dir)
            result = WorkflowStatusService.resolve(
                self._finished_detail(
                    是否达标=False,
                    人工接受例外=True,
                ),
                self._summary(),
                finished_dir,
            )
            self.assertEqual(result.stage, STAGE_COMPLETED)
            self.assertEqual(result.ink_status, INK_STATUS_EXCEPTION)
            self.assertEqual(result.markers, (MARKER_INK_EXCEPTION,))

    def test_coordination_projection_waits_for_ink_and_accepts_manual_exception(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            library_dir = Path(directory)
            finished_dir = library_dir / config.DIR_FINISHED_FILES
            self._install_stage_file(
                library_dir,
                config.DIR_REVIEWED_FILES,
                "甲-审核.png",
            )
            self._install_finished(finished_dir)
            base = {
                "审核文件": "甲-审核.png",
            }
            pending = project_stage_status(
                {
                    **self._finished_detail(是否达标=False),
                    **base,
                },
                self._summary(),
                finished_dir,
                PHASE_COORDINATION,
            )
            exception = WorkflowStatusService.project_stage(
                {
                    **self._finished_detail(
                        是否达标=False,
                        人工接受例外=True,
                    ),
                    **base,
                },
                self._summary(),
                finished_dir,
                PHASE_COORDINATION,
            )

            self.assertTrue(pending.admitted)
            self.assertFalse(pending.completed)
            self.assertEqual(pending.status, STAGE_PENDING_COORDINATION)
            self.assertEqual(pending.ink_status, INK_STATUS_PENDING)
            self.assertEqual(pending.markers, (MARKER_INK_PENDING,))
            self.assertTrue(exception.completed)
            self.assertEqual(exception.status, STATUS_COORDINATED)
            self.assertEqual(exception.ink_status, INK_STATUS_EXCEPTION)
            self.assertEqual(exception.markers, (MARKER_INK_EXCEPTION,))

    def test_projection_filters_ignore_auxiliary_markers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            library_dir = Path(directory)
            self._install_stage_file(
                library_dir,
                config.DIR_INTERMEDIATE_FILES,
                "甲.png",
            )
            projection = project_stage_status(
                {
                    "状态": config.STATUS_PENDING_MANUAL_REVIEW,
                    "中间文件": "甲.png",
                    "自动优化": {
                        "方案": {
                            "结构复核": {"状态": "需人工核对"},
                        },
                    },
                },
                self._summary(),
                library_dir / config.DIR_FINISHED_FILES,
                PHASE_REVIEW,
            )

            self.assertEqual(projection.markers, (MARKER_STRUCTURE_REVIEW,))
            self.assertTrue(projection.matches_status(PHASE_FILTER_ALL))
            self.assertTrue(projection.matches_status(STAGE_PENDING_REVIEW))
            self.assertFalse(projection.matches_status(STATUS_REVIEWED))

    def test_stage_projection_is_immutable_and_rejects_unknown_phase(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            library_dir = Path(directory)
            self._install_stage_file(
                library_dir,
                config.DIR_ORIGINAL_FILES,
                "甲.tif",
            )
            projection = project_stage_status(
                {
                    "状态": config.STATUS_PENDING_OPTIMIZATION,
                    "原始文件": "甲.tif",
                },
                self._summary(),
                library_dir / config.DIR_FINISHED_FILES,
                PHASE_OPTIMIZATION,
            )
            self.assertIsInstance(projection, WorkflowStageProjection)
            self.assertIs(projection.workflow, projection.workflow)
            self.assertFalse(projection.has_valid_finished)
            self.assertFalse(projection.has_marker(MARKER_FILE_ERROR))
            with self.assertRaises(FrozenInstanceError):
                projection.status = STATUS_OPTIMIZED  # type: ignore[misc]
            with self.assertRaisesRegex(ValueError, "未知制作阶段"):
                project_stage_status(
                    {},
                    self._summary(),
                    library_dir / config.DIR_FINISHED_FILES,
                    "未知阶段",
                )

    def test_enabled_ink_contract_mismatches_remain_pending(self) -> None:
        cases = (
            {"保存后复测": False},
            {"保存后墨色": None},
            {"方法": "旧方法"},
            {"方法版本": 1},
            {"基准": 181.0},
            {"是否达标": False},
        )
        with tempfile.TemporaryDirectory() as directory:
            finished_dir = Path(directory)
            self._install_finished(finished_dir)
            for updates in cases:
                with self.subTest(updates=updates):
                    result = resolve_workflow_status(
                        self._finished_detail(**updates),
                        self._summary(),
                        finished_dir,
                    )
                    self.assertEqual(result.stage, STAGE_PENDING_COORDINATION)
                    self.assertEqual(result.ink_status, INK_STATUS_PENDING)
                    self.assertEqual(result.markers, (MARKER_INK_PENDING,))
                    self.assertTrue(result.has_valid_finished)

    def test_invalid_summary_contract_cannot_complete_enabled_ink(self) -> None:
        summaries = (
            {},
            self._summary(enabled="是"),
            {**self._summary(), "墨色方法": ""},
            {**self._summary(), "墨色方法版本": 0},
            {**self._summary(), "墨色基准": float("nan")},
        )
        with tempfile.TemporaryDirectory() as directory:
            finished_dir = Path(directory)
            self._install_finished(finished_dir)
            for summary in summaries:
                with self.subTest(summary=summary):
                    result = resolve_workflow_status(
                        self._finished_detail(),
                        summary,
                        finished_dir,
                    )
                    self.assertEqual(result.stage, STAGE_PENDING_COORDINATION)
                    self.assertEqual(result.markers, (MARKER_INK_PENDING,))

    def test_disabled_ink_requires_disabled_record_and_saved_recheck(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            finished_dir = Path(directory)
            self._install_finished(finished_dir)
            completed = resolve_workflow_status(
                self._finished_detail(
                    启用=False,
                    保存后复测=True,
                    保存后墨色=176.0,
                ),
                self._summary(enabled=False),
                finished_dir,
            )
            wrong_mode = resolve_workflow_status(
                self._finished_detail(),
                self._summary(enabled=False),
                finished_dir,
            )
            missing_recheck = resolve_workflow_status(
                self._finished_detail(
                    启用=False,
                    保存后复测=False,
                ),
                self._summary(enabled=False),
                finished_dir,
            )
            self.assertEqual(completed.stage, STAGE_COMPLETED)
            self.assertEqual(completed.ink_status, INK_STATUS_DISABLED)
            self.assertEqual(completed.markers, ())
            for result in (wrong_mode, missing_recheck):
                self.assertEqual(result.stage, STAGE_PENDING_COORDINATION)
                self.assertEqual(result.ink_status, INK_STATUS_PENDING)
                self.assertEqual(result.markers, (MARKER_INK_PENDING,))

    def test_dirty_result_is_orthogonal_to_completed_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            finished_dir = Path(directory)
            self._install_finished(finished_dir)
            result = resolve_workflow_status(
                self._finished_detail(),
                self._summary(),
                finished_dir,
                dirty=True,
            )
            self.assertEqual(result.stage, STAGE_COMPLETED)
            self.assertEqual(result.markers, (MARKER_UNSAVED,))
            self.assertEqual(result.ink_status, INK_STATUS_ACHIEVED)
            self.assertTrue(result.has_valid_finished)

    def test_result_and_inputs_remain_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            library_dir = Path(directory)
            preview_dir = library_dir / config.DIR_INTERMEDIATE_FILES
            preview_dir.mkdir()
            (preview_dir / "甲.png").write_bytes(b"preview")
            detail = {"状态": config.STATUS_REVIEWED, "中间文件": "甲.png"}
            summary = self._summary()
            detail_before = dict(detail)
            summary_before = dict(summary)
            result = resolve_workflow_status(
                detail,
                summary,
                library_dir / config.DIR_FINISHED_FILES,
            )

            with self.assertRaises(FrozenInstanceError):
                result.stage = STAGE_COMPLETED  # type: ignore[misc]
            self.assertEqual(detail, detail_before)
            self.assertEqual(summary, summary_before)
            self.assertTrue(result.matches_stage(STAGE_PENDING_COORDINATION))
            self.assertTrue(result.matches_stage(STAGE_FILTER_ALL))
            self.assertTrue(result.matches_marker(MARKER_FILTER_ALL))
            self.assertFalse(result.matches_marker(MARKER_FILE_ERROR))


if __name__ == "__main__":
    unittest.main()
