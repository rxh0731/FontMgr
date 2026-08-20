"""定制经文排版三栏工作台。"""

from __future__ import annotations

import os
import threading
from collections import Counter
from collections.abc import Callable

from PIL.ImageQt import ImageQt
from PySide6.QtCore import QSignalBlocker, QSize, Qt, Slot
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import config
from core.custom_scripture_layout import (
    CustomBoardParameters,
    CustomGridGeometry,
    CustomLayoutResult,
    CustomLayoutTemplateParameters,
    ParsedCustomScripture,
    allocate_custom_boards,
    parse_custom_scripture,
)
from core.scripture_layout import LayoutParameters
from data.custom_layout_template_store import (
    DEFAULT_CUSTOM_TEMPLATE_ID,
    CustomLayoutTemplateStore,
)
from services.scripture_layout_service import (
    OUTPUT_FORMAT_AUTO,
    BoardOutputPlan,
    GenerationCancelled,
    GenerationProgress,
    GenerationResult,
    board_output_path,
    generate_psd_boards,
    plan_board_output,
    render_board_preview,
)
from ui.pages.scripture_layout_page import (
    CUSTOM_TEMPLATE_ID,
    ScriptureLayoutPage,
)
from ui.workers import FunctionWorker


class CustomScriptureLayoutPage(ScriptureLayoutPage):
    """空行分版、每行成列且逐列自动等高的定制排版页面。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        self._custom_board_parameters: list[CustomBoardParameters] = [
            CustomBoardParameters()
        ]
        self._editing_board_index = 0
        self._loading_custom_controls = False
        self._parsed_custom: ParsedCustomScripture | None = None
        self._custom_result: CustomLayoutResult | None = None
        super().__init__(parent)

    def _create_template_store(self) -> CustomLayoutTemplateStore:
        return CustomLayoutTemplateStore(config.CUSTOM_LAYOUT_TEMPLATE_FILE)

    def _default_template_id(self) -> str:
        return DEFAULT_CUSTOM_TEMPLATE_ID

    def _build_ui(self) -> None:
        super()._build_ui()
        for label in self.findChildren(QLabel):
            if label.property("role") == "pageTitle":
                label.setText("定制经文排版")
                break
        self._text_edit.setPlaceholderText(
            "在这里粘贴经文正文，或点击右上角打开文件。每个非空行作为一列，空行分隔版面。"
        )
        self._splitter.set_preferred_sizes(300, 560, 640)

    def _build_parameter_panel(self) -> QWidget:
        panel = self._card()
        panel.setMinimumWidth(540)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        heading_row = QHBoxLayout()
        heading = QLabel("版面设置")
        heading.setObjectName("sectionTitle")
        heading_row.addWidget(heading)
        heading_row.addStretch(1)
        self._match_label = QLabel("正文 0 / 参数 1")
        self._match_label.setProperty("role", "muted")
        heading_row.addWidget(self._match_label)
        self._save_template_button = QPushButton("模板保存")
        self._save_template_button.setObjectName("saveLayoutTemplateButton")
        self._save_template_button.setEnabled(False)
        self._save_template_button.clicked.connect(self._save_custom_template)
        heading_row.addWidget(self._save_template_button)
        layout.addLayout(heading_row)

        self._board_parameter_tabs = QTabBar()
        self._board_parameter_tabs.setObjectName("customBoardParameterTabs")
        self._board_parameter_tabs.setAccessibleName("版面与输出参数选择")
        self._board_parameter_tabs.setDrawBase(False)
        self._board_parameter_tabs.setExpanding(False)
        self._board_parameter_tabs.setUsesScrollButtons(True)
        self._board_parameter_tabs.setElideMode(Qt.TextElideMode.ElideRight)
        self._board_parameter_tabs.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self._board_parameter_tabs.currentChanged.connect(
            self._board_parameter_selection_changed
        )
        layout.addWidget(self._board_parameter_tabs)

        self._ignored_pages_label = QLabel()
        self._ignored_pages_label.setObjectName("customIgnoredPagesLabel")
        self._ignored_pages_label.setProperty("role", "muted")
        self._ignored_pages_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._ignored_pages_label.setWordWrap(True)
        layout.addWidget(self._ignored_pages_label)

        self._cell_height_spin = self._double_spin(0.1, 1000.0, " mm")
        self._cell_width_spin = self._double_spin(0.1, 1000.0, " mm")
        self._base_row_gap_spin = self._double_spin(0.0, 1000.0, " mm")
        self._base_count_spin = self._spin(1, 500)
        self._column_gap_spin = self._double_spin(0.0, 1000.0, " mm")
        self._dpi_spin = self._spin(72, 1200)
        self._frame_top_spin = self._double_spin(0.0, 1000.0, " mm")
        self._frame_bottom_spin = self._double_spin(0.0, 1000.0, " mm")
        self._frame_left_spin = self._double_spin(0.0, 1000.0, " mm")
        self._frame_right_spin = self._double_spin(0.0, 1000.0, " mm")
        self._canvas_top_spin = self._double_spin(0.0, 1000.0, " mm")
        self._canvas_bottom_spin = self._double_spin(0.0, 1000.0, " mm")
        self._canvas_left_spin = self._double_spin(0.0, 1000.0, " mm")
        self._canvas_right_spin = self._double_spin(0.0, 1000.0, " mm")

        self._settings_tabs = QTabWidget()
        self._settings_tabs.setObjectName("customLayoutSettingsTabs")
        self._settings_tabs.addTab(self._build_board_parameter_tab(), "版面参数")
        self._settings_tabs.addTab(self._build_custom_output_tab(), "输出参数")
        self._settings_tabs.tabBar().hide()
        layout.addWidget(self._settings_tabs, 1)

        self._custom_controls = (
            self._cell_height_spin,
            self._cell_width_spin,
            self._base_row_gap_spin,
            self._base_count_spin,
            self._column_gap_spin,
            self._dpi_spin,
            self._frame_top_spin,
            self._frame_bottom_spin,
            self._frame_left_spin,
            self._frame_right_spin,
            self._canvas_top_spin,
            self._canvas_bottom_spin,
            self._canvas_left_spin,
            self._canvas_right_spin,
        )
        for control in self._custom_controls:
            control.setMinimumWidth(90)
            control.valueChanged.connect(self._custom_parameter_changed)
        for control in (
            self._draw_outer_frame_check,
            self._include_punctuation_check,
            self._add_annotations_check,
        ):
            control.toggled.connect(self._custom_parameter_changed)
        self._refresh_board_parameter_tabs(0)
        self._load_board_controls(0)
        return panel

    def _build_board_parameter_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setObjectName("customBoardParameterScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(12, 10, 12, 12)
        content_layout.setSpacing(12)

        parameter_heading = QHBoxLayout()
        self._board_parameter_heading = QLabel("第 1 版 · 基础尺寸")
        self._board_parameter_heading.setObjectName("sectionTitle")
        parameter_heading.addWidget(self._board_parameter_heading)
        parameter_heading.addStretch(1)
        direction = QLabel("固定排版方式：竖排 / 从右向左")
        direction.setProperty("role", "muted")
        parameter_heading.addWidget(direction)
        self._remove_board_button = QPushButton("删除本版")
        self._remove_board_button.clicked.connect(self._remove_board_parameters)
        parameter_heading.addWidget(self._remove_board_button)
        content_layout.addLayout(parameter_heading)

        basic_grid = QGridLayout()
        basic_grid.setHorizontalSpacing(10)
        basic_grid.setVerticalSpacing(8)
        basic_grid.setColumnStretch(1, 1)
        basic_grid.setColumnStretch(3, 1)
        self._add_parameter_grid_field(
            basic_grid, 0, 0, "画布 DPI：", self._dpi_spin
        )
        self._add_parameter_grid_field(
            basic_grid, 0, 2, "固定列间距：", self._column_gap_spin
        )
        self._add_parameter_grid_field(
            basic_grid, 1, 0, "单元格高：", self._cell_height_spin
        )
        self._add_parameter_grid_field(
            basic_grid, 1, 2, "单元格宽：", self._cell_width_spin
        )
        self._add_parameter_grid_field(
            basic_grid, 2, 0, "基准列字数：", self._base_count_spin
        )
        self._add_parameter_grid_field(
            basic_grid, 2, 2, "基准行间距：", self._base_row_gap_spin
        )
        content_layout.addLayout(basic_grid)

        self._baseline_height_label = QLabel("基准列总高度：0.00 mm")
        self._baseline_height_label.setObjectName("customBaselineHeightLabel")
        self._baseline_height_label.setProperty("role", "muted")
        self._baseline_height_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        content_layout.addWidget(self._baseline_height_label)

        frame_heading = QLabel("框线与边距")
        frame_heading.setObjectName("sectionTitle")
        content_layout.addWidget(frame_heading)
        self._draw_outer_frame_check = QCheckBox("绘制大框")
        content_layout.addWidget(self._draw_outer_frame_check)
        margin_grid = QGridLayout()
        margin_grid.setHorizontalSpacing(8)
        margin_grid.setVerticalSpacing(8)
        for column, text in enumerate(("上", "下", "左", "右"), start=1):
            label = QLabel(text)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            margin_grid.addWidget(label, 0, column)
            margin_grid.setColumnStretch(column, 1)
        self._add_margin_grid_row(
            margin_grid,
            1,
            "大框边距：",
            (
                self._frame_top_spin,
                self._frame_bottom_spin,
                self._frame_left_spin,
                self._frame_right_spin,
            ),
        )
        self._add_margin_grid_row(
            margin_grid,
            2,
            "画布边距：",
            (
                self._canvas_top_spin,
                self._canvas_bottom_spin,
                self._canvas_left_spin,
                self._canvas_right_spin,
            ),
        )
        content_layout.addLayout(margin_grid)

        other_heading = QLabel("其他规则")
        other_heading.setObjectName("sectionTitle")
        content_layout.addWidget(other_heading)
        option_row = QHBoxLayout()
        self._include_punctuation_check = QCheckBox("包含标点符号")
        self._include_punctuation_check.setChecked(False)
        self._add_annotations_check = QCheckBox("尺寸标注")
        option_row.addWidget(self._include_punctuation_check)
        option_row.addWidget(self._add_annotations_check)
        option_row.addStretch(1)
        content_layout.addLayout(option_row)

        metric_heading = QLabel("逐列等高结果")
        metric_heading.setObjectName("sectionTitle")
        content_layout.addWidget(metric_heading)
        self._column_metric_table = QTableWidget(0, 4)
        self._column_metric_table.setObjectName("customColumnMetricTable")
        self._column_metric_table.setHorizontalHeaderLabels(
            ("列", "实际字数", "单元格高", "实际行间距")
        )
        self._column_metric_table.verticalHeader().hide()
        self._column_metric_table.verticalHeader().setDefaultSectionSize(30)
        self._column_metric_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self._column_metric_table.setSelectionMode(
            QTableWidget.SelectionMode.NoSelection
        )
        self._column_metric_table.setMinimumHeight(190)
        self._column_metric_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        content_layout.addWidget(self._column_metric_table, 1)
        scroll.setWidget(content)
        return scroll

    def _build_custom_output_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setObjectName("customOutputParameterScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(12, 12, 12, 12)
        heading = QLabel("输出文件")
        heading.setObjectName("sectionTitle")
        content_layout.addWidget(heading)
        output_form = self._build_output_form()
        output_form.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        content_layout.addWidget(output_form)
        content_layout.addStretch(1)
        scroll.setWidget(content)
        return scroll

    @staticmethod
    def _add_parameter_grid_field(
        grid: QGridLayout,
        row: int,
        column: int,
        text: str,
        control: QWidget,
    ) -> None:
        label = QLabel(text)
        label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        grid.addWidget(label, row, column)
        grid.addWidget(control, row, column + 1)

    @staticmethod
    def _add_margin_grid_row(
        grid: QGridLayout,
        row: int,
        text: str,
        controls: tuple[QWidget, QWidget, QWidget, QWidget],
    ) -> None:
        label = QLabel(text)
        label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        grid.addWidget(label, row, 0)
        for column, control in enumerate(controls, start=1):
            grid.addWidget(control, row, column)

    def _build_output_form(self) -> QWidget:
        content = super()._build_output_form()
        self._output_name_edit.setText("定制经文排版")
        return content

    def _template_choice_changed(self) -> None:
        if self._applying_template:
            return
        template_id = self._selected_template_id()
        if template_id == CUSTOM_TEMPLATE_ID:
            return
        try:
            template = self._template_store.get(template_id)
        except KeyError as exc:
            QMessageBox.warning(self, "模板载入失败", str(exc))
            return
        self._template_combo.setToolTip(template.description)
        self._apply_template(template_id)

    def _update_template_state(self) -> None:
        if self._applying_template or not hasattr(self, "_save_template_button"):
            return
        if self._template_baseline is None:
            return
        try:
            dirty = self._collect_parameters() != self._template_baseline
        except ValueError:
            dirty = True
        self._template_dirty = dirty
        custom_index = self._template_combo.findData(CUSTOM_TEMPLATE_ID)
        with QSignalBlocker(self._template_combo):
            if dirty:
                if custom_index < 0:
                    self._template_combo.insertItem(
                        0,
                        "自定义模板",
                        CUSTOM_TEMPLATE_ID,
                    )
                    custom_index = 0
                self._template_combo.setCurrentIndex(custom_index)
                self._template_combo.setToolTip("当前多版参数尚未保存为模板")
            else:
                if custom_index >= 0:
                    self._template_combo.removeItem(custom_index)
                applied_index = self._template_combo.findData(
                    self._applied_template_id
                )
                if applied_index >= 0:
                    self._template_combo.setCurrentIndex(applied_index)
                    try:
                        template = self._template_store.get(
                            self._applied_template_id
                        )
                        self._template_combo.setToolTip(template.description)
                    except KeyError:
                        self._template_combo.setToolTip("")
        self._save_template_button.setEnabled(not self.is_running and dirty)

    def _refresh_board_parameter_tabs(self, selected: int | None = None) -> None:
        if not hasattr(self, "_board_parameter_tabs"):
            return
        page_count = len(self._parsed_custom.pages) if self._parsed_custom else 0
        previous_parameter_count = len(self._custom_board_parameters)
        keep_output_page = (
            selected is None
            and self._board_parameter_tabs.currentIndex()
            == previous_parameter_count + 1
        )
        target = self._editing_board_index if selected is None else selected
        with QSignalBlocker(self._board_parameter_tabs):
            while self._board_parameter_tabs.count():
                self._board_parameter_tabs.removeTab(0)
            for index, _parameters in enumerate(self._custom_board_parameters):
                if index < page_count:
                    page = self._parsed_custom.pages[index]
                    counts = "/".join(str(len(column)) for column in page.columns)
                    detail = f"{len(page.columns)} 列 · {counts or '0'} 字"
                    state = "已匹配"
                else:
                    detail = "尚无对应正文"
                    state = "等待正文"
                tab_index = self._board_parameter_tabs.addTab(f"第 {index + 1} 版")
                self._board_parameter_tabs.setTabToolTip(
                    tab_index,
                    f"第 {index + 1} 版：{detail}，{state}",
                )
            add_tab = self._board_parameter_tabs.addTab("＋")
            self._board_parameter_tabs.setTabToolTip(add_tab, "新增版面参数")
            output_tab = self._board_parameter_tabs.addTab("输出参数")
            self._board_parameter_tabs.setTabToolTip(
                output_tab,
                "设置所有版面共用的输出目录、文件名和文件格式",
            )
            if keep_output_page:
                self._board_parameter_tabs.setCurrentIndex(output_tab)
            elif self._custom_board_parameters:
                self._board_parameter_tabs.setCurrentIndex(
                    min(max(0, target), len(self._custom_board_parameters) - 1)
                )
        if hasattr(self, "_settings_tabs"):
            self._settings_tabs.setCurrentIndex(1 if keep_output_page else 0)
        parameter_count = len(self._custom_board_parameters)
        if page_count > parameter_count:
            first_ignored = parameter_count + 1
            ignored_text = (
                f"第 {first_ignored} 版无参数，将忽略"
                if first_ignored == page_count
                else f"第 {first_ignored}–{page_count} 版无参数，将忽略"
            )
            self._ignored_pages_label.setText(ignored_text)
            self._ignored_pages_label.setToolTip(
                f"正文多出 {page_count - parameter_count} 版；生成时不会输出这些正文。"
            )
        elif parameter_count > page_count:
            waiting = parameter_count - page_count
            self._ignored_pages_label.setText(f"{waiting} 块版面参数尚无对应正文")
            self._ignored_pages_label.setToolTip(
                "没有对应正文的版面参数会保留，但不会生成文件。"
            )
        else:
            self._ignored_pages_label.setText("全部正文均已匹配")
            self._ignored_pages_label.setToolTip("正文版数与参数版数一致。")
        self._remove_board_button.setEnabled(len(self._custom_board_parameters) > 1)
        self._update_match_status()

    def _update_match_status(self) -> None:
        page_count = len(self._parsed_custom.pages) if self._parsed_custom else 0
        parameter_count = len(self._custom_board_parameters)
        self._match_label.setText(f"正文 {page_count} / 参数 {parameter_count}")
        if page_count > parameter_count:
            self._match_label.setToolTip(
                f"正文多出 {page_count - parameter_count} 版；多出的正文将忽略，不生成文件。"
            )
        elif parameter_count > page_count:
            self._match_label.setToolTip(
                f"参数多出 {parameter_count - page_count} 块；没有对应正文的参数不会生成文件。"
            )
        else:
            self._match_label.setToolTip("正文版数与参数版数一致。")

    def _board_parameter_selection_changed(self, row: int) -> None:
        if self._loading_custom_controls or row < 0:
            return
        parameter_count = len(self._custom_board_parameters)
        if row == parameter_count:
            previous = self._editing_board_index
            with QSignalBlocker(self._board_parameter_tabs):
                self._board_parameter_tabs.setCurrentIndex(previous)
            self._add_board_parameters()
            return
        if row == parameter_count + 1:
            self._store_current_board_controls()
            self._settings_tabs.setCurrentIndex(1)
            return
        self._store_current_board_controls()
        if row >= parameter_count:
            with QSignalBlocker(self._board_parameter_tabs):
                self._board_parameter_tabs.setCurrentIndex(
                    self._editing_board_index
                )
            return
        self._settings_tabs.setCurrentIndex(0)
        self._editing_board_index = row
        self._load_board_controls(row)
        self._current_board = min(row, max(0, len(self._boards) - 1))
        self._update_board_controls()
        if self._boards and row < len(self._boards):
            self._render_current_preview()

    def _parameter_from_controls(self) -> CustomBoardParameters:
        result = CustomBoardParameters(
            dpi=self._dpi_spin.value(),
            cell_width_mm=self._cell_width_spin.value(),
            cell_height_mm=self._cell_height_spin.value(),
            base_row_gap_mm=self._base_row_gap_spin.value(),
            base_column_characters=self._base_count_spin.value(),
            column_gap_mm=self._column_gap_spin.value(),
            draw_outer_frame=self._draw_outer_frame_check.isChecked(),
            frame_top_mm=self._frame_top_spin.value(),
            frame_bottom_mm=self._frame_bottom_spin.value(),
            frame_left_mm=self._frame_left_spin.value(),
            frame_right_mm=self._frame_right_spin.value(),
            canvas_top_mm=self._canvas_top_spin.value(),
            canvas_bottom_mm=self._canvas_bottom_spin.value(),
            canvas_left_mm=self._canvas_left_spin.value(),
            canvas_right_mm=self._canvas_right_spin.value(),
        )
        result.validate()
        return result

    def _store_current_board_controls(self) -> None:
        if self._loading_custom_controls or not self._custom_board_parameters:
            return
        index = min(self._editing_board_index, len(self._custom_board_parameters) - 1)
        try:
            self._custom_board_parameters[index] = self._parameter_from_controls()
        except ValueError:
            return

    def _load_board_controls(self, index: int) -> None:
        if not self._custom_board_parameters:
            return
        parameters = self._custom_board_parameters[index]
        self._loading_custom_controls = True
        try:
            values = (
                (self._dpi_spin, parameters.dpi),
                (self._cell_width_spin, parameters.cell_width_mm),
                (self._cell_height_spin, parameters.cell_height_mm),
                (self._base_row_gap_spin, parameters.base_row_gap_mm),
                (self._base_count_spin, parameters.base_column_characters),
                (self._column_gap_spin, parameters.column_gap_mm),
                (self._frame_top_spin, parameters.frame_top_mm),
                (self._frame_bottom_spin, parameters.frame_bottom_mm),
                (self._frame_left_spin, parameters.frame_left_mm),
                (self._frame_right_spin, parameters.frame_right_mm),
                (self._canvas_top_spin, parameters.canvas_top_mm),
                (self._canvas_bottom_spin, parameters.canvas_bottom_mm),
                (self._canvas_left_spin, parameters.canvas_left_mm),
                (self._canvas_right_spin, parameters.canvas_right_mm),
            )
            for control, value in values:
                with QSignalBlocker(control):
                    control.setValue(value)
            with QSignalBlocker(self._draw_outer_frame_check):
                self._draw_outer_frame_check.setChecked(parameters.draw_outer_frame)
        finally:
            self._loading_custom_controls = False
        self._board_parameter_heading.setText(f"第 {index + 1} 版参数")
        self._baseline_height_label.setText(
            f"基准列总高度：{parameters.baseline_height_mm:.2f} mm"
        )
        self._refresh_column_metrics()

    def _custom_parameter_changed(self, _value: object = None) -> None:
        if self._loading_custom_controls:
            return
        self._store_current_board_controls()
        self._load_board_controls(self._editing_board_index)
        self._schedule_preview()

    def _add_board_parameters(self) -> None:
        self._store_current_board_controls()
        choices = ["使用默认参数"] + [
            f"复制第 {index + 1} 版参数"
            for index in range(len(self._custom_board_parameters))
        ]
        choice, accepted = QInputDialog.getItem(
            self,
            "新增版面参数",
            "选择新参数的初始值：",
            choices,
            0,
            False,
        )
        if not accepted:
            return
        if choice == "使用默认参数":
            parameters = CustomBoardParameters()
        else:
            source_index = choices.index(choice) - 1
            parameters = self._custom_board_parameters[source_index]
        self._custom_board_parameters.append(parameters)
        self._editing_board_index = len(self._custom_board_parameters) - 1
        self._refresh_board_parameter_tabs()
        self._settings_tabs.setCurrentIndex(0)
        self._load_board_controls(self._editing_board_index)
        self._schedule_preview()

    def _remove_board_parameters(self) -> None:
        if len(self._custom_board_parameters) <= 1:
            QMessageBox.information(self, "无法删除", "至少需要保留一块版面参数。")
            return
        index = self._editing_board_index
        reply = QMessageBox.question(
            self,
            "删除版面参数",
            f"确定删除第 {index + 1} 版参数吗？正文内容不会被删除。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        del self._custom_board_parameters[index]
        self._editing_board_index = min(index, len(self._custom_board_parameters) - 1)
        self._refresh_board_parameter_tabs(self._editing_board_index)
        self._settings_tabs.setCurrentIndex(0)
        self._load_board_controls(self._editing_board_index)
        self._schedule_preview()

    def _collect_parameters(self) -> CustomLayoutTemplateParameters:
        self._store_current_board_controls()
        result = CustomLayoutTemplateParameters(
            boards=tuple(self._custom_board_parameters),
            include_punctuation=self._include_punctuation_check.isChecked(),
            add_annotations=self._add_annotations_check.isChecked(),
        )
        result.validate()
        return result

    def _apply_parameters(self, parameters: CustomLayoutTemplateParameters) -> None:
        parameters.validate()
        self._custom_board_parameters = list(parameters.boards)
        self._editing_board_index = 0
        self._loading_custom_controls = True
        try:
            with QSignalBlocker(self._include_punctuation_check):
                self._include_punctuation_check.setChecked(
                    parameters.include_punctuation
                )
            with QSignalBlocker(self._add_annotations_check):
                self._add_annotations_check.setChecked(parameters.add_annotations)
        finally:
            self._loading_custom_controls = False
        self._refresh_board_parameter_tabs(0)
        self._load_board_controls(0)

    def _rebuild_layout(self, preparsed: object | None = None) -> None:
        del preparsed
        text = self._text_edit.toPlainText()
        include_punctuation = self._include_punctuation_check.isChecked()
        available = self._glyph_index.characters if self._glyph_index is not None else None
        parsed = parse_custom_scripture(text, available, include_punctuation)
        self._parsed_custom = parsed
        self._update_scripture_statistics(parsed, source_loaded=self._glyph_index is not None)
        self._update_missing_results(parsed if self._glyph_index is not None else None)
        self._refresh_board_parameter_tabs()
        if self._glyph_index is None:
            self._boards = ()
            self._custom_result = None
            self._show_preview_message("请先载入并检查字图来源")
            self._update_board_controls()
            return
        try:
            template_parameters = self._collect_parameters()
            result = allocate_custom_boards(parsed, template_parameters)
        except ValueError as exc:
            self._boards = ()
            self._custom_result = None
            self._show_preview_message(str(exc))
            self._update_board_controls()
            return
        self._custom_result = result
        self._boards = result.boards
        if not self._boards:
            self._current_board = 0
            self._show_preview_message("请输入经文正文")
            self._update_board_controls()
            return
        self._current_board = min(self._current_board, len(self._boards) - 1)
        self._update_board_controls()
        self._refresh_column_metrics()
        self._render_current_preview()

    def _refresh_column_metrics(self) -> None:
        if not hasattr(self, "_column_metric_table"):
            return
        geometry: CustomGridGeometry | None = None
        if (
            self._custom_result is not None
            and self._editing_board_index < len(self._custom_result.geometries)
        ):
            geometry = self._custom_result.geometries[self._editing_board_index]
        self._column_metric_table.setRowCount(0 if geometry is None else len(geometry.column_metrics))
        if geometry is None:
            return
        for row, metric in enumerate(reversed(geometry.column_metrics)):
            values = (
                str(row + 1),
                str(metric.character_count),
                f"{metric.cell_height_mm:.2f}",
                f"{metric.row_gap_mm:.2f}",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._column_metric_table.setItem(row, column, item)

    def _render_current_preview(
        self,
        parameters: LayoutParameters | None = None,
    ) -> None:
        del parameters
        if (
            not self._boards
            or self._glyph_index is None
            or self._custom_result is None
        ):
            return
        index = self._current_board
        board = self._boards[index]
        captured_parameters = self._custom_result.parameters[index]
        geometry = self._custom_result.geometries[index]
        glyph_index = self._glyph_index
        show_guides = self._preview_guides_check.isChecked()
        document_size = QSize(geometry.canvas_width, geometry.canvas_height)
        self._preview_document_size = document_size
        display_size = self._preview_target_size(document_size)
        device_ratio = max(1.0, float(self.devicePixelRatioF()))
        requested_size = QSize(
            max(1, round(display_size.width() * device_ratio)),
            max(1, round(display_size.height() * device_ratio)),
        )
        render_size = self._bounded_preview_render_size(requested_size, document_size)
        render_context = (
            board,
            captured_parameters,
            geometry,
            show_guides,
            id(glyph_index),
        )
        if self._reuse_current_preview(render_context, render_size):
            return
        self._preview_generation += 1
        generation = self._preview_generation
        if self._preview_cancel is not None:
            self._preview_cancel.set()
        cancel_event = threading.Event()
        self._preview_cancel = cancel_event
        if self._preview_image.isNull():
            self._preview_label.set_message("正在生成高清字图预览…")
        self._preview_progress.show()
        self._preview_progress.setRange(0, 0)
        self._preview_progress.setFormat("正在准备高清预览…")

        def render(progress_callback: Callable[[object], None]) -> QImage:
            try:
                image = render_board_preview(
                    board,
                    glyph_index,
                    captured_parameters,
                    (render_size.width(), render_size.height()),
                    geometry=geometry,
                    show_guides=show_guides,
                    progress_callback=progress_callback,
                    cancel_check=cancel_event.is_set,
                )
                try:
                    return QImage(ImageQt(image)).copy()
                finally:
                    image.close()
            except GenerationCancelled:
                return QImage()

        worker = FunctionWorker(render, with_progress=True)
        self._preview_worker = worker
        self._workers.add(worker)
        worker.signals.progress.connect(
            lambda value, token=generation, task=worker: self._preview_progress_changed(
                token, value, task
            )
        )
        worker.signals.finished.connect(
            lambda result,
            token=generation,
            task=worker,
            ratio=device_ratio,
            context=render_context: self._preview_ready(
                token,
                result,
                task,
                ratio,
                context,
            )
        )
        worker.signals.failed.connect(
            lambda message, token=generation, task=worker: self._preview_failed(
                token, message, task
            )
        )
        self._thread_pool.start(worker)

    def _change_board(self, offset: int) -> None:
        """切换预览时同步右侧版面参数，避免两个当前版号不一致。"""

        if not self._boards:
            return
        new_index = max(
            0,
            min(len(self._boards) - 1, self._current_board + offset),
        )
        if new_index == self._current_board:
            return
        self._board_parameter_tabs.setCurrentIndex(new_index)

    def _confirm_page_mismatch(
        self,
        page_count: int,
        parameter_count: int,
    ) -> None:
        if page_count == parameter_count:
            return
        if page_count > parameter_count:
            message = (
                f"正文识别为 {page_count} 版，当前只有 {parameter_count} 块版面参数。\n\n"
                f"第 {parameter_count + 1} 至第 {page_count} 版正文将忽略，不生成文件。"
            )
        else:
            message = (
                f"正文识别为 {page_count} 版，当前有 {parameter_count} 块版面参数。\n\n"
                "没有对应正文的多余版面参数不会生成文件。"
            )
        QMessageBox.information(self, "正文与版面参数数量不一致", message)

    def _start_generation(self) -> None:
        if self.is_running:
            return
        if self._glyph_index is None:
            QMessageBox.warning(self, "无法生成", "请先载入并检查字图来源。")
            return
        try:
            template_parameters = self._collect_parameters()
            parsed = parse_custom_scripture(
                self._text_edit.toPlainText(),
                self._glyph_index.characters,
                template_parameters.include_punctuation,
            )
            result = allocate_custom_boards(parsed, template_parameters)
            if not result.boards:
                raise ValueError("请输入要生成的经文正文。")
            self._confirm_page_mismatch(len(parsed.pages), len(template_parameters.boards))
            selected = self._parse_board_selection(len(result.boards))
            output_dir = self._output_path_edit.text().strip()
            if not output_dir:
                raise ValueError("请选择 PSD 输出目录。")
            output_base_name = self._output_name_edit.text().strip()
            output_format = self._output_format_combo.currentText() or OUTPUT_FORMAT_AUTO
            compress_psd = self._compress_psd_check.isChecked()
            first_parameters = result.parameters[0]
            board_output_path(
                output_dir,
                1,
                first_parameters.dpi,
                output_base_name,
                total_boards=len(result.boards),
            )
            os.makedirs(output_dir, exist_ok=True)
            selected_missing: Counter[str] = Counter(
                placement.character
                for board in result.boards
                if board.number in selected
                for placement in board.placements
                if placement.missing
            )
            if not self._confirm_missing_characters(selected_missing):
                return
            parameter_map = {
                board.number: result.parameters[index]
                for index, board in enumerate(result.boards)
            }
            geometry_map = {
                board.number: result.geometries[index]
                for index, board in enumerate(result.boards)
            }
            plans: dict[int, BoardOutputPlan] = {
                board.number: plan_board_output(
                    board,
                    self._glyph_index,
                    parameter_map[board.number],
                    output_format,
                    geometry=geometry_map[board.number],
                )
                for board in result.boards
                if board.number in selected
            }
            decisions = self._resolve_conflicts(
                output_dir,
                selected,
                len(result.boards),
                first_parameters.dpi,
                output_base_name,
                plans,
            )
            if decisions is None:
                return
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "无法生成", str(exc))
            return

        boards = result.boards
        glyph_index = self._glyph_index
        cancel_event = threading.Event()
        self._generation_cancel = cancel_event

        def generate(progress_callback: Callable[[object], None]) -> GenerationResult:
            try:
                return generate_psd_boards(
                    boards,
                    glyph_index,
                    first_parameters,
                    output_dir,
                    board_parameters=parameter_map,
                    board_geometries=geometry_map,
                    task_name="定制经文排版",
                    output_base_name=output_base_name,
                    compress_psd=compress_psd,
                    output_format=output_format,
                    selected_boards=selected,
                    conflict_decisions=decisions,
                    progress_callback=progress_callback,
                    cancel_event=cancel_event,
                )
            except GenerationCancelled:
                return GenerationResult((), True)

        worker = FunctionWorker(generate, with_progress=True)
        self._generation_worker = worker
        self._workers.add(worker)
        worker.signals.progress.connect(self._generation_progress)
        worker.signals.finished.connect(
            lambda value, task=worker: self._generation_finished(value, task)
        )
        worker.signals.failed.connect(
            lambda message, task=worker: self._generation_failed(message, task)
        )
        self._set_running(True)
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setFormat("正在准备定制版面分层 PSD…")
        self._thread_pool.start(worker)

    def _generation_finished(self, result: object, worker: FunctionWorker) -> None:
        self._finish_worker(worker)
        if not isinstance(result, GenerationResult):
            QMessageBox.warning(self, "生成失败", "排版任务返回了无法识别的结果。")
            return
        if result.stopped:
            self._progress_bar.setRange(0, 1)
            self._progress_bar.setValue(0)
            self._progress_bar.setFormat("任务已停止；已完整生成的版面予以保留")
            self.status_message.emit("定制经文排版已停止")
            return
        completed = sum(not board.skipped for board in result.boards)
        skipped = sum(board.skipped for board in result.boards)
        missing = sum(board.missing_characters for board in result.boards)
        self._progress_bar.setValue(self._progress_bar.maximum())
        summary = f"生成完成：新增或覆盖 {completed} 版，跳过 {skipped} 版"
        if missing:
            summary += f"，缺字留空 {missing} 处"
        self._progress_bar.setFormat(summary)
        QMessageBox.information(
            self,
            "定制经文排版完成",
            f"已生成 {completed} 个分层文件，跳过 {skipped} 个已有文件。",
        )

    def _generation_failed(self, message: str, worker: FunctionWorker) -> None:
        self._finish_worker(worker)
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("生成失败")
        QMessageBox.critical(self, "定制经文排版失败", message)

    def _set_running(self, running: bool) -> None:
        super()._set_running(running)
        if hasattr(self, "_board_parameter_tabs"):
            self._board_parameter_tabs.setEnabled(not running)
            self._remove_board_button.setEnabled(
                not running and len(self._custom_board_parameters) > 1
            )
            for control in self._custom_controls:
                control.setEnabled(not running)
            self._draw_outer_frame_check.setEnabled(not running)
            self._include_punctuation_check.setEnabled(not running)
            self._add_annotations_check.setEnabled(not running)

    def _request_home(self) -> None:
        if self.is_running:
            QMessageBox.information(
                self,
                "排版任务正在执行",
                "请先点击“停止生成”，任务停止后即可返回首页。",
            )
            return
        self.home_requested.emit()
