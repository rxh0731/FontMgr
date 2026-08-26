"""应用内使用说明页面。"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True, slots=True)
class HelpSection:
    """一个说明主题中的内容分节。"""

    title: str
    paragraphs: tuple[str, ...] = ()
    items: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class HelpTopic:
    """可导航、可检索的说明主题。"""

    key: str
    title: str
    summary: str
    keywords: tuple[str, ...]
    sections: tuple[HelpSection, ...]

    @property
    def search_text(self) -> str:
        parts = [self.title, self.summary, *self.keywords]
        for section in self.sections:
            parts.extend((section.title, *section.paragraphs, *section.items))
        return " ".join(parts).casefold()


HELP_TOPICS: tuple[HelpTopic, ...] = (
    HelpTopic(
        "quick_start",
        "快速开始",
        "从创建字库到获得成品，先掌握主流程和首页入口。",
        ("首页", "开始", "流程", "字库"),
        (
            HelpSection(
                "第一次制作字库",
                items=(
                    "在首页点击“新建字库”，选择已经按文字命名的图片目录，并确认识别结果和字库参数。",
                    "按“字库添加 → 自动优化 → 手工审核 → 整体协调”的顺序推进字形。",
                    "整体协调完成后进入“导出最终成品”，选择输出规则并执行导出。",
                ),
            ),
            HelpSection(
                "继续已有工作",
                paragraphs=(
                    "在首页字库表中选中目标字库。制作流程会显示每一阶段的已完成数量；直接点击需要继续的阶段即可。",
                    "“可导出”是独立审计结果，不等同于已经完成整体协调。导出前仍会执行更完整的文件和状态核对。",
                ),
            ),
            HelpSection(
                "其他独立功能",
                items=(
                    "通用经文排版：按统一版式生成经文版面。",
                    "定制经文排版：按空行分版，并逐列设置版面参数。",
                    "文字统计：统计经文用字，并与本系统字库或外部字图目录比较。",
                    "图片实验室：面向整幅碑文拓片、书页和文字扫描件的独立处理功能。",
                ),
            ),
        ),
    ),
    HelpTopic(
        "library_workflow",
        "字库制作流程",
        "理解六阶段图片、四种持久状态和各工作台的准入关系。",
        ("阶段", "状态", "源图", "成品", "数据库"),
        (
            HelpSection(
                "六阶段图片",
                items=(
                    "源图：保留导入图片的原始副本。",
                    "灰度底稿和字形掩码：供图像分析与自动优化使用。",
                    "自动优化稿：保存自动算法选定的结果。",
                    "人工修订稿：保存手工审核后的结果。",
                    "最终成品：保存整体协调后的实际输出图片。",
                ),
            ),
            HelpSection(
                "状态如何推进",
                paragraphs=(
                    "内部状态依次为“待优化、待审核、审核通过、成品已生成”。页面会按当前阶段显示为“待优化/已优化”“待审核/已审核”“待协调/已协调”。",
                    "状态和对应阶段文件必须同时有效。上游重新处理后，如果下游结果已经失效，程序会撤销相应的完成状态，避免把旧图片误当成新成品。",
                ),
            ),
            HelpSection(
                "列表中的提示",
                paragraphs=(
                    "“未保存修改、结构需核对、墨色待确认、人工例外、文件异常”是辅助提示，不是新的制作阶段。遇到文件异常时，应先核对对应阶段图片和数据库，再继续批量处理。",
                ),
            ),
        ),
    ),
    HelpTopic(
        "glyph_navigation",
        "字形列表与搜索",
        "在当前制作阶段内筛选、搜索、定位和修正具体字形。",
        ("搜索", "筛选", "定位", "修正字形名称", "文件名", "无结果"),
        (
            HelpSection(
                "状态筛选范围",
                paragraphs=(
                    "自动优化、手工审核和整体协调的状态筛选只作用于当前制作阶段。选择“已协调”后，搜索范围就是当前已经进入整体协调阶段且投影为已协调的字形，不会跨到待协调或尚未完成手工审核的记录。",
                ),
            ),
            HelpSection(
                "文字搜索范围",
                items=(
                    "输入归属字、字形编号或文件名，按回车或点击“搜索”执行；输入过程不会反复重建列表。",
                    "搜索不会匹配“已审核”等状态文字、阶段名称、辅助提示或风险说明。",
                    "每次非空搜索都从当前排序的第一条匹配结果开始定位；没有匹配项时列表明确显示为空。",
                    "清空搜索文字会立即恢复当前阶段和状态筛选下的全部字形，并尽量保留仍然可见的当前选择。",
                ),
            ),
            HelpSection(
                "列表与工作区同步",
                paragraphs=(
                    "在字形列表中选中一项时，中间工作区会同步定位；整体协调的比较区选中字形时，左侧列表也会滚动到相同记录。父级字组只用于汇总，不能替代具体字形成为编辑对象。",
                ),
            ),
            HelpSection(
                "修正字形名称",
                paragraphs=(
                    "在具体字形上打开右键菜单并选择“修正字形名称”。系统只接受一个有效汉字，并会使用稳定变体 ID 同步六阶段已有文件名和数据库记录；图片内容、优化、审核及协调结果保持不变。",
                ),
            ),
        ),
    ),
    HelpTopic(
        "create_and_add",
        "新建字库与字库添加",
        "导入图片、复核文字识别结果，并安全创建或扩充字库。",
        ("导入", "添加", "扫描", "繁体", "OpenCC", "重复"),
        (
            HelpSection(
                "准备图片",
                paragraphs=(
                    "使用清晰、方向正确的单字图片。文件名中的文字用于初步识别；同一个字可以导入多个字形，系统会分配稳定的四位序号。",
                ),
            ),
            HelpSection(
                "识别与人工复核",
                items=(
                    "选择图片目录后先扫描，不会立刻写入字库。",
                    "在复核列表中检查原字以及简体转繁体的 s2t、s2tw、s2hk 候选，选择符合碑帖或古文语境的文字。",
                    "重复图片会在正式导入时跳过；无法确定的文字应先人工确认，不要只依赖自动转换。",
                ),
            ),
            HelpSection(
                "创建与添加的区别",
                paragraphs=(
                    "“新建字库”会建立新的字库目录、参数和 SQLite 数据库；“字库添加”沿用当前字库参数，将新图片加入已有流程。两种操作都会在结束反馈中显示耗时。",
                ),
            ),
        ),
    ),
    HelpTopic(
        "optimization",
        "自动优化",
        "生成候选图、比较评分并保存自动优化稿。",
        ("候选", "算法", "评分", "结构", "批量自动优化"),
        (
            HelpSection(
                "单字处理",
                items=(
                    "从左侧列表选择待优化字形，中间查看原图和候选效果。",
                    "需要进一步尝试时，可围绕当前方案继续探索，或从原图重新生成候选。",
                    "确认后点击“采用并保存”，结果进入自动优化稿，并允许进入手工审核。",
                ),
            ),
            HelpSection(
                "结构保护提示",
                paragraphs=(
                    "评分用于比较候选，不代表文字一定正确。出现“结构需核对”时，应在手工审核中重点检查缺笔、粘连、污点和误删。",
                ),
            ),
            HelpSection(
                "批量自动优化",
                paragraphs=(
                    "“批量自动优化”处理全部待优化字形。点击停止后，系统会让当前单字事务完整提交或回滚；已经完成的字形保留，未开始的字形保持待优化。",
                ),
            ),
        ),
    ),
    HelpTopic(
        "manual_review",
        "手工审核",
        "校正自动优化稿，使用变换、画笔和橡皮完成字形审核。",
        ("画笔", "橡皮", "自由变换", "透视", "审核", "压感"),
        (
            HelpSection(
                "常用操作",
                items=(
                    "自由变换可移动、等比缩放、单轴拉伸、旋转、透视扭曲和斜切字形。",
                    "画笔用于补画，橡皮用于清除；支持绘图板压力，笔尾可临时擦除。",
                    "“保存修改稿”只保存当前编辑结果；“保存并审核通过”同时将该字形推进到整体协调阶段。",
                    "“恢复上次保存”会丢弃当前未保存修改，回到最近一次保存的人工稿。",
                ),
            ),
            HelpSection(
                "画布边界",
                paragraphs=(
                    "田字格外仍保留与田字格同中心的扩展编辑范围。可在允许范围内补画并自动扩展透明画布；超出限制的落笔不会写入图片。",
                ),
            ),
            HelpSection(
                "批量手工审核",
                paragraphs=(
                    "批量审核会自动生成或校验人工稿并推进状态，但不能替代逐字判断。停止时已经完成的字形保留，当前单字完整提交或回滚。",
                ),
            ),
        ),
    ),
    HelpTopic(
        "coordination",
        "整体协调",
        "在全库比较中统一字形大小、位置、形态和墨色，并生成最终成品。",
        ("协调", "未保存", "保存全部修改", "保存本字", "墨色", "最终成品"),
        (
            HelpSection(
                "全库比较",
                items=(
                    "中间区域一次显示当前筛选下的全部字形，列数会随窗口宽度自动调整；普通滚轮只滚动列表。",
                    "选中字形后可直接移动和缩放。列表与比较区会同步选中和定位。",
                    "红色未保存提示表示该字形的几何、墨色或像素修订与上次保存结果不同。滚动离开可视区不会丢失修改。",
                ),
            ),
            HelpSection(
                "单字精调",
                paragraphs=(
                    "按回车或进入单字精调后，可使用与手工审核相同的自由变换、画笔和橡皮。点击“保存本字”只写入当前字形的最终成品和协调记录；“保存并下一字”保存成功后继续到下一字。",
                ),
            ),
            HelpSection(
                "保存全部修改",
                paragraphs=(
                    "比较模式下“保存全部修改”只保存当前确实有未保存修改的字形，不重复写入其他字形。即使修改了很多字并多次滚屏，脏状态仍按字形独立保留。",
                ),
            ),
            HelpSection(
                "批量整体协调",
                paragraphs=(
                    "“批量整体协调”会以整批事务处理全库需要生成或重建的成品，统一计算墨色并更新摘要。整批成功后一起提交；任一关键记录或图片失败时整体回滚，不留下半批结果。进入最终提交后，停止请求会等待提交完整结束。",
                ),
            ),
            HelpSection(
                "墨色模式",
                paragraphs=(
                    "默认跟随全库墨色基准，也可为单字选择保留本字或人工例外。保存后会按最终画布重新实测；墨色未达到当前契约时会显示“墨色待确认”，不能仅凭已经生成文件判断为全部完成。",
                ),
            ),
        ),
    ),
    HelpTopic(
        "export",
        "字库导出",
        "核对最终成品，并以统一或自定义规格安全导出 PNG。",
        ("导出", "PNG", "透明区", "DPI", "覆盖"),
        (
            HelpSection(
                "导出前核对",
                paragraphs=(
                    "导出页只纳入已完成手工审核、进入整体协调阶段的字形。已有成品显示最终图片；待协调项可能显示上游稿件用于识别，但不能作为有效成品导出。",
                ),
            ),
            HelpSection(
                "输出方式",
                items=(
                    "按照字库参数：使用字库的 DPI、宽度和高度。",
                    "去除透明区：按实际内容裁切透明边缘。",
                    "自定义参数：指定 DPI、宽度和高度，并选择完整图片或实际文字区域的缩放方式。",
                ),
            ),
            HelpSection(
                "文件安全",
                paragraphs=(
                    "导出前会集中处理同名文件的覆盖、跳过或取消。批次写入使用临时目录和恢复机制；失败或取消时不会留下未经确认的部分结果。",
                ),
            ),
        ),
    ),
    HelpTopic(
        "scripture_layout",
        "通用经文排版",
        "选择字图来源、检查缺字、调整版式并生成分层 PSD。",
        ("排版", "PSD", "模板", "预览", "经文"),
        (
            HelpSection(
                "基本流程",
                items=(
                    "选择本系统字库或外部字图目录，并输入、粘贴或载入经文内容。",
                    "先检查字图来源和缺失文字，再设置版面、文字、行列与输出参数。",
                    "在预览中检查分版结果，确认后点击“开始生成”。生成完成反馈会显示总耗时。",
                ),
            ),
            HelpSection(
                "模板",
                paragraphs=(
                    "排版模板用于保存通用经文排版参数。可以新建、保存、导入或导出模板；这套模板属于排版功能，与首页“图片实验室”无关。",
                ),
            ),
            HelpSection(
                "停止生成",
                paragraphs=(
                    "生成期间可点击“停止生成”。程序会在当前版的安全边界停止并清理未完成输出，已经完整生成的结果按页面反馈处理。",
                ),
            ),
        ),
    ),
    HelpTopic(
        "custom_layout",
        "定制经文排版",
        "用空行划分版面，并为每一列设置不同的版式参数。",
        ("定制排版", "空行", "分版", "逐列"),
        (
            HelpSection(
                "内容结构",
                paragraphs=(
                    "每个非空行作为一列，空行用于分隔版面。先整理经文换行，再设置板框、列参数、文字间距和输出位置。",
                ),
            ),
            HelpSection(
                "使用建议",
                items=(
                    "先用少量文字验证列顺序、方向和间距，再载入完整经文。",
                    "生成前检查缺字和输出目录，避免完成后才发现字图来源不完整。",
                    "生成任务与通用排版使用相同的安全停止原则。",
                ),
            ),
        ),
    ),
    HelpTopic(
        "text_statistics",
        "文字统计",
        "统计经文字符、频次与缺字，并比较本系统字库或外部字图。",
        ("文字统计", "频次", "缺字", "异体", "比较"),
        (
            HelpSection(
                "统计内容",
                paragraphs=(
                    "选择经文文件，或直接输入、粘贴内容。程序统计有效汉字、不同文字数量和使用频次，并单独列出缺失文字。",
                ),
            ),
            HelpSection(
                "字图比较",
                items=(
                    "选择本系统字库时，列表来自程序数据库中的全部可用字库，而不是旧 JSON 状态索引。",
                    "也可以选择外部字图目录进行比较。文件名识别规则与排版功能保持一致。",
                    "统计结果用于准备字图和发现缺字，不会修改字库状态或图片文件。",
                ),
            ),
        ),
    ),
    HelpTopic(
        "image_lab",
        "图片实验室",
        "独立处理整幅碑文拓片、手稿和其他文字图片扫描件。",
        ("拓片", "扫描件", "整幅", "手稿", "背景清理", "白色清理层", "PSD", "PSB", "Photoshop"),
        (
            HelpSection(
                "功能定位",
                paragraphs=(
                    "图片实验室独立于字库制作流程和经文排版，不读取字库数据库，也不改变任何字形状态。它面向整幅碑文拓片、手稿和文字扫描件，是进入 Photoshop 精修前的自动预处理工具；原稿始终保持不变。",
                ),
            ),
            HelpSection(
                "智能清理与预览",
                items=(
                    "打开图片后，程序在后台生成缩放预览。多通道通用识别同时分析明暗、颜色和笔画尺度，不要求用户判断图片是黑白、灰度还是彩色。",
                    "清理强度越高，去除的弱背景越多；“保护浅色和残损笔迹”适合残碑和淡墨手稿，“清除孤立小噪点”只处理缺乏稳定笔画证据的微小区域。误删笔画的风险高于残留少量噪点，因此默认策略偏向保守。",
                    "“原稿 / 清理效果 / 白色清理层 / 待核对区域”用于切换观察方式。红色区域表示算法证据不足，建议放大核对，不会自动修改字形。",
                ),
            ),
            HelpSection(
                "人工修补",
                items=(
                    "清除背景工具会在清理层增加白色遮盖，保护文字工具会擦除对应遮盖并重新显露原稿。笔触只记录在图片实验室项目中。",
                    "普通滚轮滚屏，Ctrl+滚轮缩放；Ctrl+Z 撤销最近一笔，Ctrl+S 保存项目。",
                    "项目文件使用 .fontlab 扩展名，只保存原稿路径、处理参数和人工笔画。原稿被外部改动后，程序会拒绝直接复用旧项目，避免输出与预览不一致。",
                ),
            ),
            HelpSection(
                "完整尺寸导出",
                items=(
                    "“导出 Photoshop 文件”是后续精修的推荐方式。PSD/PSB 从下到上包含“原稿（锁定）”“白色清理层”“笔画修补”三层；擦除白色清理层可恢复误清理内容，缺失笔画应在笔画修补层补画。",
                    "程序保留原稿 DPI 和 sRGB 配置；超过 PSD 单边 30,000 像素或预计接近 2 GB 时自动改用 PSB。清理效果和白色透明清理层也可分别导出为 TIFF 或 PNG。",
                    "停止导出会等当前分块安全结束，删除临时文件，并且不会覆盖已有目标文件。",
                ),
            ),
        ),
    ),
    HelpTopic(
        "settings",
        "设置与数据维护",
        "配置程序级默认值、常用目录、性能模式和数据库检查。",
        ("设置", "目录", "性能", "缓存", "数据库", "DPI"),
        (
            HelpSection(
                "常规和目录",
                paragraphs=(
                    "可设置新建字库使用的默认 DPI、画布宽高，以及默认图片目录、导出目录和排版输出目录。页面中的临时选择仍可覆盖这些默认值。",
                ),
            ),
            HelpSection(
                "性能与缓存",
                paragraphs=(
                    "自动模式按处理器能力分配后台任务；保守模式减少并行任务，适合内存较小或同时运行其他大型软件的电脑。“释放闲置内存”只清理可重建的页面缓存，不删除字库图片。",
                ),
            ),
            HelpSection(
                "数据维护",
                paragraphs=(
                    "可打开字库、程序数据库和日志所在目录，并核对程序数据库完整性。完整性异常时应停止继续写入，先备份配置目录和相关字库。",
                ),
            ),
        ),
    ),
    HelpTopic(
        "shortcuts",
        "快捷键与鼠标操作",
        "集中查看审核、协调和排版工作台的键盘与鼠标操作。",
        ("快捷键", "鼠标", "滚轮", "Ctrl", "Shift", "Alt", "空格"),
        (
            HelpSection(
                "手工审核与单字精调",
                items=(
                    "Ctrl+S：保存当前字形。",
                    "Ctrl+Z / Ctrl+Y：撤销 / 重做。",
                    "[ / ]：减小 / 增大画笔或橡皮笔触。",
                    "空格 + 鼠标或绘图笔拖动：临时平移画布。",
                    "Ctrl + 滚轮：缩放画布。",
                    "画笔或橡皮工具下普通滚轮：调整笔触大小。",
                    "Shift + 四角拖动：等比约束。",
                    "Alt + 拖动：以中心为锚点缩放。",
                    "Ctrl + 四角拖动：透视扭曲。",
                    "Ctrl + 四边拖动：斜切。",
                ),
            ),
            HelpSection(
                "整体协调比较区",
                items=(
                    "Ctrl+S：按当前模式保存；比较模式保存全部未保存修改，精调模式保存当前字形。",
                    "Ctrl+Z / Ctrl+Y：撤销 / 重做当前编辑。",
                    "字形列表选中项按 Enter：进入单字精调。",
                    "普通滚轮：滚动全库字形列表。",
                    "Ctrl + 滚轮：缩放选中的字形。",
                ),
            ),
            HelpSection(
                "排版编辑与预览",
                items=(
                    "正文编辑区支持系统标准的剪切、复制、粘贴、撤销、重做和全选快捷键。",
                    "预览区滚轮：缩放版面。",
                    "预览区左键或中键拖动：平移版面。",
                ),
            ),
        ),
    ),
    HelpTopic(
        "data_safety",
        "数据安全与故障恢复",
        "了解数据库、图片事务、批处理停止和未保存修改保护。",
        ("SQLite", "恢复", "事务", "停止", "崩溃", "备份", "未保存"),
        (
            HelpSection(
                "数据保存位置",
                paragraphs=(
                    "程序级设置和索引保存在“配置/fontmgr.sqlite3”；每个字库的结构化记录保存在字库目录下的“font_library.sqlite3”。图片仍保存在六个阶段目录中，不写入数据库。",
                ),
            ),
            HelpSection(
                "图片与数据库一致性",
                paragraphs=(
                    "会改变阶段图片的操作使用可恢复图片事务，并与数据库状态一起提交。普通失败会回滚；程序意外退出后，下次启动会依据事务记录完成恢复，避免只改了图片或只改了状态。",
                ),
            ),
            HelpSection(
                "批处理停止",
                paragraphs=(
                    "自动优化和手工审核按单字提交，停止后保留已完成字形。整体协调采用整批事务，提交前停止不会改变本批成品；进入提交后会完整提交或回滚。运行期间应使用页面上的停止按钮，不要直接结束程序。",
                ),
            ),
            HelpSection(
                "离开页面",
                paragraphs=(
                    "手工审核和整体协调检测到未保存修改时，会要求保存、放弃或取消。选择放弃会恢复最近一次保存基线；选择取消会留在当前页面并保留草稿。",
                ),
            ),
        ),
    ),
)


class HelpPage(QWidget):
    """提供主题导航、全文检索和结构化正文的使用说明。"""

    home_requested = Signal()
    status_message = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("helpPage")
        self._topics = HELP_TOPICS
        self._visible_topics: list[HelpTopic] = []
        self._build_ui()
        self._apply_filter("")

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(22, 18, 22, 20)
        root.setSpacing(12)

        header = QHBoxLayout()
        title = QLabel("使用说明")
        title.setProperty("role", "pageTitle")
        header.addWidget(title)
        header.addStretch(1)
        home_button = QPushButton("返回首页")
        home_button.clicked.connect(self.home_requested)
        header.addWidget(home_button)
        root.addLayout(header)

        search_row = QHBoxLayout()
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("搜索功能、操作、状态或快捷键")
        self._search_edit.setClearButtonEnabled(True)
        self._search_edit.textChanged.connect(self._apply_filter)
        search_row.addWidget(self._search_edit, 1)
        self._result_label = QLabel()
        self._result_label.setProperty("role", "muted")
        self._result_label.setMinimumWidth(90)
        self._result_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        search_row.addWidget(self._result_label)
        root.addLayout(search_row)

        content = QHBoxLayout()
        content.setSpacing(12)
        self._topic_list = QListWidget()
        self._topic_list.setObjectName("helpTopicList")
        self._topic_list.setMinimumWidth(210)
        self._topic_list.setMaximumWidth(260)
        self._topic_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._topic_list.currentRowChanged.connect(self._show_topic)
        content.addWidget(self._topic_list)

        article_frame = QFrame()
        article_frame.setProperty("role", "card")
        article_layout = QVBoxLayout(article_frame)
        article_layout.setContentsMargins(0, 0, 0, 0)
        self._article = QTextBrowser()
        self._article.setObjectName("helpArticle")
        self._article.setOpenExternalLinks(False)
        self._article.setFrameShape(QFrame.Shape.NoFrame)
        self._article.setStyleSheet(
            "QTextBrowser#helpArticle { background:#202630; padding:20px 24px; }"
        )
        article_layout.addWidget(self._article)
        content.addWidget(article_frame, 1)
        root.addLayout(content, 1)

    def _apply_filter(self, query: str) -> None:
        normalized = " ".join(query.casefold().split())
        tokens = normalized.split()
        selected_key = ""
        current = self._topic_list.currentItem()
        if current is not None:
            selected_key = str(current.data(Qt.ItemDataRole.UserRole) or "")

        self._visible_topics = [
            topic
            for topic in self._topics
            if not tokens or all(token in topic.search_text for token in tokens)
        ]
        self._topic_list.blockSignals(True)
        self._topic_list.clear()
        selected_row = 0
        for row, topic in enumerate(self._visible_topics):
            item = QListWidgetItem(topic.title)
            item.setData(Qt.ItemDataRole.UserRole, topic.key)
            self._topic_list.addItem(item)
            if topic.key == selected_key:
                selected_row = row
        self._topic_list.blockSignals(False)

        count = len(self._visible_topics)
        self._result_label.setText(f"{count} 个主题")
        if count:
            self._topic_list.setCurrentRow(selected_row)
            self._show_topic(selected_row)
            if normalized:
                self.status_message.emit(f"使用说明：找到 {count} 个相关主题")
        else:
            self._article.setHtml(self._empty_result_html(query))
            self.status_message.emit("使用说明：没有找到相关主题")

    def _show_topic(self, row: int) -> None:
        if not 0 <= row < len(self._visible_topics):
            return
        topic = self._visible_topics[row]
        self._article.setHtml(self._topic_html(topic))
        self._article.verticalScrollBar().setValue(0)

    @staticmethod
    def _topic_html(topic: HelpTopic) -> str:
        sections: list[str] = []
        for section in topic.sections:
            body = "".join(f"<p>{escape(text)}</p>" for text in section.paragraphs)
            if section.items:
                body += "<ul>" + "".join(
                    f"<li>{escape(item)}</li>" for item in section.items
                ) + "</ul>"
            sections.append(f"<h2>{escape(section.title)}</h2>{body}")
        return f"""
        <html><head><style>
        body {{ color:#E8EDF5; font-family:'Microsoft YaHei UI'; font-size:14px; line-height:1.65; }}
        h1 {{ color:#F4F7FB; font-size:25px; margin:0 0 8px 0; }}
        h2 {{ color:#E8EDF5; font-size:17px; margin:22px 0 7px 0; }}
        p {{ margin:5px 0 10px 0; }}
        p.lead {{ color:#AEB9C8; font-size:15px; margin-bottom:18px; }}
        ul {{ margin:5px 0 12px 0; padding-left:22px; }}
        li {{ margin:5px 0; }}
        </style></head><body>
        <h1>{escape(topic.title)}</h1>
        <p class="lead">{escape(topic.summary)}</p>
        {''.join(sections)}
        </body></html>
        """

    @staticmethod
    def _empty_result_html(query: str) -> str:
        return f"""
        <html><head><style>
        body {{ color:#E8EDF5; font-family:'Microsoft YaHei UI'; text-align:center; padding-top:80px; }}
        h1 {{ font-size:21px; }} p {{ color:#AEB9C8; font-size:14px; }}
        </style></head><body>
        <h1>没有找到相关说明</h1>
        <p>未找到与“{escape(query.strip())}”匹配的主题，请尝试“保存”“滚轮”“墨色”或“数据库”。</p>
        </body></html>
        """
