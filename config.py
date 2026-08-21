# config.py — 全局常量、路径、默认参数

import os
import sys

if getattr(sys, "frozen", False):
    SCRIPT_DIR: str = os.path.dirname(os.path.abspath(sys.executable))
    RESOURCE_DIR: str = getattr(sys, "_MEIPASS", SCRIPT_DIR)
else:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    RESOURCE_DIR = SCRIPT_DIR
CONFIG_DIR: str = os.path.join(SCRIPT_DIR, "配置")
APP_DATABASE_FILE: str = os.path.join(CONFIG_DIR, "fontmgr.sqlite3")
REGISTRY_FILE: str = os.path.join(CONFIG_DIR, "算法注册表.json")
TEMPLATE_FILE: str = os.path.join(CONFIG_DIR, "优化模板.json")
GLOBAL_CONFIG_FILE: str = os.path.join(CONFIG_DIR, "用户设置.json")
LIBRARY_SUMMARY_CACHE_FILE: str = os.path.join(CONFIG_DIR, "字库状态索引.json")
LAYOUT_TEMPLATE_FILE: str = os.path.join(CONFIG_DIR, "通用经文排版模板.json")
LEGACY_LAYOUT_TEMPLATE_FILE: str = os.path.join(CONFIG_DIR, "排版模板.json")
CUSTOM_LAYOUT_TEMPLATE_FILE: str = os.path.join(CONFIG_DIR, "定制经文排版模板.json")
ZIKU_ROOT: str = os.path.join(SCRIPT_DIR, "字库")
LOG_FILE: str = os.path.join(SCRIPT_DIR, "font_editor.log")
LOG_MAX_BYTES: int = 2 * 1024 * 1024

WELCOME_BG_TITLE: str = "般若波羅蜜多心經"
WELCOME_BG_BODY: str = (
    "觀自在菩薩行深般若波羅蜜多時照見五蘊皆空度一切苦厄"
    "舍利子色不異空空不異色色即是空空即是色受想行識亦復如是"
    "舍利子是諸法空相不生不滅不垢不淨不增不減"
    "是故空中無色無受想行識無眼耳鼻舌身意無色聲香味觸法無眼界乃至無意識界"
    "無無明亦無無明盡乃至無老死亦無老死盡無苦集滅道無智亦無得"
    "以無所得故菩提薩埵依般若波羅蜜多故心無罣礙無罣礙故無有恐怖遠離顛倒夢想究竟涅槃"
    "三世諸佛依般若波羅蜜多故得阿耨多羅三藐三菩提"
    "故知般若波羅蜜多是大神咒是大明咒是無上咒是無等等咒能除一切苦真實不虛"
    "故說般若波羅蜜多咒即說咒曰揭諦揭諦波羅揭諦波羅僧揭諦菩提薩婆訶"
)
WELCOME_BG_TEXT: str = WELCOME_BG_TITLE + WELCOME_BG_BODY
WELCOME_FONT_FAMILIES: tuple = ("楷体", "KaiTi", "MingLiU", "細明體", "SimSun", "宋体", "Microsoft YaHei", "微软雅黑")
WELCOME_MIN_FONT_SIZE: int = 8

DIR_ORIGINAL_FILES: str = "01_源图"
DIR_GRAY_MASTER_FILES: str = "02_灰度底稿"
DIR_CLEAN_MASK_FILES: str = "03_字形掩码"
DIR_INTERMEDIATE_FILES: str = "04_自动优化稿"
DIR_REVIEWED_FILES: str = "05_人工修订稿"
DIR_FINISHED_FILES: str = "06_最终成品"

STATUS_PENDING_OPTIMIZATION: str = "待优化"
STATUS_PENDING_MANUAL_REVIEW: str = "待审核"
STATUS_REVIEWED: str = "审核通过"
STATUS_FINISHED: str = "成品已生成"
ALL_STATUSES: tuple[str, ...] = (
    STATUS_PENDING_OPTIMIZATION,
    STATUS_PENDING_MANUAL_REVIEW,
    STATUS_REVIEWED,
    STATUS_FINISHED,
)
STATUS_COLORS: dict[str, str] = {
    STATUS_PENDING_OPTIMIZATION: "#888888",
    STATUS_PENDING_MANUAL_REVIEW: "#FF8C00",
    STATUS_REVIEWED: "#4169E1",
    STATUS_FINISHED: "#228B22",
}

MAX_IMPORT_WORKERS: int = max(1, (os.cpu_count() or 4) // 2)
THUMB_CACHE_MAX: int = 200
ICON_FILE: str = os.path.join(RESOURCE_DIR, "FontEditor.ico")
WINDOW_ICON_FILE: str = os.path.join(
    RESOURCE_DIR,
    "assets",
    "font_editor_icon_color_blocks.png",
)
WINDOWS_APP_USER_MODEL_ID: str = "RuanXiaohua.FontMgr"
