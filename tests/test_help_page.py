"""应用内使用说明页面回归测试。"""

from __future__ import annotations

import unittest

from PySide6.QtWidgets import QApplication, QPushButton

from ui.pages.help_page import HELP_TOPICS, HelpPage


class HelpPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_topics_cover_main_workflows_and_data_safety(self) -> None:
        titles = {topic.title for topic in HELP_TOPICS}

        self.assertGreaterEqual(len(HELP_TOPICS), 14)
        self.assertTrue(
            {
                "快速开始",
                "字形列表与搜索",
                "新建字库与字库添加",
                "自动优化",
                "手工审核",
                "整体协调",
                "字库导出",
                "通用经文排版",
                "定制经文排版",
                "文字统计",
                "图片实验室",
                "设置与数据维护",
                "快捷键与鼠标操作",
                "数据安全与故障恢复",
            }.issubset(titles)
        )

        search_topic = next(
            topic for topic in HELP_TOPICS if topic.title == "字形列表与搜索"
        )
        self.assertIn("修正字形名称", search_topic.search_text)

    def test_search_filters_topics_and_renders_selected_article(self) -> None:
        page = HelpPage()
        page._search_edit.setText("Ctrl+S")
        self.app.processEvents()

        titles = {
            page._topic_list.item(index).text()
            for index in range(page._topic_list.count())
        }
        self.assertEqual(titles, {"图片实验室", "快捷键与鼠标操作"})
        self.assertIn("Ctrl+S", page._article.toPlainText())
        page.deleteLater()

    def test_search_accepts_multiple_terms(self) -> None:
        page = HelpPage()
        page._search_edit.setText("整幅 清理层")
        self.app.processEvents()

        self.assertEqual(page._topic_list.count(), 1)
        self.assertEqual(page._topic_list.currentItem().text(), "图片实验室")
        page.deleteLater()

    def test_empty_search_result_has_clear_feedback(self) -> None:
        page = HelpPage()
        page._search_edit.setText("不存在的操作名称")
        self.app.processEvents()

        self.assertEqual(page._topic_list.count(), 0)
        self.assertIn("没有找到相关说明", page._article.toPlainText())
        page.deleteLater()

    def test_topic_navigation_changes_article(self) -> None:
        page = HelpPage()
        target_row = next(
            index
            for index, topic in enumerate(HELP_TOPICS)
            if topic.title == "整体协调"
        )
        page._topic_list.setCurrentRow(target_row)
        self.app.processEvents()

        article = page._article.toPlainText()
        self.assertIn("整体协调", article)
        self.assertIn("保存全部修改", article)
        self.assertIn("批量整体协调", article)
        page.deleteLater()

    def test_home_button_emits_navigation_signal(self) -> None:
        page = HelpPage()
        emissions: list[bool] = []
        page.home_requested.connect(lambda: emissions.append(True))

        home_buttons = [
            child
            for child in page.findChildren(QPushButton)
            if child.text() == "返回首页"
        ]
        self.assertEqual(len(home_buttons), 1)
        home_buttons[0].click()

        self.assertEqual(emissions, [True])
        page.deleteLater()


if __name__ == "__main__":
    unittest.main()
