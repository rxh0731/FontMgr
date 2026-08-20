from __future__ import annotations

import inspect
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from services.glyph_service import GlyphService
from ui.pages.consistency_page import ConsistencyPage
from ui.pages.export_page import ExportPage
from ui.pages.optimization_page import OptimizationPage
from ui.pages.review_page import ReviewPage
from ui.theme import apply_theme
from ui.widgets.glyph_rename_dialog import GlyphRenameDialog


class GlyphRenameDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])
        apply_theme(cls.app)

    def test_dialog_previews_managed_filename_and_stage_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "名称窗口测试"
            service = GlyphService("名称窗口测试", str(root))
            service.ensure_dirs()
            source_path = Path(service.get_workflow_dirs()["原图"]) / "錯-0001.png"
            image = QImage(32, 32, QImage.Format.Format_ARGB32)
            image.fill(QColor("white"))
            self.assertTrue(image.save(str(source_path), "PNG"))
            variant_id = service.add_original(
                "錯",
                source_path.name,
                source_path.name,
                "a" * 32,
            )
            service.save()

            dialog = GlyphRenameDialog(service, variant_id)
            dialog._char_edit.setText("正")
            self.app.processEvents()

            self.assertTrue(dialog._confirm_button.isEnabled())
            self.assertEqual(dialog._new_filename_label.text(), "正-0001.png")
            self.assertEqual(dialog._stage_list.count(), 1)
            self.assertIn("图片内容和制作状态保持不变", dialog._validation_label.text())
            dialog.close()

    def test_workbenches_expose_rename_only_from_context_menu(self) -> None:
        for page_class in (OptimizationPage, ReviewPage, ConsistencyPage, ExportPage):
            with self.subTest(page=page_class.__name__):
                source = inspect.getsource(page_class)
                self.assertIn('menu.addAction("修正字形名称…")', source)
                self.assertNotIn('QPushButton("修正字形名称")', source)


if __name__ == "__main__":
    unittest.main()
