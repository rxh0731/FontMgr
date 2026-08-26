# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files


project_root = Path(SPEC).resolve().parent
rapidocr_datas = collect_data_files("rapidocr")

a = Analysis(
    [str(project_root / "main.py")],
    pathex=[str(project_root)],
    binaries=[],
    datas=[
        (str(project_root / "FontEditor.ico"), "."),
        (
            str(project_root / "assets" / "font_editor_icon_color_blocks.png"),
            "assets",
        ),
        *rapidocr_datas,
    ],
    hiddenimports=[
        "rapidocr.main",
        "rapidocr.inference_engine.onnxruntime",
    ],
    hookspath=[],
    hooksconfig={},
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="字库编辑器",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=[str(project_root / "FontEditor.ico")],
)
