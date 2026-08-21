# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


project_root = Path(SPEC).resolve().parent

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
    ],
    hiddenimports=[],
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
