# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs


project_root = Path(SPEC).resolve().parent
rapidocr_datas = collect_data_files("rapidocr")
opencc_datas, opencc_binaries, opencc_hiddenimports = collect_all("opencc")
psd_tools_datas, psd_tools_binaries, psd_tools_hiddenimports = collect_all("psd_tools")
onnxruntime_binaries = collect_dynamic_libs("onnxruntime")

a = Analysis(
    [str(project_root / "main.py")],
    pathex=[str(project_root)],
    binaries=[
        *opencc_binaries,
        *psd_tools_binaries,
        *onnxruntime_binaries,
    ],
    datas=[
        (str(project_root / "FontEditor.ico"), "."),
        (
            str(project_root / "assets" / "font_editor_icon_color_blocks.png"),
            "assets",
        ),
        *rapidocr_datas,
        *opencc_datas,
        *psd_tools_datas,
    ],
    hiddenimports=[
        *opencc_hiddenimports,
        *psd_tools_hiddenimports,
        "rapidocr.main",
        "rapidocr.inference_engine.onnxruntime",
        "rapidocr.inference_engine.onnxruntime.main",
        "rapidocr.inference_engine.onnxruntime.provider_config",
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
