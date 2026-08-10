# FontEditor-PySide6

这是从 `D:\Code\FontEditor` 分离出的 PySide6 改造项目。旧版项目保持原状，可随时返回继续维护。

## 当前状态

- 已复制与 GUI 无关的业务层：`core`、`data`、`services`、`utils`。
- 已复制旧版 Tk 界面到 `reference/tk_ui`，仅供行为和布局参考。
- 已复制并验证流畅的 PySide6 画布原型到 `prototype`。
- 已复制算法注册表、应用图标和小型测试字库。
- 已建立现行规范 `字库编辑_PySide6_开发说明书.md`，旧版说明书已归档到 `reference`。
- `ui` 是新 PySide6 正式界面的开发目录，目前尚未开始正式迁移。

## 开始开发

在 PowerShell 中执行：

```powershell
cd D:\Code\FontEditor-PySide6
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\prototype\PySide6画布性能测试.py
```

如果系统 `py` 不可用，可使用已经安装的兼容 Python 创建 `.venv`。

## 新会话建议提示

> 请继续开发 `D:\Code\FontEditor-PySide6`。先阅读 `AGENTS.md`、`字库编辑_PySide6_开发说明书.md` 和 `README.md`，核对当前 Git 状态，然后按开发说明书从 PySide6 应用骨架和手工审核页面开始。`D:\Code\FontEditor` 只读且必须保持不变。

## 隔离说明

新项目没有复制旧版的 `.venv`、Git 历史、日志、打包目录、便携版、完整业务字库或备份。测试字库是约 1.5 MB 的独立副本，因此新项目的读写不会影响旧版数据。
