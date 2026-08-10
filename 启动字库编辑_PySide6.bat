@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo [提示] 首次启动，正在创建 Python 3.12 虚拟环境……
    where uv >nul 2>nul
    if not errorlevel 1 (
        uv venv --python 3.12 .venv
        if errorlevel 1 goto :setup_failed
        echo [提示] 正在安装项目依赖，请稍候……
        uv pip install --python "%PYTHON_EXE%" -r requirements.txt
    ) else (
        py -3.12 -m venv .venv
        if errorlevel 1 goto :setup_failed
        echo [提示] 正在安装项目依赖，请稍候……
        "%PYTHON_EXE%" -m pip install -r requirements.txt
    )
    if errorlevel 1 goto :setup_failed
)

"%PYTHON_EXE%" main.py
if errorlevel 1 (
    echo.
    echo [错误] 字库编辑器启动失败，退出码：%errorlevel%
    pause
)
exit /b %errorlevel%

:setup_failed
echo.
echo [错误] 运行环境创建失败，请检查网络连接和 Python 3.12。
pause
exit /b 1
