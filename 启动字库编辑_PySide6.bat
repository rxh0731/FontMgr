@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo [错误] 未找到项目虚拟环境：%PYTHON_EXE%
    echo 请先创建虚拟环境并安装 requirements.txt 中的依赖。
    pause
    exit /b 1
)

"%PYTHON_EXE%" main.py
if errorlevel 1 (
    echo.
    echo [错误] 字库编辑器启动失败，退出码：%errorlevel%
    pause
)

endlocal
