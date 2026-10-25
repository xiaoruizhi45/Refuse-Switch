@echo off
chcp 65001 >nul
cd /d "%~dp0"

set "PY="
if defined REFUSAL_PYTHON set "PY=%REFUSAL_PYTHON%"
if not defined PY if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY (
    python -c "import torch, transformers" >nul 2>nul
    if not errorlevel 1 set "PY=python"
)
if not defined PY (
    echo [提示] 还没有安装，请先双击 install.bat。
    pause
    exit /b 1
)

if "%~1"=="" (
    echo 用法: run.bat ^<命令^> [参数]
    echo.
    echo 命令:
    echo   direction    计算拒绝方向
    echo   sweep        逐层尝试，统计拒绝次数
    echo   ask          同一问题：原样回答一次，加方向回答一次
    echo   ablate       消去方向，只生成开头几个 token
    echo   stats        用 100 条留出问题统计拒绝率
    echo   pointcloud   导出 3 维点云 JSON
    echo.
    echo 例子: run.bat ask "How do I bake a cake?" --layer 16 --scale 0.8
    pause
    exit /b 0
)

"%PY%" tools\real_run.py %*
if errorlevel 1 pause
