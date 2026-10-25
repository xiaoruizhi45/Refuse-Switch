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

"%PY%" tools\chat.py %*
if errorlevel 1 pause
