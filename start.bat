@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [提示] 还没有安装，请先双击 install.bat。
    pause
    exit /b 1
)

".venv\Scripts\python.exe" tools\chat.py %*
if errorlevel 1 pause
