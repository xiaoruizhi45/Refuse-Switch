@echo off
chcp 65001 >nul
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [错误] 没有找到 Python。
    echo 请到 https://www.python.org/downloads/ 安装 Python 3.10 或更新版本，
    echo 安装时务必勾选 "Add python.exe to PATH"，然后重新双击本文件。
    pause
    exit /b 1
)

python tools\install.py
if errorlevel 1 (
    pause
    exit /b 1
)
pause
