@echo off
cd /d "%~dp0.."
setlocal enabledelayedexpansion
echo ========================================
echo    YOLO CV CCTV - Starting System
echo ========================================
echo.

REM Check if venv exists
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Please create venv first: python -m venv .venv
    echo Then install packages: pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo Starting CCTV system...
echo.
REM Set the Python path directly instead of using activate.bat
set "PATH=%CD%\.venv\Scripts;%PATH%"
"%CD%\.venv\Scripts\python.exe" main.py
pause
