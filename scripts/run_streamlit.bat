@echo off
cd /d "%~dp0.."
setlocal enabledelayedexpansion
echo ========================================
echo    Starting Streamlit Dashboard
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

REM Set the Python path directly
set "PATH=%CD%\.venv\Scripts;%PATH%"

REM Check if streamlit is installed
echo Checking required packages...
"%CD%\.venv\Scripts\python.exe" -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Installing streamlit...
    "%CD%\.venv\Scripts\pip.exe" install streamlit
    echo.
)

echo.
echo Starting Streamlit Dashboard...
echo Dashboard will open at: http://localhost:8501
echo.
echo Press CTRL+C to stop the server
echo.

"%CD%\.venv\Scripts\python.exe" -m streamlit run streamlit_dashboard.py
pause
