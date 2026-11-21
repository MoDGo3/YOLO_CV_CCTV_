@echo off
cd /d "%~dp0.."
setlocal enabledelayedexpansion
echo ========================================
echo    Starting FastAPI Dashboard
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

REM Check if required packages are installed
echo Checking required packages...
"%CD%\.venv\Scripts\python.exe" -c "import uvicorn, fastapi" 2>nul
if errorlevel 1 (
    echo Installing uvicorn and fastapi...
    "%CD%\.venv\Scripts\pip.exe" install uvicorn fastapi
    echo.
)

echo.
echo ========================================
echo    Starting FastAPI Server...
echo ========================================
echo.
echo Dashboard will be available at:
echo   - Local:    http://localhost:8080
echo   - API Docs: http://localhost:8080/docs
echo   - ReDoc:    http://localhost:8080/redoc
echo.
echo Press CTRL+C to stop the server
echo.

"%CD%\.venv\Scripts\python.exe" -m uvicorn src.web.dashboard:app --reload --host localhost --port 8080
pause
