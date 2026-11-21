@echo off
cd /d "%~dp0.."
echo ========================================
echo    YOLO CV CCTV - All Services Launcher
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

REM Check if ngrok is available
where ngrok >nul 2>&1
if errorlevel 1 (
    echo WARNING: ngrok not found! Public URL will not be available.
    echo Install ngrok from: https://ngrok.com/download
    echo.
    set NGROK_AVAILABLE=0
) else (
    set NGROK_AVAILABLE=1
)

REM Activate venv
call .venv\Scripts\activate.bat

REM Check if required packages are installed
echo Checking required packages...
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Installing streamlit...
    pip install streamlit
)

python -c "import uvicorn, fastapi" 2>nul
if errorlevel 1 (
    echo Installing uvicorn and fastapi...
    pip install uvicorn fastapi
)

echo.
echo ========================================
echo    Starting All Services...
echo ========================================
echo.

REM Start FastAPI in new window
echo [1/3] Starting FastAPI Dashboard...
echo       URL: http://localhost:8080
start "FastAPI Dashboard - Port 8080" cmd /k "cd /d %~dp0.. && call .venv\Scripts\activate.bat && python -m uvicorn src.web.dashboard:app --reload --host localhost --port 8080"
timeout /t 2 /nobreak >nul

REM Start Streamlit in new window
echo [2/3] Starting Streamlit Dashboard...
echo       URL: http://localhost:8501
start "Streamlit Dashboard - Port 8501" cmd /k "cd /d %~dp0.. && call .venv\Scripts\activate.bat && python -m streamlit run streamlit_dashboard.py"
timeout /t 2 /nobreak >nul

REM Start ngrok if available
if %NGROK_AVAILABLE%==1 (
    echo [3/3] Starting ngrok tunnel...
    echo       Check ngrok window for public URL
    start "ngrok - Public Tunnel" cmd /k "cd /d %~dp0.. && ngrok http 8501"
    timeout /t 2 /nobreak >nul
) else (
    echo [3/3] Skipping ngrok (not installed)
)

echo.
echo ========================================
echo    All Services Started!
echo ========================================
echo.
echo FastAPI Dashboard:    http://localhost:8080
echo   - API Docs:         http://localhost:8080/docs
echo   - ReDoc:            http://localhost:8080/redoc
echo.
echo Streamlit Dashboard:  http://localhost:8501
echo.
if %NGROK_AVAILABLE%==1 (
    echo ngrok Public Tunnel:  Check ngrok window for https:// URL
    echo.
)
echo.
echo Press any key to stop all services...
pause >nul

REM Kill all spawned processes when user presses a key
echo.
echo Stopping services...
taskkill /FI "WINDOWTITLE eq FastAPI Dashboard*" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq Streamlit Dashboard*" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq ngrok*" /T /F >nul 2>&1
echo Services stopped.
pause
