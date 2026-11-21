@echo off
echo ========================================
echo    Starting ngrok Tunnel
echo ========================================
echo.

REM Check if ngrok is available
where ngrok >nul 2>&1
if errorlevel 1 (
    echo ERROR: ngrok not found in PATH!
    echo.
    echo Please install ngrok:
    echo   1. Download from: https://ngrok.com/download
    echo   2. Extract ngrok.exe to this folder or add to PATH
    echo   3. Run: ngrok config add-authtoken YOUR_TOKEN
    echo.
    pause
    exit /b 1
)

REM Default port
set PORT=8501

REM Check if user wants to specify a different port
if not "%1"=="" (
    set PORT=%1
)

echo.
echo ========================================
echo    Creating Public Tunnel...
echo ========================================
echo.
echo Local Port:  %PORT%
echo.
echo Waiting for ngrok to start...
echo Once running, look for "Forwarding" line below
echo Copy the https://xxxxx.ngrok-free.app URL
echo.
echo Press CTRL+C to stop ngrok
echo.

ngrok http %PORT%
pause
