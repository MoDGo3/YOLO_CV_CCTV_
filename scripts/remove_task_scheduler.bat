@echo off
echo ========================================
echo    Remove Auto-Start Task Scheduler
echo ========================================
echo.

REM Check for admin privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: Administrator privileges required!
    echo Please right-click this file and select "Run as administrator"
    echo.
    pause
    exit /b 1
)

REM Delete scheduled task
echo Removing scheduled task...
schtasks /delete /tn "YOLO_CV_CCTV_AutoStart" /f

if %errorLevel% equ 0 (
    echo.
    echo Task removed successfully!
    echo The system will no longer start automatically.
) else (
    echo.
    echo No task found or failed to remove.
)

echo.
pause
