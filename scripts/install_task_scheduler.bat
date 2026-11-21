@echo off
echo ========================================
echo    Install Auto-Start via Task Scheduler
echo ========================================
echo.
echo This will create a Windows scheduled task to run
echo the CCTV system at startup (even before login)
echo.
echo Requires Administrator privileges!
echo.
pause

REM Check for admin privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: Administrator privileges required!
    echo Please right-click this file and select "Run as administrator"
    echo.
    pause
    exit /b 1
)

REM Create scheduled task
echo Creating scheduled task...
schtasks /create /tn "YOLO_CV_CCTV_AutoStart" /tr "\"%~dp0run.bat\"" /sc onstart /ru "%USERNAME%" /rl highest /f

if %errorLevel% equ 0 (
    echo.
    echo ========================================
    echo    Installation Complete!
    echo ========================================
    echo.
    echo Task Name: YOLO_CV_CCTV_AutoStart
    echo Trigger:   At system startup
    echo.
    echo The system will run automatically when Windows starts.
    echo.
    echo To view/modify task:
    echo   1. Press Win+R
    echo   2. Type: taskschd.msc
    echo   3. Find "YOLO_CV_CCTV_AutoStart"
    echo.
    echo To remove:
    echo   Run: remove_task_scheduler.bat
    echo.
) else (
    echo.
    echo ERROR: Failed to create scheduled task!
    echo.
)

pause
