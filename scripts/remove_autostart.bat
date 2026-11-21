@echo off
echo ========================================
echo    Remove Auto-Start from Windows Boot
echo ========================================
echo.

REM Get Startup folder path
set STARTUP=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup

REM Check if shortcut exists
if exist "%STARTUP%\YOLO_CV_CCTV.lnk" (
    del "%STARTUP%\YOLO_CV_CCTV.lnk"
    echo Auto-start shortcut removed successfully!
    echo.
    echo The system will no longer start automatically on login.
) else (
    echo No auto-start shortcut found.
    echo The system is not set to auto-start.
)

echo.
pause
