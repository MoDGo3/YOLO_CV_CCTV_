@echo off
echo ========================================
echo    Install Auto-Start on Windows Boot
echo ========================================
echo.
echo This will create a shortcut in Windows Startup folder
echo The CCTV system will run automatically when you login
echo.
pause

REM Get Startup folder path
set STARTUP=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup

REM Create VBScript to make shortcut
echo Set oWS = WScript.CreateObject("WScript.Shell") > CreateShortcut.vbs
echo sLinkFile = "%STARTUP%\YOLO_CV_CCTV.lnk" >> CreateShortcut.vbs
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> CreateShortcut.vbs
echo oLink.TargetPath = "%~dp0run.bat" >> CreateShortcut.vbs
echo oLink.WorkingDirectory = "%~dp0.." >> CreateShortcut.vbs
echo oLink.Description = "YOLO CV CCTV Auto Start" >> CreateShortcut.vbs
echo oLink.IconLocation = "%SystemRoot%\System32\SHELL32.dll,21" >> CreateShortcut.vbs
echo oLink.Save >> CreateShortcut.vbs

REM Run VBScript
cscript //nologo CreateShortcut.vbs

REM Clean up
del CreateShortcut.vbs

echo.
echo ========================================
echo    Installation Complete!
echo ========================================
echo.
echo Shortcut created at:
echo %STARTUP%\YOLO_CV_CCTV.lnk
echo.
echo The system will now start automatically when you login.
echo.
echo To remove auto-start:
echo 1. Press Win+R
echo 2. Type: shell:startup
echo 3. Delete "YOLO_CV_CCTV.lnk"
echo.
pause
