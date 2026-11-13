@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo No virtual environment found. Running installation...
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
    if errorlevel 1 (
        echo Installation failed. See the diagnostic above.
        pause
        exit /b 1
    )
)
".venv\Scripts\python.exe" -m app.main
if errorlevel 1 (
    echo Startup failed. Run repair.ps1 and check logs\app.log.
    pause
    exit /b 1
)
