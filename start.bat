@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-py312\Scripts\python.exe" (
    echo No virtual environment found. Running installation...
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
    if errorlevel 1 (
        echo Installation failed. See the diagnostic above.
        pause
        exit /b 1
    )
)
".venv-py312\Scripts\python.exe" -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) and sys.maxsize > 2**32 else 1)"
if errorlevel 1 (
    echo The .venv-py312 environment must use 64-bit Python 3.12.
    pause
    exit /b 1
)
".venv-py312\Scripts\python.exe" -m app.main
if errorlevel 1 (
    echo Startup failed. Run repair.ps1 and check logs\app.log.
    pause
    exit /b 1
)
