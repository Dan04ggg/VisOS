@echo off
cd /d "%~dp0"

REM Try "python" first (standard Windows install), then "python3" (some custom setups)
python --version >nul 2>&1
if not errorlevel 1 (
    python run.py %*
    goto :end
)

python3 --version >nul 2>&1
if not errorlevel 1 (
    python3 run.py %*
    goto :end
)

echo.
echo  ERROR: Python 3 is not installed or not in PATH.
echo  Please install Python 3.9+ from https://python.org
echo.
pause
exit /b 1

:end
