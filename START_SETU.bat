@echo off
setlocal enabledelayedexpansion
title SETU - offline comprehension engine
cd /d "%~dp0"

echo.
echo   SETU - offline comprehension engine
echo   ===================================
echo.

rem ---------------------------------------------------------------- python
set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

if not defined PY (
    where py >nul 2>&1 && set "PY=py -3"
)
if not defined PY (
    where python >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo   Python 3.10 or newer was not found.
    echo   Install it from https://python.org and run this file again.
    echo.
    pause
    exit /b 1
)

rem ------------------------------------------------------- first-run setup
if not exist ".venv\Scripts\python.exe" (
    echo   First run - creating a private environment. This takes a few minutes.
    echo.
    %PY% -m venv .venv
    if errorlevel 1 (
        echo   Could not create the environment. Is Python 3.10+ installed?
        pause
        exit /b 1
    )
    set "PY=.venv\Scripts\python.exe"
    "!PY!" -m pip install --upgrade pip --quiet
    echo   Installing SETU...
    "!PY!" -m pip install -e . --quiet
    if errorlevel 1 (
        echo   Install failed. See the messages above.
        pause
        exit /b 1
    )
    echo   Installing the on-device runtime...
    "!PY!" -m pip install "onnxruntime>=1.22" --quiet
    echo.
    echo   Done. Model files are optional - SETU runs without them and
    echo   labels anything it could not do for real.
    echo.
)

set "PY=.venv\Scripts\python.exe"

rem ------------------------------------------------------------- self-check
echo   Checking this machine...
echo.
"%PY%" -m setu.cli doctor
echo.

rem ---------------------------------------------------------------- launch
echo   Starting SETU on http://127.0.0.1:8756
echo   A browser window will open. Close this window to stop the server.
echo.
start "" http://127.0.0.1:8756
"%PY%" -m setu.server.app

echo.
echo   SETU has stopped.
pause
