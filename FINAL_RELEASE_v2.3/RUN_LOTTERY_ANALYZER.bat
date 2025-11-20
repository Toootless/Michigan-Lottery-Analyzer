@echo off
setlocal
title Michigan Lottery Analyzer v2.3 - Final Release

REM Get the directory where this batch file is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo.
echo =================================================
echo    Michigan Lottery Analyzer v2.3
echo    Final Release - Production Ready
echo =================================================
echo.
echo Checking Python installation...

REM Try to find Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    echo.
    pause
    exit /b 1
)

echo Python found!
echo.
echo Installing/checking required packages...
pip install streamlit pandas numpy requests beautifulsoup4 matplotlib seaborn plotly >nul 2>&1
if errorlevel 1 (
    echo Warning: Some packages may not have installed correctly
    echo The application will still attempt to start...
)

echo.
echo Starting the application...
echo Browser will open automatically at http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo.

REM Start the Streamlit application
python -m streamlit run src/MichiganLotteryAnalyzer.py

echo.
echo Application stopped.
pause