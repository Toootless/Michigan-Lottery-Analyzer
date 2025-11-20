@echo off
setlocal
title Michigan Lottery Analyzer v2.3 - With Conda Environment

REM Get the directory where this batch file is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo.
echo =================================================
echo    Michigan Lottery Analyzer v2.3
echo    Using Conda Environment - Production Ready
echo =================================================
echo.

REM Check if the conda environment exists
if exist "..\..\.conda\python.exe" (
    echo Using conda environment...
    set "PYTHON_EXE=..\..\.conda\python.exe"
) else (
    echo Conda environment not found, using system Python...
    set "PYTHON_EXE=python"
)

echo.
echo Starting the application...
echo Browser will open automatically at http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo.

REM Start the Streamlit application
"%PYTHON_EXE%" -m streamlit run src/MichiganLotteryAnalyzer.py

echo.
echo Application stopped.
pause