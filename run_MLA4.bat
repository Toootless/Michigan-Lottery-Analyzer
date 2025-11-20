@echo off
REM Simple double-click launcher for MLA4 using PowerShell script
setlocal
set SCRIPT_DIR=%~dp0
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%run_MLA4.ps1"
endlocal