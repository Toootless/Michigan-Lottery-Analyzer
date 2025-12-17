Param(
    [switch]$DryRun,
    [string]$Port = "8506",
    [string]$Address = "localhost"
)

# Purpose: Single-entry launcher for MLA4 Streamlit app.
# It detects local .conda environment python, or falls back to a named conda env (mla4-py311), then launches Streamlit.
# Usage:  ./run_MLA4.ps1  (normal) | ./run_MLA4.ps1 -DryRun | ./run_MLA4.ps1 -Port 8510

function Write-Info($msg) { Write-Host "[MLA4] $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "[MLA4] $msg" -ForegroundColor Yellow }
function Write-Err($msg)  { Write-Host "[MLA4] $msg" -ForegroundColor Red }

$ErrorActionPreference = 'Stop'

$workspaceRoot = Split-Path -Parent $PSCommandPath
Set-Location $workspaceRoot

$localPython = Join-Path $workspaceRoot ".conda\Scripts\python.exe"
$condaCmd = Get-Command conda -ErrorAction SilentlyContinue
$condaExe = if ($condaCmd) { $condaCmd.Source } else { $null }
$fallbackEnv  = "mla4-py311"

if ($DryRun) { Write-Info "DryRun mode: will not execute Streamlit." }

if (Test-Path $localPython) {
    Write-Info "Using local venv python: $localPython"
    $exe = $localPython
    $launchArgs = @('-m','streamlit','run','MLA4.py','--server.address', $Address,'--server.port', $Port)
    $fullCmd = '"' + $exe + '" ' + ($launchArgs -join ' ')
} elseif ($condaExe) {
    Write-Info "Local python not found, will use conda run -n $fallbackEnv"
    $exe = $condaExe
    $launchArgs = @('run','-n', $fallbackEnv,'python','-m','streamlit','run','MLA4.py','--server.address', $Address,'--server.port', $Port)
    $fullCmd = '"' + $exe + '" ' + ($launchArgs -join ' ')
} else {
    Write-Err "No local venv and conda not found. Please install Miniconda/Anaconda or create .conda environment."
    exit 1
}

Write-Info "Launch command: $fullCmd"

if ($DryRun) { Write-Info "DryRun complete."; exit 0 }

try {
    # Start Streamlit without messing up quoting; keep in current console for logs
    & $exe $launchArgs
} catch {
    Write-Err "Failed to launch Streamlit: $($_.Exception.Message)"
    exit 1
}
