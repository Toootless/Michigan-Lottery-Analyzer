Param(
    [string]$EnvName = "mla4-py311",
    [string]$PythonVersion = "3.11"
)

# Purpose: Optional bootstrap to create a conda env and install requirements.
# Usage:   ./setup_env.ps1  (creates env if missing)

function Write-Info($msg) { Write-Host "[SETUP] $msg" -ForegroundColor Cyan }
function Write-Err($msg)  { Write-Host "[SETUP] $msg" -ForegroundColor Red }

$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent $PSCommandPath
Set-Location $workspaceRoot

$conda = (Get-Command conda -ErrorAction SilentlyContinue)?.Source
if (-not $conda) { Write-Err "conda not found on PATH. Please install Miniconda/Anaconda first."; exit 1 }

# Check env exists
Write-Info "Checking for conda env '$EnvName'..."
$envs = & $conda env list --json | ConvertFrom-Json
$exists = $false
foreach ($p in $envs.envs) { if ($p -like "*\$EnvName") { $exists = $true; break } }

if (-not $exists) {
    Write-Info "Creating env '$EnvName' with Python $PythonVersion..."
    & $conda create -y -n $EnvName python=$PythonVersion
} else {
    Write-Info "Env '$EnvName' already exists."
}

$req = Join-Path $workspaceRoot "requirements.txt"
if (Test-Path $req) {
    Write-Info "Installing requirements.txt into '$EnvName'..."
    & $conda run -n $EnvName python -m pip install --upgrade pip
    & $conda run -n $EnvName python -m pip install -r "$req"
} else {
    Write-Info "No requirements.txt found; skipping package install."
}

Write-Info "Done. Use:  & $conda run -n $EnvName pwsh -NoProfile -File run_MLA4.ps1"
