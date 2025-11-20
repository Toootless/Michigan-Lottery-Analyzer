param(
  [switch]$Headless,
  [string]$EnvName = 'mla4-py311'
)

# Determine repo root relative to this script
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python   = Join-Path $RepoRoot ".conda/python.exe"
$Entry    = Join-Path $RepoRoot "MLA4.py"

if (-not (Test-Path $Entry)) {
  Write-Host "Entry file not found at $Entry" -ForegroundColor Red
  exit 1
}

$headlessArg = if ($Headless) { 'true' } else { 'false' }

# Prefer a Conda environment if available; fall back to repo .conda/python.exe
$condaExe = $null
try {
  $condaExe = (Get-Command conda -ErrorAction Stop).Source
} catch {
  $userConda = Join-Path $env:USERPROFILE 'anaconda3\Scripts\conda.exe'
  if (Test-Path $userConda) { $condaExe = $userConda }
}

if ($condaExe) {
  Write-Host "Using conda environment '$EnvName' to run MLA4" -ForegroundColor Cyan
  $procArgs = @('run','-n', $EnvName, 'python','-m','streamlit','run', $Entry,'--server.headless', $headlessArg,'--server.address','localhost','--server.port','8506')
  $p = Start-Process -FilePath $condaExe -ArgumentList $procArgs -PassThru -WindowStyle Hidden
  Write-Host "Started Streamlit (PID: $($p.Id)) at http://localhost:8506" -ForegroundColor Green
  exit 0
} else {
  if (-not (Test-Path $Python)) {
    Write-Host "Conda not found and repo Python not found at $Python" -ForegroundColor Red
    exit 1
  }
  Write-Host "Conda not found; using repo Python at $Python" -ForegroundColor Yellow
  Start-Process -FilePath $Python -ArgumentList @('-m','streamlit','run', $Entry,'--server.headless', $headlessArg,'--server.address','localhost','--server.port','8506') -WindowStyle Hidden | Out-Null
  Write-Host "Started Streamlit with repo Python at http://localhost:8506" -ForegroundColor Green
  exit 0
}
