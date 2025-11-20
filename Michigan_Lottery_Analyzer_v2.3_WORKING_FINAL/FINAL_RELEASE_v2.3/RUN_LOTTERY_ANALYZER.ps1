# Michigan Lottery Analyzer v2.3 - PowerShell Launcher
# Production Ready Release

# Set window title
$Host.UI.RawUI.WindowTitle = "Michigan Lottery Analyzer v2.3 - Final Release"

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host ""
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "   Michigan Lottery Analyzer v2.3" -ForegroundColor Yellow
Write-Host "   Final Release - Production Ready" -ForegroundColor Yellow  
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""

# Check for Python
Write-Host "Checking Python installation..." -ForegroundColor Green
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
    } else {
        throw "Python not found"
    }
} catch {
    Write-Host "❌ Error: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ and try again" -ForegroundColor Red
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# Check/install required packages
Write-Host ""
Write-Host "Checking required packages..." -ForegroundColor Green
$packages = @("streamlit", "pandas", "numpy", "requests", "beautifulsoup4", "matplotlib", "seaborn", "plotly")

foreach ($package in $packages) {
    try {
        python -c "import $package" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ $package is available" -ForegroundColor Green
        } else {
            Write-Host "⚠️  Installing $package..." -ForegroundColor Yellow
            pip install $package --quiet
        }
    } catch {
        Write-Host "⚠️  Could not verify $package" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Starting the application..." -ForegroundColor Green
Write-Host "Browser will open automatically at http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the application" -ForegroundColor Yellow
Write-Host ""

# Start the Streamlit application
try {
    python -m streamlit run src/MichiganLotteryAnalyzer.py
} catch {
    Write-Host ""
    Write-Host "❌ Error starting the application: $_" -ForegroundColor Red
    Write-Host ""
}

Write-Host ""
Write-Host "Application stopped." -ForegroundColor Yellow
Read-Host "Press Enter to exit"