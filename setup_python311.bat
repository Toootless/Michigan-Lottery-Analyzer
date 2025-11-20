@echo off
REM Python 3.11 Setup Script for Lottery Analyzer
REM This script will set up Python 3.11 environment with CUDA support

echo ====================================================
echo Python 3.11 Setup for Lottery Analyzer
echo ====================================================

echo.
echo Step 1: Checking Python version...
python --version
echo.

echo Step 2: Creating Python 3.11 virtual environment...
python -m venv .venv311
echo Virtual environment created successfully!

echo.
echo Step 3: Activating virtual environment...
call .venv311\Scripts\activate.bat

echo.
echo Step 4: Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Step 5: Installing PyTorch with CUDA support...
echo This may take a few minutes...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo.
echo Step 6: Installing core dependencies...
pip install -r requirements_py311.txt

echo.
echo Step 7: Testing CUDA installation...
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device Count: {torch.cuda.device_count()}'); print(f'GPU Name: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'GPU Name: None'); print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB' if torch.cuda.is_available() else 'GPU Memory: None')"

echo.
echo ====================================================
echo Setup Complete!
echo ====================================================
echo.
echo To run the application:
echo 1. Make sure the virtual environment is activated:
echo    .venv311\Scripts\activate
echo 2. Run the application:
echo    streamlit run MichiganLotteryAnalyzer.py
echo.
echo Your RTX 3060 should now have full CUDA support!
echo ====================================================

pause