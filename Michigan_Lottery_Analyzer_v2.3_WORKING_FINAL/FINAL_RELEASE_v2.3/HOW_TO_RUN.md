# How to Run Michigan Lottery Analyzer v2.3

## 🚀 Quick Start Options

### Option 1: Standard Batch File (Recommended)
**File**: `RUN_LOTTERY_ANALYZER.bat`
- **How**: Double-click the file
- **Requirements**: Python 3.8+ installed on your system
- **Features**: 
  - Automatic package installation
  - Error checking
  - Works from any location

### Option 2: PowerShell Script (Advanced)
**File**: `RUN_LOTTERY_ANALYZER.ps1`
- **How**: Right-click → "Run with PowerShell"
- **Requirements**: PowerShell enabled, Python 3.8+
- **Features**:
  - Enhanced error messages
  - Better package verification
  - Colored output

### Option 3: Conda Environment (For Development)
**File**: `RUN_LOTTERY_ANALYZER_CONDA.bat`
- **How**: Double-click the file
- **Requirements**: Must be in same directory structure as original project
- **Features**:
  - Uses specific conda environment if available
  - Fallback to system Python

### Option 4: Manual Command Line
**Commands**:
```bash
cd FINAL_RELEASE_v2.3
pip install -r requirements.txt
python -m streamlit run src/MichiganLotteryAnalyzer.py
```

## 🔧 Troubleshooting

### "Python not found" Error:
1. Install Python 3.8+ from https://python.org
2. Make sure "Add Python to PATH" is checked during installation
3. Restart your command prompt/PowerShell

### "Package installation failed" Error:
1. Try running as Administrator
2. Update pip: `python -m pip install --upgrade pip`
3. Install packages manually: `pip install streamlit pandas numpy requests beautifulsoup4`

### Application doesn't start:
1. Check if port 8501 is already in use
2. Try: `python -m streamlit run src/MichiganLotteryAnalyzer.py --server.port 8502`
3. Restart your computer and try again

### Browser doesn't open automatically:
1. Manually navigate to: http://localhost:8501
2. Make sure your firewall isn't blocking the connection
3. Try a different browser (Chrome, Firefox, Edge)

## 📋 System Requirements

### Minimum:
- **OS**: Windows 10, macOS 10.15, or Linux Ubuntu 18.04+
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum
- **Storage**: 500MB free space
- **Internet**: Required for web features and purchase integration

### Recommended:
- **Python**: 3.11 for best compatibility
- **RAM**: 8GB for smooth performance
- **Browser**: Chrome or Firefox for best experience

## 🎯 First Time Setup

1. **Extract the ZIP file** to your desired location
2. **Double-click** `RUN_LOTTERY_ANALYZER.bat`
3. **Wait** for automatic package installation (first time only)
4. **Browser opens** automatically to http://localhost:8501
5. **Start analyzing** lottery numbers!

## 🎫 Using the Purchase Feature

### For Online Games (Powerball, Mega Millions, etc.):
1. Go to "🎯 Smart Number Suggestions" tab
2. Generate your number recommendations
3. Click the red "🎫 Buy [Game] Tickets" button
4. Complete purchase on Michigan Lottery website

### For Daily Games (Daily 3, Daily 4):
1. Generate suggestions in the app
2. Use the retailer finder link provided
3. Visit an authorized Michigan Lottery retailer
4. Purchase tickets with your suggested numbers

## ⚠️ Important Notes

- **Age Requirement**: Must be 18+ to purchase lottery tickets
- **Entertainment Only**: This tool is for educational/entertainment purposes
- **No Guarantees**: Lottery drawings are random; no system guarantees wins
- **Responsible Gaming**: Please gamble responsibly and within your means

---

**Need Help?** Check the comprehensive documentation in the `docs/` folder!

*Michigan Lottery Analyzer v2.3 - Ready to Win!* 🎲🎯