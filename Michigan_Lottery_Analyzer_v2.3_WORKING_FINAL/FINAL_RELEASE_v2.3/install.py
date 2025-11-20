#!/usr/bin/env python3
"""
Michigan Lottery Analyzer v2.3 - Installation Script
Automated setup and configuration for the lottery analysis application
"""

import os
import sys
import subprocess
import platform
import json
from pathlib import Path

class LotteryAnalyzerInstaller:
    def __init__(self):
        self.python_version = sys.version_info
        self.platform = platform.system()
        self.current_dir = Path.cwd()
        self.requirements = [
            'streamlit>=1.25.0',
            'pandas>=1.5.0',
            'numpy>=1.21.0',
            'requests>=2.28.0',
            'beautifulsoup4>=4.11.0',
            'matplotlib>=3.5.0',
            'seaborn>=0.11.0',
            'plotly>=5.10.0'
        ]
        self.optional_requirements = [
            'pytesseract>=0.3.10',
            'pillow>=9.2.0',
            'pdf2image>=1.16.0',
            'pymupdf>=1.20.0',
            'transformers>=4.21.0',
            'torch>=1.12.0'
        ]
    
    def check_python_version(self):
        """Check if Python version meets requirements"""
        print("🐍 Checking Python version...")
        if self.python_version < (3, 8):
            print("❌ Python 3.8+ is required. Current version:", 
                  f"{self.python_version.major}.{self.python_version.minor}")
            return False
        elif self.python_version >= (3, 14):
            print("⚠️  Python 3.14+ detected. Some features may have compatibility issues.")
            print("💡 Python 3.11 is recommended for optimal performance.")
        else:
            print(f"✅ Python {self.python_version.major}.{self.python_version.minor} detected - Compatible!")
        return True
    
    def create_virtual_environment(self):
        """Create virtual environment for the application"""
        print("\n🏗️  Creating virtual environment...")
        venv_path = self.current_dir / "lottery_env"
        
        try:
            subprocess.run([sys.executable, '-m', 'venv', str(venv_path)], 
                          check=True, capture_output=True)
            print("✅ Virtual environment created successfully!")
            return venv_path
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
            return None
    
    def install_requirements(self, venv_path=None):
        """Install required packages"""
        print("\n📦 Installing required packages...")
        
        # Determine pip command
        if venv_path:
            if self.platform == "Windows":
                pip_cmd = str(venv_path / "Scripts" / "pip.exe")
            else:
                pip_cmd = str(venv_path / "bin" / "pip")
        else:
            pip_cmd = "pip"
        
        # Install core requirements
        for package in self.requirements:
            try:
                print(f"Installing {package}...")
                subprocess.run([pip_cmd, 'install', package], 
                              check=True, capture_output=True)
                print(f"✅ {package} installed successfully!")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install {package}: {e}")
                return False
        
        # Install optional requirements (with error handling)
        print("\n📦 Installing optional packages...")
        for package in self.optional_requirements:
            try:
                print(f"Installing {package} (optional)...")
                subprocess.run([pip_cmd, 'install', package], 
                              check=True, capture_output=True)
                print(f"✅ {package} installed successfully!")
            except subprocess.CalledProcessError as e:
                print(f"⚠️  Optional package {package} failed to install: {e}")
                print("   (This won't affect core functionality)")
        
        return True
    
    def verify_data_files(self):
        """Verify required data files exist"""
        print("\n📊 Verifying data files...")
        
        data_dir = self.current_dir / "data" / "final_integrated_data"
        required_files = [
            "Powerball_Complete.csv",
            "Mega_Millions_Complete.csv", 
            "Fantasy_5_Complete.csv",
            "Daily_3_Evening_Complete.csv",
            "Daily_3_Midday_Complete.csv",
            "Daily_4_Evening_Complete.csv",
            "Daily_4_Midday_Complete.csv",
            "Lucky_for_Life_Complete.csv",
            "Lotto_47_Complete.csv"
        ]
        
        missing_files = []
        for file_name in required_files:
            file_path = data_dir / file_name
            if file_path.exists():
                print(f"✅ {file_name} found")
            else:
                print(f"❌ {file_name} missing")
                missing_files.append(file_name)
        
        if missing_files:
            print(f"\n⚠️  {len(missing_files)} data files are missing.")
            print("   The application will use web scraping as fallback.")
        else:
            print("\n✅ All data files verified!")
        
        return len(missing_files) == 0
    
    def create_directories(self):
        """Create necessary directories"""
        print("\n📁 Creating directory structure...")
        
        directories = [
            "logs",
            "data/generated_suggestions", 
            "data/predictions",
            "data/backup"
        ]
        
        for dir_name in directories:
            dir_path = self.current_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {dir_name}")
    
    def test_installation(self, venv_path=None):
        """Test the installation by importing key modules"""
        print("\n🧪 Testing installation...")
        
        # Determine python command
        if venv_path:
            if self.platform == "Windows":
                python_cmd = str(venv_path / "Scripts" / "python.exe")
            else:
                python_cmd = str(venv_path / "bin" / "python")
        else:
            python_cmd = sys.executable
        
        test_script = '''
import streamlit
import pandas
import numpy
import requests
import matplotlib
import plotly
print("✅ All core modules imported successfully!")
'''
        
        try:
            result = subprocess.run([python_cmd, '-c', test_script], 
                                  check=True, capture_output=True, text=True)
            print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Import test failed: {e}")
            return False
    
    def generate_run_script(self, venv_path=None):
        """Generate convenient run script"""
        print("\n📝 Creating run script...")
        
        if self.platform == "Windows":
            script_name = "run_lottery_analyzer.bat"
            if venv_path:
                python_path = str(venv_path / "Scripts" / "python.exe")
            else:
                python_path = "python"
            
            script_content = f'''@echo off
echo Starting Michigan Lottery Analyzer v2.3...
echo.
"{python_path}" -m streamlit run src/MichiganLotteryAnalyzer.py
pause
'''
        else:
            script_name = "run_lottery_analyzer.sh"
            if venv_path:
                python_path = str(venv_path / "bin" / "python")
            else:
                python_path = "python3"
            
            script_content = f'''#!/bin/bash
echo "Starting Michigan Lottery Analyzer v2.3..."
echo
{python_path} -m streamlit run src/MichiganLotteryAnalyzer.py
'''
        
        script_path = self.current_dir / script_name
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        if self.platform != "Windows":
            os.chmod(script_path, 0o755)
        
        print(f"✅ Run script created: {script_name}")
        return script_path
    
    def print_summary(self, venv_path=None, script_path=None):
        """Print installation summary"""
        print("\n" + "="*60)
        print("🎉 MICHIGAN LOTTERY ANALYZER v2.3 INSTALLATION COMPLETE!")
        print("="*60)
        
        print("\n📋 INSTALLATION SUMMARY:")
        print(f"   • Python Version: {self.python_version.major}.{self.python_version.minor}")
        print(f"   • Platform: {self.platform}")
        print(f"   • Installation Directory: {self.current_dir}")
        
        if venv_path:
            print(f"   • Virtual Environment: {venv_path}")
        
        print("\n🚀 HOW TO RUN:")
        if script_path:
            if self.platform == "Windows":
                print(f"   Double-click: {script_path.name}")
            else:
                print(f"   Run: ./{script_path.name}")
        
        print("\n   Or manually:")
        if venv_path:
            if self.platform == "Windows":
                print(f"   {venv_path}\\Scripts\\python.exe -m streamlit run src/MichiganLotteryAnalyzer.py")
            else:
                print(f"   {venv_path}/bin/python -m streamlit run src/MichiganLotteryAnalyzer.py")
        else:
            print("   python -m streamlit run src/MichiganLotteryAnalyzer.py")
        
        print("\n🌐 ACCESS:")
        print("   Local URL: http://localhost:8501")
        print("   Network URL: http://[your-ip]:8501")
        
        print("\n✨ KEY FEATURES:")
        print("   • 🎯 Smart Number Suggestions with AI Learning")
        print("   • 🎫 Purchase Integration (Online & Retailer)")
        print("   • 📊 Advanced Statistical Analysis")
        print("   • 🌐 Real-time Web Data Integration")
        print("   • 📈 Interactive Data Visualization")
        print("   • 🤖 Machine Learning Improvements")
        
        print("\n📚 DOCUMENTATION:")
        print("   • README.md - Project overview")
        print("   • docs/USER_GUIDE.md - Detailed usage instructions")
        print("   • docs/TECHNICAL_DOCS.md - Development documentation")
        print("   • docs/CHANGELOG_v2.3.md - Version history")
        
        print("\n⚠️  IMPORTANT NOTES:")
        print("   • This is for entertainment/educational purposes only")
        print("   • Must be 18+ to purchase lottery tickets")
        print("   • Please gamble responsibly")
        
        print("\nReady to analyze lottery numbers! 🎲🎯")
        print("="*60)
    
    def run(self):
        """Run the complete installation process"""
        print("🎲 MICHIGAN LOTTERY ANALYZER v2.3 - INSTALLER")
        print("="*50)
        
        # Check Python version
        if not self.check_python_version():
            return False
        
        # Ask user about virtual environment
        use_venv = input("\n🤔 Create virtual environment? (recommended) [Y/n]: ").lower()
        venv_path = None
        
        if use_venv != 'n':
            venv_path = self.create_virtual_environment()
            if not venv_path:
                print("Continuing without virtual environment...")
        
        # Install requirements
        if not self.install_requirements(venv_path):
            print("❌ Installation failed!")
            return False
        
        # Create directories
        self.create_directories()
        
        # Verify data files
        self.verify_data_files()
        
        # Test installation
        if not self.test_installation(venv_path):
            print("⚠️  Installation test failed, but continuing...")
        
        # Generate run script
        script_path = self.generate_run_script(venv_path)
        
        # Print summary
        self.print_summary(venv_path, script_path)
        
        return True

if __name__ == "__main__":
    installer = LotteryAnalyzerInstaller()
    success = installer.run()
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)