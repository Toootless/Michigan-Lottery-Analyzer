"""
Python Version and Compatibility Checker
Lottery Analyzer Project
"""

import sys
import platform
import subprocess

def check_python_version():
    """Check current Python version and compatibility"""
    print("=" * 60)
    print("🐍 PYTHON VERSION COMPATIBILITY CHECK")
    print("=" * 60)
    
    # Current Python version
    python_version = sys.version_info
    print(f"📋 Current Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    print(f"📋 Platform: {platform.system()} {platform.release()}")
    print(f"📋 Architecture: {platform.architecture()[0]}")
    
    # Check if version is compatible
    if python_version.major == 3 and python_version.minor == 11:
        print("✅ RECOMMENDED: Python 3.11 - Full compatibility expected")
        compatibility_score = "EXCELLENT"
    elif python_version.major == 3 and python_version.minor in [9, 10]:
        print("✅ GOOD: Python 3.9-3.10 - Good compatibility")
        compatibility_score = "GOOD"
    elif python_version.major == 3 and python_version.minor == 12:
        print("⚠️  CAUTION: Python 3.12 - Some library compatibility issues possible")
        compatibility_score = "CAUTION"
    elif python_version.major == 3 and python_version.minor >= 13:
        print("❌ WARNING: Python 3.13+ - Significant compatibility issues expected")
        compatibility_score = "POOR"
    else:
        print("❌ UNSUPPORTED: Python version not supported")
        compatibility_score = "UNSUPPORTED"
    
    print()
    
    # Check installed packages
    print("📦 CHECKING INSTALLED PACKAGES:")
    print("-" * 40)
    
    packages_to_check = [
        ("torch", "PyTorch - AI/ML Framework"),
        ("transformers", "Transformers - NLP Library"),
        ("streamlit", "Streamlit - Web Framework"),
        ("pandas", "Pandas - Data Analysis"),
        ("numpy", "NumPy - Numerical Computing")
    ]
    
    installed_packages = []
    missing_packages = []
    
    for package, description in packages_to_check:
        try:
            __import__(package)
            version = __import__(package).__version__
            print(f"✅ {package} {version} - {description}")
            installed_packages.append(package)
        except ImportError:
            print(f"❌ {package} - MISSING - {description}")
            missing_packages.append(package)
    
    print()
    
    # CUDA Check
    print("🎯 CUDA COMPATIBILITY CHECK:")
    print("-" * 40)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA Available: {torch.cuda.is_available()}")
            print(f"🎮 GPU Count: {torch.cuda.device_count()}")
            print(f"🚀 GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            cuda_status = "WORKING"
        else:
            print("⚠️  PyTorch installed but CUDA not available")
            cuda_status = "NO_CUDA"
    except ImportError:
        print("❌ PyTorch not installed")
        cuda_status = "NO_PYTORCH"
    
    # Check nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ NVIDIA GPU Driver detected via nvidia-smi")
        else:
            print("❌ nvidia-smi command failed")
    except Exception:
        print("❌ nvidia-smi not available")
    
    print()
    
    # Recommendations
    print("🎯 RECOMMENDATIONS:")
    print("-" * 40)
    
    if compatibility_score == "EXCELLENT":
        print("✅ Your Python 3.11 setup is optimal!")
        if cuda_status == "WORKING":
            print("🚀 Full GPU acceleration should work perfectly")
        elif missing_packages:
            print("📦 Install missing packages for full functionality:")
            for package in missing_packages:
                if package == "torch":
                    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
                else:
                    print(f"   pip install {package}")
    else:
        print("⚠️  Consider upgrading to Python 3.11 for optimal compatibility")
        print("📝 Follow the PYTHON_311_MIGRATION.md guide")
        print("🔧 Or run setup_python311.bat for automated setup")
    
    print()
    print("=" * 60)
    return {
        'python_version': f"{python_version.major}.{python_version.minor}.{python_version.micro}",
        'compatibility': compatibility_score,
        'cuda_status': cuda_status,
        'installed_packages': installed_packages,
        'missing_packages': missing_packages
    }

if __name__ == "__main__":
    result = check_python_version()
    
    # Summary
    print("📊 SUMMARY:")
    print(f"   Python: {result['python_version']} ({result['compatibility']})")
    print(f"   CUDA: {result['cuda_status']}")
    print(f"   Packages: {len(result['installed_packages'])}/{len(result['installed_packages']) + len(result['missing_packages'])} installed")
    print("=" * 60)