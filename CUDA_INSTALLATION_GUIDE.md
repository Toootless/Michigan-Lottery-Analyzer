# NVIDIA CUDA Toolkit 13.1 Installation Guide - December 2025

## Overview
Your RTX 3060 GPU supports CUDA 13.1 (latest stable). This guide provides step-by-step installation for maximum performance and compatibility.

## Prerequisites
- NVIDIA RTX 3060 ✅
- Windows 11 (22H2) recommended or Windows 10
- NVIDIA drivers 560+ (latest recommended)
- Internet connection (~3-4 GB download)
- Admin privileges on computer

## System Compatibility Check
```powershell
# Run in PowerShell as Admin
nvidia-smi

# Expected output should show:
# NVIDIA-SMI 560.xx (driver version)
# RTX 3060, RTX 3060 Laptop, or similar
# CUDA Capability: 8.6
```

## Download CUDA Toolkit 13.1

### Direct Download Link
https://developer.nvidia.com/cuda-13-1-0-download-archive

### Installation Options:
1. **Local Installer (Recommended)**: Full offline setup (~3GB)
2. **Network Installer**: Downloads during installation (~200MB)
3. **Portable Archive**: Extract and use (advanced users)

### Step-by-Step Download:
1. Visit: https://developer.nvidia.com/cuda-13-1-0-download-archive
2. Select:
   - **Operating System**: Windows
   - **Architecture**: x86_64
   - **Version**: Windows 11 (or Windows 10)
   - **Installer Type**: exe (local)
3. Click "Download" (~3.2GB file)

## Installation Steps

### Step 1: Verify Prerequisites
```powershell
# Check GPU drivers
nvidia-smi

# Check available disk space (need ~8GB free)
Get-Volume C: | Select-Object SizeRemaining

# Check Windows version
[System.Environment]::OSVersion.VersionString
```

### Step 2: Disable Conflicting Software
Before installation, temporarily disable:
- Antivirus (Windows Defender can interfere)
- VPN connections
- Third-party GPU monitor software
- Any NVIDIA GeForce Experience auto-updates

### Step 3: Run CUDA Installer
1. Double-click the downloaded `.exe` file
2. **If prompted**: "The following packages will be uninstalled" - click OK
3. Select **"Express Setup"** (recommended for most users)
4. **IMPORTANT**: Ensure these are checked:
   - ☑️ CUDA Toolkit
   - ☑️ cuDNN (if available - for deep learning)
   - ☑️ NVIDIA Nsight Systems (optional profiling)
5. Accept default installation path: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1`
6. Click "Install" and wait (5-15 minutes depending on speed)
7. Restart your computer when prompted

### Step 4: Verify Installation
```powershell
# After restart, open new PowerShell window and check:
nvcc --version

# Expected output:
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2024 NVIDIA Corporation
# Built on xxx
# Cuda compilation tools, release 13.1, Vxxx
# Build xxxxx

# Also verify GPU detection:
nvidia-smi

# Expected output shows your RTX 3060
```

### Step 5: Set Environment Variables (Critical!)
Open Windows System Environment Variables (or PowerShell as Admin):

#### GUI Method:
1. Right-click "This PC" → Properties
2. Click "Advanced system settings"
3. Click "Environment Variables" button
4. Click "New" under "System variables"
5. Add these variables:
   - **CUDA_HOME**: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1`
   - **CUDA_PATH**: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1`
   - **CUDA_PATH_V13_1**: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1`
   - **CUDACXX**: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe`

6. Edit **PATH** variable and add:
   - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin`
   - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\libnvvp`
   - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\include`

#### PowerShell Method:
```powershell
# Run as Administrator
$cudaPath = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1'

# Set environment variables
[System.Environment]::SetEnvironmentVariable('CUDA_HOME', $cudaPath, [System.EnvironmentVariableTarget]::Machine)
[System.Environment]::SetEnvironmentVariable('CUDA_PATH', $cudaPath, [System.EnvironmentVariableTarget]::Machine)
[System.Environment]::SetEnvironmentVariable('CUDACXX', "$cudaPath\bin\nvcc.exe", [System.EnvironmentVariableTarget]::Machine)

# Verify
$env:CUDA_HOME
$env:CUDA_PATH
```

### Step 6: Reinstall PyTorch with CUDA 13.1
```powershell
# Uninstall existing PyTorch
pip uninstall -y torch torchvision torchaudio torchtext

# Clear pip cache
pip cache purge

# Install PyTorch with CUDA 13.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu131 --no-cache-dir

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU: {torch.cuda.is_available()}')"

# Expected output:
# PyTorch: 2.1.2+cu131
# CUDA: 13.1
# GPU: True
```

### Step 7: Verify GPU Access
```powershell
python << 'EOF'
import torch

print("=" * 50)
print("CUDA VERIFICATION")
print("=" * 50)

# Check CUDA availability
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"PyTorch Version: {torch.__version__}")

# Check GPU info
if torch.cuda.is_available():
    print(f"\nGPU Information:")
    print(f"  Device Name: {torch.cuda.get_device_name(0)}")
    print(f"  Compute Capability: {torch.cuda.get_device_capability(0)}")
    print(f"  Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"  Current Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    # Test tensor operation
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')
    z = torch.matmul(x, y)
    print(f"\n✅ GPU Tensor Operations: SUCCESS")
else:
    print("❌ CUDA not available - check installation")

print("=" * 50)
EOF
```

### Step 8: Install Additional Tools (Optional)

#### cuDNN (Deep Learning Acceleration)
```powershell
# Download from: https://developer.nvidia.com/cudnn-downloads
# Requires NVIDIA Developer Account (free)

# After download, extract to:
# C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1

# Verify
python -c "import torch; print(f'cuDNN: {torch.backends.cudnn.enabled}')"
```

#### TensorRT (Inference Optimization)
```powershell
pip install tensorrt-cu131

# Verify
python -c "import tensorrt as trt; print(f'TensorRT: {trt.__version__}')"
```

#### NVIDIA Nsight Systems (Profiling)
- Installed with CUDA Toolkit
- Use to profile GPU performance
- Launch: `nsys` command

## Troubleshooting

### Issue: "nvcc: not found" after installation
**Solution:**
1. Verify CUDA path: `Test-Path 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe'`
2. Restart PowerShell/Command Prompt
3. Manually add to PATH if needed

### Issue: PyTorch still shows CUDA unavailable
**Solution:**
```powershell
# Complete reinstall
pip uninstall -y torch torchvision torchaudio
pip cache purge

# Install with specific Python version
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu131 --force-reinstall --no-cache-dir

# Check Python/CUDA compatibility
python -c "import sys; print(f'Python {sys.version_info.major}.{sys.version_info.minor}')"
```

### Issue: CUDA out of memory during operation
**Solution:**
```powershell
# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()"

# Monitor memory usage
nvidia-smi -l 1  # Update every 1 second
```

### Issue: Installation hangs or fails
**Solution:**
1. Disable antivirus temporarily
2. Run installer as Administrator
3. Delete `%temp%\NVIDIA` folder if it exists
4. Restart computer and try again

## Performance Benchmarking

```powershell
python << 'EOF'
import torch
import time

print("CUDA Performance Test (RTX 3060)")
print("=" * 50)

# Warmup
x = torch.randn(1024, 1024, device='cuda')
y = torch.randn(1024, 1024, device='cuda')
for _ in range(5):
    z = torch.matmul(x, y)

# Benchmark
sizes = [512, 1024, 2048, 4096]
for size in sizes:
    x = torch.randn(size, size, device='cuda')
    y = torch.randn(size, size, device='cuda')
    
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(100):
        z = torch.matmul(x, y)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    gflops = (2 * size**3 * 100) / (elapsed * 1e9)
    print(f"{size}x{size}: {elapsed:.4f}s ({gflops:.1f} GFLOPS)")

print("=" * 50)
EOF
```

## Running MLA4 with CUDA

```powershell
# Set GPU to use (0 for single GPU systems)
$env:CUDA_VISIBLE_DEVICES = "0"

# Optional: Enable CUDA launch blocking for debugging
# $env:CUDA_LAUNCH_BLOCKING = "1"

# Start application
python -m streamlit run MLA4.py --server.address localhost --server.port 8520

# Monitor GPU usage in another terminal
nvidia-smi -l 1
```

## System Requirements Summary

| Component | Requirement | Status |
|-----------|-------------|--------|
| GPU | NVIDIA RTX 3060 | ✅ |
| VRAM | 12 GB GDDR6 | ✅ |
| Compute Capability | 8.6+ | ✅ |
| Driver Version | 560+ | ✅ |
| CUDA Toolkit | 13.1 | ✅ |
| cuDNN | 8.9.0+ (optional) | Optional |
| Python | 3.10-3.12 | ✅ |
| PyTorch | 2.1.2+ with CUDA 13.1 | Recommended |
| Disk Space | 8 GB free | Required |

## Next Steps
1. ✅ Install CUDA Toolkit 13.1
2. ✅ Install PyTorch with CUDA support
3. ✅ Verify GPU access works
4. Run MLA4: `python -m streamlit run MLA4.py --server.address localhost --server.port 8520`

## Additional Resources
- NVIDIA CUDA 13.1 Docs: https://docs.nvidia.com/cuda/cuda-13-1-0/
- PyTorch Installation: https://pytorch.org/get-started/locally/
- RTX 3060 Specifications: https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/
- NVIDIA Developer: https://developer.nvidia.com/

---
**Last Updated**: December 11, 2025
**CUDA Version**: 13.1 (Latest Stable)
**PyTorch Version**: 2.1.2+ with CUDA 13.1 support
**Driver Version**: 560+

## Support
For detailed NVIDIA CUDA Toolkit documentation:
https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/
