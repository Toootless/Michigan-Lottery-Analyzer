# CUDA 13.1 Setup Guide for RTX 3060 - December 2025 Edition

## Prerequisites
- NVIDIA RTX 3060 (12GB VRAM)
- Windows 10/11 (22H2 or later recommended)
- NVIDIA drivers 560+ (supports CUDA 13.1)
- Python 3.10-3.12 (3.14 not yet fully supported by PyTorch)

## System Information
- **GPU**: NVIDIA RTX 3060 (12GB GDDR6)
- **CUDA Compute Capability**: 8.6
- **Max CUDA Cores**: 3584
- **Memory Bandwidth**: 360 GB/s

## Step 1: Verify NVIDIA Drivers
```powershell
nvidia-smi
```
Expected output should show:
- CUDA Capability Major/Minor: 8.6
- Driver Version: 560.0 or higher
- CUDA Version: 13.1

If driver is outdated, download from: https://www.nvidia.com/Download/driverDetails.aspx/225369

## Step 2: Install CUDA Toolkit 13.1
1. Download: https://developer.nvidia.com/cuda-13-1-0-download-archive
2. Select: Windows → x86_64 → Windows 11 → exe (local)
3. Run installer and accept default path: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1`
4. Installation takes 5-10 minutes

## Step 3: Set Environment Variables
Add/Update these in Windows System Environment Variables:

```
CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
CUDA_PATH_V13_1=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
CUDACXX=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe
PATH (append): C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin
PATH (append): C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\libnvvp
CUDNN_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1 (if cuDNN installed)
```

**Note**: Restart terminal/IDE after setting environment variables

## Step 4: Install PyTorch with CUDA 13.1
```powershell
pip uninstall -y torch torchvision torchaudio

# For CUDA 13.1 (RTX 3060 native support)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu131 --no-cache-dir

# Alternative: PyTorch 2.2+ with CUDA 12.1 (also works with RTX 3060)
# pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
```

## Step 5: Install Additional CUDA Components (Optional but Recommended)
```powershell
# cuDNN (for optimized deep learning operations)
# Download from: https://developer.nvidia.com/cudnn-downloads
# Extract to: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1

# TensorRT (for optimized inference)
# pip install tensorrt-cu131

# Triton (for custom CUDA kernels in PyTorch)
# pip install triton>=2.1.0
```

## Step 6: Verify CUDA Installation
```powershell
# Check CUDA compilation
nvcc --version

# Check GPU availability
python -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0)}'); print(f'CUDA Version: {torch.version.cuda}')"

# Verify compute capability
python -c "import torch; print(f'Compute Capability: {torch.cuda.get_device_capability(0)}')"
```

Expected output:
```
PyTorch CUDA: True
GPU Name: NVIDIA RTX 3060
CUDA Version: 13.1
Compute Capability: (8, 6)
```

## Step 7: Test Performance
```powershell
python -c "
import torch
import time

# Create test tensor
x = torch.randn(1000, 1000, device='cuda')
y = torch.randn(1000, 1000, device='cuda')

# Warmup
for _ in range(10):
    z = torch.matmul(x, y)

# Benchmark
start = time.time()
for _ in range(100):
    z = torch.matmul(x, y)
torch.cuda.synchronize()
elapsed = time.time() - start

print(f'1000x1000 matmul x100: {elapsed:.4f}s')
print(f'Throughput: {100000 / elapsed:.0f} ops/s')
"
```

## Application-Specific Dependencies

### For MLA4 (Michigan Lottery Analyzer)
```powershell
pip install streamlit>=1.28.0
pip install pandas numpy scipy scikit-learn
pip install transformers>=4.35.0  # Requires PyTorch
pip install prophet>=1.1.5
pip install beautifulsoup4 lxml
pip install pillow pdf2image pytesseract
pip install selenium webdriver-manager
```

### For TensorFlow Users (Alternative to PyTorch)
```powershell
# TensorFlow with CUDA 13.1 support
pip install tensorflow[and-cuda]==2.15.0

# Note: TensorFlow 2.15+ supports CUDA 12.3+ but works optimally with CUDA 13.1
```

### For JAX Users (Alternative Framework)
```powershell
# JAX with CUDA 13.1
pip install jax[cuda131]==0.4.20

# Verify
python -c "import jax; print(jax.devices())"
```

## Environment Variable Configuration (PowerShell)
```powershell
# Add to PowerShell profile for persistence
$env:CUDA_HOME = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1'
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1'
$env:CUDACXX = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe'
$env:PATH += ';C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin'
$env:PATH += ';C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\libnvvp'

# Verify
Write-Host "CUDA_HOME: $env:CUDA_HOME"
nvcc --version
```

## Model Recommendations for RTX 3060 (12GB VRAM)

### Lightweight Models (2-4GB)
- `DistilBERT` - Fast NLP
- `GPT2` - Quick generation
- `MobileNet` - Efficient vision
- **Tokens/sec**: 100-200

### Standard Models (4-8GB)
- `BERT-large` - Advanced NLP
- `ResNet-50` - Vision tasks
- `Mistral-7B` (quantized) - LLM
- **Tokens/sec**: 50-100

### Large Models (8GB+)
- `Llama-2-7B` (8-bit) - Full LLM
- `ViT-large` - Vision transformer
- Requires 8-bit quantization
- **Tokens/sec**: 20-50

## Memory Optimization Flags
```python
import torch

# Enable memory efficient attention
torch.cuda.empty_cache()

# Use mixed precision (float16)
from torch.cuda.amp import autocast
with autocast():
    output = model(input)

# Gradient checkpointing
model.gradient_checkpointing_enable()

# 8-bit quantization
from bitsandbytes.nn import Linear8bitLt
model = model.to_8bit()

# Monitor memory
print(torch.cuda.memory_allocated() / 1024**3, 'GB used')
print(torch.cuda.memory_reserved() / 1024**3, 'GB reserved')
```

## Performance Optimization Tips
1. **Pin Memory**: Use `pin_memory=True` in DataLoaders
2. **Batch Size**: Use 64-128 for RTX 3060
3. **Mixed Precision**: FP16 can speed up 2-3x
4. **Gradient Accumulation**: Simulate larger batches
5. **NVIDIA CUDA Graphs**: Reduce CPU overhead

## Troubleshooting

### "CUDA out of memory"
```powershell
# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()"

# Monitor memory in real-time
Watch-Object { nvidia-smi } -Interval 1
```

### "cuda runtime error"
1. Update NVIDIA drivers: 560+
2. Restart Python kernel
3. Check for GPU memory leaks in code

### "CUDA libraries not found"
1. Verify environment variables (see Step 3)
2. Restart terminal/PowerShell
3. Check installation path exists

### "PyTorch still CPU-only"
```powershell
# Uninstall all torch packages
pip uninstall -y torch torchvision torchaudio torchtext

# Reinstall with CUDA 13.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu131 --no-cache-dir --force-reinstall
```

## Running MLA4 with CUDA
```powershell
# Verify setup
python -m streamlit run MLA4.py --server.address localhost --server.port 8520

# With CUDA memory monitoring
$env:CUDA_LAUNCH_BLOCKING=0  # Set to 1 for debugging
$env:CUDA_VISIBLE_DEVICES=0  # Use GPU 0 (if multiple GPUs)
python -m streamlit run MLA4.py --server.address localhost --server.port 8520
```

## System Specifications Summary
```
GPU: NVIDIA RTX 3060 (12GB GDDR6)
Compute Capability: 8.6
CUDA Toolkit: 13.1
cuDNN: 8.9.0+ (optional)
NVIDIA Driver: 560+
PyTorch: 2.1.2+ with CUDA 13.1
Python: 3.10-3.12
TensorRT: 8.6+ (optional)
```

## Additional Resources
- NVIDIA CUDA Documentation: https://docs.nvidia.com/cuda/
- PyTorch Official: https://pytorch.org/
- NVIDIA Developer: https://developer.nvidia.com/
- RTX 3060 Specs: https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/
- CUDA Compute Capability 8.6: https://docs.nvidia.com/cuda/archive/cuda-13-1-0/

---
**Last Updated**: December 11, 2025
**CUDA Version**: 13.1
**PyTorch Version**: 2.1.2+
**Python Versions Tested**: 3.10, 3.11, 3.12