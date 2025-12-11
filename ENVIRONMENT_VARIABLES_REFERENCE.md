# Environment Variables Reference - CUDA 13.1 & Dependencies

## NVIDIA CUDA & GPU Configuration

### Core CUDA Variables
```
CUDA_HOME = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
CUDA_PATH = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
CUDA_PATH_V13_1 = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
CUDACXX = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe
CUDNN_PATH = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
```

### GPU Execution Control
```
CUDA_VISIBLE_DEVICES = 0                    # Which GPU to use (0 for first GPU)
CUDA_LAUNCH_BLOCKING = 0                    # 0=async (faster), 1=sync (debug)
CUDA_DEVICE_ORDER = PCI_BUS_ID              # GPU ordering by PCI ID
CUDA_FORCE_PTX_JIT = 1                      # Force PTX JIT compilation
```

### Memory Management
```
CUDA_MANAGED_FORCE_UNIFIED_MEMORY = 0       # Unified memory for RTX 3060
CUDA_DEVICE_MAX_CONNECTIONS = 32            # Max concurrent kernels
CUDNN_BATCHNORM_RUNNING_AVERAGE_DECAY = 0.9  # BatchNorm decay rate
```

### Performance Tuning
```
TF_CPP_MIN_LOG_LEVEL = 2                    # Suppress TensorFlow verbose output (0-3)
CUDA_CACHE_DISABLE = 0                      # Don't disable kernel cache
NCCL_DEBUG = INFO                           # NCCL logging level
CUBLAS_WORKSPACE_CONFIG = :4294967296       # cuBLAS workspace (4GB)
```

## Python & PyTorch Environment

### Path Configuration
```
PATH (add) = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin
PATH (add) = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\libnvvp
PATH (add) = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\include
PYTHONPATH = (auto-set by Python)
```

### PyTorch Specific
```
PYTORCH_CUDA_ALLOC_CONF = max_split_size_mb=3000  # Prevent fragmentation
PYTORCH_ENABLE_MPS_FALLBACK = 1              # Enable fallback to CPU if needed
TORCH_HOME = %USERPROFILE%\.cache\torch      # Model cache location
TORCH_CUDNN_ENABLED = 1                      # Enable cuDNN optimization
```

### Development & Debugging
```
TORCH_SHOW_CPP_STACKTRACES = 1              # Show C++ errors
PYTORCH_DEBUG = 1                           # Enable PyTorch debug mode
PYTHONDONTWRITEBYTECODE = 1                # Don't create .pyc files
PYTHONUTF8 = 1                              # Use UTF-8 encoding
```

## Deep Learning Framework Configuration

### TensorFlow (if using)
```
TF_CPP_MIN_LOG_LEVEL = 2                    # 0=all, 1=info, 2=warning, 3=error
TF_FORCE_GPU_ALLOW_GROWTH = true            # Gradual GPU memory allocation
TF_GPU_THREAD_MODE = gpu_private            # GPU thread affinity
TF_CUDA_VISIBLE_DEVICES = 0                 # GPU device selection
```

### JAX (if using)
```
JAX_PLATFORMS = cuda                         # Use CUDA backend
XLA_FLAGS = --xla_gpu_cuda_data_dir=/path/to/cuda  # CUDA path for JAX
JAX_FORCE_MULTIPLE_COMPUTE_CAPABILITY = 8.6 # Force compute capability
```

### Transformers & Hugging Face
```
HF_HOME = %USERPROFILE%\.cache\huggingface  # Model cache location
HF_DATASETS_CACHE = %USERPROFILE%\.cache\huggingface\datasets
TRANSFORMERS_CACHE = %USERPROFILE%\.cache\huggingface
```

## Application-Specific (MLA4)

### Streamlit Configuration
```
STREAMLIT_THEME_PRIMARYCOLOR = #0084ff
STREAMLIT_THEME_BACKGROUNDCOLOR = #ffffff
STREAMLIT_LOGGER_LEVEL = info              # info, debug, warning, error
STREAMLIT_CLIENT_LOGGER_LEVEL = info
```

### Lottery Analyzer (MLA4)
```
MLA4_CACHE_SIZE = 1024                      # Cache size in MB
MLA4_GPU_ENABLED = 1                        # Enable GPU (1=yes, 0=no)
MLA4_BATCH_SIZE = 64                        # Default batch size for analysis
MLA4_MAX_WORKERS = 4                        # Number of processing workers
```

## System & Windows Configuration

### Windows Runtime
```
TEMP = %USERPROFILE%\AppData\Local\Temp
TMP = %USERPROFILE%\AppData\Local\Temp
TMPDIR = %USERPROFILE%\AppData\Local\Temp
USERPROFILE = C:\Users\<YourUsername>
HOMEPATH = \Users\<YourUsername>
```

### Developer Tools
```
VCINSTALLDIR = C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\
WindowsSDKVersion = 10.0.22621.0
INCLUDE = (Visual Studio include paths)
LIB = (Visual Studio library paths)
```

## How to Set Environment Variables

### Windows GUI Method
1. Right-click "This PC" → Properties
2. Click "Advanced system settings"
3. Click "Environment Variables"
4. Add/Edit variables in "System variables" section
5. Restart terminal/IDE after changes

### PowerShell Method
```powershell
# For current session only
$env:CUDA_HOME = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1'

# Permanently (requires Admin)
[System.Environment]::SetEnvironmentVariable('CUDA_HOME', 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1', [System.EnvironmentVariableTarget]::Machine)

# Verify
Write-Host $env:CUDA_HOME
```

### Command Prompt Method
```batch
# For current session only
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1

# Permanently (requires Admin)
setx CUDA_HOME "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1" /M
```

### Python Script
```python
import os

# Set for current process
os.environ['CUDA_HOME'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Check
print(f"CUDA_HOME: {os.environ.get('CUDA_HOME')}")
```

## Verification Script

```powershell
# SaveAs: verify_env.ps1
Write-Host "=== CUDA & GPU Environment Verification ===" -ForegroundColor Cyan

Write-Host "`n1. CUDA Paths:"
Write-Host "CUDA_HOME: $env:CUDA_HOME"
Write-Host "CUDA_PATH: $env:CUDA_PATH"

Write-Host "`n2. GPU Status:"
nvidia-smi -L

Write-Host "`n3. CUDA Compiler:"
nvcc --version

Write-Host "`n4. Python CUDA Access:"
python -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

Write-Host "`n5. Environment Variables Summary:"
$cudaVars = @('CUDA_HOME', 'CUDA_PATH', 'CUDA_VISIBLE_DEVICES', 'TORCH_HOME', 'HF_HOME')
foreach ($var in $cudaVars) {
    $value = [System.Environment]::GetEnvironmentVariable($var, [System.EnvironmentVariableTarget]::Machine)
    if ($value) {
        Write-Host "$var = $value"
    }
}

Write-Host "`n✅ Verification Complete" -ForegroundColor Green
```

## Troubleshooting Environment Variables

### Issue: Changes not taking effect
**Solution:**
- Close and reopen PowerShell/CMD
- Restart IDE (VSCode, PyCharm, etc.)
- Restart computer if still not working

### Issue: "CUDA not found" despite PATH updates
**Solution:**
```powershell
# Verify path includes CUDA
$env:PATH -split ';' | Where-Object { $_ -like '*CUDA*' }

# If empty, manually add:
$cudaPath = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin'
$env:PATH = $env:PATH + ";$cudaPath"
```

### Issue: Multiple CUDA versions installed
**Solution:**
```powershell
# Check all CUDA installations
Get-ChildItem -Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\*" -Directory | ForEach-Object { $_.Name }

# Set CUDA_HOME to desired version (13.1 recommended)
[System.Environment]::SetEnvironmentVariable('CUDA_HOME', 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1', [System.EnvironmentVariableTarget]::Machine)
```

## Performance Tuning Recommendations

### For RTX 3060 (Gaming/ML Hybrid)
```
CUDA_VISIBLE_DEVICES = 0
CUDA_LAUNCH_BLOCKING = 0
CUBLAS_WORKSPACE_CONFIG = :16:8  # Smaller workspace
TF_CPP_MIN_LOG_LEVEL = 2
```

### For Data Science (High Performance)
```
CUDA_VISIBLE_DEVICES = 0
CUDA_LAUNCH_BLOCKING = 0
CUBLAS_WORKSPACE_CONFIG = :4294967296  # 4GB workspace
PYTORCH_CUDA_ALLOC_CONF = max_split_size_mb=3000
TF_FORCE_GPU_ALLOW_GROWTH = true
```

### For Debugging
```
CUDA_VISIBLE_DEVICES = 0
CUDA_LAUNCH_BLOCKING = 1
CUDA_DEVICE_ORDER = PCI_BUS_ID
TORCH_SHOW_CPP_STACKTRACES = 1
PYTORCH_DEBUG = 1
TF_CPP_MIN_LOG_LEVEL = 0
```

## Compatibility Matrix

| Framework | CUDA 13.1 | Status |
|-----------|-----------|--------|
| PyTorch 2.1.2+ | ✅ | Full Support |
| TensorFlow 2.15+ | ✅ | Full Support |
| JAX 0.4.20+ | ✅ | Full Support |
| ONNX Runtime | ✅ | Full Support |
| OpenCV (cuda) | ✅ | Full Support |
| Triton | ✅ | Full Support |

## Additional Resources
- NVIDIA CUDA Documentation: https://docs.nvidia.com/cuda/
- Environment Variable Reference: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars
- PyTorch CUDA: https://pytorch.org/docs/stable/notes/cuda.html
- TensorFlow CUDA: https://www.tensorflow.org/install/gpu

---
**Last Updated**: December 11, 2025
**CUDA Version**: 13.1
**Compatible OS**: Windows 10/11
**GPU Support**: RTX 3060 and newer
