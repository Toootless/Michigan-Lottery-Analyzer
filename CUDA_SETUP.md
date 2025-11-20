# Local CUDA Setup Guide for RTX 3060

## Prerequisites
- NVIDIA RTX 3060 (12GB VRAM)
- Windows 10/11
- NVIDIA drivers (latest)
- Python 3.8+

## Step 1: Install CUDA Toolkit
1. Download CUDA 11.8 or 12.1 from NVIDIA website
2. Install following the installer instructions
3. Verify installation: `nvcc --version`

## Step 2: Install PyTorch with CUDA
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Step 3: Install Dependencies
```bash
pip install -r requirements_local.txt
```

## Step 4: Verify CUDA Setup
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Step 5: Run the Application
```bash
streamlit run MichiganLotteryAnalyzer.py
```

## Alternative: Ollama Setup (Easier)
1. Download Ollama from https://ollama.ai/
2. Install and run: `ollama serve`
3. Pull a model: `ollama pull llama2`
4. Run the application - it will automatically detect Ollama

## Model Recommendations for RTX 3060 (12GB VRAM)

### Small Models (2-4GB VRAM)
- `distilgpt2` - Very fast, basic capabilities
- `microsoft/DialoGPT-medium` - Good for conversation

### Medium Models (4-8GB VRAM)  
- `microsoft/DialoGPT-large` - Better responses
- `gpt2` - Classic GPT-2

### Large Models (8GB+ VRAM)
- Requires quantization (8-bit or 4-bit)
- Consider using Ollama with quantized models

## Memory Optimization Tips
1. Use `torch.float16` instead of `torch.float32`
2. Enable gradient checkpointing
3. Use smaller batch sizes
4. Consider 8-bit quantization with bitsandbytes

## Troubleshooting

### CUDA Out of Memory
- Use smaller models
- Reduce max_length parameter
- Enable 8-bit quantization
- Close other GPU applications

### Slow Performance
- Verify CUDA is being used: check device in logs
- Update GPU drivers
- Ensure adequate cooling

### Model Loading Issues
- Check internet connection for model download
- Verify sufficient disk space
- Try alternative models

## Performance Expectations on RTX 3060
- Small models: ~50-100 tokens/second
- Medium models: ~20-50 tokens/second  
- Large models (quantized): ~10-20 tokens/second

## Supported Models
The application will automatically select appropriate models based on available VRAM and performance requirements.