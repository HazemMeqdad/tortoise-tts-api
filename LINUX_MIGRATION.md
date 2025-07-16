# Tortoise TTS Linux Migration Guide

This guide helps you migrate your Tortoise TTS setup from Windows to Linux for optimal performance with DeepSpeed.

## 🚀 Performance Expectations

### Windows (Current)
- **Generation Time**: 1500-2000ms 
- **Real-time Target**: ❌ (>500ms)
- **DeepSpeed**: ❌ Disabled (compatibility issues)

### Linux (Expected)
- **Generation Time**: 200-500ms (3-10x faster)
- **Real-time Target**: ✅ (<500ms)
- **DeepSpeed**: ✅ Enabled (10x speedup)

## 📋 Prerequisites

### Linux System Requirements
- Ubuntu 20.04+ / CentOS 8+ / Similar Linux distribution
- Python 3.8+
- NVIDIA GPU with CUDA 11.8+
- 8GB+ RAM
- 20GB+ free disk space

### GPU Requirements
- RTX 4060 (your current GPU) ✅
- CUDA Compute Capability 7.0+
- 8GB+ VRAM

## 🔧 Installation Steps

### 1. Clone/Copy Your Project
```bash
# Copy your project files to Linux
scp -r tortoise-tts/ user@linux-server:/home/user/
```

### 2. Run the Linux Setup Script
```bash
cd tortoise-tts
chmod +x setup_linux.sh
./setup_linux.sh
```

### 3. Manual Installation (Alternative)
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install DeepSpeed
pip install deepspeed

# Install API requirements
pip install -r requirements_api.txt

# Install Tortoise TTS
pip install -e .
```

## 🚀 Running the API

### Option 1: Quick Start
```bash
python start_api.py
```

### Option 2: Manual Start
```bash
source venv/bin/activate
uvicorn api:app --host 0.0.0.0 --port 8888 --reload
```

## 🧪 Testing Performance

### 1. Run Tests
```bash
python test_optimization.py
```

### 2. Test API Endpoint
```bash
curl -X POST "http://localhost:8888/synthesize" \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world","voice":"random","preset":"ultra_realtime"}' \
  --output test.wav
```

### 3. Check Performance Headers
```bash
curl -X POST "http://localhost:8888/synthesize" \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world","voice":"random","preset":"ultra_realtime"}' \
  -I
```

Expected headers:
- `X-Generation-Time-Ms`: Should be <500ms
- `X-Is-Realtime`: Should be "True"
- `X-RTF`: Should be <1.0

## 🔍 Troubleshooting

### DeepSpeed Installation Issues
```bash
# Install build dependencies
sudo apt-get update
sudo apt-get install build-essential ninja-build

# Reinstall DeepSpeed
pip uninstall deepspeed
pip install deepspeed --no-cache-dir
```

### CUDA Issues
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# If CUDA is missing, install:
# https://developer.nvidia.com/cuda-downloads
```

### Performance Not Improving
1. Verify GPU is being used: Check logs for "Using device: cuda"
2. Verify DeepSpeed is enabled: Check `/health` endpoint
3. Monitor GPU usage: `nvidia-smi` while generating

## 📊 Performance Comparison

| Metric | Windows | Linux |
|--------|---------|--------|
| Generation Time | 1500-2000ms | 200-500ms |
| Real-time Capable | ❌ | ✅ |
| DeepSpeed | ❌ | ✅ |
| RTF | 0.9-2.0x | 0.2-0.5x |

## 🎯 Expected Results

With DeepSpeed enabled on Linux, you should see:
- **Generation times**: 200-500ms (real-time!)
- **RTF**: 0.2-0.5x (much faster than real-time)
- **Consistent performance**: Less variation between runs

## 🆘 Support

If you encounter issues:
1. Check the health endpoint: `curl http://localhost:8888/health`
2. Review the logs during startup
3. Run the test script: `python test_optimization.py`
4. Verify all optimizations are enabled in the health response

## 🎉 Success Indicators

You've successfully migrated when:
- ✅ Health endpoint shows `"use_deepspeed": true`
- ✅ Generation time consistently <500ms
- ✅ API responds with `"X-Is-Realtime": "True"`
- ✅ RTF values are <1.0

Your RTX 4060 should now be capable of true real-time voice synthesis!
