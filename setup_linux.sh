#!/bin/bash
# Linux setup script for Tortoise TTS with DeepSpeed optimization

echo "🐧 Setting up Tortoise TTS for Linux with DeepSpeed..."
echo "=================================================="

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "❌ This script is designed for Linux systems only."
    exit 1
fi

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️ NVIDIA GPU not detected. DeepSpeed will be disabled."
    USE_DEEPSPEED=false
else
    echo "✅ NVIDIA GPU detected."
    nvidia-smi
    USE_DEEPSPEED=true
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🔨 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install basic requirements
echo "📦 Installing basic requirements..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install fastapi uvicorn
pip install scipy numpy
pip install requests

# Install DeepSpeed if GPU is available
if [ "$USE_DEEPSPEED" = true ]; then
    echo "🚀 Installing DeepSpeed for maximum performance..."
    pip install deepspeed
    
    # Test DeepSpeed installation
    python -c "import deepspeed; print('✅ DeepSpeed installed successfully')" || {
        echo "❌ DeepSpeed installation failed. Continuing without DeepSpeed..."
    }
fi

# Install Tortoise TTS
echo "🎤 Installing Tortoise TTS..."
pip install -e .

echo ""
echo "🎉 Setup complete!"
echo ""
echo "🚀 To start the optimized API server:"
echo "   source venv/bin/activate"
echo "   uvicorn api:app --host 0.0.0.0 --port 8888 --reload"
echo ""
echo "🧪 To test the setup:"
echo "   python test_optimization.py"
echo ""
echo "🔧 To test with curl:"
echo "   curl -X POST \"http://localhost:8888/synthesize\" \\"
echo "     -H \"Content-Type: application/json\" \\"
echo "     -d '{\"text\":\"Hello world\",\"voice\":\"random\",\"preset\":\"ultra_realtime\"}' \\"
echo "     --output test.wav"
echo ""
if [ "$USE_DEEPSPEED" = true ]; then
    echo "⚡ DeepSpeed is enabled for maximum performance!"
    echo "   Expected performance: <500ms generation time"
else
    echo "⚠️ DeepSpeed is disabled (no GPU detected)"
    echo "   Expected performance: 1-2s generation time"
fi
