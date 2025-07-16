#!/usr/bin/env python3
"""
Cross-platform startup script for Tortoise TTS API
Automatically detects platform and enables appropriate optimizations.
"""

import os
import sys
import platform
import subprocess
import torch

def check_requirements():
    """Check if all requirements are installed."""
    try:
        import fastapi
        import uvicorn
        import scipy
        import numpy
        from tortoise.api_fast import TextToSpeech
        print("✅ All requirements satisfied")
        return True
    except ImportError as e:
        print(f"❌ Missing requirement: {e}")
        return False

def check_deepspeed():
    """Check if DeepSpeed is available and working."""
    try:
        import deepspeed
        print("✅ DeepSpeed is available")
        return True
    except ImportError:
        print("ℹ️ DeepSpeed not available")
        return False

def main():
    print("🚀 Tortoise TTS API Startup")
    print("=" * 40)
    
    # Platform detection
    system = platform.system()
    print(f"🖥️ Platform: {system}")
    
    # GPU detection
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🎮 GPU: {gpu_name}")
    else:
        print("⚠️ No GPU detected - using CPU")
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements not met. Please install:")
        print("   pip install -r requirements_api.txt")
        if system == "Linux" and torch.cuda.is_available():
            print("   pip install deepspeed  # For Linux GPU optimization")
        sys.exit(1)
    
    # Check DeepSpeed availability
    deepspeed_available = check_deepspeed()
    if system == "Linux" and torch.cuda.is_available() and deepspeed_available:
        print("🚀 DeepSpeed optimization enabled!")
    elif system == "Linux" and torch.cuda.is_available() and not deepspeed_available:
        print("⚠️ DeepSpeed not installed. Install with: pip install deepspeed")
    
    # Start the API server
    print("\n🌐 Starting API server...")
    print("   Access: http://localhost:8888")
    print("   Docs: http://localhost:8888/docs")
    print("   Health: http://localhost:8888/health")
    print("\n⏹️ Press Ctrl+C to stop")
    
    try:
        # Import and run uvicorn
        import uvicorn
        uvicorn.run(
            "api:app",
            host="0.0.0.0",
            port=8888,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"\n❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
