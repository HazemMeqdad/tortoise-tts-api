#!/usr/bin/env python3
"""
Test script to verify the optimized TTS API works correctly.
Tests the api.py file with voice selection and performance monitoring.
"""

import sys
import time
import torch
import requests
import json
import os
import platform

def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        from tortoise.api_fast import TextToSpeech
        print("✅ tortoise.api_fast imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import tortoise.api_fast: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import numpy: {e}")
        return False
    
    try:
        from scipy.io import wavfile
        print("✅ scipy.io.wavfile imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import scipy.io.wavfile: {e}")
        return False
    
    try:
        import fastapi
        print("✅ fastapi imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import fastapi: {e}")
        return False
    
    return True

def test_gpu_setup():
    """Test GPU availability and memory."""
    print("\n🎮 Testing GPU setup...")
    
    print(f"🖥️ Platform: {platform.system()}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        available_memory = torch.cuda.mem_get_info()[0] / 1024**3
        
        print(f"✅ GPU: {gpu_name}")
        print(f"💾 Total Memory: {total_memory:.1f} GB")
        print(f"💾 Available Memory: {available_memory:.1f} GB")
        
        if "RTX 4060" in gpu_name:
            print("🎯 RTX 4060 detected - optimizations should work great!")
        
        # Check DeepSpeed availability
        is_linux = platform.system() == 'Linux'
        if is_linux:
            print("🚀 Linux detected - DeepSpeed will be enabled for maximum performance!")
        else:
            print("🪟 Windows detected - DeepSpeed will be disabled for compatibility")
        
        return True
    else:
        print("❌ No GPU available - will use CPU (slower)")
        return False

def test_optimized_tts():
    """Test the optimized TTS configuration directly."""
    print("\n🚀 Testing optimized TTS directly...")
    
    try:
        from tortoise.api_fast import TextToSpeech
        
        # Test with optimizations (matching api.py configuration)
        print("⚙️ Initializing TTS with optimizations...")
        
        # Enable DeepSpeed on Linux, disable on Windows for compatibility
        is_linux = platform.system() == 'Linux'
        use_deepspeed = is_linux and torch.cuda.is_available()
        
        print(f"🖥️ Platform: {platform.system()}")
        print(f"🚀 DeepSpeed: {'Enabled' if use_deepspeed else 'Disabled'}")
        
        tts = TextToSpeech(
            use_deepspeed=use_deepspeed,   # Enable on Linux, disable on Windows
            kv_cache=True,                 # 5x faster according to changelog  
            half=True,                     # Half precision for speed and memory
            autoregressive_batch_size=16,  # Optimal for RTX 4060
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        test_text = "Hello, this is a test of the optimized Tortoise TTS system."
        
        print("🎤 Generating test audio...")
        start_time = time.time()
        
        # Use the same method as api.py
        settings = {
            'num_autoregressive_samples': 1, 
            'diffusion_iterations': 1,
            'cond_free': False,
            'temperature': 0.7,
            'length_penalty': 1.0,
            'repetition_penalty': 2.0,
            'top_p': 0.8,
            'cond_free_k': 2.0,
            'diffusion_temperature': 1.0,
            'k': 1,
            'verbose': False
        }
        
        audio_generator = tts.tts_stream(test_text, voice_samples=None, **settings)
        audio = next(audio_generator)
        
        generation_time = time.time() - start_time
        
        if isinstance(audio, torch.Tensor):
            audio_length = audio.shape[-1] / 24000
        else:
            audio_length = len(audio) / 24000
        
        rtf = generation_time / audio_length
        
        print(f"✅ Audio generated successfully!")
        print(f"⏱️ Generation time: {generation_time:.2f}s ({generation_time*1000:.0f}ms)")
        print(f"🎵 Audio length: {audio_length:.2f}s")
        print(f"⚡ RTF: {rtf:.2f}x")
        
        # Real-time is defined as < 500ms
        is_realtime = generation_time < 0.5
        if is_realtime:
            print("🎉 SUCCESS: Real-time capable (<500ms)!")
        else:
            print("⚠️ WARNING: Slower than real-time (>500ms)")
        
        return True
        
    except Exception as e:
        print(f"❌ TTS test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_endpoints():
    """Test the API endpoints if server is running."""
    print("\n🌐 Testing API endpoints...")
    
    base_url = "http://localhost:8888"
    
    try:
        # Test health endpoint
        print("🔍 Testing health endpoint...")
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check passed: {health_data['status']}")
            print(f"🎮 Device: {health_data['device']}")
            print(f"🖥️ Platform: {health_data.get('platform', 'Unknown')}")
            print(f"🚀 DeepSpeed: {'Enabled' if health_data['optimizations']['use_deepspeed'] else 'Disabled'}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
        # Test voices endpoint
        print("🎵 Testing voices endpoint...")
        response = requests.get(f"{base_url}/voices", timeout=5)
        if response.status_code == 200:
            voices_data = response.json()
            print(f"✅ Found {voices_data['total']} voices")
            if voices_data['total'] > 0:
                print(f"📝 Available voices: {voices_data['voices'][:5]}...")  # Show first 5
        else:
            print(f"❌ Voices endpoint failed: {response.status_code}")
            
        # Test synthesis endpoint
        print("🎤 Testing synthesis endpoint...")
        payload = {
            "text": "Hello, this is a test of the API.",
            "voice": "random",
            "preset": "ultra_realtime"
        }
        
        start_time = time.time()
        response = requests.post(
            f"{base_url}/synthesize", 
            json=payload, 
            timeout=30
        )
        api_time = time.time() - start_time
        
        if response.status_code == 200:
            print(f"✅ Synthesis successful!")
            print(f"⏱️ API response time: {api_time:.2f}s")
            
            # Check headers
            headers = response.headers
            if 'X-Generation-Time-Ms' in headers:
                gen_time_ms = int(headers['X-Generation-Time-Ms'])
                print(f"🚀 Generation time: {gen_time_ms}ms")
                print(f"⚡ RTF: {headers.get('X-RTF', 'N/A')}")
                print(f"🎵 Voice: {headers.get('X-Voice', 'N/A')}")
                print(f"⚙️ Preset: {headers.get('X-Preset', 'N/A')}")
                
                is_realtime = gen_time_ms < 500
                if is_realtime:
                    print("🎉 SUCCESS: API is real-time capable!")
                else:
                    print("⚠️ WARNING: API is slower than real-time")
            
            return True
        else:
            print(f"❌ Synthesis failed: {response.status_code}")
            print(f"❌ Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ API server not running. Start it with:")
        print("   uvicorn api:app --host \"0.0.0.0\" --port \"8888\" --reload")
        return False
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Tortoise TTS API Optimization Test")
    print("=" * 50)
    
    # Run tests
    imports_ok = test_imports()
    gpu_ok = test_gpu_setup()
    
    if imports_ok:
        tts_ok = test_optimized_tts()
        api_ok = test_api_endpoints()
    else:
        print("❌ Skipping TTS tests due to import failures")
        tts_ok = False
        api_ok = False
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"GPU: {'✅ PASS' if gpu_ok else '❌ FAIL'}")
    print(f"TTS Direct: {'✅ PASS' if tts_ok else '❌ FAIL'}")
    print(f"API Endpoints: {'✅ PASS' if api_ok else '❌ FAIL'}")
    
    if imports_ok and tts_ok:
        print("\n🎉 TTS engine is working! Your optimized setup is ready.")
        if api_ok:
            print("🌐 API is running and responsive!")
        else:
            print("\n💡 To start the API server:")
            print('   uvicorn api:app --host "0.0.0.0" --port "8888" --reload')
        
        print("\n🔧 Test with curl:")
        print('   curl -X POST "http://localhost:8888/synthesize" \\')
        print('     -H "Content-Type: application/json" \\')
        print('     -d \'{"text":"Hello world","voice":"random","preset":"ultra_realtime"}\' \\')
        print('     --output test.wav')
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
