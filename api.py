from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import io
import torch
import numpy as np
from scipy.io import wavfile
from tortoise.api_fast import TextToSpeech
from tortoise.utils.audio import load_audio
import time
import os
import platform

app = FastAPI()

# Initialize TTS model with all optimizations enabled
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Enable DeepSpeed on Linux, disable on Windows for compatibility
is_linux = platform.system() == 'Linux'
use_deepspeed = is_linux and torch.cuda.is_available()

print(f"Platform: {platform.system()}")
print(f"DeepSpeed: {'Enabled' if use_deepspeed else 'Disabled'}")

# Enable all performance optimizations
tts = TextToSpeech(
    use_deepspeed=use_deepspeed,   # Enable on Linux, disable on Windows
    kv_cache=True,                 # 5x faster according to changelog  
    half=True,                     # Half precision for speed and memory
    autoregressive_batch_size=16,  # Optimal for RTX 4060
    device=device
)

class SynthesizePayload(BaseModel):
    text: str
    voice: str = "random"
    preset: str = "ultra_realtime"  # Default to ultra_realtime for <500ms target

def get_preset_settings(preset):
    """
    Returns the settings for a given preset, compatible with tts_stream method.
    """
    # Base settings for all presets
    settings = {
        'temperature': 0.8, 
        'length_penalty': 1.0, 
        'repetition_penalty': 2.0,
        'top_p': 0.8,
        'cond_free_k': 2.0, 
        'diffusion_temperature': 1.0,
        'cond_free': True,
        'k': 1,
        'verbose': False  # Reduce console output
    }
    
    # Preset-specific settings
    presets = {
        'ultra_realtime': {
            'num_autoregressive_samples': 1, 
            'diffusion_iterations': 1,  # Absolute minimum for <500ms
            'cond_free': False,  # Disable conditioning-free for speed
            'temperature': 0.7,  # Slightly lower for more deterministic output
        },
        'ultra_fast': {
            'num_autoregressive_samples': 1, 
            'diffusion_iterations': 10
        },
        'fast': {
            'num_autoregressive_samples': 32, 
            'diffusion_iterations': 50
        },
        'standard': {
            'num_autoregressive_samples': 256, 
            'diffusion_iterations': 200
        },
        'high_quality': {
            'num_autoregressive_samples': 256, 
            'diffusion_iterations': 400
        },
    }
    
    # Update with preset-specific settings
    if preset in presets:
        settings.update(presets[preset])
    else:
        # Default to ultra_realtime for unknown presets
        settings.update(presets['ultra_realtime'])
    
    return settings

@app.post("/synthesize")
async def synthesize(payload: SynthesizePayload):
    try:
        # Record start time for performance measurement
        start_time = time.time()
        
        # Generate audio using the streaming TTS method which accepts diffusion parameters
        try:
            voice_samples = load_voice_samples(payload.voice)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        audio_generator = tts.tts_stream(
            payload.text,
            voice_samples=voice_samples,
            **get_preset_settings(payload.preset)
        )
        
        # Get the first (and only) audio chunk from the generator
        pcm_audio = next(audio_generator)
        
        # Record generation time
        generation_time = time.time() - start_time
        
        # Convert tensor to numpy array
        if isinstance(pcm_audio, torch.Tensor):
            audio_data = pcm_audio.cpu().numpy()
        else:
            audio_data = pcm_audio
        
        # Handle multi-dimensional audio data
        if len(audio_data.shape) > 1:
            # Take the first channel if multi-channel
            audio_data = audio_data[0] if audio_data.shape[0] == 1 else audio_data.flatten()
        
        # Calculate audio length and RTF
        audio_length = len(audio_data) / 24000  # 24kHz sample rate
        rtf = generation_time / audio_length
        
        # Log performance metrics
        print(f"🎤 Generated {audio_length:.2f}s audio in {generation_time:.2f}s")
        
        # Real-time is defined as generation time < 500ms
        is_realtime = generation_time < 0.5
        rtf_status = "(REAL-TIME!)" if is_realtime else "(slower than real-time)"
        print(f"⚡ RTF: {rtf:.2f}x | Gen Time: {generation_time*1000:.0f}ms {rtf_status}")
        
        # Normalize audio to 16-bit PCM range
        audio_data = np.clip(audio_data, -1.0, 1.0)
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Convert to WAV format using scipy
        bio = io.BytesIO()
        wavfile.write(bio, 24000, audio_data)  # 24000 Hz sample rate (Tortoise default)
        bio.seek(0)
        
        # Add performance headers
        headers = {
            "X-Generation-Time": str(generation_time),
            "X-Generation-Time-Ms": str(int(generation_time * 1000)),
            "X-Audio-Length": str(audio_length),
            "X-RTF": str(rtf),
            "X-Preset": payload.preset,
            "X-Voice": payload.voice,
            "X-Is-Realtime": str(generation_time < 0.5)
        }
        
        return StreamingResponse(bio, media_type="audio/wav", headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/synthesize_stream")
async def synthesize_stream(payload: SynthesizePayload):
    """
    Streaming synthesis endpoint for real-time audio generation.
    Returns audio chunks as they are generated.
    """
    try:
        def generate_audio():
            # Use the streaming TTS method which accepts diffusion parameters
            try:
                voice_samples = load_voice_samples(payload.voice)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            
            audio_generator = tts.tts_stream(
                payload.text,
                voice_samples=voice_samples,
                **get_preset_settings(payload.preset)
            )
            
            for audio_chunk in audio_generator:
                if isinstance(audio_chunk, torch.Tensor):
                    audio_data = audio_chunk.cpu().numpy()
                else:
                    audio_data = audio_chunk
                
                # Handle multi-dimensional audio data
                if len(audio_data.shape) > 1:
                    audio_data = audio_data[0] if audio_data.shape[0] == 1 else audio_data.flatten()
                
                # Normalize audio to 16-bit PCM range
                audio_data = np.clip(audio_data, -1.0, 1.0)
                audio_data = (audio_data * 32767).astype(np.int16)
                
                # Convert to WAV format
                bio = io.BytesIO()
                wavfile.write(bio, 24000, audio_data)
                bio.seek(0)
                yield bio.read()
        
        return StreamingResponse(generate_audio(), media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint to verify API is running."""
    return {
        "status": "healthy",
        "device": device,
        "platform": platform.system(),
        "optimizations": {
            "use_deepspeed": use_deepspeed,
            "kv_cache": True,
            "half": True,
            "autoregressive_batch_size": 16
        }
    }

@app.get("/voices")
async def list_voices():
    """List all available voices."""
    voices_dir = os.path.join("tortoise", "voices")
    try:
        if os.path.exists(voices_dir):
            voices = [d for d in os.listdir(voices_dir) if os.path.isdir(os.path.join(voices_dir, d))]
            return {
                "voices": voices,
                "total": len(voices),
                "note": "Use 'random' for random voice selection"
            }
        else:
            return {"error": "Voices directory not found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def load_voice_samples(voice_name):
    """
    Load voice samples for a given voice name.
    Returns None for random voice, or a list of audio tensors for specific voices.
    """
    if voice_name == "random":
        return None
    
    voice_dir = os.path.join("tortoise", "voices", voice_name)
    if not os.path.exists(voice_dir):
        raise ValueError(f"Voice '{voice_name}' not found in voices directory")
    
    # Load all .wav files from the voice directory
    voice_samples = []
    for file in os.listdir(voice_dir):
        if file.endswith(('.wav', '.mp3', '.flac')):
            file_path = os.path.join(voice_dir, file)
            try:
                # Load audio using tortoise's audio loading function
                audio = load_audio(file_path, 22050)  # 22050 Hz for voice samples
                voice_samples.append(audio)
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
                continue
    
    if not voice_samples:
        raise ValueError(f"No valid audio files found in voice directory: {voice_dir}")
    
    return voice_samples

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
