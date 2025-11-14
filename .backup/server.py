import asyncio
import websockets
import soundfile as sf
import numpy as np
import os
import torch
import wave
import matplotlib.pyplot as plt
from scipy.signal import resample
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from aiohttp import web
import webrtcvad
import time
import json

# Constants
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)
RAW_AUDIO_PATH = os.path.join(STATIC_DIR, "recorded_raw.wav")
RESAMPLED_AUDIO_PATH = os.path.join(STATIC_DIR, "recorded.wav")

# Load Whisper model
print("[*] Loading model...")
processor = WhisperProcessor.from_pretrained("local_model/processor")
model = WhisperForConditionalGeneration.from_pretrained("local_model/model").to("cpu").eval()
print("[+] Whisper model loaded.")

# Tamil syllable to Latin mapping
tamil_to_latin = {
    "பா": "pa", "டா": "ta", "கா": "ka"
}

# Tamil normalization
def normalize_syllables(text):
    patterns = {
        "பாட்டாக": ["பா", "டா", "கா"],
        "பாட்டா": ["பா", "டா"],
        "பாக்கா": ["பா", "கா"],
        "காப்பா": ["கா", "பா"],
        "டாக்கா": ["டா", "கா"],
        "பா": ["பா"], "ப": ["பா"], "ப்": ["பா"], "ப்ப": ["பா"],
        "ட்டா": ["டா"], "ட்": ["டா"], "ட": ["டா"], "டா": ["டா"],
        "க்கா": ["கா"], "க்": ["கா"], "க": ["கா"], "கா": ["கா"],
    }
    result = []
    i = 0
    while i < len(text):
        matched = False
        for k in sorted(patterns.keys(), key=lambda x: -len(x)):
            if text[i:i+len(k)] == k:
                result.extend(patterns[k])
                i += len(k)
                matched = True
                break
        if not matched:
            i += 1
    return result

# Use VAD for voiced regions
def get_vad_segments(audio, sr, frame_ms=30):
    vad = webrtcvad.Vad(2)
    frame_len = int(sr * frame_ms / 1000)
    audio_bytes = (audio * 32768).astype(np.int16).tobytes()
    voiced = []

    for i in range(0, len(audio_bytes), frame_len * 2):
        frame = audio_bytes[i:i + frame_len * 2]
        if len(frame) < frame_len * 2:
            break
        voiced.append(vad.is_speech(frame, sr))

    segments = []
    start = None
    for i, is_voiced in enumerate(voiced):
        if is_voiced and start is None:
            start = i
        elif not is_voiced and start is not None:
            segments.append((start * frame_ms / 1000, i * frame_ms / 1000))
            start = None
    if start is not None:
        segments.append((start * frame_ms / 1000, len(voiced) * frame_ms / 1000))
    return segments

# Plot waveform with syllables
def plot_waveform(audio_path, syllables, timestamps, save_path):
    audio, sr = sf.read(audio_path)
    duration = len(audio) / sr
    times = np.linspace(0, duration, len(audio))

    plt.figure(figsize=(12, 3))
    plt.plot(times, audio, color='lime')
    plt.title("Waveform with Recognized Syllables", color='white')
    plt.xlabel("Time (s)", color='white')
    plt.ylabel("Amplitude", color='white')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.gca().set_facecolor("black")
    plt.gcf().patch.set_facecolor("black")
    plt.tick_params(colors='white')

    for t, syl in zip(timestamps, syllables):
        plt.axvline(t, color='cyan', linestyle='--', linewidth=1)
        plt.text(t, 0.6, tamil_to_latin.get(syl, syl), fontsize=12, color="cyan", ha='center',
                 bbox=dict(facecolor='black', edgecolor='cyan', alpha=0.7))

    plt.tight_layout()
    plt.savefig(save_path, facecolor='black')
    plt.close()
    print(f"[+] Saved plot: {save_path}")

# WebSocket handler
async def handle(websocket):
    print("[+] Client connected.")
    audio_data = bytearray()

    try:
        async for message in websocket:
            if isinstance(message, str) and message == "END":
                print("[*] Finalizing...")

                # Save raw audio
                with wave.open(RAW_AUDIO_PATH, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(48000)
                    wf.writeframes(audio_data)

                # Resample to 16kHz
                audio, _ = sf.read(RAW_AUDIO_PATH)
                audio_16k = resample(audio, int(len(audio) * 16000 / 48000))
                sf.write(RESAMPLED_AUDIO_PATH, audio_16k, 16000)

                duration = len(audio_16k) / 16000

                # Transcribe
                print("[*] Transcribing...")
                inputs = processor(audio_16k, sampling_rate=16000, return_tensors="pt")
                with torch.no_grad():
                    predicted_ids = model.generate(inputs.input_features)
                transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                print(f"[TRANSCRIPTION]: {transcription}")

                # Normalize
                syllables = normalize_syllables(transcription)
                print(f"[SYLLABLES]: {' '.join(syllables)}")

                # VAD segments → estimate timestamps
                segments = get_vad_segments(audio_16k, 16000)
                if len(segments) >= len(syllables):
                    used = segments[:len(syllables)]
                    times = [(s + e) / 2 for s, e in used]
                else:
                    times = np.linspace(0.1, duration - 0.1, len(syllables))

                syllable_data = [{"text": syl, "time": round(t, 2)} for syl, t in zip(syllables, times)]

                # Plot + send result
                timestamp = str(int(time.time()))
                image_path = f"{STATIC_DIR}/waveform_with_syllables_{timestamp}.png"
                plot_waveform(RESAMPLED_AUDIO_PATH, syllables, times, image_path)

                await websocket.send(json.dumps({
                    "syllables": syllable_data,
                    "timestamp": timestamp,
                    "duration": round(duration, 2)
                }))

            else:
                audio_data.extend(message)

    except Exception as e:
        print(f"[!] WebSocket Error: {e}")

# Main server
async def main():
    async with websockets.serve(handle, "0.0.0.0", 8765, ping_interval=None):
        app = web.Application()
        app.add_routes([
            web.static("/static", "./static"),
            web.get("/", lambda r: web.FileResponse("static/index.html"))
        ])
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", 8000)
        await site.start()
        print("[*] HTTP server running at http://localhost:8000")
        print("[*] WebSocket server at ws://localhost:8765")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
