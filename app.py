# app.py
# FastAPI backend for speech feature extraction

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import subprocess
import os
import soundfile as sf
import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
from scipy.signal import find_peaks

import base64
import time
import wave
from io import BytesIO

import matplotlib.pyplot as plt
from scipy.signal import resample

import webrtcvad
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration


# -----------------------------------------------------------
# Load Whisper model
# -----------------------------------------------------------
processor = WhisperProcessor.from_pretrained("local_model/processor")
model = WhisperForConditionalGeneration.from_pretrained("local_model/model").to("cpu").eval()

# -----------------------------------------------------------
# Tamil normalization
# -----------------------------------------------------------
tamil_to_latin = {
    "பா": "pa", "டா": "ta", "கா": "ka"
}

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
    out = []
    i = 0
    while i < len(text):
        for key in sorted(patterns, key=lambda x: -len(x)):
            if text.startswith(key, i):
                out.extend(patterns[key])
                i += len(key)
                break
        else:
            i += 1
    return out

# -----------------------------------------------------------
# VAD
# -----------------------------------------------------------
def get_vad_segments(audio, sr):
    vad = webrtcvad.Vad(2)
    frame_ms = 30
    frame_len = int(sr * frame_ms / 1000)

    bytes_data = (audio * 32768).astype(np.int16).tobytes()
    voiced = []

    for i in range(0, len(bytes_data), frame_len * 2):
        frame = bytes_data[i:i + frame_len * 2]
        if len(frame) < frame_len * 2:
            break
        voiced.append(vad.is_speech(frame, sr))

    segments = []
    start = None
    for i, v in enumerate(voiced):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segments.append((start * 0.03, i * 0.03))
            start = None

    if start is not None:
        segments.append((start * 0.03, len(voiced) * 0.03))

    return segments

# -----------------------------------------------------------
# Waveform plot → Base64
# -----------------------------------------------------------
def make_waveform(audio, sr, syllables, timestamps):
    times = np.linspace(0, len(audio)/sr, len(audio))

    plt.figure(figsize=(12, 3))
    plt.plot(times, audio)

    for t, syl in zip(timestamps, syllables):
        plt.axvline(t)
        plt.text(t, 0.6, tamil_to_latin.get(syl, syl), ha="center")

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    return base64.b64encode(buf.read()).decode()



# ---------------------------------------------------------
# ✅ FastAPI Setup
# ---------------------------------------------------------
app = FastAPI()
origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://localhost:3000",    # if using React/Vite
    "http://127.0.0.1:3000"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---------------------------------------------------------
# ✅ Instantaneous Jitter + Shimmer
# ---------------------------------------------------------
def instantaneous_jitter_shimmer(pitch_times, pitch_vals, amp_times, amp_vals):
    import numpy as np

    mask = (~np.isnan(pitch_vals)) & (pitch_vals > 0)
    t = pitch_times[mask]
    f0 = pitch_vals[mask]

    result = {
        "jitter_times": [],
        "jitter_values": [],
        "shimmer_times": [],
        "shimmer_values": []
    }

    if len(f0) < 2:
        return result

    # ===== JITTER =====
    periods = 1.0 / f0
    jitter_vals = np.abs(np.diff(periods)) / periods[:-1]    # fractional jitter
    jitter_times = t[1:]

    result["jitter_times"] = jitter_times.tolist()
    result["jitter_values"] = (jitter_vals * 100).tolist()   # %


    # ===== SHIMMER ===== (RMS used as amplitude proxy)
    amp_times = np.array(amp_times)
    amp_vals  = np.array(amp_vals)

    # convert RMS-dB → linear amplitude if needed
    if np.nanmax(amp_vals) - np.nanmin(amp_vals) > 1 and np.nanmin(amp_vals) < 0:
        amp_vals = 10 ** (amp_vals / 20)

    idx = np.argsort(amp_times)
    amp_interp = np.interp(t, amp_times[idx], amp_vals[idx])

    if len(amp_interp) >= 2:
        shimmer_vals = np.abs(np.diff(amp_interp)) / (amp_interp[:-1] + 1e-12)
        shimmer_times = t[1:]

        result["shimmer_times"] = shimmer_times.tolist()
        result["shimmer_values"] = (shimmer_vals * 100).tolist()    # %

    return result


# ---------------------------------------------------------
# ✅ Utility
# ---------------------------------------------------------
def safe_to_list(x):
    if x is None: return []
    return np.asarray(x).astype(float).tolist()


# ---------------------------------------------------------
# ✅ Extract Features
# ---------------------------------------------------------
def extract_features(path):

    # Load
    samples, sr = sf.read(path, dtype="float32")
    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)

    duration = len(samples) / sr

    # ---------- Pitch ----------
    try:
        snd = parselmouth.Sound(path)
        pitch = snd.to_pitch(pitch_floor=75.0, pitch_ceiling=600.0)
        pitch_vals = np.nan_to_num(pitch.selected_array["frequency"]).astype(float)
        pitch_times = pitch.xs()
        print(pitch_times,pitch_vals)
        point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75.0, 600.0)
        jitter_local = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer_local = parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        # loud_db= librosa.load(snd, sr=None)
        # print("loudness",loud_db)
        print("Jitter (local):", jitter_local)
        print("Shimmer (local):", shimmer_local)
    except:
        pitch_vals, pitch_times = [], []
        jitter_local=None
        shimmer_local=None

    # ---------- Jitter / Shimmer ----------
    # try:
    #     jitter_local = call(snd, "Get jitter (local)", 0, 0, 75, 600, 1.3, 1.6)
    # except:
    #     jitter_local = None

    # try:
    #     shimmer_local = call(snd, "Get shimmer (local)", 0, 0, 75, 600, 1.3, 1.6)
    # except:
    #     shimmer_local = None

    # ---------- RMS ----------
    hop = int(0.01 * sr)
    win = int(0.03 * sr)
    rms = librosa.feature.rms(y=samples, frame_length=win, hop_length=hop)[0]
    rms_t = librosa.frames_to_time(np.arange(len(rms)), sr=sr)

    # ---------- Loudness (dB) ----------
    # safe = np.where(rms <= 1e-12, 1e-12, rms)
    
    # loud_db = librosa.amplitude_to_db(safe, ref=np.max)

    loud_db = 20 * np.log10(rms/ 20e-6)
    # calibration_offset = 70 - np.max(loud_db)
    # loud_db = loud_db + calibration_offset
    loud_db = np.nan_to_num(loud_db, nan=0.0, posinf=0.0, neginf=0.0)
    loud_db = loud_db.tolist()
    # ---------- Zero Crossing ----------
    zcr = librosa.feature.zero_crossing_rate(samples, frame_length=win, hop_length=hop)[0]
    zcr_t = librosa.frames_to_time(np.arange(len(zcr)), sr=sr)

    # ---------- Spectral Centroid ----------
    spec_cent = librosa.feature.spectral_centroid(y=samples, sr=sr, hop_length=hop)[0]
    spec_t = librosa.frames_to_time(np.arange(len(spec_cent)), sr=sr)

    # ---------- Speech Rate (rough syllable count) ----------
    thr = np.percentile(rms, 100)
    peaks, _ = find_peaks(rms, height=thr, distance=max(1, int(0.08 / (hop / sr))))
    speech_rate = len(peaks) / duration if duration > 0 else 0
        # ---------- Instantaneous Jitter / Shimmer ----------
    inst = instantaneous_jitter_shimmer(
        pitch_times,
        pitch_vals,
        rms_t,
        rms
    )

    return {
        "duration": duration,
        "sr": sr,
        "pitch": {
            "times": safe_to_list(pitch_times),
            "values": safe_to_list(pitch_vals),
        },
        "jitter_local": float(jitter_local) if jitter_local is not None else None,
        "shimmer_local": float(shimmer_local) if shimmer_local is not None else None,
        "rms": {
            "times": safe_to_list(rms_t),
            "values": safe_to_list(rms),
        },
        "loudness": {
            "times": safe_to_list(rms_t),
            "values": safe_to_list(loud_db),
        },
        "jitter": {
            "times":   inst["jitter_times"],
            "values":  inst["jitter_values"],
        },
        "shimmer": {
            "times":   inst["shimmer_times"],
            "values":  inst["shimmer_values"],
        },
        "zcr": {
            "times": safe_to_list(zcr_t),
            "values": safe_to_list(zcr),
        },
        "spectralCentroid": {
            "times": safe_to_list(spec_t),
            "values": safe_to_list(spec_cent),
        },
        "speech_rate_sps": float(speech_rate),
    }


# ---------------------------------------------------------
# ✅ WebM → WAV conversion
# ---------------------------------------------------------
def convert_to_wav(input_bytes):
    temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")

    temp_in.write(input_bytes)
    temp_in.close()

    cmd = [
        "ffmpeg", "-y", "-i", temp_in.name,
        "-ar", "16000", "-ac", "1",   # 16k mono
        temp_out.name
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return temp_out.name
    except:
        raise HTTPException(500, "FFmpeg audio conversion failed")
    finally:
        os.unlink(temp_in.name)


# ---------------------------------------------------------
# ✅ API endpoint
# ---------------------------------------------------------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    data = await file.read()        # raw input
    wav_path = convert_to_wav(data) # convert > wav

    try:
        features = extract_features(wav_path)
    except Exception as e:
        raise HTTPException(500, f"Feature extraction error: {e}")
    finally:
        try: os.unlink(wav_path)
        except: pass

    return features


@app.post("/process-pata-ka")
async def process_pataka(file: UploadFile = File(...)):
    # -------------------------------------------
    # Read bytes
    # -------------------------------------------
    audio_bytes = await file.read()

    audio_48k, sr = sf.read(BytesIO(audio_bytes))
    if sr != 48000:
        raise ValueError("Client must record at 48kHz.")

    audio_16k = resample(audio_48k, int(len(audio_48k) * 16000 / 48000))
    duration = len(audio_16k) / 16000

    inputs = processor(audio_16k, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        ids = model.generate(inputs.input_features)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    print("Recogonized text: ",text)

    syllables = normalize_syllables(text)
    print("All syllables", syllables)

    segments = get_vad_segments(audio_16k, 16000)
    if len(segments) >= len(syllables):
        timestamps = [(s+e)/2 for s, e in segments[:len(syllables)]]
    else:
        timestamps = np.linspace(0.1, duration - 0.1, len(syllables))

    png_b64 = make_waveform(audio_16k, 16000, syllables, timestamps)
    out={
        "syllables": [{"text": s, "time": round(t, 2)} for s, t in zip(syllables, timestamps)],
        "duration": round(duration, 2),
        "waveform_png": png_b64
    }
    print(out)

    return out

# ---------------------------------------------------------
# RUN
# uvicorn app:app --reload
# ---------------------------------------------------------