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


# ---------------------------------------------------------
# ✅ FastAPI Setup
# ---------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


# ---------------------------------------------------------
# RUN
# uvicorn app:app --reload
# ---------------------------------------------------------
