from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import parselmouth
import tempfile
import numpy as np

app = FastAPI()

# CORS FIX
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze_vowel")
async def analyze_vowel(file: UploadFile = File(...)):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(await file.read())
    tmp.close()

    snd = parselmouth.Sound(tmp.name)
    formant = snd.to_formant_burg()

    f1 = formant.get_value_at_time(1, snd.duration/2)
    f2 = formant.get_value_at_time(2, snd.duration/2)

    return {"f1": f1, "f2": f2}
