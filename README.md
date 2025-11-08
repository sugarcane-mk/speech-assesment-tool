# Speech Assessment — Final Working Package

This textdoc contains **two files** you can copy-save and run locally:

1. `index.html` — frontend (single-file) using Plotly, records audio, shows live loudness, sends WAV to backend, renders interactive panes with click handlers and an annotations panel.
2. `app.py` — FastAPI backend using `parselmouth` (Praat wrapper) that accepts `POST /analyze` with a WAV file and returns pitch, jitter, shimmer, RMS, speech-rate, etc.

---

## How to run (local)

1. Save `index.html` and `app.py` as shown above. Create `requirements.txt`.
2. Create a Python virtualenv and install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Run the FastAPI server:

```bash
uvicorn app:app --reload --port 8000
```

4. Serve the `index.html` (open directly in modern browser or serve with a tiny static server). For CORS simplicity the FastAPI allows all origins; opening `index.html` from `file://` should work but some browsers restrict `getUserMedia` on file URIs — best to use a tiny static server:

```bash
python -m http.server 5500
# then open http://localhost:5500/index.html
```

5. Click **Start Recording**, speak, **Stop**. The page will show client-side plots immediately and then the server-enhanced plots with jitter/shimmer/praat-based metrics.

---
