// main.js — complete frontend logic
// Assumes index.html contains the containers with ids:
// startBtn, stopBtn, status, meterFill, panes, annotations, downloadWav, playBack, processingOverlay
// and Plotly is already loaded (via script tag)

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const statusEl = document.getElementById("status");
const meterFill = document.getElementById("meterFill");
const panes = document.getElementById("panes");
const annotationsEl = document.getElementById("annotations");
const downloadWavBtn = document.getElementById("downloadWav");
const playBackBtn = document.getElementById("playBack");
const processingOverlay = document.getElementById("processingOverlay");

let mediaStream = null;
let mediaRecorder = null;
let recordedChunks = [];
let audioCtx = null;
let liveAnalyser = null;
let rafId = null;
let sampleRate = 48000;
let recordedFloat32 = null;

let patakaRecorder = null;
let audioChunks = [];
let recordingStream = null;
let isRecording = false;

// Helpers
function showProcessing() {
  if (processingOverlay) processingOverlay.style.display = "flex";
}
function hideProcessing() {
  if (processingOverlay) processingOverlay.style.display = "none";
}
function setStatus(text) { if (statusEl) statusEl.textContent = text; }
function rmsToDb(rms) { if (!rms || rms <= 1e-12) return -120; return 20 * Math.log10(rms); }

// Defensive get element
function E(id) { return document.getElementById(id); }

// Start recording
async function startRecording() {
  try {
    startBtn.disabled = true;
    stopBtn.disabled = false;
    setStatus("Recording...");
    recordedChunks = [];

    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(mediaStream);

    mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) recordedChunks.push(e.data);
    };

    mediaRecorder.onstop = onStopRecording;
    mediaRecorder.start();

    // Live loudness display
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    sampleRate = audioCtx.sampleRate;
    const source = audioCtx.createMediaStreamSource(mediaStream);
    liveAnalyser = audioCtx.createAnalyser();
    liveAnalyser.fftSize = 2048;
    source.connect(liveAnalyser);

    const data = new Float32Array(liveAnalyser.fftSize);
    function updateMeter() {
      liveAnalyser.getFloatTimeDomainData(data);
      let s = 0;
      for (let i = 0; i < data.length; i++) s += data[i] * data[i];
      const rms = Math.sqrt(s / data.length);
      const db = rmsToDb(rms);
      // Normalize -60..0 dB to 0..1
      const norm = Math.min(1, Math.max(0, (db + 60) / 60));
      if (meterFill) meterFill.style.width = (norm * 100) + "%";
      rafId = requestAnimationFrame(updateMeter);
    }
    updateMeter();
  } catch (e) {
    console.error("startRecording error", e);
    setStatus("Error starting recording");
    startBtn.disabled = false;
    stopBtn.disabled = true;
  }
}

// Stop recording
function stopRecording() {
  stopBtn.disabled = true;
  startBtn.disabled = false;
  setStatus("Processing...");
  showProcessing();

  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
  // stop streams
  if (mediaStream) {
    mediaStream.getTracks().forEach(t => t.stop());
    mediaStream = null;
  }
  if (audioCtx) {
    try { audioCtx.close(); } catch (e) { /* ignore */ }
    audioCtx = null;
  }
  if (rafId) {
    try { cancelAnimationFrame(rafId); } catch (e) {}
    rafId = null;
  }
}

// When mediaRecorder stops
async function onStopRecording() {
  const blob = new Blob(recordedChunks, { type: "audio/wav" });
  // decode to Float32 for waveform (and enable download/play)
  try {
    // create AudioContext to decode
    const decodeCtx = new (window.OfflineAudioContext || window.AudioContext)(1, 1, 44100);
    const arrayBuffer = await blob.arrayBuffer();
    let decoded;
    try {
      decoded = await (new Promise((resolve, reject) => {
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        ctx.decodeAudioData(arrayBuffer, buffer => {
          resolve(buffer);
          try { ctx.close(); } catch (e) {}
        }, err => { reject(err); });
      }));
      // Use channel data
      if (decoded && decoded.numberOfChannels > 0) {
        recordedFloat32 = decoded.getChannelData(0).slice(0);
        sampleRate = decoded.sampleRate || sampleRate;
      } else {
        recordedFloat32 = null;
      }
    } catch (err) {
      // decoding failed — leave recordedFloat32 null but still send blob to server
      console.warn("decodeAudioData failed:", err);
      recordedFloat32 = null;
    }
  } catch (e) {
    console.error("onStopRecording decode error", e);
    recordedFloat32 = null;
  }

  // enable download and playback
  if (downloadWavBtn) downloadWavBtn.disabled = false;
  if (playBackBtn) playBackBtn.disabled = false;

  // send to server
  await sendToServerAndPlot(blob);
}

// Send audio blob to server and plot returned features
async function sendToServerAndPlot(wavBlob) {
  setStatus("Analyzing on server...");
  showProcessing();

  const fd = new FormData();
  fd.append("file", wavBlob, "recording.wav");

  try {
    const res = await fetch("http://localhost:8000/analyze", { method: "POST", body: fd });
    if (!res.ok) {
      const text = await res.text();
      console.error("Server returned non-OK:", res.status, text);
      setStatus("Server error: " + res.status);
      return;
    }
    const json = await res.json();
    hideProcessing();
    setStatus("Done");
    renderPlotsFromServer(json);
  } catch (err) {
    console.error("sendToServerAndPlot error:", err);
    hideProcessing();
    setStatus("Network or server error");
  }
}

// Safely create / clear pane container
function clearPanes() {
  if (!panes) return;
  panes.innerHTML = "";
}

// Create pane helper (reuses existing element if present)
function createPane(plotId, title, height = 200) {
  if (!panes) return null;

  const wrapper = document.createElement("div");
  wrapper.className = "pane";
  wrapper.id = "pane-" + plotId;
  wrapper.style.display = "none";

  const h = document.createElement("h4");
  h.textContent = title;
  wrapper.appendChild(h);

  const plotDiv = document.createElement("div");
  plotDiv.id = plotId;
  plotDiv.style.width = "100%";
  plotDiv.style.height = height + "px";
  wrapper.appendChild(plotDiv);

  panes.appendChild(wrapper);

  // ✅ auto-show if feature already selected
  const toggleElem = document.querySelector(`[data-feature="${plotId}"]`);
  if (toggleElem && toggleElem.classList.contains("active")) {
    wrapper.style.display = "block";
  }

  return plotDiv;
}


// Add annotation entry
function addAnnotation(label, t, v) {
  if (!annotationsEl) return;
  const div = document.createElement("div");
  div.className = "anno-item";
  const valStr = (typeof v === "number") ? v.toFixed(4) : String(v);
  div.textContent = `${label} — t=${t.toFixed(3)}s  value=${valStr}`;
  annotationsEl.prepend(div);
}

// Attach click-to-annotate to Plotly plot
function attachPlotlyClick(plotId, label) {
  const el = document.getElementById(plotId);
  if (!el) return;
  el.on('plotly_click', (data) => {
    const p = data.points && data.points[0];
    if (!p) return;
    addAnnotation(label, p.x, p.y);
  });
  // click to expand (toggle height)
  el.addEventListener('dblclick', () => {
    const curr = el.style.height || "";
    if (!curr || curr === "200px") {
      el.style.height = "600px";
    } else {
      el.style.height = "200px";
    }
    // relayout to trigger Plotly redraw
    Plotly.Plots.resize(el);
  });
}

// Render waveform (downsample for plotting)
function plotWaveformFromFloat32(buffer, sr) {
  const id = "plot-waveform";
  // create pane
  createPane(id, "Waveform", 200);
  const el = document.getElementById(id);
  if (!el) return;
  if (!buffer || buffer.length === 0) {
    Plotly.newPlot(id, [{ x: [], y: [], mode: "lines" }], { margin: { t: 10 } });
    return;
  }
  const maxPoints = 20000;
  const step = Math.max(1, Math.floor(buffer.length / maxPoints));
  const x = [];
  const y = [];
  for (let i = 0; i < buffer.length; i += step) {
    x.push(i / sr);
    y.push(buffer[i]);
  }
  Plotly.newPlot(id, [{ x, y, mode: "lines", line: { width: 1 } }], { margin: { t: 10 } });
  attachPlotlyClick(id, "Waveform");
}

// === Updated renderPlotsFromServer ===
function renderPlotsFromServer(json) {
  clearPanes();

  // Summary box
  const summary = document.createElement("div");
  summary.className = "summary-box pane";
  summary.innerHTML = `
    <strong>Summary</strong><br/><br/>
    <strong>Duration:</strong> ${json.duration?.toFixed?.(3) ?? "n/a"} s<br/>
    <strong>Sample rate:</strong> ${json.sr ?? "n/a"}<br/>
    <strong>Speech rate:</strong> ${(json.speech_rate_sps ? (json.speech_rate_sps * 60).toFixed(1) : "n/a")} syl/min<br/>
    <strong>Jitter (local):</strong> ${json.jitter_local ?? "n/a"}<br/>
    <strong>Shimmer (local):</strong> ${json.shimmer_local ?? "n/a"}<br/>
  `;
  panes.appendChild(summary);

  // Waveform
  if (recordedFloat32) {
    const plotDiv = plotWaveformFromFloat32(recordedFloat32, sampleRate);
    const wrapper = document.getElementById("pane-plot-waveform");
    if (!wrapper) return;
  }

  // Short-time RMS
  if (json.rms && json.rms.values) {
    createPane("plot-rms", "Short-time RMS", 200);
    Plotly.newPlot("plot-rms", [{ x: json.rms.times, y: json.rms.values, mode: "lines" }], { margin: { t: 10 } });
    attachPlotlyClick("plot-rms", "RMS");
  }

  // Pitch
  console.log(json)
  if (json.pitch.times && json.pitch.values) {
    createPane("plot-pitch", "Pitch (Hz)", 200);
    Plotly.newPlot("plot-pitch", [{ x: json.pitch.times, y: json.pitch.values, mode: "lines" }], { margin: { t: 10 }, yaxis: { title: "Hz" } });
    attachPlotlyClick("plot-pitch", "Pitch");
  }

  // Jitter
  if (json.jitter && json.jitter.values) {
    const times = json.jitter.times || json.jitter.values.map((_, i) => i * (1/100));
    createPane("plot-jitter", "Jitter (per-voiced)", 200);
    Plotly.newPlot("plot-jitter", [{ x: times, y: json.jitter.values, mode: "lines" }], { margin: { t: 10 } });
    attachPlotlyClick("plot-jitter", "Jitter");
  }

  // Shimmer
  if (json.shimmer && json.shimmer.values) {
    const times = json.shimmer.times || json.shimmer.values.map((_, i) => i * (1/100));
    createPane("plot-shimmer", "Shimmer (per-voiced)", 200);
    Plotly.newPlot("plot-shimmer", [{ x: times, y: json.shimmer.values, mode: "lines" }], { margin: { t: 10 } });
    attachPlotlyClick("plot-shimmer", "Shimmer");
  }

  // ZCR
  if (json.zcr && json.zcr.values) {
    createPane("plot-zcr", "Zero Crossing Rate", 200);
    Plotly.newPlot("plot-zcr", [{ x: json.zcr.times, y: json.zcr.values, mode: "lines" }], { margin: { t: 10 } });
    attachPlotlyClick("plot-zcr", "ZCR");
  }

  // Spectral Centroid
  if (json.spectralCentroid && json.spectralCentroid.values) {
    createPane("plot-spectralCentroid", "Spectral Centroid (Hz)", 200);
    Plotly.newPlot("plot-spectralCentroid", [{ x: json.spectralCentroid.times, y: json.spectralCentroid.values, mode: "lines" }], { margin: { t: 10 } });
    attachPlotlyClick("plot-spectralCentroid", "Spectral Centroid");
  }

  // Loudness
  if (json.loudness && json.loudness.values) {
    createPane("plot-loudness", "Loudness (dB)", 200);
    Plotly.newPlot("plot-loudness", [{ x: json.loudness.times, y: json.loudness.values, mode: "lines" }], { margin: { t: 10 }, yaxis: { title: "dB" } });
    attachPlotlyClick("plot-loudness", "Loudness dB");
  }
}

// DOWNLOAD / PLAYBACK helpers
if (downloadWavBtn) {
  downloadWavBtn.addEventListener("click", () => {
    if (!recordedChunks || recordedChunks.length === 0) return;
    const blob = new Blob(recordedChunks, { type: "audio/wav" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "recording.wav";
    a.click();
  });
}
if (playBackBtn) {
  playBackBtn.addEventListener("click", () => {
    if (!recordedChunks || recordedChunks.length === 0) return;
    const blob = new Blob(recordedChunks, { type: "audio/wav" });
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    audio.play();
  });
}

// Hook buttons (existing ids in HTML)
if (startBtn) startBtn.addEventListener("click", startRecording);
if (stopBtn) stopBtn.addEventListener("click", stopRecording);
featureList.addEventListener('click', e => {
  const it = e.target.closest('.feature-item');
  if (!it) return;
  
  const fid = it.dataset.feature;   // e.g. "plot-rms"
  it.classList.toggle('active');

  const pane = document.getElementById('pane-' + fid);
  if (pane) {
    pane.style.display = it.classList.contains('active') ? 'block' : 'none';
    if (pane.style.display === "block") {
      const plot = document.getElementById(fid);
      if (plot) Plotly.Plots.resize(plot);   // ✅ force redraw
    }
  }
});

// pa-ta-ka logic
// document.getElementById("patakaBtn").addEventListener("click", startRecordingPataka);

// async function startRecordingPataka() {
//   document.getElementById("prompt").innerText = "Requesting microphone...";

//   const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

//   document.getElementById("prompt").innerText = "Recording... Say Pa-Ta-Ka";
  
//   audioChunks = [];
//   patakaRecorder = new MediaRecorder(stream);

//   patakaRecorder.ondataavailable = e => audioChunks.push(e.data);

//   patakaRecorder.onstop = () => {
//     document.getElementById("prompt").innerText = "Processing...";

//     const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
//     convertBlobToWavAndUpload(audioBlob);
//   };

//   patakaRecorder.start();

//   // Auto-stop after 3 seconds
//   setTimeout(() => patakaRecorder.stop(), 3000);
// }
document.getElementById("patakaBtn").addEventListener("click", toggleRecording);

async function toggleRecording() {
  if (!isRecording) {
    startRecordingPataka();
  } else {
    stopRecordingPataka();
  }
}

async function startRecordingPataka() {
  document.getElementById("prompt").innerText = "Requesting microphone...";

  recordingStream = await navigator.mediaDevices.getUserMedia({ audio: true });

  document.getElementById("prompt").innerText = "Recording... Say Pa-Ta-Ka";

  const btn = document.getElementById("patakaBtn");
  btn.textContent = "⛔ Stop";

  audioChunks = [];
  patakaRecorder = new MediaRecorder(recordingStream);
  isRecording = true;

  patakaRecorder.ondataavailable = e => audioChunks.push(e.data);

  patakaRecorder.onstop = () => {
    document.getElementById("prompt").innerText = "Processing...";
    btn.textContent = "🎤 Start Pa-Ta-Ka";
    isRecording = false;

    const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
    convertBlobToWavAndUpload(audioBlob);

    // close the mic
    recordingStream.getTracks().forEach(t => t.stop());
  };

  patakaRecorder.start();
}

function stopRecordingPataka() {
  if (patakaRecorder && patakaRecorder.state === "recording") {
    patakaRecorder.stop();
    document.getElementById("prompt").innerText = "Stopping...";
  }
}

async function convertBlobToWavAndUpload(blob) {
  const arrayBuffer = await blob.arrayBuffer();
  const audioCtx = new AudioContext({ sampleRate: 48000 });
  const decoded = await audioCtx.decodeAudioData(arrayBuffer);

  // PCM float → WAV (16-bit)
  const wavBlob = pcmToWavBlob(decoded);

  uploadToServer(wavBlob);
}

function pcmToWavBlob(audioBuffer) {
  const numOfChannels = audioBuffer.numberOfChannels;
  const sampleRate = audioBuffer.sampleRate;
  const bitDepth = 16;

  let samples;
  if (numOfChannels === 2) {
    const left = audioBuffer.getChannelData(0);
    const right = audioBuffer.getChannelData(1);
    const length = left.length + right.length;
    const interleaved = new Float32Array(length);

    for (let i = 0; i < left.length; i++) {
      interleaved[i * 2] = left[i];
      interleaved[i * 2 + 1] = right[i];
    }
    samples = interleaved;
  } else {
    samples = audioBuffer.getChannelData(0);
  }

  const wavBuffer = encodeWAV(samples, sampleRate, numOfChannels, bitDepth);
  return new Blob([wavBuffer], { type: "audio/wav" });
}

function encodeWAV(samples, sampleRate, numChannels, bitDepth) {
  const bytesPerSample = bitDepth / 8;
  const blockAlign = numChannels * bytesPerSample;
  const buffer = new ArrayBuffer(44 + samples.length * bytesPerSample);
  const view = new DataView(buffer);

  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * bytesPerSample, true);
  writeString(view, 8, "WAVE");

  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitDepth, true);

  writeString(view, 36, "data");
  view.setUint32(40, samples.length * bytesPerSample, true);

  floatTo16BitPCM(view, 44, samples);

  return buffer;
}

function writeString(view, offset, text) {
  for (let i = 0; i < text.length; i++) {
    view.setUint8(offset + i, text.charCodeAt(i));
  }
}

function floatTo16BitPCM(view, offset, samples) {
  for (let i = 0; i < samples.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
}

async function uploadToServer(wavBlob) {
  const formData = new FormData();
  formData.append("file", wavBlob, "recording.wav");

  try {
    const response = await fetch("http://localhost:8000/process-pata-ka", {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    document.getElementById("prompt").innerText = "Analysis Complete";

    // Show syllables
    const sDiv = document.getElementById("syllables");
    sDiv.innerHTML = "";
    data.syllables.forEach(syl => {
      const p = document.createElement("p");
      p.textContent = `${syl.text} — ${syl.time}s`;
      sDiv.appendChild(p);
    });

    // Draw waveform PNG
    const img = new Image();
    img.onload = () => {
      const canvas = document.getElementById("waveCanvas");
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    };
    img.src = "data:image/png;base64," + data.waveform_png;

  } catch (err) {
    console.error(err);
    document.getElementById("prompt").innerText = "Error processing audio.";
  }
}



console.log("main.js loaded");
