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
// === Updated createPane ===
function createPane(plotId, title, height = 200) {
  if (!panes) return null;

  // Wrapper pane
  const wrapper = document.createElement("div");
  wrapper.className = "pane";
  wrapper.id = "pane-" + plotId; // ✅ Added for feature toggle

  // Title
  const h = document.createElement("h4");
  h.textContent = title;
  wrapper.appendChild(h);

  // Plot div
  const plotDiv = document.createElement("div");
  plotDiv.id = plotId;
  plotDiv.style.width = "100%";
  plotDiv.style.height = height + "px";
  wrapper.appendChild(plotDiv);

  panes.appendChild(wrapper);
  return plotDiv;
}

// function createPane(plotId, title, height = 200) {
//   if (!panes) return null;
//   // create wrapper pane
//   const wrapper = document.createElement("div");
//   wrapper.className = "pane";
//   const h = document.createElement("h4");
//   h.textContent = title;
//   wrapper.appendChild(h);

//   const plotDiv = document.createElement("div");
//   plotDiv.id = plotId;
//   plotDiv.style.width = "100%";
//   plotDiv.style.height = height + "px";
//   wrapper.appendChild(plotDiv);
//   panes.appendChild(wrapper);
//   return plotDiv;
// }

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

// // Render all plots using server JSON
// function renderPlotsFromServer(json) {
//   clearPanes();

//   // Summary box
//   const summary = document.createElement("div");
//   summary.className = "summary-box pane";
//   summary.innerHTML = `
//     <strong>Duration:</strong> ${json.duration?.toFixed?.(3) ?? "n/a"} s<br/>
//     <strong>Sample rate:</strong> ${json.sample_rate ?? "n/a"}<br/>
//     <strong>Speech rate:</strong> ${(json.speech_rate_sps ? (json.speech_rate_sps * 60).toFixed(1) : "n/a")} syl/min<br/>
//     <strong>Jitter (local):</strong> ${json.jitter_local ?? "n/a"}<br/>
//     <strong>Shimmer (local):</strong> ${json.shimmer_local ?? "n/a"}<br/>
//   `;
//   panes.appendChild(summary);

//   // Waveform (local decoded buffer)
//   if (recordedFloat32) {
//     plotWaveformFromFloat32(recordedFloat32, sampleRate);
//   } else {
//     // create empty waveform pane
//     createPane("plot-waveform", "Waveform", 200);
//     Plotly.newPlot("plot-waveform", [{ x: [], y: [], mode: "lines" }], { margin: { t: 10 } });
//   }

//   // RMS
//   if (json.rms && json.rms.times && json.rms.values) {
//     createPane("plot-rms", "Short-time RMS", 200);
//     Plotly.newPlot("plot-rms", [{ x: json.rms.times, y: json.rms.values, mode: "lines" }], { margin: { t: 10 } });
//     attachPlotlyClick("plot-rms", "RMS");
//   }

//   // Loudness dB
//   if (json.loudness && json.loudness.times && json.loudness.values) {
//     createPane("plot-loudness", "Loudness (dB)", 200);
//     Plotly.newPlot("plot-loudness", [{ x: json.loudness.times, y: json.loudness.values, mode: "lines" }], { margin: { t: 10 }, yaxis: { title: "dB" } });
//     attachPlotlyClick("plot-loudness", "Loudness dB");
//     // update meter to last dB
//     const lastDb = json.loudness.values[json.loudness.values.length - 1];
//     if (typeof lastDb === "number") {
//       const norm = Math.min(1, Math.max(0, (lastDb + 60) / 60));
//       meterFill.style.width = (norm * 100) + "%";
//     }
//   }

//   // Pitch
//   if (json.pitch_times && json.pitch_vals) {
//     createPane("plot-pitch", "Pitch (Hz)", 200);
//     Plotly.newPlot("plot-pitch", [{ x: json.pitch_times, y: json.pitch_vals, mode: "lines+markers" }], { margin: { t: 10 }, yaxis: { title: "Hz" } });
//     attachPlotlyClick("plot-pitch", "Pitch");
//   }

//   // Jitter
//   if (json.jitter && json.jitter.times && json.jitter.values) {
//     createPane("plot-jitter", "Jitter (per-voiced)", 200);
//     Plotly.newPlot("plot-jitter", [{ x: json.jitter.times, y: json.jitter.values, mode: "lines+markers" }], { margin: { t: 10 } });
//     attachPlotlyClick("plot-jitter", "Jitter");
//   }

//   // Shimmer
//   if (json.shimmer && json.shimmer.times && json.shimmer.values) {
//     createPane("plot-shimmer", "Shimmer (per-voiced)", 200);
//     Plotly.newPlot("plot-shimmer", [{ x: json.shimmer.times, y: json.shimmer.values, mode: "lines+markers" }], { margin: { t: 10 } });
//     attachPlotlyClick("plot-shimmer", "Shimmer");
//   }

//   // ZCR
//   if (json.zcr && json.zcr.times && json.zcr.values) {
//     createPane("plot-zcr", "Zero Crossing Rate", 200);
//     Plotly.newPlot("plot-zcr", [{ x: json.zcr.times, y: json.zcr.values, mode: "lines" }], { margin: { t: 10 } });
//     attachPlotlyClick("plot-zcr", "ZCR");
//   }

//   // Spectral Centroid
//   if (json.spectralCentroid && json.spectralCentroid.times && json.spectralCentroid.values) {
//     createPane("plot-spectralCentroid", "Spectral Centroid (Hz)", 200);
//     Plotly.newPlot("plot-spectralCentroid", [{ x: json.spectralCentroid.times, y: json.spectralCentroid.values, mode: "lines" }], { margin: { t: 10 } });
//     attachPlotlyClick("plot-spectralCentroid", "Spectral Centroid");
//   }
// }


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
featureList.addEventListener('click', e=>{
  const it = e.target.closest('.feature-item'); if(!it) return;
  const fid = it.dataset.feature; it.classList.toggle('active'); const pane = document.getElementById('pane-'+fid);
  if(pane) pane.style.display = it.classList.contains('active') ? 'block' : 'none';
});

console.log("main.js loaded");
