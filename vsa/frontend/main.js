console.log("main.js loaded");

let isRecording = false;
let audioContext, processor, inputNode, stream;
let pcmChunks = [];

const vowelData = {}; // store F1/F2 for A, I, U

const recordBtn = document.getElementById("recordBtn");
const vowelSelect = document.getElementById("vowel");

// ------------------ BUTTON CLICK --------------------
recordBtn.onclick = async () => {
    const vowel = vowelSelect.value;

    if (!isRecording) {
        await startRecording(vowel);
    } else {
        await stopRecording(vowel);
    }
};

// ------------------ START RECORDING (WAV) --------------------
async function startRecording(vowel) {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioContext = new AudioContext({ sampleRate: 44100 });

    inputNode = audioContext.createMediaStreamSource(stream);
    processor = audioContext.createScriptProcessor(4096, 1, 1);

    pcmChunks = [];

    processor.onaudioprocess = (e) => {
        const data = e.inputBuffer.getChannelData(0);
        pcmChunks.push(new Float32Array(data)); // store PCM
    };

    inputNode.connect(processor);
    processor.connect(audioContext.destination);

    isRecording = true;
    recordBtn.textContent = "⏹️ Stop Recording";
    document.getElementById("status").textContent = `Recording… Say /${vowel}/`;
}

// ------------------ STOP RECORDING + UPLOAD --------------------
async function stopRecording(vowel) {
    processor.disconnect();
    inputNode.disconnect();
    stream.getTracks().forEach((t) => t.stop());

    isRecording = false;
    recordBtn.textContent = "🎙️ Start Recording";
    document.getElementById("status").textContent = "Processing…";

    // Convert PCM → WAV blob
    const wavBlob = encodeWAV(pcmChunks, audioContext.sampleRate);

    const formData = new FormData();
    formData.append("file", wavBlob, `vowel_${vowel}.wav`);
    formData.append("vowel", vowel);

    try {
        const res = await fetch("http://127.0.0.1:8000/analyze_vowel", {
            method: "POST",
            body: formData,
        });

        const data = await res.json();

        // Save F1/F2
        vowelData[vowel] = { F1: data.F1_Hz, F2: data.F2_Hz };

        document.getElementById("resultsBox").textContent =
            JSON.stringify(vowelData, null, 2);

        plotVowelTriangle();
        document.getElementById("status").textContent = "Done ✓";

    } catch (err) {
        console.error(err);
        document.getElementById("status").textContent = "Error uploading audio";
    }
}

// ------------------ WAV ENCODER --------------------
function encodeWAV(chunks, sampleRate) {
    const totalSamples = chunks.reduce((s, c) => s + c.length, 0);
    const buffer = new ArrayBuffer(44 + totalSamples * 2);
    const view = new DataView(buffer);

    // Header
    writeString(view, 0, "RIFF");
    view.setUint32(4, 36 + totalSamples * 2, true);
    writeString(view, 8, "WAVE");
    writeString(view, 12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, 1, true); // mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, "data");
    view.setUint32(40, totalSamples * 2, true);

    // PCM data
    let offset = 44;
    chunks.forEach(chunk => {
        for (let i = 0; i < chunk.length; i++, offset += 2) {
            let s = Math.max(-1, Math.min(1, chunk[i]));
            view.setInt16(offset, s * 0x7FFF, true);
        }
    });

    return new Blob([buffer], { type: "audio/wav" });
}

function writeString(view, offset, str) {
    for (let i = 0; i < str.length; i++) {
        view.setUint8(offset + i, str.charCodeAt(i));
    }
}

// ------------------ PLOT TRIANGLE --------------------
function plotVowelTriangle() {
    if (!vowelData["a"] || !vowelData["i"] || !vowelData["u"]) return;

    const vowels = ["a", "i", "u"];
    const F1 = vowels.map(v => vowelData[v].F1);
    const F2 = vowels.map(v => vowelData[v].F2);

    const trace = {
        x: F2,       // F2 on X
        y: F1,       // F1 on Y
        mode: "markers+lines+text",
        text: vowels.map(v => "/" + v + "/"),
        textposition: "top center",
        marker: { size: 12 },
        line: { shape: "linear" }
    };

    const layout = {
        title: "Vowel Space Area (A–I–U)",
        xaxis: { title: "F2 (Hz)", autorange: "reversed" },
        yaxis: { title: "F1 (Hz)", autorange: "reversed" }
    };

    Plotly.newPlot("plot", [trace], layout);
}
