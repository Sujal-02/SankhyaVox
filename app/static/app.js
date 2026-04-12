/* ═══════════════════════════════════════════════════════════════════════
   SankhyaVox – Client-side logic
   ═══════════════════════════════════════════════════════════════════════ */

(function () {
  "use strict";

  // ── DOM refs ──────────────────────────────────────────────────────
  const checkpointSelect = document.getElementById("checkpoint-select");
  const modeSelect = document.getElementById("mode-select");
  const btnRecord = document.getElementById("btn-record");
  const btnSubmit = document.getElementById("btn-submit");
  const fileInput = document.getElementById("file-input");
  const resultLabel1 = document.getElementById("result-label-1");
  const resultDisplay1 = document.getElementById("result-display-1");
  const resultMeta1 = document.getElementById("result-meta-1");
  const resultDisplay2 = document.getElementById("result-display-2");
  const resultMeta2 = document.getElementById("result-meta-2");
  const rank2 = document.getElementById("rank-2");
  const resultDisplay3 = document.getElementById("result-display-3");
  const resultMeta3 = document.getElementById("result-meta-3");
  const rank3 = document.getElementById("rank-3");
  const statusBar = document.getElementById("status-bar");
  const playbackBar = document.getElementById("playback-bar");
  const audioOriginal = document.getElementById("audio-original");
  const audioProcessed = document.getElementById("audio-processed");
  let audioObjectUrl = null;   // revocable URL for playback

  // ── State ─────────────────────────────────────────────────────────
  let checkpointsList = [];   // [ "checkpoints/hmm_classifier.pkl", ... ]
  let audioBlob = null;       // File or recorded Blob
  let audioFileName = "";
  let mediaRecorder = null;
  let isRecording = false;

  // ── Helpers ───────────────────────────────────────────────────────
  function setStatus(msg, cls) {
    statusBar.className = "status-bar" + (cls ? " " + cls : "");
    statusBar.innerHTML = msg;
  }

  function enableSubmit() {
    btnSubmit.disabled = !audioBlob;
  }

  // ── Load checkpoints on init ──────────────────────────────────────
  async function loadCheckpoints() {
    try {
      const res = await fetch("/api/checkpoints");
      checkpointsList = await res.json();
      populateCheckpoints();
    } catch (e) {
      setStatus("Could not load checkpoints", "error");
    }
  }

  function populateCheckpoints() {
    checkpointSelect.innerHTML = '<option value="">default</option>';
    checkpointsList.forEach((f) => {
      const opt = document.createElement("option");
      opt.value = f;
      opt.textContent = f.split("/").pop();
      checkpointSelect.appendChild(opt);
    });
  }

  // ── File attach ───────────────────────────────────────────────────
  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (!file) return;
    audioBlob = file;
    audioFileName = file.name;
    setStatus("Attached: " + file.name);
    enableSubmit();
    // Reset recording state if any
    stopRecording(true);
  });

  // ── Recording ─────────────────────────────────────────────────────
  btnRecord.addEventListener("click", () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  });

  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const chunks = [];
      mediaRecorder = new MediaRecorder(stream, { mimeType: getSupportedMime() });

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };

      mediaRecorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        if (chunks.length) {
          audioBlob = new Blob(chunks, { type: mediaRecorder.mimeType });
          const ext = audioBlob.type.includes("webm") ? ".webm" : ".wav";
          audioFileName = "recording" + ext;
          setStatus("Recorded: " + (audioBlob.size / 1024).toFixed(1) + " KB");
          enableSubmit();
        }
      };

      mediaRecorder.start();
      isRecording = true;
      btnRecord.classList.add("recording");
      btnRecord.querySelector("span").textContent = "Stop";
      setStatus("Recording…", "loading");
    } catch (e) {
      setStatus("Microphone access denied", "error");
    }
  }

  function stopRecording(silent) {
    if (mediaRecorder && isRecording) {
      mediaRecorder.stop();
    }
    isRecording = false;
    btnRecord.classList.remove("recording");
    btnRecord.querySelector("span").textContent = "Record";
    if (silent) return;
  }

  function getSupportedMime() {
    const types = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/wav"];
    for (const t of types) {
      if (MediaRecorder.isTypeSupported(t)) return t;
    }
    return "";
  }

  // ── Submit / Decode ───────────────────────────────────────────────
  btnSubmit.addEventListener("click", submitDecode);

  async function submitDecode() {
    if (!audioBlob) return;

    btnSubmit.disabled = true;
    resultLabel1.textContent = "";
    resultDisplay1.className = "result-number";
    resultDisplay1.textContent = "—";
    resultMeta1.textContent = "";
    resultDisplay2.textContent = "—";
    resultMeta2.textContent = "";
    rank2.hidden = true;
    resultDisplay3.textContent = "—";
    resultMeta3.textContent = "";
    rank3.hidden = true;
    playbackBar.hidden = true;
    audioOriginal.pause();
    audioOriginal.removeAttribute("src");
    audioProcessed.pause();
    audioProcessed.removeAttribute("src");
    setStatus('<span class="spinner"></span>Decoding…', "loading");

    const formData = new FormData();
    formData.append("audio", audioBlob, audioFileName);
    formData.append("checkpoint", checkpointSelect.value);
    formData.append("mode", modeSelect.value);

    try {
      const res = await fetch("/api/decode", { method: "POST", body: formData });
      const data = await res.json();

      if (!res.ok || data.error) {
        resultDisplay1.className = "result-number error";
        resultDisplay1.textContent = data.error || "Unknown error";
        setStatus("Decoding failed", "error");
        return;
      }

      if (data.mode === "isolated") {
        resultLabel1.textContent = data.label;
        resultDisplay1.className = "result-number success";
        resultDisplay1.textContent = data.token;
        resultMeta1.textContent =
          "Label: " + data.label +
          "  •  Score: " + data.score;
        // Show rank 2 & 3 from top3
        if (data.top3 && data.top3.length >= 2) {
          resultDisplay2.className = "result-number success";
          resultDisplay2.textContent = data.top3[1][0];
          resultMeta2.textContent = "Score: " + data.top3[1][1].toFixed(4);
          rank2.hidden = false;
        }
        if (data.top3 && data.top3.length >= 3) {
          resultDisplay3.className = "result-number success";
          resultDisplay3.textContent = data.top3[2][0];
          resultMeta3.textContent = "Score: " + data.top3[2][1].toFixed(4);
          rank3.hidden = false;
        }
        setStatus("Isolated token recognised");
        showPlayback(data.processed_audio);
      } else if (data.result >= 0) {
        resultDisplay1.className = "result-number success";
        resultDisplay1.textContent = data.result;
        resultMeta1.textContent =
          "Tokens: " + data.tokens.join(" + ") +
          "  •  Score: " + data.score;
        setStatus("Recognised with HMM");
        showPlayback(data.processed_audio);
      } else {
        resultDisplay1.className = "result-number error";
        resultDisplay1.textContent = "?";
        resultMeta1.textContent = "Tokens: " + (data.tokens || []).join(" + ");
        setStatus("Recognition failed – no valid grammar match", "error");
        showPlayback(data.processed_audio);
      }
    } catch (e) {
      resultDisplay1.className = "result-number error";
      resultDisplay1.textContent = "Error";
      setStatus("Network error: " + e.message, "error");
    } finally {
      enableSubmit();
    }
  }

  // ── Audio playback ────────────────────────────────────────────────
  function showPlayback(processedUrl) {
    if (!audioBlob) return;
    if (audioObjectUrl) URL.revokeObjectURL(audioObjectUrl);
    audioObjectUrl = URL.createObjectURL(audioBlob);
    audioOriginal.src = audioObjectUrl;
    if (processedUrl) {
      audioProcessed.src = processedUrl;
    }
    playbackBar.hidden = false;
  }

  // ── Boot ──────────────────────────────────────────────────────────
  loadCheckpoints();
})();
