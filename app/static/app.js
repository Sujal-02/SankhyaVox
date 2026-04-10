/* ═══════════════════════════════════════════════════════════════════════
   SankhyaVox – Client-side logic
   ═══════════════════════════════════════════════════════════════════════ */

(function () {
  "use strict";

  // ── DOM refs ──────────────────────────────────────────────────────
  const checkpointSelect = document.getElementById("checkpoint-select");
  const btnRecord = document.getElementById("btn-record");
  const btnSubmit = document.getElementById("btn-submit");
  const fileInput = document.getElementById("file-input");
  const resultDisplay = document.getElementById("result-display");
  const resultMeta = document.getElementById("result-meta");
  const statusBar = document.getElementById("status-bar");
  const playbackBar = document.getElementById("playback-bar");
  const audioPlayer = document.getElementById("audio-player");
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
    resultDisplay.className = "result-number";
    resultDisplay.textContent = "—";
    resultMeta.textContent = "";
    playbackBar.hidden = true;
    audioPlayer.pause();
    audioPlayer.removeAttribute("src");
    setStatus('<span class="spinner"></span>Decoding…', "loading");

    const formData = new FormData();
    formData.append("audio", audioBlob, audioFileName);
    formData.append("checkpoint", checkpointSelect.value);

    try {
      const res = await fetch("/api/decode", { method: "POST", body: formData });
      const data = await res.json();

      if (!res.ok || data.error) {
        resultDisplay.className = "result-number error";
        resultDisplay.textContent = data.error || "Unknown error";
        setStatus("Decoding failed", "error");
        return;
      }

      if (data.result >= 0) {
        resultDisplay.className = "result-number success";
        resultDisplay.textContent = data.result;
        resultMeta.textContent =
          "Tokens: " + data.tokens.join(" + ") +
          "  •  Score: " + data.score;
        setStatus("Recognised with HMM");
        showPlayback();
      } else {
        resultDisplay.className = "result-number error";
        resultDisplay.textContent = "?";
        resultMeta.textContent = "Tokens: " + (data.tokens || []).join(" + ");
        setStatus("Recognition failed – no valid grammar match", "error");
        showPlayback();
      }
    } catch (e) {
      resultDisplay.className = "result-number error";
      resultDisplay.textContent = "Error";
      setStatus("Network error: " + e.message, "error");
    } finally {
      enableSubmit();
    }
  }

  // ── Audio playback ────────────────────────────────────────────────
  function showPlayback() {
    if (!audioBlob) return;
    if (audioObjectUrl) URL.revokeObjectURL(audioObjectUrl);
    audioObjectUrl = URL.createObjectURL(audioBlob);
    audioPlayer.src = audioObjectUrl;
    playbackBar.hidden = false;
  }

  // ── Boot ──────────────────────────────────────────────────────────
  loadCheckpoints();
})();
