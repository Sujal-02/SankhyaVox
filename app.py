#!/usr/bin/env python3
"""
SankhyaVox Flask Server
app.py

Usage:
    python app.py
    Open: http://localhost:5000

Endpoints:
    GET  /            web UI
    POST /predict     predict digits from uploaded audio
    GET  /health      liveness check
    GET  /info        model metadata and per-token accuracy
    GET  /debug       verbose decode trace on uploaded audio (dev only)

Audio formats accepted: wav, m4a, mp3, webm, ogg (anything ffmpeg handles)
"""
import os
import json
import tempfile

from flask import Flask, request, jsonify, render_template

from src.inference import SankhyaVoxInference

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "best")

print("Loading SankhyaVox model...")
model = SankhyaVoxInference.load(MODEL_DIR)
print(f"Ready. Model: {model.label}\n")


# ── Helper: save uploaded file to temp, call model, clean up ─────────────────

def _predict_from_request(req, verbose=False):
    if "file" not in req.files:
        return {"error": "No audio file. Send as multipart form-data, key='file'."}, 400

    f = req.files["file"]
    if not f or f.filename == "":
        return {"error": "Empty filename."}, 400

    # Determine suffix safely (default .wav if no extension)
    fname  = f.filename or "audio.wav"
    ext    = fname.rsplit(".", 1)[-1].lower() if "." in fname else "wav"
    suffix = f".{ext}"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        f.save(tmp_path)

    try:
        result = model.predict_file(tmp_path, verbose=verbose)
    except Exception as e:
        return {"error": str(e)}, 500
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return result, 200


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept audio file and return recognition result.

    Returns JSON:
        {
          "number":     int,       # 0–99  (−1 on failure)
          "tokens":     list[str], # e.g. ["pancha", "dasha"]
          "score":      float,     # HMM log-likelihood
          "model":      str,
          "devanagari": str,       # e.g. "पञ्च–दश"
          "confidence": str,       # "High" / "Medium" / "Low" / "None"
          "success":    bool
        }
    """
    result, code = _predict_from_request(request)
    return jsonify(result), code


@app.route("/debug", methods=["POST"])
def debug():
    """
    Same as /predict but runs decoder in verbose mode.
    Useful during development to understand segmentation decisions.
    """
    result, code = _predict_from_request(request, verbose=True)
    return jsonify(result), code


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": model.label})


@app.route("/info")
def info():
    report_path = os.path.join(MODEL_DIR, "eval_report.json")
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as fh:
            report = json.load(fh)
    else:
        report = {"note": "eval_report.json not found in model directory"}
    return jsonify(report)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)