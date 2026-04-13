"""
SankhyaVox – Flask Web Application.

Provides a web UI for Sanskrit numeral recognition:
  - Optional HMM checkpoint selection
  - Upload or record audio
  - Display recognised integer (0–99)

Usage:
    python app/server.py
"""

import os
import sys
import glob
import tempfile
import traceback
import uuid
from pathlib import Path

import soundfile as sf
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.hmm_classifier import SankhyaHMM
from src.decoder import decode_audio

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

UPLOAD_DIR = PROJECT_ROOT / "temp" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
DEFAULT_CHECKPOINT = str(CHECKPOINTS_DIR / "hmm_classifier_4_4.pkl")

# Cache: (checkpoint_path, loaded model)
_hmm_cache: dict = {"path": DEFAULT_CHECKPOINT, "model": SankhyaHMM(checkpoint_path=DEFAULT_CHECKPOINT)}


def get_hmm(checkpoint: str) -> SankhyaHMM:
    """Return a cached HMM, reloading only when the checkpoint changes."""
    ckpt = checkpoint if checkpoint else DEFAULT_CHECKPOINT
    if ckpt != _hmm_cache["path"]:
        _hmm_cache["path"] = ckpt
        _hmm_cache["model"] = SankhyaHMM(checkpoint_path=ckpt)
    return _hmm_cache["model"]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/checkpoints", methods=["GET"])
def list_checkpoints():
    """Return available HMM checkpoint files."""
    pattern = str(CHECKPOINTS_DIR / "*hmm*.pkl")
    files = sorted(glob.glob(pattern))
    result = [os.path.relpath(f, PROJECT_ROOT).replace("\\", "/") for f in files]
    return jsonify(result)


@app.route("/temp/uploads/<filename>")
def serve_temp(filename):
    """Serve a temporary audio file."""
    return send_from_directory(str(UPLOAD_DIR), filename)


@app.route("/api/decode", methods=["POST"])
def decode():
    """Decode an uploaded audio file using the HMM classifier."""
    checkpoint = request.form.get("checkpoint", "").strip()
    mode = request.form.get("mode", "compound").strip()
    isolated = mode == "isolated"

    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save uploaded file to temp
    suffix = Path(audio_file.filename).suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(
        dir=str(UPLOAD_DIR), suffix=suffix, delete=False
    )
    try:
        audio_file.save(tmp.name)
        tmp.close()

        result = decode_audio(
            get_hmm(checkpoint),
            audio_path=tmp.name,
            verbose=False,
            isolated=isolated,
            return_audio=True,
        )

        # Last element is the preprocessed audio waveform
        preprocessed_audio = result[-1]

        # Save preprocessed audio as a servable temp WAV
        proc_name = f"proc_{uuid.uuid4().hex[:12]}.wav"
        proc_path = str(UPLOAD_DIR / proc_name)
        from src.config import SAMPLE_RATE
        sf.write(proc_path, preprocessed_audio, SAMPLE_RATE, subtype="PCM_16")
        proc_url = f"/temp/uploads/{proc_name}"

        if isolated:
            token: str = result[0]  # type: ignore[assignment]
            label: int = result[1]  # type: ignore[assignment]
            debug: dict = result[2]  # type: ignore[assignment]
            return jsonify({
                "mode": "isolated",
                "token": token,
                "label": label,
                "score": debug["best_score"],
                "ranked": debug["ranked"],
                "processed_audio": proc_url,
            })
        else:
            integer: int = result[0]  # type: ignore[assignment]
            tokens: list = result[1]  # type: ignore[assignment]
            debug: dict = result[2]  # type: ignore[assignment]
            return jsonify({
                "mode": "compound",
                "result": integer,
                "tokens": tokens,
                "score": debug["viterbi_score"],
                "active_windows": debug["active_windows"],
                "processed_audio": proc_url,
            })

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Decoding failed: {str(e)}"}), 500
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
