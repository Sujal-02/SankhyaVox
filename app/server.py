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
from pathlib import Path

from flask import Flask, jsonify, render_template, request

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.demo_decode_sound import decode_audio

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

UPLOAD_DIR = PROJECT_ROOT / "temp" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


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


@app.route("/api/decode", methods=["POST"])
def decode():
    """Decode an uploaded audio file using the HMM classifier."""
    checkpoint = request.form.get("checkpoint", "").strip()

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

        ckpt = checkpoint if checkpoint else None

        result, tokens, debug = decode_audio(
            audio_path=tmp.name,
            checkpoint_path=ckpt,
            verbose=False,
        )

        return jsonify({
            "result": result,
            "tokens": tokens,
            "score": debug["viterbi_score"],
            "active_windows": debug["active_windows"],
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
