"""
SankhyaVox — Feature Extraction Module
src/features.py

Single source of truth for all audio preprocessing and MFCC extraction.
Parameters MUST match the training notebook (sankhya.ipynb) exactly:
  SAMPLE_RATE = 16000
  N_MFCC      = 13
  HOP_MS      = 10   → hop_length = 160
  WIN_MS      = 25   → n_fft      = 400
  delta mode  = 'nearest'
  CMVN        = per-utterance zero-mean unit-variance
"""

import subprocess
import tempfile
import os
import numpy as np
import librosa
import warnings
warnings.filterwarnings("ignore")

# ── Constants (must match notebook) ──────────────────────────────────────────
SAMPLE_RATE = 16_000
N_MFCC      = 13
N_FEATURES  = 39   # 13 + delta + delta-delta
HOP_MS      = 10
WIN_MS      = 25


# ── Audio loading ─────────────────────────────────────────────────────────────

def _find_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def load_audio(path, target_sr=SAMPLE_RATE):
    """
    Load any audio file → mono float32 at target_sr.
    Handles: wav, m4a, mp3, webm, ogg (via ffmpeg fallback).
    """
    # Try direct load first
    try:
        audio, sr = librosa.load(path, sr=target_sr, mono=True)
        if len(audio) > 0:
            return audio.astype("float32"), target_sr
    except Exception:
        pass

    # Fallback: convert via ffmpeg (handles browser formats: webm, ogg)
    ffmpeg = _find_ffmpeg()
    wav_path = path + "._converted.wav"
    try:
        result = subprocess.run(
            [ffmpeg, "-y", "-i", path,
             "-ar", str(target_sr), "-ac", "1", "-f", "wav", wav_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=15,
        )
        if result.returncode == 0 and os.path.exists(wav_path):
            audio, sr = librosa.load(wav_path, sr=target_sr, mono=True)
            return audio.astype("float32"), target_sr
    except Exception:
        pass
    finally:
        if os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
            except OSError:
                pass

    raise ValueError(
        f"Cannot load audio from '{path}'. "
        "Ensure ffmpeg is installed for webm/ogg/mp3 support."
    )


def preprocess_audio(audio):
    """DC removal + peak normalisation to 95% (matches training notebook)."""
    audio = audio - np.mean(audio)
    peak  = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio / peak * 0.95
    return audio.astype("float32")


def extract_mfcc(audio, sr=SAMPLE_RATE):
    """
    Extract 39-dim MFCC [T × 39] matching the training notebook exactly.
    
    Steps:
      1. Pre-emphasis filter (0.97)
      2. librosa MFCC: n_mfcc=13, hop=160, n_fft=400
      3. Delta and delta-delta with mode='nearest'
      4. Per-utterance CMVN (zero mean, unit variance)
    """
    hop = int(sr * HOP_MS / 1000)   # 160 samples at 16kHz
    win = int(sr * WIN_MS / 1000)   # 400 samples at 16kHz

    # Pre-emphasis
    pre = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

    mfcc = librosa.feature.mfcc(
        y=pre, sr=sr, n_mfcc=N_MFCC,
        hop_length=hop, n_fft=win,
    )  # [13 × T]

    # mode='nearest' avoids crash when n_frames < 9 (short VAD segments)
    delta  = librosa.feature.delta(mfcc, mode='nearest')          # [13 × T]
    delta2 = librosa.feature.delta(mfcc, order=2, mode='nearest') # [13 × T]

    features = np.vstack([mfcc, delta, delta2]).T  # [T × 39]

    # Per-utterance CMVN
    mean = np.mean(features, axis=0, keepdims=True)
    std  = np.std(features,  axis=0, keepdims=True) + 1e-8
    return ((features - mean) / std).astype("float32")


def load_and_extract(path):
    """
    Convenience: file path → preprocessed MFCC [T × 39].
    Handles all audio formats via load_audio.
    """
    audio, sr = load_audio(path)
    audio     = preprocess_audio(audio)
    return extract_mfcc(audio, sr)