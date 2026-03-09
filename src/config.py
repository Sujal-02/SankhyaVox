"""
SankhyaVox – Central project configuration.
All hyperparameters and paths are defined here for easy tuning.
"""

import os
from pathlib import Path

# ── Project Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
RAW_DIR      = DATA_DIR / "raw"          # data/raw/<SpeakerID>/<SpeakerID>_<token>_raw.wav
SEGMENT_DIR  = DATA_DIR / "segments"     # data/segments/<SpeakerID>/<SpeakerID>_<token>_<rep>.wav
FEATURE_DIR  = DATA_DIR / "features"     # data/features/<SpeakerID>/<SpeakerID>_<token>_<rep>.npy
MODEL_DIR    = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results"

# ── Audio Settings ─────────────────────────────────────────────────────────────
SAMPLE_RATE      = 16_000     # Hz – target sample rate for all audio
PRE_EMPHASIS     = 0.97       # Pre-emphasis filter coefficient
PEAK_NORM_DBFS   = -3.0       # Peak normalisation target (dBFS)

# ── MFCC Feature Settings ─────────────────────────────────────────────────────
FRAME_LENGTH_MS    = 25       # ms
FRAME_SHIFT_MS     = 10       # ms
FRAME_LENGTH       = int(SAMPLE_RATE * FRAME_LENGTH_MS / 1000)   # 400 samples
FRAME_SHIFT        = int(SAMPLE_RATE * FRAME_SHIFT_MS  / 1000)   # 160 samples
N_FFT              = 512
N_MELS             = 26       # Mel filter banks
N_MFCC             = 13       # Cepstral coefficients (c0 .. c12)
USE_DELTAS         = True     # Append Δ and ΔΔ → 39 dims
FEATURE_DIM        = N_MFCC * (3 if USE_DELTAS else 1)   # 39
APPLY_CMVN         = True     # Cepstral Mean-Variance Normalisation

# ── Segmentation Settings ─────────────────────────────────────────────────────
EXPECTED_REPS       = 10
ENERGY_THRESHOLD    = 0.015   # Relative to 95th-percentile RMS
MIN_SPEECH_DUR_S    = 0.20    # Minimum speech segment duration (seconds)
MIN_SILENCE_DUR_S   = 0.40    # Minimum silence gap between repetitions
PAD_MS              = 80      # Silence padding kept around each segment
HIGHPASS_CUTOFF     = 80      # Hz – high-pass filter cutoff

# ── Vocabulary ─────────────────────────────────────────────────────────────────
VOCAB = [
    "shunya",    # 0
    "eka",       # 1
    "dvi",       # 2
    "tri",       # 3
    "catur",     # 4
    "pancha",    # 5
    "shat",      # 6
    "sapta",     # 7
    "ashta",     # 8
    "nava",      # 9
    "dasha",     # 10
    "vimsati",   # 20
    "shata",     # 100  (for future 0-999 extension)
]

TOKEN_TO_IDX = {tok: i for i, tok in enumerate(VOCAB)}
IDX_TO_TOKEN = {i: tok for i, tok in enumerate(VOCAB)}

# ── HMM Settings ──────────────────────────────────────────────────────────────
# Number of emitting states per word model (matches phoneme complexity)
HMM_STATES = {
    "shunya":  6,
    "eka":     5,
    "dvi":     5,
    "tri":     5,
    "catur":   7,
    "pancha":  7,
    "shat":    5,
    "sapta":   7,
    "ashta":   7,
    "nava":    6,
    "dasha":   6,
    "vimsati": 9,
    "shata":   6,
    "SIL":     3,
}

GMM_MIXTURES       = 1        # Start with 1; increase to 3 or 5 if data allows
BAUM_WELCH_ITERS   = 15       # EM iterations for training
CONVERGENCE_THRESH = 1e-4     # Log-likelihood convergence threshold

# ── Baseline Model Settings ───────────────────────────────────────────────────
KNN_K              = 5        # k for k-NN (tuned via CV)
SVM_KERNEL         = "rbf"
SVM_C_RANGE        = [0.1, 1, 10, 100]
SVM_GAMMA_RANGE    = [1e-4, 1e-3, 1e-2, 0.1]

# ── Evaluation Settings ──────────────────────────────────────────────────────
N_CV_FOLDS         = 5        # Speaker-out cross-validation folds
TRAIN_SPEAKERS     = 7
VAL_SPEAKERS       = 2
TEST_SPEAKERS      = 1
