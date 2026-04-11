"""
SankhyaVox – Central project configuration.
All hyperparameters, paths, and constants are defined here for easy tuning.
"""

from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# PROJECT PATHS
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT   = Path(__file__).resolve().parent.parent

# DVC-tracked raw human recordings (immutable source, never modified in-place)
DATA_DIR       = PROJECT_ROOT / "data"

# Runtime-generated processed outputs (git-ignored, regenerable)
PROCESSED_DIR  = PROJECT_ROOT / "data_processed"

# Human data paths (source → convert → segment → features)
HUMAN_RAW_DIR  = PROCESSED_DIR / "human" / "raw"        # converted 16 kHz mono WAVs
SEGMENT_DIR    = PROCESSED_DIR / "human" / "segments"    # individual utterance clips
FEATURE_DIR    = PROCESSED_DIR / "human" / "features"    # MFCC .npy files

# TTS data paths (generate → features; no segmentation needed)
TTS_DIR        = PROCESSED_DIR / "tts"
TTS_SEGMENT_DIR = TTS_DIR / "segments"
TTS_FEATURE_DIR = TTS_DIR / "features"

# Augmented data paths (reserved for future noise/pitch augmentation)
AUG_SEGMENT_DIR = PROCESSED_DIR / "augmented" / "segments"
AUG_FEATURE_DIR = PROCESSED_DIR / "augmented" / "features"

# Trained model artifacts and evaluation outputs
MODEL_DIR      = PROJECT_ROOT / "models"
RESULTS_DIR    = PROJECT_ROOT / "results"

# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLE_RATE      = 16_000     # Hz – target sample rate for all audio
PRE_EMPHASIS     = 0.97       # Pre-emphasis filter coefficient
PEAK_NORM_DBFS   = -3.0       # Peak normalisation target (dBFS)

# ═══════════════════════════════════════════════════════════════════════════════
# MFCC FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

FRAME_LENGTH_MS    = 25       # ms per frame
FRAME_SHIFT_MS     = 10       # ms hop between frames
FRAME_LENGTH       = int(SAMPLE_RATE * FRAME_LENGTH_MS / 1000)   # 400 samples
FRAME_SHIFT        = int(SAMPLE_RATE * FRAME_SHIFT_MS  / 1000)   # 160 samples
N_FFT              = 512
N_MELS             = 26       # Mel filter banks
N_MFCC             = 13       # Cepstral coefficients (c0 … c12)
USE_DELTAS         = True     # Append Δ and ΔΔ → 39 dims total
FEATURE_DIM        = N_MFCC * (3 if USE_DELTAS else 1)   # 39
APPLY_CMVN         = True     # Per-utterance cepstral mean-variance normalisation

# ═══════════════════════════════════════════════════════════════════════════════
# SEGMENTATION (energy-based VAD for splitting repeated-utterance recordings)
# ═══════════════════════════════════════════════════════════════════════════════

EXPECTED_REPS       = 10      # Repetitions per raw recording file
ENERGY_THRESHOLD    = 0.3     # Adaptive threshold ratio between noise floor and peak
MIN_SPEECH_DUR_S    = 0.20    # Minimum speech segment duration (seconds)
MIN_SILENCE_DUR_S   = 0.40    # Minimum silence gap between repetitions (seconds)
PAD_MS              = 80      # Silence padding kept around each segment (ms)
HIGHPASS_CUTOFF     = 80      # Hz – high-pass filter cutoff for DC removal

# ═══════════════════════════════════════════════════════════════════════════════
# SEGMENT QA THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════

QA_MIN_DURATION     = 0.15    # Segments shorter than this are likely errors (sec)
QA_MAX_DURATION     = 1.50    # Segments longer than this may contain >1 word (sec)

# ═══════════════════════════════════════════════════════════════════════════════
# VOCABULARY
# The 13 base Sanskrit tokens that compose all numbers 0–99
# ═══════════════════════════════════════════════════════════════════════════════

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
    "shata",     # 100
]

TOKEN_TO_IDX = {tok: i for i, tok in enumerate(VOCAB)}
IDX_TO_TOKEN = {i: tok for i, tok in enumerate(VOCAB)}

# ═══════════════════════════════════════════════════════════════════════════════
# NUMERIC ID MAPPING
# Zero-padded 3-digit IDs used in the processed file naming convention:
#   <SpeakerId>_<numericId>_<rep>.wav   e.g. S01_001_03.wav
# ═══════════════════════════════════════════════════════════════════════════════

TOKEN_TO_NUMERIC_ID = {
    "shunya":  "000",
    "eka":     "001",
    "dvi":     "002",
    "tri":     "003",
    "catur":   "004",
    "pancha":  "005",
    "shat":    "006",
    "sapta":   "007",
    "ashta":   "008",
    "nava":    "009",
    "dasha":   "010",
    "vimsati": "020",
    "shata":   "100",
}
NUMERIC_ID_TO_TOKEN = {v: k for k, v in TOKEN_TO_NUMERIC_ID.items()}

# Numeric value each token represents (used as classification labels)
TOKEN_TO_VALUE = {
    "shunya": 0, "eka": 1, "dvi": 2, "tri": 3, "catur": 4,
    "pancha": 5, "shat": 6, "sapta": 7, "ashta": 8, "nava": 9,
    "dasha": 10, "vimsati": 20, "shata": 100,
}
VALUE_TO_TOKEN = {v: k for k, v in TOKEN_TO_VALUE.items()}

# ═══════════════════════════════════════════════════════════════════════════════
# TTS GENERATION
# Voice IDs and augmentation ranges for Edge TTS synthetic data
# ═══════════════════════════════════════════════════════════════════════════════

TTS_VOICES = {
    "TTS01": "en-IN-PrabhatNeural",
    "TTS02": "en-IN-NeerjaNeural",
    "TTS03": "en-US-GuyNeural",
    "TTS04": "en-US-AriaNeural",
}
TTS_RATES   = ["-10%", "-5%", "+0%", "+5%", "+10%"]
TTS_PITCHES = ["-2Hz", "-1Hz", "+0Hz", "+1Hz", "+2Hz"]
TTS_REPS    = 10   # Repetitions per token per voice

# ═══════════════════════════════════════════════════════════════════════════════
# AUGMENTATION
# Pitch shifts (semitones) and speed factors whose permutations are applied
# ═══════════════════════════════════════════════════════════════════════════════

AUG_PITCHES = [-2, 0, +2]         # Semitones (0 = unchanged pitch)
AUG_SPEEDS  = [0.9, 1.0, 1.1]    # Speed/rate factors (1.0 = unchanged speed)

# ═══════════════════════════════════════════════════════════════════════════════
# HMM MODEL
# Number of emitting states per word model (matched to phoneme complexity)
# ═══════════════════════════════════════════════════════════════════════════════

# State count per token — sized to phonetic complexity
HMM_STATES = {
    "shunya":  15,
    "eka":     9,
    "dvi":     9,
    "tri":     9,
    "catur":   15,
    "pancha":  15,
    "shat":    9,
    "sapta":   15,
    "ashta":   12,
    "nava":    12,
    "dasha":   12,
    "vimsati": 21,
    "shata":   12,
}

# Per-token GMM mixtures — more for confused tokens
GMM_MIXTURES = {
    "shunya":  3,
    "eka":     3,
    "dvi":     3,
    "tri":     3,
    "catur":   3,
    "pancha":  3,
    "shat":    3,
    "sapta":   3,
    "ashta":   3,
    "nava":    3,
    "dasha":   3,
    "vimsati": 3,
    "shata":   3,
}

BAUM_WELCH_ITERS   = 150       # EM iterations for training
CONVERGENCE_THRESH = 1e-4     # Log-likelihood convergence threshold

# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE MODEL SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

KNN_K              = 5        # k for k-NN (tuned via CV)
SVM_KERNEL         = "rbf"
SVM_C_RANGE        = [0.1, 1, 10, 100]
SVM_GAMMA_RANGE    = [1e-4, 1e-3, 1e-2, 0.1]

# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

N_CV_FOLDS         = 5        # Speaker-out cross-validation folds
TRAIN_SPEAKERS     = 7
VAL_SPEAKERS       = 2
TEST_SPEAKERS      = 1
