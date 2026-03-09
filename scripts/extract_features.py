"""
SankhyaVox – MFCC Feature Extraction.

Extracts 39-dim MFCC vectors (13 static + 13Δ + 13ΔΔ) with optional CMVN.

Usage:
    python scripts/extract_features.py                 # batch all segments
    python scripts/extract_features.py --input file.wav
"""

import argparse
import glob
import os
import sys

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import (
    APPLY_CMVN,
    FEATURE_DIR,
    FRAME_LENGTH,
    FRAME_SHIFT,
    N_FFT,
    N_MELS,
    N_MFCC,
    PRE_EMPHASIS,
    SAMPLE_RATE,
    SEGMENT_DIR,
    USE_DELTAS,
)


def preprocess_audio(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Audio pre-processing pipeline:
    1. DC offset removal (subtract mean)
    2. Pre-emphasis filter
    3. Peak normalisation to -3 dBFS
    """
    # DC offset removal
    audio = audio - np.mean(audio)

    # Pre-emphasis
    audio = np.append(audio[0], audio[1:] - PRE_EMPHASIS * audio[:-1])

    # Peak normalisation to -3 dBFS
    peak = np.max(np.abs(audio))
    if peak > 0:
        target_peak = 10 ** (-3.0 / 20.0)  # -3 dBFS ≈ 0.708
        audio = audio * (target_peak / peak)

    return audio.astype(np.float32)


def extract_mfcc(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    n_fft: int = N_FFT,
    hop_length: int = FRAME_SHIFT,
    win_length: int = FRAME_LENGTH,
    n_mels: int = N_MELS,
    use_deltas: bool = USE_DELTAS,
    apply_cmvn: bool = APPLY_CMVN,
) -> np.ndarray:
    """
    Extract MFCC features from preprocessed audio.

    Returns
    -------
    features : np.ndarray of shape (n_frames, feature_dim)
        39-dim by default (13 MFCC + 13Δ + 13ΔΔ).
    """
    # Compute MFCCs
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        window="hamming",
    )  # shape: (n_mfcc, n_frames)

    features = mfcc

    # Append delta and delta-delta
    if use_deltas:
        delta = librosa.feature.delta(mfcc, order=1)
        delta2 = librosa.feature.delta(mfcc, order=2)
        features = np.vstack([mfcc, delta, delta2])  # (39, n_frames)

    # Transpose to (n_frames, feature_dim)
    features = features.T

    # Cepstral Mean-Variance Normalisation (per-utterance)
    if apply_cmvn:
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True)
        std[std < 1e-10] = 1e-10  # prevent division by zero
        features = (features - mean) / std

    return features.astype(np.float32)


def process_file(wav_path: str, output_dir: str) -> str:
    """Extract features from a single WAV file and save as .npy."""
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    audio = preprocess_audio(audio, SAMPLE_RATE)
    features = extract_mfcc(audio, SAMPLE_RATE)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(wav_path))[0]
    out_path = os.path.join(output_dir, f"{basename}.npy")
    np.save(out_path, features)
    return out_path


def batch_extract(seg_dir: str = str(SEGMENT_DIR), feat_dir: str = str(FEATURE_DIR)):
    """Extract features for all segmented WAV files."""
    wavs = sorted(glob.glob(f"{seg_dir}/**/*.wav", recursive=True))
    if not wavs:
        print(f"No WAV files found in {seg_dir}")
        return

    count = 0
    for wav_path in wavs:
        # Mirror directory structure
        rel = os.path.relpath(os.path.dirname(wav_path), seg_dir)
        out_dir = os.path.join(feat_dir, rel)
        out_path = process_file(wav_path, out_dir)
        count += 1
        if count % 50 == 0:
            print(f"  Processed {count} files...")

    print(f"\nDone. Extracted features for {count} files → {feat_dir}")


def main():
    parser = argparse.ArgumentParser(description="SankhyaVox MFCC Feature Extraction")
    parser.add_argument("--input", type=str, help="Single WAV file")
    parser.add_argument("--seg-dir", type=str, default=str(SEGMENT_DIR))
    parser.add_argument("--feat-dir", type=str, default=str(FEATURE_DIR))
    args = parser.parse_args()

    if args.input:
        out = process_file(args.input, args.feat_dir)
        print(f"Features saved to {out}")
    else:
        batch_extract(args.seg_dir, args.feat_dir)


if __name__ == "__main__":
    main()
