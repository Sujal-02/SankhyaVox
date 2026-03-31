"""
SankhyaVox – Automated Audio Segmentation Pipeline.

Takes raw repeated-utterance recordings and splits them into individual
labelled utterances using energy-based Voice Activity Detection (VAD).

Usage:
    python scripts/segment.py                          # batch-process all raw files
    python scripts/segment.py --input data/raw/S01/S01_eka_raw.wav
"""

import argparse
import glob
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import (
    ENERGY_THRESHOLD,
    EXPECTED_REPS,
    HIGHPASS_CUTOFF,
    MIN_SILENCE_DUR_S,
    MIN_SPEECH_DUR_S,
    PAD_MS,
    RAW_DIR,
    SAMPLE_RATE,
    SEGMENT_DIR,
)


def butter_highpass(cutoff: float = HIGHPASS_CUTOFF, fs: int = SAMPLE_RATE, order: int = 4):
    """Design a Butterworth high-pass filter."""
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="high")
    return b, a


def segment_utterances(
    input_path: str,
    speaker_id: str,
    token_name: str,
    output_dir: str,
    expected_reps: int = EXPECTED_REPS,
    energy_threshold: float = ENERGY_THRESHOLD,
    min_speech_dur: float = MIN_SPEECH_DUR_S,
    min_silence_dur: float = MIN_SILENCE_DUR_S,
    target_fs: int = SAMPLE_RATE,
    pad_ms: int = PAD_MS,
) -> int:
    """
    Segment a raw repeated-utterance recording into individual WAV files.

    Parameters
    ----------
    input_path : path to the raw WAV file
    speaker_id : e.g. "S01"
    token_name : e.g. "eka"
    output_dir : directory to write individual segments

    Returns
    -------
    Number of segments extracted.
    """
    audio, fs = sf.read(input_path)

    # Convert to mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if fs != target_fs:
        from scipy.signal import resample_poly
        audio = resample_poly(audio, target_fs, fs).astype(np.float32)
        fs = target_fs

    # High-pass filter to remove DC / low-frequency rumble
    b, a = butter_highpass(cutoff=HIGHPASS_CUTOFF, fs=fs)
    audio = filtfilt(b, a, audio).astype(np.float32)

    # Short-time RMS energy (25 ms frames, 10 ms shift)
    frame_len = int(0.025 * fs)
    frame_shift = int(0.010 * fs)
    frames = [audio[i : i + frame_len] for i in range(0, len(audio) - frame_len, frame_shift)]
    rms = np.array([np.sqrt(np.mean(f ** 2)) for f in frames])

    # Adaptive threshold grid search to find EXACTLY expected_reps
    from scipy.ndimage import binary_closing, binary_opening
    
    best_segs = []
    best_diff = float('inf')
    found_exact = False
    
    for min_sil in [0.4, 0.3, 0.2, 0.15, 0.1]:
        for thresh_mult in np.linspace(0.01, 1.0, 100):
            threshold = thresh_mult * np.percentile(rms, 95)
            is_speech_test = rms > threshold
            
            kernel_close = max(1, int(min_sil / 0.010))
            is_speech_test = binary_closing(is_speech_test, structure=np.ones(kernel_close))
            
            kernel_open = max(1, int(min_speech_dur / 0.010))
            is_speech_test = binary_opening(is_speech_test, structure=np.ones(kernel_open))
            
            boundaries = []
            in_seg = False
            start = 0
            for i, s in enumerate(is_speech_test):
                if s and not in_seg:
                    start = max(0, i - int(pad_ms / 10))
                    in_seg = True
                elif not s and in_seg:
                    end = min(len(is_speech_test), i + int(pad_ms / 10))
                    boundaries.append((start, end))
                    in_seg = False
            if in_seg:
                boundaries.append((start, len(is_speech_test)))
                
            if len(boundaries) == expected_reps:
                best_segs = boundaries
                found_exact = True
                break
                
            if abs(len(boundaries) - expected_reps) < best_diff:
                best_diff = abs(len(boundaries) - expected_reps)
                best_segs = boundaries
                
        if found_exact:
            break

    # Convert to sample-level boundaries
    segs = [(b[0] * frame_shift, b[1] * frame_shift) for b in best_segs]

    if len(segs) != expected_reps:
        print(f"  ⚠ WARNING: Expected {expected_reps} segments, found {len(segs)} in {input_path}")

    os.makedirs(output_dir, exist_ok=True)
    for idx, (s, e) in enumerate(segs, 1):
        clip = audio[s:e]
        out_name = f"{speaker_id}_{token_name}_{idx:02d}.wav"
        sf.write(os.path.join(output_dir, out_name), clip, fs)
        print(f"  Saved: {out_name} ({(e - s) / fs:.2f}s)")

    return len(segs)


def batch_segment(raw_dir: str = str(RAW_DIR), out_dir: str = str(SEGMENT_DIR)):
    """
    Process all raw recordings in the directory tree.

    Expected structure:
        data/raw/S01/S01_eka_raw.wav
        data/raw/S01/S01_pancha_raw.wav
        ...
    """
    wavs = sorted(glob.glob(f"{raw_dir}/**/*_raw.wav", recursive=True))
    if not wavs:
        print(f"No *_raw.wav files found in {raw_dir}")
        return

    total_segs = 0
    for wav_path in wavs:
        basename = os.path.basename(wav_path).replace("_raw.wav", "")
        parts = basename.split("_")  # e.g. ['S01', 'eka']
        spk = parts[0]
        tok = "_".join(parts[1:])
        spk_out = os.path.join(out_dir, spk)
        print(f"Processing {wav_path} ...")
        n = segment_utterances(wav_path, spk, tok, spk_out)
        total_segs += n
        print(f"  → {n} segments extracted")

    print(f"\nDone. Total segments: {total_segs}")


def main():
    parser = argparse.ArgumentParser(description="SankhyaVox Audio Segmentation")
    parser.add_argument("--input", type=str, help="Single raw WAV file to segment")
    parser.add_argument("--speaker", type=str, help="Speaker ID (required with --input)")
    parser.add_argument("--token", type=str, help="Token name (required with --input)")
    parser.add_argument("--raw-dir", type=str, default=str(RAW_DIR), help="Raw recordings directory")
    parser.add_argument("--out-dir", type=str, default=str(SEGMENT_DIR), help="Output segments directory")
    args = parser.parse_args()

    if args.input:
        if not args.speaker or not args.token:
            parser.error("--speaker and --token are required with --input")
        segment_utterances(args.input, args.speaker, args.token, args.out_dir)
    else:
        batch_segment(args.raw_dir, args.out_dir)


if __name__ == "__main__":
    main()
