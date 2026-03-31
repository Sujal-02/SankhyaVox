"""
SankhyaVox – MFCC Spread Visualisation.

For each token, plots the per-utterance average MFCC profile for every
segment across all speakers on a single axis.

  - Tight clustering  → consistent pronunciation ✓
  - Two visible bands → pronunciation split (e.g. "char" vs "cha-TUR") ✗

Usage:
    python scripts/visualise_mfcc.py                    # all risky tokens
    python scripts/visualise_mfcc.py --token dvi        # single token
    python scripts/visualise_mfcc.py --all              # all 13 tokens
"""

import argparse
import glob
import os
import sys

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import (
    APPLY_CMVN,
    FRAME_LENGTH,
    FRAME_SHIFT,
    N_FFT,
    N_MELS,
    N_MFCC,
    SAMPLE_RATE,
    SEGMENT_DIR,
    VOCAB,
)

# Tokens most at risk of Hindi-bleed / syllable-dropping
RISKY_TOKENS = ["dvi", "catur", "pancha", "dasha", "vimsati", "ashta"]

# Expected duration ranges per token group (seconds) — from the tech report
DURATION_THRESHOLDS = {
    "dvi":     (0.22, 0.55),
    "tri":     (0.22, 0.55),
    "shat":    (0.22, 0.55),
    "eka":     (0.30, 0.70),
    "nava":    (0.30, 0.70),
    "shata":   (0.30, 0.70),
    "catur":   (0.40, 0.90),
    "pancha":  (0.40, 0.90),
    "sapta":   (0.40, 0.90),
    "ashta":   (0.40, 0.90),
    "shunya":  (0.40, 0.90),
    "dasha":   (0.40, 0.90),
    "vimsati": (0.55, 1.10),
}


def _mfcc_mean(wav_path: str) -> np.ndarray:
    """Return the per-frame mean MFCC vector (shape: n_mfcc,) for a file."""
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    mfcc = librosa.feature.mfcc(
        y=audio.astype(np.float32),
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=FRAME_SHIFT,
        win_length=FRAME_LENGTH,
        n_mels=N_MELS,
        window="hamming",
    )  # shape: (n_mfcc, T)

    if APPLY_CMVN:
        mean = mfcc.mean(axis=1, keepdims=True)
        std = mfcc.std(axis=1, keepdims=True)
        std[std < 1e-10] = 1e-10
        mfcc = (mfcc - mean) / std

    return mfcc.mean(axis=1)  # average over time → (n_mfcc,)


def plot_token_mfcc_spread(
    token_name: str,
    seg_dir: str = str(SEGMENT_DIR),
    output_dir: str = "plots",
    show: bool = True,
) -> str | None:
    """
    Plot the MFCC spread for a single token across all speakers/segments.

    Returns the path to the saved figure (or None if no files found).
    """
    pattern = os.path.join(seg_dir, "**", f"*_{token_name}_*.wav")
    paths = sorted(glob.glob(pattern, recursive=True))

    if not paths:
        print(f"  No segments found for token '{token_name}' in {seg_dir}")
        return None

    # Gather per-utterance mean MFCCs
    means = []
    labels = []
    for p in paths:
        try:
            means.append(_mfcc_mean(p))
            labels.append(os.path.basename(p))
        except Exception as e:
            print(f"  Skipping {p}: {e}")

    if not means:
        return None

    means = np.array(means)         # (N_files, n_mfcc)
    grand_mean = means.mean(axis=0) # (n_mfcc,)
    x = np.arange(N_MFCC)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#1e1e2e")

    # Individual utterances (semi-transparent)
    for row in means:
        ax.plot(x, row, alpha=0.35, linewidth=0.9, color="#88c0d0")

    # Grand mean
    ax.plot(x, grand_mean, color="#bf616a", linewidth=2.2, label="Grand mean", zorder=5)

    # Styling
    for spine in ax.spines.values():
        spine.set_edgecolor("#4c566a")
    ax.tick_params(colors="#d8dee9")
    ax.set_xlabel("MFCC coefficient index", color="#d8dee9")
    ax.set_ylabel("Mean value (CMVN)", color="#d8dee9")
    ax.set_title(
        f"MFCC spread for  '{token_name}'  ·  {len(means)} segments",
        color="#eceff4", fontsize=12, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"c{i}" for i in x], fontsize=8)
    ax.legend(labelcolor="#d8dee9", facecolor="#2e3440", edgecolor="#4c566a")
    ax.grid(axis="y", color="#3b4252", linewidth=0.5, linestyle="--")

    # Annotation: tight vs spread
    spread = means.std(axis=0).mean()
    spread_label = "✓ Consistent" if spread < 0.5 else "⚠ Spread — check pronunciation"
    ax.annotate(
        f"Avg std={spread:.3f}  {spread_label}",
        xy=(0.02, 0.05), xycoords="axes fraction",
        color="#a3be8c" if spread < 0.5 else "#ebcb8b",
        fontsize=9,
    )

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"mfcc_spread_{token_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    if show:
        plt.show()
    plt.close()
    return out_path


def verify_segment_durations(
    seg_dir: str = str(SEGMENT_DIR),
    thresholds: dict = DURATION_THRESHOLDS,
) -> list:
    """
    Flag segments whose duration falls outside expected range.
    Short → likely dropped syllable.  Long → likely merged segment.
    """
    wavs = sorted(glob.glob(os.path.join(seg_dir, "**", "*.wav"), recursive=True))
    issues = []

    for wav_path in wavs:
        basename = os.path.splitext(os.path.basename(wav_path))[0]
        # Parse token from filename: S01_eka_01 → eka
        parts = basename.rsplit("_", 1)
        if len(parts) != 2:
            continue
        token_parts = parts[0].split("_", 1)
        if len(token_parts) != 2:
            continue
        token = token_parts[1]

        try:
            info = sf.info(wav_path)
            dur = info.frames / info.samplerate
        except Exception:
            continue

        lo, hi = thresholds.get(token, (0.15, 1.50))
        if dur < lo:
            issues.append((wav_path, dur, "TOO SHORT — possible dropped syllable"))
        elif dur > hi:
            issues.append((wav_path, dur, "TOO LONG  — possible merged segment"))

    if issues:
        print(f"\n{'=' * 62}")
        print(f"  {len(issues)} segments need review:")
        print(f"{'=' * 62}")
        for path, dur, reason in issues:
            print(f"  [{dur:.2f}s]  {reason}")
            print(f"           {os.path.basename(path)}")
    else:
        print("✓ All segments passed duration check.")

    return issues


def main():
    parser = argparse.ArgumentParser(description="SankhyaVox MFCC Visualisation + Duration Check")
    parser.add_argument("--token", type=str, help="Plot spread for a specific token")
    parser.add_argument("--all", action="store_true", help="Plot all 13 vocabulary tokens")
    parser.add_argument("--seg-dir", type=str, default=str(SEGMENT_DIR))
    parser.add_argument("--out-dir", type=str, default="plots")
    parser.add_argument("--duration-check", action="store_true",
                        help="Run duration filter check only")
    parser.add_argument("--no-show", action="store_true",
                        help="Save plots without displaying them")
    args = parser.parse_args()

    if args.duration_check:
        verify_segment_durations(args.seg_dir)
        return

    tokens = VOCAB if args.all else ([args.token] if args.token else RISKY_TOKENS)
    print(f"Generating MFCC spread plots for: {tokens}")
    for tok in tokens:
        plot_token_mfcc_spread(tok, args.seg_dir, args.out_dir, show=not args.no_show)

    print("\nDone. Run with --duration-check to validate segment lengths.")


if __name__ == "__main__":
    main()
