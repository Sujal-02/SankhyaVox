"""
SankhyaVox – Demo: Feature Visualisation.

Generates waveform, spectrogram, and MFCC plots for a sample of
processed audio, plus a cross-speaker comparison grid.

Saves all figures to results/viz/ and also displays them interactively.

Prerequisites:
    Run DataPipeline().build() first to populate data_processed/.

Usage:
    python scripts/demo_visualize_feat.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import SankhyaVoxDataset
from src.config import RESULTS_DIR
from src.viz import plot_comparison, plot_mfcc, plot_spectrogram, plot_waveform

VIZ_DIR = RESULTS_DIR / "viz"
VIZ_DIR.mkdir(parents=True, exist_ok=True)


def main():
    ds = SankhyaVoxDataset()
    if len(ds) == 0:
        print("No samples found. Run DataPipeline().build() first.")
        return

    print(f"Dataset: {repr(ds)}")
    print(f"Saving visualisations to {VIZ_DIR}\n")

    # Pick the first sample for single-file plots
    sample = ds[0]
    wav_path = sample["audio_path"]
    npy_path = ds.df.iloc[0]["npy_path"]
    # Resolve npy_path relative to processed_dir
    npy_abs = str(ds._root / npy_path)
    label = f"{sample['speaker_id']}_{sample['token']}"

    print(f"=== Single sample: {label} ===")

    # Waveform
    print("  Plotting waveform...")
    plot_waveform(
        wav_path,
        title=f"Waveform — {label}",
        save_path=str(VIZ_DIR / f"{label}_waveform.png"),
    )

    # Mel spectrogram
    print("  Plotting spectrogram...")
    plot_spectrogram(
        wav_path,
        title=f"Mel Spectrogram — {label}",
        save_path=str(VIZ_DIR / f"{label}_spectrogram.png"),
    )

    # MFCC heatmap (from .npy features)
    print("  Plotting MFCC heatmap...")
    plot_mfcc(
        npy_abs,
        title=f"MFCC — {label}",
        save_path=str(VIZ_DIR / f"{label}_mfcc.png"),
    )

    # MFCC heatmap (from .wav directly, for comparison)
    print("  Plotting MFCC from audio...")
    plot_mfcc(
        wav_path,
        title=f"MFCC (from wav) — {label}",
        save_path=str(VIZ_DIR / f"{label}_mfcc_from_wav.png"),
    )

    # ── Cross-speaker comparison ──────────────────────────────────────────
    # Pick one token and collect one sample per speaker
    target_token = sample["token"]
    speakers = ds.speakers
    comparison_wavs = []
    comparison_npys = []

    for spk in speakers:
        spk_ds = ds.filter(speaker=spk)
        for i in range(len(spk_ds)):
            s = spk_ds[i]
            if s["token"] == target_token:
                comparison_wavs.append(s["audio_path"])
                comparison_npys.append(str(ds._root / spk_ds.df.iloc[i]["npy_path"]))
                break

    if len(comparison_wavs) >= 2:
        print(f"\n=== Cross-speaker comparison for '{target_token}' ({len(comparison_wavs)} speakers) ===")

        print("  Plotting waveform comparison...")
        plot_comparison(
            comparison_wavs,
            kind="waveform",
            save_path=str(VIZ_DIR / f"compare_{target_token}_waveform.png"),
        )

        print("  Plotting spectrogram comparison...")
        plot_comparison(
            comparison_wavs,
            kind="spectrogram",
            save_path=str(VIZ_DIR / f"compare_{target_token}_spectrogram.png"),
        )

        print("  Plotting MFCC comparison...")
        plot_comparison(
            comparison_npys,
            kind="mfcc",
            save_path=str(VIZ_DIR / f"compare_{target_token}_mfcc.png"),
        )
    else:
        print(f"\n  Skipping comparison: only {len(comparison_wavs)} speaker(s) have '{target_token}'")

    print(f"\nDone. All figures saved to {VIZ_DIR}")


if __name__ == "__main__":
    main()
