"""
SankhyaVox – Demo: Feature Visualisation.

Generates waveform, spectrogram, and MFCC plots for a given audio file,
plus an optional preprocessed-vs-original comparison.

Saves all figures to results/viz/ and also displays them interactively.

Usage:
    python scripts/demo_visualize_feat.py path/to/recording.wav
    python scripts/demo_visualize_feat.py path/to/recording.wav --output-dir results/viz
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RESULTS_DIR
from src.viz import plot_mfcc, plot_spectrogram, plot_waveform


def main():
    parser = argparse.ArgumentParser(
        description="Visualise waveform, spectrogram, and MFCC for an audio file."
    )
    parser.add_argument(
        "audio", type=str, help="Path to the input audio file (WAV, MP3, M4A, etc.)."
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save plots (default: results/viz).",
    )
    args = parser.parse_args()

    audio = Path(args.audio)
    if not audio.exists():
        print(f"Error: File not found: {audio}")
        sys.exit(1)

    base_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR / "viz"
    label = audio.stem
    viz_dir = base_dir / label
    viz_dir.mkdir(parents=True, exist_ok=True)

    print(f"Audio: {audio}")
    print(f"Saving visualisations to {viz_dir}\n")

    # Waveform
    print("  Plotting waveform...")
    plot_waveform(
        str(audio),
        title=f"Waveform — {label}",
        save_path=str(viz_dir / "waveform.png"),
    )

    # Mel spectrogram
    print("  Plotting spectrogram...")
    plot_spectrogram(
        str(audio),
        title=f"Mel Spectrogram — {label}",
        save_path=str(viz_dir / "spectrogram.png"),
    )

    # MFCC heatmap
    print("  Plotting MFCC heatmap...")
    plot_mfcc(
        str(audio),
        title=f"MFCC — {label}",
        save_path=str(viz_dir / "mfcc.png"),
    )

    print(f"\nDone. All figures saved to {viz_dir}")


if __name__ == "__main__":
    # Examples:
    #   python scripts/demo_visualize_feat.py data_processed/human/segments/S01/S01_007_01.wav
    #   python scripts/demo_visualize_feat.py recording.wav --output-dir my_plots/
    main()
