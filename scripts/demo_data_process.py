"""
SankhyaVox – Demo: Data Processing Pipeline.

Selectively runs data pipelines based on CLI flags:
  --human      Convert, segment, QA, extract features, generate CSV for human data
  --tts        Generate TTS segments, extract features, generate CSV
  --augmented  (reserved for future augmentation pipeline)

If no flags are given, all enabled routes run.

Usage:
    python scripts/demo_data_process.py --human
    python scripts/demo_data_process.py --tts
    python scripts/demo_data_process.py --human --tts
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import DataPipeline, SankhyaVoxDataset


def run_human(pipe: DataPipeline) -> None:
    """Full human data pipeline: convert → segment → QA → features → CSV."""

    print("=" * 70)
    print("  Human: Converting raw audio")
    print("=" * 70)
    n_converted = pipe.convert()
    print(f"  Converted {n_converted} files.\n")

    print("=" * 70)
    print("  Human: Segmenting recordings")
    print("=" * 70)
    n_segments = pipe.segment()
    print(f"  Total segments: {n_segments}\n")

    print("=" * 70)
    print("  Human: Segment QA")
    print("=" * 70)
    issues = pipe.validate(source="human", mode="qa")
    print(f"  Issues found: {len(issues)}\n")

    print("=" * 70)
    print("  Human: Extracting MFCC features")
    print("=" * 70)
    n_features = pipe.extract_features("human")
    print(f"  Features extracted: {n_features}\n")

    print("=" * 70)
    print("  Human: Generating metadata CSV")
    print("=" * 70)
    pipe.generate_csv("human")
    print()


def run_tts(pipe: DataPipeline) -> None:
    """TTS pipeline: generate segments → extract features → CSV."""

    print("=" * 70)
    print("  TTS: Generating synthetic data")
    print("=" * 70)
    n_tts = pipe.generate_tts()
    print(f"  TTS files generated: {n_tts}\n")

    print("=" * 70)
    print("  TTS: Extracting MFCC features")
    print("=" * 70)
    n_features = pipe.extract_features("tts")
    print(f"  TTS features extracted: {n_features}\n")

    print("=" * 70)
    print("  TTS: Generating metadata CSV")
    print("=" * 70)
    pipe.generate_csv("tts")
    print()


def run_augmented(pipe: DataPipeline) -> None:
    """Augmented pipeline (placeholder)."""
    print("=" * 70)
    print("  Augmented: Not yet implemented")
    print("=" * 70)
    print()


def main():
    parser = argparse.ArgumentParser(description="SankhyaVox data processing pipeline")
    parser.add_argument("--human", action="store_true", help="Run the human data pipeline")
    parser.add_argument("--tts", action="store_true", help="Run the TTS data pipeline")
    parser.add_argument("--augmented", action="store_true", help="Run the augmented data pipeline (placeholder)")
    args = parser.parse_args()

    # If no flags given, run all
    run_all = not (args.human or args.tts or args.augmented)

    pipe = DataPipeline()

    if args.human or run_all:
        run_human(pipe)

    if args.tts or run_all:
        run_tts(pipe)

    if args.augmented or run_all:
        run_augmented(pipe)

    # ── Load into SankhyaVoxDataset ───────────────────────────────────────
    print("=" * 70)
    print("  Loading SankhyaVoxDataset")
    print("=" * 70)
    ds = SankhyaVoxDataset()
    print(ds.summary())
    print()

    if len(ds) > 0:
        sample = ds[0]
        print(f"  Sample ds[0]:")
        print(f"    Speaker:        {sample['speaker_id']}")
        print(f"    Token:          {sample['token']}")
        print(f"    Label:          {sample['label']}")
        print(f"    Feature shape:  {sample['feature'].shape}")
        print(f"    Audio source:   {sample['audio_source']}")
        print(f"    Audio path:     {sample['audio_path']}")
        print()

        print(f"  Human samples:  {len(ds.filter(category='human'))}")
        print(f"  TTS samples:    {len(ds.filter(category='tts'))}")
        print(f"  Augmented:      {len(ds.filter(category='augmented'))}")
    else:
        print("  No samples found. Ensure data/ contains raw recordings.")

    print("\n" + "=" * 70)
    print("  Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
