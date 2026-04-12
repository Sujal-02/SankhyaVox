"""
SankhyaVox – Demo: Data Processing Pipeline.

Selectively runs data pipelines based on CLI flags:
  --human      Convert, segment, QA, extract features, generate CSV for human data
  --tts        Generate TTS segments, extract features, generate CSV
  --augment    Augment segments and extract features + CSV for augmented data
               Accepts: human, tts, all, or a path to a speaker segment dir.
               If no value given, defaults to "all".

If no flags are given, all enabled routes run.

Usage:
    python scripts/demo_data_process.py --human
    python scripts/demo_data_process.py --tts
    python scripts/demo_data_process.py --augment human
    python scripts/demo_data_process.py --augment data_processed/human/segments/S02
    python scripts/demo_data_process.py --human --tts --augment all
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


def run_augmented(pipe: DataPipeline, source: str) -> None:
    """Augmentation pipeline: augment segments → extract features → CSV."""

    print("=" * 70)
    print(f"  Augmented: Augmenting segments (source={source})")
    print("=" * 70)
    n_aug = pipe.augment(source)
    print(f"  Augmented files: {n_aug}\n")

    print("=" * 70)
    print("  Augmented: Extracting MFCC features")
    print("=" * 70)
    n_features = pipe.extract_features("augmented")
    print(f"  Augmented features extracted: {n_features}\n")

    print("=" * 70)
    print("  Augmented: Generating metadata CSV")
    print("=" * 70)
    pipe.generate_csv("augmented")
    print()


def main():
    parser = argparse.ArgumentParser(description="SankhyaVox data processing pipeline")
    parser.add_argument("--human", action="store_true", help="Run the human data pipeline")
    parser.add_argument("--tts", action="store_true", help="Run the TTS data pipeline")
    parser.add_argument(
        "--augment",
        nargs="?",
        const="all",
        default=None,
        metavar="SOURCE",
        help="Run augmentation pipeline. SOURCE: human, tts, all (default), "
             "or a path to a speaker segment directory.",
    )
    args = parser.parse_args()

    # If no flags given, run all
    run_all = not (args.human or args.tts or args.augment is not None)

    pipe = DataPipeline()

    if args.human or run_all:
        run_human(pipe)

    if args.tts or run_all:
        run_tts(pipe)

    if args.augment is not None or run_all:
        aug_source = args.augment if args.augment is not None else "all"
        run_augmented(pipe, aug_source)

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
    print("  Data Preprocessing complete.")
    print("=" * 70)


if __name__ == "__main__":
    # Example: process human pipeline and augment only S02
    #   python scripts/demo_data_process.py --human --augment data_processed/human/segments/S02
    main()
