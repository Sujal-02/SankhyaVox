"""
SankhyaVox – Demo: Data Processing Pipeline.

Runs the full data pipeline end-to-end:
  1. Convert raw audio from data/ → data_processed/human/raw/
  2. Segment repeated-utterance recordings into individual clips
  3. Run segment QA
  4. Extract 39-dim MFCC features
  5. Load the processed data into a SankhyaVoxDataset and print a summary

Usage:
    python scripts/demo_data_process.py
"""

import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import DataPipeline, SankhyaVoxDataset


def main():
    pipe = DataPipeline()

    # ── Step 1: Convert raw audio to standardised 16 kHz mono WAV ─────────
    print("=" * 70)
    print("  Step 1: Converting raw audio")
    print("=" * 70)
    n_converted = pipe.convert()
    print(f"  Converted {n_converted} files.\n")

    # ── Step 2: Segment into individual utterances ────────────────────────
    print("=" * 70)
    print("  Step 2: Segmenting recordings")
    print("=" * 70)
    n_segments = pipe.segment()
    print(f"  Total segments: {n_segments}\n")

    # ── Step 3: Run segment QA ────────────────────────────────────────────
    print("=" * 70)
    print("  Step 3: Segment QA")
    print("=" * 70)
    issues = pipe.validate(source="human", mode="qa")
    print(f"  Issues found: {len(issues)}\n")

    # ── Step 4: Extract MFCC features ─────────────────────────────────────
    print("=" * 70)
    print("  Step 4: Extracting human features")
    print("=" * 70)
    n_features = pipe.extract_features("human")
    print(f"  Features extracted: {n_features}\n")

    # ── Step 5: Generate TTS data ─────────────────────────────────────────
    print("=" * 70)
    print("  Step 5: Generating TTS data")
    print("=" * 70)
    try:
        n_tts = pipe.generate_tts()
        print(f"  TTS files generated: {n_tts}")
        print("  Extracting TTS features...")
        n_tts_feat = pipe.extract_features("tts")
        print(f"  TTS features extracted: {n_tts_feat}\n")
    except Exception as e:
        print(f"  TTS generation skipped: {e}")
        print("  (Install edge_tts to enable: pip install edge_tts)\n")

    # ── Step 6: Load into SankhyaVoxDataset ───────────────────────────────
    print("=" * 70)
    print("  Step 6: Loading SankhyaVoxDataset")
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

        print(f"  Human samples:  {len(ds.human)}")
        print(f"  TTS samples:    {len(ds.tts)}")
        print(f"  Augmented:      {len(ds.augmented)}")
    else:
        print("  No samples found. Ensure data/ contains raw recordings.")

    print("\n" + "=" * 70)
    print("  Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
