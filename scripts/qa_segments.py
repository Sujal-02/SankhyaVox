"""
SankhyaVox – Segment Quality Assurance (QA) Script.

Validates segmented audio files for correctness:
  - Checks each speaker has exactly 10 segments per token
  - Flags segments that are too short (<150 ms) or too long (>1.5 s)
  - Reports missing tokens per speaker
  - Prints a summary table

Usage:
    python scripts/qa_segments.py
    python scripts/qa_segments.py --seg-dir data/segments
"""

import argparse
import glob
import os
import sys

import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import EXPECTED_REPS, SEGMENT_DIR, VOCAB

# Duration thresholds (seconds)
MIN_DUR = 0.15   # segments shorter than this are likely mis-segments
MAX_DUR = 1.50   # segments longer than this might contain multiple words


def qa_segments(seg_dir: str = str(SEGMENT_DIR)):
    """Run QA checks on all segmented files."""

    wavs = sorted(glob.glob(f"{seg_dir}/**/*.wav", recursive=True))
    if not wavs:
        print(f"No WAV files found in {seg_dir}")
        return

    # Organise files by speaker and token
    # Expected filename: S01_eka_01.wav
    speaker_tokens = {}   # {speaker: {token: [files]}}
    issues = []

    for wav_path in wavs:
        basename = os.path.splitext(os.path.basename(wav_path))[0]
        parts = basename.rsplit("_", 1)  # split off rep number
        if len(parts) != 2 or not parts[1].isdigit():
            issues.append(("NAMING", wav_path, "Cannot parse rep number"))
            continue

        prefix = parts[0]  # e.g. "S01_eka"
        prefix_parts = prefix.split("_", 1)
        if len(prefix_parts) != 2:
            issues.append(("NAMING", wav_path, "Cannot parse speaker/token"))
            continue

        speaker = prefix_parts[0]
        token = prefix_parts[1]

        speaker_tokens.setdefault(speaker, {}).setdefault(token, []).append(wav_path)

        # Check duration
        try:
            info = sf.info(wav_path)
            dur = info.duration
            if dur < MIN_DUR:
                issues.append(("SHORT", wav_path, f"{dur:.3f}s < {MIN_DUR}s"))
            elif dur > MAX_DUR:
                issues.append(("LONG", wav_path, f"{dur:.3f}s > {MAX_DUR}s"))
        except Exception as e:
            issues.append(("READ_ERR", wav_path, str(e)))

    # --- Report ---
    print("=" * 70)
    print("  SankhyaVox — Segment QA Report")
    print("=" * 70)

    total_files = len(wavs)
    total_speakers = len(speaker_tokens)
    print(f"\n  Total files : {total_files}")
    print(f"  Speakers    : {total_speakers}")

    # Per-speaker summary
    print(f"\n{'Speaker':<10} {'Tokens':<8} {'Files':<8} {'Missing Tokens'}")
    print("-" * 60)

    all_expected_tokens = set(VOCAB)  # the 13 base tokens

    for speaker in sorted(speaker_tokens):
        tokens = speaker_tokens[speaker]
        found_tokens = set(tokens.keys())
        missing = all_expected_tokens - found_tokens
        n_files = sum(len(v) for v in tokens.values())
        missing_str = ", ".join(sorted(missing)) if missing else "—"
        print(f"  {speaker:<8} {len(tokens):<8} {n_files:<8} {missing_str}")

        # Check rep counts
        for token in sorted(tokens):
            count = len(tokens[token])
            if count != EXPECTED_REPS:
                issues.append(("REP_COUNT", f"{speaker}/{token}",
                               f"Expected {EXPECTED_REPS}, found {count}"))

    # Issues
    if issues:
        print(f"\n⚠  Issues Found: {len(issues)}")
        print("-" * 60)
        for kind, location, detail in issues:
            loc = os.path.basename(location) if os.sep in location else location
            print(f"  [{kind:<10}] {loc:<35} {detail}")
    else:
        print("\n✓  No issues found. All segments look good!")

    print("\n" + "=" * 70)
    return issues


def main():
    parser = argparse.ArgumentParser(description="SankhyaVox Segment QA")
    parser.add_argument("--seg-dir", type=str, default=str(SEGMENT_DIR),
                        help="Segmented recordings directory")
    args = parser.parse_args()
    qa_segments(args.seg_dir)


if __name__ == "__main__":
    main()
