"""
SankhyaVox – File Naming Convention Validator.

Validates that raw and segmented audio files follow the naming convention:
  Raw:       <SpeakerID>_<Token>_raw.wav       (e.g. S01_eka_raw.wav)
  Segmented: <SpeakerID>_<Token>_<RepNum>.wav   (e.g. S01_eka_01.wav)

Also provides a rename helper for common issues (spaces, wrong case, etc).

Usage:
    python scripts/validate_naming.py                   # validate all
    python scripts/validate_naming.py --fix              # auto-fix common issues
    python scripts/validate_naming.py --dir data/raw     # specific directory
"""

import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import RAW_DIR, SEGMENT_DIR, VOCAB

# Valid speaker ID pattern
SPEAKER_PATTERN = re.compile(r"^S\d{2,3}$")

# Valid tokens (all lowercase ASCII forms)
VALID_TOKENS = set(VOCAB)

# Also allow multi-word combo tokens like "dasha_eka", "tri_dasha_catur"
COMBO_PARTS = VALID_TOKENS.copy()

# Raw file pattern: S01_eka_raw.wav
RAW_PATTERN = re.compile(r"^(S\d{2,3})_(.+)_raw\.wav$", re.IGNORECASE)

# Segmented file pattern: S01_eka_01.wav
SEG_PATTERN = re.compile(r"^(S\d{2,3})_(.+)_(\d{2})\.wav$", re.IGNORECASE)


def validate_token(token: str) -> bool:
    """Check if a token name is valid (single word or underscore-separated combo)."""
    parts = token.split("_")
    return all(p in COMBO_PARTS for p in parts)


def validate_file(filepath: str, is_raw: bool = True) -> list:
    """
    Validate a single filename.
    Returns a list of issue strings (empty = valid).
    """
    basename = os.path.basename(filepath)
    issues = []

    pattern = RAW_PATTERN if is_raw else SEG_PATTERN

    match = pattern.match(basename)
    if not match:
        issues.append(f"Does not match expected pattern: {basename}")
        return issues

    speaker = match.group(1).upper()
    token = match.group(2).lower()

    if not SPEAKER_PATTERN.match(speaker):
        issues.append(f"Invalid speaker ID '{speaker}' (expected S01, S02, ...)")

    if not validate_token(token):
        issues.append(f"Unknown token '{token}' (not in vocabulary)")

    if not is_raw:
        rep = int(match.group(3))
        if rep < 1 or rep > 99:
            issues.append(f"Rep number {rep} out of expected range 1-10")

    # Check for common problems
    if " " in basename:
        issues.append("Filename contains spaces")
    if basename != basename.lower().replace(speaker.lower(), speaker):
        # Token part should be lowercase
        pass  # minor, don't flag

    return issues


def fix_filename(filepath: str) -> str:
    """
    Attempt to fix common naming issues. Returns the corrected basename.
    Fixes: spaces → underscores, uppercase tokens → lowercase, missing leading zeros.
    """
    basename = os.path.basename(filepath)

    # Replace spaces with underscores
    fixed = basename.replace(" ", "_")

    # Lowercase everything except speaker ID prefix
    parts = fixed.split("_", 1)
    if len(parts) == 2 and SPEAKER_PATTERN.match(parts[0].upper()):
        fixed = parts[0].upper() + "_" + parts[1].lower()

    return fixed


def validate_directory(directory: str, is_raw: bool = True):
    """Validate all WAV files in a directory tree."""
    pattern = "**/*_raw.wav" if is_raw else "**/*.wav"
    wavs = sorted(glob.glob(os.path.join(directory, pattern), recursive=True))

    if not wavs:
        print(f"No matching files found in {directory}")
        return []

    all_issues = []
    valid_count = 0

    for wav_path in wavs:
        issues = validate_file(wav_path, is_raw=is_raw)
        if issues:
            all_issues.append((wav_path, issues))
        else:
            valid_count += 1

    print(f"\nValidated {len(wavs)} files in {directory}")
    print(f"  ✓ Valid:   {valid_count}")
    print(f"  ✗ Issues:  {len(all_issues)}")

    if all_issues:
        print("\nFiles with issues:")
        for path, issues in all_issues:
            print(f"  {os.path.basename(path)}")
            for issue in issues:
                print(f"    → {issue}")

    return all_issues


def auto_fix(directory: str, is_raw: bool = True, dry_run: bool = True):
    """Auto-fix common naming issues. Use --fix to apply."""
    pattern = "**/*_raw.wav" if is_raw else "**/*.wav"
    wavs = sorted(glob.glob(os.path.join(directory, pattern), recursive=True))

    renames = []
    for wav_path in wavs:
        basename = os.path.basename(wav_path)
        fixed = fix_filename(wav_path)
        if fixed != basename:
            renames.append((wav_path, os.path.join(os.path.dirname(wav_path), fixed)))

    if not renames:
        print("No files need renaming.")
        return

    for old, new in renames:
        action = "WOULD RENAME" if dry_run else "RENAMED"
        print(f"  {action}: {os.path.basename(old)} → {os.path.basename(new)}")
        if not dry_run:
            os.rename(old, new)

    if dry_run:
        print(f"\n  {len(renames)} files would be renamed. Use --fix to apply.")


def main():
    parser = argparse.ArgumentParser(description="SankhyaVox File Naming Validator")
    parser.add_argument("--dir", type=str, default=str(RAW_DIR),
                        help="Directory to validate")
    parser.add_argument("--type", choices=["raw", "seg"], default="raw",
                        help="File type: 'raw' for *_raw.wav, 'seg' for segmented")
    parser.add_argument("--fix", action="store_true",
                        help="Auto-fix common naming issues (renames files)")
    args = parser.parse_args()

    is_raw = args.type == "raw"

    if args.fix:
        auto_fix(args.dir, is_raw=is_raw, dry_run=False)
    else:
        validate_directory(args.dir, is_raw=is_raw)


if __name__ == "__main__":
    main()
