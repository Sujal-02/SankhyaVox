"""
SankhyaVox – Audio Segmentor.

Modular pipeline for splitting raw repeated-utterance recordings into
individually labelled single-utterance WAV clips.

Each raw file contains multiple repetitions of a single token separated
by silence.  This module detects speech regions via energy-based VAD and
writes each segment following the naming convention:

    <SpeakerId>_<numericId>_<rep>.wav      e.g.  S01_001_03.wav
"""

import glob
import os
import re
from typing import Any, List, Tuple, cast

import numpy as np
import soundfile as sf
from scipy.ndimage import binary_closing, binary_opening
from scipy.signal import butter, filtfilt
from tqdm import tqdm

from src.config import (
    ENERGY_THRESHOLD,
    EXPECTED_REPS,
    HIGHPASS_CUTOFF,
    MIN_SILENCE_DUR_S,
    MIN_SPEECH_DUR_S,
    NUMERIC_ID_TO_TOKEN,
    PAD_MS,
    QA_MAX_DURATION,
    QA_MIN_DURATION,
    SAMPLE_RATE,
    TOKEN_TO_NUMERIC_ID,
    VOCAB,
)

# ── Naming patterns ────────────────────────────────────────────────────────────

SPEAKER_PATTERN = re.compile(r"^(S|TTS)\d{2,3}$")
VALID_TOKENS = set(VOCAB)
VALID_NUMERIC_IDS = set(TOKEN_TO_NUMERIC_ID.values())
RAW_FILE_PATTERN = re.compile(r"^(S\d{2,3})_(.+)_raw\.wav$", re.IGNORECASE)
SEGMENT_FILE_PATTERN = re.compile(r"^((S|TTS)\d{2,3})_(\d{3})_(\d{2})\.wav$")


# ═══════════════════════════════════════════════════════════════════════════════
#  AUDIO I/O
# ═══════════════════════════════════════════════════════════════════════════════


def load_audio(path: str, target_fs: int = SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """Load an audio file, convert to mono, and resample to *target_fs*."""
    audio, fs = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if fs != target_fs:
        from scipy.signal import resample_poly

        audio = resample_poly(audio, target_fs, fs).astype(np.float32)
        fs = target_fs
    return audio.astype(np.float32), fs


# ═══════════════════════════════════════════════════════════════════════════════
#  SIGNAL PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════


def apply_highpass(
    audio: np.ndarray,
    fs: int = SAMPLE_RATE,
    cutoff: float = HIGHPASS_CUTOFF,
) -> np.ndarray:
    """Apply a 4th-order Butterworth high-pass filter."""
    nyq = 0.5 * fs
    result = cast(tuple[Any, Any], butter(4, cutoff / nyq, btype="high"))
    b, a = result[0], result[1]
    return filtfilt(b, a, audio).astype(np.float32)


def compute_rms_energy(
    audio: np.ndarray,
    fs: int = SAMPLE_RATE,
    frame_ms: float = 25.0,
    shift_ms: float = 10.0,
) -> np.ndarray:
    """Compute short-time RMS energy (one value per frame)."""
    frame_len = int(frame_ms / 1000 * fs)
    frame_shift = int(shift_ms / 1000 * fs)
    frames = [
        audio[i : i + frame_len]
        for i in range(0, len(audio) - frame_len, frame_shift)
    ]
    return np.array([np.sqrt(np.mean(f**2)) for f in frames])


# ═══════════════════════════════════════════════════════════════════════════════
#  VOICE ACTIVITY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════


def detect_speech_regions(
    rms: np.ndarray,
    threshold_ratio: float = ENERGY_THRESHOLD,
    min_speech_s: float = MIN_SPEECH_DUR_S,
    min_silence_s: float = MIN_SILENCE_DUR_S,
    shift_ms: float = 10.0,
) -> np.ndarray:
    """Return a boolean mask indicating speech frames after morphological smoothing."""
    threshold = threshold_ratio * np.percentile(rms, 95)
    is_speech = rms > threshold

    close_kernel = int(min_silence_s / (shift_ms / 1000))
    open_kernel = int(min_speech_s / (shift_ms / 1000))
    is_speech = binary_closing(is_speech, structure=np.ones(close_kernel))
    is_speech = binary_opening(is_speech, structure=np.ones(open_kernel))
    return is_speech


def find_boundaries(
    speech_mask: np.ndarray, pad_frames: int
) -> List[Tuple[int, int]]:
    """Extract ``(start, end)`` frame indices for each contiguous speech region."""
    boundaries: List[Tuple[int, int]] = []
    in_seg = False
    start = 0
    for i, active in enumerate(speech_mask):
        if active and not in_seg:
            start = max(0, i - pad_frames)
            in_seg = True
        elif not active and in_seg:
            end = min(len(speech_mask), i + pad_frames)
            boundaries.append((start, end))
            in_seg = False
    if in_seg:
        boundaries.append((start, len(speech_mask)))
    return boundaries


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════


def segment_file(
    input_path: str,
    speaker_id: str,
    numeric_id: str,
    output_dir: str,
    expected_reps: int = EXPECTED_REPS,
) -> int:
    """
    Segment one raw recording into individual WAV clips.

    Output naming: ``<speaker_id>_<numeric_id>_<rep>.wav``

    Returns the number of segments extracted.
    """
    audio, fs = load_audio(input_path)
    audio = apply_highpass(audio, fs)

    rms = compute_rms_energy(audio, fs)
    speech_mask = detect_speech_regions(rms)

    pad_frames = int(PAD_MS / 10)
    frame_shift_samples = int(0.010 * fs)
    boundaries = find_boundaries(speech_mask, pad_frames)

    # Frame-index → sample-index
    segments = [
        (s * frame_shift_samples, e * frame_shift_samples) for s, e in boundaries
    ]

    if len(segments) != expected_reps:
        print(
            f"  WARNING: Expected {expected_reps}, found {len(segments)} "
            f"in {input_path}"
        )

    os.makedirs(output_dir, exist_ok=True)
    for idx, (s, e) in enumerate(segments, 1):
        clip = audio[s:e]
        filename = f"{speaker_id}_{numeric_id}_{idx:02d}.wav"
        sf.write(os.path.join(output_dir, filename), clip, fs)
        print(f"  Saved: {filename} ({(e - s) / fs:.2f}s)")

    return len(segments)


def batch_segment(raw_dir: str, out_dir: str) -> int:
    """
    Segment all ``*_raw.wav`` files under *raw_dir*.

    Parses each filename to extract speaker ID and token name, maps the
    token to its numeric ID, and writes segments into ``out_dir/<speaker>/``.
    """
    wavs = sorted(glob.glob(f"{raw_dir}/**/*_raw.wav", recursive=True))
    if not wavs:
        print(f"No *_raw.wav files found in {raw_dir}")
        return 0

    total = 0
    for wav_path in tqdm(wavs, desc="Segmenting", unit="file"):
        basename = os.path.basename(wav_path).replace("_raw.wav", "")
        parts = basename.split("_", 1)
        if len(parts) != 2:
            tqdm.write(f"  SKIP: Cannot parse {basename}")
            continue

        speaker, token = parts[0], parts[1]
        numeric_id = TOKEN_TO_NUMERIC_ID.get(token)
        if numeric_id is None:
            tqdm.write(f"  SKIP: Unknown token '{token}' in {basename}")
            continue

        speaker_out = os.path.join(out_dir, speaker)
        n = segment_file(wav_path, speaker, numeric_id, speaker_out)
        total += n

    print(f"Done. Total segments: {total}")
    return total


# ═══════════════════════════════════════════════════════════════════════════════
#  SEGMENT QUALITY ASSURANCE
# ═══════════════════════════════════════════════════════════════════════════════


def qa_segments(seg_dir: str) -> list:
    """
    Run QA checks on all segmented files.

    Checks filename parsing, duration bounds, token coverage, and rep counts.
    Returns a list of ``(issue_type, location, detail)`` tuples.
    """
    wavs = sorted(glob.glob(f"{seg_dir}/**/*.wav", recursive=True))
    if not wavs:
        print(f"No WAV files found in {seg_dir}")
        return []

    speaker_tokens: dict = {}
    issues: list = []

    for wav_path in wavs:
        basename = os.path.splitext(os.path.basename(wav_path))[0]
        parts = basename.split("_")
        if len(parts) != 3 or not parts[2].isdigit():
            issues.append(("NAMING", wav_path, "Cannot parse filename"))
            continue

        speaker, numeric_id, _rep_str = parts
        if numeric_id not in VALID_NUMERIC_IDS:
            issues.append(("NAMING", wav_path, f"Unknown numeric ID '{numeric_id}'"))
            continue

        speaker_tokens.setdefault(speaker, {}).setdefault(numeric_id, []).append(
            wav_path
        )

        try:
            info = sf.info(wav_path)
            dur = info.duration
            if dur < QA_MIN_DURATION:
                issues.append(("SHORT", wav_path, f"{dur:.3f}s < {QA_MIN_DURATION}s"))
            elif dur > QA_MAX_DURATION:
                issues.append(("LONG", wav_path, f"{dur:.3f}s > {QA_MAX_DURATION}s"))
        except Exception as e:
            issues.append(("READ_ERR", wav_path, str(e)))

    # ── Report ─────────────────────────────────────────────────────────────
    print("=" * 70)
    print("  SankhyaVox — Segment QA Report")
    print("=" * 70)
    print(f"\n  Total files : {len(wavs)}")
    print(f"  Speakers    : {len(speaker_tokens)}")

    expected_ids = set(TOKEN_TO_NUMERIC_ID.values())
    print(f"\n{'Speaker':<10} {'Tokens':<8} {'Files':<8} {'Missing'}")
    print("-" * 60)

    for speaker in sorted(speaker_tokens):
        found = speaker_tokens[speaker]
        missing_ids = expected_ids - set(found.keys())
        n_files = sum(len(v) for v in found.values())
        missing_labels = [NUMERIC_ID_TO_TOKEN.get(m, m) for m in sorted(missing_ids)]
        missing_str = ", ".join(missing_labels) if missing_labels else "—"
        print(f"  {speaker:<8} {len(found):<8} {n_files:<8} {missing_str}")

        for nid in sorted(found):
            count = len(found[nid])
            if count != EXPECTED_REPS:
                issues.append(
                    (
                        "REP_COUNT",
                        f"{speaker}/{nid}",
                        f"Expected {EXPECTED_REPS}, found {count}",
                    )
                )

    if issues:
        print(f"\n  Issues Found: {len(issues)}")
        print("-" * 60)
        for kind, location, detail in issues:
            loc = os.path.basename(location) if os.sep in location else location
            print(f"  [{kind:<10}] {loc:<35} {detail}")
    else:
        print("\n  No issues found. All segments look good!")

    print("\n" + "=" * 70)
    return issues


# ═══════════════════════════════════════════════════════════════════════════════
#  NAMING VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════


def validate_naming(directory: str, is_raw: bool = True) -> list:
    """
    Validate file naming in a directory tree.

    *Raw* files: ``<SpeakerId>_<token>_raw.wav``
    *Segments*:  ``<SpeakerId>_<numericId>_<rep>.wav``
    """
    pattern_str = "**/*_raw.wav" if is_raw else "**/*.wav"
    wavs = sorted(glob.glob(os.path.join(directory, pattern_str), recursive=True))
    if not wavs:
        print(f"No matching files found in {directory}")
        return []

    file_re = RAW_FILE_PATTERN if is_raw else SEGMENT_FILE_PATTERN
    all_issues: list = []
    valid_count = 0

    for wav_path in wavs:
        basename = os.path.basename(wav_path)
        match = file_re.match(basename)
        if not match:
            all_issues.append(
                (wav_path, [f"Does not match expected pattern: {basename}"])
            )
        else:
            issues: list = []
            if is_raw:
                speaker = match.group(1).upper()
                token = match.group(2).lower()
                if not SPEAKER_PATTERN.match(speaker):
                    issues.append(f"Invalid speaker ID '{speaker}'")
                if token not in VALID_TOKENS:
                    issues.append(f"Unknown token '{token}'")
            else:
                numeric_id = match.group(3)
                if numeric_id not in VALID_NUMERIC_IDS:
                    issues.append(f"Unknown numeric ID '{numeric_id}'")

            if issues:
                all_issues.append((wav_path, issues))
            else:
                valid_count += 1

    print(f"\nValidated {len(wavs)} files in {directory}")
    print(f"  Valid:   {valid_count}")
    print(f"  Issues:  {len(all_issues)}")
    if all_issues:
        print("\nFiles with issues:")
        for path, iss in all_issues:
            print(f"  {os.path.basename(path)}")
            for i in iss:
                print(f"    -> {i}")

    return all_issues


def fix_naming(directory: str, is_raw: bool = True, dry_run: bool = True) -> list:
    """Auto-fix common naming issues (spaces, case) in raw files."""
    pattern_str = "**/*_raw.wav" if is_raw else "**/*.wav"
    wavs = sorted(glob.glob(os.path.join(directory, pattern_str), recursive=True))
    renames = []
    for wav_path in wavs:
        basename = os.path.basename(wav_path)
        fixed = basename.replace(" ", "_")
        parts = fixed.split("_", 1)
        if len(parts) == 2 and SPEAKER_PATTERN.match(parts[0].upper()):
            fixed = parts[0].upper() + "_" + parts[1].lower()
        if fixed != basename:
            renames.append((wav_path, os.path.join(os.path.dirname(wav_path), fixed)))

    if not renames:
        print("No files need renaming.")
        return renames

    for old, new in renames:
        action = "WOULD RENAME" if dry_run else "RENAMED"
        print(f"  {action}: {os.path.basename(old)} -> {os.path.basename(new)}")
        if not dry_run:
            os.rename(old, new)

    if dry_run:
        print(f"\n  {len(renames)} files would be renamed. Pass dry_run=False to apply.")

    return renames
