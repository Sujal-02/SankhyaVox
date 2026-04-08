"""
SankhyaVox – Audio Segmentor.

Splits raw repeated-utterance recordings into individually labelled
single-utterance WAV clips.

Each raw file contains multiple repetitions of a single token separated
by silence.  This module detects speech regions via energy-based VAD,
refines outlier-long segments using cross-correlation, and writes each
segment following the naming convention:

    <SpeakerId>_<numericId>_<rep>.wav      e.g.  S01_001_03.wav

Public API
----------
segment_file   – segment a single raw recording
segment_all    – walk a directory tree and segment every *_raw.wav
qa_segments    – run QA checks on segmented files
validate_naming / fix_naming – check / auto-fix file naming
"""

import glob
import os
import re
from typing import List, Tuple

import numpy as np
import soundfile as sf
from scipy.ndimage import binary_closing, binary_opening
from scipy.signal import butter, correlate, filtfilt, find_peaks
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

# ── Constants ──────────────────────────────────────────────────────────────────

_FRAME_MS = 25.0
_SHIFT_MS = 10.0
_OUTLIER_RATIO = 1.5   # segments longer than this × median get re-split
_SHORT_RATIO = 0.5     # segments shorter than this × median get merged with neighbour


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════


def _load_and_preprocess(path: str) -> Tuple[np.ndarray, int]:
    """Load audio, convert to mono, resample to SAMPLE_RATE, and high-pass."""
    audio, fs = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if fs != SAMPLE_RATE:
        from scipy.signal import resample_poly
        audio = resample_poly(audio, SAMPLE_RATE, fs).astype(np.float32)
        fs = SAMPLE_RATE
    audio = audio.astype(np.float32)

    # 4th-order Butterworth high-pass to remove DC / low-freq rumble
    nyq = 0.5 * fs
    b, a = butter(4, HIGHPASS_CUTOFF / nyq, btype="high")
    audio = filtfilt(b, a, audio).astype(np.float32)
    return audio, fs


def _vad(audio: np.ndarray, fs: int) -> List[Tuple[int, int]]:
    """Energy-based VAD → list of (start_sample, end_sample) boundaries.

    1. Compute short-time RMS energy.
    2. Adaptive threshold between noise floor and peak.
    3. Morphological close/open to smooth.
    4. Extract contiguous regions with padding.
    """
    frame_len = int(_FRAME_MS / 1000 * fs)
    frame_shift = int(_SHIFT_MS / 1000 * fs)

    # RMS energy per frame
    rms = np.array([
        np.sqrt(np.mean(audio[i : i + frame_len] ** 2))
        for i in range(0, len(audio) - frame_len, frame_shift)
    ])

    # Adaptive threshold: noise_floor + ratio * (peak - noise_floor)
    noise_floor = np.median(rms)
    peak = np.percentile(rms, 95)
    is_speech = rms > (noise_floor + ENERGY_THRESHOLD * (peak - noise_floor))

    # Morphological smoothing (in frames)
    close_k = int(MIN_SILENCE_DUR_S / (_SHIFT_MS / 1000))
    open_k = int(MIN_SPEECH_DUR_S / (_SHIFT_MS / 1000))
    is_speech = binary_closing(is_speech, structure=np.ones(close_k))
    is_speech = binary_opening(is_speech, structure=np.ones(open_k))

    # Extract contiguous regions (frame indices → sample indices)
    pad = int(PAD_MS / _SHIFT_MS)
    boundaries: List[Tuple[int, int]] = []
    in_seg, start = False, 0
    for i, active in enumerate(is_speech):
        if active and not in_seg:
            start = max(0, i - pad)
            in_seg = True
        elif not active and in_seg:
            boundaries.append((start, min(len(is_speech), i + pad)))
            in_seg = False
    if in_seg:
        boundaries.append((start, len(is_speech)))

    # Frame → sample
    return [(s * frame_shift, e * frame_shift) for s, e in boundaries]


def _find_majority(durations: np.ndarray) -> Tuple[float, float]:
    """Find the majority cluster of segment durations.

    Returns ``(center, spread)`` where *center* is the median duration of
    the largest cluster and *spread* is its IQR.  Segments whose duration
    falls within ``center ± 1.5 × spread`` are considered "normal".

    Uses a simple 1-D clustering: sort durations, find the longest run
    of values where consecutive gaps are small (< 50% of running mean).
    """
    if len(durations) < 3:
        med = float(np.median(durations))
        return med, med * 0.25

    order = np.argsort(durations)
    sorted_d = durations[order]

    # Walk sorted durations; break into a new group when the jump exceeds
    # 50 % of the current group mean.
    groups: List[List[float]] = [[sorted_d[0]]]
    for d in sorted_d[1:]:
        group_mean = np.mean(groups[-1])
        if d - group_mean > 0.5 * group_mean:
            groups.append([d])
        else:
            groups[-1].append(d)

    # Majority = largest group
    majority = max(groups, key=len)
    majority = np.array(majority)
    center = float(np.median(majority))
    q1, q3 = float(np.percentile(majority, 25)), float(np.percentile(majority, 75))
    spread = max(q3 - q1, center * 0.15)  # floor at 15 % of center
    return center, spread


def _pick_template(clips: List[np.ndarray]) -> np.ndarray:
    """Choose the clip whose waveform best represents the group.

    Resamples all clips to a common length, then picks the one with the
    highest average normalised cross-correlation against every other clip.
    """
    # Resample all clips to the median length for fair comparison
    target_len = int(np.median([len(c) for c in clips]))
    resampled = [np.interp(
        np.linspace(0, 1, target_len), np.linspace(0, 1, len(c)), c
    ) for c in clips]

    # Pairwise peak cross-correlation (symmetric, so compute upper triangle)
    n = len(resampled)
    scores = np.zeros(n)
    for i in range(n):
        a = (resampled[i] - resampled[i].mean()) / (resampled[i].std() + 1e-12)
        for j in range(i + 1, n):
            b = (resampled[j] - resampled[j].mean()) / (resampled[j].std() + 1e-12)
            peak = np.max(correlate(a, b, mode="same")) / target_len
            scores[i] += peak
            scores[j] += peak

    return clips[int(np.argmax(scores))]


def _refine(
    segs: List[Tuple[int, int]], audio: np.ndarray, fs: int,
) -> List[np.ndarray]:
    """Majority-based two-pass refinement of VAD segments.

    1. Compute durations of all initial segments.
    2. Find the **majority cluster** — the largest group of segments with
       similar duration.  This represents the "true" single-utterance length.
    3. **Merge** consecutive segments that are too short compared to the
       majority (likely split syllables of one word).
    4. **Split** segments that are too long compared to the majority
       (likely merged repetitions) using cross-correlation with a template
       built from majority-duration clips.
    """
    if len(segs) < 3:
        return [audio[s:e] for s, e in segs]

    durations = np.array([(e - s) / fs for s, e in segs])
    center, spread = _find_majority(durations)
    short_thresh = center - 1.5 * spread   # below this → merge
    long_thresh = center + 1.5 * spread     # above this → split

    # ── Pass 1: merge consecutive short segments ──────────────────────────
    merged: List[Tuple[int, int]] = [segs[0]]
    for s, e in segs[1:]:
        prev_s, prev_e = merged[-1]
        prev_dur = (prev_e - prev_s) / fs
        curr_dur = (e - s) / fs
        if prev_dur < short_thresh or curr_dur < short_thresh:
            merged[-1] = (prev_s, e)  # bridge the gap
        else:
            merged.append((s, e))
    segs = merged

    # Extract clips
    clips = [audio[s:e] for s, e in segs]

    # ── Pass 2: split long segments via cross-correlation ─────────────────
    # Build template from majority-duration clips by picking the one with
    # highest average waveform similarity to the others.
    normal = [c for c in clips if len(c) / fs <= long_thresh]
    if not normal or len(normal) < 2:
        return clips

    tmpl = _pick_template(normal)

    refined: List[np.ndarray] = []
    for clip in clips:
        if len(clip) / fs <= long_thresh:
            refined.append(clip)
            continue

        # Normalised cross-correlation
        c_norm = (clip - clip.mean()) / (clip.std() + 1e-12)
        t_norm = (tmpl - tmpl.mean()) / (tmpl.std() + 1e-12)
        xcorr = correlate(c_norm, t_norm, mode="same") / len(tmpl)

        peaks, _ = find_peaks(
            xcorr, height=0.25 * xcorr.max(), distance=int(len(tmpl) * 0.6)
        )

        if len(peaks) < 2:
            refined.append(clip)
            continue

        half = len(tmpl) // 2
        for i, pk in enumerate(peaks):
            s = max(0, pk - half) if i == 0 else (peaks[i - 1] + pk) // 2
            e = min(len(clip), pk + half) if i == len(peaks) - 1 else (pk + peaks[i + 1]) // 2
            refined.append(clip[s:e])

    return refined


def segment_file(
    input_path: str,
    speaker_id: str,
    numeric_id: str,
    output_dir: str,
    expected_reps: int = EXPECTED_REPS,
) -> int:
    """Segment one raw recording into individual WAV clips.

    Pipeline: load → high-pass → VAD → merge short → split long → write.
    Returns the number of segments written.
    """
    audio, fs = _load_and_preprocess(input_path)
    segments = _vad(audio, fs)
    clips = _refine(segments, audio, fs)

    if len(clips) != expected_reps:
        print(
            f"  WARNING: Expected {expected_reps}, found {len(clips)} "
            f"in {input_path}"
        )

    os.makedirs(output_dir, exist_ok=True)
    for idx, clip in enumerate(clips, 1):
        fname = f"{speaker_id}_{numeric_id}_{idx:02d}.wav"
        sf.write(os.path.join(output_dir, fname), clip, fs)

    return len(clips)


def segment_all(raw_dir: str, out_dir: str) -> int:
    """Walk *raw_dir* for ``*_raw.wav`` files and segment each one.

    Returns the total number of segments extracted.
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

        n = segment_file(wav_path, speaker, numeric_id,
                         os.path.join(out_dir, speaker))
        total += n

    print(f"Done. Total segments: {total}")
    return total


# ═══════════════════════════════════════════════════════════════════════════════
#  SEGMENT QUALITY ASSURANCE
# ═══════════════════════════════════════════════════════════════════════════════


def qa_segments(seg_dir: str) -> list:
    """Run QA checks on all segmented files.

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

        speaker, numeric_id, _ = parts
        if numeric_id not in VALID_NUMERIC_IDS:
            issues.append(("NAMING", wav_path, f"Unknown numeric ID '{numeric_id}'"))
            continue

        speaker_tokens.setdefault(speaker, {}).setdefault(numeric_id, []).append(
            wav_path
        )

        try:
            dur = sf.info(wav_path).duration
            if dur < QA_MIN_DURATION:
                issues.append(("SHORT", wav_path, f"{dur:.3f}s < {QA_MIN_DURATION}s"))
            elif dur > QA_MAX_DURATION:
                issues.append(("LONG", wav_path, f"{dur:.3f}s > {QA_MAX_DURATION}s"))
        except Exception as e:
            issues.append(("READ_ERR", wav_path, str(e)))

    # ── Report ─────────────────────────────────────────────────────────────
    expected_ids = set(TOKEN_TO_NUMERIC_ID.values())
    print("=" * 70)
    print("  SankhyaVox — Segment QA Report")
    print("=" * 70)
    print(f"\n  Total files : {len(wavs)}")
    print(f"  Speakers    : {len(speaker_tokens)}")
    print(f"\n{'Speaker':<10} {'Tokens':<8} {'Files':<8} {'Missing'}")
    print("-" * 60)

    for speaker in sorted(speaker_tokens):
        found = speaker_tokens[speaker]
        missing = expected_ids - set(found.keys())
        n_files = sum(len(v) for v in found.values())
        labels = [NUMERIC_ID_TO_TOKEN.get(m, m) for m in sorted(missing)]
        print(f"  {speaker:<8} {len(found):<8} {n_files:<8} "
              f"{', '.join(labels) if labels else '—'}")

        for nid in sorted(found):
            if len(found[nid]) != EXPECTED_REPS:
                issues.append((
                    "REP_COUNT", f"{speaker}/{nid}",
                    f"Expected {EXPECTED_REPS}, found {len(found[nid])}",
                ))

    if issues:
        print(f"\n  Issues Found: {len(issues)}")
        print("-" * 60)
        for kind, loc, detail in issues:
            name = os.path.basename(loc) if os.sep in loc else loc
            print(f"  [{kind:<10}] {name:<35} {detail}")
    else:
        print("\n  No issues found. All segments look good!")
    print("\n" + "=" * 70)
    return issues


# ═══════════════════════════════════════════════════════════════════════════════
#  NAMING VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════


def validate_naming(directory: str, is_raw: bool = True) -> list:
    """Validate file naming in a directory tree."""
    pattern = "**/*_raw.wav" if is_raw else "**/*.wav"
    wavs = sorted(glob.glob(os.path.join(directory, pattern), recursive=True))
    if not wavs:
        print(f"No matching files found in {directory}")
        return []

    file_re = RAW_FILE_PATTERN if is_raw else SEGMENT_FILE_PATTERN
    all_issues: list = []
    valid = 0

    for wav_path in wavs:
        basename = os.path.basename(wav_path)
        match = file_re.match(basename)
        if not match:
            all_issues.append((wav_path, [f"Does not match pattern: {basename}"]))
            continue

        errs: list = []
        if is_raw:
            if not SPEAKER_PATTERN.match(match.group(1).upper()):
                errs.append(f"Invalid speaker ID '{match.group(1)}'")
            if match.group(2).lower() not in VALID_TOKENS:
                errs.append(f"Unknown token '{match.group(2)}'")
        else:
            if match.group(3) not in VALID_NUMERIC_IDS:
                errs.append(f"Unknown numeric ID '{match.group(3)}'")

        if errs:
            all_issues.append((wav_path, errs))
        else:
            valid += 1

    print(f"\nValidated {len(wavs)} files — Valid: {valid}, Issues: {len(all_issues)}")
    for path, errs in all_issues:
        print(f"  {os.path.basename(path)}")
        for e in errs:
            print(f"    -> {e}")
    return all_issues


def fix_naming(directory: str, is_raw: bool = True, dry_run: bool = True) -> list:
    """Auto-fix common naming issues (spaces, case) in raw files."""
    pattern = "**/*_raw.wav" if is_raw else "**/*.wav"
    wavs = sorted(glob.glob(os.path.join(directory, pattern), recursive=True))
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
        tag = "WOULD RENAME" if dry_run else "RENAMED"
        print(f"  {tag}: {os.path.basename(old)} -> {os.path.basename(new)}")
        if not dry_run:
            os.rename(old, new)

    if dry_run:
        print(f"\n  {len(renames)} files would be renamed. Pass dry_run=False to apply.")
    return renames
