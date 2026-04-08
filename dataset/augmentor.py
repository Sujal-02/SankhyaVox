"""
SankhyaVox – Audio Augmentor.

Applies pitch-shift and speed-perturbation augmentations to segmented
WAV files.  All permutations of ``AUG_PITCHES × AUG_SPEEDS`` are applied
to every source file.

Output naming convention
------------------------
Given a source speaker folder ``S01/`` containing ``S01_000_01.wav``,
the augmented output is written to::

    augmented/segments/augS01/augS01_000_01_p<i>_f<j>.wav

where ``<i>`` indexes the pitch setting and ``<j>`` indexes the speed
setting (both zero-based).  The identity permutation ``(pitch=0, speed=1.0)``
is skipped so no exact copy of the original is produced.

Public API
----------
augment(subject_dir)  – augment all WAVs under one speaker directory
"""

import os
from itertools import product
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from src.config import (
    AUG_PITCHES,
    AUG_SPEEDS,
    AUG_SEGMENT_DIR,
    SAMPLE_RATE,
)


def _augment_audio(
    audio: np.ndarray,
    sr: int,
    pitch_semitones: float,
    speed_factor: float,
) -> np.ndarray:
    """
    Apply pitch shift and speed perturbation to an audio signal.

    Parameters
    ----------
    audio : 1-D float array
    sr : sample rate
    pitch_semitones : pitch shift in semitones (0 = no shift)
    speed_factor : time-stretch factor (>1 = faster / higher freq, <1 = slower)

    Returns
    -------
    Augmented audio as 1-D float32 array at the same sample rate.
    """
    y = audio.copy()

    # Speed perturbation (changes duration *and* effective frequency content)
    if speed_factor != 1.0:
        y = librosa.effects.time_stretch(y, rate=speed_factor)

    # Pitch shift (independent of speed)
    if pitch_semitones != 0:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_semitones)

    return y.astype(np.float32)


def augment(
    subject_dir: str,
    output_root: Optional[str] = None,
    pitches: Optional[list] = None,
    speeds: Optional[list] = None,
) -> int:
    """
    Augment all WAV files under *subject_dir* with every combination
    of pitch and speed settings.

    Parameters
    ----------
    subject_dir : str
        Path to a single speaker segment folder, e.g.
        ``data_processed/human/segments/S01``
    output_root : str, optional
        Root directory for augmented output.  Defaults to
        ``config.AUG_SEGMENT_DIR``.
    pitches : list[float], optional
        Pitch shifts in semitones (default: ``AUG_PITCHES``).
    speeds : list[float], optional
        Speed factors (default: ``AUG_SPEEDS``).

    Returns
    -------
    Number of augmented files written.
    """
    subject_dir = Path(subject_dir)
    output_root = Path(output_root) if output_root else AUG_SEGMENT_DIR
    pitches = pitches if pitches is not None else AUG_PITCHES
    speeds = speeds if speeds is not None else AUG_SPEEDS

    speaker_name = subject_dir.name  # e.g. "S01" or "TTS02"
    aug_speaker = f"aug{speaker_name}"
    out_dir = output_root / aug_speaker
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(subject_dir.glob("*.wav"))
    if not wav_files:
        print(f"  No WAV files found in {subject_dir}")
        return 0

    combos = [
        (pi, p, fi, f)
        for (pi, p), (fi, f) in product(enumerate(pitches), enumerate(speeds))
    ]

    total = len(wav_files) * len(combos)
    count = 0
    with tqdm(total=total, desc=f"Augmenting {speaker_name}", unit="file") as pbar:
        for wav_path in wav_files:
            audio, sr = sf.read(str(wav_path))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
                sr = SAMPLE_RATE

            stem = wav_path.stem  # e.g. "S01_000_01"
            aug_stem = f"aug{stem}"

            for pi, pitch, fi, speed in combos:
                aug_audio = _augment_audio(audio, sr, pitch, speed)
                out_name = f"{aug_stem}_p{pi}_f{fi}.wav"
                out_path = out_dir / out_name
                sf.write(str(out_path), aug_audio, sr, subtype="PCM_16")
                count += 1
                pbar.update(1)

    print(f"  {speaker_name}: wrote {count} augmented files -> {out_dir}")
    return count
