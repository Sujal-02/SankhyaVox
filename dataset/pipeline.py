"""
SankhyaVox – Data Pipeline.

Single entry point for all data preparation: audio format conversion,
segmentation, TTS generation, feature extraction, validation, and
single-file inference preprocessing.

Usage::

    from dataset import DataPipeline

    pipe = DataPipeline()
    pipe.convert()                  # standardise raw audio to 16kHz mono WAV
    pipe.segment()                  # segment human raw recordings
    pipe.generate_tts()             # create TTS synthetic data
    pipe.extract_features()         # MFCC for all sources
    pipe.build()                    # full human pipeline in one call

    # Live / inference (single file, no segmentation)
    features = pipe.process_single("path/to/test.m4a")
"""

import glob
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from src.config import (
    APPLY_CMVN,
    AUG_FEATURE_DIR,
    AUG_SEGMENT_DIR,
    DATA_DIR,
    FEATURE_DIR,
    FRAME_LENGTH,
    FRAME_SHIFT,
    HUMAN_RAW_DIR,
    N_FFT,
    N_MELS,
    N_MFCC,
    NUMERIC_ID_TO_TOKEN,
    PRE_EMPHASIS,
    PROCESSED_DIR,
    SAMPLE_RATE,
    SEGMENT_DIR,
    TOKEN_TO_VALUE,
    TTS_FEATURE_DIR,
    TTS_REPS,
    TTS_SEGMENT_DIR,
    USE_DELTAS,
)


class DataPipeline:
    """
    Orchestrates all data preparation for SankhyaVox.

    Each method is idempotent and can be called independently.

    Parameters
    ----------
    data_dir : Path, optional
        Override DVC-tracked raw data root (default: ``config.DATA_DIR``).
    processed_dir : Path, optional
        Override processed output root (default: ``config.PROCESSED_DIR``).
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
    ):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.processed_dir = Path(processed_dir) if processed_dir else PROCESSED_DIR

        # Human data paths
        self.human_raw_dir = self.processed_dir / "human" / "raw"
        self.human_segment_dir = self.processed_dir / "human" / "segments"
        self.human_feature_dir = self.processed_dir / "human" / "features"

        # TTS data paths (generated audio goes directly to segments)
        self.tts_segment_dir = self.processed_dir / "tts" / "segments"
        self.tts_feature_dir = self.processed_dir / "tts" / "features"

        # Augmented data paths (reserved for future augmentation)
        self.aug_segment_dir = self.processed_dir / "augmented" / "segments"
        self.aug_feature_dir = self.processed_dir / "augmented" / "features"

    # ── Audio format conversion ───────────────────────────────────────────

    _AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".mp4", ".aac", ".flac", ".ogg", ".wma"}

    @staticmethod
    def _convert_single_file(
        input_path: str,
        output_path: str,
        sample_rate: int = SAMPLE_RATE,
    ) -> bool:
        """
        Convert a single audio file to 16 kHz mono 16-bit PCM WAV using ffmpeg.

        Returns ``True`` on success, ``False`` on failure.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # If already a conforming WAV, try a fast check before shelling out
        if input_path.lower().endswith(".wav"):
            try:
                info = sf.info(input_path)
                if (
                    info.samplerate == sample_rate
                    and info.channels == 1
                    and info.subtype == "PCM_16"
                ):
                    shutil.copy2(input_path, output_path)
                    return True
            except Exception:
                pass  # fall through to ffmpeg

        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ar", str(sample_rate),
            "-ac", "1",
            "-sample_fmt", "s16",
            "-loglevel", "error",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  FAILED: {os.path.basename(input_path)}: {result.stderr.strip()}")
            return False
        return True

    def convert(
        self,
        input_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> int:
        """
        Batch-convert raw audio from *input_dir* to standardised 16 kHz
        mono PCM WAV files in *output_dir*.

        Preserves the ``<SpeakerId>/`` subdirectory structure.

        Parameters
        ----------
        input_dir : str, optional
            Source directory (default: DVC-tracked ``data/``).
        output_dir : str, optional
            Destination (default: ``data_processed/human/raw/``).

        Returns
        -------
        Number of files successfully converted.
        """
        src = Path(input_dir) if input_dir else self.data_dir
        dst = Path(output_dir) if output_dir else self.human_raw_dir

        files = [
            p
            for p in sorted(src.rglob("*"))
            if p.is_file() and p.suffix.lower() in self._AUDIO_EXTENSIONS
        ]
        if not files:
            print(f"No audio files found in {src}")
            return 0

        count = 0
        for fpath in tqdm(files, desc="Converting", unit="file"):
            rel = fpath.relative_to(src)
            out_path = dst / rel.with_suffix(".wav")
            ok = self._convert_single_file(str(fpath), str(out_path))
            if ok:
                count += 1

        print(f"Done. Converted {count}/{len(files)} files -> {dst}")
        return count

    # ── Segmentation ──────────────────────────────────────────────────────

    def segment(
        self,
        raw_dir: Optional[str] = None,
        out_dir: Optional[str] = None,
    ) -> int:
        """Segment raw repeated-utterance recordings into individual clips."""
        from dataset.segmentor import segment_all

        rd = raw_dir or str(self.human_raw_dir)
        od = out_dir or str(self.human_segment_dir)
        return segment_all(rd, od)

    # ── TTS Generation ────────────────────────────────────────────────────

    def generate_tts(self, reps: int = TTS_REPS, **kwargs) -> int:
        """Generate TTS synthetic audio data."""
        from dataset.generator import generate

        return generate(str(self.tts_segment_dir), reps=reps, **kwargs)

    # ── Augmentation ──────────────────────────────────────────────────────

    def augment(self, source: str = "human") -> int:
        """
        Augment segmented WAVs with pitch/speed permutations.

        Parameters
        ----------
        source : str
            One of ``"human"``, ``"tts"``, ``"all"``, or an explicit path
            to a single speaker segment directory (e.g.
            ``data_processed/human/segments/S02``).

        Returns
        -------
        Total number of augmented files written.
        """
        from dataset.augmentor import augment as _augment

        if source == "all":
            return self.augment("human") + self.augment("tts")

        # Explicit path to a specific speaker folder
        p = Path(source)
        if p.is_dir():
            return _augment(str(p), output_root=str(self.aug_segment_dir))

        # Walk known category directories
        if source == "human":
            seg_root = self.human_segment_dir
        elif source == "tts":
            seg_root = self.tts_segment_dir
        else:
            raise ValueError(
                f"Unknown source '{source}'. Use 'human', 'tts', 'all', "
                "or a path to a speaker segment directory."
            )

        subjects = sorted(
            d for d in Path(seg_root).iterdir() if d.is_dir()
        )
        if not subjects:
            print(f"No speaker directories found in {seg_root}")
            return 0

        total = 0
        for subj in subjects:
            total += _augment(str(subj), output_root=str(self.aug_segment_dir))
        return total

    # ── Validation ────────────────────────────────────────────────────────

    def validate(
        self,
        source: str = "human",
        mode: str = "raw",
        fix: bool = False,
    ) -> list:
        """
        Validate file naming or run segment QA.

        Parameters
        ----------
        source : ``"human"`` or ``"tts"``
        mode : ``"raw"``, ``"segments"``, or ``"qa"``
        fix : auto-fix naming issues when ``True``
        """
        from dataset.segmentor import fix_naming, qa_segments, validate_naming

        if mode == "qa":
            seg_dir = str(
                self.human_segment_dir if source == "human" else self.tts_segment_dir
            )
            return qa_segments(seg_dir)

        is_raw = mode == "raw"
        if is_raw:
            target = str(self.human_raw_dir)
        else:
            target = str(
                self.human_segment_dir
                if source == "human"
                else self.tts_segment_dir
            )

        if fix:
            return fix_naming(target, is_raw=is_raw, dry_run=False)
        return validate_naming(target, is_raw=is_raw)

    # ── Feature extraction ────────────────────────────────────────────────

    @staticmethod
    def _preprocess_audio(audio: np.ndarray) -> np.ndarray:
        """DC removal → pre-emphasis → peak normalisation."""
        audio = audio - np.mean(audio)
        audio = np.append(audio[0], audio[1:] - PRE_EMPHASIS * audio[:-1])
        peak = np.max(np.abs(audio))
        if peak > 0:
            target_peak = 10 ** (-3.0 / 20.0)
            audio = audio * (target_peak / peak)
        return audio.astype(np.float32)

    @staticmethod
    def _extract_mfcc(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
        """Extract MFCC features (default 39-dim with deltas + CMVN)."""
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=FRAME_SHIFT,
            win_length=FRAME_LENGTH,
            n_mels=N_MELS,
            window="hamming",
        )
        features = mfcc
        if USE_DELTAS:
            delta = librosa.feature.delta(mfcc, order=1)
            delta2 = librosa.feature.delta(mfcc, order=2)
            features = np.vstack([mfcc, delta, delta2])
        features = features.T
        if APPLY_CMVN:
            mean = np.mean(features, axis=0, keepdims=True)
            std = np.std(features, axis=0, keepdims=True)
            std[std < 1e-10] = 1e-10
            features = (features - mean) / std
        return features.astype(np.float32)

    def _process_wav(self, wav_path: str, output_dir: str) -> str:
        """Extract features from a single WAV and save as ``.npy``."""
        audio, sr = sf.read(wav_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        audio = self._preprocess_audio(audio)
        features = self._extract_mfcc(audio)

        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(wav_path))[0]
        out_path = os.path.join(output_dir, f"{basename}.npy")
        np.save(out_path, features)
        return out_path

    def extract_features(self, source: Optional[str] = None) -> int:
        """
        Extract MFCC features for all segmented WAV files.

        Parameters
        ----------
        source : ``"human"``, ``"tts"``, ``"augmented"``, or ``None`` (all).
        """
        pairs = []
        if source is None or source == "human":
            pairs.append((self.human_segment_dir, self.human_feature_dir))
        if source is None or source == "tts":
            pairs.append((self.tts_segment_dir, self.tts_feature_dir))
        if source is None or source == "augmented":
            pairs.append((self.aug_segment_dir, self.aug_feature_dir))

        all_wavs = []
        for seg_dir, feat_dir in pairs:
            wavs = sorted(glob.glob(f"{seg_dir}/**/*.wav", recursive=True))
            for w in wavs:
                all_wavs.append((w, seg_dir, feat_dir))

        if not all_wavs:
            print("No WAV files found for feature extraction.")
            return 0

        for wav_path, seg_dir, feat_dir in tqdm(all_wavs, desc="Extracting features", unit="file"):
            rel = os.path.relpath(os.path.dirname(wav_path), str(seg_dir))
            out = os.path.join(str(feat_dir), rel)
            self._process_wav(wav_path, out)

        print(f"Done. Extracted features for {len(all_wavs)} files.")
        return len(all_wavs)

    # ── CSV metadata generation ───────────────────────────────────────────

    _CATEGORY_MAP = {
        "human":     ("human_segment_dir", "human_feature_dir"),
        "tts":       ("tts_segment_dir",   "tts_feature_dir"),
        "augmented": ("aug_segment_dir",   "aug_feature_dir"),
    }

    def generate_csv(self, category: str) -> Path:
        """
        Generate a metadata CSV for one category and save it to
        ``<processed_dir>/<category>.csv``.

        Columns: npy_path, wav_path, speaker, numeric_id, label,
        sanskrit_label, rep[, aug_pitch, aug_speed]
        """
        seg_attr, feat_attr = self._CATEGORY_MAP[category]
        feat_dir = Path(getattr(self, feat_attr))
        seg_dir  = Path(getattr(self, seg_attr))

        rows: List[Dict[str, Any]] = []
        for npy_path in sorted(feat_dir.rglob("*.npy")):
            parts = npy_path.stem.split("_")

            # Augmented files: augS01_000_01_p0_f1  → 5 parts
            if len(parts) == 5 and parts[3].startswith("p") and parts[4].startswith("f"):
                speaker = parts[0]          # "augS01"
                numeric_id = parts[1]       # "000"
                rep_str = parts[2]          # "01"
                aug_pitch = parts[3]        # "p0"
                aug_speed = parts[4]        # "f1"
            elif len(parts) == 3 and parts[2].isdigit():
                speaker, numeric_id, rep_str = parts
                aug_pitch = None
                aug_speed = None
            else:
                continue

            if not rep_str.isdigit():
                continue

            token = NUMERIC_ID_TO_TOKEN.get(numeric_id)
            if token is None:
                continue

            rel = npy_path.relative_to(feat_dir)
            wav_path = seg_dir / rel.with_suffix(".wav")

            row: Dict[str, Any] = {
                "npy_path":       npy_path.relative_to(self.processed_dir).as_posix(),
                "wav_path":       wav_path.relative_to(self.processed_dir).as_posix(),
                "speaker":        speaker,
                "numeric_id":     numeric_id,
                "label":          TOKEN_TO_VALUE.get(token, -1),
                "sanskrit_label": token,
                "rep":            int(rep_str),
            }
            if aug_pitch is not None:
                row["aug_pitch"] = aug_pitch
                row["aug_speed"] = aug_speed
            rows.append(row)

        csv_path = self.processed_dir / f"{category}.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"Saved {csv_path} ({len(rows)} rows)")
        return csv_path

    def generate_all_csvs(self) -> List[Path]:
        """Generate CSVs for all categories that have features on disk."""
        paths = []
        for cat in ("human", "tts", "augmented"):
            _, feat_attr = self._CATEGORY_MAP[cat]
            feat_dir = Path(getattr(self, feat_attr))
            if feat_dir.exists() and any(feat_dir.rglob("*.npy")):
                paths.append(self.generate_csv(cat))
        return paths

    # ── Full pipeline ─────────────────────────────────────────────────────

    def build(self, source: str = "human") -> None:
        """Run the full pipeline: convert → segment → QA → extract features → generate CSVs."""
        print("=== Converting raw audio ===")
        self.convert()
        print("\n=== Segmenting raw recordings ===")
        self.segment()
        print("\n=== Running segment QA ===")
        self.validate(source=source, mode="qa")
        print("\n=== Extracting features ===")
        self.extract_features(source=source)
        print("\n=== Generating metadata CSVs ===")
        self.generate_all_csvs()
        print("\n=== Pipeline complete ===")

    # ── Single-file inference ─────────────────────────────────────────────

    def process_single(
        self,
        audio_path: str,
        output_npy: Optional[str] = None,
    ) -> np.ndarray:
        """
        Standardise and extract features from a single audio file.

        Intended for live testing / inference where the user says the
        number once (no repeated-utterance segmentation needed).

        Steps: convert to 16 kHz mono → preprocess → extract MFCC.

        Parameters
        ----------
        audio_path : str
            Path to any supported audio file.
        output_npy : str, optional
            If given, save the feature array to this ``.npy`` path.

        Returns
        -------
        features : np.ndarray of shape ``(n_frames, 39)``
        """
        import tempfile

        src = Path(audio_path)
        needs_convert = True

        # Check if already conforming
        if src.suffix.lower() == ".wav":
            try:
                info = sf.info(str(src))
                if (
                    info.samplerate == SAMPLE_RATE
                    and info.channels == 1
                    and info.subtype == "PCM_16"
                ):
                    needs_convert = False
            except Exception:
                pass

        if needs_convert:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            ok = self._convert_single_file(str(src), tmp.name)
            if not ok:
                raise RuntimeError(f"Failed to convert {audio_path}")
            wav_path = tmp.name
        else:
            wav_path = str(src)

        try:
            audio, sr = sf.read(wav_path)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

            audio = self._preprocess_audio(audio)
            features = self._extract_mfcc(audio)

            if output_npy:
                os.makedirs(os.path.dirname(output_npy), exist_ok=True)
                np.save(output_npy, features)

            return features
        finally:
            if needs_convert:
                os.unlink(wav_path)
