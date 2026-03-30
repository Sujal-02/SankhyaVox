"""
SankhyaVox – Dataset class.

Clean, indexed interface over processed feature data for training and
evaluation.  Backed by a ``pandas.DataFrame`` with metadata derived from
the standardised file naming convention:

    <SpeakerId>_<numericId>_<rep>.npy

Supports global indexing and per-category views (human, tts, augmented).
"""

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    NUMERIC_ID_TO_TOKEN,
    PROCESSED_DIR,
    TOKEN_TO_VALUE,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  CATEGORY VIEW
# ═══════════════════════════════════════════════════════════════════════════════


class CategoryView:
    """Read-only indexed view over a single category of the dataset."""

    def __init__(self, df: pd.DataFrame):
        self._df = df.reset_index(drop=True)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a sample dict for the given index."""
        row = self._df.iloc[idx]
        return {
            "audio_path": row["wav_path"],
            "audio_source": row["category"],
            "speaker_id": row["speaker"],
            "token": row["sanskrit_label"],
            "label": int(row["label"]),
            "feature": np.load(row["npy_path"]),
        }

    def __len__(self) -> int:
        return len(self._df)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for i in range(len(self)):
            yield self[i]

    @property
    def df(self) -> pd.DataFrame:
        """Underlying metadata DataFrame for this category."""
        return self._df

    def __repr__(self) -> str:
        return f"CategoryView(samples={len(self)})"


# ═══════════════════════════════════════════════════════════════════════════════
#  SANKHYAVOX DATASET
# ═══════════════════════════════════════════════════════════════════════════════


class SankhyaVoxDataset:
    """
    Unified dataset for SankhyaVox training and evaluation.

    Scans the processed data directory and builds a pandas DataFrame
    with columns:

    ============== ================================================
    ``npy_path``   absolute path to the ``.npy`` feature file
    ``wav_path``   absolute path to the source ``.wav`` segment
    ``category``   ``"human"`` | ``"tts"`` | ``"augmented"``
    ``speaker``    speaker ID  (e.g. ``"S01"``, ``"TTS02"``)
    ``numeric_id`` 3-digit token ID  (e.g. ``"001"``)
    ``label``      integer value the token represents (e.g. ``1``)
    ``sanskrit_label`` token name  (e.g. ``"eka"``)
    ``rep``        repetition number
    ============== ================================================

    Parameters
    ----------
    processed_dir : Path, optional
        Root of the processed data tree.  Default: ``config.PROCESSED_DIR``.
    categories : list of str, optional
        Which categories to load.  Default: all present on disk.
    """

    CATEGORIES = ("human", "tts", "augmented")

    def __init__(
        self,
        processed_dir: Optional[Path] = None,
        categories: Optional[List[str]] = None,
    ):
        self._root = Path(processed_dir) if processed_dir else PROCESSED_DIR
        self._categories = categories or list(self.CATEGORIES)
        self._df = self._scan()

    # ── Scanning ──────────────────────────────────────────────────────────

    def _scan(self) -> pd.DataFrame:
        """Walk feature directories and build the metadata DataFrame."""
        rows: List[Dict[str, Any]] = []

        for category in self._categories:
            feat_dir = self._root / category / "features"
            seg_dir = self._root / category / "segments"

            if not feat_dir.exists():
                continue

            for npy_path in sorted(feat_dir.rglob("*.npy")):
                stem = npy_path.stem  # e.g. S01_001_01
                parts = stem.split("_")
                if len(parts) != 3 or not parts[2].isdigit():
                    continue

                speaker, numeric_id, rep_str = parts
                token = NUMERIC_ID_TO_TOKEN.get(numeric_id)
                if token is None:
                    continue

                label = TOKEN_TO_VALUE.get(token, -1)

                # Derive wav_path by mirroring the directory structure
                rel = npy_path.relative_to(feat_dir)
                wav_path = seg_dir / rel.with_suffix(".wav")

                rows.append(
                    {
                        "npy_path": str(npy_path),
                        "wav_path": str(wav_path),
                        "category": category,
                        "speaker": speaker,
                        "numeric_id": numeric_id,
                        "label": label,
                        "sanskrit_label": token,
                        "rep": int(rep_str),
                    }
                )

        return pd.DataFrame(rows)

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def df(self) -> pd.DataFrame:
        """Full metadata DataFrame."""
        return self._df

    @property
    def human(self) -> CategoryView:
        """View over human-recorded samples only."""
        return CategoryView(self._df[self._df["category"] == "human"])

    @property
    def tts(self) -> CategoryView:
        """View over TTS-generated samples only."""
        return CategoryView(self._df[self._df["category"] == "tts"])

    @property
    def augmented(self) -> CategoryView:
        """View over augmented samples only."""
        return CategoryView(self._df[self._df["category"] == "augmented"])

    @property
    def speakers(self) -> List[str]:
        """Sorted list of unique speaker IDs across all categories."""
        return sorted(self._df["speaker"].unique().tolist())

    @property
    def tokens(self) -> List[str]:
        """Sorted list of unique Sanskrit token labels."""
        return sorted(self._df["sanskrit_label"].unique().tolist())

    # ── Indexing ──────────────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Return a sample dict for the given global index.

        Keys:
            ``audio_path``   — path to the source ``.wav`` segment
            ``audio_source`` — category (``"human"``, ``"tts"``, ``"augmented"``)
            ``speaker_id``   — e.g. ``"S01"``
            ``token``        — Sanskrit label (e.g. ``"eka"``)
            ``label``        — integer value (e.g. ``1``)
            ``feature``      — MFCC ndarray of shape ``(n_frames, 39)``
        """
        row = self._df.iloc[idx]
        return {
            "audio_path": row["wav_path"],
            "audio_source": row["category"],
            "speaker_id": row["speaker"],
            "token": row["sanskrit_label"],
            "label": int(row["label"]),
            "feature": np.load(row["npy_path"]),
        }

    def __len__(self) -> int:
        return len(self._df)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for i in range(len(self)):
            yield self[i]

    # ── Filtering / Splitting ─────────────────────────────────────────────

    def filter(
        self,
        category: Optional[str] = None,
        speaker: Optional[str] = None,
        label: Optional[int] = None,
    ) -> "SankhyaVoxDataset":
        """Return a new dataset filtered by the given criteria."""
        mask = pd.Series(True, index=self._df.index)
        if category is not None:
            mask &= self._df["category"] == category
        if speaker is not None:
            mask &= self._df["speaker"] == speaker
        if label is not None:
            mask &= self._df["label"] == label

        filtered = SankhyaVoxDataset.__new__(SankhyaVoxDataset)
        filtered._root = self._root
        filtered._categories = self._categories
        filtered._df = self._df[mask].reset_index(drop=True)
        return filtered

    def split_by_speakers(
        self,
        train: List[str],
        val: List[str],
        test: List[str],
    ) -> Tuple["SankhyaVoxDataset", "SankhyaVoxDataset", "SankhyaVoxDataset"]:
        """Split dataset into train / val / test by speaker IDs."""

        def _subset(spk_list: List[str]) -> "SankhyaVoxDataset":
            ds = SankhyaVoxDataset.__new__(SankhyaVoxDataset)
            ds._root = self._root
            ds._categories = self._categories
            ds._df = self._df[self._df["speaker"].isin(spk_list)].reset_index(
                drop=True
            )
            return ds

        return _subset(train), _subset(val), _subset(test)

    # ── Display ───────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        cats = self._df["category"].value_counts().to_dict() if len(self._df) else {}
        cat_str = ", ".join(f"{k}={v}" for k, v in sorted(cats.items()))
        return f"SankhyaVoxDataset(samples={len(self)}, {cat_str})"

    def summary(self) -> str:
        """Return a formatted summary of the dataset contents."""
        lines = [repr(self)]
        if not self._df.empty:
            lines.append(f"  Speakers: {', '.join(self.speakers)}")
            lines.append(f"  Tokens:   {', '.join(self.tokens)}")
            for cat in self.CATEGORIES:
                sub = self._df[self._df["category"] == cat]
                if not sub.empty:
                    lines.append(
                        f"  [{cat}] {len(sub)} samples, "
                        f"{sub['speaker'].nunique()} speakers"
                    )
        return "\n".join(lines)
