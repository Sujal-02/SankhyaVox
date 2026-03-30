"""
SankhyaVox – Dataset class.

Lightweight, CSV-backed indexed dataset for training and evaluation.
Loads pre-generated metadata CSVs (``human.csv``, ``tts.csv``,
``augmented.csv``) from the processed data directory instead of
scanning the filesystem on every init.

CSV columns:
    npy_path, wav_path, speaker, numeric_id, label, sanskrit_label, rep
"""

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR


class SankhyaVoxDataset:
    """
    Unified dataset for SankhyaVox training and evaluation.

    Reads ``<processed_dir>/human.csv``, ``tts.csv``, ``augmented.csv``
    and concatenates them into a single DataFrame.

    Parameters
    ----------
    processed_dir : Path or str, optional
        Root of the processed data tree.  Default: ``config.PROCESSED_DIR``.
    categories : list of str, optional
        Which categories to load.  Default: all CSVs present on disk.

    ``ds[i]`` returns::

        {
            "audio_path":   str,      # path to source .wav segment
            "audio_source": str,      # "human" | "tts" | "augmented"
            "speaker_id":   str,      # e.g. "S01"
            "token":        str,      # Sanskrit label, e.g. "eka"
            "label":        int,      # numeric value, e.g. 1
            "feature":      ndarray,  # shape (n_frames, 39)
        }
    """

    CATEGORIES = ("human", "tts", "augmented")

    def __init__(
        self,
        processed_dir: Optional[Path] = None,
        categories: Optional[List[str]] = None,
    ):
        self._root = Path(processed_dir) if processed_dir else PROCESSED_DIR
        self._categories = categories or list(self.CATEGORIES)
        self._df = self._load_csvs()

    # ── Loading ───────────────────────────────────────────────────────────

    def _load_csvs(self) -> pd.DataFrame:
        """Load and concatenate category CSV files."""
        frames = []
        for cat in self._categories:
            csv_path = self._root / f"{cat}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df["category"] = cat
                frames.append(df)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def df(self) -> pd.DataFrame:
        """Full metadata DataFrame."""
        return self._df

    @property
    def speakers(self) -> List[str]:
        """Sorted list of unique speaker IDs."""
        if self._df.empty:
            return []
        return sorted(self._df["speaker"].unique().tolist())

    @property
    def tokens(self) -> List[str]:
        """Sorted list of unique Sanskrit token labels."""
        if self._df.empty:
            return []
        return sorted(self._df["sanskrit_label"].unique().tolist())

    # ── Indexing ──────────────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a sample dict for the given global index."""
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

    # ── Batch access ──────────────────────────────────────────────────────

    def get_Xy(self) -> Tuple[List[np.ndarray], List[int]]:
        """Return ``(features_list, labels_list)`` for model training."""
        X, y = [], []
        for i in range(len(self)):
            s = self[i]
            X.append(s["feature"])
            y.append(s["label"])
        return X, y

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
