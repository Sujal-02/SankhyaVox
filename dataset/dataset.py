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
        self._root = (Path(processed_dir) if processed_dir else PROCESSED_DIR).resolve()
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
        npy_abs = str(self._root / row["npy_path"])
        wav_abs = str(self._root / row["wav_path"])
        return {
            "audio_path": wav_abs,
            "audio_source": row["category"],
            "speaker_id": row["speaker"],
            "token": row["sanskrit_label"],
            "label": int(row["label"]),
            "feature": np.load(npy_abs),
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
        """Return a detailed formatted summary of the dataset contents.

        For each category present, prints a table with speakers as columns,
        tokens as rows, and sample counts as cell values plus a Total column.
        Then prints a consolidated table aggregating across all speakers per
        category.
        """
        if self._df.empty:
            return repr(self) + "\n  (no samples loaded)"

        lines = [repr(self)]
        lines.append(f"  Speakers: {', '.join(self.speakers)}")
        lines.append(f"  Tokens:   {', '.join(self.tokens)}")
        lines.append("")

        # Build token display labels: "token (label)"
        token_label_map = (
            self._df[["sanskrit_label", "label"]]
            .drop_duplicates()
            .sort_values("label")
        )
        ordered_tokens = token_label_map["sanskrit_label"].tolist()
        token_display = {
            row.sanskrit_label: f"{row.sanskrit_label} ({row.label})"
            for row in token_label_map.itertuples()
        }

        # ── Per-category detail tables ────────────────────────────────────
        for cat in self.CATEGORIES:
            sub = self._df[self._df["category"] == cat]
            if sub.empty:
                continue

            speakers = sorted(sub["speaker"].unique())
            lines.append(f"  ┌─ {cat.upper()} ({len(sub)} samples, "
                         f"{len(speakers)} speakers) ─────────")

            # Build cross-tab: rows=token, cols=speaker
            ct = pd.crosstab(sub["sanskrit_label"], sub["speaker"])
            # Ensure all tokens and speakers present
            ct = ct.reindex(index=ordered_tokens, columns=speakers, fill_value=0)
            ct["Total"] = ct.sum(axis=1)

            # Format header
            row_label_width = max(len(d) for d in token_display.values()) + 2
            col_width = max(max((len(s) for s in speakers), default=5), 5) + 1
            total_width = max(5, len("Total")) + 1

            header = "  " + "".rjust(row_label_width)
            for sp in speakers:
                header += sp.rjust(col_width)
            header += "Total".rjust(total_width)
            lines.append(header)
            lines.append("  " + "─" * (row_label_width + col_width * len(speakers) + total_width))

            for tok in ordered_tokens:
                display = token_display[tok]
                row_str = "  " + display.rjust(row_label_width)
                for sp in speakers:
                    row_str += str(ct.at[tok, sp]).rjust(col_width)
                row_str += str(ct.at[tok, "Total"]).rjust(total_width)
                lines.append(row_str)

            # Column totals
            lines.append("  " + "─" * (row_label_width + col_width * len(speakers) + total_width))
            totals_row = "  " + "Total".rjust(row_label_width)
            for sp in speakers:
                totals_row += str(int(ct[sp].sum())).rjust(col_width)
            totals_row += str(int(ct["Total"].sum())).rjust(total_width)
            lines.append(totals_row)
            lines.append("")

        # ── Consolidated table across categories ──────────────────────────
        cats_present = [c for c in self.CATEGORIES if c in self._df["category"].values]
        if len(cats_present) > 1:
            lines.append("  ┌─ CONSOLIDATED (all categories) ─────────")

            ct_all = pd.crosstab(self._df["sanskrit_label"], self._df["category"])
            ct_all = ct_all.reindex(index=ordered_tokens, columns=cats_present, fill_value=0)
            ct_all["Total"] = ct_all.sum(axis=1)

            row_label_width = max(len(d) for d in token_display.values()) + 2
            cat_width = max(max((len(c) for c in cats_present), default=9), 9) + 1
            total_width = max(5, len("Total")) + 1

            header = "  " + "".rjust(row_label_width)
            for c in cats_present:
                header += c.rjust(cat_width)
            header += "Total".rjust(total_width)
            lines.append(header)
            lines.append("  " + "─" * (row_label_width + cat_width * len(cats_present) + total_width))

            for tok in ordered_tokens:
                display = token_display[tok]
                row_str = "  " + display.rjust(row_label_width)
                for c in cats_present:
                    row_str += str(ct_all.at[tok, c]).rjust(cat_width)
                row_str += str(ct_all.at[tok, "Total"]).rjust(total_width)
                lines.append(row_str)

            lines.append("  " + "─" * (row_label_width + cat_width * len(cats_present) + total_width))
            totals_row = "  " + "Total".rjust(row_label_width)
            for c in cats_present:
                totals_row += str(int(ct_all[c].sum())).rjust(cat_width)
            totals_row += str(int(ct_all["Total"].sum())).rjust(total_width)
            lines.append(totals_row)
            lines.append("")

        return "\n".join(lines)
