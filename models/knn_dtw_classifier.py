"""
SankhyaVox – k-NN + DTW Baseline Classifier.

Uses Dynamic Time Warping distance between variable-length MFCC
sequences and k-Nearest Neighbours for classification.
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute DTW distance between two feature sequences."""
    n, m = len(a), len(b)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = np.sum((a[i - 1] - b[j - 1]) ** 2)
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    return float(np.sqrt(cost[n, m]))


class KNNDTWClassifier:
    """
    k-NN classifier with DTW distance on raw MFCC sequences.

    Parameters
    ----------
    k : int
        Number of neighbours.
    checkpoint_path : str or Path, optional
        If given, load a previously saved model.
    """

    def __init__(
        self,
        k: int = 5,
        checkpoint_path: Optional[str] = None,
    ):
        self.k = k
        self._X: list[np.ndarray] = []
        self._y: list[int] = []

        if checkpoint_path:
            self.load(checkpoint_path)

    def fit(self, X: list[np.ndarray], y: list[int]) -> "KNNDTWClassifier":
        """Store training data (lazy — no model fitting needed)."""
        self._X = list(X)
        self._y = list(y)
        return self

    def predict_one(self, query: np.ndarray) -> int:
        """Predict label for a single feature sequence."""
        dists = [_dtw_distance(query, ref) for ref in self._X]
        idx = np.argsort(dists)[: self.k]
        labels = [self._y[i] for i in idx]
        # Majority vote
        counts: dict[int, int] = {}
        for l in labels:
            counts[l] = counts.get(l, 0) + 1
        return max(counts, key=lambda l: counts[l])

    def predict(self, X: list[np.ndarray]) -> np.ndarray:
        """Predict class labels for a list of feature sequences."""
        return np.array([self.predict_one(x) for x in X])

    def score(self, X: list[np.ndarray], y: list[int]) -> float:
        """Return accuracy on the given data."""
        preds = self.predict(X)
        return float(np.mean(preds == np.array(y)))

    def save(self, path: str) -> None:
        """Save training data to a pickle file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"k": self.k, "X": self._X, "y": self._y}, f)
        print(f"Saved KNNDTWClassifier -> {path}")

    def load(self, path: str) -> None:
        """Load training data from a pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.k = data["k"]
        self._X = data["X"]
        self._y = data["y"]
        print(f"Loaded KNNDTWClassifier <- {path}")
