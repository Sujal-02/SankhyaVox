"""
SankhyaVox – k-NN + DTW Baseline Classifier.

k-Nearest Neighbours with Dynamic Time Warping distance on
per-utterance normalised MFCC sequences.  Uses Sakoe-Chiba band
constraint for speed and distance-weighted voting for accuracy.
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np


class KNNDTWClassifier:
    """
    k-NN classifier with DTW distance on MFCC sequences.

    Parameters
    ----------
    k : int
        Number of neighbours.
    sakoe_chiba_radius : int
        DTW warping-band width (frames).  Constrains alignment to a
        diagonal band, improving both speed and accuracy.
    checkpoint_path : str or Path, optional
        If given, load a previously saved model.
    """

    def __init__(
        self,
        k: int = 3,
        sakoe_chiba_radius: int = 10,
        checkpoint_path: Optional[str] = None,
    ):
        self.k = k
        self.sakoe_chiba_radius = sakoe_chiba_radius
        self._X: list[np.ndarray] = []
        self._y: list[int] = []

        if checkpoint_path:
            self.load(checkpoint_path)

    @staticmethod
    def _transform(features: np.ndarray) -> np.ndarray:
        """
        Transform MFCC features for DTW comparison.

        1. Keeps only the first 13 static MFCC coefficients, stripping
           delta and delta-delta.  DTW already captures temporal dynamics
           through its elastic warping, so explicit deltas are redundant
           and add noise that hurts template matching.
        2. Applies per-utterance z-normalisation (zero mean, unit variance
           per coefficient) so distances measure spectral-shape similarity
           rather than absolute magnitude — making the classifier robust
           to recording-level gain and channel variation.
        """
        feat = features[:, :13]
        mean = feat.mean(axis=0, keepdims=True)
        std = feat.std(axis=0, keepdims=True)
        std[std < 1e-8] = 1.0
        return (feat - mean) / std

    def _dtw_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute DTW distance with Sakoe-Chiba band constraint."""
        n, m = len(a), len(b)
        r = self.sakoe_chiba_radius
        cost = np.full((n + 1, m + 1), np.inf)
        cost[0, 0] = 0.0
        for i in range(1, n + 1):
            j_start = max(1, i - r)
            j_end = min(m, i + r)
            for j in range(j_start, j_end + 1):
                d = np.sum((a[i - 1] - b[j - 1]) ** 2)
                cost[i, j] = d + min(
                    cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1]
                )
        return float(np.sqrt(cost[n, m]))

    def fit(self, X: list[np.ndarray], y: list[int]) -> "KNNDTWClassifier":
        """
        Store transformed training data (lazy — no model fitting).

        Parameters
        ----------
        X : list of ndarray, each (n_frames, 39)
        y : list of int labels
        """
        self._X = [self._transform(x) for x in X]
        self._y = list(y)
        return self

    def _predict_one(self, query: np.ndarray) -> int:
        """Predict with distance-weighted k-NN voting."""
        dists = np.array([self._dtw_distance(query, ref) for ref in self._X])
        idx = np.argsort(dists)[: self.k]
        # Inverse-distance weighted voting
        weights: dict[int, float] = {}
        for i in idx:
            w = 1.0 / (dists[i] + 1e-8)
            weights[self._y[i]] = weights.get(self._y[i], 0.0) + w
        return max(weights, key=lambda l: weights[l])

    def predict(self, X: list[np.ndarray]) -> np.ndarray:
        """Predict class labels for a list of feature sequences."""
        transformed = [self._transform(x) for x in X]
        return np.array([self._predict_one(x) for x in transformed])

    def save(self, path: str) -> None:
        """Save model (stored training templates) to a pickle file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "k": self.k,
                "sakoe_chiba_radius": self.sakoe_chiba_radius,
                "X": self._X,
                "y": self._y,
            }, f)
        print(f"Saved KNNDTWClassifier -> {path}")

    def load(self, path: str) -> None:
        """Load model from a pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.k = data["k"]
        self.sakoe_chiba_radius = data["sakoe_chiba_radius"]
        self._X = data["X"]
        self._y = data["y"]
        print(f"Loaded KNNDTWClassifier <- {path}")
