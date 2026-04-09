"""
SankhyaVox – GMM Baseline Classifier.

Per-class Gaussian Mixture Model fitted on rich statistical MFCC summaries.
Classification by maximum log-likelihood across per-class GMMs.
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class GMMClassifier:
    """
    GMM baseline: one GMM per class, classify by max log-likelihood.

    Parameters
    ----------
    n_components : int
        Number of Gaussian components per class GMM.
    checkpoint_path : str or Path, optional
        If given, load a previously saved model.
    """

    def __init__(
        self,
        n_components: int = 8,
        checkpoint_path: Optional[str] = None,
    ):
        self.n_components = n_components
        self.models: dict[int, GaussianMixture] = {}
        self.scaler = StandardScaler()

        if checkpoint_path:
            self.load(checkpoint_path)

    @staticmethod
    def _transform(features: np.ndarray) -> np.ndarray:
        """
        Transform variable-length MFCC (n_frames, 39) to a fixed-length vector.

        Computes per-coefficient statistics: mean, std, min, max, median,
        q25, q75, and average frame-to-frame delta.
        Output: 39 * 8 = 312 dimensions.

        Rationale: The previous mean+std approach (78 dims) discards
        distribution shape and temporal dynamics entirely.  Adding
        min/max captures the coefficient range, percentiles capture
        distribution skew, and the delta-mean preserves the average
        rate-of-change across frames — all of which help GMMs
        discriminate between tokens that have similar means but
        different articulatory trajectories.
        """
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        fmin = features.min(axis=0)
        fmax = features.max(axis=0)
        med = np.median(features, axis=0)
        q25 = np.percentile(features, 25, axis=0)
        q75 = np.percentile(features, 75, axis=0)
        if len(features) > 1:
            delta_mean = np.diff(features, axis=0).mean(axis=0)
        else:
            delta_mean = np.zeros(features.shape[1])
        return np.concatenate([mean, std, fmin, fmax, med, q25, q75, delta_mean])

    def fit(self, X: list[np.ndarray], y: list[int]) -> "GMMClassifier":
        """
        Fit one GMM per class on transformed, scaled features.

        Parameters
        ----------
        X : list of ndarray, each (n_frames, feat_dim)
        y : list of int labels
        """
        # Transform all samples
        summaries: dict[int, list[np.ndarray]] = {}
        all_vecs = []
        for feat, label in zip(X, y):
            vec = self._transform(feat)
            summaries.setdefault(label, []).append(vec)
            all_vecs.append(vec)

        # Fit scaler on all training data, then scale
        self.scaler.fit(np.array(all_vecs))

        for label, vecs in summaries.items():
            data = self.scaler.transform(np.array(vecs))
            n_comp = min(self.n_components, len(data))
            gmm = GaussianMixture(
                n_components=n_comp,
                covariance_type="diag",
                max_iter=200,
                n_init=3,
                random_state=42,
            )
            gmm.fit(data)
            self.models[label] = gmm

        return self

    def predict(self, X: list[np.ndarray]) -> np.ndarray:
        """Predict class labels for a list of feature sequences."""
        preds = []
        for feat in X:
            vec = self.scaler.transform(self._transform(feat).reshape(1, -1))
            best_label = max(self.models, key=lambda l: self.models[l].score(vec))
            preds.append(best_label)
        return np.array(preds)

    def save(self, path: str) -> None:
        """Save model to a pickle file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "n_components": self.n_components,
                "models": self.models,
                "scaler": self.scaler,
            }, f)
        print(f"Saved GMMClassifier -> {path}")

    def load(self, path: str) -> None:
        """Load model from a pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.n_components = data["n_components"]
        self.models = data["models"]
        self.scaler = data["scaler"]
        print(f"Loaded GMMClassifier <- {path}")
