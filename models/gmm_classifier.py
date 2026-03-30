"""
SankhyaVox – GMM Baseline Classifier.

Per-class Gaussian Mixture Model fitted on summarised MFCC features
(mean + std per coefficient → 78-dim fixed-length vector).
Classification by maximum log-likelihood.
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.mixture import GaussianMixture


class GMMClassifier:
    """
    GMM baseline: one GMM per class, classify by max log-likelihood.

    Parameters
    ----------
    n_components : int
        Number of Gaussian components per class GMM.
    n_classes : int
        Number of target classes.
    checkpoint_path : str or Path, optional
        If given, load a previously saved model.
    """

    def __init__(
        self,
        n_components: int = 3,
        n_classes: int = 13,
        checkpoint_path: Optional[str] = None,
    ):
        self.n_components = n_components
        self.n_classes = n_classes
        self.models: dict[int, GaussianMixture] = {}

        if checkpoint_path:
            self.load(checkpoint_path)

    @staticmethod
    def summarise(features: np.ndarray) -> np.ndarray:
        """Summarise variable-length MFCC to fixed-length vector (mean + std)."""
        return np.concatenate([features.mean(axis=0), features.std(axis=0)])

    def fit(self, X: list[np.ndarray], y: list[int]) -> "GMMClassifier":
        """
        Fit one GMM per class.

        Parameters
        ----------
        X : list of ndarray, each (n_frames, feat_dim)
        y : list of int labels
        """
        summaries: dict[int, list[np.ndarray]] = {}
        for feat, label in zip(X, y):
            summaries.setdefault(label, []).append(self.summarise(feat))

        for label, vecs in summaries.items():
            data = np.array(vecs)
            n_comp = min(self.n_components, len(data))
            gmm = GaussianMixture(n_components=n_comp, covariance_type="diag", random_state=42)
            gmm.fit(data)
            self.models[label] = gmm

        return self

    def predict(self, X: list[np.ndarray]) -> np.ndarray:
        """Predict class labels for a list of feature sequences."""
        preds = []
        for feat in X:
            vec = self.summarise(feat).reshape(1, -1)
            best_label = max(self.models, key=lambda l: self.models[l].score(vec))
            preds.append(best_label)
        return np.array(preds)

    def score(self, X: list[np.ndarray], y: list[int]) -> float:
        """Return accuracy on the given data."""
        preds = self.predict(X)
        return float(np.mean(preds == np.array(y)))

    def save(self, path: str) -> None:
        """Save model to a pickle file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"n_components": self.n_components, "models": self.models}, f)
        print(f"Saved GMMClassifier -> {path}")

    def load(self, path: str) -> None:
        """Load model from a pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.n_components = data["n_components"]
        self.models = data["models"]
        print(f"Loaded GMMClassifier <- {path}")
