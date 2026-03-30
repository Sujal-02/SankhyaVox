"""
SankhyaVox – SVM Baseline Classifier.

RBF-kernel SVM on fixed-length summarised MFCC features
(mean + std per coefficient → 78-dim vector).
Hyperparameters (C, gamma) tuned via grid search.
"""

import pickle
from pathlib import Path
from typing import Literal, List, Optional

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import SVM_C_RANGE, SVM_GAMMA_RANGE, SVM_KERNEL


class SVMClassifier:
    """
    SVM baseline with RBF kernel and optional grid search.

    Parameters
    ----------
    kernel : str
        SVM kernel type.
    C : float, optional
        Regularisation parameter.  If ``None``, will be tuned via grid search.
    gamma : float or str, optional
        Kernel coefficient.  If ``None``, will be tuned via grid search.
    checkpoint_path : str or Path, optional
        If given, load a previously saved model.
    """

    KernelType = Literal["linear", "poly", "rbf", "sigmoid", "precomputed"]

    def __init__(
        self,
        kernel: KernelType = "rbf",
        C: Optional[float] = None,
        gamma: Optional[float] = None,
        checkpoint_path: Optional[str] = None,
    ):
        self.kernel: SVMClassifier.KernelType = kernel
        self.C = C
        self.gamma = gamma
        self.scaler = StandardScaler()
        self.model: Optional[SVC] = None

        if checkpoint_path:
            self.load(checkpoint_path)

    @staticmethod
    def summarise(features: np.ndarray) -> np.ndarray:
        """Summarise variable-length MFCC to fixed-length vector (mean + std)."""
        return np.concatenate([features.mean(axis=0), features.std(axis=0)])

    def fit(
        self,
        X: list[np.ndarray],
        y: list[int],
        grid_search: bool = True,
        cv: int = 3,
    ) -> "SVMClassifier":
        """
        Fit the SVM.

        Parameters
        ----------
        X : list of ndarray, each (n_frames, feat_dim)
        y : list of int labels
        grid_search : bool
            If True and C/gamma are None, run grid search over
            ``config.SVM_C_RANGE`` and ``config.SVM_GAMMA_RANGE``.
        cv : int
            Cross-validation folds for grid search.
        """
        data = np.array([self.summarise(feat) for feat in X])
        labels = np.array(y)

        data = self.scaler.fit_transform(data)

        if grid_search and (self.C is None or self.gamma is None):
            param_grid = {"C": SVM_C_RANGE, "gamma": SVM_GAMMA_RANGE}
            gs = GridSearchCV(
                SVC(kernel=self.kernel),
                param_grid,
                cv=cv,
                scoring="accuracy",
                n_jobs=-1,
            )
            gs.fit(data, labels)
            self.model = gs.best_estimator_
            self.C = gs.best_params_["C"]
            self.gamma = gs.best_params_["gamma"]
            print(f"Grid search best: C={self.C}, gamma={self.gamma}, "
                  f"acc={gs.best_score_:.3f}")
        else:
            self.model = SVC(
                kernel=self.kernel,
                C=self.C or 1.0,
                gamma=self.gamma or "scale",
            )
            self.model.fit(data, labels)

        return self

    def predict(self, X: list[np.ndarray]) -> np.ndarray:
        """Predict class labels for a list of feature sequences."""
        assert self.model is not None, "Model not fitted yet. Call fit() first."
        data = np.array([self.summarise(feat) for feat in X])
        data = self.scaler.transform(data)
        return self.model.predict(data)

    def score(self, X: list[np.ndarray], y: list[int]) -> float:
        """Return accuracy on the given data."""
        preds = self.predict(X)
        return float(np.mean(preds == np.array(y)))

    def save(self, path: str) -> None:
        """Save model + scaler to a pickle file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "kernel": self.kernel, "C": self.C, "gamma": self.gamma,
                "scaler": self.scaler, "model": self.model,
            }, f)
        print(f"Saved SVMClassifier -> {path}")

    def load(self, path: str) -> None:
        """Load model + scaler from a pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.kernel = data["kernel"]
        self.C = data["C"]
        self.gamma = data["gamma"]
        self.scaler = data["scaler"]
        self.model = data["model"]
        print(f"Loaded SVMClassifier <- {path}")
