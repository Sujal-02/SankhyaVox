"""
SankhyaVox – SVM Baseline Classifier.

RBF-kernel SVM on rich statistical MFCC summaries with grid-searched
hyperparameters (C, gamma).  Self-contained — no project imports.
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class SVMClassifier:
    """
    SVM baseline with RBF kernel and optional grid search.

    Parameters
    ----------
    kernel : str
        SVM kernel type.
    C : float, optional
        Regularisation parameter.  If ``None``, tuned via grid search.
    gamma : float or str, optional
        Kernel coefficient.  If ``None``, tuned via grid search.
    checkpoint_path : str or Path, optional
        If given, load a previously saved model.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        C: Optional[float] = None,
        gamma: Optional[float] = None,
        checkpoint_path: Optional[str] = None,
    ):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.scaler = StandardScaler()
        self.model: Optional[SVC] = None

        if checkpoint_path:
            self.load(checkpoint_path)

    @staticmethod
    def _transform(features: np.ndarray) -> np.ndarray:
        """
        Transform variable-length MFCC (n_frames, 39) to a fixed-length vector.

        Computes per-coefficient: mean, std, min, max, median, 10th and
        90th percentiles, inter-quartile range, and mean absolute
        frame-to-frame change.  Plus normalised log frame count as a
        duration proxy.
        Output: 39 * 9 + 1 = 352 dimensions.

        Rationale: SVMs with RBF kernels measure pairwise distances in
        feature space.  Richer statistics (percentile tails, IQR, delta
        magnitude) spread class-discriminative information across more
        dimensions so the RBF kernel can find better separating surfaces.
        The log-frame-count encodes utterance duration, which differs
        substantially across tokens (e.g. "dvi" is short, "vimsati" is
        long) and is a strong discriminative cue.
        """
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        fmin = features.min(axis=0)
        fmax = features.max(axis=0)
        med = np.median(features, axis=0)
        q10 = np.percentile(features, 10, axis=0)
        q90 = np.percentile(features, 90, axis=0)
        iqr = np.percentile(features, 75, axis=0) - np.percentile(features, 25, axis=0)
        if len(features) > 1:
            delta_abs_mean = np.abs(np.diff(features, axis=0)).mean(axis=0)
        else:
            delta_abs_mean = np.zeros(features.shape[1])
        n_frames = np.array([np.log1p(len(features))])
        return np.concatenate([
            mean, std, fmin, fmax, med, q10, q90, iqr, delta_abs_mean, n_frames,
        ])

    def fit(
        self,
        X: list[np.ndarray],
        y: list[int],
        grid_search: bool = True,
        cv: int = 3,
    ) -> "SVMClassifier":
        """
        Fit the SVM with optional grid search.

        Parameters
        ----------
        X : list of ndarray, each (n_frames, feat_dim)
        y : list of int labels
        grid_search : bool
            If True and C/gamma are None, run grid search.
        cv : int
            Cross-validation folds for grid search.
        """
        data = np.array([self._transform(feat) for feat in X])
        labels = np.array(y)
        data = self.scaler.fit_transform(data)

        if grid_search and (self.C is None or self.gamma is None):
            param_grid = {
                "C": [0.01, 0.1, 1, 10, 100, 1000],
                "gamma": [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1],
            }
            gs = GridSearchCV(
                SVC(kernel=self.kernel),
                param_grid,
                cv=cv,
                scoring="accuracy",
                n_jobs=-1,
                verbose=1,
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
        data = np.array([self._transform(feat) for feat in X])
        data = self.scaler.transform(data)
        return self.model.predict(data)

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
