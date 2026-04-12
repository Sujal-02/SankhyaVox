"""
SankhyaVox – HMM (GMM-HMM) Classifier.

Per-class left-to-right (Bakis) GMM-HMM fitted on variable-length MFCC
sequences.  Classification by maximum per-frame log-likelihood.
"""

import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
from hmmlearn.hmm import GMMHMM

from src.config import (
    BAUM_WELCH_ITERS,
    FEATURE_DIM,
    GMM_MIXTURES,
    HMM_STATES,
    VOCAB,
    VALUE_TO_TOKEN,
)

_DEFAULT_MIX = 2  # fallback when a token is missing from the mix map


class SankhyaHMM:
    """
    HMM baseline: one Bakis GMM-HMM per class, classify by max
    per-frame log-likelihood.

    Parameters
    ----------
    n_iter : int
        EM (Baum-Welch) iterations per model.
    mix_map : dict[str, int], optional
        Token → number of GMM components per state.
        Defaults to ``config.GMM_MIXTURES``.
    states_map : dict[str, int], optional
        Token → number of HMM states.  Defaults to ``config.HMM_STATES``.
    checkpoint_path : str or Path, optional
        If given, load a previously saved model.
    """

    def __init__(
        self,
        n_iter: int = BAUM_WELCH_ITERS,
        mix_map: Optional[dict[str, int]] = None,
        states_map: Optional[dict[str, int]] = None,
        checkpoint_path: Optional[str] = None,
    ):
        self.n_iter = n_iter
        self.mix_map = dict(mix_map or GMM_MIXTURES)
        self.states_map = dict(states_map or HMM_STATES)
        self.models: dict[int, GMMHMM] = {}
        self._label_to_token: dict[int, str] = {}

        if checkpoint_path:
            self.load(checkpoint_path)

    # ── Topology ──────────────────────────────────────────────────────────

    @staticmethod
    def _bakis_transmat(n_states: int) -> np.ndarray:
        A = np.zeros((n_states, n_states))
        for i in range(n_states):
            if i < n_states - 2:
                A[i, i]     = 0.5   # stay
                A[i, i + 1] = 0.35  # advance
                A[i, i + 2] = 0.15  # skip
            elif i < n_states - 1:
                A[i, i]     = 0.6
                A[i, i + 1] = 0.4
            else:
                A[i, i] = 1.0
        return A

    @staticmethod
    def _bakis_startprob(n_states: int) -> np.ndarray:
        pi = np.full(n_states, 1e-10)
        pi[0] = 1.0
        pi /= pi.sum()
        return pi

    # ── Training ──────────────────────────────────────────────────────────

    def _fit_one(self, label: int, sequences: list[np.ndarray]) -> GMMHMM:
        """Fit a single GMM-HMM for one class label."""
        token = VALUE_TO_TOKEN.get(label, str(label))
        n_states = self.states_map.get(token, 5)
        n_mix = self.mix_map.get(token, _DEFAULT_MIX)

        X = np.concatenate(sequences, axis=0)
        lengths = [seq.shape[0] for seq in sequences]

        model = GMMHMM(
            n_components=n_states,
            n_mix=n_mix,
            n_iter=self.n_iter,
            covariance_type="diag",
            verbose=False,
            init_params="mcw",   # skip random init of startprob/transmat
            params="tmcw",        #  was "stmcw": freeze s and t to preserve Bakis
            covars_prior=1e-2,   # variance floor prior
            covars_weight=1.0,   # ADD: prior strength
        )
        model.startprob_ = self._bakis_startprob(n_states)
        model.transmat_ = self._bakis_transmat(n_states)

        model.fit(X, lengths)
        return model

    def fit(self, X: list[np.ndarray], y: list[int]) -> "SankhyaHMM":
        """
        Fit one GMM-HMM per class.

        Parameters
        ----------
        X : list of ndarray, each (n_frames, 39)
        y : list of int labels
        """
        # Group sequences by label
        groups: dict[int, list[np.ndarray]] = {}
        for feat, label in zip(X, y):
            if feat.shape[0] < 3:
                continue
            groups.setdefault(label, []).append(feat)

        for label, seqs in sorted(groups.items()):
            token = VALUE_TO_TOKEN.get(label, str(label))
            self._label_to_token[label] = token
            t0 = time.perf_counter()
            self.models[label] = self._fit_one(label, seqs)
            elapsed = time.perf_counter() - t0
            n_states = self.states_map.get(token, 5)
            n_mix = self.mix_map.get(token, _DEFAULT_MIX)
            model = self.models[label]
            status = f"converged ({model.monitor_.iter} iters)" if model.monitor_.converged else f"no convergence ({model.monitor_.iter} iters)"
            print(
                f"  {token:>8s} (label={label:>3d}): "
                f"{len(seqs):4d} seqs, {n_states} states, "
                f"{n_mix} mix, {elapsed:.1f}s, {status}"
            )

        return self

    # ── Scoring / Prediction ──────────────────────────────────────────────

    def score(self, label: int, mfcc: np.ndarray) -> float:
        """Per-frame log-likelihood of *mfcc* under the model for *label*."""
        if label not in self.models:
            return -1e9
        try:
            return self.models[label].score(mfcc) / mfcc.shape[0]
        except Exception as e:
            # This reveals which models are broken
            print(f"  ⚠ score failed for label={label}: {e}")
            return -1e9

    def predict(self, X: list[np.ndarray]) -> np.ndarray:
        """Predict class labels for a list of MFCC sequences."""
        preds = []
        for feat in X:
            best_label = max(
                self.models, key=lambda l: self.score(l, feat)
            )
            preds.append(best_label)
        return np.array(preds)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save model to a pickle file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "n_iter": self.n_iter,
                    "mix_map": self.mix_map,
                    "states_map": self.states_map,
                    "models": self.models,
                    "label_to_token": self._label_to_token,
                },
                f,
            )
        print(f"Saved SankhyaHMM -> {path}")

    def load(self, path: str) -> None:
        """Load model from a pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.n_iter = data["n_iter"]
        self.mix_map = data.get("mix_map", data.get("n_mix", GMM_MIXTURES))
        if isinstance(self.mix_map, int):
            self.mix_map = {tok: self.mix_map for tok in VOCAB}
        self.states_map = data["states_map"]
        self.models = data["models"]
        self._label_to_token = data.get("label_to_token", {})
        print(f"Loaded SankhyaHMM <- {path}")
