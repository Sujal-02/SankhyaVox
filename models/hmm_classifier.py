from __future__ import annotations

"""
SankhyaVox – HMM (GMM-HMM) Classifier — Pomegranate backend.

Drop-in replacement for the hmmlearn-based SankhyaHMM (see hmm_classifier_old.py).
Per-class left-to-right (Bakis) GMM-HMM fitted on variable-length MFCC
sequences.  Classification by maximum per-frame log-likelihood.

Key differences from the hmmlearn version:
  - PyTorch-backed (pomegranate ≥ 1.0), enabling optional GPU acceleration.
  - Manual Baum-Welch loop freezes Bakis topology (edges & starts restored
    after each M-step) while allowing emission GMM parameters to update.
  - Sequences are grouped by length to avoid padding artefacts.
  - Checkpoints are saved with ``torch.save`` (not pickle); incompatible
    with hmmlearn checkpoints — use hmm_classifier_old.py for those.

Requires: pomegranate >= 1.0.0, torch >= 2.0.0
"""

import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from pomegranate.distributions import Normal
from pomegranate.gmm import GeneralMixtureModel
from pomegranate.hmm import DenseHMM

from src.config import (
    BAUM_WELCH_ITERS,
    FEATURE_DIM,
    GMM_MIXTURES,
    HMM_STATES,
    VOCAB,
    VALUE_TO_TOKEN,
)

_DEFAULT_MIX = 2  # fallback when a token is missing from the mix map
_EM_TOL = 0.1     # log-likelihood improvement threshold for convergence


class SankhyaHMM:
    """
    HMM classifier: one Bakis GMM-HMM per class, classify by max
    per-frame log-likelihood.  Pomegranate (PyTorch) backend.

    API-compatible with the hmmlearn version in ``hmm_classifier_old.py``.

    Parameters
    ----------
    n_iter : int
        Maximum EM (Baum-Welch) iterations per model.
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
        self.models: dict[int, DenseHMM] = {}
        self._label_to_token: dict[int, str] = {}

        if checkpoint_path:
            self.load(checkpoint_path)

    # ── Topology ──────────────────────────────────────────────────────────

    @staticmethod
    def _bakis_edges(n_states: int) -> list[list[float]]:
        """Left-to-right transition matrix: stay or advance, no skip."""
        edges = [[0.0] * n_states for _ in range(n_states)]
        for i in range(n_states - 1):
            edges[i][i] = 0.6
            edges[i][i + 1] = 0.4
        edges[-1][-1] = 1.0
        return edges

    @staticmethod
    def _bakis_starts(n_states: int) -> list[float]:
        """Start probabilities: always begin in state 0."""
        starts = [1e-10] * n_states
        starts[0] = 1.0
        total = sum(starts)
        return [s / total for s in starts]

    # ── Model construction ────────────────────────────────────────────────

    def _build_model(self, n_states: int, n_mix: int) -> DenseHMM:
        """Construct a DenseHMM with Bakis topology and GMM emissions."""
        dists = [
            GeneralMixtureModel(
                [Normal(
                    means=torch.zeros(FEATURE_DIM),
                    covs=torch.ones(FEATURE_DIM),
                    covariance_type="diag",
                    min_cov=1e-2,
                 )
                 for _ in range(n_mix)]
            )
            for _ in range(n_states)
        ]
        return DenseHMM(
            distributions=dists,
            edges=self._bakis_edges(n_states),
            starts=self._bakis_starts(n_states),
            verbose=False,
        )

    # ── Training ──────────────────────────────────────────────────────────

    @staticmethod
    def _prepare_batches(sequences: list[np.ndarray]) -> list[torch.Tensor]:
        """Group variable-length sequences by frame count → list of 3-D tensors."""
        length_groups: dict[int, list[np.ndarray]] = defaultdict(list)
        for seq in sequences:
            length_groups[seq.shape[0]].append(seq)

        batches: list[torch.Tensor] = []
        for _length, seqs in sorted(length_groups.items()):
            batches.append(torch.from_numpy(np.stack(seqs)).float())
        return batches

    def _fit_one(
        self, label: int, sequences: list[np.ndarray]
    ) -> tuple[DenseHMM, bool, int]:
        """Fit a single GMM-HMM for one class label.

        Returns (model, converged, n_iters).
        """
        token = VALUE_TO_TOKEN.get(label, str(label))
        n_states = self.states_map.get(token, 5)
        n_mix = self.mix_map.get(token, _DEFAULT_MIX)

        model = self._build_model(n_states, n_mix)
        batches = self._prepare_batches(sequences)

        # Snapshot Bakis topology (log-space) for restoring after each M-step
        assert model.edges is not None and model.starts is not None
        frozen_edges = model.edges.clone()
        frozen_starts = model.starts.clone()

        # ── Manual Baum-Welch with frozen topology ────────────────────────
        last_logp: Optional[float] = None
        converged = False
        n_iters = 0

        for iteration in range(self.n_iter):
            logp = 0.0
            for batch in batches:
                logp += float(model.summarize(batch).sum())
            model.from_summaries()

            # Restore Bakis edges & starts (freeze topology)
            with torch.no_grad():
                assert model.edges is not None and model.starts is not None
                model.edges.copy_(frozen_edges)
                model.starts.copy_(frozen_starts)

            n_iters = iteration + 1

            if last_logp is not None and abs(logp - last_logp) < _EM_TOL:
                converged = True
                break
            last_logp = logp

        return model, converged, n_iters

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
            model, converged, n_iters = self._fit_one(label, seqs)
            self.models[label] = model
            elapsed = time.perf_counter() - t0
            n_states = self.states_map.get(token, 5)
            n_mix = self.mix_map.get(token, _DEFAULT_MIX)
            status = (
                f"converged ({n_iters} iters)"
                if converged
                else f"no convergence ({n_iters} iters)"
            )
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
            X = torch.from_numpy(mfcc).unsqueeze(0).float()
            ll = self.models[label].log_probability(X).item()
            return ll / mfcc.shape[0]
        except Exception as e:
            print(f"  \u26a0 score failed for label={label}: {e}")
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
        """Save model to file (torch format, incompatible with hmmlearn checkpoints)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data: dict = {
            "backend": "pomegranate",
            "n_iter": self.n_iter,
            "mix_map": self.mix_map,
            "states_map": self.states_map,
            "label_to_token": self._label_to_token,
            "model_configs": {},
        }
        for label, model in self.models.items():
            token = self._label_to_token.get(label, str(label))
            data["model_configs"][label] = {
                "state_dict": model.state_dict(),
                "n_states": self.states_map.get(token, 5),
                "n_mix": self.mix_map.get(token, _DEFAULT_MIX),
            }
        torch.save(data, path)
        print(f"Saved SankhyaHMM (pomegranate) -> {path}")

    def load(self, path: str) -> None:
        """Load model from file (torch format)."""
        data = torch.load(path, map_location="cpu", weights_only=False)
        self.n_iter = data["n_iter"]
        self.mix_map = data.get("mix_map", GMM_MIXTURES)
        if isinstance(self.mix_map, int):
            self.mix_map = {tok: self.mix_map for tok in VOCAB}
        self.states_map = data["states_map"]
        self._label_to_token = data.get("label_to_token", {})
        self.models = {}
        for label, cfg in data["model_configs"].items():
            model = self._build_model(cfg["n_states"], cfg["n_mix"])
            model.load_state_dict(cfg["state_dict"])
            self.models[label] = model
        print(f"Loaded SankhyaHMM (pomegranate) <- {path}")
