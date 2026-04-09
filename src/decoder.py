"""
SankhyaVox – Grammar-Constrained Viterbi Decoder (0–99).

Decodes a compound Sanskrit number utterance into an integer using:
  1. Sliding-window HMM scoring against all 13 token models
  2. Energy-based silence gating (skip silent windows before Viterbi)
  3. Grammar-constrained Viterbi search over non-silent windows

Requires a trained ``SankhyaHMM`` checkpoint (one GMM-HMM per token).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import TOKEN_TO_VALUE, VALUE_TO_TOKEN, VOCAB
from src.grammar import all_valid_sequences, tokens_to_number


def _build_successor_map() -> Dict[str, set]:
    """Token → set of tokens that can legally follow it."""
    sequences = all_valid_sequences()
    succ: Dict[str, set] = {tok: set() for tok in VOCAB}
    succ["SIL"] = set(VOCAB)
    for seq in sequences.values():
        for i in range(len(seq) - 1):
            succ[seq[i]].add(seq[i + 1])
        succ[seq[-1]].add("SIL")
    return succ


class GrammarConstrainedDecoder:
    """
    Decode continuous Sanskrit digit speech → integer (0–99).

    Pipeline
    --------
    1. Slide a fixed-width window over the MFCC matrix.
    2. Score each window against all 13 token HMMs (per-frame LL).
    3. Gate out silence windows by RMS energy threshold.
    4. Run grammar-constrained Viterbi over active windows.
    5. Collapse consecutive duplicate tokens → grammar lookup → integer.

    Parameters
    ----------
    hmm : SankhyaHMM
        Trained HMM bank (one model per token label).
    window_frames : int
        Sliding window width in MFCC frames (default 25 = 250 ms @ 100 fps).
    hop_frames : int
        Sliding window hop in MFCC frames (default 5 = 50 ms).
    silence_db : float
        Windows with RMS energy below this dBFS threshold are skipped.
    """

    def __init__(
        self,
        hmm,
        window_frames: int = 25,
        hop_frames: int = 5,
        silence_db: float = -35.0,
    ):
        self.hmm = hmm
        self.window_frames = window_frames
        self.hop_frames = hop_frames
        self.silence_db = silence_db

        self.vocab = VOCAB
        self.n = len(VOCAB)
        self.tok2idx = {t: i for i, t in enumerate(VOCAB)}
        self.idx2tok = {i: t for i, t in enumerate(VOCAB)}

        # Label (int) each vocab token maps to in the HMM bank
        self._tok_labels = [TOKEN_TO_VALUE[t] for t in VOCAB]

        # Grammar successor map
        successors = _build_successor_map()

        # Boolean transition matrix [n × n]: allowed[i, j] = True
        # if vocab[j] can legally follow vocab[i]
        self.allowed = np.zeros((self.n, self.n), dtype=bool)
        for tok, succs in successors.items():
            i = self.tok2idx.get(tok)
            if i is None:
                continue
            for s in succs:
                j = self.tok2idx.get(s)
                if j is not None:
                    self.allowed[i, j] = True
        # Self-loops: same token can span consecutive windows
        np.fill_diagonal(self.allowed, True)

    # ── 1. Sliding-window score matrix ────────────────────────────────────

    def _compute_scores(
        self, mfcc: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        scores : ndarray of shape (N_windows, n_vocab)
        is_sil : bool ndarray of shape (N_windows,)
        """
        T = mfcc.shape[0]
        W = self.window_frames
        H = self.hop_frames

        positions = list(range(0, max(1, T - W + 1), H))
        N = len(positions)
        scores = np.full((N, self.n), -1e9, dtype=np.float64)
        is_sil = np.zeros(N, dtype=bool)

        for wi, start in enumerate(positions):
            window = mfcc[start: start + W]
            if window.shape[0] < W:
                window = np.pad(
                    window, ((0, W - window.shape[0]), (0, 0)), mode="edge"
                )

            rms_db = 20.0 * np.log10(
                np.sqrt(np.mean(window ** 2)) + 1e-10
            )
            if rms_db < self.silence_db:
                is_sil[wi] = True
                continue

            for ti, label in enumerate(self._tok_labels):
                scores[wi, ti] = self.hmm.score(label, window)

        return scores, is_sil

    # ── 2. Grammar-constrained Viterbi ────────────────────────────────────

    def _viterbi(
        self, scores: np.ndarray, is_sil: np.ndarray
    ) -> Tuple[List[str], float]:
        """
        Returns
        -------
        collapsed : list of token strings (consecutive dupes merged)
        path_score : float
        """
        active_idx = np.where(~is_sil)[0]
        if len(active_idx) == 0:
            return [], 0.0

        active_scores = scores[active_idx]
        M = len(active_idx)

        vit = np.full((M, self.n), -np.inf)
        back = np.full((M, self.n), -1, dtype=np.int32)

        # Any token can start
        vit[0] = active_scores[0]

        for t in range(1, M):
            for s in range(self.n):
                emit = active_scores[t, s]
                if emit < -1e8:
                    continue
                prev = vit[t - 1].copy()
                prev[~self.allowed[:, s]] = -np.inf
                best_prev = int(np.argmax(prev))
                if prev[best_prev] > -np.inf:
                    vit[t, s] = prev[best_prev] + emit
                    back[t, s] = best_prev

        end_s = int(np.argmax(vit[M - 1]))
        path_score = float(vit[M - 1, end_s])
        if path_score == -np.inf:
            return [], -np.inf

        # Backtrack
        path = [end_s]
        s = end_s
        for t in range(M - 1, 0, -1):
            s = int(back[t, s])
            if s < 0:
                break
            path.append(s)
        path.reverse()

        # Collapse consecutive identical tokens
        collapsed: List[str] = []
        for s in path:
            tok = self.idx2tok[s]
            if not collapsed or collapsed[-1] != tok:
                collapsed.append(tok)

        return collapsed, path_score

    # ── 3. Grammar lookup ─────────────────────────────────────────────────

    @staticmethod
    def _sequence_to_int(tokens: List[str]) -> int:
        """Exact grammar match first, then try contiguous sub-sequences."""
        r = tokens_to_number(tokens)
        if r is not None:
            return r
        # Fallback: try sub-sequences of length 1..3
        for length in range(1, min(4, len(tokens) + 1)):
            for start in range(len(tokens) - length + 1):
                r = tokens_to_number(tokens[start: start + length])
                if r is not None:
                    return r
        return -1

    # ── 4. Public API ─────────────────────────────────────────────────────

    def decode(
        self, mfcc: np.ndarray, verbose: bool = False
    ) -> Tuple[int, List[str], dict]:
        """
        Decode an MFCC matrix into a recognised integer.

        Parameters
        ----------
        mfcc : ndarray of shape (T, 39)
        verbose : bool
            Print debug information.

        Returns
        -------
        integer : int
            Recognised number (0–99), or -1 on failure.
        tokens : list of str
            Decoded token sequence.
        debug : dict
            Diagnostic info (score matrix shape, active windows, etc.).
        """
        if verbose:
            print(f"  Input MFCC : {mfcc.shape}  ({mfcc.shape[0] / 100:.2f}s)")

        scores, is_sil = self._compute_scores(mfcc)
        n_active = int((~is_sil).sum())

        if verbose:
            print(f"  Score matrix: {scores.shape}")
            print(f"  Active windows: {n_active} / {len(is_sil)}")

        token_seq, score = self._viterbi(scores, is_sil)

        if verbose:
            print(f"  Viterbi path : {token_seq}  (score={score:.3f})")

        integer_result = self._sequence_to_int(token_seq)

        if verbose:
            print(f"  Grammar      : {token_seq} → {integer_result}")

        debug = {
            "score_matrix_shape": list(scores.shape),
            "active_windows": n_active,
            "raw_token_sequence": token_seq,
            "viterbi_score": round(score, 4),
            "recognized_integer": integer_result,
        }
        return integer_result, token_seq, debug
