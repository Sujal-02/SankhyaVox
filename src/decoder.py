"""
SankhyaVox — Decoder  (src/decoder.py)

THE BUG THAT WAS CAUSING ALL DOUBLE-DIGIT FAILURES:
----------------------------------------------------
Previous versions extracted MFCC + CMVN from the full utterance
("pancha dasha"), then split the normalised matrix at frame t.

  full_audio -> extract_mfcc() -> CMVN(full) -> split at t
  mfcc[0:t] scored against "pancha" HMM  <- WRONG stats

CMVN normalises using mean/std of the ENTIRE utterance.
A sub-segment cut from "pancha dasha" has different CMVN stats
than an isolated "pancha" clip used at training.
HMM has never seen features with these stats -> garbage scores.

THE FIX:
  Boundary search operates on raw AUDIO, not pre-normalised MFCC.
  For each candidate split sample s:
    audio[0:s]   -> extract_mfcc() -> CMVN over those samples only
    audio[s:end] -> extract_mfcc() -> CMVN over those samples only
  Each segment normalised exactly like training. HMMs work correctly.
"""

import numpy as np
import librosa

_SR            = 16_000
_HOP_MS        = 10
_WIN_MS        = 25
_N_MFCC        = 13
_MIN_SEG_MS    = 120    # min segment length ms (shorter = likely noise)
_VAD_TOP_DB    = 18     # lower than training (28) catches shallow natural dips
_VAD_TOP_DB_LO = 12     # second-pass fallback
_VAD_MAX_SEGS  = 3
_ENERGY_SMOOTH = 5      # frames for RMS envelope smoothing


def _extract_mfcc(audio, sr=_SR):
    """
    Extract 39-dim MFCC with per-segment CMVN.
    MUST match training notebook exactly:
      n_mfcc=13, hop=160, n_fft=400, mode='nearest', CMVN per clip.
    Called once per segment so stats match training (isolated word clips).
    """
    if len(audio) < sr * 0.05:
        return None
    hop  = int(sr * _HOP_MS / 1000)
    win  = int(sr * _WIN_MS / 1000)
    pre  = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
    mfcc = librosa.feature.mfcc(y=pre, sr=sr, n_mfcc=_N_MFCC,
                                 hop_length=hop, n_fft=win)
    d1   = librosa.feature.delta(mfcc, mode='nearest')
    d2   = librosa.feature.delta(mfcc, order=2, mode='nearest')
    feat = np.vstack([mfcc, d1, d2]).T
    mean = feat.mean(axis=0, keepdims=True)
    std  = feat.std(axis=0,  keepdims=True) + 1e-8
    return ((feat - mean) / std).astype('float32')


def _score_audio_segment(audio_seg, hmm_system, sr=_SR):
    """
    THE KEY FIX:
    audio_seg -> extract_mfcc (with its OWN CMVN) -> per-frame LL per token.
    CMVN applied per-segment, not per-utterance. Matches training.
    Returns {} if segment too short or all models fail.
    """
    mfcc = _extract_mfcc(audio_seg, sr)
    if mfcc is None or mfcc.shape[0] < 4:
        return {}
    scores = {}
    import warnings
    for tok, model in hmm_system.models.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                scores[tok] = model.score(mfcc) / mfcc.shape[0]
        except Exception:
            scores[tok] = -1e9
    return scores


def _best(scores):
    if not scores:
        return None, -1e9
    tok = max(scores, key=scores.get)
    return tok, scores[tok]


def _vad_split_audio(audio, sr, top_db):
    """Split audio into word-level segments, with smart merging."""
    min_samp  = int(_MIN_SEG_MS / 1000 * sr)
    intervals = librosa.effects.split(
        audio, top_db=top_db,
        frame_length=int(sr * _WIN_MS / 1000),
        hop_length=int(sr * _HOP_MS / 1000),
    )
    valid = [(s, e) for s, e in intervals if (e - s) >= min_samp]
    while len(valid) > _VAD_MAX_SEGS:
        gaps = [valid[i+1][0] - valid[i][1] for i in range(len(valid) - 1)]
        at   = int(np.argmin(gaps))
        valid = valid[:at] + [(valid[at][0], valid[at+1][1])] + valid[at+2:]
    return [audio[s:e].copy() for s, e in valid]


def _candidate_split_samples(audio, sr, n_splits=1):
    """
    Find n_splits candidate split positions (samples) using energy envelope.
    Falls back to evenly-spaced when signal is flat (continuous speech).
    """
    hop     = int(sr * _HOP_MS / 1000)
    rms     = librosa.feature.rms(y=audio, frame_length=hop * 4,
                                  hop_length=hop)[0]
    T       = len(rms)
    margin  = max(4, int(_MIN_SEG_MS / 1000 * sr / hop))
    kernel  = np.ones(_ENERGY_SMOOTH) / _ENERGY_SMOOTH
    rms_sm  = np.convolve(rms, kernel, mode='same')
    search  = rms_sm[margin: T - margin]

    if len(search) < 3:
        return [len(audio) // 2]

    local_min = [i + margin for i in range(1, len(search) - 1)
                 if search[i] <= search[i-1] and search[i] <= search[i+1]]

    if np.var(search) >= 1e-6 and len(local_min) >= n_splits:
        local_min.sort(key=lambda i: rms_sm[i])
        return [f * hop for f in sorted(local_min[:n_splits])]

    # Flat / no clear dip — evenly spaced
    margin_s = margin * hop
    span     = len(audio) - 2 * margin_s
    return [margin_s + span * (i + 1) // (n_splits + 1) for i in range(n_splits)]


def _boundary_search_1(audio, sr, hmm_system, grammar):
    """
    Best single split for 2-token numbers.
    Operates on RAW AUDIO — each half gets its own CMVN. Matches training.
    """
    two_tok = {tuple(seq): num for num, seq in grammar.items() if len(seq) == 2}
    if not two_tok:
        return -1, [], -np.inf

    guided  = _candidate_split_samples(audio, sr, n_splits=1)
    min_s   = int(_MIN_SEG_MS / 1000 * sr)
    start_s = max(min_s, int(len(audio) * 0.15))
    end_s   = min(len(audio) - min_s, int(len(audio) * 0.85))
    step_s  = int(0.020 * sr)   # 20ms grid
    grid    = list(range(start_s, end_s, step_s))
    cands   = sorted(set(guided + grid))

    best_score, best_result = -np.inf, (-1, [], -np.inf)
    for s in cands:
        sc_l = _score_audio_segment(audio[:s],  hmm_system, sr)
        sc_r = _score_audio_segment(audio[s:],  hmm_system, sr)
        if not sc_l or not sc_r:
            continue
        tok_l, ll_l = _best(sc_l)
        tok_r, ll_r = _best(sc_r)
        if (tok_l, tok_r) in two_tok:
            combined = ll_l + ll_r
            if combined > best_score:
                best_score  = combined
                best_result = (two_tok[(tok_l, tok_r)], [tok_l, tok_r], combined)

    return best_result


def _boundary_search_2(audio, sr, hmm_system, grammar):
    """
    Best two split points for 3-token numbers.
    Each of three segments gets its own CMVN. Matches training.
    """
    three_tok = {tuple(seq): num for num, seq in grammar.items() if len(seq) == 3}
    if not three_tok:
        return -1, [], -np.inf

    min_s  = int(_MIN_SEG_MS / 1000 * sr)
    step1  = int(0.030 * sr)
    step2  = int(0.020 * sr)

    best_score, best_result = -np.inf, (-1, [], -np.inf)
    for t1 in range(min_s, max(min_s + 1, int(len(audio) * 0.60)), step1):
        sc1 = _score_audio_segment(audio[:t1], hmm_system, sr)
        if not sc1:
            continue
        tok1, ll1 = _best(sc1)
        for t2 in range(t1 + min_s, len(audio) - min_s, step2):
            sc2 = _score_audio_segment(audio[t1:t2], hmm_system, sr)
            sc3 = _score_audio_segment(audio[t2:],   hmm_system, sr)
            if not sc2 or not sc3:
                continue
            tok2, ll2 = _best(sc2)
            tok3, ll3 = _best(sc3)
            triplet = (tok1, tok2, tok3)
            if triplet in three_tok:
                combined = ll1 + ll2 + ll3
                if combined > best_score:
                    best_score  = combined
                    best_result = (three_tok[triplet], list(triplet), combined)

    return best_result


class SegmentFirstDecoder:
    """
    Cascaded decoder. All paths operate on raw audio so CMVN
    matches training (per-segment, not per-utterance).

    decode(audio, sr)  — PRIMARY entry point used by inference.py
    decode_mfcc(mfcc)  — isolated-token eval only (single token, no split)
    """

    def __init__(self, hmm_system, grammar_fn=None):
        self.hmm = hmm_system
        from src.grammar import all_valid_sequences
        self._g   = all_valid_sequences() if grammar_fn is None else grammar_fn()
        self._s2n = {tuple(seq): num for num, seq in self._g.items()}

    def _tok2num(self, tokens):
        return self._s2n.get(tuple(tokens), -1)

    def _classify_seg(self, audio_seg, sr):
        sc = _score_audio_segment(audio_seg, self.hmm, sr)
        if not sc:
            return None, {}, 0.0
        tok, ll = _best(sc)
        sv  = sorted(sc.values(), reverse=True)
        gap = sv[0] - sv[1] if len(sv) > 1 else 0.0
        return tok, sc, gap

    def _repair(self, tokens):
        for length in range(min(3, len(tokens)), 0, -1):
            for start in range(len(tokens) - length + 1):
                n = self._tok2num(tokens[start:start+length])
                if n >= 0:
                    return n
        return -1

    def decode(self, audio, sr=_SR, verbose=False):
        """
        Decode raw audio -> (number 0-99, total_score).
        Returns (-1, -1e9) on failure.
        """
        if audio is None or len(audio) < int(sr * 0.05):
            return -1, -1e9

        if verbose:
            print(f"  [dec] {len(audio)/sr:.3f}s audio")

        # Path 1: VAD segmentation
        for top_db in [_VAD_TOP_DB, _VAD_TOP_DB_LO]:
            segs = _vad_split_audio(audio, sr, top_db)
            if verbose:
                print(f"  [dec] VAD top_db={top_db}: {len(segs)} seg(s) "
                      f"{[f'{len(s)/sr*1000:.0f}ms' for s in segs]}")
            if 1 <= len(segs) <= _VAD_MAX_SEGS:
                tokens, total_ll, ok = [], 0.0, True
                for i, seg in enumerate(segs):
                    tok, sc, gap = self._classify_seg(seg, sr)
                    if tok is None:
                        ok = False; break
                    tokens.append(tok)
                    total_ll += sc.get(tok, -1e9)
                    if verbose:
                        top3 = sorted(sc.items(), key=lambda x: x[1], reverse=True)[:3]
                        print(f"    seg[{i}] {len(seg)/sr*1000:.0f}ms -> {tok} "
                              f"gap={gap:.2f}  "
                              + "  ".join(f"{t}={v:.2f}" for t, v in top3))
                if ok and tokens:
                    num = self._tok2num(tokens)
                    if num >= 0:
                        if verbose: print(f"  [dec] VAD OK: {num} {tokens}")
                        return num, total_ll
                    num = self._repair(tokens)
                    if num >= 0:
                        if verbose: print(f"  [dec] VAD+repair: {num}")
                        return num, total_ll

        # Path 2: boundary search (works for continuous speech, no pause needed)
        if verbose:
            print(f"  [dec] boundary search...")
        num2, toks2, sc2 = _boundary_search_1(audio, sr, self.hmm, self._g)
        num3, toks3, sc3 = _boundary_search_2(audio, sr, self.hmm, self._g)
        if verbose:
            print(f"  [dec] 2-split: {num2} {toks2} sc={sc2:.3f}")
            print(f"  [dec] 3-split: {num3} {toks3} sc={sc3:.3f}")

        # Path 3: single-token
        tok1, sc1, _ = self._classify_seg(audio, sr)
        num1  = self._tok2num([tok1]) if tok1 else -1
        sco1  = sc1.get(tok1, -1e9) if tok1 else -1e9
        if verbose:
            print(f"  [dec] 1-token: {tok1} ({num1}) sc={sco1:.3f}")

        candidates = []
        if num1 >= 0: candidates.append((num1, sco1 / 1))
        if num2 >= 0: candidates.append((num2, sc2  / 2))
        if num3 >= 0: candidates.append((num3, sc3  / 3))

        if not candidates:
            if verbose: print("  [dec] all paths failed")
            return -1, -1e9

        best_num, _ = max(candidates, key=lambda x: x[1])
        if verbose: print(f"  [dec] result: {best_num}")
        return best_num, _

    def decode_mfcc(self, mfcc):
        """Single-token classification from pre-computed MFCC (eval only)."""
        import warnings
        scores = {}
        for tok, model in self.hmm.models.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    scores[tok] = model.score(mfcc) / max(mfcc.shape[0], 1)
            except Exception:
                scores[tok] = -1e9
        tok, ll = _best(scores)
        return (self._tok2num([tok]) if tok else -1), ll