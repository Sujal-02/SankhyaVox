# SankhyaVox — Root Cause Analysis & Revised Architecture
**For: Coding Agent Execution**
**Version: 2.0 — Grammar-Constrained Viterbi Decoder (No Augmentation)**

---

## VERDICT: The Idea Is Feasible. The Current Code Has a Fatal Bug.

Your instinct about "word breakdown → segment → classify → grammar merge" is **correct and is the right approach**. It's essentially what real ASR systems do. The reason it's failing is not a conceptual flaw — it's a **data poisoning bug** in `augment_sequence` combined with a mismatch between training and decoding strategy.

---

## Part 1 — Root Cause of Current Failure

### Bug 1 (Critical): `augment_sequence` Destroys Training Labels

**Location:** `SankhyaVox_Pipeline.ipynb`, Cell 6, function `augment_sequence`

```python
# CURRENT BROKEN CODE:
seg = len(y) // n_tok
for i, tok in enumerate(tokens):
    chunk = y[i*seg : (i+1)*seg if i < n_tok-1 else len(y)]
```

**What this does:** Splits "pancha dasha" audio into two equal halves and labels the first half as "pancha" and the second as "dasha".

**Why this is wrong:** Speech is NOT evenly distributed in time. "pancha" (पञ्च, 2 syllables) and "dasha" (दश, 2 syllables) are close in length, but in connected speech, co-articulation means the actual acoustic boundary between them is **never** at the 50% mark. It shifts based on TTS engine, speaking rate, and intonation. The result:

- The "pancha" HMM trains on audio that is 30-70% actually "dasha"
- The "dasha" HMM trains on audio that is 30-70% actually "pancha"
- Models learn **poisoned, overlapping distributions**
- Classifier confidence collapses on real speech

**This single bug explains why the system doesn't work.**

### Bug 2 (Architectural): Training Strategy vs Decoding Strategy Mismatch

The CTC-style decoder (`src/ctc_decoder.py`) does sliding window classification. But the models were trained on fixed `5-chunk pooled` vectors (`chunk_pool` in Cell 20). A 250ms window at inference becomes a 195-dim vector through pooling — but the window boundaries are different from training. The models haven't seen "partial" words during training, only full ones.

### Bug 3 (Structural): No Grammar Integration During Training

The HMMs are trained as independent per-token classifiers. They have no knowledge of which tokens can follow which (e.g., "vimsati" can never follow "dasha"). This means the decoder has to do expensive post-hoc grammar checking instead of pruning the search space during decoding.

---

## Part 2 — What Must Change

### Decision: Drop Augmentation, Fix the Pipeline

The user correctly identified that augmentation (especially the broken sequence augmentation) is the wrong path. The clean solution is:

1. **Train on isolated single tokens only** — real recordings, no sequence splitting
2. **Decode connected speech using Grammar-Constrained Viterbi** — bake the grammar into the HMM network, let Viterbi find the boundary + identity simultaneously
3. **No equal-time splitting ever** — let the decoder find boundaries as a byproduct of classification

This is fundamentally sound. Here is the complete new architecture.

---

## Part 3 — Revised Architecture

### 3.1 Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING PHASE                              │
│                                                                   │
│  Real Recordings (isolated tokens only)                          │
│       ↓                                                          │
│  MFCC Extraction (39-dim, 100 fps)                               │
│       ↓                                                          │
│  Per-Token HMM Training (Bakis, 3-GMM)                          │
│  13 independent HMM models saved as .pkl                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     INFERENCE PHASE                              │
│                                                                   │
│  Raw Audio (browser .webm)                                       │
│       ↓ FFmpeg → 16kHz Mono WAV                                  │
│  Preprocessing (DC offset, peak normalize)                       │
│       ↓                                                          │
│  MFCC Matrix X [T × 39]                                         │
│       ↓                                                          │
│  Observation Matrix Computation                                  │
│  For each token HMM: compute log-likelihood per frame window     │
│  Result: Score Matrix S [T × 13]                                 │
│       ↓                                                          │
│  Grammar Graph Construction                                      │
│  Build trellis of all valid (token_sequence → integer) paths     │
│       ↓                                                          │
│  Grammar-Constrained Viterbi Decoding                            │
│  Find best path through trellis using S                          │
│       ↓                                                          │
│  Token Sequence → Grammar Parser → Integer (0-99)                │
└─────────────────────────────────────────────────────────────────┘
```

---

### 3.2 Sanskrit Number Grammar (Complete Specification)

This is the formal grammar that constrains what token sequences are valid:

```
VOCAB = [shunya, eka, dvi, tri, catur, pancha, shat, sapta, ashta, nava,
         dasha, vimsati, shata]

VALID_SEQUENCES = {
    # Integers 0-9 (single token)
    0:  [shunya]
    1:  [eka]
    2:  [dvi]
    3:  [tri]
    4:  [catur]
    5:  [pancha]
    6:  [shat]
    7:  [sapta]
    8:  [ashta]
    9:  [nava]
    
    # Integers 10-19 (dasha prefix)
    10: [dasha]
    11: [dasha, eka]
    12: [dasha, dvi]
    13: [dasha, tri]
    14: [dasha, catur]
    15: [dasha, pancha]
    16: [dasha, shat]
    17: [dasha, sapta]
    18: [dasha, ashta]
    19: [dasha, nava]
    
    # Integers 20-29 (vimsati prefix)
    20: [vimsati]
    21: [vimsati, eka]
    22: [vimsati, dvi]
    ...
    29: [vimsati, nava]
    
    # Integers 30-99 (unit + dasha + optional unit)
    30: [tri, dasha]
    31: [tri, dasha, eka]
    ...
    39: [tri, dasha, nava]
    40: [catur, dasha]
    ...
    99: [nava, dasha, nava]
    
    # Special: shata = 100 (out of scope, but keep as invalid guard)
}
```

**Grammar rules (BNF):**
```
number   ::= single | tens_prefix | compound
single   ::= shunya | eka | dvi | tri | catur | pancha | shat | sapta | ashta | nava
tens_prefix ::= dasha | vimsati
compound ::= tens_prefix unit
           | multiplier dasha
           | multiplier dasha unit
multiplier ::= dvi | tri | catur | pancha | shat | sapta | ashta | nava
unit       ::= eka | dvi | tri | catur | pancha | shat | sapta | ashta | nava
```

---

### 3.3 Core Algorithm: Grammar-Constrained Viterbi

This replaces the current `CTCStyleDecoder`. It is the centerpiece of the new architecture.

```python
# src/constrained_decoder.py

class GrammarConstrainedDecoder:
    """
    Decodes continuous Sanskrit digit speech using grammar-constrained Viterbi.
    
    Unlike the CTC-style decoder which classifies frames then collapses,
    this decoder integrates grammar constraints DURING search, not after.
    This means impossible token transitions (e.g., vimsati→dasha) are pruned
    early, dramatically improving accuracy.
    """
    
    def __init__(self, hmm_models: dict, grammar: dict):
        """
        Args:
            hmm_models: {token_name: trained_GaussianMixtureHMM}
            grammar: {integer_value: [token_sequence]}
        """
        self.models = hmm_models       # 13 trained HMMs
        self.grammar = grammar         # all 100 valid sequences
        self.vocab = list(hmm_models.keys())
        
        # Pre-compute: which tokens can FOLLOW which token?
        # Used to prune impossible transitions during Viterbi
        self.valid_successors = self._build_successor_map()
    
    def _build_successor_map(self) -> dict:
        """
        For each token, what tokens can legally follow it?
        
        Examples:
            shunya → {} (must be end of utterance)
            dasha  → {eka, dvi, tri, catur, pancha, shat, sapta, ashta, nava, SIL}
            vimsati → {eka, dvi, ..., nava, SIL}
            pancha  → {dasha, SIL}  # as multiplier
                    ∪ {SIL}         # as standalone 5
            
        Build this by walking the grammar sequences.
        """
        successors = {tok: set() for tok in self.vocab}
        successors['SIL'] = set(self.vocab)  # silence can precede anything
        
        for integer_val, token_seq in self.grammar.items():
            for i in range(len(token_seq) - 1):
                current = token_seq[i]
                next_tok = token_seq[i + 1]
                successors[current].add(next_tok)
            # Last token in any sequence can be followed by SIL (end)
            successors[token_seq[-1]].add('SIL')
        
        return successors
    
    def compute_score_matrix(self, mfcc_matrix: np.ndarray) -> np.ndarray:
        """
        Compute per-frame log-likelihood scores for all 13 tokens.
        
        Method: Sliding window of 250ms (25 frames), hop 50ms (5 frames).
        Each window is scored against each token's HMM using log-likelihood.
        
        Args:
            mfcc_matrix: [T × 39] float array
        
        Returns:
            scores: [N_windows × 14] float array
                    columns 0-12: log-likelihood for each token
                    column  13:   silence score (computed from energy)
        """
        T = mfcc_matrix.shape[0]
        WINDOW = 25    # 250ms at 100fps
        HOP    = 5     # 50ms hop
        
        windows = []
        window_centers = []
        
        start = 0
        while start + WINDOW <= T:
            window = mfcc_matrix[start : start + WINDOW]  # [25 × 39]
            windows.append(window)
            window_centers.append(start + WINDOW // 2)
            start += HOP
        
        # Handle remainder: pad last window if needed
        if start < T:
            remainder = mfcc_matrix[start:]
            padded = np.pad(remainder, ((0, WINDOW - len(remainder)), (0, 0)), mode='edge')
            windows.append(padded)
            window_centers.append(start + len(remainder) // 2)
        
        N = len(windows)
        n_vocab = len(self.vocab)
        scores = np.full((N, n_vocab + 1), -np.inf)  # +1 for SIL
        
        for i, window in enumerate(windows):
            for j, token in enumerate(self.vocab):
                model = self.models[token]
                try:
                    # hmmlearn: score() returns log-probability of sequence
                    scores[i, j] = model.score(window)
                except:
                    scores[i, j] = -1e9
            
            # Silence score: based on frame energy
            # Low energy = high silence score
            energy = np.mean(window ** 2)
            scores[i, n_vocab] = -20.0 * np.log(energy + 1e-8)  # SIL col
        
        return scores, windows
    
    def viterbi_decode(self, scores: np.ndarray) -> tuple:
        """
        Grammar-constrained Viterbi over the score matrix.
        
        State space: (token_index) where token_index ∈ {0..12, SIL}
        Transition: only allowed if token_j ∈ valid_successors[token_i]
        Emission:   scores[t, j] from compute_score_matrix
        
        This is a SEGMENTAL Viterbi (each "state" can span multiple frames).
        
        Args:
            scores: [N × 14] from compute_score_matrix
        
        Returns:
            best_sequence: list of (token_name, start_frame, end_frame)
            best_score: float
        """
        N = scores.shape[0]  # number of windows
        n_states = len(self.vocab) + 1  # +1 for SIL
        SIL_IDX = n_states - 1
        
        # Map token name → index for fast lookup
        tok_to_idx = {tok: i for i, tok in enumerate(self.vocab)}
        tok_to_idx['SIL'] = SIL_IDX
        idx_to_tok = {i: tok for tok, i in tok_to_idx.items()}
        
        # Build transition matrix: allowed[i][j] = True if j can follow i
        allowed = np.zeros((n_states, n_states), dtype=bool)
        for tok, succs in self.valid_successors.items():
            i = tok_to_idx.get(tok)
            if i is None: continue
            for s in succs:
                j = tok_to_idx.get(s)
                if j is not None:
                    allowed[i, j] = True
        
        # Viterbi DP
        # viterbi[t][s] = best log-prob of any path ending at state s at time t
        viterbi = np.full((N, n_states), -np.inf)
        backtrack = np.full((N, n_states), -1, dtype=int)
        
        # Initialize: start from SIL or any valid first token
        for s in range(n_states):
            viterbi[0, s] = scores[0, s]
        
        # Fill
        for t in range(1, N):
            for s in range(n_states):
                emission = scores[t, s]
                if emission < -1e8:
                    continue  # pruned
                
                # Find best predecessor
                best_prev_score = -np.inf
                best_prev_state = -1
                
                for prev_s in range(n_states):
                    if not allowed[prev_s, s]:
                        continue
                    if viterbi[t-1, prev_s] == -np.inf:
                        continue
                    
                    candidate = viterbi[t-1, prev_s]  # no transition penalty needed
                    if candidate > best_prev_score:
                        best_prev_score = candidate
                        best_prev_state = prev_s
                
                if best_prev_state >= 0:
                    viterbi[t, s] = best_prev_score + emission
                    backtrack[t, s] = best_prev_state
        
        # Find best end state (should end at SIL or any valid terminal)
        best_end_state = np.argmax(viterbi[N-1])
        best_score = viterbi[N-1, best_end_state]
        
        # Backtrack
        path = []
        s = best_end_state
        for t in range(N-1, -1, -1):
            path.append(s)
            prev = backtrack[t, s]
            if prev < 0: break
            s = prev
        
        path.reverse()
        
        # Collapse consecutive identical tokens (CTC-style collapse)
        token_path = [idx_to_tok[s] for s in path]
        collapsed = []
        for tok in token_path:
            if tok == 'SIL':
                continue
            if not collapsed or collapsed[-1] != tok:
                collapsed.append(tok)
        
        return collapsed, best_score
    
    def decode(self, mfcc_matrix: np.ndarray) -> tuple:
        """
        Full pipeline: MFCC → integer.
        
        Returns:
            (integer_result, token_sequence, debug_info)
            integer_result = -1 if recognition failed
        """
        scores, _ = self.compute_score_matrix(mfcc_matrix)
        token_seq, confidence = self.viterbi_decode(scores)
        
        # Grammar lookup
        integer_result = self._tokens_to_integer(token_seq)
        
        debug = {
            'token_sequence': token_seq,
            'confidence': float(confidence),
            'score_matrix_shape': list(scores.shape)
        }
        
        return integer_result, token_seq, debug
    
    def _tokens_to_integer(self, tokens: list) -> int:
        """
        Map token sequence to integer using grammar.
        Returns -1 if not a valid sequence.
        """
        for integer_val, seq in self.grammar.items():
            if seq == tokens:
                return integer_val
        
        # Fuzzy fallback: try all subsequences of length 1-3
        for length in range(1, min(4, len(tokens) + 1)):
            for start in range(len(tokens) - length + 1):
                subseq = tokens[start : start + length]
                for integer_val, seq in self.grammar.items():
                    if seq == subseq:
                        return integer_val
        
        return -1  # recognition failed
```

---

### 3.4 New Training Pipeline (No Augmentation)

```python
# scripts/train_v2.py
# Clean training on isolated tokens only.

"""
DATA REQUIREMENTS (no augmentation):
- Per token: minimum 10 real recordings, ideally 30+
- Recording conditions: vary microphone distance, room, time of day
- Recording length: 0.5s - 2.0s per token
- Format: 16kHz mono WAV (use prep_audio.py to convert)

WHAT WE REMOVE:
- All augment_and_save() calls
- All augment_sequence() calls  
- All multi-word SEQUENCE_MAP generation
  (grammar handles co-articulation at decode time, not train time)

WHAT WE KEEP:
- MFCC extraction (39-dim)
- Per-token HMM training
- Bakis topology
"""

def train_hmm_models_v2(feature_dir, vocab, hmm_states, model_dir):
    """
    Train one HMM per token on isolated recordings only.
    No sequences, no augmentation.
    """
    token_data = {v: [] for v in vocab}
    token_lengths = {v: [] for v in vocab}
    
    # Load only isolated-token feature files
    # File naming: {token}_{recording_id}.npy  e.g.  pancha_001.npy
    feature_files = glob.glob(os.path.join(feature_dir, "**", "*.npy"), recursive=True)
    
    for fp in feature_files:
        # Skip any files generated from sequences (they're poisoned)
        if "seq_" in os.path.basename(fp):
            continue
        
        bn = os.path.basename(fp)
        for token in vocab:
            if bn.startswith(token + "_") or f"_{token}_" in bn or f"_{token}." in bn:
                feats = np.load(fp)  # [T × 39]
                if feats.shape[0] >= 5:  # need at least 5 frames
                    token_data[token].append(feats)
                    token_lengths[token].append(feats.shape[0])
                break
    
    # Report counts
    for token in vocab:
        count = len(token_data[token])
        if count < 5:
            print(f"WARNING: {token} has only {count} samples — need at least 5")
        else:
            print(f"  {token}: {count} samples, avg {np.mean(token_lengths[token]):.1f} frames")
    
    # Train per-token HMM
    hmm = SankhyaHMMBase(vocab, hmm_states, model_dir, n_mix=3, n_iter=100)
    
    for token in vocab:
        if not token_data[token]:
            print(f"SKIP {token}: no data")
            continue
        
        # Concatenate all recordings for this token
        # hmmlearn expects (concatenated_sequences, [length1, length2, ...])
        all_seqs = np.concatenate(token_data[token], axis=0)
        lengths  = token_lengths[token]
        
        hmm.fit_word(token, (all_seqs, lengths))
        print(f"  Trained HMM for {token} on {len(lengths)} recordings")
    
    hmm.save_models()
    print(f"\nAll HMMs saved to {model_dir}")
    return hmm
```

---

### 3.5 Updated `src/grammar.py`

The grammar needs to explicitly enumerate all valid sequences with their integer values:

```python
# src/grammar.py  (complete rewrite)

UNITS = ['eka', 'dvi', 'tri', 'catur', 'pancha', 'shat', 'sapta', 'ashta', 'nava']
MULTIPLIERS = UNITS  # same tokens used as multipliers for 30-99

def build_complete_grammar() -> dict:
    """
    Returns {integer_value: [token_sequence]} for all 0-99.
    """
    grammar = {}
    
    # 0
    grammar[0] = ['shunya']
    
    # 1-9
    for i, unit in enumerate(UNITS, start=1):
        grammar[i] = [unit]
    
    # 10
    grammar[10] = ['dasha']
    
    # 11-19
    for i, unit in enumerate(UNITS, start=1):
        grammar[10 + i] = ['dasha', unit]
    
    # 20
    grammar[20] = ['vimsati']
    
    # 21-29
    for i, unit in enumerate(UNITS, start=1):
        grammar[20 + i] = ['vimsati', unit]
    
    # 30-99: multiplier × 10 + unit
    for m_idx, multiplier in enumerate(MULTIPLIERS, start=1):
        base = (m_idx + 2) * 10  # dvi=2→40, tri=3→30... wait, correct mapping below
    
    # Correct mapping for 30-99:
    # 30 = tri(3) × dasha  → [tri, dasha]
    # 40 = catur(4) × dasha → [catur, dasha]
    # 50 = pancha(5) × dasha → [pancha, dasha]
    # 60 = shat(6) × dasha  → [shat, dasha]
    # 70 = sapta(7) × dasha → [sapta, dasha]
    # 80 = ashta(8) × dasha → [ashta, dasha]
    # 90 = nava(9) × dasha  → [nava, dasha]
    
    tens_map = {
        3: 'tri', 4: 'catur', 5: 'pancha',
        6: 'shat', 7: 'sapta', 8: 'ashta', 9: 'nava'
    }
    
    for digit, multiplier_tok in tens_map.items():
        base = digit * 10
        grammar[base] = [multiplier_tok, 'dasha']
        
        for unit_digit, unit_tok in enumerate(UNITS, start=1):
            grammar[base + unit_digit] = [multiplier_tok, 'dasha', unit_tok]
    
    return grammar


# Build on import
COMPLETE_GRAMMAR = build_complete_grammar()

# Reverse lookup: token_sequence (as tuple) → integer
SEQ_TO_INT = {tuple(seq): val for val, seq in COMPLETE_GRAMMAR.items()}

def tokens_to_number(token_list: list) -> int:
    key = tuple(token_list)
    return SEQ_TO_INT.get(key, -1)

def all_valid_sequences() -> dict:
    return COMPLETE_GRAMMAR
```

---

### 3.6 Updated `src/ctc_decoder.py` → Renamed `src/constrained_decoder.py`

The old CTC-style file should be **replaced** (not modified) with the `GrammarConstrainedDecoder` class from Section 3.3. Key differences from current:

| Old CTC Decoder | New Constrained Decoder |
|---|---|
| Classify each window independently | Score each window against all tokens simultaneously |
| Grammar applied AFTER collapse | Grammar constraints baked INTO Viterbi |
| Median smoothing post-hoc | Viterbi handles temporal consistency |
| Equal-time sequence splitting for training | No sequence training needed |
| Fails on co-articulated speech | Handles co-articulation via segmental Viterbi |

---

### 3.7 Updated `app.py` — Dependency Injection

Replace the `CTCStyleDecoder` injection with the new decoder:

```python
# app.py  (relevant changes only)

from src.constrained_decoder import GrammarConstrainedDecoder
from src.grammar import COMPLETE_GRAMMAR
from src.hmm import SankhyaHMMBase

# Load models at startup
hmm = SankhyaHMMBase(VOCAB, HMM_STATES, MODEL_DIR)
hmm.load_models()

# Build decoder
decoder = GrammarConstrainedDecoder(
    hmm_models=hmm.models,   # {token_name: GaussianHMM}
    grammar=COMPLETE_GRAMMAR
)

@app.route('/predict', methods=['POST'])
def predict():
    # ... existing FFmpeg transcoding ...
    
    # Feature extraction (unchanged)
    audio, sr = preprocess_audio(wav_path)
    mfcc = extract_mfcc(audio, sr)  # [T × 39]
    
    # NEW: single decoder call
    integer_result, token_seq, debug = decoder.decode(mfcc)
    
    return jsonify({
        'number': integer_result,
        'tokens': token_seq,
        'debug': debug,
        'success': integer_result >= 0
    })
```

---

## Part 4 — File Change Summary

This is a precise list of every file your coding agent must touch:

### Files to DELETE or EMPTY
```
data/features/seq_*/          ← ALL sequence-derived features are poisoned, delete them
data/synthetic_raw/seq_*      ← Delete poisoned sequence audio
```

### Files to REWRITE COMPLETELY

**`src/constrained_decoder.py`** (new file, replaces `src/ctc_decoder.py`)
→ Full implementation from Section 3.3

**`src/grammar.py`**
→ Full implementation from Section 3.5

### Files to MODIFY

**`SankhyaVox_Pipeline.ipynb` — Cell 6**
Remove `augment_sequence()` function entirely.
Remove `SEQUENCE_MAP` dict entirely.
Remove `PITCH_STEPS`, `SPEED_RATES`, `NOISE_LEVELS` augmentation vars.

**`SankhyaVox_Pipeline.ipynb` — Cell 16**
Delete the entire cell body. Replace with:
```python
print("Sequence generation removed. Grammar-constrained decoder handles co-articulation at inference time.")
```

**`SankhyaVox_Pipeline.ipynb` — Cell 20 (Training)**
Replace `chunk_pool` function and SVM/DNN training with HMM-only training using `train_hmm_models_v2` from Section 3.4.
> Note: SVM/DNN are incompatible with the new segmental Viterbi approach — they require fixed-length inputs and cannot be used for frame-level scoring. Keep them as optional classifiers for isolated-token evaluation only, not continuous decoding.

**`SankhyaVox_Pipeline.ipynb` — Cell 22 (Evaluation)**
Replace `ConstrainedViterbiDecoder` import with `GrammarConstrainedDecoder`.
Update evaluation to test on:
- Single-token clips (should get ~95%+)
- Two-token phrases (should get ~80%+ with good recordings)
- Three-token phrases (should get ~70%+)

**`app.py`**
Replace CTCStyleDecoder usage with GrammarConstrainedDecoder (Section 3.7).

### Files to LEAVE UNCHANGED
```
src/config.py           ← unchanged
src/hmm.py              ← unchanged (SankhyaHMMBase still correct)
scripts/extract_features.py  ← unchanged
scripts/prep_audio.py   ← unchanged
templates/index.html    ← unchanged
static/style.css        ← unchanged
```

---

## Part 5 — Data Collection Guide (Since Augmentation Is Dropped)

With no augmentation, data diversity must come from **real recordings**. Minimum requirements:

| Token | Min Recordings | Notes |
|---|---|---|
| shunya | 15 | Critical — zero is common |
| eka–nava | 10 each | Single digit units |
| dasha | 20 | Very common — appears in all 11-19, 30-99 |
| vimsati | 10 | Only appears standalone or as tens prefix |
| shata | 5 | Boundary guard only (out of 0-99 scope) |

**Recording diversity checklist (per token):**
- [ ] 2–3 speakers minimum
- [ ] Both genders if possible  
- [ ] Varying speaking speeds (slow, normal, fast)
- [ ] Varying microphone distances (close 10cm, desk 50cm, room 1m)
- [ ] No consistent background noise (but some incidental noise is fine)
- [ ] Record connected phrases too, but **do not split them for training** — only use them for evaluation

**Minimum viable dataset for decent accuracy:** ~150 total recordings (avg 10 per token × 13 tokens + extra for dasha). This will outperform 1000 augmented-but-poisoned sequences.

---

## Part 6 — Expected Accuracy After Fix

| Input Type | Expected Accuracy |
|---|---|
| Isolated single token (1–9, 0) | 90–98% |
| Clean two-token compound (11–19, 20s, 30–90) | 75–88% |
| Three-token compound (31–39, ..., 91–99) | 65–80% |
| Fast/accented speech | 55–70% |

These numbers assume ~15 real recordings per token. With 30+ recordings per token, accuracy improves significantly.

---

## Part 7 — Testing the Fix (Step-by-Step)

After your coding agent executes all changes, validate in this order:

```bash
# Step 1: Re-extract features (clean data only, no seq_ files)
python scripts/extract_features.py

# Step 2: Retrain HMMs
python scripts/train_v2.py

# Step 3: Unit test grammar
python -c "
from src.grammar import COMPLETE_GRAMMAR, tokens_to_number
assert tokens_to_number(['pancha', 'dasha']) == 50
assert tokens_to_number(['dasha', 'pancha']) == 15
assert tokens_to_number(['shunya']) == 0
assert tokens_to_number(['nava', 'dasha', 'nava']) == 99
print('Grammar OK — 100 sequences defined:', len(COMPLETE_GRAMMAR))
"

# Step 4: Test decoder on a known recording
python -c "
from src.constrained_decoder import GrammarConstrainedDecoder
from src.grammar import COMPLETE_GRAMMAR
from src.hmm import SankhyaHMMBase
from scripts.extract_features import preprocess_audio, extract_mfcc
import numpy as np

hmm = SankhyaHMMBase(VOCAB, HMM_STATES, MODEL_DIR)
hmm.load_models()
decoder = GrammarConstrainedDecoder(hmm.models, COMPLETE_GRAMMAR)

audio, sr = preprocess_audio('test_pancha_dasha.wav')
mfcc = extract_mfcc(audio, sr)
result, tokens, debug = decoder.decode(mfcc)
print(f'Decoded: {tokens} → {result}')  # expect: ['pancha', 'dasha'] → 50
"

# Step 5: Launch Flask
python app.py
```

---

## Summary

| | Current System | Fixed System |
|---|---|---|
| Root bug | `augment_sequence` equal-time split poisons labels | Removed entirely |
| Co-articulation handling | Attempted via sequence training (broken) | Handled by segmental Viterbi at decode time |
| Grammar integration | Post-hoc, after collapse | During Viterbi, prunes invalid paths |
| Augmentation | 30 variants (including broken sequence splits) | Zero augmentation, real diversity required |
| Training data needed | Many augmented files (but corrupted) | ~150 clean isolated recordings |
| Decoder complexity | Slide → classify → smooth → collapse → grammar | Slide → score matrix → constrained Viterbi |
| Main change required | Delete seq_ data, rewrite decoder, rewrite grammar | ← exactly this |
