"""
Debug script to trace Viterbi decoding on a test file.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))
import joblib
from src.config import MODEL_DIR
from src.constrained_decoder import GrammarConstrainedDecoder
from src.grammar import COMPLETE_GRAMMAR
import soundfile as sf
import librosa
from scripts.extract_features import preprocess_audio, extract_mfcc

# Find S04_ashta_06.wav
test_wav = r"c:\Users\rakes\OneDrive\Desktop\Sujal\SankhyaVox\data\segments\S04\S04_ashta_06.wav"
if not os.path.exists(test_wav):
    print(f"File not found: {test_wav}")
    sys.exit(1)

audio, sr = sf.read(test_wav)
if audio.ndim > 1: audio = audio.mean(axis=1)
if sr != 16000: audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

audio = preprocess_audio(audio)
X = extract_mfcc(audio)
print(f"Extracted MFCC shape: {X.shape}")

model_path = os.path.join(str(MODEL_DIR), "synth_5mix.pkl")
models = joblib.load(model_path)
decoder = GrammarConstrainedDecoder(models, COMPLETE_GRAMMAR)

scores, windows = decoder.compute_score_matrix(X)
print(f"Score matrix shape: {scores.shape}")

print("\n--- Scores (First Window) ---")
for tok, idx in zip(decoder.vocab + ['SIL'], range(len(decoder.vocab)+1)):
    print(f"{tok:>10}: {scores[0, idx]:.2f}")

print("\n--- Viterbi DP Trace ---")
# Run exact code from viterbi_decode to see what happens
N = scores.shape[0]
n_states = len(decoder.vocab) + 1
allowed = np.zeros((n_states, n_states), dtype=bool)
tok_to_idx = {tok: i for i, tok in enumerate(decoder.vocab + ['SIL'])}
for tok, succs in decoder.valid_successors.items():
    i = tok_to_idx.get(tok)
    for s in succs:
        j = tok_to_idx.get(s)
        if j is not None: allowed[i, j] = True

viterbi = np.full((N, n_states), -np.inf)
backtrack = np.full((N, n_states), -1, dtype=int)
for s in range(n_states): viterbi[0, s] = scores[0, s]

for t in range(1, N):
    for s in range(n_states):
        emission = scores[t, s]
        if emission < -1e8: continue
        best_prev = -np.inf
        best_prev_s = -1
        for prev_s in range(n_states):
            if not allowed[prev_s, s]: continue
            if viterbi[t-1, prev_s] > best_prev:
                best_prev = viterbi[t-1, prev_s]
                best_prev_s = prev_s
        if best_prev_s >= 0:
            viterbi[t, s] = best_prev + emission
            backtrack[t, s] = best_prev_s

# Find best end state
print("\n--- End State Candidates ---")
for s in range(n_states):
    if viterbi[N-1, s] > -np.inf:
        print(f"{decoder.vocab[s] if s < len(decoder.vocab) else 'SIL'}: {viterbi[N-1, s]:.2f}")

best_end = np.argmax(viterbi[N-1])
best_score = viterbi[N-1, best_end]
print(f"\nBest End State: {best_end} ({decoder.vocab[best_end] if best_end < len(decoder.vocab) else 'SIL'}) with score {best_score}")

# Backtrack
path = []
s = best_end
for t in range(N-1, -1, -1):
    path.append(s)
    prev = backtrack[t, s]
    if prev < 0:
        print(f"Path broken at t={t}, state={s}")
        break
    s = prev
path.reverse()

idx_to_tok = {v: k for k, v in tok_to_idx.items()}
token_path = [idx_to_tok[s] for s in path]
print("\nRaw Path:", token_path)

collapsed = []
for tok in token_path:
    if tok == 'SIL': continue
    if not collapsed or collapsed[-1] != tok:
        collapsed.append(tok)
print("\nCollapsed:", collapsed)
