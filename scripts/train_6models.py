"""
SankhyaVox — Train 6 HMM Model Variants.

Trains 3 real-data models and 3 synthetic-data models with different GMM complexities.
Each model set = 13 per-token HMMs saved as a single .pkl file.

Usage:
    python scripts/train_6models.py
"""

import os
import sys
import glob
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import VOCAB, HMM_STATES, MODEL_DIR, FEATURE_DIR
from src.hmm import SankhyaHMMBase

# ── Model Configurations ──
CONFIGS = [
    # (name,       data_source, n_mix, n_iter)
    ("real_3mix",   "real",      3,     25),
    ("real_5mix",   "real",      5,     40),
    ("real_8mix",   "real",      8,     50),
    ("synth_3mix",  "synth",     3,     25),
    ("synth_5mix",  "synth",     5,     40),
    ("synth_8mix",  "synth",     8,     50),
]


def load_features(source: str) -> dict:
    """
    Load feature .npy files, grouped by token.
    
    source='real'  → only data/features/S* directories
    source='synth' → only data/features/synth_* directories
    """
    feat_dir = str(FEATURE_DIR)
    token_data = {v: [] for v in VOCAB}
    token_lengths = {v: [] for v in VOCAB}

    # Find all .npy files
    all_npy = sorted(glob.glob(os.path.join(feat_dir, "**", "*.npy"), recursive=True))

    for fp in all_npy:
        # Determine which source this file belongs to
        rel = os.path.relpath(fp, feat_dir)
        top_dir = rel.split(os.sep)[0] if os.sep in rel else ""

        if source == "real":
            # Only S01, S02, ... directories
            if not top_dir.startswith("S"):
                continue
        elif source == "synth":
            # Only synth_* directories
            if not top_dir.startswith("synth_"):
                continue

        # Match token from filename
        bn = os.path.basename(fp).replace(".npy", "")
        matched_token = None
        for token in VOCAB:
            # Check if the filename ends with _<token> or contains _<token>_
            if bn.endswith(f"_{token}") or f"_{token}_" in bn:
                matched_token = token
                break

        if matched_token is None:
            continue

        feats = np.load(fp)
        if feats.shape[0] >= 5 and feats.ndim == 2:
            token_data[matched_token].append(feats)
            token_lengths[matched_token].append(feats.shape[0])

    return token_data, token_lengths


def train_model(name: str, source: str, n_mix: int, n_iter: int):
    """Train a single model configuration."""
    print(f"\n{'='*60}")
    print(f"  Training: {name}")
    print(f"  Source: {source} | GMM mixtures: {n_mix} | EM iters: {n_iter}")
    print(f"{'='*60}")

    token_data, token_lengths = load_features(source)

    # Report counts
    total_samples = 0
    for token in VOCAB:
        count = len(token_data[token])
        total_samples += count
        if count < 5:
            print(f"  ⚠ {token}: only {count} samples")
        else:
            avg_frames = np.mean(token_lengths[token]) if token_lengths[token] else 0
            print(f"  ✓ {token}: {count} samples, avg {avg_frames:.0f} frames")

    if total_samples == 0:
        print(f"  ✗ NO DATA for {source}! Skipping.")
        return

    print(f"\n  Total: {total_samples} samples across {len(VOCAB)} tokens")

    # Train
    os.makedirs(str(MODEL_DIR), exist_ok=True)
    hmm = SankhyaHMMBase(VOCAB, HMM_STATES, str(MODEL_DIR), n_mix=n_mix, n_iter=n_iter)

    trained_count = 0
    for token in VOCAB:
        if not token_data[token]:
            print(f"  SKIP {token}: no data")
            continue

        all_seqs = np.concatenate(token_data[token], axis=0)
        lengths = token_lengths[token]

        try:
            hmm.fit_word(token, (all_seqs, lengths))
            trained_count += 1
        except Exception as e:
            print(f"  FAILED {token}: {e}")

    # Save to custom path
    import joblib
    out_path = os.path.join(str(MODEL_DIR), f"{name}.pkl")
    joblib.dump(hmm.models, out_path)
    print(f"\n  ✓ Saved {trained_count} token HMMs → {out_path}")


def main():
    print("=" * 60)
    print("  SankhyaVox — 6-Model Training Pipeline")
    print("=" * 60)

    for name, source, n_mix, n_iter in CONFIGS:
        train_model(name, source, n_mix, n_iter)

    print("\n" + "=" * 60)
    print("  ALL 6 MODELS TRAINED!")
    print("  Model files in:", str(MODEL_DIR))
    print("=" * 60)


if __name__ == "__main__":
    main()
