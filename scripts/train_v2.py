import os
import sys
import glob
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.hmm import SankhyaHMMBase

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

if __name__ == "__main__":
    from src.config import VOCAB, HMM_STATES, MODEL_DIR
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Training per-token HMMs (No Augmentation)...")
    # 'data/features/' is the default path relative to the root SankhyaVox directory
    # since we added the root to sys.path, we can pass root-relative paths
    feature_directory = os.path.join(os.path.dirname(__file__), '..', 'data', 'features')
    train_hmm_models_v2(feature_directory, VOCAB, HMM_STATES, MODEL_DIR)
