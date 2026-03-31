import os
import glob
import numpy as np
import sys
import joblib

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import FEATURE_DIR, MODEL_DIR, VOCAB, HMM_STATES, GMM_MIXTURES, BAUM_WELCH_ITERS, TOKEN_TO_IDX

from src.hmm import SankhyaHMMBase

def load_features_by_token():
    print(f"Reading features from {FEATURE_DIR}...")
    token_data = {v: [] for v in VOCAB}
    token_lengths = {v: [] for v in VOCAB}

    feature_files = glob.glob(os.path.join(FEATURE_DIR, "**", "*.npy"), recursive=True)
    if not feature_files:
        raise FileNotFoundError(f"No feature files found in {FEATURE_DIR}.")

    valid_count = 0
    for fpath in feature_files:
        basename = os.path.basename(fpath)
        token = None
        for v in VOCAB:
            if f"_{v}_" in basename or basename.startswith(v + "_"):
                token = v
                break

        if token and token in token_data:
            feats = np.load(fpath)
            if feats.shape[0] > 0:
                token_data[token].append(feats)
                token_lengths[token].append(feats.shape[0])
                valid_count += 1

    print(f"Loaded {valid_count} feature files across {len(VOCAB)} tokens.")
    return token_data, token_lengths

def extract_fixed_size_chunks(X, num_chunks=5):
    """Pools variable length MFCCs into 5 temporal chunks for static models like SVM/DNN."""
    n_frames = X.shape[0]
    if n_frames < num_chunks:
        X = np.pad(X, ((0, num_chunks - n_frames), (0, 0)), mode='edge')
        n_frames = num_chunks
    embeddings = []
    chunk_size = n_frames / float(num_chunks)
    for i in range(num_chunks):
        s = int(i * chunk_size)
        e = int((i + 1) * chunk_size)
        if e == s: e = s + 1
        embeddings.append(np.mean(X[s:e], axis=0))
    return np.concatenate(embeddings)

def main():
    token_data, token_lengths = load_features_by_token()
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # ── 1. Train Classical HMM ──
    print("\n--- Training Classical HMM ---")
    hmm_system = SankhyaHMMBase(VOCAB, HMM_STATES, str(MODEL_DIR), n_mix=3, n_iter=BAUM_WELCH_ITERS)
    
    for token in VOCAB:
        if len(token_data[token]) == 0: continue
        X_concat = np.concatenate(token_data[token], axis=0)
        lengths = token_lengths[token]
        hmm_system.fit_word(token, (X_concat, lengths))
    hmm_system.save_models()
    print("✓ HMM Models saved.")

    # ── Build Fixed-Size Dataset for SVM & DNN ──
    X_fixed = []
    y_fixed = []
    for token in VOCAB:
        for feat_seq in token_data[token]:
            emb = extract_fixed_size_chunks(feat_seq, num_chunks=5)
            X_fixed.append(emb)
            y_fixed.append(TOKEN_TO_IDX[token])
            
    if not X_fixed:
        print("No valid data to train SVM/DNN.")
        return
        
    X_fixed = np.array(X_fixed)
    y_fixed = np.array(y_fixed)

    # ── 2. Train RBF SVM ──
    print("\n--- Training Goated SVM Pipeline ---")
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', C=10.0, gamma='scale', probability=True))
    ])
    svm_pipeline.fit(X_fixed, y_fixed)
    joblib.dump(svm_pipeline, os.path.join(MODEL_DIR, "svm_model.pkl"))
    print(f"✓ SVM Model saved. Train Acc: {svm_pipeline.score(X_fixed, y_fixed)*100:.2f}%")

    # ── 3. Train DNN (MLP) ──
    print("\n--- Training Deep Neural Network (MLP) ---")
    dnn_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, alpha=0.001, early_stopping=True))
    ])
    dnn_pipeline.fit(X_fixed, y_fixed)
    joblib.dump(dnn_pipeline, os.path.join(MODEL_DIR, "dnn_model.pkl"))
    print(f"✓ DNN Model saved. Train Acc: {dnn_pipeline.score(X_fixed, y_fixed)*100:.2f}%")

    print("\nAll Multi-Model Training Complete!")

if __name__ == "__main__":
    main()
