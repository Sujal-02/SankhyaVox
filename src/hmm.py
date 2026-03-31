import os
import joblib
import numpy as np
from hmmlearn import hmm

# Attempt to configure hmmlearn left-to-right (Bakis) model
def build_left_to_right_hmm(n_states, n_mixtures=1, n_iter=15, n_features=39):
    """
    Builds a left-to-right HMM with GMM emissions.
    States only transition to themselves or the next state.
    """
    startprob = np.zeros(n_states)
    startprob[0] = 1.0

    transmat = np.zeros((n_states, n_states))
    for i in range(n_states):
        if i == n_states - 1:
            transmat[i, i] = 1.0
        else:
            transmat[i, i] = 0.5
            transmat[i, i + 1] = 0.5
            
    model = hmm.GMMHMM(
        n_components=n_states, 
        n_mix=n_mixtures, 
        covariance_type="diag", 
        init_params="mcw",  # K-Means init
        n_iter=n_iter, 
        tol=1e-4, 
        random_state=42
    )

    model.startprob_ = startprob
    model.transmat_ = transmat

    return model

class SankhyaHMMBase:
    def __init__(self, vocab_config, state_config, model_dir, n_mix=1, n_iter=15):
        self.vocab = vocab_config
        self.state_config = state_config
        self.model_dir = model_dir
        self.n_mix = n_mix
        self.n_iter = n_iter
        self.models = {}

    def fit_word(self, word, X_lengths_tuple):
        """
        Train an HMM for a specific word given concatenated features X 
        and a list of lengths (hmmlearn format).
        """
        X, lengths = X_lengths_tuple
        n_states = self.state_config.get(word, 5)
        model = build_left_to_right_hmm(n_states, n_mixtures=self.n_mix, n_iter=self.n_iter)
        print(f"Training HMM for '{word}' with {n_states} states, {self.n_mix} mixtures, {len(lengths)} examples...")
        
        # Train
        model.fit(X, lengths)
        self.models[word] = model

    def save_models(self):
        """Save trained models to disk."""
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, "sankhya_hmm_models.pkl")
        joblib.dump(self.models, model_path)
        print(f"Models saved to {model_path}")

    def load_models(self):
        """Load trained models from disk."""
        model_path = os.path.join(self.model_dir, "sankhya_hmm_models.pkl")
        if os.path.exists(model_path):
            self.models = joblib.load(model_path)
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
    def score_frame_sequence(self, word, X):
        """Compute log likelihood of sequence X for a single word model."""
        if word not in self.models:
            return float("-inf")
        return self.models[word].score(X)
