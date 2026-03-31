import numpy as np

class ConstrainedViterbiDecoder:
    """
    Decodes an observation sequence into a grammar-valid sequence of words.
    Uses the trained word HMMs to score the sequence.
    For simplicity in this pipeline, we will use a sequence-level constrained decoding.
    We score every valid token sequence (0-99) and find the sequence that yields 
    the maximum composite log-likelihood.
    
    This is an approximation of full continuous Viterbi, perfectly suitable for 
    isolated connected-digit segments, since we have only 100 valid paths.
    """
    def __init__(self, hmm_system, all_valid_sequences_fn):
        self.hmm_system = hmm_system
        # dict: number -> list of word tokens
        self.valid_paths = all_valid_sequences_fn() 

    def decode(self, X):
        """
        Given MFCC features X for an entire utterance, parse it into the best 
        Sanskrit number 0-99.
        
        Because hmmlearn word models don't naturally compose into a single unified 
        continuous hmmlearn object easily, and the total valid sequences = 100,
        we can use word-level forced alignment or window sliding. 
        For isolated components, we can roughly approximate by scoring the whole 
        clip against composite models, or just use the model sequence.
        
        A true continuous Viterbi concatenates the states. Here, we build a 
        large state-transition matrix for each of the 100 numbers and score X.
        """
        best_score = float("-inf")
        best_number = None

        for number, tokens in self.valid_paths.items():
            score = self.score_composite_model(tokens, X)
            if score > best_score:
                best_score = score
                best_number = number

        return best_number, best_score

    def score_composite_model(self, tokens, X):
        """
        Concatenates the HMMs in `tokens` to score the sequence X.
        Since hmmlearn doesn't support easy concatenation, we do it manually.
        """
        from hmmlearn.hmm import GMMHMM
        
        # Sum total states, mixtures, dims
        total_states = sum(self.hmm_system.models[t].n_components for t in tokens)
        n_mix = self.hmm_system.models[tokens[0]].n_mix
        n_features = X.shape[1]
        
        # New model
        comp = GMMHMM(n_components=total_states, n_mix=n_mix, covariance_type="diag")
        comp.startprob_ = np.zeros(total_states)
        comp.startprob_[0] = 1.0
        
        comp.transmat_ = np.zeros((total_states, total_states))
        comp.means_ = np.zeros((total_states, n_mix, n_features))
        comp.covars_ = np.zeros((total_states, n_mix, n_features))
        comp.weights_ = np.zeros((total_states, n_mix))
        
        # Fill in parameters
        state_offset = 0
        for i, token in enumerate(tokens):
            model = self.hmm_system.models[token]
            n_states = model.n_components
            
            # Means, covars, weights
            comp.means_[state_offset:state_offset+n_states] = model.means_
            comp.covars_[state_offset:state_offset+n_states] = model.covars_
            comp.weights_[state_offset:state_offset+n_states] = model.weights_
            
            # Transition matrix
            comp.transmat_[state_offset:state_offset+n_states, state_offset:state_offset+n_states] = model.transmat_
            
            # Fix exit transition of this word to enter the next word
            if i < len(tokens) - 1:
                # Last state of current word transitions into first state of next word
                comp.transmat_[state_offset+n_states-1, state_offset+n_states-1] = 0.5
                comp.transmat_[state_offset+n_states-1, state_offset+n_states] = 0.5
                
            state_offset += n_states
            
        # Ensure rows sum to 1
        for row in range(total_states):
            s = np.sum(comp.transmat_[row])
            if s > 0:
                comp.transmat_[row] /= s
            else:
                comp.transmat_[row, row] = 1.0
                
        # hmmlearn expects n_features to be explicitly set when we manually assemble a model
        comp.n_features = n_features
        
        try:
            logprob = comp.score(X)
            return logprob
        except Exception as e:
            # If the sequence is too short for the number of states, it will fail
            return float("-inf")
