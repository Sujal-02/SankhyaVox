import numpy as np

class GrammarConstrainedDecoder:
    """
    Decodes continuous Sanskrit digit speech using grammar-constrained Viterbi.
    
    Integrates grammar constraints DURING search so impossible token 
    transitions are pruned early. Handles 0-99 recognition.
    """
    
    def __init__(self, hmm_models: dict, grammar: dict):
        self.models = hmm_models       # {token_name: trained GMMHMM}
        self.grammar = grammar         # {integer_value: [token_sequence]}
        self.vocab = list(hmm_models.keys())
        self.valid_successors = self._build_successor_map()
    
    def _build_successor_map(self) -> dict:
        """
        Build valid successor tokens for each token from grammar rules.
        Every token can also follow itself (self-transition = token sustains).
        """
        successors = {tok: set() for tok in self.vocab}
        
        for integer_val, token_seq in self.grammar.items():
            for i in range(len(token_seq) - 1):
                successors[token_seq[i]].add(token_seq[i + 1])
        
        # Self-transitions: every token can sustain across windows
        for tok in self.vocab:
            successors[tok].add(tok)
            
        return successors
    
    def compute_score_matrix(self, mfcc_matrix: np.ndarray) -> tuple:
        """
        Score sliding windows against all 13 token HMMs.
        
        Returns:
            scores: [N_windows × 13 tokens]
        """
        T = mfcc_matrix.shape[0]
        WINDOW = 25    # ~250ms at 100fps
        HOP    = 5     # ~50ms hop
        
        windows = []
        start = 0
        while start + WINDOW <= T:
            windows.append(mfcc_matrix[start : start + WINDOW])
            start += HOP
        
        # Handle remainder
        if start < T and T - start >= 5:
            remainder = mfcc_matrix[start:]
            padded = np.pad(remainder, ((0, WINDOW - len(remainder)), (0, 0)), mode='edge')
            windows.append(padded)
        
        if not windows:
            if T >= 5:
                padded = np.pad(mfcc_matrix, ((0, max(0, WINDOW - T)), (0, 0)), mode='edge')
                windows.append(padded)
            else:
                return np.zeros((1, len(self.vocab))), []
        
        N = len(windows)
        n_vocab = len(self.vocab)
        scores = np.full((N, n_vocab), -np.inf)
        
        for i, window in enumerate(windows):
            for j, token in enumerate(self.vocab):
                model = self.models.get(token)
                if model is None:
                    continue
                try:
                    scores[i, j] = model.score(window)
                except:
                    scores[i, j] = -1e9
        
        return scores, windows
    
    def viterbi_decode(self, scores: np.ndarray) -> tuple:
        """
        Grammar-constrained Viterbi over score matrix.
        Only valid grammar sequences are allowed.
        """
        N = scores.shape[0]
        n_states = len(self.vocab)
        tok_to_idx = {tok: i for i, tok in enumerate(self.vocab)}
        idx_to_tok = {i: tok for tok, i in tok_to_idx.items()}
        
        # 1. Which tokens can START a sequence?
        valid_starts = set()
        for seq in self.grammar.values():
            valid_starts.add(seq[0])
            
        # 2. Which tokens can END a sequence?
        valid_ends = set()
        for seq in self.grammar.values():
            valid_ends.add(seq[-1])
        
        # Build allowed transition matrix
        allowed = np.zeros((n_states, n_states), dtype=bool)
        for tok, succs in self.valid_successors.items():
            i = tok_to_idx.get(tok)
            if i is None: continue
            for s in succs:
                j = tok_to_idx.get(s)
                if j is not None: allowed[i, j] = True
        
        # Viterbi DP
        viterbi = np.full((N, n_states), -np.inf)
        backtrack = np.full((N, n_states), -1, dtype=int)
        
        # Init: only valid start tokens
        for tok in valid_starts:
            s = tok_to_idx.get(tok)
            if s is not None:
                viterbi[0, s] = scores[0, s]
        
        # Fill
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
        
        # Find best end state (MUST be a valid end token)
        best_end = -1
        best_score = -np.inf
        for tok in valid_ends:
            s = tok_to_idx.get(tok)
            if s is not None and viterbi[N-1, s] > best_score:
                best_score = viterbi[N-1, s]
                best_end = s
                
        if best_end == -1:
            return [], -9999.0
            
        # Backtrack
        path = []
        s = best_end
        for t in range(N-1, -1, -1):
            path.append(s)
            prev = backtrack[t, s]
            if prev < 0: break
            s = prev
        path.reverse()
        
        # Collapse consecutive identical tokens
        token_path = [idx_to_tok[s] for s in path]
        collapsed = []
        for tok in token_path:
            if not collapsed or collapsed[-1] != tok:
                collapsed.append(tok)
        
        return collapsed, best_score
        
        return collapsed, best_score
    
    def decode(self, mfcc_matrix: np.ndarray) -> tuple:
        """
        Full pipeline: MFCC → integer.
        Returns (integer_result, token_sequence, debug_info).
        integer_result = -1 if recognition failed.
        """
        scores, _ = self.compute_score_matrix(mfcc_matrix)
        token_seq, confidence = self.viterbi_decode(scores)
        
        integer_result = self._tokens_to_integer(token_seq)
        
        debug = {
            'token_sequence': token_seq,
            'confidence': float(confidence),
            'score_matrix_shape': list(scores.shape)
        }
        
        return integer_result, token_seq, debug
    
    def _tokens_to_integer(self, tokens: list) -> int:
        """Map token sequence → integer using grammar. -1 if invalid."""
        # Exact match
        for integer_val, seq in self.grammar.items():
            if seq == tokens:
                return integer_val
        
        # Fuzzy: try subsequences
        for length in range(1, min(4, len(tokens) + 1)):
            for start in range(len(tokens) - length + 1):
                subseq = tokens[start : start + length]
                for integer_val, seq in self.grammar.items():
                    if seq == subseq:
                        return integer_val
        
        return -1
