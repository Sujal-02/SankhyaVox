import os
import sys
import glob
import random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import VOCAB, HMM_STATES, MODEL_DIR, FEATURE_DIR
from src.hmm import SankhyaHMMBase
from src.decoder import ConstrainedViterbiDecoder
from src.grammar import all_valid_sequences, tokens_to_number

def get_true_label(filename):
    """Extract true word token and its numeric value from filename."""
    basename = os.path.basename(filename)
    # Filename format: S01_eka_01.npy
    for v in VOCAB:
        if f"_{v}_" in basename or basename.startswith(v + "_"):
            return v, tokens_to_number([v])
    return None, None

def main():
    print("Loading HMM system...")
    hmm_system = SankhyaHMMBase(VOCAB, HMM_STATES, str(MODEL_DIR))
    try:
        hmm_system.load_models()
        print(f"Successfully loaded {len(hmm_system.models)} word models.")
    except Exception as e:
        print(f"Failed to load models. Have you run scripts/train.py? Error: {e}")
        return

    print("Initializing Grammar-constrained Viterbi Decoder...")
    decoder = ConstrainedViterbiDecoder(hmm_system, all_valid_sequences)

    print("\n--- Model Evaluation (Subset of Training Data) ---")
    
    # Read all feature files
    feature_files = glob.glob(os.path.join(FEATURE_DIR, "**", "*.npy"), recursive=True)
    
    if not feature_files:
        print(f"No feature files found in {FEATURE_DIR}.")
        print("Please run python scripts/extract_features.py first.")
        return

    # Randomly select up to 15 samples from the dataset to test
    random.seed(42)  # For reproducible results
    n_samples = min(15, len(feature_files))
    test_samples = random.sample(feature_files, n_samples)
    
    correct = 0
    total = 0
    
    print(f"{'File':<25} | {'True Number':<12} | {'Predicted':<10} | {'Status'}")
    print("-" * 65)
    
    for fpath in test_samples:
        true_token, true_number = get_true_label(fpath)
        if true_token is None:
            continue
            
        total += 1
        
        # Extract features
        X = np.load(fpath)
        
        # Decode sequence
        # Note: A single isolated digit is perfectly legal in our grammar (e.g. "eka" = 1)
        pred_number, score = decoder.decode(X)
        
        # Format output
        filename = os.path.basename(fpath)
        
        if pred_number == true_number:
            status = "✅ PASS"
            correct += 1
        else:
            status = "❌ FAIL"
            
        print(f"{filename:<25} | {str(true_number):<12} | {str(pred_number):<10} | {status}")

    if total > 0:
        accuracy = (correct / total) * 100
        print("-" * 65)
        print(f"Test complete! Accuracy on {total} random samples: {accuracy:.2f}% ({correct}/{total})")
    else:
        print("No valid labels found for testing.")

if __name__ == "__main__":
    main()
