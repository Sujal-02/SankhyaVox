"""
SankhyaVox — Evaluate 6 HMM Model Variants.

Loads each of the 6 trained model .pkl files, builds a GrammarConstrainedDecoder,
and evaluates on held-out real speaker segments (isolated tokens).

Usage:
    python scripts/evaluate_6models.py
"""

import os
import sys
import glob
import numpy as np
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import VOCAB, HMM_STATES, MODEL_DIR, FEATURE_DIR
from src.constrained_decoder import GrammarConstrainedDecoder
from src.grammar import COMPLETE_GRAMMAR

MODEL_NAMES = [
    "real_3mix", "real_5mix", "real_8mix",
    "synth_3mix", "synth_5mix", "synth_8mix",
]


def load_test_data():
    """
    Load real speaker feature files as test data.
    Returns list of (token_label, feature_array) tuples.
    """
    feat_dir = str(FEATURE_DIR)
    test_items = []

    for speaker_dir in sorted(glob.glob(os.path.join(feat_dir, "S*"))):
        for npy_path in sorted(glob.glob(os.path.join(speaker_dir, "*.npy"))):
            bn = os.path.basename(npy_path).replace(".npy", "")
            
            # Find which token this file belongs to
            matched_token = None
            for token in VOCAB:
                if bn.endswith(f"_{token}") or f"_{token}_" in bn:
                    matched_token = token
                    break
                # Also handle format: S01_pancha_01
                parts = bn.split("_")
                for part in parts:
                    if part == token:
                        matched_token = token
                        break
                if matched_token:
                    break

            if matched_token is None:
                continue

            feats = np.load(npy_path)
            if feats.shape[0] >= 5 and feats.ndim == 2:
                test_items.append((matched_token, feats, npy_path))

    return test_items


def evaluate_model(model_name: str, test_items: list) -> dict:
    """
    Evaluate a single model on isolated token test clips.
    
    For isolated tokens, the decoder should return a single-token sequence
    that maps to 0-9 (or 10/20/100 for dasha/vimsati/shata).
    """
    model_path = os.path.join(str(MODEL_DIR), f"{model_name}.pkl")
    if not os.path.exists(model_path):
        return {"error": f"Model not found: {model_path}"}

    models = joblib.load(model_path)

    # Build decoder
    decoder = GrammarConstrainedDecoder(
        hmm_models=models,
        grammar=COMPLETE_GRAMMAR
    )

    # Token → expected integer for single-token clips
    token_to_int = {
        'shunya': 0, 'eka': 1, 'dvi': 2, 'tri': 3, 'catur': 4,
        'pancha': 5, 'shat': 6, 'sapta': 7, 'ashta': 8, 'nava': 9,
        'dasha': 10, 'vimsati': 20, 'shata': -1  # shata not in 0-99
    }

    results = {tok: {"correct": 0, "total": 0} for tok in VOCAB}
    total_correct = 0
    total_count = 0

    for true_token, feats, path in test_items:
        expected_int = token_to_int.get(true_token)
        if expected_int is None:
            continue

        try:
            # Decode
            scores, _ = decoder.compute_score_matrix(feats)
            token_seq, confidence = decoder.viterbi_decode(scores)

            # For isolated tokens, check if the decoded sequence
            # contains the correct token
            predicted_int = decoder._tokens_to_integer(token_seq)

            # Also check: does the decoded token sequence contain
            # the expected token at all?
            token_match = true_token in token_seq

            # Primary metric: exact integer match
            # For shata (out of 0-99 scope), check token match only
            if expected_int == -1:
                correct = token_match
            else:
                correct = (predicted_int == expected_int)

            if correct:
                total_correct += 1
                results[true_token]["correct"] += 1

            results[true_token]["total"] += 1
            total_count += 1
        except Exception as e:
            results[true_token]["total"] += 1
            total_count += 1

    return {
        "per_token": results,
        "total_correct": total_correct,
        "total_count": total_count,
        "accuracy": total_correct / total_count * 100 if total_count > 0 else 0,
    }


def main():
    print("=" * 70)
    print("  SankhyaVox — 6-Model Evaluation")
    print("=" * 70)

    # Load test data
    test_items = load_test_data()
    print(f"\n  Test set: {len(test_items)} isolated token clips from real speakers\n")

    if not test_items:
        print("  ERROR: No test data found in data/features/S*/")
        return

    # Token distribution
    from collections import Counter
    token_counts = Counter(t for t, _, _ in test_items)
    for tok in VOCAB:
        print(f"    {tok}: {token_counts.get(tok, 0)} clips")

    # Evaluate each model
    all_results = {}
    for name in MODEL_NAMES:
        print(f"\n{'─'*70}")
        print(f"  Evaluating: {name}")
        print(f"{'─'*70}")

        result = evaluate_model(name, test_items)
        if "error" in result:
            print(f"  ✗ {result['error']}")
            continue

        all_results[name] = result
        print(f"  Overall: {result['total_correct']}/{result['total_count']} = {result['accuracy']:.1f}%")

        # Per-token breakdown
        for tok in VOCAB:
            r = result["per_token"][tok]
            if r["total"] > 0:
                acc = r["correct"] / r["total"] * 100
                bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
                print(f"    {tok:10s} {r['correct']:3d}/{r['total']:3d}  {bar} {acc:5.1f}%")

    # Summary comparison table
    if all_results:
        print(f"\n{'='*70}")
        print("  COMPARISON TABLE")
        print(f"{'='*70}")
        print(f"  {'Model':<15s} {'Accuracy':>10s} {'Correct':>10s} {'Total':>8s}")
        print(f"  {'─'*45}")

        best_name = ""
        best_acc = 0
        for name in MODEL_NAMES:
            if name in all_results:
                r = all_results[name]
                marker = ""
                if r["accuracy"] > best_acc:
                    best_acc = r["accuracy"]
                    best_name = name
                print(f"  {name:<15s} {r['accuracy']:>9.1f}% {r['total_correct']:>10d} {r['total_count']:>8d}")

        print(f"\n  🏆 BEST MODEL: {best_name} ({best_acc:.1f}%)")
        print(f"     To use it in app.py, set MODEL_FILE = '{best_name}.pkl'")

    print()


if __name__ == "__main__":
    main()
