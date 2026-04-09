"""
SankhyaVox – Demo: Decode a compound Sanskrit number from a WAV file.

Loads a trained SankhyaHMM checkpoint and the grammar-constrained decoder,
processes a raw audio file through the data pipeline, and prints the
recognised integer (0–99).

Usage:
    python scripts/demo_decode_sound.py path/to/recording.wav
    python scripts/demo_decode_sound.py path/to/recording.wav --checkpoint checkpoints/hmm_classifier.pkl
    python scripts/demo_decode_sound.py path/to/recording.wav --verbose
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.pipeline import DataPipeline
from models.hmm_classifier import SankhyaHMM
from src.decoder import GrammarConstrainedDecoder
from src.grammar import number_to_tokens


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode a compound Sanskrit number (0–99) from a WAV file."
    )
    parser.add_argument(
        "audio", type=str, help="Path to the input audio file (WAV, MP3, M4A, etc.)."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/hmm_classifier.pkl",
        help="Path to the trained SankhyaHMM pickle (default: checkpoints/hmm_classifier.pkl).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print detailed decoding info."
    )
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: audio file not found: {audio_path}")
        sys.exit(1)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found: {ckpt_path}")
        print("  Train the HMM first (see notebooks/train_hmm.ipynb) and save to checkpoints/.")
        sys.exit(1)

    # Load model
    hmm = SankhyaHMM(checkpoint_path=str(ckpt_path))

    # Build decoder
    decoder = GrammarConstrainedDecoder(hmm)

    # Process audio → MFCC features
    pipe = DataPipeline()
    print(f"Processing: {audio_path}")
    mfcc = pipe.process_single(str(audio_path))
    print(f"  MFCC shape: {mfcc.shape}  ({mfcc.shape[0] / 100:.2f}s)")

    # Decode
    result, tokens, debug = decoder.decode(mfcc, verbose=args.verbose)

    # Output
    print()
    if result >= 0:
        expected_tokens = number_to_tokens(result)
        print(f"  Result:  {result}")
        print(f"  Tokens:  {' + '.join(tokens)}")
        print(f"  Grammar: {' + '.join(expected_tokens)}")
        print(f"  Score:   {debug['viterbi_score']}")
    else:
        print(f"  Recognition failed.")
        print(f"  Decoded tokens: {tokens}")
        print(f"  Score: {debug['viterbi_score']}")
        sys.exit(1)


if __name__ == "__main__":
    # Example: decode a compound Sanskrit number from a WAV file
    #   python scripts/demo_decode_sound.py path/to/recording.wav
    #   python scripts/demo_decode_sound.py path/to/recording.wav --checkpoint checkpoints/hmm_classifier.pkl
    #   python scripts/demo_decode_sound.py path/to/recording.wav --verbose
    main()
