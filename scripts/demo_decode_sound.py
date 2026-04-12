"""
SankhyaVox – Demo: Decode a compound Sanskrit number from a WAV file.

Loads a trained SankhyaHMM checkpoint and the grammar-constrained decoder,
processes a raw audio file through the data pipeline, and prints the
recognised integer (0–99).

Usage:
    python scripts/demo_decode_sound.py path/to/recording.wav
    python scripts/demo_decode_sound.py path/to/recording.wav --checkpoint checkpoints/hmm_classifier.pkl
    python scripts/demo_decode_sound.py path/to/recording.wav --verbose
    python scripts/demo_decode_sound.py path/to/recording.wav --isolated --verbose
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.hmm_classifier import SankhyaHMM
from src.decoder import DEFAULT_CHECKPOINT, decode_audio
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
        default=None,
        help="Path to the trained SankhyaHMM pickle (default: checkpoints/hmm_classifier.pkl).",
    )
    parser.add_argument(
        "--isolated", action="store_true",
        help="Predict a single isolated token instead of a compound number.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print detailed decoding info."
    )
    args = parser.parse_args()

    ckpt = args.checkpoint or DEFAULT_CHECKPOINT
    hmm = SankhyaHMM(checkpoint_path=ckpt)

    try:
        result = decode_audio(
            hmm, args.audio, args.verbose, args.isolated
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print()
    if args.isolated:
        token, label, debug = result
        print(f"  Token: {token}")
        print(f"  Label: {label}")
        print(f"  Score: {debug['best_score']}")
    else:
        integer: int = result[0]  # type: ignore[assignment]
        tokens: list[str] = result[1]  # type: ignore[assignment]
        debug: dict = result[2]  # type: ignore[assignment]
        if integer >= 0:
            expected_tokens = number_to_tokens(integer)
            print(f"  Result:  {integer}")
            print(f"  Tokens:  {' + '.join(tokens)}")
            print(f"  Grammar: {' + '.join(expected_tokens)}")
            print(f"  Score:   {debug['viterbi_score']}")
        else:
            print(f"  Recognition failed.")
            print(f"  Decoded tokens: {tokens}")
            print(f"  Score: {debug['viterbi_score']}")
            sys.exit(1)


if __name__ == "__main__":
    # Examples:
    #   python scripts/demo_decode_sound.py data_processed/human/segments/S01/S01_007_01.wav
    #   python scripts/demo_decode_sound.py recording.wav --checkpoint checkpoints/hmm_classifier.pkl
    #   python scripts/demo_decode_sound.py recording.wav -v
    #   python scripts/demo_decode_sound.py recording.wav --isolated -v
    main()
