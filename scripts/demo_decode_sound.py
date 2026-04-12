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
from typing import Optional

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.pipeline import DataPipeline
from models.hmm_classifier import SankhyaHMM
from src.decoder import GrammarConstrainedDecoder
from src.grammar import number_to_tokens

DEFAULT_CHECKPOINT = "checkpoints/hmm_classifier.pkl"


def decode_audio(audio_path: str, checkpoint_path: Optional[str] = None,
                 verbose: bool = False):
    """Decode a single audio file and return (integer, tokens, debug)."""
    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT

    audio = Path(audio_path)
    if not audio.exists():
        raise FileNotFoundError(f"Audio file not found: {audio}")

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    hmm = SankhyaHMM(checkpoint_path=str(ckpt_path))
    decoder = GrammarConstrainedDecoder(hmm)

    pipe = DataPipeline()
    mfcc = pipe.process_single(str(audio))

    if verbose:
        print(f"Processing: {audio}")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"  MFCC shape: {mfcc.shape}  ({mfcc.shape[0] / 100:.2f}s)")

    result, tokens, debug = decoder.decode(mfcc, verbose=verbose)
    return result, tokens, debug


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
        "--verbose", "-v", action="store_true", help="Print detailed decoding info."
    )
    args = parser.parse_args()

    try:
        result, tokens, debug = decode_audio(
            args.audio, args.checkpoint, args.verbose
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

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
    # Examples:
    #   python scripts/demo_decode_sound.py data_processed/human/segments/S01/S01_007_01.wav
    #   python scripts/demo_decode_sound.py recording.wav --checkpoint checkpoints/hmm_classifier.pkl
    #   python scripts/demo_decode_sound.py recording.wav -v
    main()
