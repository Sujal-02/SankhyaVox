"""
SankhyaVox – Demo: Generate compound Sanskrit number audio via TTS.

Synthesises isolated token WAVs using Edge TTS, then concatenates them
with silence gaps to produce a compound number utterance.

Usage:
    python scripts/demo_generate_number.py 57
    python scripts/demo_generate_number.py 0 15 42 99

Output:
    temp/pancha_dasha_sapta.wav   (for 57)
"""

import argparse
import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf

from src.config import SAMPLE_RATE
from src.grammar import number_to_tokens

# ── Settings ──────────────────────────────────────────────────────────────────

SILENCE_GAP_MS = 0      # ms of silence between tokens
TTS_VOICE = "en-IN-PrabhatNeural"
OUTPUT_DIR = Path("temp")


async def _synthesise_token(text: str, voice: str, out_path: str) -> None:
    """Synthesise a single token to a WAV file using Edge TTS."""
    import edge_tts
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(out_path)


def synthesise_token(text: str, voice: str, out_path: str) -> None:
    """Synchronous wrapper for Edge TTS synthesis."""
    asyncio.run(_synthesise_token(text, voice, out_path))


def generate_compound(number: int) -> Path:
    """
    Generate a compound Sanskrit number WAV file.

    Returns the output path.
    """
    tokens = number_to_tokens(number)
    out_name = "_".join(tokens) + ".wav"
    out_path = OUTPUT_DIR / out_name

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Synthesise each token individually
    clips = []
    tmp_files = []
    try:
        for tok in tokens:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            tmp_files.append(tmp.name)

            print(f"  Synthesising '{tok}' ...", end=" ", flush=True)
            synthesise_token(tok, TTS_VOICE, tmp.name)

            audio, sr = sf.read(tmp.name)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SAMPLE_RATE:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            clips.append(audio.astype(np.float32))
            print("done")
    finally:
        for f in tmp_files:
            if os.path.exists(f):
                os.unlink(f)

    # Concatenate with silence gaps
    gap = np.zeros(int(SAMPLE_RATE * SILENCE_GAP_MS / 1000), dtype=np.float32)
    parts = []
    for i, clip in enumerate(clips):
        parts.append(clip)
        if i < len(clips) - 1:
            parts.append(gap)

    compound = np.concatenate(parts)

    # Peak normalise
    peak = np.max(np.abs(compound))
    if peak > 1e-6:
        compound = compound / peak * 0.95

    sf.write(str(out_path), compound, SAMPLE_RATE)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate compound Sanskrit number audio (0-99) via TTS."
    )
    parser.add_argument(
        "numbers",
        type=int,
        nargs="+",
        help="One or more integers in range 0-99.",
    )
    args = parser.parse_args()

    for n in args.numbers:
        if not 0 <= n <= 99:
            print(f"Error: {n} is out of range 0-99, skipping.")
            continue

        tokens = number_to_tokens(n)
        print(f"\n{n} → {' + '.join(tokens)}")
        out = generate_compound(n)
        print(f"  Saved: {out}")


if __name__ == "__main__":
    # Example: generate compound audio for numbers
    #   python scripts/demo_generate_number.py 57
    #   python scripts/demo_generate_number.py 0 15 42 99
    main()
