"""
SankhyaVox – TTS Data Generator.

Generates synthetic audio files for each token and speaker using
Microsoft Edge's TTS API with randomised rate/pitch for augmentation.

Output naming follows: <SpeakerId>_<numericId>_<rep>.wav
Files are written into per-speaker subdirectories.
"""

import asyncio
import os
import random
from typing import Dict, Optional

import edge_tts

from src.config import (
    TOKEN_TO_NUMERIC_ID,
    TTS_PITCHES,
    TTS_RATES,
    TTS_REPS,
    TTS_VOICES,
)

# TTS speaks the romanised token text, keyed by numeric ID
_TTS_TEXT: Dict[str, str] = {nid: tok for tok, nid in TOKEN_TO_NUMERIC_ID.items()}


async def _generate(
    output_dir: str,
    voices: Optional[Dict[str, str]] = None,
    reps: int = TTS_REPS,
) -> int:
    """Generate TTS audio files.  Returns the number of files created."""
    voices = voices or TTS_VOICES
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    for speaker_id, voice in voices.items():
        speaker_dir = os.path.join(output_dir, speaker_id)
        os.makedirs(speaker_dir, exist_ok=True)

        for numeric_id, text in _TTS_TEXT.items():
            for rep in range(1, reps + 1):
                rate = random.choice(TTS_RATES)
                pitch = random.choice(TTS_PITCHES)
                filename = f"{speaker_id}_{numeric_id}_{rep:02d}.wav"
                filepath = os.path.join(speaker_dir, filename)

                communicate = edge_tts.Communicate(
                    text=text, voice=voice, rate=rate, pitch=pitch
                )
                await communicate.save(filepath)
                count += 1
                print(f"  Saved: {filename}")

    print(f"\nDone. Generated {count} TTS files -> {output_dir}")
    return count


def generate(
    output_dir: str,
    voices: Optional[Dict[str, str]] = None,
    reps: int = TTS_REPS,
) -> int:
    """Synchronous wrapper for TTS generation."""
    return asyncio.run(_generate(output_dir, voices, reps))
