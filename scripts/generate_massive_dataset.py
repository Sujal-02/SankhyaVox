"""
SankhyaVox — Massive Synthetic Data Generator.

Uses 3 TTS engines (gTTS, edge-tts, pyttsx3) with deep audio augmentations
to generate 1800+ synthetic training samples for the 13 Sanskrit digit tokens.

Usage:
    pip install gTTS edge-tts pyttsx3 imageio-ffmpeg
    python scripts/generate_massive_dataset.py
"""

import os
import sys
import numpy as np
import soundfile as sf
import librosa
import tempfile
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import VOCAB, SAMPLE_RATE
from scripts.extract_features import process_file

# ── Devanagari Text Map (for TTS engines) ──
DEVANAGARI_MAP = {
    "shunya":  "शून्य",
    "eka":     "एक",
    "dvi":     "द्वि",
    "tri":     "त्रि",
    "catur":   "चतुर्",
    "pancha":  "पञ्च",
    "shat":    "षट्",
    "sapta":   "सप्त",
    "ashta":   "अष्ट",
    "nava":    "नव",
    "dasha":   "दश",
    "vimsati": "विंशति",
    "shata":   "शत",
}

RAW_SYNTH_DIR = os.path.join("data", "synthetic_raw")
FEAT_SYNTH_DIR = os.path.join("data", "features")
os.makedirs(RAW_SYNTH_DIR, exist_ok=True)

# ── TTS Voice Configurations ──
GTTS_TLDS = ['co.in', 'com', 'co.uk', 'com.au', 'ca']   # 5 accent variants
EDGE_VOICES = ['hi-IN-SwaraNeural', 'hi-IN-MadhurNeural'] # 2 neural voices

# ── Augmentation Matrix ──
PITCH_STEPS  = [-4, -2, 0, 2, 4]     # 5 pitch variants (simulates different vocal tracts)
SPEED_RATES  = [0.85, 1.0, 1.15]      # 3 speed variants
NOISE_LEVELS = [0.0, 0.004]           # clean + light noise

generated_count = 0

def augment_and_save(audio, sr, token, variation_name):
    """Apply augmentation matrix and save each variant + its features."""
    global generated_count

    # Trim silence
    trimmed, _ = librosa.effects.trim(audio, top_db=30)
    if len(trimmed) > int(sr * 0.1):
        audio = trimmed

    for p in PITCH_STEPS:
        for s in SPEED_RATES:
            for n in NOISE_LEVELS:
                y = audio.copy()

                # Time stretch
                if s != 1.0:
                    y = librosa.effects.time_stretch(y=y, rate=s)

                # Pitch shift
                if p != 0:
                    y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=p)

                # Add noise
                if n > 0:
                    y = y + np.random.normal(0, n, len(y))

                # Save WAV
                tag = f"synth_{variation_name}_p{p}_s{int(s*100)}_n{int(n*10000)}"
                out_name = f"{tag}_{token}.wav"
                out_path = os.path.join(RAW_SYNTH_DIR, out_name)
                sf.write(out_path, y.astype(np.float32), sr)

                # Extract features
                feat_out_dir = os.path.join(FEAT_SYNTH_DIR, f"synth_{variation_name}")
                try:
                    process_file(out_path, feat_out_dir)
                    generated_count += 1
                except Exception:
                    pass


def generate_gtts(token, dev_text, ffmpeg_exe):
    """Generate audio using Google TTS with multiple accent TLDs."""
    from gtts import gTTS
    for tld in GTTS_TLDS:
        print(f"    gTTS ({tld})...", end=" ", flush=True)
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as fp:
            temp_mp3 = fp.name
        try:
            tts = gTTS(text=dev_text, lang='hi', tld=tld)
            tts.save(temp_mp3)

            temp_wav = temp_mp3.replace('.mp3', '.wav')
            subprocess.run([
                ffmpeg_exe, "-y", "-i", temp_mp3,
                "-ar", str(SAMPLE_RATE), "-ac", "1", temp_wav
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            audio, sr = sf.read(temp_wav)
            augment_and_save(audio, sr, token, f"gtts_{tld.replace('.', '')}")
            os.remove(temp_wav)
            print("✓")
        except Exception as e:
            print(f"SKIP ({e})")
        finally:
            if os.path.exists(temp_mp3):
                os.remove(temp_mp3)


def generate_edge_tts(token, dev_text, ffmpeg_exe):
    """Generate audio using Microsoft Edge Neural TTS."""
    for voice in EDGE_VOICES:
        print(f"    Edge ({voice})...", end=" ", flush=True)
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as fp:
            temp_mp3 = fp.name
        try:
            subprocess.run([
                "edge-tts", "--voice", voice, "--text", dev_text,
                "--write-media", temp_mp3
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            temp_wav = temp_mp3.replace('.mp3', '.wav')
            subprocess.run([
                ffmpeg_exe, "-y", "-i", temp_mp3,
                "-ar", str(SAMPLE_RATE), "-ac", "1", temp_wav
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            audio, sr = sf.read(temp_wav)
            augment_and_save(audio, sr, token, voice.replace('-', ''))
            os.remove(temp_wav)
            print("✓")
        except Exception as e:
            print(f"SKIP ({e})")
        finally:
            if os.path.exists(temp_mp3):
                os.remove(temp_mp3)


def generate_pyttsx3(token, dev_text):
    """Generate audio using pyttsx3 offline TTS (Windows SAPI5)."""
    print(f"    pyttsx3 (offline)...", end=" ", flush=True)
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)

        temp_wav = os.path.join(tempfile.gettempdir(), f"pyttsx3_{token}.wav")
        engine.save_to_file(dev_text, temp_wav)
        engine.runAndWait()

        if os.path.exists(temp_wav) and os.path.getsize(temp_wav) > 1000:
            audio, sr = sf.read(temp_wav)
            if sr != SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
                sr = SAMPLE_RATE
            augment_and_save(audio, sr, token, "pyttsx3")
            print("✓")
        else:
            print("SKIP (empty output)")

        if os.path.exists(temp_wav):
            os.remove(temp_wav)
    except Exception as e:
        print(f"SKIP ({e})")


def main():
    global generated_count
    print("=" * 60)
    print("  SankhyaVox — Massive Synthetic Data Generator")
    print("  Engines: gTTS + Edge-TTS + pyttsx3")
    print(f"  Augmentations: {len(PITCH_STEPS)} pitch × {len(SPEED_RATES)} speed × {len(NOISE_LEVELS)} noise = {len(PITCH_STEPS)*len(SPEED_RATES)*len(NOISE_LEVELS)} per voice")
    print("=" * 60)

    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    for token in VOCAB:
        if token not in DEVANAGARI_MAP:
            continue

        dev_text = DEVANAGARI_MAP[token]
        print(f"\n[{token}] ({dev_text})")

        generate_gtts(token, dev_text, ffmpeg_exe)
        generate_edge_tts(token, dev_text, ffmpeg_exe)
        generate_pyttsx3(token, dev_text)

    print("\n" + "=" * 60)
    print(f"  DONE! Generated {generated_count} synthetic feature files.")
    print(f"  Raw audio → {RAW_SYNTH_DIR}")
    print(f"  Features  → {FEAT_SYNTH_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
