"""
SankhyaVox — Inference Module  (src/inference.py)

CHANGE FROM PREVIOUS VERSION:
  Old: extract_mfcc(audio) -> decoder.decode(mfcc)
  New: decoder.decode(audio, sr)   <- passes raw audio

The decoder now handles MFCC extraction PER SEGMENT internally,
so CMVN stats match training (per-word, not per-utterance).
"""

import os
import json
import subprocess
import tempfile
import joblib
import numpy as np
import warnings
warnings.filterwarnings('ignore')

_SR = 16_000


def _find_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return 'ffmpeg'


def _load_audio(path, sr=_SR):
    """Load any audio format -> (float32 array, sr). Uses ffmpeg fallback."""
    import librosa
    try:
        audio, _ = librosa.load(path, sr=sr, mono=True)
        if len(audio) > 0:
            return audio.astype('float32'), sr
    except Exception:
        pass
    # ffmpeg fallback (handles webm, ogg, mp3 from browser)
    ffmpeg  = _find_ffmpeg()
    out_wav = path + '._conv.wav'
    try:
        r = subprocess.run(
            [ffmpeg, '-y', '-i', path, '-ar', str(sr), '-ac', '1', '-f', 'wav', out_wav],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=15,
        )
        if r.returncode == 0:
            audio, _ = librosa.load(out_wav, sr=sr, mono=True)
            return audio.astype('float32'), sr
    except Exception:
        pass
    finally:
        if os.path.exists(out_wav):
            try: os.unlink(out_wav)
            except OSError: pass
    raise ValueError(f"Cannot load '{path}'. Install ffmpeg for webm/ogg support.")


def _preprocess(audio):
    """DC removal + peak normalisation (matches training)."""
    audio = audio - np.mean(audio)
    peak  = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio / peak * 0.95
    return audio.astype('float32')


def _confidence(score, n_tokens):
    """Map total HMM score to High/Medium/Low string."""
    if n_tokens == 0 or score <= -1e8:
        return 0.0, 'None'
    pf = score / max(n_tokens * 40, 1)      # per-frame LL estimate
    cf = max(0.0, min(1.0, (pf + 60.0) / 30.0))
    label = 'High' if cf >= 0.65 else ('Medium' if cf >= 0.35 else 'Low')
    return round(cf, 3), label


class _HMMSystem:
    def __init__(self, models):
        self.models = models


class SankhyaVoxInference:
    """
    Load:
        model = SankhyaVoxInference.load('models/best')
    Predict:
        result = model.predict_file('recording.wav')
        result = model.predict_audio(audio_np, sr=16000)
    Result keys: number, tokens, devanagari, score, confidence, success, [error]
    """

    def __init__(self, model_dir, config, hmm_system):
        self.model_dir  = model_dir
        self.config     = config
        self.label      = config.get('label', 'unknown')
        self.vocab      = config.get('vocab', [])
        self.devanagari = config.get('devanagari', {})
        self._hmm       = hmm_system
        from src.decoder  import SegmentFirstDecoder
        from src.grammar  import all_valid_sequences
        self._decoder = SegmentFirstDecoder(hmm_system, all_valid_sequences)
        from src.grammar  import COMPLETE_GRAMMAR
        self._grammar = COMPLETE_GRAMMAR

    @classmethod
    def load(cls, model_dir):
        model_dir   = os.path.abspath(model_dir)
        config_path = os.path.join(model_dir, 'config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f'config.json not found in {model_dir}')
        with open(config_path, encoding='utf-8') as f:
            config = json.load(f)
        vocab, models = config.get('vocab', []), {}
        for tok in vocab:
            pkl = os.path.join(model_dir, f'hmm_{tok}.pkl')
            if os.path.exists(pkl):
                try:
                    models[tok] = joblib.load(pkl)
                    print(f'  ✓ {tok}')
                except Exception as e:
                    print(f'  ✗ {tok}: {e}')
            else:
                print(f'  [WARN] missing hmm_{tok}.pkl')
        if not models:
            raise RuntimeError(f'No HMM models loaded from {model_dir}')
        print(f'  {len(models)}/{len(vocab)} models [{config.get("label","")}]')
        return cls(model_dir, config, _HMMSystem(models))

    def predict_audio(self, audio, sr=_SR, verbose=False):
        """Predict from raw numpy array."""
        audio  = np.asarray(audio, dtype='float32')
        if len(audio) < sr * 0.05:
            return self._err('Audio too short (< 50ms)')
        audio  = _preprocess(audio)

        # FIXED: pass raw audio to decoder, not pre-computed MFCC
        # Decoder extracts MFCC per-segment with per-segment CMVN.
        try:
            number, score = self._decoder.decode(audio, sr=sr, verbose=verbose)
        except Exception as e:
            return self._err(f'Decoder error: {e}')

        tokens      = list(self._grammar.get(number, []))
        deva        = '–'.join(self.devanagari.get(t, t) for t in tokens)
        cf, cf_str  = _confidence(score, len(tokens))
        return {
            'number':     int(number),
            'tokens':     tokens,
            'devanagari': deva,
            'score':      float(round(score, 2)),
            'confidence': cf_str,
            'model':      self.label,
            'success':    number >= 0,
        }

    def predict_file(self, path, verbose=False):
        """Predict from file path (wav, m4a, mp3, webm, ogg)."""
        try:
            audio, sr = _load_audio(path)
        except Exception as e:
            return self._err(f'Audio load failed: {e}')
        return self.predict_audio(audio, sr, verbose=verbose)

    def _err(self, reason):
        return {'number': -1, 'tokens': [], 'devanagari': '', 'score': 0.0,
                'confidence': 'None', 'model': self.label,
                'success': False, 'error': reason}