"""
SankhyaVox — Audio Word Segmenter.

Splits raw audio (16kHz waveform) into individual word segments BEFORE
MFCC extraction. This gives 160x better temporal resolution than splitting
MFCCs and produces clean, isolated word chunks that the already-working
single-word classifiers can handle.

Pipeline:
    Raw audio → Preprocess → Segment into words → Extract MFCCs per word → Classify
"""

import numpy as np
import librosa
from src.config import SAMPLE_RATE


def segment_audio(audio, sr=SAMPLE_RATE, min_word_sec=0.15, min_silence_sec=0.08):
    """
    Segment raw audio waveform into individual word chunks.
    
    Uses a multi-stage approach:
    1. Compute short-time energy envelope
    2. Adaptive threshold based on audio statistics
    3. Find speech regions with minimum duration constraints
    4. Return list of audio arrays, one per word
    
    Parameters
    ----------
    audio : np.ndarray
        Raw audio waveform (mono, 16kHz)
    sr : int
        Sample rate
    min_word_sec : float
        Minimum word duration in seconds (rejects clicks/pops)
    min_silence_sec : float
        Minimum silence gap to split words
    
    Returns
    -------
    segments : list of np.ndarray
        Each element is a raw audio chunk for one word
    """
    if len(audio) < int(sr * 0.1):
        return [audio]
    
    # ── Stage 1: Short-time energy envelope ──
    # Use a 20ms window with 5ms hop for fine-grained energy
    frame_len = int(sr * 0.020)  # 320 samples
    hop_len = int(sr * 0.005)    # 80 samples
    
    # RMS energy per frame
    rms = librosa.feature.rms(y=audio, frame_length=frame_len, hop_length=hop_len)[0]
    
    # Smooth energy with a wider window to bridge micro-gaps within words
    smooth_size = max(3, int(0.030 * sr / hop_len))  # 30ms smoothing window
    kernel = np.ones(smooth_size) / smooth_size
    smoothed = np.convolve(rms, kernel, mode='same')
    
    # ── Stage 2: Adaptive threshold ──
    # Use the quietest 10% as noise floor, set threshold above it
    sorted_energy = np.sort(smoothed)
    noise_floor = np.mean(sorted_energy[:max(1, len(sorted_energy) // 10)])
    peak_energy = np.percentile(smoothed, 95)
    
    # Threshold = noise_floor + 15% of dynamic range
    # This is more robust than a fixed percentage of peak
    dynamic_range = peak_energy - noise_floor
    threshold = noise_floor + 0.15 * dynamic_range
    
    is_speech = smoothed > threshold
    
    # ── Stage 3: Find speech regions ──
    min_word_frames = int(min_word_sec * sr / hop_len)
    min_silence_frames = int(min_silence_sec * sr / hop_len)
    
    regions = []
    in_speech = False
    seg_start = 0
    silence_count = 0
    
    for i in range(len(is_speech)):
        if is_speech[i]:
            if not in_speech:
                seg_start = i
                in_speech = True
            silence_count = 0
        else:
            if in_speech:
                silence_count += 1
                if silence_count >= min_silence_frames:
                    seg_end = i - silence_count
                    if (seg_end - seg_start) >= min_word_frames:
                        regions.append((seg_start, seg_end))
                    in_speech = False
                    silence_count = 0
    
    # Last segment
    if in_speech:
        seg_end = len(is_speech)
        if (seg_end - seg_start) >= min_word_frames:
            regions.append((seg_start, seg_end))
    
    # If no regions found, return whole audio
    if not regions:
        return [audio]
    
    # ── Stage 4: Convert frame indices to audio samples ──
    # Add small padding around each segment (20ms)
    pad_samples = int(sr * 0.020)
    segments = []
    
    for frame_start, frame_end in regions:
        sample_start = max(0, frame_start * hop_len - pad_samples)
        sample_end = min(len(audio), frame_end * hop_len + pad_samples)
        chunk = audio[sample_start:sample_end]
        if len(chunk) > int(sr * 0.05):  # At least 50ms
            segments.append(chunk)
    
    if not segments:
        return [audio]
    
    # Cap at 3 words max (grammar limit)
    if len(segments) > 3:
        # Merge segments with smallest gaps
        while len(segments) > 3:
            # Find smallest gap and merge
            min_gap_idx = 0
            min_gap = float('inf')
            for i in range(len(regions) - 1):
                gap = regions[i+1][0] - regions[i][1]
                if gap < min_gap:
                    min_gap = gap
                    min_gap_idx = i
            # Merge in audio space
            merged_start = regions[min_gap_idx][0] * hop_len
            merged_end = regions[min_gap_idx + 1][1] * hop_len
            merged = audio[max(0, merged_start - pad_samples):min(len(audio), merged_end + pad_samples)]
            # Rebuild segments list
            new_regions = regions[:min_gap_idx] + [(regions[min_gap_idx][0], regions[min_gap_idx+1][1])] + regions[min_gap_idx+2:]
            regions = new_regions
            segments = []
            for fs, fe in regions:
                ss = max(0, fs * hop_len - pad_samples)
                se = min(len(audio), fe * hop_len + pad_samples)
                segments.append(audio[ss:se])
    
    return segments


def force_split_audio(audio, n_parts, sr=SAMPLE_RATE):
    """
    Force-split audio into n_parts at energy minima.
    Used as fallback when segment_audio detects only 1 segment
    but we want to try multi-word interpretations.
    
    Returns list of audio chunks.
    """
    if n_parts == 1:
        return [audio]
    
    # Energy envelope (fine-grained)
    frame_len = int(sr * 0.020)
    hop_len = int(sr * 0.005)
    rms = librosa.feature.rms(y=audio, frame_length=frame_len, hop_length=hop_len)[0]
    
    # Heavy smoothing to find broad valleys
    smooth_size = max(5, len(rms) // 8)
    kernel = np.ones(smooth_size) / smooth_size
    smoothed = np.convolve(rms, kernel, mode='same')
    
    n_rms = len(smoothed)
    splits = []
    
    if n_parts == 2:
        # Find deepest valley in middle 60%
        s, e = int(n_rms * 0.2), int(n_rms * 0.8)
        if e > s:
            valley = np.argmin(smoothed[s:e]) + s
            splits = [valley]
    
    elif n_parts == 3:
        s1, e1 = int(n_rms * 0.12), int(n_rms * 0.42)
        s2, e2 = int(n_rms * 0.58), int(n_rms * 0.88)
        if e1 > s1 and e2 > s2:
            v1 = np.argmin(smoothed[s1:e1]) + s1
            v2 = np.argmin(smoothed[s2:e2]) + s2
            splits = [v1, v2]
    
    if not splits:
        # Even spacing as last resort
        chunk_size = n_rms // n_parts
        splits = [chunk_size * i for i in range(1, n_parts)]
    
    # Convert to sample positions
    sample_splits = [s * hop_len for s in splits]
    
    chunks = []
    prev = 0
    for sp in sample_splits:
        sp = max(prev + int(sr * 0.05), min(len(audio) - int(sr * 0.05), sp))
        chunks.append(audio[prev:sp])
        prev = sp
    chunks.append(audio[prev:])
    
    return [c for c in chunks if len(c) > int(sr * 0.03)]
