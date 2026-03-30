"""
SankhyaVox – Feature Visualisation Tools.

Provides functions for plotting spectrograms, MFCC heatmaps, and
waveform overlays from either raw audio files or pre-extracted .npy
feature arrays.

Usage::

    from src.viz import plot_waveform, plot_spectrogram, plot_mfcc

    plot_waveform("data_processed/human/segments/S01/S01_001_01.wav")
    plot_spectrogram("data_processed/human/segments/S01/S01_001_01.wav")
    plot_mfcc("data_processed/human/features/S01/S01_001_01.npy")
"""

from pathlib import Path
from typing import Optional

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from src.config import (
    FEATURE_DIM,
    FRAME_LENGTH,
    FRAME_SHIFT,
    N_FFT,
    N_MELS,
    N_MFCC,
    RESULTS_DIR,
    SAMPLE_RATE,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def _load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load audio, convert to mono, return (audio, sr)."""
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), sr


def _save_or_show(fig: plt.Figure, save_path: Optional[str]) -> None:
    """Save to file if *save_path* given, otherwise display interactively."""
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path}")
    else:
        plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  WAVEFORM
# ═══════════════════════════════════════════════════════════════════════════════


def plot_waveform(
    audio_path: str,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 3),
) -> plt.Figure:
    """Plot the time-domain waveform of an audio file."""
    audio, sr = _load_audio(audio_path)
    t = np.arange(len(audio)) / sr

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(t, audio, linewidth=0.4, color="#1f77b4")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title or Path(audio_path).stem)
    ax.set_xlim(0, t[-1])
    fig.tight_layout()

    _save_or_show(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  SPECTROGRAM
# ═══════════════════════════════════════════════════════════════════════════════


def plot_spectrogram(
    audio_path: str,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 4),
) -> plt.Figure:
    """Plot a Mel spectrogram from a WAV file."""
    audio, sr = _load_audio(audio_path)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    S = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=N_FFT,
        hop_length=FRAME_SHIFT, win_length=FRAME_LENGTH,
        n_mels=N_MELS,
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=figsize)
    img = librosa.display.specshow(
        S_db, sr=sr, hop_length=FRAME_SHIFT,
        x_axis="time", y_axis="mel", ax=ax, cmap="magma",
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title or f"Mel Spectrogram — {Path(audio_path).stem}")
    fig.tight_layout()

    _save_or_show(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  MFCC HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════


def plot_mfcc(
    source: str,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show_deltas: bool = True,
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """
    Plot an MFCC heatmap.

    Parameters
    ----------
    source : str
        Path to a ``.npy`` feature file *or* a ``.wav`` audio file.
        If ``.npy``, expects shape ``(n_frames, feature_dim)``.
        If ``.wav``, MFCCs are computed on the fly.
    show_deltas : bool
        If True and features have 39 dims, show static / delta / delta-delta
        in three stacked panels.  Otherwise show a single heatmap.
    """
    path = Path(source)

    if path.suffix == ".npy":
        features = np.load(str(path))  # (n_frames, dim)
    else:
        audio, sr = _load_audio(str(path))
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        mfcc = librosa.feature.mfcc(
            y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT,
            hop_length=FRAME_SHIFT, win_length=FRAME_LENGTH, n_mels=N_MELS,
            window="hamming",
        )
        delta = librosa.feature.delta(mfcc, order=1)
        delta2 = librosa.feature.delta(mfcc, order=2)
        features = np.vstack([mfcc, delta, delta2]).T  # (n_frames, 39)

    label = title or path.stem
    n_frames, dim = features.shape

    if show_deltas and dim >= 3 * N_MFCC:
        static = features[:, :N_MFCC].T
        deltas = features[:, N_MFCC : 2 * N_MFCC].T
        delta2s = features[:, 2 * N_MFCC : 3 * N_MFCC].T

        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        for ax, data, sub in zip(axes, [static, deltas, delta2s], ["Static", "Delta", "Delta-Delta"]):
            img = ax.imshow(data, aspect="auto", origin="lower", cmap="coolwarm")
            ax.set_ylabel(sub)
            fig.colorbar(img, ax=ax, format="%.1f")
        axes[0].set_title(f"MFCC — {label}")
        axes[-1].set_xlabel("Frame")
    else:
        fig, ax = plt.subplots(figsize=figsize)
        img = ax.imshow(features.T, aspect="auto", origin="lower", cmap="coolwarm")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Coefficient")
        ax.set_title(f"MFCC — {label}")
        fig.colorbar(img, ax=ax, format="%.1f")

    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPARISON GRID
# ═══════════════════════════════════════════════════════════════════════════════


def plot_comparison(
    paths: list[str],
    kind: str = "mfcc",
    save_path: Optional[str] = None,
    figsize_per_row: tuple = (10, 2.5),
) -> plt.Figure:
    """
    Plot multiple samples in a vertical grid for side-by-side comparison.

    Parameters
    ----------
    paths : list of str
        Paths to ``.npy`` or ``.wav`` files.
    kind : ``"mfcc"``, ``"spectrogram"``, or ``"waveform"``
    """
    n = len(paths)
    fig, axes = plt.subplots(n, 1, figsize=(figsize_per_row[0], figsize_per_row[1] * n), squeeze=False)

    for i, p in enumerate(paths):
        ax = axes[i, 0]
        label = Path(p).stem

        if kind == "waveform":
            audio, sr = _load_audio(p)
            t = np.arange(len(audio)) / sr
            ax.plot(t, audio, linewidth=0.4)
            ax.set_ylabel(label, fontsize=8)
            ax.set_xlim(0, t[-1])

        elif kind == "spectrogram":
            audio, sr = _load_audio(p)
            if sr != SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            S = librosa.feature.melspectrogram(
                y=audio, sr=SAMPLE_RATE, n_fft=N_FFT,
                hop_length=FRAME_SHIFT, win_length=FRAME_LENGTH, n_mels=N_MELS,
            )
            S_db = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(S_db, sr=SAMPLE_RATE, hop_length=FRAME_SHIFT,
                                     x_axis="time", y_axis="mel", ax=ax, cmap="magma")
            ax.set_ylabel(label, fontsize=8)

        else:  # mfcc
            if p.endswith(".npy"):
                features = np.load(p)[:, :N_MFCC].T
            else:
                audio, sr = _load_audio(p)
                if sr != SAMPLE_RATE:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
                features = librosa.feature.mfcc(
                    y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT,
                    hop_length=FRAME_SHIFT, win_length=FRAME_LENGTH, n_mels=N_MELS,
                    window="hamming",
                )
            ax.imshow(features, aspect="auto", origin="lower", cmap="coolwarm")
            ax.set_ylabel(label, fontsize=8)

        if i < n - 1:
            ax.set_xticklabels([])

    axes[-1, 0].set_xlabel("Frame" if kind != "waveform" else "Time (s)")
    fig.suptitle(f"{kind.title()} Comparison", fontweight="bold")
    fig.tight_layout()

    _save_or_show(fig, save_path)
    return fig
