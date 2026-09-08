"""Minimal STFT/ISTFT used by the classical spectral baselines.

A Hann window with 75% overlap satisfies the constant-overlap-add condition, so
synthesis is exact when the spectrum is unmodified. Window length defaults to 32 ms
which is the usual choice for spectral subtraction and Wiener filtering; it is
deliberately not the neural model's 20 ms frame, because these baselines are
supposed to represent the traditional approach rather than a tuned variant of ours.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def stft(x: np.ndarray, n_fft: int = 1536, hop: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(spectrogram (frames, bins), window)``."""
    hop = hop or n_fft // 4
    window = np.hanning(n_fft + 1)[:n_fft].astype(np.float64)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n_frames = 1 + (len(x) - n_fft) // hop
    frames = np.lib.stride_tricks.as_strided(
        np.ascontiguousarray(x, dtype=np.float64),
        shape=(n_frames, n_fft),
        strides=(x.dtype.itemsize * hop, x.dtype.itemsize),
    ).copy()
    return np.fft.rfft(frames * window, axis=1), window


def istft(spec: np.ndarray, window: np.ndarray, hop: int | None = None, length: int | None = None) -> np.ndarray:
    """Weighted overlap-add inverse of :func:`stft`."""
    n_fft = len(window)
    hop = hop or n_fft // 4
    frames = np.fft.irfft(spec, n=n_fft, axis=1) * window
    n = (spec.shape[0] - 1) * hop + n_fft
    out = np.zeros(n, dtype=np.float64)
    norm = np.zeros(n, dtype=np.float64)
    w2 = window**2
    for i in range(spec.shape[0]):
        s = i * hop
        out[s : s + n_fft] += frames[i]
        norm[s : s + n_fft] += w2
    out /= np.maximum(norm, _EPS)
    if length is not None:
        out = out[:length] if len(out) >= length else np.pad(out, (0, length - len(out)))
    return out.astype(np.float32)


def initial_noise_psd(
    mag2: np.ndarray, n_frames: int, floor: float = _EPS
) -> np.ndarray:
    """Noise power spectrum estimated from the first ``n_frames`` frames."""
    k = max(1, min(n_frames, mag2.shape[0]))
    return np.maximum(np.mean(mag2[:k], axis=0), floor)


def track_noise_psd(
    mag2: np.ndarray,
    init_frames: int = 10,
    smoothing: float = 0.95,
    update_threshold: float = 3.0,
) -> np.ndarray:
    """Per-frame noise PSD estimate with a simple recursive minimum-style tracker.

    The estimate is only updated when the frame looks noise-dominated (its power is
    within ``update_threshold`` of the current noise estimate), which is the
    classical stationarity assumption these baselines are built on and the reason
    they struggle with sirens, passing vehicles and gunshots.
    """
    psd = initial_noise_psd(mag2, init_frames)
    out = np.empty_like(mag2)
    for i in range(mag2.shape[0]):
        frame = mag2[i]
        ratio = float(np.mean(frame) / (np.mean(psd) + _EPS))
        if ratio < update_threshold:
            psd = smoothing * psd + (1.0 - smoothing) * frame
        out[i] = psd
    return np.maximum(out, _EPS)
