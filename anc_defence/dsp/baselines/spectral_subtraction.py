"""Magnitude spectral subtraction (Boll, 1979) - single channel.

Estimate the noise magnitude spectrum, subtract a scaled version of it from the
noisy magnitude, keep the noisy phase, resynthesise::

    |S| = max( |Y| - alpha*|N| , beta*|Y| )

``alpha`` over-subtracts to compensate for the variance of the noise estimate;
``beta`` is a spectral floor that trades residual noise against musical noise.

Known weaknesses, which are the point of including it as a baseline: the noise
estimate assumes stationarity, so it lags behind sirens and passing vehicles; and
the half-wave rectification leaves isolated spectral peaks that sound like tonal
"musical noise".
"""

from __future__ import annotations

import numpy as np

from .stft import istft, stft, track_noise_psd

_EPS = 1e-12


def spectral_subtraction(
    noisy: np.ndarray,
    sample_rate: int = 48000,
    n_fft: int = 1536,
    hop: int | None = None,
    alpha: float = 2.0,
    beta: float = 0.02,
    noise_init_s: float = 0.25,
    adaptive_noise: bool = True,
) -> np.ndarray:
    """Enhance ``noisy`` by magnitude spectral subtraction.

    Args:
        alpha: over-subtraction factor.
        beta: spectral floor as a fraction of the noisy magnitude.
        noise_init_s: leading audio assumed to be noise only.
        adaptive_noise: keep updating the noise estimate on noise-dominated frames.
    """
    x = np.asarray(noisy, dtype=np.float64)
    hop = hop or n_fft // 4
    spec, window = stft(x, n_fft, hop)
    mag = np.abs(spec)
    mag2 = mag**2

    init_frames = max(1, int(round(noise_init_s * sample_rate / hop)))
    if adaptive_noise:
        noise_psd = track_noise_psd(mag2, init_frames=init_frames)
    else:
        noise_psd = np.broadcast_to(
            np.maximum(np.mean(mag2[:init_frames], axis=0), _EPS), mag2.shape
        )

    clean_mag = np.maximum(mag - alpha * np.sqrt(noise_psd), beta * mag)
    out_spec = spec * (clean_mag / (mag + _EPS))
    return istft(out_spec, window, hop, length=len(x))
