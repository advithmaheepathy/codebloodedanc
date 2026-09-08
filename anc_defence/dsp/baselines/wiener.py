"""Decision-directed Wiener filtering - single channel.

A posteriori SNR ``gamma = |Y|^2 / N``; a priori SNR ``xi`` estimated with the
decision-directed rule of Ephraim and Malah::

    xi[k,t] = a * G[k,t-1]^2 * gamma[k,t-1] + (1-a) * max(gamma[k,t]-1, 0)
    G[k,t]  = xi / (1 + xi)

The gain is floored so the filter attenuates rather than gates, which is what keeps
Wiener filtering less prone to musical noise than plain spectral subtraction. It
still depends on a slowly-updated noise PSD, so it degrades in exactly the
non-stationary conditions this project targets.
"""

from __future__ import annotations

import numpy as np

from .stft import istft, stft, track_noise_psd

_EPS = 1e-12


def wiener_filter(
    noisy: np.ndarray,
    sample_rate: int = 48000,
    n_fft: int = 1536,
    hop: int | None = None,
    dd_alpha: float = 0.98,
    gain_floor_db: float = -25.0,
    noise_init_s: float = 0.25,
    adaptive_noise: bool = True,
) -> np.ndarray:
    """Enhance ``noisy`` with a decision-directed Wiener gain."""
    x = np.asarray(noisy, dtype=np.float64)
    hop = hop or n_fft // 4
    spec, window = stft(x, n_fft, hop)
    mag2 = np.abs(spec) ** 2

    init_frames = max(1, int(round(noise_init_s * sample_rate / hop)))
    if adaptive_noise:
        noise_psd = track_noise_psd(mag2, init_frames=init_frames)
    else:
        noise_psd = np.broadcast_to(
            np.maximum(np.mean(mag2[:init_frames], axis=0), _EPS), mag2.shape
        )

    gain_floor = 10.0 ** (gain_floor_db / 20.0)
    n_frames, n_bins = mag2.shape
    gains = np.empty((n_frames, n_bins), dtype=np.float64)
    g_prev = np.full(n_bins, gain_floor)
    gamma_prev = np.ones(n_bins)

    for t in range(n_frames):
        gamma = mag2[t] / (noise_psd[t] + _EPS)
        xi = dd_alpha * (g_prev**2) * gamma_prev + (1.0 - dd_alpha) * np.maximum(gamma - 1.0, 0.0)
        g = xi / (1.0 + xi)
        g = np.maximum(g, gain_floor)
        gains[t] = g
        g_prev, gamma_prev = g, gamma

    return istft(spec * gains, window, hop, length=len(x))
