"""Echo/noise return loss enhancement for the adaptive stage.

ERLE = 10*log10( mean(d^2) / mean(e^2) ), computed per block to give a convergence
curve and summarised over the run. It measures how much energy the adaptive filter
removed, which is exactly what it is supposed to do, and says nothing about whether
the removed energy was noise or speech. That is why it is always reported alongside
the speech-distortion metrics.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

_EPS = 1e-20


def erle_curve(primary: np.ndarray, residual: np.ndarray, block_size: int = 480) -> np.ndarray:
    """Per-block ERLE in dB."""
    d = np.asarray(primary, dtype=np.float64)
    e = np.asarray(residual, dtype=np.float64)
    n = min(len(d), len(e)) // block_size
    if n == 0:
        return np.zeros(0)
    d = d[: n * block_size].reshape(n, block_size)
    e = e[: n * block_size].reshape(n, block_size)
    num = np.mean(d**2, axis=1) + _EPS
    den = np.mean(e**2, axis=1) + _EPS
    return 10.0 * np.log10(num / den)


def erle_summary(
    curve: np.ndarray,
    block_size: int = 480,
    sample_rate: int = 48000,
    noise_only_mask: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Summary statistics for an ERLE curve.

    ``noise_only_mask`` (per block) restricts the ``noise_only`` figure to blocks
    where the talker is silent, which is the honest place to quote noise reduction.
    """
    c = np.asarray(curve, dtype=np.float64)
    finite = c[np.isfinite(c)]
    out: dict[str, float] = {
        "erle_mean_db": float(np.mean(finite)) if finite.size else float("nan"),
        "erle_median_db": float(np.median(finite)) if finite.size else float("nan"),
        "erle_p95_db": float(np.percentile(finite, 95)) if finite.size else float("nan"),
        "erle_final_db": float(np.mean(finite[-50:])) if finite.size else float("nan"),
        "erle_max_db": float(np.max(finite)) if finite.size else float("nan"),
        "duration_s": float(len(c) * block_size / sample_rate),
    }
    if noise_only_mask is not None and len(noise_only_mask) >= len(c):
        mask = np.asarray(noise_only_mask[: len(c)], dtype=bool)
        sel = c[mask & np.isfinite(c)]
        out["erle_noise_only_db"] = float(np.mean(sel)) if sel.size else float("nan")
        out["noise_only_fraction"] = float(np.mean(mask))
    return out


def noise_reduction_db(
    before: np.ndarray, after: np.ndarray, noise_mask: np.ndarray
) -> float:
    """Level drop in dB measured only where the talker is silent.

    This is the number a listener would call "noise reduction": it excludes speech
    regions entirely, so speech attenuation cannot inflate it.
    """
    b = np.asarray(before, dtype=np.float64)
    a = np.asarray(after, dtype=np.float64)
    n = min(len(b), len(a), len(noise_mask))
    mask = np.asarray(noise_mask[:n], dtype=bool)
    if mask.sum() < 2:
        return float("nan")
    pb = float(np.mean(b[:n][mask] ** 2)) + _EPS
    pa = float(np.mean(a[:n][mask] ** 2)) + _EPS
    return float(10.0 * np.log10(pb / pa))
