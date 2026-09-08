"""Channel and sensor augmentation for the evaluation mixtures.

These model the ways a real radio/headset chain damages the signal before any
enhancement gets to see it. Clipping in particular is called out by the problem
statement, and it matters for the adaptive stage: a clipped primary channel is a
non-linear function of the noise, so no linear filter can cancel it fully, and the
neural stage has to pick up the difference.

Every augmentation reports exactly what it did so the manifest can record it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import signal

from ..config import AugmentCfg

_EPS = 1e-20


@dataclass
class AugmentRecord:
    """What was applied to one mixture."""

    clipped: bool = False
    clip_threshold: float = float("nan")
    clipped_sample_fraction: float = 0.0
    level_gain_db: float = 0.0
    spectral_tilt_db_per_khz: float = 0.0
    mic_self_noise_dbfs: Optional[float] = None
    applied: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "clipped": self.clipped,
            "clip_threshold": round(float(self.clip_threshold), 4)
            if np.isfinite(self.clip_threshold)
            else None,
            "clipped_sample_fraction": round(self.clipped_sample_fraction, 6),
            "level_gain_db": round(self.level_gain_db, 2),
            "spectral_tilt_db_per_khz": round(self.spectral_tilt_db_per_khz, 3),
            "mic_self_noise_dbfs": self.mic_self_noise_dbfs,
            "applied": list(self.applied),
        }


def apply_clipping(x: np.ndarray, threshold: float) -> tuple[np.ndarray, float]:
    """Hard-clip at ``threshold`` (relative to the current peak).

    Returns the clipped signal and the fraction of samples that hit the limit.
    """
    peak = float(np.max(np.abs(x))) + _EPS
    limit = threshold * peak
    clipped_mask = np.abs(x) > limit
    fraction = float(np.mean(clipped_mask))
    return np.clip(x, -limit, limit).astype(np.float32), fraction


def apply_spectral_tilt(x: np.ndarray, db_per_khz: float, sample_rate: int) -> np.ndarray:
    """First-order spectral tilt, standing in for microphone/channel colouration."""
    if abs(db_per_khz) < 1e-6:
        return x
    n_fft = 2048
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    gain_db = db_per_khz * (freqs / 1000.0)
    gain = 10.0 ** (gain_db / 20.0)
    # Zero-phase FIR from the desired magnitude response.
    taps = np.fft.irfft(gain, n=n_fft)
    taps = np.roll(taps, n_fft // 2)[n_fft // 2 - 64 : n_fft // 2 + 64]
    taps *= np.hanning(len(taps))
    taps /= np.sum(np.abs(taps)) + _EPS
    return signal.fftconvolve(x, taps, mode="same").astype(np.float32)


def add_mic_self_noise(
    x: np.ndarray, level_dbfs: float, rng: np.random.Generator, dc_offset: float = 0.0
) -> np.ndarray:
    """Add a white noise floor and an optional DC offset, as a real capsule would."""
    amplitude = 10.0 ** (level_dbfs / 20.0)
    return (x + rng.standard_normal(len(x)).astype(np.float32) * amplitude + dc_offset).astype(
        np.float32
    )


def draw_level_gain_db(cfg: AugmentCfg, rng: np.random.Generator) -> float:
    """Draw the capture-chain level gain.

    This is drawn separately from the rest of the chain because it must be applied to
    the primary channel *and* to the clean target: a gain difference between a mixture
    and its target silently biases every scale-sensitive metric (segmental SNR, LSD,
    speech attenuation, PESQ). Only the genuinely destructive steps - spectral tilt,
    sensor noise and clipping - are applied to the primary alone.
    """
    if rng.random() >= cfg.level_prob:
        return 0.0
    return float(rng.uniform(*cfg.level_range_db))


def augment_channel(
    x: np.ndarray,
    cfg: AugmentCfg,
    rng: np.random.Generator,
    sample_rate: int = 48000,
    add_self_noise: bool = True,
    apply_level: bool = True,
) -> tuple[np.ndarray, AugmentRecord]:
    """Apply the configured augmentation chain to one channel.

    With ``apply_level=False`` the level draw is skipped; the caller is expected to
    have applied a matched gain to the mixture and its target already.
    """
    y = np.asarray(x, dtype=np.float32).copy()
    rec = AugmentRecord()

    if apply_level and rng.random() < cfg.level_prob:
        gain_db = float(rng.uniform(*cfg.level_range_db))
        y = (y * 10.0 ** (gain_db / 20.0)).astype(np.float32)
        rec.level_gain_db = gain_db
        rec.applied.append("level")

    if rng.random() < cfg.spectral_tilt_prob:
        tilt = float(rng.uniform(*cfg.spectral_tilt_db_per_khz))
        y = apply_spectral_tilt(y, tilt, sample_rate)
        rec.spectral_tilt_db_per_khz = tilt
        rec.applied.append("spectral_tilt")

    if add_self_noise and cfg.mic_self_noise_dbfs is not None:
        y = add_mic_self_noise(y, cfg.mic_self_noise_dbfs, rng)
        rec.mic_self_noise_dbfs = cfg.mic_self_noise_dbfs
        rec.applied.append("mic_self_noise")

    # Clipping is applied last: it is the final stage of a real capture chain.
    if rng.random() < cfg.clipping_prob:
        threshold = float(rng.uniform(cfg.clipping_threshold_min, cfg.clipping_threshold_max))
        y, fraction = apply_clipping(y, threshold)
        rec.clipped = True
        rec.clip_threshold = threshold
        rec.clipped_sample_fraction = fraction
        rec.applied.append("clipping")

    return y, rec
