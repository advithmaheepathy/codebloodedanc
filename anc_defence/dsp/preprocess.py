"""Per-channel pre-processing: DC removal, high-pass filtering and numeric guards.

The same filter is applied to the primary and reference channels, but each channel
gets its own ``Preprocessor`` instance because the filter state must not be shared.
State is carried between calls so that block-by-block processing produces exactly
the same output as processing the whole signal at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import signal

from ..config import PreprocessCfg
from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class GuardCounters:
    nonfinite_frames: int = 0
    nonfinite_samples: int = 0
    denormals_flushed: int = 0
    clipped_frames: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "nonfinite_frames": self.nonfinite_frames,
            "nonfinite_samples": self.nonfinite_samples,
            "denormals_flushed": self.denormals_flushed,
            "clipped_frames": self.clipped_frames,
        }

    def any(self) -> bool:
        return bool(self.nonfinite_frames or self.clipped_frames)


class Preprocessor:
    """Stateful DC-removal + high-pass chain for one channel."""

    def __init__(self, cfg: PreprocessCfg, sample_rate: int, name: str = "channel") -> None:
        self.cfg = cfg
        self.sample_rate = sample_rate
        self.name = name
        self.counters = GuardCounters()
        self._warned_nonfinite = False
        self._gain = 10.0 ** (cfg.reference_gain_db / 20.0) if name == "reference" else 1.0

        sos_sections: list[np.ndarray] = []
        if cfg.dc_removal:
            sos_sections.append(
                signal.butter(1, cfg.dc_cutoff_hz, btype="highpass", fs=sample_rate, output="sos")
            )
        if cfg.highpass.enabled:
            sos_sections.append(
                signal.butter(
                    cfg.highpass.order, cfg.highpass.cutoff_hz, btype="highpass", fs=sample_rate, output="sos"
                )
            )
        self.sos: Optional[np.ndarray] = np.vstack(sos_sections) if sos_sections else None
        self._zi = signal.sosfilt_zi(self.sos) * 0.0 if self.sos is not None else None

    # -------------------------------------------------------------------- api
    def reset(self) -> None:
        if self.sos is not None:
            self._zi = signal.sosfilt_zi(self.sos) * 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        """Filter one block (or a whole signal), carrying state across calls."""
        y = np.asarray(x, dtype=np.float32)
        y = self._guard(y)
        if self._gain != 1.0:
            y = y * self._gain
        if self.sos is not None:
            y, self._zi = signal.sosfilt(self.sos, y, zi=self._zi)
            y = y.astype(np.float32, copy=False)
        if self.cfg.guard_nonfinite:
            floor = self.cfg.denormal_floor
            small = np.abs(y) < floor
            if small.any():
                self.counters.denormals_flushed += int(small.sum())
                y = np.where(small, 0.0, y).astype(np.float32, copy=False)
        return y

    def _guard(self, y: np.ndarray) -> np.ndarray:
        if not self.cfg.guard_nonfinite:
            return y
        bad = ~np.isfinite(y)
        if bad.any():
            self.counters.nonfinite_frames += 1
            self.counters.nonfinite_samples += int(bad.sum())
            if not self._warned_nonfinite:
                log.warning(
                    "%s: non-finite samples detected (%d in this block); zeroing them. "
                    "This is logged once per channel.",
                    self.name,
                    int(bad.sum()),
                )
                self._warned_nonfinite = True
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        if np.any(np.abs(y) >= 0.999):
            self.counters.clipped_frames += 1
        return y


@dataclass
class GainCalibration:
    """Result of comparing primary and reference levels on noise-only audio."""

    primary_dbfs: float
    reference_dbfs: float
    suggested_reference_gain_db: float
    coherence: float = field(default=float("nan"))

    def as_dict(self) -> dict[str, float]:
        return {
            "primary_dbfs": round(self.primary_dbfs, 2),
            "reference_dbfs": round(self.reference_dbfs, 2),
            "suggested_reference_gain_db": round(self.suggested_reference_gain_db, 2),
            "broadband_coherence": round(self.coherence, 4),
        }


def calibrate_reference_gain(
    primary: np.ndarray, reference: np.ndarray, sample_rate: int = 48000
) -> GainCalibration:
    """Measure the primary/reference level difference on noise-only audio.

    The suggested gain brings the reference to the same level as the primary, which
    keeps the NLMS step size in a sensible range. Broadband coherence is reported
    as a sanity check: a very low value means the reference carries little
    information about the noise in the primary and cancellation will be poor.
    """
    from ..audio.io import dbfs, match_length

    p, r = match_length(primary, reference)
    p_db, r_db = dbfs(p), dbfs(r)
    coh = float("nan")
    if len(p) > 4096:
        nper = 2048
        f, cxy = signal.coherence(p, r, fs=sample_rate, nperseg=nper)
        band = (f >= 100.0) & (f <= 8000.0)
        if band.any():
            coh = float(np.mean(cxy[band]))
    return GainCalibration(
        primary_dbfs=p_db,
        reference_dbfs=r_db,
        suggested_reference_gain_db=float(p_db - r_db),
        coherence=coh,
    )
