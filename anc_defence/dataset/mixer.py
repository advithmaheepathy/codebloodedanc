"""Mixing: SNR control, impulsive event injection and two-channel construction.

Two-channel construction follows the runtime signal model, with the crucial
difference from the naive setup spelled out:

    primary   = speech + sum_i  ( noise_i * rir_i )        <- what the talker's mic hears
    reference = sum_i  noise_i                             <- the DRY noise
    target    = speech                                     <- the clean reference for metrics

The reference is *not* the waveform that was added to the primary. It is the same
source before it travelled through the room, so the adaptive filter has to identify
the acoustic path. Optionally a small amount of speech is leaked into the reference,
which is the real-world failure mode that makes NLMS cancel speech.

SNR is measured on active speech only (frames within 40 dB of the loudest frame), so
leading and trailing silence cannot skew the mixture level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..audio.io import tile_to
from ..config import ImpulsiveCfg
from ..utils.logging import get_logger

log = get_logger(__name__)

_EPS = 1e-20


def active_rms(x: np.ndarray, sample_rate: int = 48000, frame_ms: float = 20.0, range_db: float = 40.0) -> float:
    """RMS over active frames only, an approximation of ITU-T P.56 active level."""
    x = np.asarray(x, dtype=np.float64)
    frame = max(16, int(round(sample_rate * frame_ms / 1000.0)))
    n = len(x) // frame
    if n == 0:
        return float(np.sqrt(np.mean(x**2))) if x.size else 0.0
    frames = x[: n * frame].reshape(n, frame)
    power = np.mean(frames**2, axis=1)
    peak = power.max()
    if peak <= 0:
        return 0.0
    keep = power > peak * 10.0 ** (-range_db / 10.0)
    return float(np.sqrt(np.mean(power[keep])))


def scale_for_snr(
    speech: np.ndarray, noise: np.ndarray, snr_db: float, sample_rate: int = 48000
) -> tuple[np.ndarray, float]:
    """Scale ``noise`` so that mixing it with ``speech`` yields ``snr_db``."""
    s_rms = active_rms(speech, sample_rate)
    n_rms = float(np.sqrt(np.mean(np.square(np.asarray(noise, dtype=np.float64))))) + _EPS
    if s_rms <= 0:
        return np.zeros_like(noise), 0.0
    target_n_rms = s_rms / (10.0 ** (snr_db / 20.0))
    gain = target_n_rms / n_rms
    return (np.asarray(noise, dtype=np.float32) * gain).astype(np.float32), float(gain)


@dataclass
class ImpulsiveEventRecord:
    onset_s: float
    duration_s: float
    peak_dbfs: float
    source: str
    crest_factor_db: float

    def as_dict(self) -> dict[str, object]:
        return {
            "onset_s": round(self.onset_s, 4),
            "duration_s": round(self.duration_s, 4),
            "peak_dbfs": round(self.peak_dbfs, 2),
            "source": self.source,
            "crest_factor_db": round(self.crest_factor_db, 2),
        }


def inject_impulsive_events(
    length: int,
    event_pool: Sequence[np.ndarray],
    cfg: ImpulsiveCfg,
    rng: np.random.Generator,
    sample_rate: int = 48000,
    source_names: Optional[Sequence[str]] = None,
) -> tuple[np.ndarray, list[ImpulsiveEventRecord]]:
    """Build a sparse impulsive track: discrete events at random inter-arrival times.

    Peak levels are set per event and the waveform is **not** peak-normalised
    afterwards, so the high crest factor that makes these events hard survives into
    the mixture. That is the whole point of treating them separately from
    continuous noise.
    """
    track = np.zeros(length, dtype=np.float32)
    records: list[ImpulsiveEventRecord] = []
    if not event_pool or length <= 0:
        return track, records

    duration_s = length / sample_rate
    rate = float(rng.uniform(cfg.events_per_minute_min, cfg.events_per_minute_max))
    n_events = max(1, int(round(rate * duration_s / 60.0)))
    min_gap = int(cfg.min_gap_s * sample_rate)

    onset = int(rng.integers(0, max(1, int(0.2 * sample_rate))))
    for _ in range(n_events):
        if onset >= length - 64:
            break
        idx = int(rng.integers(0, len(event_pool)))
        event = np.asarray(event_pool[idx], dtype=np.float32)
        name = source_names[idx] if source_names is not None else f"event_{idx}"

        # Trim silence so the onset lands where we put it.
        nz = np.flatnonzero(np.abs(event) > 0.02 * (np.max(np.abs(event)) + _EPS))
        if nz.size:
            event = event[nz[0] : min(len(event), nz[-1] + 1)]
        if event.size < 32:
            continue

        peak_dbfs = float(rng.uniform(cfg.peak_level_dbfs_min, cfg.peak_level_dbfs_max))
        target_peak = 10.0 ** (peak_dbfs / 20.0)
        current_peak = float(np.max(np.abs(event))) + _EPS
        scaled = event * (target_peak / current_peak)

        end = min(length, onset + len(scaled))
        seg = scaled[: end - onset]
        track[onset:end] += seg

        rms = float(np.sqrt(np.mean(np.square(seg.astype(np.float64))))) + _EPS
        records.append(
            ImpulsiveEventRecord(
                onset_s=onset / sample_rate,
                duration_s=len(seg) / sample_rate,
                peak_dbfs=peak_dbfs,
                source=name,
                crest_factor_db=float(20.0 * np.log10(float(np.max(np.abs(seg))) / rms)),
            )
        )

        gap = int(rng.exponential(max(1.0, sample_rate * 60.0 / max(rate, 1e-6))))
        onset = onset + len(seg) + max(min_gap, gap)

    return track, records


@dataclass
class NoiseComponent:
    """One noise source that went into a mixture."""

    path: str
    category: str
    noise_type: str
    rir_index: Optional[int]
    gain: float
    is_impulsive_track: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "category": self.category,
            "noise_type": self.noise_type,
            "rir_index": self.rir_index,
            "gain": round(float(self.gain), 6),
            "is_impulsive_track": self.is_impulsive_track,
        }


@dataclass
class Mixture:
    """A complete two-channel example plus every parameter that produced it."""

    primary: np.ndarray
    reference: np.ndarray
    target: np.ndarray
    noise_at_primary: np.ndarray
    sample_rate: int
    snr_db: float
    speech_path: str
    speaker: str
    category: str
    components: list[NoiseComponent] = field(default_factory=list)
    impulsive_events: list[ImpulsiveEventRecord] = field(default_factory=list)
    leakage_db: Optional[float] = None
    rir_significant_ms: float = float("nan")
    seed: int = 0
    augment: dict[str, object] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return len(self.primary) / self.sample_rate

    def manifest_entry(self, name: str, paths: dict[str, str]) -> dict[str, object]:
        return {
            "id": name,
            "duration_s": round(self.duration_s, 3),
            "sample_rate": self.sample_rate,
            "snr_db": round(float(self.snr_db), 3),
            "category": self.category,
            "speech_path": self.speech_path,
            "speaker": self.speaker,
            "leakage_db": self.leakage_db,
            "rir_significant_ms": self.rir_significant_ms,
            "seed": self.seed,
            "components": [c.as_dict() for c in self.components],
            "impulsive_events": [e.as_dict() for e in self.impulsive_events],
            "n_impulsive_events": len(self.impulsive_events),
            "augment": self.augment,
            "files": paths,
        }


def leak_speech_into_reference(
    reference: np.ndarray, speech: np.ndarray, leakage_db: float
) -> np.ndarray:
    """Add attenuated speech to the reference channel.

    ``leakage_db`` is relative to the speech level in the primary. This is the
    mechanism behind the classic failure: the adaptive filter sees speech in the
    reference and learns to subtract speech from the primary.
    """
    gain = 10.0 ** (leakage_db / 20.0)
    n = min(len(reference), len(speech))
    out = np.asarray(reference, dtype=np.float32).copy()
    out[:n] += (np.asarray(speech[:n], dtype=np.float32) * gain).astype(np.float32)
    return out


def prepare_noise(
    noise: np.ndarray, length: int, rng: np.random.Generator, allow_tile: bool = True
) -> np.ndarray:
    """Crop or loop a noise clip to the required length from a random offset."""
    x = np.asarray(noise, dtype=np.float32)
    if len(x) >= length:
        start = int(rng.integers(0, len(x) - length + 1))
        return x[start : start + length].copy()
    if not allow_tile:
        out = np.zeros(length, dtype=np.float32)
        out[: len(x)] = x
        return out
    return tile_to(x, length, rng)
