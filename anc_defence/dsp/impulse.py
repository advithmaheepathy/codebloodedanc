"""Impulsive-event detection.

Gunshots, machine-gun bursts and explosions are short, very high crest-factor
events. They matter twice over:

* the adaptive filter sees a huge instantaneous gradient and can destabilise, so
  detection feeds the impulse guard in the NLMS stage;
* evaluation needs to score suppression *at* the events rather than averaged over
  the whole file, so detection also drives the event-local metrics.

Detection is deliberately simple: a frame is impulsive when its crest factor
(peak / RMS) exceeds a threshold and its peak rises well above the running
background level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

_EPS = 1e-12


def crest_factor_db(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    if rms <= 0.0:
        return 0.0
    return float(20.0 * np.log10((float(np.max(np.abs(x))) + _EPS) / rms))


@dataclass
class ImpulseDetector:
    """Streaming detector with a slowly tracked background level."""

    crest_factor_db_threshold: float = 12.0
    background_smoothing: float = 0.98
    peak_rise_db: float = 10.0
    background_rms: Optional[float] = None
    last_crest_db: float = 0.0
    last_peak_rise_db: float = 0.0

    def reset(self) -> None:
        self.background_rms = None

    def process_frame(self, frame: np.ndarray) -> bool:
        x = np.asarray(frame, dtype=np.float32)
        if x.size == 0:
            return False
        rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
        peak = float(np.max(np.abs(x)))
        if self.background_rms is None:
            self.background_rms = max(rms, _EPS)
        crest = self.last_crest_db = crest_factor_db(x)
        rise = 20.0 * np.log10((peak + _EPS) / (self.background_rms + _EPS))
        self.last_peak_rise_db = float(rise)
        impulsive = bool(crest > self.crest_factor_db_threshold and rise > self.peak_rise_db)
        # Only let quiet/steady frames update the background estimate.
        if not impulsive:
            a = self.background_smoothing
            self.background_rms = a * self.background_rms + (1.0 - a) * rms
        return impulsive


@dataclass
class ImpulseEvent:
    """One detected transient, in samples and seconds."""

    start: int
    end: int
    sample_rate: int
    peak: float

    @property
    def start_s(self) -> float:
        return self.start / self.sample_rate

    @property
    def end_s(self) -> float:
        return self.end / self.sample_rate

    @property
    def duration_s(self) -> float:
        return (self.end - self.start) / self.sample_rate

    def as_dict(self) -> dict[str, float]:
        return {
            "start_s": round(self.start_s, 4),
            "end_s": round(self.end_s, 4),
            "duration_s": round(self.duration_s, 4),
            "peak": round(float(self.peak), 6),
        }


@dataclass
class ImpulseScan:
    frame_flags: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    events: list[ImpulseEvent] = field(default_factory=list)
    frame_size: int = 480
    hop: int = 480

    def sample_mask(self, n_samples: int) -> np.ndarray:
        mask = np.zeros(n_samples, dtype=bool)
        for i, flag in enumerate(self.frame_flags):
            if flag:
                s = i * self.hop
                mask[s : min(s + self.frame_size, n_samples)] = True
        return mask


def scan_impulsive(
    x: np.ndarray,
    sample_rate: int,
    frame_size: int = 480,
    hop: Optional[int] = None,
    crest_db: float = 12.0,
    peak_rise_db: float = 10.0,
    merge_gap_s: float = 0.05,
    min_duration_s: float = 0.002,
) -> ImpulseScan:
    """Find impulsive frames and group them into events.

    Adjacent flagged frames closer than ``merge_gap_s`` are merged into one event,
    which keeps a machine-gun burst from being reported as dozens of events.
    """
    hop = hop or frame_size
    n_frames = max(0, 1 + (len(x) - frame_size) // hop) if len(x) >= frame_size else 0
    det = ImpulseDetector(crest_factor_db_threshold=crest_db, peak_rise_db=peak_rise_db)
    flags = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        flags[i] = det.process_frame(x[i * hop : i * hop + frame_size])

    events: list[ImpulseEvent] = []
    merge_gap = int(round(merge_gap_s * sample_rate))
    min_dur = int(round(min_duration_s * sample_rate))
    i = 0
    while i < n_frames:
        if not flags[i]:
            i += 1
            continue
        start = i * hop
        j = i
        while j + 1 < n_frames and (flags[j + 1] or (j + 2 < n_frames and flags[j + 2])):
            j += 1
        end = min(len(x), j * hop + frame_size)
        if end - start >= min_dur:
            if events and start - events[-1].end <= merge_gap:
                prev = events[-1]
                seg = x[prev.start : end]
                events[-1] = ImpulseEvent(prev.start, end, sample_rate, float(np.max(np.abs(seg))))
            else:
                seg = x[start:end]
                events.append(ImpulseEvent(start, end, sample_rate, float(np.max(np.abs(seg)))))
        i = j + 1

    return ImpulseScan(frame_flags=flags, events=events, frame_size=frame_size, hop=hop)
