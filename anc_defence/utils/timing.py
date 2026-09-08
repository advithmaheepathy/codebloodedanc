"""Timing, latency and real-time-factor bookkeeping.

Per-stage timings are accumulated in preallocated lists and summarised as
mean/median/p95/p99/max. The real-time factor (RTF) is processing time divided by
the duration of audio processed: RTF < 1 means faster than real time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class TimingSummary:
    """Summary statistics for one stage, all times in milliseconds."""

    name: str
    count: int
    total_ms: float
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    audio_s: float
    rtf: float

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "stage": self.name,
            "count": self.count,
            "total_ms": round(self.total_ms, 3),
            "mean_ms": round(self.mean_ms, 4),
            "median_ms": round(self.median_ms, 4),
            "p95_ms": round(self.p95_ms, 4),
            "p99_ms": round(self.p99_ms, 4),
            "max_ms": round(self.max_ms, 4),
            "audio_s": round(self.audio_s, 3),
            "rtf": round(self.rtf, 5),
        }


class StageTimer:
    """Accumulates per-call wall times and the amount of audio each call covered."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._times_ms: list[float] = []
        self._audio_s: float = 0.0
        self._t0: Optional[float] = None

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop(self, audio_s: float = 0.0) -> float:
        if self._t0 is None:
            raise RuntimeError(f"StageTimer({self.name}).stop() called before start()")
        dt_ms = (time.perf_counter() - self._t0) * 1e3
        self._t0 = None
        self._times_ms.append(dt_ms)
        self._audio_s += audio_s
        return dt_ms

    def add(self, dt_ms: float, audio_s: float = 0.0) -> None:
        self._times_ms.append(dt_ms)
        self._audio_s += audio_s

    @property
    def samples_ms(self) -> np.ndarray:
        return np.asarray(self._times_ms, dtype=np.float64)

    @property
    def audio_s(self) -> float:
        return self._audio_s

    def summary(self) -> TimingSummary:
        t = self.samples_ms
        if t.size == 0:
            return TimingSummary(self.name, 0, 0, 0, 0, 0, 0, 0, self._audio_s, 0.0)
        total_ms = float(t.sum())
        rtf = (total_ms / 1e3) / self._audio_s if self._audio_s > 0 else float("nan")
        return TimingSummary(
            name=self.name,
            count=int(t.size),
            total_ms=total_ms,
            mean_ms=float(t.mean()),
            median_ms=float(np.median(t)),
            p95_ms=float(np.percentile(t, 95)),
            p99_ms=float(np.percentile(t, 99)),
            max_ms=float(t.max()),
            audio_s=self._audio_s,
            rtf=float(rtf),
        )


@dataclass
class TimingRegistry:
    """Named collection of stage timers."""

    stages: dict[str, StageTimer] = field(default_factory=dict)

    def timer(self, name: str) -> StageTimer:
        if name not in self.stages:
            self.stages[name] = StageTimer(name)
        return self.stages[name]

    def summaries(self) -> list[TimingSummary]:
        return [t.summary() for t in self.stages.values()]

    def as_records(self) -> list[dict[str, float | int | str]]:
        return [s.as_dict() for s in self.summaries()]

    def merge(self, other: "TimingRegistry") -> None:
        for name, timer in other.stages.items():
            mine = self.timer(name)
            for dt in timer.samples_ms:
                mine.add(float(dt))
            mine._audio_s += timer.audio_s


class Stopwatch:
    """Context manager: ``with Stopwatch(reg.timer('nlms'), audio_s=0.01):``"""

    __slots__ = ("_timer", "_audio_s")

    def __init__(self, timer: StageTimer, audio_s: float = 0.0) -> None:
        self._timer = timer
        self._audio_s = audio_s

    def __enter__(self) -> "Stopwatch":
        self._timer.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._timer.stop(self._audio_s)


def latency_breakdown(
    sample_rate: int,
    hop_size: int,
    input_blocks: int,
    output_blocks: int,
    algorithmic_frames: int,
    chunk_s: Optional[float] = None,
) -> dict[str, float]:
    """Algorithmic + buffering latency budget in milliseconds.

    This is the *theoretical* budget from block sizes and model lookahead. Any
    figure reported as measured must come from an actual loopback measurement.
    """
    hop_ms = 1e3 * hop_size / sample_rate
    out: dict[str, float] = {
        "input_buffer_ms": input_blocks * hop_ms,
        "output_buffer_ms": output_blocks * hop_ms,
        "model_algorithmic_ms": algorithmic_frames * hop_ms,
    }
    if chunk_s is not None:
        out["chunk_buffer_ms"] = chunk_s * 1e3
    out["total_ms"] = float(sum(out.values()))
    return out
