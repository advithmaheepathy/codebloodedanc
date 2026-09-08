"""System-level metrics: real-time factor, latency, dropouts and resource use.

The real-time factor is the figure that decides whether the pipeline could run on a
smaller machine. It is reported per stage and in total, and separately with the
neural stage pinned to a single CPU thread, which is the portability evidence for an
embedded target. No claim is made about hardware that was not measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from ..utils.timing import TimingRegistry, TimingSummary


@dataclass
class SystemMetrics:
    rtf_total: float = float("nan")
    rtf_by_stage: dict[str, float] = field(default_factory=dict)
    per_stage: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: dict[str, float] = field(default_factory=dict)
    measured_latency_ms: Optional[float] = None
    xruns: int = 0
    dropped_samples: int = 0
    underruns: int = 0
    ring_high_water: int = 0
    ring_capacity: int = 0
    peak_rss_mb: Optional[float] = None
    cpu_percent_mean: Optional[float] = None
    audio_seconds: float = 0.0
    wall_seconds: float = 0.0
    single_thread: bool = False
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rtf_total": round(self.rtf_total, 5) if np.isfinite(self.rtf_total) else None,
            "rtf_by_stage": {k: round(v, 5) for k, v in self.rtf_by_stage.items()},
            "per_stage": self.per_stage,
            "latency_ms": self.latency_ms,
            "measured_latency_ms": self.measured_latency_ms,
            "xruns": self.xruns,
            "dropped_samples": self.dropped_samples,
            "underruns": self.underruns,
            "ring_high_water": self.ring_high_water,
            "ring_capacity": self.ring_capacity,
            "ring_high_water_fraction": (
                round(self.ring_high_water / self.ring_capacity, 4) if self.ring_capacity else None
            ),
            "peak_rss_mb": self.peak_rss_mb,
            "cpu_percent_mean": self.cpu_percent_mean,
            "audio_seconds": round(self.audio_seconds, 3),
            "wall_seconds": round(self.wall_seconds, 3),
            "single_thread": self.single_thread,
            "warnings": self.warnings,
        }


def collect_system_metrics(
    timings: TimingRegistry,
    audio_seconds: float,
    wall_seconds: float,
    latency_ms: Optional[dict[str, float]] = None,
    ring_stats: Optional[dict[str, int]] = None,
    resource_records: Optional[Sequence[dict[str, float]]] = None,
    peak_rss_mb: Optional[float] = None,
    single_thread: bool = False,
    rtf_budget: float = 0.5,
) -> SystemMetrics:
    """Assemble the system metrics block for the report."""
    summaries: list[TimingSummary] = timings.summaries()
    m = SystemMetrics(
        per_stage=[s.as_dict() for s in summaries],
        latency_ms=dict(latency_ms or {}),
        audio_seconds=audio_seconds,
        wall_seconds=wall_seconds,
        single_thread=single_thread,
        peak_rss_mb=peak_rss_mb,
    )
    total_ms = sum(s.total_ms for s in summaries)
    m.rtf_total = (total_ms / 1e3) / audio_seconds if audio_seconds > 0 else float("nan")
    for s in summaries:
        if s.audio_s > 0:
            m.rtf_by_stage[s.name] = s.rtf
        elif audio_seconds > 0:
            m.rtf_by_stage[s.name] = (s.total_ms / 1e3) / audio_seconds

    if ring_stats:
        m.xruns = int(ring_stats.get("overruns", 0))
        m.dropped_samples = int(ring_stats.get("dropped_samples", 0))
        m.underruns = int(ring_stats.get("underruns", 0))
        m.ring_high_water = int(ring_stats.get("high_water", 0))
        m.ring_capacity = int(ring_stats.get("capacity", 0))

    if resource_records:
        cpu = [r["cpu_percent"] for r in resource_records if "cpu_percent" in r]
        if cpu:
            m.cpu_percent_mean = round(float(np.mean(cpu)), 1)
        rss = [r["rss_mb"] for r in resource_records if "rss_mb" in r]
        if rss and m.peak_rss_mb is None:
            m.peak_rss_mb = round(float(np.max(rss)), 1)

    if np.isfinite(m.rtf_total) and m.rtf_total > rtf_budget:
        m.warnings.append(
            f"total RTF {m.rtf_total:.3f} exceeds the {rtf_budget} budget: this configuration is "
            "not comfortably real time on this machine"
        )
    if m.xruns:
        m.warnings.append(f"{m.xruns} buffer overrun(s), {m.dropped_samples} samples dropped")
    if m.underruns:
        m.warnings.append(f"{m.underruns} output underrun(s)")
    return m


def memory_growth(resource_records: Sequence[dict[str, float]], min_samples: int = 20) -> dict[str, Any]:
    """Least-squares RSS trend, used to check for leaks during long runs."""
    rss = [r["rss_mb"] for r in resource_records if "rss_mb" in r]
    t = [r["t_s"] for r in resource_records if "t_s" in r]
    if len(rss) < min_samples:
        return {"samples": len(rss), "slope_mb_per_min": None, "verdict": "insufficient data"}
    slope = float(np.polyfit(t, rss, 1)[0]) * 60.0
    return {
        "samples": len(rss),
        "start_mb": round(rss[0], 1),
        "end_mb": round(rss[-1], 1),
        "peak_mb": round(max(rss), 1),
        "slope_mb_per_min": round(slope, 3),
        "verdict": "stable" if abs(slope) < 2.0 else "growing",
    }


def measure_loopback_latency(
    reference: np.ndarray, captured: np.ndarray, sample_rate: int
) -> Optional[float]:
    """Empirical latency from a known impulse in ``reference`` found in ``captured``.

    Returns milliseconds, or None if no clear peak was found. Cross-correlation is
    used rather than a naive threshold so the measurement survives a noisy capture.
    """
    ref = np.asarray(reference, dtype=np.float64)
    cap = np.asarray(captured, dtype=np.float64)
    if ref.size < 8 or cap.size < 8:
        return None
    n = int(2 ** np.ceil(np.log2(len(ref) + len(cap))))
    corr = np.fft.irfft(np.fft.rfft(cap, n) * np.conj(np.fft.rfft(ref, n)), n=n)
    peak = int(np.argmax(np.abs(corr[: len(cap)])))
    if np.abs(corr[peak]) < 4.0 * float(np.mean(np.abs(corr[: len(cap)]))):
        return None
    return 1000.0 * peak / sample_rate
