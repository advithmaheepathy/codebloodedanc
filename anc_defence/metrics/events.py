"""Event-local metrics for impulsive noise.

PESQ and STOI average over a whole file, so a gunshot that punches a hole in the
speech barely moves them. These metrics score the transients directly:

``transient_suppression_db``
    Peak level drop at each detected event, input versus output. Positive is good.

``speech_dropout``
    Around each event, compare the enhanced speech energy with the clean speech
    energy in the same window. If the enhancer gates the speech along with the
    transient, this shows up as a large negative value. A dropout is counted when
    the loss exceeds ``dropout_threshold_db`` in a window where the clean signal
    actually has speech.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..dsp.impulse import ImpulseEvent, scan_impulsive

_EPS = 1e-20


@dataclass
class EventMetrics:
    n_events: int = 0
    events_per_minute: float = float("nan")
    transient_suppression_db_mean: float = float("nan")
    transient_suppression_db_min: float = float("nan")
    peak_suppression_db_mean: float = float("nan")
    speech_dropout_count: int = 0
    speech_dropout_worst_db: float = float("nan")
    per_event: list[dict[str, float]] = field(default_factory=list)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n_events": self.n_events,
            "events_per_minute": self.events_per_minute,
            "transient_suppression_db_mean": self.transient_suppression_db_mean,
            "transient_suppression_db_min": self.transient_suppression_db_min,
            "peak_suppression_db_mean": self.peak_suppression_db_mean,
            "speech_dropout_count": self.speech_dropout_count,
            "speech_dropout_worst_db": self.speech_dropout_worst_db,
        }


def _band_power(x: np.ndarray) -> float:
    return float(np.mean(np.square(x, dtype=np.float64))) + _EPS


def event_local_metrics(
    primary: np.ndarray,
    output: np.ndarray,
    sample_rate: int = 48000,
    clean: Optional[np.ndarray] = None,
    events: Optional[list[ImpulseEvent]] = None,
    crest_db: float = 12.0,
    guard_ms: float = 200.0,
    dropout_threshold_db: float = 6.0,
) -> EventMetrics:
    """Score suppression at impulsive events and check for speech dropouts.

    Events are detected on the *input* so the same regions are scored regardless of
    how much the enhancer removed.
    """
    d = np.asarray(primary, dtype=np.float64)
    y = np.asarray(output, dtype=np.float64)
    n = min(len(d), len(y))
    d, y = d[:n], y[:n]
    c = np.asarray(clean, dtype=np.float64)[:n] if clean is not None else None

    if events is None:
        events = scan_impulsive(d, sample_rate, crest_db=crest_db).events

    m = EventMetrics(n_events=len(events))
    if n == 0:
        return m
    m.events_per_minute = len(events) / (n / sample_rate / 60.0)
    if not events:
        return m

    guard = int(round(guard_ms * sample_rate / 1000.0))
    suppression: list[float] = []
    peak_suppression: list[float] = []
    dropouts: list[float] = []

    for ev in events:
        s, e = max(0, ev.start), min(n, ev.end)
        if e <= s:
            continue
        p_in, p_out = _band_power(d[s:e]), _band_power(y[s:e])
        supp = 10.0 * np.log10(p_in / p_out)
        peak_in = float(np.max(np.abs(d[s:e]))) + _EPS
        peak_out = float(np.max(np.abs(y[s:e]))) + _EPS
        peak_supp = 20.0 * np.log10(peak_in / peak_out)
        suppression.append(supp)
        peak_suppression.append(peak_supp)

        record = {
            "start_s": round(s / sample_rate, 4),
            "duration_s": round((e - s) / sample_rate, 4),
            "suppression_db": round(float(supp), 2),
            "peak_suppression_db": round(float(peak_supp), 2),
        }

        if c is not None:
            # Speech either side of the event: did the enhancer gate it away too?
            w0, w1 = max(0, s - guard), min(n, e + guard)
            clean_win = c[w0:w1]
            if _band_power(clean_win) > 1e-8:
                loss_db = 10.0 * np.log10(_band_power(clean_win) / _band_power(y[w0:w1]))
                record["speech_loss_db"] = round(float(loss_db), 2)
                if loss_db > dropout_threshold_db:
                    dropouts.append(float(loss_db))
        m.per_event.append(record)

    if suppression:
        m.transient_suppression_db_mean = float(np.mean(suppression))
        m.transient_suppression_db_min = float(np.min(suppression))
        m.peak_suppression_db_mean = float(np.mean(peak_suppression))
    m.speech_dropout_count = len(dropouts)
    m.speech_dropout_worst_db = float(np.max(dropouts)) if dropouts else float("nan")
    return m
