"""Dataset statistics report.

Summarises what was actually built: duration per category and noise type, the SNR
distribution, speaker counts, impulsive event density and RIR properties. Written as
JSON next to the manifest and rendered into the PDF report.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class DatasetStats:
    n_examples: int = 0
    total_minutes: float = 0.0
    minutes_by_category: dict[str, float] = field(default_factory=dict)
    minutes_by_noise_type: dict[str, float] = field(default_factory=dict)
    examples_by_subset: dict[str, int] = field(default_factory=dict)
    speakers: list[str] = field(default_factory=list)
    utterances_per_speaker: dict[str, int] = field(default_factory=dict)
    snr_histogram: dict[str, int] = field(default_factory=dict)
    snr_min: float = float("nan")
    snr_max: float = float("nan")
    snr_mean: float = float("nan")
    duration_s_values: list[float] = field(default_factory=list)
    snr_values: list[float] = field(default_factory=list)
    n_impulsive_events: int = 0
    events_per_minute: float = float("nan")
    event_peak_dbfs: list[float] = field(default_factory=list)
    event_crest_db: list[float] = field(default_factory=list)
    clipped_fraction: float = 0.0
    leakage_values_db: list[float] = field(default_factory=list)
    rir: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_examples": self.n_examples,
            "total_minutes": round(self.total_minutes, 3),
            "minutes_by_category": {k: round(v, 3) for k, v in self.minutes_by_category.items()},
            "minutes_by_noise_type": {
                k: round(v, 3) for k, v in sorted(self.minutes_by_noise_type.items())
            },
            "examples_by_subset": self.examples_by_subset,
            "n_speakers": len(self.speakers),
            "speakers": self.speakers,
            "utterances_per_speaker": self.utterances_per_speaker,
            "snr_histogram": self.snr_histogram,
            "snr_min_db": round(self.snr_min, 2),
            "snr_max_db": round(self.snr_max, 2),
            "snr_mean_db": round(self.snr_mean, 2),
            "n_impulsive_events": self.n_impulsive_events,
            "events_per_minute": round(self.events_per_minute, 2)
            if np.isfinite(self.events_per_minute)
            else None,
            "event_peak_dbfs_mean": round(float(np.mean(self.event_peak_dbfs)), 2)
            if self.event_peak_dbfs
            else None,
            "event_crest_db_mean": round(float(np.mean(self.event_crest_db)), 2)
            if self.event_crest_db
            else None,
            "clipped_example_fraction": round(self.clipped_fraction, 4),
            "leakage_values_db": sorted(set(self.leakage_values_db)),
            "rir": self.rir,
            "warnings": self.warnings,
        }


def compute_stats(manifest: dict[str, Any], snr_bin_width: float = 5.0) -> DatasetStats:
    """Summarise a dataset manifest."""
    entries: Sequence[dict[str, Any]] = manifest.get("examples", [])  # type: ignore[assignment]
    st = DatasetStats(n_examples=len(entries))
    st.rir = dict(manifest.get("rir", {}))  # type: ignore[arg-type]
    st.warnings = list(manifest.get("warnings", []))  # type: ignore[arg-type]

    cat_minutes: Counter[str] = Counter()
    type_minutes: Counter[str] = Counter()
    subset_counts: Counter[str] = Counter()
    speaker_counts: Counter[str] = Counter()
    snr_bins: Counter[str] = Counter()
    n_clipped = 0

    for e in entries:
        dur_min = float(e.get("duration_s", 0.0)) / 60.0
        st.total_minutes += dur_min
        st.duration_s_values.append(float(e.get("duration_s", 0.0)))
        cat_minutes[str(e.get("category", "unknown"))] += dur_min
        subset_counts[str(e.get("id", "?")).split("/")[0]] += 1
        speaker_counts[str(e.get("speaker", "?"))] += 1

        for c in e.get("components", []):
            type_minutes[str(c.get("noise_type", "unknown"))] += dur_min

        snr = float(e.get("snr_db", float("nan")))
        if np.isfinite(snr):
            st.snr_values.append(snr)
            lo = np.floor(snr / snr_bin_width) * snr_bin_width
            snr_bins[f"[{lo:.0f},{lo + snr_bin_width:.0f})"] += 1

        events = e.get("impulsive_events", [])
        st.n_impulsive_events += len(events)
        for ev in events:
            st.event_peak_dbfs.append(float(ev.get("peak_dbfs", float("nan"))))
            st.event_crest_db.append(float(ev.get("crest_factor_db", float("nan"))))

        if bool(e.get("augment", {}).get("primary", {}).get("clipped", False)):
            n_clipped += 1
        leak = e.get("leakage_db")
        if leak is not None:
            st.leakage_values_db.append(float(leak))

    st.minutes_by_category = dict(cat_minutes)
    st.minutes_by_noise_type = dict(type_minutes)
    st.examples_by_subset = dict(subset_counts)
    st.utterances_per_speaker = dict(sorted(speaker_counts.items()))
    st.speakers = sorted(speaker_counts)
    st.snr_histogram = dict(sorted(snr_bins.items(), key=lambda kv: float(kv[0].split(",")[0][1:])))
    if st.snr_values:
        st.snr_min = float(np.min(st.snr_values))
        st.snr_max = float(np.max(st.snr_values))
        st.snr_mean = float(np.mean(st.snr_values))
    if st.total_minutes > 0:
        st.events_per_minute = st.n_impulsive_events / st.total_minutes
    st.clipped_fraction = n_clipped / max(1, len(entries))
    return st


def write_stats(manifest_path: Path | str, out_path: Optional[Path] = None) -> tuple[Path, DatasetStats]:
    """Compute and write ``stats.json`` beside the manifest."""
    mp = Path(manifest_path)
    if mp.is_dir():
        mp = mp / "manifest.json"
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    stats = compute_stats(manifest)
    out = Path(out_path) if out_path else mp.parent / "stats.json"
    out.write_text(json.dumps(stats.as_dict(), indent=2), encoding="utf-8")
    log.info(
        "dataset stats: %d examples, %.1f minutes, categories %s",
        stats.n_examples,
        stats.total_minutes,
        {k: round(v, 1) for k, v in stats.minutes_by_category.items()},
    )
    return out, stats


def format_stats_table(stats: DatasetStats) -> list[tuple[str, str]]:
    """Key/value rows for the PDF report."""
    rows = [
        ("Examples", str(stats.n_examples)),
        ("Total duration", f"{stats.total_minutes:.2f} min"),
        ("Speakers", str(len(stats.speakers))),
        (
            "SNR range",
            f"{stats.snr_min:.1f} to {stats.snr_max:.1f} dB (mean {stats.snr_mean:.1f})",
        ),
        ("Impulsive events", str(stats.n_impulsive_events)),
        ("Clipped examples", f"{100.0 * stats.clipped_fraction:.0f}%"),
    ]
    for cat, minutes in sorted(stats.minutes_by_category.items()):
        rows.append((f"  {cat}", f"{minutes:.2f} min"))
    if stats.rir:
        rows.append(
            (
                "RIR significant length",
                f"{stats.rir.get('significant_ms_min', '?')}-{stats.rir.get('significant_ms_max', '?')} ms",
            )
        )
        rows.append(
            (
                "RIR RT60 estimate",
                f"{stats.rir.get('t60_estimate_s_min', '?')}-{stats.rir.get('t60_estimate_s_max', '?')} s",
            )
        )
    return rows
