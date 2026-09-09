"""Aggregation of per-example metrics by category, noise type, method and SNR.

A single averaged number hides the structure that matters here: impulsive noise
behaves nothing like a steady engine, and SNR improvement is bounded by how much
noise there was to remove in the first place. Everything is therefore reported
grouped, and the headline figure is quoted over an explicitly stated input-SNR range.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from ..config import MetricsCfg, TargetsCfg

NUMERIC_FIELDS = (
    "pesq",
    "stoi",
    "estoi",
    "si_sdr",
    "snr_direct",
    "output_snr_db",
    "residual_noise_snr_db",
    "snr_gain_db",
    "segmental_snr",
    "lsd",
    "speech_attenuation_db",
    "speech_distortion_db",
    "snr_improvement_db",
    "d_pesq",
    "d_stoi",
    "d_estoi",
    "d_segsnr_db",
    "d_lsd",
    "erle_mean_db",
    "erle_final_db",
    "erle_noise_only_db",
    "noise_reduction_db",
    "rms_reduction_db",
    "rms_reduction_pct",
    "transient_suppression_db_mean",
    "peak_suppression_db_mean",
    "speech_dropout_count",
    "n_events",
    "nlms_erle_mean_db",
    "nlms_erle_final_db",
    "nlms_adapting_fraction",
    "nlms_rollbacks",
    "agc_gain_mean_db",
    "agc_gain_range_db",
    "agc_held_fraction",
    "agc_limited_frames",
    "agc_clipped_samples",
    "output_level_dbfs",
    "level_error_db",
    "measured_input_snr_db",
    "diverged",
    "rtf",
)


@dataclass
class MetricRecord:
    """One measurement: one example, one method, one tap point."""

    example_id: str
    subset: str
    category: str
    noise_type: str
    snr_db: float
    method: str
    tap: str
    values: dict[str, float] = field(default_factory=dict)

    def flat(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "example_id": self.example_id,
            "subset": self.subset,
            "category": self.category,
            "noise_type": self.noise_type,
            "snr_db": self.snr_db,
            "method": self.method,
            "tap": self.tap,
        }
        out.update({k: v for k, v in self.values.items()})
        return out


def snr_bucket_label(snr_db: float, buckets: Sequence[tuple[float, float]]) -> str:
    for lo, hi in buckets:
        if lo <= snr_db < hi:
            return f"[{lo:g},{hi:g})"
    return "out_of_range"


def aggregate(
    records: Iterable[MetricRecord],
    group_by: Sequence[str] = ("method",),
    fields: Sequence[str] = NUMERIC_FIELDS,
    buckets: Optional[Sequence[tuple[float, float]]] = None,
) -> list[dict[str, Any]]:
    """Mean of each numeric field, grouped by the requested keys.

    ``group_by`` may include ``snr_bucket``, which is derived from ``snr_db``.
    Missing values are ignored rather than treated as zero.
    """
    groups: dict[tuple[Any, ...], list[MetricRecord]] = {}
    for rec in records:
        key_parts: list[Any] = []
        for key in group_by:
            if key == "snr_bucket":
                key_parts.append(snr_bucket_label(rec.snr_db, buckets or []))
            else:
                key_parts.append(getattr(rec, key, "?"))
        groups.setdefault(tuple(key_parts), []).append(rec)

    rows: list[dict[str, Any]] = []
    for key, items in groups.items():
        row: dict[str, Any] = dict(zip(group_by, key))
        row["n"] = len(items)
        for f in fields:
            vals = [
                r.values[f]
                for r in items
                if f in r.values and r.values[f] is not None and np.isfinite(r.values[f])
            ]
            if vals:
                row[f] = float(np.mean(vals))
        rows.append(row)
    rows.sort(key=lambda r: tuple(str(r.get(k, "")) for k in group_by))
    return rows


def filter_by_snr(
    records: Sequence[MetricRecord], snr_range: tuple[float, float]
) -> list[MetricRecord]:
    lo, hi = snr_range
    return [r for r in records if lo <= r.snr_db <= hi]


@dataclass
class TargetCheck:
    name: str
    value: float
    target: float
    comparison: str
    passed: bool
    scope: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def verdict(self) -> str:
        if not np.isfinite(self.value):
            return "N/A"
        return "PASS" if self.passed else "FAIL"


def check_targets(
    row: dict[str, Any], targets: TargetsCfg, scope: str = ""
) -> list[TargetCheck]:
    """Evaluate the mandated problem-statement targets against one aggregate row."""
    checks: list[TargetCheck] = []

    # The mandated "SNR > 15 dB" is an absolute output SNR, not the SI-SDR improvement.
    # Prefer the residual-noise output SNR (speech vs residual noise, the classical ANC
    # definition) where the noise-only track made it available; otherwise fall back to the
    # component output SNR from the clean-reference decomposition. Both are absolute
    # output-quality figures, which is what the target and its siblings (STOI, PESQ) are.
    snr_key = "residual_noise_snr_db" if np.isfinite(
        float(row.get("residual_noise_snr_db", float("nan")))
    ) else "output_snr_db"

    specs = (
        ("Output SNR", snr_key, targets.snr_db, ">"),
        ("STOI", "stoi", targets.stoi, ">"),
        ("PESQ (wideband)", "pesq", targets.pesq, ">"),
    )
    for label, key, target, comparison in specs:
        value = float(row.get(key, float("nan")))
        passed = bool(np.isfinite(value) and (value > target if comparison == ">" else value < target))
        checks.append(
            TargetCheck(
                name=label,
                value=value,
                target=target,
                comparison=comparison,
                passed=passed,
                scope=scope,
            )
        )
    return checks


def headline_summary(
    records: Sequence[MetricRecord],
    method: str,
    cfg: MetricsCfg,
    targets: TargetsCfg,
) -> dict[str, Any]:
    """Headline numbers for one method over the stated input-SNR range."""
    lo, hi = cfg.headline_snr_range
    selected = [r for r in records if r.method == method and lo <= r.snr_db <= hi]
    if not selected:
        return {"method": method, "n": 0, "snr_range_db": [lo, hi]}
    agg = aggregate(selected, group_by=("method",))[0]

    # Reports carry no pass/fail target table: measured figures are reported on their own
    # terms, with the definition each one uses stated alongside.
    #
    # The SI-SDR improvement is a different quantity from output SNR. It
    # is bounded by the noise that was present, so averaging it across near-clean inputs
    # measures the corpus rather than the system; it is therefore also summarised over the
    # noisy end of the range where it is meaningful.
    slo, shi = cfg.suppression_snr_range
    suppression = [r for r in selected if slo <= r.snr_db <= shi]
    snri_row = aggregate(suppression, group_by=("method",))[0] if suppression else agg

    per_category = aggregate(selected, group_by=("category",))
    return {
        "method": method,
        "n": len(selected),
        "snr_range_db": [lo, hi],
        "aggregate": agg,
        "output_snr_db": agg.get("output_snr_db"),
        "residual_noise_snr_db": agg.get("residual_noise_snr_db"),
        "snr_gain_db": agg.get("snr_gain_db"),
        "suppression_snr_range_db": [slo, shi],
        "suppression_n": len(suppression),
        "snr_improvement_db_full_range": agg.get("snr_improvement_db"),
        "snr_improvement_db_suppression_range": snri_row.get("snr_improvement_db"),
        "per_category": per_category,
    }


def write_csv(path: Path | str, rows: Sequence[dict[str, Any]]) -> Path:
    """Write aggregate or per-record rows as CSV with a stable column order."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return p
    columns: list[str] = []
    for row in rows:
        for k in row:
            if k not in columns:
                columns.append(k)
    with p.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: (round(v, 4) if isinstance(v, float) and np.isfinite(v) else v)
                    for k, v in row.items()
                }
            )
    return p


def format_table(
    rows: Sequence[dict[str, Any]], columns: Sequence[str], headers: Optional[Sequence[str]] = None
) -> list[list[str]]:
    """Render aggregate rows as a list-of-lists table for the PDF."""
    head = list(headers) if headers else [c.replace("_", " ") for c in columns]
    table = [head]
    for row in rows:
        line: list[str] = []
        for c in columns:
            v = row.get(c)
            if v is None:
                line.append("-")
            elif isinstance(v, float):
                line.append("-" if not np.isfinite(v) else f"{v:.3f}" if abs(v) < 10 else f"{v:.2f}")
            else:
                line.append(str(v))
        table.append(line)
    return table
