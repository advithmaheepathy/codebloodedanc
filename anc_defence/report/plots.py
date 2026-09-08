"""Figures for the PDF report.

All plots are rendered through the Agg backend so nothing needs a display, and every
figure is written to the session's ``figures/`` directory as a PNG that the PDF then
embeds. Waveform and spectrogram panels for a given run share identical scales, so
stages can be compared by eye without being misled by autoscaling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..utils.logging import get_logger  # noqa: E402

log = get_logger(__name__)

_EPS = 1e-12
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf", "#7f7f7f"]


def _save(fig: plt.Figure, path: Path, dpi: int = 110) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def waveforms(
    taps: Mapping[str, np.ndarray], sample_rate: int, path: Path, dpi: int = 110, title: str = ""
) -> Path:
    """Stacked waveforms, one panel per tap point, identical y limits."""
    names = list(taps)
    n = len(names)
    peak = max((float(np.max(np.abs(v))) for v in taps.values() if len(v)), default=1.0) or 1.0
    fig, axes = plt.subplots(n, 1, figsize=(9, 1.5 * n + 0.6), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, name, colour in zip(axes, names, PALETTE * 4):
        x = np.asarray(taps[name], dtype=np.float32)
        t = np.arange(len(x)) / sample_rate
        ax.plot(t, x, linewidth=0.4, color=colour)
        ax.set_ylim(-1.05 * peak, 1.05 * peak)
        ax.set_ylabel(name.replace("_", "\n"), fontsize=7)
        ax.grid(alpha=0.25, linewidth=0.4)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("time (s)", fontsize=8)
    if title:
        axes[0].set_title(f"{title} (shared amplitude scale)", fontsize=9)
    return _save(fig, path, dpi)


def spectrograms(
    taps: Mapping[str, np.ndarray],
    sample_rate: int,
    path: Path,
    dpi: int = 110,
    n_fft: int = 1024,
    title: str = "",
    db_range: float = 80.0,
) -> Path:
    """Spectrogram per tap point on one shared colour scale."""
    names = list(taps)
    n = len(names)
    specs: list[np.ndarray] = []
    for name in names:
        x = np.asarray(taps[name], dtype=np.float64)
        if len(x) < n_fft:
            x = np.pad(x, (0, n_fft - len(x)))
        hop = n_fft // 4
        frames = 1 + (len(x) - n_fft) // hop
        window = np.hanning(n_fft)
        S = np.empty((n_fft // 2 + 1, frames))
        for i in range(frames):
            S[:, i] = np.abs(np.fft.rfft(x[i * hop : i * hop + n_fft] * window))
        specs.append(20.0 * np.log10(S + _EPS))
    vmax = max(float(np.max(s)) for s in specs)
    vmin = vmax - db_range

    fig, axes = plt.subplots(n, 1, figsize=(9, 1.9 * n + 0.6), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]
    img = None
    for ax, name, S in zip(axes, names, specs):
        duration = len(taps[name]) / sample_rate
        img = ax.imshow(
            S,
            origin="lower",
            aspect="auto",
            extent=(0.0, duration, 0.0, sample_rate / 2000.0),
            vmin=vmin,
            vmax=vmax,
            cmap="magma",
        )
        ax.set_ylabel(f"{name}\nkHz", fontsize=7)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("time (s)", fontsize=8)
    if title:
        axes[0].set_title(f"{title} (shared colour scale, {db_range:.0f} dB range)", fontsize=9)
    if img is not None:
        cbar = fig.colorbar(img, ax=axes, pad=0.01, fraction=0.02)
        cbar.set_label("dB", fontsize=7)
        cbar.ax.tick_params(labelsize=6)
    return _save(fig, path, dpi)


def erle_plot(
    curve: np.ndarray,
    block_size: int,
    sample_rate: int,
    path: Path,
    dpi: int = 110,
    convergence_s: Optional[float] = None,
    speech_flags: Optional[Sequence[bool]] = None,
    title: str = "NLMS adaptive stage",
) -> Path:
    """ERLE over time with speech regions shaded and the convergence point marked."""
    c = np.asarray(curve, dtype=np.float64)
    t = np.arange(len(c)) * block_size / sample_rate
    fig, ax = plt.subplots(figsize=(9, 2.8))
    if speech_flags is not None and len(speech_flags) >= len(c):
        flags = np.asarray(speech_flags[: len(c)], dtype=bool)
        ax.fill_between(
            t, -100, 100, where=flags, color="#cccccc", alpha=0.5, step="mid",
            label="VAD: talker active (adaptation frozen)",
        )
    ax.plot(t, c, linewidth=0.8, color=PALETTE[0], label="ERLE")
    if len(c) > 20:
        window = max(5, len(c) // 50)
        smooth = np.convolve(c, np.ones(window) / window, mode="same")
        ax.plot(t, smooth, linewidth=1.6, color=PALETTE[1], label=f"ERLE ({window}-block mean)")
    if convergence_s is not None and np.isfinite(convergence_s):
        ax.axvline(convergence_s, color=PALETTE[2], linestyle="--", linewidth=1.2,
                   label=f"convergence {convergence_s:.2f} s")
    finite = c[np.isfinite(c)]
    if finite.size:
        lo = max(-10.0, float(np.percentile(finite, 1)) - 2.0)
        hi = float(np.percentile(finite, 99)) + 3.0
        ax.set_ylim(lo, max(hi, lo + 5.0))
    ax.set_xlabel("time (s)", fontsize=8)
    ax.set_ylabel("ERLE (dB)", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.3, linewidth=0.4)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="lower right")
    return _save(fig, path, dpi)


def nlms_diagnostics(
    diagnostics: Any, path: Path, dpi: int = 110
) -> Path:
    """Filter norm, effective step size and adaptation state over time."""
    t = diagnostics.time_axis()
    fig, axes = plt.subplots(3, 1, figsize=(9, 5.0), sharex=True)
    axes[0].plot(t, diagnostics.weight_norm, linewidth=0.9, color=PALETTE[0])
    axes[0].set_ylabel("||w||", fontsize=8)
    axes[0].set_title("Adaptive filter diagnostics", fontsize=9)

    axes[1].plot(t, diagnostics.mu_effective, linewidth=0.9, color=PALETTE[3])
    axes[1].set_ylabel("effective mu", fontsize=8)

    states = list(dict.fromkeys(diagnostics.state))
    code = {s: i for i, s in enumerate(states)}
    axes[2].step(t, [code[s] for s in diagnostics.state], where="post", linewidth=0.9, color=PALETTE[4])
    axes[2].set_yticks(range(len(states)))
    axes[2].set_yticklabels(states, fontsize=6)
    axes[2].set_ylabel("state", fontsize=8)
    axes[2].set_xlabel("time (s)", fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3, linewidth=0.4)
        ax.tick_params(labelsize=7)
    return _save(fig, path, dpi)


def latency_histogram(
    per_stage: Sequence[dict[str, Any]], stage_samples: Mapping[str, np.ndarray], path: Path, dpi: int = 110
) -> Path:
    """Per-stage processing time distributions with p95/p99 marked."""
    stages = [s for s in stage_samples if len(stage_samples[s]) > 1]
    if not stages:
        fig, ax = plt.subplots(figsize=(9, 2.2))
        ax.text(0.5, 0.5, "no timing samples recorded", ha="center", va="center", fontsize=9)
        ax.axis("off")
        return _save(fig, path, dpi)
    fig, axes = plt.subplots(1, len(stages), figsize=(4.2 * len(stages), 2.6), squeeze=False)
    for ax, stage, colour in zip(axes[0], stages, PALETTE):
        samples = np.asarray(stage_samples[stage], dtype=np.float64)
        ax.hist(samples, bins=min(50, max(10, len(samples) // 8)), color=colour, alpha=0.8)
        for label, value, style in (
            ("p50", float(np.median(samples)), "-"),
            ("p95", float(np.percentile(samples, 95)), "--"),
            ("p99", float(np.percentile(samples, 99)), ":"),
        ):
            ax.axvline(value, linestyle=style, color="black", linewidth=1.0, label=f"{label} {value:.2f} ms")
        ax.set_title(stage, fontsize=9)
        ax.set_xlabel("processing time per call (ms)", fontsize=7)
        ax.set_ylabel("count", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3, linewidth=0.4)
    return _save(fig, path, dpi)


def method_comparison_bars(
    rows: Sequence[dict[str, Any]],
    metric: str,
    path: Path,
    group_key: str = "category",
    method_key: str = "method",
    dpi: int = 110,
    ylabel: Optional[str] = None,
    target: Optional[float] = None,
    title: Optional[str] = None,
) -> Path:
    """Grouped bars: one group per noise category, one bar per method."""
    methods = list(dict.fromkeys(str(r[method_key]) for r in rows if metric in r))
    groups = list(dict.fromkeys(str(r[group_key]) for r in rows if metric in r))
    if not methods or not groups:
        fig, ax = plt.subplots(figsize=(9, 2.2))
        ax.text(0.5, 0.5, f"no data for {metric}", ha="center", va="center", fontsize=9)
        ax.axis("off")
        return _save(fig, path, dpi)

    values = {(str(r[method_key]), str(r[group_key])): float(r[metric]) for r in rows if metric in r}
    x = np.arange(len(groups))
    width = min(0.8 / len(methods), 0.18)
    fig, ax = plt.subplots(figsize=(max(7.0, 1.9 * len(groups) + 3.0), 3.2))
    for i, method in enumerate(methods):
        heights = [values.get((method, g), np.nan) for g in groups]
        ax.bar(
            x + (i - (len(methods) - 1) / 2) * width,
            heights,
            width=width,
            label=method,
            color=PALETTE[i % len(PALETTE)],
        )
    if target is not None:
        ax.axhline(target, color="black", linestyle="--", linewidth=1.1, label=f"target {target:g}")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=7)
    ax.set_ylabel(ylabel or metric.replace("_", " "), fontsize=8)
    ax.set_title(title or f"{metric.replace('_', ' ')} by {group_key}", fontsize=9)
    ax.grid(alpha=0.3, axis="y", linewidth=0.4)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, ncol=min(4, len(methods)))
    return _save(fig, path, dpi)


def snr_bucket_plot(
    rows: Sequence[dict[str, Any]], path: Path, metric: str = "snr_improvement_db", dpi: int = 110,
    target: Optional[float] = None,
) -> Path:
    """Improvement as a function of input SNR: shows where the ceiling bites."""
    return method_comparison_bars(
        rows,
        metric,
        path,
        group_key="snr_bucket",
        dpi=dpi,
        ylabel="SNR improvement (dB)",
        target=target,
        title="SNR improvement by input SNR bucket (the metric ceilings out at high input SNR)",
    )


def resource_timeline(records: Sequence[Mapping[str, float]], path: Path, dpi: int = 110) -> Path:
    """CPU and memory over the run."""
    if not records:
        fig, ax = plt.subplots(figsize=(9, 2.0))
        ax.text(0.5, 0.5, "no resource samples recorded", ha="center", va="center", fontsize=9)
        ax.axis("off")
        return _save(fig, path, dpi)
    t = [r.get("t_s", 0.0) for r in records]
    cpu = [r.get("cpu_percent", np.nan) for r in records]
    rss = [r.get("rss_mb", np.nan) for r in records]
    fig, ax1 = plt.subplots(figsize=(9, 2.4))
    ax1.plot(t, cpu, color=PALETTE[0], linewidth=1.0, label="CPU (%)")
    ax1.set_xlabel("time (s)", fontsize=8)
    ax1.set_ylabel("CPU (%)", fontsize=8, color=PALETTE[0])
    ax1.tick_params(labelsize=7)
    ax2 = ax1.twinx()
    ax2.plot(t, rss, color=PALETTE[1], linewidth=1.0, label="RSS (MB)")
    ax2.set_ylabel("process RSS (MB)", fontsize=8, color=PALETTE[1])
    ax2.tick_params(labelsize=7)
    ax1.set_title("Host resource usage during the run", fontsize=9)
    ax1.grid(alpha=0.3, linewidth=0.4)
    return _save(fig, path, dpi)


def rir_plot(h: np.ndarray, sample_rate: int, path: Path, dpi: int = 110,
             significant_taps: Optional[int] = None, filter_length: Optional[int] = None) -> Path:
    """RIR with its envelope, significant length and the NLMS filter length marked."""
    h = np.asarray(h, dtype=np.float64)
    t_ms = np.arange(len(h)) * 1000.0 / sample_rate
    peak = float(np.max(np.abs(h))) + _EPS
    env_db = 20.0 * np.log10(np.abs(h) / peak + _EPS)
    fig, axes = plt.subplots(2, 1, figsize=(9, 3.6), sharex=True)
    axes[0].plot(t_ms, h, linewidth=0.5, color=PALETTE[0])
    axes[0].set_ylabel("amplitude", fontsize=8)
    axes[0].set_title("Room impulse response applied to the noise path", fontsize=9)
    axes[1].plot(t_ms, env_db, linewidth=0.5, color=PALETTE[0])
    axes[1].set_ylim(-80, 3)
    axes[1].set_ylabel("dB re peak", fontsize=8)
    axes[1].set_xlabel("time (ms)", fontsize=8)
    if significant_taps:
        ms = significant_taps * 1000.0 / sample_rate
        for ax in axes:
            ax.axvline(ms, color=PALETTE[1], linestyle="--", linewidth=1.1,
                       label=f"significant length {ms:.0f} ms")
    if filter_length:
        ms = filter_length * 1000.0 / sample_rate
        for ax in axes:
            ax.axvline(ms, color=PALETTE[2], linestyle=":", linewidth=1.3,
                       label=f"NLMS filter {ms:.0f} ms")
    axes[1].legend(fontsize=6, loc="upper right")
    for ax in axes:
        ax.grid(alpha=0.3, linewidth=0.4)
        ax.tick_params(labelsize=7)
    return _save(fig, path, dpi)


def dataset_overview(stats: Any, path: Path, dpi: int = 110) -> Path:
    """SNR histogram, minutes per category and event peak distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.7))
    if stats.snr_values:
        axes[0].hist(stats.snr_values, bins=12, color=PALETTE[0], alpha=0.85)
    axes[0].set_title("input SNR distribution", fontsize=9)
    axes[0].set_xlabel("SNR (dB)", fontsize=7)

    cats = sorted(stats.minutes_by_category)
    axes[1].bar(cats, [stats.minutes_by_category[c] for c in cats], color=PALETTE[2], alpha=0.85)
    axes[1].set_title("minutes per category", fontsize=9)
    axes[1].tick_params(axis="x", labelrotation=20)

    if stats.event_peak_dbfs:
        axes[2].hist(stats.event_peak_dbfs, bins=12, color=PALETTE[1], alpha=0.85)
    axes[2].set_title("impulsive event peak level", fontsize=9)
    axes[2].set_xlabel("dBFS", fontsize=7)

    for ax in axes:
        ax.grid(alpha=0.3, axis="y", linewidth=0.4)
        ax.tick_params(labelsize=6)
    return _save(fig, path, dpi)


def live_levels(
    input_dbfs: Sequence[float], output_dbfs: Sequence[float], path: Path, dpi: int = 110,
    interval_s: float = 0.1, erle: Optional[Sequence[float]] = None,
) -> Path:
    """Input/output level over a live run, with ERLE if the adaptive stage ran."""
    t = np.arange(len(input_dbfs)) * interval_s
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.plot(t, input_dbfs, linewidth=0.9, color=PALETTE[0], label="input level (dBFS)")
    ax.plot(
        np.arange(len(output_dbfs)) * interval_s, output_dbfs, linewidth=0.9,
        color=PALETTE[2], label="output level (dBFS)",
    )
    ax.set_xlabel("time (s)", fontsize=8)
    ax.set_ylabel("dBFS", fontsize=8)
    ax.grid(alpha=0.3, linewidth=0.4)
    ax.tick_params(labelsize=7)
    if erle is not None and len(erle):
        ax2 = ax.twinx()
        ax2.plot(np.linspace(0, t[-1] if len(t) else 0, len(erle)), erle,
                 linewidth=0.8, color=PALETTE[1], alpha=0.8, label="ERLE (dB)")
        ax2.set_ylabel("ERLE (dB)", fontsize=8, color=PALETTE[1])
        ax2.tick_params(labelsize=7)
    ax.set_title("Live run levels", fontsize=9)
    ax.legend(fontsize=6, loc="lower left")
    return _save(fig, path, dpi)


def stage_difference(
    before: np.ndarray,
    after: np.ndarray,
    sample_rate: int,
    path: Path,
    dpi: int = 110,
    label_before: str = "input",
    label_after: str = "output",
    n_fft: int = 1024,
) -> Path:
    """What one stage changed: waveform overlay, removed signal, and spectral delta.

    The bottom panel is ``after - before`` in dB per time-frequency bin: blue is energy
    removed, red is energy added. For a noise suppressor it should be blue in the noise
    and neutral on the speech; red patches mean the stage is inventing energy.
    """
    n = min(len(before), len(after))
    a = np.asarray(before[:n], dtype=np.float64)
    b = np.asarray(after[:n], dtype=np.float64)
    t = np.arange(n) / sample_rate

    hop = n_fft // 4
    if n < n_fft:
        a = np.pad(a, (0, n_fft - n))
        b = np.pad(b, (0, n_fft - n))
    frames = 1 + (len(a) - n_fft) // hop
    window = np.hanning(n_fft)
    Sa = np.empty((n_fft // 2 + 1, frames))
    Sb = np.empty_like(Sa)
    for i in range(frames):
        s = i * hop
        Sa[:, i] = np.abs(np.fft.rfft(a[s : s + n_fft] * window))
        Sb[:, i] = np.abs(np.fft.rfft(b[s : s + n_fft] * window))
    delta = 20.0 * np.log10(Sb + _EPS) - 20.0 * np.log10(Sa + _EPS)
    limit = float(np.percentile(np.abs(delta), 98)) or 1.0

    fig, axes = plt.subplots(3, 1, figsize=(9, 6.2), sharex=True)
    peak = max(float(np.max(np.abs(a))), float(np.max(np.abs(b)))) or 1.0
    axes[0].plot(t, a[:n], linewidth=0.4, color=PALETTE[7], label=label_before)
    axes[0].plot(t, b[:n], linewidth=0.4, color=PALETTE[0], label=label_after, alpha=0.85)
    axes[0].set_ylim(-1.05 * peak, 1.05 * peak)
    axes[0].set_ylabel("amplitude", fontsize=8)
    axes[0].legend(fontsize=6, loc="upper right")
    axes[0].set_title(f"{label_before} vs {label_after}: what this stage changed", fontsize=9)

    # Removed content, gain-aligned so a pure level change does not show up as removal.
    denom = float(np.dot(b[:n], b[:n]))
    alpha = float(np.dot(a[:n], b[:n])) / denom if denom > _EPS else 1.0
    removed = a[:n] - alpha * b[:n]
    axes[1].plot(t, removed, linewidth=0.4, color=PALETTE[1])
    axes[1].set_ylim(-1.05 * peak, 1.05 * peak)
    axes[1].set_ylabel("removed", fontsize=8)

    img = axes[2].imshow(
        delta,
        origin="lower",
        aspect="auto",
        extent=(0.0, n / sample_rate, 0.0, sample_rate / 2000.0),
        vmin=-limit,
        vmax=limit,
        cmap="coolwarm",
    )
    axes[2].set_ylabel("kHz", fontsize=8)
    axes[2].set_xlabel("time (s)", fontsize=8)
    cbar = fig.colorbar(img, ax=axes[2], pad=0.01, fraction=0.04)
    cbar.set_label("dB change (blue = removed)", fontsize=6)
    cbar.ax.tick_params(labelsize=6)
    for ax in axes:
        ax.grid(alpha=0.25, linewidth=0.4)
        ax.tick_params(labelsize=7)
    return _save(fig, path, dpi)


def agc_trace(diagnostics: Any, path: Path, dpi: int = 110) -> Path:
    """Applied gain, tracked speech level and limiter activity over time."""
    t = diagnostics.time_axis()
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.0), sharex=True)
    axes[0].plot(t, diagnostics.gain_db, linewidth=1.0, color=PALETTE[0], label="applied gain")
    axes[0].plot(
        t, diagnostics.speech_level_db, linewidth=0.8, color=PALETTE[3], alpha=0.8,
        label="tracked speech level",
    )
    if diagnostics.speech_flags:
        flags = np.asarray(diagnostics.speech_flags, dtype=bool)
        lo, hi = axes[0].get_ylim()
        axes[0].fill_between(t, lo, hi, where=~flags, color="#dddddd", alpha=0.6, step="mid",
                             label="pause (gain held)")
        axes[0].set_ylim(lo, hi)
    axes[0].set_ylabel("dB", fontsize=8)
    axes[0].set_title("Volume normalisation: gain is held during pauses so residual noise is not pumped up",
                      fontsize=9)
    axes[0].legend(fontsize=6, loc="lower right")

    axes[1].plot(t, diagnostics.limiter_reduction_db, linewidth=0.9, color=PALETTE[1])
    axes[1].set_ylabel("limiter\nreduction (dB)", fontsize=8)
    axes[1].set_xlabel("time (s)", fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3, linewidth=0.4)
        ax.tick_params(labelsize=7)
    return _save(fig, path, dpi)


def category_snr_heatmap(
    rows: Sequence[dict[str, Any]],
    metric: str,
    path: Path,
    dpi: int = 110,
    title: Optional[str] = None,
    target: Optional[float] = None,
) -> Path:
    """Metric as a category x input-SNR grid for one method."""
    cats = sorted({str(r["category"]) for r in rows if metric in r})
    snrs = sorted({float(r["snr_db"]) for r in rows if metric in r})
    if not cats or not snrs:
        fig, ax = plt.subplots(figsize=(9, 2.2))
        ax.text(0.5, 0.5, f"no data for {metric}", ha="center", va="center", fontsize=9)
        ax.axis("off")
        return _save(fig, path, dpi)
    grid = np.full((len(cats), len(snrs)), np.nan)
    for r in rows:
        if metric not in r:
            continue
        i = cats.index(str(r["category"]))
        j = snrs.index(float(r["snr_db"]))
        grid[i, j] = float(r[metric])

    fig, ax = plt.subplots(figsize=(1.15 * len(snrs) + 3.4, 0.5 * len(cats) + 1.9))
    img = ax.imshow(grid, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(snrs)))
    ax.set_xticklabels([f"{s:g}" for s in snrs], fontsize=7)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats, fontsize=7)
    ax.set_xlabel("input SNR (dB)", fontsize=8)
    ax.set_title(title or metric.replace("_", " "), fontsize=9)
    for i in range(len(cats)):
        for j in range(len(snrs)):
            if np.isfinite(grid[i, j]):
                colour = "white" if grid[i, j] < np.nanmean(grid) else "black"
                mark = ""
                if target is not None:
                    mark = "" if grid[i, j] > target else "*"
                ax.text(j, i, f"{grid[i, j]:.2f}{mark}", ha="center", va="center",
                        fontsize=6, color=colour)
    cbar = fig.colorbar(img, ax=ax, pad=0.02, fraction=0.03)
    cbar.ax.tick_params(labelsize=6)
    if target is not None:
        ax.set_xlabel(f"input SNR (dB)   -   * marks cells below the target of {target:g}", fontsize=7)
    return _save(fig, path, dpi)
