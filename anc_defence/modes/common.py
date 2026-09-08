"""Shared report assembly for every mode.

Keeps the session metadata block, the scope statement and the standard figure set
identical across modes, so two reports from different modes are directly comparable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..audio.io import dbfs, peak_dbfs, save_aligned_multichannel, save_wav
from ..config import Config
from ..metrics.system import SystemMetrics
from ..pipeline import PipelineResult
from ..report import plots
from ..report.export import ReportData
from ..report.pdf import INTERPRETATION_NOTES
from ..utils.logging import get_logger
from ..utils.platform_info import EDGE_READINESS_NOTE
from ..utils.session import Session

log = get_logger(__name__)


SCOPE_NOTES: list[str] = [
    "This is a <b>single-microphone</b> system: capture, pre-processing, DeepFilterNet3, volume "
    "normalisation. There is no adaptive filter in the delivered path because a one-microphone "
    "system has no noise reference to give it. The two adaptive placements that were tried are "
    "measured and reported as rejected designs rather than left unexplained.",
    "The neural stage is <b>pretrained DeepFilterNet3, unmodified</b>. No model was trained or "
    "fine-tuned in this project. A training pipeline is designed and documented in README.md but "
    "was <b>not executed</b>.",
    "Audio is carried at 48 kHz because the model is a 48 kHz model. The supplied corpus is 16 kHz "
    "and is <b>upsampled on load</b>, so the 8-24 kHz band is empty and this evaluation does not "
    "exercise the model's fullband capability. The reported metrics are unaffected: PESQ is scored "
    "at 16 kHz and STOI at 10 kHz.",
    EDGE_READINESS_NOTE,
]


def session_metadata_rows(
    session: Session,
    cfg: Config,
    model_info: Optional[Mapping[str, Any]] = None,
    extra: Optional[Sequence[tuple[str, str]]] = None,
) -> list[tuple[str, str]]:
    """The session table that heads every report."""
    host = session.host
    cpu = host.get("cpu", {})
    torch_info = host.get("torch", {})
    git = host.get("git", {})
    packages = host.get("packages", {})

    rows: list[tuple[str, str]] = [
        ("Generated", str(host.get("timestamp", ""))),
        ("Mode", session.mode),
        ("Session directory", str(session.root)),
        ("Pipeline order", cfg.pipeline.order),
        (
            "Host",
            f"{host.get('os', '?')} &middot; {cpu.get('processor', '?')} &middot; "
            f"{cpu.get('physical_cores', '?')} cores / {cpu.get('logical_cores', '?')} threads &middot; "
            f"{cpu.get('total_ram_gb', '?')} GB RAM",
        ),
        ("Jetson hardware", "not present (is_jetson = False)" if not host.get("is_jetson") else "present"),
        (
            "Python / torch",
            f"{host.get('python', '?')} / {torch_info.get('version', 'n/a')} "
            f"(CUDA available: {torch_info.get('cuda_available')})",
        ),
        (
            "Key packages",
            ", ".join(
                f"{k} {v}"
                for k, v in packages.items()
                if v and k in ("deepfilternet", "numpy", "scipy", "pesq", "pystoi", "pyroomacoustics")
            ),
        ),
        (
            "Git",
            f"{str(git.get('commit') or 'n/a')[:12]} on {git.get('branch', '?')}"
            + (" (uncommitted changes present)" if git.get("dirty") else ""),
        ),
        ("Sample rate / hop", f"{cfg.audio.sample_rate} Hz / {cfg.audio.hop_size} samples "
                              f"({1000.0 * cfg.audio.hop_size / cfg.audio.sample_rate:.0f} ms)"),
    ]
    if model_info:
        rows += [
            (
                "Neural model",
                f"{model_info.get('name')} &middot; {model_info.get('n_parameters', 0) / 1e6:.2f}M params "
                f"&middot; {model_info.get('checkpoint')} &middot; fine-tuned: "
                f"{model_info.get('fine_tuned')}",
            ),
            (
                "Neural backend",
                f"{model_info.get('device')} &middot; {model_info.get('num_threads')} torch thread(s) "
                f"&middot; atten_lim_db: {model_info.get('atten_lim_db')} "
                f"&middot; post-filter: {model_info.get('post_filter')}",
            ),
            (
                "Model algorithmic latency",
                f"{model_info.get('algorithmic_latency_ms')} ms "
                f"(frame 10 ms + STFT {model_info.get('latency_breakdown_ms', {}).get('stft_istft_ms', '?')} ms "
                f"+ lookahead {model_info.get('effective_lookahead_frames')} frames)",
            ),
        ]
    rows += [
        (
            "Volume normalisation",
            f"{cfg.normalise.mode} &middot; target {cfg.normalise.target_dbfs} dBFS active speech "
            f"&middot; gain range [{cfg.normalise.min_gain_db}, {cfg.normalise.max_gain_db}] dB "
            f"&middot; hold during pause: {cfg.normalise.hold_during_pause} &middot; "
            f"limiter ceiling {cfg.normalise.limiter_ceiling_dbfs} dBFS with "
            f"{cfg.normalise.limiter_lookahead_ms} ms look-ahead",
        ),
        (
            "Neural framing",
            f"{cfg.neural.streaming.mode} &middot; chunk {cfg.neural.streaming.chunk_s} s "
            f"&middot; overlap {cfg.neural.streaming.overlap} &middot; "
            f"offline framing: {cfg.neural.streaming.offline_framing}",
        ),
        ("Seed", str(cfg.run.seed)),
    ]
    if extra:
        rows += list(extra)
    return rows


def level_rows(taps: Mapping[str, np.ndarray]) -> list[list[str]]:
    """RMS and peak level at every tap point: the quickest sanity check there is."""
    rows = [["Tap point", "RMS (dBFS)", "Peak (dBFS)", "Samples"]]
    for name, x in taps.items():
        rows.append([name, f"{dbfs(x):.1f}", f"{peak_dbfs(x):.1f}", str(len(x))])
    return rows


def system_table(system: SystemMetrics) -> list[list[str]]:
    rows = [["Stage", "Calls", "Mean (ms)", "p95 (ms)", "p99 (ms)", "Max (ms)", "RTF"]]
    for s in system.per_stage:
        rows.append(
            [
                str(s.get("stage")),
                str(s.get("count")),
                f"{s.get('mean_ms', 0):.3f}",
                f"{s.get('p95_ms', 0):.3f}",
                f"{s.get('p99_ms', 0):.3f}",
                f"{s.get('max_ms', 0):.3f}",
                f"{s.get('rtf', float('nan')):.4f}",
            ]
        )
    rows.append(["TOTAL", "", "", "", "", "", f"{system.rtf_total:.4f}"])
    return rows


def latency_table(latency_ms: Mapping[str, float], measured: Optional[float] = None) -> list[list[str]]:
    rows = [["Component", "Latency (ms)"]]
    order = [
        ("block_ms", "Input block (one hop)"),
        ("nlms_ms", "NLMS (overlap-save, no added delay)"),
        ("frame_ms", "Model frame"),
        ("stft_istft_ms", "Model STFT/ISTFT (n_fft - hop)"),
        ("model_lookahead_ms", "Model lookahead (2 frames)"),
        ("total_algorithmic_ms", "Model algorithmic subtotal"),
        ("chunk_buffer_ms", "Chunk buffering (live path only)"),
        ("total_theoretical_ms", "Total theoretical"),
    ]
    for key, label in order:
        if key in latency_ms:
            rows.append([label, f"{latency_ms[key]:.1f}"])
    if measured is not None:
        rows.append(["Measured (loopback)", f"{measured:.1f}"])
    return rows


def save_stage_wavs(session: Session, taps: Mapping[str, np.ndarray], sample_rate: int) -> dict[str, str]:
    """Write one WAV per tap point plus a sample-aligned multichannel file."""
    written: dict[str, str] = {}
    for name, x in taps.items():
        p = save_wav(session.audio_path(name), x, sample_rate)
        written[name] = str(p)
    path, order = save_aligned_multichannel(
        session.audio_path("all_stages_aligned"), taps, sample_rate
    )
    written["all_stages_aligned"] = str(path)
    written["all_stages_aligned_channel_order"] = ", ".join(order)
    return written


def add_signal_figures(
    data: ReportData,
    session: Session,
    taps: Mapping[str, np.ndarray],
    sample_rate: int,
    result: Optional[PipelineResult] = None,
    dpi: int = 110,
    prefix: str = "",
    max_seconds: float = 12.0,
) -> None:
    """Waveforms, spectrograms and the adaptive-stage diagnostics."""
    limit = int(max_seconds * sample_rate)
    trimmed = {k: (v[:limit] if len(v) > limit else v) for k, v in taps.items()}
    tag = f"{prefix}_" if prefix else ""

    data.add_figure(
        plots.waveforms(trimmed, sample_rate, session.figure_path(f"{tag}waveforms"), dpi,
                        title="Waveforms per pipeline stage"),
        "Waveforms at each tap point on a shared amplitude scale. The reference channel is the dry "
        "noise the adaptive filter is given; it is not the signal that was added to the primary.",
    )
    data.add_figure(
        plots.spectrograms(trimmed, sample_rate, session.figure_path(f"{tag}spectrograms"), dpi,
                           title="Spectrograms per pipeline stage"),
        "Spectrograms on a shared colour scale, so stage-to-stage differences are real and not an "
        "artefact of autoscaling.",
    )

    if result is not None and getattr(result, "nlms", None) is not None:
        diag = result.nlms.diagnostics
        from ..dsp.nlms import convergence_time_s

        t_conv = convergence_time_s(np.asarray(diag.erle_db), diag.block_size, diag.sample_rate)
        data.add_figure(
            plots.erle_plot(
                np.asarray(diag.erle_db),
                diag.block_size,
                diag.sample_rate,
                session.figure_path(f"{tag}erle"),
                dpi,
                convergence_s=t_conv,
                speech_flags=diag.speech_flags,
            ),
            "ERLE over time. Shaded spans are frames where the VAD reported the talker active and "
            "adaptation was frozen: without that freeze the filter learns to cancel speech.",
        )
        data.add_figure(
            plots.nlms_diagnostics(diag, session.figure_path(f"{tag}nlms_diagnostics"), dpi),
            "Filter norm, effective step size and adaptation state. Rollbacks appear as "
            "'rolled_back' in the state track.",
        )


def add_system_figures(
    data: ReportData,
    session: Session,
    result: Optional[PipelineResult],
    resource_records: Sequence[Mapping[str, float]] = (),
    dpi: int = 110,
    prefix: str = "",
) -> None:
    tag = f"{prefix}_" if prefix else ""
    if result is not None:
        stage_samples = {name: t.samples_ms for name, t in result.timings.stages.items()}
        data.add_figure(
            plots.latency_histogram(
                [s.as_dict() for s in result.timings.summaries()],
                stage_samples,
                session.figure_path(f"{tag}stage_timings"),
                dpi,
            ),
            "Per-call processing time distribution for each stage, with p50/p95/p99 marked. This is "
            "compute time, not latency.",
        )
    if resource_records:
        data.add_figure(
            plots.resource_timeline(resource_records, session.figure_path(f"{tag}resources"), dpi),
            "Host CPU and process memory over the run. A flat memory trace is the check for leaks.",
        )


def finalise(data: ReportData) -> dict[str, Path]:
    """Write the PDF, JSON and CSV artefacts for a completed run."""
    from ..report.export import write_json_report, write_metrics_csv
    from ..report.pdf import build_pdf_report

    data.interpretation = INTERPRETATION_NOTES
    written: dict[str, Path] = {}
    cfg = data.session.config
    if cfg.report.write_json:
        written["json"] = write_json_report(data)
    if cfg.report.write_csv:
        csv_path = write_metrics_csv(data)
        if csv_path:
            written["csv"] = csv_path
    if cfg.report.write_pdf:
        written["pdf"] = build_pdf_report(data)
    return written
