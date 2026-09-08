"""Offline mode: files in, enhanced audio plus a full report out.

Three shapes, one code path:

* **corpus** (default) - the supplied ``dataset_plain``. A category x SNR balanced
  subset is evaluated with every method, producing the comparison tables, the
  per-category and per-SNR breakdowns and the target verdicts.
* **manifest** - a generated dataset manifest, same treatment.
* **single file** - ``--primary`` with optional ``--clean``. Produces per-stage audio,
  the stage-difference figures and a report for that one file.

This is the deliverable that must never break: no audio hardware, no threading in the
single-file path, and it degrades gracefully when the clean reference is absent.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from ..audio.io import load_audio, match_length
from ..config import Config
from ..evaluate import (
    ALL_METHODS,
    DEFAULT_METHOD,
    DELIVERED_METHODS,
    METHOD_NOTES,
    REJECTED_METHODS,
    MethodRunner,
    evaluate_manifest,
)
from ..metrics.categories import (
    MetricRecord,
    aggregate,
    check_targets,
    format_table,
    headline_summary,
)
from ..metrics.intrusive import compute_intrusive
from ..metrics.system import collect_system_metrics
from ..pipeline import Pipeline
from ..report import plots
from ..report.export import ReportData
from ..utils.logging import get_logger
from ..utils.platform_info import ResourceSampler
from ..utils.session import Session, create_session
from .common import (
    SCOPE_NOTES,
    add_signal_figures,
    add_system_figures,
    finalise,
    latency_table,
    level_rows,
    save_stage_wavs,
    session_metadata_rows,
    system_table,
)

log = get_logger(__name__)

COMPARISON_COLUMNS = (
    "method",
    "n",
    "pesq",
    "stoi",
    "estoi",
    "si_sdr",
    "snr_improvement_db",
    "segmental_snr",
    "lsd",
    "speech_attenuation_db",
    "noise_reduction_db",
    "rtf",
)
COMPARISON_HEADERS = (
    "method",
    "n",
    "PESQ",
    "STOI",
    "ESTOI",
    "SI-SDR dB",
    "SNRi dB",
    "segSNR dB",
    "LSD dB",
    "sp.atten dB",
    "noise red. dB",
    "RTF",
)


def run_offline(
    cfg: Config,
    session: Optional[Session] = None,
    methods: Sequence[str] = DELIVERED_METHODS,
    limit: Optional[int] = None,
    title: str = "Offline evaluation",
    use_corpus: bool = True,
) -> dict[str, Path]:
    """Entry point for ``anc run --mode offline_file`` and ``anc evaluate``."""
    session = session or create_session(cfg, "offline_file")
    log.info("pipeline: %s", Pipeline(cfg, load_model=False).describe())
    sampler = ResourceSampler(interval_s=0.5).start()
    t0 = time.perf_counter()

    if cfg.offline.primary is not None:
        data = _run_single(cfg, session, title)
    else:
        data = _run_batch(cfg, session, methods, limit, title, use_corpus)

    sampler.stop()
    add_system_figures(data, session, None, sampler.as_records(), cfg.report.dpi)
    data.payload.setdefault("resources", sampler.as_records())
    data.payload["wall_seconds"] = round(time.perf_counter() - t0, 3)
    data.payload["peak_rss_mb"] = sampler.peak_rss_mb()
    return finalise(data)


# ---------------------------------------------------------------- single file


def _run_single(cfg: Config, session: Session, title: str) -> ReportData:
    sr = cfg.audio.sample_rate
    primary = load_audio(cfg.offline.primary, sr)
    clean = load_audio(cfg.offline.clean, sr) if cfg.offline.clean else None
    reference = load_audio(cfg.offline.reference, sr) if cfg.offline.reference else None
    if clean is not None:
        primary, clean = match_length(primary, clean)
    if reference is not None:
        reference = reference[: len(primary)]

    log.info(
        "input: %s (%.2f s)%s",
        Path(cfg.offline.primary).name,
        len(primary) / sr,
        f", clean {Path(cfg.offline.clean).name}" if cfg.offline.clean else " (no clean target)",
    )

    pipeline = Pipeline(cfg)
    result = pipeline.process(primary, reference)
    taps = result.taps()

    data = ReportData(
        session=session,
        title=title,
        mode=session.mode,
        subtitle=f"{pipeline.describe()} &middot; single file &middot; {len(primary) / sr:.2f} s",
        scope_notes=list(SCOPE_NOTES),
    )
    data.metadata_rows = session_metadata_rows(
        session,
        cfg,
        result.model_info,
        extra=[
            ("Input", str(cfg.offline.primary)),
            ("Clean target", str(cfg.offline.clean) if cfg.offline.clean else "none (intrusive metrics unavailable)"),
        ],
    )

    records: list[MetricRecord] = []
    if clean is not None:
        base = compute_intrusive(clean, result.primary, sr, cfg.metrics)
        rows = [["Tap point", "PESQ", "STOI", "ESTOI", "SI-SDR dB", "segSNR dB", "LSD dB",
                 "sp.atten dB", "SNRi dB"]]
        for name, signal in taps.items():
            if name == "reference":
                continue
            m = compute_intrusive(clean[: len(signal)], signal, sr, cfg.metrics)
            snri = m.si_sdr - base.si_sdr
            rows.append([
                name, f"{m.pesq:.3f}", f"{m.stoi:.3f}", f"{m.estoi:.3f}", f"{m.si_sdr:.2f}",
                f"{m.segmental_snr:.2f}", f"{m.lsd:.2f}", f"{m.speech_attenuation_db:.2f}",
                f"{snri:+.2f}",
            ])
            values = m.as_dict()
            values["snr_improvement_db"] = snri
            records.append(
                MetricRecord(
                    example_id=Path(cfg.offline.primary).stem, subset="single", category="unknown",
                    noise_type="unknown", snr_db=base.si_sdr, method=cfg.pipeline.order,
                    tap=name, values=values,
                )
            )
        data.add_table("Metrics by pipeline stage", rows,
                       "One row per tap point, so each stage's contribution is separable. "
                       "Level-sensitive metrics are gain-aligned because the pipeline ends with a "
                       "volume normaliser.")
        if records:
            data.target_checks = check_targets(records[-1].values, cfg.report.targets, scope="single file")
            data.target_scope = "Single-file run, evaluated at the pipeline output."
    else:
        data.scope_notes.append(
            "No clean reference was supplied, so PESQ, STOI, ESTOI, SI-SDR and SNR improvement "
            "cannot be computed. Only levels, normalisation behaviour and timing are reported."
        )

    data.add_table("Signal levels", level_rows(taps), "RMS and peak level at each tap point.")
    _add_normalise_section(cfg, data, session, result)

    if cfg.report.write_wav:
        artefacts = dict(taps)
        artefacts["removed_by_pipeline"] = result.removed()
        data.payload["audio_files"] = save_stage_wavs(session, artefacts, sr)
    add_signal_figures(data, session, taps, sr, result, cfg.report.dpi)
    _add_stage_diff_figures(cfg, data, session, result, sr)

    system = collect_system_metrics(
        result.timings,
        audio_seconds=len(result.primary) / sr,
        wall_seconds=sum(s.total_ms for s in result.timings.summaries()) / 1e3,
        latency_ms=result.latency_ms,
        single_thread=(cfg.neural.num_threads == 1),
    )
    data.add_table("Processing time and real-time factor", system_table(system))
    data.add_table("Latency budget", latency_table(result.latency_ms),
                   "Theoretical budget. Offline runs have no audio-device buffering.")
    data.warnings.extend(system.warnings)
    data.payload.update({
        "system": system.as_dict(),
        "model": result.model_info,
        "pipeline": {"order": result.order, "describe": pipeline.describe(), "notes": result.notes},
        "normalise": result.normalise_summary,
        "guard_counters": result.guard_counters,
        "metrics_by_tap": [r.flat() for r in records],
    })
    data.csv_rows = [r.flat() for r in records]
    return data


# --------------------------------------------------------------------- batch


def _run_batch(
    cfg: Config,
    session: Session,
    methods: Sequence[str],
    limit: Optional[int],
    title: str,
    use_corpus: bool,
) -> ReportData:
    method_list = list(methods) if methods else [DEFAULT_METHOD]
    if len(method_list) == 1 and method_list[0] == "all":
        method_list = list(ALL_METHODS)

    manifest, corpus, subset, source_rows = _load_source(cfg, use_corpus)

    def progress(i: int, total: int, name: str) -> None:
        if i == 1 or i % 20 == 0 or i == total:
            log.info("  [%d/%d] %s", i, total, name)

    result = evaluate_manifest(cfg, manifest, method_list, limit, progress)

    data = ReportData(
        session=session,
        title=title,
        mode=session.mode,
        subtitle=(
            f"{result.n_examples} examples &middot; {len(result.methods)} methods &middot; "
            f"{result.audio_seconds / 60.0:.1f} min of audio &middot; "
            f"{result.wall_seconds:.0f} s on {result.workers} worker(s) "
            f"({result.throughput:.1f}x real time)"
        ),
        scope_notes=list(SCOPE_NOTES),
    )
    data.metadata_rows = session_metadata_rows(
        session, cfg, result.model_info,
        extra=[
            ("Methods compared", ", ".join(result.methods)),
            ("Examples evaluated", str(result.n_examples)),
            ("Evaluation workers", f"{result.workers} process(es), 1 torch thread each"),
        ] + source_rows,
    )

    output_records = [r for r in result.records if r.tap in ("output", "input")]

    # ---- headline and targets -------------------------------------------------
    headline = headline_summary(output_records, DEFAULT_METHOD, cfg.metrics, cfg.report.targets)
    if headline.get("aggregate"):
        data.target_checks = check_targets(headline["aggregate"], cfg.report.targets)
        lo, hi = cfg.metrics.headline_snr_range
        data.target_scope = (
            f"Delivered pipeline <b>{DEFAULT_METHOD}</b> over input SNR {lo:g} to {hi:g} dB "
            f"({headline['n']} measurements, category-balanced). SNR improvement is SI-SDR of the "
            f"output minus SI-SDR of the unprocessed input, and is bounded by how much noise was "
            f"present, so it is also reported per input SNR below."
        )
    data.payload["headline"] = headline

    # ---- method comparison ---------------------------------------------------
    delivered_rows = _order_methods(
        aggregate([r for r in output_records if r.method in DELIVERED_METHODS], group_by=("method",))
    )
    data.add_table(
        "Method comparison: delivered pipeline versus classical baselines",
        format_table(delivered_rows, COMPARISON_COLUMNS, COMPARISON_HEADERS),
        "Same examples, same metrics, same code path. 'unprocessed' is the floor. Speech "
        "attenuation catches any method that improves its noise figures by muting the talker; "
        "noise reduction is measured only in talker-silent regions.",
    )

    rejected_present = [m for m in result.methods if m in REJECTED_METHODS]
    if rejected_present:
        rejected_rows = _order_methods(
            aggregate(
                [r for r in output_records if r.method in rejected_present + [DEFAULT_METHOD, "unprocessed"]],
                group_by=("method",),
            )
        )
        data.add_table(
            "Rejected two-microphone designs, measured for the record",
            format_table(rejected_rows, COMPARISON_COLUMNS, COMPARISON_HEADERS),
            "These need a noise reference. They were given the corpus's noise-only track - the "
            "exact noise, sample aligned - which is a better reference than any real "
            "single-microphone system could obtain, so these rows are an upper bound. "
            "'dfn_then_nlms' shows why the adaptive stage was dropped: the model applies a "
            "time-varying non-linear gain, so the linear relationship between reference and "
            "residual is broken. 'nlms_then_dfn' would need a second physical microphone, which "
            "the single-microphone framing rules out.",
        )

    # ---- per category and per SNR -------------------------------------------
    cat_rows = aggregate(output_records, group_by=("category", "method"))
    cat_table = [["category", "method", "n", "PESQ", "STOI", "SNRi dB", "sp.atten dB", "noise red. dB"]]
    for row in sorted(cat_rows, key=lambda r: (str(r["category"]), _method_rank(str(r["method"])))):
        if row["method"] not in DELIVERED_METHODS:
            continue
        cat_table.append([
            str(row["category"]), str(row["method"]), str(row["n"]),
            _fmt(row.get("pesq")), _fmt(row.get("stoi")), _fmt(row.get("snr_improvement_db"), "+.2f"),
            _fmt(row.get("speech_attenuation_db"), ".2f"), _fmt(row.get("noise_reduction_db"), ".2f"),
        ])
    data.add_table("Results per noise category", cat_table,
                   "Gunshot is impulsive and is the hardest case; the other five are stationary or "
                   "non-stationary. Averaging across categories would hide this.")

    target_rows = [["Category", "Taxonomy", "n", "PESQ", "STOI", "SNRi dB",
                    "PESQ>2.5", "STOI>0.85", "SNRi>15dB"]]
    from ..dataset.plain import taxonomy_of

    for row in headline.get("per_category", []):
        checks = {c.name: c for c in check_targets(row, cfg.report.targets)}
        target_rows.append([
            str(row.get("category")), taxonomy_of(str(row.get("category"))), str(row.get("n")),
            _fmt(row.get("pesq")), _fmt(row.get("stoi")), _fmt(row.get("snr_improvement_db"), "+.2f"),
            checks["PESQ (wideband)"].verdict, checks["STOI"].verdict,
            checks["SNR improvement"].verdict,
        ])
    data.add_table(f"Targets per noise category ({DEFAULT_METHOD})", target_rows,
                   "The mandated targets evaluated per category rather than as one average.")

    snr_rows = aggregate(
        [r for r in output_records if r.method in (DEFAULT_METHOD, "unprocessed", "dfn_only")],
        group_by=("method", "snr_db"),
    )
    snr_table = [["method", "input SNR dB", "n", "PESQ", "STOI", "SI-SDR dB", "SNRi dB"]]
    for row in sorted(snr_rows, key=lambda r: (_method_rank(str(r["method"])), float(r["snr_db"]))):
        snr_table.append([
            str(row["method"]), f"{float(row['snr_db']):g}", str(row["n"]),
            _fmt(row.get("pesq")), _fmt(row.get("stoi")), _fmt(row.get("si_sdr"), ".2f"),
            _fmt(row.get("snr_improvement_db"), "+.2f"),
        ])
    data.add_table("Results per input SNR", snr_table,
                   "SNR improvement is bounded above by the noise present: at +15 dB input there is "
                   "little left to remove, so the figure necessarily falls. The low-SNR rows are the "
                   "interesting ones for a defence scenario.")

    # ---- ablation ------------------------------------------------------------
    ablation = [["Configuration", "n", "PESQ", "STOI", "SI-SDR dB", "sp.atten dB", "Output level dBFS"]]
    for method in ("unprocessed", "dfn_only", DEFAULT_METHOD):
        rows = aggregate([r for r in output_records if r.method == method], group_by=("method",))
        if not rows:
            continue
        row = rows[0]
        ablation.append([
            METHOD_NOTES.get(method, method), str(row["n"]), _fmt(row.get("pesq")),
            _fmt(row.get("stoi")), _fmt(row.get("si_sdr"), ".2f"),
            _fmt(row.get("speech_attenuation_db"), ".2f"),
            _fmt(row.get("agc_gain_mean_db"), "+.2f") if method == DEFAULT_METHOD else "-",
        ])
    data.add_table("Ablation: contribution of each stage", ablation,
                   "The normalisation stage is a gain stage: it is expected to leave the "
                   "gain-aligned quality metrics unchanged and to fix the output level. If PESQ or "
                   "STOI move noticeably between dfn_only and the full pipeline, the AGC is "
                   "distorting and needs its limiter or attack times revisited.")

    if corpus is not None:
        from ..dataset.plain import corpus_stats_rows

        data.add_table(
            "Evaluation corpus",
            [["Property", "Value"]] + [[k, v] for k, v in corpus_stats_rows(corpus, subset)],
            "The supplied corpus. Note the category imbalance, which is why a balanced subset is "
            "used for the headline numbers.",
        )

    # ---- figures -------------------------------------------------------------
    dpi = cfg.report.dpi
    bar_rows = aggregate(
        [r for r in output_records if r.method in DELIVERED_METHODS], group_by=("method", "category")
    )
    for metric, label, target in (
        ("pesq", "PESQ (wideband)", cfg.report.targets.pesq),
        ("stoi", "STOI", cfg.report.targets.stoi),
        ("snr_improvement_db", "SNR improvement (dB)", cfg.report.targets.snr_improvement_db),
    ):
        data.add_figure(
            plots.method_comparison_bars(
                bar_rows, metric, session.figure_path(f"compare_{metric}"), dpi=dpi,
                ylabel=label, target=target, title=f"{label} by noise category and method",
            ),
            f"{label} for every delivered method, split by noise category. Dashed line is the target.",
        )

    heat_rows = [
        {**row, "snr_db": row["snr_db"]}
        for row in aggregate(
            [r for r in output_records if r.method == DEFAULT_METHOD],
            group_by=("category", "snr_db"),
        )
    ]
    for metric, label, target in (
        ("pesq", "PESQ", cfg.report.targets.pesq),
        ("snr_improvement_db", "SNR improvement (dB)", cfg.report.targets.snr_improvement_db),
    ):
        data.add_figure(
            plots.category_snr_heatmap(
                heat_rows, metric, session.figure_path(f"heatmap_{metric}"), dpi=dpi,
                title=f"{label}: {DEFAULT_METHOD}, per category and input SNR", target=target,
            ),
            f"{label} for the delivered pipeline across every category and input SNR cell.",
        )

    # ---- worked example -----------------------------------------------------
    _add_worked_examples(cfg, session, data, manifest)

    data.payload.update({
        "methods": result.methods,
        "method_notes": METHOD_NOTES,
        "n_examples": result.n_examples,
        "audio_seconds": result.audio_seconds,
        "evaluation_wall_seconds": result.wall_seconds,
        "evaluation_workers": result.workers,
        "throughput_x_realtime": result.throughput,
        "aggregate_by_method": delivered_rows,
        "aggregate_by_method_category": cat_rows,
        "aggregate_by_snr": snr_rows,
        "model": result.model_info,
        "evaluation_warnings": result.warnings,
        "corpus": corpus.summary() if corpus is not None else {},
    })
    data.warnings.extend(result.warnings[:20])
    data.csv_rows = result.flat_records()
    return data


def _load_source(
    cfg: Config, use_corpus: bool
) -> tuple[dict[str, Any], Optional[Any], Optional[list[Any]], list[tuple[str, str]]]:
    """Resolve where the examples come from: the supplied corpus or a manifest."""
    if use_corpus and cfg.offline.manifest is None:
        from ..dataset.plain import build_manifest, load_plain_corpus, stratified_subset

        corpus = load_plain_corpus(cfg.plain)
        subset = stratified_subset(
            corpus, cfg.plain.per_cell, cfg.run.seed, cfg.plain.categories, cfg.plain.snr_values
        )
        manifest = build_manifest(corpus, subset)
        rows = [
            ("Corpus", f"{cfg.plain.root} ({len(corpus.examples)} labelled examples)"),
            ("Subset", f"{len(subset)} examples, {cfg.plain.per_cell} per (category, SNR) cell"),
            ("Native rate", f"{cfg.plain.native_sample_rate} Hz upsampled to {cfg.audio.sample_rate} Hz"),
        ]
        return manifest, corpus, subset, rows

    from ..dataset.build import load_manifest

    manifest = load_manifest(cfg.offline.manifest)
    return manifest, None, None, [("Dataset manifest", str(cfg.offline.manifest))]


def _fmt(value: Any, spec: str = ".3f") -> str:
    if value is None:
        return "-"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "-" if not np.isfinite(f) else format(f, spec)


def _method_rank(method: str) -> int:
    order = {name: i for i, name in enumerate(ALL_METHODS)}
    return order.get(method, 99)


def _order_methods(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda r: _method_rank(str(r.get("method"))))


def _add_normalise_section(
    cfg: Config, data: ReportData, session: Session, result: Any
) -> None:
    """Table and figure for the volume normalisation stage."""
    summary = result.normalise_summary or {}
    if not summary or summary.get("mode") == "off":
        return
    rows = [["Quantity", "Value"]]
    labels = (
        ("mode", "Mode"),
        ("gain_mean_db", "Mean applied gain (dB)"),
        ("gain_min_db", "Minimum gain (dB)"),
        ("gain_max_db", "Maximum gain (dB)"),
        ("gain_range_db", "Gain range (dB)"),
        ("speech_fraction", "Fraction of frames with speech"),
        ("held_fraction", "Fraction of frames with gain held (pause)"),
        ("limited_frames", "Frames the limiter acted on"),
        ("limiter_max_reduction_db", "Largest limiter reduction (dB)"),
        ("clipped_samples", "Samples clipped by the safety clamp"),
        ("latency_ms", "Latency added by this stage (ms)"),
    )
    for key, label in labels:
        if key in summary:
            v = summary[key]
            rows.append([label, f"{v:.3f}" if isinstance(v, float) else str(v)])
    data.add_table(
        "Volume normalisation", rows,
        "The gain is held during pauses: without that a naive AGC raises its gain when the talker "
        "stops and audibly pumps the residual noise. Clipped samples should be zero, since the "
        "look-ahead limiter is there to prevent clipping rather than clip.",
    )
    diag = getattr(getattr(result, "normaliser", None), "diagnostics", None)
    data.payload["normalise"] = summary


def _add_stage_diff_figures(
    cfg: Config, data: ReportData, session: Session, result: Any, sr: int, prefix: str = ""
) -> None:
    """Per-stage 'what changed' figures, and the AGC gain trace."""
    tag = f"{prefix}_" if prefix else ""
    taps = result.taps(include_reference=False)
    names = list(taps)
    limit = int(12.0 * sr)
    for before_name, after_name in zip(names, names[1:]):
        before = taps[before_name][:limit]
        after = taps[after_name][:limit]
        data.add_figure(
            plots.stage_difference(
                before, after, sr,
                session.figure_path(f"{tag}diff_{before_name}_to_{after_name}"),
                cfg.report.dpi, label_before=before_name, label_after=after_name,
            ),
            f"What changed between <b>{before_name}</b> and <b>{after_name}</b>. Middle panel is the "
            f"removed content, gain-aligned so a level change alone does not appear as removal. "
            f"Bottom panel is the spectral delta: blue is energy taken out, red is energy added.",
        )


def _add_worked_examples(
    cfg: Config, session: Session, data: ReportData, manifest: dict[str, Any]
) -> None:
    """Process one impulsive and one stationary example end to end for the figures."""
    examples = manifest.get("examples", [])
    if not examples:
        return
    sr = cfg.audio.sample_rate
    pipeline = Pipeline(cfg)

    def pick(predicate) -> Optional[dict[str, Any]]:
        return next((e for e in examples if predicate(e)), None)

    wanted = [
        ("gunshot", pick(lambda e: e.get("category") == "gunshot" and float(e.get("snr_db", 99)) <= 0)),
        ("helicopter", pick(lambda e: e.get("category") == "helicopter" and float(e.get("snr_db", 99)) <= 0)),
    ]
    worked: list[dict[str, Any]] = []
    for label, entry in wanted:
        if entry is None:
            continue
        files = entry.get("files", {})
        if "primary" not in files:
            continue
        try:
            primary = load_audio(files["primary"], sr)[: int(cfg.plain.max_duration_s * sr)]
            clean = load_audio(files["target"], sr)[: len(primary)] if "target" in files else None
        except FileNotFoundError:
            continue
        result = pipeline.process(primary)
        taps = result.taps(include_reference=False)
        if clean is not None:
            figure_taps = {"clean_target": clean[: len(result.primary)], **taps}
        else:
            figure_taps = taps

        add_signal_figures(data, session, figure_taps, sr, result, cfg.report.dpi, prefix=f"ex_{label}")
        _add_stage_diff_figures(cfg, data, session, result, sr, prefix=f"ex_{label}")
        if result.normalise_summary and pipeline.normaliser is not None:
            diag = pipeline.normaliser.diagnostics
            if diag is not None and diag.gain_db:
                data.add_figure(
                    plots.agc_trace(diag, session.figure_path(f"ex_{label}_agc"), cfg.report.dpi),
                    f"Volume normalisation on the {label} example: applied gain, tracked speech "
                    f"level and limiter activity. Grey spans are pauses where the gain is held.",
                )
        if cfg.report.write_wav:
            artefacts = {f"{label}_{k}": v for k, v in figure_taps.items()}
            artefacts[f"{label}_removed_by_pipeline"] = result.removed()
            data.payload.setdefault("worked_example_audio", {}).update(
                save_stage_wavs(session, artefacts, sr)
            )
        worked.append({
            "label": label,
            "id": entry.get("id"),
            "category": entry.get("category"),
            "snr_db": entry.get("snr_db"),
            "normalise": result.normalise_summary,
        })
    data.payload["worked_examples"] = worked
