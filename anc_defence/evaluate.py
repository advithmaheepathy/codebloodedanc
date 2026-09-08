"""Evaluation harness: method comparison, per-stage metrics and the ablation matrix.

Every method runs over the same examples with the same metrics through the same code
path, so the comparison table is apples to apples.

Delivered methods
-----------------
``unprocessed``          the floor: no processing at all
``spectral_subtraction`` classical single-channel baseline
``wiener``               classical single-channel baseline
``dfn_only``             DeepFilterNet3 alone
``dfn_then_normalise``   the delivered pipeline: model then volume normalisation

Rejected designs, measured for the record
-----------------------------------------
``nlms_only``, ``nlms_then_dfn``, ``dfn_then_nlms`` need a noise reference. They are
evaluated using the corpus's noise-only file, which is the exact noise, sample
aligned - a reference no real single-microphone system could ever have. Whatever they
score is therefore an upper bound, and the report presents them that way.

Speed
-----
Batch evaluation runs across worker processes, each with a single torch thread and its
own model instance. That is both the fast path and the source of the single-thread RTF
figure used as portability evidence.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from .audio.io import load_audio, match_length
from .config import Config
from .dsp.baselines import BlockLms, spectral_subtraction, wiener_filter
from .dsp.baselines.lms import LMS_DEFAULT_MU
from .dsp.normalise import active_speech_dbfs
from .dsp.vad import speech_mask
from .enhance.streaming import DfnModel
from .metrics.categories import MetricRecord
from .metrics.erle import erle_summary, noise_reduction_db, rms_reduction
from .metrics.events import event_local_metrics
from .metrics.intrusive import (
    compute_intrusive,
    improvement,
    mixture_snr_db,
    residual_noise_snr_db,
)
from .pipeline import Pipeline, PipelineResult
from .utils.logging import get_logger
from .utils.timing import TimingRegistry

log = get_logger(__name__)

DELIVERED_METHODS = (
    "unprocessed",
    "spectral_subtraction",
    "wiener",
    "dfn_only",
    "dfn_then_normalise",
)
REJECTED_METHODS = ("nlms_only", "nlms_then_dfn", "dfn_then_nlms", "lms")
ALL_METHODS = DELIVERED_METHODS + REJECTED_METHODS
DEFAULT_METHOD = "dfn_then_normalise"

PIPELINE_METHODS = {
    "dfn_only": "dfn_only",
    "dfn_then_normalise": "dfn_then_normalise",
    "nlms_only": "nlms_only",
    "nlms_then_dfn": "nlms_then_dfn",
    "dfn_then_nlms": "dfn_then_nlms",
}

METHOD_NEEDS_REFERENCE = {"lms", "nlms_only", "nlms_then_dfn", "dfn_then_nlms"}

METHOD_NOTES = {
    "unprocessed": "no processing; the floor every other row is measured against",
    "spectral_subtraction": "classical, single channel, assumes a slowly varying noise spectrum",
    "wiener": "classical, single channel, decision-directed a priori SNR",
    "dfn_only": "pretrained DeepFilterNet3, unmodified",
    "dfn_then_normalise": "delivered pipeline: DeepFilterNet3 then speech-aware volume normalisation",
    "lms": "classical fixed-step LMS; needs a noise reference (rejected design)",
    "nlms_only": "adaptive filter alone; needs a noise reference (rejected design)",
    "nlms_then_dfn": "adaptive filter before the model; needs a second microphone (rejected design)",
    "dfn_then_nlms": "adaptive filter after the model; breaks the linear reference relationship (rejected design)",
}


@dataclass
class MethodOutput:
    """One method's result for one example."""

    name: str
    output: np.ndarray
    stage_outputs: dict[str, np.ndarray] = field(default_factory=dict)
    nlms_summary: dict[str, Any] = field(default_factory=dict)
    normalise_summary: dict[str, Any] = field(default_factory=dict)
    erle: Optional[np.ndarray] = None
    timings: Optional[TimingRegistry] = None
    diverged: bool = False
    notes: list[str] = field(default_factory=list)
    # Per-sample gain the normaliser applied. Quality metrics are computed on the
    # gain-compensated output so that a time-varying gain is not scored as distortion;
    # level is reported separately from the uncompensated signal.
    gain_envelope: Optional[np.ndarray] = None


class MethodRunner:
    """Builds and runs the requested methods, sharing one loaded model between them."""

    def __init__(self, cfg: Config, methods: Sequence[str] = DELIVERED_METHODS) -> None:
        unknown = [m for m in methods if m not in ALL_METHODS]
        if unknown:
            raise ValueError(f"unknown method(s): {unknown}. Available: {list(ALL_METHODS)}")
        self.cfg = cfg
        self.methods = list(methods)
        self.sample_rate = cfg.audio.sample_rate
        self._model: Optional[DfnModel] = None
        self._pipelines: dict[str, Pipeline] = {}
        if any(m in PIPELINE_METHODS and "dfn" in m for m in self.methods):
            self._model = DfnModel(cfg.neural, self.sample_rate).load()

    def _pipeline(self, order: str) -> Pipeline:
        if order not in self._pipelines:
            cfg = self.cfg.model_copy(deep=True)
            cfg.pipeline.order = order  # type: ignore[assignment]
            self._pipelines[order] = Pipeline(cfg, model=self._model)
        return self._pipelines[order]

    @property
    def model_info(self) -> dict[str, Any]:
        return self._model.info.as_dict() if self._model else {}

    @staticmethod
    def requires_reference(method: str) -> bool:
        return method in METHOD_NEEDS_REFERENCE

    def run(
        self, method: str, primary: np.ndarray, reference: Optional[np.ndarray]
    ) -> MethodOutput:
        sr = self.sample_rate
        if method == "unprocessed":
            return MethodOutput(name=method, output=np.asarray(primary, dtype=np.float32))

        if method in ("spectral_subtraction", "wiener"):
            timings = TimingRegistry()
            t = timings.timer(method)
            t.start()
            y = (
                spectral_subtraction(primary, sr)
                if method == "spectral_subtraction"
                else wiener_filter(primary, sr)
            )
            t.stop(audio_s=len(primary) / sr)
            return MethodOutput(name=method, output=y, timings=timings)

        if method == "lms":
            if reference is None:
                return MethodOutput(
                    name=method,
                    output=np.asarray(primary, dtype=np.float32),
                    notes=["no reference channel available"],
                )
            timings = TimingRegistry()
            t = timings.timer("lms")
            t.start()
            result = BlockLms(
                filter_length=self.cfg.nlms.filter_length,
                block_size=self.cfg.audio.hop_size,
                mu=LMS_DEFAULT_MU,
            ).process(primary, reference)
            t.stop(audio_s=len(result.output) / sr)
            return MethodOutput(
                name=method,
                output=result.output,
                erle=result.erle_db,
                timings=timings,
                diverged=result.diverged,
                notes=["classical LMS diverged and was disabled"] if result.diverged else [],
            )

        pipe = self._pipeline(PIPELINE_METHODS[method])
        pipe.timings = TimingRegistry()
        result: PipelineResult = pipe.process(primary, reference)
        envelope = None
        if pipe.normaliser is not None and pipe.normaliser.enabled:
            envelope = pipe.normaliser.gain_envelope(len(result.output))
        return MethodOutput(
            name=method,
            output=result.output,
            stage_outputs=dict(result.stage_outputs),
            nlms_summary=result.nlms.diagnostics.summary() if result.nlms else {},
            normalise_summary=result.normalise_summary,
            erle=result.erle_curve if result.nlms else None,
            timings=result.timings,
            notes=list(result.notes),
            gain_envelope=envelope,
        )


# ------------------------------------------------------------------- scoring


def evaluate_example(
    cfg: Config,
    runner: MethodRunner,
    primary: np.ndarray,
    clean: Optional[np.ndarray],
    example_id: str,
    subset: str,
    category: str,
    noise_type: str,
    snr_db: float,
    reference: Optional[np.ndarray] = None,
    noise_only: Optional[np.ndarray] = None,
    taxonomy: str = "unknown",
    keep_signals: bool = False,
) -> tuple[list[MetricRecord], dict[str, MethodOutput]]:
    """Run every requested method on one example and score every tap point."""
    sr = cfg.audio.sample_rate
    records: list[MetricRecord] = []
    outputs: dict[str, MethodOutput] = {}

    arrays = [primary]
    if clean is not None:
        arrays.append(clean)
    if reference is not None:
        arrays.append(reference)
    trimmed = match_length(*arrays)
    primary = trimmed[0]
    idx = 1
    if clean is not None:
        clean = trimmed[idx]
        idx += 1
    if reference is not None:
        reference = trimmed[idx]

    base_metrics = compute_intrusive(clean, primary, sr, cfg.metrics) if clean is not None else None
    noise_mask = ~speech_mask(clean, cfg.vad, sr) if clean is not None else None
    measured_input_snr = (
        mixture_snr_db(clean, noise_only[: len(clean)])
        if clean is not None and noise_only is not None
        else float("nan")
    )

    def add(
        method: str,
        tap: str,
        signal: np.ndarray,
        extra: Optional[dict[str, float]] = None,
        gain_envelope: Optional[np.ndarray] = None,
    ) -> None:
        n = min(len(signal), len(primary))
        sig = signal[:n]
        values: dict[str, float] = {}

        # Level is measured on the signal as delivered; quality is measured on the
        # gain-compensated signal. A volume normaliser applies a known, invertible,
        # time-varying gain: SI-SDR is invariant to a constant scale but not to a gain
        # ride, so scoring quality without compensating would penalise the normaliser
        # for doing exactly what it is there to do.
        values["output_level_dbfs"] = active_speech_dbfs(sig, sr)
        scored = sig
        if gain_envelope is not None and len(gain_envelope) >= n:
            env = np.asarray(gain_envelope[:n], dtype=np.float32)
            scored = np.divide(sig, np.maximum(env, 1e-6)).astype(np.float32)
            values["gain_compensated"] = 1.0
        if clean is not None:
            m = compute_intrusive(clean[:n], scored, sr, cfg.metrics)
            values.update(m.as_dict())
            if base_metrics is not None:
                values.update(improvement(base_metrics, m))
        if noise_mask is not None and cfg.metrics.erle:
            values["noise_reduction_db"] = noise_reduction_db(primary[:n], sig, noise_mask[:n])
        # Whole-signal RMS drop, for like-for-like comparison against tools that quote
        # it as their headline "noise reduction". See metrics.erle.rms_reduction.
        values.update(rms_reduction(primary[:n], sig))
        if cfg.metrics.events and taxonomy == "impulsive":
            ev = event_local_metrics(
                primary[:n], sig, sr, clean=clean[:n] if clean is not None else None
            )
            values.update({k: float(v) for k, v in ev.as_dict().items() if isinstance(v, (int, float))})
        if np.isfinite(measured_input_snr):
            values["measured_input_snr_db"] = measured_input_snr
        if extra:
            values.update(extra)
        records.append(
            MetricRecord(
                example_id=example_id,
                subset=subset,
                category=category,
                noise_type=noise_type,
                snr_db=snr_db,
                method=method,
                tap=tap,
                values=values,
            )
        )

    add("unprocessed", "input", primary)

    # Cache for the residual-noise output SNR: identical for dfn_only and
    # dfn_then_normalise (normalisation is gain-only), so compute it at most once.
    _residual_noise_snr: list[Optional[float]] = [None]

    for method in runner.methods:
        if method == "unprocessed":
            continue
        if runner.requires_reference(method) and reference is None:
            continue
        out = runner.run(method, primary, reference)
        extra: dict[str, float] = {}
        if out.erle is not None and len(out.erle):
            extra.update(erle_summary(np.asarray(out.erle), cfg.audio.hop_size, sr))
        if out.timings is not None:
            summaries = out.timings.summaries()
            audio_s = max(len(out.output) / sr, 1e-9)
            extra["rtf"] = (sum(s.total_ms for s in summaries) / 1e3) / audio_s
        for key in ("erle_mean_db", "erle_final_db", "adapting_fraction", "rollbacks"):
            if key in out.nlms_summary:
                extra[f"nlms_{key}"] = float(out.nlms_summary[key])
        for key in ("gain_mean_db", "gain_range_db", "held_fraction", "limited_frames",
                    "clipped_samples"):
            if key in out.normalise_summary:
                extra[f"agc_{key}"] = float(out.normalise_summary[key])
        extra["diverged"] = 1.0 if out.diverged else 0.0
        if out.gain_envelope is not None:
            extra["level_error_db"] = abs(
                active_speech_dbfs(out.output, sr) - cfg.normalise.target_dbfs
            )

        # Intermediate taps for multi-stage methods, so each stage is separable.
        for tap_name, tap_signal in out.stage_outputs.items():
            if tap_name != f"after_{_final_stage_label(method)}":
                add(method, tap_name, tap_signal)
        # Classical output SNR (speech power over residual-noise power) for the neural
        # method, using the kept clean and noise-only tracks. This is the least arguable
        # reading of the mandated "SNR > 15 dB" target. It needs two extra model calls per
        # example, so it is only done for dfn_only (the model's own output SNR, which the
        # delivered pipeline inherits since normalisation is gain-only) and only where the
        # noise-only reference exists.
        if (
            method in ("dfn_only", "dfn_then_normalise")
            and clean is not None
            and noise_only is not None
            and runner._model is not None
        ):
            if _residual_noise_snr[0] is None:
                k = min(len(clean), len(noise_only))
                enh_speech = runner._model.enhance_array(clean[:k], count_time=False)
                enh_noise = runner._model.enhance_array(noise_only[:k], count_time=False)
                _residual_noise_snr[0] = residual_noise_snr_db(enh_speech, enh_noise)
            if np.isfinite(_residual_noise_snr[0]):
                extra["residual_noise_snr_db"] = float(_residual_noise_snr[0])

        add(method, "output", out.output, extra, gain_envelope=out.gain_envelope)
        outputs[method] = (
            out
            if keep_signals
            else MethodOutput(
                name=out.name,
                output=np.zeros(0, dtype=np.float32),
                nlms_summary=out.nlms_summary,
                normalise_summary=out.normalise_summary,
                diverged=out.diverged,
                notes=out.notes,
            )
        )

    return records, outputs


def _final_stage_label(method: str) -> str:
    if method.endswith("normalise"):
        return "normalise"
    if method.endswith("nlms") or method == "nlms_only":
        return "nlms"
    return "deepfilternet"


# ---------------------------------------------------------------- batch runner


@dataclass
class EvaluationResult:
    records: list[MetricRecord] = field(default_factory=list)
    model_info: dict[str, Any] = field(default_factory=dict)
    methods: list[str] = field(default_factory=list)
    n_examples: int = 0
    wall_seconds: float = 0.0
    audio_seconds: float = 0.0
    workers: int = 1
    warnings: list[str] = field(default_factory=list)

    def flat_records(self) -> list[dict[str, Any]]:
        return [r.flat() for r in self.records]

    @property
    def throughput(self) -> float:
        """Audio seconds processed per wall second across all workers and methods."""
        return self.audio_seconds / self.wall_seconds if self.wall_seconds > 0 else float("nan")


# Worker-process globals: the model is loaded once per process, not once per example.
_WORKER_STATE: dict[str, Any] = {}


def _worker_init(cfg_dict: dict[str, Any], methods: list[str], threads: int) -> None:  # pragma: no cover
    import torch

    torch.set_num_threads(max(1, threads))
    from .config import Config as _Config

    cfg = _Config(**cfg_dict)
    _WORKER_STATE["cfg"] = cfg
    _WORKER_STATE["runner"] = MethodRunner(cfg, methods)


def _worker_run(entry: dict[str, Any]) -> tuple[list[dict[str, Any]], float, Optional[str]]:  # pragma: no cover
    cfg: Config = _WORKER_STATE["cfg"]
    runner: MethodRunner = _WORKER_STATE["runner"]
    try:
        records, audio_s = _evaluate_entry(cfg, runner, entry)
        return [r.flat() for r in records], audio_s, None
    except Exception as exc:  # noqa: BLE001 - one bad file must not kill the run
        return [], 0.0, f"{entry.get('id')}: {type(exc).__name__}: {exc}"


def _evaluate_entry(
    cfg: Config, runner: MethodRunner, entry: dict[str, Any]
) -> tuple[list[MetricRecord], float]:
    sr = cfg.audio.sample_rate
    files = entry.get("files", {})
    max_n = int(cfg.plain.max_duration_s * sr)

    primary = load_audio(files["primary"], sr)[:max_n]
    clean = load_audio(files["target"], sr)[:max_n] if "target" in files else None
    noise_only = load_audio(files["noise"], sr)[:max_n] if "noise" in files else None
    # The rejected adaptive designs are given the noise-only track as their reference,
    # which is the most favourable reference that could possibly exist.
    reference = noise_only if any(runner.requires_reference(m) for m in runner.methods) else None

    components = entry.get("components", [])
    noise_type = components[0].get("noise_type", "unknown") if components else "unknown"
    records, _ = evaluate_example(
        cfg=cfg,
        runner=runner,
        primary=primary,
        clean=clean,
        example_id=str(entry.get("id", "?")),
        subset=str(entry.get("id", "")).split("/")[0],
        category=str(entry.get("category", "unknown")),
        noise_type=str(noise_type),
        snr_db=float(entry.get("snr_db", float("nan"))),
        reference=reference,
        noise_only=noise_only,
        taxonomy=str(entry.get("taxonomy", "unknown")),
    )
    return records, len(primary) / sr


def evaluate_manifest(
    cfg: Config,
    manifest: dict[str, Any],
    methods: Sequence[str] = DELIVERED_METHODS,
    limit: Optional[int] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
    workers: Optional[int] = None,
) -> EvaluationResult:
    """Evaluate every example in a manifest with every requested method."""
    entries: list[dict[str, Any]] = list(manifest.get("examples", []))
    if limit:
        entries = entries[:limit]
    method_list = list(methods)

    n_workers = workers if workers is not None else cfg.offline.workers
    if n_workers <= 0:
        n_workers = max(1, min(8, (os.cpu_count() or 2) - 1))
    n_workers = min(n_workers, max(1, len(entries)))

    result = EvaluationResult(methods=method_list, workers=n_workers)
    t0 = time.perf_counter()

    if n_workers == 1:
        runner = MethodRunner(cfg, method_list)
        result.model_info = runner.model_info
        for i, entry in enumerate(entries):
            try:
                records, audio_s = _evaluate_entry(cfg, runner, entry)
            except Exception as exc:  # noqa: BLE001
                msg = f"{entry.get('id')}: {type(exc).__name__}: {exc}"
                log.warning("skipping %s", msg)
                result.warnings.append(msg)
                continue
            result.records.extend(records)
            result.n_examples += 1
            result.audio_seconds += audio_s
            if progress:
                progress(i + 1, len(entries), str(entry.get("id", "")))
    else:
        cfg_dict = cfg.to_plain()
        log.info("evaluating on %d worker process(es), 1 torch thread each", n_workers)
        with ProcessPoolExecutor(
            max_workers=n_workers, initializer=_worker_init, initargs=(cfg_dict, method_list, 1)
        ) as pool:
            futures = {pool.submit(_worker_run, entry): entry for entry in entries}
            done = 0
            for future in as_completed(futures):
                flat, audio_s, error = future.result()
                done += 1
                if error:
                    log.warning("skipping %s", error)
                    result.warnings.append(error)
                else:
                    result.records.extend(MetricRecord(**_unflatten(row)) for row in flat)
                    result.n_examples += 1
                    result.audio_seconds += audio_s
                if progress:
                    progress(done, len(entries), str(futures[future].get("id", "")))
        # The model info is identical in every worker; report it from the parent.
        result.model_info = _model_info_for(cfg, method_list)

    result.wall_seconds = time.perf_counter() - t0
    log.info(
        "evaluated %d examples x %d methods in %.1f s (%.1f s of audio, %.1fx real time)",
        result.n_examples,
        len(result.methods),
        result.wall_seconds,
        result.audio_seconds,
        result.throughput,
    )
    return result


def _unflatten(row: dict[str, Any]) -> dict[str, Any]:
    """Rebuild MetricRecord kwargs from a flat row."""
    keys = ("example_id", "subset", "category", "noise_type", "snr_db", "method", "tap")
    return {
        **{k: row[k] for k in keys},
        "values": {k: v for k, v in row.items() if k not in keys},
    }


def _model_info_for(cfg: Config, methods: Sequence[str]) -> dict[str, Any]:
    if not any("dfn" in m for m in methods):
        return {}
    try:
        return DfnModel(cfg.neural, cfg.audio.sample_rate).load().info.as_dict()
    except Exception:  # pragma: no cover
        return {}
