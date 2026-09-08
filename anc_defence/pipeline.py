"""Stage orchestration.

Delivered pipeline (single microphone):

    capture -> preprocess -> DeepFilterNet3 -> volume normalisation -> output

``preprocess`` is DC removal plus an 80 Hz high-pass. Audio is carried at 48 kHz
because the model is a 48 kHz model; 16 kHz sources are upsampled on load.

Why there is no adaptive filter in this path
--------------------------------------------
An NLMS stage needs a reference microphone carrying noise that is correlated with the
noise in the primary channel. With one microphone there is no such signal. The two
placements that were tried are both dead ends, and both are still measurable through
this module so the report can show the numbers rather than assert them:

* ``dfn_then_nlms`` - the adaptive filter runs on the model's output. The model
  applies a time-varying, non-linear gain, so the linear relationship between the
  reference and the residual is broken and the filter damages the speech.
* ``nlms_then_dfn`` - works, but requires a second physical microphone, which the
  single-microphone problem framing rules out.

Both are retained as ``REJECTED`` orders and are evaluated using the corpus's
noise-only file as the reference. That is the most favourable reference possible -
the exact noise, sample-aligned, which no real system could obtain - so the result is
an upper bound on what the rejected designs could achieve.

Metrics are taken at every tap point, so each stage's contribution is separable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from .audio.io import match_length
from .config import Config
from .dsp.nlms import NlmsResult, create_adaptive_filter
from .dsp.normalise import Normaliser
from .dsp.preprocess import Preprocessor
from .enhance.streaming import ChunkedEnhancer, DfnModel
from .utils.logging import get_logger
from .utils.timing import TimingRegistry

log = get_logger(__name__)

DFN = "deepfilternet"
NORMALISE = "normalise"
NLMS = "nlms"

# Stage sequence per configured order.
STAGE_SEQUENCE: dict[str, tuple[str, ...]] = {
    "dfn_then_normalise": (DFN, NORMALISE),
    "dfn_only": (DFN,),
    "normalise_only": (NORMALISE,),
    "passthrough": (),
    # Rejected two-microphone designs, kept only so they can be measured.
    "nlms_then_dfn": (NLMS, DFN),
    "dfn_then_nlms": (DFN, NLMS),
    "nlms_only": (NLMS,),
}

REJECTED_ORDERS = ("nlms_then_dfn", "dfn_then_nlms", "nlms_only")

STAGE_LABELS = {
    DFN: "deepfilternet",
    NORMALISE: "normalise",
    NLMS: "nlms",
}


@dataclass
class PipelineResult:
    """Signals at every tap point plus everything measured on the way through."""

    primary: np.ndarray
    output: np.ndarray
    stage_outputs: dict[str, np.ndarray] = field(default_factory=dict)
    stages: tuple[str, ...] = ()
    order: str = "dfn_then_normalise"
    sample_rate: int = 48000
    reference: Optional[np.ndarray] = None
    nlms: Optional[NlmsResult] = None
    normalise_summary: dict[str, Any] = field(default_factory=dict)
    timings: TimingRegistry = field(default_factory=TimingRegistry)
    model_info: dict[str, Any] = field(default_factory=dict)
    latency_ms: dict[str, float] = field(default_factory=dict)
    guard_counters: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return len(self.primary) / self.sample_rate

    @property
    def after_stage1(self) -> np.ndarray:
        """First stage output, or the input when the pipeline is a pass-through."""
        if self.stages:
            return self.stage_outputs[f"after_{STAGE_LABELS[self.stages[0]]}"]
        return self.primary

    @property
    def stage1(self) -> Optional[str]:
        return STAGE_LABELS[self.stages[0]] if self.stages else None

    @property
    def stage2(self) -> Optional[str]:
        return STAGE_LABELS[self.stages[1]] if len(self.stages) > 1 else None

    @property
    def erle_curve(self) -> np.ndarray:
        return self.nlms.erle_curve if self.nlms is not None else np.zeros(0)

    def taps(self, include_reference: bool = True) -> dict[str, np.ndarray]:
        """Named tap points in signal-flow order."""
        out: dict[str, np.ndarray] = {"input": self.primary}
        if include_reference and self.reference is not None:
            out["reference"] = self.reference
        out.update(self.stage_outputs)
        if not self.stages:
            out["output"] = self.output
        return out

    def removed(self) -> np.ndarray:
        """What the pipeline took out: input minus output, level-matched.

        Useful in the dashboard as a playable track: if speech is audible in here, the
        pipeline is removing something it should not be.
        """
        n = min(len(self.primary), len(self.output))
        a = self.primary[:n].astype(np.float64)
        b = self.output[:n].astype(np.float64)
        denom = float(np.dot(b, b))
        alpha = float(np.dot(a, b)) / denom if denom > 1e-20 else 1.0
        return (a - alpha * b).astype(np.float32)


class Pipeline:
    """The two-stage enhancement pipeline."""

    def __init__(
        self,
        cfg: Config,
        model: Optional[DfnModel] = None,
        load_model: bool = True,
    ) -> None:
        self.cfg = cfg
        self.sample_rate = cfg.audio.sample_rate
        self.block_size = cfg.audio.hop_size
        self.order = cfg.pipeline.order
        self.stages = STAGE_SEQUENCE[self.order]
        self.timings = TimingRegistry()

        self.pre_primary = Preprocessor(cfg.preprocess, self.sample_rate, "primary")
        self.pre_reference = Preprocessor(cfg.preprocess, self.sample_rate, "reference")

        self.model: Optional[DfnModel] = None
        self.enhancer: Optional[ChunkedEnhancer] = None
        if DFN in self.stages and cfg.neural.enabled and load_model:
            self.model = model or DfnModel(cfg.neural, self.sample_rate).load()
            self.enhancer = ChunkedEnhancer(
                model=self.model,
                chunk_s=cfg.neural.streaming.chunk_s,
                overlap=cfg.neural.streaming.overlap,
                crossfade_ms=cfg.neural.streaming.crossfade_ms,
                sample_rate=self.sample_rate,
            )

        self.normaliser: Optional[Normaliser] = None
        if NORMALISE in self.stages:
            self.normaliser = Normaliser(cfg.normalise, cfg.vad, self.sample_rate, self.block_size)

        self.nlms = None
        if NLMS in self.stages:
            if self.order in REJECTED_ORDERS:
                log.info(
                    "pipeline order '%s' is a rejected two-microphone design, retained for "
                    "measurement only; it is not the delivered pipeline",
                    self.order,
                )
            self.nlms = create_adaptive_filter(
                cfg.nlms, cfg.vad, self.sample_rate, self.block_size, use_safeguards=True
            )

    # ------------------------------------------------------------------ helpers
    def reset(self) -> None:
        self.pre_primary.reset()
        self.pre_reference.reset()
        if self.nlms is not None:
            self.nlms.reset()
        if self.enhancer is not None:
            self.enhancer.reset()
        if self.normaliser is not None:
            self.normaliser.reset()

    def _run_neural(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            return x
        timer = self.timings.timer("neural")
        timer.start()
        if self.cfg.neural.streaming.offline_framing == "chunked" and self.enhancer is not None:
            y = self.enhancer.process_signal(x)
        else:
            y = self.model.enhance_array(x, count_time=False)
        timer.stop(audio_s=len(x) / self.sample_rate)
        return y

    def _run_normalise(self, x: np.ndarray) -> np.ndarray:
        if self.normaliser is None or not self.normaliser.enabled:
            return x
        timer = self.timings.timer("normalise")
        timer.start()
        y = self.normaliser.process(x)
        timer.stop(audio_s=len(x) / self.sample_rate)
        return y

    def _run_nlms(
        self, d: np.ndarray, x: Optional[np.ndarray]
    ) -> tuple[np.ndarray, Optional[NlmsResult]]:
        if self.nlms is None or x is None:
            if self.nlms is not None and x is None:
                log.warning(
                    "order '%s' includes an adaptive stage but no reference channel was supplied; "
                    "it is a pass-through for this run",
                    self.order,
                )
            return d, None
        timer = self.timings.timer("nlms")
        timer.start()
        d2, x2 = match_length(d, x)
        result = self.nlms.process(d2, x2)
        timer.stop(audio_s=len(result.output) / self.sample_rate)
        return result.output, result

    # ------------------------------------------------------------------ offline
    def process(
        self, primary: np.ndarray, reference: Optional[np.ndarray] = None
    ) -> PipelineResult:
        """Run the configured pipeline over whole signals."""
        self.reset()
        notes: list[str] = []

        timer = self.timings.timer("preprocess")
        timer.start()
        signal = self.pre_primary.process(np.asarray(primary, dtype=np.float32))
        ref = (
            self.pre_reference.process(np.asarray(reference, dtype=np.float32))
            if reference is not None
            else None
        )
        timer.stop(audio_s=len(signal) / self.sample_rate)
        input_tap = signal.copy()

        nlms_result: Optional[NlmsResult] = None
        stage_outputs: dict[str, np.ndarray] = {}
        for stage in self.stages:
            if stage == DFN:
                signal = self._run_neural(signal)
            elif stage == NORMALISE:
                signal = self._run_normalise(signal)
            elif stage == NLMS:
                signal, nlms_result = self._run_nlms(signal, ref)
                if self.order == "dfn_then_nlms":
                    notes.append(
                        "the adaptive filter is operating on the model's output, whose relationship "
                        "to the reference is no longer linear"
                    )
            stage_outputs[f"after_{STAGE_LABELS[stage]}"] = signal.copy()

        # Every tap must share a length for metrics and the multichannel export.
        lengths = [len(input_tap), len(signal)] + [len(v) for v in stage_outputs.values()]
        n = min(lengths)
        input_tap = input_tap[:n]
        signal = signal[:n]
        stage_outputs = {k: v[:n] for k, v in stage_outputs.items()}

        return PipelineResult(
            primary=input_tap,
            output=signal,
            stage_outputs=stage_outputs,
            stages=self.stages,
            order=self.order,
            sample_rate=self.sample_rate,
            reference=ref[:n] if ref is not None else None,
            nlms=nlms_result,
            normalise_summary=self.normaliser.summary() if self.normaliser else {},
            timings=self.timings,
            model_info=self.model.info.as_dict() if self.model else {},
            latency_ms=self.latency_breakdown(),
            guard_counters={
                **{f"primary_{k}": v for k, v in self.pre_primary.counters.as_dict().items()},
                **{f"reference_{k}": v for k, v in self.pre_reference.counters.as_dict().items()},
            },
            notes=notes,
        )

    # ------------------------------------------------------------------ streaming
    def process_block(
        self, block: np.ndarray, reference_block: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Live path: one hop in, however many samples are ready out.

        The neural stage buffers into chunks, so this returns an empty array until a
        chunk completes and then a burst of ``chunk * (1 - overlap)`` samples.
        """
        signal = self.pre_primary.process(np.asarray(block, dtype=np.float32))
        for stage in self.stages:
            if stage == DFN:
                signal = self.enhancer.push(signal) if self.enhancer is not None else signal
            elif stage == NORMALISE:
                if signal.size and self.normaliser is not None and self.normaliser.enabled:
                    signal = self.normaliser.process(signal)
            elif stage == NLMS:
                if reference_block is None or self.nlms is None:
                    continue
                ref = self.pre_reference.process(np.asarray(reference_block, dtype=np.float32))
                signal = self.nlms.process_block(signal, ref)
            if signal.size == 0:
                return signal
        return signal

    def flush(self) -> np.ndarray:
        """Drain buffered audio at the end of a live run."""
        tail = self.enhancer.flush() if self.enhancer is not None else np.zeros(0, dtype=np.float32)
        if tail.size and NORMALISE in self.stages and self.normaliser is not None:
            tail = self.normaliser.process(tail)
        return tail

    # ----------------------------------------------------------------- latency
    def latency_breakdown(self) -> dict[str, float]:
        """Theoretical latency budget. Measured figures come from a loopback test."""
        hop_ms = 1000.0 * self.block_size / self.sample_rate
        out: dict[str, float] = {"block_ms": hop_ms}
        if self.model is not None:
            out.update(self.model.info.latency_breakdown_ms())
        if self.enhancer is not None:
            out["chunk_buffer_ms"] = self.enhancer.latency_ms
        if self.normaliser is not None and self.normaliser.enabled:
            out["normalise_lookahead_ms"] = self.normaliser.latency_ms
        if self.nlms is not None:
            out["nlms_ms"] = 0.0
        out["total_theoretical_ms"] = (
            out.get("block_ms", 0.0)
            + out.get("total_algorithmic_ms", 0.0)
            + out.get("chunk_buffer_ms", 0.0)
            + out.get("normalise_lookahead_ms", 0.0)
            + out.get("nlms_ms", 0.0)
        )
        return {k: round(v, 3) for k, v in out.items()}

    def describe(self) -> str:
        parts = ["capture", "preprocess"] + [STAGE_LABELS[s] for s in self.stages] + ["output"]
        return " -> ".join(parts)

    @property
    def is_rejected_design(self) -> bool:
        return self.order in REJECTED_ORDERS
