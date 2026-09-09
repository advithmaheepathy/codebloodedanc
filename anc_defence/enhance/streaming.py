"""Neural stage: pretrained DeepFilterNet3, wrapped for block-based use.

Model facts, read from the installed checkpoint's configuration rather than assumed:

===========================  ==========================================
sample rate / hop / FFT      48000 Hz / 480 (10 ms) / 960 (20 ms)
parameters                   2,135,484
``conv_lookahead``           2 frames
``df_lookahead``             2 frames
deep filter order            5
===========================  ==========================================

Algorithmic latency
-------------------
``deepfilternet3.py`` shifts the encoder features by ``conv_lookahead`` and the
spectrogram by ``df_lookahead``, and asserts ``conv_lookahead >= df_lookahead``.
Both shifts act on the same time axis, so they do **not** add: the output at frame
``t`` depends on input up to frame ``t+2``. The STFT/ISTFT loop adds ``n_fft - hop``
as noted in ``df/enhance.py``. Therefore:

    10 ms (frame)  +  10 ms (n_fft - hop)  +  20 ms (2-frame lookahead)  =  40 ms

That is the model's floor, independent of implementation, and it matches the 40 ms
reported in the DeepFilterNet2 paper for this architecture.

Framing strategies
------------------
``ChunkedEnhancer``
    Overlapping chunks through the offline enhancer with a crossfade at the seams.
    One inference implementation, one set of weights, no model surgery, and it cannot
    fail. Its latency is one chunk, which is **not** low latency: it is a demo path
    and the report labels it as such.

``PerHopEnhancer``
    True frame-by-frame inference with persistent recurrent and convolution state.
    Not implemented: see the class docstring for what it requires and why it was not
    attempted inside this project's time budget.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..config import NeuralCfg
from ..utils.logging import get_logger
from ..utils.timing import StageTimer

log = get_logger(__name__)


@dataclass
class ModelInfo:
    """Everything the report needs to identify the model and its latency."""

    name: str = "DeepFilterNet3"
    checkpoint: str = ""
    epoch: Optional[int] = None
    sample_rate: int = 48000
    hop_size: int = 480
    fft_size: int = 960
    n_parameters: int = 0
    conv_lookahead: int = 0
    df_lookahead: int = 0
    df_order: int = 0
    nb_erb: int = 0
    nb_df: int = 0
    device: str = "cpu"
    num_threads: Optional[int] = None
    atten_lim_db: Optional[float] = None
    post_filter: bool = False
    fine_tuned: bool = False

    @property
    def lookahead_frames(self) -> int:
        """Not the sum: both lookaheads shift the same time axis."""
        return max(self.conv_lookahead, self.df_lookahead)

    @property
    def algorithmic_latency_ms(self) -> float:
        frame_ms = 1000.0 * self.hop_size / self.sample_rate
        stft_ms = 1000.0 * (self.fft_size - self.hop_size) / self.sample_rate
        return frame_ms + stft_ms + self.lookahead_frames * frame_ms

    def latency_breakdown_ms(self) -> dict[str, float]:
        frame_ms = 1000.0 * self.hop_size / self.sample_rate
        return {
            "frame_ms": frame_ms,
            "stft_istft_ms": 1000.0 * (self.fft_size - self.hop_size) / self.sample_rate,
            "model_lookahead_ms": self.lookahead_frames * frame_ms,
            "total_algorithmic_ms": self.algorithmic_latency_ms,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "checkpoint": self.checkpoint,
            "epoch": self.epoch,
            "fine_tuned": self.fine_tuned,
            "sample_rate": self.sample_rate,
            "hop_size": self.hop_size,
            "fft_size": self.fft_size,
            "n_parameters": self.n_parameters,
            "conv_lookahead_frames": self.conv_lookahead,
            "df_lookahead_frames": self.df_lookahead,
            "effective_lookahead_frames": self.lookahead_frames,
            "df_order": self.df_order,
            "nb_erb": self.nb_erb,
            "nb_df": self.nb_df,
            "device": self.device,
            "num_threads": self.num_threads,
            "atten_lim_db": self.atten_lim_db,
            "post_filter": self.post_filter,
            "algorithmic_latency_ms": round(self.algorithmic_latency_ms, 2),
            "latency_breakdown_ms": {
                k: round(v, 2) for k, v in self.latency_breakdown_ms().items()
            },
        }


class DfnModel:
    """Thin wrapper around the pretrained DeepFilterNet3 inference path.

    This is the only place inference happens; every framing strategy calls
    :meth:`enhance_array`, so there is exactly one implementation of the neural
    stage and no risk of the offline and live paths diverging.
    """

    def __init__(self, cfg: NeuralCfg, sample_rate: int = 48000) -> None:
        self.cfg = cfg
        self.sample_rate = sample_rate
        self._model = None
        self._df_state = None
        self._torch = None
        self.info = ModelInfo()
        self.timer = StageTimer("neural")
        self._loaded = False

    # ------------------------------------------------------------------- load
    def load(self) -> "DfnModel":
        if self._loaded:
            return self
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("PyTorch is required for the neural stage") from exc
        self._torch = torch

        if self.cfg.num_threads is not None:
            torch.set_num_threads(int(self.cfg.num_threads))

        device = self.cfg.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "neural.device='cuda' but torch.cuda.is_available() is False. Install a CUDA "
                "build of PyTorch, or set neural.device=cpu. This machine's CPU real-time "
                "factor is well under 1.0, so CPU is a valid choice."
            )

        # df.enhance picks its own device via df.utils.get_device(), which defaults to
        # cuda:0 whenever torch.cuda.is_available() is True - independent of where we put
        # the model with model.to(device) below. On a CUDA machine with neural.device=cpu
        # this puts the weights on CPU but feeds them CUDA tensors:
        # "Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should
        # be the same". get_device() reads the DEVICE env var first if present, so setting
        # it here forces the library to agree with our own device choice. This has to
        # happen before the first `import df.*`, because df.utils evaluates torch.device()
        # lazily but other df modules may cache torch.cuda state at import time.
        os.environ["DEVICE"] = device

        from df.enhance import init_df

        base_dir = str(self.cfg.model_base_dir) if self.cfg.model_base_dir else None
        model, df_state, _ = init_df(
            model_base_dir=base_dir,
            post_filter=self.cfg.post_filter,
            log_level="warning",
            config_allow_defaults=True,
            default_model=self.cfg.model,
        )
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)

        self._model = model
        self._df_state = df_state

        from df.model import ModelParams

        params = ModelParams()
        self.info = ModelInfo(
            name=self.cfg.model,
            checkpoint=base_dir or "packaged pretrained weights",
            sample_rate=df_state.sr(),
            hop_size=df_state.hop_size(),
            fft_size=df_state.fft_size(),
            n_parameters=sum(p.numel() for p in model.parameters()),
            conv_lookahead=int(getattr(params, "conv_lookahead", 0)),
            df_lookahead=int(getattr(params, "df_lookahead", 0)),
            df_order=int(getattr(params, "df_order", 0)),
            nb_erb=int(getattr(params, "nb_erb", 0)),
            nb_df=int(getattr(params, "nb_df", 0)),
            device=device,
            num_threads=torch.get_num_threads(),
            atten_lim_db=self.cfg.atten_lim_db,
            post_filter=self.cfg.post_filter,
            fine_tuned=False,
        )
        if self.info.sample_rate != self.sample_rate:
            raise RuntimeError(
                f"model sample rate {self.info.sample_rate} does not match the pipeline rate "
                f"{self.sample_rate}"
            )
        self._loaded = True
        log.info(
            "loaded %s on %s: %.2fM parameters, %d threads, algorithmic latency %.0f ms",
            self.info.name,
            device,
            self.info.n_parameters / 1e6,
            self.info.num_threads or 0,
            self.info.algorithmic_latency_ms,
        )
        if self.cfg.warmup_frames > 0:
            self.warmup(self.cfg.warmup_frames)
        return self

    def warmup(self, n_frames: int = 10) -> float:
        """Run a throwaway inference. The first call is much slower than the rest."""
        n = max(1, n_frames) * self.info.hop_size
        dummy = np.zeros(n, dtype=np.float32)
        import time

        t0 = time.perf_counter()
        self._enhance_raw(dummy)
        dt = (time.perf_counter() - t0) * 1e3
        log.debug("model warmup took %.1f ms", dt)
        return dt

    # ---------------------------------------------------------------- inference
    def _enhance_raw(self, x: np.ndarray) -> np.ndarray:
        torch = self._torch
        assert torch is not None and self._model is not None
        from df.enhance import enhance

        with torch.inference_mode():
            tensor = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(0)
            if self.info.device == "cuda":
                tensor = tensor.pin_memory().to("cuda", non_blocking=True)
            out = enhance(
                self._model,
                self._df_state,
                tensor,
                pad=True,
                atten_lim_db=self.cfg.atten_lim_db,
            )
            return out.squeeze(0).detach().float().cpu().numpy()

    def enhance_array(self, x: np.ndarray, count_time: bool = True) -> np.ndarray:
        """Enhance a whole array. ``pad=True`` keeps the output time-aligned."""
        if not self._loaded:
            self.load()
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.size == 0:
            return x
        if count_time:
            self.timer.start()
        y = self._enhance_raw(x)
        if count_time:
            self.timer.stop(audio_s=len(x) / self.sample_rate)
        if len(y) < len(x):
            y = np.pad(y, (0, len(x) - len(y)))
        return np.ascontiguousarray(y[: len(x)], dtype=np.float32)

    def rtf(self) -> float:
        return self.timer.summary().rtf


# ------------------------------------------------------------------- chunked


@dataclass
class ChunkedEnhancer:
    """Overlapping-chunk framing with a crossfade at the seams.

    Latency is one chunk in the worst case and ``chunk - hop`` in the best case,
    plus processing time. This is a demo path, not a low-latency path.
    """

    model: DfnModel
    chunk_s: float = 1.5
    overlap: float = 0.5
    crossfade_ms: float = 30.0
    sample_rate: int = 48000
    context_s: float = 0.5

    _buffer: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    _tail: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    _context: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    _primed: bool = False
    chunks_processed: int = 0

    def __post_init__(self) -> None:
        self.chunk_samples = max(self.model.info.hop_size, int(round(self.chunk_s * self.sample_rate)))
        self.hop_samples = max(1, int(round(self.chunk_samples * (1.0 - self.overlap))))
        self.context_samples = max(0, int(round(self.context_s * self.sample_rate)))
        self.ramp_samples = min(
            self.hop_samples, max(0, int(round(self.crossfade_ms * self.sample_rate / 1000.0)))
        )
        self._fade_in = (
            np.linspace(0.0, 1.0, self.ramp_samples, dtype=np.float32)
            if self.ramp_samples > 0
            else np.zeros(0, dtype=np.float32)
        )
        self._fade_out = 1.0 - self._fade_in

    # ------------------------------------------------------------------ props
    @property
    def latency_ms(self) -> float:
        """Worst-case buffering latency introduced by the chunking, in ms."""
        return 1000.0 * self.chunk_samples / self.sample_rate

    def latency_breakdown_ms(self) -> dict[str, float]:
        model = self.model.info.latency_breakdown_ms()
        out = dict(model)
        out["chunk_buffer_ms"] = self.latency_ms
        out["chunk_hop_ms"] = 1000.0 * self.hop_samples / self.sample_rate
        out["total_algorithmic_plus_buffer_ms"] = model["total_algorithmic_ms"] + self.latency_ms
        return out

    def reset(self) -> None:
        self._buffer = np.zeros(0, dtype=np.float32)
        self._tail = np.zeros(0, dtype=np.float32)
        self._context = np.zeros(0, dtype=np.float32)
        self._primed = False
        self.chunks_processed = 0

    # ------------------------------------------------------------------ stream
    def push(self, block: np.ndarray) -> np.ndarray:
        """Feed input; returns however many output samples are ready (possibly none)."""
        self._buffer = np.concatenate([self._buffer, np.asarray(block, dtype=np.float32)])
        out_parts: list[np.ndarray] = []
        while len(self._buffer) >= self.chunk_samples:
            chunk = self._buffer[: self.chunk_samples]
            self._buffer = self._buffer[self.hop_samples :]
            out_parts.append(self._process_chunk(chunk))
        return (
            np.concatenate(out_parts).astype(np.float32)
            if out_parts
            else np.zeros(0, dtype=np.float32)
        )

    def _process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        # Prepend previously-seen audio as context so the model's recurrent state is
        # warm for the samples we keep, then discard the context region from the output.
        # A cold state at every chunk boundary is what limits agreement with the
        # reference offline path.
        if self.context_samples > 0 and self._context.size:
            context = self._context[-self.context_samples :]
            y_full = self.model.enhance_array(np.concatenate([context, chunk]))
            y = y_full[len(context) :]
        else:
            y = self.model.enhance_array(chunk)
        if self.context_samples > 0:
            self._context = chunk[-self.context_samples :].copy()
        self.chunks_processed += 1
        emit = y[: self.hop_samples].copy()
        if self._primed and self.ramp_samples > 0 and len(self._tail) >= self.ramp_samples:
            emit[: self.ramp_samples] = (
                self._tail[: self.ramp_samples] * self._fade_out
                + emit[: self.ramp_samples] * self._fade_in
            )
        self._tail = y[self.hop_samples :].copy()
        self._primed = True
        return emit

    def flush(self) -> np.ndarray:
        """Drain whatever is left at the end of a run."""
        out_parts: list[np.ndarray] = []
        if len(self._buffer) > 0:
            padded = np.zeros(self.chunk_samples, dtype=np.float32)
            n = min(len(self._buffer), self.chunk_samples)
            padded[:n] = self._buffer[:n]
            if self.context_samples > 0 and self._context.size:
                context = self._context[-self.context_samples :]
                y = self.model.enhance_array(np.concatenate([context, padded]))[len(context) :]
            else:
                y = self.model.enhance_array(padded)
            self.chunks_processed += 1
            emit = y[:n].copy()
            if self._primed and self.ramp_samples > 0 and len(self._tail) >= self.ramp_samples:
                k = min(self.ramp_samples, len(emit))
                emit[:k] = self._tail[:k] * self._fade_out[:k] + emit[:k] * self._fade_in[:k]
            out_parts.append(emit)
            self._buffer = np.zeros(0, dtype=np.float32)
        elif self._primed and len(self._tail) > 0:
            out_parts.append(self._tail.copy())
        self._tail = np.zeros(0, dtype=np.float32)
        return (
            np.concatenate(out_parts).astype(np.float32)
            if out_parts
            else np.zeros(0, dtype=np.float32)
        )

    # ----------------------------------------------------------------- offline
    def process_signal(self, x: np.ndarray) -> np.ndarray:
        """Run a whole signal through the chunked path (used to verify the live path)."""
        self.reset()
        out = self.push(x)
        out = np.concatenate([out, self.flush()])
        if len(out) < len(x):
            out = np.pad(out, (0, len(x) - len(out)))
        return out[: len(x)].astype(np.float32)


class PerHopEnhancer:
    """True per-hop streaming inference. Not implemented.

    What it would take: DeepFilterNet3's Python forward pass consumes a whole
    spectrogram and its GRUs run over the time axis in one call, so single-frame
    operation needs (a) hidden state threaded through every recurrent layer, (b) ring
    buffers for each convolution's time context, and (c) state for the deep-filter
    operation's 5-frame window. The upstream project implements this in Rust
    (``deep-filter``, the LADSPA plugin) but exposes only offline ``enhance()`` from
    Python.

    That is a substantial piece of work with a real chance of silently diverging from
    the reference implementation, so it was explicitly de-scoped in favour of the
    chunked path, which reuses the reference inference code unmodified. If it is
    built later, the gate is numerical agreement with :meth:`DfnModel.enhance_array`
    on a test file within a stated tolerance.
    """

    def __init__(self, *_: object, **__: object) -> None:
        raise NotImplementedError(
            "neural.streaming.mode='per_hop' is not implemented. Use 'chunked'.\n"
            "Reason: DeepFilterNet's Python API only exposes offline enhance(); true "
            "frame-by-frame operation requires threading GRU hidden state and convolution "
            "context buffers through the model, which was de-scoped. The chunked path uses "
            "the reference inference code unmodified and its latency is reported honestly "
            "as one chunk."
        )


def create_enhancer(
    cfg: NeuralCfg, sample_rate: int = 48000, model: Optional[DfnModel] = None
) -> ChunkedEnhancer:
    """Build the configured framing strategy around a loaded model."""
    dfn = model or DfnModel(cfg, sample_rate).load()
    if cfg.streaming.mode == "per_hop":
        PerHopEnhancer()  # raises with the explanation above
    return ChunkedEnhancer(
        model=dfn,
        chunk_s=cfg.streaming.chunk_s,
        overlap=cfg.streaming.overlap,
        crossfade_ms=cfg.streaming.crossfade_ms,
        sample_rate=sample_rate,
        context_s=cfg.streaming.context_s,
    )


def chunk_equivalence(
    model: DfnModel,
    x: np.ndarray,
    chunk_s: float,
    overlap: float = 0.5,
    crossfade_ms: float = 30.0,
    sample_rate: int = 48000,
    context_s: float = 0.5,
) -> dict[str, float]:
    """Compare chunked output against whole-file output on the same input.

    Used to choose the chunk size from measurement rather than by assumption: the
    reported ``si_sdr_vs_offline`` is how closely the chunked path reproduces the
    reference result, in dB.
    """
    from ..metrics.intrusive import si_sdr

    reference = model.enhance_array(x, count_time=False)
    enhancer = ChunkedEnhancer(
        model=model,
        chunk_s=chunk_s,
        overlap=overlap,
        crossfade_ms=crossfade_ms,
        sample_rate=sample_rate,
        context_s=context_s,
    )
    chunked = enhancer.process_signal(x)
    n = min(len(reference), len(chunked))
    return {
        "chunk_s": chunk_s,
        "overlap": overlap,
        "context_s": context_s,
        "si_sdr_vs_offline_db": si_sdr(reference[:n], chunked[:n]),
        "worst_case_latency_ms": enhancer.latency_ms,
        "chunks": enhancer.chunks_processed,
        "rtf": model.timer.summary().rtf,
    }
