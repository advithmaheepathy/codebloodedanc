"""Standalone live two-microphone NLMS noise canceller.

This is a self-contained path, deliberately separate from the delivered single-microphone
DeepFilterNet pipeline. It exists because the hardware for a genuine primary/reference
microphone pair became available (a Hollyland-style wireless kit whose two transmitters
arrive as the left and right channels of one stereo input device), and the problem
statement explicitly describes a "primary + reference microphone" adaptive-filter design.

Signal model (matches the reference MATLAB implementation)::

    primary  (d) = LEFT  channel  = speech + noise, from the mic on the talker
    reference(x) = RIGHT channel  = the noise, from the mic placed toward the noise source
    output   (e) = d - w^T x      = residual after subtracting the estimated noise = speech

The adaptive filter is the project's existing NLMS (``create_adaptive_filter``); this
module only adds the live two-channel capture, the block loop, and the metrics. It does
NOT import or run DeepFilterNet - it is NLMS only, by request.

There is no noise injection here and no neural stage: what the reference microphone
actually picks up is the reference. That is the whole point of having two real mics.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..audio.devices import channel_independence, check_stereo_input, device_name, resolve_device
from ..audio.io import dbfs
from ..audio.ringbuffer import RingBuffer
from ..config import Config
from ..dsp.nlms import create_adaptive_filter
from ..utils.logging import get_logger
from ..utils.timing import TimingRegistry

log = get_logger(__name__)

_EPS = 1e-20


@dataclass
class NlmsLiveStats:
    """Per-run counters and per-block time series for the report/dashboard."""

    blocks_in: int = 0
    blocks_out: int = 0
    output_underruns: int = 0
    callback_status_flags: list[str] = field(default_factory=list)

    primary_dbfs: list[float] = field(default_factory=list)
    reference_dbfs: list[float] = field(default_factory=list)
    output_dbfs: list[float] = field(default_factory=list)
    erle_db: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        def _m(v: list[float]) -> Optional[float]:
            arr = np.asarray([x for x in v if np.isfinite(x)], dtype=np.float64)
            return round(float(np.mean(arr)), 2) if arr.size else None

        return {
            "blocks_in": self.blocks_in,
            "blocks_out": self.blocks_out,
            "output_underruns": self.output_underruns,
            "callback_status_flags": list(dict.fromkeys(self.callback_status_flags)),
            "mean_primary_dbfs": _m(self.primary_dbfs),
            "mean_reference_dbfs": _m(self.reference_dbfs),
            "mean_output_dbfs": _m(self.output_dbfs),
            "mean_erle_db": _m(self.erle_db),
        }


class NlmsLiveEngine:
    """Captures a stereo microphone pair and runs block-wise NLMS in real time.

    The two channels of one input device are the two microphones: channel 0 (left) is the
    primary (speech + noise), channel 1 (right) is the noise reference.
    """

    def __init__(self, cfg: Config, primary_channel: int = 0, reference_channel: int = 1) -> None:
        self.cfg = cfg
        self.sr = cfg.audio.sample_rate
        self.block = cfg.live.blocksize
        self.primary_channel = int(primary_channel)
        self.reference_channel = int(reference_channel)
        self.stats = NlmsLiveStats()
        self.timings = TimingRegistry()

        # The adaptive filter, from the shared NLMS machinery. Safeguards on: they freeze
        # adaptation during double-talk and reference silence and roll back on divergence,
        # which is exactly what a live two-mic setup needs.
        self.filter = create_adaptive_filter(
            cfg.nlms, cfg.vad, self.sr, self.block, use_safeguards=True
        )

        capacity = max(int(cfg.live.ring_capacity_s * self.sr), 8 * self.block)
        self.in_ring = RingBuffer(capacity, channels=2)  # L=primary, R=reference
        self.out_ring = RingBuffer(capacity, channels=1)
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._monitor_gain = 10.0 ** (cfg.live.monitor_gain_db / 20.0)

        self.recorded_primary: list[np.ndarray] = []
        self.recorded_reference: list[np.ndarray] = []
        self.recorded_output: list[np.ndarray] = []

    # ---------------------------------------------------------------- callbacks
    def _input_callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            self.stats.callback_status_flags.append(str(status))
        # Keep both channels; the worker splits primary/reference.
        self.in_ring.write(indata[:, :2])
        self.stats.blocks_in += 1

    def _output_callback(self, outdata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            self.stats.callback_status_flags.append(str(status))
        data = self.out_ring.read(frames, timeout=0.0)
        if data is None:
            outdata[:] = 0.0
            self.stats.output_underruns += 1
        else:
            outdata[:, 0:1] = data * self._monitor_gain
            if outdata.shape[1] > 1:
                outdata[:, 1:] = outdata[:, 0:1]
        self.stats.blocks_out += 1

    # ------------------------------------------------------------------ worker
    def _worker_loop(self) -> None:
        scratch = np.empty((self.block, 2), dtype=np.float32)
        while not self._stop.is_set():
            got = self.in_ring.read(self.block, out=scratch, timeout=0.2)
            if got is None:
                continue
            primary = np.ascontiguousarray(got[:, self.primary_channel])
            reference = np.ascontiguousarray(got[:, self.reference_channel])
            self.recorded_primary.append(primary.copy())
            self.recorded_reference.append(reference.copy())

            timer = self.timings.timer("nlms")
            timer.start()
            # e = d - w^T x : the residual is the enhanced speech.
            out = self.filter.process_block(primary, reference)
            timer.stop(audio_s=self.block / self.sr)

            self.recorded_output.append(out.copy())
            self.stats.primary_dbfs.append(dbfs(primary))
            self.stats.reference_dbfs.append(dbfs(reference))
            self.stats.output_dbfs.append(dbfs(out))
            in_p = float(np.mean(primary.astype(np.float64) ** 2)) + _EPS
            out_p = float(np.mean(out.astype(np.float64) ** 2)) + _EPS
            self.stats.erle_db.append(10.0 * float(np.log10(in_p / out_p)))

            if self.cfg.live.monitor and out.size:
                self.out_ring.write(out.reshape(-1, 1))

    # -------------------------------------------------------------------- run
    def run(self, duration_s: float) -> None:
        import sounddevice as sd

        in_dev = resolve_device(self.cfg.live.input_device, "input")
        out_dev = (
            resolve_device(self.cfg.live.output_device, "output")
            if self.cfg.live.monitor
            else None
        )
        check_stereo_input(in_dev, self.sr)
        log.info("two-mic NLMS input device: %s (stereo)", device_name(in_dev, "input"))
        log.info(
            "primary = channel %d, reference = channel %d",
            self.primary_channel,
            self.reference_channel,
        )
        if self.cfg.live.monitor:
            log.warning(
                "USE HEADPHONES for monitoring: playing the output through speakers feeds it "
                "back into the microphones and will howl."
            )

        self._worker = threading.Thread(target=self._worker_loop, name="nlms-worker", daemon=True)
        self._worker.start()

        streams: list[Any] = []
        try:
            streams.append(
                sd.InputStream(
                    samplerate=self.sr, blocksize=self.block, device=in_dev, channels=2,
                    dtype="float32", callback=self._input_callback,
                )
            )
            if self.cfg.live.monitor:
                streams.append(
                    sd.OutputStream(
                        samplerate=self.sr, blocksize=self.block, device=out_dev, channels=1,
                        dtype="float32", callback=self._output_callback,
                    )
                )
            for s in streams:
                s.start()
            time.sleep(max(0.0, duration_s))
        finally:
            self._stop.set()
            for s in streams:
                try:
                    s.stop()
                    s.close()
                except Exception:  # pragma: no cover - teardown best effort
                    pass
            self.in_ring.close()
            if self._worker is not None:
                self._worker.join(timeout=2.0)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _join(chunks: list[np.ndarray]) -> np.ndarray:
        return (
            np.concatenate(chunks).astype(np.float32)
            if chunks
            else np.zeros(0, dtype=np.float32)
        )

    def signals(self) -> dict[str, np.ndarray]:
        """The three recorded streams, equal length, for playback and plotting."""
        primary = self._join(self.recorded_primary)
        reference = self._join(self.recorded_reference)
        output = self._join(self.recorded_output)
        n = min(len(primary), len(reference), len(output)) if output.size else len(primary)
        return {
            "primary": primary[:n],
            "reference": reference[:n],
            "output": output[:n],
        }

    def metrics(self) -> dict[str, Any]:
        """Reference-free live metrics, plus channel-independence sanity check."""
        sig = self.signals()
        stereo = np.stack(
            [sig["primary"], sig["reference"]], axis=1
        ) if sig["primary"].size else np.zeros((0, 2))
        indep = channel_independence(stereo)

        stats = self.stats.as_dict()
        primary, output = sig["primary"], sig["output"]
        n = min(len(primary), len(output))
        overall_erle = float("nan")
        if n:
            in_p = float(np.mean(primary[:n].astype(np.float64) ** 2)) + _EPS
            out_p = float(np.mean(output[:n].astype(np.float64) ** 2)) + _EPS
            overall_erle = 10.0 * float(np.log10(in_p / out_p))

        summary = self.filter.diagnostics.summary() if hasattr(self.filter, "diagnostics") else {}
        return {
            "channel_correlation": round(indep["correlation"], 4),
            "primary_dbfs": round(indep["left_dbfs"], 1),
            "reference_dbfs": round(indep["right_dbfs"], 1),
            "overall_erle_db": round(overall_erle, 2) if np.isfinite(overall_erle) else None,
            "stats": stats,
            "nlms": summary,
        }
