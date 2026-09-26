"""Live single-microphone mode.

    microphone -> preprocess -> DeepFilterNet3 -> volume normalisation -> headphones

Optionally a noise file can be mixed into the captured signal digitally at a chosen
SNR (``--noise-file``). That is a demo aid: it makes the effect obvious in a quiet
room and, because the pre-mix microphone recording is kept, it is the only live
configuration where PESQ, STOI and SI-SDR can be computed.

Real-time discipline
--------------------
The audio callbacks only copy samples into preallocated ring buffers and bump
counters. No allocation, no locks beyond the ring's wait/notify, no logging and no
file I/O in a callback. All DSP runs on a worker thread.

Latency
-------
The neural stage runs in chunked framing, so output lags input by roughly one chunk
(1.5 s by default) plus the model's 40 ms algorithmic latency, plus 5 ms of limiter
look-ahead, plus device buffering. This is a demo path. It is not low-latency
streaming and the report does not claim it is.
"""

from __future__ import annotations

import collections
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from ..audio.devices import check_settings, device_name, resolve_device
from ..audio.io import dbfs, load_audio
from ..audio.ringbuffer import RingBuffer
from ..config import Config
from ..dsp.vad import speech_mask
from ..metrics.erle import noise_reduction_db
from ..metrics.intrusive import compute_intrusive
from ..metrics.nonintrusive import DnsmosEstimator, estimate_snr_db
from ..metrics.system import collect_system_metrics
from ..pipeline import Pipeline
from ..report import plots
from ..report.export import ReportData
from ..utils.logging import get_logger
from ..utils.platform_info import ResourceSampler
from ..utils.session import Session, create_session
from ..utils.timing import TimingRegistry
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


@dataclass
class LiveStats:
    blocks_in: int = 0
    blocks_out: int = 0
    input_dbfs: list[float] = field(default_factory=list)
    output_dbfs: list[float] = field(default_factory=list)
    callback_status_flags: list[str] = field(default_factory=list)
    output_underruns: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "blocks_in": self.blocks_in,
            "blocks_out": self.blocks_out,
            "output_underruns": self.output_underruns,
            "callback_status_flags": list(dict.fromkeys(self.callback_status_flags)),
            "mean_input_dbfs": round(float(np.mean(self.input_dbfs)), 2) if self.input_dbfs else None,
            "mean_output_dbfs": round(float(np.mean(self.output_dbfs)), 2) if self.output_dbfs else None,
        }


class _DemoDelayQueue:
    """Holds already-processed output blocks for a fixed extra delay before release.

    This exists purely so a live demo (in particular a single-take video recording)
    can separate the noisy input and the enhanced output in time, without which they
    audibly/visually overlap at the pipeline's real (sub-second) latency. It does not
    touch the samples at all: a block that entered the pipeline is bit-identical to
    the block that eventually comes out of this queue. Only the *release time* is
    delayed, by holding each block until ``delay_s`` seconds have elapsed since it was
    produced.

    With ``delay_s == 0`` this is a no-op passthrough (verified by the queue draining
    immediately), so the real minimum-latency figure the report computes is completely
    unaffected by whether this feature is used.
    """

    def __init__(self, delay_s: float, sample_rate: int) -> None:
        self.delay_s = max(0.0, float(delay_s))
        self.sample_rate = sample_rate
        self._queue: collections.deque[tuple[float, np.ndarray]] = collections.deque()

    def push(self, block: np.ndarray, now: Optional[float] = None) -> None:
        if block.size == 0:
            return
        self._queue.append(((now if now is not None else time.perf_counter()), block))

    def pop_ready(self, now: Optional[float] = None) -> list[np.ndarray]:
        """Return, in order, every block whose hold time has elapsed."""
        if self.delay_s <= 0.0:
            out = [b for _, b in self._queue]
            self._queue.clear()
            return out
        now = now if now is not None else time.perf_counter()
        ready: list[np.ndarray] = []
        while self._queue and (now - self._queue[0][0]) >= self.delay_s:
            ready.append(self._queue.popleft()[1])
        return ready

    def flush(self) -> list[np.ndarray]:
        """Release everything still queued, regardless of hold time (end of run)."""
        out = [b for _, b in self._queue]
        self._queue.clear()
        return out

    def pending_seconds(self) -> float:
        if not self._queue:
            return 0.0
        return sum(len(b) for _, b in self._queue) / self.sample_rate


class LiveEngine:
    """Microphone capture, worker-thread DSP and monitored output."""

    def __init__(self, cfg: Config, session: Session) -> None:
        self.cfg = cfg
        self.session = session
        self.sr = cfg.audio.sample_rate
        self.block = cfg.live.blocksize
        self.stats = LiveStats()
        self.timings = TimingRegistry()
        self.pipeline = Pipeline(cfg)
        self.name = f"live ({cfg.pipeline.order})"

        capacity = max(int(cfg.live.ring_capacity_s * self.sr), 8 * self.block)
        self.in_ring = RingBuffer(capacity, channels=1)
        self.out_ring = RingBuffer(capacity, channels=1)
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._monitor_gain = 10.0 ** (cfg.live.monitor_gain_db / 20.0)
        self._demo_delay = _DemoDelayQueue(cfg.live.demo_delay_s, self.sr)

        self.recorded_output: list[np.ndarray] = []
        self.recorded_mic: list[np.ndarray] = []
        self.recorded_input: list[np.ndarray] = []

        # Optional digital noise injection, purely a demo aid.
        self.noise: Optional[np.ndarray] = None
        self._noise_pos = 0
        self._noise_gain = 0.0
        self._gain_ready = False
        self._mic_level_estimate = 0.0
        self._calibration_blocks = max(1, int(cfg.live.calibration_s * self.sr / self.block))
        self._blocks_seen = 0
        if cfg.live.noise_file is not None:
            self.noise = load_audio(cfg.live.noise_file, self.sr)
            log.info(
                "injecting noise from %s (%.2f s, %.1f dBFS) at %.1f dB SNR",
                Path(cfg.live.noise_file).name,
                len(self.noise) / self.sr,
                dbfs(self.noise),
                cfg.live.noise_snr_db,
            )

    # ---------------------------------------------------------------- callbacks
    def _input_callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            self.stats.callback_status_flags.append(str(status))
        self.in_ring.write(indata[:, 0:1])
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

    # ------------------------------------------------------------------ noise
    def _next_noise(self, n: int) -> np.ndarray:
        if self.noise is None or len(self.noise) == 0:
            return np.zeros(n, dtype=np.float32)
        out = np.empty(n, dtype=np.float32)
        filled = 0
        while filled < n:
            take = min(n - filled, len(self.noise) - self._noise_pos)
            out[filled : filled + take] = self.noise[self._noise_pos : self._noise_pos + take]
            filled += take
            self._noise_pos = (self._noise_pos + take) % len(self.noise)
        return out

    def _inject(self, mic: np.ndarray) -> np.ndarray:
        if self.noise is None:
            return mic
        self._blocks_seen += 1
        noise = self._next_noise(len(mic))
        if not self._gain_ready:
            rms = float(np.sqrt(np.mean(np.square(mic.astype(np.float64)))))
            self._mic_level_estimate = max(self._mic_level_estimate, rms)
            if self._blocks_seen >= self._calibration_blocks:
                n_rms = float(np.sqrt(np.mean(np.square(self.noise.astype(np.float64))))) + 1e-20
                target = self._mic_level_estimate / (10.0 ** (self.cfg.live.noise_snr_db / 20.0))
                self._noise_gain = float(target / n_rms)
                self._gain_ready = True
                log.info(
                    "calibrated: mic %.1f dBFS, noise gain %.4f for %.1f dB SNR",
                    20.0 * np.log10(self._mic_level_estimate + 1e-20),
                    self._noise_gain,
                    self.cfg.live.noise_snr_db,
                )
            return mic
        return (mic + noise * self._noise_gain).astype(np.float32)

    # ------------------------------------------------------------------ worker
    def _worker_loop(self) -> None:
        scratch = np.empty((self.block, 1), dtype=np.float32)
        while not self._stop.is_set():
            got = self.in_ring.read(self.block, out=scratch, timeout=0.2)
            if got is None:
                continue
            mic = np.ascontiguousarray(got[:, 0])
            self.recorded_mic.append(mic.copy())
            signal = self._inject(mic)
            self.recorded_input.append(signal.copy())
            self.stats.input_dbfs.append(dbfs(signal))

            timer = self.timings.timer("pipeline")
            timer.start()
            out = self.pipeline.process_block(signal)
            timer.stop(audio_s=self.block / self.sr)

            if out.size:
                self._demo_delay.push(out)
            for ready in self._demo_delay.pop_ready():
                self.stats.output_dbfs.append(dbfs(ready))
                self.recorded_output.append(ready.copy())
                if self.cfg.live.monitor:
                    self.out_ring.write(ready.reshape(-1, 1))

    # -------------------------------------------------------------------- run
    def run(self, duration_s: float, dashboard: Optional[Callable[["LiveEngine", float], None]] = None) -> None:
        """Start capture, run ``dashboard`` (or the default terminal one) for
        ``duration_s`` seconds, then stop and drain everything.

        ``dashboard`` lets a caller (e.g. the Streamlit app) drive its own polling
        loop against this same engine - reading ``self.stats``, ``self.recorded_input``
        etc. - instead of the terminal rich/plain dashboard. It must block for
        approximately ``duration_s`` seconds (or return early on user interrupt); the
        stream teardown and buffered-output flush happen after it returns either way.
        """
        import sounddevice as sd

        in_dev = resolve_device(self.cfg.live.input_device, "input")
        out_dev = resolve_device(self.cfg.live.output_device, "output") if self.cfg.live.monitor else None
        check_settings(in_dev, self.sr, 1, "input")
        if self.cfg.live.monitor:
            check_settings(out_dev, self.sr, 1, "output")

        log.info("input device:  %s", device_name(in_dev, "input"))
        if self.cfg.live.monitor:
            log.info("output device: %s", device_name(out_dev, "output"))
            log.warning(
                "USE HEADPHONES. Monitoring the enhanced signal through speakers feeds it back into "
                "the microphone and will howl."
            )

        self._worker = threading.Thread(target=self._worker_loop, name="dsp-worker", daemon=True)
        self._worker.start()

        streams: list[Any] = []
        try:
            streams.append(
                sd.InputStream(
                    samplerate=self.sr, blocksize=self.block, device=in_dev, channels=1,
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
            if dashboard is not None:
                dashboard(self, duration_s)
            else:
                from ..ui.dashboard import LiveDashboard

                LiveDashboard(self, duration_s).run()
        except KeyboardInterrupt:
            log.info("interrupted by user; finalising the session")
        finally:
            for s in streams:
                try:
                    s.stop()
                    s.close()
                except Exception:  # pragma: no cover
                    pass
            self._stop.set()
            self.in_ring.close()
            self.out_ring.close()
            if self._worker is not None:
                self._worker.join(timeout=3.0)
            tail = self.pipeline.flush()
            if tail.size:
                self._demo_delay.push(tail)
            # End of run: release everything still held rather than losing it, even if
            # its hold time has not fully elapsed yet.
            for ready in self._demo_delay.flush():
                self.stats.output_dbfs.append(dbfs(ready))
                self.recorded_output.append(ready.copy())

        log.info(
            "captured %d blocks, emitted %d, xruns %d, underruns %d",
            self.stats.blocks_in, self.stats.blocks_out,
            self.in_ring.overruns, self.stats.output_underruns,
        )

    # ------------------------------------------------------------------ output
    def signal(self, parts: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, dtype=np.float32)

    def ring_stats(self) -> dict[str, int]:
        s = self.in_ring.stats()
        s["underruns"] = self.stats.output_underruns
        return s

    def measure_pipeline_latency(self) -> dict[str, float]:
        """Measured streaming latency: how long after a sample is fed does it emerge.

        This is the delay a talker actually experiences, and it is not the same as the
        model's algorithmic latency. DeepFilterNet's ``enhance`` internally
        delay-compensates (``pad=True``), so a whole-file call is time-aligned. In the
        live path, however, samples are held until a full neural chunk has accumulated
        before the model runs, so the output for input sample ``k`` is not emitted until
        roughly one chunk later. This probe drives the real ``process_block`` stream one
        block at a time and records, for a mark placed at a known input sample, how many
        output samples had already been emitted when that input was fed - which is
        exactly the buffering delay the listener hears.

        It excludes the audio device's own input/output buffering, which is hardware
        dependent and reported separately from live.blocksize.
        """
        from ..pipeline import Pipeline

        probe = Pipeline(self.cfg, model=self.pipeline.model)
        probe.reset()
        sr = self.sr

        # Feed silence one block at a time and count how many input samples go in before
        # the pipeline emits its first output sample. In a streaming pipeline the output
        # for input sample k cannot appear before its neural chunk has filled, so this
        # "fed in before first out" count is the buffering delay the talker experiences.
        # It is measured on the real process_block path, so it reflects the actual chunk
        # size and the limiter look-ahead, not a theoretical sum.
        fed = 0
        fed_at_first_output = None
        for _ in range(int(6.0 * sr) // self.block):
            y = probe.process_block(np.zeros(self.block, dtype=np.float32))
            fed += self.block
            if y.size:
                fed_at_first_output = fed
                break
        if fed_at_first_output is None:
            return {"measured_latency_ms": float("nan")}

        # Sample 0 entered at t=0 and its processed counterpart is not emitted until the
        # buffer has filled enough for the first neural chunk to run. The number of input
        # samples fed before the first output block is exactly that fill time, i.e. the
        # buffering delay every input sample experiences.
        buffering_ms = 1000.0 * fed_at_first_output / sr
        limiter_ms = self.pipeline.normaliser.latency_ms if self.pipeline.normaliser else 0.0
        device_ms = 1000.0 * self.block / sr  # one block each direction, hardware dependent
        total_ms = buffering_ms + limiter_ms + device_ms
        return {
            "measured_latency_ms": round(total_ms, 1),
            "buffering_ms": round(buffering_ms, 1),
            "limiter_lookahead_ms": round(limiter_ms, 1),
            "device_io_ms": round(device_ms, 1),
        }

    @property
    def processor(self) -> "LiveEngine":
        """The dashboard reads ``engine.processor.name`` and ``.nlms``."""
        return self

    @property
    def nlms(self) -> Any:
        return self.pipeline.nlms

    @property
    def normaliser(self) -> Any:
        return self.pipeline.normaliser


# (chunk_s, context_s, crossfade_ms, post_filter). Measured on 12 corpus examples at
# <= 0 dB SNR, with no attenuation cap. Noise reduction is the level drop in
# talker-silent regions; speech attenuation is how much quieter the talker became.
#
#   profile           chunk   latency   noise red.  sp.atten
#   low_latency       250 ms   265 ms    +37.5 dB    10.6 dB
#   balanced          500 ms   515 ms    +37.5 dB     9.6 dB
#   quality             1 s   1015 ms    +37.5 dB     8.1 dB
#   max_suppression     1 s   1015 ms    +44.7 dB     8.6 dB   (post-filter on)
#
# 60 ms chunks were tried as the low_latency default and rejected: they removed less
# noise (+33.9 dB) and damaged the speech far more (12.4 dB) than 250 ms, in exchange
# for 190 ms less delay. Not a good trade.
LATENCY_PROFILES = {
    "low_latency": (0.25, 0.25, 10.0, False),
    "balanced": (0.5, 0.25, 20.0, False),
    "quality": (1.0, 0.25, 30.0, False),
    "max_suppression": (1.0, 0.25, 30.0, True),
}


def apply_latency_profile(cfg: Config) -> Config:
    """Resolve live.latency_profile into concrete streaming parameters.

    Returns a copy: the profile is a convenience over neural.streaming, and 'custom'
    leaves whatever is already configured untouched.
    """
    profile = cfg.live.latency_profile
    if profile == "custom":
        return cfg
    chunk_s, context_s, crossfade_ms, post_filter = LATENCY_PROFILES[profile]
    out = cfg.model_copy(deep=True)
    out.neural.streaming.chunk_s = chunk_s
    out.neural.streaming.context_s = context_s
    out.neural.streaming.crossfade_ms = crossfade_ms
    out.neural.streaming.overlap = 0.0  # small chunks crossfade, they do not overlap-hop
    out.neural.post_filter = post_filter
    return out


def run_live(
    cfg: Config,
    session: Optional[Session] = None,
    dashboard: Optional[Callable[["LiveEngine", float], None]] = None,
) -> dict[str, Path]:
    """``anc run --mode live_mic``.

    ``dashboard``, if given, replaces the terminal rich/plain dashboard - see
    :meth:`LiveEngine.run`. Used by the Streamlit app to drive its own live-updating
    widgets against the same engine instance.
    """
    cfg = apply_latency_profile(cfg)
    session = session or create_session(cfg, "live_mic")
    log.info(
        "pipeline: %s  |  latency profile: %s (chunk %.0f ms, context %.0f ms)  |  "
        "demo output delay: %.1f s",
        Pipeline(cfg, load_model=False).describe(),
        cfg.live.latency_profile,
        cfg.neural.streaming.chunk_s * 1000,
        cfg.neural.streaming.context_s * 1000,
        cfg.live.demo_delay_s,
    )
    sampler = ResourceSampler(interval_s=0.5).start()
    engine = LiveEngine(cfg, session)
    t0 = time.perf_counter()
    engine.run(cfg.live.duration_s, dashboard=dashboard)
    wall = time.perf_counter() - t0
    sampler.stop()

    sr = cfg.audio.sample_rate
    output = engine.signal(engine.recorded_output)
    mic = engine.signal(engine.recorded_mic)
    pipeline_input = engine.signal(engine.recorded_input)

    taps: dict[str, np.ndarray] = {}
    if engine.noise is not None and mic.size:
        taps["mic_premix_pseudo_clean"] = mic
    taps["input"] = pipeline_input if pipeline_input.size else mic
    taps["output"] = output

    data = ReportData(
        session=session,
        title="Live single-microphone run",
        mode=session.mode,
        subtitle=(
            f"{cfg.live.duration_s:.0f} s requested &middot; {len(output) / sr:.1f} s of output "
            f"&middot; chunked neural framing ({cfg.neural.streaming.chunk_s} s chunks)"
        ),
        scope_notes=list(SCOPE_NOTES)
        + [
            "Live output uses <b>chunked</b> neural framing: it lags the input by about one chunk "
            f"({cfg.neural.streaming.chunk_s} s) plus the model's 40 ms algorithmic latency plus "
            f"{cfg.normalise.limiter_lookahead_ms} ms of limiter look-ahead plus device buffering. "
            "This is a demo path, not low-latency streaming.",
        ]
        + (
            [
                f"An additional <b>{cfg.live.demo_delay_s:.1f} s artificial delay</b> was applied to "
                "the output for this run (live.demo_delay_s). It is a pure post-processing hold: the "
                "audio samples are not altered in any way, only the moment they are released is "
                "pushed back. This exists only to make input and output separable in a single-take "
                "recording; it is <b>not</b> part of the system's real latency, which is reported in "
                "the Latency budget table below with this delay excluded.",
            ]
            if cfg.live.demo_delay_s > 0.0
            else []
        ),
    )
    data.metadata_rows = session_metadata_rows(
        session, cfg, engine.pipeline.model.info.as_dict() if engine.pipeline.model else None,
        extra=[
            ("Input device", device_name(resolve_device(cfg.live.input_device, "input"), "input")),
            (
                "Output device",
                device_name(resolve_device(cfg.live.output_device, "output"), "output")
                if cfg.live.monitor else "monitoring disabled",
            ),
            ("Requested duration", f"{cfg.live.duration_s:.1f} s"),
            (
                "Demo output delay",
                f"{cfg.live.demo_delay_s:.1f} s (artificial, demo aid only - see note below)"
                if cfg.live.demo_delay_s > 0.0 else "none (0 s, real minimum latency shown below)",
            ),
            (
                "Injected noise",
                f"{Path(cfg.live.noise_file).name} at {cfg.live.noise_snr_db:g} dB SNR"
                if cfg.live.noise_file else "none (room noise only)",
            ),
        ],
    )

    # ---- metrics -------------------------------------------------------------
    if engine.noise is not None and mic.size and output.size:
        n = min(len(mic), len(output), len(pipeline_input))
        base = compute_intrusive(mic[:n], pipeline_input[:n], sr, cfg.metrics)
        enhanced = compute_intrusive(mic[:n], output[:n], sr, cfg.metrics)
        rows = [["Metric", "With injected noise", "Enhanced output", "Change"]]
        for label, key in (
            ("PESQ (wb)", "pesq"), ("STOI", "stoi"), ("ESTOI", "estoi"),
            ("SI-SDR dB", "si_sdr"), ("segSNR dB", "segmental_snr"), ("LSD dB", "lsd"),
        ):
            b, e = getattr(base, key), getattr(enhanced, key)
            rows.append([label, f"{b:.3f}", f"{e:.3f}", f"{e - b:+.3f}"])
        data.add_table(
            "Intrusive metrics against the pre-mix microphone recording", rows,
            "The pre-mix microphone signal is the reference. It contains the real room's own noise, "
            "so it is a pseudo-clean reference rather than a true clean signal: absolute values are "
            "pessimistic, but the change from input to output is meaningful.",
        )
        values = enhanced.as_dict()
        values["snr_improvement_db"] = enhanced.si_sdr - base.si_sdr
        data.target_scope = (
            "Live run, measured against the pre-mix microphone recording. That reference is a "
            "quiet-room capture, not studio-clean, so the absolute figures read pessimistically; "
            "the change from input to output is the meaningful part."
        )
        data.payload["intrusive"] = {"input": base.as_dict(), "output": enhanced.as_dict()}
    else:
        data.scope_notes.append(
            "No clean reference exists for this run, so only non-intrusive metrics, levels and "
            "timing are reported."
        )

    dnsmos = DnsmosEstimator() if cfg.metrics.dnsmos else None
    rows = [["Signal", "Estimated SNR (dB)", "Level (dBFS)", "DNSMOS OVRL"]]
    nonintrusive: dict[str, float] = {}
    for name in ("input", "output"):
        sig = taps.get(name)
        if sig is None or not sig.size:
            continue
        ovrl = "not enabled"
        if dnsmos is not None:
            score = dnsmos.score(sig, sr)
            ovrl = f"{score.dnsmos_ovrl:.2f}"
            nonintrusive[f"{name}_dnsmos_ovrl"] = float(score.dnsmos_ovrl)
        est, lvl = estimate_snr_db(sig, sr), dbfs(sig)
        nonintrusive[f"{name}_estimated_snr_db"] = float(est)
        nonintrusive[f"{name}_level_dbfs"] = float(lvl)
        rows.append([name, f"{est:.1f}", f"{lvl:.1f}", ovrl])
    if "input_estimated_snr_db" in nonintrusive and "output_estimated_snr_db" in nonintrusive:
        nonintrusive["estimated_snr_change_db"] = (
            nonintrusive["output_estimated_snr_db"] - nonintrusive["input_estimated_snr_db"]
        )
    data.add_table("Non-intrusive metrics", rows,
                   "Available without a reference. The output level should sit close to the "
                   f"normaliser target of {cfg.normalise.target_dbfs:g} dBFS.")

    # Reference-free noise reduction: the level drop between input and output measured only
    # where the talker is silent. This needs no clean reference, so unlike PESQ or STOI it
    # is available on any live run, and it is the figure a listener would call "noise
    # cancellation". With the pipeline bypassed it is 0 dB by construction, which is what
    # makes an ANC-off run a usable baseline.
    pin, pout = taps.get("input"), taps.get("output")
    if pin is not None and pout is not None and pin.size and pout.size:
        n = min(len(pin), len(pout))
        silent = ~speech_mask(pin[:n], cfg.vad, sr)
        live_nr = noise_reduction_db(pin[:n], pout[:n], silent)
        speech_kept = float(np.mean(silent)) if silent.size else float("nan")
        if np.isfinite(live_nr):
            nonintrusive["noise_reduction_db"] = float(live_nr)
            nonintrusive["silent_fraction"] = speech_kept
            data.add_table(
                "Noise removed (reference-free)",
                [
                    ["Quantity", "Value"],
                    ["Level drop in talker-silent regions", f"{live_nr:+.2f} dB"],
                    ["Fraction of the run the talker was silent", f"{100.0 * speech_kept:.0f} %"],
                ],
                "Measured between the pipeline input and output in the regions the voice-activity "
                "detector marked as speech-free, so attenuating the talker cannot inflate it. This "
                "is the headline live figure: it needs no clean reference, and it is 0 dB when the "
                "pipeline is bypassed.",
            )
    data.payload["non_intrusive"] = nonintrusive

    if engine.normaliser is not None and engine.normaliser.enabled:
        summary = engine.normaliser.summary()
        nrows = [["Quantity", "Value"]]
        for key in ("mode", "gain_mean_db", "gain_range_db", "held_fraction", "limited_frames",
                    "clipped_samples", "latency_ms"):
            if key in summary:
                v = summary[key]
                nrows.append([key.replace("_", " "), f"{v:.3f}" if isinstance(v, float) else str(v)])
        data.add_table("Volume normalisation", nrows,
                       "Gain held during pauses so residual noise is not pumped up.")
        diag = engine.normaliser.diagnostics
        if diag is not None and diag.gain_db:
            data.add_figure(
                plots.agc_trace(diag, session.figure_path("live_agc"), cfg.report.dpi),
                "Applied gain, tracked speech level and limiter activity over the run.",
            )
        data.payload["normalise"] = summary

    data.add_table("Signal levels", level_rows(taps))

    if cfg.report.write_wav and cfg.live.save_streams:
        data.payload["audio_files"] = save_stage_wavs(session, taps, sr)
    add_signal_figures(data, session, taps, sr, None, cfg.report.dpi)
    if taps["input"].size and output.size:
        data.add_figure(
            plots.stage_difference(
                taps["input"][: 12 * sr], output[: 12 * sr], sr,
                session.figure_path("live_diff"), cfg.report.dpi,
                label_before="input", label_after="output",
            ),
            "What the pipeline changed: removed content and the spectral delta.",
        )
    data.add_figure(
        plots.live_levels(
            engine.stats.input_dbfs, engine.stats.output_dbfs,
            session.figure_path("live_levels"), cfg.report.dpi,
            interval_s=cfg.audio.hop_size / sr,
        ),
        "Input and output level over the run.",
    )
    add_system_figures(data, session, None, sampler.as_records(), cfg.report.dpi)

    audio_seconds = max(len(taps["input"]), 1) / sr
    system = collect_system_metrics(
        engine.timings, audio_seconds=audio_seconds, wall_seconds=wall,
        latency_ms=engine.pipeline.latency_breakdown(), ring_stats=engine.ring_stats(),
        resource_records=sampler.as_records(), peak_rss_mb=sampler.peak_rss_mb(),
        single_thread=(cfg.neural.num_threads == 1),
    )
    data.add_table("Processing time and real-time factor", system_table(system))
    latency = engine.pipeline.latency_breakdown()
    measured = engine.measure_pipeline_latency()
    data.add_table(
        "Latency budget",
        latency_table(latency, measured=measured.get("measured_latency_ms")),
        f"Latency profile: <b>{cfg.live.latency_profile}</b>, {cfg.neural.streaming.chunk_s * 1000:.0f} ms "
        f"neural chunk. The measured figure is an impulse pushed through the real block path, so it "
        f"includes chunk buffering and the limiter look-ahead but not the audio device's own "
        f"input/output buffering (that is set by live.blocksize = {cfg.live.blocksize} samples = "
        f"{1000.0 * cfg.live.blocksize / sr:.0f} ms per direction).",
    )
    data.payload["latency"] = {**latency, **measured, "profile": cfg.live.latency_profile}
    log.info(
        "measured DSP latency: %.0f ms (profile: %s)",
        measured.get("measured_latency_ms", float("nan")),
        cfg.live.latency_profile,
    )

    ring = engine.ring_stats()
    stats = engine.stats.as_dict()
    health = [["Quantity", "Value"]]
    for label, value in (
        ("Blocks captured", stats["blocks_in"]),
        ("Blocks emitted", stats["blocks_out"]),
        ("Input ring overruns (xruns)", ring.get("overruns", 0)),
        ("Samples dropped", ring.get("dropped_samples", 0)),
        ("Output underruns", stats["output_underruns"]),
        ("Ring high-water mark", f"{ring.get('high_water', 0)} / {ring.get('capacity', 0)} samples"),
        ("PortAudio status flags", ", ".join(stats["callback_status_flags"]) or "none"),
        ("Peak process RSS", f"{sampler.peak_rss_mb()} MB"),
    ):
        health.append([label, str(value)])
    data.add_table("Real-time health", health,
                   "Zero xruns and a flat memory trace are the acceptance conditions for a live run.")

    data.warnings.extend(system.warnings)
    data.payload.update({
        "live_stats": stats, "ring": ring, "system": system.as_dict(),
        "resources": sampler.as_records(),
        "model": engine.pipeline.model.info.as_dict() if engine.pipeline.model else {},
        "latency_ms": engine.pipeline.latency_breakdown(),
    })
    return finalise(data)


# Backwards-compatible aliases used by the CLI.
run_live_dfn_only = run_live
run_live_injected = run_live
