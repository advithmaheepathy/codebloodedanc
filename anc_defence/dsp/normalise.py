"""Volume normalisation: speech-aware automatic gain control with a soft limiter.

This is the second and final stage of the pipeline. DeepFilterNet leaves the speech
at whatever level it arrived at, and after heavy suppression that is usually quiet
and inconsistent between talkers and distances. A comms chain needs the operator's
voice to arrive at a predictable level.

How it works, per 10 ms frame:

1. A VAD decides whether the frame contains speech.
2. On speech frames the active speech level is tracked (fast attack, slow release).
3. The wanted gain is ``target - measured``, clamped to a configurable range so the
   AGC can never turn a near-silent frame into full-scale hiss.
4. The applied gain moves toward the wanted gain with separate attack and release
   time constants, and is **held frozen during pauses**. This is the important part:
   a naive AGC raises its gain when the talker stops and pumps the residual noise up
   with it, which is exactly the artefact that makes suppressed audio sound worse
   than it measures.
5. Gain is interpolated per sample across the frame, so there is no zipper noise at
   frame boundaries.
6. A look-ahead soft limiter catches transients before they clip. It introduces a
   small, reported latency (5 ms by default) and is the only part of the pipeline
   that adds delay after the model.

Modes
-----
``agc``   the above. The default.
``peak``  one static gain per signal so the peak lands on the ceiling.
``rms``   one static gain per signal so the active speech level hits the target.
``off``   pass through unchanged.

Level-sensitive metrics must be measured *before* this stage, or with level
alignment, otherwise the gain change alone will move them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config import NormaliseCfg, VadCfg
from ..utils.logging import get_logger
from .vad import EnergyFlatnessVad

log = get_logger(__name__)

_EPS = 1e-12


def dbfs_of(x: np.ndarray) -> float:
    if x.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return 20.0 * np.log10(rms + _EPS)


def active_speech_dbfs(
    x: np.ndarray, sample_rate: int = 48000, frame_ms: float = 20.0, range_db: float = 40.0
) -> float:
    """Active speech level: RMS over frames within ``range_db`` of the loudest frame.

    An approximation of ITU-T P.56 active speech level. Leading and trailing silence
    cannot drag it down, which is what makes it usable as an AGC target.
    """
    x = np.asarray(x, dtype=np.float64)
    frame = max(16, int(round(sample_rate * frame_ms / 1000.0)))
    n = len(x) // frame
    if n == 0:
        return dbfs_of(x)
    power = np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1)
    peak = float(power.max())
    if peak <= 0:
        return -120.0
    keep = power > peak * 10.0 ** (-range_db / 10.0)
    return float(10.0 * np.log10(float(np.mean(power[keep])) + _EPS))


@dataclass
class NormaliseDiagnostics:
    """Per-frame record of what the normaliser did, for the report and dashboard."""

    frame_size: int = 480
    sample_rate: int = 48000
    gain_db: list[float] = field(default_factory=list)
    speech_level_db: list[float] = field(default_factory=list)
    limiter_reduction_db: list[float] = field(default_factory=list)
    speech_flags: list[bool] = field(default_factory=list)
    held_frames: int = 0
    limited_frames: int = 0
    clipped_samples: int = 0

    def time_axis(self) -> np.ndarray:
        return np.arange(len(self.gain_db), dtype=np.float64) * self.frame_size / self.sample_rate

    def summary(self) -> dict[str, float | int]:
        gains = np.asarray(self.gain_db, dtype=np.float64)
        lim = np.asarray(self.limiter_reduction_db, dtype=np.float64)
        return {
            "frames": len(self.gain_db),
            "gain_mean_db": float(np.mean(gains)) if gains.size else float("nan"),
            "gain_min_db": float(np.min(gains)) if gains.size else float("nan"),
            "gain_max_db": float(np.max(gains)) if gains.size else float("nan"),
            "gain_range_db": float(np.max(gains) - np.min(gains)) if gains.size else float("nan"),
            "speech_fraction": float(np.mean(self.speech_flags)) if self.speech_flags else float("nan"),
            "held_frames": self.held_frames,
            "held_fraction": (
                self.held_frames / len(self.gain_db) if self.gain_db else float("nan")
            ),
            "limited_frames": self.limited_frames,
            "limiter_max_reduction_db": float(np.max(lim)) if lim.size else 0.0,
            "clipped_samples": self.clipped_samples,
        }


class SpeechAgc:
    """Streaming speech-aware AGC with a look-ahead soft limiter.

    Block-by-block and whole-signal processing use the same code, so the offline
    result is exactly what the live path would produce.
    """

    def __init__(
        self,
        cfg: NormaliseCfg,
        vad_cfg: Optional[VadCfg] = None,
        sample_rate: int = 48000,
        frame_size: int = 480,
    ) -> None:
        self.cfg = cfg
        self.sample_rate = sample_rate
        self.frame_size = int(frame_size)
        self.vad = EnergyFlatnessVad(vad_cfg or VadCfg(), sample_rate, frame_size=self.frame_size)

        frame_s = self.frame_size / sample_rate
        self._attack = float(np.exp(-frame_s / max(1e-4, cfg.attack_ms / 1000.0)))
        self._release = float(np.exp(-frame_s / max(1e-4, cfg.release_ms / 1000.0)))
        self._level_attack = float(np.exp(-frame_s / max(1e-4, cfg.level_attack_ms / 1000.0)))
        self._level_release = float(np.exp(-frame_s / max(1e-4, cfg.level_release_ms / 1000.0)))

        self.lookahead = max(0, int(round(cfg.limiter_lookahead_ms * sample_rate / 1000.0)))
        self._delay = np.zeros(self.lookahead, dtype=np.float32)
        self._ceiling = 10.0 ** (cfg.limiter_ceiling_dbfs / 20.0)
        self._lim_release = float(np.exp(-frame_s / max(1e-4, cfg.limiter_release_ms / 1000.0)))

        self._gain_db = 0.0
        self._speech_level_db: Optional[float] = None
        self._limiter_gain = 1.0
        self._prev_total_gain: Optional[float] = None
        self._peak_db: Optional[float] = None
        self._peak_decay_db = cfg.peak_decay_db_per_s * frame_s
        self.diagnostics = NormaliseDiagnostics(self.frame_size, sample_rate)

    # ------------------------------------------------------------------- props
    @property
    def latency_ms(self) -> float:
        """Extra latency this stage adds: the limiter's look-ahead."""
        return 1000.0 * self.lookahead / self.sample_rate

    def reset(self) -> None:
        self._delay = np.zeros(self.lookahead, dtype=np.float32)
        self._gain_db = 0.0
        self._speech_level_db = None
        self._limiter_gain = 1.0
        self._prev_total_gain = None
        self._peak_db = None
        self.vad.reset()
        self.diagnostics = NormaliseDiagnostics(self.frame_size, self.sample_rate)

    # ------------------------------------------------------------------ inverse
    def gain_envelope(self, n_samples: int, aligned: bool = True) -> np.ndarray:
        """The exact per-sample gain that was applied, reconstructed from the trace.

        The AGC ramps linearly between frame gains, so the envelope can be rebuilt
        exactly. Dividing the output by it recovers the signal as it was before
        normalisation, which is how quality metrics are separated from level: a
        time-varying gain is not scale invariant, so SI-SDR would otherwise read the
        gain ride as distortion.

        ``aligned`` shifts the envelope by the limiter look-ahead to match the
        delay-compensated output that :meth:`process` returns.
        """
        gains_db = np.asarray(self.diagnostics.gain_db, dtype=np.float64)
        if gains_db.size == 0:
            return np.ones(n_samples, dtype=np.float32)
        gains = 10.0 ** (gains_db / 20.0)
        env = np.empty(gains.size * self.frame_size, dtype=np.float64)
        start = gains[0]
        for i, end in enumerate(gains):
            env[i * self.frame_size : (i + 1) * self.frame_size] = np.linspace(
                start, end, self.frame_size, endpoint=True
            )
            start = end
        if aligned and self.lookahead > 0:
            env = env[self.lookahead :]
        if env.size < n_samples:
            env = np.concatenate([env, np.full(n_samples - env.size, gains[-1])])
        return env[:n_samples].astype(np.float32)

    # ------------------------------------------------------------------- block
    def process_block(self, x: np.ndarray) -> np.ndarray:
        """Normalise one frame. Output is delayed by the limiter look-ahead."""
        cfg = self.cfg
        frame = np.asarray(x, dtype=np.float32)
        n = frame.size
        if n == 0:
            return frame

        speech = self.vad.process_frame(frame)
        frame_db = dbfs_of(frame)

        # Running peak with a slow decay. A frame close to the peak counts as active
        # even when the VAD stays quiet, which is the fallback for continuously active
        # audio: the VAD's noise floor adapts up to a signal that never pauses, after
        # which it reports no speech and, without this, the AGC would do nothing at all.
        if self._peak_db is None:
            self._peak_db = frame_db
        else:
            self._peak_db = max(frame_db, self._peak_db - self._peak_decay_db)
        loud = frame_db > self._peak_db - cfg.active_range_db
        active = speech or loud

        # 1. Track the level on active frames. Without ``hold_during_pause`` the
        # estimate follows every frame above the gate, which is what a naive AGC does
        # and exactly why a naive AGC pumps the noise up in pauses.
        track = (active or not cfg.hold_during_pause) and frame_db > cfg.gate_dbfs
        if track:
            if self._speech_level_db is None:
                self._speech_level_db = frame_db
            else:
                a = self._level_attack if frame_db > self._speech_level_db else self._level_release
                self._speech_level_db = a * self._speech_level_db + (1.0 - a) * frame_db

        # 2. Wanted gain from the tracked level.
        if self._speech_level_db is None:
            wanted_db = self._gain_db
        else:
            wanted_db = float(
                np.clip(cfg.target_dbfs - self._speech_level_db, cfg.min_gain_db, cfg.max_gain_db)
            )

        # 3. Move toward it, but hold during pauses so residual noise is not pumped up.
        held = False
        if active or not cfg.hold_during_pause:
            a = self._attack if wanted_db > self._gain_db else self._release
            self._gain_db = a * self._gain_db + (1.0 - a) * wanted_db
        else:
            held = True
            self.diagnostics.held_frames += 1

        agc_gain = 10.0 ** (self._gain_db / 20.0)

        # 4. Look-ahead limiter: decide on the *upcoming* frame, apply to the delayed one.
        peak_ahead = float(np.max(np.abs(frame))) * agc_gain
        wanted_lim = 1.0 if peak_ahead <= self._ceiling else self._ceiling / (peak_ahead + _EPS)
        if wanted_lim < self._limiter_gain:
            self._limiter_gain = wanted_lim  # instant attack: never let a peak through
            self.diagnostics.limited_frames += 1
        else:
            self._limiter_gain = (
                self._lim_release * self._limiter_gain + (1.0 - self._lim_release) * wanted_lim
            )

        total_gain = agc_gain * self._limiter_gain

        # 5. Emit the delayed samples with a per-sample gain ramp (no zipper noise).
        if self.lookahead > 0:
            buffer = np.concatenate([self._delay, frame])
            out = buffer[:n].copy()
            self._delay = buffer[n:][-self.lookahead :].copy()
        else:
            out = frame.copy()

        start = self._prev_total_gain if self._prev_total_gain is not None else total_gain
        ramp = np.linspace(start, total_gain, n, endpoint=True, dtype=np.float32)
        self._prev_total_gain = total_gain
        out = (out * ramp).astype(np.float32)

        # 6. Final safety clamp. Should be a no-op if the limiter is doing its job.
        over = np.abs(out) > 1.0
        if over.any():
            self.diagnostics.clipped_samples += int(over.sum())
            np.clip(out, -1.0, 1.0, out=out)

        d = self.diagnostics
        d.gain_db.append(20.0 * float(np.log10(total_gain + _EPS)))
        d.speech_level_db.append(
            self._speech_level_db if self._speech_level_db is not None else float("nan")
        )
        d.limiter_reduction_db.append(-20.0 * float(np.log10(self._limiter_gain + _EPS)))
        d.speech_flags.append(bool(speech))
        return out

    # ----------------------------------------------------------------- offline
    def process(self, x: np.ndarray, align: bool = True) -> np.ndarray:
        """Whole-signal normalisation, block by block, exactly as the live path does.

        With ``align=True`` the limiter's look-ahead delay is removed from the result
        so the output stays sample-aligned with the input. That matters: an
        uncompensated 5 ms shift leaves PESQ untouched (it realigns internally) while
        destroying SI-SDR, which would look like a quality collapse when nothing but
        the timing had changed. The delay is real in the live path and is reported in
        the latency budget either way.
        """
        x = np.asarray(x, dtype=np.float32)
        n_blocks = len(x) // self.frame_size
        parts: list[np.ndarray] = []
        for b in range(n_blocks):
            s = b * self.frame_size
            parts.append(self.process_block(x[s : s + self.frame_size]))
        remainder = len(x) - n_blocks * self.frame_size
        if remainder:
            parts.append(self.process_block(x[n_blocks * self.frame_size :]))
        if not parts:
            return x
        out = np.concatenate(parts).astype(np.float32)
        # Flush the look-ahead delay line so no audio is lost off the end.
        if self.lookahead > 0:
            tail = self._delay * (self._prev_total_gain or 1.0)
            out = np.concatenate([out, tail.astype(np.float32)])
            if align:
                out = out[self.lookahead :]
        return out[: len(x)] if len(out) >= len(x) else np.pad(out, (0, len(x) - len(out)))


# ----------------------------------------------------------------- static modes


def peak_normalise(x: np.ndarray, ceiling_dbfs: float = -1.0) -> tuple[np.ndarray, float]:
    """One static gain so the peak lands on the ceiling."""
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= 0:
        return np.asarray(x, dtype=np.float32), 0.0
    gain = (10.0 ** (ceiling_dbfs / 20.0)) / peak
    return (np.asarray(x, dtype=np.float32) * gain).astype(np.float32), 20.0 * float(np.log10(gain))


def rms_normalise(
    x: np.ndarray,
    target_dbfs: float = -26.0,
    sample_rate: int = 48000,
    ceiling_dbfs: float = -1.0,
    max_gain_db: float = 30.0,
) -> tuple[np.ndarray, float]:
    """One static gain so the active speech level hits the target, then peak-safe."""
    level = active_speech_dbfs(x, sample_rate)
    gain_db = float(np.clip(target_dbfs - level, -max_gain_db, max_gain_db))
    y = (np.asarray(x, dtype=np.float32) * 10.0 ** (gain_db / 20.0)).astype(np.float32)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    ceiling = 10.0 ** (ceiling_dbfs / 20.0)
    if peak > ceiling:
        trim = ceiling / peak
        y = (y * trim).astype(np.float32)
        gain_db += 20.0 * float(np.log10(trim))
    return y, gain_db


# --------------------------------------------------------------------- factory


class Normaliser:
    """Dispatches to the configured normalisation mode behind one interface."""

    def __init__(
        self,
        cfg: NormaliseCfg,
        vad_cfg: Optional[VadCfg] = None,
        sample_rate: int = 48000,
        frame_size: int = 480,
    ) -> None:
        self.cfg = cfg
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.agc = (
            SpeechAgc(cfg, vad_cfg, sample_rate, frame_size) if cfg.mode == "agc" else None
        )
        self.static_gain_db: float = 0.0

    @property
    def enabled(self) -> bool:
        return self.cfg.mode != "off"

    @property
    def latency_ms(self) -> float:
        return self.agc.latency_ms if self.agc is not None else 0.0

    def reset(self) -> None:
        if self.agc is not None:
            self.agc.reset()
        self.static_gain_db = 0.0

    def process_block(self, x: np.ndarray) -> np.ndarray:
        """Streaming path. Static modes need the whole signal, so they pass through
        here and are applied in :meth:`process`; the live path always uses ``agc``."""
        if self.agc is not None:
            return self.agc.process_block(x)
        return np.asarray(x, dtype=np.float32)

    def process(self, x: np.ndarray) -> np.ndarray:
        mode = self.cfg.mode
        if mode == "off":
            return np.asarray(x, dtype=np.float32)
        if mode == "agc":
            assert self.agc is not None
            return self.agc.process(x)
        if mode == "peak":
            y, self.static_gain_db = peak_normalise(x, self.cfg.limiter_ceiling_dbfs)
            return y
        if mode == "rms":
            y, self.static_gain_db = rms_normalise(
                x, self.cfg.target_dbfs, self.sample_rate, self.cfg.limiter_ceiling_dbfs,
                self.cfg.max_gain_db,
            )
            return y
        raise ValueError(f"unknown normalisation mode: {mode}")

    def gain_envelope(self, n_samples: int) -> np.ndarray:
        """Per-sample applied gain, for separating quality from level in the metrics."""
        if self.agc is not None:
            return self.agc.gain_envelope(n_samples)
        gain = 10.0 ** (self.static_gain_db / 20.0)
        return np.full(n_samples, gain, dtype=np.float32)

    def summary(self) -> dict[str, float | int | str]:
        out: dict[str, float | int | str] = {"mode": self.cfg.mode}
        if self.agc is not None:
            out.update(self.agc.diagnostics.summary())
        else:
            out["static_gain_db"] = round(self.static_gain_db, 2)
        out["latency_ms"] = round(self.latency_ms, 2)
        return out

    @property
    def diagnostics(self) -> Optional[NormaliseDiagnostics]:
        return self.agc.diagnostics if self.agc is not None else None
