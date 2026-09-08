"""NLMS adaptive noise cancellation: shared machinery and the time-domain oracle.

Signal model
------------
``d`` (primary)   = speech + noise picked up by the talker's microphone
``x`` (reference) = the noise, correlated with the noise in ``d`` through an
                    unknown acoustic path
``e`` (output)    = d - w^T x, the residual after subtracting the estimated noise

Update (normalised least mean squares)::

    y[n] = w^T x[n]
    e[n] = d[n] - y[n]
    w   <- (1 - leak) * w + mu * e[n] * x[n] / (eps + ||x[n]||^2)

Why the safeguards matter more than the update rule
---------------------------------------------------
Plain NLMS applied to real signals fails in three specific ways, and each one is
guarded here:

* **Double talk.** If the reference contains any speech leakage, or the talker is
  active while the filter adapts, NLMS minimises total residual energy by
  cancelling *speech*. Adaptation is frozen (or heavily slowed) while the VAD
  reports speech.
* **Silent reference.** With no reference energy the normalised update divides by
  almost nothing and the weights random-walk. Adaptation is skipped below a level
  threshold.
* **Impulsive noise and divergence.** A single gunshot produces an enormous
  gradient. The update is clipped (or frozen through the transient), the weight
  norm and the output/input power ratio are monitored, and the filter is rolled
  back to the last healthy checkpoint if it misbehaves.

Two implementations share this machinery: this module's sample-wise
:class:`TimeDomainNlms` (the correctness oracle, too slow for long filters in real
time) and :mod:`anc_defence.dsp.fdaf`'s partitioned-block frequency-domain filter
(the real-time default).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np

from ..config import NlmsCfg, VadCfg
from ..utils.logging import get_logger
from .impulse import ImpulseDetector
from .vad import EnergyFlatnessVad

log = get_logger(__name__)

_EPS = 1e-20

# Adaptation state codes recorded per block for the diagnostics plot.
ADAPTING = "adapting"
FROZEN_SPEECH = "frozen_speech"
SCALED_SPEECH = "scaled_speech"
FROZEN_SILENCE = "frozen_ref_silence"
FROZEN_IMPULSE = "frozen_impulse"
CLIPPED_IMPULSE = "clipped_impulse"
ROLLED_BACK = "rolled_back"

_STATE_ORDER = (
    ADAPTING,
    SCALED_SPEECH,
    CLIPPED_IMPULSE,
    FROZEN_SPEECH,
    FROZEN_SILENCE,
    FROZEN_IMPULSE,
    ROLLED_BACK,
)


@dataclass
class NlmsDiagnostics:
    """Per-block diagnostics, sized by the number of processed blocks."""

    block_size: int = 480
    sample_rate: int = 48000
    erle_db: list[float] = field(default_factory=list)
    weight_norm: list[float] = field(default_factory=list)
    mu_effective: list[float] = field(default_factory=list)
    state: list[str] = field(default_factory=list)
    reference_dbfs: list[float] = field(default_factory=list)
    mse: list[float] = field(default_factory=list)
    speech_flags: list[bool] = field(default_factory=list)
    impulse_flags: list[bool] = field(default_factory=list)
    rollbacks: int = 0
    divergence_events: int = 0

    def time_axis(self) -> np.ndarray:
        n = len(self.erle_db)
        return np.arange(n, dtype=np.float64) * self.block_size / self.sample_rate

    def state_counts(self) -> dict[str, int]:
        counts = {k: 0 for k in _STATE_ORDER}
        for s in self.state:
            counts[s] = counts.get(s, 0) + 1
        return counts

    def summary(self) -> dict[str, float | int | dict[str, int]]:
        erle = np.asarray(self.erle_db, dtype=np.float64)
        finite = erle[np.isfinite(erle)]
        return {
            "blocks": len(self.erle_db),
            "erle_mean_db": float(np.mean(finite)) if finite.size else float("nan"),
            "erle_median_db": float(np.median(finite)) if finite.size else float("nan"),
            "erle_final_db": float(np.mean(finite[-50:])) if finite.size else float("nan"),
            "weight_norm_final": self.weight_norm[-1] if self.weight_norm else float("nan"),
            "adapting_fraction": (
                self.state.count(ADAPTING) / len(self.state) if self.state else float("nan")
            ),
            "speech_fraction": (
                float(np.mean(self.speech_flags)) if self.speech_flags else float("nan")
            ),
            "impulse_fraction": (
                float(np.mean(self.impulse_flags)) if self.impulse_flags else float("nan")
            ),
            "rollbacks": self.rollbacks,
            "divergence_events": self.divergence_events,
            "state_counts": self.state_counts(),
        }


@dataclass
class NlmsResult:
    """Output of one full pass of the adaptive stage."""

    output: np.ndarray  # residual e, the enhanced signal
    estimated_noise: np.ndarray  # y, the filter's estimate of the noise in d
    weights: np.ndarray
    diagnostics: NlmsDiagnostics

    @property
    def erle_curve(self) -> np.ndarray:
        return np.asarray(self.diagnostics.erle_db, dtype=np.float64)


class AdaptiveFilter(Protocol):
    """Interface shared by the time-domain and frequency-domain implementations."""

    filter_length: int
    block_size: int

    def reset(self) -> None: ...

    def process_block(self, d: np.ndarray, x: np.ndarray) -> np.ndarray: ...

    def process(self, d: np.ndarray, x: np.ndarray) -> NlmsResult: ...

    def impulse_response(self) -> np.ndarray: ...


# --------------------------------------------------------------------- guards


class SafeguardEngine:
    """Decides, per block, whether and how strongly the filter may adapt."""

    def __init__(
        self,
        cfg: NlmsCfg,
        vad_cfg: VadCfg,
        sample_rate: int,
        block_size: int,
    ) -> None:
        self.cfg = cfg
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.vad = EnergyFlatnessVad(vad_cfg, sample_rate, frame_size=block_size)
        self.impulse = ImpulseDetector(crest_factor_db_threshold=cfg.impulse_guard.crest_factor_db)
        self._ref_silence_lin = 10.0 ** (cfg.safeguards.reference_silence_dbfs / 20.0)
        self._logged_divergence = False
        self._ratio_ema_db: Optional[float] = None

    def reset(self) -> None:
        self.vad.reset()
        self.impulse.reset()
        self._ratio_ema_db = None

    def decide(self, d_block: np.ndarray, x_block: np.ndarray) -> tuple[float, str, bool, bool]:
        """Return (mu_scale, state_code, speech_flag, impulse_flag)."""
        cfg = self.cfg
        speech = self.vad.process_frame(d_block)
        impulsive = (
            self.impulse.process_frame(x_block) if cfg.impulse_guard.mode != "off" else False
        )
        ref_rms = float(np.sqrt(np.mean(np.square(x_block, dtype=np.float64))) + _EPS)

        if ref_rms < self._ref_silence_lin:
            return 0.0, FROZEN_SILENCE, speech, impulsive
        if speech and cfg.double_talk.mode == "freeze":
            return 0.0, FROZEN_SPEECH, speech, impulsive
        if impulsive and cfg.impulse_guard.mode == "freeze":
            return 0.0, FROZEN_IMPULSE, speech, impulsive
        if speech and cfg.double_talk.mode == "scale":
            state = CLIPPED_IMPULSE if impulsive and cfg.impulse_guard.mode == "clip" else SCALED_SPEECH
            return cfg.double_talk.mu_scale, state, speech, impulsive
        if impulsive and cfg.impulse_guard.mode == "clip":
            return 1.0, CLIPPED_IMPULSE, speech, impulsive
        return 1.0, ADAPTING, speech, impulsive

    def check_divergence(
        self, weights: np.ndarray, in_power: float, out_power: float
    ) -> tuple[bool, str]:
        """True when the filter has diverged and should be rolled back."""
        return self.check_divergence_scalar(float(np.linalg.norm(weights)), in_power, out_power)

    def check_divergence_scalar(
        self, weight_norm: float, in_power: float, out_power: float
    ) -> tuple[bool, str]:
        """Same check, for implementations that track the norm incrementally.

        The power-ratio test runs on a smoothed ratio rather than a single block. A
        transient - a gunshot arriving in the reference before its reverberant copy
        arrives in the primary - can legitimately push one block's output above its
        input without the filter having diverged at all, and rolling back on that
        would throw away good convergence. A single block is still enough to trigger
        if it is catastrophically bad.
        """
        sg = self.cfg.safeguards
        norm = weight_norm
        if not np.isfinite(norm) or norm > sg.weight_norm_limit:
            return True, f"weight norm {norm:.3g} exceeded limit {sg.weight_norm_limit:.3g}"
        if in_power > _EPS and out_power > _EPS:
            ratio_db = 10.0 * np.log10(out_power / in_power)
            self._ratio_ema_db = (
                ratio_db
                if self._ratio_ema_db is None
                else 0.75 * self._ratio_ema_db + 0.25 * ratio_db
            )
            if ratio_db > sg.divergence_margin_db + 12.0:
                return True, f"output exceeded input by {ratio_db:.1f} dB in a single block"
            if self._ratio_ema_db > sg.divergence_margin_db:
                return True, (
                    f"smoothed output/input ratio {self._ratio_ema_db:.1f} dB exceeded the "
                    f"{sg.divergence_margin_db:.1f} dB margin"
                )
        return False, ""

    def log_divergence_once(self, reason: str) -> None:
        if not self._logged_divergence:
            log.warning("NLMS divergence detected (%s); rolling back to last good state", reason)
            self._logged_divergence = True
        else:
            log.debug("NLMS divergence: %s", reason)


# ------------------------------------------------------------- time domain


class TimeDomainNlms:
    """Sample-wise NLMS. Correctness oracle for the frequency-domain filter.

    Cost is O(filter_length) per sample in Python, so this is intended for tests
    and short offline checks, not for real-time use with long filters.
    """

    def __init__(
        self,
        cfg: NlmsCfg,
        vad_cfg: Optional[VadCfg] = None,
        sample_rate: int = 48000,
        block_size: int = 480,
        use_safeguards: bool = True,
    ) -> None:
        self.cfg = cfg
        self.sample_rate = sample_rate
        self.block_size = int(block_size)
        self.filter_length = int(cfg.filter_length)
        self.use_safeguards = use_safeguards
        self.guards = (
            SafeguardEngine(cfg, vad_cfg or VadCfg(), sample_rate, self.block_size)
            if use_safeguards
            else None
        )
        self.w = np.zeros(self.filter_length, dtype=np.float64)
        self._xbuf = np.zeros(self.filter_length, dtype=np.float64)  # newest sample first
        self._power = 0.0
        self._checkpoint = self.w.copy()
        self._blocks_since_checkpoint = 0
        self.diagnostics = NlmsDiagnostics(block_size=self.block_size, sample_rate=sample_rate)
        if self.filter_length > 1024:
            log.warning(
                "time-domain NLMS with %d taps is slow (O(L) per sample); "
                "use impl='fdaf' for anything longer than a few seconds of audio",
                self.filter_length,
            )

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.w.fill(0.0)
        self._xbuf.fill(0.0)
        self._power = 0.0
        self._checkpoint = self.w.copy()
        self._blocks_since_checkpoint = 0
        self.diagnostics = NlmsDiagnostics(block_size=self.block_size, sample_rate=self.sample_rate)
        if self.guards is not None:
            self.guards.reset()

    def impulse_response(self) -> np.ndarray:
        return self.w.astype(np.float64, copy=True)

    # ------------------------------------------------------------------ block
    def process_block(self, d: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Process one block of primary/reference samples and return the residual."""
        d = np.asarray(d, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        if d.shape != x.shape:
            raise ValueError(f"primary and reference blocks differ in shape: {d.shape} vs {x.shape}")

        mu_scale, state, speech, impulsive = (
            self.guards.decide(d.astype(np.float32), x.astype(np.float32))
            if self.guards is not None
            else (1.0, ADAPTING, False, False)
        )
        mu = self.cfg.mu * mu_scale
        leak = self.cfg.leakage
        eps = self.cfg.eps
        clip_sigma = (
            self.cfg.impulse_guard.update_clip_sigma
            if (self.guards is not None and self.cfg.impulse_guard.mode == "clip")
            else None
        )

        e = np.empty_like(d)
        y = np.empty_like(d)
        w = self.w
        xbuf = self._xbuf
        for n in range(d.size):
            # Shift the newest reference sample into the front of the tap buffer.
            xbuf[1:] = xbuf[:-1]
            xbuf[0] = x[n]
            y_n = float(np.dot(w, xbuf))
            e_n = d[n] - y_n
            y[n] = y_n
            e[n] = e_n
            if mu > 0.0:
                norm = float(np.dot(xbuf, xbuf))
                step = mu * e_n / (eps + norm)
                update = step * xbuf
                if clip_sigma is not None:
                    limit = clip_sigma * float(np.sqrt(np.mean(np.square(update)))) + _EPS
                    np.clip(update, -limit, limit, out=update)
                if leak > 0.0:
                    w *= 1.0 - leak
                w += update

        self._record(d, x, e, state, mu, speech, impulsive)
        return e.astype(np.float32, copy=False)

    def _record(
        self,
        d: np.ndarray,
        x: np.ndarray,
        e: np.ndarray,
        state: str,
        mu: float,
        speech: bool,
        impulsive: bool,
    ) -> None:
        in_power = float(np.mean(np.square(d)))
        out_power = float(np.mean(np.square(e)))
        diverged = False
        if self.guards is not None and self.cfg.safeguards.rollback:
            diverged, reason = self.guards.check_divergence(self.w, in_power, out_power)
            if diverged:
                self.guards.log_divergence_once(reason)
                self.w[:] = self._checkpoint
                self.diagnostics.rollbacks += 1
                self.diagnostics.divergence_events += 1
                state = ROLLED_BACK
            else:
                self._blocks_since_checkpoint += 1
                if self._blocks_since_checkpoint >= self.cfg.safeguards.checkpoint_interval_frames:
                    self._checkpoint = self.w.copy()
                    self._blocks_since_checkpoint = 0

        erle = 10.0 * np.log10((in_power + _EPS) / (out_power + _EPS))
        diag = self.diagnostics
        diag.erle_db.append(float(erle))
        diag.weight_norm.append(float(np.linalg.norm(self.w)))
        diag.mu_effective.append(float(mu))
        diag.state.append(state)
        diag.reference_dbfs.append(
            float(20.0 * np.log10(np.sqrt(np.mean(np.square(x))) + _EPS))
        )
        diag.mse.append(out_power)
        diag.speech_flags.append(bool(speech))
        diag.impulse_flags.append(bool(impulsive))

    # ----------------------------------------------------------------- offline
    def process(self, d: np.ndarray, x: np.ndarray) -> NlmsResult:
        """Run over whole signals, block by block, exactly as the live path does."""
        d = np.asarray(d, dtype=np.float32)
        x = np.asarray(x, dtype=np.float32)
        n = min(len(d), len(x))
        n_blocks = n // self.block_size
        out = np.zeros(n_blocks * self.block_size, dtype=np.float32)
        for b in range(n_blocks):
            s = b * self.block_size
            out[s : s + self.block_size] = self.process_block(
                d[s : s + self.block_size], x[s : s + self.block_size]
            )
        est = (d[: len(out)] - out).astype(np.float32)
        return NlmsResult(
            output=out, estimated_noise=est, weights=self.impulse_response(), diagnostics=self.diagnostics
        )


def create_adaptive_filter(
    cfg: NlmsCfg,
    vad_cfg: Optional[VadCfg] = None,
    sample_rate: int = 48000,
    block_size: int = 480,
    use_safeguards: bool = True,
) -> AdaptiveFilter:
    """Factory: returns the configured NLMS implementation behind one interface."""
    if cfg.impl == "time":
        return TimeDomainNlms(cfg, vad_cfg, sample_rate, block_size, use_safeguards)
    from .fdaf import PartitionedFdafNlms

    return PartitionedFdafNlms(cfg, vad_cfg, sample_rate, block_size, use_safeguards)


def normalised_misalignment_db(estimated: np.ndarray, truth: np.ndarray) -> float:
    """10*log10(||w - h||^2 / ||h||^2): how far the filter is from the true path."""
    n = max(len(estimated), len(truth))
    a = np.zeros(n)
    b = np.zeros(n)
    a[: len(estimated)] = estimated
    b[: len(truth)] = truth
    denom = float(np.dot(b, b))
    if denom <= 0:
        return float("nan")
    return float(10.0 * np.log10((float(np.sum((a - b) ** 2)) + _EPS) / denom))


def convergence_time_s(
    erle_db: np.ndarray, block_size: int, sample_rate: int, within_db: float = 3.0
) -> float:
    """Time to come within ``within_db`` of the steady-state ERLE and stay there."""
    erle = np.asarray(erle_db, dtype=np.float64)
    finite = np.isfinite(erle)
    if finite.sum() < 10:
        return float("nan")
    erle = np.where(finite, erle, 0.0)
    tail = erle[-max(10, len(erle) // 10) :]
    steady = float(np.mean(tail))
    target = steady - within_db
    reached = np.flatnonzero(erle >= target)
    if reached.size == 0:
        return float("nan")
    # First index after which the curve never drops back below the target.
    idx = reached[0]
    for i in reached:
        if np.all(erle[i:] >= target - 1e-9):
            idx = i
            break
    return float(idx * block_size / sample_rate)
