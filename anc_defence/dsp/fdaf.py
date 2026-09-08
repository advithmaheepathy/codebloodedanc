"""Partitioned-block frequency-domain NLMS (overlap-save FDAF / multi-delay filter).

This is the real-time default. At 48 kHz a useful filter spans thousands of taps
(3840 taps = 80 ms), which sample-wise NLMS cannot deliver inside a 10 ms block
budget in Python. The frequency-domain form splits the filter into ``P`` partitions
of ``B`` taps and replaces the per-sample convolution with a handful of FFTs.

Structure (block index ``m``, block size ``B``, FFT size ``N = 2B``):

    frame_m  = [x_{m-1} , x_m]                        (2B samples, overlap-save)
    X_m      = rfft(frame_m)
    Y_m      = sum_p  W_p * X_{m-p}
    y_m      = irfft(Y_m)[B:]                          (aliasing lands in [0:B])
    e_m      = d_m - y_m
    E_m      = rfft([0 , e_m])
    grad_p   = conj(X_{m-p}) * E_m                     (= sum_n e[n] x[n-k])
    W_p     <- (1-leak) W_p + mu * grad_p / (power + eps)

The gradient is constrained (its second half is zeroed in the time domain) so the
filter is equivalent to the linear-convolution NLMS of :mod:`anc_defence.dsp.nlms`
rather than a circular approximation. The per-bin power is halved because the
2B-point transform of a B-sample block doubles the apparent power, which keeps the
meaning of ``mu`` close to the sample-wise implementation.

Latency: none beyond the block itself. Overlap-save produces the current block's
output from the current block's input.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..config import NlmsCfg, VadCfg
from ..utils.logging import get_logger
from .nlms import (
    ADAPTING,
    CLIPPED_IMPULSE,
    ROLLED_BACK,
    NlmsDiagnostics,
    NlmsResult,
    SafeguardEngine,
)

log = get_logger(__name__)

_EPS = 1e-20


class PartitionedFdafNlms:
    """Frequency-domain NLMS with the same safeguards as the time-domain filter."""

    def __init__(
        self,
        cfg: NlmsCfg,
        vad_cfg: Optional[VadCfg] = None,
        sample_rate: int = 48000,
        block_size: int = 480,
        use_safeguards: bool = True,
        constrained: bool = True,
    ) -> None:
        self.cfg = cfg
        self.sample_rate = sample_rate
        self.block_size = int(block_size)
        self.constrained = constrained
        self.n_partitions = int(np.ceil(cfg.filter_length / self.block_size))
        self.filter_length = self.n_partitions * self.block_size
        if self.filter_length != cfg.filter_length:
            log.info(
                "filter_length %d rounded up to %d (%d partitions of %d taps) for the FDAF",
                cfg.filter_length,
                self.filter_length,
                self.n_partitions,
                self.block_size,
            )
        self.fft_size = 2 * self.block_size
        self.n_bins = self.fft_size // 2 + 1

        self.guards = (
            SafeguardEngine(cfg, vad_cfg or VadCfg(), sample_rate, self.block_size)
            if use_safeguards
            else None
        )

        # Preallocated state: no allocation in the steady-state block path.
        self._W = np.zeros((self.n_partitions, self.n_bins), dtype=np.complex128)
        self._Xh = np.zeros((self.n_partitions, self.n_bins), dtype=np.complex128)
        self._frame = np.zeros(self.fft_size, dtype=np.float64)
        self._err_frame = np.zeros(self.fft_size, dtype=np.float64)
        self._power = np.zeros(self.n_bins, dtype=np.float64)
        self._checkpoint = self._W.copy()
        self._blocks_since_checkpoint = 0
        self._power_initialised = False

        self.diagnostics = NlmsDiagnostics(block_size=self.block_size, sample_rate=sample_rate)

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self._W.fill(0)
        self._Xh.fill(0)
        self._frame.fill(0.0)
        self._err_frame.fill(0.0)
        self._power.fill(0.0)
        self._power_initialised = False
        self._checkpoint = self._W.copy()
        self._blocks_since_checkpoint = 0
        self.diagnostics = NlmsDiagnostics(block_size=self.block_size, sample_rate=self.sample_rate)
        if self.guards is not None:
            self.guards.reset()

    def impulse_response(self) -> np.ndarray:
        """Equivalent time-domain filter, ``filter_length`` taps."""
        taps = np.zeros(self.filter_length, dtype=np.float64)
        for p in range(self.n_partitions):
            block = np.fft.irfft(self._W[p], n=self.fft_size)
            taps[p * self.block_size : (p + 1) * self.block_size] = block[: self.block_size]
        return taps

    @property
    def weights(self) -> np.ndarray:
        return self.impulse_response()

    def _weight_norm(self) -> float:
        """L2 norm of the equivalent time-domain filter, via Parseval.

        Cheaper than an inverse FFT per block: for a real signal of length N,
        sum |g[n]|^2 = (1/N) (|G_0|^2 + 2*sum_{k=1}^{N/2-1}|G_k|^2 + |G_{N/2}|^2).
        """
        mag2 = np.abs(self._W) ** 2
        total = mag2[:, 0].sum() + mag2[:, -1].sum() + 2.0 * mag2[:, 1:-1].sum()
        return float(np.sqrt(total / self.fft_size))

    # ------------------------------------------------------------------ block
    def process_block(self, d: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Process one block and return the residual (the enhanced output)."""
        B = self.block_size
        d64 = np.asarray(d, dtype=np.float64)
        x64 = np.asarray(x, dtype=np.float64)
        if d64.size != B or x64.size != B:
            raise ValueError(f"FDAF expects blocks of exactly {B} samples, got {d64.size}/{x64.size}")

        # Slide the overlap-save frame: [previous block, current block].
        self._frame[:B] = self._frame[B:]
        self._frame[B:] = x64
        X = np.fft.rfft(self._frame)

        # Newest input spectrum first: _Xh[p] holds X_{m-p}.
        if self.n_partitions > 1:
            self._Xh[1:] = self._Xh[:-1]
        self._Xh[0] = X

        Y = np.einsum("pf,pf->f", self._W, self._Xh)
        y = np.fft.irfft(Y, n=self.fft_size)[B:]
        e = d64 - y

        mu_scale, state, speech, impulsive = (
            self.guards.decide(d64.astype(np.float32), x64.astype(np.float32))
            if self.guards is not None
            else (1.0, ADAPTING, False, False)
        )
        mu = self.cfg.mu * mu_scale

        # Track per-bin reference power across all partitions.
        inst_power = np.einsum("pf,pf->f", self._Xh, self._Xh.conj()).real
        if not self._power_initialised:
            self._power[:] = inst_power
            self._power_initialised = True
        else:
            a = self.cfg.power_smoothing
            self._power *= a
            self._power += (1.0 - a) * inst_power

        if mu > 0.0:
            self._err_frame[:B] = 0.0
            self._err_frame[B:] = e
            E = np.fft.rfft(self._err_frame)
            # Halving compensates the 2B-point transform of a B-sample block.
            denom = 0.5 * self._power + self.cfg.eps + _EPS
            grad = self._Xh.conj() * (E / denom)
            if self.constrained:
                g = np.fft.irfft(grad, n=self.fft_size, axis=1)
                g[:, B:] = 0.0
                if impulsive and self.cfg.impulse_guard.mode == "clip":
                    limit = self.cfg.impulse_guard.update_clip_sigma * float(
                        np.sqrt(np.mean(np.square(g)))
                    ) + _EPS
                    np.clip(g, -limit, limit, out=g)
                grad = np.fft.rfft(g, axis=1)
            if self.cfg.leakage > 0.0:
                self._W *= 1.0 - self.cfg.leakage
            self._W += mu * grad

        self._record(d64, x64, e, state, mu, speech, impulsive)
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
        w_norm = self._weight_norm()
        if self.guards is not None and self.cfg.safeguards.rollback:
            diverged, reason = self.guards.check_divergence_scalar(w_norm, in_power, out_power)
            if diverged:
                self.guards.log_divergence_once(reason)
                self._W[:] = self._checkpoint
                self.diagnostics.rollbacks += 1
                self.diagnostics.divergence_events += 1
                state = ROLLED_BACK
            else:
                self._blocks_since_checkpoint += 1
                if self._blocks_since_checkpoint >= self.cfg.safeguards.checkpoint_interval_frames:
                    self._checkpoint = self._W.copy()
                    self._blocks_since_checkpoint = 0

        erle = 10.0 * np.log10((in_power + _EPS) / (out_power + _EPS))
        diag = self.diagnostics
        diag.erle_db.append(float(erle))
        diag.weight_norm.append(w_norm)
        diag.mu_effective.append(float(mu))
        diag.state.append(state if state != CLIPPED_IMPULSE or self.cfg.impulse_guard.mode == "clip" else ADAPTING)
        diag.reference_dbfs.append(float(20.0 * np.log10(np.sqrt(np.mean(np.square(x))) + _EPS)))
        diag.mse.append(out_power)
        diag.speech_flags.append(bool(speech))
        diag.impulse_flags.append(bool(impulsive))

    # ---------------------------------------------------------------- offline
    def process(self, d: np.ndarray, x: np.ndarray) -> NlmsResult:
        """Run over whole signals block by block, identical to the live path."""
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
            output=out,
            estimated_noise=est,
            weights=self.impulse_response(),
            diagnostics=self.diagnostics,
        )
