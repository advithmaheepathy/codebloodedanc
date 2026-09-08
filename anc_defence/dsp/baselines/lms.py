"""Classical LMS adaptive noise cancellation using the reference channel.

Textbook LMS is the same structure as NLMS with the power normalisation removed::

    y[n] = w^T x[n]
    e[n] = d[n] - y[n]
    w   <- w + mu * e[n] * x[n]

The missing normalisation is the whole point of the comparison. Convergence and
stability now depend on the *absolute* input power: the stability bound is roughly
``mu < 2 / (L * E[x^2])``, so a step size tuned for one noise level diverges when
the level rises, which is precisely what happens when a gunshot or a passing vehicle
enters the reference. NLMS removes that dependence by dividing by the instantaneous
tap-vector power.

Two implementations:

:func:`time_domain_lms`
    Exact sample-wise textbook form. O(L) per sample in Python, so it is used for
    short signals and for the equivalence test.

:class:`BlockLms`
    The same fixed-step update accumulated over a block and applied in the frequency
    domain, so that long filters can be run over the whole evaluation set in
    reasonable time. The step size is still *not* power-normalised, so the classical
    weakness is preserved. This is the variant used for the reported baseline, and
    the report says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ...utils.logging import get_logger

log = get_logger(__name__)

_EPS = 1e-20

# Chosen by sweeping mu on eight real evaluation examples with a 7680-tap filter, and
# picking the largest value that was mostly stable. Larger values diverge on nearly
# every example, which is the classical weakness this baseline exists to demonstrate:
#   mu=0.050 -> 8/8 diverged      mu=0.005 -> 6/8 diverged
#   mu=0.020 -> 8/8 diverged      mu=0.002 -> 5/8 diverged
#   mu=0.010 -> 7/8 diverged      mu=0.001 -> 1/8 diverged, 3.4 dB ERLE
LMS_DEFAULT_MU = 0.001


@dataclass
class LmsResult:
    output: np.ndarray
    estimated_noise: np.ndarray
    weights: np.ndarray
    erle_db: np.ndarray = field(default_factory=lambda: np.zeros(0))
    diverged: bool = False


def time_domain_lms(
    primary: np.ndarray,
    reference: np.ndarray,
    filter_length: int = 256,
    mu: float = 0.01,
    normalise_mu_by_power: bool = True,
    divergence_limit: float = 1e6,
) -> LmsResult:
    """Sample-wise fixed-step LMS.

    Args:
        mu: step size. If ``normalise_mu_by_power`` is True, ``mu`` is interpreted
            relative to the stability bound ``2/(L*E[x^2])`` measured once at the
            start, which is how an engineer would actually pick it in practice: you
            measure the level, set the step, and it stops being valid when the level
            changes. The adaptation itself remains unnormalised.
    """
    d = np.asarray(primary, dtype=np.float64)
    x = np.asarray(reference, dtype=np.float64)
    n = min(len(d), len(x))
    d, x = d[:n], x[:n]
    L = int(filter_length)

    step = mu
    if normalise_mu_by_power:
        power = float(np.mean(x**2)) + _EPS
        step = mu * (2.0 / (L * power))

    w = np.zeros(L)
    xbuf = np.zeros(L)
    e = np.empty(n)
    y = np.empty(n)
    diverged = False
    for i in range(n):
        xbuf[1:] = xbuf[:-1]
        xbuf[0] = x[i]
        y_i = float(np.dot(w, xbuf))
        e_i = d[i] - y_i
        y[i] = y_i
        e[i] = e_i
        w += step * e_i * xbuf
        if not diverged and (not np.isfinite(e_i) or abs(e_i) > divergence_limit):
            diverged = True
            log.warning("classical LMS diverged at sample %d (step=%.3g)", i, step)
            break
    if diverged:
        e[i:] = d[i:]
        y[i:] = 0.0
    return LmsResult(
        output=np.nan_to_num(e).astype(np.float32),
        estimated_noise=np.nan_to_num(y).astype(np.float32),
        weights=w,
        diverged=diverged,
    )


class BlockLms:
    """Fixed-step LMS accumulated per block and applied via FFT.

    Same overlap-save structure as the NLMS FDAF but with a constant step size
    instead of per-bin power normalisation.
    """

    def __init__(
        self,
        filter_length: int = 3840,
        block_size: int = 480,
        mu: float = LMS_DEFAULT_MU,
        reference_power: Optional[float] = None,
        divergence_limit: float = 1e6,
    ) -> None:
        self.block_size = int(block_size)
        self.n_partitions = int(np.ceil(filter_length / self.block_size))
        self.filter_length = self.n_partitions * self.block_size
        self.fft_size = 2 * self.block_size
        self.n_bins = self.fft_size // 2 + 1
        self.mu = float(mu)
        self.divergence_limit = divergence_limit
        self.diverged = False
        self._ref_power = reference_power
        self._W = np.zeros((self.n_partitions, self.n_bins), dtype=np.complex128)
        self._Xh = np.zeros((self.n_partitions, self.n_bins), dtype=np.complex128)
        self._frame = np.zeros(self.fft_size, dtype=np.float64)
        self._err = np.zeros(self.fft_size, dtype=np.float64)

    def impulse_response(self) -> np.ndarray:
        taps = np.zeros(self.filter_length)
        for p in range(self.n_partitions):
            taps[p * self.block_size : (p + 1) * self.block_size] = np.fft.irfft(
                self._W[p], n=self.fft_size
            )[: self.block_size]
        return taps

    def process(self, primary: np.ndarray, reference: np.ndarray) -> LmsResult:
        d = np.asarray(primary, dtype=np.float64)
        x = np.asarray(reference, dtype=np.float64)
        n = min(len(d), len(x))
        B = self.block_size
        n_blocks = n // B

        # Step size scaled once by the measured reference power, mirroring how a
        # fixed-step LMS is tuned in practice. It stays fixed thereafter.
        power = self._ref_power if self._ref_power is not None else float(np.mean(x[:n] ** 2)) + _EPS
        step = self.mu * (2.0 / (self.filter_length * power))

        out = np.zeros(n_blocks * B, dtype=np.float32)
        erle = np.zeros(n_blocks)
        for b in range(n_blocks):
            s = b * B
            self._frame[:B] = self._frame[B:]
            self._frame[B:] = x[s : s + B]
            X = np.fft.rfft(self._frame)
            if self.n_partitions > 1:
                self._Xh[1:] = self._Xh[:-1]
            self._Xh[0] = X

            Y = np.einsum("pf,pf->f", self._W, self._Xh)
            y = np.fft.irfft(Y, n=self.fft_size)[B:]
            e = d[s : s + B] - y
            out[s : s + B] = e.astype(np.float32)

            in_p = float(np.mean(d[s : s + B] ** 2)) + _EPS
            out_p = float(np.mean(e**2)) + _EPS
            erle[b] = 10.0 * np.log10(in_p / out_p)

            if not np.isfinite(out_p) or out_p > self.divergence_limit:
                self.diverged = True
                log.warning(
                    "block LMS diverged at block %d of %d (step=%.3g); disabling the filter for "
                    "this example and passing the primary through unchanged",
                    b,
                    n_blocks,
                    step,
                )
                # A diverged fixed-step filter is switched off, which is what a fielded
                # system would have to do. Returning the partially-exploded signal
                # instead would make the baseline look worse than the algorithm is.
                out[:] = d[: len(out)].astype(np.float32)
                erle[:] = 0.0
                break

            self._err[:B] = 0.0
            self._err[B:] = e
            E = np.fft.rfft(self._err)
            grad = self._Xh.conj() * E
            g = np.fft.irfft(grad, n=self.fft_size, axis=1)
            g[:, B:] = 0.0
            self._W += step * np.fft.rfft(g, axis=1)

        est = (d[: len(out)] - out).astype(np.float32)
        return LmsResult(
            output=out,
            estimated_noise=est,
            weights=self.impulse_response(),
            erle_db=erle,
            diverged=self.diverged,
        )


def lms_cancel(
    primary: np.ndarray,
    reference: np.ndarray,
    filter_length: int = 3840,
    block_size: int = 480,
    mu: float = 0.05,
) -> np.ndarray:
    """Convenience wrapper returning just the enhanced signal."""
    return BlockLms(filter_length, block_size, mu).process(primary, reference).output
