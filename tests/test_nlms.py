"""Correctness tests for the adaptive stage.

The reference problem is a known FIR path: the primary channel is the reference
signal convolved with a known impulse response, so a correctly working filter must
converge to that impulse response and cancel the primary almost completely.
"""

from __future__ import annotations

import numpy as np
import pytest

from anc_defence.config import ImpulseGuardCfg, NlmsCfg, NlmsSafeguardCfg, VadCfg
from anc_defence.dsp.fdaf import PartitionedFdafNlms
from anc_defence.dsp.nlms import (
    TimeDomainNlms,
    convergence_time_s,
    normalised_misalignment_db,
)

SR = 48000
BLOCK = 480


def known_fir(n_taps: int = 96, seed: int = 7) -> np.ndarray:
    """Decaying random impulse response standing in for an acoustic path."""
    rng = np.random.default_rng(seed)
    h = rng.standard_normal(n_taps) * np.exp(-np.arange(n_taps) / (n_taps / 4.0))
    return (h / np.linalg.norm(h)).astype(np.float64)


def fir_problem(
    n_samples: int, h: np.ndarray, seed: int = 11, noise_floor: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Reference (white noise) and primary (reference convolved with ``h``)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n_samples).astype(np.float32) * 0.1
    d = np.convolve(x, h)[:n_samples].astype(np.float32)
    if noise_floor > 0:
        d = d + rng.standard_normal(n_samples).astype(np.float32) * noise_floor
    return d, x


def plain_cfg(filter_length: int, mu: float = 0.5) -> NlmsCfg:
    """NLMS with all safeguards neutral, for pure-algorithm tests."""
    return NlmsCfg(
        filter_length=filter_length,
        mu=mu,
        eps=1e-8,
        leakage=0.0,
        power_smoothing=0.5,
        impulse_guard=ImpulseGuardCfg(mode="off"),
        safeguards=NlmsSafeguardCfg(rollback=False, reference_silence_dbfs=-200.0),
    )


# --------------------------------------------------------------------------- 1


def test_time_domain_nlms_converges_on_known_fir():
    h = known_fir(96)
    d, x = fir_problem(SR // 2, h)  # 0.5 s keeps the O(L) loop fast
    cfg = plain_cfg(filter_length=128, mu=0.5)
    filt = TimeDomainNlms(cfg, VadCfg(enabled=False), SR, BLOCK, use_safeguards=False)
    result = filt.process(d, x)

    erle = result.erle_curve
    final_erle = float(np.mean(erle[-10:]))
    misalignment = normalised_misalignment_db(result.weights[: len(h)], h)

    assert final_erle > 25.0, f"time-domain NLMS ERLE too low: {final_erle:.1f} dB"
    assert misalignment < -20.0, f"filter did not approach the true path: {misalignment:.1f} dB"


# --------------------------------------------------------------------------- 2


def test_fdaf_converges_and_matches_time_domain():
    h = known_fir(96)
    d, x = fir_problem(SR, h)
    cfg = plain_cfg(filter_length=BLOCK, mu=0.5)

    td = TimeDomainNlms(cfg, VadCfg(enabled=False), SR, BLOCK, use_safeguards=False)
    fd = PartitionedFdafNlms(cfg, VadCfg(enabled=False), SR, BLOCK, use_safeguards=False)

    r_td = td.process(d[: SR // 2], x[: SR // 2])
    r_fd = fd.process(d, x)

    erle_td = float(np.mean(r_td.erle_curve[-10:]))
    erle_fd = float(np.mean(r_fd.erle_curve[-10:]))
    mis_td = normalised_misalignment_db(r_td.weights[: len(h)], h)
    mis_fd = normalised_misalignment_db(r_fd.weights[: len(h)], h)

    assert erle_fd > 25.0, f"FDAF ERLE too low: {erle_fd:.1f} dB"
    assert mis_fd < -20.0, f"FDAF misalignment too high: {mis_fd:.1f} dB"
    # Both implementations must identify the same physical path.
    assert abs(mis_fd - mis_td) < 12.0, (
        f"implementations disagree: time {mis_td:.1f} dB vs freq {mis_fd:.1f} dB"
    )
    assert erle_td > 25.0


# --------------------------------------------------------------------------- 3


def test_fdaf_long_filter_covers_longer_path():
    """A path longer than one partition needs several partitions to be identified."""
    h = known_fir(1200, seed=3)
    d, x = fir_problem(3 * SR, h)
    cfg = plain_cfg(filter_length=1440, mu=0.5)
    fd = PartitionedFdafNlms(cfg, VadCfg(enabled=False), SR, BLOCK, use_safeguards=False)
    result = fd.process(d, x)

    assert fd.n_partitions == 3
    final_erle = float(np.mean(result.erle_curve[-20:]))
    assert final_erle > 20.0, f"multi-partition FDAF ERLE too low: {final_erle:.1f} dB"
    assert normalised_misalignment_db(result.weights[: len(h)], h) < -15.0


# --------------------------------------------------------------------------- 4


def test_double_talk_freeze_protects_speech():
    """With speech in the primary and none in the reference, freezing adaptation
    must stop the filter from learning to cancel the speech."""
    rng = np.random.default_rng(5)
    n = 2 * SR
    h = known_fir(96)
    x = (rng.standard_normal(n) * 0.05).astype(np.float32)
    noise_at_mic = np.convolve(x, h)[:n].astype(np.float32)

    t = np.arange(n) / SR
    speech = (0.2 * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)
    speech[: n // 2] = 0.0  # talker silent for the first half
    d = noise_at_mic + speech

    frozen = NlmsCfg(
        filter_length=BLOCK,
        mu=0.5,
        eps=1e-8,
        power_smoothing=0.5,
        impulse_guard=ImpulseGuardCfg(mode="off"),
        safeguards=NlmsSafeguardCfg(rollback=False, reference_silence_dbfs=-200.0),
    )
    frozen.double_talk.mode = "freeze"
    unprotected = frozen.model_copy(deep=True)
    unprotected.double_talk.mode = "off"

    def speech_error(cfg: NlmsCfg) -> float:
        filt = PartitionedFdafNlms(cfg, VadCfg(), SR, BLOCK, use_safeguards=True)
        out = filt.process(d, x).output
        half = len(out) // 2
        # How much of the clean speech survived in the second half.
        target = speech[half : len(out)]
        got = out[half:]
        return float(np.mean((got - target) ** 2) / (np.mean(target**2) + 1e-20))

    err_frozen = speech_error(frozen)
    err_open = speech_error(unprotected)

    assert err_frozen < err_open, (
        f"double-talk freeze did not help: frozen {err_frozen:.4f} vs open {err_open:.4f}"
    )
    assert err_frozen < 0.25, f"speech badly distorted even with freeze: {err_frozen:.4f}"


# --------------------------------------------------------------------------- 5


def test_impulse_robust_update_does_not_diverge():
    """A burst of gunshot-like impulses in the reference must not blow the filter up."""
    rng = np.random.default_rng(9)
    n = 2 * SR
    h = known_fir(96)
    x = (rng.standard_normal(n) * 0.02).astype(np.float32)
    # Machine-gun burst: 8 impulses, 0.1 s apart, ~30 dB above the background.
    for k in range(8):
        idx = SR // 2 + int(k * 0.1 * SR)
        x[idx : idx + 40] += rng.standard_normal(40).astype(np.float32) * 0.9
    d = np.convolve(x, h)[:n].astype(np.float32)

    cfg = NlmsCfg(filter_length=BLOCK, mu=0.5, eps=1e-8, power_smoothing=0.5)
    cfg.impulse_guard = ImpulseGuardCfg(mode="clip", update_clip_sigma=4.0)
    cfg.double_talk.mode = "off"
    filt = PartitionedFdafNlms(cfg, VadCfg(enabled=False), SR, BLOCK, use_safeguards=True)
    result = filt.process(d, x)

    assert np.all(np.isfinite(result.output)), "impulse burst produced non-finite output"
    assert np.all(np.isfinite(result.weights))
    peak_out = float(np.max(np.abs(result.output)))
    peak_in = float(np.max(np.abs(d)))
    assert peak_out < 5.0 * peak_in, f"output blew up: {peak_out:.3f} vs input {peak_in:.3f}"
    final_erle = float(np.mean(result.erle_curve[-20:]))
    assert final_erle > 10.0, f"filter lost convergence after impulses: {final_erle:.1f} dB"


# --------------------------------------------------------------------------- 6


def test_reference_silence_guard_freezes_adaptation():
    n = SR
    d = (np.random.default_rng(1).standard_normal(n) * 0.05).astype(np.float32)
    x = np.zeros(n, dtype=np.float32)  # silent reference
    cfg = NlmsCfg(filter_length=BLOCK, mu=0.5)
    filt = PartitionedFdafNlms(cfg, VadCfg(), SR, BLOCK, use_safeguards=True)
    result = filt.process(d, x)

    assert np.allclose(result.weights, 0.0), "filter adapted on a silent reference"
    assert np.allclose(result.output, d[: len(result.output)], atol=1e-6)


# --------------------------------------------------------------------------- 7


def test_convergence_time_is_measurable():
    h = known_fir(96)
    d, x = fir_problem(SR, h)
    cfg = plain_cfg(filter_length=BLOCK, mu=0.5)
    fd = PartitionedFdafNlms(cfg, VadCfg(enabled=False), SR, BLOCK, use_safeguards=False)
    result = fd.process(d, x)
    t_conv = convergence_time_s(result.erle_curve, BLOCK, SR, within_db=3.0)
    assert np.isfinite(t_conv)
    assert 0.0 < t_conv < 1.0


@pytest.mark.parametrize("filter_length", [480, 960])
def test_impulse_response_length(filter_length: int):
    cfg = plain_cfg(filter_length=filter_length)
    fd = PartitionedFdafNlms(cfg, VadCfg(enabled=False), SR, BLOCK, use_safeguards=False)
    assert fd.impulse_response().shape == (filter_length,)
