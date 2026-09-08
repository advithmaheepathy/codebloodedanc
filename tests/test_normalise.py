"""Tests for the volume normalisation stage.

The properties that matter for this stage are: it reaches the target level, it does
not pump the noise up during pauses, it does not clip, and it is exactly invertible so
quality metrics can be separated from level.
"""

from __future__ import annotations

import numpy as np
import pytest

from anc_defence.config import NormaliseCfg, VadCfg
from anc_defence.dsp.normalise import (
    Normaliser,
    SpeechAgc,
    active_speech_dbfs,
    peak_normalise,
    rms_normalise,
)

SR = 48000
FRAME = 480


def speech_like(
    duration_s: float = 4.0,
    level_dbfs: float = -40.0,
    sample_rate: int = SR,
    seed: int = 0,
    pause_from: float = 0.5,
    pause_floor_dbfs: float = -90.0,
) -> np.ndarray:
    """Amplitude-modulated harmonic stack, then a pause with a residual noise floor.

    Not real speech, but it has the syllabic envelope and harmonic structure the VAD
    and the level tracker key off, which is what these tests exercise.

    ``pause_floor_dbfs`` matters: after neural suppression the residual noise in a pause
    typically sits around -50 to -40 dBFS, which is *above* the AGC's gate. A pause far
    below the gate is protected by the gate alone and cannot show the difference between
    holding the gain and letting it chase the floor.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * sample_rate)
    t = np.arange(n) / sample_rate
    f0 = 130.0
    sig = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12))
    envelope = 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 3.0 * t))
    sig = (sig * envelope).astype(np.float32)

    cut = int(pause_from * n)
    rms = float(np.sqrt(np.mean(np.square(sig[:cut].astype(np.float64)))))
    sig = (sig * (10.0 ** (level_dbfs / 20.0) / (rms + 1e-20))).astype(np.float32)

    noise = rng.standard_normal(n - cut).astype(np.float32)
    noise *= 10.0 ** (pause_floor_dbfs / 20.0) / (
        float(np.sqrt(np.mean(np.square(noise.astype(np.float64))))) + 1e-20
    )
    sig[cut:] = noise
    return sig


# --------------------------------------------------------------------------- 1


@pytest.mark.parametrize("input_level_dbfs", [-45.0, -35.0, -20.0])
def test_agc_reaches_the_target_level(input_level_dbfs: float):
    """Whatever the talker's level, the output should land near the target."""
    cfg = NormaliseCfg(target_dbfs=-26.0)
    x = speech_like(level_dbfs=input_level_dbfs)
    agc = SpeechAgc(cfg, VadCfg(), SR, FRAME)
    y = agc.process(x)

    # Measure over the active half only: the pause is not meant to be brought up.
    half = len(x) // 2
    achieved = active_speech_dbfs(y[:half], SR)
    assert abs(achieved - cfg.target_dbfs) < 4.0, (
        f"input {input_level_dbfs} dBFS -> output {achieved:.1f} dBFS, "
        f"target {cfg.target_dbfs} dBFS"
    )


# --------------------------------------------------------------------------- 2


def test_agc_holds_gain_during_pauses():
    """The gain must not climb during a pause, or residual noise gets pumped up.

    This is the difference between an AGC that helps and one that makes suppressed
    audio sound worse than it measures.
    """
    # The pause floor has to sit in the window where the two behaviours differ: above
    # the AGC's gate (-60 dBFS), so the level tracker is allowed to see it, and more
    # than active_range_db (20 dB) below the speech, so it is genuinely a pause.
    # -35 dBFS speech with a -58 dBFS floor is 23 dB down and above the gate.
    x = speech_like(level_dbfs=-35.0, pause_from=0.5, pause_floor_dbfs=-58.0)

    holding = SpeechAgc(NormaliseCfg(hold_during_pause=True), VadCfg(), SR, FRAME)
    y_hold = holding.process(x)
    pumping = SpeechAgc(NormaliseCfg(hold_during_pause=False), VadCfg(), SR, FRAME)
    y_pump = pumping.process(x)

    half = len(x) // 2
    noise_hold = active_speech_dbfs(y_hold[half:], SR)
    noise_pump = active_speech_dbfs(y_pump[half:], SR)

    assert holding.diagnostics.held_frames > 0, "no frames were held: no pause was detected"
    assert noise_hold < noise_pump - 2.0, (
        f"holding did not keep the pause quieter: held {noise_hold:.1f} dBFS vs "
        f"free-running {noise_pump:.1f} dBFS"
    )

    # And the gain trace itself must not rise through the pause.
    gains = np.asarray(holding.diagnostics.gain_db)
    pause_gains = gains[len(gains) // 2 :]
    assert pause_gains.max() - pause_gains.min() < 3.0, "gain moved during the pause"


# --------------------------------------------------------------------------- 3


def test_gain_envelope_inverts_exactly():
    """Dividing the output by the recorded envelope must recover the input.

    This is what lets quality metrics be computed without the normaliser's gain being
    scored as distortion.
    """
    x = speech_like(level_dbfs=-40.0)
    agc = SpeechAgc(NormaliseCfg(), VadCfg(), SR, FRAME)
    y = agc.process(x)
    env = agc.gain_envelope(len(y))

    recovered = y / np.maximum(env, 1e-9)
    n = min(len(x), len(recovered))
    # Ignore the first and last frame: the look-ahead delay flush is not invertible there.
    a = x[FRAME : n - FRAME]
    b = recovered[FRAME : n - FRAME]
    error = float(np.sqrt(np.mean((a - b) ** 2))) / (float(np.sqrt(np.mean(a**2))) + 1e-20)
    assert error < 0.02, f"envelope inversion error {error:.4f} is too large"


# --------------------------------------------------------------------------- 4


def test_agc_output_does_not_clip():
    """A loud input must be limited, not clipped."""
    x = speech_like(level_dbfs=-6.0)
    x = np.clip(x * 3.0, -1.5, 1.5).astype(np.float32)  # deliberately hot
    cfg = NormaliseCfg(limiter_ceiling_dbfs=-1.0)
    agc = SpeechAgc(cfg, VadCfg(), SR, FRAME)
    y = agc.process(x)

    ceiling = 10.0 ** (cfg.limiter_ceiling_dbfs / 20.0)
    peak = float(np.max(np.abs(y)))
    assert peak <= 1.0, f"output exceeded full scale: {peak:.4f}"
    assert agc.diagnostics.clipped_samples == 0, (
        f"{agc.diagnostics.clipped_samples} samples hit the safety clamp; the limiter "
        "should have caught them first"
    )
    assert peak <= ceiling * 1.5, f"peak {peak:.4f} is well above the ceiling {ceiling:.4f}"


# --------------------------------------------------------------------------- 5


def test_agc_preserves_length_and_alignment():
    """Output length matches input, and the limiter delay is compensated offline.

    An uncompensated delay leaves PESQ untouched (it realigns internally) while
    destroying SI-SDR, which looks like a quality collapse when only timing changed.
    """
    x = speech_like(duration_s=3.0, level_dbfs=-30.0)
    agc = SpeechAgc(NormaliseCfg(limiter_lookahead_ms=5.0), VadCfg(), SR, FRAME)
    y = agc.process(x)
    assert len(y) == len(x)

    # Cross-correlate to confirm the peak is at zero lag.
    n = 2 ** int(np.ceil(np.log2(2 * len(x))))
    corr = np.fft.irfft(np.fft.rfft(y, n) * np.conj(np.fft.rfft(x, n)), n=n)
    lag = int(np.argmax(np.abs(corr[: SR // 10])))
    assert lag < FRAME, f"output is offset by {lag} samples ({1000 * lag / SR:.1f} ms)"


# --------------------------------------------------------------------------- 6


def test_static_modes():
    x = speech_like(level_dbfs=-40.0)

    y_peak, gain_peak = peak_normalise(x, ceiling_dbfs=-1.0)
    assert abs(20 * np.log10(np.max(np.abs(y_peak))) - (-1.0)) < 0.1
    assert gain_peak > 0

    y_rms, _ = rms_normalise(x, target_dbfs=-26.0, sample_rate=SR)
    half = len(x) // 2
    assert abs(active_speech_dbfs(y_rms[:half], SR) - (-26.0)) < 2.0


def test_off_mode_is_a_passthrough():
    x = speech_like()
    norm = Normaliser(NormaliseCfg(mode="off"), VadCfg(), SR, FRAME)
    assert not norm.enabled
    assert np.allclose(norm.process(x), x)


# --------------------------------------------------------------------------- 7


def test_block_and_whole_signal_agree():
    """The live path and the offline path must produce the same samples."""
    x = speech_like(duration_s=2.0, level_dbfs=-35.0)
    n_blocks = len(x) // FRAME
    x = x[: n_blocks * FRAME]

    whole = SpeechAgc(NormaliseCfg(), VadCfg(), SR, FRAME).process(x, align=False)
    streaming = SpeechAgc(NormaliseCfg(), VadCfg(), SR, FRAME)
    parts = [streaming.process_block(x[i * FRAME : (i + 1) * FRAME]) for i in range(n_blocks)]
    blocked = np.concatenate(parts)

    n = min(len(whole), len(blocked))
    assert np.allclose(whole[:n], blocked[:n], atol=1e-6), (
        "block-by-block processing diverged from whole-signal processing"
    )


def test_max_gain_is_respected():
    """A near-silent input must not be amplified without limit."""
    cfg = NormaliseCfg(max_gain_db=12.0, target_dbfs=-26.0)
    x = (speech_like(level_dbfs=-70.0)).astype(np.float32)
    agc = SpeechAgc(cfg, VadCfg(), SR, FRAME)
    agc.process(x)
    gains = np.asarray(agc.diagnostics.gain_db)
    assert gains.max() <= cfg.max_gain_db + 0.5, f"gain reached {gains.max():.1f} dB"
