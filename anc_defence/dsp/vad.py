"""Voice activity detection.

Purpose here is not transcription-grade VAD: it exists to tell the adaptive filter
when the talker is active, because adapting during speech is what makes NLMS cancel
speech instead of noise.

The detector combines two cheap cues:

* frame energy relative to a tracked noise floor (fast down, slow up), and
* spectral flatness, which is low for voiced speech and high for broadband noise.

A frame counts as speech when energy exceeds the floor by ``energy_margin_db`` and
flatness is below ``flatness_threshold``. A hangover keeps the decision latched so
adaptation does not restart inside short pauses between words.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import VadCfg
from ..utils.logging import get_logger

log = get_logger(__name__)

_EPS = 1e-12


@dataclass
class VadState:
    speech: bool = False
    energy_db: float = -120.0
    noise_floor_db: float = -120.0
    flatness: float = 1.0
    hangover_left: int = 0
    speech_run: int = 0


class EnergyFlatnessVad:
    """Streaming frame-based VAD with persistent noise-floor tracking."""

    def __init__(self, cfg: VadCfg, sample_rate: int, frame_size: Optional[int] = None) -> None:
        self.cfg = cfg
        self.sample_rate = sample_rate
        self.frame_size = int(frame_size or round(sample_rate * cfg.frame_ms / 1000.0))
        self._hangover_frames = max(0, int(round(cfg.hangover_ms / cfg.frame_ms)))
        self._min_speech_frames = max(1, int(round(cfg.min_speech_ms / cfg.frame_ms)))
        self._window = np.hanning(self.frame_size).astype(np.float32)
        self.state = VadState()
        self._initialised = False

    def reset(self) -> None:
        self.state = VadState()
        self._initialised = False

    # ------------------------------------------------------------------ frame
    def process_frame(self, frame: np.ndarray) -> bool:
        """Update the detector with one frame and return the speech decision."""
        if not self.cfg.enabled:
            return False
        x = np.asarray(frame, dtype=np.float32)
        energy = float(np.mean(np.square(x, dtype=np.float64)))
        energy_db = 10.0 * np.log10(energy + _EPS)

        if not self._initialised:
            self.state.noise_floor_db = energy_db
            self._initialised = True
        else:
            a = self.cfg.noise_floor_smoothing
            if energy_db < self.state.noise_floor_db:
                # Fall quickly toward a new, quieter floor.
                self.state.noise_floor_db = 0.7 * self.state.noise_floor_db + 0.3 * energy_db
            else:
                self.state.noise_floor_db = a * self.state.noise_floor_db + (1.0 - a) * energy_db

        flatness = spectral_flatness(x * self._window[: len(x)] if len(x) == self.frame_size else x)
        loud = energy_db > self.state.noise_floor_db + self.cfg.energy_margin_db
        peaky = flatness < self.cfg.flatness_threshold
        raw = bool(loud and peaky)

        if raw:
            self.state.speech_run += 1
        else:
            self.state.speech_run = 0

        if self.state.speech_run >= self._min_speech_frames:
            self.state.hangover_left = self._hangover_frames
            speech = True
        elif self.state.hangover_left > 0:
            self.state.hangover_left -= 1
            speech = True
        else:
            speech = False

        self.state.energy_db = energy_db
        self.state.flatness = flatness
        self.state.speech = speech
        return speech

    # ---------------------------------------------------------------- offline
    def process(self, x: np.ndarray, hop: Optional[int] = None) -> np.ndarray:
        """Run over a whole signal. Returns a bool array, one entry per frame."""
        hop = hop or self.frame_size
        n_frames = max(0, 1 + (len(x) - self.frame_size) // hop) if len(x) >= self.frame_size else 0
        out = np.zeros(n_frames, dtype=bool)
        for i in range(n_frames):
            out[i] = self.process_frame(x[i * hop : i * hop + self.frame_size])
        return out


def spectral_flatness(frame: np.ndarray) -> float:
    """Geometric mean divided by arithmetic mean of the power spectrum, in [0, 1].

    Near 1 for white noise, near 0 for a pure tone or strongly voiced speech.
    """
    if frame.size == 0:
        return 1.0
    spec = np.abs(np.fft.rfft(frame.astype(np.float64)))
    power = np.square(spec) + _EPS
    # Skip DC, which carries no information about voicing.
    power = power[1:]
    if power.size == 0:
        return 1.0
    geo = np.exp(np.mean(np.log(power)))
    arith = np.mean(power)
    return float(np.clip(geo / (arith + _EPS), 0.0, 1.0))


def frames_to_samples(mask: np.ndarray, frame_size: int, hop: int, n_samples: int) -> np.ndarray:
    """Expand a per-frame boolean mask to a per-sample mask."""
    out = np.zeros(n_samples, dtype=bool)
    for i, flag in enumerate(mask):
        if flag:
            start = i * hop
            out[start : min(start + frame_size, n_samples)] = True
    return out


def speech_mask(
    x: np.ndarray, cfg: VadCfg, sample_rate: int, hop: Optional[int] = None
) -> np.ndarray:
    """Convenience wrapper: per-sample speech mask for a whole signal."""
    vad = EnergyFlatnessVad(cfg, sample_rate)
    hop = hop or vad.frame_size
    mask = vad.process(x, hop=hop)
    return frames_to_samples(mask, vad.frame_size, hop, len(x))
