"""Room impulse responses for the noise path.

This is the piece that stops the adaptive stage from being trivially easy. If the
NLMS reference were the same waveform that was added to the primary channel, the
filter would only have to learn a gain and a delay, and the measured ERLE would say
nothing about whether the algorithm works. Instead:

* the noise is convolved with a simulated room impulse response before it is added
  to the primary channel, and
* the NLMS reference is the **dry** noise.

The filter therefore has to identify a real acoustic path, which is the problem it
actually solves in the field. Every mixture records the RIR's room dimensions, RT60
and significant length so the report can show that the configured filter length
covers the path.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import RirCfg
from ..utils.logging import get_logger

log = get_logger(__name__)

_EPS = 1e-20


@dataclass
class RirInfo:
    """One impulse response and everything measurable about it."""

    index: int
    sample_rate: int
    room_dim: tuple[float, float, float]
    rt60_target_s: float
    absorption: float
    max_order: int
    source_pos: tuple[float, float, float]
    mic_pos: tuple[float, float, float]
    source_mic_distance_m: float
    n_taps: int
    peak_index: int
    significant_taps: int
    significant_ms: float
    t60_estimate_s: float
    direct_to_reverberant_db: float
    truncate_db: float
    simulated: bool = True

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def rir_properties(
    h: np.ndarray, sample_rate: int, truncate_db: float = -40.0
) -> tuple[int, int, float, float]:
    """Return ``(peak_index, significant_taps, t60_estimate_s, drr_db)``.

    ``significant_taps`` is the tap index after which the envelope stays below
    ``truncate_db`` relative to the peak: this is the length the adaptive filter has
    to cover. ``t60_estimate_s`` comes from the slope of the Schroeder integral
    between -5 dB and -25 dB, extrapolated to -60 dB.
    """
    h = np.asarray(h, dtype=np.float64)
    if h.size == 0:
        return 0, 0, float("nan"), float("nan")
    energy = h**2
    peak_index = int(np.argmax(np.abs(h)))
    peak = float(np.max(np.abs(h))) + _EPS

    env_db = 20.0 * np.log10(np.abs(h) / peak + _EPS)
    above = np.flatnonzero(env_db > truncate_db)
    significant = int(above[-1] + 1) if above.size else h.size

    # Schroeder backward integration for the decay estimate.
    schroeder = np.cumsum(energy[::-1])[::-1]
    schroeder_db = 10.0 * np.log10(schroeder / (schroeder[0] + _EPS) + _EPS)
    t60 = float("nan")
    try:
        i5 = int(np.flatnonzero(schroeder_db <= -5.0)[0])
        i25 = int(np.flatnonzero(schroeder_db <= -25.0)[0])
        if i25 > i5:
            slope = (schroeder_db[i25] - schroeder_db[i5]) / ((i25 - i5) / sample_rate)
            if slope < 0:
                t60 = float(-60.0 / slope)
    except IndexError:
        pass

    # Direct-to-reverberant ratio: 2.5 ms window around the direct path.
    win = max(1, int(0.0025 * sample_rate))
    direct = energy[max(0, peak_index - 8) : peak_index + win].sum()
    reverb = energy[peak_index + win :].sum()
    drr = float(10.0 * np.log10((direct + _EPS) / (reverb + _EPS)))
    return peak_index, significant, t60, drr


class RirBank:
    """A bank of simulated RIRs, cached on disk so builds are reproducible and fast."""

    def __init__(
        self,
        cfg: RirCfg,
        sample_rate: int = 48000,
        seed: int = 4242,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.cfg = cfg
        self.sample_rate = sample_rate
        self.seed = seed
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.rirs: list[np.ndarray] = []
        self.info: list[RirInfo] = []

    # ------------------------------------------------------------------ build
    def build(self) -> "RirBank":
        if self.cache_dir is not None and self._load_cache():
            return self
        try:
            import pyroomacoustics as pra
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "pyroomacoustics is required to simulate RIRs. Install it, or set "
                "dataset.rir.enabled=false (which makes the reference identical to the "
                "noise in the primary channel and renders ERLE meaningless)."
            ) from exc

        rng = np.random.default_rng(self.seed)
        cfg = self.cfg
        for i in range(cfg.n_rirs):
            dims = np.array(
                [
                    rng.uniform(lo, hi)
                    for lo, hi in zip(cfg.room_dim_min, cfg.room_dim_max)
                ]
            )
            rt60 = float(rng.uniform(cfg.rt60_min_s, cfg.rt60_max_s))
            try:
                absorption, max_order = pra.inverse_sabine(rt60, dims.tolist())
            except ValueError:
                # Requested RT60 is not achievable in this room; nudge it up.
                rt60 = cfg.rt60_max_s
                absorption, max_order = pra.inverse_sabine(rt60, dims.tolist())
            max_order = int(min(max_order, cfg.max_order))

            margin = 0.4
            src = np.array([rng.uniform(margin, d - margin) for d in dims])
            mic = np.array([rng.uniform(margin, d - margin) for d in dims])
            # Keep source and mic apart so the path is non-trivial.
            while np.linalg.norm(src - mic) < 0.8:
                mic = np.array([rng.uniform(margin, d - margin) for d in dims])

            room = pra.ShoeBox(
                dims.tolist(),
                fs=self.sample_rate,
                materials=pra.Material(absorption),
                max_order=max_order,
            )
            room.add_source(src.tolist())
            room.add_microphone(mic.reshape(3, 1))
            room.compute_rir()
            h = np.asarray(room.rir[0][0], dtype=np.float32)
            # Normalise so convolution does not change the overall noise level much.
            h = h / (np.linalg.norm(h) + _EPS)

            peak, significant, t60, drr = rir_properties(h, self.sample_rate, cfg.truncate_db)
            self.rirs.append(h)
            self.info.append(
                RirInfo(
                    index=i,
                    sample_rate=self.sample_rate,
                    room_dim=tuple(round(float(d), 3) for d in dims),  # type: ignore[arg-type]
                    rt60_target_s=round(rt60, 4),
                    absorption=round(float(absorption), 4),
                    max_order=max_order,
                    source_pos=tuple(round(float(v), 3) for v in src),  # type: ignore[arg-type]
                    mic_pos=tuple(round(float(v), 3) for v in mic),  # type: ignore[arg-type]
                    source_mic_distance_m=round(float(np.linalg.norm(src - mic)), 3),
                    n_taps=int(h.size),
                    peak_index=peak,
                    significant_taps=significant,
                    significant_ms=round(1000.0 * significant / self.sample_rate, 2),
                    t60_estimate_s=round(t60, 4) if np.isfinite(t60) else float("nan"),
                    direct_to_reverberant_db=round(drr, 2),
                    truncate_db=cfg.truncate_db,
                )
            )
        log.info(
            "simulated %d RIRs: significant length %.1f-%.1f ms, RT60 estimate %.2f-%.2f s",
            len(self.rirs),
            min(i.significant_ms for i in self.info),
            max(i.significant_ms for i in self.info),
            min(i.t60_estimate_s for i in self.info),
            max(i.t60_estimate_s for i in self.info),
        )
        if self.cache_dir is not None:
            self._save_cache()
        return self

    # ------------------------------------------------------------------ cache
    def _cache_paths(self) -> tuple[Path, Path]:
        assert self.cache_dir is not None
        tag = f"rir_{self.cfg.n_rirs}_{self.seed}_{self.sample_rate}"
        return self.cache_dir / f"{tag}.npz", self.cache_dir / f"{tag}.json"

    def _save_cache(self) -> None:
        npz, meta = self._cache_paths()
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz, **{f"h{i}": h for i, h in enumerate(self.rirs)})
        meta.write_text(
            json.dumps([i.as_dict() for i in self.info], indent=2), encoding="utf-8"
        )
        log.debug("cached RIR bank to %s", npz)

    def _load_cache(self) -> bool:
        npz, meta = self._cache_paths()
        if not (npz.is_file() and meta.is_file()):
            return False
        try:
            data = np.load(npz)
            self.rirs = [data[f"h{i}"] for i in range(len(data.files))]
            self.info = [RirInfo(**d) for d in json.loads(meta.read_text(encoding="utf-8"))]
        except Exception as exc:
            log.warning("could not load cached RIR bank (%s); regenerating", exc)
            return False
        # JSON turns tuples into lists; restore them for type consistency.
        for info in self.info:
            info.room_dim = tuple(info.room_dim)  # type: ignore[assignment]
            info.source_pos = tuple(info.source_pos)  # type: ignore[assignment]
            info.mic_pos = tuple(info.mic_pos)  # type: ignore[assignment]
        log.info("loaded %d cached RIRs from %s", len(self.rirs), npz.name)
        return True

    # ----------------------------------------------------------------- access
    def __len__(self) -> int:
        return len(self.rirs)

    def pick(self, rng: np.random.Generator) -> tuple[np.ndarray, RirInfo]:
        if not self.rirs:
            raise RuntimeError("RIR bank is empty; call build() first")
        i = int(rng.integers(0, len(self.rirs)))
        return self.rirs[i], self.info[i]

    def max_significant_ms(self) -> float:
        return max((i.significant_ms for i in self.info), default=0.0)

    def summary(self) -> dict[str, object]:
        if not self.info:
            return {}
        sig = [i.significant_ms for i in self.info]
        t60 = [i.t60_estimate_s for i in self.info if np.isfinite(i.t60_estimate_s)]
        drr = [i.direct_to_reverberant_db for i in self.info]
        return {
            "n_rirs": len(self.info),
            "sample_rate": self.sample_rate,
            "truncate_db": self.cfg.truncate_db,
            "significant_ms_min": round(min(sig), 2),
            "significant_ms_mean": round(float(np.mean(sig)), 2),
            "significant_ms_max": round(max(sig), 2),
            "t60_estimate_s_min": round(min(t60), 3) if t60 else float("nan"),
            "t60_estimate_s_max": round(max(t60), 3) if t60 else float("nan"),
            "drr_db_min": round(min(drr), 2),
            "drr_db_max": round(max(drr), 2),
            "rooms": [i.room_dim for i in self.info],
        }


def convolve_rir(x: np.ndarray, h: np.ndarray, keep_length: bool = True) -> np.ndarray:
    """Convolve and (by default) trim back to the input length.

    The direct-path delay of the RIR is preserved, which is what makes the adaptive
    filter's job realistic: it must model both the delay and the reverberant tail.
    """
    y = np.convolve(np.asarray(x, dtype=np.float64), np.asarray(h, dtype=np.float64))
    if keep_length:
        y = y[: len(x)]
    return y.astype(np.float32)


def filter_length_check(
    filter_length: int, significant_taps: int, sample_rate: int = 48000
) -> dict[str, object]:
    """Is the configured NLMS filter long enough for this path?"""
    ratio = filter_length / max(1, significant_taps)
    return {
        "filter_length": filter_length,
        "filter_length_ms": round(1000.0 * filter_length / sample_rate, 2),
        "rir_significant_taps": significant_taps,
        "rir_significant_ms": round(1000.0 * significant_taps / sample_rate, 2),
        "coverage_ratio": round(float(ratio), 2),
        "adequate": bool(ratio >= 1.0),
    }
