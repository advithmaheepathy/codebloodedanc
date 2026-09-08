"""Audio file input/output.

Internally everything is float32, mono per channel, at 48 kHz. Files at other
rates are resampled on load with a log message; anything that is not WAV is
decoded through libsndfile (FLAC, OGG) where available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np
import soundfile as sf

from ..utils.logging import get_logger

log = get_logger(__name__)

DEFAULT_SR = 48000


def load_audio(
    path: Path | str,
    target_sr: int = DEFAULT_SR,
    mono: bool = True,
    channel: Optional[int] = None,
) -> np.ndarray:
    """Read an audio file as float32 at ``target_sr``.

    Args:
        path: file to read. WAV/FLAC/OGG via libsndfile.
        target_sr: resample to this rate if needed.
        mono: downmix multi-channel input by averaging.
        channel: instead of downmixing, take this channel index.

    Returns:
        1-D float32 array (or 2-D ``(n, channels)`` when ``mono`` is False).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"audio file not found: {p}")
    try:
        data, sr = sf.read(str(p), dtype="float32", always_2d=True)
    except Exception as exc:
        raise RuntimeError(
            f"could not decode {p}: {exc}. WAV/FLAC/OGG are supported through libsndfile; "
            "MP3 support depends on the libsndfile build. Convert to WAV with ffmpeg if needed."
        ) from exc

    if channel is not None:
        if channel >= data.shape[1]:
            raise ValueError(f"{p} has {data.shape[1]} channel(s); channel {channel} requested")
        out = data[:, channel]
    elif mono:
        out = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    else:
        out = data

    if sr != target_sr:
        log.info("resampling %s from %d Hz to %d Hz", p.name, sr, target_sr)
        out = resample(out, sr, target_sr)
    return np.ascontiguousarray(out, dtype=np.float32)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """High-quality resampling (soxr if available, otherwise scipy polyphase)."""
    if sr_in == sr_out:
        return x
    try:
        import soxr

        return soxr.resample(x, sr_in, sr_out, quality="HQ").astype(np.float32, copy=False)
    except ImportError:  # pragma: no cover - soxr is a hard dependency
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(sr_in, sr_out)
        return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32, copy=False)


def save_wav(path: Path | str, x: np.ndarray, sr: int = DEFAULT_SR, float32: bool = True) -> Path:
    """Write a mono or multi-channel float32 WAV, creating parent directories."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(x, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    subtype = "FLOAT" if float32 else "PCM_16"
    sf.write(str(p), data, sr, subtype=subtype)
    return p


def save_aligned_multichannel(
    path: Path | str, streams: Mapping[str, np.ndarray], sr: int = DEFAULT_SR
) -> tuple[Path, list[str]]:
    """Write every stage of the pipeline into one sample-aligned multi-channel WAV.

    Returns the path and the channel order, which the report records because WAV
    has nowhere to store channel names.
    """
    names = list(streams.keys())
    n = max(len(streams[k]) for k in names)
    stacked = np.zeros((n, len(names)), dtype=np.float32)
    for i, name in enumerate(names):
        s = np.asarray(streams[name], dtype=np.float32)
        stacked[: len(s), i] = s
    save_wav(path, stacked, sr)
    return Path(path), names


# ------------------------------------------------------------------ small utils


def match_length(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Truncate all inputs to the shortest common length."""
    n = min(len(a) for a in arrays)
    return tuple(np.ascontiguousarray(a[:n]) for a in arrays)


def pad_to(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    out = np.zeros(n, dtype=x.dtype)
    out[: len(x)] = x
    return out


def tile_to(x: np.ndarray, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Loop a signal until it is ``n`` samples long, optionally from a random offset."""
    if len(x) == 0:
        return np.zeros(n, dtype=np.float32)
    start = int(rng.integers(0, len(x))) if rng is not None and len(x) > 1 else 0
    reps = int(np.ceil((n + start) / len(x)))
    return np.tile(x, reps)[start : start + n].astype(np.float32, copy=False)


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def dbfs(x: np.ndarray, floor_db: float = -120.0) -> float:
    r = rms(x)
    return max(floor_db, 20.0 * np.log10(r)) if r > 0 else floor_db


def peak_dbfs(x: np.ndarray, floor_db: float = -120.0) -> float:
    p = float(np.max(np.abs(x))) if x.size else 0.0
    return max(floor_db, 20.0 * np.log10(p)) if p > 0 else floor_db


def crest_factor_db(x: np.ndarray) -> float:
    """Peak-to-RMS ratio in dB. High values indicate transient/impulsive content."""
    r = rms(x)
    if r <= 0:
        return 0.0
    return float(20.0 * np.log10(np.max(np.abs(x)) / r))


def scale_to_dbfs(x: np.ndarray, target_dbfs: float) -> np.ndarray:
    r = rms(x)
    if r <= 0:
        return x
    gain = 10.0 ** (target_dbfs / 20.0) / r
    return (x * gain).astype(np.float32, copy=False)


def safe_peak_normalise(x: np.ndarray, ceiling: float = 0.99) -> tuple[np.ndarray, float]:
    """Scale down only if the peak exceeds the ceiling. Returns (audio, gain)."""
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= ceiling or peak == 0.0:
        return x, 1.0
    gain = ceiling / peak
    return (x * gain).astype(np.float32, copy=False), gain


def iter_audio_files(directory: Path | str, extensions: Iterable[str] = (".wav", ".flac", ".ogg")) -> list[Path]:
    d = Path(directory)
    if not d.is_dir():
        return []
    exts = {e.lower() for e in extensions}
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in exts and p.is_file())
