"""Intrusive metrics: those that need the clean reference signal.

Definitions used here, stated explicitly because "SNR improvement" is ambiguous in
the literature and the reported headline number depends on the choice:

``si_sdr``
    Scale-invariant signal-to-distortion ratio. The test signal is projected onto
    the clean signal, ``alpha = <y,s>/<s,s>``; the aligned part ``alpha*s`` is
    signal and everything else is distortion. Because it is scale invariant it is
    unaffected by any overall gain the enhancer applies.

``output SNR``
    Reported as SI-SDR of the processed signal. Projecting is the only defensible
    way to separate "surviving speech" from "everything else" once a non-linear
    enhancer has been applied.

``snr_improvement``
    ``si_sdr(enhanced) - si_sdr(noisy)``, i.e. SI-SDRi. This is the primary figure.

``snr_direct``
    ``10*log10(||s||^2 / ||y - s||^2)`` without projection. Sensitive to gain, so it
    is reported as a secondary column only.

``segmental_snr``
    Frame-wise SNR averaged over frames, clamped to [-10, +35] dB per the usual
    convention so that silent frames cannot dominate the average.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from ..audio.io import match_length, resample
from ..config import MetricsCfg
from ..utils.logging import get_logger

log = get_logger(__name__)

_EPS = 1e-12
_PESQ_SR = 16000

_pesq_warned = False
_stoi_warned = False


@dataclass
class IntrusiveMetrics:
    """All intrusive metrics for one signal against one clean reference."""

    pesq: float = float("nan")
    stoi: float = float("nan")
    estoi: float = float("nan")
    si_sdr: float = float("nan")
    snr_direct: float = float("nan")
    segmental_snr: float = float("nan")
    lsd: float = float("nan")
    speech_attenuation_db: float = float("nan")
    speech_distortion_db: float = float("nan")
    extra: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        out = {k: v for k, v in asdict(self).items() if k != "extra"}
        out.update(self.extra)
        return {k: (float(v) if v is not None else float("nan")) for k, v in out.items()}


def si_sdr(reference: np.ndarray, test: np.ndarray) -> float:
    """Scale-invariant SDR in dB. Also serves as the projected output SNR."""
    ref, tst = match_length(
        np.asarray(reference, dtype=np.float64), np.asarray(test, dtype=np.float64)
    )
    ref = ref - ref.mean()
    tst = tst - tst.mean()
    denom = float(np.dot(ref, ref))
    if denom <= _EPS:
        return float("nan")
    alpha = float(np.dot(tst, ref)) / denom
    projection = alpha * ref
    noise = tst - projection
    num = float(np.dot(projection, projection))
    den = float(np.dot(noise, noise))
    if num <= _EPS or den <= _EPS:
        return float("nan")
    return float(10.0 * np.log10(num / den))


def snr_db(reference: np.ndarray, test: np.ndarray) -> float:
    """Unprojected SNR: signal power over the power of the deviation from it."""
    ref, tst = match_length(
        np.asarray(reference, dtype=np.float64), np.asarray(test, dtype=np.float64)
    )
    num = float(np.dot(ref, ref))
    err = tst - ref
    den = float(np.dot(err, err))
    if num <= _EPS or den <= _EPS:
        return float("nan")
    return float(10.0 * np.log10(num / den))


def mixture_snr_db(clean: np.ndarray, noise: np.ndarray) -> float:
    """True input SNR of a synthesised mixture, from its known components."""
    c, n = match_length(np.asarray(clean, dtype=np.float64), np.asarray(noise, dtype=np.float64))
    cp, npow = float(np.dot(c, c)), float(np.dot(n, n))
    if cp <= _EPS or npow <= _EPS:
        return float("nan")
    return float(10.0 * np.log10(cp / npow))


def segmental_snr(
    reference: np.ndarray,
    test: np.ndarray,
    sample_rate: int = 48000,
    frame_ms: float = 20.0,
    floor_db: float = -10.0,
    ceil_db: float = 35.0,
    active_only: bool = True,
) -> float:
    """Frame-wise SNR averaged over frames, clamped to a sane range."""
    ref, tst = match_length(
        np.asarray(reference, dtype=np.float64), np.asarray(test, dtype=np.float64)
    )
    frame = max(16, int(round(sample_rate * frame_ms / 1000.0)))
    n_frames = len(ref) // frame
    if n_frames == 0:
        return float("nan")
    ref_f = ref[: n_frames * frame].reshape(n_frames, frame)
    tst_f = tst[: n_frames * frame].reshape(n_frames, frame)
    sig = np.sum(ref_f**2, axis=1)
    err = np.sum((tst_f - ref_f) ** 2, axis=1)
    if active_only:
        # Ignore frames more than 40 dB below the loudest frame: they are silence.
        keep = sig > (sig.max() * 1e-4)
        if keep.sum() < 1:
            return float("nan")
        sig, err = sig[keep], err[keep]
    snr = 10.0 * np.log10((sig + _EPS) / (err + _EPS))
    return float(np.mean(np.clip(snr, floor_db, ceil_db)))


def log_spectral_distance(
    reference: np.ndarray, test: np.ndarray, n_fft: int = 1024, hop: Optional[int] = None
) -> float:
    """RMS difference of log power spectra in dB. Lower is closer to the clean signal."""
    ref, tst = match_length(
        np.asarray(reference, dtype=np.float64), np.asarray(test, dtype=np.float64)
    )
    hop = hop or n_fft // 2
    if len(ref) < n_fft:
        return float("nan")
    window = np.hanning(n_fft)
    n_frames = 1 + (len(ref) - n_fft) // hop
    total = 0.0
    for i in range(n_frames):
        s = i * hop
        a = np.abs(np.fft.rfft(ref[s : s + n_fft] * window)) ** 2 + _EPS
        b = np.abs(np.fft.rfft(tst[s : s + n_fft] * window)) ** 2 + _EPS
        diff = 10.0 * np.log10(a) - 10.0 * np.log10(b)
        total += float(np.sqrt(np.mean(diff**2)))
    return total / max(1, n_frames)


def speech_component_metrics(clean: np.ndarray, test: np.ndarray) -> tuple[float, float]:
    """How much of the speech survived, and how distorted what survived is.

    Returns ``(attenuation_db, distortion_db)``:

    * attenuation: positive means the enhancer made the speech quieter overall.
    * distortion: energy of the non-scalable part of the error relative to the
      speech energy, in dB. More negative is better.
    """
    ref, tst = match_length(
        np.asarray(clean, dtype=np.float64), np.asarray(test, dtype=np.float64)
    )
    denom = float(np.dot(ref, ref))
    if denom <= _EPS:
        return float("nan"), float("nan")
    alpha = float(np.dot(tst, ref)) / denom
    if alpha <= 0:
        return float("inf"), float("inf")
    attenuation = float(-20.0 * np.log10(alpha))
    residual = tst - alpha * ref
    distortion = float(10.0 * np.log10((float(np.dot(residual, residual)) + _EPS) / denom))
    return attenuation, distortion


def pesq_wb(clean: np.ndarray, test: np.ndarray, sample_rate: int = 48000, mode: str = "wb") -> float:
    """Wideband PESQ (ITU-T P.862.2), computed at 16 kHz as the standard requires."""
    global _pesq_warned
    try:
        from pesq import PesqError, pesq as _pesq
    except ImportError:
        if not _pesq_warned:
            log.warning("pesq not installed; PESQ will be reported as NaN")
            _pesq_warned = True
        return float("nan")

    ref, tst = match_length(clean, test)
    if sample_rate != _PESQ_SR:
        ref = resample(np.asarray(ref, dtype=np.float32), sample_rate, _PESQ_SR)
        tst = resample(np.asarray(tst, dtype=np.float32), sample_rate, _PESQ_SR)
    if len(ref) < _PESQ_SR // 4:  # PESQ needs a fraction of a second to work with
        return float("nan")
    try:
        return float(
            _pesq(_PESQ_SR, np.asarray(ref, np.float32), np.asarray(tst, np.float32), mode, on_error=PesqError.RETURN_VALUES)
        )
    except Exception as exc:
        log.debug("PESQ failed: %s", exc)
        return float("nan")


def stoi_scores(
    clean: np.ndarray, test: np.ndarray, sample_rate: int = 48000, extended: bool = False
) -> float:
    """STOI or ESTOI via pystoi (which resamples internally to 10 kHz)."""
    global _stoi_warned
    try:
        from pystoi import stoi as _stoi
    except ImportError:
        if not _stoi_warned:
            log.warning("pystoi not installed; STOI/ESTOI will be reported as NaN")
            _stoi_warned = True
        return float("nan")
    ref, tst = match_length(
        np.asarray(clean, dtype=np.float64), np.asarray(test, dtype=np.float64)
    )
    if len(ref) < sample_rate // 2:
        return float("nan")
    try:
        return float(_stoi(ref, tst, sample_rate, extended=extended))
    except Exception as exc:
        log.debug("STOI failed: %s", exc)
        return float("nan")


def compute_intrusive(
    clean: np.ndarray,
    test: np.ndarray,
    sample_rate: int = 48000,
    cfg: Optional[MetricsCfg] = None,
) -> IntrusiveMetrics:
    """Compute the configured set of intrusive metrics for one signal."""
    cfg = cfg or MetricsCfg()
    clean, test = match_length(
        np.asarray(clean, dtype=np.float32), np.asarray(test, dtype=np.float32)
    )
    m = IntrusiveMetrics()

    # Level-sensitive metrics are computed on a gain-aligned copy. The pipeline ends
    # with a volume normaliser, so without this the gain change alone would move
    # segmental SNR and LSD and look like a quality change. PESQ has its own internal
    # level alignment, and STOI and SI-SDR are scale invariant, so they use the signal
    # as it actually is.
    aligned = test
    if cfg.level_align:
        denom = float(np.dot(clean.astype(np.float64), clean.astype(np.float64)))
        if denom > _EPS:
            alpha = float(np.dot(test.astype(np.float64), clean.astype(np.float64))) / denom
            if alpha > _EPS:
                aligned = (test / alpha).astype(np.float32)
        m.extra["level_align_gain_db"] = (
            float(-20.0 * np.log10(alpha)) if denom > _EPS and alpha > _EPS else float("nan")
        )

    if cfg.pesq:
        m.pesq = pesq_wb(clean, test, sample_rate, cfg.pesq_mode)
    if cfg.stoi:
        m.stoi = stoi_scores(clean, test, sample_rate, extended=False)
    if cfg.estoi:
        m.estoi = stoi_scores(clean, test, sample_rate, extended=True)
    if cfg.sisdr:
        m.si_sdr = si_sdr(clean, test)
    if cfg.snr:
        m.snr_direct = snr_db(clean, aligned)
    if cfg.segsnr:
        m.segmental_snr = segmental_snr(clean, aligned, sample_rate, cfg.segsnr_frame_ms)
    if cfg.lsd:
        m.lsd = log_spectral_distance(clean, aligned)
    m.speech_attenuation_db, m.speech_distortion_db = speech_component_metrics(clean, test)
    return m


def improvement(before: IntrusiveMetrics, after: IntrusiveMetrics) -> dict[str, float]:
    """Deltas between two metric sets. For LSD and distortion, lower is better."""
    return {
        "d_pesq": after.pesq - before.pesq,
        "d_stoi": after.stoi - before.stoi,
        "d_estoi": after.estoi - before.estoi,
        "snr_improvement_db": after.si_sdr - before.si_sdr,
        "d_snr_direct_db": after.snr_direct - before.snr_direct,
        "d_segsnr_db": after.segmental_snr - before.segmental_snr,
        "d_lsd": after.lsd - before.lsd,
    }
