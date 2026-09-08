"""Non-intrusive metrics: available without a clean reference, including live runs.

DNSMOS P.835 predicts the three ITU-T P.835 subjective scores (SIG for speech
quality, BAK for background intrusiveness, OVRL for overall) from the signal alone.
It needs the official ONNX models from the Microsoft DNS-Challenge repository, which
are not bundled here. If they are absent, or onnxruntime is not installed, the
metrics are reported as unavailable rather than silently replaced with something
else.

Also here: an estimated SNR that needs no reference, computed from the ratio of
speech-active to speech-inactive frame energy, and noise-only level measurement for
"how much quieter did the background get".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..audio.io import resample
from ..utils.logging import get_logger

log = get_logger(__name__)

_EPS = 1e-12
DNSMOS_SR = 16000

# Official model files, from https://github.com/microsoft/DNS-Challenge (MIT licence).
DNSMOS_FILES = {
    "sig_bak_ovr": "DNSMOS/sig_bak_ovrl.onnx",
    "p808": "DNSMOS/model_v8.onnx",
}
DNSMOS_BASE_URL = "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/"


@dataclass
class NonIntrusiveMetrics:
    dnsmos_sig: float = float("nan")
    dnsmos_bak: float = float("nan")
    dnsmos_ovrl: float = float("nan")
    dnsmos_p808: float = float("nan")
    estimated_snr_db: float = float("nan")
    available: bool = False
    reason: str = ""
    extra: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "dnsmos_sig": self.dnsmos_sig,
            "dnsmos_bak": self.dnsmos_bak,
            "dnsmos_ovrl": self.dnsmos_ovrl,
            "dnsmos_p808": self.dnsmos_p808,
            "estimated_snr_db": self.estimated_snr_db,
            "dnsmos_available": self.available,
        }
        if self.reason:
            out["dnsmos_reason"] = self.reason
        out.update(self.extra)
        return out


def estimate_snr_db(
    x: np.ndarray, sample_rate: int = 48000, frame_ms: float = 20.0, percentile_noise: float = 10.0
) -> float:
    """Reference-free SNR estimate from the frame-energy distribution.

    The noise floor is taken as a low percentile of frame energies and the speech
    level as a high percentile. Crude, but it needs no clean signal, so it is the only
    SNR figure available during a live run.
    """
    x = np.asarray(x, dtype=np.float64)
    frame = max(16, int(round(sample_rate * frame_ms / 1000.0)))
    n = len(x) // frame
    if n < 4:
        return float("nan")
    power = np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1)
    noise = float(np.percentile(power, percentile_noise)) + _EPS
    speech = float(np.percentile(power, 95.0)) + _EPS
    if speech <= noise:
        return float("nan")
    return float(10.0 * np.log10((speech - noise) / noise))


class DnsmosEstimator:
    """DNSMOS P.835 wrapper. Reports unavailability instead of guessing."""

    def __init__(self, model_dir: Optional[Path] = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else Path("cache/dnsmos")
        self._session = None
        self._p808_session = None
        self.available = False
        self.reason = ""
        self._checked = False

    def _load(self) -> bool:
        if self._checked:
            return self.available
        self._checked = True
        try:
            import onnxruntime as ort
        except ImportError:
            self.reason = (
                "onnxruntime is not installed. Install the optional extra: "
                "pip install .[dnsmos]"
            )
            log.info("DNSMOS unavailable: %s", self.reason)
            return False
        primary = self.model_dir / "sig_bak_ovrl.onnx"
        if not primary.is_file():
            self.reason = (
                f"DNSMOS ONNX model not found at {primary}. Download sig_bak_ovrl.onnx from "
                "https://github.com/microsoft/DNS-Challenge (DNSMOS directory, MIT licence) "
                "and place it there, or run `anc fetch-data --dnsmos`."
            )
            log.info("DNSMOS unavailable: %s", self.reason)
            return False
        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            self._session = ort.InferenceSession(
                str(primary), sess_options=opts, providers=["CPUExecutionProvider"]
            )
            self.available = True
        except Exception as exc:
            self.reason = f"failed to load DNSMOS model: {exc}"
            log.warning(self.reason)
            return False
        return self.available

    def score(self, x: np.ndarray, sample_rate: int = 48000) -> NonIntrusiveMetrics:
        m = NonIntrusiveMetrics(estimated_snr_db=estimate_snr_db(x, sample_rate))
        if not self._load():
            m.reason = self.reason
            return m
        audio = np.asarray(x, dtype=np.float32)
        if sample_rate != DNSMOS_SR:
            audio = resample(audio, sample_rate, DNSMOS_SR)
        # The model scores 9-second segments; average over as many as fit.
        seg = 9 * DNSMOS_SR
        if len(audio) < seg:
            audio = np.pad(audio, (0, seg - len(audio)))
        scores: list[np.ndarray] = []
        assert self._session is not None
        input_name = self._session.get_inputs()[0].name
        for start in range(0, max(1, len(audio) - seg + 1), seg):
            chunk = audio[start : start + seg]
            if len(chunk) < seg:
                break
            try:
                out = self._session.run(None, {input_name: chunk[None, :].astype(np.float32)})[0]
                scores.append(np.asarray(out).reshape(-1))
            except Exception as exc:
                m.reason = f"DNSMOS inference failed: {exc}"
                log.warning(m.reason)
                return m
        if not scores:
            m.reason = "no full 9 s segment available for DNSMOS"
            return m
        mean = np.mean(np.stack(scores), axis=0)
        m.available = True
        if mean.size >= 3:
            m.dnsmos_sig, m.dnsmos_bak, m.dnsmos_ovrl = (
                float(mean[0]),
                float(mean[1]),
                float(mean[2]),
            )
        return m


def fetch_dnsmos_models(dest: Path | str = "cache/dnsmos", timeout: float = 60.0) -> list[Path]:
    """Download the official DNSMOS ONNX models (MIT licence) into the cache."""
    import urllib.request

    out_dir = Path(dest)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, rel in DNSMOS_FILES.items():
        target = out_dir / Path(rel).name
        if target.is_file():
            written.append(target)
            continue
        url = DNSMOS_BASE_URL + rel
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "anc-defence/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                target.write_bytes(resp.read())
            written.append(target)
            log.info("downloaded %s", target.name)
        except Exception as exc:
            log.warning("could not download DNSMOS model %s (%s): %s", name, url, exc)
    return written
