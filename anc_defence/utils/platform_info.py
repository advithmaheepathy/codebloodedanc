"""Host, toolchain and resource information for the report.

Everything here describes the machine the run actually happened on. No claim is
made about hardware that was not used: the Jetson deployment path is prepared but
unvalidated, and the report says so.
"""

from __future__ import annotations

import platform
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Optional

from .logging import get_logger

log = get_logger(__name__)

_PACKAGES = (
    "numpy",
    "scipy",
    "torch",
    "torchaudio",
    "deepfilternet",
    "DeepFilterLib",
    "soundfile",
    "sounddevice",
    "pystoi",
    "pesq",
    "pyroomacoustics",
    "onnxruntime",
    "matplotlib",
    "reportlab",
)

EDGE_READINESS_NOTE = (
    "All timings in this report were measured on the host described above. "
    "No NVIDIA Jetson hardware was used. The single-thread CPU real-time factor is "
    "reported as portability evidence only: it indicates the workload fits inside a "
    "single CPU core's real-time budget on this machine, which is a necessary but not "
    "sufficient condition for running on an embedded target. No performance claim is "
    "made about the Jetson Orin NX."
)


def package_versions() -> dict[str, Optional[str]]:
    out: dict[str, Optional[str]] = {}
    for name in _PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def git_info() -> dict[str, Any]:
    def _run(args: list[str]) -> Optional[str]:
        try:
            res = subprocess.run(
                args, capture_output=True, text=True, timeout=5, check=False
            )
            return res.stdout.strip() if res.returncode == 0 else None
        except Exception:
            return None

    commit = _run(["git", "rev-parse", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status) if status is not None else None,
    }


def torch_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "available": False,
        "version": None,
        "cuda_available": False,
        "cuda_version": None,
        "device_name": None,
        "num_threads": None,
    }
    try:
        import torch

        info.update(
            available=True,
            version=torch.__version__,
            cuda_available=bool(torch.cuda.is_available()),
            cuda_version=getattr(torch.version, "cuda", None),
            num_threads=torch.get_num_threads(),
        )
        if torch.cuda.is_available():
            info["device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        log.debug("torch info unavailable: %s", exc)
    return info


def cpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "processor": platform.processor() or platform.machine(),
        "machine": platform.machine(),
        "logical_cores": None,
        "physical_cores": None,
        "total_ram_gb": None,
        "max_freq_mhz": None,
    }
    try:
        import psutil

        info["logical_cores"] = psutil.cpu_count(logical=True)
        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["total_ram_gb"] = round(psutil.virtual_memory().total / 2**30, 2)
        freq = psutil.cpu_freq()
        if freq is not None:
            info["max_freq_mhz"] = round(freq.max, 1)
    except Exception as exc:  # pragma: no cover
        log.debug("psutil unavailable: %s", exc)
    return info


def collect_host_info() -> dict[str, Any]:
    """Full snapshot of the machine and toolchain for the session record."""
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "cpu": cpu_info(),
        "torch": torch_info(),
        "packages": package_versions(),
        "git": git_info(),
        "is_jetson": _is_jetson(),
        "edge_readiness_note": EDGE_READINESS_NOTE,
    }


def _is_jetson() -> bool:
    """True only on real Tegra hardware. Kept so the report can never mislabel."""
    try:
        with open("/etc/nv_tegra_release", "r", encoding="utf-8") as fh:
            return bool(fh.readline())
    except OSError:
        return False


@dataclass
class ResourceSample:
    t: float
    cpu_percent: float
    rss_mb: float
    ram_percent: float


@dataclass
class ResourceSampler:
    """Background sampler for the CPU/memory timeline in the report.

    Runs in its own thread at a low rate; never touched from the audio callback.
    """

    interval_s: float = 0.5
    samples: list[ResourceSample] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: Optional[threading.Thread] = None
    _t0: float = 0.0

    def start(self) -> "ResourceSampler":
        try:
            import psutil  # noqa: F401
        except Exception as exc:  # pragma: no cover
            log.warning("resource sampling disabled (psutil unavailable: %s)", exc)
            return self
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, name="resource-sampler", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:  # pragma: no cover - timing dependent
        import psutil

        proc = psutil.Process()
        proc.cpu_percent(None)
        psutil.cpu_percent(None)
        while not self._stop.wait(self.interval_s):
            try:
                mem = psutil.virtual_memory()
                self.samples.append(
                    ResourceSample(
                        t=time.perf_counter() - self._t0,
                        cpu_percent=psutil.cpu_percent(None),
                        rss_mb=proc.memory_info().rss / 2**20,
                        ram_percent=mem.percent,
                    )
                )
            except Exception:
                break

    def stop(self) -> "ResourceSampler":
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return self

    def as_records(self) -> list[dict[str, float]]:
        return [
            {
                "t_s": round(s.t, 3),
                "cpu_percent": s.cpu_percent,
                "rss_mb": round(s.rss_mb, 2),
                "ram_percent": s.ram_percent,
            }
            for s in self.samples
        ]

    def peak_rss_mb(self) -> Optional[float]:
        return max((s.rss_mb for s in self.samples), default=None)
