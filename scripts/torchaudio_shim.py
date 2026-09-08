#!/usr/bin/env python3
"""Install a minimal ``torchaudio`` compatibility shim. Last-resort, opt-in.

Why this exists
---------------
``deepfilternet`` imports torchaudio at module load time:

    df/io.py:  import torchaudio as ta
               from torchaudio.backend.common import AudioMetaData

but it only *uses* torchaudio inside ``df.io.load_audio`` / ``save_audio`` and in its
training and evaluation scripts. This project never calls those: all file I/O goes
through soundfile, and resampling goes through soxr. The inference path
(``init_df`` + ``enhance``) touches no torchaudio function at all.

That matters on JetPack 5.1.3, where NVIDIA publishes a CUDA build of torch
(``torch-2.1.0a0+41361538.nv23.06-cp38``) but **no matching torchaudio wheel**. The
options are a 20-40 minute source build, a PyPI wheel that is built against a different
libtorch and usually fails with undefined symbols, or this shim.

Safety
------
Every stubbed callable **raises** with an explanation. Nothing returns fake audio or a
silently wrong result. If any code path in this project ever did need real torchaudio,
it would fail loudly and immediately, not produce a plausible-looking wrong answer.

The shim refuses to install if a working torchaudio is already importable, and
``anc selftest`` reports when the shim is active so it can never be mistaken for the
real thing.

Usage
-----
    python scripts/torchaudio_shim.py --install     # only if torchaudio is missing
    python scripts/torchaudio_shim.py --check
    python scripts/torchaudio_shim.py --remove
"""

from __future__ import annotations

import argparse
import shutil
import site
import sys
from pathlib import Path

MARKER = "anc_defence_torchaudio_shim"

SHIM_INIT = '''"""Minimal torchaudio stand-in installed by anc_defence.

THIS IS NOT TORCHAUDIO. It exists only so that `import torchaudio` succeeds, because
deepfilternet imports it at module load but does not use it on the inference path.

Every function here raises. If you are reading this because something raised, the fix is
to install a real torchaudio built against your installed torch:

    # matching the JetPack 5.1.3 torch 2.1.0 build
    pip install --no-build-isolation "git+https://github.com/pytorch/audio.git@v2.1.0"
"""

__version__ = "0.0.0+anc_defence_shim"
IS_ANC_DEFENCE_SHIM = True

from dataclasses import dataclass
from typing import Any, Optional


def _unavailable(name: str) -> Any:
    def _raise(*_args: Any, **_kwargs: Any) -> Any:
        raise NotImplementedError(
            "torchaudio." + name + " was called, but only a compatibility shim is "
            "installed. This project does not use torchaudio on its inference path, so "
            "reaching here means a code path changed. Install a real torchaudio built "
            "against your torch:\\n"
            '  pip install --no-build-isolation "git+https://github.com/pytorch/audio.git@v2.1.0"'
        )

    return _raise


@dataclass
class AudioMetaData:
    """Mirrors torchaudio.backend.common.AudioMetaData."""

    sample_rate: int = 0
    num_frames: int = 0
    num_channels: int = 0
    bits_per_sample: int = 0
    encoding: str = "UNKNOWN"


load = _unavailable("load")
save = _unavailable("save")
info = _unavailable("info")
list_audio_backends = lambda: []          # noqa: E731
get_audio_backend = lambda: None          # noqa: E731
set_audio_backend = _unavailable("set_audio_backend")

from . import backend, compliance, functional, transforms  # noqa: E402,F401
'''

SHIM_BACKEND_INIT = "from . import common  # noqa: F401\n"

SHIM_BACKEND_COMMON = '''"""torchaudio.backend.common shim."""

from .. import AudioMetaData  # noqa: F401
'''

SHIM_FUNCTIONAL = '''"""torchaudio.functional shim. Every entry raises if called."""

from .. import _unavailable

resample = _unavailable("functional.resample")
highpass_biquad = _unavailable("functional.highpass_biquad")
lowpass_biquad = _unavailable("functional.lowpass_biquad")
'''

SHIM_TRANSFORMS = '''"""torchaudio.transforms shim. Every entry raises if instantiated."""

from .. import _unavailable


class Resample:
    def __init__(self, *args, **kwargs):
        _unavailable("transforms.Resample")()


class Spectrogram:
    def __init__(self, *args, **kwargs):
        _unavailable("transforms.Spectrogram")()
'''

SHIM_COMPLIANCE_INIT = "from . import kaldi  # noqa: F401\n"

SHIM_KALDI = '''"""torchaudio.compliance.kaldi shim."""

from ... import _unavailable

resample_waveform = _unavailable("compliance.kaldi.resample_waveform")
'''

FILES = {
    "__init__.py": SHIM_INIT,
    "backend/__init__.py": SHIM_BACKEND_INIT,
    "backend/common.py": SHIM_BACKEND_COMMON,
    "functional/__init__.py": SHIM_FUNCTIONAL,
    "transforms/__init__.py": SHIM_TRANSFORMS,
    "compliance/__init__.py": SHIM_COMPLIANCE_INIT,
    "compliance/kaldi.py": SHIM_KALDI,
}


def site_packages() -> Path:
    for candidate in site.getsitepackages():
        p = Path(candidate)
        if p.name == "site-packages" and p.is_dir():
            return p
    return Path(site.getsitepackages()[0])


def torchaudio_status() -> tuple[bool, str]:
    """(importable, description)."""
    try:
        import torchaudio
    except Exception as exc:
        return False, f"not importable: {type(exc).__name__}: {exc}"
    if getattr(torchaudio, "IS_ANC_DEFENCE_SHIM", False):
        return True, f"SHIM active (version {torchaudio.__version__})"
    return True, f"real torchaudio {getattr(torchaudio, '__version__', '?')}"


def install(force: bool = False) -> int:
    ok, desc = torchaudio_status()
    if ok and "SHIM" not in desc and not force:
        print(f"torchaudio is already available: {desc}")
        print("Refusing to shadow a real installation. Use --force to override.")
        return 0

    target = site_packages() / "torchaudio"
    if target.exists():
        if not (target / MARKER).exists() and not force:
            print(f"{target} exists and is not our shim. Refusing to overwrite.")
            return 1
        shutil.rmtree(target)

    for rel, content in FILES.items():
        path = target / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (target / MARKER).write_text(
        "Installed by anc_defence scripts/torchaudio_shim.py. Safe to delete.\n",
        encoding="utf-8",
    )
    print(f"installed the torchaudio shim at {target}")
    print("Verifying that deepfilternet imports...")
    try:
        import df.enhance  # noqa: F401

        print("  df.enhance imports successfully")
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        return 1
    print(
        "\nThis is a shim, not torchaudio. It only satisfies the import.\n"
        "`anc selftest` will report that the shim is active."
    )
    return 0


def remove() -> int:
    target = site_packages() / "torchaudio"
    if not target.exists():
        print("no torchaudio directory found")
        return 0
    if not (target / MARKER).exists():
        print(f"{target} is not our shim. Not removing it.")
        return 1
    shutil.rmtree(target)
    print(f"removed {target}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--install", action="store_true")
    group.add_argument("--check", action="store_true")
    group.add_argument("--remove", action="store_true")
    parser.add_argument("--force", action="store_true", help="overwrite an existing torchaudio")
    args = parser.parse_args()

    if args.check:
        ok, desc = torchaudio_status()
        print(f"torchaudio: {desc}")
        return 0 if ok else 1
    if args.install:
        return install(force=args.force)
    return remove()


if __name__ == "__main__":
    sys.exit(main())
