"""Audio device enumeration and selection.

Devices can be chosen by index or by a case-insensitive substring of their name, so
``--input-device "Microphone Array"`` works without hunting for an index that changes
when hardware is plugged in.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..utils.logging import get_logger

log = get_logger(__name__)


class DeviceError(RuntimeError):
    """Raised with actionable guidance when a device cannot be used."""


def _sd() -> Any:
    try:
        import sounddevice as sd
    except OSError as exc:  # pragma: no cover - platform dependent
        raise DeviceError(
            f"PortAudio could not be loaded ({exc}). Live modes need a working audio backend; "
            "the offline file mode does not and will still run."
        ) from exc
    except ImportError as exc:
        raise DeviceError("sounddevice is not installed; live modes are unavailable") from exc
    return sd


def list_devices() -> list[dict[str, Any]]:
    """Every audio device with the fields needed to choose one."""
    sd = _sd()
    hostapis = sd.query_hostapis()
    out: list[dict[str, Any]] = []
    for i, dev in enumerate(sd.query_devices()):
        api = hostapis[dev["hostapi"]]["name"] if dev.get("hostapi") is not None else "?"
        out.append(
            {
                "index": i,
                "name": dev["name"],
                "hostapi": api,
                "max_input_channels": dev["max_input_channels"],
                "max_output_channels": dev["max_output_channels"],
                "default_samplerate": dev["default_samplerate"],
                "is_default_input": i == sd.default.device[0],
                "is_default_output": i == sd.default.device[1],
            }
        )
    return out


def format_device_table(devices: Optional[list[dict[str, Any]]] = None) -> str:
    """Human-readable device listing for ``anc list-devices``."""
    devices = devices if devices is not None else list_devices()
    lines = [
        f"{'idx':>4}  {'in':>3} {'out':>3}  {'rate':>7}  {'host api':<22} name",
        "-" * 100,
    ]
    for d in devices:
        marks = ""
        if d["is_default_input"]:
            marks += " [default in]"
        if d["is_default_output"]:
            marks += " [default out]"
        lines.append(
            f"{d['index']:>4}  {d['max_input_channels']:>3} {d['max_output_channels']:>3}  "
            f"{d['default_samplerate']:>7.0f}  {d['hostapi'][:22]:<22} {d['name']}{marks}"
        )
    lines.append("")
    lines.append(
        "Select with --input-device / --output-device using either the index or part of the name."
    )
    lines.append(
        "Only one microphone is used in the live modes of this build; a second reference mic is not "
        "required and not supported."
    )
    return "\n".join(lines)


def resolve_device(spec: Optional[str], kind: str = "input") -> Optional[int]:
    """Resolve an index or name substring to a device index.

    Returns None for the system default. Raises :class:`DeviceError` with the device
    list included when the spec is ambiguous or matches nothing.
    """
    if spec is None or str(spec).strip() == "":
        return None
    spec = str(spec).strip()
    devices = list_devices()
    channel_key = "max_input_channels" if kind == "input" else "max_output_channels"

    if spec.lstrip("-").isdigit():
        index = int(spec)
        match = next((d for d in devices if d["index"] == index), None)
        if match is None:
            raise DeviceError(f"no audio device with index {index}. Run `anc list-devices`.")
        if match[channel_key] < 1:
            raise DeviceError(
                f"device {index} ('{match['name']}') has no {kind} channels. Run `anc list-devices`."
            )
        return index

    matches = [
        d for d in devices if spec.lower() in d["name"].lower() and d[channel_key] >= 1
    ]
    if not matches:
        raise DeviceError(
            f"no {kind} device matching '{spec}'.\n\n{format_device_table(devices)}"
        )
    if len(matches) > 1:
        names = ", ".join(f"{d['index']}:{d['name']}" for d in matches[:6])
        log.warning("'%s' matched %d %s devices (%s); using the first", spec, len(matches), kind, names)
    return int(matches[0]["index"])


def check_stereo_input(device: Optional[int], sample_rate: int) -> None:
    """Verify a device can capture 2 channels at ``sample_rate``.

    Used by the two-microphone NLMS path, where the two transmitters of a stereo mic
    kit arrive as the left and right channels of a single input device.
    """
    check_settings(device, sample_rate, channels=2, kind="input")


def channel_independence(stereo: "Any") -> dict[str, float]:
    """How different the two channels of a stereo capture are.

    Returns the Pearson correlation between L and R and their individual RMS levels in
    dBFS. A correlation near 1.0 means the device is duplicating one source to both
    channels (useless as a primary/reference microphone pair); a low correlation with a
    level difference means two genuinely independent microphones.
    """
    arr = np.asarray(stereo, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return {"correlation": float("nan"), "left_dbfs": float("nan"), "right_dbfs": float("nan")}
    left, right = arr[:, 0], arr[:, 1]
    lc, rc = left - left.mean(), right - right.mean()
    denom = float(np.linalg.norm(lc) * np.linalg.norm(rc)) + 1e-20
    corr = float(np.dot(lc, rc) / denom)

    def _dbfs(x: np.ndarray) -> float:
        r = float(np.sqrt(np.mean(x**2)))
        return 20.0 * float(np.log10(r)) if r > 0 else -120.0

    return {"correlation": corr, "left_dbfs": _dbfs(left), "right_dbfs": _dbfs(right)}


def device_name(index: Optional[int], kind: str = "input") -> str:
    if index is None:
        try:
            sd = _sd()
            index = sd.default.device[0 if kind == "input" else 1]
        except Exception:
            return "system default"
    try:
        devices = list_devices()
        match = next((d for d in devices if d["index"] == index), None)
        return f"{index}: {match['name']} ({match['hostapi']})" if match else str(index)
    except Exception:
        return str(index)


def check_settings(
    device: Optional[int], sample_rate: int, channels: int = 1, kind: str = "input"
) -> None:
    """Verify a device can run at the requested rate, with a clear error if not."""
    sd = _sd()
    try:
        if kind == "input":
            sd.check_input_settings(device=device, samplerate=sample_rate, channels=channels, dtype="float32")
        else:
            sd.check_output_settings(device=device, samplerate=sample_rate, channels=channels, dtype="float32")
    except Exception as exc:
        raise DeviceError(
            f"{kind} device {device_name(device, kind)} cannot run at {sample_rate} Hz with "
            f"{channels} channel(s): {exc}\n"
            f"The pipeline is fixed at {sample_rate} Hz because the neural model is. On Windows, "
            "set the device's sample rate in Sound Control Panel > Device Properties > Advanced, "
            "or pick a different device with `anc list-devices`."
        ) from exc
