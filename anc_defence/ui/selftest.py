"""``anc selftest``: environment check plus a tiny end-to-end pipeline run.

Prints a go/no-go summary. Each check is independent, so one failure does not hide the
rest, and every failure carries the action needed to fix it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ..config import Config
from ..utils.logging import get_logger

log = get_logger(__name__)

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
SKIP = "SKIP"


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""
    action: str = ""


def _run(name: str, fn: Callable[[], tuple[str, str, str]]) -> Check:
    try:
        status, detail, action = fn()
    except Exception as exc:  # noqa: BLE001 - a selftest must never crash
        return Check(name, FAIL, f"{type(exc).__name__}: {exc}", "see the traceback in the log")
    return Check(name, status, detail, action)


def run_selftest(cfg: Config) -> int:
    checks: list[Check] = []

    # ------------------------------------------------------------------ imports
    def check_imports() -> tuple[str, str, str]:
        missing: list[str] = []
        versions: list[str] = []
        for module, label in (
            ("numpy", "numpy"),
            ("scipy", "scipy"),
            ("soundfile", "soundfile"),
            ("torch", "torch"),
            ("df", "deepfilternet"),
            ("pystoi", "pystoi"),
            ("pesq", "pesq"),
            ("pyroomacoustics", "pyroomacoustics"),
            ("reportlab", "reportlab"),
            ("matplotlib", "matplotlib"),
        ):
            try:
                mod = __import__(module)
                versions.append(f"{label} {getattr(mod, '__version__', '?')}")
            except Exception:
                missing.append(label)
        if missing:
            return FAIL, f"missing: {', '.join(missing)}", "pip install -e .[dev]"
        return PASS, ", ".join(versions[:6]), ""

    checks.append(_run("python packages", check_imports))

    # -------------------------------------------------------------------- numpy
    def check_numpy() -> tuple[str, str, str]:
        import numpy

        if numpy.__version__.startswith("2."):
            return (
                FAIL,
                f"numpy {numpy.__version__}",
                "deepfilternet 0.5.6 requires numpy<2; pip install 'numpy==1.26.4'",
            )
        return PASS, f"numpy {numpy.__version__} (<2 as deepfilternet requires)", ""

    checks.append(_run("numpy version", check_numpy))

    # -------------------------------------------------------------------- torch
    def check_torch() -> tuple[str, str, str]:
        import torch

        detail = f"torch {torch.__version__}, threads {torch.get_num_threads()}"
        if torch.cuda.is_available():
            return PASS, f"{detail}, CUDA {torch.version.cuda} on {torch.cuda.get_device_name(0)}", ""
        if cfg.neural.device == "cuda":
            return (
                FAIL,
                f"{detail}, CUDA unavailable but neural.device=cuda",
                "install a CUDA build of torch or set neural.device=cpu",
            )
        return PASS, f"{detail}, CPU only (neural.device={cfg.neural.device})", ""

    checks.append(_run("torch", check_torch))

    # -------------------------------------------------------------- model load
    model = None

    def check_model() -> tuple[str, str, str]:
        nonlocal model
        from ..enhance.streaming import DfnModel

        t0 = time.perf_counter()
        model = DfnModel(cfg.neural, cfg.audio.sample_rate).load()
        dt = time.perf_counter() - t0
        info = model.info
        return (
            PASS,
            f"{info.name}, {info.n_parameters / 1e6:.2f}M params, {info.sample_rate} Hz, "
            f"algorithmic latency {info.algorithmic_latency_ms:.0f} ms, loaded in {dt:.1f} s",
            "",
        )

    checks.append(_run("neural model", check_model))

    # ------------------------------------------------------------------- warmup
    def check_inference() -> tuple[str, str, str]:
        if model is None:
            return SKIP, "model not loaded", ""
        sr = cfg.audio.sample_rate
        rng = np.random.default_rng(0)
        audio = (0.05 * rng.standard_normal(2 * sr)).astype(np.float32)
        model.enhance_array(audio)  # warm
        t0 = time.perf_counter()
        out = model.enhance_array(audio)
        dt = time.perf_counter() - t0
        rtf = dt / (len(audio) / sr)
        if not np.all(np.isfinite(out)):
            return FAIL, "output contained non-finite samples", "check the model checkpoint"
        status = PASS if rtf < 1.0 else WARN
        return (
            status,
            f"RTF {rtf:.3f} on {len(audio) / sr:.0f} s ({'faster' if rtf < 1 else 'SLOWER'} than real time)",
            "" if rtf < 1.0 else "reduce load or use fewer metrics; the live path will drop out",
        )

    checks.append(_run("neural inference", check_inference))

    # --------------------------------------------------------------------- NLMS
    def check_nlms() -> tuple[str, str, str]:
        from ..dsp.fdaf import PartitionedFdafNlms
        from ..dsp.nlms import normalised_misalignment_db

        sr = cfg.audio.sample_rate
        rng = np.random.default_rng(3)
        h = rng.standard_normal(96) * np.exp(-np.arange(96) / 24.0)
        h /= np.linalg.norm(h)
        x = (rng.standard_normal(sr) * 0.1).astype(np.float32)
        d = np.convolve(x, h)[:sr].astype(np.float32)
        local = cfg.nlms.model_copy(deep=True)
        local.filter_length = cfg.audio.hop_size
        local.mu = 0.5
        local.double_talk.mode = "off"
        filt = PartitionedFdafNlms(local, cfg.vad, sr, cfg.audio.hop_size, use_safeguards=False)
        result = filt.process(d, x)
        erle = float(np.mean(result.erle_curve[-10:]))
        mis = normalised_misalignment_db(result.weights[:96], h)
        if erle < 15.0:
            return FAIL, f"ERLE only {erle:.1f} dB on a known FIR path", "the adaptive stage is broken"
        return PASS, f"ERLE {erle:.1f} dB, filter misalignment {mis:.1f} dB on a known FIR path", ""

    checks.append(_run("NLMS convergence", check_nlms))

    # ------------------------------------------------------------------ metrics
    def check_metrics() -> tuple[str, str, str]:
        from ..metrics.intrusive import compute_intrusive

        sr = cfg.audio.sample_rate
        t = np.arange(2 * sr) / sr
        clean = (0.2 * np.sin(2 * np.pi * 220 * t) * (1 + 0.4 * np.sin(2 * np.pi * 2.5 * t))).astype(np.float32)
        noisy = (clean + 0.05 * np.random.default_rng(1).standard_normal(len(t))).astype(np.float32)
        m = compute_intrusive(clean, noisy, sr)
        parts = [f"PESQ {m.pesq:.2f}", f"STOI {m.stoi:.3f}", f"SI-SDR {m.si_sdr:.1f} dB"]
        if not np.isfinite(m.pesq):
            return WARN, ", ".join(parts) + " (PESQ unavailable)", "pip install pesq"
        return PASS, ", ".join(parts), ""

    checks.append(_run("metrics", check_metrics))

    # ------------------------------------------------------------------- report
    def check_report() -> tuple[str, str, str]:
        import matplotlib

        matplotlib.use("Agg")
        from reportlab.pdfgen import canvas  # noqa: F401

        return PASS, "matplotlib Agg backend and reportlab both usable offline", ""

    checks.append(_run("report toolchain", check_report))

    # ------------------------------------------------------------------ devices
    def check_devices() -> tuple[str, str, str]:
        from ..audio.devices import DeviceError, list_devices

        try:
            devices = list_devices()
        except DeviceError as exc:
            return WARN, str(exc), "live modes unavailable; offline mode still works"
        inputs = [d for d in devices if d["max_input_channels"] > 0]
        outputs = [d for d in devices if d["max_output_channels"] > 0]
        if not inputs:
            return WARN, "no input devices found", "live modes unavailable"
        default_in = next((d for d in inputs if d["is_default_input"]), inputs[0])
        return (
            PASS,
            f"{len(inputs)} input / {len(outputs)} output device(s); default input: {default_in['name']}",
            "",
        )

    checks.append(_run("audio devices", check_devices))

    # ------------------------------------------------------- device sample rate
    def check_device_rate() -> tuple[str, str, str]:
        from ..audio.devices import DeviceError, check_settings, resolve_device

        try:
            dev = resolve_device(cfg.live.input_device, "input")
            check_settings(dev, cfg.audio.sample_rate, 1, "input")
        except DeviceError as exc:
            return WARN, str(exc).splitlines()[0], "pick another device or change its Windows sample rate"
        return PASS, f"input device accepts {cfg.audio.sample_rate} Hz mono float32", ""

    checks.append(_run("device sample rate", check_device_rate))

    # ------------------------------------------------------------ end-to-end
    def check_pipeline() -> tuple[str, str, str]:
        """End-to-end check on real speech where available.

        A synthetic tone must not be used here: DeepFilterNet is a *speech* enhancer,
        so it correctly suppresses a sine wave as noise, and the check would fail for
        the wrong reason. If no real speech is present the neural stage is excluded
        and only the adaptive stage is exercised.
        """
        from ..dataset.sources import load_speech_files
        from ..metrics.intrusive import si_sdr
        from ..pipeline import Pipeline

        sr = cfg.audio.sample_rate
        rng = np.random.default_rng(5)
        speech_files = load_speech_files(cfg.dataset.clean_dir)
        local = cfg.model_copy(deep=True)
        local.neural.streaming.offline_framing = "whole_file"
        note = ""

        if speech_files:
            from ..audio.io import load_audio

            speech = load_audio(speech_files[0], sr)[: 4 * sr]
            if len(speech) < sr:
                speech = np.pad(speech, (0, sr - len(speech)))
            speech = (speech * (0.06 / (float(np.sqrt(np.mean(speech**2))) + 1e-20))).astype(np.float32)
        else:
            # No real speech: exercise the adaptive stage only and say so.
            local.pipeline.order = "nlms_only"  # type: ignore[assignment]
            t = np.arange(3 * sr) / sr
            speech = (0.06 * np.sin(2 * np.pi * 200 * t) * (1 + 0.5 * np.sin(2 * np.pi * 2.0 * t))).astype(
                np.float32
            )
            note = " (no real speech available: neural stage excluded, synthetic tone used)"

        h = rng.standard_normal(400) * np.exp(-np.arange(400) / 90.0)
        h /= np.linalg.norm(h)
        dry = (0.06 * rng.standard_normal(len(speech))).astype(np.float32)
        wet = np.convolve(dry, h)[: len(speech)].astype(np.float32)
        primary = (speech + wet).astype(np.float32)

        pipe = Pipeline(local, model=model)
        result = pipe.process(primary, dry)

        before = si_sdr(speech[: len(result.primary)], result.primary)
        after = si_sdr(speech[: len(result.output)], result.output)
        if not np.all(np.isfinite(result.output)):
            return FAIL, "pipeline output contained non-finite samples", "check the DSP stages"
        status = PASS if after > before else WARN
        return (
            status,
            f"{pipe.describe()}: SI-SDR {before:.1f} -> {after:.1f} dB ({after - before:+.1f} dB){note}",
            "" if after > before else "no improvement; investigate before demoing",
        )

    checks.append(_run("end-to-end pipeline", check_pipeline))

    # --------------------------------------------------------------- data files
    def check_data() -> tuple[str, str, str]:
        from ..dataset.sources import index_noise_dir, load_speech_files

        speech = load_speech_files(cfg.dataset.clean_dir)
        noise = index_noise_dir(cfg.dataset.noise_dir)
        total_noise = sum(len(v) for v in noise.values())
        if not speech and not total_noise:
            return WARN, "no evaluation corpora present", "run `anc fetch-data`"
        empty = [k for k, v in noise.items() if not v]
        detail = (
            f"{len(speech)} speech file(s) from {len({p.parent.name for p in speech})} speaker(s), "
            f"{total_noise} noise file(s) {({k: len(v) for k, v in noise.items()})}"
        )
        if empty or not speech:
            return WARN, detail + f"; missing: {empty or 'speech'}", "run `anc fetch-data`"
        return PASS, detail, ""

    checks.append(_run("evaluation corpora", check_data))

    # ------------------------------------------------------------------- output
    width = max(len(c.name) for c in checks) + 2
    print("\nself-test\n" + "=" * 78)
    for c in checks:
        marker = {PASS: "[ok]  ", FAIL: "[FAIL]", WARN: "[warn]", SKIP: "[skip]"}[c.status]
        print(f"{marker} {c.name:<{width}} {c.detail}")
        if c.action:
            print(f"{'':<{width + 7}} -> {c.action}")
    failures = [c for c in checks if c.status == FAIL]
    warnings = [c for c in checks if c.status == WARN]
    print("=" * 78)
    if failures:
        print(f"NO-GO: {len(failures)} check(s) failed, {len(warnings)} warning(s).")
        return 1
    if warnings:
        print(f"GO with {len(warnings)} warning(s). Offline mode is usable; see the notes above.")
        return 0
    print("GO: everything passed.")
    return 0
