"""Streamlit dashboard.

Launch with ``anc dashboard`` (or ``streamlit run anc_defence/ui/app.py``).

Five tabs:

``Overview``    the latest evaluation session: headline figures, method comparison,
                per-category and per-SNR results, and the generated charts.
``Explore``     pick any corpus example, run the pipeline on it, and inspect every
                stage: waveform, spectrogram, the spectral delta showing exactly what
                each stage changed, and a player for every stage plus the removed noise.
``Corpus``      what is in the supplied dataset and how balanced it is.
``Live``        record from the microphone, process, and compare before and after.
``System``      host, model, latency budget and real-time factor.

Everything is pure Python, so this runs unchanged on a Jetson.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any, Optional

# Some OpenSSL-backed hashlib builds (observed on JetPack 5.1.x / Python 3.8 aarch64)
# raise "'usedforsecurity' is an invalid keyword argument for openssl_md5()" (and the
# same for sha1/sha256/sha512) even though the stdlib has accepted that kwarg on every
# hash constructor since Python 3.9. `usedforsecurity` is only a FIPS-auditing
# annotation - dropping it changes nothing about the hash itself - so catching the
# TypeError and retrying without it is safe everywhere, not just on this platform.
#
# This bites more than one call site:
#   - Streamlit's own cache hasher (streamlit/util.py: create_fast_hasher) falls back to
#     hashlib.new("md5", usedforsecurity=False), which crashes every cached function -
#     including the one behind the live-mic run - right after a session completes.
#   - Starlette's static/file response code computes an ETag with
#     hashlib.md5(data, usedforsecurity=False) directly (not via hashlib.new), which is
#     the likely cause of the report download button not working.
#
# All of these do a fresh `hashlib.<name>(...)` attribute lookup at call time rather than
# `from hashlib import md5` at import time, so patching the shared hashlib module's
# attributes here fixes every one of them regardless of when the calling library was
# imported - this only needs to run before the *call*, not before the *import*.
for _algo in ("md5", "sha1", "sha256", "sha512", "new"):
    _orig = getattr(hashlib, _algo)

    def _compat(*args, _orig=_orig, **kwargs):
        try:
            return _orig(*args, **kwargs)
        except TypeError:
            kwargs.pop("usedforsecurity", None)
            return _orig(*args, **kwargs)

    setattr(hashlib, _algo, _compat)
del _algo, _orig

import numpy as np
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anc_defence.config import Config, load_config  # noqa: E402
from anc_defence.dsp.normalise import active_speech_dbfs  # noqa: E402

SR = 48000
PLOT_HEIGHT = 260

st.set_page_config(
    page_title="ANC for defence comms - SIH 26052",
    page_icon="~",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------- caching


@st.cache_data(show_spinner=False)
def get_config(overrides: tuple[str, ...] = ()) -> Config:
    path = REPO_ROOT / "configs" / "default.yaml"
    return load_config([path] if path.is_file() else [], list(overrides))


@st.cache_resource(show_spinner="Loading DeepFilterNet3...")
def get_model(device: str, threads: Optional[int]):
    from anc_defence.enhance.streaming import DfnModel

    cfg = get_config()
    neural = cfg.neural.model_copy(deep=True)
    neural.device = device  # type: ignore[assignment]
    neural.num_threads = threads
    return DfnModel(neural, cfg.audio.sample_rate).load()


@st.cache_data(show_spinner="Indexing the corpus...")
def get_corpus_records() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from anc_defence.dataset.plain import load_plain_corpus

    cfg = get_config()
    try:
        corpus = load_plain_corpus(cfg.plain)
    except FileNotFoundError:
        # The corpus is gitignored (licence unverified) and is not on every machine that
        # runs the dashboard, e.g. a fresh Jetson clone. The offline tabs need it; the
        # Live microphone tab and the report Overview do not. Degrade to an empty result
        # here rather than crashing the whole page, matching how `anc selftest` treats
        # the same condition as a warning, not a hard failure.
        return [], {}
    return [e.as_dict() for e in corpus.examples], corpus.summary()


@st.cache_data(show_spinner=False)
def load_wav(path: str, max_seconds: float = 12.0) -> np.ndarray:
    from anc_defence.audio.io import load_audio

    return load_audio(path, SR)[: int(max_seconds * SR)]


def sessions() -> list[Path]:
    root = REPO_ROOT / "sessions"
    if not root.is_dir():
        return []
    return sorted((d for d in root.iterdir() if d.is_dir()), reverse=True)


@st.cache_data(show_spinner=False)
def load_session(path: str) -> dict[str, Any]:
    p = Path(path) / "metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


# ----------------------------------------------------------------- audio utils


def wav_bytes(x: np.ndarray, sample_rate: int = SR) -> bytes:
    """Encode float audio as 16-bit WAV for the browser's audio player."""
    import soundfile as sf

    buf = io.BytesIO()
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    safe = x / peak * 0.98 if peak > 1.0 else x
    sf.write(buf, np.asarray(safe, dtype=np.float32), sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _round(df, precision: int = 3):
    """Round float columns for display without pandas Styler.

    pandas Styler.background_gradient imports matplotlib, and pandas 3.0 requires a newer
    matplotlib than the numpy<2 constraint (from deepfilternet) allows. A plain rounded
    DataFrame avoids the import entirely.
    """
    import pandas as pd

    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(precision)
    return out


def player(label: str, x: np.ndarray, note: str = "") -> None:
    st.caption(f"**{label}**" + (f" - {note}" if note else ""))
    if x.size:
        st.audio(wav_bytes(x), format="audio/wav")
    else:
        st.info("no audio")


# ---------------------------------------------------------------------- plots


def waveform_figure(taps: dict[str, np.ndarray], sample_rate: int = SR):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(taps)
    peak = max((float(np.max(np.abs(v))) for v in taps.values() if v.size), default=1.0) or 1.0
    fig, axes = plt.subplots(len(names), 1, figsize=(11, 1.35 * len(names) + 0.5), sharex=True)
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        x = taps[name]
        ax.plot(np.arange(len(x)) / sample_rate, x, linewidth=0.35, color="#1f77b4")
        ax.set_ylim(-1.05 * peak, 1.05 * peak)
        ax.set_ylabel(name.replace("after_", "").replace("_", "\n"), fontsize=7)
        ax.grid(alpha=0.25, linewidth=0.4)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("time (s)", fontsize=8)
    fig.tight_layout()
    return fig


def spectrogram_figure(taps: dict[str, np.ndarray], sample_rate: int = SR, n_fft: int = 1024):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(taps)
    specs = []
    for name in names:
        x = np.asarray(taps[name], dtype=np.float64)
        if len(x) < n_fft:
            x = np.pad(x, (0, n_fft - len(x)))
        hop = n_fft // 4
        frames = 1 + (len(x) - n_fft) // hop
        w = np.hanning(n_fft)
        S = np.empty((n_fft // 2 + 1, frames))
        for i in range(frames):
            S[:, i] = np.abs(np.fft.rfft(x[i * hop : i * hop + n_fft] * w))
        specs.append(20.0 * np.log10(S + 1e-12))
    vmax = max(float(np.max(s)) for s in specs)
    vmin = vmax - 80.0
    fig, axes = plt.subplots(len(names), 1, figsize=(11, 1.8 * len(names) + 0.5), sharex=True)
    if len(names) == 1:
        axes = [axes]
    for ax, name, S in zip(axes, names, specs):
        dur = len(taps[name]) / sample_rate
        im = ax.imshow(S, origin="lower", aspect="auto", extent=(0, dur, 0, sample_rate / 2000),
                       vmin=vmin, vmax=vmax, cmap="magma")
        ax.set_ylabel(f"{name.replace('after_', '')}\nkHz", fontsize=7)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("time (s)", fontsize=8)
    fig.colorbar(im, ax=axes, pad=0.01, fraction=0.02).ax.tick_params(labelsize=6)
    return fig


def live_waveform_figure(
    input_signal: np.ndarray, output_signal: np.ndarray, sample_rate: int = SR, window_s: float = 12.0
):
    """Stacked input/output waveform, most recent ``window_s`` seconds, for the live-updating panel.

    Same visual convention as the offline ``waveforms`` plot (shared style, one panel per
    tap, fixed layout) so a viewer sees the same kind of chart whether it is live or in a
    report - just scrolling. The x-axis is anchored to the newest sample rather than to
    session start, so it behaves like a scrolling strip-chart instead of stretching wider
    forever as the session runs.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    window_n = max(1, int(window_s * sample_rate))
    inp = input_signal[-window_n:] if input_signal.size else input_signal
    out = output_signal[-window_n:] if output_signal.size else output_signal
    peak = max(
        (float(np.max(np.abs(v))) for v in (inp, out) if v.size), default=1.0
    ) or 1.0

    fig, axes = plt.subplots(2, 1, figsize=(11, 3.4), sharex=False)
    for ax, name, sig, colour in (
        (axes[0], "input", inp, "#1f77b4"),
        (axes[1], "output", out, "#d62728"),
    ):
        if sig.size:
            t_end = len(sig) / sample_rate
            t0 = max(0.0, t_end - window_s)
            t = t0 + np.arange(len(sig)) / sample_rate
            ax.plot(t, sig, linewidth=0.4, color=colour)
            ax.set_xlim(t0, max(t0 + window_s, t_end))
        ax.set_ylim(-1.05 * peak, 1.05 * peak)
        ax.set_ylabel(name, fontsize=8)
        ax.grid(alpha=0.25, linewidth=0.4)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("time (s)", fontsize=8)
    axes[0].set_title(f"Live waveform (last {window_s:.0f} s, shared amplitude scale)", fontsize=9)
    fig.tight_layout()
    return fig


def run_live_with_live_chart(run_cfg: "Config", seconds: float, refresh_s: float = 0.4):
    """Run a live session while redrawing a scrolling input/output waveform in-place.

    ``run_live`` (and the ``LiveEngine`` it drives) blocks for the session duration on
    whichever thread calls it, and Streamlit only redraws widgets between script reruns -
    it has no built-in animation loop. To get a chart that visibly updates *during* the
    session rather than only appearing once it ends, the engine is run on a background
    thread and this (the main Streamlit script) thread polls its already-thread-safe
    stats/recorded buffers on a timer, redrawing an ``st.empty()`` placeholder - the same
    pattern ``LiveDashboard`` uses for the terminal meters, just targeting a matplotlib
    figure instead of a rich table.
    """
    import threading

    from anc_defence.modes.live import LiveEngine, run_live
    from anc_defence.utils.session import create_session

    session = create_session(run_cfg, "live_mic")
    engine_holder: dict[str, Any] = {}
    result_holder: dict[str, Any] = {}

    def _capture_engine(engine: LiveEngine, duration_s: float) -> None:
        # Installed as run_live's dashboard callback: records the engine for the polling
        # loop below, then blocks for the session exactly as the terminal dashboard would,
        # so stream teardown / flushing in LiveEngine.run() still happens at the right time.
        engine_holder["engine"] = engine
        end = time.perf_counter() + duration_s
        while time.perf_counter() < end:
            time.sleep(0.05)

    def _worker() -> None:
        try:
            result_holder["artefacts"] = run_live(run_cfg, session=session, dashboard=_capture_engine)
        except Exception as exc:  # noqa: BLE001
            result_holder["error"] = exc

    thread = threading.Thread(target=_worker, name="streamlit-live-session", daemon=True)
    thread.start()

    chart = st.empty()
    meters = st.empty()
    # Wait for the engine to exist (stream startup) before the first redraw.
    t_wait = time.perf_counter()
    while "engine" not in engine_holder and thread.is_alive() and time.perf_counter() - t_wait < 10.0:
        time.sleep(0.05)

    engine = engine_holder.get("engine")
    t0 = time.perf_counter()
    while thread.is_alive() or (engine is not None and time.perf_counter() - t0 < seconds + 2.0):
        if engine is not None:
            inp = engine.signal(engine.recorded_input) if engine.recorded_input else np.zeros(0, dtype=np.float32)
            outp = engine.signal(engine.recorded_output) if engine.recorded_output else np.zeros(0, dtype=np.float32)
            chart.pyplot(live_waveform_figure(inp, outp, SR), use_container_width=True)
            in_db = engine.stats.input_dbfs[-1] if engine.stats.input_dbfs else float("-inf")
            out_db = engine.stats.output_dbfs[-1] if engine.stats.output_dbfs else float("-inf")
            elapsed = min(seconds, time.perf_counter() - t0)
            meters.caption(
                f"elapsed {elapsed:4.1f} s / {seconds:.0f} s  &middot;  "
                f"input {in_db:5.1f} dBFS  &middot;  output {out_db:5.1f} dBFS  &middot;  "
                f"blocks in/out {engine.stats.blocks_in}/{engine.stats.blocks_out}"
            )
        if not thread.is_alive():
            break
        time.sleep(refresh_s)

    thread.join(timeout=5.0)

    # Leave the final frame on screen (do not clear the placeholders): for a demo
    # recording, the graph should still be showing the full input/output waveform
    # after the session ends, not disappear the instant capture stops.
    if engine is not None:
        inp = engine.signal(engine.recorded_input) if engine.recorded_input else np.zeros(0, dtype=np.float32)
        outp = engine.signal(engine.recorded_output) if engine.recorded_output else np.zeros(0, dtype=np.float32)
        chart.pyplot(
            live_waveform_figure(inp, outp, SR, window_s=max(seconds, len(inp) / SR if inp.size else seconds)),
            use_container_width=True,
        )
        meters.caption(
            f"Session finished - {seconds:.0f} s  &middot;  "
            f"blocks in/out {engine.stats.blocks_in}/{engine.stats.blocks_out}"
        )

    if "error" in result_holder:
        st.error(f"Live session failed: {result_holder['error']}")
        return {}
    return result_holder.get("artefacts", {})


def delta_figure(before: np.ndarray, after: np.ndarray, sample_rate: int = SR, n_fft: int = 1024):
    """Spectral difference: blue is energy removed, red is energy added."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(len(before), len(after))
    a, b = np.asarray(before[:n], np.float64), np.asarray(after[:n], np.float64)
    if n < n_fft:
        a, b = np.pad(a, (0, n_fft - n)), np.pad(b, (0, n_fft - n))
    hop = n_fft // 4
    frames = 1 + (len(a) - n_fft) // hop
    w = np.hanning(n_fft)
    Sa = np.empty((n_fft // 2 + 1, frames))
    Sb = np.empty_like(Sa)
    for i in range(frames):
        s = i * hop
        Sa[:, i] = np.abs(np.fft.rfft(a[s : s + n_fft] * w))
        Sb[:, i] = np.abs(np.fft.rfft(b[s : s + n_fft] * w))
    delta = 20 * np.log10(Sb + 1e-12) - 20 * np.log10(Sa + 1e-12)
    limit = float(np.percentile(np.abs(delta), 98)) or 1.0
    fig, ax = plt.subplots(figsize=(11, 3.0))
    im = ax.imshow(delta, origin="lower", aspect="auto",
                   extent=(0, n / sample_rate, 0, sample_rate / 2000),
                   vmin=-limit, vmax=limit, cmap="coolwarm")
    ax.set_xlabel("time (s)", fontsize=8)
    ax.set_ylabel("kHz", fontsize=8)
    ax.tick_params(labelsize=7)
    cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03)
    cb.set_label("dB change  (blue = removed, red = added)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------- sidebar

st.sidebar.title("ANC for defence comms")
st.sidebar.caption("SIH problem statement 26052 - single microphone")

cfg = get_config()
st.sidebar.markdown("**Delivered pipeline**")
st.sidebar.code("mic -> preprocess\n    -> DeepFilterNet3\n    -> volume normalisation\n    -> out", language=None)

with st.sidebar.expander("Pipeline settings", expanded=False):
    st.write(f"Sample rate: **{cfg.audio.sample_rate} Hz** (corpus is 16 kHz, upsampled)")
    st.write(f"Hop: **{cfg.audio.hop_size}** samples ({1000 * cfg.audio.hop_size / cfg.audio.sample_rate:.0f} ms)")
    st.write(f"Model: **{cfg.neural.model}**, pretrained, unmodified")
    st.write(f"Normaliser: **{cfg.normalise.mode}**, target **{cfg.normalise.target_dbfs} dBFS**")
    st.write(f"Hold gain during pauses: **{cfg.normalise.hold_during_pause}**")

device = st.sidebar.selectbox("Neural device", ["cpu", "cuda"], index=0)
threads = st.sidebar.selectbox(
    "Torch threads", ["default", "1 (portability figure)"], index=0
)
thread_count = 1 if threads.startswith("1") else None

st.sidebar.divider()
st.sidebar.caption(
    "No model was trained in this project: the neural stage is the pretrained "
    "DeepFilterNet3 checkpoint, unmodified."
)

tab_overview, tab_explore, tab_corpus, tab_live, tab_system = st.tabs(
    ["Overview", "Explore a file", "Corpus", "Live microphone", "System"]
)


# ------------------------------------------------------------------- overview

with tab_overview:
    st.header("Evaluation results")
    all_sessions = sessions()
    if not all_sessions:
        st.warning("No sessions yet. Run `anc evaluate` to produce one.")
    else:
        labels = [d.name for d in all_sessions]
        pick = st.selectbox("Session", labels, index=0)
        session_dir = all_sessions[labels.index(pick)]
        payload = load_session(str(session_dir))

        if not payload:
            st.warning(f"{session_dir.name} has no metrics.json")
        else:
            st.caption(payload.get("title", ""))

            # No pass/fail target verdicts: the headline figures are shown as measured,
            # with the definition each one uses stated in the scope note.
            head = payload.get("headline", {})
            scope = payload.get("measurement", {}).get("scope", "")
            if head.get("n"):
                st.subheader("Headline figures")
                if scope:
                    st.caption(scope.replace("<b>", "**").replace("</b>", "**")
                               .replace("<i>", "*").replace("</i>", "*"))
                agg = head.get("aggregate", {})
                cols = st.columns(5)
                for col, (label, key, fmt) in zip(cols, (
                    ("Output SNR dB", "residual_noise_snr_db", "{:.2f}"),
                    ("SNR gain dB", "snr_gain_db", "{:+.2f}"),
                    ("Noise removed dB", "noise_reduction_db", "{:+.1f}"),
                    ("STOI", "stoi", "{:.3f}"),
                    ("PESQ", "pesq", "{:.3f}"),
                )):
                    v = agg.get(key)
                    col.metric(label, "n/a" if v is None else fmt.format(v))

            rows = payload.get("aggregate_by_method", [])
            if rows:
                st.subheader("Method comparison")
                import pandas as pd

                keep = ["method", "n", "pesq", "stoi", "estoi", "si_sdr", "snr_improvement_db",
                        "segmental_snr", "lsd", "speech_attenuation_db", "noise_reduction_db",
                        "rms_reduction_db", "rms_reduction_pct", "output_level_dbfs", "rtf"]
                df = pd.DataFrame(rows)
                df = df[[c for c in keep if c in df.columns]]
                # Plain formatting only: pandas' .background_gradient() imports matplotlib
                # and pandas 3.0 wants a newer matplotlib than the numpy<2 pin allows.
                st.dataframe(_round(df), use_container_width=True)
                st.caption(
                    "Quality metrics for the full pipeline are computed on the gain-compensated "
                    "output, so the volume normaliser is not scored for changing level - which is "
                    "the one thing it is supposed to do. `output_level_dbfs` shows the level it "
                    "actually delivered."
                )

            per_cat = payload.get("headline", {}).get("per_category", [])
            if per_cat:
                st.subheader("Per noise category")
                import pandas as pd

                df = pd.DataFrame(per_cat)
                cols = [c for c in ("category", "n", "pesq", "stoi", "snr_improvement_db",
                                    "speech_attenuation_db", "noise_reduction_db") if c in df.columns]
                st.dataframe(_round(df[cols]), use_container_width=True)

            figures = sorted((session_dir / "figures").glob("*.png"))
            if figures:
                st.subheader("Figures")
                names = [f.stem for f in figures]
                chosen = st.multiselect(
                    "Show", names,
                    default=[n for n in names if n.startswith("compare") or n.startswith("heatmap")][:3],
                )
                for name in chosen:
                    st.image(str(figures[names.index(name)]), caption=name, use_container_width=True)

            pdf = session_dir / "report.pdf"
            if pdf.is_file():
                st.download_button(
                    "Download the PDF report", pdf.read_bytes(), file_name=pdf.name,
                    mime="application/pdf",
                )


# -------------------------------------------------------------------- explore

with tab_explore:
    st.header("Run the pipeline on one file and see what each stage changed")
    records, summary = get_corpus_records()
    if not records:
        st.warning("Corpus not found. Check `plain.root` in configs/default.yaml.")
    else:
        c1, c2, c3 = st.columns([1, 1, 2])
        cats = sorted({r["category"] for r in records})
        cat = c1.selectbox("Noise category", cats, index=cats.index("gunshot") if "gunshot" in cats else 0)
        snrs = sorted({r["snr_db"] for r in records if r["category"] == cat})
        snr = c2.selectbox("Input SNR (dB)", snrs, index=0)
        pool = [r for r in records if r["category"] == cat and r["snr_db"] == snr]
        ids = [r["id"] for r in pool]
        chosen_id = c3.selectbox(f"Example ({len(ids)} available)", ids, index=0)
        record = pool[ids.index(chosen_id)]

        order = st.radio(
            "Pipeline", ["dfn_then_normalise", "dfn_only", "normalise_only", "passthrough"],
            horizontal=True, index=0,
        )

        if st.button("Process", type="primary"):
            from anc_defence.pipeline import Pipeline

            run_cfg = cfg.model_copy(deep=True)
            run_cfg.pipeline.order = order  # type: ignore[assignment]
            run_cfg.neural.device = device  # type: ignore[assignment]
            run_cfg.neural.num_threads = thread_count
            model = get_model(device, thread_count) if "dfn" in order else None

            noisy = load_wav(record["noisy_path"], cfg.plain.max_duration_s)
            clean = load_wav(record["clean_path"], cfg.plain.max_duration_s)
            n = min(len(noisy), len(clean))
            noisy, clean = noisy[:n], clean[:n]

            with st.spinner("Processing..."):
                t0 = time.perf_counter()
                pipe = Pipeline(run_cfg, model=model)
                result = pipe.process(noisy)
                elapsed = time.perf_counter() - t0

            taps = result.taps(include_reference=False)
            st.success(
                f"{pipe.describe()}  -  {n / SR:.2f} s of audio in {elapsed:.2f} s "
                f"(RTF {elapsed / (n / SR):.3f})"
            )

            # ---- metrics ------------------------------------------------------
            from anc_defence.metrics.intrusive import compute_intrusive

            base = compute_intrusive(clean, result.primary, SR, cfg.metrics)
            env = (
                pipe.normaliser.gain_envelope(len(result.output))
                if pipe.normaliser is not None and pipe.normaliser.enabled
                else None
            )
            rows = []
            for name, sig in taps.items():
                scored = sig
                if env is not None and name == f"after_normalise":
                    scored = sig / np.maximum(env[: len(sig)], 1e-6)
                m = compute_intrusive(clean[: len(scored)], scored, SR, cfg.metrics)
                rows.append({
                    "stage": name,
                    "level dBFS": round(active_speech_dbfs(sig, SR), 1),
                    "PESQ": round(m.pesq, 3),
                    "STOI": round(m.stoi, 3),
                    "ESTOI": round(m.estoi, 3),
                    "SI-SDR dB": round(m.si_sdr, 2),
                    "SNRi dB": round(m.si_sdr - base.si_sdr, 2),
                    "segSNR dB": round(m.segmental_snr, 2),
                    "speech atten dB": round(m.speech_attenuation_db, 2),
                })
            import pandas as pd

            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            m1, m2, m3, m4 = st.columns(4)
            final = rows[-1]
            m1.metric("PESQ", f"{final['PESQ']:.3f}", f"{final['PESQ'] - base.pesq:+.3f}")
            m2.metric("STOI", f"{final['STOI']:.3f}", f"{final['STOI'] - base.stoi:+.3f}")
            m3.metric("SNR improvement", f"{final['SNRi dB']:+.2f} dB")
            m4.metric("Output level", f"{final['level dBFS']:.1f} dBFS",
                      f"target {cfg.normalise.target_dbfs:g}")

            # ---- audio --------------------------------------------------------
            st.subheader("Listen")
            cols = st.columns(min(4, len(taps) + 2))
            player_taps = {"clean target (reference)": clean, **taps}
            player_taps["removed by the pipeline"] = result.removed()
            for i, (name, sig) in enumerate(player_taps.items()):
                with cols[i % len(cols)]:
                    note = ""
                    if name == "removed by the pipeline":
                        note = "if you hear speech here, the pipeline is taking too much"
                    player(name, sig, note)

            # ---- figures ------------------------------------------------------
            st.subheader("Waveforms")
            st.pyplot(waveform_figure({"clean target": clean, **taps}), use_container_width=True)

            st.subheader("Spectrograms (shared colour scale)")
            st.pyplot(spectrogram_figure({"clean target": clean, **taps}), use_container_width=True)

            st.subheader("What each stage changed")
            names = list(taps)
            stage_pairs = list(zip(["input"] + names, names))
            for before_name, after_name in stage_pairs:
                if before_name == after_name:
                    continue
                before = taps.get(before_name, result.primary)
                st.markdown(f"**{before_name}** to **{after_name}**")
                st.pyplot(delta_figure(before, taps[after_name]), use_container_width=True)

            if pipe.normaliser is not None and pipe.normaliser.diagnostics is not None:
                diag = pipe.normaliser.diagnostics
                if diag.gain_db:
                    st.subheader("Volume normalisation")
                    from anc_defence.report import plots
                    import tempfile

                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
                        path = Path(fh.name)
                    plots.agc_trace(diag, path, dpi=110)
                    st.image(str(path), use_container_width=True)
                    st.json(pipe.normaliser.summary())


# --------------------------------------------------------------------- corpus

with tab_corpus:
    st.header("Supplied corpus")
    records, summary = get_corpus_records()
    if not records:
        st.warning("Corpus not found.")
    else:
        import pandas as pd

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Labelled examples", summary.get("labelled_examples", 0))
        c2.metric("Unlabelled files", summary.get("unlabelled_files", 0))
        c3.metric("Categories", len(summary.get("categories", {})))
        c4.metric("Native rate", f"{summary.get('native_sample_rate', 0)} Hz")

        st.caption(
            "The unlabelled files have no metadata row (no category, no SNR), so they are excluded "
            "from scored results and kept only for listening."
        )

        counts = summary.get("categories", {})
        taxonomy = summary.get("category_taxonomy_map", {})
        notes = summary.get("taxonomy_notes", {})
        total = max(1, sum(counts.values()))
        df = pd.DataFrame([
            {
                "category": k,
                "taxonomy": taxonomy.get(k, "?"),
                "examples": v,
                "share %": round(100.0 * v / total, 1),
                "stands in for": notes.get(k, ""),
            }
            for k, v in sorted(counts.items(), key=lambda kv: -kv[1])
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.warning(
            f"Gunshot is {df.iloc[0]['share %']:.0f}% of the corpus. An unweighted average over the "
            "whole set would essentially be a gunshot score, so the evaluation draws a "
            "category x SNR balanced subset and every table is broken out per category."
        )

        st.subheader("Distribution")
        c1, c2 = st.columns(2)
        c1.bar_chart(df.set_index("category")["examples"])
        snr_counts = summary.get("snr_distribution", {})
        c2.bar_chart(pd.DataFrame({"examples": snr_counts.values()}, index=list(snr_counts.keys())))


# ----------------------------------------------------------------------- live

with tab_live:
    st.header("Live microphone session")
    st.caption(
        "Runs a full live session from here and writes a complete session directory with "
        "report.pdf and metrics.json, exactly as the terminal `anc run --mode live_mic` does. "
        "Record one session with ANC off and one with ANC on, then compare them in the Overview tab."
    )

    anc_on = st.toggle(
        "ANC", value=True,
        help="On: mic -> preprocess -> DeepFilterNet3 -> volume normalisation. "
             "Off: the microphone signal passes straight through, no suppression and no "
             "level control. Off is the baseline the 'on' run is compared against.",
    )
    if anc_on:
        st.success("**ANC ON** - full pipeline: DeepFilterNet3 + volume normalisation")
    else:
        st.warning(
            "**ANC OFF** - passthrough. The report will show no noise removed and no level "
            "correction. This is the intended baseline, not a fault."
        )

    try:
        from anc_defence.audio.devices import format_device_table, list_devices

        devices = [d for d in list_devices() if d["max_input_channels"] > 0]
        names = [f"{d['index']}: {d['name']}" for d in devices]
        if not names:
            st.error("No input devices found. A microphone is required for a live session.")
            raise RuntimeError("no input device")

        col_a, col_b = st.columns(2)
        with col_a:
            chosen = st.selectbox("Input device", names, index=0)
            seconds = st.slider("Session length (s)", 5, 60, 20)
        with col_b:
            profile = st.selectbox(
                "Latency profile",
                ["low_latency", "balanced", "quality", "max_suppression"],
                index=0,
                help="Larger chunks remove more noise but lag further behind.",
                disabled=not anc_on,
            )
            monitor = st.checkbox(
                "Monitor to speakers/headphones", value=False,
                help="Plays the processed audio out live. Use headphones - speakers will feed "
                     "back into the microphone and ruin the recording.",
            )
            demo_delay = st.slider(
                "Demo output delay (s)", 0.0, 10.0, 0.0, 0.5,
                help="Purely a demo aid: holds the already-processed output before releasing it, so "
                     "input and output do not overlap when recorded in one take (e.g. for a video). "
                     "Changes nothing about processing - 0 s is the real, minimum-latency behaviour, "
                     "and the report's measured latency figure always reflects that real minimum, "
                     "not this setting. Set once before starting; it cannot be changed mid-session.",
            )

        # Injecting a known noise at a known SNR gives the run a pseudo-clean reference (the
        # pre-mix microphone capture), which is the only way a live run can report PESQ, STOI
        # and SI-SDR. Without it only levels and a reference-free SNR estimate are available.
        with st.expander(
            "Inject noise digitally (enables PESQ / STOI / SI-SDR on a live run)", expanded=False
        ):
            st.caption(
                "Mixes a noise file into the captured microphone signal at a chosen SNR and keeps "
                "the pre-mix capture as the reference. Without this, a live run has no clean "
                "reference, so only levels and a reference-free SNR estimate can be reported."
            )
            inject = st.checkbox("Inject noise", value=False)
            noise_path: Optional[str] = None
            inject_snr = 0.0
            if inject:
                records, _ = get_corpus_records()
                noise_options = sorted(
                    {r["noise_path"] for r in records if r.get("noise_path")}
                )
                if noise_options:
                    labels = [Path(p).name for p in noise_options[:400]]
                    pick = st.selectbox("Noise file (from the corpus)", labels, index=0)
                    noise_path = noise_options[labels.index(pick)]
                    inject_snr = st.slider("Injected SNR (dB)", -10.0, 15.0, 0.0, 2.5)
                else:
                    st.info("No noise-only files found in the corpus metadata.")

        if st.button(
            f"Run {seconds} s live session with ANC {'ON' if anc_on else 'OFF'}",
            type="primary",
        ):
            run_cfg = cfg.model_copy(deep=True)
            run_cfg.pipeline.order = (  # type: ignore[assignment]
                "dfn_then_normalise" if anc_on else "passthrough"
            )
            # Tags the session directory, so the two runs are told apart in Overview.
            run_cfg.run.name = "anc_on" if anc_on else "anc_off"
            run_cfg.neural.device = device  # type: ignore[assignment]
            run_cfg.neural.num_threads = thread_count
            run_cfg.live.duration_s = float(seconds)
            run_cfg.live.input_device = chosen.split(":")[0]
            run_cfg.live.monitor = bool(monitor)
            run_cfg.live.latency_profile = profile if anc_on else "low_latency"  # type: ignore[assignment]
            run_cfg.live.demo_delay_s = float(demo_delay)
            if noise_path:
                run_cfg.live.noise_file = Path(noise_path)
                run_cfg.live.noise_snr_db = float(inject_snr)

            status = st.empty()
            status.info(f"Recording {seconds} s - speak now.")
            if demo_delay > 0.0:
                st.caption(
                    f"Demo output delay is **{demo_delay:g} s**: the waveform panel below and the "
                    "monitored/recorded output will lag the input by that much on top of the real "
                    "pipeline latency."
                )

            artefacts = run_live_with_live_chart(run_cfg, seconds)

            if artefacts:
                status.empty()
                anchor = artefacts.get("pdf") or artefacts.get("json") or artefacts.get("csv")
                session_dir = Path(anchor).parent
                st.success(f"Session written: `{session_dir.name}`")
                st.session_state["last_live_session"] = str(session_dir)
                load_session.clear()

                pdf = session_dir / "report.pdf"
                if pdf.is_file():
                    st.download_button(
                        "Download report.pdf", pdf.read_bytes(),
                        file_name=f"{session_dir.name}_report.pdf", mime="application/pdf",
                    )
                st.caption(f"Full report and figures: `{session_dir}`")

                for name, label in (
                    ("input", "recorded input"), ("output", "pipeline output"),
                ):
                    wav = session_dir / "audio" / f"{name}.wav"
                    if wav.is_file():
                        st.markdown(f"**{label}**")
                        st.audio(str(wav))

        # ---- ANC off vs on comparison ---------------------------------------
        live_sessions = [d for d in sessions() if "live_mic" in d.name]
        off = [d for d in live_sessions if d.name.endswith("anc_off")]
        on = [d for d in live_sessions if d.name.endswith("anc_on")]
        if off and on:
            st.divider()
            st.subheader("ANC off versus ANC on")
            st.caption(
                "Most recent run of each. Both used the same microphone and the same code path; "
                "the only difference is whether the suppression and normalisation stages ran."
            )
            rows = []
            for label, d in (("ANC off", off[0]), ("ANC on", on[0])):
                p = d / "metrics.json"
                if not p.is_file():
                    continue
                pl = json.loads(p.read_text(encoding="utf-8"))
                ni = pl.get("non_intrusive", {})
                rows.append({
                    "run": label,
                    "session": d.name,
                    "pipeline": pl.get("config", {}).get("pipeline", {}).get("order", "?"),
                    "noise removed dB": (
                        None if ni.get("noise_reduction_db") is None
                        else round(ni["noise_reduction_db"], 2)
                    ),
                    "est. SNR change dB": (
                        None if ni.get("estimated_snr_change_db") is None
                        else round(ni["estimated_snr_change_db"], 2)
                    ),
                    "output level dBFS": (
                        None if ni.get("output_level_dbfs") is None
                        else round(ni["output_level_dbfs"], 1)
                    ),
                    "latency ms": pl.get("latency", {}).get("measured_latency_ms"),
                    "RTF": pl.get("system", {}).get("rtf_total"),
                })
            if rows:
                import pandas as pd

                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                for label, d in (("ANC off", off[0]), ("ANC on", on[0])):
                    wav = d / "audio" / "output.wav"
                    if wav.is_file():
                        st.markdown(f"**{label} - output**")
                        st.audio(str(wav))

        last = st.session_state.get("last_live_session")
        if last:
            st.caption(f"Most recent live session: `{Path(last).name}`. "
                       f"Switch the ANC toggle and run again to produce the paired session.")

        with st.expander("All audio devices"):
            st.code(format_device_table(), language=None)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Audio device access failed: {exc}")
        st.caption("The offline tabs do not need audio hardware.")


# --------------------------------------------------------------------- system

with tab_system:
    st.header("System and performance")
    from anc_defence.utils.platform_info import collect_host_info

    host = collect_host_info()
    c1, c2, c3 = st.columns(3)
    cpu = host.get("cpu", {})
    c1.metric("CPU cores", f"{cpu.get('physical_cores')} / {cpu.get('logical_cores')} threads")
    c2.metric("RAM", f"{cpu.get('total_ram_gb')} GB")
    c3.metric("Jetson hardware", "yes" if host.get("is_jetson") else "no")

    st.subheader("Host")
    st.json({k: v for k, v in host.items() if k not in ("edge_readiness_note",)})
    st.info(host.get("edge_readiness_note", ""))

    st.subheader("Latency budget")
    from anc_defence.pipeline import Pipeline

    pipe = Pipeline(cfg, load_model=False)
    breakdown = {
        "input block (one hop)": 1000 * cfg.audio.hop_size / cfg.audio.sample_rate,
        "model frame": 10.0,
        "model STFT/ISTFT (n_fft - hop)": 10.0,
        "model lookahead (2 frames)": 20.0,
        "model algorithmic subtotal": 40.0,
        "chunk buffering (live path)": cfg.neural.streaming.chunk_s * 1000.0,
        "limiter look-ahead": cfg.normalise.limiter_lookahead_ms,
    }
    import pandas as pd

    st.dataframe(
        pd.DataFrame({"component": breakdown.keys(), "latency (ms)": breakdown.values()}),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "The 40 ms model figure is fixed by the architecture: one 10 ms frame, 10 ms for the "
        "STFT/ISTFT loop (n_fft - hop), and a 2-frame lookahead. The two lookaheads in the config "
        "are not additive - the code asserts conv_lookahead >= df_lookahead and both shift the same "
        "time axis. Chunk buffering applies to the live path only and is buffering, not streaming "
        "latency."
    )

    st.subheader("Measure the real-time factor now")
    bench_seconds = st.slider("Audio duration (s)", 2, 20, 5)
    if st.button("Run benchmark"):
        model = get_model(device, thread_count)
        rng = np.random.default_rng(0)
        t = np.arange(int(bench_seconds * SR)) / SR
        audio = (0.15 * np.sin(2 * np.pi * 180 * t) + 0.05 * rng.standard_normal(len(t))).astype(np.float32)
        model.enhance_array(audio)  # warm
        t0 = time.perf_counter()
        model.enhance_array(audio)
        dt = time.perf_counter() - t0
        rtf = dt / bench_seconds
        c1, c2, c3 = st.columns(3)
        c1.metric("RTF", f"{rtf:.4f}")
        c2.metric("Faster than real time by", f"{1 / rtf:.1f}x")
        c3.metric("Torch threads", str(model.info.num_threads))
        if rtf < 1.0:
            st.success(
                f"Real-time capable on this host with {model.info.num_threads} thread(s). "
                "This is portability evidence only - no claim is made about hardware that was not "
                "measured."
            )
        else:
            st.error("Slower than real time in this configuration.")
