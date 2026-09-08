"""Streamlit dashboard.

Launch with ``anc dashboard`` (or ``streamlit run anc_defence/ui/app.py``).

Five tabs:

``Overview``    the latest evaluation session: target verdicts, method comparison,
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

import io
import json
import time
from pathlib import Path
from typing import Any, Optional

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
    corpus = load_plain_corpus(cfg.plain)
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
            targets = payload.get("targets", {})
            checks = targets.get("checks", [])
            if checks:
                st.subheader("Mandated targets")
                st.caption(targets.get("scope", "").replace("<b>", "**").replace("</b>", "**"))
                cols = st.columns(len(checks))
                for col, c in zip(cols, checks):
                    value = c.get("value")
                    shown = "n/a" if value is None else f"{value:.3f}"
                    verdict = "PASS" if c.get("passed") else "FAIL"
                    col.metric(
                        f"{c['name']}  ({c['comparison']} {c['target']:g})",
                        shown,
                        verdict,
                        delta_color="normal" if c.get("passed") else "inverse",
                    )

            rows = payload.get("aggregate_by_method", [])
            if rows:
                st.subheader("Method comparison")
                import pandas as pd

                keep = ["method", "n", "pesq", "stoi", "estoi", "si_sdr", "snr_improvement_db",
                        "segmental_snr", "lsd", "speech_attenuation_db", "noise_reduction_db",
                        "output_level_dbfs", "rtf"]
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
    st.header("Live microphone")
    st.caption(
        "Records from the default input device, runs the pipeline, and compares before and after. "
        "Use headphones if you enable monitoring elsewhere; this tab does not play back live."
    )
    try:
        from anc_defence.audio.devices import format_device_table, list_devices

        devices = [d for d in list_devices() if d["max_input_channels"] > 0]
        names = [f"{d['index']}: {d['name']}" for d in devices]
        chosen = st.selectbox("Input device", names, index=0 if names else None)
        seconds = st.slider("Recording length (s)", 2, 20, 6)

        if st.button("Record and process", type="primary"):
            import sounddevice as sd

            index = int(chosen.split(":")[0])
            with st.spinner(f"Recording {seconds} s... speak now"):
                captured = sd.rec(
                    int(seconds * SR), samplerate=SR, channels=1, dtype="float32", device=index
                )
                sd.wait()
            mic = np.ascontiguousarray(captured[:, 0])

            from anc_defence.pipeline import Pipeline

            run_cfg = cfg.model_copy(deep=True)
            run_cfg.neural.device = device  # type: ignore[assignment]
            run_cfg.neural.num_threads = thread_count
            with st.spinner("Processing..."):
                t0 = time.perf_counter()
                pipe = Pipeline(run_cfg, model=get_model(device, thread_count))
                result = pipe.process(mic)
                elapsed = time.perf_counter() - t0

            st.success(f"processed {seconds} s in {elapsed:.2f} s (RTF {elapsed / seconds:.3f})")
            taps = result.taps(include_reference=False)
            from anc_defence.metrics.nonintrusive import estimate_snr_db

            c1, c2, c3 = st.columns(3)
            c1.metric("Input level", f"{active_speech_dbfs(mic, SR):.1f} dBFS")
            c2.metric("Output level", f"{active_speech_dbfs(result.output, SR):.1f} dBFS",
                      f"target {cfg.normalise.target_dbfs:g}")
            c3.metric("Estimated SNR change",
                      f"{estimate_snr_db(result.output, SR) - estimate_snr_db(mic, SR):+.1f} dB")

            cols = st.columns(3)
            with cols[0]:
                player("recorded input", mic)
            with cols[1]:
                player("pipeline output", result.output)
            with cols[2]:
                player("removed", result.removed(), "should contain no intelligible speech")

            st.pyplot(waveform_figure(taps), use_container_width=True)
            st.pyplot(spectrogram_figure(taps), use_container_width=True)
            st.markdown("**input to output**")
            st.pyplot(delta_figure(result.primary, result.output), use_container_width=True)
            if pipe.normaliser is not None:
                st.json(pipe.normaliser.summary())

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
