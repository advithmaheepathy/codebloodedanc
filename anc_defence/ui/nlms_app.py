"""Standalone Streamlit page for the two-microphone NLMS noise canceller.

Deliberately separate from the main dashboard (anc_defence/ui/app.py): this is the
NLMS-only, two-microphone demo, matching the reference MATLAB implementation. It does not
load or run DeepFilterNet.

Launch:  anc nlms-dashboard      (or:  streamlit run anc_defence/ui/nlms_app.py)

Microphones:
  LEFT  channel  = primary   = speech + noise   (mic on the talker)
  RIGHT channel  = reference = noise            (mic toward the noise source)
  output = primary - NLMS estimate of the noise = enhanced speech
"""

from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path
from typing import Optional

# OpenSSL-backed hashlib builds (JetPack 5.1.x / Python 3.8 aarch64) reject
# `usedforsecurity=` on the hash constructors that Streamlit and Starlette call. Patch
# all of them to retry without the (FIPS-annotation-only) kwarg. Must run before Streamlit
# imports. See anc_defence/ui/app.py for the full explanation.
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anc_defence.config import Config, load_config  # noqa: E402

SR_PLACEHOLDER = 48000

st.set_page_config(
    page_title="Two-mic NLMS noise canceller - SIH 26052",
    page_icon="~",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def get_config() -> Config:
    path = REPO_ROOT / "configs" / "default.yaml"
    return load_config([path] if path.is_file() else [])


def wav_bytes(x: np.ndarray, sample_rate: int) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    safe = x / peak * 0.98 if peak > 1.0 else x
    sf.write(buf, np.asarray(safe, dtype=np.float32), sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def player(label: str, x: np.ndarray, sr: int, note: str = "") -> None:
    st.caption(f"**{label}**" + (f" - {note}" if note else ""))
    if x.size:
        st.audio(wav_bytes(x, sr), format="audio/wav")
    else:
        st.info("no audio")


def level_and_erle_plot(stats, sr: int, block: int):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    interval = block / sr
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4.4), sharex=True)

    tp = np.arange(len(stats.primary_dbfs)) * interval
    ax1.plot(tp, stats.primary_dbfs, lw=0.9, color="#1f77b4", label="primary (speech+noise)")
    ax1.plot(np.arange(len(stats.reference_dbfs)) * interval, stats.reference_dbfs,
             lw=0.9, color="#ff7f0e", label="reference (noise)")
    ax1.plot(np.arange(len(stats.output_dbfs)) * interval, stats.output_dbfs,
             lw=1.1, color="#2ca02c", label="output (enhanced)")
    ax1.set_ylabel("level (dBFS)", fontsize=8)
    ax1.legend(fontsize=6, loc="lower left")
    ax1.grid(alpha=0.3, lw=0.4)
    ax1.tick_params(labelsize=7)
    ax1.set_title("Live levels and noise reduction (ERLE) over the run", fontsize=9)

    te = np.arange(len(stats.erle_db)) * interval
    ax2.plot(te, stats.erle_db, lw=0.9, color="#9467bd")
    ax2.axhline(0.0, color="#999", lw=0.5)
    ax2.set_ylabel("ERLE (dB)", fontsize=8)
    ax2.set_xlabel("time (s)", fontsize=8)
    ax2.grid(alpha=0.3, lw=0.4)
    ax2.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def waveform_plot(sig: dict, sr: int):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(len(sig["primary"]), len(sig["output"]))
    if n == 0:
        return None
    t = np.arange(n) / sr
    fig, axes = plt.subplots(3, 1, figsize=(9, 5), sharex=True, sharey=True)
    for ax, key, colour, title in (
        (axes[0], "primary", "#1f77b4", "Primary  (LEFT mic: speech + noise)"),
        (axes[1], "reference", "#ff7f0e", "Reference  (RIGHT mic: noise)"),
        (axes[2], "output", "#2ca02c", "NLMS output  (enhanced speech = primary - estimated noise)"),
    ):
        ax.plot(t, sig[key][:n], lw=0.5, color=colour)
        ax.set_title(title, fontsize=8, loc="left")
        ax.grid(alpha=0.3, lw=0.4)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("time (s)", fontsize=8)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------- sidebar

cfg = get_config()
st.sidebar.title("Two-microphone NLMS")
st.sidebar.caption("SIH 26052 - adaptive noise cancellation, no AI model")
st.sidebar.code(
    "LEFT  mic -> primary   (speech + noise)\n"
    "RIGHT mic -> reference (noise only)\n"
    "output = primary - w*reference",
    language=None,
)
st.sidebar.markdown(
    "This is the classical adaptive-filter path: **NLMS only, two microphones, no neural "
    "network.** It is separate from the single-microphone DeepFilterNet dashboard."
)

st.header("Live two-microphone NLMS noise canceller")
st.caption(
    "Uses one stereo input device whose two channels are the two microphones. Put the "
    "primary mic on the talker and the reference mic toward the noise source. The filter "
    "estimates the noise from the reference and subtracts it from the primary."
)

# --------------------------------------------------------------- device select
try:
    from anc_defence.audio.devices import channel_independence, format_device_table, list_devices

    devices = [d for d in list_devices() if d["max_input_channels"] >= 2]
    if not devices:
        st.error(
            "No 2-channel input device found. The two-microphone NLMS demo needs a stereo "
            "input (both mics on one device as left/right). A mono microphone cannot provide "
            "a separate noise reference."
        )
        st.stop()

    names = [f"{d['index']}: {d['name']} ({d['max_input_channels']} ch)" for d in devices]
    col_a, col_b = st.columns(2)
    with col_a:
        chosen = st.selectbox("Stereo input device (2 mics)", names, index=0)
        seconds = st.slider("Session length (s)", 3, 60, 15)
    with col_b:
        swap = st.checkbox(
            "Swap channels (reference = LEFT)", value=False,
            help="By default LEFT is the primary (talker) and RIGHT is the noise reference. "
                 "Tick this if your mics are wired the other way round.",
        )
        monitor = st.checkbox(
            "Monitor output to headphones", value=False,
            help="Plays the enhanced output live. Use headphones - speakers feed back into "
                 "the mics.",
        )

    index = int(chosen.split(":")[0])

    st.divider()
    st.subheader("Step 1 - check the two channels are independent")
    st.caption(
        "Records 3 s and measures how different the two channels are. Speak into the PRIMARY "
        "mic only. A low correlation with the primary louder than the reference means two "
        "genuinely separate microphones - which is what NLMS needs. A correlation near 1.0 "
        "means the device is duplicating one mic to both channels (NLMS cannot work then)."
    )
    if st.button("Record 3 s and check channels"):
        import sounddevice as sd

        sr = cfg.audio.sample_rate
        with st.spinner("Recording 3 s - speak into the primary mic..."):
            rec = sd.rec(int(3 * sr), samplerate=sr, channels=2, dtype="float32", device=index)
            sd.wait()
        indep = channel_independence(rec)
        c1, c2, c3 = st.columns(3)
        c1.metric("L<->R correlation", f"{indep['correlation']:.3f}",
                  help="1.0 = identical (bad). Near 0 = independent (good).")
        c2.metric("Left level", f"{indep['left_dbfs']:.1f} dBFS")
        c3.metric("Right level", f"{indep['right_dbfs']:.1f} dBFS")
        if indep["correlation"] > 0.98:
            st.error(
                "Channels are essentially identical - the device is duplicating one mic to "
                "both channels. NLMS has no independent noise reference. Switch the receiver / "
                "Windows capture to true stereo (two separate transmitters on L and R)."
            )
        else:
            st.success(
                "Channels are independent - genuine two-microphone capture. Good to run NLMS."
            )

    st.divider()
    st.subheader("Step 2 - run the live NLMS session")
    if st.button("Run NLMS session", type="primary"):
        from anc_defence.modes.nlms_live import NlmsLiveEngine

        run_cfg = cfg.model_copy(deep=True)
        run_cfg.live.input_device = str(index)
        run_cfg.live.monitor = bool(monitor)
        sr = run_cfg.audio.sample_rate
        primary_ch, reference_ch = (1, 0) if swap else (0, 1)

        engine = NlmsLiveEngine(run_cfg, primary_channel=primary_ch, reference_channel=reference_ch)
        status = st.empty()
        status.info(f"Recording {seconds} s - talk into the primary mic now.")
        with st.spinner(f"Running NLMS for {seconds} s..."):
            try:
                engine.run(float(seconds))
            except Exception as exc:  # noqa: BLE001
                status.empty()
                st.error(f"NLMS session failed: {exc}")
                st.stop()
        status.empty()

        m = engine.metrics()
        sig = engine.signals()

        st.subheader("Result")
        k1, k2, k3, k4 = st.columns(4)
        erle = m.get("overall_erle_db")
        k1.metric("Noise reduction (ERLE)", "n/a" if erle is None else f"{erle:+.1f} dB",
                  help="10*log10(primary power / output power). How much energy the filter removed.")
        k2.metric("Channel correlation", f"{m['channel_correlation']:.3f}",
                  help="Sanity check that the two mics were independent during the run.")
        k3.metric("Primary level", f"{m['primary_dbfs']:.1f} dBFS")
        k4.metric("Reference level", f"{m['reference_dbfs']:.1f} dBFS")

        nlms = m.get("nlms", {})
        if nlms:
            st.caption(
                f"Adapting {100 * nlms.get('adapting_fraction', 0):.0f}% of the run · "
                f"final ERLE {nlms.get('erle_final_db', float('nan')):.1f} dB · "
                f"rollbacks {nlms.get('rollbacks', 0)} · "
                f"filter {run_cfg.nlms.filter_length} taps, mu {run_cfg.nlms.mu}"
            )

        st.markdown("### Listen")
        c1, c2, c3 = st.columns(3)
        with c1:
            player("primary (speech + noise)", sig["primary"], sr)
        with c2:
            player("reference (noise)", sig["reference"], sr)
        with c3:
            player("NLMS output (enhanced)", sig["output"], sr)

        st.markdown("### Waveforms")
        wf = waveform_plot(sig, sr)
        if wf is not None:
            st.pyplot(wf, use_container_width=True)

        st.markdown("### Levels and ERLE over the run")
        st.pyplot(level_and_erle_plot(engine.stats, sr, engine.block), use_container_width=True)

        health = m.get("stats", {})
        st.caption(
            f"Blocks in/out {health.get('blocks_in')}/{health.get('blocks_out')} · "
            f"output underruns {health.get('output_underruns')} · "
            f"status flags: {', '.join(health.get('callback_status_flags') or ['none'])}"
        )

    with st.expander("All input devices"):
        st.code(format_device_table(), language=None)

except Exception as exc:  # noqa: BLE001
    st.error(f"Audio device access failed: {exc}")
    st.caption("This demo needs a stereo microphone input device.")
