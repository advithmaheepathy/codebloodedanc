"""Isolate the cost of one live-waveform-panel redraw tick, with no Streamlit/audio.

Run this directly on the target machine (e.g. the Jetson) to see how much of the
live panel's lag is matplotlib/PNG-encode cost versus something else (network,
Streamlit's own overhead, contention with the audio worker thread). Compare the
printed numbers against ``refresh_s`` (0.7 s default) in
``anc_defence.ui.app.run_live_with_live_chart`` - if a single figure+encode already
costs a large fraction of that budget, the fix is a cheaper figure (or a plain line
chart instead of matplotlib); if it's a small fraction, the lag is coming from
somewhere else in the loop and this script will have shown that.

Usage:
    python scripts/bench_live_chart.py
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    from anc_defence.ui.app import live_waveform_figure

    sr = 48000
    window_s = 12.0
    n_ticks = 15

    rng = np.random.default_rng(0)
    # Simulate a session that has been running a while: a big, slowly growing signal,
    # same shape the real polling loop would hand to live_waveform_figure after
    # _tail_from_blocks has already trimmed it to the window.
    window_n = int(window_s * sr)
    inp = rng.standard_normal(window_n).astype(np.float32) * 0.1
    outp = rng.standard_normal(window_n).astype(np.float32) * 0.05

    print(f"window: {window_s:.0f} s ({window_n} samples/channel), {n_ticks} ticks\n")

    figure_times = []
    encode_times = []
    for i in range(n_ticks):
        t0 = time.perf_counter()
        fig = live_waveform_figure(inp, outp, sr, window_s=window_s)
        t1 = time.perf_counter()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        t2 = time.perf_counter()
        import matplotlib.pyplot as plt

        plt.close(fig)
        figure_times.append((t1 - t0) * 1000.0)
        encode_times.append((t2 - t1) * 1000.0)

    def stats(label: str, values: list[float]) -> None:
        arr = np.asarray(values)
        print(
            f"{label:32s} mean {arr.mean():7.1f} ms   p95 {np.percentile(arr, 95):7.1f} ms   "
            f"max {arr.max():7.1f} ms"
        )

    stats("figure build + render (Agg)", figure_times)
    stats("PNG encode", encode_times)
    total = np.asarray(figure_times) + np.asarray(encode_times)
    stats("total per tick", list(total))
    print(f"\nrefresh_s budget: 700 ms  ->  headroom per tick: {700 - total.mean():.0f} ms (mean)")


if __name__ == "__main__":
    main()
