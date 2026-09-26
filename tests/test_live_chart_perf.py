"""Tests for the live waveform panel's performance helpers.

``_tail_from_blocks`` and ``_downsample_for_plot`` (in ``anc_defence.ui.app``) exist to
keep each redraw tick of the Streamlit live-mic panel cheap and bounded by the display
window, not by how long the session has run - see the docstrings in ``app.py`` for why
(a Jetson's CPU is also running the single-threaded DSP worker at the same time, so
matplotlib re-rendering tens of thousands of raw points every ~0.4-0.7 s is expensive
there in a way it is not on a faster laptop).

``app.py`` does a fair amount of Streamlit-specific work at import time (page config,
etc.), so these tests import it directly - Streamlit tolerates being imported outside a
running app for pure function access, it just will not render anything.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

APP_PATH = Path(__file__).resolve().parents[1] / "anc_defence" / "ui" / "app.py"


@pytest.fixture(scope="module")
def app_module():
    spec = importlib.util.spec_from_file_location("anc_defence_ui_app_under_test", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestTailFromBlocks:
    def test_empty_blocks_returns_empty(self, app_module):
        out = app_module._tail_from_blocks([], max_samples=1000)
        assert out.size == 0

    def test_fewer_samples_than_window_returns_everything(self, app_module):
        blocks = [np.ones(10, dtype=np.float32) * i for i in range(3)]
        out = app_module._tail_from_blocks(blocks, max_samples=1000)
        expected = np.concatenate(blocks)
        np.testing.assert_array_equal(out, expected)

    def test_returns_only_the_trailing_window(self, app_module):
        # 5 blocks of 100 samples, each filled with its own index -> easy to check which
        # blocks survived the trim. Walking backwards, block 4 + block 3 + block 2
        # (300 samples) is the first point >= the 250-sample request, then the result
        # is trimmed to exactly the last 250 - so the boundary falls inside block 2.
        blocks = [np.full(100, i, dtype=np.float32) for i in range(5)]
        out = app_module._tail_from_blocks(blocks, max_samples=250)
        assert len(out) == 250
        np.testing.assert_array_equal(out[-100:], np.full(100, 4, dtype=np.float32))
        np.testing.assert_array_equal(out[50:150], np.full(100, 3, dtype=np.float32))
        np.testing.assert_array_equal(out[:50], np.full(50, 2, dtype=np.float32))

    def test_cost_bounded_by_window_not_history_length(self, app_module):
        # Many small blocks simulating a long session: the function must not need to
        # touch every block once enough recent ones satisfy the window.
        many_blocks = [np.zeros(10, dtype=np.float32) for _ in range(100_000)]
        many_blocks[-1] = np.ones(10, dtype=np.float32)  # marker at the very end
        out = app_module._tail_from_blocks(many_blocks, max_samples=20)
        assert len(out) == 20
        np.testing.assert_array_equal(out[-10:], np.ones(10, dtype=np.float32))


class TestDownsampleForPlot:
    def test_short_signal_is_returned_unchanged(self, app_module):
        x = np.arange(100, dtype=np.float32)
        idx, vals = app_module._downsample_for_plot(x, max_points=1500)
        np.testing.assert_array_equal(idx, np.arange(100))
        np.testing.assert_array_equal(vals, x)

    def test_long_signal_is_capped_in_point_count(self, app_module):
        x = np.random.default_rng(0).standard_normal(500_000).astype(np.float32)
        idx, vals = app_module._downsample_for_plot(x, max_points=1500)
        assert len(vals) <= 1500 * 2
        assert len(idx) == len(vals)

    def test_transient_peak_is_not_lost_between_bins(self, app_module):
        # A single large spike hidden inside an otherwise quiet signal: a naive stride
        # decimation (x[::step]) would very likely step over it. Min/max envelope
        # decimation must not.
        x = np.zeros(200_000, dtype=np.float32)
        spike_idx = 123_456
        x[spike_idx] = 0.97
        _, vals = app_module._downsample_for_plot(x, max_points=1000)
        assert np.max(vals) >= 0.97

    def test_monotonic_time_axis(self, app_module):
        x = np.random.default_rng(1).standard_normal(300_000).astype(np.float32)
        idx, _ = app_module._downsample_for_plot(x, max_points=800)
        assert np.all(np.diff(idx) >= 0)
