"""Tests for the live-mode demo output delay queue.

The delay queue is a pure demo aid (see ``anc_defence.modes.live._DemoDelayQueue``):
it must never alter samples, only when they are released, and it must be a true no-op
at ``delay_s == 0`` so the real minimum-latency path is unaffected by the feature's
existence.
"""

from __future__ import annotations

import numpy as np
import pytest

from anc_defence.modes.live import _DemoDelayQueue

SR = 48000


def test_zero_delay_is_immediate_passthrough():
    q = _DemoDelayQueue(delay_s=0.0, sample_rate=SR)
    block = np.arange(480, dtype=np.float32)
    q.push(block, now=0.0)
    ready = q.pop_ready(now=0.0)
    assert len(ready) == 1
    np.testing.assert_array_equal(ready[0], block)
    assert q.pending_seconds() == 0.0


def test_positive_delay_holds_until_elapsed():
    q = _DemoDelayQueue(delay_s=2.0, sample_rate=SR)
    block = np.ones(480, dtype=np.float32)
    q.push(block, now=0.0)

    # Not enough time has passed yet: nothing is released.
    assert q.pop_ready(now=1.0) == []
    assert q.pending_seconds() == pytest.approx(480 / SR)

    # Exactly at the hold time (and beyond), it is released.
    ready = q.pop_ready(now=2.0)
    assert len(ready) == 1
    np.testing.assert_array_equal(ready[0], block)
    assert q.pending_seconds() == 0.0


def test_samples_are_never_modified():
    q = _DemoDelayQueue(delay_s=3.0, sample_rate=SR)
    rng = np.random.default_rng(0)
    block = rng.standard_normal(480).astype(np.float32)
    original = block.copy()
    q.push(block, now=0.0)
    ready = q.pop_ready(now=10.0)
    np.testing.assert_array_equal(ready[0], original)


def test_multiple_blocks_release_in_order():
    q = _DemoDelayQueue(delay_s=1.0, sample_rate=SR)
    blocks = [np.full(10, i, dtype=np.float32) for i in range(5)]
    for i, b in enumerate(blocks):
        q.push(b, now=float(i) * 0.1)  # pushed at t=0.0, 0.1, ..., 0.4

    # At t=1.05, only blocks pushed at t<=0.05 have crossed the 1 s hold.
    ready = q.pop_ready(now=1.05)
    assert len(ready) == 1
    np.testing.assert_array_equal(ready[0], blocks[0])

    # Later, the rest arrive in the same order they were pushed.
    ready = q.pop_ready(now=2.0)
    assert len(ready) == 4
    for got, expected in zip(ready, blocks[1:]):
        np.testing.assert_array_equal(got, expected)


def test_flush_releases_everything_regardless_of_hold_time():
    q = _DemoDelayQueue(delay_s=10.0, sample_rate=SR)
    block = np.ones(20, dtype=np.float32)
    q.push(block, now=0.0)
    assert q.pop_ready(now=0.1) == []  # far from the 10 s hold
    flushed = q.flush()
    assert len(flushed) == 1
    np.testing.assert_array_equal(flushed[0], block)
    assert q.pending_seconds() == 0.0


def test_empty_block_is_ignored():
    q = _DemoDelayQueue(delay_s=1.0, sample_rate=SR)
    q.push(np.zeros(0, dtype=np.float32), now=0.0)
    assert q.pending_seconds() == 0.0
    assert q.pop_ready(now=5.0) == []


@pytest.mark.parametrize("delay_s", [-1.0, -0.001])
def test_negative_delay_is_clamped_to_zero(delay_s):
    q = _DemoDelayQueue(delay_s=delay_s, sample_rate=SR)
    assert q.delay_s == 0.0
    block = np.ones(5, dtype=np.float32)
    q.push(block, now=0.0)
    ready = q.pop_ready(now=0.0)
    assert len(ready) == 1


def test_config_default_and_bounds():
    from anc_defence.config import LiveCfg

    assert LiveCfg().demo_delay_s == 0.0
    LiveCfg(demo_delay_s=10.0)  # upper bound ok
    LiveCfg(demo_delay_s=0.0)  # lower bound ok
    with pytest.raises(Exception):
        LiveCfg(demo_delay_s=10.1)
    with pytest.raises(Exception):
        LiveCfg(demo_delay_s=-0.1)
