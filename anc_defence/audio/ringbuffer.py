"""Single-producer / single-consumer ring buffer for the real-time path.

The audio callback is the producer and must never allocate, lock, log or touch the
filesystem. It only copies samples into this preallocated buffer and bumps a
counter. All DSP happens on a worker thread, which is the consumer.

Correctness relies on there being exactly one producer thread and one consumer
thread. Read and write indices are only ever advanced by their own side, so no
lock is needed for the data itself; a lock is used only for the wait/notify
handshake so the consumer can block instead of spinning.
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np


class RingBuffer:
    """Fixed-capacity float32 ring buffer.

    Attributes:
        overruns: writes that had to drop samples because the buffer was full.
        underruns: reads that could not be satisfied within the timeout.
        high_water: largest number of samples ever queued.
    """

    def __init__(self, capacity: int, channels: int = 1) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = int(capacity)
        self._channels = int(channels)
        self._buf = np.zeros((self._capacity, self._channels), dtype=np.float32)
        self._write = 0
        self._read = 0
        self._available = 0
        self._cv = threading.Condition(threading.Lock())
        self.overruns = 0
        self.dropped_samples = 0
        self.underruns = 0
        self.high_water = 0
        self._closed = False

    # ------------------------------------------------------------------ props
    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def channels(self) -> int:
        return self._channels

    def available(self) -> int:
        with self._cv:
            return self._available

    def free(self) -> int:
        with self._cv:
            return self._capacity - self._available

    # ----------------------------------------------------------------- writer
    def write(self, data: np.ndarray) -> int:
        """Copy ``data`` in. Called from the audio callback: no allocation here.

        Oldest samples are dropped if the buffer is full, and the overrun counter
        is incremented. Returns the number of samples written.
        """
        arr = data if data.ndim == 2 else data[:, None]
        n = arr.shape[0]
        if n == 0:
            return 0
        with self._cv:
            if n > self._capacity:
                arr = arr[-self._capacity :]
                n = self._capacity
            overflow = self._available + n - self._capacity
            if overflow > 0:
                # Drop the oldest data; the consumer has fallen behind.
                self._read = (self._read + overflow) % self._capacity
                self._available -= overflow
                self.overruns += 1
                self.dropped_samples += overflow
            first = min(n, self._capacity - self._write)
            self._buf[self._write : self._write + first] = arr[:first]
            if first < n:
                self._buf[: n - first] = arr[first:]
            self._write = (self._write + n) % self._capacity
            self._available += n
            if self._available > self.high_water:
                self.high_water = self._available
            self._cv.notify_all()
        return n

    # ----------------------------------------------------------------- reader
    def read(self, n: int, out: Optional[np.ndarray] = None, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """Read exactly ``n`` samples, blocking until available or timeout.

        Returns None on timeout (counted as an underrun) or after ``close()``.
        Pass a preallocated ``out`` array to avoid allocation in steady state.
        """
        if n <= 0:
            raise ValueError("n must be positive")
        if n > self._capacity:
            raise ValueError("cannot read more than the buffer capacity")
        with self._cv:
            if not self._cv.wait_for(lambda: self._available >= n or self._closed, timeout=timeout):
                self.underruns += 1
                return None
            if self._available < n:
                return None  # closed and drained
            dest = out if out is not None else np.empty((n, self._channels), dtype=np.float32)
            view = dest if dest.ndim == 2 else dest[:, None]
            first = min(n, self._capacity - self._read)
            view[:first] = self._buf[self._read : self._read + first]
            if first < n:
                view[first:] = self._buf[: n - first]
            self._read = (self._read + n) % self._capacity
            self._available -= n
            self._cv.notify_all()
        return dest

    def close(self) -> None:
        """Wake any blocked reader so shutdown does not hang."""
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    def reset(self) -> None:
        with self._cv:
            self._write = 0
            self._read = 0
            self._available = 0
            self._closed = False
            self._buf.fill(0.0)

    def stats(self) -> dict[str, int]:
        with self._cv:
            return {
                "capacity": self._capacity,
                "channels": self._channels,
                "available": self._available,
                "high_water": self.high_water,
                "overruns": self.overruns,
                "dropped_samples": self.dropped_samples,
                "underruns": self.underruns,
            }
