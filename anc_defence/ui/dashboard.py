"""Text dashboard for live runs.

Runs on the main thread and only *reads* counters that the audio callbacks and the
DSP worker write. It never touches the audio path, so a slow terminal cannot cause a
dropout. Falls back to plain periodic printing if ``rich`` is unavailable.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from ..utils.logging import get_logger

log = get_logger(__name__)


def _bar(value_db: float, floor: float = -60.0, width: int = 22) -> str:
    """Simple level meter from a dBFS value."""
    if not np.isfinite(value_db):
        return " " * width
    frac = max(0.0, min(1.0, (value_db - floor) / (0.0 - floor)))
    filled = int(round(frac * width))
    return "#" * filled + "-" * (width - filled)


class LiveDashboard:
    """Live status display for a :class:`anc_defence.modes.live.LiveEngine`."""

    def __init__(self, engine: Any, duration_s: float, refresh_hz: float = 8.0) -> None:
        self.engine = engine
        self.duration_s = duration_s
        self.refresh_s = 1.0 / max(1.0, refresh_hz)

    # ------------------------------------------------------------------ fields
    def _rows(self, elapsed: float) -> list[tuple[str, str]]:
        eng = self.engine
        stats = eng.stats
        in_db = stats.input_dbfs[-1] if stats.input_dbfs else float("-inf")
        out_db = stats.output_dbfs[-1] if stats.output_dbfs else float("-inf")
        ring = eng.in_ring.stats()

        nlms = getattr(eng.processor, "nlms", None)
        erle = "-"
        adapt = "-"
        vad = "-"
        if nlms is not None and getattr(nlms, "diagnostics", None) is not None:
            diag = nlms.diagnostics
            if diag.erle_db:
                recent = [v for v in diag.erle_db[-25:] if np.isfinite(v)]
                if recent:
                    erle = f"{float(np.mean(recent)):+.1f} dB"
            if diag.state:
                adapt = diag.state[-1]
            if diag.speech_flags:
                vad = "SPEECH" if diag.speech_flags[-1] else "noise only"

        timer = eng.timings.stages.get("pipeline")
        summary = timer.summary() if timer is not None else None
        rtf = f"{summary.rtf:.3f}" if summary and np.isfinite(summary.rtf) else "-"
        p95 = f"{summary.p95_ms:.2f} ms" if summary and summary.count else "-"

        cpu = "-"
        try:
            import psutil

            cpu = f"{psutil.cpu_percent(None):.0f}%"
        except Exception:
            pass

        remaining = max(0.0, self.duration_s - elapsed)
        return [
            ("elapsed / remaining", f"{elapsed:5.1f} s / {remaining:5.1f} s"),
            ("input level", f"{in_db:6.1f} dBFS  [{_bar(in_db)}]"),
            ("output level", f"{out_db:6.1f} dBFS  [{_bar(out_db)}]"),
            ("ERLE (recent)", erle),
            ("VAD", vad),
            ("adaptation", adapt),
            ("RTF / p95 per block", f"{rtf} / {p95}"),
            ("blocks in / out", f"{stats.blocks_in} / {stats.blocks_out}"),
            ("xruns / underruns", f"{ring.get('overruns', 0)} / {stats.output_underruns}"),
            (
                "ring high-water",
                f"{ring.get('high_water', 0)} / {ring.get('capacity', 0)} samples",
            ),
            ("CPU", cpu),
        ]

    # --------------------------------------------------------------------- run
    def run(self) -> None:
        try:
            self._run_rich()
        except ImportError:
            self._run_plain()

    def _run_rich(self) -> None:
        from rich.console import Console
        from rich.live import Live
        from rich.panel import Panel
        from rich.table import Table

        console = Console()
        start = time.perf_counter()

        def render() -> Panel:
            table = Table.grid(padding=(0, 2))
            table.add_column(justify="right", style="bold cyan", no_wrap=True)
            table.add_column(justify="left")
            for key, value in self._rows(time.perf_counter() - start):
                table.add_row(key, value)
            return Panel(
                table,
                title=f"[bold]{self.engine.processor.name}[/bold]  (Ctrl-C to stop early)",
                subtitle="headphones recommended: speaker monitoring will feed back",
                border_style="cyan",
            )

        with Live(render(), console=console, refresh_per_second=6, transient=False) as live:
            try:
                while time.perf_counter() - start < self.duration_s:
                    time.sleep(self.refresh_s)
                    live.update(render())
                live.update(render())
            except KeyboardInterrupt:
                live.update(render())
                raise

    def _run_plain(self) -> None:  # pragma: no cover - fallback path
        start = time.perf_counter()
        next_print = 0.0
        while True:
            elapsed = time.perf_counter() - start
            if elapsed >= self.duration_s:
                break
            if elapsed >= next_print:
                parts = [f"{k}={v}" for k, v in self._rows(elapsed)[:6]]
                print("  ".join(parts), flush=True)
                next_print = elapsed + 1.0
            time.sleep(0.1)


class ProgressReporter:
    """Minimal progress line for long offline runs."""

    def __init__(self, total: int, label: str = "processing") -> None:
        self.total = total
        self.label = label
        self._last = 0.0

    def __call__(self, index: int, total: Optional[int] = None, name: str = "") -> None:
        total = total or self.total
        now = time.perf_counter()
        if index == 1 or index == total or now - self._last > 2.0:
            self._last = now
            pct = 100.0 * index / max(1, total)
            log.info("%s %d/%d (%.0f%%) %s", self.label, index, total, pct, name)
