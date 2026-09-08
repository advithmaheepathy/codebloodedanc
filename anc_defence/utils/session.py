"""Session directories.

Every run writes into ``sessions/YYYY-MM-DD_HH-MM-SS_<mode>/``:

    config.yaml      fully resolved effective configuration
    session.log      full debug log for the run
    report.pdf       generated report
    metrics.json     machine-readable results
    metrics.csv      flat table of the same results
    audio/           WAV artefacts, one per pipeline tap point
    figures/         PNG figures used by the PDF
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..config import Config
from .logging import WarningCollector, get_logger, setup_logging
from .platform_info import collect_host_info

log = get_logger(__name__)


@dataclass
class Session:
    root: Path
    mode: str
    started_at: float
    config: Config
    host: dict[str, Any]
    warnings: WarningCollector
    notes: list[str] = field(default_factory=list)

    # ---------------------------------------------------------------- layout
    @property
    def audio_dir(self) -> Path:
        return self.root / "audio"

    @property
    def figures_dir(self) -> Path:
        return self.root / "figures"

    @property
    def log_path(self) -> Path:
        return self.root / "session.log"

    @property
    def config_path(self) -> Path:
        return self.root / "config.yaml"

    @property
    def pdf_path(self) -> Path:
        return self.root / "report.pdf"

    @property
    def json_path(self) -> Path:
        return self.root / "metrics.json"

    @property
    def csv_path(self) -> Path:
        return self.root / "metrics.csv"

    def audio_path(self, name: str) -> Path:
        return self.audio_dir / f"{name}.wav"

    def figure_path(self, name: str) -> Path:
        return self.figures_dir / f"{name}.png"

    # ----------------------------------------------------------------- misc
    @property
    def elapsed_s(self) -> float:
        return time.time() - self.started_at

    def note(self, message: str) -> None:
        """Record an operator-visible note that appears in the report."""
        self.notes.append(message)
        log.info("note: %s", message)


def create_session(cfg: Config, mode: str) -> Session:
    """Create the session directory, start file logging, dump the effective config."""
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{stamp}_{mode}" + (f"_{cfg.run.name}" if cfg.run.name else "")
    root = Path(cfg.run.session_root) / name
    root.mkdir(parents=True, exist_ok=True)
    (root / "audio").mkdir(exist_ok=True)
    (root / "figures").mkdir(exist_ok=True)

    setup_logging(cfg.run.log_level, root / "session.log")
    warnings = WarningCollector().install()

    session = Session(
        root=root,
        mode=mode,
        started_at=time.time(),
        config=cfg,
        host=collect_host_info(),
        warnings=warnings,
    )
    session.config_path.write_text(cfg.to_yaml(), encoding="utf-8")
    log.info("session directory: %s", root.resolve())
    return session


def find_session(path: Optional[Path], session_root: Path) -> Path:
    """Resolve a session directory, defaulting to the most recent one."""
    if path is not None:
        p = Path(path)
        if not p.is_dir():
            raise FileNotFoundError(f"session directory not found: {p}")
        return p
    candidates = sorted((d for d in Path(session_root).glob("*") if d.is_dir()), reverse=True)
    if not candidates:
        raise FileNotFoundError(f"no sessions found under {session_root}")
    return candidates[0]
