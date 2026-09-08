"""Structured logging.

One log file per session plus a console stream. Real-time audio callbacks must
never log: they run on a driver thread with a hard deadline, and the logging module
allocates and takes locks. Callbacks increment counters instead, and worker threads
report those counters.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_CONSOLE_FMT = "%(asctime)s %(levelname)-7s %(name)-28s %(message)s"
_FILE_FMT = "%(asctime)s %(levelname)-7s %(name)s %(filename)s:%(lineno)d %(message)s"
_DATE_FMT = "%H:%M:%S"

_configured = False


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> logging.Logger:
    """Configure the root logger. Idempotent within a process."""
    global _configured
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    if not _configured:
        console = logging.StreamHandler(stream=sys.stderr)
        console.setLevel(getattr(logging, level.upper(), logging.INFO))
        console.setFormatter(logging.Formatter(_CONSOLE_FMT, datefmt=_DATE_FMT))
        console.set_name("console")
        root.addHandler(console)
        # Third-party chatter that is not useful at INFO.
        for noisy in ("matplotlib", "PIL", "h5py", "numba"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        _configured = True
    else:
        for handler in root.handlers:
            if handler.get_name() == "console":
                handler.setLevel(getattr(logging, level.upper(), logging.INFO))

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        for handler in list(root.handlers):
            if handler.get_name() == "session-file":
                root.removeHandler(handler)
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_FILE_FMT))
        file_handler.set_name("session-file")
        root.addHandler(file_handler)

    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class WarningCollector(logging.Handler):
    """Collects WARNING+ records so the report can list everything that went wrong."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[dict[str, object]] = []

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - trivial
        self.records.append(
            {
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "time": record.created,
            }
        )

    def install(self) -> "WarningCollector":
        logging.getLogger().addHandler(self)
        return self

    def messages(self) -> list[str]:
        return [f"[{r['level']}] {r['logger']}: {r['message']}" for r in self.records]
