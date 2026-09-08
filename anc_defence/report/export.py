"""Report data container plus JSON and CSV export.

Every run produces the same three artefacts from one :class:`ReportData` object: a
PDF, a JSON file with all the numbers, and a CSV table. Modes fill the container;
they never format anything themselves.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from ..metrics.categories import TargetCheck
from ..utils.logging import get_logger
from ..utils.session import Session

log = get_logger(__name__)


@dataclass
class ReportTable:
    """One table in the report. ``rows`` includes the header row."""

    name: str
    rows: list[list[str]]
    caption: str = ""
    col_widths: Optional[Sequence[float]] = None
    font_size: float = 6.5


@dataclass
class ReportFigure:
    path: Path
    caption: str = ""
    width_cm: float = 17.0


@dataclass
class ReportData:
    """Everything the report needs, filled in by whichever mode ran."""

    session: Session
    title: str
    mode: str
    subtitle: str = ""
    metadata_rows: list[tuple[str, str]] = field(default_factory=list)
    scope_notes: list[str] = field(default_factory=list)
    target_checks: list[TargetCheck] = field(default_factory=list)
    target_scope: str = ""
    tables: list[ReportTable] = field(default_factory=list)
    figures: list[ReportFigure] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    interpretation: list[tuple[str, str]] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)
    csv_rows: list[dict[str, Any]] = field(default_factory=list)

    def add_table(self, name: str, rows: list[list[str]], caption: str = "", **kwargs: Any) -> None:
        if rows:
            self.tables.append(ReportTable(name=name, rows=rows, caption=caption, **kwargs))

    def add_figure(self, path: Optional[Path], caption: str = "", width_cm: float = 17.0) -> None:
        if path is not None and Path(path).is_file():
            self.figures.append(ReportFigure(path=Path(path), caption=caption, width_cm=width_cm))


def _jsonable(obj: Any) -> Any:
    """Make numpy and Path values JSON-serialisable, and NaN explicit as null."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if not np.isfinite(f) else round(f, 6)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


def write_json_report(data: ReportData) -> Path:
    """Write ``metrics.json`` with the full machine-readable record of the run."""
    payload = {
        "title": data.title,
        "mode": data.mode,
        "session_dir": str(data.session.root),
        "started_at": data.session.started_at,
        "elapsed_s": round(data.session.elapsed_s, 3),
        "host": data.session.host,
        "config": data.session.config.to_plain(),
        "scope_notes": data.scope_notes,
        "targets": {
            "scope": data.target_scope,
            "checks": [c.as_dict() for c in data.target_checks],
            "all_passed": all(c.passed for c in data.target_checks) if data.target_checks else None,
        },
        "warnings": data.warnings + data.session.warnings.messages(),
        "notes": data.session.notes,
        **data.payload,
    }
    path = data.session.json_path
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    log.info("wrote %s", path)
    return path


def write_metrics_csv(data: ReportData) -> Optional[Path]:
    """Write ``metrics.csv``: one row per example/method/tap measurement."""
    if not data.csv_rows:
        return None
    from ..metrics.categories import write_csv

    path = write_csv(data.session.csv_path, data.csv_rows)
    log.info("wrote %s", path)
    return path
