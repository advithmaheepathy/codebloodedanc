"""Report generation: figures, PDF, and machine-readable exports."""

from .export import write_json_report, write_metrics_csv
from .pdf import build_pdf_report

__all__ = ["build_pdf_report", "write_json_report", "write_metrics_csv"]
