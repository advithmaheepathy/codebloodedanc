"""Hybrid adaptive-filter + neural noise cancellation for defence voice comms.

SIH 2026, problem statement 26052.

Scope note: this package implements the offline/benchmark path and two live
single-microphone modes on commodity hardware. Dataset generation is limited to a
small *evaluation* set; no model training is performed here. See README.md for the
full statement of what is designed versus what is executed.
"""

__version__ = "0.1.0"
