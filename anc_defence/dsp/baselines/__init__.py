"""Classical single- and dual-channel noise reduction baselines.

These exist to be beaten. The problem statement argues that traditional methods
assume stationarity and distort speech under dynamic conditions; the only way to
support that claim is to run them on the same test material with the same metrics.

* :mod:`spectral_subtraction` - magnitude spectral subtraction (single channel)
* :mod:`wiener` - decision-directed Wiener filtering (single channel)
* :mod:`lms` - classical fixed-step LMS using the reference channel

All three share the STFT helpers in :mod:`stft`.
"""

from .lms import BlockLms, lms_cancel, time_domain_lms
from .spectral_subtraction import spectral_subtraction
from .wiener import wiener_filter

__all__ = [
    "spectral_subtraction",
    "wiener_filter",
    "BlockLms",
    "lms_cancel",
    "time_domain_lms",
]
