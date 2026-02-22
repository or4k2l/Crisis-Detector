"""
Crisis Detector – public API.

All symbols that were previously importable from the single-file
``crisis_detector`` module remain importable from this package:

    from crisis_detector import CrisisDetector
    from crisis_detector import load_finance_data, load_economic_data
    from crisis_detector import load_seismic_data
    from crisis_detector import load_gravitational_wave_data
    from crisis_detector import load_eeg_data
"""

from .detector import CrisisDetector
from .loaders import (
    load_economic_data,
    load_eeg_data,
    load_finance_data,
    load_gravitational_wave_data,
    load_seismic_data,
)

__all__ = [
    "CrisisDetector",
    "load_finance_data",
    "load_economic_data",
    "load_seismic_data",
    "load_gravitational_wave_data",
    "load_eeg_data",
]
