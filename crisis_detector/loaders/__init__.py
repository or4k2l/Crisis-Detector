"""Loaders sub-package – re-exports all domain-specific loader functions."""

from .finance import load_economic_data, load_finance_data
from .gravitational import load_gravitational_wave_data
from .neuro import load_eeg_data
from .seismic import load_seismic_data

__all__ = [
    "load_finance_data",
    "load_economic_data",
    "load_seismic_data",
    "load_gravitational_wave_data",
    "load_eeg_data",
]
