"""Neurophysiology data loader: load_eeg_data."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def load_eeg_data(sample_dataset: str = "sample") -> np.ndarray | None:
    """
    Load EEG data using MNE.

    Args:
        sample_dataset: MNE sample dataset name.

    Returns:
        NumPy array of EEG signal, or ``None`` if an error occurs.
    """
    try:
        import mne

        data_path = mne.datasets.sample.data_path()
        raw_fname = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"
        raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)

        eeg_data = raw.get_data(picks="eeg")[0]
        logger.info(f"Successfully loaded EEG data from {sample_dataset}")
        return eeg_data
    except Exception as e:
        logger.error(f"Error loading EEG data: {e}", exc_info=True)
        return None
