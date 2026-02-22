"""Shared utility functions for signal preprocessing."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def prepare_signal(
    data: np.ndarray | pd.Series | pd.DataFrame,
    timestamps: np.ndarray | None = None,
    column: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract and clean a 1-D signal from various input formats.

    Handles ``np.ndarray``, ``pd.Series`` and ``pd.DataFrame`` inputs,
    strips NaN values, and returns matching (signal, timestamps) arrays.

    Args:
        data: Input time-series data (1-D array, Series, or DataFrame).
        timestamps: Optional array of timestamps corresponding to *data*.
        column: Column name to use when *data* is a ``DataFrame``.

    Returns:
        Tuple of ``(signal_values, timestamps)`` with NaNs removed.
    """
    if isinstance(data, pd.DataFrame):
        if column is None:
            column = data.columns[0]
        signal_values: np.ndarray = np.asarray(data[column].values).flatten()
        if timestamps is None and isinstance(data.index, pd.DatetimeIndex):
            timestamps = data.index.values
    elif isinstance(data, pd.Series):
        signal_values = np.asarray(data.values).flatten()
        if timestamps is None and isinstance(data.index, pd.DatetimeIndex):
            timestamps = data.index.values
    else:
        signal_values = np.asarray(data).flatten()

    if timestamps is None:
        timestamps = np.arange(len(signal_values))

    # Remove NaN values
    valid_mask = ~np.isnan(signal_values)
    signal_values = signal_values[valid_mask]
    timestamps = timestamps[valid_mask]

    return signal_values, timestamps
