"""Gravitational wave data loader: load_gravitational_wave_data."""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def load_gravitational_wave_data(
    detector: str = "H1",
    start_time: int = 1126259446,
    end_time: int = 1126259478,
) -> Optional[np.ndarray]:
    """
    Load gravitational wave data using gwpy.

    Args:
        detector: Detector name (``H1``, ``L1``, ``V1``).
        start_time: GPS start time.
        end_time: GPS end time.

    Returns:
        NumPy array of gravitational wave strain data, or ``None`` if an
        error occurs.
    """
    try:
        from gwpy.timeseries import TimeSeries

        data = TimeSeries.fetch_open_data(detector, start_time, end_time, cache=True)
        logger.info(f"Successfully loaded gravitational wave data from {detector}")
        return data.value
    except Exception as e:
        logger.error(f"Error loading gravitational wave data: {e}", exc_info=True)
        return None
