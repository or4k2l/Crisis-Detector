"""Seismic data loader: load_seismic_data."""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def load_seismic_data(
    network: str = "IU",
    station: str = "ANMO",
    location: str = "00",
    channel: str = "BHZ",
    starttime: str = "2023-01-01",
    endtime: str = "2023-01-02",
) -> Optional[np.ndarray]:
    """
    Load seismic data using ObsPy.

    Args:
        network: Seismic network code.
        station: Station code.
        location: Location code.
        channel: Channel code.
        starttime: Start time (UTC).
        endtime: End time (UTC).

    Returns:
        NumPy array of seismic signal, or ``None`` if an error occurs.
    """
    try:
        from obspy import UTCDateTime
        from obspy.clients.fdsn import Client

        client = Client("IRIS")
        st = client.get_waveforms(
            network,
            station,
            location,
            channel,
            UTCDateTime(starttime),
            UTCDateTime(endtime),
        )
        logger.info(f"Successfully loaded seismic data from {network}.{station}")
        return st[0].data
    except Exception as e:
        logger.error(f"Error loading seismic data: {e}", exc_info=True)
        return None
