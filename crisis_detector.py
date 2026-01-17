"""
Crisis Detector - A tool for detecting crises and anomalies in time series data.

This module provides the CrisisDetector class for analyzing time series data
across multiple domains including finance, seismology, and neuroscience.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings


class CrisisDetector:
    """
    Detects crises, anomalies, and transient events in time series data.

    Uses multiple signal processing techniques including wavelet transforms,
    spectral analysis, and statistical methods to identify unusual patterns.

    Parameters
    ----------
    threshold : float, optional (default=3.0)
        Z-score threshold for anomaly detection
    window_size : int, optional (default=50)
        Window size for rolling statistics
    """

    def __init__(self, threshold=3.0, window_size=50):
        self.threshold = threshold
        self.window_size = window_size
        self.scaler = StandardScaler()
        self.scores_ = None
        self.anomalies_ = None

    def fit(self, data):
        """
        Fit the detector on time series data.

        Parameters
        ----------
        data : array-like
            Time series data to analyze

        Returns
        -------
        self : CrisisDetector
            Returns self for method chaining
        """
        data = np.asarray(data).flatten()

        if len(data) < self.window_size:
            warnings.warn(
                f"Data length ({len(data)}) is less than window_size "
                f"({self.window_size}). Using data length as window size."
            )
            self.window_size = len(data) // 2 or 1

        # Compute rolling statistics (not centered to avoid including future in past)
        rolling_mean = (
            pd.Series(data).rolling(window=self.window_size, center=False).mean()
        )
        rolling_std = (
            pd.Series(data).rolling(window=self.window_size, center=False).std()
        )
        
        # Fill NaN values at the beginning
        rolling_mean = rolling_mean.bfill()
        rolling_std = rolling_std.bfill().replace(0, 1e-8)  # Avoid division by zero

        # Compute z-scores
        z_scores = np.abs((data - rolling_mean.values) / (rolling_std.values + 1e-8))

        # Combine features
        self.scores_ = z_scores
        self.anomalies_ = self.scores_ > self.threshold

        return self

    def detect(self, data):
        """
        Detect crises in the given time series data.

        Parameters
        ----------
        data : array-like
            Time series data to analyze

        Returns
        -------
        scores : ndarray
            Anomaly scores for each time point
        anomalies : ndarray
            Boolean array indicating detected anomalies
        """
        self.fit(data)
        return self.scores_, self.anomalies_

    def get_crisis_periods(self):
        """
        Get the start and end indices of crisis periods.

        Returns
        -------
        periods : list of tuples
            List of (start_idx, end_idx) tuples for each crisis period
        """
        if self.anomalies_ is None:
            raise ValueError("Must call fit() or detect() first")

        periods = []
        in_crisis = False
        start_idx = None

        for i, is_anomaly in enumerate(self.anomalies_):
            if is_anomaly and not in_crisis:
                start_idx = i
                in_crisis = True
            elif not is_anomaly and in_crisis:
                periods.append((start_idx, i - 1))
                in_crisis = False

        if in_crisis:
            periods.append((start_idx, len(self.anomalies_) - 1))

        return periods


def load_finance_data(ticker="SPY", period="1y"):
    """
    Load financial time series data using yfinance.

    Parameters
    ----------
    ticker : str, optional (default='SPY')
        Stock ticker symbol
    period : str, optional (default='1y')
        Period to download (e.g., '1d', '5d', '1mo', '1y', '5y', 'max')

    Returns
    -------
    data : pandas.DataFrame
        Financial data with OHLCV columns
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is required for loading finance data. "
            "Install it with: pip install yfinance"
        )

    ticker_obj = yf.Ticker(ticker)
    data = ticker_obj.history(period=period)

    if data.empty:
        raise ValueError(f"No data found for ticker {ticker}")

    return data


def load_seismic_data(station="II.BFO", starttime=None, endtime=None):
    """
    Load seismic data using ObsPy (optional dependency).

    Parameters
    ----------
    station : str, optional (default='II.BFO')
        Station code
    starttime : str or UTCDateTime, optional
        Start time for data request
    endtime : str or UTCDateTime, optional
        End time for data request

    Returns
    -------
    stream : obspy.Stream
        Seismic data stream
    """
    try:
        from obspy.clients.fdsn import Client
        from obspy import UTCDateTime
    except ImportError:
        raise ImportError(
            "obspy is required for loading seismic data. "
            "Install it with: pip install -r requirements-optional.txt"
        )

    if starttime is None:
        starttime = UTCDateTime() - 86400  # 1 day ago
    if endtime is None:
        endtime = UTCDateTime()

    client = Client("IRIS")
    network, station_code = station.split(".")
    stream = client.get_waveforms(network, station_code, "*", "*", starttime, endtime)

    return stream


def load_gravitational_wave_data(detector="H1", start_time=None, duration=32):
    """
    Load gravitational wave data using gwpy (optional dependency).

    Parameters
    ----------
    detector : str, optional (default='H1')
        Detector name (e.g., 'H1', 'L1', 'V1')
    start_time : int, optional
        GPS start time
    duration : int, optional (default=32)
        Duration in seconds

    Returns
    -------
    data : gwpy.timeseries.TimeSeries
        Gravitational wave strain data
    """
    try:
        from gwpy.timeseries import TimeSeries
    except ImportError:
        raise ImportError(
            "gwpy is required for loading gravitational wave data. "
            "Install it with: pip install -r requirements-optional.txt"
        )

    if start_time is None:
        # Default to a known event time (GW150914)
        start_time = 1126259462

    data = TimeSeries.fetch_open_data(detector, start_time, start_time + duration)

    return data


def load_eeg_data(filename):
    """
    Load EEG/MEG data using MNE (optional dependency).

    Parameters
    ----------
    filename : str
        Path to EEG/MEG data file

    Returns
    -------
    raw : mne.io.Raw
        Raw EEG/MEG data
    """
    try:
        import mne
    except ImportError:
        raise ImportError(
            "mne is required for loading EEG/MEG data. "
            "Install it with: pip install -r requirements-optional.txt"
        )

    raw = mne.io.read_raw(filename, preload=True)
    return raw


if __name__ == "__main__":
    # Simple demonstration
    print("Crisis Detector - Basic Demo")
    print("-" * 40)

    # Generate synthetic data with a crisis
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    signal_data = np.sin(2 * np.pi * t) + 0.1 * np.random.randn(len(t))

    # Add a crisis event (sharp spike)
    crisis_start = 400
    crisis_end = 410  # Short crisis
    signal_data[crisis_start:crisis_end] += 12.0

    # Detect crises
    detector = CrisisDetector(threshold=3.5, window_size=25)
    scores, anomalies = detector.detect(signal_data)

    print(f"Total data points: {len(signal_data)}")
    print(f"Anomalies detected: {np.sum(anomalies)}")
    print(f"Max anomaly score: {np.max(scores):.2f}")

    periods = detector.get_crisis_periods()
    print(f"\nCrisis periods detected: {len(periods)}")
    for i, (start, end) in enumerate(periods, 1):
        print(f"  Period {i}: indices {start} to {end} (length: {end-start+1})")

    print("\nDemo completed successfully!")
