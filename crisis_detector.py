"""
Crisis Detector - A Python module for detecting anomalies and crisis events in time series data.

This module provides tools for analyzing various types of time series data including
financial markets, seismic data, and physiological signals to identify crisis events
and anomalies using statistical and machine learning methods.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, Union, Dict


class CrisisDetector:
    """
    A class for detecting crisis events and anomalies in time series data.

    This detector uses multiple statistical and machine learning methods to identify
    anomalous patterns that may indicate crisis events in various domains including
    financial markets, seismic activity, and physiological data.

    Parameters
    ----------
    window_size : int, default=20
        Size of the rolling window for statistical calculations
    threshold : float, default=3.0
        Z-score threshold for anomaly detection
    min_crisis_duration : int, default=1
        Minimum duration (in time steps) for a valid crisis event
    """

    def __init__(
        self,
        window_size: int = 20,
        threshold: float = 3.0,
        min_crisis_duration: int = 1,
    ):
        self.window_size = window_size
        self.threshold = threshold
        self.min_crisis_duration = min_crisis_duration
        self.scaler = StandardScaler()
        self._is_fitted = False

    def fit(self, data: Union[np.ndarray, pd.Series, pd.DataFrame]) -> "CrisisDetector":
        """
        Fit the detector to training data.

        Parameters
        ----------
        data : array-like
            Training data for fitting the detector

        Returns
        -------
        self : CrisisDetector
            Fitted detector instance
        """
        data_array = self._validate_input(data)

        # Fit the scaler
        if data_array.ndim == 1:
            data_array = data_array.reshape(-1, 1)
        self.scaler.fit(data_array)
        self._is_fitted = True

        return self

    def detect(
        self,
        data: Union[np.ndarray, pd.Series, pd.DataFrame],
        return_scores: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Detect crisis events in the input data.

        Parameters
        ----------
        data : array-like
            Time series data to analyze
        return_scores : bool, default=True
            If True, return both crisis flags and anomaly scores

        Returns
        -------
        crisis_flags : np.ndarray
            Binary array indicating crisis events (1) or normal periods (0)
        scores : np.ndarray, optional
            Anomaly scores for each time point (returned if return_scores=True)
        """
        data_array = self._validate_input(data)

        # Calculate anomaly scores using multiple methods
        z_scores = self._calculate_z_scores(data_array)
        volatility_scores = self._calculate_volatility(data_array)

        # Combine scores
        combined_scores = (np.abs(z_scores) + volatility_scores) / 2.0

        # Detect crisis events
        crisis_flags = self._identify_crisis_events(combined_scores)

        if return_scores:
            return crisis_flags, combined_scores
        return crisis_flags

    def _validate_input(
        self, data: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> np.ndarray:
        """Validate and convert input data to numpy array."""
        if isinstance(data, pd.DataFrame):
            if data.shape[1] == 1:
                data_array = data.iloc[:, 0].values
            else:
                raise ValueError("DataFrame must have exactly one column")
        elif isinstance(data, pd.Series):
            data_array = data.values
        elif isinstance(data, np.ndarray):
            data_array = data.flatten()
        else:
            try:
                data_array = np.array(data).flatten()
            except Exception as e:
                raise ValueError(f"Could not convert input to numpy array: {e}")

        if len(data_array) < self.window_size:
            raise ValueError(
                f"Data length ({len(data_array)}) must be at least "
                f"window_size ({self.window_size})"
            )

        return data_array

    def _calculate_z_scores(self, data: np.ndarray) -> np.ndarray:
        """Calculate rolling z-scores for anomaly detection."""
        z_scores = np.zeros_like(data)

        for i in range(len(data)):
            if i < self.window_size:
                # Use available data for initial window
                window_data = data[: i + 1]
            else:
                window_data = data[i - self.window_size : i]

            if len(window_data) > 1:
                mean = np.mean(window_data)
                std = np.std(window_data)
                if std > 0:
                    z_scores[i] = (data[i] - mean) / std
                else:
                    z_scores[i] = 0
            else:
                z_scores[i] = 0

        return z_scores

    def _calculate_volatility(self, data: np.ndarray) -> np.ndarray:
        """Calculate rolling volatility scores."""
        volatility = np.zeros_like(data)

        # Calculate returns
        returns = np.diff(data, prepend=data[0])

        for i in range(len(data)):
            if i < self.window_size:
                window_returns = returns[: i + 1]
            else:
                window_returns = returns[i - self.window_size : i]

            if len(window_returns) > 1:
                volatility[i] = np.std(window_returns)
            else:
                volatility[i] = 0

        # Normalize volatility
        if np.max(volatility) > 0:
            volatility = volatility / np.max(volatility)

        return volatility

    def _identify_crisis_events(self, scores: np.ndarray) -> np.ndarray:
        """Identify crisis events based on anomaly scores."""
        crisis_flags = np.zeros_like(scores, dtype=int)
        crisis_flags[scores > self.threshold] = 1

        # Apply minimum duration filter
        if self.min_crisis_duration > 1:
            crisis_flags = self._apply_duration_filter(crisis_flags)

        return crisis_flags

    def _apply_duration_filter(self, flags: np.ndarray) -> np.ndarray:
        """Filter out crisis events shorter than minimum duration."""
        filtered = np.zeros_like(flags)
        in_crisis = False
        crisis_start = 0

        for i in range(len(flags)):
            if flags[i] == 1 and not in_crisis:
                in_crisis = True
                crisis_start = i
            elif flags[i] == 0 and in_crisis:
                crisis_duration = i - crisis_start
                if crisis_duration >= self.min_crisis_duration:
                    filtered[crisis_start:i] = 1
                in_crisis = False

        # Handle crisis at end of data
        if in_crisis:
            crisis_duration = len(flags) - crisis_start
            if crisis_duration >= self.min_crisis_duration:
                filtered[crisis_start:] = 1

        return filtered


def load_finance_data(
    ticker: str = "^GSPC",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = "1y",
) -> pd.DataFrame:
    """
    Load financial data from Yahoo Finance.

    Parameters
    ----------
    ticker : str, default="^GSPC"
        Stock ticker symbol (default is S&P 500)
    start_date : str, optional
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format
    period : str, default="1y"
        Time period to fetch if dates not specified

    Returns
    -------
    data : pd.DataFrame
        DataFrame with financial data including OHLCV columns
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is required for loading financial data. "
            "Install it with: pip install yfinance"
        )

    if start_date and end_date:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    else:
        data = yf.download(ticker, period=period, progress=False)

    if data.empty:
        raise ValueError(f"No data found for ticker {ticker}")

    return data


def load_synthetic_data(
    n_samples: int = 1000,
    noise_level: float = 0.1,
    n_crises: int = 3,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic time series data with crisis events.

    Parameters
    ----------
    n_samples : int, default=1000
        Number of samples to generate
    noise_level : float, default=0.1
        Standard deviation of Gaussian noise
    n_crises : int, default=3
        Number of crisis events to inject
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    data : np.ndarray
        Generated time series data
    true_labels : np.ndarray
        Ground truth labels (1 for crisis, 0 for normal)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate base signal
    t = np.linspace(0, 10, n_samples)
    data = np.sin(t) + noise_level * np.random.randn(n_samples)

    # Add crisis events (spikes)
    true_labels = np.zeros(n_samples, dtype=int)
    crisis_positions = np.random.choice(
        range(100, n_samples - 100), size=n_crises, replace=False
    )

    for pos in crisis_positions:
        spike_duration = np.random.randint(5, 20)
        spike_magnitude = np.random.uniform(3, 6)
        for i in range(spike_duration):
            if pos + i < n_samples:
                data[pos + i] += spike_magnitude
                true_labels[pos + i] = 1

    return data, true_labels


def analyze_crisis(
    data: Union[np.ndarray, pd.Series, pd.DataFrame],
    detector: Optional[CrisisDetector] = None,
    **detector_kwargs,
) -> Dict[str, Union[np.ndarray, float, int]]:
    """
    Perform comprehensive crisis analysis on time series data.

    Parameters
    ----------
    data : array-like
        Time series data to analyze
    detector : CrisisDetector, optional
        Pre-configured detector instance
    **detector_kwargs
        Arguments to pass to CrisisDetector constructor if detector not provided

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'crisis_flags': Binary crisis indicators
        - 'scores': Anomaly scores
        - 'n_crises': Number of crisis events detected
        - 'crisis_ratio': Proportion of time in crisis
    """
    if detector is None:
        detector = CrisisDetector(**detector_kwargs)

    crisis_flags, scores = detector.detect(data, return_scores=True)

    # Count distinct crisis events
    n_crises = 0
    in_crisis = False
    for flag in crisis_flags:
        if flag == 1 and not in_crisis:
            n_crises += 1
            in_crisis = True
        elif flag == 0:
            in_crisis = False

    results = {
        "crisis_flags": crisis_flags,
        "scores": scores,
        "n_crises": n_crises,
        "crisis_ratio": np.sum(crisis_flags) / len(crisis_flags),
    }

    return results
