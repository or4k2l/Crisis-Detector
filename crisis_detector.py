"""
Crisis Detector - A unified framework for detecting anomalies and crises across multiple domains.

This module provides the CrisisDetector class that can analyze time-series data from various
domains including finance, seismology, gravitational waves, and neurophysiology to identify
potential crisis events using statistical and machine learning techniques.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, Union, List

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CrisisDetector:
    """
    A multi-domain crisis detection system that identifies anomalies and crisis events
    in time-series data using statistical analysis and machine learning.

    The detector uses a sliding window approach combined with multiple detection methods:
    - Statistical thresholds (Z-score, moving average)
    - Isolation Forest for anomaly detection
    - Volatility and rate-of-change analysis

    Attributes:
        window_size (int): Size of the sliding window for analysis
        threshold (float): Z-score threshold for anomaly detection
        min_crisis_duration (int): Minimum duration for a crisis event
        use_isolation_forest (bool): Whether to use Isolation Forest
        contamination (float): Expected proportion of outliers for Isolation Forest
    """

    def __init__(
        self,
        window_size: int = 50,
        threshold: float = 2.5,
        min_crisis_duration: int = 5,
        use_isolation_forest: bool = True,
        contamination: float = 0.1,
    ):
        """
        Initialize the Crisis Detector.

        Args:
            window_size: Size of the sliding window for rolling statistics
            threshold: Z-score threshold for flagging anomalies
            min_crisis_duration: Minimum consecutive points to constitute a crisis
            use_isolation_forest: Whether to use Isolation Forest for anomaly detection
            contamination: Expected proportion of outliers in the data
        """
        self.window_size = window_size
        self.threshold = threshold
        self.min_crisis_duration = min_crisis_duration
        self.use_isolation_forest = use_isolation_forest
        self.contamination = contamination

    def process_signal(
        self,
        data: Union[np.ndarray, pd.Series, pd.DataFrame],
        timestamps: Optional[np.ndarray] = None,
        column: Optional[str] = None,
    ) -> Dict:
        """
        Process a time-series signal to detect crisis events.

        Args:
            data: Input time-series data (1D array, Series, or DataFrame)
            timestamps: Optional timestamps corresponding to data points
            column: Column name to use if data is a DataFrame

        Returns:
            Dictionary containing:
                - 'signal': Original signal values
                - 'timestamps': Time indices
                - 'crisis_score': Anomaly scores for each point
                - 'crisis_regions': Boolean mask of detected crisis regions
                - 'volatility': Rolling volatility measure
                - 'z_scores': Statistical z-scores
                - 'anomalies': Anomaly flags from multiple methods
                - 'metrics': Summary statistics
        """
        # Handle different input types
        if isinstance(data, pd.DataFrame):
            if column is None:
                column = data.columns[0]
            signal_values = data[column].values
            if timestamps is None and isinstance(data.index, pd.DatetimeIndex):
                timestamps = data.index.values
        elif isinstance(data, pd.Series):
            signal_values = data.values
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

        n_points = len(signal_values)
        
        # Handle edge case: too few points for analysis
        if n_points < 2:
            logger.warning(f"Signal has only {n_points} point(s), returning minimal results")
            return {
                "signal": signal_values,
                "timestamps": timestamps,
                "crisis_score": np.zeros(n_points),
                "crisis_regions": np.zeros(n_points, dtype=bool),
                "volatility": np.zeros(n_points),
                "z_scores": np.zeros(n_points),
                "anomalies": np.zeros(n_points, dtype=bool),
                "metrics": {
                    "total_points": n_points,
                    "crisis_points": 0,
                    "crisis_ratio": 0.0,
                    "n_crisis_events": 0,
                    "mean_crisis_score": 0.0,
                    "max_crisis_score": 0.0,
                    "mean_signal": np.mean(signal_values) if n_points > 0 else 0.0,
                    "std_signal": 0.0,
                    "n_anomalies": 0,
                },
            }

        # Calculate rolling statistics
        rolling_mean = (
            pd.Series(signal_values)
            .rolling(window=self.window_size, center=True)
            .mean()
            .values
        )
        rolling_std = (
            pd.Series(signal_values)
            .rolling(window=self.window_size, center=True)
            .std()
            .values
        )

        # Calculate z-scores
        z_scores = np.zeros_like(signal_values)
        valid_std = rolling_std > 0
        z_scores[valid_std] = (
            signal_values[valid_std] - rolling_mean[valid_std]
        ) / rolling_std[valid_std]

        # Calculate volatility (rate of change)
        volatility = np.abs(np.gradient(signal_values))
        volatility_norm = (volatility - np.nanmean(volatility)) / (
            np.nanstd(volatility) + 1e-10
        )

        # Initialize anomaly detection
        anomalies = np.zeros(n_points, dtype=bool)

        # Method 1: Z-score threshold
        anomalies = anomalies | (np.abs(z_scores) > self.threshold)

        # Method 2: Volatility threshold
        volatility_threshold = np.nanpercentile(volatility_norm, 95)
        anomalies = anomalies | (volatility_norm > volatility_threshold)

        # Method 3: Isolation Forest (if enabled)
        # Note: Requires minimum 100 points for statistical reliability
        # and stable anomaly detection performance
        if self.use_isolation_forest and n_points >= 100:
            logger.debug(f"Applying Isolation Forest to {n_points} data points")
            features = np.column_stack([signal_values, volatility_norm, z_scores])

            # Remove NaN rows for Isolation Forest
            valid_features = ~np.any(np.isnan(features), axis=1)
            if np.sum(valid_features) >= 10:
                iso_forest = IsolationForest(
                    contamination=self.contamination, random_state=42
                )
                iso_predictions = np.zeros(n_points)
                iso_predictions[valid_features] = iso_forest.fit_predict(
                    features[valid_features]
                )
                anomalies = anomalies | (iso_predictions == -1)

        # Combine into crisis score (0-1 scale)
        crisis_score = np.zeros(n_points)
        crisis_score = np.abs(z_scores) / (self.threshold * 2)  # Normalized z-score
        if volatility_threshold > 0:
            crisis_score += (
                volatility_norm / volatility_threshold
            )  # Normalized volatility
        crisis_score = np.clip(crisis_score, 0, 1)

        # Identify continuous crisis regions
        crisis_regions = self._identify_crisis_regions(anomalies)

        # Calculate summary metrics
        metrics = self._calculate_metrics(
            signal_values, crisis_score, crisis_regions, anomalies
        )

        return {
            "signal": signal_values,
            "timestamps": timestamps,
            "crisis_score": crisis_score,
            "crisis_regions": crisis_regions,
            "volatility": volatility,
            "z_scores": z_scores,
            "anomalies": anomalies,
            "metrics": metrics,
        }

    def _identify_crisis_regions(self, anomalies: np.ndarray) -> np.ndarray:
        """
        Identify continuous regions of crisis based on anomaly flags.

        Args:
            anomalies: Boolean array of anomaly flags

        Returns:
            Boolean array with crisis regions (filtered by min_crisis_duration)
        """
        crisis_regions = np.zeros_like(anomalies, dtype=bool)

        # Find continuous sequences of anomalies
        changes = np.diff(np.concatenate([[0], anomalies.astype(int), [0]]))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]

        # Filter by minimum duration
        for start, end in zip(starts, ends):
            if end - start >= self.min_crisis_duration:
                crisis_regions[start:end] = True

        return crisis_regions

    def _calculate_metrics(
        self,
        signal: np.ndarray,
        crisis_score: np.ndarray,
        crisis_regions: np.ndarray,
        anomalies: np.ndarray,
    ) -> Dict:
        """
        Calculate summary metrics for the detection results.

        Args:
            signal: Original signal values
            crisis_score: Anomaly scores
            crisis_regions: Boolean mask of crisis regions
            anomalies: Boolean mask of anomalies

        Returns:
            Dictionary of summary statistics
        """
        n_crisis_points = np.sum(crisis_regions)
        n_total_points = len(signal)

        # Find number of distinct crisis events
        changes = np.diff(np.concatenate([[0], crisis_regions.astype(int), [0]]))
        n_crisis_events = np.sum(changes == 1)

        metrics = {
            "total_points": n_total_points,
            "crisis_points": n_crisis_points,
            "crisis_ratio": (
                n_crisis_points / n_total_points if n_total_points > 0 else 0
            ),
            "n_crisis_events": n_crisis_events,
            "mean_crisis_score": np.mean(crisis_score),
            "max_crisis_score": np.max(crisis_score),
            "mean_signal": np.mean(signal),
            "std_signal": np.std(signal),
            "n_anomalies": np.sum(anomalies),
        }

        return metrics

    def plot_analysis(
        self,
        results: Dict,
        title: str = "Crisis Detection Analysis",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 10),
    ) -> plt.Figure:
        """
        Create a comprehensive visualization of the crisis detection results.

        Args:
            results: Output dictionary from process_signal()
            title: Plot title
            save_path: Optional path to save the figure
            figsize: Figure size (width, height)

        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

        signal = results["signal"]
        timestamps = results["timestamps"]
        crisis_score = results["crisis_score"]
        crisis_regions = results["crisis_regions"]
        volatility = results["volatility"]
        z_scores = results["z_scores"]

        # Plot 1: Original signal with crisis regions
        axes[0].plot(timestamps, signal, "b-", linewidth=1, alpha=0.7, label="Signal")
        if np.any(crisis_regions):
            axes[0].fill_between(
                timestamps,
                signal.min(),
                signal.max(),
                where=crisis_regions,
                alpha=0.3,
                color="red",
                label="Crisis Regions",
            )
        axes[0].set_ylabel("Signal Value")
        axes[0].set_title(title)
        axes[0].legend(loc="best")
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Crisis score
        axes[1].plot(
            timestamps, crisis_score, "r-", linewidth=1.5, label="Crisis Score"
        )
        axes[1].axhline(y=0.5, color="k", linestyle="--", alpha=0.5, label="Threshold")
        axes[1].fill_between(timestamps, 0, crisis_score, alpha=0.3, color="red")
        axes[1].set_ylabel("Crisis Score")
        axes[1].set_ylim([0, 1])
        axes[1].legend(loc="best")
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Z-scores
        axes[2].plot(timestamps, z_scores, "g-", linewidth=1, label="Z-Score")
        axes[2].axhline(y=self.threshold, color="r", linestyle="--", alpha=0.5)
        axes[2].axhline(y=-self.threshold, color="r", linestyle="--", alpha=0.5)
        axes[2].axhline(y=0, color="k", linestyle="-", alpha=0.3)
        axes[2].set_ylabel("Z-Score")
        axes[2].legend(loc="best")
        axes[2].grid(True, alpha=0.3)

        # Plot 4: Volatility
        axes[3].plot(timestamps, volatility, "m-", linewidth=1, label="Volatility")
        axes[3].set_ylabel("Volatility")
        axes[3].set_xlabel("Time")
        axes[3].legend(loc="best")
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Plot saved to {save_path}")

        return fig


# Data loader functions for various domains


def load_finance_data(
    ticker: str = "^GSPC", start_date: str = "2020-01-01", end_date: str = "2023-12-31"
) -> Optional[pd.DataFrame]:
    """
    Load financial time-series data using yfinance.

    Args:
        ticker: Stock ticker symbol (default: S&P 500)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        DataFrame with financial data including OHLCV columns
    """
    try:
        import yfinance as yf

        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        logger.info(f"Successfully loaded finance data for {ticker}")
        return data
    except Exception as e:
        logger.error(f"Error loading finance data: {e}", exc_info=True)
        return None


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
        network: Seismic network code
        station: Station code
        location: Location code
        channel: Channel code
        starttime: Start time (UTC)
        endtime: End time (UTC)

    Returns:
        Numpy array of seismic signal or None if error
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


def load_gravitational_wave_data(
    detector: str = "H1", start_time: int = 1126259446, end_time: int = 1126259478
) -> Optional[np.ndarray]:
    """
    Load gravitational wave data using gwpy.

    Args:
        detector: Detector name (H1, L1, V1)
        start_time: GPS start time
        end_time: GPS end time

    Returns:
        Numpy array of gravitational wave strain data or None if error
    """
    try:
        from gwpy.timeseries import TimeSeries

        data = TimeSeries.fetch_open_data(detector, start_time, end_time, cache=True)
        logger.info(f"Successfully loaded gravitational wave data from {detector}")
        return data.value
    except Exception as e:
        logger.error(f"Error loading gravitational wave data: {e}", exc_info=True)
        return None


def load_eeg_data(sample_dataset: str = "sample") -> Optional[np.ndarray]:
    """
    Load EEG data using MNE.

    Args:
        sample_dataset: MNE sample dataset name

    Returns:
        Numpy array of EEG signal or None if error
    """
    try:
        import mne

        # Load MNE sample data
        data_path = mne.datasets.sample.data_path()
        raw_fname = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"
        raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)

        # Get EEG data from first channel
        eeg_data = raw.get_data(picks="eeg")[0]
        logger.info(f"Successfully loaded EEG data from {sample_dataset}")
        return eeg_data
    except Exception as e:
        logger.error(f"Error loading EEG data: {e}", exc_info=True)
        return None


def load_economic_data(dataset: str = "GDP") -> Optional[pd.DataFrame]:
    """
    Load economic indicator data using statsmodels.

    Args:
        dataset: Economic indicator name

    Returns:
        DataFrame with economic data or None if error
    """
    try:
        from statsmodels.datasets import macrodata

        data = macrodata.load_pandas().data
        logger.info(f"Successfully loaded economic data: {dataset}")
        return data
    except Exception as e:
        logger.error(f"Error loading economic data: {e}", exc_info=True)
        return None
