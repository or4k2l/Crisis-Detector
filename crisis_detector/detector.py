"""CrisisDetector – core detection class."""

import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .utils import prepare_signal
from .visualization import plot_analysis as _plot_analysis

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class CrisisDetector:
    """
    A multi-domain crisis detection system that identifies anomalies and crisis
    events in time-series data using statistical analysis and machine learning.

    The detector uses a sliding window approach combined with multiple detection
    methods:

    - Statistical thresholds (Z-score, moving average)
    - Isolation Forest for anomaly detection
    - Volatility and rate-of-change analysis

    Attributes:
        window_size (int): Size of the sliding window for analysis.
        threshold (float): Z-score threshold for anomaly detection.
        min_crisis_duration (int): Minimum duration for a crisis event.
        use_isolation_forest (bool): Whether to use Isolation Forest.
        contamination (float): Expected proportion of outliers for Isolation Forest.
    """

    def __init__(
        self,
        window_size: int = 50,
        threshold: float = 2.5,
        min_crisis_duration: int = 5,
        use_isolation_forest: bool = True,
        contamination: float = 0.1,
    ) -> None:
        """
        Initialize the Crisis Detector.

        Args:
            window_size: Size of the sliding window for rolling statistics.
            threshold: Z-score threshold for flagging anomalies.
            min_crisis_duration: Minimum consecutive points to constitute a
                crisis.
            use_isolation_forest: Whether to use Isolation Forest for anomaly
                detection.
            contamination: Expected proportion of outliers in the data.
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
            data: Input time-series data (1-D array, Series, or DataFrame).
            timestamps: Optional timestamps corresponding to data points.
            column: Column name to use if *data* is a DataFrame.

        Returns:
            Dictionary containing:

            - ``signal``: Original signal values.
            - ``timestamps``: Time indices.
            - ``crisis_score``: Anomaly scores for each point.
            - ``crisis_regions``: Boolean mask of detected crisis regions.
            - ``volatility``: Rolling volatility measure.
            - ``z_scores``: Statistical z-scores.
            - ``anomalies``: Anomaly flags from multiple methods.
            - ``metrics``: Summary statistics.
        """
        signal_values, timestamps = prepare_signal(data, timestamps, column)
        n_points = len(signal_values)

        # Handle edge case: too few points for analysis
        if n_points < 2:
            logger.warning(
                f"Signal has only {n_points} point(s), returning minimal results"
            )
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
        crisis_score += np.abs(z_scores) / (self.threshold * 2)  # Normalized z-score
        if volatility_threshold > 0:
            crisis_score += volatility_norm / volatility_threshold  # Normalized volatility
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
            anomalies: Boolean array of anomaly flags.

        Returns:
            Boolean array with crisis regions (filtered by
            ``min_crisis_duration``).
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
            signal: Original signal values.
            crisis_score: Anomaly scores.
            crisis_regions: Boolean mask of crisis regions.
            anomalies: Boolean mask of anomalies.

        Returns:
            Dictionary of summary statistics.
        """
        n_crisis_points = np.sum(crisis_regions)
        n_total_points = len(signal)

        # Find number of distinct crisis events
        changes = np.diff(np.concatenate([[0], crisis_regions.astype(int), [0]]))
        n_crisis_events = np.sum(changes == 1)

        return {
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

    def plot_analysis(
        self,
        results: Dict,
        title: str = "Crisis Detection Analysis",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 10),
    ) -> plt.Figure:
        """
        Create a comprehensive visualization of the crisis detection results.

        Delegates to :func:`crisis_detector.visualization.plot_analysis`.

        Args:
            results: Output dictionary from :meth:`process_signal`.
            title: Plot title.
            save_path: Optional path to save the figure.
            figsize: Figure size ``(width, height)`` in inches.

        Returns:
            Matplotlib ``Figure`` object.
        """
        return _plot_analysis(
            results,
            title=title,
            save_path=save_path,
            figsize=figsize,
            threshold=self.threshold,
        )
