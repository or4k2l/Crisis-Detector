"""
Basic tests for the CrisisDetector class.

Tests the core functionality with synthetic data including sine waves
and transient spikes to verify anomaly detection capabilities.
"""

import numpy as np
import pytest
from crisis_detector import CrisisDetector


def test_detector_initialization():
    """Test that CrisisDetector initializes with correct parameters."""
    detector = CrisisDetector(threshold=3.0, window_size=50)
    assert detector.threshold == 3.0
    assert detector.window_size == 50
    assert detector.scores_ is None
    assert detector.anomalies_ is None


def test_detector_with_synthetic_sine_wave():
    """Test detector on synthetic sine wave with a transient spike."""
    # Generate synthetic data: sine wave + noise
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    signal = np.sin(2 * np.pi * t) + 0.1 * np.random.randn(len(t))

    # Add a transient spike (crisis event)
    spike_start = 400
    spike_end = 450
    signal[spike_start:spike_end] += 5.0  # Large amplitude spike

    # Run detector
    detector = CrisisDetector(threshold=3.0, window_size=50)
    scores, anomalies = detector.detect(signal)

    # Assertions
    assert scores is not None, "Scores should not be None"
    assert len(scores) == len(signal), "Scores length should match input length"
    assert np.all(np.isfinite(scores)), "All scores should be finite"
    assert isinstance(anomalies, np.ndarray), "Anomalies should be numpy array"
    assert anomalies.dtype == bool, "Anomalies should be boolean array"

    # Check that spike region has high scores
    spike_scores = scores[spike_start:spike_end]
    normal_scores = scores[: spike_start - 50]  # Well before spike
    assert np.mean(spike_scores) > np.mean(
        normal_scores
    ), "Spike region should have higher scores than normal region"


def test_detector_detects_anomalies():
    """Test that detector identifies anomalies in data with obvious outliers."""
    # Create data with clear outliers
    np.random.seed(123)
    data = np.random.randn(500) * 0.5  # Normal data
    data[250:260] += 10.0  # Clear outliers

    detector = CrisisDetector(threshold=2.5, window_size=30)
    scores, anomalies = detector.detect(data)

    # Should detect some anomalies
    assert np.sum(anomalies) > 0, "Should detect at least some anomalies"

    # Check that outlier region has higher scores than normal region
    outlier_scores = scores[250:260]
    normal_scores = scores[:200]
    assert np.mean(outlier_scores) > np.mean(
        normal_scores
    ), "Outlier region should have higher scores than normal region"


def test_get_crisis_periods():
    """Test that crisis periods are correctly identified."""
    # Create realistic data with clear crisis periods
    np.random.seed(555)
    data = np.random.randn(200) * 0.5  # Normal variation
    data[50:60] += 8.0  # First crisis
    data[150:165] += 8.0  # Second crisis

    detector = CrisisDetector(threshold=2.0, window_size=20)
    detector.detect(data)

    periods = detector.get_crisis_periods()

    assert len(periods) >= 1, "Should detect at least one crisis period"
    assert all(
        isinstance(p, tuple) and len(p) == 2 for p in periods
    ), "Each period should be a tuple of (start, end)"
    assert all(
        start <= end for start, end in periods
    ), "Start index should be <= end index"


def test_detector_handles_short_data():
    """Test that detector handles data shorter than window size."""
    short_data = np.array([1, 2, 3, 4, 5])
    detector = CrisisDetector(threshold=3.0, window_size=50)

    # Should not raise an error
    scores, anomalies = detector.detect(short_data)

    assert len(scores) == len(short_data)
    assert len(anomalies) == len(short_data)
    assert np.all(np.isfinite(scores))


def test_detector_fit_and_detect_consistency():
    """Test that fit() and detect() produce consistent results."""
    np.random.seed(456)
    data = np.random.randn(300)

    detector1 = CrisisDetector(threshold=3.0, window_size=40)
    scores1, anomalies1 = detector1.detect(data)

    detector2 = CrisisDetector(threshold=3.0, window_size=40)
    detector2.fit(data)
    scores2 = detector2.scores_
    anomalies2 = detector2.anomalies_

    np.testing.assert_array_almost_equal(scores1, scores2)
    np.testing.assert_array_equal(anomalies1, anomalies2)


def test_detector_scores_are_non_negative():
    """Test that anomaly scores are non-negative."""
    np.random.seed(789)
    data = np.random.randn(400)

    detector = CrisisDetector(threshold=3.0, window_size=50)
    scores, _ = detector.detect(data)

    assert np.all(scores >= 0), "All anomaly scores should be non-negative"


def test_crisis_periods_error_without_fit():
    """Test that get_crisis_periods raises error if called before fit."""
    detector = CrisisDetector()

    with pytest.raises(ValueError, match="Must call fit"):
        detector.get_crisis_periods()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
