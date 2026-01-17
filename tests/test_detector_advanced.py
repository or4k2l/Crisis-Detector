"""
Advanced tests for the Crisis Detector module.

These tests cover edge cases, domain-specific scenarios, and performance aspects.
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path to import crisis_detector
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crisis_detector import CrisisDetector  # noqa: E402


class TestCrisisDetectorAdvanced:
    """Advanced test suite for CrisisDetector."""

    def test_empty_signal(self):
        """Test handling of empty signal."""
        detector = CrisisDetector()
        signal = np.array([])
        
        with pytest.raises(Exception):
            detector.process_signal(signal)

    def test_single_point_signal(self):
        """Test handling of single data point."""
        detector = CrisisDetector(window_size=5)
        signal = np.array([1.0])
        
        results = detector.process_signal(signal)
        assert len(results["signal"]) == 1

    def test_all_nan_signal(self):
        """Test handling of signal with all NaN values."""
        detector = CrisisDetector()
        signal = np.array([np.nan] * 100)
        
        results = detector.process_signal(signal)
        assert len(results["signal"]) == 0

    def test_partial_nan_signal(self):
        """Test handling of signal with some NaN values."""
        detector = CrisisDetector()
        signal = np.random.randn(100)
        signal[20:30] = np.nan  # Insert NaN region
        
        results = detector.process_signal(signal)
        # Should process remaining valid points
        assert len(results["signal"]) == 90

    def test_constant_signal(self):
        """Test handling of constant signal (no variation)."""
        detector = CrisisDetector()
        signal = np.ones(200) * 5.0
        
        results = detector.process_signal(signal)
        # No crises should be detected in constant signal
        assert results["metrics"]["n_crisis_events"] == 0

    def test_multiple_crises(self):
        """Test detection of multiple distinct crisis events."""
        np.random.seed(123)
        signal = np.random.randn(500) * 0.1
        
        # Add three distinct spikes
        signal[100:110] += 10
        signal[250:260] += 10
        signal[400:410] += 10
        
        detector = CrisisDetector(window_size=20, threshold=2.0, min_crisis_duration=5)
        results = detector.process_signal(signal)
        
        # Should detect at least 2 crisis events (may merge nearby ones)
        assert results["metrics"]["n_crisis_events"] >= 2

    def test_min_crisis_duration_filtering(self):
        """Test that short anomalies are filtered by min_crisis_duration."""
        np.random.seed(456)
        signal = np.random.randn(300) * 0.1
        
        # Add very short spike (3 points)
        signal[150:153] += 10
        
        # With min_crisis_duration=5, should not detect it
        detector = CrisisDetector(window_size=20, threshold=2.0, min_crisis_duration=5)
        results = detector.process_signal(signal)
        
        assert results["metrics"]["n_crisis_events"] == 0
        
        # With min_crisis_duration=2, should detect it
        detector2 = CrisisDetector(window_size=20, threshold=2.0, min_crisis_duration=2)
        results2 = detector2.process_signal(signal)
        
        assert results2["metrics"]["n_crisis_events"] >= 1

    def test_threshold_sensitivity(self):
        """Test that higher thresholds reduce detections."""
        np.random.seed(789)
        signal = np.random.randn(300)
        signal[150:160] += 3  # Moderate spike
        
        # Low threshold - should detect
        detector_sensitive = CrisisDetector(threshold=1.5)
        results_sensitive = detector_sensitive.process_signal(signal)
        
        # High threshold - should not detect
        detector_strict = CrisisDetector(threshold=5.0)
        results_strict = detector_strict.process_signal(signal)
        
        assert results_sensitive["metrics"]["n_crisis_events"] >= results_strict["metrics"]["n_crisis_events"]

    def test_isolation_forest_toggle(self):
        """Test that isolation forest can be disabled."""
        np.random.seed(321)
        signal = np.random.randn(200)
        signal[100:110] += 5
        
        detector_with_if = CrisisDetector(use_isolation_forest=True)
        detector_without_if = CrisisDetector(use_isolation_forest=False)
        
        results_with = detector_with_if.process_signal(signal)
        results_without = detector_without_if.process_signal(signal)
        
        # Both should work, but may have different results
        assert "crisis_score" in results_with
        assert "crisis_score" in results_without

    def test_pandas_series_input(self):
        """Test processing pandas Series input."""
        import pandas as pd
        
        signal = pd.Series(np.random.randn(100), name="test_series")
        detector = CrisisDetector()
        results = detector.process_signal(signal)
        
        assert len(results["signal"]) == 100

    def test_pandas_dataframe_input(self):
        """Test processing pandas DataFrame input."""
        import pandas as pd
        
        df = pd.DataFrame({
            "value": np.random.randn(100),
            "other": np.random.randn(100)
        })
        
        detector = CrisisDetector()
        results = detector.process_signal(df, column="value")
        
        assert len(results["signal"]) == 100

    def test_custom_timestamps(self):
        """Test processing with custom timestamps."""
        signal = np.random.randn(100)
        timestamps = np.arange(0, 1000, 10)  # Custom time indices
        
        detector = CrisisDetector()
        results = detector.process_signal(signal, timestamps=timestamps)
        
        assert np.array_equal(results["timestamps"], timestamps)

    def test_datetime_index(self):
        """Test processing with datetime index."""
        import pandas as pd
        
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        signal = pd.Series(np.random.randn(100), index=dates)
        
        detector = CrisisDetector()
        results = detector.process_signal(signal)
        
        # Should preserve datetime timestamps
        assert len(results["timestamps"]) == 100

    def test_metrics_consistency(self):
        """Test that metrics are internally consistent."""
        np.random.seed(654)
        signal = np.random.randn(200)
        signal[100:110] += 5
        
        detector = CrisisDetector()
        results = detector.process_signal(signal)
        
        metrics = results["metrics"]
        
        # Crisis points should not exceed total points
        assert metrics["crisis_points"] <= metrics["total_points"]
        
        # Crisis ratio should be between 0 and 1
        assert 0 <= metrics["crisis_ratio"] <= 1
        
        # Crisis ratio calculation should be correct
        expected_ratio = metrics["crisis_points"] / metrics["total_points"]
        assert abs(metrics["crisis_ratio"] - expected_ratio) < 1e-10

    def test_small_signal_no_isolation_forest(self):
        """Test that Isolation Forest is skipped for small signals."""
        # Signal with <100 points should skip Isolation Forest
        signal = np.random.randn(50)
        
        detector = CrisisDetector(use_isolation_forest=True)
        results = detector.process_signal(signal)
        
        # Should still work without errors
        assert "crisis_score" in results
        assert len(results["signal"]) == 50

    def test_plot_generation(self):
        """Test that plot generation doesn't crash."""
        signal = np.random.randn(100)
        signal[50:60] += 5
        
        detector = CrisisDetector()
        results = detector.process_signal(signal)
        
        # Should generate plot without errors
        fig = detector.plot_analysis(results, title="Test Plot")
        assert fig is not None
        
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_volatility_calculation(self):
        """Test that volatility is calculated correctly."""
        # Signal with known rate of change
        signal = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        
        detector = CrisisDetector()
        results = detector.process_signal(signal)
        
        # Volatility should be relatively constant for linear signal
        volatility = results["volatility"]
        assert len(volatility) == len(signal)
        assert np.all(np.isfinite(volatility))

    def test_large_signal_performance(self):
        """Test performance with large signal (10K points)."""
        np.random.seed(999)
        signal = np.random.randn(10000)
        
        detector = CrisisDetector()
        results = detector.process_signal(signal)
        
        # Should complete without errors
        assert len(results["signal"]) == 10000
        assert "metrics" in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
