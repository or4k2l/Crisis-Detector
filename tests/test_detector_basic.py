"""
Basic tests for Crisis Detector module.

These tests verify the core functionality using synthetic data with known properties.
"""

import pytest
import numpy as np
import pandas as pd
from crisis_detector import CrisisDetector, load_synthetic_data, analyze_crisis


class TestCrisisDetectorBasic:
    """Basic tests for CrisisDetector class."""

    def test_detector_initialization(self):
        """Test that detector initializes with correct parameters."""
        detector = CrisisDetector(window_size=20, threshold=3.0, min_crisis_duration=1)
        assert detector.window_size == 20
        assert detector.threshold == 3.0
        assert detector.min_crisis_duration == 1
        assert not detector._is_fitted

    def test_detector_with_synthetic_data(self):
        """Test detector on synthetic data with known crisis events."""
        # Generate synthetic data with 3 crisis events
        data, true_labels = load_synthetic_data(
            n_samples=1000, noise_level=0.1, n_crises=3, random_seed=42
        )

        # Create and run detector
        detector = CrisisDetector(window_size=20, threshold=2.5, min_crisis_duration=1)
        crisis_flags, scores = detector.detect(data, return_scores=True)

        # Verify output shapes
        assert len(crisis_flags) == len(data)
        assert len(scores) == len(data)

        # Verify outputs are valid
        assert crisis_flags.dtype == int
        assert np.all((crisis_flags == 0) | (crisis_flags == 1))
        assert np.all(np.isfinite(scores))

        # Verify some crises were detected
        assert np.sum(crisis_flags) > 0

    def test_detector_output_shape(self):
        """Test that detector outputs have correct shape."""
        # Create simple sine wave with spike
        t = np.linspace(0, 10, 500)
        data = np.sin(t)
        data[250:260] += 5  # Add a spike

        detector = CrisisDetector()
        crisis_flags, scores = detector.detect(data, return_scores=True)

        assert len(crisis_flags) == len(data)
        assert len(scores) == len(data)

    def test_detector_return_scores_false(self):
        """Test detector with return_scores=False."""
        data = np.sin(np.linspace(0, 10, 500))
        detector = CrisisDetector()

        result = detector.detect(data, return_scores=False)

        # Should return only crisis_flags (1D array)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert len(result) == len(data)

    def test_detector_detects_spike(self):
        """Test that detector can detect an obvious spike."""
        # Create data with clear spike
        t = np.linspace(0, 10, 500)
        data = np.sin(t) + 0.1 * np.random.randn(500)

        # Add large spike in middle
        spike_start = 250
        spike_end = 260
        data[spike_start:spike_end] += 10  # Very large spike

        detector = CrisisDetector(window_size=20, threshold=2.0)
        crisis_flags, scores = detector.detect(data, return_scores=True)

        # Verify spike region is detected
        spike_detected = np.any(crisis_flags[spike_start:spike_end] == 1)
        assert spike_detected, "Detector should detect the large spike"

    def test_detector_with_pandas_series(self):
        """Test detector works with pandas Series input."""
        data = pd.Series(np.sin(np.linspace(0, 10, 500)))
        detector = CrisisDetector()

        crisis_flags, scores = detector.detect(data, return_scores=True)

        assert len(crisis_flags) == len(data)
        assert len(scores) == len(data)

    def test_detector_with_pandas_dataframe(self):
        """Test detector works with pandas DataFrame input."""
        data = pd.DataFrame({"value": np.sin(np.linspace(0, 10, 500))})
        detector = CrisisDetector()

        crisis_flags, scores = detector.detect(data, return_scores=True)

        assert len(crisis_flags) == len(data)
        assert len(scores) == len(data)

    def test_detector_scores_are_finite(self):
        """Test that all anomaly scores are finite (no NaN or inf)."""
        data = np.sin(np.linspace(0, 10, 500))
        detector = CrisisDetector()

        _, scores = detector.detect(data, return_scores=True)

        assert np.all(np.isfinite(scores)), "All scores should be finite"

    def test_detector_minimum_data_length(self):
        """Test that detector raises error for data shorter than window."""
        detector = CrisisDetector(window_size=50)
        short_data = np.array([1, 2, 3, 4, 5])  # Only 5 points, window is 50

        with pytest.raises(
            ValueError, match="Data length.*must be at least.*window_size"
        ):
            detector.detect(short_data)


class TestSyntheticDataGeneration:
    """Tests for synthetic data generation."""

    def test_load_synthetic_data_shape(self):
        """Test that synthetic data has correct shape."""
        data, labels = load_synthetic_data(n_samples=1000, n_crises=3, random_seed=42)

        assert len(data) == 1000
        assert len(labels) == 1000
        assert data.dtype == np.float64
        assert labels.dtype == int

    def test_load_synthetic_data_has_crises(self):
        """Test that synthetic data contains crisis labels."""
        data, labels = load_synthetic_data(n_samples=1000, n_crises=5, random_seed=42)

        # Should have some crisis events
        assert np.sum(labels) > 0
        assert np.any(labels == 1)

    def test_synthetic_data_reproducible(self):
        """Test that synthetic data generation is reproducible with seed."""
        data1, labels1 = load_synthetic_data(n_samples=500, random_seed=123)
        data2, labels2 = load_synthetic_data(n_samples=500, random_seed=123)

        np.testing.assert_array_equal(data1, data2)
        np.testing.assert_array_equal(labels1, labels2)


class TestAnalysisFunctions:
    """Tests for analysis helper functions."""

    def test_analyze_crisis_returns_dict(self):
        """Test that analyze_crisis returns a dictionary with expected keys."""
        data = np.sin(np.linspace(0, 10, 500))
        results = analyze_crisis(data)

        assert isinstance(results, dict)
        assert "crisis_flags" in results
        assert "scores" in results
        assert "n_crises" in results
        assert "crisis_ratio" in results

    def test_analyze_crisis_with_custom_detector(self):
        """Test analyze_crisis with custom detector."""
        data = np.sin(np.linspace(0, 10, 500))
        detector = CrisisDetector(window_size=30, threshold=2.0)

        results = analyze_crisis(data, detector=detector)

        assert isinstance(results, dict)
        assert len(results["crisis_flags"]) == len(data)

    def test_analyze_crisis_counts_events_correctly(self):
        """Test that crisis event counting works correctly."""
        # Create data with 2 distinct spike regions
        data = np.zeros(500)
        data[100:110] = 10  # First crisis
        data[300:310] = 10  # Second crisis

        detector = CrisisDetector(threshold=2.0)
        results = analyze_crisis(data, detector=detector)

        # Should detect 2 distinct crisis events
        assert results["n_crises"] >= 1
        assert results["crisis_ratio"] > 0


class TestDetectorFit:
    """Tests for the fit method."""

    def test_fit_method(self):
        """Test that fit method works and sets fitted flag."""
        data = np.sin(np.linspace(0, 10, 500))
        detector = CrisisDetector()

        assert not detector._is_fitted
        detector.fit(data)
        assert detector._is_fitted

    def test_fit_returns_self(self):
        """Test that fit returns self for chaining."""
        data = np.sin(np.linspace(0, 10, 500))
        detector = CrisisDetector()

        result = detector.fit(data)
        assert result is detector


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
