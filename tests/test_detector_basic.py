"""
Basic tests for the Crisis Detector module.

These tests validate the core functionality of the CrisisDetector class
using synthetic data to ensure the detector works correctly without
requiring external data downloads.
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path to import crisis_detector
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crisis_detector import CrisisDetector


class TestCrisisDetectorBasic:
    """Basic test suite for CrisisDetector."""
    
    def test_detector_initialization(self):
        """Test that the detector can be initialized with default parameters."""
        detector = CrisisDetector()
        assert detector.window_size == 50
        assert detector.threshold == 2.5
        assert detector.min_crisis_duration == 5
        assert detector.use_isolation_forest is True
        
    def test_detector_custom_parameters(self):
        """Test that the detector accepts custom parameters."""
        detector = CrisisDetector(
            window_size=30,
            threshold=3.0,
            min_crisis_duration=10,
            use_isolation_forest=False
        )
        assert detector.window_size == 30
        assert detector.threshold == 3.0
        assert detector.min_crisis_duration == 10
        assert detector.use_isolation_forest is False
        
    def test_process_simple_signal(self):
        """Test processing a simple synthetic signal."""
        # Create a simple sine wave
        t = np.linspace(0, 10 * np.pi, 500)
        signal = np.sin(t)
        
        detector = CrisisDetector(window_size=20, threshold=2.0)
        results = detector.process_signal(signal)
        
        # Validate output structure
        assert 'signal' in results
        assert 'timestamps' in results
        assert 'crisis_score' in results
        assert 'crisis_regions' in results
        assert 'volatility' in results
        assert 'z_scores' in results
        assert 'anomalies' in results
        assert 'metrics' in results
        
        # Validate output shapes
        assert len(results['signal']) == len(signal)
        assert len(results['crisis_score']) == len(signal)
        assert len(results['timestamps']) == len(signal)
        
    def test_process_signal_with_spike(self):
        """Test processing a signal with a clear anomaly (spike)."""
        # Create signal with a spike (crisis)
        n_points = 300
        signal = np.random.randn(n_points) * 0.5  # Low noise
        
        # Add a clear spike in the middle
        spike_start = 140
        spike_end = 160
        signal[spike_start:spike_end] += 10  # Large spike
        
        detector = CrisisDetector(
            window_size=20,
            threshold=2.0,
            min_crisis_duration=5
        )
        results = detector.process_signal(signal)
        
        # The spike should be detected as a crisis
        assert np.any(results['crisis_regions'])
        assert np.any(results['anomalies'])
        
        # Crisis score should be elevated during the spike
        crisis_during_spike = results['crisis_score'][spike_start:spike_end]
        crisis_before_spike = results['crisis_score'][:spike_start-20]
        assert np.mean(crisis_during_spike) > np.mean(crisis_before_spike)
        
    def test_metrics_calculation(self):
        """Test that metrics are calculated correctly."""
        # Simple signal
        signal = np.random.randn(200)
        
        detector = CrisisDetector(window_size=20)
        results = detector.process_signal(signal)
        
        metrics = results['metrics']
        
        # Validate metric keys
        assert 'total_points' in metrics
        assert 'crisis_points' in metrics
        assert 'crisis_ratio' in metrics
        assert 'n_crisis_events' in metrics
        assert 'mean_crisis_score' in metrics
        assert 'max_crisis_score' in metrics
        assert 'mean_signal' in metrics
        assert 'std_signal' in metrics
        
        # Validate metric values
        assert metrics['total_points'] == len(signal)
        assert 0 <= metrics['crisis_ratio'] <= 1
        assert np.isfinite(metrics['mean_crisis_score'])
        assert np.isfinite(metrics['max_crisis_score'])
        assert metrics['max_crisis_score'] >= 0
        
    def test_no_nans_in_output(self):
        """Test that crisis scores don't contain NaN values."""
        signal = np.random.randn(150)
        
        detector = CrisisDetector(window_size=20)
        results = detector.process_signal(signal)
        
        # Crisis score should not have NaN (even at boundaries)
        assert np.all(np.isfinite(results['crisis_score']))
        
    def test_empty_signal_handling(self):
        """Test handling of very short signals."""
        signal = np.array([1.0, 2.0, 3.0])
        
        detector = CrisisDetector(window_size=10)
        results = detector.process_signal(signal)
        
        # Should still return valid results
        assert len(results['signal']) == len(signal)
        assert 'metrics' in results
        
    def test_process_with_nan_values(self):
        """Test that NaN values in input are handled correctly."""
        signal = np.random.randn(200)
        signal[50:55] = np.nan  # Inject some NaN values
        
        detector = CrisisDetector(window_size=20)
        results = detector.process_signal(signal)
        
        # Output signal should have NaN removed
        assert len(results['signal']) < len(signal)
        assert not np.any(np.isnan(results['signal']))
        assert np.all(np.isfinite(results['crisis_score']))


class TestCrisisDetectorPlotting:
    """Test suite for plotting functionality."""
    
    def test_plot_analysis_runs(self):
        """Test that plot_analysis runs without errors."""
        signal = np.sin(np.linspace(0, 4 * np.pi, 200))
        
        detector = CrisisDetector(window_size=20)
        results = detector.process_signal(signal)
        
        # This should not raise an exception
        fig = detector.plot_analysis(results, title="Test Plot")
        assert fig is not None
        
        # Clean up
        import matplotlib.pyplot as plt
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
