# Crisis Detector

[![CI](https://github.com/or4k2l/Crisis-Detector/workflows/CI/badge.svg)](https://github.com/or4k2l/Crisis-Detector/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A unified framework for detecting anomalies and crisis events across multiple domains including finance, seismology, gravitational waves, and neurophysiology.

## Overview

Crisis Detector is a powerful Python library that identifies potential crisis events in time-series data using a combination of statistical analysis and machine learning techniques. The detector uses a sliding window approach with multiple detection methods:

- **Statistical Thresholds**: Z-score and moving average analysis
- **Isolation Forest**: Machine learning-based anomaly detection
- **Volatility Analysis**: Rate-of-change and trend detection
- **Multi-domain Support**: Works with finance, seismic, gravitational wave, and EEG data

## Features

- 🔍 **Multi-method Detection**: Combines statistical and ML approaches for robust crisis identification
- 📊 **Comprehensive Visualization**: Generates detailed plots showing signal analysis and detected crises
- 🌐 **Multi-domain Support**: Built-in data loaders for various scientific and financial domains
- ⚙️ **Configurable**: Easily adjust detection sensitivity and parameters
- 🧪 **Well-tested**: Comprehensive test suite with synthetic data validation
- 📈 **Production-ready**: CI/CD pipeline with automated testing and linting

## Installation

### From source

```bash
git clone https://github.com/or4k2l/Crisis-Detector.git
cd Crisis-Detector
pip install -r requirements.txt
```

### Requirements

- Python 3.9+
- numpy, pandas, scipy
- scikit-learn, matplotlib
- statsmodels, yfinance
- mne, gwpy, obspy (for domain-specific data loading)

## Quick Start

### Basic Usage

```python
from crisis_detector import CrisisDetector
import numpy as np

# Create a detector instance
detector = CrisisDetector(
    window_size=50,
    threshold=2.5,
    min_crisis_duration=5
)

# Generate or load your time-series data
signal = np.random.randn(1000)

# Process the signal
results = detector.process_signal(signal)

# Visualize the results
detector.plot_analysis(results, title="My Analysis", save_path="output.png")

# Access metrics
print(f"Detected {results['metrics']['n_crisis_events']} crisis events")
print(f"Crisis ratio: {results['metrics']['crisis_ratio']:.2%}")
```

### Financial Market Analysis

```python
from crisis_detector import CrisisDetector, load_finance_data

# Load S&P 500 data
data = load_finance_data(ticker="^GSPC", start_date="2020-01-01", end_date="2023-12-31")

# Initialize detector
detector = CrisisDetector(window_size=20, threshold=2.5)

# Analyze closing prices
results = detector.process_signal(data, column='Close')

# Generate visualization
detector.plot_analysis(results, title="S&P 500 Crisis Detection", save_path="sp500_analysis.png")
```

### Domain-Specific Data Loading

The library includes built-in loaders for multiple domains:

```python
from crisis_detector import (
    load_finance_data,      # Financial markets (yfinance)
    load_seismic_data,      # Seismology (obspy)
    load_gravitational_wave_data,  # Gravitational waves (gwpy)
    load_eeg_data,          # Neurophysiology (mne)
    load_economic_data      # Economic indicators (statsmodels)
)

# Load financial data
finance_data = load_finance_data(ticker="^GSPC")

# Load seismic data
seismic_data = load_seismic_data(network="IU", station="ANMO")

# Load gravitational wave data
gw_data = load_gravitational_wave_data(detector="H1")

# Load EEG data
eeg_data = load_eeg_data()
```

## Running the Demo

A demo script is provided that demonstrates the detector on S&P 500 financial data:

```bash
python scripts/run_demo.py
```

This will:
1. Download recent S&P 500 data
2. Run the crisis detector
3. Print summary statistics
4. Save a visualization to `plots/sp500_demo.png`

## Example Outputs

The Crisis Detector generates comprehensive visualizations showing:

1. **Original Signal with Crisis Regions**: Highlights detected crisis periods
2. **Crisis Score Timeline**: Shows the anomaly score over time
3. **Z-Score Analysis**: Statistical deviation from rolling mean
4. **Volatility**: Rate of change and trend analysis

### Example Plots

Below are example outputs from the detector on various datasets:

#### S&P 500 Financial Data (2020-2023)
![S&P 500 Analysis](https://github.com/user-attachments/assets/example1.png)

#### Seismic Event Detection
![Seismic Analysis](https://github.com/user-attachments/assets/example2.png)

#### Gravitational Wave Detection
![Gravitational Wave Analysis](https://github.com/user-attachments/assets/example3.png)

*Note: The plots show crisis detection on real-world data from various domains. Crisis regions are highlighted in red, showing periods of significant anomalies.*

## Configuration

The `CrisisDetector` class accepts several parameters for customization:

```python
detector = CrisisDetector(
    window_size=50,              # Size of rolling window for statistics
    threshold=2.5,               # Z-score threshold for anomaly detection
    min_crisis_duration=5,       # Minimum consecutive anomalies for a crisis
    use_isolation_forest=True,   # Enable ML-based detection
    contamination=0.1            # Expected proportion of outliers
)
```

### Parameter Guidelines

- **window_size**: Larger windows smooth out noise but may miss short crises (20-100 typical)
- **threshold**: Higher values reduce false positives but may miss subtle crises (2.0-3.0 typical)
- **min_crisis_duration**: Filters out brief spikes (3-10 typical)
- **contamination**: Adjust based on expected anomaly rate in your domain (0.01-0.2 typical)

## Testing

Run the test suite to validate the installation:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_detector_basic.py -v

# Run with coverage
pytest tests/ --cov=crisis_detector --cov-report=html
```

## Development

### Code Quality

The project uses black for code formatting and flake8 for linting:

```bash
# Format code
black .

# Check formatting
black --check .

# Run linter
flake8 . --max-line-length=100
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linters
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Project Structure

```
Crisis-Detector/
├── crisis_detector.py          # Main detector implementation
├── scripts/
│   └── run_demo.py            # Demo script
├── tests/
│   └── test_detector_basic.py # Test suite
├── .github/
│   └── workflows/
│       └── ci.yml             # CI/CD pipeline
├── requirements.txt           # Dependencies
├── README.md                  # Documentation
├── LICENSE                    # MIT License
└── .gitignore                # Git ignore rules
```

## Use Cases

### Financial Markets
- Market crash detection
- Volatility spike identification
- Trading anomaly detection

### Seismology
- Earthquake detection
- Aftershock identification
- Seismic event classification

### Gravitational Waves
- Binary merger detection
- Glitch identification
- Signal quality monitoring

### Neurophysiology
- Seizure detection
- Brain activity anomalies
- Sleep stage transitions

## API Reference

### CrisisDetector Class

**Methods:**

- `process_signal(data, timestamps=None, column=None)`: Process time-series data and detect crises
- `plot_analysis(results, title, save_path=None)`: Visualize detection results
- `_identify_crisis_regions(anomalies)`: Internal method to identify continuous crisis regions
- `_calculate_metrics(signal, crisis_score, crisis_regions, anomalies)`: Calculate summary statistics

**Returns:**

The `process_signal` method returns a dictionary containing:
- `signal`: Processed signal values
- `timestamps`: Time indices
- `crisis_score`: Anomaly scores (0-1 scale)
- `crisis_regions`: Boolean mask of crisis regions
- `volatility`: Rolling volatility measure
- `z_scores`: Statistical z-scores
- `anomalies`: Boolean mask of detected anomalies
- `metrics`: Dictionary of summary statistics

## Performance Considerations

- For large datasets (>100K points), consider downsampling or processing in chunks
- Isolation Forest is disabled for signals with <100 points
- Adjust `window_size` based on your data frequency and expected crisis duration
- Use `contamination` parameter to tune sensitivity for your specific domain

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Uses scikit-learn for machine learning components
- Integrates with domain-specific libraries (yfinance, obspy, gwpy, mne)
- Inspired by research in multi-domain anomaly detection

## Citation

If you use this software in your research, please cite:

```bibtex
@software{crisis_detector,
  title={Crisis Detector: A Multi-Domain Anomaly Detection Framework},
  author={or4k2l},
  year={2026},
  url={https://github.com/or4k2l/Crisis-Detector}
}
```

## Contact

- GitHub: [@or4k2l](https://github.com/or4k2l)
- Issues: [GitHub Issues](https://github.com/or4k2l/Crisis-Detector/issues)

## Roadmap

- [ ] Add support for multivariate time-series
- [ ] Implement additional ML models (LSTM, Transformer)
- [ ] Add real-time streaming detection
- [ ] Create interactive web dashboard
- [ ] Add more domain-specific loaders
- [ ] Improve documentation and tutorials

---

**Made with ❤️ for data scientists, researchers, and engineers working with time-series data.**