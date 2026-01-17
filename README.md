# Crisis Detector

A Python toolkit for detecting anomalies and crisis events in time series data across multiple domains including financial markets, seismic activity, and physiological signals.

## Features

- **Multi-domain support**: Analyze financial data, seismic signals, physiological data, and more
- **Statistical methods**: Z-score analysis, volatility detection, and rolling window statistics
- **Easy-to-use API**: Simple interface for detecting crisis events in time series
- **Flexible configuration**: Customizable thresholds and detection parameters
- **Built-in data loaders**: Load financial data from Yahoo Finance or generate synthetic test data

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/or4k2l/Crisis-Detector.git
cd Crisis-Detector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install heavy dependencies for specialized domains:
```bash
pip install -r requirements-optional.txt
```

### Basic Usage

```python
from crisis_detector import CrisisDetector, load_finance_data

# Load S&P 500 data
data = load_finance_data("^GSPC", period="1y")
prices = data['Close']

# Create and configure detector
detector = CrisisDetector(
    window_size=20,
    threshold=3.0,
    min_crisis_duration=1
)

# Detect crisis events
crisis_flags, scores = detector.detect(prices, return_scores=True)

# Analyze results
n_crises = crisis_flags.sum()
print(f"Detected {n_crises} crisis events")
```

### Run Demo

Run the included demo script to see the detector in action:

```bash
python scripts/run_demo.py
```

This will analyze S&P 500 data and save plots to the `plots/` directory.

### Run Tests

```bash
pytest tests/
```

## API Reference

### CrisisDetector

Main class for crisis detection.

**Parameters:**
- `window_size` (int): Size of rolling window for statistics (default: 20)
- `threshold` (float): Z-score threshold for anomaly detection (default: 3.0)
- `min_crisis_duration` (int): Minimum duration for valid crisis events (default: 1)

**Methods:**
- `fit(data)`: Fit the detector to training data
- `detect(data, return_scores=True)`: Detect crisis events in data

### Helper Functions

- `load_finance_data(ticker, start_date, end_date, period)`: Load financial data from Yahoo Finance
- `load_synthetic_data(n_samples, noise_level, n_crises)`: Generate synthetic data with crisis events
- `analyze_crisis(data, detector)`: Perform comprehensive crisis analysis

## Development

### Code Quality

Run linting checks:
```bash
black .
flake8 .
```

### Testing

Run the test suite:
```bash
pytest tests/ -v
```

## Optional Dependencies

The `requirements-optional.txt` file contains heavy dependencies for specialized domains:
- `mne`: For EEG/physiological signal analysis
- `gwpy`: For gravitational wave data
- `obspy`: For seismic data analysis

These are not required for basic functionality and are excluded from CI to keep builds fast.

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.