# Crisis-Detector

A Python toolkit for detecting crises, anomalies, and transient events in time series data across multiple domains including finance, seismology, neuroscience, and gravitational wave astronomy.

## Features

- **Multi-domain support**: Analyze financial data, seismic signals, EEG/MEG, and gravitational waves
- **Multiple detection methods**: Statistical analysis, spectral features, and signal processing
- **Flexible data loaders**: Built-in functions for loading data from various sources
- **Easy-to-use API**: Simple interface for crisis detection and analysis

## Quick Start

### Installation

Install core dependencies:
```bash
pip install -r requirements.txt
```

For optional heavy dependencies (seismic, gravitational waves, EEG/MEG):
```bash
pip install -r requirements-optional.txt
```

### Basic Usage

```python
import numpy as np
from crisis_detector import CrisisDetector

# Generate or load your time series data
data = np.random.randn(1000)

# Create detector and analyze
detector = CrisisDetector(threshold=3.0, window_size=50)
scores, anomalies = detector.detect(data)

# Get crisis periods
periods = detector.get_crisis_periods()
print(f"Detected {len(periods)} crisis periods")
```

### Financial Data Analysis

```python
from crisis_detector import CrisisDetector, load_finance_data

# Load S&P 500 data
data = load_finance_data(ticker='SPY', period='1y')

# Detect crises in closing prices
detector = CrisisDetector(threshold=3.0)
scores, anomalies = detector.detect(data['Close'].values)
```

### Running the Demo

```bash
python scripts/run_demo.py
```

This will download S&P 500 data and demonstrate crisis detection with visualization.

## Optional Dependencies

The following dependencies are optional and only required for specific data types:

- **mne** (>= 1.0.0): For EEG/MEG neurophysiology data
- **gwpy** (>= 3.0.0): For gravitational wave data (LIGO/Virgo)
- **obspy** (>= 1.3.0): For seismological data

Install them only if you need these specific features:
```bash
pip install -r requirements-optional.txt
```

## Development

### Running Tests

```bash
pytest
```

### Code Formatting and Linting

```bash
black .
flake8
```

## Data Policy

This repository does **not** include any raw datasets. All data is downloaded at runtime via the provided loader functions (`load_finance_data`, `load_seismic_data`, etc.).

## License

MIT License - see [LICENSE](LICENSE) file for details.

Copyright (c) 2026 or4k2l

## Contributing

Contributions are welcome! Please ensure:
- All tests pass (`pytest`)
- Code is formatted (`black .`)
- Linting passes (`flake8`)
