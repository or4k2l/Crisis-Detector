#!/usr/bin/env python3
"""
Demo script for Crisis Detector

This script demonstrates the basic usage of the CrisisDetector class
on financial market data (S&P 500). It downloads recent data, runs
the detector, and saves a visualization plot.
"""

import os
import sys
import warnings

# Add parent directory to path to import crisis_detector
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crisis_detector import CrisisDetector, load_finance_data  # noqa: E402

warnings.filterwarnings("ignore")


def main():
    """Run the demo analysis on S&P 500 data."""
    print("=" * 60)
    print("Crisis Detector Demo - S&P 500 Analysis")
    print("=" * 60)

    # Load financial data
    print("\nLoading S&P 500 data from 2020-2024...")
    data = load_finance_data(
        ticker="^GSPC", start_date="2020-01-01", end_date="2023-12-31"
    )

    if data is None or len(data) == 0:
        print(
            "Error: Could not load financial data. Please check your internet connection."
        )
        return 1

    print(f"Loaded {len(data)} data points")

    # Initialize the detector with reasonable parameters for financial data
    print("\nInitializing Crisis Detector...")
    detector = CrisisDetector(
        window_size=20,  # 20-day rolling window
        threshold=2.5,  # 2.5 standard deviations
        min_crisis_duration=5,  # At least 5 days
        use_isolation_forest=True,
        contamination=0.05,  # Expect 5% outliers
    )

    # Process the closing price
    print("Processing signal and detecting crises...")
    results = detector.process_signal(data, column="Close")

    # Print summary metrics
    print("\n" + "-" * 60)
    print("Detection Results:")
    print("-" * 60)
    metrics = results["metrics"]
    print(f"Total data points:     {metrics['total_points']}")
    print(f"Crisis points:         {metrics['crisis_points']}")
    print(f"Crisis ratio:          {metrics['crisis_ratio']:.2%}")
    print(f"Number of crises:      {metrics['n_crisis_events']}")
    print(f"Mean crisis score:     {metrics['mean_crisis_score']:.3f}")
    print(f"Max crisis score:      {metrics['max_crisis_score']:.3f}")
    print(f"Signal mean:           ${metrics['mean_signal']:.2f}")
    print(f"Signal std dev:        ${metrics['std_signal']:.2f}")

    # Create output directory
    plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Generate and save visualization
    output_path = os.path.join(plots_dir, "sp500_demo.png")
    print("\nGenerating visualization...")
    detector.plot_analysis(
        results, title="Crisis Detection: S&P 500 (2020-2023)", save_path=output_path
    )

    print(f"\nDemo complete! Plot saved to: {output_path}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
