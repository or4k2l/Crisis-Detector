#!/usr/bin/env python3
"""
Crisis Detector Demo Script

This script demonstrates the basic usage of the Crisis Detector on financial data.
It loads S&P 500 data, runs the detector, and saves visualization plots.

Usage:
    python scripts/run_demo.py [--ticker TICKER] [--period PERIOD]

Examples:
    python scripts/run_demo.py
    python scripts/run_demo.py --ticker ^GSPC --period 2y
    python scripts/run_demo.py --ticker AAPL --period 6mo
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path to import crisis_detector
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from crisis_detector import CrisisDetector, load_finance_data, analyze_crisis


def setup_output_directory():
    """Create plots directory if it doesn't exist."""
    plots_dir = Path(__file__).parent.parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    return plots_dir


def plot_crisis_detection(
    data, crisis_flags, scores, ticker, output_dir, threshold=3.0
):
    """
    Create and save visualization plots.

    Parameters
    ----------
    data : pd.Series
        Price data
    crisis_flags : np.ndarray
        Binary crisis indicators
    scores : np.ndarray
        Anomaly scores
    ticker : str
        Stock ticker symbol
    output_dir : Path
        Directory to save plots
    threshold : float, default=3.0
        Detection threshold to show on plot
        Directory to save plots
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))

    # Plot 1: Price data with crisis highlights
    ax1.plot(data.index, data.values, label="Price", color="blue", linewidth=1.5)
    crisis_periods = np.where(crisis_flags == 1)[0]
    if len(crisis_periods) > 0:
        ax1.scatter(
            data.index[crisis_periods],
            data.values[crisis_periods],
            color="red",
            s=20,
            alpha=0.6,
            label="Crisis Events",
            zorder=5,
        )
    ax1.set_ylabel("Price", fontsize=12)
    ax1.set_title(f"Crisis Detection for {ticker}", fontsize=14, fontweight="bold")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Anomaly scores
    ax2.plot(data.index, scores, label="Anomaly Score", color="orange", linewidth=1.5)
    ax2.axhline(
        y=threshold, color="red", linestyle="--", linewidth=1, label="Threshold"
    )
    ax2.fill_between(
        data.index,
        0,
        scores,
        where=(scores > threshold),
        alpha=0.3,
        color="red",
        label="Crisis Region",
    )
    ax2.set_ylabel("Anomaly Score", fontsize=12)
    ax2.set_title("Anomaly Scores Over Time", fontsize=12)
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Crisis flags
    ax3.fill_between(
        data.index,
        0,
        crisis_flags,
        step="post",
        alpha=0.5,
        color="red",
        label="Crisis Period",
    )
    ax3.set_ylabel("Crisis Flag", fontsize=12)
    ax3.set_xlabel("Date", fontsize=12)
    ax3.set_title("Crisis Events Timeline", fontsize=12)
    ax3.set_ylim(-0.1, 1.3)
    ax3.legend(loc="best")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_file = (
        output_dir / f"crisis_detection_{ticker.replace('^', '').replace('/', '_')}.png"
    )
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✓ Plot saved to: {output_file}")

    plt.close()


def print_summary(results, ticker, period):
    """Print analysis summary."""
    print("\n" + "=" * 60)
    print("CRISIS DETECTION SUMMARY")
    print("=" * 60)
    print(f"Ticker:               {ticker}")
    print(f"Period:               {period}")
    print(f"Data points:          {len(results['crisis_flags'])}")
    print(f"Crisis events:        {results['n_crises']}")
    print(f"Crisis ratio:         {results['crisis_ratio']:.2%}")
    print(f"Max anomaly score:    {np.max(results['scores']):.2f}")
    print(f"Mean anomaly score:   {np.mean(results['scores']):.2f}")
    print("=" * 60)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run Crisis Detector demo on financial data"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="^GSPC",
        help="Stock ticker symbol (default: ^GSPC for S&P 500)",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="1y",
        help="Time period to analyze (default: 1y). Examples: 1mo, 3mo, 6mo, 1y, 2y, 5y",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=20,
        help="Rolling window size for statistics (default: 20)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="Anomaly detection threshold (default: 3.0)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("CRISIS DETECTOR DEMO")
    print("=" * 60)
    print(f"Loading data for {args.ticker} (period: {args.period})...")

    try:
        # Load financial data
        data = load_finance_data(ticker=args.ticker, period=args.period)
        prices = data["Close"]

        print(f"✓ Loaded {len(prices)} data points")
        print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

        # Run crisis detection
        print("\nRunning crisis detection...")
        detector = CrisisDetector(
            window_size=args.window_size,
            threshold=args.threshold,
            min_crisis_duration=1,
        )

        results = analyze_crisis(prices, detector=detector)

        print("✓ Detection complete")

        # Print summary
        print_summary(results, args.ticker, args.period)

        # Create visualizations
        print("\nGenerating plots...")
        output_dir = setup_output_directory()
        plot_crisis_detection(
            prices,
            results["crisis_flags"],
            results["scores"],
            args.ticker,
            output_dir,
            threshold=args.threshold,
        )

        print("\n✓ Demo completed successfully!")
        print("  Check the 'plots/' directory for visualizations.\n")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("  Make sure you have installed all required dependencies:")
        print("  pip install -r requirements.txt\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
