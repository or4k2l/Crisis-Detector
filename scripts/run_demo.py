#!/usr/bin/env python3
"""
Demo script for Crisis Detector on S&P 500 financial data.

This script demonstrates the crisis detection capabilities by:
1. Downloading recent S&P 500 data using yfinance
2. Running the CrisisDetector on closing prices
3. Visualizing the results with detected crisis periods highlighted
"""

import sys
import os

# Add parent directory to path to import crisis_detector
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from crisis_detector import CrisisDetector, load_finance_data  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Run Crisis Detector demo on S&P 500 data"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="SPY",
        help="Stock ticker symbol (default: SPY)",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="1y",
        help="Time period (default: 1y)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="Anomaly detection threshold (default: 3.0)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=20,
        help="Rolling window size (default: 20)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save plot to file instead of displaying",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plots/sp500_crisis_detection.png",
        help="Output plot filename (default: plots/sp500_crisis_detection.png)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Crisis Detector - S&P 500 Demo")
    print("=" * 60)
    print(f"\nTicker: {args.ticker}")
    print(f"Period: {args.period}")
    print(f"Threshold: {args.threshold}")
    print(f"Window Size: {args.window_size}")
    print("\nDownloading data...")

    try:
        # Load financial data
        data = load_finance_data(ticker=args.ticker, period=args.period)
        print(f"Downloaded {len(data)} data points")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")

        # Extract closing prices
        prices = data["Close"].values
        dates = data.index

        # Run crisis detector
        print("\nRunning crisis detection...")
        detector = CrisisDetector(
            threshold=args.threshold, window_size=args.window_size
        )
        scores, anomalies = detector.detect(prices)

        # Get crisis periods
        periods = detector.get_crisis_periods()

        print("\nResults:")
        print(f"  Total anomalies detected: {np.sum(anomalies)}")
        print(f"  Max anomaly score: {np.max(scores):.2f}")
        print(f"  Number of crisis periods: {len(periods)}")

        if periods:
            print("\nCrisis periods:")
            for i, (start, end) in enumerate(periods, 1):
                start_date = dates[start]
                end_date = dates[end]
                duration = end - start + 1
                print(
                    f"  {i}. {start_date.date()} to {end_date.date()} "
                    f"({duration} days)"
                )

        # Visualization
        print("\nGenerating visualization...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Plot 1: Price with crisis periods highlighted
        ax1.plot(dates, prices, "b-", linewidth=1.5, label="Close Price")

        # Highlight crisis periods
        for start, end in periods:
            ax1.axvspan(
                dates[start],
                dates[end],
                alpha=0.3,
                color="red",
                label="Crisis" if start == periods[0][0] else "",
            )

        ax1.set_ylabel("Price ($)", fontsize=12)
        ax1.set_title(
            f"{args.ticker} Crisis Detection (Threshold={args.threshold})",
            fontsize=14,
            fontweight="bold",
        )
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Anomaly scores
        ax2.plot(dates, scores, "g-", linewidth=1, label="Anomaly Score")
        ax2.axhline(
            y=args.threshold, color="r", linestyle="--", linewidth=2, label="Threshold"
        )
        ax2.fill_between(
            dates,
            0,
            scores,
            where=anomalies,
            alpha=0.3,
            color="red",
            label="Detected Anomalies",
        )

        ax2.set_xlabel("Date", fontsize=12)
        ax2.set_ylabel("Anomaly Score", fontsize=12)
        ax2.set_title("Anomaly Scores Over Time", fontsize=12)
        ax2.legend(loc="best")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if args.save:
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(args.output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            plt.savefig(args.output, dpi=300, bbox_inches="tight")
            print(f"\nPlot saved to: {args.output}")
        else:
            print("\nDisplaying plot (close window to exit)...")
            plt.show()

        print("\nDemo completed successfully!")
        return 0

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
