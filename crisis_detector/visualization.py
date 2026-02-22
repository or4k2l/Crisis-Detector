"""Visualization logic for Crisis Detector results."""

import logging
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_analysis(
    results: Dict,
    title: str = "Crisis Detection Analysis",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
    threshold: float = 2.5,
) -> plt.Figure:
    """
    Create a comprehensive visualization of crisis detection results.

    Args:
        results: Output dictionary from ``CrisisDetector.process_signal()``.
        title: Plot title.
        save_path: Optional file path to save the figure.
        figsize: Figure size ``(width, height)`` in inches.
        threshold: Z-score threshold used during detection (drawn as reference
            lines on the Z-score sub-plot).

    Returns:
        Matplotlib ``Figure`` object.
    """
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    signal = results["signal"]
    timestamps = results["timestamps"]
    crisis_score = results["crisis_score"]
    crisis_regions = results["crisis_regions"]
    volatility = results["volatility"]
    z_scores = results["z_scores"]

    # Plot 1: Original signal with crisis regions
    axes[0].plot(timestamps, signal, "b-", linewidth=1, alpha=0.7, label="Signal")
    if np.any(crisis_regions):
        axes[0].fill_between(
            timestamps,
            signal.min(),
            signal.max(),
            where=crisis_regions,
            alpha=0.3,
            color="red",
            label="Crisis Regions",
        )
    axes[0].set_ylabel("Signal Value")
    axes[0].set_title(title)
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Crisis score
    axes[1].plot(timestamps, crisis_score, "r-", linewidth=1.5, label="Crisis Score")
    axes[1].axhline(y=0.5, color="k", linestyle="--", alpha=0.5, label="Threshold")
    axes[1].fill_between(timestamps, 0, crisis_score, alpha=0.3, color="red")
    axes[1].set_ylabel("Crisis Score")
    axes[1].set_ylim([0, 1])
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Z-scores
    axes[2].plot(timestamps, z_scores, "g-", linewidth=1, label="Z-Score")
    axes[2].axhline(y=threshold, color="r", linestyle="--", alpha=0.5)
    axes[2].axhline(y=-threshold, color="r", linestyle="--", alpha=0.5)
    axes[2].axhline(y=0, color="k", linestyle="-", alpha=0.3)
    axes[2].set_ylabel("Z-Score")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.3)

    # Plot 4: Volatility
    axes[3].plot(timestamps, volatility, "m-", linewidth=1, label="Volatility")
    axes[3].set_ylabel("Volatility")
    axes[3].set_xlabel("Time")
    axes[3].legend(loc="best")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved to {save_path}")

    return fig
