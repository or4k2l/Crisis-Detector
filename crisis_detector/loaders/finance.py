"""Financial data loaders: load_finance_data and load_economic_data."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def load_finance_data(
    ticker: str = "^GSPC",
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
) -> pd.DataFrame | None:
    """
    Load financial time-series data using yfinance.

    Args:
        ticker: Stock ticker symbol (default: S&P 500).
        start_date: Start date in ``YYYY-MM-DD`` format.
        end_date: End date in ``YYYY-MM-DD`` format.

    Returns:
        DataFrame with financial data including OHLCV columns, or ``None``
        if an error occurs.
    """
    try:
        import yfinance as yf

        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        # Handle MultiIndex columns from newer yfinance versions (>=0.2.31)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        logger.info(f"Successfully loaded finance data for {ticker}")
        return data
    except Exception as e:
        logger.error(f"Error loading finance data: {e}", exc_info=True)
        return None


def load_economic_data(dataset: str = "macrodata") -> pd.DataFrame | None:
    """
    Load economic indicator data using statsmodels.

    Currently only ``"macrodata"`` is supported. Passing any other name will
    log a warning and still load ``macrodata``.

    Args:
        dataset: Economic indicator name. Only ``"macrodata"`` is supported.

    Returns:
        DataFrame with economic data, or ``None`` if an error occurs.
    """
    try:
        from statsmodels.datasets import macrodata

        if dataset != "macrodata":
            logger.warning(
                f"Dataset '{dataset}' is not supported; falling back to 'macrodata'."
            )
        data = macrodata.load_pandas().data
        logger.info(f"Successfully loaded economic data: {dataset}")
        return data
    except Exception as e:
        logger.error(f"Error loading economic data: {e}", exc_info=True)
        return None
