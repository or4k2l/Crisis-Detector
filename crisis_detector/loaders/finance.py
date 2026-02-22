"""Financial data loaders: load_finance_data and load_economic_data."""

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def load_finance_data(
    ticker: str = "^GSPC",
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
) -> Optional[pd.DataFrame]:
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
        logger.info(f"Successfully loaded finance data for {ticker}")
        return data
    except Exception as e:
        logger.error(f"Error loading finance data: {e}", exc_info=True)
        return None


def load_economic_data(dataset: str = "GDP") -> Optional[pd.DataFrame]:
    """
    Load economic indicator data using statsmodels.

    Args:
        dataset: Economic indicator name.

    Returns:
        DataFrame with economic data, or ``None`` if an error occurs.
    """
    try:
        from statsmodels.datasets import macrodata

        data = macrodata.load_pandas().data
        logger.info(f"Successfully loaded economic data: {dataset}")
        return data
    except Exception as e:
        logger.error(f"Error loading economic data: {e}", exc_info=True)
        return None
