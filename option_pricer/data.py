"""Data loading and synthetic price series generation for NIFTY 50."""
from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd


def synthetic_prices(
    start: str = "2018-01-01",
    end: str = "2024-06-30",
    s0: float = 10500.0,
    mu: float = 0.13,
    sigma: float = 0.16,
    seed: int = 42,
) -> pd.Series:
    """Generate realistic synthetic daily index prices via Geometric Brownian Motion.

    Parameters
    ----------
    start : str
        Start date 'YYYY-MM-DD'.
    end : str
        End date 'YYYY-MM-DD'.
    s0 : float
        Initial index level (e.g., 10,500 in early 2018).
    mu : float
        Annualized expected return / drift.
    sigma : float
        Annualized volatility.
    seed : int
        RNG seed for reproducible data.

    Returns
    -------
    pd.Series
        Synthetic prices with daily DatetimeIndex named 'Close'.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)
    if n == 0:
        raise ValueError("End date must be after start date.")

    dt = 1.0 / 248.0
    drift = (mu - 0.5 * sigma**2) * dt
    shocks = sigma * np.sqrt(dt) * rng.standard_normal(n)
    shocks[0] = 0.0  # start exactly at s0
    log_returns = drift + shocks
    log_returns[0] = 0.0

    prices = s0 * np.exp(np.cumsum(log_returns))
    series = pd.Series(prices, index=dates, name="Close")
    return series


def load_prices(
    start: str = "2018-01-01",
    csv_path: Optional[Union[str, Path]] = None,
    ticker: str = "^NSEI",
) -> pd.Series:
    """Load historical closing prices for NIFTY 50.

    Parameters
    ----------
    start : str
        Start date 'YYYY-MM-DD'.
    csv_path : str or Path, optional
        Path to local CSV containing 'Date' and 'Close' columns.
    ticker : str, default '^NSEI'
        Yahoo Finance ticker symbol.

    Returns
    -------
    pd.Series
        Historical close prices with DatetimeIndex named 'Close'.
    """
    # 1. Custom CSV provided
    if csv_path is not None:
        p = Path(csv_path)
        if not p.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        df = pd.read_csv(p)
        date_col = next(
            (c for c in df.columns if c.strip().lower() in ("date", "datetime", "timestamp")),
            df.columns[0],
        )
        close_col = next(
            (c for c in df.columns if c.strip().lower() in ("close", "adj close", "adj_close", "price")),
            df.columns[1] if len(df.columns) > 1 else df.columns[0],
        )
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).set_index(date_col)
        prices = df[close_col].astype(float).dropna()
        prices.name = "Close"
        if start:
            prices = prices.loc[pd.to_datetime(start):]
        return prices

    # 2. Try online download via yfinance
    try:
        import yfinance as yf
        data = yf.download(ticker, start=start, auto_adjust=True, progress=False, timeout=10)
        if data is not None and not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                if "Close" in data.columns.levels[0]:
                    prices = data["Close"].iloc[:, 0]
                else:
                    prices = data.iloc[:, 0]
            elif "Close" in data.columns:
                prices = data["Close"]
            else:
                prices = data.iloc[:, 0]
            prices = prices.astype(float).dropna()
            prices.name = "Close"
            if len(prices) > 0:
                return prices
    except Exception:
        pass

    # 3. Fallback to bundled data/nifty.csv if exists
    default_csv = Path("data/nifty.csv")
    if default_csv.exists():
        print(f"Network unavailable for '{ticker}'; using bundled {default_csv}")
        return load_prices(start=start, csv_path=default_csv)

    # 4. If neither worked, raise informative error
    raise RuntimeError(
        f"Unable to download market data for '{ticker}' and no CSV found at '{default_csv}'. "
        "Use --synthetic for offline demo or provide --csv path/to/nifty.csv."
    )
