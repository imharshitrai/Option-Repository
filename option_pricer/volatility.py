"""Realized volatility estimation with lag to prevent look-ahead bias."""
from typing import Union
import numpy as np
import pandas as pd

# Annualization factor for NSE (typically 248 trading days)
DEFAULT_TRADING_DAYS = 248


def realized_volatility(
    prices: Union[pd.Series, pd.DataFrame, np.ndarray],
    window: int = 20,
    annual_trading_days: int = DEFAULT_TRADING_DAYS,
    lag: int = 1,
) -> Union[pd.Series, np.ndarray]:
    """Calculate rolling annualized realized volatility from daily prices.

    The volatility is estimated from daily log returns:
        r_t = ln(P_t / P_{t-1})
        vol_t = std(r_{t-window+1:t}) * sqrt(annual_trading_days)

    By default, lag=1 shifts the resulting volatility by 1 observation so that
    pricing at time t uses only information available up to time t-1,
    avoiding look-ahead bias.

    Parameters
    ----------
    prices : pd.Series or np.ndarray
        Historical price series.
    window : int, default 20
        Lookback window in trading days.
    annual_trading_days : int, default 248
        Annual trading days (248 for NSE).
    lag : int, default 1
        Number of periods to shift forward to eliminate look-ahead bias.

    Returns
    -------
    pd.Series or np.ndarray
        Lagged rolling annualized volatility.
    """
    if isinstance(prices, pd.Series):
        log_ret = np.log(prices / prices.shift(1))
        rolling_std = log_ret.rolling(window=window).std(ddof=1)
        ann_vol = rolling_std * np.sqrt(annual_trading_days)
        if lag > 0:
            ann_vol = ann_vol.shift(lag)
        return ann_vol
    elif isinstance(prices, pd.DataFrame):
        log_ret = np.log(prices / prices.shift(1))
        rolling_std = log_ret.rolling(window=window).std(ddof=1)
        ann_vol = rolling_std * np.sqrt(annual_trading_days)
        if lag > 0:
            ann_vol = ann_vol.shift(lag)
        return ann_vol
    else:
        # np.ndarray fallback
        s = pd.Series(prices)
        res = realized_volatility(
            s, window=window, annual_trading_days=annual_trading_days, lag=lag
        )
        return res.values
