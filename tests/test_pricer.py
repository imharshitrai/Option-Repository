"""Unit tests for option_pricer: put-call parity, reference values, Greeks, and look-ahead bias."""
import numpy as np
import pandas as pd
import pytest

from option_pricer import (
    call_price,
    put_price,
    call_delta,
    put_delta,
    gamma,
    vega,
    put_call_parity_gap,
    realized_volatility,
    synthetic_prices,
)


def test_put_call_parity_scalar():
    """Verify Put-Call parity gap is negligible for scalars."""
    S, K, T, r, sigma = 24000.0, 24000.0, 30.0 / 365.0, 0.07, 0.13
    gap = put_call_parity_gap(S, K, T, r, sigma)
    assert abs(gap) < 1e-10


def test_put_call_parity_vector():
    """Verify Put-Call parity gap is negligible across arrays."""
    spots = np.linspace(15000, 30000, 50)
    strikes = np.linspace(15000, 30000, 50)
    T = 45.0 / 365.0
    r = 0.065
    vols = np.linspace(0.10, 0.40, 50)

    gap = put_call_parity_gap(spots, strikes, T, r, vols)
    assert np.max(np.abs(gap)) < 1e-10


def test_reference_values_standard():
    """Verify prices against standard Black-Scholes benchmark (Hull)."""
    # S=100, K=100, T=1, r=0.05, sigma=0.20
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    c = call_price(S, K, T, r, sigma)
    p = put_price(S, K, T, r, sigma)

    # Standard analytical reference values: C = 10.45058..., P = 5.57353...
    assert pytest.approx(10.45058, abs=1e-4) == c
    assert pytest.approx(5.57353, abs=1e-4) == p


def test_reference_values_nifty():
    """Verify prices for typical NIFTY 50 values (README example)."""
    S, K, T, r, sigma = 24000.0, 24000.0, 30.0 / 365.0, 0.07, 0.13
    c = call_price(S, K, T, r, sigma)
    d_c = call_delta(S, K, T, r, sigma)
    d_p = put_delta(S, K, T, r, sigma)

    assert c > 0
    assert 0.5 < d_c < 0.6  # slightly ITM forward due to interest rate
    assert -0.5 < d_p < -0.4
    assert pytest.approx(d_c - 1.0, abs=1e-10) == d_p


def test_greeks_finite_difference():
    """Verify closed-form Greeks match central finite difference approximations."""
    S, K, T, r, sigma = 24000.0, 24200.0, 30.0 / 365.0, 0.07, 0.15

    # 1. Delta: dC/dS
    h_s = 1e-3
    fd_delta_c = (call_price(S + h_s, K, T, r, sigma) - call_price(S - h_s, K, T, r, sigma)) / (2 * h_s)
    fd_delta_p = (put_price(S + h_s, K, T, r, sigma) - put_price(S - h_s, K, T, r, sigma)) / (2 * h_s)
    assert pytest.approx(call_delta(S, K, T, r, sigma), rel=1e-4) == fd_delta_c
    assert pytest.approx(put_delta(S, K, T, r, sigma), rel=1e-4) == fd_delta_p

    # 2. Gamma: d2C/dS2 = dDelta/dS
    fd_gamma = (call_delta(S + h_s, K, T, r, sigma) - call_delta(S - h_s, K, T, r, sigma)) / (2 * h_s)
    assert pytest.approx(gamma(S, K, T, r, sigma), rel=1e-4) == fd_gamma

    # 3. Vega: dC/dsigma
    h_vol = 1e-4
    fd_vega = (call_price(S, K, T, r, sigma + h_vol) - call_price(S, K, T, r, sigma - h_vol)) / (2 * h_vol)
    assert pytest.approx(vega(S, K, T, r, sigma), rel=1e-4) == fd_vega


def test_look_ahead_bias():
    """Ensure that volatility calculation on day t uses only information up to day t-1."""
    prices = synthetic_prices(start="2023-01-01", end="2023-06-30", seed=10)
    vol_original = realized_volatility(prices, window=20, lag=1)

    # Mutate price at index 50
    mutated_prices = prices.copy()
    mutated_prices.iloc[50] = mutated_prices.iloc[50] * 1.5
    vol_mutated = realized_volatility(mutated_prices, window=20, lag=1)

    # Values up to day 50 should be completely identical (no look-ahead)
    pd.testing.assert_series_equal(
        vol_original.iloc[:51],
        vol_mutated.iloc[:51],
        check_exact=True,
    )

    # Value on day 51 must reflect the shock at day 50
    assert vol_original.iloc[51] != vol_mutated.iloc[51]


def test_synthetic_prices():
    """Verify synthetic price generation properties."""
    prices = synthetic_prices(start="2023-01-01", end="2023-12-31")
    assert isinstance(prices, pd.Series)
    assert len(prices) > 200
    assert (prices > 0).all()
    assert isinstance(prices.index, pd.DatetimeIndex)
