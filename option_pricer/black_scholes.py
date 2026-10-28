"""Black-Scholes pricing, closed-form Greeks, and put-call parity checks."""
from typing import Any, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import norm


def _d1_d2(
    S: Union[float, np.ndarray, pd.Series],
    K: Union[float, np.ndarray, pd.Series],
    T: Union[float, np.ndarray, pd.Series],
    r: Union[float, np.ndarray, pd.Series],
    sigma: Union[float, np.ndarray, pd.Series],
    q: Union[float, np.ndarray, pd.Series] = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate d1 and d2 for Black-Scholes formula.

    Returns (d1, d2, sqrt_T, valid_mask).
    """
    S_arr = np.asarray(S, dtype=float)
    K_arr = np.asarray(K, dtype=float)
    T_arr = np.asarray(T, dtype=float)
    r_arr = np.asarray(r, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)
    q_arr = np.asarray(q, dtype=float)

    # Valid when time to expiry > 0 and volatility > 0
    valid = (T_arr > 0) & (sigma_arr > 0) & (S_arr > 0) & (K_arr > 0)
    sqrt_T = np.sqrt(np.maximum(T_arr, 0.0))
    vol_sqrt_T = np.where(valid, sigma_arr * sqrt_T, 1.0)

    # Numerator of d1
    num = np.log(np.maximum(S_arr, 1e-12) / np.maximum(K_arr, 1e-12)) + (
        r_arr - q_arr + 0.5 * sigma_arr**2
    ) * T_arr
    d1 = np.where(valid, num / vol_sqrt_T, 0.0)
    d2 = np.where(valid, d1 - sigma_arr * sqrt_T, 0.0)

    return d1, d2, sqrt_T, valid


def _wrap_output(
    val: np.ndarray,
    *inputs: Any,
) -> Union[float, np.ndarray, pd.Series]:
    """Preserve input types (float scalar or pd.Series if provided)."""
    if all(np.isscalar(x) for x in inputs):
        return float(np.asarray(val).item())
    for x in inputs:
        if isinstance(x, pd.Series):
            return pd.Series(val, index=x.index)
    return val


def call_price(
    S: Union[float, np.ndarray, pd.Series],
    K: Union[float, np.ndarray, pd.Series],
    T: Union[float, np.ndarray, pd.Series],
    r: Union[float, np.ndarray, pd.Series],
    sigma: Union[float, np.ndarray, pd.Series],
    q: Union[float, np.ndarray, pd.Series] = 0.0,
) -> Union[float, np.ndarray, pd.Series]:
    """Calculate Black-Scholes European call option price."""
    d1, d2, _, valid = _d1_d2(S, K, T, r, sigma, q)
    S_arr = np.asarray(S, dtype=float)
    K_arr = np.asarray(K, dtype=float)
    T_arr = np.asarray(T, dtype=float)
    r_arr = np.asarray(r, dtype=float)
    q_arr = np.asarray(q, dtype=float)

    disc_r = np.exp(-r_arr * T_arr)
    disc_q = np.exp(-q_arr * T_arr)

    price = disc_q * S_arr * norm.cdf(d1) - disc_r * K_arr * norm.cdf(d2)
    # Intrinsic payoff when expired (T <= 0)
    expired_payoff = np.maximum(S_arr - K_arr, 0.0)
    res = np.where(valid, price, expired_payoff)
    return _wrap_output(res, S, K, T, r, sigma, q)


def put_price(
    S: Union[float, np.ndarray, pd.Series],
    K: Union[float, np.ndarray, pd.Series],
    T: Union[float, np.ndarray, pd.Series],
    r: Union[float, np.ndarray, pd.Series],
    sigma: Union[float, np.ndarray, pd.Series],
    q: Union[float, np.ndarray, pd.Series] = 0.0,
) -> Union[float, np.ndarray, pd.Series]:
    """Calculate Black-Scholes European put option price."""
    d1, d2, _, valid = _d1_d2(S, K, T, r, sigma, q)
    S_arr = np.asarray(S, dtype=float)
    K_arr = np.asarray(K, dtype=float)
    T_arr = np.asarray(T, dtype=float)
    r_arr = np.asarray(r, dtype=float)
    q_arr = np.asarray(q, dtype=float)

    disc_r = np.exp(-r_arr * T_arr)
    disc_q = np.exp(-q_arr * T_arr)

    price = disc_r * K_arr * norm.cdf(-d2) - disc_q * S_arr * norm.cdf(-d1)
    # Intrinsic payoff when expired (T <= 0)
    expired_payoff = np.maximum(K_arr - S_arr, 0.0)
    res = np.where(valid, price, expired_payoff)
    return _wrap_output(res, S, K, T, r, sigma, q)


def call_delta(
    S: Union[float, np.ndarray, pd.Series],
    K: Union[float, np.ndarray, pd.Series],
    T: Union[float, np.ndarray, pd.Series],
    r: Union[float, np.ndarray, pd.Series],
    sigma: Union[float, np.ndarray, pd.Series],
    q: Union[float, np.ndarray, pd.Series] = 0.0,
) -> Union[float, np.ndarray, pd.Series]:
    """Calculate European call option Delta (dC/dS)."""
    d1, _, _, valid = _d1_d2(S, K, T, r, sigma, q)
    S_arr = np.asarray(S, dtype=float)
    K_arr = np.asarray(K, dtype=float)
    T_arr = np.asarray(T, dtype=float)
    q_arr = np.asarray(q, dtype=float)

    delta = np.exp(-q_arr * T_arr) * norm.cdf(d1)
    expired_delta = np.where(S_arr > K_arr, 1.0, np.where(S_arr == K_arr, 0.5, 0.0))
    res = np.where(valid, delta, expired_delta)
    return _wrap_output(res, S, K, T, r, sigma, q)


def put_delta(
    S: Union[float, np.ndarray, pd.Series],
    K: Union[float, np.ndarray, pd.Series],
    T: Union[float, np.ndarray, pd.Series],
    r: Union[float, np.ndarray, pd.Series],
    sigma: Union[float, np.ndarray, pd.Series],
    q: Union[float, np.ndarray, pd.Series] = 0.0,
) -> Union[float, np.ndarray, pd.Series]:
    """Calculate European put option Delta (dP/dS)."""
    d1, _, _, valid = _d1_d2(S, K, T, r, sigma, q)
    S_arr = np.asarray(S, dtype=float)
    K_arr = np.asarray(K, dtype=float)
    T_arr = np.asarray(T, dtype=float)
    q_arr = np.asarray(q, dtype=float)

    delta = np.exp(-q_arr * T_arr) * (norm.cdf(d1) - 1.0)
    expired_delta = np.where(S_arr < K_arr, -1.0, np.where(S_arr == K_arr, -0.5, 0.0))
    res = np.where(valid, delta, expired_delta)
    return _wrap_output(res, S, K, T, r, sigma, q)


def gamma(
    S: Union[float, np.ndarray, pd.Series],
    K: Union[float, np.ndarray, pd.Series],
    T: Union[float, np.ndarray, pd.Series],
    r: Union[float, np.ndarray, pd.Series],
    sigma: Union[float, np.ndarray, pd.Series],
    q: Union[float, np.ndarray, pd.Series] = 0.0,
) -> Union[float, np.ndarray, pd.Series]:
    """Calculate European option Gamma (d2C/dS2 = d2P/dS2)."""
    d1, _, sqrt_T, valid = _d1_d2(S, K, T, r, sigma, q)
    S_arr = np.asarray(S, dtype=float)
    T_arr = np.asarray(T, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)
    q_arr = np.asarray(q, dtype=float)

    denom = np.where(valid, S_arr * sigma_arr * sqrt_T, 1.0)
    g = np.where(valid, np.exp(-q_arr * T_arr) * norm.pdf(d1) / denom, 0.0)
    return _wrap_output(g, S, K, T, r, sigma, q)


def vega(
    S: Union[float, np.ndarray, pd.Series],
    K: Union[float, np.ndarray, pd.Series],
    T: Union[float, np.ndarray, pd.Series],
    r: Union[float, np.ndarray, pd.Series],
    sigma: Union[float, np.ndarray, pd.Series],
    q: Union[float, np.ndarray, pd.Series] = 0.0,
) -> Union[float, np.ndarray, pd.Series]:
    """Calculate European option Vega (dC/dsigma = dP/dsigma).

    Note: Vega is expressed per 1.00 (100%) change in volatility.
    Divide by 100 for vega per 1 vol point.
    """
    d1, _, sqrt_T, valid = _d1_d2(S, K, T, r, sigma, q)
    S_arr = np.asarray(S, dtype=float)
    T_arr = np.asarray(T, dtype=float)
    q_arr = np.asarray(q, dtype=float)

    v = np.where(valid, S_arr * np.exp(-q_arr * T_arr) * norm.pdf(d1) * sqrt_T, 0.0)
    return _wrap_output(v, S, K, T, r, sigma, q)


def put_call_parity_gap(
    S: Union[float, np.ndarray, pd.Series],
    K: Union[float, np.ndarray, pd.Series],
    T: Union[float, np.ndarray, pd.Series],
    r: Union[float, np.ndarray, pd.Series],
    sigma: Union[float, np.ndarray, pd.Series],
    q: Union[float, np.ndarray, pd.Series] = 0.0,
) -> Union[float, np.ndarray, pd.Series]:
    """Calculate the Put-Call Parity discrepancy: C - P - (S*e^(-qT) - K*e^(-rT)).

    In theoretical Black-Scholes, this evaluates to 0 within machine epsilon.
    """
    c = call_price(S, K, T, r, sigma, q)
    p = put_price(S, K, T, r, sigma, q)
    S_arr = np.asarray(S, dtype=float)
    K_arr = np.asarray(K, dtype=float)
    T_arr = np.asarray(T, dtype=float)
    r_arr = np.asarray(r, dtype=float)
    q_arr = np.asarray(q, dtype=float)

    parity_synthetic = S_arr * np.exp(-q_arr * T_arr) - K_arr * np.exp(-r_arr * T_arr)
    gap = c - p - parity_synthetic
    return _wrap_output(gap, S, K, T, r, sigma, q)
