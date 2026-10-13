"""End-to-end demo: NIFTY 50 data -> realised vol -> ATM 30-day option prices.

Usage:
    python run_analysis.py                       # download ^NSEI via yfinance
    python run_analysis.py --csv data/nifty.csv  # use a local CSV
    python run_analysis.py --synthetic           # offline demo (fake data)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from option_pricer import (call_price, put_price, call_delta, put_delta, gamma,
                           vega, put_call_parity_gap, realized_volatility,
                           load_prices)
from option_pricer.data import synthetic_prices

OUT = Path("outputs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--rate", type=float, default=0.07, help="risk-free rate (India ~6-7%%)")
    ap.add_argument("--days", type=int, default=30, help="calendar days to expiry")
    ap.add_argument("--window", type=int, default=20)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    prices = synthetic_prices() if args.synthetic else load_prices(start=args.start, csv_path=args.csv)
    label = "Synthetic index (demo)" if args.synthetic else "NIFTY 50"
    r, T = args.rate, args.days / 365

    df = prices.to_frame("S")
    df["vol"] = realized_volatility(prices, window=args.window)
    df = df.dropna()
    df["K"] = df["S"]  # at-the-money
    df["call"] = call_price(df.S, df.K, T, r, df.vol)
    df["put"] = put_price(df.S, df.K, T, r, df.vol)
    df["delta_c"] = call_delta(df.S, df.K, T, r, df.vol)
    df["delta_p"] = put_delta(df.S, df.K, T, r, df.vol)
    df["gamma"] = gamma(df.S, df.K, T, r, df.vol)
    df["vega"] = vega(df.S, df.K, T, r, df.vol) / 100  # per 1 vol point

    gap = np.abs(put_call_parity_gap(df.S, df.K, T, r, df.vol)).max()
    print(f"Max |put-call parity gap| over {len(df)} days: {gap:.2e}")
    assert gap < 1e-8, "Put-call parity violated"
    df.to_csv(OUT / "atm_option_prices.csv")
    print(df.tail(3).round(4).to_string())

    # 1. Volatility and ATM prices through time
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(df.index, df.vol * 100); ax[0].set_ylabel("20d realised vol (%)")
    ax[0].set_title(f"{label}: lagged realised volatility")
    ax[1].plot(df.index, df.call, label="ATM call"); ax[1].plot(df.index, df.put, label="ATM put")
    ax[1].set_ylabel("Option price (index pts)"); ax[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "vol_and_prices.png", dpi=130); plt.close(fig)

    # 2. Sensitivities around the latest observation
    S0, sig0 = df.S.iloc[-1], df.vol.iloc[-1]
    spots = np.linspace(0.85 * S0, 1.15 * S0, 200)
    vols = np.linspace(0.05, 0.60, 200)
    taus = np.linspace(1 / 365, 90 / 365, 200)

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax[0, 0].plot(spots, call_price(spots, S0, T, r, sig0), label="Call")
    ax[0, 0].plot(spots, put_price(spots, S0, T, r, sig0), label="Put")
    ax[0, 0].set(title="Price vs spot", xlabel="Spot"); ax[0, 0].legend()
    ax[0, 1].plot(vols * 100, call_price(S0, S0, T, r, vols), label="Call")
    ax[0, 1].plot(vols * 100, put_price(S0, S0, T, r, vols), label="Put")
    ax[0, 1].set(title="Price vs volatility", xlabel="Vol (%)"); ax[0, 1].legend()
    ax[0, 2].plot(taus * 365, call_price(S0, S0, taus, r, sig0), label="Call")
    ax[0, 2].plot(taus * 365, put_price(S0, S0, taus, r, sig0), label="Put")
    ax[0, 2].set(title="Price vs time to expiry", xlabel="Days"); ax[0, 2].legend()
    ax[1, 0].plot(spots, call_delta(spots, S0, T, r, sig0), label="Call")
    ax[1, 0].plot(spots, put_delta(spots, S0, T, r, sig0), label="Put")
    ax[1, 0].set(title="Delta vs spot", xlabel="Spot"); ax[1, 0].legend()
    ax[1, 1].plot(spots, gamma(spots, S0, T, r, sig0)); ax[1, 1].set(title="Gamma vs spot", xlabel="Spot")
    ax[1, 2].plot(spots, vega(spots, S0, T, r, sig0) / 100); ax[1, 2].set(title="Vega vs spot (per vol pt)", xlabel="Spot")
    fig.suptitle(f"{label}: option value and Greeks (K = {S0:,.0f}, vol = {sig0:.1%})")
    fig.tight_layout(); fig.savefig(OUT / "sensitivities.png", dpi=130); plt.close(fig)
    print(f"Saved results to {OUT}/")


if __name__ == "__main__":
    main()
