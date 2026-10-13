# Option Pricer

A Black-Scholes pricer for European calls and puts with closed-form Greeks, applied to **NIFTY 50** index data.

**Stack:** Python, NumPy, SciPy, Matplotlib (plus pandas and yfinance for data handling)

## What it does

- Prices European calls and puts with closed-form **Delta, Gamma and Vega**.
- Validates the pricer with **put-call parity** checks (call and put prices are internally consistent) and finite-difference checks on the Greeks.
- Estimates volatility from the **20-day annualised realised volatility** of NIFTY 50 daily log returns, **lagged by one day** to avoid look-ahead bias.
- Prices **at-the-money 30-day options** each day from that volatility.
- Plots how option value and Greeks respond to **volatility, spot price and time to expiry**.

NIFTY 50 index options on NSE are European-style and cash-settled, so Black-Scholes is a natural fit.

## Project structure

```
option-pricer/
├── option_pricer/
│   ├── black_scholes.py   # pricing + Greeks + parity check
│   ├── volatility.py      # lagged rolling realised volatility
│   └── data.py            # NIFTY 50 loader (yfinance / CSV) + synthetic demo data
├── tests/test_pricer.py   # parity, reference values, Greeks, look-ahead tests
├── run_analysis.py        # end-to-end pipeline and plots
├── requirements.txt
└── outputs/               # generated CSV and charts
```

## Quick start

```bash
pip install -r requirements.txt
python -m pytest                      # run the tests
python run_analysis.py                # download ^NSEI and run the analysis
python run_analysis.py --csv data/nifty.csv   # or use your own CSV (Date, Close columns)
python run_analysis.py --synthetic    # offline demo on simulated data (not real prices)
```

Options: `--rate 0.07` (risk-free rate, default 7%), `--days 30` (calendar days to expiry), `--window 20` (vol window), `--start 2018-01-01`.

Outputs are written to `outputs/`: `atm_option_prices.csv`, `vol_and_prices.png`, `sensitivities.png`.

## Usage as a library

```python
from option_pricer import call_price, put_price, call_delta, gamma, vega

S, K, T, r, sigma = 24000, 24000, 30/365, 0.07, 0.13
print(call_price(S, K, T, r, sigma), call_delta(S, K, T, r, sigma))
```

## Modelling notes

- Annualisation uses **248 trading days** (NSE), configurable in `volatility.py`.
- Risk-free rate defaults to 7% (roughly Indian T-bill / G-sec levels); override with `--rate`.
- Time to expiry uses calendar days / 365; dividend yield `q` defaults to 0.
- Vega is returned per 1.00 change in volatility by the library; the analysis script reports it per 1 vol point.
- Realised volatility is a backward-looking proxy, not implied volatility, so model prices will differ from NSE market quotes.

## License

MIT
