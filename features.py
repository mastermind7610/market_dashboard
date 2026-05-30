"""
features.py — THE FEATURE LAYER
=================================
Takes the clean OHLCV frame from data_loader and adds the three
signals our dashboard actually plots:

    log_return    -> daily log return of close
    volatility    -> rolling annualized std of log returns
    momentum      -> trailing N-day cumulative log return

Each is one small function. build_features() wires them together so
the rest of the system imports one thing and gets a fully enriched
frame back.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from data_loader import load_prices

# Number of trading days in a year. Used to annualize daily volatility.
# (Calendar days would be 365, but markets are closed on weekends/holidays
# so vol scales by sqrt(252), not sqrt(365). Getting this wrong is one of
# the most common bugs in junior quant code.)
TRADING_DAYS = 252


def add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Daily log return: r_t = ln(close_t / close_{t-1}).

    Why log instead of simple (% change)?
      1. Time-additive: log returns sum cleanly across periods,
         so a 21-day return is just the rolling sum of 21 daily ones.
      2. Symmetric: a +10% then -10% sequence nets to ~0 in logs,
         but to -1% in simple returns. The log version reflects what
         actually happened to your portfolio more honestly.
      3. Closer to Gaussian in the small, which most ML/quant models
         secretly assume.
    """
    out = df.copy()
    out["log_return"] = np.log(out["close"] / out["close"].shift(1))
    return out


def add_volatility(df: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """
    Rolling annualized volatility of log returns.

    Window choice is a tradeoff:
      - 21 days  (~1 trading month) reacts quickly, but is noisier
      - 63 days  (~3 months) is smoother but lags real regime shifts
      - 252 days (~1 year) is the textbook "long-run" vol, very lagged

    Annualization: daily std * sqrt(252). Volatility scales with the
    sqrt of time under the random-walk assumption that underlies most
    pricing models.
    """
    out = df.copy()
    out[f"volatility_{window}d"] = (
        out["log_return"].rolling(window).std() * np.sqrt(TRADING_DAYS)
    )
    return out


def add_momentum(df: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """
    Trailing N-day momentum = sum of log returns over the last N days.

    This is the classic "trend" feature: positive = the stock has been
    going up recently, negative = down. We'll use it as the COLOR
    dimension in the scatter plot — letting the eye see whether high-
    vol high-return points are also trending up (potentially healthy
    trends) vs trending down (potentially capitulation).

    63 days ≈ 3 months, a common momentum lookback in academic factor
    literature (Jegadeesh-Titman 1993 used 3-12 month windows).
    """
    out = df.copy()
    out[f"momentum_{window}d"] = out["log_return"].rolling(window).sum()
    return out


def build_features(
    df: pd.DataFrame,
    vol_window: int = 21,
    mom_window: int = 63,
) -> pd.DataFrame:
    """
    Apply the full feature pipeline.

    Order matters: returns first, because volatility and momentum both
    depend on them. This is the contract the plot layer will consume.
    """
    out = add_log_returns(df)
    out = add_volatility(out, window=vol_window)
    out = add_momentum(out, window=mom_window)

    # Tag the windows in attrs so the plot layer can label axes
    # without having to be told the parameter values separately.
    out.attrs["vol_window"] = vol_window
    out.attrs["mom_window"] = mom_window
    out.attrs["ticker"] = df.attrs.get("ticker", "?")
    return out


if __name__ == "__main__":
    prices = load_prices("aapl_raw.csv")
    feats = build_features(prices)

    print(f"Ticker:    {feats.attrs['ticker']}")
    print(f"Rows:      {len(feats)}")
    print(f"Columns:   {list(feats.columns)}")
    print(f"\nLast 5 rows (rounded):")
    print(feats.tail(5).round(4))

    print(f"\nFeature summary:")
    print(feats[["log_return",
                 f"volatility_{feats.attrs['vol_window']}d",
                 f"momentum_{feats.attrs['mom_window']}d"]].describe().round(4))