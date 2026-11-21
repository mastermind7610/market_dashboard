"""
features.py — THE FEATURE LAYER
=================================
Takes the clean OHLCV frame from data_loader and adds the three
signals the dashboard plots, plus a one-row regime snapshot:

    log_return    -> daily log return of close
    volatility    -> rolling annualized std of log returns
    momentum      -> trailing N-day cumulative log return

Each is one small function. build_features() wires them together so
the rest of the system imports one thing and gets a fully enriched
frame back. regime_snapshot() reads only the last warmed-up row:
that is the "where are we now" answer. The path stays on the chart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Number of trading days in a year. Used to annualize daily volatility.
# (Calendar days would be 365, but markets are closed on weekends/holidays
# so vol scales by sqrt(252), not sqrt(365). Getting this wrong is one of
# the most common bugs in junior quant code.)
TRADING_DAYS = 252

# 21 days ≈ one trading month: quick enough to see a regime change.
# 63 days ≈ one quarter, the short end of the Jegadeesh-Titman
# momentum window (3–12 months).
DEFAULT_VOL_WINDOW = 21
DEFAULT_MOM_WINDOW = 63

# Cumulative log return inside this band is too small to call a trend.
# 0.05 over the default 63-day window is about a 5% move: drift, not a
# regime. The chart still shows the exact value either way.
MOMENTUM_FLAT_BAND = 0.05

# Volatility is judged against THIS file, not a universal cutoff such
# as "20% is low". The tails of the loaded history are the quiet and
# stress regimes; the middle is typical for this stock, in this sample.
VOL_QUIET_PERCENTILE = 0.20
VOL_ELEVATED_PERCENTILE = 0.80


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


def add_volatility(df: pd.DataFrame, window: int = DEFAULT_VOL_WINDOW) -> pd.DataFrame:
    """
    Rolling annualized volatility of log returns.

    Window choice is a tradeoff:
      - 21 days  (~1 trading month) reacts quickly, but is noisier
      - 63 days  (~3 months) is smoother but lags real regime shifts
      - 252 days (~1 year) is the textbook "long-run" vol, very lagged

    Annualization: daily std * sqrt(252). Volatility scales with the
    sqrt of time under the random-walk assumption that underlies most
    pricing models. pandas rolling std is the sample std (ddof=1),
    which is the usual estimator for a window this short.
    """
    out = df.copy()
    out[f"volatility_{window}d"] = (
        out["log_return"].rolling(window).std() * np.sqrt(TRADING_DAYS)
    )
    return out


def add_momentum(df: pd.DataFrame, window: int = DEFAULT_MOM_WINDOW) -> pd.DataFrame:
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
    vol_window: int = DEFAULT_VOL_WINDOW,
    mom_window: int = DEFAULT_MOM_WINDOW,
) -> pd.DataFrame:
    """
    Apply the full feature pipeline.

    Order matters: returns first, because volatility and momentum both
    depend on them. This is the contract the plot layer will consume.
    Early NaNs are the rolling warmup. They are intentional.
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


def _trend_label(momentum: float) -> str:
    """Map a cumulative log return onto uptrend / downtrend / flat."""
    if momentum > MOMENTUM_FLAT_BAND:
        return "uptrend"
    if momentum < -MOMENTUM_FLAT_BAND:
        return "downtrend"
    return "flat"


def _volatility_label(percentile: float) -> str:
    """Map a within-sample percentile onto quiet / typical / elevated."""
    if percentile >= VOL_ELEVATED_PERCENTILE:
        return "elevated"
    if percentile <= VOL_QUIET_PERCENTILE:
        return "quiet"
    return "typical"


def regime_snapshot(df: pd.DataFrame) -> dict:
    """
    Latest risk regime a person can brief from.

    Uses the last row where volatility and momentum are both available
    (after the rolling warmup). Current vol is ranked against the same
    series, so "elevated" means elevated for this stock inside this
    file, and the peak records whether a worse episode already
    happened earlier on the chart.
    """
    vol_window = int(df.attrs.get("vol_window", DEFAULT_VOL_WINDOW))
    mom_window = int(df.attrs.get("mom_window", DEFAULT_MOM_WINDOW))
    vol_col = f"volatility_{vol_window}d"
    mom_col = f"momentum_{mom_window}d"
    needed = ["close", "log_return", vol_col, mom_col]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"Feature frame is missing {missing}.")

    ready = df.dropna(subset=needed)
    if ready.empty:
        raise ValueError(
            "Not enough history for a regime snapshot. "
            f"Need more than {max(vol_window, mom_window)} trading days "
            "after the first return."
        )

    last = ready.iloc[-1]
    vol_hist = ready[vol_col]
    vol_now = float(last[vol_col])
    # Mid-rank on ties, so a perfectly flat vol series reads as typical
    # rather than as a spike at the 100th percentile.
    percentile = float(
        (vol_hist < vol_now).mean() + 0.5 * (vol_hist == vol_now).mean()
    )
    momentum = float(last[mom_col])
    return {
        "ticker": df.attrs.get("ticker", "?"),
        "as_of": ready.index[-1],
        "history_start": df.index.min(),
        "close": float(last["close"]),
        "daily_log_return": float(last["log_return"]),
        "volatility": vol_now,
        "volatility_median": float(vol_hist.median()),
        "volatility_percentile": percentile,
        "volatility_regime": _volatility_label(percentile),
        "volatility_peak": float(vol_hist.max()),
        "volatility_peak_date": vol_hist.idxmax(),
        "momentum": momentum,
        "momentum_simple": float(np.exp(momentum) - 1.0),
        "trend": _trend_label(momentum),
        "vol_window": vol_window,
        "mom_window": mom_window,
        "n_obs": int(len(ready)),
    }


if __name__ == "__main__":
    from data_loader import load_prices

    prices = load_prices()
    feats = build_features(prices)
    snapshot = regime_snapshot(feats)

    print(f"Ticker:    {feats.attrs['ticker']}")
    print(f"Rows:      {len(feats)}")
    print(f"Columns:   {list(feats.columns)}")
    print("\nLast 5 rows (rounded):")
    print(feats.tail(5).round(4))

    print("\nFeature summary:")
    print(feats[["log_return",
                 f"volatility_{feats.attrs['vol_window']}d",
                 f"momentum_{feats.attrs['mom_window']}d"]].describe().round(4))
    print("\nLatest regime:")
    for key, value in snapshot.items():
        print(f"  {key}: {value}")
