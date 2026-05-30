"""
data_loader.py — THE DATA LAYER
=================================
Single responsibility: turn a raw CSV into a clean, validated,
time-indexed price series.

Everything downstream (features, plots) trusts the output of this
module completely. That trust is the whole point: parsing, sorting,
type-coercion and NaN-handling happen HERE and nowhere else.
"""

from __future__ import annotations
import pandas as pd


# The raw file uses ticker-prefixed columns (AAPL.Close, ...).
# We map them to clean, source-agnostic names so the rest of the
# system never knows or cares which ticker this is.
_COLUMN_MAP = {
    "Date": "date",
    "AAPL.Open": "open",
    "AAPL.High": "high",
    "AAPL.Low": "low",
    "AAPL.Close": "close",
    "AAPL.Volume": "volume",
}

_OHLC = ["open", "high", "low", "close"]


def load_prices(path: str, ticker: str = "AAPL") -> pd.DataFrame:
    """
    Load raw OHLCV data and return a clean, sorted, DatetimeIndex'd frame.

    Returns
    -------
    DataFrame indexed by date with columns:
        open, high, low, close, volume   (all numeric, no NaNs)
    plus an attribute df.attrs['ticker'] carrying the symbol.
    """
    raw = pd.read_csv(path)

    # 1. Select + rename only the columns we trust. Drop the rest
    #    (moving-average columns 'dn/mavg/up' belong to the feature
    #    layer's job, not the data layer's — we recompute our own).
    df = raw[list(_COLUMN_MAP)].rename(columns=_COLUMN_MAP)

    # 2. Parse dates and make them the index. A monotonic DatetimeIndex
    #    is non-negotiable: returns, rolling windows and event lookups
    #    all assume time runs strictly forward.
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # 3. Coerce to numeric. Anything unparseable becomes NaN so it
    #    gets caught by the validation step instead of silently
    #    poisoning a calculation later.
    for col in _OHLC + ["volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Validate + clean. We forward-fill small gaps (a halted/holiday
    #    tick) rather than dropping rows, because dropping rows breaks
    #    the assumption of evenly-spaced trading days that volatility
    #    windows rely on.
    df = _validate(df)

    df.attrs["ticker"] = ticker
    return df


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    """Guard rails. Raise loudly on structural problems; heal small ones."""
    if df.empty:
        raise ValueError("Loaded an empty price frame.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Date index is not sorted ascending.")
    if df.index.has_duplicates:
        raise ValueError("Duplicate dates in index — data integrity issue.")

    # Heal isolated NaNs; fail if a whole column is missing.
    n_missing = int(df[_OHLC].isna().sum().sum())
    if n_missing:
        df[_OHLC] = df[_OHLC].ffill().bfill()
    if df[_OHLC].isna().any().any():
        raise ValueError("Unrecoverable NaNs remain after fill.")

    return df


if __name__ == "__main__":
    prices = load_prices("aapl_raw.csv")
    print(f"Ticker:   {prices.attrs['ticker']}")
    print(f"Rows:     {len(prices)}")
    print(f"Range:    {prices.index.min().date()} -> {prices.index.max().date()}")
    print(f"Columns:  {list(prices.columns)}")
    print(f"NaNs:     {int(prices.isna().sum().sum())}")
    print("\nHead:")
    print(prices.head(3).round(2))
