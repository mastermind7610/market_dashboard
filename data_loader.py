"""
data_loader.py — THE DATA LAYER
=================================
Single responsibility: turn a raw CSV into a clean, validated,
time-indexed price series.

Everything downstream (features, plots) trusts the output of this
module completely. That trust is the whole point: parsing, sorting,
type-coercion and NaN-handling happen HERE and nowhere else.

The bundled sample (aapl_raw.csv) is the default, so the demo runs
with no API key and no working-directory tricks. A caller can still
point at another file in the same shape.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# Resolved next to this module, not against the process working directory.
# `python dashboard.py` then works after a clone even if the shell is
# sitting somewhere else.
SAMPLE_CSV = Path(__file__).resolve().parent / "aapl_raw.csv"

_CANONICAL = ("date", "open", "high", "low", "close", "volume")
_OHLC = ["open", "high", "low", "close"]
_PRICE_COLUMNS = _OHLC + ["volume"]

# Float noise from a vendor feed should not fail a bar that is
# economically a valid print. A full dollar of inversion still fails.
_OHLC_TOLERANCE = 1e-6


def load_prices(path: str | Path | None = None, ticker: str | None = None) -> pd.DataFrame:
    """
    Load raw OHLCV data and return a clean, sorted, DatetimeIndex'd frame.

    `path` defaults to the bundled Apple sample. Column names may be
    generic (`Date`, `Open`, `High`, `Low`, `Close`, `Volume`) or
    ticker-prefixed (`AAPL.Open`, ...). The prefix becomes the ticker
    label unless `ticker` is passed explicitly.

    Returns
    -------
    DataFrame indexed by date with columns:
        open, high, low, close, volume   (all numeric, no NaNs)
    plus an attribute df.attrs['ticker'] carrying the symbol.
    """
    csv_path = SAMPLE_CSV if path is None else Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Price file not found: {csv_path}. "
            "The demo default is the bundled aapl_raw.csv next to this module."
        )

    try:
        raw = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"{csv_path.name} has no rows.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Could not parse {csv_path.name}: {exc}") from exc

    mapping, inferred = _resolve_columns(list(raw.columns))
    df = raw[list(mapping)].rename(columns=mapping)

    # A monotonic DatetimeIndex is non-negotiable: returns, rolling
    # windows and event lookups all assume time runs strictly forward.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("The Date column has values that are not dates.")
    df = df.set_index("date").sort_index()

    # Anything unparseable becomes NaN so validation sees it, instead of
    # a stray string silently poisoning a return later.
    for col in _PRICE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = _validate(df)
    df.attrs["ticker"] = _ticker_label(ticker, inferred)
    return df


def _normalize_header(column: str) -> tuple[str, str | None]:
    """
    'AAPL.Open' -> ('open', 'AAPL'); 'Close' -> ('close', None).

    Extra columns (adjusted close, vendor moving averages) do not match
    a canonical name and are ignored by the caller.
    """
    text = column.strip()
    if "." in text:
        prefix, _, name = text.rpartition(".")
        return name.strip().lower(), (prefix.strip().upper() or None)
    return text.lower(), None


def _resolve_columns(columns: list[str]) -> tuple[dict[str, str], str | None]:
    """Map the file's headers onto canonical names and infer one ticker."""
    groups: dict[str, list[str]] = {name: [] for name in _CANONICAL}
    prefixes: list[str] = []
    for column in columns:
        canonical, prefix = _normalize_header(column)
        if canonical not in groups:
            continue
        groups[canonical].append(column)
        if prefix:
            prefixes.append(prefix)

    missing = [name for name in _CANONICAL if not groups[name]]
    if missing:
        found = ", ".join(columns) or "(none)"
        needed = ", ".join(missing)
        raise ValueError(
            f"CSV is missing required columns: {needed}. Found: {found}. "
            "Use Date, Open, High, Low, Close, Volume, "
            "or ticker-prefixed names such as AAPL.Close."
        )

    ambiguous = {name: cols for name, cols in groups.items() if len(cols) > 1}
    if ambiguous:
        raise ValueError(f"CSV has more than one column for {ambiguous}.")

    mapping = {groups[name][0]: name for name in _CANONICAL}
    unique_prefixes = set(prefixes)
    inferred = next(iter(unique_prefixes)) if len(unique_prefixes) == 1 else None
    return mapping, inferred


def _ticker_label(explicit: str | None, inferred: str | None) -> str:
    """An explicit label wins. Otherwise use the column prefix."""
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip().upper()
    return inferred or "UNKNOWN"


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    """Guard rails. Raise loudly on structural problems; heal small ones."""
    if df.empty:
        raise ValueError("Loaded an empty price frame.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Date index is not sorted ascending.")
    if df.index.has_duplicates:
        raise ValueError("Duplicate dates in index — data integrity issue.")

    # Heal isolated NaNs (a halted print, a blank volume cell). Dropping
    # the row would punch a hole in the evenly-spaced trading-day grid
    # that the volatility window assumes.
    n_missing = int(df[_PRICE_COLUMNS].isna().sum().sum())
    if n_missing:
        df[_PRICE_COLUMNS] = df[_PRICE_COLUMNS].ffill().bfill()
    if df[_PRICE_COLUMNS].isna().any().any():
        raise ValueError("Unrecoverable NaNs remain after fill.")
    if (df["volume"] < 0).any():
        raise ValueError("Volume contains negative values.")

    # A bar whose high is below its low cannot be plotted honestly.
    bad = (
        (df["high"] + _OHLC_TOLERANCE < df["low"])
        | (df["high"] + _OHLC_TOLERANCE < df[["open", "close"]].max(axis=1))
        | (df["low"] - _OHLC_TOLERANCE > df[["open", "close"]].min(axis=1))
    )
    if bad.any():
        raise ValueError(f"{int(bad.sum())} rows have impossible OHLC relationships.")

    return df


if __name__ == "__main__":
    prices = load_prices()
    print(f"Ticker:   {prices.attrs['ticker']}")
    print(f"Rows:     {len(prices)}")
    print(f"Range:    {prices.index.min().date()} -> {prices.index.max().date()}")
    print(f"Columns:  {list(prices.columns)}")
    print(f"NaNs:     {int(prices.isna().sum().sum())}")
    print("\nHead:")
    print(prices.head(3).round(2))
