"""
data_loader.py — THE DATA LAYER
=================================
Turn a bundled CSV or a Yahoo Finance download into one clean price
frame. Features and charts trust this module for parsing, adjustment,
caching, and ticker checks. They do not fetch data themselves.

Live bars use yfinance with auto_adjust=True, so open, high, low, and
close are split- and dividend-adjusted together. The adjusted close is
copied into ``adj_close`` and is the only series the analysis uses.
The bundled Apple file keeps its own adjusted column. No API key.
"""

from __future__ import annotations

import json
import logging
import re
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SAMPLE_CSV = ROOT / "aapl_raw.csv"
SPY_SAMPLE = ROOT / "spy_sample.csv"
CACHE_DIR = ROOT / ".cache"

# A fresh download is reused for this long. The on-screen "fetched"
# time stays at the original download, not the moment the cache was read.
CACHE_TTL_HOURS = 12

YAHOO_SOURCE = "Yahoo Finance via yfinance (auto-adjusted OHLC)"
OFFLINE_AAPL_SOURCE = (
    "Offline demo — bundled aapl_raw.csv "
    "(plotly/datasets, AAPL.Adjusted used as the adjusted close)"
)
OFFLINE_SPY_SOURCE = (
    "Offline demo — bundled spy_sample.csv "
    "(Yahoo Finance auto-adjusted SPY bars saved 2026-11-22, not a live request)"
)

_SYMBOL = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
_CANONICAL = ("date", "open", "high", "low", "close", "volume")
_ADJUSTED_ALIASES = {"adj_close", "adjclose", "adjusted", "adj close"}
_OHLC = ["open", "high", "low", "close"]
_PRICE_COLUMNS = _OHLC + ["volume", "adj_close"]
_OHLC_TOLERANCE = 1e-6

_QUOTE_TYPES = {"EQUITY", "ETF"}


class MarketDataError(Exception):
    """A data problem the interface can show as a sentence."""


@dataclass(frozen=True)
class PriceHistory:
    """One symbol, cleaned, plus where it came from and when."""

    frame: pd.DataFrame
    ticker: str
    source: str
    fetched_at: pd.Timestamp | None
    latest_bar: pd.Timestamp
    from_cache: bool


@dataclass(frozen=True)
class SymbolHit:
    symbol: str
    name: str
    quote_type: str
    exchange: str


def normalize_symbol(raw: str) -> str:
    """
    Accept a ticker the user typed. Reject blanks and anything that is
    not a symbol shape, before a network call.
    """
    text = "" if raw is None else str(raw).strip().upper()
    if not text:
        raise MarketDataError("Enter a ticker symbol.")
    if not _SYMBOL.match(text):
        raise MarketDataError(
            f"'{raw}' is not a usable ticker. Use letters, digits, dots, "
            "or a dash, as in AAPL, BRK-B, or BRK.B."
        )
    return text


def load_prices(path: str | Path | None = None, ticker: str | None = None) -> pd.DataFrame:
    """
    Load a CSV into a sorted frame with a DatetimeIndex and columns
    open, high, low, close, volume, adj_close. No NaNs.

    ``path`` defaults to the bundled Apple file. When the file has no
    adjusted column, adj_close is set equal to close.
    """
    csv_path = SAMPLE_CSV if path is None else Path(path)
    if not csv_path.is_file():
        raise MarketDataError(f"Price file not found: {csv_path.name}.")

    try:
        raw = pd.read_csv(csv_path, comment="#")
    except pd.errors.EmptyDataError as exc:
        raise MarketDataError(f"{csv_path.name} has no rows.") from exc
    except pd.errors.ParserError as exc:
        raise MarketDataError(f"Could not parse {csv_path.name}: {exc}") from exc

    mapping, inferred = _resolve_columns(list(raw.columns))
    df = raw[list(mapping)].rename(columns=mapping)
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise MarketDataError("The Date column has values that are not dates.")
    df = df.set_index("date").sort_index()
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index)).tz_localize(None).normalize()
    df.index.name = "date"

    for col in _PRICE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = _validate(df)
    df.attrs["ticker"] = _ticker_label(ticker, inferred)
    return df


def load_offline_histories() -> tuple[PriceHistory, PriceHistory]:
    """Bundled AAPL and the saved SPY sample. No network."""
    apple = load_prices(SAMPLE_CSV, ticker="AAPL")
    spy = load_prices(SPY_SAMPLE, ticker="SPY")
    return (
        _as_history(apple, "AAPL", OFFLINE_AAPL_SOURCE, fetched_at=None, from_cache=False),
        _as_history(spy, "SPY", OFFLINE_SPY_SOURCE, fetched_at=None, from_cache=False),
    )


def fetch_history(
    symbol: str,
    *,
    refresh: bool = False,
    cache_dir: Path | None = None,
) -> PriceHistory:
    """
    Daily history for one symbol. A cached file younger than
    CACHE_TTL_HOURS is returned unless ``refresh`` is set.
    """
    ticker = normalize_symbol(symbol)
    directory = CACHE_DIR if cache_dir is None else Path(cache_dir)
    if not refresh:
        cached = _read_cache(ticker, directory)
        if cached is not None:
            return cached

    frame = _download_yahoo(ticker)
    fetched_at = pd.Timestamp(datetime.now(timezone.utc))
    _write_cache(ticker, frame, fetched_at, directory)
    return _as_history(frame, ticker, YAHOO_SOURCE, fetched_at=fetched_at, from_cache=False)


def search_symbols(query: str, limit: int = 8) -> list[SymbolHit]:
    """
    Company or fund name to a short list of equity and ETF symbols.
    Futures and other quote types are left out so the list is tradable names.
    """
    text = "" if query is None else str(query).strip()
    if len(text) < 2:
        raise MarketDataError("Type at least two letters to search.")

    try:
        import yfinance as yf

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = yf.Search(text, max_results=max(limit * 3, 10))
        quotes = list(getattr(result, "quotes", None) or [])
    except MarketDataError:
        raise
    except Exception as exc:
        raise MarketDataError(
            "Symbol search is unavailable right now. Enter a ticker directly."
        ) from exc

    hits: list[SymbolHit] = []
    seen: set[str] = set()
    for quote in quotes:
        quote_type = str(quote.get("quoteType") or "")
        if quote_type not in _QUOTE_TYPES:
            continue
        symbol = str(quote.get("symbol") or "").strip().upper()
        if not symbol or symbol in seen or not _SYMBOL.match(symbol):
            continue
        seen.add(symbol)
        name = str(quote.get("shortname") or quote.get("longname") or symbol)
        exchange = str(quote.get("exchDisp") or quote.get("exchange") or "")
        hits.append(SymbolHit(symbol, name, quote_type, exchange))
        if len(hits) >= limit:
            break
    return hits


def slice_history(frame: pd.DataFrame, start, end) -> pd.DataFrame:
    """Inclusive calendar slice. The caller checks that it is non-empty."""
    start_ts = pd.Timestamp(start).tz_localize(None).normalize()
    end_ts = pd.Timestamp(end).tz_localize(None).normalize()
    return frame.loc[(frame.index >= start_ts) & (frame.index <= end_ts)].copy()


def _download_yahoo(ticker: str) -> pd.DataFrame:
    """Full available daily history. Empty responses become MarketDataError."""
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    try:
        import yfinance as yf

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(
                ticker,
                period="max",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
    except Exception as exc:
        raise MarketDataError(
            f"Yahoo Finance did not return data for {ticker}. {exc}"
        ) from exc

    frame = _normalize_yahoo(raw, ticker)
    if frame.empty:
        raise MarketDataError(
            f"No price history for {ticker}. Check the symbol, "
            "or switch to the offline Apple demo."
        )
    return frame


def _normalize_yahoo(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    yfinance returns a MultiIndex even for one symbol, and with
    auto_adjust=True the close is already the adjusted close.
    """
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=_PRICE_COLUMNS)

    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame.columns = [str(column).strip().lower().replace(" ", "_") for column in frame.columns]
    if "adj_close" not in frame.columns:
        if "close" not in frame.columns:
            return pd.DataFrame(columns=_PRICE_COLUMNS)
        frame["adj_close"] = frame["close"]

    keep = [column for column in _PRICE_COLUMNS if column in frame.columns]
    frame = frame[keep]
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).tz_localize(None).normalize()
    frame.index.name = "date"
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    for column in _PRICE_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = _validate(frame)
    frame.attrs["ticker"] = ticker
    return frame


def _as_history(
    frame: pd.DataFrame,
    ticker: str,
    source: str,
    *,
    fetched_at: pd.Timestamp | None,
    from_cache: bool,
) -> PriceHistory:
    frame.attrs["ticker"] = ticker
    frame.attrs["source"] = source
    latest = pd.Timestamp(frame.index.max())
    return PriceHistory(
        frame=frame,
        ticker=ticker,
        source=source,
        fetched_at=None if fetched_at is None else pd.Timestamp(fetched_at),
        latest_bar=latest,
        from_cache=from_cache,
    )


def _cache_paths(ticker: str, directory: Path) -> tuple[Path, Path]:
    return directory / f"{ticker}.csv", directory / f"{ticker}.json"


def _read_cache(ticker: str, directory: Path) -> PriceHistory | None:
    csv_path, meta_path = _cache_paths(ticker, directory)
    if not csv_path.is_file() or not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        fetched_at = pd.Timestamp(meta["fetched_at"])
        age = pd.Timestamp(datetime.now(timezone.utc)) - fetched_at
        if age > pd.Timedelta(hours=CACHE_TTL_HOURS):
            return None
        frame = pd.read_csv(csv_path, index_col="date", parse_dates=True)
        frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
        frame.index.name = "date"
        for column in _PRICE_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = _validate(frame)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    return _as_history(
        frame,
        ticker,
        str(meta.get("source") or YAHOO_SOURCE),
        fetched_at=fetched_at,
        from_cache=True,
    )


def _write_cache(
    ticker: str,
    frame: pd.DataFrame,
    fetched_at: pd.Timestamp,
    directory: Path,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    csv_path, meta_path = _cache_paths(ticker, directory)
    frame.to_csv(csv_path)
    meta_path.write_text(
        json.dumps(
            {
                "ticker": ticker,
                "source": YAHOO_SOURCE,
                "fetched_at": pd.Timestamp(fetched_at).isoformat(),
                "latest_bar": pd.Timestamp(frame.index.max()).date().isoformat(),
            }
        ),
        encoding="utf-8",
    )


def _normalize_header(column: str) -> tuple[str, str | None]:
    """'AAPL.Open' -> ('open', 'AAPL'); 'Adj Close' -> ('adj_close', None)."""
    text = column.strip()
    if "." in text:
        prefix, _, name = text.rpartition(".")
        canonical = name.strip().lower().replace(" ", "_")
    else:
        prefix, canonical = None, text.lower().replace(" ", "_")
    if canonical in _ADJUSTED_ALIASES:
        canonical = "adj_close"
    ticker = prefix.strip().upper() if prefix else None
    return canonical, (ticker or None)


def _resolve_columns(columns: list[str]) -> tuple[dict[str, str], str | None]:
    wanted = set(_CANONICAL) | {"adj_close"}
    groups: dict[str, list[str]] = {name: [] for name in wanted}
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
        raise MarketDataError(
            f"CSV is missing required columns: {', '.join(missing)}. Found: {found}. "
            "Use Date, Open, High, Low, Close, Volume, "
            "or ticker-prefixed names such as AAPL.Close."
        )

    ambiguous = {name: cols for name, cols in groups.items() if len(cols) > 1}
    if ambiguous:
        raise MarketDataError(f"CSV has more than one column for {ambiguous}.")

    order = [name for name in (*_CANONICAL, "adj_close") if groups[name]]
    mapping = {groups[name][0]: name for name in order}
    unique_prefixes = set(prefixes)
    inferred = next(iter(unique_prefixes)) if len(unique_prefixes) == 1 else None
    return mapping, inferred


def _ticker_label(explicit: str | None, inferred: str | None) -> str:
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip().upper()
    return inferred or "UNKNOWN"


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    """Raise on a broken frame. Fill an isolated gap rather than drop the day."""
    if df.empty:
        raise MarketDataError("Loaded an empty price frame.")
    if not df.index.is_monotonic_increasing:
        raise MarketDataError("Date index is not sorted ascending.")
    if df.index.has_duplicates:
        raise MarketDataError("Duplicate dates in the price file.")

    n_missing = int(df[_PRICE_COLUMNS].isna().sum().sum())
    if n_missing:
        df[_PRICE_COLUMNS] = df[_PRICE_COLUMNS].ffill().bfill()
    if df[_PRICE_COLUMNS].isna().any().any():
        raise MarketDataError("Unrecoverable gaps remain after filling.")
    if (df["volume"] < 0).any():
        raise MarketDataError("Volume contains negative values.")
    if (df["adj_close"] <= 0).any() or (df["close"] <= 0).any():
        raise MarketDataError("Prices must be positive so returns are defined.")

    # The raw high/low test uses the unadjusted OHLC when the file has
    # both. Adjusted close can sit on a different scale, so it is not
    # compared with the raw high and low.
    bad = (
        (df["high"] + _OHLC_TOLERANCE < df["low"])
        | (df["high"] + _OHLC_TOLERANCE < df[_OHLC].max(axis=1))
        | (df["low"] - _OHLC_TOLERANCE > df[_OHLC].min(axis=1))
    )
    if bad.any():
        raise MarketDataError(f"{int(bad.sum())} rows have impossible OHLC relationships.")
    return df


if __name__ == "__main__":
    apple, spy = load_offline_histories()
    print(apple.ticker, apple.source)
    print(f"  rows {len(apple.frame)}  {apple.frame.index.min().date()} -> {apple.latest_bar.date()}")
    print(spy.ticker, spy.source)
    print(f"  rows {len(spy.frame)}  {spy.frame.index.min().date()} -> {spy.latest_bar.date()}")
