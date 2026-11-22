"""
features.py — THE FEATURE LAYER
=================================
Performance and risk for one selected window, plus the sentences that
describe those numbers. Every sentence is filled from a field on
PeriodStats. This module does not fetch data and does not draw charts.

Definitions, in one place so the label cannot drift from the prose:

* Cumulative return is end adjusted close / start adjusted close − 1.
* Annualized volatility is the sample standard deviation (N−1) of daily
  log returns in the window, times sqrt(252).
* Maximum drawdown is the lowest close-to-prior-peak decline inside the
  window. The peak is the highest adjusted close on or before the trough.
* Moving averages are simple averages of adjusted close. They are built
  on the full loaded history, then shown on the selected dates, so a
  200-day average can appear inside a shorter window.
* The volatility label ranks the window's annualized volatility against
  this symbol's own rolling 21-day annualized volatilities on the full
  loaded history. Quiet is a percentile rank of 20 or below. Elevated is
  80 or above. Between them is typical. The rank counts the share of
  baseline windows strictly below the period vol, plus half the ties.
  Fewer than 30 baseline windows means no label.
* The benchmark gap uses an inner join of dates. Both sides are adjusted
  closes. It is the difference of the two cumulative simple returns, in
  percentage points, not a regression alpha.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MA_SHORT = 50
MA_LONG = 200
ROLLING_VOL_WINDOW = 21
VOL_QUIET_PERCENTILE = 0.20
VOL_ELEVATED_PERCENTILE = 0.80
MIN_RETURNS_FOR_VOL = 2
MIN_BASELINE_WINDOWS = 30
MIN_PRICES = 2

PRESETS = ("1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "Max")
_PRESET_MONTHS = {"1M": 1, "3M": 3, "6M": 6, "1Y": 12, "3Y": 36, "5Y": 60}


class AnalysisError(ValueError):
    """The selected window cannot support a briefing."""


@dataclass(frozen=True)
class PeriodStats:
    ticker: str
    benchmark: str | None
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    n_days: int
    first_price: float
    last_price: float
    cumulative_return: float
    annualized_volatility: float | None
    n_returns: int
    max_drawdown: float | None
    drawdown_peak: pd.Timestamp | None
    drawdown_trough: pd.Timestamp | None
    drawdown_peak_price: float | None
    drawdown_trough_price: float | None
    vol_percentile: float | None
    vol_label: str | None
    vol_baseline_count: int
    ma_short: float | None
    ma_long: float | None
    benchmark_cumulative_return: float | None
    aligned_stock_return: float | None
    excess_return: float | None
    aligned_days: int
    dropped_stock_sessions: int


@dataclass(frozen=True)
class ChartData:
    price: pd.Series
    ma_short: pd.Series
    ma_long: pd.Series
    drawdown: pd.Series
    rolling_vol: pd.Series
    aligned: pd.DataFrame | None


@dataclass(frozen=True)
class PeriodReport:
    stats: PeriodStats
    charts: ChartData
    sentences: tuple[str, ...]
    notes: tuple[str, ...]


def methodology_notes() -> tuple[str, ...]:
    """The rules, with the same constants the calculations use."""
    return (
        "Cumulative return is the adjusted close on the last session of the "
        "window divided by the adjusted close on the first session, minus one.",
        "Annualized volatility is the sample standard deviation, with divisor "
        f"N−1, of the daily log returns inside the window, multiplied by the "
        f"square root of {TRADING_DAYS}. Log returns are the natural log of "
        "today's adjusted close over yesterday's. At least "
        f"{MIN_RETURNS_FOR_VOL} daily returns are required.",
        "Maximum drawdown is the deepest decline from a prior peak to a later "
        "close inside the window. The peak is the highest adjusted close on "
        "or before the trough. A window that never closes below a prior peak "
        "has a drawdown of zero.",
        f"The {MA_SHORT}-day and {MA_LONG}-day lines are simple moving averages "
        "of adjusted close. Each average is computed on the full loaded history "
        "and then restricted to the selected dates, so an average can be drawn "
        "at the start of a short window when earlier bars were loaded.",
        volatility_rule_text(),
        "The benchmark comparison keeps only dates present in both series. "
        "Both prices are adjusted closes. Each cumulative return runs from the "
        "first shared session to the last. The gap is the stock's cumulative "
        "return minus the benchmark's, reported in percentage points. It is "
        "not a regression alpha and it is not compared with the VIX.",
    )


def volatility_rule_text() -> str:
    quiet = f"{VOL_QUIET_PERCENTILE:.0%}"
    elevated = f"{VOL_ELEVATED_PERCENTILE:.0%}"
    return (
        f"The volatility label ranks the selected window's annualized volatility "
        f"against this symbol's rolling {ROLLING_VOL_WINDOW}-day annualized "
        f"volatilities on the full loaded history, not only inside the window. "
        f"Quiet means a percentile rank of {quiet} or below. Elevated means "
        f"{elevated} or above. A rank between those cuts is typical. The rank "
        f"is the share of baseline windows strictly below the period volatility, "
        f"plus half the share that tie with it. The label is withheld when fewer "
        f"than {MIN_BASELINE_WINDOWS} baseline windows exist."
    )


def metric_help() -> dict[str, str]:
    """One line per headline number, drawn from the same definitions."""
    return {
        "cumulative_return": methodology_notes()[0],
        "annualized_volatility": (
            f"Sample standard deviation of daily log returns, times sqrt({TRADING_DAYS})."
        ),
        "max_drawdown": "Deepest close-to-prior-peak decline inside the selected window.",
        "excess_return": (
            "Stock cumulative return minus benchmark cumulative return, "
            "on shared sessions only, in percentage points."
        ),
    }


def window_for_preset(index: pd.DatetimeIndex, preset: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Calendar window ending on the last loaded bar."""
    if preset not in PRESETS:
        raise AnalysisError(f"Unknown range '{preset}'.")
    last = pd.Timestamp(index.max()).normalize()
    first = pd.Timestamp(index.min()).normalize()
    if preset == "Max":
        start = first
    elif preset == "YTD":
        start = pd.Timestamp(year=int(last.year), month=1, day=1)
    else:
        start = last - pd.DateOffset(months=_PRESET_MONTHS[preset])
    if start < first:
        start = first
    return start, last


def resolve_window(index: pd.DatetimeIndex, start, end) -> tuple[pd.Timestamp, pd.Timestamp, tuple[str, ...]]:
    """
    Clamp a requested calendar range to the loaded bars and say when
    the clamp moved a boundary.
    """
    first = pd.Timestamp(index.min()).normalize()
    last = pd.Timestamp(index.max()).normalize()
    start_ts = pd.Timestamp(start).tz_localize(None).normalize()
    end_ts = pd.Timestamp(end).tz_localize(None).normalize()
    if end_ts < start_ts:
        raise AnalysisError("The end date is before the start date.")
    if end_ts < first or start_ts > last:
        raise AnalysisError(
            f"No trading days between {start_ts.date()} and {end_ts.date()}. "
            f"Loaded history runs from {first.date()} to {last.date()}."
        )
    notes: list[str] = []
    if start_ts < first:
        notes.append(
            f"The requested start {start_ts.date()} is before the first loaded bar, "
            f"so the window begins on {first.date()}."
        )
        start_ts = first
    if end_ts > last:
        notes.append(
            f"The requested end {end_ts.date()} is after the last loaded bar, "
            f"so the window ends on {last.date()}."
        )
        end_ts = last
    return start_ts, end_ts, tuple(notes)


def cumulative_return(prices: pd.Series) -> float:
    """End price over start price, minus one. Prices must be positive."""
    if len(prices) < MIN_PRICES:
        raise AnalysisError("A cumulative return needs two adjusted closes.")
    start = float(prices.iloc[0])
    end = float(prices.iloc[-1])
    if start <= 0 or end <= 0:
        raise AnalysisError("Adjusted closes must be positive to compute a return.")
    return end / start - 1.0


def daily_log_returns(prices: pd.Series) -> pd.Series:
    """r_t = ln(P_t / P_{t-1}). The first observation is NaN by construction."""
    return np.log(prices.astype(float) / prices.astype(float).shift(1))


def annualized_volatility(log_returns: pd.Series) -> float:
    """
    Sample daily standard deviation annualized with sqrt(252).
    Requires at least two finite returns; one return has no sample spread.
    """
    clean = pd.Series(log_returns).replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < MIN_RETURNS_FOR_VOL:
        raise AnalysisError(
            f"Annualized volatility needs at least {MIN_RETURNS_FOR_VOL} daily returns."
        )
    return float(clean.std(ddof=1) * np.sqrt(TRADING_DAYS))


def max_drawdown(prices: pd.Series) -> dict:
    """
    Deepest peak-to-trough decline inside ``prices``.

    The running peak is the maximum adjusted close up to each day. The
    trough is the day that ratio is smallest. The peak date is the day
    that running peak was last set on or before the trough.
    """
    series = prices.astype(float).dropna()
    if series.empty:
        raise AnalysisError("Maximum drawdown needs a price series.")
    running_peak = series.cummax()
    drawdown = series / running_peak - 1.0
    trough = drawdown.idxmin()
    depth = float(drawdown.loc[trough])
    peak_level = float(running_peak.loc[trough])
    at_peak = series.loc[:trough] >= peak_level - 1e-9
    peak = series.loc[:trough].loc[at_peak].index[-1]
    return {
        "max_drawdown": depth,
        "peak": pd.Timestamp(peak),
        "trough": pd.Timestamp(trough),
        "peak_price": float(series.loc[peak]),
        "trough_price": float(series.loc[trough]),
        "series": drawdown,
    }


def moving_average(prices: pd.Series, window: int) -> pd.Series:
    """Simple moving average. The first window−1 values stay missing."""
    if window < MIN_PRICES:
        raise AnalysisError("A moving-average window must be at least 2.")
    return prices.astype(float).rolling(window, min_periods=window).mean()


def rolling_annualized_volatility(log_returns: pd.Series, window: int = ROLLING_VOL_WINDOW) -> pd.Series:
    """Trailing annualized sample volatility. Warmup rows stay NaN."""
    return log_returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(TRADING_DAYS)


def percentile_rank(value: float, sample: pd.Series) -> float:
    """Mid-rank: share strictly below, plus half the ties."""
    clean = pd.Series(sample).dropna()
    if clean.empty:
        raise AnalysisError("A percentile rank needs a baseline sample.")
    below = float((clean < value).mean())
    tied = float((clean == value).mean())
    return below + 0.5 * tied


def volatility_label(percentile: float) -> str:
    if percentile >= VOL_ELEVATED_PERCENTILE:
        return "elevated"
    if percentile <= VOL_QUIET_PERCENTILE:
        return "quiet"
    return "typical"


def align_adjusted(stock: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    """
    Inner join on the timestamp. No forward fill: a session missing on
    either side is left out of the comparison rather than invented.
    """
    frame = pd.concat(
        {"stock": stock.astype(float), "benchmark": benchmark.astype(float)},
        axis=1,
        join="inner",
    ).dropna()
    frame = frame.sort_index()
    return frame[~frame.index.duplicated(keep="last")]


def build_period(
    history: pd.DataFrame,
    *,
    ticker: str,
    start,
    end,
    benchmark: pd.DataFrame | None = None,
    benchmark_ticker: str | None = None,
) -> PeriodReport:
    """Stats, chart series, and sentences for one window of ``history``."""
    prices = _adjusted(history)
    start_ts, end_ts, clamp_notes = resolve_window(prices.index, start, end)
    window = prices.loc[start_ts:end_ts].dropna()
    if len(window) < MIN_PRICES:
        raise AnalysisError(
            f"{ticker} has {len(window)} trading day(s) between {start_ts.date()} "
            f"and {end_ts.date()}. A return needs two adjusted closes. "
            "Widen the range."
        )

    # Averages and the volatility baseline use the whole loaded file.
    # The headline return, drawdown, and period vol use only the window.
    full_returns = daily_log_returns(prices)
    baseline = rolling_annualized_volatility(full_returns).dropna()
    ma_short_full = moving_average(prices, MA_SHORT)
    ma_long_full = moving_average(prices, MA_LONG)

    period_returns = daily_log_returns(window).dropna()
    period_vol = (
        annualized_volatility(period_returns)
        if len(period_returns) >= MIN_RETURNS_FOR_VOL
        else None
    )
    vol_percentile = None
    vol_label = None
    if period_vol is not None and len(baseline) >= MIN_BASELINE_WINDOWS:
        vol_percentile = percentile_rank(period_vol, baseline)
        vol_label = volatility_label(vol_percentile)

    drawdown = max_drawdown(window)
    # The printed average is the value on the last session of the window.
    # It stays blank while that session is still inside the warmup.
    last_day = window.index[-1]
    last_ma_short = _value_on(ma_short_full, last_day)
    last_ma_long = _value_on(ma_long_full, last_day)

    aligned = None
    bench_return = None
    aligned_stock_return = None
    excess = None
    aligned_days = 0
    dropped = 0
    if benchmark is not None and benchmark_ticker:
        bench_px = _adjusted(benchmark)
        aligned = align_adjusted(window, bench_px)
        aligned_days = int(len(aligned))
        dropped = int(len(window) - aligned_days)
        if aligned_days >= MIN_PRICES:
            aligned_stock_return = cumulative_return(aligned["stock"])
            bench_return = cumulative_return(aligned["benchmark"])
            excess = aligned_stock_return - bench_return

    stats = PeriodStats(
        ticker=ticker,
        benchmark=benchmark_ticker,
        window_start=pd.Timestamp(window.index[0]),
        window_end=pd.Timestamp(window.index[-1]),
        n_days=int(len(window)),
        first_price=float(window.iloc[0]),
        last_price=float(window.iloc[-1]),
        cumulative_return=cumulative_return(window),
        annualized_volatility=period_vol,
        n_returns=int(len(period_returns)),
        max_drawdown=float(drawdown["max_drawdown"]),
        drawdown_peak=drawdown["peak"],
        drawdown_trough=drawdown["trough"],
        drawdown_peak_price=float(drawdown["peak_price"]),
        drawdown_trough_price=float(drawdown["trough_price"]),
        vol_percentile=vol_percentile,
        vol_label=vol_label,
        vol_baseline_count=int(len(baseline)),
        ma_short=last_ma_short,
        ma_long=last_ma_long,
        benchmark_cumulative_return=bench_return,
        aligned_stock_return=aligned_stock_return,
        excess_return=excess,
        aligned_days=aligned_days,
        dropped_stock_sessions=dropped,
    )
    charts = ChartData(
        price=window,
        ma_short=ma_short_full.reindex(window.index),
        ma_long=ma_long_full.reindex(window.index),
        drawdown=drawdown["series"],
        rolling_vol=rolling_annualized_volatility(full_returns).reindex(window.index),
        aligned=aligned,
    )
    sentences = clamp_notes + describe_period(stats)
    return PeriodReport(stats=stats, charts=charts, sentences=sentences, notes=clamp_notes)


def describe_period(stats: PeriodStats) -> tuple[str, ...]:
    """Plain-English sentences. Each number is a field on ``stats``."""
    start = stats.window_start.date().isoformat()
    end = stats.window_end.date().isoformat()
    lines = [
        (
            f"{stats.ticker} adjusted close moved from {stats.first_price:.2f} on {start} "
            f"to {stats.last_price:.2f} on {end}: a cumulative return of "
            f"{stats.cumulative_return:+.1%} across {stats.n_days} trading days."
        )
    ]
    lines.append(_volatility_sentence(stats))
    lines.append(_drawdown_sentence(stats))
    lines.append(_average_sentence(stats))
    lines.append(_benchmark_sentence(stats))
    return tuple(line for line in lines if line)


def _volatility_sentence(stats: PeriodStats) -> str:
    if stats.annualized_volatility is None:
        return (
            f"Annualized volatility is omitted: the window has {stats.n_returns} daily "
            f"return(s), and the sample standard deviation needs at least {MIN_RETURNS_FOR_VOL}."
        )
    text = (
        f"Annualized volatility is {stats.annualized_volatility:.1%}, "
        f"from {stats.n_returns} daily log returns."
    )
    if stats.n_returns < ROLLING_VOL_WINDOW:
        text += (
            f" That uses fewer than {ROLLING_VOL_WINDOW} returns, so the annualized "
            "figure is a short sample."
        )
    if stats.vol_label is None or stats.vol_percentile is None:
        text += (
            f" No quiet/typical/elevated label is attached: the loaded history has "
            f"{stats.vol_baseline_count} rolling {ROLLING_VOL_WINDOW}-day windows and "
            f"the label needs at least {MIN_BASELINE_WINDOWS}."
        )
        return text
    rank = stats.vol_percentile * 100
    text += (
        f" The label is {stats.vol_label}: this {stats.annualized_volatility:.1%} sits at "
        f"percentile rank {rank:.1f} of the {stats.vol_baseline_count} rolling "
        f"{ROLLING_VOL_WINDOW}-day volatilities in the loaded history. "
        f"Quiet is {VOL_QUIET_PERCENTILE:.0%} or below, elevated is "
        f"{VOL_ELEVATED_PERCENTILE:.0%} or above, and the rank between them is typical."
    )
    return text


def _drawdown_sentence(stats: PeriodStats) -> str:
    if stats.max_drawdown is None or stats.drawdown_peak is None or stats.drawdown_trough is None:
        return ""
    if abs(stats.max_drawdown) < 1e-12:
        return (
            "Maximum drawdown is 0.0%: the adjusted close never finished below "
            "a prior peak inside this window."
        )
    return (
        f"Maximum drawdown is {stats.max_drawdown:.1%}, from a peak of "
        f"{stats.drawdown_peak_price:.2f} on {stats.drawdown_peak.date().isoformat()} "
        f"to a trough of {stats.drawdown_trough_price:.2f} on "
        f"{stats.drawdown_trough.date().isoformat()}."
    )


def _average_sentence(stats: PeriodStats) -> str:
    parts: list[str] = []
    if stats.ma_short is None:
        parts.append(
            f"The {MA_SHORT}-day average is not available on {stats.window_end.date().isoformat()}: "
            f"that average needs {MA_SHORT} earlier adjusted closes."
        )
    else:
        relation = _relation(stats.last_price, stats.ma_short)
        parts.append(
            f"On {stats.window_end.date().isoformat()} the close is {relation} the "
            f"{MA_SHORT}-day average of {stats.ma_short:.2f}."
        )
    if stats.ma_long is None:
        parts.append(
            f"The {MA_LONG}-day average is not available on {stats.window_end.date().isoformat()}: "
            f"that average needs {MA_LONG} earlier adjusted closes."
        )
    else:
        relation = _relation(stats.last_price, stats.ma_long)
        parts.append(
            f"The close is {relation} the {MA_LONG}-day average of {stats.ma_long:.2f}."
        )
    return " ".join(parts)


def _benchmark_sentence(stats: PeriodStats) -> str:
    if not stats.benchmark:
        return ""
    if stats.aligned_days < MIN_PRICES or stats.benchmark_cumulative_return is None:
        return (
            f"The {stats.benchmark} comparison is omitted: only {stats.aligned_days} "
            "shared session(s), and a return needs two."
        )
    gap = float(stats.excess_return)
    points = gap * 100
    # 0.05 percentage points prints as 0.0 at one decimal place.
    if abs(points) < 0.05:
        text = (
            f"On the {stats.aligned_days} shared sessions, {stats.ticker} and "
            f"{stats.benchmark} both returned {stats.aligned_stock_return:+.1%}."
        )
    else:
        direction = "higher" if gap > 0 else "lower"
        text = (
            f"On the {stats.aligned_days} shared sessions, {stats.ticker} returned "
            f"{stats.aligned_stock_return:+.1%} and {stats.benchmark} returned "
            f"{stats.benchmark_cumulative_return:+.1%}. {stats.ticker} is "
            f"{abs(points):.1f} percentage points {direction}."
        )
    if stats.dropped_stock_sessions == 0:
        text += " Every session in the stock window has a benchmark print."
    else:
        text += (
            f" {stats.dropped_stock_sessions} {stats.ticker} session(s) have no "
            f"{stats.benchmark} print and are left out of the comparison."
        )
    return text


def _relation(price: float, average: float) -> str:
    if price > average:
        return "above"
    if price < average:
        return "below"
    return "equal to"


def _adjusted(frame: pd.DataFrame) -> pd.Series:
    if "adj_close" not in frame.columns:
        raise AnalysisError("The price frame has no adjusted close.")
    series = frame["adj_close"].astype(float).sort_index()
    series = series[~series.index.duplicated(keep="last")].dropna()
    if (series <= 0).any():
        raise AnalysisError("Adjusted closes must be positive.")
    return series


def _value_on(series: pd.Series, when: pd.Timestamp) -> float | None:
    if when not in series.index or pd.isna(series.loc[when]):
        return None
    return float(series.loc[when])


if __name__ == "__main__":
    from data_loader import load_offline_histories

    apple, spy = load_offline_histories()
    start, end = window_for_preset(apple.frame.index, "Max")
    report = build_period(
        apple.frame,
        ticker="AAPL",
        start=start,
        end=end,
        benchmark=spy.frame,
        benchmark_ticker="SPY",
    )
    for sentence in report.sentences:
        print(sentence)
    print("---")
    for note in methodology_notes():
        print(note)
