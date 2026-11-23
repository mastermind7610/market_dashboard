"""
features.py — THE FEATURE LAYER
=================================
Performance and risk for one selected window, a side-by-side comparison
of several symbols, and the sentences that describe those numbers. Every
sentence is filled from a field on PeriodStats. This module does not
fetch data and does not draw charts.

Definitions, in one place so the label cannot drift from the prose:

* Cumulative return is end adjusted close / start adjusted close − 1.
* Annualized return (CAGR) is (1 + cumulative return) raised to
  365.25 / calendar days in the window, minus one. It is withheld for
  windows shorter than 360 calendar days.
* Annualized volatility is the sample standard deviation (N−1) of daily
  log returns in the window, times sqrt(252).
* Sharpe ratio is the mean daily simple return above the daily risk-free
  rate, divided by the sample standard deviation of those excess
  returns, times sqrt(252).
* Sortino ratio has the same numerator. The denominator is the downside
  deviation: the root mean square of the below-rate days, with every
  day in the divisor.
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
* Beta is the sample covariance of the stock's and the benchmark's daily
  simple returns on shared sessions, divided by the benchmark's sample
  variance. Correlation is the Pearson correlation of the same pairs.
* Up capture is the stock's mean daily return on days the benchmark
  rose, divided by the benchmark's mean on those days. Down capture is
  the same ratio on days the benchmark fell.
* Monthly returns chain inside the window: each month's last adjusted
  close over the previous month's last, with the window's first close
  as the base of the first month. Compounded, they equal the cumulative
  return.
* 95% historical Value at Risk is the loss magnitude of the 5th
  percentile of daily simple returns in the window. A VaR of 2.1%
  means that on 95% of days the return was better than −2.1%.
* Calendar-year returns are the cumulative return from the first close
  in each calendar year inside the window to the last close of that
  year (or the window end for the final partial year).
* A portfolio is a weighted mix of the selected symbols on shared
  sessions. Weights are normalized to sum to one. With no rebalance,
  the mix drifts with prices after the first day. Monthly or quarterly
  rebalance resets to the target weights on the first shared session
  of each period.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252
DAYS_PER_YEAR = 365.25
MA_SHORT = 50
MA_LONG = 200
MIN_MA_WINDOW = 2
MAX_MA_WINDOW = 400
ROLLING_VOL_WINDOW = 21
# About three months of sessions: long enough for a stable regression
# slope, short enough that the line still moves when the relationship does.
ROLLING_BETA_WINDOW = 63
ROLLING_SHARPE_WINDOW = ROLLING_BETA_WINDOW
VOL_QUIET_PERCENTILE = 0.20
VOL_ELEVATED_PERCENTILE = 0.80
MIN_RETURNS_FOR_VOL = 2
MIN_BASELINE_WINDOWS = 30
MIN_PRICES = 2
# Ratios built from a mean daily return are noise on a handful of days.
# Twenty returns is one month of sessions.
MIN_RETURNS_FOR_RATIOS = 20
# Capture ratios average each side separately, so each side needs its own sample.
MIN_CAPTURE_DAYS = 10
# A one-year preset starts on the first session on or after the anchor
# date, so it can span a few days less than 365. Anything shorter than
# this is a part-year, and compounding a part-year into a yearly rate
# reports a return nobody earned.
MIN_ANNUALIZE_CALENDAR_DAYS = 360
DEFAULT_RISK_FREE = 0.0
RISK_FREE_MIN = 0.0
RISK_FREE_MAX = 0.20
BETA_HIGH = 1.2
BETA_LOW = 0.8
CORRELATION_STRONG = 0.7
CORRELATION_WEAK = 0.3
REBASE_LEVEL = 100.0
_EPSILON = 1e-12
# Historical VaR: the left-tail percentile of daily simple returns.
VAR_CONFIDENCE = 0.95
VAR_PERCENTILE = 1.0 - VAR_CONFIDENCE
MIN_RETURNS_FOR_VAR = 20
PORTFOLIO_LABEL = "Portfolio"
REBALANCE_NONE = "none"
REBALANCE_MONTHLY = "monthly"
REBALANCE_QUARTERLY = "quarterly"
REBALANCE_OPTIONS = (REBALANCE_NONE, REBALANCE_MONTHLY, REBALANCE_QUARTERLY)
# Weights that sum within half a percentage point are treated as intentional
# and renormalized. Larger gaps get the same treatment with an explicit note.
WEIGHT_SUM_TOLERANCE = 0.005
# Transaction cost as a fraction of traded notional. 10 bps = 0.001.
DEFAULT_COST_BPS = 0.0
COST_BPS_MIN = 0.0
COST_BPS_MAX = 100.0
BPS_PER_UNIT = 10_000.0

PRESETS = ("1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "Max")
_PRESET_MONTHS = {"1M": 1, "3M": 3, "6M": 6, "1Y": 12, "3Y": 36, "5Y": 60}

# Column name -> Python format string. The page and the Markdown summary
# both print the comparison table through this one mapping.
TABLE_FORMATS = {
    "Return": "{:+.1%}",
    "Annualized return": "{:+.1%}",
    "Volatility": "{:.1%}",
    "Max drawdown": "{:.1%}",
    "Sharpe": "{:.2f}",
    "Sortino": "{:.2f}",
    "Beta": "{:.2f}",
    "Correlation": "{:.2f}",
    "Up capture": "{:.0%}",
    "Down capture": "{:.0%}",
    "Vs benchmark (pp)": "{:+.1f}",
    "Sessions": "{:.0f}",
}
MISSING = "—"


class AnalysisError(ValueError):
    """The selected window cannot support a briefing."""


@dataclass(frozen=True)
class PeriodStats:
    ticker: str
    benchmark: str | None
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    n_days: int
    calendar_days: int
    first_price: float
    last_price: float
    cumulative_return: float
    annualized_return: float | None
    annualized_volatility: float | None
    n_returns: int
    risk_free_rate: float
    sharpe: float | None
    sortino: float | None
    max_drawdown: float | None
    drawdown_peak: pd.Timestamp | None
    drawdown_trough: pd.Timestamp | None
    drawdown_peak_price: float | None
    drawdown_trough_price: float | None
    best_day: pd.Timestamp | None
    best_day_return: float | None
    worst_day: pd.Timestamp | None
    worst_day_return: float | None
    positive_day_share: float | None
    vol_percentile: float | None
    vol_label: str | None
    vol_baseline_count: int
    ma_short_window: int
    ma_long_window: int
    ma_short: float | None
    ma_long: float | None
    benchmark_cumulative_return: float | None
    aligned_stock_return: float | None
    excess_return: float | None
    aligned_days: int
    dropped_stock_sessions: int
    return_pairs: int
    beta: float | None
    beta_label: str | None
    correlation: float | None
    correlation_label: str | None
    up_capture: float | None
    down_capture: float | None
    up_days: int
    down_days: int
    value_at_risk_95: float | None
    parametric_var_95: float | None
    cvar_95: float | None
    alpha: float | None
    drawdown_days_to_trough: int | None
    drawdown_days_to_recovery: int | None
    drawdown_recovered: bool | None
    longest_underwater_days: int | None


@dataclass(frozen=True)
class ChartData:
    price: pd.Series
    ohlc: pd.DataFrame
    ma_short: pd.Series
    ma_long: pd.Series
    drawdown: pd.Series
    rolling_vol: pd.Series
    returns: pd.Series
    benchmark_returns: pd.Series | None
    monthly: pd.Series
    rolling_beta: pd.Series | None
    rolling_correlation: pd.Series | None
    aligned: pd.DataFrame | None
    calendar_years: pd.Series
    rolling_sharpe: pd.Series | None = None
    rolling_sortino: pd.Series | None = None


@dataclass(frozen=True)
class PeriodReport:
    stats: PeriodStats
    charts: ChartData
    sentences: tuple[str, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class DiversificationStats:
    """How a weight mix spreads risk across the selected symbols."""

    weights: dict[str, float]
    diversification_ratio: float | None
    effective_n: float | None
    risk_contribution: dict[str, float]
    portfolio_volatility: float | None
    weighted_avg_volatility: float | None


@dataclass(frozen=True)
class PortfolioCostReport:
    """Turnover and fees from a cost-aware portfolio path."""

    cost_bps: float
    total_turnover: float
    total_cost_drag: float
    rebalance_count: int
    average_turnover: float


@dataclass(frozen=True)
class Comparison:
    """Several symbols over one requested range, optional portfolio, plus the benchmark."""

    rows: tuple[PeriodStats, ...]
    benchmark: str | None
    rebased: pd.DataFrame | None
    sentences: tuple[str, ...]
    portfolio: str | None = None
    correlation_matrix: pd.DataFrame | None = None
    diversification: DiversificationStats | None = None


# ---------------------------------------------------------------------------
# Definitions shown on the page
# ---------------------------------------------------------------------------


def methodology_notes(ma_short: int = MA_SHORT, ma_long: int = MA_LONG) -> tuple[str, ...]:
    """The rules, with the same constants the calculations use."""
    return (
        "Cumulative return is the adjusted close on the last session of the "
        "window divided by the adjusted close on the first session, minus one.",
        "Annualized return (CAGR) is one plus the cumulative return, raised to "
        f"{DAYS_PER_YEAR} divided by the calendar days between the first and "
        "last session, minus one. It is withheld when the window spans fewer "
        f"than {MIN_ANNUALIZE_CALENDAR_DAYS} calendar days, because compounding "
        "a few months into a yearly rate reports a return nobody earned.",
        "Annualized volatility is the sample standard deviation, with divisor "
        f"N−1, of the daily log returns inside the window, multiplied by the "
        f"square root of {TRADING_DAYS}. Log returns are the natural log of "
        "today's adjusted close over yesterday's. At least "
        f"{MIN_RETURNS_FOR_VOL} daily returns are required.",
        "The Sharpe ratio is the mean daily simple return above the daily "
        "risk-free rate, divided by the sample standard deviation of those "
        f"excess returns, multiplied by the square root of {TRADING_DAYS}. The "
        "daily risk-free rate is one plus the annual rate, raised to "
        f"1/{TRADING_DAYS}, minus one. The annual rate is the sidebar setting.",
        "The Sortino ratio uses the same numerator as Sharpe. Its denominator "
        "is the downside deviation: the square root of the mean squared "
        "shortfall below the risk-free rate, where days at or above the rate "
        "count as zero and every day is in the divisor. Both ratios need at "
        f"least {MIN_RETURNS_FOR_RATIOS} daily returns.",
        "Maximum drawdown is the deepest decline from a prior peak to a later "
        "close inside the window. The peak is the highest adjusted close on "
        "or before the trough. Days to trough count trading sessions from that "
        "peak to the trough; days to recovery count from the trough to the first "
        "later close at or above the peak price (omitted when unrecovered). "
        "Longest underwater is the longest run of sessions with price below the "
        "running peak. A window that never closes below a prior peak has a "
        "drawdown of zero.",
        "Best day and worst day are the largest and smallest daily simple "
        "returns inside the window. The share of gains is the fraction of "
        "daily returns above zero.",
        f"The {ma_short}-day and {ma_long}-day lines are simple moving averages "
        "of adjusted close. Each average is computed on the full loaded history "
        "and then restricted to the selected dates, so an average can be drawn "
        "at the start of a short window when earlier bars were loaded.",
        volatility_rule_text(),
        "The benchmark comparison keeps only dates present in both series. "
        "Both prices are adjusted closes. Each cumulative return runs from the "
        "first shared session to the last. The gap is the stock's cumulative "
        "return minus the benchmark's, reported in percentage points. It is "
        "not a regression alpha and it is not compared with the VIX.",
        beta_rule_text(),
        correlation_rule_text(),
        "Up capture is the stock's mean daily return on the days the benchmark "
        "rose, divided by the benchmark's mean daily return on those days. Down "
        "capture is the same ratio on the days the benchmark fell. 100% means "
        "the stock's average move matched the benchmark's on that kind of day. "
        f"Each side needs at least {MIN_CAPTURE_DAYS} days. This is a daily "
        "arithmetic version, not the monthly compounded figure fund fact "
        "sheets print.",
        f"Rolling beta and rolling correlation use the trailing "
        f"{ROLLING_BETA_WINDOW} shared daily returns, computed on the full "
        "loaded history and then shown on the selected dates.",
        "Monthly returns chain inside the window: each calendar month's last "
        "adjusted close over the previous month's last, with the window's first "
        "close as the base of the first month. The first and last months can be "
        "partial. Compounded, the monthly figures equal the cumulative return.",
        "The side-by-side table gives each symbol its own sessions inside the "
        "requested range. The rebased chart starts every line at "
        f"{REBASE_LEVEL:.0f} on the first session all of them share.",
        f"95% historical Value at Risk is minus the {VAR_PERCENTILE:.0%} "
        "percentile of daily simple returns in the window. It needs at least "
        f"{MIN_RETURNS_FOR_VAR} daily returns. A VaR of 2% means that on "
        f"{VAR_CONFIDENCE:.0%} of days the return was better than −2%.",
        f"95% parametric (Gaussian) VaR uses the sample mean μ and sample "
        f"standard deviation σ of daily simple returns: −(μ + z σ), floored at "
        f"zero, where z is the standard normal inverse CDF at "
        f"{1.0 - VAR_CONFIDENCE:.0%}. Same minimum return count as historical VaR.",
        f"95% Conditional VaR (expected shortfall) is minus the mean of daily "
        f"simple returns at or below the historical VaR quantile. It answers "
        f"how bad the average loss was on the worst "
        f"{1.0 - VAR_CONFIDENCE:.0%} of days.",
        "Jensen's alpha is annualised as daily_alpha × "
        f"{TRADING_DAYS}. Daily alpha is the average stock excess return minus "
        "beta times the average benchmark excess return, using the same daily "
        "risk-free rate as Sharpe. It needs the same shared-return count as beta.",
        f"Rolling Sharpe and Sortino use the trailing {ROLLING_SHARPE_WINDOW} "
        "daily simple returns of the focus symbol, annualised the same way as "
        "the full-window ratios.",
        "The correlation heatmap is the Pearson correlation matrix of daily "
        "simple returns across the selected symbols on shared sessions.",
        "Diversification ratio is the weighted average of standalone "
        "volatilities divided by portfolio volatility (both from the sample "
        "covariance of daily returns). Effective number of holdings is 1 / Σ w_i². "
        "Risk contribution for holding i is its share of portfolio variance: "
        "w_i (Σw)_i / σ_p².",
        "Transaction cost is charged in basis points of turnover at inception "
        "and at each rebalance. Turnover for a reset is half the L1 distance "
        "between pre-trade weights and the target. Portfolio value is scaled by "
        "(1 − turnover × bps / 10_000). Buy-and-hold charges only the opening trade.",
        "Calendar-year returns chain inside each year of the window: the last "
        "close of that year (or the window end) over the first close of that "
        "year (or the window start for a partial first year), minus one.",
        "Portfolio weights are fractions of capital that sum to one. Equal "
        "weight is the default. Weights that do not sum to one are "
        "renormalized and named in a note. A zero-weight symbol is left out "
        "of the mix. With no rebalance the mix drifts after day one; monthly "
        "or quarterly rebalance resets to the target weights on the first "
        "shared session of each period.",
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


def beta_rule_text() -> str:
    return (
        "Beta is the sample covariance of the stock's and the benchmark's daily "
        "simple returns on shared sessions, divided by the benchmark's sample "
        f"variance. The label is high at {BETA_HIGH:.2f} or above, low from zero "
        f"to {BETA_LOW:.2f}, inverse below zero, and market-like between "
        f"{BETA_LOW:.2f} and {BETA_HIGH:.2f}. It needs at least "
        f"{MIN_RETURNS_FOR_RATIOS} shared daily returns."
    )


def correlation_rule_text() -> str:
    return (
        "Correlation is the Pearson correlation of the same daily return pairs. "
        f"The label is strong when its size is {CORRELATION_STRONG:.2f} or more, "
        f"weak when it is below {CORRELATION_WEAK:.2f}, and moderate between "
        "them. A negative value is labeled negative."
    )


def metric_help() -> dict[str, str]:
    """One line per headline number, drawn from the same definitions."""
    notes = methodology_notes()
    return {
        "cumulative_return": notes[0],
        "annualized_return": notes[1],
        "annualized_volatility": (
            f"Sample standard deviation of daily log returns, times sqrt({TRADING_DAYS})."
        ),
        "sharpe": notes[3],
        "sortino": notes[4],
        "max_drawdown": "Deepest close-to-prior-peak decline inside the selected window.",
        "excess_return": (
            "Stock cumulative return minus benchmark cumulative return, "
            "on shared sessions only, in percentage points."
        ),
        "beta": beta_rule_text(),
        "correlation": correlation_rule_text(),
        "value_at_risk_95": (
            f"Loss magnitude of the {VAR_PERCENTILE:.0%} percentile of daily "
            f"simple returns (95% historical VaR)."
        ),
        # Keep these help strings ASCII-plain: some Streamlit front ends drop
        # the tooltip icon when the help text contains special math glyphs.
        "parametric_var_95": (
            "Gaussian VaR from the sample mean and standard deviation of daily "
            "simple returns: -(mean + z * std), floored at zero."
        ),
        "cvar_95": (
            "Conditional VaR / expected shortfall: average loss on days at or "
            "below the historical VaR quantile."
        ),
        "alpha": (
            "Annualised Jensen's alpha versus the benchmark: 252 * "
            "(average stock excess return - beta * average benchmark excess return)."
        ),
        "drawdown_days_to_trough": (
            "Trading sessions from the peak of the deepest drawdown to its trough."
        ),
        "drawdown_days_to_recovery": (
            "Trading sessions from that trough to the first later close at or "
            "above the peak price. Blank when the drawdown is unrecovered."
        ),
    }


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


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
            f"The range starts on {start_ts.date()}, before this symbol's first loaded bar, "
            f"so its window begins on {first.date()}."
        )
        start_ts = first
    if end_ts > last:
        notes.append(
            f"The range ends on {end_ts.date()}, after this symbol's last loaded bar, "
            f"so its window ends on {last.date()}."
        )
        end_ts = last
    return start_ts, end_ts, tuple(notes)


def validate_average_windows(ma_short: int, ma_long: int) -> tuple[int, int]:
    """Two whole-number windows, the short one strictly shorter."""
    try:
        short, long = int(ma_short), int(ma_long)
    except (TypeError, ValueError) as exc:
        raise AnalysisError("Moving-average windows must be whole numbers.") from exc
    if short < MIN_MA_WINDOW or long > MAX_MA_WINDOW:
        raise AnalysisError(
            f"Moving-average windows must be between {MIN_MA_WINDOW} and {MAX_MA_WINDOW} days."
        )
    if short >= long:
        raise AnalysisError(
            f"The short average ({short} days) must be shorter than the long average ({long} days)."
        )
    return short, long


def validate_risk_free(rate: float) -> float:
    """An annual rate as a fraction, inside the range the page offers."""
    try:
        value = float(rate)
    except (TypeError, ValueError) as exc:
        raise AnalysisError("The risk-free rate must be a number.") from exc
    if not np.isfinite(value) or value < RISK_FREE_MIN or value > RISK_FREE_MAX:
        raise AnalysisError(
            f"The risk-free rate must be between {RISK_FREE_MIN:.0%} and {RISK_FREE_MAX:.0%} a year."
        )
    return value


# ---------------------------------------------------------------------------
# Single-series calculations
# ---------------------------------------------------------------------------


def cumulative_return(prices: pd.Series) -> float:
    """End price over start price, minus one. Prices must be positive."""
    if len(prices) < MIN_PRICES:
        raise AnalysisError("A cumulative return needs two adjusted closes.")
    start = float(prices.iloc[0])
    end = float(prices.iloc[-1])
    if start <= 0 or end <= 0:
        raise AnalysisError("Adjusted closes must be positive to compute a return.")
    return end / start - 1.0


def annualized_return(prices: pd.Series) -> float:
    """
    Compound annual growth rate by calendar time.

    Calendar days, not sessions, are the exponent: a year of holding is
    a year whether it had 250 sessions or 253.
    """
    total = cumulative_return(prices)
    days = _calendar_days(prices.index)
    if days < MIN_ANNUALIZE_CALENDAR_DAYS:
        raise AnalysisError(
            f"An annualized return needs at least {MIN_ANNUALIZE_CALENDAR_DAYS} calendar days."
        )
    return float((1.0 + total) ** (DAYS_PER_YEAR / days) - 1.0)


def daily_log_returns(prices: pd.Series) -> pd.Series:
    """r_t = ln(P_t / P_{t-1}). The first observation is NaN by construction."""
    return np.log(prices.astype(float) / prices.astype(float).shift(1))


def daily_simple_returns(prices: pd.Series) -> pd.Series:
    """R_t = P_t / P_{t-1} − 1. The first observation is NaN by construction."""
    series = prices.astype(float)
    return series / series.shift(1) - 1.0


def annualized_volatility(log_returns: pd.Series) -> float:
    """
    Sample daily standard deviation annualized with sqrt(252).
    Requires at least two finite returns; one return has no sample spread.
    """
    clean = _finite(log_returns)
    if len(clean) < MIN_RETURNS_FOR_VOL:
        raise AnalysisError(
            f"Annualized volatility needs at least {MIN_RETURNS_FOR_VOL} daily returns."
        )
    return float(clean.std(ddof=1) * np.sqrt(TRADING_DAYS))


def daily_risk_free(annual_rate: float) -> float:
    """The daily rate that compounds to ``annual_rate`` over 252 sessions."""
    return float((1.0 + annual_rate) ** (1.0 / TRADING_DAYS) - 1.0)


def sharpe_ratio(returns: pd.Series, risk_free: float = DEFAULT_RISK_FREE) -> float:
    """
    Mean excess daily return over its sample standard deviation,
    annualized with sqrt(252). Undefined when the returns do not vary.
    """
    excess = _excess_returns(returns, risk_free)
    spread = float(excess.std(ddof=1))
    if spread <= _EPSILON:
        raise AnalysisError("The Sharpe ratio is undefined when daily returns do not vary.")
    return float(excess.mean() / spread * np.sqrt(TRADING_DAYS))


def downside_deviation(returns: pd.Series, risk_free: float = DEFAULT_RISK_FREE) -> float:
    """
    Root mean square of the below-rate days. Days at or above the rate
    count as zero and stay in the divisor, so a window with few bad days
    is not penalized as if every day had been one.
    """
    excess = _excess_returns(returns, risk_free)
    shortfall = excess.clip(upper=0.0)
    return float(np.sqrt((shortfall**2).mean()))


def sortino_ratio(returns: pd.Series, risk_free: float = DEFAULT_RISK_FREE) -> float:
    """Mean excess daily return over the downside deviation, times sqrt(252)."""
    excess = _excess_returns(returns, risk_free)
    downside = downside_deviation(returns, risk_free)
    if downside <= _EPSILON:
        raise AnalysisError("The Sortino ratio is undefined when no day falls below the risk-free rate.")
    return float(excess.mean() / downside * np.sqrt(TRADING_DAYS))


def max_drawdown(prices: pd.Series) -> dict:
    """
    Deepest peak-to-trough decline inside ``prices``, with recovery timing.

    The running peak is the maximum adjusted close up to each day. The
    trough is the day that ratio is smallest. The peak date is the day
    that running peak was last set on or before the trough. Recovery is
    the first later session that closes at or above the peak price.
    Longest underwater is the longest run of sessions with a negative
    drawdown (price below its running peak).
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
    peak_price = float(series.loc[peak])
    peak_pos = int(series.index.get_loc(peak))
    trough_pos = int(series.index.get_loc(trough))
    if isinstance(peak_pos, slice):
        peak_pos = peak_pos.start
    if isinstance(trough_pos, slice):
        trough_pos = trough_pos.start
    days_to_trough = max(trough_pos - peak_pos, 0)
    after = series.iloc[trough_pos:]
    recovered_mask = after >= peak_price - 1e-9
    recovered = after.loc[recovered_mask]
    if abs(depth) < _EPSILON:
        recovery = trough
        days_to_recovery = 0
        did_recover = True
    elif recovered.empty:
        recovery = None
        days_to_recovery = None
        did_recover = False
    else:
        # Skip the trough itself when it is still below the peak.
        first_idx = recovered.index[0]
        if first_idx == trough and len(recovered) > 1:
            first_idx = recovered.index[1]
        elif first_idx == trough:
            recovery = None
            days_to_recovery = None
            did_recover = False
            first_idx = None
        if first_idx is not None:
            recovery = first_idx
            recovery_pos = int(series.index.get_loc(recovery))
            if isinstance(recovery_pos, slice):
                recovery_pos = recovery_pos.start
            days_to_recovery = max(recovery_pos - trough_pos, 0)
            did_recover = True
    underwater = (drawdown < -_EPSILON).astype(int)
    longest = 0
    run = 0
    for flag in underwater.to_numpy():
        if flag:
            run += 1
            if run > longest:
                longest = run
        else:
            run = 0
    return {
        "max_drawdown": depth,
        "peak": pd.Timestamp(peak) if isinstance(peak, pd.Timestamp) or hasattr(peak, "date") else peak,
        "trough": pd.Timestamp(trough) if isinstance(trough, pd.Timestamp) or hasattr(trough, "date") else trough,
        "peak_price": peak_price,
        "trough_price": float(series.loc[trough]),
        "days_to_trough": days_to_trough,
        "days_to_recovery": days_to_recovery,
        "recovered": did_recover,
        "recovery": recovery,
        "longest_underwater_days": int(longest),
        "series": drawdown,
    }


def extreme_days(returns: pd.Series) -> dict:
    """The largest and smallest daily return, each with its date."""
    clean = _finite(returns)
    if clean.empty:
        raise AnalysisError("Best and worst day need at least one daily return.")
    best = clean.idxmax()
    worst = clean.idxmin()
    return {
        "best_day": pd.Timestamp(best),
        "best_day_return": float(clean.loc[best]),
        "worst_day": pd.Timestamp(worst),
        "worst_day_return": float(clean.loc[worst]),
    }


def positive_share(returns: pd.Series) -> float:
    """Fraction of daily returns strictly above zero."""
    clean = _finite(returns)
    if clean.empty:
        raise AnalysisError("The share of gains needs at least one daily return.")
    return float((clean > 0).mean())


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


def monthly_returns(prices: pd.Series) -> pd.Series:
    """
    Calendar-month returns that chain inside ``prices``.

    Each month ends on its last close in the series. The first month
    starts from the first close in the series, not from a month-end
    outside it, so the product of (1 + month) is the cumulative return
    of the whole series.
    """
    series = prices.astype(float).dropna()
    if len(series) < MIN_PRICES:
        raise AnalysisError("Monthly returns need two adjusted closes.")
    month_end = series.groupby(series.index.to_period("M")).last()
    base = month_end.shift(1)
    base.iloc[0] = float(series.iloc[0])
    return month_end / base - 1.0


def monthly_table(monthly: pd.Series) -> pd.DataFrame:
    """Years down the side, months 1–12 across. Months outside the window stay NaN."""
    frame = pd.DataFrame(
        {
            "year": monthly.index.year,
            "month": monthly.index.month,
            "value": monthly.to_numpy(),
        }
    )
    table = frame.pivot(index="year", columns="month", values="value")
    return table.reindex(columns=range(1, 13))


def adjusted_ohlc(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Open, high, low and close on the adjusted-close scale, plus volume.

    Each bar is scaled by adjusted close over raw close, the way Yahoo's
    auto-adjust does it. Candles then sit on the same scale as the
    moving averages, which are built from adjusted close. Live Yahoo
    bars are already adjusted, so their factor is one.
    """
    factor = frame["adj_close"].astype(float) / frame["close"].astype(float)
    return pd.DataFrame(
        {
            "open": frame["open"].astype(float) * factor,
            "high": frame["high"].astype(float) * factor,
            "low": frame["low"].astype(float) * factor,
            "close": frame["adj_close"].astype(float),
            "volume": frame["volume"].astype(float),
        },
        index=frame.index,
    )


# ---------------------------------------------------------------------------
# Stock-versus-benchmark calculations
# ---------------------------------------------------------------------------


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


def paired_returns(aligned: pd.DataFrame) -> pd.DataFrame:
    """
    Daily simple returns between consecutive shared sessions. Both
    columns span the same two dates, so each row is a fair pair.
    """
    return (aligned / aligned.shift(1) - 1.0).dropna()


def beta(stock_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Sample covariance with the benchmark over the benchmark's sample variance."""
    pairs = _pairs(stock_returns, benchmark_returns)
    variance = float(pairs["benchmark"].var(ddof=1))
    if variance <= _EPSILON:
        raise AnalysisError("Beta is undefined when the benchmark's returns do not vary.")
    covariance = float(pairs["stock"].cov(pairs["benchmark"]))
    return covariance / variance


def correlation(stock_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Pearson correlation of paired daily returns."""
    pairs = _pairs(stock_returns, benchmark_returns)
    if float(pairs["stock"].std(ddof=1)) <= _EPSILON or float(pairs["benchmark"].std(ddof=1)) <= _EPSILON:
        raise AnalysisError("Correlation is undefined when either series does not vary.")
    return float(pairs["stock"].corr(pairs["benchmark"]))


def capture_ratio(stock_returns: pd.Series, benchmark_returns: pd.Series, *, side: str) -> float:
    """
    The stock's mean daily return over the benchmark's, on the days the
    benchmark rose (``side="up"``) or fell (``side="down"``).
    """
    if side not in ("up", "down"):
        raise AnalysisError("Capture side must be 'up' or 'down'.")
    pairs = _pairs(stock_returns, benchmark_returns, minimum=1)
    days = pairs[pairs["benchmark"] > 0] if side == "up" else pairs[pairs["benchmark"] < 0]
    if days.empty:
        raise AnalysisError(f"No days on which the benchmark moved {side}.")
    return float(days["stock"].mean() / days["benchmark"].mean())


def rolling_beta(
    stock_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = ROLLING_BETA_WINDOW,
) -> pd.Series:
    """Trailing beta over ``window`` paired returns. Warmup rows stay NaN."""
    pairs = _pairs(stock_returns, benchmark_returns, minimum=1)
    covariance = pairs["stock"].rolling(window, min_periods=window).cov(pairs["benchmark"])
    variance = pairs["benchmark"].rolling(window, min_periods=window).var(ddof=1)
    return covariance / variance.where(variance > _EPSILON)


def rolling_correlation(
    stock_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = ROLLING_BETA_WINDOW,
) -> pd.Series:
    """Trailing Pearson correlation over ``window`` paired returns."""
    pairs = _pairs(stock_returns, benchmark_returns, minimum=1)
    return pairs["stock"].rolling(window, min_periods=window).corr(pairs["benchmark"])


def beta_label(value: float) -> str:
    if value < 0:
        return "inverse"
    if value >= BETA_HIGH:
        return "high"
    if value <= BETA_LOW:
        return "low"
    return "market-like"


def correlation_label(value: float) -> str:
    size = abs(value)
    if size >= CORRELATION_STRONG:
        strength = "strong"
    elif size < CORRELATION_WEAK:
        strength = "weak"
    else:
        strength = "moderate"
    return f"{strength} negative" if value < 0 else strength


def rebase(prices: pd.DataFrame, level: float = REBASE_LEVEL) -> pd.DataFrame:
    """Every column divided by its first row, times ``level``."""
    if prices.empty:
        raise AnalysisError("Rebasing needs at least one shared session.")
    return prices / prices.iloc[0] * level


def shared_sessions(series: dict[str, pd.Series]) -> pd.DataFrame:
    """Inner join of several price series. One column per symbol, no fill."""
    frame = pd.concat({name: values.astype(float) for name, values in series.items()}, axis=1, join="inner")
    return frame.dropna().sort_index()


# ---------------------------------------------------------------------------
# Value at Risk and calendar-year returns
# ---------------------------------------------------------------------------


def historical_var(returns: pd.Series, *, confidence: float = VAR_CONFIDENCE) -> float:
    """
    95% historical VaR as a positive loss fraction.

    The left-tail percentile of daily simple returns, negated. A result
    of 0.021 means that on 95% of days the return was better than -2.1%.
    """
    if not 0.5 < confidence < 1.0:
        raise AnalysisError("VaR confidence must sit between 50% and 100%.")
    clean = _finite(returns)
    if len(clean) < MIN_RETURNS_FOR_VAR:
        raise AnalysisError(
            f"Historical VaR needs at least {MIN_RETURNS_FOR_VAR} daily returns."
        )
    tail = float(np.quantile(clean.to_numpy(), 1.0 - confidence))
    # A positive tail (every day was a gain) still reports zero loss floor.
    return float(max(-tail, 0.0))


def parametric_var(returns: pd.Series, *, confidence: float = VAR_CONFIDENCE) -> float:
    """
    95% parametric (Gaussian) VaR as a positive loss fraction.

    Uses the sample mean and sample standard deviation of daily simple
    returns. VaR = −(μ + z_{1−c} σ), floored at zero. z comes from the
    standard normal inverse CDF via ``statistics.NormalDist``.
    """
    from statistics import NormalDist

    if not 0.5 < confidence < 1.0:
        raise AnalysisError("VaR confidence must sit between 50% and 100%.")
    clean = _finite(returns)
    if len(clean) < MIN_RETURNS_FOR_VAR:
        raise AnalysisError(
            f"Parametric VaR needs at least {MIN_RETURNS_FOR_VAR} daily returns."
        )
    mean = float(clean.mean())
    std = float(clean.std(ddof=1))
    if std <= _EPSILON:
        return float(max(-mean, 0.0))
    z = NormalDist().inv_cdf(1.0 - confidence)
    return float(max(-(mean + z * std), 0.0))


def conditional_var(returns: pd.Series, *, confidence: float = VAR_CONFIDENCE) -> float:
    """
    95% historical Conditional VaR (expected shortfall).

    The mean of daily simple returns at or below the left-tail quantile,
    reported as a positive loss. Needs the same minimum return count as VaR.
    """
    if not 0.5 < confidence < 1.0:
        raise AnalysisError("CVaR confidence must sit between 50% and 100%.")
    clean = _finite(returns)
    if len(clean) < MIN_RETURNS_FOR_VAR:
        raise AnalysisError(
            f"Conditional VaR needs at least {MIN_RETURNS_FOR_VAR} daily returns."
        )
    threshold = float(np.quantile(clean.to_numpy(), 1.0 - confidence))
    tail = clean[clean <= threshold + 1e-15]
    if tail.empty:
        raise AnalysisError("Conditional VaR found no returns in the left tail.")
    return float(max(-float(tail.mean()), 0.0))


def jensen_alpha(
    stock_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free: float = DEFAULT_RISK_FREE,
) -> float:
    """
    Annualised Jensen's alpha versus the benchmark.

    Daily alpha is the average stock excess return minus beta times the
    average benchmark excess return, using the daily risk-free rate
    ``(1 + annual)^(1/252) − 1``. Annualised as daily_alpha × 252.
    """
    pairs = _pairs(stock_returns, benchmark_returns)
    daily_rf = daily_risk_free(risk_free)
    stock_excess = pairs["stock"] - daily_rf
    bench_excess = pairs["benchmark"] - daily_rf
    variance = float(pairs["benchmark"].var(ddof=1))
    if variance <= _EPSILON:
        raise AnalysisError("Alpha is undefined when the benchmark's returns do not vary.")
    beta_value = float(pairs["stock"].cov(pairs["benchmark"]) / variance)
    daily_alpha = float(stock_excess.mean() - beta_value * bench_excess.mean())
    return daily_alpha * TRADING_DAYS


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise Pearson correlations of daily simple returns.

    ``returns`` is one column per symbol on shared sessions. Columns that
    do not vary are left as NaN against every peer.
    """
    if returns is None or returns.empty or returns.shape[1] < 1:
        raise AnalysisError("A correlation matrix needs return columns.")
    frame = returns.astype(float).dropna(how="any")
    if len(frame) < MIN_RETURNS_FOR_RATIOS:
        raise AnalysisError(
            f"A correlation matrix needs at least {MIN_RETURNS_FOR_RATIOS} shared daily returns."
        )
    return frame.corr()


def diversification_stats(
    returns: pd.DataFrame,
    weights: dict[str, float],
) -> DiversificationStats:
    """
    Diversification ratio, effective number of holdings, and risk contribution.

    Diversification ratio is the weighted average of standalone volatilities
    divided by the portfolio volatility. Effective N is 1 / Σ w_i².
    Risk contribution for i is w_i × (Σw)_i / σ_p² (share of portfolio variance).
    """
    cleaned, _ = normalize_weights(weights, list(weights))
    cols = [symbol for symbol in cleaned if symbol in returns.columns]
    if len(cols) < 1:
        raise AnalysisError("Diversification analysis needs weights that match return columns.")
    frame = returns.loc[:, cols].astype(float).dropna(how="any")
    if len(frame) < MIN_RETURNS_FOR_RATIOS:
        raise AnalysisError(
            f"Diversification analysis needs at least {MIN_RETURNS_FOR_RATIOS} shared daily returns."
        )
    w = np.array([cleaned[column] for column in cols], dtype=float)
    # Renormalize to the columns present.
    w = w / w.sum()
    weight_map = {column: float(value) for column, value in zip(cols, w, strict=True)}
    cov = frame.cov().to_numpy(dtype=float)
    vols = np.sqrt(np.diag(cov))
    port_var = float(w @ cov @ w)
    if port_var <= _EPSILON:
        raise AnalysisError("Portfolio volatility is zero; diversification ratios are undefined.")
    port_vol = float(np.sqrt(port_var))
    weighted_avg = float(w @ vols)
    ratio = weighted_avg / port_vol if port_vol > _EPSILON else None
    effective_n = float(1.0 / float(np.sum(w * w)))
    marginal = cov @ w
    contrib = w * marginal / port_var
    risk = {column: float(value) for column, value in zip(cols, contrib, strict=True)}
    return DiversificationStats(
        weights=weight_map,
        diversification_ratio=ratio,
        effective_n=effective_n,
        risk_contribution=risk,
        portfolio_volatility=port_vol * np.sqrt(TRADING_DAYS),
        weighted_avg_volatility=weighted_avg * np.sqrt(TRADING_DAYS),
    )


def rolling_sharpe(
    returns: pd.Series,
    risk_free: float = DEFAULT_RISK_FREE,
    window: int = ROLLING_SHARPE_WINDOW,
) -> pd.Series:
    """Trailing annualized Sharpe over ``window`` daily simple returns."""
    clean = returns.astype(float)
    daily_rf = daily_risk_free(risk_free)
    excess = clean - daily_rf
    mean = excess.rolling(window, min_periods=window).mean()
    std = excess.rolling(window, min_periods=window).std(ddof=1)
    return (mean / std.where(std > _EPSILON)) * np.sqrt(TRADING_DAYS)


def rolling_sortino(
    returns: pd.Series,
    risk_free: float = DEFAULT_RISK_FREE,
    window: int = ROLLING_SHARPE_WINDOW,
) -> pd.Series:
    """
    Trailing annualized Sortino over ``window`` daily simple returns.

    Downside deviation uses the same definition as ``downside_deviation``:
    squared shortfalls below the daily risk-free rate, including zeros for
    non-shortfall days, with every day of the window in the divisor.
    """
    clean = returns.astype(float)
    daily_rf = daily_risk_free(risk_free)
    excess = clean - daily_rf
    mean = excess.rolling(window, min_periods=window).mean()

    def _downside(values: np.ndarray) -> float:
        shortfall = np.minimum(values - daily_rf, 0.0)
        return float(np.sqrt(np.mean(shortfall * shortfall)))

    downside = clean.rolling(window, min_periods=window).apply(_downside, raw=True)
    return (mean / downside.where(downside > _EPSILON)) * np.sqrt(TRADING_DAYS)


def validate_cost_bps(cost_bps: float) -> float:
    try:
        value = float(cost_bps)
    except (TypeError, ValueError) as exc:
        raise AnalysisError("Transaction cost must be a number of basis points.") from exc
    if value < COST_BPS_MIN or value > COST_BPS_MAX:
        raise AnalysisError(
            f"Transaction cost must sit between {COST_BPS_MIN:g} and {COST_BPS_MAX:g} bps."
        )
    return value


def calendar_year_returns(prices: pd.Series) -> pd.Series:
    """
    Cumulative return inside each calendar year of ``prices``.

    The first year starts at the first close in the series. The last year
    ends at the last close. Partial years are allowed and labelled by year.
    """
    series = prices.astype(float).dropna()
    if len(series) < MIN_PRICES:
        raise AnalysisError("Calendar-year returns need two adjusted closes.")
    years = series.index.year
    out: dict[int, float] = {}
    for year in sorted(set(years)):
        chunk = series.loc[years == year]
        if len(chunk) < MIN_PRICES:
            # A single print in a year has no return inside that year.
            continue
        out[int(year)] = float(chunk.iloc[-1] / chunk.iloc[0] - 1.0)
    if not out:
        raise AnalysisError("No calendar year inside the window has two closes.")
    return pd.Series(out, name="calendar_return")


# ---------------------------------------------------------------------------
# Portfolio mix of several symbols
# ---------------------------------------------------------------------------


def validate_rebalance(rebalance: str) -> str:
    key = (rebalance or REBALANCE_NONE).strip().lower()
    if key not in REBALANCE_OPTIONS:
        raise AnalysisError(
            f"Unknown rebalance '{rebalance}'. Use none, monthly, or quarterly."
        )
    return key


def normalize_weights(
    weights: dict[str, float] | None,
    symbols: list[str] | tuple[str, ...],
) -> tuple[dict[str, float], tuple[str, ...]]:
    """
    Turn raw weights into a usable mix.

    Missing symbols default to equal weight across the list. Zero-weight
    symbols are dropped. Negative weights are rejected. Weights that do
    not sum to one are renormalized and named in a note.
    """
    notes: list[str] = []
    ordered = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
    if not ordered:
        raise AnalysisError("A portfolio needs at least one symbol.")
    ordered = list(dict.fromkeys(ordered))

    if not weights:
        equal = 1.0 / len(ordered)
        return {symbol: equal for symbol in ordered}, (
            f"Equal weight: each of the {len(ordered)} symbols starts at {equal:.1%}.",
        )

    cleaned: dict[str, float] = {}
    for symbol in ordered:
        raw = weights.get(symbol, weights.get(symbol.lower(), 0.0))
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise AnalysisError(f"Weight for {symbol} is not a number.") from exc
        if value < 0:
            raise AnalysisError(
                f"Weight for {symbol} is {value}. Portfolio weights cannot be negative."
            )
        if value > _EPSILON:
            cleaned[symbol] = value

    skipped = [symbol for symbol in ordered if symbol not in cleaned]
    if skipped:
        notes.append(
            f"{', '.join(skipped)} "
            f"{'has' if len(skipped) == 1 else 'have'} zero weight and "
            f"{'is' if len(skipped) == 1 else 'are'} left out of the portfolio."
        )
    if not cleaned:
        raise AnalysisError(
            "Every symbol has zero weight. Give at least one symbol a positive weight."
        )

    total = float(sum(cleaned.values()))
    if total <= _EPSILON:
        raise AnalysisError("Portfolio weights sum to zero.")
    if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
        notes.append(
            f"Weights summed to {total:.1%}, so each was scaled to total 100%."
        )
    elif abs(total - 1.0) > _EPSILON:
        notes.append(
            f"Weights summed to {total:.1%} and were nudged to total 100%."
        )
    normalized = {symbol: value / total for symbol, value in cleaned.items()}
    return normalized, tuple(notes)


def portfolio_value(
    prices: pd.DataFrame,
    weights: dict[str, float],
    *,
    rebalance: str = REBALANCE_NONE,
    cost_bps: float = DEFAULT_COST_BPS,
) -> pd.Series:
    """Portfolio level starting at 1.0. See ``portfolio_backtest`` for costs."""
    value, _ = portfolio_backtest(prices, weights, rebalance=rebalance, cost_bps=cost_bps)
    return value


def portfolio_backtest(
    prices: pd.DataFrame,
    weights: dict[str, float],
    *,
    rebalance: str = REBALANCE_NONE,
    cost_bps: float = DEFAULT_COST_BPS,
) -> tuple[pd.Series, PortfolioCostReport]:
    """
    Portfolio level starting at 1.0 on the first shared session.

    ``prices`` is a column per symbol of adjusted closes on shared dates.
    With no rebalance the opening mix drifts. Monthly or quarterly
    rebalance resets to ``weights`` on the first session of each period.

    ``cost_bps`` is charged on turnover at each rebalance (and once at
    inception). Turnover for a reset is half the L1 distance between the
    pre-trade weights and the target weights. Cost drag compounds as
    value *= (1 − turnover × bps / 10000).

    Look-ahead rule: holdings for day ``t`` are set at ``t-1`` (or at the
    prior rebalance). The return from ``t-1`` to ``t`` always uses those
    pre-trade weights. A period change on day ``t`` resets weights after
    that mark, so the new mix first earns the move from ``t`` to ``t+1``.
    Shared dates are an inner join with no forward fill.
    """
    mode = validate_rebalance(rebalance)
    bps = validate_cost_bps(cost_bps)
    cost_rate = bps / BPS_PER_UNIT
    if prices is None or prices.empty:
        raise AnalysisError("A portfolio needs shared price history.")
    missing = [symbol for symbol in weights if symbol not in prices.columns]
    if missing:
        raise AnalysisError(f"No prices for portfolio symbol(s): {', '.join(missing)}.")
    frame = prices.loc[:, list(weights)].astype(float).dropna(how="any")
    if len(frame) < MIN_PRICES:
        raise AnalysisError(
            "The portfolio needs at least two shared sessions across its symbols."
        )

    target = np.array([weights[column] for column in frame.columns], dtype=float)
    levels = frame.to_numpy(dtype=float)
    value = np.empty(len(frame), dtype=float)
    # Open at target weights; charge inception turnover of 1.0 (fully invested).
    # Opening weights are known before the first bar (no look-ahead).
    open_turnover = 1.0
    open_cost = open_turnover * cost_rate
    holdings = target * (1.0 - open_cost) / levels[0]
    value[0] = 1.0 - open_cost
    periods = frame.index.to_period("M" if mode == REBALANCE_MONTHLY else "Q")
    turnovers = [open_turnover]
    rebalance_count = 1  # inception counts as the opening trade

    for i in range(1, len(frame)):
        # Mark to market with holdings fixed at t-1. Day-t return uses only
        # pre-rebalance weights, so tomorrow's target never earns today's move.
        pre_value = float(np.dot(holdings, levels[i]))
        value[i] = pre_value
        # Calendar period change is known from the date itself. Execute the
        # reset after today's mark, at today's close — weights for day i+1.
        if mode != REBALANCE_NONE and periods[i] != periods[i - 1]:
            if pre_value > _EPSILON:
                current = holdings * levels[i] / pre_value
            else:
                current = target.copy()
            turnover = 0.5 * float(np.sum(np.abs(target - current)))
            fee = turnover * cost_rate
            post = pre_value * (1.0 - fee)
            holdings = target * post / levels[i]
            value[i] = post
            turnovers.append(turnover)
            rebalance_count += 1
    total_turnover = float(sum(turnovers))
    # Fee drag is the sum of turnover × bps on traded notional (not a path twin).
    fee_sum = float(sum(t * cost_rate for t in turnovers))
    report = PortfolioCostReport(
        cost_bps=bps,
        total_turnover=total_turnover,
        total_cost_drag=fee_sum,
        rebalance_count=int(rebalance_count),
        average_turnover=total_turnover / max(rebalance_count, 1),
    )
    return pd.Series(value, index=frame.index, name=PORTFOLIO_LABEL), report


def portfolio_frame(value: pd.Series) -> pd.DataFrame:
    """Synthetic OHLC so build_period can treat the portfolio like a symbol."""
    level = value.astype(float)
    return pd.DataFrame(
        {
            "open": level,
            "high": level,
            "low": level,
            "close": level,
            "volume": 0.0,
            "adj_close": level,
        },
        index=level.index,
    )


def aligned_export(
    reports: dict[str, PeriodReport],
    *,
    benchmark_report: PeriodReport | None = None,
    portfolio_report: PeriodReport | None = None,
) -> pd.DataFrame:
    """
    Shared-session prices and daily returns for CSV download.

    Columns are ``{symbol}_price`` and ``{symbol}_return``. The portfolio
    and benchmark are included when present. Returns on the first row
    stay blank.
    """
    series: dict[str, pd.Series] = {
        symbol: report.charts.price for symbol, report in reports.items()
    }
    if portfolio_report is not None:
        series[PORTFOLIO_LABEL] = portfolio_report.charts.price
    if benchmark_report is not None:
        series[benchmark_report.stats.ticker] = benchmark_report.charts.price
    shared = shared_sessions(series)
    if shared.empty:
        raise AnalysisError("No shared sessions to export.")
    columns: dict[str, pd.Series] = {}
    for name in shared.columns:
        columns[f"{name}_price"] = shared[name]
        columns[f"{name}_return"] = shared[name].pct_change()
    return pd.DataFrame(columns, index=shared.index)


# ---------------------------------------------------------------------------
# One symbol over one window
# ---------------------------------------------------------------------------


def build_period(
    history: pd.DataFrame,
    *,
    ticker: str,
    start,
    end,
    benchmark: pd.DataFrame | None = None,
    benchmark_ticker: str | None = None,
    ma_short: int = MA_SHORT,
    ma_long: int = MA_LONG,
    risk_free: float = DEFAULT_RISK_FREE,
) -> PeriodReport:
    """Stats, chart series, and sentences for one window of ``history``."""
    short_window, long_window = validate_average_windows(ma_short, ma_long)
    rate = validate_risk_free(risk_free)
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
    ma_short_full = moving_average(prices, short_window)
    ma_long_full = moving_average(prices, long_window)
    returns = daily_simple_returns(window).dropna()
    drawdown = max_drawdown(window)
    last_day = window.index[-1]

    risk = _risk_fields(window, full_returns)
    reward = _reward_fields(window, returns, rate)
    relative, relative_charts = _relative_fields(prices, window, benchmark, benchmark_ticker, rate)

    years = _optional(calendar_year_returns, window)
    if years is None:
        years = pd.Series(dtype=float, name="calendar_return")
    stats = PeriodStats(
        ticker=ticker,
        benchmark=benchmark_ticker if benchmark is not None else None,
        window_start=pd.Timestamp(window.index[0]),
        window_end=pd.Timestamp(last_day),
        n_days=int(len(window)),
        calendar_days=_calendar_days(window.index),
        first_price=float(window.iloc[0]),
        last_price=float(window.iloc[-1]),
        cumulative_return=cumulative_return(window),
        risk_free_rate=rate,
        max_drawdown=float(drawdown["max_drawdown"]),
        drawdown_peak=drawdown["peak"],
        drawdown_trough=drawdown["trough"],
        drawdown_peak_price=float(drawdown["peak_price"]),
        drawdown_trough_price=float(drawdown["trough_price"]),
        drawdown_days_to_trough=drawdown["days_to_trough"],
        drawdown_days_to_recovery=drawdown["days_to_recovery"],
        drawdown_recovered=drawdown["recovered"],
        longest_underwater_days=drawdown["longest_underwater_days"],
        ma_short_window=short_window,
        ma_long_window=long_window,
        # The printed average is the value on the last session of the window.
        # It stays blank while that session is still inside the warmup.
        ma_short=_value_on(ma_short_full, last_day),
        ma_long=_value_on(ma_long_full, last_day),
        value_at_risk_95=_optional(historical_var, returns),
        parametric_var_95=_optional(parametric_var, returns),
        cvar_95=_optional(conditional_var, returns),
        **risk,
        **reward,
        **relative,
    )
    simple_full = daily_simple_returns(prices)
    charts = ChartData(
        price=window,
        ohlc=adjusted_ohlc(history).reindex(window.index),
        ma_short=ma_short_full.reindex(window.index),
        ma_long=ma_long_full.reindex(window.index),
        drawdown=drawdown["series"],
        rolling_vol=rolling_annualized_volatility(full_returns).reindex(window.index),
        returns=returns,
        monthly=monthly_returns(window),
        calendar_years=years,
        rolling_sharpe=rolling_sharpe(simple_full, rate).reindex(window.index),
        rolling_sortino=rolling_sortino(simple_full, rate).reindex(window.index),
        **relative_charts,
    )
    sentences = clamp_notes + describe_period(stats)
    return PeriodReport(stats=stats, charts=charts, sentences=sentences, notes=clamp_notes)


def _risk_fields(window: pd.Series, full_returns: pd.Series) -> dict:
    """Period volatility and where it ranks in the symbol's own history."""
    baseline = rolling_annualized_volatility(full_returns).dropna()
    period_returns = daily_log_returns(window).dropna()
    period_vol = _optional(annualized_volatility, period_returns)
    percentile = None
    label = None
    if period_vol is not None and len(baseline) >= MIN_BASELINE_WINDOWS:
        percentile = percentile_rank(period_vol, baseline)
        label = volatility_label(percentile)
    return {
        "annualized_volatility": period_vol,
        "n_returns": int(len(period_returns)),
        "vol_percentile": percentile,
        "vol_label": label,
        "vol_baseline_count": int(len(baseline)),
    }


def _reward_fields(window: pd.Series, returns: pd.Series, risk_free: float) -> dict:
    """Annualized return, the two risk-adjusted ratios, and the extreme days."""
    enough = len(returns) >= MIN_RETURNS_FOR_RATIOS
    extremes = _optional(extreme_days, returns) or {
        "best_day": None,
        "best_day_return": None,
        "worst_day": None,
        "worst_day_return": None,
    }
    return {
        "annualized_return": _optional(annualized_return, window),
        "sharpe": _optional(sharpe_ratio, returns, risk_free) if enough else None,
        "sortino": _optional(sortino_ratio, returns, risk_free) if enough else None,
        "positive_day_share": _optional(positive_share, returns),
        **extremes,
    }


def _relative_fields(
    prices: pd.Series,
    window: pd.Series,
    benchmark: pd.DataFrame | None,
    benchmark_ticker: str | None,
    risk_free: float = DEFAULT_RISK_FREE,
) -> tuple[dict, dict]:
    """Everything that needs the benchmark: the gap, beta, alpha, correlation, capture."""
    stats = {
        "benchmark_cumulative_return": None,
        "aligned_stock_return": None,
        "excess_return": None,
        "aligned_days": 0,
        "dropped_stock_sessions": 0,
        "return_pairs": 0,
        "beta": None,
        "beta_label": None,
        "correlation": None,
        "correlation_label": None,
        "up_capture": None,
        "down_capture": None,
        "up_days": 0,
        "down_days": 0,
        "alpha": None,
    }
    charts = {
        "aligned": None,
        "benchmark_returns": None,
        "rolling_beta": None,
        "rolling_correlation": None,
    }
    if benchmark is None or not benchmark_ticker:
        return stats, charts

    bench_px = _adjusted(benchmark)
    aligned = align_adjusted(window, bench_px)
    charts["aligned"] = aligned
    stats["aligned_days"] = int(len(aligned))
    stats["dropped_stock_sessions"] = int(len(window) - len(aligned))
    if len(aligned) >= MIN_PRICES:
        stats["aligned_stock_return"] = cumulative_return(aligned["stock"])
        stats["benchmark_cumulative_return"] = cumulative_return(aligned["benchmark"])
        stats["excess_return"] = stats["aligned_stock_return"] - stats["benchmark_cumulative_return"]

    pairs = paired_returns(aligned)
    charts["benchmark_returns"] = pairs["benchmark"]
    stats["return_pairs"] = int(len(pairs))
    stats["up_days"] = int((pairs["benchmark"] > 0).sum())
    stats["down_days"] = int((pairs["benchmark"] < 0).sum())
    if len(pairs) >= MIN_RETURNS_FOR_RATIOS:
        stats["beta"] = _optional(beta, pairs["stock"], pairs["benchmark"])
        stats["correlation"] = _optional(correlation, pairs["stock"], pairs["benchmark"])
        stats["alpha"] = _optional(jensen_alpha, pairs["stock"], pairs["benchmark"], risk_free)
    if stats["beta"] is not None:
        stats["beta_label"] = beta_label(stats["beta"])
    if stats["correlation"] is not None:
        stats["correlation_label"] = correlation_label(stats["correlation"])
    if stats["up_days"] >= MIN_CAPTURE_DAYS:
        stats["up_capture"] = _optional(capture_ratio, pairs["stock"], pairs["benchmark"], side="up")
    if stats["down_days"] >= MIN_CAPTURE_DAYS:
        stats["down_capture"] = _optional(capture_ratio, pairs["stock"], pairs["benchmark"], side="down")

    # Like the moving averages, the rolling lines are built on every
    # shared session in the loaded files so a short window still has one.
    full_pairs = paired_returns(align_adjusted(prices, bench_px))
    if len(full_pairs) >= ROLLING_BETA_WINDOW:
        charts["rolling_beta"] = rolling_beta(full_pairs["stock"], full_pairs["benchmark"]).reindex(window.index)
        charts["rolling_correlation"] = rolling_correlation(
            full_pairs["stock"], full_pairs["benchmark"]
        ).reindex(window.index)
    return stats, charts


# ---------------------------------------------------------------------------
# Several symbols side by side
# ---------------------------------------------------------------------------


def build_comparison(
    reports: dict[str, PeriodReport],
    benchmark_report: PeriodReport | None = None,
    portfolio_report: PeriodReport | None = None,
    weights: dict[str, float] | None = None,
) -> Comparison:
    """
    Line up the symbols' reports. Each row keeps that symbol's own
    sessions. The rebased frame keeps only sessions every line shares.
    An optional portfolio row sits above the benchmark.
    """
    if not reports:
        raise AnalysisError("A comparison needs at least one symbol.")
    rows = [report.stats for report in reports.values()]
    series = {symbol: report.charts.price for symbol, report in reports.items()}
    portfolio_name = None
    if portfolio_report is not None:
        portfolio_name = portfolio_report.stats.ticker
        rows.append(portfolio_report.stats)
        series[portfolio_name] = portfolio_report.charts.price
    benchmark = None
    if benchmark_report is not None:
        benchmark = benchmark_report.stats.ticker
        rows.append(benchmark_report.stats)
        series[benchmark] = benchmark_report.charts.price
    shared = shared_sessions(series)
    rebased = rebase(shared) if len(shared) >= MIN_PRICES else None
    symbol_rows = tuple(report.stats for report in reports.values())
    if portfolio_report is not None:
        symbol_rows = symbol_rows + (portfolio_report.stats,)
    benchmark_stats = None if benchmark_report is None else benchmark_report.stats
    sentences = describe_comparison(symbol_rows, benchmark_stats, rebased)

    # Correlation and diversification use only the selected symbols (not the
    # synthetic portfolio row or the benchmark), on shared simple returns.
    symbol_returns = {
        symbol: report.charts.returns for symbol, report in reports.items()
    }
    corr = None
    divers = None
    if len(symbol_returns) >= 2:
        shared_returns = pd.concat(symbol_returns, axis=1, join="inner").dropna(how="any")
        if len(shared_returns) >= MIN_RETURNS_FOR_RATIOS:
            corr = _optional(correlation_matrix, shared_returns)
            if weights:
                divers = _optional(diversification_stats, shared_returns, weights)
            else:
                equal = {symbol: 1.0 for symbol in shared_returns.columns}
                divers = _optional(diversification_stats, shared_returns, equal)
            if divers is not None:
                sentences = sentences + _diversification_sentences(divers)

    return Comparison(
        rows=tuple(rows),
        benchmark=benchmark,
        rebased=rebased,
        sentences=sentences,
        portfolio=portfolio_name,
        correlation_matrix=corr,
        diversification=divers,
    )


def comparison_table(comparison: Comparison) -> pd.DataFrame:
    """One row per symbol, numbers as fractions. Missing figures stay NaN."""
    records = []
    for stats in comparison.rows:
        is_benchmark = stats.ticker == comparison.benchmark
        is_portfolio = comparison.portfolio is not None and stats.ticker == comparison.portfolio
        gap = None if stats.excess_return is None else stats.excess_return * 100.0
        if is_benchmark:
            label = f"{stats.ticker} (benchmark)"
        elif is_portfolio:
            label = f"{stats.ticker}"
        else:
            label = stats.ticker
        records.append(
            {
                "Symbol": label,
                "Return": stats.cumulative_return,
                "Annualized return": stats.annualized_return,
                "Volatility": stats.annualized_volatility,
                "Max drawdown": stats.max_drawdown,
                "Sharpe": stats.sharpe,
                "Sortino": stats.sortino,
                "Beta": stats.beta,
                "Correlation": stats.correlation,
                "Up capture": stats.up_capture,
                "Down capture": stats.down_capture,
                "Vs benchmark (pp)": gap,
                "Sessions": stats.n_days,
            }
        )
    table = pd.DataFrame.from_records(records).set_index("Symbol")
    return table.astype(float)


def format_cell(column: str, value) -> str:
    """One table figure as text. A figure the window could not support prints as a dash."""
    if value is None or pd.isna(value):
        return MISSING
    return TABLE_FORMATS[column].format(value)


def describe_comparison(
    rows: tuple[PeriodStats, ...],
    benchmark: PeriodStats | None,
    rebased: pd.DataFrame | None,
) -> tuple[str, ...]:
    """Ranking sentences. Empty for a single symbol, which has nothing to rank."""
    if len(rows) < 2:
        return ()
    lines = [_return_ranking(rows, benchmark)]
    lines.append(_volatility_range(rows))
    lines.append(_drawdown_range(rows))
    lines.append(_sharpe_ranking(rows))
    lines.append(_window_mismatch(rows, rebased))
    return tuple(line for line in lines if line)


def _return_ranking(rows: tuple[PeriodStats, ...], benchmark: PeriodStats | None) -> str:
    ordered = sorted(rows, key=lambda stats: stats.cumulative_return, reverse=True)
    listing = ", ".join(f"{stats.ticker} {stats.cumulative_return:+.1%}" for stats in ordered)
    text = f"Ranked by cumulative return: {listing}."
    if benchmark is not None:
        ahead = [stats.ticker for stats in ordered if stats.cumulative_return > benchmark.cumulative_return]
        text += (
            f" {benchmark.ticker}, the benchmark, returned {benchmark.cumulative_return:+.1%} "
            f"over its own sessions in the range, and {len(ahead)} of the {len(rows)} symbols finished ahead of it."
        )
    return text


def _volatility_range(rows: tuple[PeriodStats, ...]) -> str:
    measured = [stats for stats in rows if stats.annualized_volatility is not None]
    if len(measured) < 2:
        return ""
    calm = min(measured, key=lambda stats: stats.annualized_volatility)
    wild = max(measured, key=lambda stats: stats.annualized_volatility)
    return (
        f"Annualized volatility runs from {calm.annualized_volatility:.1%} ({calm.ticker}) "
        f"to {wild.annualized_volatility:.1%} ({wild.ticker})."
    )


def _drawdown_range(rows: tuple[PeriodStats, ...]) -> str:
    measured = [stats for stats in rows if stats.max_drawdown is not None]
    if len(measured) < 2:
        return ""
    deepest = min(measured, key=lambda stats: stats.max_drawdown)
    shallowest = max(measured, key=lambda stats: stats.max_drawdown)
    return (
        f"The deepest drawdown is {deepest.ticker}'s {deepest.max_drawdown:.1%}; "
        f"the shallowest is {shallowest.ticker}'s {shallowest.max_drawdown:.1%}."
    )


def _sharpe_ranking(rows: tuple[PeriodStats, ...]) -> str:
    measured = [stats for stats in rows if stats.sharpe is not None]
    if len(measured) < 2:
        return ""
    best = max(measured, key=lambda stats: stats.sharpe)
    worst = min(measured, key=lambda stats: stats.sharpe)
    return (
        f"Per unit of volatility, {best.ticker} ranks first with a Sharpe ratio of "
        f"{best.sharpe:.2f} and {worst.ticker} last at {worst.sharpe:.2f}, "
        f"at a risk-free rate of {best.risk_free_rate:.1%} a year."
    )


def _window_mismatch(rows: tuple[PeriodStats, ...], rebased: pd.DataFrame | None) -> str:
    """Say so when one symbol's sessions start later than another's."""
    earliest = min(stats.window_start for stats in rows)
    late = [stats for stats in rows if stats.window_start > earliest]
    if not late:
        return ""
    listing = ", ".join(f"{stats.ticker} on {stats.window_start.date().isoformat()}" for stats in late)
    rows_word = "that row covers" if len(late) == 1 else "those rows cover"
    text = (
        f"Not every symbol covers the same dates. The earliest starts on "
        f"{earliest.date().isoformat()}; starting later: {listing}, so {rows_word} a shorter window."
    )
    if rebased is not None:
        text += (
            f" The rebased chart starts on {rebased.index[0].date().isoformat()}, "
            "the first session all of them share."
        )
    return text


# ---------------------------------------------------------------------------
# Sentences for one symbol
# ---------------------------------------------------------------------------


def describe_period(stats: PeriodStats) -> tuple[str, ...]:
    """Plain-English sentences. Each number is a field on ``stats``."""
    lines = [
        _return_sentence(stats),
        _volatility_sentence(stats),
        _ratio_sentence(stats),
        _var_sentence(stats),
        _drawdown_sentence(stats),
        _extreme_sentence(stats),
        _average_sentence(stats),
        _benchmark_sentence(stats),
        _beta_sentence(stats),
        _alpha_sentence(stats),
        _capture_sentence(stats),
        _var_extras_sentence(stats),
    ]
    return tuple(line for line in lines if line)


def _return_sentence(stats: PeriodStats) -> str:
    start = stats.window_start.date().isoformat()
    end = stats.window_end.date().isoformat()
    text = (
        f"{stats.ticker} adjusted close moved from {stats.first_price:.2f} on {start} "
        f"to {stats.last_price:.2f} on {end}: a cumulative return of "
        f"{stats.cumulative_return:+.1%} across {stats.n_days} trading days."
    )
    if stats.annualized_return is None:
        return text + (
            f" No annualized return is shown: the window spans {stats.calendar_days} "
            f"calendar days and annualizing needs at least {MIN_ANNUALIZE_CALENDAR_DAYS}."
        )
    return text + (
        f" Over {stats.calendar_days} calendar days that compounds to "
        f"{stats.annualized_return:+.1%} a year."
    )


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


def _ratio_sentence(stats: PeriodStats) -> str:
    rate = f"{stats.risk_free_rate:.1%}"
    if stats.sharpe is None:
        return (
            f"Sharpe and Sortino ratios are omitted: they need at least "
            f"{MIN_RETURNS_FOR_RATIOS} daily returns that vary, and this window has {stats.n_returns}."
        )
    text = f"The Sharpe ratio is {stats.sharpe:.2f}"
    if stats.sortino is None:
        text += f" at a risk-free rate of {rate} a year. The Sortino ratio is omitted: no day fell below that rate."
    else:
        text += f" and the Sortino ratio is {stats.sortino:.2f}, at a risk-free rate of {rate} a year."
    if stats.sharpe > 0:
        return text + " Both are positive because the average daily return was above that rate."
    if stats.sharpe < 0:
        return text + " Both are negative because the average daily return was below that rate."
    return text


def _var_sentence(stats: PeriodStats) -> str:
    if stats.value_at_risk_95 is None:
        return (
            f"95% historical Value at Risk is omitted: it needs at least "
            f"{MIN_RETURNS_FOR_VAR} daily returns, and this window has {stats.n_returns}."
        )
    return (
        f"95% historical Value at Risk is {stats.value_at_risk_95:.1%}: on "
        f"{VAR_CONFIDENCE:.0%} of the {stats.n_returns} days the return was better than "
        f"−{stats.value_at_risk_95:.1%}."
    )


def describe_calendar_years(years: pd.Series, *, ticker: str) -> str:
    """One sentence listing each calendar year's return inside the window."""
    if years is None or len(years) == 0:
        return ""
    listing = ", ".join(f"{int(year)} {value:+.1%}" for year, value in years.items())
    return f"{ticker} calendar-year returns inside the window: {listing}."


def _drawdown_sentence(stats: PeriodStats) -> str:
    if stats.max_drawdown is None or stats.drawdown_peak is None or stats.drawdown_trough is None:
        return ""
    if abs(stats.max_drawdown) < _EPSILON:
        return (
            "Maximum drawdown is 0.0%: the adjusted close never finished below "
            "a prior peak inside this window."
        )
    text = (
        f"Maximum drawdown is {stats.max_drawdown:.1%}, from a peak of "
        f"{stats.drawdown_peak_price:.2f} on {stats.drawdown_peak.date().isoformat()} "
        f"to a trough of {stats.drawdown_trough_price:.2f} on "
        f"{stats.drawdown_trough.date().isoformat()}."
    )
    if stats.drawdown_days_to_trough is not None:
        text += f" That decline spanned {stats.drawdown_days_to_trough} trading sessions."
    if stats.drawdown_recovered:
        text += (
            f" Price recovered to the prior peak in "
            f"{stats.drawdown_days_to_recovery} trading sessions after the trough."
        )
    elif stats.drawdown_recovered is False:
        text += " The window ends before price recovered to that prior peak."
    if stats.longest_underwater_days is not None:
        text += (
            f" The longest stretch below a running peak lasted "
            f"{stats.longest_underwater_days} trading sessions."
        )
    return text


def _alpha_sentence(stats: PeriodStats) -> str:
    if stats.benchmark is None:
        return ""
    if stats.alpha is None:
        return (
            f"Jensen's alpha versus {stats.benchmark} is omitted: it needs at least "
            f"{MIN_RETURNS_FOR_RATIOS} shared daily returns that vary, and this window has "
            f"{stats.return_pairs}."
        )
    sign = "above" if stats.alpha > 0 else "below" if stats.alpha < 0 else "in line with"
    return (
        f"Annualised Jensen's alpha versus {stats.benchmark} is {stats.alpha:+.1%}: "
        f"after adjusting for beta, average daily excess return was {sign} the "
        f"benchmark's, scaled by {TRADING_DAYS} trading days."
    )


def _var_extras_sentence(stats: PeriodStats) -> str:
    parts: list[str] = []
    if stats.parametric_var_95 is not None:
        parts.append(
            f"95% parametric (Gaussian) VaR is {stats.parametric_var_95:.1%}"
        )
    if stats.cvar_95 is not None:
        parts.append(
            f"95% Conditional VaR (expected shortfall) is {stats.cvar_95:.1%}, "
            f"the average loss on days at or below the historical VaR threshold"
        )
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0] + "."
    return parts[0] + "; " + parts[1] + "."


def _diversification_sentences(stats: DiversificationStats) -> tuple[str, ...]:
    lines: list[str] = []
    if stats.diversification_ratio is not None and stats.portfolio_volatility is not None:
        lines.append(
            f"The diversification ratio is {stats.diversification_ratio:.2f}: "
            f"weighted standalone volatility is {stats.weighted_avg_volatility:.1%} a year "
            f"versus portfolio volatility of {stats.portfolio_volatility:.1%}."
        )
    if stats.effective_n is not None:
        lines.append(
            f"The effective number of holdings is {stats.effective_n:.2f} "
            f"(1 / Σ w² over the {len(stats.weights)} weights)."
        )
    if stats.risk_contribution:
        ranked = sorted(stats.risk_contribution.items(), key=lambda item: -item[1])
        listing = ", ".join(f"{name} {share:.0%}" for name, share in ranked)
        lines.append(f"Risk contribution by weight: {listing}.")
    return tuple(lines)


def _extreme_sentence(stats: PeriodStats) -> str:
    if stats.best_day is None or stats.worst_day is None or stats.positive_day_share is None:
        return ""
    return (
        f"The best day was {stats.best_day_return:+.1%} on {stats.best_day.date().isoformat()} "
        f"and the worst was {stats.worst_day_return:+.1%} on {stats.worst_day.date().isoformat()}. "
        f"{stats.positive_day_share:.0%} of the {stats.n_returns} daily moves were gains."
    )


def _average_sentence(stats: PeriodStats) -> str:
    end = stats.window_end.date().isoformat()
    parts: list[str] = []
    if stats.ma_short is None:
        parts.append(
            f"The {stats.ma_short_window}-day average is not available on {end}: "
            f"that average needs {stats.ma_short_window} earlier adjusted closes."
        )
    else:
        relation = _relation(stats.last_price, stats.ma_short)
        parts.append(
            f"On {end} the close is {relation} the "
            f"{stats.ma_short_window}-day average of {stats.ma_short:.2f}."
        )
    if stats.ma_long is None:
        parts.append(
            f"The {stats.ma_long_window}-day average is not available on {end}: "
            f"that average needs {stats.ma_long_window} earlier adjusted closes."
        )
    else:
        relation = _relation(stats.last_price, stats.ma_long)
        parts.append(
            f"The close is {relation} the {stats.ma_long_window}-day average of {stats.ma_long:.2f}."
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


def _beta_sentence(stats: PeriodStats) -> str:
    if not stats.benchmark:
        return ""
    if stats.beta is None or stats.correlation is None:
        return (
            f"Beta and correlation against {stats.benchmark} are omitted: they need at least "
            f"{MIN_RETURNS_FOR_RATIOS} shared daily returns that vary, and this window has "
            f"{stats.return_pairs}."
        )
    return (
        f"Against {stats.benchmark}, beta is {stats.beta:.2f} "
        f"({stats.beta_label}: {_beta_rule(stats.beta_label)}) and the correlation of daily "
        f"returns is {stats.correlation:.2f} ({stats.correlation_label}: "
        f"{_correlation_rule(stats.correlation)}), from {stats.return_pairs} shared daily returns. "
        f"A beta of {stats.beta:.2f} means {stats.ticker} moved about {abs(stats.beta):.2f}% "
        f"on average, in the {'opposite' if stats.beta < 0 else 'same'} direction, "
        f"for each 1% move in {stats.benchmark}."
    )


def _capture_sentence(stats: PeriodStats) -> str:
    if not stats.benchmark:
        return ""
    if stats.up_capture is None or stats.down_capture is None:
        if stats.return_pairs == 0:
            return ""
        return (
            f"Up and down capture are omitted: {stats.benchmark} rose on {stats.up_days} "
            f"day(s) and fell on {stats.down_days}, and each side needs at least {MIN_CAPTURE_DAYS}."
        )
    return (
        f"On the {stats.up_days} days {stats.benchmark} rose, {stats.ticker}'s average daily "
        f"return was {stats.up_capture:.0%} of {stats.benchmark}'s (up capture). On the "
        f"{stats.down_days} days {stats.benchmark} fell, it was {stats.down_capture:.0%} "
        "(down capture)."
    )


def _beta_rule(label: str) -> str:
    rules = {
        "high": f"{BETA_HIGH:.2f} or above",
        "low": f"zero to {BETA_LOW:.2f}",
        "inverse": "below zero",
        "market-like": f"between {BETA_LOW:.2f} and {BETA_HIGH:.2f}",
    }
    return rules[label]


def _correlation_rule(value: float) -> str:
    size = abs(value)
    if size >= CORRELATION_STRONG:
        return f"size {CORRELATION_STRONG:.2f} or above"
    if size < CORRELATION_WEAK:
        return f"size below {CORRELATION_WEAK:.2f}"
    return f"size between {CORRELATION_WEAK:.2f} and {CORRELATION_STRONG:.2f}"


def _relation(price: float, average: float) -> str:
    if price > average:
        return "above"
    if price < average:
        return "below"
    return "equal to"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _adjusted(frame: pd.DataFrame) -> pd.Series:
    if "adj_close" not in frame.columns:
        raise AnalysisError("The price frame has no adjusted close.")
    series = frame["adj_close"].astype(float).sort_index()
    series = series[~series.index.duplicated(keep="last")].dropna()
    if (series <= 0).any():
        raise AnalysisError("Adjusted closes must be positive.")
    return series


def _finite(values: pd.Series) -> pd.Series:
    return pd.Series(values).astype(float).replace([np.inf, -np.inf], np.nan).dropna()


def _excess_returns(returns: pd.Series, risk_free: float) -> pd.Series:
    clean = _finite(returns)
    if len(clean) < MIN_RETURNS_FOR_VOL:
        raise AnalysisError(
            f"A risk-adjusted ratio needs at least {MIN_RETURNS_FOR_VOL} daily returns."
        )
    return clean - daily_risk_free(risk_free)


def _pairs(stock_returns: pd.Series, benchmark_returns: pd.Series, *, minimum: int = MIN_RETURNS_FOR_VOL) -> pd.DataFrame:
    """Returns present on both sides, on the dates they share."""
    frame = pd.concat(
        {"stock": _finite(stock_returns), "benchmark": _finite(benchmark_returns)},
        axis=1,
        join="inner",
    ).dropna()
    if len(frame) < minimum:
        raise AnalysisError(f"This calculation needs at least {minimum} paired daily return(s).")
    return frame


def _calendar_days(index: pd.Index) -> int:
    if not isinstance(index, pd.DatetimeIndex):
        raise AnalysisError("Calendar spans need dated prices.")
    return int((index[-1] - index[0]).days)


def _optional(calculation, *args, **kwargs):
    """A figure this window cannot support is reported as missing, not as an error."""
    try:
        return calculation(*args, **kwargs)
    except AnalysisError:
        return None


def _value_on(series: pd.Series, when: pd.Timestamp) -> float | None:
    if when not in series.index or pd.isna(series.loc[when]):
        return None
    return float(series.loc[when])


if __name__ == "__main__":
    from data_loader import load_offline_history

    bundled = {symbol: load_offline_history(symbol) for symbol in ("AAPL", "MSFT", "SPY")}
    start, end = window_for_preset(bundled["AAPL"].frame.index, "Max")
    reports = {
        symbol: build_period(
            bundled[symbol].frame,
            ticker=symbol,
            start=start,
            end=end,
            benchmark=bundled["SPY"].frame,
            benchmark_ticker="SPY",
        )
        for symbol in ("AAPL", "MSFT")
    }
    spy_report = build_period(bundled["SPY"].frame, ticker="SPY", start=start, end=end)
    prices = shared_sessions({symbol: reports[symbol].charts.price for symbol in reports})
    weights, weight_notes = normalize_weights(None, list(reports))
    value = portfolio_value(prices, weights, rebalance=REBALANCE_MONTHLY)
    portfolio = build_period(
        portfolio_frame(value),
        ticker=PORTFOLIO_LABEL,
        start=start,
        end=end,
        benchmark=bundled["SPY"].frame,
        benchmark_ticker="SPY",
        ma_short=20,
        ma_long=50,
    )
    comparison = build_comparison(reports, spy_report, portfolio)
    for sentence in reports["AAPL"].sentences:
        print(sentence)
    print("---")
    print(describe_calendar_years(reports["AAPL"].charts.calendar_years, ticker="AAPL"))
    print("---")
    for note in weight_notes:
        print(note)
    for sentence in comparison.sentences:
        print(sentence)
    print("---")
    print(comparison_table(comparison).round(3).to_string())
