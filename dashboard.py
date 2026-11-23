"""
dashboard.py — THE ORCHESTRATOR
================================
The only place a briefing is assembled:

    prices  →  features  →  sentences  →  charts  →  markdown

One briefing covers up to five symbols and one benchmark over one
requested range. One of the symbols is in focus: the detailed charts
and the sentences describe that one, and the side-by-side table and the
rebased chart cover all of them.

``python dashboard.py`` reads live Yahoo Finance data.
``python dashboard.py --offline`` uses the bundled files and makes no
network call.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import reduce
from pathlib import Path

import pandas as pd

from data_loader import (
    CompanySnapshot,
    MarketDataError,
    PriceHistory,
    fetch_company_snapshot,
    fetch_history,
    load_offline_history,
    normalize_symbol,
)
from features import (
    DEFAULT_RISK_FREE,
    MA_LONG,
    MA_SHORT,
    PORTFOLIO_LABEL,
    PRESETS,
    REBALANCE_NONE,
    REBALANCE_OPTIONS,
    ROLLING_BETA_WINDOW,
    ROLLING_VOL_WINDOW,
    TABLE_FORMATS,
    AnalysisError,
    Comparison,
    PeriodReport,
    aligned_export,
    build_comparison,
    build_period,
    comparison_table,
    describe_calendar_years,
    format_cell,
    methodology_notes,
    monthly_table,
    normalize_weights,
    portfolio_frame,
    portfolio_value,
    shared_sessions,
    validate_average_windows,
    validate_rebalance,
    validate_risk_free,
    volatility_rule_text,
    window_for_preset,
)
from plots import (
    candlestick_chart,
    comparison_chart,
    drawdown_chart,
    monthly_heatmap,
    price_chart,
    return_histogram,
    rolling_relationship_chart,
    volatility_chart,
)


DEFAULT_TICKER = "AAPL"
DEFAULT_BENCHMARK = "SPY"
DEFAULT_PRESET = "1Y"
OUTPUT_MD = "analysis.md"
# Five lines plus a benchmark is as many as one chart and one table
# can carry before the colours and the columns stop being readable.
MAX_SYMBOLS = 5
PERCENT = 100.0
BENCHMARK_CHOICES = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq-100",
    "DIA": "Dow Jones Industrial Average",
    "IWM": "Russell 2000",
}


@dataclass(frozen=True)
class Briefing:
    histories: dict[str, PriceHistory]
    failed: dict[str, str]
    benchmark: PriceHistory | None
    benchmark_error: str | None
    reports: dict[str, PeriodReport]
    benchmark_report: PeriodReport | None
    comparison: Comparison
    focus: str
    figures: dict
    markdown: str
    notes: tuple[str, ...]
    portfolio_report: PeriodReport | None = None
    portfolio_weights: dict[str, float] | None = None
    rebalance: str = REBALANCE_NONE
    snapshot: CompanySnapshot | None = None
    export_csv: str | None = None

    @property
    def stock(self) -> PriceHistory:
        """The symbol in focus."""
        return self.histories[self.focus]

    @property
    def report(self) -> PeriodReport:
        """The report for the symbol in focus."""
        return self.reports[self.focus]

    @property
    def stale(self) -> tuple[str, ...]:
        """Symbols served from an expired download because Yahoo did not answer."""
        loaded = list(self.histories.values())
        if self.benchmark is not None:
            loaded.append(self.benchmark)
        return tuple(history.ticker for history in loaded if history.stale)


def run_briefing(
    *,
    ticker: str | None = None,
    tickers: list[str] | tuple[str, ...] | None = None,
    benchmark: str = DEFAULT_BENCHMARK,
    focus: str | None = None,
    start=None,
    end=None,
    preset: str | None = DEFAULT_PRESET,
    offline: bool = False,
    refresh: bool = False,
    cache_dir: Path | None = None,
    ma_short: int = MA_SHORT,
    ma_long: int = MA_LONG,
    risk_free: float = DEFAULT_RISK_FREE,
    weights: dict[str, float] | None = None,
    rebalance: str = REBALANCE_NONE,
    portfolio: bool = False,
) -> Briefing:
    """
    One briefing. Pass ``tickers`` for several symbols, or ``ticker``
    for one. ``preset`` is used when start or end is omitted. Pass both
    dates to ignore the preset.

    A symbol that cannot be loaded or has no sessions in the range is
    reported in ``failed`` and the rest of the briefing still runs. The
    call raises only when no symbol at all can be briefed.
    """
    # Settings are checked before any download, so a bad window fails fast.
    validate_average_windows(ma_short, ma_long)
    validate_risk_free(risk_free)
    symbols, bench_symbol, rejected, notes = resolve_symbols(ticker, tickers, benchmark)

    histories, failed = _load_symbols(symbols, offline=offline, refresh=refresh, cache_dir=cache_dir)
    failed = {**rejected, **failed}
    if not histories:
        raise MarketDataError(_join_failures(failed))
    bench_history, bench_error = _load_benchmark(bench_symbol, offline=offline, refresh=refresh, cache_dir=cache_dir)

    start, end = _requested_window(histories, start, end, preset)
    reports, unbriefed = _build_reports(
        histories,
        bench_history,
        start=start,
        end=end,
        ma_short=ma_short,
        ma_long=ma_long,
        risk_free=risk_free,
    )
    failed.update(unbriefed)
    if not reports:
        raise AnalysisError(_join_failures(unbriefed))
    histories = {symbol: histories[symbol] for symbol in reports}

    benchmark_report, window_error = _benchmark_report(
        bench_history, start=start, end=end, ma_short=ma_short, ma_long=ma_long, risk_free=risk_free
    )
    bench_error = bench_error or window_error

    chosen = _choose_focus(focus, reports)
    rebalance_mode = validate_rebalance(rebalance)
    portfolio_report = None
    weight_map = None
    portfolio_notes: tuple[str, ...] = ()
    # A portfolio needs two or more loaded symbols so the mix has something to mix.
    if portfolio and len(reports) >= 2:
        portfolio_report, weight_map, portfolio_notes = _build_portfolio(
            reports,
            benchmark_report=benchmark_report,
            bench_history=bench_history,
            weights=weights,
            rebalance=rebalance_mode,
            start=start,
            end=end,
            ma_short=ma_short,
            ma_long=ma_long,
            risk_free=risk_free,
        )
        notes = notes + portfolio_notes
    elif portfolio and len(reports) < 2:
        notes = notes + ("Portfolio mode needs at least two loaded symbols.",)

    comparison = build_comparison(reports, benchmark_report, portfolio_report)
    figures = _build_figures(reports[chosen], benchmark_report, comparison)
    snapshot = _load_snapshot(chosen, offline=offline)
    export_csv = aligned_export(
        reports, benchmark_report=benchmark_report, portfolio_report=portfolio_report
    ).to_csv(index_label="date", float_format="%.8f")
    markdown = render_markdown(
        histories=histories,
        benchmark=bench_history,
        benchmark_error=bench_error,
        reports=reports,
        comparison=comparison,
        focus=chosen,
        failed=failed,
        notes=notes,
        portfolio_report=portfolio_report,
        portfolio_weights=weight_map,
        rebalance=rebalance_mode,
        snapshot=snapshot,
    )
    return Briefing(
        histories=histories,
        failed=failed,
        benchmark=bench_history,
        benchmark_error=bench_error,
        reports=reports,
        benchmark_report=benchmark_report,
        comparison=comparison,
        focus=chosen,
        figures=figures,
        markdown=markdown,
        notes=notes,
        portfolio_report=portfolio_report,
        portfolio_weights=weight_map,
        rebalance=rebalance_mode,
        snapshot=snapshot,
        export_csv=export_csv,
    )


def resolve_symbols(
    ticker: str | None,
    tickers: list[str] | tuple[str, ...] | None,
    benchmark: str | None,
) -> tuple[list[str], str, dict[str, str], tuple[str, ...]]:
    """
    Clean the request before anything is downloaded.

    Returns the symbols to brief (upper case, no repeats, benchmark
    removed), the benchmark, any entries that are not ticker-shaped with
    the reason, and notes about what was changed.
    """
    raw = list(tickers) if tickers else [ticker or DEFAULT_TICKER]
    bench_symbol = normalize_symbol(benchmark or DEFAULT_BENCHMARK)
    symbols: list[str] = []
    rejected: dict[str, str] = {}
    for entry in raw:
        try:
            symbol = normalize_symbol(entry)
        except MarketDataError as exc:
            rejected[str(entry).strip() or "(blank)"] = str(exc)
            continue
        if symbol not in symbols:
            symbols.append(symbol)
    if not symbols:
        raise MarketDataError(_join_failures(rejected))

    notes: list[str] = []
    if bench_symbol in symbols:
        if len(symbols) == 1:
            raise MarketDataError("Choose a benchmark that is a different symbol from the stock.")
        symbols.remove(bench_symbol)
        notes.append(
            f"{bench_symbol} is the benchmark, so it is drawn once, as the benchmark line, "
            "and not repeated as a compared symbol."
        )
    if len(symbols) > MAX_SYMBOLS:
        raise MarketDataError(
            f"Compare at most {MAX_SYMBOLS} symbols at a time. {len(symbols)} were given."
        )
    return symbols, bench_symbol, rejected, tuple(notes)


def render_markdown(
    *,
    histories: dict[str, PriceHistory],
    benchmark: PriceHistory | None,
    benchmark_error: str | None,
    reports: dict[str, PeriodReport],
    comparison: Comparison,
    focus: str,
    failed: dict[str, str],
    notes: tuple[str, ...],
    portfolio_report: PeriodReport | None = None,
    portfolio_weights: dict[str, float] | None = None,
    rebalance: str = REBALANCE_NONE,
    snapshot: CompanySnapshot | None = None,
) -> str:
    """Downloadable summary. The prose is the same sentences as the page."""
    report = reports[focus]
    stats = report.stats
    lines = [
        f"# {', '.join(reports)} compared with {stats.benchmark or 'no benchmark'}",
        "",
        "## Data",
        "",
    ]
    lines.extend(_source_line(history) for history in histories.values())
    lines.append(_source_line(benchmark) if benchmark is not None else "- Benchmark: not loaded")
    lines.extend(
        [
            f"- Window for {focus}: {stats.window_start.date().isoformat()} to "
            f"{stats.window_end.date().isoformat()}, {stats.n_days} trading days",
            f"- Moving averages: {stats.ma_short_window} and {stats.ma_long_window} days",
            f"- Risk-free rate: {stats.risk_free_rate:.1%} a year",
        ]
    )
    if portfolio_report is not None and portfolio_weights is not None:
        weight_text = ", ".join(f"{symbol} {weight:.1%}" for symbol, weight in portfolio_weights.items())
        lines.append(f"- Portfolio weights: {weight_text} (rebalance: {rebalance})")
    if snapshot is not None:
        lines.append(f"- Company snapshot: {_snapshot_line(snapshot)}")
    lines.extend(["", "## Side by side", ""])
    lines.extend(_markdown_table(comparison_table(comparison)))
    lines.extend(["", f"## {focus} in detail", ""])
    lines.extend(_result_lines(stats))
    year_line = describe_calendar_years(report.charts.calendar_years, ticker=focus)
    if year_line:
        lines.extend(["", year_line])
    lines.extend(["", "### Summary", ""])
    lines.extend(report.sentences)
    if portfolio_report is not None:
        lines.extend(["", "## Portfolio", ""])
        lines.extend(_result_lines(portfolio_report.stats))
        lines.extend(portfolio_report.sentences)
    if comparison.sentences:
        lines.extend(["", "## Across the symbols", ""])
        lines.extend(comparison.sentences)
    lines.extend(["", "## How the figures are defined", ""])
    lines.extend(f"- {note}" for note in methodology_notes(stats.ma_short_window, stats.ma_long_window))
    lines.extend(_footnotes(notes, failed, benchmark_error))
    lines.extend(
        [
            "",
            "The volatility label is a rank inside one symbol's own history. "
            "The benchmark gap is a difference of cumulative returns on shared dates.",
            "",
        ]
    )
    return "\n".join(lines)


def _build_portfolio(
    reports: dict[str, PeriodReport],
    *,
    benchmark_report: PeriodReport | None,
    bench_history: PriceHistory | None,
    weights: dict[str, float] | None,
    rebalance: str,
    start,
    end,
    ma_short: int,
    ma_long: int,
    risk_free: float,
) -> tuple[PeriodReport, dict[str, float], tuple[str, ...]]:
    """Build a portfolio report on shared sessions of the loaded symbols."""
    weight_map, notes = normalize_weights(weights, list(reports))
    series = {symbol: reports[symbol].charts.price for symbol in weight_map}
    shared = shared_sessions(series)
    value = portfolio_value(shared, weight_map, rebalance=rebalance)
    bench_frame = None if bench_history is None else bench_history.frame
    bench_ticker = None if benchmark_report is None else benchmark_report.stats.ticker
    report = build_period(
        portfolio_frame(value),
        ticker=PORTFOLIO_LABEL,
        start=start,
        end=end,
        benchmark=bench_frame,
        benchmark_ticker=bench_ticker,
        ma_short=ma_short,
        ma_long=ma_long,
        risk_free=risk_free,
    )
    mode_note = (
        "Portfolio rebalances monthly."
        if rebalance == "monthly"
        else "Portfolio rebalances quarterly."
        if rebalance == "quarterly"
        else "Portfolio is buy-and-hold after the opening weights."
    )
    return report, weight_map, notes + (mode_note,)


def _load_snapshot(symbol: str, *, offline: bool) -> CompanySnapshot | None:
    try:
        return fetch_company_snapshot(symbol, offline=offline)
    except MarketDataError:
        return None


def _snapshot_line(snapshot: CompanySnapshot) -> str:
    bits = [snapshot.ticker]
    if snapshot.name:
        bits.append(snapshot.name)
    if snapshot.sector:
        bits.append(snapshot.sector)
    if snapshot.market_cap is not None:
        bits.append(f"mkt cap {_fmt_cap(snapshot.market_cap)}")
    if snapshot.pe_ratio is not None:
        bits.append(f"P/E {snapshot.pe_ratio:.1f}")
    if snapshot.dividend_yield is not None:
        bits.append(f"div {snapshot.dividend_yield:.2%}")
    bits.append(f"[{snapshot.source}]")
    return " · ".join(bits)


def _fmt_cap(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1e12:
        return f"{value / 1e12:.2f}T"
    if abs_value >= 1e9:
        return f"{value / 1e9:.2f}B"
    if abs_value >= 1e6:
        return f"{value / 1e6:.2f}M"
    return f"{value:,.0f}"


def _load_symbols(

    symbols: list[str],
    *,
    offline: bool,
    refresh: bool,
    cache_dir: Path | None,
) -> tuple[dict[str, PriceHistory], dict[str, str]]:
    """Each symbol on its own, so one bad ticker does not sink the others."""
    histories: dict[str, PriceHistory] = {}
    failed: dict[str, str] = {}
    for symbol in symbols:
        try:
            histories[symbol] = _load_one(symbol, offline=offline, refresh=refresh, cache_dir=cache_dir)
        except MarketDataError as exc:
            failed[symbol] = str(exc)
    return histories, failed


def _load_one(symbol: str, *, offline: bool, refresh: bool, cache_dir: Path | None) -> PriceHistory:
    if offline:
        return load_offline_history(symbol)
    return fetch_history(symbol, refresh=refresh, cache_dir=cache_dir)


def _load_benchmark(symbol: str, *, offline: bool, refresh: bool, cache_dir: Path | None):
    try:
        return _load_one(symbol, offline=offline, refresh=refresh, cache_dir=cache_dir), None
    except MarketDataError as exc:
        return None, str(exc)


def _requested_window(histories: dict[str, PriceHistory], start, end, preset: str | None):
    """
    Explicit dates win. Otherwise the preset is measured back from the
    latest bar any compared symbol has, so every symbol is asked for the
    same calendar range and a symbol with less history shows less.
    """
    if start is not None and end is not None:
        return start, end
    if preset is None:
        raise AnalysisError("Provide a date range or a preset.")
    sessions = reduce(lambda left, right: left.union(right), (h.frame.index for h in histories.values()))
    return window_for_preset(sessions, preset)


def _build_reports(
    histories: dict[str, PriceHistory],
    benchmark: PriceHistory | None,
    *,
    start,
    end,
    ma_short: int,
    ma_long: int,
    risk_free: float,
) -> tuple[dict[str, PeriodReport], dict[str, str]]:
    reports: dict[str, PeriodReport] = {}
    failed: dict[str, str] = {}
    for symbol, history in histories.items():
        try:
            reports[symbol] = build_period(
                history.frame,
                ticker=symbol,
                start=start,
                end=end,
                benchmark=None if benchmark is None else benchmark.frame,
                benchmark_ticker=None if benchmark is None else benchmark.ticker,
                ma_short=ma_short,
                ma_long=ma_long,
                risk_free=risk_free,
            )
        except AnalysisError as exc:
            failed[symbol] = str(exc)
    return reports, failed


def _benchmark_report(
    benchmark: PriceHistory | None,
    *,
    start,
    end,
    ma_short: int,
    ma_long: int,
    risk_free: float,
) -> tuple[PeriodReport | None, str | None]:
    """The benchmark's own row: its return and risk over the same requested range."""
    if benchmark is None:
        return None, None
    try:
        report = build_period(
            benchmark.frame,
            ticker=benchmark.ticker,
            start=start,
            end=end,
            ma_short=ma_short,
            ma_long=ma_long,
            risk_free=risk_free,
        )
    except AnalysisError as exc:
        return None, f"{benchmark.ticker} cannot be used as the benchmark for this range. {exc}"
    return report, None


def _choose_focus(focus: str | None, reports: dict[str, PeriodReport]) -> str:
    """The requested symbol when it was briefed, otherwise the first one that was."""
    wanted = (focus or "").strip().upper()
    if wanted in reports:
        return wanted
    return next(iter(reports))


def _build_figures(
    report: PeriodReport,
    benchmark_report: PeriodReport | None,
    comparison: Comparison,
) -> dict:
    """Every chart for the symbol in focus, plus the one chart that shows all symbols."""
    stats = report.stats
    charts = report.charts
    averages = dict(ticker=stats.ticker, short_window=stats.ma_short_window, long_window=stats.ma_long_window)
    bench_name = stats.benchmark
    bench_drawdown = None if benchmark_report is None else benchmark_report.charts.drawdown
    relationship = None
    if bench_name:
        relationship = rolling_relationship_chart(
            charts.rolling_beta,
            charts.rolling_correlation,
            ticker=stats.ticker,
            benchmark=bench_name,
            window=ROLLING_BETA_WINDOW,
        )
    return {
        "comparison": comparison_chart(comparison.rebased, benchmark=comparison.benchmark, focus=stats.ticker),
        "price": price_chart(charts.price, charts.ma_short, charts.ma_long, **averages),
        "candles": candlestick_chart(charts.ohlc, charts.ma_short, charts.ma_long, **averages),
        "drawdown": drawdown_chart(
            charts.drawdown,
            ticker=stats.ticker,
            benchmark_drawdown=bench_drawdown,
            benchmark=comparison.benchmark,
        ),
        "volatility": volatility_chart(
            charts.rolling_vol,
            ticker=stats.ticker,
            window=ROLLING_VOL_WINDOW,
            period_vol=stats.annualized_volatility,
        ),
        "relationship": relationship,
        "histogram": return_histogram(
            charts.returns,
            ticker=stats.ticker,
            benchmark_returns=charts.benchmark_returns,
            benchmark=bench_name,
        ),
        "heatmap": monthly_heatmap(monthly_table(charts.monthly), ticker=stats.ticker),
    }


def _source_line(history: PriceHistory) -> str:
    return (
        f"- {history.ticker}: {history.source}. Latest bar "
        f"{history.latest_bar.date().isoformat()}. Fetched: {_fetched_label(history)}."
    )


def _result_lines(stats) -> list[str]:
    return [
        f"- Cumulative return: {_fmt_optional(stats.cumulative_return)}",
        f"- Annualized return: {_fmt_optional(stats.annualized_return)}",
        f"- Annualized volatility: {_fmt_optional(stats.annualized_volatility)}",
        f"- Volatility label: {stats.vol_label or 'not available'}",
        f"- Sharpe ratio: {_fmt_optional(stats.sharpe, '{:.2f}')}",
        f"- Sortino ratio: {_fmt_optional(stats.sortino, '{:.2f}')}",
        f"- 95% historical VaR: {_fmt_optional(stats.value_at_risk_95)}",
        f"- Maximum drawdown: {_fmt_optional(stats.max_drawdown)}",
        (
            f"- Versus {stats.benchmark}: "
            f"{_fmt_optional(None if stats.excess_return is None else stats.excess_return * 100, '{:+.1f}')} "
            "percentage points"
            if stats.benchmark
            else "- Versus benchmark: not available"
        ),
        f"- Beta: {_fmt_optional(stats.beta, '{:.2f}')}",
        f"- Correlation: {_fmt_optional(stats.correlation, '{:.2f}')}",
    ]



def _markdown_table(table: pd.DataFrame) -> list[str]:
    columns = [column for column in TABLE_FORMATS if column in table.columns]
    lines = [
        "| Symbol | " + " | ".join(columns) + " |",
        "|---|" + "---:|" * len(columns),
    ]
    for symbol, row in table.iterrows():
        cells = [format_cell(column, row[column]) for column in columns]
        lines.append(f"| {symbol} | " + " | ".join(cells) + " |")
    return lines


def _footnotes(notes: tuple[str, ...], failed: dict[str, str], benchmark_error: str | None) -> list[str]:
    lines: list[str] = []
    for note in notes:
        lines.extend(["", f"Note: {note}"])
    for symbol, reason in failed.items():
        lines.extend(["", f"Not included — {symbol}: {reason}"])
    if benchmark_error:
        lines.extend(["", f"Benchmark note: {benchmark_error}"])
    return lines


def _join_failures(failed: dict[str, str]) -> str:
    """One symbol keeps its own message. Several are listed by symbol."""
    if len(failed) == 1:
        return next(iter(failed.values()))
    return " ".join(f"{symbol}: {reason}" for symbol, reason in failed.items())


def _fetched_label(history: PriceHistory | None) -> str:
    if history is None or history.fetched_at is None:
        return "bundled file, no live refresh"
    stamp = pd.Timestamp(history.fetched_at)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    if history.stale:
        cache = " (older download, Yahoo did not answer)"
    elif history.from_cache:
        cache = " (cached download)"
    else:
        cache = ""
    return stamp.strftime("%Y-%m-%d %H:%M UTC") + cache


def _parse_weight_args(entries: list[str] | None) -> dict[str, float] | None:
    if not entries:
        return None
    weights: dict[str, float] = {}
    for entry in entries:
        if "=" not in entry:
            raise AnalysisError(f"Weight '{entry}' must look like AAPL=0.4.")
        symbol, raw = entry.split("=", 1)
        weights[normalize_symbol(symbol)] = float(raw)
    return weights


def _fmt_optional(value: float | None, pattern: str = "{:.1%}") -> str:
    if value is None:
        return "not available"
    return pattern.format(value)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Brief one or more stocks against a benchmark. "
            "Live data comes from Yahoo Finance. No API key."
        )
    )
    parser.add_argument("--ticker", default=None, help="One symbol. Same as --tickers with one entry.")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help=f"Up to {MAX_SYMBOLS} symbols to compare, separated by spaces.",
    )
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--focus", default=None, help="Which symbol the sentences describe. Defaults to the first.")
    parser.add_argument("--preset", default=DEFAULT_PRESET, choices=PRESETS)
    parser.add_argument("--start", default=None, help="YYYY-MM-DD. Overrides --preset.")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD. Overrides --preset.")
    parser.add_argument("--ma-short", type=int, default=MA_SHORT, help="Short moving-average window, in days.")
    parser.add_argument("--ma-long", type=int, default=MA_LONG, help="Long moving-average window, in days.")
    parser.add_argument(
        "--risk-free",
        type=float,
        default=DEFAULT_RISK_FREE * PERCENT,
        help="Annual risk-free rate in percent, used by Sharpe and Sortino.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use the bundled AAPL, MSFT, SPY and QQQ files. No network.",
    )
    parser.add_argument("--refresh", action="store_true", help="Ignore a fresh cache and download again.")
    parser.add_argument(
        "--portfolio",
        action="store_true",
        help="Build an equal-weight (or --weight) portfolio of the symbols.",
    )
    parser.add_argument(
        "--weight",
        action="append",
        default=None,
        metavar="SYMBOL=FRACTION",
        help="Portfolio weight, e.g. AAPL=0.4. Repeat per symbol. Default is equal weight.",
    )
    parser.add_argument(
        "--rebalance",
        default=REBALANCE_NONE,
        choices=REBALANCE_OPTIONS,
        help="Portfolio rebalance schedule: none, monthly, or quarterly.",
    )
    parser.add_argument("--output", default=OUTPUT_MD)
    parser.add_argument("--csv", default=None, help="Also write the aligned price/return CSV to this path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = _parse_args(argv)
    start = end = None
    preset = args.preset
    if args.start or args.end:
        if not (args.start and args.end):
            raise AnalysisError("Pass both --start and --end, or neither.")
        start, end = args.start, args.end
        preset = None
    weight_map = _parse_weight_args(args.weight)
    briefing = run_briefing(
        ticker=args.ticker,
        tickers=args.tickers,
        benchmark=args.benchmark,
        focus=args.focus,
        start=start,
        end=end,
        preset=preset,
        offline=args.offline,
        refresh=args.refresh,
        ma_short=args.ma_short,
        ma_long=args.ma_long,
        risk_free=args.risk_free / PERCENT,
        weights=weight_map,
        rebalance=args.rebalance,
        portfolio=args.portfolio or bool(weight_map),
    )
    path = Path(args.output)
    path.write_text(briefing.markdown, encoding="utf-8")
    if args.csv and briefing.export_csv:
        Path(args.csv).write_text(briefing.export_csv, encoding="utf-8")
        print(f"Saved CSV to {args.csv}")
    for sentence in briefing.report.sentences:
        print(sentence)
    for sentence in briefing.comparison.sentences:
        print(sentence)
    for symbol, reason in briefing.failed.items():
        print(f"Not included — {symbol}: {reason}")
    print(volatility_rule_text())
    print(f"Saved summary to {args.output}")
    return path.resolve()


if __name__ == "__main__":
    try:
        main()
    except (MarketDataError, AnalysisError) as exc:
        raise SystemExit(str(exc)) from exc
