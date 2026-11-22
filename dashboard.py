"""
dashboard.py — THE ORCHESTRATOR
================================
The only place a briefing is assembled:

    prices  →  features  →  sentences  →  charts  →  markdown

``python dashboard.py`` reads live Yahoo Finance data.
``python dashboard.py --offline`` uses the bundled Apple file and the
saved SPY sample, and makes no network call.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data_loader import (
    MarketDataError,
    PriceHistory,
    fetch_history,
    load_offline_histories,
    normalize_symbol,
)
from features import (
    MA_LONG,
    MA_SHORT,
    PRESETS,
    ROLLING_VOL_WINDOW,
    AnalysisError,
    PeriodReport,
    methodology_notes,
    volatility_rule_text,
    window_for_preset,
    build_period,
)
from plots import comparison_chart, drawdown_chart, price_chart, volatility_chart


DEFAULT_TICKER = "AAPL"
DEFAULT_BENCHMARK = "SPY"
DEFAULT_PRESET = "1Y"
OUTPUT_MD = "analysis.md"


@dataclass(frozen=True)
class Briefing:
    stock: PriceHistory
    benchmark: PriceHistory | None
    benchmark_error: str | None
    report: PeriodReport
    figures: dict
    markdown: str


def run_briefing(
    *,
    ticker: str = DEFAULT_TICKER,
    benchmark: str = DEFAULT_BENCHMARK,
    start=None,
    end=None,
    preset: str | None = DEFAULT_PRESET,
    offline: bool = False,
    refresh: bool = False,
    cache_dir: Path | None = None,
) -> Briefing:
    """
    One briefing. ``preset`` is used when start or end is omitted.
    Pass both dates to ignore the preset.
    """
    if offline:
        stock, bundled_bench = load_offline_histories()
        if ticker not in (None, "", DEFAULT_TICKER) or benchmark not in (None, "", DEFAULT_BENCHMARK):
            symbol = ticker or DEFAULT_TICKER
            other = benchmark or DEFAULT_BENCHMARK
            if symbol != DEFAULT_TICKER or other != DEFAULT_BENCHMARK:
                raise MarketDataError(
                    "The offline demo only includes bundled AAPL and SPY. "
                    "Switch to live market data for other symbols."
                )
        bench_history: PriceHistory | None = bundled_bench
        bench_error = None
    else:
        stock_symbol = normalize_symbol(ticker)
        bench_symbol = normalize_symbol(benchmark)
        if stock_symbol == bench_symbol:
            raise MarketDataError(
                "Choose a benchmark that is a different symbol from the stock."
            )
        stock = fetch_history(stock_symbol, refresh=refresh, cache_dir=cache_dir)
        bench_history, bench_error = _load_benchmark(bench_symbol, refresh=refresh, cache_dir=cache_dir)

    if start is None or end is None:
        if preset is None:
            raise AnalysisError("Provide a date range or a preset.")
        start, end = window_for_preset(stock.frame.index, preset)

    report = build_period(
        stock.frame,
        ticker=stock.ticker,
        start=start,
        end=end,
        benchmark=None if bench_history is None else bench_history.frame,
        benchmark_ticker=None if bench_history is None else bench_history.ticker,
    )
    figures = {
        "price": price_chart(
            report.charts.price,
            report.charts.ma_short,
            report.charts.ma_long,
            ticker=stock.ticker,
            short_window=MA_SHORT,
            long_window=MA_LONG,
        ),
        "drawdown": drawdown_chart(report.charts.drawdown, ticker=stock.ticker),
        "comparison": comparison_chart(
            report.charts.aligned,
            ticker=stock.ticker,
            benchmark=bench_history.ticker if bench_history is not None else DEFAULT_BENCHMARK,
        ),
        "volatility": volatility_chart(
            report.charts.rolling_vol,
            ticker=stock.ticker,
            window=ROLLING_VOL_WINDOW,
            period_vol=report.stats.annualized_volatility,
        ),
    }
    markdown = render_markdown(stock, bench_history, bench_error, report)
    return Briefing(
        stock=stock,
        benchmark=bench_history,
        benchmark_error=bench_error,
        report=report,
        figures=figures,
        markdown=markdown,
    )


def render_markdown(
    stock: PriceHistory,
    benchmark: PriceHistory | None,
    benchmark_error: str | None,
    report: PeriodReport,
) -> str:
    """Downloadable summary. The prose is the same sentences as the page."""
    stats = report.stats
    fetched = _fetched_label(stock)
    bench_fetched = _fetched_label(benchmark) if benchmark is not None else "unavailable"
    lines = [
        f"# {stock.ticker} compared with {stats.benchmark or 'no benchmark'}",
        "",
        f"- Stock source: {stock.source}",
        f"- Stock latest bar: {stock.latest_bar.date().isoformat()}",
        f"- Stock fetched: {fetched}",
        f"- Benchmark source: {benchmark.source if benchmark is not None else 'not loaded'}",
        f"- Benchmark fetched: {bench_fetched}",
        f"- Window: {stats.window_start.date().isoformat()} to {stats.window_end.date().isoformat()}",
        f"- Trading days: {stats.n_days}",
        "",
        "## Results",
        "",
        f"- Cumulative return: {stats.cumulative_return:+.1%}",
        f"- Annualized volatility: {_fmt_optional(stats.annualized_volatility)}",
        f"- Maximum drawdown: {_fmt_optional(stats.max_drawdown)}",
        f"- Volatility label: {stats.vol_label or 'not ranked'}",
        (
            f"- Versus {stats.benchmark}: {stats.excess_return * 100:+.1f} percentage points"
            if stats.excess_return is not None and stats.benchmark
            else "- Versus benchmark: not available"
        ),
        "",
        "## Summary",
        "",
    ]
    lines.extend(report.sentences)
    lines.extend(["", "## How the figures are defined", ""])
    lines.extend(f"- {note}" for note in methodology_notes())
    if benchmark_error:
        lines.extend(["", f"Benchmark note: {benchmark_error}"])
    lines.extend(
        [
            "",
            "The volatility label is a rank inside this symbol's own history. "
            "The benchmark gap is a difference of cumulative returns on shared dates.",
            "",
        ]
    )
    return "\n".join(lines)


def _load_benchmark(symbol: str, *, refresh: bool, cache_dir: Path | None):
    try:
        return fetch_history(symbol, refresh=refresh, cache_dir=cache_dir), None
    except MarketDataError as exc:
        return None, str(exc)


def _fetched_label(history: PriceHistory | None) -> str:
    if history is None or history.fetched_at is None:
        return "bundled file, no live refresh"
    stamp = pd.Timestamp(history.fetched_at)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    cache = " (cached download)" if history.from_cache else ""
    return stamp.strftime("%Y-%m-%d %H:%M UTC") + cache


def _fmt_optional(value: float | None) -> str:
    if value is None:
        return "not available"
    return f"{value:.1%}"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Brief one stock against a benchmark. "
            "Live data comes from Yahoo Finance. No API key."
        )
    )
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--preset", default=DEFAULT_PRESET, choices=PRESETS)
    parser.add_argument("--start", default=None, help="YYYY-MM-DD. Overrides --preset.")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD. Overrides --preset.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use the bundled AAPL and SPY files. No network.",
    )
    parser.add_argument("--refresh", action="store_true", help="Ignore a fresh cache and download again.")
    parser.add_argument("--output", default=OUTPUT_MD)
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
    briefing = run_briefing(
        ticker=args.ticker,
        benchmark=args.benchmark,
        start=start,
        end=end,
        preset=preset,
        offline=args.offline,
        refresh=args.refresh,
    )
    path = Path(args.output)
    path.write_text(briefing.markdown, encoding="utf-8")
    for sentence in briefing.report.sentences:
        print(sentence)
    print(volatility_rule_text())
    print(f"Saved summary to {args.output}")
    return path.resolve()


if __name__ == "__main__":
    try:
        main()
    except (MarketDataError, AnalysisError) as exc:
        raise SystemExit(str(exc)) from exc
