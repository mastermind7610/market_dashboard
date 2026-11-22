"""Offline briefings, multi-symbol requests, loader guards, the cache, outages, and a live Yahoo check."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from dashboard import MAX_SYMBOLS, main, resolve_symbols, run_briefing
from data_loader import (
    CACHE_TTL_HOURS,
    OFFLINE_BENCHMARKS,
    MarketDataError,
    fetch_history,
    load_offline_histories,
    load_offline_history,
    load_prices,
    normalize_symbol,
    offline_span,
    offline_symbols,
    search_symbols,
)
from features import AnalysisError, comparison_table

FIGURE_NAMES = (
    "comparison",
    "price",
    "candles",
    "drawdown",
    "volatility",
    "relationship",
    "histogram",
    "heatmap",
)


class OfflineBriefingTests(unittest.TestCase):
    def test_bundled_files_cover_the_sample_window(self) -> None:
        apple, spy = load_offline_histories()
        self.assertEqual(apple.ticker, "AAPL")
        self.assertEqual(spy.ticker, "SPY")
        self.assertEqual(int(apple.frame.isna().sum().sum()), 0)
        self.assertEqual(int(spy.frame.isna().sum().sum()), 0)
        self.assertIn("adj_close", apple.frame.columns)
        self.assertEqual(apple.frame.index.min().date().isoformat(), "2015-02-17")
        self.assertEqual(apple.frame.index.max().date().isoformat(), "2017-02-16")
        self.assertLess(spy.frame.index.min(), apple.frame.index.min())
        self.assertGreaterEqual(spy.frame.index.max(), apple.frame.index.max())
        # The bundled adjusted column is not a copy of the raw close.
        self.assertGreater(
            abs(float(apple.frame["adj_close"].iloc[0]) - float(apple.frame["close"].iloc[0])),
            0.5,
        )

    def test_offline_briefing_has_a_benchmark_gap_and_a_download(self) -> None:
        briefing = run_briefing(offline=True, preset="Max")
        stats = briefing.report.stats
        self.assertEqual(stats.ticker, "AAPL")
        self.assertEqual(stats.benchmark, "SPY")
        self.assertGreater(stats.n_days, 400)
        self.assertIsNotNone(stats.annualized_volatility)
        self.assertLess(stats.max_drawdown, 0)
        self.assertIsNotNone(stats.excess_return)
        self.assertGreater(stats.aligned_days, 400)
        self.assertIn("AAPL", briefing.report.sentences[0])
        self.assertIn("SPY", " ".join(briefing.report.sentences))
        self.assertIn("percentile rank", " ".join(briefing.report.sentences))
        self.assertIn(stats.vol_label, " ".join(briefing.report.sentences))
        self.assertIn("Yahoo Finance", briefing.markdown)
        self.assertIn("plotly/datasets", briefing.markdown)
        self.assertIsNotNone(briefing.figures["price"])
        self.assertIsNotNone(briefing.figures["comparison"])
        self.assertIsNone(briefing.stock.fetched_at)

    def test_offline_rejects_symbols_that_are_not_bundled(self) -> None:
        with self.assertRaises(MarketDataError) as caught:
            run_briefing(ticker="NVDA", offline=True, preset="Max")
        self.assertIn("offline demo only includes", str(caught.exception))
        with self.assertRaises(MarketDataError):
            load_offline_history("NVDA")

    def test_every_bundled_file_loads_clean(self) -> None:
        self.assertEqual(offline_symbols(), ("AAPL", "MSFT", "SPY", "QQQ"))
        for symbol in offline_symbols():
            history = load_offline_history(symbol)
            self.assertEqual(history.ticker, symbol)
            self.assertEqual(int(history.frame.isna().sum().sum()), 0)
            self.assertGreater(len(history.frame), 500)
            self.assertIn("Offline demo", history.source)
            self.assertIsNone(history.fetched_at)
            self.assertFalse(history.stale)
        for symbol in OFFLINE_BENCHMARKS:
            self.assertIn(symbol, offline_symbols())
        first, last = offline_span()
        self.assertEqual(first.date().isoformat(), "2015-02-02")
        self.assertEqual(last.date().isoformat(), "2017-02-17")

    def test_new_figures_are_present_and_consistent_offline(self) -> None:
        briefing = run_briefing(offline=True, preset="Max")
        stats = briefing.report.stats
        for value in (stats.annualized_return, stats.sharpe, stats.sortino, stats.beta, stats.correlation):
            self.assertIsNotNone(value)
        self.assertIsNotNone(stats.up_capture)
        self.assertIsNotNone(stats.down_capture)
        self.assertEqual(stats.return_pairs, stats.aligned_days - 1)
        self.assertEqual(stats.up_days + stats.down_days <= stats.return_pairs, True)
        self.assertGreater(stats.best_day_return, 0)
        self.assertLess(stats.worst_day_return, 0)
        # A two-year window: the annualized figure sits between zero and the total.
        self.assertGreater(stats.annualized_return, 0)
        self.assertLess(stats.annualized_return, stats.cumulative_return)
        # The heatmap's months compound back to the headline return.
        compounded = (1.0 + briefing.report.charts.monthly).prod() - 1.0
        self.assertAlmostEqual(compounded, stats.cumulative_return, places=10)
        for name in FIGURE_NAMES:
            self.assertIsNotNone(briefing.figures[name], name)
        text = " ".join(briefing.report.sentences)
        for phrase in ("Sharpe ratio is", "beta is", "up capture", "best day was", "compounds to"):
            self.assertIn(phrase, text)


class MultiSymbolTests(unittest.TestCase):
    def test_two_symbols_against_a_chosen_benchmark(self) -> None:
        briefing = run_briefing(
            tickers=["aapl", "MSFT", "AAPL"],
            benchmark="qqq",
            focus="msft",
            offline=True,
            preset="Max",
        )
        self.assertEqual(list(briefing.reports), ["AAPL", "MSFT"])
        self.assertEqual(briefing.focus, "MSFT")
        self.assertEqual(briefing.report.stats.ticker, "MSFT")
        self.assertEqual(briefing.report.stats.benchmark, "QQQ")
        self.assertEqual(briefing.failed, {})
        self.assertEqual([row.ticker for row in briefing.comparison.rows], ["AAPL", "MSFT", "QQQ"])
        lines = briefing.comparison.rebased
        self.assertEqual(list(lines.columns), ["AAPL", "MSFT", "QQQ"])
        self.assertEqual(list(lines.iloc[0]), [100.0, 100.0, 100.0])
        # AAPL's file starts on 17 Feb 2015, later than the others, so that is where the lines begin.
        self.assertEqual(lines.index[0].date().isoformat(), "2015-02-17")
        self.assertEqual(len(briefing.figures["comparison"].data), 3)
        table = comparison_table(briefing.comparison)
        self.assertAlmostEqual(
            table.loc["MSFT", "Return"], briefing.reports["MSFT"].stats.cumulative_return, places=12
        )
        self.assertIn("| MSFT |", briefing.markdown)
        self.assertIn("| QQQ (benchmark) |", briefing.markdown)
        self.assertIn("## Across the symbols", briefing.markdown)
        self.assertIn("Ranked by cumulative return", " ".join(briefing.comparison.sentences))

    def test_focus_changes_the_sentences_but_not_the_table(self) -> None:
        on_apple = run_briefing(tickers=["AAPL", "MSFT"], focus="AAPL", offline=True, preset="1Y")
        on_microsoft = run_briefing(tickers=["AAPL", "MSFT"], focus="MSFT", offline=True, preset="1Y")
        self.assertIn("AAPL adjusted close", " ".join(on_apple.report.sentences))
        self.assertNotIn("MSFT adjusted close", " ".join(on_apple.report.sentences))
        self.assertIn("MSFT adjusted close", on_microsoft.report.sentences[0])
        # AAPL's bundled file ends a day before the others, and the briefing says so.
        self.assertIn("so its window ends on 2017-02-16", on_apple.report.sentences[0])
        self.assertTrue(comparison_table(on_apple.comparison).equals(comparison_table(on_microsoft.comparison)))
        unknown = run_briefing(tickers=["AAPL", "MSFT"], focus="TSLA", offline=True, preset="1Y")
        self.assertEqual(unknown.focus, "AAPL")

    def test_one_bad_symbol_is_left_out_and_the_rest_still_run(self) -> None:
        briefing = run_briefing(tickers=["AAPL", "NVDA", "bad sym"], offline=True, preset="1Y")
        self.assertEqual(list(briefing.reports), ["AAPL"])
        self.assertEqual(sorted(briefing.failed), ["NVDA", "bad sym"])
        self.assertIn("offline demo only includes", briefing.failed["NVDA"])
        self.assertIn("not a usable ticker", briefing.failed["bad sym"])
        self.assertIn("Not included — NVDA", briefing.markdown)

    def test_every_symbol_bad_is_an_error(self) -> None:
        with self.assertRaises(MarketDataError) as caught:
            run_briefing(tickers=["NVDA", "TSLA"], offline=True, preset="1Y")
        self.assertIn("NVDA", str(caught.exception))
        self.assertIn("TSLA", str(caught.exception))

    def test_the_benchmark_is_not_compared_with_itself(self) -> None:
        briefing = run_briefing(tickers=["AAPL", "SPY"], benchmark="SPY", offline=True, preset="1Y")
        self.assertEqual(list(briefing.reports), ["AAPL"])
        self.assertIn("SPY is the benchmark", briefing.notes[0])
        with self.assertRaises(MarketDataError):
            run_briefing(tickers=["SPY"], benchmark="SPY", offline=True, preset="1Y")

    def test_request_cleanup(self) -> None:
        symbols, benchmark, rejected, notes = resolve_symbols(None, [" msft ", "MSFT", "brk-b"], "qqq")
        self.assertEqual(symbols, ["MSFT", "BRK-B"])
        self.assertEqual(benchmark, "QQQ")
        self.assertEqual(rejected, {})
        self.assertEqual(notes, ())
        self.assertEqual(resolve_symbols(None, None, None)[:2], (["AAPL"], "SPY"))
        too_many = [f"SYM{letter}" for letter in "ABCDEF"]
        self.assertGreater(len(too_many), MAX_SYMBOLS)
        with self.assertRaises(MarketDataError) as caught:
            resolve_symbols(None, too_many, "SPY")
        self.assertIn(f"at most {MAX_SYMBOLS}", str(caught.exception))

    def test_settings_flow_through_and_bad_settings_stop_early(self) -> None:
        briefing = run_briefing(offline=True, preset="Max", ma_short=20, ma_long=100, risk_free=0.03)
        stats = briefing.report.stats
        self.assertEqual((stats.ma_short_window, stats.ma_long_window), (20, 100))
        self.assertAlmostEqual(stats.risk_free_rate, 0.03)
        zero_rate = run_briefing(offline=True, preset="Max").report.stats
        # A higher hurdle lowers the ratio and leaves the return alone.
        self.assertLess(stats.sharpe, zero_rate.sharpe)
        self.assertAlmostEqual(stats.cumulative_return, zero_rate.cumulative_return, places=12)
        self.assertIn("20-day average", " ".join(briefing.report.sentences))
        self.assertIn("Risk-free rate: 3.0% a year", briefing.markdown)
        with patch("dashboard.load_offline_history") as loader:
            with self.assertRaises(AnalysisError):
                run_briefing(offline=True, ma_short=200, ma_long=50)
            with self.assertRaises(AnalysisError):
                run_briefing(offline=True, risk_free=5.0)
            loader.assert_not_called()

    def test_a_symbol_with_no_sessions_in_the_range_is_left_out(self) -> None:
        # AAPL's bundled file starts on 17 Feb 2015. MSFT's starts on 2 Feb.
        briefing = run_briefing(
            tickers=["AAPL", "MSFT"], offline=True, start="2015-02-02", end="2015-02-13", preset=None
        )
        self.assertEqual(list(briefing.reports), ["MSFT"])
        self.assertIn("AAPL", briefing.failed)
        self.assertIn("No trading days", briefing.failed["AAPL"])

    def test_command_line_writes_a_multi_symbol_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "out.md"
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                path = main(
                    ["--offline", "--tickers", "AAPL", "MSFT", "--benchmark", "QQQ", "--preset", "Max",
                     "--ma-short", "20", "--ma-long", "100", "--risk-free", "2", "--output", str(output)]
                )
            text = path.read_text(encoding="utf-8")
        self.assertIn("Ranked by cumulative return", printed.getvalue())
        self.assertIn("# AAPL, MSFT compared with QQQ", text)
        self.assertIn("Moving averages: 20 and 100 days", text)
        self.assertIn("Risk-free rate: 2.0% a year", text)
        self.assertIn("| Symbol | Return |", text)

    def test_a_shorter_preset_changes_the_window_and_the_return(self) -> None:
        full = run_briefing(offline=True, preset="Max")
        year_to_date = run_briefing(offline=True, preset="YTD")
        self.assertLess(year_to_date.report.stats.n_days, full.report.stats.n_days)
        self.assertNotEqual(
            year_to_date.report.stats.cumulative_return,
            full.report.stats.cumulative_return,
        )
        self.assertEqual(year_to_date.report.stats.window_start.date().isoformat(), "2017-01-03")

    def test_one_day_window_is_a_clear_error(self) -> None:
        with self.assertRaises(AnalysisError) as caught:
            run_briefing(offline=True, start="2016-06-06", end="2016-06-06", preset=None)
        self.assertIn("two adjusted closes", str(caught.exception))

    def test_missing_and_garbage_symbols_never_need_the_network(self) -> None:
        with self.assertRaises(MarketDataError):
            normalize_symbol("  ")
        with self.assertRaises(MarketDataError):
            normalize_symbol("AAPL;rm")
        self.assertEqual(normalize_symbol("brk-b"), "BRK-B")

    def test_generic_csv_and_duplicate_dates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "book.csv"
            path.write_text(
                "Date,Open,High,Low,Close,Volume\n"
                "2024-01-02,10,11,9,10.5,100\n"
                "2024-01-03,10,11,9,11,100\n",
                encoding="utf-8",
            )
            frame = load_prices(path, ticker="demo")
            self.assertEqual(frame.attrs["ticker"], "DEMO")
            self.assertAlmostEqual(float(frame["adj_close"].iloc[-1]), 11.0)
            dup = Path(temp_dir) / "dup.csv"
            dup.write_text(
                "Date,Open,High,Low,Close,Volume\n"
                "2024-01-02,10,11,9,10,100\n"
                "2024-01-02,10,11,9,10,100\n",
                encoding="utf-8",
            )
            with self.assertRaises(MarketDataError):
                load_prices(dup)

    def test_cache_is_reused_until_refresh(self) -> None:
        frame = load_prices()
        calls = {"n": 0}

        def fake_download(ticker: str) -> pd.DataFrame:
            calls["n"] += 1
            out = frame.copy()
            out.attrs["ticker"] = ticker
            return out

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("data_loader._download_yahoo", fake_download):
                first = fetch_history("AAPL", refresh=True, cache_dir=temp_dir)
                second = fetch_history("AAPL", refresh=False, cache_dir=temp_dir)
                third = fetch_history("AAPL", refresh=True, cache_dir=temp_dir)
        self.assertEqual(calls["n"], 2)
        self.assertFalse(first.from_cache)
        self.assertTrue(second.from_cache)
        self.assertEqual(second.fetched_at, first.fetched_at)
        self.assertFalse(third.from_cache)

    def test_an_outage_serves_the_expired_download_and_says_so(self) -> None:
        frame = load_prices()

        def working(ticker: str) -> pd.DataFrame:
            out = frame.copy()
            out.attrs["ticker"] = ticker
            return out

        def down(ticker: str) -> pd.DataFrame:
            raise MarketDataError(f"No price history for {ticker}.")

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("data_loader._download_yahoo", working):
                first = fetch_history("AAPL", refresh=True, cache_dir=temp_dir)
            # Age the saved download past the reuse window.
            meta_path = Path(temp_dir) / "AAPL.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            old = datetime.now(timezone.utc) - timedelta(hours=CACHE_TTL_HOURS + 6)
            meta["fetched_at"] = pd.Timestamp(old).isoformat()
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            with patch("data_loader._download_yahoo", down):
                fallback = fetch_history("AAPL", cache_dir=temp_dir)
                with self.assertRaises(MarketDataError):
                    fetch_history("MSFT", cache_dir=temp_dir)
                briefing = run_briefing(tickers=["AAPL", "MSFT"], benchmark="SPY", cache_dir=temp_dir)
        self.assertFalse(first.stale)
        self.assertTrue(fallback.stale)
        self.assertTrue(fallback.from_cache)
        self.assertLess(fallback.fetched_at, first.fetched_at)
        self.assertEqual(len(fallback.frame), len(frame))
        # The briefing still runs on what is on disk, and names what it could not get.
        self.assertEqual(list(briefing.reports), ["AAPL"])
        self.assertEqual(briefing.stale, ("AAPL",))
        self.assertIn("MSFT", briefing.failed)
        self.assertIsNotNone(briefing.benchmark_error)
        self.assertIsNone(briefing.report.stats.beta)
        self.assertIn("older download", briefing.markdown)

    def test_an_outage_with_nothing_on_disk_is_a_clear_error(self) -> None:
        def down(ticker: str) -> pd.DataFrame:
            raise MarketDataError(f"No price history for {ticker}.")

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("data_loader._download_yahoo", down):
                with self.assertRaises(MarketDataError) as caught:
                    run_briefing(tickers=["AAPL"], cache_dir=temp_dir)
        self.assertIn("No price history for AAPL", str(caught.exception))


class LiveYahooTests(unittest.TestCase):
    """Hits Yahoo Finance. Skips only when the host cannot be reached."""

    def test_live_aapl_spy_search_and_a_bad_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                apple = fetch_history("AAPL", refresh=True, cache_dir=temp_dir)
                spy = fetch_history("SPY", refresh=True, cache_dir=temp_dir)
                hits = search_symbols("microsoft")
            except MarketDataError as exc:
                self.skipTest(f"Yahoo Finance unreachable: {exc}")
        self.assertGreater(len(apple.frame), 2000)
        self.assertEqual(apple.source.startswith("Yahoo Finance"), True)
        self.assertIsNotNone(apple.fetched_at)
        today = datetime.now(timezone.utc).date()
        self.assertLessEqual((today - apple.latest_bar.date()).days, 7)
        self.assertLessEqual((today - spy.latest_bar.date()).days, 7)
        self.assertTrue(any(hit.symbol == "MSFT" for hit in hits))
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(MarketDataError):
                fetch_history("ZZZNOTREAL", refresh=True, cache_dir=temp_dir)

    def test_live_multi_symbol_briefing_with_a_bad_ticker(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                briefing = run_briefing(
                    tickers=["AAPL", "MSFT", "ZZZNOTREAL"],
                    benchmark="QQQ",
                    preset="1Y",
                    refresh=True,
                    cache_dir=temp_dir,
                )
            except MarketDataError as exc:
                self.skipTest(f"Yahoo Finance unreachable: {exc}")
        self.assertEqual(list(briefing.reports), ["AAPL", "MSFT"])
        self.assertIn("ZZZNOTREAL", briefing.failed)
        self.assertEqual(briefing.stale, ())
        for report in briefing.reports.values():
            stats = report.stats
            self.assertEqual(stats.benchmark, "QQQ")
            self.assertGreater(stats.n_days, 200)
            self.assertIsNotNone(stats.beta)
            self.assertGreaterEqual(stats.correlation, -1.0)
            self.assertLessEqual(stats.correlation, 1.0)
            self.assertIsNotNone(stats.sharpe)
            self.assertIsNotNone(stats.annualized_return)
        self.assertEqual(list(briefing.comparison.rebased.iloc[0]), [100.0, 100.0, 100.0])
        for name in FIGURE_NAMES:
            self.assertIsNotNone(briefing.figures[name], name)


if __name__ == "__main__":
    unittest.main()
