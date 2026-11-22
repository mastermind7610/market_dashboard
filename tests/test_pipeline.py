"""Offline briefing, loader guards, cache, and a live Yahoo check."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from dashboard import run_briefing
from data_loader import (
    MarketDataError,
    fetch_history,
    load_offline_histories,
    load_prices,
    normalize_symbol,
    search_symbols,
)
from features import AnalysisError


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

    def test_offline_rejects_other_symbols(self) -> None:
        with self.assertRaises(MarketDataError):
            run_briefing(ticker="MSFT", offline=True, preset="Max")

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


if __name__ == "__main__":
    unittest.main()
