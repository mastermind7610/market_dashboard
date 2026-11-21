"""Lightweight tests for the offline risk-regime briefing."""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dashboard import analyze, format_regime, main
from data_loader import SAMPLE_CSV, load_prices
from features import TRADING_DAYS, build_features, regime_snapshot


def _write_prices(
    path: Path,
    n: int = 40,
    prefix: str | None = None,
    start: str = "2024-01-02",
    *,
    break_ohlc: bool = False,
    duplicate_date: bool = False,
    volume_nan_at: int | None = None,
    reverse: bool = False,
    extra_close: bool = False,
) -> None:
    dates = list(pd.bdate_range(start, periods=n))
    if duplicate_date and n > 1:
        dates[1] = dates[0]
    close = 100 + np.linspace(0, 5, n)
    open_ = close - 0.2
    high = close + 0.5
    low = close - 0.5
    if break_ohlc:
        high[3] = low[3] - 1
    volume = np.full(n, 1_000_000.0)
    if volume_nan_at is not None:
        volume[volume_nan_at] = np.nan
    frame = pd.DataFrame(
        {
            "Date": dates,
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )
    if extra_close:
        frame["AdjClose"] = close
    if prefix:
        frame = frame.rename(
            columns={
                "Open": f"{prefix}.Open",
                "High": f"{prefix}.High",
                "Low": f"{prefix}.Low",
                "Close": f"{prefix}.Close",
                "Volume": f"{prefix}.Volume",
            }
        )
    if reverse:
        frame = frame.iloc[::-1]
    frame.to_csv(path, index=False)


class PipelineTests(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    def test_sample_contract(self) -> None:
        prices = load_prices()
        self.assertEqual(prices.attrs["ticker"], "AAPL")
        self.assertEqual(list(prices.columns), ["open", "high", "low", "close", "volume"])
        self.assertEqual(int(prices.isna().sum().sum()), 0)
        self.assertTrue(prices.index.is_monotonic_increasing)
        self.assertFalse(prices.index.has_duplicates)
        self.assertEqual(prices.index.min().date().isoformat(), "2015-02-17")
        self.assertEqual(prices.index.max().date().isoformat(), "2017-02-16")
        self.assertAlmostEqual(float(prices.iloc[0]["close"]), 127.830002, places=4)
        self.assertTrue(SAMPLE_CSV.is_file())

    def test_sample_found_from_another_directory(self) -> None:
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            try:
                prices = load_prices()
            finally:
                os.chdir(previous)
        self.assertEqual(len(prices), 506)

    def test_generic_columns_and_explicit_ticker(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "generic.csv"
            _write_prices(path, prefix=None)
            prices = load_prices(path, ticker="demo")
        self.assertEqual(prices.attrs["ticker"], "DEMO")
        self.assertEqual(int(prices.isna().sum().sum()), 0)

    def test_prefixed_ticker_is_inferred(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "msft.csv"
            _write_prices(path, prefix="MSFT")
            prices = load_prices(path)
        self.assertEqual(prices.attrs["ticker"], "MSFT")

    def test_missing_file_and_missing_column(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_prices("/tmp/market-dashboard-missing.csv")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nocol.csv"
            path.write_text("Date,Open\n2024-01-02,1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_prices(path)

    def test_duplicate_dates_and_impossible_bars(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dup = Path(temp_dir) / "dup.csv"
            bad = Path(temp_dir) / "bad.csv"
            _write_prices(dup, duplicate_date=True)
            _write_prices(bad, break_ohlc=True)
            with self.assertRaises(ValueError):
                load_prices(dup)
            with self.assertRaises(ValueError):
                load_prices(bad)

    def test_unsorted_input_is_sorted_and_volume_gap_is_filled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "messy.csv"
            _write_prices(path, reverse=True, volume_nan_at=4)
            prices = load_prices(path)
        self.assertTrue(prices.index.is_monotonic_increasing)
        self.assertEqual(int(prices["volume"].isna().sum()), 0)
        self.assertEqual(float(prices["volume"].iloc[4]), 1_000_000.0)

    def test_log_return_warmup_and_annualization(self) -> None:
        prices = load_prices()
        feats = build_features(prices, vol_window=21, mom_window=63)
        self.assertTrue(pd.isna(feats["log_return"].iloc[0]))
        manual = float(np.log(feats["close"].iloc[1] / feats["close"].iloc[0]))
        self.assertAlmostEqual(float(feats["log_return"].iloc[1]), manual, places=10)

        # Index 21 is the first window that no longer touches the leading NaN.
        window = feats["log_return"].iloc[1:22]
        expected = float(window.std(ddof=1) * np.sqrt(TRADING_DAYS))
        self.assertAlmostEqual(float(feats["volatility_21d"].iloc[21]), expected, places=10)
        self.assertTrue(feats["volatility_21d"].iloc[:21].isna().all())
        self.assertTrue(feats["momentum_63d"].iloc[:63].isna().all())
        self.assertEqual(feats.attrs["vol_window"], 21)
        self.assertEqual(feats.attrs["mom_window"], 63)
        self.assertEqual(feats.attrs["ticker"], "AAPL")

    def test_sample_briefing(self) -> None:
        result = analyze()
        regime = result.regime
        text = format_regime(regime)
        self.assertEqual(regime["ticker"], "AAPL")
        self.assertEqual(regime["as_of"].date().isoformat(), "2017-02-16")
        self.assertEqual(regime["trend"], "uptrend")
        self.assertEqual(regime["volatility_regime"], "typical")
        self.assertGreater(regime["volatility_peak"], regime["volatility"])
        self.assertAlmostEqual(regime["close"], 135.350006, places=4)
        self.assertIn("AAPL", text)
        self.assertIn("2017-02-16", text)
        self.assertIn("uptrend", text)
        self.assertIn("typical", text)
        self.assertIn("21.2%", text)
        self.assertGreaterEqual(len(result.figure.axes), 4)
        texts = [ann.get_text() for ax in result.figure.axes for ann in ax.texts]
        self.assertTrue(any("China shock" in text for text in texts))

    def test_events_outside_the_file_are_not_drawn(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "later.csv"
            _write_prices(path, n=120, start="2024-01-02")
            result = analyze(path, vol_window=21, mom_window=63)
        texts = [ann.get_text() for ax in result.figure.axes for ann in ax.texts]
        self.assertFalse(any("China shock" in text for text in texts))
        self.assertEqual(result.regime["ticker"], "UNKNOWN")

    def test_short_history_and_tiny_window_fail_clearly(self) -> None:
        with self.assertRaises(ValueError):
            analyze(vol_window=1)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "short.csv"
            _write_prices(path, n=10)
            with self.assertRaises(ValueError):
                analyze(path, vol_window=21, mom_window=63)

    def test_main_writes_png_without_using_the_cwd_sample_lookup(self) -> None:
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    out = main(
                        ["--output", "briefing.png", "--vol-window", "21", "--mom-window", "63"]
                    )
            finally:
                os.chdir(previous)
            self.assertTrue(out.is_file())
            self.assertGreater(out.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
