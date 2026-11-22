"""Known-answer tests for returns, volatility, drawdown, averages, and alignment."""

from __future__ import annotations

import math
import unittest

import pandas as pd

from features import (
    MA_LONG,
    MA_SHORT,
    VOL_ELEVATED_PERCENTILE,
    AnalysisError,
    align_adjusted,
    annualized_volatility,
    build_period,
    cumulative_return,
    describe_period,
    max_drawdown,
    moving_average,
    percentile_rank,
    volatility_label,
)


class KnownAnswerTests(unittest.TestCase):
    def test_cumulative_return(self) -> None:
        self.assertAlmostEqual(cumulative_return(pd.Series([100.0, 80.0])), -0.20)
        self.assertAlmostEqual(cumulative_return(pd.Series([100.0, 110.0, 121.0])), 0.21)
        with self.assertRaises(AnalysisError):
            cumulative_return(pd.Series([100.0]))

    def test_annualized_volatility_matches_hand_calculation(self) -> None:
        # Sample standard deviation of these four log returns, times sqrt(252).
        # mean = 0.00125
        # variance divisor is N-1 = 3
        # annualized result = 0.24571324750611234
        returns = pd.Series([0.01, -0.02, 0.015, 0.0])
        mean = sum(returns) / len(returns)
        variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
        expected = math.sqrt(variance) * math.sqrt(252)
        self.assertAlmostEqual(expected, 0.24571324750611234, places=12)
        self.assertAlmostEqual(annualized_volatility(returns), expected, places=12)
        with self.assertRaises(AnalysisError):
            annualized_volatility(pd.Series([0.01]))

    def test_max_drawdown_peak_and_trough(self) -> None:
        index = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08"])
        prices = pd.Series([100.0, 120.0, 110.0, 90.0, 95.0], index=index)
        result = max_drawdown(prices)
        self.assertAlmostEqual(result["max_drawdown"], -0.25)
        self.assertEqual(result["peak"].date().isoformat(), "2020-01-03")
        self.assertEqual(result["trough"].date().isoformat(), "2020-01-07")
        self.assertAlmostEqual(result["peak_price"], 120.0)
        self.assertAlmostEqual(result["trough_price"], 90.0)

    def test_drawdown_is_zero_on_a_rising_path(self) -> None:
        prices = pd.Series([10.0, 11.0, 12.0])
        result = max_drawdown(prices)
        self.assertAlmostEqual(result["max_drawdown"], 0.0)

    def test_moving_average_warmup(self) -> None:
        average = moving_average(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]), 3)
        self.assertTrue(pd.isna(average.iloc[0]))
        self.assertTrue(pd.isna(average.iloc[1]))
        self.assertEqual([float(value) for value in average.iloc[2:]], [2.0, 3.0, 4.0])

    def test_alignment_drops_unmatched_dates_and_does_not_fill(self) -> None:
        stock = pd.Series(
            [10.0, 11.0, 12.0],
            index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
        )
        benchmark = pd.Series(
            [20.0, 22.0, 21.0],
            index=pd.to_datetime(["2020-01-03", "2020-01-06", "2020-01-07"]),
        )
        aligned = align_adjusted(stock, benchmark)
        self.assertEqual(
            [stamp.date().isoformat() for stamp in aligned.index],
            ["2020-01-03", "2020-01-06"],
        )
        self.assertAlmostEqual(cumulative_return(aligned["stock"]), 12.0 / 11.0 - 1.0)
        self.assertAlmostEqual(cumulative_return(aligned["benchmark"]), 22.0 / 20.0 - 1.0)
        self.assertNotIn("2020-01-07", [stamp.date().isoformat() for stamp in aligned.index])

    def test_period_uses_adjusted_close_not_the_raw_close(self) -> None:
        dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
        stock = _frame([10.0, 10.0, 10.0], [100.0, 110.0, 121.0], dates)
        bench_dates = pd.to_datetime(["2020-01-03", "2020-01-06", "2020-01-07"])
        benchmark = _frame([1.0, 1.0, 1.0], [200.0, 220.0, 210.0], bench_dates)
        report = build_period(
            stock,
            ticker="DEMO",
            start="2020-01-01",
            end="2020-01-31",
            benchmark=benchmark,
            benchmark_ticker="BENCH",
        )
        self.assertAlmostEqual(report.stats.cumulative_return, 0.21)
        self.assertAlmostEqual(report.stats.first_price, 100.0)
        self.assertAlmostEqual(report.stats.last_price, 121.0)
        self.assertEqual(report.stats.aligned_days, 2)
        self.assertEqual(report.stats.dropped_stock_sessions, 1)
        self.assertAlmostEqual(report.stats.aligned_stock_return, 121.0 / 110.0 - 1.0)
        self.assertAlmostEqual(report.stats.benchmark_cumulative_return, 220.0 / 200.0 - 1.0)
        text = " ".join(report.sentences)
        self.assertIn("+21.0%", text)
        self.assertIn("2020-01-02", text)
        self.assertIn("BENCH", text)
        self.assertIsNone(report.stats.vol_label)
        self.assertIn("No quiet/typical/elevated label", text)

    def test_volatility_label_thresholds(self) -> None:
        calm = pd.Series([0.10] * 40)
        self.assertAlmostEqual(percentile_rank(0.50, calm), 1.0)
        self.assertEqual(volatility_label(1.0), "elevated")
        self.assertEqual(volatility_label(0.0), "quiet")
        self.assertEqual(volatility_label(0.50), "typical")
        self.assertGreaterEqual(VOL_ELEVATED_PERCENTILE, 0.80)

    def test_short_crash_window_is_elevated_against_a_calm_history(self) -> None:
        calm = [100.0 * math.exp(0.0004 * step) for step in range(80)]
        # Identical daily moves have zero volatility. The window has to
        # alternate large up and down days so the standard deviation is high.
        shocks = (0.90, 1.12, 0.86, 1.15, 0.88, 1.10)
        crash = []
        price = calm[-1]
        for shock in shocks:
            price *= shock
            crash.append(price)
        dates = pd.bdate_range("2020-01-02", periods=len(calm) + len(crash))
        frame = _frame([1.0] * len(dates), calm + crash, dates)
        window_start = dates[-6]
        report = build_period(
            frame,
            ticker="DEMO",
            start=window_start,
            end=dates[-1],
        )
        self.assertEqual(report.stats.vol_label, "elevated")
        self.assertGreaterEqual(report.stats.vol_percentile, VOL_ELEVATED_PERCENTILE)
        text = " ".join(describe_period(report.stats))
        self.assertIn("elevated", text)
        self.assertIn(f"{report.stats.vol_percentile * 100:.1f}", text)

    def test_sentences_quote_the_calculated_drawdown(self) -> None:
        index = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08"])
        # Pad a long calm history so the label has a baseline, then the window is the drawdown path.
        calm_dates = pd.bdate_range("2019-01-02", periods=60)
        calm_prices = [50.0 + step * 0.05 for step in range(60)]
        frame = _frame(
            [1.0] * (60 + 5),
            calm_prices + [100.0, 120.0, 110.0, 90.0, 95.0],
            calm_dates.append(index),
        )
        report = build_period(frame, ticker="DEMO", start="2020-01-02", end="2020-01-08")
        self.assertAlmostEqual(report.stats.max_drawdown, -0.25)
        text = " ".join(report.sentences)
        self.assertIn("-25.0%", text)
        self.assertIn("2020-01-03", text)
        self.assertIn("2020-01-07", text)
        self.assertIn("120.00", text)
        self.assertIn("90.00", text)

    def test_moving_averages_use_history_before_the_window(self) -> None:
        dates = pd.bdate_range("2020-01-02", periods=MA_LONG + 5)
        prices = [float(step + 1) for step in range(len(dates))]
        frame = _frame(prices, prices, dates)
        report = build_period(
            frame,
            ticker="DEMO",
            start=dates[-3],
            end=dates[-1],
        )
        self.assertIsNotNone(report.stats.ma_short)
        self.assertIsNotNone(report.stats.ma_long)
        expected_long = sum(prices[-MA_LONG:]) / MA_LONG
        # The last day of the window is the last price. The 200-day average
        # on that day uses the last 200 prices, which includes days before the window.
        self.assertAlmostEqual(report.stats.ma_long, expected_long, places=8)
        self.assertLess(report.stats.n_days, MA_SHORT)


def _frame(close: list[float], adjusted: list[float], index) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": [1_000.0] * len(close),
            "adj_close": adjusted,
        },
        index=pd.DatetimeIndex(index),
    )


if __name__ == "__main__":
    unittest.main()
