"""
Known-answer tests for the metrics added with the multi-symbol briefing:
annualized return, Sharpe, Sortino, beta, correlation, capture, extreme
days, rolling beta and correlation, monthly returns, adjusted candles,
rebasing, the labels, and the side-by-side comparison.

Every expected figure is worked out in the test with plain arithmetic,
not by calling the function under test a second time.
"""

from __future__ import annotations

import math
import unittest

import pandas as pd

from features import (
    BETA_HIGH,
    BETA_LOW,
    CORRELATION_STRONG,
    CORRELATION_WEAK,
    MIN_ANNUALIZE_CALENDAR_DAYS,
    MIN_RETURNS_FOR_RATIOS,
    MISSING,
    TRADING_DAYS,
    AnalysisError,
    adjusted_ohlc,
    annualized_return,
    beta,
    beta_label,
    build_comparison,
    build_period,
    capture_ratio,
    comparison_table,
    correlation,
    correlation_label,
    daily_risk_free,
    daily_simple_returns,
    downside_deviation,
    extreme_days,
    format_cell,
    monthly_returns,
    monthly_table,
    paired_returns,
    positive_share,
    rebase,
    rolling_beta,
    rolling_correlation,
    shared_sessions,
    sharpe_ratio,
    sortino_ratio,
    validate_average_windows,
    validate_risk_free,
)

# Four daily returns used by the ratio tests. Mean 0.00125, sample
# standard deviation 0.0154784796, one day below zero (−0.02).
FOUR_RETURNS = [0.01, -0.02, 0.015, 0.0]


class ReturnTests(unittest.TestCase):
    def test_simple_returns(self) -> None:
        returns = daily_simple_returns(pd.Series([100.0, 110.0, 99.0]))
        self.assertTrue(pd.isna(returns.iloc[0]))
        self.assertAlmostEqual(returns.iloc[1], 0.10)
        self.assertAlmostEqual(returns.iloc[2], -0.10)

    def test_annualized_return_over_one_year(self) -> None:
        # 100 to 110 across 365 calendar days: 1.10 ** (365.25 / 365) − 1.
        prices = pd.Series([100.0, 110.0], index=pd.to_datetime(["2021-01-01", "2022-01-01"]))
        self.assertAlmostEqual(annualized_return(prices), 0.10007181138351062, places=12)

    def test_annualized_return_over_two_years_is_about_ten_percent(self) -> None:
        # 100 to 121 across 731 calendar days: 1.21 ** (365.25 / 731) − 1.
        prices = pd.Series([100.0, 121.0], index=pd.to_datetime(["2019-01-01", "2021-01-01"]))
        self.assertAlmostEqual(annualized_return(prices), 0.09992829153182337, places=12)
        self.assertAlmostEqual(annualized_return(prices), 0.10, places=3)

    def test_annualized_return_is_withheld_for_a_part_year(self) -> None:
        short = pd.Series([100.0, 105.0], index=pd.to_datetime(["2021-01-01", "2021-06-30"]))
        self.assertLess((short.index[-1] - short.index[0]).days, MIN_ANNUALIZE_CALENDAR_DAYS)
        with self.assertRaises(AnalysisError):
            annualized_return(short)

    def test_annualized_return_needs_dates(self) -> None:
        with self.assertRaises(AnalysisError):
            annualized_return(pd.Series([100.0, 110.0]))

    def test_extreme_days_and_share_of_gains(self) -> None:
        index = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
        returns = pd.Series([0.01, -0.03, 0.04, 0.0], index=index)
        result = extreme_days(returns)
        self.assertEqual(result["best_day"].date().isoformat(), "2020-01-06")
        self.assertAlmostEqual(result["best_day_return"], 0.04)
        self.assertEqual(result["worst_day"].date().isoformat(), "2020-01-03")
        self.assertAlmostEqual(result["worst_day_return"], -0.03)
        # Two of four returns are above zero. A flat day is not a gain.
        self.assertAlmostEqual(positive_share(returns), 0.5)
        with self.assertRaises(AnalysisError):
            extreme_days(pd.Series([], dtype=float))


class RatioTests(unittest.TestCase):
    def test_sharpe_matches_hand_calculation(self) -> None:
        mean = sum(FOUR_RETURNS) / 4
        variance = sum((value - mean) ** 2 for value in FOUR_RETURNS) / 3
        expected = mean / math.sqrt(variance) * math.sqrt(TRADING_DAYS)
        self.assertAlmostEqual(expected, 1.2819821609014552, places=12)
        self.assertAlmostEqual(sharpe_ratio(pd.Series(FOUR_RETURNS)), expected, places=12)

    def test_sharpe_subtracts_the_daily_risk_free_rate(self) -> None:
        # 5% a year is 1.05 ** (1/252) − 1 a day. The spread is unchanged
        # by subtracting a constant, so only the mean moves.
        self.assertAlmostEqual(daily_risk_free(0.05), 0.00019363050654397362, places=15)
        self.assertAlmostEqual(sharpe_ratio(pd.Series(FOUR_RETURNS), 0.05), 1.0833974767449057, places=10)
        self.assertLess(sharpe_ratio(pd.Series(FOUR_RETURNS), 0.05), sharpe_ratio(pd.Series(FOUR_RETURNS)))

    def test_sharpe_is_undefined_for_flat_or_single_returns(self) -> None:
        with self.assertRaises(AnalysisError):
            sharpe_ratio(pd.Series([0.01, 0.01, 0.01]))
        with self.assertRaises(AnalysisError):
            sharpe_ratio(pd.Series([0.01]))

    def test_downside_deviation_keeps_every_day_in_the_divisor(self) -> None:
        # One shortfall of −0.02 among four days: sqrt(0.02**2 / 4) = 0.01.
        self.assertAlmostEqual(downside_deviation(pd.Series(FOUR_RETURNS)), 0.01, places=12)

    def test_sortino_matches_hand_calculation(self) -> None:
        # Mean 0.00125 over downside deviation 0.01, times sqrt(252).
        expected = 0.00125 / 0.01 * math.sqrt(TRADING_DAYS)
        self.assertAlmostEqual(expected, 1.9843134832984426, places=12)
        self.assertAlmostEqual(sortino_ratio(pd.Series(FOUR_RETURNS)), expected, places=12)
        self.assertAlmostEqual(sortino_ratio(pd.Series(FOUR_RETURNS), 0.05), 1.6607786271293632, places=10)

    def test_sortino_is_undefined_without_a_losing_day(self) -> None:
        with self.assertRaises(AnalysisError):
            sortino_ratio(pd.Series([0.01, 0.02, 0.0, 0.005]))

    def test_a_losing_window_has_negative_ratios(self) -> None:
        losses = pd.Series([-0.01, 0.005, -0.02, -0.003])
        self.assertLess(sharpe_ratio(losses), 0)
        self.assertLess(sortino_ratio(losses), 0)


class BenchmarkRelationshipTests(unittest.TestCase):
    def test_beta_and_correlation_match_hand_calculation(self) -> None:
        benchmark = [0.01, -0.01, 0.02, 0.0]
        stock = [0.02, -0.01, 0.01, 0.01]
        mean_b = sum(benchmark) / 4
        mean_s = sum(stock) / 4
        covariance = sum((s - mean_s) * (b - mean_b) for s, b in zip(stock, benchmark)) / 3
        variance_b = sum((b - mean_b) ** 2 for b in benchmark) / 3
        variance_s = sum((s - mean_s) ** 2 for s in stock) / 3
        self.assertAlmostEqual(covariance / variance_b, 0.7, places=12)
        self.assertAlmostEqual(beta(pd.Series(stock), pd.Series(benchmark)), 0.7, places=12)
        self.assertAlmostEqual(
            correlation(pd.Series(stock), pd.Series(benchmark)),
            covariance / math.sqrt(variance_b * variance_s),
            places=12,
        )
        self.assertAlmostEqual(correlation(pd.Series(stock), pd.Series(benchmark)), 0.7181848464596079, places=12)

    def test_a_scaled_copy_has_that_beta_and_perfect_correlation(self) -> None:
        benchmark = pd.Series([0.01, -0.02, 0.005, 0.03, -0.01])
        self.assertAlmostEqual(beta(benchmark * 2.0, benchmark), 2.0, places=12)
        self.assertAlmostEqual(correlation(benchmark * 2.0, benchmark), 1.0, places=12)
        self.assertAlmostEqual(beta(benchmark * -0.5, benchmark), -0.5, places=12)
        self.assertAlmostEqual(correlation(benchmark * -0.5, benchmark), -1.0, places=12)

    def test_beta_uses_only_dates_both_sides_have(self) -> None:
        dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
        benchmark = pd.Series([0.01, -0.02, 0.03, 0.01], index=dates)
        # The stock is three times the benchmark where it trades, and has
        # an extra date the benchmark lacks. That date must be ignored.
        stock = pd.Series(
            [0.03, -0.06, 0.09, 0.50],
            index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-08"]),
        )
        self.assertAlmostEqual(beta(stock, benchmark), 3.0, places=12)

    def test_flat_benchmark_has_no_beta_or_correlation(self) -> None:
        flat = pd.Series([0.0, 0.0, 0.0, 0.0])
        moving = pd.Series([0.01, -0.01, 0.02, 0.0])
        with self.assertRaises(AnalysisError):
            beta(moving, flat)
        with self.assertRaises(AnalysisError):
            correlation(moving, flat)

    def test_capture_ratios(self) -> None:
        benchmark = pd.Series([0.01, 0.02, -0.01, -0.03, 0.0])
        stock = pd.Series([0.02, 0.01, -0.005, -0.045, 0.01])
        # Up days: stock mean 0.015 over benchmark mean 0.015.
        self.assertAlmostEqual(capture_ratio(stock, benchmark, side="up"), 1.0, places=12)
        # Down days: stock mean −0.025 over benchmark mean −0.02. The flat
        # benchmark day belongs to neither side.
        self.assertAlmostEqual(capture_ratio(stock, benchmark, side="down"), 1.25, places=12)

    def test_capture_needs_a_day_on_that_side(self) -> None:
        only_up = pd.Series([0.01, 0.02, 0.005])
        with self.assertRaises(AnalysisError):
            capture_ratio(only_up, only_up, side="down")
        with self.assertRaises(AnalysisError):
            capture_ratio(only_up, only_up, side="sideways")

    def test_rolling_beta_and_correlation_warm_up_then_hold(self) -> None:
        dates = pd.bdate_range("2020-01-02", periods=6)
        benchmark = pd.Series([0.01, -0.02, 0.005, 0.03, -0.01, 0.02], index=dates)
        stock = benchmark * 1.5
        betas = rolling_beta(stock, benchmark, window=3)
        correlations = rolling_correlation(stock, benchmark, window=3)
        self.assertTrue(betas.iloc[:2].isna().all())
        self.assertTrue(correlations.iloc[:2].isna().all())
        for value in betas.iloc[2:]:
            self.assertAlmostEqual(value, 1.5, places=10)
        for value in correlations.iloc[2:]:
            self.assertAlmostEqual(value, 1.0, places=10)

    def test_paired_returns_span_the_same_two_sessions(self) -> None:
        dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
        aligned = pd.DataFrame({"stock": [10.0, 11.0, 9.9], "benchmark": [100.0, 102.0, 102.0]}, index=dates)
        pairs = paired_returns(aligned)
        self.assertEqual(len(pairs), 2)
        self.assertAlmostEqual(pairs["stock"].iloc[0], 0.10)
        self.assertAlmostEqual(pairs["benchmark"].iloc[0], 0.02)
        self.assertAlmostEqual(pairs["stock"].iloc[1], -0.10)
        self.assertAlmostEqual(pairs["benchmark"].iloc[1], 0.0)


class LabelTests(unittest.TestCase):
    def test_beta_label_cuts(self) -> None:
        self.assertEqual(beta_label(BETA_HIGH), "high")
        self.assertEqual(beta_label(1.19), "market-like")
        self.assertEqual(beta_label(1.0), "market-like")
        self.assertEqual(beta_label(0.81), "market-like")
        self.assertEqual(beta_label(BETA_LOW), "low")
        self.assertEqual(beta_label(0.0), "low")
        self.assertEqual(beta_label(-0.1), "inverse")

    def test_correlation_label_cuts(self) -> None:
        self.assertEqual(correlation_label(CORRELATION_STRONG), "strong")
        self.assertEqual(correlation_label(0.69), "moderate")
        self.assertEqual(correlation_label(CORRELATION_WEAK), "moderate")
        self.assertEqual(correlation_label(0.29), "weak")
        self.assertEqual(correlation_label(-0.8), "strong negative")
        self.assertEqual(correlation_label(-0.1), "weak negative")

    def test_settings_are_validated(self) -> None:
        self.assertEqual(validate_average_windows(50, 200), (50, 200))
        self.assertEqual(validate_average_windows(20.0, 100.0), (20, 100))
        for short, long in ((200, 50), (50, 50), (1, 10), (10, 401), ("x", 10)):
            with self.assertRaises(AnalysisError):
                validate_average_windows(short, long)
        self.assertAlmostEqual(validate_risk_free(0.03), 0.03)
        for rate in (-0.01, 0.5, float("nan"), "x"):
            with self.assertRaises(AnalysisError):
                validate_risk_free(rate)


class MonthlyAndCandleTests(unittest.TestCase):
    def test_monthly_returns_chain_to_the_cumulative_return(self) -> None:
        index = pd.to_datetime(
            ["2020-01-15", "2020-01-31", "2020-02-14", "2020-02-28", "2020-03-02", "2020-03-10"]
        )
        prices = pd.Series([100.0, 110.0, 105.0, 121.0, 120.0, 108.9], index=index)
        monthly = monthly_returns(prices)
        self.assertEqual([str(period) for period in monthly.index], ["2020-01", "2020-02", "2020-03"])
        # January starts from the first close in the series, not from a
        # December close the series does not contain.
        self.assertAlmostEqual(monthly.iloc[0], 110.0 / 100.0 - 1.0)
        self.assertAlmostEqual(monthly.iloc[1], 121.0 / 110.0 - 1.0)
        self.assertAlmostEqual(monthly.iloc[2], 108.9 / 121.0 - 1.0)
        compounded = (1.0 + monthly).prod() - 1.0
        self.assertAlmostEqual(compounded, 108.9 / 100.0 - 1.0, places=12)

    def test_a_month_with_one_session_at_the_start_returns_zero(self) -> None:
        index = pd.to_datetime(["2020-01-31", "2020-02-03", "2020-02-28"])
        monthly = monthly_returns(pd.Series([100.0, 101.0, 103.0], index=index))
        self.assertAlmostEqual(monthly.iloc[0], 0.0)
        self.assertAlmostEqual(monthly.iloc[1], 0.03)

    def test_monthly_table_has_twelve_columns_and_blank_months(self) -> None:
        index = pd.to_datetime(["2019-11-29", "2019-12-31", "2020-01-31", "2020-02-28"])
        table = monthly_table(monthly_returns(pd.Series([100.0, 102.0, 99.96, 103.9584], index=index)))
        self.assertEqual(list(table.columns), list(range(1, 13)))
        self.assertEqual(list(table.index), [2019, 2020])
        self.assertAlmostEqual(table.loc[2019, 12], 0.02)
        self.assertAlmostEqual(table.loc[2020, 1], -0.02)
        self.assertAlmostEqual(table.loc[2020, 2], 0.04)
        self.assertTrue(pd.isna(table.loc[2019, 1]))
        self.assertTrue(pd.isna(table.loc[2020, 3]))

    def test_monthly_returns_need_two_prices(self) -> None:
        with self.assertRaises(AnalysisError):
            monthly_returns(pd.Series([100.0], index=pd.to_datetime(["2020-01-02"])))

    def test_candles_are_scaled_onto_the_adjusted_close(self) -> None:
        frame = pd.DataFrame(
            {
                "open": [8.0, 10.0],
                "high": [12.0, 11.0],
                "low": [6.0, 9.0],
                "close": [10.0, 10.0],
                "volume": [500.0, 700.0],
                "adj_close": [5.0, 10.0],
            },
            index=pd.to_datetime(["2020-01-02", "2020-01-03"]),
        )
        candles = adjusted_ohlc(frame)
        # Day one: adjusted close is half the raw close, so every price halves.
        self.assertEqual(list(candles.iloc[0][["open", "high", "low", "close"]]), [4.0, 6.0, 3.0, 5.0])
        # Day two: factor one, nothing moves. Volume is never scaled.
        self.assertEqual(list(candles.iloc[1][["open", "high", "low", "close"]]), [10.0, 11.0, 9.0, 10.0])
        self.assertEqual(list(candles["volume"]), [500.0, 700.0])


class RebaseTests(unittest.TestCase):
    def test_shared_sessions_and_rebase(self) -> None:
        one = pd.Series([10.0, 11.0, 12.0], index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]))
        two = pd.Series([50.0, 55.0, 44.0], index=pd.to_datetime(["2020-01-03", "2020-01-06", "2020-01-07"]))
        shared = shared_sessions({"ONE": one, "TWO": two})
        self.assertEqual([stamp.date().isoformat() for stamp in shared.index], ["2020-01-03", "2020-01-06"])
        lines = rebase(shared)
        self.assertEqual(list(lines.iloc[0]), [100.0, 100.0])
        self.assertAlmostEqual(lines["ONE"].iloc[1], 12.0 / 11.0 * 100.0)
        self.assertAlmostEqual(lines["TWO"].iloc[1], 110.0)
        with self.assertRaises(AnalysisError):
            rebase(shared.iloc[0:0])


class PeriodIntegrationTests(unittest.TestCase):
    """build_period on a stock that moves exactly twice the benchmark each day."""

    @classmethod
    def setUpClass(cls) -> None:
        dates = pd.bdate_range("2020-01-02", periods=41)
        # The benchmark alternates +1% and −0.5%, so it rises on 20 days
        # and falls on 20. The stock's daily return is exactly double.
        moves = [0.01 if step % 2 == 0 else -0.005 for step in range(40)]
        bench_prices = [100.0]
        stock_prices = [50.0]
        for move in moves:
            bench_prices.append(bench_prices[-1] * (1.0 + move))
            stock_prices.append(stock_prices[-1] * (1.0 + 2.0 * move))
        cls.moves = moves
        cls.stock = _frame(stock_prices, dates)
        cls.benchmark = _frame(bench_prices, dates)
        cls.report = build_period(
            cls.stock,
            ticker="DOUBLE",
            start=dates[0],
            end=dates[-1],
            benchmark=cls.benchmark,
            benchmark_ticker="BENCH",
            ma_short=3,
            ma_long=5,
            risk_free=0.0,
        )

    def test_beta_correlation_and_capture(self) -> None:
        stats = self.report.stats
        self.assertEqual(stats.return_pairs, 40)
        self.assertAlmostEqual(stats.beta, 2.0, places=9)
        self.assertAlmostEqual(stats.correlation, 1.0, places=9)
        self.assertEqual(stats.beta_label, "high")
        self.assertEqual(stats.correlation_label, "strong")
        self.assertEqual((stats.up_days, stats.down_days), (20, 20))
        self.assertAlmostEqual(stats.up_capture, 2.0, places=9)
        self.assertAlmostEqual(stats.down_capture, 2.0, places=9)

    def test_ratios_match_the_hand_formula(self) -> None:
        stats = self.report.stats
        stock_moves = [2.0 * move for move in self.moves]
        mean = sum(stock_moves) / 40
        variance = sum((value - mean) ** 2 for value in stock_moves) / 39
        self.assertAlmostEqual(stats.sharpe, mean / math.sqrt(variance) * math.sqrt(252), places=9)
        downside = math.sqrt(sum(min(value, 0.0) ** 2 for value in stock_moves) / 40)
        self.assertAlmostEqual(stats.sortino, mean / downside * math.sqrt(252), places=9)
        self.assertAlmostEqual(stats.best_day_return, 0.02, places=12)
        self.assertAlmostEqual(stats.worst_day_return, -0.01, places=12)
        self.assertAlmostEqual(stats.positive_day_share, 0.5)

    def test_custom_average_windows_are_used_and_named(self) -> None:
        stats = self.report.stats
        closes = list(self.stock["adj_close"])
        self.assertEqual((stats.ma_short_window, stats.ma_long_window), (3, 5))
        self.assertAlmostEqual(stats.ma_short, sum(closes[-3:]) / 3, places=9)
        self.assertAlmostEqual(stats.ma_long, sum(closes[-5:]) / 5, places=9)
        text = " ".join(self.report.sentences)
        self.assertIn("3-day average", text)
        self.assertIn("5-day average", text)

    def test_sentences_quote_the_new_figures_and_their_rules(self) -> None:
        text = " ".join(self.report.sentences)
        self.assertIn("beta is 2.00", text)
        self.assertIn("high: 1.20 or above", text)
        self.assertIn("correlation of daily returns is 1.00", text)
        self.assertIn("strong: size 0.70 or above", text)
        self.assertIn("200% of BENCH's (up capture)", text)
        self.assertIn(f"Sharpe ratio is {self.report.stats.sharpe:.2f}", text)
        self.assertIn("best day was +2.0%", text)
        # 41 sessions is under a year, so the annualized figure is withheld and said so.
        self.assertIsNone(self.report.stats.annualized_return)
        self.assertIn("No annualized return is shown", text)

    def test_chart_series_cover_the_window(self) -> None:
        charts = self.report.charts
        self.assertEqual(len(charts.returns), 40)
        self.assertEqual(len(charts.benchmark_returns), 40)
        self.assertEqual(list(charts.ohlc.columns), ["open", "high", "low", "close", "volume"])
        self.assertEqual(len(charts.ohlc), 41)
        compounded = (1.0 + charts.monthly).prod() - 1.0
        self.assertAlmostEqual(compounded, self.report.stats.cumulative_return, places=12)
        # 40 paired returns is fewer than the 63 the rolling lines need.
        self.assertIsNone(charts.rolling_beta)

    def test_a_week_long_window_omits_what_it_cannot_support(self) -> None:
        dates = self.stock.index
        report = build_period(
            self.stock,
            ticker="DOUBLE",
            start=dates[0],
            end=dates[5],
            benchmark=self.benchmark,
            benchmark_ticker="BENCH",
        )
        stats = report.stats
        self.assertLess(stats.n_returns, MIN_RETURNS_FOR_RATIOS)
        self.assertIsNone(stats.sharpe)
        self.assertIsNone(stats.sortino)
        self.assertIsNone(stats.beta)
        self.assertIsNone(stats.up_capture)
        text = " ".join(report.sentences)
        self.assertIn("Sharpe and Sortino ratios are omitted", text)
        self.assertIn("Beta and correlation against BENCH are omitted", text)
        self.assertIn("Up and down capture are omitted", text)

    def test_without_a_benchmark_the_relative_figures_are_blank(self) -> None:
        dates = self.stock.index
        report = build_period(self.stock, ticker="DOUBLE", start=dates[0], end=dates[-1])
        self.assertIsNone(report.stats.benchmark)
        self.assertIsNone(report.stats.beta)
        self.assertIsNone(report.stats.excess_return)
        self.assertIsNotNone(report.stats.sharpe)
        self.assertNotIn("beta", " ".join(report.sentences).lower())

    def test_bad_settings_are_rejected(self) -> None:
        dates = self.stock.index
        with self.assertRaises(AnalysisError):
            build_period(self.stock, ticker="DOUBLE", start=dates[0], end=dates[-1], ma_short=10, ma_long=5)
        with self.assertRaises(AnalysisError):
            build_period(self.stock, ticker="DOUBLE", start=dates[0], end=dates[-1], risk_free=2.0)


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dates = pd.bdate_range("2020-01-02", periods=30)
        bench = [100.0 * (1.0 + 0.001 * step) for step in range(30)]
        winner = [20.0 * (1.0 + 0.01 * step) for step in range(30)]
        loser = [80.0 * (1.0 - 0.005 * step) for step in range(30)]
        cls.dates = dates
        cls.benchmark = _frame(bench, dates)
        cls.winner = _frame(winner, dates)
        # The loser only starts trading on the eleventh session.
        cls.loser = _frame(loser[10:], dates[10:])
        cls.reports = {
            name: build_period(
                frame,
                ticker=name,
                start=dates[0],
                end=dates[-1],
                benchmark=cls.benchmark,
                benchmark_ticker="BENCH",
            )
            for name, frame in (("WIN", cls.winner), ("LOSE", cls.loser))
        }
        cls.bench_report = build_period(cls.benchmark, ticker="BENCH", start=dates[0], end=dates[-1])
        cls.comparison = build_comparison(cls.reports, cls.bench_report)

    def test_rows_keep_selection_order_and_end_with_the_benchmark(self) -> None:
        self.assertEqual([row.ticker for row in self.comparison.rows], ["WIN", "LOSE", "BENCH"])
        self.assertEqual(self.comparison.benchmark, "BENCH")

    def test_every_line_starts_at_100_on_the_first_shared_session(self) -> None:
        lines = self.comparison.rebased
        self.assertEqual(list(lines.columns), ["WIN", "LOSE", "BENCH"])
        # The first shared session is the loser's first day, not the range start.
        self.assertEqual(lines.index[0], self.dates[10])
        self.assertEqual(list(lines.iloc[0]), [100.0, 100.0, 100.0])
        self.assertAlmostEqual(lines["WIN"].iloc[-1], (1.0 + 0.29) / (1.0 + 0.10) * 100.0, places=9)
        self.assertEqual(len(lines), 20)

    def test_table_holds_the_same_numbers_as_the_reports(self) -> None:
        table = comparison_table(self.comparison)
        self.assertEqual(list(table.index), ["WIN", "LOSE", "BENCH (benchmark)"])
        self.assertAlmostEqual(table.loc["WIN", "Return"], 0.29, places=12)
        self.assertAlmostEqual(table.loc["LOSE", "Return"], (1.0 - 0.145) / (1.0 - 0.05) - 1.0, places=12)
        self.assertEqual(table.loc["WIN", "Sessions"], 30)
        self.assertEqual(table.loc["LOSE", "Sessions"], 20)
        self.assertAlmostEqual(
            table.loc["WIN", "Vs benchmark (pp)"],
            (0.29 - 0.029) * 100.0,
            places=9,
        )
        # The benchmark is not measured against itself.
        self.assertTrue(pd.isna(table.loc["BENCH (benchmark)", "Beta"]))
        self.assertTrue(pd.isna(table.loc["BENCH (benchmark)", "Vs benchmark (pp)"]))

    def test_cells_print_through_one_format_table(self) -> None:
        self.assertEqual(format_cell("Return", 0.2912), "+29.1%")
        self.assertEqual(format_cell("Max drawdown", -0.304), "-30.4%")
        self.assertEqual(format_cell("Sharpe", 1.2345), "1.23")
        self.assertEqual(format_cell("Up capture", 1.18), "118%")
        self.assertEqual(format_cell("Vs benchmark (pp)", -6.34), "-6.3")
        self.assertEqual(format_cell("Beta", float("nan")), MISSING)
        self.assertEqual(format_cell("Beta", None), MISSING)

    def test_sentences_rank_the_symbols_and_flag_the_late_start(self) -> None:
        text = " ".join(self.comparison.sentences)
        self.assertIn("Ranked by cumulative return: WIN +29.0%, LOSE -10.0%", text)
        self.assertIn("BENCH, the benchmark, returned +2.9%", text)
        self.assertIn("1 of the 2 symbols finished ahead of it", text)
        self.assertIn("The deepest drawdown is LOSE's", text)
        self.assertIn(f"starting later: LOSE on {self.dates[10].date().isoformat()}, so that row covers", text)
        self.assertIn(f"The rebased chart starts on {self.dates[10].date().isoformat()}", text)

    def test_one_symbol_has_no_ranking_sentences(self) -> None:
        single = build_comparison({"WIN": self.reports["WIN"]}, self.bench_report)
        self.assertEqual(single.sentences, ())
        self.assertEqual([row.ticker for row in single.rows], ["WIN", "BENCH"])
        with self.assertRaises(AnalysisError):
            build_comparison({})

    def test_a_missing_benchmark_still_compares_the_symbols(self) -> None:
        alone = build_comparison(self.reports, None)
        self.assertIsNone(alone.benchmark)
        self.assertEqual(list(alone.rebased.columns), ["WIN", "LOSE"])
        self.assertNotIn("the benchmark", " ".join(alone.sentences))


def _frame(prices: list[float], index) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": [1_000.0] * len(prices),
            "adj_close": prices,
        },
        index=pd.DatetimeIndex(index),
    )


if __name__ == "__main__":
    unittest.main()
