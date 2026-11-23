"""Hand-checked tests for Phase 2 risk, diversification, and cost-aware backtests."""

from __future__ import annotations

import math
import unittest
from statistics import NormalDist

import numpy as np
import pandas as pd

from features import (
    REBALANCE_MONTHLY,
    REBALANCE_NONE,
    conditional_var,
    correlation_matrix,
    diversification_stats,
    historical_var,
    jensen_alpha,
    max_drawdown,
    parametric_var,
    portfolio_backtest,
    portfolio_value,
    rolling_sharpe,
)


class DrawdownRecoveryTests(unittest.TestCase):
    def test_peak_trough_and_recovery_timing(self) -> None:
        index = pd.to_datetime(
            ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09"]
        )
        # Peak 120 on day 2, trough 90 on day 4 (−25%), recovers to 120 on day 6.
        prices = pd.Series([100.0, 120.0, 110.0, 90.0, 100.0, 120.0], index=index)
        result = max_drawdown(prices)
        self.assertAlmostEqual(result["max_drawdown"], -0.25)
        self.assertEqual(result["days_to_trough"], 2)
        self.assertTrue(result["recovered"])
        self.assertEqual(result["days_to_recovery"], 2)
        self.assertGreaterEqual(result["longest_underwater_days"], 2)

    def test_unrecovered_drawdown(self) -> None:
        index = pd.bdate_range("2020-01-02", periods=5)
        prices = pd.Series([100.0, 110.0, 80.0, 85.0, 90.0], index=index)
        result = max_drawdown(prices)
        self.assertFalse(result["recovered"])
        self.assertIsNone(result["days_to_recovery"])


class VarFamilyTests(unittest.TestCase):
    def test_parametric_var_matches_normal_formula(self) -> None:
        returns = pd.Series([0.01, -0.02, 0.015, 0.0] + [0.001] * 16)
        mean = float(returns.mean())
        std = float(returns.std(ddof=1))
        z = NormalDist().inv_cdf(0.05)
        expected = max(-(mean + z * std), 0.0)
        self.assertAlmostEqual(parametric_var(returns), expected, places=12)

    def test_cvar_is_at_least_historical_var(self) -> None:
        rng = np.random.default_rng(1)
        returns = pd.Series(rng.normal(0.0, 0.01, 100))
        h = historical_var(returns)
        c = conditional_var(returns)
        self.assertGreaterEqual(c + 1e-12, h)

    def test_jensen_alpha_zero_when_stock_is_beta_times_benchmark(self) -> None:
        # r_s = 1.5 * r_b every day, rf = 0 → alpha = 0, beta = 1.5.
        bench = pd.Series([0.01, -0.02, 0.015, 0.0, 0.005] * 8)
        stock = 1.5 * bench
        alpha = jensen_alpha(stock, bench, risk_free=0.0)
        self.assertAlmostEqual(alpha, 0.0, places=10)


class DiversificationTests(unittest.TestCase):
    def test_correlation_matrix_and_effective_n(self) -> None:
        rng = np.random.default_rng(7)
        idx = pd.bdate_range("2020-01-02", periods=40)
        # Shared factor plus idiosyncratic noise so portfolio vol stays positive.
        factor = rng.normal(0.0, 0.01, 40)
        a = pd.Series(factor + rng.normal(0.0, 0.008, 40), index=idx)
        b = pd.Series(-0.4 * factor + rng.normal(0.0, 0.008, 40), index=idx)
        frame = pd.DataFrame({"A": a, "B": b})
        corr = correlation_matrix(frame)
        self.assertAlmostEqual(float(corr.loc["A", "A"]), 1.0)
        self.assertLess(float(corr.loc["A", "B"]), 0.0)
        stats = diversification_stats(frame, {"A": 0.5, "B": 0.5})
        self.assertAlmostEqual(stats.effective_n, 2.0)
        self.assertAlmostEqual(sum(stats.risk_contribution.values()), 1.0, places=8)
        self.assertGreater(stats.diversification_ratio, 1.0)


class CostAwareBacktestTests(unittest.TestCase):
    def test_zero_cost_matches_legacy_path(self) -> None:
        index = pd.to_datetime(
            ["2020-01-02", "2020-01-15", "2020-01-31", "2020-02-03", "2020-02-14", "2020-02-28"]
        )
        frame = pd.DataFrame(
            {"A": [100.0, 110.0, 120.0, 120.0, 120.0, 120.0], "B": [100.0, 100.0, 100.0, 110.0, 120.0, 130.0]},
            index=index,
        )
        weights = {"A": 0.5, "B": 0.5}
        legacy = portfolio_value(frame, weights, rebalance=REBALANCE_MONTHLY, cost_bps=0.0)
        valued, report = portfolio_backtest(frame, weights, rebalance=REBALANCE_MONTHLY, cost_bps=0.0)
        self.assertAlmostEqual(float(legacy.iloc[-1]), float(valued.iloc[-1]))
        self.assertAlmostEqual(float(valued.iloc[-1]), 0.575 + 0.575 * 130.0 / 110.0)
        self.assertEqual(report.cost_bps, 0.0)

    def test_positive_cost_reduces_terminal_value(self) -> None:
        # Buy-and-hold: only inception turnover differs, so terminal value
        # scales by (1 − cost) while holdings stay proportional.
        index = pd.bdate_range("2020-01-02", periods=40)
        frame = pd.DataFrame(
            {
                "A": np.linspace(100, 130, 40),
                "B": np.linspace(100, 110, 40),
            },
            index=index,
        )
        weights = {"A": 0.5, "B": 0.5}
        free, free_report = portfolio_backtest(
            frame, weights, rebalance=REBALANCE_NONE, cost_bps=0.0
        )
        costly, report = portfolio_backtest(
            frame, weights, rebalance=REBALANCE_NONE, cost_bps=50.0
        )
        self.assertAlmostEqual(float(free.iloc[0]), 1.0)
        self.assertAlmostEqual(float(costly.iloc[0]), 0.995)
        self.assertAlmostEqual(float(costly.iloc[-1]), float(free.iloc[-1]) * 0.995, places=10)
        self.assertAlmostEqual(report.total_turnover, 1.0)
        self.assertEqual(report.rebalance_count, 1)
        self.assertEqual(free_report.rebalance_count, 1)

    def test_monthly_rebalance_reports_turnover(self) -> None:
        index = pd.bdate_range("2020-01-02", periods=60)
        frame = pd.DataFrame(
            {
                "A": np.linspace(100, 130, 60),
                "B": np.linspace(100, 90, 60),
            },
            index=index,
        )
        _, report = portfolio_backtest(
            frame, {"A": 0.5, "B": 0.5}, rebalance=REBALANCE_MONTHLY, cost_bps=10.0
        )
        self.assertGreater(report.total_turnover, 1.0)
        self.assertGreater(report.rebalance_count, 1)
        self.assertGreater(report.total_cost_drag, 0.0)


class RollingSharpeTests(unittest.TestCase):
    def test_rolling_sharpe_warmup(self) -> None:
        # Alternating returns so the window standard deviation is positive.
        returns = pd.Series([0.01, -0.005] * 15)
        series = rolling_sharpe(returns, risk_free=0.0, window=20)
        self.assertTrue(series.iloc[:19].isna().all())
        self.assertFalse(math.isnan(float(series.iloc[19])))


class LookAheadGuardTests(unittest.TestCase):
    def test_rebalance_day_return_uses_prior_weights_only(self) -> None:
        # A jumps 10% on the first February session; B is flat. Prior equal
        # holdings are still 50/50 in shares from January, so the day return
        # must be +5% (not +10% from post-rebalance 100% A, and not 0%).
        index = pd.to_datetime(
            ["2020-01-02", "2020-01-31", "2020-02-03", "2020-02-04"]
        )
        frame = pd.DataFrame(
            {
                "A": [100.0, 100.0, 110.0, 110.0],
                "B": [100.0, 100.0, 100.0, 100.0],
            },
            index=index,
        )
        value = portfolio_value(frame, {"A": 0.5, "B": 0.5}, rebalance=REBALANCE_MONTHLY)
        self.assertAlmostEqual(float(value.iloc[1]), 1.0)
        # Mark with prior weights: 0.5*1.10 + 0.5*1.00 = 1.05, then reset.
        self.assertAlmostEqual(float(value.iloc[2]), 1.05)
        day_return = float(value.iloc[2] / value.iloc[1] - 1.0)
        self.assertAlmostEqual(day_return, 0.05)
        # A look-ahead bug that applied February weights before the jump
        # would earn ~10% (all A) or 0% (if it skipped the day). Either fails.
        self.assertNotAlmostEqual(day_return, 0.10)
        self.assertNotAlmostEqual(day_return, 0.0)

    def test_shared_sessions_drop_missing_dates_without_ffill(self) -> None:
        from features import shared_sessions

        a = pd.Series(
            [100.0, 110.0, 120.0],
            index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
        )
        b = pd.Series(
            [50.0, 55.0],
            index=pd.to_datetime(["2020-01-02", "2020-01-06"]),
        )
        shared = shared_sessions({"A": a, "B": b})
        self.assertEqual(list(shared.index), list(pd.to_datetime(["2020-01-02", "2020-01-06"])))
        # No invented mid-point for B on 2020-01-03.
        self.assertNotIn(pd.Timestamp("2020-01-03"), shared.index)


if __name__ == "__main__":
    unittest.main()
