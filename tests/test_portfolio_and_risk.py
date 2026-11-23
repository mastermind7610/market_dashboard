"""Portfolio weights, historical VaR, calendar-year returns, snapshots, and CSV export."""

from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from dashboard import run_briefing
from data_loader import CompanySnapshot, MarketDataError, fetch_company_snapshot, search_symbols
from features import (
    MIN_RETURNS_FOR_VAR,
    PORTFOLIO_LABEL,
    REBALANCE_MONTHLY,
    REBALANCE_NONE,
    REBALANCE_QUARTERLY,
    AnalysisError,
    aligned_export,
    build_period,
    calendar_year_returns,
    comparison_table,
    describe_calendar_years,
    historical_var,
    normalize_weights,
    portfolio_frame,
    portfolio_value,
    validate_rebalance,
)


def _prices(values: list[float], start: str = "2020-01-02") -> pd.Series:
    index = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=index, dtype=float)


def _ohlc(values: list[float], start: str = "2020-01-02") -> pd.DataFrame:
    series = _prices(values, start)
    return pd.DataFrame(
        {
            "open": series,
            "high": series,
            "low": series,
            "close": series,
            "volume": 1_000.0,
            "adj_close": series,
        },
        index=series.index,
    )


class WeightNormalizationTests(unittest.TestCase):
    def test_equal_weight_when_none_given(self) -> None:
        weights, notes = normalize_weights(None, ["AAPL", "MSFT"])
        self.assertEqual(weights, {"AAPL": 0.5, "MSFT": 0.5})
        self.assertTrue(any("Equal weight" in note for note in notes))

    def test_weights_not_summing_to_one_are_scaled(self) -> None:
        weights, notes = normalize_weights({"AAPL": 0.3, "MSFT": 0.3}, ["AAPL", "MSFT"])
        self.assertAlmostEqual(weights["AAPL"], 0.5)
        self.assertAlmostEqual(weights["MSFT"], 0.5)
        self.assertTrue(any("scaled to total 100%" in note for note in notes))

    def test_weights_slightly_off_one_are_nudged(self) -> None:
        weights, notes = normalize_weights({"AAPL": 0.501, "MSFT": 0.501}, ["AAPL", "MSFT"])
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=12)
        self.assertTrue(any("nudged" in note or "scaled" in note for note in notes))

    def test_zero_weight_is_dropped(self) -> None:
        weights, notes = normalize_weights(
            {"AAPL": 0.6, "MSFT": 0.0, "GOOG": 0.4},
            ["AAPL", "MSFT", "GOOG"],
        )
        self.assertEqual(set(weights), {"AAPL", "GOOG"})
        self.assertAlmostEqual(weights["AAPL"], 0.6)
        self.assertAlmostEqual(weights["GOOG"], 0.4)
        self.assertTrue(any("MSFT" in note and "zero weight" in note for note in notes))

    def test_all_zero_weights_are_rejected(self) -> None:
        with self.assertRaises(AnalysisError) as caught:
            normalize_weights({"AAPL": 0.0, "MSFT": 0.0}, ["AAPL", "MSFT"])
        self.assertIn("zero weight", str(caught.exception))

    def test_negative_weight_is_rejected(self) -> None:
        with self.assertRaises(AnalysisError) as caught:
            normalize_weights({"AAPL": 1.2, "MSFT": -0.2}, ["AAPL", "MSFT"])
        self.assertIn("negative", str(caught.exception))

    def test_empty_symbol_list_is_rejected(self) -> None:
        with self.assertRaises(AnalysisError):
            normalize_weights(None, [])


class PortfolioValueTests(unittest.TestCase):
    def test_buy_and_hold_tracks_weighted_growth(self) -> None:
        # A doubles, B stays flat. Equal weight buy-and-hold ends at 1.5.
        frame = pd.DataFrame(
            {
                "A": [100.0, 120.0, 200.0],
                "B": [50.0, 50.0, 50.0],
            },
            index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
        )
        value = portfolio_value(frame, {"A": 0.5, "B": 0.5}, rebalance=REBALANCE_NONE)
        self.assertAlmostEqual(float(value.iloc[0]), 1.0)
        self.assertAlmostEqual(float(value.iloc[-1]), 1.5)

    def test_monthly_rebalance_resets_on_month_change(self) -> None:
        # Two months: A rises in January, B rises in February.
        # Equal-weight buy-and-hold keeps the January overweight in A.
        # Monthly rebalance resets to 50/50 on the first February session.
        index = pd.to_datetime(
            ["2020-01-02", "2020-01-15", "2020-01-31", "2020-02-03", "2020-02-14", "2020-02-28"]
        )
        frame = pd.DataFrame(
            {
                "A": [100.0, 110.0, 120.0, 120.0, 120.0, 120.0],
                "B": [100.0, 100.0, 100.0, 110.0, 120.0, 130.0],
            },
            index=index,
        )
        weights = {"A": 0.5, "B": 0.5}
        held = portfolio_value(frame, weights, rebalance=REBALANCE_NONE)
        monthly = portfolio_value(frame, weights, rebalance=REBALANCE_MONTHLY)
        self.assertAlmostEqual(float(held.iloc[0]), 1.0)
        self.assertAlmostEqual(float(monthly.iloc[0]), 1.0)
        # Hand path for monthly: end-Jan value 1.10; rebalance at Feb 3 into
        # 0.55/120 of A and 0.55/110 of B; end-Feb = 0.55 + 0.65 = 1.20.
        self.assertAlmostEqual(float(monthly.iloc[2]), 1.10)
        self.assertAlmostEqual(float(monthly.iloc[-1]), 1.20)
        # Buy-and-hold keeps 0.005 shares of each: 0.6 + 0.65 = 1.25.
        self.assertAlmostEqual(float(held.iloc[-1]), 1.25)
        self.assertNotAlmostEqual(float(monthly.iloc[-1]), float(held.iloc[-1]))

    def test_quarterly_rebalance_is_accepted(self) -> None:
        frame = pd.DataFrame(
            {"A": [10.0, 11.0, 12.0], "B": [20.0, 19.0, 21.0]},
            index=pd.to_datetime(["2020-01-02", "2020-04-01", "2020-04-02"]),
        )
        value = portfolio_value(frame, {"A": 0.5, "B": 0.5}, rebalance=REBALANCE_QUARTERLY)
        self.assertEqual(value.name, PORTFOLIO_LABEL)
        self.assertAlmostEqual(float(value.iloc[0]), 1.0)

    def test_bad_rebalance_is_rejected(self) -> None:
        with self.assertRaises(AnalysisError):
            validate_rebalance("weekly")

    def test_portfolio_frame_feeds_build_period(self) -> None:
        value = _prices([1.0, 1.1, 1.21])
        report = build_period(
            portfolio_frame(value),
            ticker=PORTFOLIO_LABEL,
            start=value.index[0],
            end=value.index[-1],
        )
        self.assertAlmostEqual(report.stats.cumulative_return, 0.21)


class HistoricalVarTests(unittest.TestCase):
    def test_var_matches_hand_percentile(self) -> None:
        # Twenty returns: the 5th percentile of a sorted list of 20 is
        # between the lowest and second-lowest under numpy's default
        # linear quantile. Build a sequence whose 5% quantile is exact.
        returns = pd.Series([-0.05] + [-0.01] * 18 + [0.02])
        expected = -float(np.quantile(returns.to_numpy(), 0.05))
        self.assertAlmostEqual(historical_var(returns), expected, places=12)
        self.assertGreater(historical_var(returns), 0.0)

    def test_all_gains_report_zero_var(self) -> None:
        returns = pd.Series([0.01] * 25)
        self.assertAlmostEqual(historical_var(returns), 0.0)

    def test_too_few_returns_are_rejected(self) -> None:
        with self.assertRaises(AnalysisError) as caught:
            historical_var(pd.Series([0.01] * (MIN_RETURNS_FOR_VAR - 1)))
        self.assertIn(str(MIN_RETURNS_FOR_VAR), str(caught.exception))

    def test_build_period_exposes_var(self) -> None:
        # A long enough window so VaR has enough daily returns.
        values = [100.0 * (1.01 if step % 2 == 0 else 0.99) for step in range(40)]
        report = build_period(
            _ohlc(values),
            ticker="DEMO",
            start="2020-01-02",
            end="2020-12-31",
        )
        self.assertIsNotNone(report.stats.value_at_risk_95)
        self.assertGreaterEqual(report.stats.value_at_risk_95, 0.0)
        self.assertIn("Value at Risk", " ".join(report.sentences))


class CalendarYearTests(unittest.TestCase):
    def test_partial_years_use_first_and_last_close(self) -> None:
        # 2020: 100 -> 110. 2021: 110 -> 121.
        index = pd.to_datetime(["2020-06-01", "2020-12-31", "2021-06-01", "2021-12-31"])
        prices = pd.Series([100.0, 110.0, 110.0, 121.0], index=index)
        years = calendar_year_returns(prices)
        self.assertAlmostEqual(float(years.loc[2020]), 0.10)
        self.assertAlmostEqual(float(years.loc[2021]), 0.10)

    def test_single_print_year_is_skipped(self) -> None:
        index = pd.to_datetime(["2020-01-02", "2020-06-01", "2021-06-01"])
        prices = pd.Series([100.0, 110.0, 120.0], index=index)
        years = calendar_year_returns(prices)
        self.assertIn(2020, years.index)
        self.assertNotIn(2021, years.index)

    def test_describe_calendar_years_names_best_and_worst(self) -> None:
        years = pd.Series({2020: 0.10, 2021: -0.20, 2022: 0.05})
        sentence = describe_calendar_years(years, ticker="DEMO")
        self.assertIn("2020", sentence)
        self.assertIn("2021", sentence)
        self.assertIn("DEMO", sentence)

    def test_too_few_prices_are_rejected(self) -> None:
        with self.assertRaises(AnalysisError):
            calendar_year_returns(pd.Series([100.0], index=pd.to_datetime(["2020-01-02"])))


class SnapshotTests(unittest.TestCase):
    def test_offline_placeholder_is_labelled(self) -> None:
        snap = fetch_company_snapshot("AAPL", offline=True)
        self.assertEqual(snap.name, "Apple Inc.")
        self.assertEqual(snap.sector, "Technology")
        self.assertTrue(snap.offline)
        self.assertIn("Offline demo placeholder", snap.source)
        self.assertIsNotNone(snap.market_cap)
        self.assertIsNotNone(snap.pe_ratio)
        self.assertIsNotNone(snap.dividend_yield)

    def test_offline_unknown_symbol_has_missing_fields(self) -> None:
        snap = fetch_company_snapshot("MSFT", offline=True)
        self.assertEqual(snap.ticker, "MSFT")
        # Known offline symbols have fields; invent one outside the map via patch.
        with patch.dict("data_loader.OFFLINE_SNAPSHOTS", {}, clear=True):
            empty = fetch_company_snapshot("MSFT", offline=True)
        self.assertIsNone(empty.name)
        self.assertIsNone(empty.market_cap)
        self.assertIsNone(empty.pe_ratio)
        self.assertIn("no fundamentals bundled", empty.source.lower())

    def test_live_snapshot_leaves_missing_fields_blank(self) -> None:
        sparse = {
            "shortName": "Sparse Co",
            "symbol": "SPARSE",
            "trailingPegRatio": 1.0,
        }
        with patch("yfinance.Ticker") as ticker:
            ticker.return_value.info = sparse
            snap = fetch_company_snapshot("SPARSE", offline=False)
        self.assertEqual(snap.name, "Sparse Co")
        self.assertIsNone(snap.sector)
        self.assertIsNone(snap.market_cap)
        self.assertIsNone(snap.pe_ratio)
        self.assertIsNone(snap.dividend_yield)
        self.assertIsNone(snap.fifty_two_week_low)
        self.assertFalse(snap.offline)

    def test_dividend_yield_above_25_percent_is_scaled(self) -> None:
        info = {
            "shortName": "Yield Co",
            "symbol": "YLD",
            "trailingPegRatio": 1.0,
            "dividendYield": 0.40,  # Yahoo sometimes means 0.40%, not 40%.
        }
        with patch("yfinance.Ticker") as ticker:
            ticker.return_value.info = info
            snap = fetch_company_snapshot("YLD", offline=False)
        self.assertAlmostEqual(snap.dividend_yield, 0.004)

    def test_empty_info_is_an_error(self) -> None:
        with patch("yfinance.Ticker") as ticker:
            ticker.return_value.info = {}
            with self.assertRaises(MarketDataError):
                fetch_company_snapshot("ZZZ", offline=False)


class ExportAndBriefingTests(unittest.TestCase):
    def test_offline_portfolio_briefing_includes_row_and_csv(self) -> None:
        briefing = run_briefing(
            tickers=["AAPL", "MSFT"],
            benchmark="SPY",
            offline=True,
            preset="Max",
            portfolio=True,
            weights={"AAPL": 0.5, "MSFT": 0.5},
            rebalance=REBALANCE_MONTHLY,
        )
        self.assertIsNotNone(briefing.portfolio_report)
        self.assertEqual(briefing.portfolio_weights, {"AAPL": 0.5, "MSFT": 0.5})
        self.assertEqual(briefing.rebalance, REBALANCE_MONTHLY)
        table = comparison_table(briefing.comparison)
        self.assertIn(PORTFOLIO_LABEL, table.index)
        self.assertIn("SPY (benchmark)", list(table.index))
        self.assertIn(PORTFOLIO_LABEL, briefing.comparison.rebased.columns)
        self.assertIsNotNone(briefing.export_csv)
        self.assertIn("AAPL_price", briefing.export_csv)
        self.assertIn("Portfolio_price", briefing.export_csv)
        self.assertIn("SPY_price", briefing.export_csv)
        self.assertIn("Portfolio", briefing.markdown)
        self.assertIsNotNone(briefing.snapshot)
        self.assertEqual(briefing.snapshot.ticker, "AAPL")
        self.assertTrue(briefing.snapshot.offline)
        self.assertIsNotNone(briefing.report.stats.value_at_risk_95)
        self.assertIsNotNone(briefing.report.charts.calendar_years)
        self.assertGreater(len(briefing.report.charts.calendar_years), 0)

    def test_uneven_weights_are_renormalized_in_briefing(self) -> None:
        briefing = run_briefing(
            tickers=["AAPL", "MSFT"],
            offline=True,
            preset="Max",
            portfolio=True,
            weights={"AAPL": 0.2, "MSFT": 0.2},
        )
        self.assertAlmostEqual(briefing.portfolio_weights["AAPL"], 0.5)
        self.assertAlmostEqual(briefing.portfolio_weights["MSFT"], 0.5)
        self.assertTrue(any("scaled to total 100%" in note for note in briefing.notes))

    def test_zero_weight_symbol_is_left_out(self) -> None:
        briefing = run_briefing(
            tickers=["AAPL", "MSFT"],
            offline=True,
            preset="Max",
            portfolio=True,
            weights={"AAPL": 1.0, "MSFT": 0.0},
        )
        self.assertEqual(briefing.portfolio_weights, {"AAPL": 1.0})
        self.assertTrue(any("MSFT" in note and "zero weight" in note for note in briefing.notes))

    def test_aligned_export_includes_returns(self) -> None:
        briefing = run_briefing(tickers=["AAPL", "MSFT"], offline=True, preset="Max", portfolio=True)
        frame = aligned_export(
            briefing.reports,
            benchmark_report=briefing.benchmark_report,
            portfolio_report=briefing.portfolio_report,
        )
        self.assertIn("AAPL_return", frame.columns)
        self.assertTrue(math.isnan(frame["AAPL_return"].iloc[0]))
        self.assertFalse(math.isnan(frame["AAPL_return"].iloc[1]))

    def test_portfolio_mode_with_one_symbol_adds_a_note(self) -> None:
        briefing = run_briefing(
            tickers=["AAPL"],
            offline=True,
            preset="Max",
            portfolio=True,
        )
        self.assertIsNone(briefing.portfolio_report)
        self.assertTrue(any("at least two" in note.lower() for note in briefing.notes))


class CompanySearchTests(unittest.TestCase):
    def test_apple_suggests_aapl(self) -> None:
        try:
            hits = search_symbols("apple")
        except MarketDataError as exc:
            self.skipTest(f"Yahoo Finance unreachable: {exc}")
        self.assertTrue(hits)
        self.assertEqual(hits[0].symbol, "AAPL")
        self.assertIn("Apple", hits[0].name)


if __name__ == "__main__":
    unittest.main()
