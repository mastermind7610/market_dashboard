"""
Phase 6 production-readiness guards: timeouts, stale refresh, gap drops,
share-link weights, build stamp, and independent formula spot-checks.
"""

from __future__ import annotations

import math
import tempfile
import unittest
from concurrent.futures import TimeoutError as FuturesTimeout
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from data_loader import (
    DOWNLOAD_TIMEOUT_SECONDS,
    MarketDataError,
    _run_with_timeout,
    _validate,
    fetch_history,
    load_prices,
)
from features import (
    TRADING_DAYS,
    annualized_volatility,
    cumulative_return,
    daily_log_returns,
    daily_simple_returns,
    historical_var,
    parametric_var,
    sharpe_ratio,
)
from version_info import PACKAGE_VERSION, build_label, git_sha

APP = str(Path(__file__).resolve().parent.parent / "app.py")


class FormulaSpotCheckTests(unittest.TestCase):
    def test_core_formulas_match_independent_hand_calcs(self) -> None:
        idx = pd.bdate_range("2020-01-02", periods=40)
        growth = np.array([0.01, -0.005] * 20, dtype=float)
        prices = pd.Series(100.0 * np.cumprod(1.0 + growth), index=idx, name="adj_close")
        self.assertAlmostEqual(
            cumulative_return(prices),
            float(prices.iloc[-1] / prices.iloc[0] - 1.0),
        )
        log_returns = daily_log_returns(prices)
        hand_vol = float(log_returns.std(ddof=1) * math.sqrt(TRADING_DAYS))
        self.assertAlmostEqual(annualized_volatility(log_returns), hand_vol)
        simple = daily_simple_returns(prices)
        hand_sharpe = float(simple.mean() / simple.std(ddof=1) * math.sqrt(TRADING_DAYS))
        self.assertAlmostEqual(sharpe_ratio(simple, 0.0), hand_sharpe)
        # Historical VaR drops non-finite returns first (same as features._finite).
        from features import _finite

        clean = _finite(simple)
        q05 = float(np.quantile(clean.to_numpy(), 0.05))
        self.assertAlmostEqual(historical_var(simple), max(0.0, -q05), places=12)
        from statistics import NormalDist

        mu = float(clean.mean())
        sigma = float(clean.std(ddof=1))
        z = NormalDist().inv_cdf(0.05)
        hand_parametric = max(0.0, -(mu + z * sigma))
        self.assertAlmostEqual(parametric_var(simple), hand_parametric, places=12)


class DownloadTimeoutTests(unittest.TestCase):
    def test_run_with_timeout_raises_on_deadline(self) -> None:
        def hang() -> None:
            import time

            time.sleep(5)

        with self.assertRaises(FuturesTimeout):
            _run_with_timeout(hang, 0.05)
        self.assertGreaterEqual(DOWNLOAD_TIMEOUT_SECONDS, 10)

    def test_yahoo_timeout_becomes_market_data_error(self) -> None:
        with patch("data_loader._run_with_timeout", side_effect=FuturesTimeout()):
            with self.assertRaises(MarketDataError) as caught:
                from data_loader import _download_yahoo_once

                _download_yahoo_once("AAPL")
        self.assertIn("timed out", str(caught.exception).lower())


class StaleRefreshTests(unittest.TestCase):
    def test_failed_refresh_marks_fresh_cache_stale(self) -> None:
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
            self.assertFalse(first.stale)
            with patch("data_loader._download_yahoo", down):
                fallback = fetch_history("AAPL", refresh=True, cache_dir=temp_dir)
        self.assertTrue(fallback.stale)
        self.assertTrue(fallback.from_cache)
        self.assertEqual(fallback.fetched_at, first.fetched_at)


class GapHandlingTests(unittest.TestCase):
    def test_validate_drops_nan_rows_instead_of_filling(self) -> None:
        idx = pd.bdate_range("2020-01-02", periods=5)
        frame = pd.DataFrame(
            {
                "open": [1, 1, np.nan, 1, 1],
                "high": [1, 1, 1, 1, 1],
                "low": [1, 1, 1, 1, 1],
                "close": [1, 1, 1, 1, 1],
                "volume": [1, 1, 1, 1, 1],
                "adj_close": [1, 1, 1, 1, 1],
            },
            index=idx,
        )
        clean = _validate(frame)
        self.assertEqual(len(clean), 4)
        self.assertEqual(clean.attrs.get("dropped_gap_rows"), 1)
        self.assertFalse(clean.isna().any().any())


class ShareLinkAndBuildStampTests(unittest.TestCase):
    def test_share_link_includes_unequal_weights_and_cost(self) -> None:
        page = AppTest.from_file(APP, default_timeout=60)
        page.query_params["data"] = "Offline"
        page.query_params["symbols"] = ["AAPL", "MSFT"]
        page.query_params["range"] = "Max"
        page.query_params["portfolio"] = "True"
        page = page.run()
        self.assertEqual(list(page.exception), [])
        page.sidebar.number_input(key="w_AAPL").set_value(70.0).run()
        page.sidebar.number_input(key="w_MSFT").set_value(30.0).run()
        page.sidebar.number_input(key="cost_bps").set_value(25.0).run()
        self.assertEqual(list(page.exception), [])
        link = page.code[0].value
        query = parse_qs(urlsplit(link).query)
        self.assertEqual(query.get("portfolio"), ["True"])
        self.assertEqual(query.get("w_AAPL"), ["70"])
        self.assertEqual(query.get("w_MSFT"), ["30"])
        self.assertEqual(query.get("cost_bps"), ["25"])

    def test_build_stamp_appears_on_page(self) -> None:
        page = AppTest.from_file(APP, default_timeout=60)
        page.query_params["data"] = "Offline"
        page = page.run()
        captions = " ".join(c.value for c in page.caption)
        self.assertIn(f"Build v{PACKAGE_VERSION}", captions)
        self.assertIn("reboot the Streamlit Cloud app", captions)
        label = build_label()
        self.assertTrue(label.startswith(f"v{PACKAGE_VERSION}"))
        # SHA may be missing in some sandboxes; when present it is short hex.
        sha = git_sha()
        if sha is not None:
            self.assertRegex(sha, r"^[0-9a-f]{7}$")


if __name__ == "__main__":
    unittest.main()
