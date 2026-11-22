"""
The page itself, driven headless with Streamlit's AppTest.

Offline mode makes these deterministic: the bundled files never change.
The live-mode cases replace the Yahoo download with a stub and point the
cache at a temporary folder, so they need no network and leave the real
cache alone.
"""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from dashboard import run_briefing
from data_loader import MarketDataError, load_prices

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT_SECONDS = 60
CARD_COUNT = 10
# Height of the header bar Streamlit Community Cloud draws over the page.
CLOUD_HEADER_REM = 3.75


def _page(**params) -> AppTest:
    page = AppTest.from_file(APP, default_timeout=TIMEOUT_SECONDS)
    for name, value in params.items():
        page.query_params[name] = value
    return page.run()


def _cards(page: AppTest) -> dict[str, str]:
    return {card.label: card.value for card in page.metric}


def _html(page: AppTest, marker: str) -> str:
    return " ".join(block.value for block in page.markdown if marker in block.value)


class OfflinePageTests(unittest.TestCase):
    def test_default_offline_view_renders_every_section(self) -> None:
        page = _page(data="Offline")
        self.assertEqual(list(page.exception), [])
        self.assertEqual(list(page.error), [])
        self.assertEqual(page.title[0].value, "AAPL vs SPY")
        self.assertIn("Market briefing · offline demo", _html(page, 'class="kicker"'))
        self.assertEqual(len(page.metric), CARD_COUNT)
        self.assertEqual([tab.label for tab in page.tabs], ["Compare", "Price", "Drawdown", "Risk", "Returns"])
        self.assertIn("Side by side", [heading.value for heading in page.subheader])
        self.assertIn("What the numbers say about AAPL", [heading.value for heading in page.subheader])
        self.assertIn("bundled aapl_raw.csv", _html(page, "status-card"))
        self.assertIn("does not call Yahoo Finance", page.sidebar.info[0].value)

    def test_cards_match_the_calculation(self) -> None:
        page = _page(data="Offline", range="Max")
        stats = run_briefing(offline=True, preset="Max").report.stats
        cards = _cards(page)
        self.assertEqual(cards["Cumulative return"], f"{stats.cumulative_return:+.1%}")
        self.assertEqual(cards["Annualized return"], f"{stats.annualized_return:+.1%}")
        self.assertEqual(cards["Annualized volatility"], f"{stats.annualized_volatility:.1%}")
        self.assertEqual(cards["Maximum drawdown"], f"{stats.max_drawdown:.1%}")
        self.assertEqual(cards["Vs SPY"], f"{stats.excess_return * 100:+.1f} pp")
        self.assertEqual(cards["Sharpe ratio"], f"{stats.sharpe:.2f}")
        self.assertEqual(cards["Sortino ratio"], f"{stats.sortino:.2f}")
        self.assertEqual(cards["Beta vs SPY"], f"{stats.beta:.2f}")
        self.assertEqual(cards["Correlation vs SPY"], f"{stats.correlation:.2f}")
        self.assertEqual(cards["Volatility label"], stats.vol_label.capitalize())
        # The numbers the owner checked on the hosted app in the last round.
        self.assertEqual(cards["Cumulative return"], "+10.1%")
        self.assertEqual(cards["Maximum drawdown"], "-30.4%")
        self.assertEqual(cards["Vs SPY"], "-6.3 pp")

    def test_a_link_reproduces_a_multi_symbol_view(self) -> None:
        page = _page(
            data="Offline",
            symbols=["AAPL", "MSFT"],
            benchmark="QQQ",
            range="Max",
            focus="MSFT",
            ma_short="20",
            ma_long="100",
            rf="2",
        )
        self.assertEqual(list(page.exception), [])
        self.assertEqual(page.title[0].value, "AAPL, MSFT vs QQQ")
        self.assertEqual(page.sidebar.multiselect(key="symbols").value, ["AAPL", "MSFT"])
        self.assertEqual(page.sidebar.selectbox(key="benchmark").value, "QQQ")
        self.assertEqual(page.button_group(key="range").value, "Max")
        self.assertEqual(page.button_group(key="focus").value, "MSFT")
        self.assertEqual(page.sidebar.number_input(key="ma_short").value, 20)
        self.assertEqual(page.sidebar.number_input(key="ma_long").value, 100)
        self.assertAlmostEqual(page.sidebar.number_input(key="rf").value, 2.0)
        headings = [heading.value for heading in page.subheader]
        self.assertIn("What the numbers say about MSFT", headings)
        self.assertIn("Across the symbols", headings)

        expected = run_briefing(
            tickers=["AAPL", "MSFT"],
            benchmark="QQQ",
            focus="MSFT",
            offline=True,
            preset="Max",
            ma_short=20,
            ma_long=100,
            risk_free=0.02,
        ).report.stats
        cards = _cards(page)
        self.assertEqual(cards["Cumulative return"], f"{expected.cumulative_return:+.1%}")
        self.assertEqual(cards["Sharpe ratio"], f"{expected.sharpe:.2f}")
        self.assertEqual(cards["Beta vs QQQ"], f"{expected.beta:.2f}")
        summary = _html(page, "summary-card")
        self.assertIn("20-day average", summary)
        self.assertIn("risk-free rate of 2.0% a year", summary)
        self.assertIn("Ranked by cumulative return: MSFT", summary)

        table = _html(page, "compare-wrap")
        self.assertIn('<tr class="focus"><th scope="row">MSFT</th>', table)
        self.assertIn("QQQ (benchmark)", table)
        # The copyable link carries every setting that is not a default.
        link = page.code[0].value
        for part in ("data=Offline", "symbols=AAPL&symbols=MSFT", "benchmark=QQQ", "range=Max",
                     "focus=MSFT", "ma_short=20", "ma_long=100", "rf=2"):
            self.assertIn(part, link)

    def test_the_default_view_has_an_empty_link(self) -> None:
        page = _page()
        if page.error:
            self.skipTest(f"Live data unavailable: {page.error[0].value}")
        self.assertEqual(page.code[0].value, "(default view)")

    def test_custom_range_from_a_link(self) -> None:
        page = _page(data="Offline", range="Custom", start="2016-01-04", end="2016-06-30")
        self.assertEqual(list(page.exception), [])
        self.assertEqual(page.date_input(key="start").value.isoformat(), "2016-01-04")
        self.assertEqual(page.date_input(key="end").value.isoformat(), "2016-06-30")
        expected = run_briefing(offline=True, start="2016-01-04", end="2016-06-30", preset=None).report.stats
        cards = _cards(page)
        self.assertEqual(cards["Cumulative return"], f"{expected.cumulative_return:+.1%}")
        # Six months is a part-year, so the annualized card is a dash, not a number.
        self.assertEqual(cards["Annualized return"], "—")
        self.assertIn("2016-01-04 to 2016-06-30", _html(page, "status-card").replace("\n", " ").replace("  ", " "))
        self.assertIn("range=Custom&start=2016-01-04&end=2016-06-30", page.code[0].value)

    def test_a_one_day_range_is_an_error_not_a_crash(self) -> None:
        page = _page(data="Offline", range="Custom", start="2016-06-06", end="2016-06-06")
        self.assertEqual(list(page.exception), [])
        self.assertEqual(len(page.metric), 0)
        self.assertIn("A return needs two adjusted closes", page.error[0].value)

    def test_changing_the_range_and_the_focus_updates_the_page(self) -> None:
        page = _page(data="Offline", symbols=["AAPL", "MSFT"])
        one_year = _cards(page)["Cumulative return"]
        page.button_group(key="range").set_value("Max").run()
        self.assertEqual(_cards(page)["Cumulative return"], "+10.1%")
        self.assertNotEqual(one_year, "+10.1%")
        page.button_group(key="focus").set_value("MSFT").run()
        self.assertIn("What the numbers say about MSFT", [heading.value for heading in page.subheader])
        self.assertEqual(
            _cards(page)["Cumulative return"],
            f"{run_briefing(tickers=['AAPL', 'MSFT'], focus='MSFT', offline=True, preset='Max').report.stats.cumulative_return:+.1%}",
        )

    def test_switching_the_benchmark_changes_the_relative_cards(self) -> None:
        page = _page(data="Offline", range="Max")
        against_spy = _cards(page)
        page.sidebar.selectbox(key="benchmark").set_value("QQQ").run()
        against_qqq = _cards(page)
        self.assertIn("Vs QQQ", against_qqq)
        self.assertNotIn("Vs SPY", against_qqq)
        self.assertNotEqual(against_spy["Beta vs SPY"], against_qqq["Beta vs QQQ"])
        # The stock's own figures do not depend on the benchmark.
        self.assertEqual(against_spy["Cumulative return"], against_qqq["Cumulative return"])
        self.assertEqual(against_spy["Sharpe ratio"], against_qqq["Sharpe ratio"])

    def test_average_windows_in_the_wrong_order_are_explained(self) -> None:
        page = _page(data="Offline")
        page.sidebar.number_input(key="ma_short").set_value(250).run()
        self.assertEqual(list(page.exception), [])
        self.assertIn("must be shorter than the long average", page.error[0].value)

    def test_a_symbol_the_offline_demo_lacks_is_dropped_from_a_link(self) -> None:
        page = _page(data="Offline", symbols=["NVDA"])
        self.assertEqual(list(page.exception), [])
        self.assertEqual(page.sidebar.multiselect(key="symbols").value, ["AAPL"])
        self.assertEqual(page.title[0].value, "AAPL vs SPY")

    def test_clearing_every_symbol_asks_for_one(self) -> None:
        page = _page(data="Offline")
        page.sidebar.multiselect(key="symbols").set_value([]).run()
        self.assertEqual(list(page.exception), [])
        self.assertEqual(len(page.metric), 0)
        self.assertTrue(any("Add at least one symbol" in note.value for note in page.info))

    def test_the_label_above_the_title_clears_the_hosted_header(self) -> None:
        page = _page(data="Offline")
        css = _html(page, "<style>")
        match = re.search(r"\.block-container \{padding-top: ([0-9.]+)rem;", css)
        self.assertIsNotNone(match)
        self.assertGreater(float(match.group(1)), CLOUD_HEADER_REM)


class LivePageWithStubbedYahooTests(unittest.TestCase):
    """Live mode with the download replaced, so outages and bad tickers can be staged."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        cache = patch("data_loader.CACHE_DIR", Path(self.temp.name))
        cache.start()
        self.addCleanup(cache.stop)
        self.frame = load_prices()

    def _download(self, known: tuple[str, ...]):
        def fake(ticker: str) -> pd.DataFrame:
            if ticker not in known:
                raise MarketDataError(f"No price history for {ticker}. Check the symbol.")
            out = self.frame.copy()
            out.attrs["ticker"] = ticker
            return out

        return fake

    def test_one_unknown_ticker_is_a_warning_and_the_rest_still_show(self) -> None:
        with patch("data_loader._download_yahoo", self._download(("AAPL", "SPY"))):
            page = _page(symbols=["AAPL", "ZZZNOTREAL"])
        self.assertEqual(list(page.exception), [])
        self.assertEqual(list(page.error), [])
        self.assertEqual(page.title[0].value, "AAPL vs SPY")
        self.assertEqual(len(page.metric), CARD_COUNT)
        warnings = " ".join(note.value for note in page.warning)
        self.assertIn("ZZZNOTREAL was left out", warnings)
        self.assertIn("No price history for ZZZNOTREAL", warnings)

    def test_only_unknown_tickers_is_an_error_with_a_way_out(self) -> None:
        with patch("data_loader._download_yahoo", self._download(("SPY",))):
            page = _page(symbols=["ZZZNOTREAL"])
        self.assertEqual(list(page.exception), [])
        self.assertEqual(len(page.metric), 0)
        self.assertIn("No price history for ZZZNOTREAL", page.error[0].value)
        self.assertTrue(any("offline demo is available" in note.value for note in page.info))

    def test_a_yahoo_outage_with_nothing_cached_points_to_the_offline_demo(self) -> None:
        with patch("data_loader._download_yahoo", self._download(())):
            page = _page()
        self.assertEqual(list(page.exception), [])
        self.assertIn("No price history for AAPL", page.error[0].value)
        self.assertTrue(any("offline demo is available" in note.value for note in page.info))

    def test_a_missing_benchmark_keeps_the_stock_and_says_what_is_missing(self) -> None:
        with patch("data_loader._download_yahoo", self._download(("AAPL",))):
            page = _page(benchmark="NOBENCH")
        self.assertEqual(list(page.exception), [])
        cards = _cards(page)
        self.assertEqual(len(cards), CARD_COUNT)
        self.assertEqual(cards["Vs benchmark"], "—")
        self.assertEqual(cards["Beta vs benchmark"], "—")
        self.assertNotEqual(cards["Cumulative return"], "—")
        self.assertIn("Benchmark unavailable", " ".join(note.value for note in page.warning))

    def test_typed_tickers_are_tidied(self) -> None:
        with patch("data_loader._download_yahoo", self._download(("AAPL", "MSFT", "SPY"))):
            page = _page()
            page.sidebar.multiselect(key="symbols").set_value(["AAPL", " msft ", "MSFT"]).run()
        self.assertEqual(list(page.exception), [])
        self.assertEqual(page.sidebar.multiselect(key="symbols").value, ["AAPL", "MSFT"])
        self.assertEqual(page.title[0].value, "AAPL, MSFT vs SPY")


if __name__ == "__main__":
    unittest.main()
