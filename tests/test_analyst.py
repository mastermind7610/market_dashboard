"""
Phase 4 analyst: typed facts, rule-based narrative, optional LLM (off by default).

Guarantees:
- every number in the rule-based narrative is traceable to AnalystFacts
- LLM stays off unless explicitly enabled with a key
- not-financial-advice disclaimer is always present
- missing data produces notes instead of invented figures
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from analyst import (
    LLM_MODE,
    NOT_FINANCIAL_ADVICE,
    RULE_BASED_MODE,
    UNAVAILABLE_MODE,
    AnalystFacts,
    FactValue,
    NarrativeClaim,
    assert_claims_traceable,
    build_analyst_pack,
    extract_facts,
    optional_llm_brief,
    rule_based_brief,
    untraceable_numbers,
)
from dashboard import run_briefing
from features import PORTFOLIO_LABEL, comparison_table


def _offline_pack(**kwargs):
    briefing = run_briefing(
        tickers=["AAPL", "MSFT"],
        offline=True,
        preset="Max",
        portfolio=True,
        **kwargs,
    )
    return briefing, build_analyst_pack(
        focus_report=briefing.report,
        comparison=briefing.comparison,
        portfolio_report=briefing.portfolio_report,
        cost_report=briefing.cost_report,
        enable_llm=False,
    )


class AnalystFactsTests(unittest.TestCase):
    def test_facts_come_from_computed_reports(self) -> None:
        briefing, pack = _offline_pack()
        self.assertEqual(pack.facts.focus, "AAPL")
        self.assertEqual(pack.facts.benchmark, "SPY")
        keys = pack.facts.as_map()
        self.assertIn("focus.cumulative_return", keys)
        self.assertAlmostEqual(
            float(keys["focus.cumulative_return"].value),
            briefing.report.stats.cumulative_return,
        )
        self.assertEqual(
            keys["focus.cumulative_return"].display,
            f"{briefing.report.stats.cumulative_return:+.1%}",
        )
        self.assertIn("portfolio.cumulative_return", keys)
        self.assertTrue(any(key.startswith("diversification.") for key in keys))
        self.assertIn("cost.cost_bps", keys)

    def test_briefing_attaches_analyst_pack_by_default(self) -> None:
        briefing = run_briefing(offline=True, preset="Max")
        self.assertIsNotNone(briefing.analyst)
        self.assertEqual(briefing.analyst.rule_based.mode, RULE_BASED_MODE)
        self.assertIsNone(briefing.analyst.llm)


class PortfolioBenchmarkWindowTests(unittest.TestCase):
    def test_portfolio_narrative_matches_table_not_focus_spy_row(self) -> None:
        """
        Offline AAPL+MSFT Max with MSFT in focus: the SPY table row is +21.4%
        on SPY's own sessions, while the portfolio's aligned SPY print is
        +16.5%. The analyst must quote the portfolio-window numbers and name
        that window — never the focus-only or SPY-own-row figure.
        """
        briefing = run_briefing(
            tickers=["AAPL", "MSFT"],
            focus="MSFT",
            offline=True,
            preset="Max",
            portfolio=True,
        )
        pack = briefing.analyst
        self.assertIsNotNone(pack)
        self.assertIsNotNone(briefing.portfolio_report)
        assert_claims_traceable(pack.rule_based, pack.facts)

        port = briefing.portfolio_report.stats
        focus = briefing.report.stats
        table = comparison_table(briefing.comparison)
        port_return = float(table.loc[PORTFOLIO_LABEL, "Return"])
        spy_own = next(
            float(table.loc[idx, "Return"])
            for idx in table.index
            if str(idx).startswith("SPY")
        )

        self.assertAlmostEqual(port_return, port.cumulative_return)
        self.assertAlmostEqual(port.benchmark_cumulative_return, 0.165, places=2)
        self.assertAlmostEqual(spy_own, 0.214, places=2)
        # Focus (MSFT) shares SPY's longer own-session print — that is the trap.
        self.assertAlmostEqual(focus.benchmark_cumulative_return, spy_own, places=3)
        self.assertNotAlmostEqual(
            port.benchmark_cumulative_return,
            focus.benchmark_cumulative_return,
            places=2,
        )

        claim = next(c for c in pack.rule_based.claims if c.topic == "portfolio")
        port_disp = f"{port.cumulative_return:+.1%}"
        bench_disp = f"{port.benchmark_cumulative_return:+.1%}"
        start = port.window_start.date().isoformat()
        end = port.window_end.date().isoformat()
        self.assertIn(port_disp, claim.text)
        self.assertIn(bench_disp, claim.text)
        self.assertIn(start, claim.text)
        self.assertIn(end, claim.text)
        self.assertIn("shared sessions", claim.text)
        self.assertNotIn(f"{spy_own:+.1%}", claim.text)
        self.assertIn("portfolio.benchmark_cumulative_return", claim.fact_keys)
        self.assertNotIn("focus.benchmark_cumulative_return", claim.fact_keys)


class TraceabilityTests(unittest.TestCase):
    def test_rule_based_claims_are_fully_traceable(self) -> None:
        _, pack = _offline_pack()
        assert_claims_traceable(pack.rule_based, pack.facts)
        self.assertGreaterEqual(len(pack.rule_based.claims), 3)

    def test_invented_number_is_detected(self) -> None:
        _, pack = _offline_pack()
        fake = NarrativeClaim(
            text="AAPL will return +99.9% next year.",
            fact_keys=("focus.cumulative_return",),
            topic="hype",
        )
        self.assertIn("+99.9%", untraceable_numbers(fake.text, pack.facts))

    def test_iso_dates_in_window_are_not_flagged(self) -> None:
        facts = AnalystFacts(
            focus="AAPL",
            benchmark="SPY",
            window_start="2015-02-17",
            window_end="2017-02-16",
            items=(
                FactValue("focus.cumulative_return", "Cumulative return", 0.101, "+10.1%"),
            ),
        )
        text = "AAPL returned +10.1% over 2015-02-17 to 2017-02-16."
        self.assertEqual(untraceable_numbers(text, facts), [])


class DisclaimerAndSafetyTests(unittest.TestCase):
    def test_disclaimer_is_always_present(self) -> None:
        _, pack = _offline_pack()
        self.assertEqual(pack.rule_based.disclaimer, NOT_FINANCIAL_ADVICE)
        self.assertNotIn("guaranteed", pack.rule_based.disclaimer.lower())
        self.assertIn("not financial advice", pack.rule_based.disclaimer.lower())
        for claim in pack.rule_based.claims:
            lowered = claim.text.lower()
            self.assertNotIn("buy now", lowered)
            self.assertNotIn("guaranteed profit", lowered)
            self.assertNotIn("will outperform", lowered)

    def test_missing_fields_produce_notes_not_fabricated_claims(self) -> None:
        sparse = AnalystFacts(
            focus="ZZZ",
            benchmark=None,
            window_start="2020-01-01",
            window_end="2020-01-02",
            items=(),
        )
        brief = rule_based_brief(sparse)
        self.assertEqual(brief.claims, ())
        self.assertTrue(brief.missing_notes)
        self.assertEqual(brief.disclaimer, NOT_FINANCIAL_ADVICE)


class OptionalLlmTests(unittest.TestCase):
    def test_llm_stays_off_by_default(self) -> None:
        briefing = run_briefing(offline=True, preset="Max", enable_llm_analyst=False)
        self.assertIsNone(briefing.analyst.llm)

    def test_llm_without_key_is_unavailable_not_invented(self) -> None:
        briefing = run_briefing(
            offline=True,
            preset="Max",
            enable_llm_analyst=True,
            llm_api_key=None,
        )
        llm = briefing.analyst.llm
        self.assertIsNotNone(llm)
        self.assertEqual(llm.mode, UNAVAILABLE_MODE)
        self.assertEqual(llm.claims, ())
        self.assertEqual(llm.disclaimer, NOT_FINANCIAL_ADVICE)
        self.assertTrue(any("no API key" in note for note in llm.missing_notes))

    def test_llm_claims_drop_untraceable_numbers(self) -> None:
        _, pack = _offline_pack()
        forged = '{"claims":[{"topic":"hype","text":"Expect +77.7% next quarter.","fact_keys":[]}]}'
        with patch("analyst._chat_completion", return_value=forged):
            brief = optional_llm_brief(pack.facts, api_key="sk-test")
        self.assertEqual(brief.mode, UNAVAILABLE_MODE)
        self.assertEqual(brief.claims, ())

    def test_llm_keeps_only_traceable_claims(self) -> None:
        _, pack = _offline_pack()
        good_display = pack.facts.as_map()["focus.cumulative_return"].display
        payload = {
            "claims": [
                {
                    "topic": "performance",
                    "text": f"Focus cumulative return was {good_display}.",
                    "fact_keys": ["focus.cumulative_return"],
                },
                {
                    "topic": "hype",
                    "text": "Secret alpha of +88.8% awaits.",
                    "fact_keys": [],
                },
            ]
        }
        import json

        with patch("analyst._chat_completion", return_value=json.dumps(payload)):
            brief = optional_llm_brief(pack.facts, api_key="sk-test")
        self.assertEqual(brief.mode, LLM_MODE)
        self.assertEqual(len(brief.claims), 1)
        assert_claims_traceable(brief, pack.facts)


class ExtractHelpersTests(unittest.TestCase):
    def test_extract_facts_skips_none_optional_metrics(self) -> None:
        briefing = run_briefing(offline=True, preset="1M")
        facts = extract_facts(
            focus_report=briefing.report,
            comparison=briefing.comparison,
        )
        # Short windows may omit annualized return; that must not crash.
        self.assertEqual(facts.focus, briefing.focus)
        self.assertTrue(facts.items)


if __name__ == "__main__":
    unittest.main()
