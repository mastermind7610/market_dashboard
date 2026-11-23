"""
analyst.py — OPTIONAL ANALYST LAYER
===================================
Builds a typed fact sheet from already-computed briefing numbers, then
writes a rule-based narrative that may only quote those facts.

This module does not load prices and does not draw charts. It sits beside
the orchestration layer: dashboard.py gathers a Briefing, then asks here
for an AnalystPack.

LLM interpretation is optional, off by default, and never required to
run the app. No API keys are read from the repository.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass

from features import (
    Comparison,
    DiversificationStats,
    PeriodReport,
    PeriodStats,
    PortfolioCostReport,
)

NOT_FINANCIAL_ADVICE = (
    "Not financial advice. This briefing explains historical numbers already "
    "computed from the selected window. It does not recommend buys, sells, "
    "or position sizes, and it does not promise future profits or losses."
)

RULE_BASED_MODE = "rule_based"
LLM_MODE = "llm"
UNAVAILABLE_MODE = "unavailable"

# Optional LLM hook (stdlib only). Off unless the caller passes a key.
DEFAULT_LLM_BASE_URL = "https://api.openai.com/v1"
DEFAULT_LLM_MODEL = "gpt-4o-mini"
LLM_TIMEOUT_SECONDS = 20
LLM_MAX_CLAIMS = 8

# Number tokens we allow a narrative to repeat from the fact sheet.
_NUMBER_TOKEN = re.compile(
    r"(?<![A-Za-z])"
    r"[+-]?"
    r"(?:\d{1,3}(?:,\d{3})+|\d+)"
    r"(?:\.\d+)?"
    r"%?"
    r"(?![A-Za-z])"
)
# ISO dates in claims are labels from the window, not inventable market figures.
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
# Methodology confidence labels that appear in VaR wording (not market data).
_METHODOLOGY_NUMBER_TOKENS = frozenset({"95", "95%"})


@dataclass(frozen=True)
class FactValue:
    """One computed field the narrative is allowed to quote."""

    key: str
    label: str
    value: float | int | str | None
    display: str


@dataclass(frozen=True)
class AnalystFacts:
    """Typed fact sheet for one briefing focus (and optional portfolio)."""

    focus: str
    benchmark: str | None
    window_start: str
    window_end: str
    items: tuple[FactValue, ...]

    def as_map(self) -> dict[str, FactValue]:
        return {item.key: item for item in self.items}

    def displays(self) -> set[str]:
        return {item.display for item in self.items if item.display and item.display != "—"}


@dataclass(frozen=True)
class NarrativeClaim:
    """One analyst sentence, grounded on fact keys."""

    text: str
    fact_keys: tuple[str, ...]
    topic: str


@dataclass(frozen=True)
class AnalystBrief:
    """Interpretation block. Always carries the not-financial-advice note."""

    mode: str
    title: str
    claims: tuple[NarrativeClaim, ...]
    disclaimer: str
    missing_notes: tuple[str, ...]
    source_label: str


@dataclass(frozen=True)
class AnalystPack:
    """Facts plus rule-based brief; optional LLM brief stays None when off."""

    facts: AnalystFacts
    rule_based: AnalystBrief
    llm: AnalystBrief | None = None


def build_analyst_pack(
    *,
    focus_report: PeriodReport,
    comparison: Comparison,
    portfolio_report: PeriodReport | None = None,
    cost_report: PortfolioCostReport | None = None,
    enable_llm: bool = False,
    llm_api_key: str | None = None,
    llm_base_url: str = DEFAULT_LLM_BASE_URL,
    llm_model: str = DEFAULT_LLM_MODEL,
) -> AnalystPack:
    """
    Fact sheet + rule-based narrative. LLM runs only when explicitly enabled
    and a caller-supplied API key is present.
    """
    facts = extract_facts(
        focus_report=focus_report,
        comparison=comparison,
        portfolio_report=portfolio_report,
        cost_report=cost_report,
    )
    rule_based = rule_based_brief(facts)
    llm_brief = None
    if enable_llm:
        llm_brief = optional_llm_brief(
            facts,
            api_key=llm_api_key,
            base_url=llm_base_url,
            model=llm_model,
        )
    return AnalystPack(facts=facts, rule_based=rule_based, llm=llm_brief)


def extract_facts(
    *,
    focus_report: PeriodReport,
    comparison: Comparison,
    portfolio_report: PeriodReport | None = None,
    cost_report: PortfolioCostReport | None = None,
) -> AnalystFacts:
    """Pull a flat, display-ready fact list from computed reports only."""
    focus = focus_report.stats
    items: list[FactValue] = []
    items.extend(_period_facts(focus, prefix="focus"))
    if portfolio_report is not None:
        items.extend(_period_facts(portfolio_report.stats, prefix="portfolio"))
    if comparison.diversification is not None:
        items.extend(_diversification_facts(comparison.diversification))
    if cost_report is not None:
        items.extend(_cost_facts(cost_report))
    skip = {focus.ticker}
    if portfolio_report is not None:
        skip.add(portfolio_report.stats.ticker)
    for row in comparison.rows:
        if row.ticker in skip:
            continue
        # Keep peer returns for comparison sentences; prefix with ticker.
        items.append(
            _fact(
                f"peer.{row.ticker}.cumulative_return",
                f"{row.ticker} cumulative return",
                row.cumulative_return,
                _pct(row.cumulative_return, signed=True),
            )
        )
        if row.max_drawdown is not None:
            items.append(
                _fact(
                    f"peer.{row.ticker}.max_drawdown",
                    f"{row.ticker} max drawdown",
                    row.max_drawdown,
                    _pct(row.max_drawdown),
                )
            )
    return AnalystFacts(
        focus=focus.ticker,
        benchmark=focus.benchmark,
        window_start=focus.window_start.date().isoformat(),
        window_end=focus.window_end.date().isoformat(),
        items=tuple(items),
    )


def rule_based_brief(facts: AnalystFacts) -> AnalystBrief:
    """
    Template analyst. Every numeric token in a claim must appear in the
    fact sheet displays (enforced by ``assert_claims_traceable`` in tests).
    """
    m = facts.as_map()
    claims: list[NarrativeClaim] = []
    missing: list[str] = []

    def have(*keys: str) -> bool:
        return all(key in m and m[key].value is not None for key in keys)

    # Performance versus the benchmark when both cumulative returns exist.
    if have("focus.cumulative_return", "focus.benchmark_cumulative_return", "focus.excess_return"):
        claims.append(
            NarrativeClaim(
                topic="performance",
                fact_keys=(
                    "focus.cumulative_return",
                    "focus.benchmark_cumulative_return",
                    "focus.excess_return",
                ),
                text=(
                    f"{facts.focus} returned {m['focus.cumulative_return'].display} over "
                    f"{facts.window_start} to {facts.window_end}, while "
                    f"{facts.benchmark or 'the benchmark'} returned "
                    f"{m['focus.benchmark_cumulative_return'].display} "
                    f"(gap {m['focus.excess_return'].display})."
                ),
            )
        )
    elif have("focus.cumulative_return"):
        claims.append(
            NarrativeClaim(
                topic="performance",
                fact_keys=("focus.cumulative_return",),
                text=(
                    f"{facts.focus} returned {m['focus.cumulative_return'].display} "
                    f"from {facts.window_start} to {facts.window_end}."
                ),
            )
        )
    else:
        missing.append("Focus cumulative return is unavailable for this window.")

    if have("focus.annualized_volatility", "focus.vol_label"):
        claims.append(
            NarrativeClaim(
                topic="risk",
                fact_keys=("focus.annualized_volatility", "focus.vol_label"),
                text=(
                    f"Annualized volatility was {m['focus.annualized_volatility'].display}, "
                    f"labelled {m['focus.vol_label'].display} against this symbol's own history."
                ),
            )
        )
    elif have("focus.annualized_volatility"):
        claims.append(
            NarrativeClaim(
                topic="risk",
                fact_keys=("focus.annualized_volatility",),
                text=(
                    f"Annualized volatility was {m['focus.annualized_volatility'].display}."
                ),
            )
        )
    else:
        missing.append("Volatility is unavailable for this window.")

    if have("focus.max_drawdown"):
        keys = ["focus.max_drawdown"]
        extra = ""
        if have("focus.drawdown_days_to_trough"):
            keys.append("focus.drawdown_days_to_trough")
            extra += f" Peak to trough spanned {m['focus.drawdown_days_to_trough'].display} sessions."
        if have("focus.longest_underwater_days"):
            keys.append("focus.longest_underwater_days")
            extra += (
                f" The longest underwater stretch lasted "
                f"{m['focus.longest_underwater_days'].display} sessions."
            )
        claims.append(
            NarrativeClaim(
                topic="drawdown",
                fact_keys=tuple(keys),
                text=f"Maximum drawdown was {m['focus.max_drawdown'].display}.{extra}",
            )
        )

    if have("focus.value_at_risk_95", "focus.cvar_95"):
        claims.append(
            NarrativeClaim(
                topic="tail_risk",
                fact_keys=("focus.value_at_risk_95", "focus.cvar_95"),
                text=(
                    f"95% historical VaR was {m['focus.value_at_risk_95'].display}; "
                    f"95% Conditional VaR was {m['focus.cvar_95'].display}."
                ),
            )
        )
    elif have("focus.value_at_risk_95"):
        claims.append(
            NarrativeClaim(
                topic="tail_risk",
                fact_keys=("focus.value_at_risk_95",),
                text=f"95% historical VaR was {m['focus.value_at_risk_95'].display}.",
            )
        )

    if have("focus.sharpe", "focus.sortino"):
        claims.append(
            NarrativeClaim(
                topic="risk_adjusted",
                fact_keys=("focus.sharpe", "focus.sortino"),
                text=(
                    f"Sharpe was {m['focus.sharpe'].display} and Sortino was "
                    f"{m['focus.sortino'].display} at the selected risk-free rate."
                ),
            )
        )

    if have("focus.beta", "focus.correlation"):
        claims.append(
            NarrativeClaim(
                topic="benchmark_relationship",
                fact_keys=("focus.beta", "focus.correlation"),
                text=(
                    f"Beta versus {facts.benchmark or 'the benchmark'} was "
                    f"{m['focus.beta'].display} with correlation "
                    f"{m['focus.correlation'].display}."
                ),
            )
        )

    if have("focus.alpha"):
        claims.append(
            NarrativeClaim(
                topic="alpha",
                fact_keys=("focus.alpha",),
                text=(
                    f"Annualised Jensen alpha versus {facts.benchmark or 'the benchmark'} "
                    f"was {m['focus.alpha'].display}."
                ),
            )
        )

    # Portfolio vs benchmark must use the portfolio's own aligned window —
    # not the focus symbol's. The benchmark's own table row can cover a
    # longer own-session span (e.g. offline SPY Max +21.4% vs +16.5% on
    # the shared AAPL+MSFT portfolio sessions).
    if have("portfolio.cumulative_return", "portfolio.benchmark_cumulative_return"):
        port_keys = [
            "portfolio.cumulative_return",
            "portfolio.benchmark_cumulative_return",
        ]
        window_bits = ""
        if have("portfolio.window_start", "portfolio.window_end"):
            port_keys.extend(["portfolio.window_start", "portfolio.window_end"])
            window_bits = (
                f" from {m['portfolio.window_start'].display} "
                f"to {m['portfolio.window_end'].display}"
            )
        claims.append(
            NarrativeClaim(
                topic="portfolio",
                fact_keys=tuple(port_keys),
                text=(
                    f"The portfolio returned {m['portfolio.cumulative_return'].display}"
                    f"{window_bits} on shared sessions with "
                    f"{facts.benchmark or 'the benchmark'}; "
                    f"{facts.benchmark or 'the benchmark'} returned "
                    f"{m['portfolio.benchmark_cumulative_return'].display} "
                    f"over those same sessions."
                ),
            )
        )
    elif have("portfolio.cumulative_return"):
        claims.append(
            NarrativeClaim(
                topic="portfolio",
                fact_keys=("portfolio.cumulative_return",),
                text=(
                    f"The portfolio returned {m['portfolio.cumulative_return'].display}."
                ),
            )
        )

    if have("diversification.ratio", "diversification.effective_n"):
        claims.append(
            NarrativeClaim(
                topic="diversification",
                fact_keys=("diversification.ratio", "diversification.effective_n"),
                text=(
                    f"Diversification ratio was {m['diversification.ratio'].display} with "
                    f"effective holdings {m['diversification.effective_n'].display}."
                ),
            )
        )

    if have("cost.total_turnover", "cost.cost_bps"):
        claims.append(
            NarrativeClaim(
                topic="costs",
                fact_keys=("cost.total_turnover", "cost.cost_bps", "cost.total_cost_drag"),
                text=(
                    f"Transaction cost setting was {m['cost.cost_bps'].display} bps with "
                    f"total turnover {m['cost.total_turnover'].display}"
                    + (
                        f" and fee drag {m['cost.total_cost_drag'].display}."
                        if have("cost.total_cost_drag")
                        else "."
                    )
                ),
            )
        )

    if not claims:
        missing.append("Not enough computed fields to draft an analyst summary.")

    return AnalystBrief(
        mode=RULE_BASED_MODE,
        title="Rule-based analyst (deterministic)",
        claims=tuple(claims),
        disclaimer=NOT_FINANCIAL_ADVICE,
        missing_notes=tuple(missing),
        source_label="Templates filled only from AnalystFacts",
    )


def optional_llm_brief(
    facts: AnalystFacts,
    *,
    api_key: str | None,
    base_url: str = DEFAULT_LLM_BASE_URL,
    model: str = DEFAULT_LLM_MODEL,
) -> AnalystBrief:
    """
    Optional LLM interpretation. Requires an explicit API key from the
    caller. On any failure, returns an unavailable brief — never invents
    numbers locally to fill the gap.
    """
    if not api_key or not str(api_key).strip():
        return AnalystBrief(
            mode=UNAVAILABLE_MODE,
            title="Optional LLM analyst (off)",
            claims=(),
            disclaimer=NOT_FINANCIAL_ADVICE,
            missing_notes=(
                "LLM analyst is enabled in the UI but no API key was provided. "
                "The rule-based analyst above still applies.",
            ),
            source_label="No request sent",
        )

    payload_facts = {
        item.key: {"label": item.label, "display": item.display, "value": item.value}
        for item in facts.items
        if item.value is not None
    }
    system = (
        "You are a markets briefing assistant. Use ONLY the JSON facts provided. "
        "Do not invent prices, returns, or risk figures. Do not recommend trades "
        "or promise profits. Reply with JSON: "
        '{"claims":[{"topic":"...","text":"...","fact_keys":["..."]}]} '
        f"with at most {LLM_MAX_CLAIMS} claims. Every number in text must match a "
        "display string from the facts."
    )
    user = json.dumps(
        {
            "focus": facts.focus,
            "benchmark": facts.benchmark,
            "window_start": facts.window_start,
            "window_end": facts.window_end,
            "facts": payload_facts,
        },
        default=str,
    )
    try:
        content = _chat_completion(
            api_key=str(api_key).strip(),
            base_url=base_url.rstrip("/"),
            model=model,
            system=system,
            user=user,
        )
        claims = _parse_llm_claims(content, facts)
        # Drop any claim that fails local traceability.
        safe = tuple(
            claim
            for claim in claims
            if not untraceable_numbers(claim.text, facts)
        )
        dropped = len(claims) - len(safe)
        notes: list[str] = []
        if dropped:
            notes.append(
                f"Dropped {dropped} LLM claim(s) that quoted numbers absent from the fact sheet."
            )
        if not safe:
            notes.append("LLM returned no traceable claims; showing rule-based analyst only.")
            return AnalystBrief(
                mode=UNAVAILABLE_MODE,
                title="Optional LLM analyst (no traceable output)",
                claims=(),
                disclaimer=NOT_FINANCIAL_ADVICE,
                missing_notes=tuple(notes),
                source_label=f"{model} via {base_url}",
            )
        return AnalystBrief(
            mode=LLM_MODE,
            title="Optional LLM analyst (model interpretation)",
            claims=safe,
            disclaimer=NOT_FINANCIAL_ADVICE,
            missing_notes=tuple(notes),
            source_label=f"{model} via {base_url}",
        )
    except Exception as exc:  # noqa: BLE001 — surface any transport/parse failure cleanly
        return AnalystBrief(
            mode=UNAVAILABLE_MODE,
            title="Optional LLM analyst (unavailable)",
            claims=(),
            disclaimer=NOT_FINANCIAL_ADVICE,
            missing_notes=(f"LLM request failed: {exc}",),
            source_label=f"{model} via {base_url}",
        )


def untraceable_numbers(text: str, facts: AnalystFacts) -> list[str]:
    """
    Return number tokens in ``text`` that do not appear in the fact sheet.

    ISO dates and methodology confidence labels (e.g. 95% VaR) are allowed.
    This is the guardrail tests use against invented market data.
    """
    allowed = set(facts.displays())
    allowed.update(_METHODOLOGY_NUMBER_TOKENS)
    allowed.add(facts.window_start)
    allowed.add(facts.window_end)
    allowed.add(facts.focus)
    if facts.benchmark:
        allowed.add(facts.benchmark)
    # Also allow bare year fragments from the window.
    allowed.add(facts.window_start[:4])
    allowed.add(facts.window_end[:4])
    # Allow integers that appear as whole displays without percent signs.
    for item in facts.items:
        if isinstance(item.value, int):
            allowed.add(str(item.value))
        if isinstance(item.value, float) and item.display.endswith("%"):
            # Accept the unsigned percent body too (e.g. 10.1 from +10.1%).
            body = item.display.lstrip("+").rstrip("%")
            allowed.add(body)
            allowed.add(body + "%")
            allowed.add("+" + body + "%")
            allowed.add("-" + body.lstrip("-") + "%")

    # Mask ISO dates so month/day fragments are not treated as inventable figures.
    scanned = _ISO_DATE.sub("DATE", text)
    offenders: list[str] = []
    for match in _NUMBER_TOKEN.finditer(scanned):
        token = match.group(0)
        if token in allowed:
            continue
        # Comma-stripped variant.
        if token.replace(",", "") in allowed:
            continue
        offenders.append(token)
    return offenders


def assert_claims_traceable(brief: AnalystBrief, facts: AnalystFacts) -> None:
    """Raise AssertionError if any claim invents a number."""
    for claim in brief.claims:
        bad = untraceable_numbers(claim.text, facts)
        if bad:
            raise AssertionError(
                f"Claim invents number(s) {bad}: {claim.text!r} (keys={claim.fact_keys})"
            )
        for key in claim.fact_keys:
            if key not in facts.as_map():
                raise AssertionError(f"Claim cites unknown fact key {key!r}: {claim.text!r}")


def facts_table_rows(facts: AnalystFacts) -> tuple[tuple[str, str, str], ...]:
    """Rows for the on-page computed-facts table: key, label, display."""
    return tuple((item.key, item.label, item.display) for item in facts.items)


# ---------------------------------------------------------------------------
# Fact builders
# ---------------------------------------------------------------------------


def _period_facts(stats: PeriodStats, *, prefix: str) -> list[FactValue]:
    rows: list[FactValue] = [
        _fact(f"{prefix}.ticker", "Ticker", stats.ticker, stats.ticker),
        _fact(
            f"{prefix}.window_start",
            "Window start",
            stats.window_start.date().isoformat(),
            stats.window_start.date().isoformat(),
        ),
        _fact(
            f"{prefix}.window_end",
            "Window end",
            stats.window_end.date().isoformat(),
            stats.window_end.date().isoformat(),
        ),
        _fact(
            f"{prefix}.cumulative_return",
            "Cumulative return",
            stats.cumulative_return,
            _pct(stats.cumulative_return, signed=True),
        ),
        _fact(
            f"{prefix}.n_days",
            "Sessions",
            stats.n_days,
            str(stats.n_days),
        ),
    ]
    optional_float = [
        ("annualized_return", "Annualized return", True),
        ("annualized_volatility", "Annualized volatility", False),
        ("sharpe", "Sharpe", False),
        ("sortino", "Sortino", False),
        ("max_drawdown", "Max drawdown", False),
        ("value_at_risk_95", "95% historical VaR", False),
        ("parametric_var_95", "95% parametric VaR", False),
        ("cvar_95", "95% Conditional VaR", False),
        ("beta", "Beta", False),
        ("correlation", "Correlation", False),
        ("alpha", "Jensen alpha", True),
        ("excess_return", "Gap vs benchmark", True),
        ("benchmark_cumulative_return", "Benchmark cumulative return", True),
        ("up_capture", "Up capture", False),
        ("down_capture", "Down capture", False),
    ]
    for name, label, signed in optional_float:
        value = getattr(stats, name)
        if value is None:
            continue
        if name in {"sharpe", "sortino", "beta", "correlation"}:
            display = f"{value:.2f}"
        elif name in {"up_capture", "down_capture"}:
            display = _pct(value)
        else:
            display = _pct(value, signed=signed)
        rows.append(_fact(f"{prefix}.{name}", label, value, display))

    if stats.vol_label:
        rows.append(_fact(f"{prefix}.vol_label", "Volatility label", stats.vol_label, stats.vol_label))
    if stats.drawdown_days_to_trough is not None:
        rows.append(
            _fact(
                f"{prefix}.drawdown_days_to_trough",
                "Days peak to trough",
                stats.drawdown_days_to_trough,
                str(stats.drawdown_days_to_trough),
            )
        )
    if stats.drawdown_days_to_recovery is not None:
        rows.append(
            _fact(
                f"{prefix}.drawdown_days_to_recovery",
                "Days trough to recovery",
                stats.drawdown_days_to_recovery,
                str(stats.drawdown_days_to_recovery),
            )
        )
    if stats.longest_underwater_days is not None:
        rows.append(
            _fact(
                f"{prefix}.longest_underwater_days",
                "Longest underwater sessions",
                stats.longest_underwater_days,
                str(stats.longest_underwater_days),
            )
        )
    return rows


def _diversification_facts(stats: DiversificationStats) -> list[FactValue]:
    rows: list[FactValue] = []
    if stats.diversification_ratio is not None:
        rows.append(
            _fact(
                "diversification.ratio",
                "Diversification ratio",
                stats.diversification_ratio,
                f"{stats.diversification_ratio:.2f}",
            )
        )
    if stats.effective_n is not None:
        rows.append(
            _fact(
                "diversification.effective_n",
                "Effective N",
                stats.effective_n,
                f"{stats.effective_n:.2f}",
            )
        )
    for symbol, weight in stats.weights.items():
        rows.append(
            _fact(
                f"diversification.weight.{symbol}",
                f"{symbol} weight",
                weight,
                _pct(weight),
            )
        )
    for symbol, share in stats.risk_contribution.items():
        rows.append(
            _fact(
                f"diversification.risk.{symbol}",
                f"{symbol} risk contribution",
                share,
                _pct(share),
            )
        )
    return rows


def _cost_facts(report: PortfolioCostReport) -> list[FactValue]:
    return [
        _fact("cost.cost_bps", "Cost setting (bps)", report.cost_bps, f"{report.cost_bps:g}"),
        _fact(
            "cost.total_turnover",
            "Total turnover",
            report.total_turnover,
            f"{report.total_turnover:.2f}",
        ),
        _fact(
            "cost.total_cost_drag",
            "Fee drag",
            report.total_cost_drag,
            _pct(report.total_cost_drag),
        ),
        _fact(
            "cost.rebalance_count",
            "Reset count",
            report.rebalance_count,
            str(report.rebalance_count),
        ),
    ]


def _fact(key: str, label: str, value: float | int | str | None, display: str) -> FactValue:
    return FactValue(key=key, label=label, value=value, display=display)


def _pct(value: float, *, signed: bool = False) -> str:
    if signed:
        return f"{value:+.1%}"
    return f"{value:.1%}"


# ---------------------------------------------------------------------------
# Optional LLM transport (stdlib only)
# ---------------------------------------------------------------------------


def _chat_completion(
    *,
    api_key: str,
    base_url: str,
    model: str,
    system: str,
    user: str,
) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("LLM base URL must be an https:// endpoint")
    body = json.dumps(
        {
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    # Scheme checked above; stdlib urlopen is the only free transport used here.
    with urllib.request.urlopen(request, timeout=LLM_TIMEOUT_SECONDS) as response:  # nosec B310
        payload = json.loads(response.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"]


def _parse_llm_claims(content: str, facts: AnalystFacts) -> tuple[NarrativeClaim, ...]:
    data = json.loads(content)
    raw = data.get("claims") or []
    known = facts.as_map()
    claims: list[NarrativeClaim] = []
    for entry in raw[:LLM_MAX_CLAIMS]:
        if not isinstance(entry, Mapping):
            continue
        text = str(entry.get("text") or "").strip()
        topic = str(entry.get("topic") or "general").strip() or "general"
        keys = tuple(
            str(key)
            for key in (entry.get("fact_keys") or ())
            if str(key) in known
        )
        if not text:
            continue
        claims.append(NarrativeClaim(text=text, fact_keys=keys, topic=topic))
    return tuple(claims)


if __name__ == "__main__":
    # Smoke: offline briefing → facts → rule-based brief must be traceable.
    from dashboard import run_briefing

    briefing = run_briefing(
        tickers=["AAPL", "MSFT"],
        offline=True,
        preset="Max",
        portfolio=True,
    )
    pack = build_analyst_pack(
        focus_report=briefing.report,
        comparison=briefing.comparison,
        portfolio_report=briefing.portfolio_report,
        cost_report=briefing.cost_report,
        enable_llm=False,
    )
    assert_claims_traceable(pack.rule_based, pack.facts)
    print(f"facts={len(pack.facts.items)} claims={len(pack.rule_based.claims)}")
    print(pack.rule_based.claims[0].text)
    print(pack.rule_based.disclaimer)
