"""
app.py — the briefing a person actually uses.

The sidebar collects up to five symbols, a benchmark, and the chart
settings. The page collects the date range and which symbol is in
focus, then shows the numbers, the sentences those numbers produce, the
side-by-side table, and the charts. This file does not calculate
returns. Offline mode is an explicit choice and reads only the bundled
files.

Every control that changes the view is bound to a URL query parameter,
so the address bar always describes what is on screen and a copied link
reopens the same view.
"""

from __future__ import annotations

import html
from datetime import date, timedelta
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

import streamlit as st

from dashboard import (
    BENCHMARK_CHOICES,
    DEFAULT_BENCHMARK,
    DEFAULT_PRESET,
    DEFAULT_TICKER,
    MAX_SYMBOLS,
    run_briefing,
)
from data_loader import (
    CACHE_TTL_HOURS,
    OFFLINE_BENCHMARKS,
    MarketDataError,
    filter_offline_selection,
    offline_span,
    offline_symbols,
    search_symbols,
)
# Metric helpers and portfolio constants live in features.py in this repo
# (not the unrelated PyPI package of the same name).
from features import (
    DEFAULT_RISK_FREE,
    DEFAULT_COST_BPS,
    COST_BPS_MAX,
    COST_BPS_MIN,
    MA_LONG,
    MA_SHORT,
    MAX_MA_WINDOW,
    MIN_MA_WINDOW,
    MISSING,
    PORTFOLIO_LABEL,
    PRESETS,
    REBALANCE_MONTHLY,
    REBALANCE_NONE,
    REBALANCE_OPTIONS,
    REBALANCE_QUARTERLY,
    RISK_FREE_MAX,
    RISK_FREE_MIN,
    ROLLING_BETA_WINDOW,
    ROLLING_VOL_WINDOW,
    TABLE_FORMATS,
    AnalysisError,
    comparison_table,
    describe_calendar_years,
    format_cell,
    metric_help,
    methodology_notes,
    volatility_rule_text,
)
from plots import PLOTLY_CONFIG

# The option text is also the value a shared link carries (?data=Offline),
# because a bound widget writes what it displays.
LIVE = "Live"
OFFLINE = "Offline"
MODE_CAPTIONS = ("Yahoo Finance prices", "Bundled demo files, no network")
CUSTOM = "Custom"
RANGE_OPTIONS = (*PRESETS, CUSTOM)
LINE = "Line"
CANDLES = "Candlestick"
TABS = ("Compare", "Price", "Drawdown", "Risk", "Returns")
# In-page section nav. "All sections" keeps the full briefing visible.
SECTION_ALL = "All sections"
SECTION_OVERVIEW = "Overview"
SECTION_COMPARE = "Comparison"
SECTION_PORTFOLIO = "Portfolio"
SECTION_CHARTS = "Charts"
SECTION_NARRATIVE = "Narrative"
SECTIONS = (
    SECTION_ALL,
    SECTION_OVERVIEW,
    SECTION_COMPARE,
    SECTION_PORTFOLIO,
    SECTION_CHARTS,
    SECTION_NARRATIVE,
)
# Suggestions only. In live mode any Yahoo Finance ticker can be typed.
POPULAR = ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "XOM", "JNJ", "KO", "BRK-B")
PERCENT = 100.0
RISK_FREE_STEP = 0.25
DEFAULT_CUSTOM_DAYS = 365
EARLIEST_PICKABLE = date(1970, 1, 1)
BOUND = "query-params"
DATA_MODE_LOCK = "_data_mode_lock"
# Streamlit Community Cloud draws its own header bar, 3.75rem tall, over
# the top of the page. Content has to start below it or the first line
# (the small label above the title) is covered on the hosted app while
# looking fine locally, where the bar is transparent.
CLOUD_HEADER_REM = 3.75
TOP_PADDING_REM = CLOUD_HEADER_REM + 0.5
# Widget keys double as the query-parameter names in a shared link.
SELECTION_KEYS = ("symbols", "benchmark", "focus", "start", "end")

PAGE_CSS = """
<style>
    .block-container {padding-top: TOP_PADDING; padding-bottom: 3rem; max-width: 1180px;}
    h1 {font-weight: 650; letter-spacing: -0.03em; margin-bottom: 0.15rem; padding-top: 0.1rem;}
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e4e8ee;
        border-radius: 12px;
        padding: 0.75rem 0.9rem 0.55rem 0.9rem;
        box-shadow: 0 1px 2px rgba(28, 36, 48, 0.04);
        overflow: visible;
    }
    div[data-testid="stMetricLabel"],
    div[data-testid="stTooltipHoverTarget"],
    div[data-testid="stMetricLabel"] > div {
        overflow: visible !important;
    }
    div[data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
    .kicker {
        color: #1f4b99;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.2rem;
    }
    .status-card {
        background: #ffffff;
        border: 1px solid #e4e8ee;
        border-radius: 12px;
        padding: 0.7rem 0.95rem;
        margin: 0.4rem 0 1rem 0;
        color: #3c4756;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    .status-card .muted {color: #5c6b7a;}
    .summary-card {
        background: #ffffff;
        border-left: 4px solid #1f4b99;
        border-radius: 0 12px 12px 0;
        padding: 0.3rem 1rem 0.6rem 1rem;
        margin: 0.4rem 0 0.8rem 0;
    }
    .compare-wrap {
        background: #ffffff;
        border: 1px solid #e4e8ee;
        border-radius: 12px;
        overflow-x: auto;
        margin: 0.4rem 0 0.5rem 0;
    }
    table.compare {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.86rem;
        font-variant-numeric: tabular-nums;
    }
    table.compare th, table.compare td {
        padding: 0.5rem 0.7rem;
        text-align: right;
        white-space: nowrap;
        border: none;
        border-bottom: 1px solid #eef1f5;
    }
    table.compare thead th {
        color: #5c6b7a;
        font-weight: 600;
        font-size: 0.76rem;
        background: #f8fafc;
        white-space: normal;
        vertical-align: bottom;
    }
    table.compare th:first-child {text-align: left;}
    table.compare tbody tr:last-child th, table.compare tbody tr:last-child td {border-bottom: none;}
    table.compare tr.focus th, table.compare tr.focus td {background: #eef4ff;}
    table.compare tr.portfolio th, table.compare tr.portfolio td {background: #f4fbf6; font-weight: 600;}
    .snapshot-card {
        background: #ffffff;
        border: 1px solid #e4e8ee;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        margin: 0.35rem 0 0.9rem 0;
        color: #3c4756;
        font-size: 0.92rem;
        line-height: 1.55;
    }
    .snapshot-card .snap-title {font-weight: 650; color: #1c2430; margin-bottom: 0.2rem;}
    .snapshot-card .muted {color: #5c6b7a; font-size: 0.84rem;}
    .year-table {width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums;}
    .year-table th, .year-table td {padding: 0.35rem 0.55rem; border-bottom: 1px solid #eef1f5; text-align: right;}
    .year-table th:first-child, .year-table td:first-child {text-align: left;}
    div[data-testid="stTabs"] {margin-top: 0.4rem;}
    .callout {
        background: #fff8eb;
        border: 1px solid #f0d9a8;
        border-left: 4px solid #d97706;
        border-radius: 0 12px 12px 0;
        padding: 0.7rem 0.95rem;
        margin: 0.5rem 0 0.9rem 0;
        color: #3c4756;
        font-size: 0.92rem;
        line-height: 1.5;
    }
    .callout strong {color: #1c2430;}
    .error-card {
        background: #fff5f5;
        border: 1px solid #f0c4c4;
        border-left: 4px solid #b91c1c;
        border-radius: 0 12px 12px 0;
        padding: 0.7rem 0.95rem;
        margin: 0.5rem 0 0.9rem 0;
        color: #3c4756;
        font-size: 0.92rem;
        line-height: 1.5;
    }
    @media (max-width: 900px) {
        .block-container {padding-left: 1rem; padding-right: 1rem; padding-top: TOP_PADDING;}
        h1 {font-size: 1.55rem; line-height: 1.2;}
        div[data-testid="stMetric"] {padding: 0.55rem 0.65rem 0.4rem 0.65rem;}
        div[data-testid="stMetricValue"] {font-size: 1.05rem !important;}
        div[data-testid="stMetricLabel"] {font-size: 0.75rem !important;}
        table.compare {font-size: 0.78rem;}
        table.compare th, table.compare td {padding: 0.4rem 0.45rem;}
        .summary-card {padding: 0.2rem 0.7rem 0.45rem 0.7rem;}
        .status-card {font-size: 0.84rem;}
        section[data-testid="stSidebar"] {width: min(100vw, 22rem);}
        div[data-testid="stTabs"] button {padding-left: 0.55rem; padding-right: 0.55rem;}
    }
    @media (max-width: 640px) {
        .block-container {padding-left: 0.7rem; padding-right: 0.7rem;}
        h1 {font-size: 1.35rem;}
        .kicker {font-size: 0.7rem; letter-spacing: 0.1em;}
        div[data-testid="stHorizontalBlock"] {gap: 0.45rem;}
        .callout, .error-card {font-size: 0.86rem; padding: 0.6rem 0.75rem;}
    }
</style>
""".replace("TOP_PADDING", f"{TOP_PADDING_REM}rem")


# ---------------------------------------------------------------------------
# Sidebar: what to load
# ---------------------------------------------------------------------------


def _on_data_mode_change() -> None:
    """
    Live and offline offer different symbols and dates.

    Pin the chosen mode in session so a query-param remount cannot snap
    the radio back to Live on the first click (bind writes the URL a
    tick later). Symbol filtering for Offline runs in the normal sidebar
    path after the radio sticks — rewriting symbols here remounted
    sibling widgets and made the first Offline click look like a no-op.
    """
    mode = st.session_state.get("data")
    if mode in (LIVE, OFFLINE):
        st.session_state[DATA_MODE_LOCK] = mode
    # Focus is mode-specific; custom dates are clamped by the range picker.
    st.session_state.pop("focus", None)
    if mode != OFFLINE:
        st.session_state.pop("offline_drop_notes", None)
        st.session_state.pop("offline_intent_symbols", None)
        st.session_state.pop("offline_intent_benchmark", None)
        st.session_state.pop("offline_intent_captured", None)
        return
    # Capture intent before bound widgets drop unavailable options.
    requested = list(st.session_state.get("symbols") or [])
    requested.extend(_symbols_named_in_query())
    if requested:
        st.session_state.offline_intent_symbols = list(dict.fromkeys(
            [str(s).strip().upper() for s in requested if str(s).strip()]
        ))
    bench = (
        st.session_state.get("benchmark")
        or _benchmark_named_in_query()
        or DEFAULT_BENCHMARK
    )
    st.session_state.offline_intent_benchmark = str(bench).strip().upper()


def _capture_offline_intent() -> None:
    """
    Remember symbols and benchmark named in the URL before the offline
    widgets drop values they cannot hold. Bound widgets remove unknown
    options from the query string during the run; reading first keeps
    enough to name what was left out.
    """
    mode = st.query_params.get("data") or st.session_state.get("data")
    if mode != OFFLINE:
        return
    if st.session_state.get("offline_intent_captured"):
        return
    symbols = _symbols_named_in_query()
    bench = _benchmark_named_in_query()
    if symbols:
        st.session_state.offline_intent_symbols = symbols
    if bench:
        st.session_state.offline_intent_benchmark = bench
    st.session_state.offline_intent_captured = True


def _symbols_named_in_query() -> list[str]:
    """Symbols named in the query string, including ones offline will refuse."""
    values = list(st.query_params.get_all("symbols") or [])
    address = st.context.url or ""
    if address:
        values.extend(parse_qs(urlsplit(address).query).get("symbols", []))
    cleaned: list[str] = []
    for value in values:
        text = str(value).strip().upper()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def _benchmark_named_in_query() -> str | None:
    value = st.query_params.get("benchmark")
    if value:
        text = str(value).strip().upper()
        if text:
            return text
    address = st.context.url or ""
    if not address:
        return None
    values = parse_qs(urlsplit(address).query).get("benchmark", [])
    if not values:
        return None
    text = str(values[0]).strip().upper()
    return text or None


def _apply_offline_filter(symbols: list[str], benchmark: str) -> tuple[list[str], str, tuple[str, ...]]:
    """
    Drop symbols (and a benchmark) the offline demo cannot serve, and
    remember the notice for the main page. Uses the URL intent captured
    before widgets ran, so a link that names NVDA still produces a message.
    """
    requested = list(
        dict.fromkeys(
            [
                *[str(s).strip().upper() for s in symbols if str(s).strip()],
                *list(st.session_state.get("offline_intent_symbols") or []),
                *_symbols_named_in_query(),
            ]
        )
    )
    intent_bench = st.session_state.get("offline_intent_benchmark")
    kept, bench, notes = filter_offline_selection(
        requested, intent_bench or _benchmark_named_in_query() or benchmark
    )
    previous = tuple(st.session_state.get("offline_drop_notes") or ())
    st.session_state.offline_drop_notes = list(notes or previous)
    # An intentional empty multiselect must stay empty; only fill AAPL when
    # something was requested and nothing offline-available remained.
    widget_empty = not [str(s).strip() for s in symbols if str(s).strip()]
    if widget_empty and not (
        st.session_state.get("offline_intent_symbols") or _symbols_named_in_query()
    ):
        kept = []
    if list(st.session_state.get("symbols") or []) != kept:
        st.session_state.symbols = kept
    if kept and st.session_state.get("benchmark") != bench:
        st.session_state.benchmark = bench
    return kept, (bench if kept else benchmark), tuple(st.session_state.offline_drop_notes)


def _tidy_symbols() -> None:
    """Typed tickers become upper case, and a repeat is dropped."""
    typed = st.session_state.get("symbols") or []
    cleaned = [str(entry).strip().upper() for entry in typed if str(entry).strip()]
    st.session_state.symbols = list(dict.fromkeys(cleaned))


def _tidy_benchmark() -> None:
    typed = st.session_state.get("benchmark")
    if typed:
        st.session_state.benchmark = str(typed).strip().upper()


def _add_search_hit() -> None:
    label = st.session_state.get("search_choice") or ""
    symbol = label.split("—")[0].strip().upper()
    current = list(st.session_state.get("symbols") or [])
    if not symbol or symbol in current:
        return
    if len(current) >= MAX_SYMBOLS:
        st.session_state.search_error = (
            f"The comparison already has {MAX_SYMBOLS} symbols. Remove one, then add {symbol}."
        )
        return
    st.session_state.symbols = [*current, symbol]


def _mode() -> bool:
    # Restore a just-chosen mode before the bound radio hydrates from a URL
    # that may still omit ``data`` (Live is the default and is left out of
    # the share link). Without this lock the first Offline click can flash
    # back to Live until a second click writes the query string.
    locked = st.session_state.get(DATA_MODE_LOCK)
    if locked in (LIVE, OFFLINE) and st.session_state.get("data") != locked:
        st.session_state["data"] = locked
    mode = st.sidebar.radio(
        "Data",
        (LIVE, OFFLINE),
        captions=MODE_CAPTIONS,
        key="data",
        bind=BOUND,
        on_change=_on_data_mode_change,
        help="Live data is Yahoo Finance. The offline demo reads only the files shipped with this app.",
    )
    st.session_state[DATA_MODE_LOCK] = mode
    offline = mode == OFFLINE
    if offline:
        first, last = _offline_span()
        st.sidebar.info(
            f"Bundled files for {', '.join(offline_symbols())}, "
            f"{first:%d %b %Y} – {last:%d %b %Y}. This mode does not call Yahoo Finance."
        )
    return offline


def _symbol_picker(offline: bool) -> list[str]:
    if offline:
        options = list(offline_symbols())
        guidance = "The offline demo ships these four files."
        # Bound widgets drop values that are not in options. Align session
        # state first so the multiselect opens on what offline can show.
        current = [str(s).strip().upper() for s in (st.session_state.get("symbols") or []) if str(s).strip()]
        intent = list(st.session_state.get("offline_intent_symbols") or [])
        kept, _, _ = filter_offline_selection(
            current or intent or _symbols_named_in_query(),
            st.session_state.get("benchmark") or st.session_state.get("offline_intent_benchmark"),
        )
        if current or intent or _symbols_named_in_query():
            st.session_state.symbols = kept
    else:
        chosen = st.session_state.get("symbols") or []
        options = list(dict.fromkeys([*POPULAR, *chosen]))
        guidance = "Pick from the list, or type any Yahoo Finance ticker and press Enter."
    return st.sidebar.multiselect(
        "Symbols",
        options,
        default=[DEFAULT_TICKER],
        key="symbols",
        max_selections=MAX_SYMBOLS,
        accept_new_options=not offline,
        bind=BOUND,
        on_change=_tidy_symbols,
        placeholder="Add a ticker",
        help=f"Up to {MAX_SYMBOLS} symbols. {guidance}",
    )


def _benchmark_picker(offline: bool) -> str:
    if offline:
        options = list(OFFLINE_BENCHMARKS)
        current = (
            st.session_state.get("benchmark")
            or st.session_state.get("offline_intent_benchmark")
            or _benchmark_named_in_query()
            or DEFAULT_BENCHMARK
        )
        _, bench, _ = filter_offline_selection(st.session_state.get("symbols") or [], current)
        st.session_state.benchmark = bench
    else:
        chosen = st.session_state.get("benchmark")
        options = list(dict.fromkeys([*BENCHMARK_CHOICES, *([chosen] if chosen else [])]))
    picked = st.sidebar.selectbox(
        "Benchmark",
        options,
        index=options.index(DEFAULT_BENCHMARK) if DEFAULT_BENCHMARK in options else 0,
        key="benchmark",
        accept_new_options=not offline,
        bind=BOUND,
        on_change=_tidy_benchmark,
        help="The yardstick for the gap, beta, correlation and capture. In live mode you can type another ticker.",
    )
    benchmark = picked or DEFAULT_BENCHMARK
    st.sidebar.caption(BENCHMARK_CHOICES.get(benchmark, "Custom benchmark"))
    return benchmark


def _symbol_search() -> None:
    with st.sidebar.form("finder"):
        query = st.text_input("Find a company or fund", placeholder="Apple, Microsoft, S&P 500")
        search = st.form_submit_button("Search")
    if search:
        try:
            st.session_state.search_hits = search_symbols(query)
            st.session_state.search_error = ""
        except MarketDataError as exc:
            st.session_state.search_hits = []
            st.session_state.search_error = str(exc)
    error = st.session_state.get("search_error") or ""
    hits = st.session_state.get("search_hits") or []
    if hits:
        labels = [f"{hit.symbol} — {hit.name}" for hit in hits]
        st.sidebar.selectbox("Matches", labels, key="search_choice")
        st.sidebar.button("Add to comparison", on_click=_add_search_hit)
    if error:
        st.sidebar.warning(error)


def _settings(symbols: list[str]) -> dict:
    with st.sidebar.expander("Chart and risk settings"):
        short = st.number_input(
            "Short moving average (days)",
            min_value=MIN_MA_WINDOW,
            max_value=MAX_MA_WINDOW - 1,
            value=MA_SHORT,
            step=1,
            key="ma_short",
            bind=BOUND,
        )
        long = st.number_input(
            "Long moving average (days)",
            min_value=MIN_MA_WINDOW + 1,
            max_value=MAX_MA_WINDOW,
            value=MA_LONG,
            step=1,
            key="ma_long",
            bind=BOUND,
        )
        rate = st.number_input(
            "Risk-free rate (% a year)",
            min_value=RISK_FREE_MIN * PERCENT,
            max_value=RISK_FREE_MAX * PERCENT,
            value=DEFAULT_RISK_FREE * PERCENT,
            step=RISK_FREE_STEP,
            format="%.2f",
            key="rf",
            bind=BOUND,
            help="Used by the Sharpe and Sortino ratios. Zero is the default so the assumption is visible, not guessed.",
        )
    portfolio = _portfolio_controls(symbols)
    return {
        "ma_short": int(short),
        "ma_long": int(long),
        "risk_free": float(rate) / PERCENT,
        **portfolio,
    }


def _portfolio_controls(symbols: list[str]) -> dict:
    """Weights and rebalance. Equal weight is the default when portfolio mode is on."""
    usable = [symbol for symbol in symbols if symbol]
    enabled = st.sidebar.checkbox(
        "Portfolio mode",
        value=False,
        key="portfolio",
        bind=BOUND,
        help="Mix the selected symbols into one portfolio line and table row versus the benchmark.",
    )
    if not enabled:
        return {
            "portfolio": False,
            "portfolio_blocked": False,
            "weights": None,
            "rebalance": REBALANCE_NONE,
            "cost_bps": DEFAULT_COST_BPS,
        }
    if len(usable) < 2:
        named = ", ".join(usable) if usable else "none selected"
        st.sidebar.warning(
            f"Portfolio mode is on, but a mix needs **at least two symbols**. "
            f"Right now: {named}. Add another ticker (for example MSFT next to AAPL), "
            "then weights, rebalance, the green portfolio line, and construction charts appear."
        )
        return {
            "portfolio": False,
            "portfolio_blocked": True,
            "weights": None,
            "rebalance": REBALANCE_NONE,
            "cost_bps": DEFAULT_COST_BPS,
        }

    labels = {
        REBALANCE_NONE: "None (buy and hold)",
        REBALANCE_MONTHLY: "Monthly",
        REBALANCE_QUARTERLY: "Quarterly",
    }
    rebalance = st.sidebar.selectbox(
        "Rebalance",
        REBALANCE_OPTIONS,
        index=0,
        format_func=lambda key: labels[key],
        key="rebalance",
        bind=BOUND,
        help="Monthly or quarterly resets the mix to the target weights on the first shared session of each period.",
    )
    equal = 100.0 / len(usable)
    st.sidebar.caption(f"Weights as percent of the portfolio. Equal weight is {equal:.1f}% each.")
    weights: dict[str, float] = {}
    for symbol in usable:
        percent = st.sidebar.number_input(
            f"{symbol} weight (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(round(equal, 2)),
            step=1.0,
            key=f"w_{symbol}",
            help="Zero leaves this symbol out of the mix. Weights are scaled to 100% if they do not already sum to it.",
        )
        weights[symbol] = float(percent) / PERCENT
    cost = st.sidebar.number_input(
        "Transaction cost (bps)",
        min_value=float(COST_BPS_MIN),
        max_value=float(COST_BPS_MAX),
        value=float(DEFAULT_COST_BPS),
        step=1.0,
        key="cost_bps",
        bind=BOUND,
        help="Charged on turnover at inception and at each rebalance. 10 bps = 0.10% of traded notional.",
    )
    return {
        "portfolio": True,
        "portfolio_blocked": False,
        "weights": weights,
        "rebalance": rebalance,
        "cost_bps": float(cost),
    }


def _sidebar() -> dict:
    st.sidebar.header("Briefing")
    _capture_offline_intent()
    offline = _mode()
    symbols = _symbol_picker(offline)
    benchmark = _benchmark_picker(offline)
    notes: tuple[str, ...] = ()
    if offline:
        symbols, benchmark, notes = _apply_offline_filter(symbols, benchmark)
    else:
        st.session_state.pop("offline_drop_notes", None)
        st.session_state.pop("offline_intent_symbols", None)
        st.session_state.pop("offline_intent_benchmark", None)
        st.session_state.pop("offline_intent_captured", None)
    if not offline:
        _symbol_search()
    settings = _settings(symbols)
    refresh = False
    if not offline:
        refresh = st.sidebar.button("Refresh prices")
        st.sidebar.caption(
            f"A download is reused for {CACHE_TTL_HOURS} hours. "
            "Refresh fetches Yahoo Finance again."
        )
    st.sidebar.caption(
        "Zooming a chart, or pressing a chart's range buttons, changes the view only. "
        "The metrics use the range selected above the briefing."
    )
    return {
        "offline": offline,
        "symbols": symbols,
        "benchmark": benchmark,
        "refresh": refresh,
        "offline_notes": notes,
        **settings,
    }


# ---------------------------------------------------------------------------
# Page: range and focus
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _offline_span() -> tuple[date, date]:
    first, last = offline_span()
    return first.date(), last.date()


def _range_picker(offline: bool) -> dict:
    preset = st.segmented_control(
        "Date range",
        RANGE_OPTIONS,
        default=DEFAULT_PRESET,
        required=True,
        key="range",
        bind=BOUND,
        help="Presets count back from the latest bar. Custom takes any two dates.",
    )
    if preset != CUSTOM:
        return {"preset": preset, "start": None, "end": None}
    if offline:
        earliest, latest = _offline_span()
        default_start = earliest
    else:
        earliest, latest = EARLIEST_PICKABLE, date.today()
        default_start = latest - timedelta(days=DEFAULT_CUSTOM_DAYS)
    left, right, _ = st.columns((1, 1, 2))
    start = left.date_input(
        "Start", value=default_start, min_value=earliest, max_value=latest, key="start", bind=BOUND
    )
    end = right.date_input(
        "End", value=latest, min_value=earliest, max_value=latest, key="end", bind=BOUND
    )
    return {"preset": None, "start": start, "end": end}


def _focus_picker(symbols: list[str], benchmark: str) -> str | None:
    """Which symbol the cards, sentences and single-symbol charts describe."""
    candidates = [symbol for symbol in symbols if symbol != benchmark]
    if st.session_state.get("focus") not in candidates:
        st.session_state.pop("focus", None)
    if len(candidates) < 2:
        return candidates[0] if candidates else None
    return st.segmented_control(
        "In focus",
        candidates,
        default=candidates[0],
        required=True,
        key="focus",
        bind=BOUND,
        help="The cards, the sentences and the single-symbol charts describe this symbol. "
        "The table and the Compare chart cover all of them.",
    )


def _title(symbols: list[str], benchmark: str) -> str:
    names = [symbol for symbol in symbols if symbol != benchmark] or [DEFAULT_TICKER]
    return f"{', '.join(names)} vs {benchmark}"


# ---------------------------------------------------------------------------
# Page: results
# ---------------------------------------------------------------------------


def _warnings(briefing, offline_notes: tuple[str, ...] = ()) -> None:
    for note in offline_notes:
        st.info(note)
    for note in briefing.notes:
        st.info(note)
    for symbol, reason in briefing.failed.items():
        st.warning(f"{symbol} was left out. {reason}")
    if briefing.benchmark_error:
        st.warning(f"Benchmark unavailable. {briefing.benchmark_error}")
    if briefing.stale:
        st.warning(
            f"Yahoo Finance did not answer for {', '.join(briefing.stale)}. "
            f"The page is showing the previous download, which is older than {CACHE_TTL_HOURS} hours. "
            "The fetch time below is when that download was made."
        )


def _fetched_text(history) -> str:
    if history.fetched_at is None:
        return "bundled file, no live refresh"
    stamp = history.fetched_at
    stamp = stamp.tz_convert("UTC") if stamp.tzinfo else stamp.tz_localize("UTC")
    if history.stale:
        freshness = "older download, Yahoo did not answer"
    elif history.from_cache:
        freshness = "cached download"
    else:
        freshness = "just downloaded"
    return f"fetched {stamp:%Y-%m-%d %H:%M} UTC · {freshness}"


def _status(briefing) -> None:
    loaded = list(briefing.histories.values())
    if briefing.benchmark is not None:
        loaded.append(briefing.benchmark)
    rows = "".join(
        f"<strong>{html.escape(history.ticker)}.</strong> {html.escape(history.source)} "
        f"· latest bar {history.latest_bar.date().isoformat()} "
        f'· <span class="muted">{_fetched_text(history)}</span><br>'
        for history in loaded
    )
    window = briefing.report.stats
    bench = briefing.benchmark.ticker if briefing.benchmark is not None else "unavailable"
    st.markdown(
        f"""
        <div class="status-card">
        {rows}
        <strong>This briefing.</strong> {html.escape(briefing.focus)} from
        {window.window_start.date().isoformat()} to {window.window_end.date().isoformat()}
        · {window.n_days} trading days · benchmark {html.escape(bench)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _number(value: float | None, pattern: str) -> str:
    return MISSING if value is None else pattern.format(value)


def _metrics(stats) -> None:
    help_text = metric_help()
    bench = stats.benchmark or "benchmark"
    gap = MISSING if stats.excess_return is None else f"{stats.excess_return * PERCENT:+.1f} pp"
    label = (stats.vol_label or "not ranked").capitalize()
    top = st.columns(5)
    top[0].metric("Cumulative return", f"{stats.cumulative_return:+.1%}", help=help_text["cumulative_return"])
    top[1].metric("Annualized return", _number(stats.annualized_return, "{:+.1%}"), help=help_text["annualized_return"])
    top[2].metric("Annualized volatility", _number(stats.annualized_volatility, "{:.1%}"), help=help_text["annualized_volatility"])
    top[3].metric("Maximum drawdown", _number(stats.max_drawdown, "{:.1%}"), help=help_text["max_drawdown"])
    top[4].metric(f"Vs {bench}", gap, help=help_text["excess_return"])
    bottom = st.columns(5)
    bottom[0].metric("Sharpe ratio", _number(stats.sharpe, "{:.2f}"), help=help_text["sharpe"])
    bottom[1].metric("Sortino ratio", _number(stats.sortino, "{:.2f}"), help=help_text["sortino"])
    bottom[2].metric(f"Beta vs {bench}", _number(stats.beta, "{:.2f}"), help=help_text["beta"])
    bottom[3].metric(f"Correlation vs {bench}", _number(stats.correlation, "{:.2f}"), help=help_text["correlation"])
    bottom[4].metric("95% hist. VaR", _number(stats.value_at_risk_95, "{:.1%}"), help=help_text["value_at_risk_95"])
    extra = st.columns(5)
    # Required keys (not .get(..., "")) so Streamlit always gets a real help
    # string and draws the tooltip icon — empty help hides the icon.
    # Keep labels short: long "95% param. VaR" labels were clipping the
    # help icon on Streamlit Cloud's metric chrome.
    extra[0].metric(
        "Parametric VaR",
        _number(stats.parametric_var_95, "{:.1%}"),
        help=help_text["parametric_var_95"],
    )
    extra[1].metric(
        "CVaR / ES",
        _number(stats.cvar_95, "{:.1%}"),
        help=help_text["cvar_95"],
    )
    extra[2].metric(
        f"Alpha vs {bench}",
        _number(stats.alpha, "{:+.1%}"),
        help=help_text["alpha"],
    )
    extra[3].metric(
        "Peak to trough",
        "—" if stats.drawdown_days_to_trough is None else str(stats.drawdown_days_to_trough),
        help=help_text["drawdown_days_to_trough"],
    )
    extra[4].metric(
        "Trough to recovery",
        "—" if stats.drawdown_days_to_recovery is None else str(stats.drawdown_days_to_recovery),
        help=help_text["drawdown_days_to_recovery"],
    )
    st.caption(
        f"Volatility label: **{label}**. {volatility_rule_text()} "
        f"Longest underwater: {stats.longest_underwater_days if stats.longest_underwater_days is not None else '—'} sessions."
    )


def _card(sentences) -> None:
    blocks = "".join(f"<p>{html.escape(sentence)}</p>" for sentence in sentences)
    st.markdown(f'<div class="summary-card">{blocks}</div>', unsafe_allow_html=True)


def _snapshot(briefing) -> None:
    snap = briefing.snapshot
    if snap is None:
        return
    st.subheader(f"Company snapshot · {briefing.focus}")
    title = snap.name or snap.ticker
    bits = []
    if snap.sector:
        bits.append(html.escape(snap.sector))
    if snap.industry:
        bits.append(html.escape(snap.industry))
    facts = []
    if snap.market_cap is not None:
        facts.append(f"<strong>Market cap</strong> {_fmt_cap_html(snap.market_cap)}")
    if snap.pe_ratio is not None:
        facts.append(f"<strong>P/E</strong> {snap.pe_ratio:.1f}")
    if snap.dividend_yield is not None:
        facts.append(f"<strong>Dividend yield</strong> {snap.dividend_yield:.2%}")
    if snap.fifty_two_week_low is not None and snap.fifty_two_week_high is not None:
        facts.append(
            f"<strong>52-week range</strong> {snap.fifty_two_week_low:.2f} – {snap.fifty_two_week_high:.2f}"
        )
    missing = []
    if snap.market_cap is None:
        missing.append("market cap")
    if snap.pe_ratio is None:
        missing.append("P/E")
    if snap.dividend_yield is None:
        missing.append("dividend yield")
    body = " · ".join(facts) if facts else "No valuation fields were available for this symbol."
    meta = " · ".join(bits)
    note = html.escape(snap.source)
    if missing:
        note += f" · Missing: {', '.join(missing)}"
    st.markdown(
        f'<div class="snapshot-card"><div class="snap-title">{html.escape(title)}</div>'
        f'<div>{meta}</div><div>{body}</div>'
        f'<div class="muted">{note}</div></div>',
        unsafe_allow_html=True,
    )


def _fmt_cap_html(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1e12:
        return f"{value / 1e12:.2f}T"
    if abs_value >= 1e9:
        return f"{value / 1e9:.2f}B"
    if abs_value >= 1e6:
        return f"{value / 1e6:.2f}M"
    return f"{value:,.0f}"


def _calendar_years(briefing) -> None:
    years = briefing.report.charts.calendar_years
    if years is None or len(years) == 0:
        return
    st.subheader(f"Calendar-year returns · {briefing.focus}")
    rows = "".join(
        f"<tr><th scope='row'>{int(year)}</th><td>{value:+.1%}</td></tr>"
        for year, value in years.items()
    )
    st.markdown(
        f'<div class="compare-wrap"><table class="year-table">'
        f"<thead><tr><th>Year</th><th>Return</th></tr></thead><tbody>{rows}</tbody></table></div>",
        unsafe_allow_html=True,
    )
    sentence = describe_calendar_years(years, ticker=briefing.focus)
    if sentence:
        st.caption(sentence)


def _cost_and_diversification(briefing) -> None:
    """Portfolio trading costs, diversification stats, and allocation charts."""
    cost = briefing.cost_report
    divers = briefing.comparison.diversification
    figures = briefing.figures
    allocation = figures.get("allocation")
    risk_bars = figures.get("risk_contribution")
    if cost is None and divers is None and allocation is None and risk_bars is None:
        return
    st.subheader("Portfolio construction details")
    if cost is not None:
        cols = st.columns(4)
        cols[0].metric("Cost setting (bps)", f"{cost.cost_bps:g}")
        cols[1].metric("Total turnover", f"{cost.total_turnover:.2f}")
        cols[2].metric("Fee drag (sum)", f"{cost.total_cost_drag:.2%}")
        cols[3].metric("Resets (incl. open)", str(cost.rebalance_count))
        st.caption(
            "Turnover is half the L1 weight distance at each reset; inception counts as 1.0. "
            "Fee drag is the sum of turnover x bps on traded notional."
        )
    if divers is not None:
        dcols = st.columns(3)
        ratio = "—" if divers.diversification_ratio is None else f"{divers.diversification_ratio:.2f}"
        dcols[0].metric("Diversification ratio", ratio)
        dcols[1].metric("Effective N", f"{divers.effective_n:.2f}")
        ranked = sorted(divers.risk_contribution.items(), key=lambda item: -item[1])
        listing = ", ".join(f"{name} {share:.0%}" for name, share in ranked)
        dcols[2].metric("Top risk contribution", ranked[0][0] if ranked else "—")
        st.caption(f"Risk contribution by weight: {listing}.")
    # Full-width stacked charts (not nested columns) so Plotly gets a real
    # width on first paint — side-by-side columns were blank on Streamlit Cloud.
    if allocation is not None:
        st.markdown("**Target weights**")
        _show(allocation, "Allocation chart needs portfolio weights.")
        st.caption("Target weights after renormalization. Zero-weight symbols are left out.")
    if risk_bars is not None:
        st.markdown("**Risk contribution**")
        _show(risk_bars, "Risk contribution needs a diversified mix.")
        st.caption("Each bar is that holding's share of portfolio variance.")


def _section_nav() -> str:
    """Compact jump control for the main briefing sections."""
    chosen = st.segmented_control(
        "On this page",
        SECTIONS,
        default=SECTION_ALL,
        required=True,
        key="section",
        bind=BOUND,
        help="Jump to a block of the briefing. All sections keeps the full page visible.",
    )
    return chosen or SECTION_ALL


def _show_section(current: str, name: str) -> bool:
    return current in (SECTION_ALL, name)


def _portfolio_blocked_callout(symbols: list[str]) -> None:
    named = ", ".join(symbols) if symbols else "none selected yet"
    st.markdown(
        f'<div class="callout"><strong>Portfolio mode needs two symbols.</strong> '
        f"It is on, but the mix is waiting. Selected now: {html.escape(named)}. "
        "Add another ticker in the sidebar (for example MSFT next to AAPL). "
        "Then target weights, rebalance, the green portfolio line, allocation charts, "
        "and construction metrics appear.</div>",
        unsafe_allow_html=True,
    )


def _summary(briefing) -> None:
    stats = briefing.report.stats
    st.subheader(f"What the numbers say about {briefing.focus}")
    _card(briefing.report.sentences)
    if briefing.portfolio_report is not None:
        st.subheader("What the numbers say about the portfolio")
        _card(briefing.portfolio_report.sentences)
    if briefing.comparison.sentences:
        st.subheader("Across the symbols")
        _card(briefing.comparison.sentences)
    with st.expander("How every figure and label is defined"):
        for note in methodology_notes(stats.ma_short_window, stats.ma_long_window):
            st.markdown(f"- {note}")
        st.markdown("Full formulas and limitations: see `docs/METHODOLOGY.md` in the repository.")


def _table(briefing) -> None:
    st.subheader("Side by side")
    table = comparison_table(briefing.comparison)
    st.markdown(_table_html(table, focus=briefing.focus), unsafe_allow_html=True)
    st.caption(
        "Each row uses that symbol's own sessions inside the selected range. Beta, correlation, capture "
        "and the gap are measured against the benchmark on shared sessions. The blue shaded row is the "
        "symbol in focus; the green shaded row is the portfolio when Portfolio mode is on."
    )


def _table_html(table, *, focus: str) -> str:
    """
    The comparison as a plain HTML table. Every cell goes through the
    same formats the downloaded summary uses, so the two always agree.
    """
    columns = [column for column in TABLE_FORMATS if column in table.columns]
    head = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = []
    for symbol, row in table.iterrows():
        name = str(symbol)
        if name == focus:
            css = "focus"
        elif name == PORTFOLIO_LABEL:
            css = "portfolio"
        else:
            css = ""
        cells = "".join(f"<td>{html.escape(format_cell(column, row[column]))}</td>" for column in columns)
        body.append(f'<tr class="{css}"><th scope="row">{html.escape(name)}</th>{cells}</tr>')
    return (
        '<div class="compare-wrap"><table class="compare">'
        f"<thead><tr><th>Symbol</th>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )


def _share_query(request: dict) -> str:
    """The query string for this view. A setting left at its default is omitted, as the address bar does."""
    pairs: list[tuple[str, str]] = []
    if request["offline"]:
        pairs.append(("data", OFFLINE))
    if list(request["symbols"]) != [DEFAULT_TICKER]:
        pairs.extend(("symbols", symbol) for symbol in request["symbols"])
    if request["benchmark"] != DEFAULT_BENCHMARK:
        pairs.append(("benchmark", request["benchmark"]))
    if request["preset"] is None:
        pairs.extend([("range", CUSTOM), ("start", request["start"].isoformat()), ("end", request["end"].isoformat())])
    elif request["preset"] != DEFAULT_PRESET:
        pairs.append(("range", request["preset"]))
    compared = [symbol for symbol in request["symbols"] if symbol != request["benchmark"]]
    if request["focus"] and compared and request["focus"] != compared[0]:
        pairs.append(("focus", request["focus"]))
    if request["ma_short"] != MA_SHORT:
        pairs.append(("ma_short", str(request["ma_short"])))
    if request["ma_long"] != MA_LONG:
        pairs.append(("ma_long", str(request["ma_long"])))
    if request["risk_free"] != DEFAULT_RISK_FREE:
        pairs.append(("rf", f"{request['risk_free'] * PERCENT:g}"))
    if request.get("tab", TABS[0]) != TABS[0]:
        pairs.append(("tab", request["tab"]))
    if request.get("chart", LINE) != LINE:
        pairs.append(("chart", request["chart"]))
    if request.get("section", SECTION_ALL) != SECTION_ALL:
        pairs.append(("section", request["section"]))
    if request.get("portfolio"):
        pairs.append(("portfolio", "True"))
        if request.get("rebalance") and request["rebalance"] != REBALANCE_NONE:
            pairs.append(("rebalance", request["rebalance"]))
    return urlencode(pairs)


def _share_link(request: dict) -> str:
    query = _share_query(request)
    address = st.context.url
    if not address:
        return f"?{query}" if query else "(default view)"
    parts = urlsplit(address)
    return urlunsplit((parts.scheme, parts.netloc, parts.path or "/", query, ""))


def _take_away(briefing, request: dict) -> None:
    left, mid, right = st.columns((1, 1, 2))
    with left:
        st.download_button(
            "Download analysis summary",
            data=briefing.markdown,
            file_name=f"{'_'.join(briefing.reports).lower()}_briefing.md",
            mime="text/markdown",
        )
    with mid:
        if briefing.export_csv:
            st.download_button(
                "Download price/return CSV",
                data=briefing.export_csv,
                file_name=f"{'_'.join(briefing.reports).lower()}_prices.csv",
                mime="text/csv",
                help="Aligned adjusted closes and daily returns for every symbol, the portfolio when present, and the benchmark.",
            )
    with right:
        with st.expander("Link to this view"):
            st.code(_share_link(request), language=None)
            st.caption(
                "The address bar updates the same way as you change symbols, benchmark, range, focus, "
                "settings and the open chart tab. Opening the link loads the same view with the prices "
                "available at that time."
            )



def _show(figure, missing: str) -> None:
    if figure is None:
        st.info(missing)
    else:
        st.plotly_chart(figure, width="stretch", theme=None, config=PLOTLY_CONFIG)


def _charts(briefing) -> None:
    figures = briefing.figures
    stats = briefing.report.stats
    compare_tab, price_tab, drawdown_tab, risk_tab, returns_tab = st.tabs(TABS, key="tab", bind=BOUND)
    with compare_tab:
        _show(figures["comparison"], "The comparison chart needs two sessions that every symbol shares.")
        st.caption(
            "Every line starts at 100 on the first session all of them share, so the height of a line is "
            "its growth since that day. The benchmark is dashed. The portfolio, when on, is the solid green "
            "line. The symbol in focus is the thicker line among the individual symbols."
        )
    with price_tab:
        style = st.segmented_control(
            "Chart type", (LINE, CANDLES), default=LINE, required=True, key="chart", bind=BOUND
        )
        if style == CANDLES:
            _show(figures["candles"], "No bars to draw.")
            st.caption(
                "Open, high, low and close are on the adjusted-close scale, so the candles and the "
                f"{stats.ma_short_window}- and {stats.ma_long_window}-day averages line up. Volume is underneath."
            )
        else:
            _show(figures["price"], "No prices to draw.")
            st.caption(
                "Hover a date for the adjusted close and both averages. The slider under the chart and the "
                "buttons above it zoom the view. Change the average windows under Chart and risk settings."
            )
    with drawdown_tab:
        _show(figures["drawdown"], "No drawdown to draw.")
        st.caption(
            "Each point is that day's adjusted close divided by the highest close so far in the window, "
            "minus one. The dashed line is the benchmark's own drawdown over the same range."
        )
    with risk_tab:
        _show(figures["volatility"], "No volatility to draw.")
        st.caption(
            f"The line is the trailing {ROLLING_VOL_WINDOW}-day annualized volatility. "
            "The dashed line is the volatility of the whole selected window."
        )
        _show(
            figures["relationship"],
            f"Rolling beta and correlation need a benchmark and at least {ROLLING_BETA_WINDOW} shared daily returns.",
        )
        st.caption(
            f"Beta and correlation over the trailing {ROLLING_BETA_WINDOW} shared daily returns. "
            "A beta near 1 moves like the benchmark. A correlation near 0 moves independently of it."
        )
        _show(figures.get("rolling_ratios"), "Rolling Sharpe/Sortino need a longer history.")
        st.caption(
            f"Trailing {ROLLING_BETA_WINDOW}-day annualized Sharpe and Sortino for the symbol in focus."
        )
        _show(figures.get("correlation"), "A correlation heatmap needs at least two symbols with shared returns.")
        st.caption("Pairwise Pearson correlations of daily simple returns across the selected symbols.")
    with returns_tab:
        _show(figures["histogram"], "No daily returns to draw.")
        st.caption(
            "How often each size of daily move happened. The benchmark is the grey shape behind, "
            "in the same bins. The dashed line is the mean daily return."
        )
        _show(figures["heatmap"], "No monthly returns to draw.")
        st.caption(
            "Monthly returns inside the selected range. The first and last months can be partial. "
            "Compounding every cell gives the cumulative return at the top of the page."
        )


def _brief(request: dict):
    names = ", ".join([*request["symbols"], request["benchmark"]])
    spinner = (
        f"Reading the bundled files for {names}…"
        if request["offline"]
        else f"Loading {names} from Yahoo Finance…"
    )
    with st.spinner(spinner):
        return run_briefing(
            tickers=request["symbols"],
            benchmark=request["benchmark"],
            focus=request["focus"],
            start=request["start"],
            end=request["end"],
            preset=request["preset"],
            offline=request["offline"],
            refresh=request["refresh"],
            ma_short=request["ma_short"],
            ma_long=request["ma_long"],
            risk_free=request["risk_free"],
            weights=request.get("weights"),
            rebalance=request.get("rebalance") or REBALANCE_NONE,
            portfolio=bool(request.get("portfolio")),
            cost_bps=float(request.get("cost_bps") or DEFAULT_COST_BPS),
        )


def main() -> None:
    st.set_page_config(page_title="Market briefing", layout="wide", initial_sidebar_state="expanded")
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    request = _sidebar()
    kicker = "Market briefing · offline demo" if request["offline"] else "Market briefing"
    st.markdown(f'<div class="kicker">{kicker}</div>', unsafe_allow_html=True)
    # The headline is filled in again after loading, so it names what was briefed.
    headline = st.empty()
    headline.title(_title(request["symbols"], request["benchmark"]))
    st.write(
        f"Pick up to {MAX_SYMBOLS} symbols and a benchmark. Read each one's return, risk and drawdown over "
        "a window you choose, and see how closely it moved with the benchmark on the dates both traded."
    )
    request.update(_range_picker(request["offline"]))
    section = _section_nav()
    request["section"] = section
    offline_notes = tuple(request.get("offline_notes") or ())
    if not request["symbols"]:
        for note in offline_notes:
            st.info(note)
        st.info("Add at least one symbol in the sidebar to build a briefing.")
        return
    if request.get("portfolio_blocked"):
        _portfolio_blocked_callout(request["symbols"])
    # Prefer a focus the user already chose; the dashboard falls back if
    # that symbol failed to load. The picker below only offers loaded ones.
    request["focus"] = st.session_state.get("focus")

    try:
        briefing = _brief(request)
    except MarketDataError as exc:
        for note in offline_notes:
            st.info(note)
        st.error(str(exc))
        if not request["offline"]:
            st.info("The offline demo is available under Data in the sidebar. It does not use the network.")
        return
    except AnalysisError as exc:
        for note in offline_notes:
            st.info(note)
        st.error(str(exc))
        return

    loaded = list(briefing.reports)
    chosen = _focus_picker(loaded, request["benchmark"])
    if chosen and chosen != briefing.focus:
        # The picker only lists symbols that loaded; rebuild so the cards,
        # sentences and single-symbol charts match the new focus.
        briefing = _brief({**request, "symbols": loaded, "focus": chosen})
        loaded = list(briefing.reports)
    request["focus"] = chosen or briefing.focus
    request["symbols"] = loaded

    headline.title(_title(loaded, request["benchmark"]))
    _warnings(briefing, offline_notes=offline_notes)
    if _show_section(section, SECTION_OVERVIEW):
        _status(briefing)
        _snapshot(briefing)
        _metrics(briefing.report.stats)
    if _show_section(section, SECTION_COMPARE):
        _table(briefing)
    if _show_section(section, SECTION_PORTFOLIO):
        _cost_and_diversification(briefing)
    if _show_section(section, SECTION_CHARTS):
        _charts(briefing)
    if _show_section(section, SECTION_NARRATIVE):
        _calendar_years(briefing)
        _summary(briefing)
    request["tab"] = st.session_state.get("tab") or TABS[0]
    request["chart"] = st.session_state.get("chart") or LINE
    request["section"] = st.session_state.get("section") or SECTION_ALL
    _take_away(briefing, request)


main()
