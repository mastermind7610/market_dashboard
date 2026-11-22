"""
app.py — the briefing a person actually uses.

The sidebar collects a symbol, a benchmark, and a date range. The page
shows the numbers, the sentences those numbers produce, and the charts.
This file does not calculate returns. Offline mode is an explicit choice
and reads only the bundled files.
"""

from __future__ import annotations

import html
from datetime import date, timedelta

import streamlit as st

from dashboard import DEFAULT_BENCHMARK, DEFAULT_TICKER, run_briefing
from data_loader import (
    CACHE_TTL_HOURS,
    MarketDataError,
    search_symbols,
)
from features import (
    PRESETS,
    AnalysisError,
    metric_help,
    methodology_notes,
    volatility_rule_text,
)
from plots import PLOTLY_CONFIG

EXAMPLES = ("AAPL", "MSFT", "NVDA", "JPM", "XOM")
RANGE_OPTIONS = (*PRESETS, "Custom")

PAGE_CSS = """
<style>
    .block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1180px;}
    h1 {font-weight: 650; letter-spacing: -0.03em; margin-bottom: 0.15rem;}
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e4e8ee;
        border-radius: 12px;
        padding: 0.75rem 0.9rem 0.55rem 0.9rem;
        box-shadow: 0 1px 2px rgba(28, 36, 48, 0.04);
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
        font-size: 0.92rem;
        line-height: 1.45;
    }
    .summary-card {
        background: #ffffff;
        border-left: 4px solid #1f4b99;
        border-radius: 0 12px 12px 0;
        padding: 0.3rem 1rem 0.6rem 1rem;
        margin: 0.4rem 0 0.8rem 0;
    }
    div[data-testid="stTabs"] {margin-top: 0.4rem;}
</style>
"""


def _use_example() -> None:
    choice = st.session_state.get("example_pick")
    if choice:
        st.session_state.symbol = choice


def _use_search_hit() -> None:
    label = st.session_state.get("search_choice") or ""
    symbol = label.split("—")[0].strip().upper()
    if symbol:
        st.session_state.symbol = symbol


def _sidebar() -> dict:
    st.sidebar.header("Briefing")
    mode = st.sidebar.radio(
        "Data",
        ("Live market data", "Offline demo"),
        help="Live data is Yahoo Finance. The offline demo reads only the files shipped with this app.",
    )
    offline = mode == "Offline demo"
    if offline:
        st.sidebar.info(
            "Bundled Apple history (17 Feb 2015 – 16 Feb 2017) and a saved SPY file "
            "for the same years. This mode does not call Yahoo Finance."
        )
        symbol = DEFAULT_TICKER
        benchmark = DEFAULT_BENCHMARK
    else:
        st.sidebar.pills(
            "Examples",
            EXAMPLES,
            selection_mode="single",
            key="example_pick",
            on_change=_use_example,
        )
        if "symbol" not in st.session_state:
            st.session_state.symbol = DEFAULT_TICKER
        symbol = st.sidebar.text_input("Symbol", key="symbol").strip().upper()
        if "benchmark" not in st.session_state:
            st.session_state.benchmark = DEFAULT_BENCHMARK
        benchmark = st.sidebar.text_input("Benchmark", key="benchmark").strip().upper()
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
        if error:
            st.sidebar.warning(error)
        elif hits:
            labels = [f"{hit.symbol} — {hit.name}" for hit in hits]
            st.sidebar.selectbox("Matches", labels, key="search_choice")
            st.sidebar.button("Use selected symbol", on_click=_use_search_hit)

    preset = st.sidebar.selectbox(
        "Date range",
        RANGE_OPTIONS,
        index=RANGE_OPTIONS.index("Max" if offline else "1Y"),
    )
    start = end = None
    if preset == "Custom":
        default_start = date(2015, 2, 17) if offline else date.today() - timedelta(days=365)
        default_end = date(2017, 2, 16) if offline else date.today()
        start = st.sidebar.date_input("Start", value=default_start)
        end = st.sidebar.date_input("End", value=default_end)
    refresh = False
    if not offline:
        refresh = st.sidebar.button("Refresh prices")
        st.sidebar.caption(
            f"A download is reused for {CACHE_TTL_HOURS} hours. "
            "Refresh fetches Yahoo Finance again."
        )
    st.sidebar.caption("Zooming a chart does not change the metrics. The metrics use this date range.")
    return {
        "offline": offline,
        "ticker": symbol or DEFAULT_TICKER,
        "benchmark": benchmark or DEFAULT_BENCHMARK,
        "preset": None if preset == "Custom" else preset,
        "start": start,
        "end": end,
        "refresh": refresh,
    }


def _status(briefing) -> None:
    stock = briefing.stock
    latest = stock.latest_bar.date().isoformat()
    if stock.fetched_at is None:
        fetched = "Bundled file. No live refresh."
    else:
        stamp = stock.fetched_at.tz_convert("UTC") if stock.fetched_at.tzinfo else stock.fetched_at.tz_localize("UTC")
        fetched = stamp.strftime("%Y-%m-%d %H:%M UTC")
        fetched += " · cached download" if stock.from_cache else " · just downloaded"
    bench = briefing.benchmark.ticker if briefing.benchmark is not None else "unavailable"
    bench_source = briefing.benchmark.source if briefing.benchmark is not None else briefing.benchmark_error
    window = briefing.report.stats
    st.markdown(
        f"""
        <div class="status-card">
        <strong>Source.</strong> {stock.source}<br>
        <strong>Latest available bar.</strong> {latest}
        &nbsp;·&nbsp; <strong>Fetched.</strong> {fetched}<br>
        <strong>This briefing.</strong> {window.window_start.date().isoformat()}
        to {window.window_end.date().isoformat()}
        · {window.n_days} trading days
        · benchmark {bench}<br>
        <span style="color:#5c6b7a">{bench_source}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _metrics(stats) -> None:
    help_text = metric_help()
    vol = "—" if stats.annualized_volatility is None else f"{stats.annualized_volatility:.1%}"
    drawdown = "—" if stats.max_drawdown is None else f"{stats.max_drawdown:.1%}"
    if stats.excess_return is None:
        gap = "—"
        gap_label = "Vs benchmark"
    else:
        gap = f"{stats.excess_return * 100:+.1f} pp"
        gap_label = f"Vs {stats.benchmark}"
    label = (stats.vol_label or "not ranked").capitalize()
    one, two, three, four, five = st.columns(5)
    one.metric("Cumulative return", f"{stats.cumulative_return:+.1%}", help=help_text["cumulative_return"])
    two.metric("Annualized volatility", vol, help=help_text["annualized_volatility"])
    three.metric("Maximum drawdown", drawdown, help=help_text["max_drawdown"])
    four.metric(gap_label, gap, help=help_text["excess_return"])
    five.metric("Volatility label", label, help=volatility_rule_text())


def _summary(briefing) -> None:
    st.subheader("What the numbers say")
    blocks = "".join(f"<p>{html.escape(sentence)}</p>" for sentence in briefing.report.sentences)
    st.markdown(f'<div class="summary-card">{blocks}</div>', unsafe_allow_html=True)
    st.caption(volatility_rule_text())
    with st.expander("Definitions used above"):
        for note in methodology_notes():
            st.markdown(f"- {note}")
    st.download_button(
        "Download analysis summary",
        data=briefing.markdown,
        file_name=f"{briefing.stock.ticker.lower()}_briefing.md",
        mime="text/markdown",
    )


def _charts(briefing) -> None:
    figures = briefing.figures
    price_tab, drawdown_tab, compare_tab, vol_tab = st.tabs(
        ("Price and averages", "Drawdown", "Versus benchmark", "Rolling volatility")
    )
    with price_tab:
        st.plotly_chart(figures["price"], width="stretch", theme=None, config=PLOTLY_CONFIG)
        st.caption("Hover a date for the adjusted close and both averages. The range slider under the chart zooms the view.")
    with drawdown_tab:
        st.plotly_chart(figures["drawdown"], width="stretch", theme=None, config=PLOTLY_CONFIG)
        st.caption("Each point is that day's adjusted close divided by the highest close so far in the window, minus one.")
    with compare_tab:
        if briefing.benchmark_error:
            st.warning(briefing.benchmark_error)
        chart = figures["comparison"]
        if chart is None:
            st.info("The comparison chart needs two shared sessions.")
        else:
            st.plotly_chart(chart, width="stretch", theme=None, config=PLOTLY_CONFIG)
            st.caption("Both lines are rebased to 100 on the first date they share. Dollar levels are not overlaid.")
    with vol_tab:
        st.plotly_chart(figures["volatility"], width="stretch", theme=None, config=PLOTLY_CONFIG)
        st.caption("The line is the trailing 21-day annualized volatility. The dashed line is the volatility of the whole selected window.")


def main() -> None:
    st.set_page_config(page_title="Market briefing", layout="wide", initial_sidebar_state="expanded")
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    request = _sidebar()
    st.markdown('<div class="kicker">Market briefing</div>', unsafe_allow_html=True)
    title_bench = request["benchmark"] if not request["offline"] else "SPY"
    st.title(f"{request['ticker']} and {title_bench}")
    st.write(
        "Select a stock, read its return, volatility, and drawdown over a window you choose, "
        "and compare that path with a benchmark on the dates both actually traded."
    )

    spinner = (
        "Reading the bundled Apple and SPY files…"
        if request["offline"]
        else f"Loading {request['ticker']} and {request['benchmark']} from Yahoo Finance…"
    )
    try:
        with st.spinner(spinner):
            briefing = run_briefing(
                ticker=request["ticker"],
                benchmark=request["benchmark"],
                start=request["start"],
                end=request["end"],
                preset=request["preset"],
                offline=request["offline"],
                refresh=request["refresh"],
            )
    except (MarketDataError, AnalysisError) as exc:
        st.error(str(exc))
        if not request["offline"]:
            st.info("The offline Apple demo is available under Data in the sidebar. It does not use the network.")
        return

    if briefing.benchmark_error:
        st.warning(briefing.benchmark_error)
    _status(briefing)
    _metrics(briefing.report.stats)
    _summary(briefing)
    _charts(briefing)


main()
