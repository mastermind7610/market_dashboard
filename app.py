"""
app.py — browser demo of the same briefing dashboard.py runs.

This file only collects a file path and two window lengths, then
displays what dashboard.analyze returns. It does not parse prices
or compute features. The default path is the bundled sample, so the
page works with no API key.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import streamlit as st

from dashboard import OUTPUT_DPI, analyze, format_context, format_regime
from data_loader import SAMPLE_CSV
from features import DEFAULT_MOM_WINDOW, DEFAULT_VOL_WINDOW

# Slider bounds stay inside what the bundled ~500-day sample can support.
# Five days is the shortest vol window that is still a window. 126 days
# is about six months, and still leaves plotted points after warmup.
VOL_WINDOW_MIN = 5
VOL_WINDOW_MAX = 63
MOM_WINDOW_MIN = 21
MOM_WINDOW_MAX = 126

st.set_page_config(page_title="Risk regime briefing", layout="wide")
st.title("Single-stock risk regime")
st.write(
    "A one-screen briefing before a meeting: where has this stock traded, "
    "is recent volatility quiet or elevated for its own history, and is the "
    "trailing quarter an uptrend or a downtrend?"
)
st.caption(
    "Demo mode reads the bundled Apple sample (17 Feb 2015 – 16 Feb 2017). "
    "No API key and no market-data account."
)

with st.sidebar:
    st.header("Briefing inputs")
    vol_window = st.slider(
        "Volatility window (trading days)",
        min_value=VOL_WINDOW_MIN,
        max_value=VOL_WINDOW_MAX,
        value=DEFAULT_VOL_WINDOW,
        step=1,
        help="Rolling window for annualized volatility. 21 days is about one month.",
    )
    mom_window = st.slider(
        "Momentum window (trading days)",
        min_value=MOM_WINDOW_MIN,
        max_value=MOM_WINDOW_MAX,
        value=DEFAULT_MOM_WINDOW,
        step=1,
        help="Trailing cumulative log return. 63 days is about one quarter.",
    )
    ticker = st.text_input(
        "Ticker label (optional)",
        value="",
        help="Leave blank to use the column prefix, such as AAPL from AAPL.Close.",
    )
    upload = st.file_uploader(
        "Optional CSV (replaces the sample)",
        type=["csv"],
        help="Date plus open/high/low/close/volume, or AAPL.Open-style names.",
    )
    st.caption("An uploaded file stays on this machine. Nothing is sent to a data vendor.")

if upload is not None:
    # One stable path per machine, overwritten on each rerun. The file
    # is the optional real-data input; the bundled sample remains the default.
    upload_dir = Path(tempfile.gettempdir()) / "market_dashboard_upload"
    upload_dir.mkdir(exist_ok=True)
    csv_path = upload_dir / "upload.csv"
    csv_path.write_bytes(upload.getvalue())
    source = f"uploaded file ({upload.name})"
else:
    csv_path = SAMPLE_CSV
    source = f"bundled sample ({SAMPLE_CSV.name})"

try:
    result = analyze(
        csv_path,
        vol_window=vol_window,
        mom_window=mom_window,
        ticker=ticker or None,
    )
except Exception as exc:
    st.error(f"Could not build the briefing: {exc}")
    st.stop()

regime = result.regime
close_col, vol_col, regime_col, trend_col = st.columns(4)
close_col.metric("Close", f"{regime['close']:.2f}")
vol_col.metric(f"{regime['vol_window']}d vol", f"{regime['volatility']:.1%}")
regime_col.metric("Vol regime", regime["volatility_regime"])
trend_col.metric(f"{regime['mom_window']}d trend", regime["trend"])

st.subheader("Regime read")
st.info(format_regime(regime))
st.caption(f"{format_context(regime)} Source: {source}.")

buffer = io.BytesIO()
result.figure.savefig(
    buffer,
    format="png",
    dpi=OUTPUT_DPI,
    bbox_inches="tight",
    facecolor="white",
)
st.download_button(
    "Download PNG",
    data=buffer.getvalue(),
    file_name=f"{str(regime['ticker']).lower()}_dashboard.png",
    mime="image/png",
)
st.pyplot(result.figure, clear_figure=True)
st.caption(
    "Panels, left to right: close (event callouts appear only when the file "
    "covers those dates), daily log returns, rolling annualized volatility, "
    "and return versus volatility. Scatter color is momentum and dot size is "
    "the absolute return."
)
