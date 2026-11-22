"""
plots.py — THE PLOT LAYER
==========================
Interactive Plotly charts. This module does not read files and does not
compute returns. Zooming a chart does not change the numbers in the briefing.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

COLOR_PRICE = "#1f4b99"
COLOR_MA_SHORT = "#d97706"
COLOR_MA_LONG = "#64748b"
COLOR_BENCH = "#0f766e"
COLOR_DRAWDOWN = "#b45309"
COLOR_VOL = "#1f4b99"
GRID = "#e6ebf2"
FONT = "Segoe UI, Helvetica Neue, Arial, sans-serif"

PLOTLY_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def price_chart(
    price: pd.Series,
    ma_short: pd.Series,
    ma_long: pd.Series,
    *,
    ticker: str,
    short_window: int,
    long_window: int,
) -> go.Figure:
    """Adjusted close with the two simple moving averages. Rangeslider on."""
    figure = _figure(f"{ticker} adjusted close")
    _add_line(figure, price, ticker, COLOR_PRICE, width=2.2, hover="Adjusted close")
    _add_line(figure, ma_short, f"{short_window}-day average", COLOR_MA_SHORT, width=1.4, hover="Average")
    _add_line(figure, ma_long, f"{long_window}-day average", COLOR_MA_LONG, width=1.4, hover="Average")
    figure.update_layout(xaxis_rangeslider_visible=True, yaxis_title="Adjusted close")
    figure.update_yaxes(tickprefix="")
    return figure


def drawdown_chart(drawdown: pd.Series, *, ticker: str) -> go.Figure:
    """Underwater chart of close-to-prior-peak declines inside the window."""
    figure = _figure(f"{ticker} drawdown from the prior peak")
    figure.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.to_numpy(),
            name="Drawdown",
            mode="lines",
            line=dict(color=COLOR_DRAWDOWN, width=1.6),
            fill="tozeroy",
            fillcolor="rgba(180, 83, 9, 0.18)",
            hovertemplate="%{x|%Y-%m-%d}<br>Drawdown: %{y:.1%}<extra></extra>",
        )
    )
    figure.update_layout(yaxis_title="Decline from peak", yaxis_tickformat=".0%")
    return figure


def comparison_chart(
    aligned: pd.DataFrame,
    *,
    ticker: str,
    benchmark: str,
) -> go.Figure | None:
    """
    Both adjusted-close paths rebased to 100 on the first shared session.
    Levels from different files are not plotted in dollars against each other.
    """
    if aligned is None or len(aligned) < 2:
        return None
    base = aligned.iloc[0]
    rebased = aligned / base * 100.0
    figure = _figure(f"{ticker} and {benchmark}, rebased to 100")
    _add_line(figure, rebased["stock"], ticker, COLOR_PRICE, width=2.2, hover="Index")
    _add_line(figure, rebased["benchmark"], benchmark, COLOR_BENCH, width=2.0, hover="Index")
    figure.update_layout(yaxis_title="Rebased to 100 on the first shared session")
    return figure


def volatility_chart(
    rolling: pd.Series,
    *,
    ticker: str,
    window: int,
    period_vol: float | None,
) -> go.Figure:
    """Rolling annualized volatility, with the window's own vol as a line."""
    figure = _figure(f"{ticker} rolling {window}-day annualized volatility")
    clean = rolling.dropna()
    figure.add_trace(
        go.Scatter(
            x=clean.index,
            y=clean.to_numpy(),
            name=f"{window}-day vol",
            mode="lines",
            line=dict(color=COLOR_VOL, width=1.8),
            hovertemplate="%{x|%Y-%m-%d}<br>Annualized vol: %{y:.1%}<extra></extra>",
        )
    )
    if period_vol is not None:
        figure.add_hline(
            y=period_vol,
            line_dash="dash",
            line_color="#b45309",
            annotation_text=f"Window vol {period_vol:.1%}",
            annotation_position="top left",
        )
    figure.update_layout(yaxis_title="Annualized volatility", yaxis_tickformat=".0%")
    return figure


def _figure(title: str) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=18, color="#1c2430")),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        hovermode="x unified",
        margin=dict(l=56, r=24, t=64, b=48),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        font=dict(family=FONT, color="#1c2430", size=13),
        height=480,
    )
    figure.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    figure.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    return figure


def _add_line(
    figure: go.Figure,
    series: pd.Series,
    name: str,
    color: str,
    *,
    width: float,
    hover: str,
) -> None:
    clean = series.dropna()
    figure.add_trace(
        go.Scatter(
            x=clean.index,
            y=clean.to_numpy(),
            name=name,
            mode="lines",
            line=dict(color=color, width=width),
            hovertemplate=f"%{{x|%Y-%m-%d}}<br>{hover}: %{{y:.2f}}<extra>{name}</extra>",
        )
    )
