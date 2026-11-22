"""
plots.py — THE PLOT LAYER
==========================
Interactive Plotly charts. This module does not read files and does not
compute returns. It draws the series the feature layer hands it.
Zooming a chart, or pressing one of its range buttons, changes the view
only. The numbers in the briefing still describe the selected window.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLOR_PRICE = "#1f4b99"
COLOR_MA_SHORT = "#d97706"
COLOR_MA_LONG = "#64748b"
COLOR_BENCH = "#334155"
COLOR_DRAWDOWN = "#b45309"
COLOR_VOL = "#1f4b99"
COLOR_UP = "#15803d"
COLOR_DOWN = "#b91c1c"
COLOR_REFERENCE = "#94a3b8"
# One colour per compared symbol, in selection order. The benchmark is
# always the dashed slate line, so a symbol never shares its colour.
SERIES_COLORS = ("#1f4b99", "#d97706", "#7c3aed", "#be123c", "#0891b2")
# Red below zero, white at zero, green above. Monthly returns read as a calendar.
HEATMAP_SCALE = ((0.0, "#b91c1c"), (0.5, "#ffffff"), (1.0, "#15803d"))
GRID = "#e6ebf2"
INK = "#1c2430"
FONT = "Segoe UI, Helvetica Neue, Arial, sans-serif"

CHART_HEIGHT = 480
CANDLE_HEIGHT = 580
CANDLE_ROW_HEIGHTS = (0.74, 0.26)
HEATMAP_BASE_HEIGHT = 170
HEATMAP_ROW_HEIGHT = 34
HISTOGRAM_BINS = 50
FOCUS_LINE_WIDTH = 2.6
LINE_WIDTH = 1.8
MONTH_LABELS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

# Range buttons that jump the view. A button longer than the data on
# the chart would do nothing, so each chart keeps only the shorter ones.
_RANGE_STEPS = (
    ("1M", 31, dict(count=1, step="month", stepmode="backward")),
    ("3M", 92, dict(count=3, step="month", stepmode="backward")),
    ("6M", 183, dict(count=6, step="month", stepmode="backward")),
    ("YTD", 0, dict(count=1, step="year", stepmode="todate")),
    ("1Y", 366, dict(count=1, step="year", stepmode="backward")),
    ("5Y", 1827, dict(count=5, step="year", stepmode="backward")),
)

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
    _add_range_buttons(figure, price.index)
    return figure


def candlestick_chart(
    ohlc: pd.DataFrame,
    ma_short: pd.Series,
    ma_long: pd.Series,
    *,
    ticker: str,
    short_window: int,
    long_window: int,
) -> go.Figure:
    """
    Daily candles on the adjusted scale, the two averages, and volume
    underneath. Weekends are cut out of the axis so candles sit side by
    side. Exchange holidays still leave a one-bar gap.
    """
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=list(CANDLE_ROW_HEIGHTS),
    )
    figure.add_trace(
        go.Candlestick(
            x=ohlc.index,
            open=ohlc["open"],
            high=ohlc["high"],
            low=ohlc["low"],
            close=ohlc["close"],
            name=ticker,
            increasing_line_color=COLOR_UP,
            decreasing_line_color=COLOR_DOWN,
        ),
        row=1,
        col=1,
    )
    for series, window, color in ((ma_short, short_window, COLOR_MA_SHORT), (ma_long, long_window, COLOR_MA_LONG)):
        clean = series.dropna()
        figure.add_trace(
            go.Scatter(
                x=clean.index,
                y=clean.to_numpy(),
                name=f"{window}-day average",
                mode="lines",
                line=dict(color=color, width=1.4),
                hovertemplate="%{y:.2f}",
            ),
            row=1,
            col=1,
        )
    rising = ohlc["close"] >= ohlc["open"]
    figure.add_trace(
        go.Bar(
            x=ohlc.index,
            y=ohlc["volume"],
            name="Volume",
            marker_color=[COLOR_UP if up else COLOR_DOWN for up in rising],
            marker_line_width=0,
            opacity=0.55,
            hovertemplate="%{y:,.0f}",
        ),
        row=2,
        col=1,
    )
    _style(figure, f"{ticker} daily candles and volume", height=CANDLE_HEIGHT)
    figure.update_layout(xaxis_rangeslider_visible=False, showlegend=True)
    figure.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    figure.update_yaxes(title_text="Adjusted price", row=1, col=1)
    figure.update_yaxes(title_text="Volume", row=2, col=1)
    _add_range_buttons(figure, ohlc.index)
    return figure


def drawdown_chart(
    drawdown: pd.Series,
    *,
    ticker: str,
    benchmark_drawdown: pd.Series | None = None,
    benchmark: str | None = None,
) -> go.Figure:
    """Underwater chart of close-to-prior-peak declines inside the window."""
    figure = _figure(f"{ticker} drawdown from the prior peak")
    figure.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.to_numpy(),
            name=ticker,
            mode="lines",
            line=dict(color=COLOR_DRAWDOWN, width=1.6),
            fill="tozeroy",
            fillcolor="rgba(180, 83, 9, 0.18)",
            hovertemplate="%{x|%Y-%m-%d}<br>Drawdown: %{y:.1%}<extra>" + ticker + "</extra>",
        )
    )
    if benchmark_drawdown is not None and benchmark:
        figure.add_trace(
            go.Scatter(
                x=benchmark_drawdown.index,
                y=benchmark_drawdown.to_numpy(),
                name=benchmark,
                mode="lines",
                line=dict(color=COLOR_BENCH, width=1.4, dash="dash"),
                hovertemplate="%{x|%Y-%m-%d}<br>Drawdown: %{y:.1%}<extra>" + benchmark + "</extra>",
            )
        )
    figure.update_layout(yaxis_title="Decline from peak", yaxis_tickformat=".0%")
    _add_range_buttons(figure, drawdown.index)
    return figure


def comparison_chart(
    rebased: pd.DataFrame | None,
    *,
    benchmark: str | None = None,
    focus: str | None = None,
) -> go.Figure | None:
    """
    One line per column of ``rebased``, which the feature layer has
    already set to 100 on the first shared session. The benchmark is
    dashed. The symbol in focus is drawn thicker.
    """
    if rebased is None or len(rebased) < 2:
        return None
    symbols = [column for column in rebased.columns if column != benchmark]
    names = ", ".join(symbols) + (f" and {benchmark}" if benchmark in rebased.columns else "")
    figure = _figure(f"{names}, rebased to 100")
    for position, symbol in enumerate(symbols):
        color = SERIES_COLORS[position % len(SERIES_COLORS)]
        width = FOCUS_LINE_WIDTH if symbol == focus else LINE_WIDTH
        _add_line(figure, rebased[symbol], symbol, color, width=width, hover="Index")
    if benchmark in rebased.columns:
        _add_line(figure, rebased[benchmark], benchmark, COLOR_BENCH, width=LINE_WIDTH, hover="Index", dash="dash")
    figure.add_hline(y=100.0, line_color=COLOR_REFERENCE, line_width=1, line_dash="dot")
    figure.update_layout(yaxis_title="Rebased to 100 on the first shared session")
    _add_range_buttons(figure, rebased.index)
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


def rolling_relationship_chart(
    rolling_beta: pd.Series | None,
    rolling_correlation: pd.Series | None,
    *,
    ticker: str,
    benchmark: str,
    window: int,
) -> go.Figure | None:
    """
    Trailing beta and trailing correlation against the benchmark on one
    axis. The dotted lines mark beta 1 (moves like the benchmark) and 0.
    """
    if rolling_beta is None or rolling_correlation is None:
        return None
    if rolling_beta.dropna().empty or rolling_correlation.dropna().empty:
        return None
    figure = _figure(f"{ticker} versus {benchmark}: rolling {window}-day beta and correlation")
    _add_line(figure, rolling_beta, "Beta", COLOR_PRICE, width=2.0, hover="Beta")
    _add_line(figure, rolling_correlation, "Correlation", COLOR_MA_SHORT, width=1.8, hover="Correlation")
    figure.add_hline(y=1.0, line_color=COLOR_REFERENCE, line_width=1, line_dash="dot")
    figure.add_hline(y=0.0, line_color=COLOR_REFERENCE, line_width=1, line_dash="dot")
    figure.update_layout(yaxis_title="Beta and correlation")
    return figure


def return_histogram(
    returns: pd.Series,
    *,
    ticker: str,
    benchmark_returns: pd.Series | None = None,
    benchmark: str | None = None,
) -> go.Figure:
    """
    How often each size of daily move happened. The benchmark's moves
    are drawn behind in the same bins, so a wider stock shape reads as
    wider at a glance. The dashed line is the stock's mean daily return.
    """
    figure = _figure(f"{ticker} daily returns")
    clean = returns.dropna()
    if benchmark_returns is not None and benchmark:
        figure.add_trace(
            go.Histogram(
                x=benchmark_returns.dropna(),
                name=benchmark,
                nbinsx=HISTOGRAM_BINS,
                bingroup="daily",
                marker_color=COLOR_BENCH,
                opacity=0.35,
                hovertemplate="%{x}<br>" + benchmark + " days: %{y}<extra></extra>",
            )
        )
    figure.add_trace(
        go.Histogram(
            x=clean,
            name=ticker,
            nbinsx=HISTOGRAM_BINS,
            bingroup="daily",
            marker_color=COLOR_PRICE,
            opacity=0.75,
            hovertemplate="%{x}<br>" + ticker + " days: %{y}<extra></extra>",
        )
    )
    if not clean.empty:
        mean = float(clean.mean())
        figure.add_vline(
            x=mean,
            line_dash="dash",
            line_color=COLOR_MA_SHORT,
            annotation_text=f"Mean {mean:+.2%}",
            annotation_position="top right",
        )
    figure.update_layout(
        barmode="overlay",
        hovermode="closest",
        xaxis_title="Daily simple return",
        xaxis_tickformat=".1%",
        yaxis_title="Number of days",
    )
    return figure


def monthly_heatmap(table: pd.DataFrame, *, ticker: str) -> go.Figure:
    """
    Years down the side, months across, one coloured cell per monthly
    return. The colour scale is symmetric, so white is always zero.
    """
    values = table.to_numpy(dtype=float)
    labels = [["" if pd.isna(value) else f"{value:+.1%}" for value in row] for row in values]
    bound = float(pd.DataFrame(values).abs().max().max()) if table.size else 0.0
    figure = go.Figure(
        go.Heatmap(
            z=values,
            x=list(MONTH_LABELS),
            y=[str(year) for year in table.index],
            text=labels,
            texttemplate="%{text}",
            colorscale=[list(stop) for stop in HEATMAP_SCALE],
            zmid=0.0,
            zmin=-bound,
            zmax=bound,
            xgap=3,
            ygap=3,
            hoverongaps=False,
            # Every cell prints its own figure, so a colour bar would only repeat it.
            showscale=False,
            hovertemplate="%{x} %{y}<br>Return: %{text}<extra></extra>",
        )
    )
    _style(
        figure,
        f"{ticker} monthly returns",
        height=HEATMAP_BASE_HEIGHT + HEATMAP_ROW_HEIGHT * max(len(table.index), 1),
    )
    figure.update_layout(hovermode="closest")
    figure.update_xaxes(showgrid=False, side="top", ticks="")
    figure.update_yaxes(showgrid=False, autorange="reversed", type="category", ticks="")
    return figure


def _figure(title: str) -> go.Figure:
    figure = go.Figure()
    _style(figure, title, height=CHART_HEIGHT)
    return figure


def _style(figure: go.Figure, title: str, *, height: int) -> None:
    """One look for every chart: white panel, left title, legend under it."""
    figure.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=18, color=INK)),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        hovermode="x unified",
        margin=dict(l=56, r=24, t=96, b=48),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        font=dict(family=FONT, color=INK, size=13),
        height=height,
    )
    figure.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    figure.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)


def _add_range_buttons(figure: go.Figure, index: pd.Index) -> None:
    """Quick-zoom buttons above the plot, right-aligned so the legend keeps the left."""
    if len(index) < 2:
        return
    span_days = (index[-1] - index[0]).days
    starts_before_january = index[0].year < index[-1].year
    buttons = []
    for label, minimum_days, step in _RANGE_STEPS:
        fits = starts_before_january if label == "YTD" else span_days > minimum_days
        if fits:
            buttons.append(dict(label=label, **step))
    if not buttons:
        return
    buttons.append(dict(label="All", step="all"))
    figure.update_xaxes(
        rangeselector=dict(
            buttons=buttons,
            x=1,
            xanchor="right",
            y=1.02,
            yanchor="bottom",
            bgcolor="#eef2f7",
            activecolor="#c7d6ee",
            font=dict(size=12),
        ),
        row=1 if _has_subplots(figure) else None,
        col=1 if _has_subplots(figure) else None,
    )


def _has_subplots(figure: go.Figure) -> bool:
    return getattr(figure, "_grid_ref", None) is not None


def _add_line(
    figure: go.Figure,
    series: pd.Series,
    name: str,
    color: str,
    *,
    width: float,
    hover: str,
    dash: str | None = None,
) -> None:
    clean = series.dropna()
    figure.add_trace(
        go.Scatter(
            x=clean.index,
            y=clean.to_numpy(),
            name=name,
            mode="lines",
            line=dict(color=color, width=width, dash=dash),
            hovertemplate=f"%{{x|%Y-%m-%d}}<br>{hover}: %{{y:.2f}}<extra>{name}</extra>",
        )
    )


if __name__ == "__main__":
    # A drawn-by-hand series is enough to prove every chart builds. Real
    # prices arrive through dashboard.py, which is the only orchestrator.
    dates = pd.bdate_range("2024-01-02", periods=300)
    wave = pd.Series([100.0 + (step % 17) - 0.02 * step for step in range(300)], index=dates)
    bars = pd.DataFrame(
        {"open": wave, "high": wave + 1.0, "low": wave - 1.0, "close": wave + 0.5, "volume": 1_000.0},
        index=dates,
    )
    moves = wave / wave.shift(1) - 1.0
    months = pd.DataFrame([[0.01 * month for month in range(1, 13)]], index=[2024], columns=range(1, 13))
    lines = pd.DataFrame({"DEMO": wave, "BENCH": wave * 0.9}, index=dates)
    built = {
        "price": price_chart(wave, wave.rolling(20).mean(), wave.rolling(60).mean(), ticker="DEMO", short_window=20, long_window=60),
        "candles": candlestick_chart(bars, wave.rolling(20).mean(), wave.rolling(60).mean(), ticker="DEMO", short_window=20, long_window=60),
        "drawdown": drawdown_chart(wave / wave.cummax() - 1.0, ticker="DEMO"),
        "comparison": comparison_chart(lines / lines.iloc[0] * 100.0, benchmark="BENCH", focus="DEMO"),
        "volatility": volatility_chart(moves.rolling(21).std(), ticker="DEMO", window=21, period_vol=0.2),
        "relationship": rolling_relationship_chart(moves.rolling(63).mean(), moves.rolling(63).mean(), ticker="DEMO", benchmark="BENCH", window=63),
        "histogram": return_histogram(moves, ticker="DEMO"),
        "heatmap": monthly_heatmap(months, ticker="DEMO"),
    }
    for name, chart in built.items():
        print(f"{name}: {len(chart.data)} trace(s)")
