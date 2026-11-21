"""
plots.py — THE PLOT LAYER
==========================
Pure rendering. This module never reads a CSV. It draws the 2×2
dashboard the orchestrator asks for:

    price, with callouts when the file actually covers those dates
    daily log returns
    rolling annualized volatility
    return-vs-volatility scatter (color = momentum, size = |return|)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- Styling constants ------------------------------------------------------
FIGSIZE = (14, 9)
PRICE_COLOR = "steelblue"
RETURN_COLOR = "gray"
VOL_COLOR = "darkorange"
SCATTER_CMAP = "RdYlGn"

# Maps |log return| onto marker area. A 2% day (0.02) becomes area 16,
# large enough to see and small enough not to hide the rest of the cloud.
# The legend uses the same scale, so the swatches match the points.
RETURN_SIZE_SCALE = 800
SIZE_LEGEND = (
    (0.005, "|r| = 0.5%"),
    (0.02, "|r| = 2%"),
    (0.05, "|r| = 5%"),
)

# Callouts that sit inside the bundled AAPL sample. They are skipped
# when the nearest trading day is farther away than this, so a file
# from another year is not labeled with Apple's calendar.
EVENT_SNAP_TOLERANCE_DAYS = 7
# Dollar offsets were chosen for the bundled sample, whose median close
# is about $113. Scaling by median / REFERENCE_PRICE keeps those labels
# put on the sample and on-chart for a stock that trades at another level.
REFERENCE_PRICE = 113.0
EVENTS = (
    ("2015-08-24", "China shock\n(Aug '15)", 25),
    ("2016-09-09", "iPhone 7 launch\n(Sep '16)", 20),
)


def setup_figure() -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    fig.patch.set_facecolor("white")
    for ax in axes.ravel():
        ax.set_facecolor("white")
    return fig, axes


def plot_price(ax: plt.Axes, df: pd.DataFrame) -> None:
    ax.set_title(f"{df.attrs.get('ticker', '?')} Close Price", fontweight="bold")
    ax.set_ylabel("Price (USD)")
    ax.grid(True, alpha=0.3)
    ax.plot(df.index, df["close"], color=PRICE_COLOR, linewidth=1.5)


def plot_returns(ax: plt.Axes, df: pd.DataFrame) -> None:
    ax.set_title("Daily Log Returns", fontweight="bold")
    ax.set_ylabel("log return")
    ax.grid(True, alpha=0.3)
    ax.plot(df.index, df["log_return"], color=RETURN_COLOR, linewidth=0.8)
    ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.5)


def plot_volatility(ax: plt.Axes, df: pd.DataFrame) -> None:
    vol_window = df.attrs.get("vol_window", 21)
    vol_col = f"volatility_{vol_window}d"
    ax.set_title(f"Rolling {vol_window}-Day Annualized Volatility", fontweight="bold")
    ax.set_ylabel("annualized σ")
    ax.grid(True, alpha=0.3)
    ax.plot(df.index, df[vol_col], color=VOL_COLOR, linewidth=1.5)


def plot_scatter(ax: plt.Axes, df: pd.DataFrame):
    """Return the scatter handle so the colorbar can attach to it."""
    vol_window = df.attrs.get("vol_window", 21)
    mom_window = df.attrs.get("mom_window", 63)
    vol_col = f"volatility_{vol_window}d"
    mom_col = f"momentum_{mom_window}d"

    ax.set_title("Return vs Volatility (color = momentum, size = |return|)",
                 fontweight="bold")
    ax.set_xlabel(f"{vol_window}d annualized volatility")
    ax.set_ylabel("daily log return")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.5)

    # Warmup rows have no vol/momentum yet. Dropping them here is
    # display logic: the feature layer keeps the NaNs on purpose.
    sub = df[[vol_col, "log_return", mom_col]].dropna()

    sc = ax.scatter(
        x=sub[vol_col],
        y=sub["log_return"],
        c=sub[mom_col],
        s=sub["log_return"].abs() * RETURN_SIZE_SCALE,
        cmap=SCATTER_CMAP,
        alpha=0.6,
        edgecolors="black",
        linewidths=0.3,
    )
    return sc


def add_colorbar(fig: plt.Figure, ax: plt.Axes, scatter_handle, label: str) -> None:
    """
    Attach a colorbar so the gradient is a scale: red is negative
    momentum, green is positive. Without it the colors are decoration.
    """
    cbar = fig.colorbar(scatter_handle, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)


def add_size_legend(ax: plt.Axes) -> None:
    """
    Manual size legend. Matplotlib will not build one for marker area,
    so these are empty scatters: swatches for the legend, no data.
    """
    handles = []
    for magnitude, label in SIZE_LEGEND:
        handle = ax.scatter(
            [],
            [],
            s=magnitude * RETURN_SIZE_SCALE,
            color="gray",
            alpha=0.6,
            edgecolors="black",
            linewidths=0.3,
            label=label,
        )
        handles.append(handle)

    ax.legend(
        handles=handles,
        title="dot size",
        loc="upper left",
        fontsize=8,
        title_fontsize=8,
        framealpha=0.9,
        labelspacing=1.2,
    )


def add_event_annotations(ax_price: plt.Axes, df: pd.DataFrame) -> None:
    """
    Arrows on the price chart for dates the loaded file actually covers.

    The two events below are real episodes inside the bundled sample.
    A nearest-date lookup without a tolerance would pin those labels
    onto whatever history the caller loaded. The tolerance stops that.
    """
    if df.empty:
        return

    median_price = float(df["close"].median())
    scale = median_price / REFERENCE_PRICE if median_price > 0 else 1.0
    tolerance = pd.Timedelta(days=EVENT_SNAP_TOLERANCE_DAYS)

    for date_str, label, y_offset in EVENTS:
        target = pd.Timestamp(date_str)
        idx_pos = df.index.get_indexer([target], method="nearest")[0]
        if idx_pos < 0:
            continue
        actual_date = df.index[idx_pos]
        if abs(actual_date - target) > tolerance:
            continue
        price = float(df.loc[actual_date, "close"])
        text_y = price + y_offset * scale
        ax_price.annotate(
            label,
            xy=(actual_date, price),
            xytext=(actual_date, text_y),
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
            fontsize=8,
            ha="center",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="gray",
                alpha=0.9,
            ),
        )


def build_dashboard(df: pd.DataFrame) -> plt.Figure:
    """Wire the four panels and the three reading aids. No file I/O."""
    fig, axes = setup_figure()
    ticker = df.attrs.get("ticker", "Market")
    start = df.index.min().date().isoformat()
    end = df.index.max().date().isoformat()
    fig.suptitle(
        f"{ticker} market dashboard    {start} to {end}",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    plot_price(axes[0, 0], df)
    plot_returns(axes[0, 1], df)
    plot_volatility(axes[1, 0], df)
    scatter = plot_scatter(axes[1, 1], df)

    mom_window = df.attrs.get("mom_window", 63)
    add_colorbar(fig, axes[1, 1], scatter, label=f"{mom_window}d momentum")
    add_size_legend(axes[1, 1])
    add_event_annotations(axes[0, 0], df)

    # Leave a strip at the top so the suptitle is not clipped.
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


if __name__ == "__main__":
    from data_loader import load_prices
    from features import build_features

    prices = load_prices()
    feats = build_features(prices)
    fig = build_dashboard(feats)
    fig.savefig("dashboard.png", dpi=120, bbox_inches="tight", facecolor="white")
    print("Saved dashboard.png")
