"""
plots.py — THE PLOT LAYER
=========================
2x2 dashboard panels:
    - price (with event annotations)
    - drawdown from peak
    - rolling annualized volatility
    - return vs volatility scatter (color = momentum, size = |return|)

Polish: colorbar, custom size legend, event annotations.
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
DRAWDOWN_COLOR = "firebrick"
SCATTER_CMAP = "RdYlGn"


# ---- Figure scaffolding (DONE) ---------------------------------------------
def setup_figure() -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    fig.suptitle("Market Dashboard", fontsize=16, fontweight="bold", y=0.995)
    return fig, axes


# ---- Plot 1: Price trend (TODO #1 filled) ----------------------------------
def plot_price(ax: plt.Axes, df: pd.DataFrame) -> None:
    ax.set_title(f"{df.attrs.get('ticker', '?')} Close Price", fontweight="bold")
    ax.set_ylabel("Price (USD)")
    ax.grid(True, alpha=0.3)
    ax.plot(df.index, df["close"], color=PRICE_COLOR, linewidth=1.5)


# ---- Plot 2: Daily returns (kept for self-tests / reuse) -------------------
def plot_returns(ax: plt.Axes, df: pd.DataFrame) -> None:
    ax.set_title("Daily Log Returns", fontweight="bold")
    ax.set_ylabel("log return")
    ax.grid(True, alpha=0.3)
    ax.plot(df.index, df["log_return"], color=RETURN_COLOR, linewidth=0.8)
    ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.5)


# ---- Plot 2b: Drawdown from peak ------------------------------------------
def plot_drawdown(ax: plt.Axes, df: pd.DataFrame) -> None:
    """
    Peak-to-trough loss over time. Filled area makes depth of underwater
    periods obvious at a glance — complementary to volatility (noise) vs
    drawdown (how far below the high-water mark).
    """
    ax.set_title("Drawdown from Peak", fontweight="bold")
    ax.set_ylabel("drawdown")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.5)
    ax.fill_between(
        df.index,
        df["drawdown"],
        0,
        color=DRAWDOWN_COLOR,
        alpha=0.35,
        linewidth=0,
    )
    ax.plot(df.index, df["drawdown"], color=DRAWDOWN_COLOR, linewidth=1.0)


# ---- Plot 3: Volatility (TODO #3 filled) -----------------------------------
def plot_volatility(ax: plt.Axes, df: pd.DataFrame) -> None:
    vol_window = df.attrs.get("vol_window", 21)
    vol_col = f"volatility_{vol_window}d"
    ax.set_title(f"Rolling {vol_window}-Day Annualized Volatility", fontweight="bold")
    ax.set_ylabel("annualized σ")
    ax.grid(True, alpha=0.3)
    ax.plot(df.index, df[vol_col], color=VOL_COLOR, linewidth=1.5)


# ---- Plot 4: Scatter (TODO #4 filled, now returns sc) ----------------------
def plot_scatter(ax: plt.Axes, df: pd.DataFrame):
    """Returns the scatter handle so the colorbar can attach to it."""
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

    sub = df[[vol_col, "log_return", mom_col]].dropna()

    sc = ax.scatter(
        x=sub[vol_col],
        y=sub["log_return"],
        c=sub[mom_col],
        s=sub["log_return"].abs() * 800,
        cmap=SCATTER_CMAP,
        alpha=0.6,
        edgecolors="black",
        linewidths=0.3,
    )
    return sc


# ============================================================================
# STEP 3b POLISH LAYER — three new TODOs below
# ============================================================================


def add_colorbar(fig: plt.Figure, ax: plt.Axes, scatter_handle, label: str) -> None:
    """
    Attach a colorbar to the scatter plot so the reader knows what the
    color gradient actually means (red = negative momentum, green = positive).
    Without a colorbar, the colors are decoration. With one, they're a scale.
    """
    # TODO #5 ----------------------------------------------------------------
    # Use fig.colorbar(...). It needs:
    #   - the scatter handle (first positional argument)
    #   - ax=ax           — tells it which subplot to attach next to
    #   - shrink=0.85     — makes the bar 85% the height of the subplot
    #   - pad=0.02        — small gap between subplot and bar
    #
    # Save the return value into a variable named `cbar`. We'll then
    # call two methods on it:
    #   cbar.set_label(label, fontsize=9)
    #   cbar.ax.tick_params(labelsize=8)
    #
    # If you get stuck, paste this into Claude Code:
    #   @plots.py I'm on TODO #5 — adding a colorbar. Don't write the
    #   code. Just confirm: which matplotlib function attaches a
    #   colorbar to a figure given a scatter handle?
    # ------------------------------------------------------------------------
    cbar = fig.colorbar(scatter_handle, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)


def add_size_legend(ax: plt.Axes) -> None:
    """
    Build a manual legend showing what the dot sizes mean. matplotlib's
    default legend handles COLOR encoding automatically, but for SIZE
    encoding we have to create 'proxy' markers — fake scatter points
    with no data, just visible as legend swatches.
    """
    sample_returns = [0.005, 0.02, 0.05]              # 0.5%, 2%, 5% magnitude
    labels = ["|r| = 0.5%", "|r| = 2%", "|r| = 5%"]

    handles = []
    for r, lbl in zip(sample_returns, labels):
        # TODO #6 ------------------------------------------------------------
        # Create one proxy scatter point per size. The trick: call
        # ax.scatter([], [], ...) — empty x/y means NO data is plotted,
        # but the legend still gets a swatch.
        #
        # Use these arguments:
        #   [], []                          — empty data
        #   s = r * 800                     — SAME 800 magnification as the real scatter
        #   color = "gray"                  — neutral color since we're not encoding momentum here
        #   alpha = 0.6
        #   edgecolors = "black"
        #   linewidths = 0.3
        #   label = lbl                     — the legend text
        #
        # Save the return value into a variable named `h` and append
        # it to handles (do `handles.append(h)`).
        #
        # Stuck? Ask Claude Code:
        #   @plots.py I'm on TODO #6. Without writing it, confirm: how
        #   do I create a scatter marker that appears in a legend but
        #   plots no actual data?
        # --------------------------------------------------------------------
        h = ax.scatter([], [], s=r * 800, color="gray", alpha=0.6, edgecolors="black", linewidths=0.3, label=lbl)
        handles.append(h)

    # TODO #7 ----------------------------------------------------------------
    # Now actually attach the legend to the chart. Call ax.legend(...) with:
    #   handles = handles
    #   title = "dot size"
    #   loc = "upper left"
    #   fontsize = 8
    #   title_fontsize = 8
    #   framealpha = 0.9
    #   labelspacing = 1.2
    # ------------------------------------------------------------------------
    ax.legend(handles=handles, title="dot size", loc="upper left", fontsize=8, title_fontsize=8, framealpha=0.9, labelspacing=1.2)


def add_event_annotations(ax_price: plt.Axes, df: pd.DataFrame) -> None:
    """
    Add arrows + text labels on the price chart pointing at specific dates.
    Anchors the abstract wiggles to real-world events the reader can recall.
    """
    events = [
        ("2015-08-24", "China shock\n(Aug '15)", 25),
        ("2016-09-09", "iPhone 7 launch\n(Sep '16)", 20),
    ]

    for date_str, label, y_offset in events:
        target = pd.Timestamp(date_str)
        # Snap to nearest actual trading day (markets close on weekends).
        idx_pos = df.index.get_indexer([target], method="nearest")[0]
        actual_date = df.index[idx_pos]
        price = df.loc[actual_date, "close"]

        # TODO #8 ------------------------------------------------------------
        # Use ax_price.annotate(...) with:
        #   label                        — first positional, the text to show
        #   xy = (actual_date, price)    — where the ARROW POINT lands
        #   xytext = (actual_date, price + y_offset)
        #                                — where the TEXT BOX sits (above the point)
        #   arrowprops = dict(arrowstyle="->", color="black", lw=0.8)
        #   fontsize = 8
        #   ha = "center"                — horizontal alignment of text
        #   bbox = dict(boxstyle="round,pad=0.3", facecolor="white",
        #               edgecolor="gray", alpha=0.9)
        #
        # The structure: a text label PLUS an arrow connecting the label
        # to the data point. xy=where the arrow points, xytext=where the
        # text sits.
        #
        # Claude Code prompt if stuck:
        #   @plots.py I'm on TODO #8 — annotating dates on a price chart.
        #   Don't write code. Just explain: in ax.annotate(), what's the
        #   difference between `xy` and `xytext`?
        # --------------------------------------------------------------------
        ax_price.annotate(label, xy=(actual_date, price), xytext=(actual_date, price + y_offset), arrowprops=dict(arrowstyle="->", color="black", lw=0.8), fontsize=8, ha="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9))

# ---- Orchestrator (DONE) ---------------------------------------------------
def build_dashboard(df: pd.DataFrame) -> plt.Figure:
    """
    Wires the four base plots PLUS the three polish helpers.
    """
    fig, axes = setup_figure()
    plot_price(axes[0, 0], df)
    # Drawdown replaces raw returns in the grid: daily log returns remain
    # visible as the scatter y-axis, while this panel answers "how deep
    # below the peak am I?"
    plot_drawdown(axes[0, 1], df)
    plot_volatility(axes[1, 0], df)
    sc = plot_scatter(axes[1, 1], df)

    # NEW for Step 3b — these three calls turn a working chart into a
    # readable one. They will do nothing until you fill in the TODOs.
    mom_window = df.attrs.get("mom_window", 63)
    add_colorbar(fig, axes[1, 1], sc, label=f"{mom_window}d momentum")
    add_size_legend(axes[1, 1])
    add_event_annotations(axes[0, 0], df)

    fig.tight_layout()
    return fig


# ---- Self-test -------------------------------------------------------------
if __name__ == "__main__":
    from data_loader import load_prices
    from features import build_features

    prices = load_prices("aapl_raw.csv")
    feats = build_features(prices)
    fig = build_dashboard(feats)
    fig.savefig("dashboard.png", dpi=120, bbox_inches="tight")
    print("Saved dashboard.png")