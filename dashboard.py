"""
dashboard.py — THE ORCHESTRATOR
================================
The only place the pipeline runs end to end:

    data_loader  →  features  →  regime snapshot  →  plots  →  PNG

`python dashboard.py` briefs the bundled Apple sample and writes
dashboard.png. No API key, no network. Pass --csv to brief another
file in the same column layout; that path is optional.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import pandas as pd

from data_loader import SAMPLE_CSV, load_prices
from features import (
    DEFAULT_MOM_WINDOW,
    DEFAULT_VOL_WINDOW,
    build_features,
    regime_snapshot,
)
from plots import build_dashboard


# Shortest window that is still a window. A 1-day "rolling" std is empty.
MIN_WINDOW = 2
OUTPUT_DPI = 120
OUTPUT_PNG = "dashboard.png"


class Analysis(NamedTuple):
    """Everything the CLI and the browser demo display."""

    features: pd.DataFrame
    figure: plt.Figure
    regime: dict


def analyze(
    input_csv: str | Path | None = None,
    vol_window: int = DEFAULT_VOL_WINDOW,
    mom_window: int = DEFAULT_MOM_WINDOW,
    ticker: str | None = None,
) -> Analysis:
    """
    Run one risk briefing.

    The latest row becomes the written regime. The figure keeps the
    whole path, so a calm ending can still show an earlier spike.
    """
    vol_window = int(vol_window)
    mom_window = int(mom_window)
    if vol_window < MIN_WINDOW or mom_window < MIN_WINDOW:
        raise ValueError(f"Windows must be at least {MIN_WINDOW} trading days.")

    prices = load_prices(input_csv, ticker=ticker)
    feats = build_features(prices, vol_window=vol_window, mom_window=mom_window)
    regime = regime_snapshot(feats)
    figure = build_dashboard(feats)
    return Analysis(feats, figure, regime)


def _iso(value) -> str:
    return pd.Timestamp(value).date().isoformat()


def _ordinal(percentile: float) -> str:
    """0.44 -> '44th'. Used only in the sentence a person reads."""
    n = int(round(percentile * 100))
    n = min(100, max(0, n))
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def format_regime(regime: dict) -> str:
    """One paragraph: price, vol regime, trend, and the worst vol in the file."""
    as_of = _iso(regime["as_of"])
    peak_on = _iso(regime["volatility_peak_date"])
    vol_window = regime["vol_window"]
    mom_window = regime["mom_window"]
    trend = regime["trend"]
    if trend == "flat":
        trend_clause = (
            f"{mom_window}-day momentum is flat "
            f"({regime['momentum_simple']:+.1%} over the window)."
        )
    else:
        article = "an" if trend == "uptrend" else "a"
        trend_clause = (
            f"{mom_window}-day momentum is {article} {trend}: "
            f"{regime['momentum_simple']:+.1%} over the window."
        )
    return (
        f"{regime['ticker']} as of {as_of}: "
        f"close {regime['close']:.2f} "
        f"(session log return {regime['daily_log_return']:+.2%}). "
        f"{vol_window}-day annualized volatility is {regime['volatility']:.1%}, "
        f"{regime['volatility_regime']} for this file "
        f"({_ordinal(regime['volatility_percentile'])} percentile, "
        f"median {regime['volatility_median']:.1%}). "
        f"{trend_clause} "
        f"Peak {vol_window}-day vol in this file was "
        f"{regime['volatility_peak']:.1%} on {peak_on}."
    )


def format_context(regime: dict) -> str:
    """The coverage line under the briefing: what history this read used."""
    return (
        f"File covers {_iso(regime['history_start'])} to {_iso(regime['as_of'])}. "
        f"{regime['n_obs']} days have both features fully warmed up."
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Brief the risk regime of one stock from an OHLCV CSV. "
            "Defaults to the bundled Apple sample. No API key is used."
        )
    )
    parser.add_argument(
        "--csv",
        default=None,
        help=f"OHLCV CSV. Defaults to the bundled sample ({SAMPLE_CSV.name}).",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_PNG,
        help=f"Where to write the dashboard image (default: {OUTPUT_PNG}).",
    )
    parser.add_argument("--vol-window", type=int, default=DEFAULT_VOL_WINDOW)
    parser.add_argument("--mom-window", type=int, default=DEFAULT_MOM_WINDOW)
    parser.add_argument(
        "--ticker",
        default=None,
        help="Title label. Inferred from column prefixes such as AAPL.Close when omitted.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    """Brief the sample (or --csv) and save the figure. Returns the PNG path."""
    args = _parse_args(argv)
    result = analyze(
        args.csv,
        vol_window=args.vol_window,
        mom_window=args.mom_window,
        ticker=args.ticker,
    )
    result.figure.savefig(
        args.output,
        dpi=OUTPUT_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(result.figure)

    print(format_regime(result.regime))
    print(format_context(result.regime))
    out = Path(args.output).resolve()
    print(f"Saved dashboard to {args.output}")
    return out


if __name__ == "__main__":
    main()
