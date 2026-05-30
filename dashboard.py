"""
dashboard.py — THE ORCHESTRATOR
================================
This is THE file you run. It wires the three layers together:

    data_loader  →  features  →  plots  →  saved PNG

Everything below is composition: import each layer, call them in order,
save the result. No new logic — just gluing the modules together.
"""

from __future__ import annotations
from pathlib import Path

from data_loader import load_prices
from features import build_features
from plots import build_dashboard


def main(
    input_csv: str = "aapl_raw.csv",
    output_png: str = "dashboard.png",
) -> Path:
    """
    Run the full pipeline end-to-end.

    Parameters
    ----------
    input_csv  : path to the raw OHLCV file
    output_png : where to save the rendered dashboard

    Returns
    -------
    Path to the saved image.
    """
    # TODO #9 ----------------------------------------------------------------
    # Call load_prices() with input_csv. Save the result to a variable
    # named `prices`.
    # ------------------------------------------------------------------------
    prices = load_prices(input_csv)


    # TODO #10 ---------------------------------------------------------------
    # Call build_features() on `prices`. Save the result to `feats`.
    # ------------------------------------------------------------------------
    feats = build_features(prices)


    # TODO #11 ---------------------------------------------------------------
    # Call build_dashboard() on `feats`. Save the result to `fig`.
    # ------------------------------------------------------------------------
    fig = build_dashboard(feats)


    # TODO #12 ---------------------------------------------------------------
    # Save `fig` to disk using fig.savefig(...) with arguments:
    #   output_png, dpi=120, bbox_inches="tight"
    # ------------------------------------------------------------------------
    fig.savefig(output_png, dpi=120, bbox_inches="tight")


    out = Path(output_png).resolve()
    print(f"✅ Saved dashboard to {out}")
    return out


if __name__ == "__main__":
    main()
