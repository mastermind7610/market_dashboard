# Market Dashboard

A 2×2 matplotlib dashboard for single-stock market analysis, built with a clean modular architecture — data → features → plots → orchestrator. Real historical OHLCV data, real quant-grade feature engineering, and a polished four-panel visualization.

![Dashboard preview](dashboard.png)

## What it shows

Four panels, each answering a different question about the stock:

| Panel | Question it answers |
|---|---|
| **Price** | Where has the stock been? *(annotated with key events)* |
| **Daily log returns** | What does the day-to-day "heartbeat" look like? |
| **Rolling annualized volatility** | How has the risk regime changed over time? |
| **Return vs volatility scatter** | Are big-move days happening in uptrends or downtrends? *(color = momentum, size = \|return\|)* |

Built on real Apple (AAPL) historical data, Feb 2015 – Feb 2017.

## Architecture

One-directional data flow. One file per responsibility. Each layer hands the next a guaranteed-clean contract.

```
data_loader.py  →  features.py  →  plots.py  →  dashboard.py
   (clean OHLCV)    (signals)       (renders)    (orchestrates)
```

- **`data_loader.py`** — loads + validates raw CSV into a strict OHLCV frame (zero NaNs, sorted DatetimeIndex)
- **`features.py`** — adds log returns, rolling annualized volatility, trailing momentum
- **`plots.py`** — pure rendering layer: four subplots, colorbar, custom size legend, event annotations
- **`dashboard.py`** — single entry point that wires it all together

## How to run

```bash
pip install numpy pandas matplotlib
python dashboard.py
```

Output: `dashboard.png` in the project root.

## Quant / ML concepts applied

- **Log returns** instead of simple returns — time-additive across periods, symmetric on up/down moves
- **Annualization with √252** — trading days per year, not calendar days (one of the most common junior-quant bugs)
- **Causal rolling windows** — every feature uses only past data, so the scatter represents decisions you could plausibly have made *at the time* (no lookahead bias)
- **Visual encoding of four dimensions** in one chart — x (volatility), y (return), color (momentum), size (magnitude)

## Data source

Daily OHLCV from the public [plotly/datasets](https://github.com/plotly/datasets) repository.

## File structure

```
market_dashboard/
├── data_loader.py     # data layer
├── features.py        # feature engineering layer
├── plots.py           # plotting layer
├── dashboard.py       # orchestrator (run this)
├── aapl_raw.csv       # raw input data
├── dashboard.png      # rendered output
└── README.md
```
