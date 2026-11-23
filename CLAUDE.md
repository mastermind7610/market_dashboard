# Market Dashboard — Project Conventions

## Architecture
Modular layers, one-directional data flow:

    data_loader.py → features.py → plots.py → dashboard.py
                              ↘
                     analyst.py ↗

Never bypass a layer. `plots.py` does not read CSV or call `data_loader` /
`features`. `features.py` does not call Plotly. `analyst.py` only reads
already-computed reports (never prices or charts). `dashboard.py` is the
only orchestrator and is the only place that runs end-to-end. `app.py` is
the Streamlit UI: it collects inputs and displays the briefing.

## Style
- Small, single-purpose functions. One concept per function.
- Trading / ML reasoning lives in comments BEFORE the code, not after.
- No magic numbers — use named constants (TRADING_DAYS = 252, etc.).
- Self-test at the bottom of every module: `if __name__ == "__main__"`.
- Prefer typed public functions; CI runs mypy on core modules and ruff on the tree.

## Data contracts between layers
- `load_prices()` / `fetch_history()` return clean OHLC frames (or
  `PriceHistory`) with DatetimeIndex. Zero NaNs in the price columns used
  for analysis. attrs / history fields carry ticker and source metadata.
- `build_period()` / `build_comparison()` produce typed reports and
  comparison rows. Early NaNs in rolling series are expected (warmup).
- `build_dashboard` path: `run_briefing()` returns a `Briefing` (figures,
  markdown, optional `AnalystPack`). Saving files is the orchestrator's job.

## When asked to add code
- Ask before introducing a new *runtime* dependency. CI/dev tools listed
  in `requirements-dev.txt` are fine when the mission asks for them.
- Prefer reviewing my code over rewriting it when I ask for review.
- If I leave a `# TODO` comment with hints, do NOT fill it in unless
  I explicitly ask — those are deliberate learning blanks.
