# Market Dashboard — Project Conventions

## Architecture
Modular layers, one file per layer, one-directional data flow:

    data_loader.py  →  features.py  →  plots.py  →  dashboard.py

Never bypass a layer. `plots.py` does not read CSV. `features.py` does
not call matplotlib. `dashboard.py` is the only orchestrator and is
the only place that runs end-to-end.

## Style
- Small, single-purpose functions. One concept per function.
- Trading / ML reasoning lives in comments BEFORE the code, not after.
- No magic numbers — use named constants (TRADING_DAYS = 252, etc.).
- Self-test at the bottom of every module: `if __name__ == "__main__"`.

## Data contracts between layers
- `load_prices()` returns: DataFrame with DatetimeIndex and columns
  open/high/low/close/volume. Zero NaNs. attrs['ticker'] set.
- `build_features()` adds: log_return, volatility_{N}d, momentum_{N}d.
  Early NaNs are expected and intentional (rolling-window warmup).
  attrs['vol_window'], attrs['mom_window'], attrs['ticker'] set.
- `build_dashboard()` returns: a matplotlib Figure. No side effects
  besides drawing; saving is the orchestrator's job.

## When asked to add code
- Ask before introducing a new dependency.
- Prefer reviewing my code over rewriting it when I ask for review.
- If I leave a `# TODO` comment with hints, do NOT fill it in unless
  I explicitly ask — those are deliberate learning blanks.
