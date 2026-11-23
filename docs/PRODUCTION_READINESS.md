# Production readiness audit (Phase 6)

Audit of the Market Briefing repository and hosted Streamlit app after Phases 1–5. Findings ranked by severity. **Fixed in this phase** are marked; remaining items are intentional limits or deferred work.

Date of audit: 23 Nov 2026 (America/Toronto). Build identity: `version_info.build_label()` (footer).

## Summary

| Severity | Found | Fixed this phase | Remaining |
|---|---:|---:|---:|
| Critical | 0 | 0 | 0 |
| High | 1 | 1 | 0 |
| Medium | 6 | 5 | 1 (mypy gradual) |
| Low | 8 | 3 | 5 |

Core risk math (cumulative return, log-vol × √252, Sharpe/Sortino, hist/parametric VaR, CVaR, Jensen α, rebalance look-ahead, diversification) matches `docs/METHODOLOGY.md` and independent spot-checks in `tests/test_production_readiness.py`.

---

## High

### H1. Yahoo downloads could hang Streamlit — **FIXED**
- **Evidence:** `yf.download` / search / snapshot had no deadline; only the optional LLM path used a timeout.
- **Fix:** `DOWNLOAD_TIMEOUT_SECONDS=30` (search/snapshot 15s) via `ThreadPoolExecutor` in `data_loader.py`. Timeouts become `MarketDataError` and fall through to cache/offline messaging.

---

## Medium

### M1. Share links omitted portfolio weights / cost — **FIXED**
- **Evidence:** README listed `w_*` params; widgets were unbound; `_share_query` skipped weights and `cost_bps`.
- **Fix:** Bind `w_{SYMBOL}` and include unequal weights + non-default `cost_bps` in the share link.

### M2. Failed Refresh with fresh cache looked successful — **FIXED**
- **Evidence:** Download failure used `allow_expired` cache but `stale` was only set when TTL expired.
- **Fix:** Any download-failure fallback sets `stale=True`; UI warning covers expired **or** failed-refresh cases.

### M3. Intra-symbol OHLC NaNs were silently ffill/bfill — **FIXED**
- **Evidence:** `_validate` forward/back-filled gaps, inventing prices inside one series.
- **Fix:** Drop rows with any NaN in price columns; record `dropped_gap_rows` on attrs. Cross-symbol joins remain inner joins (no fill).

### M4. No runtime version stamp — **FIXED**
- **Evidence:** Impossible to tell if Community Cloud was on latest `main` after a silent non-redeploy.
- **Fix:** Footer caption `Build v0.6.0 · <sha>` from `version_info.py`, with reboot guidance when SHA is stale.

### M5. Deploy docs pointed at an old preview branch — **FIXED**
- **Evidence:** README still named `cursor/portfolio-and-fundamentals-5ee9`.
- **Fix:** Publish/redeploy section now targets `main` and documents the **reboot** step when auto-redeploy skips a push.

### M6. mypy excludes `features.py` / `app.py` — **REMAINS**
- Gradual typing deliberately covers `analyst`, `data_loader`, `plots`, `dashboard` first. Extending to `features`/`app` is safe follow-up but large; left open.

---

## Low (selected)

| ID | Topic | Status |
|---|---|---|
| L1 | `diversification_stats` docstring said `/σ_p` instead of `/σ_p²` | **FIXED** |
| L2 | `prefers-reduced-motion` not respected | **FIXED** (CSS) |
| L3 | Plotly charts lack text alternatives beyond captions | Remains (captions exist; full alt summaries deferred) |
| L4 | No explicit “delisted” label when Yahoo still returns history | Remains |
| L5 | Cold start still pulls `period=max` serially | Remains (timeouts mitigate hangs; window-cap deferred) |
| L6 | Focus change may rebuild a full briefing | Remains |
| L7 | LLM `urlopen` follows redirects after https check | Remains (base URL not user-settable) |
| L8 | Contrast of muted `#5c6b7a` on white is borderline for tiny text | Remains (documented) |

---

## Error handling & data quality (spot check)

| Case | Behavior |
|---|---|
| Unknown / empty Yahoo ticker | Clear error; empty frames never cached; offline demo offered |
| Partial multi-symbol failure | Failed symbols listed; rest of briefing continues |
| One-day range | `AnalysisError` with widen-range message |
| Portfolio with one symbol | Callout; mix withheld |
| Offline missing symbols | Drop + notice; fallback to AAPL when nothing remains |
| Short windows | Sharpe/VaR/etc. show `—` with help text |
| Yahoo outage with disk cache | Stale history served and labelled |
| Yahoo outage, no cache | Hard error |

---

## Security

- No committed secrets; `.streamlit/secrets.toml` gitignored.
- LLM key is session-only password input; blank keys stripped; never URL-bound.
- Symbol shape validated; HTML paths use `html.escape`.
- CI: `pip-audit` on `requirements.txt`, `bandit` on app modules (free).

---

## Accessibility / mobile

- Responsive CSS at 900px / 640px; comparison table scrolls horizontally.
- Metric help tooltips; section nav; reduced-motion preference honored for scroll-behavior.
- Remaining: richer chart text alternatives; formal contrast audit.

---

## Deployment (Streamlit Community Cloud)

1. App tracks branch **`main`**, entry `app.py`, Python **3.12**, secrets empty.
2. A push *usually* redeploys, but **not always**. After merge, open the app → footer SHA.
3. If the SHA is behind `main`, use **Reboot app** (or Redeploy) in the Streamlit Cloud dashboard.
4. Smoke: Offline demo → Max → Portfolio AAPL+MSFT → Narrative analyst sentence; Live AAPL/SPY 1Y cards load or show a clear Yahoo error.

---

## What remains (not blocking)

- Cap Yahoo history to the requested window + MA warmup; optional parallel fetch.
- Extend mypy to `features.py` / `app.py`.
- Delisted / halted labeling when quote metadata is available.
- Richer a11y text for Plotly figures.

---

## Verification

- Unittest suite including `tests/test_production_readiness.py`.
- CI jobs: unittest, ruff + mypy, pip-audit + bandit.
- Live: reboot Cloud app and confirm footer SHA matches merged `main`.
