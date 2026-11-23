# Methodology

Plain-English definitions, formulas, assumptions, limitations, and data-quality notes for every figure the Market Briefing computes. Constants match `features.py`.

## Shared conventions

| Symbol | Meaning | Default / constant |
|---|---|---|
| \(P_t\) | Adjusted close on session \(t\) | Yahoo `auto_adjust=True` / offline CSV |
| \(r_t\) | Daily simple return \(P_t / P_{t-1} - 1\) | Used by Sharpe, Sortino, beta, VaR, portfolio |
| \(\ell_t\) | Daily log return \(\ln(P_t / P_{t-1})\) | Used by annualized volatility |
| \(N\) | Trading days per year | `TRADING_DAYS = 252` |
| \(R_f\) | Annual risk-free rate (sidebar) | `DEFAULT_RISK_FREE = 0` |
| \(r_f\) | Daily risk-free \((1+R_f)^{1/252}-1\) | |
| Shared dates | Inner join of calendars | No forward fill, no back fill |

## Return and risk

**Cumulative return.** \(P_T / P_0 - 1\).

**Annualized return (CAGR).** \((1 + R_{\mathrm{cum}})^{365.25 / \Delta_{\mathrm{cal}}} - 1\). Withheld when calendar span &lt; 360 days.

**Annualized volatility.** \(\mathrm{std}(\ell_t;\, ddof=1) \times \sqrt{252}\). Needs ≥ 2 daily log returns.

**Maximum drawdown.** Deepest \(P_t / \max_{s\le t} P_s - 1\). Peak is the last session at that running peak on or before the trough.

- **Days to trough:** trading sessions from peak to trough.
- **Days to recovery:** sessions from trough to the first later close at or above the peak price; omitted if unrecovered.
- **Longest underwater:** longest run of sessions with price below the running peak.

## Risk-adjusted returns

**Sharpe.** \(\sqrt{252}\cdot \overline{r_t - r_f} / \mathrm{std}(r_t - r_f;\, ddof=1)\). Needs ≥ 20 daily returns.

**Sortino.** Same numerator; denominator is downside deviation \(\sqrt{\mathrm{mean}(\min(r_t-r_f,0)^2)}\) with every day in the divisor.

**Rolling Sharpe / Sortino.** Same formulas on a trailing 63-day window (`ROLLING_SHARPE_WINDOW`).

## Value at Risk family

All VaR figures need ≥ 20 daily simple returns and are reported as **positive loss fractions**.

**Historical 95% VaR.** \(-\,q_{0.05}(r_t)\), floored at 0.

**Parametric (Gaussian) 95% VaR.** \(-(\mu + z_{0.05}\sigma)\), floored at 0, with sample \(\mu,\sigma\) (\(ddof=1\)) and \(z_{0.05}\) from the standard normal inverse CDF.

**Conditional VaR / expected shortfall (95%).** \(-\,\mathrm{mean}\{r_t : r_t \le q_{0.05}\}\), floored at 0.

## Against the benchmark

Dates are an **inner join**. Gap = stock cumulative return − benchmark cumulative return on shared sessions (percentage points). **Not** regression alpha.

**Beta.** \(\mathrm{Cov}(r^s, r^b) / \mathrm{Var}(r^b)\) on shared daily simple returns (≥ 20).

**Jensen's alpha (annualised).** Daily \(\alpha = \overline{r^s - r_f} - \beta\,\overline{r^b - r_f}\), then \(\alpha_{\mathrm{ann}} = 252 \cdot \alpha\).

**Correlation.** Pearson of the same pairs. **Up / down capture** use arithmetic means on up/down benchmark days (≥ 10 each side).

**Rolling beta / correlation.** Trailing 63 shared daily returns.

## Portfolio construction

**Weights.** Fractions summing to 1. Missing → equal weight. Zeros dropped. Negatives rejected. Non-unit sums renormalized with a note.

**Look-ahead rule (enforced in `portfolio_backtest`).**

1. Holdings for day \(t\) are fixed at \(t-1\) (or at the prior rebalance).
2. The return from \(t-1\) to \(t\) always uses those **pre-trade** weights.
3. A month/quarter change on day \(t\) resets weights **after** that mark, at day-\(t\) closes, so the new mix first earns \(t \to t+1\).
4. Shared asset dates are an inner join; missing sessions are dropped, never filled forward.

**Transaction costs.** `cost_bps` on turnover at inception and each reset. Turnover = \(\tfrac12\|w_{\mathrm{target}} - w_{\mathrm{pre}}\|_1\). Value is scaled by \(1 - \mathrm{turnover}\times\mathrm{bps}/10\,000\). Buy-and-hold charges only the opening trade (turnover 1.0).

## Diversification

Built from the sample covariance of daily simple returns on shared sessions (≥ 20), with portfolio weights \(w\).

- **Diversification ratio.** \((\sum_i w_i \sigma_i) / \sigma_p\).
- **Effective N.** \(1 / \sum_i w_i^2\).
- **Risk contribution.** Share of variance \(w_i (\Sigma w)_i / \sigma_p^2\).

The correlation heatmap is the Pearson matrix of the same return columns.

## Assumptions

- One price series per symbol: adjusted close. Corporate actions follow Yahoo’s adjustment.
- The risk-free rate is a constant you set, not a bill curve.
- Rebalance decisions use the calendar period of the trading date; execution is at that day’s close after mark-to-market.
- Costs are proportional to turnover only — no bid–ask, slippage schedule, taxes, or cash drag.
- Long-only weights. No leverage, shorts, or options.

## Limitations and data quality

- Yahoo Finance is best-effort and unauthenticated. Empty answers are not cached; cold starts may retry then fall back to a 12-hour disk cache when present.
- Streamlit Community Cloud wipes local disk on restart, so the cache only covers symbols loaded since the last restart.
- Daily (not monthly) returns mean Sharpe/Sortino/beta/capture will not match Portfolio Visualizer or fund fact sheets for the same window.
- Historical and parametric VaR assume the sample window is representative; Gaussian VaR understates fat tails.
- CVaR here is historical expected shortfall of daily returns, not a multi-day horizon loss.
- Offline AAPL is a 2017-vintage Plotly file; other offline files were snapshotted later — adjustments may not align across symbols.
- Company snapshot fields come from yfinance `info`; blanks stay blank. Offline snapshots are labelled placeholders.
- Correlation and diversification need overlapping calendars; a thin overlap shrinks the sample.
- Look-ahead tests live in `tests/test_advanced_quant.py` (`LookAheadGuardTests`). They fail if rebalance-day returns use post-trade weights or if missing dates are forward-filled.

## Where the same text appears

- On-page expander: `methodology_notes()` in `features.py`.
- README section **How the figures are defined**.
- This file: full formulas, look-ahead rules, and data-quality notes.
