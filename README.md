# Stock Sensitivity to Pure Volatility Shocks

Per-stock estimates of how nine large-cap equities respond to changes in VIX that are unrelated to the S&P 500's own move.

## Question

Do individual stocks have a stable exposure to volatility surprises once the market return is controlled for? The starting hypothesis was that these betas are negative (stocks fall when volatility rises beyond what the market move explains).

## Data

- Daily SPX, VIX, and equity prices for nine stocks (NVDA, GOOGL, AAPL, MSFT, WMT, XOM, KO, JNJ, JPM), 2006 to 2026
- Pulled from Interactive Brokers via `ib_async`
- The basket was hand-picked, so this is a nine-stock study and not a cross-sectional pricing test

## Method

**1. Isolate the volatility shock.** Regress the VIX change on the SPX return and keep the residual:

$$\Delta \ln \text{VIX}_t = a + b \, r^{SPX}_t + \varepsilon_t$$

The residual $\varepsilon_t$ is the "pure" volatility shock. Checks on the orthogonalization:

| Check | Result |
|---|---|
| $\text{corr}(\varepsilon_t, r^{SPX}_t)$ | $\approx 6 \times 10^{-16}$ |
| Slope $b$ | $-4.64$ |
| $R^2$ | $0.547$ |
| Frisch-Waugh-Lovell (JPM) | raw and orthogonalized coefficients both $0.00064250$ |

The FWL match is expected, since orthogonalizing is equivalent to controlling for $r^{SPX}_t$ directly. It confirms the pipeline is wired correctly.

**2. Per-stock regressions.** For each stock $i$:

$$r_{i,t} = \alpha_i + \beta_i \, r^{SPX}_t + \gamma_i \, \varepsilon_t + u_{i,t}$$

The object of interest is $\gamma_i$, reported as the effect of a 1-standard-deviation shock in basis points. Standard errors are HAC.

**3. Two shock definitions.** The shock is built from either the level change $\Delta \text{VIX}_t$ or the log change $\Delta \ln \text{VIX}_t$. The log version is the primary specification: level changes are dominated by 2008 and 2020, when VIX itself was very large, so the level regression is close to a crisis-only regression.

## Results

### Full sample: effect of a 1-SD volatility shock (bps)

| Stock | Level shock (bps) | $t$ | Log shock (bps) | $t$ |
|---|---:|---:|---:|---:|
| NVDA | -6.29 | -1.37 | -15.07 | -4.10 |
| GOOGL | 3.40 | 1.11 | -4.98 | -2.33 |
| AAPL | 3.68 | 0.98 | -2.15 | -0.99 |
| MSFT | 0.93 | 0.38 | -0.77 | -0.38 |
| WMT | -2.72 | -1.05 | 1.55 | 0.66 |
| XOM | -4.06 | -1.29 | 3.29 | 1.12 |
| KO | -1.85 | -0.63 | 3.39 | 1.48 |
| JNJ | 1.06 | 0.37 | 4.37 | 1.83 |
| JPM | 7.19 | 1.63 | 11.88 | 2.75 |

- Under the level shock, nothing is significant at the 5% level.
- Under the log shock, three names are: NVDA (negative), GOOGL (negative), JPM (positive).
- Five names change sign between specifications (GOOGL, AAPL, MSFT, WMT, XOM). None of those sign changes is statistically significant on either side.
- With nine tests, a Bonferroni cutoff is $p < 0.0056$ ($|t| \gtrsim 2.77$). NVDA clears it comfortably, JPM just misses it, and GOOGL does not clear it.

### Subperiod stability (log shock, bps and $t$-stats)

| Stock | 2006-12 | $t$ | 2013-19 | $t$ | 2020-26 | $t$ | Wald $p$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| KO | -3.15 | -0.82 | -3.79 | -1.63 | 13.95 | 3.34 | 0.001 |
| JNJ | 4.58 | 1.28 | 2.45 | 0.82 | 15.81 | 3.45 | 0.047 |
| AAPL | -8.00 | -1.41 | 4.75 | 1.34 | 1.78 | 0.46 | 0.160 |
| JPM | 17.40 | 1.49 | -1.38 | -0.53 | 4.58 | 1.04 | 0.178 |
| NVDA | -20.99 | -2.65 | -7.25 | -1.10 | -17.91 | -2.53 | 0.348 |
| MSFT | 0.46 | 0.11 | 7.14 | 2.28 | 3.90 | 1.02 | 0.433 |
| XOM | 0.01 | 0.00 | 0.28 | 0.11 | 7.85 | 1.36 | 0.473 |
| GOOGL | 2.62 | 0.53 | -2.29 | -0.88 | -3.29 | -0.69 | 0.628 |
| WMT | 0.48 | 0.11 | 2.96 | 1.05 | 5.16 | 1.23 | 0.751 |

The Wald $p$-value tests whether the volatility beta is equal across the three subperiods.

## Findings

1. **NVDA is the one robust negative exposure.** Negative in all three subperiods and highly significant in the full sample under the log shock.
2. **KO and JNJ turn positive after 2020.** Both are flat or negative before 2020 and strongly positive in 2020-26. KO's stability test survives the Bonferroni cutoff ($p = 0.001$). JNJ ($p = 0.047$) does not, so it is suggestive only. This looks like a regime effect and not a stable exposure.
3. **JPM's full-sample positive beta is fragile.** The subperiod signs move around and none is individually significant.
4. **The rest are indistinguishable from zero** once the market return is controlled for, and they are stable across subperiods.
5. **The original hypothesis of broadly negative vol betas is not supported.** Results depend meaningfully on how the shock is defined.

## Limitations

- Nine hand-picked stocks, so no cross-sectional inference (no Fama-MacBeth or portfolio sorts) and limited ability to generalize.
- HAC lag-length sensitivity has not yet been checked.
- Subperiod splits are chosen by calendar and not by estimated breaks, and each subperiod has fewer observations than the full sample.
- Findings 2 and 3 involve many comparisons (9 stocks, 3 subperiods, 2 shock definitions), so only the results that survive multiple-testing adjustment should be read as more than suggestive.

## Code

- `build_panel.py`: builds the SPX/VIX/equity panel
- `orthogonalize.py`: orthogonalizes the VIX change against SPX returns
- `per_stock_betas.py`: per-stock regressions with HAC standard errors

Stack: pandas, NumPy, statsmodels, matplotlib.
