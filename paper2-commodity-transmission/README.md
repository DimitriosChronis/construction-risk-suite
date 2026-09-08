# Paper 2: Global Commodity Transmission to European Construction Cost Inflation

**A Vine Copula Network Topology and System VARX Analysis**

![Status](https://img.shields.io/badge/Status-Accepted_--_In_Production_(CME)-brightgreen)

> Chronis, D. (2026). *Global Commodity Transmission to European Construction
> Cost Inflation: A Vine Copula Network Topology and System VARX Analysis.*
> **Construction Management and Economics (Taylor & Francis), in press.**
> DOI: [10.1080/01446193.2026.2732227](https://doi.org/10.1080/01446193.2026.2732227)

---

## Scope

- **Vine copula network topology** — a 10-D R-vine over five Greek construction
  price indices and five US commodity benchmarks (299 monthly obs, 2000–2024).
  US Fuel PPI is the dominant contemporaneous hub (vine-strength 0.464
  full-period; 0.776 post-COVID — directionally robust given the 60-month
  post-COVID sample). The post-COVID vine root shifts from US Cement PPI to
  US Brent.
- **System VARX transmission analysis** — all five Greek series jointly
  endogenous, the five US series exogenous. The system specification shows the
  bilateral framing **overstates** US Steel PPI's contribution to Greek steel
  cost variance (bivariate FEVD 22.2% vs. system FEVD 13.5% on the matched
  channel) and, more critically, that for Greek Fuel/Energy and PVC the
  matched US benchmark fails to Granger-cause the Greek series in the system
  (p = 0.757, p = 0.903) while the joint **cross-material** US block does
  (p < 0.001, p = 0.037).
- **Regime and robustness layers** — Chow structural breaks (4/5 pairs at
  COVID-19), regime-conditional FEVD with moving-block bootstrap CIs,
  Cholesky-ordering sensitivity, vine-conditional FEVD (US-explained variance
  3.11× higher in vine-identified joint-tail months), and cross-EU Granger
  robustness on Germany, France, Italy, Spain (7/8 pairs significant).

---

## Key Finding

A **two-layer transmission architecture**, with the system layer qualifying
the bilateral story:

| Layer | Method | Dominant variable | Mechanism |
|-------|--------|-------------------|-----------|
| Contemporaneous | R-vine copula | US Fuel PPI | Organises simultaneous co-movement across all commodities |
| Sequential | System VARX / Granger | US Steel PPI (matched); cross-material US block for Fuel & PVC | Persistent shocks via 1–4-month supply-chain delays |

For Greek Steel, Cement/Concrete and General/Brent the bilateral channel
remains directionally informative; for Greek Fuel/Energy and PVC the apparent
transmission operates through **cross-material spillovers**, which the
bilateral specification cannot identify.

---

## Scripts (run in order)

| # | Script | Purpose | Key output |
|---|--------|---------|------------|
| 01 | `01_global_data_download.py` | Download US PPI series from FRED API | `data/raw/global_commodities_monthly.csv` |
| 02 | `02_align_datasets.py` | Align ELSTAT + FRED, compute log-returns | `data/processed/aligned_log_returns.csv` |
| 03 | `03_vine_network_topology.py` | Fit 10-D R-vine copula, extract Tree 1 | `c1_vine_structure.csv` |
| 03b | `03b_network_centrality.py` | Vine-strength centrality by regime + bootstrap | `c1b_centrality_measures.csv` |
| 04 | `04_tail_concordance_lag.py` | Lag sweep (0–6M) + bivariate Granger tests | `c2_granger_causality.csv` |
| 04b | `04b_var_irf.py` | Bivariate VAR, orthogonalised IRF, FEVD (legacy spec) | `c2b_fevd_table.csv` |
| 04c | `04c_structural_break.py` | Chow structural break tests (COVID-19, Ukraine) | `c2c_structural_breaks.csv` |
| 04d | `04d_system_varx.py` | **System VARX**: joint FEVD + block-exogeneity tests | `c2d_system_fevd.csv`, `c2d_block_exogeneity.csv` |
| 04e | `04e_cholesky_sensitivity.py` | Reverse-ordering IRF sensitivity | `c2e_cholesky_sensitivity.csv` |
| 04f | `04f_regime_fevd_bootstrap.py` | Pre/post-COVID FEVD + moving-block bootstrap CIs | `c2d_regime_fevd_bootstrap_ci_*.csv` |
| 04g | `04g_copula_conditional_fevd.py` | Vine-conditional FEVD (tail vs calm months) | `c2g_vine_conditional_fevd.csv` |
| 05 | `05_cost_translation.py` | Legacy FEVD → EUR P95 contingency (textbook benchmark) | `c5_cost_translation.csv` |
| 05b | `05b_revised_contingency.py` | **Revised contingency**: system FEVD + block-bootstrap VaR, 4 specifications | `c2_5b_revised_contingency.csv` |
| 06 | `06_oos_forecast.py` | Rolling OOS forecast + Diebold–Mariano tests | `c6_oos_forecast.csv` |
| 08 | `08_publication_figures.py` | Core publication figures | `results/figures/` |
| 08b | `08b_revision_figures.py` | System-VARX / revision figures (fig_R1–R6) | `results/figures/` |
| 09 | `09_cross_eu_robustness.py` | Cross-EU Granger causality (DE, FR, IT, ES) | `c6_eu_granger.csv` |

---

## Key Outputs

### Tables (`results/tables/`)
| File | Description |
|------|-------------|
| `c1_vine_structure.csv` | Pair-copula families and parameters (all trees) |
| `c1b_centrality_measures.csv` | Vine-strength centrality by regime |
| `c2_granger_causality.csv` | Bivariate Granger p-values at lags 1–8M |
| `c2b_var_summary.csv` / `c2b_fevd_table.csv` | Bivariate VAR summaries and FEVD (legacy) |
| `c2c_structural_breaks.csv` | Chow F-statistics at COVID-19 and Ukraine breaks |
| `c2d_system_fevd.csv` | **System FEVD** — all five US shocks jointly |
| `c2d_block_exogeneity.csv` | Matched vs cross-material Granger in the system |
| `c2d_bivariate_vs_system_fevd.csv` | Bilateral overstatement comparison |
| `c2d_regime_fevd_bootstrap_ci_*.csv` | Pre/post-COVID FEVD bootstrap CIs |
| `c2e_cholesky_sensitivity.csv` | Reverse-ordering cumulative IRF comparison |
| `c2g_vine_conditional_fevd.csv` | Tail-conditional vs calm-month FEVD (3.11×) |
| `c2_5b_revised_contingency.csv` | EUR contingency under 4 specifications |
| `c2_5b_scaling_comparison.csv` | √H vs block-bootstrap VaR ratios |
| `c2_5b_portfolio_totals.csv` | Contingency totals (all specifications) |
| `c6_oos_forecast.csv` | OOS RMSE vs naive/AR + Diebold–Mariano p-values |
| `c6_eu_granger.csv` | Cross-EU Granger p-values (DE, FR, IT, ES) |

### Figures (`results/figures/`)
Core: `fig1_kendall_heatmap`, `fig_c1b_centrality`, `fig2_lag_heatmap`,
`fig3_rolling_tau`, `fig_c2b_irf_all`, `fig5_fevd_bar`,
`fig_c2c_structural_break_steel`, `fig7_eu_robustness`, `fig_c6_oos_forecast`.
Revision/system layer: `fig_R1_bivariate_vs_system_fevd`,
`fig_R2_block_exogeneity`, `fig_R3_cholesky_sensitivity`,
`fig_R4_regime_conditional_fevd`, `fig_R5_revised_contingency`,
`fig_R6_scaling_comparison`.

---

## Data Sources

| Series | Source | FRED Code | n |
|--------|--------|-----------|---|
| Greek General, Concrete, Steel, Fuel, PVC indices | [ELSTAT SPC23](https://www.statistics.gr/en/statistics/-/publication/SPC23/) | — | 300 monthly obs |
| US Brent Crude Oil | [FRED](https://fred.stlouisfed.org) | `DCOILBRENTEU` | 300 |
| US Steel PPI | FRED | `WPU101` | 300 |
| US Cement PPI | FRED | `WPU1321` | 300 |
| US PVC PPI (plastic pipe) | FRED | `WPU0721` | 300 |
| US Fuel PPI | FRED | `WPU0553` | 300 |
| DE/FR/IT/ES construction indices | FRED | `DEUPRCNTO01IXOBM` etc. | 291 |

**Aligned dataset:** 299 monthly log-returns × 10 series (February 2000 – December 2024)

ELSTAT data requires manual download. US and EU data are downloaded
automatically by `01_global_data_download.py`. See
[`../shared-data/README.md`](../shared-data/README.md) for full instructions.

---

## Global Parameters

```python
SEED           = 42
BASE_COST      = 2_300_000    # EUR reference project
HORIZON        = 24           # months
ROLLING_WINDOW = 24           # months
MAX_LAG        = 8            # VAR AIC selection
N_BOOT         = 5000         # contingency block bootstrap (05b)
BLOCK_LEN      = 6            # moving-block length, months (05b)
BOOTSTRAP_REPS = 200          # centrality stability (03b)
```

---

## Results Summary

### Network centrality (vine-strength)

| Variable | Full | Pre-COVID | Post-COVID |
|----------|------|-----------|------------|
| US Fuel PPI | **0.464** | 0.447 | **0.776** |
| US Steel PPI | 0.276 | 0.262 | 0.000 |
| US PVC PPI | 0.090 | 0.090 | 0.401 |
| US Cement PPI | 0.090 | 0.090 | 0.000 |
| US Brent | 0.000 | 0.000 | 0.093 |

Post-COVID values rest on 60 months and are directionally robust rather than
precise point estimates.

### System VARX: matched vs cross-material channels

| Greek series | System FEVD (matched) | System FEVD (cross-material) | Matched Granger p | Cross-material block p | Verdict |
|--------------|----------------------:|-----------------------------:|------------------:|-----------------------:|---------|
| Steel | 13.5% | 13.3% | 0.003 | 0.124 | bilateral informative |
| Concrete | 9.8% | 4.8% | <0.001 | 0.528 | bilateral informative |
| General/Brent | 12.8% | 13.6% | <0.001 | 0.0495 | cross-material matters |
| Fuel/Energy | 0.1% | 28.6% | 0.757 | <0.001 | **cross-material dominates** |
| PVC | 0.1% | 15.2% | 0.903 | 0.037 | **cross-material dominates** |

Regime-conditional FEVD: Greek Concrete's US-explained share rises 1.8% →
30.9% post-COVID (bootstrap intervals non-overlapping); Greek Steel's falls
15.4% → 7.9% (intervals overlap — directional only).

### EUR P95 contingency (EUR 2.3M reference project, H = 24M)

| Specification | Total |
|---------------|------:|
| (A) Bilateral FEVD + √H — textbook benchmark | EUR 45,174 (1.96%) |
| **(D) System FEVD + block-bootstrap VaR — preferred** | **EUR 28,510 (1.24%)** incl. cross-material, of which EUR 9,043 on the matched channel |

The difference is attributable in roughly equal measure to the system-FEVD
correction and to the serial-correlation-preserving bootstrap scaling.

### Cross-EU robustness (7/8 pairs significant)

| US Series | DE | FR | IT | ES |
|-----------|----|----|----|----|
| Steel PPI | ✓ p<0.001 | ✓ p=0.022 | ✓ p=0.004 | ✗ p=0.076 |
| Cement PPI | ✓ p<0.001 | ✓ p=0.040 | ✓ p<0.001 | ✓ p<0.001 |

---

## Procurement Decision Rules (as published)

| Signal | Lead | Action |
|--------|------|--------|
| US Cement PPI 3M momentum above its 80th percentile | 1M | Lock cement contracts |
| US Steel PPI — **regime-conditional rule**: activate only when composite US commodity volatility > P67 **and** US Steel PPI 3M momentum > P80 | 4M | Accelerate steel procurement (thresholds are illustrative heuristics, not out-of-sample-optimised) |
| US Brent 3M momentum above its 80th percentile | 1M | Review energy-intensive line items |

For Fuel/Energy and PVC no matched-channel rule is offered: their transmission
operates through cross-material spillovers, so portfolio-level monitoring of
the joint US block is recommended instead.

---

## Citation

```bibtex
@article{chronis2026transmission,
  author  = {Chronis, Dimitrios},
  title   = {Global Commodity Transmission to European Construction
             Cost Inflation: A Vine Copula Network Topology and
             System VARX Analysis},
  journal = {Construction Management and Economics},
  year    = {2026},
  doi     = {10.1080/01446193.2026.2732227},
  note    = {in press},
  url     = {https://github.com/DimitriosChronis/construction-risk-suite}
}
```
