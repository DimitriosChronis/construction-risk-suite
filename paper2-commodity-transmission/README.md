# Paper 2: Global Commodity Transmission to European Construction Cost Inflation

**A Vine Copula Network Topology and VAR-IRF Analysis**

> Chronis, D. (2026). *Global Commodity Transmission to European Construction Cost Inflation: A Vine Copula Network Topology and VAR-IRF Analysis*. Submitted to *Construction Management and Economics*, Taylor & Francis.

**Contributions: C1 + C2 + C6 (cross-EU robustness)**

---

## Scope

- **C1** — 10-D R-vine copula network topology of US–Greek construction cost linkages. US Fuel PPI identified as dominant hub (vine-strength 0.464 full-period; 0.776 post-COVID). Post-COVID vine root shifts from US Cement PPI to US Brent.
- **C2** — Transmission mechanism: VAR/IRF/FEVD shock propagation + Chow structural break tests. Steel Granger-causes Greek costs at 4-month lag (FEVD 22.2% at 12M); Cement at 1-month lag (FEVD 7.3%).
- **C6** — Cross-EU robustness: US Steel PPI and Cement PPI tested against Germany, France, Italy, and Spain. 7 of 8 pairs pass Granger significance.

---

## Key Finding

This paper reveals a **two-layer transmission architecture**:

| Layer | Method | Dominant variable | Mechanism |
|-------|--------|-------------------|-----------|
| Contemporaneous | R-vine copula | US Fuel PPI | Organises simultaneous co-movement across all commodities |
| Sequential | VAR/Granger | US Steel PPI | Transmits persistent shocks via 4-month supply-chain delay |

These are complementary findings, not contradictory — Fuel organises the network topology while Steel drives the causal procurement signal.

---

## Scripts (run in order)

| # | Script | Purpose | Key output |
|---|--------|---------|------------|
| 01 | `01_global_data_download.py` | Download US PPI series from FRED API | `data/raw/global_commodities_monthly.csv` |
| 02 | `02_align_datasets.py` | Align ELSTAT + FRED, compute log-returns | `data/processed/aligned_log_returns.csv` |
| 03 | `03_vine_network_topology.py` | Fit 10-D R-vine copula, extract Tree 1 | `results/tables/c1_vine_structure.csv` |
| 03b | `03b_network_centrality.py` | Vine-strength centrality by regime + bootstrap (B=200) | `results/tables/c1b_centrality.csv` |
| 04 | `04_tail_concordance_lag.py` | Lag sweep (0–6M) + Granger causality tests | `results/tables/c2_granger.csv` |
| 04b | `04b_var_irf.py` | Bivariate VAR, orthogonalised IRF, FEVD | `results/tables/c2b_fevd_table.csv` |
| 04c | `04c_structural_break.py` | Chow structural break tests (COVID-19, Ukraine) | `results/tables/c2c_structural_breaks.csv` |
| 05 | `05_cost_translation.py` | FEVD → EUR P95 contingency + decision rules | `results/tables/c5_eur_contingency.csv` |
| 06 | `06_oos_forecast.py` | Rolling OOS forecast + Diebold-Mariano tests | `results/tables/c6_oos_forecast.csv` |
| 08 | `08_publication_figures.py` | All publication-quality figures | `results/figures/` |
| 09 | `09_cross_eu_robustness.py` | Cross-EU Granger causality (DE, FR, IT, ES) | `results/tables/c6_eu_granger.csv` |

---

## Key Outputs

### Tables
| File | Description |
|------|-------------|
| `c1_vine_structure.csv` | 45 pair-copula families and parameters (all 9 trees) |
| `c1b_centrality.csv` | Vine-strength centrality by regime (full, pre-COVID, post-COVID) |
| `c2_granger.csv` | Granger causality p-values at lags 1–8M for all 5 pairs |
| `c2b_var_summary.csv` | VAR model summaries (optimal lag, AIC) |
| `c2b_fevd_table.csv` | FEVD at 1, 6, 12-month horizons for all pairs |
| `c2c_structural_breaks.csv` | Chow F-statistics and Δρ at COVID-19 and Ukraine breaks |
| `c5_eur_contingency.csv` | P95 EUR contingency by material for EUR 2.3M reference project |
| `c6_oos_forecast.csv` | OOS RMSE vs naive and AR benchmarks + Diebold-Mariano p-values |
| `c6_eu_granger.csv` | Cross-EU Granger p-values (DE, FR, IT, ES) |

### Figures
| File | Description |
|------|-------------|
| `fig1_kendall_heatmap.pdf` | 10×10 Kendall τ matrix (full sample) |
| `fig_c1b_centrality.pdf` | Vine-strength centrality by regime |
| `fig2_lag_heatmap.pdf` | Spearman cross-correlation at lags 0–6M |
| `fig3_rolling_tau.pdf` | Rolling 24M Kendall τ with COVID/Ukraine markers |
| `fig_c2b_irf_all.pdf` | Orthogonalised IRF for all 5 pairs (95% bootstrap CIs) |
| `fig5_fevd_bar.pdf` | FEVD bar chart at 1, 6, 12-month horizons |
| `fig_c2c_structural_break_steel.pdf` | Steel rolling correlation with Chow break dates |
| `fig7_eu_robustness.pdf` | Cross-EU Granger −log₁₀(p) values |
| `fig_c5_eur_contingency.pdf` | EUR P95 contingency by material |
| `fig_c6_oos_forecast.pdf` | OOS forecast vs naive and AR benchmarks |

---

## Data Sources

| Series | Source | FRED Code | n |
|--------|--------|-----------|---|
| Greek General, Concrete, Steel, Fuel, PVC indices | [ELSTAT SPC23](https://www.statistics.gr/en/statistics/-/publication/SPC23/) | — | 300 monthly obs |
| US Brent Crude Oil | [FRED](https://fred.stlouisfed.org) | `DCOILBRENTEU` | 300 |
| US Steel PPI | FRED | `WPU101` | 300 |
| US Cement PPI | FRED | `WPU1321` | 300 |
| US PVC PPI | FRED | `WPU0721` | 300 |
| US Fuel PPI | FRED | `WPU0553` | 300 |
| DE/FR/IT/ES construction indices | FRED | `DEUPRCNTO01IXOBM` etc. | 291 |

**Aligned dataset:** 299 monthly log-returns × 10 series (February 2000 – December 2024)

ELSTAT data requires manual download. US and EU data are downloaded automatically by `01_global_data_download.py`. See [`../shared-data/README.md`](../shared-data/README.md) for full instructions.

---

## Global Parameters

```python
SEED           = 42
BASE_COST      = 2_300_000    # EUR
HORIZON        = 24           # months
ROLLING_WINDOW = 24           # months
BOOTSTRAP_REPS = 200          # centrality stability
MAX_LAG        = 8            # VAR AIC selection
```

---

## Results Summary

### C1 — Network Centrality

| Variable | Full | Pre-COVID | Post-COVID |
|----------|------|-----------|------------|
| US Fuel PPI | **0.464** | 0.447 | **0.776** |
| US Steel PPI | 0.276 | 0.262 | 0.000 |
| US PVC PPI | 0.090 | 0.090 | 0.401 |
| US Cement PPI | 0.090 | 0.090 | 0.000 |
| US Brent | 0.000 | 0.000 | 0.093 |

### C2 — Transmission (FEVD at 12 months)

| Pair | Granger p | Lead | FEVD 12M | EUR Contingency |
|------|-----------|------|----------|-----------------|
| Steel | 0.011* | 4M | 22.2% | EUR 22,779 |
| Brent/General | <0.001*** | 1M | 12.1% | EUR 25,388 |
| Cement | <0.001*** | 1M | 7.3% | EUR 5,102 |
| PVC | 0.061 | — | 4.9% | EUR 5,039 |
| Fuel/Energy | 0.125 | — | 2.9% | EUR 12,254 |
| **Total** | | | | **EUR 45,174** |

### C6 — Cross-EU (7/8 pairs significant)

| US Series | DE | FR | IT | ES |
|-----------|----|----|----|----|
| Steel PPI | ✓ p<0.001 | ✓ p=0.022 | ✓ p=0.004 | ✗ p=0.076 |
| Cement PPI | ✓ p<0.001 | ✓ p=0.040 | ✓ p<0.001 | ✓ p<0.001 |

---

## Procurement Decision Rules

| Signal | Threshold | Lead | Action | Contingency |
|--------|-----------|------|--------|-------------|
| US Steel PPI 3M MA | > 2.22% (80th pctile) | 4M | Accelerate steel procurement | EUR 22,779 |
| US Cement PPI 3M MA | > 0.60% (80th pctile) | 1M | Lock cement contracts | EUR 5,102 |
| US Brent 3M MA | > 8.02% (80th pctile) | 1M | Review energy-intensive items | EUR 25,388 |

---

## Citation

```bibtex
@article{chronis2026paper2,
  author  = {Chronis, Dimitrios},
  title   = {Global Commodity Transmission to European Construction
             Cost Inflation: A Vine Copula Network Topology and
             VAR-IRF Analysis},
  journal = {Construction Management and Economics},
  year    = {2026},
  note    = {under review},
  url     = {https://github.com/dimitrioschronis/construction-risk-suite}
}
```