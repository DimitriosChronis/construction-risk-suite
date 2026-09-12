# Paper 6 — State-Dependent Cross-Border Co-Movement of European Construction Costs

**A Copula Network and Spillover Analysis of Eight Euro-Area Countries, 2000–2026**

![Status](https://img.shields.io/badge/Status-Under_Review-yellow)

> Chronis, D. (2026). *State-Dependent Cross-Border Co-Movement of
> European Construction Costs: A Copula Network and Spillover Analysis
> of Eight Euro-Area Countries, 2000–2026.* Manuscript under review.
> Journal details are withheld during active peer review.

This folder contains the complete, reproducible analysis pipeline.
**Code and result tables only** — raw data are re-downloadable from
the public sources below; the manuscript is not distributed here.

Builds on two published companion papers:
[10.1080/01446193.2026.2732227](https://doi.org/10.1080/01446193.2026.2732227)
(CME) and
[10.1061/JCCEE5/CPENG-8417](https://doi.org/10.1061/JCCEE5/CPENG-8417)
(ASCE JCCE).

---

## What the pipeline does

A balanced monthly panel of residential construction-cost indices for
eight euro-area countries — Greece, Spain, Italy, Portugal
(Mediterranean) and the Netherlands, Ireland, Finland, Lithuania
(Northern) — January 2000 to July 2026, analysed in four layers:

1. **Rolling pair-Gumbel copula network** (windows 24/36M, tau
   inversion, MLE-validated) with crisis-vs-stable amplification
   under four pre-registered exogenous EU crisis windows and
   leak-free (point-in-time) endogenous volatility labels.
2. **Directional layer**: pairwise Granger network (BH-FDR
   controlled) + order-invariant generalized Diebold–Yilmaz
   spillovers.
3. **Clustering** (Mediterranean vs Northern partition test) and a
   **panel transmission regression** of US material-price volatility
   into national cost growth (country/month FE, cluster SEs, wild
   cluster bootstrap).
4. **Robustness**: quarterly 10-country layer incl. DE/FR, COST
   indicator subset, extended (+NO, +PL) set, label-definition sweep,
   deseasonalisation, Greek data-source swap, copula-family and
   estimator cross-validation, block-length sweep.

## Scripts (run in order)

| # | Script | Purpose |
|---|--------|---------|
| 01 | `01_eurostat_cci.py` | Ingest: Eurostat API (monthly + quarterly) + ELSTAT |
| 02 | `02_harmonise_panel.py` | Balanced panels + coverage report |
| 03 | `03_volatility_features.py` | Rolling volatility + leak-free labels |
| 03b | `03b_bridge_validation.py` | Greek micro–macro bridge checks |
| 04 | `04_cross_country_topology.py` | Rolling pair-copula network (H1) |
| 04b | `04b_h1_subgroups.py` | Bloc subgroups + endogenous labels |
| 04c | `04c_h1_episodes.py` | Per-episode analysis + bootstrap CIs |
| 05 | `05_granger_network.py` | Granger network + generalized DY |
| 06 | `06_clustering.py` | Hierarchical clustering test |
| 07 | `07_ukraine_case_study.py` | 2022 energy-shock case study |
| 08 | `08_cross_country_panel.py` | Panel FE transmission regression |
| 08b | `08b_h4_wildboot.py` | Wild cluster bootstrap |
| 10 | `10_quarterly_robustness.py` | Quarterly layer (adds DE, FR) |
| 11 | `11_indicator_robustness.py` | Indicator / country-set robustness |
| 12 | `12_copula_families.py` | Copula-family + estimator validation |
| 13 | `13_extended_robustness.py` | Label sweep, seasonality, GR source |
| 09 | `09_publication_figures.py` | Figures (PDF + EPS + PNG) |
| 97–99 | `9*_*.py` | Digest + consistency harnesses |

**One-shot driver:**

```bash
cd src
python run_all.py                 # full pipeline (~3 min end-to-end)
python run_all.py --skip-download # reuse cached raw data
```

## Data sources (all public)

| Source | Series | Access |
|--------|--------|--------|
| Eurostat `sts_copi_m` / `sts_copi_q` | Residential construction producer-price / cost indices | keyless REST API (automated) |
| ELSTAT | Material Cost Index for New Residential Buildings (Greece) | manual download from statistics.gr → `data/raw/elstat_spc23_*.xls` |
| FRED | US PPI series: Brent, steel, cement, fuel, PVC | `pandas-datareader` (automated) |

Raw and processed data are not committed; the ingest scripts rebuild
them. Result tables (CSV) under `results/tables/` are versioned so
every number in the manuscript can be traced.

## Dependencies

```bash
pip install -r requirements.txt
```

## License

MIT License.
