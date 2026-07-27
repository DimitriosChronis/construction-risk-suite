# Paper 5 — From Public Procurement Data to Procurement Triggers

**An Automated Pipeline for Portfolio-Level Construction Cost Risk
Monitoring across Public Infrastructure Projects**

> Chronis, D. (2026). *From Public Procurement Data to Procurement
> Triggers: An Automated Pipeline for Portfolio-Level Construction Cost
> Risk Monitoring across Public Infrastructure Projects*.
> Submitted to *Automation in Construction*.

This folder contains the complete, reproducible analysis pipeline for
the paper. **Code and result tables only** — raw/processed data and the
manuscript are not distributed here (the data are re-downloadable from
the public sources below).

---

## What the paper does

A four-layer framework that runs directly on public procurement
records and material price indices:

1. **Ingestion** — 1,310 competitively-awarded Greek public-construction
   contracts (Diavgeia platform, Δ.2.2 awards, 2014–2024, total budget
   €4.51B), classified into road / bridge / pipeline / building / other,
   plus ELSTAT material price indices (concrete, steel, fuel, PVC).
2. **Dependence analytics** — rolling 24-month pair-Gumbel copula
   network of type-level cost returns; directional lead–lag
   source/receiver decomposition; portfolio Expected Shortfall
   (α = 0.95) with an explicit diversification-benefit metric;
   dual-target LSTM crisis classifier; HMM regime model.
3. **Aggregation** — Composite Systemic Risk Index (CSRI): four
   components (mean λ_U, mean contagion index, max contagion index,
   1 − diversification benefit), each standardised by a trailing
   60-month rolling z-score, equal 1/4 weights.
4. **Decision output** — per-project contingency rules by type ×
   duration, procurement-timing prioritisation for source types, and
   +1σ index alerts.

**Public data only.** No internal, proprietary, or commercially
sensitive project data are used anywhere in this work.

---

## Headline results (all reproducible from this folder)

- Mean upper tail dependence amplifies **+8.1%** in crisis regimes
  (λ_U 0.853 → 0.922; permutation p < 0.001; 1,000-block-bootstrap CI).
- Portfolio ES₉₅ rises **1.50% → 2.75%** (1.83×) and the
  diversification benefit collapses toward zero in crisis.
- **Roads and pipelines are net systemic sources** of cost contagion;
  buildings and other works are receivers. (The bridge category has
  n = 1 project in-sample; its classification carries an explicit
  "interpret with caution" caveat.)
- The CSRI registers the two **commodity/energy-driven** stress periods
  (2008–09 GFC peak +1.71σ; above +1σ from Nov-2021, before the
  Feb-2022 invasion of Ukraine) while staying quiet in the 2012 and
  2015 **fiscal-liquidity** crises — construction-cost systemic stress
  is empirically distinct from macro-financial stress.
- Dual-target LSTM evaluation: exogenous macro-calendar target
  **AUC = 0.744** out-of-sample; the endogenous volatility-derived
  target yields a spuriously perfect AUC via circular-label leakage
  (documented as a methodological warning).
- Counterfactual: contingency budgets sized on stable-period
  diversification under-provision by **€19.5M (+1.09%)** over the
  crisis-period portfolio.
- Sensitivity: source/receiver roles stable for tail thresholds
  q ∈ {0.85, 0.90}; CSRI robust to component re-weighting
  (corr ≥ 0.97 vs equal weights, incl. PCA-derived weights).

---

## Data sources

| Source | Data |
|--------|------|
| [Diavgeia](https://diavgeia.gov.gr/) | Greek public contract awards (Δ.2.2, ΚΑΤΑΚΥΡΩΣΗ) |
| ELSTAT SPC23 | Monthly material price indices, 2001–2025 |
| FRED | US PPI commodity series (LSTM features) |
| [OpenTender](https://opentender.eu/) | EU procurement archive (external validation) |
| World Bank IEG | Infrastructure project ratings (external validation) |
| Eurostat/OECD | 13-country construction-cost panel (external validation) |

---

## Scripts (run in order)

| # | Script | Purpose |
|---|--------|---------|
| 01 | `01_diavgeia_download.py` | Download Δ.2.2 awards from the Diavgeia API |
| 02 | `02_portfolio_construction.py` | Classification, material-share weights, type-level return panel |
| 03 | `03_dynamic_copula.py` | Rolling 24M Gumbel λ_U network + regime labels |
| 04 | `04_contagion_index.py` | Contagion index + directional lead–lag OUT/IN/NET flows |
| 05 | `05_portfolio_es.py` | Portfolio ES (α=0.95), diversification benefit, contingency |
| 06 | `06_lstm_portfolio_agent.py` | Dual-target LSTM ensemble + logistic/XGBoost benchmarks |
| 07 | `07_systemic_risk_index.py` | CSRI (4 components, rolling 60M z), Granger, block bootstrap |
| 08 | `08_publication_figures.py` | Publication figures |
| 09 | `09_robustness.py` | Copula family / threshold / weights / permutation checks |
| 10 | `10_rvine.py` | R-vine comparison |
| 11 | `11_benchmarks.py` | DCC-GARCH + Diebold–Yilmaz benchmarks |
| 12 | `12_lead_time_curve.py` | LSTM AUC vs forecast horizon |
| 13 | `13_case_studies.py` | Crisis case studies + counterfactual €-gap + contingency rules |
| 14 | `14_validation_ted.py` | EU OpenTender external check |
| 15 | `15_validation_worldbank.py` | World Bank IEG external check |
| 16 | `16_eurostat_validation.py` | EU/OECD construction-cost validation |
| 17 | `17_panel_regression.py` | 13-country panel regression (lagged volatility) |
| 18 | `18_hmm_csri.py` | HMM regime classification of the CSRI |
| 19 | `19_aic_framework_figure.py` | Pipeline overview figure |
| 20 | `20_sensitivity_analysis.py` | Tail-threshold (q) + CSRI weight sensitivity |

Key parameters (fixed across the pipeline):

```python
SEED           = 42
COPULA_WINDOW  = 24      # months, rolling
TAIL_Q         = 0.90    # lead–lag joint-exceedance quantile
CRISIS_PCT     = 0.75    # endogenous regime threshold (P75)
ALPHA          = 0.95    # Expected Shortfall level
ZSCORE_WIN     = 60      # CSRI rolling z-score window
N_BOOTSTRAP    = 1000    # block bootstrap (block = 12)
# LSTM: hidden=64, layers=2, dropout=0.3, lr=1e-3, batch=16,
#       epochs=150 (no early stopping), lookback=6, lead=2,
#       ensemble seeds [0,1,2,3,4], walk-forward min_train=60, step=6
```

---

## Reproducibility

```bash
pip install -r requirements.txt   # or: conda env create -f environment.yml
cd src
python 01_diavgeia_download.py    # then 02 ... 20 in order
pytest ../tests -v
```

Every table and figure in the manuscript is generated by these scripts;
result tables are versioned under `results/` (figures and processed
data are regenerated locally and not versioned).

---

## Citation

```bibtex
@article{chronis2026procurement,
  author  = {Chronis, Dimitrios},
  title   = {From Public Procurement Data to Procurement Triggers:
             An Automated Pipeline for Portfolio-Level Construction
             Cost Risk Monitoring across Public Infrastructure
             Projects},
  journal = {Automation in Construction},
  year    = {2026},
  note    = {submitted}
}
```
