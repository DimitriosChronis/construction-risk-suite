# Quantifying Tail Dependence in Construction Cost Overruns

> **A fully reproducible Monte Carlo simulation framework using Gumbel Copulas and R-Vine Copulas to quantify systemic tail risk in construction project portfolios.**

![Status](https://img.shields.io/badge/Status-Ready_for_Submission-yellow)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Scripts](https://img.shields.io/badge/Scripts-12-orange)

---

## Paper

**"From Statistical Error to Profit Erosion: Quantifying Tail Dependence in Construction Cost Overruns"**
Dimitrios Chronis - School of Civil Engineering, NTUA

---

## Executive Summary

Standard industry methods (PERT, Independent Monte Carlo) underestimate extreme downside risks by assuming asset independence or Gaussian dependence. During systemic crises (2008, 2021-2022 Energy Shock), this produces systematic capital shortfalls.

This repository implements a **Project Lifecycle Risk Model** that:

1. **Rejects Normality** via Jarque-Bera tests on log-returns of 5 Greek ELSTAT construction price indices
2. **Models Tail Dependence** using Gumbel Copulas (Marshall-Olkin / Kanter sampler) and Kendall's tau
3. **Validates Externally** against the ENR 20-city Construction Cost Index (4/5 series significant)
4. **Quantifies Bootstrap Uncertainty** - 1,000-resample percentile CI on the hidden risk gap
5. **Cross-checks via R-Vine** - 4D pyvinecopulib vine agrees within 0.04% at P85
6. **Retrospective Plausibility** - Egnatia Odos motorway (EUR 3.5B budget, EUR 5.93B final) calibration check

---

## Directory Structure

```
quant-risk-copula/
├── data/
│   ├── raw/                         # Raw Excel files from ELSTAT
│   └── processed/
│       └── clean_returns.csv        # Price INDEX levels, 2000-2024, 5 series
├── results/
│   ├── figures/                     # Publication-ready figures (300 DPI PNG)
│   └── tables/                      # CSV + LaTeX output tables
├── src/
│   ├── 01_data_processing.py        # ETL pipeline (OpenPyXL, Pandas)
│   ├── 02_fit_marginals.py          # Jarque-Bera normality tests + kurtosis
│   ├── 03_detailed_simulation.py    # Core engine: Gumbel / Gaussian / Independent (100k sims)
│   ├── 04_generate_figures.py       # Legacy draft figures (not used in paper)
│   ├── 05_master_scenarios.py       # Sensitivity & stress tests (Table 3)
│   ├── 06_copula_gof_table.py       # Goodness-of-fit: PIT + MLE, 4 copulas x 2 regimes
│   ├── 07_enr_validation.py         # External validation vs ENR CCI (annual YoY)
│   ├── 08_volatility_cap.py         # Empirical justification for 15% volatility cap
│   ├── 09_vine_copula.py            # 4D R-vine copula (pyvinecopulib) comparison
│   ├── 10_publication_figures.py    # ASCE publication figures (Fig. 3-7, 300 DPI)
│   ├── 11_bootstrap_ci.py           # Parametric bootstrap CI on hidden risk gap (B=1000)
│   └── 12_egnatia_validation.py     # Retrospective plausibility: Egnatia Odos 2000-2009
├── paper_v10.tex                    # Full LaTeX source (XeLaTeX, 32 references)
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/DimitriosChronis/quant-risk-copula.git
cd quant-risk-copula
pip install -r requirements.txt
```

> **Note:** `pyvinecopulib==0.7.5` is pinned. Newer versions may change the API.

---

## Execution Pipeline

Run scripts in order to fully replicate the paper.

### Step 1 - Data preparation (run once)
```bash
python src/01_data_processing.py   # ETL from ELSTAT Excel -> clean_returns.csv
python src/02_fit_marginals.py     # Normality tests on log-returns
```

### Step 2 - Core simulation
```bash
python src/03_detailed_simulation.py   # 100,000-sim Gumbel / Gaussian / Independent
python src/05_master_scenarios.py      # Sensitivity & stress tests (Table 3)
```

### Step 3 - Goodness-of-fit and validation
```bash
python src/06_copula_gof_table.py      # GoF table (Table 2): Frank/Gumbel/Clayton/Gaussian
python src/07_enr_validation.py        # ENR CCI external alignment
python src/08_volatility_cap.py        # Volatility cap (15%) justification
```

### Step 4 - Robustness and uncertainty
```bash
python src/09_vine_copula.py           # R-vine vs Gumbel comparison (Table 4)
python src/11_bootstrap_ci.py          # Bootstrap CI on hidden risk gap (Table 3 note)
python src/12_egnatia_validation.py    # Egnatia Odos retrospective check (Table 5)
```

### Step 5 - Figures
```bash
python src/10_publication_figures.py   # All publication figures (Fig. 1-7)
```

---

## Key Results

### Main Simulation (100,000 sims, base cost EUR 2,300,000, T=24 months)

| Model | P85 (EUR) | P99 (EUR) | vs. Independent |
|---|---|---|---|
| Independent (industry) | 2,346,xxx | 2,4xx,xxx | - |
| Gaussian | 2,37x,xxx | - | +xx,xxx |
| **Gumbel (full period)** | **2,384,546** | **2,523,257** | **+Hidden Risk** |
| R-Vine (4D) | 2,379,322 | 2,472,711 | agrees within 0.04% |

### Goodness-of-Fit (AIC, lower is better)

| Regime | Best Copula | AIC | Gumbel AIC |
|---|---|---|---|
| Stable (2014-2019, n=72) | Frank | -19.5 | -18.2 |
| Crisis (2021-2024, n=48) | **Gumbel** | **-9.7** | **-9.7** |

### Bootstrap CI on Hidden Risk Gap (B=1,000, T=36 months)

| Regime | Gap Mean (EUR) | 95% CI (EUR) |
|---|---|---|
| Crisis (2021-2024) | 6,438 | [515 ; 17,435] |
| Full Period (2000-2024) | 19,965 | [14,498 ; 25,971] |

### Egnatia Odos Retrospective (calibrated 2000-2009, tau=0.227, theta=1.293)

| Duration | Portfolio | P85 Gap (EUR) | Gap % of base |
|---|---|---|---|
| T=24M | Standard (30/30/20/20) | 15,729 | 0.68% |
| T=36M | Standard (30/30/20/20) | 18,176 | 0.79% |
| T=36M | Motorway (35/30/25/10) | 18,463 | **0.80%** |

The 0.80% per-phase material-cost gap is consistent with the systematic escalation pattern observed in the Egnatia Odos overrun (EUR 3.5B budget vs EUR 5.93B final cost over multiple construction phases).

---

## Data

- **Source:** Greek ELSTAT (Hellenic Statistical Authority) monthly price indices
- **Coverage:** January 2000 – December 2024 (n=299 observations)
- **Series:** `General_Index`, `Concrete`, `Steel`, `Fuel_Energy`, `PVC_Pipes`
- **File:** `data/processed/clean_returns.csv`

> **Important:** The CSV contains price **INDEX LEVELS**, not pre-computed returns.
> All scripts compute log-returns internally: `log(df / df.shift(1)).dropna()`
> Do not apply this transformation manually before running scripts.

---

## Key Technical Details

### Gumbel Copula Sampler
Marshall-Olkin algorithm with Kanter's method for Positive Stable distributions:
```python
alpha = 1.0 / theta                           # Stable index
V = stable_sample(alpha, n)                   # Kanter algorithm
U = np.column_stack([
    np.exp(-E_i / V) for E_i in exponentials  # Gumbel margins
])
```

### Two-Theta Distinction (critical for reproducibility)
| Parameter | Value | Source |
|---|---|---|
| `theta_MLE` (fitted, full period) | 1.503 | MLE on n=299 log-return pseudo-obs |
| `theta_crisis` (fitted, crisis) | 1.141 | MLE on n=48 crisis pseudo-obs |
| `theta_stress` (scenario) | **6.67** | Corresponds to peak rolling tau=0.85 |

The stress scenario (theta=6.67, lambda_U=0.80) is a **forward-looking shock scenario**, not a fitted parameter. This distinction is explicit throughout the paper.

### Volatility Throttling
Monthly return volatility is capped at **15%** - the empirical 100th percentile of observed ELSTAT log-returns (max observed: Fuel_Energy at 1.955% monthly std, annualised ~6.8%). The 15% cap provides a conservative but non-infinite bound for tail simulation.

---

## Requirements

```
numpy>=1.24
pandas>=2.0
scipy>=1.10
matplotlib>=3.7
statsmodels>=0.14
pyvinecopulib==0.7.5
```

Install via: `pip install -r requirements.txt`

---

## Reproducibility

All stochastic components use `numpy.random.default_rng(SEED=42)`.
Full pipeline runtime on a standard laptop: approximately 3-5 minutes.


---

## Author

**Dimitrios Chronis**
School of Civil Engineering, National Technical University of Athens (NTUA)
