# Quantifying Tail Dependence in Construction Cost Overruns

> **A fully reproducible Monte Carlo framework using Gumbel and regular-vine (R-vine) copulas to quantify systemic tail risk—and its monetary cost—in construction project portfolios.**

![Status](https://img.shields.io/badge/Status-Under_Review-yellow)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Scripts](https://img.shields.io/badge/Analysis_scripts-12-orange)

---

## Paper

**"From Statistical Error to Profit Erosion: Quantifying Tail Dependence in Construction Cost Overruns"**
Dimitrios Chronis — School of Civil Engineering, National Technical University of Athens (NTUA)

---

## Executive Summary

Standard contingency tools (PERT, independent or Gaussian-correlated Monte Carlo) underestimate extreme downside risk because the Gaussian dependence structure has **zero upper tail dependence** by construction. During systemic crises (the 2008 Global Financial Crisis and the 2021–2024 energy shock), several construction materials escalate **simultaneously**—exactly the scenario these models cannot represent.

This repository quantifies, in euros, the contingency shortfall that results. It:

1. **Rejects normality** via Jarque–Bera tests on log-returns of five Greek ELSTAT construction price indices (all reject at *p* < 0.001).
2. **Selects the dependence model** by maximum likelihood estimation (MLE) across **five copula families**—Gaussian, Student-t, Gumbel, Clayton, Frank—within stable and crisis regimes.
3. **Samples the Gumbel copula correctly** via the Marshall–Olkin / Kanter positive-stable algorithm, self-validated against the analytic relation *τ* = 1 − 1/*θ*.
4. **Reports a range, not a point**: the "hidden risk gap" is given from the fitted base case to an explicit stress scenario.
5. **Validates externally** against the Engineering News-Record (ENR) Construction Cost Index (4/5 series significant).
6. **Cross-checks via R-vine**: a four-dimensional `pyvinecopulib` vine agrees with the bivariate Gumbel within 0.04 % at the 85th percentile (P85).
7. **Quantifies bootstrap uncertainty**: a 1,000-resample confidence interval on the gap.
8. **Retrospective plausibility**: an Egnatia Odos motorway calibration check.

---

## Headline Result — The Hidden Risk Gap (P85)

The gap is the extra P85 contingency that a Gaussian/independence model omits, reported as a **range** bounded by the fitted crisis dependence and an explicit stress scenario (base cost EUR 2,300,000; crisis-regime volatilities; *N* = 100,000 paths).

| Horizon | Base case (θ = 1.141, λ_U = 0.17) | Stress (θ = 6.67, λ_U = 0.89) |
|---|---|---|
| 12 months | EUR 5,688 (+0.24 %) | EUR 31,023 (+1.31 %) |
| 24 months | **EUR 8,133 (+0.34 %)** | **EUR 44,461 (+1.86 %)** |
| 36 months | EUR 10,046 (+0.42 %) | EUR 55,012 (+2.29 %) |

For a contractor on a 4–5 % net margin, this is **7–11 % of net profit** in the base case and **39–60 %** under stress.

---

## Directory Structure

```
paper1-profit-erosion/
├── data/
│   ├── raw/                         # Raw Excel from ELSTAT (elstat_data.xlsx)
│   └── processed/
│       └── clean_returns.csv        # Price INDEX levels, 2000–2024, 5 series
├── results/
│   ├── figures/                     # Publication figures (300 dpi PNG + vector PDF)
│   └── tables/                      # CSV + LaTeX output tables
├── src/
│   ├── 01_data_processing.py        # ETL pipeline (OpenPyXL, Pandas)
│   ├── 02_fit_marginals.py          # Jarque–Bera normality tests + kurtosis
│   ├── 03_detailed_simulation.py    # Core engine: Gumbel / Gaussian / Independent
│   ├── 04_generate_figures.py       # Legacy draft figures (not used in paper)
│   ├── 05_master_scenarios.py       # Sensitivity & stress tests
│   ├── 06_copula_gof_table.py       # GoF: PIT + MLE, 5 copulas × 2 regimes
│   ├── 07_enr_validation.py         # External validation vs ENR CCI
│   ├── 08_volatility_cap.py         # Empirical justification of 15 % volatility cap
│   ├── 09_vine_copula.py            # 4D R-vine copula (pyvinecopulib) comparison
│   ├── 10_publication_figures.py    # ASCE publication figures (S-curve, rolling τ, density)
│   ├── 11_bootstrap_ci.py           # Parametric bootstrap CI on the gap (B = 1000)
│   ├── 12_egnatia_validation.py     # Retrospective: Egnatia Odos 2000–2009
│   └── 13_gumbel_gap_table.py       # Marshall–Olkin sampler; base-vs-stress gap table
├── requirements.txt
└── README.md
```

> **Script count:** the `src/` folder contains 13 files, of which **12 are the analysis scripts that reproduce the paper** (`01`–`03`, `05`–`13`); `04_generate_figures.py` is a legacy draft retained for reference and is **not** part of the published pipeline.

---

## Installation

```bash
git clone https://github.com/DimitriosChronis/construction-risk-suite.git
cd construction-risk-suite/paper1-profit-erosion
pip install -r requirements.txt
```

> **Note:** `pyvinecopulib==0.7.5` is pinned; newer versions may change the API.

---

## Execution Pipeline

Run scripts in order to replicate the paper.

```bash
# 1. Data preparation (run once)
python src/01_data_processing.py        # ELSTAT Excel -> clean_returns.csv
python src/02_fit_marginals.py          # Normality tests on log-returns

# 2. Core simulation
python src/03_detailed_simulation.py    # Gumbel / Gaussian / Independent
python src/05_master_scenarios.py       # Sensitivity & stress tests

# 3. Selection and validation
python src/06_copula_gof_table.py       # Five-family GoF table (Table 2)
python src/07_enr_validation.py         # ENR CCI external alignment (Table 3)
python src/08_volatility_cap.py         # Volatility-cap justification (Fig. 1)

# 4. Robustness and uncertainty
python src/09_vine_copula.py            # R-vine vs Gumbel (Table 5)
python src/11_bootstrap_ci.py           # Bootstrap CI on the gap
python src/12_egnatia_validation.py     # Egnatia Odos retrospective (Table 6)
python src/13_gumbel_gap_table.py       # Base-vs-stress hidden-risk gap (Table 4)

# 5. Figures
python src/10_publication_figures.py    # S-curve, rolling tau, Gumbel density
```

---

## Key Results

### Five-family goodness of fit (AIC, lower is better)

| Regime | Best copula | AIC | Gumbel AIC | Note |
|---|---|---|---|---|
| Stable (2014–2019, *n* = 72) | Frank | −19.5 | −18.2 | Tail-independent regime |
| **Crisis (2021–2024, *n* = 48)** | **Gumbel** | **−9.7** | **−9.7** | Gumbel wins AIC **and** BIC |

In the crisis regime the full ranking is Gumbel (−9.7) ≻ Frank (−1.1) ≻ Student-t (+1.1) ≻ Gaussian (+1.4) ≻ Clayton (+2.0). The elliptical copulas attain the highest raw log-likelihoods (good central fit) but are penalised for their six and seven parameters. The Student-t degrees of freedom fall from **ν = 35.0** (stable, effectively Gaussian) to **ν = 6.5** (crisis), independently signalling emergent tail dependence.

### Bootstrap CI on the hidden risk gap (*B* = 1,000)

| Sample | Gap mean (EUR) | 95 % CI (EUR) |
|---|---|---|
| Crisis (2021–2024, *n* = 48) | 6,438 | [515 ; 17,435] |
| Full period (2000–2024, *n* = 299) | 19,965 | [14,498 ; 25,971] |

### R-vine robustness (24-month crisis, *N* = 20,000)

| Percentile | R-vine (EUR) | Gumbel (EUR) | Δ |
|---|---|---|---|
| P85 | 2,379,322 | 2,378,314 | **+0.04 %** |
| P99 | 2,472,711 | 2,511,476 | −1.54 % (conservative) |

### Egnatia Odos retrospective (calibrated 2000–2009, τ = 0.227, θ = 1.293)

| Duration | Portfolio | P85 gap (EUR) | % of base |
|---|---|---|---|
| 24 M | Standard (30/30/20/20) | 15,729 | 0.68 % |
| 36 M | Standard (30/30/20/20) | 18,176 | 0.79 % |
| 36 M | Motorway (35/30/25/10) | 18,463 | **0.80 %** |

The 0.80 % per-phase material-cost gap is consistent with the systematic escalation in the Egnatia Odos overrun (EUR 3.5 B budget vs EUR 5.93 B final cost over multiple construction phases).

---

## Data

- **Source:** Greek ELSTAT (Hellenic Statistical Authority) monthly construction price indices.
- **Coverage:** January 2000 – December 2024 (*n* = 299 log-return observations).
- **Series:** `General_Index`, `Concrete`, `Steel`, `Fuel_Energy`, `PVC_Pipes`.
- **File:** `data/processed/clean_returns.csv`.

> **Important:** the CSV holds price **INDEX LEVELS**, not returns. All scripts compute log-returns internally as `log(df / df.shift(1)).dropna()`. Do not pre-transform.

---

## Key Technical Details

### Gumbel copula sampler (`13_gumbel_gap_table.py`)
Exchangeable Gumbel copula via the Marshall–Olkin algorithm with Kanter's method for the positive-stable mixing variable:

```python
alpha = 1.0 / theta                                   # stable index in (0,1)
S = sample_positive_stable(alpha, n, rng)             # Kanter (1975)
E = rng.exponential(1.0, size=(n, d))                 # iid unit exponentials
U = np.exp(-np.power(E / S[:, None], alpha))          # Gumbel copula sample
```

The sampler is **self-validated**: empirical Kendall's τ matches the analytic 1 − 1/θ to within 0.002 at θ ∈ {1.141, 1.503, 6.67}.

### Base case versus stress scenario (critical for reproducibility)

| Parameter | Value | Source | λ_U |
|---|---|---|---|
| θ (MLE, full period) | 1.503 | MLE on *n* = 299 pseudo-obs | 0.41 |
| θ (MLE, crisis) — **base case** | 1.141 | MLE on *n* = 48 crisis pseudo-obs | 0.17 |
| θ (stress scenario) | **6.67** | Peak rolling τ = 0.85 | **0.89** |

The stress value (θ = 6.67) is a **transparent upper-bound scenario** anchored at the most extreme observed dependence—not a fitted parameter that persists for the horizon. The base case and stress case are reported **jointly** so the realised contingency requirement is understood to lie between them.

### Volatility throttling
Monthly return volatility is capped at **15 %**—above the 100th percentile of observed ELSTAT log-returns (maximum 10.31 %, Fuel/Energy). The cap is never binding under historical conditions; it is a numerical safeguard only.

---

## Requirements

```
numpy>=1.24
pandas>=2.0
scipy>=1.10
matplotlib>=3.7
statsmodels>=0.14
pyvinecopulib==0.7.5
openpyxl>=3.1
```

Install via `pip install -r requirements.txt`.

---

## Reproducibility

All stochastic components use `numpy.random.default_rng(42)`. Full-pipeline runtime on a standard laptop is approximately 3–5 minutes. Figures are emitted as both 300 dpi PNG and vector PDF.

---

## Author

**Dimitrios Chronis**
School of Civil Engineering, National Technical University of Athens (NTUA)
ORCID: [0009-0001-9557-4175](https://orcid.org/0009-0001-9557-4175)

## License

MIT License — see `LICENSE`.
