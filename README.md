# Construction Cost Risk — Global Risk Suite

> **A copula-based framework for quantifying, transmitting, and hedging systemic tail risk in construction project portfolios.**

Dimitrios Chronis — School of Civil Engineering, National Technical University of Athens (NTUA)
ORCID: [0009-0001-9557-4175](https://orcid.org/0009-0001-9557-4175)

---

## Overview

This monorepo contains the reproducible code for a five-paper research programme on construction cost risk. The framework applies vine copula models, VAR transmission analysis, and Monte Carlo simulation to Greek ELSTAT construction price indices (2000–2024), demonstrating that standard industry risk methods systematically underestimate extreme downside exposure.

**Core thesis:** Independence and Gaussian assumptions used in traditional Monte Carlo ignore tail dependence between construction materials. During systemic crises (2008 GFC, 2021–2022 energy shock), this produces systematic capital shortfalls that propagate across project phases and national borders — and can be quantified, predicted, and ultimately automated.

---

## The Five Papers

| # | Title | Status | Journal | Folder |
|---|-------|--------|---------|--------|
| 1 | From Statistical Error to Profit Erosion: Quantifying Tail Dependence in Construction Cost Overruns | Under review | ASCE JCEM | [`paper1-profit-erosion/`](paper1-profit-erosion/) |
| 2 | Global Commodity Transmission to European Construction Cost Inflation: A Vine Copula Network Topology and VAR-IRF Analysis | Under review | Construction Management and Economics | [`paper2-commodity-transmission/`](paper2-commodity-transmission/) |
| 3 | Tail Risk Quantification and Cross-Border Hedging for Construction Cost Overruns: Expected Shortfall, Lifecycle Phasing, and Basis Risk Analysis Using Vine Copulas | Under review | Automation in Construction | [`paper3-es-hedging/`](paper3-es-hedging/) |
| 4 | A Hybrid LSTM-Copula Agent for Dynamic Regime Detection and Procurement Optimization in Construction Cost Risk | In preparation |  | — |
| 5 | Multi-Agent Ecosystem for Construction Supply Chain Risk: Cross-Currency and Cross-Commodity Coordination | In preparation | TBD | — |

---

## Research Narrative

The five papers form a self-contained research programme:

- **Paper 1 — The Problem:** Gaussian models miss tail dependence. EUR 45,806 hidden risk gap erodes 38–48% of contractor net profit.
- **Paper 2 — The Network:** US commodity shocks transmit to European construction costs with 1–4 month lags. Fuel PPI is the contemporaneous hub; Steel PPI is the primary causal transmitter.
- **Paper 3 — The Quantification:** Basel III ES(99%) reaches EUR 2.94M in crisis (28% overrun). Hedging via US futures is structurally impossible for 4 of 5 materials due to non-cointegration.
- **Paper 4 — The Intelligence:** A hybrid LSTM-Copula agent automates regime detection and generates dynamic procurement recommendations in real time.
- **Paper 5 — The Scale:** Multiple coordinated agents cover the full supply chain across currencies and commodities for global construction portfolios.

---

## Repository Structure

```
construction-risk-suite/
├── README.md                              ← you are here
├── .gitignore
├── requirements.txt                       # Combined dependencies (all papers)
├── shared-data/
│   └── README.md                          # ELSTAT + FRED data source guide
├── paper1-profit-erosion/                 # Paper 1: Gumbel copula tail risk
│   ├── README.md
│   ├── src/                               # 12 analysis scripts
│   ├── data/
│   └── results/
│       ├── tables/
│       └── figures/
├── paper2-commodity-transmission/         # Paper 2: VAR/IRF network topology
│   ├── README.md
│   ├── src/                               # 11 analysis scripts
│   ├── data/
│   └── results/
│       ├── tables/
│       └── figures/
└── paper3-es-hedging/                     # Paper 3: ES + lifecycle + hedging
    ├── README.md
    ├── src/                               # 17 analysis scripts
    ├── data/
    └── results/
        ├── tables/
        └── figures/
```

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/dimitrioschronis/construction-risk-suite
cd construction-risk-suite
pip install -r requirements.txt

# Run Paper 1
cd paper1-profit-erosion
python src/01_data_processing.py
# ... see each paper's README for full pipeline

# Run Paper 2
cd ../paper2-commodity-transmission
python src/01_global_data_download.py
python src/02_align_datasets.py
# ... see paper2 README for full pipeline

# Run Paper 3
cd ../paper3-es-hedging
python src/01_global_data_download.py
# ... see paper3 README for full pipeline
```

---

## Data Sources

All papers use the same underlying datasets:

| Source | Series | Access | Used in |
|--------|--------|--------|---------|
| [ELSTAT SPC23](https://www.statistics.gr/en/statistics/-/publication/SPC23/) | Greek construction cost indices (5 series, monthly 2000–2024) | Manual download | Papers 1, 2, 3 |
| [FRED](https://fred.stlouisfed.org) | US commodity PPIs (Brent, Steel, Cement, PVC, Fuel) | API (auto) | Papers 2, 3 |
| [FRED](https://fred.stlouisfed.org) | EU construction indices (DE, FR, IT, ES) | API (auto) | Paper 2 |

Raw data files are not committed to this repository. See [`shared-data/README.md`](shared-data/README.md) for download instructions and FRED API setup.

---

## Key Contributions

| ID | Contribution | Paper |
|----|-------------|-------|
| C1 | 10-D R-vine copula network topology — US Fuel PPI as dominant hub, post-COVID structural reorganisation | 2 |
| C2 | VAR/IRF/FEVD transmission mechanism — Steel 4M lag, Cement 1M lag, structural breaks | 2 |
| C3 | Basel III ES(99%) via vine copula Monte Carlo — 7.2× Gumbel premium in crisis vs stable | 3 |
| C4 | Lifecycle phase decomposition — superstructure phase +14.2% overrun, bootstrap CIs | 3 |
| C5 | Cross-border hedging — structural basis risk via Engle-Granger, HE < 3% for 4/5 materials | 3 |
| C6 | Cross-EU robustness — 7/8 US→EU Granger pairs significant | 2 |

---

## Global Parameters

All scripts share these global parameters:

```python
SEED           = 42
BASE_COST      = 2_300_000    # EUR
HORIZON        = 24           # months
N_SIMS         = 100_000      # Monte Carlo simulations
CONFIDENCE     = 0.99         # ES level (Paper 3)
ROLLING_WINDOW = 24           # months
BOOTSTRAP_REPS = 500          # Paper 3 lifecycle CIs
MAX_LAG        = 8            # VAR lag selection
```

---

## Dependencies

See [`requirements.txt`](requirements.txt) for pinned versions. Key packages:

```
pyvinecopulib==0.7.5
statsmodels>=0.14
scipy>=1.11
pandas>=2.0
numpy>=1.25
matplotlib>=3.7
fredapi
```

---

## Citation

If you use this code, please cite the relevant paper(s):

```bibtex
@article{chronis2026paper1,
  author  = {Chronis, Dimitrios},
  title   = {From Statistical Error to Profit Erosion:
             Quantifying Tail Dependence in Construction Cost Overruns},
  journal = {Journal of Construction Engineering and Management},
  year    = {2026},
  note    = {Manuscript No. COENG-19311, under review}
}

@article{chronis2026paper2,
  author  = {Chronis, Dimitrios},
  title   = {Global Commodity Transmission to European Construction
             Cost Inflation: A Vine Copula Network Topology and
             VAR-IRF Analysis},
  journal = {Construction Management and Economics},
  year    = {2026},
  note    = {under review}
}

@article{chronis2026paper3,
  author  = {Chronis, Dimitrios},
  title   = {Tail Risk Quantification and Cross-Border Hedging for
             Construction Cost Overruns: Expected Shortfall, Lifecycle
             Phasing, and Basis Risk Analysis Using Vine Copulas},
  journal = {International Journal of Project Management},
  year    = {2026},
  note    = {under review}
}
```

---

## License

MIT License — see `LICENSE` for details.
