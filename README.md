# Global Construction Risk Intelligence Suite

> **An 8-paper research programme on automated construction cost risk: from tail dependence quantification to global megaproject intelligence.**

Dimitrios Chronis — School of Civil Engineering, National Technical University of Athens (NTUA)
ORCID: [0009-0001-9557-4175](https://orcid.org/0009-0001-9557-4175)
Email: dimitrischronis7@gmail.com

---

## Overview

This monorepo contains the reproducible code for an 8-paper research series on construction cost risk intelligence. The framework applies vine copula models, LSTM ensemble agents, VAR transmission analysis, and Monte Carlo simulation to construction price indices (2000–2024), demonstrating that standard industry risk methods systematically underestimate extreme downside exposure.

**Core thesis:** Independence and Gaussian assumptions used in traditional Monte Carlo ignore tail dependence between construction materials. During systemic crises (2008 GFC, 2021–2022 energy shock), this produces systematic capital shortfalls that propagate across project phases and national borders — and can be quantified, predicted, and ultimately automated with 13–16 months of early warning.

---

## The Research Series

| # | Title | Status | Journal | Folder |
|---|-------|--------|---------|--------|
| 1 | From Statistical Error to Profit Erosion: Quantifying Tail Dependence in Construction Cost Overruns Using Gumbel Copulas | Under Review | JCEM (ASCE) | [`paper1-profit-erosion/`](paper1-profit-erosion/) |
| 2 | Global Commodity Transmission to European Construction Cost Inflation: A Vine Copula Network Topology and VAR-IRF Analysis | Under Review | Construction Management and Economics | [`paper2-commodity-transmission/`](paper2-commodity-transmission/) |
| 3 | An Automated Tail Risk Quantification and Procurement Decision Framework for Construction Cost Overruns: Expected Shortfall, Lifecycle Phasing, and Basis Risk Analysis Using Vine Copulas | Under Review | Journal of Building Engineering | [`paper3-es-hedging/`](paper3-es-hedging/) |
| 4 | Real-Time Regime Detection for Construction Cost Risk: An LSTM-Copula Agent with Explainable Procurement Triggers | Under Review | Automation in Construction (IF 14.4) | [`paper4-lstm-agent/`](paper4-lstm-agent/) |
| 5 | Systemic Risk Contagion in Construction Cost Portfolios: A Dynamic Vine Copula Network Approach to Cross-Project Tail Dependence | In preparation | — | — |
| 6 | Pan-European Construction Cost Risk Intelligence: Vine Copula-LSTM Validation Across Southern European Markets | In preparation | — | — |
| 7 | Multi-Currency Construction Cost Intelligence for Global Megaproject Portfolios: Shanghai Steel, Gulf Construction, and FX Volatility | — | — | — |
| 8 | A Three-Layer Cascade Theory of Global Construction Cost Crises: Financial Markets, Commodity Networks, and Domestic Prices | — | — | — |

---

## Research Narrative

Each paper answers one question and opens the next:

```
P1: "The problem is bigger than you think"     → Tail dependence exists
P2: "It comes from there"                      → US→Greek transmission mapped
P3: "Quantify it and automate the response"    → ES(99%) + 6 procurement rules
P4: "See it 16 months before it arrives"       → LSTM agent AUC = 0.926
P5: "Scale to the full portfolio"              → Cross-project contagion
P6: "Valid across all of Europe"               → Pan-EU risk index
P7: "Valid anywhere in the world"              → Dubai, London, Singapore
P8: "It is a universal law"                    → Financial → Commodity → Construction
```

---

## Key Results (Papers 1–4)

| Paper | Key Finding | Value |
|-------|------------|-------|
| P1 | Hidden tail risk gap vs Gaussian | EUR 45,806 per project |
| P2 | US→Greek transmission lag | Steel: 4M, Fuel: 1M |
| P3 | Crisis ES(99%) overrun | EUR 2.94M (+28%) |
| P3 | Superstructure phase overrun risk | +14.2% (bootstrap CI confirmed) |
| P4 | **LSTM ensemble AUC** | **0.926 [95% CI: 0.854–0.983]** |
| P4 | **GFC 2008 early warning** | **13 months before peak** |
| P4 | **COVID 2021 early warning** | **16 months before peak** |
| P4 | Economic saving vs static rules | EUR 4,001,160 over 72 months |
| P4 | False alarm reduction | 89% (1 vs 9 false alarms) |

---

## Repository Structure

```
construction-risk-suite/
├── README.md
├── .gitignore
├── requirements.txt
├── shared-data/
│   └── README.md                    # ELSTAT + FRED data source guide
├── paper1-profit-erosion/           # Gumbel copula tail risk
│   ├── README.md
│   ├── src/                         # 12 analysis scripts
│   ├── data/raw/.gitkeep
│   ├── data/processed/.gitkeep
│   └── results/.gitkeep
├── paper2-commodity-transmission/   # VAR/IRF network topology
│   ├── README.md
│   ├── src/                         # 11 analysis scripts
│   ├── data/raw/.gitkeep
│   ├── data/processed/.gitkeep
│   └── results/.gitkeep
├── paper3-es-hedging/               # ES + lifecycle + hedging
│   ├── README.md
│   ├── src/                         # 17 analysis scripts
│   ├── data/raw/.gitkeep
│   ├── data/processed/.gitkeep
│   └── results/.gitkeep
└── paper4-lstm-agent/               # LSTM-Copula agent
    ├── README.md
    ├── src/                         # 16 scripts + utils.py
    │   ├── utils.py
    │   ├── 01_data_preparation.py
    │   ├── 02_lstm_regime_classification.py
    │   ├── 03_shap_explanations.py
    │   ├── 04_walk_forward_validation.py
    │   ├── 05_benchmarks.py
    │   ├── 06_bootstrap_auc.py
    │   ├── 07_robustness_checks.py
    │   ├── 08_rule6_comparison.py
    │   ├── 09_calibration.py
    │   ├── 10_granger_causality.py
    │   ├── 11_crisis_backtests.py
    │   ├── 12_decision_rules.py
    │   ├── 13_economic_value.py
    │   ├── 14_ablation_study.py
    │   ├── 15_temporal_shap.py
    │   └── 16_publication_figures.py
    ├── data/raw/.gitkeep
    ├── data/processed/.gitkeep
    └── results/.gitkeep
```

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/dimitrioschronis/construction-risk-suite
cd construction-risk-suite
pip install -r requirements.txt

# Run Paper 4 (LSTM Agent) — recommended entry point
cd paper4-lstm-agent/src
python run_all.py        # Runs all 16 scripts sequentially (~67 min)

# Or run individual scripts:
python 01_data_preparation.py
python 02_lstm_regime_classification.py
# ... see paper4 README for full pipeline
```

---

## Paper 4 — LSTM Agent Pipeline

```
01_data_preparation.py          → Feature engineering (20 features, 294 obs)
02_lstm_regime_classification.py → 4×4 AUC matrix (materials × lead times)
03_shap_explanations.py         → KernelSHAP feature attribution
04_walk_forward_validation.py   → Expanding window OOS validation
05_benchmarks.py                → 6-model DeLong comparison
06_bootstrap_auc.py             → Bootstrap CI (B=1,000) + permutation test
07_robustness_checks.py         → Sensitivity: lead/lookback/threshold
08_rule6_comparison.py          → Paper 3 vs Paper 4 + Youden's J
09_calibration.py               → Isotonic regression (ECE: 0.199→0.123)
10_granger_causality.py         → Bivariate Granger: US PPI → Greek vol
11_crisis_backtests.py          → GFC 2008 (13M lead) + COVID 2021 (16M lead)
12_decision_rules.py            → Adaptive rules R1–R8
13_economic_value.py            → EUR simulation: saving EUR 4,001,160
14_ablation_study.py            → Component contribution analysis
15_temporal_shap.py             → Quarterly SHAP evolution
16_publication_figures.py       → All publication figures
```

---

## Global Parameters

```python
# Paper 4 (LSTM Agent)
SEED             = 42
ENSEMBLE_SEEDS   = [42, 43, 44, 45, 46]
LOOKBACK         = 6       # months of US history as LSTM input
LEAD             = 2       # months ahead to predict
N_FEATURES       = 20      # all features (ablation confirmed)
OPT_THRESHOLD    = 0.875   # Youden's J optimal
CRISIS_PCT       = 0.75    # P75 for crisis definition
HIDDEN_SIZE      = 64
N_LAYERS         = 2
DROPOUT          = 0.3
LR               = 5e-4
PATIENCE         = 20

# Papers 1–3 (Copula)
BASE_COST        = 2_300_000   # EUR
HORIZON          = 24          # months
N_SIMS           = 100_000     # Monte Carlo paths
CONFIDENCE       = 0.99        # ES level
BOOTSTRAP_REPS   = 500
```

---

## Data Sources

| Source | Series | Access | Papers |
|--------|--------|--------|--------|
| [ELSTAT SPC23](https://www.statistics.gr/en/statistics/-/publication/SPC23/) | Greek construction cost indices (monthly 2000–2024) | Manual download | 1, 2, 3, 4 |
| [FRED](https://fred.stlouisfed.org) | US PPIs: Steel (WPU1017), Cement (WPU1321), Fuel (WPU0553), PVC (WPU0911), Brent (DCOILBRENTEU) | API (automated) | 2, 3, 4 |

Raw data files are not committed. See [`shared-data/README.md`](shared-data/README.md) for download instructions.

---

## Dependencies

```bash
pip install -r requirements.txt
```

Key packages:
```
torch>=2.0
shap>=0.44
xgboost>=1.7
scikit-learn>=1.3
pyvinecopulib==0.7.5
statsmodels>=0.14
pandas>=2.0
numpy>=1.25
matplotlib>=3.7
fredapi
```

---

## Citation

If you use this code, please cite the relevant paper

---

## License

MIT License — see `LICENSE` for details.
