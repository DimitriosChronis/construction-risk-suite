# Paper 4: Real-Time Crisis-Regime Detection for Construction Fuel and Energy Costs

**An Explainable LSTM Early-Warning Agent**

> Chronis, D. (2026). *Real-Time Crisis-Regime Detection for Construction Fuel
> and Energy Costs: An Explainable LSTM Early-Warning Agent.* Manuscript under
> review (ASCE *Journal of Computing in Civil Engineering*).

**Contributions: C6 + C7 + C8**

> **Note on this revision.** All results below come from a strictly **leak-free
> (point-in-time / causal) re-analysis** of the entire pipeline: the crisis
> percentile threshold at each month uses only data available up to that month,
> so no future information enters label construction. The leak-free labels agree
> with the original global-threshold labels on **97.7%** of months for the
> headline material, so the framework is validated rather than overturned; the
> corrected numbers are slightly more conservative and are reported honestly
> throughout.

---

## Scope

This paper upgrades Paper 3's static Rule R6 (reactive, threshold-based) to a
**predictive, LSTM-driven regime-detection agent** that anticipates
construction **fuel/energy** cost crises before they manifest in Greek ELSTAT
price indices. The skill is **material-specific** and we state this plainly: the
agent is effective for Fuel/Energy and moderately effective for PVC, but **not**
predictive for Steel or Concrete (no usable upstream US signal — confirmed by
Granger tests).

- **C6** — LSTM 5-seed ensemble binary classifier: US PPI signals (20 features)
  → Greek construction crisis regime. **AUC = 0.908 [95% CI: 0.824–0.980]** for
  Fuel/Energy at lead = 2M (permutation *p* < 0.001). Significantly outperforms
  **3/5** benchmarks (DeLong *p* < 0.05); statistically tied with the GRU
  ensemble.
- **C7** — SHAP explainability: US Cement PPI 3M volatility is the dominant
  leading indicator. Temporal SHAP reveals regime-dependent importance (US Fuel
  PPI 6M volatility amplified **3.26×** during crisis; Brent crude fades to
  **0.31×** — supply-chain decoupling).
- **C8** — Crisis backtests (strict out-of-sample): the COVID-19 commodity shock
  is detected strongly (**AUC 0.95, recall 0.95**) but with a **short lead** to
  the price peak; the 2008 episode is detected weakly (AUC 0.26) given the tiny
  pre-2007 training set. Upgraded rules R1–R8 with adaptive contingency save
  **EUR 2.35M** vs the Paper 3 static rule over 72 months, cutting false alarms
  from 9 to 3.

---

## Key Findings (leak-free)

| Metric | Result |
|--------|--------|
| Best AUC (Fuel/Energy, lead = 2M) | **0.908** [95% CI: 0.824–0.980] (5-seed ensemble, 20 features) |
| Optimal threshold (Youden's J) | **0.953** (J = 0.681, TPR = 0.76, FPR = 0.079) |
| Walk-forward (44 expanding windows) | **Pooled OOS AUC = 0.780** (0.880 in the 2020–2024 crisis era) |
| Beats benchmarks (DeLong *p* < 0.05) | **3/5** models (LogReg, XGBoost, ARIMA) |
| LSTM vs GRU Ensemble | −0.006 AUC (*p* = 0.84, statistically tied) |
| LSTM vs Random Forest | +0.068 AUC (*p* = 0.14, n.s.; but RF crisis recall = 0.04, inert) |
| Material scope | **Fuel/Energy (0.91), PVC (0.83)**; Steel/Concrete not predictable |
| False alarms (optimal thr) | **3** (vs 9 for Paper 3 static) |
| EUR saving vs Paper 3 static | **EUR 2.35M** (over 72 test months) |
| Top SHAP feature | **US_Cement_PPI_vol3** |
| Most crisis-amplified feature | **US_Fuel_PPI_vol6** (3.26× crisis/stable ratio) |
| Most impactful component (ablation) | **2-month lead** (ΔAUC = −0.093 when removed) |
| Calibration | ECE **0.155 → 0.093** (isotonic) |
| Label leakage diagnostic | **97.7%** causal-vs-global agreement (Fuel/Energy) |
| Paper 3 → Paper 4 upgrade | **Reactive → Predictive (lead = 2M)** |

---

## Architecture

### LSTM Model

```
Input: (batch, lookback=6, features=20)
    |
LSTM(hidden=64, layers=2, dropout=0.3)  ->  BatchNorm1d(64)  ->  Dropout(0.3)
    |
Linear(64 -> 1)  ->  Sigmoid  ->  P(crisis) in [0, 1]
    |
x5 seeds {42,43,44,45,46} -> ensemble mean probability
```

**Training:** BCEWithLogitsLoss + pos_weight (class balancing, no oversampling) +
Adam (lr = 5e-4, weight_decay = 1e-4) + gradient clipping (norm = 1.0) +
early stopping (patience = 20). Input scaler fit on training data only.

---

## Key Results

### Benchmark Comparison (Fuel/Energy, lead = 2M; DeLong pairwise)

| Rank | Model | AUC | 95% CI | Crisis recall | Beats LSTM? |
|------|-------|-----|--------|---------------|-------------|
| 1 | GRU Ensemble | 0.914 | [0.804, 0.981] | 0.96 | tied (*p* = 0.86) |
| 2 | **LSTM Ensemble (Paper 4)** | **0.908** | [0.817, 0.975] | 0.80 | — |
| 3 | Random Forest | 0.840 | [0.740, 0.929] | **0.04** | n.s. (*p* = 0.14)* |
| 4 | XGBoost | 0.786 | [0.669, 0.884] | 0.40 | LSTM wins (*p* = 0.008) |
| 5 | Logistic Regression | 0.737 | [0.600, 0.870] | 0.44 | LSTM wins (*p* < 0.001) |
| 6 | ARIMA-Threshold | 0.708 | [0.581, 0.826] | 0.48 | LSTM wins (*p* = 0.002) |

LSTM significantly beats **3/5** models. It ties the GRU ensemble (expected at
this sample size). *Random forest ranks third on AUC but issues almost no crisis
alerts (recall 0.04), making it operationally inert despite a competitive
ranking statistic.

### Three-Way Rule R6 Comparison

| Metric | Paper 3 Static | LSTM (thr = 0.50) | LSTM (optimal = 0.953) |
|--------|---------------|-------------------|------------------------|
| AUC | 0.899 | 0.908 | 0.908 |
| Recall | 0.80 | 0.80 | 0.72 |
| Precision | 0.714 | 0.690 | **0.857** |
| F1 | 0.755 | 0.741 | **0.783** |
| FPR | 0.211 | 0.237 | **0.079** |
| Missed crises | 5 | 5 | 7 |
| False alarms | 8 | 9 | **3** |

### Economic Value Simulation (72 test months)

| Strategy | Total EUR | Missed Crises | False Alarms |
|----------|-----------|---------------|--------------|
| A: No hedging | 169.62M | 25 | 0 |
| B: Paper 3 static | 170.02M | 5 | 8 |
| **C: Paper 4 LSTM** | **167.67M** | 7 | **3** |

**LSTM saves EUR 2.35M vs static** over 72 months, with 3 false alarms vs 9. The
saving is path-dependent and sensitive to the decision threshold.

### Ablation Study (AUC impact of removing each component; baseline 0.908)

| Removed Component | AUC | ΔAUC |
|-------------------|-----|------|
| Full model (baseline, 20 feat) | **0.908** | — |
| 2M lead time (→ 0M nowcast) | 0.815 | **−0.093** |
| Full lookback (6M → 3M) | 0.843 | −0.065 |
| Batch normalization | 0.848 | −0.060 |
| Ensemble averaging | 0.882 | −0.026 |
| Dropout | 0.897 | −0.012 |
| LSTM cell (→ GRU) | 0.914 | +0.005 |
| SHAP selection (20 → 14 feat) | 0.909 | +0.001 |
| pos_weight balancing | 0.925 | +0.017 |

**Key finding:** the temporal components (lead, lookback) carry the performance.
A **14-feature SHAP-selected subset matches the full 20-feature model**, so the
model can be simplified without loss; dropout, the LSTM/GRU choice, and class
balancing are near-neutral.

### Temporal SHAP (Crisis vs Stable feature importance)

| Feature | Crisis \|SHAP\| | Stable \|SHAP\| | Ratio |
|---------|---------|---------|-------|
| US_Fuel_PPI_vol6 | 0.0255 | 0.0078 | **3.26×** |
| US_PVC_PPI_ret | 0.0234 | 0.0122 | **1.91×** |
| US_Fuel_PPI_vol3 | 0.0160 | 0.0091 | **1.76×** |
| US_PVC_PPI_mom3 | 0.2509 | 0.1439 | **1.74×** |
| US_Steel_PPI_mom3 | 0.1486 | 0.0969 | **1.53×** |
| US_Brent_mom3 | 0.0234 | 0.0443 | **0.53×** (fades) |
| US_Brent_vol6 | 0.0229 | 0.0746 | **0.31×** (fades) |

Feature importance is regime-dependent: Brent crude signals dominate in stable
periods but fade during crises, while fuel/PVC supply-chain signals are amplified
— this state-switching justifies a nonlinear LSTM over linear models.

### Crisis Backtests (strict out-of-sample)

| Episode | Train end | First alert | Peak | Lead (to peak) | AUC |
|---------|-----------|-------------|------|----------------|-----|
| GFC 2008 | 2006-12 | 2007-09 | 2008-10 | 13 M | 0.26 |
| COVID 2021 | 2018-12 | 2021-05 | 2021-06 | 1 M | **0.95** |

The COVID episode is detected reliably (AUC 0.95) but with a **short lead** to the
price peak. The 2008 episode shows weak discrimination given only ~6 years of
mostly pre-crisis training data. We therefore claim **reliable detection** of
fuel/energy crises, **not** a guaranteed multi-month early warning.

---

## Scripts (run in order)

All scripts use `utils.py` for shared LSTM classes, leak-free causal labels,
5-seed ensemble training, and helpers. Run `python run_all.py` to execute 01–16.

| # | Script | Purpose |
|---|--------|---------|
| 01 | `01_data_preparation.py` | Engineer 20 US PPI features; **leak-free causal crisis labels** + leakage diagnostic |
| 02 | `02_lstm_regime_classification.py` | 5-seed ensemble across 4 materials × 4 lead times |
| 03 | `03_shap_explanations.py` | KernelSHAP feature attribution + ranking |
| 04 | `04_walk_forward_validation.py` | 44 expanding quarterly windows; **pooled OOS AUC** |
| 05 | `05_benchmarks.py` | 6-model comparison (LR, RF, XGB, GRU, ARIMA, LSTM) + DeLong |
| 06 | `06_bootstrap_auc.py` | Bootstrap AUC CI (B = 1000) + permutation + DeLong |
| 07 | `07_robustness_checks.py` | Sensitivity to lead/lookback/vol_window and **crisis percentile P67–P90** |
| 08 | `08_rule6_comparison.py` | Paper 3 vs Paper 4; Youden's J threshold; 3-way comparison |
| 09 | `09_calibration.py` | Post-hoc calibration (Platt, isotonic) |
| 10 | `10_granger_causality.py` | Granger causality US → Greek |
| 11 | `11_crisis_backtests.py` | GFC 2008 + COVID 2021 episode backtests |
| 12 | `12_decision_rules.py` | Upgraded R1–R8 rules with adaptive contingency |
| 13 | `13_economic_value.py` | Month-by-month EUR simulation: no-hedge vs static vs LSTM |
| 14 | `14_ablation_study.py` | 9 experiments removing one component at a time |
| 15 | `15_temporal_shap.py` | Quarterly SHAP evolution + crisis vs stable |
| 16 | `16_publication_figures.py` | Publication figures (runs last; reads all results) |

---

## Global Parameters

```python
SEED           = 42
LOOKBACK       = 6          # months of US history as LSTM input
LEAD           = 2          # months ahead to predict
ENSEMBLE_SEEDS = [42, 43, 44, 45, 46]   # 5-seed ensemble
N_FEATURES     = 20         # 14 SHAP-selected features perform equivalently
OPT_THRESHOLD  = 0.953      # Youden's J optimal
CRISIS_PCT     = 0.75       # P75 threshold for crisis definition (point-in-time)
VOL_WINDOW     = 6          # rolling window for crisis vol computation
MIN_HIST       = 36         # causal-label burn-in (months)
EPOCHS         = 150
BATCH_SIZE     = 16
HIDDEN_SIZE    = 64
N_LAYERS       = 2
DROPOUT        = 0.3
LR             = 5e-4
PATIENCE       = 20
TRAIN_RATIO    = 0.75
BASE_COST      = 2_300_000  # EUR (from Paper 3)
C_CRISIS       = 644_866    # EUR crisis contingency (Paper 3 ES, Rule R8 max)
C_STABLE       = 94_721     # EUR stable contingency (Paper 3 ES, Rule R8 min)
```

---

## Data Sources

Same aligned dataset as Papers 1–3.

| Series | Source | Used for |
|--------|--------|---------|
| Greek Concrete, Steel, Fuel, PVC indices | [ELSTAT SPC23](https://www.statistics.gr/en/statistics/-/publication/SPC23/) | Crisis labels (targets) |
| US Steel, Cement, Fuel, PVC PPIs + Brent | [FRED](https://fred.stlouisfed.org) | Input features (20 engineered) |

**Aligned dataset:** after the 36-month causal-label burn-in, the usable sample
is **259 monthly observations** (June 2003 – December 2024); Fuel/Energy has 65
crisis months (25.1%).

---

## Dependencies

```
torch>=2.0
shap>=0.44
xgboost>=1.7
scikit-learn>=1.3
pandas>=2.0
numpy>=1.25
matplotlib>=3.7
statsmodels>=0.14
```

Install: `pip install torch shap xgboost scikit-learn pandas numpy matplotlib statsmodels`

---

## License

MIT License.
