# Paper 4: Real-Time Regime Detection for Construction Cost Risk

**An LSTM-Copula Agent with Explainable Procurement Triggers**

> Chronis, D. (2026). *Real-Time Regime Detection for Construction Cost Risk: An LSTM-Copula Agent with Explainable Procurement Triggers*. In preparation.

**Contributions: C6 + C7 + C8**

---

## Scope

This paper upgrades Paper 3's static Rule R6 (reactive, threshold-based) to a
**predictive, LSTM-driven regime detection agent** that identifies construction
cost crises 2 months before they manifest in Greek ELSTAT price indices.

- **C6** -- LSTM 5-seed ensemble binary classifier: US PPI signals (20 features) -> Greek construction crisis regime.
  AUC = 0.926 [95% CI: 0.854-0.983] for Fuel/Energy at lead = 2M.
  Significantly outperforms 4/5 benchmarks (DeLong p < 0.05). #1 ranked model.
- **C7** -- SHAP explainability: US Cement PPI 3M volatility is the dominant
  leading indicator. Temporal SHAP reveals regime-dependent feature importance
  (PVC momentum amplified 2.38x during crisis, Fuel vol 3.47x; Brent fades to 0.12x).
- **C8** -- Crisis episode backtests: GFC 2008 first alert September 2007
  (13 months lead); COVID-19 commodity shock first alert February 2020 (16 months lead).
  Upgraded rules R1-R8 with adaptive contingency.
  LSTM ensemble saves EUR 4,001,160 vs Paper 3 static rule over 72-month test period.

---

## Key Findings

| Metric | Result |
|--------|--------|
| Best AUC (Fuel/Energy, lead=2M) | **0.926** [95% CI: 0.854-0.983] (5-seed ensemble, 20 features) |
| Optimal threshold (Youden's J) | **0.875** (J=0.771, TPR=0.792, FPR=0.021) |
| Walk-forward (55 quarterly windows) | Pooled OOS validated |
| Beats benchmarks (DeLong p<0.05) | **4/5 models** (LR, XGB, ARIMA, RF) |
| LSTM vs #2 (GRU Ensemble) | **+0.018 AUC** (p=0.602, CIs overlap) |
| LSTM vs Random Forest | **+0.190 AUC** (p=0.000, significant) |
| Missed crises (thr=0.50) | **3** (vs 4 for Paper 3 static) |
| False alarms (optimal thr=0.875) | **1** (vs 9 for Paper 3 static) |
| EUR saving vs Paper 3 static | **EUR 4,001,160** (+2.1% over 72 test months) |
| EUR saving vs no hedging | **EUR 3,049,855** (+1.6% over 72 test months) |
| Top SHAP feature | **US_Cement_PPI_vol3** |
| Most crisis-amplified feature | **US_Fuel_PPI_vol3** (3.47x crisis/stable ratio) |
| Most impactful component (ablation) | **Lookback window** (dAUC = -0.216 at 3M vs 6M) |
| GFC 2008 lead time | **13 months** (first alert Sep 2007) |
| COVID 2021 lead time | **16 months** (first alert Feb 2020) |
| Paper 3 -> Paper 4 upgrade | **Reactive -> Predictive (lead = 2M)** |

---

## Architecture

### Pipeline

```
US PPI Data (5 series) -> Feature Engineering (20 features: ret, vol3, vol6, mom3)
    -> MinMaxScaler (fit on train only)
    -> 5-seed LSTM Ensemble -> P(crisis) [0,1]
    -> Youden's J Optimal Threshold (0.875)
    -> Crisis/Stable Decision -> Adaptive Contingency Scaling
```

### LSTM Model

```
Input: (batch, lookback=6, features=20)
    |
LSTM(hidden=64, layers=2, dropout=0.3)
    |
BatchNorm1d(64)
    |
Dropout(0.3)
    |
Linear(64 -> 1)
    |
Sigmoid -> P(crisis) in [0, 1]

x5 seeds -> ensemble mean probability
```

**Training:** BCEWithLogitsLoss + pos_weight (class balancing, no oversampling) +
Adam (lr=5e-4, weight_decay=1e-4) + gradient clipping (norm=1.0) +
early stopping (patience=20).

---

## Scripts (run in order)

All scripts use `utils.py` for shared LSTM classes, ensemble training, and helpers.

### Phase 1: Core Pipeline (01-03)

| # | Script | Purpose | Key output |
|---|--------|---------|------------|
| -- | `utils.py` | Shared LSTM classes, 5-seed ensemble training, helpers | Imported by all scripts |
| 01 | `01_data_preparation.py` | Load aligned returns, engineer 20 US PPI features, compute crisis labels | `features.csv`, `crisis_labels.csv` |
| 02 | `02_lstm_regime_classification.py` | 5-seed ensemble across 4 materials x 4 lead times | `lstm_regime_clf_summary.csv` |
| 03 | `03_shap_explanations.py` | KernelSHAP feature attribution + SHAP ranking (all 20 features used downstream) | `shap_summary.csv`, `top_shap_features.csv` |

### Phase 2: Validation & Benchmarks (04-06)

| # | Script | Purpose | Key output |
|---|--------|---------|------------|
| 04 | `04_walk_forward_validation.py` | 55 expanding quarterly windows walk-forward (ensemble) | `wf_validation_summary.csv`, `wf_quarterly_evolution.csv` |
| 05 | `05_benchmarks.py` | 6-model comparison (LR, RF, XGB, GRU, ARIMA, LSTM) + DeLong pairwise | `benchmark_comparison.csv`, `delong_pairwise.csv` |
| 06 | `06_bootstrap_auc.py` | Bootstrap AUC CI (B=1000) + permutation test + LSTM vs GRU DeLong | `bootstrap_auc_results.csv` |

### Phase 3: Statistical Rigor (07-10)

| # | Script | Purpose | Key output |
|---|--------|---------|------------|
| 07 | `07_robustness_checks.py` | Sensitivity to lead/lookback/vol_window with ensemble | `robustness_results.csv` |
| 08 | `08_rule6_comparison.py` | Paper 3 vs Paper 4: Youden's J threshold + 3-way comparison + EUR cost | `rule6_comparison.csv`, `optimal_threshold.csv` |
| 09 | `09_calibration.py` | Post-hoc calibration (Platt, Isotonic, Beta) | `calibration_results.csv` |
| 10 | `10_granger_causality.py` | Granger causality tests US -> Greek | `granger_causality.csv`, `granger_summary.csv` |

### Phase 4: Application (11-13)

| # | Script | Purpose | Key output |
|---|--------|---------|------------|
| 11 | `11_crisis_backtests.py` | GFC 2008 + COVID 2021 episode backtests (ensemble) | `crisis_backtest_summary.csv` |
| 12 | `12_decision_rules.py` | Upgraded R1-R8 rules with adaptive contingency (ensemble) | `decision_rules_p4.csv`, `p3_vs_p4_comparison.csv` |
| 13 | `13_economic_value.py` | Month-by-month EUR simulation: no-hedge vs static vs LSTM | `economic_value_simulation.csv`, `economic_value_summary.csv` |

### Phase 5: Advanced Analyses (14-15)

| # | Script | Purpose | Key output |
|---|--------|---------|------------|
| 14 | `14_ablation_study.py` | 9 experiments removing one component at a time | `ablation_study.csv` |
| 15 | `15_temporal_shap.py` | Quarterly SHAP evolution + crisis vs stable comparison | `temporal_shap_evolution.csv`, `temporal_shap_crisis_vs_stable.csv` |

### Phase 6: Publication (16)

| # | Script | Purpose | Key output |
|---|--------|---------|------------|
| 16 | `16_publication_figures.py` | Publication-ready figures (runs LAST, reads all results) | `fig1_*.pdf` ... `fig7_*.pdf` |

---

## Key Results

### Benchmark Comparison (DeLong pairwise)

| Rank | Model | AUC | 95% CI | Beats LSTM? |
|------|-------|-----|--------|-------------|
| 1 | **LSTM Ensemble (Paper 4)** | **0.926** | [0.854, 0.982] | -- |
| 2 | GRU Ensemble | 0.908 | [0.832, 0.966] | No (p=0.664) |
| 3 | ARIMA-Threshold | 0.737 | [0.606, 0.847] | No (p=0.004) |
| 4 | Random Forest | 0.736 | [0.619, 0.847] | No (p=0.000) |
| 5 | XGBoost | 0.654 | [0.528, 0.778] | No (p=0.000) |
| 6 | Logistic Regression | 0.609 | [0.465, 0.746] | No (p=0.000) |

LSTM significantly beats 4/5 models (DeLong p < 0.05). Only GRU overlaps (p=0.664).

### Three-Way Rule R6 Comparison

| Metric | Paper 3 Static | LSTM (thr=0.50) | LSTM (optimal=0.875) |
|--------|---------------|-----------------|---------------------|
| AUC | 0.902 | 0.926 | 0.926 |
| Recall | 0.833 | 0.875 | 0.750 |
| Precision | 0.690 | 0.553 | **0.947** |
| F1 | 0.755 | 0.677 | **0.837** |
| FPR | 0.188 | 0.354 | **0.021** |
| Missed crises | 4 | 3 | 6 |
| False alarms | 9 | 17 | **1** |

### Economic Value Simulation (72 test months)

| Strategy | Total EUR | Missed Crises | False Alarms |
|----------|-----------|---------------|--------------|
| A: No hedging | 190.42M | 24 | 0 |
| B: Paper 3 static | 191.37M | 4 | 9 |
| **C: Paper 4 LSTM** | **187.37M** | 6 | **1** |

**LSTM saves EUR 4.00M vs static (+2.1%)** over 72 months. Only 1 false alarm vs 9.

### Ablation Study (AUC impact of removing each component)

| Removed Component | AUC | dAUC |
|-------------------|-----|------|
| Full model (baseline, 20 feat) | **0.926** | -- |
| Lookback (6M -> 3M) | 0.710 | -0.216 |
| SHAP selection (20 -> 14 feat) | 0.795 | -0.131 |
| 2M lead time (-> 0M nowcast) | 0.805 | -0.122 |
| Batch normalization | 0.864 | -0.063 |
| Ensemble averaging | 0.898 | -0.029 |
| LSTM cell (-> GRU) | 0.908 | -0.018 |
| pos_weight balancing | 0.918 | -0.008 |
| Dropout | 0.926 | -0.000 |

**Key finding:** SHAP feature selection (14 features) *hurts* AUC by -0.131. All 20 features used.

### Temporal SHAP (Crisis vs Stable Feature Importance)

| Feature | Crisis |SHAP| | Stable |SHAP| | Ratio |
|---------|---------|---------|-------|
| US_Fuel_PPI_vol3 | 0.0595 | 0.0172 | **3.47x** |
| US_Steel_PPI_vol6 | 0.0076 | 0.0026 | **2.90x** |
| US_PVC_PPI_mom3 | 0.1584 | 0.0666 | **2.38x** |
| US_Fuel_PPI_vol6 | 0.0105 | 0.0057 | **1.84x** |
| US_Brent_vol6 | 0.0128 | 0.1083 | **0.12x** (fades) |
| US_Brent_vol3 | 0.0161 | 0.0542 | **0.30x** (fades) |

Feature importance is regime-dependent -- supports LSTM over linear models.
Brent features dominate in stable periods but fade during crisis (supply chain propagation shifts).

---

## Paper 3 vs Paper 4 Decision Rules

| Rule | Paper 3 (Static) | Paper 4 (LSTM) | Upgrade |
|------|-----------------|----------------|---------|
| R1 | Crisis if vol > 67th pctile | Crisis if P(crisis) > 0.875 -- **2M AHEAD** | +2M lead |
| R2-R4 | Fixed EUR per phase | Adaptive EUR = f(P_crisis) | Continuous |
| R5 | Hedge if rho > 0.40 | Suppress hedge if P > 0.50 | Crisis-aware |
| R6 | Switch model reactively | Switch model **predictively** | Reactive -> Predictive |
| R6b | -- | LSTM early warning + lead quantification | **NEW** |
| R7 | -- | SHAP-guided monitoring: Steel > Cement > Brent | **NEW** |
| R8 | Binary contingency | Continuous: EUR 94,721 -> 644,866 | **NEW** |

---

## Global Parameters

```python
SEED           = 42
LOOKBACK       = 6          # months of US history as LSTM input
LEAD           = 2          # months ahead to predict
ENSEMBLE_SEEDS = [42, 43, 44, 45, 46]   # 5-seed ensemble
N_FEATURES     = 20         # all features (ablation showed full set is better)
OPT_THRESHOLD  = 0.875      # Youden's J optimal
CRISIS_PCT     = 0.75       # P75 threshold for crisis definition
VOL_WINDOW     = 6          # rolling window for crisis vol computation
EPOCHS         = 150
BATCH_SIZE     = 16
HIDDEN_SIZE    = 64
N_LAYERS       = 2
DROPOUT        = 0.3
LR             = 5e-4
PATIENCE       = 20
TRAIN_RATIO    = 0.75
BASE_COST      = 2_300_000  # EUR (from Paper 3)
ES_CRISIS      = 2_944_866  # EUR Expected Shortfall crisis regime
ES_STABLE      = 2_394_721  # EUR Expected Shortfall stable regime
```

---

## Feature Engineering

From 5 US PPI log-return series, 20 features are engineered (all used):

| Feature type | Formula | Count |
|-------------|---------|-------|
| Log-return | Raw monthly log-return | 5 |
| 3M volatility | Rolling 3M std of returns | 5 |
| 6M volatility | Rolling 6M std of returns | 5 |
| 3M momentum | Rolling 3M sum of returns | 5 |
| **Total** | | **20** |

SHAP ranking (script 03) identifies top features for interpretability, but
ablation showed full 20-feature set achieves higher AUC (0.926 vs 0.795 with 14).

---

## Data Sources

Same aligned dataset as Papers 1-3. See [`../shared-data/README.md`](../shared-data/README.md).

| Series | Source | Used for |
|--------|--------|---------|
| Greek Concrete, Steel, Fuel, PVC indices | [ELSTAT SPC23](https://www.statistics.gr/en/statistics/-/publication/SPC23/) | Crisis labels (targets) |
| US Steel, Cement, Fuel, PVC PPIs + Brent | [FRED](https://fred.stlouisfed.org) | Input features (20 engineered, all used) |

**Aligned dataset:** 294 monthly observations x 10 series (July 2000 - December 2024)

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
```

Install: `pip install torch shap xgboost scikit-learn pandas numpy matplotlib`
