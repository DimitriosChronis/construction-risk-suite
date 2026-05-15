# Paper 5 — Data Dictionary

One table per CSV in `data/processed/`. Files are grouped by the pipeline
step that produces them. Units: monetary values in **EUR**; returns are
**log-returns** on a monthly index; Kendall's τ and λ_U are unitless and
lie in [−1, 1] and [0, 1] respectively.

All date columns are **month-start** ISO-8601 strings (`YYYY-MM-01`).

---

## 02 — Portfolio construction

### `portfolio_metadata.csv`
One row per Διαύγεια contract.

| column | type | description |
|---|---|---|
| project_id | str | Internal id `P#####` |
| ada | str | Διαύγεια "ΑΔΑ" decision code |
| type | str | Asset class ∈ {road, bridge, pipeline, building, other} |
| subject | str | Contract title (Greek) |
| organization | str | Awarding authority |
| budget_eur | float | Contract value (EUR) |
| budget_category | str | Bucket label (Small / Mid / Large / Mega) |
| duration_m | int | Project duration in months |
| start_month | date | First active month |
| end_month | date | Last active month |
| w_concrete | float | Weight of concrete in theoretical cost mix |
| w_steel | float | Weight of steel |
| w_fuel | float | Weight of fuel/energy |
| w_pvc | float | Weight of PVC/plastics |

### `portfolio_summary.csv`
Single-row summary of the portfolio.

| column | type | description |
|---|---|---|
| n_projects | int | Total contracts retained |
| period_start / period_end | date | Portfolio time-span |
| total_months | int | Distinct active months |
| total_budget_eur | float | Sum of contract values |
| median_budget_eur | float | Median contract value |
| mean_duration_months | float | Mean project duration |
| mean_active_per_month | float | Average open contracts per month |
| peak_active_per_month | int | Max concurrent contracts |
| n_road, n_bridge, n_pipeline, n_building, n_other | int | Count per asset class |
| mean_w_concrete / _steel / _fuel / _pvc | float | Portfolio-average cost mix |

### `portfolio_project_returns.csv`
Long-form panel of per-project monthly log-returns.

| column | type | description |
|---|---|---|
| date | date | Month-start |
| project_id | str | Project id |
| ret | float | Log-return of theoretical project cost index |

### `portfolio_project_returns_wide.csv`
Same data in wide format: `date` + one column per `project_id`.

### `portfolio_type_returns.csv`
Empirical value-weighted log-returns by asset class. Columns: `date`,
`road`, `bridge`, `pipeline`, `building`, `other`. NaN where no active
projects of that class.

### `portfolio_type_returns_theoretical.csv`
Theoretical counterpart built from ELSTAT / FRED material PPIs so that
the series exists for the full 2000–2025 period.

| column | type | description |
|---|---|---|
| Date | date | Month-start |
| road / bridge / pipeline / building / other | float | Log-return of synthetic cost index per class |

---

## 03 — Dynamic copula

All files are one row per month, 10 pair columns of the form
`<typeA>__<typeB>` (five asset classes → C(5,2)=10 pairs).

### `dynamic_copula_tau.csv`
Rolling Kendall's τ per pair (window = 36 months).

### `dynamic_copula_lambdaU.csv`
Rolling Gumbel upper-tail dependence `λ_U = 2 − 2^(1−τ)` per pair.

### `dynamic_copula_empirical_lambdaU.csv`
Non-parametric empirical upper-tail dependence (top-quantile hit rate).

### `dynamic_copula_network_avg.csv`
Monthly network aggregates.

| column | type | description |
|---|---|---|
| date | date | Month-start |
| avg_tau | float | Mean of 10 pairwise τ |
| avg_lambdaU | float | Mean of 10 Gumbel λ_U |
| avg_lambdaU_emp | float | Mean empirical λ_U |
| regime | str | Crisis / stable label from CSRI quantile rule |
| regime_exo | str | Exogenous regime label (pre_2012, 2012, 2015, 2022, post_2022) |

### `dynamic_copula_regimes.csv`
`date`, `regime`, `regime_exo` only — stand-alone regime calendar.

---

## 04 — Contagion index

### `contagion_index.csv`
Per-type contagion index CI_t. Columns: `date`, `road`, `bridge`,
`pipeline`, `building`, `other`. CI ∈ [0, 1].

### `contagion_flows.csv`
Directed source/receiver flows. For each type T: `OUT_T`, `IN_T`, `NET_T`
(= OUT − IN).

### `contagion_classification.csv`
Static classification of each type over stable vs crisis regimes.

| column | type | description |
|---|---|---|
| type | str | Asset class |
| CI_full / CI_stable / CI_crisis | float | Mean CI in each regime |
| amplification | float | CI_crisis / CI_stable |
| OUT, IN, NET | float | Mean directed flows |
| role | str | source / receiver / neutral |

---

## 05 — Portfolio ES + contingency

### `portfolio_es_by_regime.csv`
ES₉₅ of the 5-type portfolio under different weightings × regimes.

| column | type | description |
|---|---|---|
| weights | str | `equal` / `budget` / `active` / `inverse_vol` |
| regime | str | `full` / `stable` / `crisis` |
| n_months | int | Observations in sub-sample |
| ES_sum | float | Σ ES of each type (comonotone upper bound) |
| ES_portfolio | float | ES of the portfolio log-return |
| ratio | float | ES_portfolio / ES_sum |
| div_benefit | float | 1 − ratio (diversification benefit) |
| ES_road / ES_bridge / ES_pipeline / ES_building / ES_other | float | Marginal ES per type |

### `portfolio_es_rolling.csv`
Rolling ES (24M window).

| column | type | description |
|---|---|---|
| date | date | Month-start |
| ES_sum / ES_portfolio / ratio / div_benefit | float | See above |
| regime | str | Regime label at `date` |

### `project_contingency.csv`
Per-project €-contingency recommendation.

| column | type | description |
|---|---|---|
| project_id, ada, type, budget_eur, duration_m | — | From metadata |
| es_crisis_type | float | ES₉₅ of the project's type under crisis |
| contingency_pct | float | Recommended contingency (fraction of budget) |
| contingency_eur | float | `contingency_pct × budget_eur` |

---

## 06 — LSTM portfolio agent

### `agent_features.csv`
Feature matrix fed to the walk-forward LSTM.

| column | type | description |
|---|---|---|
| (index) | date | Month-start |
| US_Brent_ret / US_Steel_PPI_ret / ... _ret | float | 1M log-return of each FRED driver |
| ..._vol3, ..._vol6 | float | 3M / 6M rolling standard deviation |
| ..._mom3 | float | 3M momentum |
| network_avg_lambdaU / network_avg_tau | float | From 03 |
| max_CI_type / mean_CI | float | From 04 |
| y | int | 1 if month `t+lead` is in crisis regime, else 0 |

### `agent_predictions.csv`
Per-model probability forecasts. Columns: `date`, `lstm`, `xgb`,
`logit`, `actual`.

### `agent_walkforward_auc.csv`
Rolling walk-forward AUC of each model (columns: `date`, `lstm`, `xgb`,
`logit`, `actual`).

---

## 07 — Systemic risk index

### `csri_monthly.csv`
Composite Systemic Risk Index and its components.

| column | type | description |
|---|---|---|
| date | date | Month-start |
| CSRI | float | Mean of four component z-scores |
| lambdaU / mean_CI / max_CI / loss_div_ben | float | Raw components |
| z_lambdaU / z_mean_CI / z_max_CI / z_loss_div_ben | float | 60M rolling z-scores |
| regime | str | Crisis / stable from CSRI quantile |
| regime_exo | str | Exogenous regime label |

### `granger_results.csv`
Granger causality of CSRI → each type's CI.

| column | type | description |
|---|---|---|
| type | str | Asset class |
| lag | int | Lag order tested |
| F_stat | float | F statistic |
| p_value | float | p-value |
| reject_05 | bool | Rejects H₀ at 5 % |
| n_obs | int | Observations |

### `bootstrap_ci.csv`
Stationary block-bootstrap confidence intervals for headline statistics.

| column | type | description |
|---|---|---|
| statistic | str | Name (`lambdaU_full`, `lambdaU_crisis`, `amp`, ...) |
| point | float | Observed value |
| mean_boot / se_boot | float | Bootstrap mean & SE |
| ci_low / ci_high | float | 2.5 / 97.5 percentile |
| n_boot | int | Number of bootstrap replications |

---

## 09 — Robustness suite

### `robust_copula_family.csv`
Copula-family sensitivity of λ_U amplification.

| column | type | description |
|---|---|---|
| family | str | Gumbel_U / Student_U(nu=4) / Clayton_L |
| full / stable / crisis | float | Mean λ over each regime |
| amp_c_s | float | crisis / stable |

### `robust_threshold.csv`
Sensitivity to the crisis-threshold quantile.

| column | type | description |
|---|---|---|
| threshold_q | float | Quantile cutoff (0.60–0.90) |
| n_crisis | int | Crisis months under this threshold |
| DB_stable / DB_crisis | float | Diversification benefit in each regime |
| DB_drop | float | DB_stable − DB_crisis |
| lambdaU_amp | float | Crisis / stable λ_U |

### `robust_weights.csv`
Same shape as `portfolio_es_by_regime.csv` — ES under alternative
portfolio weightings (equal / budget / inverse-vol).

### `robust_permutation.csv`
Permutation test: is the observed λ_U amplification distinguishable
from a random shuffle of regime labels?

| column | type | description |
|---|---|---|
| n_perm | int | Number of permutations (10 000) |
| n_crisis / n_stable | int | Sub-sample sizes |
| amp_observed | float | Real amplification |
| amp_null_mean | float | Mean under the null |
| amp_null_ci_lo / amp_null_ci_hi | float | 95 % null band |
| p_value_one_sided | float | Fraction of null ≥ observed |

---

## 10 — R-vine copula

### `rvine_fit_summary.csv`
One row per fitted vine model.

| column | type | description |
|---|---|---|
| model | str | Vine label (e.g. `R-vine (all families)`) |
| loglik | float | Log-likelihood |
| aic / bic | float | Information criteria |
| npars | int | Number of parameters |

### `rvine_structure.csv`
Edge list of the fitted vine.

| column | type | description |
|---|---|---|
| vine | str | Sample label (`full` / `stable` / `crisis`) |
| tree | int | Tree level |
| edge | int | Edge index within the tree |
| family | str | Bivariate copula family name |
| rotation | int | Rotation (0 / 90 / 180 / 270) |
| tau | float | Kendall's τ on the edge |
| lambda_U | float | Upper-tail dependence on the edge |
| parameters | str | Comma-joined parameter vector |

### `rvine_comparison.csv`
R-vine vs. Gaussian benchmark by sub-sample.

| column | type | description |
|---|---|---|
| sample | str | `full` / `stable` / `crisis` |
| n | int | Observations |
| loglik / aic / bic | float | Fit statistics |
| tree1_avg_lambdaU | float | Mean λ_U on the tree-1 edges |

---

## 11 — DCC-GARCH + Diebold–Yilmaz benchmarks

### `dcc_correlations.csv`
Monthly DCC(1,1) pairwise correlations plus `dcc_avg` = mean of the 10
pairwise correlations.

### `dcc_vs_lambdaU.csv`
Regime-level comparison of DCC ρ̄ vs λ_U.

| column | type | description |
|---|---|---|
| regime | str | full / stable / crisis |
| n | int | Months |
| dcc_avg | float | Mean DCC correlation |
| lambdaU | float | Mean Gumbel λ_U |
| ratio | float | λ_U / dcc_avg |

### `dy_spillover.csv`
Diebold–Yilmaz GFEVD matrix (row = "from", col = "to"); last column
`FROM_others` is 100 minus diagonal.

### `dy_classification.csv`
Comparison of DY vs P5 source/receiver roles.

| column | type | description |
|---|---|---|
| type | str | Asset class |
| DY_NET / P5_NET | float | Net directional score |
| DY_role / P5_role | str | source / receiver |
| agree | bool | Roles match |

---

## 12 — LSTM lead-time curve

### `lead_time_auc.csv`
Walk-forward AUC curve as the forecast horizon grows.

| column | type | description |
|---|---|---|
| lead | int | Lead time (months: 1, 2, 3, 6, 12) |
| model | str | `lstm` / `xgb` / `logit` |
| seed | int | Random seed |
| auc | float | Walk-forward AUC at this lead |

---

## 13 — Case studies

### `case_study_windows.csv`
One row per historical crisis window.

| column | type | description |
|---|---|---|
| code | str | Short id (`GR2012`, `GR2015`, `UKR2022`) |
| name | str | Human-readable label |
| start / end | date | Window bounds (YYYY-MM) |
| trigger | str | Economic trigger |
| n_months | int | Length of window |
| CSRI_pre / CSRI_crisis | float | Mean CSRI before vs during |
| CSRI_peak | float | Max CSRI in window |
| CSRI_peak_date | date | When peak occurred |
| lambdaU_crisis / lambdaU_pre | float | Network λ_U pre vs during |
| top_source / top_receiver | str | Most active types in the window |

### `counterfactual_gap.csv`
The practitioner's €-gap: loss of the diversification benefit the
stable-regime owner *thought* they had.

| column | type | description |
|---|---|---|
| regime | str | `full` / `crisis` / `stable` |
| ES_sum | float | Σ ES of marginal types (comonotone upper bound) |
| ES_P | float | ES of the portfolio log-return |
| DB_actual | float | Observed diversification benefit `1 − ES_P/ES_sum` |
| DB_assumed | float | Stable-regime DB the practitioner assumed |
| naive_budget_eur | float | `(1 − DB_assumed) × ES_sum × total_budget` |
| true_loss_eur | float | `ES_P × total_budget` |
| shortfall_eur | float | true_loss − naive_budget |
| shortfall_pct | float | shortfall / naive_budget |

### `contingency_rules.csv`
Decision rules: contingency % per (type × duration bucket).

| column | type | description |
|---|---|---|
| type | str | Asset class |
| duration_bucket | str | short (<18M) / medium (18-30M) / long (>30M) |
| n | int | Projects in bucket |
| median_pct / mean_pct / p75_pct | float | Contingency distribution |
| mean_budget | float | Mean project budget in bucket |
| rule_pct | float | Recommended % (P75, rounded) |
| rule_text | str | Human-readable rule (`If type=... AND duration=... → add x.x%`) |
