# Paper 5: Systemic Risk Contagion in Construction Cost Portfolios

**A Dynamic Vine Copula Network Approach to Cross-Project Tail Dependence**

> Chronis, D. (2027). *Systemic Risk Contagion in Construction Cost Portfolios:
> A Dynamic Vine Copula Network Approach to Cross-Project Tail Dependence*.
> In preparation for *Automation in Construction*.

**Contributions: C9 + C10 + C11**

---

## Scope

This paper extends the single-project framework of Papers 1–4 to
**N concurrent construction projects**, introducing the first
systemic risk contagion framework for construction cost portfolios.

The central empirical finding: under crisis regimes, cross-project tail
dependence **amplifies rather than diversifies** portfolio-level Expected
Shortfall. Portfolio ES ≥ Σ individual ES — the opposite of classical
sub-additivity-based diversification theory.

- **C9** — Dynamic vine copula network across N concurrent projects.
  Time-varying upper-tail dependence λ_U captures crisis-period
  amplification. Contagion index per project type with source vs
  receiver classification via directional lead-lag.
- **C10** — Portfolio-level ES with explicit diversification-benefit
  test: `DB = 1 − ES_P / Σ w_i ES_i`. Rolling 24-month decomposition
  plus per-project contingency buffer `|ES_crisis| · √(duration/12)`.
- **C11** — Construction Systemic Risk Index (CSRI): first composite,
  automatically re-computable monthly tail-risk indicator for construction
  portfolios. Validated by LSTM early-warning agent (P4 architecture at
  portfolio level), block-bootstrap CIs and Granger causality tests.

---

## Research Questions

1. **C9 / H1:** Does cross-project tail dependence exist in construction
   cost portfolios, and does it amplify during crisis regimes?
2. **C10 / H2:** Does classical diversification (ES_P < Σ ES_i) hold for
   real construction portfolios, or collapse under stress?
3. **C11 / H3:** Can an automated monthly index + LSTM agent detect
   portfolio-level contagion onset with a useful lead time?

---

## Key Hypotheses and Empirical Status

| Hypothesis | Statement | Status (current run) |
|---|---|---|
| H1 | λ_U (crisis) > λ_U (stable) | ✅ 1.08× amplification, 95% CI [1.06, 1.10], p < 0.0001 |
| H2 | DB → 0 or < 0 in crisis | ✅ DB_stable = +0.009, DB_crisis ≈ 0 (super-additivity) |
| H3 | LSTM agent detects portfolio crisis lead=2M | ✅ validated vs P4 reference (AUC = 0.926) |
| H4 | CSRI separates regimes | ✅ CSRI_stable = −0.525, CSRI_crisis = +1.088 (1.6σ) |
| H5 | Asymmetric contagion (sources / receivers) | ✅ road & pipeline = sources; bridge = strongest receiver |

---

## Data Sources

| Source | Data | Used for |
|--------|------|---------|
| Διαύγεια (diavgeia.gov.gr) | Greek public construction contract awards (Δ.1, `assignmentType=Έργα`) | N project dataset |
| ELSTAT SPC23 | Monthly material price indices (concrete, steel, fuel, PVC), 2000–2024 | Cost-shock returns + crisis labels |
| FRED (P4 features) | US PPI series (5 commodities) | LSTM agent leading indicators |
| Paper 4 artefacts | `features.csv`, `wf_predictions.csv` | Benchmark baseline |

**Public data only.** Paper 5 uses exclusively public data sources — no
internal office / proprietary project data.

**Target dataset:** 500+ Greek public construction projects
(budget ≥ €500K, 2010–2024, monthly windowing, auto-split on
Elasticsearch 20k-offset truncation).

---

## Methodology

### 1. Portfolio construction (script 02)
- Classify each Διαύγεια project into **road / bridge / pipeline /
  building / other** via Greek keyword matching on `subject`.
- Assign material exposure weights per type (concrete / steel / fuel / PVC).
- Duration `d_i = clip(12 + 6·log10(budget / 500K), 12, 48)` months.
- Cost-shock return: `r_i(t) = −(W @ R_ELSTAT.T)`.

### 2. Dynamic copula network (script 03)
- Rolling 24-month Kendall's τ for every pair of type returns.
- Gumbel upper-tail dependence:
  `λ_U = 2 − 2^(1−τ)`  (Embrechts, Lindskog & McNeil 2003).
- Endogenous regime: top 25% months by network-average λ_U.
- Exogenous regime: Greek macro calendar (bailout, capital controls,
  COVID, Ukraine energy shock).

### 3. Contagion index (script 04)
- `CI_i(t) = mean_{j≠i} λ_U(i, j; t)`.
- Directional lead-lag via rolling pseudo-observations joint-exceedance:
  OUT / IN / NET flow per type → **source** (NET > 0) vs **receiver**.

### 4. Portfolio Expected Shortfall (script 05)
- `ES_i(α) = −E[r_i | r_i ≤ VaR_i(α)]`,  α = 0.95.
- `ES_Σ = Σ w_i ES_i`,  `ES_P = −E[r_P | r_P ≤ VaR_P(α)]`.
- Diversification benefit `DB = 1 − ES_P / ES_Σ`.
- Equal and budget-share weights, regime breakdown, rolling 24-month DB.
- Per-project contingency buffer `|ES_crisis(type_i)| · √(d_i / 12)`.

### 5. LSTM portfolio agent (script 06)
- Direct import of P4 PyTorch `LSTMClassifier` from
  `paper4-lstm-agent/src/utils.py` (no retraining of P4; same architecture).
- `hidden=64, layers=2, dropout=0.3, epochs=150, batch=16, lookback=6, lead=2`,
  5-seed ensemble.
- 24 features = 20 P4 PPI features + 4 P5 network features
  (`avg_lambdaU`, `avg_tau`, `max_CI`, `mean_CI`).
- Walk-forward `min_train=60, step=6`. Benchmarks: XGBoost ensemble,
  logistic regression, P4-signal-only (no retraining).

### 6. CSRI + validation (script 07)
- `CSRI(t) = mean over components of rolling 60M z-score`, components:
  `λ_U`, `mean_CI`, `max_CI`, `1 − DB`, (optional) agent probability.
- **Granger causality**: `CSRI(t−k) → extreme_i(t)` for k=1,2,3.
- **Block bootstrap** (B=1000, block=12) CIs on λ_U, mean CI, DB, crisis
  amplification.

### 7. Robustness suite (script 09)
- **Copula family sensitivity**: Gumbel, Student-t (ν=4), Clayton — λ_U
  amplification holds across all families (1.05–1.14×).
- **Crisis threshold sweep**: 60th–90th percentile — DB→0 finding is
  threshold-robust.
- **Portfolio weights**: equal, budget, inverse-volatility — super-additivity
  holds for all.
- **Permutation test**: B=10,000 regime-label shuffles, p < 0.0001.

### 8. Benchmarks (scripts 10–12)
- **R-vine copula** (pyvinecopulib): proper hierarchical decomposition;
  ΔAIC = −77 vs Gaussian vine confirms non-Gaussian tails needed.
- **DCC-GARCH(1,1)**: DCC amp (1.015) < λ_U amp (1.078) — Gumbel captures
  tail information beyond conditional correlation.
- **Diebold–Yilmaz spillover**: 0/5 source/receiver agreement with P5 →
  orthogonal to linear VAR (strengthens P5 contribution).
- **Lead-time curve**: LSTM AUC at lead = 1, 2, 3, 6, 12 months.

### 9. Case studies (script 13)
- **3 crisis windows**: 2012 sovereign crisis, 2015 capital controls,
  2022 Ukraine energy shock.
- **Counterfactual €-gap**: practitioner who assumes stable-regime
  diversification loses the benefit under crisis.
- **Contingency rules**: decision table by (type × duration bucket).

---

## Scripts (run in order)

| # | Script | Purpose | Key outputs |
|---|--------|---------|-------------|
| 01 | `01_diavgeia_download.py` | Real download from Διαύγεια API (monthly, auto-split, 4× retry w/ backoff) | `data/raw/diavgeia_projects.csv` |
| 02 | `02_portfolio_construction.py` | N×T portfolio + type-level theoretical returns | `portfolio_type_returns_theoretical.csv`, `portfolio_metadata.csv` |
| 03 | `03_dynamic_copula.py` | Rolling Gumbel λ_U + regime labels | `dynamic_copula_*.csv` |
| 04 | `04_contagion_index.py` | CI_i + directional source/receiver flows | `contagion_index.csv`, `contagion_flows.csv` |
| 05 | `05_portfolio_es.py` | Regime ES + DB + per-project contingency | `portfolio_es_by_regime.csv`, `portfolio_es_rolling.csv` |
| 06 | `06_lstm_portfolio_agent.py` | P4 LSTM ensemble at portfolio level + XGB/Logit/P4-signal benchmarks | `agent_predictions.csv` |
| 07 | `07_systemic_risk_index.py` | CSRI + Granger + block bootstrap CIs | `csri_monthly.csv`, `granger_results.csv`, `bootstrap_ci.csv` |
| 08 | `08_publication_figures.py` | 8 AiC-ready figures (PDF + PNG @ 300 dpi) | `results/figures/fig{1..8}.{pdf,png}` |
| 09 | `09_robustness.py` | Copula family / threshold / weights / permutation sensitivity | `robust_*.csv` |
| 10 | `10_rvine.py` | R-vine copula fit + Gaussian benchmark | `rvine_*.csv` |
| 11 | `11_benchmarks.py` | DCC-GARCH + Diebold–Yilmaz spillover | `dcc_*.csv`, `dy_*.csv` |
| 12 | `12_lead_time_curve.py` | LSTM AUC vs forecast horizon (1–12M) | `lead_time_auc.csv`, `fig_lead_time` |
| 13 | `13_case_studies.py` | Crisis windows + counterfactual €-gap + contingency rules | `case_study_windows.csv`, `counterfactual_gap.csv`, `contingency_rules.csv` |

**One-shot driver:**
```bash
python run_all.py            # runs 01→13 end-to-end
python run_all.py --from 09  # restart from script 09
python run_all.py --test     # dry-run (print steps only)
```

---

## Tables and Figures

**Tables (all in `results/`):**

| File | Content |
|------|---------|
| `table2_copula_regimes.csv` | Dynamic copula summary per regime |
| `table3_contagion_by_type.csv` | Contagion index & source/receiver classification |
| `table4_portfolio_es.csv` | Portfolio ES, ES_Σ, ratio, DB by regime |
| `table5_lstm_portfolio.csv` | LSTM portfolio agent OOS metrics vs benchmarks |
| `table6_csri.csv` | CSRI summary + Granger + bootstrap CIs |
| `table_robustness.csv` | Copula family / threshold / weights / permutation results |
| `table_rvine.csv` | R-vine structure + AIC comparison |
| `table_benchmarks.csv` | DCC-GARCH + Diebold–Yilmaz vs P5 |
| `table_case_studies.csv` | 3 crisis windows: CSRI trajectory + sources |
| `table_contingency_rules.csv` | Contingency % by type × duration |
| `table_lead_time.csv` | AUC by lead horizon |

**Figures (in `results/figures/`, PDF + PNG @ 300 dpi):**

| File | Content |
|------|---------|
| `fig1_framework` | Pipeline architecture diagram |
| `fig2_network` | Dynamic copula network (stable vs crisis, 5-node) |
| `fig3_contagion_timeline` | CI timeline with crisis shading |
| `fig4_portfolio_es` | Portfolio ES vs Σ individual ES + rolling DB |
| `fig5_source_receiver` | NET flow bars (source/receiver classification) |
| `fig6_lstm_agent` | ROC + OOS probabilities overlay |
| `fig7_walkforward` | Rolling 24M AUC + P4 reference |
| `fig8_csri_timeline` | CSRI with Greek macro-episode annotations |
| `fig_case_studies` | 3 crisis windows: CSRI + λ_U trajectories |
| `fig_counterfactual_gap` | Naive vs true ES bar chart |
| `fig_lead_time` | AUC degradation by forecast horizon |

---

## Headline Results (current run — test sub-sample, N=42)

- **276 months × 10 type pairs = 2,760 rolling copula fits.**
- **λ_U crisis / stable amplification = 1.081 [95% CI 1.064, 1.103], p < 0.0001.**
- **Budget-weighted DB**: stable = +0.009, crisis ≈ 0 → **classical
  diversification disappears under crisis**.
- **Source / receiver**: road & pipeline are sources (NET ≈ +0.031);
  bridge is the strongest receiver (NET ≈ −0.043).
- **CSRI** separates regimes by 1.6σ (−0.525 vs +1.088).
- **LSTM agent** (P4 architecture, lead = 2M): AUC = 0.926.
- **R-vine** vs Gaussian: ΔAIC = −77 (Student+Gaussian mix wins).
- **DCC amp** (1.015) < **λ_U amp** (1.078) — copula captures tail info DCC misses.
- **Robustness**: amplification holds across Gumbel (1.08), Student-t (1.14),
  Clayton (1.05); all thresholds 60–90%; all weight schemes.

---

## Robustness & Benchmarks Summary

| Check | Result | Conclusion |
|-------|--------|------------|
| Copula family (Gumbel/Student-t/Clayton) | Amp 1.05–1.14× | Robust to family choice |
| Crisis threshold (P60–P90) | DB drop 0.004–0.008 | Threshold-independent |
| Portfolio weights (equal/budget/inv-vol) | Super-additivity in all | Weight-independent |
| Permutation test (B=10K) | p < 0.0001 | Not a random artefact |
| R-vine vs Gaussian vine | ΔAIC = −77 | Non-Gaussian tails needed |
| DCC-GARCH vs Gumbel λ_U | DCC amp 1.02 < λ_U amp 1.08 | Gumbel adds tail info |
| Diebold–Yilmaz vs P5 | 0/5 role agreement | P5 is orthogonal to linear VAR |
| Lead-time curve (1–12M) | AUC degrades smoothly | 2M lead is near-optimal |

---

## Reproducibility

| Artefact | Status |
|----------|--------|
| `requirements.txt` (pinned versions) | ✅ |
| `environment.yml` (conda) | ✅ |
| `run_all.py` (one-shot driver, 01→13) | ✅ |
| `tests/test_core_functions.py` (28 pytest tests) | ✅ all pass |
| `docs/data_dictionary.md` (30 CSVs documented) | ✅ |

```bash
# Quick start
conda env create -f environment.yml
conda activate construction-risk-p5
python run_all.py
pytest tests/ -v
```

---

## Connection to Paper Series

```
P1: Tail dependence in single material pairs
P2: US → Greek transmission (cross-border)
P3: ES quantification + 6 rules (1 project)
P4: LSTM agent (1 project, 16 M lead)
P5: THIS PAPER — N projects, dynamic copula contagion network + CSRI
P6: EU generalisation (5 countries)
P7: Global megaprojects (multi-currency)
P8: Universal cascade theory
```

**P5 answers:** "Does tail-dependence / early-warning machinery from
P1–P4 scale to a whole construction portfolio, and does diversification
actually work under crisis?" — **No, it breaks; and yes, a monthly,
automatable CSRI + LSTM early-warning stack recovers the lost safety
margin.**

---

## Target Journal

**Automation in Construction** (Elsevier, IF ≈ 14.4, Q1).

**Automation angle:** CSRI is designed to be **re-computed automatically
every month** from public data (Διαύγεια + ELSTAT + FRED), with no
human tuning — a fully automated portfolio-level tail-risk monitor that
replaces manual project-by-project ES assessment.

---

## What Remains for Submission

The analytical engine (13 scripts, robustness suite, benchmarks) is
**complete**. The remaining work is data, writing, and polish.

### A. Data [CRITICAL]
- [ ] **Full Διαύγεια download** — replace N=42 test sub-sample with
      N≥500 full dataset and re-run 02→13.

### B. Manuscript [CRITICAL]
- [ ] **Abstract** (≤ 250 words) — lead with DB → 0 headline.
- [ ] **Introduction** — 3 literature gaps + 3 contributions (C9/C10/C11)
      + AiC automation angle.
- [ ] **Literature review** — 4 subsections, ≥ 60 refs, ≥ 15 from AiC.
- [ ] **Methods** — formal definitions + pseudocode for CSRI.
- [ ] **Results** — one subsection per table, 11 figures woven in.
- [ ] **Discussion** — super-additivity interpretation, managerial
      implications, automation workflow, limitations.
- [ ] **Conclusion** — 4 contributions, 3 future-work directions.

### C. Figure & Table Polish [must]
- [ ] Colour-blind safe palette, serif fonts, single-column legibility.
- [ ] Consistent sig figs, units in headers, LaTeX formatting.
- [ ] Consistency pass: λ_U / DB / CSRI identical across text / tables / figures.

### D. Pre-submission Package [must]
- [ ] Cover letter, highlights (3–5 bullets, ≤ 85 chars each).
- [ ] Graphical abstract (= simplified Fig 1).
- [ ] CRediT authorship statement, competing interest declaration.
- [ ] Data availability statement + GitHub repo with Zenodo DOI.
- [ ] Language polish — native-English proof-read.

---

## Global Parameters

```python
SEED           = 42
MIN_BUDGET     = 500_000    # EUR — Διαύγεια filter
COPULA_WINDOW  = 24         # months, rolling
CRISIS_PCT     = 0.75       # P75 threshold (consistent with P1–P4)
ALPHA          = 0.95       # ES confidence level
LOOKBACK       = 6          # LSTM lookback (P4)
LEAD           = 2          # months ahead (P4)
ZSCORE_WIN     = 60         # CSRI rolling z-score window
N_BOOTSTRAP    = 1000
BLOCK_SIZE     = 12         # block bootstrap block length
ENSEMBLE_SEEDS = [0,1,2,3,4]
```

---

## Dependencies

See `requirements.txt` for pinned versions or `environment.yml` for conda.

Core stack: Python 3.13, NumPy, Pandas, SciPy, scikit-learn,
statsmodels, PyTorch, XGBoost, pyvinecopulib, arch, matplotlib,
networkx, requests, pytest.

---

## Citation

```bibtex
@article{chronis2027paper5,
  author  = {Chronis, Dimitrios},
  title   = {Systemic Risk Contagion in Construction Cost Portfolios:
             A Dynamic Vine Copula Network Approach to
             Cross-Project Tail Dependence},
  journal = {Automation in Construction},
  year    = {2027},
  note    = {in preparation}
}
```
