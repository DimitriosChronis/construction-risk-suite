# Paper 3: Tail Risk Quantification and Cross-Border Hedging for Construction Cost Overruns

**Expected Shortfall, Lifecycle Phasing, and Basis Risk Analysis Using Vine Copulas**

> Chronis, D. (2026). *Tail Risk Quantification and Cross-Border Hedging for Construction Cost Overruns: Expected Shortfall, Lifecycle Phasing, and Basis Risk Analysis Using Vine Copulas*. Submitted to *Automation in Construction*, Elsevier.

**Contributions: C3 + C4 + C5**

---

## Scope

- **C3** — Basel III Expected Shortfall ES(99%) via 4-D Gumbel vine copula Monte Carlo. Regime-conditional analysis (stable 2014–2019 vs crisis 2021–2024). Gumbel premium 6× larger in crisis than stable. Formal Kupiec + Christoffersen backtesting.
- **C4** — Construction lifecycle phase decomposition: Foundation (8M), Superstructure (10M), Completion (6M). Phase-specific ES with bootstrap 95% CIs (B=500). Superstructure phase carries +14.2% overrun risk.
- **C5** — Cross-border hedging analysis: OLS hedge ratios, rolling hedge effectiveness, Engle–Granger cointegration, basis risk gap quantification. Steel only pair achieving HE = 25.5%; collapses to 2.7% in crisis.

---

## Key Finding

This paper reveals that **tail risk is regime-conditional, phase-heterogeneous, and structurally unhedgeable**:

| Finding | Result |
|---------|--------|
| Crisis ES(99%) | EUR 2,944,866 — 28% overrun above EUR 2.3M base |
| Gumbel premium amplification | 6.0× in crisis vs stable (EUR +28,733 vs EUR −4,784) |
| Dominant tail risk driver | Fuel/Energy: 53% of tail losses despite 20% weight (4.28× amplification) |
| Highest-risk phase | Superstructure: ES99 = EUR 1,313,153, +14.2% overrun |
| Hedging | Steel HE = 25.5% full period → 2.7% in crisis. 4/5 pairs non-cointegrated |
| Policy implication | Contractual risk transfer dominates financial hedging for small open economies |

---

## Scripts (run in order)

| # | Script | Purpose | Key output |
|---|--------|---------|------------|
| 01 | `01_global_data_download.py` | Download US PPI series from FRED API | `data/raw/global_commodities_monthly.csv` |
| 02 | `02_align_datasets.py` | Align ELSTAT + FRED, compute log-returns | `data/processed/aligned_log_returns.csv` |
| 05 | `05_expected_shortfall.py` | ES(99%) under Independent/Gaussian/Gumbel × 3 regimes | `results/tables/c3_es_comparison.csv` |
| 05b | `05b_rolling_es_backtest.py` | Rolling 24M ES + Kupiec/Christoffersen backtests | `results/tables/c3b_rolling_es.csv` |
| 05c | `05c_regime_switching_es.py` | Regime-conditional ES (stable vs crisis) + Gumbel premium | `results/tables/c3c_regime_es.csv` |
| 05d | `05d_es_backtest_formal.py` | Formal expanding-window backtest (39 windows) | `results/tables/c3d_backtest.csv` |
| 05e | `05e_decision_rules.py` | ES → procurement decision rules (R1–R6) | `results/tables/c3e_decision_rules.csv` |
| 05f | `05f_es_decomposition.py` | Marginal ES contribution per material | `results/tables/c3f_es_decomposition.csv` |
| 06 | `06_lifecycle_phasing.py` | Phase-specific ES (Foundation/Superstructure/Completion) | `results/tables/c4_lifecycle_es.csv` |
| 06b | `06b_bootstrap_ci.py` | Bootstrap 95% CI on phase ES (B=500, N=20,000 each) | `results/tables/c4b_bootstrap_ci.csv` |
| 06c | `06c_phase_sensitivity.py` | Weight sensitivity ±10pp on dominant material | `results/tables/c4c_phase_sensitivity.csv` |
| 07 | `07_hedging_quantification.py` | Joint 6D vine copula + OLS hedge ratios + ES reduction | `results/tables/c5_hedge_es.csv` |
| 07b | `07b_hedge_effectiveness.py` | Static + rolling HE (full period vs crisis) | `results/tables/c5b_hedge_effectiveness.csv` |
| 07c | `07c_basis_risk_breakeven.py` | Basis risk gap (HE=25%, HE=50%) + break-even cost | `results/tables/c5c_basis_risk.csv` |
| 07d | `07d_cointegration_test.py` | Engle–Granger cointegration tests (price levels) | `results/tables/c5d_cointegration.csv` |
| 07e | `07e_rolling_correlation.py` | Rolling 24M Pearson + Kendall correlation | `results/tables/c5e_rolling_corr.csv` |
| 08 | `08_publication_figures.py` | All publication-quality figures | `results/figures/` |

---

## Key Outputs

### Tables
| File | Description |
|------|-------------|
| `c3_es_comparison.csv` | ES(99%), P85, P99 under 3 copulas × 3 regimes |
| `c3b_rolling_es.csv` | Rolling 24M Gumbel vs Gaussian ES (every 6M) |
| `c3c_regime_es.csv` | Regime-conditional ES + Gumbel premium |
| `c3d_backtest.csv` | Kupiec PoF + Christoffersen independence test results |
| `c3e_decision_rules.csv` | Procurement decision rules R1–R6 with EUR amounts |
| `c3f_es_decomposition.csv` | Marginal ES contribution + amplification factor per material |
| `c4_lifecycle_es.csv` | Phase-specific ES(99%), P85, P99, overrun % |
| `c4b_bootstrap_ci.csv` | Bootstrap 95% CI for phase ES (B=500) |
| `c4c_phase_sensitivity.csv` | ES sensitivity to ±10pp weight perturbation |
| `c5_hedge_es.csv` | Unhedged vs hedged ES, net benefit |
| `c5b_hedge_effectiveness.csv` | HE full period vs crisis for all pairs |
| `c5c_basis_risk.csv` | Basis risk gap to HE=25% and HE=50% targets |
| `c5d_cointegration.csv` | Engle–Granger test statistics and p-values |
| `c5e_rolling_corr.csv` | Rolling correlation: mean, std, hedgeable % of windows |

### Figures
| File | Description |
|------|-------------|
| `fig1_es_comparison.pdf` | ES(99%) and P85 across copulas and regimes |
| `fig2_rolling_es.pdf` | Rolling ES: Gumbel vs Gaussian (2000–2024) |
| `fig3_regime_es.pdf` | Regime-conditional ES bar chart + Gumbel premium |
| `fig_c3c_regime_timeline.pdf` | Greek log-returns with stable/crisis shading |
| `fig_c3f_es_decomposition.pdf` | Marginal ES contribution by material |
| `fig5_lifecycle_profile.pdf` | Phase-specific P85 and ES(99%) |
| `fig6_bootstrap_ci.pdf` | Bootstrap 95% CI on phase ES |
| `fig8_hedge_effectiveness.pdf` | HE full period vs crisis |
| `fig7_hedge_waterfall.pdf` | ES reduction waterfall (unhedged → hedged) |
| `fig9_basis_risk.pdf` | Basis risk gap for HE=25% and HE=50% targets |
| `fig_c5b_rolling_hr.pdf` | Rolling 24M hedge ratios for all pairs |
| `fig_c5e_rolling_correlation.pdf` | Rolling correlation with HE=25% threshold line |

---

## Data Sources

Same as Papers 1 and 2. See [`../shared-data/README.md`](../shared-data/README.md).

| Series | Source | Used for |
|--------|--------|---------|
| Greek Concrete, Steel, Fuel, PVC indices | [ELSTAT SPC23](https://www.statistics.gr/en/statistics/-/publication/SPC23/) | C3, C4 (4D vine) |
| US Steel, Cement, Fuel, PVC PPIs + Brent | [FRED](https://fred.stlouisfed.org) | C5 (hedging analysis) |

**Aligned dataset:** 299 monthly log-returns × 10 series (February 2000 – December 2024)

---

## Global Parameters

```python
SEED           = 42
BASE_COST      = 2_300_000    # EUR
HORIZON        = 24           # months
N_SIMS         = 100_000      # Monte Carlo (50,000 for rolling/sensitivity)
CONFIDENCE     = 0.99         # ES level
BOOTSTRAP_REPS = 500          # lifecycle phase CIs
ROLLING_WINDOW = 24           # months
```

---

## Results Summary

### C3 — Expected Shortfall ES(99%) — Base cost EUR 2,300,000, T=24 months

| Regime | Copula | ES(99%) EUR | Overrun % |
|--------|--------|-------------|-----------|
| Full (2000–2024) | Independent | 2,600,429 | +13.1% |
| Full (2000–2024) | Gaussian | 2,646,364 | +15.1% |
| Full (2000–2024) | Gumbel | 2,649,424 | +15.2% |
| Stable (2014–2019) | Gumbel | 2,394,721 | +4.1% |
| **Crisis (2021–2024)** | **Gumbel** | **2,944,866** | **+28.0%** |

**Gumbel premium:** EUR −4,784 (stable) → EUR +28,733 (crisis) — **6.0× amplification**

**Marginal ES decomposition (crisis regime):**

| Material | Weight | ES Contribution | Share | Amplification |
|----------|--------|-----------------|-------|---------------|
| Fuel/Energy | 20% | EUR 339,721 | 53% | **4.28×** |
| Steel | 30% | EUR 153,932 | 24% | 1.26× |
| Concrete | 30% | EUR 97,498 | 15% | 1.17× |
| PVC | 20% | EUR 86,671 | 8% | 1.15× |

### C4 — Lifecycle Phase ES (Gumbel, crisis regime)

| Phase | Budget | Months | ES(99%) | Overrun | Bootstrap 95% CI |
|-------|--------|--------|---------|---------|-----------------|
| Foundation | EUR 690,000 | 1–8 | EUR 765,763 | +11.0% | [740K, 788K] |
| **Superstructure** | **EUR 1,150,000** | **9–18** | **EUR 1,313,153** | **+14.2%** | **[1,258K, 1,355K]** |
| Completion | EUR 460,000 | 19–24 | EUR 518,244 | +12.7% | [498K, 535K] |

### C5 — Cross-Border Hedging

| Pair | HE full | HE crisis | Cointegrated? | Gap to HE=25% |
|------|---------|-----------|---------------|---------------|
| **Steel** | **25.5%** | **2.7%** | **Yes (marginal)** | **−0.005 ✓** |
| Cement | 2.4% | 0.7% | No | +0.345 |
| Fuel | 0.5% | 0.4% | No | +0.427 |
| PVC | 3.0% | 0.0% | No | +0.326 |

Net simulation-based hedge benefit: **EUR 45,923** (1.6% of crisis ES) — economically marginal relative to EUR 645,000 total overrun.

---

## Procurement Decision Rules

| Rule | Trigger | Action | EUR |
|------|---------|--------|-----|
| R1 | Crisis regime detected | Set total contingency to ES(99%)−base (28%) | 644,866 |
| R2 | Superstructure start | Allocate 55% of contingency; pre-purchase steel | 163,153 |
| R3 | Foundation start | Allocate 25%; lock concrete via framework agreement | 75,763 |
| R4 | Completion start | Allocate 20%; monitor fuel; re-tender if fuel > P80 | 58,244 |
| R5 | Steel procurement > EUR 200K | Consider Steel PPI swap only if rolling ρ > 0.40 | 46,504 |
| R6 | Monthly monitoring | If composite vol > 67th pctile → switch to crisis ES | 181,617 |

---

## Citation

```bibtex
@article{chronis2026paper3,
  author  = {Chronis, Dimitrios},
  title   = {Tail Risk Quantification and Cross-Border Hedging for
             Construction Cost Overruns: Expected Shortfall, Lifecycle
             Phasing, and Basis Risk Analysis Using Vine Copulas},
  journal = {Automation in Construction},
  year    = {2026},
  note    = {under review},
  url     = {https://github.com/dimitrioschronis/construction-risk-suite}
}
```
