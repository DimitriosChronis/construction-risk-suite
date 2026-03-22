"""
06_lifecycle_phasing.py
C4 -- Dynamic Lifecycle Phasing

Three construction phases with different commodity weight profiles.
Computes phase-specific ES(99%) using Gumbel copula.

Phases:
    Foundation     (M1-8):   Concrete-heavy (50%)
    Superstructure (M9-18):  Steel-heavy (50%)
    Completion     (M19-24): Fuel/PVC-heavy (60%)

Input:  data/processed/elstat_log_returns.csv
Output: results/tables/c4_lifecycle_es.csv
        results/figures/fig_c4_lifecycle_profile.pdf
"""

import pandas as pd
import numpy as np
from scipy.stats import norm, rankdata
import pyvinecopulib as pv
import matplotlib.pyplot as plt
import os

SEED      = 42
N_SIMS    = 100_000
BASE_COST = 2_300_000

PHASES = [
    ("Foundation",     8,  np.array([0.50, 0.20, 0.20, 0.10]), 0.30),
    ("Superstructure", 10, np.array([0.20, 0.50, 0.20, 0.10]), 0.50),
    ("Completion",     6,  np.array([0.15, 0.25, 0.35, 0.25]), 0.20),
]

COLS = ["GR_Concrete", "GR_Steel", "GR_Fuel_Energy", "GR_PVC_Pipes"]

# Use crisis regime (conservative)
CRISIS_START = "2021-01-01"
CRISIS_END   = "2024-12-01"

SCRIPT_DIR = os.path.dirname(__file__)
DATA_PATH  = os.path.join(SCRIPT_DIR, "..", "data", "processed",
                           "elstat_log_returns.csv")
OUT_FIG    = os.path.join(SCRIPT_DIR, "..", "results", "figures")
OUT_TAB    = os.path.join(SCRIPT_DIR, "..", "results", "tables")


def pseudo_obs(arr):
    n = arr.shape[0]
    return np.array([rankdata(arr[:, j]) / (n + 1)
                     for j in range(arr.shape[1])]).T


def simulate_all_phases(log_returns_df, phases, n_sims, base_cost, seed):
    """Simulate all phases from a single continuous copula draw.

    One vine is fitted, and all 24 months are drawn in a single block so
    that cross-phase temporal dependence is preserved.  The monthly draws
    are then split and weighted according to each phase's profile.
    """
    mu      = log_returns_df.mean().values
    sigma   = log_returns_df.std().values
    n_assets = log_returns_df.shape[1]
    total_months = sum(m for _, m, _, _ in phases)

    # Fit Gumbel vine once
    u_data = pseudo_obs(log_returns_df.values)
    controls = pv.FitControlsVinecop(
        family_set=[pv.BicopFamily.gumbel],
        selection_criterion="aic",
        num_threads=1
    )
    vine = pv.Vinecop(d=n_assets)
    vine.select(np.asfortranarray(u_data), controls)

    # Single continuous draw for the entire project horizon
    u_sim = vine.simulate(n_sims * total_months, seeds=[seed])
    u_sim = np.clip(u_sim, 1e-6, 1 - 1e-6)
    u_sim = u_sim.reshape(n_sims, total_months, n_assets)
    z = norm.ppf(u_sim)
    log_rets = mu + sigma * z  # (n_sims, total_months, n_assets)

    # Split into phases and aggregate with phase-specific weights
    phase_costs = {}
    t0 = 0
    for name, months, weights, budget_share in phases:
        budget = base_cost * budget_share
        phase_z = log_rets[:, t0:t0 + months, :]
        simple_returns = np.exp(phase_z) - 1.0
        portfolio_simple = (simple_returns * weights).sum(axis=2)
        portfolio_log = np.log1p(portfolio_simple)
        total_log_return = portfolio_log.sum(axis=1)
        phase_costs[name] = (budget, budget * np.exp(total_log_return))
        t0 += months

    return phase_costs


def es(costs, alpha):
    threshold = np.quantile(costs, alpha)
    tail = costs[costs >= threshold]
    return tail.mean() if len(tail) > 0 else threshold


def main():
    os.makedirs(OUT_FIG, exist_ok=True)
    os.makedirs(OUT_TAB, exist_ok=True)

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    available = [c for c in COLS if c in df.columns]
    crisis = df[available].loc[CRISIS_START:CRISIS_END].dropna()
    print(f"Crisis regime: {crisis.shape[0]} obs")

    print("Simulating all phases from single continuous copula draw...")
    phase_costs = simulate_all_phases(crisis, PHASES, N_SIMS, BASE_COST, SEED)

    rows = []
    for name, months, weights, budget_share in PHASES:
        budget, costs = phase_costs[name]

        p85  = np.quantile(costs, 0.85)
        p99  = np.quantile(costs, 0.99)
        es99 = es(costs, 0.99)
        overrun_pct = (es99 / budget - 1) * 100

        rows.append({
            "Phase":          name,
            "Months":         months,
            "Budget_EUR":     round(budget),
            "P85_EUR":        round(p85),
            "P99_EUR":        round(p99),
            "ES99_EUR":       round(es99),
            "ES99_Overrun_%": round(overrun_pct, 1),
        })
        print(f"  {name:16s}  budget={budget:>10,.0f}  P85={p85:>10,.0f}  "
              f"ES99={es99:>10,.0f}  overrun={overrun_pct:+.1f}%")

    res_df = pd.DataFrame(rows)
    res_df.to_csv(os.path.join(OUT_TAB, "c4_lifecycle_es.csv"), index=False)

    # Bar chart
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#2196F3", "#FF5722", "#4CAF50"]
    x = range(len(res_df))
    bars_p85 = ax.bar([i - 0.15 for i in x], res_df["P85_EUR"] / 1e6,
                      width=0.3, color=colors, alpha=0.5, label="P85")
    bars_es  = ax.bar([i + 0.15 for i in x], res_df["ES99_EUR"] / 1e6,
                      width=0.3, color=colors, edgecolor="black",
                      linewidth=0.8, label="ES(99%)")

    ax.set_xticks(list(x))
    ax.set_xticklabels(res_df["Phase"])
    ax.set_ylabel("Cost (EUR millions)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG, "fig_c4_lifecycle_profile.pdf"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("\nSaved: c4_lifecycle_es.csv, fig_c4_lifecycle_profile.pdf")


if __name__ == "__main__":
    main()
