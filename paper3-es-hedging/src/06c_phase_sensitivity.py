"""
06c_phase_sensitivity.py
Fix 3: Phase Weight Sensitivity Analysis

Perturbs dominant material weight by +/-10pp and recomputes ES(99%).
Shows how sensitive phase risk is to weight assumptions.

Input:  data/processed/elstat_log_returns.csv
Output: results/tables/c4c_phase_sensitivity.csv
"""

import os
import warnings

import numpy as np
import pandas as pd
import pyvinecopulib as pv
from scipy.stats import norm, rankdata

warnings.filterwarnings("ignore")

SEED       = 42
N_SIMS     = 50_000
BASE_COST  = 2_300_000
COLS       = ["GR_Concrete", "GR_Steel", "GR_Fuel_Energy", "GR_PVC_Pipes"]
CRISIS_START = "2021-01-01"
CRISIS_END   = "2024-12-01"

# Base phases: (name, months, weights[Concrete,Steel,Fuel,PVC], budget_share)
PHASES = [
    ("Foundation",     8,  np.array([0.50, 0.20, 0.20, 0.10]), 0.30),
    ("Superstructure", 10, np.array([0.20, 0.50, 0.20, 0.10]), 0.50),
    ("Completion",     6,  np.array([0.15, 0.25, 0.35, 0.25]), 0.20),
]

# Dominant material index per phase
DOMINANT = [0, 1, 2]  # Concrete, Steel, Fuel

SCRIPT_DIR = os.path.dirname(__file__)
DATA_PATH  = os.path.join(SCRIPT_DIR, "..", "data", "processed",
                           "elstat_log_returns.csv")
OUT_TAB    = os.path.join(SCRIPT_DIR, "..", "results", "tables")


def pseudo_obs(arr):
    n = arr.shape[0]
    return np.array([rankdata(arr[:, j]) / (n + 1)
                     for j in range(arr.shape[1])]).T


def simulate_phase_es(train_df, weights, n_months, budget, n_sims, seed):
    """Gumbel vine simulation for one phase, return ES(99%)."""
    mu = train_df.mean().values
    sigma = train_df.std().values
    n_assets = len(weights)

    u_data = pseudo_obs(train_df.values)
    controls = pv.FitControlsVinecop(
        family_set=[pv.BicopFamily.gumbel],
        selection_criterion="aic", num_threads=1)
    vine = pv.Vinecop(d=n_assets)
    vine.select(np.asfortranarray(u_data), controls)

    u_sim = vine.simulate(n_sims * n_months, seeds=[seed])
    u_sim = np.clip(u_sim, 1e-6, 1 - 1e-6).reshape(n_sims, n_months, n_assets)
    z = norm.ppf(u_sim)

    log_rets = mu + sigma * z
    simple_rets = np.exp(log_rets) - 1.0
    port_simple = (simple_rets * weights).sum(axis=2)
    port_log = np.log1p(port_simple)
    total_log = port_log.sum(axis=1)
    costs = budget * np.exp(total_log)

    var99 = np.quantile(costs, 0.99)
    tail = costs[costs >= var99]
    return tail.mean() if len(tail) > 0 else var99


def perturb_weights(base_weights, dominant_idx, delta):
    """Shift dominant material weight by delta, redistribute to others."""
    w = base_weights.copy()
    old_dom = w[dominant_idx]
    new_dom = max(0.05, min(0.90, old_dom + delta))
    actual_delta = new_dom - old_dom
    w[dominant_idx] = new_dom

    # Redistribute proportionally among non-dominant
    others = [i for i in range(len(w)) if i != dominant_idx]
    other_sum = sum(w[i] for i in others)
    if other_sum > 0:
        for i in others:
            w[i] -= actual_delta * (w[i] / other_sum)
            w[i] = max(0.01, w[i])

    # Renormalize to sum to 1
    w = w / w.sum()
    return w


def main():
    os.makedirs(OUT_TAB, exist_ok=True)

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    available = [c for c in COLS if c in df.columns]
    crisis = df[available].loc[CRISIS_START:CRISIS_END].dropna()
    print(f"Crisis data: {len(crisis)} obs\n")

    mat_names = ["Concrete", "Steel", "Fuel", "PVC"]
    deltas = [-0.10, -0.05, 0.0, +0.05, +0.10]

    rows = []
    for phase_idx, (name, months, base_w, budget_share) in enumerate(PHASES):
        budget = BASE_COST * budget_share
        dom_idx = DOMINANT[phase_idx]
        dom_name = mat_names[dom_idx]
        print(f"Phase: {name} (dominant: {dom_name}, base={base_w[dom_idx]:.0%})")

        for delta in deltas:
            w = perturb_weights(base_w, dom_idx, delta)
            es99 = simulate_phase_es(crisis, w, months, budget, N_SIMS, SEED)
            overrun = (es99 / budget - 1) * 100

            label = f"{delta:+.0%}" if delta != 0 else "Base"
            print(f"  {label:>6s}: w={w}  ES99=EUR {es99:,.0f}  overrun={overrun:+.1f}%")

            rows.append({
                "Phase": name,
                "Dominant_material": dom_name,
                "Perturbation": f"{delta:+.0%}" if delta != 0 else "Base",
                "Dom_weight_%": round(w[dom_idx] * 100, 1),
                "Weights": f"[{w[0]:.2f},{w[1]:.2f},{w[2]:.2f},{w[3]:.2f}]",
                "Budget_EUR": round(budget),
                "ES99_EUR": round(es99),
                "Overrun_%": round(overrun, 1),
            })

    res_df = pd.DataFrame(rows)
    res_df.to_csv(os.path.join(OUT_TAB, "c4c_phase_sensitivity.csv"), index=False)
    print(f"\nSaved: c4c_phase_sensitivity.csv")

    # Sensitivity range
    for name in ["Foundation", "Superstructure", "Completion"]:
        sub = res_df[res_df["Phase"] == name]
        es_range = sub["ES99_EUR"].max() - sub["ES99_EUR"].min()
        base_es = sub[sub["Perturbation"] == "Base"]["ES99_EUR"].values[0]
        print(f"  {name}: ES99 range = EUR {es_range:,.0f} "
              f"({es_range/base_es*100:.1f}% of base ES)")


if __name__ == "__main__":
    main()
