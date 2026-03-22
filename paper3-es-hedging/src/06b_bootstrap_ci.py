"""
06b_bootstrap_ci.py
C4 upgrade -- Bootstrap Confidence Intervals for Lifecycle ES

Resamples crisis data 500 times to produce 95% CI on phase-specific ES(99%).

Input:  data/processed/elstat_log_returns.csv
Output: results/tables/c4b_bootstrap_ci.csv
"""

import pandas as pd
import numpy as np
from scipy.stats import norm, rankdata
import pyvinecopulib as pv
import os

SEED       = 42
N_SIMS     = 20_000    # per bootstrap iteration (reduced for speed)
N_BOOT     = 500       # bootstrap iterations
BASE_COST  = 2_300_000

PHASES = [
    ("Foundation",     8,  np.array([0.50, 0.20, 0.20, 0.10]), 0.30),
    ("Superstructure", 10, np.array([0.20, 0.50, 0.20, 0.10]), 0.50),
    ("Completion",     6,  np.array([0.15, 0.25, 0.35, 0.25]), 0.20),
]

COLS = ["GR_Concrete", "GR_Steel", "GR_Fuel_Energy", "GR_PVC_Pipes"]
CRISIS_START = "2021-01-01"
CRISIS_END   = "2024-12-01"

SCRIPT_DIR = os.path.dirname(__file__)
DATA_PATH  = os.path.join(SCRIPT_DIR, "..", "data", "processed",
                           "elstat_log_returns.csv")
OUT_TAB    = os.path.join(SCRIPT_DIR, "..", "results", "tables")


def pseudo_obs(arr):
    n = arr.shape[0]
    return np.array([rankdata(arr[:, j]) / (n + 1)
                     for j in range(arr.shape[1])]).T


def simulate_phase_es(log_returns_df, weights, n_months, budget, n_sims, seed):
    """Quick Gumbel vine simulation for one phase, return ES(99%)."""
    rng = np.random.default_rng(seed)
    mu = log_returns_df.mean().values
    sigma = log_returns_df.std().values
    n_assets = len(weights)

    u_data = pseudo_obs(log_returns_df.values)
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


def main():
    os.makedirs(OUT_TAB, exist_ok=True)

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    available = [c for c in COLS if c in df.columns]
    crisis = df[available].loc[CRISIS_START:CRISIS_END].dropna()
    n_crisis = len(crisis)
    print(f"Crisis data: {n_crisis} obs")
    print(f"Bootstrap iterations: {N_BOOT}")

    results = []
    for name, months, weights, budget_share in PHASES:
        budget = BASE_COST * budget_share
        print(f"\n{name} ({months}M, budget=EUR {budget:,.0f}):")

        boot_es = []
        for b in range(N_BOOT):
            # Resample crisis data with replacement
            sample = crisis.sample(n=n_crisis, replace=True,
                                   random_state=SEED + b)
            sample = sample.reset_index(drop=True)
            es99 = simulate_phase_es(sample, weights, months, budget,
                                     N_SIMS, SEED + b)
            boot_es.append(es99)

            if (b + 1) % 100 == 0:
                print(f"  Bootstrap {b+1}/{N_BOOT}...")

        boot_es = np.array(boot_es)
        point = np.mean(boot_es)
        ci_low = np.percentile(boot_es, 2.5)
        ci_high = np.percentile(boot_es, 97.5)
        ci_width = ci_high - ci_low

        results.append({
            "Phase": name,
            "ES99_point_EUR": round(point),
            "ES99_CI_lower": round(ci_low),
            "ES99_CI_upper": round(ci_high),
            "CI_width_EUR": round(ci_width),
            "CI_width_%": round(ci_width / budget * 100, 1),
            "n_bootstrap": N_BOOT,
        })
        print(f"  ES99 = EUR {point:,.0f} [{ci_low:,.0f} ; {ci_high:,.0f}]")

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(OUT_TAB, "c4b_bootstrap_ci.csv"), index=False)
    print(f"\nSaved: c4b_bootstrap_ci.csv")
    print(res_df.to_string(index=False))


if __name__ == "__main__":
    main()
