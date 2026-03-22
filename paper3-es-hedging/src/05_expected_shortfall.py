"""
05_expected_shortfall.py
C3 -- Basel III Expected Shortfall ES(99%)

Upgrades Paper 1 percentile-based risk (P85) to coherent risk measure ES(99%).
Runs 100k Monte Carlo simulations under three copula types x three regimes.

Input:  data/processed/elstat_log_returns.csv
Output: results/tables/c3_es_comparison.csv
"""

import pandas as pd
import numpy as np
from scipy.stats import norm, rankdata
import pyvinecopulib as pv
import warnings
import os

warnings.filterwarnings("ignore")

SEED       = 42
N_SIMS     = 100_000
BASE_COST  = 2_300_000
HORIZON    = 24
WEIGHTS    = np.array([0.30, 0.30, 0.20, 0.20])   # Concrete, Steel, Fuel, PVC

# Regime dates
REGIMES = {
    "Full (2000-2024)":    ("2000-02-01", "2024-12-01"),
    "Stable (2014-2019)":  ("2014-01-01", "2019-12-01"),
    "Crisis (2021-2024)":  ("2021-01-01", "2024-12-01"),
}

SCRIPT_DIR = os.path.dirname(__file__)
DATA_PATH  = os.path.join(SCRIPT_DIR, "..", "data", "processed",
                           "elstat_log_returns.csv")
OUT_TAB    = os.path.join(SCRIPT_DIR, "..", "results", "tables")

# Greek material columns (in weight order)
COLS = ["GR_Concrete", "GR_Steel", "GR_Fuel_Energy", "GR_PVC_Pipes"]


def pseudo_obs(arr):
    n = arr.shape[0]
    return np.array([rankdata(arr[:, j]) / (n + 1)
                     for j in range(arr.shape[1])]).T


def simulate_costs(log_returns_df, copula_type, n_sims, horizon, weights,
                   base_cost, seed):
    """Run Monte Carlo simulation with specified copula type."""
    rng = np.random.default_rng(seed)
    mu    = log_returns_df.mean().values
    sigma = log_returns_df.std().values
    n_assets = len(weights)

    if copula_type == "Independent":
        z = rng.standard_normal((n_sims, horizon, n_assets))

    elif copula_type == "Gaussian":
        corr = log_returns_df.corr().values
        L = np.linalg.cholesky(corr)
        raw = rng.standard_normal((n_sims, horizon, n_assets))
        z = np.array([r @ L.T for r in raw])

    elif copula_type == "Gumbel":
        u_data = pseudo_obs(log_returns_df.values)
        controls = pv.FitControlsVinecop(
            family_set=[pv.BicopFamily.gumbel],
            selection_criterion="aic",
            num_threads=1
        )
        vine = pv.Vinecop(d=n_assets)
        vine.select(np.asfortranarray(u_data), controls)
        u_sim = vine.simulate(n_sims * horizon, seeds=[seed])
        u_sim = np.clip(u_sim, 1e-6, 1 - 1e-6)
        u_sim = u_sim.reshape(n_sims, horizon, n_assets)
        z = norm.ppf(u_sim)
    else:
        raise ValueError(f"Unknown copula: {copula_type}")

    log_returns = mu + sigma * z
    # Convert log-returns to simple returns before cross-asset aggregation
    simple_returns = np.exp(log_returns) - 1.0
    portfolio_simple = (simple_returns * weights).sum(axis=2)
    # Convert back to log-returns for temporal aggregation
    portfolio_log = np.log1p(portfolio_simple)
    total_log_return = portfolio_log.sum(axis=1)
    return base_cost * np.exp(total_log_return)


def es(costs, alpha):
    """Expected Shortfall at level alpha."""
    threshold = np.quantile(costs, alpha)
    tail = costs[costs >= threshold]
    return tail.mean() if len(tail) > 0 else threshold


def main():
    os.makedirs(OUT_TAB, exist_ok=True)

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    available = [c for c in COLS if c in df.columns]
    if len(available) < 4:
        print(f"WARNING: Only {len(available)} columns found: {available}")
    df_use = df[available]

    results = []
    for regime_name, (start, end) in REGIMES.items():
        subset = df_use.loc[start:end].dropna()
        n_obs = len(subset)
        if n_obs < 12:
            print(f"SKIP {regime_name}: only {n_obs} obs")
            continue

        print(f"\n{'='*60}")
        print(f"Regime: {regime_name} (n={n_obs})")
        print(f"{'='*60}")

        for cop in ["Independent", "Gaussian", "Gumbel"]:
            costs = simulate_costs(subset, cop, N_SIMS, HORIZON,
                                   WEIGHTS, BASE_COST, SEED)

            p85  = np.quantile(costs, 0.85)
            p95  = np.quantile(costs, 0.95)
            p99  = np.quantile(costs, 0.99)
            es95 = es(costs, 0.95)
            es99 = es(costs, 0.99)

            results.append({
                "Regime":     regime_name,
                "Copula":     cop,
                "n_obs":      n_obs,
                "P85_EUR":    round(p85),
                "P95_EUR":    round(p95),
                "P99_EUR":    round(p99),
                "ES95_EUR":   round(es95),
                "ES99_EUR":   round(es99),
                "ES99_vs_P85": round(es99 - p85),
            })
            print(f"  {cop:12s}  P85={p85:>11,.0f}  P99={p99:>11,.0f}  "
                  f"ES99={es99:>11,.0f}  gap={es99-p85:>9,.0f}")

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(OUT_TAB, "c3_es_comparison.csv"), index=False)
    print(f"\nSaved: c3_es_comparison.csv")
    print(res_df.to_string(index=False))


if __name__ == "__main__":
    main()
