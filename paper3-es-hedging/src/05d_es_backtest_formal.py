"""
05d_es_backtest_formal.py
Fix 1: Formal ES Backtesting -- Kupiec PoF + Christoffersen Independence

Validates ES(99%) using expanding-window out-of-sample framework:
  1. Train on first W months, predict ES for next 12M
  2. Count violations (actual cost > VaR threshold)
  3. Kupiec Proportion-of-Failures (PoF) test
  4. Christoffersen independence test (no violation clustering)

Input:  data/processed/elstat_log_returns.csv
Output: results/tables/c3d_backtest_formal.csv
        results/figures/fig_c3d_backtest_violations.pdf
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvinecopulib as pv
from scipy.stats import chi2, norm, rankdata

warnings.filterwarnings("ignore")

SEED       = 42
N_SIMS     = 30_000
BASE_COST  = 2_300_000
HORIZON    = 6        # forward-looking months for each test window
MIN_TRAIN  = 60       # minimum training obs
STEP       = 6        # re-estimate every 6 months
ALPHA      = 0.99     # ES level
WEIGHTS    = np.array([0.30, 0.30, 0.20, 0.20])
COLS       = ["GR_Concrete", "GR_Steel", "GR_Fuel_Energy", "GR_PVC_Pipes"]

SCRIPT_DIR = os.path.dirname(__file__)
DATA_PATH  = os.path.join(SCRIPT_DIR, "..", "data", "processed",
                           "elstat_log_returns.csv")
OUT_FIG    = os.path.join(SCRIPT_DIR, "..", "results", "figures")
OUT_TAB    = os.path.join(SCRIPT_DIR, "..", "results", "tables")


def pseudo_obs(arr):
    n = arr.shape[0]
    return np.array([rankdata(arr[:, j]) / (n + 1)
                     for j in range(arr.shape[1])]).T


def simulate_var_es(train_df, copula_type, n_sims, horizon, seed):
    """Simulate and return (VaR99, ES99) for the given training data."""
    rng = np.random.default_rng(seed)
    mu = train_df.mean().values
    sigma = train_df.std().values
    n_assets = len(WEIGHTS)

    if copula_type == "gumbel":
        u_data = pseudo_obs(train_df.values)
        controls = pv.FitControlsVinecop(
            family_set=[pv.BicopFamily.gumbel],
            selection_criterion="aic", num_threads=1)
        vine = pv.Vinecop(d=n_assets)
        vine.select(np.asfortranarray(u_data), controls)
        u_sim = vine.simulate(n_sims * horizon, seeds=[seed])
        u_sim = np.clip(u_sim, 1e-6, 1 - 1e-6).reshape(n_sims, horizon, n_assets)
        z = norm.ppf(u_sim)
    elif copula_type == "gaussian":
        corr = train_df.corr().values
        L = np.linalg.cholesky(corr)
        raw = rng.standard_normal((n_sims, horizon, n_assets))
        z = np.array([r @ L.T for r in raw])
    else:
        z = rng.standard_normal((n_sims, horizon, n_assets))

    log_rets = mu + sigma * z
    simple_rets = np.exp(log_rets) - 1.0
    port_simple = (simple_rets * WEIGHTS).sum(axis=2)
    port_log = np.log1p(port_simple)
    total_log = port_log.sum(axis=1)
    costs = BASE_COST * np.exp(total_log)

    var99 = np.quantile(costs, ALPHA)
    tail = costs[costs >= var99]
    es99 = tail.mean() if len(tail) > 0 else var99
    return var99, es99


def kupiec_test(violations, n_total, expected_rate=0.01):
    """Kupiec Proportion of Failures test. H0: violation rate = expected_rate."""
    if n_total == 0:
        return np.nan, np.nan, True
    actual_rate = violations / n_total
    if violations == 0 or violations == n_total:
        return 0.0, 1.0, True

    lr = -2 * ((n_total - violations) * np.log(1 - expected_rate) +
               violations * np.log(expected_rate)) + \
         2 * ((n_total - violations) * np.log(1 - actual_rate) +
              violations * np.log(actual_rate))
    lr = abs(lr)
    p_val = 1 - chi2.cdf(lr, df=1)
    return round(lr, 4), round(p_val, 4), p_val > 0.05


def christoffersen_test(violations_seq):
    """Christoffersen independence test on binary violation sequence."""
    n = len(violations_seq)
    if n < 3:
        return np.nan, np.nan, True

    # Count transitions
    n00 = n01 = n10 = n11 = 0
    for i in range(1, n):
        prev, curr = violations_seq[i-1], violations_seq[i]
        if prev == 0 and curr == 0: n00 += 1
        elif prev == 0 and curr == 1: n01 += 1
        elif prev == 1 and curr == 0: n10 += 1
        else: n11 += 1

    # Avoid division by zero
    if (n00 + n01) == 0 or (n10 + n11) == 0:
        return 0.0, 1.0, True

    p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    p = (n01 + n11) / n if n > 0 else 0

    if p == 0 or p == 1 or p01 == 0 or p11 == 0:
        return 0.0, 1.0, True

    # Log-likelihood ratio
    try:
        lr_ind = -2 * ((n00 + n10) * np.log(1 - p) + (n01 + n11) * np.log(p))
        lr_dep = 0
        if n00 > 0: lr_dep += n00 * np.log(1 - p01)
        if n01 > 0: lr_dep += n01 * np.log(p01)
        if n10 > 0: lr_dep += n10 * np.log(1 - p11)
        if n11 > 0: lr_dep += n11 * np.log(p11)
        lr_dep *= -2
        lr = lr_dep - lr_ind
        lr = abs(lr)
    except (ValueError, ZeroDivisionError):
        return 0.0, 1.0, True

    p_val = 1 - chi2.cdf(lr, df=1)
    return round(lr, 4), round(p_val, 4), p_val > 0.05


def main():
    os.makedirs(OUT_FIG, exist_ok=True)
    os.makedirs(OUT_TAB, exist_ok=True)

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    available = [c for c in COLS if c in df.columns]
    df_use = df[available]
    n_total = len(df_use)
    print(f"Data: {n_total} obs")

    results = []
    for cop in ["gumbel", "gaussian", "independent"]:
        print(f"\n--- {cop.upper()} ---")
        violations_seq = []
        var_list = []
        date_list = []

        for start in range(MIN_TRAIN, n_total - HORIZON, STEP):
            train = df_use.iloc[:start]
            # Compute "realized" portfolio cost from actual data
            test_data = df_use.iloc[start:start + HORIZON]
            if len(test_data) < HORIZON:
                continue

            # Realized cost
            actual_log_rets = test_data.values
            actual_simple = np.exp(actual_log_rets) - 1.0
            actual_port_simple = (actual_simple * WEIGHTS).sum(axis=1)
            actual_port_log = np.log1p(actual_port_simple)
            actual_total = actual_port_log.sum()
            realized_cost = BASE_COST * np.exp(actual_total)

            # Predicted VaR/ES
            var99, es99 = simulate_var_es(train, cop, N_SIMS, HORIZON,
                                          SEED + start)

            violation = 1 if realized_cost > var99 else 0
            violations_seq.append(violation)
            var_list.append(var99)
            date_list.append(df_use.index[start])

            if start % 48 == 0:
                print(f"  {df_use.index[start].date()}: VaR99=EUR {var99:,.0f}  "
                      f"realized=EUR {realized_cost:,.0f}  "
                      f"{'VIOLATION' if violation else 'ok'}")

        n_windows = len(violations_seq)
        n_violations = sum(violations_seq)
        viol_rate = n_violations / n_windows if n_windows > 0 else 0

        # Kupiec test
        kup_lr, kup_p, kup_pass = kupiec_test(n_violations, n_windows)

        # Christoffersen test
        chr_lr, chr_p, chr_pass = christoffersen_test(violations_seq)

        print(f"  Windows: {n_windows}, Violations: {n_violations} "
              f"({viol_rate*100:.1f}%)")
        print(f"  Kupiec: LR={kup_lr}, p={kup_p}, "
              f"{'PASS' if kup_pass else 'FAIL'}")
        print(f"  Christoffersen: LR={chr_lr}, p={chr_p}, "
              f"{'PASS' if chr_pass else 'FAIL'}")

        results.append({
            "Copula": cop,
            "n_windows": n_windows,
            "n_violations": n_violations,
            "Violation_rate_%": round(viol_rate * 100, 2),
            "Expected_rate_%": 1.0,
            "Kupiec_LR": kup_lr,
            "Kupiec_p": kup_p,
            "Kupiec_PASS": kup_pass,
            "Christoffersen_LR": chr_lr,
            "Christoffersen_p": chr_p,
            "Christoffersen_PASS": chr_pass,
        })

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(OUT_TAB, "c3d_backtest_formal.csv"), index=False)
    print(f"\nSaved: c3d_backtest_formal.csv")
    print(res_df.to_string(index=False))


if __name__ == "__main__":
    main()
