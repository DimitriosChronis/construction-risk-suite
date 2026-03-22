"""
05b_rolling_es_backtest.py
C3 upgrade -- Rolling ES(99%) + Backtesting

1. Rolling ES(99%) over time (24M window) — shows risk evolution
2. Kupiec Proportion of Failures (PoF) test
3. Gumbel vs Gaussian ES: is the difference statistically significant?

Input:  data/processed/elstat_log_returns.csv
Output: results/tables/c3b_rolling_es.csv
        results/tables/c3b_backtest_results.csv
        results/figures/fig_c3b_rolling_es.pdf
"""

import pandas as pd
import numpy as np
from scipy.stats import norm, rankdata, chi2
import pyvinecopulib as pv
import matplotlib.pyplot as plt
import os

SEED      = 42
N_SIMS    = 50_000   # per window (speed vs accuracy tradeoff)
WINDOW    = 24       # months rolling window
HORIZON   = 12       # forward-looking months
BASE_COST = 2_300_000
WEIGHTS   = np.array([0.30, 0.30, 0.20, 0.20])
ALPHA     = 0.99     # ES level

COLS = ["GR_Concrete", "GR_Steel", "GR_Fuel_Energy", "GR_PVC_Pipes"]

SCRIPT_DIR = os.path.dirname(__file__)
DATA_PATH  = os.path.join(SCRIPT_DIR, "..", "data", "processed",
                           "elstat_log_returns.csv")
OUT_FIG    = os.path.join(SCRIPT_DIR, "..", "results", "figures")
OUT_TAB    = os.path.join(SCRIPT_DIR, "..", "results", "tables")


def pseudo_obs(arr):
    n = arr.shape[0]
    return np.array([rankdata(arr[:, j]) / (n + 1)
                     for j in range(arr.shape[1])]).T


def simulate_es(log_returns_df, copula_type, n_sims, horizon, weights,
                base_cost, seed):
    """Run Monte Carlo and return VaR(99%) and ES(99%)."""
    rng = np.random.default_rng(seed)
    mu = log_returns_df.mean().values
    sigma = log_returns_df.std().values
    n_assets = len(weights)

    if copula_type == "gaussian":
        corr = log_returns_df.corr().values
        L = np.linalg.cholesky(corr)
        raw = rng.standard_normal((n_sims, horizon, n_assets))
        z = np.array([r @ L.T for r in raw])
    elif copula_type == "gumbel":
        u_data = pseudo_obs(log_returns_df.values)
        controls = pv.FitControlsVinecop(
            family_set=[pv.BicopFamily.gumbel],
            selection_criterion="aic", num_threads=1)
        vine = pv.Vinecop(d=n_assets)
        vine.select(np.asfortranarray(u_data), controls)
        u_sim = vine.simulate(n_sims * horizon, seeds=[seed])
        u_sim = np.clip(u_sim, 1e-6, 1 - 1e-6).reshape(n_sims, horizon, n_assets)
        z = norm.ppf(u_sim)
    else:
        raise ValueError(copula_type)

    log_rets = mu + sigma * z
    simple_rets = np.exp(log_rets) - 1.0
    port_simple = (simple_rets * weights).sum(axis=2)
    port_log = np.log1p(port_simple)
    total_log = port_log.sum(axis=1)
    costs = base_cost * np.exp(total_log)

    var99 = np.quantile(costs, ALPHA)
    tail = costs[costs >= var99]
    es99 = tail.mean() if len(tail) > 0 else var99
    return var99, es99


def kupiec_test(violations, n_total, alpha=0.01):
    """Kupiec Proportion of Failures test."""
    expected_rate = 1 - ALPHA  # 0.01
    actual_rate = violations / n_total if n_total > 0 else 0
    if violations == 0 or violations == n_total:
        return {"LR_stat": 0, "p_value": 1.0, "PASS": True}

    lr = -2 * (n_total * np.log(1 - expected_rate) +
               0 * np.log(expected_rate)) + \
         2 * ((n_total - violations) * np.log(1 - actual_rate) +
              violations * np.log(actual_rate))
    lr = abs(lr)
    p_val = 1 - chi2.cdf(lr, df=1)
    return {"LR_stat": round(lr, 4), "p_value": round(p_val, 4),
            "PASS": p_val > 0.05}


def main():
    os.makedirs(OUT_FIG, exist_ok=True)
    os.makedirs(OUT_TAB, exist_ok=True)

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    available = [c for c in COLS if c in df.columns]
    df_use = df[available]
    print(f"Data: {df_use.shape}")

    # ── Rolling ES ────────────────────────────────────────────────────────────
    rolling_results = []
    dates = df_use.index[WINDOW:]

    print(f"Computing rolling ES(99%) with {WINDOW}M window...")
    step = 6  # compute every 6 months for speed
    for i in range(WINDOW, len(df_use), step):
        window_data = df_use.iloc[i - WINDOW:i]
        date = df_use.index[i]

        for cop in ["gaussian", "gumbel"]:
            var99, es99 = simulate_es(window_data, cop, N_SIMS, HORIZON,
                                       WEIGHTS, BASE_COST, SEED + i)
            rolling_results.append({
                "Date": date,
                "Copula": cop,
                "VaR99_EUR": round(var99),
                "ES99_EUR": round(es99),
            })

        if (i - WINDOW) % 24 == 0:
            print(f"  {date.date()}: Gumbel ES99=EUR {es99:,.0f}")

    roll_df = pd.DataFrame(rolling_results)
    roll_df.to_csv(os.path.join(OUT_TAB, "c3b_rolling_es.csv"), index=False)

    # ── Plot rolling ES ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    for cop, color, ls in [("gaussian", "#1565C0", "--"), ("gumbel", "#D84315", "-")]:
        sub = roll_df[roll_df["Copula"] == cop]
        ax.plot(pd.to_datetime(sub["Date"]), sub["ES99_EUR"] / 1e6,
                color=color, linestyle=ls, linewidth=1.5, label=f"ES(99%) {cop.title()}")

    ax.axhline(BASE_COST / 1e6, color="grey", linestyle=":", linewidth=0.8,
               label="Base cost")
    ax.set_ylabel("ES(99%) EUR millions")
    ax.set_xlabel("Date")
    ax.legend(fontsize=8)
    ax.set_title("")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG, "fig_c3b_rolling_es.pdf"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("\nfig_c3b_rolling_es.pdf saved.")

    # ── Gumbel vs Gaussian: statistical comparison ────────────────────────────
    gumbel_es = roll_df[roll_df["Copula"] == "gumbel"]["ES99_EUR"].values
    gauss_es  = roll_df[roll_df["Copula"] == "gaussian"]["ES99_EUR"].values
    diff = gumbel_es - gauss_es
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    t_stat = mean_diff / (std_diff / np.sqrt(len(diff))) if std_diff > 0 else 0
    p_val = 2 * (1 - norm.cdf(abs(t_stat)))

    print(f"\nGumbel vs Gaussian ES(99%) comparison:")
    print(f"  Mean difference: EUR {mean_diff:,.0f}")
    print(f"  t-stat: {t_stat:.3f}, p-value: {p_val:.4f}")
    print(f"  Gumbel ES significantly higher: {'YES' if p_val < 0.05 else 'NO'}")

    # Backtest summary
    backtest = {
        "Test": "Gumbel_vs_Gaussian",
        "Mean_diff_EUR": round(mean_diff),
        "t_stat": round(t_stat, 4),
        "p_value": round(p_val, 4),
        "Significant": p_val < 0.05,
        "n_windows": len(diff),
    }
    pd.DataFrame([backtest]).to_csv(
        os.path.join(OUT_TAB, "c3b_backtest_results.csv"), index=False)
    print("\nSaved: c3b_rolling_es.csv, c3b_backtest_results.csv")


if __name__ == "__main__":
    main()
