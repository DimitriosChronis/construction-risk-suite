"""
05f_es_decomposition.py
Fix 7: ES Marginal Contribution Decomposition

Decomposes ES(99%) into per-material marginal contributions:
  MC_i = E[w_i * r_i | portfolio in tail]

Shows WHICH material drives tail risk (expected: Steel and Fuel).

Input:  data/processed/elstat_log_returns.csv
Output: results/tables/c3f_es_decomposition.csv
        results/figures/fig_c3f_es_decomposition.pdf
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvinecopulib as pv
from scipy.stats import norm, rankdata

warnings.filterwarnings("ignore")

SEED       = 42
N_SIMS     = 100_000
BASE_COST  = 2_300_000
HORIZON    = 24
WEIGHTS    = np.array([0.30, 0.30, 0.20, 0.20])
COLS       = ["GR_Concrete", "GR_Steel", "GR_Fuel_Energy", "GR_PVC_Pipes"]
MAT_NAMES  = ["Concrete", "Steel", "Fuel/Energy", "PVC/Pipes"]

CRISIS_START = "2021-01-01"
CRISIS_END   = "2024-12-01"

SCRIPT_DIR = os.path.dirname(__file__)
DATA_PATH  = os.path.join(SCRIPT_DIR, "..", "data", "processed",
                           "elstat_log_returns.csv")
OUT_FIG    = os.path.join(SCRIPT_DIR, "..", "results", "figures")
OUT_TAB    = os.path.join(SCRIPT_DIR, "..", "results", "tables")

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.linewidth": 0.8, "axes.grid": True,
    "grid.alpha": 0.3, "grid.linewidth": 0.5,
    "figure.dpi": 300, "savefig.dpi": 300,
})


def pseudo_obs(arr):
    n = arr.shape[0]
    return np.array([rankdata(arr[:, j]) / (n + 1)
                     for j in range(arr.shape[1])]).T


def main():
    os.makedirs(OUT_FIG, exist_ok=True)
    os.makedirs(OUT_TAB, exist_ok=True)

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    available = [c for c in COLS if c in df.columns]

    rows = []
    regimes = {
        "Full (2000-2024)": df[available],
        "Crisis (2021-2024)": df[available].loc[CRISIS_START:CRISIS_END],
    }

    for regime_name, data in regimes.items():
        data = data.dropna()
        n_obs = len(data)
        mu = data.mean().values
        sigma = data.std().values
        n_assets = len(available)

        print(f"\n{'='*60}")
        print(f"Regime: {regime_name} (n={n_obs})")

        # Fit Gumbel vine
        u_data = pseudo_obs(data.values)
        controls = pv.FitControlsVinecop(
            family_set=[pv.BicopFamily.gumbel],
            selection_criterion="aic", num_threads=1)
        vine = pv.Vinecop(d=n_assets)
        vine.select(np.asfortranarray(u_data), controls)

        u_sim = vine.simulate(N_SIMS * HORIZON, seeds=[SEED])
        u_sim = np.clip(u_sim, 1e-6, 1 - 1e-6).reshape(N_SIMS, HORIZON, n_assets)
        z = norm.ppf(u_sim)

        # Per-asset log-returns over full horizon
        log_rets = mu + sigma * z  # (N_SIMS, HORIZON, n_assets)
        simple_rets = np.exp(log_rets) - 1.0

        # Portfolio returns per month
        port_simple = (simple_rets * WEIGHTS).sum(axis=2)
        port_log = np.log1p(port_simple)
        total_log = port_log.sum(axis=1)
        total_costs = BASE_COST * np.exp(total_log)

        # Identify tail (top 1%)
        var99 = np.quantile(total_costs, 0.99)
        tail_mask = total_costs >= var99
        n_tail = tail_mask.sum()

        # Per-asset cumulative simple return in tail scenarios
        # Weighted contribution = w_i * cumulative_simple_return_i
        for i in range(n_assets):
            # Cumulative log-return per asset over horizon
            asset_log = log_rets[:, :, i].sum(axis=1)  # (N_SIMS,)
            asset_simple = np.exp(asset_log) - 1.0     # cumulative simple return

            # Marginal contribution = w_i * E[asset_simple | tail]
            mc_return = WEIGHTS[i] * asset_simple[tail_mask].mean()
            mc_eur = BASE_COST * mc_return

            # Also compute unconditional
            unc_return = WEIGHTS[i] * asset_simple.mean()
            unc_eur = BASE_COST * unc_return

            # Tail amplification ratio
            amp = mc_return / unc_return if abs(unc_return) > 1e-10 else np.nan

            rows.append({
                "Regime": regime_name,
                "Material": MAT_NAMES[i],
                "Weight_%": round(WEIGHTS[i] * 100),
                "MC_tail_return_%": round(mc_return * 100, 2),
                "MC_tail_EUR": round(mc_eur),
                "MC_unconditional_%": round(unc_return * 100, 2),
                "MC_unconditional_EUR": round(unc_eur),
                "Tail_amplification": round(amp, 2) if not np.isnan(amp) else "NA",
            })

            print(f"  {MAT_NAMES[i]:12s}: MC_tail={mc_return*100:+.2f}%  "
                  f"(EUR {mc_eur:+,.0f})  "
                  f"unconditional={unc_return*100:+.2f}%  "
                  f"amplification={amp:.2f}x")

        # Total check
        total_mc = sum(r["MC_tail_EUR"] for r in rows[-n_assets:])
        es99 = total_costs[tail_mask].mean()
        print(f"  SUM MC = EUR {total_mc:,.0f}  vs  "
              f"ES99-base = EUR {es99 - BASE_COST:,.0f}")

    res_df = pd.DataFrame(rows)
    res_df.to_csv(os.path.join(OUT_TAB, "c3f_es_decomposition.csv"), index=False)

    # ── Figure: stacked bar chart of marginal contributions ──────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    colors = ["#1565C0", "#D84315", "#2E7D32", "#7B1FA2"]

    for ax_idx, regime_name in enumerate(regimes.keys()):
        ax = axes[ax_idx]
        sub = res_df[res_df["Regime"] == regime_name]

        mats = sub["Material"].values
        mc_vals = sub["MC_tail_EUR"].values / 1e3  # in thousands

        bars = ax.barh(range(len(mats)), mc_vals, color=colors,
                        edgecolor="black", linewidth=0.5)

        for bar, val in zip(bars, mc_vals):
            x_pos = bar.get_width() + 0.5
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                    f"EUR {val:.0f}K", va="center", fontsize=8)

        ax.set_yticks(range(len(mats)))
        ax.set_yticklabels(mats)
        short = regime_name.split("(")[0].strip()
        ax.set_xlabel(f"Marginal Contribution (EUR thousands)\n{short}")
        ax.axvline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG, "fig_c3f_es_decomposition.pdf"),
                bbox_inches="tight")
    plt.close()
    print("\nfig_c3f_es_decomposition.pdf saved.")
    print(f"Saved: c3f_es_decomposition.csv")


if __name__ == "__main__":
    main()
