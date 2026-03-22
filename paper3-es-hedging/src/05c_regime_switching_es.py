"""
05c_regime_switching_es.py
Regime-Switching Expected Shortfall (C3 Enhancement)

Identifies two market regimes via rolling composite volatility:
  Stable regime : rolling vol <= 33rd percentile
  Crisis regime : rolling vol >= 67th percentile

For each regime, fits a 4D vine copula (Gumbel / Gaussian separately)
on the regime-specific pseudo-observations, then simulates 50,000
portfolio paths and computes ES(99%).

Key finding (expected):
  Stable : Gumbel premium over Gaussian  ~  small (< EUR 15K)
  Crisis : Gumbel premium over Gaussian  >>  large (> EUR 200K)
  => tail-risk premium is REGIME-CONDITIONAL, not unconditional

This directly addresses the reviewer question raised by 05b
(non-significant overall t-test): the Gumbel advantage is episodic.

Outputs:
  results/tables/c3c_regime_es.csv
  results/figures/fig_c3c_regime_es.pdf
  results/figures/fig_c3c_regime_vol.pdf
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvinecopulib as pv
from scipy.stats import kendalltau, norm

warnings.filterwarnings("ignore")
np.random.seed(42)

SCRIPT_DIR = os.path.dirname(__file__)
DATA_PATH  = os.path.join(SCRIPT_DIR, "..", "data", "processed",
                           "aligned_log_returns.csv")
OUT_FIG    = os.path.join(SCRIPT_DIR, "..", "results", "figures")
OUT_TAB    = os.path.join(SCRIPT_DIR, "..", "results", "tables")

GR_COLS    = ["GR_Concrete", "GR_Steel", "GR_Fuel_Energy", "GR_PVC_Pipes"]
WEIGHTS    = np.array([0.30, 0.30, 0.20, 0.20])
BASE_COST  = 2_300_000
MONTHS     = 24
N_SIM      = 50_000
VOL_CAP    = 0.15   # annualised sigma cap (from 08_volatility_cap)

# Named regimes — consistent with Paper 1 GoF regime classification
# Stable: post-crisis calm before commodity super-cycle (n=72 in Paper 1)
# Crisis: post-COVID inflation shock (n=48 in Paper 1)
STABLE_START = "2014-01-01"
STABLE_END   = "2019-12-01"
CRISIS_START = "2021-01-01"
CRISIS_END   = "2024-12-01"


# ── Helpers ───────────────────────────────────────────────────────────────────

def pseudo_obs_manual(data: np.ndarray) -> np.ndarray:
    """Rank-based pseudo-observations for copula fitting."""
    n, d = data.shape
    u = np.empty_like(data, dtype=np.float64)
    for j in range(d):
        ranks = data[:, j].argsort().argsort() + 1
        u[:, j] = ranks / (n + 1.0)
    return np.asfortranarray(u)


def avg_kendall_tau(returns: np.ndarray) -> float:
    """Average pairwise Kendall tau."""
    n_vars = returns.shape[1]
    taus = []
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            tau, _ = kendalltau(returns[:, i], returns[:, j])
            taus.append(tau)
    return float(np.mean(taus))


def fit_vine_simulate(u: np.ndarray, family: str, n_sim: int) -> np.ndarray:
    """
    Fit a d-dimensional vine copula restricted to 'family',
    then simulate n_sim samples. Returns (n_sim, d) array in [0,1].
    """
    d = u.shape[1]
    if family == "gumbel":
        fset = [pv.BicopFamily.gumbel]
    else:
        fset = [pv.BicopFamily.gaussian]

    controls = pv.FitControlsVinecop(
        family_set=fset,
        selection_criterion="aic"
    )
    vine = pv.Vinecop(d)
    vine.select(np.asfortranarray(u), controls)
    try:
        sim = vine.simulate(n_sim, seeds=[42])
    except TypeError:
        sim = vine.simulate(n_sim)
    return sim


def compute_es(sim_u: np.ndarray, sigma: np.ndarray) -> dict:
    """
    Transform simulated copula samples to portfolio costs.
    sigma : per-asset monthly std dev (annualised inside)
    """
    z = norm.ppf(np.clip(sim_u, 1e-6, 1 - 1e-6))
    horizon_sigma = np.minimum(sigma * np.sqrt(MONTHS), VOL_CAP)
    log_returns   = z * horizon_sigma  # (n_sim, d) single-period 24M log-returns
    # Convert to simple returns before cross-asset aggregation
    simple_returns = np.exp(log_returns) - 1.0
    port_simple    = simple_returns @ WEIGHTS
    total_costs    = BASE_COST * (1.0 + port_simple)
    p85  = float(np.percentile(total_costs, 85))
    es99 = float(np.mean(total_costs[total_costs >=
                                      np.percentile(total_costs, 99)]))
    return {"P85": p85, "ES99": es99}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_FIG, exist_ok=True)
    os.makedirs(OUT_TAB, exist_ok=True)

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    gr = df[GR_COLS].dropna()
    print(f"Data: {gr.shape[0]} obs  x  {gr.shape[1]} series")

    # ── Named regime slices ───────────────────────────────────────────────
    # Consistent with Paper 1 GoF regime dates.
    # Stable 2014-2019: calm period, low commodity volatility, low dependence
    # Crisis 2021-2024: post-COVID inflation shock, high joint price rises
    stable_ret = gr.loc[STABLE_START:STABLE_END].values
    crisis_ret = gr.loc[CRISIS_START:CRISIS_END].values

    print(f"\nNamed regime windows (consistent with Paper 1):")
    print(f"  Stable  ({STABLE_START} to {STABLE_END}): n={len(stable_ret)} months")
    print(f"  Crisis  ({CRISIS_START} to {CRISIS_END}): n={len(crisis_ret)} months")

    if len(stable_ret) < 20 or len(crisis_ret) < 20:
        print("  WARNING: too few obs in a regime — adjust percentile thresholds")
        return

    tau_stable = avg_kendall_tau(stable_ret)
    tau_crisis = avg_kendall_tau(crisis_ret)
    sig_stable = stable_ret.std(axis=0)
    sig_crisis = crisis_ret.std(axis=0)

    print(f"\n  Avg Kendall tau: stable={tau_stable:.3f}  "
          f"crisis={tau_crisis:.3f}")

    # ── Pseudo-observations per regime ────────────────────────────────────
    u_stable = pseudo_obs_manual(stable_ret)
    u_crisis = pseudo_obs_manual(crisis_ret)

    # ── Vine fitting + Monte Carlo ─────────────────────────────────────────
    combos = [
        ("stable", "gumbel",   u_stable, sig_stable),
        ("stable", "gaussian", u_stable, sig_stable),
        ("crisis", "gumbel",   u_crisis, sig_crisis),
        ("crisis", "gaussian", u_crisis, sig_crisis),
    ]

    results = {}
    for regime, copula, u, sig in combos:
        key = f"{regime}_{copula}"
        print(f"\nFitting {copula} vine on {regime} regime "
              f"(n={u.shape[0]}) ...")
        try:
            sim_u = fit_vine_simulate(u, copula, N_SIM)
            res   = compute_es(sim_u, sig)
            results[key] = res
            print(f"  P85 = EUR {res['P85']:,.0f}   "
                  f"ES99 = EUR {res['ES99']:,.0f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results[key] = {"P85": np.nan, "ES99": np.nan}

    # ── Gumbel premium per regime ──────────────────────────────────────────
    prem_stable = (results["stable_gumbel"]["ES99"] -
                   results["stable_gaussian"]["ES99"])
    prem_crisis = (results["crisis_gumbel"]["ES99"] -
                   results["crisis_gaussian"]["ES99"])

    print(f"\nGumbel premium over Gaussian (ES99):")
    print(f"  Stable regime: EUR {prem_stable:+,.0f}")
    print(f"  Crisis regime: EUR {prem_crisis:+,.0f}")
    print(f"\n  => Crisis/Stable ratio: "
          f"{abs(prem_crisis)/max(abs(prem_stable),1):.1f}x")

    # ── Save table ────────────────────────────────────────────────────────
    table_rows = []
    for regime, copula, u, sig in combos:
        key = f"{regime}_{copula}"
        r   = results[key]
        table_rows.append({
            "Regime":    regime,
            "Copula":    copula,
            "n_obs":     u.shape[0],
            "Avg_tau":   round(tau_stable if regime == "stable"
                               else tau_crisis, 3),
            "P85_EUR":   round(r["P85"],  0),
            "ES99_EUR":  round(r["ES99"], 0),
        })
    # Add premium rows
    table_rows.append({
        "Regime": "premium_stable", "Copula": "Gumbel-Gaussian",
        "n_obs": "", "Avg_tau": "",
        "P85_EUR": "", "ES99_EUR": round(prem_stable, 0)
    })
    table_rows.append({
        "Regime": "premium_crisis", "Copula": "Gumbel-Gaussian",
        "n_obs": "", "Avg_tau": "",
        "P85_EUR": "", "ES99_EUR": round(prem_crisis, 0)
    })

    pd.DataFrame(table_rows).to_csv(
        os.path.join(OUT_TAB, "c3c_regime_es.csv"), index=False)

    # ── Figure 1: ES bar chart ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    regime_labels = ["Stable", "Crisis"]
    g_es = [results["stable_gumbel"]["ES99"],
            results["crisis_gumbel"]["ES99"]]
    n_es = [results["stable_gaussian"]["ES99"],
            results["crisis_gaussian"]["ES99"]]

    x = np.arange(2)
    w = 0.35
    ax.bar(x - w / 2, [v / 1e6 for v in g_es], w,
           label="Gumbel", color="#1565C0", edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, [v / 1e6 for v in n_es], w,
           label="Gaussian", color="#78909C", edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(regime_labels)
    ax.set_ylabel("ES(99%)  EUR million")
    ax.legend(fontsize=8)

    for i, (gv, nv) in enumerate(zip(g_es, n_es)):
        prem = gv - nv
        ypos = max(gv, nv) / 1e6 + 0.008
        color = "#C62828" if abs(prem) > 50_000 else "#555555"
        ax.annotate(f"prem: EUR {prem:+,.0f}",
                    xy=(i, ypos), ha="center", fontsize=7, color=color)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG, "fig_c3c_regime_es.pdf"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("\nfig_c3c_regime_es.pdf saved.")

    # ── Figure 2: GR composite index with regime shading ──────────────────
    composite = gr.mean(axis=1)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(composite.index, composite.values,
            color="#1565C0", linewidth=1, label="Avg GR log-return")
    ax.axvspan(pd.Timestamp(STABLE_START), pd.Timestamp(STABLE_END),
               alpha=0.20, color="#388E3C", label="Stable regime (2014-2019)")
    ax.axvspan(pd.Timestamp(CRISIS_START), pd.Timestamp(CRISIS_END),
               alpha=0.20, color="#D84315", label="Crisis regime (2021-2024)")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Avg GR log-return")
    ax.set_xlabel("Date")
    ax.legend(fontsize=7, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG, "fig_c3c_regime_timeline.pdf"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("fig_c3c_regime_timeline.pdf saved.")
    print(f"\nSaved: c3c_regime_es.csv")


if __name__ == "__main__":
    main()
