"""
03b_network_centrality.py
C1 upgrade -- Network Centrality Analysis

Fits R-Vine on 5D global commodities ONLY (US series).
Computes network centrality measures to identify exogenous driver.
Compares pre-COVID vs post-COVID network structure.

Input:  data/processed/global_log_returns.csv
Output: results/tables/c1b_centrality_measures.csv
        results/tables/c1b_regime_comparison.csv
        results/figures/fig_c1b_centrality.pdf
"""

import pandas as pd
import numpy as np
from scipy.stats import rankdata, kendalltau
import pyvinecopulib as pv
import matplotlib.pyplot as plt
import os

SEED = 42
SCRIPT_DIR = os.path.dirname(__file__)
DATA_PATH  = os.path.join(SCRIPT_DIR, "..", "data", "processed",
                           "global_log_returns.csv")
OUT_FIG    = os.path.join(SCRIPT_DIR, "..", "results", "figures")
OUT_TAB    = os.path.join(SCRIPT_DIR, "..", "results", "tables")

# Regime splits
REGIMES = {
    "Pre-COVID (2000-2019)": ("2000-02-01", "2019-12-01"),
    "Post-COVID (2020-2024)": ("2020-01-01", "2024-12-01"),
    "Full (2000-2024)": ("2000-02-01", "2024-12-01"),
}


def pseudo_obs(df):
    n = len(df)
    u = np.zeros_like(df.values, dtype=float)
    for j in range(df.shape[1]):
        u[:, j] = rankdata(df.iloc[:, j]) / (n + 1)
    return u


def fit_vine(u, d):
    controls = pv.FitControlsVinecop(
        family_set=[pv.BicopFamily.gumbel, pv.BicopFamily.frank,
                    pv.BicopFamily.clayton, pv.BicopFamily.gaussian,
                    pv.BicopFamily.student],
        selection_criterion="aic", num_threads=1
    )
    vine = pv.Vinecop(d=d)
    vine.select(np.asfortranarray(u), controls)
    return vine


def extract_tree1_edges(vine, col_names):
    """Extract Tree 1 variable pairs from the R-vine structure matrix.

    In pyvinecopulib the structure matrix M is d x d (1-indexed variable
    labels).  For Tree 1 (tree_idx=0), each column j encodes one edge
    that connects M[0, j] and M[d-1, j] (the diagonal element).
    """
    M = np.array(vine.matrix)  # d x d, 1-indexed
    d = M.shape[0]
    edges = []
    for j in range(d - 1):
        # 1-indexed variable labels -> 0-indexed column positions
        var_a = int(M[0, j]) - 1
        var_b = int(M[d - 1, j]) - 1
        bc = vine.get_pair_copula(0, j)
        edges.append((col_names[var_a], col_names[var_b], abs(bc.tau),
                       str(bc.family)))
    return edges


def compute_centrality(df, vine):
    """Compute degree and strength centrality from Tree 1 edges."""
    col_names = list(df.columns)

    degree = {c: 0 for c in col_names}
    strength = {c: 0.0 for c in col_names}

    edges = extract_tree1_edges(vine, col_names)
    for name_a, name_b, tau_val, _ in edges:
        degree[name_a] += 1
        degree[name_b] += 1
        strength[name_a] += tau_val
        strength[name_b] += tau_val

    centrality = pd.DataFrame({
        "Variable": col_names,
        "Degree": [degree[c] for c in col_names],
        "Strength": [round(strength[c], 4) for c in col_names],
    })
    centrality["Rank"] = centrality["Strength"].rank(ascending=False).astype(int)
    centrality = centrality.sort_values("Rank")
    return centrality


def main():
    os.makedirs(OUT_FIG, exist_ok=True)
    os.makedirs(OUT_TAB, exist_ok=True)

    df_full = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    print(f"Global data: {df_full.shape}")

    regime_results = []

    for regime_name, (start, end) in REGIMES.items():
        df = df_full.loc[start:end].dropna()
        if len(df) < 24:
            print(f"SKIP {regime_name}: only {len(df)} obs")
            continue

        print(f"\n{'='*60}")
        print(f"{regime_name} (n={len(df)})")
        print(f"{'='*60}")

        u = pseudo_obs(df)
        vine = fit_vine(u, df.shape[1])
        ll = vine.loglik(np.asfortranarray(u))
        aic = vine.aic(np.asfortranarray(u))

        centrality = compute_centrality(df, vine)
        centrality["Regime"] = regime_name
        centrality["n_obs"] = len(df)
        centrality["AIC"] = round(aic, 1)
        regime_results.append(centrality)

        print(f"  AIC={aic:.1f}, LL={ll:.1f}")
        print(centrality[["Variable", "Degree", "Strength", "Rank"]].to_string(index=False))

        # Vine order
        order = list(vine.order)
        print(f"  Vine root: {df.columns[order[0]-1]}")

    # Combine and save
    all_cent = pd.concat(regime_results, ignore_index=True)
    all_cent.to_csv(os.path.join(OUT_TAB, "c1b_centrality_measures.csv"), index=False)

    # Regime comparison: who is central in each regime?
    pivot = all_cent.pivot_table(index="Variable", columns="Regime",
                                 values="Strength", aggfunc="first")
    pivot.to_csv(os.path.join(OUT_TAB, "c1b_regime_comparison.csv"))
    print("\nCentrality across regimes:")
    print(pivot.round(4))

    # ── Bootstrap stability for post-COVID sub-sample (n=60) ────────────
    print("\n" + "="*60)
    print("Bootstrap stability check (Post-COVID, B=200)")
    print("="*60)
    df_post = df_full.loc["2020-01-01":"2024-12-01"].dropna()
    B = 200
    rng = np.random.default_rng(SEED)
    boot_strength = {c: [] for c in df_post.columns}
    boot_roots = []
    for b in range(B):
        idx = rng.choice(len(df_post), size=len(df_post), replace=True)
        df_b = df_post.iloc[idx]
        u_b = pseudo_obs(df_b)
        try:
            vine_b = fit_vine(u_b, df_b.shape[1])
            cent_b = compute_centrality(df_b, vine_b)
            for _, row in cent_b.iterrows():
                boot_strength[row["Variable"]].append(row["Strength"])
            order_b = list(vine_b.order)
            boot_roots.append(df_b.columns[order_b[0]-1])
        except Exception:
            pass

    print(f"  Successful bootstrap samples: {len(boot_roots)}/{B}")
    print(f"  Vine-strength centrality (bootstrap mean +/- sd):")
    boot_summary_rows = []
    for c in df_post.columns:
        vals = boot_strength[c]
        if vals:
            m, s = np.mean(vals), np.std(vals)
            print(f"    {c:18s}  mean={m:.4f}  sd={s:.4f}  "
                  f"95%CI=[{np.percentile(vals,2.5):.4f}, {np.percentile(vals,97.5):.4f}]")
            boot_summary_rows.append({
                "Variable": c, "Boot_mean": round(m,4),
                "Boot_sd": round(s,4),
                "CI_lo": round(np.percentile(vals,2.5),4),
                "CI_hi": round(np.percentile(vals,97.5),4)
            })

    from collections import Counter
    root_counts = Counter(boot_roots)
    print(f"  Root frequency (top 3):")
    for root, cnt in root_counts.most_common(3):
        print(f"    {root}: {cnt}/{len(boot_roots)} ({100*cnt/len(boot_roots):.1f}%)")

    pd.DataFrame(boot_summary_rows).to_csv(
        os.path.join(OUT_TAB, "c1b_bootstrap_stability.csv"), index=False)

    # Bar chart
    fig, axes = plt.subplots(1, len(REGIMES), figsize=(14, 4), sharey=True)
    for ax, regime_name in zip(axes, REGIMES.keys()):
        sub = all_cent[all_cent["Regime"] == regime_name].sort_values("Strength", ascending=True)
        if sub.empty:
            continue
        short = [v.replace("US_", "") for v in sub["Variable"]]
        ax.barh(short, sub["Strength"], color="#1565C0", edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Strength Centrality")
        ax.set_title(regime_name.split("(")[0].strip(), fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG, "fig_c1b_centrality.pdf"), dpi=300, bbox_inches="tight")
    plt.close()
    print("\nfig_c1b_centrality.pdf saved.")


if __name__ == "__main__":
    main()
