"""
03_vine_network_topology.py
C1 -- Global Commodity Network Topology

Fits R-Vine copula on global + Greek commodity log-returns (10D).
Identifies the network structure: which commodities are central nodes,
and whether Energy (Brent/Fuel) is the exogenous driver.

Input:  data/processed/aligned_log_returns.csv
Output: results/tables/c1_vine_structure.csv
        results/tables/c1_kendall_matrix.csv
        results/figures/fig_c1_kendall_heatmap.pdf
"""

import pandas as pd
import numpy as np
from scipy.stats import rankdata, kendalltau
import pyvinecopulib as pv
import matplotlib.pyplot as plt
import seaborn as sns
import os

SEED      = 42
SCRIPT_DIR = os.path.dirname(__file__)
DATA_PATH  = os.path.join(SCRIPT_DIR, "..", "data", "processed",
                           "aligned_log_returns.csv")
OUT_FIG    = os.path.join(SCRIPT_DIR, "..", "results", "figures")
OUT_TAB    = os.path.join(SCRIPT_DIR, "..", "results", "tables")


def pseudo_obs(df):
    """Rank-based pseudo-observations (PIT)."""
    n = len(df)
    u = np.zeros_like(df.values, dtype=float)
    for j in range(df.shape[1]):
        u[:, j] = rankdata(df.iloc[:, j]) / (n + 1)
    return u


def kendall_matrix(df):
    """Full pairwise Kendall tau matrix."""
    cols = df.columns
    n = len(cols)
    mat = np.eye(n)
    pmat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            tau, p = kendalltau(df.iloc[:, i], df.iloc[:, j])
            mat[i, j] = mat[j, i] = tau
            pmat[i, j] = pmat[j, i] = p
    return pd.DataFrame(mat, index=cols, columns=cols), pd.DataFrame(pmat, index=cols, columns=cols)


def fit_vine(u, d):
    """Fit R-Vine copula."""
    controls = pv.FitControlsVinecop(
        family_set=[pv.BicopFamily.gumbel,
                    pv.BicopFamily.frank,
                    pv.BicopFamily.clayton,
                    pv.BicopFamily.gaussian,
                    pv.BicopFamily.student],
        selection_criterion="aic",
        num_threads=1
    )
    vine = pv.Vinecop(d=d)
    vine.select(np.asfortranarray(u), controls)
    return vine


def main():
    os.makedirs(OUT_FIG, exist_ok=True)
    os.makedirs(OUT_TAB, exist_ok=True)

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    print(f"Data: {df.shape[0]} obs x {df.shape[1]} series")
    print(f"Columns: {list(df.columns)}")

    # ---- Kendall tau matrix ------------------------------------------------
    tau_mat, p_mat = kendall_matrix(df)
    tau_mat.to_csv(os.path.join(OUT_TAB, "c1_kendall_matrix.csv"))
    print("\nKendall tau matrix:")
    print(tau_mat.round(3))

    # ---- Heatmap -----------------------------------------------------------
    short_names = [c.replace("GR_", "GR:").replace("US_", "US:")
                   for c in df.columns]
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(tau_mat, dtype=bool), k=1)
    sns.heatmap(tau_mat.values, mask=mask, annot=True, fmt=".2f",
                xticklabels=short_names, yticklabels=short_names,
                cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG, "fig_c1_kendall_heatmap.pdf"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("\nfig_c1_kendall_heatmap.pdf saved.")

    # ---- Fit R-Vine --------------------------------------------------------
    u = pseudo_obs(df)
    print(f"\nFitting 10D R-Vine copula (n={len(df)})...")
    vine = fit_vine(u, df.shape[1])
    print("R-Vine fitted successfully.")

    ll = vine.loglik(np.asfortranarray(u))
    aic = vine.aic(np.asfortranarray(u))
    bic = vine.bic(np.asfortranarray(u))
    print(f"  Log-likelihood: {ll:.2f}")
    print(f"  AIC: {aic:.2f}")
    print(f"  BIC: {bic:.2f}")

    # ---- Extract pair-copula families --------------------------------------
    d = df.shape[1]
    rows = []
    col_names = list(df.columns)
    structure_order = list(vine.order)  # variable ordering

    for tree_idx in range(vine.trunc_lvl):
        n_edges = d - 1 - tree_idx
        for edge_idx in range(n_edges):
            bc = vine.get_pair_copula(tree_idx, edge_idx)
            fam = str(bc.family)
            tau_val = round(bc.tau, 4)
            params = bc.parameters
            theta = float(np.round(params[0], 4)) if len(params) > 0 else None
            rows.append({
                "Tree":   tree_idx + 1,
                "Edge":   edge_idx,
                "Family": fam,
                "Tau":    tau_val,
                "Theta":  theta,
            })

    vine_df = pd.DataFrame(rows)
    vine_df.to_csv(os.path.join(OUT_TAB, "c1_vine_structure.csv"), index=False)
    print(f"\nVine pair-copulas ({len(rows)} edges):")
    print(vine_df.to_string(index=False))

    # ---- Summary -----------------------------------------------------------
    tree1 = vine_df[vine_df["Tree"] == 1]
    print(f"\nTree 1 edges: {len(tree1)}")
    print(f"Average |tau| in Tree 1: {tree1['Tau'].abs().mean():.4f}")
    print(f"Max |tau| in Tree 1: {tree1['Tau'].abs().max():.4f}")
    print(f"Vine variable order: {structure_order}")
    print(f"  -> mapped: {[col_names[i-1] for i in structure_order]}")


if __name__ == "__main__":
    main()
