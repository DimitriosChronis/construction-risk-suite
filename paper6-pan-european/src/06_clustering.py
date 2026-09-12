"""
06_clustering.py
================
H2: does hierarchical clustering of the 8 country CCI return series
separate a Mediterranean cluster (GR, ES, IT, PT) from a Northern
cluster (NL, IE, FI, LT)?

Distance = 1 - Spearman rank correlation of monthly log-returns.
Linkages: Ward, average, complete (sensitivity built in).
Test: permutation on the pre-specified 2-cluster cut -- does the
cut reproduce the MED/NORTH partition, and is the silhouette of the
observed partition higher than under 1,000 random relabelings?

Inputs : data/processed/cci_features_primary.csv
Outputs: results/tables/table_clustering.csv
         results/tables/table_h2_summary.csv
"""

import os

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(SCRIPT_DIR, "..", "data", "processed")
TAB = os.path.join(SCRIPT_DIR, "..", "results", "tables")

MED = {"GR", "ES", "IT", "PT"}
B_PERM = 1000
SEED = 42


def silhouette(D, labels):
    n = len(labels)
    vals = []
    for i in range(n):
        same = [j for j in range(n) if labels[j] == labels[i] and j != i]
        other = [j for j in range(n) if labels[j] != labels[i]]
        if not same or not other:
            return np.nan
        a = np.mean([D[i, j] for j in same])
        b = np.mean([D[i, j] for j in other])
        vals.append((b - a) / max(a, b))
    return float(np.mean(vals))


def main():
    os.makedirs(TAB, exist_ok=True)
    rng = np.random.default_rng(SEED)
    print("=" * 70)
    print("Paper 6 -- H2 Mediterranean vs Northern clustering")
    print("=" * 70)

    feat = pd.read_csv(os.path.join(PROC, "cci_features_primary.csv"),
                       parse_dates=["date"])
    wide = feat.pivot(index="date", columns="country",
                      values="log_return").sort_index().dropna()
    countries = list(wide.columns)
    rho, _ = spearmanr(wide.values)
    D = 1.0 - rho
    np.fill_diagonal(D, 0.0)

    med_labels = np.array([1 if c in MED else 0 for c in countries])
    obs_sil_med = silhouette(D, med_labels)

    rows = []
    for method in ["ward", "average", "complete"]:
        Z = linkage(squareform(D, checks=False), method=method)
        cut = cut_tree(Z, n_clusters=2).flatten()
        # does the 2-cut reproduce MED/NORTH (up to label swap)?
        agree = max((cut == med_labels).mean(),
                    (cut == 1 - med_labels).mean())
        sil_cut = silhouette(D, cut)
        # permutation p for the pre-specified MED/NORTH silhouette
        cnt = 0
        for _ in range(B_PERM):
            perm = rng.permutation(med_labels)
            s = silhouette(D, perm)
            if s is not np.nan and s >= obs_sil_med:
                cnt += 1
        p_perm = (cnt + 1) / (B_PERM + 1)
        rows.append({"linkage": method,
                     "cut2_matches_mednorth_pct": round(agree, 3),
                     "silhouette_cut2": round(sil_cut, 3),
                     "silhouette_medNorth": round(obs_sil_med, 3),
                     "perm_p_medNorth": round(p_perm, 4),
                     "cluster_assignment": "|".join(
                         f"{c}:{k}" for c, k in zip(countries, cut))})
        print(f"[{method:8s}] 2-cut vs MED/NORTH agreement: {agree:.0%} | "
              f"silhouette(cut)={sil_cut:.3f} "
              f"sil(MED/NORTH)={obs_sil_med:.3f} perm p={p_perm:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB, "table_clustering.csv"), index=False)
    df[["linkage", "cut2_matches_mednorth_pct", "silhouette_medNorth",
        "perm_p_medNorth"]].to_csv(
        os.path.join(TAB, "table_h2_summary.csv"), index=False)
    print("\nDONE -- 06_clustering.py")


if __name__ == "__main__":
    main()
