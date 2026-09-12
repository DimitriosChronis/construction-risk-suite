"""
05_granger_network.py
=====================
H3: directional Granger network among the 8 primary CCI series, plus
the Diebold-Yilmaz spillover index on an 8-variable VAR.

* Pairwise Granger on log-returns (lag by AIC, max 6), 8x8 matrix.
* Diebold-Yilmaz: generalized FEVD (Pesaran-Shin) at h=12 from a
  VAR(p*) on the balanced common sample.
* Periphery-vs-core flow: MED->NORTH significant links vs reverse.
* Fallback (pre-registered): PCA common factor if network empty.

Inputs : data/processed/cci_features_primary.csv
Outputs: results/tables/table_granger_network.csv
         results/tables/table_dy_spillover.csv
         results/tables/table_h3_summary.csv
"""

import os
import warnings
from itertools import permutations

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(SCRIPT_DIR, "..", "data", "processed")
TAB = os.path.join(SCRIPT_DIR, "..", "results", "tables")

MAX_LAG = 6
H = 12
MED = {"GR", "ES", "IT", "PT"}


def main():
    os.makedirs(TAB, exist_ok=True)
    print("=" * 70)
    print("Paper 6 -- H3 Granger network + Diebold-Yilmaz")
    print("=" * 70)

    feat = pd.read_csv(os.path.join(PROC, "cci_features_primary.csv"),
                       parse_dates=["date"])
    wide = feat.pivot(index="date", columns="country",
                      values="log_return").sort_index().dropna()
    countries = list(wide.columns)
    print(f"balanced sample: {len(wide)} months "
          f"[{wide.index.min():%Y-%m} -> {wide.index.max():%Y-%m}]")

    # ---------------- pairwise Granger ----------------
    rows = []
    for a, b in permutations(countries, 2):   # does a -> b ?
        data = wide[[b, a]].values            # [caused, causing]
        best_p, best_lag, best_aic = np.nan, np.nan, np.inf
        res = grangercausalitytests(data, maxlag=MAX_LAG, verbose=False)
        for lag in range(1, MAX_LAG + 1):
            aic = res[lag][1][1].aic          # restricted-model list: [0]=OLS restr? use unrestricted
            p = res[lag][0]["ssr_ftest"][1]
            if aic < best_aic:
                best_aic, best_p, best_lag = aic, p, lag
        rows.append({"causing": a, "caused": b, "lag": int(best_lag),
                     "p_value": round(float(best_p), 4),
                     "flow": ("MED->N" if a in MED and b not in MED else
                              "N->MED" if a not in MED and b in MED else
                              "within")})
    gdf = pd.DataFrame(rows)
    # Benjamini-Hochberg FDR across the 56 directed tests
    # (multiplicity-adjusted counts reported alongside raw p)
    pv = gdf["p_value"].values
    order = np.argsort(pv)
    m = len(pv)
    bh = np.zeros(m, dtype=bool)
    thresh = 0.0
    for rank, idx in enumerate(order, start=1):
        if pv[idx] <= 0.05 * rank / m:
            thresh = pv[idx]
    gdf["bh_fdr_sig"] = (gdf["p_value"] <= thresh).astype(int)
    gdf.to_csv(os.path.join(TAB, "table_granger_network.csv"), index=False)

    sig = gdf[gdf["p_value"] < 0.05]
    n_bh = int(gdf["bh_fdr_sig"].sum())
    print(f"significant links (p<0.05): {len(sig)}/{len(gdf)} "
          f"({len(sig)/len(gdf):.0%}) | BH-FDR 5%: {n_bh}/{len(gdf)}")
    for f, g in sig.groupby("flow"):
        print(f"  {f:7s}: {len(g)} links")

    # ---------------- Diebold-Yilmaz (GENERALIZED, order-invariant) --
    # Pesaran-Shin generalized FEVD, row-normalised (DY 2012); does
    # NOT depend on variable ordering, unlike orthogonalised FEVD.
    var = VAR(wide)
    p_star = var.select_order(MAX_LAG).aic
    fit = var.fit(p_star or 1)
    Sigma = np.asarray(fit.sigma_u)
    A = fit.ma_rep(H - 1)                     # (H, k, k)
    k = Sigma.shape[0]
    num = np.zeros((k, k))
    den = np.zeros(k)
    for h in range(H):
        Ah = A[h]
        AS = Ah @ Sigma
        num += (AS ** 2)
        den += np.einsum("ij,jk,ik->i", Ah, Sigma, Ah)
    gf = num / np.sqrt(np.diag(Sigma))[None, :] ** 2
    gf = gf / den[:, None]
    gf = gf / gf.sum(axis=1, keepdims=True)   # row-normalise
    dy = pd.DataFrame(gf, index=countries, columns=countries)
    to_others = (dy.sum(axis=0) - np.diag(dy)) * 100
    from_others = (1 - np.diag(dy)) * 100
    total = float((dy.sum().sum() - np.trace(dy)) / len(countries)) * 100
    out = pd.DataFrame({"to_others_pct": to_others.round(2),
                        "from_others_pct": from_others.round(2),
                        "net_pct": (to_others - from_others).round(2)})
    out.loc["TOTAL_spillover_pct"] = [total, np.nan, np.nan]
    out.to_csv(os.path.join(TAB, "table_dy_spillover.csv"))
    print(f"DY total spillover: {total:.1f}%  (VAR lag p*={p_star})")
    net_sorted = (to_others - from_others).sort_values(ascending=False)
    print("net transmitters:", ", ".join(
        f"{c}:{v:+.1f}" for c, v in net_sorted.items()))

    med_to_n = len(sig[sig["flow"] == "MED->N"])
    n_to_med = len(sig[sig["flow"] == "N->MED"])
    pd.DataFrame([{
        "n_links_sig": len(sig), "share_sig": round(len(sig)/len(gdf), 3),
        "med_to_north_sig": med_to_n, "north_to_med_sig": n_to_med,
        "dy_total_spillover_pct": round(total, 2),
        "var_lag": int(p_star or 1),
        "top_net_transmitter": net_sorted.index[0],
    }]).to_csv(os.path.join(TAB, "table_h3_summary.csv"), index=False)

    print("\nDONE -- 05_granger_network.py")


if __name__ == "__main__":
    main()
