"""
09_publication_figures.py
=========================
G4: publication figures. Every figure is saved as PDF + EPS + PNG
(300 dpi) at creation.

fig1_lambdaU_timeline : panel-mean lambda_U (w=24, w=36) with the
                        exogenous EU crisis windows shaded (H1)
fig2_granger_network  : Granger p heatmap + DY net-spillover bars (H3)
fig3_clustering       : Spearman correlation heatmap + Ward dendrogram
                        (H2 -- the convergence/null finding)
fig4_bridge           : CSRI vs GR sigma6 timeline + lagged Spearman
fig5_h4_effects       : transmission betas core vs Mediterranean
fig6_ukraine          : sigma6 shock ratios + panel lambda_U 2021-23
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(SCRIPT_DIR, "..", "data", "processed")
TAB = os.path.join(SCRIPT_DIR, "..", "results", "tables")
FIG = os.path.join(SCRIPT_DIR, "..", "results", "figures")

plt.rcParams.update({"font.family": "serif", "font.size": 10})

EXOG = [("2008-09", "2009-12"), ("2011-07", "2012-12"),
        ("2020-03", "2020-12"), ("2022-02", "2022-12")]
MED = {"GR", "ES", "IT", "PT"}


def save(fig, name):
    for ext in ("pdf", "eps", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {name}.pdf/.eps/.png")


def shade(ax):
    for a, b in EXOG:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), alpha=0.15,
                   color="red", lw=0)


def fig1():
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for w, c in [(24, "#2c3e50"), (36, "#2980b9")]:
        ts = pd.read_csv(os.path.join(PROC, f"lambdaU_pairs_w{w}.csv"),
                         index_col=0, parse_dates=True)
        pm = ts.mean(axis=1).dropna()
        ax.plot(pm.index, pm.values, lw=1.6, color=c,
                label=f"panel mean $\\lambda_U$ (w={w}M)")
    shade(ax)
    ax.set_ylabel("Panel-mean upper tail dependence")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_title("Cross-country tail dependence, eight euro-area "
                 "construction-cost indices (shaded: EU crisis windows)")
    save(fig, "fig1_lambdaU_timeline")


def fig2():
    g = pd.read_csv(os.path.join(TAB, "table_granger_network.csv"))
    dy = pd.read_csv(os.path.join(TAB, "table_dy_spillover.csv"),
                     index_col=0).drop(index="TOTAL_spillover_pct")
    countries = sorted(set(g["causing"]) | set(g["caused"]))
    n = len(countries)
    M = np.full((n, n), np.nan)
    for _, r in g.iterrows():
        M[countries.index(r["causing"]), countries.index(r["caused"])] = \
            r["p_value"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    ax = axes[0]
    im = ax.imshow(M, cmap="RdYlGn_r", vmin=0, vmax=0.10)
    ax.set_xticks(range(n)); ax.set_xticklabels(countries)
    ax.set_yticks(range(n)); ax.set_yticklabels(countries)
    ax.set_xlabel("caused"); ax.set_ylabel("causing")
    for i in range(n):
        for j in range(n):
            if i != j and M[i, j] == M[i, j]:
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if M[i, j] < 0.05 else "black")
    plt.colorbar(im, ax=ax, label="Granger p-value")
    ax.set_title("Pairwise Granger causality (AIC lag)")
    ax = axes[1]
    net = dy["net_pct"].astype(float).sort_values()
    colors = ["#c0392b" if c in MED else "#2980b9" for c in net.index]
    ax.barh(net.index, net.values, color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Diebold-Yilmaz net spillover (%)")
    ax.set_title("Net transmitters (red: Mediterranean)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    save(fig, "fig2_granger_network")


def fig3():
    feat = pd.read_csv(os.path.join(PROC, "cci_features_primary.csv"),
                       parse_dates=["date"])
    wide = feat.pivot(index="date", columns="country",
                      values="log_return").sort_index().dropna()
    rho, _ = spearmanr(wide.values)
    countries = list(wide.columns)
    D = 1 - rho; np.fill_diagonal(D, 0)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    ax = axes[0]
    im = ax.imshow(rho, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(countries))); ax.set_xticklabels(countries)
    ax.set_yticks(range(len(countries))); ax.set_yticklabels(countries)
    for i in range(len(countries)):
        for j in range(len(countries)):
            ax.text(j, i, f"{rho[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if rho[i, j] < 0.6 else "k")
    plt.colorbar(im, ax=ax, label="Spearman correlation")
    ax.set_title("Return correlations 2000-2026")
    ax = axes[1]
    Z = linkage(squareform(D, checks=False), method="ward")
    dendrogram(Z, labels=countries, ax=ax, color_threshold=0)
    ax.set_ylabel("distance (1 - rank correlation)")
    ax.set_title("Ward dendrogram: no Mediterranean/Northern split "
                 "(H2 null)")
    fig.tight_layout()
    save(fig, "fig3_clustering")


def fig4():
    feat = pd.read_csv(os.path.join(PROC, "cci_features_primary.csv"),
                       parse_dates=["date"])
    gr = feat[feat["country"] == "GR"].set_index("date")["sigma6"]
    csri = pd.read_csv(os.path.join(
        SCRIPT_DIR, "..", "..", "paper5-portfolio-contagion", "data",
        "processed", "csri_monthly.csv"), index_col=0,
        parse_dates=True)["CSRI"]
    lag = pd.read_csv(os.path.join(TAB, "table_bridge_lagged.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5),
                             gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    ax.plot(csri.index, csri.values, color="#2c3e50", lw=1.4,
            label="CSRI (micro, P5 pipeline)")
    ax2 = ax.twinx()
    ax2.plot(gr.index, gr.values, color="#c0392b", lw=1.4,
             label="GR CCI 6M volatility (macro)")
    ax.set_ylabel("CSRI (z)"); ax2.set_ylabel("sigma6")
    ax.set_title("Bridge: microdata systemic-risk index vs macro "
                 "cost-index volatility (complementary facets)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    ax = axes[1]
    ax.bar(lag["lag_csri_leads"], lag["spearman"], color="#2980b9")
    ax.axhline(0.5, color="red", ls="--", lw=1,
               label="pre-registered threshold")
    ax.set_xlabel("lag (CSRI leads, months)")
    ax.set_ylabel("Spearman rho")
    ax.set_ylim(0, 0.6)
    ax.legend(fontsize=8)
    ax.set_title("Lagged correlation")
    fig.tight_layout()
    save(fig, "fig4_bridge")


def fig5():
    h4 = pd.read_csv(os.path.join(TAB, "table_h4_summary.csv")).iloc[0]
    wb = pd.read_csv(os.path.join(TAB, "table_h4_wildboot.csv")).iloc[0]
    b1 = h4["beta1_core_ppsigma"]
    b2 = h4["beta2_med_extra_ppsigma"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(["Northern (core)", "Mediterranean"], [b1, b1 + b2],
           color=["#2980b9", "#c0392b"])
    for x, v in zip([0, 1], [b1, b1 + b2]):
        ax.text(x, v + 0.003, f"{v:.3f}", ha="center", fontweight="bold")
    ax.set_ylabel("pp CCI growth per 1 sigma US material volatility "
                  "(lag 1M)")
    ax.set_title(f"H4: transmission {h4['med_to_core_ratio']:.1f}x "
                 f"stronger in the Mediterranean bloc\n"
                 f"(interaction wild-cluster bootstrap p = "
                 f"{wb['wild_boot_p']:.3f}; provisional)")
    ax.grid(alpha=0.3, axis="y")
    save(fig, "fig5_h4_effects")


def fig6():
    case = pd.read_csv(os.path.join(TAB, "table_ukraine_case.csv"))
    net = pd.read_csv(os.path.join(TAB, "table_ukraine_network.csv"),
                      index_col=0, parse_dates=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    ax = axes[0]
    colors = ["#c0392b" if c in MED else "#2980b9"
              for c in case["country"]]
    ax.bar(case["country"], case["shock_ratio"], color=colors)
    ax.axhline(1, color="k", lw=0.8)
    ax.set_ylabel("sigma6 Ukraine window / 2019 baseline")
    ax.set_title("2022 energy shock by country (red: Mediterranean)")
    ax.grid(alpha=0.3, axis="y")
    ax = axes[1]
    ax.plot(net.index, net["panel_mean_lambdaU"], color="#2c3e50", lw=1.8)
    ax.axvspan(pd.Timestamp("2022-02"), pd.Timestamp("2022-12"),
               alpha=0.15, color="red", lw=0)
    peak = net["panel_mean_lambdaU"].idxmax()
    ax.annotate(f"all-time peak {net['panel_mean_lambdaU'].max():.2f}\n"
                f"({peak:%Y-%m})",
                xy=(peak, net["panel_mean_lambdaU"].max()),
                xytext=(peak, net["panel_mean_lambdaU"].max() * 0.8),
                arrowprops=dict(arrowstyle="->"), fontsize=8)
    ax.set_ylabel("panel-mean lambda_U (w=24)")
    ax.set_title("Network tail dependence around the invasion")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "fig6_ukraine")


def main():
    os.makedirs(FIG, exist_ok=True)
    print("=" * 70)
    print("Paper 6 -- G4 publication figures (PDF+EPS+PNG)")
    print("=" * 70)
    for f in [fig1, fig2, fig3, fig4, fig5, fig6]:
        f()
    print("\nDONE -- 09_publication_figures.py")


if __name__ == "__main__":
    main()
