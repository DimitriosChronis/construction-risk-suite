"""
08b_revision_figures.py
=======================
Publication-ready figures generated from the revision scripts
04d_system_varx, 04e_cholesky_sensitivity, and 05b_revised_contingency.

Produces (PDF + PNG):
  fig_R1_bivariate_vs_system_fevd  -- side-by-side bars
  fig_R2_block_exogeneity          -- p-value heatmap
  fig_R3_cholesky_sensitivity      -- IRF cumulative diff
  fig_R4_regime_conditional_fevd   -- pre/post-COVID FEVD bars
  fig_R5_revised_contingency       -- A/B/D contingency stack
  fig_R6_scaling_comparison        -- sqrtH vs bootstrap VaR

Run AFTER 04d, 04e, 05b have produced their CSVs.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "font.family"    : "serif",
    "font.size"      : 10,
    "axes.titlesize" : 11,
    "axes.labelsize" : 10,
    "legend.fontsize": 8,
    "figure.dpi"     : 150,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TAB_DIR    = os.path.join(SCRIPT_DIR, "..", "results", "tables")
FIG_DIR    = os.path.join(SCRIPT_DIR, "..", "results", "figures")


def _save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(FIG_DIR, f"{name}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {name}.pdf / .png")


def fig_bivariate_vs_system():
    cmp = pd.read_csv(os.path.join(TAB_DIR,
                                    "c2d_bivariate_vs_system_fevd.csv"))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(cmp))
    w = 0.30
    b1 = ax.bar(x - w, cmp["Bivariate_FEVD"], w,
                color="#7f8c8d", label="Bivariate (v1)")
    b2 = ax.bar(x,     cmp["System_FEVD_matched"], w,
                color="#2980b9", label="System: matched US")
    b3 = ax.bar(x + w, cmp["System_FEVD_cross_total"], w,
                color="#c0392b", label="System: cross-material total")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("GR_", "") for s in cmp["Greek_series"]],
                       rotation=15, ha="right")
    ax.set_ylabel("FEVD share at h = 12 months")
    ax.set_title("Bivariate vs System FEVD: matched + cross-material spillovers")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    for bars in (b1, b2, b3):
        for bar in bars:
            v = bar.get_height()
            if v > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                        f"{v:.2f}", ha="center", fontsize=8)
    _save(fig, "fig_R1_bivariate_vs_system_fevd")


def fig_block_exogeneity():
    be = pd.read_csv(os.path.join(TAB_DIR, "c2d_block_exogeneity.csv"))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    pmat = be[["p_matched", "p_cross_material"]].values
    im = ax.imshow(pmat, cmap="RdYlGn_r", vmin=0, vmax=0.10, aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Matched US Granger p",
                        "Cross-material US block-exogeneity p"])
    ax.set_yticks(range(len(be)))
    ax.set_yticklabels([s.replace("GR_", "") for s in be["Greek_series"]])
    for i in range(len(be)):
        for j in range(2):
            v = pmat[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=10,
                    color="white" if v < 0.05 else "black",
                    fontweight="bold" if v < 0.05 else "normal")
    plt.colorbar(im, ax=ax, label="p-value")
    ax.set_title("System Granger causality: matched vs cross-material US")
    _save(fig, "fig_R2_block_exogeneity")


def fig_cholesky_sensitivity():
    biv = pd.read_csv(os.path.join(TAB_DIR,
                                   "c2e_cholesky_sensitivity.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: cumulative IRF under both orderings
    ax = axes[0]
    x = np.arange(len(biv))
    w = 0.40
    ax.bar(x - w/2, biv["cum_IRF_US_first"], w,
           color="#2980b9", label="US-first (published)")
    ax.bar(x + w/2, biv["cum_IRF_GR_first"], w,
           color="#e67e22", label="GR-first (reversed)")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(biv["pair"], rotation=15, ha="right")
    ax.set_ylabel("Cumulative 12M IRF")
    ax.set_title("(a) US -> GR cumulative IRF under reversed ordering")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    # Panel B: % difference
    ax = axes[1]
    colors = ["#27ae60" if abs(d) < 10 else "#c0392b"
              for d in biv["pct_diff"]]
    ax.bar(x, biv["pct_diff"], color=colors)
    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(10, color="gray", ls="--", lw=0.5,
               label="+/- 10% reference")
    ax.axhline(-10, color="gray", ls="--", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(biv["pair"], rotation=15, ha="right")
    ax.set_ylabel("% difference (GR-first vs US-first)")
    ax.set_title("(b) Ordering sensitivity")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    _save(fig, "fig_R3_cholesky_sensitivity")


def fig_regime_conditional_fevd():
    path = os.path.join(TAB_DIR, "c2d_regime_conditional_fevd.csv")
    if not os.path.exists(path):
        print("  [skip] regime-conditional FEVD file not found")
        return
    reg = pd.read_csv(path)
    if reg.empty:
        print("  [skip] regime-conditional FEVD file is empty")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = reg.pivot(index="Greek_series", columns="regime",
                      values="FEVD_matched")
    if "pre_COVID" in pivot.columns and "post_COVID" in pivot.columns:
        x = np.arange(len(pivot))
        w = 0.35
        ax.bar(x - w/2, pivot["pre_COVID"], w,
               color="#3498db", label="Pre-COVID (<2020-03)")
        ax.bar(x + w/2, pivot["post_COVID"], w,
               color="#c0392b", label="Post-COVID (>=2020-03)")
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("GR_", "") for s in pivot.index],
                           rotation=15, ha="right")
        ax.set_ylabel("System FEVD share at h = 12")
        ax.set_title("Regime-conditional FEVD: pre/post-COVID structural break")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        _save(fig, "fig_R4_regime_conditional_fevd")


def fig_revised_contingency():
    cont = pd.read_csv(os.path.join(TAB_DIR,
                                     "c2_5b_revised_contingency.csv"))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    labels = [s.replace("GR_", "") for s in cont["Greek_series"]]
    x = np.arange(len(cont))
    w = 0.20
    series = [
        ("EUR_A_v1_published",        "A: v1 published",      "#7f8c8d"),
        ("EUR_B_sysFEVD_sqrtH",       "B: sysFEVD + sqrtH",   "#2980b9"),
        ("EUR_C_bivFEVD_boot",        "C: bivFEVD + boot",    "#16a085"),
        ("EUR_D_sysFEVD_boot",        "D: sysFEVD + boot (revised)", "#c0392b"),
    ]
    for i, (col, label, color) in enumerate(series):
        offset = (i - 1.5) * w
        ax.bar(x + offset, cont[col].astype(float), w,
               color=color, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("EUR contingency (per EUR 2.3M project)")
    ax.set_title("Revised EUR contingency: 4 specifications")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    _save(fig, "fig_R5_revised_contingency")


def fig_scaling_comparison():
    sc = pd.read_csv(os.path.join(TAB_DIR,
                                   "c2_5b_scaling_comparison.csv"))
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(sc))
    w = 0.40
    ax.bar(x - w/2, sc["VaR_sqrtH"], w,
           color="#7f8c8d", label="sqrt-H scaling")
    ax.bar(x + w/2, sc["VaR_bootstrap"], w,
           color="#c0392b", label="Block bootstrap (L=6, B=5000)")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("GR_", "") for s in sc["Greek_series"]],
                       rotation=15, ha="right")
    ax.set_ylabel("24-month one-sided 95% VaR (log return)")
    ax.set_title("Time scaling: sqrt-H vs serial-correlation-preserving bootstrap")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    for i, row in sc.iterrows():
        ax.annotate(f"x{row['ratio_boot_to_sqrtH']:.2f}",
                    (x[i] + w/2, row["VaR_bootstrap"]),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8)
    _save(fig, "fig_R6_scaling_comparison")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print("=" * 70)
    print("Paper 2 -- Revision figures (08b)")
    print("=" * 70)
    fig_bivariate_vs_system()
    fig_block_exogeneity()
    fig_cholesky_sensitivity()
    fig_regime_conditional_fevd()
    fig_revised_contingency()
    fig_scaling_comparison()
    print("\nDONE -- all revision figures generated.")


if __name__ == "__main__":
    main()
