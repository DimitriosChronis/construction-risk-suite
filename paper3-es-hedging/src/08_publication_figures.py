"""
08_publication_figures.py
Paper 2B -- Publication-ready figures for C3 + C4 + C5.

Generates from saved CSV results:
    fig1 - ES(99%) comparison: Independent/Gaussian/Gumbel x 3 regimes  [from 05]
    fig2 - Rolling ES(99%) time series                                   [from 05b]
    fig3 - Regime-conditional ES bar chart (stable vs crisis)            [from 05c]
    fig4 - Regime timeline with shaded windows                           [from 05c]
    fig5 - Lifecycle phase ES profile                                    [from 06]
    fig6 - Bootstrap CI error bars on phase ES                           [from 06b]
    fig7 - Hedge benefit waterfall                                       [from 07]
    fig8 - Hedge effectiveness: full vs crisis                           [from 07b]
    fig9 - Basis risk gap bar chart                                      [from 07c]

Style: ASCE-compatible, no embedded titles, serif font, 300 DPI.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

SCRIPT_DIR = os.path.dirname(__file__)
OUT_FIG    = os.path.join(SCRIPT_DIR, "..", "results", "figures")
OUT_TAB    = os.path.join(SCRIPT_DIR, "..", "results", "tables")

# ASCE style
plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       10,
    "axes.linewidth":  0.8,
    "axes.grid":       True,
    "grid.alpha":      0.3,
    "grid.linewidth":  0.5,
    "figure.dpi":      300,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
})


def fig1_es_comparison():
    """ES(99%) comparison across regimes and copula types (C3)."""
    path = os.path.join(OUT_TAB, "c3_es_comparison.csv")
    if not os.path.exists(path):
        print("SKIP fig1: c3_es_comparison.csv not found")
        return
    df = pd.read_csv(path)
    regimes = df["Regime"].unique()
    copulas = df["Copula"].unique()
    colors = {"Independent": "#90CAF9", "Gaussian": "#FFE082", "Gumbel": "#EF9A9A"}
    fig, axes = plt.subplots(1, len(regimes), figsize=(12, 4), sharey=True)
    if len(regimes) == 1:
        axes = [axes]
    for ax, regime in zip(axes, regimes):
        sub = df[df["Regime"] == regime]
        x = np.arange(len(sub))
        w = 0.35
        ax.bar(x - w / 2, sub["P85_EUR"].values / 1e6, w, label="P85",
               color=[colors.get(c, "#ccc") for c in sub["Copula"]],
               alpha=0.6, edgecolor="grey", linewidth=0.5)
        ax.bar(x + w / 2, sub["ES99_EUR"].values / 1e6, w, label="ES(99%)",
               color=[colors.get(c, "#ccc") for c in sub["Copula"]],
               edgecolor="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["Copula"], fontsize=8)
        short = regime.split("(")[0].strip()
        ax.set_xlabel(short, fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    axes[0].set_ylabel("Cost (EUR millions)")
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG, "fig1_es_comparison.pdf"))
    plt.close()
    print("fig1_es_comparison.pdf saved.")


def fig2_rolling_es():
    """Rolling ES(99%) time series (C3)."""
    path = os.path.join(OUT_TAB, "c3b_rolling_es.csv")
    if not os.path.exists(path):
        print("SKIP fig2: c3b_rolling_es.csv not found")
        return
    roll = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(10, 4))
    for cop, color, ls in [("gaussian", "#1565C0", "--"),
                            ("gumbel", "#D84315", "-")]:
        sub = roll[roll["Copula"] == cop]
        ax.plot(pd.to_datetime(sub["Date"]), sub["ES99_EUR"] / 1e6,
                color=color, linestyle=ls, linewidth=1.5,
                label=f"ES(99%) {cop.title()}")
    ax.axhline(2.3, color="grey", linestyle=":", linewidth=0.8,
               label="Base cost")
    ax.set_ylabel("ES(99%) EUR millions")
    ax.set_xlabel("Date")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG, "fig2_rolling_es.pdf"))
    plt.close()
    print("fig2_rolling_es.pdf saved.")


def fig3_regime_es():
    """Regime-conditional ES: stable vs crisis (C3)."""
    path = os.path.join(OUT_TAB, "c3c_regime_es.csv")
    if not os.path.exists(path):
        print("SKIP fig3: c3c_regime_es.csv not found")
        return
    df = pd.read_csv(path)
    core = df[df["Regime"].isin(["stable", "crisis"])]
    if core.empty:
        print("SKIP fig3: no stable/crisis data")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    regimes = ["stable", "crisis"]
    gumbel_es = [float(core[(core["Regime"] == r) & (core["Copula"] == "gumbel")]["ES99_EUR"].iloc[0])
                 for r in regimes]
    gauss_es = [float(core[(core["Regime"] == r) & (core["Copula"] == "gaussian")]["ES99_EUR"].iloc[0])
                for r in regimes]
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w / 2, [v / 1e6 for v in gumbel_es], w,
           label="Gumbel", color="#1565C0", edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, [v / 1e6 for v in gauss_es], w,
           label="Gaussian", color="#78909C", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Stable\n(2014-2019)", "Crisis\n(2021-2024)"])
    ax.set_ylabel("ES(99%) EUR millions")
    ax.legend(fontsize=8)
    for i, (gv, nv) in enumerate(zip(gumbel_es, gauss_es)):
        prem = gv - nv
        ypos = max(gv, nv) / 1e6 + 0.005
        color = "#C62828" if abs(prem) > 50_000 else "#555555"
        ax.annotate(f"premium: EUR {prem:+,.0f}",
                    xy=(i, ypos), ha="center", fontsize=7, color=color)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG, "fig3_regime_es.pdf"))
    plt.close()
    print("fig3_regime_es.pdf saved.")


def fig4_regime_timeline():
    """Regime timeline already generated by 05c."""
    path = os.path.join(OUT_FIG, "fig_c3c_regime_timeline.pdf")
    if os.path.exists(path):
        print("fig4: fig_c3c_regime_timeline.pdf already exists (from 05c).")
    else:
        print("fig4: Run 05c_regime_switching_es.py to generate.")


def fig5_lifecycle_profile():
    """Phase-specific ES profile (C4)."""
    path = os.path.join(OUT_TAB, "c4_lifecycle_es.csv")
    if not os.path.exists(path):
        print("SKIP fig5: c4_lifecycle_es.csv not found")
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(df))
    w = 0.3
    colors_p85 = ["#90CAF9", "#FFAB91", "#A5D6A7"]
    colors_es  = ["#1565C0", "#D84315", "#2E7D32"]
    ax.bar(x - w / 2, df["P85_EUR"] / 1e6, w,
           color=colors_p85, edgecolor="grey", linewidth=0.5, label="P85")
    bars = ax.bar(x + w / 2, df["ES99_EUR"] / 1e6, w,
                  color=colors_es, edgecolor="black", linewidth=0.8,
                  label="ES(99%)")
    for bar, pct in zip(bars, df["ES99_Overrun_%"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"+{pct:.0f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Phase"])
    ax.set_ylabel("Phase Cost (EUR millions)")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG, "fig5_lifecycle_profile.pdf"))
    plt.close()
    print("fig5_lifecycle_profile.pdf saved.")


def fig6_bootstrap_ci():
    """Bootstrap CI on lifecycle ES (C4)."""
    path = os.path.join(OUT_TAB, "c4b_bootstrap_ci.csv")
    if not os.path.exists(path):
        print("SKIP fig6: c4b_bootstrap_ci.csv not found")
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(df))
    colors = ["#1565C0", "#D84315", "#2E7D32"]
    ax.bar(x, df["ES99_point_EUR"] / 1e6, color=colors,
           edgecolor="black", linewidth=0.8)
    ax.errorbar(x, df["ES99_point_EUR"] / 1e6,
                yerr=[(df["ES99_point_EUR"] - df["ES99_CI_lower"]) / 1e6,
                      (df["ES99_CI_upper"] - df["ES99_point_EUR"]) / 1e6],
                fmt="none", ecolor="black", capsize=5, linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Phase"])
    ax.set_ylabel("ES(99%) EUR millions")
    for i, row in df.iterrows():
        ax.text(i, row["ES99_CI_upper"] / 1e6 + 0.003,
                f"[{row['ES99_CI_lower']/1e6:.3f}, {row['ES99_CI_upper']/1e6:.3f}]",
                ha="center", fontsize=7, color="#555555")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG, "fig6_bootstrap_ci.pdf"))
    plt.close()
    print("fig6_bootstrap_ci.pdf saved.")


def fig7_hedge_waterfall():
    """Hedge benefit waterfall (C5)."""
    path = os.path.join(OUT_TAB, "c5_hedge_quantification.csv")
    if not os.path.exists(path):
        print("SKIP fig7: c5_hedge_quantification.csv not found")
        return
    df = pd.read_csv(path)
    total = df[df["Instrument"].str.contains("TOTAL", na=False)]
    if total.empty or "ES99_Unhedged" not in total.columns:
        print("SKIP fig7: no TOTAL row")
        return
    row = total.iloc[0]
    es_un  = row["ES99_Unhedged"] / 1e6
    es_red = row["ES99_Reduction"] / 1e6
    h_cost = row["Hedge_Cost_EUR"] / 1e6
    es_h   = row["ES99_Hedged"] / 1e6
    fig, ax = plt.subplots(figsize=(7, 4))
    cats = ["ES(99%)\nUnhedged", "Risk\nReduction", "Hedge\nCost", "ES(99%)\nHedged"]
    bots = [0, es_un - es_red, es_un - es_red, 0]
    hts  = [es_un, es_red, h_cost, es_h]
    cols = ["#EF9A9A", "#A5D6A7", "#FFE082", "#90CAF9"]
    bars = ax.bar(cats, hts, bottom=bots, color=cols,
                  edgecolor="black", linewidth=0.8, width=0.5)
    for bar, val in zip(bars, [es_un, es_red, h_cost, es_h]):
        y = bar.get_y() + bar.get_height() / 2
        ax.text(bar.get_x() + bar.get_width() / 2, y,
                f"EUR {val:.3f}M", ha="center", va="center",
                fontsize=8, fontweight="bold")
    ax.set_ylabel("Cost (EUR millions)")
    ax.set_ylim(0, es_un * 1.1)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG, "fig7_hedge_waterfall.pdf"))
    plt.close()
    print("fig7_hedge_waterfall.pdf saved.")


def fig8_hedge_effectiveness():
    """Hedge effectiveness: full vs crisis (C5)."""
    path = os.path.join(OUT_TAB, "c5b_hedge_effectiveness.csv")
    if not os.path.exists(path):
        print("SKIP fig8: c5b_hedge_effectiveness.csv not found")
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w / 2, df["HE_full_%"], w, label="Full period",
           color="#1565C0", edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, df["HE_crisis_%"], w, label="Crisis (2021-24)",
           color="#D84315", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Pair"])
    ax.set_ylabel("Hedge Effectiveness (%)")
    ax.legend(fontsize=8)
    ax.axhline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG, "fig8_hedge_effectiveness.pdf"))
    plt.close()
    print("fig8_hedge_effectiveness.pdf saved.")


def fig9_basis_risk():
    """Basis risk gap bar chart (C5)."""
    path = os.path.join(OUT_TAB, "c5c_basis_risk.csv")
    if not os.path.exists(path):
        print("SKIP fig9: c5c_basis_risk.csv not found")
        return
    df = pd.read_csv(path)
    df = df[df["Weight"].notna()]
    if df.empty:
        print("SKIP fig9: no weighted pairs")
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(df))
    w = 0.28
    ax.bar(x - w, df["gap_full_HE_25pct"], w,
           label="Full: gap to HE=25%", color="#1565C0",
           edgecolor="black", linewidth=0.5)
    ax.bar(x, df["gap_full_HE_50pct"], w,
           label="Full: gap to HE=50%", color="#D84315",
           edgecolor="black", linewidth=0.5)
    # Crisis gap
    crisis_gaps = pd.to_numeric(df["gap_crisis_HE_25pct"], errors="coerce")
    ax.bar(x + w, crisis_gaps, w,
           label="Crisis: gap to HE=25%", color="#78909C",
           edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Pair"])
    ax.set_ylabel("Basis Risk Gap (rho_required - rho_current)")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG, "fig9_basis_risk.pdf"))
    plt.close()
    print("fig9_basis_risk.pdf saved.")


def main():
    os.makedirs(OUT_FIG, exist_ok=True)
    print("Generating Paper 2B publication figures...\n")
    fig1_es_comparison()
    fig2_rolling_es()
    fig3_regime_es()
    fig4_regime_timeline()
    fig5_lifecycle_profile()
    fig6_bootstrap_ci()
    fig7_hedge_waterfall()
    fig8_hedge_effectiveness()
    fig9_basis_risk()
    print("\nDone.")


if __name__ == "__main__":
    main()
