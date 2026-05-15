"""
Paper 5 — Publication Figures
===================================
Produces the 8 AiC-ready figures for Paper 5.

Figure 1  Framework architecture (schematic pipeline)
Figure 2  Dynamic copula network  (nodes = types, edges = avg λ_U)
Figure 3  Contagion index timeline  (CI_i(t), crisis shading)
Figure 4  Portfolio ES vs sum of individual ES  (by regime)
Figure 5  Source vs Receiver classification  (NET flow per type)
Figure 6  LSTM portfolio agent — ROC + predictions overlay
Figure 7  Walk-forward validation (per-window AUC)
Figure 8  CSRI timeline (with ±1σ bands & crisis shading)

Outputs are saved as both PDF (vector, for paper) and PNG (300 dpi, preview)
into ../results/.
"""

import io
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

try:
    import networkx as nx
    HAVE_NX = True
except ImportError:
    HAVE_NX = False

from sklearn.metrics import roc_curve, roc_auc_score

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
PROC_DIR    = "../data/processed/"
RESULTS_DIR = "../results/"
FIG_DIR     = os.path.join(RESULTS_DIR, "figures/")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "figure.dpi":         120,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "axes.axisbelow":     True,
})

TYPE_COLORS = {
    "road":     "#E74C3C",
    "bridge":   "#3498DB",
    "pipeline": "#27AE60",
    "building": "#F39C12",
    "other":    "#8E44AD",
}
CRISIS_COLOR  = "#F8D7DA"
STABLE_COLOR  = "#D4EDDA"
ACCENT        = "#2C3E50"


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  Saved: {FIG_DIR}{name}.{{pdf,png}}")


def shade_crisis(ax, regime_series):
    """Shade x-regions where regime == 'crisis'."""
    in_crisis = False
    start = None
    for t, r in regime_series.items():
        if r == "crisis" and not in_crisis:
            start = t; in_crisis = True
        elif r != "crisis" and in_crisis:
            ax.axvspan(start, t, color=CRISIS_COLOR, alpha=0.6, zorder=0)
            in_crisis = False
    if in_crisis:
        ax.axvspan(start, regime_series.index[-1], color=CRISIS_COLOR,
                   alpha=0.6, zorder=0)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 64)
print("Paper 5 — Publication Figures")
print("=" * 64)
print("\nLoading data")

net  = pd.read_csv(PROC_DIR + "dynamic_copula_network_avg.csv",
                    index_col=0, parse_dates=True).sort_index()
lamU = pd.read_csv(PROC_DIR + "dynamic_copula_lambdaU.csv",
                    index_col=0, parse_dates=True).sort_index()
reg  = pd.read_csv(PROC_DIR + "dynamic_copula_regimes.csv",
                    index_col=0, parse_dates=True).sort_index()
CI   = pd.read_csv(PROC_DIR + "contagion_index.csv",
                    index_col=0, parse_dates=True).sort_index()
flows = pd.read_csv(PROC_DIR + "contagion_flows.csv",
                    index_col=0, parse_dates=True).sort_index()
es_reg = pd.read_csv(PROC_DIR + "portfolio_es_by_regime.csv")
es_roll = pd.read_csv(PROC_DIR + "portfolio_es_rolling.csv",
                      index_col=0, parse_dates=True).sort_index()
# Use walkforward file (backward-compatible column names: lstm, xgb, logit, actual)
agent = pd.read_csv(PROC_DIR + "agent_walkforward_auc.csv",
                    parse_dates=["date"]).set_index("date").sort_index()
wf    = pd.read_csv(PROC_DIR + "agent_walkforward_auc.csv")
csri  = pd.read_csv(PROC_DIR + "csri_monthly.csv",
                    index_col=0, parse_dates=True).sort_index()
clf   = pd.read_csv(PROC_DIR + "contagion_classification.csv")

TYPES = list(CI.columns)
print(f"  Types: {TYPES}")
print(f"  λ_U coverage: {lamU.index.min():%Y-%m} → {lamU.index.max():%Y-%m}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — FRAMEWORK ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
print("\nFig 1: Framework architecture")

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.set_xlim(0, 10); ax.set_ylim(0, 6)
ax.axis("off")

boxes = [
    (0.3, 4.5, 1.9, 1.0, "Διαύγεια\nawards API", "#D6EAF8"),
    (2.5, 4.5, 1.9, 1.0, "ELSTAT\nprice indices", "#D6EAF8"),
    (4.7, 4.5, 1.9, 1.0, "US PPI\nfeatures (P4)", "#D6EAF8"),
    (0.3, 2.8, 2.5, 1.0, "02 Portfolio\nconstruction\n(N×T matrix)", "#FCF3CF"),
    (3.1, 2.8, 2.5, 1.0, "03 Dynamic\ncopula\n(λ_U rolling)", "#FCF3CF"),
    (5.9, 2.8, 2.5, 1.0, "04 Contagion\nindex CI_i\n(src↔recv)", "#FCF3CF"),
    (0.3, 1.1, 2.5, 1.0, "05 Portfolio\nES\n(DB → 0 crisis)", "#FADBD8"),
    (3.1, 1.1, 2.5, 1.0, "06 LSTM\nportfolio agent\n(lead=2M)", "#FADBD8"),
    (5.9, 1.1, 2.5, 1.0, "07 CSRI\n+ Granger\n+ bootstrap", "#FADBD8"),
    (8.1, 2.2, 1.7, 1.6, "08\nFigures &\nTables\n(Q1)", "#D5F5E3"),
]
for x, y, w, h, text, color in boxes:
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                  boxstyle="round,pad=0.08", lw=1.2,
                  ec=ACCENT, fc=color))
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=9)

arrows = [
    ((1.25, 4.5), (1.55, 3.8)),
    ((3.45, 4.5), (4.35, 3.8)),
    ((5.65, 4.5), (4.35, 3.8)),
    ((2.8, 3.3), (3.1, 3.3)),
    ((5.6, 3.3), (5.9, 3.3)),
    ((1.55, 2.8), (1.55, 2.1)),
    ((4.35, 2.8), (4.35, 2.1)),
    ((7.15, 2.8), (7.15, 2.1)),
    ((2.8, 1.6), (3.1, 1.6)),
    ((5.6, 1.6), (5.9, 1.6)),
    ((8.4, 2.1), (8.8, 2.2)),
    ((8.4, 3.3), (8.8, 3.0)),
]
for (x1, y1), (x2, y2) in arrows:
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                 arrowstyle="->", lw=1.3, color=ACCENT,
                 mutation_scale=12))

ax.text(5, 5.85, "Paper 5 — Portfolio Contagion Framework",
        ha="center", fontsize=12, fontweight="bold", color=ACCENT)
save(fig, "fig1_framework")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — DYNAMIC COPULA NETWORK
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 2: Dynamic copula network")

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

def draw_network(ax, avg_lamU_vec, title):
    """Draw a circular 5-node graph with edge widths ∝ λ_U."""
    n = len(TYPES)
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, n, endpoint=False)
    pos = {t: (np.cos(a), np.sin(a)) for t, a in zip(TYPES, angles)}

    # Edges
    w_max = max(avg_lamU_vec.values()) or 1.0
    for (a, b), w in avg_lamU_vec.items():
        xa, ya = pos[a]; xb, yb = pos[b]
        ax.plot([xa, xb], [ya, yb],
                lw=0.5 + 5 * (w / w_max),
                color=ACCENT, alpha=0.35 + 0.55 * (w / w_max),
                zorder=1)

    # Nodes — color by role (from classification)
    role_color = {"source": "#C0392B", "receiver": "#2874A6"}
    role_map = dict(zip(clf["type"], clf["role"]))
    for t, (x, y) in pos.items():
        ax.scatter(x, y, s=1400,
                    c=role_color.get(role_map.get(t, "receiver")),
                    edgecolors=ACCENT, lw=1.5, zorder=2)
        ax.text(x, y, t, ha="center", va="center",
                 color="white", fontsize=9, fontweight="bold", zorder=3)

    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=11)

def avg_over_mask(mask):
    out = {}
    sub = lamU.loc[mask]
    for col in sub.columns:
        a, b = col.split("__")
        out[(a, b)] = sub[col].mean()
    return out

draw_network(axes[0], avg_over_mask(reg["regime"] == "stable"),
             "STABLE regime  (edge width ∝ mean λ_U)")
draw_network(axes[1], avg_over_mask(reg["regime"] == "crisis"),
             "CRISIS regime  (edge width ∝ mean λ_U)")

fig.legend(handles=[
    mpatches.Patch(color="#C0392B", label="Source (net outflow)"),
    mpatches.Patch(color="#2874A6", label="Receiver (net inflow)"),
], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))

fig.suptitle("Figure 2 — Dynamic copula contagion network (λ_U)",
             fontsize=12, fontweight="bold", y=1.0)
save(fig, "fig2_network")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — CONTAGION INDEX TIMELINE
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 3: Contagion index timeline")

fig, ax = plt.subplots(figsize=(10, 4.5))
shade_crisis(ax, reg["regime"])
for t in TYPES:
    ax.plot(CI.index, CI[t], color=TYPE_COLORS[t], lw=1.3, label=t)
ax.set_ylabel("Contagion index  CI$_i$(t)")
ax.set_title("Figure 3 — Time-varying contagion index per project type",
             fontweight="bold")
ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.25))
ax.set_ylim(CI.min().min() * 0.98, 1.0)
save(fig, "fig3_contagion_timeline")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — PORTFOLIO ES vs Σ INDIVIDUAL ES
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 4: Portfolio ES vs sum of individual ES")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

bar_df = es_reg[es_reg["weights"] == "budget"].set_index("regime")
regs = ["stable", "crisis", "full"]
bar_df = bar_df.loc[[r for r in regs if r in bar_df.index]]
x = np.arange(len(bar_df))
w = 0.35
axes[0].bar(x - w/2, bar_df["ES_sum"],       w, label="Σ ES$_i$",
            color="#85C1E9", edgecolor=ACCENT)
axes[0].bar(x + w/2, bar_df["ES_portfolio"], w, label="ES$_P$",
            color="#E74C3C", edgecolor=ACCENT)
axes[0].set_xticks(x); axes[0].set_xticklabels(bar_df.index)
axes[0].set_ylabel("Expected Shortfall (α=0.95)")
axes[0].set_title("ES_P vs Σ ES$_i$ by regime  (budget weights)")
axes[0].legend()

# Rolling DB
axes[1].axhline(0, color=ACCENT, lw=0.8, ls="--")
shade_crisis(axes[1], reg["regime"].reindex(es_roll.index))
axes[1].plot(es_roll.index, es_roll["div_benefit"],
             color="#8E44AD", lw=1.3)
axes[1].set_ylabel("Diversification benefit  DB(t) = 1 − ES$_P$/ΣES$_i$")
axes[1].set_title("Rolling 24M diversification benefit")

fig.suptitle("Figure 4 — Portfolio ES and diversification benefit",
             fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "fig4_portfolio_es")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — SOURCE vs RECEIVER
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 5: Source vs receiver classification")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

clf_sorted = clf.sort_values("NET", ascending=True)
colors = ["#2874A6" if v < 0 else "#C0392B" for v in clf_sorted["NET"]]
axes[0].barh(clf_sorted["type"], clf_sorted["NET"],
             color=colors, edgecolor=ACCENT)
axes[0].axvline(0, color=ACCENT, lw=1)
axes[0].set_xlabel("NET = OUT − IN")
axes[0].set_title("Net directional intensity")

axes[1].barh(clf_sorted["type"], clf_sorted["OUT"], color="#E67E22",
             edgecolor=ACCENT, label="OUT (leads)")
axes[1].barh(clf_sorted["type"], -clf_sorted["IN"], color="#2980B9",
             edgecolor=ACCENT, label="−IN (follows)")
axes[1].axvline(0, color=ACCENT, lw=1)
axes[1].legend(loc="lower right")
axes[1].set_title("Bidirectional flow decomposition")
axes[1].set_xlabel("λ$^{lead}_U$ intensity")

fig.suptitle("Figure 5 — Source vs receiver classification",
             fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "fig5_source_receiver")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — LSTM agent: ROC + predictions overlay
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 6: LSTM agent ROC + predictions")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

y_true = agent["actual"].values
for model, color in [("lstm", "#E74C3C"), ("xgb", "#27AE60"),
                      ("logit", "#2874A6")]:
    if model in agent.columns and agent[model].notna().sum() > 20:
        mask = ~agent[model].isna()
        p = agent.loc[mask, model].values
        y = y_true[mask.values]
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        axes[0].plot(fpr, tpr, color=color, lw=1.5,
                     label=f"{model.upper()}  (AUC={auc:.3f})")
axes[0].plot([0, 1], [0, 1], "--", color=ACCENT, lw=0.8)
axes[0].set_xlabel("False positive rate")
axes[0].set_ylabel("True positive rate")
axes[0].set_title("ROC — walk-forward OOS")
axes[0].legend(loc="lower right")
axes[0].set_aspect("equal"); axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)

# Predictions overlay
shade_crisis(axes[1], reg["regime"].reindex(agent.index))
for model, color in [("lstm", "#E74C3C"), ("xgb", "#27AE60")]:
    if model in agent.columns and agent[model].notna().sum() > 20:
        axes[1].plot(agent.index, agent[model], color=color, lw=1.2,
                     label=f"P({model.upper()})", alpha=0.9)
axes[1].axhline(0.5, color=ACCENT, ls="--", lw=0.8)
axes[1].set_ylim(0, 1); axes[1].set_ylabel("P(portfolio crisis, t+2)")
axes[1].set_title("Out-of-sample probabilities")
axes[1].legend(loc="upper left")

fig.suptitle("Figure 6 — LSTM portfolio agent (P4 arch.) — OOS performance",
             fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "fig6_lstm_agent")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 7 — WALK-FORWARD
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 7: Walk-forward validation")

fig, ax = plt.subplots(figsize=(10, 4.5))

# Compute rolling 24M AUC from agent_predictions.csv (date-indexed, has lstm/xgb/logit/actual)
roll_auc = {}
for m in ("lstm", "xgb", "logit"):
    if m not in agent.columns:
        continue
    df = agent[[m, "actual"]].dropna().sort_index()
    if df.empty:
        continue
    aucs = []
    idxs = []
    for i in range(24, len(df) + 1):
        sub = df.iloc[i - 24: i]
        if len(np.unique(sub["actual"])) < 2:
            continue
        aucs.append(roc_auc_score(sub["actual"], sub[m]))
        idxs.append(df.index[i - 1])
    if aucs:
        roll_auc[m] = pd.Series(aucs, index=idxs)

for m, col in [("lstm", "#E74C3C"), ("xgb", "#27AE60"),
                ("logit", "#2874A6")]:
    if m in roll_auc:
        ax.plot(roll_auc[m].index, roll_auc[m].values,
                color=col, lw=1.4, label=m.upper())

ax.axhline(0.5, color=ACCENT, ls="--", lw=0.8, label="chance")
ax.axhline(0.926, color="#F39C12", ls=":", lw=1.0,
           label="P4 ref (0.926)")
ax.set_ylim(0.3, 1.02); ax.set_ylabel("Rolling 24M AUC")
ax.set_title("Figure 7 — Walk-forward validation  (rolling 24-month AUC)",
             fontweight="bold")
ax.legend(ncol=4, loc="lower center")
save(fig, "fig7_walkforward")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 8 — CSRI TIMELINE
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 8: CSRI timeline")

fig, ax = plt.subplots(figsize=(10, 4.5))
shade_crisis(ax, reg["regime"].reindex(csri.index))
ax.plot(csri.index, csri["CSRI"], color=ACCENT, lw=1.5, label="CSRI")
ax.axhline(0, color="grey", lw=0.6)
ax.axhline(1, color="#C0392B", ls="--", lw=1.0, label="crisis threshold (+1σ)")
ax.axhline(-1, color="#27AE60", ls="--", lw=1.0, label="quiet threshold (−1σ)")
ax.fill_between(csri.index, -1, 1, color="#F4F6F6", alpha=0.4, zorder=0)

# Annotate Greek macro episodes
macro = [
    ("2010-05", "GR bailout I"),
    ("2015-07", "GR capital controls"),
    ("2020-03", "COVID-19"),
    ("2022-02", "Ukraine energy shock"),
]
for d, lbl in macro:
    t = pd.Timestamp(d)
    if csri.index.min() <= t <= csri.index.max():
        val = csri["CSRI"].get(t, np.nan)
        if not np.isnan(val):
            ax.annotate(lbl, xy=(t, val), xytext=(t, val + 0.6),
                        fontsize=8, ha="center",
                        arrowprops=dict(arrowstyle="->", lw=0.6,
                                         color=ACCENT))

ax.set_ylabel("CSRI  (rolling 60M z-score)")
ax.set_title("Figure 8 — Construction Systemic Risk Index (CSRI) timeline",
             fontweight="bold")
ax.legend(loc="upper left", ncol=3)
save(fig, "fig8_csri_timeline")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("DONE — 8 publication figures saved")
print("=" * 64)
print(f"  Directory: {FIG_DIR}")
for i in range(1, 9):
    for ext in ("pdf", "png"):
        pass
print("\nPipeline complete. Paper 5 scripts 01-08 ready.")
print("=" * 64)
