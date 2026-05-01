"""
Paper 4 -- Publication Figures (canonical AiC numbering)
================================================================
Generates ALL publication figures, named in the order they appear
in the manuscript (fig1 through fig10). This is the FINAL script --
run after all other scripts (01-15) have produced their CSV results.

Manuscript figure order (Section -> filename):
  Sec 1    fig1_lstm_architecture.pdf       Framework overview
  Sec 3.8  fig2_decision_logic.pdf          Decision flowchart (P_hat zones)
  Sec 4.1  fig3_regime_classification.pdf   AUC heatmap + ROC
  Sec 4.1  fig4_benchmarks.pdf              6-model comparison + DeLong
  Sec 4.1  fig5_walk_forward.pdf            Walk-forward AUC distribution
  Sec 4.2  fig6_shap.pdf                    SHAP global feature importance
  Sec 4.2  fig7_temporal_shap.pdf           Quarterly SHAP + amplification
  Sec 4.3  fig8_crisis_backtests.pdf        GFC + COVID episode backtests
  Sec 4.3  fig9_decision_rules.pdf          Adaptive rules timeline (R1-R8)
  Sec 4.3  fig10_contingency_savings.pdf    EUR savings simulation

Run from: construction-risk-suite/paper4-lstm-agent/src/
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, roc_curve, f1_score

import torch

from utils import (make_sequences, train_ensemble, predict_ensemble,
                   DEFAULT_SEEDS)

warnings.filterwarnings("ignore")

# ==============================================================================
# PARAMETERS
# ==============================================================================
PROCESSED_DIR  = "../data/processed/"
RESULTS_DIR    = "../results/"
SEED           = 42
LOOKBACK       = 6
LEAD           = 2
TARGET_MAT     = "GR_Fuel_Energy"
EPOCHS         = 150
BATCH_SIZE     = 16
HIDDEN_SIZE    = 64
N_LAYERS       = 2
DROPOUT        = 0.3
LR             = 5e-4
PATIENCE       = 20
TRAIN_RATIO    = 0.75

# Decision-rule thresholds (Section 3.8 of manuscript)
P_LOW          = 0.30
P_MEDIUM       = 0.50
P_HIGH         = 0.70
ALERT_THRESH   = 0.875            # Youden-optimal threshold
CONTINGENCY_STABLE = 94_721
CONTINGENCY_CRISIS = 644_866
BASE_COST          = 2_300_000

ENSEMBLE_SEEDS = DEFAULT_SEEDS

# AiC journal style
plt.rcParams.update({
    "font.family"    : "serif",
    "font.size"      : 10,
    "axes.titlesize" : 11,
    "axes.labelsize" : 10,
    "legend.fontsize": 8,
    "figure.dpi"     : 150,
})

np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 64)
print("Paper 4 -- Publication Figures (canonical AiC numbering)")
print("=" * 64)

# ==============================================================================
# LOAD DATA + PRE-COMPUTED RESULTS
# ==============================================================================
print("\nLoading data and results...")

features = pd.read_csv(PROCESSED_DIR + "features.csv",
                       index_col=0, parse_dates=True)
labels   = pd.read_csv(PROCESSED_DIR + "crisis_labels.csv",
                       index_col=0, parse_dates=True)
clf_res  = pd.read_csv(RESULTS_DIR + "lstm_regime_clf_summary.csv")
shap_res = pd.read_csv(RESULTS_DIR + "shap_summary.csv")

# Optional CSVs (skip elegantly if missing)
def _opt(name):
    p = RESULTS_DIR + name
    return pd.read_csv(p) if os.path.exists(p) else None

bench_res     = _opt("benchmark_comparison.csv")
delong_res    = _opt("delong_pairwise.csv")
econ_res      = _opt("economic_value_summary.csv")
econ_sim      = _opt("economic_value_simulation.csv")
wf_qtr        = _opt("wf_quarterly_evolution.csv")
temporal_evol = _opt("temporal_shap_evolution.csv")
temporal_cmp  = _opt("temporal_shap_crisis_vs_stable.csv")
contingency_t = _opt("contingency_timeline.csv")

FEAT_COLS = list(features.columns)
common    = features.index.intersection(labels.dropna().index)
X_raw     = features.loc[common].values
y_raw     = labels.loc[common, TARGET_MAT].values.astype(int)
dates     = common

print(f"  Data: {len(X_raw)} obs, {len(FEAT_COLS)} features")

# ==============================================================================
# TRAIN ENSEMBLE FOR FIG3 (ROC) AND FIG8 (BACKTESTS)
# ==============================================================================
print("\nTraining main ensemble (used in fig3, fig8) ...")

scaler = MinMaxScaler()
n_raw_tr = int(len(X_raw) * TRAIN_RATIO)
scaler.fit(X_raw[:n_raw_tr])
X_sc = scaler.transform(X_raw)

X_seq, y_seq, d_seq = make_sequences(X_sc, y_raw, LOOKBACK, LEAD, dates=dates)
n_tr = int(len(X_seq) * TRAIN_RATIO)
X_tr, X_te = X_seq[:n_tr], X_seq[n_tr:]
y_tr, y_te = y_seq[:n_tr], y_seq[n_tr:]
d_te       = d_seq[n_tr:]

models = train_ensemble(X_tr, y_tr, seeds=ENSEMBLE_SEEDS,
                        epochs=EPOCHS, batch_size=BATCH_SIZE,
                        hidden_size=HIDDEN_SIZE, n_layers=N_LAYERS,
                        dropout=DROPOUT, lr=LR, patience=PATIENCE,
                        verbose=False)
probs_te = predict_ensemble(models, X_te)
auc_main = roc_auc_score(y_te, probs_te) if y_te.sum() > 0 else np.nan

print(f"  Ensemble OOS AUC: {auc_main:.3f}")


# ==============================================================================
# FIG 1 -- Framework Architecture
# ==============================================================================
print("\nFIG 1: Framework architecture")

fig, ax = plt.subplots(1, 1, figsize=(14, 7))
ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis("off")
fig.suptitle("Automated LSTM-Copula Framework for Construction Cost Risk",
             fontsize=13, fontweight="bold", y=0.98)

boxes = [
    (0.3, 3.0, 2.8, 1.6, "#d5e8d4", "#82b366",
     "LAYER 1\nDATA INGESTION",
     "FRED API (monthly)\nUS PPI: Fuel, Steel,\nCement, PVC, Brent"),
    (3.5, 3.0, 2.8, 1.6, "#dae8fc", "#6c8ebf",
     "LAYER 2\nLSTM AGENT",
     "5-seed ensemble\nP(crisis) in [0,1]\nLead = 2 months"),
    (6.7, 3.0, 2.8, 1.6, "#ffe6cc", "#d6b656",
     "LAYER 3\nVINE COPULA ENGINE",
     "Gumbel ES(99%)\nPhase decomposition\nBootstrap CIs"),
    (9.9, 3.0, 2.8, 1.6, "#f8cecc", "#b85450",
     "LAYER 4\nDECISION OUTPUT",
     "Rules R1-R8\nEUR-denominated\nProcurement triggers"),
]
for x, y, w, h, fc, ec, title, body in boxes:
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                           facecolor=fc, edgecolor=ec, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h-0.35, title, ha="center", va="top",
            fontsize=9, fontweight="bold")
    ax.text(x+w/2, y+h/2-0.2, body, ha="center", va="center",
            fontsize=8, style="italic")

for x_start in [3.1, 6.3, 9.5]:
    ax.annotate("", xy=(x_start+0.4, 3.8),
                xytext=(x_start, 3.8),
                arrowprops=dict(arrowstyle="->", lw=2, color="#555555"))

# SHAP feedback box
rect2 = FancyBboxPatch((3.5, 1.2), 2.8, 1.2, boxstyle="round,pad=0.1",
                        facecolor="#e1d5e7", edgecolor="#9673a6", linewidth=1.5)
ax.add_patch(rect2)
ax.text(5.0, 1.95, "SHAP EXPLAINER",
        ha="center", va="center", fontsize=9, fontweight="bold")
ax.text(5.0, 1.55, "Feature attribution\n(audit + Rule R7)",
        ha="center", va="center", fontsize=8, style="italic")
ax.annotate("", xy=(4.9, 3.0), xytext=(4.9, 2.4),
            arrowprops=dict(arrowstyle="<->", lw=1.5, color="#9673a6"))

ax.text(7.0, 1.5,
        "Key results:\n"
        f"  AUC = {auc_main:.3f} (Fuel, lead=2M, 5-seed)\n"
        f"  Optimal threshold = {ALERT_THRESH} (Youden J)\n"
        "  COVID lead = 16M | GFC lead = 13M\n"
        "  LSTM saves EUR 4.0M vs static (72M test)",
        ha="left", va="center", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="#f5f5f5", alpha=0.8))

plt.tight_layout()
p = RESULTS_DIR + "fig1_lstm_architecture.pdf"
plt.savefig(p, bbox_inches="tight"); print(f"  Saved: {p}"); plt.close()


# ==============================================================================
# FIG 2 -- Decision Logic Flowchart  (NEW)
# ==============================================================================
print("FIG 2: Decision-logic flowchart")

fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.set_xlim(0, 14); ax.set_ylim(0, 9); ax.axis("off")
fig.suptitle("Decision Logic: From LSTM Probability to Procurement Action",
             fontsize=13, fontweight="bold", y=0.98)

# Top: P_hat input
top_box = FancyBboxPatch((5.0, 7.4), 4.0, 1.0, boxstyle="round,pad=0.12",
                         facecolor="#dae8fc", edgecolor="#1565C0", linewidth=2)
ax.add_patch(top_box)
ax.text(7.0, 7.9, r"LSTM Ensemble Probability $\hat{p} \in [0,1]$",
        ha="center", va="center", fontsize=11, fontweight="bold")

# Three decision zones
zones = [
    (0.3, 4.0, 4.0, 2.6, "#d5e8d4", "#2e7d32",
     "STABLE ZONE",
     r"$\hat{p} < 0.30$",
     "Standard contingency\nEUR 94,721 (R1)\n"
     "Steel-PPI swap if\n"
     r"$\rho > 0.40$ (R5)"),
    (4.9, 4.0, 4.2, 2.6, "#fff2cc", "#bf8f00",
     "ELEVATED ZONE",
     r"$0.30 \leq \hat{p} < 0.875$",
     "Adaptive contingency (R8)\n"
     "Suppress hedging (R5)\n"
     "SHAP-prioritised\n"
     "monitoring (R7)"),
    (9.7, 4.0, 4.0, 2.6, "#f8cecc", "#b71c1c",
     "CRISIS ZONE",
     r"$\hat{p} \geq 0.875$  (Youden J = 0.771)",
     "Crisis ES model (R6b)\n"
     "Max contingency\n"
     "EUR 644,866\n"
     "Pre-purchase trigger"),
]
for x, y, w, h, fc, ec, title, cond, action in zones:
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                          facecolor=fc, edgecolor=ec, linewidth=1.8)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h-0.35, title, ha="center", va="top",
            fontsize=10, fontweight="bold", color=ec)
    ax.text(x+w/2, y+h-0.85, cond, ha="center", va="top",
            fontsize=9, style="italic")
    ax.text(x+w/2, y+h/2-0.55, action, ha="center", va="center",
            fontsize=8.5)

# Arrows from input to zones
for cx in [2.3, 7.0, 11.7]:
    arr = FancyArrowPatch((7.0, 7.4), (cx, 6.65),
                          arrowstyle="->", mutation_scale=15,
                          lw=1.5, color="#555555")
    ax.add_patch(arr)

# Bottom: outputs
out_box = FancyBboxPatch((1.0, 1.5), 12.0, 1.6, boxstyle="round,pad=0.12",
                         facecolor="#f5f5f5", edgecolor="#424242", linewidth=1.5)
ax.add_patch(out_box)
ax.text(7.0, 2.65, "AUTOMATED OUTPUT (no human intervention)",
        ha="center", va="center", fontsize=10, fontweight="bold")
ax.text(7.0, 2.05,
        "Updates project DB with EUR contingency  |  "
        "Dispatches procurement-rule trigger  |  "
        "Logs SHAP attribution for audit",
        ha="center", va="center", fontsize=9, style="italic")

# Cycle-time annotation
ax.text(7.0, 0.7,
        r"End-to-end cycle: $\leq 15$ minutes from FRED publication to action",
        ha="center", va="center", fontsize=9.5,
        bbox=dict(boxstyle="round", facecolor="#fff9c4",
                  edgecolor="#f57f17", alpha=0.9))

# Arrows zones -> output
for cx in [2.3, 7.0, 11.7]:
    arr = FancyArrowPatch((cx, 4.0), (cx, 3.1),
                          arrowstyle="->", mutation_scale=12,
                          lw=1.2, color="#555555")
    ax.add_patch(arr)

plt.tight_layout()
p = RESULTS_DIR + "fig2_decision_logic.pdf"
plt.savefig(p, bbox_inches="tight"); print(f"  Saved: {p}"); plt.close()


# ==============================================================================
# FIG 3 -- Regime Classification  (AUC heatmap + ROC)
# ==============================================================================
print("FIG 3: Regime classification")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("LSTM Regime Classification: US PPI -> Greek Construction Crisis",
             fontsize=12, fontweight="bold")

# Heatmap
ax = axes[0]
pivot = clf_res.pivot(index="material", columns="lead", values="AUC")
pivot.index = [m.replace("GR_", "") for m in pivot.index]
im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0.40, vmax=0.95, aspect="auto")
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels([f"{l}M" for l in pivot.columns])
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index)
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        v = pivot.values[i, j]
        ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                fontsize=11, fontweight="bold",
                color="white" if v < 0.55 else "black")
plt.colorbar(im, ax=ax, label="AUC")
ax.set_xlabel("Lead time (months)")
ax.set_ylabel("Greek material")
ax.set_title("AUC by material x lead time\n(5-seed ensemble)")

# ROC
ax = axes[1]
if y_te.sum() > 0:
    fpr, tpr, _ = roc_curve(y_te, probs_te)
    ax.plot(fpr, tpr, color="#e74c3c", lw=2.5,
            label=f"LSTM Ensemble (AUC = {auc_main:.3f})")
    ax.fill_between(fpr, tpr, alpha=0.12, color="#e74c3c")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
alerts_te = (probs_te > ALERT_THRESH).astype(int)
f1_val = f1_score(y_te, alerts_te, zero_division=0)
rec_val = (alerts_te[y_te == 1] == 1).mean() if y_te.sum() > 0 else 0
ax.set_title(f"ROC Curve -- {TARGET_MAT.replace('GR_','')} | lead={LEAD}M")
ax.legend(); ax.grid(alpha=0.3)
ax.text(0.55, 0.15,
        f"AUC = {auc_main:.3f}\nF1  = {f1_val:.3f}\n"
        f"Recall = {rec_val:.3f}\nThreshold = {ALERT_THRESH}",
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle="round", facecolor="#fff9c4", alpha=0.9))

plt.tight_layout()
p = RESULTS_DIR + "fig3_regime_classification.pdf"
plt.savefig(p, bbox_inches="tight"); print(f"  Saved: {p}"); plt.close()


# ==============================================================================
# FIG 4 -- Benchmark Comparison
# ==============================================================================
print("FIG 4: Benchmark comparison")

if bench_res is not None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Model Comparison: LSTM Ensemble vs 5 Benchmarks",
                 fontsize=12, fontweight="bold")

    ax = axes[0]
    bench_sorted = bench_res.sort_values("AUC", ascending=True)
    mcol = "Model" if "Model" in bench_sorted.columns else "model"
    colors_bench = ["#e74c3c" if "LSTM" in m else "#3498db"
                    for m in bench_sorted[mcol]]
    bars = ax.barh(range(len(bench_sorted)), bench_sorted["AUC"],
                   color=colors_bench, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(bench_sorted)))
    ax.set_yticklabels(bench_sorted[mcol], fontsize=9)
    for bar, v in zip(bars, bench_sorted["AUC"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=10, fontweight="bold")
    ax.axvline(0.5, color="gray", linestyle=":", lw=1)
    ax.set_xlabel("AUC")
    ax.set_title("AUC Comparison\n(LSTM Ensemble outperforms benchmarks)")
    ax.set_xlim(0, 1.05)
    ax.grid(alpha=0.3, axis="x")

    ax = axes[1]
    if delong_res is not None and len(delong_res) > 0:
        ma = "Model_A" if "Model_A" in delong_res.columns else "model_A"
        mb = "Model_B" if "Model_B" in delong_res.columns else "model_B"
        pv = "p_value"
        models_list = sorted(set(delong_res[ma].tolist() +
                                 delong_res[mb].tolist()))
        n_models = len(models_list)
        pval_matrix = np.ones((n_models, n_models))
        for _, row in delong_res.iterrows():
            i = models_list.index(row[ma])
            j = models_list.index(row[mb])
            pval_matrix[i, j] = row[pv]
            pval_matrix[j, i] = row[pv]

        im = ax.imshow(pval_matrix, cmap="RdYlGn_r", vmin=0, vmax=0.10,
                       aspect="auto")
        ax.set_xticks(range(n_models))
        ax.set_xticklabels([m[:10] for m in models_list],
                           rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n_models))
        ax.set_yticklabels([m[:10] for m in models_list], fontsize=8)
        for i in range(n_models):
            for j in range(n_models):
                if i != j:
                    v = pval_matrix[i, j]
                    sig = "*" if v < 0.05 else ""
                    ax.text(j, i, f"{v:.3f}{sig}",
                            ha="center", va="center", fontsize=7)
        plt.colorbar(im, ax=ax, label="p-value")
        ax.set_title("DeLong Pairwise p-values\n(* = p<0.05)")
    else:
        ax.text(0.5, 0.5, "DeLong results not available",
                ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    p = RESULTS_DIR + "fig4_benchmarks.pdf"
    plt.savefig(p, bbox_inches="tight"); print(f"  Saved: {p}"); plt.close()
else:
    print("  SKIPPED: no benchmark results found")


# ==============================================================================
# FIG 5 -- Walk-Forward Validation
# ==============================================================================
print("FIG 5: Walk-forward validation")

if wf_qtr is not None and len(wf_qtr) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    auc_col = "auc" if "auc" in wf_qtr.columns else "cumulative_auc"
    valid_auc = wf_qtr[wf_qtr[auc_col].notna() & (wf_qtr[auc_col] > 0)]
    mean_auc = valid_auc[auc_col].mean()
    std_auc  = valid_auc[auc_col].std()
    fig.suptitle(f"Walk-Forward Validation: "
                 f"{len(wf_qtr)} expanding windows | "
                 f"Mean AUC = {mean_auc:.3f} +/- {std_auc:.3f}",
                 fontsize=12, fontweight="bold")

    ax = axes[0]
    colors_b = ["#2ecc71" if a > 0.70 else "#f39c12" if a > 0.50
                else "#e74c3c" for a in valid_auc[auc_col]]
    ax.bar(range(len(valid_auc)), valid_auc[auc_col].values,
           color=colors_b, edgecolor="black", linewidth=0.3, width=0.8)
    ax.axhline(mean_auc, color="blue", linestyle="--", lw=2,
               label=f"Mean AUC = {mean_auc:.3f}")
    ax.axhline(0.50, color="red", linestyle=":", lw=1, alpha=0.7,
               label="Random = 0.50")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Walk-forward window")
    ax.set_ylabel("Out-of-sample AUC")
    ax.set_title(f"AUC per window ({len(valid_auc)} valid of {len(wf_qtr)})")
    ax.legend(); ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    ax.hist(valid_auc[auc_col], bins=15, color="#3498db", edgecolor="black",
            alpha=0.8)
    ax.axvline(mean_auc, color="red", linestyle="--", lw=2,
               label=f"Mean = {mean_auc:.3f}")
    ax.axvline(0.50, color="gray", linestyle=":", lw=1.5,
               label="Random = 0.50")
    ax.set_xlabel("AUC")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of OOS AUC across windows")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    p = RESULTS_DIR + "fig5_walk_forward.pdf"
    plt.savefig(p, bbox_inches="tight"); print(f"  Saved: {p}"); plt.close()
else:
    print("  SKIPPED: no walk-forward results found")


# ==============================================================================
# FIG 6 -- SHAP Feature Importance (global)
# ==============================================================================
print("FIG 6: SHAP feature importance")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("SHAP Explainability: Which US Signals Drive Greek Fuel Crisis?",
             fontsize=12, fontweight="bold")

ax = axes[0]
top15 = shap_res.head(15)
feat_colors = []
for f in top15["feature"]:
    if "Fuel" in f:    feat_colors.append("#e74c3c")
    elif "Steel" in f: feat_colors.append("#3498db")
    elif "Cement" in f:feat_colors.append("#2ecc71")
    elif "PVC" in f:   feat_colors.append("#9b59b6")
    else:              feat_colors.append("#f39c12")

ax.barh(range(len(top15)), top15["mean_abs_shap"],
        color=feat_colors, edgecolor="black", linewidth=0.3)
ax.set_yticks(range(len(top15)))
ax.set_yticklabels([f.replace("US_", "").replace("_PPI", "")
                    for f in top15["feature"]], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("Top 15 features by mean |SHAP|")
ax.grid(alpha=0.3, axis="x")
patches = [mpatches.Patch(color="#e74c3c", label="Fuel PPI"),
           mpatches.Patch(color="#3498db", label="Steel PPI"),
           mpatches.Patch(color="#2ecc71", label="Cement PPI"),
           mpatches.Patch(color="#9b59b6", label="PVC PPI"),
           mpatches.Patch(color="#f39c12", label="Brent")]
ax.legend(handles=patches, fontsize=7, loc="lower right")

ax = axes[1]
def get_series(f):
    for s in ["Brent", "Steel_PPI", "Cement_PPI", "PVC_PPI", "Fuel_PPI"]:
        if s in f: return s.replace("_PPI", "")
    return f

shap_res_copy = shap_res.copy()
shap_res_copy["series"] = shap_res_copy["feature"].apply(get_series)
s_imp = shap_res_copy.groupby("series")["mean_abs_shap"].sum().sort_values(
    ascending=False)
s_colors = {"Steel": "#3498db", "Cement": "#2ecc71", "Brent": "#f39c12",
            "PVC": "#9b59b6", "Fuel": "#e74c3c"}
bars3 = ax.bar(range(len(s_imp)), s_imp.values,
               color=[s_colors.get(s, "#95a5a6") for s in s_imp.index],
               edgecolor="black", linewidth=0.5)
for bar, v in zip(bars3, s_imp.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + s_imp.values.max() * 0.01,
            f"{v:.4f}", ha="center", fontsize=10)
ax.set_xticks(range(len(s_imp)))
ax.set_xticklabels(s_imp.index, fontsize=11)
ax.set_ylabel("Total |SHAP|")
ax.set_title("Importance by US Series")
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
p = RESULTS_DIR + "fig6_shap.pdf"
plt.savefig(p, bbox_inches="tight"); print(f"  Saved: {p}"); plt.close()


# ==============================================================================
# FIG 7 -- Temporal SHAP (NEW: quarterly evolution + amplification)
# ==============================================================================
print("FIG 7: Temporal SHAP")

if temporal_evol is not None and temporal_cmp is not None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Temporal SHAP: Regime-Dependent Feature Importance",
                 fontsize=12, fontweight="bold")

    # Left: top-4 features over quarters (line plot)
    ax = axes[0]
    quarters = sorted(temporal_evol["quarter"].astype(str).unique())
    overall_top = (temporal_evol.groupby("feature")["mean_abs_shap"]
                                 .mean()
                                 .nlargest(4).index.tolist())
    colors_line = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    for i, feat in enumerate(overall_top):
        vals = []
        for q in quarters:
            sub = temporal_evol[(temporal_evol["quarter"].astype(str) == q) &
                                (temporal_evol["feature"] == feat)]
            vals.append(sub["mean_abs_shap"].values[0]
                        if len(sub) > 0 else 0)
        ax.plot(range(len(quarters)), vals, "o-",
                color=colors_line[i], lw=1.6, markersize=4,
                label=feat.replace("US_", "").replace("_PPI", ""))
    # Crisis shading
    if "crisis_rate" in temporal_evol.columns:
        cr = [temporal_evol[temporal_evol["quarter"].astype(str) == q]
              ["crisis_rate"].iloc[0] if (temporal_evol["quarter"].astype(str) == q).any()
              else 0 for q in quarters]
        for j, c in enumerate(cr):
            if c > 0.5:
                ax.axvspan(j - 0.5, j + 0.5, color="red", alpha=0.08)
    step = max(1, len(quarters) // 10)
    ax.set_xticks(range(0, len(quarters), step))
    ax.set_xticklabels([quarters[i] for i in range(0, len(quarters), step)],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean |SHAP|")
    ax.set_title("(a) Top-4 feature importance over time\n"
                 "(red shading = crisis quarters)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    # Right: amplification bar (crisis / stable)
    ax = axes[1]
    cmp = temporal_cmp.copy()
    if "ratio_crisis_stable" in cmp.columns:
        cmp = cmp.dropna(subset=["ratio_crisis_stable"])
        cmp["abs_log_ratio"] = np.abs(np.log(cmp["ratio_crisis_stable"]
                                             .replace(0, np.nan)))
        cmp = cmp.sort_values("abs_log_ratio", ascending=False).head(10)
        cmp = cmp.sort_values("ratio_crisis_stable", ascending=True)
        bar_colors = ["#e74c3c" if r > 1 else "#3498db"
                      for r in cmp["ratio_crisis_stable"]]
        ax.barh(range(len(cmp)), cmp["ratio_crisis_stable"],
                color=bar_colors, edgecolor="black", linewidth=0.4)
        ax.axvline(1.0, color="black", lw=1)
        ax.set_yticks(range(len(cmp)))
        ax.set_yticklabels([f.replace("US_", "").replace("_PPI", "")
                            for f in cmp["feature"]], fontsize=8)
        ax.set_xlabel("Crisis / Stable amplification")
        ax.set_title("(b) Regime amplification\n"
                     "red = amplified in crisis, blue = suppressed")
        ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    p = RESULTS_DIR + "fig7_temporal_shap.pdf"
    plt.savefig(p, bbox_inches="tight"); print(f"  Saved: {p}"); plt.close()
else:
    print("  SKIPPED: temporal SHAP CSVs missing (run 15_temporal_shap.py)")


# ==============================================================================
# FIG 8 -- Crisis Backtests (GFC + COVID, ensemble retrained per episode)
# ==============================================================================
print("FIG 8: Crisis backtests")

EPISODES = {
    "GFC 2008": {
        "train_end"  : "2006-12-01",
        "test_start" : "2007-01-01",
        "test_end"   : "2010-12-01",
        "crisis_peak": "2008-10-01",
        "color"      : "#3498db",
        "label"      : "Global Financial Crisis 2008-2009",
    },
    "COVID 2021": {
        "train_end"  : "2018-12-01",
        "test_start" : "2019-01-01",
        "test_end"   : "2023-12-01",
        "crisis_peak": "2021-06-01",
        "color"      : "#e74c3c",
        "label"      : "COVID-19 Commodity Shock 2021-2022",
    },
}

fig, axes = plt.subplots(2, 1, figsize=(16, 11))
fig.suptitle(
    "Crisis Episode Backtests (5-seed Ensemble, strict OOS)\n"
    "LSTM trained on pre-crisis data only -- early detection before Greek cost spike",
    fontsize=12, fontweight="bold"
)

for ax, (ep_name, ep) in zip(axes, EPISODES.items()):
    train_mask = dates <= ep["train_end"]
    test_mask  = (dates >= ep["test_start"]) & (dates <= ep["test_end"])
    X_tr_r = X_raw[train_mask]; y_tr_r = y_raw[train_mask]
    X_te_r = X_raw[test_mask];  y_te_r = y_raw[test_mask]
    d_te_r = dates[test_mask]

    sc2 = MinMaxScaler(); sc2.fit(X_tr_r)
    X_tr_s = sc2.transform(X_tr_r); X_te_s = sc2.transform(X_te_r)
    X_tr_q, y_tr_q, _ = make_sequences(
        X_tr_s, y_tr_r, LOOKBACK, LEAD, dates=dates[train_mask])
    X_te_q, y_te_q, d_q = make_sequences(
        X_te_s, y_te_r, LOOKBACK, LEAD, dates=d_te_r)

    if len(y_te_q) == 0:
        ax.text(0.5, 0.5, f"{ep_name}: no test sequences",
                ha="center", va="center", transform=ax.transAxes)
        continue

    ep_models = train_ensemble(X_tr_q, y_tr_q, seeds=ENSEMBLE_SEEDS,
                               epochs=EPOCHS, batch_size=BATCH_SIZE,
                               hidden_size=HIDDEN_SIZE, n_layers=N_LAYERS,
                               dropout=DROPOUT, lr=LR, patience=PATIENCE,
                               verbose=False)
    pr2 = predict_ensemble(ep_models, X_te_q)
    al2 = (pr2 > ALERT_THRESH).astype(int)
    dates_plot2 = pd.to_datetime([str(d) for d in d_q])
    peak_ts = pd.Timestamp(ep["crisis_peak"])
    early = [pd.Timestamp(str(d)) for d, a in zip(d_q, al2)
             if a == 1 and pd.Timestamp(str(d)) < peak_ts]
    first_alert = min(early) if early else None
    lead_m = ((peak_ts.year - first_alert.year) * 12 +
              peak_ts.month - first_alert.month) if first_alert else 0

    ax.fill_between(dates_plot2, 0, 1,
                    where=(y_te_q == 1), color="red", alpha=0.15,
                    label="Actual crisis period")
    ax.plot(dates_plot2, pr2, color=ep["color"], lw=2,
            label=f"LSTM P(crisis) -- train: 2000->{ep['train_end'][:7]}")
    ax.axhline(ALERT_THRESH, color="gray", linestyle="--", lw=1, alpha=0.7,
               label=f"Threshold ({ALERT_THRESH})")
    ax.axvline(peak_ts, color="darkred", linestyle="-.", lw=2,
               label=f"Crisis peak ({ep['crisis_peak'][:7]})")
    if first_alert:
        ax.axvline(first_alert, color="green", linestyle="-.", lw=2,
                   label=f"First alert ({first_alert.strftime('%Y-%m')})")
        ax.axvspan(first_alert, peak_ts, color="green", alpha=0.07)
        ax.annotate(f"Lead: {lead_m}M",
                    xy=(first_alert, 0.88), xytext=(first_alert, 0.97),
                    fontsize=12, fontweight="bold", color="green",
                    ha="center",
                    arrowprops=dict(arrowstyle="->", color="green", lw=1.5))
    auc_str = ""
    if y_te_q.sum() > 0 and y_te_q.sum() < len(y_te_q):
        auc_v = roc_auc_score(y_te_q, pr2)
        auc_str = f" | AUC={auc_v:.3f}"
    ax.set_title(f"{ep['label']}{auc_str}")
    ax.set_ylabel("P(crisis)"); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.grid(alpha=0.3)

plt.tight_layout()
p = RESULTS_DIR + "fig8_crisis_backtests.pdf"
plt.savefig(p, bbox_inches="tight"); print(f"  Saved: {p}"); plt.close()


# ==============================================================================
# FIG 9 -- Adaptive Decision Rules timeline (R1, R6b, R8)
# ==============================================================================
print("FIG 9: Adaptive decision rules")

if contingency_t is not None:
    ct = contingency_t.copy()
    ct["date"] = pd.to_datetime(ct["date"])
    ct = ct.sort_values("date").reset_index(drop=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle("Adaptive Decision Rules: Paper 3 (Static) vs Paper 4 (LSTM)",
                 fontsize=12, fontweight="bold")

    # Top panel: P(crisis) with regime thresholds
    ax = axes[0]
    if "actual_crisis" in ct.columns:
        ax.fill_between(ct["date"], 0, 1,
                        where=(ct["actual_crisis"] == 1),
                        color="red", alpha=0.15, label="Actual crisis")
    ax.plot(ct["date"], ct["P_crisis"], color="#e74c3c", lw=2,
            label=r"LSTM $\hat{p}$ (lead = 2M)")
    ax.axhline(P_LOW,    color="green",  linestyle=":", lw=1,
               label=f"P_LOW = {P_LOW}")
    ax.axhline(P_MEDIUM, color="orange", linestyle="--", lw=1.5,
               label=f"P_MEDIUM = {P_MEDIUM}")
    ax.axhline(P_HIGH,   color="red",    linestyle="-.", lw=1,
               label=f"P_HIGH = {P_HIGH}")
    ax.axhline(ALERT_THRESH, color="black", linestyle="-", lw=1,
               label=f"Optimal = {ALERT_THRESH} (Youden)")
    ax.set_ylabel(r"$\hat{p}$ (crisis probability)")
    ax.set_ylim(0, 1.05)
    ax.set_title("(a) LSTM probability with decision thresholds (Rule R6b)")
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.3)

    # Bottom panel: adaptive contingency vs static
    ax = axes[1]
    if "actual_crisis" in ct.columns:
        ax.fill_between(ct["date"], 0, 1,
                        where=(ct["actual_crisis"] == 1),
                        transform=ax.get_xaxis_transform(),
                        color="red", alpha=0.10)
    ax.plot(ct["date"], ct["contingency_EUR"], color="#3498db", lw=2,
            label="Paper 4: Adaptive contingency (R8)")
    ax.axhline(CONTINGENCY_CRISIS, color="red", linestyle="--", lw=1.5,
               label=f"Paper 3 crisis = EUR {CONTINGENCY_CRISIS:,}")
    ax.axhline(CONTINGENCY_STABLE, color="green", linestyle="--", lw=1.5,
               label=f"Paper 3 stable = EUR {CONTINGENCY_STABLE:,}")
    ax.fill_between(ct["date"], ct["contingency_EUR"], CONTINGENCY_CRISIS,
                    where=(ct["contingency_EUR"] < CONTINGENCY_CRISIS),
                    color="green", alpha=0.15, label="Saving vs static crisis")
    ax.set_ylabel("Contingency Reserve (EUR)")
    ax.set_title("(b) Adaptive contingency (R8): continuous EUR scaling")
    ax.legend(fontsize=8, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"EUR {x:,.0f}"))
    ax.grid(alpha=0.3)

    plt.tight_layout()
    p = RESULTS_DIR + "fig9_decision_rules.pdf"
    plt.savefig(p, bbox_inches="tight"); print(f"  Saved: {p}"); plt.close()
else:
    print("  SKIPPED: contingency_timeline.csv missing (run 12_decision_rules.py)")


# ==============================================================================
# FIG 10 -- Contingency Savings (economic value)
# ==============================================================================
print("FIG 10: Contingency savings")

if econ_res is not None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Economic Value: No-Hedge vs P3 Static vs P4 LSTM",
                 fontsize=12, fontweight="bold")

    # Left: total bars
    ax = axes[0]
    strategies = econ_res["strategy"].values if "strategy" in econ_res.columns \
        else [f"S{i}" for i in range(len(econ_res))]
    tcol = next((c for c in econ_res.columns if "total" in c.lower()),
                econ_res.columns[1])
    totals = econ_res[tcol].values
    colors_econ = ["#e74c3c", "#95a5a6", "#2980b9"][:len(strategies)]
    bars = ax.bar(range(len(strategies)), totals / 1e6,
                  color=colors_econ, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(totals)/1e6 * 0.01,
                f"EUR {v/1e6:.2f}M", ha="center", fontsize=10,
                fontweight="bold")
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, fontsize=10)
    ax.set_ylabel("Total Cost (EUR millions)")
    ax.set_title("(a) Total procurement cost over test period")
    ax.grid(alpha=0.3, axis="y")

    # Right: cumulative cost over time (if simulation available)
    ax = axes[1]
    if econ_sim is not None and "date" in econ_sim.columns:
        sim = econ_sim.copy()
        sim["date"] = pd.to_datetime(sim["date"])
        sim = sim.sort_values("date")
        if "actual_crisis" in sim.columns:
            ax.fill_between(sim["date"], 0, 1,
                            where=(sim["actual_crisis"] == 1),
                            transform=ax.get_xaxis_transform(),
                            color="red", alpha=0.10, label="Crisis period")
        for col, color, lbl in [
            ("cost_no_hedge", "#e74c3c", "No hedging"),
            ("cost_static",   "#95a5a6", "P3 Static"),
            ("cost_lstm",     "#2980b9", "P4 LSTM")
        ]:
            if col in sim.columns:
                ax.plot(sim["date"], sim[col].cumsum() / 1e6,
                        color=color, lw=2, label=lbl)
        ax.set_ylabel("Cumulative Cost (EUR M)")
        ax.set_xlabel("Date")
        ax.set_title("(b) Cumulative cost timeline")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(alpha=0.3)
    else:
        # Fall back to a horizontal saving annotation
        ax.text(0.5, 0.5, "Cumulative simulation not available\n"
                "(economic_value_simulation.csv missing)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    p = RESULTS_DIR + "fig10_contingency_savings.pdf"
    plt.savefig(p, bbox_inches="tight"); print(f"  Saved: {p}"); plt.close()
else:
    print("  SKIPPED: economic_value_summary.csv missing")


# ==============================================================================
# CLEAN UP LEGACY FILES (delete old confusing names so Overleaf is unambiguous)
# ==============================================================================
print("\nCleaning up legacy figure filenames ...")

LEGACY = [
    "fig2_regime_classification.pdf",   # -> renamed to fig3_*
    "fig3_walk_forward.pdf",            # -> renamed to fig5_*
    "fig4_shap.pdf",                    # -> renamed to fig6_*
    "fig5_crisis_backtests.pdf",        # -> renamed to fig8_*
    "fig6_benchmarks.pdf",              # -> renamed to fig4_*
    "fig6_decision_rules.pdf",          # -> renamed to fig9_*
    "fig7_economic_value.pdf",          # -> renamed to fig10_*
    "fig7_contingency_savings.pdf",     # old draft name, regenerated as fig10_*
    "fig_p4_temporal_shap.pdf",         # -> renamed to fig7_*
    "fig1b_decision_logic.pdf",         # -> renamed to fig2_*
    "fig8_temporal_shap.pdf",           # old draft name, regenerated as fig7_*
]
for fn in LEGACY:
    p = RESULTS_DIR + fn
    if os.path.exists(p):
        os.remove(p)
        print(f"  Removed legacy: {fn}")


# ==============================================================================
# DONE
# ==============================================================================
print("\n" + "=" * 64)
print("DONE -- Canonical AiC publication figures generated:")
print("=" * 64)
canonical = [
    "fig1_lstm_architecture.pdf",
    "fig2_decision_logic.pdf",
    "fig3_regime_classification.pdf",
    "fig4_benchmarks.pdf",
    "fig5_walk_forward.pdf",
    "fig6_shap.pdf",
    "fig7_temporal_shap.pdf",
    "fig8_crisis_backtests.pdf",
    "fig9_decision_rules.pdf",
    "fig10_contingency_savings.pdf",
]
for fn in canonical:
    p = RESULTS_DIR + fn
    status = "OK" if os.path.exists(p) else "MISSING"
    print(f"  [{status}] {fn}")
print(f"\n  Ensemble OOS AUC: {auc_main:.3f}")
print(f"  Figures saved to: {RESULTS_DIR}")
print("=" * 64)
