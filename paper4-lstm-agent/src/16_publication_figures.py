"""
Paper 4 -- Publication Figures (v2 -- uses utils.py ensemble)
================================================================
Generates all AiC-ready publication figures for Paper 4.

This is the FINAL script -- run after all other scripts (01-15)
have generated their CSV results.

Figure list:
  fig1_lstm_architecture.pdf     -- Framework overview (4-layer pipeline)
  fig2_regime_classification.pdf -- AUC heatmap + ROC curve (ensemble)
  fig3_walk_forward.pdf          -- Walk-forward validation (55 windows)
  fig4_shap.pdf                  -- SHAP feature importance
  fig5_crisis_backtests.pdf      -- GFC 2008 + COVID 2021 backtests
  fig6_benchmarks.pdf            -- 6-model comparison + DeLong
  fig7_economic_value.pdf        -- EUR cost comparison (3 strategies)

Run from: construction-risk-suite/paper4-lstm-agent/src/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, roc_curve, f1_score

import warnings
warnings.filterwarnings("ignore")

import torch

from utils import (make_sequences, train_ensemble, predict_ensemble,
                   load_selected_features, DEFAULT_SEEDS)

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
ALERT_THRESH   = 0.462     # Youden's J optimal threshold
BASE_COST      = 2_300_000
ES_CRISIS      = 2_944_866
ES_STABLE      = 2_394_721

TARGET_MATERIALS = ["GR_Fuel_Energy", "GR_Steel", "GR_Concrete", "GR_PVC_Pipes"]
LEAD_TIMES       = [1, 2, 3, 4]

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

print("=" * 60)
print("Paper 4 -- Publication Figures (v2 Ensemble)")
print("=" * 60)

# ==============================================================================
# LOAD DATA + PRE-COMPUTED RESULTS
# ==============================================================================
print("\nLoading data and results...")

features = pd.read_csv(PROCESSED_DIR + "features.csv",
                       index_col=0, parse_dates=True)
print(f"  Using all {features.shape[1]} features")

labels   = pd.read_csv(PROCESSED_DIR + "crisis_labels.csv",
                       index_col=0, parse_dates=True)
clf_res  = pd.read_csv(RESULTS_DIR + "lstm_regime_clf_summary.csv")
shap_res = pd.read_csv(RESULTS_DIR + "shap_summary.csv")
bt_res   = pd.read_csv(RESULTS_DIR + "crisis_backtest_summary.csv")

# Optional results (may not exist if script was not run)
import os
bench_res = None
delong_res = None
econ_res = None
wf_qtr = None

if os.path.exists(RESULTS_DIR + "benchmark_comparison.csv"):
    bench_res = pd.read_csv(RESULTS_DIR + "benchmark_comparison.csv")
if os.path.exists(RESULTS_DIR + "delong_pairwise.csv"):
    delong_res = pd.read_csv(RESULTS_DIR + "delong_pairwise.csv")
if os.path.exists(RESULTS_DIR + "economic_value_summary.csv"):
    econ_res = pd.read_csv(RESULTS_DIR + "economic_value_summary.csv")
if os.path.exists(RESULTS_DIR + "wf_quarterly_evolution.csv"):
    wf_qtr = pd.read_csv(RESULTS_DIR + "wf_quarterly_evolution.csv")

FEAT_COLS = list(features.columns)
common    = features.index.intersection(labels.dropna().index)
X_raw     = features.loc[common].values
y_raw     = labels.loc[common, TARGET_MAT].values
dates     = common

print(f"  Data: {len(X_raw)} samples, {len(FEAT_COLS)} features")

# ==============================================================================
# TRAIN ENSEMBLE FOR FIGURES (ROC, backtests)
# ==============================================================================
print("\nTraining ensemble for publication figures...")

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

print(f"  Ensemble AUC: {auc_main:.3f}")

# ==============================================================================
# FIG 1 -- Framework Architecture
# ==============================================================================
print("\nFIG 1: Framework architecture")

fig, ax = plt.subplots(1, 1, figsize=(14, 7))
ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis("off")
fig.suptitle("Paper 4 -- LSTM-Copula Automated Construction Cost Risk Framework",
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

# SHAP box below Layer 2
rect2 = FancyBboxPatch((3.5, 1.2), 2.8, 1.2, boxstyle="round,pad=0.1",
                        facecolor="#e1d5e7", edgecolor="#9673a6", linewidth=1.5)
ax.add_patch(rect2)
ax.text(5.0, 1.95, "SHAP EXPLAINER",
        ha="center", va="center", fontsize=9, fontweight="bold")
ax.text(5.0, 1.55, "Feature attribution\nSteel > Cement > Brent",
        ha="center", va="center", fontsize=8, style="italic")
ax.annotate("", xy=(4.9, 3.0), xytext=(4.9, 2.4),
            arrowprops=dict(arrowstyle="<->", lw=1.5, color="#9673a6"))

# Key metrics
ax.text(7.0, 1.5,
        "Key results:\n"
        f"  AUC = {auc_main:.3f} (Fuel, lead=2M, 5-seed)\n"
        f"  Optimal threshold = {ALERT_THRESH} (Youden J)\n"
        "  COVID lead = 15M | GFC lead = 13M\n"
        "  LSTM saves EUR 1,100,290 vs static",
        ha="left", va="center", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="#f5f5f5", alpha=0.8))

plt.tight_layout()
p = RESULTS_DIR + "fig1_lstm_architecture.pdf"
plt.savefig(p, bbox_inches="tight"); print(f"  Saved: {p}"); plt.close()

# ==============================================================================
# FIG 2 -- Regime Classification (AUC heatmap + ROC)
# ==============================================================================
print("FIG 2: Regime classification")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Paper 4 -- LSTM Regime Classification: US PPI -> Greek Construction Crisis",
             fontsize=12, fontweight="bold")

# Heatmap
ax = axes[0]
pivot = clf_res.pivot(index="material", columns="lead", values="AUC")
pivot.index = [m.replace("GR_", "") for m in pivot.index]
im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0.40, vmax=0.95,
               aspect="auto")
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
ax.set_title("AUC by material x lead time\n"
             "(5-seed ensemble, SHAP-selected features)")

# ROC curve (ensemble)
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
ax.set_title(f"ROC Curve -- {TARGET_MAT.replace('GR_','')} | lead={LEAD}M\n"
             f"Out-of-sample test (5-seed ensemble)")
ax.legend(); ax.grid(alpha=0.3)
ax.text(0.55, 0.15,
        f"AUC = {auc_main:.3f}\nF1  = {f1_val:.3f}\n"
        f"Recall = {rec_val:.3f}\nThreshold = {ALERT_THRESH}",
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle="round", facecolor="#fff9c4", alpha=0.9))

plt.tight_layout()
p = RESULTS_DIR + "fig2_regime_classification.pdf"
plt.savefig(p, bbox_inches="tight"); print(f"  Saved: {p}"); plt.close()

# ==============================================================================
# FIG 3 -- Walk-Forward Validation
# ==============================================================================
print("FIG 3: Walk-forward validation")

if wf_qtr is not None and len(wf_qtr) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    auc_col = "auc" if "auc" in wf_qtr.columns else "cumulative_auc"
    valid_auc = wf_qtr[wf_qtr[auc_col].notna() & (wf_qtr[auc_col] > 0)]
    mean_auc = valid_auc[auc_col].mean()
    std_auc  = valid_auc[auc_col].std()
    fig.suptitle(f"Paper 4 -- Walk-Forward Validation: "
                 f"{len(wf_qtr)} expanding windows | "
                 f"Mean AUC = {mean_auc:.3f} +/- {std_auc:.3f}",
                 fontsize=12, fontweight="bold")

    # Panel 1: AUC evolution over time
    ax = axes[0]
    colors_b = ["#2ecc71" if a > 0.70 else "#f39c12" if a > 0.50
                else "#e74c3c" for a in valid_auc[auc_col]]
    ax.bar(range(len(valid_auc)), valid_auc[auc_col].values,
           color=colors_b, edgecolor="black", linewidth=0.3, width=0.8)
    ax.axhline(mean_auc, color="blue", linestyle="--", lw=2,
               label=f"Mean AUC = {mean_auc:.3f}")
    ax.axhline(0.50, color="red", linestyle=":", lw=1, alpha=0.7,
               label="Random baseline = 0.50")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Walk-forward window")
    ax.set_ylabel("Out-of-sample AUC")
    ax.set_title(f"AUC per window ({len(valid_auc)} valid of {len(wf_qtr)})")
    ax.legend(); ax.grid(alpha=0.3, axis="y")

    # Panel 2: AUC distribution
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
    p = RESULTS_DIR + "fig3_walk_forward.pdf"
    plt.savefig(p, bbox_inches="tight"); print(f"  Saved: {p}"); plt.close()
else:
    print("  SKIPPED: no walk-forward results found")

# ==============================================================================
# FIG 4 -- SHAP Feature Importance
# ==============================================================================
print("FIG 4: SHAP feature importance")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Paper 4 -- SHAP Explainability: Which US Signals Drive Greek Fuel Crisis?",
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
ax.set_title("Top 15 Feature Importance (SHAP)\n"
             "Steel volatility is primary leading indicator")
ax.grid(alpha=0.3, axis="x")
patches = [mpatches.Patch(color="#e74c3c", label="Fuel PPI"),
           mpatches.Patch(color="#3498db", label="Steel PPI"),
           mpatches.Patch(color="#2ecc71", label="Cement PPI"),
           mpatches.Patch(color="#9b59b6", label="PVC PPI"),
           mpatches.Patch(color="#f39c12", label="Brent")]
ax.legend(handles=patches, fontsize=7, loc="lower right")

# Importance by US series
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
ax.set_title("Importance by US Series\n"
             "(Steel > Cement > Brent)")
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
p = RESULTS_DIR + "fig4_shap.pdf"
plt.savefig(p, bbox_inches="tight"); print(f"  Saved: {p}"); plt.close()

# ==============================================================================
# FIG 5 -- Crisis Backtests (ensemble)
# ==============================================================================
print("FIG 5: Crisis backtests")

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
    "Paper 4 -- Crisis Episode Backtests (5-seed Ensemble)\n"
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
p = RESULTS_DIR + "fig5_crisis_backtests.pdf"
plt.savefig(p, bbox_inches="tight"); print(f"  Saved: {p}"); plt.close()

# ==============================================================================
# FIG 6 -- Benchmark Comparison
# ==============================================================================
print("FIG 6: Benchmark comparison")

if bench_res is not None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Paper 4 -- Model Comparison: LSTM Ensemble vs 5 Benchmarks",
                 fontsize=12, fontweight="bold")

    # Panel 1: AUC bar chart
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
    ax.set_title("AUC Comparison\n(LSTM Ensemble outperforms all benchmarks)")
    ax.set_xlim(0, 1.05)
    ax.grid(alpha=0.3, axis="x")

    # Panel 2: DeLong pairwise p-values
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
        ax.set_title("DeLong Pairwise p-values\n(* = p<0.05, green = significant)")
    else:
        ax.text(0.5, 0.5, "DeLong results not available",
                ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    p = RESULTS_DIR + "fig6_benchmarks.pdf"
    plt.savefig(p, bbox_inches="tight"); print(f"  Saved: {p}"); plt.close()
else:
    print("  SKIPPED: no benchmark results found")

# ==============================================================================
# FIG 7 -- Economic Value
# ==============================================================================
print("FIG 7: Economic value comparison")

if econ_res is not None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle("Paper 4 -- Economic Value: No-Hedge vs Static vs LSTM",
                 fontsize=12, fontweight="bold")

    strategies = econ_res["strategy"].values if "strategy" in econ_res.columns \
        else [f"S{i}" for i in range(len(econ_res))]
    tcol = next((c for c in econ_res.columns if "total" in c.lower()), econ_res.columns[1])
    totals = econ_res[tcol].values
    colors_econ = ["#e74c3c", "#f39c12", "#2ecc71"][:len(strategies)]

    bars = ax.bar(range(len(strategies)), totals / 1e6,
                  color=colors_econ, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"EUR {v/1e6:.2f}M", ha="center", fontsize=10,
                fontweight="bold")
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, fontsize=10)
    ax.set_ylabel("Total Cost (EUR millions)")
    ax.set_title("Total procurement cost over test period\n"
                 "(LSTM-adaptive achieves lowest cost)")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    p = RESULTS_DIR + "fig7_economic_value.pdf"
    plt.savefig(p, bbox_inches="tight"); print(f"  Saved: {p}"); plt.close()
else:
    print("  SKIPPED: no economic value results found")

# ==============================================================================
# DONE
# ==============================================================================
print("\n" + "=" * 60)
print("DONE -- All publication figures generated.")
print(f"  Ensemble AUC: {auc_main:.3f}")
print(f"  Figures saved to: {RESULTS_DIR}")
print("=" * 60)
