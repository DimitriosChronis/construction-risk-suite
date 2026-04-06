"""
Paper 4 -- Robustness Checks (v2 -- ensemble + feature selection)
===================================================================
Tests that results are NOT sensitive to parameter choices.

Key question: "Does AUC > 0.85 hold across different:
  - LOOKBACK windows (3M, 6M, 9M, 12M)?
  - CRISIS thresholds (P70, P75, P80, P85)?
  - LEAD times (1M, 2M, 3M, 4M)?
  - VOL windows (3M, 6M, 9M)?"

v2 changes (Phase 1F):
- Uses shared utils.py (ensemble, helpers)
- 5-seed ensemble per parameter combination
- Uses SHAP-selected features where applicable

Outputs:
  - results/robustness_summary.csv
  - results/fig_p4_robustness.pdf

Run from: construction-risk-suite/paper4-lstm-agent/src/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings("ignore")

import torch

from utils import (make_sequences, train_ensemble, predict_ensemble,
                   DEFAULT_SEEDS)

# ==============================================================================
# PARAMETERS
# ==============================================================================
PROCESSED_DIR = "../data/processed/"
RESULTS_DIR   = "../results/"
SEED          = 42
TARGET_MAT    = "GR_Fuel_Energy"
TRAIN_RATIO   = 0.75
EPOCHS        = 100   # reduced for speed across many experiments
BATCH_SIZE    = 16
HIDDEN_SIZE   = 64
N_LAYERS      = 2
DROPOUT       = 0.3
LR            = 5e-4
PATIENCE      = 15

# Ensemble
ENSEMBLE_SEEDS = DEFAULT_SEEDS

# Robustness grids
LOOKBACK_GRID   = [3, 6, 9, 12]
CRISIS_PCT_GRID = [0.70, 0.75, 0.80, 0.85]
LEAD_GRID       = [1, 2, 3, 4]
VOL_WIN_GRID    = [3, 6, 9]

BASE_LOOKBACK   = 6
BASE_CRISIS_PCT = 0.75
BASE_LEAD       = 2
BASE_VOL_WIN    = 6

np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 60)
print("Paper 4 -- Robustness Checks (Ensemble)")
print(f"Target: {TARGET_MAT} | {len(ENSEMBLE_SEEDS)}-seed ensemble")
print("=" * 60)

# ==============================================================================
# 1. LOAD RAW DATA
# ==============================================================================
raw = pd.read_csv(PROCESSED_DIR + "aligned_returns.csv",
                  index_col=0, parse_dates=True)
US_COLS = [c for c in raw.columns if c.startswith("US_")]

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
def build_features(raw, us_cols, vol_window):
    """Build US PPI features with given vol_window."""
    parts = []
    parts.append(raw[us_cols].rename(
        columns={c: f"{c}_ret" for c in us_cols}))
    for c in us_cols:
        parts.append(raw[c].rolling(3).std().rename(f"{c}_vol3"))
    for c in us_cols:
        parts.append(raw[c].rolling(vol_window).std().rename(
            f"{c}_vol{vol_window}"))
    for c in us_cols:
        parts.append(raw[c].rolling(3).sum().rename(f"{c}_mom3"))
    return pd.concat(parts, axis=1).dropna()


def build_labels(raw, target_mat, crisis_pct, vol_window):
    """Build crisis labels with given crisis_pct and vol_window."""
    vol = raw[target_mat].rolling(vol_window).std()
    thr = vol.quantile(crisis_pct)
    return (vol > thr).astype(int), thr


def run_experiment(raw, us_cols, target_mat, lookback, lead,
                   crisis_pct, vol_window):
    """Run one ensemble LSTM experiment with given parameters."""
    features = build_features(raw, us_cols, vol_window)
    labels, _ = build_labels(raw, target_mat, crisis_pct, vol_window)
    common = features.index.intersection(labels.dropna().index)
    X_raw  = features.loc[common].values
    y_raw  = labels.loc[common].values

    if len(X_raw) < lookback + lead + 20:
        return np.nan
    if y_raw.sum() < 5:
        return np.nan

    # Fit scaler on training portion only
    scaler = MinMaxScaler()
    n_raw_tr = int(len(X_raw) * TRAIN_RATIO)
    scaler.fit(X_raw[:n_raw_tr])
    X_sc = scaler.transform(X_raw)

    X_seq, y_seq = make_sequences(X_sc, y_raw, lookback, lead)

    if len(X_seq) < 20:
        return np.nan

    n_tr = int(len(X_seq) * TRAIN_RATIO)
    X_tr, X_te = X_seq[:n_tr], X_seq[n_tr:]
    y_tr, y_te = y_seq[:n_tr], y_seq[n_tr:]

    if y_te.sum() == 0 or y_te.sum() == len(y_te):
        return np.nan

    # Train ensemble
    models = train_ensemble(X_tr, y_tr, seeds=ENSEMBLE_SEEDS,
                            epochs=EPOCHS, batch_size=BATCH_SIZE,
                            hidden_size=HIDDEN_SIZE, n_layers=N_LAYERS,
                            dropout=DROPOUT, lr=LR, patience=PATIENCE)
    probs = predict_ensemble(models, X_te)

    if len(np.unique(y_te)) < 2:
        return np.nan
    return roc_auc_score(y_te, probs)


# ==============================================================================
# 3. ROBUSTNESS EXPERIMENTS
# ==============================================================================
all_results = []

# 3a: Vary LOOKBACK
print("\nExperiment A: Varying LOOKBACK window")
print("-" * 50)
for lb in LOOKBACK_GRID:
    print(f"  LOOKBACK={lb}M...", end=" ", flush=True)
    auc = run_experiment(raw, US_COLS, TARGET_MAT,
                         lb, BASE_LEAD, BASE_CRISIS_PCT, BASE_VOL_WIN)
    print(f"AUC = {auc:.3f}" if not np.isnan(auc) else "SKIP")
    all_results.append({
        "experiment": "LOOKBACK",
        "param_name": "lookback_months",
        "param_value": lb,
        "AUC": auc,
        "is_base": lb == BASE_LOOKBACK
    })

# 3b: Vary CRISIS_PCT
print("\nExperiment B: Varying CRISIS threshold (percentile)")
print("-" * 50)
for cp in CRISIS_PCT_GRID:
    print(f"  CRISIS_PCT=P{int(cp*100)}...", end=" ", flush=True)
    auc = run_experiment(raw, US_COLS, TARGET_MAT,
                         BASE_LOOKBACK, BASE_LEAD, cp, BASE_VOL_WIN)
    print(f"AUC = {auc:.3f}" if not np.isnan(auc) else "SKIP")
    all_results.append({
        "experiment": "CRISIS_PCT",
        "param_name": "crisis_percentile",
        "param_value": int(cp * 100),
        "AUC": auc,
        "is_base": cp == BASE_CRISIS_PCT
    })

# 3c: Vary LEAD
print("\nExperiment C: Varying LEAD time")
print("-" * 50)
for lead in LEAD_GRID:
    print(f"  LEAD={lead}M...", end=" ", flush=True)
    auc = run_experiment(raw, US_COLS, TARGET_MAT,
                         BASE_LOOKBACK, lead, BASE_CRISIS_PCT, BASE_VOL_WIN)
    print(f"AUC = {auc:.3f}" if not np.isnan(auc) else "SKIP")
    all_results.append({
        "experiment": "LEAD",
        "param_name": "lead_months",
        "param_value": lead,
        "AUC": auc,
        "is_base": lead == BASE_LEAD
    })

# 3d: Vary VOL_WINDOW
print("\nExperiment D: Varying VOL window (crisis definition)")
print("-" * 50)
for vw in VOL_WIN_GRID:
    print(f"  VOL_WIN={vw}M...", end=" ", flush=True)
    auc = run_experiment(raw, US_COLS, TARGET_MAT,
                         BASE_LOOKBACK, BASE_LEAD, BASE_CRISIS_PCT, vw)
    print(f"AUC = {auc:.3f}" if not np.isnan(auc) else "SKIP")
    all_results.append({
        "experiment": "VOL_WINDOW",
        "param_name": "vol_window_months",
        "param_value": vw,
        "AUC": auc,
        "is_base": vw == BASE_VOL_WIN
    })

# ==============================================================================
# 4. SUMMARY
# ==============================================================================
df_rob = pd.DataFrame(all_results)
df_rob.to_csv(RESULTS_DIR + "robustness_summary.csv", index=False)

print("\n" + "=" * 60)
print("ROBUSTNESS SUMMARY (Ensemble)")
print("=" * 60)

for exp in ["LOOKBACK", "CRISIS_PCT", "LEAD", "VOL_WINDOW"]:
    sub = df_rob[df_rob["experiment"] == exp].dropna(subset=["AUC"])
    if len(sub) == 0:
        continue
    aucs = sub["AUC"].values
    print(f"\n  {exp}:")
    print(f"    Range: [{aucs.min():.3f}, {aucs.max():.3f}]")
    print(f"    Mean +/- std: {aucs.mean():.3f} +/- {aucs.std():.3f}")
    print(f"    All > 0.70: {'YES' if (aucs > 0.70).all() else 'NO'}")
    print(f"    All > 0.80: {'YES' if (aucs > 0.80).all() else 'NO'}")

overall_aucs = df_rob.dropna(subset=["AUC"])["AUC"].values
print(f"\n  OVERALL:")
print(f"    Experiments run: {len(overall_aucs)}")
print(f"    AUC range: [{overall_aucs.min():.3f}, {overall_aucs.max():.3f}]")
print(f"    AUC > 0.70 in all: "
      f"{'YES' if (overall_aucs > 0.70).all() else 'NO'}")

# ==============================================================================
# 5. FIGURES
# ==============================================================================
print("\nGenerating figures...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Paper 4 -- Robustness Checks (Ensemble)\n"
    f"AUC stability across parameter variations | "
    f"{len(ENSEMBLE_SEEDS)}-seed ensemble",
    fontsize=12, fontweight="bold"
)

exp_configs = [
    ("LOOKBACK",   "Lookback window (months)",  "A"),
    ("CRISIS_PCT", "Crisis threshold (P%)",      "B"),
    ("LEAD",       "Lead time (months)",          "C"),
    ("VOL_WINDOW", "Volatility window (months)", "D"),
]

for ax, (exp, xlabel, panel) in zip(axes.flat, exp_configs):
    sub = df_rob[df_rob["experiment"] == exp].dropna(subset=["AUC"])
    if len(sub) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        continue

    x_vals = sub["param_value"].values
    y_vals = sub["AUC"].values
    is_base = sub["is_base"].values

    colors = ["#e74c3c" if b else "#3498db" for b in is_base]
    bars = ax.bar(range(len(x_vals)), y_vals,
                  color=colors, edgecolor="black", linewidth=0.5, width=0.6)
    ax.axhline(0.70, color="green", linestyle="--", lw=1.5,
               label="Threshold = 0.70")
    ax.axhline(0.80, color="orange", linestyle=":", lw=1.5,
               label="Strong = 0.80")

    for bar, v, b in zip(bars, y_vals, is_base):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold" if b else "normal")

    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([str(x) for x in x_vals])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("AUC")
    ax.set_ylim(0.5, 1.05)
    ax.set_title(f"({panel}) {exp}\n"
                 f"Range: [{y_vals.min():.3f}, {y_vals.max():.3f}] "
                 f"| Base (red): AUC={y_vals[is_base][0]:.3f}"
                 if is_base.any() else "")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", label="Base parameter"),
        Patch(facecolor="#3498db", label="Alternative"),
    ]
    ax.legend(handles=legend_elements, fontsize=7)
    ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
fig_path = RESULTS_DIR + "fig_p4_robustness.pdf"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Figure saved: {fig_path}")
plt.show()

print("\n" + "=" * 60)
print("DONE -- Robustness checks complete.")
print("Next: run 08_rule6_comparison.py")
print("=" * 60)
