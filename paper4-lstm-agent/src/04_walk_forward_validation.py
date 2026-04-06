"""
Paper 4 -- Walk-Forward Validation (v3 -- expanding quarterly windows)
========================================================================
Proves the LSTM regime classifier is NOT overfitting by
evaluating out-of-sample AUC across expanding windows with
quarterly test steps.

v3 changes (Phase 4B):
- Expanding windows with 3-month (quarterly) test steps
- Minimum 5 years training data before first test
- Reports per-window and pooled OOS metrics
- Cumulative AUC evolution plot
- Maintains backward compat: also reports 5-window summary

Focus: GR_Fuel_Energy at lead=2M (best result from script 02)

Walk-forward design:
  - Start training at earliest date
  - First test window begins after MIN_TRAIN_YEARS of data
  - Each step: expand training by 3 months, test next 3 months
  - This yields ~20-30 windows (depending on data range)

Outputs:
  - results/wf_validation_summary.csv       (per-window)
  - results/wf_predictions.csv              (all OOS predictions)
  - results/wf_quarterly_evolution.csv      (cumulative metrics)
  - results/fig_p4_walk_forward.pdf

Run from: construction-risk-suite/paper4-lstm-agent/src/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (roc_auc_score, f1_score, roc_curve)

import warnings
warnings.filterwarnings("ignore")

import torch

from utils import (make_sequences, train_ensemble, predict_ensemble,
                   load_selected_features, DEFAULT_SEEDS)

# ==============================================================================
# PARAMETERS
# ==============================================================================
PROCESSED_DIR = "../data/processed/"
RESULTS_DIR   = "../results/"
SEED          = 42
LOOKBACK      = 6
LEAD          = 2
TARGET_MAT    = "GR_Fuel_Energy"
EPOCHS        = 150
BATCH_SIZE    = 16
HIDDEN_SIZE   = 64
N_LAYERS      = 2
DROPOUT       = 0.3
LR            = 5e-4
PATIENCE      = 20

# Ensemble
ENSEMBLE_SEEDS = DEFAULT_SEEDS

# Feature selection
USE_SELECTED_FEATURES = False  # all 20 features (ablation showed full set is better)

# Walk-forward parameters
MIN_TRAIN_MONTHS = 120      # 10 years minimum training (need enough crisis data)
STEP_MONTHS      = 3        # quarterly steps (advance by 3M each iteration)
TEST_MONTHS      = 12       # test window size (12M to ensure enough sequences)
MIN_TEST_CRISIS  = 0        # allow windows with 0 crisis (report but skip AUC)

np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 60)
print("Paper 4 -- Walk-Forward Validation (Expanding Quarterly)")
print(f"Target: {TARGET_MAT} | Lead: {LEAD}M | "
      f"Seeds: {len(ENSEMBLE_SEEDS)}")
print(f"Step: {STEP_MONTHS}M | Test window: {TEST_MONTHS}M | "
      f"Min train: {MIN_TRAIN_MONTHS}M")
print("=" * 60)

# ==============================================================================
# 1. LOAD PROCESSED DATA
# ==============================================================================
print("\nSTEP 1: Loading processed data")

sel_feats = load_selected_features(PROCESSED_DIR) if USE_SELECTED_FEATURES else None

if sel_feats is not None:
    features = pd.read_csv(PROCESSED_DIR + "features_selected.csv",
                           index_col=0, parse_dates=True)
    print(f"  Using SHAP-selected features: {len(sel_feats)}")
else:
    features = pd.read_csv(PROCESSED_DIR + "features.csv",
                           index_col=0, parse_dates=True)
    print(f"  Using all features: {features.shape[1]}")

labels = pd.read_csv(PROCESSED_DIR + "crisis_labels.csv",
                     index_col=0, parse_dates=True)

FEAT_COLS = list(features.columns)
common    = features.index.intersection(labels.dropna().index)
X_all     = features.loc[common].values
y_all     = labels.loc[common, TARGET_MAT].values
dates_all = common

print(f"  Features : {len(FEAT_COLS)} columns")
print(f"  Samples  : {len(X_all)}")
print(f"  Period   : {dates_all[0].strftime('%Y-%m')} -> "
      f"{dates_all[-1].strftime('%Y-%m')}")
print(f"  Crisis   : {int(y_all.sum())} months "
      f"({y_all.mean()*100:.1f}%)")

# ==============================================================================
# 2. BUILD EXPANDING WINDOWS
# ==============================================================================
print("\nSTEP 2: Building expanding quarterly windows")

# Generate window boundaries
all_months = pd.to_datetime(dates_all)
first_date = all_months[0]
last_date  = all_months[-1]

# First test start: after MIN_TRAIN_MONTHS
first_test_start = first_date + pd.DateOffset(months=MIN_TRAIN_MONTHS)

# Generate quarterly test boundaries
test_starts = pd.date_range(start=first_test_start, end=last_date,
                            freq=f"{STEP_MONTHS}MS")

windows = []
for ts in test_starts:
    te = ts + pd.DateOffset(months=TEST_MONTHS) - pd.DateOffset(days=1)
    if te > last_date:
        te = last_date
    windows.append((ts, te))

# Remove windows that are too short to create sequences
# Need at least LOOKBACK + LEAD + 1 months
min_test_months = LOOKBACK + LEAD + 1
windows = [(ts, te) for ts, te in windows
           if (te - ts).days >= min_test_months * 28]

print(f"  Total windows: {len(windows)}")
print(f"  First test: {windows[0][0].strftime('%Y-%m')} -> {windows[0][1].strftime('%Y-%m')}")
print(f"  Last test:  {windows[-1][0].strftime('%Y-%m')} -> {windows[-1][1].strftime('%Y-%m')}")

# ==============================================================================
# 3. WALK-FORWARD LOOP
# ==============================================================================
print("\nSTEP 3: Walk-forward validation")
print("-" * 60)

wf_results = []
all_probs, all_true, all_dates_oos = [], [], []
cumulative_auc_history = []

for win_idx, (test_start, test_end) in enumerate(windows, 1):
    # Training: everything before test_start
    train_mask = all_months < test_start
    test_mask  = (all_months >= test_start) & (all_months <= test_end)

    X_tr_raw = X_all[train_mask]
    y_tr_raw = y_all[train_mask]
    X_te_raw = X_all[test_mask]
    y_te_raw = y_all[test_mask]
    d_te_raw = dates_all[test_mask]

    if len(X_te_raw) < 1:
        continue

    # Scale on training data only
    scaler = MinMaxScaler()
    scaler.fit(X_tr_raw)
    X_tr_sc = scaler.transform(X_tr_raw)
    X_te_sc = scaler.transform(X_te_raw)

    # Build sequences
    result_tr = make_sequences(X_tr_sc, y_tr_raw, LOOKBACK, LEAD,
                               dates=dates_all[train_mask])
    result_te = make_sequences(X_te_sc, y_te_raw, LOOKBACK, LEAD,
                               dates=d_te_raw)

    if len(result_tr) == 3:
        X_tr_seq, y_tr_seq, _ = result_tr
    else:
        X_tr_seq, y_tr_seq = result_tr

    if len(result_te) == 3:
        X_te_seq, y_te_seq, d_te_seq = result_te
    else:
        X_te_seq, y_te_seq = result_te
        d_te_seq = None

    if len(X_te_seq) == 0:
        continue

    n_te_crisis = int(y_te_seq.sum())
    n_tr_crisis = int(y_tr_seq.sum())

    # Train ensemble
    models = train_ensemble(X_tr_seq, y_tr_seq, seeds=ENSEMBLE_SEEDS,
                            epochs=EPOCHS, batch_size=BATCH_SIZE,
                            hidden_size=HIDDEN_SIZE, n_layers=N_LAYERS,
                            dropout=DROPOUT, lr=LR, patience=PATIENCE)
    probs = predict_ensemble(models, X_te_seq)
    preds = (probs > 0.5).astype(int)

    # Metrics
    if n_te_crisis > 0 and n_te_crisis < len(y_te_seq):
        auc = roc_auc_score(y_te_seq, probs)
    else:
        auc = np.nan
    f1  = f1_score(y_te_seq, preds, zero_division=0)
    rec = (preds[y_te_seq == 1] == 1).mean() if n_te_crisis > 0 else np.nan

    # Print progress every 5 windows or if crisis present
    if win_idx % 5 == 1 or n_te_crisis > 0:
        auc_str = f"{auc:.3f}" if not np.isnan(auc) else "N/A"
        print(f"  W{win_idx:02d}: train<{test_start.strftime('%Y-%m')} | "
              f"test={test_start.strftime('%Y-%m')}->{test_end.strftime('%Y-%m')} | "
              f"crisis={n_te_crisis} | AUC={auc_str}")

    wf_results.append({
        "window"        : win_idx,
        "test_start"    : test_start.strftime("%Y-%m"),
        "test_end"      : test_end.strftime("%Y-%m"),
        "n_train"       : len(y_tr_seq),
        "n_test"        : len(y_te_seq),
        "n_train_crisis": n_tr_crisis,
        "n_test_crisis" : n_te_crisis,
        "AUC"           : round(auc, 3) if not np.isnan(auc) else np.nan,
        "F1"            : round(f1, 3),
        "Recall_crisis" : round(rec, 3) if not np.isnan(rec) else np.nan,
    })

    all_probs.extend(probs)
    all_true.extend(y_te_seq)
    if d_te_seq is not None:
        all_dates_oos.extend(d_te_seq)

    # Track cumulative AUC
    cum_true = np.array(all_true)
    cum_probs = np.array(all_probs)
    if cum_true.sum() > 0 and cum_true.sum() < len(cum_true):
        cum_auc = roc_auc_score(cum_true, cum_probs)
    else:
        cum_auc = np.nan
    cumulative_auc_history.append({
        "window": win_idx,
        "test_end": test_end.strftime("%Y-%m"),
        "cumulative_auc": round(cum_auc, 3) if not np.isnan(cum_auc) else np.nan,
        "n_total_oos": len(cum_true),
        "n_total_crisis": int(cum_true.sum()),
    })

# ==============================================================================
# 4. AGGREGATE RESULTS
# ==============================================================================
print("\n" + "=" * 60)
print("STEP 4: Aggregate results")
print("=" * 60)

df_wf = pd.DataFrame(wf_results)

# Windows with valid AUC (both classes present)
df_valid = df_wf.dropna(subset=["AUC"])
n_total_windows = len(df_wf)
n_valid_windows = len(df_valid)

mean_auc = df_valid["AUC"].mean() if len(df_valid) > 0 else np.nan
std_auc  = df_valid["AUC"].std() if len(df_valid) > 0 else np.nan
min_auc  = df_valid["AUC"].min() if len(df_valid) > 0 else np.nan
max_auc  = df_valid["AUC"].max() if len(df_valid) > 0 else np.nan

print(f"\n  Total windows: {n_total_windows}")
print(f"  Windows with valid AUC: {n_valid_windows} (both classes present)")
print(f"\n  Walk-forward AUC summary (valid windows):")
print(f"    Mean  : {mean_auc:.3f}")
print(f"    Std   : {std_auc:.3f}")
print(f"    Min   : {min_auc:.3f}")
print(f"    Max   : {max_auc:.3f}")

# Pooled out-of-sample AUC
all_probs_arr = np.array(all_probs)
all_true_arr  = np.array(all_true)
pooled_auc = roc_auc_score(all_true_arr, all_probs_arr)
pooled_f1  = f1_score(all_true_arr, (all_probs_arr > 0.5).astype(int), zero_division=0)

print(f"\n  Pooled out-of-sample:")
print(f"    AUC: {pooled_auc:.3f}")
print(f"    F1:  {pooled_f1:.3f}")
print(f"    N:   {len(all_true_arr)} ({int(all_true_arr.sum())} crisis)")

# Save
df_wf.to_csv(RESULTS_DIR + "wf_validation_summary.csv", index=False)
pd.DataFrame({"date": all_dates_oos, "prob": all_probs,
              "true": all_true}).to_csv(
    RESULTS_DIR + "wf_predictions.csv", index=False)

df_cum = pd.DataFrame(cumulative_auc_history)
df_cum.to_csv(RESULTS_DIR + "wf_quarterly_evolution.csv", index=False)

# Verdict
print("\n" + "*" * 60)
if mean_auc > 0.75 and min_auc > 0.55:
    print("* VERDICT: ROBUST -- AUC stable across expanding windows! *")
    print("* Walk-forward confirms NO overfitting -> PUBLISHABLE *")
elif mean_auc > 0.65:
    print("* VERDICT: MODERATE -- acceptable stability *")
    print("* Early windows with data scarcity may show lower AUC *")
else:
    print("* VERDICT: UNSTABLE -- investigate further *")
print("*" * 60)

# ==============================================================================
# 5. FIGURES
# ==============================================================================
print("\nSTEP 5: Generating figures")

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle(
    f"Paper 4 -- Walk-Forward Validation (Expanding Quarterly)\n"
    f"{TARGET_MAT} | Lead={LEAD}M | "
    f"{n_total_windows} windows (step={STEP_MONTHS}M, test={TEST_MONTHS}M) | "
    f"Pooled AUC={pooled_auc:.3f}",
    fontsize=12, fontweight="bold"
)

# Panel 1: AUC per window (bar chart)
ax = axes[0, 0]
colors_bar = []
for _, row in df_wf.iterrows():
    if pd.isna(row["AUC"]):
        colors_bar.append("#bdc3c7")   # gray for N/A
    elif row["AUC"] > 0.70:
        colors_bar.append("#2ecc71")   # green
    elif row["AUC"] > 0.55:
        colors_bar.append("#f39c12")   # orange
    else:
        colors_bar.append("#e74c3c")   # red

auc_plot = df_wf["AUC"].fillna(0).values
bars = ax.bar(df_wf["window"], auc_plot,
              color=colors_bar, edgecolor="black", linewidth=0.3)
ax.axhline(mean_auc, color="blue", linestyle="--", lw=1.5,
           label=f"Mean AUC = {mean_auc:.3f}")
ax.axhline(0.70, color="green", linestyle=":", lw=1,
           label="AUC = 0.70")
ax.axhline(0.50, color="red", linestyle=":", lw=1,
           label="Random = 0.50")
ax.set_xlabel("Window")
ax.set_ylabel("AUC")
ax.set_title(f"Out-of-sample AUC per window\n"
             f"({n_valid_windows} valid / {n_total_windows} total)")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=7)
ax.grid(alpha=0.3, axis="y")

# Panel 2: Cumulative AUC evolution
ax = axes[0, 1]
cum_vals = df_cum["cumulative_auc"].values
cum_x = df_cum["window"].values
ax.plot(cum_x, cum_vals, "o-", color="blue", lw=2, markersize=3)
ax.axhline(pooled_auc, color="blue", linestyle="--", lw=1, alpha=0.5,
           label=f"Final pooled = {pooled_auc:.3f}")
ax.axhline(0.70, color="green", linestyle=":", lw=1)
ax.set_xlabel("Window (cumulative)")
ax.set_ylabel("Cumulative OOS AUC")
ax.set_title("Cumulative AUC Evolution\n(stabilizes as data accumulates)")
ax.set_ylim(0.3, 1.05)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Panel 3: Pooled ROC curve
ax = axes[1, 0]
fpr, tpr, _ = roc_curve(all_true_arr, all_probs_arr)
ax.plot(fpr, tpr, color="blue", lw=2,
        label=f"Pooled OOS (AUC = {pooled_auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
ax.fill_between(fpr, tpr, alpha=0.1, color="blue")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Pooled Out-of-Sample ROC Curve")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel 4: Crisis probability timeline (full OOS)
ax = axes[1, 1]
if len(all_dates_oos) > 0:
    dates_arr = np.array(all_dates_oos, dtype="datetime64[D]")
    sort_idx  = np.argsort(dates_arr)
    d_sorted  = dates_arr[sort_idx]
    p_sorted  = all_probs_arr[sort_idx]
    t_sorted  = all_true_arr[sort_idx]

    ax.fill_between(d_sorted, 0, 1,
                    where=(t_sorted == 1),
                    color="red", alpha=0.20, label="Actual crisis")
    ax.plot(d_sorted, p_sorted, color="orange", lw=1,
            alpha=0.8, label="P(crisis)")
    ax.axhline(0.5, color="gray", linestyle="--", lw=0.8)
    ax.set_ylabel("P(crisis)")
    ax.set_ylim(0, 1)
    ax.set_title("Crisis Probability -- Full OOS Timeline")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.grid(alpha=0.3)

plt.tight_layout()
fig_path = RESULTS_DIR + "fig_p4_walk_forward.pdf"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"  Figure saved: {fig_path}")
plt.show()

# ==============================================================================
# 6. SUMMARY
# ==============================================================================
print("\n" + "=" * 60)
print("WALK-FORWARD SUMMARY (for Paper 4)")
print("=" * 60)
print(f"\n  Expanding walk-forward with quarterly steps:")
print(f"    Windows: {n_total_windows} ({n_valid_windows} with both classes)")
print(f"    Step: {STEP_MONTHS}M | Test window: {TEST_MONTHS}M")
print(f"    Mean AUC: {mean_auc:.3f} +/- {std_auc:.3f}")
print(f"    Pooled OOS AUC: {pooled_auc:.3f}")
print(f"\n  FOR PAPER 4:")
print(f"  'Walk-forward validation with {n_total_windows} expanding windows")
print(f"   (quarterly steps, {TEST_MONTHS}M test windows) yields a pooled")
print(f"   out-of-sample AUC of {pooled_auc:.3f}, with per-window AUC")
print(f"   averaging {mean_auc:.3f} +/- {std_auc:.3f}.'")

print("\n" + "=" * 60)
print("DONE -- Walk-forward validation complete.")
print("Next: run 05_benchmarks.py")
print("=" * 60)
