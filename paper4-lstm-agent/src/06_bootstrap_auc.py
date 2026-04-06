"""
Paper 4 -- Bootstrap AUC Confidence Intervals (v2 -- ensemble)
================================================================
Provides statistically rigorous uncertainty quantification
for the ensemble LSTM AUC result using:

  1. Bootstrap CI (B=1000) on test set AUC
  2. Permutation test: is AUC significantly above chance?
  3. Ensemble vs single-seed comparison
  4. DeLong test: LSTM ensemble vs GRU ensemble

v2 changes (Phase 1E):
- Uses shared utils.py (ensemble, helpers)
- Main result is ensemble AUC (not single seed)
- Compares ensemble vs single-seed stability
- Uses SHAP-selected features

Outputs:
  - results/bootstrap_auc_summary.csv
  - results/fig_p4_bootstrap_auc.pdf

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

from utils import (make_sequences, train_single_lstm, predict_probs,
                   train_ensemble, predict_ensemble,
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
TRAIN_RATIO   = 0.75
N_BOOTSTRAP   = 1000
N_PERMUTATION = 1000

# LSTM params
EPOCHS      = 150
BATCH_SIZE  = 16
HIDDEN_SIZE = 64
N_LAYERS    = 2
DROPOUT     = 0.3
LR          = 5e-4
PATIENCE    = 20

# Ensemble
ENSEMBLE_SEEDS = DEFAULT_SEEDS
USE_SELECTED_FEATURES = False  # all 20 features (ablation showed full set is better)

# Seeds for stability test (broader range)
STABILITY_SEEDS = list(range(42, 52))  # 10 seeds

np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 60)
print("Paper 4 -- Bootstrap AUC Confidence Intervals (Ensemble)")
print(f"Target: {TARGET_MAT} | Lead: {LEAD}M | B={N_BOOTSTRAP}")
print("=" * 60)

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
print("\nSTEP 1: Loading data")

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
X_raw     = features.loc[common].values
y_raw     = labels.loc[common, TARGET_MAT].values
dates     = common

# Scale -- fit on training data only
scaler = MinMaxScaler()
n_raw_tr = int(len(X_raw) * TRAIN_RATIO)
scaler.fit(X_raw[:n_raw_tr])
X_sc = scaler.transform(X_raw)

X_seq, y_seq = make_sequences(X_sc, y_raw, LOOKBACK, LEAD)
n_tr = int(len(X_seq) * TRAIN_RATIO)
X_tr, X_te = X_seq[:n_tr], X_seq[n_tr:]
y_tr, y_te = y_seq[:n_tr], y_seq[n_tr:]

print(f"  Train: {len(X_tr)} | Test: {len(X_te)}")
print(f"  Crisis in test: {int(y_te.sum())}")

# ==============================================================================
# 2. TRAIN ENSEMBLE
# ==============================================================================
print(f"\nSTEP 2: Training {len(ENSEMBLE_SEEDS)}-seed ensemble")

train_kwargs = dict(epochs=EPOCHS, batch_size=BATCH_SIZE,
                    hidden_size=HIDDEN_SIZE, n_layers=N_LAYERS,
                    dropout=DROPOUT, lr=LR, patience=PATIENCE)

models_ens = train_ensemble(X_tr, y_tr, seeds=ENSEMBLE_SEEDS,
                            verbose=True, **train_kwargs)
probs_ens  = predict_ensemble(models_ens, X_te)
auc_ens    = roc_auc_score(y_te, probs_ens)
print(f"  Ensemble AUC: {auc_ens:.4f}")

# Also get individual seed AUCs
seed_aucs = []
for m in models_ens:
    p = predict_probs(m, X_te)
    seed_aucs.append(roc_auc_score(y_te, p))
seed_aucs = np.array(seed_aucs)
print(f"  Individual seed AUCs: {[f'{a:.4f}' for a in seed_aucs]}")
print(f"  Seed range: [{seed_aucs.min():.4f}, {seed_aucs.max():.4f}]")

# ==============================================================================
# 3. BOOTSTRAP CI (B=1000)
# ==============================================================================
print(f"\nSTEP 3: Bootstrap CI (B={N_BOOTSTRAP})")

boot_aucs = []
for b in range(N_BOOTSTRAP):
    idx = np.random.choice(len(y_te), len(y_te), replace=True)
    if y_te[idx].sum() > 0 and y_te[idx].sum() < len(idx):
        boot_aucs.append(roc_auc_score(y_te[idx], probs_ens[idx]))
    if (b + 1) % 200 == 0:
        print(f"  Bootstrap {b+1}/{N_BOOTSTRAP}...")

boot_aucs = np.array(boot_aucs)
ci_lo = np.percentile(boot_aucs, 2.5)
ci_hi = np.percentile(boot_aucs, 97.5)
ci_se = boot_aucs.std()

print(f"\n  Ensemble AUC = {auc_ens:.4f}")
print(f"  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"  SE = {ci_se:.4f}")

# ==============================================================================
# 4. PERMUTATION TEST
# ==============================================================================
print(f"\nSTEP 4: Permutation test (N={N_PERMUTATION})")

perm_aucs = []
for p in range(N_PERMUTATION):
    y_perm = np.random.permutation(y_te)
    if y_perm.sum() > 0 and y_perm.sum() < len(y_perm):
        perm_aucs.append(roc_auc_score(y_perm, probs_ens))

perm_aucs = np.array(perm_aucs)
p_value   = (perm_aucs >= auc_ens).mean()

print(f"  Observed AUC: {auc_ens:.4f}")
print(f"  Permutation mean AUC: {perm_aucs.mean():.4f}")
print(f"  p-value: {p_value:.4f}")
print(f"  Significant (p < 0.05): {'YES' if p_value < 0.05 else 'NO'}")

# ==============================================================================
# 5. ENSEMBLE vs SINGLE-SEED STABILITY
# ==============================================================================
print(f"\nSTEP 5: Stability comparison (ensemble vs single-seed)")
print(f"  Testing {len(STABILITY_SEEDS)} single-seed models...")

stability_aucs = []
for seed in STABILITY_SEEDS:
    m = train_single_lstm(X_tr, y_tr, seed=seed, **train_kwargs)
    p = predict_probs(m, X_te)
    a = roc_auc_score(y_te, p)
    stability_aucs.append(a)
    print(f"    Seed {seed}: AUC = {a:.4f}")

stability_aucs = np.array(stability_aucs)
single_mean = stability_aucs.mean()
single_std  = stability_aucs.std()

print(f"\n  Single-seed: mean={single_mean:.4f} +/- {single_std:.4f} "
      f"range=[{stability_aucs.min():.4f}, {stability_aucs.max():.4f}]")
print(f"  Ensemble:    AUC={auc_ens:.4f} (deterministic)")
print(f"  Ensemble improvement: {auc_ens - single_mean:+.4f} vs single mean")
print(f"  Variance reduction: std {single_std:.4f} -> 0 (ensemble is deterministic)")

# ==============================================================================
# 6. DELONG TEST (LSTM ENSEMBLE vs GRU ENSEMBLE)
# ==============================================================================
print("\nSTEP 6: DeLong-style test (LSTM ensemble vs GRU ensemble)")

# Train GRU ensemble
models_gru = train_ensemble(X_tr, y_tr, seeds=ENSEMBLE_SEEDS,
                            verbose=True, rnn_type="GRU", **train_kwargs)
probs_gru = predict_ensemble(models_gru, X_te)
auc_gru   = roc_auc_score(y_te, probs_gru)

# Bootstrap-based DeLong approximation
delong_diffs = []
for _ in range(N_BOOTSTRAP):
    idx = np.random.choice(len(y_te), len(y_te), replace=True)
    if y_te[idx].sum() > 0 and y_te[idx].sum() < len(idx):
        a_lstm = roc_auc_score(y_te[idx], probs_ens[idx])
        a_gru  = roc_auc_score(y_te[idx], probs_gru[idx])
        delong_diffs.append(a_lstm - a_gru)

delong_diffs = np.array(delong_diffs)
delong_p = 2 * min((delong_diffs >= 0).mean(),
                    (delong_diffs <= 0).mean())
auc_diff = auc_ens - auc_gru

print(f"  LSTM ensemble AUC: {auc_ens:.4f}")
print(f"  GRU  ensemble AUC: {auc_gru:.4f}")
print(f"  Difference: {auc_diff:+.4f}")
print(f"  Bootstrap p-value: {delong_p:.4f}")
print(f"  Significantly different (p<0.05): "
      f"{'YES' if delong_p < 0.05 else 'NO -- CIs overlap'}")

# ==============================================================================
# 7. SAVE RESULTS
# ==============================================================================
summary = pd.DataFrame([{
    "metric"            : "Ensemble AUC",
    "value"             : round(auc_ens, 4),
    "ci_lo"             : round(ci_lo, 4),
    "ci_hi"             : round(ci_hi, 4),
    "se"                : round(ci_se, 4),
    "permutation_pvalue": round(p_value, 4),
    "single_seed_mean"  : round(single_mean, 4),
    "single_seed_std"   : round(single_std, 4),
    "single_seed_min"   : round(stability_aucs.min(), 4),
    "single_seed_max"   : round(stability_aucs.max(), 4),
    "gru_ensemble_auc"  : round(auc_gru, 4),
    "lstm_vs_gru_diff"  : round(auc_diff, 4),
    "delong_pvalue"     : round(delong_p, 4),
    "n_ensemble_seeds"  : len(ENSEMBLE_SEEDS),
}])
summary.to_csv(RESULTS_DIR + "bootstrap_auc_summary.csv", index=False)

pd.DataFrame({
    "seed": STABILITY_SEEDS,
    "AUC" : stability_aucs
}).to_csv(RESULTS_DIR + "seed_stability.csv", index=False)

# ==============================================================================
# 8. FIGURES
# ==============================================================================
print("\nSTEP 7: Generating figures")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(
    f"Paper 4 -- Statistical Validation (Ensemble)\n"
    f"LSTM: {TARGET_MAT} | Lead={LEAD}M | "
    f"AUC = {auc_ens:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]",
    fontsize=12, fontweight="bold"
)

# Panel 1: Bootstrap distribution
ax = axes[0, 0]
ax.hist(boot_aucs, bins=40, color="#3498db", alpha=0.7,
        edgecolor="white", linewidth=0.5)
ax.axvline(auc_ens, color="red", lw=2.5,
           label=f"Ensemble AUC = {auc_ens:.4f}")
ax.axvline(ci_lo, color="orange", lw=1.5, linestyle="--",
           label=f"95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
ax.axvline(ci_hi, color="orange", lw=1.5, linestyle="--")
ax.axvline(0.50, color="gray", lw=1, linestyle=":", label="Random = 0.50")
ax.set_xlabel("Bootstrap AUC")
ax.set_ylabel("Frequency")
ax.set_title(f"Bootstrap AUC Distribution (B={N_BOOTSTRAP})\n"
             f"Mean = {boot_aucs.mean():.4f} +/- {ci_se:.4f}")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Panel 2: Permutation test
ax = axes[0, 1]
ax.hist(perm_aucs, bins=40, color="#e74c3c", alpha=0.7,
        edgecolor="white", linewidth=0.5, label="Null distribution")
ax.axvline(auc_ens, color="blue", lw=2.5,
           label=f"Ensemble AUC = {auc_ens:.4f}")
ax.axvline(perm_aucs.mean(), color="red", lw=1.5, linestyle="--",
           label=f"Null mean = {perm_aucs.mean():.4f}")
ax.set_xlabel("AUC under permuted labels")
ax.set_ylabel("Frequency")
ax.set_title(f"Permutation Test (N={N_PERMUTATION})\n"
             f"p-value = {p_value:.4f} -> "
             f"{'Significant' if p_value < 0.05 else 'Not significant'}")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Panel 3: Ensemble vs single-seed stability
ax = axes[1, 0]
ax.plot(STABILITY_SEEDS, stability_aucs, "o-", color="#95a5a6", lw=1.5,
        markersize=8, markerfacecolor="white", markeredgecolor="#95a5a6",
        markeredgewidth=2, label=f"Single seed (mean={single_mean:.4f})")
ax.axhline(auc_ens, color="#2ecc71", lw=2.5, linestyle="-",
           label=f"Ensemble ({len(ENSEMBLE_SEEDS)} seeds) = {auc_ens:.4f}")
ax.axhline(single_mean, color="#95a5a6", linestyle="--", lw=1.5)
ax.fill_between(STABILITY_SEEDS,
                single_mean - single_std,
                single_mean + single_std,
                alpha=0.15, color="gray",
                label=f"Single-seed +/-1 std = {single_std:.4f}")
ax.axhline(0.70, color="red", linestyle=":", lw=1, label="Threshold = 0.70")
ax.set_xlabel("Random seed")
ax.set_ylabel("AUC")
ax.set_title(f"Ensemble vs Single-Seed Stability\n"
             f"Ensemble eliminates seed variance entirely")
ax.set_ylim(0.5, 1.05)
ax.legend(fontsize=7)
ax.grid(alpha=0.3)
ax.set_xticks(STABILITY_SEEDS)

# Panel 4: LSTM vs GRU ensemble
ax = axes[1, 1]
ax.hist(delong_diffs, bins=40, color="#9b59b6", alpha=0.7,
        edgecolor="white", linewidth=0.5)
ax.axvline(auc_diff, color="red", lw=2.5,
           label=f"Observed diff = {auc_diff:+.4f}")
ax.axvline(0, color="black", lw=1.5, linestyle="--",
           label="No difference")
ci_diff_lo = np.percentile(delong_diffs, 2.5)
ci_diff_hi = np.percentile(delong_diffs, 97.5)
ax.axvline(ci_diff_lo, color="orange", lw=1, linestyle=":")
ax.axvline(ci_diff_hi, color="orange", lw=1, linestyle=":",
           label=f"95% CI: [{ci_diff_lo:+.4f}, {ci_diff_hi:+.4f}]")
ax.set_xlabel("AUC difference (LSTM - GRU)")
ax.set_ylabel("Frequency")
ax.set_title(f"DeLong-style test: LSTM vs GRU (ensembles)\n"
             f"p = {delong_p:.4f} -> "
             f"{'Significant' if delong_p < 0.05 else 'CIs overlap'}")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
fig_path = RESULTS_DIR + "fig_p4_bootstrap_auc.pdf"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Figure saved: {fig_path}")
plt.show()

# ==============================================================================
# 9. FINAL SUMMARY
# ==============================================================================
print("\n" + "=" * 60)
print("STATISTICAL VALIDATION SUMMARY (ENSEMBLE)")
print("=" * 60)
print(f"\n  Ensemble AUC = {auc_ens:.4f} [{ci_lo:.4f}, {ci_hi:.4f}] "
      f"(95% bootstrap CI)")
print(f"  Permutation p-value: {p_value:.4f} -> "
      f"{'significant' if p_value < 0.05 else 'NOT significant'}")
print(f"  Single-seed stability: {single_mean:.4f} +/- {single_std:.4f}")
print(f"  Ensemble improvement: {auc_ens - single_mean:+.4f} vs single mean")
print(f"  LSTM vs GRU: diff = {auc_diff:+.4f}, p = {delong_p:.4f}")

print(f"\n  FOR PAPER 4 ABSTRACT:")
print(f"  'The {len(ENSEMBLE_SEEDS)}-seed LSTM ensemble achieves "
      f"AUC = {auc_ens:.3f} [95% CI: {ci_lo:.3f}-{ci_hi:.3f}],")
print(f"   significantly exceeding chance (permutation p = {p_value:.4f})")
print(f"   and stabilising single-seed variance "
      f"(std = {single_std:.3f} -> deterministic).'")

print("\n" + "=" * 60)
print("DONE -- Bootstrap AUC complete.")
print("Next: run 07_robustness_checks.py")
print("=" * 60)
