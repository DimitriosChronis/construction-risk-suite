"""
Paper 4 -- LSTM Regime Classification (v3 — ensemble + feature selection)
==========================================================================
Refined hypothesis: US PPI series can CLASSIFY whether Greek
construction costs will enter a CRISIS REGIME 1-4 months ahead.

v3 changes (Phase 1C):
- Uses shared utils.py (LSTMClassifier, ensemble, helpers)
- 5-seed ensemble averaging for stable predictions
- Optional SHAP-selected features (14 of 20)
- Loads pre-processed data from data/processed/

Run from: construction-risk-suite/paper4-lstm-agent/src/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (roc_auc_score, roc_curve, f1_score)

import warnings
warnings.filterwarnings("ignore")

import torch

from utils import (LSTMClassifier, make_sequences, train_ensemble,
                   predict_ensemble, predict_probs, load_selected_features,
                   DEFAULT_SEEDS)

print(f"Backend: torch")

# ==============================================================================
# PARAMETERS
# ==============================================================================
PROCESSED_DIR = "../data/processed/"
RESULTS_DIR   = "../results/"
SEED          = 42
LOOKBACK      = 6
LEAD_TIMES    = [1, 2, 3, 4]
VOL_WINDOW    = 6
CRISIS_PCT    = 0.75
TRAIN_RATIO   = 0.75
EPOCHS        = 150
BATCH_SIZE    = 16
HIDDEN_SIZE   = 64
N_LAYERS      = 2
DROPOUT       = 0.3
LR            = 5e-4

# Ensemble seeds
ENSEMBLE_SEEDS = DEFAULT_SEEDS  # [42, 43, 44, 45, 46]

# Feature selection: use SHAP-selected features if available
USE_SELECTED_FEATURES = False  # all 20 features (ablation showed full set is better)

# Target materials
TARGET_MATERIALS = ["GR_Fuel_Energy", "GR_Steel", "GR_Concrete", "GR_PVC_Pipes"]

np.random.seed(SEED)
torch.manual_seed(SEED)

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
print("\n" + "=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

df = pd.read_csv(PROCESSED_DIR + "aligned_returns.csv",
                 index_col=0, parse_dates=True)
print(f"Loaded: {df.shape[0]} obs x {df.shape[1]} series")
print(f"Period: {df.index[0].strftime('%Y-%m')} to "
      f"{df.index[-1].strftime('%Y-%m')}")

# ==============================================================================
# 2. LOAD FEATURES
# ==============================================================================
print("\n" + "=" * 60)
print("STEP 2: Loading features")
print("=" * 60)

# Try selected features first, fall back to full
sel_feats = load_selected_features(PROCESSED_DIR) if USE_SELECTED_FEATURES else None

if sel_feats is not None:
    features = pd.read_csv(PROCESSED_DIR + "features_selected.csv",
                           index_col=0, parse_dates=True)
    print(f"Using SHAP-selected features: {len(sel_feats)} columns")
else:
    features = pd.read_csv(PROCESSED_DIR + "features.csv",
                           index_col=0, parse_dates=True)
    print(f"Using all features: {features.shape[1]} columns")

FEAT_COLS = list(features.columns)
print(f"Features: {len(FEAT_COLS)} columns")
print(f"Feature period: {features.index[0].strftime('%Y-%m')} -> "
      f"{features.index[-1].strftime('%Y-%m')}")

# ==============================================================================
# 3. DEFINE CRISIS REGIMES PER MATERIAL
# ==============================================================================
print("\n" + "=" * 60)
print("STEP 3: Crisis regime definition per material")
print("=" * 60)

crisis_regimes = {}
for mat in TARGET_MATERIALS:
    vol = df[mat].rolling(VOL_WINDOW).std()
    thr = vol.quantile(CRISIS_PCT)
    crisis = (vol > thr).astype(int)
    crisis_regimes[mat] = crisis
    n_crisis = crisis.sum()
    print(f"  {mat}: threshold={thr:.4f}, "
          f"crisis months={n_crisis} ({n_crisis/len(crisis)*100:.1f}%)")

# ==============================================================================
# 4. RUN EXPERIMENT PER MATERIAL x LEAD TIME
# ==============================================================================
print("\n" + "=" * 60)
print("STEP 4: Training per material x lead time")
print(f"  Ensemble: {len(ENSEMBLE_SEEDS)} seeds {ENSEMBLE_SEEDS}")
print("=" * 60)

all_results = []
best_overall = {"AUC": 0}

for mat in TARGET_MATERIALS:
    for lead in LEAD_TIMES:
        # Align features and labels
        common = features.index.intersection(crisis_regimes[mat].dropna().index)
        X_raw = features.loc[common].values
        y_raw = crisis_regimes[mat].loc[common].values

        # Split BEFORE scaling to prevent data leakage
        n_raw_tr = int(len(X_raw) * TRAIN_RATIO)
        scaler = MinMaxScaler()
        scaler.fit(X_raw[:n_raw_tr])
        X_sc = scaler.transform(X_raw)

        X_seq, y_seq = make_sequences(X_sc, y_raw, LOOKBACK, lead)
        dates_seq = common[LOOKBACK + lead: LOOKBACK + lead + len(y_seq)]

        n_tr = int(len(X_seq) * TRAIN_RATIO)
        X_tr, X_te = X_seq[:n_tr], X_seq[n_tr:]
        y_tr, y_te = y_seq[:n_tr], y_seq[n_tr:]
        dates_te   = dates_seq[n_tr:]

        if y_te.sum() == 0 or y_te.sum() == len(y_te):
            continue

        # Train ensemble
        models = train_ensemble(X_tr, y_tr, seeds=ENSEMBLE_SEEDS,
                                epochs=EPOCHS, batch_size=BATCH_SIZE,
                                hidden_size=HIDDEN_SIZE, n_layers=N_LAYERS,
                                dropout=DROPOUT, lr=LR)
        probs = predict_ensemble(models, X_te)
        preds = (probs > 0.5).astype(int)

        auc = roc_auc_score(y_te, probs)
        f1  = f1_score(y_te, preds, zero_division=0)
        rec = (preds[y_te == 1] == 1).mean() if y_te.sum() > 0 else 0

        print(f"  {mat} | lead={lead}M | AUC={auc:.3f} | "
              f"F1={f1:.3f} | Recall(crisis)={rec:.3f}")

        result = {
            "material": mat, "lead": lead,
            "AUC": auc, "F1": f1, "Recall_crisis": rec,
            "n_test": len(y_te), "n_crisis_test": int(y_te.sum()),
            "probs": probs, "preds": preds,
            "y_true": y_te, "dates": dates_te
        }
        all_results.append(result)

        if auc > best_overall["AUC"]:
            best_overall = result

# ==============================================================================
# 5. VERDICT
# ==============================================================================
print("\n" + "=" * 60)
print("STEP 5: RESULTS SUMMARY")
print("=" * 60)

df_res = pd.DataFrame([{
    "material": r["material"], "lead": r["lead"],
    "AUC": round(r["AUC"], 3),
    "F1":  round(r["F1"], 3),
    "Recall_crisis": round(r["Recall_crisis"], 3)
} for r in all_results])

print("\nAll results:")
print(df_res.to_string(index=False))

print(f"\nBest result:")
print(f"  Material : {best_overall['material']}")
print(f"  Lead     : {best_overall['lead']} month(s)")
print(f"  AUC      : {best_overall['AUC']:.3f}")
print(f"  F1       : {best_overall['F1']:.3f}")
print(f"  Recall   : {best_overall['Recall_crisis']:.3f}")
print(f"  Ensemble : {len(ENSEMBLE_SEEDS)} seeds")

print("\n" + "*" * 60)
if best_overall["AUC"] > 0.70:
    print("* VERDICT: STRONG SIGNAL -- Paper 4 is HIGHLY VIABLE! *")
elif best_overall["AUC"] > 0.60:
    print("* VERDICT: MODERATE SIGNAL -- Paper 4 is VIABLE with tuning *")
else:
    print("* VERDICT: WEAK SIGNAL -- needs different features *")
print("*" * 60)

# ==============================================================================
# 6. FIGURES
# ==============================================================================
print("\nGenerating figures...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(
    "Paper 4 -- LSTM Crisis Regime Classification (Ensemble)\n"
    f"US PPI -> Greek Construction Cost Crisis (lead 1-4 months) | "
    f"{len(ENSEMBLE_SEEDS)}-seed ensemble | {len(FEAT_COLS)} features",
    fontsize=13, fontweight="bold"
)

# Panel 1: AUC heatmap
ax = axes[0, 0]
pivot = df_res.pivot(index="material", columns="lead", values="AUC")
pivot.index = [m.replace("GR_", "") for m in pivot.index]
im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0.4, vmax=0.95,
               aspect="auto")
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels([f"{l}M" for l in pivot.columns])
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index)
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        ax.text(j, i, f"{pivot.values[i,j]:.3f}",
                ha="center", va="center", fontsize=10, fontweight="bold")
plt.colorbar(im, ax=ax)
ax.set_title("AUC by material x lead time\n(green = good, red = bad)")
ax.set_xlabel("Lead time")
ax.set_ylabel("Greek material")

# Panel 2: ROC curve for best result
ax = axes[0, 1]
fpr, tpr, _ = roc_curve(best_overall["y_true"], best_overall["probs"])
ax.plot(fpr, tpr, color="blue", lw=2,
        label=f"LSTM ensemble (AUC = {best_overall['AUC']:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
ax.fill_between(fpr, tpr, alpha=0.1, color="blue")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title(f"ROC Curve -- Best: {best_overall['material'].replace('GR_','')}"
             f" lead={best_overall['lead']}M")
ax.legend()
ax.grid(alpha=0.3)

# Panel 3: Crisis detection timeline for best result
ax = axes[1, 0]
dates_te = best_overall["dates"]
y_te     = best_overall["y_true"]
probs_te = best_overall["probs"]

ax.fill_between(dates_te, 0, 1, where=(y_te == 1),
                color="red", alpha=0.25, label="Actual crisis")
ax.plot(dates_te, probs_te, color="orange", lw=1.5,
        label="Predicted P(crisis)")
ax.axhline(0.5, color="gray", linestyle="--", lw=1)
ax.set_ylabel("P(crisis)")
ax.set_ylim(0, 1)
ax.set_title(f"Crisis probability timeline -- "
             f"{best_overall['material'].replace('GR_','')} "
             f"lead={best_overall['lead']}M")
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.grid(alpha=0.3)

# Panel 4: F1 heatmap
ax = axes[1, 1]
pivot_f1 = df_res.pivot(index="material", columns="lead", values="F1")
pivot_f1.index = [m.replace("GR_", "") for m in pivot_f1.index]
im2 = ax.imshow(pivot_f1.values, cmap="RdYlGn", vmin=0.0, vmax=0.8,
                aspect="auto")
ax.set_xticks(range(len(pivot_f1.columns)))
ax.set_xticklabels([f"{l}M" for l in pivot_f1.columns])
ax.set_yticks(range(len(pivot_f1.index)))
ax.set_yticklabels(pivot_f1.index)
for i in range(len(pivot_f1.index)):
    for j in range(len(pivot_f1.columns)):
        ax.text(j, i, f"{pivot_f1.values[i,j]:.3f}",
                ha="center", va="center", fontsize=10, fontweight="bold")
plt.colorbar(im2, ax=ax)
ax.set_title("F1 Score by material x lead time")
ax.set_xlabel("Lead time")
ax.set_ylabel("Greek material")

plt.tight_layout()
plt.savefig(RESULTS_DIR + "fig_lstm_regime_clf.pdf", dpi=150,
            bbox_inches="tight")
print("Figure saved: ../results/fig_lstm_regime_clf.pdf")
plt.show()

# ==============================================================================
# 7. SAVE RESULTS
# ==============================================================================
df_res.to_csv(RESULTS_DIR + "lstm_regime_clf_summary.csv", index=False)
print("Results saved: ../results/lstm_regime_clf_summary.csv")

print("\n" + "=" * 60)
print("DONE.")
print("=" * 60)
