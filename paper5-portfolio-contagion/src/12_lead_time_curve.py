"""
Paper 5 — LSTM lead-time curve
===================================
Reviewer-proofing: a Q1 referee will ALWAYS ask "why lead=2 months?
what does AUC look like at other lead times?"

This script re-runs the walk-forward LSTM portfolio agent of script 06
for multiple lead values (1, 2, 3, 6, 12 months ahead) and produces
the canonical "AUC vs lead time" curve.

To keep runtime under control:
    - Uses the same P4 PyTorch LSTM architecture as 06 (hidden=64, 2 layers)
    - Shorter: 3 seeds instead of 5, 80 epochs instead of 150
    - Same walk-forward: min_train=60, step=6

The LSTM at lead=2 from script 06 is the headline of Table 5; this
script just adds the curve as a supporting robustness appendix figure.

Inputs:
  ../paper4-lstm-agent/src/utils.py                (P4 torch LSTM)
  data/processed/agent_features.csv                (from 06)

Outputs:
  data/processed/lead_time_auc.csv      (lead, model, auc, brier, n_oos)
  results/table_lead_time.csv           (compact table for appendix)
  results/figures/fig_lead_time.pdf/png
"""

import io
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

# P4 torch LSTM
P4_SRC = os.path.abspath("../../paper4-lstm-agent/src")
if P4_SRC not in sys.path:
    sys.path.insert(0, P4_SRC)
from utils import make_sequences, train_ensemble, predict_ensemble  # noqa

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
PROC_DIR    = "../data/processed/"
RESULTS_DIR = "../results/"
FIG_DIR     = RESULTS_DIR + "figures/"
os.makedirs(FIG_DIR, exist_ok=True)

LEADS      = [1, 2, 3, 6, 12]
LOOKBACK   = 6
MIN_TRAIN  = 60
STEP       = 6
SEEDS      = [0, 1, 2]          # 3 seeds for speed
LSTM_KW    = dict(hidden_size=64, n_layers=2, dropout=0.3,
                   epochs=80, batch_size=16, lr=1e-3)

print("=" * 64)
print("Paper 5 — LSTM lead-time curve")
print(f"  leads = {LEADS},  seeds = {len(SEEDS)},  epochs = {LSTM_KW['epochs']}")
print("=" * 64)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════════
print("\nLoading agent features (from 06)")

feats_path = PROC_DIR + "agent_features.csv"
if not os.path.exists(feats_path):
    sys.exit(f"  ✗ Missing {feats_path} — run 06 first")

feat = pd.read_csv(feats_path, index_col=0, parse_dates=True).sort_index()
y = feat["y"].astype(int).to_numpy()
X = feat.drop(columns=["y"]).to_numpy()
dates = feat.index
print(f"  Feature matrix: {X.shape}  |  base rate: {y.mean():.1%}")


# ══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD FOR ONE LEAD
# ══════════════════════════════════════════════════════════════════════════════
def fit_xgb_ensemble(X_tr, y_tr, X_te):
    probs = np.zeros(len(X_te))
    for s in SEEDS:
        m = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=s, verbosity=0,
        )
        m.fit(X_tr, y_tr)
        probs += m.predict_proba(X_te)[:, 1]
    return probs / len(SEEDS)


def walk_forward(lead: int) -> dict:
    """Run walk-forward for a single lead. Return pooled OOS metrics."""
    n = len(X)
    lstm_p  = np.full(n, np.nan)
    xgb_p   = np.full(n, np.nan)
    logit_p = np.full(n, np.nan)

    for end in range(MIN_TRAIN, n - STEP + 1, STEP):
        tr_slice = slice(0, end)
        te_slice = slice(end, min(end + STEP, n))
        y_tr = y[tr_slice]
        if y_tr.sum() < 3 or (len(y_tr) - y_tr.sum()) < 3:
            continue

        scaler = StandardScaler().fit(X[tr_slice])
        X_tr_sc = scaler.transform(X[tr_slice])
        X_te_sc = scaler.transform(X[te_slice])

        # XGBoost
        if HAVE_XGB:
            xgb_p[te_slice] = fit_xgb_ensemble(X_tr_sc, y_tr, X_te_sc)

        # Logistic
        lr = LogisticRegression(max_iter=500, random_state=0, C=1.0)
        lr.fit(X_tr_sc, y_tr)
        logit_p[te_slice] = lr.predict_proba(X_te_sc)[:, 1]

        # LSTM ensemble on sequences
        end_union = min(end + STEP, n)
        X_union = np.vstack([X_tr_sc, X_te_sc])
        y_union = y[:end_union]
        X_seq, y_seq = make_sequences(X_union, y_union, LOOKBACK, lead)
        target_idx = np.arange(LOOKBACK + lead, LOOKBACK + lead + len(y_seq))
        train_mask = target_idx < end
        test_mask  = (target_idx >= end) & (target_idx < end_union)
        if train_mask.sum() < 20 or test_mask.sum() == 0:
            continue
        models = train_ensemble(
            X_seq[train_mask], y_seq[train_mask],
            seeds=SEEDS, verbose=False, **LSTM_KW,
        )
        probs = predict_ensemble(models, X_seq[test_mask])
        for j, ti in enumerate(target_idx[test_mask]):
            if ti < n:
                lstm_p[ti] = probs[j]

    def metrics(p):
        mask = ~np.isnan(p)
        if mask.sum() < 10 or len(np.unique(y[mask])) < 2:
            return {"n": int(mask.sum()), "auc": np.nan, "brier": np.nan}
        return {
            "n":     int(mask.sum()),
            "auc":   float(roc_auc_score(y[mask], p[mask])),
            "brier": float(brier_score_loss(y[mask], p[mask])),
        }

    return {
        "lstm":  metrics(lstm_p),
        "xgb":   metrics(xgb_p),
        "logit": metrics(logit_p),
    }


# ══════════════════════════════════════════════════════════════════════════════
# RUN ALL LEADS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nRunning walk-forward across {len(LEADS)} lead values")

rows = []
for lead in LEADS:
    print(f"\n  lead = {lead:2d} M  ...", flush=True)
    res = walk_forward(lead)
    for model, m in res.items():
        rows.append({"lead": lead, "model": model, **m})
    print(f"    LSTM AUC={res['lstm']['auc']:.3f}  "
          f"XGB AUC={res['xgb']['auc']:.3f}  "
          f"Logit AUC={res['logit']['auc']:.3f}")

curve = pd.DataFrame(rows)
print("\n\nLead-time table:")
print(curve.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ══════════════════════════════════════════════════════════════════════════════
# SAVE TABLE + FIGURE
# ══════════════════════════════════════════════════════════════════════════════
print("\nSaving")

curve.to_csv(PROC_DIR + "lead_time_auc.csv", index=False)
curve.to_csv(RESULTS_DIR + "table_lead_time.csv", index=False)

# Plot
fig, ax = plt.subplots(figsize=(7, 4.2))
colors = {"lstm": "#E74C3C", "xgb": "#27AE60", "logit": "#2874A6"}
for model in ("lstm", "xgb", "logit"):
    sub = curve[curve["model"] == model].sort_values("lead")
    ax.plot(sub["lead"], sub["auc"], marker="o", lw=1.6,
             color=colors[model], label=model.upper())
ax.axhline(0.5, ls="--", color="#7F8C8D", lw=0.8, label="chance")
ax.axhline(0.926, ls=":", color="#F39C12", lw=1.0, label="P4 ref (0.926)")
ax.set_xticks(LEADS)
ax.set_xlabel("Lead time (months)")
ax.set_ylabel("Pooled OOS AUC")
ax.set_title("Figure — LSTM portfolio agent lead-time curve", fontweight="bold")
ax.set_ylim(0.3, 1.02)
ax.grid(True, alpha=0.3)
ax.legend(loc="lower left", ncol=2)
fig.tight_layout()

for ext in ("pdf", "png"):
    fig.savefig(FIG_DIR + f"fig_lead_time.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"  Saved: {PROC_DIR}lead_time_auc.csv")
print(f"  Saved: {RESULTS_DIR}table_lead_time.csv")
print(f"  Saved: {FIG_DIR}fig_lead_time.{{pdf,png}}")

print("\n" + "=" * 64)
print("DONE — Lead-time curve complete")
print("=" * 64)
best = curve[curve["model"] == "lstm"].sort_values("auc", ascending=False).iloc[0]
print(f"  Best LSTM lead: {int(best['lead'])} M  (AUC = {best['auc']:.3f})")
print("=" * 64)
