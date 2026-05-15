"""
Paper 5 — LSTM Portfolio Agent  (P4 logic at portfolio level)
===================================
Robustness / generalization check: extend P4's LSTM ensemble from predicting
a SINGLE material (GR_Fuel_Energy) crisis to predicting the WHOLE construction
PORTFOLIO regime produced by 03_dynamic_copula.py.

DUAL-TARGET EVALUATION (Option C):
  1. Endogenous target — top 25% months by network-average λ_U.
     Expected AUC ≈ 1.0 (features ← same ELSTAT prices → regime label).
     Reported as an upper bound / sanity check.
  2. Exogenous target — Greek macro crisis calendar (2012 sovereign,
     2015 capital controls, 2022 Ukraine energy shock).
     This is the FAIR metric: features cannot trivially reconstruct
     external macro events.

P4 setup (replicated exactly — same PyTorch code via sys.path import):
    LSTMClassifier(hidden=64, layers=2, dropout=0.3)
    LOOKBACK=6,  LEAD=2,  EPOCHS=150,  BATCH=16,  5-seed ensemble
    P4 headline:  AUC = 0.926 on GR_Fuel_Energy crisis, lead=2M

Input features (24 total):
    - P4's 20 US PPI features        (ret / vol3 / vol6 / mom3 for 5 commodities)
    - 4 P5 portfolio features        (network_avg_lambdaU, network_avg_tau,
                                       max_CI_type, mean_CI)

Benchmarks (all via walk-forward, min_train=60, step=6, lead=2):
    1. LSTM ensemble       — real P4 architecture, 5 seeds  (main model)
    2. XGBoost ensemble    — non-neural benchmark, 5 seeds
    3. Logistic regression — linear baseline
    4. P4 signal only      — evaluate P4's published wf_predictions.csv
                              against the P5 portfolio target (no retraining)
"""

import io
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

# Import P4's actual PyTorch LSTM utilities (same architecture as P4 paper)
P4_SRC = os.path.abspath("../../paper4-lstm-agent/src")
if P4_SRC not in sys.path:
    sys.path.insert(0, P4_SRC)
from utils import LSTMClassifier, make_sequences, train_ensemble, predict_ensemble  # noqa

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS  (mirror P4 exactly)
# ══════════════════════════════════════════════════════════════════════════════
PROC_DIR    = "../data/processed/"
RESULTS_DIR = "../results/"
P4_FEATURES_PATH = "../../paper4-lstm-agent/data/processed/features.csv"
P4_WF_PATH       = "../../paper4-lstm-agent/results/wf_predictions.csv"

LEAD         = 2          # predict t+LEAD portfolio regime
LOOKBACK     = 6          # P4 setting
MIN_TRAIN    = 60         # minimum training months
STEP         = 6          # walk-forward step

LSTM_KWARGS = dict(
    hidden_size=64,
    n_layers=2,
    dropout=0.3,
    epochs=150,
    batch_size=16,
    lr=1e-3,
)
SEEDS = [0, 1, 2, 3, 4]
N_SEEDS = len(SEEDS)

# Exogenous crisis windows (Greek macro events)
EXO_CRISIS_WINDOWS = [
    ("2011-07", "2012-12"),   # 2012 sovereign / PSI
    ("2015-06", "2015-12"),   # 2015 capital controls
    ("2022-02", "2022-12"),   # 2022 Ukraine energy shock
]

os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 64)
print(f"Paper 5 — LSTM Portfolio Agent  (lookback={LOOKBACK}, lead={LEAD}, "
      f"{N_SEEDS}-seed ensemble)")
print("  DUAL-TARGET: endogenous (λ_U quantile) + exogenous (macro calendar)")
print("=" * 64)


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD FEATURES + TARGETS
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 1: Loading features + targets")

if not os.path.exists(P4_FEATURES_PATH):
    sys.exit(f"  ✗ Missing P4 features: {P4_FEATURES_PATH}")

feat_p4 = pd.read_csv(P4_FEATURES_PATH, parse_dates=["Date"]).set_index("Date").sort_index()
print(f"  P4 features: {feat_p4.shape}  "
      f"({feat_p4.index.min():%Y-%m} → {feat_p4.index.max():%Y-%m})")

regimes = pd.read_csv(os.path.join(PROC_DIR, "dynamic_copula_regimes.csv"),
                      index_col=0, parse_dates=True).sort_index()
network = pd.read_csv(os.path.join(PROC_DIR, "dynamic_copula_network_avg.csv"),
                      index_col=0, parse_dates=True).sort_index()
CI = pd.read_csv(os.path.join(PROC_DIR, "contagion_index.csv"),
                 index_col=0, parse_dates=True).sort_index()

# --- Endogenous target (top 25% λ_U) ---
y_endo = (regimes["regime"] == "crisis").astype(int)
print(f"  Endogenous target: {y_endo.sum()}/{len(y_endo)} crisis months "
      f"(base rate {y_endo.mean():.1%})")

# --- Exogenous target (Greek macro calendar) ---
y_exo = pd.Series(0, index=regimes.index, name="crisis_exo")
for start, end in EXO_CRISIS_WINDOWS:
    mask = (y_exo.index >= pd.Timestamp(start)) & (y_exo.index <= pd.Timestamp(end))
    y_exo[mask] = 1
print(f"  Exogenous target:  {y_exo.sum()}/{len(y_exo)} crisis months "
      f"(base rate {y_exo.mean():.1%})")


# ══════════════════════════════════════════════════════════════════════════════
# 2. BUILD AGENT FEATURE MATRIX
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 2: Merging P4 features with P5 portfolio features")

portfolio_features = pd.DataFrame({
    "network_avg_lambdaU": network["avg_lambdaU"],
    "network_avg_tau":     network["avg_tau"],
    "max_CI_type":         CI.max(axis=1),
    "mean_CI":             CI.mean(axis=1),
})

X_full = feat_p4.join(portfolio_features, how="inner").dropna()
y_endo_aligned = y_endo.reindex(X_full.index).dropna().astype(int)
y_exo_aligned  = y_exo.reindex(X_full.index).dropna().astype(int)

# Use intersection of both targets
common_idx = y_endo_aligned.index.intersection(y_exo_aligned.index)
X_full = X_full.loc[common_idx]
y_endo_aligned = y_endo_aligned.loc[common_idx]
y_exo_aligned  = y_exo_aligned.loc[common_idx]

print(f"  Merged feature matrix: {X_full.shape}")
print(f"  Endogenous crisis months in feature range: {y_endo_aligned.sum()}")
print(f"  Exogenous  crisis months in feature range: {y_exo_aligned.sum()}")

X_full.join(y_endo_aligned.rename("y_endo")).join(y_exo_aligned.rename("y_exo")).to_csv(
    os.path.join(PROC_DIR, "agent_features.csv"))


# ══════════════════════════════════════════════════════════════════════════════
# 3. WALK-FORWARD VALIDATION (reusable for both targets)
# ══════════════════════════════════════════════════════════════════════════════
def run_walkforward(X_full_df, y_series, target_label):
    """Run full walk-forward with LSTM + XGB + Logit. Returns probs dict."""
    print(f"\n{'─'*64}")
    print(f"  Walk-forward: TARGET = {target_label}")
    print(f"  min_train={MIN_TRAIN}, step={STEP}, lookback={LOOKBACK}")
    print(f"{'─'*64}")

    X_arr = X_full_df.to_numpy()
    y_arr = y_series.to_numpy()

    lstm_probs  = np.full(len(X_arr), np.nan)
    xgb_probs   = np.full(len(X_arr), np.nan)
    logit_probs = np.full(len(X_arr), np.nan)

    n_windows = 0
    for end in range(MIN_TRAIN, len(X_arr) - STEP + 1, STEP):
        tr_idx = slice(0, end)
        te_idx = slice(end, min(end + STEP, len(X_arr)))

        y_tr_flat = y_arr[tr_idx]
        if y_tr_flat.sum() < 3 or (len(y_tr_flat) - y_tr_flat.sum()) < 3:
            continue

        # --- Standardise ---
        scaler = StandardScaler().fit(X_arr[tr_idx])
        X_tr_sc = scaler.transform(X_arr[tr_idx])
        X_te_sc = scaler.transform(X_arr[te_idx])

        # --- XGBoost (flat features) ---
        if HAVE_XGB:
            probs = np.zeros(len(X_te_sc))
            for s in SEEDS:
                m = XGBClassifier(
                    n_estimators=200, max_depth=3, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    eval_metric="logloss", random_state=s, verbosity=0,
                )
                m.fit(X_tr_sc, y_tr_flat)
                probs += m.predict_proba(X_te_sc)[:, 1]
            xgb_probs[te_idx] = probs / N_SEEDS

        # --- Logistic (flat features) ---
        lr = LogisticRegression(max_iter=500, random_state=0, C=1.0)
        lr.fit(X_tr_sc, y_tr_flat)
        logit_probs[te_idx] = lr.predict_proba(X_te_sc)[:, 1]

        # --- LSTM ensemble (sequences) ---
        end_union = min(end + STEP, len(X_arr))
        X_union = np.vstack([X_tr_sc, X_te_sc])
        y_union = y_arr[:end_union]
        X_seq, y_seq = make_sequences(X_union, y_union, LOOKBACK, LEAD)

        target_idx = np.arange(LOOKBACK + LEAD, LOOKBACK + LEAD + len(y_seq))
        train_mask = target_idx < end
        test_mask  = (target_idx >= end) & (target_idx < end_union)

        if train_mask.sum() < 20 or test_mask.sum() == 0:
            n_windows += 1
            continue

        models = train_ensemble(
            X_seq[train_mask], y_seq[train_mask],
            seeds=SEEDS, verbose=False, **LSTM_KWARGS,
        )
        probs = predict_ensemble(models, X_seq[test_mask])
        for j, ti in enumerate(target_idx[test_mask]):
            if ti < len(lstm_probs):
                lstm_probs[ti] = probs[j]

        n_windows += 1

    print(f"  Completed {n_windows} walk-forward windows")
    return {
        "LSTM (P4 arch, ens=5)": lstm_probs,
        "XGBoost (ens=5)":       xgb_probs,
        "LogReg":                logit_probs,
    }


def pooled_metrics(probs, truth):
    mask = ~np.isnan(probs)
    if mask.sum() < 10 or len(np.unique(truth[mask])) < 2:
        return {"n_oos": int(mask.sum()), "auc": np.nan, "brier": np.nan}
    return {
        "n_oos": int(mask.sum()),
        "auc":   float(roc_auc_score(truth[mask], probs[mask])),
        "brier": float(brier_score_loss(truth[mask], probs[mask])),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. RUN BOTH TARGETS
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 3: Running walk-forward for BOTH targets")

# --- Endogenous ---
endo_probs = run_walkforward(X_full, y_endo_aligned, "endogenous (λ_U quantile)")

# --- Exogenous ---
exo_probs = run_walkforward(X_full, y_exo_aligned, "exogenous (macro calendar)")


# ══════════════════════════════════════════════════════════════════════════════
# 5. EVALUATE  (pooled OOS AUC)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("STEP 4: Pooled out-of-sample metrics (Table 5)")
print("=" * 64)

table5_rows = []

# Endogenous results
y_arr_endo = y_endo_aligned.to_numpy()
for name, pr in endo_probs.items():
    m = pooled_metrics(pr, y_arr_endo)
    table5_rows.append({"target": "endogenous", "model": name, **m})

# Exogenous results
y_arr_exo = y_exo_aligned.to_numpy()
for name, pr in exo_probs.items():
    m = pooled_metrics(pr, y_arr_exo)
    table5_rows.append({"target": "exogenous", "model": name, **m})

# --- P4 baseline: evaluate P4's own wf_predictions against BOTH targets ---
if os.path.exists(P4_WF_PATH):
    p4wf = pd.read_csv(P4_WF_PATH, parse_dates=["date"]).set_index("date").sort_index()
    p4wf = p4wf[~p4wf.index.duplicated(keep="first")]

    for tgt_name, y_tgt in [("endogenous", y_endo_aligned), ("exogenous", y_exo_aligned)]:
        y_nodup = y_tgt[~y_tgt.index.duplicated(keep="first")]
        common = p4wf.index.intersection(y_nodup.index)
        if len(common) > 20:
            y_al = y_nodup.loc[common].to_numpy()
            p4_al = p4wf.loc[common, "prob"].to_numpy()
            mask = ~np.isnan(p4_al)
            y_al, p4_al = y_al[mask], p4_al[mask]
            if len(np.unique(y_al)) >= 2:
                table5_rows.append({
                    "target": tgt_name,
                    "model":  "P4 signal only (lead=2)",
                    "n_oos":  int(len(y_al)),
                    "auc":    float(roc_auc_score(y_al, p4_al)),
                    "brier":  float(brier_score_loss(y_al, p4_al)),
                })

table5 = pd.DataFrame(table5_rows).sort_values(["target", "auc"], ascending=[True, False])

print("\n--- Endogenous target (λ_U quantile — upper bound, expected ~1.0) ---")
endo_rows = table5[table5["target"] == "endogenous"]
print(endo_rows.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\n--- Exogenous target (macro calendar — FAIR metric) ---")
exo_rows = table5[table5["target"] == "exogenous"]
print(exo_rows.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ══════════════════════════════════════════════════════════════════════════════
# 6. SAVE
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 5: Saving")

dates = X_full.index

# Save predictions for both targets
pred_df = pd.DataFrame({"date": dates})
for name_key, short in [("LSTM (P4 arch, ens=5)", "lstm"),
                         ("XGBoost (ens=5)", "xgb"),
                         ("LogReg", "logit")]:
    pred_df[f"{short}_endo"] = endo_probs[name_key]
    pred_df[f"{short}_exo"]  = exo_probs[name_key]
pred_df["actual_endo"] = y_arr_endo
pred_df["actual_exo"]  = y_arr_exo
pred_df.to_csv(os.path.join(PROC_DIR, "agent_predictions.csv"), index=False)

# Walk-forward AUC (keep backward compat column names for 08)
wf_df = pd.DataFrame({
    "date":   dates,
    "lstm":   endo_probs["LSTM (P4 arch, ens=5)"],
    "xgb":    endo_probs["XGBoost (ens=5)"],
    "logit":  endo_probs["LogReg"],
    "actual": y_arr_endo,
})
wf_df.to_csv(os.path.join(PROC_DIR, "agent_walkforward_auc.csv"), index=False)

table5.to_csv(os.path.join(RESULTS_DIR, "table5_lstm_portfolio.csv"), index=False)
print(f"  Saved: {PROC_DIR}agent_predictions.csv")
print(f"  Saved: {PROC_DIR}agent_walkforward_auc.csv")
print(f"  Saved: {RESULTS_DIR}table5_lstm_portfolio.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 7. FINAL
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("DONE — Portfolio LSTM agent trained & validated (DUAL TARGET)")
print("=" * 64)

# Report exogenous as the main metric
exo_best = exo_rows.iloc[0] if len(exo_rows) > 0 else None
endo_best = endo_rows.iloc[0] if len(endo_rows) > 0 else None

if endo_best is not None:
    print(f"  Endogenous (upper bound): {endo_best['model']:35s}  AUC = {endo_best['auc']:.3f}")
if exo_best is not None:
    print(f"  Exogenous  (fair metric): {exo_best['model']:35s}  AUC = {exo_best['auc']:.3f}")
print(f"  P4 reference (GR_Fuel_Energy, lead=2): AUC = 0.926")
print(f"\nNext: run 07_systemic_risk_index.py")
print("=" * 64)
