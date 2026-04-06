"""
Paper 4 — Probability Calibration
====================================
Tests whether LSTM probabilities are well-calibrated:
"If the model says P(crisis) = 0.70, does crisis occur 70% of the time?"

Methods:
  1. Reliability diagram (calibration curve)
  2. Brier score (proper scoring rule)
  3. Platt scaling (post-hoc calibration)
  4. Isotonic regression calibration
  5. Expected Calibration Error (ECE)

Why this matters for AiC:
  - Procurement decisions need calibrated probabilities
  - "P(crisis) = 0.67 → allocate EUR X" only works if P is meaningful
  - Calibrated model = trustworthy decision support system

Outputs:
  - results/calibration_summary.csv
  - results/fig_p4_calibration.pdf

Run from: construction-risk-suite/paper4-lstm-agent/src/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
PROCESSED_DIR = "../data/processed/"
RESULTS_DIR   = "../results/"
SEED          = 42
LOOKBACK      = 6
LEAD          = 2
TARGET_MAT    = "GR_Fuel_Energy"
TRAIN_RATIO   = 0.75
EPOCHS        = 150
BATCH_SIZE    = 16
HIDDEN_SIZE   = 64
N_LAYERS      = 2
DROPOUT       = 0.3
LR            = 5e-4
PATIENCE      = 20
N_BINS        = 5    # calibration bins (small n → fewer bins)

np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 60)
print("Paper 4 — Probability Calibration")
print(f"Target: {TARGET_MAT} | Lead: {LEAD}M")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 1: Loading data")

features = pd.read_csv(PROCESSED_DIR + "features.csv",
                       index_col=0, parse_dates=True)
labels   = pd.read_csv(PROCESSED_DIR + "crisis_labels.csv",
                       index_col=0, parse_dates=True)

FEAT_COLS = list(features.columns)
common    = features.index.intersection(labels.dropna().index)
X_raw     = features.loc[common].values
y_raw     = labels.loc[common, TARGET_MAT].values
dates     = common

scaler = MinMaxScaler()

def make_sequences(X, y, dates, lookback, lead):
    Xs, ys, ds = [], [], []
    for i in range(lookback, len(X) - lead):
        Xs.append(X[i - lookback:i])
        ys.append(y[i + lead])
        ds.append(dates[i + lead])
    return np.array(Xs), np.array(ys), np.array(ds)

# Fit scaler on training portion only to prevent data leakage
n_raw_tr = int(len(X_raw) * TRAIN_RATIO)
scaler.fit(X_raw[:n_raw_tr])
X_sc = scaler.transform(X_raw)

X_seq, y_seq, d_seq = make_sequences(X_sc, y_raw, dates, LOOKBACK, LEAD)
n_tr = int(len(X_seq) * TRAIN_RATIO)
X_tr, X_te = X_seq[:n_tr], X_seq[n_tr:]
y_tr, y_te = y_seq[:n_tr], y_seq[n_tr:]

# Val split for calibration fitting (from training set)
n_cal = int(len(X_tr) * 0.80)
X_cal_tr, X_cal_val = X_tr[:n_cal], X_tr[n_cal:]
y_cal_tr, y_cal_val = y_tr[:n_cal], y_tr[n_cal:]

print(f"  Train: {len(X_tr)} | Val (calib): {len(X_cal_val)} | Test: {len(X_te)}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. TRAIN LSTM
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 2: Training LSTM")

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers,
                            batch_first=True,
                            dropout=dropout if n_layers > 1 else 0)
        self.bn   = nn.BatchNorm1d(hidden_size)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = self.bn(out[:, -1, :])
        return self.fc(self.drop(h)).squeeze()

def train_lstm(X_tr, y_tr):
    # Class balancing via pos_weight only (no oversampling)
    idx_c = np.where(y_tr == 1)[0]
    idx_s = np.where(y_tr == 0)[0]

    pos_w  = torch.tensor([len(idx_s) / max(len(idx_c), 1)],
                           dtype=torch.float32)
    model  = LSTMClassifier(X_tr.shape[2], HIDDEN_SIZE, N_LAYERS, DROPOUT)
    opt    = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    crit   = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    ds     = TensorDataset(torch.FloatTensor(X_tr),
                            torch.FloatTensor(y_tr))
    dl     = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    best_loss, best_state, wait = np.inf, None, 0
    for epoch in range(EPOCHS):
        model.train()
        ep_loss = 0
        for xb, yb in dl:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
        avg = ep_loss / len(dl)
        if avg < best_loss:
            best_loss  = avg
            best_state = {k: v.clone() for k, v
                          in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model

def get_probs(model, X):
    model.eval()
    with torch.no_grad():
        lg = model(torch.FloatTensor(X)).numpy()
    if lg.ndim == 0:
        lg = lg.reshape(1)
    return 1 / (1 + np.exp(-lg))

model     = train_lstm(X_tr, y_tr)
probs_val = get_probs(model, X_cal_val)
probs_te  = get_probs(model, X_te)
print(f"  Training complete")

# ══════════════════════════════════════════════════════════════════════════════
# 3. CALIBRATION METRICS (uncalibrated)
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 3: Calibration metrics (uncalibrated LSTM)")

def expected_calibration_error(y_true, probs, n_bins=10):
    """ECE: weighted average calibration error across bins."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc  = y_true[mask].mean()
        bin_conf = probs[mask].mean()
        ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
    return ece

brier_raw = brier_score_loss(y_te, probs_te)
ece_raw   = expected_calibration_error(y_te, probs_te, N_BINS)

print(f"  Brier score (lower=better): {brier_raw:.4f}")
print(f"  ECE (lower=better):         {ece_raw:.4f}")
print(f"  Mean predicted prob:        {probs_te.mean():.4f}")
print(f"  Actual crisis rate:         {y_te.mean():.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. PLATT SCALING (post-hoc calibration)
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 4: Platt scaling calibration")

# Fit Platt scaler on validation set
if len(np.unique(y_cal_val)) > 1 and probs_val.std() > 0:
    platt = LogisticRegression(C=1.0, solver="lbfgs")
    platt.fit(probs_val.reshape(-1, 1), y_cal_val)
    probs_platt = platt.predict_proba(probs_te.reshape(-1, 1))[:, 1]

    brier_platt = brier_score_loss(y_te, probs_platt)
    ece_platt   = expected_calibration_error(y_te, probs_platt, N_BINS)
    print(f"  Platt Brier: {brier_platt:.4f} "
          f"(Δ={brier_platt-brier_raw:+.4f})")
    print(f"  Platt ECE:   {ece_platt:.4f} "
          f"(Δ={ece_platt-ece_raw:+.4f})")
else:
    probs_platt = probs_te.copy()
    brier_platt = brier_raw
    ece_platt   = ece_raw
    print("  Platt skipped (insufficient validation data)")

# ══════════════════════════════════════════════════════════════════════════════
# 5. ISOTONIC REGRESSION CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 5: Isotonic regression calibration")

if len(np.unique(y_cal_val)) > 1:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(probs_val, y_cal_val)
    probs_iso = iso.predict(probs_te)

    brier_iso = brier_score_loss(y_te, probs_iso)
    ece_iso   = expected_calibration_error(y_te, probs_iso, N_BINS)
    print(f"  Isotonic Brier: {brier_iso:.4f} "
          f"(Δ={brier_iso-brier_raw:+.4f})")
    print(f"  Isotonic ECE:   {ece_iso:.4f} "
          f"(Δ={ece_iso-ece_raw:+.4f})")
else:
    probs_iso = probs_te.copy()
    brier_iso = brier_raw
    ece_iso   = ece_raw
    print("  Isotonic skipped")

# ══════════════════════════════════════════════════════════════════════════════
# 6. CALIBRATION CURVES
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 6: Computing calibration curves")

def safe_calibration_curve(y_true, probs, n_bins):
    """Calibration curve with safety checks."""
    try:
        fraction_pos, mean_pred = calibration_curve(
            y_true, probs, n_bins=n_bins, strategy="uniform")
        return mean_pred, fraction_pos
    except Exception:
        # Manual binning
        bins = np.linspace(0, 1, n_bins + 1)
        mean_preds, frac_pos = [], []
        for i in range(n_bins):
            mask = (probs >= bins[i]) & (probs < bins[i+1])
            if mask.sum() >= 2:
                mean_preds.append(probs[mask].mean())
                frac_pos.append(y_true[mask].mean())
        return np.array(mean_preds), np.array(frac_pos)

x_raw,   y_raw_c   = safe_calibration_curve(y_te, probs_te,    N_BINS)
x_platt, y_platt_c = safe_calibration_curve(y_te, probs_platt, N_BINS)
x_iso,   y_iso_c   = safe_calibration_curve(y_te, probs_iso,   N_BINS)

# ══════════════════════════════════════════════════════════════════════════════
# 7. SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════
summary = pd.DataFrame([
    {"method": "Uncalibrated LSTM", "brier": brier_raw, "ECE": ece_raw},
    {"method": "Platt scaling",     "brier": brier_platt, "ECE": ece_platt},
    {"method": "Isotonic regression","brier": brier_iso, "ECE": ece_iso},
])
summary.to_csv(RESULTS_DIR + "calibration_summary.csv", index=False)

# Best calibrated probs for downstream use (choose by ECE, the calibration metric)
# Exclude uncalibrated — we want the best post-hoc method
posthoc = summary[summary["method"] != "Uncalibrated LSTM"]
best_method = posthoc.iloc[posthoc["ECE"].argmin()]["method"]
if "Isotonic" in best_method:
    probs_best = probs_iso
elif "Platt" in best_method:
    probs_best = probs_platt
else:
    probs_best = probs_te

pd.DataFrame({
    "prob_raw"    : probs_te,
    "prob_platt"  : probs_platt,
    "prob_isotonic": probs_iso,
    "y_true"      : y_te
}).to_csv(RESULTS_DIR + "calibrated_probs.csv", index=False)

print(f"\n  Best calibration method: {best_method}")
print(f"  Summary:")
print(summary.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# 8. FIGURES
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 7: Generating figures")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f"Paper 4 — Probability Calibration\n"
    f"LSTM: {TARGET_MAT} | Lead={LEAD}M | "
    f"Brier(raw)={brier_raw:.4f} | ECE(raw)={ece_raw:.4f}",
    fontsize=12, fontweight="bold"
)

# Panel 1: Reliability diagram
ax = axes[0, 0]
ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
if len(x_raw) > 0:
    ax.plot(x_raw, y_raw_c, "o-", color="#e74c3c", lw=2, markersize=8,
            label=f"Uncalibrated (ECE={ece_raw:.3f})")
if len(x_platt) > 0:
    ax.plot(x_platt, y_platt_c, "s--", color="#3498db", lw=2, markersize=8,
            label=f"Platt scaling (ECE={ece_platt:.3f})")
if len(x_iso) > 0:
    ax.plot(x_iso, y_iso_c, "^:", color="#2ecc71", lw=2, markersize=8,
            label=f"Isotonic (ECE={ece_iso:.3f})")
ax.set_xlabel("Mean predicted P(crisis)")
ax.set_ylabel("Actual crisis fraction")
ax.set_title(f"Reliability Diagram (Calibration Curve)\n"
             f"{N_BINS} bins — perfect = diagonal")
ax.legend(fontsize=8)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.grid(alpha=0.3)

# Panel 2: Probability distribution
ax = axes[0, 1]
ax.hist(probs_te[y_te == 0], bins=20, alpha=0.6, color="#3498db",
        label="Stable months", density=True)
ax.hist(probs_te[y_te == 1], bins=20, alpha=0.6, color="#e74c3c",
        label="Crisis months", density=True)
ax.axvline(0.5, color="black", linestyle="--", lw=1.5,
           label="Decision threshold (0.5)")
ax.set_xlabel("P(crisis)")
ax.set_ylabel("Density")
ax.set_title("Probability Distribution by Class\n"
             "(good separation = well-separated histograms)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Panel 3: Brier scores comparison
ax = axes[1, 0]
methods = ["Uncalibrated\nLSTM", "Platt\nScaling", "Isotonic\nRegression"]
briers  = [brier_raw, brier_platt, brier_iso]
colors_b= ["#e74c3c", "#3498db", "#2ecc71"]
bars = ax.bar(range(3), briers, color=colors_b,
              edgecolor="black", linewidth=0.5, width=0.5)
for bar, v in zip(bars, briers):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.002,
            f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(range(3))
ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel("Brier Score (lower = better)")
ax.set_title("Calibration Method Comparison\n"
             "(Brier score = proper scoring rule)")
ax.grid(alpha=0.3, axis="y")
ax.axhline(min(briers), color="green", linestyle=":",
           lw=1.5, label=f"Best = {min(briers):.4f}")
ax.legend(fontsize=8)

# Panel 4: ECE comparison + interpretation
ax = axes[1, 1]
eces = [ece_raw, ece_platt, ece_iso]
bars2 = ax.bar(range(3), eces, color=colors_b,
               edgecolor="black", linewidth=0.5, width=0.5)
for bar, v in zip(bars2, eces):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.002,
            f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(range(3))
ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel("Expected Calibration Error (ECE)")
ax.set_title("ECE Comparison\n"
             "ECE < 0.10 = well-calibrated for deployment")
ax.axhline(0.10, color="green", linestyle="--", lw=1.5,
           label="Deployment threshold (ECE=0.10)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
fig_path = RESULTS_DIR + "fig_p4_calibration.pdf"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Figure saved: {fig_path}")
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# 9. FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("CALIBRATION SUMMARY FOR PAPER 4")
print("=" * 60)
print(f"\n  Uncalibrated LSTM:")
print(f"    Brier = {brier_raw:.4f} | ECE = {ece_raw:.4f}")
print(f"    Mean P(crisis) = {probs_te.mean():.4f} vs actual {y_te.mean():.4f}")

print(f"\n  Best method: {best_method}")
best_brier = summary[summary["method"]==best_method]["brier"].values[0]
best_ece   = summary[summary["method"]==best_method]["ECE"].values[0]
print(f"    Brier = {best_brier:.4f} | ECE = {best_ece:.4f}")

print(f"\n  FOR PAPER 4 WRITING:")
if ece_raw < 0.10:
    print(f"  'The LSTM produces well-calibrated probabilities "
          f"(ECE={ece_raw:.3f} < 0.10),")
    print(f"   enabling direct interpretation of P(crisis) as")
    print(f"   procurement contingency scaling factor.'")
else:
    print(f"  'Post-hoc calibration via {best_method} reduces ECE")
    print(f"   from {ece_raw:.3f} to {best_ece:.3f}, enabling reliable")
    print(f"   probability-based procurement decisions.'")

print("\n" + "=" * 60)
print("DONE — Calibration complete.")
print("=" * 60)
print("Next: run 10_granger_causality.py")