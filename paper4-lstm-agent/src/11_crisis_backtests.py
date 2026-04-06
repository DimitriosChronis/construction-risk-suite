"""
Paper 4 -- Crisis Episode Backtests (v2 -- ensemble + selected features)
==========================================================================
THE KILLER RESULT for Paper 4.

Tests whether the LSTM ensemble, trained ONLY on pre-crisis data,
would have detected the two major crises BEFORE they appeared
in Greek construction cost data:

  Episode 1: Global Financial Crisis (GFC) 2008-2009
    - Train: 2000-2006 only
    - Test:  2007-2010

  Episode 2: COVID-19 Commodity Shock 2021-2022
    - Train: 2000-2018 only
    - Test:  2019-2023

v2 changes:
- Uses utils.py (ensemble, helpers)
- 5-seed ensemble per episode
- SHAP-selected features (14 of 20)
- Fixed Unicode for Windows cp1253

Outputs:
  - results/crisis_backtest_summary.csv
  - results/fig_p4_crisis_backtests.pdf

Run from: construction-risk-suite/paper4-lstm-agent/src/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, f1_score

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
ALERT_THRESH  = 0.5

ENSEMBLE_SEEDS = DEFAULT_SEEDS
USE_SELECTED_FEATURES = False  # all 20 features (ablation showed full set is better)

# Crisis episodes
EPISODES = {
    "GFC 2008": {
        "train_end"  : "2006-12-01",
        "test_start" : "2007-01-01",
        "test_end"   : "2010-12-01",
        "crisis_peak": "2008-10-01",
        "color"      : "#e74c3c",
        "label"      : "GFC 2008-2009",
    },
    "COVID 2021": {
        "train_end"  : "2018-12-01",
        "test_start" : "2019-01-01",
        "test_end"   : "2023-12-01",
        "crisis_peak": "2021-06-01",
        "color"      : "#e67e22",
        "label"      : "COVID Commodity Shock 2021-2022",
    },
}

np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 60)
print("Paper 4 -- Crisis Episode Backtests (Ensemble)")
print(f"Target: {TARGET_MAT} | Lead: {LEAD}M | Seeds: {len(ENSEMBLE_SEEDS)}")
print("=" * 60)

# ==============================================================================
# 1. LOAD DATA
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

labels = pd.read_csv(PROCESSED_DIR + "crisis_labels.csv",
                     index_col=0, parse_dates=True)

FEAT_COLS = list(features.columns)
common    = features.index.intersection(labels.dropna().index)
X_all     = features.loc[common].values
y_all     = labels.loc[common, TARGET_MAT].values
dates_all = common

print(f"  Features : {len(FEAT_COLS)}")
print(f"  Period   : {dates_all[0].strftime('%Y-%m')} -> "
      f"{dates_all[-1].strftime('%Y-%m')}")

# ==============================================================================
# 2. RUN CRISIS BACKTESTS
# ==============================================================================
print("\nSTEP 2: Running crisis backtests")
print("-" * 60)

results = {}

for ep_name, ep in EPISODES.items():
    print(f"\n{'='*50}")
    print(f"Episode: {ep['label']}")
    print(f"  Train  : 2000 -> {ep['train_end'][:7]}")
    print(f"  Test   : {ep['test_start'][:7]} -> {ep['test_end'][:7]}")
    print(f"  Crisis peak: {ep['crisis_peak'][:7]}")

    # Split
    train_mask = dates_all <= ep["train_end"]
    test_mask  = ((dates_all >= ep["test_start"]) &
                  (dates_all <= ep["test_end"]))

    X_tr_raw = X_all[train_mask]
    y_tr_raw = y_all[train_mask]
    X_te_raw = X_all[test_mask]
    y_te_raw = y_all[test_mask]
    d_te     = dates_all[test_mask]

    print(f"  Train samples: {len(X_tr_raw)} | "
          f"Train crisis: {int(y_tr_raw.sum())}")
    print(f"  Test samples : {len(X_te_raw)} | "
          f"Test crisis : {int(y_te_raw.sum())}")

    # Scale
    scaler = MinMaxScaler()
    scaler.fit(X_tr_raw)
    X_tr_sc = scaler.transform(X_tr_raw)
    X_te_sc = scaler.transform(X_te_raw)

    # Sequences
    X_tr_seq, y_tr_seq, _ = make_sequences(
        X_tr_sc, y_tr_raw, LOOKBACK, LEAD, dates=dates_all[train_mask])
    X_te_seq, y_te_seq, d_te_seq = make_sequences(
        X_te_sc, y_te_raw, LOOKBACK, LEAD, dates=d_te)

    if len(y_te_seq) == 0:
        print("  SKIPPED: no test sequences")
        continue

    # Train ensemble
    print(f"  Training {len(ENSEMBLE_SEEDS)}-seed ensemble on pre-crisis data...")
    models = train_ensemble(X_tr_seq, y_tr_seq, seeds=ENSEMBLE_SEEDS,
                            epochs=EPOCHS, batch_size=BATCH_SIZE,
                            hidden_size=HIDDEN_SIZE, n_layers=N_LAYERS,
                            dropout=DROPOUT, lr=LR, patience=PATIENCE,
                            verbose=True)
    probs = predict_ensemble(models, X_te_seq)
    alerts = (probs > ALERT_THRESH).astype(int)

    # -- Find first alert before crisis peak -----------------------------------
    peak_date  = pd.Timestamp(ep["crisis_peak"])
    alert_dates = d_te_seq[alerts == 1]
    early_alerts = [d for d in alert_dates
                    if pd.Timestamp(str(d)) < peak_date]

    if early_alerts:
        first_alert = min([pd.Timestamp(str(d)) for d in early_alerts])
        lead_months = (peak_date.year - first_alert.year) * 12 + \
                      (peak_date.month - first_alert.month)
        print(f"\n  * FIRST ALERT: {first_alert.strftime('%Y-%m')}")
        print(f"  * CRISIS PEAK: {peak_date.strftime('%Y-%m')}")
        print(f"  * LEAD TIME  : {lead_months} months before peak")
    else:
        first_alert = None
        lead_months = 0
        print(f"  X No early alert detected before crisis peak")

    # Metrics
    if y_te_seq.sum() > 0 and y_te_seq.sum() < len(y_te_seq):
        auc = roc_auc_score(y_te_seq, probs)
        f1  = f1_score(y_te_seq, alerts, zero_division=0)
        rec = (alerts[y_te_seq == 1] == 1).mean()
        print(f"  AUC={auc:.3f} | F1={f1:.3f} | Recall={rec:.3f}")
    else:
        auc = f1 = rec = np.nan
        print(f"  No crisis labels in test period")

    results[ep_name] = {
        "episode"     : ep_name,
        "label"       : ep["label"],
        "train_end"   : ep["train_end"],
        "test_period" : f"{ep['test_start'][:7]}->{ep['test_end'][:7]}",
        "crisis_peak" : ep["crisis_peak"],
        "first_alert" : first_alert.strftime("%Y-%m") if first_alert else "None",
        "lead_months" : lead_months,
        "AUC"         : auc,
        "F1"          : f1,
        "Recall"      : rec,
        "probs"       : probs,
        "y_true"      : y_te_seq,
        "dates"       : d_te_seq,
        "color"       : ep["color"],
    }

# ==============================================================================
# 3. SUMMARY
# ==============================================================================
print("\n" + "=" * 60)
print("STEP 3: Summary")
print("=" * 60)

summary_rows = []
for ep_name, r in results.items():
    print(f"\n  {r['label']}:")
    print(f"    First alert : {r['first_alert']}")
    print(f"    Lead time   : {r['lead_months']} months before peak")
    print(f"    AUC         : {r['AUC']:.3f}" if not np.isnan(r['AUC'])
          else "    AUC         : N/A")
    summary_rows.append({k: v for k, v in r.items()
                         if k not in ["probs", "y_true", "dates", "color"]})

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(RESULTS_DIR + "crisis_backtest_summary.csv", index=False)

# Verdict
print("\n" + "*" * 60)
gfc_lead   = results.get("GFC 2008", {}).get("lead_months", 0)
covid_lead = results.get("COVID 2021", {}).get("lead_months", 0)

if gfc_lead >= 2 and covid_lead >= 2:
    print("* VERDICT: KILLER RESULT -- agent detects BOTH crises early! *")
    print(f"* GFC: {gfc_lead}M lead | COVID: {covid_lead}M lead *")
elif gfc_lead >= 2 or covid_lead >= 2:
    print("* VERDICT: PARTIAL -- detects one crisis early *")
else:
    print("* VERDICT: No early detection -- check threshold *")
print("*" * 60)

# ==============================================================================
# 4. FIGURES
# ==============================================================================
print("\nSTEP 4: Generating figures")

fig, axes = plt.subplots(2, 1, figsize=(16, 12))
fig.suptitle(
    f"Paper 4 -- Crisis Episode Backtests (Ensemble)\n"
    f"LSTM trained on pre-crisis data only -> "
    f"detecting {TARGET_MAT.replace('GR_','')} crises in advance",
    fontsize=13, fontweight="bold"
)

for ax, (ep_name, r) in zip(axes, results.items()):
    ep      = EPISODES[ep_name]
    dates   = np.array(r["dates"], dtype="datetime64[D]")
    probs   = r["probs"]
    y_true  = r["y_true"]
    color   = r["color"]

    ax.fill_between(dates, 0, 1,
                    where=(y_true == 1),
                    color="red", alpha=0.15, label="Actual crisis")
    ax.plot(dates, probs, color=color, lw=2,
            label=f"LSTM P(crisis) -- lead={LEAD}M")
    ax.axhline(ALERT_THRESH, color="gray", linestyle="--",
               lw=1, alpha=0.8, label=f"Alert threshold ({ALERT_THRESH})")

    peak_date = pd.Timestamp(ep["crisis_peak"])
    ax.axvline(peak_date, color="darkred", linestyle="-.", lw=1.5,
               label=f"Crisis peak ({ep['crisis_peak'][:7]})")

    if r["first_alert"] != "None":
        first_alert_ts = pd.Timestamp(r["first_alert"])
        ax.axvline(first_alert_ts, color="green",
                   linestyle="-.", lw=1.5,
                   label=f"First alert ({r['first_alert']})")
        ax.annotate(
            f"Lead: {r['lead_months']}M",
            xy=(first_alert_ts, 0.85),
            xytext=(first_alert_ts, 0.95),
            fontsize=11, fontweight="bold", color="green",
            ha="center",
            arrowprops=dict(arrowstyle="->", color="green")
        )
        ax.axvspan(first_alert_ts, peak_date,
                   color="green", alpha=0.08,
                   label=f"Early warning window")

    auc_str = f"AUC={r['AUC']:.3f}" if not np.isnan(r['AUC']) else ""
    ax.set_title(
        f"{r['label']} | Train: 2000->{r['train_end'][:7]} | "
        f"Test: {r['test_period']} | {auc_str}",
        fontsize=10
    )
    ax.set_ylabel("P(crisis)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.grid(alpha=0.3)

plt.tight_layout()
fig_path = RESULTS_DIR + "fig_p4_crisis_backtests.pdf"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Figure saved: {fig_path}")
plt.show()

print("\n" + "=" * 60)
print("DONE -- Crisis backtests complete.")
print(f"GFC 2008 lead  : {results.get('GFC 2008',{}).get('lead_months',0)}M")
print(f"COVID 2021 lead: {results.get('COVID 2021',{}).get('lead_months',0)}M")
print("Next: run 12_decision_rules.py")
print("=" * 60)
