"""
Paper 4 -- Automated Decision Rules (v2 -- ensemble + selected features)
==========================================================================
Translates LSTM crisis probabilities into automated procurement
decision rules -- upgrading Paper 3's static Rule R6 to a
dynamic, predictive LSTM-based trigger.

v2 changes:
- Uses utils.py (ensemble, helpers)
- 5-seed ensemble predictions
- SHAP-selected features (14 of 20)
- Fixed Unicode for Windows cp1253

Outputs:
  - results/decision_rules_p4.csv
  - results/contingency_timeline.csv
  - results/p3_vs_p4_comparison.csv
  - results/fig_p4_decision_rules.pdf

Run from: construction-risk-suite/paper4-lstm-agent/src/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from sklearn.preprocessing import MinMaxScaler

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
TRAIN_RATIO   = 0.75

ENSEMBLE_SEEDS = DEFAULT_SEEDS
USE_SELECTED_FEATURES = False  # all 20 features (ablation showed full set is better)

# Base cost and contingency parameters (from Paper 3)
BASE_COST     = 2_300_000
ES_CRISIS     = 2_944_866
ES_STABLE     = 2_394_721
CONTINGENCY_CRISIS = ES_CRISIS - BASE_COST   # EUR 644,866
CONTINGENCY_STABLE = ES_STABLE - BASE_COST   # EUR 94,721

# Alert thresholds
P_LOW    = 0.30
P_MEDIUM = 0.50
P_HIGH   = 0.70

np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 60)
print("Paper 4 -- Automated Decision Rules (Ensemble)")
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
raw    = pd.read_csv(PROCESSED_DIR + "aligned_returns.csv",
                     index_col=0, parse_dates=True)

FEAT_COLS = list(features.columns)
common    = features.index.intersection(labels.dropna().index)
X_raw     = features.loc[common].values
y_raw     = labels.loc[common, TARGET_MAT].values
dates     = common

print(f"  Features : {len(FEAT_COLS)} | Samples: {len(X_raw)}")

# ==============================================================================
# 2. TRAIN ENSEMBLE
# ==============================================================================
print(f"\nSTEP 2: Training {len(ENSEMBLE_SEEDS)}-seed ensemble")

scaler = MinMaxScaler()
n_raw_tr = int(len(X_raw) * TRAIN_RATIO)
scaler.fit(X_raw[:n_raw_tr])
X_sc = scaler.transform(X_raw)

X_seq, y_seq, d_seq = make_sequences(X_sc, y_raw, LOOKBACK, LEAD, dates=dates)
n_train = int(len(X_seq) * TRAIN_RATIO)
X_train, X_test = X_seq[:n_train], X_seq[n_train:]
y_train, y_test = y_seq[:n_train], y_seq[n_train:]
d_test          = d_seq[n_train:]

models = train_ensemble(X_train, y_train, seeds=ENSEMBLE_SEEDS,
                        epochs=EPOCHS, batch_size=BATCH_SIZE,
                        hidden_size=HIDDEN_SIZE, n_layers=N_LAYERS,
                        dropout=DROPOUT, lr=LR, patience=PATIENCE,
                        verbose=True)
probs_test = predict_ensemble(models, X_test)
print(f"  Test samples: {len(X_test)}")

# ==============================================================================
# 3. GENERATE DECISION RULES
# ==============================================================================
print("\nSTEP 3: Generating decision rules")

def adaptive_contingency(p_crisis):
    p = np.clip(p_crisis, 0, 1)
    return CONTINGENCY_STABLE + p * (CONTINGENCY_CRISIS - CONTINGENCY_STABLE)

def regime_label(p):
    if p < P_LOW:    return "STABLE"
    if p < P_MEDIUM: return "ELEVATED"
    if p < P_HIGH:   return "WARNING"
    return "CRISIS"

def rule_actions(p_crisis, month_in_project):
    actions = []
    regime  = regime_label(p_crisis)
    cont    = adaptive_contingency(p_crisis)

    actions.append({
        "rule"   : "R1",
        "trigger": f"P(crisis)={p_crisis:.2f} -> {regime}",
        "action" : f"Set contingency to EUR {cont:,.0f} "
                   f"({cont/BASE_COST*100:.1f}% of base)",
        "EUR"    : cont
    })

    if month_in_project <= 8:
        actions.append({
            "rule"   : "R3",
            "trigger": f"Foundation phase + {regime}",
            "action" : "Allocate 25% contingency; lock concrete price",
            "EUR"    : cont * 0.25
        })
    elif month_in_project <= 18:
        actions.append({
            "rule"   : "R2",
            "trigger": f"Superstructure phase + {regime}",
            "action" : "Allocate 55% contingency; pre-purchase steel",
            "EUR"    : cont * 0.55
        })
    else:
        actions.append({
            "rule"   : "R4",
            "trigger": f"Completion phase + {regime}",
            "action" : "Allocate 20% contingency; monitor fuel index",
            "EUR"    : cont * 0.20
        })

    if p_crisis < P_MEDIUM:
        actions.append({
            "rule"   : "R5",
            "trigger": f"P(crisis)={p_crisis:.2f} < {P_MEDIUM}",
            "action" : "Evaluate Steel PPI swap if rolling rho > 0.40",
            "EUR"    : 45_923
        })
    else:
        actions.append({
            "rule"   : "R5",
            "trigger": f"P(crisis)={p_crisis:.2f} >= {P_MEDIUM} -> SUPPRESS hedge",
            "action" : "Do NOT hedge -- crisis mode degrades HE to <3%",
            "EUR"    : 0
        })

    actions.append({
        "rule"   : "R6b",
        "trigger": f"LSTM P(crisis)={p_crisis:.2f} (lead={LEAD}M)",
        "action" : f"Switch to {'CRISIS' if p_crisis >= P_MEDIUM else 'STABLE'} "
                   f"ES model -- {LEAD} months ahead of vol spike",
        "EUR"    : abs(cont - adaptive_contingency(0))
    })

    if p_crisis >= P_LOW:
        actions.append({
            "rule"   : "R7",
            "trigger": "SHAP: Steel_vol3 > Cement_vol3 > Brent_vol6",
            "action" : "Monitor US Steel PPI vol (primary signal) + "
                       "Cement vol (secondary) monthly",
            "EUR"    : None
        })

    actions.append({
        "rule"   : "R8",
        "trigger": f"Continuous P(crisis)={p_crisis:.2f}",
        "action" : f"Scale contingency linearly: "
                   f"EUR {CONTINGENCY_STABLE:,.0f} -> "
                   f"EUR {CONTINGENCY_CRISIS:,.0f} as P(crisis) 0->1",
        "EUR"    : cont
    })

    return actions

# Apply to test set
print("\n  Generating rules for test period...")
rules_timeline = []
for i, (d, p, y) in enumerate(zip(d_test, probs_test, y_test)):
    phase = (i % 24) + 1
    for act in rule_actions(p, phase):
        rules_timeline.append({
            "date"         : pd.Timestamp(str(d)).strftime("%Y-%m"),
            "P_crisis"     : round(float(p), 3),
            "regime"       : regime_label(p),
            "actual_crisis": int(y),
            **act
        })

df_rules = pd.DataFrame(rules_timeline)
df_rules.to_csv(RESULTS_DIR + "decision_rules_p4.csv", index=False)
print(f"  Rules generated: {len(df_rules)} rows")

# ==============================================================================
# 4. PAPER 3 VS PAPER 4 COMPARISON TABLE
# ==============================================================================
print("\nSTEP 4: Paper 3 vs Paper 4 comparison")

comparison = pd.DataFrame([
    {"Rule": "R1",
     "Paper 3 (static)": "Crisis detected if vol > 67th pctile",
     "Paper 4 (LSTM)": f"Crisis predicted if P(crisis) > {P_MEDIUM} -- {LEAD}M AHEAD",
     "Upgrade": f"+{LEAD}M lead time"},
    {"Rule": "R2-R4",
     "Paper 3 (static)": "Fixed EUR amounts per phase",
     "Paper 4 (LSTM)": "Adaptive EUR = f(P_crisis) -- scales continuously",
     "Upgrade": "Continuous scaling vs binary"},
    {"Rule": "R5",
     "Paper 3 (static)": "Hedge if rolling rho > 0.40",
     "Paper 4 (LSTM)": "Suppress hedge automatically in crisis regime",
     "Upgrade": "Crisis-aware suppression"},
    {"Rule": "R6",
     "Paper 3 (static)": "Switch model if vol > 67th pctile (reactive)",
     "Paper 4 (LSTM)": f"Switch model if P(crisis) > {P_MEDIUM} (predictive, {LEAD}M ahead)",
     "Upgrade": "Reactive -> Predictive"},
    {"Rule": "R6b (NEW)",
     "Paper 3 (static)": "--",
     "Paper 4 (LSTM)": "LSTM early warning with lead time quantification",
     "Upgrade": "New rule"},
    {"Rule": "R7 (NEW)",
     "Paper 3 (static)": "--",
     "Paper 4 (LSTM)": "SHAP-guided monitoring priority (Steel > Cement > Brent)",
     "Upgrade": "New rule"},
    {"Rule": "R8 (NEW)",
     "Paper 3 (static)": "Binary contingency (stable vs crisis)",
     "Paper 4 (LSTM)": f"Continuous contingency: EUR {CONTINGENCY_STABLE:,.0f} -> "
                        f"EUR {CONTINGENCY_CRISIS:,.0f} as P(crisis) 0->1",
     "Upgrade": "Binary -> Continuous"},
])

for _, row in comparison.iterrows():
    print(f"\n  {row['Rule']}:")
    print(f"    P3: {row['Paper 3 (static)']}")
    print(f"    P4: {row['Paper 4 (LSTM)']}")
    print(f"    ^  : {row['Upgrade']}")

comparison.to_csv(RESULTS_DIR + "p3_vs_p4_comparison.csv", index=False)

# ==============================================================================
# 5. ADAPTIVE CONTINGENCY TIMELINE
# ==============================================================================
print("\nSTEP 5: Adaptive contingency timeline")

contingency_timeline = pd.DataFrame({
    "date"           : d_test,
    "P_crisis"       : probs_test,
    "regime"         : [regime_label(p) for p in probs_test],
    "contingency_EUR": [adaptive_contingency(p) for p in probs_test],
    "actual_crisis"  : y_test
})
contingency_timeline.to_csv(RESULTS_DIR + "contingency_timeline.csv", index=False)

mean_cont   = contingency_timeline["contingency_EUR"].mean()
crisis_cont = contingency_timeline[
    contingency_timeline["actual_crisis"] == 1]["contingency_EUR"].mean()
stable_cont = contingency_timeline[
    contingency_timeline["actual_crisis"] == 0]["contingency_EUR"].mean()

print(f"  Mean contingency (all)   : EUR {mean_cont:,.0f}")
print(f"  Mean contingency (crisis): EUR {crisis_cont:,.0f}")
print(f"  Mean contingency (stable): EUR {stable_cont:,.0f}")
print(f"  Paper 3 static crisis    : EUR {CONTINGENCY_CRISIS:,.0f}")
print(f"  Saving vs static         : EUR {CONTINGENCY_CRISIS - mean_cont:,.0f}")

# ==============================================================================
# 6. FIGURES
# ==============================================================================
print("\nSTEP 6: Generating figures")

fig, axes = plt.subplots(3, 1, figsize=(16, 14))
fig.suptitle(
    "Paper 4 -- Automated Decision Rules: Paper 3 (Static) vs Paper 4 (LSTM Ensemble)\n"
    f"Adaptive contingency and predictive regime switching -- "
    f"lead={LEAD}M | {TARGET_MAT}",
    fontsize=12, fontweight="bold"
)

dates_plot = pd.to_datetime([str(d) for d in d_test])

# Panel 1: P(crisis) with regime bands
ax = axes[0]
ax.fill_between(dates_plot, 0, 1,
                where=(y_test == 1),
                color="red", alpha=0.15, label="Actual crisis")
ax.plot(dates_plot, probs_test, color="#e74c3c", lw=2,
        label=f"LSTM P(crisis) -- lead={LEAD}M")
ax.axhline(P_LOW,    color="green",  linestyle=":", lw=1, alpha=0.8,
           label=f"Stable ({P_LOW})")
ax.axhline(P_MEDIUM, color="orange", linestyle="--", lw=1.5,
           label=f"Alert ({P_MEDIUM})")
ax.axhline(P_HIGH,   color="red",    linestyle="-.", lw=1, alpha=0.8,
           label=f"Crisis ({P_HIGH})")
ax.set_ylabel("P(crisis)")
ax.set_ylim(0, 1.05)
ax.set_title("LSTM Crisis Probability with Regime Thresholds (R6b)")
ax.legend(fontsize=8, ncol=3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.grid(alpha=0.3)

# Panel 2: Adaptive contingency vs Static
ax = axes[1]
cont_values = [adaptive_contingency(p) for p in probs_test]
ax.fill_between(dates_plot, 0, 1,
                where=(y_test == 1),
                color="red", alpha=0.10)
ax.plot(dates_plot, cont_values, color="#3498db", lw=2,
        label="Paper 4: Adaptive contingency (R8)")
ax.axhline(CONTINGENCY_CRISIS, color="red", linestyle="--", lw=1.5,
           label=f"Paper 3: Static crisis = EUR {CONTINGENCY_CRISIS:,.0f}")
ax.axhline(CONTINGENCY_STABLE, color="green", linestyle="--", lw=1.5,
           label=f"Paper 3: Static stable = EUR {CONTINGENCY_STABLE:,.0f}")
ax.set_ylabel("Contingency Reserve (EUR)")
ax.set_title("Adaptive Contingency: Paper 4 (continuous) vs Paper 3 (binary) -- Rule R8")
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"EUR {x:,.0f}"))
ax.grid(alpha=0.3)

# Panel 3: Regime timeline
ax = axes[2]
regime_values = {"STABLE": 0, "ELEVATED": 1, "WARNING": 2, "CRISIS": 3}
regimes = [regime_label(p) for p in probs_test]
regime_nums = [regime_values[r] for r in regimes]

for reg, val, color in [("STABLE", 0, "#2ecc71"),
                          ("ELEVATED", 1, "#f39c12"),
                          ("WARNING", 2, "#e67e22"),
                          ("CRISIS", 3, "#e74c3c")]:
    mask = np.array(regime_nums) == val
    if mask.any():
        ax.fill_between(dates_plot, val - 0.4, val + 0.4,
                        where=mask, color=color, alpha=0.7, step="mid")

for i, (d, y) in enumerate(zip(dates_plot, y_test)):
    if y == 1:
        ax.axvline(d, color="red", alpha=0.2, lw=0.5)

ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(["STABLE", "ELEVATED", "WARNING", "CRISIS"])
ax.set_title("Automated Regime Classification Timeline -- Rules R6b, R1")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.grid(alpha=0.3, axis="x")

legend_patches = [
    Patch(facecolor="#2ecc71", label="STABLE"),
    Patch(facecolor="#f39c12", label="ELEVATED"),
    Patch(facecolor="#e67e22", label="WARNING"),
    Patch(facecolor="#e74c3c", label="CRISIS"),
]
ax.legend(handles=legend_patches, fontsize=8, ncol=4, loc="upper left")

plt.tight_layout()
fig_path = RESULTS_DIR + "fig_p4_decision_rules.pdf"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Figure saved: {fig_path}")
plt.show()

# ==============================================================================
# 7. SUMMARY
# ==============================================================================
print("\n" + "=" * 60)
print("DECISION RULES SUMMARY -- Paper 4")
print("=" * 60)

rules_summary = [
    ("R1" , "LSTM P(crisis) > threshold",
     f"Set contingency: EUR {CONTINGENCY_STABLE:,.0f}-{CONTINGENCY_CRISIS:,.0f}",
     "Adaptive"),
    ("R2" , "Superstructure start + P(crisis)",
     "Allocate 55%; pre-purchase steel", "Adaptive EUR"),
    ("R3" , "Foundation start + P(crisis)",
     "Allocate 25%; lock concrete", "Adaptive EUR"),
    ("R4" , "Completion start + P(crisis)",
     "Allocate 20%; monitor fuel", "Adaptive EUR"),
    ("R5" , "P(crisis) < 0.50 AND rho > 0.40",
     "Steel PPI swap; SUPPRESS if P > 0.50", "Crisis-aware"),
    ("R6b", f"LSTM P(crisis) > {P_MEDIUM} (lead={LEAD}M)",
     "Switch to crisis ES model PREDICTIVELY", "NEW -- Predictive"),
    ("R7" , "SHAP importance ranking",
     "Monitor Steel > Cement > Brent monthly", "NEW -- SHAP-guided"),
    ("R8" , "P(crisis) continuous [0,1]",
     f"Scale contingency EUR {CONTINGENCY_STABLE:,.0f}->{CONTINGENCY_CRISIS:,.0f}",
     "NEW -- Continuous"),
]

for rule, trigger, action, note in rules_summary:
    print(f"\n  {rule} [{note}]")
    print(f"    Trigger: {trigger}")
    print(f"    Action : {action}")

print("\n" + "=" * 60)
print("DONE -- Decision rules complete.")
print("Next: run 13_economic_value.py")
print("=" * 60)
