"""
Paper 5 -- B2c: Hidden Markov Model regime classification of CSRI
==================================================================
Estimates a 2-state Gaussian Hidden Markov Model on the monthly
Composite Systemic Risk Index. Provides a probabilistic
classification of stable vs crisis regimes that complements the
75th-percentile rolling-volatility threshold used elsewhere in
the paper.

Inputs:
  data/processed/csri_monthly.csv
Outputs:
  data/processed/csri_hmm_states.csv
  results/table_hmm_regimes.csv
  results/figures/fig_hmm_csri.{pdf,png}
"""

import os
import io
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

PROC_DIR = "../data/processed/"
RES_DIR  = "../results/"
FIG_DIR  = "../results/figures/"

print("=" * 64)
print("Paper 5 -- B2c: HMM regime classification of CSRI")
print("=" * 64)


# ---------------------------------------------------------------------------
# 1. LOAD CSRI
# ---------------------------------------------------------------------------
path = os.path.join(PROC_DIR, "csri_monthly.csv")
if not os.path.exists(path):
    print(f"  [error] {path} not found. Run 07_systemic_risk_index first.")
    sys.exit(1)

csri = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
csri = csri.dropna(subset=["CSRI"]).reset_index(drop=True)
print(f"  CSRI: {len(csri)} months, "
      f"{csri['date'].min().date()} -> {csri['date'].max().date()}")


# ---------------------------------------------------------------------------
# 2. FIT 2-STATE GAUSSIAN HMM
# ---------------------------------------------------------------------------
print("\nSTEP 2: Fitting 2-state Gaussian HMM ...")

try:
    from hmmlearn.hmm import GaussianHMM
    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False
    print("  [warn] hmmlearn not installed; using statsmodels MarkovRegression")

if HAS_HMMLEARN:
    X = csri["CSRI"].values.reshape(-1, 1)
    np.random.seed(42)
    model = GaussianHMM(n_components=2, covariance_type="diag",
                         n_iter=200, random_state=42, tol=1e-4)
    model.fit(X)
    states = model.predict(X)
    posterior = model.predict_proba(X)

    means = model.means_.flatten()
    stds  = np.sqrt(model.covars_.flatten())

    # Reorder so state 0 = stable (low mean), state 1 = crisis (high mean)
    if means[0] > means[1]:
        states    = 1 - states
        posterior = posterior[:, [1, 0]]
        means     = means[::-1]
        stds      = stds[::-1]
        model.transmat_ = model.transmat_[[1, 0]][:, [1, 0]]

    transmat = model.transmat_
    expected_dur = 1.0 / (1.0 - np.diag(transmat))

    print(f"  Stable state:  mean={means[0]:+.3f}  std={stds[0]:.3f}")
    print(f"  Crisis state:  mean={means[1]:+.3f}  std={stds[1]:.3f}")
    print(f"  Transition matrix:\n{transmat.round(3)}")
    print(f"  Expected duration -- stable: {expected_dur[0]:.1f} months")
    print(f"  Expected duration -- crisis: {expected_dur[1]:.1f} months")

    csri["hmm_state"]      = states
    csri["hmm_p_crisis"]   = posterior[:, 1]
else:
    from statsmodels.tsa.regime_switching.markov_regression \
        import MarkovRegression
    mod = MarkovRegression(csri["CSRI"].values, k_regimes=2,
                            switching_variance=True)
    res = mod.fit(disp=False)
    # statsmodels returns smoothed_marginal_probabilities as (k_regimes, T)
    smoothed = np.asarray(res.smoothed_marginal_probabilities)
    if smoothed.ndim == 2 and smoothed.shape[0] == 2:
        # shape (2, T)
        s_0, s_1 = smoothed[0], smoothed[1]
    else:
        # shape (T, 2)
        s_0, s_1 = smoothed[:, 0], smoothed[:, 1]
    states    = (s_1 > 0.5).astype(int)
    posterior = np.column_stack([s_0, s_1])

    # statsmodels' MarkovRegression: get regime-specific intercepts
    means = [float(res.params[i]) for i in range(2)]
    # Variances: try labelled access, fallback to last 2 params
    try:
        stds = [float(np.sqrt(res.params[f"sigma2[{i}]"])) for i in range(2)]
    except Exception:
        stds = [float(np.sqrt(res.params[-2])),
                float(np.sqrt(res.params[-1]))]

    if means[0] > means[1]:
        states    = 1 - states
        posterior = posterior[:, [1, 0]]
        means.reverse(); stds.reverse()

    transmat = np.asarray(res.regime_transition)
    if transmat.ndim == 3:
        transmat = transmat[:, :, 0]
    expected_dur = 1.0 / (1.0 - np.diag(transmat))
    csri["hmm_state"]    = states
    csri["hmm_p_crisis"] = posterior[:, 1]
    print(f"  Stable mean={means[0]:.3f}  Crisis mean={means[1]:.3f}")
    print(f"  Stable std={stds[0]:.3f}  Crisis std={stds[1]:.3f}")
    print(f"  Transition matrix:\n{transmat.round(3)}")
    print(f"  Expected duration -- stable: {expected_dur[0]:.1f} months")
    print(f"  Expected duration -- crisis: {expected_dur[1]:.1f} months")


# ---------------------------------------------------------------------------
# 3. REGIME SUMMARY
# ---------------------------------------------------------------------------
n_stable = int((csri["hmm_state"] == 0).sum())
n_crisis = int((csri["hmm_state"] == 1).sum())
print(f"\n  Stable months:  {n_stable}  ({100*n_stable/len(csri):.1f}%)")
print(f"  Crisis months:  {n_crisis}  ({100*n_crisis/len(csri):.1f}%)")

if "regime" in csri.columns:
    cross = pd.crosstab(csri["regime"], csri["hmm_state"],
                          rownames=["vol-quantile regime"],
                          colnames=["HMM state"])
    print("\n  Crosstab vs vol-quantile regime:")
    print(cross)

# Crisis episodes timing
csri["hmm_state_diff"] = csri["hmm_state"].diff().fillna(0)
crisis_starts = csri.loc[csri["hmm_state_diff"] == 1, "date"].dt.strftime("%Y-%m").tolist()
crisis_ends   = csri.loc[csri["hmm_state_diff"] == -1, "date"].dt.strftime("%Y-%m").tolist()
print(f"\n  Crisis-state starts:  {crisis_starts}")
print(f"  Crisis-state ends:    {crisis_ends}")


# ---------------------------------------------------------------------------
# 4. SAVE
# ---------------------------------------------------------------------------
csri.to_csv(os.path.join(PROC_DIR, "csri_hmm_states.csv"), index=False)

summary = pd.DataFrame([
    {"metric": "stable_state_mean",     "value": float(means[0])},
    {"metric": "crisis_state_mean",     "value": float(means[1])},
    {"metric": "stable_state_std",      "value": float(stds[0])},
    {"metric": "crisis_state_std",      "value": float(stds[1])},
    {"metric": "n_stable_months",       "value": float(n_stable)},
    {"metric": "n_crisis_months",       "value": float(n_crisis)},
    {"metric": "P(stable->stable)",     "value": float(transmat[0, 0])},
    {"metric": "P(crisis->crisis)",     "value": float(transmat[1, 1])},
    {"metric": "expected_dur_stable",   "value": float(expected_dur[0])},
    {"metric": "expected_dur_crisis",   "value": float(expected_dur[1])},
    {"metric": "n_crisis_episodes",     "value": float(len(crisis_starts))},
])
summary.to_csv(os.path.join(RES_DIR, "table_hmm_regimes.csv"), index=False)
print(f"\n  Saved: table_hmm_regimes.csv")


# ---------------------------------------------------------------------------
# 5. FIGURE
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

ax = axes[0]
ax.plot(csri["date"], csri["CSRI"], color="#1A237E", lw=1.4, label="CSRI")
ax.fill_between(csri["date"], -10, 10,
                where=(csri["hmm_state"] == 1),
                color="#F44336", alpha=0.15,
                label="HMM crisis state")
ax.axhline(0, color="black", lw=0.5)
ax.set_ylim(csri["CSRI"].min() - 0.3, csri["CSRI"].max() + 0.3)
ax.set_ylabel("CSRI ($z$-score)")
ax.set_title("(a) CSRI timeline with HMM crisis-state shading")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(csri["date"], csri["hmm_p_crisis"], color="#C62828", lw=1.6)
ax.fill_between(csri["date"], 0, csri["hmm_p_crisis"],
                color="#C62828", alpha=0.30)
ax.axhline(0.5, color="gray", ls="--", lw=1)
ax.set_ylim(0, 1.05)
ax.set_ylabel(r"$P(\text{crisis state}\,|\,\text{CSRI}_{1:t})$")
ax.set_xlabel("Date")
ax.set_title("(b) Smoothed posterior probability of crisis state")
ax.grid(alpha=0.3)

plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG_DIR, f"fig_hmm_csri.{ext}"),
                dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: fig_hmm_csri.{{pdf,png}}")

print("\n" + "=" * 64)
print("DONE -- 18_hmm_csri.py")
print("=" * 64)
