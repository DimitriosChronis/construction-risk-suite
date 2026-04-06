"""
Paper 4 -- Granger Causality Tests
====================================
Provides econometric foundation for the LSTM lead time result.

Before using ML, we need to show formally that:
  US PPI series Granger-cause Greek construction crises

This answers: "Is the lead time real or spurious?"

Tests:
  1. Bivariate Granger causality: each US series → Greek crisis
  2. Multivariate VAR Granger causality: all US → Greek (jointly)
  3. Instantaneous causality test
  4. Optimal lag selection (AIC/BIC)
  5. Impulse Response Functions (IRF) — how long does the effect last?

Outputs:
  - results/granger_causality.csv
  - results/fig_p4_granger.pdf

Run from: construction-risk-suite/paper4-lstm-agent/src/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.stats.stattools import durbin_watson
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
PROCESSED_DIR = "../data/processed/"
RESULTS_DIR   = "../results/"
SEED          = 42
TARGET_MAT    = "GR_Fuel_Energy"
MAX_LAGS      = 4      # test lags 1-4 months
ALPHA         = 0.05
VOL_WINDOW    = 6
CRISIS_PCT    = 0.75

np.random.seed(SEED)

print("=" * 60)
print("Paper 4 -- Granger Causality Tests")
print(f"Target: {TARGET_MAT} | Max lags: {MAX_LAGS}M")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 1: Loading data")

raw = pd.read_csv(PROCESSED_DIR + "aligned_returns.csv",
                  index_col=0, parse_dates=True)

GR_COLS = [c for c in raw.columns if c.startswith("GR_")]
US_COLS = [c for c in raw.columns if c.startswith("US_")]

# Build Greek crisis vol series
gr_vol   = raw[TARGET_MAT].rolling(VOL_WINDOW).std()
thr      = gr_vol.quantile(CRISIS_PCT)
gr_crisis = (gr_vol > thr).astype(float)

# Align
common    = raw.index.intersection(gr_crisis.dropna().index)
raw_al    = raw.loc[common]
crisis_al = gr_crisis.loc[common]
vol_al    = gr_vol.loc[common]

print(f"  Period: {common[0].strftime('%Y-%m')} -> "
      f"{common[-1].strftime('%Y-%m')}")
print(f"  N = {len(common)} months")
print(f"  Crisis rate: {crisis_al.mean()*100:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# 2. STATIONARITY TESTS (required for Granger)
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 2: ADF stationarity tests")
print("-" * 50)

adf_results = []
test_series = {**{c: raw_al[c] for c in US_COLS},
               TARGET_MAT: vol_al}

for name, series in test_series.items():
    result = adfuller(series.dropna(), autolag="AIC")
    stationary = result[1] < ALPHA
    adf_results.append({
        "series"    : name,
        "adf_stat"  : round(result[0], 4),
        "p_value"   : round(result[1], 4),
        "stationary": stationary
    })
    print(f"  {name:30s}: p={result[1]:.4f} "
          f"-> {'I(0) OK' if stationary else 'I(1) -- use returns'}")

df_adf = pd.DataFrame(adf_results)

# ══════════════════════════════════════════════════════════════════════════════
# 3. BIVARIATE GRANGER CAUSALITY
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 3: Bivariate Granger causality (US -> Greek vol)")
print("-" * 60)

granger_results = []

for us_col in US_COLS:
    # Build bivariate dataframe: [Greek vol, US series]
    df_bi = pd.DataFrame({
        "greek_vol": vol_al,
        "us_series": raw_al[us_col]
    }).dropna()

    try:
        gc_result = grangercausalitytests(
            df_bi[["greek_vol", "us_series"]],
            maxlag=MAX_LAGS, verbose=False
        )

        # Extract p-values per lag
        for lag in range(1, MAX_LAGS + 1):
            f_stat = gc_result[lag][0]["ssr_ftest"][0]
            p_val  = gc_result[lag][0]["ssr_ftest"][1]
            sig    = p_val < ALPHA

            granger_results.append({
                "US_series" : us_col,
                "lag_months": lag,
                "F_stat"    : round(f_stat, 4),
                "p_value"   : round(p_val, 4),
                "significant": sig
            })

            marker = "***" if p_val < 0.01 else "**" if p_val < 0.05 \
                     else "*" if p_val < 0.10 else ""
            print(f"  {us_col:25s} lag={lag}M: "
                  f"F={f_stat:.3f}, p={p_val:.4f} {marker}")

    except Exception as e:
        print(f"  {us_col}: ERROR -- {e}")

df_granger = pd.DataFrame(granger_results)

# Summary: which series are significant?
print("\n  SIGNIFICANT Granger causes (p < 0.05):")
sig = df_granger[df_granger["significant"]]
if len(sig) > 0:
    for _, row in sig.iterrows():
        print(f"    {row['US_series']} at lag {row['lag_months']}M "
              f"(p={row['p_value']:.4f})")
else:
    print("    None at p < 0.05")

# ══════════════════════════════════════════════════════════════════════════════
# 4. MULTIVARIATE VAR GRANGER (joint test)
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 4: Multivariate VAR Granger causality (joint test)")
print("-" * 60)

# Build VAR dataset: Greek vol + all US series
df_var = pd.concat([vol_al.rename("greek_vol")] +
                   [raw_al[c] for c in US_COLS],
                   axis=1).dropna()

# Optimal lag selection
var_model = VAR(df_var)
lag_order = var_model.select_order(maxlags=MAX_LAGS)
optimal_lag = lag_order.aic

print(f"  Optimal lag (AIC): {optimal_lag}M")
print(f"  Lag order summary:")
for criterion in ["aic", "bic", "hqic"]:
    val = getattr(lag_order, criterion)
    print(f"    {criterion.upper()}: {val}M")

# Fit VAR
try:
    var_fitted = var_model.fit(optimal_lag)

    # Granger causality: do US series jointly cause Greek vol?
    gc_test = var_fitted.test_causality(
        "greek_vol",
        [c for c in US_COLS],
        kind="f"
    )
    print(f"\n  Joint Granger causality (all US -> Greek vol):")
    print(f"    F-stat: {gc_test.test_statistic:.4f}")
    print(f"    p-value: {gc_test.pvalue:.4f}")
    print(f"    Significant: {'YES' if gc_test.pvalue < ALPHA else 'NO'}")

    var_joint_p = gc_test.pvalue
    var_joint_f = gc_test.test_statistic

    # IRF — Impulse Response Functions
    irf = var_fitted.irf(periods=6)
    irf_values = irf.irfs  # shape: (periods, n_vars, n_vars)

    # Response of greek_vol to each US shock
    var_names = list(df_var.columns)
    greek_idx = var_names.index("greek_vol")
    irf_to_greek = {}
    for us_col in US_COLS:
        us_idx = var_names.index(us_col)
        irf_to_greek[us_col] = irf_values[:, greek_idx, us_idx]

except Exception as e:
    print(f"  VAR fitting error: {e}")
    var_joint_p = np.nan
    var_joint_f = np.nan
    irf_to_greek = {}

# ══════════════════════════════════════════════════════════════════════════════
# 5. SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════
df_granger.to_csv(RESULTS_DIR + "granger_causality.csv", index=False)
df_adf.to_csv(RESULTS_DIR + "adf_stationarity.csv", index=False)

summary_row = pd.DataFrame([{
    "n_significant_bivariate" : df_granger["significant"].sum(),
    "best_us_series"          : df_granger[df_granger["significant"]].sort_values("p_value")["US_series"].iloc[0] if df_granger["significant"].any() else "None",
    "best_lag"                : df_granger[df_granger["significant"]].sort_values("p_value")["lag_months"].iloc[0] if df_granger["significant"].any() else np.nan,
    "var_joint_F"             : var_joint_f,
    "var_joint_p"             : var_joint_p,
    "optimal_lag_AIC"         : optimal_lag,
}])
summary_row.to_csv(RESULTS_DIR + "granger_summary.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# 6. FIGURES
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 5: Generating figures")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(
    f"Paper 4 -- Granger Causality: US PPI -> Greek Construction Cost Volatility\n"
    f"Target: {TARGET_MAT} | Max lag: {MAX_LAGS}M",
    fontsize=12, fontweight="bold"
)

# Panel 1: Granger p-values heatmap
ax = axes[0, 0]
pivot = df_granger.pivot(index="US_series", columns="lag_months",
                          values="p_value")
pivot.index = [i.replace("US_", "").replace("_PPI", "") for i in pivot.index]
im = ax.imshow(pivot.values, cmap="RdYlGn_r", vmin=0, vmax=0.20,
               aspect="auto")
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels([f"{l}M" for l in pivot.columns])
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index)
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        v = pivot.values[i, j]
        sig = "*" if v < 0.05 else ""
        ax.text(j, i, f"{v:.3f}{sig}",
                ha="center", va="center", fontsize=9,
                fontweight="bold" if v < 0.05 else "normal")
plt.colorbar(im, ax=ax, label="p-value (green=significant)")
ax.set_xlabel("Lead time (months)")
ax.set_title("Granger Causality p-values\n"
             "(*p<0.05 — green = US Granger-causes Greek vol)")

# Panel 2: F-statistics
ax = axes[0, 1]
pivot_f = df_granger.pivot(index="US_series", columns="lag_months",
                             values="F_stat")
pivot_f.index = [i.replace("US_", "").replace("_PPI", "")
                 for i in pivot_f.index]
for i, series in enumerate(pivot_f.index):
    ax.plot(range(1, MAX_LAGS + 1), pivot_f.loc[series].values,
            "o-", lw=1.5, markersize=6, label=series)
ax.axhline(3.84, color="red", linestyle="--", lw=1.5,
           label="F critical (p=0.05, df=1)")
ax.set_xlabel("Lag (months)")
ax.set_ylabel("F-statistic")
ax.set_title("Granger F-statistics by Lag\n(above dashed line = significant)")
ax.legend(fontsize=8)
ax.set_xticks(range(1, MAX_LAGS + 1))
ax.grid(alpha=0.3)

# Panel 3: IRF
ax = axes[1, 0]
if irf_to_greek:
    periods = list(range(len(list(irf_to_greek.values())[0])))
    for us_col, irf_vals in irf_to_greek.items():
        label = us_col.replace("US_", "").replace("_PPI", "")
        ax.plot(periods, irf_vals, "o-", lw=1.5, markersize=5, label=label)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Months after shock")
    ax.set_ylabel("Response of Greek vol")
    ax.set_title("Impulse Response Functions (VAR)\n"
                 "Response of Greek vol to unit US PPI shock")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
else:
    ax.text(0.5, 0.5, "IRF not available\n(VAR fitting failed)",
            ha="center", va="center", transform=ax.transAxes)

# Panel 4: Timeline of Greek vol with significant US series
ax = axes[1, 1]
sig_series = df_granger[df_granger["significant"]].sort_values("p_value")
ax2 = ax.twinx()
ax.fill_between(common, 0, (crisis_al > 0).astype(float),
                color="red", alpha=0.15, label="Crisis regime")
ax.plot(common, vol_al, color="blue", lw=1.5,
        label=f"{TARGET_MAT} vol")
ax.set_ylabel("Greek vol", color="blue")
ax.tick_params(axis="y", labelcolor="blue")

if len(sig_series) > 0:
    best_us = sig_series.iloc[0]["US_series"]
    best_lag = int(sig_series.iloc[0]["lag_months"])
    us_lagged = raw_al[best_us].shift(best_lag)
    us_norm = (us_lagged - us_lagged.mean()) / us_lagged.std() * vol_al.std() + vol_al.mean()
    ax2.plot(common, us_norm.loc[common], color="orange", lw=1.5,
             linestyle="--", alpha=0.8,
             label=f"{best_us} (lag={best_lag}M, normalized)")
    ax2.set_ylabel("US PPI (normalized)", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")

ax.set_title(f"Greek Vol vs Best Granger Predictor\n"
             f"(leading by {best_lag if len(sig_series)>0 else 'N/A'}M)")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.grid(alpha=0.3)

plt.tight_layout()
fig_path = RESULTS_DIR + "fig_p4_granger.pdf"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Figure saved: {fig_path}")
plt.show()

print("\n" + "=" * 60)
print("DONE -- Granger causality complete.")
print(f"Significant bivariate pairs: {df_granger['significant'].sum()}")
print(f"VAR joint p-value: {var_joint_p:.4f}" if not np.isnan(var_joint_p)
      else "VAR: not available")
print("Next: run 11_crisis_backtests.py")
print("=" * 60) 