"""
06_oos_forecast.py
Out-of-Sample Rolling-Window VAR Forecast Validation

For each matched US->Greek pair:
  1. Train bivariate VAR on expanding window (start=2000, min train=180 obs)
  2. 1-step-ahead forecast of Greek series, rolling month-by-month (2015-2024)
  3. Compare vs naive forecast (random walk: forecast = last observed value)
  4. Metrics: RMSE, MAE, Diebold-Mariano test
  5. Also compare: VAR-with-US-input vs AR(p)-only-Greek (no US information)
     This isolates the marginal value of US information.

Input:  data/processed/aligned_log_returns.csv
Output: results/tables/c6_oos_forecast.csv
        results/figures/fig_c6_oos_forecast.pdf
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.ar_model import AutoReg
from scipy import stats
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "processed",
                          "aligned_log_returns.csv")
OUT_FIG = os.path.join(SCRIPT_DIR, "..", "results", "figures")
OUT_TAB = os.path.join(SCRIPT_DIR, "..", "results", "tables")

# OOS period: 2015-01 to 2024-12 (~120 months)
OOS_START = "2015-01-01"
MIN_TRAIN = 180  # minimum training observations

PAIRS = [
    ("US_Steel_PPI",  "GR_Steel",        "Steel"),
    ("US_Fuel_PPI",   "GR_Fuel_Energy",  "Fuel/Energy"),
    ("US_Cement_PPI", "GR_Concrete",     "Cement/Concrete"),
    ("US_PVC_PPI",    "GR_PVC_Pipes",    "PVC/Plastic"),
    ("US_Brent",      "GR_General_Index", "Brent/General"),
]

# Series that need differencing (from ADF tests)
NONSTAT = {"US_Cement_PPI"}


def diebold_mariano(e1, e2, h=1):
    """Diebold-Mariano test for equal predictive accuracy.
    H0: both forecasts equally accurate.
    Returns (DM statistic, p-value). Negative DM = model 1 better."""
    d = e1**2 - e2**2
    d_bar = np.mean(d)
    n = len(d)
    # Newey-West variance with h-1 lags
    gamma_0 = np.var(d, ddof=1)
    if h > 1:
        for k in range(1, h):
            gamma_k = np.cov(d[k:], d[:-k])[0, 1]
            gamma_0 += 2 * (1 - k / h) * gamma_k
    se = np.sqrt(gamma_0 / n)
    if se < 1e-12:
        return 0.0, 1.0
    dm = d_bar / se
    pval = 2 * (1 - stats.norm.cdf(abs(dm)))
    return dm, pval


def main():
    os.makedirs(OUT_FIG, exist_ok=True)
    os.makedirs(OUT_TAB, exist_ok=True)

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    print(f"Data: {df.shape}")
    print(f"OOS period: {OOS_START} onwards")

    results = []
    fig, axes = plt.subplots(len(PAIRS), 1, figsize=(10, 3 * len(PAIRS)),
                              sharex=False)

    for idx, (us_col, gr_col, label) in enumerate(PAIRS):
        if us_col not in df.columns or gr_col not in df.columns:
            continue

        print(f"\n{'='*60}")
        print(f"OOS Forecast: {us_col} -> {gr_col} ({label})")
        print(f"{'='*60}")

        pair_df = df[[us_col, gr_col]].dropna().copy()

        # Apply differencing if needed
        if us_col in NONSTAT:
            pair_df[us_col] = pair_df[us_col].diff()
            pair_df = pair_df.dropna()
            print(f"  First-differenced: {us_col}")

        # Define OOS indices
        oos_mask = pair_df.index >= OOS_START
        oos_dates = pair_df.index[oos_mask]

        if len(oos_dates) < 12:
            print(f"  SKIP: too few OOS observations ({len(oos_dates)})")
            continue

        print(f"  Training: {pair_df.index[0].strftime('%Y-%m')} to "
              f"{oos_dates[0].strftime('%Y-%m')}")
        print(f"  OOS: {oos_dates[0].strftime('%Y-%m')} to "
              f"{oos_dates[-1].strftime('%Y-%m')} ({len(oos_dates)} months)")

        actual = []
        fc_var = []
        fc_ar = []
        fc_naive = []
        fc_dates = []

        for i, oos_date in enumerate(oos_dates):
            oos_pos = pair_df.index.get_loc(oos_date)

            if oos_pos < MIN_TRAIN:
                continue

            train = pair_df.iloc[:oos_pos]
            true_val = pair_df[gr_col].iloc[oos_pos]

            # --- VAR forecast (with US info) ---
            try:
                var_model = VAR(train)
                lo = var_model.select_order(maxlags=8)
                p = max(lo.aic, 1)
                fitted = var_model.fit(p)
                fc = fitted.forecast(train.values[-p:], steps=1)
                var_pred = fc[0, 1]  # Greek is column index 1
            except Exception:
                var_pred = train[gr_col].iloc[-1]

            # --- AR forecast (Greek only, no US info) ---
            try:
                ar_model = AutoReg(train[gr_col].values, lags=min(4, len(train) - 1))
                ar_fit = ar_model.fit()
                ar_pred = ar_fit.forecast(steps=1)[0]
            except Exception:
                ar_pred = train[gr_col].iloc[-1]

            # --- Naive forecast (random walk = last value) ---
            naive_pred = train[gr_col].iloc[-1]

            actual.append(true_val)
            fc_var.append(var_pred)
            fc_ar.append(ar_pred)
            fc_naive.append(naive_pred)
            fc_dates.append(oos_date)

        actual = np.array(actual)
        fc_var = np.array(fc_var)
        fc_ar = np.array(fc_ar)
        fc_naive = np.array(fc_naive)

        # Errors
        e_var = actual - fc_var
        e_ar = actual - fc_ar
        e_naive = actual - fc_naive

        # Metrics
        rmse_var = np.sqrt(np.mean(e_var**2))
        rmse_ar = np.sqrt(np.mean(e_ar**2))
        rmse_naive = np.sqrt(np.mean(e_naive**2))

        mae_var = np.mean(np.abs(e_var))
        mae_ar = np.mean(np.abs(e_ar))
        mae_naive = np.mean(np.abs(e_naive))

        # RMSE improvement vs naive
        rmse_improv_vs_naive = (1 - rmse_var / rmse_naive) * 100
        rmse_improv_vs_ar = (1 - rmse_var / rmse_ar) * 100

        # Diebold-Mariano tests
        dm_vs_naive, dm_p_naive = diebold_mariano(e_var, e_naive)
        dm_vs_ar, dm_p_ar = diebold_mariano(e_var, e_ar)

        print(f"  RMSE: VAR={rmse_var:.6f}  AR={rmse_ar:.6f}  "
              f"Naive={rmse_naive:.6f}")
        print(f"  MAE:  VAR={mae_var:.6f}  AR={mae_ar:.6f}  "
              f"Naive={mae_naive:.6f}")
        print(f"  RMSE improvement vs Naive: {rmse_improv_vs_naive:+.1f}%")
        print(f"  RMSE improvement vs AR:    {rmse_improv_vs_ar:+.1f}%")
        print(f"  DM test (VAR vs Naive): DM={dm_vs_naive:.3f}, "
              f"p={dm_p_naive:.4f}"
              f" {'***' if dm_p_naive < 0.001 else '**' if dm_p_naive < 0.01 else '*' if dm_p_naive < 0.05 else ''}")
        print(f"  DM test (VAR vs AR):    DM={dm_vs_ar:.3f}, "
              f"p={dm_p_ar:.4f}"
              f" {'***' if dm_p_ar < 0.001 else '**' if dm_p_ar < 0.01 else '*' if dm_p_ar < 0.05 else ''}")

        results.append({
            "Pair": label,
            "N_oos": len(actual),
            "RMSE_VAR": round(rmse_var, 6),
            "RMSE_AR": round(rmse_ar, 6),
            "RMSE_Naive": round(rmse_naive, 6),
            "MAE_VAR": round(mae_var, 6),
            "MAE_AR": round(mae_ar, 6),
            "MAE_Naive": round(mae_naive, 6),
            "RMSE_improv_vs_Naive_%": round(rmse_improv_vs_naive, 1),
            "RMSE_improv_vs_AR_%": round(rmse_improv_vs_ar, 1),
            "DM_vs_Naive": round(dm_vs_naive, 3),
            "DM_p_Naive": round(dm_p_naive, 4),
            "DM_vs_AR": round(dm_vs_ar, 3),
            "DM_p_AR": round(dm_p_ar, 4),
        })

        # Plot
        ax = axes[idx]
        ax.plot(fc_dates, actual, "k-", linewidth=0.8, alpha=0.6, label="Actual")
        ax.plot(fc_dates, fc_var, "b-", linewidth=1.2, label="VAR (US+GR)")
        ax.plot(fc_dates, fc_ar, "r--", linewidth=0.8, alpha=0.7, label="AR (GR only)")
        ax.set_ylabel(label, fontsize=9)
        ax.legend(fontsize=6, loc="upper right")
        # Annotate RMSE improvement
        txt = f"RMSE vs Naive: {rmse_improv_vs_naive:+.1f}%"
        if rmse_improv_vs_naive > 0:
            txt += " (better)"
        ax.text(0.02, 0.85, txt, transform=ax.transAxes, fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          alpha=0.8))

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG, "fig_c6_oos_forecast.pdf"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("\nfig_c6_oos_forecast.pdf saved.")

    # Save results table
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(OUT_TAB, "c6_oos_forecast.csv"), index=False)
    print(f"\n{'='*60}")
    print("OUT-OF-SAMPLE FORECAST SUMMARY")
    print(f"{'='*60}")
    print(res_df[["Pair", "RMSE_VAR", "RMSE_AR", "RMSE_Naive",
                   "RMSE_improv_vs_Naive_%", "DM_p_Naive",
                   "RMSE_improv_vs_AR_%", "DM_p_AR"]].to_string(index=False))

    # Count how many pairs show improvement
    n_better_naive = (res_df["RMSE_improv_vs_Naive_%"] > 0).sum()
    n_better_ar = (res_df["RMSE_improv_vs_AR_%"] > 0).sum()
    n_sig_naive = (res_df["DM_p_Naive"] < 0.05).sum()
    n_sig_ar = (res_df["DM_p_AR"] < 0.05).sum()

    print(f"\nVAR beats Naive: {n_better_naive}/{len(res_df)} pairs "
          f"({n_sig_naive} significant)")
    print(f"VAR beats AR:    {n_better_ar}/{len(res_df)} pairs "
          f"({n_sig_ar} significant)")


if __name__ == "__main__":
    main()
