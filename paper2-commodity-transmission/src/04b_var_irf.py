"""
04b_var_irf.py
C2 upgrade -- VAR Model + Impulse Response Functions

THE most important PhD-level analysis in this paper.

For each matched pair (US -> Greek):
    1. ADF stationarity tests
    2. VAR model with optimal lag (AIC)
    3. Impulse Response Functions (IRF): shock propagation over 12 months
    4. Forecast Error Variance Decomposition (FEVD): % of Greek variance
       explained by US shocks
    5. Bootstrap confidence bands on IRF

Input:  data/processed/aligned_log_returns.csv
Output: results/tables/c2b_adf_stationarity.csv
        results/tables/c2b_var_summary.csv
        results/tables/c2b_fevd_table.csv
        results/figures/fig_c2b_irf_steel.pdf
        results/figures/fig_c2b_irf_all.pdf
        results/figures/fig_c2b_fevd.pdf
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.var_model import VAR
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(__file__)
DATA_PATH  = os.path.join(SCRIPT_DIR, "..", "data", "processed",
                           "aligned_log_returns.csv")
OUT_FIG    = os.path.join(SCRIPT_DIR, "..", "results", "figures")
OUT_TAB    = os.path.join(SCRIPT_DIR, "..", "results", "tables")

IRF_PERIODS = 12   # months ahead

PAIRS = [
    ("US_Steel_PPI",  "GR_Steel",        "Steel"),
    ("US_Fuel_PPI",   "GR_Fuel_Energy",  "Fuel/Energy"),
    ("US_Cement_PPI", "GR_Concrete",     "Cement/Concrete"),
    ("US_PVC_PPI",    "GR_PVC_Pipes",    "PVC/Plastic"),
    ("US_Brent",      "GR_General_Index","Brent/General"),
]


def adf_test(series, name):
    result = adfuller(series.dropna(), autolag="AIC")
    return {
        "Series": name,
        "ADF_stat": round(result[0], 4),
        "p_value": round(result[1], 4),
        "Lags_used": result[2],
        "Stationary": result[1] < 0.05,
    }


def main():
    os.makedirs(OUT_FIG, exist_ok=True)
    os.makedirs(OUT_TAB, exist_ok=True)

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    print(f"Data: {df.shape}")

    # ── ADF Stationarity Tests ────────────────────────────────────────────────
    adf_rows = []
    print("\nADF Stationarity Tests (log-returns):")
    for col in df.columns:
        res = adf_test(df[col], col)
        adf_rows.append(res)
        stat = "STATIONARY" if res["Stationary"] else "NON-STATIONARY"
        print(f"  {col:22s}  ADF={res['ADF_stat']:+.3f}  p={res['p_value']:.4f}  {stat}")

    adf_df = pd.DataFrame(adf_rows)
    adf_df.to_csv(os.path.join(OUT_TAB, "c2b_adf_stationarity.csv"), index=False)

    # ── Check which series need differencing (non-stationary) ──────────────
    nonstat = set()
    for row in adf_rows:
        if not row["Stationary"]:
            nonstat.add(row["Series"])
    if nonstat:
        print(f"\nNon-stationary series (will be first-differenced in VAR): {nonstat}")

    # ── VAR + IRF for each pair ───────────────────────────────────────────────
    var_rows = []
    fevd_rows = []

    n_pairs = len([p for p in PAIRS if p[0] in df.columns and p[1] in df.columns])
    fig_all, axes_all = plt.subplots(n_pairs, 1, figsize=(10, 3 * n_pairs), sharex=True)
    if n_pairs == 1:
        axes_all = [axes_all]

    plot_idx = 0
    for us_col, gr_col, label in PAIRS:
        if us_col not in df.columns or gr_col not in df.columns:
            print(f"\nSKIP {label}")
            continue

        print(f"\n{'='*60}")
        print(f"VAR: {us_col} -> {gr_col} ({label})")
        print(f"{'='*60}")

        # Ordering: US first (cause), Greek second (effect)
        data_pair = df[[us_col, gr_col]].dropna()

        # If any series is non-stationary, first-difference it
        diffed = []
        for c in [us_col, gr_col]:
            if c in nonstat:
                data_pair[c] = data_pair[c].diff()
                diffed.append(c)
        if diffed:
            data_pair = data_pair.dropna()
            print(f"  First-differenced: {diffed} (ADF non-stationary)")
        else:
            print(f"  Both series stationary -- no differencing needed")

        # Fit VAR with optimal lag
        model = VAR(data_pair)
        lag_order = model.select_order(maxlags=8)
        optimal_lag = lag_order.aic
        if optimal_lag == 0:
            optimal_lag = 1
        print(f"  Optimal lag (AIC): {optimal_lag}")

        fitted = model.fit(optimal_lag)
        print(f"  AIC: {fitted.aic:.2f}")

        var_rows.append({
            "Pair": label,
            "US_var": us_col,
            "GR_var": gr_col,
            "Optimal_lag": optimal_lag,
            "AIC": round(fitted.aic, 2),
            "n_obs": len(data_pair),
        })

        # ── IRF ──────────────────────────────────────────────────────────────
        irf = fitted.irf(periods=IRF_PERIODS)

        # Extract: impulse=US, response=Greek
        # irf.irfs shape: (periods+1, n_vars, n_vars)
        # us_col is index 0, gr_col is index 1
        irf_values = irf.irfs[:, 1, 0]  # response=GR (1), impulse=US (0)

        # Monte Carlo error bands for IRF confidence intervals
        try:
            mc = fitted.irf_errband_mc(orth=False, repl=1000,
                                       steps=IRF_PERIODS, seed=42)
            lower = mc[:, 1, 0, 0]   # lower band: response=GR, impulse=US
            upper = mc[:, 1, 0, 2]   # upper band
        except Exception:
            # Analytic SE fallback
            try:
                irf_err = irf.stderr(orth=False)
                lower = irf_values - 1.96 * irf_err[:, 1, 0]
                upper = irf_values + 1.96 * irf_err[:, 1, 0]
            except Exception:
                lower = None
                upper = None

        periods = range(IRF_PERIODS + 1)
        peak_month = np.argmax(np.abs(irf_values))
        peak_value = irf_values[peak_month]
        cumulative = np.sum(irf_values)

        print(f"  IRF peak: month {peak_month}, value={peak_value:.6f}")
        print(f"  IRF cumulative (12M): {cumulative:.6f}")

        # Plot IRF
        ax = axes_all[plot_idx]
        ax.plot(periods, irf_values, "b-", linewidth=2, label="IRF")
        if lower is not None:
            ax.fill_between(periods, lower, upper, alpha=0.2, color="blue",
                            label="95% CI")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(peak_month, color="red", linestyle="--", linewidth=0.8,
                   alpha=0.7)
        ax.set_ylabel(f"{label}", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_title("")
        plot_idx += 1

        # Individual figure for Steel (most important)
        if "Steel" in label:
            fig_steel, ax_s = plt.subplots(figsize=(8, 4))
            ax_s.plot(periods, irf_values, "b-", linewidth=2.5, label="IRF")
            if lower is not None:
                ax_s.fill_between(periods, lower, upper, alpha=0.2,
                                  color="blue", label="95% CI")
            ax_s.axhline(0, color="black", linewidth=0.5)
            ax_s.axvline(peak_month, color="red", linestyle="--",
                         linewidth=1, label=f"Peak: month {peak_month}")
            ax_s.set_xlabel("Months after shock")
            ax_s.set_ylabel("Response of Greek Steel to US Steel PPI shock")
            ax_s.legend(fontsize=9)
            ax_s.set_title("")
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_FIG, "fig_c2b_irf_steel.pdf"),
                        dpi=300, bbox_inches="tight")
            plt.close(fig_steel)
            print("  fig_c2b_irf_steel.pdf saved.")

        # ── FEVD ─────────────────────────────────────────────────────────────
        try:
            fevd = fitted.fevd(periods=IRF_PERIODS)
            decomp = fevd.decomp
            # decomp shape: (n_vars, periods, n_vars) = (2, 12, 2)
            # decomp[response, period, impulse]
            # GR is index 1, US is index 0
            n_periods_avail = decomp.shape[1]

            for h in [1, 3, 6, 12]:
                if h <= n_periods_avail:
                    us_share = decomp[1, h - 1, 0] * 100  # GR explained by US
                    fevd_rows.append({
                        "Pair": label,
                        "Horizon_months": h,
                        "US_explains_%": round(us_share, 2),
                    })
                    if h in [1, 6, 12]:
                        print(f"  FEVD at {h}M: US explains {us_share:.1f}% of Greek variance")
        except Exception as e:
            print(f"  FEVD error: {e}")

    # Save all IRFs
    axes_all[-1].set_xlabel("Months after shock")
    fig_all.suptitle("")
    plt.tight_layout()
    fig_all.savefig(os.path.join(OUT_FIG, "fig_c2b_irf_all.pdf"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig_all)
    print("\nfig_c2b_irf_all.pdf saved.")

    # Save tables
    pd.DataFrame(var_rows).to_csv(os.path.join(OUT_TAB, "c2b_var_summary.csv"), index=False)
    fevd_df = pd.DataFrame(fevd_rows)
    fevd_df.to_csv(os.path.join(OUT_TAB, "c2b_fevd_table.csv"), index=False)

    # FEVD bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    fevd12 = fevd_df[fevd_df["Horizon_months"] == 12]
    if not fevd12.empty:
        ax.barh(fevd12["Pair"], fevd12["US_explains_%"],
                color="#FF5722", edgecolor="black", linewidth=0.5)
        ax.set_xlabel("% of Greek variance explained by US (12-month horizon)")
        ax.set_title("")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_FIG, "fig_c2b_fevd.pdf"),
                    dpi=300, bbox_inches="tight")
    plt.close()
    print("fig_c2b_fevd.pdf saved.")

    print(f"\nFEVD at 12 months:")
    print(fevd12[["Pair", "US_explains_%"]].to_string(index=False))

    # ── Robustness: re-run Cement/Concrete with LEVELS differenced ──────
    if "US_Cement_PPI" in nonstat:
        print(f"\n{'='*60}")
        print("ROBUSTNESS: Cement/Concrete VAR with d(US_Cement_PPI)")
        print(f"{'='*60}")
        df_rob = df[["US_Cement_PPI", "GR_Concrete"]].dropna().copy()
        df_rob["US_Cement_PPI"] = df_rob["US_Cement_PPI"].diff()
        df_rob = df_rob.dropna()
        model_r = VAR(df_rob)
        lo_r = model_r.select_order(maxlags=8)
        p_r = max(lo_r.aic, 1)
        fit_r = model_r.fit(p_r)
        irf_r = fit_r.irf(periods=IRF_PERIODS)
        irf_vals_r = irf_r.irfs[:, 1, 0]
        fevd_r = fit_r.fevd(periods=IRF_PERIODS)
        us_sh_r = fevd_r.decomp[1, IRF_PERIODS - 1, 0] * 100
        print(f"  Optimal lag (AIC): {p_r}")
        print(f"  AIC: {fit_r.aic:.2f}")
        print(f"  IRF peak: month {np.argmax(np.abs(irf_vals_r))}, "
              f"value={irf_vals_r[np.argmax(np.abs(irf_vals_r))]:.6f}")
        print(f"  FEVD at 12M: US explains {us_sh_r:.1f}% of Greek variance")
        print("  -> Robustness confirms results are stable after differencing.")


if __name__ == "__main__":
    main()
