"""
Module: 11_bootstrap_ci.py
Description: Parametric Bootstrap Confidence Intervals for the Hidden Risk Gap.
             Quantifies estimation uncertainty in the Gumbel P85 risk gap arising
             from finite-sample Kendall tau estimation in the Crisis Regime (n~47).
             Method: Percentile bootstrap, B=1000 resamples.
             Two analyses:
               (A) CRISIS REGIME  (2021-2024): tau bootstrapped from actual crisis data
               (B) FULL PERIOD    (2000-2024): tau bootstrapped from full dataset
             Output: bootstrap_ci_summary.csv  (for embedding in paper Table 3 note)
                     bootstrap_ci_results.csv   (raw bootstrap draws)
Author: Dimitrios Chronis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import norm, kendalltau
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ── CONFIG ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_PATH   = BASE_DIR / 'data' / 'processed' / 'clean_returns.csv'
RESULTS_DIR = BASE_DIR / 'results' / 'tables'

B_RESAMPLES     = 1000       # bootstrap iterations
N_SIMS          = 20_000     # MC sims per bootstrap (speed/accuracy balance)
DURATION_MONTHS = 36         # worst-case scenario (Table 3 row: Crisis 36M)
BASE_COST       = 2_300_000.0
MAX_VOL         = 0.15       # same cap as 03_detailed_simulation.py
WEIGHTS = {'Concrete': 0.30, 'Steel': 0.30, 'Fuel_Energy': 0.20, 'PVC_Pipes': 0.20}

CRISIS_START = 2021
CRISIS_END   = 2024
SEED         = 42


# ── GUMBEL SAMPLER (Marshall-Olkin, identical to 03_detailed_simulation.py) ─
def gumbel_sample(theta: float, n_sims: int, n_assets: int,
                  rng: np.random.Generator) -> np.ndarray:
    alpha = 1.0 / theta
    if alpha >= 0.99:
        return rng.uniform(0, 1, (n_sims, n_assets))
    U = rng.uniform(1e-4, np.pi - 1e-4, n_sims)
    W = rng.exponential(1.0, n_sims) + 1e-4
    term1 = np.sin(alpha * U) / (np.sin(U) ** (1.0 / alpha))
    term2 = (np.sin((1.0 - alpha) * U) / W) ** ((1.0 - alpha) / alpha)
    V = np.clip(term1 * term2, 1e-9, 1e5)[:, None]
    Y = rng.uniform(0, 1, (n_sims, n_assets))
    return np.exp(-(-np.log(np.clip(Y, 1e-9, 1 - 1e-9)) / V) ** alpha)


# ── COST CALCULATOR ──────────────────────────────────────────────────────────
def calc_cost(u_mat: np.ndarray, stds: list, col_names: list) -> np.ndarray:
    u_mat  = np.clip(u_mat, 1e-6, 1 - 1e-6)
    impact = np.zeros(u_mat.shape[0])
    for i, col in enumerate(col_names):
        if col in WEIGHTS:
            ret     = norm.ppf(u_mat[:, i], loc=0.0, scale=stds[i])
            impact += WEIGHTS[col] * (np.exp(ret) - 1.0)
    return BASE_COST * (1.0 + impact)


# ── ONE BOOTSTRAP ITERATION ──────────────────────────────────────────────────
def run_one_bootstrap(boot_returns: pd.DataFrame,
                      rng: np.random.Generator) -> tuple:
    """
    Fit tau/theta from one bootstrap resample, run mini Monte Carlo,
    return (p85_gap_eur, avg_tau, theta, p85_ind, p85_gum).
    """
    col_names = list(boot_returns.columns)
    n_assets  = len(col_names)
    scale     = np.sqrt(DURATION_MONTHS)

    # Volatility (vol-throttled, zero-drift — matches paper methodology)
    stds = []
    for col in col_names:
        _, std = norm.fit(boot_returns[col])
        stds.append(min(std, MAX_VOL) * scale)

    # Average pairwise Kendall tau
    tau_vals = []
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            t, _ = kendalltau(boot_returns.iloc[:, i], boot_returns.iloc[:, j])
            tau_vals.append(t)
    avg_tau = float(np.clip(np.mean(tau_vals), 0.05, 0.95))
    theta   = 1.0 / (1.0 - avg_tau)

    # Independent model
    u_ind      = rng.uniform(0, 1, (N_SIMS, n_assets))
    costs_ind  = calc_cost(u_ind, stds, col_names)
    p85_ind    = float(np.percentile(costs_ind, 85))

    # Gumbel model
    u_gum      = gumbel_sample(theta, N_SIMS, n_assets, rng)
    costs_gum  = calc_cost(u_gum, stds, col_names)
    p85_gum    = float(np.percentile(costs_gum, 85))

    return p85_gum - p85_ind, avg_tau, theta, p85_ind, p85_gum


# ── BOOTSTRAP ENGINE ─────────────────────────────────────────────────────────
def bootstrap_analysis(returns: pd.DataFrame,
                       label: str,
                       rng: np.random.Generator) -> dict:
    n_obs = len(returns)
    logger.info(f"[{label}] n_obs={n_obs}, B={B_RESAMPLES}, N_sims={N_SIMS}, T={DURATION_MONTHS}M")

    gaps, taus, thetas, p85_inds, p85_gums = [], [], [], [], []

    for b in range(B_RESAMPLES):
        if (b + 1) % 200 == 0:
            logger.info(f"  [{label}] Bootstrap {b+1}/{B_RESAMPLES}...")
        idx       = rng.integers(0, n_obs, size=n_obs)
        boot_data = returns.iloc[idx].reset_index(drop=True)
        try:
            gap, tau, theta, p85i, p85g = run_one_bootstrap(boot_data, rng)
            gaps.append(gap);   taus.append(tau)
            thetas.append(theta); p85_inds.append(p85i); p85_gums.append(p85g)
        except Exception as e:
            logger.warning(f"  [{label}] Bootstrap {b} skipped: {e}")

    gaps   = np.array(gaps)
    taus   = np.array(taus)
    thetas = np.array(thetas)

    summary = {
        'label':            label,
        'n_obs':            n_obs,
        'n_valid':          len(gaps),
        'duration_months':  DURATION_MONTHS,
        # Tau
        'tau_mean':         round(float(np.mean(taus)), 3),
        'tau_median':       round(float(np.median(taus)), 3),
        'tau_ci_low_95':    round(float(np.percentile(taus, 2.5)), 3),
        'tau_ci_high_95':   round(float(np.percentile(taus, 97.5)), 3),
        # Theta
        'theta_mean':       round(float(np.mean(thetas)), 3),
        'theta_median':     round(float(np.median(thetas)), 3),
        'theta_ci_low_95':  round(float(np.percentile(thetas, 2.5)), 3),
        'theta_ci_high_95': round(float(np.percentile(thetas, 97.5)), 3),
        # P85 Gap (EUR)
        'gap_mean_eur':     round(float(np.mean(gaps))),
        'gap_median_eur':   round(float(np.median(gaps))),
        'gap_ci_low_95':    round(float(np.percentile(gaps, 2.5))),
        'gap_ci_high_95':   round(float(np.percentile(gaps, 97.5))),
        'gap_pct_mean':     round(float(np.mean(gaps)) / BASE_COST * 100, 2),
        'gap_pct_ci_low':   round(float(np.percentile(gaps, 2.5)) / BASE_COST * 100, 2),
        'gap_pct_ci_high':  round(float(np.percentile(gaps, 97.5)) / BASE_COST * 100, 2),
    }

    # Console report
    logger.info(f"\n{'='*60}")
    logger.info(f"  BOOTSTRAP RESULTS: {label}")
    logger.info(f"  n_obs={n_obs}, B={len(gaps)} valid resamples")
    logger.info(f"  Kendall tau : mean={summary['tau_mean']:.3f},"
                f" 95% CI [{summary['tau_ci_low_95']:.3f}, {summary['tau_ci_high_95']:.3f}]")
    logger.info(f"  Gumbel theta: mean={summary['theta_mean']:.3f},"
                f" 95% CI [{summary['theta_ci_low_95']:.3f}, {summary['theta_ci_high_95']:.3f}]")
    logger.info(f"  P85 Gap EUR : mean={summary['gap_mean_eur']:,},"
                f" median={summary['gap_median_eur']:,},"
                f" 95% CI [{summary['gap_ci_low_95']:,}, {summary['gap_ci_high_95']:,}]")
    logger.info(f"  P85 Gap %   : mean={summary['gap_pct_mean']:.2f}%,"
                f" 95% CI [{summary['gap_pct_ci_low']:.2f}%, {summary['gap_pct_ci_high']:.2f}%]")
    logger.info(f"{'='*60}")

    return summary, gaps, taus, thetas


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df_all = pd.read_csv(DATA_PATH, index_col='Date', parse_dates=True)
    all_returns = np.log(df_all / df_all.shift(1)).dropna()

    col_order = ['Concrete', 'Steel', 'Fuel_Energy', 'PVC_Pipes']
    all_returns = all_returns[col_order]

    crisis_mask    = (all_returns.index.year >= CRISIS_START) & \
                     (all_returns.index.year <= CRISIS_END)
    crisis_returns = all_returns[crisis_mask]
    full_returns   = all_returns

    rng = np.random.default_rng(SEED)

    # ── ANALYSIS A: CRISIS REGIME ────────────────────────────────────────────
    logger.info("\n>>> ANALYSIS A: Crisis Regime Bootstrap (2021-2024)")
    summary_crisis, gaps_c, taus_c, thetas_c = bootstrap_analysis(
        crisis_returns, "Crisis (2021-2024)", rng
    )

    # ── ANALYSIS B: FULL PERIOD ──────────────────────────────────────────────
    logger.info("\n>>> ANALYSIS B: Full-Period Bootstrap (2000-2024)")
    summary_full, gaps_f, taus_f, thetas_f = bootstrap_analysis(
        full_returns, "Full Period (2000-2024)", rng
    )

    # ── SAVE RAW RESULTS ─────────────────────────────────────────────────────
    raw = pd.DataFrame({
        'crisis_gap_eur':   gaps_c,
        'crisis_tau':       taus_c,
        'crisis_theta':     thetas_c,
        'full_gap_eur':     gaps_f[:len(gaps_c)],
        'full_tau':         taus_f[:len(gaps_c)],
        'full_theta':       thetas_f[:len(gaps_c)],
    })
    raw.to_csv(RESULTS_DIR / 'bootstrap_ci_results.csv', index=False)
    logger.info(f"Raw bootstrap draws -> {RESULTS_DIR / 'bootstrap_ci_results.csv'}")

    # ── SAVE SUMMARY ─────────────────────────────────────────────────────────
    summary_df = pd.DataFrame([summary_crisis, summary_full])
    summary_df.to_csv(RESULTS_DIR / 'bootstrap_ci_summary.csv', index=False)
    logger.info(f"Summary -> {RESULTS_DIR / 'bootstrap_ci_summary.csv'}")

    # ── PAPER-READY SNIPPET ──────────────────────────────────────────────────
    logger.info("\n\n>>> PAPER-READY NUMBERS (copy into Table 3 footnote / Section 4.5):")
    logger.info(f"Crisis Regime (2021-2024), T=36 months, B={summary_crisis['n_valid']} resamples:")
    logger.info(f"  Hidden Risk Gap: EUR {summary_crisis['gap_mean_eur']:,}"
                f"  (95% CI: EUR {summary_crisis['gap_ci_low_95']:,}"
                f" -- EUR {summary_crisis['gap_ci_high_95']:,})")
    logger.info(f"  As % of base cost: {summary_crisis['gap_pct_mean']:.2f}%"
                f"  (95% CI: {summary_crisis['gap_pct_ci_low']:.2f}%"
                f" -- {summary_crisis['gap_pct_ci_high']:.2f}%)")
    logger.info(f"  Kendall tau range: [{summary_crisis['tau_ci_low_95']:.3f},"
                f" {summary_crisis['tau_ci_high_95']:.3f}]")
    logger.info(f"  Gumbel theta range: [{summary_crisis['theta_ci_low_95']:.3f},"
                f" {summary_crisis['theta_ci_high_95']:.3f}]")

    logger.info(f"\nFull Period (2000-2024), T=36 months, B={summary_full['n_valid']} resamples:")
    logger.info(f"  Hidden Risk Gap: EUR {summary_full['gap_mean_eur']:,}"
                f"  (95% CI: EUR {summary_full['gap_ci_low_95']:,}"
                f" -- EUR {summary_full['gap_ci_high_95']:,})")
    logger.info(f"  As % of base cost: {summary_full['gap_pct_mean']:.2f}%"
                f"  (95% CI: {summary_full['gap_pct_ci_low']:.2f}%"
                f" -- {summary_full['gap_pct_ci_high']:.2f}%)")

    logger.info("\nDone.")


if __name__ == '__main__':
    main()
