"""
Module: 13_gumbel_gap_table.py
Description:
    Correct exchangeable Gumbel-copula Monte-Carlo simulator using
    Marshall-Olkin (positive-stable) sampling. Computes the P85
    "hidden risk gap" under BOTH the MLE-fitted base-case dependence
    (theta = 1.141, tau ~ 0.12) AND the stress-scenario dependence
    (theta = 6.67, tau = 0.85), across 12/24/36-month horizons.

    Addresses Reviewer #1's central concern that the headline EUR 45,806
    figure derived from the stress scenario only, with no base-case
    counterpart. This script reports the full range.

    Reproducibility: fully seeded; sampler self-validated against the
    analytic Gumbel relation tau = 1 - 1/theta.

Author: Dimitrios Chronis
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, kendalltau

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_PATH  = BASE_DIR / "data" / "processed" / "clean_returns.csv"
TABLES_DIR = BASE_DIR / "results" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

SEED       = 42
N_SIMS     = 100_000
BASE_COST  = 2_300_000.0          # EUR
HORIZONS   = [12, 24, 36]         # months
P_LEVEL    = 85                   # percentile for capital adequacy

# Representative project portfolio weights
WEIGHTS = {"Concrete": 0.30, "Steel": 0.30, "Fuel_Energy": 0.20, "PVC_Pipes": 0.20}
MATERIALS = list(WEIGHTS.keys())

# Dependence parameterisations (Gumbel theta)
THETA_MLE    = 1.141     # MLE fit, Crisis Regime 2021-2024 (tau = 0.124)
THETA_STRESS = 6.67      # stress, from peak rolling tau = 0.85
THETA_STABLE = 1.168     # MLE fit, Stable Regime 2014-2019 (tau = 0.144)


# --------------------------------------------------------------------------
# Exchangeable Gumbel copula sampler (Marshall-Olkin / positive stable)
# --------------------------------------------------------------------------
def sample_positive_stable(alpha, size, rng):
    """Sample a positive (one-sided) alpha-stable subordinator with
    Laplace transform E[exp(-tS)] = exp(-t**alpha), 0 < alpha < 1,
    via the Kanter (1975) representation."""
    U = rng.uniform(0.0, np.pi, size=size)
    W = rng.exponential(1.0, size=size)
    term1 = np.sin(alpha * U) / np.power(np.sin(U), 1.0 / alpha)
    term2 = np.power(np.sin((1.0 - alpha) * U) / W, (1.0 - alpha) / alpha)
    return term1 * term2


def sample_gumbel_copula(n, d, theta, rng):
    """Return n x d uniform pseudo-samples from the d-dimensional
    exchangeable Gumbel copula with parameter theta >= 1.
    Marshall-Olkin algorithm: U_i = exp(-(E_i / S)^(1/theta)),
    where S is positive-stable(1/theta) and E_i ~ Exp(1) iid."""
    if theta <= 1.0 + 1e-9:                       # independence limit
        return rng.random((n, d))
    alpha = 1.0 / theta
    S = sample_positive_stable(alpha, n, rng)[:, None]    # (n, 1)
    E = rng.exponential(1.0, size=(n, d))                 # (n, d)
    return np.exp(-np.power(E / S, alpha))


def sample_gaussian_copula(n, d, rho, rng):
    """n x d uniform samples from an equicorrelated Gaussian copula."""
    C = np.full((d, d), rho)
    np.fill_diagonal(C, 1.0)
    L = np.linalg.cholesky(C)
    Z = rng.standard_normal((n, d)) @ L.T
    return norm.cdf(Z)


# --------------------------------------------------------------------------
# Cost model
# --------------------------------------------------------------------------
def project_cost(U, sigma_monthly, horizon, weights):
    """Map copula uniforms U (n x d) to project cost.
    Marginal of the cumulative T-month log-return is N(0, sigma*sqrt(T))
    under the zero-drift assumption; dependence is supplied by the copula.
    Cost: C = C0 * exp( sum_i w_i * r_i^(T) )."""
    scale = sigma_monthly * np.sqrt(horizon)          # (d,)
    R = norm.ppf(U) * scale                            # (n, d) cumulative log-returns
    w = np.array([weights[m] for m in MATERIALS])      # (d,)
    return BASE_COST * np.exp(R @ w)


def p_level(cost, p=P_LEVEL):
    return np.percentile(cost, p)


# --------------------------------------------------------------------------
# Sampler self-validation
# --------------------------------------------------------------------------
def validate_sampler():
    rng = np.random.default_rng(SEED)
    print("Sampler validation (empirical tau vs analytic 1 - 1/theta):")
    for theta in (1.141, 1.503, 6.67):
        U = sample_gumbel_copula(200_000, 2, theta, rng)
        tau_emp = kendalltau(U[:, 0], U[:, 1]).statistic
        tau_ana = 1.0 - 1.0 / theta
        print(f"  theta={theta:5.3f}  tau_emp={tau_emp:.4f}  "
              f"tau_analytic={tau_ana:.4f}  diff={abs(tau_emp-tau_ana):.4f}")
    print()


# --------------------------------------------------------------------------
# Main analysis
# --------------------------------------------------------------------------
def main():
    np.seterr(all="ignore")
    validate_sampler()

    df = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True)
    ret = np.log(df / df.shift(1)).dropna()

    crisis = ret[(ret.index.year >= 2021) & (ret.index.year <= 2024)]
    sigma_crisis = np.array([crisis[m].std() for m in MATERIALS])

    print(f"Crisis-regime monthly volatilities (n = {len(crisis)}):")
    for m, s in zip(MATERIALS, sigma_crisis):
        print(f"  {m:12s}: {s:.5f}")
    print()

    rows = []
    for theta, label in [(THETA_MLE, "Base case (MLE)"),
                         (THETA_STRESS, "Stress scenario")]:
        tau = 1.0 - 1.0 / theta
        lam_u = 2.0 - 2.0 ** (1.0 / theta)
        rho_g = np.sin(np.pi * tau / 2.0)            # Gaussian rho matching tau
        for H in HORIZONS:
            rng = np.random.default_rng(SEED)        # reset for paired comparison
            U_ind = rng.random((N_SIMS, len(MATERIALS)))
            U_gau = sample_gaussian_copula(N_SIMS, len(MATERIALS), rho_g, rng)
            U_gum = sample_gumbel_copula(N_SIMS, len(MATERIALS), theta, rng)

            c_ind = project_cost(U_ind, sigma_crisis, H, WEIGHTS)
            c_gau = project_cost(U_gau, sigma_crisis, H, WEIGHTS)
            c_gum = project_cost(U_gum, sigma_crisis, H, WEIGHTS)

            p_ind, p_gau, p_gum = p_level(c_ind), p_level(c_gau), p_level(c_gum)
            gap_gum = p_gum - p_ind
            gap_gau = p_gau - p_ind

            rows.append({
                "Dependence":   label,
                "theta":        round(theta, 3),
                "tau":          round(tau, 3),
                "lambda_U":     round(lam_u, 3),
                "Horizon_m":    H,
                "Indep_P85":    round(p_ind),
                "Gaussian_P85": round(p_gau),
                "Gumbel_P85":   round(p_gum),
                "Gap_Gumbel_EUR": round(gap_gum),
                "Gap_Gumbel_pct": round(100.0 * gap_gum / p_ind, 2),
                "Gap_Gaussian_EUR": round(gap_gau),
            })

    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "gumbel_gap_mle_vs_stress.csv", index=False)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print("HIDDEN-RISK GAP: BASE CASE (MLE) vs STRESS SCENARIO")
    print("=" * 96)
    print(out.to_string(index=False))
    print()
    print(f"Saved -> {TABLES_DIR / 'gumbel_gap_mle_vs_stress.csv'}")

    # Headline summary for manuscript reframing
    base36 = out[(out.Dependence == "Base case (MLE)") & (out.Horizon_m == 36)].iloc[0]
    str36  = out[(out.Dependence == "Stress scenario") & (out.Horizon_m == 36)].iloc[0]
    print()
    print("HEADLINE RANGE (36-month horizon, P85):")
    print(f"  Base case  (MLE,   tau=0.12, lambda_U=0.17): "
          f"EUR {base36.Gap_Gumbel_EUR:,.0f}  (+{base36.Gap_Gumbel_pct}%)")
    print(f"  Stress     (worst, tau=0.85, lambda_U=0.80): "
          f"EUR {str36.Gap_Gumbel_EUR:,.0f}  (+{str36.Gap_Gumbel_pct}%)")


if __name__ == "__main__":
    main()
