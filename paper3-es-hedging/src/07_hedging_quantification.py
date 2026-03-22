"""
07_hedging_quantification.py
C5 -- Hedging Quantification with Real Instruments

Computes:
    1. OLS hedge ratios (min-variance) for US vs Greek matched pairs
    2. Hedge cost in EUR (roll cost + bid-ask spread)
    3. Hedged variance reduction
    4. Net benefit: ES reduction - hedge cost

Input:  data/processed/aligned_log_returns.csv
        results/tables/c3_es_comparison.csv
Output: results/tables/c5_hedge_quantification.csv
"""

import pandas as pd
import numpy as np
from scipy.stats import norm, rankdata
import pyvinecopulib as pv
import os

BASE_COST      = 2_300_000
HORIZON_MONTHS = 24
N_SIMS         = 100_000
SEED           = 42

# Full portfolio weights (Concrete, Steel, Fuel, PVC)
PORTFOLIO_WEIGHTS = {"GR_Concrete": 0.30, "GR_Steel": 0.30,
                     "GR_Fuel_Energy": 0.20, "GR_PVC_Pipes": 0.20}

CRISIS_START = "2021-01-01"
CRISIS_END   = "2024-12-01"

# Hedge instruments
HEDGES = [
    {
        "instrument": "Brent Crude Futures (CME BZ)",
        "us_col": "US_Brent",
        "gr_col": "GR_Fuel_Energy",
        "weight": 0.20,    # Fuel share of project
        "roll_cost_annual": 0.003,   # 0.3% annual roll
        "spread_cost": 0.001,        # 0.1% bid-ask
    },
    {
        "instrument": "Steel PPI Swap (OTC proxy)",
        "us_col": "US_Steel_PPI",
        "gr_col": "GR_Steel",
        "weight": 0.30,
        "roll_cost_annual": 0.005,
        "spread_cost": 0.002,
    },
]

SCRIPT_DIR = os.path.dirname(__file__)
DATA_PATH  = os.path.join(SCRIPT_DIR, "..", "data", "processed",
                           "aligned_log_returns.csv")
ES_PATH    = os.path.join(SCRIPT_DIR, "..", "results", "tables",
                           "c3_es_comparison.csv")
OUT_TAB    = os.path.join(SCRIPT_DIR, "..", "results", "tables")


def ols_hedge_ratio(spot, futures):
    """Minimum variance hedge ratio via OLS."""
    valid = pd.concat([spot, futures], axis=1).dropna()
    if len(valid) < 12:
        return 0.0, 0.0
    y = valid.iloc[:, 0].values
    X = valid.iloc[:, 1].values
    X_aug = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
    # R-squared
    y_hat = X_aug @ beta
    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return beta[1], r2


def hedge_cost_eur(notional, hedge_ratio, roll_annual, spread, horizon_m):
    """Total cost of maintaining hedge over horizon."""
    hedged_notional = notional * abs(hedge_ratio)
    years = horizon_m / 12
    roll = hedged_notional * roll_annual * years
    bid_ask = hedged_notional * spread
    return roll + bid_ask


def main():
    os.makedirs(OUT_TAB, exist_ok=True)

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    print(f"Data: {df.shape}")

    # Crisis-period subset for regime-appropriate hedge ratios
    df_crisis = df.loc[CRISIS_START:CRISIS_END].dropna()

    rows = []
    total_hedge_cost = 0

    for h in HEDGES:
        if h["us_col"] not in df.columns or h["gr_col"] not in df.columns:
            print(f"  SKIP: {h['instrument']}")
            continue

        # Full-period hedge ratio (for reference)
        hr_full, r2_full = ols_hedge_ratio(df[h["gr_col"]], df[h["us_col"]])

        # Crisis-period hedge ratio (used for simulation)
        hr, r2 = ols_hedge_ratio(df_crisis[h["gr_col"]],
                                 df_crisis[h["us_col"]])

        notional = BASE_COST * h["weight"]
        cost = hedge_cost_eur(notional, hr, h["roll_cost_annual"],
                              h["spread_cost"], HORIZON_MONTHS)
        var_reduction_pct = r2 * 100
        total_hedge_cost += cost

        rows.append({
            "Instrument":       h["instrument"],
            "Greek_Series":     h["gr_col"],
            "US_Series":        h["us_col"],
            "Hedge_Ratio":      round(hr, 4),
            "Hedge_Ratio_Full": round(hr_full, 4),
            "R_squared":        round(r2, 4),
            "R_squared_Full":   round(r2_full, 4),
            "Var_Reduction_%":  round(var_reduction_pct, 1),
            "Notional_EUR":     round(notional),
            "Hedge_Cost_EUR":   round(cost),
        })
        print(f"  {h['instrument']:35s}  h_crisis={hr:.4f}  "
              f"h_full={hr_full:.4f}  R2={r2:.4f}  cost=EUR {cost:,.0f}")

    # ---- Simulation-based ES comparison (unhedged vs hedged) ---------------
    hedge_ratios = {r["Greek_Series"]: (r["Hedge_Ratio"], r["US_Series"])
                    for r in rows}

    # Use aligned data for joint GR+US simulation
    gr_cols = sorted(PORTFOLIO_WEIGHTS.keys())
    us_cols = sorted(set(r["US_Series"] for r in rows))
    sim_cols = gr_cols + us_cols
    sim_cols = [c for c in sim_cols if c in df.columns]

    crisis = df[sim_cols].loc[CRISIS_START:CRISIS_END].dropna()
    n_obs = len(crisis)
    print(f"\n  Simulation-based ES: crisis n={n_obs}, {len(sim_cols)}D")

    mu    = crisis.mean().values
    sigma = crisis.std().values
    n_assets = len(sim_cols)

    # Fit vine on joint GR+US crisis data
    u_data = np.array([rankdata(crisis.iloc[:, j]) / (n_obs + 1)
                       for j in range(n_assets)]).T
    controls = pv.FitControlsVinecop(
        family_set=[pv.BicopFamily.gumbel, pv.BicopFamily.frank,
                    pv.BicopFamily.gaussian, pv.BicopFamily.student],
        selection_criterion="aic", num_threads=1
    )
    vine = pv.Vinecop(d=n_assets)
    vine.select(np.asfortranarray(u_data), controls)

    u_sim = vine.simulate(N_SIMS * HORIZON_MONTHS, seeds=[SEED])
    u_sim = np.clip(u_sim, 1e-6, 1 - 1e-6)
    u_sim = u_sim.reshape(N_SIMS, HORIZON_MONTHS, n_assets)
    z = norm.ppf(u_sim)
    log_rets = mu + sigma * z  # (N_SIMS, HORIZON, n_assets)

    # Unhedged portfolio: only GR series
    gr_indices = [sim_cols.index(c) for c in gr_cols]
    gr_weights = np.array([PORTFOLIO_WEIGHTS[c] for c in gr_cols])
    gr_log = log_rets[:, :, gr_indices]
    gr_simple = np.exp(gr_log) - 1.0
    port_unhedged = (gr_simple * gr_weights).sum(axis=2)
    cost_unhedged = BASE_COST * np.exp(np.log1p(port_unhedged).sum(axis=1))

    # Hedged portfolio: subtract h * US_return for hedged assets
    gr_hedged_simple = gr_simple.copy()
    for gr_col, (hr, us_col) in hedge_ratios.items():
        if us_col in sim_cols:
            gi = gr_cols.index(gr_col)
            ui = sim_cols.index(us_col)
            us_simple = np.exp(log_rets[:, :, ui]) - 1.0
            gr_hedged_simple[:, :, gi] -= hr * us_simple

    port_hedged = (gr_hedged_simple * gr_weights).sum(axis=2)
    cost_hedged = BASE_COST * np.exp(np.log1p(port_hedged).sum(axis=1))

    # ES(99%)
    def es99(costs):
        q = np.quantile(costs, 0.99)
        return costs[costs >= q].mean()

    es99_unhedged = es99(cost_unhedged)
    es99_hedged   = es99(cost_hedged)
    es99_reduction = es99_unhedged - es99_hedged
    net_benefit = es99_reduction - total_hedge_cost

    print(f"  ES99 unhedged:  EUR {es99_unhedged:>12,.0f}")
    print(f"  ES99 hedged:    EUR {es99_hedged:>12,.0f}")
    print(f"  ES99 reduction: EUR {es99_reduction:>12,.0f}")
    print(f"  Hedge cost:     EUR {total_hedge_cost:>12,.0f}")
    print(f"  Net benefit:    EUR {net_benefit:>12,.0f}")

    rows.append({
        "Instrument":       "TOTAL (simulation-based)",
        "Greek_Series":     "",
        "US_Series":        "",
        "Hedge_Ratio":      None,
        "R_squared":        None,
        "Var_Reduction_%":  None,
        "Notional_EUR":     round(BASE_COST),
        "Hedge_Cost_EUR":   round(total_hedge_cost),
        "ES99_Unhedged":    round(es99_unhedged),
        "ES99_Hedged":      round(es99_hedged),
        "ES99_Reduction":   round(es99_reduction),
        "Net_Benefit_EUR":  round(net_benefit),
    })

    hedge_df = pd.DataFrame(rows)
    hedge_df.to_csv(os.path.join(OUT_TAB, "c5_hedge_quantification.csv"),
                    index=False)
    print(f"\nSaved: c5_hedge_quantification.csv")


if __name__ == "__main__":
    main()
