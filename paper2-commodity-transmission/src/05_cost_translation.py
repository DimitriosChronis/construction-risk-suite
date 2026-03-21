"""
05_cost_translation.py
Translates FEVD percentages and IRF cumulative responses into EUR
contingency amounts for a reference construction project.

Uses the same base-cost parameters as Paper 1 for consistency:
  Base cost = EUR 2,300,000
  Weights:  Concrete 30%, Steel 30%, Fuel/Energy 20%, PVC 20%
  Horizon: 24 months

For each material pair:
  1. Computes historical volatility (annualised) of Greek log-returns
  2. Uses FEVD% to attribute share of Greek variance to US shocks
  3. Computes EUR impact = base_cost * weight * sigma * sqrt(horizon) * FEVD_share
  4. Builds a decision-rule table based on US PPI percentile thresholds

Input:  data/processed/aligned_log_returns.csv
        results/tables/c2b_fevd_table.csv
Output: results/tables/c5_cost_translation.csv
        results/tables/c5_decision_rules.csv
        results/figures/fig_c5_eur_contingency.pdf
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "processed",
                          "aligned_log_returns.csv")
FEVD_PATH = os.path.join(SCRIPT_DIR, "..", "results", "tables",
                          "c2b_fevd_table.csv")
OUT_FIG = os.path.join(SCRIPT_DIR, "..", "results", "figures")
OUT_TAB = os.path.join(SCRIPT_DIR, "..", "results", "tables")

# Project parameters (consistent with Paper 1)
BASE_COST = 2_300_000  # EUR
HORIZON_MONTHS = 24
WEIGHTS = {
    "Steel": 0.30,
    "Cement/Concrete": 0.30,
    "Fuel/Energy": 0.20,
    "PVC/Plastic": 0.20,
    "Brent/General": 1.00,  # General index = full project
}

# Map pair labels to Greek column names
GR_COL_MAP = {
    "Steel": "GR_Steel",
    "Cement/Concrete": "GR_Concrete",
    "Fuel/Energy": "GR_Fuel_Energy",
    "PVC/Plastic": "GR_PVC_Pipes",
    "Brent/General": "GR_General_Index",
}

US_COL_MAP = {
    "Steel": "US_Steel_PPI",
    "Cement/Concrete": "US_Cement_PPI",
    "Fuel/Energy": "US_Fuel_PPI",
    "PVC/Plastic": "US_PVC_PPI",
    "Brent/General": "US_Brent",
}

# Granger lead times (from 04_tail_concordance_lag.py / 04b_var_irf.py)
LEAD_MONTHS = {
    "Steel": 4,
    "Cement/Concrete": 1,
    "Fuel/Energy": 0,  # not Granger-significant
    "PVC/Plastic": 0,  # not Granger-significant
    "Brent/General": 1,
}


def main():
    os.makedirs(OUT_FIG, exist_ok=True)
    os.makedirs(OUT_TAB, exist_ok=True)

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    fevd_df = pd.read_csv(FEVD_PATH)

    # Get FEVD at 12 months
    fevd12 = fevd_df[fevd_df["Horizon_months"] == 12].copy()

    print("=" * 70)
    print("COST TRANSLATION: FEVD -> EUR Contingency")
    print(f"Base cost: EUR {BASE_COST:,.0f}  |  Horizon: {HORIZON_MONTHS}M")
    print("=" * 70)

    rows = []
    for _, frow in fevd12.iterrows():
        pair = frow["Pair"]
        fevd_pct = frow["US_explains_%"] / 100.0
        gr_col = GR_COL_MAP.get(pair)
        us_col = US_COL_MAP.get(pair)
        weight = WEIGHTS.get(pair, 0)
        lead = LEAD_MONTHS.get(pair, 0)

        if gr_col is None or gr_col not in df.columns:
            continue

        # Annualised volatility of Greek series
        sigma_monthly = df[gr_col].std()
        sigma_annual = sigma_monthly * np.sqrt(12)

        # Multi-month volatility over project horizon
        sigma_horizon = sigma_monthly * np.sqrt(HORIZON_MONTHS)

        # EUR impact: portion of material cost at risk from US shocks
        # = base_cost * material_weight * horizon_volatility * sqrt(FEVD_share)
        # Using P95 (1.645 sigma) as the contingency level
        if pair == "Brent/General":
            # General index applies to full project
            eur_at_risk = BASE_COST * sigma_horizon * np.sqrt(fevd_pct) * 1.645
        else:
            eur_at_risk = BASE_COST * weight * sigma_horizon * np.sqrt(fevd_pct) * 1.645

        # Also compute: what happens when US PPI has a 1-sigma shock?
        us_sigma = df[us_col].std() if us_col in df.columns else 0
        # Using IRF cumulative to translate US shock -> GR impact
        # EUR from a 1-sigma US shock = base * weight * IRF_cum * us_sigma
        # (simplified: use FEVD-based approach which is more robust)

        rows.append({
            "Pair": pair,
            "Material_weight": f"{weight:.0%}",
            "GR_sigma_monthly_%": round(sigma_monthly * 100, 3),
            "GR_sigma_annual_%": round(sigma_annual * 100, 2),
            "FEVD_12M_%": round(fevd_pct * 100, 1),
            "Lead_months": lead,
            "EUR_contingency_P95": round(eur_at_risk, 0),
            "Pct_of_base_cost": round(eur_at_risk / BASE_COST * 100, 2),
        })

        print(f"\n  {pair}:")
        print(f"    Weight={weight:.0%}, sigma_m={sigma_monthly*100:.3f}%, "
              f"sigma_a={sigma_annual*100:.2f}%")
        print(f"    FEVD={fevd_pct*100:.1f}%, Lead={lead}M")
        print(f"    -> EUR contingency (P95) = EUR {eur_at_risk:,.0f} "
              f"({eur_at_risk/BASE_COST*100:.2f}% of base)")

    result_df = pd.DataFrame(rows)
    result_df.to_csv(os.path.join(OUT_TAB, "c5_cost_translation.csv"),
                     index=False)

    # Total contingency (sum of material-specific, excluding General)
    material_rows = result_df[result_df["Pair"] != "Brent/General"]
    total_material = material_rows["EUR_contingency_P95"].sum()
    general_row = result_df[result_df["Pair"] == "Brent/General"]
    general_eur = general_row["EUR_contingency_P95"].values[0] if len(general_row) > 0 else 0

    print(f"\n{'='*70}")
    print(f"TOTAL material-specific contingency (P95): EUR {total_material:,.0f} "
          f"({total_material/BASE_COST*100:.2f}%)")
    print(f"General Index contingency (P95):           EUR {general_eur:,.0f} "
          f"({general_eur/BASE_COST*100:.2f}%)")
    print(f"{'='*70}")

    # ── Decision Rules Table ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DECISION RULES (US PPI triggers)")
    print("=" * 70)

    decision_rows = []
    for _, frow in fevd12.iterrows():
        pair = frow["Pair"]
        us_col = US_COL_MAP.get(pair)
        lead = LEAD_MONTHS.get(pair, 0)
        if us_col is None or us_col not in df.columns or lead == 0:
            continue

        us_series = df[us_col].dropna()
        p80 = np.percentile(us_series, 80)
        p90 = np.percentile(us_series, 90)

        # Get EUR contingency for this pair
        pair_eur = result_df[result_df["Pair"] == pair]["EUR_contingency_P95"].values
        eur_val = pair_eur[0] if len(pair_eur) > 0 else 0

        decision_rows.append({
            "Signal": f"US {pair.split('/')[0]} PPI",
            "Trigger": f"3M MA > 80th pctile ({p80*100:.2f}%)",
            "Lead_time_months": lead,
            "Action": f"{'Accelerate' if lead >= 3 else 'Lock'} {pair.split('/')[0].lower()} procurement",
            "EUR_contingency": f"EUR {eur_val:,.0f}",
            "Pct_of_base": f"{eur_val/BASE_COST*100:.1f}%",
        })

        print(f"  {pair}: If US 3M MA > {p80*100:.2f}% (80th pctile) "
              f"-> {lead}M lead -> EUR {eur_val:,.0f}")

    dec_df = pd.DataFrame(decision_rows)
    dec_df.to_csv(os.path.join(OUT_TAB, "c5_decision_rules.csv"), index=False)

    # ── Bar chart: EUR contingency by material ───────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    mat_df = result_df[result_df["Pair"] != "Brent/General"].sort_values(
        "EUR_contingency_P95", ascending=True)
    colors = ["#FF9800" if row["Lead_months"] > 0 else "#90A4AE"
              for _, row in mat_df.iterrows()]
    bars = ax.barh(mat_df["Pair"], mat_df["EUR_contingency_P95"],
                   color=colors, edgecolor="black", linewidth=0.5)

    # Add EUR labels on bars
    for bar, val in zip(bars, mat_df["EUR_contingency_P95"]):
        ax.text(bar.get_width() + 500, bar.get_y() + bar.get_height() / 2,
                f"EUR {val:,.0f}", va="center", fontsize=8)

    ax.set_xlabel("P95 Contingency from US Transmission (EUR)")
    ax.set_xlim(0, mat_df["EUR_contingency_P95"].max() * 1.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#FF9800", edgecolor="black", label="Granger-significant"),
        Patch(facecolor="#90A4AE", edgecolor="black", label="Not significant"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG, "fig_c5_eur_contingency.pdf"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("\nfig_c5_eur_contingency.pdf saved.")


if __name__ == "__main__":
    main()
