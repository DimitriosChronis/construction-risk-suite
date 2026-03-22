"""
05e_decision_rules.py
Fix 6: Decision Rules Table for Practitioners

Translates ES results into actionable procurement triggers:
  - What ES threshold triggers action?
  - What action should the project manager take?
  - What is the EUR contingency per phase?

Input:  results/tables/c3_es_comparison.csv
        results/tables/c4_lifecycle_es.csv
        results/tables/c5b_hedge_effectiveness.csv
Output: results/tables/c3e_decision_rules.csv
"""

import os
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(__file__)
OUT_TAB    = os.path.join(SCRIPT_DIR, "..", "results", "tables")
BASE_COST  = 2_300_000


def main():
    os.makedirs(OUT_TAB, exist_ok=True)

    # Load results
    es_df = pd.read_csv(os.path.join(OUT_TAB, "c3_es_comparison.csv"))
    lc_df = pd.read_csv(os.path.join(OUT_TAB, "c4_lifecycle_es.csv"))

    # Crisis Gumbel row
    crisis_gumbel = es_df[(es_df["Regime"].str.contains("Crisis")) &
                           (es_df["Copula"] == "Gumbel")].iloc[0]

    rules = []

    # Rule 1: Overall contingency
    es99_total = crisis_gumbel["ES99_EUR"]
    contingency_total = es99_total - BASE_COST
    rules.append({
        "Rule": "R1",
        "Trigger": "Project inception",
        "Condition": "Crisis regime detected (rolling vol > 67th percentile)",
        "Action": "Set total contingency to ES(99%) - base cost",
        "EUR_amount": round(contingency_total),
        "Pct_of_base": round(contingency_total / BASE_COST * 100, 1),
        "Phase": "All",
    })

    # Rule 2-4: Phase-specific contingency
    for _, row in lc_df.iterrows():
        phase = row["Phase"]
        budget = row["Budget_EUR"]
        es99 = row["ES99_EUR"]
        contingency = es99 - budget
        overrun_pct = row["ES99_Overrun_%"]

        if phase == "Superstructure":
            action = ("Allocate 55% of contingency; pre-purchase steel "
                      "at contract signing; add price escalation clause")
            rule_id = "R2"
        elif phase == "Foundation":
            action = ("Allocate 25% of contingency; lock concrete price "
                      "via supplier framework agreement")
            rule_id = "R3"
        else:
            action = ("Allocate 20% of contingency; monitor fuel index; "
                      "trigger re-tender if fuel exceeds P80")
            rule_id = "R4"

        rules.append({
            "Rule": rule_id,
            "Trigger": f"{phase} phase start (Month {row['Months']})",
            "Condition": f"ES(99%) overrun > {overrun_pct:.0f}% of phase budget",
            "Action": action,
            "EUR_amount": round(contingency),
            "Pct_of_base": round(contingency / BASE_COST * 100, 1),
            "Phase": phase,
        })

    # Rule 5: Hedging decision
    rules.append({
        "Rule": "R5",
        "Trigger": "Steel procurement > EUR 200,000",
        "Condition": "Rolling rho(GR_Steel, US_Steel_PPI) > 0.40",
        "Action": ("Consider Steel PPI swap (OTC); expected HE=25%; "
                   "do NOT hedge other materials (HE<3%)"),
        "EUR_amount": 46504,
        "Pct_of_base": round(46504 / BASE_COST * 100, 1),
        "Phase": "Superstructure",
    })

    # Rule 6: Regime monitoring
    rules.append({
        "Rule": "R6",
        "Trigger": "Monthly monitoring",
        "Condition": "12M rolling composite vol crosses 67th percentile",
        "Action": ("Switch from stable to crisis ES model; "
                   "increase contingency from P85 to ES(99%)"),
        "EUR_amount": round(es99_total - crisis_gumbel["P85_EUR"]),
        "Pct_of_base": round((es99_total - crisis_gumbel["P85_EUR"])
                              / BASE_COST * 100, 1),
        "Phase": "All",
    })

    rules_df = pd.DataFrame(rules)
    rules_df.to_csv(os.path.join(OUT_TAB, "c3e_decision_rules.csv"),
                     index=False)

    print("Decision Rules for Practitioners:")
    print("=" * 80)
    for _, r in rules_df.iterrows():
        print(f"\n{r['Rule']}: {r['Trigger']}")
        print(f"  IF: {r['Condition']}")
        print(f"  THEN: {r['Action']}")
        print(f"  EUR: {r['EUR_amount']:,}  ({r['Pct_of_base']}% of base)")

    print(f"\nSaved: c3e_decision_rules.csv")


if __name__ == "__main__":
    main()
