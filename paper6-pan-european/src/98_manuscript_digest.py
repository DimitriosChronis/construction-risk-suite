"""
98_manuscript_digest.py
=======================
Prints every headline number destined for the manuscript, straight
from the final result tables (single source of truth for writing;
99_verify + the Delta-phase manuscript sweep check against these).
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(SCRIPT_DIR, "..", "data", "processed")
TAB = os.path.join(SCRIPT_DIR, "..", "results", "tables")


def t(name):
    return pd.read_csv(os.path.join(TAB, name))


def main():
    print("=" * 72)
    print("PAPER 6 MANUSCRIPT DIGEST (all values from final tables)")
    print("=" * 72)

    cov = t("../..//data/processed/panel_coverage.csv".replace("//", "/")) \
        if False else pd.read_csv(os.path.join(PROC, "panel_coverage.csv"))
    p = cov[cov["panel"] == "primary"]
    print("\n[PANEL]")
    print(f"  8 countries, obs per country: {p['n_obs'].min()}-"
          f"{p['n_obs'].max()}, span {p['start'].min()} -> {p['end'].max()}")

    print("\n[BRIDGE]")
    bc = t("table_bridge_correlations.csv")
    for _, r in bc.iterrows():
        print(f"  A {r['metric']}: {r['estimate']:+.3f} (p={r['p_value']:.1e}, n={int(r['n'])})")
    bl = t("table_bridge_lagged.csv")
    best = bl.loc[bl["spearman"].idxmax()]
    print(f"  A best lag: {int(best['lag_csri_leads'])}M rho={best['spearman']:.3f}")
    bi = t("table_bridge_indicator_check.csv")
    for _, r in bi.iterrows():
        print(f"  B ELSTAT~EL {r['indicator']}: pearson {r['pearson']:.3f} "
              f"spearman {r['spearman']:.3f} (n={int(r['n_quarters'])}Q)")

    print("\n[H1 CORE]")
    h1 = t("table_h1_summary.csv")
    for _, r in h1.iterrows():
        print(f"  w={int(r['window'])}: pair-mean amp {r['mean_amplification']:.3f}, "
              f"pairs>1 {int(r['pairs_amp_gt_1'])}/28, panel diff {r['panel_diff']:+.4f} "
              f"perm p={r['panel_perm_p']:.3f}, stable {r['mean_lambdaU_stable']:.3f} "
              f"crisis {r['mean_lambdaU_crisis']:.3f}")
    sub = t("table_h1_subgroups.csv")
    for _, r in sub.iterrows():
        print(f"  {int(r['window'])} {r['labels']:5s} {r['bloc']:7s} "
              f"amp={r['amplification']:.3f} p={r['perm_p']:.4f}")
    ep = t("table_h1_episodes.csv")
    for _, r in ep[ep["window"] == 24].iterrows():
        print(f"  ep24 {r['episode']:9s} amp={r['amplification']:.3f} "
              f"peak_pct={r['peak_percentile']:.3f} n={int(r['n_months'])}")
    ci = t("table_h1_bootstrap_ci.csv")
    for _, r in ci.iterrows():
        print(f"  CI w={int(r['window'])} blk={int(r['block'])}: amp {r['amplification']:.3f} "
              f"[{r['ci_lo_95']:.3f},{r['ci_hi_95']:.3f}] excl1={int(r['excludes_1'])} "
              f"sign_p={r['sign_test_p']}")

    print("\n[H3]")
    h3 = t("table_h3_summary.csv").iloc[0]
    print(f"  links {int(h3['n_links_sig'])}/56 ({h3['share_sig']:.0%}), "
          f"MED->N {int(h3['med_to_north_sig'])}, N->MED {int(h3['north_to_med_sig'])}, "
          f"DY total {h3['dy_total_spillover_pct']:.1f}%, lag {int(h3['var_lag'])}, "
          f"top {h3['top_net_transmitter']}")
    g = t("table_granger_network.csv")
    print(f"  BH-FDR sig: {int(g['bh_fdr_sig'].sum())}/56")
    dy = pd.read_csv(os.path.join(TAB, "table_dy_spillover.csv"), index_col=0)
    dy = dy.drop(index="TOTAL_spillover_pct")
    print("  net:", ", ".join(f"{i}:{v:+.1f}" for i, v in
                              dy["net_pct"].astype(float).sort_values(
                                  ascending=False).items()))

    print("\n[H2]")
    h2 = t("table_h2_summary.csv")
    for _, r in h2.iterrows():
        print(f"  {r['linkage']}: agree {r['cut2_matches_mednorth_pct']:.2f} "
              f"sil {r['silhouette_medNorth']:.3f} p={r['perm_p_medNorth']:.3f}")

    print("\n[H4]")
    h4 = t("table_h4_summary.csv").iloc[0]
    wb = t("table_h4_wildboot.csv").iloc[0]
    print(f"  b1={h4['beta1_core_ppsigma']:.4f} b2={h4['beta2_med_extra_ppsigma']:.4f} "
          f"ratio={h4['med_to_core_ratio']:.2f} p_cl={h4['beta2_p_cluster']:.4f} "
          f"p_tFE={h4['beta2_p_timeFE']:.4f} n={int(h4['n_obs'])} "
          f"| wild p={wb['wild_boot_p']:.4f}")
    pi = t("table_panel_interaction.csv")
    m1 = pi[(pi["model"] == "M1_countryFE")].iloc[0]
    print(f"  aggregate b={m1['beta_pp_per_sigma']:.4f} (p={m1['p_value']:.1e})")

    print("\n[UKRAINE]")
    uk = t("table_ukraine_case.csv")
    print("  " + "; ".join(f"{r['country']}:{r['shock_ratio']:.2f}x"
                           f"(+{int(r['months_invasion_to_peak'])}M)"
                           for _, r in uk.iterrows()))
    un = pd.read_csv(os.path.join(TAB, "table_ukraine_network.csv"),
                     index_col=0, parse_dates=True)
    pk = un["panel_mean_lambdaU"].idxmax()
    print(f"  panel lambdaU peak {pk:%Y-%m}: {un['panel_mean_lambdaU'].max():.3f} "
          f"(pctile {un.loc[pk,'full_sample_percentile']:.3f})")

    print("\n[ROBUSTNESS]")
    q1 = t("table_quarterly_h1.csv").iloc[0]
    print(f"  quarterly({int(q1['n_countries'])}): amp {q1['amplification']:.3f} "
          f"p={q1['panel_perm_p']:.3f}")
    q3 = t("table_quarterly_h3.csv")
    dfr = q3[q3["group"] == "DE/FR involved"]
    print(f"  quarterly DE/FR links: "
          f"{(dfr['p_min_lag12'] < 0.05).sum()}/{len(dfr)}")
    ir = t("table_indicator_robustness.csv")
    print(ir.to_string(index=False))
    g3 = t("table_g3_summary.csv")
    print(g3[["regime", "gumbel_best", "gumbel_within2",
              "mean_lambdaU_pooled"]].to_string(index=False))
    cv = t("table_estimator_crossval.csv")
    r_cv = np.corrcoef(cv["theta_mle"], cv["theta_tau"])[0, 1]
    print(f"  estimator crossval r = {r_cv:.3f}")
    lb = t("table_robustness_labels.csv")
    print(f"  label sweep: MED-MED p<0.05 in "
          f"{(lb['MEDMED_p'] < 0.05).sum()}/{len(lb)} configs; "
          f"p<0.06 in {(lb['MEDMED_p'] < 0.06).sum()}/{len(lb)}")
    se = t("table_robustness_seasonal.csv").iloc[0]
    print(f"  deseasonalised: ALL {se['ALL_amp']:.3f} (p={se['ALL_p']:.3f}) "
          f"MED-MED {se['MEDMED_amp']:.3f} (p={se['MEDMED_p']:.3f})")
    gs = t("table_robustness_gr_source.csv")
    print(f"  GR-source (Eurostat EL): out "
          f"{((gs['causing']=='GR') & (gs['p_min_lag12']<0.05)).sum()}/7, "
          f"in {((gs['caused']=='GR') & (gs['p_min_lag12']<0.05)).sum()}/7, "
          f"network {(gs['p_min_lag12']<0.05).sum()}/56")
    dep = t("table_deployment_summary.csv")
    print(f"  deployment: {len(dep)} stages, "
          f"{dep['seconds'].sum()/60:.1f} min, "
          f"all_ok={int((dep['status']=='ok').all())}")


if __name__ == "__main__":
    main()
