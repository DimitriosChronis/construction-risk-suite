"""
Paper 5 — Validation against World Bank IEG Project Performance Ratings
=========================================================================
Downloads World Bank Independent Evaluation Group (IEG) project ratings
and tests whether infrastructure projects approved during high
material-price-volatility periods receive worse outcome ratings.

This provides CROSS-COUNTRY validation: the P5 framework is built on
Greek data, but if the same material-price dynamics affect global
infrastructure outcomes, the mechanism is general.

Validation tests:
  1. Outcome rating vs material-price volatility at approval time
  2. Outcome distribution: high-vol vs low-vol approval periods
  3. Sector breakdown: transport / water / energy

Data source:
  World Bank IEG Project Performance Ratings
  https://datacatalog.worldbank.org/search/dataset/0038059
  Columns: Project ID, Project Name, Global Practice, Outcome (1-6 scale),
           Approval FY, Country

  World Bank Projects API (for project amounts):
  https://search.worldbank.org/api/v3/projects

Inputs:
  - data/raw/ieg_ratings.csv                    (downloaded)
  - data/processed/csri_monthly.csv             (from 07, for vol proxy)
  - ../../paper4-lstm-agent/data/processed/features.csv (FRED PPI returns)

Outputs:
  - data/processed/wb_infra_ratings.csv
  - data/processed/wb_validation_results.csv
  - results/table_validation_wb.csv
  - results/figures/fig_validation_wb.{pdf,png}
"""

import io
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
RAW_DIR  = "../data/raw/"
PROC_DIR = "../data/processed/"
RES_DIR  = "../results/"
FIG_DIR  = "../results/figures/"
P4_FEATURES = "../../paper4-lstm-agent/data/processed/features.csv"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# IEG data download — direct CSV from FinancesOne bulk files
IEG_CSV_URL = (
    "https://financesonefiles.worldbank.org/f-one/DS00053/RS00055/"
    "IEG_World_Bank_Project_Performance_Ratings.csv"
)
IEG_ALT_URLS = [
    "https://finances.worldbank.org/api/views/rq9d-pctf/rows.csv"
    "?accessType=DOWNLOAD&api_foundry=true",
]
IEG_MANUAL_URL = (
    "https://datacatalog.worldbank.org/search/dataset/0038059/"
    "ieg-world-bank-project-performance-ratings"
)

# WB Projects API
WB_PROJECTS_API = "https://search.worldbank.org/api/v3/projects"

# Infrastructure-related Global Practices
INFRA_PRACTICES = [
    "Transport",
    "Water",
    "Energy & Extractives",
    "Urban, Resilience and Land",
    "Social, Urban, Rural and Resilience Global Practice",
]

# Outcome rating mapping (IEG uses text labels)
OUTCOME_MAP = {
    "Highly Unsatisfactory": 1,
    "Unsatisfactory":        2,
    "Moderately Unsatisfactory": 3,
    "Moderately Satisfactory":   4,
    "Satisfactory":          5,
    "Highly Satisfactory":   6,
}

print("=" * 64)
print("Paper 5 — Validation: World Bank IEG Infrastructure Ratings")
print("=" * 64)


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD / DOWNLOAD IEG DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 1: Loading IEG project performance ratings")


def download_ieg():
    import requests

    cache = os.path.join(RAW_DIR, "ieg_ratings.csv")
    if os.path.exists(cache):
        print(f"  Using cached: {cache}")
        return pd.read_csv(cache, low_memory=False)

    urls = [IEG_CSV_URL] + IEG_ALT_URLS
    print(f"  Downloading IEG ratings (trying {len(urls)} URLs)...", flush=True)
    for i, url in enumerate(urls, 1):
        try:
            print(f"  [{i}/{len(urls)}] {url[:80]}...", flush=True)
            r = requests.get(url, timeout=120,
                             headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200 and len(r.content) > 1000:
                with open(cache, "wb") as f:
                    f.write(r.content)
                print(f"  ✓ Downloaded: {len(r.content)/1e6:.1f} MB")
                return pd.read_csv(cache, low_memory=False)
            else:
                print(f"    status={r.status_code}, size={len(r.content)}")
        except Exception as e:
            print(f"    failed: {e}")
            continue

    print("  ✗ Auto-download failed.")
    print(f"  Please download manually from:")
    print(f"    {IEG_MANUAL_URL}")
    print(f"  Save as: {cache}")
    return pd.DataFrame()


ieg = download_ieg()

if ieg.empty:
    print("\n  ✗ No IEG data available. Continuing with material-price-only analysis.")
else:
    print(f"  IEG records: {len(ieg):,}")
    print(f"  Columns: {list(ieg.columns)}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. BUILD MATERIAL PRICE VOLATILITY PROXY (from FRED PPI)
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 2: Building global material-price volatility proxy")

if os.path.exists(P4_FEATURES):
    feat = pd.read_csv(P4_FEATURES, parse_dates=["Date"]).set_index("Date").sort_index()
    # Use average of all PPI volatility columns as a global price stress indicator
    vol_cols = [c for c in feat.columns if "vol6" in c.lower() or "vol3" in c.lower()]
    if vol_cols:
        feat["mat_vol"] = feat[vol_cols].mean(axis=1)
        # Annual average for matching with IEG approval FY
        feat["year"] = feat.index.year
        annual_vol = feat.groupby("year")["mat_vol"].mean()
        print(f"  Material-price volatility: {len(annual_vol)} years")
        print(f"  Range: {annual_vol.min():.4f} → {annual_vol.max():.4f}")

        # Define high-vol years (top 25%)
        vol_threshold = annual_vol.quantile(0.75)
        high_vol_years = set(annual_vol[annual_vol >= vol_threshold].index)
        print(f"  High-vol years (P75): {sorted(high_vol_years)}")
    else:
        print("  No volatility columns found in P4 features")
        annual_vol = pd.Series(dtype=float)
        high_vol_years = set()
else:
    print(f"  P4 features not found: {P4_FEATURES}")
    annual_vol = pd.Series(dtype=float)
    high_vol_years = set()


# ══════════════════════════════════════════════════════════════════════════════
# 3. FILTER INFRASTRUCTURE PROJECTS + MERGE VOLATILITY
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 3: Filtering infrastructure projects")

results = []

if not ieg.empty:
    # Detect column names (they vary by download format)
    col_candidates = {
        "practice": ["Global Practice", "GLBL_PRCTC", "global_practice"],
        "outcome":  ["Outcome", "OUTCOME", "outcome"],
        "approval": ["Approval FY", "APRVL_FY", "approval_fy"],
        "country":  ["Country / Economy", "COUNTRY_NAME", "country"],
        "project":  ["Project Name", "PROJ_NAME", "project_name"],
        "proj_id":  ["Project ID", "PROJ_ID", "project_id"],
    }

    col_map = {}
    for key, candidates in col_candidates.items():
        for c in candidates:
            if c in ieg.columns:
                col_map[key] = c
                break
        if key not in col_map:
            # Try case-insensitive match
            for c in ieg.columns:
                if any(cand.lower() == c.lower() for cand in candidates):
                    col_map[key] = c
                    break

    print(f"  Column mapping: {col_map}")

    if "practice" in col_map and "outcome" in col_map:
        # Filter infrastructure
        practice_col = ieg[col_map["practice"]].astype(str)
        infra_mask = practice_col.str.contains(
            "|".join(INFRA_PRACTICES), case=False, na=False)
        infra = ieg[infra_mask].copy()
        print(f"  Infrastructure projects: {len(infra):,}")

        # Map outcome to numeric
        outcome_col = col_map["outcome"]
        infra["outcome_num"] = infra[outcome_col].map(OUTCOME_MAP)
        # Try numeric directly if mapping fails
        if infra["outcome_num"].isna().all():
            infra["outcome_num"] = pd.to_numeric(infra[outcome_col],
                                                  errors="coerce")

        infra = infra.dropna(subset=["outcome_num"])
        print(f"  With valid outcome: {len(infra):,}")

        # Get approval year
        if "approval" in col_map:
            infra["approval_fy"] = pd.to_numeric(
                infra[col_map["approval"]], errors="coerce")
            infra = infra.dropna(subset=["approval_fy"])
            infra["approval_fy"] = infra["approval_fy"].astype(int)
        else:
            infra["approval_fy"] = np.nan

        # Merge volatility
        if len(annual_vol) > 0 and infra["approval_fy"].notna().any():
            infra["mat_vol"] = infra["approval_fy"].map(annual_vol)
            infra["high_vol"] = infra["approval_fy"].isin(high_vol_years).astype(int)

            matched = infra.dropna(subset=["mat_vol"])
            print(f"  Matched with volatility data: {len(matched):,}")

            # Save
            save_cols = ["approval_fy", "outcome_num", "mat_vol", "high_vol"]
            if "country" in col_map:
                save_cols.insert(0, col_map["country"])
            if "practice" in col_map:
                save_cols.insert(0, col_map["practice"])
            matched[save_cols].to_csv(
                os.path.join(PROC_DIR, "wb_infra_ratings.csv"), index=False)

            # ══════════════════════════════════════════════════════════════
            # TEST 1: High-vol vs low-vol outcome ratings
            # ══════════════════════════════════════════════════════════════
            print("\n  --- Test 1: High-vol vs Low-vol outcome ratings ---")
            hv = matched.loc[matched["high_vol"] == 1, "outcome_num"]
            lv = matched.loc[matched["high_vol"] == 0, "outcome_num"]

            if len(hv) >= 10 and len(lv) >= 10:
                t_stat, t_pval = stats.ttest_ind(hv, lv, equal_var=False)
                u_stat, u_pval = stats.mannwhitneyu(lv, hv,
                                                     alternative="greater")

                print(f"    High-vol outcome: mean={hv.mean():.2f}  "
                      f"n={len(hv)}")
                print(f"    Low-vol outcome:  mean={lv.mean():.2f}  "
                      f"n={len(lv)}")
                print(f"    Difference:       {hv.mean() - lv.mean():.3f}")
                print(f"    Welch t-test:     t={t_stat:.3f}, p={t_pval:.4f}")
                print(f"    Mann-Whitney U:   p={u_pval:.4f} (low > high)")

                results.append({
                    "test": "WB: High-vol vs Low-vol outcome",
                    "high_vol_mean": hv.mean(),
                    "low_vol_mean": lv.mean(),
                    "difference": hv.mean() - lv.mean(),
                    "t_stat": t_stat, "t_pval": t_pval,
                    "u_stat": u_stat, "u_pval": u_pval,
                    "n_high": len(hv), "n_low": len(lv),
                })

            # ══════════════════════════════════════════════════════════════
            # TEST 2: Correlation volatility ~ outcome
            # ══════════════════════════════════════════════════════════════
            print("\n  --- Test 2: Correlation mat_vol ~ outcome ---")
            r_corr, r_pval = stats.spearmanr(
                matched["mat_vol"], matched["outcome_num"])
            print(f"    Spearman ρ = {r_corr:.3f}, p = {r_pval:.4f}")

            results.append({
                "test": "WB: Spearman corr(mat_vol, outcome)",
                "spearman_rho": r_corr,
                "p_value": r_pval,
                "n": len(matched),
            })

            # ══════════════════════════════════════════════════════════════
            # TEST 3: By sector
            # ══════════════════════════════════════════════════════════════
            if "practice" in col_map:
                print("\n  --- Test 3: By sector ---")
                for sector in matched[col_map["practice"]].unique():
                    sec = matched[matched[col_map["practice"]] == sector]
                    hv_s = sec.loc[sec["high_vol"] == 1, "outcome_num"]
                    lv_s = sec.loc[sec["high_vol"] == 0, "outcome_num"]
                    if len(hv_s) >= 5 and len(lv_s) >= 5:
                        diff = hv_s.mean() - lv_s.mean()
                        _, p = stats.mannwhitneyu(lv_s, hv_s,
                                                   alternative="greater")
                        sig = "*" if p < 0.05 else ""
                        print(f"    {sector:40s}  diff={diff:+.3f}  "
                              f"p={p:.3f}{sig}  "
                              f"(n_hv={len(hv_s)}, n_lv={len(lv_s)})")
    else:
        print("  Could not find practice/outcome columns")


# ══════════════════════════════════════════════════════════════════════════════
# 4. FIGURES
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 4: Creating validation figures")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (a) Material price volatility timeline
    ax = axes[0]
    if len(annual_vol) > 0:
        colors = ["#F44336" if y in high_vol_years else "#2196F3"
                  for y in annual_vol.index]
        ax.bar(annual_vol.index, annual_vol.values, color=colors, alpha=0.8)
        ax.axhline(vol_threshold, color="red", ls="--", lw=1,
                   label=f"P75 threshold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Mean PPI volatility (6M)")
        ax.set_title("(a) Material-price volatility by year")
        ax.legend()

    # (b) Outcome rating by vol regime
    ax = axes[1]
    if results and any(r["test"].startswith("WB: High-vol") for r in results):
        r0 = [r for r in results if r["test"].startswith("WB: High-vol")][0]
        bars = ax.bar(["Low-vol years", "High-vol years"],
                      [r0["low_vol_mean"], r0["high_vol_mean"]],
                      color=["#2196F3", "#F44336"], alpha=0.8)
        ax.set_ylabel("Mean IEG outcome rating (1-6)")
        ax.set_title("(b) WB infra outcome by vol regime")
        for bar, val in zip(bars, [r0["low_vol_mean"], r0["high_vol_mean"]]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", fontsize=10)
        pval = r0.get("u_pval", 1)
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."
        ax.set_title(f"(b) WB infra outcome ({sig}, p={pval:.3f})")

    # (c) Scatter: volatility vs outcome (annual means)
    ax = axes[2]
    if not ieg.empty and "approval_fy" in locals().get("matched", pd.DataFrame()).columns:
        annual_outcome = matched.groupby("approval_fy")["outcome_num"].mean()
        common_years = annual_vol.index.intersection(annual_outcome.index)
        if len(common_years) > 5:
            ax.scatter(annual_vol.loc[common_years],
                       annual_outcome.loc[common_years],
                       s=40, alpha=0.7, color="#333")
            # Label years
            for y in common_years:
                ax.annotate(str(y),
                           (annual_vol.loc[y], annual_outcome.loc[y]),
                           fontsize=7, alpha=0.7)
            ax.set_xlabel("Material-price volatility")
            ax.set_ylabel("Mean IEG outcome rating")
            ax.set_title("(c) Annual vol vs outcome")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig_validation_wb.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_DIR}fig_validation_wb.{{pdf,png}}")

except Exception as e:
    print(f"  Figure error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 5: Saving results")

if results:
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(PROC_DIR, "wb_validation_results.csv"),
                  index=False)
    res_df.to_csv(os.path.join(RES_DIR, "table_validation_wb.csv"),
                  index=False)
    print(f"  Saved: {RES_DIR}table_validation_wb.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 6. SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("DONE — World Bank IEG infrastructure validation")
print("=" * 64)

if results:
    for r in results:
        test = r["test"]
        if "High-vol" in test:
            diff = r["difference"]
            pval = r.get("u_pval", r.get("p_value", 1))
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."
            print(f"  {test}: diff={diff:+.3f} ({sig})")
        elif "Spearman" in test:
            print(f"  {test}: rho={r['spearman_rho']:.3f}, p={r['p_value']:.4f}")
else:
    print("  No validation tests completed (data unavailable)")

print(f"\nPipeline validation complete.")
print("=" * 64)
