"""
Paper 5 — Validation against OpenTender.eu real Greek cost data
=================================================================
Loads EU public procurement data (OpenTender.eu OCDS format) and
validates the P5 theoretical-returns framework against OBSERVED
tender-to-award cost deviations for Greek construction contracts.

Data source:
  OpenTender.eu — OCDS format, Greece
  https://data.open-contracting.org/en/publication/55/
  Multi-CSV archive: main.csv + tender_items.csv + awards.csv
  Linked via main_ocid / main_id

Validation tests:
  1. Crisis vs stable overrun comparison (t-test + Mann-Whitney)
  2. Regression: overrun ~ CSRI + log(budget) + year
  3. Overrun by crisis episode

Inputs:
  - data/raw/opentender_gr_full.csv.tar.gz     (auto-downloaded)
  - data/processed/csri_monthly.csv             (from 07)

Outputs:
  - data/processed/ted_greek_construction.csv
  - data/processed/ted_validation_results.csv
  - results/table_validation_ted.csv
  - results/table_validation_episodes.csv
  - results/figures/fig_validation_ted.{pdf,png}
"""

import io
import os
import sys
import tarfile
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
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

OPENTENDER_GR_URL = (
    "https://data.open-contracting.org/en/publication/55/"
    "download?name=full.csv.tar.gz"
)
TAR_PATH = os.path.join(RAW_DIR, "opentender_gr_full.csv.tar.gz")

# Exogenous crisis windows
EXO_CRISIS_WINDOWS = [
    ("2011-07", "2012-12"),
    ("2015-06", "2015-12"),
    ("2022-02", "2022-12"),
]

MIN_EST_VALUE = 50_000
MAX_OVERRUN   = 5.0
MIN_OVERRUN   = -0.90

print("=" * 64)
print("Paper 5 — Validation: Greek Construction (OpenTender.eu)")
print("=" * 64)


# ══════════════════════════════════════════════════════════════════════════════
# 1. DOWNLOAD ARCHIVE IF NEEDED
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 1: Loading OpenTender.eu Greek data")

if not os.path.exists(TAR_PATH):
    import requests
    print(f"  Downloading from {OPENTENDER_GR_URL}...", flush=True)
    r = requests.get(OPENTENDER_GR_URL, timeout=300, stream=True)
    r.raise_for_status()
    with open(TAR_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
    print(f"  Downloaded: {os.path.getsize(TAR_PATH) / 1e6:.1f} MB")
else:
    print(f"  Using cached: {TAR_PATH} ({os.path.getsize(TAR_PATH)/1e6:.1f} MB)")


# ══════════════════════════════════════════════════════════════════════════════
# 2. EXTRACT & MERGE CSVs FROM ARCHIVE
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 2: Extracting and merging OCDS tables")


def read_from_tar(tar, name, **kwargs):
    """Read a CSV from inside the tar.gz archive."""
    f = tar.extractfile(name)
    if f is None:
        print(f"  Warning: {name} not found in archive")
        return pd.DataFrame()
    return pd.read_csv(f, low_memory=False, **kwargs)


with tarfile.open(TAR_PATH, "r:gz") as tar:
    # 1. Main table — has tender_value_amount (estimated value)
    print("  Loading main.csv...", flush=True)
    main = read_from_tar(tar, "full/main.csv")
    print(f"    Records: {len(main):,}")

    # 2. Tender items — has CPV classification
    print("  Loading tender_items.csv...", flush=True)
    items = read_from_tar(tar, "full/tender_items.csv")
    print(f"    Records: {len(items):,}")

    # 3. Awards — has award value
    print("  Loading awards.csv...", flush=True)
    awards = read_from_tar(tar, "full/awards.csv")
    print(f"    Records: {len(awards):,}")

    # 4. Tender lots — may have lot-level estimated values
    print("  Loading tender_lots.csv...", flush=True)
    lots = read_from_tar(tar, "full/tender_lots.csv")
    print(f"    Records: {len(lots):,}")

# --- Filter construction CPV codes (45*) ---
print("\n  Filtering CPV 45* (construction works)...")
if "classification_id" in items.columns:
    items["cpv_str"] = items["classification_id"].astype(str)
    construction_items = items[items["cpv_str"].str.startswith("45")]
    construction_ocids = set(construction_items["main_ocid"].dropna().unique())
    print(f"    Construction tenders (CPV 45*): {len(construction_ocids):,}")
else:
    print("  ✗ No classification_id column in tender_items")
    sys.exit(1)

# --- Filter main table to construction ---
main_constr = main[main["ocid"].isin(construction_ocids)].copy()
print(f"    Main table (construction): {len(main_constr):,}")

# --- Get estimated value from main or lots ---
main_constr["est_value"] = pd.to_numeric(
    main_constr["tender_value_amount"], errors="coerce")

# If main doesn't have estimated value, try lots
if main_constr["est_value"].notna().sum() < 100 and not lots.empty:
    print("  Trying lot-level estimated values...")
    lot_cols = [c for c in lots.columns if "value" in c.lower() and "amount" in c.lower()]
    if lot_cols:
        lots["lot_value"] = pd.to_numeric(lots[lot_cols[0]], errors="coerce")
        lot_vals = lots.groupby("main_ocid")["lot_value"].sum()
        main_constr["est_value_lot"] = main_constr["ocid"].map(lot_vals)
        # Use lot value where main value is missing
        main_constr["est_value"] = main_constr["est_value"].fillna(
            main_constr["est_value_lot"])

est_count = main_constr["est_value"].notna().sum()
print(f"    With estimated value: {est_count:,}")

# --- Aggregate awards per tender ---
awards["award_amount"] = pd.to_numeric(awards["value_amount"], errors="coerce")
award_agg = awards.groupby("main_ocid")["award_amount"].sum().reset_index()
award_agg.columns = ["ocid", "award_value"]

# --- Merge ---
merged = main_constr.merge(award_agg, on="ocid", how="inner")
print(f"    Merged (main + awards): {len(merged):,}")

# Parse date
merged["award_date"] = pd.to_datetime(merged["date"], errors="coerce", utc=True)
merged["award_date"] = merged["award_date"].dt.tz_localize(None)

# Duration
if "tender_contractPeriod_durationInDays" in merged.columns:
    merged["duration_days"] = pd.to_numeric(
        merged["tender_contractPeriod_durationInDays"], errors="coerce")
    merged["duration_m"] = (merged["duration_days"] / 30.44).round(0)
else:
    merged["duration_m"] = np.nan


# ══════════════════════════════════════════════════════════════════════════════
# 3. COMPUTE OVERRUN + CLEAN
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 3: Computing cost overrun")

gr = merged.dropna(subset=["est_value", "award_value", "award_date"]).copy()
gr = gr[(gr["est_value"] >= MIN_EST_VALUE) & (gr["award_value"] > 0)].copy()

gr["overrun"] = gr["award_value"] / gr["est_value"] - 1.0
gr = gr[(gr["overrun"] >= MIN_OVERRUN) & (gr["overrun"] <= MAX_OVERRUN)].copy()

gr["award_month"] = gr["award_date"].dt.to_period("M").dt.to_timestamp()
gr["award_year"]  = gr["award_date"].dt.year
gr["log_est"]     = np.log(gr["est_value"])

print(f"  Clean Greek construction contracts: {len(gr):,}")
if len(gr) == 0:
    print("  ✗ No records after cleaning. Check data.")
    sys.exit(1)

print(f"  Period: {gr['award_date'].min():%Y-%m} → {gr['award_date'].max():%Y-%m}")
print(f"  Estimated value: EUR {gr['est_value'].min():,.0f} → EUR {gr['est_value'].max():,.0f}")
print(f"  Award value: EUR {gr['award_value'].min():,.0f} → EUR {gr['award_value'].max():,.0f}")
print(f"  Overrun: mean={gr['overrun'].mean():.1%}, "
      f"median={gr['overrun'].median():.1%}, "
      f"std={gr['overrun'].std():.1%}")

# Save
gr[["award_date", "award_month", "award_year", "est_value", "award_value",
    "overrun", "log_est", "duration_m"]].to_csv(
    os.path.join(PROC_DIR, "ted_greek_construction.csv"), index=False)
print(f"  Saved: {PROC_DIR}ted_greek_construction.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 4. ASSIGN CRISIS LABELS + MERGE CSRI
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 4: Assigning crisis labels + merging CSRI")

gr["crisis_exo"] = 0
for start, end in EXO_CRISIS_WINDOWS:
    mask = ((gr["award_date"] >= pd.Timestamp(start)) &
            (gr["award_date"] <= pd.Timestamp(end)))
    gr.loc[mask, "crisis_exo"] = 1

n_crisis = gr["crisis_exo"].sum()
n_stable = len(gr) - n_crisis
print(f"  Crisis contracts: {n_crisis} ({n_crisis/len(gr):.1%})")
print(f"  Stable contracts: {n_stable} ({n_stable/len(gr):.1%})")

csri_path = os.path.join(PROC_DIR, "csri_monthly.csv")
if os.path.exists(csri_path):
    csri = pd.read_csv(csri_path, parse_dates=["date"]).set_index("date")
    gr = gr.merge(csri[["CSRI"]].rename(columns={"CSRI": "csri_at_award"}),
                  left_on="award_month", right_index=True, how="left")
    print(f"  CSRI merged: {gr['csri_at_award'].notna().sum()} matches")
else:
    gr["csri_at_award"] = np.nan
    print("  Warning: csri_monthly.csv not found")


# ══════════════════════════════════════════════════════════════════════════════
# 5. TEST 1: Crisis vs Stable overrun
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 5: Test 1 — Crisis vs Stable overrun")

crisis_or = gr.loc[gr["crisis_exo"] == 1, "overrun"]
stable_or = gr.loc[gr["crisis_exo"] == 0, "overrun"]

results = []

if len(crisis_or) >= 10 and len(stable_or) >= 10:
    t_stat, t_pval = stats.ttest_ind(crisis_or, stable_or, equal_var=False)
    u_stat, u_pval = stats.mannwhitneyu(crisis_or, stable_or,
                                         alternative="greater")

    print(f"  Crisis overrun:  mean={crisis_or.mean():.3f} "
          f"median={crisis_or.median():.3f}  n={len(crisis_or)}")
    print(f"  Stable overrun:  mean={stable_or.mean():.3f} "
          f"median={stable_or.median():.3f}  n={len(stable_or)}")
    print(f"  Difference:      {crisis_or.mean() - stable_or.mean():+.3f}")
    print(f"  Welch t-test:    t={t_stat:.3f}, p={t_pval:.4f}")
    print(f"  Mann-Whitney U:  U={u_stat:.0f}, p={u_pval:.4f} (one-sided)")

    results.append({
        "test": "Crisis vs Stable overrun",
        "crisis_mean": crisis_or.mean(),
        "stable_mean": stable_or.mean(),
        "difference": crisis_or.mean() - stable_or.mean(),
        "t_stat": t_stat, "t_pval": t_pval,
        "u_stat": u_stat, "u_pval": u_pval,
        "n_crisis": len(crisis_or), "n_stable": len(stable_or),
    })
else:
    print(f"  Insufficient data (crisis={len(crisis_or)}, stable={len(stable_or)})")


# ══════════════════════════════════════════════════════════════════════════════
# 6. TEST 2: Regression overrun ~ CSRI + controls
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 6: Test 2 — OLS regression: overrun ~ CSRI + log(budget) + year")

try:
    import statsmodels.api as sm

    reg_data = gr.dropna(subset=["csri_at_award", "overrun", "log_est"]).copy()

    if len(reg_data) >= 30:
        X = reg_data[["csri_at_award", "log_est", "award_year"]].copy()
        X = sm.add_constant(X)
        y = reg_data["overrun"]

        model = sm.OLS(y, X).fit(cov_type="HC1")
        print(model.summary2().tables[1].to_string())

        beta_csri = model.params.get("csri_at_award", np.nan)
        p_csri    = model.pvalues.get("csri_at_award", np.nan)
        r2        = model.rsquared

        results.append({
            "test": "OLS: overrun ~ CSRI + controls",
            "beta_csri": beta_csri,
            "p_csri": p_csri,
            "r_squared": r2,
            "n": len(reg_data),
        })
        print(f"\n  CSRI coefficient: {beta_csri:.4f} (p={p_csri:.4f})")
        print(f"  Interpretation: 1σ CSRI increase → {beta_csri:.1%} higher overrun")
    else:
        print(f"  Insufficient CSRI-matched data: {len(reg_data)}")

except ImportError:
    print("  statsmodels not available — skipping regression")


# ══════════════════════════════════════════════════════════════════════════════
# 7. TEST 3: Overrun by episode
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 7: Test 3 — Overrun by crisis episode")

episodes = {
    "Pre-crisis (2009-2011H1)": ("2009-01", "2011-06"),
    "2012 Sovereign":           ("2011-07", "2012-12"),
    "Recovery (2013-2015H1)":   ("2013-01", "2015-05"),
    "2015 Capital Controls":    ("2015-06", "2015-12"),
    "Expansion (2016-2021)":    ("2016-01", "2021-12"),
    "2022 Ukraine Shock":       ("2022-02", "2022-12"),
    "Post-shock (2023+)":       ("2023-01", "2025-12"),
}

episode_rows = []
for label, (start, end) in episodes.items():
    mask = ((gr["award_date"] >= pd.Timestamp(start)) &
            (gr["award_date"] <= pd.Timestamp(end)))
    sub = gr.loc[mask, "overrun"]
    if len(sub) >= 5:
        episode_rows.append({
            "episode": label,
            "n": len(sub),
            "mean_overrun": sub.mean(),
            "median_overrun": sub.median(),
            "std_overrun": sub.std(),
            "p75_overrun": sub.quantile(0.75),
        })
        print(f"  {label:30s}  n={len(sub):>5}  "
              f"mean={sub.mean():+.3f}  median={sub.median():+.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. FIGURES
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 8: Creating validation figures")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Overrun distribution: crisis vs stable
    ax = axes[0, 0]
    bins = np.linspace(-0.5, 1.5, 50)
    if len(stable_or) > 0:
        ax.hist(stable_or, bins=bins, alpha=0.6,
                label=f"Stable (n={len(stable_or)})",
                color="#2196F3", density=True)
    if len(crisis_or) > 0:
        ax.hist(crisis_or, bins=bins, alpha=0.6,
                label=f"Crisis (n={len(crisis_or)})",
                color="#F44336", density=True)
    if len(stable_or) > 0:
        ax.axvline(stable_or.mean(), color="#1565C0", ls="--", lw=2)
    if len(crisis_or) > 0:
        ax.axvline(crisis_or.mean(), color="#C62828", ls="--", lw=2)
    ax.set_xlabel("Cost overrun (award / estimated - 1)")
    ax.set_ylabel("Density")
    ax.set_title("(a) Overrun distribution: crisis vs stable")
    ax.legend()

    # (b) Monthly median overrun timeline
    ax = axes[0, 1]
    monthly = gr.groupby("award_month")["overrun"].agg(["median", "count"])
    monthly = monthly[monthly["count"] >= 3]
    if len(monthly) > 0:
        ax.plot(monthly.index, monthly["median"], color="#333", lw=1)
        ax.fill_between(monthly.index, monthly["median"],
                        alpha=0.3, color="#2196F3")
        for start, end in EXO_CRISIS_WINDOWS:
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                       alpha=0.2, color="red")
    ax.set_xlabel("Award date")
    ax.set_ylabel("Median overrun")
    ax.set_title("(b) Monthly median overrun + crisis shading")

    # (c) Overrun vs CSRI scatter
    ax = axes[1, 0]
    reg_sub = gr.dropna(subset=["csri_at_award"])
    if len(reg_sub) > 20:
        ax.scatter(reg_sub["csri_at_award"], reg_sub["overrun"],
                   alpha=0.15, s=8, color="#666")
        bins_csri = pd.qcut(reg_sub["csri_at_award"], 10, duplicates="drop")
        binned = reg_sub.groupby(bins_csri, observed=True)["overrun"].mean()
        bin_centers = reg_sub.groupby(bins_csri, observed=True)["csri_at_award"].mean()
        ax.plot(bin_centers, binned, "ro-", lw=2, ms=6, label="Decile means")
        ax.set_xlabel("CSRI at award month")
        ax.set_ylabel("Overrun")
        ax.set_title("(c) Overrun vs CSRI (decile means)")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Insufficient CSRI overlap",
                ha="center", va="center", transform=ax.transAxes)

    # (d) Overrun by episode
    ax = axes[1, 1]
    if episode_rows:
        ep_df = pd.DataFrame(episode_rows)
        colors = ["#F44336" if "2012" in e or "2015" in e or "2022" in e
                  else "#2196F3" for e in ep_df["episode"]]
        ax.barh(range(len(ep_df)), ep_df["mean_overrun"],
                color=colors, alpha=0.8)
        ax.set_yticks(range(len(ep_df)))
        ax.set_yticklabels(ep_df["episode"], fontsize=8)
        ax.set_xlabel("Mean overrun")
        ax.set_title("(d) Mean overrun by episode")
        ax.axvline(0, color="black", lw=0.5)
        for i, row in ep_df.iterrows():
            ax.text(max(row["mean_overrun"] + 0.005, 0.01), i,
                    f"n={row['n']}", va="center", fontsize=7)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig_validation_ted.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_DIR}fig_validation_ted.{{pdf,png}}")

except Exception as e:
    print(f"  Figure error: {e}")
    import traceback; traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
# 9. SAVE
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 9: Saving results")

if results:
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(PROC_DIR, "ted_validation_results.csv"),
                  index=False)
    res_df.to_csv(os.path.join(RES_DIR, "table_validation_ted.csv"),
                  index=False)
    print(f"  Saved: {RES_DIR}table_validation_ted.csv")

if episode_rows:
    pd.DataFrame(episode_rows).to_csv(
        os.path.join(RES_DIR, "table_validation_episodes.csv"), index=False)
    print(f"  Saved: {RES_DIR}table_validation_episodes.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 10. SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("DONE — OpenTender.eu Greek construction validation")
print("=" * 64)
print(f"  Contracts:  {len(gr):,} Greek construction (CPV 45*)")
print(f"  Period:     {gr['award_date'].min():%Y-%m} → {gr['award_date'].max():%Y-%m}")
if results:
    r0 = results[0]
    print(f"  Crisis overrun:  {r0.get('crisis_mean', 0):.1%}")
    print(f"  Stable overrun:  {r0.get('stable_mean', 0):.1%}")
    diff = r0.get("difference", 0)
    pval = r0.get("u_pval", 1)
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."
    print(f"  Difference:      {diff:+.1%} ({sig}, p={pval:.4f})")
if any(r.get("test", "").startswith("OLS") for r in results):
    r_ols = [r for r in results if r["test"].startswith("OLS")][0]
    print(f"  CSRI → overrun:  β={r_ols['beta_csri']:.4f} (p={r_ols['p_csri']:.4f})")
print(f"\nNext: run 15_validation_worldbank.py")
print("=" * 64)
