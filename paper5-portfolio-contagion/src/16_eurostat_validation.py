"""
Paper 5 -- B2a: Cross-country construction-cost validation via FRED
====================================================================
Strengthens the external validation by extending the World Bank IEG
analysis with a focused panel of 17 EU/OECD countries' construction
cost indices, downloaded from FRED.

Hypothesis: country-year combinations with high material-price
volatility (top 25% by year) experience higher construction-cost
inflation than low-volatility country-years.

Data:
  - Construction cost indices: OECD PRCNTO01 series via FRED
    (Germany, France, Italy, Spain, Netherlands, Belgium, Austria,
    Portugal, Finland, Sweden, Denmark, Ireland, Luxembourg,
    UK, Norway, Poland, Czech Republic) -- 17 countries
  - Material-price volatility proxy: FRED PPI 6-month rolling
    standard deviation of log-returns (Steel + Cement + Fuel)

Outputs:
  data/processed/eurostat_panel.csv
  results/table_eurostat_validation.csv
  results/figures/fig_eurostat_validation.{pdf,png}
"""

import os
import io
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

RAW_DIR  = "../data/raw/"
PROC_DIR = "../data/processed/"
RES_DIR  = "../results/"
FIG_DIR  = "../results/figures/"
P4_FEAT  = "../../paper4-lstm-agent/data/processed/features.csv"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# 17 OECD countries with PRCNTO01 construction-cost series
COUNTRIES = {
    "DE": "DEUPRCNTO01IXOBM",   # Germany
    "FR": "FRAPRCNTO01IXOBM",   # France
    "IT": "ITAPRCNTO01IXOBM",   # Italy
    "ES": "ESPPRCNTO01IXOBM",   # Spain
    "NL": "NLDPRCNTO01IXOBM",   # Netherlands
    "BE": "BELPRCNTO01IXOBM",   # Belgium
    "AT": "AUTPRCNTO01IXOBM",   # Austria
    "PT": "PRTPRCNTO01IXOBM",   # Portugal
    "FI": "FINPRCNTO01IXOBM",   # Finland
    "SE": "SWEPRCNTO01IXOBM",   # Sweden
    "DK": "DNKPRCNTO01IXOBM",   # Denmark
    "IE": "IRLPRCNTO01IXOBM",   # Ireland
    "LU": "LUXPRCNTO01IXOBM",   # Luxembourg
    "GB": "GBRPRCNTO01IXOBM",   # United Kingdom
    "NO": "NORPRCNTO01IXOBM",   # Norway
    "PL": "POLPRCNTO01IXOBM",   # Poland
    "CZ": "CZEPRCNTO01IXOBM",   # Czech Republic
}

print("=" * 64)
print("Paper 5 -- B2a: Eurostat-style cross-country validation")
print("=" * 64)


# ---------------------------------------------------------------------------
# 1. DOWNLOAD CONSTRUCTION COST INDICES VIA FRED
# ---------------------------------------------------------------------------
def download_fred_series(series_id):
    import requests
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return None
        df = pd.read_csv(io.StringIO(r.text))
        date_col = df.columns[0]
        val_col  = df.columns[1]
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).rename(columns={val_col: series_id})
        df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
        return df.dropna()
    except Exception as e:
        print(f"  [warn] {series_id}: {e}")
        return None


print("\nSTEP 1: Downloading construction cost indices ...")
panel_path = os.path.join(PROC_DIR, "eurostat_panel.csv")
if os.path.exists(panel_path):
    panel = pd.read_csv(panel_path, parse_dates=["date"])
    print(f"  cached panel: {panel.shape}")
else:
    frames = []
    for code, sid in COUNTRIES.items():
        cache = os.path.join(RAW_DIR, f"{code}_{sid}.csv")
        if os.path.exists(cache):
            df = pd.read_csv(cache, parse_dates=["date"], index_col="date")
        else:
            df = download_fred_series(sid)
            if df is not None:
                df.index.name = "date"
                df.to_csv(cache)
        if df is None or df.empty:
            print(f"  [skip] {code}: no data")
            continue
        df = df.rename(columns={sid: "index"})
        df["country"] = code
        df["log_return"] = np.log(df["index"] / df["index"].shift(1))
        df = df.reset_index()
        frames.append(df)
        print(f"  {code}: {len(df)} months")

    panel = pd.concat(frames, ignore_index=True)
    panel.to_csv(panel_path, index=False)

print(f"\n  Panel: {len(panel):,} country-month obs, "
      f"{panel['country'].nunique()} countries")


# ---------------------------------------------------------------------------
# 2. BUILD MATERIAL VOLATILITY PROXY FROM PAPER 4 FEATURES
# ---------------------------------------------------------------------------
print("\nSTEP 2: Building material-volatility proxy (P4 PPI features)")

if not os.path.exists(P4_FEAT):
    print(f"  [error] P4 features not found at {P4_FEAT}")
    sys.exit(1)

feat = pd.read_csv(P4_FEAT, parse_dates=["Date"]).set_index("Date").sort_index()
vol_cols = [c for c in feat.columns if "vol6" in c.lower()]
feat["mat_vol"] = feat[vol_cols].mean(axis=1)
feat["year"]   = feat.index.year
annual_vol = feat.groupby("year")["mat_vol"].mean()

q75 = float(annual_vol.quantile(0.75))
high_years = set(annual_vol[annual_vol >= q75].index)
print(f"  high-vol years (P75): {sorted(high_years)}")


# ---------------------------------------------------------------------------
# 3. ANNUAL PANEL: Δlog construction index by country-year
# ---------------------------------------------------------------------------
print("\nSTEP 3: Annual log-returns by country-year")

panel["year"] = panel["date"].dt.year
ann = (panel.groupby(["country", "year"])["log_return"]
              .sum()
              .reset_index()
              .rename(columns={"log_return": "annual_log_return"}))
ann["mat_vol"]  = ann["year"].map(annual_vol)
ann["high_vol"] = ann["year"].isin(high_years).astype(int)
ann = ann.dropna(subset=["mat_vol"])
print(f"  Annual panel: {len(ann)} country-years, "
      f"{ann['country'].nunique()} countries, "
      f"{ann['year'].nunique()} years")


# ---------------------------------------------------------------------------
# 4. STATISTICAL TESTS
# ---------------------------------------------------------------------------
print("\nSTEP 4: Statistical tests")
results = []

hi = ann.loc[ann["high_vol"] == 1, "annual_log_return"]
lo = ann.loc[ann["high_vol"] == 0, "annual_log_return"]
t_stat, t_pval = stats.ttest_ind(hi, lo, equal_var=False)
u_stat, u_pval = stats.mannwhitneyu(hi, lo, alternative="greater")
print(f"  Test 1: high-vol vs low-vol annual returns")
print(f"    n_high={len(hi)}, n_low={len(lo)}")
print(f"    mean_high={hi.mean():.4f}, mean_low={lo.mean():.4f}")
print(f"    diff={hi.mean()-lo.mean():.4f}, t={t_stat:.3f}, p={t_pval:.4f}")
print(f"    Mann-Whitney p={u_pval:.4f} (high > low)")

results.append({
    "test": "EU17: high-vol vs low-vol annual return",
    "n_high": len(hi), "n_low": len(lo),
    "mean_high": float(hi.mean()), "mean_low": float(lo.mean()),
    "diff": float(hi.mean() - lo.mean()),
    "t_stat": float(t_stat), "t_pval": float(t_pval),
    "u_pval": float(u_pval),
})

rho, sp_pval = stats.spearmanr(ann["mat_vol"], ann["annual_log_return"])
print(f"\n  Test 2: Spearman correlation (mat_vol, annual_return)")
print(f"    rho={rho:.4f}, p={sp_pval:.4f}, n={len(ann)}")

results.append({
    "test": "EU17: Spearman corr(mat_vol, annual_return)",
    "n": len(ann), "spearman_rho": float(rho), "p_value": float(sp_pval),
})

print("\n  Test 3: Per-country effect size (high - low mean)")
country_eff = []
for c in sorted(ann["country"].unique()):
    sub = ann[ann["country"] == c]
    if (sub["high_vol"] == 1).sum() < 2 or (sub["high_vol"] == 0).sum() < 2:
        continue
    diff = (sub.loc[sub["high_vol"] == 1, "annual_log_return"].mean()
            - sub.loc[sub["high_vol"] == 0, "annual_log_return"].mean())
    country_eff.append({"country": c, "high_minus_low": float(diff)})
country_eff_df = pd.DataFrame(country_eff)
print(country_eff_df)
n_pos = (country_eff_df["high_minus_low"] > 0).sum()
n_tot = len(country_eff_df)
print(f"  countries with high>low: {n_pos}/{n_tot}")

results.append({
    "test": "EU17: countries with high>low",
    "n_total":   n_tot, "n_positive": int(n_pos),
    "share_pos": float(n_pos / n_tot) if n_tot else np.nan,
})


# ---------------------------------------------------------------------------
# 5. SAVE RESULTS
# ---------------------------------------------------------------------------
res_df = pd.DataFrame(results)
res_df.to_csv(os.path.join(RES_DIR, "table_eurostat_validation.csv"),
              index=False)
country_eff_df.to_csv(os.path.join(RES_DIR,
                                    "table_eurostat_per_country.csv"),
                     index=False)

# ---------------------------------------------------------------------------
# 6. FIGURE
# ---------------------------------------------------------------------------
print("\nSTEP 5: Figure")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# (a) annual_log_return by year, high-vol shaded
ax = axes[0]
yearly = ann.groupby("year").agg(
    mean_ret=("annual_log_return", "mean"),
    high_vol=("high_vol",          "first"),
).reset_index()
colors = ["#F44336" if h else "#2196F3" for h in yearly["high_vol"]]
ax.bar(yearly["year"], yearly["mean_ret"] * 100,
       color=colors, alpha=0.85, edgecolor="black", linewidth=0.4)
ax.set_xlabel("Year")
ax.set_ylabel("Mean annual log-return (%)")
ax.set_title("(a) EU17 mean construction-cost log-return\n"
             "(red = high-vol year, blue = low-vol)")
ax.grid(alpha=0.3, axis="y")

# (b) Distribution
ax = axes[1]
ax.hist(lo * 100, bins=20, color="#2196F3", alpha=0.55,
        label=f"Low-vol years (n={len(lo)})", edgecolor="black")
ax.hist(hi * 100, bins=20, color="#F44336", alpha=0.55,
        label=f"High-vol years (n={len(hi)})", edgecolor="black")
ax.axvline(lo.mean() * 100, color="#1976D2", ls="--", lw=2)
ax.axvline(hi.mean() * 100, color="#C62828", ls="--", lw=2)
ax.set_xlabel("Annual log-return (%)")
ax.set_ylabel("Frequency")
ax.set_title(f"(b) Distribution by regime\n"
             f"diff = {(hi.mean()-lo.mean())*100:.2f}%, p = {u_pval:.3f}")
ax.legend()
ax.grid(alpha=0.3)

# (c) Per-country effect
ax = axes[2]
country_eff_df = country_eff_df.sort_values("high_minus_low")
bar_colors = ["#F44336" if v > 0 else "#1976D2"
              for v in country_eff_df["high_minus_low"]]
ax.barh(range(len(country_eff_df)),
        country_eff_df["high_minus_low"] * 100,
        color=bar_colors, edgecolor="black", linewidth=0.4)
ax.set_yticks(range(len(country_eff_df)))
ax.set_yticklabels(country_eff_df["country"])
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("Mean (high-vol) - mean (low-vol), %")
ax.set_title(f"(c) Per-country effect\n"
             f"{n_pos}/{n_tot} countries: high > low")
ax.grid(alpha=0.3, axis="x")

plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG_DIR, f"fig_eurostat_validation.{ext}"),
                dpi=300, bbox_inches="tight")
plt.close()
print(f"  saved: fig_eurostat_validation.{{pdf,png}}")

print("\n" + "=" * 64)
print("DONE -- 16_eurostat_validation.py")
print("=" * 64)
