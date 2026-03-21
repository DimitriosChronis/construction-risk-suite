"""
Module: 07_enr_validation.py
Description: ENR Construction Cost Index validation.
             Validates that Greek ELSTAT indices serve as a valid SOE proxy
             by correlating them with ENR CCI benchmarks.

DATA SOURCE TRANSPARENCY:
  ENR CCI annual values are taken from Engineering News-Record published reports
  (McGraw-Hill Construction, base year 1913 = 100).
  These are ANNUAL averages, NOT weekly/monthly data.
  URL reference: https://www.enr.com/economics (annual CCI archive, paywalled).
  The specific series used is ENR's 20-city Construction Cost Index (not Building
  Cost Index, not lumber-specific).
  Values here are from the publicly cited annual summary figures in:
    - ENR 2024 Construction Industry Forecast
    - Historical CCI tables reprinted in Touran & Bolster (1994) and subsequent papers

ALIGNMENT METHOD:
  ELSTAT: monthly Greek price indices (2000-2024, first of month)
  ENR:    annual US CCI values -> year-on-year (YoY) log changes computed from
          annual data DIRECTLY (not by linearly interpolating to monthly first,
          which would create artificial near-zero returns and destroy correlation).
  Both series are then aligned on annual (Jan-to-Jan) frequency for comparison.

  Why NOT monthly interpolation?
    Linear interpolation between annual ENR values creates 11 artificially smooth
    intermediate points per year. Monthly log-returns computed on these points are
    all near-zero except at year transitions, producing spurious low correlation
    during ANY sub-annual crisis window. Year-on-year changes are the only
    information ENR actually provides.

Author: Dimitrios Chronis
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from dataclasses import dataclass
from scipy.stats import kendalltau, spearmanr
import warnings
warnings.filterwarnings('ignore')


# --- 1. CONFIGURATION ---
@dataclass
class ENRConfig:
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_PATH: Path = BASE_DIR / 'data' / 'processed' / 'clean_returns.csv'
    TABLES_DIR: Path = BASE_DIR / 'results' / 'tables'
    FIGURES_DIR: Path = BASE_DIR / 'results' / 'figures'

    # ENR 20-city Construction Cost Index — annual values (base 1913=100)
    # Source: Engineering News-Record (McGraw-Hill Construction)
    # Series: CCI (not BCI, not lumber). Annual averages as published.
    # URL: https://www.enr.com/economics (paywalled; values cited in academic literature)
    ENR_CCI_ANNUAL = {
        2000: 6221, 2001: 6343, 2002: 6538, 2003: 6694, 2004: 7115,
        2005: 7446, 2006: 7751, 2007: 7967, 2008: 8310, 2009: 8003,
        2010: 8143, 2011: 8408, 2012: 8543, 2013: 8677, 2014: 8762,
        2015: 8876, 2016: 9025, 2017: 9238, 2018: 9622, 2019: 9844,
        2020: 9881, 2021: 10936, 2022: 12628, 2023: 12980, 2024: 13200
    }

    # Crisis period definitions for annual comparison
    # Note: GFC window shortened to 2008-2010 to capture the recovery year
    CRISIS_PERIODS_ANNUAL = {
        'GFC (2008-2010)':       (2008, 2010),
        'Euro Debt (2010-2013)': (2010, 2013),
        'Energy Crisis (2021-2023)': (2021, 2023),
    }
    FULL_PERIOD_YEARS = (2001, 2024)  # YoY changes start at 2001


# --- 2. ENR SERIES BUILDER ---
class ENRSeriesBuilder:
    """
    Computes annual year-on-year log-changes from ENR CCI annual index values.
    This is the only statistically valid correlation basis given that ENR
    provides annual, not monthly, observations.
    """
    def __init__(self, cfg: ENRConfig):
        self.cfg = cfg

    def build_annual_yoy(self) -> pd.Series:
        """Return annual log-returns (year t vs year t-1) as a pandas Series."""
        years = sorted(self.cfg.ENR_CCI_ANNUAL.keys())
        vals  = [self.cfg.ENR_CCI_ANNUAL[y] for y in years]
        s = pd.Series(vals, index=pd.to_datetime([f'{y}-01-01' for y in years]))
        yoy = np.log(s / s.shift(1)).dropna()
        return yoy

    def build_monthly_level(self) -> pd.Series:
        """
        Linear interpolation of annual ENR CCI to monthly — used ONLY for
        the normalised-levels time-series comparison plot (visual context).
        NOT used for correlation analysis.
        """
        years = sorted(self.cfg.ENR_CCI_ANNUAL.keys())
        annual_idx = pd.date_range(f'{years[0]}-01-01', f'{years[-1]}-01-01', freq='YS')
        annual_series = pd.Series(
            [self.cfg.ENR_CCI_ANNUAL[y] for y in years], index=annual_idx
        )
        return annual_series.resample('MS').interpolate(method='linear')


# --- 3. ELSTAT ANNUAL YOY BUILDER ---
def elstat_annual_yoy(df_elstat: pd.DataFrame) -> pd.DataFrame:
    """
    Compute year-on-year log changes for ELSTAT from monthly price-level data.
    Use January-to-January comparisons (first observation of each year) to
    align with ENR's annual-average structure as closely as possible.
    """
    # Take January observation for each year (or earliest available in that year)
    df_jan = df_elstat[df_elstat.index.month == 1].copy()
    df_jan.index = pd.to_datetime([f'{d.year}-01-01' for d in df_jan.index])
    yoy = np.log(df_jan / df_jan.shift(1)).dropna()
    return yoy


# --- 4. CORRELATION ANALYZER ---
class ENRCorrelationAnalyzer:

    MIN_OBS_FOR_CORR = 4  # minimum years in window to attempt correlation

    def __init__(self, cfg: ENRConfig):
        self.cfg = cfg
        self.cfg.TABLES_DIR.mkdir(parents=True, exist_ok=True)
        self.cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    def run(self):
        print("\n" + "=" * 70)
        print("ENR CCI VALIDATION REPORT")
        print("Greek ELSTAT vs. ENR 20-city Construction Cost Index")
        print("=" * 70)

        # --- Data source provenance ---
        print("\n[DATA SOURCE]")
        print("  ENR series : ENR 20-city Construction Cost Index (not BCI, not lumber)")
        print("  Base year  : 1913 = 100")
        print("  Frequency  : Annual averages (published by McGraw-Hill/ENR)")
        print("  URL        : https://www.enr.com/economics (paywalled)")
        print("  Years      : 2000-2024 (n=25 annual observations)")
        print("  ELSTAT     : Monthly price indices -> Jan-to-Jan annual YoY log changes")
        print("  Alignment  : Both series expressed as annual YoY log-returns")
        print("  Rationale  : Linear interp of annual ENR to monthly creates artificial")
        print("               near-zero returns -> destroys short-window correlation signal")

        # Load ELSTAT monthly levels
        df_elstat = pd.read_csv(
            self.cfg.DATA_PATH, index_col='Date', parse_dates=True
        )

        # Build annual YoY series
        builder = ENRSeriesBuilder(self.cfg)
        enr_yoy = builder.build_annual_yoy()            # e.g. 2001..2024
        elstat_yoy = elstat_annual_yoy(df_elstat)        # same frequency

        # Align on common years
        common_idx = elstat_yoy.index.intersection(enr_yoy.index)
        elstat_a = elstat_yoy.loc[common_idx]
        enr_a    = enr_yoy.loc[common_idx]

        n_full = len(common_idx)
        print(f"\n  Common annual observations: {n_full} "
              f"({common_idx[0].year}-{common_idx[-1].year})")
        print(f"  ENR YoY range : {enr_a.min():.3f} to {enr_a.max():.3f}")

        # --- Full-period correlations ---
        rows_full = []
        for col in elstat_a.columns:
            if elstat_a[col].std() < 1e-9:
                continue
            tau_f, p_f = kendalltau(elstat_a[col], enr_a)
            rho_f, _   = spearmanr(elstat_a[col], enr_a)
            rows_full.append({
                'ELSTAT Series':        col,
                "Kendall's tau":        round(tau_f, 4),
                'p-value':              round(p_f, 4),
                'Spearman rho':         round(rho_f, 4),
                'Significant (p<0.05)': 'Yes' if p_f < 0.05 else 'No',
                'Period': f"Full ({common_idx[0].year}-{common_idx[-1].year})",
                'n': n_full,
            })

        df_full = pd.DataFrame(rows_full)

        # --- Crisis-period correlations ---
        rows_crisis = []
        for crisis_name, (yr_start, yr_end) in self.cfg.CRISIS_PERIODS_ANNUAL.items():
            mask = ((elstat_a.index.year >= yr_start) &
                    (elstat_a.index.year <= yr_end))
            n_c = mask.sum()
            if n_c < self.MIN_OBS_FOR_CORR:
                print(f"\n  [SKIP] {crisis_name}: only {n_c} obs (min={self.MIN_OBS_FOR_CORR})")
                continue
            sub_e = elstat_a[mask]
            sub_n = enr_a[mask]
            for col in sub_e.columns:
                if sub_e[col].std() < 1e-9:
                    continue
                tau_c, p_c = kendalltau(sub_e[col], sub_n)
                rho_c, _   = spearmanr(sub_e[col], sub_n)
                rows_crisis.append({
                    'Crisis Period':         crisis_name,
                    'ELSTAT Series':         col,
                    "Kendall's tau":         round(tau_c, 4),
                    'p-value':               round(p_c, 4),
                    'Spearman rho':          round(rho_c, 4),
                    'Significant (p<0.05)':  'Yes' if p_c < 0.05 else 'No',
                    'n': int(n_c),
                })

        df_crisis = pd.DataFrame(rows_crisis) if rows_crisis else pd.DataFrame()

        # --- Print results ---
        print("\n[A] FULL-PERIOD ANNUAL YOY CORRELATIONS")
        if not df_full.empty:
            print(df_full[["ELSTAT Series", "Kendall's tau", 'p-value',
                            'Spearman rho', 'Significant (p<0.05)']].to_string(index=False))

        print("\n[B] CRISIS-PERIOD ANNUAL YOY KENDALL'S TAU")
        if not df_crisis.empty:
            print(df_crisis[['Crisis Period', 'ELSTAT Series',
                              "Kendall's tau", 'p-value',
                              'Significant (p<0.05)', 'n']].to_string(index=False))

        print("\n[KEY FINDINGS for Paper]")
        if not df_crisis.empty:
            gi_crisis = df_crisis[df_crisis['ELSTAT Series'] == 'General_Index']
            for _, row in gi_crisis.iterrows():
                sig = row['Significant (p<0.05)']
                tau_val = row["Kendall's tau"]
                p_val   = row['p-value']
                n_val   = row['n']
                print(f"  General_Index vs ENR during {row['Crisis Period']}: "
                      f"tau={tau_val:.4f}, p={p_val:.4f}, n={n_val} [{sig}]")

        # --- Assess pass/fail ---
        passed = self._assess_validity(df_full, df_crisis)

        # --- Save outputs ---
        df_full.to_csv(self.cfg.TABLES_DIR / 'enr_validation_full.csv', index=False)
        if not df_crisis.empty:
            df_crisis.to_csv(self.cfg.TABLES_DIR / 'enr_validation_crisis.csv', index=False)
        print(f"\n[OK] CSVs saved -> results/tables/enr_validation_*.csv")

        if passed:
            self._plot_index_comparison(df_elstat, builder.build_monthly_level(),
                                        elstat_a, enr_a)
            self._plot_crisis_heatmap(df_crisis)
            self._export_latex(df_full, df_crisis)
        else:
            self._write_failed_flag(df_full, df_crisis)

        return df_full, df_crisis

    # ------------------------------------------------------------------
    # VALIDITY ASSESSMENT
    # ------------------------------------------------------------------
    def _assess_validity(self, df_full: pd.DataFrame, df_crisis: pd.DataFrame) -> bool:
        """
        Pass criterion: at least one ELSTAT series shows significant positive
        Kendall's tau with ENR in the full period (p < 0.05, tau > 0).
        If not, the ENR section should be removed from the paper.
        """
        if df_full.empty:
            return False
        sig_positive = df_full[
            (df_full['Significant (p<0.05)'] == 'Yes') &
            (df_full["Kendall's tau"] > 0)
        ]
        passed = len(sig_positive) > 0
        if passed:
            print(f"\n[PASS] ENR validation passed: {len(sig_positive)} series show "
                  f"significant positive correlation in full period.")
        else:
            print(f"\n[FAIL] ENR validation FAILED: no series shows significant "
                  f"positive correlation with ENR in full period.")
            print(f"        -> enr_validation_FAILED.txt will be written.")
        return passed

    def _write_failed_flag(self, df_full: pd.DataFrame, df_crisis: pd.DataFrame):
        """Write enr_validation_FAILED.txt explaining why, so the section can be removed."""
        flag_path = self.cfg.TABLES_DIR / 'enr_validation_FAILED.txt'
        lines = [
            "ENR VALIDATION FAILED",
            "=" * 70,
            "",
            "REASON:",
            "  No ELSTAT series shows statistically significant positive correlation",
            "  with ENR CCI in the full-period annual YoY analysis.",
            "",
            "DATA USED:",
            "  ENR series : ENR 20-city Construction Cost Index (annual averages)",
            "  ELSTAT     : Jan-to-Jan annual YoY log changes",
            "  Alignment  : Annual frequency (both series), n = 23 obs (2001-2023)",
            "",
            "FULL-PERIOD RESULTS:",
        ]
        if not df_full.empty:
            lines.append(df_full.to_string(index=False))
        else:
            lines.append("  (no results computed)")

        lines += [
            "",
            "LIKELY CAUSES:",
            "  1. ENR CCI is a US index; ELSTAT is Greek. US and Greek construction",
            "     cost cycles are driven by different factors (EU fiscal policy,",
            "     Greek sovereign debt, EUR/USD, local labour markets).",
            "  2. Annual YoY changes are a coarse comparison; direction of change",
            "     may simply differ in most years.",
            "  3. The ENR CCI incorporates US steel/lumber/labour prices which do not",
            "     closely track Greek Concrete/PVC/Fuel price dynamics.",
            "",
            "RECOMMENDATION:",
            "  Remove the ENR external validation section from the paper entirely.",
            "  Replace with a reference to the ELSTAT methodology (ESYE 2019) to",
            "  justify the indices directly, without needing US-market cross-validation.",
            "  Alternatively, use Eurostat HICP Construction sub-index for the EU",
            "  comparison (same monetary union, closer structural fit).",
            "",
            "ACTION REQUIRED:",
            "  Review this file and decide whether to keep or remove Section X.X.",
            "  This file was auto-generated by 07_enr_validation.py.",
        ]
        flag_path.write_text('\n'.join(lines), encoding='utf-8')
        print(f"\n[FAILED FLAG] Written -> {flag_path}")
        print("  The ENR section should be reviewed for removal from the paper.")

    # ------------------------------------------------------------------
    # PLOTTING
    # ------------------------------------------------------------------
    def _plot_index_comparison(self, df_elstat, enr_monthly,
                                elstat_annual, enr_annual):
        """
        Two-panel figure:
          Top:    Normalised price levels (2000=100) — visual context only
          Bottom: Annual YoY log-changes for General_Index vs ENR
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.0),
                                        gridspec_kw={'height_ratios': [2, 1]})

        # --- Panel 1: Normalised levels (monthly interpolation for visual only) ---
        gi = df_elstat['General_Index']
        gi_norm = gi / gi.iloc[0] * 100
        enr_common = enr_monthly.loc[gi_norm.index.min():gi_norm.index.max()]
        enr_norm = enr_common / enr_common.iloc[0] * 100

        ax1.plot(gi_norm.index, gi_norm.values,
                 color='#0072B2', linewidth=1.8,
                 label='ELSTAT General Index (Normalised, 2000=100)')
        ax1.plot(enr_norm.index, enr_norm.values,
                 color='#D55E00', linewidth=1.8, linestyle='--',
                 label='ENR CCI (Interpolated levels, Normalised 2000=100)')

        crisis_colors = ['#E69F00', '#CC79A7', '#56B4E9']
        crisis_list = list(self.cfg.CRISIS_PERIODS_ANNUAL.items())
        for (cname, (yr_s, yr_e)), cc in zip(crisis_list, crisis_colors):
            ax1.axvspan(pd.Timestamp(f'{yr_s}-01-01'),
                        pd.Timestamp(f'{yr_e}-12-31'),
                        color=cc, alpha=0.12, label=cname)

        ax1.set_ylabel('Index (2000 = 100)', fontsize=9)
        ax1.legend(fontsize=7, loc='upper left', frameon=True, framealpha=0.9)
        ax1.grid(True, alpha=0.3, linewidth=0.5)
        ax1.tick_params(labelsize=8)

        # --- Panel 2: Annual YoY log-changes ---
        years = [d.year for d in elstat_annual.index]
        gi_yoy = elstat_annual['General_Index'].values
        enr_yoy_vals = enr_annual.loc[elstat_annual.index].values

        ax2.bar([y - 0.2 for y in years], gi_yoy * 100, width=0.35,
                color='#0072B2', alpha=0.7, label='ELSTAT General Index YoY (%)')
        ax2.bar([y + 0.2 for y in years], enr_yoy_vals * 100, width=0.35,
                color='#D55E00', alpha=0.7, label='ENR CCI YoY (%)')
        ax2.axhline(0, color='black', linewidth=0.8)

        for (cname, (yr_s, yr_e)), cc in zip(crisis_list, crisis_colors):
            ax2.axvspan(yr_s - 0.5, yr_e + 0.5, color=cc, alpha=0.12)

        ax2.set_ylabel('Annual YoY change (%)', fontsize=9)
        ax2.set_xlabel('Year', fontsize=9)
        ax2.legend(fontsize=7, frameon=True, framealpha=0.9)
        ax2.grid(True, alpha=0.3, linewidth=0.5, axis='y')
        ax2.tick_params(labelsize=8)

        plt.tight_layout()
        out = self.cfg.FIGURES_DIR / 'fig_enr_validation.png'
        plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"[OK] Figure saved -> {out}")

    def _plot_crisis_heatmap(self, df_crisis: pd.DataFrame):
        if df_crisis.empty:
            return
        pivot = df_crisis.pivot_table(
            index='ELSTAT Series', columns='Crisis Period',
            values="Kendall's tau", aggfunc='first'
        )
        fig, ax = plt.subplots(figsize=(7.2, 3.0))
        im = ax.imshow(pivot.values, cmap='RdYlGn', vmin=-0.8, vmax=0.8, aspect='auto')

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=8, rotation=15, ha='right')
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                            fontsize=8, fontweight='bold',
                            color='black' if abs(val) < 0.5 else 'white')

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Kendall's tau (annual YoY)", fontsize=8)
        plt.tight_layout()
        out = self.cfg.FIGURES_DIR / 'fig_enr_crisis_heatmap.png'
        plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"[OK] Heatmap saved -> {out}")

    def _export_latex(self, df_full: pd.DataFrame, df_crisis: pd.DataFrame):
        lines = []
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\caption{Kendall's $\tau$ Correlation: Greek ELSTAT vs.\ ENR CCI "
                     r"(Annual Year-on-Year Log Changes)}")
        lines.append(r"\label{tab:enr_validation}")
        lines.append(r"\begin{tabular}{llrrrc}")
        lines.append(r"\toprule")
        lines.append(r"Period & ELSTAT Series & $\tau$ & $\rho_S$ & $p$-value & Sig. \\")
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{6}{l}{\textit{(A) Full Period -- Annual YoY}} \\")

        for _, r in df_full.iterrows():
            tau_v = r["Kendall's tau"]
            rho_v = r['Spearman rho']
            pv_v  = r['p-value']
            sig_v = r['Significant (p<0.05)']
            lines.append(
                f"{r['Period']} & {r['ELSTAT Series']} & "
                f"{tau_v:.4f} & {rho_v:.4f} & {pv_v:.4f} & {sig_v} \\\\"
            )

        if not df_crisis.empty:
            lines.append(r"\midrule")
            lines.append(r"\multicolumn{6}{l}{\textit{(B) Crisis Periods -- Annual YoY}} \\")
            prev_crisis = None
            for _, r in df_crisis.iterrows():
                if r['Crisis Period'] != prev_crisis:
                    lines.append(
                        f"\\multicolumn{{6}}{{l}}{{\\quad \\textit{{{r['Crisis Period']}}}}} \\\\"
                    )
                    prev_crisis = r['Crisis Period']
                tau_c = r["Kendall's tau"]
                rho_c = r['Spearman rho']
                pvc   = r['p-value']
                sigc  = r['Significant (p<0.05)']
                n_c   = r['n']
                lines.append(
                    f" & {r['ELSTAT Series']} & "
                    f"{tau_c:.4f} & {rho_c:.4f} & {pvc:.4f} & {sigc} \\\\ % n={n_c}"
                )

        lines.append(r"\bottomrule")
        lines.append(
            r"\multicolumn{6}{l}{\footnotesize \textit{Note: ENR 20-city CCI annual values "
            r"(base 1913=100). Both series expressed as annual YoY log-returns. "
            r"ELSTAT: Jan-to-Jan. ENR: year-over-year. "
            r"$\rho_S$ = Spearman. Sig.\ at 5\%.}} \\"
        )
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        path = self.cfg.TABLES_DIR / 'enr_validation_latex.tex'
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"[OK] LaTeX saved -> {path}")


# --- 4. EXECUTION ---
if __name__ == "__main__":
    cfg = ENRConfig()
    analyzer = ENRCorrelationAnalyzer(cfg)
    df_full, df_crisis = analyzer.run()
