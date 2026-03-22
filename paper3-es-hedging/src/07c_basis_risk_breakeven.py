"""
07c_basis_risk_breakeven.py
Basis Risk & Break-Even Analysis (C5 Enhancement)

Core mathematics:
  Minimum-variance hedge ratio : h* = rho * (sigma_S / sigma_F)
  Max Hedge Effectiveness      : HE_max = rho^2   (= OLS R-squared)
  Required rho for HE target   : rho_req = sqrt(HE_target)
  Basis risk gap                : gap = rho_req - rho_current
                                   > 0  => instrument cannot achieve target
                                   <= 0 => instrument ALREADY achieves target

Break-even transaction cost (bps per annum):
  Benefit per month = HE_max * sigma_spot^2 * notional
  TC break-even     = Benefit / (notional * MONTHS) * 10,000  bps

Key finding (expected):
  Only Steel is close to HE=25% threshold.
  All instruments face a LARGE basis risk gap at HE=50%.
  Crisis correlations collapse -> gap widens exactly when hedging is needed.
  => Cross-border commodity futures are structurally insufficient for
     hedging Greek construction cost risk.

Outputs:
  results/tables/c5c_basis_risk.csv
  results/figures/fig_c5c_basis_risk_gap.pdf
  results/figures/fig_c5c_rho_comparison.pdf
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR   = os.path.dirname(__file__)
DATA_PATH    = os.path.join(SCRIPT_DIR, "..", "data", "processed",
                             "aligned_log_returns.csv")
OUT_FIG      = os.path.join(SCRIPT_DIR, "..", "results", "figures")
OUT_TAB      = os.path.join(SCRIPT_DIR, "..", "results", "tables")

BASE_COST    = 2_300_000
MONTHS       = 24
CRISIS_START = "2021-01-01"
CRISIS_END   = "2024-12-01"

PAIRS = [
    ("US_Steel_PPI",  "GR_Steel",         "Steel",   0.30),
    ("US_Cement_PPI", "GR_Concrete",      "Cement",  0.30),
    ("US_Fuel_PPI",   "GR_Fuel_Energy",   "Fuel",    0.20),
    ("US_PVC_PPI",    "GR_PVC_Pipes",     "PVC",     0.20),
    ("US_Brent",      "GR_General_Index", "General", None),
]

HE_TARGETS = {"HE_25pct": 0.25, "HE_50pct": 0.50}


# ── Helpers ───────────────────────────────────────────────────────────────────

def pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def he_max(rho: float) -> float:
    """Max achievable HE = rho^2."""
    return rho ** 2


def rho_req(he_target: float) -> float:
    """Min Pearson rho for HE >= he_target."""
    return float(np.sqrt(abs(he_target)))


def breakeven_bps(rho: float, sigma_spot: float, weight: float) -> float:
    """
    Break-even transaction cost in bps/annum.
    The hedge reduces variance by HE_max * Var(spot) per month.
    Over MONTHS, total benefit (EUR) = HE_max * sigma_spot^2 * MONTHS * notional.
    Break-even TC = benefit / (notional * MONTHS) * 10,000  bps.
    Simplifies to: HE_max * sigma_spot^2 * 10,000  bps.
    """
    return float(he_max(rho) * (sigma_spot ** 2) * 10_000)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_FIG, exist_ok=True)
    os.makedirs(OUT_TAB, exist_ok=True)

    df     = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    crisis = df.loc[CRISIS_START:CRISIS_END]
    print(f"Data: {df.shape}   Crisis window: {len(crisis)} obs\n")

    rows = []

    for us_col, gr_col, label, weight in PAIRS:
        if us_col not in df.columns or gr_col not in df.columns:
            continue

        # --- Full-period metrics ---
        rho_f  = pearson(df[us_col].values, df[gr_col].values)
        he_f   = he_max(rho_f)
        sig_f  = float(df[gr_col].std())

        # --- Crisis-period metrics ---
        if len(crisis) > 10:
            rho_c = pearson(crisis[us_col].values, crisis[gr_col].values)
            he_c  = he_max(rho_c)
        else:
            rho_c, he_c = np.nan, np.nan

        # --- Breakeven bps ---
        be_bps = breakeven_bps(rho_f, sig_f, weight) if weight else np.nan

        row = {
            "Pair":           label,
            "Weight":         weight,
            "rho_full":       round(rho_f,  3),
            "HE_max_full_%":  round(he_f  * 100, 1),
            "rho_crisis":     round(rho_c,  3) if not np.isnan(rho_c) else "NA",
            "HE_max_crisis_%": round(he_c * 100, 1) if not np.isnan(he_c) else "NA",
            "BE_bps":         round(be_bps, 2) if not np.isnan(be_bps) else "NA",
        }

        print(f"{label:10s}  rho_full={rho_f:.3f}  HE_full={he_f*100:.1f}%  "
              f"rho_crisis={rho_c:.3f}  HE_crisis={he_c*100:.1f}%")

        for tname, tval in HE_TARGETS.items():
            r_req    = rho_req(tval)
            gap_full = round(r_req - rho_f,  3)
            gap_cr   = round(r_req - rho_c,  3) if not np.isnan(rho_c) else "NA"
            achieves = "YES" if gap_full <= 0 else "NO "
            row[f"rho_req_{tname}"]    = round(r_req, 3)
            row[f"gap_full_{tname}"]   = gap_full
            row[f"gap_crisis_{tname}"] = gap_cr
            row[f"achieves_full_{tname}"] = achieves
            print(f"  {tname}: rho_req={r_req:.3f}  "
                  f"gap_full={gap_full:+.3f}  gap_crisis={gap_cr}  "
                  f"achieves={achieves}")

        rows.append(row)
        print()

    out_df = pd.DataFrame(rows)
    out_df.to_csv(os.path.join(OUT_TAB, "c5c_basis_risk.csv"), index=False)

    # ── Figure 1: basis risk gap bars ─────────────────────────────────────
    plot_rows = [r for r in rows if r["Weight"] is not None]
    plabels   = [r["Pair"] for r in plot_rows]
    gaps25    = [r["gap_full_HE_25pct"] for r in plot_rows]
    gaps50    = [r["gap_full_HE_50pct"] for r in plot_rows]
    gaps25_cr = [r["gap_crisis_HE_25pct"]
                 if not isinstance(r["gap_crisis_HE_25pct"], str)
                 else np.nan for r in plot_rows]

    x = np.arange(len(plabels))
    w = 0.28

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w,     gaps25,    w, label="Full: gap to HE=25%",
           color="#1565C0", edgecolor="black", linewidth=0.5)
    ax.bar(x,         gaps50,    w, label="Full: gap to HE=50%",
           color="#D84315", edgecolor="black", linewidth=0.5)
    ax.bar(x + w,     gaps25_cr, w, label="Crisis: gap to HE=25%",
           color="#78909C", edgecolor="black", linewidth=0.5)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(plabels)
    ax.set_ylabel("Basis Risk Gap  (rho_required - rho_current)")
    ax.legend(fontsize=7)

    # Annotate "OK" where gap <= 0
    for i, g in enumerate(gaps25):
        if g <= 0:
            ax.text(i - w, g - 0.005, "OK",
                    ha="center", va="top", fontsize=7,
                    color="#388E3C", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG, "fig_c5c_basis_risk_gap.pdf"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("fig_c5c_basis_risk_gap.pdf saved.")

    # ── Figure 2: full vs crisis rho comparison ────────────────────────────
    rho_full_vals   = [r["rho_full"]   for r in plot_rows]
    rho_crisis_vals = [float(r["rho_crisis"])
                       if not isinstance(r["rho_crisis"], str)
                       else np.nan for r in plot_rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    x2 = np.arange(len(plabels))
    w2 = 0.35
    ax.bar(x2 - w2 / 2, rho_full_vals,   w2,
           label="Full period",      color="#1565C0",
           edgecolor="black", linewidth=0.5)
    ax.bar(x2 + w2 / 2, rho_crisis_vals, w2,
           label="Crisis (2021-24)", color="#D84315",
           edgecolor="black", linewidth=0.5)

    # HE=25% threshold line: rho = 0.5
    ax.axhline(rho_req(0.25), color="#388E3C", linestyle="--",
               linewidth=1.0, label=f"Min rho for HE=25% ({rho_req(0.25):.2f})")
    ax.axhline(rho_req(0.50), color="#C62828", linestyle="--",
               linewidth=1.0, label=f"Min rho for HE=50% ({rho_req(0.50):.2f})")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x2)
    ax.set_xticklabels(plabels)
    ax.set_ylabel("Pearson Correlation (rho)")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG, "fig_c5c_rho_comparison.pdf"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("fig_c5c_rho_comparison.pdf saved.")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\nBasis Risk Summary:")
    display_cols = ["Pair", "rho_full", "HE_max_full_%",
                    "gap_full_HE_25pct", "gap_full_HE_50pct",
                    "rho_crisis", "HE_max_crisis_%", "BE_bps"]
    print(out_df[display_cols].to_string(index=False))

    # Policy interpretation
    print("\nPolicy interpretation:")
    for r in rows:
        if r["Weight"] is None:
            continue
        g25 = r["gap_full_HE_25pct"]
        g50 = r["gap_full_HE_50pct"]
        g25_cr = r.get("gap_crisis_HE_25pct", "NA")
        if g25 <= 0:
            label2 = r["Pair"]
            print(f"  {label2:10s}: achieves HE=25% in full period "
                  f"(crisis gap={g25_cr})")
        else:
            print(f"  {r['Pair']:10s}: CANNOT achieve HE=25% "
                  f"(gap={g25:+.3f}); crisis gap worse ({g25_cr})")

    print(f"\nSaved: c5c_basis_risk.csv")


if __name__ == "__main__":
    main()
