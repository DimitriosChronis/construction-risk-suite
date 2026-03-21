"""
Module: 10_publication_figures.py
Description: Regenerates all three core paper figures (fig3, fig4, fig5) to
             strict ASCE publication standards:
               - 300 DPI minimum
               - Single-column (3.5 in) or double-column (7.2 in) width
               - Times New Roman or equivalent serif font
               - Axis labels with units in parentheses
               - No chart-junk (minimal grid, no top/right spines)
               - Colourblind-safe palette (IBM / Wong palette)
               - No embedded figure captions (ASCE: captions in LaTeX only)
Author: Dimitrios Chronis
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import json
import warnings
from pathlib import Path
from dataclasses import dataclass
from scipy.stats import norm
warnings.filterwarnings('ignore')


# ── ASCE STYLE CONSTANTS ──────────────────────────────────────────────────────
ASCE_SINGLE_W = 3.50   # inches
ASCE_DOUBLE_W = 7.20   # inches
ASCE_DPI      = 300

# Wong (2011) colourblind-safe palette  ─ ASCE compatible
C_BLUE    = '#0072B2'   # Gaussian
C_ORANGE  = '#D55E00'   # Gumbel
C_GREY    = '#999999'   # Independent
C_GREEN   = '#009E73'   # Systemic/safe
C_AMBER   = '#E69F00'   # Crisis 1
C_PURPLE  = '#CC79A7'   # Crisis 2
C_SKY     = '#56B4E9'   # Accent

ASCE_RC = {
    # Font
    'font.family':          'serif',
    'font.serif':           ['Times New Roman', 'DejaVu Serif', 'Georgia'],
    'mathtext.fontset':     'dejavuserif',
    # Sizes (ASCE specifies ≥ 8 pt after reduction)
    'axes.labelsize':        9,
    'axes.titlesize':        9,
    'xtick.labelsize':       8,
    'ytick.labelsize':       8,
    'legend.fontsize':       8,
    'legend.title_fontsize': 8,
    'figure.titlesize':      9,
    # Lines / ticks
    'axes.linewidth':        0.7,
    'lines.linewidth':       1.6,
    'xtick.major.width':     0.7,
    'ytick.major.width':     0.7,
    'xtick.direction':       'in',
    'ytick.direction':       'in',
    # Grid
    'axes.grid':             True,
    'grid.alpha':            0.25,
    'grid.linewidth':        0.4,
    # Resolution
    'figure.dpi':            ASCE_DPI,
    'savefig.dpi':           ASCE_DPI,
    'savefig.bbox':         'tight',
    'savefig.facecolor':    'white',
}

def _apply_asce_spines(ax):
    """Remove top and right spines (ASCE clean style)."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def _fmt_eur(x, _):
    """Formatter: EUR values → '2.30M' style."""
    if abs(x) >= 1e6:
        return f'{x/1e6:.2f}M'
    return f'{x/1e3:.0f}k'


# ── CONFIG ────────────────────────────────────────────────────────────────────
@dataclass
class ASCEFigConfig:
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_PATH: Path = BASE_DIR / 'data' / 'processed' / 'clean_returns.csv'
    SIM_PATH:  Path = BASE_DIR / 'results' / 'tables' / 'simulation_results.csv'
    FIG_DIR:   Path = BASE_DIR / 'results' / 'figures'
    CRISIS_SHADING = [
        ('2008-01-01', '2013-12-31', C_AMBER,  'Greek Financial Crisis (2008-2013)'),
        ('2021-01-01', '2024-06-30', C_ORANGE, 'Energy/Inflation Crisis (2021-2024)'),
    ]


# ── FIGURE 3: CDF S-CURVES ────────────────────────────────────────────────────
def fig3_scurves(cfg: ASCEFigConfig):
    """
    Fig. 3. Cumulative distribution functions of total project cost under three
    dependence assumptions. Inset: tail region near the 85th percentile.

    X-axis: data-driven from P0.5 to P99.5 across all three models, so the
    entire meaningful distribution is visible within the base-cost neighbourhood.
    Base cost = EUR 2,300,000; expected range EUR 2.0M -- 2.7M after log-return fix.
    """
    sim_path = cfg.SIM_PATH
    if not sim_path.exists():
        print(f"[WARN] {sim_path} not found — run 03_detailed_simulation.py first.")
        return

    df = pd.read_csv(sim_path)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Sanity check: warn if simulation data looks like it was computed on price levels
    gumbel_p85 = df['Gumbel'].quantile(0.85)
    base_cost  = 2_300_000.0
    if gumbel_p85 > base_cost * 3:
        print(f"[WARN] fig3: Gumbel P85={gumbel_p85:,.0f} is >{base_cost*3:,.0f}."
              f" Simulation may have been run on price levels, not log-returns."
              f" Re-run 03_detailed_simulation.py first.")

    # ── Data-driven x-axis limits ──────────────────────────────────────────────
    # Use P0.5 (left tail) to P99.5 (right tail) across all three models so the
    # full distribution is shown without extreme outliers distorting the scale.
    all_vals = pd.concat([df[c] for c in ['Independent', 'Gaussian', 'Gumbel']
                          if c in df.columns])
    x_lo = all_vals.quantile(0.005)
    x_hi = all_vals.quantile(0.995)
    # Round to nearest 50k for clean tick labels
    x_lo = np.floor(x_lo / 50_000) * 50_000
    x_hi = np.ceil(x_hi  / 50_000) * 50_000

    fig, ax = plt.subplots(figsize=(ASCE_DOUBLE_W, 3.8))

    styles = {
        'Independent': (C_GREY,   '-',  'Independent'),
        'Gaussian':    (C_BLUE,   '--', 'Gaussian Copula'),
        'Gumbel':      (C_ORANGE, '-',  'Gumbel Copula'),
    }

    for col, (color, ls, label) in styles.items():
        if col not in df.columns:
            continue
        data = np.sort(df[col].values)
        cdf  = np.arange(1, len(data) + 1) / len(data)
        ax.plot(data, cdf, color=color, linestyle=ls,
                linewidth=1.8, label=label, alpha=0.92)

    # P85 reference line
    ax.axhline(0.85, color='black', linewidth=0.8, linestyle=':')
    ax.text(x_lo + (x_hi - x_lo) * 0.01, 0.856, 'P85',
            fontsize=7, va='bottom', color='black')

    # Base cost vertical marker
    ax.axvline(base_cost, color=C_GREEN, linewidth=0.9, linestyle='--', alpha=0.7)
    ax.text(base_cost, 0.04, 'Base\nCost',
            fontsize=6.5, ha='center', va='bottom', color=C_GREEN,
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor=C_GREEN, alpha=0.8, linewidth=0.5))

    ax.set_xlabel('Total Project Cost (EUR)', fontsize=9)
    ax.set_ylabel('Cumulative Probability $F(x)$', fontsize=9)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-0.02, 1.04)
    # Tick every 50k within the data range
    tick_step = 50_000
    ax.xaxis.set_major_locator(mticker.MultipleLocator(tick_step))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_eur))
    ax.legend(loc='upper left', frameon=True, framealpha=0.9,
              edgecolor='#AAAAAA')
    _apply_asce_spines(ax)
    ax.tick_params(labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    # ── Inset: tail zoom around P85 ───────────────────────────────────────────
    axins = inset_axes(ax, width='36%', height='44%',
                       loc='lower right', borderpad=1.8)

    p85_lo = min(df[c].quantile(0.85) for c in styles if c in df.columns)
    p85_hi = max(df[c].quantile(0.85) for c in styles if c in df.columns)
    center = (p85_lo + p85_hi) / 2.0
    # Span: encompass spread between models + 20k padding on each side
    span = max(abs(p85_hi - p85_lo) * 2.5, 40_000)
    ins_lo = center - span
    ins_hi = center + span

    for col, (color, ls, _) in styles.items():
        if col not in df.columns:
            continue
        data = np.sort(df[col].values)
        cdf  = np.arange(1, len(data) + 1) / len(data)
        mask = (data >= ins_lo) & (data <= ins_hi)
        if mask.any():
            axins.plot(data[mask], cdf[mask], color=color,
                       linestyle=ls, linewidth=1.5)

    axins.axhline(0.85, color='black', linewidth=0.7, linestyle=':')
    axins.set_xlim(ins_lo, ins_hi)
    axins.set_ylim(0.80, 0.90)
    axins.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
    axins.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_eur))
    axins.set_title('Tail Detail @ P85', fontsize=7, fontweight='bold')
    axins.tick_params(labelsize=6)
    plt.setp(axins.get_xticklabels(), rotation=20, ha='right')
    axins.spines['top'].set_visible(False)
    axins.spines['right'].set_visible(False)
    axins.grid(True, alpha=0.2, linewidth=0.3)

    plt.tight_layout()
    out = cfg.FIG_DIR / 'fig3_scurve_asce.png'
    plt.savefig(out, dpi=ASCE_DPI)
    plt.close()
    print(f'[OK] fig3 -> {out}')
    print(f'     x-axis range: EUR {x_lo:,.0f} -- EUR {x_hi:,.0f}')
    print(f'     Gumbel P85={gumbel_p85:,.0f}, P99={df["Gumbel"].quantile(0.99):,.0f}')


# ── FIGURE 4: ROLLING SYSTEMIC RISK ──────────────────────────────────────────
def fig4_rolling_risk(cfg: ASCEFigConfig):
    """
    Fig. 4. Rolling 24-month average Spearman rank correlation among the four
    construction cost components, with crisis-period shading.
    """
    df_prices = pd.read_csv(cfg.DATA_PATH, index_col='Date', parse_dates=True)
    returns   = np.log(df_prices / df_prices.shift(1)).dropna()

    cols   = ['Concrete', 'Steel', 'Fuel_Energy', 'PVC_Pipes']
    rets   = returns[cols]
    window = 24
    n      = len(cols)

    ranked      = rets.rank(pct=True)
    rolling_mat = ranked.rolling(window=window).corr()
    avg_corr    = rolling_mat.groupby(level=0).apply(
        lambda x: (x.values.sum() - n) / (n * (n - 1))
    )

    fig, ax = plt.subplots(figsize=(ASCE_DOUBLE_W, 3.4))

    ax.plot(avg_corr.index, avg_corr.values,
            color=C_GREEN, linewidth=1.8,
            label='24-Month Rolling Avg. Spearman $\\rho$')
    ax.fill_between(avg_corr.index, avg_corr.values,
                    alpha=0.12, color=C_GREEN)

    # Horizontal reference lines
    for level, label in [(0.33, 'Stable\u00a0$\\rho$=0.33'),
                          (0.85, 'Crisis\u00a0$\\rho$=0.85')]:
        ax.axhline(level, color='black', linewidth=0.7,
                   linestyle='--', alpha=0.55)
        ax.text(avg_corr.index[-1], level + 0.012,
                label, fontsize=7, ha='right', va='bottom', color='#444444')

    # Crisis shading
    legend_patches = [
        Line2D([0], [0], color=C_GREEN, linewidth=1.8,
               label='24-Month Rolling Avg. Spearman $\\rho$')
    ]
    shade_colors = [C_AMBER, C_ORANGE]
    for (start, end, color, label), sc in zip(cfg.CRISIS_SHADING, shade_colors):
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   color=sc, alpha=0.14)
        legend_patches.append(
            mpatches.Patch(facecolor=sc, alpha=0.4, label=label)
        )

    ax.set_ylabel('Average Spearman $\\rho$ (rank correlation)', fontsize=9)
    ax.set_xlabel('Date', fontsize=9)
    ax.set_ylim(-0.1, 1.05)
    ax.legend(handles=legend_patches, loc='upper left',
              frameon=True, framealpha=0.9, fontsize=7, edgecolor='#AAAAAA')
    _apply_asce_spines(ax)
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    out = cfg.FIG_DIR / 'fig4_rolling_risk_asce.png'
    plt.savefig(out, dpi=ASCE_DPI)
    plt.close()
    print(f'[OK] fig4 -> {out}')


# ── FIGURE 5: 3-D GUMBEL COPULA DENSITY ──────────────────────────────────────
def fig5_3d_density(cfg: ASCEFigConfig):
    """
    Fig. 5. Joint density of the bivariate Gumbel copula (theta=6.67) plotted
    on a standard normal probability scale. The concentration of mass in the
    upper tail illustrates asymmetric co-movement during cost shocks.
    """
    from mpl_toolkits.mplot3d import Axes3D

    rng   = np.random.default_rng(42)
    theta = 6.67  # tau=0.85 crisis regime: theta = 1/(1-tau) = 1/0.15 = 6.67
    alpha = 1.0 / theta
    n     = 8_000

    # Marshall-Olkin Gumbel simulation
    U_raw = rng.uniform(1e-4, np.pi - 1e-4, n)
    W_raw = rng.exponential(1.0, n) + 1e-4
    t1    = np.sin(alpha * U_raw) / (np.sin(U_raw) ** (1 / alpha))
    t2    = (np.sin((1 - alpha) * U_raw) / W_raw) ** ((1 - alpha) / alpha)
    V     = np.clip(t1 * t2, 1e-9, 1e5)[:, None]

    Y     = rng.uniform(0, 1, (n, 2))
    u_cop = np.exp(- (-np.log(np.clip(Y, 1e-9, 1 - 1e-9)) / V) ** alpha)

    # Normal-quantile transform
    x = norm.ppf(np.clip(u_cop[:, 0], 1e-6, 1 - 1e-6))
    y = norm.ppf(np.clip(u_cop[:, 1], 1e-6, 1 - 1e-6))
    valid = np.isfinite(x) & np.isfinite(y)
    x, y  = x[valid], y[valid]

    bins  = 28
    hist, xe, ye = np.histogram2d(x, y, bins=bins, range=[[-3, 3], [-3, 3]])
    xpos  = ((xe[:-1] + xe[1:]) / 2).repeat(bins)
    ypos  = np.tile((ye[:-1] + ye[1:]) / 2, bins)
    dz    = hist.ravel()
    dx    = dy = (xe[1] - xe[0]) * 0.85

    fig   = plt.figure(figsize=(ASCE_DOUBLE_W * 0.6, 4.8))
    ax    = fig.add_subplot(111, projection='3d')

    cmap      = plt.cm.plasma
    max_h     = dz.max() if dz.max() > 0 else 1
    rgba      = [cmap(k / max_h) for k in dz]

    ax.bar3d(xpos, ypos, np.zeros_like(dz),
             dx, dy, dz, color=rgba, alpha=0.86, zsort='average', shade=True)

    ax.set_xlabel('Cost Component A\n(Std. Normal)', fontsize=8, labelpad=6)
    ax.set_ylabel('Cost Component B\n(Std. Normal)', fontsize=8, labelpad=6)
    ax.set_zlabel('Joint Frequency', fontsize=8, labelpad=4)
    ax.view_init(elev=32, azim=225)
    ax.tick_params(labelsize=7)

    # Colour-bar annotation
    sm  = plt.cm.ScalarMappable(cmap=cmap,
                                 norm=plt.Normalize(vmin=0, vmax=max_h))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.08, aspect=15)
    cbar.set_label('Joint Frequency', fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    plt.tight_layout()
    out = cfg.FIG_DIR / 'fig5_3d_density_asce.png'
    plt.savefig(out, dpi=ASCE_DPI)
    plt.close()
    print(f'[OK] fig5 -> {out}')


# ── EXECUTION ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    plt.rcParams.update(ASCE_RC)

    cfg = ASCEFigConfig()
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)

    print('\nGenerating Publication-quality figures...')
    fig3_scurves(cfg)
    fig4_rolling_risk(cfg)
    fig5_3d_density(cfg)

    print('\n[DONE] All Publication figures saved to results/figures/')
    print('  fig3_scurve_asce.png')
    print('  fig4_rolling_risk_asce.png')
    print('  fig5_3d_density_asce.png')
