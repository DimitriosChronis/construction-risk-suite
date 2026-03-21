"""
Module: 04_generate_figures.py
Description: Generates publication-quality figures (S-Curves, Rolling Risk, 3D Density).
             Includes auto-cleaning of Infinite/NaN values to prevent crashes.
Author: Dimitrios Chronis
Standards: Publication-Ready (300 DPI), OOP, Type Hinting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import json
import os
from pathlib import Path
from dataclasses import dataclass
from scipy.stats import norm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import Axes3D

# --- 1. LOGGING & STYLE SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(module)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set Professional Plotting Style
try:
    plt.style.use('seaborn-v0_8-paper')
except OSError:
    plt.style.use('seaborn-paper')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300
})

# --- 2. CONFIGURATION ---
@dataclass
class FigConfig:
    """Configuration for figure generation."""
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_PATH: Path = BASE_DIR / 'data' / 'processed' / 'clean_returns.csv'
    RESULTS_DIR: Path = BASE_DIR / 'results' / 'tables'
    FIG_DIR: Path = BASE_DIR / 'results' / 'figures'
    
    # Visual Settings
    COLORS = {'Gaussian': '#0072B2', 'Gumbel': '#D55E00', 'Independent': 'gray'}
    CRISIS_SHADING = [
        ('2008-01-01', '2013-12-31', '#E69F00', 'Financial Crisis'),
        ('2021-01-01', '2024-12-31', '#D55E00', 'Energy Crisis')
    ]

# --- 3. VISUALIZER CLASS ---
class RiskVisualizer:
    
    def __init__(self, config: FigConfig):
        self.cfg = config
        self.cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)

    def run(self):
        logger.info("Starting Visualization Engine...")
        
        self.plot_scurves()
        self.plot_rolling_risk()
        self.plot_3d_density()
        
        logger.info("All figures generated successfully.")

    def plot_scurves(self):
        """Generates the CDF S-Curve with P85 Zoom Inset. Includes Data Cleaning."""
        sim_path = self.cfg.RESULTS_DIR / 'simulation_results.csv'
        if not sim_path.exists():
            logger.warning("Simulation results not found. Skipping S-Curves.")
            return

        logger.info("Generating Fig 6: S-Curves...")
        df = pd.read_csv(sim_path)
        
        initial_len = len(df)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        if len(df) < initial_len:
            logger.warning(f"Dropped {initial_len - len(df)} infinite/NaN simulation runs.")
            
        if df.empty:
            logger.error("No valid data left to plot S-Curves.")
            return

        df = df[df['Gumbel'] < 100_000_000] # Sanity check

        fig, ax = plt.subplots(figsize=(10, 6))
        
        for col, color in self.cfg.COLORS.items():
            if col in df.columns:
                data = np.sort(df[col])
                y = np.arange(1, len(data)+1) / len(data)
                ax.plot(data, y, label=col, color=color, linewidth=2.5, alpha=0.9)
        
        ax.axhline(0.85, color='black', linestyle='--', alpha=0.6)
        ax.text(df['Independent'].min(), 0.86, 'P85 Confidence', fontsize=10, fontweight='bold')
        
        ax.set_title("Total Project Cost CDF (Cumulative Distribution Function)", fontweight='bold')
        ax.set_xlabel("Total Cost (€)")
        ax.set_ylabel("Probability P(X ≤ x)")
        ax.legend(loc='upper left', frameon=True)
        ax.grid(True, alpha=0.3)
        
        # --- Dynamic Zoom ---
        axins = inset_axes(ax, width="40%", height="40%", loc='lower right', borderpad=2)
        
        try:
            p85_g = np.percentile(df['Gumbel'], 85)
            p85_i = np.percentile(df['Independent'], 85)
            center = (p85_g + p85_i) / 2
            span = abs(p85_g - p85_i) * 3
            span = max(span, 100000) 
        except Exception:
            center = df.mean().mean()
            span = 500000

        for col, color in self.cfg.COLORS.items():
            if col in df.columns:
                data = np.sort(df[col])
                y = np.arange(1, len(data)+1) / len(data)
                mask = (data > center - span) & (data < center + span)
                if np.any(mask):
                    axins.plot(data[mask], y[mask], color=color, linewidth=2.5)

        axins.set_xlim(center - span/2, center + span/2)
        axins.set_ylim(0.80, 0.90)
        axins.set_title("Tail Risk Zoom @ P85", fontsize=9, fontweight='bold')
        axins.grid(True, alpha=0.3)
        axins.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.2f}M'))
        
        out_file = self.cfg.FIG_DIR / 'fig3_scurve_zoom.png'
        plt.savefig(out_file, bbox_inches='tight')
        plt.close()

    def plot_rolling_risk(self):
        """Generates the Rolling Spearman Correlation plot with Crisis Shading."""
        if not self.cfg.DATA_PATH.exists(): return
        
        logger.info("Generating Fig 4: Rolling Systemic Risk...")
        df_prices = pd.read_csv(self.cfg.DATA_PATH, index_col='Date', parse_dates=True)
        returns = np.log(df_prices / df_prices.shift(1)).dropna()
        
        window = 24
        ranked = returns.rank(pct=True)
        rolling_corr = ranked.rolling(window=window).corr()
        
        n = returns.shape[1]
        avg_corr = rolling_corr.groupby(level=0).apply(lambda x: (x.values.sum() - n) / (n * (n - 1)))
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(avg_corr.index, avg_corr, color='#2ca02c', linewidth=2, label='Avg. Systemic Correlation')
        
        # --- Crisis Shading ---
        for start, end, color, label in self.cfg.CRISIS_SHADING:
            try:
                ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=color, alpha=0.15, label=label)
            except ValueError: pass
            
        ax.set_title("Evolution of Systemic Risk (Rolling 24-month Rank Correlation)", fontweight='bold')
        ax.set_ylabel("Correlation Coefficient (τ)")
        ax.legend(loc='upper left', frameon=True)
        ax.grid(True, alpha=0.3)
        
        out_file = self.cfg.FIG_DIR / 'fig4_rolling_risk.png'
        plt.savefig(out_file, bbox_inches='tight')
        plt.close()

    def plot_3d_density(self):
        """Generates the 3D Bar Plot."""
        logger.info("Generating Fig 5: 3D Copula Density...")
        
        theta = 4.0 # Set for visualization purpose matching paper text
        
        n_samples = 5000
        alpha = 1.0 / theta
        g = np.random.gumbel(0, 1, n_samples)
        noise = np.random.uniform(0, 1, (n_samples, 2))
        
        term_common = np.exp(-g/alpha)
        u_dep = np.zeros((n_samples, 2))
        for i in range(2):
            term_indep = (-np.log(noise[:, i]))**theta
            u_dep[:, i] = np.exp(-(term_indep + term_common)**alpha)
            
        x, y = norm.ppf(u_dep[:, 0]), norm.ppf(u_dep[:, 1])
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        hist, xedges, yedges = np.histogram2d(x, y, bins=30, range=[[-3, 3], [-3, 3]])
        xpos, ypos = np.meshgrid(xedges[:-1] + 0.1, yedges[:-1] + 0.1, indexing="ij")
        xpos = xpos.ravel(); ypos = ypos.ravel(); zpos = 0
        dx = dy = 0.2 * np.ones_like(zpos)
        dz = hist.ravel()
        
        cmap = plt.cm.viridis
        max_height = np.max(dz) if len(dz) > 0 else 1
        rgba = [cmap(k/max_height) for k in dz]
        
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, alpha=0.8, zsort='average')
        
        ax.set_title(f"Tail Dependence Structure (Gumbel Copula, $\\theta={theta:.2f}$)", fontweight='bold')
        ax.set_xlabel('Asset A (Z-Score)')
        ax.set_ylabel('Asset B (Z-Score)')
        ax.set_zlabel('Joint Density')
        ax.view_init(elev=35, azim=225)
        
        out_file = self.cfg.FIG_DIR / 'fig5_3d_density.png'
        plt.savefig(out_file, bbox_inches='tight')
        plt.close()

# --- 4. EXECUTION ---
if __name__ == "__main__":
    config = FigConfig()
    RiskVisualizer(config).run()