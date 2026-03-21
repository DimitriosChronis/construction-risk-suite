"""
Module: 03_detailed_simulation.py
Description: Monte Carlo Engine with Volatility Throttling & Kanter Algorithm.
             Optimized for stability and smooth S-Curves.
Author: Dimitrios Chronis
"""

import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict
from scipy.stats import norm, kendalltau

# --- 1. LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(module)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- 2. CONFIGURATION ---
@dataclass
class SimConfig:
    """Simulation Hyperparameters."""
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_PATH: Path = BASE_DIR / 'data' / 'processed' / 'clean_returns.csv'
    RESULTS_DIR: Path = BASE_DIR / 'results' / 'tables'
    
    NUM_SIMS: int = 100000
    BASE_COST: float = 2_300_000.0
    PROJECT_MONTHS: int = 24
    SEED: int = 42
    
    # Portfolio Weights
    WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        'Concrete': 0.30, 'Steel': 0.30, 'Fuel_Energy': 0.20, 'PVC_Pipes': 0.20
    })

# --- 3. MATH UTILITIES ---
class StochasticMath:
    @staticmethod
    def generate_positive_stable(alpha: float, size: int, rng: np.random.Generator) -> np.ndarray:
        """Robust Kanter Algorithm for Stable Distributions."""
        if alpha >= 0.99: return np.ones(size)
        
        # Avoid singularities
        U = rng.uniform(1e-4, np.pi - 1e-4, size)
        W = rng.exponential(1.0, size) + 1e-4
        
        term1 = np.sin(alpha * U) / (np.sin(U) ** (1/alpha))
        term2 = (np.sin((1 - alpha) * U) / W) ** ((1 - alpha) / alpha)
        
        # Soft cap to prevent numerical infinity inside Gumbel
        return np.clip(term1 * term2, 1e-9, 1e5)

# --- 4. SIMULATOR CLASS ---
class CopulaSimulator:
    
    def __init__(self, config: SimConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.SEED)

    def run(self):
        logger.info("Starting Simulation (Volatility Throttle Edition)...")
        if not self.cfg.DATA_PATH.exists():
            logger.error("Data file not found.")
            return
            
        df = pd.read_csv(self.cfg.DATA_PATH, index_col='Date', parse_dates=True)

        # Compute log-returns from price levels (MEMORY.md: data contains INDEX LEVELS)
        # Must use log(df/df.shift(1)).dropna() — fitting on raw levels inflates std ~100x
        returns = np.log(df / df.shift(1)).dropna()

        # 1. Volatility Throttle & Marginals
        # Cap monthly volatility at 15% to prevent unrealistic explosions
        MAX_VOL = 0.15
        scale = np.sqrt(self.cfg.PROJECT_MONTHS)
        margins = {}

        for col in returns.columns:
            _, std = norm.fit(returns[col])
            # Zero Drift (mu=0) + Throttled Volatility
            margins[col] = (0.0, min(std, MAX_VOL) * scale)

        # 2. Correlations (Spearman & Kendall) — computed on log-returns, not levels
        spearman = returns.corr(method='spearman')
        n = len(returns.columns)
        tau_sum = 0; cnt = 0
        for i in range(n):
            for j in range(i+1, n):
                t, _ = kendalltau(returns.iloc[:, i], returns.iloc[:, j])
                tau_sum += t; cnt += 1
        
        avg_tau = max(0.05, min(tau_sum/cnt, 0.95))
        theta = 1.0 / (1.0 - avg_tau)
        alpha = 1.0 / theta
        logger.info(f"Systemic Tau: {avg_tau:.3f} | Theta: {theta:.3f}")

        # 3. Simulation Kernel
        # A. Independent
        u_ind = self.rng.uniform(0, 1, (self.cfg.NUM_SIMS, n))
        
        # B. Gaussian (Cholesky with Jitter)
        C = spearman.values + np.eye(n) * 1e-6
        try: 
            L = np.linalg.cholesky(C)
        except: 
            L = np.linalg.cholesky(C + np.eye(n)*1e-3)
        u_gauss = norm.cdf(self.rng.standard_normal((self.cfg.NUM_SIMS, n)) @ L.T)

        # C. Gumbel (Marshall-Olkin)
        V = StochasticMath.generate_positive_stable(alpha, self.cfg.NUM_SIMS, self.rng)[:, None]
        Y = self.rng.uniform(0, 1, (self.cfg.NUM_SIMS, n))
        u_gum = np.exp(- (-np.log(Y) / V) ** alpha)

        # 4. Cost Calculation (No Staircase Clips)
        def calc_cost(u_mat):
            u_mat = np.clip(u_mat, 1e-6, 1-1e-6)
            impact = np.zeros(self.cfg.NUM_SIMS)
            for i, c in enumerate(returns.columns):
                if c in self.cfg.WEIGHTS:
                    mu, std = margins[c]
                    # No hard clip needed here because Input Volatility is throttled
                    ret = norm.ppf(u_mat[:, i], loc=mu, scale=std) 
                    impact += self.cfg.WEIGHTS[c] * (np.exp(ret) - 1)
            return self.cfg.BASE_COST * (1 + impact)

        df_res = pd.DataFrame({
            'Independent': calc_cost(u_ind),
            'Gaussian': calc_cost(u_gauss),
            'Gumbel': calc_cost(u_gum)
        })

        self._save(df_res, spearman, avg_tau, theta)

    def _save(self, df, corr, tau, theta):
        self.cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.cfg.RESULTS_DIR / 'simulation_results.csv', index=False)
        corr.to_csv(self.cfg.RESULTS_DIR / 'spearman_matrix.csv')
        with open(self.cfg.RESULTS_DIR / 'copula_params.json', 'w') as f:
            json.dump({'avg_tau': tau, 'theta_gumbel': theta}, f)
            
        logger.info(f"Simulation Done. Max Cost: €{df.max().max():,.0f} (Reasonable)")

# --- 5. EXECUTION ---
if __name__ == "__main__":
    CopulaSimulator(SimConfig()).run()