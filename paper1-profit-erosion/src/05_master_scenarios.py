"""
Module: 05_master_scenarios.py
Description: Advanced Risk Quantification Engine.
             Features: Higham's Algorithm, Sensitivity Analysis (Time/Profile), Stress Testing.
             Includes solutions for Static Weights, Time Horizon, and Parametric Stress.
Author: Dimitrios Chronis
Standards: Enterprise-Grade (OOP, Logging, Numerical Stability)
"""

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from scipy.stats import norm

# --- 1. ENTERPRISE LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(module)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- 2. CONFIGURATION OBJECT (Data Class) ---
@dataclass
class SimulationConfig:
    """Immutable configuration for the Risk Simulation Engine."""
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_PATH: Path = BASE_DIR / 'data' / 'processed' / 'clean_returns.csv'
    RESULTS_DIR: Path = BASE_DIR / 'results' / 'final_analysis'
    
    # Base Simulation Parameters
    NUM_SIMS: int = 100000        
    BASE_COST: float = 2_300_000.0
    RANDOM_SEED: int = 42
    
    # Solution 3: Parametric Stress (Variable Black Swan Factor)
    BLACK_SWAN_FACTOR: float = 1.15  
    STRESS_CORRELATION: float = 0.85 

    # Solution 2: Temporal Scaling (Time Horizon Sensitivity)
    DURATIONS: List[int] = field(default_factory=lambda: [12, 24, 36])

# --- 3. NUMERICAL LINEAR ALGEBRA UTILS ---
class MatrixMath:
    """Advanced Linear Algebra utilities for Financial Engineering."""

    @staticmethod
    def nearest_pd(A: np.ndarray) -> np.ndarray:
        """Higham's Algorithm (2002) for Nearest Correlation Matrix."""
        B = (A + A.T) / 2
        _, s, V = np.linalg.svd(B)
        H = np.dot(V.T, np.dot(np.diag(s), V))
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2

        if MatrixMath._is_pd(A3): return A3
        
        spacing = np.spacing(np.linalg.norm(A))
        I = np.eye(A.shape[0])
        k = 1
        while not MatrixMath._is_pd(A3):
            mineig = np.min(np.real(np.linalg.eigvals(A3)))
            A3 += I * (-mineig * k**2 + spacing)
            k += 1
        return A3

    @staticmethod
    def _is_pd(B):
        try:
            _ = np.linalg.cholesky(B)
            return True
        except np.linalg.LinAlgError:
            return False

# --- 4. RISK ENGINE (Core Logic) ---
class RiskEngine:
    
    def __init__(self, config: SimulationConfig):
        self.cfg = config
        self.rng = np.random.default_rng(self.cfg.RANDOM_SEED)
        self.cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
        if not self.cfg.DATA_PATH.exists():
            logger.critical(f"Data not found at {self.cfg.DATA_PATH}")
            sys.exit(1)
        self.df_data = pd.read_csv(self.cfg.DATA_PATH, index_col='Date', parse_dates=True)

    def get_regime_parameters(self, start_year: int, end_year: int, duration_months: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Extracts Volatility and Correlation. Scales volatility based on Duration.
        """
        df_subset = self.df_data[(self.df_data.index.year >= start_year) & (self.df_data.index.year <= end_year)]
        
        if len(df_subset) < 12:
            raise ValueError(f"Insufficient data for {start_year}-{end_year}")

        returns = np.log(df_subset / df_subset.shift(1)).dropna()
        n = len(returns.columns)

        # Solution 2 Implementation: Dynamic Volatility Scaling
        scale_factor = np.sqrt(duration_months)
        mus, stds = [], []
        
        for col in returns.columns:
            mu, std = norm.fit(returns[col])
            # Zero Drift Assumption (Methodology) -> Use 0 instead of mu
            mus.append(0.0) 
            stds.append(std * scale_factor)
            
        # Correlation Logic
        is_crisis = (2008 <= start_year <= 2012) or (2021 <= start_year <= 2024)
        
        if is_crisis:
            target_corr = self.cfg.STRESS_CORRELATION
        else:
            c = returns.corr().values
            target_corr = (np.sum(c) - n) / (n * (n - 1))
            target_corr = max(0.1, target_corr) # Floor at 0.1
            
        return np.array(mus), np.array(stds), target_corr

    def run_simulation(self, scenarios: Dict, profiles: Dict) -> pd.DataFrame:
        results = []
        col_names = self.df_data.columns
        
        for duration in self.cfg.DURATIONS:
            logger.info(f"--- Simulating Project Duration: {duration} Months ---")
            
            for sc_name, (start, end) in scenarios.items():
                try:
                    mus, stds, target_corr = self.get_regime_parameters(start, end, duration)
                    n_cols = len(mus)
                    
                    # --- A. INDEPENDENT MODEL ---
                    Z_ind = self.rng.standard_normal((self.cfg.NUM_SIMS, n_cols))
                    
                    # --- B. SYSTEMIC MODEL (Cholesky with Higham) ---
                    C = np.full((n_cols, n_cols), target_corr)
                    np.fill_diagonal(C, 1.0)
                    C_robust = MatrixMath.nearest_pd(C)
                    L = np.linalg.cholesky(C_robust)
                    Z_sys = self.rng.standard_normal((self.cfg.NUM_SIMS, n_cols)) @ L.T
                    
                    if 'Crisis' in sc_name:
                        market_factor = np.mean(Z_sys, axis=1)
                        mask = market_factor > 1.645
                        Z_sys[mask] *= self.cfg.BLACK_SWAN_FACTOR
                        
                    # --- COST CALCULATION ---
                    for prof_name, weights in profiles.items():
                        c_ind, c_sys = np.zeros(self.cfg.NUM_SIMS), np.zeros(self.cfg.NUM_SIMS)
                        
                        for i, col in enumerate(col_names):
                            if col in weights:
                                w = weights[col]
                                # GBM Approximation with Volatility Throttling logic inherent in stds
                                r_ind = Z_ind[:, i] * stds[i] + mus[i]
                                r_sys = Z_sys[:, i] * stds[i] + mus[i]
                                
                                c_ind += w * (np.exp(r_ind) - 1)
                                c_sys += w * (np.exp(r_sys) - 1)
                                
                        # Metrics (P85)
                        p85_ind = np.percentile(self.cfg.BASE_COST * (1 + c_ind), 85)
                        p85_sys = np.percentile(self.cfg.BASE_COST * (1 + c_sys), 85)
                        
                        gap = p85_sys - p85_ind
                        hidden_risk_pct = (gap / p85_ind) * 100
                        
                        results.append({
                            'Duration': duration,
                            'Period': sc_name,
                            'Profile': prof_name,
                            'Corr': f"{target_corr:.2f}",
                            'Industry P85': f"€{p85_ind/1e6:.2f}M",
                            'Quant P85': f"€{p85_sys/1e6:.2f}M",
                            'Hidden Risk': f"+{hidden_risk_pct:.1f}%",
                            'Shortfall (€)': f"€{gap:,.0f}"
                        })
                except Exception as e:
                    logger.error(f"Error processing {sc_name}: {e}")

        df = pd.DataFrame(results)
        df = df.sort_values(by=['Duration', 'Period', 'Profile'])
        
        print("\nFINAL SENSITIVITY & STRESS TEST REPORT")
        print("=========================================")
        print(df.to_string(index=False))
        
        df.to_csv(self.cfg.RESULTS_DIR / 'sensitivity_analysis.csv', index=False)
        logger.info(f"Analysis Complete. Results saved to {self.cfg.RESULTS_DIR}")
        return df

if __name__ == "__main__":
    # Define Scenarios & Profiles
    scenarios = {
        'Stable (2014-2019)': (2014, 2019),
        'Crisis (2021-2024)': (2021, 2024)
    }
    
    # --- CRITICAL UPDATE: MATCHING PAPER METHODOLOGY ---
    # Representative Project: 30% Concrete, 30% Steel, 20% Fuel, 20% PVC
    profiles = {
        'Representative Project': {
            'Concrete': 0.30, 
            'Steel': 0.30, 
            'Fuel_Energy': 0.20, 
            'PVC_Pipes': 0.20
        }
    }
    
    RiskEngine(SimulationConfig()).run_simulation(scenarios, profiles)