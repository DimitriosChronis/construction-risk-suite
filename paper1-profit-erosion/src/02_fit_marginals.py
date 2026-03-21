"""
Module: 02_fit_marginals.py
Description: Statistical analysis of asset returns. Calculates volatility,
             skewness, kurtosis, and performs normality tests to justify
             the use of Copulas.
Author: Dimitrios Chronis
"""

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
from dataclasses import dataclass
from scipy.stats import norm, skew, kurtosis, jarque_bera

# --- 1. LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(module)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- 2. CONFIGURATION ---
@dataclass
class StatsConfig:
    """Configuration paths for statistical analysis."""
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_PATH: Path = BASE_DIR / 'data' / 'processed' / 'clean_returns.csv'
    RESULTS_DIR: Path = BASE_DIR / 'results' / 'tables'
    OUTPUT_FILE: str = 'descriptive_stats.csv'

# --- 3. ANALYZER CLASS ---
class MarginalAnalyzer:
    """
    Performs univariate statistical analysis on time series data.
    Focuses on identifying 'Fat Tails' (Kurtosis) and Non-Normality.
    """

    def __init__(self, config: StatsConfig):
        self.cfg = config

    def run(self):
        """Main execution flow."""
        logger.info("Starting Marginal Distribution Analysis...")
        
        # Load
        df_returns = self._load_and_transform_data()
        if df_returns is None:
            sys.exit(1)

        # Analyze
        stats_df = self._calculate_metrics(df_returns)
        
        # Save
        self._save_results(stats_df)
        logger.info("Analysis completed successfully.")

    def _load_and_transform_data(self) -> pd.DataFrame:
        """Loads price data and converts to Log-Returns."""
        if not self.cfg.DATA_PATH.exists():
            logger.error(f"Data file not found: {self.cfg.DATA_PATH}")
            return None

        try:
            logger.info("Loading price data...")
            df = pd.read_csv(self.cfg.DATA_PATH, index_col='Date', parse_dates=True)
            
            # Calculate Log Returns: ln(P_t / P_t-1)
            # This is standard in Quant Finance for time-additivity
            returns = np.log(df / df.shift(1)).dropna()
            
            logger.info(f"Computed returns for {len(returns)} periods.")
            return returns
            
        except Exception as e:
            logger.critical(f"Failed to load/transform data: {e}")
            return None

    def _calculate_metrics(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculates comprehensive risk metrics for each asset."""
        logger.info("Calculating risk metrics (Mean, Vol, Skew, Kurt, JB-Test)...")
        
        stats = []
        for col in returns.columns:
            series = returns[col]
            
            # 1. Basic Moments
            mu, std = norm.fit(series)
            
            # 2. Higher Moments (The "Why Copula?" justification)
            sk = skew(series)
            kt = kurtosis(series) # Excess Kurtosis (Normal = 0)
            
            # 3. Normality Test (Jarque-Bera)
            # H0: Data is Normal. If p < 0.05, reject H0 (Data is NOT Normal)
            jb_stat, jb_p = jarque_bera(series)
            normality = "REJECTED" if jb_p < 0.05 else "Accepted"

            stats.append({
                'Material': col,
                'Monthly Mean': f"{mu:.4f}",
                'Annual Volatility': f"{std * np.sqrt(12):.2%}", # Annualized
                'Skewness': f"{sk:.2f}",
                'Excess Kurtosis': f"{kt:.2f}",  # High value (>1) implies Fat Tails
                'Min Return (Max Loss)': f"{series.min():.2%}",
                'Normality (p<0.05)': normality
            })
            
        return pd.DataFrame(stats)

    def _save_results(self, df: pd.DataFrame):
        """Saves the statistics table."""
        try:
            self.cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            out_path = self.cfg.RESULTS_DIR / self.cfg.OUTPUT_FILE
            
            df.to_csv(out_path, index=False)
            logger.info(f"Statistics table saved to: {out_path}")
            
            # Print table to console for quick verify
            print("\nASSET RISK PROFILE")
            print("=====================")
            print(df.to_string(index=False))
            print("=====================\n")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

# --- 4. EXECUTION ---
if __name__ == "__main__":
    config = StatsConfig()
    analyzer = MarginalAnalyzer(config)
    analyzer.run()