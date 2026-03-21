"""
Module: 01_data_processing.py
Description: ETL (Extract, Transform, Load) pipeline for ELSTAT construction cost data.
Author: Dimitrios Chronis
Standards: PEP8, Type Hinting, Robust Error Handling, Logging
"""

import pandas as pd
import logging
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional

# --- 1. SETUP LOGGING ---
# Ρύθμιση επαγγελματικού logging για παρακολούθηση της εκτέλεσης
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(module)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- 2. CONFIGURATION ---
@dataclass
class DataConfig:
    """Configuration class containing paths and schema definitions."""
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    RAW_PATH: Path = BASE_DIR / 'data' / 'raw' / 'elstat_data.xlsx'
    PROCESSED_PATH: Path = BASE_DIR / 'data' / 'processed' / 'clean_returns.csv'
    
    # Παράμετροι ανάγνωσης Excel
    HEADER_ROW: int = 12  # Οι τίτλοι βρίσκονται στη γραμμή 13 (0-based index 12)
    
    # Χαρτογράφηση στηλών βάσει Index για ασφάλεια (Index -> Internal Name)
    COLUMN_MAP: Dict[int, str] = None

    def __post_init__(self):
        # Ορίζουμε το mapping εδώ (immutable default argument workaround)
        if self.COLUMN_MAP is None:
            self.COLUMN_MAP = {
                0: 'Date',
                1: 'General_Index',
                2: 'Concrete',     # Κονίες
                7: 'Steel',        # Μεταλλικά
                16: 'Fuel_Energy', # Καύσιμα/Ενέργεια
                8: 'PVC_Pipes'     # Σωλήνες
            }

# --- 3. ETL PIPELINE CLASS ---
class DataIngestionPipeline:
    """
    Handles the ingestion, cleaning, and storage of financial time series data.
    """
    
    def __init__(self, config: DataConfig):
        self.cfg = config

    def run(self):
        """Executes the full pipeline lifecycle."""
        logger.info("Starting Data Ingestion Pipeline...")
        
        # Step 1: Extract
        df_raw = self._extract()
        if df_raw is None:
            logger.critical("Pipeline failed during extraction.")
            sys.exit(1)

        # Step 2: Transform
        df_clean = self._transform(df_raw)
        
        # Step 3: Load
        self._load(df_clean)
        logger.info("Pipeline completed successfully.")

    def _extract(self) -> Optional[pd.DataFrame]:
        """Reads the raw Excel file."""
        if not self.cfg.RAW_PATH.exists():
            logger.error(f"File not found at: {self.cfg.RAW_PATH}")
            return None
        
        try:
            logger.info(f"Reading Excel: {self.cfg.RAW_PATH.name}")
            # Χρήση openpyxl engine για .xlsx
            df = pd.read_excel(
                self.cfg.RAW_PATH, 
                skiprows=self.cfg.HEADER_ROW, 
                engine='openpyxl'
            )
            return df
        except Exception as e:
            logger.error(f"Failed to read Excel: {e}")
            logger.info("Hint: Ensure 'openpyxl' is installed (pip install openpyxl)")
            return None

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans, formats, and validates the data."""
        logger.info("Transforming raw data...")
        
        # 1. Rename Columns safely using indices
        try:
            # Δημιουργία λεξικού με τα πραγματικά ονόματα στηλών από το DataFrame
            rename_dict = {df.columns[idx]: name for idx, name in self.cfg.COLUMN_MAP.items()}
            df = df.rename(columns=rename_dict)
        except IndexError as e:
            logger.critical(f"Column index mismatch. Check Excel structure. Error: {e}")
            sys.exit(1)

        # 2. Date Parsing & Indexing
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date']).set_index('Date').sort_index()

        # 3. Filter Study Period (2000-2024)
        # Χρήση string slicing για robust datetime indexing
        try:
            df = df.loc['2000-01-01':'2024-12-31'].copy()
        except KeyError:
            logger.warning("Date range 2000-2024 partially or fully missing.")

        # 4. Select Target Columns
        target_cols = list(self.cfg.COLUMN_MAP.values())
        target_cols.remove('Date') # Date is index now
        
        # Keep only columns that exist (in case of rename failure)
        df = df[[c for c in target_cols if c in df.columns]]

        # 5. Numeric Conversion (Robust)
        for col in df.columns:
            if df[col].dtype == object:
                # Αντικατάσταση κόμματος με τελεία και καθαρισμός κενών
                df[col] = df[col].astype(str).str.replace(',', '.').str.strip()
            
            # Μετατροπή σε float, λάθη γίνονται NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 6. Drop NaN rows (Strict Quality Control for Copula)
        initial_rows = len(df)
        df = df.dropna()
        dropped = initial_rows - len(df)
        
        if dropped > 0:
            logger.warning(f"Dropped {dropped} rows containing NaN values.")
        
        logger.info(f"Data shape after cleaning: {df.shape}")
        return df

    def _load(self, df: pd.DataFrame):
        """Saves the processed data to CSV."""
        try:
            self.cfg.PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.cfg.PROCESSED_PATH)
            
            logger.info(f"Processed data saved to: {self.cfg.PROCESSED_PATH}")
            logger.info(f"Data Range: {df.index.min().date()} to {df.index.max().date()}")
            
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")
            sys.exit(1)

# --- 4. EXECUTION ENTRY POINT ---
if __name__ == "__main__":
    # Initialize Config
    config = DataConfig()
    
    # Run Pipeline
    pipeline = DataIngestionPipeline(config)
    pipeline.run()