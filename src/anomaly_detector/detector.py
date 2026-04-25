from __future__ import annotations
import pandas as pd

from .checks.rare_values import check_rare_values
from .checks.null_values import check_null_values
from .checks.duplicate_rows import check_duplicate_rows
from .checks.numerical_outliers import check_numerical_outliers
from .checks.type_inconsistency import check_type_inconsistency
from .checks.logical_outliers import check_logical_outliers
from .checks.auto_multivariate import check_auto_multivariate
from .report import AnomalyReport

class AnomalyDetector:
    def __init__(self, df: pd.DataFrame, mode: str = 'basic', kwargs: dict = None) -> None:
        if df.empty:
            raise ValueError("DataFrame is empty — nothing to analyse.")
        
        self.df = df.copy()
        self.mode = mode.lower()
        self.config = kwargs or {}

    # --- CONFIGURATION GENERATOR ---

    @staticmethod
    def suggest_config(df: pd.DataFrame) -> dict:
        """
        Analyzes the dataframe to suggest a starting configuration
        based on current data distributions.
        """
        config = {
            "rare_threshold": max(2, int(len(df) * 0.01)), # 1% of data or min 2
            "null_threshold_pct": 10.0,
            "logical_rules": {},
            "patterns": {}
        }
        
        for col in df.columns:
            # 1. Suggest ranges for numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                # Using float() to ensure it serializes nicely if you export to JSON later
                col_min = float(df[col].min())
                col_max = float(df[col].max())
                config["logical_rules"][col] = {"min": col_min, "max": col_max}
                
            # 2. Suggest Regex patterns for small categorical lists
            elif pd.api.types.is_object_dtype(df[col]):
                unique_vals = df[col].dropna().unique()
                # If there are 5 or fewer unique strings, it's likely a strict category
                if len(unique_vals) <= 5:
                    # Escape strings just in case, and format into a regex OR group
                    clean_vals = [str(v).replace(r'(', r'\(').replace(r')', r'\)') for v in unique_vals]
                    pattern = f"^({'|'.join(clean_vals)})$"
                    config["patterns"][col] = pattern
                    
        return config

    # --- FACTORY METHODS ---
    
    @classmethod
    def Basic(cls, df: pd.DataFrame, **kwargs) -> AnomalyDetector:
        """Runs the standard core checks (Rare, Null, Duplicates, Outliers, Types, Logic)."""
        return cls(df, mode='basic', kwargs=kwargs)

    @classmethod
    def Auto(cls, df: pd.DataFrame, **kwargs) -> AnomalyDetector:
        """Runs unsupervised ML to find weird combinations of data."""
        return cls(df, mode='auto', kwargs=kwargs)

    @classmethod
    def Full(cls, df: pd.DataFrame, **kwargs) -> AnomalyDetector:
        """Runs the Basic checks AND the Auto ML checks."""
        return cls(df, mode='full', kwargs=kwargs)

    def run(self) -> AnomalyReport:
        findings = {}
        
        # 1. Run Basic Checks
        if self.mode in ['basic', 'full']:
            findings["rare_values"] = check_rare_values(self.df, self.config.get('rare_threshold', 5))
            findings["null_values"] = check_null_values(self.df, self.config.get('null_threshold_pct', 5.0))
            findings["duplicate_rows"] = check_duplicate_rows(self.df)
            findings["numerical_outliers"] = check_numerical_outliers(self.df)
            findings["type_inconsistency"] = check_type_inconsistency(self.df)
            findings["logical_outliers"] = check_logical_outliers(self.df, self.config.get('logical_rules', {}))
            
        # 2. Run Auto Checks
        if self.mode in ['auto', 'full']:
            # The contamination parameter can also be overridden via kwargs
            findings["auto_multivariate"] = check_auto_multivariate(self.df, self.config.get('contamination', 0.02))
            
        return AnomalyReport(findings, self.df.shape, self.mode)
