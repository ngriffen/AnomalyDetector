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
        based on current statistical data distributions.
        """
        config = {
            "rare_threshold": max(2, int(len(df) * 0.01)), # 1% of data or min 2
            "null_threshold_pct": 10.0,
            "logical_rules": {},
            "patterns": {}
        }
        
        for col in df.columns:
            # 1. Suggest ranges for numeric columns using Percentile Clipping (5th to 95th)
            # This ensures the "Ideal" baseline ignores existing extreme outliers.
            if pd.api.types.is_numeric_dtype(df[col]):
                low, high = df[col].quantile([0.05, 0.95])
                config["logical_rules"][col] = {"min": float(low), "max": float(high)}
                
            # 2. Suggest Regex patterns for categorical lists
            elif pd.api.types.is_object_dtype(df[col]):
                # Only include values that appear more than 10% of the time
                counts = df[col].value_counts(normalize=True)
                top_tier = counts[counts > 0.10].index.tolist()
                
                # If there are 1-5 dominant categories, lock them in as the pattern
                if 0 < len(top_tier) <= 5:
                    clean_vals = [str(v).replace(r'(', r'\(').replace(r')', r'\)') for v in top_tier]
                    pattern = f"^({'|'.join(clean_vals)})$"
                    config["patterns"][col] = pattern
                    
        return config

    # --- FACTORY METHODS ---
    
    @classmethod
    def Basic(cls, df: pd.DataFrame, **kwargs) -> AnomalyDetector:
        """Runs standard core checks (Rare, Null, Duplicates, Outliers, Types, Logic)."""
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
            
        # Return the report, passing self.df so visualize() can access the data
        return AnomalyReport(self.df, findings, self.mode)
