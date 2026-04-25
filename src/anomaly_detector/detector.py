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

    # --- FACTORY METHODS ---
    
    @classmethod
    def Basic(cls, df: pd.DataFrame, **kwargs) -> AnomalyDetector:
        """Runs the standard 6 checks (Rare, Null, Duplicates, Outliers, Types, Logic)."""
        return cls(df, mode='basic', kwargs=kwargs)

    @classmethod
    def Auto(cls, df: pd.DataFrame, **kwargs) -> AnomalyDetector:
        """Runs unsupervised ML to find weird combinations of data."""
        return cls(df, mode='auto', kwargs=kwargs)

    @classmethod
    def Full(cls, df: pd.DataFrame, **kwargs) -> AnomalyDetector:
        """Runs the 6 Basic checks AND the Auto ML checks."""
        return cls(df, mode='full', kwargs=kwargs)

    def run(self) -> AnomalyReport:
        findings = {}
        
        # 1. Run Basic Checks (The Core 6)
        if self.mode in ['basic', 'full']:
            findings["rare_values"] = check_rare_values(self.df, self.config.get('rare_threshold', 5))
            findings["null_values"] = check_null_values(self.df, self.config.get('null_threshold_pct', 5.0))
            findings["duplicate_rows"] = check_duplicate_rows(self.df)
            findings["numerical_outliers"] = check_numerical_outliers(self.df)
            findings["type_inconsistency"] = check_type_inconsistency(self.df)
            findings["logical_outliers"] = check_logical_outliers(self.df, self.config.get('logical_rules', {}))
            
        # 2. Run Auto Checks
        if self.mode in ['auto', 'full']:
            findings["auto_multivariate"] = check_auto_multivariate(self.df, self.config.get('contamination', 0.02))
            
        return AnomalyReport(findings, self.df.shape, self.mode)
