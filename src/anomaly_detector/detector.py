"""
AnomalyDetector — the main entry point for running checks on a DataFrame.
"""

from __future__ import annotations

import pandas as pd

# These relative imports link to the files in the checks folder
from .checks.rare_values import check_rare_values
from .checks.null_values import check_null_values
from .checks.duplicate_rows import check_duplicate_rows
from .report import AnomalyReport


class AnomalyDetector:
    """
    Run a suite of anomaly checks against a pandas DataFrame.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        rare_threshold: int = 5,
        rare_max_categories: int = 50,
        null_threshold_pct: float = 5.0,
        duplicate_subset: list[str] | None = None,
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("DataFrame is empty — nothing to analyse.")

        self.df = df.copy()
        self.rare_threshold = rare_threshold
        self.rare_max_categories = rare_max_categories
        self.null_threshold_pct = null_threshold_pct
        self.duplicate_subset = duplicate_subset

    def run(self) -> AnomalyReport:
        """Execute all enabled checks and return a consolidated report."""
        findings: dict[str, list[dict]] = {}

        # 1. Rare Values
        findings["rare_values"] = check_rare_values(
            self.df,
            threshold=self.rare_threshold,
            max_categories=self.rare_max_categories,
        )

        # 2. Null Values
        findings["null_values"] = check_null_values(
            self.df,
            threshold_pct=self.null_threshold_pct,
        )

        # 3. Duplicate Rows
        findings["duplicate_rows"] = check_duplicate_rows(
            self.df,
            subset=self.duplicate_subset,
        )

        return AnomalyReport(findings=findings, df_shape=self.df.shape)
