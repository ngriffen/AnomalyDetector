"""
AnomalyDetector — the main entry point for running checks on a DataFrame.
"""

from __future__ import annotations

import pandas as pd

from .checks.rare_values import check_rare_values
from .report import AnomalyReport


class AnomalyDetector:
    """
    Run a suite of anomaly checks against a pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to analyse.
    rare_threshold : int
        Values that appear fewer than or equal to this many times are
        considered "rare" and will be flagged. Default is 5.
    rare_max_categories : int
        Only inspect columns whose total number of unique values is at
        most this number.  Very-high-cardinality columns (e.g. free-text
        IDs) would produce noise rather than signal.  Default is 50.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        rare_threshold: int = 5,
        rare_max_categories: int = 50,
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("DataFrame is empty — nothing to analyse.")

        self.df = df.copy()
        self.rare_threshold = rare_threshold
        self.rare_max_categories = rare_max_categories

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> AnomalyReport:
        """
        Execute all enabled checks and return a consolidated report.

        Returns
        -------
        AnomalyReport
        """
        findings: dict[str, list[dict]] = {}

        findings["rare_values"] = check_rare_values(
            self.df,
            threshold=self.rare_threshold,
            max_categories=self.rare_max_categories,
        )

        return AnomalyReport(findings=findings, df_shape=self.df.shape)
