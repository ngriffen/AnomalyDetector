from __future__ import annotations
import pandas as pd
from .checks.rare_values import check_rare_values
from .checks.null_values import check_null_values
from .checks.duplicate_rows import check_duplicate_rows
from .report import AnomalyReport

class AnomalyDetector:
    def __init__(self, df, rare_threshold=5, null_threshold_pct=5.0):
        self.df = df
        self.rare_threshold = rare_threshold
        self.null_threshold_pct = null_threshold_pct

    def run(self) -> AnomalyReport:
        findings = {
            "rare_values": check_rare_values(self.df, threshold=self.rare_threshold),
            "null_values": check_null_values(self.df, threshold_pct=self.null_threshold_pct),
            "duplicate_rows": check_duplicate_rows(self.df)
        }
        return AnomalyReport(findings, self.df.shape)
