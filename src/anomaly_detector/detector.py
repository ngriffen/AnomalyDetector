from .checks.rare_values import check_rare_values
from .checks.null_values import check_null_values
from .checks.duplicate_rows import check_duplicate_rows
from .checks.numerical_outliers import check_numerical_outliers
from .checks.type_inconsistency import check_type_inconsistency
from .checks.logical_outliers import check_logical_outliers
from .report import AnomalyReport

class AnomalyDetector:
    def __init__(self, df, rare_threshold=5, null_threshold_pct=5.0, logical_rules=None):
        self.df = df
        self.rare_threshold = rare_threshold
        self.null_threshold_pct = null_threshold_pct
        self.logical_rules = logical_rules or {} # Pass rules here

    def run(self) -> AnomalyReport:
        findings = {
            "rare_values": check_rare_values(self.df, self.rare_threshold),
            "null_values": check_null_values(self.df, self.null_threshold_pct),
            "duplicate_rows": check_duplicate_rows(self.df),
            "numerical_outliers": check_numerical_outliers(self.df),
            "type_inconsistency": check_type_inconsistency(self.df),
            "logical_outliers": check_logical_outliers(self.df, rules=self.logical_rules)
        }
        return AnomalyReport(findings, self.df.shape)
