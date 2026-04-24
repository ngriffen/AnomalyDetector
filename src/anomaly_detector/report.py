"""
AnomalyReport — Dynamic reporting engine for StatGuard.
Supports Rare Values, Nulls, Duplicates (with row detail), 
Outliers, Type Inconsistency, and Logical Rules.
"""

from __future__ import annotations
from typing import Any
import pandas as pd

class AnomalyReport:
    """Container for anomaly-detection findings with pretty-printing."""

    def __init__(
        self,
        findings: dict[str, list[dict]],
        df_shape: tuple[int, int],
    ) -> None:
        self.findings = findings
        self.df_shape = df_shape

    def summary(self) -> str:
        """Returns a human-readable text summary with color coding."""
        # Color codes for Colab/Terminal
        HEADER = '\033[95m'
        BLUE = '\033[94m'
        YEL = '\033[93m'
        RED = '\033[91m'
        END = '\033[0m'
        BOLD = '\033[1m'

        rows, cols = self.df_shape
        lines = [
            f"{HEADER}{'='*70}{END}",
            f"  {BOLD}DATA QUALITY AUDIT REPORT{END}",
            f"{HEADER}{'='*70}{END}",
            f"  Dataset: {rows:,} rows × {cols:,} columns",
            ""
        ]

        # --- [1] Anomalies Detected (Rare Values) ---
        rv = self.findings.get("rare_values", [])
        lines.append(f"{BLUE}1] Anomalies Detected (Rare Values){END}")
        lines.append("-" * 70)
        if not rv:
            lines.append("  ✓ No rare values detected.")
        else:
            for item in rv:
                lines.append(f"  • Col: '{item['column']}' | Value: {item['value']!r:<15} (Count: {item['count']})")

        # --- [2] Null Values Detected ---
        nv = self.findings.get("null_values", [])
        lines.append(f"\n{BLUE}2] Null Values Detected{END}")
        lines.append("-" * 70)
        if not nv:
            lines.append("  ✓ No high null-rate columns detected.")
        else:
            for item in nv:
                lines.append(f"  • Col: '{item['column']}' -> {YEL}{item['null_pct']}% missing{END}")

        # --- [3] Duplicates Detected ---
        dv = self.findings.get("duplicate_rows", [])
        lines.append(f"\n{BLUE}3] Duplicates Detected{END}")
        lines.append("-" * 70)
        if not dv:
            lines.append("  ✓ No duplicate rows detected.")
        else:
            # Group by total count (stored in first item)
            total_dupes = len(dv)
            lines.append(f"  {YEL}Found {total_dupes} duplicate rows:{END}")
            for item in dv[:10]: # Show first 10
                idx = item['row_index']
                attr_str = ", ".join([f"{k}: {v}" for k, v in item['attributes'].items()])
                if len(attr_str) > 80: attr_str = attr_str[:77] + "..."
                lines.append(f"    • Row {idx:>4}: {attr_str}")
            if total_dupes > 10:
                lines.append(f"    ... and {total_dupes - 10} more.")

        # --- [4] Statistical Outliers ---
        so = self.findings.get("numerical_outliers", [])
        lines.append(f"\n{BLUE}4] Statistical Outliers (IQR Method){END}")
        lines.append("-" * 70)
        if not so:
            lines.append("  ✓ No statistical outliers detected.")
        else:
            for item in so:
                lines.append(f"  • Col: '{item['column']}' | Found {item['count']} outliers")
                lines.append(f"    Normal Range: {item['bounds']} | Samples: {item['samples']}")

        # --- [5] Type Inconsistency ---
        ti = self.findings.get("type_inconsistency", [])
        lines.append(f"\n{BLUE}5] Type Inconsistency{END}")
        lines.append("-" * 70)
        if not ti:
            lines.append("  ✓ All columns have consistent data types.")
        else:
            for item in ti:
                lines.append(f"  • Col: '{item['column']}' | Mixed types: {YEL}{item['types_found']}{END}")

        # --- [6] Logical Outliers (Scalable) ---
        lo = self.findings.get("logical_outliers", [])
        lines.append(f"\n{BLUE}6] Logical Outliers (Rule Violations){END}")
        lines.append("-" * 70)
        if not lo:
            lines.append("  ✓ No logical rule violations detected.")
        else:
            for item in lo:
                lines.append(f"  • Col: '{item['column']}' | {RED}{item['issue']}{END}")
                lines.append(f"    Violations: {item['count']} | Sample Indices: {item['sample_indices']}")

        lines.append(f"\n{HEADER}{'='*70}{END}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()
