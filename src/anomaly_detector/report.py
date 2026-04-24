from __future__ import annotations
from typing import Any
import pandas as pd

class AnomalyReport:
    def __init__(self, findings: dict[str, list[dict]], df_shape: tuple[int, int]) -> None:
        self.findings = findings
        self.df_shape = df_shape

    def summary(self) -> str:
        HEADER, BLUE, YEL, RED, END = '\033[95m', '\033[94m', '\033[93m', '\033[91m', '\033[0m'
        BOLD = '\033[1m'

        lines = [
            f"{HEADER}{'='*70}{END}",
            f"  {BOLD}DATA QUALITY AUDIT REPORT{END}",
            f"{HEADER}{'='*70}{END}",
            f"  Dataset: {self.df_shape[0]:,} rows × {self.df_shape[1]:,} columns",
            ""
        ]

        # Sections 1-5 (Briefly outlined for the full file)
        self._add_section(lines, "1] Anomalies Detected (Rare)", "rare_values", BLUE, END)
        self._add_section(lines, "2] Null Values Detected", "null_values", BLUE, END)
        self._add_section(lines, "3] Duplicates Detected", "duplicate_rows", BLUE, END)
        self._add_section(lines, "4] Statistical Outliers (IQR)", "numerical_outliers", BLUE, END)
        self._add_section(lines, "5] Type Inconsistency", "type_inconsistency", BLUE, END)

        # --- [6] Logical Outliers (Detailed Version) ---
        lo = self.findings.get("logical_outliers", [])
        lines.append(f"\n{BLUE}6] Logical Outliers (Rule Violations){END}")
        lines.append("-" * 70)
        if not lo:
            lines.append("  ✓ No logical rule violations detected.")
        else:
            for item in lo:
                lines.append(f"  • Col: '{item['column']}' | {RED}{item['issue']}{END} (Total: {item['count']})")
                for detail in item['details']:
                    lines.append(f"    - Row {detail['row']:>4}: {detail['val']!r}")
                
                if item['count'] > 10:
                    lines.append(f"    ... and {item['count'] - 10} more.")

        lines.append(f"\n{HEADER}{'='*70}{END}")
        return "\n".join(lines)

    def _add_section(self, lines, title, key, color, end):
        """Helper to keep the summary clean."""
        data = self.findings.get(key, [])
        lines.append(f"\n{color}{title}{end}")
        lines.append("-" * 70)
        if not data:
            lines.append("  ✓ None")
        else:
            # General formatting for other sections
            for item in data[:5]:
                lines.append(f"  • {str(item)}")

    def __str__(self) -> str:
        return self.summary()
