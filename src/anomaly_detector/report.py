from __future__ import annotations
from typing import Any
import pandas as pd

class AnomalyReport:
    def __init__(self, findings: dict[str, list[dict]], df_shape: tuple[int, int]) -> None:
        self.findings = findings
        self.df_shape = df_shape

    def summary(self) -> str:
        # Colors for the terminal/Colab
        HEADER, BLUE, YEL, END = '\033[95m', '\033[94m', '\033[93m', '\033[0m'
        BOLD = '\033[1m'

        lines = [
            f"{HEADER}{'='*60}{END}",
            f"  {BOLD}DATA QUALITY AUDIT{END}",
            f"{HEADER}{'='*60}{END}",
            f"  Dataset: {self.df_shape[0]} rows x {self.df_shape[1]} columns",
            ""
        ]

        # --- [1] Anomalies Detected (Rare Values) ---
        rv = self.findings.get("rare_values", [])
        lines.append(f"{BLUE}1] Anomalies Detected{END} ({len(rv)} flags)")
        lines.append("-" * 60)
        if not rv:
            lines.append("  ✓ No rare values detected.")
        else:
            for item in rv:
                lines.append(f"  • Col: '{item['column']}' | Value: {item['value']!r} (Count: {item['count']})")

        # --- [2] Null Values Detected ---
        nv = self.findings.get("null_values", [])
        lines.append(f"\n{BLUE}2] Null Values Detected{END} ({len(nv)} flags)")
        lines.append("-" * 60)
        if not nv:
            lines.append("  ✓ No high-null columns detected.")
        else:
            for item in nv:
                lines.append(f"  • Col: '{item['column']}' | {YEL}{item['null_pct']}% missing{END}")

        # --- [3] Duplicates Detected ---
        dv = self.findings.get("duplicate_rows", [])
        lines.append(f"\n{BLUE}3] Duplicates Detected{END}")
        lines.append("-" * 60)
        if not dv:
            lines.append("  ✓ No duplicate rows detected.")
        else:
            total = dv[0]['total_count']
            lines.append(f"  {YEL}Found {total} duplicate rows:{END}")
            
            # Show the first 10 duplicates so the report doesn't get too long
            for item in dv[:10]:
                idx = item['row_index']
                # Create a clean string of the attributes
                attr_str = ", ".join([f"{k}: {v}" for k, v in item['attributes'].items()])
                # Truncate if very long
                if len(attr_str) > 70: attr_str = attr_str[:67] + "..."
                
                lines.append(f"  • Row {idx:>4}: {attr_str}")
            
            if total > 10:
                lines.append(f"  ... and {total - 10} more duplicate rows.")

        lines.append(f"\n{HEADER}{'='*60}{END}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()
