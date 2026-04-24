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
            f"  {BOLD}ANOMALY REPORT{END}",
            f"{HEADER}{'='*70}{END}",
            f"  Dataset: {self.df_shape[0]:,} rows × {self.df_shape[1]:,} columns",
            ""
        ]

        # --- [1] Anomalies Detected (Rare Values) ---
        rv = self.findings.get("rare_values", [])
        lines.append(f"{BLUE}1] Anomalies Detected (Rare Values){END}")
        lines.append("-" * 70)
        if not rv:
            lines.append("  ✓ No rare values detected.")
        else:
            # Group by column for cleaner output
            by_col = {}
            for item in rv:
                by_col.setdefault(item["column"], []).append(item)
                
            for col, items in by_col.items():
                lines.append(f"  • Col: '{col}'")
                for item in items:
                    # Formatting strings to align beautifully and display proper %
                    lines.append(f"    - {item['value']!r:<15} | Count: {item['count']:<4} | {item['pct']:.2f}%")

        # --- [2] Null Values Detected ---
        nv = self.findings.get("null_values", [])
        lines.append(f"\n{BLUE}2] Null Values Detected{END}")
        lines.append("-" * 70)
        if not nv:
            lines.append("  ✓ No high null-rate columns detected.")
        else:
            for item in nv:
                lines.append(f"  • '{item['column']}': {YEL}{item['null_pct']:.1f}% missing{END} ({item['null_count']} rows)")

        # --- [3] Duplicates Detected ---
        dv = self.findings.get("duplicate_rows", [])
        lines.append(f"\n{BLUE}3] Duplicates Detected{END}")
        lines.append("-" * 70)
        if not dv:
            lines.append("  ✓ No duplicate rows detected.")
        else:
            lines.append(f"  {YEL}Found {len(dv)} distinct duplicated records:{END}")
            for i, item in enumerate(dv[:5], 1): # Show top 5 groups
                attr_str = ", ".join([f"{k}: {v}" for k, v in item['attributes'].items()])
                if len(attr_str) > 65: attr_str = attr_str[:62] + "..."
                lines.append(f"  {i}. {attr_str}")
                lines.append(f"     ↳ {item['occurrences']} occurrences at rows: {item['row_indices']}")
            if len(dv) > 5:
                lines.append(f"  ... and {len(dv) - 5} more duplicated groups.")

        # --- [4] Statistical Outliers (IQR) ---
        so = self.findings.get("numerical_outliers", [])
        lines.append(f"\n{BLUE}4] Statistical Outliers (IQR Method){END}")
        lines.append("-" * 70)
        if not so:
            lines.append("  ✓ No statistical outliers detected.")
        else:
            for item in so:
                lines.append(f"  • '{item['column']}' ({item['count']} outliers found)")
                # Format numbers with commas (e.g. 50,000)
                lines.append(f"    - Expected Normal Range: {item['bounds'][0]:,} to {item['bounds'][1]:,}")
                
                # Create a clean string of the outliers and their rows
                outlier_strs = [f"{d['val']:,} (Row {d['row']})" for d in item['details']]
                lines.append(f"    - Outliers Detected: {', '.join(outlier_strs)}")

        # --- [5] Type Inconsistency ---
        ti = self.findings.get("type_inconsistency", [])
        lines.append(f"\n{BLUE}5] Type Inconsistency{END}")
        lines.append("-" * 70)
        if not ti:
            lines.append("  ✓ All columns have consistent data types.")
        else:
            for item in ti:
                lines.append(f"  • Col: '{item['column']}' | Mixed types: {YEL}{item['types_found']}{END}")

        # --- [6] Logical Outliers ---
        lo = self.findings.get("logical_outliers", [])
        lines.append(f"\n{BLUE}6] Logical Outliers (Rule Violations){END}")
        lines.append("-" * 70)
        if not lo:
            lines.append("  ✓ No logical rule violations detected.")
        else:
            for item in lo:
                lines.append(f"  • Col: '{item['column']}' | {RED}{item['issue']}{END} (Total: {item['count']})")
                for detail in item['details'][:5]:
                    lines.append(f"    - Row {detail['row']:>4}: {detail['val']!r}")
                if item['count'] > 5:
                    lines.append(f"    ... and {item['count'] - 5} more.")
                    
        # --- [7] Executive Summary ---
        lines.append(f"\n{HEADER}--- EXECUTIVE SUMMARY ---{END}")
        
        # 1. Rare
        lines.append(f"  [1] Rare Values:  {'✓ Pass' if not rv else f'! {len(rv)}'}")
        
        # 2. Nulls
        lines.append(f"  [2] Null Values:  {'✓ Pass' if not nv else f'! {len(nv)}'}")
        
        # 3. Duplicates
        lines.append(f"  [3] Duplicates:   {'✓ Pass' if not dv else f'! {len(dv)} groups'}")
        
        # 4. Outliers (Summing the total outlier count)
        outlier_total = sum(item['count'] for item in so) if so else 0
        lines.append(f"  [4] Outliers:     {'✓ Pass' if not so else f'! {outlier_total}'}")
        
        # 5. Types
        lines.append(f"  [5] Mixed Types:  {'✓ Pass' if not ti else f'! {len(ti)} columns'}")
        
        # 6. Logical (Summing the total logical violations)
        logic_total = sum(item['count'] for item in lo) if lo else 0
        lines.append(f"  [6] Logic Rules:  {'✓ Pass' if not lo else f'! {logic_total} rule violations'}")

        lines.append(f"\n{HEADER}{'='*70}{END}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()
