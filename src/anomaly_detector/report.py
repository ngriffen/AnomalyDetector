from __future__ import annotations
import pandas as pd

class AnomalyReport:
    def __init__(self, findings: dict[str, list[dict]], df_shape: tuple[int, int], mode: str) -> None:
        self.findings = findings
        self.df_shape = df_shape
        self.mode = mode

    def summary(self) -> str:
        HEADER, BLUE, YEL, RED, END, BOLD = '\033[95m', '\033[94m', '\033[93m', '\033[91m', '\033[0m', '\033[1m'

        lines = [
            f"{HEADER}{'='*80}{END}",
            f"  {BOLD}DATA QUALITY AUDIT REPORT [{self.mode.upper()} MODE]{END}",
            f"{HEADER}{'='*80}{END}",
            f"  Dataset: {self.df_shape[0]:,} rows × {self.df_shape[1]:,} columns",
            ""
        ]

        # --- AUTO DETECTOR SECTION ---
        if self.mode in ['auto', 'full']:
            am = self.findings.get("auto_multivariate", [])
            lines.append(f"\n{BLUE}[AUTO] Unsupervised Multivariate Anomalies{END}")
            lines.append("-" * 80)
            
            if not am:
                lines.append("  ✓ No multivariate anomalies detected.")
            elif "error" in am[0]:
                lines.append(f"  {RED}Error: {am[0]['error']}{END}")
            else:
                for item in am:
                    lines.append(f"  {RED}! Detected {item['count']} rows with highly anomalous feature combinations.{END}")
                    for detail in item['details'][:3]:
                        # Format the row dictionary cleanly
                        row_str = ", ".join([f"{k}: {v}" for k, v in detail['val'].items()])
                        lines.append(f"    - Row {detail['row']:>4}: {row_str}")

        # --- BASIC DETECTOR SECTIONS (Summarized for brevity here, keep your existing logic) ---
        if self.mode in ['basic', 'full']:
            # ... (Insert your existing detailed print logic for sections 1 through 7 here) ...
            lines.append(f"\n  {BLUE}(Basic Detailed Output Handled Here){END}")

       # --- EXECUTIVE SUMMARY ---
        lines.append(f"\n{HEADER}--- ANOMALY SUMMARY ---{END}")
        
        f = self.findings # Short alias to keep lines readable

        if self.mode in ['basic', 'full']:
            # Use ' (single quotes) inside the { } to avoid SyntaxErrors
            lines.append(f"  [1] Rare Values:    {'✓ Pass' if not f.get('rare_values') else f'! {len(f.get(\"rare_values\", [])):<3} | Anomalies'}")
            lines.append(f"  [2] Null Values:    {'✓ Pass' if not f.get('null_values') else f'! {len(f.get(\"null_values\", [])):<3} | Null Columns'}")
            lines.append(f"  [3] Duplicates:     {'✓ Pass' if not f.get('duplicate_rows') else f'! {len(f.get(\"duplicate_rows\", [])):<3} | Duplicated Groups'}")
            
            outlier_total = sum(item['count'] for item in f.get('numerical_outliers', []))
            lines.append(f"  [4] Outliers:       {'✓ Pass' if not f.get('numerical_outliers') else f'! {outlier_total:<3} | Statistical Outliers'}")
            
            lines.append(f"  [5] Mixed Types:    {'✓ Pass' if not f.get('type_inconsistency') else f'! {len(f.get(\"type_inconsistency\", [])):<3} | Mixed Type Columns'}")
            
            logic_total = sum(item['count'] for item in f.get('logical_outliers', []))
            lines.append(f"  [6] Logic Rules:    {'✓ Pass' if not f.get('logical_outliers') else f'! {logic_total:<3} | Rule Violations'}")
            
            pattern_total = sum(item['count'] for item in f.get('pattern_violations', []))
            lines.append(f"  [7] Regex Patterns: {'✓ Pass' if not f.get('pattern_violations') else f'! {pattern_total:<3} | Pattern Mismatches'}")

        if self.mode in ['auto', 'full']:
            am = f.get('auto_multivariate', [])
            auto_count = am[0]['count'] if am and 'count' in am[0] else 0
            lines.append(f"  [AUTO] Machine ML:  {'✓ Pass' if not am else f'! {auto_count:<3} | Multivariate Anomalies'}")
                      
        lines.append(f"{HEADER}{'='*80}{END}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()
