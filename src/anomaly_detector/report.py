from __future__ import annotations
from typing import Any
import pandas as pd

class AnomalyReport:
    def __init__(self, findings: dict[str, list[dict]], df_shape: tuple[int, int], mode: str) -> None:
        self.findings = findings
        self.df_shape = df_shape
        self.mode = mode

    def summary(self) -> str:
        # Formatting Codes
        HEADER, BLUE, YEL, RED, END = '\033[95m', '\033[94m', '\033[93m', '\033[91m', '\033[0m'
        BOLD = '\033[1m'

        lines = [
            f"{HEADER}{'='*80}{END}",
            f"  {BOLD}DATA QUALITY AUDIT REPORT [{self.mode.upper()} MODE]{END}",
            f"{HEADER}{'='*80}{END}",
            f"  Dataset: {self.df_shape[0]:,} rows × {self.df_shape[1]:,} columns",
            ""
        ]

        # --- [AUTO] Unsupervised Section ---
        if self.mode in ['auto', 'full']:
            am = self.findings.get('auto_multivariate', [])
            lines.append(f"{BLUE}[AUTO] Unsupervised Multivariate Anomalies{END}")
            lines.append("-" * 80)
            if not am:
                lines.append("  ✓ No multivariate anomalies detected.")
            elif 'error' in am[0]:
                lines.append(f"  {RED}Error: {am[0]['error']}{END}")
            else:
                for item in am:
                    lines.append(f"  {RED}! Detected {item['count']} rows with highly anomalous feature combinations.{END}")
                    for detail in item.get('details', [])[:3]:
                        row_str = ", ".join([f"{k}: {v}" for k, v in detail['val'].items()])
                        if len(row_str) > 70: row_str = row_str[:67] + "..."
                        lines.append(f"    - Row {detail['row']:>4}: {row_str}")

        # --- BASIC Detailed Sections ---
        if self.mode in ['basic', 'full']:
            # [1] Rare Values
            rv = self.findings.get('rare_values', [])
            lines.append(f"\n{BLUE}1] Anomalies Detected (Rare Values){END}")
            lines.append("-" * 80)
            if not rv:
                lines.append("  ✓ No rare values detected.")
            else:
                by_col = {}
                for item in rv:
                    by_col.setdefault(item['column'], []).append(item)
                for col, items in by_col.items():
                    lines.append(f"  • Col: '{col}'")
                    for item in items:
                        lines.append(f"    - {str(item['value']):<15} | Count: {item['count']:<4} | {item['pct']:.2f}%")

            # [2] Null Values
            nv = self.findings.get('null_values', [])
            lines.append(f"\n{BLUE}2] Null Values Detected{END}")
            lines.append("-" * 80)
            if not nv:
                lines.append("  ✓ No high null-rate columns detected.")
            else:
                for item in nv:
                    lines.append(f"  • '{item['column']}': {YEL}{item['null_pct']:.1f}% missing{END} ({item['null_count']} rows)")

            # [3] Duplicates
            dv = self.findings.get('duplicate_rows', [])
            lines.append(f"\n{BLUE}3] Duplicates Detected{END}")
            lines.append("-" * 80)
            if not dv:
                lines.append("  ✓ No duplicate rows detected.")
            else:
                lines.append(f"  {YEL}Found {len(dv)} distinct duplicated records:{END}")
                for i, item in enumerate(dv[:5], 1):
                    attr_str = ", ".join([f"{k}: {v}" for k, v in item['attributes'].items()])
                    if len(attr_str) > 70: attr_str = attr_str[:67] + "..."
                    lines.append(f"  {i}. {attr_str}")
                    lines.append(f"     ↳ {item['occurrences']} occurrences at rows: {item['row_indices']}")

            # [4] Statistical Outliers
            so = self.findings.get('numerical_outliers', [])
            lines.append(f"\n{BLUE}4] Statistical Outliers (IQR Method){END}")
            lines.append("-" * 80)
            if not so:
                lines.append("  ✓ No statistical outliers detected.")
            else:
                for item in so:
                    lines.append(f"  • '{item['column']}' ({item['count']} outliers)")
                    lines.append(f"    - Expected Range: {item['bounds'][0]:,} to {item['bounds'][1]:,}")
                    outlier_strs = [f"{d['val']:,} (Row {d['row']})" for d in item['details'][:5]]
                    lines.append(f"    - Sample: {', '.join(outlier_strs)}")

            # [5] Type Inconsistency
            ti = self.findings.get('type_inconsistency', [])
            lines.append(f"\n{BLUE}5] Type Inconsistency{END}")
            lines.append("-" * 80)
            if not ti:
                lines.append("  ✓ All columns have consistent data types.")
            else:
                for item in ti:
                    lines.append(f"  • Col: '{item['column']}' | Mixed types: {YEL}{item['types_found']}{END}")

            # [6] Logical Outliers
            lo = self.findings.get('logical_outliers', [])
            lines.append(f"\n{BLUE}6] Logical Outliers (Rule Violations){END}")
            lines.append("-" * 80)
            if not lo:
                lines.append("  ✓ No logical rule violations detected.")
            else:
                for item in lo:
                    lines.append(f"  • Col: '{item['column']}' | {RED}{item['issue']}{END} ({item['count']} total)")
                    for detail in item['details'][:5]:
                        lines.append(f"    - Row {detail['row']:>4}: {detail['val']!r}")

            # [7] Pattern Violations
            pv = self.findings.get('pattern_violations', [])
            lines.append(f"\n{BLUE}7] Pattern Violations (Regex){END}")
            lines.append("-" * 80)
            if not pv:
                lines.append("  ✓ All patterns matched.")
            else:
                for item in pv:
                    lines.append(f"  • '{item['column']}' | {RED}{item['issue']}{END} ({item['count']} errors)")
                    for detail in item['details'][:5]:
                        lines.append(f"    - Row {detail['row']:>4}: {detail['val']!r}")

        # --- ANOMALY SUMMARY ---
        lines.append(f"\n{HEADER}--- ANOMALY REPORT ---{END}")
        f = self.findings

        if self.mode in ['basic', 'full']:
            lines.append(f"  [1] Rare Values:    {'✓ Pass' if not f.get('rare_values') else f'! {len(f.get('rare_values', [])):<3} | Anomalies'}")
            lines.append(f"  [2] Null Values:    {'✓ Pass' if not f.get('null_values') else f'! {len(f.get('null_values', [])):<3} | Null Columns'}")
            lines.append(f"  [3] Duplicates:     {'✓ Pass' if not f.get('duplicate_rows') else f'! {len(f.get('duplicate_rows', [])):<3} | Duplicated Groups'}")
            
            out_cnt = sum(item['count'] for item in f.get('numerical_outliers', []))
            lines.append(f"  [4] Outliers:       {'✓ Pass' if not f.get('numerical_outliers') else f'! {out_cnt:<3} | Statistical Outliers'}")
            lines.append(f"  [5] Mixed Types:    {'✓ Pass' if not f.get('type_inconsistency') else f'! {len(f.get('type_inconsistency', [])):<3} | Mixed Type Columns'}")
            
            log_cnt = sum(item['count'] for item in f.get('logical_outliers', []))
            lines.append(f"  [6] Logic Rules:    {'✓ Pass' if not f.get('logical_outliers') else f'! {log_cnt:<3} | Rule Violations'}")
            
            pat_cnt = sum(item['count'] for item in f.get('pattern_violations', []))
            lines.append(f"  [7] Regex Patterns: {'✓ Pass' if not f.get('pattern_violations') else f'! {pat_cnt:<3} | Pattern Mismatches'}")

        if self.mode in ['auto', 'full']:
            am_data = f.get('auto_multivariate', [])
            auto_cnt = am_data[0]['count'] if am_data and 'count' in am_data[0] else 0
            lines.append(f"  [AUTO] Machine ML:  {'✓ Pass' if not am_data else f'! {auto_cnt:<3} | Multivariate Anomalies'}")

        lines.append(f"{HEADER}{'='*80}{END}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()
