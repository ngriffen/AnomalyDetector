from __future__ import annotations
import pandas as pd

class AnomalyReport:
    def __init__(self, findings: dict[str, list[dict]], df_shape: tuple[int, int], mode: str) -> None:
        self.findings = findings
        self.df_shape = df_shape
        self.mode = mode

    def summary(self) -> str:
        HEADER, BLUE, YEL, RED, END = '\033[95m', '\033[94m', '\033[93m', '\033[91m', '\033[0m'
        BOLD = '\033[1m'

        lines = [
            f"{HEADER}{'='*80}{END}",
            f"  {BOLD} ANOMALY REPORT [{self.mode.upper()} MODE]{END}",
            f"{HEADER}{'='*80}{END}",
            f"  Dataset: {self.df_shape[0]:,} rows × {self.df_shape[1]:,} columns",
            ""
        ]

        # --- [AUTO] Section ---
        if self.mode in ['auto', 'full']:
            am = self.findings.get('auto_multivariate', [])
            lines.append(f"\n{BLUE}[AUTO] Unsupervised Multivariate Anomalies{END}")
            lines.append("-" * 80)
            if not am:
                lines.append("  ✓ No multivariate anomalies detected.")
            else:
                for item in am:
                    lines.append(f"  {RED}! Detected {item['count']} rows with anomalous combinations.{END}")
                    for detail in item.get('details', []):
                        suspects = detail.get('suspects', [])
                        tag = detail.get('tag', 'Unknown')
                        
                        # Formatting the suspect string
                        if tag == "Multivariate Mix":
                            sus_label = f"{YEL}[Complex Interaction: {', '.join(suspects)}]{END}"
                        else:
                            sus_label = f"{RED}[Suspect: {', '.join(suspects)}]{END}"
                        
                        row_str = ", ".join([f"{k}: {v}" for k, v in detail['val'].items()])
                        lines.append(f"    - Row {detail['row']:>4}: {sus_label}")
                        lines.append(f"               Data: {row_str}")

        # --- BASIC Sections (1-6) ---
        if self.mode in ['basic', 'full']:
            # 1. Rare
            rv = self.findings.get('rare_values', [])
            lines.append(f"\n{BLUE}1] Anomalies Detected (Rare Values){END}")
            lines.append("-" * 80)
            if not rv: lines.append("  ✓ No rare values detected.")
            else:
                for item in rv:
                    lines.append(f"    - Col: '{item['column']}' | Value: {str(item['value']):<10} | Count: {item['count']} ({item['pct']:.2f}%)")

            # 2. Nulls
            nv = self.findings.get('null_values', [])
            lines.append(f"\n{BLUE}2] Null Values Detected{END}")
            lines.append("-" * 80)
            if not nv: lines.append("  ✓ No high null-rate columns.")
            else:
                for item in nv:
                    lines.append(f"  • '{item['column']}': {YEL}{item['null_pct']:.1f}% missing{END}")

            # 3. Duplicates
            dv = self.findings.get('duplicate_rows', [])
            lines.append(f"\n{BLUE}3] Duplicates Detected{END}")
            lines.append("-" * 80)
            if not dv:
                lines.append("  ✓ No duplicate rows detected.")
            else:
                lines.append(f"  {YEL}Found {len(dv)} groups of identical rows:{END}")
                for item in dv:
                    # item['attributes'] contains the actual data values that were duplicated
                    attr_str = ", ".join([f"{k}: {v}" for k, v in item['attributes'].items()])
                    if len(attr_str) > 70: attr_str = attr_str[:67] + "..."
                    
                    lines.append(f"  • Row Indices: {item['row_indices']}")
                    lines.append(f"    Data: {attr_str}")

            # 4. Outliers
            so = self.findings.get('numerical_outliers', [])
            lines.append(f"\n{BLUE}4] Statistical Outliers (IQR Method){END}")
            lines.append("-" * 80)
            if not so: lines.append("  ✓ No statistical outliers.")
            else:
                for item in so:
                    lines.append(f"  • '{item['column']}' ({item['count']} outliers)")

            # 5. Type Inconsistency
            ti = self.findings.get('type_inconsistency', [])
            lines.append(f"\n{BLUE}5] Type Inconsistency{END}")
            lines.append("-" * 80)
            if not ti: lines.append("  ✓ All types consistent.")
            else:
                for item in ti:
                    lines.append(f"  • '{item['column']}' | Types: {YEL}{item['types_found']}{END}")

            # 6. Logical Outliers
            lo = self.findings.get('logical_outliers', [])
            lines.append(f"\n{BLUE}6] Logical Outliers (Rule Violations){END}")
            lines.append("-" * 80)
            if not lo: lines.append("  ✓ No rule violations.")
            else:
                for item in lo:
                    lines.append(f"  • '{item['column']}' | {RED}{item['issue']}{END} ({item['count']} total)")

        # --- ANOMALY SUMMARY ---
        lines.append(f"\n{HEADER}--- ANOMALY SUMMARY ---{END}")
        f = self.findings

        if self.mode in ['basic', 'full']:
            lines.append(f"  [1] Rare Values:    {'✓ Pass' if not f.get('rare_values') else f'{len(f.get('rare_values', [])):<3}'}")
            lines.append(f"  [2] Null Values:    {'✓ Pass' if not f.get('null_values') else f'{len(f.get('null_values', [])):<3}'}")
            lines.append(f"  [3] Duplicates:     {'✓ Pass' if not f.get('duplicate_rows') else f'{len(f.get('duplicate_rows', [])):<3}'}")
            
            out_cnt = sum(item['count'] for item in f.get('numerical_outliers', []))
            lines.append(f"  [4] Outliers:       {'✓ Pass' if not f.get('numerical_outliers') else f'{out_cnt:<3}'}")
            
            lines.append(f"  [5] Mixed Types:    {'✓ Pass' if not f.get('type_inconsistency') else f'{len(f.get('type_inconsistency', [])):<3}'}")
            
            log_cnt = sum(item['count'] for item in f.get('logical_outliers', []))
            lines.append(f"  [6] Logic Rules:    {'✓ Pass' if not f.get('logical_outliers') else f'{log_cnt:<3}'}")

        if self.mode in ['auto', 'full']:
            am_data = f.get('auto_multivariate', [])
            auto_cnt = am_data[0]['count'] if am_data and 'count' in am_data[0] else 0
            lines.append(f"  [AUTO] Machine ML:  {'✓ Pass' if not am_data else f'{auto_cnt:<3}'}")

        lines.append(f"{HEADER}{'='*80}{END}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()
