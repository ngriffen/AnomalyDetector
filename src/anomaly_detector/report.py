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
        lines.append(f"\n{HEADER}--- EXECUTIVE SUMMARY ---{END}")
        
        if self.mode in ['basic', 'full']:
            lines.append(f"  [1] Rare Values:    {'✓ Pass' if not self.findings.get('rare_values') else f'! {len(self.findings.get(\"rare_values\")):<3} | Anomalies'}")
            # ... (Add the rest of your 6 basic summary lines here) ...

        if self.mode in ['auto', 'full']:
            am = self.findings.get("auto_multivariate", [])
            lines.append(f"  [AUTO] Machine ML:  {'✓ Pass' if not am else f'! {am[0][\"count\"] if \"count\" in am[0] else \"ERR\":<3} | Multivariate Anomalies'}")

        lines.append(f"{HEADER}{'='*80}{END}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()
