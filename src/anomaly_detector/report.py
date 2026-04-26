from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np

class AnomalyReport:
    def __init__(self, df: pd.DataFrame, findings: dict[str, list[dict]], mode: str) -> None:
        self.df = df
        self.findings = findings
        self.df_shape = df.shape
        self.mode = mode

    def summary(self) -> str:
        # --- UI Configuration ---
        HEADER, BLUE, YEL, RED, END = '\033[95m', '\033[94m', '\033[93m', '\033[91m', '\033[0m'
        BOLD = '\033[1m'

        lines = [
            f"{HEADER}{'='*80}{END}",
            f"  {BOLD}ANOMALY REPORT [{self.mode.upper()} MODE]{END}",
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
            else:
                for item in am:
                    lines.append(f"  {RED}! Detected {item['count']} rows with anomalous combinations.{END}")
                    for detail in item.get('details', []):
                        suspects = detail.get('suspects', [])
                        tag = detail.get('tag', 'Unknown')
                        
                        if tag == "Multivariate Mix":
                            sus_label = f"{YEL}[Complex Interaction: {', '.join(suspects)}]{END}"
                        else:
                            sus_label = f"{RED}[Suspect: {', '.join(suspects)}]{END}"
                        
                        row_str = ", ".join([f"{k}: {v}" for k, v in detail['val'].items()])
                        lines.append(f"    - Row {detail['row']:>4}: {sus_label}")
                        lines.append(f"               Data: {row_str}")

        # --- BASIC Detailed Sections ---
        if self.mode in ['basic', 'full']:
            # 1. Rare Values
            rv = self.findings.get('rare_values', [])
            lines.append(f"\n{BLUE}1] Anomalies Detected (Rare Values){END}")
            lines.append("-" * 80)
            if not rv: 
                lines.append("  ✓ No rare values detected.")
            else:
                for item in rv:
                    lines.append(f"    - Col: '{item['column']}' | Value: {str(item['value']):<10} | Count: {item['count']} ({item['pct']:.2f}%)")

            # 2. Null Values
            nv = self.findings.get('null_values', [])
            lines.append(f"\n{BLUE}2] Null Values Detected{END}")
            lines.append("-" * 80)
            if not nv: 
                lines.append("  ✓ No high null-rate columns.")
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
                    attr_str = ", ".join([f"{k}: {v}" for k, v in item['attributes'].items()])
                    if len(attr_str) > 75: attr_str = attr_str[:72] + "..."
                    lines.append(f"  • Row Indices: {item['row_indices']}")
                    lines.append(f"    Data: {attr_str}")

            # 4. Outliers
            so = self.findings.get('numerical_outliers', [])
            lines.append(f"\n{BLUE}4] Statistical Outliers (IQR Method){END}")
            lines.append("-" * 80)
            if not so: 
                lines.append("  ✓ No statistical outliers.")
            else:
                for item in so:
                    lines.append(f"  • '{item['column']}' ({item['count']} outliers)")

            # 5. Type Inconsistency
            ti = self.findings.get('type_inconsistency', [])
            lines.append(f"\n{BLUE}5] Type Inconsistency{END}")
            lines.append("-" * 80)
            if not ti: 
                lines.append("  ✓ All types consistent.")
            else:
                for item in ti:
                    lines.append(f"  • '{item['column']}' | Types: {YEL}{item['types_found']}{END}")

            # 6. Logical Outliers
            lo = self.findings.get('logical_outliers', [])
            lines.append(f"\n{BLUE}6] Logical Outliers (Rule Violations){END}")
            lines.append("-" * 80)
            if not lo: 
                lines.append("  ✓ No rule violations.")
            else:
                for item in lo:
                    lines.append(f"  • '{item['column']}' | {RED}{item['issue']}{END} ({item['count']} total)")

        # --- EXECUTIVE SUMMARY ---
        lines.append(f"\n{HEADER}--- ANOMALY SUMMARY ---{END}")
        f = self.findings
        if self.mode in ['basic', 'full']:
            rv_cnt = len(f.get('rare_values', []))
            nv_cnt = len(f.get('null_values', []))
            dv_cnt = len(f.get('duplicate_rows', []))
            out_cnt = sum(item['count'] for item in f.get('numerical_outliers', []))
            ti_cnt = len(f.get('type_inconsistency', []))
            log_cnt = sum(item['count'] for item in f.get('logical_outliers', []))
            lines.append(f"  [1] Rare Values:    {'✓ Pass' if rv_cnt == 0 else f'{rv_cnt:<3}'}")
            lines.append(f"  [2] Null Values:    {'✓ Pass' if nv_cnt == 0 else f'{nv_cnt:<3}'}")
            lines.append(f"  [3] Duplicates:     {'✓ Pass' if dv_cnt == 0 else f'{dv_cnt:<3}'}")
            lines.append(f"  [4] Outliers:       {'✓ Pass' if out_cnt == 0 else f'{out_cnt:<3}'}")
            lines.append(f"  [5] Mixed Types:    {'✓ Pass' if ti_cnt == 0 else f'{ti_cnt:<3}'}")
            lines.append(f"  [6] Logic Rules:    {'✓ Pass' if log_cnt == 0 else f'{log_cnt:<3}'}")
        if self.mode in ['auto', 'full']:
            am_data = f.get('auto_multivariate', [])
            auto_cnt = am_data[0]['count'] if am_data and 'count' in am_data[0] else 0
            lines.append(f"  [AUTO] Machine ML:  {'✓ Pass' if auto_cnt == 0 else f'{auto_cnt:<3}'}")

        lines.append(f"{HEADER}{'='*80}{END}")
        return "\n".join(lines)

    def visualize(self) -> None:
        """Generates a visual dashboard of the detected anomalies."""
        sns.set_theme(style="whitegrid")
        
        # Determine which plots to generate
        plots_to_make = []
        if self.findings.get('null_values'): plots_to_make.append('nulls')
        if self.findings.get('numerical_outliers'): plots_to_make.append('outliers')
        if self.findings.get('rare_values'): plots_to_make.append('rare')
        if self.findings.get('auto_multivariate'): plots_to_make.append('auto')

        if not plots_to_make:
            print("No anomalies found to visualize.")
            return

        # FIXED GRID LOGIC: Always flattens axes into a usable list
        num_plots = len(plots_to_make)
        cols = 2 if num_plots > 1 else 1
        rows = math.ceil(num_plots / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows), squeeze=False)
        axes_list = axes.flatten()
        idx = 0

        # 1. Null Values
        if 'nulls' in plots_to_make:
            ax = axes_list[idx]
            null_data = self.findings['null_values']
            sns.barplot(x=[i['null_pct'] for i in null_data], y=[i['column'] for i in null_data], ax=ax, palette="Reds_r")
            ax.set_title("Missing Data Percentage", fontweight='bold')
            ax.set_xlim(0, 100)
            idx += 1

        # 2. Outliers
        if 'outliers' in plots_to_make:
            ax = axes_list[idx]
            out_cols = [i['column'] for i in self.findings['numerical_outliers'][:3]]
            melted = self.df[out_cols].melt()
            sns.boxplot(data=melted, x='value', y='variable', ax=ax, palette="Set2")
            ax.set_title("Statistical Outliers (Top 3 Columns)", fontweight='bold')
            idx += 1

        # 3. Rare Values
        if 'rare' in plots_to_make:
            ax = axes_list[idx]
            target = self.findings['rare_values'][0]['column']
            sns.countplot(y=self.df[target], ax=ax, order=self.df[target].value_counts().index, palette="viridis")
            ax.set_title(f"Rare Values in '{target}'", fontweight='bold')
            idx += 1

        # 4. Auto ML (Multivariate)
        if 'auto' in plots_to_make:
            ax = axes_list[idx]
            am_details = self.findings['auto_multivariate'][0]['details']
            anomaly_idx = [d['row'] for d in am_details]
            
            num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols) >= 2:
                x, y = num_cols[0], num_cols[1]
                # Highlighting anomalies in red
                colors = ['#ff4d4d' if i in anomaly_idx else '#4d79ff' for i in self.df.index]
                ax.scatter(self.df[x], self.df[y], c=colors, alpha=0.6, edgecolors='w', s=50)
                ax.set_title(f"ML Detection: {x} vs {y}", fontweight='bold')
                ax.set_xlabel(x); ax.set_ylabel(y)
            idx += 1

        # Remove extra empty subplots
        for i in range(idx, len(axes_list)):
            fig.delaxes(axes_list[i])

        plt.tight_layout()
        plt.show()

    def __str__(self) -> str:
        return self.summary()
