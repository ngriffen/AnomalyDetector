"""
AnomalyReport — holds all findings from a detector run and provides
pretty-printing and export helpers.
"""

from __future__ import annotations

from typing import Any
import pandas as pd


class AnomalyReport:
    """Container for anomaly-detection findings."""

    def __init__(
        self,
        findings: dict[str, list[dict]],
        df_shape: tuple[int, int],
    ) -> None:
        self.findings = findings
        self.df_shape = df_shape

    @property
    def rare_values(self) -> list[dict]:
        return self.findings.get("rare_values", [])

    @property
    def null_values(self) -> list[dict]:
        return self.findings.get("null_values", [])

    @property
    def duplicate_rows(self) -> list[dict]:
        return self.findings.get("duplicate_rows", [])

    def rare_values_df(self) -> pd.DataFrame:
        """Return rare-value findings as a tidy DataFrame."""
        data = self.rare_values
        if not data:
            return pd.DataFrame(
                columns=["column", "value", "count", "pct", "total_rows", "unique_vals"]
            )
        return pd.DataFrame(data)

    def summary(self) -> str:
        """Return a human-readable text summary of all findings."""
        rows, cols = self.df_shape
        
        # Color codes for a clean terminal/Colab output
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        WARNING = '\033[93m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'

        lines: list[str] = [
            f"{HEADER}{'=' * 60}{ENDC}",
            f"  {BOLD}Anomaly Report{ENDC}",
            f"{HEADER}{'=' * 60}{ENDC}",
            f"  Dataset : {rows:,} rows × {cols:,} columns",
            "",
        ]

        # ---- Section 1: Anomalies Detected (Rare Values) ----
        rv = self.rare_values
        lines.append(f"{OKBLUE}[1] Anomalies Detected  ({len(rv)} flag(s)){ENDC}")
        lines.append("-" * 60)
        if not rv:
            lines.append("    ✓ No rare values detected.")
        else:
            by_col: dict[str, list[dict]] = {}
            for item in rv:
                by_col.setdefault(item["column"], []).append(item)

            for col, items in by_col.items():
                lines.append(f"  Column : '{col}'  ({items[0]['unique_vals']} unique vals)")
                for item in items:
                    lines.append(
                        f"    • {_fmt_value(item['value'])!r:<20} "
                        f"count={item['count']:>4}  "
                        f"({item['pct']:.2f}%)"
                    )
                lines.append("")

        # ---- Section 2: Null Values Detected ----
        nv = self.null_values
        lines.append(f"{OKBLUE}[2] Null Values Detected  ({len(nv)} flag(s)){ENDC}")
        lines.append("-" * 60)
        if not nv:
            lines.append("    ✓ No high null-rate columns detected.")
        else:
            for item in nv:
                lines.append(f"  Column : '{item['column']}' -> {WARNING}{item['null_pct']:.2f}% missing{ENDC}")
            lines.append("")

        # ---- Section 3: Duplicates Detected ----
        dupes = self.duplicate_rows
        dupe_count = dupes[0]['count'] if dupes else 0
        lines.append(f"{OKBLUE}[3] Duplicates Detected{ENDC}")
        lines.append("-" * 60)
        if dupe_count == 0:
            lines.append("    ✓ No duplicate rows detected.")
        else:
            lines.append(f"  {WARNING}Found {dupe_count:,} duplicate rows.{ENDC}")
            if dupes[0].get('subset') != "all_columns":
                lines.append(f"  (Subset checked: {dupes[0]['subset']})")
        
        lines.append("")
        lines.append(f"{HEADER}{'=' * 60}{ENDC}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

def _fmt_value(val: Any) -> str:
    s = str(val)
    return s if len(s) <= 30 else s[:27] + "..."
