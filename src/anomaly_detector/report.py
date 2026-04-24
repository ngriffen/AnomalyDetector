"""
AnomalyReport — holds all findings from a detector run and provides
pretty-printing and export helpers.
"""

from __future__ import annotations

import textwrap
from typing import Any

import pandas as pd


class AnomalyReport:
    """
    Container for anomaly-detection findings.

    Attributes
    ----------
    findings : dict[str, list[dict]]
        Raw findings keyed by check name.
    df_shape : tuple[int, int]
        (rows, columns) of the analysed DataFrame.
    """

    def __init__(
        self,
        findings: dict[str, list[dict]],
        df_shape: tuple[int, int],
    ) -> None:
        self.findings = findings
        self.df_shape = df_shape

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def rare_values(self) -> list[dict]:
        """Return the raw rare-value findings list."""
        return self.findings.get("rare_values", [])

    def rare_values_df(self) -> pd.DataFrame:
        """Return rare-value findings as a tidy DataFrame."""
        data = self.rare_values
        if not data:
            return pd.DataFrame(
                columns=["column", "value", "count", "pct", "total_rows", "unique_vals"]
            )
        return pd.DataFrame(data)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """
        Return a human-readable text summary of all findings.
        """
        rows, cols = self.df_shape
        lines: list[str] = [
            "=" * 60,
            "  StatGuard Anomaly Report",
            "=" * 60,
            f"  Dataset : {rows:,} rows × {cols:,} columns",
            "",
        ]

        # ---- Rare values -------------------------------------------------
        rv = self.rare_values
        lines.append(f"[1] Rare Values  ({len(rv)} flag(s))")
        lines.append("-" * 60)
        if not rv:
            lines.append("    ✓ No rare values detected.")
        else:
            # Group by column for readability
            by_col: dict[str, list[dict]] = {}
            for item in rv:
                by_col.setdefault(item["column"], []).append(item)

            for col, items in by_col.items():
                lines.append(f"  Column : '{col}'  ({items[0]['unique_vals']} unique vals)")
                for item in items:
                    flag = (
                        f"    • {_fmt_value(item['value'])!r:<20} "
                        f"count={item['count']:>4}  "
                        f"({item['pct']:.2f}% of non-null rows)"
                    )
                    lines.append(flag)
                lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def to_csv(self, path: str) -> None:
        """Export all rare-value findings to a CSV file."""
        self.rare_values_df().to_csv(path, index=False)
        print(f"Rare-value findings written to: {path}")

    def to_excel(self, path: str) -> None:
        """Export findings to an Excel workbook (requires openpyxl)."""
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            self.rare_values_df().to_excel(
                writer, sheet_name="rare_values", index=False
            )
        print(f"Report written to: {path}")

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # noqa: D105
        n = len(self.rare_values)
        return f"<AnomalyReport shape={self.df_shape} rare_value_flags={n}>"

    def __str__(self) -> str:  # noqa: D105
        return self.summary()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _fmt_value(val: Any) -> str:
    """Truncate long string values for display."""
    s = str(val)
    return s if len(s) <= 30 else s[:27] + "..."
