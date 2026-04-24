"""
check_rare_values
-----------------
Scans every column in the DataFrame and flags individual values whose
frequency (absolute count) is at or below `threshold`.

A "rare" value could indicate:
  - A data-entry error          ("Femlae" instead of "Female")
  - A genuine statistical outlier in a categorical/discrete column
  - A miscoded sentinel value   (-999, "N/A ", "na", etc.)
  - A legitimate but extremely uncommon category worth review

Only columns whose total unique-value count is ≤ max_categories are
inspected; ultra-high-cardinality columns (raw IDs, free-text) would
flood the results with noise.
"""

from __future__ import annotations

import pandas as pd


def check_rare_values(
    df: pd.DataFrame,
    threshold: int = 5,
    max_categories: int = 50,
) -> list[dict]:
    """
    Parameters
    ----------
    df : pd.DataFrame
    threshold : int
        Counts ≤ this value are flagged as rare.
    max_categories : int
        Skip columns with more unique values than this.

    Returns
    -------
    list[dict]
        One entry per (column, value) pair that is flagged.
        Each dict has the keys:
          - column      : str   — column name
          - value       : any   — the rare value itself
          - count       : int   — how many times it appears
          - pct         : float — percentage of non-null rows
          - total_rows  : int   — non-null row count for that column
          - unique_vals : int   — total unique values in the column
    """
    findings: list[dict] = []
    n_rows = len(df)

    for col in df.columns:
        series = df[col]

        # Drop nulls — nulls are handled by a dedicated check (future)
        non_null = series.dropna()
        if non_null.empty:
            continue

        n_non_null = len(non_null)
        value_counts = non_null.value_counts(sort=True, ascending=True)
        n_unique = len(value_counts)

        # Skip columns that are too high-cardinality to be useful
        if n_unique > max_categories:
            continue

        # Flag values whose count is at or below the threshold
        rare = value_counts[value_counts <= threshold]

        for value, count in rare.items():
            findings.append(
                {
                    "column": col,
                    "value": value,
                    "count": int(count),
                    "pct": round(count / n_non_null * 100, 2),
                    "total_rows": n_non_null,
                    "unique_vals": n_unique,
                }
            )

    # Sort so the rarest values surface first
    findings.sort(key=lambda x: (x["column"], x["count"]))
    return findings
