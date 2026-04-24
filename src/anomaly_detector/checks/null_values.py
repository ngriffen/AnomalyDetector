import pandas as pd

def check_null_values(df: pd.DataFrame, threshold_pct: float = 5.0) -> list[dict]:
    """
    Flags columns where the percentage of missing values is >= threshold_pct.
    """
    findings = []
    null_counts = df.isnull().sum()
    null_pcts = (null_counts / len(df)) * 100

    for col in df.columns:
        pct = null_pcts[col]
        if pct >= threshold_pct:
            findings.append({
                "column": col,
                "null_count": int(null_counts[col]),
                "null_pct": round(pct, 2),
                "issue_type": "high_null_rate"
            })
    return findings