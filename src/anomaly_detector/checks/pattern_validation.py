import pandas as pd

def check_pattern_validation(df: pd.DataFrame, patterns: dict = None) -> list[dict]:
    findings = []
    if not patterns:
        return findings
    for col, pattern in patterns.items():
        if col not in df.columns: continue
        series = df[col].dropna().astype(str)
        is_match = series.str.match(pattern, na=False)
        violators = df.loc[series[~is_match].index]
        if not violators.empty:
            findings.append({
                "column": col,
                "issue": f"Regex Mismatch ({pattern})",
                "count": len(violators),
                "details": [{"row": idx, "val": val} for idx, val in violators[col].head(5).items()]
            })
    return findings
