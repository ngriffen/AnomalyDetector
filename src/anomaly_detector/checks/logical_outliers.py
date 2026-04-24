import pandas as pd

def check_logical_outliers(df: pd.DataFrame, rules: dict = None) -> list[dict]:
    """
    Checks for logical violations and captures precise indices and values.
    """
    findings = []
    if not rules:
        return findings

    for col, constraints in rules.items():
        if col not in df.columns:
            continue

        series = df[col]
        
        # 1. Range Checks
        if "min" in constraints:
            violators = df[series < constraints["min"]]
            if not violators.empty:
                findings.append(_format_violation(col, f"Value < {constraints['min']}", violators))
        
        if "max" in constraints:
            violators = df[series > constraints["max"]]
            if not violators.empty:
                findings.append(_format_violation(col, f"Value > {constraints['max']}", violators))

        # 2. Allowed Values
        if "allowed" in constraints:
            violators = df[~series.isin(constraints["allowed"]) & series.notnull()]
            if not violators.empty:
                findings.append(_format_violation(col, "Value not in allowed list", violators))

        # 3. Regex
        if "regex" in constraints:
            violators = df[~series.astype(str).str.contains(constraints["regex"], na=False)]
            if not violators.empty:
                findings.append(_format_violation(col, "Value does not match pattern", violators))

    return findings

def _format_violation(col, issue, violators_df):
    """Captures precise indices and the values that failed."""
    # We take up to the first 10 for the report to keep it readable
    sample_indices = violators_df.index.tolist()[:10]
    sample_values = violators_df[col].iloc[:10].tolist()
    
    return {
        "column": col,
        "issue": issue,
        "count": len(violators_df),
        "details": [
            {"row": idx, "val": val} for idx, val in zip(sample_indices, sample_values)
        ]
    }
