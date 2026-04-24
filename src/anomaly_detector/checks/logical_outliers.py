import pandas as pd

def check_logical_outliers(df: pd.DataFrame, rules: dict = None) -> list[dict]:
    """
    Checks for logical violations based on a provided rules dictionary.
    
    Example rules format:
    rules = {
        "age": {"min": 0, "max": 120},
        "score": {"min": 0, "max": 100},
        "rating": {"allowed": [1, 2, 3, 4, 5]},
        "email": {"regex": r'[^@]+@[^@]+\.[^@]+'}
    }
    """
    findings = []
    if not rules:
        return findings

    for col, constraints in rules.items():
        if col not in df.columns:
            continue

        series = df[col]
        
        # 1. Range Checks (Min/Max)
        if "min" in constraints:
            violators = df[series < constraints["min"]]
            if not violators.empty:
                findings.append(_format_violation(col, f"Value < {constraints['min']}", violators))
        
        if "max" in constraints:
            violators = df[series > constraints["max"]]
            if not violators.empty:
                findings.append(_format_violation(col, f"Value > {constraints['max']}", violators))

        # 2. Allowed Values Check
        if "allowed" in constraints:
            violators = df[~series.isin(constraints["allowed"]) & series.notnull()]
            if not violators.empty:
                findings.append(_format_violation(col, "Value not in allowed list", violators))

        # 3. Regex Pattern Check (for Strings)
        if "regex" in constraints:
            violators = df[~series.astype(str).str.contains(constraints["regex"], na=False)]
            if not violators.empty:
                findings.append(_format_violation(col, "Value does not match pattern", violators))

    return findings

def _format_violation(col, issue, violators_df):
    """Helper to standardize finding output."""
    return {
        "column": col,
        "issue": issue,
        "count": len(violators_df),
        "sample_indices": violators_df.index.tolist()[:5]
    }