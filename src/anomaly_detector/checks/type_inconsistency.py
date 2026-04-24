import pandas as pd

def check_type_inconsistency(df: pd.DataFrame) -> list[dict]:
    findings = []
    for col in df.columns:
        # Check if the column has multiple data types (excluding None)
        types = df[col].dropna().map(type).unique()
        if len(types) > 1:
            findings.append({
                "column": col,
                "types_found": [t.__name__ for t in types],
                "issue": "Mixed data types detected"
            })
    return findings