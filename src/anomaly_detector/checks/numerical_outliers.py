import pandas as pd

def check_numerical_outliers(df: pd.DataFrame) -> list[dict]:
    findings = []
    numeric_cols = df.select_dtypes(include=['number']).columns

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty: continue
            
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        
        outliers = series[(series < lower) | (series > upper)]
        if not outliers.empty:
            # Capture the exact index and value for the report
            sample_indices = outliers.index.tolist()[:5]
            sample_values = outliers.tolist()[:5]
            
            findings.append({
                "column": col,
                "count": len(outliers),
                "bounds": (round(lower, 2), round(upper, 2)),
                "details": [{"row": idx, "val": val} for idx, val in zip(sample_indices, sample_values)]
            })
    return findings
