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
            findings.append({
                "column": col,
                "count": len(outliers),
                "bounds": (round(lower, 2), round(upper, 2)),
                "samples": outliers.unique().tolist()[:3]
            })
    return findings