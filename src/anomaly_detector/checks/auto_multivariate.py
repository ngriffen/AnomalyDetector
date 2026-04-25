import pandas as pd
import numpy as np

def check_auto_multivariate(df: pd.DataFrame, contamination: float = 0.01) -> list[dict]:
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        return [{"error": "Missing dependency. Please run: pip install scikit-learn"}]

    numeric_df = df.select_dtypes(include=['number']).dropna()
    if len(numeric_df) < 10:
        return []

    # Calculate global stats to find what 'normal' looks like
    means = numeric_df.mean()
    stds = numeric_df.std().replace(0, 1) # Prevent division by zero

    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(numeric_df)
    outliers_idx = numeric_df[preds == -1].index
    
    findings = []
    if len(outliers_idx) > 0:
        details = []
        for idx in outliers_idx:
            row_data = df.iloc[idx].to_dict()
            numeric_row = numeric_df.loc[idx]
            
            # Find which columns are the most 'extreme' (highest Z-score)
            z_scores = abs((numeric_row - means) / stds)
            # We flag columns that are > 2 standard deviations away
            suspect_cols = z_scores[z_scores > 2.0].index.tolist()
            
            details.append({
                "row": idx, 
                "val": row_data,
                "suspects": suspect_cols # Pass the 'culprit' columns to the report
            })

        findings.append({
            "issue": "Multivariate Anomaly",
            "count": len(outliers_idx),
            "details": details
        })

    return findings
