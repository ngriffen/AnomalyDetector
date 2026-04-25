import pandas as pd
import numpy as np

def check_auto_multivariate(df: pd.DataFrame, contamination: float = 0.01) -> list[dict]:
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        return [{"error": "Missing dependency. Please run: pip install scikit-learn"}]

    numeric_df = df.select_dtypes(include=['number']).dropna()
    if len(numeric_df) < 10: return []

    means = numeric_df.mean()
    stds = numeric_df.std().replace(0, 1)

    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(numeric_df)
    outliers_idx = numeric_df[preds == -1].index
    
    findings = []
    if len(outliers_idx) > 0:
        details = []
        for idx in outliers_idx:
            row_data = df.iloc[idx].to_dict()
            numeric_row = numeric_df.loc[idx]
            
            # Calculate deviance for all columns
            z_scores = abs((numeric_row - means) / stds)
            
            # Logic: If any col is > 2.0, it's a primary suspect.
            # Otherwise, take the top 2 highest contributors.
            primary_suspects = z_scores[z_scores > 2.0].index.tolist()
            
            if not primary_suspects:
                # Get top 2 contributors to the "weirdness"
                top_2 = z_scores.nlargest(2).index.tolist()
                suspect_list = [f"{col}*" for col in top_2] # * denotes secondary contributor
                tag = "Multivariate Mix"
            else:
                suspect_list = primary_suspects
                tag = "Specific"

            details.append({
                "row": idx, 
                "val": row_data,
                "suspects": suspect_list,
                "tag": tag
            })

        findings.append({
            "issue": "Multivariate Anomaly",
            "count": len(outliers_idx),
            "details": details
        })

    return findings
