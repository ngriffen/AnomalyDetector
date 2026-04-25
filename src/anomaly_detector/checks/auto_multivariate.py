import pandas as pd

def check_auto_multivariate(df: pd.DataFrame, contamination: float = 0.01) -> list[dict]:
    """
    Uses Unsupervised Machine Learning (Isolation Forest) to automatically 
    detect rows that are anomalous across multiple numeric dimensions.
    """
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        return [{"error": "Missing dependency. Please run: pip install scikit-learn"}]

    findings = []
    
    # Analyze numeric columns only to avoid processing errors
    numeric_df = df.select_dtypes(include=['number']).dropna()
    
    # Lowered threshold to 10 so it triggers more easily in testing/small datasets
    if len(numeric_df) < 10:
        return findings

    # Train the Auto-Detector
    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(numeric_df)
    
    # -1 means anomaly, 1 means normal
    outliers_idx = numeric_df[preds == -1].index
    
    if len(outliers_idx) > 0:
        violators = df.loc[outliers_idx]
        
        # REMOVED [:5] limit: Capture ALL indices and ALL row data
        all_indices = violators.index.tolist()
        all_rows = violators.to_dict(orient='records')
        
        findings.append({
            "issue": "Multivariate Anomaly",
            "count": len(violators),
            "details": [
                {"row": idx, "val": row_data} 
                for idx, row_data in zip(all_indices, all_rows)
            ]
        })

    return findings
