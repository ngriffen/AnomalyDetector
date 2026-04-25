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
    
    # We only analyze numeric columns for the auto-detector to prevent string errors
    numeric_df = df.select_dtypes(include=['number']).dropna()
    
    # Not enough data to confidently run ML
    if len(numeric_df) < 50:
        return findings

    # Train the Auto-Detector
    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(numeric_df)
    
    # -1 means anomaly, 1 means normal
    outliers_idx = numeric_df[preds == -1].index
    
    if len(outliers_idx) > 0:
        violators = df.loc[outliers_idx]
        
        sample_indices = violators.index.tolist()[:5]
        # Store the full row dictionary for the report
        sample_rows = violators.iloc[:5].to_dict(orient='records')
        
        findings.append({
            "issue": "Multivariate Anomaly Detected",
            "count": len(violators),
            "details": [{"row": idx, "val": row_data} for idx, row_data in zip(sample_indices, sample_rows)]
        })

    return findings