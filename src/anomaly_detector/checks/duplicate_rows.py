import pandas as pd

def check_duplicate_rows(df: pd.DataFrame, subset: list[str] | None = None) -> list[dict]:
    # keep=False marks ALL duplicates so we can group and count them accurately
    duplicates = df[df.duplicated(subset=subset, keep=False)]
    
    findings = []
    if not duplicates.empty:
        # Group by the identical values to count occurrences
        cols_to_group = subset if subset else df.columns.tolist()
        grouped = duplicates.groupby(cols_to_group, dropna=False)
        
        for _, group in grouped:
            if len(group) > 1:
                findings.append({
                    "attributes": group.iloc[0].to_dict(),
                    "occurrences": len(group),
                    "row_indices": group.index.tolist()
                })
                
        # Sort so the most duplicated items appear first
        findings.sort(key=lambda x: x["occurrences"], reverse=True)
            
    return findings
