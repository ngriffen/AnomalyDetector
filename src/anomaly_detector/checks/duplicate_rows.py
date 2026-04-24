import pandas as pd

def check_duplicate_rows(df: pd.DataFrame, subset: list[str] | None = None) -> list[dict]:
    """
    Flags duplicate rows and captures their index and attributes.
    """
    # keep='first' marks the 2nd, 3rd, etc. occurrences as True
    is_duplicate = df.duplicated(subset=subset, keep='first')
    duplicate_indices = df.index[is_duplicate].tolist()
    
    findings = []
    if duplicate_indices:
        for idx in duplicate_indices:
            # Capture the values of the row to show the "attributes"
            # We convert to dict for easy reporting
            row_values = df.loc[idx].to_dict()
            
            findings.append({
                "row_index": idx,
                "attributes": row_values,
                "subset": subset or "all_columns",
                "total_count": len(duplicate_indices)
            })
            
    return findings
