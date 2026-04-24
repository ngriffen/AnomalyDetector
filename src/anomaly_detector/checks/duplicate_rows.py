import pandas as pd

def check_duplicate_rows(df: pd.DataFrame, subset: list[str] | None = None) -> list[dict]:
    """
    Flags if there are any duplicate rows in the DataFrame.
    """
    # Find all duplicates except the first occurrence
    duplicates = df.duplicated(subset=subset, keep='first')
    count = duplicates.sum()

    if count > 0:
        return [{
            "issue_type": "duplicate_rows",
            "count": int(count),
            "subset": subset or "all_columns"
        }]
    return []