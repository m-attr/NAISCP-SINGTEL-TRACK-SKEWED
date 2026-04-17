import pandas as pd
import numpy as np

def classify_features(
    df: pd.DataFrame,
    primary_key: str = "CustomerID",
    time_col: str = "Month",
    target_col: str = "ChurnStatus",
    sample_rows: int | None = None
):
    """
    Dynamically sorts columns into mathematical buckets based on their properties,
    ignoring the column names.

    cat_low -> low-cardinality: text or discrete numbers with a small menu of options (usually under 15-50 unique values).
    cat_high -> high-cardinality: text or discrete numbers with a large menu of options (usually over 50 unique values).
    """
    col_groups = {
        "numerical": [],
        "binary": [],
        "cat_low": [],
        "cat_high": [],
        "cat_high_te": [],
    }

    skip_cols = {primary_key, time_col, target_col}
    
    scan_df = df
    if sample_rows is not None and len(df) > sample_rows:
        scan_df = df.iloc[:sample_rows]

    n_rows = max(1, len(scan_df))
    low_card_ceiling = max(12, int(0.02 * n_rows))

    for col in df.columns:
        if col in skip_cols:
            continue
            
        series = scan_df[col]
        n_unique = int(series.nunique(dropna=True))
        dtype = series.dtype

        if n_unique <= 1:
            continue
        
        # Rule 1: Binary (Exactly 2 unique values, regardless of text or number)
        if n_unique == 2:
            col_groups["binary"].append(col)
            continue
            
        # Rule 2: Text/String Columns
        if (
            pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
        ):
            if n_unique > max(64, low_card_ceiling):
                col_groups["cat_high"].append(col)
                col_groups["cat_high_te"].append(col)
            else:
                col_groups["cat_low"].append(col)
            continue
            
        # Rule 3: Numeric Columns
        if pd.api.types.is_numeric_dtype(dtype):
            numeric_series = pd.to_numeric(series, errors="coerce").dropna()
            if numeric_series.empty:
                continue

            arr = numeric_series.to_numpy(dtype=np.float64, copy=False)
            integer_like = bool(np.isclose(np.mod(arr, 1.0), 0.0, atol=1e-8).mean() >= 0.98)

            # Discrete numeric columns with low support are treated as categorical.
            if n_unique <= low_card_ceiling and (integer_like or n_unique <= 20):
                col_groups["cat_low"].append(col)
            else:
                col_groups["numerical"].append(col)
                
    return col_groups
