import hashlib
import os
import time
from pathlib import Path

import pandas as pd
from config import PIPELINE_CONFIG
from schema_utils import detect_identifier_column, detect_target_column, detect_time_column

try:
    import pyarrow as pa
    import pyarrow.csv as pa_csv
except Exception:
    pa = None
    pa_csv = None

PARQUET_CACHE_DIR = Path(".parquet_cache")



def _read_csv_head_pyarrow(path: str, nrows: int, block_size: int = 1 << 24) -> pd.DataFrame:
    if pa_csv is None:
        raise ImportError("pyarrow.csv is unavailable")

    reader = pa_csv.open_csv(path, read_options=pa_csv.ReadOptions(block_size=block_size))
    batches = []
    remaining = int(nrows)

    for batch in reader:
        if remaining <= 0:
            break
        # Slice the pyarrow batch directly (zero-copy) instead of converting to pandas first
        if batch.num_rows > remaining:
            batch = batch.slice(0, remaining)
        batches.append(batch)
        remaining -= batch.num_rows

    if not batches:
        return pd.DataFrame()
    
    # Combine in C++ memory space, then convert to pandas once
    import pyarrow as pa
    table = pa.Table.from_batches(batches)
    return table.to_pandas()


def _ensure_cache_dir() -> None:
    if not PARQUET_CACHE_DIR.exists():
        PARQUET_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(csv_path: str) -> Path:
    digest = hashlib.sha256(os.path.abspath(csv_path).encode()).hexdigest()[:12]
    stem = Path(csv_path).stem
    return PARQUET_CACHE_DIR / f"{stem}_{digest}.parquet"


def _read_pyarrow_full(path: str, nrows: int | None = None) -> pd.DataFrame:
    if pa_csv is None:
        raise ImportError("pyarrow.csv is unavailable")

    if nrows is not None and nrows > 0:
        return _read_csv_head_pyarrow(path, nrows=nrows)

    return pd.read_csv(path, engine="pyarrow")


def _try_load_parquet(path: str) -> pd.DataFrame | None:
    cache = _cache_path(path)
    if cache.exists() and cache.stat().st_mtime >= os.path.getmtime(path):
        print("Loaded parquet cache for", path)
        return pd.read_parquet(cache)
    return None


def _write_parquet_cache(path: str, df: pd.DataFrame) -> None:
    _ensure_cache_dir()
    cache = _cache_path(path)
    try:
        df.to_parquet(cache, engine="pyarrow", compression="snappy")
    except Exception:
        pass


def _read_csv_fast(
    path: str,
    nrows: int | None = None,
) -> pd.DataFrame:
    """
    Prefer pyarrow for every load and cache the full-file reads as parquet.
    """
    try:
        if nrows is None:
            cached = _try_load_parquet(path)
            if cached is not None:
                return cached
            df = _read_pyarrow_full(path)
            _write_parquet_cache(path, df)
            print("Parser selected: pyarrow (cached full read)")
            return df
        print("Parser selected: pyarrow (partial read)")
        return _read_pyarrow_full(path, nrows=nrows)
    except Exception as e:
        print(f"Parser fallback: pandas-c (pyarrow path failed: {type(e).__name__}: {e})")

    file_mb = os.path.getsize(path) / (1024 ** 2)
    use_low_memory = file_mb >= 1500
    return pd.read_csv(path, low_memory=use_low_memory, memory_map=(not use_low_memory), nrows=nrows)


def _optimize_dtypes_inplace(
    df: pd.DataFrame,
    keep_string_cols: set[str] | None = None,
    category_sample_rows: int = 100000,
    max_unique_for_category: int = 1024,
    max_unique_ratio_for_category: float = 0.20,
) -> pd.DataFrame:
    """
    Compact dtypes to reduce memory and speed downstream operations.
    """
    if keep_string_cols is None:
        keep_string_cols = set()

    # 1) Downcast numerics where possible.
    num_cols = df.select_dtypes(include=["number"]).columns
    for col in num_cols:
        series = df[col]
        if pd.api.types.is_integer_dtype(series.dtype):
            df[col] = pd.to_numeric(series, errors="coerce", downcast="integer")
        elif pd.api.types.is_float_dtype(series.dtype):
            df[col] = pd.to_numeric(series, errors="coerce", downcast="float")

    # 2) Convert low-cardinality text columns to category.
    text_cols = df.select_dtypes(include=["object", "string", "category"]).columns
    for col in text_cols:
        if col in keep_string_cols:
            # Keep IDs/time columns as string for stability.
            try:
                df[col] = df[col].astype("string")
            except Exception:
                pass
            continue

        non_na = df[col].dropna()
        if non_na.empty:
            continue

        sample = non_na.iloc[: min(len(non_na), category_sample_rows)]
        est_unique = int(sample.nunique(dropna=True))
        est_ratio = est_unique / max(1, len(sample))

        if est_unique <= max_unique_for_category and est_ratio <= max_unique_ratio_for_category:
            try:
                df[col] = df[col].astype("category")
            except Exception:
                pass
        else:
            # Avoid Python-object strings when possible.
            try:
                df[col] = df[col].astype("string")
            except Exception:
                pass

    return df

def load_raw_datasets(
    train_path: str,
    test_path: str,
    max_train_rows: int | None = None,
    max_test_rows: int | None = None,
    optimize_dtypes: bool = True,
):
    """
    Load the raw CSV files into memory (DataFrames) 
    """

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Testing file not found at {test_path}")

    train_size_gb = os.path.getsize(train_path) / (1024 ** 3)
    test_size_gb = os.path.getsize(test_path) / (1024 ** 3)
    print(f"Reading train CSV ({train_size_gb:.2f} GB) with optimized parser...")
    t0 = time.time()
    if max_train_rows is not None and max_train_rows > 0:
        print(f"Train row cap active at load time: {max_train_rows}")
    train_df = _read_csv_fast(train_path, nrows=max_train_rows)
    t1 = time.time()
    print(f"Train loaded in {(t1 - t0):.2f}s")

    print(f"Reading test CSV ({test_size_gb:.2f} GB) with optimized parser...")
    if max_test_rows is not None and max_test_rows > 0:
        print(f"Test row cap active at load time: {max_test_rows}")
    test_df = _read_csv_fast(test_path, nrows=max_test_rows)
    t2 = time.time()
    print(f"Test loaded in {(t2 - t1):.2f}s")

    if optimize_dtypes:
        train_only_cols = [c for c in train_df.columns if c not in test_df.columns]
        # REFACTORED: centralized schema column names from shared config.
        id_col_train = PIPELINE_CONFIG.id_column if PIPELINE_CONFIG.id_column in train_df.columns else detect_identifier_column(train_df)
        id_col_test = PIPELINE_CONFIG.id_column if PIPELINE_CONFIG.id_column in test_df.columns else detect_identifier_column(test_df)
        time_col_train = PIPELINE_CONFIG.time_column if PIPELINE_CONFIG.time_column in train_df.columns else detect_time_column(train_df)
        time_col_test = PIPELINE_CONFIG.time_column if PIPELINE_CONFIG.time_column in test_df.columns else detect_time_column(test_df)
        target_col = PIPELINE_CONFIG.target_column if PIPELINE_CONFIG.target_column in train_df.columns else (
            train_only_cols[0] if len(train_only_cols) == 1 else detect_target_column(train_df)
        )

        keep_train = {c for c in [id_col_train, time_col_train, target_col] if c}
        keep_test = {c for c in [id_col_test, time_col_test] if c}

        train_df = _optimize_dtypes_inplace(
            train_df,
            keep_string_cols=keep_train,
        )
        test_df = _optimize_dtypes_inplace(
            test_df,
            keep_string_cols=keep_test,
        )

    print(f"Successfully loaded {len(train_df)} training rows and {len(test_df)} test rows.")
    train_mem_gb = train_df.memory_usage(deep=False).sum() / (1024 ** 3)
    test_mem_gb = test_df.memory_usage(deep=False).sum() / (1024 ** 3)
    print(f"Approx in-memory footprint (shallow): train={train_mem_gb:.2f} GB, test={test_mem_gb:.2f} GB")
    return train_df, test_df

def load_single_dataset(
    path: str,
    max_rows: int | None = None,
    dataset_name: str = "dataset",
    keep_string_cols: set[str] | None = None
):
    """
    Load a single CSV file with the same optimized path as the paired loader.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{dataset_name} file not found at {path}")

    size_gb = os.path.getsize(path) / (1024 ** 3)
    print(f"Reading {dataset_name} CSV ({size_gb:.2f} GB) with optimized parser...")
    t0 = time.time()
    if max_rows is not None and max_rows > 0:
        print(f"{dataset_name} row cap active at load time: {max_rows}")
    df = _read_csv_fast(path, nrows=max_rows)
    t1 = time.time()
    print(f"{dataset_name} loaded in {(t1 - t0):.2f}s")

    if keep_string_cols is None:
        keep_string_cols = set()
    df = _optimize_dtypes_inplace(df, keep_string_cols=keep_string_cols)

    mem_gb = df.memory_usage(deep=False).sum() / (1024 ** 3)
    print(f"{dataset_name} rows loaded: {len(df)} (approx shallow memory={mem_gb:.2f} GB)")
    return df

def split_features_and_target(df: pd.DataFrame, target_column: str | None = None):
    """
    Separate the target values (y) from feature columns (X).
    The detected/selected target column is removed from X before training.
    """

    if target_column is None:
        # REFACTORED: default target resolved from centralized config.
        target_column = PIPELINE_CONFIG.target_column

    if target_column not in df.columns:
        target_column = detect_target_column(df)

    if target_column is not None and target_column in df.columns:
        # Pop target to avoid duplicating a very large dataframe in memory.
        target_series = df.pop(target_column)
        y = target_series.map({"Yes": 1, "No": 0, "yes": 1, "no": 0, "Y": 1, "N": 0, "y": 1, "n": 0})
        if y.isna().all():
            if pd.api.types.is_numeric_dtype(target_series):
                y = pd.to_numeric(target_series, errors="coerce")
            else:
                normalized = target_series.astype("string").str.strip().str.lower()
                uniques = [u for u in normalized.dropna().unique().tolist()]
                if len(uniques) == 2:
                    positive_tokens = {"1", "yes", "true", "y", "positive", "pos", "churn", "fraud", "default"}
                    pos = next((u for u in uniques if str(u) in positive_tokens), None)
                    if pos is None:
                        pos = sorted([str(u) for u in uniques])[1]
                    y = (normalized == str(pos)).astype("int8")
                else:
                    y = pd.to_numeric(target_series, errors="coerce")

        y = pd.to_numeric(y, errors="coerce").fillna(0).astype("int8")
        return df, y
    else:
        # if no target column, just return the original df and None for y -> hidden dataset
        return df, None
