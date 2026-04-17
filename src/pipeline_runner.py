import gc
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import numpy as np

import pandas as pd
import polars as pl
from sklearn.cluster import MiniBatchKMeans

from data_loader import split_features_and_target
from data_preprocessor import (
    apply_missing_fill_plan,
    apply_ordinal_encoder,
    clean_and_engineer_features,
)
from drift_mitigator import apply_structure_plan, apply_target_encoding_plan, apply_winsor_plan
from model_trainer import predict_with_model
from schema_utils import detect_identifier_column, detect_target_column, detect_time_column

try:
    import pyarrow.csv as pa_csv
except Exception:
    pa_csv = None


def load_previous_test_auprc(metrics_path: str = "latest_metrics.json") -> float | None:
    """
    Read previously recorded test AU-PRC if available.
    """
    if not os.path.exists(metrics_path):
        return None
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        val = payload.get("test_auprc")
        if val is None:
            return None
        out = float(val)
        return out if np.isfinite(out) else None
    except Exception:
        return None


def should_revert_feature_engineering(
    previous_test_auprc: float | None,
    current_test_auprc: float | None,
    max_drop: float = 0.01,
) -> bool:
    """
    Return True when new feature blocks should be reverted for dropping too much.
    """
    if previous_test_auprc is None or current_test_auprc is None:
        return False
    if not np.isfinite(previous_test_auprc) or not np.isfinite(current_test_auprc):
        return False
    return float(current_test_auprc) < (float(previous_test_auprc) - float(max_drop))


def _apply_missing_indicator_columns(df: pd.DataFrame, missing_indicators: list[str]):
    indicator_payload: dict[str, object] = {}
    for ind_col in missing_indicators:
        src_col = ind_col.replace("__was_missing", "")
        if src_col in df.columns:
            indicator_payload[ind_col] = df[src_col].isna().astype(np.int8)
        else:
            indicator_payload[ind_col] = 0
    if indicator_payload:
        df = df.assign(**indicator_payload)
    return df


def _compress_chunk_inplace(
    df: pd.DataFrame,
    id_col_hint: str | None = None,
    time_col_hint: str | None = None,
    target_col_hint: str | None = None,
) -> pd.DataFrame:
    """
    Lightweight in-memory compression for chunk processing.
    """
    num_cols = df.select_dtypes(include=["number"]).columns
    for col in num_cols:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer", errors="coerce")
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float", errors="coerce")

    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    id_col = id_col_hint if (id_col_hint and id_col_hint in df.columns) else detect_identifier_column(df)
    time_col = time_col_hint if (time_col_hint and time_col_hint in df.columns) else detect_time_column(df)
    target_col = target_col_hint if (target_col_hint and target_col_hint in df.columns) else detect_target_column(df)
    protected_cols = {c for c in [id_col, time_col, target_col] if c}
    for col in obj_cols:
        if col in protected_cols:
            continue
        sample = df[col].dropna().head(5000)
        if not sample.empty:
            uniq_ratio = sample.nunique(dropna=True) / max(1, len(sample))
            if uniq_ratio <= 0.25:
                try:
                    df[col] = df[col].astype("category")
                except Exception:
                    pass
    return df


def _iter_partition_chunks(csv_path: str, initial_chunk_rows: int):
    """
    Yield dataframe partitions with adaptive fallback if parser OOM occurs.
    """
    chunk_rows = max(10000, int(initial_chunk_rows))

    if pa_csv is not None:
        block_size = max(1 << 20, min(1 << 26, chunk_rows * 80))
        while True:
            try:
                reader = pa_csv.open_csv(
                    csv_path,
                    read_options=pa_csv.ReadOptions(block_size=block_size),
                )
                for batch in reader:
                    chunk = batch.to_pandas()
                    if not chunk.empty:
                        yield chunk
                return
            except Exception as e:
                msg = str(e).lower()
                if "out of memory" in msg and block_size > (1 << 20):
                    block_size = max(1 << 20, block_size // 2)
                    print(f"WARNING: Arrow OOM encountered. Retrying with smaller block size={block_size}.")
                    continue
                break

    while True:
        try:
            for chunk in pd.read_csv(
                csv_path,
                chunksize=chunk_rows,
                low_memory=True,
                memory_map=True,
            ):
                yield chunk
            return
        except Exception as e:
            msg = str(e).lower()
            if "out of memory" in msg and chunk_rows > 10000:
                chunk_rows = max(10000, chunk_rows // 2)
                print(f"WARNING: CSV parser OOM encountered. Retrying with smaller chunk rows={chunk_rows}.")
                continue
            raise


def build_partition_reference(
    csv_path: str,
    target_rows: int,
    chunksize: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Build an exact-size random reference sample using Polars lazy scanning.
    This avoids Python-level row loops and keeps sampling out-of-core.
    """
    _ = chunksize  # Controlled by runtime plan; retained for interface compatibility.

    if target_rows <= 0:
        return pd.DataFrame()

    lf = pl.scan_csv(csv_path, try_parse_dates=False, infer_schema_length=1000)
    total_rows = int(lf.select(pl.len().alias("_n")).collect(streaming=True).item(0, 0))
    if total_rows <= 0:
        return pd.DataFrame()

    if target_rows >= total_rows:
        return lf.collect(streaming=True).to_pandas()

    rng = np.random.default_rng(random_state)
    selected_idx = np.sort(rng.choice(total_rows, size=target_rows, replace=False))
    selected_series = pl.Series(name="_selected_idx", values=selected_idx)

    sampled = (
        lf.with_row_index("_row_idx")
        .filter(pl.col("_row_idx").is_in(selected_series))
        .drop("_row_idx")
        .collect(streaming=True)
    )

    if sampled.height > target_rows:
        sampled = sampled.sample(n=target_rows, seed=random_state)

    return sampled.to_pandas()


def _build_cluster_feature_matrix(
    df: pd.DataFrame,
    max_numeric_cols: int = 120,
    max_categorical_cols: int = 24,
) -> np.ndarray:
    num_df = df.select_dtypes(include=["number", "bool"]).copy()
    if num_df.shape[1] > max_numeric_cols:
        var_order = num_df.var(axis=0, skipna=True).sort_values(ascending=False).index[:max_numeric_cols]
        num_df = num_df[var_order]

    for col in num_df.columns:
        col_s = pd.to_numeric(num_df[col], errors="coerce")
        if col_s.isna().all():
            num_df[col] = 0.0
        else:
            num_df[col] = col_s.fillna(col_s.median())

    blocks: list[np.ndarray] = []
    if num_df.shape[1] > 0:
        blocks.append(num_df.to_numpy(dtype=np.float32, copy=False))

    cat_cols = list(df.select_dtypes(include=["object", "string", "category"]).columns)[:max_categorical_cols]
    for col in cat_cols:
        codes = df[col].astype("category").cat.codes.to_numpy(copy=False)
        codes = np.where(codes < 0, -1, codes).astype(np.float32, copy=False)
        blocks.append(codes.reshape(-1, 1))

    if not blocks:
        return np.zeros((len(df), 1), dtype=np.float32)

    X = np.concatenate(blocks, axis=1).astype(np.float32, copy=False)
    mean = np.nanmean(X, axis=0, keepdims=True)
    std = np.nanstd(X, axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    X = (X - mean) / std
    return np.nan_to_num(X, nan=0.0, posinf=6.0, neginf=-6.0)


def _allocate_cluster_quota(labels: np.ndarray, target_rows: int) -> np.ndarray:
    counts = np.bincount(labels)
    if counts.sum() <= 0:
        return np.zeros(len(counts), dtype=np.int64)

    raw = counts / counts.sum() * int(target_rows)
    quota = np.floor(raw).astype(np.int64)
    remainders = raw - quota
    remaining = int(target_rows - quota.sum())

    if remaining > 0:
        order = np.argsort(remainders)[::-1]
        for k in order[:remaining]:
            quota[k] += 1

    quota = np.minimum(quota, counts)
    deficit = int(target_rows - quota.sum())
    if deficit > 0:
        spare = counts - quota
        order = np.argsort(spare)[::-1]
        for k in order:
            if deficit <= 0:
                break
            add = int(min(spare[k], deficit))
            quota[k] += add
            deficit -= add

    return quota


def _select_cluster_distance_rows(
    df: pd.DataFrame,
    target_rows: int,
    random_state: int = 42,
    strategy: str = "furthest",
    max_clusters: int = 128,
) -> pd.DataFrame:
    selected = _select_cluster_distance_indices(
        df,
        target_rows=target_rows,
        random_state=random_state,
        strategy=strategy,
        max_clusters=max_clusters,
    )
    if selected.size <= 0:
        return df.iloc[0:0].copy()
    return df.iloc[selected].reset_index(drop=True)


def _select_cluster_distance_indices(
    df: pd.DataFrame,
    target_rows: int,
    random_state: int = 42,
    strategy: str = "furthest",
    max_clusters: int = 128,
) -> np.ndarray:
    if target_rows <= 0 or df.empty:
        return np.array([], dtype=np.int64)
    if target_rows >= len(df):
        return np.arange(len(df), dtype=np.int64)

    X = _build_cluster_feature_matrix(df)
    n_rows = X.shape[0]
    n_clusters = int(min(max_clusters, max(8, int(np.sqrt(max(1, n_rows)) // 2))))
    n_clusters = max(2, min(n_clusters, target_rows, n_rows))

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=min(4096, max(256, n_rows // 20)),
        n_init=5,
        max_iter=100,
    )
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_
    dist = np.sum((X - centers[labels]) ** 2, axis=1)

    quota = _allocate_cluster_quota(labels, target_rows)
    selected: list[int] = []
    rng = np.random.default_rng(random_state)

    for cluster_id, take_n in enumerate(quota):
        if take_n <= 0:
            continue
        idx = np.where(labels == cluster_id)[0]
        if len(idx) == 0:
            continue

        if strategy == "closest":
            ranked = idx[np.argsort(dist[idx])]
        elif strategy == "random":
            ranked = rng.permutation(idx)
        else:
            ranked = idx[np.argsort(dist[idx])[::-1]]

        selected.extend(ranked[: int(take_n)].tolist())

    if len(selected) < target_rows:
        selected_set = set(selected)
        remaining_idx = np.array([i for i in range(n_rows) if i not in selected_set], dtype=np.int64)
        if len(remaining_idx) > 0:
            if strategy == "closest":
                fill_ranked = remaining_idx[np.argsort(dist[remaining_idx])]
            elif strategy == "random":
                fill_ranked = rng.permutation(remaining_idx)
            else:
                fill_ranked = remaining_idx[np.argsort(dist[remaining_idx])[::-1]]
            need = target_rows - len(selected)
            selected.extend(fill_ranked[:need].tolist())

    selected = selected[:target_rows]
    return np.array(selected, dtype=np.int64)


def _apply_partition_cluster_pruning(
    model_df: pd.DataFrame,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict]:
    enabled = os.getenv("ENABLE_PARTITION_CLUSTER_PRUNING", "1").strip().lower() in ("1", "true", "yes", "on")
    if not enabled:
        return model_df, {"applied": False, "reason": "disabled"}
    if model_df.empty:
        return model_df, {"applied": False, "reason": "empty"}

    keep_ratio = float(os.getenv("PARTITION_CLUSTER_PRUNE_KEEP_RATIO", "0.65"))
    keep_ratio = float(np.clip(keep_ratio, 0.20, 1.0))
    if keep_ratio >= 0.999:
        return model_df, {"applied": False, "reason": "keep_ratio_1"}

    min_rows = int(os.getenv("PARTITION_CLUSTER_PRUNE_MIN_ROWS", "50000"))
    min_rows = max(10000, min_rows)

    n_rows = len(model_df)
    if n_rows <= min_rows:
        return model_df, {"applied": False, "reason": "below_min_rows", "rows": int(n_rows)}

    target_rows = int(max(min_rows, min(n_rows, int(round(n_rows * keep_ratio)))))
    if target_rows >= n_rows:
        return model_df, {"applied": False, "reason": "target_not_smaller", "rows": int(n_rows)}

    pool_cap = int(os.getenv("PARTITION_CLUSTER_PRUNE_MAX_POOL_ROWS", "300000"))
    pool_cap = max(50000, pool_cap)

    if n_rows > pool_cap:
        pool_df = model_df.sample(n=pool_cap, random_state=random_state).reset_index(drop=True)
    else:
        pool_df = model_df

    target_rows = min(target_rows, len(pool_df))
    strategy = os.getenv("PARTITION_CLUSTER_PRUNE_STRATEGY", "furthest").strip().lower()
    if strategy not in {"furthest", "closest", "random"}:
        strategy = "furthest"

    max_clusters = int(os.getenv("PARTITION_CLUSTER_PRUNE_MAX_CLUSTERS", "128"))
    max_clusters = max(8, min(512, max_clusters))

    try:
        pruned_df = _select_cluster_distance_rows(
            pool_df,
            target_rows=target_rows,
            random_state=random_state,
            strategy=strategy,
            max_clusters=max_clusters,
        )
    except Exception as e:
        print(f"WARNING: Partition cluster pruning failed; using unpruned sample. Reason: {e}")
        return model_df, {"applied": False, "reason": f"error:{e}"}

    return pruned_df, {
        "applied": True,
        "strategy": strategy,
        "keep_ratio": float(keep_ratio),
        "original_rows": int(n_rows),
        "pool_rows": int(len(pool_df)),
        "sampled_rows": int(len(pruned_df)),
        "max_clusters": int(max_clusters),
    }


def apply_inmemory_cluster_pruning(
    X_train_model: pd.DataFrame,
    y_train_model: pd.Series,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    enabled = os.getenv("ENABLE_INMEM_CLUSTER_PRUNING", "0").strip().lower() in ("1", "true", "yes", "on")
    if not enabled:
        return X_train_model, y_train_model, {"applied": False, "reason": "disabled"}
    if X_train_model.empty or y_train_model is None or len(X_train_model) != len(y_train_model):
        return X_train_model, y_train_model, {"applied": False, "reason": "empty-or-invalid-target"}

    keep_ratio = float(os.getenv("PARTITION_CLUSTER_PRUNE_KEEP_RATIO", "0.65"))
    keep_ratio = float(np.clip(keep_ratio, 0.20, 1.0))
    if keep_ratio >= 0.999:
        return X_train_model, y_train_model, {"applied": False, "reason": "keep_ratio_1"}

    min_rows = int(os.getenv("PARTITION_CLUSTER_PRUNE_MIN_ROWS", "50000"))
    min_rows = max(10000, min_rows)

    n_rows = len(X_train_model)
    if n_rows <= min_rows:
        return X_train_model, y_train_model, {"applied": False, "reason": "below_min_rows", "rows": int(n_rows)}

    target_rows = int(max(min_rows, min(n_rows, int(round(n_rows * keep_ratio)))))
    if target_rows >= n_rows:
        return X_train_model, y_train_model, {"applied": False, "reason": "target_not_smaller", "rows": int(n_rows)}

    strategy = os.getenv("PARTITION_CLUSTER_PRUNE_STRATEGY", "furthest").strip().lower()
    if strategy not in {"furthest", "closest", "random"}:
        strategy = "furthest"

    max_clusters = int(os.getenv("PARTITION_CLUSTER_PRUNE_MAX_CLUSTERS", "128"))
    max_clusters = max(8, min(512, max_clusters))

    y_num = pd.to_numeric(pd.Series(y_train_model), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    pos_mask = y_num > 0.5
    pos_idx = np.where(pos_mask)[0]
    neg_idx = np.where(~pos_mask)[0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        selected = _select_cluster_distance_indices(
            X_train_model,
            target_rows=target_rows,
            random_state=random_state,
            strategy=strategy,
            max_clusters=max_clusters,
        )
    else:
        pos_rate = float(len(pos_idx) / max(1, n_rows))
        n_pos_target = int(round(target_rows * pos_rate))
        n_pos_target = min(max(1, n_pos_target), len(pos_idx))
        n_neg_target = min(max(1, target_rows - n_pos_target), len(neg_idx))

        pos_sel_rel = _select_cluster_distance_indices(
            X_train_model.iloc[pos_idx],
            target_rows=n_pos_target,
            random_state=random_state,
            strategy=strategy,
            max_clusters=max_clusters,
        )
        neg_sel_rel = _select_cluster_distance_indices(
            X_train_model.iloc[neg_idx],
            target_rows=n_neg_target,
            random_state=random_state + 1,
            strategy=strategy,
            max_clusters=max_clusters,
        )
        selected = np.concatenate([pos_idx[pos_sel_rel], neg_idx[neg_sel_rel]])

    rng = np.random.default_rng(random_state)
    if len(selected) > target_rows:
        selected = selected[:target_rows]
    rng.shuffle(selected)

    X_out = X_train_model.iloc[selected].reset_index(drop=True)
    y_out = y_train_model.iloc[selected].reset_index(drop=True)

    return X_out, y_out, {
        "applied": True,
        "strategy": strategy,
        "keep_ratio": float(keep_ratio),
        "original_rows": int(n_rows),
        "sampled_rows": int(len(X_out)),
        "max_clusters": int(max_clusters),
        "original_positive_rate": float(np.mean(y_num > 0.5)),
        "sampled_positive_rate": float(pd.to_numeric(y_out, errors="coerce").fillna(0.0).gt(0.5).mean()),
    }


def build_partition_train_slices(
    train_csv_path: str,
    test_csv_path: str,
    detection_train_rows: int,
    model_train_rows: int,
    detection_test_rows: int,
    chunksize: int,
    random_state: int = 42,
):
    """
    Build separate partition samples:
    - small detection sample for drift detection/temporal scoring
    - larger model-train sample for final model fitting
    - test reference sample for fast evaluation/drift comparison
    """
    model_df = build_partition_reference(
        csv_path=train_csv_path,
        target_rows=model_train_rows,
        chunksize=chunksize,
        random_state=random_state,
    )
    if model_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    model_df, prune_meta = _apply_partition_cluster_pruning(model_df, random_state=random_state)
    if prune_meta.get("applied", False):
        print(
            "SUCCESS: Partition cluster-distance pruning applied "
            f"({prune_meta.get('sampled_rows')}/{prune_meta.get('original_rows')} rows, "
            f"strategy={prune_meta.get('strategy')})."
        )
    else:
        print(f"SUCCESS: Partition cluster-distance pruning skipped ({prune_meta.get('reason')}).")

    detect_n = min(max(1, int(detection_train_rows)), len(model_df))
    detect_df = model_df.sample(n=detect_n, random_state=random_state + 1).reset_index(drop=True)

    test_df = build_partition_reference(
        csv_path=test_csv_path,
        target_rows=detection_test_rows,
        chunksize=chunksize,
        random_state=random_state + 2,
    )

    return detect_df, model_df, test_df


def _process_and_predict_chunk(
    chunk_idx: int,
    chunk: pd.DataFrame,
    model,
    missing_indicators: list[str],
    fill_plan: dict,
    structure_plan: dict,
    winsor_plan: dict,
    te_plan: dict,
    pre_encode_cols: list[str],
    encoder,
    cat_cols: list[str],
    final_model_cols: list[str],
    id_col_hint: str | None,
    time_col_hint: str | None,
    target_col_hint: str,
    predict_batch_size: int,
) -> tuple[int, str, np.ndarray | None, np.ndarray, int]:
    chunk = _compress_chunk_inplace(
        chunk,
        id_col_hint=id_col_hint,
        time_col_hint=time_col_hint,
        target_col_hint=target_col_hint,
    )

    resolved_id_col = id_col_hint
    out_id_name = resolved_id_col or "row_id"
    if resolved_id_col is not None and resolved_id_col in chunk.columns:
        ids_values = chunk[resolved_id_col].values
    else:
        ids_values = None

    X_chunk, _ = split_features_and_target(chunk, target_column=target_col_hint)

    X_chunk = _apply_missing_indicator_columns(X_chunk, missing_indicators)
    X_chunk = apply_missing_fill_plan(X_chunk, fill_plan)
    X_chunk = clean_and_engineer_features(X_chunk)
    X_chunk = apply_structure_plan(X_chunk, structure_plan)
    X_chunk = apply_winsor_plan(X_chunk, winsor_plan, align_test_distribution=True)
    X_chunk = apply_target_encoding_plan(X_chunk, te_plan)

    if time_col_hint:
        X_chunk = X_chunk.drop(columns=[time_col_hint], errors="ignore")

    X_chunk = X_chunk.reindex(columns=pre_encode_cols, fill_value=0)
    X_chunk = apply_ordinal_encoder(X_chunk, encoder, cat_cols)

    X_chunk = X_chunk.reindex(columns=final_model_cols, fill_value=0)

    preds = predict_with_model(model, X_chunk, predict_batch_size=predict_batch_size)
    preds = np.asarray(preds)
    if ids_values is not None and len(preds) != len(ids_values):
        raise ValueError(
            f"Prediction length mismatch in chunk {chunk_idx}: preds={len(preds)} ids={len(ids_values)}"
        )

    row_count = int(len(preds))
    del chunk, X_chunk
    return chunk_idx, out_id_name, ids_values, preds, row_count


def partition_predict_full_test(
    test_path: str,
    model,
    missing_indicators: list[str],
    fill_plan: dict,
    structure_plan: dict,
    winsor_plan: dict,
    te_plan: dict,
    pre_encode_cols: list[str],
    encoder,
    cat_cols: list[str],
    final_model_cols: list[str],
    id_col_hint: str | None = None,
    time_col_hint: str | None = None,
    target_col_hint: str = "ChurnStatus",
    predict_batch_size: int = 250000,
    chunksize: int = 200000,
    output_path: str = "prediction.csv",
):
    """
    Stream full test CSV in chunks, apply mitigation/encoding pipeline,
    and write predictions once via an in-memory, threaded execution path.
    """
    all_ids: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    total_rows = 0
    chunk_idx = 0
    resolved_id_col = id_col_hint
    resolved_time_col = time_col_hint
    out_id_name = resolved_id_col or "row_id"

    max_workers = int(os.getenv("PREDICT_THREAD_WORKERS", str(max(1, min(2, os.cpu_count() or 1)))))
    max_workers = max(1, min(2, max_workers))
    max_inflight = max_workers * 2

    csv_batch_size_raw = os.getenv("PREDICTION_CSV_BATCH_SIZE", "262144")
    try:
        csv_batch_size = max(1024, int(csv_batch_size_raw))
    except Exception:
        csv_batch_size = 65536
    csv_kwargs: dict[str, object] = {"batch_size": csv_batch_size}
    csv_float_precision_raw = os.getenv("PREDICTION_CSV_FLOAT_PRECISION", "").strip()
    if csv_float_precision_raw:
        try:
            csv_kwargs["float_precision"] = max(1, int(csv_float_precision_raw))
        except Exception:
            pass

    gc_interval_raw = os.getenv("PREDICT_GC_INTERVAL_CHUNKS", "8")
    try:
        gc_interval = max(1, int(gc_interval_raw))
    except Exception:
        gc_interval = 8

    pending_results: dict[int, tuple[str, np.ndarray | None, np.ndarray, int]] = {}
    next_chunk_to_flush = 1

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: dict[object, int] = {}

        def _drain_ready() -> None:
            nonlocal total_rows, out_id_name, next_chunk_to_flush
            if not futures:
                return
            done, _ = wait(
                list(futures.keys()),
                return_when=FIRST_COMPLETED,
            )
            for fut in done:
                idx = futures.pop(fut)
                result_idx, id_name, ids_values, preds, row_count = fut.result()
                if result_idx != idx:
                    raise RuntimeError(f"Chunk index mismatch: expected {idx}, got {result_idx}")
                pending_results[idx] = (id_name, ids_values, preds, row_count)

            while next_chunk_to_flush in pending_results:
                id_name, ids_values, preds, row_count = pending_results.pop(next_chunk_to_flush)
                out_id_name = id_name or out_id_name

                if ids_values is None:
                    ids_values = np.arange(total_rows, total_rows + row_count)
                all_ids.append(np.asarray(ids_values))
                all_preds.append(np.asarray(preds))

                total_rows += int(row_count)
                if (next_chunk_to_flush % 10) == 0:
                    print(f"INFO: Prediction in-memory progress: chunks={next_chunk_to_flush}, rows_ready={total_rows:,}")
                if (next_chunk_to_flush % gc_interval) == 0:
                    gc.collect()
                next_chunk_to_flush += 1

        for chunk in _iter_partition_chunks(test_path, initial_chunk_rows=chunksize):
            chunk_idx += 1
            if chunk_idx == 1:
                if resolved_id_col is None or resolved_id_col not in chunk.columns:
                    resolved_id_col = "CustomerID" if "CustomerID" in chunk.columns else detect_identifier_column(chunk)
                    out_id_name = resolved_id_col or out_id_name
                if resolved_time_col is None or resolved_time_col not in chunk.columns:
                    resolved_time_col = "Month" if "Month" in chunk.columns else detect_time_column(chunk)

            fut = executor.submit(
                _process_and_predict_chunk,
                chunk_idx,
                chunk,
                model,
                missing_indicators,
                fill_plan,
                structure_plan,
                winsor_plan,
                te_plan,
                pre_encode_cols,
                encoder,
                cat_cols,
                final_model_cols,
                resolved_id_col,
                resolved_time_col,
                target_col_hint,
                predict_batch_size,
            )
            futures[fut] = chunk_idx

            if len(futures) >= max_inflight:
                _drain_ready()

        while futures:
            _drain_ready()

    write_start = time.perf_counter()
    if all_ids and all_preds:
        merged_ids = np.concatenate(all_ids, axis=0)
        merged_preds = np.concatenate(all_preds, axis=0)
        pl.DataFrame(
            {
                out_id_name: merged_ids,
                "probability_score": merged_preds,
            }
        ).write_csv(output_path, **csv_kwargs)
    else:
        pl.DataFrame(
            {
                out_id_name: np.array([], dtype=np.int64),
                "probability_score": np.array([], dtype=np.float64),
            }
        ).write_csv(output_path, **csv_kwargs)

    write_duration = time.perf_counter() - write_start
    print(
        "INFO: Prediction CSV write complete: "
        f"rows={total_rows:,}, seconds={write_duration:.2f}, batch_size={csv_batch_size}"
    )

    gc.collect()

    return total_rows
