import numpy as np
import pandas as pd
from config import PIPELINE_CONFIG
from schema_utils import detect_time_column


def _safe_ratio_map(train_series: pd.Series, test_series: pd.Series) -> dict:
    train_freq = train_series.value_counts(normalize=True, dropna=False)
    test_freq = test_series.value_counts(normalize=True, dropna=False)
    keys = set(train_freq.index).union(set(test_freq.index))
    ratio_map = {}
    eps = 1e-6
    for k in keys:
        tr = float(train_freq.get(k, 0.0))
        te = float(test_freq.get(k, 0.0))
        ratio = (te + eps) / (tr + eps)
        ratio_map[k] = float(np.clip(ratio, 0.50, 2.50))
    return ratio_map


def _choose_target_rows(n_rows: int) -> int:
    # Keep enough rows for stability while reducing compute on large sets.
    # Hard safety cap: strategic sampling should never target above 1.2x observed rows.
    if n_rows <= 20_000:
        proposed = n_rows
    elif n_rows <= 100_000:
        proposed = int(n_rows * 0.75)
    elif n_rows <= 1_000_000:
        proposed = int(n_rows * 0.55)
    else:
        proposed = int(n_rows * 0.40)

    max_allowed = max(1, int(n_rows * 1.2))
    return int(min(proposed, max_allowed))


def apply_strategic_sampling(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test_ref: pd.DataFrame,
    drift_report: dict,
    feature_buckets: dict,
    random_state: int = 42,
):
    """
    Drift-aware strategic sampling on the training set.
    Returns sampled (X_train, y_train, metadata).
    """
    n_rows = len(X_train)
    if y_train is None or n_rows == 0:
        return X_train, y_train, {"applied": False, "reason": "missing-target-or-empty"}

    target_rows = _choose_target_rows(n_rows)
    if target_rows >= n_rows:
        return X_train, y_train, {"applied": False, "reason": "small-dataset", "rows": n_rows}

    rng = np.random.default_rng(random_state)
    score = np.ones(n_rows, dtype=np.float64)

    # 1) Temporal alignment.
    # REFACTORED: centralized time column name from shared config.
    time_col = PIPELINE_CONFIG.time_column if PIPELINE_CONFIG.time_column in X_train.columns else detect_time_column(X_train)
    if time_col and time_col in X_train.columns and time_col in X_test_ref.columns:
        month_ratio = _safe_ratio_map(X_train[time_col], X_test_ref[time_col])
        month_vals = X_train[time_col].astype("object").map(month_ratio)
        score *= month_vals.fillna(1.0).to_numpy(dtype=np.float64, copy=False)

    drift_cols = [c for c in drift_report.keys() if c in X_train.columns and c in X_test_ref.columns]
    cat_cols = set(feature_buckets.get("binary", [])) | set(feature_buckets.get("cat_low", [])) | set(feature_buckets.get("cat_high", []))
    num_cols = set(feature_buckets.get("numerical", []))

    # 2) Categorical/binary drift alignment.
    drift_cat_cols = [c for c in drift_cols if c in cat_cols][:12]
    for col in drift_cat_cols:
        ratio_map = _safe_ratio_map(X_train[col], X_test_ref[col])
        col_vals = X_train[col].astype("object").map(ratio_map)
        col_factor = col_vals.fillna(1.0).to_numpy(dtype=np.float64, copy=False)
        score *= np.clip(col_factor, 0.70, 1.80)

    # 3) Numerical drift alignment.
    drift_num_cols = [c for c in drift_cols if c in num_cols][:12]
    if drift_num_cols:
        train_num_df = X_train[drift_num_cols].apply(pd.to_numeric, errors="coerce")
        test_num_df = X_test_ref[drift_num_cols].apply(pd.to_numeric, errors="coerce")

        q10 = test_num_df.quantile(0.10)
        q90 = test_num_df.quantile(0.90)
        meds = test_num_df.median()
        spreads = (q90 - q10)

        valid_mask = spreads.to_numpy(dtype=np.float64, copy=False) > 1e-9
        valid_cols = spreads.index[valid_mask].tolist()

        if valid_cols:
            mat = train_num_df[valid_cols].to_numpy(dtype=np.float32, copy=False)
            med_arr = meds[valid_cols].to_numpy(dtype=np.float32, copy=False)
            spread_arr = spreads[valid_cols].to_numpy(dtype=np.float32, copy=False)

            # Vectorized matrix path: compute per-cell modifiers in one pass.
            z = np.abs(mat - med_arr[None, :]) / (spread_arr[None, :] + 1e-9)
            modifiers = np.clip(1.0 + np.exp(-z), 0.80, 1.80)
            row_factor = np.prod(np.nan_to_num(modifiers, nan=1.0, posinf=1.8, neginf=0.8), axis=1)
            score *= row_factor.astype(np.float64, copy=False)

    score = np.nan_to_num(score, nan=1.0, posinf=2.0, neginf=0.5)
    score = np.clip(score, 1e-6, None)

    y_np = y_train.to_numpy(dtype=np.int8, copy=False)
    idx_all = np.arange(n_rows)
    pos_idx = idx_all[y_np == 1]
    neg_idx = idx_all[y_np == 0]

    pos_rate = float(y_train.mean())
    n_pos_target = int(round(target_rows * pos_rate))
    n_pos_target = min(max(1, n_pos_target), len(pos_idx))
    n_neg_target = min(max(1, target_rows - n_pos_target), len(neg_idx))

    pos_probs = score[pos_idx] / score[pos_idx].sum()
    neg_probs = score[neg_idx] / score[neg_idx].sum()

    pos_pick = rng.choice(pos_idx, size=n_pos_target, replace=False, p=pos_probs)
    neg_pick = rng.choice(neg_idx, size=n_neg_target, replace=False, p=neg_probs)

    keep_idx = np.concatenate([pos_pick, neg_pick])
    rng.shuffle(keep_idx)

    X_out = X_train.iloc[keep_idx].reset_index(drop=True)
    y_out = y_train.iloc[keep_idx].reset_index(drop=True)

    metadata = {
        "applied": True,
        "original_rows": int(n_rows),
        "sampled_rows": int(len(X_out)),
        "sample_fraction": float(len(X_out) / max(1, n_rows)),
        "target_rows": int(target_rows),
        "original_positive_rate": float(pos_rate),
        "sampled_positive_rate": float(y_out.mean()),
    }
    return X_out, y_out, metadata
