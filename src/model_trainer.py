import os
import gc
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from sklearn.metrics import average_precision_score, precision_recall_curve

# Avoid noisy loky core-detection warnings on some Windows environments.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

def _as_dataframe(X):
    if isinstance(X, pd.DataFrame):
        return X.copy()
    return pd.DataFrame(X)


def _as_polars_dataframe(X):
    if isinstance(X, pl.DataFrame):
        return X.clone()
    if isinstance(X, pd.DataFrame):
        return pl.from_pandas(X, include_index=False)
    return pl.DataFrame(X)


def _prepare_training_frame(
    X,
    categorical_feature_names: list[str] | None = None,
):
    """
    Prepare train matrix for LightGBM with a Polars-first path.
    - categorical/object/string -> stable int32 codes
    - numeric -> float32
    Returns prepared Polars frame, categorical feature names, and category maps.
    """
    out = _as_polars_dataframe(X)
    explicit = set(categorical_feature_names or [])
    inferred = set(
        c for c, dt in zip(out.columns, out.dtypes)
        if dt in (pl.String, pl.Categorical, pl.Enum, pl.Object)
    )
    cat_cols = [c for c in out.columns if c in explicit or c in inferred]
    cat_set = set(cat_cols)

    category_maps: dict[str, dict[str, int]] = {}
    exprs = []
    for col in out.columns:
        if col in cat_set:
            cat_series = (
                out
                .get_column(col)
                .cast(pl.String, strict=False)
                .fill_null("__missing__")
            )
            categories = [str(v) for v in cat_series.unique().sort().to_list()]
            cat_map = {v: i for i, v in enumerate(categories)}
            category_maps[col] = cat_map

            exprs.append(
                pl.col(col)
                .cast(pl.String, strict=False)
                .fill_null("__missing__")
                .replace(cat_map, default=-1)
                .cast(pl.Int32)
                .alias(col)
            )
        else:
            exprs.append(
                pl.col(col)
                .cast(pl.Float32, strict=False)
                .alias(col)
            )

    out = out.select(exprs).rechunk()
    return out, cat_cols, category_maps


def _prepare_inference_frame(
    X,
    train_columns: list[str],
    category_maps: dict[str, dict[str, int]],
):
    """
    Prepare inference matrix using train-time schema and category maps.
    Unknown categories map to -1.
    """
    src = _as_polars_dataframe(X)
    exprs = []

    for col in train_columns:
        if col in src.columns:
            base_expr = pl.col(col)
        else:
            base_expr = pl.lit(None)

        if col in category_maps:
            exprs.append(
                base_expr
                .cast(pl.String, strict=False)
                .fill_null("__missing__")
                .replace(category_maps[col], default=-1)
                .cast(pl.Int32)
                .alias(col)
            )
        else:
            exprs.append(
                base_expr
                .cast(pl.Float32, strict=False)
                .alias(col)
            )

    return src.select(exprs).rechunk()


def _to_float32_array(X):
    """
    Prepare a contiguous float32 matrix once so batch slicing stays zero-copy.
    """
    frame = _as_dataframe(X)
    arr = frame.to_numpy(dtype=np.float32, copy=False)
    return np.ascontiguousarray(arr, dtype=np.float32)

def _predict_scores(model, X_data):
    if hasattr(model, "predict_proba") and isinstance(X_data, np.ndarray):
        return model.predict_proba(X_data)[:, 1]
    try:
        return model.predict(X_data)
    except Exception:
        if isinstance(X_data, pl.DataFrame):
            return model.predict(X_data.to_numpy())
        return model.predict(np.asarray(X_data))

def _predict_in_batches(model, X_pl: pl.DataFrame, batch_size: int):
    n_rows = X_pl.height
    if n_rows == 0:
        return np.empty(0, dtype=np.float32)

    preds = np.empty(n_rows, dtype=np.float32)
    for start in range(0, n_rows, batch_size):
        end = min(start + batch_size, n_rows)
        batch_arrow = X_pl.slice(start, end - start).to_arrow()
        preds[start:end] = np.asarray(_predict_scores(model, batch_arrow), dtype=np.float32)
    return preds


def _build_pr_curve_payload(y_true: np.ndarray | pd.Series | None, y_scores: np.ndarray | None):
    if y_true is None or y_scores is None:
        return None
    try:
        y_arr = pd.to_numeric(pd.Series(y_true), errors="coerce").fillna(0).astype(np.int8).to_numpy()
        s_arr = np.asarray(y_scores, dtype=np.float64)
        if y_arr.size == 0 or s_arr.size == 0 or y_arr.size != s_arr.size:
            return None

        precision, recall, _ = precision_recall_curve(y_arr, s_arr)
        # Keep recall increasing for plotting consistency.
        if recall.size >= 2 and recall[0] > recall[-1]:
            recall = recall[::-1]
            precision = precision[::-1]

        return {
            "recall": np.clip(recall, 0.0, 1.0).astype(float).tolist(),
            "precision": np.clip(precision, 0.0, 1.0).astype(float).tolist(),
        }
    except Exception:
        return None

def predict_with_model(model, X_test: pd.DataFrame, predict_batch_size: int = 250000):
    """
    Predict probabilities with a trained model using memory-safe batching.
    """
    X_test_pl = _as_polars_dataframe(X_test).rechunk()
    if X_test_pl.height > predict_batch_size:
        return _predict_in_batches(model, X_test_pl, batch_size=predict_batch_size)
    return np.asarray(_predict_scores(model, X_test_pl.to_arrow()), dtype=np.float32)


def _to_target_array(y):
    if y is None:
        return None
    y_series = pd.Series(y)
    y_num = y_series.map({"Yes": 1, "No": 0, "yes": 1, "no": 0})
    if y_num.isna().any():
        y_num = pd.to_numeric(y_series, errors="coerce")
    return y_num.fillna(0).astype(np.int8).to_numpy()


def train_and_predict_raw_baseline(
    train_df_raw: pd.DataFrame,
    test_df_raw: pd.DataFrame,
    target_column: str,
    predict_batch_size: int = 250000,
    return_pr_curves: bool = False,
):
    """
    Train a strict baseline directly from raw train/test frames.
    No cleaning, feature engineering, mitigation, or schema-altering transforms are applied.
    """
    train_raw = train_df_raw.copy(deep=True)
    test_raw = test_df_raw.copy(deep=True)

    if target_column not in train_raw.columns:
        raise ValueError(f"Target column '{target_column}' not found in raw training data.")

    y_train_raw = _to_target_array(train_raw.pop(target_column))

    if target_column in test_raw.columns:
        y_test_raw = _to_target_array(test_raw.pop(target_column))
    else:
        y_test_raw = None

    baseline_weights = np.ones(len(train_raw), dtype=np.float32)

    return train_and_predict(
        train_raw,
        y_train_raw,
        test_raw,
        y_test_raw,
        baseline_weights,
        predict_batch_size=predict_batch_size,
        return_pr_curves=return_pr_curves,
    )

def train_and_predict(
    X_train,
    y_train,
    X_test,
    y_test,
    weights: np.ndarray,
    predict_batch_size: int = 250000,
    categorical_feature_names: list[str] | None = None,
    return_model: bool = False,
    return_pr_curves: bool = False,
):
    """
    Trains the AI and calculates AU-PRC.
    Dynamically checks if test answers are available (Public Test) or hidden (Evaluation).
    """
    X_train_prepared, cat_names, category_maps = _prepare_training_frame(
        X_train,
        categorical_feature_names=categorical_feature_names,
    )
    X_test_prepared = _prepare_inference_frame(
        X_test,
        train_columns=X_train_prepared.columns,
        category_maps=category_maps,
    )

    y_arr = _to_target_array(y_train)
    w_arr = np.asarray(weights, dtype=np.float32) if weights is not None else None

    gc.collect()
    train_dataset = lgb.Dataset(
        X_train_prepared.to_arrow(),
        label=y_arr,
        weight=w_arr,
        categorical_feature=cat_names if cat_names else "auto",
        free_raw_data=True,
    )
    gc.collect()

    params = {
        "verbosity": -1,
        "objective": "binary",
        "is_unbalance": True,
        "random_state": 42,
        "importance_type": "gain",
    }
    model = lgb.train(params=params, train_set=train_dataset)
    del train_dataset
    gc.collect()
    
    # Calculate AU-PRC on the full training set.
    train_preds = np.asarray(_predict_scores(model, X_train_prepared.to_arrow()), dtype=np.float32)
    train_auprc = average_precision_score(y_arr, train_preds)
    train_pr_curve = _build_pr_curve_payload(y_arr, train_preds)
    
    # Predict the final test set
    predictions = predict_with_model(model, X_test_prepared, predict_batch_size=predict_batch_size)
    
    # 2. Test-set AU-PRC only when labels are available.
    if y_test is not None:
        y_test_arr = _to_target_array(y_test)
        test_auprc = average_precision_score(y_test_arr, predictions)
        test_pr_curve = _build_pr_curve_payload(y_test_arr, predictions)
    else:
        # Hidden evaluation set has no labels; do not train an auxiliary model.
        test_auprc = None
        test_pr_curve = None

    # Return the predictions and both scores so main.py can format them
    if return_model and return_pr_curves:
        return predictions, train_auprc, test_auprc, model, train_pr_curve, test_pr_curve
    if return_model:
        return predictions, train_auprc, test_auprc, model
    if return_pr_curves:
        return predictions, train_auprc, test_auprc, train_pr_curve, test_pr_curve
    return predictions, train_auprc, test_auprc

def create_submission(test_ids: pd.Series, predictions: np.ndarray, output_path: str = "prediction.csv"):
    """
    Stitches the original IDs and the new predictions together into a final CSV using Polars for maximum speed.
    """
    # 1. Convert directly to a Polars DataFrame instead of Pandas
    submission_df_pl = pl.DataFrame({
        'CustomerID': test_ids.to_list(), 
        'probability_score': predictions
    })
    
    # 2. Write to CSV using Polars' multithreaded Rust engine
    submission_df_pl.write_csv(output_path)

    return submission_df_pl