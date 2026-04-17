import pandas as pd
import numpy as np
import polars as pl
import os
from scipy.spatial import cKDTree
from scipy import stats
from config import PIPELINE_CONFIG
from schema_utils import detect_identifier_column, detect_time_column
from utils import _env_truthy


def _drift_entry(drift_report: dict, col: str) -> dict:
    # REFACTORED: structured drift payload lookup.
    entry = drift_report.get(col, {})
    return entry if isinstance(entry, dict) else {}


def _metric_score(entry: dict, metric: str) -> float:
    metrics = entry.get("metrics", {}) if isinstance(entry, dict) else {}
    raw = metrics.get(metric, 0.0)
    if isinstance(raw, dict):
        raw = raw.get("score", 0.0)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _categorical_drift_score(entry: dict) -> float:
    return max(_metric_score(entry, "jsd"), _metric_score(entry, "cramers_v"))


def _binary_shift_score(entry: dict) -> float:
    return _metric_score(entry, "binary_max_diff")


def _distance_score(entry: dict) -> float:
    try:
        base = float(entry.get("distance_score", 0.0))
    except (TypeError, ValueError, AttributeError):
        base = 0.0
    fallback = max(
        _metric_score(entry, "ks_stat"),
        _metric_score(entry, "psi"),
        _metric_score(entry, "nz_ks_stat"),
        _metric_score(entry, "nz_psi"),
        _metric_score(entry, "cramers_v"),
        _metric_score(entry, "jsd"),
        _metric_score(entry, "binary_max_diff"),
        _metric_score(entry, "zero_mass_diff"),
    )
    return max(base, fallback)


def _is_numerical_distribution_drift(entry: dict) -> bool:
    return any(
        _metric_score(entry, metric) > 0.0
        for metric in ("ks_stat", "psi", "nz_ks_stat", "nz_psi", "zero_mass_diff")
    )

def _to_numeric_target(y_train: pd.Series):
    y_num = y_train.map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})
    if y_num.isna().any():
        try:
            y_num = y_train.astype(float)
        except (ValueError, TypeError):
            y_num = (y_train == y_train.iloc[0]).astype(int)
    return y_num.astype(float)


def apply_fast_target_encoding(
    pl_df: pl.LazyFrame,
    target_col: str,
    cat_cols: list,
    smoothing: float = 1000.0,
) -> pl.LazyFrame:
    """
    Rust-accelerated smoothed target encoding via Polars window expressions.
    """
    global_mean_expr = pl.col(target_col).mean()

    exprs = []
    for col in cat_cols:
        cat_count = pl.col(target_col).count().cast(pl.Float64).over(col)
        cat_mean = pl.col(target_col).mean().over(col)
        smoothed_te = ((cat_count * cat_mean) + (float(smoothing) * global_mean_expr)) / (cat_count + float(smoothing))
        exprs.append(smoothed_te.alias(col))

    return pl_df.with_columns(exprs)

def _sorted_time_values(time_series: pd.Series):
    as_str = time_series.astype(str)
    try:
        parsed = pd.to_datetime(as_str, errors='coerce', format='mixed')
    except Exception:
        parsed = pd.to_datetime(as_str, errors='coerce')
    helper = pd.DataFrame({'raw': as_str, 'parsed': parsed}).drop_duplicates('raw')
    helper = helper.sort_values('parsed', kind='stable')
    return helper['raw'].tolist()


def _build_recency_weights(time_series: pd.Series, strength: float = 2.0) -> pd.Series:
    """
    Build normalized recency weights in [exp(0), exp(strength)] by inferred time order.
    """
    if time_series.empty:
        return pd.Series(dtype=np.float64)

    ordered_times = _sorted_time_values(time_series)
    if len(ordered_times) < 2:
        return pd.Series(np.ones(len(time_series), dtype=np.float64), index=time_series.index)

    denom = float(max(1, len(ordered_times) - 1))
    rank_map = {t: i / denom for i, t in enumerate(ordered_times)}
    ranks = pd.to_numeric(
        time_series.astype("string").map(rank_map),
        errors="coerce",
    ).fillna(0.5).to_numpy(dtype=np.float64, copy=False)

    safe_strength = float(np.clip(strength, -4.0, 4.0))
    weights = np.exp(np.clip(ranks, 0.0, 1.0) * safe_strength)
    weights = weights / max(1e-12, float(np.mean(weights)))
    return pd.Series(weights.astype(np.float64, copy=False), index=time_series.index)


def _categorical_temporal_instability_score(
    category_series: pd.Series,
    y_num: pd.Series,
    time_series: pd.Series,
) -> float:
    """
    Weighted early-vs-late target-rate drift for a categorical feature.
    Returns 0 when temporal structure is insufficient.
    """
    if len(category_series) == 0 or len(y_num) == 0 or len(time_series) == 0:
        return 0.0

    ordered_times = _sorted_time_values(time_series)
    if len(ordered_times) < 6:
        return 0.0

    split_idx = max(1, int(len(ordered_times) * 0.6))
    split_idx = min(split_idx, len(ordered_times) - 1)
    early_times = set(ordered_times[:split_idx])
    late_times = set(ordered_times[split_idx:])
    if not early_times or not late_times:
        return 0.0

    time_arr = time_series.astype("string").to_numpy(copy=False)
    early_mask = np.isin(time_arr, list(early_times))
    late_mask = np.isin(time_arr, list(late_times))
    if int(np.sum(early_mask)) < 200 or int(np.sum(late_mask)) < 200:
        return 0.0

    cat_all = category_series.astype("string").fillna("__missing__")
    global_mean = float(np.mean(pd.to_numeric(y_num, errors="coerce").fillna(0.0)))

    early_df = pd.DataFrame(
        {
            "__cat": cat_all[early_mask],
            "__target": pd.to_numeric(y_num[early_mask], errors="coerce").fillna(global_mean),
        }
    )
    late_df = pd.DataFrame(
        {
            "__cat": cat_all[late_mask],
            "__target": pd.to_numeric(y_num[late_mask], errors="coerce").fillna(global_mean),
        }
    )

    early_rate = early_df.groupby("__cat", observed=False)["__target"].mean()
    late_rate = late_df.groupby("__cat", observed=False)["__target"].mean()
    aligned = pd.concat([early_rate, late_rate], axis=1, keys=["early", "late"]).fillna(global_mean)
    if aligned.empty:
        return 0.0

    freq = cat_all.value_counts(normalize=True)
    mapped_w = pd.Series(aligned.index.map(freq), index=aligned.index)
    w = pd.to_numeric(mapped_w, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    if w.sum() <= 0:
        return 0.0
    w = w / w.sum()

    instability = float(np.sum(np.abs(aligned["early"].values - aligned["late"].values) * w))
    if not np.isfinite(instability):
        return 0.0
    return float(np.clip(instability, 0.0, 1.0))


def add_categorical_shift_ratio_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    drift_scores: dict,
    feature_buckets: dict,
    max_cols: int = 8,
    eps: float = 1e-6,
):
    """
    Add dynamic numeric features that encode train-vs-test category prior shift.
    Each feature is log((p_test(cat)+eps)/(p_train(cat)+eps)) mapped per row.
    """
    if X_train.empty or X_test.empty or max_cols <= 0:
        return X_train, X_test, []

    available_cols = set(X_train.columns).intersection(set(X_test.columns))
    candidate_scores: list[tuple[str, float]] = []
    include_high_card = _env_truthy("CATEGORY_SHIFT_RATIO_INCLUDE_HIGH_CARD", "0")

    for col, meta in drift_scores.items():
        if col not in available_cols:
            continue
        ft = str(meta.get("feature_type", ""))
        allowed_feature_types = {"binary", "cat_low"}
        if include_high_card:
            allowed_feature_types.add("cat_high")
        if ft not in allowed_feature_types:
            continue
        if not bool(meta.get("flagged", False)):
            continue
        if str(col).endswith("__was_missing"):
            continue
        score = float(meta.get("distance_score", 0.0))
        candidate_scores.append((col, score))

    if not candidate_scores:
        fallback_cols: list[str] = []
        fallback_buckets = ["binary", "cat_low"]
        if include_high_card:
            fallback_buckets.append("cat_high")
        for bucket in fallback_buckets:
            for col in feature_buckets.get(bucket, []):
                if col in available_cols:
                    if str(col).endswith("__was_missing"):
                        continue
                    fallback_cols.append(col)
        # preserve order while removing duplicates
        seen = set()
        unique_fallback = []
        for col in fallback_cols:
            if col in seen:
                continue
            seen.add(col)
            unique_fallback.append(col)
        candidate_scores = [(col, 0.0) for col in unique_fallback]

    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    selected_cols = [col for col, _ in candidate_scores[:max_cols]]
    if not selected_cols:
        return X_train, X_test, []

    added_cols: list[str] = []
    safe_eps = float(max(1e-9, eps))

    for col in selected_cols:
        tr = X_train[col].astype("string").fillna("__missing__")
        te = X_test[col].astype("string").fillna("__missing__")

        p_train = tr.value_counts(normalize=True)
        p_test = te.value_counts(normalize=True)
        cats = p_train.index.union(p_test.index)
        p = p_train.reindex(cats, fill_value=0.0)
        q = p_test.reindex(cats, fill_value=0.0)

        shift_log_ratio = np.log((q + safe_eps) / (p + safe_eps))
        # Shrink rare-category shifts to reduce instability and overfitting risk.
        support = np.sqrt(np.clip(p + q, 0.0, 1.0))
        shift_log_ratio = (shift_log_ratio * support).clip(-3.0, 3.0)

        feature_name = f"{col}__shift_log_ratio"
        X_train[feature_name] = (
            pd.to_numeric(tr.map(shift_log_ratio), errors="coerce")
            .fillna(0.0)
            .astype(np.float32)
        )
        X_test[feature_name] = (
            pd.to_numeric(te.map(shift_log_ratio), errors="coerce")
            .fillna(0.0)
            .astype(np.float32)
        )
        added_cols.append(feature_name)

    return X_train, X_test, added_cols

def mitigate_structure(X_train: pd.DataFrame, X_test: pd.DataFrame, feature_buckets: dict, drift_report: dict):
    """
    Fixes structural issues: Drops corrupted IDs and bins rare categories.
    Now includes adaptive feature pruning for noisy, drifted high-cardinality features.
    """
    id_col = (
        PIPELINE_CONFIG.id_column
        if (PIPELINE_CONFIG.id_column in X_train.columns or PIPELINE_CONFIG.id_column in X_test.columns)
        else (detect_identifier_column(X_train) or detect_identifier_column(X_test))
    )
    if id_col is not None:
        X_train = X_train.drop(columns=[id_col], errors='ignore')
        X_test = X_test.drop(columns=[id_col], errors='ignore')

    for col in feature_buckets.get("cat_high", []):
        # High-card rescue mode: keep column and only collapse rare labels.
        if col in X_train.columns and col in X_test.columns:
            value_counts = X_train[col].value_counts(normalize=True)
            rare_labels = value_counts[value_counts < 0.01].index

            if isinstance(X_train[col].dtype, pd.CategoricalDtype) and 'other_rare' not in X_train[col].cat.categories:
                X_train[col] = X_train[col].cat.add_categories(['other_rare'])
            if isinstance(X_test[col].dtype, pd.CategoricalDtype) and 'other_rare' not in X_test[col].cat.categories:
                X_test[col] = X_test[col].cat.add_categories(['other_rare'])

            X_train.loc[X_train[col].isin(rare_labels), col] = 'other_rare'
            X_test.loc[X_test[col].isin(rare_labels), col] = 'other_rare'
            
    return X_train, X_test

def apply_stability_gated_pruning(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    drift_report: dict,
    max_drops: int = 2
):
    """
    Conservative pruning pass for toxic high-cardinality drift only.
    Binary / numerical / low-card features are never hard-dropped here.
    """
    if not _env_truthy("ENABLE_STABILITY_GATED_PRUNING_PASS", "0"):
        # Default behavior preserves the current high-card rescue path.
        return X_train, X_test, []

    time_col = PIPELINE_CONFIG.time_column if PIPELINE_CONFIG.time_column in X_train.columns else detect_time_column(X_train)
    if time_col is None or time_col not in X_train.columns:
        return X_train, X_test, []

    time_values = _sorted_time_values(X_train[time_col])
    if len(time_values) < 6:
        return X_train, X_test, []

    split_idx = int(len(time_values) * 0.6)
    early_times = set(time_values[:split_idx])
    late_times = set(time_values[split_idx:])

    y_num = _to_numeric_target(y_train)
    y_arr = y_num.to_numpy(dtype=np.float64, copy=False)
    time_arr = X_train[time_col].astype(str).to_numpy(copy=False)
    early_mask = np.isin(time_arr, list(early_times))
    late_mask = np.isin(time_arr, list(late_times))

    dropped = []
    scores = []
    drift_base_blend = float(np.clip(float(os.getenv("STABILITY_DRIFT_BASE_BLEND", "0.15")), 0.0, 0.8))

    for col in drift_report:
        if col not in X_train.columns or col == time_col:
            continue
        drift_entry = _drift_entry(drift_report, col)
        drift_hint = max(
            _categorical_drift_score(drift_entry),
            _binary_shift_score(drift_entry),
            _distance_score(drift_entry),
        )
        if drift_hint <= 0.0:
            continue

        if not early_mask.any() or not late_mask.any():
            continue

        series_train = X_train[col]
        series_test = X_test[col] if col in X_test.columns else pd.Series(dtype=float)

        # Drift severity from train-test marginal difference.
        if pd.api.types.is_numeric_dtype(series_train):
            tr = series_train.dropna()
            te = series_test.dropna()
            if tr.empty or te.empty:
                continue
            if len(tr) > 150000:
                tr = tr.sample(n=150000, random_state=42)
            if len(te) > 150000:
                te = te.sample(n=150000, random_state=42)
            tr_arr = tr.to_numpy(dtype=np.float64, copy=False)
            te_arr = te.to_numpy(dtype=np.float64, copy=False)
            drift_severity = stats.ks_2samp(tr_arr, te_arr, method="asymp")[0]

            # Temporal instability: sign-flip + magnitude change in correlation with target.
            e_vals = pd.to_numeric(
                X_train.loc[early_mask, col],
                errors="coerce",
            )
            l_vals = pd.to_numeric(
                X_train.loc[late_mask, col],
                errors="coerce",
            )
            e_vals = e_vals.fillna(e_vals.median()).to_numpy(dtype=np.float64, copy=False)
            l_vals = l_vals.fillna(l_vals.median()).to_numpy(dtype=np.float64, copy=False)
            e_y = y_arr[early_mask]
            l_y = y_arr[late_mask]
            if len(e_vals) < 2 or len(l_vals) < 2:
                continue
            corr_early = np.corrcoef(e_vals, e_y)[0, 1]
            corr_late = np.corrcoef(l_vals, l_y)[0, 1]
            corr_early = 0.0 if np.isnan(corr_early) else corr_early
            corr_late = 0.0 if np.isnan(corr_late) else corr_late
            instability = abs(corr_early - corr_late)
            sign_flip = corr_early * corr_late < 0
            blend_term = drift_base_blend + (1.0 - drift_base_blend) * instability
            risk = drift_severity * blend_term * (1.5 if sign_flip else 1.0)
        else:
            # Categorical drift severity via total variation distance.
            aligned = pd.DataFrame({
                'train': series_train.value_counts(normalize=True),
                'test': series_test.value_counts(normalize=True)
            }).fillna(0.0)
            drift_severity = 0.5 * np.abs(aligned['train'] - aligned['test']).sum()

            # Temporal instability in category->target mappings.
            early_df = pd.DataFrame(
                {
                    col: X_train.loc[early_mask, col],
                    "__target__": y_arr[early_mask],
                }
            )
            late_df = pd.DataFrame(
                {
                    col: X_train.loc[late_mask, col],
                    "__target__": y_arr[late_mask],
                }
            )
            early_rate = early_df.groupby(col, observed=False)['__target__'].mean()
            late_rate = late_df.groupby(col, observed=False)['__target__'].mean()
            aligned_rate = pd.concat([early_rate, late_rate], axis=1, keys=['e', 'l']).fillna(y_num.mean())
            cat_freq = series_train.value_counts(normalize=True)
            mapped_w = pd.Series(aligned_rate.index.map(cat_freq), index=aligned_rate.index)
            mapped_w = pd.to_numeric(mapped_w, errors='coerce').fillna(0.0)
            w = mapped_w.to_numpy(dtype=float, copy=False)
            if w.sum() == 0:
                continue
            w = w / w.sum()
            instability = np.sum(np.abs(aligned_rate['e'].values - aligned_rate['l'].values) * w)
            blend_term = drift_base_blend + (1.0 - drift_base_blend) * instability
            risk = drift_severity * blend_term

        scores.append((col, risk, drift_severity, instability))

    if not scores:
        return X_train, X_test, dropped

    risk_q = float(np.clip(float(os.getenv("STABILITY_RISK_QUANTILE", "0.90")), 0.0, 0.995))
    drift_q = float(np.clip(float(os.getenv("STABILITY_DRIFT_QUANTILE", "0.75")), 0.0, 0.995))
    inst_q = float(np.clip(float(os.getenv("STABILITY_INSTABILITY_QUANTILE", "0.70")), 0.0, 0.995))

    risk_floor = max(0.0, float(os.getenv("STABILITY_RISK_FLOOR", "0.0015")))
    drift_floor = max(0.0, float(os.getenv("STABILITY_DRIFT_FLOOR", "0.08")))
    inst_floor = max(0.0, float(os.getenv("STABILITY_INSTABILITY_FLOOR", "0.008")))

    risk_vals = np.asarray([float(s[1]) for s in scores], dtype=np.float64)
    drift_vals = np.asarray([float(s[2]) for s in scores], dtype=np.float64)
    inst_vals = np.asarray([float(s[3]) for s in scores], dtype=np.float64)

    risk_thr = max(risk_floor, float(np.quantile(risk_vals, risk_q)))
    drift_thr = max(drift_floor, float(np.quantile(drift_vals, drift_q)))
    inst_thr = max(inst_floor, float(np.quantile(inst_vals, inst_q)))

    high_risk = [
        x
        for x in scores
        if float(x[1]) >= risk_thr and float(x[2]) >= drift_thr and float(x[3]) >= inst_thr
    ]

    # Backstop: if no feature passes all three dynamic gates, allow a single
    # dominant outlier risk feature to be pruned when clearly separated.
    if not high_risk and _env_truthy("ENABLE_STABILITY_RISK_BACKSTOP", "1"):
        ordered = sorted(scores, key=lambda x: float(x[1]), reverse=True)
        top = ordered[0]
        med = float(np.median(risk_vals))
        mad = float(np.median(np.abs(risk_vals - med)))
        robust_sigma = 1.4826 * mad
        outlier_cut = med + 3.0 * robust_sigma
        if float(top[1]) >= max(risk_floor, outlier_cut) and float(top[2]) >= drift_floor:
            high_risk = [top]

    print(
        "      DEBUG: stability gates "
        f"(risk_thr={risk_thr:.6f}, drift_thr={drift_thr:.6f}, inst_thr={inst_thr:.6f}, "
        f"scores={len(scores)})"
    )
    preview = sorted(scores, key=lambda x: float(x[1]), reverse=True)[:5]
    if preview:
        print(
            "      DEBUG: stability top risks "
            + ", ".join(
                f"{c}(r={float(r):.6f},d={float(d):.6f},i={float(i):.6f})"
                for c, r, d, i in preview
            )
        )

    high_risk = sorted(high_risk, key=lambda x: x[1], reverse=True)[:max_drops]

    for col, _, _, _ in high_risk:
        if col in X_train.columns:
            X_train = X_train.drop(columns=[col])
        if col in X_test.columns:
            X_test = X_test.drop(columns=[col])
        dropped.append(col)

    if dropped:
        print(f"      DEBUG: stability-gated dropped columns: {dropped}")

    return X_train, X_test, dropped

def select_temporal_te_smoothing(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    feature_buckets: dict,
    drift_report: dict,
    candidates: list[int] | None = None
):
    """
    Select target-encoding base smoothing using a purely mathematical heuristic.
    No auxiliary model is trained here.
    """
    dynamic_base = max(10, int(len(X_train) * 0.005))
    if candidates is None:
        candidates = sorted(
            {
                max(10, dynamic_base // 2),
                dynamic_base,
                max(10, dynamic_base * 2),
            }
        )

    drifted_cats = [
        col for col in feature_buckets.get("cat_high", []) + feature_buckets.get("cat_low", [])
        if col in drift_report and col in X_train.columns and col in X_test.columns
    ]
    if not drifted_cats:
        return int(dynamic_base)

    base_from_n = int(dynamic_base)

    tv_vals = []
    card_vals = []
    for col in drifted_cats:
        tr_probs = X_train[col].value_counts(normalize=True)
        te_probs = X_test[col].value_counts(normalize=True)
        aligned = pd.concat([tr_probs, te_probs], axis=1).fillna(0.0)
        tv_distance = 0.5 * np.abs(aligned.iloc[:, 0] - aligned.iloc[:, 1]).sum()
        tv_vals.append(float(tv_distance))
        card_vals.append(float(X_train[col].nunique(dropna=True)))

    avg_tv = float(np.mean(tv_vals)) if tv_vals else 0.0
    avg_card = float(np.mean(card_vals)) if card_vals else 1.0

    # Increase smoothing when categorical drift and cardinality are higher.
    drift_factor = 1.0 + 2.5 * min(1.0, avg_tv)
    card_factor = 1.0 + 0.20 * np.log1p(max(1.0, avg_card))
    heuristic = int(base_from_n * drift_factor * card_factor)

    best_s = min(candidates, key=lambda c: abs(c - heuristic))
    print(
        "      DEBUG: mathematical TE smoothing selected "
        f"base={best_s} (heuristic={heuristic}, avg_tv={avg_tv:.4f}, avg_card={avg_card:.1f})"
    )
    return int(best_s)

def apply_winsorization(X_train: pd.DataFrame, X_test: pd.DataFrame, drift_report: dict, feature_buckets: dict):
    """
    Caps extreme numerical outliers for columns that failed the KS test.
    """
    for col in feature_buckets.get("numerical", []):
        if col in drift_report and _is_numerical_distribution_drift(_drift_entry(drift_report, col)):
            q_low = X_train[col].quantile(0.05)
            q_high = X_train[col].quantile(0.95)
            
            X_train[col] = X_train[col].clip(lower=q_low, upper=q_high)
            X_test[col] = X_test[col].clip(lower=q_low, upper=q_high)
            
    return X_train, X_test

def apply_adaptive_alignment(X_train: pd.DataFrame, X_test: pd.DataFrame, drift_report: dict, feature_buckets: dict):
    """
    Adaptive Distribution Alignment:
    Converts severely drifted numericals into quantiles (percentiles) independently.
    This neutralizes absolute magnitude drift (inflation) by converting values to relative rank.
    """
    for col in feature_buckets.get("numerical", []):
        if col in drift_report and _is_numerical_distribution_drift(_drift_entry(drift_report, col)):
            try:
                # qcut splits the data into 10 equal-sized buckets based on density
                # duplicates='drop' handles cases where a lot of numbers are exactly 0.0
                X_train[col] = pd.qcut(X_train[col], q=10, labels=False, duplicates='drop')
                X_test[col] = pd.qcut(X_test[col], q=10, labels=False, duplicates='drop')
            except Exception:
                # fallback to standard winsorization if quantile math fails on sparse data
                q_low = X_train[col].quantile(0.01)
                q_high = X_train[col].quantile(0.99)
                X_train[col] = X_train[col].clip(lower=q_low, upper=q_high)
                X_test[col] = X_test[col].clip(lower=q_low, upper=q_high)
            
    return X_train, X_test

def get_statistical_weights(X_train: pd.DataFrame, X_test: pd.DataFrame, drift_report: dict):
    """
    Calculates weights using Probabilistic Density Ratios.
    Student Lesson: 
    If a certain type of contract appears 20% of the time in Test, but only 10% 
    of the time in Train, we assign those Train rows a weight of 2.0 (20/10).
    """
    weights = np.ones(len(X_train))
    important_drifts = [col for col in drift_report.keys() if col in X_train.columns]
    
    for col in important_drifts:
        train_probs = X_train[col].value_counts(normalize=True)
        test_probs = X_test[col].value_counts(normalize=True)
        
        ratio_map = {}
        for val in train_probs.index:
            t_prob = train_probs[val]
            te_prob = test_probs.get(val, t_prob)
            ratio_map[val] = te_prob / (t_prob + 1e-6) # Added tiny number to avoid /0
            
        weights *= X_train[col].map(ratio_map).fillna(1.0).values
        
    weights = weights / np.mean(weights)
    weights = np.clip(weights, 0.1, 5.0)
    
    return weights

def get_temporal_similarity_weights(X_train: pd.DataFrame, X_test: pd.DataFrame, drift_report: dict, feature_buckets: dict):
    """
    Dynamically measures how similar each historical month is to the final exam (Test set).
    Student Lesson:
    We use the KS Statistic. A low KS Stat means the month is very similar to the Test set.
    We convert this into a "Similarity Score" where a higher score = more similar.
    """
    weights = np.ones(len(X_train))
    
    # We only measure distance using numerical columns that drifted
    num_drifts = [c for c in feature_buckets.get("numerical", []) if c in drift_report]
    
    time_col = PIPELINE_CONFIG.time_column if PIPELINE_CONFIG.time_column in X_train.columns else detect_time_column(X_train)
    if not num_drifts or time_col is None or time_col not in X_train.columns:
        return weights # Skip if no data available to calculate
        
    unique_times = X_train[time_col].astype("string").unique()
    month_scores = {}
    
    for time_key in unique_times:
        month_data = X_train[X_train[time_col].astype("string") == str(time_key)]
        ks_stats = []
        
        for col in num_drifts:
            # ks_2samp returns (statistic, p_value). 
            # The statistic is the actual physical distance between the curves (0 to 1)
            month_arr = month_data[col].dropna().to_numpy(dtype=np.float64, copy=False)
            test_arr = X_test[col].dropna().to_numpy(dtype=np.float64, copy=False)
            if len(month_arr) == 0 or len(test_arr) == 0:
                continue
            stat, _ = stats.ks_2samp(month_arr, test_arr, method="asymp")
            ks_stats.append(stat)
            
        # Average distance across all drifted features
        avg_distance = np.mean(ks_stats)
        
        # Invert distance into a Similarity Score (so smaller distance = higher weight)
        month_scores[str(time_key)] = 1.0 - avg_distance
        
    print(f"\n      DEBUG: Temporal Similarity Scores per Month:")
    for m, score in month_scores.items():
        print(f"        - {m}: {score:.4f}")
        
    # Map the scores to the corresponding rows in the training data
    weights = pd.to_numeric(
        X_train[time_col].astype("string").map(month_scores),
        errors='coerce'
    ).fillna(1.0).to_numpy(dtype=float, copy=False)

    # Normalize
    weights = weights / np.mean(weights)
    
    return weights

def apply_smoothed_target_encoding(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    feature_buckets: dict,
    drift_report: dict,
    base_smoothing_override: int | None = None,
    in_place: bool = True,
    n_splits: int = 5,
):
    """
    Smoothed target encoding with Polars-accelerated K-fold out-of-fold encoding.
    """
    _ = in_place
    if base_smoothing_override is None:
        base_smoothing_override = max(10, int(len(X_train) * 0.005))
    te_plan = build_target_encoding_plan(
        X_train=X_train,
        X_test_ref=X_test,
        y_train=y_train,
        feature_buckets=feature_buckets,
        drift_report=drift_report,
        base_smoothing_override=base_smoothing_override,
    )
    X_train = apply_kfold_target_encoding_with_plan(
        X_train=X_train,
        y_train=y_train,
        te_plan=te_plan,
        n_splits=n_splits,
        blend_alpha=1.0,
    )
    X_test = apply_target_encoding_plan(X_test, te_plan)
    return X_train, X_test


def apply_kfold_target_encoding_with_plan(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    te_plan: dict,
    n_splits: int = 10,
    blend_alpha: float = 0.75,
):
    """
    Apply leakage-safe OOF target encoding on train while blending with
    globally learned TE maps from detection sample.
    """
    if y_train is None or not te_plan.get("maps"):
        return X_train

    y_num = _to_numeric_target(y_train)
    y_arr = y_num.to_numpy(dtype=np.float64, copy=False)
    global_mean = float(te_plan.get("global_mean", y_num.mean()))
    smoothing_map = te_plan.get("smoothing", {})
    methods = te_plan.get("methods", {})
    n_rows = len(X_train)
    n_folds = max(2, int(n_splits))
    fold_ids = np.arange(n_rows, dtype=np.int32) % n_folds

    original_columns = X_train.columns.tolist()
    encoded_updates: dict[str, pd.Series] = {}

    for col, global_map in te_plan.get("maps", {}).items():
        if col not in X_train.columns:
            continue

        if methods.get(col) == "neutral":
            encoded_updates[col] = pd.Series(
                np.full(n_rows, global_mean, dtype=np.float32),
                index=X_train.index,
                name=col,
            )
            continue

        if methods.get(col) == "freq":
            encoded_updates[col] = pd.to_numeric(
                X_train[col].map(global_map),
                errors="coerce",
            ).fillna(0.0).astype(np.float32)
            continue

        smoothing = int(smoothing_map.get(col, 1200))
        cat_source = X_train[col].astype("string").fillna("__missing__")
        global_encoded = pd.to_numeric(
            cat_source.map(global_map),
            errors="coerce",
        ).fillna(global_mean).to_numpy(dtype=np.float32, copy=False)
        cat_series = cat_source.to_numpy(dtype=object, copy=False)
        pl_df = pl.DataFrame(
            {
                "__row_id": np.arange(n_rows, dtype=np.int64),
                "__fold_id": fold_ids.astype(np.int16, copy=False),
                "__cat": cat_series,
                "__target": y_arr.astype(np.float64, copy=False),
            }
        )
        encoded = (
            pl_df.with_columns(
                [
                    pl.col("__target").sum().over("__cat").alias("__sum_total"),
                    pl.col("__target").count().cast(pl.Float64).over("__cat").alias("__count_total"),
                    pl.col("__target").sum().over(["__cat", "__fold_id"]).alias("__sum_fold"),
                    pl.col("__target").count().cast(pl.Float64).over(["__cat", "__fold_id"]).alias("__count_fold"),
                ]
            )
            .with_columns(
                (
                    (pl.col("__sum_total") - pl.col("__sum_fold") + smoothing * global_mean)
                    / (pl.col("__count_total") - pl.col("__count_fold") + smoothing)
                ).alias("__oof_rate")
            )
            .sort("__row_id")
            .select("__oof_rate")
            .to_series()
            .fill_null(global_mean)
            .to_numpy()
            .astype(np.float32, copy=False)
        )

        blended = blend_alpha * encoded + (1.0 - blend_alpha) * global_encoded
        encoded_updates[col] = pd.Series(
            blended.astype(np.float32, copy=False),
            index=X_train.index,
            name=col,
        )

    if encoded_updates:
        updates_df = pd.DataFrame(encoded_updates, index=X_train.index)
        base_df = X_train.drop(columns=list(encoded_updates.keys()), errors="ignore")
        X_train = pd.concat([base_df, updates_df], axis=1)
        X_train = X_train.reindex(columns=original_columns)

    return X_train

def apply_adversarial_dropping(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Purely mathematical train-vs-test separability pruning (no model training).
    """
    dropped_cols = []
    time_col = PIPELINE_CONFIG.time_column if PIPELINE_CONFIG.time_column in X_train.columns else detect_time_column(X_train)
    shared_cols = [
        c for c in X_train.columns
        if c in X_test.columns and (time_col is None or c != time_col)
    ]

    adaptive_threshold_enabled = _env_truthy("ENABLE_ADAPTIVE_ADVERSARIAL_DROPPING", "1")
    ks_floor = float(os.getenv("ADVERSARIAL_KS_FLOOR", "0.25"))
    ks_cap = float(os.getenv("ADVERSARIAL_KS_CAP", "0.50"))
    tv_floor = float(os.getenv("ADVERSARIAL_TV_FLOOR", "0.35"))
    tv_cap = float(os.getenv("ADVERSARIAL_TV_CAP", "0.60"))
    fixed_ks_threshold = float(os.getenv("ADVERSARIAL_KS_FIXED", "0.35"))
    fixed_tv_threshold = float(os.getenv("ADVERSARIAL_TV_FIXED", "0.45"))

    numeric_stats: list[tuple[str, float]] = []
    categorical_stats: list[tuple[str, float]] = []

    for col in shared_cols:
        tr = X_train[col]
        te = X_test[col]
        if pd.api.types.is_numeric_dtype(tr):
            tr_s = tr.dropna()
            te_s = te.dropna()
            if tr_s.empty or te_s.empty:
                continue
            if len(tr_s) > 120000:
                tr_s = tr_s.sample(n=120000, random_state=42)
            if len(te_s) > 120000:
                te_s = te_s.sample(n=120000, random_state=42)
            ks_stat, _ = stats.ks_2samp(tr_s, te_s)
            numeric_stats.append((col, float(ks_stat)))
        else:
            aligned = pd.DataFrame({
                "train": tr.value_counts(normalize=True),
                "test": te.value_counts(normalize=True)
            }).fillna(0.0)
            tv = 0.5 * np.abs(aligned["train"] - aligned["test"]).sum()
            categorical_stats.append((col, float(tv)))

    ks_threshold = float(fixed_ks_threshold)
    tv_threshold = float(fixed_tv_threshold)
    if adaptive_threshold_enabled and numeric_stats:
        ks_vals = np.asarray([v for _, v in numeric_stats], dtype=np.float64)
        ks_threshold = float(np.clip(np.mean(ks_vals) + 2.0 * np.std(ks_vals), ks_floor, ks_cap))
    if adaptive_threshold_enabled and categorical_stats:
        tv_vals = np.asarray([v for _, v in categorical_stats], dtype=np.float64)
        tv_threshold = float(np.clip(np.mean(tv_vals) + 2.0 * np.std(tv_vals), tv_floor, tv_cap))

    for col, ks_stat in numeric_stats:
        if ks_stat >= ks_threshold:
            dropped_cols.append(col)
    for col, tv_stat in categorical_stats:
        if tv_stat >= tv_threshold:
            dropped_cols.append(col)

    dropped_cols = sorted(set(dropped_cols))

    if dropped_cols:
        X_train = X_train.drop(columns=dropped_cols, errors='ignore')
        X_test = X_test.drop(columns=dropped_cols, errors='ignore')
        print(
            "      DEBUG: mathematically dropped highly separable columns: "
            f"{dropped_cols} (ks_thr={ks_threshold:.4f}, tv_thr={tv_threshold:.4f}, adaptive={adaptive_threshold_enabled})"
        )
    else:
        print(
            "      DEBUG: adversarial dropping kept all columns "
            f"(ks_thr={ks_threshold:.4f}, tv_thr={tv_threshold:.4f}, adaptive={adaptive_threshold_enabled})"
        )
        
    return X_train, X_test

def get_distance_weights(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    calculates covariate shift weights using pure mathematical distance (1-nn).
    completely avoids ai classification to strictly comply with the one-model rule.
    """
    # 1. strictly use normalized numeric data so large numbers don't dominate the distance math
    X_tr_num = X_train.select_dtypes(include=[np.number]).fillna(0)
    X_te_num = X_test.select_dtypes(include=[np.number]).fillna(0)
    
    if X_tr_num.empty or X_te_num.empty:
        return np.ones(len(X_train))
        
    # normalize columns to 0-1 scale so distance is calculated fairly
    for col in X_tr_num.columns:
        min_val = X_tr_num[col].min()
        max_val = X_tr_num[col].max()
        if max_val > min_val:
            X_tr_num[col] = (X_tr_num[col] - min_val) / (max_val - min_val)
            X_te_num[col] = (X_te_num[col] - min_val) / (max_val - min_val)
            
    # 2. build a mathematical spatial tree of the training data
    tree = cKDTree(X_tr_num.values)
    
    # 3. for every test row, find the 1 mathematically closest training row
    # k=1 returns the exact nearest neighbor index
    distances, indices = tree.query(X_te_num.values, k=1)
    
    # 4. count how many times each training row was chosen
    weights = np.ones(len(X_train))
    unique_indices, counts = np.unique(indices, return_counts=True)
    
    # boost the weight of the chosen rows based on their hit count
    # we use a gentle multiplier (e.g., 1 hit = 1.1 weight, 2 hits = 1.2 weight)
    for idx, count in zip(unique_indices, counts):
        weights[idx] += (count * 0.1)
        
    # 5. apply the gentle soft-clip to prevent gradient explosion
    weights = np.clip(weights, 0.8, 1.2)
    
    # normalize to protect the is_unbalance=True baseline
    weights = weights / np.mean(weights)
    
    return weights

def get_temporal_similarity_weights_fast(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    drift_report: dict,
    feature_buckets: dict,
    max_rows_per_month: int = 60000,
    max_test_rows: int = 120000,
    max_drift_features: int = 8,
    feature_importance_gain: dict | None = None,
    return_month_scores: bool = False,
    sharpness_power: float = 8.0,
    softmax_temperature: float = 0.20,
    recency_blend: float = 0.35,
):
    """
    Runtime-optimized temporal similarity weights for very large datasets.
    """
    weights = np.ones(len(X_train))
    num_drifts = [c for c in feature_buckets.get("numerical", []) if c in drift_report and c in X_train.columns and c in X_test.columns]

    time_col = detect_time_column(X_train)
    if not num_drifts or time_col is None or time_col not in X_train.columns:
        if return_month_scores:
            return weights, {}
        return weights

    num_drifts = num_drifts[:max_drift_features]
    month_scores = {}
    unique_times = X_train[time_col].astype("string").unique()
    feature_importance_gain = feature_importance_gain or {}

    col_weights = np.array(
        [max(1e-6, float(feature_importance_gain.get(col, 1.0))) for col in num_drifts],
        dtype=np.float64,
    )
    col_weights = col_weights / col_weights.sum()

    test_cache = {}
    for col in num_drifts:
        te = X_test[col].dropna()
        if len(te) > max_test_rows:
            te = te.sample(n=max_test_rows, random_state=42)
        test_cache[col] = te.to_numpy(dtype=np.float64, copy=False)

    train_time_vals = X_train[time_col].astype("string")
    for time_key in unique_times:
        month_data = X_train[train_time_vals == str(time_key)]
        ks_stats = []
        ks_weights = []

        for idx, col in enumerate(num_drifts):
            tr = month_data[col].dropna()
            if len(tr) > max_rows_per_month:
                tr = tr.sample(n=max_rows_per_month, random_state=42)
            te = test_cache[col]
            tr_arr = tr.to_numpy(dtype=np.float64, copy=False)
            if len(tr_arr) == 0 or len(te) == 0:
                continue
            stat, _ = stats.ks_2samp(tr_arr, te, method="asymp")
            ks_stats.append(stat)
            ks_weights.append(col_weights[idx])

        if not ks_stats:
            month_scores[str(time_key)] = 1.0
        else:
            ks_arr = np.asarray(ks_stats, dtype=np.float64)
            w_arr = np.asarray(ks_weights, dtype=np.float64)
            w_arr = w_arr / w_arr.sum()
            month_scores[str(time_key)] = 1.0 - float(np.average(ks_arr, weights=w_arr))

    print(f"\n      DEBUG: Temporal Similarity Scores per Month:")
    for m, score in month_scores.items():
        print(f"        - {m}: {score:.4f}")

    raw_weights = pd.to_numeric(
        train_time_vals.map(month_scores),
        errors='coerce'
    ).fillna(1.0).to_numpy(dtype=float, copy=False)

    # Use controlled sharpening with soft clipping to avoid unstable over-weighting.
    safe_raw = np.clip(raw_weights, 1e-3, None)
    stretched = np.power(safe_raw, max(1.0, float(sharpness_power)))

    temp = max(0.05, float(softmax_temperature))
    z = (stretched - np.max(stretched)) / temp
    z = np.clip(z, -50.0, 50.0)
    soft = np.exp(z)
    soft = soft / np.sum(soft)
    soft = soft * len(soft)

    lo = float(np.quantile(soft, 0.05))
    hi = float(np.quantile(soft, 0.95))
    weights = np.clip(soft, lo, hi)
    weights = weights / np.mean(weights)

    # Blend with chronological recency to better extrapolate into future periods.
    # This stays dynamic by inferring order from observed time keys.
    if len(unique_times) >= 4 and recency_blend > 0.0:
        ordered_times = _sorted_time_values(pd.Series(unique_times, dtype="string"))
        if ordered_times:
            rank_map = {t: (i + 1) / float(len(ordered_times)) for i, t in enumerate(ordered_times)}
            rec = pd.to_numeric(train_time_vals.map(rank_map), errors="coerce").fillna(0.5).to_numpy(dtype=np.float64, copy=False)
            rec = np.exp(2.0 * np.clip(rec, 0.0, 1.0))
            rec = rec / max(1e-12, float(np.mean(rec)))
            weights = weights * np.power(rec, float(recency_blend))
            weights = np.clip(weights, np.quantile(weights, 0.05), np.quantile(weights, 0.95))
            weights = weights / max(1e-12, float(np.mean(weights)))

    if return_month_scores:
        return weights, month_scores
    return weights

def rank_normalize_drifted_numerics(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    drift_scores: dict,
    time_col: str | None = None,
    max_features: int = 8,
    include_cols: list[str] | None = None,
):
    """
    Apply rank-based normalization to the most drifted numeric columns.
    Uses grouped ranks by time when a valid time column is available.
    """
    if not isinstance(drift_scores, dict) or not drift_scores:
        return X_train, X_test, []

    if include_cols is not None:
        selected_cols = [
            c
            for c in include_cols
            if c in X_train.columns and c in X_test.columns and pd.api.types.is_numeric_dtype(X_train[c])
        ]
        if not selected_cols:
            return X_train, X_test, []
    else:
        ranked_candidates: list[tuple[str, float]] = []
        for col, meta in drift_scores.items():
            if col not in X_train.columns or col not in X_test.columns:
                continue
            if not pd.api.types.is_numeric_dtype(X_train[col]):
                continue
            if str(meta.get("feature_type", "")) != "numerical":
                continue
            dist = float(meta.get("distance_score", 0.0))
            if dist <= 0.0:
                continue
            ranked_candidates.append((col, dist))

        if not ranked_candidates:
            return X_train, X_test, []

        ranked_candidates = sorted(ranked_candidates, key=lambda x: x[1], reverse=True)
        selected_cols = [c for c, _ in ranked_candidates[: max(1, int(max_features))]]

    out_train = X_train.copy()
    out_test = X_test.copy()

    def _apply_group_rank(frame: pd.DataFrame, col: str, by_col: str):
        grp_cnt = frame.groupby(by_col, observed=False)[col].transform("count").replace(0, np.nan)
        grp_rank = frame.groupby(by_col, observed=False)[col].rank(method="average")
        frame[col] = pd.to_numeric(grp_rank / grp_cnt, errors="coerce").astype(np.float32)

    for col in selected_cols:
        if time_col and time_col in out_train.columns and time_col in out_test.columns:
            _apply_group_rank(out_train, col, time_col)
            _apply_group_rank(out_test, col, time_col)
        else:
            train_rank = out_train[col].rank(method="average") / max(1, len(out_train))
            test_rank = out_test[col].rank(method="average") / max(1, len(out_test))
            out_train[col] = pd.to_numeric(train_rank, errors="coerce").astype(np.float32)
            out_test[col] = pd.to_numeric(test_rank, errors="coerce").astype(np.float32)

    return out_train, out_test, selected_cols

def select_stable_predictive_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    exclude_cols: list[str] | None = None,
    keep_ratio: float = 0.72,
    min_keep: int = 28,
    max_keep: int = 180,
    max_rows: int = 120000,
    sample_weights: np.ndarray | pd.Series | None = None,
    temporal_consistency_blend: float = 0.35,
    weighted_assoc_blend: float = 0.35,
    enable_categorical_scoring: bool = False,
):
    """
    Keep features that are both predictive (target association) and stable
    (lower train-test distribution mismatch). No auxiliary model is trained.
    """
    if y_train is None or X_train.empty or X_test.empty:
        return list(X_train.columns), {"applied": False, "reason": "empty-input"}

    exclude = set(exclude_cols or [])
    shared_cols = [c for c in X_train.columns if c in X_test.columns and c not in exclude]
    if len(shared_cols) <= min_keep:
        return shared_cols, {
            "applied": False,
            "reason": "insufficient-features",
            "selected": len(shared_cols),
            "total": len(shared_cols),
        }

    y_num = _to_numeric_target(y_train).astype(np.float64)
    if y_num.nunique(dropna=True) < 2:
        return shared_cols, {"applied": False, "reason": "invalid-target"}

    temporal_consistency_blend = float(np.clip(temporal_consistency_blend, 0.0, 0.8))
    weighted_assoc_blend = float(np.clip(weighted_assoc_blend, 0.0, 0.8))
    enable_categorical_scoring = bool(enable_categorical_scoring)

    w_full = None
    if sample_weights is not None:
        w_try = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        if w_try.size == len(X_train):
            w_try = np.clip(np.nan_to_num(w_try, nan=1.0, posinf=3.0, neginf=0.0), 0.0, None)
            w_mean = float(np.mean(w_try)) if w_try.size else 0.0
            if w_mean > 0.0:
                w_full = w_try / w_mean

    def _sample(arr: np.ndarray, n: int, seed: int):
        if arr.size <= n:
            return arr
        rng = np.random.default_rng(seed)
        idx = rng.choice(arr.size, size=n, replace=False)
        return arr[idx]

    def _weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        if x.size < 3 or y.size < 3 or w.size < 3:
            return 0.0
        ww = np.clip(np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
        sw = float(np.sum(ww))
        if sw <= 1e-12:
            return 0.0
        mx = float(np.sum(ww * x) / sw)
        my = float(np.sum(ww * y) / sw)
        cx = x - mx
        cy = y - my
        vx = float(np.sum(ww * cx * cx) / sw)
        vy = float(np.sum(ww * cy * cy) / sw)
        if vx <= 1e-12 or vy <= 1e-12:
            return 0.0
        cov = float(np.sum(ww * cx * cy) / sw)
        corr = cov / max(1e-12, np.sqrt(vx * vy))
        if not np.isfinite(corr):
            return 0.0
        return float(np.clip(corr, -1.0, 1.0))

    def _categorical_assoc(cat_series: pd.Series, y_vals: np.ndarray, w_vals: np.ndarray | None) -> float:
        if len(cat_series) == 0 or y_vals.size == 0:
            return 0.0
        cat = cat_series.astype("string").fillna("__missing__")
        if w_vals is None:
            w = np.ones(len(cat), dtype=np.float64)
        else:
            w = np.asarray(w_vals, dtype=np.float64)
            if w.size != len(cat):
                return 0.0
            w = np.clip(np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)

        frame = pd.DataFrame(
            {
                "__cat": cat.to_numpy(copy=False),
                "__w": w,
                "__yw": w * y_vals,
            }
        )
        agg = frame.groupby("__cat", observed=False).agg(w_sum=("__w", "sum"), yw_sum=("__yw", "sum"))
        if agg.empty:
            return 0.0
        total_w = float(agg["w_sum"].sum())
        if total_w <= 1e-12:
            return 0.0
        global_rate = float(frame["__yw"].sum() / total_w)
        rates = agg["yw_sum"] / agg["w_sum"].replace(0.0, np.nan)
        freq = agg["w_sum"] / total_w
        score = float(np.sum(np.abs(rates.fillna(global_rate).values - global_rate) * freq.values))
        if not np.isfinite(score):
            return 0.0
        return float(np.clip(score, 0.0, 1.0))

    def _categorical_tv_distance(tr_series: pd.Series, te_series: pd.Series) -> float:
        tr = tr_series.astype("string").fillna("__missing__")
        te = te_series.astype("string").fillna("__missing__")
        tr_p = tr.value_counts(normalize=True)
        te_p = te.value_counts(normalize=True)
        aligned = pd.concat([tr_p, te_p], axis=1, keys=["tr", "te"]).fillna(0.0)
        tv = 0.5 * float(np.abs(aligned["tr"] - aligned["te"]).sum())
        return float(np.clip(tv, 0.0, 1.0))

    score_rows: list[tuple[str, float, float, float, float, float, str]] = []
    y_arr = y_num.to_numpy(dtype=np.float64, copy=False)

    for i, col in enumerate(shared_cols, start=1):
        tr_raw = X_train[col]
        te_raw = X_test[col]
        tr_num = pd.to_numeric(tr_raw, errors="coerce")
        te_num = pd.to_numeric(te_raw, errors="coerce")

        train_num_cov = float(tr_num.notna().mean()) if len(tr_num) else 0.0
        test_num_cov = float(te_num.notna().mean()) if len(te_num) else 0.0
        train_num_unique = int(tr_num.nunique(dropna=True))
        force_categorical = (
            not pd.api.types.is_numeric_dtype(tr_raw)
            and max(train_num_cov, test_num_cov) < 0.97
        )
        is_numeric_like = (
            not force_categorical
            and min(train_num_cov, test_num_cov) >= 0.70
            and train_num_unique > 8
        )
        if not enable_categorical_scoring:
            # Baseline-compatible mode: numeric-proxy scoring only.
            is_numeric_like = True

        assoc = 0.0
        assoc_weighted = 0.0
        drift = 0.0
        feature_kind = "numeric" if is_numeric_like else "categorical"

        if is_numeric_like:
            valid_mask = tr_num.notna().to_numpy(copy=False)
            if int(valid_mask.sum()) >= 80:
                x_assoc = tr_num.to_numpy(dtype=np.float64, copy=False)[valid_mask]
                y_assoc = y_arr[valid_mask]
                if np.unique(x_assoc).size > 1:
                    corr = pd.Series(x_assoc).corr(pd.Series(y_assoc), method="spearman")
                    assoc = float(abs(corr)) if pd.notna(corr) else 0.0
                    assoc_weighted = assoc
                    if w_full is not None:
                        x_rank = pd.Series(x_assoc).rank(method="average").to_numpy(dtype=np.float64, copy=False)
                        y_rank = pd.Series(y_assoc).rank(method="average").to_numpy(dtype=np.float64, copy=False)
                        w_assoc = w_full[valid_mask]
                        assoc_weighted = abs(_weighted_corr(x_rank, y_rank, w_assoc))

            tr_arr = tr_num.dropna().to_numpy(dtype=np.float64, copy=False)
            te_arr = te_num.dropna().to_numpy(dtype=np.float64, copy=False)
            if tr_arr.size >= 120 and te_arr.size >= 120:
                tr_s = _sample(tr_arr, max_rows, seed=100 + i)
                te_s = _sample(te_arr, max_rows, seed=200 + i)
                if np.unique(tr_s).size > 1 and np.unique(te_s).size > 1:
                    ks = float(stats.ks_2samp(tr_s, te_s, method="asymp").statistic)
                    q = np.linspace(0.0, 1.0, 11)
                    edges = np.unique(np.quantile(tr_s, q))
                    if edges.size >= 3:
                        tr_h, _ = np.histogram(tr_s, bins=edges)
                        te_h, _ = np.histogram(te_s, bins=edges)
                        tr_p = np.clip(tr_h / max(1, tr_h.sum()), 1e-6, None)
                        te_p = np.clip(te_h / max(1, te_h.sum()), 1e-6, None)
                        psi = float(np.sum((te_p - tr_p) * np.log(te_p / tr_p)))
                    else:
                        psi = 0.0
                    drift = max(ks, float(min(1.0, max(0.0, psi))))
        else:
            assoc = _categorical_assoc(tr_raw, y_arr, w_vals=None)
            assoc_weighted = assoc
            if w_full is not None:
                assoc_weighted = _categorical_assoc(tr_raw, y_arr, w_vals=w_full)
            if len(tr_raw) >= 120 and len(te_raw) >= 120:
                drift = _categorical_tv_distance(tr_raw, te_raw)

        miss_shift = abs(float(tr_raw.isna().mean()) - float(te_raw.isna().mean()))
        distribution_stability = 1.0 - min(0.95, 0.70 * drift + 0.30 * miss_shift)

        temporal_stability = 1.0
        if w_full is not None:
            assoc_ref = max(0.03, assoc, assoc_weighted)
            temporal_instability = abs(assoc - assoc_weighted) / assoc_ref
            temporal_stability = 1.0 - min(0.95, temporal_instability)

        predictive_strength = assoc
        if w_full is not None:
            predictive_strength = (
                (1.0 - weighted_assoc_blend) * assoc
                + weighted_assoc_blend * assoc_weighted
            )

        stability = distribution_stability
        if w_full is not None:
            stability = (
                (1.0 - temporal_consistency_blend) * distribution_stability
                + temporal_consistency_blend * temporal_stability
            )

        score = predictive_strength * max(0.05, stability)
        score_rows.append(
            (
                col,
                score,
                assoc,
                drift,
                miss_shift,
                temporal_stability,
                feature_kind,
            )
        )

    if not score_rows:
        return shared_cols, {"applied": False, "reason": "no-scores"}

    mean_drift = float(np.mean([r[3] for r in score_rows]))
    mean_temporal_instability = float(np.mean([1.0 - r[5] for r in score_rows]))
    adaptive_ratio = float(
        np.clip(
            keep_ratio - 0.16 * mean_drift - 0.10 * mean_temporal_instability,
            0.52,
            0.88,
        )
    )
    target_keep = int(len(shared_cols) * adaptive_ratio)
    target_keep = max(int(min_keep), min(int(max_keep), target_keep))

    ranked = sorted(score_rows, key=lambda r: r[1], reverse=True)
    keep_cols = [c for c, *_ in ranked[:target_keep]]
    drop_cols = [c for c in shared_cols if c not in set(keep_cols)]

    return keep_cols, {
        "applied": True,
        "selected": len(keep_cols),
        "total": len(shared_cols),
        "mean_drift": mean_drift,
        "mean_temporal_instability": mean_temporal_instability,
        "adaptive_ratio": adaptive_ratio,
        "weighted_mode": bool(w_full is not None),
        "temporal_consistency_blend": temporal_consistency_blend,
        "weighted_assoc_blend": weighted_assoc_blend,
        "top_features": [
            {
                "feature": c,
                "score": float(s),
                "association": float(a),
                "drift": float(d),
                "missing_shift": float(m),
                "temporal_stability": float(ts),
                "kind": k,
            }
            for c, s, a, d, m, ts, k in ranked[:15]
        ],
        "dropped_preview": drop_cols[:20],
    }

def build_structure_plan(
    X_train: pd.DataFrame,
    feature_buckets: dict,
    drift_report: dict,
    y_train: pd.Series | None = None,
    X_test: pd.DataFrame | None = None,
):
    """
    Build deterministic structure-mitigation rules for streamed chunks.
    """
    drop_cols = set()
    rare_label_map = {}
    cat_high_rare_thr = float(os.getenv("CAT_HIGH_RARE_THRESHOLD", "0.0015"))

    id_col = PIPELINE_CONFIG.id_column if PIPELINE_CONFIG.id_column in X_train.columns else detect_identifier_column(X_train)
    if id_col is not None and id_col in X_train.columns:
        drop_cols.add(id_col)

    # Dynamic adaptive drop threshold for severe distance drifts.
    all_distance_scores = np.asarray(
        [_distance_score(_drift_entry(drift_report, col)) for col in drift_report.keys()],
        dtype=np.float64,
    )
    all_distance_scores = all_distance_scores[np.isfinite(all_distance_scores)]
    if all_distance_scores.size:
        drift_mean = float(np.mean(all_distance_scores))
        drift_std = float(np.std(all_distance_scores))
        dynamic_threshold = drift_mean + (2.0 * drift_std)
    else:
        drift_mean = 0.0
        drift_std = 0.0
        dynamic_threshold = 0.50
    final_threshold = max(0.20, min(dynamic_threshold, 0.50))
    print(
        "      DEBUG: adaptive drop threshold "
        f"(mean={drift_mean:.4f}, std={drift_std:.4f}, raw={dynamic_threshold:.4f}, final={final_threshold:.4f})"
    )

    columns_to_drop = []
    for col in drift_report.keys():
        # High-card categorical features are rescued for TE, not dropped.
        if col in feature_buckets.get("cat_high", []):
            continue
        distance_score = _distance_score(_drift_entry(drift_report, col))
        # Only drop if the distance strictly exceeds the adaptive dynamic threshold.
        if distance_score > final_threshold:
            columns_to_drop.append(col)

    # Execute the drop immediately
    if columns_to_drop:
        X_train.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        if X_test is not None:
            X_test.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        drop_cols.update(columns_to_drop)

    for col in feature_buckets.get("cat_high", []):
        if col not in X_train.columns:
            continue
        value_counts = X_train[col].value_counts(normalize=True)
        rare_labels = value_counts[value_counts < cat_high_rare_thr].index
        rare_label_map[col] = set(rare_labels.astype(str).tolist())

    return {"drop_cols": sorted(drop_cols), "rare_label_map": rare_label_map}

def apply_structure_plan(df: pd.DataFrame, plan: dict, in_place: bool = True):
    """
    Apply precomputed structure plan to a test chunk.
    """
    _ = in_place
    out = df
    columns_to_drop = plan.get("drop_cols", [])
    # Force the dataframe to drop the toxic columns.
    if columns_to_drop:
        out.drop(columns=columns_to_drop, inplace=True, errors="ignore")
    for col, rare_labels in plan.get("rare_label_map", {}).items():
        if col not in out.columns or not rare_labels:
            continue
        if isinstance(out[col].dtype, pd.CategoricalDtype) and "other_rare" not in out[col].cat.categories:
            out[col] = out[col].cat.add_categories(["other_rare"])
        out.loc[out[col].astype(str).isin(rare_labels), col] = "other_rare"
    return out

def build_winsor_plan(X_train: pd.DataFrame, drift_report: dict, feature_buckets: dict):
    """
    Precompute numeric realignment plan with:
    - optional log scaling for range explosion
    - IQR clipping bounds
    - median-shift alignment deltas for test-time correction
    """
    bounds = {}
    for col in feature_buckets.get("numerical", []):
        if (
            col in X_train.columns
            and col in drift_report
            and _is_numerical_distribution_drift(_drift_entry(drift_report, col))
        ):
            series = pd.to_numeric(X_train[col], errors="coerce")
            non_na = series.dropna()
            if non_na.empty:
                continue

            q10 = float(non_na.quantile(0.10))
            q50 = float(non_na.quantile(0.50))
            q90 = float(non_na.quantile(0.90))
            range_ratio = abs(q90 - q10) / (abs(q50) + 1e-6)
            use_log = bool(non_na.min() >= 0 and range_ratio >= 12.0)
            if use_log:
                series_work = np.log1p(np.clip(series.to_numpy(dtype=np.float64, copy=False), 0.0, None))
                series_work = pd.Series(series_work, index=series.index)
            else:
                series_work = series

            train_median = float(series_work.median())
            q1 = float(series_work.quantile(0.25))
            q3 = float(series_work.quantile(0.75))
            iqr = q3 - q1
            if not np.isfinite(iqr) or iqr <= 1e-9:
                q_low = float(series_work.quantile(0.05))
                q_high = float(series_work.quantile(0.95))
            else:
                q_low = float(q1 - 3.0 * iqr)
                q_high = float(q3 + 3.0 * iqr)
            bounds[col] = {
                "q_low": q_low,
                "q_high": q_high,
                "use_log": use_log,
                "train_median": train_median,
                "test_median_ref": train_median,
            }
    return bounds

def calibrate_winsor_plan_to_test_reference(winsor_bounds: dict, X_test_ref: pd.DataFrame):
    """
    Update plan with test-reference medians so test chunks can be shifted
    toward train-domain medians before inference.
    """
    for col, info in winsor_bounds.items():
        if col not in X_test_ref.columns:
            continue
        series = pd.to_numeric(X_test_ref[col], errors="coerce")
        if bool(info.get("use_log", False)):
            arr = np.log1p(np.clip(series.to_numpy(dtype=np.float64, copy=False), 0.0, None))
            test_med = float(np.nanmedian(arr))
        else:
            test_med = float(series.median())
        if np.isfinite(test_med):
            info["test_median_ref"] = test_med
    return winsor_bounds

def apply_winsor_plan(
    df: pd.DataFrame,
    winsor_bounds: dict,
    in_place: bool = True,
    align_test_distribution: bool = False,
):
    """
    Apply precomputed winsor bounds to a dataframe chunk.
    """
    _ = in_place
    out = df
    for col, info in winsor_bounds.items():
        if col in out.columns:
            series = pd.to_numeric(out[col], errors="coerce")
            arr = series.to_numpy(dtype=np.float64, copy=False)
            if bool(info.get("use_log", False)):
                arr = np.log1p(np.clip(arr, 0.0, None))
            if align_test_distribution:
                shift = float(info.get("train_median", 0.0) - info.get("test_median_ref", 0.0))
                arr = arr + shift
            q_low = float(info.get("q_low", np.nanmin(arr)))
            q_high = float(info.get("q_high", np.nanmax(arr)))
            arr = np.clip(arr, q_low, q_high)
            out[col] = arr.astype(np.float32, copy=False)
    return out

def build_target_encoding_plan(
    X_train: pd.DataFrame,
    X_test_ref: pd.DataFrame,
    y_train: pd.Series,
    feature_buckets: dict,
    drift_report: dict,
    base_smoothing_override: int | None = None
):
    """
    Build reusable target-encoding maps for streamed chunk inference.
    """
    y_num = _to_numeric_target(y_train)
    dynamic_base = max(10, int(len(X_train) * 0.005))
    strong_te_floor = 700
    if base_smoothing_override is None:
        base_smoothing = int(max(dynamic_base, strong_te_floor))
    else:
        # Keep smoothing dynamic but enforce high regularization for drifted categoricals.
        bounded = int(max(10, min(int(base_smoothing_override), int(dynamic_base * 2.5))))
        base_smoothing = int(max(bounded, strong_te_floor))

    target_series = pd.Series(y_num.values, index=X_train.index).astype(np.float64)

    recency_weighted_te_enabled = _env_truthy("ENABLE_RECENCY_WEIGHTED_TE", "0")
    recency_te_strength = float(os.getenv("RECENCY_TE_STRENGTH", "2.0"))
    time_col = PIPELINE_CONFIG.time_column if PIPELINE_CONFIG.time_column in X_train.columns else detect_time_column(X_train)

    row_te_weights = pd.Series(np.ones(len(X_train), dtype=np.float64), index=X_train.index)
    if recency_weighted_te_enabled and time_col is not None and time_col in X_train.columns:
        row_te_weights = _build_recency_weights(X_train[time_col], strength=recency_te_strength)

    weight_arr = pd.to_numeric(row_te_weights, errors="coerce").fillna(1.0).to_numpy(dtype=np.float64, copy=False)
    y_arr = target_series.to_numpy(dtype=np.float64, copy=False)
    global_mean = float(np.average(y_arr, weights=weight_arr)) if weight_arr.size else float(target_series.mean())
    target_var = float(np.average(np.square(y_arr - global_mean), weights=weight_arr)) + 1e-12

    maps = {}
    smoothings = {}
    methods = {}

    drifted_cats = [
        col
        for col in (
            feature_buckets.get("cat_high", [])
            + feature_buckets.get("cat_low", [])
            + feature_buckets.get("binary", [])
        )
        if col in drift_report and col in X_train.columns and col in X_test_ref.columns
    ]

    candidate_cols = drifted_cats
    high_card_smoothing_floor = int(float(os.getenv("TE_HIGH_CARD_BASE_SMOOTHING", "1000")))
    max_smoothing_cap = int(float(os.getenv("TE_MAX_SMOOTHING", "7000")))
    dual_te_smoothing_enabled = _env_truthy("ENABLE_DUAL_TE_SMOOTHING", "0")
    te_temporal_guard_enabled = _env_truthy("ENABLE_TE_TEMPORAL_GUARD", "0")
    te_temporal_guard_min_instability = float(os.getenv("TE_TEMPORAL_GUARD_MIN_INSTABILITY", "0.18"))
    te_temporal_guard_min_drift = float(os.getenv("TE_TEMPORAL_GUARD_MIN_DRIFT", "0.14"))

    temporal_instability_map: dict[str, float] = {}
    if candidate_cols and (dual_te_smoothing_enabled or te_temporal_guard_enabled):
        if time_col is not None and time_col in X_train.columns:
            for col in candidate_cols:
                if col == time_col or col not in X_train.columns:
                    continue
                temporal_instability_map[col] = _categorical_temporal_instability_score(
                    category_series=X_train[col],
                    y_num=y_num,
                    time_series=X_train[time_col],
                )

    bin_neutral_thr = float(os.getenv("TE_BINARY_NEUTRAL_THRESHOLD", "0.15"))
    bin_freq_thr = float(os.getenv("TE_BINARY_FREQ_THRESHOLD", "0.10"))
    low_neutral_drift_thr = float(os.getenv("TE_LOWCARD_NEUTRAL_DRIFT_THRESHOLD", "0.15"))
    low_neutral_tv_thr = float(os.getenv("TE_LOWCARD_NEUTRAL_TV_THRESHOLD", "0.18"))
    low_freq_drift_thr = float(os.getenv("TE_LOWCARD_FREQ_DRIFT_THRESHOLD", "0.10"))
    low_freq_tv_thr = float(os.getenv("TE_LOWCARD_FREQ_TV_THRESHOLD", "0.12"))
    force_te_assoc_thr = float(os.getenv("TE_FORCE_TE_ASSOC_THRESHOLD", "0.22"))
    force_te_card_thr = int(os.getenv("TE_FORCE_TE_CARD_THRESHOLD", "50"))

    for col in candidate_cols:
        cat_raw = X_train[col].astype("string").fillna("__missing__")
        if recency_weighted_te_enabled:
            weighted_freq = row_te_weights.groupby(cat_raw, observed=False).sum().astype(np.float64)
            denom = float(weighted_freq.sum())
            train_probs = (weighted_freq / max(1e-12, denom)).sort_values(ascending=False)
        else:
            train_probs = X_train[col].value_counts(normalize=True)
        test_probs = X_test_ref[col].value_counts(normalize=True)
        aligned = pd.concat([train_probs, test_probs], axis=1).fillna(0.0)
        tv_distance = 0.5 * np.abs(aligned.iloc[:, 0] - aligned.iloc[:, 1]).sum()
        drift_entry = _drift_entry(drift_report, col)
        drift_score = _categorical_drift_score(drift_entry)
        binary_shift = _binary_shift_score(drift_entry)
        temporal_instability = float(temporal_instability_map.get(col, 0.0))

        feature_smoothing = int(base_smoothing * (1.0 + 3.0 * min(1.0, tv_distance)))
        if col in feature_buckets.get("cat_high", []):
            if dual_te_smoothing_enabled:
                high_scale = (1.0 + 2.5 * min(1.0, tv_distance)) * (1.0 + 1.2 * min(1.0, drift_score))
                high_scale *= (1.0 + min(0.9, 2.0 * temporal_instability))
                feature_smoothing = int(max(feature_smoothing, high_card_smoothing_floor) * high_scale)
            else:
                # Preserve existing baseline behavior when dual smoothing is disabled.
                feature_smoothing = int(
                    max(feature_smoothing, high_card_smoothing_floor)
                    * (1.0 + 3.0 * min(1.0, tv_distance))
                )
        elif dual_te_smoothing_enabled and col in feature_buckets.get("cat_low", []):
            low_scale = 1.0 + 0.8 * min(1.0, tv_distance)
            low_scale *= (1.0 + min(0.35, 1.2 * temporal_instability))
            feature_smoothing = int(max(80, feature_smoothing) * low_scale)

        feature_smoothing = int(np.clip(feature_smoothing, 10, max(20, max_smoothing_cap)))
        smoothings[col] = feature_smoothing

        if recency_weighted_te_enabled:
            weighted_sum = (target_series * row_te_weights).groupby(cat_raw, observed=False).sum().astype(np.float64)
            weighted_cnt = row_te_weights.groupby(cat_raw, observed=False).sum().astype(np.float64)
            grp_mean = (weighted_sum / (weighted_cnt + 1e-12)).fillna(global_mean)
            grp_cnt = weighted_cnt
        else:
            grp_mean = target_series.groupby(cat_raw, observed=False).mean()
            grp_cnt = target_series.groupby(cat_raw, observed=False).count().astype(np.float64)
        if float(grp_cnt.sum()) > 0.0:
            w = grp_cnt / float(grp_cnt.sum())
            assoc_score = float(np.sqrt(max(0.0, np.sum(w * np.square(grp_mean - global_mean)) / target_var)))
        else:
            assoc_score = 0.0

        force_te = (
            col in feature_buckets.get("cat_low", [])
            and int(train_probs.shape[0]) >= max(2, force_te_card_thr)
            and assoc_score >= force_te_assoc_thr
        )

        if force_te:
            feature_smoothing = int(max(feature_smoothing, strong_te_floor * 2))
            smoothings[col] = feature_smoothing

        if (
            te_temporal_guard_enabled
            and (not force_te)
            and col in feature_buckets.get("cat_high", [])
            and temporal_instability >= te_temporal_guard_min_instability
            and drift_score >= te_temporal_guard_min_drift
        ):
            maps[col] = {}
            methods[col] = "neutral"
            continue

        # Generic, non-hardcoded mitigation policy by drift severity:
        # - severe binary shift => neutralize (constant global mean)
        # - moderate binary shift => frequency encode
        # - severe low-card drift => neutralize (constant global mean)
        # - moderate low-card drift => frequency encode
        # - otherwise => smoothed target encoding
        if (not force_te) and col in feature_buckets.get("binary", []) and binary_shift >= bin_neutral_thr:
            maps[col] = {}
            methods[col] = "neutral"
            continue
        if (not force_te) and col in feature_buckets.get("binary", []) and binary_shift >= bin_freq_thr:
            maps[col] = train_probs.to_dict()
            methods[col] = "freq"
            continue
        if (not force_te) and col in feature_buckets.get("cat_low", []) and (
            drift_score >= low_neutral_drift_thr or tv_distance >= low_neutral_tv_thr
        ):
            maps[col] = {}
            methods[col] = "neutral"
            continue
        if (not force_te) and col in feature_buckets.get("cat_low", []) and (
            drift_score >= low_freq_drift_thr or tv_distance >= low_freq_tv_thr
        ):
            maps[col] = train_probs.to_dict()
            methods[col] = "freq"
            continue

        if drift_score > 0.05:
            feature_smoothing = int(max(feature_smoothing, strong_te_floor))
            smoothings[col] = feature_smoothing

        if recency_weighted_te_enabled:
            weighted_sum = (target_series * row_te_weights).groupby(cat_raw, observed=False).sum().astype(np.float64)
            weighted_cnt = row_te_weights.groupby(cat_raw, observed=False).sum().astype(np.float64)
            te_vals = (weighted_sum + float(feature_smoothing) * global_mean) / (weighted_cnt + float(feature_smoothing))
            maps[col] = {
                str(k): float(v)
                for k, v in te_vals.items()
                if v is not None and np.isfinite(v)
            }
        else:
            cat_arr = cat_raw.to_numpy(dtype=object, copy=False)
            pl_te_frame = pl.DataFrame(
                {
                    "__cat_raw": cat_arr,
                    "__cat": cat_arr,
                    "__target": target_series.to_numpy(dtype=np.float64, copy=False),
                }
            )
            encoded = apply_fast_target_encoding(
                pl_te_frame.lazy(),
                target_col="__target",
                cat_cols=["__cat"],
                smoothing=float(feature_smoothing),
            ).collect()
            cat_map_df = encoded.group_by("__cat_raw").agg(pl.col("__cat").mean().alias("__te"))
            maps[col] = {
                str(k): float(v)
                for k, v in zip(cat_map_df["__cat_raw"].to_list(), cat_map_df["__te"].to_list())
                if v is not None and np.isfinite(v)
            }
        methods[col] = "te"

    return {
        "maps": maps,
        "global_mean": global_mean,
        "smoothing": smoothings,
        "methods": methods,
        "temporal_instability": temporal_instability_map,
        "recency_weighted_te_enabled": bool(recency_weighted_te_enabled),
    }

def apply_target_encoding_plan(df: pd.DataFrame, te_plan: dict, in_place: bool = True):
    _ = in_place
    out = df
    global_mean = float(te_plan.get("global_mean", 0.5))
    methods = te_plan.get("methods", {})
    original_columns = out.columns.tolist()
    encoded_updates: dict[str, pd.Series] = {}
    for col, mapping in te_plan.get("maps", {}).items():
        if col in out.columns:
            method = methods.get(col)
            if method == "te":
                mapped = out[col].astype("string").fillna("__missing__").map(mapping)
                default_val = global_mean
            else:
                mapped = out[col].map(mapping)
                default_val = 0.0 if method == "freq" else global_mean
            encoded_updates[col] = pd.to_numeric(
                mapped,
                errors="coerce",
            ).fillna(default_val).astype(np.float32)

    if encoded_updates:
        updates_df = pd.DataFrame(encoded_updates, index=out.index)
        base_df = out.drop(columns=list(encoded_updates.keys()), errors="ignore")
        out = pd.concat([base_df, updates_df], axis=1)
        out = out.reindex(columns=original_columns)

    return out
