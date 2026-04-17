import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from scipy import stats
from scipy.spatial.distance import jensenshannon


def fast_approximate_ks(train_arr: np.ndarray, test_arr: np.ndarray, bins: int = 100) -> float:
    """
    O(N) histogram-based KS approximation for large continuous arrays.
    """
    if train_arr.size == 0 or test_arr.size == 0:
        return 0.0

    train_arr = train_arr[np.isfinite(train_arr)]
    test_arr = test_arr[np.isfinite(test_arr)]
    if train_arr.size == 0 or test_arr.size == 0:
        return 0.0

    min_val = min(np.nanmin(train_arr), np.nanmin(test_arr))
    max_val = max(np.nanmax(train_arr), np.nanmax(test_arr))

    if not np.isfinite(min_val) or not np.isfinite(max_val) or min_val == max_val:
        return 0.0

    train_hist, _ = np.histogram(train_arr, bins=bins, range=(min_val, max_val))
    test_hist, _ = np.histogram(test_arr, bins=bins, range=(min_val, max_val))

    train_cdf = np.cumsum(train_hist, dtype=np.float64) / float(train_arr.size)
    test_cdf = np.cumsum(test_hist, dtype=np.float64) / float(test_arr.size)
    return float(np.max(np.abs(train_cdf - test_cdf)))


def _approximate_ks_pvalue(ks_stat: float, n1: int, n2: int, terms: int = 5) -> float:
    """
    Asymptotic Kolmogorov p-value approximation from KS distance and sample sizes.
    """
    if not np.isfinite(ks_stat) or ks_stat <= 0.0 or n1 <= 0 or n2 <= 0:
        return 1.0

    n_eff = (float(n1) * float(n2)) / float(n1 + n2)
    if n_eff <= 0.0:
        return 1.0

    lam = (np.sqrt(n_eff) + 0.12 + 0.11 / np.sqrt(n_eff)) * float(ks_stat)
    if lam <= 0.0:
        return 1.0

    p = 0.0
    for j in range(1, max(2, int(terms) + 1)):
        p += ((-1) ** (j - 1)) * np.exp(-2.0 * (j ** 2) * (lam ** 2))
    p = 2.0 * p
    return float(np.clip(p, 0.0, 1.0))


def _detect_temporal_column_by_parseability(df: pd.DataFrame, sample_rows: int = 5000) -> str | None:
    text_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    if not text_cols:
        return None

    best_col = None
    best_ratio = 0.0
    for col in text_cols:
        s = df[col].dropna()
        if s.empty:
            continue
        if len(s) > sample_rows:
            s = s.iloc[:sample_rows]

        as_str = s.astype("string")
        try:
            parsed = pd.to_datetime(as_str, errors="coerce", format="mixed")
        except Exception:
            parsed = pd.to_datetime(as_str, errors="coerce")

        ratio = float(parsed.notna().mean())
        if ratio > best_ratio:
            best_ratio = ratio
            best_col = col

    if best_ratio >= 0.50:
        return best_col
    return None

def _sample_series(series: pd.Series, max_rows: int = 200000, random_state: int = 42):
    """
    Bound runtime and p-value inflation on very large datasets by sampling.
    """
    if len(series) > max_rows:
        series = series.sample(n=max_rows, random_state=random_state)
    return series.dropna()


def _to_numeric_target(y_train: pd.Series):
    if y_train is None:
        return None
    y_num = y_train.map({"Yes": 1, "No": 0, "yes": 1, "no": 0})
    if y_num.isna().any():
        y_num = pd.to_numeric(y_train, errors="coerce")
    return y_num.fillna(0).astype(np.int8)


def _ensure_drift_entry(drifted_features: dict, col: str, feature_type: str) -> dict:
    # REFACTORED: structured, nested drift payload per feature.
    entry = drifted_features.get(col)
    if not isinstance(entry, dict):
        entry = {
            "feature_type": feature_type,
            "distance_score": 0.0,
            "signals": [],
            "metrics": {},
        }
        drifted_features[col] = entry
    return entry


def _append_signal(
    drifted_features: dict,
    col: str,
    feature_type: str,
    metric: str,
    score: float,
    label: str,
    **extra,
):
    entry = _ensure_drift_entry(drifted_features, col, feature_type)
    signal = {
        "metric": metric,
        "score": float(score),
        "label": label,
    }
    if extra:
        signal.update(extra)
    entry["signals"].append(signal)
    metric_payload = {"score": float(score), "metric": metric}
    if extra:
        metric_payload.update(extra)
    entry["metrics"][metric] = metric_payload
    entry["distance_score"] = max(float(entry.get("distance_score", 0.0)), float(score))


def _compute_psi(train_arr: np.ndarray, test_arr: np.ndarray, n_bins: int = 10) -> float:
    """
    Population Stability Index using train-quantile bins.
    """
    if train_arr.size == 0 or test_arr.size == 0:
        return 0.0

    q = np.linspace(0.0, 1.0, n_bins + 1)
    try:
        bin_edges = np.quantile(train_arr, q)
    except Exception:
        return 0.0

    bin_edges = np.unique(bin_edges)
    if bin_edges.size < 2:
        return 0.0

    tr_hist, _ = np.histogram(train_arr, bins=bin_edges)
    te_hist, _ = np.histogram(test_arr, bins=bin_edges)

    tr_pct = tr_hist.astype(np.float64) / max(1.0, float(tr_hist.sum()))
    te_pct = te_hist.astype(np.float64) / max(1.0, float(te_hist.sum()))

    eps = 1e-6
    tr_pct = np.clip(tr_pct, eps, None)
    te_pct = np.clip(te_pct, eps, None)
    psi = np.sum((te_pct - tr_pct) * np.log(te_pct / tr_pct))
    return float(max(0.0, psi))


def _sorted_month_index(month_series: pd.Series):
    try:
        parsed = pd.to_datetime(month_series.astype(str), errors="coerce", format="mixed")
    except Exception:
        parsed = pd.to_datetime(month_series.astype(str), errors="coerce")
    helper = pd.DataFrame({"raw": month_series.astype(str), "parsed": parsed}).drop_duplicates("raw")
    if helper["parsed"].notna().any():
        helper = helper.sort_values("parsed", kind="stable")
    else:
        helper = helper.sort_values("raw", kind="stable")
    return helper["raw"].tolist()


def select_topk_features_by_gain(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_buckets: dict,
    top_k: int = 100,
    max_rows: int = 120000,
    random_state: int = 42
):
    """
    Fast feature-pruning pass using purely statistical target association scores.
    This keeps compliance with the one-model rule (no auxiliary model training).
    """
    all_cols = (
        feature_buckets.get("numerical", [])
        + feature_buckets.get("binary", [])
        + feature_buckets.get("cat_low", [])
        + feature_buckets.get("cat_high", [])
    )
    all_cols = [c for c in all_cols if c in X_train.columns]
    if y_train is None or not all_cols or top_k <= 0 or len(all_cols) <= top_k:
        return feature_buckets, {
            "applied": False,
            "reason": "disabled-or-insufficient-features",
            "selected": len(all_cols),
            "total": len(all_cols),
            "importance_gain": {},
        }

    y_num = _to_numeric_target(y_train)
    if y_num is None or y_num.nunique(dropna=True) < 2:
        return feature_buckets, {
            "applied": False,
            "reason": "invalid-target",
            "selected": len(all_cols),
            "total": len(all_cols),
            "importance_gain": {},
        }

    if len(X_train) > max_rows:
        sampled_idx = X_train.sample(n=max_rows, random_state=random_state).index
        X_fit = X_train.loc[sampled_idx, all_cols].copy()
        y_fit = y_num.loc[sampled_idx]
    else:
        X_fit = X_train[all_cols].copy()
        y_fit = y_num

    scores: dict[str, float] = {}
    y_arr = y_fit.to_numpy(dtype=np.float64, copy=False)
    y_mean = float(np.mean(y_arr))
    y_var = float(np.var(y_arr)) + 1e-12

    for col in all_cols:
        s = X_fit[col]
        if pd.api.types.is_numeric_dtype(s):
            x = pd.to_numeric(s, errors="coerce")
            if x.notna().sum() < 20:
                scores[col] = 0.0
                continue
            corr = x.corr(y_fit, method="spearman")
            scores[col] = float(abs(corr)) if pd.notna(corr) else 0.0
        else:
            cat = s.astype("string").fillna("__missing__")
            grp_mean = y_fit.groupby(cat, observed=False).mean()
            grp_cnt = y_fit.groupby(cat, observed=False).count().astype(np.float64)
            if grp_cnt.sum() <= 0:
                scores[col] = 0.0
                continue
            w = grp_cnt / grp_cnt.sum()
            # Weighted between-group variance ratio as categorical association score.
            between = float(np.sum(w * np.square(grp_mean - y_mean)))
            scores[col] = float(np.sqrt(max(0.0, between / y_var)))

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    selected_cols = {c for c, _ in ranked[:top_k]}

    # Keep high-card features available for dedicated TE mitigation downstream.
    selected_cols.update([c for c in feature_buckets.get("cat_high", []) if c in X_train.columns])

    if not selected_cols or max(scores.values(), default=0.0) <= 0.0:
        return feature_buckets, {
            "applied": False,
            "reason": "zero-statistical-score",
            "selected": len(selected_cols),
            "total": len(all_cols),
            "importance_gain": scores,
        }

    pruned_buckets = {}
    for key in ("numerical", "binary", "cat_low", "cat_high"):
        pruned_buckets[key] = [c for c in feature_buckets.get(key, []) if c in selected_cols]

    return pruned_buckets, {
        "applied": True,
        "reason": "topk-statistical-score",
        "selected": len(selected_cols),
        "total": len(all_cols),
        "importance_gain": scores,
    }

def detect_drifts(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_buckets: dict,
    fast_mode: bool = False,
    max_numeric_rows: int = 200000,
    max_categorical_rows: int = 250000,
    n_jobs: int = 1,
    return_scores: bool = False,
    use_dynamic_thresholds: bool = False,
):
    """
    Runs type-routed statistical drift tests with effect-size gates to reduce false positives on large data.
    When return_scores=True, returns (drifted_features, score_map) where score_map contains
    continuous effect-size/distance metrics for every tested feature.
    """
    drifted_features = {}
    score_map = {}
    worker_count = max(1, min(2, int(n_jobs) if n_jobs is not None else 1))

    tested_cols = []
    for key in ("numerical", "binary", "cat_low", "cat_high"):
        for col in feature_buckets.get(key, []):
            if col in train_df.columns and col in test_df.columns:
                tested_cols.append(col)
    tested_cols = list(dict.fromkeys(tested_cols))

    bucket_lookup = {}
    for key in ("numerical", "binary", "cat_low", "cat_high"):
        for col in feature_buckets.get(key, []):
            bucket_lookup[col] = key

    for col in tested_cols:
        score_map[col] = {
            "feature_type": bucket_lookup.get(col, "unknown"),
            "distance_score": 0.0,
            "flagged": False,
        }

    # 1. BINARY -> Proportion Math
    binary_max_diffs: dict[str, float] = {}
    for col in feature_buckets.get("binary", []):
        if col not in train_df.columns or col not in test_df.columns:
            continue
        train_sample = _sample_series(train_df[col], max_rows=max_categorical_rows)
        test_sample = _sample_series(test_df[col], max_rows=max_categorical_rows)
        if train_sample.empty or test_sample.empty:
            score_map.setdefault(col, {"feature_type": "binary", "distance_score": 0.0, "flagged": False})
            score_map[col]["binary_max_diff"] = 0.0
            continue

        aligned = pd.DataFrame(
            {
                "train": train_sample.value_counts(normalize=True),
                "test": test_sample.value_counts(normalize=True),
            }
        ).fillna(0)
        max_diff = float((aligned["train"] - aligned["test"]).abs().max())
        binary_max_diffs[col] = max_diff
        score_map[col]["binary_max_diff"] = max_diff
        score_map[col]["distance_score"] = max(float(score_map[col].get("distance_score", 0.0)), max_diff)

    binary_threshold = 0.05
    if use_dynamic_thresholds and binary_max_diffs:
        vals = np.asarray(list(binary_max_diffs.values()), dtype=np.float64)
        q1 = float(np.quantile(vals, 0.25))
        q3 = float(np.quantile(vals, 0.75))
        iqr = max(0.0, q3 - q1)
        dynamic = q3 + 0.5 * iqr
        binary_threshold = float(max(0.05, min(0.20, dynamic)))

    for col, max_diff in binary_max_diffs.items():
        if max_diff > binary_threshold:
            score_map[col]["flagged"] = True
            _append_signal(
                drifted_features,
                col,
                "binary",
                "binary_max_diff",
                max_diff,
                "Binary Proportion Shift",
                threshold=float(binary_threshold),
            )

    if fast_mode:
        max_numeric_rows = min(max_numeric_rows, 80000)
        max_categorical_rows = min(max_categorical_rows, 120000)

    # 2. NUMERICAL -> KS + PSI with zero-inflated guardrails
    def _numerical_drift(col):
        if col not in train_df.columns or col not in test_df.columns:
            return {
                "col": col,
                "ks_stat": 0.0,
                "ks_pvalue": 1.0,
                "psi": 0.0,
                "core_distance": 0.0,
                "zero_mass_diff": 0.0,
                "nz_ks_stat": 0.0,
                "nz_ks_pvalue": 1.0,
                "nz_psi": 0.0,
                "signals": [],
            }

        train_sample = _sample_series(train_df[col], max_rows=max_numeric_rows)
        test_sample = _sample_series(test_df[col], max_rows=max_numeric_rows)
        if train_sample.empty or test_sample.empty:
            return {
                "col": col,
                "ks_stat": 0.0,
                "ks_pvalue": 1.0,
                "psi": 0.0,
                "core_distance": 0.0,
                "zero_mass_diff": 0.0,
                "nz_ks_stat": 0.0,
                "nz_ks_pvalue": 1.0,
                "nz_psi": 0.0,
                "signals": [],
            }

        train_arr = train_sample.to_numpy(dtype=np.float64, copy=False)
        test_arr = test_sample.to_numpy(dtype=np.float64, copy=False)
        stat = float(fast_approximate_ks(train_arr, test_arr, bins=100))
        p_value = float(_approximate_ks_pvalue(stat, len(train_arr), len(test_arr)))
        psi = float(_compute_psi(train_arr, test_arr, n_bins=10))
        zero_train = float(np.mean(train_arr == 0.0))
        zero_test = float(np.mean(test_arr == 0.0))
        zero_diff = float(abs(zero_train - zero_test))
        is_zero_inflated = max(zero_train, zero_test) >= 0.60

        nz_train = train_arr[train_arr != 0.0]
        nz_test = test_arr[test_arr != 0.0]
        if nz_train.size > 20 and nz_test.size > 20:
            nz_stat = float(fast_approximate_ks(nz_train, nz_test, bins=100))
            nz_p_value = float(_approximate_ks_pvalue(nz_stat, int(nz_train.size), int(nz_test.size)))
            nz_psi = float(_compute_psi(nz_train, nz_test, n_bins=10))
        else:
            nz_stat, nz_p_value, nz_psi = 0.0, 1.0, 0.0

        signals = []
        if is_zero_inflated:
            if zero_diff > 0.08:
                signals.append(
                    {
                        "metric": "zero_mass_diff",
                        "score": float(zero_diff),
                        "label": "Zero-Mass Drift",
                        "threshold": 0.08,
                    }
                )
            if nz_p_value < 0.01 and nz_stat >= 0.05:
                signals.append(
                    {
                        "metric": "nz_ks_stat",
                        "score": float(nz_stat),
                        "label": "Numerical Drift (non-zero KS)",
                        "p_value": float(nz_p_value),
                        "threshold": 0.05,
                    }
                )
            if nz_psi > 0.15:
                signals.append(
                    {
                        "metric": "nz_psi",
                        "score": float(nz_psi),
                        "label": "PSI Drift (non-zero)",
                        "threshold": 0.15,
                    }
                )
            core_distance = max(float(zero_diff), float(nz_stat), float(min(1.0, nz_psi)))
        else:
            if p_value < 0.01 and stat >= 0.03:
                signals.append(
                    {
                        "metric": "ks_stat",
                        "score": float(stat),
                        "label": "Numerical Drift (KS)",
                        "p_value": float(p_value),
                        "threshold": 0.03,
                    }
                )
            if psi > 0.10:
                signals.append(
                    {
                        "metric": "psi",
                        "score": float(psi),
                        "label": "PSI Drift",
                        "threshold": 0.10,
                    }
                )
            core_distance = max(float(stat), float(min(1.0, psi)))

        return {
            "col": col,
            "ks_stat": float(stat),
            "ks_pvalue": float(p_value),
            "psi": float(psi),
            "core_distance": float(core_distance),
            "zero_mass_diff": float(zero_diff),
            "nz_ks_stat": float(nz_stat),
            "nz_ks_pvalue": float(nz_p_value),
            "nz_psi": float(nz_psi),
            "signals": signals,
        }

    numerical_cols = feature_buckets.get("numerical", [])
    if worker_count > 1 and len(numerical_cols) > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as ex:
            numerical_outputs = list(ex.map(_numerical_drift, numerical_cols))
    else:
        numerical_outputs = [_numerical_drift(col) for col in numerical_cols]

    for out in numerical_outputs:
        col = out["col"]
        score_map.setdefault(col, {"feature_type": "numerical", "distance_score": 0.0, "flagged": False})
        score_map[col]["ks_stat"] = out["ks_stat"]
        score_map[col]["ks_pvalue"] = out["ks_pvalue"]
        score_map[col]["psi"] = out["psi"]
        score_map[col]["zero_mass_diff"] = out["zero_mass_diff"]
        score_map[col]["nz_ks_stat"] = out["nz_ks_stat"]
        score_map[col]["nz_ks_pvalue"] = out["nz_ks_pvalue"]
        score_map[col]["nz_psi"] = out["nz_psi"]
        score_map[col]["distance_score"] = max(float(score_map[col].get("distance_score", 0.0)), float(out["core_distance"]))
        if out["signals"]:
            score_map[col]["flagged"] = True
            for signal in out["signals"]:
                extra = {k: v for k, v in signal.items() if k not in {"metric", "score", "label"}}
                _append_signal(
                    drifted_features,
                    col,
                    "numerical",
                    signal["metric"],
                    float(signal["score"]),
                    signal["label"],
                    **extra,
                )

    # 3. LOW-CARDINALITY CATEGORICAL -> Chi-Squared + Cramer's V
    def _cat_low_drift(col):
        if col not in train_df.columns or col not in test_df.columns:
            return {"col": col, "cramers_v": 0.0, "p_value": 1.0, "signal": None}

        train_sample = _sample_series(train_df[col], max_rows=max_categorical_rows)
        test_sample = _sample_series(test_df[col], max_rows=max_categorical_rows)
        if train_sample.empty or test_sample.empty:
            return {"col": col, "cramers_v": 0.0, "p_value": 1.0, "signal": None}

        aligned = pd.DataFrame({"train": train_sample.value_counts(), "test": test_sample.value_counts()}).fillna(0)
        aligned_arr = aligned.to_numpy(dtype=float, copy=False)
        stat, p_value, _, _ = stats.chi2_contingency(aligned_arr)

        n = aligned_arr.sum()
        min_dim = min(aligned.shape)
        if n == 0 or min_dim <= 1:
            return {"col": col, "cramers_v": 0.0, "p_value": float(p_value), "signal": None}

        cramer_v = float(np.sqrt((stat / n) / (min_dim - 1)))
        signal = None
        if p_value < 0.01 and cramer_v >= 0.10:
            signal = {
                "metric": "cramers_v",
                "score": cramer_v,
                "label": "Categorical Drift",
                "p_value": float(p_value),
                "threshold": 0.10,
            }
        return {"col": col, "cramers_v": cramer_v, "p_value": float(p_value), "signal": signal}

    cat_low_cols = feature_buckets.get("cat_low", [])
    if worker_count > 1 and len(cat_low_cols) > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as ex:
            cat_low_outputs = list(ex.map(_cat_low_drift, cat_low_cols))
    else:
        cat_low_outputs = [_cat_low_drift(col) for col in cat_low_cols]

    for out in cat_low_outputs:
        col = out["col"]
        score_map.setdefault(col, {"feature_type": "cat_low", "distance_score": 0.0, "flagged": False})
        score_map[col]["cramers_v"] = out["cramers_v"]
        score_map[col]["chi2_pvalue"] = out["p_value"]
        score_map[col]["distance_score"] = max(float(score_map[col].get("distance_score", 0.0)), float(out["cramers_v"]))
        if out["signal"] is not None:
            score_map[col]["flagged"] = True
            signal = out["signal"]
            _append_signal(
                drifted_features,
                col,
                "cat_low",
                signal["metric"],
                float(signal["score"]),
                signal["label"],
                p_value=float(signal["p_value"]),
                threshold=float(signal["threshold"]),
            )

    # 4. HIGH-CARDINALITY CATEGORICAL -> Jensen-Shannon Divergence (JSD)
    def _cat_high_drift(col):
        if col not in train_df.columns or col not in test_df.columns:
            return {"col": col, "jsd": 0.0, "signal": None}

        train_sample = _sample_series(train_df[col], max_rows=max_categorical_rows)
        test_sample = _sample_series(test_df[col], max_rows=max_categorical_rows)
        if train_sample.empty or test_sample.empty:
            return {"col": col, "jsd": 0.0, "signal": None}

        aligned = pd.DataFrame(
            {
                "train": train_sample.value_counts(normalize=True),
                "test": test_sample.value_counts(normalize=True),
            }
        ).fillna(0)
        train_probs = aligned["train"].to_numpy(dtype=float, copy=False)
        test_probs = aligned["test"].to_numpy(dtype=float, copy=False)
        js_div = float(jensenshannon(train_probs, test_probs))

        signal = None
        if js_div > 0.12:
            signal = {
                "metric": "jsd",
                "score": js_div,
                "label": "High-Card Drift",
                "threshold": 0.12,
            }
        return {"col": col, "jsd": js_div, "signal": signal}

    cat_high_cols = feature_buckets.get("cat_high", [])
    if worker_count > 1 and len(cat_high_cols) > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as ex:
            cat_high_outputs = list(ex.map(_cat_high_drift, cat_high_cols))
    else:
        cat_high_outputs = [_cat_high_drift(col) for col in cat_high_cols]

    for out in cat_high_outputs:
        col = out["col"]
        score_map.setdefault(col, {"feature_type": "cat_high", "distance_score": 0.0, "flagged": False})
        score_map[col]["jsd"] = out["jsd"]
        score_map[col]["distance_score"] = max(float(score_map[col].get("distance_score", 0.0)), float(out["jsd"]))
        if out["signal"] is not None:
            score_map[col]["flagged"] = True
            signal = out["signal"]
            _append_signal(
                drifted_features,
                col,
                "cat_high",
                signal["metric"],
                float(signal["score"]),
                signal["label"],
                threshold=float(signal["threshold"]),
            )

    # 5. Correlation Drift Detection (Spearman interaction structure drift)
    numerical_cols = [c for c in feature_buckets.get("numerical", []) if c in train_df.columns and c in test_df.columns]
    if len(numerical_cols) >= 2:
        tr_num = train_df[numerical_cols].apply(pd.to_numeric, errors="coerce")
        te_num = test_df[numerical_cols].apply(pd.to_numeric, errors="coerce")
        if len(tr_num) > max_numeric_rows:
            tr_num = tr_num.sample(n=max_numeric_rows, random_state=42)
        if len(te_num) > max_numeric_rows:
            te_num = te_num.sample(n=max_numeric_rows, random_state=42)

        train_med = tr_num.median(numeric_only=True)
        tr_num = tr_num.fillna(train_med).fillna(0.0)
        te_num = te_num.fillna(train_med).fillna(0.0)

        corr_train = tr_num.corr(method="spearman")
        corr_test = te_num.corr(method="spearman")
        corr_delta = (corr_train - corr_test).abs().fillna(0.0)
        np.fill_diagonal(corr_delta.values, 0.0)

        for col in numerical_cols:
            max_delta = float(corr_delta[col].max()) if col in corr_delta.columns else 0.0
            score_map[col]["interaction_max_corr_delta"] = max_delta
            if max_delta > 0.20:
                score_map[col]["flagged"] = True
                _append_signal(
                    drifted_features,
                    col,
                    "numerical",
                    "interaction_max_corr_delta",
                    max_delta,
                    "Interaction Drift",
                    threshold=0.20,
                )

    # 6. Time-aware Trend Detection (monotonic medians on detected temporal column)
    time_col = _detect_temporal_column_by_parseability(train_df)
    if time_col is not None and numerical_cols:
        month_order = _sorted_month_index(train_df[time_col])
        if len(month_order) >= 3:
            for col in numerical_cols:
                tmp = pd.DataFrame(
                    {
                        "__time": train_df[time_col].astype(str),
                        "__value": pd.to_numeric(train_df[col], errors="coerce"),
                    }
                )
                medians = tmp.groupby("__time", observed=False)["__value"].median()
                medians = medians.reindex(month_order).dropna()

                trend_strength = 0.0
                if len(medians) >= 3:
                    is_inc = bool(medians.is_monotonic_increasing)
                    is_dec = bool(medians.is_monotonic_decreasing)
                    non_constant = int(medians.nunique(dropna=True)) > 1
                    if non_constant:
                        idx = np.arange(len(medians), dtype=np.float64)
                        rho, _ = stats.spearmanr(idx, medians.to_numpy(dtype=np.float64, copy=False))
                        trend_strength = float(abs(0.0 if np.isnan(rho) else rho))

                    if non_constant and (is_inc or is_dec):
                        score_map[col]["flagged"] = True
                        direction = "increasing" if is_inc and not is_dec else "decreasing"
                        _append_signal(
                            drifted_features,
                            col,
                            "numerical",
                            "trend_spearman_abs",
                            trend_strength,
                            f"Trend Drift ({direction})",
                        )

                score_map[col]["trend_spearman_abs"] = trend_strength

    for col, entry in drifted_features.items():
        entry["distance_score"] = max(
            float(entry.get("distance_score", 0.0)),
            float(score_map.get(col, {}).get("distance_score", 0.0)),
        )
        entry["flagged"] = True

    if return_scores:
        return drifted_features, score_map
    return drifted_features

def detect_data_quality(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_buckets: dict,
    max_rows: int = 300000
):
    """
    Scans the raw data for structural and formatting anomalies before any cleaning occurs.
    """
    if len(train_df) > max_rows:
        train_df = train_df.sample(n=max_rows, random_state=42)
    if len(test_df) > max_rows:
        test_df = test_df.sample(n=max_rows, random_state=42)

    dq_features = {}
    
    # 1. Missing Value Rate Shifts
    for col in train_df.columns:
        if col in test_df.columns:
            train_na = train_df[col].isna().mean()
            test_na = test_df[col].isna().mean()
            if abs(train_na - test_na) > 0.05: # Flag if missing data shifts by >5%
                dq_features[col] = f"Data Quality: Missing value rate shifted from {train_na:.1%} to {test_na:.1%}"
                
    # 2. Case Inconsistency (e.g., "Month-to-month" vs "month-to-month")
    # We only check categorical columns for text issues
    for col in feature_buckets.get("cat_low", []) + feature_buckets.get("cat_high", []):
        if col in train_df.columns and col in test_df.columns:
            train_cats = set(train_df[col].dropna().astype(str))
            test_cats = set(test_df[col].dropna().astype(str))
            
            # Find categories in test that were not in train
            unseen = test_cats - train_cats
            if unseen:
                # Check if it is purely a capitalization issue
                train_cats_lower = set(x.lower() for x in train_cats)
                case_issues = [x for x in unseen if x.lower() in train_cats_lower]
                if case_issues:
                    dq_features[col] = f"Data Quality: Case casing drift detected (e.g., '{case_issues[0]}')"
                    
    # 3. Corrupted Primary Keys
    if "CustomerID" in test_df.columns:
        # Check for Excel scientific notation corruption (e.g., 0001-1E+43)
        has_sci = test_df['CustomerID'].astype(str).str.contains(r'E\+', na=False).any()
        if has_sci:
            dq_features["CustomerID"] = "Data Quality: Scientific notation corruption in IDs"
            
    return dq_features
