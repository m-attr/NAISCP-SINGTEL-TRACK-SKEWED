from __future__ import annotations

import gc
import json
import os
import time
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from model_trainer import train_and_predict, train_and_predict_raw_baseline

try:
    import psutil  # type: ignore
except Exception:
    psutil = None


def format_phase_box(content: str, min_inner_width: int = 28) -> str:
    """Format a single-content rectangular box using +---+ and |   | borders."""
    text = str(content).strip()
    inner_width = max(int(min_inner_width), len(text) + 2)
    border = "+" + ("-" * inner_width) + "+"
    line = "|" + text.center(inner_width) + "|"
    return "\n".join([border, line, border])


def format_ascii_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Format tabular rows into an ASCII table using +---+ and |   | borders."""
    if not headers:
        return ""

    safe_headers = [str(h) for h in headers]
    safe_rows = [["" if c is None else str(c) for c in row] for row in rows]
    col_count = len(safe_headers)
    widths = [len(safe_headers[i]) for i in range(col_count)]

    for row in safe_rows:
        for i in range(min(col_count, len(row))):
            widths[i] = max(widths[i], len(row[i]))

    def _border() -> str:
        return "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def _row(vals: list[str]) -> str:
        cells = []
        for i in range(col_count):
            v = vals[i] if i < len(vals) else ""
            cells.append(" " + v.ljust(widths[i]) + " ")
        return "|" + "|".join(cells) + "|"

    lines = [_border(), _row(safe_headers), _border()]
    for row in safe_rows:
        lines.append(_row(row))
    lines.append(_border())
    return "\n".join(lines)


def print_phase_banner(phase_number: int, phase_name: str) -> None:
    """Print a formatted phase box in the format 'Phase N: Name'."""
    label = f"Phase {int(phase_number)}: {str(phase_name).strip()}"
    print("\n" + format_phase_box(label))


def _env_truthy(name: str, default: str = "1") -> bool:
    """Read an environment flag with tolerant truthy parsing."""
    raw = os.getenv(name, default)
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _df_mem_gb(df: pd.DataFrame) -> float:
    """Estimate a DataFrame memory footprint in GB."""
    try:
        return float(df.memory_usage(deep=True).sum()) / (1024 ** 3)
    except Exception:
        return 0.0


def _short(value: Any, max_len: int = 280) -> str:
    """Render a compact debug-safe representation of any value."""
    text = repr(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _debug(enabled: bool, start_ts: float, message: str, **kwargs: Any) -> None:
    """Emit a timestamped debug line with optional memory and key-value context."""
    if not enabled:
        return
    now = datetime.now().strftime("%H:%M:%S")
    elapsed = time.perf_counter() - start_ts
    mem_text = ""
    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            mem_text = f" | ram_avail_gb={vm.available / (1024 ** 3):.2f}"
        except Exception:
            mem_text = ""
    kv = ""
    if kwargs:
        kv = " | " + ", ".join(f"{k}={_short(v)}" for k, v in kwargs.items())
    print(f"[DEBUG {now} +{elapsed:8.2f}s]{mem_text} {message}{kv}")


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to finite float; fallback to default when invalid."""
    try:
        out = float(value)
        if np.isfinite(out):
            return out
    except Exception:
        pass
    return float(default)


def _safe_float_or_none(value: Any) -> float | None:
    """Convert a value to finite float, else return None."""
    try:
        out = float(value)
        if np.isfinite(out):
            return out
    except Exception:
        pass
    return None


def _is_oom_error(exc: Exception) -> bool:
    """Detect common out-of-memory error signatures across runtimes."""
    if isinstance(exc, MemoryError):
        return True
    msg = str(exc).lower()
    patterns = (
        "out of memory",
        "memoryerror",
        "std::bad_alloc",
        "bad alloc",
        "cannot allocate memory",
        "insufficient memory",
        "resource exhausted",
    )
    return any(p in msg for p in patterns)


def _sample_rows(df: pd.DataFrame, n_rows: int, random_state: int) -> pd.DataFrame:
    """Return a defensive sampled copy bounded by n_rows."""
    if n_rows <= 0 or len(df) <= n_rows:
        return df.copy()
    return df.sample(n=n_rows, random_state=random_state).copy()


def _effective_sample_size(weights: np.ndarray) -> float:
    """Compute ESS from positive sample weights."""
    w = np.asarray(weights, dtype=np.float64)
    if w.size == 0:
        return 0.0
    w = np.clip(w, 1e-12, None)
    s1 = float(np.sum(w))
    s2 = float(np.sum(np.square(w)))
    if s2 <= 1e-18:
        return float(w.size)
    return float((s1 * s1) / s2)


def _stabilize_importance_weights(
    weights: np.ndarray,
    target_ess_ratio: float = 0.92,
    tail_clip_q: float = 0.995,
    max_shrink: float = 0.50,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply ESS-aware clipping and shrinkage to reduce weighting variance."""
    w = np.asarray(weights, dtype=np.float64)
    n = int(w.size)
    if n == 0:
        return w, {
            "applied": False,
            "reason": "empty",
        }

    w = np.clip(w, 1e-12, None)
    w = w / max(1e-12, float(np.mean(w)))

    ess_before = _effective_sample_size(w)
    ess_ratio_before = float(ess_before / max(1, n))

    q_hi = float(min(0.999, max(0.90, tail_clip_q)))
    q_lo = float(max(0.0, 1.0 - q_hi))
    lo = float(np.quantile(w, q_lo))
    hi = float(np.quantile(w, q_hi))
    w_clipped = np.clip(w, lo, hi)
    w_clipped = w_clipped / max(1e-12, float(np.mean(w_clipped)))

    ess_after_clip = _effective_sample_size(w_clipped)
    ess_ratio_after_clip = float(ess_after_clip / max(1, n))

    target = float(min(0.999, max(0.50, target_ess_ratio)))
    shrink_cap = float(min(1.0, max(0.0, max_shrink)))
    shrink = 0.0
    w_final = w_clipped

    if ess_ratio_after_clip < target and shrink_cap > 0.0:
        raw_shrink = (target - ess_ratio_after_clip) / max(target, 1e-12)
        shrink = float(min(shrink_cap, max(0.0, raw_shrink)))
        if shrink > 0.0:
            w_final = (1.0 - shrink) * w_clipped + shrink * np.ones_like(w_clipped)
            w_final = w_final / max(1e-12, float(np.mean(w_final)))

    ess_final = _effective_sample_size(w_final)
    ess_ratio_final = float(ess_final / max(1, n))

    meta: dict[str, Any] = {
        "applied": True,
        "n": n,
        "target_ess_ratio": target,
        "tail_clip_q": q_hi,
        "max_shrink": shrink_cap,
        "shrink_applied": float(shrink),
        "clip_lo": lo,
        "clip_hi": hi,
        "ess_before": float(ess_before),
        "ess_ratio_before": ess_ratio_before,
        "ess_after_clip": float(ess_after_clip),
        "ess_ratio_after_clip": ess_ratio_after_clip,
        "ess_final": float(ess_final),
        "ess_ratio_final": ess_ratio_final,
    }
    return w_final, meta


def _select_missing_shift_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_cols: int = 24,
    min_abs_shift: float = 0.03,
    quantile: float = 0.75,
    max_threshold: float = 0.25,
) -> tuple[list[str], dict[str, dict[str, float]]]:
    """Select columns with strongest missingness shift using dynamic quantile thresholds."""
    shared_cols = [c for c in train_df.columns if c in test_df.columns]
    if not shared_cols:
        return [], {}

    rows: list[tuple[str, float, float, float]] = []
    for col in shared_cols:
        train_na = float(train_df[col].isna().mean())
        test_na = float(test_df[col].isna().mean())
        shift = abs(train_na - test_na)
        if shift <= 0.0:
            continue
        rows.append((col, shift, train_na, test_na))

    if not rows:
        return [], {}

    shifts = np.asarray([r[1] for r in rows], dtype=np.float64)
    q = float(min(0.99, max(0.5, quantile)))
    max_thr = float(min(1.0, max(0.01, max_threshold)))
    dyn_thr = float(np.quantile(shifts, q)) if shifts.size else float(min_abs_shift)
    threshold = float(max(min_abs_shift, min(max_thr, dyn_thr)))

    selected = sorted([r for r in rows if r[1] >= threshold], key=lambda x: x[1], reverse=True)
    selected = selected[: max(1, int(max_cols))]

    details = {
        c: {
            "missing_shift": float(s),
            "train_missing_rate": float(tr),
            "test_missing_rate": float(te),
            "threshold": float(threshold),
        }
        for c, s, tr, te in selected
    }
    return [c for c, *_ in selected], details


def _run_strict_raw_baseline_with_fallback(
    train_df_raw: pd.DataFrame,
    test_df_raw: pd.DataFrame,
    target_column: str,
    predict_batch_size: int,
    train_cap_rows: int,
    test_cap_rows: int,
    debug_enabled: bool,
    debug_start: float,
) -> tuple[float, float | None, dict[str, list[float]] | None, dict[str, list[float]] | None]:
    """Execute strict raw baseline with OOM-aware progressive row fallback."""
    attempt_scales = [1.0, 0.60, 0.35]
    base_train_cap = min(len(train_df_raw), max(20_000, int(train_cap_rows)))
    base_test_cap = min(len(test_df_raw), max(10_000, int(test_cap_rows))) if len(test_df_raw) else 0

    for i, scale in enumerate(attempt_scales, start=1):
        attempt_train_rows = min(len(train_df_raw), max(20_000, int(base_train_cap * scale)))
        attempt_test_rows = (
            min(len(test_df_raw), max(10_000, int(base_test_cap * scale)))
            if len(test_df_raw)
            else 0
        )

        train_attempt = _sample_rows(train_df_raw, attempt_train_rows, random_state=120 + i)
        test_attempt = (
            _sample_rows(test_df_raw, attempt_test_rows, random_state=220 + i)
            if len(test_df_raw)
            else test_df_raw.copy()
        )

        print(
            "INFO: Strict raw baseline attempt "
            f"{i} with train_rows={len(train_attempt):,}, test_rows={len(test_attempt):,}."
        )

        try:
            _, pre_train_auprc, pre_test_auprc, pre_train_pr_curve, pre_test_pr_curve = train_and_predict_raw_baseline(
                train_attempt,
                test_attempt,
                target_column=target_column,
                predict_batch_size=predict_batch_size,
                return_pr_curves=True,
            )
            return pre_train_auprc, pre_test_auprc, pre_train_pr_curve, pre_test_pr_curve
        except Exception as exc:
            if not _is_oom_error(exc):
                raise
            print(f"WARNING: OOM during strict raw baseline attempt {i}. Retrying with smaller sample.")
            _debug(
                debug_enabled,
                debug_start,
                "Strict raw baseline OOM fallback triggered.",
                attempt=i,
                attempt_train_rows=attempt_train_rows,
                attempt_test_rows=attempt_test_rows,
                error=str(exc),
            )
            gc.collect()

    print("WARNING: Strict raw baseline skipped after repeated OOM fallback attempts.")
    return float("nan"), float("nan"), None, None


def _run_mitigated_training_with_fallback(
    X_train_model: pd.DataFrame,
    y_train_model: pd.Series | np.ndarray,
    X_test_ref: pd.DataFrame,
    y_test_ref: pd.Series | np.ndarray | None,
    final_sample_weights: np.ndarray,
    predict_batch_size: int,
    cat_cols: list[str],
    debug_enabled: bool,
    debug_start: float,
) -> tuple[np.ndarray, float, float | None, Any, dict[str, list[float]] | None, dict[str, list[float]] | None]:
    """Train mitigated model with OOM-aware progressive fallback."""
    attempt_scales = [1.0, 0.70, 0.45]
    n_train = len(X_train_model)

    for i, scale in enumerate(attempt_scales, start=1):
        if scale >= 0.999:
            X_train_attempt = X_train_model
            y_train_attempt = y_train_model
            w_attempt = final_sample_weights
            attempt_rows = n_train
        else:
            attempt_rows = min(n_train, max(20_000, int(n_train * scale)))
            idx = np.random.default_rng(42 + i).choice(n_train, size=attempt_rows, replace=False)
            idx = np.asarray(idx, dtype=np.int64)
            X_train_attempt = X_train_model.iloc[idx].copy()
            if hasattr(y_train_model, "iloc"):
                y_train_attempt = y_train_model.iloc[idx]
            else:
                y_train_attempt = np.asarray(y_train_model)[idx]
            w_attempt = np.asarray(final_sample_weights, dtype=np.float32)[idx]

        print(f"INFO: Mitigated model training attempt {i} with {attempt_rows:,} rows.")

        try:
            return train_and_predict(
                X_train_attempt,
                y_train_attempt,
                X_test_ref,
                y_test_ref,
                w_attempt,
                predict_batch_size=predict_batch_size,
                categorical_feature_names=cat_cols,
                return_model=True,
                return_pr_curves=True,
            )
        except Exception as exc:
            if not _is_oom_error(exc):
                raise
            print(f"WARNING: OOM during mitigated training attempt {i}. Retrying with fewer rows.")
            _debug(
                debug_enabled,
                debug_start,
                "Mitigated training OOM fallback triggered.",
                attempt=i,
                attempt_rows=attempt_rows,
                error=str(exc),
            )
            gc.collect()

    emergency_rows = min(n_train, 20_000)
    if emergency_rows < n_train:
        print("WARNING: Entering emergency low-memory training fallback with 20,000 rows.")
        idx = np.random.default_rng(99).choice(n_train, size=emergency_rows, replace=False)
        idx = np.asarray(idx, dtype=np.int64)
        X_emergency = X_train_model.iloc[idx].copy()
        if hasattr(y_train_model, "iloc"):
            y_emergency = y_train_model.iloc[idx]
        else:
            y_emergency = np.asarray(y_train_model)[idx]
        w_emergency = np.asarray(final_sample_weights, dtype=np.float32)[idx]
        return train_and_predict(
            X_emergency,
            y_emergency,
            X_test_ref,
            y_test_ref,
            w_emergency,
            predict_batch_size=predict_batch_size,
            categorical_feature_names=cat_cols,
            return_model=True,
            return_pr_curves=True,
        )

    raise RuntimeError("Mitigated training failed due to repeated OOM conditions.")


def _export_dashboard_artifacts(
    metrics: dict[str, Any],
    drift_report: dict[str, Any],
    drift_scores: dict[str, Any],
    dq_report: dict[str, Any],
    winsor_plan: dict[str, Any],
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    sample_rows: int = 5000,
) -> None:
    """Export lightweight dashboard artifacts used by the Streamlit app."""
    features = sorted(set(drift_report.keys()).union(set(drift_scores.keys())))
    rows: list[dict[str, Any]] = []

    for feature in features:
        meta = drift_scores.get(feature, {}) if isinstance(drift_scores, dict) else {}
        reason = str(drift_report.get(feature, "")) if isinstance(drift_report, dict) else ""

        distance_score = _safe_float(
            meta.get(
                "distance_score",
                max(
                    _safe_float(meta.get("ks_stat", 0.0)),
                    _safe_float(meta.get("psi", 0.0)),
                    _safe_float(meta.get("cramers_v", 0.0)),
                    _safe_float(meta.get("jsd", 0.0)),
                    _safe_float(meta.get("binary_max_diff", 0.0)),
                ),
            ),
            default=0.0,
        )
        p_value = meta.get("p_value", meta.get("ks_pvalue", meta.get("chi2_pvalue", np.nan)))
        status = "Drifted" if bool(reason) or bool(meta.get("flagged", False)) else "Stable"

        row: dict[str, Any] = {
            "feature": feature,
            "drift_score": distance_score,
            "p_value": _safe_float_or_none(p_value),
            "status": status,
            "reason": reason,
        }

        if isinstance(winsor_plan, dict) and feature in winsor_plan:
            bounds = winsor_plan.get(feature, {})
            if isinstance(bounds, dict):
                row["q_high"] = _safe_float_or_none(bounds.get("q_high"))
                row["q_low"] = _safe_float_or_none(bounds.get("q_low"))

        rows.append(row)

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "metrics": metrics,
        "features": rows,
        "winsor_bounds": {
            feat: {
                "q_low": _safe_float_or_none(info.get("q_low")),
                "q_high": _safe_float_or_none(info.get("q_high")),
            }
            for feat, info in (winsor_plan or {}).items()
            if isinstance(info, dict)
        },
        "data_quality_issues": dq_report or {},
    }

    with open("drift_report.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if isinstance(train_frame, pd.DataFrame) and not train_frame.empty:
        train_sample = train_frame.sample(n=min(sample_rows, len(train_frame)), random_state=42).reset_index(drop=True)
    else:
        train_sample = pd.DataFrame()

    if isinstance(test_frame, pd.DataFrame) and not test_frame.empty:
        test_sample = test_frame.sample(n=min(sample_rows, len(test_frame)), random_state=43).reset_index(drop=True)
    else:
        test_sample = pd.DataFrame()

    pl.from_pandas(train_sample, include_index=False).write_parquet("train_sample.parquet")
    pl.from_pandas(test_sample, include_index=False).write_parquet("test_sample.parquet")
