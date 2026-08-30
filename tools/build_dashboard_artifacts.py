"""Build bounded, real-run dashboard summaries without fitting a model.

This development utility is intentionally outside the competition pipeline. It reads
the pipeline's JSON artifacts, scans bounded deterministic samples, and writes one
versioned dashboard contract. It never reads ChurnStatus from test data.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.contracts import (  # noqa: E402
    TARGET_COLUMN,
    TIME_COLUMN,
    canonicalize_category,
    sorted_month_values,
)
from detection.detector import (  # noqa: E402
    categorical_total_variation,
    numeric_ks,
)
from mitigation.policy import (  # noqa: E402
    DROP,
    KEEP,
    KEEP_ACTIONS,
    KEEP_DRIFTED,
    KEEP_SYSTEMIC_DRIFT,
    REALIGN_OFFSET,
    REALIGN_SCALE,
    REMAP_CATEGORIES,
    REPAIR_ACTIONS,
    REVERSE_PERCENTILE,
)
from runtime.streaming import (  # noqa: E402
    collect_test_analysis_sample,
    collect_training_samples,
)

SCHEMA_NAME = "datadrift.dashboard_run"
SCHEMA_VERSION = "1.0.0"
MAX_CATEGORIES = 25
NUMERIC_BINS = 20


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required run artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _history_blocks(train: pd.DataFrame) -> list[list[str]]:
    months = sorted_month_values(train[TIME_COLUMN])
    width = 2 if len(months) >= 4 else 1
    return [months[index : index + width] for index in range(0, len(months), width)]


def _historical_shifts(
    train: pd.DataFrame,
    feature: str,
    kind: str,
    blocks: list[list[str]],
) -> list[dict[str, Any]]:
    month_text = train[TIME_COLUMN].astype("string")
    frames = [train.loc[month_text.isin(block)] for block in blocks]
    output: list[dict[str, Any]] = []
    for index in range(len(frames) - 1):
        if kind == "numeric":
            shift = numeric_ks(frames[index][feature], frames[index + 1][feature])
        else:
            shift = categorical_total_variation(
                canonicalize_category(frames[index][feature]),
                canonicalize_category(frames[index + 1][feature]),
            )
        output.append(
            {
                "from_months": blocks[index],
                "to_months": blocks[index + 1],
                "label": (
                    f"{'–'.join(blocks[index])} → "
                    f"{'–'.join(blocks[index + 1])}"
                ),
                "shift": float(shift),
            }
        )
    return output


def _finite_numeric(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    return values[np.isfinite(values)]


def _histogram(values: np.ndarray, edges: np.ndarray) -> list[int]:
    if values.size == 0:
        return [0] * (len(edges) - 1)
    return np.histogram(values, bins=edges)[0].astype(int).tolist()


def _monthly_percentiles(frame: pd.DataFrame, feature: str) -> np.ndarray:
    ranks = frame.groupby(TIME_COLUMN, observed=True)[feature].rank(
        pct=True,
        method="average",
    )
    return _finite_numeric(ranks)


def _numeric_distribution(
    recent: pd.DataFrame,
    test: pd.DataFrame,
    feature: str,
    decision: dict[str, Any],
) -> dict[str, Any]:
    action = str(decision["action"])
    if action == REVERSE_PERCENTILE:
        train_values = _monthly_percentiles(recent, feature)
        test_values = _monthly_percentiles(test, feature)
        repaired_values = 1.0 - test_values
        edges = np.linspace(0.0, 1.0, NUMERIC_BINS + 1, dtype=np.float64)
        return {
            "kind": "numeric",
            "representation": "within-month percentile",
            "bin_edges": edges.tolist(),
            "recent_training_counts": _histogram(train_values, edges),
            "test_counts": _histogram(test_values, edges),
            "repaired_test_counts": _histogram(repaired_values, edges),
            "recent_training_observations": int(train_values.size),
            "test_observations": int(test_values.size),
        }

    train_values = _finite_numeric(recent[feature])
    test_values = _finite_numeric(test[feature])
    repaired_values: np.ndarray | None = None
    if action == REALIGN_SCALE and decision.get("scale_factor") is not None:
        repaired_values = test_values / float(decision["scale_factor"])
    elif action == REALIGN_OFFSET and decision.get("offset") is not None:
        repaired_values = test_values - float(decision["offset"])
    combined_parts = [train_values, test_values]
    if repaired_values is not None:
        combined_parts.append(repaired_values)
    combined = np.concatenate(combined_parts)
    if combined.size == 0:
        edges = np.linspace(0.0, 1.0, NUMERIC_BINS + 1, dtype=np.float64)
    elif float(np.ptp(combined)) <= 1e-12:
        center = float(combined[0])
        edges = np.linspace(center - 0.5, center + 0.5, NUMERIC_BINS + 1)
    else:
        edges = np.histogram_bin_edges(combined, bins=NUMERIC_BINS)
    return {
        "kind": "numeric",
        "representation": "raw values",
        "bin_edges": edges.astype(float).tolist(),
        "recent_training_counts": _histogram(train_values, edges),
        "test_counts": _histogram(test_values, edges),
        "repaired_test_counts": (
            _histogram(repaired_values, edges)
            if repaired_values is not None
            else None
        ),
        "recent_training_observations": int(train_values.size),
        "test_observations": int(test_values.size),
    }


def _categorical_distribution(
    recent: pd.DataFrame,
    test: pd.DataFrame,
    feature: str,
    decision: dict[str, Any],
) -> dict[str, Any]:
    train_values = canonicalize_category(recent[feature])
    test_values = canonicalize_category(test[feature])
    train_counts = Counter(train_values.tolist())
    test_counts = Counter(test_values.tolist())
    mapping = {
        str(key): str(value)
        for key, value in dict(decision.get("category_mapping") or {}).items()
    }
    repaired_counts = (
        Counter(test_values.replace(mapping).tolist())
        if str(decision["action"]) == REMAP_CATEGORIES and mapping
        else None
    )
    combined = Counter(train_counts)
    combined.update(test_counts)
    if repaired_counts is not None:
        combined.update(repaired_counts)
    ordered = sorted(combined, key=lambda value: (-combined[value], str(value)))
    keep_limit = MAX_CATEGORIES
    has_other = len(ordered) > MAX_CATEGORIES
    if has_other:
        keep_limit -= 1
    categories = ordered[:keep_limit]
    remaining = ordered[keep_limit:]
    if remaining:
        categories.append("Other")

    def counts_for(source: Counter) -> list[int]:
        values = [int(source.get(category, 0)) for category in categories if category != "Other"]
        if remaining:
            values.append(int(sum(source.get(category, 0) for category in remaining)))
        return values

    recent_values = counts_for(train_counts)
    test_values_count = counts_for(test_counts)
    repaired_values_count = (
        counts_for(repaired_counts) if repaired_counts is not None else None
    )
    recent_total = max(1, int(sum(recent_values)))
    test_total = max(1, int(sum(test_values_count)))
    repaired_total = max(1, int(sum(repaired_values_count or [])))
    return {
        "kind": "categorical",
        "categories": [str(value) for value in categories],
        "recent_training_counts": recent_values,
        "test_counts": test_values_count,
        "recent_training_shares": [value / recent_total for value in recent_values],
        "test_shares": [value / test_total for value in test_values_count],
        "repaired_test_counts": repaired_values_count,
        "repaired_test_shares": (
            [value / repaired_total for value in repaired_values_count]
            if repaired_values_count is not None
            else None
        ),
        "recent_training_observations": int(sum(recent_values)),
        "test_observations": int(sum(test_values_count)),
        "category_limit": MAX_CATEGORIES,
        "other_included": bool(remaining),
    }


def _decision_metadata(action: str) -> tuple[str, str]:
    if action == REVERSE_PERCENTILE:
        return "REPAIR", "Reverse percentile representation"
    if action == REALIGN_SCALE:
        return "REPAIR", "Scale realignment"
    if action == REALIGN_OFFSET:
        return "REPAIR", "Offset realignment"
    if action == REMAP_CATEGORIES:
        return "REPAIR", "Category remapping"
    if action == DROP:
        return "DROP", "Exclude unreliable feature"
    if action == KEEP_DRIFTED:
        return "KEEP", "Preserve same-semantic population drift"
    if action == KEEP_SYSTEMIC_DRIFT:
        return "KEEP", "Preserve systemic drift"
    return "KEEP", "No mitigation applied"


def build_dashboard_run(
    train_path: Path,
    test_path: Path,
    run_artifact_dir: Path,
    external_evaluation_path: Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    metrics = _read_json(run_artifact_dir / "latest_metrics.json")
    drift_report = _read_json(run_artifact_dir / "drift_report.json")
    preparation = _read_json(run_artifact_dir / "preparation_plan.json")
    if metrics.get("model_fit_count") != 1:
        raise ValueError("Dashboard artifacts require a successful one-fit run.")

    train_sample = collect_training_samples(train_path)
    test_sample = collect_test_analysis_sample(test_path)
    train = train_sample.analysis_frame
    test = test_sample.analysis_frame
    if TARGET_COLUMN in test.columns:
        raise AssertionError("Dashboard builder must never retain test ChurnStatus.")

    evidence_rows = drift_report.get("feature_actions")
    if not isinstance(evidence_rows, list) or not evidence_rows:
        raise ValueError("drift_report.json has no feature action rows.")
    blocks = _history_blocks(train)
    if not blocks:
        raise ValueError("Training analysis sample contains no Month history.")
    recent_months = blocks[-1]
    recent = train.loc[train[TIME_COLUMN].astype("string").isin(recent_months)]

    features: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    for raw in evidence_rows:
        feature = str(raw["feature"])
        kind = str(raw["kind"])
        action = str(raw["action"])
        if action not in KEEP_ACTIONS | REPAIR_ACTIONS | {DROP}:
            raise ValueError(f"Unsupported action {action!r} for {feature!r}.")
        if feature not in train.columns or feature not in test.columns:
            raise ValueError(f"Feature {feature!r} is absent from bounded source samples.")
        display_action, technical_method = _decision_metadata(action)
        history = _historical_shifts(train, feature, kind, blocks)
        distribution = (
            _numeric_distribution(recent, test, feature, raw)
            if kind == "numeric"
            else _categorical_distribution(recent, test, feature, raw)
        )
        action_counts[display_action] += 1
        features.append(
            {
                "feature": feature,
                "type": kind,
                "status": "CHANGED" if action != KEEP else "STABLE",
                "detection_method": (
                    "Kolmogorov–Smirnov distribution comparison"
                    if kind == "numeric"
                    else "Total Variation Distance"
                ),
                "evidence": {
                    "current_shift": float(raw["current_shift"]),
                    "historical_max_shift": float(raw["historical_max_shift"]),
                    "historical_shifts": history,
                    "novelty_ratio": float(raw["novelty_ratio"]),
                    "predictive_strength": float(raw["predictive_strength"]),
                    "missing_shift": float(raw["missing_shift"]),
                    "historical_missing_shift": float(
                        raw.get("historical_missing_shift", 0.0)
                    ),
                    "missing_novelty_ratio": float(
                        raw.get("missing_novelty_ratio", 0.0)
                    ),
                    "known_category_support_ratio": float(
                        raw.get("known_category_support_ratio", 1.0)
                    ),
                    "systemic_drift_fraction": float(
                        raw.get("systemic_drift_fraction", 0.0)
                    ),
                    "systemic_drift_active": bool(
                        raw.get("systemic_drift_active", False)
                    ),
                    "unique_values": int(raw["unique_values"]),
                    "recent_unique_values": int(raw["recent_unique_values"]),
                    "test_unique_values": int(raw["test_unique_values"]),
                    "support_retention_ratio": float(raw["support_retention_ratio"]),
                    "orientation_train": float(raw["orientation_train"]),
                    "orientation_test": float(raw["orientation_test"]),
                    "orientation_anchor_count": int(raw["orientation_anchor_count"]),
                },
                "decision": {
                    "action": action,
                    "display_action": display_action,
                    "reason": str(raw["reason"]),
                    "technical_method": technical_method,
                    "repair_parameters": {
                        "estimated_scale_factor": raw.get("scale_factor"),
                        "estimated_offset": raw.get("offset"),
                        "mapped_category_count": len(raw.get("category_mapping") or {}),
                        "mapping_summary": [
                            {"current": str(key), "reference": str(value)}
                            for key, value in list(
                                dict(raw.get("category_mapping") or {}).items()
                            )[:10]
                        ],
                    },
                    "evidence_checks": dict(raw.get("evidence_checks") or {}),
                },
                "distribution": distribution,
            }
        )

    external_evaluation: dict[str, Any] = {
        "status": "not_available",
        "label": "External labelled evaluation",
        "public_test_auprc": None,
    }
    if external_evaluation_path is not None and external_evaluation_path.is_file():
        external = _read_json(external_evaluation_path)
        score = external.get("public_test_auprc")
        if score is not None:
            external_evaluation = {
                "status": "available",
                "label": "External labelled evaluation",
                "public_test_auprc": float(score),
                "row_count": int(external.get("row_count", 0)),
                "id_order_exact": bool(external.get("id_order_exact", False)),
                "probabilities_valid": bool(
                    external.get("probabilities_valid", False)
                ),
                "is_pipeline_input": False,
            }

    elapsed = time.perf_counter() - started
    return {
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "mode": "REAL_RUN",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "train_file": train_path.name,
            "test_file": test_path.name,
            "run_artifact_directory": run_artifact_dir.name,
            "bounded_summary_only": True,
            "test_target_read": False,
        },
        "summary": {
            "features_analyzed": len(features),
            "features_kept": int(action_counts["KEEP"]),
            "features_repaired": int(action_counts["REPAIR"]),
            "features_dropped": int(action_counts["DROP"]),
        },
        "run": {
            "train_auprc": float(metrics["train_auprc"]),
            "external_evaluation": external_evaluation,
            "metrics": metrics,
            "thresholds": preparation.get("thresholds", {}),
            "model_contract": {
                "library": "lightgbm",
                "version": "4.6.0",
                "fit_count": int(metrics["model_fit_count"]),
            },
        },
        "history": {
            "blocks": blocks,
            "recent_training_months": recent_months,
        },
        "features": features,
        "generation": {
            "elapsed_seconds": elapsed,
            "train_rows_scanned": train_sample.total_rows,
            "train_rows_retained": int(len(train)),
            "test_rows_scanned": test_sample.total_rows,
            "test_rows_retained": int(len(test)),
            "maximum_categories_per_feature": MAX_CATEGORIES,
            "numeric_bins": NUMERIC_BINS,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build bounded real-run artifacts for the DataDrift dashboard."
    )
    parser.add_argument("--train_data_filepath", required=True)
    parser.add_argument("--test_data_filepath", required=True)
    parser.add_argument("--run_artifact_dir", required=True)
    parser.add_argument("--external_evaluation_filepath")
    parser.add_argument("--output_filepath", required=True)
    args = parser.parse_args()

    output_path = Path(args.output_filepath).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_dashboard_run(
        Path(args.train_data_filepath).resolve(),
        Path(args.test_data_filepath).resolve(),
        Path(args.run_artifact_dir).resolve(),
        (
            Path(args.external_evaluation_filepath).resolve()
            if args.external_evaluation_filepath
            else None
        ),
    )
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "output_path": str(output_path),
                "features": payload["summary"]["features_analyzed"],
                "bytes": output_path.stat().st_size,
                "elapsed_seconds": payload["generation"]["elapsed_seconds"],
                "test_target_read": payload["source"]["test_target_read"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
