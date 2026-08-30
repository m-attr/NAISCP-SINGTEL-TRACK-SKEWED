from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from artifacts.writer import write_json
from mitigation.policy import (
    KEEP_DRIFTED,
    KEEP_SYSTEMIC_DRIFT,
    REALIGN_OFFSET,
    REALIGN_SCALE,
    REMAP_CATEGORIES,
    REPAIR_ACTIONS,
)
from model.lightgbm_model import (
    fit_once,
    get_production_model_fit_count,
    reset_production_model_fit_count,
    target_to_array,
)
from preprocessing.plan import (
    TARGET_COLUMN,
    build_preparation_plan,
    transform_training,
)
from runtime.streaming import (
    collect_test_analysis_sample,
    collect_training_samples,
    stream_predictions_to_csv,
)
from runtime.telemetry import peak_rss_bytes


def _phase(
    phase_id: int,
    title: str,
    lines: list[str] | None = None,
    tables: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": phase_id,
        "title": title,
        "lines": lines or [],
        "tables": tables or [],
    }


def run_pipeline(
    train_data_filepath: str,
    test_data_filepath: str,
    quiet: bool = True,
    on_phase: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    del quiet
    started = time.perf_counter()
    phases: list[dict[str, Any]] = []

    def emit(phase: dict[str, Any]) -> None:
        phases.append(phase)
        if on_phase is not None:
            on_phase(phase)

    try:
        reset_production_model_fit_count()
        train_path = Path(train_data_filepath)
        test_path = Path(test_data_filepath)
        if not train_path.is_file() or not test_path.is_file():
            raise FileNotFoundError(
                "Both --train_data_filepath and --test_data_filepath must point to existing CSV files."
            )

        scan_started = time.perf_counter()
        train_samples = collect_training_samples(train_path)
        test_samples = collect_test_analysis_sample(test_path)
        scan_seconds = time.perf_counter() - scan_started
        training_frame = train_samples.model_frame
        if training_frame is None or training_frame.empty:
            raise ValueError("No training rows were retained.")
        if TARGET_COLUMN not in training_frame.columns:
            raise ValueError(f"Training data must contain {TARGET_COLUMN}.")

        emit(
            _phase(
                1,
                "Bounded Data Scan",
                [
                    f"Train rows scanned: {train_samples.total_rows:,}",
                    f"Train rows retained for the one final model: {len(training_frame):,}",
                    f"Train rows retained for drift analysis: {len(train_samples.analysis_frame):,}",
                    f"Test rows scanned for unlabeled analysis: {test_samples.total_rows:,}",
                    f"Test rows retained for analysis: {len(test_samples.analysis_frame):,}",
                    f"Bounded scan time (s): {scan_seconds:.4f}",
                    "Test ChurnStatus was excluded by column selection and was never read.",
                ],
            )
        )

        target = target_to_array(training_frame[TARGET_COLUMN])
        plan_started = time.perf_counter()
        plan = build_preparation_plan(
            training_frame,
            test_samples.analysis_frame,
            target,
            drift_training_frame=train_samples.analysis_frame,
        )
        transformed_train = transform_training(training_frame, plan)
        plan_seconds = time.perf_counter() - plan_started

        action_groups: dict[str, list[str]] = {}
        for row in plan.drift:
            if row.action != "KEEP":
                action_groups.setdefault(row.action, []).append(row.feature)
        action_rows = [
            [action, ", ".join(features)]
            for action, features in sorted(action_groups.items())
        ] or [["KEEP", "No intervention"]]

        emit(
            _phase(
                2,
                "Detect Drift and Freeze the Run Plan",
                [
                    f"Dropped features: {len(plan.dropped_features)}",
                    f"Features using stored percentile maps: {len(plan.aligned_features)}",
                    f"Direction-repaired features: {len(plan.reversed_features)}",
                    f"Scale-realigned features: {len(plan.scale_factors)}",
                    f"Offset-realigned features: {len(plan.offsets)}",
                    f"Category-remapped features: {len(plan.category_remaps)}",
                    f"Final model features: {len(plan.feature_order)}",
                    f"Detection, plan, and train preparation time (s): {plan_seconds:.4f}",
                    "All feature choices were generated from this run's data; no public non-contract feature name is hardcoded.",
                ],
                [{"headers": ["Action", "Features"], "rows": action_rows}],
            )
        )

        train_started = time.perf_counter()
        model, train_auprc = fit_once(
            transformed_train,
            target,
            categorical_features=plan.categorical_features,
        )
        train_seconds = time.perf_counter() - train_started
        fit_count = get_production_model_fit_count()
        if fit_count != 1:
            raise RuntimeError(
                f"Expected exactly one production model fit, observed {fit_count}."
            )

        emit(
            _phase(
                3,
                "Train One Official LightGBM",
                [
                    f"Model fit count: {fit_count}",
                    f"Train AU-PRC: {train_auprc:.12f}",
                    f"Model training time (s): {train_seconds:.4f}",
                ],
            )
        )

        output_started = time.perf_counter()
        prediction_path = Path(
            os.getenv("PREDICTION_OUTPUT_PATH", "prediction.csv")
        )
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_result = stream_predictions_to_csv(
            model=model,
            test_path=test_path,
            plan=plan,
            output_path=prediction_path,
            chunk_rows=test_samples.read_chunk_rows,
        )
        if prediction_result["rows_written"] != test_samples.total_rows:
            raise RuntimeError(
                "Prediction output row count does not match the test row count."
            )
        plan.save_summary("preparation_plan.json")
        output_seconds = time.perf_counter() - output_started
        total_seconds = time.perf_counter() - started
        peak_rss, peak_rss_source = peak_rss_bytes()

        metrics = {
            "success": True,
            "train_auprc": train_auprc,
            "model_fit_count": fit_count,
            "train_rows_scanned": train_samples.total_rows,
            "train_rows_used": int(len(training_frame)),
            "drift_analysis_train_rows": int(len(train_samples.analysis_frame)),
            "test_rows_scanned": test_samples.total_rows,
            "test_analysis_rows": int(len(test_samples.analysis_frame)),
            "prediction_rows": int(prediction_result["rows_written"]),
            "feature_count": int(len(plan.feature_order)),
            "dropped_features": plan.dropped_features,
            "aligned_features": plan.aligned_features,
            "reversed_features": plan.reversed_features,
            "scale_realigned_features": sorted(plan.scale_factors),
            "offset_realigned_features": sorted(plan.offsets),
            "category_remapped_features": sorted(plan.category_remaps),
            "keep_drifted_features": [
                row.feature for row in plan.drift if row.action == KEEP_DRIFTED
            ],
            "systemic_kept_features": [
                row.feature
                for row in plan.drift
                if row.action == KEEP_SYSTEMIC_DRIFT
            ],
            "repair_features": [
                row.feature for row in plan.drift if row.action in REPAIR_ACTIONS
            ],
            "action_counts": {
                action: sum(row.action == action for row in plan.drift)
                for action in sorted({row.action for row in plan.drift})
            },
            "streaming": {
                "train_estimated_bytes_per_row": train_samples.estimated_bytes_per_row,
                "test_estimated_bytes_per_row": test_samples.estimated_bytes_per_row,
                "train_read_chunk_rows": train_samples.read_chunk_rows,
                "test_read_chunk_rows": test_samples.read_chunk_rows,
                "model_sample_limit": train_samples.model_sample_limit,
                "analysis_sample_limit": train_samples.analysis_sample_limit,
                "test_analysis_sample_limit": test_samples.analysis_sample_limit,
                "maximum_prediction_chunk_rows": prediction_result[
                    "maximum_chunk_rows"
                ],
                "maximum_transformed_chunk_bytes": prediction_result[
                    "maximum_transformed_chunk_bytes"
                ],
            },
            "peak_process_rss_bytes": peak_rss,
            "peak_process_rss_measurement": peak_rss_source,
            "timings_seconds": {
                "bounded_train_test_analysis_scans": scan_seconds,
                "drift_plan_and_train_preparation": plan_seconds,
                "model_training": train_seconds,
                "streamed_test_transform_prediction_and_output": prediction_result[
                    "elapsed_seconds"
                ],
                "artifact_finalization": max(
                    0.0,
                    output_seconds - prediction_result["elapsed_seconds"],
                ),
                "total": total_seconds,
            },
        }
        write_json("latest_metrics.json", metrics)
        write_json(
            "latest_metrics_summary.json",
            {
                "success": True,
                "train_auprc": train_auprc,
                "model_fit_count": fit_count,
                "test_rows": int(prediction_result["rows_written"]),
                "runtime_seconds": total_seconds,
                "bounded_memory": True,
            },
        )
        write_json(
            "drift_report.json",
            {
                "dropped_features": plan.dropped_features,
                "aligned_features": plan.aligned_features,
                "reversed_features": plan.reversed_features,
                "scale_realigned_features": sorted(plan.scale_factors),
                "offset_realigned_features": sorted(plan.offsets),
                "category_remapped_features": sorted(plan.category_remaps),
                "feature_actions": [item.as_dict() for item in plan.drift],
                "note": (
                    "Test labels were never read. Training labels were used only for transparent "
                    "training-side usefulness and stable-profile checks; exactly one final LightGBM was fitted."
                ),
            },
        )

        emit(
            _phase(
                4,
                "Stream Competition Output",
                [
                    f"Output: {prediction_result['output_path']}",
                    f"Prediction rows: {prediction_result['rows_written']:,}",
                    f"Maximum rows held in one prediction chunk: {prediction_result['maximum_chunk_rows']:,}",
                    f"Streamed prediction/output time (s): {prediction_result['elapsed_seconds']:.4f}",
                    f"Total pipeline time (s): {total_seconds:.4f}",
                ],
            )
        )
        return {
            "success": True,
            "phases": phases,
            "metrics": metrics,
            "prediction_path": prediction_result["output_path"],
        }
    except Exception as exc:
        return {
            "success": False,
            "phases": phases,
            "error": f"{type(exc).__name__}: {exc}",
        }
