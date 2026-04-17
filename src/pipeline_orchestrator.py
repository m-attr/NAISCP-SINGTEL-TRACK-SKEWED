import argparse
import copy
import gc
import json
import math
import os
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stdout
from typing import Any, Callable

import numpy as np
import pandas as pd

from data_loader import load_raw_datasets, split_features_and_target
from data_preprocessor import (
    apply_missing_fill_plan,
    build_missing_fill_plan_with_sentinels,
    clean_categories,
    clean_and_engineer_features,
    convert_string_columns_to_category,
    encode_features,
)
from drift_detector import detect_data_quality, detect_drifts, select_topk_features_by_gain
from drift_mitigator import (
    add_categorical_shift_ratio_features,
    apply_adversarial_dropping,
    apply_kfold_target_encoding_with_plan,
    apply_stability_gated_pruning,
    apply_structure_plan,
    apply_target_encoding_plan,
    apply_winsor_plan,
    build_structure_plan,
    build_target_encoding_plan,
    build_winsor_plan,
    calibrate_winsor_plan_to_test_reference,
    get_distance_weights,
    get_statistical_weights,
    get_temporal_similarity_weights_fast,
    rank_normalize_drifted_numerics,
    select_stable_predictive_features,
    select_temporal_te_smoothing,
)
from feature_classifier import classify_features
from model_trainer import create_submission
from pipeline_runner import (
    apply_inmemory_cluster_pruning,
    build_partition_train_slices,
    load_previous_test_auprc,
    partition_predict_full_test,
    should_revert_feature_engineering,
)
from runtime_manager import build_runtime_plan
from schema_utils import detect_identifier_column, detect_target_column, detect_time_column
from strategic_sampler import apply_strategic_sampling
from utils import (
    _debug,
    _df_mem_gb,
    _env_truthy,
    _export_dashboard_artifacts,
    _run_mitigated_training_with_fallback,
    _run_strict_raw_baseline_with_fallback,
    _select_missing_shift_columns,
    _stabilize_importance_weights,
    print_phase_banner,
)

ID_COLUMN = "CustomerID"
TIME_COLUMN = "Month"
TARGET_COLUMN = "ChurnStatus"


def _new_pipeline_report(train_path: str, test_path: str) -> dict[str, Any]:
    return {
        "success": False,
        "error": None,
        "train_path": train_path,
        "test_path": test_path,
        "metrics": {},
        "runtime_phase_rows": [],
        "phases": [
            {"id": 1, "title": "Data Loading", "lines": [], "tables": []},
            {"id": 2, "title": "Data Pre-Processing", "lines": [], "tables": []},
            {"id": 3, "title": "Feature Classification", "lines": [], "tables": []},
            {"id": 4, "title": "Drift Detection", "lines": [], "tables": []},
            {"id": 5, "title": "Drift Mitigation", "lines": [], "tables": []},
            {"id": 6, "title": "Model Training", "lines": [], "tables": []},
            {"id": 7, "title": "Prediction Output", "lines": [], "tables": []},
            {"id": 8, "title": "Pipeline Complete", "lines": [], "tables": []},
        ],
    }


def _phase_entry(report: dict[str, Any], phase_id: int) -> dict[str, Any]:
    for phase in report.get("phases", []):
        if int(phase.get("id", -1)) == int(phase_id):
            return phase
    fallback = {"id": int(phase_id), "title": f"Phase {phase_id}", "lines": [], "tables": []}
    report.setdefault("phases", []).append(fallback)
    return fallback


def _phase_line(report: dict[str, Any], phase_id: int, text: str) -> None:
    _phase_entry(report, phase_id).setdefault("lines", []).append(str(text))


def _phase_table(report: dict[str, Any], phase_id: int, headers: list[str], rows: list[list[Any]]) -> None:
    _phase_entry(report, phase_id).setdefault("tables", []).append(
        {
            "headers": [str(h) for h in headers],
            "rows": rows,
        }
    )


def _emit_phase(report: dict[str, Any], phase_id: int, on_phase: Callable[[dict[str, Any]], None] | None) -> None:
    if on_phase is None:
        return
    phase_payload = copy.deepcopy(_phase_entry(report, phase_id))
    on_phase(phase_payload)


def _phase1_done(report: dict[str, Any], on_phase: Callable[[dict[str, Any]], None] | None) -> None:
    _emit_phase(report, 1, on_phase)


def _phase2_done(report: dict[str, Any], on_phase: Callable[[dict[str, Any]], None] | None) -> None:
    _emit_phase(report, 2, on_phase)


def _phase3_done(report: dict[str, Any], on_phase: Callable[[dict[str, Any]], None] | None) -> None:
    _emit_phase(report, 3, on_phase)


def _phase4_done(report: dict[str, Any], on_phase: Callable[[dict[str, Any]], None] | None) -> None:
    _emit_phase(report, 4, on_phase)


def _phase5_done(report: dict[str, Any], on_phase: Callable[[dict[str, Any]], None] | None) -> None:
    _emit_phase(report, 5, on_phase)


def _phase6_done(report: dict[str, Any], on_phase: Callable[[dict[str, Any]], None] | None) -> None:
    _emit_phase(report, 6, on_phase)


def _phase7_done(report: dict[str, Any], on_phase: Callable[[dict[str, Any]], None] | None) -> None:
    _emit_phase(report, 7, on_phase)


def _phase8_done(report: dict[str, Any], on_phase: Callable[[dict[str, Any]], None] | None) -> None:
    _emit_phase(report, 8, on_phase)


def _finalize_phase_runtime(
    report: dict[str, Any],
    phase_id: int,
    phase_start_perf: float,
    total_start_perf: float,
) -> float:
    now = time.perf_counter()
    phase_seconds = max(0.0, now - phase_start_perf)
    total_seconds = max(0.0, now - total_start_perf)
    phase_title = str(_phase_entry(report, phase_id).get("title", f"Phase {phase_id}"))
    report.setdefault("runtime_phase_rows", []).append(
        [f"Phase {phase_id}: {phase_title}", f"{phase_seconds:.2f} ({total_seconds:.2f})"]
    )
    return now


def _run_pipeline_with_args(
    args: argparse.Namespace,
    on_phase: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    report = _new_pipeline_report(args.train_data_filepath, args.test_data_filepath)
    debug_enabled = _env_truthy("PIPELINE_DEBUG", "1")
    debug_start = time.perf_counter()

    total_start_time = time.time()
    total_start_perf = time.perf_counter()
    phase_start_perf = total_start_perf
    _debug(
        debug_enabled,
        debug_start,
        "Pipeline invocation started.",
        train_path=args.train_data_filepath,
        test_path=args.test_data_filepath,
        cwd=os.getcwd(),
    )

    print_phase_banner(1, "Data Loading")

    previous_test_auprc = load_previous_test_auprc("latest_metrics.json")
    adv_feature_toggle_enabled = _env_truthy("ENABLE_ADV_DYNAMIC_FEATURES", "1")
    # Default OFF to keep one-command runs deterministic; opt-in during experiments.
    feature_gating_enabled = _env_truthy("ENABLE_FEATURE_BENCH_GATING", "0")
    in_revert_pass = _env_truthy("FEATURE_REVERT_ACTIVE", "0")

    try:
        target_column_name: str | None = None
        id_column_name: str | None = None
        time_column_name: str | None = None

        train_file_gb = os.path.getsize(args.train_data_filepath) / (1024 ** 3)
        test_file_gb = os.path.getsize(args.test_data_filepath) / (1024 ** 3)
        force_partition_mode = _env_truthy("FORCE_PARTITION_MODE", "0")
        partition_mode = force_partition_mode or (max(train_file_gb, test_file_gb) >= 0.5)
        _debug(
            debug_enabled,
            debug_start,
            "Input files profiled.",
            train_file_gb=round(train_file_gb, 4),
            test_file_gb=round(test_file_gb, 4),
            force_partition_mode=force_partition_mode,
            partition_mode=partition_mode,
        )

        runtime_plan = build_runtime_plan(
            train_path=args.train_data_filepath,
            test_path=args.test_data_filepath,
            reserve_free_gb=1.5,
        )
        print(
            "Runtime planner: "
            f"RAM total={runtime_plan.total_ram_gb:.1f} GB, "
            f"available={runtime_plan.available_ram_gb:.1f} GB, "
            f"protected free target={runtime_plan.protected_free_gb:.1f} GB, "
            f"working budget={runtime_plan.working_budget_gb:.1f} GB."
        )
        for note in runtime_plan.notes:
            print(f"NOTE: {note}")
        _debug(
            debug_enabled,
            debug_start,
            "Runtime plan built.",
            total_ram_gb=runtime_plan.total_ram_gb,
            available_ram_gb=runtime_plan.available_ram_gb,
            protected_free_gb=runtime_plan.protected_free_gb,
            working_budget_gb=runtime_plan.working_budget_gb,
            extra_safety_gb=runtime_plan.extra_safety_gb,
            effective_budget_gb=runtime_plan.effective_budget_gb,
            detection_reference_rows=runtime_plan.detection_reference_rows,
            train_reference_rows=runtime_plan.train_reference_rows,
            detection_chunk_rows=runtime_plan.detection_chunk_rows,
            predict_batch_size=runtime_plan.predict_batch_size,
            stream_chunksize=runtime_plan.stream_chunksize,
            detection_cap_rows=runtime_plan.detection_cap_rows,
            model_cap_rows=runtime_plan.model_cap_rows,
            baseline_train_cap_rows=runtime_plan.baseline_train_cap_rows,
            baseline_test_cap_rows=runtime_plan.baseline_test_cap_rows,
            detection_workers=runtime_plan.detection_workers,
        )
        mode_text = "partitioned" if partition_mode else "in-memory"
        if force_partition_mode and mode_text == "partitioned":
            mode_text += " (forced)"
        print(f"SUCCESS: Pipeline mode: {mode_text}")
        _phase_line(report, 1, f"Pipeline mode: {mode_text}")

        detection_cap = int(os.getenv("DETECTION_MAX_ROWS", str(runtime_plan.detection_cap_rows)))
        model_cap = int(os.getenv("MODEL_TRAIN_MAX_ROWS", str(runtime_plan.model_cap_rows)))

        detection_train_target_rows = min(max(40_000, detection_cap), runtime_plan.detection_reference_rows)
        model_train_target_rows = min(max(80_000, model_cap), runtime_plan.train_reference_rows)
        detection_test_target_rows = min(max(40_000, detection_cap), runtime_plan.detection_reference_rows)

        baseline_train_cap = min(
            model_train_target_rows,
            int(os.getenv("RAW_BASELINE_MAX_TRAIN_ROWS", str(runtime_plan.baseline_train_cap_rows))),
        )
        baseline_test_cap = min(
            detection_test_target_rows,
            int(os.getenv("RAW_BASELINE_MAX_TEST_ROWS", str(runtime_plan.baseline_test_cap_rows))),
        )

        _phase_table(
            report,
            1,
            ["Runtime Plan", "Value"],
            [
                ["Total RAM (GB)", round(float(runtime_plan.total_ram_gb), 2)],
                ["Available RAM (GB)", round(float(runtime_plan.available_ram_gb), 2)],
                ["Working Budget (GB)", round(float(runtime_plan.working_budget_gb), 2)],
                ["Detection Workers", int(runtime_plan.detection_workers)],
                ["Predict Batch Size", int(runtime_plan.predict_batch_size)],
                ["Stream Chunk Size", int(runtime_plan.stream_chunksize)],
            ],
        )
        _phase_table(
            report,
            1,
            ["Sampling Budget", "Rows"],
            [
                ["Detection Train Target", int(detection_train_target_rows)],
                ["Model Train Target", int(model_train_target_rows)],
                ["Detection Test Target", int(detection_test_target_rows)],
                ["Baseline Train Cap", int(baseline_train_cap)],
                ["Baseline Test Cap", int(baseline_test_cap)],
            ],
        )

        pre_train_auprc = float("nan")
        pre_test_auprc = float("nan")
        pre_train_pr_curve = None
        pre_test_pr_curve = None
        print(
            "SUCCESS: Decoupled sampling budgets -> "
            f"detection_train={detection_train_target_rows:,}, "
            f"model_train={model_train_target_rows:,}, "
            f"detection_test={detection_test_target_rows:,}"
        )

        if partition_mode:
            train_detect_df, train_model_df, test_ref_df = build_partition_train_slices(
                train_csv_path=args.train_data_filepath,
                test_csv_path=args.test_data_filepath,
                detection_train_rows=detection_train_target_rows,
                model_train_rows=model_train_target_rows,
                detection_test_rows=detection_test_target_rows,
                chunksize=runtime_plan.detection_chunk_rows,
                random_state=42,
            )
            if train_model_df.empty or train_detect_df.empty:
                raise ValueError("Unable to build partition train samples from train file.")
            if test_ref_df.empty:
                raise ValueError("Unable to build partition test reference sample from test file.")
            print(
                "SUCCESS: Built partition references from whole file. "
                f"Detection train rows: {len(train_detect_df):,}, "
                f"Model train rows: {len(train_model_df):,}, "
                f"Test reference rows: {len(test_ref_df):,}"
            )

            target_column_name = TARGET_COLUMN if TARGET_COLUMN in train_model_df.columns else next((c for c in train_model_df.columns if c not in test_ref_df.columns), None)
            if target_column_name is None:
                target_column_name = detect_target_column(train_model_df)
            if target_column_name is None:
                raise ValueError("Unable to identify target column for strict raw baseline.")

            print("INFO: Running strict raw baseline on partition sample.")
            print("SUCCESS: Running strict baseline on raw partition sample with OOM-safe fallback.")
            pre_train_auprc, pre_test_auprc, pre_train_pr_curve, pre_test_pr_curve = _run_strict_raw_baseline_with_fallback(
                train_df_raw=train_model_df,
                test_df_raw=test_ref_df,
                target_column=target_column_name,
                predict_batch_size=max(10000, int(runtime_plan.predict_batch_size)),
                train_cap_rows=baseline_train_cap,
                test_cap_rows=baseline_test_cap,
                debug_enabled=debug_enabled,
                debug_start=debug_start,
            )
            _debug(
                debug_enabled,
                debug_start,
                "Strict raw baseline complete (partition mode).",
                baseline_target_column=target_column_name,
                baseline_train_rows=len(train_model_df),
                baseline_test_rows=len(test_ref_df),
                pre_train_auprc=float(pre_train_auprc),
                pre_test_auprc=(None if pre_test_auprc is None else float(pre_test_auprc)),
            )
            print("SUCCESS: Strict raw baseline metrics captured from untouched loaded partition sample.")

            id_column_name = (
                ID_COLUMN if ID_COLUMN in test_ref_df.columns else (detect_identifier_column(test_ref_df) or detect_identifier_column(train_model_df))
            )
            time_column_name = TIME_COLUMN if TIME_COLUMN in train_model_df.columns else detect_time_column(train_model_df)

            X_train_detect, y_train_detect = split_features_and_target(train_detect_df, target_column=target_column_name)
            X_train_model, y_train_model = split_features_and_target(train_model_df, target_column=target_column_name)
            X_test_ref, y_test_ref = split_features_and_target(test_ref_df, target_column=target_column_name)
            test_ids_full = None
            _debug(
                debug_enabled,
                debug_start,
                "Partition sampling materialized.",
                train_detect_rows=len(X_train_detect),
                train_model_rows=len(X_train_model),
                test_ref_rows=len(X_test_ref),
                target_col=target_column_name,
                id_col=id_column_name,
                time_col=time_column_name,
            )
            del train_detect_df, train_model_df, test_ref_df
            gc.collect()
        else:
            train_df, test_df = load_raw_datasets(
                args.train_data_filepath,
                args.test_data_filepath,
                optimize_dtypes=False,
            )
            target_column_name = TARGET_COLUMN if TARGET_COLUMN in train_df.columns else next((c for c in train_df.columns if c not in test_df.columns), None)
            if target_column_name is None:
                target_column_name = detect_target_column(train_df)
            if target_column_name is None:
                raise ValueError("Unable to identify target column for strict raw baseline.")

            print("INFO: Running strict raw baseline on in-memory sample.")
            print("SUCCESS: Running strict baseline on raw loaded data with OOM-safe fallback.")
            pre_train_auprc, pre_test_auprc, pre_train_pr_curve, pre_test_pr_curve = _run_strict_raw_baseline_with_fallback(
                train_df_raw=train_df,
                test_df_raw=test_df,
                target_column=target_column_name,
                predict_batch_size=max(10000, int(runtime_plan.predict_batch_size)),
                train_cap_rows=baseline_train_cap,
                test_cap_rows=baseline_test_cap,
                debug_enabled=debug_enabled,
                debug_start=debug_start,
            )
            _debug(
                debug_enabled,
                debug_start,
                "Strict raw baseline complete (in-memory mode).",
                baseline_target_column=target_column_name,
                baseline_train_rows=len(train_df),
                baseline_test_rows=len(test_df),
                pre_train_auprc=float(pre_train_auprc),
                pre_test_auprc=(None if pre_test_auprc is None else float(pre_test_auprc)),
            )
            print("SUCCESS: Strict raw baseline metrics captured from untouched loaded data.")

            id_column_name = ID_COLUMN if ID_COLUMN in test_df.columns else (detect_identifier_column(test_df) or detect_identifier_column(train_df))
            time_column_name = TIME_COLUMN if TIME_COLUMN in train_df.columns else detect_time_column(train_df)
            test_ids_full = test_df[id_column_name].copy() if id_column_name and id_column_name in test_df.columns else None

            train_rows = len(train_df)
            model_n = min(model_train_target_rows, train_rows)
            detect_n = min(detection_train_target_rows, model_n)
            train_model_df = (
                train_df.sample(n=model_n, random_state=42).copy()
                if model_n < train_rows
                else train_df.copy()
            )
            train_detect_df = (
                train_model_df.sample(n=detect_n, random_state=43).copy()
                if detect_n < model_n
                else train_model_df.copy()
            )
            test_n = min(detection_test_target_rows, len(test_df))
            test_ref_df = (
                test_df.sample(n=test_n, random_state=44).copy()
                if test_n < len(test_df)
                else test_df.copy()
            )

            X_train_detect, y_train_detect = split_features_and_target(train_detect_df, target_column=target_column_name)
            X_train_model, y_train_model = split_features_and_target(train_model_df, target_column=target_column_name)
            X_test_ref, y_test_ref = split_features_and_target(test_ref_df, target_column=target_column_name)

            X_train_model, y_train_model, inmem_prune_meta = apply_inmemory_cluster_pruning(
                X_train_model,
                y_train_model,
                random_state=42,
            )
            if inmem_prune_meta.get("applied", False):
                print(
                    "SUCCESS: In-memory cluster-distance pruning applied "
                    f"({inmem_prune_meta.get('sampled_rows')}/{inmem_prune_meta.get('original_rows')} rows, "
                    f"strategy={inmem_prune_meta.get('strategy')})."
                )
                _phase_line(
                    report,
                    1,
                    "In-memory model-train pruning: "
                    f"{inmem_prune_meta.get('sampled_rows')}/{inmem_prune_meta.get('original_rows')} rows "
                    f"({inmem_prune_meta.get('strategy')}).",
                )
            else:
                print(f"SUCCESS: In-memory cluster-distance pruning skipped ({inmem_prune_meta.get('reason')}).")

            _debug(
                debug_enabled,
                debug_start,
                "In-memory sampling materialized.",
                train_detect_rows=len(X_train_detect),
                train_model_rows=len(X_train_model),
                test_ref_rows=len(X_test_ref),
                train_detect_mem_gb=round(_df_mem_gb(X_train_detect), 4),
                train_model_mem_gb=round(_df_mem_gb(X_train_model), 4),
                test_ref_mem_gb=round(_df_mem_gb(X_test_ref), 4),
                target_col=target_column_name,
                id_col=id_column_name,
                time_col=time_column_name,
            )
            del train_df, test_df, train_model_df, train_detect_df, test_ref_df
            gc.collect()

        _phase_table(
            report,
            1,
            ["Dataset", "Rows", "Columns"],
            [
                ["Train Detect", len(X_train_detect), X_train_detect.shape[1]],
                ["Train Model", len(X_train_model), X_train_model.shape[1]],
                ["Test Reference", len(X_test_ref), X_test_ref.shape[1]],
            ],
        )
        _phase_table(
            report,
            1,
            ["Baseline", "Value"],
            [
                ["Train AU-PRC", round(float(pre_train_auprc), 6)],
                [
                    "Test AU-PRC",
                    (None if pre_test_auprc is None else round(float(pre_test_auprc), 6)),
                ],
            ],
        )
        _phase_table(
            report,
            1,
            ["Schema Hint", "Column"],
            [
                ["Target", target_column_name],
                ["Identifier", id_column_name],
                ["Time", time_column_name],
            ],
        )
        phase_start_perf = _finalize_phase_runtime(report, 1, phase_start_perf, total_start_perf)
        _phase1_done(report, on_phase)

        # Drop unstable ID features before feature classification to avoid noisy drift artifacts.
        if id_column_name:
            X_train_detect.drop(columns=[id_column_name], errors="ignore", inplace=True)
            X_train_model.drop(columns=[id_column_name], errors="ignore", inplace=True)
            X_test_ref.drop(columns=[id_column_name], errors="ignore", inplace=True)

        # Missingness-shift aware mitigation (dynamic, data-driven).
        missing_shift_max_cols = int(os.getenv("MISSING_SHIFT_MAX_COLS", "24"))
        missing_shift_min_abs = float(os.getenv("MISSING_SHIFT_MIN_ABS", "0.03"))
        missing_shift_quantile = float(os.getenv("MISSING_SHIFT_QUANTILE", "0.75"))
        missing_shift_max_threshold = float(os.getenv("MISSING_SHIFT_MAX_THRESHOLD", "0.25"))

        missing_shift_cols, missing_shift_meta = _select_missing_shift_columns(
            X_train_detect,
            X_test_ref,
            max_cols=max(1, missing_shift_max_cols),
            min_abs_shift=max(0.0, missing_shift_min_abs),
            quantile=missing_shift_quantile,
            max_threshold=missing_shift_max_threshold,
        )
        missing_indicators: list[str] = [f"{c}__was_missing" for c in missing_shift_cols]
        if missing_indicators:
            for frame in (X_train_detect, X_train_model, X_test_ref):
                for src_col, ind_col in zip(missing_shift_cols, missing_indicators):
                    frame[ind_col] = frame[src_col].isna().astype(np.int8) if src_col in frame.columns else 0
        fill_plan: dict = build_missing_fill_plan_with_sentinels(
            X_train_model,
            sentinel_cols=missing_shift_cols,
            numeric_sentinel=-999.0,
        )
        # Keep native NaN handling for non-sentinel columns; only apply explicit
        # sentinel strategy to high missing-shift features.
        fill_plan["num_fill"] = {}
        fill_plan["cat_cols"] = []
        X_train_detect = apply_missing_fill_plan(X_train_detect, fill_plan)
        X_train_model = apply_missing_fill_plan(X_train_model, fill_plan)
        X_test_ref = apply_missing_fill_plan(X_test_ref, fill_plan)
        _debug(
            debug_enabled,
            debug_start,
            "Missingness-shift mitigation planned and applied.",
            missing_shift_columns=missing_shift_cols,
            missing_shift_meta=missing_shift_meta,
            missing_shift_params={
                "max_cols": max(1, missing_shift_max_cols),
                "min_abs_shift": max(0.0, missing_shift_min_abs),
                "quantile": float(min(0.99, max(0.5, missing_shift_quantile))),
                "max_threshold": float(min(1.0, max(0.01, missing_shift_max_threshold))),
            },
            missing_indicators_count=len(missing_indicators),
        )

        # Standardize text categories immediately after loading/splitting.
        X_train_detect = clean_categories(X_train_detect)
        X_train_model = clean_categories(X_train_model)
        X_test_ref = clean_categories(X_test_ref)
        _debug(
            debug_enabled,
            debug_start,
            "Categorical text standardized right after load.",
            action="lowercase+strip",
            train_detect_shape=X_train_detect.shape,
            train_model_shape=X_train_model.shape,
            test_ref_shape=X_test_ref.shape,
        )

        print(f"SUCCESS: X_train_detect shape: {X_train_detect.shape}")
        print(f"SUCCESS: X_train_model shape: {X_train_model.shape}")
        print(f"SUCCESS: y_train labels found: {y_train_model is not None}")
        _phase_line(report, 2, f"Target column: {target_column_name}")
        _phase_table(
            report,
            2,
            ["Frame", "Rows", "Columns"],
            [
                ["Train Detect", X_train_detect.shape[0], X_train_detect.shape[1]],
                ["Train Model", X_train_model.shape[0], X_train_model.shape[1]],
                ["Test Reference", X_test_ref.shape[0], X_test_ref.shape[1]],
            ],
        )
        _phase_table(
            report,
            2,
            ["Pre-Processing Signal", "Value"],
            [
                ["Missing-shift columns", len(missing_shift_cols)],
                ["Missing indicators", len(missing_indicators)],
            ],
        )
        if missing_shift_meta:
            shift_rows: list[list[Any]] = []
            for col, meta in list(missing_shift_meta.items())[:10]:
                shift_rows.append(
                    [
                        col,
                        round(float(meta.get("missing_shift", 0.0)), 6),
                        round(float(meta.get("train_missing_rate", 0.0)), 6),
                        round(float(meta.get("test_missing_rate", 0.0)), 6),
                    ]
                )
            _phase_table(
                report,
                2,
                ["Missing-Shift Column", "Shift", "Train NA", "Test NA"],
                shift_rows,
            )
        if y_train_model is not None:
            print(f"SUCCESS: Churn rate in training: {y_train_model.mean():.2%}")
            y_train_num = pd.to_numeric(pd.Series(y_train_model), errors="coerce").fillna(0.0)
            pos_count = int(np.sum(y_train_num.to_numpy(dtype=np.float64, copy=False) > 0.5))
            _phase_table(
                report,
                2,
                ["Target Stats", "Value"],
                [
                    ["Positive rate", round(float(y_train_model.mean()), 6)],
                    ["Rows", int(len(y_train_model))],
                    ["Positive count", pos_count],
                ],
            )
            _debug(
                debug_enabled,
                debug_start,
                "Target distribution profiled.",
                y_mean=float(y_train_model.mean()),
                y_std=float(y_train_model.std()),
                y_min=float(y_train_model.min()),
                y_max=float(y_train_model.max()),
            )
        phase_start_perf = _finalize_phase_runtime(report, 2, phase_start_perf, total_start_perf)
        _phase2_done(report, on_phase)

        print_phase_banner(2, "Feature Classification")

        fast_mode = bool(X_train_detect.shape[0] >= 1_000_000 or X_train_detect.shape[1] >= 200)
        classifier_sample_rows = 120000 if fast_mode else None
        feature_buckets = classify_features(X_train_detect, sample_rows=classifier_sample_rows)
        _debug(
            debug_enabled,
            debug_start,
            "Feature buckets computed.",
            fast_mode=fast_mode,
            classifier_sample_rows=classifier_sample_rows,
            numerical_preview=feature_buckets.get("numerical", [])[:12],
            binary_preview=feature_buckets.get("binary", [])[:12],
            cat_low_preview=feature_buckets.get("cat_low", [])[:12],
            cat_high_preview=feature_buckets.get("cat_high", [])[:12],
        )

        print("SUCCESS: Categorised features into buckets:")
        print(f"  - Numerical: {len(feature_buckets['numerical'])} columns")
        print(f"  - Binary: {len(feature_buckets['binary'])} columns")
        print(f"  - Low-Card (cat_low): {len(feature_buckets['cat_low'])} columns")
        print(f"  - High-Card (cat_high): {len(feature_buckets['cat_high'])} columns")
        _phase_table(
            report,
            3,
            ["Category", "Count"],
            [
                ["Numerical", len(feature_buckets["numerical"])],
                ["Binary", len(feature_buckets["binary"])],
                ["Low-Card", len(feature_buckets["cat_low"])],
                ["High-Card", len(feature_buckets["cat_high"])],
            ],
        )
        _phase_table(
            report,
            3,
            ["Category", "Sample Features"],
            [
                ["Numerical", ", ".join(feature_buckets.get("numerical", [])[:8])],
                ["Binary", ", ".join(feature_buckets.get("binary", [])[:8])],
                ["Low-Card", ", ".join(feature_buckets.get("cat_low", [])[:8])],
                ["High-Card", ", ".join(feature_buckets.get("cat_high", [])[:8])],
            ],
        )
        phase_start_perf = _finalize_phase_runtime(report, 3, phase_start_perf, total_start_perf)
        _phase3_done(report, on_phase)

        drift_start_time = time.time()

        print_phase_banner(3, "Data Quality Drift Detection")

        dq_report = detect_data_quality(X_train_detect, X_test_ref, feature_buckets, max_rows=600000)
        print(f"SUCCESS: Data Quality scan complete. Found {len(dq_report)} issues.")
        for col, reason in dq_report.items():
            print(f"  - {col}: {reason}")
        _debug(
            debug_enabled,
            debug_start,
            "Data quality scan complete.",
            dq_issue_count=len(dq_report),
            dq_issues=dq_report,
        )

        print_phase_banner(4, "Data Pre-Processing")

        X_train_detect = clean_and_engineer_features(X_train_detect)
        X_train_model = clean_and_engineer_features(X_train_model)
        X_test_ref = clean_and_engineer_features(X_test_ref)

        X_train_model, X_test_ref, converted_cat_cols = convert_string_columns_to_category(X_train_model, X_test_ref)

        print_phase_banner(5, "Statistical Drift Detection")

        effective_feature_buckets, topk_meta = select_topk_features_by_gain(
            X_train_detect,
            y_train_detect,
            feature_buckets,
            top_k=100,
            max_rows=120000,
            random_state=42,
        )
        if topk_meta.get("applied"):
            print(
                "SUCCESS: Top-K feature pruning for drift tests applied "
                f"({topk_meta['selected']}/{topk_meta['total']} features kept)."
            )
        else:
            print(f"SUCCESS: Top-K feature pruning skipped ({topk_meta.get('reason', 'n/a')}).")

        drift_out = detect_drifts(
            X_train_detect,
            X_test_ref,
            effective_feature_buckets,
            fast_mode=fast_mode,
            n_jobs=runtime_plan.detection_workers,
            return_scores=True,
            use_dynamic_thresholds=_env_truthy("ENABLE_DYNAMIC_DRIFT_THRESHOLDS", "0"),
        )
        if isinstance(drift_out, tuple):
            drift_report, drift_scores = drift_out
        else:
            drift_report = drift_out
            drift_scores = {}
        if drift_scores:
            for col, meta in drift_scores.items():
                distance = float(meta.get("distance_score", 0.0))
                # Guardrail: only escalate "severe" distance for structure dropping
                # when a direct cross-dataset drift metric is severe.
                direct_vals = [
                    float(meta.get("ks_stat", 0.0)),
                    float(min(1.0, meta.get("psi", 0.0))),
                    float(meta.get("cramers_v", 0.0)),
                    float(meta.get("jsd", 0.0)),
                    float(meta.get("binary_max_diff", 0.0)),
                    float(meta.get("zero_mass_diff", 0.0)),
                    float(meta.get("nz_ks_stat", 0.0)),
                    float(min(1.0, meta.get("nz_psi", 0.0))),
                ]
                severe_direct = max(direct_vals) if direct_vals else 0.0
                if severe_direct > 0.5:
                    entry = drift_report.get(col)
                    if not isinstance(entry, dict):
                        entry = {
                            "feature_type": meta.get("feature_type", "unknown"),
                            "signals": [],
                            "metrics": {},
                            "distance_score": 0.0,
                            "flagged": True,
                        }
                    metrics = entry.setdefault("metrics", {})
                    metrics["distance_score"] = {
                        "metric": "distance_score",
                        "score": float(distance),
                    }
                    signals = entry.setdefault("signals", [])
                    signals.append(
                        {
                            "metric": "distance_score_guardrail",
                            "score": float(distance),
                            "label": "Severe Distance Drift",
                            "threshold": 0.5,
                        }
                    )
                    entry["distance_score"] = max(float(entry.get("distance_score", 0.0)), float(distance))
                    entry["flagged"] = True
                    drift_report[col] = entry
        print(f"SUCCESS: Statistical Detection complete. Found {len(drift_report)} drifted features.")
        for col, reason in drift_report.items():
            if isinstance(reason, dict):
                signals = reason.get("signals", [])
                if signals:
                    parts = []
                    for signal in signals:
                        try:
                            score = float(signal.get("score", 0.0))
                        except (TypeError, ValueError):
                            score = 0.0
                        parts.append(f"{signal.get('label', 'Drift')} ({signal.get('metric', 'metric')}: {score:.4f})")
                    print(f"  - {col}: {'; '.join(parts)}")
                else:
                    print(f"  - {col}: structured drift entry (distance_score={float(reason.get('distance_score', 0.0)):.4f})")
            else:
                print(f"  - {col}: {reason}")
        if drift_scores:
            ranked_scores = sorted(
                drift_scores.items(),
                key=lambda x: float(x[1].get("distance_score", 0.0)),
                reverse=True,
            )
            _debug(
                debug_enabled,
                debug_start,
                "Drift score map generated.",
                scored_feature_count=len(drift_scores),
                top_distance_scores=[
                    (c, round(float(m.get("distance_score", 0.0)), 6), bool(m.get("flagged", False)))
                    for c, m in ranked_scores[:15]
                ],
            )
            for col, meta in ranked_scores:
                _debug(
                    debug_enabled,
                    debug_start,
                    "Per-feature drift score.",
                    feature=col,
                    feature_type=meta.get("feature_type"),
                    distance_score=round(float(meta.get("distance_score", 0.0)), 8),
                    flagged=bool(meta.get("flagged", False)),
                    details=meta,
                )

        _phase_table(
            report,
            4,
            ["Detection Summary", "Value"],
            [
                ["Data quality issues", len(dq_report)],
                ["Drifted features", len(drift_report)],
            ],
        )
        if dq_report:
            dq_rows = [[col, reason] for col, reason in list(dq_report.items())[:10]]
            _phase_table(report, 4, ["DQ Column", "Issue"], dq_rows)
        if drift_scores:
            top_drift_rows = []
            ranked_scores = sorted(
                drift_scores.items(),
                key=lambda x: float(x[1].get("distance_score", 0.0)),
                reverse=True,
            )
            by_type: dict[str, int] = {}
            for _, meta in drift_scores.items():
                ft = str(meta.get("feature_type", "unknown"))
                by_type[ft] = by_type.get(ft, 0) + 1
            for col, meta in ranked_scores[:8]:
                top_drift_rows.append(
                    [
                        col,
                        str(meta.get("feature_type", "unknown")),
                        round(float(meta.get("distance_score", 0.0)), 6),
                        bool(meta.get("flagged", False)),
                    ]
                )
            _phase_table(report, 4, ["Feature", "Type", "Distance", "Flagged"], top_drift_rows)
            _phase_table(
                report,
                4,
                ["Feature Type", "Count"],
                [[k, v] for k, v in sorted(by_type.items(), key=lambda x: x[0])],
            )
        phase_start_perf = _finalize_phase_runtime(report, 4, phase_start_perf, total_start_perf)
        _phase4_done(report, on_phase)

        rank_norm_cols: list[str] = []
        cat_shift_ratio_cols: list[str] = []
        separability_dropped: list[str] = []

        if _env_truthy("ENABLE_STRATEGIC_SAMPLING", "0"):
            X_train_model, y_train_model, sampling_meta = apply_strategic_sampling(
                X_train_model,
                y_train_model,
                X_test_ref,
                drift_report,
                effective_feature_buckets,
                random_state=42,
            )
            print(f"SUCCESS: Strategic sampling metadata: {sampling_meta}")
            _debug(
                debug_enabled,
                debug_start,
                "Strategic sampling applied.",
                sampling_meta=sampling_meta,
                sampled_shape=X_train_model.shape,
            )

        print_phase_banner(6, "Drift Mitigation")

        structure_plan = build_structure_plan(
            X_train_detect,
            effective_feature_buckets,
            drift_report,
            y_train=y_train_detect,
            X_test=X_test_ref,
        )
        X_train_detect = apply_structure_plan(X_train_detect, structure_plan)
        X_train_model = apply_structure_plan(X_train_model, structure_plan)
        X_test_ref = apply_structure_plan(X_test_ref, structure_plan)
        print("SUCCESS: Dropped detected identifier column and adaptively pruned noisy categorical features.")
        _debug(
            debug_enabled,
            debug_start,
            "Structure plan applied.",
            drop_cols=structure_plan.get("drop_cols", []),
            drop_count=len(structure_plan.get("drop_cols", [])),
            rare_label_map_cols=len(structure_plan.get("rare_label_map", {})),
        )

        separability_drop_enabled = _env_truthy("ENABLE_SEPARABILITY_DROPPING", "0")
        if separability_drop_enabled:
            before_shared_cols = set(X_train_detect.columns).intersection(set(X_test_ref.columns))
            X_train_detect, X_test_ref = apply_adversarial_dropping(X_train_detect, X_test_ref)
            after_shared_cols = set(X_train_detect.columns).intersection(set(X_test_ref.columns))
            separability_dropped = sorted(list(before_shared_cols - after_shared_cols))
            if separability_dropped:
                X_train_model = X_train_model.drop(columns=separability_dropped, errors="ignore")
                current_drop_cols = set(structure_plan.get("drop_cols", []))
                current_drop_cols.update(separability_dropped)
                structure_plan["drop_cols"] = sorted(current_drop_cols)
                print(
                    "SUCCESS: Applied separability dropping for highly train-test separable columns: "
                    f"{separability_dropped}"
                )
            else:
                print("SUCCESS: Separability dropping evaluated; no columns exceeded separability thresholds.")
            _debug(
                debug_enabled,
                debug_start,
                "Separability dropping pass complete.",
                separability_enabled=True,
                dropped_cols=separability_dropped,
                dropped_count=len(separability_dropped),
            )

        X_train_detect, X_test_ref, pruned_cols = apply_stability_gated_pruning(
            X_train_detect,
            X_test_ref,
            y_train_detect,
            drift_report,
            max_drops=int(os.getenv("STABILITY_GATED_MAX_DROPS", "2")),
        )
        if pruned_cols:
            print(f"SUCCESS: Stability-gated pruning dropped: {pruned_cols}")
            X_train_model = X_train_model.drop(columns=pruned_cols, errors="ignore")
        else:
            print("SUCCESS: Stability-gated pruning kept all remaining columns.")
        _debug(
            debug_enabled,
            debug_start,
            "Stability-gated pruning evaluated.",
            dropped_cols=pruned_cols,
            dropped_count=len(pruned_cols),
        )
        if pruned_cols:
            current_drop_cols = set(structure_plan.get("drop_cols", []))
            current_drop_cols.update(pruned_cols)
            structure_plan["drop_cols"] = sorted(current_drop_cols)

        winsor_plan = build_winsor_plan(X_train_detect, drift_report, effective_feature_buckets)
        winsor_plan = calibrate_winsor_plan_to_test_reference(winsor_plan, X_test_ref)
        X_train_detect = apply_winsor_plan(X_train_detect, winsor_plan, align_test_distribution=False)
        X_train_model = apply_winsor_plan(X_train_model, winsor_plan, align_test_distribution=False)
        X_test_ref = apply_winsor_plan(X_test_ref, winsor_plan, align_test_distribution=True)
        print("SUCCESS: Applied numerical realignment + dynamic IQR clipping to drifted numerical columns.")

        enable_quantile_companions = _env_truthy("ENABLE_SEVERE_QUANTILE_COMPANIONS", "0")
        if enable_quantile_companions and drift_scores:
            severe_num_cols: list[str] = []
            for col, meta in drift_scores.items():
                if str(meta.get("feature_type", "")) != "numerical":
                    continue
                psi_v = float(meta.get("psi", 0.0))
                ks_v = float(meta.get("ks_stat", 0.0))
                if (psi_v >= 0.25 or ks_v >= 0.20) and col in X_train_detect.columns and col in X_test_ref.columns:
                    severe_num_cols.append(col)

            added_qbins: list[str] = []
            for col in severe_num_cols:
                tr = pd.to_numeric(X_train_detect[col], errors="coerce").dropna()
                if len(tr) < 200:
                    continue
                quantiles = np.linspace(0.0, 1.0, 8)
                edges = np.unique(np.quantile(tr.to_numpy(dtype=np.float64, copy=False), quantiles))
                if len(edges) < 4:
                    continue
                qcol = f"{col}__qbin"
                for frame in (X_train_detect, X_train_model, X_test_ref):
                    vals = pd.to_numeric(frame[col], errors="coerce")
                    frame[qcol] = pd.to_numeric(
                        pd.cut(vals, bins=edges, labels=False, include_lowest=True),
                        errors="coerce",
                    ).fillna(-1).astype(np.float32)
                added_qbins.append(qcol)

            if added_qbins:
                print(f"SUCCESS: Added severe-drift quantile companion bins: {added_qbins}")
                _debug(
                    debug_enabled,
                    debug_start,
                    "Severe quantile companions added.",
                    severe_numeric_cols=severe_num_cols,
                    added_columns=added_qbins,
                )
        _debug(
            debug_enabled,
            debug_start,
            "Winsor/alignment plan applied.",
            winsor_feature_count=len(winsor_plan),
            winsor_features=list(winsor_plan.keys()),
        )
        for col, bounds in winsor_plan.items():
            _debug(
                debug_enabled,
                debug_start,
                "Per-feature winsor bounds.",
                feature=col,
                q_low=round(float(bounds.get("q_low", 0.0)), 6),
                q_high=round(float(bounds.get("q_high", 0.0)), 6),
                use_log=bool(bounds.get("use_log", False)),
                train_median=round(float(bounds.get("train_median", 0.0)), 6),
                test_median_ref=round(float(bounds.get("test_median_ref", 0.0)), 6),
            )

        rank_norm_enabled = _env_truthy("ENABLE_RANK_NORMALIZE_DRIFTED_NUMERIC", "1")
        rank_norm_max_features = int(os.getenv("RANK_NORMALIZE_MAX_FEATURES", "12"))
        if rank_norm_enabled and drift_scores:
            rank_time_col = time_column_name if time_column_name in X_train_detect.columns else detect_time_column(X_train_detect)
            X_train_detect, X_test_ref, rank_norm_cols = rank_normalize_drifted_numerics(
                X_train_detect,
                X_test_ref,
                drift_scores,
                time_col=rank_time_col,
                max_features=max(1, rank_norm_max_features),
            )
            if rank_norm_cols:
                X_train_model, X_test_ref, _ = rank_normalize_drifted_numerics(
                    X_train_model,
                    X_test_ref,
                    drift_scores,
                    time_col=(time_column_name if time_column_name in X_train_model.columns else detect_time_column(X_train_model)),
                    max_features=max(1, rank_norm_max_features),
                    include_cols=rank_norm_cols,
                )
                print(f"SUCCESS: Rank-normalized drifted numeric features: {rank_norm_cols}")
                _debug(
                    debug_enabled,
                    debug_start,
                    "Rank normalization applied to drifted numerics.",
                    rank_norm_cols=rank_norm_cols,
                    rank_norm_max_features=rank_norm_max_features,
                )

        cat_shift_ratio_enabled = _env_truthy("ENABLE_CATEGORY_SHIFT_RATIO_FEATURES", "0")
        cat_shift_ratio_max_cols = int(os.getenv("CATEGORY_SHIFT_RATIO_MAX_COLS", "8"))
        if cat_shift_ratio_enabled and drift_scores:
            X_train_model, X_test_ref, cat_shift_ratio_cols = add_categorical_shift_ratio_features(
                X_train_model,
                X_test_ref,
                drift_scores=drift_scores,
                feature_buckets=effective_feature_buckets,
                max_cols=max(1, cat_shift_ratio_max_cols),
                eps=float(os.getenv("CATEGORY_SHIFT_RATIO_EPS", "1e-6")),
            )
            if cat_shift_ratio_cols:
                print(f"SUCCESS: Added categorical shift-ratio features: {cat_shift_ratio_cols}")
                _debug(
                    debug_enabled,
                    debug_start,
                    "Categorical shift-ratio features added.",
                    shift_ratio_enabled=cat_shift_ratio_enabled,
                    added_cols=cat_shift_ratio_cols,
                    max_cols=cat_shift_ratio_max_cols,
                )

        tuned_smoothing = select_temporal_te_smoothing(
            X_train_detect,
            y_train_detect,
            X_test_ref,
            effective_feature_buckets,
            drift_report,
        )
        te_plan = build_target_encoding_plan(
            X_train_detect,
            X_test_ref,
            y_train_detect,
            effective_feature_buckets,
            drift_report,
            base_smoothing_override=tuned_smoothing,
        )
        X_train_detect = apply_target_encoding_plan(X_train_detect, te_plan)
        X_train_model = apply_kfold_target_encoding_with_plan(
            X_train_model,
            y_train_model,
            te_plan,
            n_splits=10,
            blend_alpha=0.75,
        )
        X_test_ref = apply_target_encoding_plan(X_test_ref, te_plan)
        print(
            f"SUCCESS: Applied detection-plan target encoding + K-fold train encoding "
            f"(base={tuned_smoothing})."
        )
        te_methods = te_plan.get("methods", {})
        te_count = sum(1 for m in te_methods.values() if m == "te")
        freq_count = sum(1 for m in te_methods.values() if m == "freq")
        _debug(
            debug_enabled,
            debug_start,
            "Categorical encoding plan applied.",
            tuned_smoothing=tuned_smoothing,
            encoded_feature_count=len(te_methods),
            te_method_count=te_count,
            freq_method_count=freq_count,
            te_methods=te_methods,
        )

        print("SUCCESS: Hybrid weighting mode active with light low-similarity pruning.")

        drift_distance_vals = [
            float(m.get("distance_score", 0.0)) for m in drift_scores.values()
        ] if drift_scores else []
        drift_distance_vals = [v for v in drift_distance_vals if np.isfinite(v)]
        mean_drift_distance = float(np.mean(drift_distance_vals)) if drift_distance_vals else 0.0

        temporal_power_default = float(np.clip(5.0 + 6.0 * mean_drift_distance, 5.0, 8.0))
        temporal_power = float(os.getenv("TEMPORAL_WEIGHT_POWER", f"{temporal_power_default:.2f}"))
        temporal_softmax_temp = float(os.getenv("TEMPORAL_SOFTMAX_TEMP", "0.20"))
        temporal_recency_blend = float(os.getenv("TEMPORAL_RECENCY_BLEND", "0.05"))

        temporal_weights, month_scores = get_temporal_similarity_weights_fast(
            X_train_detect,
            X_test_ref,
            drift_report,
            effective_feature_buckets,
            feature_importance_gain=topk_meta.get("importance_gain", {}),
            return_month_scores=True,
            sharpness_power=temporal_power,
            softmax_temperature=temporal_softmax_temp,
            recency_blend=temporal_recency_blend,
        )
        print("SUCCESS: Calculated Temporal Similarity Weights.")

        active_time_col = time_column_name if time_column_name in X_train_model.columns else detect_time_column(X_train_model)
        if active_time_col and month_scores:
            temporal_weights = pd.to_numeric(
                X_train_model[active_time_col].astype("string").map(month_scores),
                errors="coerce",
            ).fillna(1.0).to_numpy(dtype=float, copy=False)
            temporal_weights = np.power(np.clip(temporal_weights, 1e-3, None), temporal_power)
        else:
            temporal_weights = np.ones(len(X_train_model), dtype=np.float32)

        q05 = float(np.quantile(temporal_weights, 0.05))
        q95 = float(np.quantile(temporal_weights, 0.95))
        temporal_weights = np.clip(temporal_weights, q05, q95)

        _debug(
            debug_enabled,
            debug_start,
            "Temporal weights computed (pre-normalization).",
            min=float(np.min(temporal_weights)),
            p10=float(np.quantile(temporal_weights, 0.10)),
            p50=float(np.quantile(temporal_weights, 0.50)),
            p90=float(np.quantile(temporal_weights, 0.90)),
            max=float(np.max(temporal_weights)),
            month_scores=month_scores,
        )

        print("SUCCESS: Light pruning disabled. Retaining all training rows.")

        temporal_weights = temporal_weights / np.mean(temporal_weights)

        stat_weight_blend = float(os.getenv("STAT_WEIGHT_BLEND", "0.10"))
        if stat_weight_blend > 0.0:
            stat_weights = get_statistical_weights(X_train_model, X_test_ref, drift_report)
            stat_weights = np.asarray(stat_weights, dtype=np.float64)
            stat_weights = np.clip(stat_weights, 0.4, 2.5)
            stat_weights = stat_weights / np.mean(stat_weights)
            temporal_weights = temporal_weights * np.power(stat_weights, stat_weight_blend)
            temporal_weights = temporal_weights / np.mean(temporal_weights)
            print(f"SUCCESS: Blended statistical shift weights (blend={stat_weight_blend:.2f}).")

        distance_weight_blend_enabled = _env_truthy("ENABLE_DISTANCE_WEIGHT_BLEND", "0")
        if distance_weight_blend_enabled:
            distance_blend = float(os.getenv("DIST_WEIGHT_BLEND", "0.20"))
            distance_weights = get_distance_weights(X_train_model, X_test_ref)
            distance_weights = np.asarray(distance_weights, dtype=np.float64)
            distance_weights = np.clip(distance_weights, 0.80, 1.20)
            distance_weights = distance_weights / max(1e-12, float(np.mean(distance_weights)))
            temporal_weights = temporal_weights * np.power(distance_weights, max(0.0, distance_blend))
            temporal_weights = temporal_weights / max(1e-12, float(np.mean(temporal_weights)))
            print(f"SUCCESS: Blended distance-based shift weights (blend={distance_blend:.2f}).")
            _debug(
                debug_enabled,
                debug_start,
                "Distance-based covariate shift weights blended.",
                distance_blend=float(distance_blend),
                dist_w_min=float(np.min(distance_weights)) if len(distance_weights) else None,
                dist_w_p50=float(np.quantile(distance_weights, 0.50)) if len(distance_weights) else None,
                dist_w_max=float(np.max(distance_weights)) if len(distance_weights) else None,
            )

        ess_stabilization_enabled = _env_truthy("ENABLE_ESS_WEIGHT_STABILIZATION", "0")
        if ess_stabilization_enabled:
            target_ess_ratio = float(os.getenv("TARGET_ESS_RATIO", "0.92"))
            weight_tail_clip_q = float(os.getenv("WEIGHT_TAIL_CLIP_Q", "0.995"))
            max_weight_shrink = float(os.getenv("MAX_WEIGHT_SHRINK", "0.50"))
            temporal_weights, weight_stability_meta = _stabilize_importance_weights(
                temporal_weights,
                target_ess_ratio=target_ess_ratio,
                tail_clip_q=weight_tail_clip_q,
                max_shrink=max_weight_shrink,
            )
            print(
                "SUCCESS: Applied ESS-aware weight stabilization "
                f"(target={weight_stability_meta.get('target_ess_ratio', 0.0):.3f}, "
                f"ESS={weight_stability_meta.get('ess_ratio_final', 0.0):.3f})."
            )
            _debug(
                debug_enabled,
                debug_start,
                "ESS-aware weight stabilization applied.",
                weight_stability_meta=weight_stability_meta,
            )

        _debug(
            debug_enabled,
            debug_start,
            "Temporal weights normalized.",
            mean=float(np.mean(temporal_weights)),
            std=float(np.std(temporal_weights)),
            min=float(np.min(temporal_weights)),
            max=float(np.max(temporal_weights)),
        )

        dynamic_month_subset_enabled = _env_truthy("ENABLE_DYNAMIC_MONTH_SUBSET_SELECTION", "0")
        if dynamic_month_subset_enabled and active_time_col and month_scores and len(X_train_model) > 0:
            keep_ratio = float(os.getenv("MONTH_SUBSET_KEEP_RATIO", "0.65"))
            keep_ratio = float(np.clip(keep_ratio, 0.35, 1.0))
            min_rows = int(os.getenv("MONTH_SUBSET_MIN_ROWS", "18000"))
            min_rows = max(5000, min_rows)

            month_sim = pd.to_numeric(
                X_train_model[active_time_col].astype("string").map(month_scores),
                errors="coerce",
            ).fillna(1.0).to_numpy(dtype=np.float64, copy=False)

            n_rows = int(len(month_sim))
            keep_n = int(max(min_rows, int(np.ceil(n_rows * keep_ratio))))
            keep_n = min(n_rows, keep_n)

            if keep_n < n_rows:
                keep_order = np.argsort(month_sim)[::-1][:keep_n]
                keep_idx = np.sort(keep_order)
                y_subset = y_train_model.iloc[keep_idx]
                if y_subset.nunique(dropna=False) >= 2:
                    X_train_model = X_train_model.iloc[keep_idx].copy()
                    y_train_model = y_subset.copy()
                    temporal_weights = np.asarray(temporal_weights, dtype=np.float64)[keep_idx]
                    temporal_weights = temporal_weights / max(1e-12, float(np.mean(temporal_weights)))
                    print(
                        "SUCCESS: Applied dynamic month-similarity subset selection "
                        f"({keep_n}/{n_rows} rows kept)."
                    )
                    _debug(
                        debug_enabled,
                        debug_start,
                        "Dynamic month-similarity subset applied.",
                        subset_enabled=dynamic_month_subset_enabled,
                        keep_ratio=keep_ratio,
                        min_rows=min_rows,
                        kept_rows=keep_n,
                        total_rows=n_rows,
                    )
                else:
                    print("SUCCESS: Dynamic month-similarity subset skipped (class diversity guardrail).")
            else:
                print("SUCCESS: Dynamic month-similarity subset skipped (keep_n >= total rows).")

        if active_time_col:
            X_train_model = X_train_model.drop(columns=[active_time_col], errors="ignore")
            X_test_ref = X_test_ref.drop(columns=[active_time_col], errors="ignore")

        pre_encode_cols = X_train_model.columns.tolist()
        if partition_mode:
            X_train_model, X_test_ref, encoder, cat_cols = encode_features(
                X_train_model,
                X_test_ref,
                return_encoder=True,
            )
            final_model_cols = X_train_model.columns.tolist()
            print("SUCCESS: Encoded all text categories to numerical values (partition mode).")
        else:
            encoder = None
            cat_cols = X_train_model.select_dtypes(include=["object", "string", "category"]).columns.tolist()
            final_model_cols = X_train_model.columns.tolist()
            print("SUCCESS: In-memory mode using native categorical handling (no ordinal encode pass).")
        _debug(
            debug_enabled,
            debug_start,
            "Feature encoding complete.",
            cat_cols_count=len(cat_cols),
            cat_cols_preview=cat_cols[:20],
            final_model_cols_count=len(final_model_cols),
            final_model_shape=X_train_model.shape,
        )

        stable_prune_enabled = _env_truthy("ENABLE_STABLE_FEATURE_PRUNING", "1")
        if stable_prune_enabled:
            weighted_prune_enabled = _env_truthy("ENABLE_WEIGHTED_STABLE_PRUNING", "0")
            categorical_scoring_enabled = _env_truthy("ENABLE_CATEGORICAL_STABLE_SCORING", "0")
            keep_cols, prune_meta = select_stable_predictive_features(
                X_train=X_train_model,
                X_test=X_test_ref,
                y_train=y_train_model,
                exclude_cols=[],
                keep_ratio=float(os.getenv("STABLE_FEATURE_KEEP_RATIO", "0.80")),
                min_keep=int(os.getenv("STABLE_FEATURE_MIN_KEEP", "28")),
                max_keep=int(os.getenv("STABLE_FEATURE_MAX_KEEP", "180")),
                sample_weights=(temporal_weights if weighted_prune_enabled else None),
                temporal_consistency_blend=float(os.getenv("STABLE_FEATURE_TEMPORAL_BLEND", "0.35")),
                weighted_assoc_blend=float(os.getenv("STABLE_FEATURE_WEIGHTED_ASSOC_BLEND", "0.35")),
                enable_categorical_scoring=categorical_scoring_enabled,
            )
            if prune_meta.get("applied") and keep_cols:
                X_train_model = X_train_model[keep_cols].copy()
                X_test_ref = X_test_ref[keep_cols].copy()
                final_model_cols = keep_cols
                print(
                    "SUCCESS: Stable predictive feature pruning applied "
                    f"({prune_meta.get('selected')}/{prune_meta.get('total')} kept)."
                )
                _debug(
                    debug_enabled,
                    debug_start,
                    "Stable predictive feature pruning applied.",
                    prune_meta=prune_meta,
                )

        final_sample_weights = temporal_weights
        print("SUCCESS: Applied temporal similarity sample weights for training.")
        print(f"SUCCESS: Mitigation complete. Final training shape: {X_train_model.shape}")
        _phase_table(
            report,
            5,
            ["Mitigation Artifact", "Value"],
            [
                ["Final train rows", X_train_model.shape[0]],
                ["Final feature count", X_train_model.shape[1]],
                ["Pruned columns", len(pruned_cols)],
                ["Winsorized columns", len(winsor_plan)],
                ["Target-encoding plan size", len(te_plan.get("methods", {}))],
            ],
        )
        _phase_table(
            report,
            5,
            ["Mitigation Switch", "Value"],
            [
                ["Stability pruning enabled", bool(_env_truthy("ENABLE_STABILITY_GATED_PRUNING_PASS", "0"))],
                ["Rank normalization enabled", bool(rank_norm_enabled)],
                ["Category shift ratio enabled", bool(cat_shift_ratio_enabled)],
                ["Dynamic month subset enabled", bool(dynamic_month_subset_enabled)],
            ],
        )
        _phase_table(
            report,
            5,
            ["Temporal Weight Stats", "Value"],
            [
                ["Min", round(float(np.min(temporal_weights)), 6)],
                ["P50", round(float(np.quantile(temporal_weights, 0.50)), 6)],
                ["Max", round(float(np.max(temporal_weights)), 6)],
                ["Std", round(float(np.std(temporal_weights)), 6)],
            ],
        )

        mitigated_col_rows: list[list[Any]] = []
        for col in pruned_cols:
            mitigated_col_rows.append([
                "Stability Prune",
                col,
                str(drift_scores.get(col, {}).get("feature_type", "unknown")),
            ])
        for col in separability_dropped:
            mitigated_col_rows.append([
                "Separability Drop",
                col,
                str(drift_scores.get(col, {}).get("feature_type", "unknown")),
            ])
        for col in winsor_plan.keys():
            mitigated_col_rows.append([
                "Winsor Align",
                col,
                str(drift_scores.get(col, {}).get("feature_type", "numerical")),
            ])
        for col in rank_norm_cols:
            mitigated_col_rows.append([
                "Rank Normalize",
                col,
                str(drift_scores.get(col, {}).get("feature_type", "numerical")),
            ])
        for col in cat_shift_ratio_cols:
            base_col = str(col).replace("__shift_ratio", "")
            mitigated_col_rows.append([
                "Shift Ratio Feature",
                col,
                str(drift_scores.get(base_col, {}).get("feature_type", "derived")),
            ])
        if mitigated_col_rows:
            _phase_table(
                report,
                5,
                ["Action", "Column", "Type"],
                mitigated_col_rows[:20],
            )
        phase_start_perf = _finalize_phase_runtime(report, 5, phase_start_perf, total_start_perf)
        _phase5_done(report, on_phase)
        _debug(
            debug_enabled,
            debug_start,
            "Mitigation phase complete.",
            final_training_shape=X_train_model.shape,
            final_test_ref_shape=X_test_ref.shape,
            final_training_mem_gb=round(_df_mem_gb(X_train_model), 4),
            final_test_mem_gb=round(_df_mem_gb(X_test_ref), 4),
        )

        if partition_mode:
            print("SUCCESS: Partition mode skips redundant pre-mitigation scans to save runtime.")
        else:
            print("SUCCESS: Full-file pre-mitigation scans skipped (in-memory mode).")

        drift_end_time = time.time()
        drift_duration = drift_end_time - drift_start_time

        print_phase_banner(7, "Model Training")
        print("Training LightGBM with advanced adaptive mitigations...")

        gc.collect()

        # Final de-fragmentation boundary before handing off to model training.
        X_train_model = X_train_model.copy()
        X_test_ref = X_test_ref.copy()

        _predictions_ref, train_auprc, test_auprc, model, post_train_pr_curve, post_test_pr_curve = _run_mitigated_training_with_fallback(
            X_train_model=X_train_model,
            y_train_model=y_train_model,
            X_test_ref=X_test_ref,
            y_test_ref=y_test_ref,
            final_sample_weights=final_sample_weights,
            predict_batch_size=max(10000, int(runtime_plan.predict_batch_size)),
            cat_cols=cat_cols,
            debug_enabled=debug_enabled,
            debug_start=debug_start,
        )
        _debug(
            debug_enabled,
            debug_start,
            "Model training complete on reference split.",
            train_auprc=float(train_auprc),
            test_auprc=(None if test_auprc is None else float(test_auprc)),
            prediction_ref_min=float(np.min(_predictions_ref)) if len(_predictions_ref) else None,
            prediction_ref_p50=float(np.quantile(_predictions_ref, 0.50)) if len(_predictions_ref) else None,
            prediction_ref_max=float(np.max(_predictions_ref)) if len(_predictions_ref) else None,
        )

        total_end_time_prewrite = time.time()
        total_duration_prewrite = total_end_time_prewrite - total_start_time

        print("\nModel Performance Metrics")
        print(f"AU-PRC on training set before mitigation: {pre_train_auprc:.4f}")
        if pre_test_auprc is not None and not (isinstance(pre_test_auprc, float) and math.isnan(pre_test_auprc)):
            print(f"AU-PRC on test reference before mitigation: {pre_test_auprc:.4f}")
        else:
            print("AU-PRC on test reference before mitigation: N/A")
        print(f"AU-PRC on training set: {train_auprc:.4f}")
        if test_auprc is not None and not (isinstance(test_auprc, float) and math.isnan(test_auprc)):
            print(f"AU-PRC on test reference after mitigation: {test_auprc:.4f}")
        else:
            print("AU-PRC on test reference after mitigation: N/A")
        print(f"\nRuntime (in seconds): {total_duration_prewrite:.2f}")
        print(f"Time taken for drift detection and mitigation: {drift_duration:.2f}")
        _phase_table(
            report,
            6,
            ["Metric", "Value"],
            [
                ["Train AU-PRC (before)", round(float(pre_train_auprc), 6)],
                [
                    "Test AU-PRC (before)",
                    (None if pre_test_auprc is None else round(float(pre_test_auprc), 6)),
                ],
                ["Train AU-PRC (after)", round(float(train_auprc), 6)],
                [
                    "Test AU-PRC (after)",
                    (None if test_auprc is None else round(float(test_auprc), 6)),
                ],
            ],
        )
        _phase_table(
            report,
            6,
            ["Prediction Score Quantile", "Value"],
            [
                ["Min", round(float(np.min(_predictions_ref)), 6)],
                ["P25", round(float(np.quantile(_predictions_ref, 0.25)), 6)],
                ["P50", round(float(np.quantile(_predictions_ref, 0.50)), 6)],
                ["P75", round(float(np.quantile(_predictions_ref, 0.75)), 6)],
                ["Max", round(float(np.max(_predictions_ref)), 6)],
            ],
        )
        phase_start_perf = _finalize_phase_runtime(report, 6, phase_start_perf, total_start_perf)
        _phase6_done(report, on_phase)

        print_phase_banner(8, "Prediction Output")
        skip_prediction_write = _env_truthy("SKIP_PREDICTION_WRITE", "0")
        default_prediction_output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "prediction.csv",
        )
        prediction_output_path = str(os.getenv("PREDICTION_OUTPUT_PATH", default_prediction_output_path))
        print(
            f"Writing predictions to disk ({prediction_output_path})... "
            "This may take a moment for large datasets."
        )
        prediction_rows_written = 0

        if skip_prediction_write:
            print("WARNING: SKIP_PREDICTION_WRITE=1 active. Prediction CSV write skipped for benchmark mode.")
        else:
            if partition_mode:
                total_streamed = partition_predict_full_test(
                    test_path=args.test_data_filepath,
                    model=model,
                    missing_indicators=missing_indicators,
                    fill_plan=fill_plan,
                    structure_plan=structure_plan,
                    winsor_plan=winsor_plan,
                    te_plan=te_plan,
                    pre_encode_cols=pre_encode_cols,
                    encoder=encoder,
                    cat_cols=cat_cols,
                    final_model_cols=final_model_cols,
                    id_col_hint=id_column_name,
                    time_col_hint=time_column_name,
                    target_col_hint=target_column_name,
                    predict_batch_size=max(10000, int(runtime_plan.predict_batch_size)),
                    chunksize=max(50000, int(runtime_plan.stream_chunksize)),
                    output_path=prediction_output_path,
                )
                print(f"SUCCESS: Partition predictions written for {total_streamed} rows.")
                prediction_rows_written = int(total_streamed)
                _debug(
                    debug_enabled,
                    debug_start,
                    "Partition prediction stream complete.",
                    streamed_rows=total_streamed,
                    stream_chunksize=max(50000, int(runtime_plan.stream_chunksize)),
                    predict_batch_size=max(10000, int(runtime_plan.predict_batch_size)),
                )
            else:
                if test_ids_full is None:
                    test_ids_full = pd.Series(np.arange(len(_predictions_ref)), name="row_id")
                create_submission(
                    test_ids_full,
                    _predictions_ref,
                    output_path=prediction_output_path,
                )
                print(f"SUCCESS: In-memory predictions written for {len(_predictions_ref)} rows.")
                prediction_rows_written = int(len(_predictions_ref))
                _debug(
                    debug_enabled,
                    debug_start,
                    "In-memory submission created.",
                    prediction_rows=len(_predictions_ref),
                )

        print("Submission saved successfully!")

        total_end_time_with_write = time.time()
        total_duration_with_write = total_end_time_with_write - total_start_time
        print(f"Total runtime including prediction write (in seconds): {total_duration_with_write:.2f}")
        _phase_table(
            report,
            7,
            ["Prediction Output", "Value"],
            [
                ["Output path", prediction_output_path],
                ["Rows written", prediction_rows_written],
                ["Write skipped", bool(skip_prediction_write)],
                ["Partition mode", bool(partition_mode)],
                ["Predict batch size", int(runtime_plan.predict_batch_size)],
            ],
        )
        phase_start_perf = _finalize_phase_runtime(report, 7, phase_start_perf, total_start_perf)
        _phase7_done(report, on_phase)

        metrics = {
            "pre_train_auprc": float(pre_train_auprc),
            "pre_test_auprc": (
                float(pre_test_auprc)
                if pre_test_auprc is not None and not (isinstance(pre_test_auprc, float) and math.isnan(pre_test_auprc))
                else None
            ),
            "train_auprc": float(train_auprc),
            "test_auprc": (
                float(test_auprc)
                if test_auprc is not None and not (isinstance(test_auprc, float) and math.isnan(test_auprc))
                else None
            ),
            "pre_train_pr_curve": pre_train_pr_curve,
            "pre_test_pr_curve": pre_test_pr_curve,
            "post_train_pr_curve": post_train_pr_curve,
            "post_test_pr_curve": post_test_pr_curve,
            "runtime": float(total_duration_prewrite),
            "runtime_with_write": float(total_duration_with_write),
            "drift_runtime": float(drift_duration),
        }

        if (
            feature_gating_enabled
            and adv_feature_toggle_enabled
            and not in_revert_pass
            and should_revert_feature_engineering(
                previous_test_auprc,
                metrics.get("test_auprc"),
                max_drop=0.01,
            )
        ):
            print("Feature Engineering Reverted: Performance Degradation Detected")
            _debug(
                debug_enabled,
                debug_start,
                "Performance gating triggered revert pass.",
                previous_test_auprc=previous_test_auprc,
                current_test_auprc=metrics.get("test_auprc"),
            )

            rerun_env = os.environ.copy()
            rerun_env["ENABLE_ADV_DYNAMIC_FEATURES"] = "0"
            rerun_env["FEATURE_REVERT_ACTIVE"] = "1"

            rerun_cmd = [
                sys.executable,
                os.path.abspath(__file__),
                "--train_data_filepath",
                args.train_data_filepath,
                "--test_data_filepath",
                args.test_data_filepath,
            ]
            rerun = subprocess.run(rerun_cmd, env=rerun_env, check=False)
            if rerun.returncode != 0:
                raise RuntimeError("Automatic baseline rerun failed after feature gating trigger.")
            report["success"] = True
            report["metrics"] = metrics
            _phase_line(report, 8, "Feature gating triggered fallback rerun.")
            phase_start_perf = _finalize_phase_runtime(report, 8, phase_start_perf, total_start_perf)
            _phase8_done(report, on_phase)
            return report

        with open("latest_metrics.json", "w") as f:
            json.dump(metrics, f)

        compact_metrics = {
            "pre_train_auprc": metrics.get("pre_train_auprc"),
            "pre_test_auprc": metrics.get("pre_test_auprc"),
            "train_auprc": metrics.get("train_auprc"),
            "test_auprc": metrics.get("test_auprc"),
            "runtime": metrics.get("runtime"),
            "runtime_with_write": metrics.get("runtime_with_write"),
            "drift_runtime": metrics.get("drift_runtime"),
        }
        with open("latest_metrics_summary.json", "w") as f:
            json.dump(compact_metrics, f, indent=2)

        _export_dashboard_artifacts(
            metrics=metrics,
            drift_report=drift_report,
            drift_scores=drift_scores,
            dq_report=dq_report,
            winsor_plan=winsor_plan,
            train_frame=X_train_model,
            test_frame=X_test_ref,
            sample_rows=5000,
        )
        _debug(
            debug_enabled,
            debug_start,
            "Pipeline completed successfully.",
            metrics=metrics,
        )
        print_phase_banner(9, "Pipeline Complete")
        report["success"] = True
        report["metrics"] = metrics
        _phase_line(report, 8, "Pipeline completed successfully.")

        phase_start_perf = _finalize_phase_runtime(report, 8, phase_start_perf, total_start_perf)
        prediction_output_runtime = max(0.0, float(total_duration_with_write) - float(total_duration_prewrite))

        runtime_phase_rows = report.get("runtime_phase_rows", [])
        if runtime_phase_rows:
            _phase_table(
                report,
                8,
                ["Phase Runtime (s)", "Value"],
                runtime_phase_rows,
            )

        _phase_table(
            report,
            8,
            ["Runtime Diagnostic", "Value"],
            [
                ["Available RAM at start (GB)", round(float(runtime_plan.available_ram_gb), 2)],
                ["Pipeline before prediction output (s)", round(float(total_duration_prewrite), 2)],
                ["Prediction output runtime (s)", round(float(prediction_output_runtime), 2)],
                ["Pipeline including prediction output (s)", round(float(total_duration_with_write), 2)],
                ["Drift/Mitigation runtime (s)", round(float(drift_duration), 2)],
            ],
        )
        _phase8_done(report, on_phase)
        return report

    except Exception as e:
        print(f"FAILED: An error occurred during testing: {e}")
        _debug(
            debug_enabled,
            debug_start,
            "Pipeline failed with exception.",
            error=str(e),
            traceback=traceback.format_exc(),
        )
        report["success"] = False
        report["error"] = str(e)
        _phase_line(report, 8, f"Failed: {e}")
        phase_start_perf = _finalize_phase_runtime(report, 8, phase_start_perf, total_start_perf)
        _phase8_done(report, on_phase)
        return report
def run_pipeline(
    train_data_filepath: str,
    test_data_filepath: str,
    quiet: bool = False,
    on_phase: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the full end-to-end pipeline using explicit file paths."""
    args = argparse.Namespace(
        train_data_filepath=train_data_filepath,
        test_data_filepath=test_data_filepath,
    )
    if quiet:
        with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink):
            return _run_pipeline_with_args(args, on_phase=on_phase)
    return _run_pipeline_with_args(args, on_phase=on_phase)


def main() -> None:
    parser = argparse.ArgumentParser(description="NAISC Singtel 2026 Pipeline")
    parser.add_argument("--train_data_filepath", type=str, required=True, help="Path to train.csv")
    parser.add_argument("--test_data_filepath", type=str, required=True, help="Path to test.csv")
    args = parser.parse_args()
    _run_pipeline_with_args(args)


if __name__ == "__main__":
    main()
