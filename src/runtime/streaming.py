from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import time
from typing import Any

import numpy as np
import pandas as pd

from model.lightgbm_model import predict_positive_class
from preprocessing.plan import (
    ID_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
    PreparationPlan,
    transform_test,
)

_SAMPLE_SEED = np.uint64(0x9E3779B97F4A7C15)
_DEFAULT_READ_CHUNK_BYTES = 64 * 1024 * 1024
_MODEL_SAMPLE_BYTE_BUDGET = 512 * 1024 * 1024
_ANALYSIS_SAMPLE_BYTE_BUDGET = 256 * 1024 * 1024
_MODEL_SAMPLE_MAX_ROWS = 200_000
_ANALYSIS_SAMPLE_MAX_ROWS = 75_000
_MIN_SAMPLE_ROWS = 20_000
_MIN_CHUNK_ROWS = 2_000
_MAX_CHUNK_ROWS = 100_000


@dataclass(frozen=True)
class SampleCollection:
    model_frame: pd.DataFrame | None
    analysis_frame: pd.DataFrame
    total_rows: int
    estimated_bytes_per_row: int
    read_chunk_rows: int
    model_sample_limit: int
    analysis_sample_limit: int
    elapsed_seconds: float


def select_test_feature_columns(path: Path) -> list[str]:
    """Exclude ChurnStatus by name before any test values are read."""
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    return [column for column in columns if column != TARGET_COLUMN]


def _estimate_bytes_per_row(frame: pd.DataFrame) -> int:
    if len(frame) == 0:
        return 1
    return max(1, int(np.ceil(frame.memory_usage(index=True, deep=True).sum() / len(frame))))


def _rows_for_bytes(bytes_per_row: int, byte_budget: int, maximum: int) -> int:
    # Never exceed the declared memory budget merely to satisfy a preferred sample size.
    affordable = max(1, int(byte_budget // max(1, bytes_per_row)))
    return min(maximum, affordable)


def _chunk_rows(bytes_per_row: int) -> int:
    return max(
        _MIN_CHUNK_ROWS,
        min(_MAX_CHUNK_ROWS, int(_DEFAULT_READ_CHUNK_BYTES // max(1, bytes_per_row))),
    )


def _priority_values(frame: pd.DataFrame, row_offset: int) -> np.ndarray:
    if ID_COLUMN in frame.columns:
        base = pd.util.hash_pandas_object(
            frame[ID_COLUMN].astype("string"), index=False
        ).to_numpy(dtype=np.uint64)
    else:
        base = np.arange(row_offset, row_offset + len(frame), dtype=np.uint64)
    if TIME_COLUMN in frame.columns:
        base ^= pd.util.hash_pandas_object(
            frame[TIME_COLUMN].astype("string"), index=False
        ).to_numpy(dtype=np.uint64)
    order = np.arange(row_offset, row_offset + len(frame), dtype=np.uint64)
    # A deterministic 64-bit mix; no Python hash randomization is involved.
    mixed = base ^ (order + _SAMPLE_SEED + (base << np.uint64(6)) + (base >> np.uint64(2)))
    return mixed


def _retain_smallest_priority(
    current: pd.DataFrame | None,
    incoming: pd.DataFrame,
    limit: int,
) -> pd.DataFrame:
    if current is None or current.empty:
        combined = incoming
    else:
        combined = pd.concat([current, incoming], ignore_index=True, copy=False)
    if len(combined) <= limit:
        return combined
    priorities = combined["__sample_priority"].to_numpy(dtype=np.uint64, copy=False)
    selected = np.argpartition(priorities, limit - 1)[:limit]
    return combined.iloc[selected].copy()


def _finalize_sample(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    if len(frame) > limit:
        priorities = frame["__sample_priority"].to_numpy(dtype=np.uint64, copy=False)
        selected = np.argpartition(priorities, limit - 1)[:limit]
        frame = frame.iloc[selected].copy()
    frame.sort_values("__row_order", inplace=True, kind="stable")
    return frame.drop(columns=["__sample_priority", "__row_order"]).reset_index(drop=True)


def collect_training_samples(path: str | Path) -> SampleCollection:
    """Scan train.csv once and retain bounded deterministic model/analysis samples."""
    started = time.perf_counter()
    source = Path(path)
    preview = pd.read_csv(source, nrows=4096)
    if TARGET_COLUMN not in preview.columns:
        raise ValueError(f"Training data must contain {TARGET_COLUMN}.")
    bytes_per_row = _estimate_bytes_per_row(preview)
    model_limit = _rows_for_bytes(
        bytes_per_row, _MODEL_SAMPLE_BYTE_BUDGET, _MODEL_SAMPLE_MAX_ROWS
    )
    analysis_limit = min(
        model_limit,
        _rows_for_bytes(
            bytes_per_row, _ANALYSIS_SAMPLE_BYTE_BUDGET, _ANALYSIS_SAMPLE_MAX_ROWS
        ),
    )
    chunk_rows = _chunk_rows(bytes_per_row)

    retained: pd.DataFrame | None = None
    total_rows = 0
    for chunk in pd.read_csv(source, chunksize=chunk_rows):
        chunk = chunk.copy()
        chunk["__row_order"] = np.arange(
            total_rows, total_rows + len(chunk), dtype=np.int64
        )
        chunk["__sample_priority"] = _priority_values(chunk, total_rows)
        total_rows += len(chunk)
        retained = _retain_smallest_priority(retained, chunk, model_limit)

    if retained is None:
        raise ValueError("Training CSV contains no rows.")
    model_frame = _finalize_sample(retained, model_limit)

    # The analysis sample is a deterministic nested subset of the model sample.
    if len(model_frame) <= analysis_limit:
        analysis_frame = model_frame.copy()
    else:
        temp = model_frame.copy()
        temp["__row_order"] = np.arange(len(temp), dtype=np.int64)
        temp["__sample_priority"] = _priority_values(temp, 0)
        analysis_frame = _finalize_sample(temp, analysis_limit)

    return SampleCollection(
        model_frame=model_frame,
        analysis_frame=analysis_frame,
        total_rows=total_rows,
        estimated_bytes_per_row=bytes_per_row,
        read_chunk_rows=chunk_rows,
        model_sample_limit=model_limit,
        analysis_sample_limit=analysis_limit,
        elapsed_seconds=time.perf_counter() - started,
    )


def collect_test_analysis_sample(path: str | Path) -> SampleCollection:
    """Scan test.csv once, never reading ChurnStatus, and retain a bounded sample."""
    started = time.perf_counter()
    source = Path(path)
    usecols = select_test_feature_columns(source)
    preview = pd.read_csv(source, usecols=usecols, nrows=4096)
    bytes_per_row = _estimate_bytes_per_row(preview)
    analysis_limit = _rows_for_bytes(
        bytes_per_row, _ANALYSIS_SAMPLE_BYTE_BUDGET, _ANALYSIS_SAMPLE_MAX_ROWS
    )
    chunk_rows = _chunk_rows(bytes_per_row)

    retained: pd.DataFrame | None = None
    total_rows = 0
    for chunk in pd.read_csv(source, usecols=usecols, chunksize=chunk_rows):
        chunk = chunk.copy()
        chunk["__row_order"] = np.arange(
            total_rows, total_rows + len(chunk), dtype=np.int64
        )
        chunk["__sample_priority"] = _priority_values(chunk, total_rows)
        total_rows += len(chunk)
        retained = _retain_smallest_priority(retained, chunk, analysis_limit)

    if retained is None:
        raise ValueError("Test CSV contains no rows.")
    analysis_frame = _finalize_sample(retained, analysis_limit)
    return SampleCollection(
        model_frame=None,
        analysis_frame=analysis_frame,
        total_rows=total_rows,
        estimated_bytes_per_row=bytes_per_row,
        read_chunk_rows=chunk_rows,
        model_sample_limit=0,
        analysis_sample_limit=analysis_limit,
        elapsed_seconds=time.perf_counter() - started,
    )


def stream_predictions_to_csv(
    *,
    model: Any,
    test_path: str | Path,
    plan: PreparationPlan,
    output_path: str | Path,
    chunk_rows: int,
) -> dict[str, Any]:
    """Transform, predict, and append each test chunk without retaining prior chunks."""
    source = Path(test_path)
    destination = Path(output_path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    usecols = select_test_feature_columns(source)

    if temporary.exists():
        temporary.unlink()
    first = True
    rows_written = 0
    maximum_chunk_rows = 0
    maximum_transformed_bytes = 0
    started = time.perf_counter()

    try:
        for chunk in pd.read_csv(source, usecols=usecols, chunksize=max(1, int(chunk_rows))):
            if ID_COLUMN not in chunk.columns:
                raise ValueError(f"Test data must contain {ID_COLUMN}.")
            identifiers = chunk[ID_COLUMN].to_numpy(copy=True)
            transformed = transform_test(chunk, plan)
            array = np.ascontiguousarray(
                transformed.to_numpy(dtype=np.float32, copy=False)
            )
            probabilities = predict_positive_class(model, array)
            if len(probabilities) != len(chunk):
                raise RuntimeError("Prediction row count does not match the current test chunk.")
            if not np.isfinite(probabilities).all():
                raise RuntimeError("Prediction output contains missing or non-finite values.")
            if ((probabilities < 0.0) | (probabilities > 1.0)).any():
                raise RuntimeError("Prediction output contains values outside [0, 1].")

            output = pd.DataFrame(
                {ID_COLUMN: identifiers, "probability_score": probabilities}
            )
            output.to_csv(temporary, mode="w" if first else "a", header=first, index=False)
            first = False
            rows_written += len(output)
            maximum_chunk_rows = max(maximum_chunk_rows, len(output))
            maximum_transformed_bytes = max(
                maximum_transformed_bytes,
                int(transformed.memory_usage(index=True, deep=True).sum()),
            )
            del output, probabilities, array, transformed, identifiers, chunk

        if rows_written == 0:
            raise ValueError("Test CSV contains no rows.")
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    return {
        "rows_written": rows_written,
        "maximum_chunk_rows": maximum_chunk_rows,
        "maximum_transformed_chunk_bytes": maximum_transformed_bytes,
        "elapsed_seconds": time.perf_counter() - started,
        "output_path": str(destination.resolve()),
    }
