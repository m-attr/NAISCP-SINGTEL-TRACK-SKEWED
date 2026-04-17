import ctypes
import os
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class RuntimePlan:
    reserve_free_gb: float
    total_ram_gb: float
    available_ram_gb: float
    protected_free_gb: float
    working_budget_gb: float
    detection_reference_rows: int
    train_reference_rows: int
    detection_chunk_rows: int
    stream_chunksize: int
    predict_batch_size: int
    detection_workers: int
    extra_safety_gb: float
    effective_budget_gb: float
    detection_cap_rows: int
    model_cap_rows: int
    baseline_train_cap_rows: int
    baseline_test_cap_rows: int
    train_row_bytes: float
    test_row_bytes: float
    notes: list[str] = field(default_factory=list)


def _bytes_to_gb(value: int) -> float:
    return float(value) / (1024 ** 3)


def _get_memory_bytes():
    """
    Returns (total_bytes, available_bytes). Uses psutil when available,
    then a Windows fallback, then a conservative default.
    """
    try:
        import psutil  # type: ignore

        mem = psutil.virtual_memory()
        return int(mem.total), int(mem.available)
    except Exception:
        pass

    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
            return int(stat.ullTotalPhys), int(stat.ullAvailPhys)
    except Exception:
        pass

    total = 16 * (1024 ** 3)
    available = 8 * (1024 ** 3)
    return total, available


def _clamp(value: int, low: int, high: int) -> int:
    return int(max(low, min(high, value)))


def _estimate_row_bytes(csv_path: str, sample_rows: int = 25000) -> float:
    """
    Estimate in-memory bytes per row from a small sample.
    """
    try:
        sample = pd.read_csv(csv_path, nrows=sample_rows, low_memory=True)
        if len(sample) == 0:
            return 2048.0
        mem = sample.memory_usage(deep=True).sum()
        return float(mem) / float(len(sample))
    except Exception:
        return 2048.0


def _rows_from_budget(
    budget_bytes: int,
    row_bytes: float,
    overhead_multiplier: float,
    low: int,
    high: int,
) -> int:
    raw = int(float(budget_bytes) / max(1.0, float(row_bytes) * float(overhead_multiplier)))
    return _clamp(raw, low, high)


def build_runtime_plan(
    train_path: str,
    test_path: str,
    reserve_free_gb: float = 1.5,
) -> RuntimePlan:
    """
    Build a machine-aware runtime plan with a fixed RAM reserve.
    """
    total_b, avail_b = _get_memory_bytes()
    protected_free_b = int(reserve_free_gb * (1024 ** 3))
    working_budget_b = max(1024 ** 3, avail_b - protected_free_b)

    notes: list[str] = []
    if avail_b <= protected_free_b:
        notes.append("Available RAM is near reserve threshold; chunk sizes were tightened.")

    train_row_bytes = _estimate_row_bytes(train_path)
    test_row_bytes = _estimate_row_bytes(test_path)
    avg_row_bytes = max(1.0, (train_row_bytes + test_row_bytes) / 2.0)

    # Keep additional free memory headroom for process spikes and OS/background usage.
    # effective_budget = (available - reserve_free) - extra_safety
    safety_fraction = float(os.getenv("CAP_MEMORY_SAFETY_FRACTION", "0.40"))
    safety_fraction = min(0.80, max(0.10, safety_fraction))
    min_extra_safety_gb = float(os.getenv("CAP_EXTRA_SAFETY_GB", "1.0"))
    min_extra_safety_b = int(max(0.0, min_extra_safety_gb) * (1024 ** 3))
    extra_safety_b = max(min_extra_safety_b, int(working_budget_b * safety_fraction))
    extra_safety_b = min(extra_safety_b, max(0, working_budget_b - (512 * 1024 * 1024)))
    effective_budget_b = max(512 * 1024 * 1024, working_budget_b - extra_safety_b)

    # Dynamic cap formula (memory-aware, conservative by design):
    # rows = stage_budget_bytes / (row_bytes * overhead_multiplier)
    # where stage budgets are fixed percentages of effective budget.
    detection_cap_rows = _rows_from_budget(
        budget_bytes=int(effective_budget_b * 0.25),
        row_bytes=max(train_row_bytes, test_row_bytes),
        overhead_multiplier=3.0,
        low=40_000,
        high=250_000,
    )
    model_cap_rows = _rows_from_budget(
        budget_bytes=int(effective_budget_b * 0.55),
        row_bytes=train_row_bytes,
        overhead_multiplier=4.0,
        low=80_000,
        high=900_000,
    )
    baseline_train_cap_rows = min(
        model_cap_rows,
        _rows_from_budget(
            budget_bytes=int(effective_budget_b * 0.20),
            row_bytes=train_row_bytes,
            overhead_multiplier=3.0,
            low=40_000,
            high=400_000,
        ),
    )
    baseline_test_cap_rows = min(
        detection_cap_rows,
        _rows_from_budget(
            budget_bytes=int(effective_budget_b * 0.10),
            row_bytes=test_row_bytes,
            overhead_multiplier=2.5,
            low=30_000,
            high=200_000,
        ),
    )

    notes.append(
        "Dynamic row caps computed from effective memory budget "
        f"(safety_fraction={safety_fraction:.2f}, extra_safety_gb={_bytes_to_gb(extra_safety_b):.2f})."
    )

    # Chunk sizes are based on current free budget after reserving 1.5 GB.
    detection_chunk_rows = _clamp(
        int((working_budget_b * 0.15) / avg_row_bytes),
        25_000,
        300_000,
    )
    stream_chunksize = _clamp(
        int((working_budget_b * 0.20) / max(1.0, test_row_bytes)),
        25_000,
        2_500_000,
    )
    predict_batch_size = _clamp(
        stream_chunksize,
        20_000,
        300_000,
    )
    detection_reference_rows = _clamp(
        int((working_budget_b * 0.25) / max(1.0, test_row_bytes)),
        200_000,
        1_200_000,
    )
    train_reference_rows = _clamp(
        int((working_budget_b * 0.35) / max(1.0, train_row_bytes)),
        250_000,
        1_500_000,
    )

    detection_workers = max(1, min(2, os.cpu_count() or 1))
    if detection_workers == 2:
        notes.append("Detection/mitigation can use 2 CPU workers where supported.")
    else:
        notes.append("Detection/mitigation will use a single CPU worker.")

    return RuntimePlan(
        reserve_free_gb=reserve_free_gb,
        total_ram_gb=_bytes_to_gb(total_b),
        available_ram_gb=_bytes_to_gb(avail_b),
        protected_free_gb=_bytes_to_gb(protected_free_b),
        working_budget_gb=_bytes_to_gb(working_budget_b),
        detection_reference_rows=detection_reference_rows,
        train_reference_rows=train_reference_rows,
        detection_chunk_rows=detection_chunk_rows,
        stream_chunksize=stream_chunksize,
        predict_batch_size=predict_batch_size,
        detection_workers=detection_workers,
        extra_safety_gb=_bytes_to_gb(extra_safety_b),
        effective_budget_gb=_bytes_to_gb(effective_budget_b),
        detection_cap_rows=detection_cap_rows,
        model_cap_rows=model_cap_rows,
        baseline_train_cap_rows=baseline_train_cap_rows,
        baseline_test_cap_rows=baseline_test_cap_rows,
        train_row_bytes=float(train_row_bytes),
        test_row_bytes=float(test_row_bytes),
        notes=notes,
    )
