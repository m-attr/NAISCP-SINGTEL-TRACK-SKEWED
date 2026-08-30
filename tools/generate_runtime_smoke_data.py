"""Generate anonymous public-shaped CSVs in chunks for runtime-only smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MONTHS = np.array(
    [
        "25-Jan",
        "25-Feb",
        "25-Mar",
        "25-Apr",
        "25-May",
        "25-Jun",
        "25-Jul",
        "25-Aug",
    ]
)


def _chunk(
    *,
    start: int,
    rows: int,
    rng: np.random.Generator,
    feature_count: int,
    training: bool,
) -> pd.DataFrame:
    latent = rng.normal(0.0, 1.0, rows)
    data: dict[str, object] = {
        "CustomerID": np.arange(start, start + rows, dtype=np.int64),
        "Month": MONTHS[np.arange(start, start + rows) % len(MONTHS)]
        if training
        else np.repeat("25-Sep", rows),
    }
    if training:
        target = latent + rng.normal(0.0, 0.8, rows) > 0.0
        data["ChurnStatus"] = np.where(target, "Yes", "No")
    numeric_count = max(1, feature_count - 4)
    for index in range(numeric_count):
        data[f"anonymous_numeric_{index:03d}"] = (
            latent * (1.0 if index < 6 else 0.15)
            + rng.normal(0.0, 0.7 + 0.03 * index, rows)
        )
    for index in range(feature_count - numeric_count):
        source = latent + rng.normal(0.0, 0.5 + 0.1 * index, rows)
        data[f"anonymous_category_{index:03d}"] = np.where(
            source < -0.7,
            "group_a",
            np.where(source > 0.9, "group_c", "group_b"),
        )
    return pd.DataFrame(data)


def _write(
    path: Path,
    total_rows: int,
    chunk_rows: int,
    feature_count: int,
    seed: int,
    training: bool,
) -> None:
    rng = np.random.default_rng(seed)
    first = True
    for start in range(0, total_rows, chunk_rows):
        rows = min(chunk_rows, total_rows - start)
        frame = _chunk(
            start=start,
            rows=rows,
            rng=rng,
            feature_count=feature_count,
            training=training,
        )
        frame.to_csv(path, mode="w" if first else "a", header=first, index=False)
        first = False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_directory", required=True)
    parser.add_argument("--train_rows", type=int, default=750_000)
    parser.add_argument("--test_rows", type=int, default=250_000)
    parser.add_argument("--chunk_rows", type=int, default=50_000)
    parser.add_argument("--feature_count", type=int, default=20)
    args = parser.parse_args()
    output = Path(args.output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write(
        output / "train.csv",
        args.train_rows,
        args.chunk_rows,
        args.feature_count,
        seed=7_001,
        training=True,
    )
    _write(
        output / "test.csv",
        args.test_rows,
        args.chunk_rows,
        args.feature_count,
        seed=7_002,
        training=False,
    )
    print(
        {
            "train_rows": args.train_rows,
            "test_rows": args.test_rows,
            "total_rows": args.train_rows + args.test_rows,
            "feature_count": args.feature_count,
            "output_directory": str(output),
        }
    )


if __name__ == "__main__":
    main()
