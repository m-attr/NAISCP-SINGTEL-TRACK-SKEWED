from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common.contracts import ID_COLUMN, TARGET_COLUMN, TIME_COLUMN
from pipeline.orchestrator import run_pipeline


def test_public_features_remain_general_when_renamed(
    public_data_paths: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch,
) -> None:
    train_path, test_path = public_data_paths
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    feature_names = [
        column
        for column in train.columns
        if column not in {ID_COLUMN, TIME_COLUMN, TARGET_COLUMN}
        and column in test.columns
    ]
    rename_map = {
        feature: f"unknown_feature_{index:03d}"
        for index, feature in enumerate(feature_names, start=1)
    }
    anonymous_train_path = tmp_path / "anonymous_train.csv"
    anonymous_test_path = tmp_path / "anonymous_test.csv"
    train.rename(columns=rename_map).to_csv(anonymous_train_path, index=False)
    test.rename(columns=rename_map).to_csv(anonymous_test_path, index=False)

    predictions: dict[str, pd.DataFrame] = {}
    reports: dict[str, dict] = {}
    for name, current_train, current_test in (
        ("named", train_path, test_path),
        ("anonymous", anonymous_train_path, anonymous_test_path),
    ):
        run_dir = tmp_path / name
        run_dir.mkdir()
        output_path = run_dir / "prediction.csv"
        monkeypatch.chdir(run_dir)
        monkeypatch.setenv("PREDICTION_OUTPUT_PATH", str(output_path))
        report = run_pipeline(str(current_train), str(current_test), quiet=True)
        assert report["success"], report.get("error")
        reports[name] = report
        predictions[name] = pd.read_csv(output_path)

    assert reports["named"]["metrics"]["model_fit_count"] == 1
    assert reports["anonymous"]["metrics"]["model_fit_count"] == 1
    assert {
        rename_map.get(feature, feature)
        for feature in reports["named"]["metrics"]["dropped_features"]
    } == set(reports["anonymous"]["metrics"]["dropped_features"])
    assert {
        rename_map.get(feature, feature)
        for feature in reports["named"]["metrics"]["reversed_features"]
    } == set(reports["anonymous"]["metrics"]["reversed_features"])
    pd.testing.assert_series_equal(
        predictions["named"][ID_COLUMN],
        predictions["anonymous"][ID_COLUMN],
    )
    np.testing.assert_array_equal(
        predictions["named"]["probability_score"].to_numpy(dtype=np.float64),
        predictions["anonymous"]["probability_score"].to_numpy(dtype=np.float64),
    )


def test_production_source_has_no_public_feature_names(public_data_paths) -> None:
    train_path, _ = public_data_paths
    train_columns = pd.read_csv(train_path, nrows=0).columns.tolist()
    public_features = [
        column
        for column in train_columns
        if column not in {ID_COLUMN, TIME_COLUMN, TARGET_COLUMN}
    ]
    source_root = Path(__file__).resolve().parents[1] / "src"
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(source_root.rglob("*.py"))
    )
    assert not [feature for feature in public_features if feature in source]
