from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.orchestrator import run_pipeline


REPO_ROOT = Path(__file__).resolve().parents[1]


def _assert_valid_prediction(path: Path, expected_test: pd.DataFrame) -> pd.DataFrame:
    prediction = pd.read_csv(path)
    assert prediction.columns.tolist() == ["CustomerID", "probability_score"]
    assert len(prediction) == len(expected_test)
    assert prediction["CustomerID"].equals(expected_test["CustomerID"])
    assert not prediction["CustomerID"].isna().any()
    assert not prediction["CustomerID"].duplicated().any()
    scores = pd.to_numeric(prediction["probability_score"], errors="coerce")
    assert scores.notna().all()
    assert np.isfinite(scores.to_numpy()).all()
    assert scores.between(0.0, 1.0, inclusive="both").all()
    return prediction


def test_public_feature_target_isolation_and_output_contract(
    bounded_public_data: tuple[Path, Path, pd.DataFrame],
    tmp_path: Path,
    monkeypatch,
) -> None:
    train_path, _, original_test = bounded_public_data
    variants = {
        "original": original_test.copy(),
        "all_no": original_test.assign(ChurnStatus="No"),
        "all_yes": original_test.assign(ChurnStatus="Yes"),
        "shuffled": original_test.assign(
            ChurnStatus=original_test["ChurnStatus"].sample(frac=1.0, random_state=4400).to_numpy()
        ),
        "absent": original_test.drop(columns=["ChurnStatus"]),
    }

    reference_scores: np.ndarray | None = None
    for name, test_variant in variants.items():
        run_dir = tmp_path / name
        run_dir.mkdir()
        test_path = run_dir / "test.csv"
        output_path = run_dir / "prediction.csv"
        test_variant.to_csv(test_path, index=False)

        monkeypatch.chdir(run_dir)
        monkeypatch.setenv("PREDICTION_OUTPUT_PATH", str(output_path))
        monkeypatch.setenv("PIPELINE_DEBUG", "0")
        report = run_pipeline(str(train_path), str(test_path), quiet=True)

        assert report["success"], report.get("error")
        assert report["metrics"]["model_fit_count"] == 1
        assert "test_auprc" not in report["metrics"]
        metric_file = json.loads((run_dir / "latest_metrics.json").read_text(encoding="utf-8"))
        assert "test_auprc" not in metric_file

        prediction = _assert_valid_prediction(output_path, original_test)
        scores = prediction["probability_score"].to_numpy(dtype=np.float64)
        if reference_scores is None:
            reference_scores = scores
        else:
            np.testing.assert_allclose(scores, reference_scores, rtol=0.0, atol=1e-12)


def test_competition_cli_smoke(
    bounded_public_data: tuple[Path, Path, pd.DataFrame],
    tmp_path: Path,
) -> None:
    train_path, test_path, expected_test = bounded_public_data
    output_path = tmp_path / "prediction.csv"
    env = os.environ.copy()
    env["PREDICTION_OUTPUT_PATH"] = str(output_path)
    env["PIPELINE_DEBUG"] = "0"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "src" / "main.py"),
            "--train_data_filepath",
            str(train_path),
            "--test_data_filepath",
            str(test_path),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Pipeline Error" not in result.stdout
    _assert_valid_prediction(output_path, expected_test)
    metrics = json.loads((tmp_path / "latest_metrics.json").read_text(encoding="utf-8"))
    assert metrics["model_fit_count"] == 1
    assert "test_auprc" not in metrics
