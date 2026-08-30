from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = REPO_ROOT / "tools" / "evaluate_public_predictions.py"


def _load_evaluator_module():
    spec = importlib.util.spec_from_file_location("external_public_evaluator", EVALUATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_external_evaluator_validates_completed_prediction(tmp_path: Path) -> None:
    labelled_test = pd.DataFrame(
        {
            "CustomerID": [1001, 1002, 1003, 1004],
            "ChurnStatus": ["No", "Yes", "No", "Yes"],
        }
    )
    prediction = pd.DataFrame(
        {
            "CustomerID": [1001, 1002, 1003, 1004],
            "probability_score": [0.1, 0.9, 0.2, 0.8],
        }
    )
    test_path = tmp_path / "labelled_test.csv"
    prediction_path = tmp_path / "prediction.csv"
    labelled_test.to_csv(test_path, index=False)
    prediction.to_csv(prediction_path, index=False)

    result = _load_evaluator_module().evaluate_predictions(prediction_path, test_path)

    assert result["public_test_auprc"] == 1.0
    assert result["row_count"] == 4
    assert result["id_order_exact"] is True
    assert result["probabilities_valid"] is True
    assert "NOT PART OF COMPETITION PIPELINE" in result["evaluator"]
