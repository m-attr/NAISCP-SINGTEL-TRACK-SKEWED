"""DEVELOPMENT / EXTERNAL EVALUATOR — NOT PART OF COMPETITION PIPELINE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


ID_COLUMN = "CustomerID"
TARGET_COLUMN = "ChurnStatus"
SCORE_COLUMN = "probability_score"
OUTPUT_COLUMNS = [ID_COLUMN, SCORE_COLUMN]


def evaluate_predictions(prediction_path: str | Path, labelled_test_path: str | Path) -> dict[str, object]:
    """Validate a completed prediction file, then calculate public-test AU-PRC."""
    predictions = pd.read_csv(prediction_path)
    labelled_test = pd.read_csv(labelled_test_path, usecols=[ID_COLUMN, TARGET_COLUMN])

    if predictions.columns.tolist() != OUTPUT_COLUMNS:
        raise ValueError(f"Prediction columns must be exactly {OUTPUT_COLUMNS}; got {predictions.columns.tolist()}.")
    if len(predictions) != len(labelled_test):
        raise ValueError(f"Row-count mismatch: prediction={len(predictions)}, labelled_test={len(labelled_test)}.")
    if not predictions[ID_COLUMN].equals(labelled_test[ID_COLUMN]):
        raise ValueError("Prediction CustomerID values do not exactly match public-test order.")
    if predictions[ID_COLUMN].isna().any() or predictions[ID_COLUMN].duplicated().any():
        raise ValueError("Prediction CustomerID values contain missing values or duplicates.")

    scores = pd.to_numeric(predictions[SCORE_COLUMN], errors="coerce")
    if scores.isna().any() or not np.isfinite(scores.to_numpy(dtype=np.float64)).all():
        raise ValueError("Prediction probabilities must be finite numeric values.")
    if not scores.between(0.0, 1.0, inclusive="both").all():
        raise ValueError("Prediction probabilities must all be within [0, 1].")

    labels = labelled_test[TARGET_COLUMN].map(
        {"Yes": 1, "No": 0, "yes": 1, "no": 0, "Y": 1, "N": 0, "y": 1, "n": 0}
    )
    if labels.isna().any():
        numeric_labels = pd.to_numeric(labelled_test[TARGET_COLUMN], errors="coerce")
        labels = labels.fillna(numeric_labels)
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("Public test target must be binary Yes/No or 0/1 values.")

    return {
        "evaluator": "DEVELOPMENT / EXTERNAL — NOT PART OF COMPETITION PIPELINE",
        "prediction_path": str(Path(prediction_path).resolve()),
        "labelled_test_path": str(Path(labelled_test_path).resolve()),
        "row_count": int(len(predictions)),
        "id_order_exact": True,
        "probabilities_valid": True,
        "public_test_auprc": float(average_precision_score(labels.astype(np.int8), scores)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="External public-label evaluator (development only).")
    parser.add_argument("--prediction_filepath", required=True)
    parser.add_argument("--labelled_test_filepath", required=True)
    parser.add_argument(
        "--output_filepath",
        help="Optional JSON evidence path written only by this external evaluator.",
    )
    args = parser.parse_args()

    result = evaluate_predictions(
        args.prediction_filepath,
        args.labelled_test_filepath,
    )
    if args.output_filepath:
        output_path = Path(args.output_filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print("DEVELOPMENT / EXTERNAL EVALUATOR — NOT PART OF COMPETITION PIPELINE")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
