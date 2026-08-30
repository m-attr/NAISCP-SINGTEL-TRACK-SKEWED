from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from model.lightgbm_model import (
    fit_once,
    get_production_model_fit_count,
    reset_production_model_fit_count,
    target_to_array,
)
from preprocessing.plan import build_preparation_plan, transform_training
from runtime.streaming import (
    collect_test_analysis_sample,
    collect_training_samples,
    stream_predictions_to_csv,
)


def test_prediction_probabilities_are_exactly_chunk_invariant(
    public_data_paths: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    train_path, test_path = public_data_paths
    training = collect_training_samples(train_path)
    test = collect_test_analysis_sample(test_path)
    assert training.model_frame is not None
    target = target_to_array(training.model_frame["ChurnStatus"])
    plan = build_preparation_plan(
        training.model_frame,
        test.analysis_frame,
        target,
        drift_training_frame=training.analysis_frame,
    )
    transformed_train = transform_training(training.model_frame, plan)
    reset_production_model_fit_count()
    model, _ = fit_once(
        transformed_train,
        target,
        categorical_features=plan.categorical_features,
    )

    reference: pd.DataFrame | None = None
    for chunk_rows in (97, 257, 1_000, 5_000, 100_000):
        output = tmp_path / f"prediction_{chunk_rows}.csv"
        result = stream_predictions_to_csv(
            model=model,
            test_path=test_path,
            plan=plan,
            output_path=output,
            chunk_rows=chunk_rows,
        )
        assert result["rows_written"] == test.total_rows
        current = pd.read_csv(output)
        if reference is None:
            reference = current
        else:
            pd.testing.assert_series_equal(
                current["CustomerID"],
                reference["CustomerID"],
            )
            np.testing.assert_array_equal(
                current["probability_score"].to_numpy(dtype=np.float64),
                reference["probability_score"].to_numpy(dtype=np.float64),
            )
    assert get_production_model_fit_count() == 1
