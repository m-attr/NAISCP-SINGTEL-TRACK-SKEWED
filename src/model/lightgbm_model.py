from __future__ import annotations

import gc
import warnings
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

_PRODUCTION_MODEL_FIT_COUNT = 0


def reset_production_model_fit_count() -> None:
    global _PRODUCTION_MODEL_FIT_COUNT
    _PRODUCTION_MODEL_FIT_COUNT = 0


def get_production_model_fit_count() -> int:
    return int(_PRODUCTION_MODEL_FIT_COUNT)


def create_official_lightgbm_model() -> LGBMClassifier:
    """Return the exact fixed model required by the challenge."""
    return LGBMClassifier(
        verbosity=-1,
        objective="binary",
        is_unbalance=True,
        random_state=42,
        importance_type="gain",
    )




def predict_positive_class(model: LGBMClassifier, features: np.ndarray) -> np.ndarray:
    """Predict the positive class without repeating a harmless sklearn name warning."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "X does not have valid feature names, but LGBMClassifier was fitted "
                "with feature names"
            ),
            category=UserWarning,
        )
        return model.predict_proba(features)[:, 1].astype(np.float32)

def target_to_array(target: pd.Series) -> np.ndarray:
    mapped = (
        target.astype("string")
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
    )
    if mapped.isna().any():
        raise ValueError("Training ChurnStatus must contain only Yes/No values.")
    return mapped.astype(np.int8).to_numpy()


def fit_once(
    train_features: pd.DataFrame,
    target: np.ndarray,
    categorical_features: list[str],
) -> tuple[LGBMClassifier, float]:
    """Fit the one permitted model exactly once and report its train AU-PRC."""
    global _PRODUCTION_MODEL_FIT_COUNT
    if _PRODUCTION_MODEL_FIT_COUNT != 0:
        raise RuntimeError("The production model has already been fitted in this pipeline run.")

    train_array = np.ascontiguousarray(
        train_features.to_numpy(dtype=np.float32, copy=False)
    )
    categorical_indices = [
        train_features.columns.get_loc(column)
        for column in categorical_features
        if column in train_features.columns
    ]

    model = create_official_lightgbm_model()
    model.fit(
        train_array,
        target,
        categorical_feature=categorical_indices if categorical_indices else "auto",
    )
    _PRODUCTION_MODEL_FIT_COUNT += 1

    train_predictions = predict_positive_class(model, train_array)
    train_auprc = float(average_precision_score(target, train_predictions))
    del train_predictions, train_array
    gc.collect()
    return model, train_auprc


def fit_once_and_predict(
    train_features: pd.DataFrame,
    target: np.ndarray,
    test_features: pd.DataFrame,
    categorical_features: list[str],
) -> tuple[LGBMClassifier, np.ndarray, np.ndarray, float]:
    """Compatibility helper: one fit followed by train/test predictions."""
    model, train_auprc = fit_once(train_features, target, categorical_features)
    train_array = np.ascontiguousarray(
        train_features.to_numpy(dtype=np.float32, copy=False)
    )
    test_array = np.ascontiguousarray(
        test_features.to_numpy(dtype=np.float32, copy=False)
    )
    train_predictions = predict_positive_class(model, train_array)
    test_predictions = predict_positive_class(model, test_array)
    return model, train_predictions, test_predictions, train_auprc


def predict_in_chunks(
    model: LGBMClassifier,
    transformed_test: pd.DataFrame,
    chunk_size: int,
) -> np.ndarray:
    """Predict already prepared rows without changing any preparation decision."""
    output = np.empty(len(transformed_test), dtype=np.float32)
    size = max(1, int(chunk_size))
    for start in range(0, len(transformed_test), size):
        stop = min(start + size, len(transformed_test))
        array = np.ascontiguousarray(
            transformed_test.iloc[start:stop].to_numpy(dtype=np.float32, copy=False)
        )
        output[start:stop] = predict_positive_class(model, array)
    return output
