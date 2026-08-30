from __future__ import annotations

import numpy as np
import pandas as pd

from common.contracts import (
    ID_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
    canonicalize_category,
    parse_competition_month_value,
)
from common.types import PreparationPlan
from detection.detector import (
    CURRENT_SHIFT_MIN,
    KS_MAX_VALUES,
    MIN_ORIENTATION_ANCHORS,
    NOVELTY_MIN,
    NOVEL_MISSING_SHIFT_MIN,
    OFFSET_SPREAD_MULTIPLIER_MIN,
    REPRESENTATION_MIN_OBSERVATIONS,
    SCALE_FACTOR_MIN,
    SHAPE_KS_MAX,
    PREDICTIVE_STRENGTH_MIN,
    RANK_MAP_MAX_POINTS,
    RANK_UNIQUE_MIN,
    REVERSE_NOVELTY_MIN,
    REVERSE_PREDICTIVE_STRENGTH_MIN,
    REVERSE_SUPPORT_RETENTION_MIN,
    REVERSE_TEST_ASSOC_MIN,
    REVERSE_TRAIN_ASSOC_MIN,
    STABLE_ANCHOR_SHIFT_MAX,
    STABLE_ANCHOR_VALIDATION_MIN,
    SYSTEMIC_DRIFT_FRACTION_MIN,
    detect_drift,
)
from mitigation.policy import (
    DROP,
    REALIGN_OFFSET,
    REALIGN_SCALE,
    REMAP_CATEGORIES,
    REVERSE_PERCENTILE,
    choose_actions,
)
from preprocessing.apply_plan import transform_test, transform_training
from preprocessing.rank_maps import build_rank_maps


def build_preparation_plan(
    training_frame: pd.DataFrame,
    test_analysis_frame: pd.DataFrame,
    target: np.ndarray,
    *,
    drift_training_frame: pd.DataFrame | None = None,
) -> PreparationPlan:
    """Detect drift, decide mitigation once, and freeze a reusable run plan."""
    required_train = {ID_COLUMN, TIME_COLUMN, TARGET_COLUMN}
    required_test = {ID_COLUMN, TIME_COLUMN}
    missing_train = required_train.difference(training_frame.columns)
    missing_test = required_test.difference(test_analysis_frame.columns)
    if missing_train:
        raise ValueError(
            f"Training data is missing required columns: {sorted(missing_train)}"
        )
    if missing_test:
        raise ValueError(
            f"Test data is missing required columns: {sorted(missing_test)}"
        )

    drift_train = (
        drift_training_frame if drift_training_frame is not None else training_frame
    )
    if TARGET_COLUMN not in drift_train.columns:
        raise ValueError("Drift training sample must contain ChurnStatus.")
    drift_target = (
        drift_train[TARGET_COLUMN]
        .astype("string")
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
    )
    if drift_target.isna().any():
        raise ValueError("Training ChurnStatus must contain only Yes/No values.")

    for value in pd.concat(
        [drift_train[TIME_COLUMN], test_analysis_frame[TIME_COLUMN]],
        ignore_index=True,
    ).dropna().unique():
        parse_competition_month_value(value)

    detection_evidence = detect_drift(
        drift_train,
        test_analysis_frame,
        drift_target.astype(np.int8).to_numpy(),
    )
    drift = choose_actions(detection_evidence)

    dropped = [row.feature for row in drift if row.action == DROP]
    aligned = [
        row.feature for row in drift if row.action == REVERSE_PERCENTILE
    ]
    reversed_features = [
        row.feature for row in drift if row.action == REVERSE_PERCENTILE
    ]
    scale_factors = {
        row.feature: float(row.scale_factor)
        for row in drift
        if row.action == REALIGN_SCALE and row.scale_factor is not None
    }
    offsets = {
        row.feature: float(row.offset)
        for row in drift
        if row.action == REALIGN_OFFSET and row.offset is not None
    }
    category_remaps = {
        row.feature: dict(row.category_mapping)
        for row in drift
        if row.action == REMAP_CATEGORIES and row.category_mapping
    }

    shared = [
        column
        for column in training_frame.columns
        if column in test_analysis_frame.columns
        and column not in {ID_COLUMN, TIME_COLUMN, TARGET_COLUMN}
        and column not in dropped
    ]
    categorical = [
        column
        for column in shared
        if not pd.api.types.is_numeric_dtype(training_frame[column])
    ]
    numeric = [column for column in shared if column not in categorical]
    aligned = [feature for feature in aligned if feature in numeric]
    reversed_features = [
        feature for feature in reversed_features if feature in aligned
    ]

    category_maps: dict[str, dict[str, int]] = {}
    for feature in categorical:
        categories = sorted(
            canonicalize_category(training_frame[feature]).unique().tolist()
        )
        category_maps[feature] = {
            value: index for index, value in enumerate(categories)
        }

    train_rank_maps = build_rank_maps(training_frame, aligned)
    test_rank_maps = build_rank_maps(test_analysis_frame, aligned)

    fill_values: dict[str, float] = {}
    for feature in numeric:
        if feature in aligned:
            fill_values[feature] = 0.5
        else:
            raw = pd.to_numeric(training_frame[feature], errors="coerce")
            fill_values[feature] = (
                float(raw.median()) if raw.notna().any() else 0.0
            )

    return PreparationPlan(
        feature_order=shared,
        categorical_features=categorical,
        numeric_features=numeric,
        dropped_features=dropped,
        aligned_features=aligned,
        reversed_features=reversed_features,
        category_maps=category_maps,
        numeric_fill_values=fill_values,
        train_rank_maps=train_rank_maps,
        test_rank_maps=test_rank_maps,
        drift=drift,
        scale_factors=scale_factors,
        offsets=offsets,
        category_remaps=category_remaps,
        thresholds={
            "current_shift_min": CURRENT_SHIFT_MIN,
            "novelty_min": NOVELTY_MIN,
            "predictive_strength_min": PREDICTIVE_STRENGTH_MIN,
            "systemic_drift_fraction_min": SYSTEMIC_DRIFT_FRACTION_MIN,
            "novel_missing_shift_min": NOVEL_MISSING_SHIFT_MIN,
            "representation_min_observations": float(REPRESENTATION_MIN_OBSERVATIONS),
            "shape_ks_max": SHAPE_KS_MAX,
            "scale_factor_min": SCALE_FACTOR_MIN,
            "offset_spread_multiplier_min": OFFSET_SPREAD_MULTIPLIER_MIN,
            "stable_anchor_shift_max": STABLE_ANCHOR_SHIFT_MAX,
            "stable_anchor_validation_min": STABLE_ANCHOR_VALIDATION_MIN,
            "minimum_orientation_anchors": float(MIN_ORIENTATION_ANCHORS),
            "reverse_train_association_min": REVERSE_TRAIN_ASSOC_MIN,
            "reverse_test_association_min": REVERSE_TEST_ASSOC_MIN,
            "reverse_novelty_min": REVERSE_NOVELTY_MIN,
            "reverse_predictive_strength_min": REVERSE_PREDICTIVE_STRENGTH_MIN,
            "reverse_support_retention_min": REVERSE_SUPPORT_RETENTION_MIN,
            "rank_unique_values_min": float(RANK_UNIQUE_MIN),
            "rank_map_max_points": float(RANK_MAP_MAX_POINTS),
            "ks_max_values": float(KS_MAX_VALUES),
        },
    )


__all__ = [
    "ID_COLUMN",
    "TARGET_COLUMN",
    "TIME_COLUMN",
    "PreparationPlan",
    "build_preparation_plan",
    "transform_test",
    "transform_training",
]
