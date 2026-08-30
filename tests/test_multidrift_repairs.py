from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from common.types import FeatureDrift, PreparationPlan
from detection.detector import (
    _categorical_remap_evidence,
    _numeric_representation_evidence,
)
from mitigation.policy import (
    REALIGN_OFFSET,
    REALIGN_SCALE,
    REMAP_CATEGORIES,
    choose_actions,
)
from preprocessing.apply_plan import transform_test


def _review_row(**overrides) -> FeatureDrift:
    values = {
        "feature": "anonymous_signal",
        "kind": "numeric",
        "current_shift": 0.95,
        "historical_max_shift": 0.02,
        "novelty_ratio": 20.0,
        "predictive_strength": 0.30,
        "unique_values": 1_000,
        "recent_unique_values": 1_000,
        "test_unique_values": 1_000,
        "support_retention_ratio": 1.0,
        "missing_shift": 0.0,
        "orientation_train": 0.0,
        "orientation_test": 0.0,
        "orientation_anchor_count": 0,
        "action": "REVIEW_NOVEL",
        "reason": "Synthetic representation evidence.",
    }
    values.update(overrides)
    return FeatureDrift(**values)


@pytest.mark.parametrize("seed", range(5))
def test_large_unit_conversion_is_detected_but_moderate_variance_is_not(
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    recent = pd.Series(rng.normal(20.0, 3.0, 3_000))
    converted = pd.Series(rng.normal(20.0, 3.0, 3_000) * 100.0)
    expanded = pd.Series(rng.normal(20.0, 5.4, 3_000))

    factor, offset, checks = _numeric_representation_evidence(recent, converted)
    assert factor == pytest.approx(100.0, rel=0.08)
    assert offset is None
    assert checks["scale_repair_safe"] is True
    decision = choose_actions(
        [_review_row(scale_factor=factor, evidence_checks=checks)]
    )[0]
    assert decision.action == REALIGN_SCALE

    expanded_factor, expanded_offset, expanded_checks = (
        _numeric_representation_evidence(recent, expanded)
    )
    assert expanded_factor is None
    assert expanded_offset is None
    assert expanded_checks["scale_repair_safe"] is False


@pytest.mark.parametrize("seed", range(5))
def test_large_offset_is_detected_but_ordinary_population_shift_is_not(
    seed: int,
) -> None:
    rng = np.random.default_rng(100 + seed)
    recent = pd.Series(rng.normal(30.0, 4.0, 3_000))
    offset_current = pd.Series(rng.normal(30.0, 4.0, 3_000) + 100.0)
    natural_shift = pd.Series(rng.normal(34.0, 4.8, 3_000))

    factor, offset, checks = _numeric_representation_evidence(
        recent,
        offset_current,
    )
    assert factor is None
    assert offset == pytest.approx(100.0, rel=0.03)
    assert checks["offset_repair_safe"] is True
    decision = choose_actions(
        [_review_row(offset=offset, evidence_checks=checks)]
    )[0]
    assert decision.action == REALIGN_OFFSET

    natural_factor, natural_offset, natural_checks = (
        _numeric_representation_evidence(recent, natural_shift)
    )
    assert natural_factor is None
    assert natural_offset is None
    assert natural_checks["offset_repair_safe"] is False


@pytest.mark.parametrize("seed", range(5))
def test_clear_category_replacement_is_mapped_but_ambiguous_frequencies_are_not(
    seed: int,
) -> None:
    rng = np.random.default_rng(200 + seed)
    recent = pd.Series(rng.choice(["old_1", "old_2", "old_3"], 5_000, p=[0.60, 0.30, 0.10]))
    current = pd.Series(rng.choice(["new_x", "new_y", "new_z"], 5_000, p=[0.60, 0.30, 0.10]))
    mapping, checks = _categorical_remap_evidence(recent, current)

    assert mapping == {"new_x": "old_1", "new_y": "old_2", "new_z": "old_3"}
    assert checks["category_remap_safe"] is True
    decision = choose_actions(
        [
            _review_row(
                kind="categorical",
                unique_values=3,
                recent_unique_values=3,
                test_unique_values=3,
                known_category_support_ratio=0.0,
                category_mapping=mapping,
                evidence_checks=checks,
            )
        ]
    )[0]
    assert decision.action == REMAP_CATEGORIES

    ambiguous_recent = pd.Series(np.tile(["old_1", "old_2", "old_3"], 1_000))
    ambiguous_current = pd.Series(np.tile(["new_x", "new_y", "new_z"], 1_000))
    ambiguous_mapping, ambiguous_checks = _categorical_remap_evidence(
        ambiguous_recent,
        ambiguous_current,
    )
    assert ambiguous_mapping == {}
    assert ambiguous_checks["category_remap_safe"] is False


def _plan(**overrides) -> PreparationPlan:
    values = {
        "feature_order": ["anonymous_signal"],
        "categorical_features": [],
        "numeric_features": ["anonymous_signal"],
        "dropped_features": [],
        "aligned_features": [],
        "reversed_features": [],
        "category_maps": {},
        "numeric_fill_values": {"anonymous_signal": 0.0},
        "train_rank_maps": {},
        "test_rank_maps": {},
        "drift": [],
    }
    values.update(overrides)
    return PreparationPlan(**values)


def test_stored_repair_parameters_are_applied_without_chunk_rediscovery() -> None:
    frame = pd.DataFrame(
        {
            "Month": ["25-Nov", "25-Dec", "25-Dec"],
            "anonymous_signal": [1_000.0, 2_000.0, 3_000.0],
        }
    )
    scale_plan = _plan(scale_factors={"anonymous_signal": 100.0})
    scaled_whole = transform_test(frame, scale_plan)
    scaled_chunked = pd.concat(
        [transform_test(frame.iloc[index : index + 1], scale_plan) for index in range(3)],
        ignore_index=True,
    )
    assert scaled_whole["anonymous_signal"].tolist() == [10.0, 20.0, 30.0]
    pd.testing.assert_frame_equal(scaled_whole, scaled_chunked, check_exact=True)

    offset_plan = _plan(offsets={"anonymous_signal": 100.0})
    offset_frame = frame.assign(anonymous_signal=[110.0, 120.0, 130.0])
    assert transform_test(offset_frame, offset_plan)["anonymous_signal"].tolist() == [
        10.0,
        20.0,
        30.0,
    ]

    category_plan = _plan(
        categorical_features=["anonymous_signal"],
        numeric_features=[],
        category_maps={"anonymous_signal": {"old_1": 0, "old_2": 1}},
        numeric_fill_values={},
        category_remaps={"anonymous_signal": {"new_x": "old_1", "new_y": "old_2"}},
    )
    category_frame = frame.assign(anonymous_signal=["new_x", "new_y", "unknown"])
    assert transform_test(category_frame, category_plan)["anonymous_signal"].tolist() == [
        0,
        1,
        -1,
    ]
