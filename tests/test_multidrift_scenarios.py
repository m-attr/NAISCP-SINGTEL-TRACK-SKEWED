from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from detection.detector import detect_drift
from mitigation.policy import (
    DROP,
    KEEP,
    KEEP_DRIFTED,
    KEEP_SYSTEMIC_DRIFT,
    REALIGN_OFFSET,
    REALIGN_SCALE,
    REMAP_CATEGORIES,
    REPAIR_ACTIONS,
    REVERSE_PERCENTILE,
    choose_actions,
)
from preprocessing.plan import build_preparation_plan, transform_test


MONTHS = [
    "25-Jan",
    "25-Feb",
    "25-Mar",
    "25-Apr",
    "25-May",
    "25-Jun",
    "25-Jul",
    "25-Aug",
]


def _base_frames(
    seed: int,
    width: int,
    *,
    rows_per_month: int = 250,
    test_rows: int = 1_000,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.random.Generator]:
    rng = np.random.default_rng(seed)
    total = len(MONTHS) * rows_per_month
    train_latent = rng.normal(0.0, 1.0, total)
    test_latent = rng.normal(0.0, 1.0, test_rows)
    target = (train_latent + rng.normal(0.0, 0.65, total) > 0.0).astype(np.int8)
    train = pd.DataFrame(
        {
            "CustomerID": np.arange(total),
            "Month": np.repeat(MONTHS, rows_per_month),
            "ChurnStatus": np.where(target == 1, "Yes", "No"),
        }
    )
    test = pd.DataFrame(
        {
            "CustomerID": np.arange(100_000, 100_000 + test_rows),
            "Month": np.repeat("25-Sep", test_rows),
        }
    )
    train_features = {
        f"anonymous_{index:03d}": train_latent
        + rng.normal(0.0, 0.55 + 0.01 * index, total)
        for index in range(width)
    }
    test_features = {
        f"anonymous_{index:03d}": test_latent
        + rng.normal(0.0, 0.55 + 0.01 * index, test_rows)
        for index in range(width)
    }
    train = pd.concat([train, pd.DataFrame(train_features)], axis=1)
    test = pd.concat([test, pd.DataFrame(test_features)], axis=1)
    return train, test, target, test_latent, rng


def _decisions(train: pd.DataFrame, test: pd.DataFrame, target: np.ndarray):
    return choose_actions(detect_drift(train, test, target))


@pytest.mark.parametrize("seed", range(5))
def test_stationary_data_has_no_speculative_repairs_or_drops(seed: int) -> None:
    train, test, target, _, _ = _base_frames(seed, width=12)
    decisions = _decisions(train, test, target)
    assert not [row for row in decisions if row.action in REPAIR_ACTIONS]
    assert not [row for row in decisions if row.action == DROP]
    assert sum(row.action == KEEP for row in decisions) >= 10


@pytest.mark.parametrize("seed", range(5))
def test_gradual_historical_population_movement_is_not_overcorrected(seed: int) -> None:
    train, test, target, test_latent, rng = _base_frames(100 + seed, width=10)
    month_offsets = np.repeat(np.linspace(0.0, 1.4, len(MONTHS)), 250)
    for index in range(10):
        feature = f"anonymous_{index:03d}"
        train[feature] = train[feature] + month_offsets
        test[feature] = test_latent + 1.6 + rng.normal(0.0, 0.65, len(test))
    decisions = _decisions(train, test, target)
    assert not [row for row in decisions if row.action in REPAIR_ACTIONS]
    assert not [row for row in decisions if row.action == DROP]


@pytest.mark.parametrize("width", [10, 50, 200])
def test_broad_population_drift_is_width_aware_and_never_mass_dropped(width: int) -> None:
    train, test, target, _, _ = _base_frames(300 + width, width=width)
    moving = int(np.ceil(width * 0.60))
    for index in range(moving):
        test[f"anonymous_{index:03d}"] += 3.5
    decisions = _decisions(train, test, target)
    assert sum(row.systemic_drift_active for row in decisions) == width
    assert sum(row.action == KEEP_SYSTEMIC_DRIFT for row in decisions) >= int(width * 0.45)
    assert not [row for row in decisions if row.action == DROP]
    assert not [row for row in decisions if row.action in REPAIR_ACTIONS]


@pytest.mark.parametrize("seed", range(5))
def test_same_label_category_population_drift_is_kept(seed: int) -> None:
    train, test, target, train_latent, rng = _base_frames(500 + seed, width=6)
    del train_latent
    feature = "anonymous_category"
    train[feature] = np.where(
        train["ChurnStatus"].eq("Yes"),
        rng.choice(["tier_a", "tier_b"], len(train), p=[0.25, 0.75]),
        rng.choice(["tier_a", "tier_b"], len(train), p=[0.75, 0.25]),
    )
    test[feature] = rng.choice(["tier_a", "tier_b"], len(test), p=[0.90, 0.10])
    decisions = _decisions(train, test, target)
    current = next(row for row in decisions if row.feature == feature)
    assert current.known_category_support_ratio == pytest.approx(1.0)
    assert current.action in {KEEP_DRIFTED, KEEP_SYSTEMIC_DRIFT}
    assert current.action != DROP


@pytest.mark.parametrize("seed", range(5))
def test_missingness_is_compared_with_its_own_history(seed: int) -> None:
    train, test, target, _, rng = _base_frames(700 + seed, width=5)
    feature = "anonymous_missing"
    train[feature] = train["anonymous_000"].copy()
    test[feature] = test["anonymous_000"].copy()
    test.loc[rng.random(len(test)) < 0.70, feature] = np.nan
    novel = next(row for row in _decisions(train, test, target) if row.feature == feature)
    assert novel.missing_shift > 0.60
    assert novel.missing_novelty_ratio > 2.0
    assert novel.action == DROP

    historical_train = train.copy()
    month_rates = {
        "25-Jan": 0.05,
        "25-Feb": 0.05,
        "25-Mar": 0.60,
        "25-Apr": 0.60,
        "25-May": 0.05,
        "25-Jun": 0.05,
        "25-Jul": 0.60,
        "25-Aug": 0.60,
    }
    for month, rate in month_rates.items():
        indices = historical_train.index[historical_train["Month"] == month]
        mask = rng.random(len(indices)) < rate
        historical_train.loc[indices[mask], feature] = np.nan
    historical_test = test.copy()
    historical_test[feature] = historical_test["anonymous_000"]
    historical_test.loc[rng.random(len(historical_test)) < 0.05, feature] = np.nan
    historical = next(
        row
        for row in _decisions(historical_train, historical_test, target)
        if row.feature == feature
    )
    assert historical.historical_missing_shift > 0.45
    assert historical.missing_novelty_ratio < 2.0
    assert historical.action != DROP


@pytest.mark.parametrize("seed", range(5))
def test_direction_reversal_requires_opposite_stable_reference_orientation(
    seed: int,
) -> None:
    train, test, target, _, rng = _base_frames(800 + seed, width=32)
    feature = "anonymous_direction"
    train[feature] = train["anonymous_000"] + rng.normal(0.0, 0.08, len(train))
    test[feature] = -test["anonymous_000"] + 3.0 + rng.normal(0.0, 0.08, len(test))
    reversed_row = next(
        row for row in _decisions(train, test, target) if row.feature == feature
    )
    assert reversed_row.orientation_anchor_count >= 8
    assert reversed_row.orientation_train > 0.08
    assert reversed_row.orientation_test < -0.05
    assert reversed_row.action == REVERSE_PERCENTILE

    same_direction = test.copy()
    same_direction[feature] = (
        same_direction["anonymous_000"]
        + 3.0
        + rng.normal(0.0, 0.08, len(same_direction))
    )
    ordinary_row = next(
        row
        for row in _decisions(train, same_direction, target)
        if row.feature == feature
    )
    assert ordinary_row.orientation_test > 0.05
    assert ordinary_row.action != REVERSE_PERCENTILE


@pytest.mark.parametrize("seed", range(5))
def test_multiple_drift_types_receive_different_actions_in_one_run(seed: int) -> None:
    train, test, target, _, rng = _base_frames(900 + seed, width=12)
    test["anonymous_000"] *= 100.0
    test["anonymous_001"] += 100.0
    for index in range(2, 8):
        test[f"anonymous_{index:03d}"] += 3.5

    category = "anonymous_category"
    train[category] = rng.choice(
        ["old_a", "old_b", "old_c"],
        len(train),
        p=[0.60, 0.30, 0.10],
    )
    # Make the anonymous category training-useful without leaking any test label.
    train.loc[train["ChurnStatus"].eq("Yes"), category] = rng.choice(
        ["old_a", "old_b", "old_c"],
        train["ChurnStatus"].eq("Yes").sum(),
        p=[0.75, 0.20, 0.05],
    )
    test[category] = rng.choice(
        ["new_x", "new_y", "new_z"],
        len(test),
        p=train[category].value_counts(normalize=True)[["old_a", "old_b", "old_c"]].to_numpy(),
    )

    decisions = {row.feature: row for row in _decisions(train, test, target)}
    assert decisions["anonymous_000"].action == REALIGN_SCALE
    assert decisions["anonymous_001"].action == REALIGN_OFFSET
    assert decisions[category].action == REMAP_CATEGORIES
    assert all(
        decisions[f"anonymous_{index:03d}"].action == KEEP_SYSTEMIC_DRIFT
        for index in range(2, 8)
    )
    assert decisions["anonymous_011"].action == KEEP


def test_complete_multidrift_plan_freezes_every_repair_parameter() -> None:
    train, test, target, _, rng = _base_frames(1_200, width=12)
    test["anonymous_000"] *= 100.0
    test["anonymous_001"] += 100.0
    for index in range(2, 8):
        test[f"anonymous_{index:03d}"] += 3.5
    category = "anonymous_category"
    train[category] = rng.choice(
        ["old_a", "old_b", "old_c"], len(train), p=[0.60, 0.30, 0.10]
    )
    train.loc[train["ChurnStatus"].eq("Yes"), category] = rng.choice(
        ["old_a", "old_b", "old_c"],
        train["ChurnStatus"].eq("Yes").sum(),
        p=[0.75, 0.20, 0.05],
    )
    probabilities = train[category].value_counts(normalize=True)[
        ["old_a", "old_b", "old_c"]
    ].to_numpy()
    test[category] = rng.choice(
        ["new_x", "new_y", "new_z"], len(test), p=probabilities
    )

    plan = build_preparation_plan(train, test, target)
    assert set(plan.scale_factors) == {"anonymous_000"}
    assert set(plan.offsets) == {"anonymous_001"}
    assert set(plan.category_remaps) == {category}
    assert plan.category_remaps[category] == {
        "new_x": "old_a",
        "new_y": "old_b",
        "new_z": "old_c",
    }

    whole = transform_test(test, plan).reset_index(drop=True)
    boundaries = np.linspace(0, len(test), 18, dtype=int)
    chunked = pd.concat(
        [
            transform_test(test.iloc[boundaries[index] : boundaries[index + 1]].copy(), plan)
            for index in range(17)
        ],
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(whole, chunked, check_exact=True)
    assert abs(float(whole["anonymous_000"].median())) < 0.2
    assert abs(float(whole["anonymous_001"].median())) < 0.2
    assert set(whole[category].unique()).issubset(
        set(plan.category_maps[category].values())
    )
