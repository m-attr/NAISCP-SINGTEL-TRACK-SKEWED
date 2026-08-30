from __future__ import annotations

from common.types import FeatureDrift
import pytest

from detection.detector import _add_systemic_context
from mitigation.policy import (
    DROP,
    KEEP,
    KEEP_DRIFTED,
    KEEP_SYSTEMIC_DRIFT,
    REPAIR,
    choose_actions,
)


def _evidence(**overrides) -> FeatureDrift:
    values = {
        "feature": "opaque_feature",
        "kind": "numeric",
        "current_shift": 0.30,
        "historical_max_shift": 0.02,
        "novelty_ratio": 4.0,
        "predictive_strength": 0.30,
        "unique_values": 100,
        "recent_unique_values": 95,
        "test_unique_values": 94,
        "support_retention_ratio": 0.98,
        "missing_shift": 0.0,
        "orientation_train": 0.12,
        "orientation_test": -0.09,
        "orientation_anchor_count": 12,
        "action": "REVIEW_NOVEL",
        "reason": "Detection evidence only.",
    }
    values.update(overrides)
    return FeatureDrift(**values)


def test_policy_exposes_keep_drop_and_repair_with_deterministic_reasons() -> None:
    keep = _evidence(action=KEEP, reason="No intervention evidence.")
    repair = _evidence()
    keep_drifted = _evidence(
        kind="categorical",
        orientation_test=0.09,
        known_category_support_ratio=0.95,
    )
    systemic = _evidence(
        orientation_test=0.09,
        systemic_drift_active=True,
        systemic_drift_fraction=0.60,
    )
    drop = _evidence(
        kind="categorical",
        orientation_test=0.09,
        known_category_support_ratio=0.0,
    )

    first = choose_actions([keep, repair, keep_drifted, systemic, drop])
    second = choose_actions([keep, repair, keep_drifted, systemic, drop])
    assert [row.action for row in first] == [
        KEEP,
        REPAIR,
        KEEP_DRIFTED,
        KEEP_SYSTEMIC_DRIFT,
        DROP,
    ]
    assert [row.reason for row in first] == [row.reason for row in second]
    assert all(row.reason.strip() for row in first)


def test_clear_reversal_overrides_systemic_keep() -> None:
    repaired = choose_actions(
        [_evidence(systemic_drift_active=True, systemic_drift_fraction=0.75)]
    )[0]
    assert repaired.action == REPAIR


def test_severe_novel_missingness_is_lossy_last_resort_drop() -> None:
    row = _evidence(
        action="REVIEW_MISSINGNESS",
        missing_shift=0.75,
        historical_missing_shift=0.01,
        missing_novelty_ratio=37.5,
        orientation_test=0.09,
    )
    assert choose_actions([row])[0].action == DROP


@pytest.mark.parametrize("width", [10, 50, 200])
def test_systemic_guard_is_proportion_based_across_widths(width: int) -> None:
    rows = [
        _evidence(
            feature=f"anonymous_{index:03d}",
            unusual_movement=index < width // 2,
            orientation_test=0.09,
        )
        for index in range(width)
    ]
    contextual = _add_systemic_context(rows)
    assert {row.systemic_drift_active for row in contextual} == {True}
    assert contextual[0].systemic_drift_fraction == pytest.approx(
        (width // 2) / width
    )


def test_no_action_is_safe_default() -> None:
    evidence = _evidence(
        action=KEEP,
        orientation_train=-1.0,
        orientation_test=1.0,
    )
    assert choose_actions([evidence])[0] == evidence
