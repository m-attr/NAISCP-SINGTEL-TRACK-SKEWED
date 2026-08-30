from __future__ import annotations

from dataclasses import replace

import numpy as np

from common.types import FeatureDrift
from detection.detector import (
    MIN_ORIENTATION_ANCHORS,
    RANK_UNIQUE_MIN,
    REVERSE_NOVELTY_MIN,
    REVERSE_PREDICTIVE_STRENGTH_MIN,
    REVERSE_SUPPORT_RETENTION_MIN,
    REVERSE_TEST_ASSOC_MIN,
    REVERSE_TRAIN_ASSOC_MIN,
)

KEEP = "KEEP"
DROP = "DROP_NOVEL_UNRELIABLE"
KEEP_DRIFTED = "KEEP_DRIFTED"
KEEP_SYSTEMIC_DRIFT = "KEEP_SYSTEMIC_DRIFT"
REVERSE_PERCENTILE = "REVERSE_PERCENTILE"
REALIGN_SCALE = "REALIGN_SCALE"
REALIGN_OFFSET = "REALIGN_OFFSET"
REMAP_CATEGORIES = "REMAP_CATEGORIES"
REPAIR = REVERSE_PERCENTILE

KEEP_ACTIONS = {KEEP, KEEP_DRIFTED, KEEP_SYSTEMIC_DRIFT}
REPAIR_ACTIONS = {
    REVERSE_PERCENTILE,
    REALIGN_SCALE,
    REALIGN_OFFSET,
    REMAP_CATEGORIES,
}
CATEGORY_SUPPORT_KEEP_MIN = 0.80
DROP_CATEGORY_SUPPORT_MAX = 0.15
DROP_MISSING_SHIFT_MIN = 0.50


def choose_actions(evidence: list[FeatureDrift]) -> list[FeatureDrift]:
    """Choose a conservative action; representation movement alone is not corruption."""
    decisions: list[FeatureDrift] = []
    for row in evidence:
        if row.action not in {"REVIEW_NOVEL", "REVIEW_MISSINGNESS"}:
            decisions.append(row)
            continue

        can_evaluate_reversal = bool(
            row.kind == "numeric"
            and row.unique_values >= RANK_UNIQUE_MIN
            and row.orientation_anchor_count >= MIN_ORIENTATION_ANCHORS
        )
        is_reverse = bool(
            can_evaluate_reversal
            and np.sign(row.orientation_train) != np.sign(row.orientation_test)
            and abs(row.orientation_train) >= REVERSE_TRAIN_ASSOC_MIN
            and abs(row.orientation_test) >= REVERSE_TEST_ASSOC_MIN
            and row.novelty_ratio >= REVERSE_NOVELTY_MIN
            and row.predictive_strength >= REVERSE_PREDICTIVE_STRENGTH_MIN
            and row.support_retention_ratio >= REVERSE_SUPPORT_RETENTION_MIN
        )

        if is_reverse:
            action = REVERSE_PERCENTILE
            reason = (
                "The feature's direction relative to a consensus of independently stable "
                "training-derived profiles reversed in the unlabeled target period, with "
                "enough novelty and training-side signal to justify repair."
            )
        elif (
            row.action == "REVIEW_NOVEL"
            and row.kind == "numeric"
            and row.scale_factor is not None
            and row.evidence_checks.get("scale_repair_safe") is True
        ):
            action = REALIGN_SCALE
            reason = (
                "Robust spread changed by a large internally consistent factor while the "
                "standardized shape remained compatible; apply the stored scale realignment."
            )
        elif (
            row.action == "REVIEW_NOVEL"
            and row.kind == "numeric"
            and row.offset is not None
            and row.evidence_checks.get("offset_repair_safe") is True
        ):
            action = REALIGN_OFFSET
            reason = (
                "A large additive displacement preserved robust spread and standardized "
                "shape; apply the stored offset realignment."
            )
        elif (
            row.action == "REVIEW_NOVEL"
            and row.kind == "categorical"
            and row.category_mapping
            and row.evidence_checks.get("category_remap_safe") is True
        ):
            action = REMAP_CATEGORIES
            reason = (
                "Known labels were replaced, but distinct frequency structure supports one "
                "unambiguous one-to-one mapping; apply only the stored mapping."
            )
        elif row.systemic_drift_active:
            action = KEEP_SYSTEMIC_DRIFT
            reason = (
                "A substantial proportion of otherwise valid features moved together; "
                "preserve this feature unless feature-specific evidence proves a safe repair."
            )
        elif (
            row.kind == "categorical"
            and row.known_category_support_ratio >= CATEGORY_SUPPORT_KEEP_MIN
        ):
            action = KEEP_DRIFTED
            reason = (
                "Category identities remain substantially intact while their proportions "
                "changed, which is more consistent with population movement than relabeling."
            )
        elif (
            row.action == "REVIEW_MISSINGNESS"
            and row.missing_shift >= DROP_MISSING_SHIFT_MIN
        ):
            action = DROP
            reason = (
                "The feature suffered severe novel information loss through missingness, "
                "and no safe reconstruction is available."
            )
        elif (
            row.kind == "categorical"
            and row.known_category_support_ratio <= DROP_CATEGORY_SUPPORT_MAX
        ):
            action = DROP
            reason = (
                "Most known category support disappeared and no unambiguous representation "
                "repair was identified; exclude this isolated unreliable feature."
            )
        else:
            action = KEEP_DRIFTED
            reason = (
                "The feature moved unusually, but no safe transformation or clear evidence "
                "of information loss was established; preserve it rather than guess."
            )

        decisions.append(replace(row, action=action, reason=reason))
    return decisions
